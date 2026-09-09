package executor

import (
	"context"
	"strings"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	"github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/usage"
	log "github.com/sirupsen/logrus"
	"github.com/sirupsen/logrus/hooks/test"
)

const xaiDiagTurnOne = `{"model":"grok-4.6","prompt_cache_key":"session-abc","instructions":"sys","tools":[],` +
	`"input":[{"type":"message","role":"user","content":"hi"},` +
	`{"type":"reasoning","encrypted_content":"enc-1","summary":[]},` +
	`{"type":"function_call_output","call_id":"c1","output":"ok"}]}`

// Turn two drops the replayed reasoning item: the H1 signature.
const xaiDiagTurnTwo = `{"model":"grok-4.6","prompt_cache_key":"session-abc","instructions":"sys","tools":[],` +
	`"input":[{"type":"message","role":"user","content":"hi"},` +
	`{"type":"function_call_output","call_id":"c1","output":"ok"}]}`

func newXAIDiagExecutor(enabled bool) *XAIExecutor {
	cfg := &config.Config{}
	cfg.XAI.PrefixDiagnostics = enabled
	return NewXAIExecutor(cfg)
}

func newXAIDiagPrepared(sessionKey string) *xaiPreparedRequest {
	return &xaiPreparedRequest{
		baseModel: "grok-4.6",
		replayScope: xaiReasoningReplayScope{
			modelName:  "grok-4.6",
			sessionKey: sessionKey,
		},
	}
}

func captureXAIDiagLogs(t *testing.T) *test.Hook {
	t.Helper()
	hook := test.NewGlobal()
	previousLevel := log.GetLevel()
	log.SetLevel(log.DebugLevel)
	t.Cleanup(func() {
		log.SetLevel(previousLevel)
		hook.Reset()
	})
	return hook
}

func xaiDiagFindMessage(hook *test.Hook, substring string) (string, bool) {
	for _, entry := range hook.AllEntries() {
		if strings.Contains(entry.Message, substring) {
			return entry.Message, true
		}
	}
	return "", false
}

func TestDiagnoseXAIPrefixDisabledByDefault(t *testing.T) {
	hook := captureXAIDiagLogs(t)
	exec := newXAIDiagExecutor(false)
	prepared := newXAIDiagPrepared("caller:aa:pck:disabled")

	exec.diagnoseXAIPrefix(context.Background(), prepared, []byte(xaiDiagTurnOne))
	exec.diagnoseXAIPrefix(context.Background(), prepared, []byte(xaiDiagTurnTwo))

	if msg, ok := xaiDiagFindMessage(hook, "xai: prefix"); ok {
		t.Fatalf("diagnostics must stay silent when the flag is off, got %q", msg)
	}
}

func TestDiagnoseXAIPrefixReportsDriftBetweenTurns(t *testing.T) {
	hook := captureXAIDiagLogs(t)
	exec := newXAIDiagExecutor(true)
	prepared := newXAIDiagPrepared("caller:aa:pck:drift")

	exec.diagnoseXAIPrefix(context.Background(), prepared, []byte(xaiDiagTurnOne))
	if _, ok := xaiDiagFindMessage(hook, "xai: prefix baseline"); !ok {
		t.Fatal("the first turn of a session should log a baseline")
	}
	hook.Reset()

	exec.diagnoseXAIPrefix(context.Background(), prepared, []byte(xaiDiagTurnTwo))
	msg, ok := xaiDiagFindMessage(hook, "xai: prefix drift")
	if !ok {
		t.Fatalf("expected a drift line, got %v", hook.AllEntries())
	}
	// instructions + tools + input[0] survive; the reasoning item at index 3 is gone.
	for _, want := range []string{
		"reused=3/4",
		"first_divergence=input[1]:function_call_output",
		"prev=reasoning",
		"reason=replay_item_missing",
	} {
		if !strings.Contains(msg, want) {
			t.Fatalf("drift line %q is missing %q", msg, want)
		}
	}
}

func TestDiagnoseXAIPrefixAppendOnlyTurnIsNotDrift(t *testing.T) {
	hook := captureXAIDiagLogs(t)
	exec := newXAIDiagExecutor(true)
	prepared := newXAIDiagPrepared("caller:aa:pck:append")

	exec.diagnoseXAIPrefix(context.Background(), prepared, []byte(xaiDiagTurnOne))
	hook.Reset()

	appended := strings.Replace(xaiDiagTurnOne,
		`{"type":"function_call_output","call_id":"c1","output":"ok"}]}`,
		`{"type":"function_call_output","call_id":"c1","output":"ok"},{"type":"message","role":"user","content":"more"}]}`, 1)
	exec.diagnoseXAIPrefix(context.Background(), prepared, []byte(appended))

	if msg, ok := xaiDiagFindMessage(hook, "xai: prefix drift"); ok {
		t.Fatalf("an append-only turn must not report drift, got %q", msg)
	}
	msg, ok := xaiDiagFindMessage(hook, "xai: prefix extended")
	if !ok {
		t.Fatalf("expected an extended line, got %v", hook.AllEntries())
	}
	if !strings.Contains(msg, "reused=5/6") {
		t.Fatalf("extended line %q should report full reuse of the previous chain", msg)
	}
}

func TestDiagnoseXAIPrefixSkipsInvalidScopeAndBody(t *testing.T) {
	hook := captureXAIDiagLogs(t)
	exec := newXAIDiagExecutor(true)

	exec.diagnoseXAIPrefix(context.Background(), newXAIDiagPrepared(""), []byte(xaiDiagTurnOne))
	exec.diagnoseXAIPrefix(context.Background(), nil, []byte(xaiDiagTurnOne))
	exec.diagnoseXAIPrefix(context.Background(), newXAIDiagPrepared("caller:aa:pck:skip"), []byte(`{"input":"not-an-array"}`))

	if msg, ok := xaiDiagFindMessage(hook, "xai: prefix"); ok {
		t.Fatalf("expected no diagnostics for an unusable scope or body, got %q", msg)
	}
}

func TestTruncateXAIDiagnosticValue(t *testing.T) {
	if got := truncateXAIDiagnosticValue("   "); got != "-" {
		t.Fatalf("blank value = %q, want %q", got, "-")
	}
	if got := truncateXAIDiagnosticValue("short-session"); got != "short-session" {
		t.Fatalf("short value = %q, want it unchanged", got)
	}
	long := "caller:0123456789abcdef:pck:0123456789abcdef"
	got := truncateXAIDiagnosticValue(long)
	if got != long[:16]+"..." {
		t.Fatalf("long value = %q, want %q", got, long[:16]+"...")
	}
}

func TestLogXAIPromptCacheUsage(t *testing.T) {
	hook := captureXAIDiagLogs(t)
	prepared := newXAIDiagPrepared("caller:aa:pck:usage")
	auth := &cliproxyauth.Auth{ID: "auth-id", Label: "xai-account.json"}

	logXAIPromptCacheUsage(context.Background(), prepared, auth, usage.Detail{
		InputTokens:  1000,
		CachedTokens: 869,
	})

	msg, ok := xaiDiagFindMessage(hook, "xai: prompt-cache")
	if !ok {
		t.Fatalf("expected a prompt-cache line, got %v", hook.AllEntries())
	}
	for _, want := range []string{
		"session=caller:aa:pck:u",
		"auth=xai-account.json",
		"model=grok-4.6",
		"input=1000",
		"cached=869",
		"hit=86.9%",
	} {
		if !strings.Contains(msg, want) {
			t.Fatalf("prompt-cache line %q is missing %q", msg, want)
		}
	}
}

func TestLogXAIPromptCacheUsageSkipsUnusableInput(t *testing.T) {
	hook := captureXAIDiagLogs(t)
	prepared := newXAIDiagPrepared("caller:aa:pck:usage")

	// No usable token count: nothing to report a ratio against.
	logXAIPromptCacheUsage(context.Background(), prepared, nil, usage.Detail{})
	logXAIPromptCacheUsage(context.Background(), nil, nil, usage.Detail{InputTokens: 10})

	if msg, ok := xaiDiagFindMessage(hook, "xai: prompt-cache"); ok {
		t.Fatalf("expected no prompt-cache line, got %q", msg)
	}
}

func TestLogXAIPromptCacheUsageFallsBackToAuthID(t *testing.T) {
	hook := captureXAIDiagLogs(t)
	prepared := newXAIDiagPrepared("caller:aa:pck:usage")

	logXAIPromptCacheUsage(context.Background(), prepared, &cliproxyauth.Auth{ID: "auth-id"}, usage.Detail{
		InputTokens:  10,
		CachedTokens: 0,
	})

	msg, ok := xaiDiagFindMessage(hook, "xai: prompt-cache")
	if !ok {
		t.Fatalf("expected a prompt-cache line, got %v", hook.AllEntries())
	}
	if !strings.Contains(msg, "auth=auth-id") || !strings.Contains(msg, "hit=0.0%") {
		t.Fatalf("prompt-cache line %q should fall back to the auth id and report a zero hit rate", msg)
	}
}
