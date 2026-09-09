package helps

import (
	"strings"
	"testing"
)

const prefixDiagBaseBody = `{
  "model": "grok-4.6",
  "prompt_cache_key": "session-abc",
  "instructions": "you are a coding agent",
  "tools": [{"type":"function","name":"read_file"}],
  "input": [
    {"type":"message","role":"user","content":"hello"},
    {"type":"reasoning","encrypted_content":"enc-1","summary":[]},
    {"type":"function_call","call_id":"c1","name":"read_file","arguments":"{}"},
    {"type":"function_call_output","call_id":"c1","output":"ok"}
  ]
}`

func TestBuildResponsesPrefixChainSegments(t *testing.T) {
	chain := BuildResponsesPrefixChain([]byte(prefixDiagBaseBody))
	// instructions + tools + 4 input items
	if chain.Len() != 6 {
		t.Fatalf("chain length = %d, want 6", chain.Len())
	}
	wantLabels := []string{
		PrefixSegmentInstructions,
		PrefixSegmentTools,
		"input[0]:message",
		"input[1]:reasoning",
		"input[2]:function_call",
		"input[3]:function_call_output",
	}
	for i, want := range wantLabels {
		if chain.Labels[i] != want {
			t.Fatalf("Labels[%d] = %q, want %q", i, chain.Labels[i], want)
		}
	}
	if chain.PromptCacheKey != "session-abc" {
		t.Fatalf("PromptCacheKey = %q, want session-abc", chain.PromptCacheKey)
	}
	for i, key := range chain.Keys {
		if len(key) != 64 {
			t.Fatalf("Keys[%d] = %q, want a 64-char hex digest", i, key)
		}
	}
}

func TestBuildResponsesPrefixChainWithoutInputArray(t *testing.T) {
	if got := BuildResponsesPrefixChain([]byte(`{"model":"grok-4.6","input":"hello"}`)); got.Len() != 0 {
		t.Fatalf("chain length = %d, want 0 for a non-array input", got.Len())
	}
	if got := BuildResponsesPrefixChain([]byte(`not json`)); got.Len() != 0 {
		t.Fatalf("chain length = %d, want 0 for invalid JSON", got.Len())
	}
}

func TestComparePrefixChainsIdenticalBodies(t *testing.T) {
	chain := BuildResponsesPrefixChain([]byte(prefixDiagBaseBody))
	reused := ComparePrefixChains(chain, chain)
	if reused != chain.Len() {
		t.Fatalf("reused = %d, want %d", reused, chain.Len())
	}
	if reason := ClassifyPrefixDrift(chain, chain, reused); reason != PrefixDriftNone {
		t.Fatalf("reason = %q, want %q", reason, PrefixDriftNone)
	}
}

func TestComparePrefixChainsAppendOnlyTurnIsFullReuse(t *testing.T) {
	previous := BuildResponsesPrefixChain([]byte(prefixDiagBaseBody))
	extended := strings.Replace(prefixDiagBaseBody,
		`{"type":"function_call_output","call_id":"c1","output":"ok"}`,
		`{"type":"function_call_output","call_id":"c1","output":"ok"},
    {"type":"message","role":"user","content":"next"}`, 1)
	next := BuildResponsesPrefixChain([]byte(extended))

	reused := ComparePrefixChains(previous, next)
	if reused != previous.Len() {
		t.Fatalf("reused = %d, want the whole previous chain (%d)", reused, previous.Len())
	}
	if reason := ClassifyPrefixDrift(previous, next, reused); reason != PrefixDriftNone {
		t.Fatalf("reason = %q, want %q for an append-only turn", reason, PrefixDriftNone)
	}
}

// A dropped replayed reasoning item is the H1 signature: the previous turn had a
// reasoning item at that slot and this one carries a different kind there.
func TestClassifyPrefixDriftReplayItemMissing(t *testing.T) {
	previous := BuildResponsesPrefixChain([]byte(prefixDiagBaseBody))
	withoutReasoning := strings.Replace(prefixDiagBaseBody,
		`{"type":"reasoning","encrypted_content":"enc-1","summary":[]},
    `, "", 1)
	next := BuildResponsesPrefixChain([]byte(withoutReasoning))

	reused := ComparePrefixChains(previous, next)
	if reused != 3 {
		t.Fatalf("reused = %d, want 3 (instructions, tools, input[0])", reused)
	}
	if got := ClassifyPrefixDrift(previous, next, reused); got != PrefixDriftReplayItemMissing {
		t.Fatalf("reason = %q, want %q", got, PrefixDriftReplayItemMissing)
	}
	if got := PrefixSegmentLabel(next, reused); got != "input[1]:function_call" {
		t.Fatalf("divergence label = %q, want input[1]:function_call", got)
	}
}

// Same slot, still a reasoning item, different bytes: encrypted_content surgery.
func TestClassifyPrefixDriftEncryptedContentStripped(t *testing.T) {
	previous := BuildResponsesPrefixChain([]byte(prefixDiagBaseBody))
	stripped := strings.Replace(prefixDiagBaseBody,
		`{"type":"reasoning","encrypted_content":"enc-1","summary":[]}`,
		`{"type":"reasoning","summary":[]}`, 1)
	next := BuildResponsesPrefixChain([]byte(stripped))

	reused := ComparePrefixChains(previous, next)
	if reused != 3 {
		t.Fatalf("reused = %d, want 3", reused)
	}
	if got := ClassifyPrefixDrift(previous, next, reused); got != PrefixDriftEncryptedStripped {
		t.Fatalf("reason = %q, want %q", got, PrefixDriftEncryptedStripped)
	}
}

func TestClassifyPrefixDriftToolsChanged(t *testing.T) {
	previous := BuildResponsesPrefixChain([]byte(prefixDiagBaseBody))
	retooled := strings.Replace(prefixDiagBaseBody,
		`"tools": [{"type":"function","name":"read_file"}]`,
		`"tools": [{"type":"function","name":"read_file"},{"type":"function","name":"write_file"}]`, 1)
	next := BuildResponsesPrefixChain([]byte(retooled))

	reused := ComparePrefixChains(previous, next)
	if reused != 1 {
		t.Fatalf("reused = %d, want 1 (instructions only)", reused)
	}
	if got := ClassifyPrefixDrift(previous, next, reused); got != PrefixDriftToolsChanged {
		t.Fatalf("reason = %q, want %q", got, PrefixDriftToolsChanged)
	}
}

func TestClassifyPrefixDriftInstructionsChanged(t *testing.T) {
	previous := BuildResponsesPrefixChain([]byte(prefixDiagBaseBody))
	next := BuildResponsesPrefixChain([]byte(strings.Replace(prefixDiagBaseBody,
		"you are a coding agent", "you are a coding agent. today is friday", 1)))

	reused := ComparePrefixChains(previous, next)
	if reused != 0 {
		t.Fatalf("reused = %d, want 0", reused)
	}
	if got := ClassifyPrefixDrift(previous, next, reused); got != PrefixDriftInstructionsChanged {
		t.Fatalf("reason = %q, want %q", got, PrefixDriftInstructionsChanged)
	}
}

// A rotated prompt_cache_key changes the upstream cache namespace even when the
// hashed prefix is byte-identical, so it must outrank every positional reason.
func TestClassifyPrefixDriftSessionKeyChangedOutranksPosition(t *testing.T) {
	previous := BuildResponsesPrefixChain([]byte(prefixDiagBaseBody))
	next := BuildResponsesPrefixChain([]byte(strings.Replace(prefixDiagBaseBody,
		`"prompt_cache_key": "session-abc"`, `"prompt_cache_key": "session-xyz"`, 1)))

	reused := ComparePrefixChains(previous, next)
	if reused != previous.Len() {
		t.Fatalf("reused = %d, want %d: prompt_cache_key is not part of the hashed prefix", reused, previous.Len())
	}
	if got := ClassifyPrefixDrift(previous, next, reused); got != PrefixDriftSessionKeyChanged {
		t.Fatalf("reason = %q, want %q", got, PrefixDriftSessionKeyChanged)
	}
}

func TestPrefixSegmentAccessorsOutOfRange(t *testing.T) {
	chain := BuildResponsesPrefixChain([]byte(prefixDiagBaseBody))
	if got := PrefixSegmentLabel(chain, -1); got != "" {
		t.Fatalf("label at -1 = %q, want empty", got)
	}
	if got := PrefixSegmentKind(chain, chain.Len()); got != "" {
		t.Fatalf("kind at len = %q, want empty", got)
	}
}
