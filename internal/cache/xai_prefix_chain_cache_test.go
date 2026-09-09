package cache

import (
	"testing"
	"time"
)

func resetXAIPrefixChainCache(t *testing.T) {
	t.Helper()
	xaiPrefixChainMu.Lock()
	xaiPrefixChainEntries = make(map[string]xaiPrefixChainEntry)
	xaiPrefixChainMu.Unlock()
	original := xaiPrefixChainNow
	t.Cleanup(func() {
		xaiPrefixChainNow = original
		xaiPrefixChainMu.Lock()
		xaiPrefixChainEntries = make(map[string]xaiPrefixChainEntry)
		xaiPrefixChainMu.Unlock()
	})
}

func sampleXAIPrefixChain(keys ...string) XAIPrefixChain {
	chain := XAIPrefixChain{PromptCacheKey: "session-abc"}
	for _, key := range keys {
		chain.Keys = append(chain.Keys, key)
		chain.Labels = append(chain.Labels, "input[x]:message")
		chain.Kinds = append(chain.Kinds, "message")
	}
	return chain
}

func TestXAIPrefixChainRoundTrip(t *testing.T) {
	resetXAIPrefixChainCache(t)
	want := sampleXAIPrefixChain("a", "b", "c")
	StoreXAIPrefixChain("grok-4.6", "sess-1", want)

	got, ok := LoadXAIPrefixChain("grok-4.6", "sess-1")
	if !ok {
		t.Fatal("LoadXAIPrefixChain reported a miss for a freshly stored chain")
	}
	if got.Len() != want.Len() {
		t.Fatalf("chain length = %d, want %d", got.Len(), want.Len())
	}
	if got.PromptCacheKey != want.PromptCacheKey {
		t.Fatalf("PromptCacheKey = %q, want %q", got.PromptCacheKey, want.PromptCacheKey)
	}
	for i := range want.Keys {
		if got.Keys[i] != want.Keys[i] {
			t.Fatalf("Keys[%d] = %q, want %q", i, got.Keys[i], want.Keys[i])
		}
	}
}

func TestXAIPrefixChainIsolatedByModelAndSession(t *testing.T) {
	resetXAIPrefixChainCache(t)
	StoreXAIPrefixChain("grok-4.6", "sess-1", sampleXAIPrefixChain("a"))

	if _, ok := LoadXAIPrefixChain("grok-4.6", "sess-2"); ok {
		t.Fatal("a different session must not read another session's chain")
	}
	if _, ok := LoadXAIPrefixChain("grok-4.6-fast", "sess-1"); ok {
		t.Fatal("a different model must not read another model's chain")
	}
}

func TestXAIPrefixChainRejectsEmptyKeyParts(t *testing.T) {
	resetXAIPrefixChainCache(t)
	StoreXAIPrefixChain("", "sess-1", sampleXAIPrefixChain("a"))
	StoreXAIPrefixChain("grok-4.6", "  ", sampleXAIPrefixChain("a"))
	// An empty chain carries no baseline and must not evict a real one.
	StoreXAIPrefixChain("grok-4.6", "sess-1", XAIPrefixChain{})

	if _, ok := LoadXAIPrefixChain("grok-4.6", "sess-1"); ok {
		t.Fatal("expected no entry to be stored")
	}
	xaiPrefixChainMu.Lock()
	size := len(xaiPrefixChainEntries)
	xaiPrefixChainMu.Unlock()
	if size != 0 {
		t.Fatalf("cache size = %d, want 0", size)
	}
}

// TTL is exercised through the injected clock: wall-clock sleeps are unreliable
// at this granularity on platforms with a coarse timer resolution.
func TestXAIPrefixChainExpiresAfterTTL(t *testing.T) {
	resetXAIPrefixChainCache(t)
	base := time.Date(2026, 9, 9, 11, 0, 0, 0, time.UTC)
	current := base
	xaiPrefixChainNow = func() time.Time { return current }

	StoreXAIPrefixChain("grok-4.6", "sess-1", sampleXAIPrefixChain("a"))

	current = base.Add(XAIPrefixChainCacheTTL)
	if _, ok := LoadXAIPrefixChain("grok-4.6", "sess-1"); !ok {
		t.Fatal("entry must survive exactly at the TTL boundary")
	}

	current = base.Add(XAIPrefixChainCacheTTL + time.Second)
	if _, ok := LoadXAIPrefixChain("grok-4.6", "sess-1"); ok {
		t.Fatal("entry must be gone past the TTL")
	}
	xaiPrefixChainMu.Lock()
	size := len(xaiPrefixChainEntries)
	xaiPrefixChainMu.Unlock()
	if size != 0 {
		t.Fatalf("expired entry was not deleted on read: cache size = %d", size)
	}
}

func TestPurgeExpiredXAIPrefixChainCache(t *testing.T) {
	resetXAIPrefixChainCache(t)
	base := time.Date(2026, 9, 9, 11, 0, 0, 0, time.UTC)
	current := base
	xaiPrefixChainNow = func() time.Time { return current }

	StoreXAIPrefixChain("grok-4.6", "old", sampleXAIPrefixChain("a"))
	current = base.Add(XAIPrefixChainCacheTTL)
	StoreXAIPrefixChain("grok-4.6", "fresh", sampleXAIPrefixChain("b"))

	purgeExpiredXAIPrefixChainCache(base.Add(XAIPrefixChainCacheTTL + time.Second))

	if _, ok := LoadXAIPrefixChain("grok-4.6", "old"); ok {
		t.Fatal("expired entry survived the purge")
	}
	if _, ok := LoadXAIPrefixChain("grok-4.6", "fresh"); !ok {
		t.Fatal("live entry was purged")
	}
}

func TestXAIPrefixChainEvictsOldestAtCapacity(t *testing.T) {
	resetXAIPrefixChainCache(t)
	base := time.Date(2026, 9, 9, 11, 0, 0, 0, time.UTC)
	current := base
	xaiPrefixChainNow = func() time.Time { return current }

	// Fill to capacity, each entry one second newer than the last.
	for i := 0; i <= XAIPrefixChainCacheMaxEntries; i++ {
		current = base.Add(time.Duration(i) * time.Second)
		StoreXAIPrefixChain("grok-4.6", sessionName(i), sampleXAIPrefixChain("a"))
	}

	xaiPrefixChainMu.Lock()
	size := len(xaiPrefixChainEntries)
	xaiPrefixChainMu.Unlock()
	wantSize := XAIPrefixChainCacheMaxEntries + 1 - XAIPrefixChainCacheEvictBatchSize
	if size != wantSize {
		t.Fatalf("cache size after eviction = %d, want %d", size, wantSize)
	}
	if _, ok := LoadXAIPrefixChain("grok-4.6", sessionName(0)); ok {
		t.Fatal("the oldest entry should have been evicted first")
	}
	if _, ok := LoadXAIPrefixChain("grok-4.6", sessionName(XAIPrefixChainCacheMaxEntries)); !ok {
		t.Fatal("the newest entry must survive eviction")
	}
}

func sessionName(index int) string {
	return "sess-" + time.Duration(index).String()
}
