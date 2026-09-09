package cache

import (
	"sort"
	"strings"
	"sync"
	"time"
)

const (
	// XAIPrefixChainCacheTTL limits how long a session's last prompt prefix
	// chain stays in process memory. It matches the reasoning replay TTL so both
	// diagnostics and replay observe the same session lifetime.
	XAIPrefixChainCacheTTL = 1 * time.Hour

	// XAIPrefixChainCacheMaxEntries bounds process memory. Oldest entries are
	// evicted first. This cache is diagnostics-only, so it is deliberately much
	// smaller than the replay cache.
	XAIPrefixChainCacheMaxEntries = 4096

	// XAIPrefixChainCacheEvictBatchSize leaves headroom after the cache reaches
	// capacity so high write volume does not rescan the map every turn.
	XAIPrefixChainCacheEvictBatchSize = 128
)

// XAIPrefixChain is a rolling hash chain over the cacheable prefix of an
// upstream Responses body, in the order the provider sees it.
//
// It lives here rather than next to its builder because this package is the
// lower layer: internal/runtime/executor/helps already imports internal/cache,
// so the type has to be declared here for the store to hold it.
type XAIPrefixChain struct {
	// Keys[i] is the rolling hash covering segments 0..i.
	Keys []string
	// Labels[i] names segment i, e.g. "instructions", "tools", "input[7]:reasoning".
	Labels []string
	// Kinds[i] is the segment's item type, e.g. "reasoning", "function_call_output".
	Kinds []string
	// PromptCacheKey is the upstream prompt_cache_key carried by the same body.
	PromptCacheKey string
	// AuthID identifies the credential this turn was sent on. The upstream
	// prompt cache is scoped per account, so a switch makes the cache cold even
	// when every hashed segment is identical.
	AuthID string
}

// Len reports the number of prefix segments.
func (chain XAIPrefixChain) Len() int {
	return len(chain.Keys)
}

type xaiPrefixChainEntry struct {
	Chain     XAIPrefixChain
	Timestamp time.Time
}

var (
	xaiPrefixChainMu      sync.Mutex
	xaiPrefixChainEntries = make(map[string]xaiPrefixChainEntry)
	// xaiPrefixChainNow is injectable so TTL tests can advance time without
	// sleeping on a wall clock.
	xaiPrefixChainNow = time.Now
)

func xaiPrefixChainCacheKey(modelName, sessionKey string) string {
	modelName = strings.TrimSpace(modelName)
	sessionKey = strings.TrimSpace(sessionKey)
	if modelName == "" || sessionKey == "" {
		return ""
	}
	// Mirror the replay cache: the session key is the continuity boundary and
	// stays independent of the selected upstream credential.
	return strings.Join([]string{"xai-prefix-chain", modelName, sessionKey}, "\x00")
}

// LoadXAIPrefixChain returns the chain stored for the previous turn of this
// session together with how long ago that turn ran. The idle gap matters because
// the upstream prompt cache expires on its own: a long pause produces a cold
// cache even when this proxy sends a byte-identical prefix.
func LoadXAIPrefixChain(modelName, sessionKey string) (XAIPrefixChain, time.Duration, bool) {
	key := xaiPrefixChainCacheKey(modelName, sessionKey)
	if key == "" {
		return XAIPrefixChain{}, 0, false
	}
	now := xaiPrefixChainNow()
	xaiPrefixChainMu.Lock()
	defer xaiPrefixChainMu.Unlock()
	entry, ok := xaiPrefixChainEntries[key]
	if !ok {
		return XAIPrefixChain{}, 0, false
	}
	idle := now.Sub(entry.Timestamp)
	if idle > XAIPrefixChainCacheTTL {
		delete(xaiPrefixChainEntries, key)
		return XAIPrefixChain{}, 0, false
	}
	if idle < 0 {
		idle = 0
	}
	return entry.Chain, idle, true
}

// StoreXAIPrefixChain records this turn's chain as the baseline for the next one.
func StoreXAIPrefixChain(modelName, sessionKey string, chain XAIPrefixChain) {
	key := xaiPrefixChainCacheKey(modelName, sessionKey)
	if key == "" || chain.Len() == 0 {
		return
	}
	cacheCleanupOnce.Do(startCacheCleanup)
	now := xaiPrefixChainNow()
	xaiPrefixChainMu.Lock()
	defer xaiPrefixChainMu.Unlock()
	xaiPrefixChainEntries[key] = xaiPrefixChainEntry{Chain: chain, Timestamp: now}
	if len(xaiPrefixChainEntries) > XAIPrefixChainCacheMaxEntries {
		evictOldestXAIPrefixChainEntriesLocked(XAIPrefixChainCacheEvictBatchSize)
	}
}

func evictOldestXAIPrefixChainEntriesLocked(count int) {
	if count <= 0 || len(xaiPrefixChainEntries) == 0 {
		return
	}
	type candidate struct {
		key       string
		timestamp time.Time
	}
	candidates := make([]candidate, 0, len(xaiPrefixChainEntries))
	for key, entry := range xaiPrefixChainEntries {
		candidates = append(candidates, candidate{key: key, timestamp: entry.Timestamp})
	}
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].timestamp.Before(candidates[j].timestamp)
	})
	if count > len(candidates) {
		count = len(candidates)
	}
	for i := 0; i < count; i++ {
		delete(xaiPrefixChainEntries, candidates[i].key)
	}
}

func purgeExpiredXAIPrefixChainCache(now time.Time) {
	xaiPrefixChainMu.Lock()
	for key, entry := range xaiPrefixChainEntries {
		if now.Sub(entry.Timestamp) > XAIPrefixChainCacheTTL {
			delete(xaiPrefixChainEntries, key)
		}
	}
	xaiPrefixChainMu.Unlock()
}
