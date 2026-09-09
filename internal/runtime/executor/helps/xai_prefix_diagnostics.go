package helps

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"

	internalcache "github.com/router-for-me/CLIProxyAPI/v7/internal/cache"
	"github.com/tidwall/gjson"
)

// Prefix segment labels that are not input items.
const (
	PrefixSegmentInstructions = "instructions"
	PrefixSegmentTools        = "tools"
)

// Drift reasons reported by ClassifyPrefixDrift.
const (
	PrefixDriftNone                = "none"
	PrefixDriftInstructionsChanged = "instructions_changed"
	PrefixDriftToolsChanged        = "tools_changed"
	PrefixDriftReplayItemMissing   = "replay_item_missing"
	PrefixDriftEncryptedStripped   = "encrypted_content_stripped"
	PrefixDriftHistoryRewritten    = "history_rewritten"
	PrefixDriftSessionKeyChanged   = "session_key_changed"
	PrefixDriftAuthSwitched        = "auth_switched"
	PrefixDriftUnknown             = "unknown"
)

// BuildResponsesPrefixChain builds a rolling hash chain over the cacheable
// prefix of a final upstream Responses body, in the exact order the provider
// sees it: instructions, then the whole tools array, then every input item as
// its raw bytes. It returns an empty chain when the body carries no input array.
//
// This deliberately does NOT reuse sdk/cliproxy/session/lcp.go's rollingPrefixKeys.
// That chain hashes canonicalized conversation turns: it drops reasoning parts and
// masks timestamps and UUIDs so that cosmetically different requests still match for
// credential affinity. Upstream prompt caching keys on the literal bytes we send, so
// this diagnostic has to hash those bytes verbatim - the two chains answer different
// questions and must not be collapsed.
func BuildResponsesPrefixChain(body []byte) internalcache.XAIPrefixChain {
	root := gjson.ParseBytes(body)
	input := root.Get("input")
	if !input.IsArray() {
		return internalcache.XAIPrefixChain{}
	}
	inputItems := input.Array()

	chain := internalcache.XAIPrefixChain{
		Keys:           make([]string, 0, len(inputItems)+2),
		Labels:         make([]string, 0, len(inputItems)+2),
		Kinds:          make([]string, 0, len(inputItems)+2),
		PromptCacheKey: strings.TrimSpace(root.Get("prompt_cache_key").String()),
	}

	var previous [sha256.Size]byte
	appendSegment := func(label, kind, segment string) {
		hash := sha256.New()
		_, _ = hash.Write(previous[:])
		_, _ = hash.Write([]byte{0})
		_, _ = hash.Write([]byte(segment))
		hash.Sum(previous[:0])
		chain.Keys = append(chain.Keys, hex.EncodeToString(previous[:]))
		chain.Labels = append(chain.Labels, label)
		chain.Kinds = append(chain.Kinds, kind)
	}

	appendSegment(PrefixSegmentInstructions, PrefixSegmentInstructions, root.Get("instructions").Raw)
	appendSegment(PrefixSegmentTools, PrefixSegmentTools, root.Get("tools").Raw)
	for index, item := range inputItems {
		kind := strings.TrimSpace(item.Get("type").String())
		if kind == "" {
			kind = "message"
		}
		appendSegment(fmt.Sprintf("input[%d]:%s", index, kind), kind, item.Raw)
	}
	return chain
}

// ComparePrefixChains reports how many leading segments the two chains share.
// The first differing segment is at that same index, so a caller can use the
// count both as the reuse length and as the divergence position.
func ComparePrefixChains(previous, next internalcache.XAIPrefixChain) int {
	limit := len(previous.Keys)
	if len(next.Keys) < limit {
		limit = len(next.Keys)
	}
	reused := 0
	for reused < limit && previous.Keys[reused] == next.Keys[reused] {
		reused++
	}
	return reused
}

// ClassifyPrefixDrift names the most likely cause of a cache miss between two
// consecutive turns. A change of credential or prompt_cache_key is reported even
// when the hashed prefix is identical, because either one makes the upstream
// cache cold on its own. It reports PrefixDriftNone when next simply extends
// previous on the same account and cache key.
func ClassifyPrefixDrift(previous, next internalcache.XAIPrefixChain, firstDivergence int) string {
	// Namespace changes outrank positional ones: when the account or the cache
	// key changes the upstream cache is cold no matter how stable the prefix is.
	if previous.AuthID != next.AuthID {
		return PrefixDriftAuthSwitched
	}
	if previous.PromptCacheKey != next.PromptCacheKey {
		return PrefixDriftSessionKeyChanged
	}
	if firstDivergence >= len(previous.Keys) || firstDivergence >= len(next.Keys) {
		return PrefixDriftNone
	}
	previousKind := previous.Kinds[firstDivergence]
	nextKind := next.Kinds[firstDivergence]
	switch {
	case previousKind == PrefixSegmentInstructions:
		return PrefixDriftInstructionsChanged
	case previousKind == PrefixSegmentTools:
		return PrefixDriftToolsChanged
	case previousKind == "reasoning" && nextKind != "reasoning":
		// The previous turn carried a reasoning item in this slot and this one
		// does not: a replayed item was dropped rather than history being edited.
		return PrefixDriftReplayItemMissing
	case previousKind == "reasoning" && nextKind == "reasoning":
		// Same slot, same kind, different bytes. encrypted_content is the only
		// field of a reasoning item this pipeline rewrites.
		return PrefixDriftEncryptedStripped
	case previousKind == nextKind:
		return PrefixDriftHistoryRewritten
	default:
		return PrefixDriftUnknown
	}
}

// PrefixSegmentKind returns the segment kind at index, or "" when out of range.
func PrefixSegmentKind(chain internalcache.XAIPrefixChain, index int) string {
	if index < 0 || index >= len(chain.Kinds) {
		return ""
	}
	return chain.Kinds[index]
}

// PrefixSegmentLabel returns the segment label at index, or "" when out of range.
func PrefixSegmentLabel(chain internalcache.XAIPrefixChain, index int) string {
	if index < 0 || index >= len(chain.Labels) {
		return ""
	}
	return chain.Labels[index]
}
