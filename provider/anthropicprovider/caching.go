// Copyright (c) Microsoft. All rights reserved.

package anthropicprovider

import (
	"github.com/anthropics/anthropic-sdk-go"
	"github.com/microsoft/agent-framework-go/message"
)

// Ephemeral5m and Ephemeral1h are the cache lifetimes Anthropic supports for a
// cache_control breakpoint. They are the anthropic-sdk-go values, surfaced here as
// named constants so callers do not have to reach for the SDK's verbose identifiers.
//
// Their type is the SDK's own anthropic.CacheControlEphemeralTTL, deliberately: this
// package already hands raw SDK types across its public surface (see MessageNewParams),
// so a provider-specific caching primitive stays consistent by doing the same rather
// than wrapping the SDK enum in a parallel type the caller would have to convert.
const (
	// Ephemeral5m caches for 5 minutes. This is also the default when no TTL is set.
	Ephemeral5m = anthropic.CacheControlEphemeralTTLTTL5m
	// Ephemeral1h caches for 1 hour.
	Ephemeral1h = anthropic.CacheControlEphemeralTTLTTL1h
)

// CacheControl describes an Anthropic cache_control breakpoint on a single content block.
type CacheControl struct {
	// TTL is the cache lifetime. The zero value ("") means the Anthropic default of
	// 5-minute ephemeral caching; Ephemeral1h selects the 1-hour tier.
	TTL anthropic.CacheControlEphemeralTTL
}

// CacheControlOption configures the CacheControl attached by WithCacheControl.
type CacheControlOption func(*CacheControl)

// WithTTL sets the cache lifetime for the breakpoint. Passing the zero value leaves
// Anthropic's default (5 minutes) in effect.
func WithTTL(ttl anthropic.CacheControlEphemeralTTL) CacheControlOption {
	return func(cc *CacheControl) { cc.TTL = ttl }
}

// cacheControlKey is the package-private key under which a cache_control marker is stored
// in a content's AdditionalProperties bag. Unexported and package-qualified so it cannot
// collide with a key a caller sets for their own purposes.
const cacheControlKey = "anthropicprovider.cacheControl"

// WithCacheControl attaches an Anthropic cache_control breakpoint to a single
// message.Content, in place, and returns it for chaining. Caching is expressed on the
// individual content block, not toggled agent-wide, so the caller decides exactly which
// blocks become cache breakpoints.
//
// The marker is stored in the content's ContentHeader.AdditionalProperties under a
// package-private key and read back in buildMessageParam, which transfers it onto whatever
// Anthropic block the content becomes: text, image, document, tool-use, or tool-result. A
// content whose block type has no cache_control field on the wire (the thinking and
// redacted_thinking variants) is a clean no-op rather than an error.
//
// Opt-in, deliberately. A cache WRITE costs more than a plain input token, so caching is a
// net loss for a short single-turn prompt; nothing here turns it on unless the caller asks.
//
// Breakpoint budget: Anthropic permits at most 4 cache_control breakpoints per request
// (the tool definitions, the system prompt, and message blocks all draw from the same
// budget). The caller is responsible for staying within it; this function does not dedupe
// or cap, and a request with too many breakpoints is rejected by the API. Higher-level
// automatic placement is intentionally left to a future change.
//
// For full control over the raw request, including breakpoints on tools or the system
// prompt, use MessageNewParams as the low-level escape hatch.
func WithCacheControl(c message.Content, opts ...CacheControlOption) message.Content {
	var cc CacheControl
	for _, opt := range opts {
		opt(&cc)
	}
	markCacheControl(c, cc)
	return c
}

// markCacheControl writes the cache_control marker into the content's own
// AdditionalProperties. Header returns the content's header by pointer, so this mutates
// the caller's content in place.
func markCacheControl(c message.Content, cc CacheControl) {
	if c == nil {
		return
	}
	h := c.Header()
	if h.AdditionalProperties == nil {
		h.AdditionalProperties = make(map[string]any)
	}
	h.AdditionalProperties[cacheControlKey] = cc
}

// cacheControlOf reads back a marker set by WithCacheControl.
func cacheControlOf(c message.Content) (CacheControl, bool) {
	cc, ok := c.Header().AdditionalProperties[cacheControlKey].(CacheControl)
	return cc, ok
}

// applyContentCacheControl transfers a WithCacheControl marker from a message.Content onto
// the Anthropic block(s) built from it. No marker means no change (opt-in). A content that
// produced no block, or whose block type has no cache_control field on the wire (thinking,
// redacted_thinking), is left uncached rather than rejected.
func applyContentCacheControl(c message.Content, blocks []anthropic.ContentBlockParamUnion) {
	cc, ok := cacheControlOf(c)
	if !ok || len(blocks) == 0 {
		return
	}
	// A breakpoint caches the prefix up to and including its block, so marking the last
	// block built for this content covers all of it.
	setCacheControl(&blocks[len(blocks)-1], cc.TTL)
}

// setCacheControl marks a content block as a cache breakpoint with the given TTL,
// reporting whether it could. The SDK's GetCacheControl returns the populated variant's
// cache_control field, or nil for a variant without one, so every cacheable block type,
// including ones the SDK adds later, is handled without a per-variant switch here.
//
// An empty TTL leaves CacheControlEphemeralParam.TTL at its zero value, which the SDK
// omits from the wire, so Anthropic applies its 5-minute default.
func setCacheControl(b *anthropic.ContentBlockParamUnion, ttl anthropic.CacheControlEphemeralTTL) bool {
	p := b.GetCacheControl()
	if p == nil {
		return false
	}
	*p = anthropic.NewCacheControlEphemeralParam()
	p.TTL = ttl
	return true
}
