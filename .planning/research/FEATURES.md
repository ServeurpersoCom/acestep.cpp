# Feature Landscape: MP3 Bitrate Control

**Domain:** Per-request audio export quality control in an AI music generation web UI + HTTP server
**Researched:** 2026-04-18
**Confidence:** HIGH — codec constraints are from the in-tree source; UX patterns are from Audacity, Adobe, and established audio tooling

---

## Context: What Exists Today

The server already has the full bitrate infrastructure:

- `g_mp3_kbps` global (default 128) controls encoding for every request
- `mp3enc_bitrate_kbps[]` in `mp3enc-tables.h` defines the 14 valid values: 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320 kbps
- `GET /props` already returns `cli.mp3_bitrate` (the server default) in the `AceProps.cli` bag
- `ServerFields` is the established pattern for per-request server-side routing that is not part of `AceRequest`
- The format control already lives in the WebUI as a `<select>` next to the format dropdown (mp3/wav16/wav24/wav32)

The feature adds the user-facing knob; the codec already handles whatever value arrives.

---

## Table Stakes

Features users expect. Missing = control feels incomplete or broken.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Bitrate dropdown scoped to MP3 format | Every audio export tool gates bitrate options on format selection; showing kbps options when WAV is selected is confusing | Low | Control should be visible/enabled only when format == "mp3" |
| Pre-population from server default | `GET /props` already returns `cli.mp3_bitrate`; not using it means the displayed value disagrees silently with what the server will actually use | Low | Read `app.props.cli.mp3_bitrate` on load; fall back to 128 if props unavailable |
| 128, 192, 320 kbps always present | These three are the universal reference points in all audio tooling; users expect to find them | Low | Part of the curated preset list below |
| Submit bitrate in JSON body, not query param | Format already uses `?format=` query param; bitrate belongs in the request body as a `ServerFields` field alongside `synth_model`/`lora`, not as a second query param | Low | Consistent with the `ServerFields` pattern |

---

## Curated Preset List

The encoder supports 14 valid values. Exposing all 14 is anti-feature territory (see below). The right subset is the 6-8 values that cover every real use case with no gaps:

| Value | Label in UI | Use Case | Notes |
|-------|-------------|----------|-------|
| 96 kbps | 96 kbps | Background music, streaming previews | Compact; audible artifacts on headphones |
| 128 kbps | 128 kbps (default) | General sharing, social media | Server default today; transparent for most consumer devices |
| 160 kbps | 160 kbps | Better sharing quality | The "safe middle ground" many tools use |
| 192 kbps | 192 kbps | High quality sharing | Considered transparency threshold in blind tests; what Spotify Premium uses |
| 256 kbps | 256 kbps | Near-lossless distribution | Indistinguishable from WAV for most listeners |
| 320 kbps | 320 kbps | Maximum quality | Audiophile / archiving; largest file size |

Rationale for omitting 32/40/48/56/64/80/112/224 kbps:
- 32-64: Degraded audio, no legitimate music use case at 48 kHz stereo; creates support questions
- 80/112: No established mental model; users cannot reason about "112 vs 128"
- 224: Orphaned between 192 and 256; no tooling convention anchors it

The 6-value list maps cleanly to the mental model users arrive with: low / medium-default / medium-high / high / very-high / maximum.

---

## Differentiators

Features that are not expected but add polish if present.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Visual "default" marker on server default option | Makes it clear which value the operator configured without forcing users to cross-reference `/props` in their head | Low | Add `(default)` suffix or `*` marker to the option matching `app.props.cli.mp3_bitrate` |
| Bitrate hidden/greyed when format is not MP3 | Eliminates the mental overhead of "why is this kbps field active when I selected WAV?" — audio tools like Audacity grey out format-specific options | Low | CSS conditional on `app.format === 'mp3'`; does not add state complexity |
| Tooltip explaining file-size impact | "320 kbps = ~2.4 MB/min" gives users a concrete reason to choose lower bitrate for quick sharing | Low | Static text in `title` attribute, no runtime calculation needed |

---

## Anti-Features

Things that seem like improvements but damage UX.

| Anti-Feature | Why It Hurts | What to Do Instead |
|--------------|--------------|-------------------|
| Expose all 14 MPEG bitrate values (32–320) | Cognitive overload; values like 40, 48, 56, 80, 112, 224 have no established meaning to users and create choice paralysis with no benefit | Curated list of 6 values (96, 128, 160, 192, 256, 320) |
| Slider control for bitrate | Sliders imply continuous range; MP3 bitrate is a small discrete set with non-linear quality steps. A slider at "171 kbps" would silently clamp to 160 or 192 — surprising behavior. Audacity tried quality-level sliders; modern tools use dropdowns for this reason | `<select>` dropdown |
| Named quality tiers ("Low / Medium / High") instead of kbps values | Audio-literate users (the primary audience for a local inference tool) think in kbps; abstracted labels hide the actual setting and make it impossible to share "I used 192 kbps" with precision | Show kbps values directly; optionally add parenthetical descriptors |
| VBR / ABR modes | Adds a second axis of choice (constant vs variable rate) that is irrelevant for this use case; the in-tree encoder is CBR-only anyway | CBR only, no mode selection |
| Persisting bitrate in localStorage / IndexedDB | PROJECT.md explicitly rules this out: no persistence layer, reload resets all UI state | Read from `app.props.cli.mp3_bitrate` on every load |
| Auto-selecting bitrate based on duration or content | Adds complexity with no clear user benefit; operator controls the default via `--mp3-bitrate` | Per-request manual selection with server default as initial value |
| Showing bitrate in the song card / export filename | File naming is not part of this project's export model; the `Song.format` field already captures "mp3" — no need for "track_128kbps.mp3" naming | No change to Song display or file naming |

---

## Feature Dependencies on Existing Code

| New Feature Touch Point | Depends On | Notes |
|------------------------|------------|-------|
| Bitrate `<select>` visibility gated on format | `app.format` reactive variable already in `RequestForm.svelte` | Simple `{#if app.format === 'mp3'}` block |
| Initial value from server default | `app.props.cli.mp3_bitrate` already delivered via `GET /props` polling in `App.svelte`; `AceProps.cli` is `Record<string, string \| number>` | Cast to `number`; fall back to `128` if `app.props` is null |
| Sending bitrate with request | `synthSubmit` and `synthSubmitWithAudio` in `api.ts` currently pass `format` as a query param; `mp3_bitrate` goes in the JSON body as a `ServerFields` field, not a query param | Extend `AceRequest` type with optional `mp3_bitrate?: number` or add to a separate per-submit bag |
| Server parsing `mp3_bitrate` | `parse_server_fields()` in `ace-server.cpp` already handles `ServerFields`; add `mp3_bitrate` field there | Server falls back to `g_mp3_kbps` if field absent (backward-compatible) |
| Validation of bitrate value | `mp3enc_bitrate_kbps[]` is the authoritative list; server should reject or clamp invalid values | Reject with HTTP 400; log the error |

---

## UI Presentation Decision

**Recommended: `<select>` dropdown with raw kbps values**

Rationale, in priority order:

1. The existing format control is already a `<select>`. Bitrate is a second dimension of the same "output format" decision — placing a `<select>` immediately adjacent to the format dropdown creates a coherent "MP3 | 192 kbps" pair that reads as a unit.

2. `<select>` is the correct control for a small discrete set of valid values with no meaningful interpolation between them.

3. The format dropdown is positioned in the bottom toolbar row between `peak_clip` input and the submit buttons. A bitrate `<select>` inserts naturally into that row, hidden when the format is not MP3.

**Rejected alternatives:**

- Radio buttons: Appropriate for 2-4 options; 6 options in a row would overflow the compact toolbar.
- Slider: Implies continuous range; discrete clamping would be silent and surprising.
- Text input: Invites arbitrary values that the codec will reject or silently clamp; no user benefit over a fixed list.

---

## MVP Recommendation

Implement exactly these, nothing more:

1. Add `mp3_bitrate` to `ServerFields` (C++ parse + use in synth worker)
2. Add `--mp3-bitrate` to `ace-server` CLI (already partially documented in help text at line 1233 of `ace-server.cpp`; wire it up)
3. Add bitrate `<select>` to `RequestForm.svelte` with 6 presets (96 / 128 / 160 / 192 / 256 / 320)
4. Hide/disable the select when `app.format !== 'mp3'`
5. Initialize value from `app.props?.cli.mp3_bitrate ?? 128`
6. Pass value in JSON body to `/synth`

Defer: everything in the Anti-Features table.

---

## Sources

- `mp3/mp3enc-tables.h` — authoritative valid bitrate list: `{0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320}`
- `tools/ace-server.cpp` — `g_mp3_kbps`, `ServerFields`, `parse_server_fields`, `/props` endpoint
- `tools/webui/src/lib/types.ts` — `AceProps.cli`, `AceRequest`, `Song`
- `tools/webui/src/lib/api.ts` — `synthSubmit`, `synthSubmitWithAudio` (format as query param pattern)
- `tools/webui/src/components/RequestForm.svelte` — existing format `<select>` at line 882
- [Audacity MP3 Export Options](https://manual.audacityteam.org/man/mp3_export_options.html) — Preset/Variable/Average/Constant mode UI; LAME preset labels
- [MP3 Bitrate Guide (CleverUtils)](https://cleverutils.com/wav-to-mp3/bitrate-guide) — 128 vs 192 vs 256 vs 320 perceptual summary
- [Audio Bitrate 101 (Unison Audio)](https://unison.audio/audio-bitrate/) — 192 kbps as transparency threshold; streaming platform reference values
- [Coding Horror MP3 experiment](https://blog.codinghorror.com/concluding-the-great-mp3-bitrate-experiment/) — blind test data supporting 192 kbps threshold claim
