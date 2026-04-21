# Research Summary — MP3 Bitrate Control Milestone

**Project:** acestep.cpp
**Milestone:** Per-request MP3 bitrate control
**Synthesized:** 2026-04-18
**Confidence:** HIGH across all four research areas

---

## Executive Summary

This milestone adds a per-request MP3 bitrate selector to an AI music generation system that
already has all required infrastructure in place. The encoder (`audio_encode_mp3`) already accepts
`int kbps` per call and creates a fresh encoder instance per request — no re-init gymnastics or
new codec work is needed. The server already exports its configured default via `GET /props`
(`cli.mp3_bitrate`), and the `ServerFields` pattern already handles per-request output routing
fields that bypass `AceRequest`. The entire feature is a targeted wiring exercise across five
existing files with no new libraries, endpoints, or source files required.

The recommended approach is: (1) add `int mp3_bitrate` to `ServerFields` with validation against
the 14-value MPEG allowlist, (2) replace the hardcoded `g_mp3_kbps` at the single encoding call
site with the per-request value falling back to the global, and (3) add a `<select>` in the WebUI
toolbar row adjacent to the format dropdown, seeded from `app.props.cli.mp3_bitrate` on load and
visible only when format is `mp3`. A curated 6-value preset list (96–320 kbps) is strongly
preferred over exposing all 14 MPEG values.

The primary risks are silent misbehavior rather than crashes: an unvalidated bitrate produces a
wrong-rate encode with no error signal, a missing seed from `/props` silently overrides a custom
`--mp3-bitrate` server flag with 128 kbps, and a stale `index.html.gz` artifact ships the old UI
without any compile-time warning. All three are prevented by straightforward, well-defined checks
documented in the pitfalls research.

---

## Key Findings

### Stack — No New Dependencies

| Component | Status | Key Point |
|-----------|--------|-----------|
| `mp3enc` / `audio_encode_mp3` | Existing, sufficient | Takes `int kbps` per call; new encoder instance per request already |
| `yyjson` (vendor) | Existing, sufficient | One `yyjson_obj_get` + `yyjson_is_int` block — identical pattern to `lora_scale` |
| `ServerFields` struct | Existing pattern | Correct home for output-routing fields; `AceRequest` is the wrong place |
| `GET /props` | Already emits `cli.mp3_bitrate` | No new endpoint; WebUI just needs to read it |
| Svelte / TypeScript | Existing, sufficient | One new reactive field; no new npm packages |
| CMake / build | No changes | `buildwebui.sh` re-run required after `.svelte` edits, as always |

Valid bitrates (authoritative from `mp3/mp3enc-tables.h`):
32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320 kbps.

---

### Features

**Table stakes — missing any makes the control feel broken:**

1. Bitrate `<select>` scoped to MP3 format — visible/enabled only when `app.format === 'mp3'`
2. Pre-population from `app.props.cli.mp3_bitrate` on load — UI must match what server will use
3. 128, 192, 320 kbps always present — universal reference points users expect
4. Bitrate sent in JSON body (not query param) — consistent with `ServerFields` pattern

**Curated preset list (6 values, not all 14):**
96 / 128 (default) / 160 / 192 / 256 / 320 kbps.
Omits 32–80 (degraded audio, no music use case), 112 (no user mental model), 224 (orphaned).

**Differentiators (low-cost polish):**
- Mark server default option with `(default)` suffix
- Hide selector for WAV formats entirely (eliminates dead-control confusion)
- Static `title` tooltip with file-size note ("320 kbps ≈ 2.4 MB/min")

**Anti-features to avoid:**
- All 14 MPEG values — choice paralysis with no benefit over 6
- Slider — implies continuous range; discrete clamping is silent and surprising
- Named quality tiers ("Low/High") — audio-literate users think in kbps
- VBR/ABR modes — encoder is CBR-only; irrelevant complexity
- localStorage persistence of bitrate — PROJECT.md prohibits persistence layer

---

### Architecture — Five Files, Six Edits, No New Files

**Data flow:**

```
RequestForm.svelte
  app.mp3Bitrate (seeds from props.cli.mp3_bitrate on mount)
  → serialized as "mp3_bitrate" in JSON body (both plain + multipart paths)

POST /synth
  parse_server_fields() → sf.mp3_bitrate (validated; 0 if absent)
  synth_worker:
    kbps = sf.mp3_bitrate > 0 ? sf.mp3_bitrate : g_mp3_kbps
    audio_encode_mp3(..., kbps, ...)   ← was g_mp3_kbps
```

**Edit map:**

| File | Edit | What |
|------|------|------|
| `tools/ace-server.cpp` | A | Add `int mp3_bitrate = 0` to `ServerFields` (line 442) |
| `tools/ace-server.cpp` | B | Parse `"mp3_bitrate"` in `parse_server_fields()` with allowlist validation; HTTP 400 if invalid |
| `tools/ace-server.cpp` | C | Replace `g_mp3_kbps` with `(sf.mp3_bitrate > 0) ? sf.mp3_bitrate : g_mp3_kbps` at line 846 |
| `tools/webui/src/lib/types.ts` | D | Add `mp3_bitrate?: number` to server-routing fields block |
| `tools/webui/src/lib/state.svelte.ts` | E | Add `mp3Bitrate: number` to `app` state + `Saved` interface; default 0 |
| `tools/webui/src/components/RequestForm.svelte` | F | Seed from props on mount; add `<select>`; include in submit JSON |

**Build order:** edits A–F are independent, but `./buildwebui.sh` must run before `ace-server`
rebuild (C++ embeds the gzipped WebUI at compile time).

**Multipart path:** no special handling — `parse_server_fields` is already called on the JSON
blob from the `"request"` form part; the new field is covered automatically.

---

### Pitfalls

**Critical (silent wrong behavior, no compile error):**

| # | Pitfall | Prevention |
|---|---------|------------|
| 1 | Invalid bitrate encodes at wrong rate silently | Validate in `parse_server_fields` against allowlist; return HTTP 400 |
| 2 | Bitrate selector active on WAV output — appears functional, does nothing | `let bitrateEnabled = $derived(app.format === 'mp3')` |
| 3 | WebUI shows 128 kbps regardless of `--mp3-bitrate` flag on first load | Seed `app.mp3Bitrate` from `app.props.cli.mp3_bitrate` in `$effect` on mount |
| 4 | `sf.mp3_bitrate = 0` on parse failure → 0-kbps encode | Fallback `(sf.mp3_bitrate > 0) ? sf.mp3_bitrate : g_mp3_kbps` at encode site |

**Moderate:**

| # | Pitfall | Prevention |
|---|---------|------------|
| 5 | `mp3_bitrate` placed in `AceRequest` instead of `ServerFields` | Keep as `app.mp3Bitrate` app-level state; inject at submit time |
| 6 | Bitrate missing from multipart `"request"` part | Ensure both `synthSubmit` and `synthSubmitWithAudio` include the field |
| 7 | Per-element bitrate loop in batch array parse | Parse from first element only (same `obj` as `synth_model`) |
| 8 | `--mp3-bitrate` startup flag accepts invalid values | Validate `g_mp3_kbps` after `atoi` against the same allowlist |

**Minor:**

| # | Pitfall | Prevention |
|---|---------|------------|
| 9 | Stale persisted bitrate causes 400 on first submit | Validate localStorage value against allowlist on load |
| 10 | `--mp3-bitrate` undiscoverable in `server.sh` | Add commented-out `--mp3-bitrate 128` line to `server.sh` |
| 11 | Stale `index.html.gz` ships old UI without warning | Run `./buildwebui.sh` before committing; verify selector present in browser |
| 12 | AI co-author trailers in commits | Do not add `Co-Authored-By:` lines (PROJECT.md rule) |

---

## Implications for Roadmap

The work naturally splits into two sequential phases:

### Phase 1 — C++ Server

**Rationale:** Server changes are independent, testable via `curl` without the WebUI, and
establish the contract the WebUI must satisfy. Shipping this first ensures the server never
accepts an unvalidated bitrate.

Delivers: `ServerFields.mp3_bitrate`, `parse_server_fields` validation, `synth_worker` fallback,
`--mp3-bitrate` startup flag validation.

Pitfalls to prevent: 1, 4, 7, 8.

Test checkpoint: `POST /synth` with `"mp3_bitrate": 200` returns HTTP 400; with `"mp3_bitrate": 192`
encodes at 192 kbps (verify with `ffprobe`).

### Phase 2 — WebUI

**Rationale:** Depends on the server contract from Phase 1; seeding from `/props` requires a
running server; integration testing requires both layers.

Delivers: `app.mp3Bitrate` state seeded from props, bitrate `<select>` in format row (6 presets),
selector hidden for non-MP3 formats, bitrate in submit JSON for both request paths, rebuilt
WebUI artifact committed.

Pitfalls to prevent: 2, 3, 5, 6, 9, 11, 12.

Test checkpoint: start server with `--mp3-bitrate 64`; confirm selector shows 64 on load; switch
to WAV32 and confirm selector hides; switch back to MP3, select 320, submit, verify with `ffprobe`.

### Research Flags

Both phases follow well-documented existing patterns (same as `lora_scale` / `app.format`).
No additional `/gsd-research-phase` calls needed.

---

## Confidence Assessment

| Area | Confidence | Basis |
|------|------------|-------|
| Stack — no new deps | HIGH | All primitives verified by direct source inspection |
| Features — preset list | HIGH | Codec constraints from in-tree source; UX from Audacity/Adobe conventions |
| Architecture — data flow | HIGH | Every line number and function signature confirmed by direct read |
| Pitfalls | HIGH | Derived from code paths that exist today; not hypothetical |

**Gaps to address during implementation:**

- `state.svelte.ts` persistence decision: `app.format` is persisted in localStorage today.
  Confirm whether `mp3Bitrate` should also persist or be explicitly excluded before committing
  the state design. PROJECT.md discourages a persistence layer; clarify intent.
- ARCHITECTURE.md suggests exposing all 14 MPEG values in the `<select>`. FEATURES.md recommends
  the curated 6-value list. Recommendation: use the 6-value list. The 14-value approach is
  technically correct but is an anti-feature per the features research.

---

## Sources (aggregated)

- `mp3/mp3enc-tables.h` — `mp3enc_bitrate_kbps[]` authoritative valid bitrate list
- `mp3/mp3enc.h` — `mp3enc_init` signature; lowpass table (lines 131–147)
- `src/audio-io.h:609` — `audio_encode_mp3` signature
- `tools/ace-server.cpp` — `ServerFields` (442), `parse_server_fields` (449), `synth_worker` (732–876), `handle_synth` (891–968), `/props` (1173–1177), `--mp3-bitrate` flag (1279–1281)
- `tools/webui/src/lib/types.ts` — `AceProps`, `AceRequest`, `Song`
- `tools/webui/src/lib/state.svelte.ts` — `app` state, `Saved` interface, format restore (line 22)
- `tools/webui/src/lib/api.ts` — `synthSubmit`, `synthSubmitWithAudio`
- `tools/webui/src/components/RequestForm.svelte` — format `<select>` (line 882)
- `.planning/PROJECT.md` — constraints, PR hygiene rules
- Audacity MP3 Export Options — mode/quality UI reference
- Coding Horror MP3 experiment — 192 kbps transparency threshold data
- Unison Audio bitrate guide — streaming platform reference values
