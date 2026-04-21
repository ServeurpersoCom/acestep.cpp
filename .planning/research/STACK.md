# Technology Stack — MP3 Bitrate Control Feature

**Project:** acestep.cpp
**Milestone:** Per-request MP3 bitrate control
**Researched:** 2026-04-18

---

## Verdict: No New Libraries or Build Changes Required

Every component needed for this feature already exists and is wired together. The entire
feature is implemented by modifying existing files only.

---

## Existing Stack Audit

### C++ Server — `tools/ace-server.cpp`

| Component | Current State | Change Needed |
|---|---|---|
| `g_mp3_kbps` (line 194) | Global `int`, default 128 | Keep; also used as fallback default |
| `ServerFields` struct (line 442) | Holds `synth_model`, `lm_model`, `lora`, `lora_scale` | Add `int mp3_kbps` field |
| `parse_server_fields()` (line 449) | Parses the four fields above from request JSON via yyjson | Add yyjson parse for `"mp3_bitrate"` key |
| `audio_encode_mp3()` call (line 846) | Passes `g_mp3_kbps` as the `kbps` argument | Change to pass `sf.mp3_kbps` |
| `GET /props` handler (line 1177) | Already emits `cli.mp3_bitrate = g_mp3_kbps` | No change needed |

The call site at line 846 is:
```cpp
encoded[b] = audio_encode_mp3(audio[b].samples, audio[b].n_samples, 48000, g_mp3_kbps, ...);
```
Changing `g_mp3_kbps` to `sf.mp3_kbps` (after `sf` is populated with a per-request value
falling back to `g_mp3_kbps`) is the complete C++ change at the encoding site.

### MP3 Encoder API — `mp3/mp3enc.h` + `src/audio-io.h`

**Critical finding: the encoder does NOT support per-call bitrate changes. Bitrate is set
at init time only via `mp3enc_init(sample_rate, channels, bitrate_kbps)` and stored in
`enc->bitrate_kbps`. There is no `mp3enc_set_bitrate()` or equivalent.**

This is not a problem. `audio_encode_mp3()` in `src/audio-io.h` already accepts `kbps` as
a parameter (line 609: `int kbps`) and creates a fresh encoder per call:

```cpp
static std::string audio_encode_mp3(const float * audio, int T_audio, int sr, int kbps,
                                    bool (*cancel)(void *) = nullptr,
                                    void * cancel_data     = nullptr)
```

Each synthesis request already creates its own encoder instance. Passing a different `kbps`
value per call is the correct and complete mechanism. No encoder re-init gymnastics needed —
a new instance is already created per request.

**Valid bitrates** are those in `mp3enc_bitrate_kbps[15]` (`mp3/mp3enc-tables.h`):
`{0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320}` kbps.
The value `0` is a table sentinel (free bitrate slot); valid user-selectable range is
32–320 kbps from the non-zero entries.

The encoder does not validate the bitrate value itself (it uses the value to look up frame
geometry from tables). Server-side validation against the known-good list is required before
passing to `audio_encode_mp3`.

### JSON Parsing — `vendor/yyjson`

Already used in `parse_server_fields()` for the same pattern (extract a numeric field from
the request JSON object). Adding `mp3_bitrate` is a single `yyjson_obj_get` + `yyjson_is_int`
block, identical in structure to the existing `lora_scale` extraction.

### Svelte WebUI — `tools/webui/src/`

| Component | Current State | Change Needed |
|---|---|---|
| `AceProps` type (`lib/types.ts:50`) | `cli` typed as `Record<string, string \| number>` — already admits `cli.mp3_bitrate` | No type change needed; field already present at runtime |
| `app.props` (`lib/state.svelte.ts:51`) | Populated from `GET /props` on load | No change needed |
| Format `<select>` (`RequestForm.svelte:882`) | Bound to `app.format`; controls WAV vs MP3 | Add adjacent bitrate `<select>` that is visible/enabled only when `app.format === 'mp3'` |
| `app.format` | Stored as a string; persisted to localStorage | No change needed |
| Request submission (`api.ts`) | Sends JSON body including server fields | Add `mp3_bitrate` to the sent JSON when format is `mp3` |

The `cli.mp3_bitrate` field is already present in the `GET /props` response today (server
emits it at line 1177). The WebUI just does not read it yet. Reading it for pre-population
is `app.props.cli.mp3_bitrate as number` — no new fetch, no new endpoint.

### Build System — CMake + WebUI Build

No CMake changes. No new source files. No new npm packages.

The WebUI build pipeline (`buildwebui.sh` → `tools/public/index.html.gz` → C++ rebuild)
is unchanged in process; the Svelte file modification triggers a required re-run, same as
any other UI change.

---

## Integration Points Summary

| Touch Point | File | Nature of Change |
|---|---|---|
| `ServerFields` struct | `tools/ace-server.cpp` | Add `int mp3_kbps` field |
| `parse_server_fields()` | `tools/ace-server.cpp` | Parse `"mp3_bitrate"` from JSON; validate against table; fall back to `g_mp3_kbps` |
| `audio_encode_mp3` call | `tools/ace-server.cpp` | Replace `g_mp3_kbps` with `sf.mp3_kbps` |
| Bitrate `<select>` | `tools/webui/src/components/RequestForm.svelte` | New `<select>` element, shown only for MP3 format |
| Props pre-population | `tools/webui/src/components/RequestForm.svelte` | Read `app.props?.cli.mp3_bitrate` to initialize the new select's value |
| Request JSON send | `tools/webui/src/lib/api.ts` or inline in submit handler | Include `mp3_bitrate` in POST body when format is `mp3` |

---

## No New Dependencies

| Category | Status |
|---|---|
| C++ libraries | None required |
| npm packages | None required |
| CMake changes | None required |
| New source files | None required |
| New API endpoints | None required (`GET /props` already serves the default) |

---

## Confidence

| Area | Confidence | Basis |
|---|---|---|
| mp3enc API is init-time only | HIGH | Direct source read of `mp3enc_init()` signature and `mp3enc_t` struct; no setter function exists |
| `audio_encode_mp3` accepts per-call `kbps` | HIGH | Direct source read of function signature at `src/audio-io.h:609` |
| `ServerFields` is the correct pattern | HIGH | Confirmed by PROJECT.md key decision and by direct inspection of existing fields |
| `GET /props` already emits `cli.mp3_bitrate` | HIGH | Direct source read of `ace-server.cpp:1177` |
| `AceProps.cli` already typed to accept the field | HIGH | Direct source read of `lib/types.ts:50`; `Record<string, string \| number>` |
| No new libraries needed | HIGH | All required primitives verified present in current codebase |
