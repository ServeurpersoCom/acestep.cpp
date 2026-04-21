# Domain Pitfalls: Per-Request MP3 Bitrate Control

**Domain:** Adding per-request MP3 bitrate control to an existing C++17 audio generation server + Svelte WebUI
**Researched:** 2026-04-18
**Codebase commit context:** acestep.cpp master, webui at current HEAD

---

## Critical Pitfalls

These mistakes cause incorrect behavior with no compile error and no obvious runtime signal.

### Pitfall 1: Invalid Bitrate Silently Produces a Corrupt or Wrong-Rate MP3

**What goes wrong:**
`mp3enc_bitrate_kbps[]` in `mp3/mp3enc-tables.h` defines exactly 15 valid values — index 0 is free-format (0 kbps, not a usable bitrate), indices 1–14 are: 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320. The encoder looks up the bitrate by scanning this table for a match. If a client sends an arbitrary integer (e.g., `200`, `0`, `-1`, `99999`) that is not in this table, the encoder either falls through to a default or encodes at the wrong rate depending on implementation. There is no existing server-side guard — `g_mp3_kbps = atoi(argv[++i])` in the `--mp3-bitrate` startup path accepts any integer without validation.

**Why it happens:**
The `parse_server_fields` function currently handles `synth_model`, `lm_model`, `lora`, and `lora_scale` with no range or allowlist checking. Adding `mp3_bitrate` as a raw integer read without consulting `mp3enc_bitrate_kbps[]` continues this pattern incorrectly.

**Consequences:**
The resulting file will play but at an unexpected bitrate (or the encoder may assert/crash with an out-of-range table index). Clients get no error. The job status will say `done` with corrupt or wrong-quality audio.

**Prevention:**
In `parse_server_fields`, after reading `mp3_bitrate` from JSON, validate against the known table:

```cpp
static const int valid_bitrates[] = {32,40,48,56,64,80,96,112,128,160,192,224,256,320};
// reject the value if it does not appear in valid_bitrates[]
```

Return HTTP 400 with an informative error message listing valid values. Do not silently clamp; clamping hides client bugs.

Also apply the same guard to the `--mp3-bitrate` startup flag (`g_mp3_kbps`) so the server fails loudly at launch with a bad flag rather than encoding silently wrong.

**Detection:**
During implementation, send `mp3_bitrate: 200` and verify a 400 error is returned. Send `mp3_bitrate: 0` and verify the same.

**Phase:** C++ server implementation phase.

---

### Pitfall 2: Bitrate Applied to WAV Output (Silent No-Op That Confuses Callers)

**What goes wrong:**
In `synth_worker` (line 843–848 of `ace-server.cpp`), the encode path branches on `output_wav`. When `output_wav == true`, `audio_encode_wav` is called and `g_mp3_kbps` is never consulted. If a client sends `mp3_bitrate: 320` together with `?format=wav16`, the bitrate value is parsed into `sf.mp3_bitrate`, carried into the lambda capture, and then silently discarded — the server encodes WAV at the requested bit depth and the bitrate value has no effect.

The specific risk: a client that inspects the response Content-Type (`audio/wav`) and sees it received WAV has no way to know whether the bitrate was honored. If the WebUI bitrate selector is enabled when format is WAV, users may believe the setting is doing something.

**Why it happens:**
`output_wav` is determined from the `?format=` query parameter (line 981–988). The bitrate field arrives in the JSON body. These two paths are orthogonal and neither checks the other.

**Prevention — Server:**
No server-side action needed (WAV has no bitrate concept). The server must not apply `mp3_bitrate` when `output_wav == true`. This is the natural result of keeping the bitrate logic inside the `else` branch of `if (output_wav)`. Do not "error" on bitrate+WAV — the client may be iterating format options while keeping the bitrate field in their request JSON.

**Prevention — WebUI:**
Disable or hide the bitrate selector when `app.format` is `wav16`, `wav24`, or `wav32`. The `<select>` for format is at `RequestForm.svelte:882–889` and `app.format` drives it. A reactive `$derived` flag is sufficient:

```ts
let bitrateEnabled = $derived(app.format === 'mp3');
```

Without this, every WAV user will see a bitrate control that appears functional but is not.

**Detection:**
Submit a request with `?format=wav16` and `mp3_bitrate: 320`. Confirm the server returns WAV and no error. Confirm the WebUI grays out the bitrate selector when a WAV format is selected.

**Phase:** WebUI implementation phase; confirm expected behavior in server phase.

---

### Pitfall 3: WebUI Bitrate State Does Not Reflect Server Default on First Load

**What goes wrong:**
`app.format` is persisted in `localStorage` and initializes to `'mp3'` on first load. There is currently no `app.mp3Bitrate` analogue — if the bitrate selector is added with a hard-coded default (e.g., 128), a server started with `--mp3-bitrate 320` will show 128 in the UI on first load even though the server default is 320. Any request submitted before the user changes the selector will explicitly send `mp3_bitrate: 128` to a 320-default server — overriding the server default with a lower quality setting the user did not choose.

**Why it happens:**
`GET /props` already returns `cli.mp3_bitrate` (line 1177 of `ace-server.cpp`). The `AceProps` interface in `types.ts` types `cli` as `Record<string, string | number>`, so `app.props?.cli?.mp3_bitrate` is already available. However, nothing currently reads this value and seeds the bitrate state.

**Prevention:**
On first load (or whenever `app.props` changes), initialize the bitrate state from `app.props.cli.mp3_bitrate` if the user has not previously saved a preference. The pattern already used for `max_batch` (line 687 of `RequestForm.svelte`: `app.props?.cli?.max_batch || 9`) is the correct model. Do not use `app.props.cli.mp3_bitrate` as the live binding — use it only to seed an `app.mp3Bitrate` state that the user can subsequently override.

If bitrate is added to `localStorage` persistence (like `app.format` is), the first-load seeding must respect the following priority: localStorage value (if present) > server default (from `/props`) > hard-coded fallback (128). This matches the existing `format` restore logic in `state.svelte.ts:22`.

**Detection:**
Start server with `--mp3-bitrate 64`. Load WebUI. Confirm the bitrate selector shows 64 without the user touching anything.

**Phase:** WebUI implementation phase.

---

### Pitfall 4: Multipart Path Does Not Call `parse_server_fields` on the Right Body

**What goes wrong:**
In `handle_synth` (line 907–968), the multipart branch calls `parse_server_fields(json_body.c_str(), &sf)` where `json_body` is the content of the `"request"` form part. The plain JSON branch calls `parse_server_fields(req.body.c_str(), &sf)`. Both paths correctly call `parse_server_fields` before the worker lambda is created. However, if the multipart `"request"` part is missing (client sends audio but omits the JSON part), the multipart branch falls through to an error at line 917 before `parse_server_fields` is ever called, and `sf` remains at its zero-initialized defaults — meaning `sf.mp3_bitrate` would be 0, not `g_mp3_kbps`.

**Why it happens:**
`ServerFields sf;` is declared with no initializer at line 898. Fields are zero-initialized (C++ zero-init for `int` in a struct with a value-initialized constructor), but `0` is not a valid bitrate. The fallback to `g_mp3_kbps` must happen explicitly.

**Prevention:**
In `synth_worker`, where `g_mp3_kbps` is currently used directly (line 846), change to:

```cpp
const int bitrate = (sf.mp3_bitrate > 0) ? sf.mp3_bitrate : g_mp3_kbps;
```

This single line makes the server default the fallback for all missing or zero values, regardless of whether the request was multipart, plain JSON, or errored before parsing. It also means a client that omits `mp3_bitrate` entirely gets the server default rather than a 0-kbps encode.

**Detection:**
Send a multipart request with audio but no `"request"` part. Confirm the server returns 400. Then send a multipart request with audio and a `"request"` part that omits `mp3_bitrate`. Confirm output is encoded at the server default bitrate.

**Phase:** C++ server implementation phase.

---

## Moderate Pitfalls

### Pitfall 5: `mp3_bitrate` Added to `AceRequest` Instead of `ServerFields`

**What goes wrong:**
`PROJECT.md` records the decision: add `mp3_bitrate` to `ServerFields`, not `AceRequest`. `AceRequest` maps to the generation parameters (model inputs). `ServerFields` maps to output routing and encoding (model outputs). Placing `mp3_bitrate` in `AceRequest` would cause it to be serialized into the LM result JSON, stored in `IndexedDB` alongside songs, included in the batch array, and potentially forwarded to the LM worker — none of which make sense for a post-generation encoding setting.

**Prevention:**
Add `int mp3_bitrate = 0;` to the `ServerFields` struct (line 442–447 of `ace-server.cpp`) and parse it in `parse_server_fields`. Do not add it to `request.h` or the TypeScript `AceRequest` interface. In the TypeScript layer, `mp3_bitrate` is app-level state (like `app.format`) not request-level state.

**Detection:**
Confirm that a JSON export of the request (the "Export" function in the WebUI, if present) does not include `mp3_bitrate`.

**Phase:** C++ server implementation phase, TypeScript state design.

---

### Pitfall 6: WebUI Sends `mp3_bitrate` Inside the JSON Body for Plain JSON Requests but Not for Multipart Requests

**What goes wrong:**
`synthSubmit` builds the body as `JSON.stringify(reqs[0])` where `reqs[0]` is an `AceRequest`. If `mp3_bitrate` is added to `AceRequest` (see Pitfall 5), it goes into the body for plain JSON but not for multipart — because `synthSubmitWithAudio` also uses `JSON.stringify(reqs[0])` for the `"request"` form part. However, if `mp3_bitrate` is instead a separate field appended directly to the serialized object before submission (like a field injected at call site), it would need to be explicitly included in both the plain JSON body and the multipart `"request"` part.

The existing pattern for `synth_model` and `lora_scale` in `api.ts` handles this correctly: those fields live inside `AceRequest` so `JSON.stringify` captures them in both paths automatically.

**Prevention:**
Whatever mechanism carries `mp3_bitrate` to the server must be present in both paths. Options:
- If `mp3_bitrate` is sent as a JSON field, include it in the request object serialized in both `synthSubmit` and `synthSubmitWithAudio`.
- If `mp3_bitrate` is sent as a query parameter (e.g., `synth?format=mp3&mp3_bitrate=320`), ensure `synthSubmitWithAudio` adds it to the URL the same way `synthSubmit` does.

The query-parameter approach avoids polluting `AceRequest` and keeps it symmetric with `?format=`, which is already a query param.

**Detection:**
Test bitrate selection with a cover/repaint task (which uses multipart with audio). Confirm the correct bitrate is used, not the server default.

**Phase:** WebUI implementation phase.

---

### Pitfall 7: `parse_server_fields` Array Path Reads From First Element Only

**What goes wrong:**
`parse_server_fields` handles batch arrays by reading server fields from the first element only (line 467–468: `if (yyjson_is_arr(root)) { obj = yyjson_arr_get_first(root); }`). This is the correct and intentional behavior — server routing fields apply to the whole job. However, when implementing `mp3_bitrate` parsing here, it must not attempt to merge or validate per-element bitrates from the rest of the array. The bitrate is job-level, not per-track.

**Prevention:**
Parse `mp3_bitrate` from the same `obj` (first element) where `synth_model` and `lora` are already parsed. Do not loop over array elements for this field.

**Phase:** C++ server implementation phase.

---

### Pitfall 8: `--mp3-bitrate` Flag Not Validated at Startup

**What goes wrong:**
Currently, `g_mp3_kbps = atoi(argv[++i])` (line 1281) accepts any integer, including zero and negative values, at server startup. If a user starts with `--mp3-bitrate 0`, all MP3 encodes for the lifetime of the server will be at an invalid bitrate. Because the startup path does not call `parse_server_fields` (the flag is handled directly in `main`), any validation added to `parse_server_fields` does not protect startup.

**Prevention:**
After setting `g_mp3_kbps` from `atoi`, validate against the same allowlist and print a usage error and exit if invalid:

```cpp
g_mp3_kbps = atoi(argv[++i]);
// validate: must be in mp3enc_bitrate_kbps[1..14]
```

**Phase:** C++ server implementation phase (same change set as the `parse_server_fields` validation).

---

## Minor Pitfalls

### Pitfall 9: WebUI Bitrate State Persists Across Format Changes in Unexpected Ways

**What goes wrong:**
`app.format` is persisted in `localStorage`. If `app.mp3Bitrate` is also persisted, a user who: sets bitrate to 320, switches to WAV32, closes the browser, re-opens — will see the bitrate selector still at 320 (correct), but the format selector at WAV32 (also correct). The interaction is fine. However, if the persisted bitrate value is later deemed invalid because the server is restarted with a different valid-bitrate list (not currently possible since the list is fixed in the MP3 standard), stale persistence could cause a 400 error on the first submit.

**Prevention:**
On load from `localStorage`, validate the persisted bitrate against the known list `[32,40,48,56,64,80,96,112,128,160,192,224,256,320]` in TypeScript before using it, defaulting to `app.props.cli.mp3_bitrate` if invalid. This mirrors the existing format validation at `state.svelte.ts:22`.

**Phase:** WebUI implementation phase.

---

### Pitfall 10: `server.sh` `--mp3-bitrate` Flag Not Documented or Exposed

**What goes wrong:**
`server.sh` is the user-facing launch script. If the `--mp3-bitrate` flag is added to `ace-server.cpp` but not to `server.sh` (or the `server.sh` help comment), server operators cannot discover or configure the default without reading C++ source.

**Prevention:**
Add a commented-out `--mp3-bitrate 128` line to `server.sh` in the output section near where other output flags would appear. The existing help text in `ace-server.cpp:1233` already lists `--mp3-bitrate`; ensure `server.sh` mirrors this.

**Phase:** Documentation and startup-script phase.

---

### Pitfall 11: PR Branch Contains WebUI Build Artifacts That Were Not Regenerated

**What goes wrong:**
`PROJECT.md` specifies: "WebUI is pre-built and committed (`tools/public/index.html.gz`); must run `./buildwebui.sh` after any `.svelte` changes, then rebuild `ace-server` to re-embed." A PR that modifies `.svelte` files but commits a stale `index.html.gz` (from a previous build, or left over from a different branch) will pass compilation but ship the old UI.

**Why it happens:**
`git diff` does not reveal that the gzipped artifact is stale relative to the source. The stale artifact commits cleanly and the server embeds it without complaint.

**Prevention:**
Before opening the PR, run `./buildwebui.sh` and verify the `index.html.gz` modification timestamp is newer than the `.svelte` files. Include the regenerated artifact in the same commit as the `.svelte` changes so reviewers see the artifact change alongside the source change.

**Detection:**
After starting the rebuilt server, confirm in a browser that the bitrate selector is present. If it is missing, the artifact was not regenerated.

**Phase:** PR submission phase (final checklist item).

---

### Pitfall 12: PR Attribution Includes AI Co-Author Lines

**What goes wrong:**
`PROJECT.md` explicitly states: "No Claude/Anthropic co-author attribution in commit messages." Including `Co-Authored-By: Claude ...` lines in commits will require a rebase before the PR can be merged.

**Prevention:**
Do not add co-author trailers. Standard Git commit message format only.

**Phase:** All commit phases.

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| C++ `ServerFields` addition | Invalid bitrate from client silently encodes wrong (Pitfall 1) | Validate against `mp3enc_bitrate_kbps[]` in `parse_server_fields`; return 400 |
| C++ `synth_worker` encode path | Per-request bitrate not wired in; global used instead | Replace `g_mp3_kbps` with `(sf.mp3_bitrate > 0) ? sf.mp3_bitrate : g_mp3_kbps` |
| C++ startup flag | `--mp3-bitrate` accepts invalid values silently | Validate `g_mp3_kbps` after `atoi` against the allowlist |
| Multipart `/synth` path | Bitrate missing from multipart `"request"` part (Pitfall 6) | Ensure both `synthSubmit` and `synthSubmitWithAudio` carry the bitrate field |
| WebUI format/bitrate interaction | Bitrate selector active on WAV output (Pitfall 2) | Derive `bitrateEnabled` from `app.format === 'mp3'`; disable selector otherwise |
| WebUI initial state | Bitrate defaults to 128 ignoring server's `--mp3-bitrate` (Pitfall 3) | Seed from `app.props?.cli?.mp3_bitrate` on first load |
| WebUI state architecture | `mp3_bitrate` added to `AceRequest` instead of app-level state (Pitfall 5) | Keep as `app.mp3Bitrate`; inject at submit time, not stored in request object |
| WebUI build | Svelte sources changed but `index.html.gz` not regenerated (Pitfall 11) | Run `./buildwebui.sh` + rebuild C++ before committing artifact |
| PR submission | AI co-author line in commit message (Pitfall 12) | Do not add; rebase required to remove if present |

---

## Sources

- `mp3/mp3enc-tables.h`: `mp3enc_bitrate_kbps[]` — the canonical valid bitrate list
- `tools/ace-server.cpp:442–447`: `ServerFields` struct — where `mp3_bitrate` belongs
- `tools/ace-server.cpp:449–490`: `parse_server_fields` — where validation must go
- `tools/ace-server.cpp:843–848`: `synth_worker` encode path — where `g_mp3_kbps` is used
- `tools/ace-server.cpp:1173–1177`: `/props` handler — `cli.mp3_bitrate` already exported
- `tools/ace-server.cpp:1279–1281`: `--mp3-bitrate` startup flag — needs same validation
- `tools/webui/src/lib/state.svelte.ts:22`: format restore validation — pattern to follow for bitrate
- `tools/webui/src/lib/api.ts:43–66`: `synthSubmit` / `synthSubmitWithAudio` — both must carry bitrate
- `tools/webui/src/components/RequestForm.svelte:882–889`: format `<select>` — reference point for bitrate selector placement
- `.planning/PROJECT.md`: constraints, MP3 codec valid bitrates, PR hygiene rules
