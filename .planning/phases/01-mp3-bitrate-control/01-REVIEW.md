---
phase: 01-mp3-bitrate-control
reviewed: 2026-04-22T00:00:00Z
depth: standard
files_reviewed: 5
files_reviewed_list:
  - server.sh
  - tools/ace-server.cpp
  - tools/webui/src/components/RequestForm.svelte
  - tools/webui/src/lib/api.ts
  - tools/webui/src/lib/state.svelte.ts
findings:
  critical: 0
  warning: 4
  info: 3
  total: 7
status: issues_found
---

# Phase 01: Code Review Report

**Reviewed:** 2026-04-22
**Depth:** standard
**Files Reviewed:** 5
**Status:** issues_found

## Summary

This review covers the `mp3_bitrate` feature implementation across all layers: the bash launch script, the C++ HTTP server, and the Svelte/TypeScript web UI. The feature is structurally sound — the server correctly validates bitrates at both the CLI level and the per-request level, the API layer correctly injects `mp3_bitrate` into the JSON body, and the UI correctly seeds from the server default. No security vulnerabilities or data-loss risks were found.

Four warnings are raised: a drift between the valid-bitrate set the UI offers and the set the server accepts; a misleading fallback label that fires for the server default rather than only for non-preset values; a silent early-return in `parse_server_fields` that discards all other parsed fields (synth_model, lm_model, lora, lora_scale) before returning the -1 sentinel, which could hide a real user misconfiguration; and a `server.sh` syntax issue where the commented-out `--mp3-bitrate` line is attached to the last real argument with a backslash continuation, making the comment fragile.

Three info items are raised: a duplicate valid-bitrates array in C++ (declared in two places), a missing `fetch` timeout on `jobResultJson` and `jobResultBlobs`, and an unused `FETCH_TIMEOUT_MS` import path in those two functions.

---

## Warnings

### WR-01: UI bitrate preset list is a subset of the server's valid set

**File:** `tools/webui/src/components/RequestForm.svelte:896-907`

**Issue:** The `<select>` for MP3 bitrate offers only six presets: `96, 128, 160, 192, 256, 320`. The server accepts fourteen: `32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320`. The UI guard in the fallback `<option>` (line 898, `![96, 128, 160, 192, 256, 320].includes(app.mp3Bitrate)`) is also hard-coded to only those six values, so if a user starts the server with `--mp3-bitrate 112` the UI will show a "server default" option for 112 but not present it in the main list when the user manually changes the value away and back. More importantly, the `validBitrates` list in `state.svelte.ts` line 20 (`[96, 128, 160, 192, 256, 320]`) is used to validate the persisted value: if the server default is any bitrate not in those six (e.g., 40 or 224), the stored value is treated as invalid and reset to 0, causing the UI to always re-seed from the server default on reload even if the user had changed it.

**Fix:** Align all three hard-coded lists. The simplest approach is a single shared constant in `config.ts`:

```ts
// tools/webui/src/lib/config.ts  (add)
export const VALID_MP3_BITRATES = [32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320];
```

Then in `state.svelte.ts`:
```ts
import { VALID_MP3_BITRATES } from './config.js';
// ...
mp3Bitrate: VALID_MP3_BITRATES.includes(parsed.mp3Bitrate) ? parsed.mp3Bitrate : 0,
```

And in `RequestForm.svelte`, render the `<option>` list from `VALID_MP3_BITRATES` rather than a static inline list.

---

### WR-02: Fallback `<option>` label fires for the server default, not only for unknown values

**File:** `tools/webui/src/components/RequestForm.svelte:898-900`

**Issue:**
```svelte
{#if ![96, 128, 160, 192, 256, 320].includes(app.mp3Bitrate)}
    <option value={app.mp3Bitrate}>{app.mp3Bitrate} kbps (server default)</option>
{/if}
```
The label "(server default)" is shown whenever `app.mp3Bitrate` is not in the preset list, which happens any time the server is configured with any bitrate outside those six. However the user may have already changed the bitrate away from the server default in a previous session — the stored value is then valid (e.g. 112) but the option is still labelled "server default", which is misleading.

**Fix:** Track the server default separately from the user-selected value:

```ts
// In state.svelte.ts, add a dedicated field:
serverMp3Bitrate: 0 as number,
```

Then in `RequestForm.svelte`, seed effect:
```ts
$effect(() => {
    if (app.props && app.serverMp3Bitrate === 0) {
        app.serverMp3Bitrate = Number(app.props.cli.mp3_bitrate) || 128;
        if (app.mp3Bitrate === 0) app.mp3Bitrate = app.serverMp3Bitrate;
    }
});
```

Use `app.serverMp3Bitrate` as the condition for the fallback option label so it only appears when the server default is genuinely outside the preset list, and is labelled "server default" only on first load before the user changes it.

---

### WR-03: `parse_server_fields` early-return on invalid bitrate silently discards all other parsed fields

**File:** `tools/ace-server.cpp:490-503`

**Issue:** When `mp3_bitrate` is present but invalid, the function sets `sf->mp3_bitrate = -1` and returns immediately:

```cpp
if (!ok) {
    yyjson_doc_free(doc);
    sf->mp3_bitrate = -1;   // sentinel: caller must return 400
    return;
}
```

Any fields that appear *after* `mp3_bitrate` in the JSON object are never parsed (though yyjson object iteration is insertion-order-dependent). More importantly, the caller already validates the sentinel and will return 400, so those fields are effectively lost on an error path — which is acceptable. However, `sf->lora`, `sf->lora_scale`, `sf->synth_model`, and `sf->lm_model` may have already been parsed into `sf` before the check but their values are irrelevant since the request is rejected. The actual bug risk is subtle: if JSON field order happens to put `mp3_bitrate` before `lm_model` in the object and the bitrate is invalid, a misleading "mp3_bitrate must be one of..." error is returned even when `lm_model` is also absent/invalid. This is a diagnostics/UX issue, not a crash, but it can confuse debugging.

**Fix:** Perform the sentinel check only after parsing all fields, so error messages are not order-dependent:

```cpp
// After all yyjson_obj_get checks:
if (mp3_bitrate_seen && !ok) {
    yyjson_doc_free(doc);
    sf->mp3_bitrate = -1;
    return;
}
sf->mp3_bitrate = req_kbps;
```

Or simply move the bitrate validation to a second pass after `yyjson_doc_free`. Either way eliminates the order dependency.

---

### WR-04: `server.sh` comment is attached to the last real flag via backslash continuation

**File:** `server.sh:14-15`

**Issue:**
```bash
./build/ace-server \
    --host 0.0.0.0 \
    --port 8085 \
    --models ./models \
    --loras ./loras \
    --max-batch 1
    #--mp3-bitrate 128    # MP3 encoding bitrate ...
```

The `--max-batch 1` line has no trailing `\`, and the next line is a comment. This is harmless as written today. However if someone copies the block and adds a new flag between `--max-batch 1` and the comment line they may accidentally omit the backslash on `--max-batch 1`, or accidentally uncomment the `--mp3-bitrate` line without adding a `\` on the preceding line and get a parse error. The comment line starting with `#` is not a flag continuation issue at the shell level, but the structural inconsistency (the block looks like the comment is an "optional flag" continuation) makes it easy to make a mistake.

**Fix:** Add a blank line between the last argument and the commented example, or move the commented example above the invocation:

```bash
./build/ace-server \
    --host 0.0.0.0 \
    --port 8085 \
    --models ./models \
    --loras ./loras \
    --max-batch 1
    #--mp3-bitrate 128   # optional: override MP3 bitrate (32,40,48,...,320)
```

No code change needed for correctness, but the comment placement directly following the last unescaped line without a blank line is a readability trap.

---

## Info

### IN-01: Valid-bitrates array duplicated in C++

**File:** `tools/ace-server.cpp:491-496` and `tools/ace-server.cpp:1325-1330`

**Issue:** The array `valid_bitrates[] = {32,40,48,56,64,80,96,112,128,160,192,224,256,320}` is declared twice as a `static const` local — once in `parse_server_fields` and once in the `--mp3-bitrate` CLI argument block in `main`. If a valid bitrate is added or removed in one place, it must be remembered to update the other.

**Fix:** Hoist to a file-scope constant:

```cpp
static const int k_valid_bitrates[] = {32,40,48,56,64,80,96,112,128,160,192,224,256,320};
static const int k_n_valid_bitrates = (int)(sizeof(k_valid_bitrates)/sizeof(k_valid_bitrates[0]));
```

Then replace both local declarations with references to the file-scope array.

---

### IN-02: `jobResultJson` and `jobResultBlobs` lack a fetch timeout

**File:** `tools/webui/src/lib/api.ts:116-134`

**Issue:** `jobStatus` (line 84) correctly applies `AbortSignal.timeout(FETCH_TIMEOUT_MS)`, but `jobResultJson` (line 117) and `jobResultBlobs` (line 124) do not. For large synth outputs the result fetch could hang indefinitely on a network stall. The current pattern relies on the browser's own TCP timeout.

**Fix:**
```ts
export async function jobResultJson(id: string): Promise<AceRequest[]> {
    const res = await fetch(`job?id=${encodeURIComponent(id)}&result=1`, {
        signal: AbortSignal.timeout(FETCH_TIMEOUT_MS)
    });
    // ...
}
```

Apply the same to `jobResultBlobs`. Note that `FETCH_TIMEOUT_MS` may need to be increased for very large audio results if it is currently set to a low value (e.g. 2000 ms); verify the constant is appropriate before applying.

---

### IN-03: `mp3_bitrate` is not forwarded through the LM call path (by design, but not documented)

**File:** `tools/webui/src/lib/api.ts:16-39`  
**File:** `tools/ace-server.cpp:717-722`

**Issue:** `lmSubmit`, `lmSubmitInspire`, and `lmSubmitFormat` never inject `mp3_bitrate` into the request body. The server-side `handle_lm` parses `ServerFields` (including `mp3_bitrate`) from the LM body and validates it — meaning an invalid `mp3_bitrate` sent to `/lm` would return 400 — but the LM pipeline never actually encodes audio, so the bitrate is irrelevant there. The validation on the LM path is dead logic. This is not a bug (the server safely ignores it), but the mismatch between "we validate it but never use it" and "the client never sends it" is a future maintenance hazard.

**Fix (documentation only):** Add a comment at the `parse_server_fields` call in `handle_lm` noting that `mp3_bitrate` is parsed for uniformity but unused by the LM pipeline:

```cpp
// mp3_bitrate is irrelevant for the LM endpoint (no audio encoding);
// parsed here so parse_server_fields can be called uniformly.
parse_server_fields(req.body.c_str(), &sf);
if (sf.mp3_bitrate == -1) { ... }
```

---

_Reviewed: 2026-04-22_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
