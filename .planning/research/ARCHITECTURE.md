# Architecture: Per-Request MP3 Bitrate Control

**Project:** acestep.cpp — MP3 bitrate milestone
**Researched:** 2026-04-18
**Confidence:** HIGH — all findings from direct codebase inspection

---

## Integration Summary

The change is small and self-contained. The established `ServerFields` pattern absorbs it cleanly:
add one `int` field, parse it in one function, thread it through the existing call chain, substitute
it for `g_mp3_kbps` at the single encoder call site, and expose it as a WebUI control that seeds from
`GET /props`. No new structs, no new endpoints, no new files — only targeted additions to five
existing files.

---

## Exact Data Flow

```
WebUI (RequestForm.svelte)
  app.mp3_bitrate (new reactive state, initialised from props.cli.mp3_bitrate on mount)
  → serialised as top-level JSON key "mp3_bitrate" in the request body
  → both paths (plain JSON body, multipart "request" part)

POST /synth  (tools/ace-server.cpp)
  handle_synth()  [line 891]
    parse_server_fields(body, &sf)   [line 920 multipart, line 963 JSON]
      reads "mp3_bitrate" int from JSON → sf.mp3_bitrate (new field)
    work_push(...sf...) captures sf by value  [line 995-998]

  synth_worker()  [line 732]
    receives sf
    at line 846:
      audio_encode_mp3(..., g_mp3_kbps, ...)   ← TODAY
      audio_encode_mp3(..., sf.mp3_bitrate, ...) ← AFTER CHANGE
```

The encoder itself (`audio_encode_mp3` in `src/audio-io.h:609`) already accepts `int kbps` as its
fourth parameter and passes it straight to `mp3enc_init`. No changes needed inside the encoder.

---

## Files to Change

### 1. `tools/ace-server.cpp` — Backend (3 independent edits)

**Edit A — `ServerFields` struct (line 442)**

Add one field:

```cpp
struct ServerFields {
    std::string synth_model;
    std::string lm_model;
    std::string lora;
    float       lora_scale;
    int         mp3_bitrate;   // 0 = use server default (g_mp3_kbps)
};
```

**Edit B — `parse_server_fields()` (line 449)**

Initialise the new field to 0 in the reset block (line 452 area) and add a parse clause:

```cpp
sf->mp3_bitrate = 0;
// ... existing fields ...
if ((v = yyjson_obj_get(obj, "mp3_bitrate")) && yyjson_is_int(v)) {
    sf->mp3_bitrate = (int) yyjson_get_int(v);
}
```

**Edit C — `synth_worker()` encoding call (line 846)**

Replace the hardcoded global with the per-request value, falling back to the global when the
request did not specify one:

```cpp
int kbps = (sf.mp3_bitrate > 0) ? sf.mp3_bitrate : g_mp3_kbps;
encoded[b] = audio_encode_mp3(audio[b].samples, audio[b].n_samples, 48000, kbps,
                               server_cancel_job, (void *) &job->cancel);
```

No validation/clamping needed in C++: `mp3enc_init` takes any int and the lowpass table uses
nearest-neighbour selection (line 152-161 of `mp3/mp3enc.h`), so an out-of-range value degrades
gracefully to the nearest valid rate. Validation belongs in the WebUI (see below).

**No changes required** in `handle_synth()`, `work_push`, or the `synth_worker` function
signature — `ServerFields` is already passed and captured by value throughout.

---

### 2. `tools/webui/src/lib/types.ts` — TypeScript type (1 edit)

Add `mp3_bitrate` as a server-routing field alongside `synth_model`, `lora_scale`, etc. The
comment block at line 33 already marks these fields as "server routing (not part of C++ AceRequest,
parsed separately)":

```typescript
// server routing (not part of C++ AceRequest, parsed separately)
synth_model?: string;
lm_model?: string;
lora?: string;
lora_scale?: number;
mp3_bitrate?: number;   // ← add here
```

---

### 3. `tools/webui/src/lib/state.svelte.ts` — App state (1 edit)

`app.format` (the output format string "mp3"/"wav16"/etc.) is a top-level field on the `app`
object, not part of `app.request`. `mp3_bitrate` should follow the same pattern since it is also
not a generation parameter — it is an output encoding setting.

Add `mp3Bitrate` to the `app` state object and the `Saved` interface, next to `format`:

```typescript
interface Saved {
    // ...
    format: string;
    mp3Bitrate: number;   // ← add
    // ...
}

export const app = $state({
    // ...
    format: saved.format,
    mp3Bitrate: saved.mp3Bitrate,   // ← add
    // ...
});
```

Default in `load()`: `mp3Bitrate: 128` (or leave 0 meaning "not set yet, will seed from props").

Seeding from props: in `App.svelte` (or `RequestForm.svelte`'s `onMount`), after `GET /props`
resolves, set `app.mp3Bitrate = app.props.cli.mp3_bitrate` if `app.mp3Bitrate` is still 0. This
mirrors how `app.format` is already preserved from `localStorage` but gets a sensible default from
the server.

The `Saved` persistence block (line 94-107) already serialises everything in `app` — adding the
field there keeps it in `localStorage` across reloads.

---

### 4. `tools/webui/src/components/RequestForm.svelte` — UI control (2 edits)

**Edit A — Seed `app.mp3Bitrate` from props on mount**

Inside the existing `$effect` that reads `app.props?.default` (around line 59), or in a separate
`$effect`, initialise bitrate from props when props arrive and no user value is stored:

```typescript
$effect(() => {
    if (app.props && !app.mp3Bitrate) {
        app.mp3Bitrate = Number(app.props.cli.mp3_bitrate) || 128;
    }
});
```

**Edit B — Add bitrate selector to the format row (around line 882)**

The existing format row (line 862-891) contains Batch, Peak clip, and the format `<select>`. Insert
a bitrate control between Peak clip and the format selector, visible only when `app.format ===
'mp3'`:

```svelte
{#if app.format === 'mp3'}
<select
    bind:value={app.mp3Bitrate}
    title="MP3 encoding bitrate (kbps). Server default shown; changes apply to this request only."
>
    {#each [32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320] as kbps}
        <option value={kbps}>{kbps}</option>
    {/each}
</select>
{/if}
```

A `<select>` with the fixed 14 valid MPEG1 bitrates avoids free-text validation entirely. The list
matches the lowpass table in `mp3/mp3enc.h` lines 131-147.

**Edit C — Include `mp3_bitrate` in JSON body sent to `/synth`**

In the `synthesize()` function (around lines 393-407), when building `toSend`, include the bitrate
when the format is MP3:

```typescript
const toSend: AceRequest[] = reqs.map(r => ({
    ...r,
    ...(app.format === 'mp3' && app.mp3Bitrate ? { mp3_bitrate: app.mp3Bitrate } : {})
}));
```

Both `synthSubmit` and `synthSubmitWithAudio` take `reqs: AceRequest[]` and serialise to JSON —
no signature changes needed. The field travels as a top-level JSON key alongside the generation
params, exactly as `synth_model` and `lora` do today.

---

## Multipart Form-Data Path

The multipart path has **no special concerns** for this feature.

How it works today (lines 907-960):
1. `req.is_multipart_form_data()` → `true`
2. The JSON part is extracted from either a `file` part named `"request"` or a `field` part named
   `"request"` (both branches checked at lines 912-919)
3. `parse_server_fields(json_body.c_str(), &sf)` is called on that same string (line 920)
4. Audio parts `"audio"` and `"ref_audio"` are extracted separately

`mp3_bitrate` lives in the JSON blob in the `"request"` part — identical to how `synth_model` and
`lora` are carried. `parse_server_fields` already parses that JSON blob in the multipart path.
Adding the new field to `parse_server_fields` handles both paths automatically. The WebUI's
`synthSubmitWithAudio` serialises `reqs[0]` to the `"request"` blob (line 61-63 of `api.ts`),
so as long as `mp3_bitrate` is in the request object it reaches the server.

The only subtlety: `synthSubmitWithAudio` currently passes the `format` string only as a query
param (`?format=wav16`), not in the JSON. That pattern is pre-existing and unrelated — MP3 bitrate
is separate from the WAV/MP3 container selection and rides in the JSON body.

---

## Component Boundaries After Change

```
WebUI state layer (state.svelte.ts)
  app.mp3Bitrate  ← new top-level field, seeds from GET /props cli.mp3_bitrate

WebUI UI layer (RequestForm.svelte)
  <select bind:value={app.mp3Bitrate}>  ← new control, visible only when format=mp3
  serialises to mp3_bitrate in AceRequest JSON body

Server HTTP layer (tools/ace-server.cpp: handle_synth)
  parse_server_fields()  ← reads mp3_bitrate int from JSON
  passes sf by value into work_push closure

Server worker layer (tools/ace-server.cpp: synth_worker)
  kbps = sf.mp3_bitrate > 0 ? sf.mp3_bitrate : g_mp3_kbps
  audio_encode_mp3(..., kbps, ...)  ← was g_mp3_kbps

Encoder (src/audio-io.h: audio_encode_mp3)
  unchanged — already takes int kbps
```

---

## Build Order

1. Edit C++ (`tools/ace-server.cpp`) — three hunks, no header changes needed
2. Edit TypeScript types (`tools/webui/src/lib/types.ts`)
3. Edit state (`tools/webui/src/lib/state.svelte.ts`)
4. Edit UI (`tools/webui/src/components/RequestForm.svelte`)
5. Run `./buildwebui.sh` — recompiles Svelte, regenerates `tools/public/index.html.gz`
6. Rebuild `ace-server` — re-embeds the updated gzip'd HTML

Steps 1-4 are independent of each other and can be done in any order, but step 5 must precede
step 6 (the C++ binary embeds the gzip'd WebUI output at compile time via an xxd-style include or
`ld -b binary`).

---

## No New Files Required

| What | Verdict |
|------|---------|
| New C++ header | No — `ServerFields` change is two lines in `ace-server.cpp` |
| New TypeScript module | No — one field added to existing `types.ts` |
| New Svelte component | No — inline `<select>` in `RequestForm.svelte` |
| New endpoint | No — `GET /props` already exposes `cli.mp3_bitrate` |
| New npm dependency | No |

---

## Edge Cases and Constraints

**Valid bitrate set (from `mp3/mp3enc.h` lowpass table, lines 131-147):**
32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320 kbps.
The encoder does not reject other values — it picks the nearest lowpass cutoff. Using a `<select>`
in the WebUI with this exact list prevents any invalid value from being submitted.

**WAV format selected:** When `app.format !== 'mp3'`, the bitrate field should be omitted from the
JSON (or set to 0) and the control hidden. The `synth_worker` only reaches the MP3 branch when
`output_wav` is false (line 845 of `ace-server.cpp`). If `mp3_bitrate` is present in the JSON but
the request ends up encoding WAV, `sf.mp3_bitrate` is simply unused — no error.

**`mp3_bitrate: 0` or absent:** `parse_server_fields` initialises `sf.mp3_bitrate = 0`. The
fallback `kbps = sf.mp3_bitrate > 0 ? sf.mp3_bitrate : g_mp3_kbps` collapses to `g_mp3_kbps`
(the server-configured default, 128 unless overridden by `--mp3-bitrate`). This preserves backward
compatibility for any client that does not send the field.

**Batch requests (array JSON body):** `parse_server_fields` already handles array vs. object at
lines 466-473 — it takes server fields from the first element. A per-request bitrate applies to
all tracks in the batch, which is the correct behaviour (all tracks in one `/synth` call share the
same output encoding).

**`server.sh --mp3-bitrate` flag:** Already implemented (lines 1233, 1280-1281 of
`ace-server.cpp`). The milestone requirement is already met on the server side for the startup
flag. The WebUI seeding from `GET /props` `cli.mp3_bitrate` is the only remaining connection.

---

## Confidence

All findings derived from direct inspection of:
- `tools/ace-server.cpp` (lines 194, 442-490, 732-876, 891-1004, 1139-1177, 1280-1281)
- `src/audio-io.h` (lines 609-794)
- `mp3/mp3enc.h` (lines 89-187)
- `tools/webui/src/lib/api.ts`
- `tools/webui/src/lib/state.svelte.ts`
- `tools/webui/src/lib/types.ts`
- `tools/webui/src/components/RequestForm.svelte`

No inference or training-data assumptions were required.
