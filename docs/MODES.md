# ACE-Step 1.5 built-in modes (acestep.cpp)

This document maps the [ACE-Step 1.5 Tutorial](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/Tutorial.md) built-in modes to the current C++ implementation.

## Task types (Tutorial: Input Control)

| `task_type`   | Description | Turbo/SFT | Base only | C++ status |
|---------------|-------------|-----------|-----------|------------|
| **text2music** | Generate from caption/lyrics (and optional reference) | ✅ | — | ✅ **Supported** |
| **cover**      | Re-synthesize with structure from source; optional timbre from reference | ✅ | — | ⚠️ **Partial** (see below) |
| **repaint**    | Local edit in time range using source as context | ✅ | — | ❌ Not implemented |
| **lego**       | Add new tracks to existing audio | — | ✅ | ❌ Base model only |
| **extract**    | Extract single track from mix | — | ✅ | ❌ Base model only |
| **complete**   | Add accompaniment to single track | — | ✅ | ❌ Base model only |

We only ship Turbo and SFT DiT weights; **lego**, **extract**, **complete** require the Base DiT and are out of scope for now.

---

## What we support today

### text2music (default)
- **Input**: `caption`, optional `lyrics`, metadata (bpm, duration, keyscale, …).
- **Flow**: LM (optional) → CoT + audio codes → DiT (context = silence) → VAE → WAV.
- **Timbre**: Always uses built-in silence latent from the DiT GGUF (no user reference yet).

### cover (when `audio_codes` are provided)
- **Input**: Same as text2music, plus **precomputed** `audio_codes` (e.g. from a previous run or from Python).
- **Flow**: Skip LM; decode `audio_codes` to latents → DiT context = decoded + silence padding → DiT → VAE → WAV.
- **Limitation**: We do **not** convert a WAV file into `audio_codes`. So “cover from a file” is only possible if you already have codes (e.g. from Python or from a prior `ace-qwen3` run). The request fields `reference_audio` and `src_audio` are accepted in JSON but **not yet used** in the pipeline.

---

## What’s not implemented yet

### reference_audio (global timbre/style)
- **Tutorial**: Load WAV → stereo 48 kHz, pad/repeat to ≥30 s → **VAE encode** → latents → feed as timbre condition into DiT.
- **C++**: Implemented. Set `reference_audio` to a **WAV or MP3 file path**. dit-vae loads the file (WAV: any sample rate resampled to 48 kHz; MP3: decoded in memory via header-only minimp3, no temp files, then resampled to 48 kHz if needed), runs the **VAE encoder** (Oobleck, in C++ in `vae.h`), and feeds the 64-d latents to the CondEncoder timbre path. No Python, no external deps. Requires a **full VAE GGUF** that includes `encoder.*` tensors (decoder-only GGUFs will print a clear error).
- **audio_cover_strength** (0.0–1.0): Implemented. When `audio_codes` are present, context latents are blended with silence: `(1 - strength)*silence + strength*decoded`.

### src_audio (Cover from file)
- **Tutorial**: Source audio is converted to **semantic codes** (melody, rhythm, chords, etc.); then DiT uses those as in cover mode.
- **C++**: That implies **audio → codes**. Likely path: WAV → VAE encode → **FSQ tokenizer** (latents → 5 Hz codes). We have the **FSQ detokenizer** (codes → latents); the tokenizer (encode) side would need to be added. Then: `src_audio` path → load WAV → VAE encode → FSQ encode → `audio_codes` → existing cover path.

### audio_cover_strength
- **Tutorial**: 0.0–1.0, how strongly generation follows reference/codes.
- **C++**: Field is in the request and parsed; no blending logic in the DiT/context path yet.

### repaint
- **Tutorial**: Specify `repainting_start` / `repainting_end` (seconds); model uses source audio as context and only generates in that interval (3–90 s).
- **C++**: Would require **masked diffusion**: context carries “given” frames; ODE only updates the repaint region. DiT’s context has a 64-channel “mask” that we currently set to 1.0; repaint would set mask per frame and the generation loop would only update unmasked frames. Not implemented.

---

## Request fields (aligned with Tutorial)

All of these are in `AceRequest` and parsed from / written to JSON. Backend behavior is as above.

| Field | Type | Purpose |
|-------|------|--------|
| `task_type` | string | `"text2music"` \| `"cover"` \| `"repaint"` \| … |
| `reference_audio` | string | Path to WAV or MP3 for timbre (implemented) |
| `src_audio` | string | Path to WAV for cover/repaint source (not used yet) |
| `audio_codes` | string | Comma-separated FSQ codes; non-empty ⇒ cover path |
| `audio_cover_strength` | float | 0.0–1.0 (parsed, not used yet) |
| `repainting_start` | float | Start time (s) for repaint (not used yet) |
| `repainting_end` | float | End time (s) for repaint (not used yet) |

See `request.h` and the README “Request JSON reference” for the full list.

---

## Summary

- **Fully supported**: text2music; cover when you supply **precomputed** `audio_codes`.
- **Schema only** (no backend): `task_type`, `reference_audio`, `src_audio`, `audio_cover_strength`, `repainting_start`/`repainting_end`.
- **To support reference_audio**: add VAE encoder, then feed its output into the existing CondEncoder timbre path.
- **To support cover from file**: add VAE encoder + FSQ tokenizer (or equivalent audio→codes), then reuse existing cover path.
- **To support repaint**: implement masked DiT generation (context mask + ODE only on repaint interval).
