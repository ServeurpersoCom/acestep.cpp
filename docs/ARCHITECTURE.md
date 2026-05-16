# Architecture

> Full technical reference for acestep.cpp. For a quick start guide, see [README.md](../README.md).

# acestep.cpp

Portable C++17 implementation of ACE-Step 1.5 music generation using GGML.
Text + lyrics in, stereo 48kHz MP3 or WAV out. Runs on CPU, CUDA, ROCm, Metal, Vulkan.

## Build

```bash
git submodule update --init

mkdir build && cd build

# macOS (Metal + Accelerate BLAS auto-enabled)
cmake ..

# Linux with NVIDIA GPU
cmake .. -DGGML_CUDA=ON

# Linux with AMD GPU (ROCm)
cmake .. -DGGML_HIP=ON

# Linux with Vulkan
cmake .. -DGGML_VULKAN=ON

cmake --build . --config Release -j$(nproc)
```

### Windows

Install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
(select "Desktop development with C++" workload) and optionally the
[CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and/or the
[Vulkan SDK](https://vulkan.lunarg.com/sdk/home).

```cmd
git submodule update --init

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

mkdir build
cd build

rem NVIDIA GPU
cmake .. -DGGML_CUDA=ON

rem AMD/Intel GPU (Vulkan)
cmake .. -DGGML_VULKAN=ON

rem all backends (CUDA + Vulkan + CPU, runtime loading)
cmake .. -DGGML_CPU_ALL_VARIANTS=ON -DGGML_CUDA=ON -DGGML_VULKAN=ON -DGGML_BACKEND_DL=ON

cmake --build . --config Release -j %NUMBER_OF_PROCESSORS%
```

Builds seven binaries: `ace-lm` (LLM), `ace-synth` (DiT + VAE), `ace-server` (HTTP server), `ace-understand` (reverse: audio -> metadata), `neural-codec` (VAE encode/decode), `mp3-codec` (MP3 encoder/decoder) and `quantize` (GGUF requantizer).

## Models

Pre-quantized GGUFs on [Hugging Face](https://huggingface.co/Serveurperso/ACE-Step-1.5-GGUF).

```bash
pip install hf
./models.sh              # Q8_0 turbo essentials (~7.7 GB)
./models.sh --all        # every model, every quant (~97 GB)
./models.sh --quant Q6_K # pick a specific quant (Q4_K_M, Q5_K_M, Q6_K, Q8_0, BF16)
./models.sh --sft        # add SFT DiT variant
./models.sh --shifts     # add shift1/shift3/continuous variants
```

Default downloads 4 files into `models/`:

| GGUF | Arch | Size |
|------|------|------|
| Qwen3-Embedding-0.6B-Q8_0.gguf | text encoder (28L, H=1024) | 748 MB |
| acestep-5Hz-lm-4B-Q8_0.gguf | Qwen3 causal LM | 4.2 GB |
| acestep-v15-turbo-Q8_0.gguf | DiT 2B + CondEncoder (24L, H=2048) | 2.4 GB |
| vae-BF16.gguf | AutoencoderOobleck | 322 MB |

Three LM sizes: 0.6B (fast), 1.7B, 4B (best quality).
Six DiT variants: turbo, sft, base, turbo-shift1, turbo-shift3, turbo-continuous.
XL (4B DiT) variants: xl-turbo, xl-sft, xl-base (32L, H=2560, higher quality, ~9.5 GB BF16).
VAE is always BF16 (small, bandwidth-bound, quality-critical).

<details>
<summary>Building GGUFs from source (checkpoints + convert)</summary>

If you want to convert from the original safetensors yourself:

```bash
pip install gguf hf
./checkpoints.sh       # download raw HF checkpoints (turbo + 4B LM)
./checkpoints.sh --all # all variants (SFT, shift1/3, 0.6B/1.7B LM)
python3 convert.py     # convert all checkpoints to GGUF (models/)
./quantize.sh          # quantize BF16 -> Q4_K_M/Q5_K_M/Q6_K/Q8_0
```

`checkpoints.sh` downloads safetensors, config.json, and tokenizer files
into `checkpoints/`. `convert.py` packs everything into self-contained
GGUF files in `models/`, bundling BPE tokenizer, silence_latent, and
config metadata so no external file is needed at runtime.

</details>

## VRAM policy

Two modes, one flag, no in-between.

**Default mode** optimises VRAM. At most one GPU module is resident at a
time. The store evicts whatever was loaded before bringing in the next
module, so VAE tile activations never sit next to DiT weights or LM
weights. This is the mode that lets the full ACE-Step stack run on
consumer cards: the DiT loads for denoising, evicts when done, the VAE
loads for decode, evicts, and so on.

**`--keep-loaded` mode** optimises latency. Everything stays resident
across requests. No reload, no eviction. This is the mode for workstations
with generous VRAM where startup overhead would dominate the request
latency. No "smart" rules kick in: if the user asks for keep-loaded, they
get exactly that.

**Invariant held under both modes.** Exactly one LM instance lives in the
process, shared between `ace-lm` (generate) and `ace-understand`.
Duplicating the Qwen3 LM would cost gigabytes and buy nothing. The
ModelStore enforces this by keying the LM on `(path, max_seq, n_kv_sets)`
with identical values across both pipelines.

## ModelStore

A single `ModelStore` instance owns every GPU module the server or the
CLIs touch: Qwen3 LM, DiT (with optional adapter), VAE encoder, VAE
decoder, Qwen3 text encoder, condition encoder, FSQ tokenizer, FSQ
detokenizer. Pipelines never load or free modules themselves. They ask
the store for a module with `store_require_*`, use it, and release it
when the scope ends. A thin RAII handle (`ModelHandle`) pairs the require
with the release so no early return, error path or exception can leak a
resident module.

The store keys each module by the fields that actually change what gets
loaded. For an LM that means `(path, max_seq, n_kv_sets)`. For a DiT that
means `(path, adapter_path, adapter_scale)`. For every other module it is
just `(path)`. Two requires with the same key return the same pointer, so
pipelines that share a module naturally share the resident weights.

CPU-only helpers (BPE merges, FSM decoding template, DiT metadata like
silence_latent and null_condition_emb) have their own accessors and stay
resident for the whole process lifetime. They total a few megabytes and
cost nothing to keep.

The chosen eviction policy (STRICT or NEVER, see above) decides what the
store does on release. STRICT unloads the module immediately once no
pipeline holds a handle on it, so VAE tiles never compete for VRAM with
an LM or a DiT. NEVER keeps everything loaded for maximum throughput on
machines that have the budget.

## CLI

`ace-lm` generates lyrics and audio codes, `ace-synth` synthesizes audio.
The input JSON is never modified. Output is always numbered: `request0.json`.

```bash
cat > /tmp/request.json << 'EOF'
{
    "caption": "Upbeat pop rock with driving guitars and catchy hooks",
    "inference_steps": 8,
    "shift": 3.0,
    "vocal_language": "fr"
}
EOF

# LLM: request.json -> request0.json (enriched with metadata + lyrics + codes)
./ace-lm \
    --models models \
    --request /tmp/request.json

# DiT+VAE: request0.json -> request00.mp3
./ace-synth \
    --models models \
    --request /tmp/request0.json
```

With an adapter (LoRA today, PEFT directory or ComfyUI single file), set
`adapter` in the JSON and point `--adapters` at a directory that contains it:

```bash
# JSON carries the adapter name, CLI passes the directory
cat > /tmp/request.json << 'EOF'
{
    "caption": "Upbeat pop rock with driving guitars and catchy hooks",
    "adapter": "best_sft_v2_2338_comfyui.safetensors",
    "adapter_scale": 1.0,
    "vocal_language": "fr"
}
EOF

./ace-synth \
    --models models \
    --adapters adapters \
    --request /tmp/request0.json
```

Generate multiple songs at once with `lm_batch_size` in the JSON:

```bash
# 2 different songs from one prompt (different lyrics, codes, metadata)
cat > /tmp/request.json << 'EOF'
{
    "caption": "Upbeat pop rock anthem with driving guitars and catchy hooks",
    "vocal_language": "fr",
    "lm_batch_size": 2
}
EOF

# LM: request.json (lm_batch_size=2) -> request0.json, request1.json
./ace-lm \
    --models models \
    --request /tmp/request.json

# DiT+VAE: both requests in one GPU batch -> request00.mp3, request10.mp3
./ace-synth \
    --models models \
    --request /tmp/request0.json /tmp/request1.json
```

`lm_batch_size` controls how many songs the LM generates. User-provided
fields are preserved in all outputs. Empty fields are filled independently
per batch item, producing genuinely different songs.
ace-synth takes all request files as CLI arguments and runs them in a
single GPU batch.

Transform an existing song with `--src-audio` (no LLM needed):

```bash
cat > /tmp/cover.json << 'EOF'
{
    "task_type": "cover",
    "caption": "Jazz piano cover with brushed drums and walking bass",
    "lyrics": "[Instrumental]"
}
EOF

./ace-synth \
    --models models \
    --src-audio song.wav \
    --request /tmp/cover.json
```

Ready-made examples in `examples/`:

```bash
cd examples
./simple.sh                    # caption only, LLM fills everything
./simple-batch.sh              # 2 songs from one prompt (lm_batch_size=2)
./partial.sh                   # caption + lyrics + duration
./full.sh                      # all metadata provided
./dit-only.sh                  # skip LLM, DiT from noise
./ace-understand.sh <audio>    # audio : understand : SFT DiT : MP3 roundtrip
./server-turbo.sh              # start HTTP server (turbo model)
./server-sft.sh                # start HTTP server (SFT model)
./client.sh                    # test server (single song)
./client-batch.py              # test server batch (2 songs)
./client-understand.sh <audio> # test /understand + /synth roundtrip
```

Each example has a `-sft` variant (SFT model, 50 steps, CFG 1.0)
alongside the turbo default (8 steps, no CFG).

## Generation modes

The LLM fills what's missing in the JSON and generates audio codes.
Empty field = "fill it". Filled = "don't touch".
All modes always output numbered files (`request0.json` .. `requestN-1.json`).
The input JSON is never modified.

**Caption only** (`lyrics=""`): two LLM passes. Phase 1 uses the "Expand"
prompt to generate an enriched caption, lyrics, and metadata (bpm, keyscale,
timesignature, duration, vocal_language) via CoT. Phase 2 reinjects the CoT
and generates audio codes using the "Generate tokens" prompt. CFG is forced
to 1.0 in phase 1 (free sampling); `lm_cfg_scale` only applies in phase 2.
With `lm_batch_size > 1`, each element runs its own phase 1,
producing N completely different songs. See `examples/simple-batch.json`.

**Caption + lyrics (+ optional metadata)**: single LLM pass. The "Generate
tokens" prompt is used directly. Missing metadata is filled via CoT, the
caption is enriched, and audio codes are generated. User-provided metadata
fields are never overwritten. `lm_cfg_scale` applies to both CoT and code
generation. See `examples/partial.json`.

**Everything provided** (caption, lyrics, bpm, duration, keyscale,
timesignature): the LLM skips CoT and generates audio codes directly.
With `lm_batch_size > 1`, all elements share the same prompt (single prefill,
KV cache copied), producing N different audio code sets. See `examples/full.json`.

**Instrumental** (`lyrics="[Instrumental]"`): treated as "lyrics provided",
so the single-pass "Generate tokens" path is used. No lyrics generation.
The DiT was trained with this exact string as the no-vocal condition.

**Passthrough** (`audio_codes` present): LLM is skipped entirely.
Run `ace-synth` to decode existing codes. See `examples/dit-only.json`.

**Cover** (`"task_type": "cover"` + `--src-audio`): no LLM needed. The source audio
(WAV or MP3, any sample rate) is resampled to 48kHz, VAE-encoded to latent
space, then passed through an FSQ roundtrip (tokenize 25Hz to 5Hz, detokenize
back to 25Hz). The lossy 5:1 temporal compression destroys micro-timings,
ornaments and transients, so the DiT diverges from the source and produces
a free reinterpretation rather than a close remix.
`audio_cover_strength` in the JSON controls how many DiT steps see the source
(0.5 = half the steps use source context, half use silence). The caption
steers the style while the source provides loose structure.
Duration is determined by the source audio.

**Cover-nofsq** (`"task_type": "cover-nofsq"` + `--src-audio`): cover variant
that skips the FSQ roundtrip. The DiT receives clean VAE latents at 25Hz,
preserving the full detail of the source. Produces remixes that stay close
to the original structure, melody, and timbre. Pass `--ref-audio` pointing to
the same file as `--src-audio` for best results.
`audio_cover_strength` works well at higher values (0.2 to 0.5) compared to
regular cover. Same JSON fields as cover, just change the task_type.

**Repaint** (`"task_type": "repaint"` + `--src-audio`):
regenerates a time region of the source audio while preserving the rest.
`repainting_start` and `repainting_end` define the region in seconds.
Default start is 0. Default end (`-1`) resolves to source start when
outpainting (start < 0) or source duration otherwise.
Negative start outpaints before the source, end beyond source duration
outpaints after. The source audio is padded with silence before VAE
encoding. `audio_cover_strength` is ignored (the mask handles everything).

```bash
# Inpaint: regenerate seconds 10-25
cat > /tmp/repaint.json << 'EOF'
{
    "task_type": "repaint",
    "caption": "Smooth jazz guitar solo with reverb",
    "lyrics": "[Instrumental]",
    "repainting_start": 10.0,
    "repainting_end": 25.0,
    "inference_steps": 50,
    "guidance_scale": 1.0,
    "shift": 1.0
}
EOF

# Outpaint: generate 5s before the song (end defaults to 0)
cat > /tmp/outpaint.json << 'EOF'
{
    "task_type": "repaint",
    "caption": "Smooth jazz intro building into the main theme",
    "lyrics": "[Instrumental]",
    "repainting_start": -5.0,
    "inference_steps": 50,
    "guidance_scale": 1.0,
    "shift": 1.0
}
EOF

./ace-synth \
    --models models \
    --src-audio song.wav \
    --request /tmp/repaint.json
```

**Lego** (`"task_type": "lego"` + `--src-audio`):
generates a new instrument track layered over an existing backing track.
Requires the `acestep-v15-base` DiT (turbo and SFT do not support lego).

```bash
cat > /tmp/lego.json << 'EOF'
{
    "synth_model": "acestep-v15-base-Q8_0.gguf",
    "caption": "electric guitar riff, funk guitar, house music, instrumental",
    "lyrics": "[Instrumental]",
    "task_type": "lego",
    "track": "guitar",
    "output_format": "wav16",
    "inference_steps": 50,
    "guidance_scale": 1.0,
    "shift": 1.0
}
EOF

./ace-synth \
    --models models \
    --src-audio backing-track.wav \
    --request /tmp/lego.json
```

Available track names for lego, extract, and complete: `vocals`, `backing_vocals`,
`drums`, `bass`, `guitar`, `keyboard`, `percussion`, `strings`, `synth`, `fx`,
`brass`, `woodwinds`.

### Model compatibility

| Task | Turbo | Base/SFT | LM used |
|------|-------|----------|---------|
| text2music | yes | yes | yes |
| cover | yes | yes | no (skipped) |
| cover-nofsq | yes | yes | no (skipped) |
| repaint | yes | yes | no (skipped) |
| lego | no | yes | yes |
| extract | no | yes | no (skipped) |
| complete | no | yes | yes |

For skipped tasks, `caption` and `lyrics` are passed verbatim to the DiT.

### DiT context per mode

What the DiT actually receives in its 128-channel context `[src(64) | mask(64)]`:

| Mode | src channels | mask value | instruction |
|------|-------------|------------|-------------|
| text2music | silence | 1.0 | "Fill the audio semantic mask..." |
| cover | FSQ(src) roundtrip | 1.0 | "Generate audio semantic tokens..." |
| cover-nofsq | raw VAE src (no FSQ) | 1.0 | "Generate audio semantic tokens..." |
| repaint | silence in zone / src outside | 0.0 outside / 1.0 in zone | "Repaint the mask area..." |
| lego (no region) | raw VAE src everywhere | 1.0 | "Generate the TRACK track..." |
| lego (with region) | raw VAE src everywhere | 0.0 outside / 1.0 in zone | "Generate the TRACK track..." |
| extract | raw VAE src | 1.0 | "Extract the TRACK track..." |
| complete | raw VAE src | 1.0 | "Complete the input track..." |

cover uses an FSQ roundtrip (tokenize 25Hz->5Hz then detokenize 5Hz->25Hz). The
lossy compression destroys source detail and the DiT diverges freely.
cover-nofsq skips this roundtrip: same instruction, clean 25Hz latents. The DiT
stays close to the source and produces faithful remixes. Pass
ref_audio = src_audio for best results.
All other tasks with source audio use raw VAE latents (no FSQ).

### Region-mode pipeline (repaint + lego with region)

Region coordinates are resolved in a unified block after mode routing:
`s.rs += left_pad_sec; s.re += left_pad_sec`. When outpainting is active,
source audio has been padded with silence before VAE encoding, so T_cover
and all downstream latent operations naturally reflect the extended canvas.

Two mechanisms do the work when a repaint region is active:

1. **DiT context conditioning**: the `context_latents` tensor fed to the DiT
   via cross-attention carries the full source latents with silence pasted
   inside [t0, t1), paired with a binary mask (1.0 inside, 0.0 outside). The
   DiT was trained on this exact shape and natively regenerates the silenced
   zone. Flow matching runs as a generic denoising loop, unaware of repaint.

2. **Latent splice** (pre-VAE decode): keeps the DiT-generated frames inside
   [t0, t1) and copies the source latents elsewhere (hard cut at frame
   boundary). Single VAE decode produces the final audio. No crossfade: the
   tiled VAE decoder smooths the seam in the waveform on its own.

Key difference repaint vs lego: repaint silences the zone in the DiT context
src (so the DiT generates fresh content there). Lego keeps the full backing
track in context even inside the zone (DiT generates a new layer that
harmonizes with it).

### CLI scope

`ace-synth` and `ace-lm` expose cover and repaint via `--src-audio` with all
model types. Lego, extract, and complete are accessible via JSON request
(`task_type` field) and the HTTP server, but have no dedicated CLI flag:
pass `--src-audio` and set `task_type` in the JSON directly. These three modes
require a base or SFT model (not turbo).


## Request JSON reference

Every field below has a default. Omitting a field from the JSON is strictly
equivalent to sending that field with its default value. Only `caption` is
effectively required: the other defaults produce a valid text2music job on
their own, but without caption the LLM has nothing to work from.

```json
{
    "caption":              "",
    "lyrics":               "",
    "bpm":                  0,
    "duration":             0,
    "keyscale":             "",
    "timesignature":        "",
    "vocal_language":       "",
    "seed":                 -1,
    "lm_batch_size":        1,
    "synth_batch_size":     1,
    "lm_temperature":       0.85,
    "lm_cfg_scale":         2.0,
    "lm_top_p":             0.9,
    "lm_top_k":             0,
    "lm_negative_prompt":   "",
    "use_cot_caption":      true,
    "audio_codes":          "",
    "inference_steps":      0,
    "guidance_scale":       0.0,
    "shift":                0.0,
    "dcw_scaler":           0.0,
    "dcw_high_scaler":      0.0,
    "dcw_mode":             "low",
    "audio_cover_strength": 1.0,
    "cover_noise_strength": 0.0,
    "repainting_start":     0,
    "repainting_end":       -1,
    "latent_shift":         0.0,
    "latent_rescale":       1.0,
    "custom_timesteps":     "",
    "task_type":            "text2music",
    "track":                "",
    "solver":               "euler",
    "lm_mode":              "generate",
    "output_format":        "mp3",
    "peak_clip":            10,
    "mp3_bitrate":          128,
    "synth_model":          "",
    "lm_model":             "",
    "adapter":              "",
    "adapter_scale":        1.0
}
```

`synth_model`, `lm_model` and `adapter` are resolved through the model
registry, scanned from `--models <dir>` (and `--adapters <dir>` for
adapters), by both the HTTP server and the CLI binaries (`ace-lm`,
`ace-synth`, `ace-understand`). Values are GGUF filenames without the
`.gguf` suffix; an empty string falls to the first matching entry of the
registry. There is no CLI flag to bypass the JSON: model selection is a
property of the request, not of the command line.

`lm_mode` picks the LM instruction: `"generate"` (full: metadata + lyrics
+ codes), `"inspire"` (short query to metadata + lyrics, no codes),
`"format"` (caption + lyrics to metadata + lyrics, no codes). `output_format`
picks the audio encoder: `"mp3"`, `"wav16"`, `"wav24"`, `"wav32"`.

### Text conditioning (ace-lm + ace-synth)

**`caption`** (string, required)
Natural language description of the music style, mood, instruments, etc.
Fed to both the LLM and the DiT text encoder.

**`lyrics`** (string, default `""`)
Controls vocal generation. Three valid states:
- `""`: LLM generates lyrics from the caption (phase 1 "Expand" prompt).
- `"[Instrumental]"`: no vocals. Passed directly to the DiT, LLM skips lyrics generation.
- Any other string: user-provided lyrics used as-is, LLM only fills missing metadata.

There is no `instrumental` flag. This field is the single source of truth for
vocal content.

### Metadata (LLM-filled if unset)

**`bpm`** (int, default `0` = unset)
Beats per minute. LLM generates one if 0.

**`duration`** (float seconds, default `0` = unset)
Target audio duration. `0` means the LLM picks it. FSM constrains LLM output
to [10, 600]s; values <= 0 after generation fall back to 120s.

**`keyscale`** (string, default `""` = unset)
Musical key and scale, e.g. `"C major"`, `"F# minor"`. LLM fills if empty.

**`timesignature`** (string, default `""` = unset)
Time signature numerator as a string, e.g. `"4"` for 4/4, `"3"` for 3/4.
LLM fills if empty.

**`vocal_language`** (string, default `""` = unset)
BCP-47 language code for lyrics, e.g. `"en"`, `"fr"`, `"ja"`. Three states:
- `""`: LLM detects the language via CoT and fills this field.
- `"unknown"`: explicit "no specific language" signal to the DiT.
- Any language code: used as-is. When lyrics are being generated, the FSM
  constrains the LLM output to that language.

### Generation control

**`seed`** (int64, default `-1` = random)
RNG seed for the DiT pipeline (Philox noise). The LM always uses a
random seed internally.

**`lm_batch_size`** (int, default `1`)
Number of LM variations. Has no effect on ace-synth.

**`synth_batch_size`** (int, default `1`)
Number of DiT variations per request. Works in all modes: text2music,
cover, repaint, lego, extract, complete. Combined with `lm_batch_size`, you get
`lm_batch_size * synth_batch_size` total outputs.

### Batching

Three rules govern all batching, in both CLIs and the server:

1. Each input JSON is executed independently, as if it were the only one.
2. `seed=-1` is resolved to a random value once per input JSON.
   An explicit seed is used as-is.
3. `lm_batch_size=N` duplicates with consecutive LM-internal seeds.
   `synth_batch_size=N` duplicates with consecutive `seed` values.

**`audio_codes`** (string, default `""`)
Comma-separated FSQ token IDs produced by ace-lm. When non-empty, the
entire LLM pass is skipped and ace-synth decodes these codes directly
(passthrough mode).

**`audio_cover_strength`** (float, default `1.0`)
Only used in `cover` mode. Fraction of DiT steps that see the source audio
as context. At `1.0` all steps use the source. At `0.0` no
steps use the source (pure text2music, source is ignored). Values below 1.0
switch DiT context to silence and encoder hidden states to text2music
instruction at the corresponding step. Lower values give more creative
freedom, higher values preserve more of the original structure.
Defaults to 1.0 for `lego`, `extract`, `complete` (context-switch inactive for these modes).
Ignored in `repaint` mode (the mask handles everything).

**`cover_noise_strength`** (float, default `0.0`)
Only used in `cover` mode. Blends initial noise with source latents before
diffusion starts. `0.0` = pure noise (default). `1.0` = start nearly identical
to the source. The schedule is truncated to the nearest timestep matching the
noise level. `cover_steps` is recalculated against the remaining steps.

**`repainting_start`** (float seconds, default `0`)
**`repainting_end`** (float seconds, default `-1`)
Region boundaries for `repaint` and `lego` modes. Default end (`-1`) resolves
to source start when outpainting (start < 0), source duration otherwise.
Negative start pads silence before, end beyond source duration pads after.
Error if end <= start after adjustment.

**`task_type`** (string, default `"text2music"`)
Controls the generation mode. This field is the single source of truth for
what the pipeline does and is always serialized in request round trips.
Values: `text2music`, `cover`, `cover-nofsq`, `repaint`, `lego`, `extract`, `complete`.

- `text2music`: standard text-to-music synthesis from silence.
- `cover`: re-synthesize source audio with a new style. FSQ roundtrip degrades
  source latents, so the DiT diverges freely. Requires `--src-audio`.
  `audio_cover_strength` controls how many DiT steps see the source.
- `cover-nofsq`: remix source audio without FSQ roundtrip. The DiT works on
  clean 25Hz VAE latents and stays close to the original. Requires `--src-audio`.
  Pass `--ref-audio` = `--src-audio` for best results.
- `repaint`: regenerate a time region of the source audio. Requires `--src-audio`.
  Negative start outpaints before, end beyond duration outpaints after.
- `lego`: generate a new instrument track in context of a backing track. Requires
  `--src-audio` and `track`. Base model only. Output is the generated track
  (behavior analogous to stem generation; the output mix vs isolated stem is
  model-dependent and unverified in this codebase).
  Supports optional region constraint via `repainting_start/end`.
- `extract`: isolate a specific stem from a mixed source. Requires `--src-audio`
  and `track`. Base model only. LM is skipped (same as cover/repaint).
- `complete`: generate a full mix from a single isolated stem. Requires `--src-audio`
  (the isolated stem, e.g. a cappella vocals) and `track` (what to add, e.g. `drums`).
  Base model only. Output duration = source duration. The DiT regenerates all frames
  conditioned on the stem; it does NOT splice or extend temporally.
  `track` can be a pre-formatted string like `"VOCALS | DRUMS"` for multi-stem.

`lego`, `extract`, and `complete` always use the full source context
(`audio_cover_strength` defaults to 1.0 and the context-switch mechanism is inactive).

**`track`** (string, default `""`)
Track name for `lego`, `extract`, and `complete` modes. Standard names: `vocals`, `backing_vocals`, `drums`,
`bass`, `guitar`, `keyboard`, `percussion`, `strings`, `synth`, `fx`, `brass`,
`woodwinds`. Non-standard names produce a warning but are passed through.

### Server model routing (ace-server only)

These fields are parsed by ace-server but are not part of the C++ `AceRequest`
struct. They select which model to load from the `--models` directory.

**`synth_model`** (string, default `""`)
DiT model filename to use for /synth (e.g. `"acestep-v15-turbo-Q8_0.gguf"`).
Empty string keeps the currently loaded DiT, or loads the first available one.

**`lm_model`** (string, default `""`)
LM model filename to use for /lm and /understand (e.g. `"acestep-5Hz-lm-4B-Q8_0.gguf"`).
Empty string keeps the currently loaded LM, or loads the first available one.

**`adapter`** (string, default `""`)
Adapter name from the `--adapters` directory (e.g. `"singer-v2.safetensors"`
or `"my-peft-adapter"`). Empty string means no adapter. Changing the adapter
reloads the DiT (deltas are merged into weights at load time). Supported
algorithm today: LoRA.

**`adapter_scale`** (float, default `1.0`)
Adapter scaling factor. Only used when `adapter` is set.

### LM sampling (ace-lm)

**`lm_temperature`** (float, default `0.85`)
Sampling temperature for both phase 1 (lyrics/metadata) and phase 2 (audio
codes). Lower = more deterministic.

**`lm_cfg_scale`** (float, default `2.0`)
Classifier-Free Guidance scale for the LM. Always active in phase 2 (audio
code generation). In phase 1, CFG is disabled whenever textual expansion is
happening (lyrics generation or CoT caption enrichment). In practice CFG
only applies to phase 1 when lyrics are provided AND `use_cot_caption=false`,
i.e. the LM is filling metadata fields without any free-text generation.
`1.0` disables CFG.

**`lm_top_p`** (float, default `0.9`)
Nucleus sampling cutoff. `1.0` disables.

**`lm_top_k`** (int, default `0` = disabled)
Top-K sampling. `0` disables hard top-K (top_p still applies).

**`lm_negative_prompt`** (string, default `""`)
Negative caption for CFG in phase 2. Empty string falls back to a
caption-less unconditional prompt.

**`use_cot_caption`** (bool, default `true`)
When `true`, the LLM enriches the user caption via CoT and the enriched
version is written to the output JSON (and fed to the DiT). When `false`,
the user caption is preserved verbatim. Only matters when the LLM runs
phase 1 (i.e. some metadata is missing). When all metadata is provided
phase 1 is skipped and the caption is never touched regardless of this flag.

### DiT flow matching (ace-synth)

**`inference_steps`** (int, default `0` = auto)
Number of diffusion denoising steps. `0` resolves from the loaded model:
turbo = `8`, base/SFT = `50`.

**`guidance_scale`** (float, default `0.0` = auto)
CFG scale for the DiT. `0.0` resolves to `1.0` (CFG disabled).
Any value > 1.0 on a turbo model is overridden to 1.0 with a warning.

**`shift`** (float, default `0.0` = auto)
Flow-matching schedule shift. Controls the timestep distribution.
`shift = s*t / (1 + (s-1)*t)`. `0.0` resolves from the loaded model:
turbo = `3.0`, base/SFT = `1.0`.

**`solver`** (string, default `"euler"`)
Diffusion solver, resolved by `solver_lookup()` in `src/solvers/`.
Accepted values:
- `"euler"`: ODE Euler, first order, 1 NFE per step. Deterministic, same seed gives same result.
- `"sde"`: SDE Ancestral, 1 NFE plus Philox renoise per step. Predicts x0 from the velocity, then re-noises to the next timestep with a fresh Philox sample seeded by `seed + step + 1`. Reproducible bit for bit per seed.
- `"dpm3m"`: DPM++ 3M, third order Adams Bashforth multistep, 1 NFE per step. Stateful, bootstraps from Euler then AB2 then AB3.
- `"stork4"`: STORK 4 (4th order ROCK4 Chebyshev sub stepping), 1 NFE per step. Stateful, sub step count tunable via `stork_substeps` (default 10).

Turbo preset: `inference_steps=8, guidance_scale=1.0, shift=3.0`.
Base/SFT preset: `inference_steps=50, guidance_scale=1.0, shift=1.0`.

## ace-lm reference

```
Usage: ./ace-lm --models <dir> --request <json> [options]

Required:
  --models <dir>         Directory of GGUF model files
  --request <json>       Input request JSON (carries lm_model)

Debug:
  --max-seq <N>          KV cache size (default: 8192)
  --no-fsm               Disable FSM constrained decoding
  --no-fa                Disable flash attention
  --no-batch-cfg         Split CFG into two separate forwards
  --clamp-fp16           Clamp hidden states to FP16 range
  --dump-logits <path>   Dump prefill logits (binary f32)
  --dump-tokens <path>   Dump prompt token IDs (CSV)
```

The LM is picked from the request JSON via `lm_model`; empty value falls
to the first LM in the registry. `lm_mode` in the JSON selects the
instruction (generate, inspire, format). Three LM sizes: 0.6B (fast),
1.7B, 4B (best quality).

Batching is controlled by `lm_batch_size` in the request JSON (default 1).
Model weights are read once per decode step for all N sequences.

## ace-synth reference

```
Usage: ./ace-synth --models <dir> --request <json...> [options]

Required:
  --models <dir>          Directory of GGUF model files
  --request <json...>     One or more request JSONs (from ace-lm --request)

Optional:
  --adapters <dir>        Directory of adapter files (enables JSON adapter field)
  --src-audio <path>      Source audio (WAV or MP3)
  --ref-audio <path>      Timbre reference audio (WAV or MP3)

Memory control:
  --vae-chunk <N>         Latent frames per tile (default: 1024)
  --vae-overlap <N>       Overlap frames per side (default: 64)

Debug:
  --no-fa                 Disable flash attention
  --no-batch-cfg          Split DiT CFG into two separate forwards
  --clamp-fp16            Clamp hidden states to FP16 range
  --dump <dir>            Dump intermediate tensors
```

Model selection comes from the first request JSON. `synth_model` picks
the DiT, `adapter` picks an adapter from `--adapters`, `output_format`
picks the output encoder (mp3, wav16, wav24, wav32). Models are loaded
once and reused across all requests.

When `adapter` is set, deltas are merged into the DiT projection weights
at load time (before QKV fusion and GPU upload). For LoRA, the safetensors
file is parsed directly, each lora_A/lora_B pair is multiplied
(`alpha/rank * scale * B @ A`), and the result is added to the base weight
in F32 before requantizing back to the original GGUF type. This is a
static merge: inference runs at full speed with no adapter overhead.
The registry accepts either a safetensors file or a directory containing
`adapter_model.safetensors` and `adapter_config.json` (PEFT format).

`--src-audio` provides source content for cover, repaint, lego, extract and
complete tasks. The audio (WAV or MP3, any sample rate) is resampled to 48kHz
and VAE-encoded once. `audio_cover_strength` in the JSON controls how many
DiT steps use the source context (default 1.0). `cover_noise_strength`
blends the initial noise with source latents to start diffusion closer to
the source (default 0.0).

`--ref-audio` provides a timbre reference, independent of the task. The audio
is VAE-encoded and fed to the 4-layer timbre encoder, which pools to a single
embedding via frame[0]. This conditions the DiT to match the tonal quality of
the reference. When omitted, the timbre encoder receives a single silence
frame (no timbre conditioning).

Batching comes from two sources: multiple `--request` files on the CLI
(or JSON array on the server), and `synth_batch_size` inside each request.
Both are combined: 2 request files with `synth_batch_size=3` yields 6 tracks
in one GPU pass.

## ace-server reference

HTTP server exposing the same pipelines as `ace-lm`, `ace-synth`, and
`ace-understand`. One binary, one port.

POST /lm, POST /synth, POST /understand and POST /vae are all **asynchronous**: they
return a job ID immediately, push the request to a FIFO queue, and the single
worker thread processes jobs in order. Clients poll GET /job?id=N for status
and fetch results with GET /job?id=N&result=1.
Cancel: POST /job?id=N&cancel=1 stops a specific job.

`--models` scans a directory for GGUF files and classifies each by its
`general.architecture` metadata into LM, Text-Enc, DiT, and VAE buckets.
All module lifetime decisions go through the [ModelStore](#modelstore):
default mode keeps one GPU module resident at a time, `--keep-loaded`
keeps the whole working set resident. Requests are serialized by a single
worker thread, no GPU mutex.

The VRAM numbers below assume Q8 quantisation of a 1.7B LM and a 2B DiT.
"Peak" is what you need under the default (STRICT) mode, where only one
module is live at once. "Working set" is what you need under
`--keep-loaded`, where everything stays resident plus the VAE tile
activations during decode. Larger LMs (4B), XL DiT (4B) or heavier
quantisation raise both columns.

| Pipeline | Modules reached | Peak (default) | Working set (`--keep-loaded`) |
|:---------|:----------------|:---------------|:------------------------------|
| LM | Qwen3 LM + KV cache | ~2-3 GB | ~2-3 GB |
| Synth | Qwen3 text-enc, cond-enc, DiT, VAE enc, VAE dec, FSQ tok/detok | ~2-3 GB (DiT or VAE tiles) | ~3-4 GB + tiles |
| Understand | Qwen3 LM, VAE enc, FSQ tok | ~2-3 GB (LM or VAE tiles) | ~2-3 GB + tiles |

VAE tile activations scale with `--vae-chunk` and `--vae-overlap`. Bigger
tiles process audio faster with fewer seams but cost more transient VRAM
during encode or decode; the project default targets 8 GB consumer cards.
Under STRICT those tiles get the full GPU budget alone; under
`--keep-loaded` they sit on top of everything else, so larger cards
earn back the latency they spend reloading.

Endpoints whose pipeline has no models in the registry return 501.

```
Usage: ./ace-server --models <dir> [options]

Required:
  --models <dir>          Directory of GGUF model files

Adapter:
  --adapters <dir>        Directory of adapters

Memory control:
  --keep-loaded           Keep models in VRAM between requests
  --vae-chunk <N>         Latent frames per tile (default: 1024)
  --vae-overlap <N>       Overlap frames per side (default: 64)

Server:
  --host <addr>           Listen address (default: 127.0.0.1)
  --port <N>              Listen port (default: 8080)
  --max-batch <N>         LM batch limit (default: 1)
  --max-seq <N>           KV cache size (default: 8192)

Debug:
  --no-fsm                Disable FSM constrained decoding
  --no-fa                 Disable flash attention
  --no-batch-cfg          Split CFG into two separate forwards (LM + DiT)
  --clamp-fp16            Clamp hidden states to FP16 range
```

Examples:

```bash
# all models in one directory
./ace-server --models /path/to/models

# with adapters
./ace-server --models /path/to/models --adapters /path/to/adapters

# custom port and batch limit
./ace-server --models /path/to/models --host 0.0.0.0 --port 8085 --max-batch 2
```

### Endpoints

```
POST /lm                        Submit LM generation, returns job ID
  body: application/json AceRequest
  response: {"id":"1"}

POST /synth                     Submit synth generation, returns job ID
  body: application/json AceRequest or [AceRequest, ...]
  body: multipart/form-data (request + audio|src_latents + ref_audio|ref_latents)
        latents win over audio when both are sent on the same side
  response: {"id":"2"}

POST /understand                Submit understand, returns job ID
  body: multipart/form-data (audio or src_latents required, optional request JSON)
  response: {"id":"3"}

POST /vae                       Submit VAE encode or decode, returns job ID
  body: multipart/form-data (exactly one of 'audio' or 'src_latents')
        'audio' -> encode path (latents out)
        'src_latents' -> decode path (audio out)
  response: {"id":"4"}

GET  /job?id=N                  Poll job status
  response: {"status":"running|done|failed|cancelled"}

GET  /job?id=N&result=1         Fetch job result
  lm:         application/json [AceRequest, ...]
  synth:      multipart/mixed (one audio part + one latent part per track, paired)
  understand: multipart/mixed (one json part + one latent part for the source)
  vae encode: application/octet-stream (raw .vae bytes, no audio echo: client already has it)
  vae decode: audio/mpeg or audio/wav (raw, no latent echo: client already has it)

POST /job?id=N&cancel=1         Cancel a specific job
  response: {"status":"cancelled"}

GET  /health                    Server health check
  response: {"status":"ok"}

GET  /props                     Server config, models, presets, defaults
  response: application/json

GET  /logs                      SSE stream of server stderr
  response: text/event-stream

GET  /                          Embedded WebUI (gzipped HTML)
```

Latent payload format (src_latents, ref_latents, synth/understand response latent parts, /vae encode response body):
raw f32 little-endian, flat [T, 64], no header. T = size / 256. Same byte
layout neural-codec writes as `.vae` files. Hard cap T <= 15000 frames
(matches the silence_latent buffer baked into the DiT GGUF), 413 over.

`lm_model`, `synth_model`, `adapter`, `adapter_scale` fields in the JSON body
select which model and adapter to load. `lm_mode` picks the LM instruction
(`"generate"`, `"inspire"`, `"format"`); `output_format` picks the audio
encoder for `/synth` (`"mp3"`, `"wav16"`, `"wav24"`, `"wav32"`).
`synth_batch_size` duplicates a request for multiple DiT variations
(clamped to 9). Error responses are JSON: `{"error":"message"}` with 400,
500, 501, or 503 status.

**GET /props** returns available models, server configuration, and the
default AceRequest (source of truth for webui dropdowns and placeholders):
```json
{
  "models": {
    "lm": ["acestep-5Hz-lm-0.6B-Q8_0.gguf", "acestep-5Hz-lm-4B-Q8_0.gguf"],
    "embedding": ["Qwen3-Embedding-0.6B-Q8_0.gguf"],
    "dit": ["acestep-v15-turbo-Q8_0.gguf", "acestep-v15-xl-turbo-Q8_0.gguf"],
    "vae": ["vae-BF16.gguf"]
  },
  "adapters": [],
  "cli": { "max_batch": 1 },
  "default": { "caption": "", "duration": 0, ... }
}
```

### Concurrency

One worker thread consumes jobs from a FIFO queue. LM, synth and
understand requests all land in that same queue and run serially: the
worker calls the pipeline, which goes through the `ModelStore` for its
GPU modules. No GPU mutex, no try_lock, no 503 on busy: the HTTP handler
enqueues and returns a job id immediately, the client polls and the
worker runs in its own time.

Completed jobs sit in memory and are evicted FIFO once the pool exceeds
32 entries, so a disconnected client can poll and fetch the result after
reconnecting. Each job has its own cancel flag, set through `POST
/job?id=N&cancel=1` and polled by the worker between DiT or LM steps.

Model loading and eviction are not this section's concern: see
[ModelStore](#modelstore) for the full story. The short version is that
`--keep-loaded` keeps everything resident across requests, the default
mode keeps one module at a time.

Request bodies are limited to 256 MB (source + reference audio, up to
10 minutes WAV each).

## neural-codec reference

GGML-native neural audio codec based on the Oobleck VAE encoder and decoder.
Serves two purposes: validating the precision of the full VAE chain (encode +
decode roundtrip), and compressing music at 6.8 kbit/s with no perceptible
difference from the original.

```
Usage: ./neural-codec --vae <gguf> --encode|--decode -i <input> [-o <output>] [--q8|--q4]

Required:
  --vae <path>            VAE GGUF file
  --encode | --decode     Encode audio to latent, or decode latent to WAV
  -i <path>               Input (WAV/MP3 for encode, latent for decode)

Output:
  -o <path>               Output file (auto-named if omitted)
  --q8                    Quantize latent to int8 (~13 kbit/s)
  --q4                    Quantize latent to int4 (~6.8 kbit/s)
  --format <fmt>          WAV format: wav16, wav24, wav32 (default: wav16)

Output naming: song.wav -> song.vae (f32) or song.nac8 (Q8) or song.nac4 (Q4)
               song.vae -> song.wav

Memory control:
  --vae-chunk <N>         Latent frames per tile (default: 1024)
  --vae-overlap <N>       Overlap frames per side (default: 64)

Latent formats (decode auto-detects):
  .vae:  flat [T, 64] f32, no header. ~51 kbit/s.
  .nac8: header + per-frame Q8. ~13 kbit/s.
  .nac4: header + per-frame Q4. ~6.8 kbit/s.
```

The encoder is the symmetric mirror of the decoder: same snake activations,
same residual units, strided conv1d for downsampling instead of transposed
conv1d for upsampling. No new GGML ops. Downsample 2x4x4x6x10 = 1920x.

48kHz stereo audio is compressed to 64-dimensional latent frames at 25 Hz.
Three output formats, decode auto-detects from file content:

| Format | Frame size | Bitrate | 3 min song | vs .vae (cossim) |
|--------|-----------|---------|------------|------------------|
| .vae   | 256B      | 51 kbit/s | 1.1 MB   | baseline         |
| .nac8  | 66B       | 13 kbit/s | 290 KB   | 0.9999           |
| .nac4  | 34B       | 6.8 kbit/s | 150 KB  | 0.989            |

NAC = Neural Audio Codec. The .nac8 and .nac4 file formats are headerless
except for a 4-byte magic (`NAC8` or `NAC4`) and a uint32 frame count. The
.vae file is the raw VAE encoder output (flat f32, no header), the same
byte payload the HTTP API exchanges as latent multipart parts.
Q8 quantization error is 39 dB below the VAE reconstruction error (free).
Q4 quantization error is 16 dB below the VAE reconstruction error (inaudible
on most material).

```bash
# encode (Q4: 6.8 kbit/s, ~150 KB for 3 minutes)
./neural-codec --vae models/vae-BF16.gguf --encode --q4 -i song.wav -o song.nac4

# encode (Q8: 13 kbit/s, ~290 KB for 3 minutes)
./neural-codec --vae models/vae-BF16.gguf --encode --q8 -i song.wav -o song.nac8

# decode (auto-detects format)
./neural-codec --vae models/vae-BF16.gguf --decode -i song.nac4 -o song_decoded.wav

# roundtrip validation: compare song.wav and song_decoded.wav with your ears
```

## mp3-codec reference

Standalone MIT-licensed MPEG1 Layer III encoder and decoder. No external
dependencies. The encoder is used by `ace-synth` for MP3 output. The decoder
uses minimp3 (CC0). Reads WAV or MP3, writes WAV or MP3 (auto-detected
from output extension).

```
Usage: ./mp3-codec -i <input> -o <output> [options]

  -i <path>     Input file (WAV or MP3)
  -o <path>     Output file (WAV or MP3)
  -b <kbps>     Bitrate for MP3 encoding (default: 128)
  --format <fmt>  WAV format: wav16, wav24, wav32 (default: wav16)

Mode is auto-detected from output extension.

Examples:
  ./mp3-codec -i song.wav -o song.mp3
  ./mp3-codec -i song.wav -o song.mp3 -b 192
  ./mp3-codec -i song.mp3 -o song.wav
  ./mp3-codec -i song.mp3 -o song.wav --format wav32
```

## ace-understand reference

Reverse pipeline: audio -> VAE encode -> FSQ tokenize -> LM understand ->
metadata + lyrics. The output JSON is reusable as ace-lm or ace-synth input.

```
Usage: ./ace-understand --models <dir> --src-audio <path> [--request <json>] [options]

Required:
  --models <dir>          Directory of GGUF model files
  --src-audio <path>      Source audio (WAV or MP3, any sample rate)

Optional:
  --request <json>        Request JSON carrying model selection and
                          sampling params (lm_model, synth_model,
                          lm_temperature, lm_top_p, lm_top_k)

When no --request is given, understand defaults apply
(temperature 0.3, top_p disabled).

Output:
  -o <json>               Output JSON (default: stdout summary)

Memory control:
  --vae-chunk <N>         Latent frames per tile (default: 1024)
  --vae-overlap <N>       Overlap frames per side (default: 64)

Debug:
  --max-seq <N>           KV cache size (default: 8192)
  --no-fsm                Disable FSM constrained decoding
  --no-fa                 Disable flash attention
  --dump <dir>            Dump tok_latents + tok_codes (skip LM)
```

Model selection comes from the request JSON (`lm_model`, `synth_model`,
`vae`).

## Architecture

```
ace-lm (Qwen3 causal LM, 0.6B/1.7B/4B)
  Phase 1 (if needed): CoT generates bpm, keyscale, timesignature, lyrics
  Phase 2: audio codes (5Hz tokens, FSQ vocabulary)
  Both phases batched: N sequences per forward, weights read once
  CFG with dual KV cache per batch element (cond + uncond)
  Output: request0.json .. requestN-1.json

ace-synth
  BPE tokenize
  Qwen3-Embedding (28L text encoder)
  CondEncoder (lyric 8L + timbre 4L + text_proj)
  FSQ detokenizer (audio codes -> flow matching source latents)
  Adapter merge (optional: LoRA safetensors delta -> dequant/merge/requant at load)
  DiT (2B: 24L H=2048, XL: 32L H=2560, flow matching)
  VAE (AutoencoderOobleck, tiled decode)
  WAV stereo 48kHz

ace-understand (reverse pipeline)
  Audio read (WAV/MP3, any rate -> 48kHz stereo)
  VAE encode (tiled, AutoencoderOobleck encoder)
  FSQ tokenize (latent -> 5Hz codes via 2L attention pooler)
  Qwen3 LM (understand prompt: codes -> CoT metadata + lyrics)
  FSM constrains CoT fields, audio codes blocked after </think>
  No CFG, no batch. Single sequence, greedy-ish (temperature=0.3)
  Output: JSON with caption, lyrics, bpm, key, duration, language
```

## LM specifics

ace-lm is not a general-purpose chat engine. It is a two-phase autoregressive
pipeline specialized for ACE-Step music generation.

Phase 1 (CoT) generates structured metadata (bpm, keyscale, timesignature, caption,
duration, language) and optionally lyrics via chain-of-thought reasoning. An FSM
(finite state machine) built from a prefix tree enforces valid field names and values
at every decode step, hard-masking invalid tokens before sampling.

Phase 2 (audio codes) generates 5Hz FSQ tokens. The FSQ codec uses levels
[8,8,8,5,5,5] producing 64000 distinct codes (8*8*8*5*5*5). The tokenizer
reserves 65535 slots (audio_code_0 to audio_code_65534) appended to the base
Qwen3 vocabulary; the 1535 extra slots are unused by the codec. A partial LM
head projects only the audio code subrange of the embedding matrix, cutting
the output GEMM by 70% compared to full-vocab projection.
Classifier-free guidance (CFG) is fused into the batch dimension: N
conditional and N unconditional sequences are packed into a single forward pass
(2*N tokens, one weight read), then combined as
`logits = uncond + scale * (cond - uncond)`. The KV cache is a single 4D tensor
`[D, max_seq, Nkv, n_sets]` shared across all batch elements and CFG paths. Shared
prompts are prefilled once and cloned to other KV sets via copy, avoiding redundant
prefills.

## Accuracy

Test logs (turbo + SFT, seed 42, Philox noise, multiple quantizations):
[`tests/`](https://github.com/ServeurpersoCom/acestep.cpp/tree/master/tests)

Each script compares GGML C++ output against the Python reference
(cosine similarity per intermediate tensor). Requires the original
ACE-Step-1.5 repo cloned alongside acestep.cpp (`../ACE-Step-1.5`).

```bash
cd tests
python3 debug-lm-logits.py    # Qwen3 LM: first-token logits GGML vs PyTorch (0.6B/1.7B/4B)
python3 debug-detok-cossim.py # FSQ detokenizer: step-by-step cossim C++ vs Python
python3 debug-dit-cossim.py   # DiT: per-layer cossim GGML vs Python (turbo/SFT, BF16/quantized)
```

## Patched GGML fork

Uses a patched GGML fork (submodule) with two new ops, a Metal im2col optimization, and
a CUDA bugfix for the Oobleck VAE decoder. All backends: CPU, CUDA, ROCm, Metal, Vulkan.
F32/F16/BF16 data types. The DiT uses only standard GGML ops and needs no patches.

The VAE reconstructs audio from latent space through 5 upsampling blocks (total 1920x),
each running a transposed convolution followed by 3 WaveNet-style residual units with
dilated convolutions and Snake activations. A single tile builds a graph of 36 snake
activations, 5 transposed convolutions, and 32 regular convolutions. At the final blocks,
sequence lengths reach 491520 timesteps, which stresses GGML ops designed for short NLP
sequences.

### `GGML_OP_SNAKE` (fused Snake activation)

Computes y = x + sin^2(a * x) * inv_b in a single kernel.
The Oobleck VAE calls this 36 times per tile. Without a fused op, each activation
requires 5 separate GGML kernels (mul, sin, sqr, mul, add), causing 5x the memory
traffic. The fused kernel reads x once and writes y once. BF16 cast nodes before/after
each snake call halve memory bandwidth at the cost of negligible precision loss
(cossim > 0.999 vs F32 baseline).

### `GGML_OP_COL2IM_1D` (scatter-add for GEMM-based conv_transpose_1d)

Gather-based reconstruction of a 1D signal from GEMM columns [K*OC, T_in] to
[T_out, OC], with fused padding crop via the p0 parameter.
Upstream `ggml_conv_transpose_1d` uses a naive kernel (one scalar FMA loop per output
element, no shared memory, no tensor cores). The VAE spends 40% of its FLOP budget on
transposed convolutions. We decompose each as `mul_mat + col2im_1d`, routing the heavy
GEMM through cuBLAS/BLAS/MPS tensor cores. The col2im_1d gather has a 2-iteration inner
loop and is pure bandwidth. BF16 cast nodes around col2im_1d halve the scatter bandwidth.

### Metal: `kernel_im2col_1d` (flat 1D dispatch)

The generic Metal `kernel_im2col` dispatches (IC, 1, OW) threadgroups with K threads
each. For the VAE's 1D convolutions with small kernels (k=1 or k=7), this wastes 78-97%
of SIMD lanes (7 or 1 active threads per 32-wide SIMD group). The dedicated
`kernel_im2col_1d` uses a flat dispatch identical to snake and col2im_1d:
(total/256, 1, 1) threadgroups with 256 threads, achieving full SIMD utilization.
The dispatch branches on `is_2D` at runtime; the 2D path and kernel are unchanged.
CUDA and Vulkan already use flat dispatch and are not affected.

VAE decode (M2 Pro 16GB, 86.8s audio @ 48kHz stereo):

| chunk | overlap | im2col    | tiles | time   |
|------:|--------:|-----------|------:|-------:|
|   256 |      64 | generic   |    17 | 71.2s  |
|  1024 |      16 | generic   |     3 | 38.9s  |
|   256 |      64 | im2col_1d |    17 | 31.8s  |
|  1024 |      16 | im2col_1d |     3 | 18.3s  |

### Bugfix: `im2col` gridDim.y overflow (CUDA)

Upstream `im2col_kernel` uses OW directly as grid dimension Y, which exceeds the CUDA
65535 gridDim limit on long sequences. The VAE calls `ggml_conv_1d` (im2col path) 32
times per tile at output widths up to 491520. Fixed with a grid-stride loop on OW and
`MIN(OW, MAX_GRIDDIM_Z)` clamping.

### Upstream divergence

The GGML submodule diverges from upstream only by the addition of
`GGML_OP_SNAKE` and `GGML_OP_COL2IM_1D`. No existing upstream kernel is
modified. These ops are required; the VAE does not work without them.

An earlier approach patched the upstream naive ops instead of adding custom
ones. Those patches were dropped. They are documented here in case someone
wants to study the naive path:

- `conv_transpose_1d`: bounded loop replacing O(T_in) brute-force, CUDA and Metal
- `im2col`: grid-stride loop on OW to fix gridDim.y overflow for large tensors
