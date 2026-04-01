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
| acestep-v15-turbo-Q8_0.gguf | DiT + CondEncoder (24L, H=2048) | 2.4 GB |
| vae-BF16.gguf | AutoencoderOobleck | 322 MB |

Three LM sizes: 0.6B (fast), 1.7B, 4B (best quality).
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
    --request /tmp/request.json \
    --lm models/acestep-5Hz-lm-4B-Q8_0.gguf

# DiT+VAE: request0.json -> request00.mp3
./ace-synth \
    --request /tmp/request0.json \
    --embedding models/Qwen3-Embedding-0.6B-Q8_0.gguf \
    --dit models/acestep-v15-turbo-Q8_0.gguf \
    --vae models/vae-BF16.gguf
```

With a LoRA adapter (PEFT directory or ComfyUI single file):

```bash
# PEFT directory (contains adapter_model.safetensors + adapter_config.json)
./ace-synth \
    --request /tmp/request0.json \
    --embedding models/Qwen3-Embedding-0.6B-Q8_0.gguf \
    --dit models/acestep-v15-turbo-Q8_0.gguf \
    --vae models/vae-BF16.gguf \
    --lora /path/to/lora-adapter

# ComfyUI single .safetensors file (alpha baked in, no config needed)
./ace-synth \
    --request /tmp/request0.json \
    --embedding models/Qwen3-Embedding-0.6B-Q8_0.gguf \
    --dit models/acestep-v15-turbo-Q8_0.gguf \
    --vae models/vae-BF16.gguf \
    --lora best_sft_v2_2338_comfyui.safetensors
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
    --request /tmp/request.json \
    --lm models/acestep-5Hz-lm-4B-Q8_0.gguf

# DiT+VAE: both requests in one GPU batch -> request00.mp3, request10.mp3
./ace-synth \
    --request /tmp/request0.json /tmp/request1.json \
    --embedding models/Qwen3-Embedding-0.6B-Q8_0.gguf \
    --dit models/acestep-v15-turbo-Q8_0.gguf \
    --vae models/vae-BF16.gguf
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
    --src-audio song.wav \
    --request /tmp/cover.json \
    --embedding models/Qwen3-Embedding-0.6B-Q8_0.gguf \
    --dit models/acestep-v15-turbo-Q8_0.gguf \
    --vae models/vae-BF16.gguf \
```

Ready-made examples in `examples/`:

```bash
cd examples
./simple.sh                    # caption only, LLM fills everything
./simple-batch.sh              # 2 songs from one prompt (lm_batch_size=2)
./partial.sh                   # caption + lyrics + duration
./full.sh                      # all metadata provided
./dit-only.sh                  # skip LLM, DiT from noise
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
space and used as DiT context instead of silence.
`audio_cover_strength` in the JSON controls how many DiT steps see the source
(0.5 = half the steps use source context, half use silence). The caption
steers the style while the source provides structure, melody, and rhythm.
Duration is determined by the source audio.

**Repaint** (`"task_type": "repaint"` + `--src-audio`):
regenerates a time region of the source audio while preserving the rest.
The DiT receives a binary mask: 1.0 inside the region (generate), 0.0 outside
(keep original). Source latents outside the region provide context; silence
fills the repaint zone. `repainting_start` and `repainting_end` define the
region in seconds. -1 on start means 0s, -1 on end means source duration.
`audio_cover_strength` is ignored in repaint mode (the mask handles everything).

```bash
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

./ace-synth \
    --src-audio song.wav \
    --request /tmp/repaint.json \
    --embedding models/Qwen3-Embedding-0.6B-Q8_0.gguf \
    --dit models/acestep-v15-sft-Q8_0.gguf \
    --vae models/vae-BF16.gguf
```

**Lego** (`"task_type": "lego"` + `--src-audio`):
generates a new instrument track layered over an existing backing track.
See `examples/lego.json` and `examples/lego.sh`.

```bash
cat > /tmp/lego.json << 'EOF'
{
    "caption": "electric guitar riff, funk guitar, house music, instrumental",
    "lyrics": "[Instrumental]",
    "task_type": "lego",
    "track": "guitar",
    "inference_steps": 50,
    "guidance_scale": 1.0,
    "shift": 1.0
}
EOF

./ace-synth \
    --src-audio backing-track.wav \
    --request /tmp/lego.json \
    --embedding models/Qwen3-Embedding-0.6B-Q8_0.gguf \
    --dit models/acestep-v15-base-Q8_0.gguf \
    --vae models/vae-BF16.gguf \
    --wav
```

Available track names for lego, extract, and complete: `vocals`, `backing_vocals`,
`drums`, `bass`, `guitar`, `keyboard`, `percussion`, `strings`, `synth`, `fx`,
`brass`, `woodwinds`.

## Request JSON reference

Only `caption` is required. All other fields default to "unset" which means
the LLM fills them, or a sensible runtime default is applied.

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
    "audio_cover_strength": 1.0,
    "cover_noise_strength": 0.0,
    "repainting_start":     -1,
    "repainting_end":       -1,
    "task_type":            "",
    "track":                "",
    "synth_model":          "",
    "lm_model":             "",
    "lora":                 "",
    "lora_scale":           1.0
}
```

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
Forced to 1.0 for `lego`, `extract`, `complete`.
Ignored in `repaint` mode (the mask handles everything).

**`cover_noise_strength`** (float, default `0.0`)
Only used in `cover` mode. Blends initial noise with source latents before
diffusion starts. `0.0` = pure noise (default). `1.0` = start nearly identical
to the source. The schedule is truncated to the nearest timestep matching the
noise level. `cover_steps` is recalculated against the remaining steps.

**`repainting_start`** (float seconds, default `-1`)
**`repainting_end`** (float seconds, default `-1`)
Region boundaries for `repaint` and `complete` modes.
`-1` on start means 0s (beginning), `-1` on end means source duration (end).
Error if end <= start after resolve.

**`task_type`** (string, default `""` = `text2music`)
Controls the generation mode. This field is the single source of truth for
what the pipeline does. Empty is equivalent to `text2music`.
Values: `text2music`, `cover`, `repaint`, `lego`, `extract`, `complete`.

- `text2music`: standard text-to-music synthesis from silence.
- `cover`: re-synthesize source audio with a new style. Requires `--src-audio`.
  `audio_cover_strength` controls how many DiT steps see the source.
- `repaint`: regenerate a time region of the source audio. Requires `--src-audio`
  and `repainting_start/end`.
- `lego`: generate a new instrument track over a backing track. Requires
  `--src-audio` and `track`.
- `extract`: isolate a track from a mix. Requires `--src-audio` and `track`.
- `complete`: fill or extend a partial track. Requires `--src-audio` and `track`.
  Uses `repainting_start/end` for region selection.

`lego`, `extract`, and `complete` force `audio_cover_strength` to 1.0
(all DiT steps see the source audio).

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

**`lora`** (string, default `""`)
LoRA adapter name from the `--loras` directory (e.g. `"singer-v2.safetensors"`
or `"my-peft-lora"`). Empty string means no LoRA. Changing the LoRA reloads
the DiT (LoRA is merged into weights at load time).

**`lora_scale`** (float, default `1.0`)
LoRA scaling factor. Only used when `lora` is set.

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

Turbo preset: `inference_steps=8, guidance_scale=1.0, shift=3.0`.
Base/SFT preset: `inference_steps=50, guidance_scale=1.0, shift=1.0`.

## ace-lm reference

```
Usage: ace-lm --request <json> --lm <gguf> [options]

Required:
  --request <json>       Input request JSON
  --lm <gguf>            5Hz LM GGUF file

Debug:
  --max-seq <N>          KV cache size (default: 8192)
  --no-fsm               Disable FSM constrained decoding
  --no-fa                Disable flash attention
  --no-batch-cfg         Split CFG into two N=1 forwards
  --clamp-fp16           Clamp hidden states to FP16 range
  --dump-logits <path>   Dump prefill logits (binary f32)
  --dump-tokens <path>   Dump prompt token IDs (CSV)
```

Three LLM sizes: 0.6B (fast), 1.7B, 4B (best quality).

Batching is controlled by `lm_batch_size` in the request JSON (default 1).
Model weights are read once per decode step for all N sequences.

## ace-synth reference

```
Usage: ace-synth --request <json...> --embedding <gguf> --dit <gguf> --vae <gguf> [options]

Required:
  --request <json...>     One or more request JSONs (from ace-lm --request)
  --embedding <gguf>      Embedding GGUF file
  --dit <gguf>            DiT GGUF file
  --vae <gguf>            VAE GGUF file

Audio:
  --src-audio <file>      Source audio (WAV or MP3)
  --ref-audio <file>      Timbre reference audio (WAV or MP3)

LoRA:
  --lora <path>           LoRA safetensors file or directory
  --lora-scale <float>    LoRA scaling factor (default: 1.0)

Output:
  Default: MP3 at 128 kbps. input.json -> input0.mp3, input1.mp3, ...
  --mp3-bitrate <kbps>    MP3 bitrate (default: 128)
  --wav                   Output WAV instead of MP3

VAE tiling (memory control):
  --vae-chunk <N>         Latent frames per tile (default: 256)
  --vae-overlap <N>       Overlap frames per side (default: 64)

Debug:
  --no-fa                 Disable flash attention
  --clamp-fp16            Clamp hidden states to FP16 range
  --dump <dir>            Dump intermediate tensors
```

Models are loaded once and reused across all requests.

When `--lora` is provided, LoRA deltas are merged into the DiT projection
weights at load time (before QKV fusion and GPU upload). The safetensors
file is parsed directly, each lora_A/lora_B pair is multiplied
(`alpha/rank * scale * B @ A`), and the result is added to the base weight
in F32 before requantizing back to the original GGUF type. This is a
static merge: inference runs at full speed with no adapter overhead.
`--lora` accepts either a safetensors file or a directory containing
`adapter_model.safetensors` and `adapter_config.json` (PEFT format).

`--src-audio` provides source content for cover, repaint, lego, extract and
complete tasks. The audio (WAV or MP3, any sample rate) is resampled to 48kHz
and VAE-encoded once. `audio_cover_strength` in the JSON controls how many
DiT steps use the source context (default 1.0). `cover_noise_strength`
blends the initial noise with source latents to start diffusion closer to
the source (default 0.0).

`--ref-audio` provides a timbre reference, independent of the task. The audio
is VAE-encoded and its first 750 latent frames (30s) are fed to the timbre
encoder. This conditions the DiT to match the tonal quality of the reference.
When omitted, the timbre encoder receives silence (no timbre conditioning).

Batching comes from two sources: multiple `--request` files on the CLI
(or JSON array on the server), and `synth_batch_size` inside each request.
Both are combined: 2 request files with `synth_batch_size=3` yields 6 tracks
in one GPU pass.

## ace-server reference

HTTP server exposing the same pipelines as `ace-lm`, `ace-synth`, and
`ace-understand` over three POST endpoints. One binary, one port.

`--models` scans a directory for GGUF files and classifies each by its
`general.architecture` metadata into LM, Text-Enc, DiT, and VAE buckets.
Models are loaded on demand when a request arrives, and swapped when the
request JSON specifies a different `synth_model`, `lm_model`, or `lora`.

| Pipeline | GGUF architectures needed | Enables | VRAM (approx) |
|:---------|:--------------------------|:--------|:--------------|
| LM | `acestep-lm` | /lm, /understand | ~7 GB (batch=1) |
| Synth | `acestep-text-enc` + `acestep-dit` + `acestep-vae` | /synth | ~12 GB |

When models for both pipelines are present, all three endpoints are active
and /synth runs concurrently with /lm (disjoint GPU memory, separate mutexes).
Endpoints whose pipeline has no models in the registry return 501.

```
Usage: ace-server --models <dir> [options]

Required:
  --models <dir>          Directory of GGUF model files

LoRA:
  --loras <dir>           Directory of LoRA adapters

VAE tiling (memory control):
  --vae-chunk <N>         Latent frames per tile (default: 256)
  --vae-overlap <N>       Overlap frames per side (default: 64)

Output:
  --mp3-bitrate <kbps>    MP3 bitrate (default: 128)

Server:
  --host <addr>           Listen address (default: 127.0.0.1)
  --port <N>              Listen port (default: 8080)
  --max-batch <N>         LM batch limit (default: 1)
  --max-seq <N>           KV cache size (default: 8192)

Debug:
  --no-fsm                Disable FSM constrained decoding
  --no-fa                 Disable flash attention
  --no-batch-cfg          Split CFG into two N=1 forwards
  --clamp-fp16            Clamp hidden states to FP16 range
```

Examples:

```bash
# all models in one directory, load on demand
./ace-server --models /path/to/models

# with LoRA adapters
./ace-server --models /path/to/models --loras /path/to/loras

# custom port and batch limit
./ace-server --models /path/to/models --host 0.0.0.0 --port 8085 --max-batch 2
```

### Endpoints

**POST /lm** accepts an AceRequest JSON body, runs the LM pipeline, and
returns a JSON array of enriched requests (`Content-Type: application/json`).
`lm_batch_size` in the input controls the array length (clamped to `1..max_batch`).
`lm_model` in the JSON selects the LM to load.
Requires at least one `acestep-lm` GGUF in `--models`.

**POST /synth** runs the synth pipeline and returns audio (MP3 or WAV via `?wav=1`).
Three input modes: a JSON array `[{req0}, {req1}]` for batch generation,
a single `application/json` object for one track, or `multipart/form-data`
with a `request` part (JSON) and an `audio` part (WAV or MP3) for
cover/repaint/lego/extract/complete. `synth_batch_size` in each request duplicates it for
multiple DiT variations (e.g. multipart with `synth_batch_size=3` produces
3 tracks from the same audio). Total track count (after expansion) is
clamped to 9.
Response format depends on batch size:
batch=1: raw audio body.
batch>1: `Content-Type: multipart/mixed`, each part is raw audio.
Metadata (seed, duration, etc) is already in the request JSON from /lm.
`synth_model`, `lora`, `lora_scale` in the JSON select the DiT and LoRA to load.
Requires `acestep-text-enc` + `acestep-dit` + `acestep-vae` GGUFs in `--models`.

**POST /understand** runs the reverse pipeline (audio -> metadata + lyrics + codes)
and returns an AceRequest JSON. Two input modes: `multipart/form-data` with an
`audio` part (required) and optional `request` part (sampling params), or plain
`application/json` with `audio_codes` for codes-only mode (skip VAE + FSQ).
Audio input mode requires LM + DiT + VAE. Codes-only mode requires just LM.

**GET /health** returns `{"status":"ok"}`.

**GET /props** returns available models, server configuration, and the
default AceRequest (source of truth for webui dropdowns and placeholders):
```json
{
  "models": {
    "lm": ["acestep-5Hz-lm-0.6B-Q8_0.gguf", "acestep-5Hz-lm-4B-Q8_0.gguf"],
    "embedding": ["Qwen3-Embedding-0.6B-Q8_0.gguf"],
    "dit": ["acestep-v15-turbo-Q8_0.gguf", "acestep-v15-base-Q8_0.gguf"],
    "vae": ["vae-BF16.gguf"]
  },
  "loras": [],
  "cli": { "max_batch": 1, "mp3_bitrate": 128 },
  "default": { "caption": "", "duration": 0, ... }
}
```

Error responses are JSON: `{"error":"message"}` with 400, 500, 501, or
503 status. 503 includes a `Retry-After` header.

### Concurrency

When both LM and synth are loaded, /synth gets its own mutex and runs
concurrently with /lm on the GPU (disjoint models, disjoint memory).
/lm and /understand always share a mutex because they use the same
Qwen3 KV cache. When only one group is loaded, everything is serial.

Each handler uses `try_lock`: if the GPU is busy, the client gets an
instant 503 with `Retry-After`. No request ever blocks waiting for
another to finish.

When the requested model differs from the one currently loaded, the old
model is freed and the new one is loaded from disk before processing the
request. If the load fails, the handler returns 500.

Request bodies are limited to 120 MB (enough for a 10-minute WAV upload
via multipart).

## neural-codec reference

GGML-native neural audio codec based on the Oobleck VAE encoder and decoder.
Serves two purposes: validating the precision of the full VAE chain (encode +
decode roundtrip), and compressing music at 6.8 kbit/s with no perceptible
difference from the original.

```
Usage: neural-codec --vae <gguf> --encode|--decode -i <input> [-o <o>] [--q8|--q4]

Required:
  --vae <path>            VAE GGUF file
  --encode | --decode     Encode audio to latent, or decode latent to WAV
  -i <path>               Input (WAV/MP3 for encode, latent for decode)

Output:
  -o <path>               Output file (auto-named if omitted)
  --q8                    Quantize latent to int8 (~13 kbit/s)
  --q4                    Quantize latent to int4 (~6.8 kbit/s)

Output naming: song.wav -> song.latent (f32) or song.nac8 (Q8) or song.nac4 (Q4)
               song.latent -> song.wav

VAE tiling (memory control):
  --vae-chunk <N>         Latent frames per tile (default: 256)
  --vae-overlap <N>       Overlap frames per side (default: 64)

Latent formats (decode auto-detects):
  f32:  flat [T, 64] f32, no header. ~51 kbit/s.
  NAC8: header + per-frame Q8. ~13 kbit/s.
  NAC4: header + per-frame Q4. ~6.8 kbit/s.
```

The encoder is the symmetric mirror of the decoder: same snake activations,
same residual units, strided conv1d for downsampling instead of transposed
conv1d for upsampling. No new GGML ops. Downsample 2x4x4x6x10 = 1920x.

48kHz stereo audio is compressed to 64-dimensional latent frames at 25 Hz.
Three output formats, decode auto-detects from file content:

| Format | Frame size | Bitrate | 3 min song | vs f32 (cossim) |
|--------|-----------|---------|------------|-----------------|
| f32    | 256B      | 51 kbit/s | 1.1 MB   | baseline        |
| NAC8   | 66B       | 13 kbit/s | 290 KB   | 0.9999          |
| NAC4   | 34B       | 6.8 kbit/s | 150 KB  | 0.989           |

NAC = Neural Audio Codec. The NAC8 and NAC4 file formats are headerless
except for a 4-byte magic (`NAC8` or `NAC4`) and a uint32 frame count.
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
Usage: mp3-codec -i <input> -o <o> [options]

  -i <path>   Input file (WAV or MP3)
  -o <path>   Output file (WAV or MP3)
  -b <kbps>   Bitrate for MP3 encoding (default: 128)

Mode is auto-detected from output extension.

Examples:
  mp3-codec -i song.wav -o song.mp3
  mp3-codec -i song.wav -o song.mp3 -b 192
  mp3-codec -i song.mp3 -o song.wav
```

## ace-understand reference

Reverse pipeline: audio (or pre-existing audio codes) -> LM understand ->
metadata + lyrics. The output JSON is reusable as ace-lm or ace-synth input.

Two input modes: `--src-audio` runs the full chain (VAE encode + FSQ tokenize +
LM), `--request` with an `audio_codes` field skips straight to the LM.

```
Usage: ace-understand [--src-audio <file> --dit <gguf> --vae <gguf> | --request <json>] --lm <gguf>

Audio input (full pipeline):
  --src-audio <file>      Source audio (WAV or MP3, any sample rate)
  --dit <gguf>            DiT GGUF (for FSQ tokenizer weights + silence_latent)
  --vae <gguf>            VAE GGUF (for audio encoding)

Code input (skip VAE + tokenizer):
  --request <json>        Request JSON with audio_codes field

Required:
  --lm <gguf>             5Hz LM GGUF file

Output:
  -o <json>               Output JSON (default: stdout summary)

Sampling params (lm_temperature, lm_top_p, lm_top_k) come from the
request JSON. Without --request, understand defaults apply
(temperature=0.3, top_p disabled).

VAE tiling:
  --vae-chunk <N>         Latent frames per tile (default: 256)
  --vae-overlap <N>       Overlap frames per side (default: 64)

Debug:
  --max-seq <N>           KV cache size (default: 8192)
  --no-fsm                Disable FSM constrained decoding
  --no-fa                 Disable flash attention
  --dump <dir>            Dump tok_latents + tok_codes (skip LM)
```

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
  LoRA merge (optional: safetensors delta -> dequant/merge/requant at load)
  DiT (24L flow matching, Euler steps)
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
