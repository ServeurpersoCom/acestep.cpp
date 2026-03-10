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

# CPU with OpenBLAS (recommended for CPU-only machines)
apt install pkg-config libopenblas-dev  # Debian/Ubuntu
cmake .. -DGGML_BLAS=ON

# Combine as needed
cmake .. -DGGML_CUDA=ON -DGGML_BLAS=ON

cmake --build . --config Release -j$(nproc)
```

Builds four binaries: `ace-qwen3` (LLM), `dit-vae` (DiT + VAE), `neural-codec` (VAE encode/decode) and `mp3-codec` (MP3 encoder/decoder).

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
./checkpoints.sh          # download raw HF checkpoints (turbo + 4B LM)
./checkpoints.sh --all    # all variants (SFT, shift1/3, 0.6B/1.7B LM)
python3 convert.py        # convert all checkpoints to GGUF (models/)
./quantize.sh             # quantize BF16 -> Q4_K_M/Q5_K_M/Q6_K/Q8_0
```

`checkpoints.sh` downloads safetensors, config.json, and tokenizer files
into `checkpoints/`. `convert.py` packs everything into self-contained
GGUF files in `models/`, bundling BPE tokenizer, silence_latent, and
config metadata so no external file is needed at runtime.

</details>

## Quick start

`ace-qwen3` generates lyrics and audio codes, `dit-vae` synthesizes audio.
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
./build/ace-qwen3 \
    --request /tmp/request.json \
    --model models/acestep-5Hz-lm-4B-Q8_0.gguf

# DiT+VAE: request0.json -> request00.mp3
./build/dit-vae \
    --request /tmp/request0.json \
    --text-encoder models/Qwen3-Embedding-0.6B-Q8_0.gguf \
    --dit models/acestep-v15-turbo-Q8_0.gguf \
    --vae models/vae-BF16.gguf
```

With a LoRA adapter (PEFT directory or ComfyUI single file):

```bash
# PEFT directory (contains adapter_model.safetensors + adapter_config.json)
./build/dit-vae \
    --request /tmp/request0.json \
    --text-encoder models/Qwen3-Embedding-0.6B-Q8_0.gguf \
    --dit models/acestep-v15-turbo-Q8_0.gguf \
    --vae models/vae-BF16.gguf \
    --lora /path/to/lora-adapter

# ComfyUI single .safetensors file (alpha baked in, no config needed)
./build/dit-vae \
    --request /tmp/request0.json \
    --text-encoder models/Qwen3-Embedding-0.6B-Q8_0.gguf \
    --dit models/acestep-v15-turbo-Q8_0.gguf \
    --vae models/vae-BF16.gguf \
    --lora best_sft_v2_2338_comfyui.safetensors
```

Generate multiple songs at once with `--batch`:

```bash
# LLM: 2 LM variations x 2 DiT variations = 4 WAVs total
# -> request0.json, request1.json (different lyrics/codes, seeds auto+0, auto+1)
./build/ace-qwen3 \
    --request /tmp/request.json \
    --model models/acestep-5Hz-lm-4B-Q8_0.gguf \
    --batch 2

# DiT+VAE: (2 DiT variations of LM output 1 and 2)
# -> request0.json -> request00.wav, request01.wav
# -> request1.json -> request10.wav, request11.wav
./build/dit-vae \
    --request /tmp/request0.json /tmp/request1.json \
    --text-encoder models/Qwen3-Embedding-0.6B-Q8_0.gguf \
    --dit models/acestep-v15-turbo-Q8_0.gguf \
    --vae models/vae-BF16.gguf \
    --batch 2
```

The LM decides song structure (lyrics, melody, rhythm via audio codes), so
LM batch variations produce genuinely different songs. DiT batch variations
only differ by initial noise, producing subtle variations of the same piece
(slightly different timbres, minor rhythmic shifts). Use LM batching for
diversity, DiT batching for cherry-picking the best render.

Transform an existing song with `--src-audio` (no LLM needed):

```bash
cat > /tmp/cover.json << 'EOF'
{
    "caption": "Jazz piano cover with brushed drums and walking bass",
    "lyrics": "[Instrumental]",
    "audio_cover_strength": 0.5
}
EOF

./build/dit-vae \
    --src-audio song.wav
    --request /tmp/cover.json \
    --text-encoder models/Qwen3-Embedding-0.6B-Q8_0.gguf \
    --dit models/acestep-v15-turbo-Q8_0.gguf \
    --vae models/vae-BF16.gguf \
```

Ready-made examples in `examples/`:

```bash
cd examples
./simple.sh           # caption only, LLM fills everything
./partial.sh          # caption + lyrics + duration
./full.sh             # all metadata provided
./dit-only.sh         # skip LLM, DiT from noise
```

Each example has a `-sft` variant (SFT model, 50 steps, CFG 7.0)
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
With `--batch N`, each element runs its own phase 1 from a different seed,
producing N completely different songs. See `examples/simple.json`.

**Caption + lyrics (+ optional metadata)**: single LLM pass. The "Generate
tokens" prompt is used directly. Missing metadata is filled via CoT, the
caption is enriched, and audio codes are generated. User-provided metadata
fields are never overwritten. `lm_cfg_scale` applies to both CoT and code
generation. See `examples/partial.json`.

**Everything provided** (caption, lyrics, bpm, duration, keyscale,
timesignature): the LLM skips CoT and generates audio codes directly.
With `--batch N`, all elements share the same prompt (single prefill,
KV cache copied). See `examples/full.json`.

**Instrumental** (`lyrics="[Instrumental]"`): treated as "lyrics provided",
so the single-pass "Generate tokens" path is used. No lyrics generation.
The DiT was trained with this exact string as the no-vocal condition.

**Passthrough** (`audio_codes` present): LLM is skipped entirely.
Run `dit-vae` to decode existing codes. See `examples/dit-only.json`.

**Reference audio** (`--src-audio` on CLI): no LLM needed. The source audio
(WAV or MP3, any sample rate) is resampled to 48kHz, VAE-encoded to latent
space and used as DiT context instead of silence.
`audio_cover_strength` in the JSON controls how many DiT steps see the source
(0.5 = half the steps use source context, half use silence). The caption
steers the style while the source provides structure, melody, and rhythm.
Duration is determined by the source audio.

**Repaint** (`--src-audio` + `repainting_start`/`repainting_end` in JSON):
regenerates a time region of the source audio while preserving the rest.
Requires the **SFT model** (the turbo model is less performant for this task).
The DiT receives a binary mask: 1.0 inside the region (generate), 0.0 outside
(keep original). Source latents outside the region provide context; silence
fills the repaint zone. Both fields default to -1
(inactive). Set one or both to activate: -1 on start means 0s, -1 on end means
source duration. `audio_cover_strength` is ignored in repaint mode (the mask
handles everything).

```bash
cat > /tmp/repaint.json << 'EOF'
{
    "caption": "Smooth jazz guitar solo with reverb",
    "lyrics": "[Instrumental]",
    "repainting_start": 10.0,
    "repainting_end": 25.0,
    "inference_steps": 50,
    "guidance_scale": 7.0,
    "shift": 1.0
}
EOF

./build/dit-vae \
    --src-audio song.wav \
    --request /tmp/repaint.json \
    --text-encoder models/Qwen3-Embedding-0.6B-Q8_0.gguf \
    --dit models/acestep-v15-sft-Q8_0.gguf \
    --vae models/vae-BF16.gguf
```

**Lego** (`--lego <track>` + `--src-audio`):
generates a new instrument track layered over an existing backing track.
Only the **base model** (`acestep-v15-base`) supports lego mode.
The track name is passed on the CLI; set `audio_cover_strength=1.0` in the
request so the source audio guides all DiT steps.
See `examples/lego.json` and `examples/lego.sh`.

Available track names: `vocals`, `backing_vocals`, `drums`, `bass`, `guitar`,
`keyboard`, `percussion`, `strings`, `synth`, `fx`, `brass`, `woodwinds`.

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
    "lm_temperature":       0.85,
    "lm_cfg_scale":         2.0,
    "lm_top_p":             0.9,
    "lm_top_k":             0,
    "lm_negative_prompt":   "",
    "use_cot_caption":      true,
    "audio_codes":          "",
    "inference_steps":      8,
    "guidance_scale":       0.0,
    "shift":                3.0,
    "audio_cover_strength": 0.5,
    "repainting_start":    -1,
    "repainting_end":      -1
}
```

### Text conditioning (ace-qwen3 + dit-vae)

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
Target audio duration. `0` means the LLM picks it. Clamped to [1, 600]s after
generation. `1` means 1 second.

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
RNG seed. Resolved once at startup to a random value if -1. Batch elements
use `seed+0`, `seed+1`, ... `seed+N-1`.

**`audio_codes`** (string, default `""`)
Comma-separated FSQ token IDs produced by ace-qwen3. When non-empty, the
entire LLM pass is skipped and dit-vae decodes these codes directly
(passthrough mode).

**`audio_cover_strength`** (float, default `0.5`)
Only used with `--src-audio`. Fraction of DiT steps that see the source audio
as context. At `1.0` all steps use the source (near passthrough). At `0.0` no
steps use the source (pure text2music, source is ignored). At `0.5` the first
half of the steps are guided by the source structure, the second half are free
to follow the caption. Lower values give more creative freedom, higher values
preserve more of the original.

**`repainting_start`** (float seconds, default `-1` = inactive)
**`repainting_end`** (float seconds, default `-1` = inactive)
Only used with `--src-audio`. When one or both are >= 0, repaint mode activates:
the DiT regenerates the `[start, end)` time region while preserving everything
else. `-1` on start means 0s (beginning), `-1` on end means source duration
(end). Error if end <= start after resolve. `audio_cover_strength` is ignored.

### LM sampling (ace-qwen3)

**`lm_temperature`** (float, default `0.85`)
Sampling temperature for both phase 1 (lyrics/metadata) and phase 2 (audio
codes). Lower = more deterministic.

**`lm_cfg_scale`** (float, default `2.0`)
Classifier-Free Guidance scale for the LM. Only active in phase 2 (audio
code generation) and in phase 1 when lyrics are already provided. When
`lyrics` is empty, phase 1 always runs with `cfg=1.0` (free sampling).
`1.0` disables CFG.

**`lm_top_p`** (float, default `0.9`)
Nucleus sampling cutoff. `1.0` disables. When `top_k=0`, an internal
pre-filter of 256 tokens is applied before top_p for performance.

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

### DiT flow matching (dit-vae)

**`inference_steps`** (int, default `8`)
Number of diffusion denoising steps. Turbo preset: `8`. SFT preset: `50`.

**`guidance_scale`** (float, default `0.0` = auto)
CFG scale for the DiT. `0.0` is resolved at runtime:
- Turbo models: forced to `1.0` (CFG disabled, turbo was trained without it).
- SFT/base models: `7.0`.
Any value > 1.0 on a turbo model is overridden to 1.0 with a warning.

**`shift`** (float, default `3.0`)
Flow-matching schedule shift. Controls the timestep distribution.
`shift = s*t / (1 + (s-1)*t)`. Turbo preset: `3.0`. SFT preset: `6.0`.

Turbo preset: `inference_steps=8, shift=3.0` (guidance_scale auto-resolved to 1.0).
SFT preset: `inference_steps=50, guidance_scale=7.0, shift=6.0`.

## ace-qwen3 reference

```
Usage: ace-qwen3 --request <json> --model <gguf> [options]

Required:
  --request <json>       Input request JSON
  --model <gguf>         Model GGUF file

Batch:
  --batch <N>            Batch N sequences (default: 1)

Output naming: input.json -> input0.json, input1.json, ... (last digit = batch index)

Debug:
  --max-seq <N>          KV cache size (default: 8192)
  --no-fsm               Disable FSM constrained decoding
  --no-fa                Disable flash attention
  --dump-logits <path>   Dump prefill logits (binary f32)
  --dump-tokens <path>   Dump prompt token IDs (CSV)
```

Three LLM sizes: 0.6B (fast), 1.7B, 4B (best quality).

Batching is always active (default N=1). Model weights are read once per
decode step for all N sequences. Phase 1 (CoT) and Phase 2 (audio codes)
are both batched with independent seeds (seed+0 .. seed+N-1).

## dit-vae reference

```
Usage: dit-vae --request <json...> --text-encoder <gguf> --dit <gguf> --vae <gguf> [options]

Required:
  --request <json...>     One or more request JSONs (from ace-qwen3 --request)
  --text-encoder <gguf>   Text encoder GGUF file
  --dit <gguf>            DiT GGUF file
  --vae <gguf>            VAE GGUF file

Reference audio:
  --src-audio <file>      Source audio (WAV or MP3, any sample rate)

LoRA:
  --lora <path>           LoRA safetensors file or directory
  --lora-scale <float>    LoRA scaling factor (default: 1.0)

Batch:
  --batch <N>             DiT variations per request (default: 1, max 9)

Output:
  Default: MP3 at 128 kbps. input.json -> input0.mp3, input1.mp3, ...
  --mp3-bitrate <kbps>    MP3 bitrate (default: 128)
  --wav                   Output WAV instead of MP3

VAE tiling (memory control):
  --vae-chunk <N>         Latent frames per tile (default: 256)
  --vae-overlap <N>       Overlap frames per side (default: 64)

Debug:
  --no-fa                 Disable flash attention
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

When `--src-audio` is provided, the source audio (WAV or MP3, any sample rate)
is resampled to 48kHz, VAE-encoded once and injected as DiT context for every
request. `audio_cover_strength` in the JSON controls how many steps use the
source (default 0.5).

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
neural-codec --vae models/vae-BF16.gguf --encode --q4 -i song.wav -o song.nac4

# encode (Q8: 13 kbit/s, ~290 KB for 3 minutes)
neural-codec --vae models/vae-BF16.gguf --encode --q8 -i song.wav -o song.nac8

# decode (auto-detects format)
neural-codec --vae models/vae-BF16.gguf --decode -i song.nac4 -o song_decoded.wav

# roundtrip validation: compare song.wav and song_decoded.wav with your ears
```

## mp3-codec reference

Standalone MIT-licensed MPEG1 Layer III encoder and decoder. No external
dependencies. The encoder is used by `dit-vae` for MP3 output. The decoder
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

## Architecture

```
ace-qwen3 (Qwen3 causal LM, 0.6B/1.7B/4B)
  Phase 1 (if needed): CoT generates bpm, keyscale, timesignature, lyrics
  Phase 2: audio codes (5Hz tokens, FSQ vocabulary)
  Both phases batched: N sequences per forward, weights read once
  CFG with dual KV cache per batch element (cond + uncond)
  Output: request0.json .. requestN-1.json

dit-vae
  BPE tokenize
  Qwen3-Embedding (28L text encoder)
  CondEncoder (lyric 8L + timbre 4L + text_proj)
  FSQ detokenizer (audio codes -> flow matching source latents)
  LoRA merge (optional: safetensors delta -> dequant/merge/requant at load)
  DiT (24L flow matching, Euler steps)
  VAE (AutoencoderOobleck, tiled decode)
  WAV stereo 48kHz
```

## Roadmap

Started as "can GGML even sing?". It can. Now make it do more.

- [ ] **Understand mode**: audio codes -> metadata + lyrics (reverse of generation)
- [x] **LoRA**: adapter loading for fine-tuned DiT models
- [ ] **JSON HTTP server**: minimal API, stable contract
- [ ] **Audio I/O**: built-in MP3 encode/decode if a clean MIT library exists, otherwise ffmpeg does the job
- [ ] **Documentation split**: README (user guide) + ARCHITECTURE.md (internals) when a UI exists
- [ ] **ACE-Step 2.0**: evaluate architecture delta, add headers/weights as needed

### Community UIs

Third-party interfaces (under active development, waiting for API and codec to stabilize):

- [acestep-cpp-ui](https://github.com/audiohacking/acestep-cpp-ui)
- [acestep.cpp-simple-GUI](https://github.com/Nurb4000/acestep.cpp-simple-GUI)
- [aceradio](https://github.com/IMbackK/aceradio)

## LM specifics

ace-qwen3 is not a general-purpose chat engine. It is a two-phase autoregressive
pipeline specialized for ACE-Step music generation.

Phase 1 (CoT) generates structured metadata (bpm, keyscale, timesignature, caption,
duration, language) and optionally lyrics via chain-of-thought reasoning. An FSM
(finite state machine) built from a prefix tree enforces valid field names and values
at every decode step, hard-masking invalid tokens before sampling.

Phase 2 (audio codes) generates 5Hz FSQ tokens from a 65535-code vocabulary appended
to the base Qwen3 tokenizer. A partial LM head projects only the audio code subrange
of the embedding matrix, cutting the output GEMM by 70% compared to full-vocab
projection. Classifier-free guidance (CFG) is fused into the batch dimension: N
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
python3 debug-lm-logits.py        # Qwen3 LM: first-token logits GGML vs PyTorch (0.6B/1.7B/4B)
python3 debug-detok-cossim.py     # FSQ detokenizer: step-by-step cossim C++ vs Python
python3 debug-dit-cossim.py       # DiT: per-layer cossim GGML vs Python (turbo/SFT, BF16/quantized)
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

## Acknowledgements

Independent implementation based on ACE-Step 1.5 by ACE Studio and StepFun.
All model weights are theirs, this is just a native backend.

```bibtex
@misc{gong2026acestep,
	title={ACE-Step 1.5: Pushing the Boundaries of Open-Source Music Generation},
	author={Junmin Gong, Yulin Song, Wenxiao Zhao, Sen Wang, Shengyuan Xu, Jing Guo},
	howpublished={\url{https://github.com/ace-step/ACE-Step-1.5}},
	year={2026},
	note={GitHub repository}
}
```

## Samples

https://github.com/user-attachments/assets/9a50c1f4-9ec0-474a-bd14-e8c6b00622a1

https://github.com/user-attachments/assets/fb606249-0269-4153-b651-bf78e05baf22

https://github.com/user-attachments/assets/e0580468-5e33-4a1f-a0f4-b914e4b9a8c2

https://github.com/user-attachments/assets/292a31f1-f97e-4060-9207-ed8364d9a794

https://github.com/user-attachments/assets/34b1b781-a5bc-46c4-90a6-615a10bc2c6a
