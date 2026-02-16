# acestep.cpp

Portable C++17 implementation of ACE-Step 1.5 music generation using GGML.
Text + lyrics in, stereo 48kHz WAV out. Runs on CPU, CUDA, Metal, Vulkan.

## Build

```bash
git submodule update --init
./build.sh
```

`build.sh` auto-detects your platform: Metal + Accelerate on macOS,
CUDA on Linux with nvcc, Vulkan if available, and OpenBLAS when found.

Manual cmake if you prefer:

```bash
mkdir build && cd build

# macOS (Metal + Accelerate BLAS auto-enabled)
cmake ..

# Linux with NVIDIA GPU
cmake .. -DGGML_CUDA=ON

# Linux with Vulkan
cmake .. -DGGML_VULKAN=ON

# CPU with OpenBLAS (recommended for CPU-only machines)
cmake .. -DGGML_BLAS=ON

# Combine as needed
cmake .. -DGGML_CUDA=ON -DGGML_BLAS=ON

cmake --build . --config Release -j$(nproc)
```

BLAS accelerates CPU matrix multiplications. On macOS, Accelerate is
enabled by default. On Linux, install `libopenblas-dev` and pass
`-DGGML_BLAS=ON`.

Builds three binaries: `ace-qwen3` (LLM), `dit-vae` (DiT + VAE),
and `pt2bin` (checkpoint converter).

## Checkpoints

```bash
./checkpoints.sh          # core models (turbo + 4B LM)
./checkpoints.sh --all    # all variants (SFT, shift1/3, 0.6B/1.7B LM)
```

Downloads from HuggingFace:

- Qwen3-Embedding-0.6B (text encoder, 1.1GB)
- acestep-v15-turbo (DiT 24L + CondEncoder, 3GB)
- vae (AutoencoderOobleck, 322MB)
- acestep-5Hz-lm-0.6B / 1.7B / 4B (autoregressive LLM)

The script also converts `silence_latent.pt` to `.bin` for each DiT
model using `build/pt2bin` (must build first).

## Quick start

Both binaries communicate through a shared JSON request file.
`ace-qwen3` reads the request, enriches it (CoT metadata + audio codes),
and overwrites it. `dit-vae` reads the enriched request and produces audio.

```bash
# Write a request
cat > /tmp/request.json << 'EOF'
{
    "caption": "Upbeat pop rock with driving guitars and catchy hooks",
    "inference_steps": 8,
    "guidance_scale": 7.0,
    "shift": 3.0,
    "vocal_language": "fr"
}
EOF

# LLM: enrich metadata + generate audio codes
./build/ace-qwen3 --model checkpoints/acestep-5Hz-lm-4B \
                  --request /tmp/request.json

# DiT + VAE: generate audio
./build/dit-vae --request /tmp/request.json \
                --dit checkpoints/acestep-v15-turbo \
                --text-encoder checkpoints/Qwen3-Embedding-0.6B \
                --vae checkpoints/vae \
                --output output.wav
```

Ready-made examples in `examples/`:

```bash
cd examples && bash simple.sh     # caption only, LLM fills everything
cd examples && bash partial.sh    # caption + lyrics + duration
cd examples && bash full.sh       # all metadata provided
cd examples && bash dit-only.sh   # skip LLM, DiT from noise
cd examples && bash all.sh        # run all examples
```

Each example has a `-sft` variant (SFT model, 50 steps, CFG 7.0)
alongside the turbo default (8 steps, no CFG).

## Generation modes

The LLM behavior depends on which fields are present in the JSON:

**Simple** (caption only): the LLM generates all metadata (bpm, key,
time signature, duration, lyrics) via chain-of-thought, then produces
audio codes. See `examples/simple.json`.

**Partial** (caption + some metadata): the LLM fills missing fields
via CoT, then generates audio codes. Provide any combination of lyrics,
duration, bpm, keyscale, timesignature. See `examples/partial.json`.

**Full** (all metadata provided): the LLM skips CoT and generates
audio codes directly. Requires caption, lyrics, bpm, duration, keyscale,
and timesignature. See `examples/full.json`.

**DiT-only** (skip LLM entirely): provide all metadata in the JSON
and run `dit-vae` alone. Audio is generated from noise without LLM
codes. See `examples/dit-only.json`.

## Request JSON reference

All fields with defaults. Only `caption` is required.

```json
{
    "caption":            "",
    "lyrics":             "",
    "instrumental":       false,
    "bpm":                0,
    "duration":           -1,
    "keyscale":           "",
    "timesignature":      "",
    "vocal_language":     "unknown",
    "task_type":          "text2music",
    "seed":               -1,
    "thinking":           true,
    "lm_temperature":     0.85,
    "lm_cfg_scale":       2.0,
    "lm_top_p":           0.9,
    "lm_negative_prompt": "NO USER INPUT",
    "audio_codes":        "",
    "inference_steps":    8,
    "guidance_scale":     7.0,
    "shift":              3.0
}
```

Key fields: `seed` -1 means random. `thinking` false skips CoT (for SFT
models or when all metadata is provided). `audio_codes` is populated by
ace-qwen3 and consumed by dit-vae (comma-separated FSQ token IDs).

Turbo preset: `inference_steps=8, shift=3.0` (no guidance_scale, turbo models don't use CFG).
SFT preset: `inference_steps=32, guidance_scale=7.0, shift=3.0, thinking=false`.

## ace-qwen3 reference

```
Usage: ace-qwen3 --model <dir> --request <json> [options]

Required:
  --model <dir>          Model directory (safetensors + config.json)
  --request <json>       Request JSON (read, enriched, overwritten)

Infra:
  --max-seq <N>          KV cache size (default: 8192)
  --no-fsm               Disable FSM constrained decoding

Debug:
  --dump-logits <path>   Dump prefill logits (binary f32)
  --dump-tokens <path>   Dump prompt token IDs (CSV)
```

Three LLM sizes: 0.6B (fast), 1.7B, 4B (best quality).

## dit-vae reference

```
Usage: dit-vae --request <json> --dit <dir> --text-encoder <dir> --vae <dir> [options]

Required:
  --request <json>        Request JSON (from ace-qwen3 --request)
  --text-encoder <dir>    Qwen3-Embedding-0.6B directory
  --dit <dir>             DiT model directory (e.g. acestep-v15-turbo)
  --vae <dir>             VAE directory

Audio:
  --noise-file <path>     Load noise from bf16 file (Philox RNG dump)
  --output <path>         Output WAV (default: output.wav)

VAE tiling (memory control):
  --vae-chunk <n>         Latent frames per tile (default: 256)
  --vae-overlap <n>       Overlap frames per side (default: 64)

Debug:
  --dump <dir>            Dump intermediate tensors
```

## Pipeline

```
ace-qwen3 (Qwen3 causal LM, 0.6B/1.7B/4B)
  -> CoT: bpm, keyscale, timesignature
  -> Audio codes (5Hz tokens, FSQ vocabulary)
  -> [optional CFG with dual KV cache]

dit-vae
  -> BPE tokenize
  -> Qwen3-Embedding (28L text encoder)
  -> CondEncoder (lyric 8L + timbre 4L + text_proj)
  -> FSQ detokenizer (audio codes -> source latents, if provided)
  -> DiT (24L flow matching, Euler steps)
  -> VAE (AutoencoderOobleck, tiled decode)
  -> WAV stereo 48kHz
```

## Accuracy

Full test log (turbo + SFT, seed 42, Philox noise):
[`tests/accuracy.log`](https://github.com/ServeurpersoCom/acestep.cpp/blob/master/tests/accuracy.log)

Run `python3 tests/debug-dit-cossim.py` to reproduce.

## Known issues

Uses a patched GGML fork (submodule). Two fixes for long-sequence audio:

- im2col.cu gridDim.y overflow when T > 65535 patches,
- conv_transpose_1d.cu O(T_in) brute-force loop too slow for VAE upsampling.

TODO: verify if these are still needed on latest GGML and submit upstream PRs.

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
