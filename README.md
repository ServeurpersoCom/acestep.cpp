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

Builds two binaries: `ace-qwen3` (LLM) and `dit-vae` (DiT + VAE).

## Checkpoints

```bash
pip install gguf
./checkpoints.sh          # core models (turbo + 4B LM)
./checkpoints.sh --all    # all variants (SFT, shift1/3, 0.6B/1.7B LM)
python3 convert.py        # convert all checkpoints to GGUF (models/)
```

`checkpoints.sh` downloads raw HuggingFace checkpoints (safetensors,
config.json, tokenizer files) into `checkpoints/`. These are the source
material for GGUF conversion, not used at runtime.

`convert.py` packs everything into self-contained GGUF files in `models/`.
Each GGUF bundles model weights with all dependent files so that no external
file is needed at runtime:

- BPE tokenizer (from `vocab.json` + `merges.txt`) in LM and text encoder
- silence_latent (from `silence_latent.pt`, transposed to [15000, 64] f32) in DiT
- full config.json as KV metadata in all models

Output models:

| GGUF | Arch | Size |
|------|------|------|
| Qwen3-Embedding-0.6B-BF16.gguf | text encoder (28L, H=1024) | 1.1 GB |
| acestep-5Hz-lm-{0.6B,1.7B,4B}-BF16.gguf | Qwen3 causal LM | 1.3 / 3.5 / 8.0 GB |
| acestep-v15-{turbo,sft,base,...}-BF16.gguf | DiT + CondEncoder (24L, H=2048) | 4.5 GB |
| vae-BF16.gguf | AutoencoderOobleck | 322 MB |

A direct GGUF downloader (skipping the safetensors intermediate) is planned
once `convert.py` supports quantized DiT exports (Q4_K, Q5_K, Q8_0).
VAE will stay in BF16 (small, bandwidth-bound, quality-critical).

## Quick start

Both binaries communicate through a shared JSON request file.
`ace-qwen3` reads the request, enriches it (CoT metadata + audio codes),
and overwrites it. `dit-vae` reads the enriched request and produces audio.

```bash
cat > /tmp/request.json << 'EOF'
{
    "caption": "Upbeat pop rock with driving guitars and catchy hooks",
    "inference_steps": 8,
    "shift": 3.0,
    "vocal_language": "fr"
}
EOF

./build/ace-qwen3 \
    --request /tmp/request.json \
    --model models/acestep-5Hz-lm-4B-BF16.gguf

./build/dit-vae \
    --request /tmp/request.json \
    --text-encoder models/Qwen3-Embedding-0.6B-BF16.gguf \
    --dit models/acestep-v15-turbo-BF16.gguf \
    --vae models/vae-BF16.gguf \
    --output output.wav
```

Ready-made examples in `examples/`:

```bash
cd examples
./simple.sh           # caption only, LLM fills everything
./partial.sh          # caption + lyrics + duration
./full.sh             # all metadata provided
./dit-only.sh         # skip LLM, DiT from noise
./all.sh              # run all examples
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
    "shift":              1.0
}
```

Key fields: `seed` -1 means random. `thinking` false skips CoT (for SFT
models or when all metadata is provided). `audio_codes` is populated by
ace-qwen3 and consumed by dit-vae (comma-separated FSQ token IDs).

Turbo preset: `inference_steps=8, shift=3.0` (no guidance_scale, turbo models don't use CFG).
Base/SFT preset: `inference_steps=32, guidance_scale=7.0, shift=1.0, thinking=false`.

## ace-qwen3 reference

```
Usage: ace-qwen3 [options]

Required:
  --request <json>       Request JSON (read, enriched, overwritten)
  --model <gguf>         Model GGUF file (from convert.py)

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
Usage: dit-vae [options]

Required:
  --request <json>        Request JSON (from ace-qwen3)
  --text-encoder <gguf>   Text encoder GGUF file
  --dit <gguf>            DiT GGUF file (from convert.py)
  --vae <gguf>            VAE GGUF file

Audio:
  --noise-file <path>     Load noise from BF16 file (Philox RNG dump)
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
