# acestep.cpp

Portable C++17 implementation of ACE-Step 1.5 music generation using GGML.
Text + lyrics in, stereo 48kHz WAV out. Runs on CPU, CUDA, Metal, Vulkan.

## Build

```bash
git submodule update --init

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

# LLM: request.json -> request0.json (enriched with lyrics + codes)
./build/ace-qwen3 \
    --request /tmp/request.json \
    --model models/acestep-5Hz-lm-4B-BF16.gguf

# DiT+VAE: request0.json -> output.wav
./build/dit-vae \
    --request /tmp/request0.json \
    --text-encoder models/Qwen3-Embedding-0.6B-BF16.gguf \
    --dit models/acestep-v15-turbo-BF16.gguf \
    --vae models/vae-BF16.gguf
# -> request00.wav
```

Generate multiple songs at once with `--batch`:

```bash
# 2 LM variations x 2 DiT variations = 4 WAVs total
./build/ace-qwen3 \
    --request /tmp/request.json \
    --model models/acestep-5Hz-lm-4B-BF16.gguf \
    --batch 2
# -> request0.json, request1.json (different lyrics/codes, seeds auto+0, auto+1)

./build/dit-vae \
    --request /tmp/request0.json /tmp/request1.json \
    --text-encoder models/Qwen3-Embedding-0.6B-BF16.gguf \
    --dit models/acestep-v15-turbo-BF16.gguf \
    --vae models/vae-BF16.gguf \
    --batch 2
# -> request00.wav, request01.wav  (2 DiT variations of LM output 0)
#    request10.wav, request11.wav  (2 DiT variations of LM output 1)
```

The LM decides song structure (lyrics, melody, rhythm via audio codes), so
LM batch variations produce genuinely different songs. DiT batch variations
only differ by initial noise, producing subtle variations of the same piece
(slightly different timbres, minor rhythmic shifts). Use LM batching for
diversity, DiT batching for cherry-picking the best render.

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

The LLM behavior depends on which fields are present in the JSON.
All modes always output numbered files (`request0.json` .. `requestN-1.json`).
The input JSON is never modified.

**Simple** (caption only): the LLM generates all metadata (bpm, key,
time signature, duration, lyrics) via chain-of-thought, then produces
audio codes. With `--batch N`, each element generates its own lyrics
and metadata from a different seed, producing N completely different
songs. See `examples/simple.json`.

**Partial** (caption + some metadata): the LLM fills missing fields
via CoT with classifier-free guidance, then generates audio codes.
Provide any combination of lyrics, duration, bpm, keyscale, timesignature.
With `--batch N`, each element fills missing fields independently.
See `examples/partial.json`.

**Full** (all metadata provided): the LLM skips CoT and generates
audio codes directly. Requires caption, lyrics, bpm, duration, keyscale,
and timesignature. With `--batch N`, all elements share the same prompt
(single prefill, KV cache copied) and produce N audio variations of
the same song. See `examples/full.json`.

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

Key fields: `seed` -1 means random (resolved once, then +1 per batch
element). `thinking` false skips CoT (for SFT models or when all metadata
is provided). `audio_codes` is generated by ace-qwen3 and consumed by
dit-vae (comma-separated FSQ token IDs).

Turbo preset: `inference_steps=8, shift=3.0` (no guidance_scale, turbo models don't use CFG).
Base/SFT preset: `inference_steps=32, guidance_scale=7.0, shift=1.0, thinking=false`.

## ace-qwen3 reference

```
Usage: ace-qwen3 [options]

Required:
  --request <json>       Input request JSON (never modified)
  --model <gguf>         Model GGUF file (from convert.py)

Generation:
  --batch <N>            Generate N sequences (default: 1)
                         Output: request0.json .. requestN-1.json

Infra:
  --max-seq <N>          KV cache size (default: 8192)
  --no-fsm               Disable FSM constrained decoding

Debug:
  --dump-logits <path>   Dump prefill logits (binary f32)
  --dump-tokens <path>   Dump prompt token IDs (CSV)
```

Three LLM sizes: 0.6B (fast), 1.7B, 4B (best quality).

Batching is always active (default N=1). Model weights are read once per
decode step for all N sequences. Phase 1 (CoT) and Phase 2 (audio codes)
are both batched with independent seeds (seed+0 .. seed+N-1).

## dit-vae reference

```
Usage: dit-vae [options]

Required:
  --request <json> ...    One or more request JSONs (output from ace-qwen3)
  --text-encoder <gguf>   Text encoder GGUF file
  --dit <gguf>            DiT GGUF file (from convert.py)
  --vae <gguf>            VAE GGUF file

Generation:
  --batch <N>             Generate N noise variations per request (default: 1)
                          Output: request0.wav .. requestN-1.wav

Audio:
  --noise-file <path>     Load noise from BF16 file (Philox RNG dump)

VAE tiling (memory control):
  --vae-chunk <n>         Latent frames per tile (default: 256)
  --vae-overlap <n>       Overlap frames per side (default: 64)

Debug:
  --dump <dir>            Dump intermediate tensors
```

Output naming is automatic: `input.json` with `--batch 2` produces
`input0.wav`, `input1.wav`. Models are loaded once and reused across
all requests.

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
  FSQ detokenizer (audio codes -> source latents)
  DiT (24L flow matching, Euler steps)
  VAE (AutoencoderOobleck, tiled decode)
  WAV stereo 48kHz
```

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
