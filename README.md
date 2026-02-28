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
apt install pkg-config libopenblas-dev  # Debian/Ubuntu
cmake .. -DGGML_BLAS=ON

# Combine as needed
cmake .. -DGGML_CUDA=ON -DGGML_BLAS=ON

cmake --build . --config Release -j$(nproc)
```

Builds two binaries: `ace-qwen3` (LLM) and `dit-vae` (DiT + VAE).

**CI (GitHub Actions)**  
- **Build**: on every push/PR, builds on Ubuntu (BLAS) and macOS (Metal); smoke test runs each binary `--help`.  
- **Test generation**: on manual trigger or push to `main`; builds, caches models, then runs short (5 s, 4 steps) generation for text2music, cover, and full pipeline (LLM → DiT → WAV). See `.github/workflows/`.

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

# LLM: request.json -> request0.json (enriched with lyrics + codes)
./build/ace-qwen3 \
    --request /tmp/request.json \
    --model models/acestep-5Hz-lm-4B-BF16.gguf

# DiT+VAE: request0.json -> request00.wav
./build/dit-vae \
    --request /tmp/request0.json \
    --text-encoder models/Qwen3-Embedding-0.6B-BF16.gguf \
    --dit models/acestep-v15-turbo-BF16.gguf \
    --vae models/vae-BF16.gguf
```

Generate multiple songs at once with `--batch`:

```bash
# LLM: 2 LM variations x 2 DiT variations = 4 WAVs total
# -> request0.json, request1.json (different lyrics/codes, seeds auto+0, auto+1)
./build/ace-qwen3 \
    --request /tmp/request.json \
    --model models/acestep-5Hz-lm-4B-BF16.gguf \
    --batch 2

# DiT+VAE: (2 DiT variations of LM output 1 and 2)
# -> request0.json -> request00.wav, request01.wav
# -> request1.json -> request10.wav, request11.wav
./build/dit-vae \
    --request /tmp/request0.json /tmp/request1.json \
    --text-encoder models/Qwen3-Embedding-0.6B-BF16.gguf \
    --dit models/acestep-v15-turbo-BF16.gguf \
    --vae models/vae-BF16.gguf \
    --batch 2
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
./test-reference.sh   # reference_audio (WAV or MP3) + audio_cover_strength
```

Each example has a `-sft` variant (SFT model, 50 steps, CFG 7.0)
alongside the turbo default (8 steps, no CFG). For **reference timbre**, set `reference_audio` to a **WAV or MP3** path; dit-vae loads it (MP3 decoded in memory via header-only minimp3, no temp files), encodes with the VAE encoder (requires a full VAE GGUF that includes encoder weights).

## Generation modes

The LLM fills what's missing in the JSON and generates audio codes.
Empty field = "fill it". Filled = "don't touch".
All modes always output numbered files (`request0.json` .. `requestN-1.json`).
The input JSON is never modified.

**Caption only**: the LLM generates lyrics, metadata (bpm, key, time
signature, duration) and audio codes. With `--batch N`, each element
generates its own lyrics and metadata from a different seed, producing
N completely different songs. See `examples/simple.json`.

**Caption + lyrics (+ optional metadata)**: the LLM fills missing
metadata via CoT, then generates audio codes. User provided fields
are preserved. See `examples/partial.json`.

**Everything provided** (caption, lyrics, bpm, duration, keyscale,
timesignature): the LLM skips CoT and generates audio codes directly.
With `--batch N`, all elements share the same prompt (single prefill,
KV cache copied). See `examples/full.json`.

**Passthrough** (`audio_codes` present): LLM is skipped entirely.
Run `dit-vae` to decode existing codes. See `examples/dit-only.json`.

## Request JSON reference

All fields with defaults. Only `caption` is required. Built-in modes (text2music, cover, repaint) and audio inputs follow the [ACE-Step 1.5 Tutorial](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/Tutorial.md); see [docs/MODES.md](docs/MODES.md) for what is implemented.

```json
{
    "task_type":          "text2music",
    "caption":            "",
    "lyrics":             "",
    "instrumental":       false,
    "bpm":                0,
    "duration":           -1,
    "keyscale":           "",
    "timesignature":      "",
    "vocal_language":     "unknown",
    "seed":               -1,
    "lm_temperature":     0.85,
    "lm_cfg_scale":       2.0,
    "lm_top_p":           0.9,
    "lm_top_k":           0,
    "lm_negative_prompt": "",
    "reference_audio":    "",
    "src_audio":          "",
    "audio_codes":        "",
    "audio_cover_strength": 1.0,
    "repainting_start":   0.0,
    "repainting_end":     0.0,
    "inference_steps":    8,
    "guidance_scale":     7.0,
    "shift":              3.0
}
```

Key fields: `seed` -1 means random (resolved once, then +1 per batch
element). `audio_codes` is generated by ace-qwen3 and consumed by
dit-vae (comma separated FSQ token IDs). When present, the LLM is
skipped entirely (cover-style generation). `reference_audio`: path to a **WAV or MP3** file for global timbre/style (MP3 decoded in memory; encoded via built-in VAE encoder; requires VAE GGUF with encoder weights). `src_audio`: not yet implemented (see docs/MODES.md).

Turbo preset: `inference_steps=8, shift=3.0` (no guidance_scale, turbo models don't use CFG).
SFT preset: `inference_steps=50, guidance_scale=4.0, shift=6.0`.

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

Batch:
  --batch <N>             DiT variations per request (default: 1, max 9)

Output naming: input.json -> input0.wav, input1.wav, ... (last digit = batch index)

VAE tiling (memory control):
  --vae-chunk <N>         Latent frames per tile (default: 256)
  --vae-overlap <N>       Overlap frames per side (default: 64)

Debug:
  --dump <dir>            Dump intermediate tensors
```

Models are loaded once and reused across all requests.

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
  DiT (24L flow matching, Euler steps)
  VAE (AutoencoderOobleck, tiled decode)
  WAV stereo 48kHz
```

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
prefills. Embedding lookup bypasses ggml_get_rows entirely: rows are read directly
from the mmap'd GGUF file on CPU, dequantized, and uploaded as F32 input tensors.
Decode uses a dedicated single-backend graph allocator (gallocr) with no scheduler
dispatch overhead, while prefill uses the multi-backend scheduler for flexibility.

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

Uses a patched GGML fork (submodule) with ops added for the Oobleck VAE decoder.

The VAE reconstructs audio from latent space through 5 upsampling blocks (total 1920x),
each running a transposed convolution followed by 3 WaveNet-style residual units with
dilated convolutions and Snake activations. A single tile builds a graph of 36 snake
activations, 5 transposed convolutions, and 32 regular convolutions. At the final blocks,
sequence lengths reach 491520 timesteps, which stresses GGML ops designed for short NLP sequences.
The DiT (flow matching diffusion transformer) uses only standard GGML ops and needs no patches.

Patches on top of upstream GGML, oldest first:

| Commit | Scope | Description |
|--------|-------|-------------|
| `8c70db84` | CUDA | `conv_transpose_1d`: replace O(T_in) brute-force loop with bounded range |
| `b65bf458` | CUDA | `im2col`: grid-stride loop on OW to fix gridDim.y overflow when T > 65535 |
| `e0e36f3c` | Metal | `conv_transpose_1d`: same bounded loop fix as CUDA |
| `2b9080bd` | CPU, CUDA, Metal | New `GGML_OP_COL2IM_1D`: scatter-add for GEMM-based conv_transpose_1d decomposition |
| `02c8041f` | CPU, CUDA, Metal | New `GGML_OP_SNAKE`: fused activation y = x + sin^2(a*x) / b (replaces 5 element-wise ops) |
| `3f60b19c` | Metal | Fix snake kernel to use current C wrapper API |
| `cb5d7067` | Vulkan | Guard `VK_EXT_layer_settings` for legacy Vulkan SDK (fixes MI50/gfx906) |
| `1f0f4214` | Vulkan | `col2im_1d`: add Vulkan backend |
| `efbf3df6` | Vulkan | `snake`: add Vulkan backend |
| `6608cd11` | Vulkan | Fix rvalue ref for `col2im_1d` and `snake` push constants |
| `06101d38` | Vulkan | Fix double-division dispatch for `col2im_1d` and `snake` |
| `91416cee` | CPU, CUDA, Metal, Vulkan | `col2im_1d`: fuse padding crop via p0 parameter (saves 5 allocs + 5 memcpy per VAE tile) |
| `20675b09` | Vulkan | `col2im_1d`, `snake`: 2D dispatch (fixes workgroup overflow on MI50) |

**Why col2im_1d**: upstream `ggml_conv_transpose_1d` uses a naive CUDA kernel (one scalar
FMA loop per output element, no shared memory, no tensor cores). The VAE spends 40% of its
FLOP budget on transposed convolutions. We decompose it as `mul_mat + col2im_1d`, routing
the heavy GEMM through cuBLAS/BLAS/MPS tensor cores. The col2im_1d gather has a 2-iteration
inner loop and is pure bandwidth.

**Why snake**: the Oobleck VAE uses Snake1d activation (x + sin^2(a*x) / b) 36 times per
tile. Without a fused op, each activation requires 5 separate GGML kernels (mul, sin, sqr,
mul, add), causing 5x the memory traffic. The fused kernel reads x once, writes y once.

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
