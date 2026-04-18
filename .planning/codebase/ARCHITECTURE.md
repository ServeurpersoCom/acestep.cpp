# Architecture

**Analysis Date:** 2026-04-18

## Pattern Overview

**Overall:** Multi-pipeline inference server — three independent but composable GGML inference pipelines wrapped by an HTTP server with a FIFO job queue and an embedded browser UI.

**Key Characteristics:**
- All model weights loaded from GGUF files via mmap; each pipeline is a self-contained opaque handle (`AceLm*`, `AceSynth*`, `AceUnderstand*`)
- GPU access is serialized by a single worker thread in the server; no per-pipeline mutexes needed
- Models are loaded on-demand per request and freed after (unless `--keep-loaded`) — zero VRAM at idle
- All pipeline operations are cancellable via a callback polled between tokens/steps

---

## System Purpose

`acestep.cpp` is a portable C++17 AI music generation engine. It implements the ACE-Step 1.5 model architecture using GGML as the compute backend. Input: a text caption + optional lyrics/metadata/source audio. Output: stereo 48 kHz audio (MP3 or WAV). Runs on CPU, CUDA, ROCm, Metal, and Vulkan.

---

## Three Pipelines

### Pipeline 1: LM (`src/pipeline-lm.h`, `src/pipeline-lm.cpp`)

**Purpose:** Text caption → enriched metadata + lyrics + audio codes (5 Hz discrete tokens)

- Loads a Qwen3 causal LM (`AceLm`) with KV cache
- Runs Chain-of-Thought (CoT) generation; outputs YAML-structured metadata fields
- FSM-constrained decoding (`src/metadata-fsm.h`) for metadata fields (BPM, key, duration, etc.)
- Produces `audio_codes` string (comma-separated FSQ integers, e.g. `"3101,11837,..."`)
- Modes: `LM_MODE_GENERATE` (full), `LM_MODE_INSPIRE` (no codes), `LM_MODE_FORMAT` (reformat only)
- Supports batched CFG (cond+uncond in one forward pass) and batch size 1–9
- Internal model shared with Understand pipeline to avoid double-loading

### Pipeline 2: Synth (`src/pipeline-synth.h`, `src/pipeline-synth.cpp`, `src/pipeline-synth-ops.cpp`)

**Purpose:** Enriched `AceRequest` → stereo 48 kHz audio

Loads four models into `AceSynth`:
- `DiTGGML` — 24-layer Diffusion Transformer (weights from DiT GGUF)
- `Qwen3GGML` (text encoder, 28L) — encodes caption text
- `CondGGML` — condition encoder: lyric encoder (8L) + timbre encoder (4L) + text projection
- `VAEGGML` / `VAEEncoder` — Oobleck VAE decoder/encoder for latent ↔ audio conversion
- `DetokGGML` / `TokGGML` — FSQ tokenizer and detokenizer (weights in DiT GGUF)
- `BPETokenizer` — Qwen3 byte-level BPE tokenizer for text

**Execution flow (ops_ primitives in `src/pipeline-synth-ops.h`):**

1. `ops_encode_src` — VAE-encode source audio to latents (cover/repaint/lego tasks)
2. `ops_resolve_params` — resolve DiT steps, guidance scale, shift; scan audio codes
3. `ops_build_schedule` — build flow-matching timestep schedule
4. `ops_resolve_T` — compute latent frame count T (25 Hz) and patch count S
5. `ops_encode_timbre` — VAE-encode reference audio → timbre features
6. `ops_encode_text` — BPE tokenize → Qwen3 text encoder → cond encoder; produces `enc_hidden` per batch item
7. `ops_build_context` — assemble DiT context tensor `[batch, T, ctx_ch]` (source latents | mask)
8. `ops_init_noise_and_repaint` — Philox PRNG noise init (matches PyTorch cuRAND); optional cover noise blend
9. `ops_dit_generate` — denoising loop: N Euler steps (8 turbo, 50 base/sft) with APG-CFG guidance
10. `ops_vae_decode_and_splice` — VAE decode latents → audio; crossfade splice for repaint regions

### Pipeline 3: Understand (`src/pipeline-understand.h`, `src/pipeline-understand.cpp`)

**Purpose:** Reverse pipeline — audio → metadata + lyrics + codes

- Audio in → VAE encode → FSQ tokenize → prepend codes as context → LM generates metadata + lyrics
- Can share LM weights from `AceLm` (no double-load when server has LM already loaded)
- Codes-only mode: skip VAE/FSQ, take `audio_codes` directly from request

---

## Task Types

Defined in `src/task-types.h`. Each task routes differently through the synth pipeline's mode block:

| Task | Key Behavior |
|------|-------------|
| `text2music` | Silence context, pure diffusion from noise |
| `cover` | VAE-encode source → FSQ roundtrip → DiT guided by degraded latents |
| `cover-nofsq` | VAE-encode source, no FSQ degradation (stays close to source structure) |
| `repaint` | Region masking: only regenerate frames within `[repainting_start, repainting_end]` |
| `lego` | Generate/add one instrument track given full mix context |
| `extract` | Separate one instrument track from a mix |
| `complete` | Complete/extend an isolated stem |

---

## Key Data Structures

### `AceRequest` (`src/request.h`)
Pure data container, no business logic. JSON-serializable. Central transfer object between all pipelines and the HTTP server. Fields include: `caption`, `lyrics`, `bpm`, `duration`, `keyscale`, `audio_codes`, `seed`, DiT control params (`inference_steps`, `guidance_scale`, `shift`), task type, repaint region, cover strength.

### `AceSynth` (`src/pipeline-synth-impl.h`)
Internal context holding all loaded synth models (`dit`, `text_enc`, `cond_enc`, `vae`, `detok`, `tok`, `bpe`). Opaque to callers of `pipeline-synth.h`.

### `SynthState` (`src/pipeline-synth-ops.h`)
Per-call working state passed by reference through all `ops_*` functions. Holds intermediate tensors, mode flags, resolved params, text encodings, DiT context, noise buffer, and debug state. Lives on stack for one `ace_synth_generate` call.

### `DiTGGML` (`src/dit.h`)
24-layer DiT weights. Per-layer: self-attention (RoPE + GQA, optional QKV fusion), cross-attention (Q from audio, KV from text encoder), AdaLN scale-shift table, SwiGLU MLP. Alternating sliding-window and full-attention layers. Weights are pre-permuted at load time to eliminate runtime transposes.

### `Qwen3LM` (`src/qwen3-lm.h`)
Causal LM with KV cache. 4D batched KV tensors for batched CFG flash attention. Supports partial lm_head copy to work around ROCm `ggml_view_2d` limitations.

### `VAEGGML` / `VAEEncoder` (`src/vae.h`, `src/vae-enc.h`)
Oobleck convolutional VAE. Decoder: 1920× upsample (5 blocks of snake + ConvTranspose + 3 ResUnits). Encoder: 1920× downsample mirror. Weight norm fused at load time. Tiled decode with configurable chunk size and overlap for VRAM control.

### `GGUFModel` (`src/gguf-weights.h`)
mmap-based GGUF file accessor. All weight loading uses memory-mapped reads; data is copied to GPU backend on `wctx_alloc`.

### `ModelRegistry` (`src/model-registry.h`)
Scans `models/` directory at server startup by reading GGUF headers only. Classifies by `general.architecture` KV into four buckets: `lm`, `dit`, `text_enc`, `vae`. Also discovers LoRA adapters (`.safetensors` ComfyUI or PEFT directory format).

---

## Inference Pipeline: Full text2music Flow

```
User input (caption, lyrics, metadata)
         │
         ▼
   [AceLm / ace-lm]
   BPE tokenize → Qwen3 causal LM (CoT) → FSM-constrained decode
   → AceRequest enriched with: bpm, duration, keyscale, timesig, audio_codes
         │
         ▼
   [AceSynth / ace-synth]
   ├─ BPE tokenize caption
   ├─ Qwen3 text encoder (28L) → caption hidden states
   ├─ CondEncoder:
   │    ├─ lyric encoder (8L) → lyric embeddings
   │    ├─ timbre encoder (4L, optional ref audio)
   │    └─ project text + cat → encoder_hidden_states [S_total, 2048]
   ├─ FSQ detokenize audio_codes → context latents [T_25Hz, 64]
   ├─ Philox noise init [batch, T, 64]
   └─ DiT denoising loop (8 or 50 Euler steps):
        each step: timestep embed → 24 layers (self-attn + cross-attn + MLP + AdaLN)
        APG-CFG: cond and uncond pass, adaptive projected guidance
         │
         ▼
   VAE decoder (Oobleck, tiled)
   latents [T_25Hz, 64] → audio [stereo, 48000 Hz]
         │
         ▼
   MP3 encode (mp3enc) or WAV write
```

---

## HTTP Server Architecture (`tools/ace-server.cpp`)

**Single-binary, single-port HTTP server using cpp-httplib.**

### Request lifecycle
1. Client POSTs to `/lm`, `/synth`, or `/understand`
2. Server validates, creates a `Job` (random 64-bit hex ID), pushes closure to `g_work_queue`
3. Returns `{"job_id": "..."}` immediately
4. Single worker thread dequeues and executes jobs in FIFO order
5. Client polls `GET /job?id=N` for status (`running`/`done`/`failed`/`cancelled`)
6. Client fetches result with `GET /job?id=N&result=1`
7. `POST /job?id=N&cancel=1` sets per-job atomic cancel flag

### Endpoints
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/lm` | POST | LM generation: caption → metadata + lyrics + codes |
| `/synth` | POST | Synthesis: codes + metadata → audio (MP3 or WAV) |
| `/understand` | POST | Reverse: audio → metadata + lyrics + codes |
| `/job` | GET/POST | Job status, result fetch, cancel |
| `/health` | GET | `{"status":"ok"}` |
| `/props` | GET | Available models, server config, default params |
| `/logs` | GET | SSE stream of stderr lines (ring buffer, 512 entries) |
| `/` | GET | Embedded WebUI (gzip-compressed HTML, served inline) |

### Model lifecycle in server
- Models NOT preloaded: zero GPU at startup
- `ensure_lm()` / `ensure_synth()` / `ensure_understand()` load on first use, swap when request specifies different model
- Understand pipeline shares LM weights from `AceLm` to avoid double-loading the 4B model
- All GPU access serialized by single worker thread

---

## GGML Backend Abstraction (`src/backend.h`)

All five model components (LM, DiT, CondEncoder, TextEncoder, VAE) use the same `backend_init()` / `backend_release()` pattern:
- `ggml_backend_load_all()` loads all compiled-in backends (CPU, CUDA, Metal, Vulkan)
- `ggml_backend_init_best()` picks the best GPU, CPU as fallback
- Reference-counted shared backend: first init allocates GPU context, subsequent inits reuse it
- CPU thread count set to `hardware_concurrency / 2` (physical cores, avoids HT contention)
- Flash attention auto-disabled on CPU-only (F32 manual attention used instead to avoid FP16 drift)

---

## FSQ / Audio Codec Layer

The system uses Finite Scalar Quantization (FSQ) as an intermediate audio representation at 5 Hz:

- **VAE** (Oobleck): audio (48 kHz) ↔ latents (25 Hz, 64 ch)
- **FSQ tokenizer** (`src/fsq-tok.h`): latents (25 Hz) → integer codes (5 Hz), 6-dim quantization with levels `[8,8,8,5,5,5]` = 8000 codebook size
- **FSQ detokenizer** (`src/fsq-detok.h`): integer codes (5 Hz) → context latents (25 Hz, 64 ch) for DiT conditioning
- LM generates audio codes at 5 Hz; 1 code = 5 frames = 0.2 s at 25 Hz = 0.192 s audio

---

## LoRA Support (`src/lora-merge.h`)

LoRA deltas are merged into base weights at load time, before QKV fusion and GPU upload:
- Supports PEFT directory format (`adapter_model.safetensors` + `adapter_config.json`)
- Supports ComfyUI single `.safetensors` format (alpha baked in)
- Merge: `weight += (alpha / rank) * scale * B @ A`, dequantized row-by-row

---

## Error Handling

**Strategy:** All public API functions return `NULL` (load functions) or `-1` (generate functions) on failure. Error details go to `stderr`. Fatal conditions (missing required tensors, backend init failure) call `exit(1)`.

**Cancellation:** All `generate` functions accept a `bool (*cancel)(void*)` callback polled between tokens (LM) and between DiT steps (Synth). Server passes per-job atomic flag via this mechanism.

---

*Architecture analysis: 2026-04-18*
