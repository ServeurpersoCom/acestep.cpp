# External Integrations

**Analysis Date:** 2026-04-18

## Model Files

### GGUF Format (primary model format)

All model weights are loaded from `.gguf` files at runtime. GGUF is the only supported inference format; no ONNX, PyTorch `.pt`, or raw safetensors loading occurs at inference time.

**GGUF loader:** custom mmap-based reader in `src/gguf-weights.h` using GGML's `gguf.h` API.

**Four model types classified by `general.architecture` GGUF key:**

| Architecture key | Role | Typical file |
|---|---|---|
| `acestep-lm` | Qwen3 causal LM (0.6B / 1.7B / 4B) | `acestep-5Hz-lm-4B-Q8_0.gguf` |
| `acestep-text-enc` | Qwen3 embedding text encoder | `Qwen3-Embedding-0.6B-Q8_0.gguf` |
| `acestep-dit` | Diffusion Transformer + CondEncoder + FSQ | `acestep-v15-turbo-Q8_0.gguf` |
| `acestep-vae` | AutoencoderOobleck VAE | `vae-BF16.gguf` |

**Quantization types supported by the `quantize` tool:**
`Q2_K`, `Q3_K_S`, `Q3_K_M`, `Q3_K_L`, `Q4_K_S`, `Q4_K_M`, `Q5_K_S`, `Q5_K_M`, `Q6_K`, `Q8_0`, `BF16`

VAE is always kept as `BF16` (never quantized). LM-0.6B/1.7B are `Q8_0` only (too small for aggressive quantization). `Q4_K_M` is explicitly unsupported for the 4B LM (breaks audio code generation). See `quantize.sh`.

### Safetensors Format (LoRA adapters only)

LoRA adapters are loaded from `.safetensors` at runtime via `src/safetensors.h` (minimal custom mmap reader). Two LoRA formats are supported:

- **PEFT directory:** contains `adapter_model.safetensors` + `adapter_config.json`
- **ComfyUI single file:** single `.safetensors` with baked-in alpha/rank

LoRA merge happens in `src/lora-merge.h` before GPU upload. Supported dtypes in LoRA files: `F32`, `BF16`, `F16`.

Safetensors is also used as the source format for `convert.py` (one-time conversion to GGUF).

## Hardware Backends (via GGML)

Backend selection is automatic: the runtime picks the best available GPU, falling back to CPU. Override via `GGML_BACKEND` environment variable (e.g., `GGML_BACKEND=CUDA0`, `GGML_BACKEND=Vulkan0`).

| Backend | Enabled by | Platforms |
|---|---|---|
| CPU (SIMD) | Always on | All; AVX/AVX2/F16C/FMA auto-detected |
| CUDA | `-DGGML_CUDA=ON` | NVIDIA Turing (sm_75) through Blackwell (sm_121a) |
| Metal | Auto on macOS | Apple Silicon (M1/M2/M3/M4) and Intel Mac |
| Vulkan | `-DGGML_VULKAN=ON` | AMD, Intel, NVIDIA (cross-vendor) |
| ROCm/HIP | `-DGGML_HIP=ON` | AMD GPUs on Linux |
| BLAS (CPU) | `-DGGML_BLAS=ON` | Any BLAS: OpenBLAS, Apple Accelerate |
| Accelerate | Auto on macOS | Apple BLAS + vDSP |

DL (dynamic loading) mode: `-DGGML_BACKEND_DL=ON` builds each backend as a separate `.so`/`.dll` loaded at runtime. Used by `buildall.sh` so a single binary set covers all hardware.

CUDA FlashAttention kernels are compiled in by default (`GGML_CUDA_FA=ON`). cuBLAS and custom MMQ kernels are both available; controlled by `GGML_CUDA_FORCE_MMQ`/`GGML_CUDA_FORCE_CUBLAS` flags.

## HTTP Server Interface

`ace-server` exposes a self-contained HTTP server on `127.0.0.1:8080` (configurable via `--host`/`--port`). No external HTTP library beyond `cpp-httplib 0.40.0` is involved.

**REST API endpoints:**

| Method | Path | Description |
|---|---|---|
| `POST` | `/lm` | Submit LM job (caption → lyrics + codes). Accepts JSON `AceRequest`. Returns `{"id": "..."}`. |
| `POST` | `/lm?mode=inspire` | LM job without audio code generation (metadata + lyrics only). |
| `POST` | `/lm?mode=format` | LM job for lyrics formatting only. |
| `POST` | `/synth` | Submit synthesis job (DiT + VAE → audio). Accepts JSON or multipart (with source/ref audio). Returns `{"id": "..."}`. |
| `POST` | `/synth?format=wav` | Same as `/synth` but returns WAV instead of MP3. |
| `POST` | `/understand` | Submit reverse pipeline job (audio → metadata + codes). Accepts multipart or JSON. |
| `GET` | `/job?id=N` | Poll job status. Returns JSON with `status`, `message`, progress. |
| `GET` | `/job?id=N&result=1` | Fetch job result (audio binary or JSON). |
| `POST` | `/job?id=N&cancel=1` | Cancel a queued/running job. |
| `GET` | `/health` | Returns `{"status":"ok"}`. |
| `GET` | `/props` | Returns available models, server config, default parameters, presets. |
| `GET` | `/logs` | Server-sent events stream of log lines. |

**Job execution model:** single worker thread, FIFO queue. All GPU access is serialized. Job IDs are random 64-bit hex strings. Completed jobs evicted FIFO when pool exceeds `MAX_JOBS`.

**Server-Sent Events:** `/logs` endpoint streams log output to the web UI (`SSE_RECONNECT_MS = 2000` on client side, defined in `tools/webui/src/lib/config.ts`).

**Vite dev proxy:** during web UI development, Vite proxies all `/lm`, `/synth`, `/understand`, `/health`, `/props`, `/logs` requests to `http://localhost:8080` (see `tools/webui/vite.config.ts`).

## Embedded Web UI

The web UI (`tools/webui/`) is a Svelte 5 + TypeScript single-page application compiled into a single self-contained HTML file and gzip-compressed into `tools/public/index.html.gz`. This file is committed to git and embedded as a C byte-array header (`build/index.html.gz.hpp`) during C++ compilation via `tools/xxd.cmake`. `ace-server` serves it directly from memory — no filesystem access required for the UI.

The web UI uses the browser's native **IndexedDB** API (via `tools/webui/src/lib/db.ts`) for local song storage. No external database or cloud sync.

## Audio File Formats

| Format | Read | Write | Notes |
|---|---|---|---|
| WAV (PCM16, float32, mono/stereo, any sample rate) | Yes | Yes (PCM16 stereo) | Reader in `src/wav.h`; custom implementation |
| MP3 | Yes | Yes (configurable kbps, default 128) | Decode: minimp3 (CC0, vendored); Encode: custom `mp3/mp3enc.h` |

All audio is resampled to **48 kHz stereo** internally via `src/audio-resample.h` (Kaiser-windowed polyphase sinc, no external library).

## External Services (Model Downloads)

**Hugging Face** — all model downloads use the `hf` CLI (`pip install hf`).

- Pre-quantized GGUFs: `Serveurperso/ACE-Step-1.5-GGUF` repo (via `models.sh`)
- Raw safetensors checkpoints: `ACE-Step/Ace-Step1.5`, `ACE-Step/acestep-5Hz-lm-4B`, and variant repos (via `checkpoints.sh`)

No Hugging Face API key or authentication is required for public repos. No network calls happen at inference time — all model data is local files.

## Conversion Pipeline (Python, offline only)

`convert.py` is a one-time offline tool (not part of inference):

1. Reads safetensors checkpoints from `checkpoints/` (sharded or single-file)
2. Parses `config.json` and `tokenizer.json` / `vocab.json` / `merges.txt`
3. Packs weights + config + tokenizer + silence latent into self-contained GGUF files in `models/`
4. Uses `gguf` Python package (`>=0.1.0`) as the GGUF writer

The resulting GGUFs are fully self-contained: no external config or tokenizer files are needed at C++ inference time.

## Noise Generation

The Philox4x32-10 PRNG in `src/philox.h` reproduces the exact noise pattern of `torch.cuda.manual_seed()` + `torch.randn()` on CUDA, enabling deterministic generation by seed across CPU and GPU backends. No PyTorch is linked at runtime.

---

*Integration audit: 2026-04-18*
