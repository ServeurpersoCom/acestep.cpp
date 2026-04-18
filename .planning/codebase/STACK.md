# Technology Stack

**Analysis Date:** 2026-04-18

## Languages

**Primary:**
- C++17 — all inference engine code (`src/`, `tools/`)
- C — vendored library code (`vendor/yyjson/yyjson.c`)

**Secondary:**
- Python 3 — model conversion script (`convert.py`), debug/test scripts (`tests/debug-*.py`, `tests/test-philox.py`)
- TypeScript 5.9.3 (resolved) — web UI frontend (`tools/webui/src/`)
- Svelte 5.55.2 (resolved) — web UI component framework (`tools/webui/src/`)
- Bash — all build, download, and utility scripts (`.sh` files at project root)
- Batch/CMD — Windows equivalents of bash scripts (`.cmd` files at project root)
- CMake — build system (`CMakeLists.txt`, `tools/version.cmake`, `tools/xxd.cmake`)

## Runtime

**C++ Environment:**
- No runtime required; statically compiled executables
- Requires pthreads (explicit on glibc < 2.34; `find_package(Threads REQUIRED)`)
- POSIX syscalls (`mmap`, `dirent`, `pipe`, `dup`) on Linux/macOS; Win32 equivalents on Windows
- Termux/Android build supported via `buildtermux.sh`

**Web UI Dev Environment:**
- Node.js (version unspecified; inferred from npm/Vite 7 requirements)
- npm — package manager; `package-lock.json` committed at `tools/webui/package-lock.json`

## Build System

**Tool:** CMake (minimum 3.14)

**Configuration file:** `CMakeLists.txt` (root), `ggml/CMakeLists.txt`, `vendor/cpp-httplib/CMakeLists.txt`

**Key CMake flags:**

| Flag | Effect |
|---|---|
| `-DGGML_CUDA=ON` | Enable NVIDIA CUDA backend |
| `-DGGML_HIP=ON` | Enable AMD ROCm/HIP backend |
| `-DGGML_VULKAN=ON` | Enable Vulkan backend |
| `-DGGML_METAL=ON` | Enable Apple Metal (auto-enabled on macOS) |
| `-DGGML_BLAS=ON` | Enable BLAS acceleration for CPU |
| `-DGGML_BACKEND_DL=ON` | Build backends as runtime-loadable `.so`/`.dll` |
| `-DGGML_CPU_ALL_VARIANTS=ON` | Build all CPU ISA variants (AVX2, AVX512, etc.) |
| `-DCMAKE_CUDA_ARCHITECTURES=...` | CUDA arch list; defaults to Turing-Blackwell range |

**Compile definitions applied by root `CMakeLists.txt`:**
- `GGML_MAX_NAME=128` — extends tensor name limit for DiT tensor paths
- `_FORTIFY_SOURCE=2` — hardened libc on non-MSVC
- `_CRT_SECURE_NO_WARNINGS` — suppress MSVC fopen deprecation

**Outputs:** Seven binaries built to `build/`:
- `ace-lm` — LLM inference (caption → lyrics + audio codes)
- `ace-synth` — Full synthesis pipeline (DiT + VAE → audio)
- `ace-server` — HTTP server with embedded web UI
- `ace-understand` — Reverse pipeline (audio → metadata + lyrics)
- `neural-codec` — VAE encode/decode standalone tool
- `mp3-codec` — MP3 encoder/decoder standalone tool
- `quantize` — GGUF BF16 → K-quant requantizer

**Version embedding:** Git commit hash baked at build time via `tools/version.cmake` → `build/version.h`; web UI reads it too via `tools/webui/vite.config.ts`.

**Web UI build:** `buildwebui.sh` (or `cd tools/webui && npm install && npm run build`); output is a single gzip-compressed `tools/public/index.html.gz` which is committed to git and embedded as a C header (`build/index.html.gz.hpp`) via `tools/xxd.cmake`.

## C++ Frameworks and Libraries

**GGML (git submodule) — version 0.9.11**
- Location: `ggml/`
- Core tensor/compute graph library providing:
  - Multi-backend dispatcher (`ggml-backend`, `ggml-alloc`, `ggml-sched`)
  - GGUF file format reader/writer (`gguf.h`)
  - CPU kernels (SIMD: SSE4.2, AVX, AVX2, AVX512, FMA, F16C, AMX)
  - CUDA backend (cuBLAS, FlashAttention CUDA kernels)
  - Metal backend (Apple Silicon GPU)
  - Vulkan backend (cross-vendor GPU)
  - HIP backend (AMD ROCm)
  - OpenCL, SYCL, WebGPU backends (optional, off by default)
  - Accelerate framework (macOS BLAS, auto-enabled on Apple)

**cpp-httplib — version 0.40.0 (vendored)**
- Location: `vendor/cpp-httplib/httplib.h`, `vendor/cpp-httplib/httplib.cpp`
- Single-file HTTP/1.1 server used by `ace-server`
- Header: `#define CPPHTTPLIB_VERSION "0.40.0"`

**yyjson — version 0.12.0 (vendored)**
- Location: `vendor/yyjson/yyjson.c`, `vendor/yyjson/yyjson.h`
- Fast JSON parser/writer (MIT license); used for `AceRequest` serialization
- Built as static library `yyjson` in CMake

**minimp3 (vendored, CC0)**
- Location: `vendor/minimp3/minimp3.h`
- Header-only MP3 decoder; included via `src/audio-io.h`

**Custom MP3 encoder (in-tree)**
- Location: `mp3/` (header-only: `mp3enc.h`, `mp3enc-*.h`, `tabs_data.h`)
- Used by `src/audio-io.h` for MP3 output

## Key In-Tree Modules (`src/`)

| Module | Purpose |
|---|---|
| `pipeline-lm.cpp/.h` | Qwen3 causal LM inference (CoT + code generation) |
| `pipeline-synth.cpp/.h` | DiT + TextEncoder + CondEncoder + VAE pipeline |
| `pipeline-understand.cpp/.h` | Reverse pipeline: audio → codes + metadata |
| `qwen3-enc.h` | Generic Qwen3 transformer encoder (text-enc, lyric-enc, timbre-enc) |
| `qwen3-lm.h` | Qwen3 causal LM |
| `dit.h` | Diffusion Transformer (AdaLN, GQA, cross-attn, FlashAttention) |
| `vae.h` | AutoencoderOobleck decoder |
| `vae-enc.h` | AutoencoderOobleck encoder |
| `cond-enc.h` | Conditional encoder (lyric + timbre) |
| `fsq-tok.h` | FSQ tokenizer (VAE latents → 5Hz codes) |
| `fsq-detok.h` | FSQ detokenizer (5Hz codes → DiT context) |
| `bpe.h` | GPT-2/Qwen3 byte-level BPE tokenizer |
| `metadata-fsm.h` | FSM-constrained metadata decoding |
| `lora-merge.h` | Runtime LoRA merge (PEFT + ComfyUI `.safetensors`) |
| `safetensors.h` | Minimal mmap-based safetensors reader |
| `gguf-weights.h` | GGUF mmap weight loader |
| `audio-io.h` | Unified audio I/O: WAV + MP3 read/write |
| `audio-resample.h` | Kaiser-windowed polyphase resampler (no external deps) |
| `philox.h` | Philox4x32-10 PRNG (matches PyTorch CUDA `torch.randn`) |
| `backend.h` | GGML backend init/selection (GPU → CPU fallback) |
| `request.h` | `AceRequest` JSON struct |
| `model-registry.h` | GGUF/LoRA directory scanner |

## C++ Standard

- **C++17** (`CMAKE_CXX_STANDARD 17`, `CMAKE_CXX_STANDARD_REQUIRED ON`)
- Compiler warnings: `-Wall -Wextra -Wshadow -Wconversion` on GCC/Clang; `/W4` on MSVC

## Python Dependencies (`ggml/requirements.txt`)

Used only for `convert.py` (safetensors → GGUF conversion) and debug scripts. Not needed at runtime.

| Package | Version | Purpose |
|---|---|---|
| `gguf` | >=0.1.0 | Write GGUF files from Python |
| `numpy` | >=2.0.2 | Tensor data manipulation |
| `torch` | ~2.5.1 (CPU wheel) | Tensor loading from checkpoints |
| `transformers` | >=4.35.2,<5.0.0 | Tokenizer/config loading |
| `accelerate` | ==0.19.0 | Model loading helpers |
| `sentencepiece` | ~0.1.98 | Tokenizer support |

Also: `pip install hf` for the `hf download` CLI used in `models.sh` and `checkpoints.sh`.

## Web UI Stack (`tools/webui/`)

| Package | Resolved Version | Role |
|---|---|---|
| Svelte | 5.55.2 | UI component framework |
| Vite | 7.3.2 | Dev server + production bundler |
| TypeScript | 5.9.3 | Type-checked frontend code |
| `@sveltejs/vite-plugin-svelte` | ^6.0.0 | Svelte integration for Vite |
| `vite-plugin-singlefile` | 2.3.2 | Inline all assets into one HTML file |
| `@lucide/svelte` | 0.577.0 | Icon library |
| Prettier | 3.8.1 | Formatter |
| `prettier-plugin-svelte` | ^3.3.0 | Svelte formatting support |
| `svelte-check` | 4.4.6 | Type-check `.svelte` files |

**TypeScript target:** ES2022 (`tools/webui/tsconfig.json`)

**Build output:** Single `index.html` (all JS/CSS inlined), then gzip-compressed to `tools/public/index.html.gz` and committed. The C++ build embeds this as a byte array header via `tools/xxd.cmake`.

## Platform Support

| Platform | Backends | Notes |
|---|---|---|
| Linux x86_64 | CPU, CUDA, ROCm/HIP, Vulkan, BLAS | Primary build target |
| macOS (Apple Silicon) | CPU, Metal, Accelerate BLAS | Metal + Accelerate auto-enabled |
| Windows (MSVC) | CPU, CUDA, Vulkan | Pre-built binaries available |
| Android/Termux | CPU, BLAS (OpenBLAS) | `buildtermux.sh`; OpenBLAS via `$PREFIX` |

---

*Stack analysis: 2026-04-18*
