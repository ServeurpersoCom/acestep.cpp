# Codebase Structure

**Analysis Date:** 2026-04-18

## Directory Layout

```
acestep.cpp/
├── src/                    # Core library headers and pipeline implementations
├── tools/                  # Executable entry points + WebUI frontend
│   ├── webui/              # Svelte browser UI (TypeScript, Vite)
│   │   └── src/
│   │       └── components/ # Svelte UI components
│   └── public/             # Built WebUI output (index.html.gz committed)
├── ggml/                   # GGML submodule (tensor ops, backends, GGUF)
├── vendor/                 # Third-party C/C++ libraries
│   ├── cpp-httplib/        # HTTP server library
│   ├── yyjson/             # Fast JSON parser
│   └── minimp3/            # MP3 decoder (header-only CC0)
├── mp3/                    # Custom MP3 encoder (header-only, in-tree)
├── models/                 # GGUF model files (runtime, not committed)
├── loras/                  # LoRA adapter files (runtime, not committed)
├── tests/                  # Integration test scripts and reference logs
├── docs/                   # Technical documentation
├── CMakeLists.txt          # Root build file
├── build*/                 # Build scripts for each backend
└── build/                  # CMake build output directory (generated)
```

---

## Directory Purposes

### `src/`
The core library. All files here are compiled into `libacestep-core.a` (except pipeline headers are header-only). No `main()` functions.

**Pipeline implementations (compiled .cpp):**
- `src/pipeline-lm.cpp` — LM pipeline implementation
- `src/pipeline-synth.cpp` — Synth pipeline orchestrator (thin; delegates to ops)
- `src/pipeline-synth-ops.cpp` — Synth pipeline primitive operations
- `src/pipeline-understand.cpp` — Understand pipeline implementation
- `src/request.cpp` — `AceRequest` JSON serialization (yyjson)

**Pipeline public API headers:**
- `src/pipeline-lm.h` — `AceLm*`, `ace_lm_load/generate/free`
- `src/pipeline-synth.h` — `AceSynth*`, `AceAudio`, `ace_synth_load/generate/free`
- `src/pipeline-understand.h` — `AceUnderstand*`, `ace_understand_load/generate/free`
- `src/request.h` — `AceRequest` struct, JSON parse/write/dump

**Model component headers (header-only, included by pipeline .cpp):**
- `src/dit.h` — DiT weight structs + GGUF load + backend init (`DiTGGML`)
- `src/dit-graph.h` — DiT ggml compute graph builders (per-layer, full forward)
- `src/dit-sampler.h` — DiT Euler flow-matching sampler with APG-CFG
- `src/qwen3-enc.h` — Qwen3 encoder backbone (text/lyric/timbre encoders, `Qwen3GGML`)
- `src/qwen3-lm.h` — Qwen3 causal LM with KV cache (`Qwen3LM`)
- `src/cond-enc.h` — Condition encoder: lyric (8L) + timbre (4L) + text projection (`CondGGML`)
- `src/vae.h` — Oobleck VAE decoder (`VAEGGML`)
- `src/vae-enc.h` — Oobleck VAE encoder (`VAEEncoder`)
- `src/fsq-tok.h` — FSQ tokenizer: latents → 5 Hz audio codes (`TokGGML`)
- `src/fsq-detok.h` — FSQ detokenizer: audio codes → context latents (`DetokGGML`)

**Infrastructure headers:**
- `src/backend.h` — Shared GGML backend init/release (GPU auto-detect, ref-counted)
- `src/gguf-weights.h` — mmap GGUF loader, tensor loading helpers (`GGUFModel`)
- `src/weight-ctx.h` — `WeightCtx` for staging tensors before GPU upload
- `src/model-registry.h` — Model directory scan and GGUF classification (`ModelRegistry`)
- `src/lora-merge.h` — Runtime LoRA delta merge into GGUF weights
- `src/safetensors.h` — Safetensors file reader (for LoRA)
- `src/bpe.h` — Qwen3/GPT-2 byte-level BPE tokenizer (`BPETokenizer`)
- `src/metadata-fsm.h` — FSM for constrained metadata decoding during LM (`MetadataFSM`)
- `src/prompt.h` — Qwen3 chat template building, CoT parsing, YAML generation (`AcePrompt`)
- `src/sampling.h` — Token sampling: temperature, top-k, top-p, multinomial
- `src/philox.h` — Philox4x32-10 PRNG matching PyTorch cuRAND (for reproducible noise)
- `src/audio-io.h` — Unified WAV/MP3 read/write (planar stereo float)
- `src/audio-resample.h` — Sample rate conversion
- `src/task-types.h` — Task identifiers and DiT instruction strings
- `src/pipeline-synth-impl.h` — `AceSynth` private struct definition (shared by two .cpp)
- `src/debug.h` — `DebugDumper` for intermediate tensor dumps
- `src/timer.h` — High-resolution timer for performance logging
- `src/wav.h` — WAV file reader

### `tools/`
Entry points (one `.cpp` per binary) and the browser WebUI.

**C++ entry points:**
- `tools/ace-server.cpp` — HTTP server; all endpoints, job queue, model lifecycle, log capture
- `tools/ace-lm.cpp` — CLI wrapper for LM pipeline
- `tools/ace-synth.cpp` — CLI wrapper for Synth pipeline
- `tools/ace-understand.cpp` — CLI wrapper for Understand pipeline
- `tools/quantize.cpp` — GGUF requantizer (BF16 → K-quants)
- `tools/neural-codec.cpp` — Standalone VAE encode/decode (WAV ↔ latent)
- `tools/mp3-codec.cpp` — Standalone MP3 encoder/decoder (no ggml)
- `tools/xxd.cmake` — CMake script: converts `index.html.gz` → C header (`index.html.gz.hpp`)
- `tools/version.cmake` — CMake script: embeds git hash into `version.h`

**WebUI (Svelte/TypeScript):**
- `tools/webui/src/App.svelte` — Root application component
- `tools/webui/src/components/RequestForm.svelte` — Generation parameter form
- `tools/webui/src/components/SongCard.svelte` — Single track result card (play, download)
- `tools/webui/src/components/SongList.svelte` — Track list with polling
- `tools/webui/src/components/Waveform.svelte` — Audio waveform visualizer
- `tools/webui/src/components/LogCard.svelte` — Live log display (SSE `/logs`)
- `tools/webui/src/components/Toast.svelte` — Notification toasts
- `tools/public/index.html.gz` — Pre-built WebUI (committed; updated by `./buildwebui.sh`)

### `ggml/`
GGML as a git submodule. Provides: tensor ops, CUDA/Metal/Vulkan/CPU backends, GGUF file format, flash attention. Not modified by this project; `GGML_MAX_NAME=128` overridden in CMakeLists.txt.

### `vendor/`
- `vendor/cpp-httplib/` — Single-header HTTP server (MIT)
- `vendor/yyjson/` — Fast JSON library compiled as static `libyyjson.a`
- `vendor/minimp3/` — Header-only MP3 decoder (CC0)

### `mp3/`
In-tree custom MP3 encoder (`mp3enc.h` + supporting headers). Header-only. Used by `audio-io.h` and `tools/mp3-codec.cpp`.

### `models/`
Runtime directory for GGUF files. Not committed. Classified at startup into four buckets by `model-registry.h`:
- `acestep-lm` architecture → LM bucket (LM GGUF)
- `acestep-dit` architecture → DiT bucket (DiT GGUF, also contains CondEncoder, FSQ weights)
- `acestep-text-enc` architecture → Text-Enc bucket (singleton, Qwen3-Embedding)
- `acestep-vae` architecture → VAE bucket (singleton, Oobleck VAE)

### `loras/`
Runtime directory for LoRA adapters. Not committed. Supports:
- `.safetensors` single-file (ComfyUI format, alpha baked in)
- Subdirectories containing `adapter_model.safetensors` (PEFT format)

### `tests/`
Integration test scripts (shell + Python) and reference output logs. Not a unit test suite. Includes:
- `tests/simple.sh` / `tests/full.sh` etc. — End-to-end generation scripts
- `tests/debug-*.py` / `tests/debug-*.sh` — Cosine similarity comparison vs Python reference
- `tests/test-philox.cpp` / `tests/test-philox.py` — Philox PRNG correctness test
- `tests/client*.sh` / `tests/client*.py` — HTTP client scripts for server testing
- `tests/*.log` — Reference output logs per backend/quantization

### `docs/`
- `docs/ARCHITECTURE.md` — Full API reference (AceRequest JSON, task types, quantization, internals)

---

## Key Source Files

| File | Owns |
|------|------|
| `src/request.h` + `src/request.cpp` | `AceRequest` struct, all JSON serialization |
| `src/pipeline-lm.h` + `src/pipeline-lm.cpp` | LM pipeline public API + implementation |
| `src/pipeline-synth.h` + `src/pipeline-synth.cpp` | Synth pipeline public API + orchestrator |
| `src/pipeline-synth-ops.h` + `src/pipeline-synth-ops.cpp` | All synthesis primitive operations |
| `src/pipeline-synth-impl.h` | `AceSynth` private struct (shared between two .cpp) |
| `src/pipeline-understand.h` + `src/pipeline-understand.cpp` | Reverse pipeline |
| `src/dit.h` | DiT weight structs + GGUF load function |
| `src/dit-graph.h` | DiT ggml compute graph construction |
| `src/dit-sampler.h` | Euler flow-matching + APG-CFG sampling loop |
| `src/task-types.h` | All task/mode identifiers and instruction strings |
| `src/model-registry.h` | Model directory scanning and classification |
| `src/backend.h` | Shared GGML backend initialization |
| `tools/ace-server.cpp` | HTTP server: all endpoints, job queue, model lifecycle |
| `CMakeLists.txt` | All build targets and dependencies |

---

## Entry Points

### `ace-server` (primary production binary)
- **Source:** `tools/ace-server.cpp`
- **Invoked by:** `./server.sh` or `server.cmd`
- **Starts:** HTTP server on `--host`/`--port` (default 127.0.0.1:8080 in code, 8085 in scripts)
- **Spawns:** One worker thread, one log-reader thread
- **Opens:** `--models` directory for `ModelRegistry`, `--loras` directory for LoRA scan

### `ace-lm` (CLI)
- **Source:** `tools/ace-lm.cpp`
- **Inputs:** `--request <json>` + `--lm <gguf>`
- **Outputs:** Enriched JSON files (`request0.json`, `request1.json`, ...) alongside input

### `ace-synth` (CLI)
- **Source:** `tools/ace-synth.cpp`
- **Inputs:** `--request <json...>` + `--embedding <gguf>` + `--dit <gguf>` + `--vae <gguf>`
- **Outputs:** Audio files (`.mp3` or `.wav`) named after input JSON files

### `ace-understand` (CLI)
- **Source:** `tools/ace-understand.cpp`
- **Inputs:** Audio file + `--lm`, `--dit`, `--vae` GGUFs
- **Outputs:** JSON with caption, lyrics, codes, metadata

### `quantize`
- **Source:** `tools/quantize.cpp`
- **Purpose:** Requantize BF16 GGUF → Q4_K_M, Q5_K_M, Q6_K, Q8_0, etc.

### `neural-codec`
- **Source:** `tools/neural-codec.cpp`
- **Purpose:** Standalone VAE encode (WAV → latent tensor) or decode (latent → WAV), no DiT

### `mp3-codec`
- **Source:** `tools/mp3-codec.cpp`
- **Purpose:** Standalone MP3 encode/decode; does not link GGML

---

## Build Organization

### CMake targets

| Target | Type | Sources | Links |
|--------|------|---------|-------|
| `acestep-core` | Static lib | `src/request.cpp`, `src/pipeline-lm.cpp`, `src/pipeline-synth.cpp`, `src/pipeline-synth-ops.cpp`, `src/pipeline-understand.cpp` | `ggml`, `yyjson` |
| `ace-server` | Executable | `tools/ace-server.cpp`, `${WEBUI_OUTPUT}` | `acestep-core`, `httplib`, ggml backends |
| `ace-lm` | Executable | `tools/ace-lm.cpp` | `acestep-core`, ggml backends |
| `ace-synth` | Executable | `tools/ace-synth.cpp` | `acestep-core`, ggml backends |
| `ace-understand` | Executable | `tools/ace-understand.cpp` | `acestep-core`, ggml backends |
| `quantize` | Executable | `tools/quantize.cpp` | ggml backends |
| `neural-codec` | Executable | `tools/neural-codec.cpp` | ggml backends |
| `mp3-codec` | Executable | `tools/mp3-codec.cpp` | `Threads::Threads`, `m` |
| `yyjson` | Static lib | `vendor/yyjson/yyjson.c` | — |
| `version` | Custom target | `tools/version.cmake` | — (generates `version.h`) |
| WebUI embed | Custom command | `tools/xxd.cmake` | generates `index.html.gz.hpp` |

### Build scripts

| Script | Backend |
|--------|---------|
| `buildcuda.sh` / `buildcuda.cmd` | CUDA (NVIDIA) |
| `buildvulkan.sh` / `buildvulkan.cmd` | Vulkan (AMD/Intel) |
| `buildcpu.sh` | CPU + BLAS only |
| `buildall.sh` / `buildall.cmd` | CUDA + Vulkan + CPU (runtime DL loading) |
| `buildtermux.sh` | Android Termux |
| `buildwebui.sh` | Rebuild WebUI only (npm + vite) |

macOS automatically enables Metal and Accelerate BLAS with any script.

### Generated files (build-time)
- `build/version.h` — Git commit hash (`ACE_VERSION` macro)
- `build/index.html.gz.hpp` — Embedded WebUI as C byte array (`INDEX_HTML_GZ[]`)
- `build/ace-server`, `build/ace-lm`, etc. — All binaries placed at `CMAKE_BINARY_DIR` root

---

## Naming Conventions

**Files:**
- `pipeline-*.h` / `pipeline-*.cpp` — Pipeline-level abstractions
- `*-enc.h` — Encoder models (vae-enc, cond-enc, qwen3-enc)
- `*-graph.h` — ggml compute graph construction
- `*-sampler.h` — Sampling/inference loops
- `fsq-*.h` — FSQ tokenizer / detokenizer
- `ace-*.cpp` — Binary entry points (in `tools/`)
- `*.h` in `src/` — Header-only or declaration headers

**Structs:**
- `AceFoo` — Public pipeline handles and data types (`AceRequest`, `AceSynth`, `AceAudio`)
- `FooGGML` — GGML model weight containers (`DiTGGML`, `VAEGGML`, `Qwen3GGML`)
- `FooGGMLConfig` — Model hyperparameters from GGUF metadata
- `FooGGMLLayer` — Per-layer weight structs
- `FooParams` — Pipeline construction parameters (`AceLmParams`, `AceSynthParams`)

---

## Where to Add New Code

**New task type (e.g. `duet`):**
1. Add constant to `src/task-types.h`
2. Add instruction string helper to `src/task-types.h`
3. Add branch in the mode-routing block in `src/pipeline-synth.cpp` (`ace_synth_generate`)
4. Add any new ops primitives to `src/pipeline-synth-ops.h` / `src/pipeline-synth-ops.cpp`

**New model component (e.g. a new encoder):**
1. Create `src/myencoder.h` following the pattern of `src/qwen3-enc.h`
2. Add struct fields to `src/pipeline-synth-impl.h` (`AceSynth`)
3. Load in `ace_synth_load` in `src/pipeline-synth.cpp`
4. Call in appropriate `ops_*` function in `src/pipeline-synth-ops.cpp`

**New HTTP endpoint:**
1. Add handler in `tools/ace-server.cpp`
2. Register with `svr.Post("/myendpoint", handle_my_endpoint)`
3. Use the job queue pattern: validate → create job → `work_push()` → return ID

**New CLI tool:**
1. Create `tools/my-tool.cpp` with `main()`
2. Add `add_executable(my-tool tools/my-tool.cpp)` + `link_ggml_backends(my-tool)` to `CMakeLists.txt`

**New WebUI component:**
1. Create `tools/webui/src/components/MyComponent.svelte`
2. Import in `tools/webui/src/App.svelte`
3. Run `./buildwebui.sh` to regenerate `tools/public/index.html.gz`
4. Rebuild `ace-server` to re-embed the updated WebUI

---

## Special Directories

**`ggml/`:**
- Purpose: GGML tensor library submodule (backends, GGUF, ops)
- Generated: No (pinned submodule commit)
- Committed: Yes (as submodule reference)

**`models/`:**
- Purpose: GGUF model files at runtime
- Generated: No (downloaded by user or `models.sh`)
- Committed: No

**`loras/`:**
- Purpose: LoRA adapter files at runtime
- Generated: No
- Committed: No

**`tools/public/`:**
- Purpose: Pre-built WebUI HTML (gzip compressed)
- Generated: Yes (`./buildwebui.sh`)
- Committed: Yes (`index.html.gz` committed so C++ build works without npm)

**`build/`:**
- Purpose: CMake build output
- Generated: Yes
- Committed: No

---

*Structure analysis: 2026-04-18*
