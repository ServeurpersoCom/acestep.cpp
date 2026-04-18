# Testing Patterns

**Analysis Date:** 2026-04-18

## Test Framework

**Runner:**
- No external test framework (no Google Test, Catch2, doctest, etc.)
- Tests are standalone C++ executables and Python scripts
- C++ test build: GNU `make` with a hand-written `Makefile` in `tests/`
- Config: `tests/Makefile`

**Assertion library:**
- None — validation is done by comparing numeric outputs (cosine similarity, exact match counts) and printing pass/fail to stdout

**Run commands:**
```bash
# Build the C++ Philox test
cd tests/
make test-philox

# Run Philox correctness check (requires CUDA + PyTorch)
cd tests/
./test-philox.py

# Run DiT cosine similarity comparison (requires models + Python torch)
cd tests/
./debug-dit-cossim.py                 # turbo BF16
./debug-dit-cossim.py --quant Q6_K    # specific quantization
./debug-dit-cossim.py --mode sft      # SFT model variant
./debug-dit-cossim.py --mode all      # all 4 model variants

# Run DiT comparison across all backends and quants (batch)
cd tests/
./debug-dit-cossim.sh

# Run LM logit comparison
cd tests/
./debug-lm-logits.sh

# CI build smoke test (from repo root)
./build/ace-lm --help
./build/ace-synth --help
./build/quantize --help
```

## Test File Organization

**Location:** `tests/` at repo root — entirely separate from `src/`.

**Naming:**
- C++ test files: `test-<component>.cpp` (e.g. `tests/test-philox.cpp`)
- Python validation scripts: `test-<component>.py` or `debug-<component>-<metric>.py`
- Shell batch runners: `debug-<component>-<metric>.sh`

**Structure:**
```
tests/
├── Makefile                    # builds test-philox from test-philox.cpp
├── test-philox.cpp             # C++ PRNG correctness binary
├── test-philox.py              # Python harness: build + compare vs PyTorch CUDA
├── debug-dit-cossim.py         # DiT layer-by-layer cosine similarity vs Python
├── debug-dit-cossim.sh         # batch runner: all backends x all quants
├── debug-lm-logits.py          # LM logit comparison vs Python reference
├── debug-lm-logits.sh          # batch runner for LM logit checks
├── debug-detok-cossim.py       # FSQ detokenizer cosine similarity check
├── debug-tok-cossim.py         # FSQ tokenizer cosine similarity check
├── request0.json               # sample request fixture for manual testing
├── CPU-BF16.log                # captured test run output (committed)
├── CPU-Q4_K_M.log
├── CPU-Q5_K_M.log
├── CPU-Q6_K.log
├── CPU-Q8_0.log
├── CUDA0-BF16.log              # committed baseline results per backend+quant
├── ...                         # (Metal, Vulkan variants too)
└── philox-noise.f32            # generated binary (gitignored, temporary)
```

**Log files committed to git:** Test output logs (`CPU-BF16.log`, `CUDA0-Q8_0.log`, `Metal-Q4_K_M.log`, `Vulkan0-Q6_K.log`, etc.) are committed as baseline reference. These capture per-layer cosine similarity scores across backend/quantization combinations.

## Types of Tests

**Unit tests (C++ binary):**
- `tests/test-philox.cpp` — validates `philox_randn()` PRNG output by writing raw `float32` to disk for external comparison.
- Built with: `g++ -std=c++17 -O2 -I../src -o test-philox test-philox.cpp -lm`
- No dependency on ggml or any model weights.

**Numerical validation (Python scripts):**
- `tests/test-philox.py` — compares C++ Philox output against `torch.randn(dtype=bfloat16)` on CUDA. Reports exact match %, cosine similarity, max diff. Threshold: `cos > 0.9999 = OK`.
- `tests/debug-dit-cossim.py` — runs full DiT inference via `ace-synth --dump-dir` and compares intermediate tensors layer-by-layer against Python ACE-Step reference using cosine similarity and max absolute error.
- `tests/debug-lm-logits.py` — compares LM logit outputs (via `ace-lm --dump-logits`) against Python reference logits.
- `tests/debug-detok-cossim.py`, `debug-tok-cossim.py` — FSQ tokenizer/detokenizer cosine similarity checks.

**Integration tests (CI smoke tests):**
- Defined in `.github/workflows/ci-build.yml`
- Run after every successful build: invoke each binary with no args (or `--help`) and check it exits cleanly:
  ```bash
  ./build/ace-lm --help 2>&1 | head -5
  ./build/ace-synth --help 2>&1 | head -5
  ./build/quantize --help 2>&1 | head -3
  ```
- Confirms all executables link and initialize correctly without model files.

**Performance / accuracy regression (log files):**
- Committed `.log` files in `tests/` capture cosine similarity scores per quantization variant.
- Compared manually against new runs to catch numerical regressions.
- Coverage: CPU, CUDA0, Metal, Vulkan0 backends × BF16, Q8_0, Q6_K, Q5_K_M, Q4_K_M quantizations.

## Debug Dump Infrastructure

The `DebugDumper` struct in `src/debug.h` provides in-process tensor dumping:
```cpp
struct DebugDumper { char dir[512]; bool enabled; };
static void debug_dump_2d(const DebugDumper * d, const char * name, const float * data, int dim0, int dim1);
```
- Dumps tensors to `<dir>/<name>.bin` in a simple binary format: `[ndims:i32][shape:i32 x ndims][data:f32 x numel]`
- Enabled via `--dump-dir <path>` CLI flag on `ace-synth` / `ace-lm`
- Python validation scripts load these binary files with `numpy.fromfile` and compare against PyTorch reference values

**Comparison metrics used in validation scripts:**
- Cosine similarity (`debug_cosine_sim` in `src/debug.h`)
- Max absolute error (`debug_max_abs_err`)
- Mean absolute error (`debug_mean_abs_err`)
- Exact match count (for PRNG validation)

## Coverage Areas

**Covered:**
- Philox4x32-10 PRNG: exact numerical match against PyTorch CUDA implementation
- DiT transformer: per-layer intermediate tensor accuracy across all quantization formats and backends
- LM (Qwen3): logit-level comparison against Python reference
- FSQ tokenizer and detokenizer: cosine similarity checks
- All binary executables: basic smoke test (link + startup)

**Not covered (no automated tests):**
- JSON request parsing (`src/request.cpp`) — no unit test
- WAV read/write (`src/wav.h`) — no unit test
- BPE tokenizer (`src/bpe.h`) — no unit test
- Audio resampling (`src/audio-resample.h`) — no unit test
- Model registry (`src/model-registry.h`) — no unit test
- Full end-to-end audio generation quality (requires models, manual listening)

## CI/CD Pipeline

**Workflows:** `.github/workflows/ci-build.yml` and `.github/workflows/release.yml`

**CI Build** (`.github/workflows/ci-build.yml`):
- Triggers: pull requests and manual dispatch
- Matrix: `ubuntu-latest` and `macos-latest`
- Ubuntu build: installs cmake, g++, libopenblas-dev; builds with `-DGGML_BLAS=ON`
- macOS build: CMake default (Metal enabled automatically on Apple Silicon)
- Post-build: smoke tests on all binaries

**Lint job** (PR-only, Ubuntu):
- `clang-format --dry-run --Werror` — checks all `*.c`, `*.h`, `*.cpp`, `*.hpp` under `src/`, `tools/`, `tests/` (excludes `ggml/`, `build/`, `vendor/`, `mp3/`)
- `cppcheck --enable=all --error-exitcode=1` with suppressed categories:
  - `missingIncludeSystem`, `missingInclude`, `cstyleCast`, `constVariable`, `constVariablePointer`, `constParameterPointer`, `variableScope`, `uselessCallsSubstr`, `useStlAlgorithm`, `shiftNegativeLHS`

**Release Build** (`.github/workflows/release.yml`):
- Triggers: published GitHub releases and manual dispatch
- Builds three targets: `linux-x64` (Ubuntu 22.04 + CUDA + Vulkan), `macos-arm64-metal`, `windows-x64` (MSVC + CUDA + Vulkan)
- Uses `ccache` to speed up builds
- Packages binaries into platform tarballs/zips and uploads to GitHub release
- Smoke test on each platform before packaging (using `continue-on-error: true`)

**No automated numerical tests in CI** — the debug/cosine-similarity validation scripts require model weights (not committed) and a GPU, so they run only locally.

---

*Testing analysis: 2026-04-18*
