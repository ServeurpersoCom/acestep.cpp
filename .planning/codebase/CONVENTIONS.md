# Coding Conventions

**Analysis Date:** 2026-04-18

## Naming Patterns

**Files:**
- Lowercase with hyphens: `pipeline-synth.cpp`, `dit-sampler.h`, `fsq-detok.h`
- Implementation split: `pipeline-synth.h` (public API) + `pipeline-synth-impl.h` (private struct) + `pipeline-synth.cpp` (orchestrator) + `pipeline-synth-ops.cpp` (primitives)
- Test files prefixed with `test-`: `tests/test-philox.cpp`

**Structs / Types:**
- PascalCase: `AceRequest`, `AceSynth`, `DiTGGML`, `BackendPair`, `WeightCtx`, `PendingCopy`
- Acronyms kept uppercase in names: `VAEGGML`, `DiTGGMLConfig`, `DiTGGMLLayer`, `GGUFModel`
- Internal helper structs use PascalCase too: `APGMomentumBuffer`, `PrefixTree`, `MetadataFSM`

**Functions:**
- `snake_case` with a module-prefix: `ace_synth_load()`, `ace_lm_generate()`, `request_parse()`, `wctx_init()`, `backend_sched_new()`
- Static helpers inside header-only files: `dit_ggml_load_temb()`, `gf_load_tensor()`, `wav_read_u16le()`
- Free functions matching struct lifecycle: `<module>_init`, `<module>_load`, `<module>_free`

**Variables / Parameters:**
- `snake_case`: `n_layers`, `hidden_size`, `src_audio`, `cancel_data`
- Single-letter or short names for tight math loops: `h`, `ic`, `p`, `oc`, `T`, `Oc`
- Descriptive names for anything crossing function boundaries
- Boolean flags prefixed with `have_` or `use_` or `is_`: `have_vae`, `use_fa`, `is_turbo`

**Constants / Macros:**
- `SCREAMING_SNAKE_CASE` macros: `DIT_GGML_MAX_LAYERS`, `QW3LM_MAX_KV_SETS`, `LM_MODE_GENERATE`
- `inline constexpr` string literals preferred over `#define` for string constants:
  ```cpp
  inline constexpr const char * TASK_TEXT2MUSIC = "text2music";
  inline constexpr const char * DIT_INSTR_REPAINT = "Repaint the mask area based on the given conditions:";
  ```
- `static constexpr` for numeric constants in header-only compute files:
  ```cpp
  static constexpr uint32_t PHILOX_M0 = 0xD2511F53u;
  static constexpr float    CURAND_2POW32_INV = 2.3283064365386963e-10f;
  ```

**Log tag prefixes (stderr):**
- Format: `[Module] Message` — e.g. `[Load]`, `[DiT]`, `[Synth-Load]`, `[WeightCtx]`, `[WAV]`, `[Debug]`, `[GGUF]`, `[LoRA]`
- FATAL errors include the word "FATAL": `[Load] FATAL: no backend available`
- Warnings include "WARNING": `[task] WARNING: 'x' is not a standard track name`

## Code Style

**Formatting tool:** `clang-format` with project `.clang-format` at root.

**Key style settings (from `.clang-format`):**
- `IndentWidth: 4`, `TabWidth: 4`, `UseTab: Never`
- `ColumnLimit: 120`
- `Standard: c++17`
- `LineEnding: LF`
- Braces: attached (`AfterFunction: false`, `AfterClass: false`, `AfterControlStatement: false`)
- `AllowShortBlocksOnASingleLine: Never` — single-line blocks are forbidden
- `AllowShortIfStatementsOnASingleLine: Never`
- `AllowShortLoopsOnASingleLine: false`
- `InsertBraces: true` — clang-format inserts braces even for single-statement bodies
- `PointerAlignment: Middle` — pointer/reference placed in middle: `float * ptr`, `const char * str`
- `ReferenceAlignment: Middle` — same for references: `const std::string & name`
- `SeparateDefinitionBlocks: Always` — blank line between each top-level definition
- `MaxEmptyLinesToKeep: 1`
- `AlignTrailingComments: Always` — trailing inline comments aligned in blocks
- `AlignConsecutiveAssignments: AcrossComments`, `AlignConsecutiveDeclarations: AcrossComments`

**Pointer/reference syntax example:**
```cpp
struct ggml_tensor * src = ggml_get_tensor(gf.meta, name.c_str());
const void *         raw = gf.mapping + gf.data_offset + offset;
```

**Trailing comments aligned in assignment blocks:**
```cpp
float lm_temperature;      // 0.85
float lm_cfg_scale;        // 2.0
float lm_top_p;            // 0.9
```

## Header / Include Organization

**Header guard:** `#pragma once` universally — no `#ifndef` guards.

**First line after `#pragma once`:** A one-line comment with filename and brief purpose:
```cpp
#pragma once
// timer.h: simple wall-clock timer (steady_clock)
```

**Include order** (enforced by `.clang-format` `IncludeBlocks: Regroup`, `SortIncludes: CaseInsensitive`):
1. Local project headers in `"quotes"` (priority 1, sorted)
2. C system headers `<*.h>` (priority 2)
3. C++ standard library `<headers>` (priority 3)

**Example from `pipeline-synth.cpp`:**
```cpp
#include "pipeline-synth.h"

#include "bpe.h"
#include "cond-enc.h"
#include "dit.h"
// ... more local headers ...

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
```

**Cross-platform guards** for OS-specific includes:
```cpp
#ifdef _WIN32
#    include <windows.h>
#else
#    include <fcntl.h>
#    include <sys/mman.h>
#endif
```
PP directives indented after `#`: `IndentPPDirectives: AfterHash`.

**MSVC suppression** for deprecation warnings via CMake: `add_compile_definitions(_CRT_SECURE_NO_WARNINGS)`.

## Memory Management Patterns

**No smart pointers** — zero uses of `std::unique_ptr` / `std::shared_ptr` in the entire project.

**Manual ownership via opaque pointers** — public API returns raw pointers, caller frees via explicit free functions:
```cpp
AceSynth * ace_synth_load(const AceSynthParams * params);  // NULL on failure
void       ace_synth_free(AceSynth * ctx);                 // caller frees

AceAudio * out;  // caller allocates array, fills via ace_synth_generate
void ace_audio_free(AceAudio * audio);
```

**`new` / `delete`** used inside pipeline `_load` / `_free` functions for opaque context structs:
```cpp
AceSynth * ctx = new AceSynth();
// ... on any failure path:
delete ctx;
return NULL;
```

**ggml context management via `WeightCtx`** — wraps `ggml_context` + `ggml_backend_buffer_t`, freed together via `wctx_free()`.

**Zero-initialization idiom** — `*gf = {}` after close, `*m = {}` after free:
```cpp
static void gf_close(GGUFModel * gf) {
    // ... resource cleanup ...
    *gf = {};
}
```

**`std::vector<float>`** used as growable scratch buffers inside structs; no custom allocators.

**`mmap` for GGUF files** (POSIX) / `CreateFileMappingA` (Win32) — weight data is memory-mapped rather than heap-copied. Pointer stored in `GGUFModel::mapping`.

**`malloc` / `free`** used in low-level header-only files for C interop buffers (`wav.h`, `bpe.h`):
```cpp
audio = (float *) malloc((size_t) n_samples * 2 * sizeof(float));
// ... caller is responsible for free(audio)
```

**`thread_local` pre-allocated buffers** used in hot sampling paths to avoid per-call malloc:
```cpp
static thread_local std::vector<float>     tmp_buf;
static thread_local std::vector<TokenProb> sorted_buf;
```

## Error Handling Approach

**No exceptions** — the codebase does not use C++ exceptions anywhere.

**Two-tier error signaling:**
1. **Fatal (non-recoverable):** `fprintf(stderr, "[...] FATAL: ...")` followed by `exit(1)`. Used when a required resource (backend, GGUF tensor, weight buffer) cannot be obtained.
2. **Soft failure (caller handles):** Functions return `false` / `NULL` / `-1` on error. Caller checks and propagates or logs.

**Return-value convention by function type:**
- Load functions: return `NULL` on failure (opaque pointer types), `false` for `bool`-returning helpers
- Generate functions: return `0` on success, `-1` on error or cancellation
- Parse functions: return `false` on malformed input
- All failures log to `stderr` before returning

**FATAL vs soft example:**
```cpp
// Soft: caller handles
if (!dit_ggml_load(&ctx->dit, params->dit_path, ...)) {
    fprintf(stderr, "[Synth-Load] FATAL: DiT load failed\n");
    delete ctx;
    return NULL;
}

// Hard: no recovery
if (!bp.backend) {
    fprintf(stderr, "[Load] FATAL: no backend available\n");
    exit(1);
}
```

**`GGML_ASSERT`** used sparingly (2 occurrences in `gguf-weights.h` and `bpe.h`) only for internal invariants.

**Cancel callbacks** — generate functions accept `bool (*cancel)(void *)` polled between steps. On cancellation, return `-1`:
```cpp
int ace_synth_generate(AceSynth * ctx, ..., bool (*cancel)(void *) = nullptr, void * cancel_data = nullptr);
```

## Common Macros and Utility Patterns

**`#define` macros** used only for integer constants and mode flags (not for code):
```cpp
#define DIT_GGML_MAX_LAYERS 32
#define LM_MODE_GENERATE    0
#define LM_MODE_INSPIRE     1
#define LM_MODE_FORMAT      2
```

**`static` functions in headers** — most implementation lives in header-only `.h` files as `static` functions or `static inline` functions, included by exactly one `.cpp` TU:
```cpp
static void dit_ggml_load_temb(DiTGGMLTembWeights * w, ...) { ... }
static bool gf_load(GGUFModel * gf, const char * path) { ... }
static void wctx_init(WeightCtx * wctx, int n_tensors) { ... }
```

**`link_ggml_backends` CMake macro** — shared compile options and ggml backend linkage applied to every target:
```cmake
macro(link_ggml_backends target)
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wshadow -Wconversion ...)
    target_link_libraries(${target} PRIVATE ggml Threads::Threads)
endmacro()
```

**Timer utility** — `Timer` struct (steady_clock) used throughout load paths:
```cpp
Timer timer;
// ... work ...
fprintf(stderr, "[Synth-Load] DiT weight load: %.1f ms\n", timer.ms());
timer.reset();
```

**`ggml_init_params` designated initializer comments** — ggml API uses positional struct init; comments label each field:
```cpp
struct ggml_init_params params = {
    /*.mem_size   =*/ ctx_size,
    /*.mem_buffer =*/ NULL,
    /*.no_alloc   =*/ true,
};
```

**`snprintf` for string formatting** — used throughout instead of `sprintf` or stream formatting:
```cpp
char prefix[128];
snprintf(prefix, sizeof(prefix), "decoder.layers.%d", i);
```

**Lambda for type-dispatch** — used in weight loading to avoid code duplication across float types:
```cpp
auto cvt = [&](auto read_fn) {
    for (int h = 0; h < H; h++) { ... }
};
if (src->type == GGML_TYPE_BF16) {
    cvt([&](int i) { return ggml_bf16_to_fp32(...); });
} else if (src->type == GGML_TYPE_F32) {
    cvt([&](int i) { return s[i]; });
}
```

**Compiler warning suppression** in CMake is scoped to specific targets and flags (`/wd4100`, `-Wno-unused-parameter`) — not project-wide disables.

---

*Convention analysis: 2026-04-18*
