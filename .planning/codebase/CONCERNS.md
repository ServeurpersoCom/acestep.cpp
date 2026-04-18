# Codebase Concerns

**Analysis Date:** 2026-04-18

---

## TODO / FIXME / HACK Comments

No `TODO`, `FIXME`, or `HACK` markers exist in `src/` or `tools/`. The codebase is clean of inline debt markers.

The vendored ggml submodule contains many upstream TODOs in `ggml/include/ggml.h` (lines 152–2689) and several backend-specific headers, but these are upstream issues in the dependency, not project issues.

---

## Hard-Coded Magic Numbers and Model Assumptions

**Silent latent capacity cap (`T <= 15000`):**
- Files: `src/pipeline-synth.cpp:92-93`, `src/pipeline-synth-ops.cpp:211-213`
- The `silence_full` buffer is always allocated as exactly `15000 * 64` floats (`~3.6 MB`). The maximum audio duration is therefore `15000 / 25 = 600 seconds (10 minutes)`. Any request for longer audio returns an error silently with no attempt to clamp gracefully.
- Impact: server returns `-1` for jobs requesting >600s; the limit is not exposed in any API documentation or user-visible error message.

**Sample rate locked to 48 kHz:**
- Files: `src/audio-io.h:210`, `src/pipeline-synth-ops.cpp:80`, `src/pipeline-synth.cpp:239,247`
- The entire pipeline assumes 48,000 Hz input/output. This is hard-coded in arithmetic (`/ 48000.0f`) throughout the synthesis ops rather than being read from any model config or constant.
- Impact: if a future model variant used a different sample rate, every arithmetic site would need manual updating.

**Latent stride hard-coded to 1920:**
- Files: `src/pipeline-synth-ops.cpp:68,80,228`, `src/pipeline-synth.cpp:239,414`
- The audio-to-latent ratio `1920` (= 10×6×4×4×2) is embedded directly in arithmetic as a literal. `src/vae-enc.h:295` documents the formula, but none of the callers use a named constant.
- Impact: matches `vae-enc.h` but is duplicated in 6+ locations. Future VAE architecture changes require grep-and-replace.

**Batch size ceiling of 9:**
- Files: `src/pipeline-synth.cpp:219`, `tools/ace-server.cpp:751,779`
- The synthesis pipeline rejects `batch_n > 9` with a hard error. `QW3LM_MAX_KV_SETS = 32` in `src/qwen3-lm.h:30` allows up to 16 cond+uncond LM slots, so the restriction is architectural (DiT memory budget at large T) but is not explained by any constant or comment at the call site.
- Impact: server silently clamps to 9 rather than returning a meaningful error for the caller.

**metas_b buffer size 512:**
- Files: `src/pipeline-synth-ops.cpp:296,350`
- A fixed `char metas_b[512]` is written with `snprintf` that concatenates four user-supplied strings (`bpm`, `timesig`, `keyscale`, `duration`). If any of those fields (coming from user JSON) are unusually long, `snprintf` silently truncates the metadata string. The value passed to the model is then wrong without any warning.

---

## Memory Safety Concerns

**Raw C malloc/free for audio output buffers:**
- Files: `src/pipeline-synth-ops.cpp:715`, `src/pipeline-synth.cpp:454`
- `out[b].samples` is set with `malloc` in `ops_vae_decode_and_splice`. Ownership is transferred to the caller via `AceAudio`, which must call `ace_audio_free`. The server's `synth_worker` (`tools/ace-server.cpp`) calls `ace_audio_free` correctly, but any future consumer that forgets will leak.
- Impact: straightforward to introduce leaks if `AceAudio` is returned across error paths without cleanup.

**`audio_planar_to_interleaved` returns malloc'd pointer with no ownership annotation:**
- File: `src/audio-io.h:348-356`
- The function comment says "caller must free()" but the return type is a raw `float *`. In `tools/ace-server.cpp` the interleaved buffers are captured in lambdas and freed inside the worker functions. This pattern is fragile: if the worker returns early on an error path (cancellation, model load failure), the free must be explicitly present on every exit path.
- The server does handle all early exits (`free(src_interleaved); free(ref_interleaved)` before each early return in `synth_worker` / `understand_worker`), but this is a manual discipline rather than an enforced pattern.

**`realloc` in MP3 decoder with unchecked return on a secondary code path:**
- File: `src/audio-io.h:130`
- Inside `audio_io_read_mp3_buf`, `pcm_buf` is grown with `realloc`. A `NULL` return from `realloc` would cause the subsequent `memcpy` to write to a null pointer. The `pcm_buf` pointer is not checked for NULL after `realloc`.

**GGUFModel mmap pointer cast and pointer arithmetic:**
- File: `src/gguf-weights.h:174-175`
- Tensor data is accessed as `gf.mapping + gf.data_offset + offset` with no bounds check against `gf.file_size`. A corrupt or truncated GGUF file (or an offset computed from a malformed GGUF header) could yield an out-of-bounds read.

---

## `exit(1)` Inside Library Headers (Not Just Binaries)

**Files:** `src/backend.h` (lines 87, 94, 107, 154), `src/gguf-weights.h` (146, 153, 197, 269), `src/dit.h` (151, 156, 190, 207, 212, 247), `src/qwen3-lm.h` (214, 506, 574, 737), `src/cond-enc.h` (294), `src/vae.h` (171, 224), `src/vae-enc.h` (48, 115), `src/qwen3-enc.h` (405, 462)

These headers call `exit(1)` on GGML allocation failures, missing tensor names, and backend init failures. Because these headers are included by the server binary (`ace-server`), a model load error terminates the entire server process rather than failing only the current request. The server handles this at the job level only for errors that propagate a return code, not for `exit(1)`.

- Impact: if a malformed or truncated model GGUF is loaded by any request, the server process terminates. With `--keep-loaded` this cannot be triggered post-startup, but it can be triggered on first request with a model that has missing tensors.
- Fix approach: convert fatal exits in headers to return-code propagation; the server already has patterns for this (`ensure_synth` returns false on failure).

---

## Global Mutable State in `backend.h`

**Files:** `src/backend.h:22-23`

```cpp
static BackendPair g_backend_cache = {};
static int         g_backend_refs  = 0;
```

These file-scope statics are **not thread-safe**. Because `backend.h` is included in multiple compilation units, each translation unit gets its own copy of these statics (the `static` keyword makes them per-TU). The current single-worker-thread server architecture avoids races in practice, but this is an implicit assumption rather than an enforced invariant.

- Impact: if the server ever moves to a multi-worker or per-request thread model, backend ref-counting will race.
- Additional risk: the `static` keyword here means `backend_init` and `backend_release` in different TUs will not share the ref-count, potentially double-freeing or double-initializing backends.

---

## ggml Submodule Points to a Fork, Not Upstream

**File:** `.gitmodules`

```
url = https://github.com/ServeurpersoCom/ggml.git
```

The ggml dependency is a private fork (`ServeurpersoCom/ggml`) rather than the canonical `ggml-org/ggml`. The submodule HEAD (`8f8c8246`) contains project-specific patches (Metal `col2im_1d`, `snake` kernels, Vulkan dispatch fixes). These patches have not been upstreamed.

- Impact: cannot pull in upstream ggml security fixes or performance improvements without a manual rebase. The fork may silently diverge from `ggml-org/ggml` over time.
- Risk: if the fork repository becomes unavailable, the build cannot clone the dependency.
- Fix approach: upstream the custom ops or document the minimum required upstream ggml version.

---

## Duplicate Text Encoding Logic

**Files:** `src/pipeline-synth-ops.cpp:284-331` (primary pass), `src/pipeline-synth-ops.cpp:339-374` (non-cover pass)

The per-batch text encoding loop in `ops_encode_text` is duplicated almost verbatim (with only the instruction string changed) for the `need_enc_switch` path. Both blocks build `text_str`, `lyric_str`, tokenize, run `qwen3_forward`, `qwen3_embed_lookup`, and `cond_ggml_forward`. Any change to the encoding format must be applied in both places.

---

## Fragile JSON Parsing in `qwen3-lm.h`

**File:** `src/qwen3-lm.h:70-120` (`qw3lm_json_int`, `qw3lm_json_float`, `qw3lm_json_bool`)

The Qwen3 config parser uses manual `strstr`/`strchr`/`atoi` to extract values from `config.json`. This will silently return the fallback value if the key appears in an unexpected position (e.g., inside a comment, a nested object, or a string value). The `adapter_config.json` parser in `src/lora-merge.h:113-159` uses the same pattern.

- Impact: a mis-formatted config.json produces silent wrong-value behavior rather than an error.

---

## LoRA Alpha Fallback Silently Defaults to Rank

**File:** `src/lora-merge.h:419-428`

```cpp
} else {
    alpha = (float) rank;
}
```

When neither a per-tensor alpha (ComfyUI baked) nor a config alpha is found, the effective scaling factor becomes `(rank/rank) * scale = 1.0 * scale`. This matches some LoRA conventions but not others. There is no warning when this fallback activates. A user providing a PEFT adapter without `adapter_config.json` will get silently wrong-strength merges.

---

## MetadataFSM CoT Parse Failures Are Non-Fatal

**File:** `src/metadata-fsm.h:634`

```cpp
fprintf(stderr, "WARNING: batch %d CoT parse incomplete\n", i);
```

When the LM generates a chain-of-thought that the FSM cannot fully parse (e.g., the model hallucinates a non-standard field order), the warning is printed but generation continues with whatever partial metadata was extracted. The downstream `ops_encode_text` will use default/empty values for any unparsed field, silently producing music with incorrect metadata conditioning.

---

## Lack of Automated Tests

**Directory:** `tests/`

The test directory contains:
- Log files from manual validation runs (`CPU-BF16.log`, `CUDA0-Q4_K_M.log`, etc.)
- Debug scripts (`debug-dit-cossim.py`, `debug-lm-logits.py`)
- A single unit test: `tests/test-philox.cpp` (tests only the Philox RNG)

There are no automated integration tests, no golden-output tests for the synthesis pipeline, no regression tests for LoRA merge correctness, and no tests for the HTTP server endpoints. All validation has been performed manually.

- Impact: refactors to `dit-sampler.h`, `pipeline-synth-ops.cpp`, or `metadata-fsm.h` carry no safety net.

---

## Audio File Load Fully Into Memory Before Decoding

**File:** `src/audio-io.h:71-96` (`audio_io_load_file`)

The entire audio file is `malloc`'d into memory before passing to the MP3/WAV decoder. For the server use case where multipart HTTP bodies are already in memory this is fine, but for CLI tools (`ace-synth`, `ace-understand`) processing large WAV files (e.g., a 10-minute 48kHz stereo WAV = ~220 MB), this doubles peak memory usage: the raw file bytes and the decoded float PCM coexist in memory simultaneously.

---

## MP3 Encoder Filter Warmup Boundary Artifact

**File:** `src/audio-io.h:660-727`

The multi-threaded MP3 encoder uses `WARMUP_FRAMES = 3` to prime each thread's encoder state. The code comment acknowledges: "one frame of lost reservoir per boundary (~26ms of slightly lower quality)." This means multi-threaded encodes produce slightly lower quality at each chunk boundary compared to sequential encoding. With `n_threads = hardware_concurrency()` on a 16-core machine this produces 15 boundaries at ~26ms each.

---

## WAV File Write Ignores `fwrite` Return Value

**Files:** `src/audio-io.h:596`, `src/audio-io.h:808`, `src/debug.h:44-46`

`fwrite` return values are not checked. A partial disk write (out of space, I/O error) produces a silently truncated output file with a success return from `audio_write_wav` / `audio_write_mp3`.

---

## Scaling Concerns

**Audio duration ceiling (600 seconds):**
- `T <= 15000` at 25 Hz = 600 seconds maximum. Silence latent is fixed at 600s pre-loaded.

**Batch ceiling (9 tracks):**
- `batch_n` is validated at `[1, 9]`. The LM side allows up to 16 KV sets (`QW3LM_MAX_KV_SETS = 32` → 16 cond + 16 uncond), but the DiT side has no equivalent named constant.

**Single worker thread in ace-server:**
- All GPU compute is serialized. Concurrent HTTP requests queue in FIFO order. Long synthesis jobs (50-step base model, 30s audio) block all other requests for the duration.

---

*Concerns audit: 2026-04-18*
