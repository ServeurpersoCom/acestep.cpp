// test-model-store.cpp: exercise ModelStore with real GGUF loads.
//
// Runs the store in both policies and prints what's resident at each step.
// Serves as living documentation of the expected require / release flow
// and catches regressions in eviction or refcounting logic.
//
// Usage:
//   ./test-model-store --lm <gguf> --dit <gguf> --vae <gguf>
//
// All three paths are required. Uses small Q8 GGUFs in practice.

#include "model-store.h"
#include "version.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

static void dump(ModelStore * s, const char * tag) {
    fprintf(stderr, "[Test] %-24s modules=%d, vram=%.1f MB\n", tag, store_gpu_module_count(s),
            (float) store_vram_bytes(s) / (1024.0f * 1024.0f));
}

// Scenario 1: STRICT policy. Load three different modules in sequence,
// each should evict the previous. At no point should more than one GPU
// module be resident.
static int scenario_strict(const char * lm_path, const char * dit_path, const char * vae_path) {
    fprintf(stderr, "\n[Test] === scenario 1: STRICT ===\n");
    ModelStore * s = store_create(EVICT_STRICT);
    dump(s, "empty");

    ModelKey k_vae_enc = { MODEL_VAE_ENC, vae_path, 0, 0, "", 1.0f };
    auto *   vae_enc   = store_require_vae_enc(s, k_vae_enc);
    if (!vae_enc) {
        fprintf(stderr, "[Test] FAIL: VAE-Enc load\n");
        store_free(s);
        return 1;
    }
    dump(s, "after require VAE-Enc");
    if (store_gpu_module_count(s) != 1) {
        fprintf(stderr, "[Test] FAIL: expected 1 module, got %d\n", store_gpu_module_count(s));
        store_free(s);
        return 1;
    }
    store_release(s, vae_enc);
    dump(s, "after release VAE-Enc");
    if (store_gpu_module_count(s) != 0) {
        fprintf(stderr, "[Test] FAIL: STRICT should unload on release, got %d modules\n", store_gpu_module_count(s));
        store_free(s);
        return 1;
    }

    ModelKey k_fsq_tok = { MODEL_FSQ_TOK, dit_path, 0, 0, "", 1.0f };
    auto *   fsq_tok   = store_require_fsq_tok(s, k_fsq_tok);
    if (!fsq_tok) {
        fprintf(stderr, "[Test] FAIL: FSQ-Tok load\n");
        store_free(s);
        return 1;
    }
    dump(s, "after require FSQ-Tok");
    store_release(s, fsq_tok);
    dump(s, "after release FSQ-Tok");

    ModelKey k_lm = { MODEL_LM, lm_path, 8192, 1, "", 1.0f };
    auto *   lm   = store_require_lm(s, k_lm);
    if (!lm) {
        fprintf(stderr, "[Test] FAIL: LM load\n");
        store_free(s);
        return 1;
    }
    dump(s, "after require LM");
    if (store_gpu_module_count(s) != 1) {
        fprintf(stderr, "[Test] FAIL: expected 1 module, got %d\n", store_gpu_module_count(s));
        store_free(s);
        return 1;
    }
    store_release(s, lm);
    dump(s, "after release LM");

    store_free(s);
    fprintf(stderr, "[Test] scenario 1: PASS\n");
    return 0;
}

// Scenario 2: NEVER policy. Same three modules, should accumulate and stay.
static int scenario_never(const char * lm_path, const char * dit_path, const char * vae_path) {
    fprintf(stderr, "\n[Test] === scenario 2: NEVER ===\n");
    ModelStore * s = store_create(EVICT_NEVER);

    ModelKey k_vae_enc = { MODEL_VAE_ENC, vae_path, 0, 0, "", 1.0f };
    auto *   vae_enc   = store_require_vae_enc(s, k_vae_enc);
    store_release(s, vae_enc);
    dump(s, "after 1st release VAE-Enc");

    ModelKey k_fsq_tok = { MODEL_FSQ_TOK, dit_path, 0, 0, "", 1.0f };
    auto *   fsq_tok   = store_require_fsq_tok(s, k_fsq_tok);
    store_release(s, fsq_tok);
    dump(s, "after 1st release FSQ-Tok");

    ModelKey k_lm = { MODEL_LM, lm_path, 8192, 1, "", 1.0f };
    auto *   lm   = store_require_lm(s, k_lm);
    store_release(s, lm);
    dump(s, "after 1st release LM");

    if (store_gpu_module_count(s) != 3) {
        fprintf(stderr, "[Test] FAIL: NEVER should keep 3 modules, got %d\n", store_gpu_module_count(s));
        store_free(s);
        return 1;
    }

    // Second require on VAE-Enc: should cache-hit, no reload.
    auto * vae_enc_2 = store_require_vae_enc(s, k_vae_enc);
    if (vae_enc_2 != vae_enc) {
        fprintf(stderr, "[Test] FAIL: cache hit expected, got different pointer\n");
        store_free(s);
        return 1;
    }
    dump(s, "after 2nd require VAE-Enc");
    store_release(s, vae_enc_2);

    store_free(s);
    fprintf(stderr, "[Test] scenario 2: PASS\n");
    return 0;
}

// Scenario 3: CPU-resident modules are shared, never counted as GPU modules.
static int scenario_cpu(const char * lm_path, const char * dit_path) {
    fprintf(stderr, "\n[Test] === scenario 3: CPU-resident ===\n");
    ModelStore * s = store_create(EVICT_STRICT);

    auto * bpe1 = store_bpe(s, lm_path);
    auto * bpe2 = store_bpe(s, lm_path);
    if (bpe1 != bpe2) {
        fprintf(stderr, "[Test] FAIL: BPE not cached\n");
        store_free(s);
        return 1;
    }

    auto * silence1 = store_silence(s, dit_path);
    auto * silence2 = store_silence(s, dit_path);
    if (silence1 != silence2) {
        fprintf(stderr, "[Test] FAIL: silence not cached\n");
        store_free(s);
        return 1;
    }

    auto * meta = store_dit_meta(s, dit_path);
    if (!meta) {
        fprintf(stderr, "[Test] FAIL: DiT meta load\n");
        store_free(s);
        return 1;
    }
    fprintf(stderr, "[Test] DiT meta: hidden=%d, layers=%d, turbo=%d, null_cond=%zu\n", meta->cfg.hidden_size,
            meta->cfg.n_layers, meta->is_turbo, meta->null_cond_cpu.size());

    if (store_gpu_module_count(s) != 0) {
        fprintf(stderr, "[Test] FAIL: CPU accessors should not allocate GPU modules\n");
        store_free(s);
        return 1;
    }

    store_free(s);
    fprintf(stderr, "[Test] scenario 3: PASS\n");
    return 0;
}

int main(int argc, char ** argv) {
    fprintf(stderr, "acestep.cpp %s - test-model-store\n\n", ACE_VERSION);

    const char * lm_path  = nullptr;
    const char * dit_path = nullptr;
    const char * vae_path = nullptr;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--lm") && i + 1 < argc) {
            lm_path = argv[++i];
        } else if (!strcmp(argv[i], "--dit") && i + 1 < argc) {
            dit_path = argv[++i];
        } else if (!strcmp(argv[i], "--vae") && i + 1 < argc) {
            vae_path = argv[++i];
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return 1;
        }
    }

    if (!lm_path || !dit_path || !vae_path) {
        fprintf(stderr, "Usage: %s --lm <gguf> --dit <gguf> --vae <gguf>\n", argv[0]);
        return 1;
    }

    int rc = 0;
    rc |= scenario_strict(lm_path, dit_path, vae_path);
    rc |= scenario_never(lm_path, dit_path, vae_path);
    rc |= scenario_cpu(lm_path, dit_path);

    if (rc == 0) {
        fprintf(stderr, "\n[Test] ALL PASS\n");
    } else {
        fprintf(stderr, "\n[Test] FAIL\n");
    }
    return rc;
}
