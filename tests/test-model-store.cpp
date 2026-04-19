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
#include "pipeline-lm.h"
#include "pipeline-understand.h"
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
    fprintf(stderr, "[Test] scenario 1: STRICT\n");
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
    fprintf(stderr, "[Test] scenario 2: NEVER\n");
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
    fprintf(stderr, "[Test] scenario 3: CPU-resident\n");
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

// Scenario 4: LM sharing invariant. The whole point of the refactor is that
// ace-lm and ace-understand share one LM instance through the store. The
// identity of an LM key is (path, max_seq, n_kv_sets); adapter_path and
// adapter_scale are DiT-only extras and must NOT participate in the LM key.
// Regression guard: if a future change reintroduces adapter_* into the LM
// hash/eq, this test fails and the LM silently duplicates in VRAM under
// --keep-loaded.
static int scenario_lm_sharing(const char * lm_path) {
    fprintf(stderr, "[Test] scenario 4: LM sharing invariant\n");
    ModelStore * s = store_create(EVICT_NEVER);

    // Two keys that differ ONLY in adapter_* fields. These are outside the
    // LM key by design. Must return the same pointer, must not reload.
    ModelKey k_a = { MODEL_LM, lm_path, 8192, 2, "", 1.0f };
    ModelKey k_b = { MODEL_LM, lm_path, 8192, 2, "garbage-adapter-path", 99.0f };

    auto * lm_a = store_require_lm(s, k_a);
    if (!lm_a) {
        fprintf(stderr, "[Test] FAIL: LM load with k_a\n");
        store_free(s);
        return 1;
    }
    auto * lm_b = store_require_lm(s, k_b);
    if (lm_b != lm_a) {
        fprintf(stderr, "[Test] FAIL: LM sharing broken: adapter_* leaked into LM key\n");
        store_free(s);
        return 1;
    }
    if (store_gpu_module_count(s) != 1) {
        fprintf(stderr, "[Test] FAIL: expected 1 LM instance, got %d\n", store_gpu_module_count(s));
        store_free(s);
        return 1;
    }

    // A third key with a different max_seq is a different LM (max_seq IS part
    // of the key because the KV cache allocation depends on it).
    ModelKey k_c  = { MODEL_LM, lm_path, 4096, 2, "", 1.0f };
    auto *   lm_c = store_require_lm(s, k_c);
    if (lm_c == lm_a) {
        fprintf(stderr, "[Test] FAIL: max_seq difference must produce distinct LM\n");
        store_free(s);
        return 1;
    }
    if (store_gpu_module_count(s) != 2) {
        fprintf(stderr, "[Test] FAIL: expected 2 LM instances, got %d\n", store_gpu_module_count(s));
        store_free(s);
        return 1;
    }

    store_release(s, lm_a);
    store_release(s, lm_b);
    store_release(s, lm_c);
    store_free(s);
    fprintf(stderr, "[Test] scenario 4: PASS\n");
    return 0;
}

// Scenario 5: kind differentiation. The VAE GGUF holds both the encoder and
// the decoder; they share the same path but are different modules. The store
// must treat (kind, path) as distinct identities even when the path collides.
static int scenario_kind_split(const char * vae_path) {
    fprintf(stderr, "[Test] scenario 5: kind differentiation\n");
    ModelStore * s = store_create(EVICT_NEVER);

    ModelKey k_enc = { MODEL_VAE_ENC, vae_path, 0, 0, "", 1.0f };
    ModelKey k_dec = { MODEL_VAE_DEC, vae_path, 0, 0, "", 1.0f };

    auto * enc = store_require_vae_enc(s, k_enc);
    auto * dec = store_require_vae_dec(s, k_dec);
    if (!enc || !dec) {
        fprintf(stderr, "[Test] FAIL: VAE enc or dec load\n");
        store_free(s);
        return 1;
    }
    if ((void *) enc == (void *) dec) {
        fprintf(stderr, "[Test] FAIL: VAE enc and dec collapsed into one module\n");
        store_free(s);
        return 1;
    }
    if (store_gpu_module_count(s) != 2) {
        fprintf(stderr, "[Test] FAIL: expected 2 modules (enc + dec), got %d\n", store_gpu_module_count(s));
        store_free(s);
        return 1;
    }

    store_release(s, enc);
    store_release(s, dec);
    store_free(s);
    fprintf(stderr, "[Test] scenario 5: PASS\n");
    return 0;
}

// Scenario 6: refcount correctness. Nested require of the same key returns
// the same pointer and bumps the refcount. Under STRICT the module only
// unloads when the refcount drops to zero, not on the first release.
//
// Not tested here: STRICT require of a DIFFERENT kind while another module
// has refcount > 0 is a programming error and the store asserts. abort()
// cannot be safely caught in this binary, so the invariant lives as a
// comment in model-store.cpp (evict_all_except).
static int scenario_refcount(const char * vae_path) {
    fprintf(stderr, "[Test] scenario 6: refcount correctness\n");
    ModelStore * s = store_create(EVICT_STRICT);

    ModelKey k = { MODEL_VAE_ENC, vae_path, 0, 0, "", 1.0f };

    auto * p1 = store_require_vae_enc(s, k);
    auto * p2 = store_require_vae_enc(s, k);
    if (p1 != p2) {
        fprintf(stderr, "[Test] FAIL: nested require must return same pointer\n");
        store_free(s);
        return 1;
    }
    if (store_gpu_module_count(s) != 1) {
        fprintf(stderr, "[Test] FAIL: nested require must not load twice, got %d modules\n", store_gpu_module_count(s));
        store_free(s);
        return 1;
    }

    // First release drops rc from 2 to 1: under STRICT the module must stay
    // resident because someone still holds the second handle.
    store_release(s, p1);
    if (store_gpu_module_count(s) != 1) {
        fprintf(stderr, "[Test] FAIL: STRICT evicted module with rc>0, got %d modules\n", store_gpu_module_count(s));
        store_free(s);
        return 1;
    }

    // Second release drops rc to 0: STRICT unloads now.
    store_release(s, p2);
    if (store_gpu_module_count(s) != 0) {
        fprintf(stderr, "[Test] FAIL: STRICT kept module after last release, got %d modules\n",
                store_gpu_module_count(s));
        store_free(s);
        return 1;
    }

    store_free(s);
    fprintf(stderr, "[Test] scenario 6: PASS\n");
    return 0;
}

// Scenario 7: ace-server integration invariant.
//
// The whole shared-LM guarantee depends on ace_lm and ace_understand building
// byte-identical LM ModelKeys from user params. If a future flag is wired into
// AceLmParams but forgotten in the AceUnderstandParams mirror that ace-server
// maintains, the two contexts will build divergent keys and the store will
// quietly load the LM twice under --keep-loaded.
//
// We simulate the exact propagation block from ace-server main() and compare
// the two keys by hand. The test must fail loudly when keys drift.
static int scenario_integration(const char * lm_path) {
    fprintf(stderr, "[Test] scenario 7: ace-server integration\n");
    ModelStore * s = store_create(EVICT_NEVER);

    AceLmParams lm_p;
    ace_lm_default_params(&lm_p);
    lm_p.model_path = lm_path;
    lm_p.max_seq    = 4096;  // non-default, exercises the propagation path
    lm_p.max_batch  = 2;

    AceUnderstandParams und_p;
    ace_understand_default_params(&und_p);
    und_p.model_path = lm_path;
    // Replicate ace-server main() propagation. Every LM-key field must land here.
    und_p.max_seq   = lm_p.max_seq;
    und_p.max_batch = lm_p.max_batch;

    AceLm * lm = ace_lm_load(s, &lm_p);
    if (!lm) {
        fprintf(stderr, "[Test] FAIL: ace_lm_load\n");
        store_free(s);
        return 1;
    }
    AceUnderstand * und = ace_understand_load(s, &und_p);
    if (!und) {
        fprintf(stderr, "[Test] FAIL: ace_understand_load\n");
        ace_lm_free(lm);
        store_free(s);
        return 1;
    }

    const ModelKey * k_lm  = ace_lm_lm_key(lm);
    const ModelKey * k_und = ace_understand_lm_key(und);
    if (!k_lm || !k_und) {
        fprintf(stderr, "[Test] FAIL: lm_key accessor returned NULL\n");
        ace_understand_free(und);
        ace_lm_free(lm);
        store_free(s);
        return 1;
    }

    bool same = k_lm->kind == k_und->kind && k_lm->path == k_und->path && k_lm->max_seq == k_und->max_seq &&
                k_lm->n_kv_sets == k_und->n_kv_sets;
    if (!same) {
        fprintf(stderr, "[Test] FAIL: LM ModelKey divergence between pipelines\n");
        fprintf(stderr, "  ace-lm:         kind=%d max_seq=%d n_kv_sets=%d\n", (int) k_lm->kind, k_lm->max_seq,
                k_lm->n_kv_sets);
        fprintf(stderr, "  ace-understand: kind=%d max_seq=%d n_kv_sets=%d\n", (int) k_und->kind, k_und->max_seq,
                k_und->n_kv_sets);
        ace_understand_free(und);
        ace_lm_free(lm);
        store_free(s);
        return 1;
    }

    ace_understand_free(und);
    ace_lm_free(lm);
    store_free(s);
    fprintf(stderr, "[Test] scenario 7: PASS\n");
    return 0;
}

int main(int argc, char ** argv) {
    fprintf(stderr, "acestep.cpp %s - test-model-store\n", ACE_VERSION);

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
    rc |= scenario_lm_sharing(lm_path);
    rc |= scenario_kind_split(vae_path);
    rc |= scenario_refcount(vae_path);
    rc |= scenario_integration(lm_path);

    if (rc == 0) {
        fprintf(stderr, "[Test] ALL PASS\n");
    } else {
        fprintf(stderr, "[Test] FAIL\n");
    }
    return rc;
}
