// pipeline-lm.cpp: ACE-Step LM pipeline implementation
//
// Wraps Qwen3 LM for caption enrichment and audio code generation.

#include "pipeline-lm.h"

#include "bpe.h"
#include "metadata-fsm.h"
#include "model-store.h"
#include "prompt.h"
#include "qwen3-lm.h"
#include "sampling.h"
#include "timer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

struct AceLm {
    ModelStore * store;
    AceLmParams  params;
    ModelKey     lm_key;
};

// Batched Phase 1: N text generations with shared prompt, different seeds.
// No CFG. Each element gets its own FSM state and RNG.
// Returns N generated text strings.
static std::vector<std::string> generate_phase1_batch(Qwen3LM *                m,
                                                      BPETokenizer *           bpe,
                                                      const std::vector<int> & prompt_tokens,
                                                      int                      max_new_tokens,
                                                      float                    temperature,
                                                      float                    top_p,
                                                      int                      top_k,
                                                      uint32_t                 base_seed,
                                                      int                      N,
                                                      MetadataFSM *            fsm_template,
                                                      bool                     lyrics_mode,
                                                      float                    cfg_scale         = 1.0f,
                                                      const std::vector<int> * uncond_tokens     = nullptr,
                                                      bool                     stop_at_reasoning = false,
                                                      bool (*cancel)(void *)                     = nullptr,
                                                      void * cancel_data                         = nullptr) {
    int  V       = m->cfg.vocab_size;
    bool use_cfg = cfg_scale > 1.0f && uncond_tokens && !uncond_tokens->empty();

    // KV sets: cond [0..N-1], uncond [N..2N-1] if CFG
    for (int i = 0; i < N; i++) {
        qw3lm_reset_kv(m, i);
    }
    if (use_cfg) {
        for (int i = 0; i < N; i++) {
            qw3lm_reset_kv(m, N + i);
        }
    }

    // Prefill cond once, set 0, copy to 1..N-1
    Timer              t_prefill;
    std::vector<float> prefill_logits(V);
    qw3lm_forward(m, prompt_tokens.data(), (int) prompt_tokens.size(), 0, prefill_logits.data());
    for (int i = 1; i < N; i++) {
        qw3lm_copy_kv(m, 0, i);
    }

    // Prefill uncond once, set N, copy to N+1..2N-1
    std::vector<float> prefill_logits_uncond(V);
    if (use_cfg) {
        qw3lm_forward(m, uncond_tokens->data(), (int) uncond_tokens->size(), N, prefill_logits_uncond.data());
        for (int i = 1; i < N; i++) {
            qw3lm_copy_kv(m, N, N + i);
        }
    }

    fprintf(stderr, "[LM-Phase1] Prefill %.0fms, %zu tokens, N=%d, CFG=%.2f\n", t_prefill.ms(), prompt_tokens.size(), N,
            cfg_scale);

    // Per-element state
    struct P1Seq {
        std::mt19937     rng;
        MetadataFSM      fsm;
        std::vector<int> gen_tokens;
        int              last_token;
        bool             codes_phase;
        bool             done;
    };

    std::vector<P1Seq> seqs(N);

    // Sample first token from shared prefill logits
    for (int i = 0; i < N; i++) {
        seqs[i].rng.seed(base_seed + i);
        if (fsm_template) {
            seqs[i].fsm = *fsm_template;
        }
        seqs[i].codes_phase = false;
        seqs[i].done        = false;

        std::vector<float> lg(prefill_logits);
        if (use_cfg) {
            for (int v = 0; v < V; v++) {
                lg[v] = prefill_logits_uncond[v] + cfg_scale * (lg[v] - prefill_logits_uncond[v]);
            }
        }
        if (fsm_template && fsm_template->enabled) {
            seqs[i].fsm.apply_mask(lg.data());
        }

        int tok = sample_top_k_p(lg.data(), V, temperature, top_p, top_k, seqs[i].rng);

        if (tok == TOKEN_IM_END) {
            seqs[i].done = true;
        } else {
            if (fsm_template && fsm_template->enabled) {
                seqs[i].fsm.update(tok);
            }
            if (tok == TOKEN_THINK_END) {
                seqs[i].codes_phase = true;
                if (stop_at_reasoning) {
                    seqs[i].done = true;
                }
            }
            seqs[i].gen_tokens.push_back(tok);
        }
        seqs[i].last_token = tok;
    }

    // KV set arrays + merged CFG arrays
    std::vector<int> cond_sets(N), uncond_sets(N);
    for (int i = 0; i < N; i++) {
        cond_sets[i]   = i;
        uncond_sets[i] = N + i;
    }

    // Batched decode
    Timer              t_decode;
    std::vector<float> logits_cond(V * N);
    std::vector<float> logits_uncond(V * N);
    std::vector<int>   tokens(N);

    // CFG: single forward with 2*N (cond + uncond)
    int                N2 = use_cfg ? 2 * N : N;
    std::vector<int>   tokens_2n(N2), sets_2n(N2);
    std::vector<float> logits_2n((size_t) V * N2);
    if (use_cfg) {
        for (int i = 0; i < N; i++) {
            sets_2n[i]     = cond_sets[i];
            sets_2n[N + i] = uncond_sets[i];
        }
    }

    int n_active = N;
    for (int i = 0; i < N; i++) {
        if (seqs[i].done) {
            n_active--;
        }
    }

    for (int step = 0; step < max_new_tokens && n_active > 0; step++) {
        if (cancel && cancel(cancel_data)) {
            fprintf(stderr, "[LM-Phase1] Cancelled at step %d\n", step);
            return {};
        }
        for (int i = 0; i < N; i++) {
            tokens[i] = seqs[i].last_token;
        }

        if (use_cfg) {
            // Single batched forward: cond[0..N-1] + uncond[N..2N-1]
            for (int i = 0; i < N; i++) {
                tokens_2n[i]     = tokens[i];
                tokens_2n[N + i] = tokens[i];
            }
            qw3lm_forward_batch(m, tokens_2n.data(), sets_2n.data(), N2, logits_2n.data());
            memcpy(logits_cond.data(), logits_2n.data(), (size_t) V * N * sizeof(float));
            memcpy(logits_uncond.data(), logits_2n.data() + (size_t) V * N, (size_t) V * N * sizeof(float));
        } else {
            qw3lm_forward_batch(m, tokens.data(), cond_sets.data(), N, logits_cond.data());
        }

        for (int i = 0; i < N; i++) {
            if (seqs[i].done) {
                continue;
            }

            float * lc = logits_cond.data() + (size_t) i * V;

            // CFG combine
            if (use_cfg) {
                float * lu = logits_uncond.data() + (size_t) i * V;
                for (int v = 0; v < V; v++) {
                    lc[v] = lu[v] + cfg_scale * (lc[v] - lu[v]);
                }
            }

            // FSM mask (before </think>)
            if (fsm_template && seqs[i].fsm.enabled && !seqs[i].codes_phase) {
                seqs[i].fsm.apply_mask(lc);
            }

            // After </think>: audio code constraint unless lyrics_mode
            if (seqs[i].codes_phase && !lyrics_mode) {
                for (int v = TOKEN_IM_END + 1; v < AUDIO_CODE_BASE; v++) {
                    lc[v] = -1e9f;
                }
            }

            int tok;
            if (seqs[i].codes_phase && !lyrics_mode) {
                // Restricted sampling: only [TOKEN_IM_END..V)
                int V_eff = V - TOKEN_IM_END;
                tok = sample_top_k_p(lc + TOKEN_IM_END, V_eff, temperature, top_p, top_k, seqs[i].rng) + TOKEN_IM_END;
            } else {
                tok = sample_top_k_p(lc, V, temperature, top_p, top_k, seqs[i].rng);
            }

            if (tok == TOKEN_IM_END) {
                seqs[i].done = true;
                n_active--;
            } else {
                if (seqs[i].fsm.enabled && !seqs[i].codes_phase) {
                    seqs[i].fsm.update(tok);
                }
                if (tok == TOKEN_THINK_END && !seqs[i].codes_phase) {
                    seqs[i].codes_phase = true;
                    if (stop_at_reasoning) {
                        seqs[i].gen_tokens.push_back(tok);
                        seqs[i].done = true;
                        n_active--;
                        continue;
                    }
                }
                seqs[i].gen_tokens.push_back(tok);
            }
            seqs[i].last_token = tok;
        }

        if ((step + 1) % 100 == 0) {
            double elapsed = t_decode.ms() / 1000.0;
            fprintf(stderr, "[LM-Phase1] Step %d, %d active, %.1f tok/s\n", step + 1, n_active,
                    (double) (step + 1) * N / elapsed);
        }
    }

    fprintf(stderr, "[LM-Phase1] Decode %.0fms\n", t_decode.ms());

    // Decode tokens to text
    std::vector<std::string> results(N);
    for (int i = 0; i < N; i++) {
        results[i] = bpe_decode(*bpe, seqs[i].gen_tokens);
        fprintf(stderr, "[LM-Phase1 Batch%d] seed=%u, %zu tokens\n", i, base_seed + i, seqs[i].gen_tokens.size());
    }
    return results;
}

// Batched Phase 2: N sequences with potentially different prompts.
// aces.size() == N: each element gets its own lyrics/metadata.
// aces.size() == 1: single prompt replicated for all N (prefill once, copy KV).
// Returns N code strings. Seeds = base_seed + 0, 1, ..., N-1.
static std::vector<std::string> run_phase2_batch(Qwen3LM *                      m,
                                                 BPETokenizer &                 bpe,
                                                 const std::vector<AcePrompt> & aces,
                                                 float                          temperature,
                                                 float                          top_p,
                                                 int                            top_k,
                                                 uint32_t                       base_seed,
                                                 int                            N,
                                                 float                          cfg_scale,
                                                 const char *                   negative_prompt,
                                                 bool                           use_batch_cfg,
                                                 bool (*cancel)(void *),
                                                 void * cancel_data) {
    int  V             = m->cfg.vocab_size;
    bool use_cfg       = cfg_scale > 1.0f;
    bool shared_prompt = ((int) aces.size() == 1);

    // Build per-element prompts
    std::vector<std::vector<int>> prompts(N), unconds(N);
    int                           max_tokens = 0;
    for (int i = 0; i < N; i++) {
        const AcePrompt & a   = shared_prompt ? aces[0] : aces[i];
        std::string       cot = build_cot_yaml(a);
        if (i == 0) {
            fprintf(stderr, "[LM-Phase2] N=%d, CoT[0]:\n%s", N, cot.c_str());
        }
        prompts[i] = build_lm_prompt_with_cot(bpe, a, cot);
        if (use_cfg) {
            unconds[i] = build_lm_prompt_uncond_with_cot(bpe, a, negative_prompt);
        }
        int mt = (int) (a.duration * 5) + 100;
        if (mt > max_tokens) {
            max_tokens = mt;
        }
    }
    fprintf(stderr, "[LM-Phase2] max_tokens: %d, CFG: %.2f, seeds: %u..%u\n", max_tokens, cfg_scale, base_seed,
            base_seed + N - 1);

    // Reset all KV sets: cond [0..N-1], uncond [N..2N-1]
    for (int i = 0; i < N; i++) {
        qw3lm_reset_kv(m, i);
    }
    if (use_cfg) {
        for (int i = 0; i < N; i++) {
            qw3lm_reset_kv(m, N + i);
        }
    }

    // Prefill: if shared prompt, prefill once + copy KV. Otherwise prefill each.
    Timer                           t_prefill;
    std::vector<std::vector<float>> prefill_logits_vec(N, std::vector<float>(V));

    if (shared_prompt) {
        qw3lm_forward(m, prompts[0].data(), (int) prompts[0].size(), 0, prefill_logits_vec[0].data());
        for (int i = 1; i < N; i++) {
            qw3lm_copy_kv(m, 0, i);
            prefill_logits_vec[i] = prefill_logits_vec[0];
        }
    } else {
        for (int i = 0; i < N; i++) {
            qw3lm_forward(m, prompts[i].data(), (int) prompts[i].size(), i, prefill_logits_vec[i].data());
        }
    }

    // Prefill uncond
    std::vector<std::vector<float>> prefill_logits_uncond_vec(N, std::vector<float>(V));
    if (use_cfg) {
        if (shared_prompt) {
            qw3lm_forward(m, unconds[0].data(), (int) unconds[0].size(), N, prefill_logits_uncond_vec[0].data());
            for (int i = 1; i < N; i++) {
                qw3lm_copy_kv(m, N, N + i);
                prefill_logits_uncond_vec[i] = prefill_logits_uncond_vec[0];
            }
        } else {
            for (int i = 0; i < N; i++) {
                qw3lm_forward(m, unconds[i].data(), (int) unconds[i].size(), N + i,
                              prefill_logits_uncond_vec[i].data());
            }
        }
    }

    double prefill_ms = t_prefill.ms();
    fprintf(stderr, "[LM-Phase2] Prefill %.0fms (%s)\n", prefill_ms,
            shared_prompt ? "shared, 1 cond + 1 uncond" : "individual, N cond + N uncond");

    // Per-sequence state
    struct BatchSeq {
        std::mt19937     rng;
        std::vector<int> audio_codes;
        int              last_token;
        bool             done;
    };

    std::vector<BatchSeq> seqs(N);

    // Sample first token from per-element prefill logits (N different seeds)
    for (int i = 0; i < N; i++) {
        seqs[i].rng.seed(base_seed + i);
        seqs[i].done = false;

        std::vector<float> lg(prefill_logits_vec[i]);  // copy
        if (use_cfg) {
            float * lu = prefill_logits_uncond_vec[i].data();
            for (int v = 0; v < V; v++) {
                lg[v] = lu[v] + cfg_scale * (lg[v] - lu[v]);
            }
        }
        // Only audio codes + EOS (codes_phase = true from start)
        for (int v = 0; v < AUDIO_CODE_BASE; v++) {
            if (v != TOKEN_IM_END) {
                lg[v] = -1e9f;
            }
        }

        int tok            = sample_top_k_p(lg.data(), V, temperature, top_p, top_k, seqs[i].rng);
        seqs[i].last_token = tok;

        if (tok == TOKEN_IM_END) {
            seqs[i].done = true;
        } else if (tok >= AUDIO_CODE_BASE && tok < AUDIO_CODE_BASE + AUDIO_CODE_COUNT) {
            seqs[i].audio_codes.push_back(tok - AUDIO_CODE_BASE);
        }
    }

    // KV set arrays for batched forward
    std::vector<int> cond_sets(N), uncond_sets(N);
    for (int i = 0; i < N; i++) {
        cond_sets[i]   = i;
        uncond_sets[i] = N + i;
    }

    // Batched decode loop.
    // partial head: pre-extracted contiguous tensor for [TOKEN_IM_END..V) rows.
    // When unavailable (alloc failed): full vocab, slightly more compute, same result.
    Timer t_decode;
    bool  partial     = (m->lm_head_phase2 != NULL);
    int   out_V       = partial ? (V - TOKEN_IM_END) : V;
    int   lm_offset   = partial ? TOKEN_IM_END : 0;
    int   lm_count    = partial ? (V - TOKEN_IM_END) : 0;
    int   eos_idx     = partial ? 0 : TOKEN_IM_END;
    int   code_offset = partial ? (AUDIO_CODE_BASE - TOKEN_IM_END) : AUDIO_CODE_BASE;

    // Pre-allocate batched arrays for the maximum possible size (N or 2*N for CFG)
    int                max_N2 = use_cfg ? 2 * N : N;
    std::vector<int>   batch_tokens(max_N2);
    std::vector<int>   batch_sets(max_N2);
    std::vector<float> batch_logits((size_t) out_V * max_N2);

    // This array maps the compact "active" index back to the original sequence index (0 to N-1)
    std::vector<int> active_to_orig(N);

    // Tiny array for CPU sampling (EOS token + Audio Codes) to prevent sorting 150,000 text logits
    int                compact_V = AUDIO_CODE_COUNT + 1;
    std::vector<float> compact_logits(compact_V);

    int n_active = N;
    for (int i = 0; i < N; i++) {
        if (seqs[i].done) {
            n_active--;
        }
    }

    for (int step = 0; step < max_tokens && n_active > 0; step++) {
        if (cancel && cancel(cancel_data)) {
            fprintf(stderr, "[LM-Phase2] Cancelled at step %d\n", step);
            return {};
        }
        int current_active = 0;

        // 1. DYNAMIC COMPACTION: Loop through all N sequences, but only gather the active ones!
        for (int i = 0; i < N; i++) {
            if (!seqs[i].done) {
                active_to_orig[current_active] = i;  // Remember that this slot belongs to sequence 'i'

                if (use_cfg) {
                    // Place the Cond token/set in the first half
                    batch_tokens[current_active] = seqs[i].last_token;
                    batch_sets[current_active]   = cond_sets[i];

                    // Place the Uncond token/set exactly n_active elements later
                    batch_tokens[n_active + current_active] = seqs[i].last_token;
                    batch_sets[n_active + current_active]   = uncond_sets[i];
                } else {
                    batch_tokens[current_active] = seqs[i].last_token;
                    batch_sets[current_active]   = cond_sets[i];
                }
                current_active++;
            }
        }

        // 2. FORWARD PASS: GPU only computes attention for n_active sequences
        if (use_cfg && !use_batch_cfg) {
            // Two separate N=1 forwards (cond, then uncond).
            // Workaround for backends where batched multi-sequence attention
            // produces wrong results (e.g. ROCm/gfx1201). Same logit layout.
            qw3lm_forward_batch(m, batch_tokens.data(), batch_sets.data(), n_active, batch_logits.data(), lm_offset,
                                lm_count);
            qw3lm_forward_batch(m, batch_tokens.data() + n_active, batch_sets.data() + n_active, n_active,
                                batch_logits.data() + (size_t) n_active * out_V, lm_offset, lm_count);
        } else {
            int actual_batch_size = use_cfg ? (2 * n_active) : n_active;
            qw3lm_forward_batch(m, batch_tokens.data(), batch_sets.data(), actual_batch_size, batch_logits.data(),
                                lm_offset, lm_count);
        }

        // 3. TARGETED CFG & LOGIT EXTRACTION
        for (int a = 0; a < n_active; a++) {
            int orig_i = active_to_orig[a];  // Map back to original sequence object

            // Pointer to the conditional logits for THIS active sequence
            float * lc = batch_logits.data() + (size_t) a * out_V;

            if (use_cfg) {
                // Pointer to the unconditional logits (offset by n_active)
                float * lu = batch_logits.data() + (size_t) (n_active + a) * out_V;

                // Targeted CFG Math: Only apply it to EOS + Audio Codes. Skip the 150,000 text tokens!
                lc[eos_idx] = lu[eos_idx] + cfg_scale * (lc[eos_idx] - lu[eos_idx]);  // EOS token
                for (int c = 0; c < AUDIO_CODE_COUNT; c++) {
                    int idx = code_offset + c;
                    lc[idx] = lu[idx] + cfg_scale * (lc[idx] - lu[idx]);
                }
            }

            // Extract ONLY the valid target tokens into the tiny compact array
            compact_logits[0] = lc[eos_idx];
            for (int c = 0; c < AUDIO_CODE_COUNT; c++) {
                compact_logits[c + 1] = lc[code_offset + c];
            }

            // CPU samples instantly because it only has to sort ~2049 items instead of 150,000+
            int compact_tok =
                sample_top_k_p(compact_logits.data(), compact_V, temperature, top_p, top_k, seqs[orig_i].rng);

            // Map the sampled index back to global vocabulary ID
            int tok = (compact_tok == 0) ? TOKEN_IM_END : (AUDIO_CODE_BASE + compact_tok - 1);

            seqs[orig_i].last_token = tok;

            if (tok == TOKEN_IM_END) {
                seqs[orig_i].done = true;
            } else {
                seqs[orig_i].audio_codes.push_back(tok - AUDIO_CODE_BASE);
            }
        }

        // 4. UPDATE ACTIVE COUNT for the next loop iteration
        int next_active_count = 0;
        int total_codes       = 0;
        for (int i = 0; i < N; i++) {
            if (!seqs[i].done) {
                next_active_count++;
            }
            total_codes += (int) seqs[i].audio_codes.size();
        }
        n_active = next_active_count;

        if ((step + 1) % 50 == 0) {
            double elapsed = t_decode.ms() / 1000.0;
            fprintf(stderr, "[LM-Phase2] Step %d, %d active, %d total codes, %.1f tok/s\n", step + 1, n_active,
                    total_codes, (double) (step + 1) * N / elapsed);
        }
    }

    double decode_ms = t_decode.ms();
    fprintf(stderr, "[LM-Phase2] Decode %.0fms\n", decode_ms);

    // Build results
    std::vector<std::string> results(N);
    for (int i = 0; i < N; i++) {
        results[i] = codes_to_string(seqs[i].audio_codes);
        fprintf(stderr, "[LM-Phase2 Batch%d] seed=%u, %zu codes\n", i, base_seed + i, seqs[i].audio_codes.size());
    }
    return results;
}

// Public API

void ace_lm_default_params(AceLmParams * p) {
    p->model_path    = NULL;
    p->max_seq       = 8192;
    p->max_batch     = 4;
    p->use_fsm       = true;
    p->use_fa        = true;
    p->use_batch_cfg = true;
    p->clamp_fp16    = false;
}

AceLm * ace_lm_load(ModelStore * store, const AceLmParams * params) {
    if (!store || !params || !params->model_path) {
        fprintf(stderr, "[Ace-LM] ERROR: store and model_path are required\n");
        return NULL;
    }

    AceLm * ctx = new AceLm();
    ctx->store  = store;
    ctx->params = *params;

    // KV sets sized for worst case: CFG needs 2x batch.
    ctx->lm_key.kind          = MODEL_LM;
    ctx->lm_key.path          = params->model_path;
    ctx->lm_key.max_seq       = params->max_seq;
    ctx->lm_key.n_kv_sets     = 2 * params->max_batch;
    ctx->lm_key.adapter_path  = "";
    ctx->lm_key.adapter_scale = 1.0f;

    fprintf(stderr, "[Ace-LM] Ready: path=%s, max_seq=%d, max_batch=%d, fa=%s, fsm=%s\n", params->model_path,
            params->max_seq, params->max_batch, params->use_fa ? "yes" : "no", params->use_fsm ? "yes" : "no");
    if (!params->use_batch_cfg) {
        fprintf(stderr, "[Ace-LM] Batched CFG disabled (split N=1 forwards)\n");
    }
    if (params->clamp_fp16) {
        fprintf(stderr, "[Ace-LM] FP16 clamp enabled\n");
    }

    return ctx;
}

int ace_lm_generate(AceLm *            ctx,
                    const AceRequest * req,
                    int                lm_batch_size,
                    AceRequest *       out,
                    const char *       dump_logits,
                    const char *       dump_tokens,
                    bool (*cancel)(void *),
                    void * cancel_data,
                    int    mode) {
    if (!ctx || !req || !out || lm_batch_size < 1) {
        return -1;
    }
    if (lm_batch_size > ctx->params.max_batch) {
        fprintf(stderr, "[Ace-LM] ERROR: lm_batch_size %d > max_batch %d\n", lm_batch_size, ctx->params.max_batch);
        return -1;
    }
    if (req->caption.empty()) {
        fprintf(stderr, "[Ace-LM] ERROR: caption is empty\n");
        return -1;
    }

    // Acquire GPU LM from the store. RAII releases it on scope exit.
    Qwen3LM * model = store_require_lm(ctx->store, ctx->lm_key);
    if (!model) {
        fprintf(stderr, "[Ace-LM] ERROR: store_require_lm failed\n");
        return -1;
    }
    ModelHandle lm_guard(ctx->store, model);

    // Runtime flags: safe to set on every require (cache-hit or fresh load).
    if (!ctx->params.use_fa) {
        model->use_flash_attn = false;
    }
    model->clamp_fp16 = ctx->params.clamp_fp16;

    // Fresh load only: allocate the partial LM head for phase2 audio codes.
    // Contiguous GPU tensor instead of ggml_view_2d on quantized weights.
    // Cached on the model itself, freed by qw3lm_free when the store evicts.
    if (!model->lm_head_buf) {
        qw3lm_build_partial_head(model, TOKEN_IM_END);
    }

    // CPU-resident tokenizer and FSM template. Owned by the store, never
    // evicted. FSM must be copied before mutation since the template is shared.
    BPETokenizer * bpe = store_bpe(ctx->store, ctx->params.model_path);
    if (!bpe) {
        fprintf(stderr, "[Ace-LM] ERROR: store_bpe failed\n");
        return -1;
    }

    MetadataFSM * fsm_template = nullptr;
    if (ctx->params.use_fsm) {
        fsm_template = store_fsm(ctx->store, ctx->params.model_path, model->cfg.vocab_size);
        if (!fsm_template) {
            fprintf(stderr, "[Ace-LM] ERROR: store_fsm failed\n");
            return -1;
        }
    }

    // Local mutable FSM for this call. A copy is mandatory: force_field and
    // apply_mask mutate state that must not bleed across requests.
    MetadataFSM local_fsm;
    if (fsm_template) {
        local_fsm = *fsm_template;
    }

    Timer t_total;

    // LM RNG seed: always random (mt19937 uses 32 bits)
    std::random_device rd;
    uint32_t           seed = rd();

    // Resolve DiT seed (pass through to output for synth pipeline)
    long long dit_seed = req->seed;
    if (dit_seed < 0) {
        dit_seed = (int64_t) rd();
    }

    // Generation params from request
    float        temperature = req->lm_temperature;
    float        top_p       = req->lm_top_p;
    int          top_k       = req->lm_top_k;
    float        cfg_scale   = req->lm_cfg_scale;
    const char * neg_prompt  = req->lm_negative_prompt.c_str();

    // Copy request -> AcePrompt (internal LLM struct)
    AcePrompt ace      = {};
    ace.caption        = req->caption;
    ace.lyrics         = req->lyrics;
    ace.duration       = req->duration;
    ace.bpm            = req->bpm;
    ace.keyscale       = req->keyscale;
    ace.timesignature  = req->timesignature;
    ace.vocal_language = req->vocal_language;

    bool user_has_codes = !req->audio_codes.empty();
    bool need_lyrics    = ace.lyrics.empty();
    bool has_all_metas  = (ace.bpm > 0 && ace.duration > 0 && !ace.keyscale.empty() && !ace.timesignature.empty());
    bool need_fill      = need_lyrics || !has_all_metas;
    bool skip_codes     = (mode == LM_MODE_INSPIRE || mode == LM_MODE_FORMAT);

    std::vector<int>       prompt;
    std::vector<AcePrompt> aces;

    // ONE path: fill what's missing, then generate codes.
    // JSON is the instruction. Empty field = "fill it". Filled = "don't touch".
    if (user_has_codes && !skip_codes) {
        fprintf(stderr, "[LM-Generate] audio_codes present, skip LM\n");
    } else if (skip_codes || need_fill) {
        // inspire/format modes always run Phase 1 with their own instruction.
        // generate mode uses the inspire instruction when lyrics are empty.
        if (mode == LM_MODE_INSPIRE || (mode == LM_MODE_GENERATE && need_lyrics)) {
            std::string sys      = std::string("# Instruction\n") + LM_INSPIRE_INSTRUCTION + "\n";
            std::string user_msg = ace.caption;
            if (ace.lyrics == "[Instrumental]") {
                user_msg += "\n\ninstrumental: true";
            }
            prompt = build_custom_prompt(*bpe, sys.c_str(), user_msg.c_str());
        } else if (mode == LM_MODE_FORMAT) {
            std::string sys      = std::string("# Instruction\n") + LM_FORMAT_INSTRUCTION + "\n";
            std::string user_msg = "# Caption\n" + ace.caption + "\n\n# Lyric\n" + ace.lyrics;
            prompt               = build_custom_prompt(*bpe, sys.c_str(), user_msg.c_str());
        } else {
            prompt = build_lm_prompt(*bpe, ace);
        }
        std::vector<int> uncond;

        // inspire/format always generate lyrics. generate mode: only when lyrics are empty.
        bool gen_lyrics = need_lyrics || skip_codes;

        // Disable CFG for ANY textual expansion (lyrics OR CoT reasoning),
        // as CFG distorts text logits and forces premature newlines.
        float fill_cfg   = (gen_lyrics || req->use_cot_caption) ? 1.0f : cfg_scale;
        float fill_top_p = top_p;
        int   fill_top_k = top_k;

        if (fill_cfg > 1.0f) {
            uncond = build_lm_prompt_uncond(*bpe, ace, neg_prompt);
        }

        local_fsm.reset();
        MetadataFSM * active_fsm = nullptr;

        if (ctx->params.use_fsm) {
            // FSM constrains CoT metadata (bpm/dur/key/lang/tsig).
            // CAPTION_VALUE is free-form (only blocks audio codes).
            // Lyrics after </think> are unconstrained.
            // Force user-provided values into the KV cache so the LM
            // generates lyrics and codes conditioned on the right metadata.
            if (ace.bpm > 0) {
                local_fsm.force_field(*bpe, MetadataFSM::BPM_VALUE, std::to_string(ace.bpm));
            }
            if (ace.duration > 0) {
                local_fsm.force_field(*bpe, MetadataFSM::DURATION_VALUE, std::to_string((int) ace.duration));
            }
            if (!ace.keyscale.empty()) {
                local_fsm.force_field(*bpe, MetadataFSM::KEYSCALE_VALUE, ace.keyscale);
            }
            if (!ace.vocal_language.empty() && ace.vocal_language != "unknown") {
                local_fsm.force_field(*bpe, MetadataFSM::LANGUAGE_VALUE, ace.vocal_language);
            }
            if (!ace.timesignature.empty()) {
                local_fsm.force_field(*bpe, MetadataFSM::TIMESIG_VALUE, ace.timesignature);
            }
            active_fsm = &local_fsm;
        }

        const char * mode_name = skip_codes ? (mode == LM_MODE_INSPIRE ? "inspire" : "format") : "fill";
        fprintf(stderr, "[LM-Generate] mode=%s lyrics=%s metas=%s | %zu tokens, CFG: %.2f, N=%d\n", mode_name,
                gen_lyrics ? "generate" : "keep", has_all_metas ? "complete" : "fill gaps", prompt.size(), fill_cfg,
                lm_batch_size);

        auto phase1_texts = generate_phase1_batch(model, bpe, prompt, 2048, temperature, fill_top_p, fill_top_k, seed,
                                                  lm_batch_size, active_fsm, gen_lyrics, fill_cfg,
                                                  uncond.empty() ? nullptr : &uncond, !gen_lyrics, cancel, cancel_data);
        if (phase1_texts.empty()) {
            return -1;
        }

        // inspire mode: empty base so the LM output overwrites everything.
        // format/generate: gap fill, user metadata preserved.
        AcePrompt parse_base = (mode == LM_MODE_INSPIRE) ? AcePrompt{} : ace;
        parse_phase1_into_aces(phase1_texts, parse_base, aces, seed, mode_name, gen_lyrics, req->use_cot_caption);

        int n_kv_reset = (fill_cfg > 1.0f) ? 2 * lm_batch_size : lm_batch_size;
        for (int i = 0; i < n_kv_reset; i++) {
            qw3lm_reset_kv(model, i);
        }
    }

    if (aces.empty()) {
        aces = { ace };
    }

    // Debug: dump tokens/logits
    if (!user_has_codes && (dump_logits || dump_tokens)) {
        std::string cot        = build_cot_yaml(aces[0]);
        auto        dbg_prompt = build_lm_prompt_with_cot(*bpe, aces[0], cot);

        if (dump_tokens) {
            FILE * f = fopen(dump_tokens, "w");
            if (f) {
                for (size_t j = 0; j < dbg_prompt.size(); j++) {
                    fprintf(f, "%s%d", j ? "," : "", dbg_prompt[j]);
                }
                fprintf(f, "\n");
                fclose(f);
                fprintf(stderr, "[LM-Debug] Tokens -> %s (%zu)\n", dump_tokens, dbg_prompt.size());
            }
        }
        if (dump_logits) {
            std::vector<float> dbg_logits(model->cfg.vocab_size);
            qw3lm_forward(model, dbg_prompt.data(), (int) dbg_prompt.size(), 0, dbg_logits.data());
            FILE * f = fopen(dump_logits, "wb");
            if (f) {
                fwrite(dbg_logits.data(), sizeof(float), model->cfg.vocab_size, f);
                fclose(f);
                fprintf(stderr, "[LM-Debug] Logits -> %s (%d floats, argmax=%d)\n", dump_logits, model->cfg.vocab_size,
                        (int) (std::max_element(dbg_logits.begin(), dbg_logits.end()) - dbg_logits.begin()));
            }
            qw3lm_reset_kv(model, 0);
        }
    }

    // Phase 2: generate audio codes (skip for inspire/format modes)
    std::vector<std::string> batch_codes(lm_batch_size);
    if (skip_codes) {
        fprintf(stderr, "[LM-Generate] %s mode, no audio code generation\n",
                mode == LM_MODE_INSPIRE ? "Inspire" : "Format");
    } else if (!user_has_codes) {
        batch_codes = run_phase2_batch(model, *bpe, aces, temperature, top_p, top_k, seed, lm_batch_size, cfg_scale,
                                       neg_prompt, ctx->params.use_batch_cfg, cancel, cancel_data);
        if (batch_codes.empty()) {
            return -1;
        }
    } else {
        fprintf(stderr, "[LM-Generate] User audio_codes present, no code generation\n");
    }

    // Write N output requests
    for (int b = 0; b < lm_batch_size; b++) {
        out[b]                = *req;
        const AcePrompt & a   = aces[b < (int) aces.size() ? b : 0];
        out[b].caption        = a.caption;
        out[b].lyrics         = a.lyrics;
        out[b].bpm            = a.bpm;
        out[b].duration       = a.duration;
        out[b].keyscale       = a.keyscale;
        out[b].timesignature  = a.timesignature;
        out[b].vocal_language = a.vocal_language;
        if (!batch_codes[b].empty()) {
            out[b].audio_codes = batch_codes[b];
        }
        out[b].seed          = dit_seed + b;
        out[b].lm_batch_size = 1;  // each output is a standalone enriched request
    }

    fprintf(stderr, "[Ace-LM] Total %.0fms | seed=%lld\n", t_total.ms(), dit_seed);
    return 0;
}

void ace_lm_free(AceLm * ctx) {
    if (!ctx) {
        return;
    }
    delete ctx;
}

const ModelKey * ace_lm_lm_key(const AceLm * ctx) {
    return ctx ? &ctx->lm_key : nullptr;
}
