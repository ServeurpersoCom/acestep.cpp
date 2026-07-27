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

// Batched Phase 1: N text generations. Prompts may either be shared (native
// variants from one request) or independent (heterogeneous HTTP batch).
// Each element gets its own KV set, FSM state and RNG.
// Returns N generated text strings.
static std::vector<std::string> generate_phase1_batch(Qwen3LM *                             m,
                                                      BPETokenizer *                        bpe,
                                                      const std::vector<std::vector<int>> & prompt_tokens,
                                                      int                                   max_new_tokens,
                                                      float                                 temperature,
                                                      float                                 top_p,
                                                      int                                   top_k,
                                                      const std::vector<uint32_t> &         seeds,
                                                      const std::vector<MetadataFSM> *      fsm_templates,
                                                      bool                                  lyrics_mode,
                                                      float                                 cfg_scale         = 1.0f,
                                                      const std::vector<std::vector<int>> * uncond_tokens     = nullptr,
                                                      bool                                  stop_at_reasoning = false,
                                                      bool                                  shared_prompt     = false,
                                                      bool (*cancel)(void *)                                  = nullptr,
                                                      void * cancel_data = nullptr) {
    int N = (int) prompt_tokens.size();
    if (N < 1 || (int) seeds.size() != N || (fsm_templates && (int) fsm_templates->size() != N)) {
        return {};
    }
    int  V       = m->cfg.vocab_size;
    bool use_cfg = cfg_scale > 1.0f && uncond_tokens && (int) uncond_tokens->size() == N;

    // KV sets: cond [0..N-1], uncond [N..2N-1] if CFG
    for (int i = 0; i < N; i++) {
        qw3lm_reset_kv(m, i);
    }
    if (use_cfg) {
        for (int i = 0; i < N; i++) {
            qw3lm_reset_kv(m, N + i);
        }
    }

    Timer                           t_prefill;
    std::vector<std::vector<float>> prefill_logits(N, std::vector<float>(V));
    qw3lm_forward(m, prompt_tokens[0].data(), (int) prompt_tokens[0].size(), 0, prefill_logits[0].data());
    for (int i = 1; i < N; i++) {
        if (shared_prompt) {
            qw3lm_copy_kv(m, 0, i);
            prefill_logits[i] = prefill_logits[0];
        } else {
            qw3lm_forward(m, prompt_tokens[i].data(), (int) prompt_tokens[i].size(), i, prefill_logits[i].data());
        }
    }

    std::vector<std::vector<float>> prefill_logits_uncond(N, std::vector<float>(V));
    if (use_cfg) {
        const auto & first_uncond = (*uncond_tokens)[0];
        qw3lm_forward(m, first_uncond.data(), (int) first_uncond.size(), N, prefill_logits_uncond[0].data());
        for (int i = 1; i < N; i++) {
            if (shared_prompt) {
                qw3lm_copy_kv(m, N, N + i);
                prefill_logits_uncond[i] = prefill_logits_uncond[0];
            } else {
                const auto & uncond = (*uncond_tokens)[i];
                qw3lm_forward(m, uncond.data(), (int) uncond.size(), N + i, prefill_logits_uncond[i].data());
            }
        }
    }

    fprintf(stderr, "[LM-Phase1] Prefill %.0fms, N=%d, CFG=%.2f, prompts=%s\n", t_prefill.ms(), N, cfg_scale,
            shared_prompt ? "shared" : "independent");

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
        seqs[i].rng.seed(seeds[i]);
        if (fsm_templates) {
            seqs[i].fsm = (*fsm_templates)[i];
        }
        seqs[i].codes_phase = false;
        seqs[i].done        = false;

        std::vector<float> lg(prefill_logits[i]);
        if (use_cfg) {
            for (int v = 0; v < V; v++) {
                lg[v] = prefill_logits_uncond[i][v] + cfg_scale * (lg[v] - prefill_logits_uncond[i][v]);
            }
        }
        if (fsm_templates && seqs[i].fsm.enabled) {
            seqs[i].fsm.apply_mask(lg.data());
        }

        int tok = sample_top_k_p(lg.data(), V, temperature, top_p, top_k, seqs[i].rng);

        if (tok == TOKEN_IM_END) {
            seqs[i].done = true;
        } else {
            if (fsm_templates && seqs[i].fsm.enabled) {
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
            if (fsm_templates && seqs[i].fsm.enabled && !seqs[i].codes_phase) {
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
        fprintf(stderr, "[LM-Phase1 Batch%d] seed=%u, %zu tokens\n", i, seeds[i], seqs[i].gen_tokens.size());
    }
    return results;
}

// Batched Phase 2: N sequences with potentially different prompts.
// aces.size() == N: each element gets its own lyrics/metadata.
// aces.size() == 1: single prompt replicated for all N (prefill once, copy KV).
// Returns N code strings. Each sequence uses the corresponding explicit seed.
static std::vector<std::string> run_phase2_batch(Qwen3LM *                      m,
                                                 BPETokenizer &                 bpe,
                                                 const std::vector<AcePrompt> & aces,
                                                 float                          temperature,
                                                 float                          top_p,
                                                 int                            top_k,
                                                 const std::vector<uint32_t> &  seeds,
                                                 float                          cfg_scale,
                                                 const char *                   negative_prompt,
                                                 bool                           use_batch_cfg,
                                                 bool                           shared_prompt,
                                                 bool (*cancel)(void *),
                                                 void * cancel_data) {
    int N = (int) aces.size();
    if (N < 1 || (int) seeds.size() != N) {
        return {};
    }
    int  V       = m->cfg.vocab_size;
    bool use_cfg = cfg_scale > 1.0f;

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
            unconds[i] = build_lm_prompt_uncond_with_cot(bpe, negative_prompt);
        }
        int mt = (int) (a.duration * 5) + 100;
        if (mt > max_tokens) {
            max_tokens = mt;
        }
    }
    fprintf(stderr, "[LM-Phase2] max_tokens: %d, CFG: %.2f, N=%d\n", max_tokens, cfg_scale, N);

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
        seqs[i].rng.seed(seeds[i]);
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
        fprintf(stderr, "[LM-Phase2 Batch%d] seed=%u, %zu codes\n", i, seeds[i], seqs[i].audio_codes.size());
    }
    return results;
}

// Public API

void ace_lm_default_params(AceLmParams * p) {
    p->model_path    = NULL;
    p->max_seq       = 8192;
    p->max_batch     = 1;
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

static AcePrompt request_to_prompt(const AceRequest & req) {
    AcePrompt ace      = {};
    ace.caption        = req.caption;
    ace.lyrics         = req.lyrics;
    ace.duration       = req.duration;
    ace.bpm            = req.bpm;
    ace.keyscale       = req.keyscale;
    ace.timesignature  = req.timesignature;
    ace.vocal_language = req.vocal_language;
    return ace;
}

static bool same_lm_batch_shape(const AceRequest & first, const AceRequest & req, int mode) {
    AcePrompt a                 = request_to_prompt(first);
    AcePrompt b                 = request_to_prompt(req);
    bool      first_has_codes   = !first.audio_codes.empty();
    bool      req_has_codes     = !req.audio_codes.empty();
    bool      first_need_lyrics = a.lyrics.empty();
    bool      req_need_lyrics   = b.lyrics.empty();
    bool      first_all_meta    = a.bpm > 0 && a.duration > 0 && !a.keyscale.empty() && !a.timesignature.empty();
    bool      req_all_meta      = b.bpm > 0 && b.duration > 0 && !b.keyscale.empty() && !b.timesignature.empty();

    return first.lm_model == req.lm_model && first.lm_temperature == req.lm_temperature &&
           first.lm_top_p == req.lm_top_p && first.lm_top_k == req.lm_top_k && first.lm_cfg_scale == req.lm_cfg_scale &&
           first.lm_negative_prompt == req.lm_negative_prompt && first.use_cot_caption == req.use_cot_caption &&
           first_has_codes == req_has_codes && first_need_lyrics == req_need_lyrics &&
           (first_need_lyrics || !first_all_meta) == (req_need_lyrics || !req_all_meta) &&
           (mode == LM_MODE_INSPIRE || mode == LM_MODE_FORMAT || first.lm_mode == req.lm_mode);
}

static int ace_lm_generate_impl(AceLm *                         ctx,
                                const std::vector<AceRequest> & requests,
                                AceRequest *                    out,
                                const char *                    dump_logits,
                                const char *                    dump_tokens,
                                bool                            shared_input,
                                bool (*cancel)(void *),
                                void * cancel_data,
                                int    mode) {
    int N = (int) requests.size();
    if (!ctx || !out || N < 1 || N > ctx->params.max_batch) {
        return -1;
    }
    for (int i = 0; i < N; i++) {
        if (requests[i].caption.empty()) {
            fprintf(stderr, "[Ace-LM] ERROR: caption is empty at batch index %d\n", i);
            return -1;
        }
        if (!same_lm_batch_shape(requests[0], requests[i], mode)) {
            fprintf(stderr, "[Ace-LM] ERROR: incompatible LM batch item at index %d\n", i);
            return -1;
        }
    }

    Qwen3LM * model = store_require_lm(ctx->store, ctx->lm_key);
    if (!model) {
        fprintf(stderr, "[Ace-LM] ERROR: store_require_lm failed\n");
        return -1;
    }
    ModelHandle lm_guard(ctx->store, model);
    if (!ctx->params.use_fa) {
        model->use_flash_attn = false;
    }
    model->clamp_fp16 = ctx->params.clamp_fp16;
    if (!model->lm_head_buf) {
        qw3lm_build_partial_head(model, TOKEN_IM_END);
    }

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

    Timer              t_total;
    const AceRequest & first       = requests[0];
    float              temperature = first.lm_temperature;
    float              top_p       = first.lm_top_p;
    int                top_k       = first.lm_top_k;
    float              cfg_scale   = first.lm_cfg_scale;
    const char *       neg_prompt  = first.lm_negative_prompt.c_str();

    std::vector<uint32_t>  lm_seeds(N);
    std::vector<AcePrompt> base_aces(N);
    for (int i = 0; i < N; i++) {
        lm_seeds[i]  = (uint32_t) requests[i].lm_seed;
        base_aces[i] = request_to_prompt(requests[i]);
    }

    bool user_has_codes = !first.audio_codes.empty();
    bool need_lyrics    = base_aces[0].lyrics.empty();
    bool has_all_metas  = base_aces[0].bpm > 0 && base_aces[0].duration > 0 && !base_aces[0].keyscale.empty() &&
                          !base_aces[0].timesignature.empty();
    bool need_fill      = need_lyrics || !has_all_metas;
    bool skip_codes     = mode == LM_MODE_INSPIRE || mode == LM_MODE_FORMAT;
    std::vector<AcePrompt> aces;

    if (user_has_codes && !skip_codes) {
        fprintf(stderr, "[LM-Generate] audio_codes present, skip LM\n");
        aces = base_aces;
    } else if (skip_codes || need_fill) {
        std::vector<std::vector<int>> prompts(N), unconds;
        bool                          gen_lyrics = need_lyrics || skip_codes;
        float                         fill_cfg   = (gen_lyrics || first.use_cot_caption) ? 1.0f : cfg_scale;
        if (fill_cfg > 1.0f) {
            unconds.resize(N);
        }
        std::vector<MetadataFSM> fsms;
        if (fsm_template) {
            fsms.resize(N);
        }

        for (int i = 0; i < N; i++) {
            const AceRequest & req = requests[i];
            const AcePrompt &  ace = base_aces[i];
            if (mode == LM_MODE_INSPIRE || (mode == LM_MODE_GENERATE && need_lyrics)) {
                std::string sys      = std::string("# Instruction\n") + LM_INSPIRE_INSTRUCTION + "\n";
                std::string user_msg = ace.caption;
                if (ace.lyrics == "[Instrumental]") {
                    user_msg += "\n\ninstrumental: true";
                }
                prompts[i] = build_custom_prompt(*bpe, sys.c_str(), user_msg.c_str());
            } else if (mode == LM_MODE_FORMAT) {
                std::string sys      = std::string("# Instruction\n") + LM_FORMAT_INSTRUCTION + "\n";
                std::string user_msg = "# Caption\n" + ace.caption + "\n\n# Lyric\n" + ace.lyrics;
                prompts[i]           = build_custom_prompt(*bpe, sys.c_str(), user_msg.c_str());
            } else {
                prompts[i] = build_lm_prompt(*bpe, ace);
            }
            if (fill_cfg > 1.0f) {
                unconds[i] = build_lm_prompt_uncond(*bpe, ace, neg_prompt);
            }
            if (fsm_template) {
                fsms[i] = *fsm_template;
                fsms[i].reset();
                fsms[i].skip_caption = !req.use_cot_caption && mode != LM_MODE_INSPIRE;
                if (ace.bpm > 0) {
                    fsms[i].force_field(*bpe, MetadataFSM::BPM_VALUE, std::to_string(ace.bpm));
                }
                if (ace.duration > 0) {
                    fsms[i].force_field(*bpe, MetadataFSM::DURATION_VALUE, std::to_string((int) ace.duration));
                }
                if (!ace.keyscale.empty()) {
                    fsms[i].force_field(*bpe, MetadataFSM::KEYSCALE_VALUE, ace.keyscale);
                }
                if (!ace.vocal_language.empty() && ace.vocal_language != "unknown") {
                    fsms[i].force_field(*bpe, MetadataFSM::LANGUAGE_VALUE, ace.vocal_language);
                }
                if (!ace.timesignature.empty()) {
                    fsms[i].force_field(*bpe, MetadataFSM::TIMESIG_VALUE, ace.timesignature);
                }
            }
        }

        const char * mode_name    = skip_codes ? (mode == LM_MODE_INSPIRE ? "inspire" : "format") : "fill";
        auto         phase1_texts = generate_phase1_batch(
            model, bpe, prompts, 2048, temperature, top_p, top_k, lm_seeds, fsm_template ? &fsms : nullptr, gen_lyrics,
            fill_cfg, unconds.empty() ? nullptr : &unconds, !gen_lyrics, shared_input, cancel, cancel_data);
        if ((int) phase1_texts.size() != N) {
            return -1;
        }
        aces.resize(N);
        for (int i = 0; i < N; i++) {
            AcePrompt                parse_base = mode == LM_MODE_INSPIRE ? AcePrompt{} : base_aces[i];
            std::vector<std::string> one_text{ phase1_texts[i] };
            std::vector<AcePrompt>   one_ace;
            bool                     parse_cot = mode == LM_MODE_INSPIRE ? true : requests[i].use_cot_caption;
            parse_phase1_into_aces(one_text, parse_base, one_ace, lm_seeds[i], mode_name, gen_lyrics, parse_cot);
            aces[i] = one_ace[0];
            if (aces[i].caption.empty()) {
                aces[i].caption = base_aces[i].caption;
            }
        }
        int n_kv_reset = fill_cfg > 1.0f ? 2 * N : N;
        for (int i = 0; i < n_kv_reset; i++) {
            qw3lm_reset_kv(model, i);
        }
    } else {
        aces = base_aces;
    }

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
            }
        }
        if (dump_logits) {
            std::vector<float> dbg_logits(model->cfg.vocab_size);
            qw3lm_forward(model, dbg_prompt.data(), (int) dbg_prompt.size(), 0, dbg_logits.data());
            FILE * f = fopen(dump_logits, "wb");
            if (f) {
                fwrite(dbg_logits.data(), sizeof(float), model->cfg.vocab_size, f);
                fclose(f);
            }
            qw3lm_reset_kv(model, 0);
        }
    }

    std::vector<std::string> batch_codes(N);
    if (skip_codes) {
        fprintf(stderr, "[LM-Generate] %s mode, no audio code generation\n",
                mode == LM_MODE_INSPIRE ? "Inspire" : "Format");
    } else if (!user_has_codes) {
        batch_codes = run_phase2_batch(model, *bpe, aces, temperature, top_p, top_k, lm_seeds, cfg_scale, neg_prompt,
                                       ctx->params.use_batch_cfg, shared_input && !need_fill, cancel, cancel_data);
        if ((int) batch_codes.size() != N) {
            return -1;
        }
    }

    for (int i = 0; i < N; i++) {
        out[i]                = requests[i];
        out[i].caption        = aces[i].caption;
        out[i].lyrics         = aces[i].lyrics;
        out[i].bpm            = aces[i].bpm;
        out[i].duration       = aces[i].duration;
        out[i].keyscale       = aces[i].keyscale;
        out[i].timesignature  = aces[i].timesignature;
        out[i].vocal_language = aces[i].vocal_language;
        if (!batch_codes[i].empty()) {
            out[i].audio_codes = batch_codes[i];
        }
        out[i].lm_batch_size = 1;
        if (mode == LM_MODE_INSPIRE) {
            out[i].use_cot_caption = true;
        }
    }
    fprintf(stderr, "[Ace-LM] Total %.0fms | heterogeneous=%s N=%d\n", t_total.ms(), shared_input ? "no" : "yes", N);
    return 0;
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
    if (!req || lm_batch_size < 1) {
        return -1;
    }
    AceRequest base = *req;
    request_resolve_lm_seed(&base);
    request_resolve_seed(&base);
    std::vector<AceRequest> expanded(lm_batch_size, base);
    for (int i = 0; i < lm_batch_size; i++) {
        expanded[i].lm_seed = base.lm_seed + i;
        expanded[i].seed    = base.seed + i;
    }
    return ace_lm_generate_impl(ctx, expanded, out, dump_logits, dump_tokens, true, cancel, cancel_data, mode);
}

int ace_lm_generate_batch(AceLm *            ctx,
                          const AceRequest * requests,
                          int                request_count,
                          AceRequest *       out,
                          bool (*cancel)(void *),
                          void * cancel_data,
                          int    mode) {
    if (!requests || request_count < 1) {
        return -1;
    }
    std::vector<AceRequest> batch(requests, requests + request_count);
    for (auto & req : batch) {
        request_resolve_lm_seed(&req);
        request_resolve_seed(&req);
        req.lm_batch_size = 1;
    }
    return ace_lm_generate_impl(ctx, batch, out, NULL, NULL, false, cancel, cancel_data, mode);
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
