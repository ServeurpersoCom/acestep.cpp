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

// True when every prompt is the same token sequence. Identical prompts
// prefill once and share the KV set through copies; distinct prompts
// prefill each sequence individually.
static bool prompts_all_equal(const std::vector<std::vector<int>> & prompts) {
    for (size_t i = 1; i < prompts.size(); i++) {
        if (prompts[i] != prompts[0]) {
            return false;
        }
    }
    return true;
}

// Batched Phase 1: N text generations, one prompt and one seed per element.
// Identical prompts are detected and prefilled once with KV copies.
// Each element gets its own FSM state and RNG.
// Returns N generated text strings.
static std::vector<std::string> generate_phase1_batch(Qwen3LM *                             m,
                                                      BPETokenizer *                        bpe,
                                                      const std::vector<std::vector<int>> & prompts,
                                                      int                                   max_new_tokens,
                                                      float                                 temperature,
                                                      float                                 top_p,
                                                      int                                   top_k,
                                                      const std::vector<uint32_t> &         seeds,
                                                      std::vector<MetadataFSM> *            fsms,
                                                      bool                                  lyrics_mode,
                                                      float                                 cfg_scale         = 1.0f,
                                                      const std::vector<std::vector<int>> * unconds           = nullptr,
                                                      bool                                  stop_at_reasoning = false,
                                                      AceProgress                              progress       = {}) {
    int  N       = (int) prompts.size();
    int  V       = m->cfg.vocab_size;
    bool use_cfg = cfg_scale > 1.0f && unconds && !unconds->empty();

    // KV sets: cond [0..N-1], uncond [N..2N-1] if CFG
    for (int i = 0; i < N; i++) {
        qw3lm_reset_kv(m, i);
    }
    if (use_cfg) {
        for (int i = 0; i < N; i++) {
            qw3lm_reset_kv(m, N + i);
        }
    }

    // Prefill cond: shared prompt runs set 0 once and copies to 1..N-1,
    // distinct prompts prefill their own set.
    bool                            shared_prompt = prompts_all_equal(prompts);
    Timer                           t_prefill;
    std::vector<std::vector<float>> prefill_logits(shared_prompt ? 1 : N, std::vector<float>(V));
    qw3lm_forward(m, prompts[0].data(), (int) prompts[0].size(), 0, prefill_logits[0].data());
    for (int i = 1; i < N; i++) {
        if (shared_prompt) {
            qw3lm_copy_kv(m, 0, i);
        } else {
            qw3lm_forward(m, prompts[i].data(), (int) prompts[i].size(), i, prefill_logits[i].data());
        }
    }

    // Prefill uncond into sets N..2N-1, same shared/distinct rule.
    bool                            shared_uncond = use_cfg && prompts_all_equal(*unconds);
    std::vector<std::vector<float>> prefill_logits_uncond(use_cfg ? (shared_uncond ? 1 : N) : 0, std::vector<float>(V));
    if (use_cfg) {
        qw3lm_forward(m, (*unconds)[0].data(), (int) (*unconds)[0].size(), N, prefill_logits_uncond[0].data());
        for (int i = 1; i < N; i++) {
            if (shared_uncond) {
                qw3lm_copy_kv(m, N, N + i);
            } else {
                qw3lm_forward(m, (*unconds)[i].data(), (int) (*unconds)[i].size(), N + i,
                              prefill_logits_uncond[i].data());
            }
        }
    }

    fprintf(stderr, "[LM-Phase1] Prefill %.0fms, %zu tokens, N=%d, CFG=%.2f, prompts=%s\n", t_prefill.ms(),
            prompts[0].size(), N, cfg_scale, shared_prompt ? "shared" : "independent");

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

    // Sample first token from per-element prefill logits
    for (int i = 0; i < N; i++) {
        seqs[i].rng.seed(seeds[i]);
        if (fsms) {
            seqs[i].fsm = (*fsms)[i];
        }
        seqs[i].codes_phase = false;
        seqs[i].done        = false;

        std::vector<float> lg(prefill_logits[shared_prompt ? 0 : i]);
        if (use_cfg) {
            const float * lu = prefill_logits_uncond[shared_uncond ? 0 : i].data();
            for (int v = 0; v < V; v++) {
                lg[v] = lu[v] + cfg_scale * (lg[v] - lu[v]);
            }
        }
        if (fsms && seqs[i].fsm.enabled) {
            seqs[i].fsm.apply_mask(lg.data());
        }

        int tok = sample_top_k_p(lg.data(), V, temperature, top_p, top_k, seqs[i].rng);

        if (tok == TOKEN_IM_END) {
            seqs[i].done = true;
        } else {
            if (fsms && seqs[i].fsm.enabled) {
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

    if (ace_cancelled(progress,
                      ACE_STAGE_LM)) {  // honour a cancel before the loop; the loop's step-0 poll sizes the bar
        fprintf(stderr, "[LM-Phase1] Cancelled at step 0\n");
        return {};
    }
    for (int step = 0; step < max_new_tokens && n_active > 0; step++) {
        if (ace_progress(progress, ACE_STAGE_LM, step, max_new_tokens)) {
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
            if (fsms && seqs[i].fsm.enabled && !seqs[i].codes_phase) {
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

// Batched Phase 2: N sequences, one prompt and one seed per element.
// Identical prompts are detected and prefilled once with KV copies.
// Returns N code strings.
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
                                                 AceProgress                    progress) {
    int  N       = (int) aces.size();
    int  V       = m->cfg.vocab_size;
    bool use_cfg = cfg_scale > 1.0f;

    // Build per-element prompts
    std::vector<std::vector<int>> prompts(N), unconds(N);
    int                           max_tokens = 0;
    for (int i = 0; i < N; i++) {
        const AcePrompt & a   = aces[i];
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
    bool shared_prompt = prompts_all_equal(prompts);
    fprintf(stderr, "[LM-Phase2] max_tokens: %d, CFG: %.2f, N=%d, prompts=%s\n", max_tokens, cfg_scale, N,
            shared_prompt ? "shared" : "independent");

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

    if (ace_cancelled(progress,
                      ACE_STAGE_LM)) {  // honour a cancel before the loop; the loop's step-0 poll sizes the bar
        fprintf(stderr, "[LM-Phase2] Cancelled at step 0\n");
        return {};
    }
    for (int step = 0; step < max_tokens && n_active > 0; step++) {
        if (ace_progress(progress, ACE_STAGE_LM, step, max_tokens)) {
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

int ace_lm_generate(AceLm *            ctx,
                    const AceRequest * reqs,
                    int                n_req,
                    AceRequest *       out,
                    const char *       dump_logits,
                    const char *       dump_tokens,
                    AceProgress        progress,
                    int                mode) {
    if (!ctx || !reqs || !out || n_req < 1) {
        return -1;
    }
    if (n_req > ctx->params.max_batch) {
        fprintf(stderr, "[Ace-LM] ERROR: n_req %d > max_batch %d\n", n_req, ctx->params.max_batch);
        return -1;
    }
    const int          N     = n_req;
    const AceRequest & first = reqs[0];
    for (int i = 0; i < N; i++) {
        if (reqs[i].caption.empty()) {
            fprintf(stderr, "[Ace-LM] ERROR: caption is empty (batch %d)\n", i);
            return -1;
        }
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

    Timer t_total;

    // mt19937 consumes the low 32 bits of lm_seed (resolved by caller).
    std::vector<uint32_t> seeds(N);
    for (int i = 0; i < N; i++) {
        seeds[i] = (uint32_t) reqs[i].lm_seed;
    }

    // Generation params: sampling configuration comes from the first request
    // and applies to the whole batch.
    float        temperature = first.lm_temperature;
    float        top_p       = first.lm_top_p;
    int          top_k       = first.lm_top_k;
    float        cfg_scale   = first.lm_cfg_scale;
    const char * neg_prompt  = first.lm_negative_prompt.c_str();

    // Copy requests -> AcePrompts (internal LLM struct)
    std::vector<AcePrompt> base(N);
    for (int i = 0; i < N; i++) {
        base[i].caption        = reqs[i].caption;
        base[i].lyrics         = reqs[i].lyrics;
        base[i].duration       = reqs[i].duration;
        base[i].bpm            = reqs[i].bpm;
        base[i].keyscale       = reqs[i].keyscale;
        base[i].timesignature  = reqs[i].timesignature;
        base[i].vocal_language = reqs[i].vocal_language;
    }

    bool user_has_codes = !first.audio_codes.empty();
    bool need_lyrics    = base[0].lyrics.empty();
    bool has_all_metas =
        (base[0].bpm > 0 && base[0].duration > 0 && !base[0].keyscale.empty() && !base[0].timesignature.empty());
    bool need_fill  = need_lyrics || !has_all_metas;
    bool skip_codes = (mode == LM_MODE_INSPIRE || mode == LM_MODE_FORMAT);

    // These flags steer the whole batch, so every request must agree on them.
    // Sampling params are not compared: the first request wins.
    for (int i = 1; i < N; i++) {
        bool i_has_codes = !reqs[i].audio_codes.empty();
        bool i_lyrics    = base[i].lyrics.empty();
        bool i_all_metas =
            (base[i].bpm > 0 && base[i].duration > 0 && !base[i].keyscale.empty() && !base[i].timesignature.empty());
        if (i_has_codes != user_has_codes || i_lyrics != need_lyrics || (i_lyrics || !i_all_metas) != need_fill ||
            reqs[i].use_cot_caption != first.use_cot_caption) {
            fprintf(stderr, "[Ace-LM] ERROR: incompatible request shape (batch %d)\n", i);
            return -1;
        }
    }

    std::vector<AcePrompt> aces;

    // ONE path: fill what's missing, then generate codes.
    // JSON is the instruction. Empty field = "fill it". Filled = "don't touch".
    if (user_has_codes && !skip_codes) {
        fprintf(stderr, "[LM-Generate] audio_codes present, skip LM\n");
        aces = base;
    } else if (skip_codes || need_fill) {
        std::vector<std::vector<int>> prompts(N);
        std::vector<std::vector<int>> unconds;

        // inspire/format always generate lyrics. generate mode: only when lyrics are empty.
        bool gen_lyrics = need_lyrics || skip_codes;

        // Disable CFG for ANY textual expansion (lyrics OR CoT reasoning),
        // as CFG distorts text logits and forces premature newlines.
        float fill_cfg   = (gen_lyrics || first.use_cot_caption) ? 1.0f : cfg_scale;
        float fill_top_p = top_p;
        int   fill_top_k = top_k;

        if (fill_cfg > 1.0f) {
            unconds.resize(N);
        }

        // Local mutable FSMs for this call. A copy per request is mandatory:
        // force_field and apply_mask mutate state that must not bleed across
        // requests, and each request forces its own metadata.
        std::vector<MetadataFSM> fsms;
        if (fsm_template) {
            fsms.assign(N, *fsm_template);
        }

        for (int i = 0; i < N; i++) {
            const AcePrompt & ace = base[i];

            // inspire/format modes always run Phase 1 with their own instruction.
            // generate mode uses the inspire instruction when lyrics are empty.
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
                // FSM constrains CoT metadata (bpm/dur/key/lang/tsig).
                // CAPTION_VALUE is free-form (only blocks audio codes).
                // Lyrics after </think> are unconstrained.
                // Force user-provided values into the KV cache so the LM
                // generates lyrics and codes conditioned on the right metadata.
                MetadataFSM & fsm = fsms[i];
                fsm.reset();

                // Caption lock (use_cot_caption=false): skip the caption zone
                // in CoT, the user-provided caption stays untouched and the LM
                // sees it via the user prompt block.
                // Inspire mode regenerates the caption from scratch, so the
                // lock is ignored there.
                fsm.skip_caption = !first.use_cot_caption && (mode != LM_MODE_INSPIRE);

                if (ace.bpm > 0) {
                    fsm.force_field(*bpe, MetadataFSM::BPM_VALUE, std::to_string(ace.bpm));
                }
                if (ace.duration > 0) {
                    fsm.force_field(*bpe, MetadataFSM::DURATION_VALUE, std::to_string((int) ace.duration));
                }
                if (!ace.keyscale.empty()) {
                    fsm.force_field(*bpe, MetadataFSM::KEYSCALE_VALUE, ace.keyscale);
                }
                if (!ace.vocal_language.empty() && ace.vocal_language != "unknown") {
                    fsm.force_field(*bpe, MetadataFSM::LANGUAGE_VALUE, ace.vocal_language);
                }
                if (!ace.timesignature.empty()) {
                    fsm.force_field(*bpe, MetadataFSM::TIMESIG_VALUE, ace.timesignature);
                }
            }
        }

        const char * mode_name = skip_codes ? (mode == LM_MODE_INSPIRE ? "inspire" : "format") : "fill";
        fprintf(stderr, "[LM-Generate] mode=%s lyrics=%s metas=%s | %zu tokens, CFG: %.2f, N=%d\n", mode_name,
                gen_lyrics ? "generate" : "keep", has_all_metas ? "complete" : "fill gaps", prompts[0].size(), fill_cfg,
                N);

        auto phase1_texts = generate_phase1_batch(
            model, bpe, prompts, 2048, temperature, fill_top_p, fill_top_k, seeds, fsm_template ? &fsms : nullptr,
            gen_lyrics, fill_cfg, unconds.empty() ? nullptr : &unconds, !gen_lyrics, progress);
        if (phase1_texts.empty()) {
            return -1;
        }

        // inspire mode: empty bases so the LM output overwrites everything.
        // format/generate: gap fill, user metadata preserved.
        std::vector<AcePrompt> parse_bases(N);
        if (mode != LM_MODE_INSPIRE) {
            parse_bases = base;
        }

        // Inspire ignores the caption lock end to end: sampling regenerates
        // and parsing accepts. Other modes honor the user flag.
        bool parse_use_cot_caption = (mode == LM_MODE_INSPIRE) ? true : first.use_cot_caption;
        parse_phase1_into_aces(phase1_texts, parse_bases, aces, seeds, mode_name, gen_lyrics, parse_use_cot_caption);

        // Caption preservation: the LM may enrich the user caption, but
        // never silently delete it. If the merge ended up with an empty
        // caption while the request had one, restore the request value.
        for (int i = 0; i < N; i++) {
            if (aces[i].caption.empty() && !base[i].caption.empty()) {
                aces[i].caption = base[i].caption;
            }
        }

        int n_kv_reset = (fill_cfg > 1.0f) ? 2 * N : N;
        for (int i = 0; i < n_kv_reset; i++) {
            qw3lm_reset_kv(model, i);
        }
    } else {
        aces = base;
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
    std::vector<std::string> batch_codes(N);
    if (skip_codes) {
        fprintf(stderr, "[LM-Generate] %s mode, no audio code generation\n",
                mode == LM_MODE_INSPIRE ? "Inspire" : "Format");
    } else if (!user_has_codes) {
        batch_codes = run_phase2_batch(model, *bpe, aces, temperature, top_p, top_k, seeds, cfg_scale, neg_prompt,
                                       ctx->params.use_batch_cfg, progress);
        if (batch_codes.empty()) {
            return -1;
        }
    } else {
        fprintf(stderr, "[LM-Generate] User audio_codes present, no code generation\n");
    }

    // Write N output requests. seed and lm_seed pass through from the input
    // requests (resolved by caller).
    for (int b = 0; b < N; b++) {
        out[b]                = reqs[b];
        const AcePrompt & a   = aces[b];
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
        out[b].lm_batch_size = 1;  // each output is a standalone enriched request

        // Backend-driven flag: report what was actually applied. Inspire
        // always regenerates the caption regardless of the input lock,
        // so the response reflects the effective state and the UI toggle
        // updates itself to match reality.
        if (mode == LM_MODE_INSPIRE) {
            out[b].use_cot_caption = true;
        }
    }

    fprintf(stderr, "[Ace-LM] Total %.0fms | N=%d seed=%lld\n", t_total.ms(), N, (long long) first.seed);
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
