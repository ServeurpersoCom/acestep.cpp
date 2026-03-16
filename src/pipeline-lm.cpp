// pipeline-lm.cpp: ACE-Step LM pipeline implementation
//
// Wraps Qwen3 LM for caption enrichment and audio code generation.

#include "pipeline-lm.h"

#include "bpe.h"
#include "metadata-fsm.h"
#include "prompt.h"
#include "qwen3-lm.h"
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
    Qwen3LM      model;
    BPETokenizer bpe;
    MetadataFSM  fsm;
    AceLmParams  params;
    double       load_ms;  // total load time (BPE + model + FSM)
};

struct TokenProb {
    int   id;
    float prob;
};

// Sampling: temperature -> top_k -> top_p -> softmax -> multinomial
// Matches nano-vLLM Sampler: div_(temperature) -> apply_top_k_top_p -> softmax -> sample
//
// Optimization: the caller compacts logits to EOS + audio codes (V=65536)
// instead of the full 150k+ vocab, so top_p sorts ~65k instead of ~150k.
// When top_k>0: nth_element O(V) finds the K-th value, mask everything below.
// When top_k=0 (disabled): no pre-filtering, top_p sees the full vocabulary.
// top_p: compact surviving tokens, sort descending O(K*log(K)), softmax,
// cumulative sum, mask tokens beyond the nucleus threshold.
// This matches nano-vLLM: apply_top_k_top_p sorts the full vocab when
// k is None, so we must not inject an artificial cap.
static int sample_top_k_p(float * logits, int V, float temperature, float top_p, int top_k, std::mt19937 & rng) {
    if (temperature <= 0.0f) {
        // greedy
        return (int) (std::max_element(logits, logits + V) - logits);
    }

    // Pre-allocated buffers (avoid malloc/free per call)
    static thread_local std::vector<float>     tmp_buf;
    static thread_local std::vector<TokenProb> sorted_buf;
    static thread_local std::vector<float>     probs_buf;

    // 1. temperature (matches nano-vLLM: logits.float().div_(temperatures))
    float inv_temp = 1.0f / temperature;
    for (int i = 0; i < V; i++) {
        logits[i] *= inv_temp;
    }

    // 2. top_k: keep top K values, set rest to -inf (skipped when top_k=0)
    if (top_k > 0 && top_k < V) {
        tmp_buf.resize(V);
        memcpy(tmp_buf.data(), logits, V * sizeof(float));
        std::nth_element(tmp_buf.begin(), tmp_buf.begin() + (top_k - 1), tmp_buf.end(), std::greater<float>());
        float threshold = tmp_buf[top_k - 1];
        for (int i = 0; i < V; i++) {
            if (logits[i] < threshold) {
                logits[i] = -INFINITY;
            }
        }
    }

    // 3. top_p: nucleus filter: only sort surviving (non-masked) tokens
    if (top_p > 0.0f && top_p < 1.0f) {
        // Compact: collect only finite logits (survived top_k)
        sorted_buf.clear();
        for (int i = 0; i < V; i++) {
            if (logits[i] > -1e30f) {
                sorted_buf.push_back({ i, logits[i] });
            }
        }

        int K = (int) sorted_buf.size();
        if (K > 0) {
            std::sort(sorted_buf.begin(), sorted_buf.end(),
                      [](const TokenProb & a, const TokenProb & b) { return a.prob > b.prob; });

            // softmax of temp-scaled logits for cumsum
            float max_val = sorted_buf[0].prob;
            float sum     = 0.0f;
            probs_buf.resize(K);
            for (int i = 0; i < K; i++) {
                probs_buf[i] = expf(sorted_buf[i].prob - max_val);
                sum += probs_buf[i];
            }
            float inv = 1.0f / sum;

            // cumulative sum, test before accumulating (shift-right trick)
            float cum = 0.0f;
            for (int i = 0; i < K; i++) {
                if (i > 0 && cum >= top_p) {
                    logits[sorted_buf[i].id] = -INFINITY;
                }
                cum += probs_buf[i] * inv;
            }
        }  // K > 0
    }

    // 4. softmax -> multinomial (only non-masked tokens matter)
    float max_val = -INFINITY;
    for (int i = 0; i < V; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
        }
    }
    float sum = 0.0f;
    for (int i = 0; i < V; i++) {
        logits[i] = expf(logits[i] - max_val);
        sum += logits[i];
    }

    std::uniform_real_distribution<float> dist(0.0f, sum);
    float                                 r   = dist(rng);
    float                                 acc = 0.0f;
    for (int i = 0; i < V; i++) {
        acc += logits[i];
        if (acc >= r) {
            return i;
        }
    }
    return 0;
}

// BPE decode (token IDs -> text)
static std::string bpe_decode(const BPETokenizer & bpe, const std::vector<int> & ids) {
    static std::unordered_map<int, uint8_t> byte_dec;
    static bool                             init = false;
    if (!init) {
        for (int b = 0; b < 256; b++) {
            int adv;
            int cp       = utf8_codepoint(bpe.byte2str[b].c_str(), &adv);
            byte_dec[cp] = (uint8_t) b;
        }
        init = true;
    }

    std::string result;
    for (int id : ids) {
        if (id == TOKEN_THINK) {
            result += "<think>";
            continue;
        }
        if (id == TOKEN_THINK_END) {
            result += "</think>";
            continue;
        }
        if (id == TOKEN_IM_START || id == TOKEN_IM_END) {
            continue;
        }
        if (id >= AUDIO_CODE_BASE) {
            continue;
        }
        if (id < 0 || id >= (int) bpe.id_to_str.size()) {
            continue;
        }
        const std::string & s = bpe.id_to_str[id];
        if (s.empty()) {
            continue;
        }
        const char * p = s.c_str();
        while (*p) {
            int  adv;
            int  cp = utf8_codepoint(p, &adv);
            auto it = byte_dec.find(cp);
            if (it != byte_dec.end()) {
                result += (char) it->second;
            }
            p += adv;
        }
    }
    return result;
}

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
                                                      long long                base_seed,
                                                      int                      N,
                                                      MetadataFSM *            fsm_template,
                                                      bool                     lyrics_mode,
                                                      float                    cfg_scale         = 1.0f,
                                                      const std::vector<int> * uncond_tokens     = nullptr,
                                                      bool                     stop_at_reasoning = false) {
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

    fprintf(stderr, "[Phase1] Prefill %.0fms, %zu tokens, N=%d, CFG=%.2f\n", t_prefill.ms(), prompt_tokens.size(), N,
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
        seqs[i].rng.seed((uint32_t) (base_seed + i));
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
            fprintf(stderr, "[Phase1] step %d, %d active, %.1f tok/s\n", step + 1, n_active,
                    (double) (step + 1) * N / elapsed);
        }
    }

    fprintf(stderr, "[Phase1] Decode %.0fms\n", t_decode.ms());

    // Decode tokens to text
    std::vector<std::string> results(N);
    for (int i = 0; i < N; i++) {
        results[i] = bpe_decode(*bpe, seqs[i].gen_tokens);
        fprintf(stderr, "[Phase1 Batch%d] seed=%lld, %zu tokens\n", i, base_seed + i, seqs[i].gen_tokens.size());
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
                                                 long long                      base_seed,
                                                 int                            N,
                                                 float                          cfg_scale,
                                                 const char *                   negative_prompt) {
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
            fprintf(stderr, "[Phase2] N=%d, CoT[0]:\n%s", N, cot.c_str());
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
    fprintf(stderr, "[Phase2] max_tokens: %d, CFG: %.2f, seeds: %lld..%lld\n", max_tokens, cfg_scale, base_seed,
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
    fprintf(stderr, "[Phase2] Prefill %.0fms (%s)\n", prefill_ms,
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
        seqs[i].rng.seed((uint32_t) (base_seed + i));
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

    // Batched decode loop, partial LM head: only project [TOKEN_IM_END..V)
    Timer t_decode;
    int   V_eff = V - TOKEN_IM_END;

    // Pre-allocate batched arrays for the maximum possible size (N or 2*N for CFG)
    int                max_N2 = use_cfg ? 2 * N : N;
    std::vector<int>   batch_tokens(max_N2);
    std::vector<int>   batch_sets(max_N2);
    std::vector<float> batch_logits((size_t) V_eff * max_N2);

    // This array maps the compact "active" index back to the original sequence index (0 to N-1)
    std::vector<int> active_to_orig(N);

    // Tiny array for CPU sampling (EOS token + Audio Codes) to prevent sorting 150,000 text logits
    int                audio_code_offset = AUDIO_CODE_BASE - TOKEN_IM_END;
    int                compact_V         = AUDIO_CODE_COUNT + 1;
    std::vector<float> compact_logits(compact_V);

    int n_active = N;
    for (int i = 0; i < N; i++) {
        if (seqs[i].done) {
            n_active--;
        }
    }

    for (int step = 0; step < max_tokens && n_active > 0; step++) {
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
        int actual_batch_size = use_cfg ? (2 * n_active) : n_active;
        qw3lm_forward_batch(m, batch_tokens.data(), batch_sets.data(), actual_batch_size, batch_logits.data(),
                            TOKEN_IM_END, V_eff);

        // 3. TARGETED CFG & LOGIT EXTRACTION
        for (int a = 0; a < n_active; a++) {
            int orig_i = active_to_orig[a];  // Map back to original sequence object

            // Pointer to the conditional logits for THIS active sequence
            float * lc = batch_logits.data() + (size_t) a * V_eff;

            if (use_cfg) {
                // Pointer to the unconditional logits (offset by n_active)
                float * lu = batch_logits.data() + (size_t) (n_active + a) * V_eff;

                // Targeted CFG Math: Only apply it to EOS + Audio Codes. Skip the 150,000 text tokens!
                lc[0] = lu[0] + cfg_scale * (lc[0] - lu[0]);  // EOS token
                for (int c = 0; c < AUDIO_CODE_COUNT; c++) {
                    int idx = audio_code_offset + c;
                    lc[idx] = lu[idx] + cfg_scale * (lc[idx] - lu[idx]);
                }
            }

            // Extract ONLY the valid target tokens into the tiny compact array
            compact_logits[0] = lc[0];
            for (int c = 0; c < AUDIO_CODE_COUNT; c++) {
                compact_logits[c + 1] = lc[audio_code_offset + c];
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
            fprintf(stderr, "[Decode] step %d, %d active, %d total codes, %.1f tok/s\n", step + 1, n_active,
                    total_codes, (double) (step + 1) * N / elapsed);
        }
    }

    double decode_ms = t_decode.ms();
    fprintf(stderr, "[Phase2] Decode %.0fms\n", decode_ms);

    // Build results
    std::vector<std::string> results(N);
    for (int i = 0; i < N; i++) {
        results[i] = codes_to_string(seqs[i].audio_codes);
        fprintf(stderr, "[Batch %d] seed=%lld, %zu codes\n", i, base_seed + i, seqs[i].audio_codes.size());
    }
    return results;
}

// Public API

void ace_lm_default_params(AceLmParams * p) {
    p->model_path = NULL;
    p->max_seq    = 8192;
    p->max_batch  = 4;
    p->use_fsm    = true;
    p->use_fa     = true;
}

AceLm * ace_lm_load(const AceLmParams * params) {
    if (!params->model_path) {
        fprintf(stderr, "[Ace-LM] ERROR: model_path is NULL\n");
        return NULL;
    }

    Timer   t_load;
    AceLm * ctx = new AceLm();
    ctx->params = *params;

    // Load BPE tokenizer from model GGUF
    if (!load_bpe_from_gguf(&ctx->bpe, params->model_path)) {
        fprintf(stderr, "[Ace-LM] ERROR: failed to load BPE from %s\n", params->model_path);
        delete ctx;
        return NULL;
    }

    // Allocate enough KV sets for worst case: CFG needs 2x batch
    int n_kv_sets = 2 * params->max_batch;

    if (!qw3lm_load(&ctx->model, params->model_path, params->max_seq, n_kv_sets)) {
        fprintf(stderr, "[Ace-LM] ERROR: failed to load model from %s\n", params->model_path);
        delete ctx;
        return NULL;
    }
    ctx->model.use_flash_attn = params->use_fa;

    // Init FSM (needs BPE + vocab size from loaded model)
    if (params->use_fsm) {
        ctx->fsm.init(ctx->bpe, ctx->model.cfg.vocab_size);
    }

    fprintf(stderr, "[Ace-LM] Loaded: vocab=%d, max_seq=%d, max_batch=%d, kv_sets=%d\n", ctx->model.cfg.vocab_size,
            params->max_seq, params->max_batch, n_kv_sets);

    ctx->load_ms = t_load.ms();
    return ctx;
}

int ace_lm_generate(AceLm *            ctx,
                    const AceRequest * req,
                    int                batch_size,
                    AceRequest *       out,
                    const char *       dump_logits,
                    const char *       dump_tokens) {
    if (!ctx || !req || !out || batch_size < 1) {
        return -1;
    }
    if (batch_size > ctx->params.max_batch) {
        fprintf(stderr, "[Ace-LM] ERROR: batch_size %d > max_batch %d\n", batch_size, ctx->params.max_batch);
        return -1;
    }
    if (req->caption.empty()) {
        fprintf(stderr, "[Ace-LM] ERROR: caption is empty\n");
        return -1;
    }

    Timer t_total;

    // Resolve seed
    long long seed = req->seed;
    if (seed < 0) {
        std::random_device rd;
        seed = (int64_t) rd() << 32 | rd();
        if (seed < 0) {
            seed = -seed;  // keep positive
        }
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

    std::vector<int>       prompt;
    std::vector<AcePrompt> aces;

    // ONE path: fill what's missing, then generate codes.
    // JSON is the instruction. Empty field = "fill it". Filled = "don't touch".
    if (user_has_codes) {
        fprintf(stderr, "[Pass] audio_codes present, skip LM\n");
    } else if (need_fill) {
        if (need_lyrics) {
            const char * sys =
                "# Instruction\n"
                "Expand the user's input into a more detailed"
                " and specific musical description:\n";
            std::string user_msg = ace.caption;
            prompt               = build_custom_prompt(ctx->bpe, sys, user_msg.c_str());
        } else {
            prompt = build_lm_prompt(ctx->bpe, ace);
        }
        std::vector<int> uncond;

        // Disable CFG for ANY textual expansion (lyrics OR CoT reasoning),
        // as CFG distorts text logits and forces premature newlines.
        float fill_cfg   = (need_lyrics || req->use_cot_caption) ? 1.0f : cfg_scale;
        float fill_top_p = top_p;
        int   fill_top_k = top_k;

        if (fill_cfg > 1.0f) {
            uncond = build_lm_prompt_uncond(ctx->bpe, ace, neg_prompt);
        }

        ctx->fsm.reset();
        MetadataFSM * active_fsm = nullptr;

        if (ctx->params.use_fsm) {
            if (need_lyrics) {
                // Free text for lyrics. Only use FSM if strictly forcing language.
                if (ace.vocal_language != "unknown" && !ace.vocal_language.empty()) {
                    ctx->fsm.force_language(ctx->bpe, ace.vocal_language);
                    active_fsm = &ctx->fsm;
                }
            } else {
                if (!req->use_cot_caption) {
                    active_fsm = &ctx->fsm;
                }
            }
        }

        fprintf(stderr, "[Fill] lyrics=%s metas=%s | %zu tokens, CFG: %.2f, N=%d\n", need_lyrics ? "generate" : "keep",
                has_all_metas ? "complete" : "fill gaps", prompt.size(), fill_cfg, batch_size);

        auto phase1_texts = generate_phase1_batch(&ctx->model, &ctx->bpe, prompt, 2048, temperature, fill_top_p,
                                                  fill_top_k, seed, batch_size, active_fsm, need_lyrics, fill_cfg,
                                                  uncond.empty() ? nullptr : &uncond, !need_lyrics);

        parse_phase1_into_aces(phase1_texts, ace, aces, seed, "Fill", need_lyrics, req->use_cot_caption);

        int n_kv_reset = (fill_cfg > 1.0f) ? 2 * batch_size : batch_size;
        for (int i = 0; i < n_kv_reset; i++) {
            qw3lm_reset_kv(&ctx->model, i);
        }
    }

    if (aces.empty()) {
        aces = { ace };
    }

    // Debug: dump tokens/logits
    if (!user_has_codes && (dump_logits || dump_tokens)) {
        std::string cot        = build_cot_yaml(aces[0]);
        auto        dbg_prompt = build_lm_prompt_with_cot(ctx->bpe, aces[0], cot);

        if (dump_tokens) {
            FILE * f = fopen(dump_tokens, "w");
            if (f) {
                for (size_t j = 0; j < dbg_prompt.size(); j++) {
                    fprintf(f, "%s%d", j ? "," : "", dbg_prompt[j]);
                }
                fprintf(f, "\n");
                fclose(f);
                fprintf(stderr, "[Debug] Tokens -> %s (%zu)\n", dump_tokens, dbg_prompt.size());
            }
        }
        if (dump_logits) {
            std::vector<float> dbg_logits(ctx->model.cfg.vocab_size);
            qw3lm_forward(&ctx->model, dbg_prompt.data(), (int) dbg_prompt.size(), 0, dbg_logits.data());
            FILE * f = fopen(dump_logits, "wb");
            if (f) {
                fwrite(dbg_logits.data(), sizeof(float), ctx->model.cfg.vocab_size, f);
                fclose(f);
                fprintf(stderr, "[Debug] Logits -> %s (%d floats, argmax=%d)\n", dump_logits, ctx->model.cfg.vocab_size,
                        (int) (std::max_element(dbg_logits.begin(), dbg_logits.end()) - dbg_logits.begin()));
            }
            qw3lm_reset_kv(&ctx->model, 0);
        }
    }

    // Phase 2: generate audio codes
    std::vector<std::string> batch_codes(batch_size);
    if (!user_has_codes) {
        batch_codes = run_phase2_batch(&ctx->model, ctx->bpe, aces, temperature, top_p, top_k, seed, batch_size,
                                       cfg_scale, neg_prompt);
    } else {
        fprintf(stderr, "[Skip] user audio_codes present, no code generation\n");
    }

    // Write N output requests
    for (int b = 0; b < batch_size; b++) {
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
        out[b].seed = seed + b;
    }

    fprintf(stderr, "[Ace-LM] Load %.0f | Total %.0fms | seed=%lld\n", ctx->load_ms, t_total.ms(), seed);
    return 0;
}

void ace_lm_free(AceLm * ctx) {
    if (!ctx) {
        return;
    }
    qw3lm_free(&ctx->model);
    delete ctx;
}
