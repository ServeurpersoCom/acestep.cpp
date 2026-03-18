#pragma once
// sampling.h: token sampling and BPE decoding shared by pipeline-lm and
// pipeline-understand.
//
// sample_top_k_p: temperature -> top_k -> top_p -> softmax -> multinomial.
// bpe_decode: token IDs -> text, skipping audio codes and special tokens.

#include "prompt.h"

#include <algorithm>
#include <cmath>
#include <random>

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

    // 3. top_p: nucleus filter.
    // the expensive part is sorting, not softmax. we compute the full-vocab
    // softmax first (two O(V) passes), then only sort the tokens above a
    // logit cutoff. the cumsum uses full-vocab-normalized probabilities, so
    // the nucleus boundary is identical to sorting all V tokens (vLLM).
    if (top_p > 0.0f && top_p < 1.0f) {
        // full-vocab softmax: exact probabilities for the cumsum.
        // tokens already masked to -inf by top_k contribute exp(-inf)=0.
        float max_logit = -INFINITY;
        for (int i = 0; i < V; i++) {
            if (logits[i] > max_logit) {
                max_logit = logits[i];
            }
        }
        float sum_exp = 0.0f;
        for (int i = 0; i < V; i++) {
            sum_exp += expf(logits[i] - max_logit);
        }
        float inv_sum = 1.0f / sum_exp;

        // compact only tokens worth sorting. a token at max-16 has relative
        // probability exp(-16) ~ 1e-7; even 65536 such tokens sum to < 0.7%
        // of the total mass. since their probabilities come from the full
        // softmax above, the cumsum reaches the same boundary as vLLM.
        float cutoff = max_logit - 16.0f;
        sorted_buf.clear();
        for (int i = 0; i < V; i++) {
            if (logits[i] >= cutoff) {
                float prob = expf(logits[i] - max_logit) * inv_sum;
                sorted_buf.push_back({ i, prob });
            } else {
                logits[i] = -INFINITY;
            }
        }

        int K = (int) sorted_buf.size();
        if (K > 0) {
            std::sort(sorted_buf.begin(), sorted_buf.end(),
                      [](const TokenProb & a, const TokenProb & b) { return a.prob > b.prob; });

            // cumsum with full-vocab-normalized probs (shift-right: test before accumulate)
            float cum = 0.0f;
            for (int i = 0; i < K; i++) {
                if (i > 0 && cum >= top_p) {
                    logits[sorted_buf[i].id] = -INFINITY;
                }
                cum += sorted_buf[i].prob;
            }
        }
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

// BPE decode (token IDs -> text).
// Skips audio code tokens, im_start/end. Expands think tags to text.
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
