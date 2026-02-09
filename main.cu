// ace-cuda: Minimal standalone Qwen3 inference engine
// Loads safetensors, runs forward pass with cuBLAS+custom kernels
// bf16 storage, fp32 compute. No PyTorch, no GGML, no deps.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include "bpe.h"
#include "safetensors.h"
#include "kernels.cuh"

// Timer for phase tracking
struct Timer {
    std::chrono::steady_clock::time_point t;
    Timer() : t(std::chrono::steady_clock::now()) {}
    double ms() const {
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t).count();
    }
};

// Model config (matches Qwen3 / ACE-Step LM)
struct QwenConfig {
    int vocab_size;
    int hidden_size;
    int intermediate_size;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    int max_seq_len;         // max context for KV cache
    float rope_theta;
    float rms_norm_eps;
    bool tie_embeddings;
};

// ACE-Step 5Hz LM 4B defaults
static QwenConfig default_config() {
    return {
        .vocab_size       = 217204,
        .hidden_size      = 2560,
        .intermediate_size = 9728,
        .n_layers         = 36,
        .n_heads          = 32,
        .n_kv_heads       = 8,
        .head_dim         = 128,
        .max_seq_len      = 8192,
        .rope_theta       = 1000000.0f,
        .rms_norm_eps     = 1e-6f,
        .tie_embeddings   = true,
    };
}

// Layer weights (all bf16, on GPU)
struct QwenLayer {
    bf16 *input_layernorm;       // [hidden_size]
    bf16 *q_proj;                // [n_heads*head_dim, hidden_size]
    bf16 *k_proj;                // [n_kv_heads*head_dim, hidden_size]
    bf16 *v_proj;                // [n_kv_heads*head_dim, hidden_size]
    bf16 *q_norm;                // [head_dim]
    bf16 *k_norm;                // [head_dim]
    bf16 *o_proj;                // [hidden_size, n_heads*head_dim]
    bf16 *post_attn_layernorm;   // [hidden_size]
    bf16 *gate_proj;             // [intermediate_size, hidden_size]
    bf16 *up_proj;               // [intermediate_size, hidden_size]
    bf16 *down_proj;             // [hidden_size, intermediate_size]
};

// Full model state (weights + KV cache + scratch buffers)
#define MAX_LAYERS 64

struct QwenModel {
    QwenConfig cfg;

    // Weights
    bf16 *embed_tokens;             // [vocab_size, hidden_size]
    QwenLayer layers[MAX_LAYERS];
    bf16 *final_norm;               // [hidden_size]
    bf16 *lm_head;                  // = embed_tokens if tied, else separate

    // KV cache: [n_layers][2][max_seq_len][kv_dim]  (0=K, 1=V)
    bf16 *kv_cache;
    bf16 *kv_cache_uncond;          // second KV cache for CFG unconditional path
    int kv_dim;                     // n_kv_heads * head_dim

    // Scratch buffers
    bf16 *buf_x;                    // [hidden_size]
    bf16 *buf_res;                  // [hidden_size]
    bf16 *buf_q;                    // [n_heads * head_dim]
    bf16 *buf_k;                    // [n_kv_heads * head_dim]
    bf16 *buf_v;                    // [n_kv_heads * head_dim]
    bf16 *buf_attn;                 // [n_heads * head_dim]
    bf16 *buf_gate;                 // [intermediate_size]
    bf16 *buf_up;                   // [intermediate_size]
    float *buf_logits_f32;          // [vocab_size]

    cublasHandle_t cublas;
};

// cuBLAS linear layer: y = W @ x  (W stored row-major [out, in])
// For decode: x is [in_dim, 1], y is [out_dim, 1]
static void linear(bf16 *y, const bf16 *x, const bf16 *W,
                   int out_dim, int in_dim, cublasHandle_t handle) {
    float alpha = 1.0f, beta = 0.0f;
    // W in memory: [out_dim, in_dim] row-major = [in_dim, out_dim] col-major
    // We want y = W @ x, so: op(A)=A^T=[out_dim,in_dim], B=x[in_dim,1], C=y[out_dim,1]
    cublasStatus_t st = cublasGemmEx(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        out_dim, 1, in_dim,
        &alpha,
        W, CUDA_R_16BF, in_dim,
        x, CUDA_R_16BF, in_dim,
        &beta,
        y, CUDA_R_16BF, out_dim,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error: %d (out=%d, in=%d)\n", st, out_dim, in_dim);
    }
}

// Linear but output to fp32 (for logits)
static void linear_f32(float *y, const bf16 *x, const bf16 *W,
                       int out_dim, int in_dim, cublasHandle_t handle) {
    float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        out_dim, 1, in_dim,
        &alpha,
        W, CUDA_R_16BF, in_dim,
        x, CUDA_R_16BF, in_dim,
        &beta,
        y, CUDA_R_32F, out_dim,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// Upload tensor from safetensors (host mmap) to GPU
static bf16 *upload_tensor(const SafeTensors &st, const std::string &name) {
    const SafeTensor &t = safe_get(st, name);
    bf16 *gpu;
    CUDA_CHECK(cudaMalloc(&gpu, t.nbytes));
    CUDA_CHECK(cudaMemcpy(gpu, t.data, t.nbytes, cudaMemcpyHostToDevice));
    return gpu;
}

// Parse config.json from model directory (auto-detect model dims)
// Falls back to default_config() if parsing fails.
static int json_int(const char *json, const char *key, int fallback) {
    char needle[128];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char *p = strstr(json, needle);
    if (!p) return fallback;
    p = strchr(p + strlen(needle), ':');
    if (!p) return fallback;
    p++;
    while (*p == ' ' || *p == '\t') p++;
    return atoi(p);
}

static float json_float(const char *json, const char *key, float fallback) {
    char needle[128];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char *p = strstr(json, needle);
    if (!p) return fallback;
    p = strchr(p + strlen(needle), ':');
    if (!p) return fallback;
    p++;
    while (*p == ' ' || *p == '\t') p++;
    return (float)atof(p);
}

static bool json_bool(const char *json, const char *key, bool fallback) {
    char needle[128];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char *p = strstr(json, needle);
    if (!p) return fallback;
    p = strchr(p + strlen(needle), ':');
    if (!p) return fallback;
    p++;
    while (*p == ' ' || *p == '\t') p++;
    return (strncmp(p, "true", 4) == 0);
}

static QwenConfig load_config(const char *model_dir) {
    QwenConfig cfg = default_config();
    char path[512];
    snprintf(path, sizeof(path), "%s/config.json", model_dir);

    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "[Config] No config.json found, using defaults\n");
        return cfg;
    }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<char> buf(len + 1, 0);
    fread(buf.data(), 1, len, f);
    fclose(f);

    const char *json = buf.data();
    cfg.vocab_size       = json_int(json, "vocab_size", cfg.vocab_size);
    cfg.hidden_size      = json_int(json, "hidden_size", cfg.hidden_size);
    cfg.intermediate_size = json_int(json, "intermediate_size", cfg.intermediate_size);
    cfg.n_layers         = json_int(json, "num_hidden_layers", cfg.n_layers);
    cfg.n_heads          = json_int(json, "num_attention_heads", cfg.n_heads);
    cfg.n_kv_heads       = json_int(json, "num_key_value_heads", cfg.n_kv_heads);
    cfg.head_dim         = json_int(json, "head_dim", cfg.head_dim);
    cfg.rope_theta       = json_float(json, "rope_theta", cfg.rope_theta);
    cfg.rms_norm_eps     = json_float(json, "rms_norm_eps", cfg.rms_norm_eps);
    cfg.tie_embeddings   = json_bool(json, "tie_word_embeddings", cfg.tie_embeddings);

    return cfg;
}

// Load full Qwen3 model from safetensors directory
static void load_model(QwenModel *m, const char *model_dir, QwenConfig cfg) {
    m->cfg = cfg;
    m->kv_dim = cfg.n_kv_heads * cfg.head_dim;

    fprintf(stderr, "[Load] Qwen3: %dL, hidden=%d, heads=%d/%d, vocab=%d\n",
            cfg.n_layers, cfg.hidden_size, cfg.n_heads, cfg.n_kv_heads, cfg.vocab_size);

    SafeTensors st;
    if (!safe_load(st, model_dir)) {
        fprintf(stderr, "FATAL: failed to load safetensors from %s\n", model_dir);
        exit(1);
    }

    // Upload weights
    Timer t_weights;

    m->embed_tokens = upload_tensor(st, "model.embed_tokens.weight");
    m->final_norm   = upload_tensor(st, "model.norm.weight");

    if (cfg.tie_embeddings) {
        m->lm_head = m->embed_tokens;
    } else {
        m->lm_head = upload_tensor(st, "lm_head.weight");
    }

    for (int l = 0; l < cfg.n_layers; l++) {
        char prefix[128];
        snprintf(prefix, sizeof(prefix), "model.layers.%d", l);
        auto name = [&](const char *suffix) {
            return std::string(prefix) + "." + suffix;
        };
        QwenLayer &ly = m->layers[l];
        ly.input_layernorm     = upload_tensor(st, name("input_layernorm.weight"));
        ly.q_proj              = upload_tensor(st, name("self_attn.q_proj.weight"));
        ly.k_proj              = upload_tensor(st, name("self_attn.k_proj.weight"));
        ly.v_proj              = upload_tensor(st, name("self_attn.v_proj.weight"));
        ly.q_norm              = upload_tensor(st, name("self_attn.q_norm.weight"));
        ly.k_norm              = upload_tensor(st, name("self_attn.k_norm.weight"));
        ly.o_proj              = upload_tensor(st, name("self_attn.o_proj.weight"));
        ly.post_attn_layernorm = upload_tensor(st, name("post_attention_layernorm.weight"));
        ly.gate_proj           = upload_tensor(st, name("mlp.gate_proj.weight"));
        ly.up_proj             = upload_tensor(st, name("mlp.up_proj.weight"));
        ly.down_proj           = upload_tensor(st, name("mlp.down_proj.weight"));

        fprintf(stderr, "[Load] Layers: %d/%d\n", l + 1, cfg.n_layers);
    }

    fprintf(stderr, "[Load] Weights (%.0fms)\n", t_weights.ms());

    // Allocate KV cache
    size_t kv_size = (size_t)cfg.n_layers * 2 * cfg.max_seq_len * m->kv_dim * sizeof(bf16);
    CUDA_CHECK(cudaMalloc(&m->kv_cache, kv_size));
    CUDA_CHECK(cudaMemset(m->kv_cache, 0, kv_size));
    CUDA_CHECK(cudaMalloc(&m->kv_cache_uncond, kv_size));
    CUDA_CHECK(cudaMemset(m->kv_cache_uncond, 0, kv_size));
    fprintf(stderr, "[Load] KV Cache: %.0f MB (x2 for CFG)\n", kv_size * 2 / 1e6);

    // Allocate scratch buffers
    int q_dim = cfg.n_heads * cfg.head_dim;
    CUDA_CHECK(cudaMalloc(&m->buf_x,            cfg.hidden_size * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&m->buf_res,           cfg.hidden_size * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&m->buf_q,             q_dim * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&m->buf_k,             m->kv_dim * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&m->buf_v,             m->kv_dim * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&m->buf_attn,          q_dim * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&m->buf_gate,          cfg.intermediate_size * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&m->buf_up,            cfg.intermediate_size * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&m->buf_logits_f32,    cfg.vocab_size * sizeof(float)));

    // cuBLAS handle
    cublasCreate(&m->cublas);
    cublasSetMathMode(m->cublas, CUBLAS_DEFAULT_MATH);
}

// KV cache pointer helpers
// Layout: [layer][kv_type][max_seq][kv_dim]
// kv_type: 0=K, 1=V
static inline bf16 *kv_ptr(QwenModel *m, int layer, int kv_type) {
    size_t offset = ((size_t)layer * 2 + kv_type) * m->cfg.max_seq_len * m->kv_dim;
    return m->kv_cache + offset;
}

// Forward one token through the model (decode mode)
// pos = current sequence position (0-indexed)
// Returns logits in m->buf_logits_f32 [vocab_size]
static void forward_token(QwenModel *m, int token_id, int pos) {
    QwenConfig &c = m->cfg;
    float scale = 1.0f / sqrtf((float)c.head_dim);

    // Embedding lookup
    embed_lookup(m->buf_x, m->embed_tokens, token_id, c.hidden_size);

    for (int l = 0; l < c.n_layers; l++) {
        QwenLayer &ly = m->layers[l];

        // Save residual
        copy_bf16(m->buf_res, m->buf_x, c.hidden_size);

        // Input LayerNorm
        rmsnorm(m->buf_x, m->buf_x, ly.input_layernorm, c.hidden_size, c.rms_norm_eps);

        // QKV projections
        linear(m->buf_q, m->buf_x, ly.q_proj, c.n_heads * c.head_dim, c.hidden_size, m->cublas);
        linear(m->buf_k, m->buf_x, ly.k_proj, m->kv_dim, c.hidden_size, m->cublas);
        linear(m->buf_v, m->buf_x, ly.v_proj, m->kv_dim, c.hidden_size, m->cublas);

        // QK-Norm (per-head RMSNorm on Q and K)
        head_rmsnorm(m->buf_q, m->buf_q, ly.q_norm, c.n_heads, c.head_dim, c.rms_norm_eps);
        head_rmsnorm(m->buf_k, m->buf_k, ly.k_norm, c.n_kv_heads, c.head_dim, c.rms_norm_eps);

        // RoPE
        rope(m->buf_q, m->buf_k, pos, c.n_heads, c.n_kv_heads, c.head_dim, c.rope_theta);

        // Store K,V in cache
        kv_store(kv_ptr(m, l, 0), m->buf_k, pos, m->kv_dim);
        kv_store(kv_ptr(m, l, 1), m->buf_v, pos, m->kv_dim);

        // Attention: Q @ K^T / sqrt(d) -> softmax -> @ V
        decode_attention(m->buf_attn, m->buf_q,
                        kv_ptr(m, l, 0), kv_ptr(m, l, 1),
                        pos + 1,  // seq_len = all positions up to and including current
                        c.n_heads, c.n_kv_heads, c.head_dim, scale);

        // O projection
        linear(m->buf_x, m->buf_attn, ly.o_proj, c.hidden_size, c.n_heads * c.head_dim, m->cublas);

        // Residual add
        add_inplace(m->buf_x, m->buf_res, c.hidden_size);

        // Save residual
        copy_bf16(m->buf_res, m->buf_x, c.hidden_size);

        // Post-attention LayerNorm
        rmsnorm(m->buf_x, m->buf_x, ly.post_attn_layernorm, c.hidden_size, c.rms_norm_eps);

        // MLP: SwiGLU
        linear(m->buf_gate, m->buf_x, ly.gate_proj, c.intermediate_size, c.hidden_size, m->cublas);
        linear(m->buf_up,   m->buf_x, ly.up_proj,   c.intermediate_size, c.hidden_size, m->cublas);
        silu_mul(m->buf_gate, m->buf_gate, m->buf_up, c.intermediate_size);
        linear(m->buf_x, m->buf_gate, ly.down_proj, c.hidden_size, c.intermediate_size, m->cublas);

        // Residual add
        add_inplace(m->buf_x, m->buf_res, c.hidden_size);
    }

    // Final LayerNorm
    rmsnorm(m->buf_x, m->buf_x, m->final_norm, c.hidden_size, c.rms_norm_eps);

    // Logits: x @ lm_head^T -> fp32
    linear_f32(m->buf_logits_f32, m->buf_x, m->lm_head,
               c.vocab_size, c.hidden_size, m->cublas);
}

// Top-p (nucleus) sampling on CPU
// Global RNG (seeded from --seed option)
static std::mt19937 g_rng(42);

struct TokenProb {
    int id;
    float prob;
};

static int sample_top_p(float *logits, int vocab_size, float temperature, float top_p) {
    // Apply temperature
    for (int i = 0; i < vocab_size; i++)
        logits[i] /= temperature;

    // Softmax
    float max_val = *std::max_element(logits, logits + vocab_size);
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        logits[i] = expf(logits[i] - max_val);
        sum += logits[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < vocab_size; i++)
        logits[i] *= inv_sum;

    // Build sorted token-prob pairs (partial sort for efficiency)
    // For large vocab, only keep top candidates
    static std::vector<TokenProb> candidates;
    candidates.clear();

    // Quick filter: only keep tokens with prob > threshold
    float threshold = 1.0f / (float)vocab_size * 0.01f;
    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] > threshold)
            candidates.push_back({i, logits[i]});
    }

    // Sort descending by probability
    std::sort(candidates.begin(), candidates.end(),
              [](const TokenProb &a, const TokenProb &b) { return a.prob > b.prob; });

    // Cumulative sum until top_p
    float cum = 0.0f;
    int n_keep = 0;
    for (size_t i = 0; i < candidates.size(); i++) {
        cum += candidates[i].prob;
        n_keep = i + 1;
        if (cum >= top_p) break;
    }

    // Renormalize
    float renorm_sum = 0.0f;
    for (int i = 0; i < n_keep; i++)
        renorm_sum += candidates[i].prob;

    // Sample (uses global g_rng, seeded from --seed)
    std::uniform_real_distribution<float> dist(0.0f, renorm_sum);
    float r = dist(g_rng);
    float acc = 0.0f;
    for (int i = 0; i < n_keep; i++) {
        acc += candidates[i].prob;
        if (acc >= r) return candidates[i].id;
    }
    return candidates[0].id;
}

// Generate tokens autoregressively
// Audio code tokens: <|audio_code_0|> = 151669, <|audio_code_65534|> = 217203
#define AUDIO_CODE_BASE  151669
#define AUDIO_CODE_COUNT 65535

static void generate(QwenModel *m, const std::vector<int> &prompt_tokens,
                     int max_new_tokens, float temperature, float top_p, int seed,
                     std::vector<int> *out_audio_codes = nullptr,
                     float cfg_scale = 1.0f,
                     const std::vector<int> *uncond_tokens = nullptr,
                     bool cot_injected = false,
                     double *out_prefill_ms = nullptr,
                     double *out_decode_ms = nullptr) {
    int pos_cond = 0;
    int pos_uncond = 0;
    int total_tokens = 0;
    int audio_code_count = 0;
    bool use_cfg = cfg_scale > 1.0f && uncond_tokens != nullptr;
    // After </think> (151668), only allow audio codes + EOS
    bool codes_phase = cot_injected;
    const int THINK_END = 151668;
    const int EOS_TOKEN = 151645;

    fprintf(stderr, "[Prefill] Cond: %zu tokens", prompt_tokens.size());
    if (use_cfg)
        fprintf(stderr, ", Uncond: %zu tokens", uncond_tokens->size());
    fprintf(stderr, "\n");

    std::vector<float> logits_cond(m->cfg.vocab_size);
    std::vector<float> logits_uncond(m->cfg.vocab_size);
    Timer t_prefill;

    // Prefill: conditional prompt
    for (size_t i = 0; i < prompt_tokens.size(); i++) {
        forward_token(m, prompt_tokens[i], pos_cond);
        pos_cond++;
    }

    // Save conditional logits from prefill
    CUDA_CHECK(cudaMemcpy(logits_cond.data(), m->buf_logits_f32,
                          m->cfg.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));

    // Prefill: unconditional prompt (into separate KV cache)
    if (use_cfg) {
        bf16 *kv_main = m->kv_cache;
        m->kv_cache = m->kv_cache_uncond;
        for (size_t i = 0; i < uncond_tokens->size(); i++) {
            forward_token(m, (*uncond_tokens)[i], pos_uncond);
            pos_uncond++;
        }
        // Save unconditional logits from prefill
        CUDA_CHECK(cudaMemcpy(logits_uncond.data(), m->buf_logits_f32,
                              m->cfg.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
        m->kv_cache = kv_main;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    size_t total_prefill = prompt_tokens.size() + (use_cfg ? uncond_tokens->size() : 0);
    double prefill_ms = t_prefill.ms();
    fprintf(stderr, "[Prefill] %.0fms (%.1f tok/s, %zu tokens)\n",
            prefill_ms, total_prefill / (prefill_ms / 1000.0), total_prefill);
    if (out_prefill_ms) *out_prefill_ms = prefill_ms;

    // Decode
    Timer t_decode;

    g_rng.seed(seed);

    for (int i = 0; i < max_new_tokens; i++) {
        // logits_cond and logits_uncond are already filled
        // (from prefill on step 0, or from previous iteration's forward passes)

        float *sample_logits = logits_cond.data();

        // Apply CFG combination
        if (use_cfg) {
            for (int v = 0; v < m->cfg.vocab_size; v++) {
                logits_cond[v] = logits_uncond[v] + cfg_scale * (logits_cond[v] - logits_uncond[v]);
            }
            sample_logits = logits_cond.data();
        }

        // Constrained decoding: after </think>, only audio codes + EOS
        if (codes_phase) {
            for (int v = 0; v < AUDIO_CODE_BASE; v++)
                if (v != EOS_TOKEN) sample_logits[v] = -1e9f;
        }

        int next_token = sample_top_p(sample_logits, m->cfg.vocab_size, temperature, top_p);

        // EOS: count + break before any display or forward
        if (next_token == EOS_TOKEN) {
            total_tokens++;
            break;
        }

        // Detect </think> -> switch to codes phase
        if (next_token == THINK_END && !codes_phase) {
            codes_phase = true;
        }

        // Collect audio codes
        if (next_token >= AUDIO_CODE_BASE && next_token < AUDIO_CODE_BASE + AUDIO_CODE_COUNT) {
            if (out_audio_codes) out_audio_codes->push_back(next_token - AUDIO_CODE_BASE);
            audio_code_count++;
        }

        // Forward conditional: same token
        forward_token(m, next_token, pos_cond);
        pos_cond++;
        // Save cond logits for next iteration
        CUDA_CHECK(cudaMemcpy(logits_cond.data(), m->buf_logits_f32,
                              m->cfg.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));

        // Forward unconditional: same token, into uncond KV cache
        if (use_cfg) {
            bf16 *kv_main = m->kv_cache;
            m->kv_cache = m->kv_cache_uncond;
            forward_token(m, next_token, pos_uncond);
            pos_uncond++;
            // Save uncond logits for NEXT iteration
            CUDA_CHECK(cudaMemcpy(logits_uncond.data(), m->buf_logits_f32,
                                  m->cfg.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
            m->kv_cache = kv_main;
        }

        total_tokens++;

        if (codes_phase && (total_tokens % 50 == 0)) {
            CUDA_CHECK(cudaDeviceSynchronize());
            double elapsed = t_decode.ms() / 1000.0;
            fprintf(stderr, "[Decode] %d/%d codes, %.1f tok/s\n",
                    audio_code_count, max_new_tokens, total_tokens / elapsed);
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    if (total_tokens > 0) {
        double decode_ms = t_decode.ms();
        double tok_s = total_tokens / (decode_ms / 1000.0);
        fprintf(stderr, "[Decode] %d/%d codes, %.1f tok/s\n",
                audio_code_count, max_new_tokens, tok_s);
        fprintf(stderr, "[Decode] %.0fms (%d tokens, %d audio codes)\n",
                decode_ms, total_tokens, audio_code_count);
        if (out_decode_ms) *out_decode_ms = decode_ms;
    }
}

// ACE-Step prompt handling
struct AcePrompt {
    std::string caption;
    std::string lyrics;
    float duration;
    int seed;
    // CoT metadata (optional, from prompt.json)
    int bpm;                    // 0 = not set
    std::string keyscale;       // e.g. "F# minor"
    std::string timesignature;  // e.g. "4"
    std::string vocal_language; // e.g. "fr"
};

// Read a JSON value (quoted string or unquoted number/bool) as string
static std::string json_value(const char *json, const char *key) {
    char needle[256];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char *p = strstr(json, needle);
    if (!p) return "";
    p += strlen(needle);
    while (*p && (*p == ' ' || *p == ':' || *p == '\t')) p++;
    if (*p == '"') {
        // Quoted string, delegate to json_string
        return "";  // caller should use json_string instead
    }
    // Unquoted value (number, bool, null)
    std::string result;
    while (*p && *p != ',' && *p != '}' && *p != ' ' && *p != '\n' && *p != '\r') {
        result += *p++;
    }
    return result;
}

static std::string json_string(const char *json, const char *key) {
    char needle[256];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char *p = strstr(json, needle);
    if (!p) return "";
    p += strlen(needle);
    while (*p && (*p == ' ' || *p == ':' || *p == '\t')) p++;
    if (*p != '"') return "";
    p++;
    std::string result;
    while (*p && *p != '"') {
        if (*p == '\\' && *(p + 1)) {
            p++;
            if (*p == 'n') result += '\n';
            else if (*p == 't') result += '\t';
            else if (*p == '"') result += '"';
            else if (*p == '\\') result += '\\';
            else { result += '\\'; result += *p; }
        } else {
            result += *p;
        }
        p++;
    }
    return result;
}

static bool load_ace_prompt(const char *path, AcePrompt *p) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", path); return false; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::string buf(sz, '\0');
    fread(&buf[0], 1, sz, f);
    fclose(f);
    const char *j = buf.c_str();
    p->caption = json_string(j, "caption");
    p->lyrics = json_string(j, "lyrics");
    p->duration = (float)atof(json_string(j, "duration").c_str());
    if (p->duration <= 0) p->duration = 120.0f;
    std::string seed_str = json_string(j, "seed");
    p->seed = seed_str.empty() ? 42 : atoi(seed_str.c_str());
    // CoT metadata (optional)
    std::string bpm_str = json_value(j, "bpm");
    p->bpm = bpm_str.empty() ? 0 : atoi(bpm_str.c_str());
    p->keyscale = json_string(j, "keyscale");
    p->timesignature = json_string(j, "timesignature");
    p->vocal_language = json_string(j, "vocal_language");
    if (p->caption.empty()) { fprintf(stderr, "ERROR: no caption in %s\n", path); return false; }
    return true;
}

// Build Qwen3 chat-template token IDs for ACE-Step 5Hz LM
// Format: <|im_start|>system\n{sys}\n<|im_end|>\n<|im_start|>user\n{user}\n<|im_end|>\n<|im_start|>assistant\n
static std::vector<int> build_lm_prompt(BPETokenizer &bpe, const AcePrompt &prompt) {
    std::vector<int> ids;
    auto append_bpe = [&](const std::string &text) {
        auto t = bpe_encode(&bpe, text, false);
        ids.insert(ids.end(), t.begin(), t.end());
    };
    // System turn
    ids.push_back(151644);  // <|im_start|>
    append_bpe("system\n# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n");
    ids.push_back(151645);  // <|im_end|>
    append_bpe("\n");
    // User turn
    ids.push_back(151644);
    append_bpe("user\n# Caption\n" + prompt.caption + "\n\n# Lyric\n" + prompt.lyrics + "\n");
    ids.push_back(151645);
    append_bpe("\n");
    // Assistant turn (generation starts here)
    ids.push_back(151644);
    append_bpe("assistant\n");
    return ids;
}

// Build unconditional prompt for CFG: caption removed (or replaced by negative_prompt), lyrics kept
static std::vector<int> build_lm_prompt_uncond(BPETokenizer &bpe, const AcePrompt &prompt,
                                                const char *negative_prompt) {
    std::vector<int> ids;
    auto append_bpe = [&](const std::string &text) {
        auto t = bpe_encode(&bpe, text, false);
        ids.insert(ids.end(), t.begin(), t.end());
    };
    // System turn (identical)
    ids.push_back(151644);
    append_bpe("system\n# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n");
    ids.push_back(151645);
    append_bpe("\n");
    // User turn: caption replaced or removed
    ids.push_back(151644);
    bool has_neg = negative_prompt && strlen(negative_prompt) > 0
                   && strcmp(negative_prompt, "NO USER INPUT") != 0;
    if (has_neg) {
        // Meaningful negative prompt: use as caption
        append_bpe("user\n# Caption\n" + std::string(negative_prompt) + "\n\n# Lyric\n" + prompt.lyrics + "\n");
    } else {
        // Default: remove caption, keep lyrics only (matches Python CoT-phase behavior)
        append_bpe("user\n# Lyric\n" + prompt.lyrics + "\n");
    }
    ids.push_back(151645);
    append_bpe("\n");
    // Assistant turn
    ids.push_back(151644);
    append_bpe("assistant\n");
    return ids;
}

// Check if field is in comma-separated list or "all"
static bool cot_has(const char *fields, const char *name) {
    if (!strcmp(fields, "all")) return true;
    const char *p = strstr(fields, name);
    if (!p) return false;
    int len = strlen(name);
    // Check boundaries: must be at start/after comma, and end at end/before comma
    if (p != fields && *(p - 1) != ',') return false;
    char after = *(p + len);
    return after == '\0' || after == ',';
}

// Build CoT YAML text from metadata (sorted alphabetically, matching Python yaml.dump)
// fields: comma-separated list of fields to include, or "all"
static std::string build_cot_text(const AcePrompt &prompt, const char *fields) {
    std::string cot = "<think>\n";
    if (cot_has(fields, "bpm") && prompt.bpm > 0)
        cot += "bpm: " + std::to_string(prompt.bpm) + "\n";
    if (cot_has(fields, "caption") && !prompt.caption.empty())
        cot += "caption: " + prompt.caption + "\n";
    if (cot_has(fields, "duration") && prompt.duration > 0)
        cot += "duration: " + std::to_string((int)prompt.duration) + "\n";
    if (cot_has(fields, "keyscale") && !prompt.keyscale.empty())
        cot += "keyscale: " + prompt.keyscale + "\n";
    if (cot_has(fields, "language") && !prompt.vocal_language.empty())
        cot += "language: " + prompt.vocal_language + "\n";
    if (cot_has(fields, "timesignature") && !prompt.timesignature.empty())
        cot += "timesignature: " + prompt.timesignature + "\n";
    cot += "</think>\n";
    return cot;
}

// Build prompt with pre-built CoT injected in assistant turn (Phase 2 mode)
// The LM will continue generating audio codes after </think>
static std::vector<int> build_lm_prompt_with_cot(BPETokenizer &bpe, const AcePrompt &prompt,
                                                  const std::string &cot_text) {
    std::vector<int> ids;
    auto append_bpe = [&](const std::string &text) {
        auto t = bpe_encode(&bpe, text, false);
        ids.insert(ids.end(), t.begin(), t.end());
    };
    // System turn
    ids.push_back(151644);
    append_bpe("system\n# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n");
    ids.push_back(151645);
    append_bpe("\n");
    // User turn
    ids.push_back(151644);
    append_bpe("user\n# Caption\n" + prompt.caption + "\n\n# Lyric\n" + prompt.lyrics + "\n");
    ids.push_back(151645);
    append_bpe("\n");
    // Assistant turn with injected CoT
    ids.push_back(151644);
    append_bpe("assistant\n" + cot_text);
    return ids;
}

// Build unconditional prompt with empty CoT for CFG (Phase 2 mode)
static std::vector<int> build_lm_prompt_uncond_with_cot(BPETokenizer &bpe, const AcePrompt &prompt,
                                                         const char *negative_prompt) {
    std::vector<int> ids;
    auto append_bpe = [&](const std::string &text) {
        auto t = bpe_encode(&bpe, text, false);
        ids.insert(ids.end(), t.begin(), t.end());
    };
    // System turn
    ids.push_back(151644);
    append_bpe("system\n# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n");
    ids.push_back(151645);
    append_bpe("\n");
    // User turn: caption replaced/removed
    ids.push_back(151644);
    bool has_neg = negative_prompt && strlen(negative_prompt) > 0
                   && strcmp(negative_prompt, "NO USER INPUT") != 0;
    if (has_neg) {
        append_bpe("user\n# Caption\n" + std::string(negative_prompt) + "\n\n# Lyric\n" + prompt.lyrics + "\n");
    } else {
        append_bpe("user\n# Lyric\n" + prompt.lyrics + "\n");
    }
    ids.push_back(151645);
    append_bpe("\n");
    // Assistant turn with empty CoT
    ids.push_back(151644);
    append_bpe("assistant\n<think>\n</think>\n");
    return ids;
}

// Main
static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s <model_dir> [options]\n", prog);
    fprintf(stderr, "  --prompt-json <file>   ACE-Step prompt JSON (caption, lyrics, duration)\n");
    fprintf(stderr, "  --output-codes <file>  Write audio codes to file (ACE-Step mode)\n");
    fprintf(stderr, "  --max-tokens <n>       Max new tokens to generate (default: 256)\n");
    fprintf(stderr, "  --temperature <f>      Sampling temperature (default: 0.8)\n");
    fprintf(stderr, "  --top-p <f>            Top-p sampling (default: 0.9)\n");
    fprintf(stderr, "  --max-seq <n>          Max sequence length for KV cache (default: 8192)\n");
    fprintf(stderr, "  --seed <n>             Random seed (default: 42)\n");
    fprintf(stderr, "  --cfg-scale <f>        Classifier-Free Guidance scale (default: 1.0 = off)\n");
    fprintf(stderr, "  --negative-prompt <s>  Negative prompt for CFG (default: \"NO USER INPUT\")\n");
    fprintf(stderr, "\nExamples:\n");
    fprintf(stderr, "  %s ./checkpoints/acestep-5Hz-lm-4B --max-tokens 512\n", prog);
    fprintf(stderr, "  %s ./checkpoints/acestep-5Hz-lm-4B --prompt-json prompt.json --output-codes codes.txt\n", prog);
}

int main(int argc, char **argv) {
    if (argc < 2 || !strcmp(argv[1], "--help") || !strcmp(argv[1], "-h")) {
        usage(argv[0]);
        return 1;
    }

    const char *model_dir = argv[1];
    std::vector<int> prompt;
    int max_tokens = 256;
    float temperature = 0.8f;
    float top_p = 0.9f;
    int max_seq = 8192;
    int seed = 42;
    float cfg_scale = 1.0f;
    const char *negative_prompt = "NO USER INPUT";
    const char *prompt_json = nullptr;
    const char *output_codes = nullptr;

    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--prompt-json") && i + 1 < argc)
            prompt_json = argv[++i];
        else if (!strcmp(argv[i], "--output-codes") && i + 1 < argc)
            output_codes = argv[++i];
        else if (!strcmp(argv[i], "--max-tokens") && i + 1 < argc)
            max_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--temperature") && i + 1 < argc)
            temperature = atof(argv[++i]);
        else if (!strcmp(argv[i], "--top-p") && i + 1 < argc)
            top_p = atof(argv[++i]);
        else if (!strcmp(argv[i], "--max-seq") && i + 1 < argc)
            max_seq = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc)
            seed = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cfg-scale") && i + 1 < argc)
            cfg_scale = atof(argv[++i]);
        else if (!strcmp(argv[i], "--negative-prompt") && i + 1 < argc)
            negative_prompt = argv[++i];
        else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    // Build prompt (from JSON or empty for free generation)
    Timer t_total;
    std::vector<int> uncond_prompt;
    BPETokenizer bpe;
    if (!load_bpe_tokenizer(&bpe, model_dir)) return 1;
    bool cot_injected = false;

    AcePrompt ace = {};
    ace.duration = 120.0f;
    ace.seed = 42;

    if (prompt_json) {
        if (!load_ace_prompt(prompt_json, &ace)) return 1;
    }

    prompt = build_lm_prompt(bpe, ace);
    if (cfg_scale > 1.0f)
        uncond_prompt = build_lm_prompt_uncond(bpe, ace, negative_prompt);
    if (seed == 42) seed = ace.seed;
    if (max_tokens == 256) max_tokens = (int)(ace.duration * 5) + 800;

    // Auto-detect CoT: if metadata present in prompt.json, inject it
    bool has_meta = ace.bpm > 0 || !ace.keyscale.empty()
                    || !ace.timesignature.empty() || !ace.vocal_language.empty();
    if (has_meta) {
        std::string cot = build_cot_text(ace, "all");
        prompt = build_lm_prompt_with_cot(bpe, ace, cot);
        if (cfg_scale > 1.0f)
            uncond_prompt = build_lm_prompt_uncond_with_cot(bpe, ace, negative_prompt);
        if (max_tokens == (int)(ace.duration * 5) + 800)
            max_tokens = (int)(ace.duration * 5) + 100;
        cot_injected = true;
    }

    fprintf(stderr, "[Prompt] %zu tokens, max_new: %d, seed: %d\n",
            prompt.size(), max_tokens, seed);
    if (cfg_scale > 1.0f)
        fprintf(stderr, "[Prompt] CFG: %.2f, Uncond: %zu tokens\n",
                cfg_scale, uncond_prompt.size());
    if (cot_injected)
        fprintf(stderr, "[Prompt] CoT injected from prompt.json\n");

    QwenConfig cfg = load_config(model_dir);
    cfg.max_seq_len = max_seq;

    Timer t_load;
    QwenModel model;
    load_model(&model, model_dir, cfg);
    double load_ms = t_load.ms();

    double prefill_ms = 0, decode_ms = 0;
    std::vector<int> audio_codes;
    generate(&model, prompt, max_tokens, temperature, top_p, seed,
             output_codes ? &audio_codes : nullptr,
             cfg_scale, uncond_prompt.empty() ? nullptr : &uncond_prompt,
             cot_injected, &prefill_ms, &decode_ms);

    // Write audio codes to file
    if (output_codes && !audio_codes.empty()) {
        FILE *f = fopen(output_codes, "w");
        if (!f) { fprintf(stderr, "ERROR: cannot write %s\n", output_codes); return 1; }
        for (size_t i = 0; i < audio_codes.size(); i++) {
            if (i > 0) fprintf(f, ",");
            fprintf(f, "%d", audio_codes[i]);
        }
        fprintf(f, "\n");
        fclose(f);
        fprintf(stderr, "[Output] %s (%zu audio codes)\n", output_codes, audio_codes.size());
    }

    fprintf(stderr, "[Ace-Qwen3] Load %.0f | Prefill %.0f | Decode %.0f | Total %.0fms\n",
            load_ms, prefill_ms, decode_ms, t_total.ms());

    return 0;
}
