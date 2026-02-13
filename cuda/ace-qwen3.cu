// ace-cuda: Minimal standalone Qwen3 inference engine
// Loads safetensors, runs forward pass with cuBLAS+custom kernels
// bf16 storage, fp32 compute. No PyTorch, no GGML, no deps.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <map>
#include <random>
#include <chrono>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include "../bpe.h"
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
    size_t kv_bytes;                // total bytes per KV cache (for reset)

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
    m->kv_bytes = kv_size;
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
// Special token IDs (Qwen3 vocab)
#define TOKEN_IM_START  151644
#define TOKEN_IM_END    151645
#define TOKEN_THINK     151667
#define TOKEN_THINK_END 151668

// Audio code tokens: <|audio_code_0|> = 151669
// Vocab has tokens up to <|audio_code_65534|> = 217203 but codebook size = 64000
// Python constrained decoding enforces MAX_AUDIO_CODE = 63999
#define AUDIO_CODE_BASE  151669
#define AUDIO_CODE_COUNT 64000

// Decode BPE token IDs back to UTF-8 text
// Handles GPT-2 byte-level encoding: each vocab token string uses remapped Unicode
// codepoints that must be converted back to raw bytes.
static std::string bpe_decode(const BPETokenizer &bpe, const std::vector<int> &ids) {
    // Build inverse byte table once: GPT-2 codepoint -> raw byte value
    static std::unordered_map<int, uint8_t> byte_dec;
    static bool init = false;
    if (!init) {
        for (int b = 0; b < 256; b++) {
            int adv;
            int cp = utf8_codepoint(bpe.byte2str[b].c_str(), &adv);
            byte_dec[cp] = (uint8_t)b;
        }
        init = true;
    }

    std::string result;
    for (int id : ids) {
        // Emit markers for think tags (needed by parser)
        if (id == TOKEN_THINK)     { result += "<think>";  continue; }
        if (id == TOKEN_THINK_END) { result += "</think>"; continue; }
        // Skip other special tokens and audio codes
        if (id == TOKEN_IM_START || id == TOKEN_IM_END) continue;
        if (id >= AUDIO_CODE_BASE) continue;
        if (id < 0 || id >= (int)bpe.id_to_str.size()) continue;
        const std::string &s = bpe.id_to_str[id];
        if (s.empty()) continue;
        // Convert GPT-2 encoded codepoints back to raw bytes
        const char *p = s.c_str();
        while (*p) {
            int adv;
            int cp = utf8_codepoint(p, &adv);
            auto it = byte_dec.find(cp);
            if (it != byte_dec.end()) result += (char)it->second;
            p += adv;
        }
    }
    return result;
}

// Constrained decoding FSM for metadata generation inside <think>...</think>
// Enforces valid field order, ranges, and values matching Python upstream.
// Reference: constrained_logits_processor.py (FSMState, MetadataConstrainedLogitsProcessor)

struct PrefixTree {
    // Map token prefix -> valid next token IDs
    std::map<std::vector<int>, std::vector<int>> nodes;

    void add(const std::vector<int> &seq) {
        for (size_t i = 0; i < seq.size(); i++) {
            std::vector<int> prefix(seq.begin(), seq.begin() + i);
            auto &v = nodes[prefix];
            if (std::find(v.begin(), v.end(), seq[i]) == v.end())
                v.push_back(seq[i]);
        }
    }

    const std::vector<int> *get(const std::vector<int> &prefix) const {
        auto it = nodes.find(prefix);
        return it != nodes.end() ? &it->second : nullptr;
    }
};

struct MetadataFSM {
    enum State {
        BPM_NAME, BPM_VALUE,
        CAPTION_NAME, CAPTION_VALUE,
        DURATION_NAME, DURATION_VALUE,
        KEYSCALE_NAME, KEYSCALE_VALUE,
        LANGUAGE_NAME, LANGUAGE_VALUE,
        TIMESIG_NAME, TIMESIG_VALUE,
        THINK_END,
        CODES,
        DISABLED
    };

    State state = DISABLED;
    int name_pos = 0;
    std::vector<int> value_acc;
    bool enabled = false;

    // Field name token sequences (what the model must emit)
    std::vector<int> bpm_name, caption_name, duration_name;
    std::vector<int> keyscale_name, language_name, timesig_name;
    // Value prefix trees
    PrefixTree bpm_tree, duration_tree, keyscale_tree, language_tree, timesig_tree;
    int newline_tok = -1;
    int think_end_tok = TOKEN_THINK_END;
    int vocab_size = 0;

    static std::vector<int> tokenize_strip(BPETokenizer &bpe,
                                           const std::string &full,
                                           const std::string &prefix) {
        std::vector<int> full_tok = bpe_encode(&bpe, full, false);
        std::vector<int> pre_tok = bpe_encode(&bpe, prefix, false);
        if (full_tok.size() >= pre_tok.size() &&
            std::equal(pre_tok.begin(), pre_tok.end(), full_tok.begin()))
            return std::vector<int>(full_tok.begin() + pre_tok.size(), full_tok.end());
        return full_tok;
    }

    void build_value_tree(BPETokenizer &bpe, PrefixTree &tree,
                          const std::string &field_prefix,
                          const std::vector<std::string> &values) {
        for (auto &val : values) {
            std::string full = field_prefix + val + "\n";
            std::vector<int> vtok = tokenize_strip(bpe, full, field_prefix);
            tree.add(vtok);
        }
    }

    void init(BPETokenizer &bpe, int vsize) {
        vocab_size = vsize;

        // Tokenize newline
        auto nl = bpe_encode(&bpe, "\n", false);
        newline_tok = nl.empty() ? -1 : nl[0];

        // Field name tokens
        bpm_name = bpe_encode(&bpe, "bpm:", false);
        caption_name = bpe_encode(&bpe, "caption:", false);
        duration_name = bpe_encode(&bpe, "duration:", false);
        keyscale_name = bpe_encode(&bpe, "keyscale:", false);
        language_name = bpe_encode(&bpe, "language:", false);
        timesig_name = bpe_encode(&bpe, "timesignature:", false);

        // BPM 30-300
        {
            std::vector<std::string> vals;
            for (int v = 30; v <= 300; v++) vals.push_back(std::to_string(v));
            build_value_tree(bpe, bpm_tree, "bpm:", vals);
        }
        // Duration 10-600
        {
            std::vector<std::string> vals;
            for (int v = 10; v <= 600; v++) vals.push_back(std::to_string(v));
            build_value_tree(bpe, duration_tree, "duration:", vals);
        }
        // Keyscale: note + accidental + " " + mode
        {
            const char *notes[] = {"A","B","C","D","E","F","G"};
            const char *accs[] = {"","b","#"};
            const char *modes[] = {
                "major","minor","dorian","phrygian","lydian","mixolydian",
                "aeolian","locrian","chromatic","blues","pentatonic",
                "harmonic minor","melodic minor"
            };
            std::vector<std::string> vals;
            for (auto n : notes)
                for (auto a : accs)
                    for (auto m : modes)
                        vals.push_back(std::string(n) + a + " " + m);
            build_value_tree(bpe, keyscale_tree, "keyscale:", vals);
        }
        // Language
        {
            std::vector<std::string> vals = {
                "en","zh","ja","ko","es","fr","de","uk","ru","pt",
                "it","ar","tr","pl","sv","nl","unknown"
            };
            build_value_tree(bpe, language_tree, "language:", vals);
        }
        // Time signature
        {
            std::vector<std::string> vals = {"2","3","4","6"};
            build_value_tree(bpe, timesig_tree, "timesignature:", vals);
        }

        fprintf(stderr, "[FSM] Prefix trees: bpm=%zu, dur=%zu, key=%zu, lang=%zu, tsig=%zu nodes\n",
                bpm_tree.nodes.size(), duration_tree.nodes.size(),
                keyscale_tree.nodes.size(), language_tree.nodes.size(),
                timesig_tree.nodes.size());
        enabled = true;
        state = BPM_NAME;
        name_pos = 0;
        value_acc.clear();
    }

    void reset() {
        state = BPM_NAME;
        name_pos = 0;
        value_acc.clear();
    }

    const std::vector<int> *current_name_tokens() const {
        switch (state) {
            case BPM_NAME: return &bpm_name;
            case CAPTION_NAME: return &caption_name;
            case DURATION_NAME: return &duration_name;
            case KEYSCALE_NAME: return &keyscale_name;
            case LANGUAGE_NAME: return &language_name;
            case TIMESIG_NAME: return &timesig_name;
            default: return nullptr;
        }
    }

    const PrefixTree *current_value_tree() const {
        switch (state) {
            case BPM_VALUE: return &bpm_tree;
            case DURATION_VALUE: return &duration_tree;
            case KEYSCALE_VALUE: return &keyscale_tree;
            case LANGUAGE_VALUE: return &language_tree;
            case TIMESIG_VALUE: return &timesig_tree;
            default: return nullptr;
        }
    }

    State next_name_state() const {
        switch (state) {
            case BPM_NAME: case BPM_VALUE: return CAPTION_NAME;
            case CAPTION_NAME: case CAPTION_VALUE: return DURATION_NAME;
            case DURATION_NAME: case DURATION_VALUE: return KEYSCALE_NAME;
            case KEYSCALE_NAME: case KEYSCALE_VALUE: return LANGUAGE_NAME;
            case LANGUAGE_NAME: case LANGUAGE_VALUE: return TIMESIG_NAME;
            case TIMESIG_NAME: case TIMESIG_VALUE: return THINK_END;
            default: return CODES;
        }
    }

    // Mask logits: set disallowed tokens to -inf
    void apply_mask(float *logits) {
        if (!enabled || state == CODES || state == DISABLED) return;

        // NAME states: force next token in field name sequence
        const std::vector<int> *name = current_name_tokens();
        if (name && name_pos < (int)name->size()) {
            int forced = (*name)[name_pos];
            for (int v = 0; v < vocab_size; v++)
                if (v != forced) logits[v] = -1e9f;
            return;
        }

        // VALUE states (except caption): use prefix tree
        const PrefixTree *tree = current_value_tree();
        if (tree) {
            const std::vector<int> *allowed = tree->get(value_acc);
            if (allowed && !allowed->empty()) {
                // Save allowed token logits, mask all, restore
                std::vector<float> saved(allowed->size());
                for (size_t i = 0; i < allowed->size(); i++)
                    saved[i] = logits[(*allowed)[i]];
                for (int v = 0; v < vocab_size; v++) logits[v] = -1e9f;
                for (size_t i = 0; i < allowed->size(); i++)
                    logits[(*allowed)[i]] = saved[i];
            } else {
                // No valid continuation: force newline to end field
                if (newline_tok >= 0) {
                    for (int v = 0; v < vocab_size; v++)
                        if (v != newline_tok) logits[v] = -1e9f;
                }
            }
            return;
        }

        // CAPTION_VALUE: block audio codes only
        if (state == CAPTION_VALUE) {
            for (int v = AUDIO_CODE_BASE; v < AUDIO_CODE_BASE + AUDIO_CODE_COUNT; v++)
                if (v < vocab_size) logits[v] = -1e9f;
            return;
        }

        // THINK_END: force </think> token
        if (state == THINK_END) {
            for (int v = 0; v < vocab_size; v++)
                if (v != think_end_tok) logits[v] = -1e9f;
            return;
        }
    }

    // Update state after a token is sampled
    void update(int token) {
        if (!enabled || state == CODES || state == DISABLED) return;

        // NAME states: advance position
        const std::vector<int> *name = current_name_tokens();
        if (name && name_pos < (int)name->size()) {
            name_pos++;
            if (name_pos >= (int)name->size()) {
                // Name complete: transition to value state
                switch (state) {
                    case BPM_NAME: state = BPM_VALUE; break;
                    case CAPTION_NAME: state = CAPTION_VALUE; break;
                    case DURATION_NAME: state = DURATION_VALUE; break;
                    case KEYSCALE_NAME: state = KEYSCALE_VALUE; break;
                    case LANGUAGE_NAME: state = LANGUAGE_VALUE; break;
                    case TIMESIG_NAME: state = TIMESIG_VALUE; break;
                    default: break;
                }
                value_acc.clear();
            }
            return;
        }

        // VALUE states (except caption): accumulate or transition on newline
        if (current_value_tree()) {
            if (token == newline_tok) {
                state = next_name_state();
                name_pos = 0;
                value_acc.clear();
            } else {
                value_acc.push_back(token);
            }
            return;
        }

        // CAPTION_VALUE: transition on newline
        if (state == CAPTION_VALUE) {
            if (token == newline_tok) {
                state = DURATION_NAME;
                name_pos = 0;
                value_acc.clear();
            }
            return;
        }

        // THINK_END: transition to CODES
        if (state == THINK_END) {
            state = CODES;
            return;
        }
    }
};

static void generate(QwenModel *m, const std::vector<int> &prompt_tokens,
                     int max_new_tokens, float temperature, float top_p, int seed,
                     std::vector<int> *out_audio_codes = nullptr,
                     float cfg_scale = 1.0f,
                     const std::vector<int> *uncond_tokens = nullptr,
                     bool cot_injected = false,
                     double *out_prefill_ms = nullptr,
                     double *out_decode_ms = nullptr,
                     bool stop_at_reasoning = false,
                     std::vector<int> *out_generated_tokens = nullptr,
                     MetadataFSM *fsm = nullptr) {
    int pos_cond = 0;
    int pos_uncond = 0;
    int total_tokens = 0;
    int audio_code_count = 0;
    bool use_cfg = cfg_scale > 1.0f && uncond_tokens != nullptr;
    // After </think> (TOKEN_THINK_END), only allow audio codes + EOS
    bool codes_phase = cot_injected;

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

        // FSM constrained decoding for metadata (before </think>)
        if (fsm && fsm->enabled && !codes_phase)
            fsm->apply_mask(sample_logits);

        // Constrained decoding: after </think>, only audio codes + EOS
        // (skip when stop_at_reasoning, we're generating CoT not codes)
        if (codes_phase && !stop_at_reasoning) {
            for (int v = 0; v < AUDIO_CODE_BASE; v++)
                if (v != TOKEN_IM_END) sample_logits[v] = -1e9f;
        }

        int next_token = sample_top_p(sample_logits, m->cfg.vocab_size, temperature, top_p);

        // Update FSM state
        if (fsm && fsm->enabled && !codes_phase)
            fsm->update(next_token);

        // EOS: count + break before any display or forward
        if (next_token == TOKEN_IM_END) {
            total_tokens++;
            break;
        }

        // Collect all generated tokens if requested
        if (out_generated_tokens) out_generated_tokens->push_back(next_token);

        // Detect </think> -> switch to codes phase or stop
        if (next_token == TOKEN_THINK_END && !codes_phase) {
            if (stop_at_reasoning) {
                total_tokens++;
                break;
            }
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

// Generate text tokens (no CFG, no constrained decoding, stops at EOS)
// Used for inspiration/simple mode Phase 1: metadata + lyrics generation
static std::vector<int> generate_text(QwenModel *m, const std::vector<int> &prompt_tokens,
                                       int max_new_tokens, float temperature, float top_p, int seed,
                                       MetadataFSM *fsm = nullptr) {
    std::vector<int> generated;
    int pos = 0;

    fprintf(stderr, "[Phase1-Prefill] %zu tokens\n", prompt_tokens.size());
    Timer t_prefill;
    for (size_t i = 0; i < prompt_tokens.size(); i++) {
        forward_token(m, prompt_tokens[i], pos);
        pos++;
    }
    std::vector<float> logits(m->cfg.vocab_size);
    CUDA_CHECK(cudaMemcpy(logits.data(), m->buf_logits_f32,
                          m->cfg.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    fprintf(stderr, "[Phase1-Prefill] %.0fms\n", t_prefill.ms());

    Timer t_decode;
    g_rng.seed(seed);
    for (int i = 0; i < max_new_tokens; i++) {
        if (fsm && fsm->enabled) fsm->apply_mask(logits.data());
        int next = sample_top_p(logits.data(), m->cfg.vocab_size, temperature, top_p);
        if (fsm && fsm->enabled) fsm->update(next);
        if (next == TOKEN_IM_END) break;
        generated.push_back(next);
        forward_token(m, next, pos);
        pos++;
        CUDA_CHECK(cudaMemcpy(logits.data(), m->buf_logits_f32,
                              m->cfg.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    fprintf(stderr, "[Phase1-Decode] %.0fms (%zu tokens)\n", t_decode.ms(), generated.size());
    return generated;
}

// ACE-Step prompt handling
struct AcePrompt {
    std::string caption;
    std::string lyrics;
    float duration;
    // CoT metadata (optional, filled by LLM or CLI)
    int bpm;                    // 0 = not set
    std::string keyscale;       // e.g. "F# minor"
    std::string timesignature;  // e.g. "4"
    std::string vocal_language; // e.g. "fr"
};

// Parse LLM output from inspiration mode: extract CoT metadata + lyrics
// Input: decoded text containing <think>YAML</think>LYRICS
static bool parse_cot_and_lyrics(const std::string &text, AcePrompt *out) {
    // Extract CoT content. Three cases:
    // 1. <think>...</think> both present: extract between them
    // 2. Only </think> present: everything before it is CoT (think was in prompt)
    // 3. Neither: treat entire text as CoT
    size_t ts = text.find("<think>");
    size_t te = text.find("</think>");
    std::string cot;
    std::string after_think;
    if (ts != std::string::npos && te != std::string::npos && te > ts) {
        cot = text.substr(ts + 7, te - ts - 7);
        after_think = text.substr(te + 8);
    } else if (te != std::string::npos) {
        cot = text.substr(0, te);
        after_think = text.substr(te + 8);
    } else {
        fprintf(stderr, "WARNING: no </think> in LLM output, treating all as CoT\n");
        cot = text;
    }

    // Parse YAML-like "key: value" with multi-line support (indented continuation)
    std::string cur_key;
    std::string cur_val;
    auto save_field = [&]() {
        if (cur_key.empty()) return;
        // Trim trailing whitespace
        while (!cur_val.empty() && (cur_val.back() == ' ' || cur_val.back() == '\n'))
            cur_val.pop_back();
        if (cur_key == "bpm")            { out->bpm = atoi(cur_val.c_str()); }
        else if (cur_key == "caption")   { out->caption = cur_val; }
        else if (cur_key == "duration")  { out->duration = (float)atof(cur_val.c_str()); }
        else if (cur_key == "keyscale")  { out->keyscale = cur_val; }
        else if (cur_key == "language")  { out->vocal_language = cur_val; }
        else if (cur_key == "timesignature") {
            out->timesignature = cur_val;
            // Strip "/4" suffix (Python llm_inference.py:1018-1019)
            auto &t = out->timesignature;
            if (t.size() >= 2 && t.compare(t.size()-2, 2, "/4") == 0)
                t = t.substr(0, t.size()-2);
        }
        cur_key.clear(); cur_val.clear();
    };

    std::istringstream ss(cot);
    std::string line;
    while (std::getline(ss, line)) {
        if (line.empty() || line[0] == '<') continue;
        // New field: no leading whitespace and contains ':'
        if (!line.empty() && line[0] != ' ' && line[0] != '\t') {
            size_t colon = line.find(':');
            if (colon != std::string::npos) {
                save_field();
                cur_key = line.substr(0, colon);
                // Lowercase the key
                for (auto &c : cur_key) c = tolower(c);
                std::string rest = line.substr(colon + 1);
                // Trim leading space
                size_t fs = rest.find_first_not_of(" \t");
                cur_val = (fs != std::string::npos) ? rest.substr(fs) : "";
                continue;
            }
        }
        // Continuation line (indented) for multi-line values like caption
        if (!cur_key.empty()) {
            cur_val += "\n" + line;
        }
    }
    save_field();

    // Extract lyrics: everything after </think>, trim "# Lyric" header
    std::string after = after_think;
    // Trim leading whitespace
    size_t first = after.find_first_not_of(" \t\n\r");
    if (first != std::string::npos) after = after.substr(first);
    // Remove "# Lyric" or "# Lyrics" header
    if (after.size() > 8 && after[0] == '#') {
        size_t nl = after.find('\n');
        if (nl != std::string::npos) {
            std::string header = after.substr(0, nl);
            // Check if it's a lyric header
            for (auto &c : header) c = tolower(c);
            if (header.find("lyric") != std::string::npos)
                after = after.substr(nl + 1);
        }
    }
    // Trim
    while (!after.empty() && (after.back() == ' ' || after.back() == '\n' || after.back() == '\r'))
        after.pop_back();
    first = after.find_first_not_of(" \t\n\r");
    out->lyrics = (first != std::string::npos) ? after.substr(first) : "";

    fprintf(stderr, "[Parse] bpm=%d duration=%.0f keyscale=%s timesig=%s lang=%s caption=%zuB lyrics=%zuB\n",
            out->bpm, out->duration, out->keyscale.c_str(), out->timesignature.c_str(),
            out->vocal_language.c_str(), out->caption.size(), out->lyrics.size());
    return out->bpm > 0 && out->duration > 0;
}

// Build chat-template prompt with custom instruction and user content.
// System: "# Instruction\n{instruction}\n\n"
// User: "{user_content}"
// + add_generation_prompt (assistant turn start)
static std::vector<int> build_custom_prompt(BPETokenizer &bpe,
                                             const std::string &instruction,
                                             const std::string &user_content) {
    std::vector<int> ids;
    auto append_bpe = [&](const std::string &text) {
        auto t = bpe_encode(&bpe, text, false);
        ids.insert(ids.end(), t.begin(), t.end());
    };
    ids.push_back(TOKEN_IM_START);
    append_bpe("system\n# Instruction\n" + instruction + "\n\n");
    ids.push_back(TOKEN_IM_END);
    append_bpe("\n");
    ids.push_back(TOKEN_IM_START);
    append_bpe("user\n" + user_content);
    ids.push_back(TOKEN_IM_END);
    append_bpe("\n");
    ids.push_back(TOKEN_IM_START);
    append_bpe("assistant\n");
    return ids;
}

// Write enriched prompt fields as individual text files in a directory
static bool write_output_dir(const char *dir, const AcePrompt &ace) {
    std::string d(dir);
    auto write_file = [&](const char *name, const std::string &val) {
        std::string path = d + "/" + name;
        FILE *f = fopen(path.c_str(), "w");
        if (!f) { fprintf(stderr, "ERROR: cannot write %s\n", path.c_str()); return; }
        fwrite(val.data(), 1, val.size(), f);
        fclose(f);
    };
    write_file("caption", ace.caption);
    write_file("lyrics", ace.lyrics);
    if (ace.bpm > 0) write_file("bpm", std::to_string(ace.bpm));
    if (ace.duration > 0) write_file("duration", std::to_string((int)ace.duration));
    if (!ace.keyscale.empty()) write_file("keyscale", ace.keyscale);
    if (!ace.timesignature.empty()) write_file("timesig", ace.timesignature);
    if (!ace.vocal_language.empty()) write_file("language", ace.vocal_language);
    fprintf(stderr, "[Output] metadata -> %s/\n", dir);
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
    ids.push_back(TOKEN_IM_START);
    append_bpe("system\n# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n");
    ids.push_back(TOKEN_IM_END);
    append_bpe("\n");
    // User turn
    ids.push_back(TOKEN_IM_START);
    append_bpe("user\n# Caption\n" + prompt.caption + "\n\n# Lyric\n" + prompt.lyrics + "\n");
    ids.push_back(TOKEN_IM_END);
    append_bpe("\n");
    // Assistant turn (generation starts here)
    ids.push_back(TOKEN_IM_START);
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
    ids.push_back(TOKEN_IM_START);
    append_bpe("system\n# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n");
    ids.push_back(TOKEN_IM_END);
    append_bpe("\n");
    // User turn: caption replaced or removed
    ids.push_back(TOKEN_IM_START);
    bool has_neg = negative_prompt && strlen(negative_prompt) > 0
                   && strcmp(negative_prompt, "NO USER INPUT") != 0;
    if (has_neg) {
        // Meaningful negative prompt: use as caption
        append_bpe("user\n# Caption\n" + std::string(negative_prompt) + "\n\n# Lyric\n" + prompt.lyrics + "\n");
    } else {
        // Default: remove caption, keep lyrics only (matches Python CoT-phase behavior)
        append_bpe("user\n# Lyric\n" + prompt.lyrics + "\n");
    }
    ids.push_back(TOKEN_IM_END);
    append_bpe("\n");
    // Assistant turn
    ids.push_back(TOKEN_IM_START);
    append_bpe("assistant\n");
    return ids;
}

// Build CoT YAML content from user metadata (matching Python yaml.dump(sort_keys=True))
// Returns ONLY the YAML lines. The <think>/</think> special tokens are handled by the caller.
// Only emits the 4 metadata fields the LLM was trained on: bpm, duration, keyscale, timesignature
// caption and language are NEVER in the CoT. They reach the LLM via user turn and DiT via text prompts.
static std::string build_cot_yaml(const AcePrompt &prompt) {
    // Match Python yaml.dump(items, allow_unicode=True, sort_keys=True)
    // Fields in alphabetical order: bpm, caption, duration, keyscale, language, timesig
    // Long values get yaml-style word-wrap at ~80 chars with 2-space indent
    auto yaml_wrap = [](const std::string &key, const std::string &val) -> std::string {
        // Match PyYAML: write words, at each space check if col > 80, if yes break
        std::string result = key + ":";
        int col = (int)(key.size() + 1);
        size_t i = 0;
        while (i < val.size()) {
            size_t end = val.find(' ', i);
            if (end == std::string::npos) end = val.size();
            std::string word = val.substr(i, end - i);
            if (col > 80) {
                result += "\n  ";
                col = 2;
            } else {
                result += " ";
                col += 1;
            }
            result += word;
            col += (int)word.size();
            i = (end < val.size()) ? end + 1 : val.size();
        }
        result += "\n";
        return result;
    };

    std::string yaml;
    if (prompt.bpm > 0)
        yaml += "bpm: " + std::to_string(prompt.bpm) + "\n";
    if (!prompt.caption.empty())
        yaml += yaml_wrap("caption", prompt.caption);
    if (prompt.duration > 0)
        yaml += "duration: " + std::to_string((int)prompt.duration) + "\n";
    if (!prompt.keyscale.empty())
        yaml += "keyscale: " + prompt.keyscale + "\n";
    if (!prompt.vocal_language.empty())
        yaml += "language: " + prompt.vocal_language + "\n";
    if (!prompt.timesignature.empty()) {
        // yaml.dump converts digit strings to int (no quotes)
        yaml += "timesignature: " + prompt.timesignature + "\n";
    }
    return yaml;
}

// Special token IDs (Qwen3 vocab)
// Build prompt with pre-built CoT injected in assistant turn (Phase 2 mode)
// Token sequence matches Python apply_chat_template exactly:
//   <|im_start|> "assistant\n" <think> "\n" {yaml} </think> "\n\n" <|im_end|> "\n"
static std::vector<int> build_lm_prompt_with_cot(BPETokenizer &bpe, const AcePrompt &prompt,
                                                  const std::string &cot_yaml) {
    std::vector<int> ids;
    auto append_bpe = [&](const std::string &text) {
        auto t = bpe_encode(&bpe, text, false);
        ids.insert(ids.end(), t.begin(), t.end());
    };
    // System turn
    ids.push_back(TOKEN_IM_START);
    append_bpe("system\n# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n");
    ids.push_back(TOKEN_IM_END);
    append_bpe("\n");
    // User turn
    ids.push_back(TOKEN_IM_START);
    append_bpe("user\n# Caption\n" + prompt.caption + "\n\n# Lyric\n" + prompt.lyrics + "\n");
    ids.push_back(TOKEN_IM_END);
    append_bpe("\n");
    // Assistant turn with injected CoT (special tokens, not BPE text)
    ids.push_back(TOKEN_IM_START);
    append_bpe("assistant\n");
    ids.push_back(TOKEN_THINK);
    append_bpe("\n" + cot_yaml);  // yaml already ends with \n
    ids.push_back(TOKEN_THINK_END);
    append_bpe("\n\n");
    ids.push_back(TOKEN_IM_END);
    append_bpe("\n");
    return ids;
}

// Build unconditional prompt with empty CoT for CFG (Phase 2 mode)
// Token sequence: <|im_start|> "assistant\n" <think> "\n\n" </think> "\n\n" <|im_end|> "\n"
static std::vector<int> build_lm_prompt_uncond_with_cot(BPETokenizer &bpe, const AcePrompt &prompt,
                                                         const char *negative_prompt) {
    std::vector<int> ids;
    auto append_bpe = [&](const std::string &text) {
        auto t = bpe_encode(&bpe, text, false);
        ids.insert(ids.end(), t.begin(), t.end());
    };
    // System turn
    ids.push_back(TOKEN_IM_START);
    append_bpe("system\n# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n");
    ids.push_back(TOKEN_IM_END);
    append_bpe("\n");
    // User turn: use negative_prompt as caption, or keep original caption
    ids.push_back(TOKEN_IM_START);
    bool has_neg = negative_prompt && strlen(negative_prompt) > 0
                   && strcmp(negative_prompt, "NO USER INPUT") != 0;
    std::string cap = has_neg ? std::string(negative_prompt) : prompt.caption;
    append_bpe("user\n# Caption\n" + cap + "\n\n# Lyric\n" + prompt.lyrics + "\n");
    ids.push_back(TOKEN_IM_END);
    append_bpe("\n");
    // Assistant turn with empty CoT (special tokens)
    ids.push_back(TOKEN_IM_START);
    append_bpe("assistant\n");
    ids.push_back(TOKEN_THINK);
    append_bpe("\n\n");  // empty CoT: just \n\n between think tags
    ids.push_back(TOKEN_THINK_END);
    append_bpe("\n\n");
    ids.push_back(TOKEN_IM_END);
    append_bpe("\n");
    return ids;
}

// Main

static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s --model <dir> [options]\n\n", prog);
    fprintf(stderr, "Model:\n");
    fprintf(stderr, "  --model <dir>          Model directory (safetensors + config.json)\n\n");
    fprintf(stderr, "Custom mode (--system + --user):\n");
    fprintf(stderr, "  --system <text>        System instruction (e.g. INSPIRED/REWRITE instruction)\n");
    fprintf(stderr, "  --user <text>          User content (query, caption+lyrics, etc.)\n\n");
    fprintf(stderr, "Standard mode (--caption):\n");
    fprintf(stderr, "  --caption <text>       Music caption/description\n");
    fprintf(stderr, "  --lyrics <text>        Lyrics text (use [Instrumental] for instrumental)\n");
    fprintf(stderr, "  --bpm <n>              BPM (0 = generate via CoT)\n");
    fprintf(stderr, "  --duration <f>         Duration in seconds (default: 120)\n");
    fprintf(stderr, "  --keyscale <text>      Key scale (e.g. \"F# minor\")\n");
    fprintf(stderr, "  --timesignature <text>  Time signature (e.g. \"4\")\n");
    fprintf(stderr, "  --language <text>      Vocal language code (e.g. \"en\")\n\n");
    fprintf(stderr, "Generation control:\n");
    fprintf(stderr, "  --cfg-scale <f>        CFG scale (default: 1.0 = off)\n");
    fprintf(stderr, "  --negative-prompt <s>  Negative prompt for CFG\n");
    fprintf(stderr, "  --no-codes             Skip audio codes generation\n");
    fprintf(stderr, "  --fsm                  Enable FSM constrained decoding for metadata\n\n");
    fprintf(stderr, "Output:\n");
    fprintf(stderr, "  --output-dir <dir>     Write codes + metadata for dit-vae\n");
    fprintf(stderr, "  --output-text <file>   Write raw LLM output text\n\n");
    fprintf(stderr, "Sampling:\n");
    fprintf(stderr, "  --max-tokens <n>       Max new tokens (default: auto)\n");
    fprintf(stderr, "  --temperature <f>      Sampling temperature (default: 0.8)\n");
    fprintf(stderr, "  --top-p <f>            Top-p sampling (default: 0.9)\n");
    fprintf(stderr, "  --max-seq <n>          Max KV cache length (default: 8192)\n");
    fprintf(stderr, "  --seed <n>             Random seed (default: random)\n");
}

// Write audio codes to file (shared helper)
static bool write_codes(const char *dir, const std::vector<int> &codes) {
    std::string path = std::string(dir) + "/codes";
    FILE *f = fopen(path.c_str(), "w");
    if (!f) { fprintf(stderr, "ERROR: cannot write %s\n", path.c_str()); return false; }
    for (size_t i = 0; i < codes.size(); i++) {
        if (i > 0) fprintf(f, ",");
        fprintf(f, "%d", codes[i]);
    }
    fprintf(f, "\n"); fclose(f);
    fprintf(stderr, "[Output] %s (%zu audio codes)\n", path.c_str(), codes.size());
    return true;
}

// Phase 2: reset KV, generate codes with CFG + CoT injected from AcePrompt
static int run_phase2(QwenModel *model, BPETokenizer &bpe, const AcePrompt &ace,
                      std::vector<int> &prompt, float temperature, float top_p, int seed,
                      float cfg_scale, const char *negative_prompt, const char *output_dir) {
    fprintf(stderr, "[Phase2] Resetting KV cache\n");
    cudaMemset(model->kv_cache, 0, model->kv_bytes);
    if (model->kv_cache_uncond)
        cudaMemset(model->kv_cache_uncond, 0, model->kv_bytes);

    std::string cot = build_cot_yaml(ace);
    prompt = build_lm_prompt_with_cot(bpe, ace, cot);
    std::vector<int> uncond;
    if (cfg_scale > 1.0f)
        uncond = build_lm_prompt_uncond_with_cot(bpe, ace, negative_prompt);

    int codes_max = (int)(ace.duration * 5) + 100;
    fprintf(stderr, "[Phase2] Prompt: %zu tokens, max_codes: %d, CFG: %.2f\n",
            prompt.size(), codes_max, cfg_scale);

    double prefill_ms = 0, decode_ms = 0;
    std::vector<int> audio_codes;
    generate(model, prompt, codes_max, temperature, top_p, seed,
             output_dir ? &audio_codes : nullptr,
             cfg_scale, uncond.empty() ? nullptr : &uncond,
             true, &prefill_ms, &decode_ms);

    if (output_dir && !audio_codes.empty())
        write_codes(output_dir, audio_codes);
    return (int)audio_codes.size();
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(argv[0]); return 1; }

    const char *model_dir = nullptr;
    std::vector<int> prompt;
    int max_tokens = 256;
    float temperature = 0.8f;
    float top_p = 0.9f;
    int max_seq = 8192;
    int seed = -1;
    float cfg_scale = 1.0f;
    const char *negative_prompt = "NO USER INPUT";
    bool no_codes = false;
    bool use_fsm = false;

    // Custom mode args
    const char *system_msg = nullptr;
    const char *user_msg = nullptr;

    // Standard mode args
    const char *cli_caption = nullptr;
    const char *cli_lyrics = nullptr;
    int cli_bpm = 0;
    float cli_duration = 120.0f;
    const char *cli_keyscale = nullptr;
    const char *cli_timesig = nullptr;
    const char *cli_language = nullptr;

    // Output args
    const char *output_dir = nullptr;
    const char *output_text = nullptr;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc)
            model_dir = argv[++i];
        else if (!strcmp(argv[i], "--system") && i + 1 < argc)
            system_msg = argv[++i];
        else if (!strcmp(argv[i], "--user") && i + 1 < argc)
            user_msg = argv[++i];
        else if (!strcmp(argv[i], "--caption") && i + 1 < argc)
            cli_caption = argv[++i];
        else if (!strcmp(argv[i], "--lyrics") && i + 1 < argc)
            cli_lyrics = argv[++i];
        else if (!strcmp(argv[i], "--bpm") && i + 1 < argc)
            cli_bpm = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--duration") && i + 1 < argc)
            cli_duration = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--keyscale") && i + 1 < argc)
            cli_keyscale = argv[++i];
        else if (!strcmp(argv[i], "--timesignature") && i + 1 < argc)
            cli_timesig = argv[++i];
        else if (!strcmp(argv[i], "--language") && i + 1 < argc)
            cli_language = argv[++i];
        else if (!strcmp(argv[i], "--output-dir") && i + 1 < argc)
            output_dir = argv[++i];
        else if (!strcmp(argv[i], "--output-text") && i + 1 < argc)
            output_text = argv[++i];
        else if (!strcmp(argv[i], "--no-codes"))
            no_codes = true;
        else if (!strcmp(argv[i], "--fsm"))
            use_fsm = true;
        else if (!strcmp(argv[i], "--max-tokens") && i + 1 < argc)
            max_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--temperature") && i + 1 < argc)
            temperature = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--top-p") && i + 1 < argc)
            top_p = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--max-seq") && i + 1 < argc)
            max_seq = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc)
            seed = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cfg-scale") && i + 1 < argc)
            cfg_scale = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--negative-prompt") && i + 1 < argc)
            negative_prompt = argv[++i];
        else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    // Validate: need --model
    if (!model_dir) {
        fprintf(stderr, "ERROR: --model required\n");
        usage(argv[0]);
        return 1;
    }

    // Validate: need either --system+--user or --caption
    if (!system_msg && !cli_caption) {
        fprintf(stderr, "ERROR: provide --system + --user (custom mode) or --caption (standard mode)\n");
        usage(argv[0]);
        return 1;
    }
    if (system_msg && !user_msg) {
        fprintf(stderr, "ERROR: --system requires --user\n");
        return 1;
    }
    if (system_msg && cli_caption) {
        fprintf(stderr, "ERROR: --system and --caption are mutually exclusive\n");
        return 1;
    }

    if (seed < 0) {
        std::random_device rd;
        seed = (int)(rd() & 0x7FFFFFFF);
    }

    Timer t_total;
    BPETokenizer bpe;
    if (!load_bpe_tokenizer(&bpe, model_dir)) return 1;

    QwenConfig qcfg = load_config(model_dir);
    qcfg.max_seq_len = max_seq;

    Timer t_load;
    QwenModel model;
    load_model(&model, model_dir, qcfg);
    double load_ms = t_load.ms();

    MetadataFSM fsm;
    if (use_fsm)
        fsm.init(bpe, model.cfg.vocab_size);

    // Helper: write raw text to file
    auto write_text = [](const char *path, const std::string &text) {
        FILE *f = fopen(path, "w");
        if (!f) { fprintf(stderr, "ERROR: cannot write %s\n", path); return; }
        fwrite(text.data(), 1, text.size(), f);
        fclose(f);
        fprintf(stderr, "[Output] %s (%zuB text)\n", path, text.size());
    };

    if (system_msg) {
        // Custom mode: Phase 1 with custom system/user prompt, then optional Phase 2
        fprintf(stderr, "[Custom] system: %.60s...\n", system_msg);

        std::vector<int> p1_prompt = build_custom_prompt(bpe, system_msg, user_msg);
        fprintf(stderr, "[Phase1] %zu tokens, seed: %d\n", p1_prompt.size(), seed);

        fsm.reset();
        std::vector<int> gen_tokens = generate_text(&model, p1_prompt, 2048, temperature, top_p, seed,
                                                    use_fsm ? &fsm : nullptr);
        std::string gen_text = bpe_decode(bpe, gen_tokens);
        fprintf(stderr, "[Phase1] %zu tokens decoded, %zuB text\n", gen_tokens.size(), gen_text.size());
        fprintf(stderr, "[Phase1] output:\n%s\n", gen_text.c_str());

        if (output_text) write_text(output_text, gen_text);

        AcePrompt ace = {};
        if (!parse_cot_and_lyrics(gen_text, &ace)) {
            fprintf(stderr, "ERROR: failed to parse Phase 1 output (missing bpm or duration)\n");
            return 1;
        }
        if (ace.duration <= 0) ace.duration = 120.0f;
        if (ace.duration > 600) ace.duration = 600.0f;

        if (output_dir) write_output_dir(output_dir, ace);

        if (!no_codes) {
            run_phase2(&model, bpe, ace, prompt, temperature, top_p, seed,
                       cfg_scale, negative_prompt, output_dir);
        }

        fprintf(stderr, "[Ace-Qwen3] Load %.0f | Total %.0fms\n", load_ms, t_total.ms());

    } else {
        // Standard mode: build AcePrompt from CLI args
        AcePrompt ace = {};
        ace.caption = cli_caption;
        ace.lyrics = cli_lyrics ? cli_lyrics : "";
        ace.duration = cli_duration;
        ace.bpm = cli_bpm;
        ace.keyscale = cli_keyscale ? cli_keyscale : "";
        ace.timesignature = cli_timesig ? cli_timesig : "";
        ace.vocal_language = cli_language ? cli_language : "";

        // Python strips "/4" suffix: "4/4" -> "4" (llm_inference.py:1018-1019)
        {
            auto &ts = ace.timesignature;
            if (ts.size() >= 2 && ts.compare(ts.size()-2, 2, "/4") == 0)
                ts = ts.substr(0, ts.size()-2);
        }

        bool has_all_metas = ace.bpm > 0 && !ace.keyscale.empty()
                             && !ace.timesignature.empty() && ace.duration > 0;

        if (has_all_metas && no_codes) {
            if (output_dir) write_output_dir(output_dir, ace);
            fprintf(stderr, "[All-metas] No codes requested, prompt written\n");
            fprintf(stderr, "[Ace-Qwen3] Load %.0f | Total %.0fms\n", load_ms, t_total.ms());

        } else if (has_all_metas) {
            std::string cot = build_cot_yaml(ace);
            prompt = build_lm_prompt_with_cot(bpe, ace, cot);
            std::vector<int> uncond;
            if (cfg_scale > 1.0f)
                uncond = build_lm_prompt_uncond_with_cot(bpe, ace, negative_prompt);
            if (max_tokens == 256) max_tokens = (int)(ace.duration * 5) + 100;

            fprintf(stderr, "[All-metas] %zu tokens, max: %d, CFG: %.2f, seed: %d\n",
                    prompt.size(), max_tokens, cfg_scale, seed);

            double prefill_ms = 0, decode_ms = 0;
            std::vector<int> audio_codes;
            generate(&model, prompt, max_tokens, temperature, top_p, seed,
                     output_dir ? &audio_codes : nullptr,
                     cfg_scale, uncond.empty() ? nullptr : &uncond,
                     true, &prefill_ms, &decode_ms);

            if (output_dir && !audio_codes.empty())
                write_codes(output_dir, audio_codes);
            if (output_dir) write_output_dir(output_dir, ace);

            fprintf(stderr, "[Ace-Qwen3] Load %.0f | Prefill %.0f | Decode %.0f | Total %.0fms\n",
                    load_ms, prefill_ms, decode_ms, t_total.ms());

        } else {
            fprintf(stderr, "[Partial-metas] Two-phase generation\n");

            prompt = build_lm_prompt(bpe, ace);
            std::vector<int> uncond;
            if (cfg_scale > 1.0f)
                uncond = build_lm_prompt_uncond(bpe, ace, negative_prompt);

            fprintf(stderr, "[Phase1] %zu tokens, CFG: %.2f, seed: %d\n",
                    prompt.size(), cfg_scale, seed);

            std::vector<int> gen_tokens;
            double p1_prefill = 0, p1_decode = 0;
            fsm.reset();
            generate(&model, prompt, 2048, temperature, top_p, seed,
                     nullptr, cfg_scale, uncond.empty() ? nullptr : &uncond,
                     false, &p1_prefill, &p1_decode,
                     true, &gen_tokens, use_fsm ? &fsm : nullptr);

            std::string gen_text = bpe_decode(bpe, gen_tokens);
            fprintf(stderr, "[Phase1] %zu tokens decoded, %zuB text\n", gen_tokens.size(), gen_text.size());
            fprintf(stderr, "[Partial-metas] CoT:\n%s\n", gen_text.c_str());

            if (output_text) write_text(output_text, gen_text);

            AcePrompt parsed = ace;
            if (!parse_cot_and_lyrics(gen_text, &parsed)) {
                fprintf(stderr, "WARNING: CoT parse incomplete, using available fields\n");
            }
            if (parsed.bpm > 0) ace.bpm = parsed.bpm;
            if (parsed.duration > 0) ace.duration = parsed.duration;
            if (!parsed.keyscale.empty()) ace.keyscale = parsed.keyscale;
            if (!parsed.timesignature.empty()) ace.timesignature = parsed.timesignature;
            if (!parsed.vocal_language.empty()) ace.vocal_language = parsed.vocal_language;
            if (!parsed.caption.empty()) ace.caption = parsed.caption;

            if (ace.duration <= 0) ace.duration = 120.0f;
            if (ace.duration > 600) ace.duration = 600.0f;

            if (output_dir) write_output_dir(output_dir, ace);

            if (!no_codes) {
                run_phase2(&model, bpe, ace, prompt, temperature, top_p, seed,
                           cfg_scale, negative_prompt, output_dir);
            }

            fprintf(stderr, "[Ace-Qwen3] Load %.0f | Total %.0fms\n", load_ms, t_total.ms());
        }
    }

    return 0;
}
