// qwen3-lm.h : Qwen3 causal LM with KV cache (GGML)
// Autoregressive text + audio code generation for ACE-Step
// Loads safetensors, supports prefill + decode, tied lm_head
#pragma once

#include "qwen3.h"    // Qwen3Layer, Qwen3Config, layer build helpers
#include "bpe.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

// LM config (superset of encoder config)
struct Qwen3LMConfig {
    int vocab_size;
    int hidden_size;
    int intermediate_size;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    int n_layers;
    float rope_theta;
    float rms_norm_eps;
    bool tie_embeddings;
    int max_seq_len;        // KV cache capacity
};

// KV cache set (one per CFG path: conditional + unconditional)
#define QW3LM_MAX_KV_SETS 2
#define QW3LM_MAX_LAYERS   64

struct Qwen3LM {
    Qwen3LMConfig cfg;

    // Weights (on backend)
    struct ggml_tensor * embed_tokens;   // [H, V] bf16
    struct ggml_tensor * final_norm;     // [H] bf16
    // lm_head = embed_tokens when tie_embeddings
    Qwen3Layer layers[QW3LM_MAX_LAYERS];

    SFWeightCtx wctx;
    ggml_backend_t backend;
    ggml_backend_t cpu_backend;
    ggml_backend_sched_t sched;

    // KV cache: per-set, per-layer [D, max_seq, Nkv] f16
    struct ggml_context  * kv_ctx;
    ggml_backend_buffer_t  kv_buf;
    struct ggml_tensor * kv_k[QW3LM_MAX_KV_SETS][QW3LM_MAX_LAYERS];
    struct ggml_tensor * kv_v[QW3LM_MAX_KV_SETS][QW3LM_MAX_LAYERS];
    int kv_pos[QW3LM_MAX_KV_SETS];
    int n_kv_sets;
};

// Parse config.json integers, floats, bools
static int qw3lm_json_int(const char * json, const char * key, int fb) {
    char needle[128];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char * p = strstr(json, needle);
    if (!p) return fb;
    p = strchr(p + strlen(needle), ':');
    if (!p) return fb;
    p++;
    while (*p == ' ' || *p == '\t') p++;
    return atoi(p);
}

static float qw3lm_json_float(const char * json, const char * key, float fb) {
    char needle[128];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char * p = strstr(json, needle);
    if (!p) return fb;
    p = strchr(p + strlen(needle), ':');
    if (!p) return fb;
    p++;
    while (*p == ' ' || *p == '\t') p++;
    return (float)atof(p);
}

static bool qw3lm_json_bool(const char * json, const char * key, bool fb) {
    char needle[128];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char * p = strstr(json, needle);
    if (!p) return fb;
    p = strchr(p + strlen(needle), ':');
    if (!p) return fb;
    p++;
    while (*p == ' ' || *p == '\t') p++;
    return (strncmp(p, "true", 4) == 0);
}

// Load config from model_dir/config.json
static Qwen3LMConfig qw3lm_load_config(const char * model_dir) {
    // 0.6B defaults
    Qwen3LMConfig c = {
        .vocab_size       = 217204,
        .hidden_size      = 1024,
        .intermediate_size = 3072,
        .n_heads          = 16,
        .n_kv_heads       = 8,
        .head_dim         = 128,
        .n_layers         = 28,
        .rope_theta       = 1000000.0f,
        .rms_norm_eps     = 1e-6f,
        .tie_embeddings   = true,
        .max_seq_len      = 8192,
    };

    char path[512];
    snprintf(path, sizeof(path), "%s/config.json", model_dir);
    FILE * f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "[LM-Config] No config.json, using 0.6B defaults\n");
        return c;
    }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<char> buf(len + 1);
    fread(buf.data(), 1, len, f);
    buf[len] = 0;
    fclose(f);
    const char * j = buf.data();

    c.vocab_size        = qw3lm_json_int(j, "vocab_size", c.vocab_size);
    c.hidden_size       = qw3lm_json_int(j, "hidden_size", c.hidden_size);
    c.intermediate_size = qw3lm_json_int(j, "intermediate_size", c.intermediate_size);
    c.n_heads           = qw3lm_json_int(j, "num_attention_heads", c.n_heads);
    c.n_kv_heads        = qw3lm_json_int(j, "num_key_value_heads", c.n_kv_heads);
    c.head_dim          = qw3lm_json_int(j, "head_dim", c.head_dim);
    c.n_layers          = qw3lm_json_int(j, "num_hidden_layers", c.n_layers);
    c.rope_theta        = qw3lm_json_float(j, "rope_theta", c.rope_theta);
    c.rms_norm_eps      = qw3lm_json_float(j, "rms_norm_eps", c.rms_norm_eps);
    c.tie_embeddings    = qw3lm_json_bool(j, "tie_word_embeddings", c.tie_embeddings);

    fprintf(stderr, "[LM-Config] %dL, H=%d, V=%d, Nh=%d, Nkv=%d, D=%d, tied=%d\n",
            c.n_layers, c.hidden_size, c.vocab_size, c.n_heads, c.n_kv_heads,
            c.head_dim, c.tie_embeddings);
    return c;
}

// Init backend (same pattern as qwen3.h)
static void qw3lm_init_backend(Qwen3LM * m) {
    ggml_backend_load_all();
    m->backend = ggml_backend_init_best();
    m->cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    fprintf(stderr, "[LM-Load] Backend: %s\n", ggml_backend_name(m->backend));

    ggml_backend_t backends[2] = { m->backend, m->cpu_backend };
    int n_backends = (m->backend == m->cpu_backend) ? 1 : 2;
    m->sched = ggml_backend_sched_new(backends, NULL, n_backends, 8192, false, true);
}

// Allocate KV cache
static void qw3lm_alloc_kv_cache(Qwen3LM * m, int n_sets) {
    const Qwen3LMConfig & c = m->cfg;
    int D   = c.head_dim;
    int Nkv = c.n_kv_heads;
    int L   = c.n_layers;
    int S   = c.max_seq_len;

    m->n_kv_sets = n_sets;

    // Each KV tensor: [D, max_seq, Nkv] f16
    int n_tensors = n_sets * L * 2;
    size_t ctx_size = (size_t)n_tensors * ggml_tensor_overhead() + 1024;
    struct ggml_init_params gp = { ctx_size, NULL, true };
    m->kv_ctx = ggml_init(gp);

    for (int s = 0; s < n_sets; s++) {
        for (int l = 0; l < L; l++) {
            m->kv_k[s][l] = ggml_new_tensor_3d(m->kv_ctx, GGML_TYPE_F16, D, S, Nkv);
            m->kv_v[s][l] = ggml_new_tensor_3d(m->kv_ctx, GGML_TYPE_F16, D, S, Nkv);
            char name[64];
            snprintf(name, sizeof(name), "kv_k_%d_%d", s, l);
            ggml_set_name(m->kv_k[s][l], name);
            snprintf(name, sizeof(name), "kv_v_%d_%d", s, l);
            ggml_set_name(m->kv_v[s][l], name);
        }
        m->kv_pos[s] = 0;
    }

    m->kv_buf = ggml_backend_alloc_ctx_tensors(m->kv_ctx, m->backend);
    if (!m->kv_buf) {
        fprintf(stderr, "[LM-KV] FATAL: failed to allocate KV cache\n");
        exit(1);
    }

    size_t kv_bytes = (size_t)n_sets * L * 2 * D * S * Nkv * ggml_type_size(GGML_TYPE_F16);
    fprintf(stderr, "[LM-KV] Allocated %d sets x %d layers, %.1f MB\n",
            n_sets, L, (float)kv_bytes / (1024 * 1024));
}

// Clear KV cache for a given set
static void qw3lm_reset_kv(Qwen3LM * m, int kv_set) {
    m->kv_pos[kv_set] = 0;
    // No need to zero memory: kv_pos tracks valid range
}

// Load model weights from safetensors
static bool qw3lm_load(Qwen3LM * m, const char * model_dir, int max_seq_len, int n_kv_sets) {
    memset(m, 0, sizeof(*m));

    m->cfg = qw3lm_load_config(model_dir);
    if (max_seq_len > 0) m->cfg.max_seq_len = max_seq_len;
    const Qwen3LMConfig & c = m->cfg;

    if (c.n_layers > QW3LM_MAX_LAYERS) {
        fprintf(stderr, "[LM-Load] FATAL: %d layers > max %d\n", c.n_layers, QW3LM_MAX_LAYERS);
        return false;
    }

    qw3lm_init_backend(m);

    SafeTensors st;
    if (!safe_load(st, model_dir)) {
        fprintf(stderr, "[LM-Load] FATAL: cannot load safetensors from %s\n", model_dir);
        return false;
    }
    fprintf(stderr, "[LM-Load] Safetensors: %zu tensors\n", st.tensors.size());

    // embed(1) + layers * 11 + final_norm(1) = 2 + n_layers * 11
    int n_tensors = 2 + c.n_layers * 11;
    sf_weight_ctx_init(&m->wctx, n_tensors);

    m->embed_tokens = sf_load_tensor(&m->wctx, st, "model.embed_tokens.weight");
    m->final_norm   = sf_load_tensor(&m->wctx, st, "model.norm.weight");

    for (int i = 0; i < c.n_layers; i++) {
        char prefix[64];
        snprintf(prefix, sizeof(prefix), "model.layers.%d", i);
        qwen3_load_layer(&m->wctx, st, &m->layers[i], prefix);
    }

    sf_weight_ctx_alloc(&m->wctx, m->backend);

    // KV cache
    qw3lm_alloc_kv_cache(m, n_kv_sets > 0 ? n_kv_sets : 1);

    return true;
}

// Build self-attention with KV cache write + read
// x: [H, n_tokens], positions: [n_tokens], mask: [kv_len, n_tokens] or NULL
static struct ggml_tensor * qw3lm_build_attn(
        struct ggml_context * ctx,
        struct ggml_cgraph  * gf,
        const Qwen3LMConfig & c,
        Qwen3Layer * ly,
        struct ggml_tensor * x,
        struct ggml_tensor * positions,
        struct ggml_tensor * mask,
        struct ggml_tensor * cache_k,    // [D, max_seq, Nkv] f16
        struct ggml_tensor * cache_v,    // [D, max_seq, Nkv] f16
        int kv_pos,
        int kv_len,
        int n_tokens) {

    int D   = c.head_dim;
    int Nh  = c.n_heads;
    int Nkv = c.n_kv_heads;
    int S   = n_tokens;

    // QKV projections
    struct ggml_tensor * q = qwen3_linear(ctx, ly->q_proj, x);  // [Nh*D, S]
    struct ggml_tensor * k = qwen3_linear(ctx, ly->k_proj, x);  // [Nkv*D, S]
    struct ggml_tensor * v = qwen3_linear(ctx, ly->v_proj, x);  // [Nkv*D, S]

    // Reshape to heads: [X*D, S] -> [D, X, S]
    q = ggml_reshape_3d(ctx, q, D, Nh,  S);
    k = ggml_reshape_3d(ctx, k, D, Nkv, S);
    v = ggml_reshape_3d(ctx, v, D, Nkv, S);

    // QK-Norm
    q = ggml_rms_norm(ctx, q, c.rms_norm_eps);
    q = ggml_mul(ctx, q, qwen3_f32(ctx, ly->q_norm));
    k = ggml_rms_norm(ctx, k, c.rms_norm_eps);
    k = ggml_mul(ctx, k, qwen3_f32(ctx, ly->k_norm));

    // RoPE (NEOX mode=2)
    q = ggml_rope_ext(ctx, q, positions, NULL,
                       D, 2, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, positions, NULL,
                       D, 2, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // Permute for flash_attn: [D, X, S] -> [D, S, X]
    q = ggml_permute(ctx, q, 0, 2, 1, 3);   // [D, S, Nh]
    k = ggml_permute(ctx, k, 0, 2, 1, 3);   // [D, S, Nkv]
    v = ggml_permute(ctx, v, 0, 2, 1, 3);   // [D, S, Nkv]

    // Make contiguous for cpy to f16 cache
    k = ggml_cont(ctx, k);
    v = ggml_cont(ctx, v);

    // Write K,V to cache at kv_pos
    // Cache layout: [D, max_seq, Nkv] f16
    size_t nb1 = (size_t)D * ggml_type_size(GGML_TYPE_F16);
    size_t nb2 = (size_t)D * c.max_seq_len * ggml_type_size(GGML_TYPE_F16);
    size_t off = (size_t)kv_pos * nb1;

    struct ggml_tensor * k_dst = ggml_view_3d(ctx, cache_k, D, S, Nkv, nb1, nb2, off);
    struct ggml_tensor * v_dst = ggml_view_3d(ctx, cache_v, D, S, Nkv, nb1, nb2, off);

    ggml_build_forward_expand(gf, ggml_cpy(ctx, k, k_dst));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, v, v_dst));

    // Read full KV from cache [0..kv_len]
    struct ggml_tensor * k_full = ggml_view_3d(ctx, cache_k, D, kv_len, Nkv, nb1, nb2, 0);
    struct ggml_tensor * v_full = ggml_view_3d(ctx, cache_v, D, kv_len, Nkv, nb1, nb2, 0);

    // Flash attention
    float scale = 1.0f / sqrtf((float)D);
    struct ggml_tensor * attn = ggml_flash_attn_ext(ctx, q, k_full, v_full, mask, scale, 0.0f, 0.0f);

    // Reshape: [D, Nh, S] -> [Nh*D, S]
    attn = ggml_reshape_2d(ctx, attn, Nh * D, S);

    // O projection
    return qwen3_linear(ctx, ly->o_proj, attn);
}

// Forward pass: token_ids[n_tokens] -> logits[vocab_size] (last token only)
// kv_set: which KV cache set to use (0=conditional, 1=unconditional for CFG)
static void qw3lm_forward(Qwen3LM * m, const int * token_ids, int n_tokens,
                            int kv_set, float * logits) {
    const Qwen3LMConfig & c = m->cfg;
    int H = c.hidden_size;
    int kv_pos = m->kv_pos[kv_set];
    int kv_len = kv_pos + n_tokens;

    if (kv_len > c.max_seq_len) {
        fprintf(stderr, "[LM-Forward] FATAL: kv_len %d > max_seq %d\n", kv_len, c.max_seq_len);
        return;
    }

    // Graph context
    size_t ctx_size = (size_t)16384 * ggml_tensor_overhead() + ggml_graph_overhead();
    struct ggml_init_params gp = { ctx_size, NULL, true };
    struct ggml_context * ctx = ggml_init(gp);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16384, false);

    // Inputs
    struct ggml_tensor * t_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(t_ids, "token_ids");
    ggml_set_input(t_ids);

    struct ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // Causal mask: needed for prefill (n_tokens > 1), not for decode (n_tokens == 1)
    struct ggml_tensor * mask = NULL;
    if (n_tokens > 1) {
        mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv_len, n_tokens);
        ggml_set_name(mask, "causal_mask");
        ggml_set_input(mask);
    }

    // Embedding lookup: [H, V] x [n_tokens] -> [H, n_tokens]
    struct ggml_tensor * hidden = ggml_get_rows(ctx, m->embed_tokens, t_ids);

    // Transformer layers
    for (int l = 0; l < c.n_layers; l++) {
        Qwen3Layer * ly = &m->layers[l];

        // Pre-attention norm
        struct ggml_tensor * norm = qwen3_rms_norm(ctx, hidden, ly->input_layernorm, c.rms_norm_eps);

        // Self-attention with KV cache
        struct ggml_tensor * attn = qw3lm_build_attn(
            ctx, gf, c, ly, norm, positions, mask,
            m->kv_k[kv_set][l], m->kv_v[kv_set][l],
            kv_pos, kv_len, n_tokens);

        // Residual
        hidden = ggml_add(ctx, hidden, attn);

        // Post-attention norm + MLP
        norm = qwen3_rms_norm(ctx, hidden, ly->post_attn_layernorm, c.rms_norm_eps);
        struct ggml_tensor * mlp = qwen3_build_mlp(ctx, ly, norm, n_tokens);
        hidden = ggml_add(ctx, hidden, mlp);
    }

    // Final norm
    hidden = qwen3_rms_norm(ctx, hidden, m->final_norm, c.rms_norm_eps);

    // Extract last token hidden state: [H, n_tokens] -> [H, 1]
    if (n_tokens > 1) {
        hidden = ggml_view_1d(ctx, hidden, H,
            (int64_t)(n_tokens - 1) * H * sizeof(float));
    }

    // LM head: logits = embed_tokens^T @ hidden -> [V, 1]
    struct ggml_tensor * lgt = ggml_mul_mat(ctx, m->embed_tokens, hidden);
    ggml_set_name(lgt, "logits");
    ggml_set_output(lgt);
    ggml_build_forward_expand(gf, lgt);

    // Schedule + allocate
    ggml_backend_sched_alloc_graph(m->sched, gf);

    // Set input data
    ggml_backend_tensor_set(t_ids, token_ids, 0, n_tokens * sizeof(int));

    {
        std::vector<int> pos_data(n_tokens);
        for (int i = 0; i < n_tokens; i++) pos_data[i] = kv_pos + i;
        ggml_backend_tensor_set(positions, pos_data.data(), 0, n_tokens * sizeof(int));
    }

    if (mask) {
        // Causal mask: [kv_len, n_tokens]
        // Row i (query at position kv_pos+i) can attend to columns [0..kv_pos+i]
        std::vector<uint16_t> mask_data((size_t)kv_len * n_tokens);
        for (int i = 0; i < n_tokens; i++) {
            int query_abs_pos = kv_pos + i;
            for (int j = 0; j < kv_len; j++) {
                float v = (j <= query_abs_pos) ? 0.0f : -INFINITY;
                mask_data[(size_t)i * kv_len + j] = ggml_fp32_to_fp16(v);
            }
        }
        ggml_backend_tensor_set(mask, mask_data.data(), 0,
            (size_t)kv_len * n_tokens * sizeof(uint16_t));
    }

    // Compute
    ggml_backend_sched_graph_compute(m->sched, gf);

    // Read logits [V]
    ggml_backend_tensor_get(lgt, logits, 0, c.vocab_size * sizeof(float));

    // Advance KV position
    m->kv_pos[kv_set] += n_tokens;

    ggml_backend_sched_reset(m->sched);
    ggml_free(ctx);
}

// Free all resources
static void qw3lm_free(Qwen3LM * m) {
    if (m->sched) ggml_backend_sched_free(m->sched);
    if (m->kv_buf) ggml_backend_buffer_free(m->kv_buf);
    if (m->kv_ctx) ggml_free(m->kv_ctx);
    if (m->backend && m->backend != m->cpu_backend) ggml_backend_free(m->backend);
    if (m->cpu_backend) ggml_backend_free(m->cpu_backend);
    sf_weight_ctx_free(&m->wctx);
    memset(m, 0, sizeof(*m));
}
