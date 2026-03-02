// qwen3-lm.h : Qwen3 causal LM with KV cache (GGML)
// Autoregressive text + audio code generation for ACE-Step
// Loads from GGUF, supports prefill + decode, tied lm_head
#pragma once

#include "qwen3-enc.h" // Qwen3Layer, Qwen3Config, layer build helpers
#include "ggml-alloc.h"
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
    int max_seq_len; // KV cache capacity
};

// KV cache set (one per CFG path: conditional + unconditional)
#define QW3LM_MAX_KV_SETS 32 // batch N * 2 (cond + uncond CFG)
#define QW3LM_MAX_LAYERS  64

struct Qwen3LM {
    Qwen3LMConfig cfg;

    // Weights (on backend)
    struct ggml_tensor * embed_tokens; // [H, V] on GPU (used by mul_mat lm_head)
    struct ggml_tensor * final_norm;   // [H]
    // lm_head = embed_tokens when tie_embeddings
    Qwen3Layer layers[QW3LM_MAX_LAYERS];

    WeightCtx wctx;
    ggml_backend_t backend;
    ggml_backend_t cpu_backend;
    ggml_backend_sched_t sched;
    bool use_flash_attn;
    bool clamp_fp16; // clamp hidden state on sub-Ampere CUDA (FP16 accumulation overflow)

    // KV cache: per-set, per-layer [D, max_seq, Nkv] f16
    struct ggml_context  * kv_ctx;
    ggml_backend_buffer_t  kv_buf;
    // 4D batched: per-layer [D, max_seq, Nkv, n_sets] for batched flash_attn
    struct ggml_tensor * kv_k4[QW3LM_MAX_LAYERS];
    struct ggml_tensor * kv_v4[QW3LM_MAX_LAYERS];
    // 3D views: per-set, per-layer [D, max_seq, Nkv] for prefill/copy_kv
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

// Load config from GGUF KV metadata (acestep.config_json)
static Qwen3LMConfig qw3lm_load_config(const GGUFModel & gf) {
    // 0.6B defaults
    Qwen3LMConfig c = {
        /*vocab_size*/        217204,
        /*hidden_size*/       1024,
        /*intermediate_size*/ 3072,
        /*n_heads*/           16,
        /*n_kv_heads*/        8,
        /*head_dim*/          128,
        /*n_layers*/          28,
        /*rope_theta*/        1000000.0f,
        /*rms_norm_eps*/      1e-6f,
        /*tie_embeddings*/    true,
        /*max_seq_len*/       8192,
    };

    const char * j = gf_get_str(gf, "acestep.config_json");
    if (!j || !j[0]) {
        fprintf(stderr, "[LM-Config] No acestep.config_json, using 0.6B defaults\n");
        return c;
    }

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
    BackendPair bp = backend_init("LM");
    m->backend = bp.backend;
    m->cpu_backend = bp.cpu_backend;
    m->sched = backend_sched_new(bp, 8192);
    m->use_flash_attn = true;
    m->clamp_fp16 = (bp.gpu_cc > 0 && bp.gpu_cc < 800);
    if (m->clamp_fp16) {
        fprintf(stderr, "[LM] FP16 clamp enabled (cc=%d)\n", bp.gpu_cc);
    }
}

// Allocate KV cache
static void qw3lm_alloc_kv_cache(Qwen3LM * m, int n_sets) {
    const Qwen3LMConfig & c = m->cfg;
    int D   = c.head_dim;
    int Nkv = c.n_kv_heads;
    int L   = c.n_layers;
    int S   = c.max_seq_len;

    m->n_kv_sets = n_sets;

    // 4D tensors [D, S, Nkv, n_sets] + 3D views [D, S, Nkv] per set
    int n_tensors = L * 2 + n_sets * L * 2;  // 4D + views
    size_t ctx_size = (size_t)n_tensors * ggml_tensor_overhead() + 1024;
    struct ggml_init_params gp = { ctx_size, NULL, true };
    m->kv_ctx = ggml_init(gp);

    for (int l = 0; l < L; l++) {
        // 4D batched tensors (allocated by backend)
        m->kv_k4[l] = ggml_new_tensor_4d(m->kv_ctx, GGML_TYPE_F16, D, S, Nkv, n_sets);
        m->kv_v4[l] = ggml_new_tensor_4d(m->kv_ctx, GGML_TYPE_F16, D, S, Nkv, n_sets);
        char name[64];
        snprintf(name, sizeof(name), "kv_k4_%d", l);
        ggml_set_name(m->kv_k4[l], name);
        snprintf(name, sizeof(name), "kv_v4_%d", l);
        ggml_set_name(m->kv_v4[l], name);

        // 3D views per set (backward compat for prefill/copy_kv)
        for (int s = 0; s < n_sets; s++) {
            size_t off = (size_t)s * D * S * Nkv * ggml_type_size(GGML_TYPE_F16);
            m->kv_k[s][l] = ggml_view_3d(m->kv_ctx, m->kv_k4[l], D, S, Nkv,
                m->kv_k4[l]->nb[1], m->kv_k4[l]->nb[2], off);
            m->kv_v[s][l] = ggml_view_3d(m->kv_ctx, m->kv_v4[l], D, S, Nkv,
                m->kv_v4[l]->nb[1], m->kv_v4[l]->nb[2], off);
        }
    }
    for (int s = 0; s < n_sets; s++)
        m->kv_pos[s] = 0;

    m->kv_buf = ggml_backend_alloc_ctx_tensors(m->kv_ctx, m->backend);
    if (!m->kv_buf) {
        fprintf(stderr, "[LM-KV] FATAL: failed to allocate KV cache\n");
        exit(1);
    }

    size_t kv_bytes = (size_t)n_sets * L * 2 * D * S * Nkv * ggml_type_size(GGML_TYPE_F16);
    fprintf(stderr, "[LM-KV] Allocated %d sets x %d layers (4D batched), %.1f MB\n",
            n_sets, L, (float)kv_bytes / (1024 * 1024));
}

// Clear KV cache for a given set
static void qw3lm_reset_kv(Qwen3LM * m, int kv_set) {
    m->kv_pos[kv_set] = 0;
    // No need to zero memory: kv_pos tracks valid range
}

// Copy KV cache from one set to another (for batched prefill sharing)
static void qw3lm_copy_kv(Qwen3LM * m, int src, int dst) {
    for (int l = 0; l < m->cfg.n_layers; l++) {
        ggml_backend_tensor_copy(m->kv_k[src][l], m->kv_k[dst][l]);
        ggml_backend_tensor_copy(m->kv_v[src][l], m->kv_v[dst][l]);
    }
    m->kv_pos[dst] = m->kv_pos[src];
}

// Load model weights from GGUF
static bool qw3lm_load(Qwen3LM * m, const char * gguf_path, int max_seq_len, int n_kv_sets) {
    *m = {};

    qw3lm_init_backend(m);

    GGUFModel gf;
    if (!gf_load(&gf, gguf_path)) {
        fprintf(stderr, "[LM-Load] FATAL: cannot load %s\n", gguf_path);
        return false;
    }

    m->cfg = qw3lm_load_config(gf);
    if (max_seq_len > 0) m->cfg.max_seq_len = max_seq_len;
    const Qwen3LMConfig & c = m->cfg;

    if (c.n_layers > QW3LM_MAX_LAYERS) {
        fprintf(stderr, "[LM-Load] FATAL: %d layers > max %d\n", c.n_layers, QW3LM_MAX_LAYERS);
        gf_close(&gf);
        return false;
    }

    // embed(1) + layers * 11 + final_norm(1) = 2 + n_layers * 11
    int n_tensors = 2 + c.n_layers * 11;
    wctx_init(&m->wctx, n_tensors);

    m->embed_tokens = gf_load_tensor(&m->wctx, gf, "model.embed_tokens.weight");
    m->final_norm   = gf_load_tensor_f32(&m->wctx, gf, "model.norm.weight");

    for (int i = 0; i < c.n_layers; i++) {
        char prefix[64];
        snprintf(prefix, sizeof(prefix), "model.layers.%d", i);
        qwen3_load_layer(&m->wctx, gf, &m->layers[i], prefix, i);
    }

    wctx_alloc(&m->wctx, m->backend);
    gf_close(&gf);

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
        struct ggml_tensor * cache_k, // [D, max_seq, Nkv] f16
        struct ggml_tensor * cache_v, // [D, max_seq, Nkv] f16
        int kv_pos,
        int kv_len,
        int n_tokens,
        bool use_flash_attn = true,
        bool clamp_fp16 = false) {

    int D   = c.head_dim;
    int Nh  = c.n_heads;
    int Nkv = c.n_kv_heads;
    int S   = n_tokens;

    // QKV projections (fused, partial, or separate)
    struct ggml_tensor * q, * k, * v;
    int q_dim  = Nh * D;
    int kv_dim = Nkv * D;
    if (ly->qkv) {
        struct ggml_tensor * qkv = qwen3_linear(ctx, ly->qkv, x);
        q = ggml_cont(ctx, ggml_view_2d(ctx, qkv, q_dim,  S, qkv->nb[1], 0));
        k = ggml_cont(ctx, ggml_view_2d(ctx, qkv, kv_dim, S, qkv->nb[1], (size_t)q_dim * qkv->nb[0]));
        v = ggml_cont(ctx, ggml_view_2d(ctx, qkv, kv_dim, S, qkv->nb[1], (size_t)(q_dim + kv_dim) * qkv->nb[0]));
    } else if (ly->qk) {
        struct ggml_tensor * qk = qwen3_linear(ctx, ly->qk, x);
        q = ggml_cont(ctx, ggml_view_2d(ctx, qk, q_dim,  S, qk->nb[1], 0));
        k = ggml_cont(ctx, ggml_view_2d(ctx, qk, kv_dim, S, qk->nb[1], (size_t)q_dim * qk->nb[0]));
        v = qwen3_linear(ctx, ly->v_proj, x);
    } else {
        q = qwen3_linear(ctx, ly->q_proj, x);
        k = qwen3_linear(ctx, ly->k_proj, x);
        v = qwen3_linear(ctx, ly->v_proj, x);
    }

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
    q = ggml_permute(ctx, q, 0, 2, 1, 3); // [D, S, Nh]
    k = ggml_permute(ctx, k, 0, 2, 1, 3); // [D, S, Nkv]
    v = ggml_permute(ctx, v, 0, 2, 1, 3); // [D, S, Nkv]

    // Make contiguous for cpy to f16 cache
    k = ggml_cont(ctx, k);
    v = ggml_cont(ctx, v);

    // Clamp V before F16 cast: sub-Ampere tensor cores accumulate in FP16,
    // V projection can overflow to inf which corrupts all subsequent attention
    if (clamp_fp16) {
        v = ggml_clamp(ctx, v, -65504.0f, 65504.0f);
    }

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

    // Attention (flash or F32 manual fallback)
    float scale = 1.0f / sqrtf((float)D);
    struct ggml_tensor * attn = use_flash_attn
        ? ggml_flash_attn_ext(ctx, q, k_full, v_full, mask, scale, 0.0f, 0.0f)
        : qwen3_attn_f32(ctx, q, k_full, v_full, mask, scale);
    if (use_flash_attn)
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);

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

    // Graph context (generous fixed allocation)
    size_t ctx_size = (size_t)16384 * ggml_tensor_overhead() + ggml_graph_overhead();
    struct ggml_init_params gp = { ctx_size, NULL, true };
    struct ggml_context * ctx = ggml_init(gp);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16384, false);

    // Inputs
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

    // Embedding via ggml_get_rows (scheduler handles backend fallback)
    struct ggml_tensor * token_ids_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(token_ids_t, "token_ids");
    ggml_set_input(token_ids_t);

    struct ggml_tensor * hidden = ggml_get_rows(ctx, m->embed_tokens, token_ids_t);

    // Transformer layers
    for (int l = 0; l < c.n_layers; l++) {
        Qwen3Layer * ly = &m->layers[l];

        // Pre-attention norm
        struct ggml_tensor * norm = qwen3_rms_norm(ctx, hidden, ly->input_layernorm, c.rms_norm_eps);

        // Self-attention with KV cache
        struct ggml_tensor * attn = qw3lm_build_attn(
            ctx, gf, c, ly, norm, positions, mask,
            m->kv_k[kv_set][l], m->kv_v[kv_set][l],
            kv_pos, kv_len, n_tokens, m->use_flash_attn, m->clamp_fp16);

        // Residual
        hidden = ggml_add(ctx, hidden, attn);
        if (m->clamp_fp16) {
            hidden = ggml_clamp(ctx, hidden, -65504.0f, 65504.0f);
        }

        // Post-attention norm + MLP
        norm = qwen3_rms_norm(ctx, hidden, ly->post_attn_layernorm, c.rms_norm_eps);
        struct ggml_tensor * mlp = qwen3_build_mlp(ctx, ly, norm, n_tokens);
        hidden = ggml_add(ctx, hidden, mlp);
        if (m->clamp_fp16) {
            hidden = ggml_clamp(ctx, hidden, -65504.0f, 65504.0f);
        }
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

    // Set token IDs
    ggml_backend_tensor_set(token_ids_t, token_ids, 0, n_tokens * sizeof(int));

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

// Batched decode forward: N tokens (1 per sequence), batched weight matmuls.
// kv_pos per element from m->kv_pos[kv_sets[i]], supports different prompt lengths.
// kv_sets[N]: which KV set each token uses.
// logits: [N * out_V] output, N logit vectors concatenated.
// lm_offset/lm_count: when lm_count>0, project only [lm_offset..lm_offset+lm_count)
//   of vocab (partial LM head). out_V = lm_count. When 0: full vocab, out_V = V.
static void qw3lm_forward_batch(Qwen3LM * m, const int * token_ids,
                                  const int * kv_sets, int N, float * logits,
                                  int lm_offset = 0, int lm_count = 0) {
    const Qwen3LMConfig & c = m->cfg;
    int D   = c.head_dim;
    int Nh  = c.n_heads;
    int Nkv = c.n_kv_heads;

    // Per-element kv_pos (supports different prompt lengths)
    int max_kv_len = 0;
    for (int i = 0; i < N; i++) {
        int kl = m->kv_pos[kv_sets[i]] + 1;
        if (kl > max_kv_len) max_kv_len = kl;
        if (kl > c.max_seq_len) {
            fprintf(stderr, "[LM-Batch] FATAL: kv_len %d > max_seq %d (set %d)\n",
                    kl, c.max_seq_len, kv_sets[i]);
            exit(1);
        }
    }

    // Graph context (generous fixed allocation, ~6 MB, negligible vs model weights)
    size_t ctx_size = (size_t)16384 * ggml_tensor_overhead() + ggml_graph_overhead_custom(16384, false);
    struct ggml_init_params gp = { ctx_size, NULL, true };
    struct ggml_context * ctx = ggml_init(gp);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16384, false);

    // Embedding via ggml_get_rows (scheduler handles backend fallback)
    struct ggml_tensor * token_ids_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    ggml_set_name(token_ids_t, "token_ids");
    ggml_set_input(token_ids_t);

    // Positions: [N], per-element kv_pos
    struct ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // Batched attention mask: [max_kv_len, 1, 1, N] f16
    // Per-element: 0 for valid KV positions, -inf for padding beyond elem kv_len
    struct ggml_tensor * attn_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, max_kv_len, 1, 1, N);
    ggml_set_name(attn_mask, "attn_mask");
    ggml_set_input(attn_mask);

    struct ggml_tensor * hidden = ggml_get_rows(ctx, m->embed_tokens, token_ids_t);

    for (int l = 0; l < c.n_layers; l++) {
        Qwen3Layer * ly = &m->layers[l];

        // Pre-attention norm [H, N]
        struct ggml_tensor * norm = qwen3_rms_norm(ctx, hidden, ly->input_layernorm, c.rms_norm_eps);

        // Batched QKV projections (fused, partial, or separate)
        struct ggml_tensor * q, * k, * v;
        int q_dim  = Nh * D;
        int kv_dim = Nkv * D;
        if (ly->qkv) {
            struct ggml_tensor * qkv = qwen3_linear(ctx, ly->qkv, norm);
            q = ggml_cont(ctx, ggml_view_2d(ctx, qkv, q_dim,  N, qkv->nb[1], 0));
            k = ggml_cont(ctx, ggml_view_2d(ctx, qkv, kv_dim, N, qkv->nb[1], (size_t)q_dim * qkv->nb[0]));
            v = ggml_cont(ctx, ggml_view_2d(ctx, qkv, kv_dim, N, qkv->nb[1], (size_t)(q_dim + kv_dim) * qkv->nb[0]));
        } else if (ly->qk) {
            struct ggml_tensor * qk = qwen3_linear(ctx, ly->qk, norm);
            q = ggml_cont(ctx, ggml_view_2d(ctx, qk, q_dim,  N, qk->nb[1], 0));
            k = ggml_cont(ctx, ggml_view_2d(ctx, qk, kv_dim, N, qk->nb[1], (size_t)q_dim * qk->nb[0]));
            v = qwen3_linear(ctx, ly->v_proj, norm);
        } else {
            q = qwen3_linear(ctx, ly->q_proj, norm);
            k = qwen3_linear(ctx, ly->k_proj, norm);
            v = qwen3_linear(ctx, ly->v_proj, norm);
        }

        // Reshape to heads: [D, Heads, N]
        q = ggml_reshape_3d(ctx, q, D, Nh,  N);
        k = ggml_reshape_3d(ctx, k, D, Nkv, N);
        v = ggml_reshape_3d(ctx, v, D, Nkv, N);

        // QK-Norm (rms_norm on dim0=D, per head per seq)
        q = ggml_rms_norm(ctx, q, c.rms_norm_eps);
        q = ggml_mul(ctx, q, qwen3_f32(ctx, ly->q_norm));
        k = ggml_rms_norm(ctx, k, c.rms_norm_eps);
        k = ggml_mul(ctx, k, qwen3_f32(ctx, ly->k_norm));

        // RoPE: positions [N] maps to dim 2 of [D, Heads, N]
        q = ggml_rope_ext(ctx, q, positions, NULL, D, 2, 0,
                           c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx, k, positions, NULL, D, 2, 0,
                           c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // Contiguous for clean slicing
        q = ggml_cont(ctx, q);
        k = ggml_cont(ctx, k);
        v = ggml_cont(ctx, v);

        // Clamp V before F16 cast (sub-Ampere FP16 accumulation overflow)
        if (m->clamp_fp16) {
            v = ggml_clamp(ctx, v, -65504.0f, 65504.0f);
        }

        // Batched attention with 4D KV cache
        float scale = 1.0f / sqrtf((float)D);

        // Per-element: write new K,V to 4D KV cache
        for (int i = 0; i < N; i++) {
            int set = kv_sets[i];
            int elem_kv_pos = m->kv_pos[set];
            size_t off_4d = (size_t)set * m->kv_k4[l]->nb[3]
                          + (size_t)elem_kv_pos * m->kv_k4[l]->nb[1];

            // Slice new K,V for element i: [D, Nkv, 1] from [D, Nkv, N]
            struct ggml_tensor * ki = ggml_view_3d(ctx, k, D, Nkv, 1,
                k->nb[1], k->nb[2], (size_t)i * k->nb[2]);
            struct ggml_tensor * vi = ggml_view_3d(ctx, v, D, Nkv, 1,
                v->nb[1], v->nb[2], (size_t)i * v->nb[2]);

            // Permute [D, Nkv, 1] -> [D, 1, Nkv] for KV cache layout
            ki = ggml_cont(ctx, ggml_permute(ctx, ki, 0, 2, 1, 3));
            vi = ggml_cont(ctx, ggml_permute(ctx, vi, 0, 2, 1, 3));

            // Write to 4D cache at (kv_pos, set)
            struct ggml_tensor * k_dst = ggml_view_3d(ctx, m->kv_k4[l],
                D, 1, Nkv, m->kv_k4[l]->nb[1], m->kv_k4[l]->nb[2], off_4d);
            struct ggml_tensor * v_dst = ggml_view_3d(ctx, m->kv_v4[l],
                D, 1, Nkv, m->kv_v4[l]->nb[1], m->kv_v4[l]->nb[2], off_4d);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, ki, k_dst));
            ggml_build_forward_expand(gf, ggml_cpy(ctx, vi, v_dst));
        }

        // Q: [D, Nh, N] -> [D, 1, Nh, N] (n_batch=1, ne3=N for batched flash_attn)
        struct ggml_tensor * q4 = ggml_reshape_4d(ctx, q, D, 1, Nh, N);

        // Batched KV read: [D, max_kv_len, Nkv, N] view of 4D cache
        int s0 = kv_sets[0]; // sets are always consecutive: [s0, s0+1, ..., s0+N-1]
        struct ggml_tensor * k_batch = ggml_view_4d(ctx, m->kv_k4[l],
            D, max_kv_len, Nkv, N,
            m->kv_k4[l]->nb[1], m->kv_k4[l]->nb[2], m->kv_k4[l]->nb[3],
            (size_t)s0 * m->kv_k4[l]->nb[3]);
        struct ggml_tensor * v_batch = ggml_view_4d(ctx, m->kv_v4[l],
            D, max_kv_len, Nkv, N,
            m->kv_v4[l]->nb[1], m->kv_v4[l]->nb[2], m->kv_v4[l]->nb[3],
            (size_t)s0 * m->kv_v4[l]->nb[3]);

        // Batched attention (flash or F32 manual fallback)
        struct ggml_tensor * attn_result = m->use_flash_attn
            ? ggml_flash_attn_ext(ctx, q4, k_batch, v_batch, attn_mask, scale, 0.0f, 0.0f)
            : qwen3_attn_f32(ctx, q4, k_batch, v_batch, attn_mask, scale);
        if (m->use_flash_attn)
            ggml_flash_attn_ext_set_prec(attn_result, GGML_PREC_F32);

        // Output: [D, Nh, 1, N] -> [Nh*D, N]
        struct ggml_tensor * attn_cat = ggml_reshape_2d(ctx, attn_result, Nh * D, N);

        // Batched O proj
        struct ggml_tensor * attn_out = qwen3_linear(ctx, ly->o_proj, attn_cat);
        hidden = ggml_add(ctx, hidden, attn_out);
        if (m->clamp_fp16) {
            hidden = ggml_clamp(ctx, hidden, -65504.0f, 65504.0f);
        }

        // Batched FFN
        norm = qwen3_rms_norm(ctx, hidden, ly->post_attn_layernorm, c.rms_norm_eps);
        struct ggml_tensor * mlp = qwen3_build_mlp(ctx, ly, norm, N);
        hidden = ggml_add(ctx, hidden, mlp);
        if (m->clamp_fp16) {
            hidden = ggml_clamp(ctx, hidden, -65504.0f, 65504.0f);
        }
    }

    // Final norm + LM head
    hidden = qwen3_rms_norm(ctx, hidden, m->final_norm, c.rms_norm_eps);
    int out_V = (lm_count > 0) ? lm_count : c.vocab_size;
    struct ggml_tensor * lm_weight = m->embed_tokens;
    if (lm_count > 0) {
        // Partial projection: only [lm_offset..lm_offset+lm_count) rows
        lm_weight = ggml_view_2d(ctx, m->embed_tokens,
            m->embed_tokens->ne[0], lm_count,
            m->embed_tokens->nb[1],
            (int64_t)lm_offset * m->embed_tokens->nb[1]);
    }
    struct ggml_tensor * lgt = ggml_mul_mat(ctx, lm_weight, hidden); // [out_V, N]
    ggml_set_name(lgt, "logits");
    ggml_set_output(lgt);
    ggml_build_forward_expand(gf, lgt);

    // Allocate
    ggml_backend_sched_alloc_graph(m->sched, gf);

    // Set token IDs
    ggml_backend_tensor_set(token_ids_t, token_ids, 0, N * sizeof(int));

    // Positions: per-element kv_pos
    {
        std::vector<int> pos_data(N);
        for (int i = 0; i < N; i++)
            pos_data[i] = m->kv_pos[kv_sets[i]];
        ggml_backend_tensor_set(positions, pos_data.data(), 0, N * sizeof(int));
    }

    // Attention mask: [max_kv_len, 1, 1, N] f16
    // 0.0 for valid KV positions, -inf for padding beyond each element's kv_len
    {
        std::vector<uint16_t> mask_data((size_t)max_kv_len * N);
        for (int i = 0; i < N; i++) {
            int kvl = m->kv_pos[kv_sets[i]] + 1; // kv_len after write
            for (int j = 0; j < max_kv_len; j++)
                mask_data[(size_t)i * max_kv_len + j] =
                    ggml_fp32_to_fp16((j < kvl) ? 0.0f : -INFINITY);
        }
        ggml_backend_tensor_set(attn_mask, mask_data.data(), 0,
            mask_data.size() * sizeof(uint16_t));
    }

    // Compute
    ggml_backend_sched_graph_compute(m->sched, gf);

    // Read logits [out_V, N]
    ggml_backend_tensor_get(lgt, logits, 0, (size_t)out_V * N * sizeof(float));

    // Advance all KV positions
    for (int i = 0; i < N; i++)
        m->kv_pos[kv_sets[i]]++;

    ggml_backend_sched_reset(m->sched);
    ggml_free(ctx);
}

// Free all resources
static void qw3lm_free(Qwen3LM * m) {
    if (m->sched) ggml_backend_sched_free(m->sched);
    if (m->kv_buf) ggml_backend_buffer_free(m->kv_buf);
    if (m->kv_ctx) ggml_free(m->kv_ctx);
    backend_release(m->backend, m->cpu_backend);
    wctx_free(&m->wctx);
    *m = {};
}
