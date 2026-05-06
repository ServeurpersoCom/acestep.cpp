// qwen3-enc.h: Qwen3 transformer encoder via ggml
//
// Generic Qwen3 backbone used by:
//   Text encoder  (Qwen3-Embedding-0.6B): 28L, H=1024, causal, vocab lookup
//   Lyric encoder (ACEStep cond):         8L,  H=2048, bidir, linear(1024->2048)
//   Timbre encoder (ACEStep cond):        4L,  H=2048, bidir, linear(64->2048)
//
// Architecture per layer:
//   RMSNorm -> Q/K/V proj -> QK-Norm -> RoPE -> GQA -> O proj -> +residual
//   RMSNorm -> gate/up proj -> SwiGLU -> down proj -> +residual
// Final: RMSNorm

#pragma once
#include "backend.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf-weights.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#define QWEN3_MAX_LAYERS 32

// Config
struct Qwen3Config {
    int   hidden_size;        // H
    int   intermediate_size;  // FFN inner dim
    int   n_heads;            // Nh (query heads)
    int   n_kv_heads;         // Nkv (key/value heads, for GQA)
    int   head_dim;           // D = H / Nh
    int   n_layers;
    float rope_theta;
    float rms_norm_eps;
    bool  is_causal;  // true for text encoder, false for lyric/timbre
};

// Per-layer weights
struct Qwen3Layer {
    struct ggml_tensor * input_layernorm;      // [H]
    struct ggml_tensor * post_attn_layernorm;  // [H]

    // Attention (fused or separate, same pattern as DiT)
    struct ggml_tensor * qkv;     // [H, (Nh+2*Nkv)*D] full fused (or NULL)
    struct ggml_tensor * qk;      // [H, (Nh+Nkv)*D] Q+K fused (or NULL)
    struct ggml_tensor * q_proj;  // [H, Nh*D]  (NULL when fused)
    struct ggml_tensor * k_proj;  // [H, Nkv*D] (NULL when fused)
    struct ggml_tensor * v_proj;  // [H, Nkv*D] (NULL when QKV fused)
    struct ggml_tensor * o_proj;  // [Nh*D, H]
    struct ggml_tensor * q_norm;  // [D]
    struct ggml_tensor * k_norm;  // [D]

    // MLP (fused or separate)
    struct ggml_tensor * gate_up;    // [H, 2*FFN] fused (or NULL)
    struct ggml_tensor * gate_proj;  // [H, FFN] (NULL when fused)
    struct ggml_tensor * up_proj;    // [H, FFN] (NULL when fused)
    struct ggml_tensor * down_proj;  // [FFN, H]
};

// Standalone model (text encoder)
struct Qwen3GGML {
    Qwen3Config cfg;
    Qwen3Layer  layers[QWEN3_MAX_LAYERS];

    // Embedding: vocab lookup table [H, vocab_size] in ggml
    struct ggml_tensor * embed_tokens;  // [H, V]
    struct ggml_tensor * final_norm;    // [H]

    // Backend
    ggml_backend_t       backend;
    ggml_backend_t       cpu_backend;
    ggml_backend_sched_t sched;
    bool                 use_flash_attn;
    WeightCtx            wctx;
};

// Helpers (pure graph ops, no side effects)
static struct ggml_tensor * qwen3_f32(struct ggml_context * ctx, struct ggml_tensor * t) {
    if (t->type == GGML_TYPE_F32) {
        return t;
    }
    return ggml_cast(ctx, t, GGML_TYPE_F32);
}

static struct ggml_tensor * qwen3_linear(struct ggml_context * ctx, struct ggml_tensor * w, struct ggml_tensor * x) {
    return ggml_mul_mat(ctx, w, x);
}

static struct ggml_tensor * qwen3_linear_bias(struct ggml_context * ctx,
                                              struct ggml_tensor *  w,
                                              struct ggml_tensor *  b,
                                              struct ggml_tensor *  x) {
    struct ggml_tensor * out = ggml_mul_mat(ctx, w, x);
    return ggml_add(ctx, out, qwen3_f32(ctx, b));
}

// F32 manual attention (fallback when flash_attn_ext is disabled).
// Works for 3D [D, S, X] and 4D [D, S, X, N] inputs.
// Returns same layout as flash_attn_ext: dims 1 and 2 swapped vs input.
static struct ggml_tensor * qwen3_attn_f32(struct ggml_context * ctx,
                                           struct ggml_tensor *  q,
                                           struct ggml_tensor *  k,
                                           struct ggml_tensor *  v,
                                           struct ggml_tensor *  mask,
                                           float                 scale) {
    struct ggml_tensor * scores = ggml_mul_mat(ctx, k, q);
    scores                      = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f);
    struct ggml_tensor * vt     = ggml_cont(ctx, ggml_transpose(ctx, v));
    struct ggml_tensor * out    = ggml_mul_mat(ctx, vt, scores);
    return ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
}

static struct ggml_tensor * qwen3_rms_norm(struct ggml_context * ctx,
                                           struct ggml_tensor *  x,
                                           struct ggml_tensor *  w,
                                           float                 eps) {
    struct ggml_tensor * n = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, n, qwen3_f32(ctx, w));
}

// Graph builders
// These build sub-graphs and return output tensors.
// They operate on ggml layout: [H, S] for hidden states.

// Self-attention: norm_in [H, S] -> attn_out [H, S]
static struct ggml_tensor * qwen3_build_self_attn(struct ggml_context * ctx,
                                                  const Qwen3Config &   c,
                                                  Qwen3Layer *          ly,
                                                  struct ggml_tensor *  x,          // [H, S]
                                                  struct ggml_tensor *  positions,  // [S] int32
                                                  struct ggml_tensor *  mask,       // [S, S] or NULL
                                                  int                   S,
                                                  bool                  use_flash_attn = true) {
    int D   = c.head_dim;
    int Nh  = c.n_heads;
    int Nkv = c.n_kv_heads;

    // 1) Q/K/V projections (fused, partial, or separate)
    struct ggml_tensor *q, *k, *v;
    int                 q_dim  = Nh * D;
    int                 kv_dim = Nkv * D;
    if (ly->qkv) {
        struct ggml_tensor * qkv = qwen3_linear(ctx, ly->qkv, x);
        q                        = ggml_cont(ctx, ggml_view_2d(ctx, qkv, q_dim, S, qkv->nb[1], 0));
        k = ggml_cont(ctx, ggml_view_2d(ctx, qkv, kv_dim, S, qkv->nb[1], (size_t) q_dim * qkv->nb[0]));
        v = ggml_cont(ctx, ggml_view_2d(ctx, qkv, kv_dim, S, qkv->nb[1], (size_t) (q_dim + kv_dim) * qkv->nb[0]));
    } else if (ly->qk) {
        struct ggml_tensor * qk = qwen3_linear(ctx, ly->qk, x);
        q                       = ggml_cont(ctx, ggml_view_2d(ctx, qk, q_dim, S, qk->nb[1], 0));
        k = ggml_cont(ctx, ggml_view_2d(ctx, qk, kv_dim, S, qk->nb[1], (size_t) q_dim * qk->nb[0]));
        v = qwen3_linear(ctx, ly->v_proj, x);
    } else {
        q = qwen3_linear(ctx, ly->q_proj, x);
        k = qwen3_linear(ctx, ly->k_proj, x);
        v = qwen3_linear(ctx, ly->v_proj, x);
    }

    // 2) Reshape to heads: [X*D, S] -> [D, X, S]
    q = ggml_reshape_3d(ctx, q, D, Nh, S);
    k = ggml_reshape_3d(ctx, k, D, Nkv, S);
    v = ggml_reshape_3d(ctx, v, D, Nkv, S);

    // 3) QK-Norm: per-head RMSNorm on D
    q = ggml_rms_norm(ctx, q, c.rms_norm_eps);
    q = ggml_mul(ctx, q, qwen3_f32(ctx, ly->q_norm));
    k = ggml_rms_norm(ctx, k, c.rms_norm_eps);
    k = ggml_mul(ctx, k, qwen3_f32(ctx, ly->k_norm));

    // 4) RoPE
    // ggml pitfall: mode=2 (NEOX half-split [i, i+D/2]), NOT mode=0 (consecutive [2i, 2i+1])
    // Python ref: rope_batch_kernel pairs ptr[d] with ptr[d+half] = NEOX
    q = ggml_rope_ext(ctx, q, positions, NULL, D, 2, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, positions, NULL, D, 2, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // 5) Permute for flash_attn_ext: [D, X, S] -> [D, S, X]
    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    // 6) Attention (flash or F32 manual fallback)
    float scale = 1.0f / sqrtf((float) D);

    // K/V come in F32 from mul_mat (encoder, no KV cache). Cast to F16 before FA,
    // mirroring llama.cpp build_attn_mha for graphs without a KV cache.
    if (use_flash_attn) {
        if (k->type == GGML_TYPE_F32) {
            k = ggml_cast(ctx, k, GGML_TYPE_F16);
        }
        if (v->type == GGML_TYPE_F32) {
            v = ggml_cast(ctx, v, GGML_TYPE_F16);
        }
    }

    struct ggml_tensor * attn = use_flash_attn ? ggml_flash_attn_ext(ctx, q, k, v, mask, scale, 0.0f, 0.0f) :
                                                 qwen3_attn_f32(ctx, q, k, v, mask, scale);
    if (use_flash_attn) {
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    }

    // 7) Reshape back: [D, Nh, S] -> [Nh*D, S]
    attn = ggml_reshape_2d(ctx, attn, Nh * D, S);

    // 8) O projection
    return qwen3_linear(ctx, ly->o_proj, attn);
}

// MLP: SwiGLU (fused gate+up or separate)
static struct ggml_tensor * qwen3_build_mlp(struct ggml_context * ctx,
                                            Qwen3Layer *          ly,
                                            struct ggml_tensor *  x,  // [H, S]
                                            int                   S) {
    (void) S;
    struct ggml_tensor * ff;
    if (ly->gate_up) {
        struct ggml_tensor * gu = qwen3_linear(ctx, ly->gate_up, x);
        ff                      = ggml_swiglu(ctx, gu);
    } else {
        struct ggml_tensor * gate = qwen3_linear(ctx, ly->gate_proj, x);
        struct ggml_tensor * up   = qwen3_linear(ctx, ly->up_proj, x);
        ff                        = ggml_swiglu_split(ctx, gate, up);
    }
    return qwen3_linear(ctx, ly->down_proj, ff);
}

// Single layer: input [H, S] -> output [H, S]
static struct ggml_tensor * qwen3_build_layer(struct ggml_context * ctx,
                                              const Qwen3Config &   c,
                                              Qwen3Layer *          ly,
                                              struct ggml_tensor *  hidden,
                                              struct ggml_tensor *  positions,
                                              struct ggml_tensor *  mask,
                                              int                   S,
                                              bool                  use_flash_attn = true) {
    // Self-attention block
    struct ggml_tensor * norm = qwen3_rms_norm(ctx, hidden, ly->input_layernorm, c.rms_norm_eps);
    struct ggml_tensor * attn = qwen3_build_self_attn(ctx, c, ly, norm, positions, mask, S, use_flash_attn);
    hidden                    = ggml_add(ctx, hidden, attn);

    // MLP block
    norm                     = qwen3_rms_norm(ctx, hidden, ly->post_attn_layernorm, c.rms_norm_eps);
    struct ggml_tensor * mlp = qwen3_build_mlp(ctx, ly, norm, S);
    hidden                   = ggml_add(ctx, hidden, mlp);

    return hidden;
}

// Full N-layer stack: input [H, S] -> output [H, S] (post final-norm)
static struct ggml_tensor * qwen3_build_layers(struct ggml_context * ctx,
                                               const Qwen3Config &   c,
                                               Qwen3Layer *          layers,
                                               struct ggml_tensor *  final_norm_w,
                                               struct ggml_tensor *  hidden,
                                               struct ggml_tensor *  positions,
                                               struct ggml_tensor *  mask,
                                               int                   S,
                                               bool                  use_flash_attn = true) {
    for (int i = 0; i < c.n_layers; i++) {
        hidden = qwen3_build_layer(ctx, c, &layers[i], hidden, positions, mask, S, use_flash_attn);
    }
    return qwen3_rms_norm(ctx, hidden, final_norm_w, c.rms_norm_eps);
}

// Loading
static void qwen3_load_layer(WeightCtx *         wctx,
                             const GGUFModel &   gf,
                             Qwen3Layer *        ly,
                             const std::string & prefix,
                             int                 layer_idx = -1) {
    ly->input_layernorm     = gf_load_tensor_f32(wctx, gf, prefix + ".input_layernorm.weight");
    ly->post_attn_layernorm = gf_load_tensor_f32(wctx, gf, prefix + ".post_attention_layernorm.weight");

    // Attention: try Q+K+V fused, then Q+K partial, then separate
    ly->qkv = gf_load_qkv_fused(wctx, gf, prefix + ".self_attn.q_proj.weight", prefix + ".self_attn.k_proj.weight",
                                prefix + ".self_attn.v_proj.weight");
    if (!ly->qkv) {
        ly->qk = gf_load_pair_fused(wctx, gf, prefix + ".self_attn.q_proj.weight", prefix + ".self_attn.k_proj.weight");
        if (ly->qk) {
            ly->v_proj = gf_load_tensor(wctx, gf, prefix + ".self_attn.v_proj.weight");
            if (layer_idx == 0) {
                fprintf(stderr, "[Qwen3] Attn: Q+K fused, V separate\n");
            }
        } else {
            ly->q_proj = gf_load_tensor(wctx, gf, prefix + ".self_attn.q_proj.weight");
            ly->k_proj = gf_load_tensor(wctx, gf, prefix + ".self_attn.k_proj.weight");
            ly->v_proj = gf_load_tensor(wctx, gf, prefix + ".self_attn.v_proj.weight");
            if (layer_idx == 0) {
                fprintf(stderr, "[Qwen3] Attn: all separate\n");
            }
        }
    } else {
        if (layer_idx == 0) {
            fprintf(stderr, "[Qwen3] Attn: Q+K+V fused\n");
        }
    }

    ly->o_proj = gf_load_tensor(wctx, gf, prefix + ".self_attn.o_proj.weight");
    ly->q_norm = gf_load_tensor_f32(wctx, gf, prefix + ".self_attn.q_norm.weight");
    ly->k_norm = gf_load_tensor_f32(wctx, gf, prefix + ".self_attn.k_norm.weight");

    // MLP: try gate+up fused, then separate
    ly->gate_up = gf_load_pair_fused(wctx, gf, prefix + ".mlp.gate_proj.weight", prefix + ".mlp.up_proj.weight");
    if (ly->gate_up) {
        if (layer_idx == 0) {
            fprintf(stderr, "[Qwen3] MLP: gate+up fused\n");
        }
    } else {
        ly->gate_proj = gf_load_tensor(wctx, gf, prefix + ".mlp.gate_proj.weight");
        ly->up_proj   = gf_load_tensor(wctx, gf, prefix + ".mlp.up_proj.weight");
        if (layer_idx == 0) {
            fprintf(stderr, "[Qwen3] MLP: gate+up separate\n");
        }
    }
    ly->down_proj = gf_load_tensor(wctx, gf, prefix + ".mlp.down_proj.weight");
}

// Load standalone text encoder (Qwen3-Embedding) from GGUF
// gguf_path: path to the .gguf file
static bool qwen3_load_text_encoder(Qwen3GGML * m, const char * gguf_path) {
    // Backend init
    BackendPair bp    = backend_init("TextEncoder");
    m->backend        = bp.backend;
    m->cpu_backend    = bp.cpu_backend;
    m->sched          = backend_sched_new(bp, 4096);
    m->use_flash_attn = bp.has_gpu;

    m->cfg = {
        /*hidden_size*/ 1024,
        /*intermediate_size*/ 3072,
        /*n_heads*/ 16,
        /*n_kv_heads*/ 8,
        /*head_dim*/ 128,
        /*n_layers*/ 28,
        /*rope_theta*/ 1000000.0f,
        /*rms_norm_eps*/ 1e-6f,
        /*is_causal*/ true,
    };

    GGUFModel gf;
    if (!gf_load(&gf, gguf_path)) {
        fprintf(stderr, "[Load] FATAL: cannot load %s\n", gguf_path);
        return false;
    }

    // embed(1) + 28 layers * 11 weights + final_norm(1) = 310
    int n_tensors = 1 + m->cfg.n_layers * 11 + 1;
    wctx_init(&m->wctx, n_tensors);

    m->embed_tokens = gf_load_tensor(&m->wctx, gf, "embed_tokens.weight");
    m->final_norm   = gf_load_tensor_f32(&m->wctx, gf, "norm.weight");

    fprintf(stderr, "[Load] TextEncoder: %dL, H=%d, Nh=%d/%d\n", m->cfg.n_layers, m->cfg.hidden_size, m->cfg.n_heads,
            m->cfg.n_kv_heads);
    for (int i = 0; i < m->cfg.n_layers; i++) {
        char prefix[64];
        snprintf(prefix, sizeof(prefix), "layers.%d", i);
        qwen3_load_layer(&m->wctx, gf, &m->layers[i], prefix, i);
    }

    if (!wctx_alloc(&m->wctx, m->backend)) {
        gf_close(&gf);
        return false;
    }
    gf_close(&gf);

    return true;
}

// Forward: token IDs -> hidden states
// token_ids: [S] int32 (CPU)
// output:    [H * S] float (CPU, caller-allocated)
// Returns hidden states in ggml layout: ne[0]=H contiguous, S rows.
static void qwen3_forward(Qwen3GGML * m, const int * token_ids, int S, float * output) {
    const Qwen3Config & c = m->cfg;
    int                 H = c.hidden_size;

    // Graph context (generous fixed allocation)
    size_t                  ctx_size = 2048 * ggml_tensor_overhead() + ggml_graph_overhead();
    struct ggml_init_params gp       = { ctx_size, NULL, true };
    struct ggml_context *   ctx      = ggml_init(gp);

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);

    // Input: token IDs [S]
    struct ggml_tensor * t_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S);
    ggml_set_name(t_ids, "token_ids");
    ggml_set_input(t_ids);

    // Embedding lookup: [H, V] x [S] -> [H, S]
    struct ggml_tensor * hidden = ggml_get_rows(ctx, m->embed_tokens, t_ids);

    // Positions: [S] int32 (0, 1, 2, ...)
    struct ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // Attention mask: causal uses fp16 mask, bidirectional uses NULL (no mask)
    struct ggml_tensor * mask = NULL;
    if (c.is_causal) {
        mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, S, S);
        ggml_set_name(mask, "attn_mask");
        ggml_set_input(mask);
    }

    // N layers + final norm
    struct ggml_tensor * out =
        qwen3_build_layers(ctx, c, m->layers, m->final_norm, hidden, positions, mask, S, m->use_flash_attn);
    ggml_set_name(out, "output");
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    // Allocate
    if (!ggml_backend_sched_alloc_graph(m->sched, gf)) {
        fprintf(stderr, "[TextEncoder] FATAL: failed to allocate graph (%d tokens)\n", S);
        exit(1);
    }

    // Set inputs
    ggml_backend_tensor_set(t_ids, token_ids, 0, S * sizeof(int));

    {
        std::vector<int> pos_data(S);
        for (int i = 0; i < S; i++) {
            pos_data[i] = i;
        }
        ggml_backend_tensor_set(positions, pos_data.data(), 0, S * sizeof(int));
    }

    if (c.is_causal) {
        std::vector<uint16_t> mask_data(S * S);
        for (int i = 0; i < S; i++) {
            for (int j = 0; j < S; j++) {
                float v              = (j <= i) ? 0.0f : -INFINITY;
                mask_data[i * S + j] = ggml_fp32_to_fp16(v);
            }
        }
        ggml_backend_tensor_set(mask, mask_data.data(), 0, S * S * sizeof(uint16_t));
    }

    // Compute
    ggml_backend_sched_graph_compute(m->sched, gf);

    // Read output [H, S]
    ggml_backend_tensor_get(out, output, 0, H * S * sizeof(float));

    ggml_backend_sched_reset(m->sched);
    ggml_free(ctx);
}

// Embedding lookup via ggml graph (reuses text encoder weights + scheduler)
// token_ids: [S] int32
// output:    [H * S] float (ggml layout: H contiguous, S tokens)
static void qwen3_embed_lookup(Qwen3GGML * m, const int * token_ids, int S, float * output) {
    int H = m->cfg.hidden_size;

    size_t                  ctx_size = 16 * ggml_tensor_overhead() + ggml_graph_overhead();
    struct ggml_init_params gp       = { ctx_size, NULL, true };
    struct ggml_context *   ctx      = ggml_init(gp);
    struct ggml_cgraph *    gf       = ggml_new_graph(ctx);

    struct ggml_tensor * t_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S);
    ggml_set_name(t_ids, "token_ids");
    ggml_set_input(t_ids);

    struct ggml_tensor * out = ggml_get_rows(ctx, m->embed_tokens, t_ids);
    ggml_set_name(out, "embed_out");
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    if (!ggml_backend_sched_alloc_graph(m->sched, gf)) {
        fprintf(stderr, "[TextEncoder] FATAL: failed to allocate graph (embed lookup, %d tokens)\n", S);
        exit(1);
    }
    ggml_backend_tensor_set(t_ids, token_ids, 0, S * sizeof(int));
    ggml_backend_sched_graph_compute(m->sched, gf);
    ggml_backend_tensor_get(out, output, 0, (size_t) H * S * sizeof(float));

    ggml_backend_sched_reset(m->sched);
    ggml_free(ctx);
}

// Free
static void qwen3_free(Qwen3GGML * m) {
    if (m->sched) {
        ggml_backend_sched_free(m->sched);
    }
    backend_release(m->backend, m->cpu_backend);
    wctx_free(&m->wctx);
    *m = {};
}
