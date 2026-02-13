#pragma once
// dit.h: ACE-Step DiT (Diffusion Transformer) via ggml compute graph
// Coexists with dit.cuh (CUDA reference). Same weights, same safetensors.
//
// Architecture: 24-layer transformer with AdaLN, GQA self-attn + cross-attn, SwiGLU MLP.
// Flow matching: 8 Euler steps (turbo schedule).
//
// ggml ops used: rms_norm, mul_mat, rope_ext, flash_attn_ext, swiglu_split,
//                conv_transpose_1d, add, mul, scale, view, reshape, permute.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "safetensors.h"

#include "debug.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Config (mirrors dit.cuh DiTConfig)

struct DiTGGMLConfig {
    int hidden_size        = 2048;
    int intermediate_size  = 6144;
    int n_heads            = 16;
    int n_kv_heads         = 8;
    int head_dim           = 128;
    int n_layers           = 24;
    int in_channels        = 192;   // after context concat
    int out_channels       = 64;    // audio_acoustic_hidden_dim
    int patch_size         = 2;
    int sliding_window     = 128;
    float rope_theta       = 1000000.0f;
    float rms_norm_eps     = 1e-6f;
};

// Layer weights

struct DiTGGMLTembWeights {
    struct ggml_tensor * linear_1_w;    // [256, hidden]
    struct ggml_tensor * linear_1_b;    // [hidden]
    struct ggml_tensor * linear_2_w;    // [hidden, hidden]
    struct ggml_tensor * linear_2_b;    // [hidden]
    struct ggml_tensor * time_proj_w;   // [hidden, 6*hidden]
    struct ggml_tensor * time_proj_b;   // [6*hidden]
};

struct DiTGGMLLayer {
    // Self-attention
    struct ggml_tensor * self_attn_norm;    // [hidden]
    struct ggml_tensor * sa_q_proj;         // [hidden, n_heads*head_dim]
    struct ggml_tensor * sa_k_proj;         // [hidden, n_kv_heads*head_dim]
    struct ggml_tensor * sa_v_proj;         // [hidden, n_kv_heads*head_dim]
    struct ggml_tensor * sa_q_norm;         // [head_dim]
    struct ggml_tensor * sa_k_norm;         // [head_dim]
    struct ggml_tensor * sa_o_proj;         // [n_heads*head_dim, hidden]

    // Cross-attention
    struct ggml_tensor * cross_attn_norm;   // [hidden]
    struct ggml_tensor * ca_q_proj;         // [hidden, n_heads*head_dim]
    struct ggml_tensor * ca_k_proj;         // [hidden, n_kv_heads*head_dim]
    struct ggml_tensor * ca_v_proj;         // [hidden, n_kv_heads*head_dim]
    struct ggml_tensor * ca_q_norm;         // [head_dim]
    struct ggml_tensor * ca_k_norm;         // [head_dim]
    struct ggml_tensor * ca_o_proj;         // [n_heads*head_dim, hidden]

    // MLP
    struct ggml_tensor * mlp_norm;          // [hidden]
    struct ggml_tensor * gate_proj;         // [hidden, intermediate]
    struct ggml_tensor * up_proj;           // [hidden, intermediate]
    struct ggml_tensor * down_proj;         // [intermediate, hidden]

    // AdaLN scale-shift table: [6*hidden] (6 rows of [hidden])
    struct ggml_tensor * scale_shift_table; // [hidden, 6] in ggml layout

    int layer_type;  // 0=sliding, 1=full
};

// Full model

#define DIT_GGML_MAX_LAYERS 32

struct DiTGGML {
    DiTGGMLConfig cfg;

    // Timestep embeddings
    DiTGGMLTembWeights time_embed;
    DiTGGMLTembWeights time_embed_r;

    // proj_in: Conv1d(in_channels, hidden, kernel=2, stride=2)
    struct ggml_tensor * proj_in_w;     // [in_ch, hidden, patch_size] raw, or ggml conv1d format
    struct ggml_tensor * proj_in_b;     // [hidden]

    // condition_embedder: Linear(hidden, hidden)
    struct ggml_tensor * cond_emb_w;    // [hidden, hidden]
    struct ggml_tensor * cond_emb_b;    // [hidden]

    // Layers
    DiTGGMLLayer layers[DIT_GGML_MAX_LAYERS];

    // Output
    struct ggml_tensor * norm_out;          // [hidden]
    struct ggml_tensor * out_scale_shift;   // [hidden, 2] in ggml layout
    struct ggml_tensor * proj_out_w;        // conv_transpose_1d weight
    struct ggml_tensor * proj_out_b;        // [out_channels]

    // Backend
    ggml_backend_t backend;
    ggml_backend_t cpu_backend;
    ggml_backend_sched_t sched;

    // Weight storage
    SFWeightCtx wctx;
};

// Load timestep embedding weights

static void dit_ggml_load_temb(DiTGGMLTembWeights * w, SFWeightCtx * wctx,
                                const SafeTensors & st, const std::string & prefix) {
    w->linear_1_w  = sf_load_tensor(wctx, st, prefix + ".linear_1.weight");
    w->linear_1_b  = sf_load_tensor(wctx, st, prefix + ".linear_1.bias");
    w->linear_2_w  = sf_load_tensor(wctx, st, prefix + ".linear_2.weight");
    w->linear_2_b  = sf_load_tensor(wctx, st, prefix + ".linear_2.bias");
    w->time_proj_w = sf_load_tensor(wctx, st, prefix + ".time_proj.weight");
    w->time_proj_b = sf_load_tensor(wctx, st, prefix + ".time_proj.bias");
}

// Load full DiT model

static bool dit_ggml_load(DiTGGML * m, const char * model_path, DiTGGMLConfig cfg) {
    m->cfg = cfg;

    // Load safetensors
    SafeTensors st;
    if (!safe_load(st, model_path)) {
        fprintf(stderr, "[Load] FATAL: cannot load safetensors from %s\n", model_path);
        return false;
    }
    fprintf(stderr, "[Load] DiT safetensors: %zu tensors\n", st.tensors.size());

    // Count tensors: temb(6*2) + proj_in(2) + cond_emb(2) + layers(19*24) + output(4) = 474
    int n_tensors = 6 * 2 + 2 + 2 + 19 * cfg.n_layers + 4;
    sf_weight_ctx_init(&m->wctx, n_tensors);

    // Timestep embeddings
    dit_ggml_load_temb(&m->time_embed,   &m->wctx, st, "decoder.time_embed");
    dit_ggml_load_temb(&m->time_embed_r, &m->wctx, st, "decoder.time_embed_r");

    // proj_in: Conv1d weight [hidden, in_ch, patch_size]
    // In ggml conv_1d format: ne[0]=patch_size, ne[1]=in_ch, ne[2]=hidden
    m->proj_in_w = sf_load_tensor(&m->wctx, st, "decoder.proj_in.1.weight");
    m->proj_in_b = sf_load_tensor(&m->wctx, st, "decoder.proj_in.1.bias");

    // condition_embedder
    m->cond_emb_w = sf_load_tensor(&m->wctx, st, "decoder.condition_embedder.weight");
    m->cond_emb_b = sf_load_tensor(&m->wctx, st, "decoder.condition_embedder.bias");

    // Layers
    for (int i = 0; i < cfg.n_layers; i++) {
        char prefix[128];
        snprintf(prefix, sizeof(prefix), "decoder.layers.%d", i);
        std::string p(prefix);
        DiTGGMLLayer & ly = m->layers[i];

        // Self-attention
        ly.self_attn_norm = sf_load_tensor(&m->wctx, st, p + ".self_attn_norm.weight");
        ly.sa_q_proj      = sf_load_tensor(&m->wctx, st, p + ".self_attn.q_proj.weight");
        ly.sa_k_proj      = sf_load_tensor(&m->wctx, st, p + ".self_attn.k_proj.weight");
        ly.sa_v_proj      = sf_load_tensor(&m->wctx, st, p + ".self_attn.v_proj.weight");
        ly.sa_q_norm      = sf_load_tensor(&m->wctx, st, p + ".self_attn.q_norm.weight");
        ly.sa_k_norm      = sf_load_tensor(&m->wctx, st, p + ".self_attn.k_norm.weight");
        ly.sa_o_proj      = sf_load_tensor(&m->wctx, st, p + ".self_attn.o_proj.weight");

        // Cross-attention
        ly.cross_attn_norm = sf_load_tensor(&m->wctx, st, p + ".cross_attn_norm.weight");
        ly.ca_q_proj       = sf_load_tensor(&m->wctx, st, p + ".cross_attn.q_proj.weight");
        ly.ca_k_proj       = sf_load_tensor(&m->wctx, st, p + ".cross_attn.k_proj.weight");
        ly.ca_v_proj       = sf_load_tensor(&m->wctx, st, p + ".cross_attn.v_proj.weight");
        ly.ca_q_norm       = sf_load_tensor(&m->wctx, st, p + ".cross_attn.q_norm.weight");
        ly.ca_k_norm       = sf_load_tensor(&m->wctx, st, p + ".cross_attn.k_norm.weight");
        ly.ca_o_proj       = sf_load_tensor(&m->wctx, st, p + ".cross_attn.o_proj.weight");

        // MLP
        ly.mlp_norm  = sf_load_tensor(&m->wctx, st, p + ".mlp_norm.weight");
        ly.gate_proj = sf_load_tensor(&m->wctx, st, p + ".mlp.gate_proj.weight");
        ly.up_proj   = sf_load_tensor(&m->wctx, st, p + ".mlp.up_proj.weight");
        ly.down_proj = sf_load_tensor(&m->wctx, st, p + ".mlp.down_proj.weight");

        // AdaLN scale_shift_table [1, 6, hidden] in safetensors
        ly.scale_shift_table = sf_load_tensor(&m->wctx, st, p + ".scale_shift_table");

        ly.layer_type = (i % 2 == 0) ? 0 : 1;  // 0=sliding, 1=full
    }

    // Output
    m->norm_out        = sf_load_tensor(&m->wctx, st, "decoder.norm_out.weight");
    m->out_scale_shift = sf_load_tensor(&m->wctx, st, "decoder.scale_shift_table");
    m->proj_out_w      = sf_load_tensor(&m->wctx, st, "decoder.proj_out.1.weight");
    m->proj_out_b      = sf_load_tensor(&m->wctx, st, "decoder.proj_out.1.bias");

    // Allocate backend buffer and copy weights
    if (!sf_weight_ctx_alloc(&m->wctx, m->backend)) {
        return false;
    }

    fprintf(stderr, "[Load] DiT: %d layers, H=%d, Nh=%d/%d, D=%d\n",
            cfg.n_layers, cfg.hidden_size, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
    return true;
}

// Backend init

static void dit_ggml_init_backend(DiTGGML * m) {
    // Load all available backends (CUDA, Metal, Vulkan...)
    ggml_backend_load_all();

    // Pick the best available backend
    m->backend = ggml_backend_init_best();
    m->cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);

    fprintf(stderr, "[Load] DiT backend: %s\n", ggml_backend_name(m->backend));

    // Scheduler: prefer GPU, fallback CPU
    ggml_backend_t backends[2] = { m->backend, m->cpu_backend };
    int n_backends = (m->backend == m->cpu_backend) ? 1 : 2;
    m->sched = ggml_backend_sched_new(backends, NULL, n_backends,
                                       8192, false, true);
}

// Graph builder: single DiT layer (self-attention block)
// Incremental approach: build and validate one block at a time.
//
// ggml tensor layout reminder:
//   [S, H] in math = ne[0]=H, ne[1]=S in ggml
//   [Nh, S, D] in math = ne[0]=D, ne[1]=S, ne[2]=Nh in ggml

// Helper: ensure tensor is f32 (cast if bf16/f16)
static struct ggml_tensor * dit_ggml_f32(
        struct ggml_context * ctx,
        struct ggml_tensor * t) {
    if (t->type == GGML_TYPE_F32) return t;
    return ggml_cast(ctx, t, GGML_TYPE_F32);
}

// Helper: RMSNorm + weight multiply
static struct ggml_tensor * dit_ggml_rms_norm_weighted(
        struct ggml_context * ctx,
        struct ggml_tensor * x,          // [H, S]
        struct ggml_tensor * weight,     // [H]
        float eps) {
    struct ggml_tensor * norm = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, norm, dit_ggml_f32(ctx, weight));
}

// Helper: Linear layer (no bias)
// weight: [in, out] in ggml (= [out, in] in PyTorch)
// input:  [in, S]
// output: [out, S]
static struct ggml_tensor * dit_ggml_linear(
        struct ggml_context * ctx,
        struct ggml_tensor * weight,
        struct ggml_tensor * input) {
    return ggml_mul_mat(ctx, weight, input);
}

// Helper: Linear layer with bias
static struct ggml_tensor * dit_ggml_linear_bias(
        struct ggml_context * ctx,
        struct ggml_tensor * weight,
        struct ggml_tensor * bias,
        struct ggml_tensor * input) {
    struct ggml_tensor * out = ggml_mul_mat(ctx, weight, input);
    return ggml_add(ctx, out, dit_ggml_f32(ctx, bias));
}

// Helper: AdaLN modulate
// out = norm * (1 + scale) + shift
// norm: [H, S], scale: [H], shift: [H]
static struct ggml_tensor * dit_ggml_adaln(
        struct ggml_context * ctx,
        struct ggml_tensor * norm,
        struct ggml_tensor * scale,
        struct ggml_tensor * shift) {
    struct ggml_tensor * scaled = ggml_mul(ctx, norm, scale);   // norm * scale
    struct ggml_tensor * out    = ggml_add(ctx, norm, scaled);  // norm + norm*scale = norm*(1+scale)
    return ggml_add(ctx, out, shift);                           // + shift
}

// Helper: Gated residual
// out = residual + x * gate
// residual: [H, S], x: [H, S], gate: [H]
// NOTE: no sigmoid, gate is a raw scaling factor (matches Python and CUDA reference)
static struct ggml_tensor * dit_ggml_gated_add(
        struct ggml_context * ctx,
        struct ggml_tensor * residual,
        struct ggml_tensor * x,
        struct ggml_tensor * gate) {
    struct ggml_tensor * gated = ggml_mul(ctx, x, gate);  // broadcast [H] over [H,S]
    return ggml_add(ctx, residual, gated);
}

// Build timestep embedding subgraph
// t_scalar: [1] f32, returns temb [H] and *out_tproj [6H]
// suffix: "_t" or "_r" for naming intermediate tensors
static struct ggml_tensor * dit_ggml_build_temb(
        struct ggml_context * ctx,
        DiTGGMLTembWeights * w,
        struct ggml_tensor * t_scalar,
        struct ggml_tensor ** out_tproj,
        const char * suffix = "") {

    // scale timestep by 1000 (diffusion convention, matches CUDA)
    struct ggml_tensor * t_scaled = ggml_scale(ctx, t_scalar, 1000.0f);

    // sinusoidal embedding: [1] -> [256]
    struct ggml_tensor * sinusoidal = ggml_timestep_embedding(ctx, t_scaled, 256, 10000);
    {
        char name[64];
        snprintf(name, sizeof(name), "sinusoidal%s", suffix);
        ggml_set_name(sinusoidal, name);
        ggml_set_output(sinusoidal);
    }

    // linear1 + silu: [256] -> [H]
    struct ggml_tensor * h = dit_ggml_linear_bias(ctx, w->linear_1_w, w->linear_1_b, sinusoidal);
    {
        char name[64];
        snprintf(name, sizeof(name), "temb_lin1%s", suffix);
        ggml_set_name(h, name);
        ggml_set_output(h);
    }
    h = ggml_silu(ctx, h);

    // linear2: [H] -> [H]
    struct ggml_tensor * temb = dit_ggml_linear_bias(ctx, w->linear_2_w, w->linear_2_b, h);

    // silu + proj: [H] -> [6H]
    struct ggml_tensor * h2 = ggml_silu(ctx, temb);
    *out_tproj = dit_ggml_linear_bias(ctx, w->time_proj_w, w->time_proj_b, h2);

    return temb;  // [H] (used for output adaln)
}

// Build self-attention sub-graph for a single layer.
// norm_sa: [H, S] pre-normalized + AdaLN-modulated hidden state
// Returns: output [H, S] (self-attention output, NOT added to residual yet)
static struct ggml_tensor * dit_ggml_build_self_attn(
        struct ggml_context * ctx,
        DiTGGML * m,
        DiTGGMLLayer * ly,
        struct ggml_tensor * norm_sa,    // [H, S] pre-normalized + AdaLN-modulated
        struct ggml_tensor * positions,  // [S] int32 position indices for RoPE
        struct ggml_tensor * mask,       // [S, S] or NULL (sliding window mask)
        int S, int layer_idx = -1) {

    DiTGGMLConfig & c = m->cfg;
    int D  = c.head_dim;
    int Nh = c.n_heads;
    int Nkv = c.n_kv_heads;

    // 1) QKV projections
    // Q: [H, S] -> [Nh*D, S]
    struct ggml_tensor * q = dit_ggml_linear(ctx, ly->sa_q_proj, norm_sa);
    // K: [H, S] -> [Nkv*D, S]
    struct ggml_tensor * k = dit_ggml_linear(ctx, ly->sa_k_proj, norm_sa);
    // V: [H, S] -> [Nkv*D, S]
    struct ggml_tensor * v = dit_ggml_linear(ctx, ly->sa_v_proj, norm_sa);

    // 3) Reshape to heads: [Nh*D, S] -> [D, Nh, S]
    //    ggml_rope_ext expects ne[2] == positions.ne[0], so rope BEFORE permute.
    //    ggml_flash_attn_ext expects [D, S, Nh], so permute AFTER rope.
    q = ggml_reshape_3d(ctx, q, D, Nh, S);
    k = ggml_reshape_3d(ctx, k, D, Nkv, S);
    v = ggml_reshape_3d(ctx, v, D, Nkv, S);

    // 4) QK-Norm: per-head RMSNorm on D dimension
    //    [D, Nh, S] rms_norm operates on ne[0]=D
    q = ggml_rms_norm(ctx, q, c.rms_norm_eps);
    q = ggml_mul(ctx, q, dit_ggml_f32(ctx, ly->sa_q_norm));
    k = ggml_rms_norm(ctx, k, c.rms_norm_eps);
    k = ggml_mul(ctx, k, dit_ggml_f32(ctx, ly->sa_k_norm));

    // 5) RoPE (bidirectional, sequential positions)
    //    Input [D, Nh, S]: ne[2]=S must match positions ne[0]=S. 
    q = ggml_rope_ext(ctx, q, positions, NULL,
                       D, 2 /*mode=NEOX*/, 0 /*n_ctx_orig*/,
                       c.rope_theta, 1.0f /*freq_scale*/,
                       0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, positions, NULL,
                       D, 2, 0,
                       c.rope_theta, 1.0f,
                       0.0f, 1.0f, 0.0f, 0.0f);

    if (layer_idx == 0) {
        ggml_set_name(q, "layer0_q_after_rope");
        ggml_set_output(q);
        ggml_set_name(k, "layer0_k_after_rope");
        ggml_set_output(k);
    }

    // 6) Permute for flash_attn_ext: [D, Nh, S] -> [D, S, Nh]
    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    // 7) Flash Attention (handles GQA: Nh Q heads, Nkv KV heads)
    //    Q[D, S, Nh], K[D, S, Nkv], V[D, S, Nkv]
    float scale = 1.0f / sqrtf((float)D);
    struct ggml_tensor * attn = ggml_flash_attn_ext(ctx, q, k, v,
                                                     mask,
                                                     scale, 0.0f, 0.0f);

    // flash_attn_ext returns [D, Nh, S] (permuted output, see ggml.h)
    // Reshape directly: [D, Nh, S] -> [D*Nh, S] = [H, S]
    attn = ggml_reshape_2d(ctx, attn, Nh * D, S);

    if (layer_idx == 0) {
        ggml_set_name(attn, "layer0_attn_out");
        ggml_set_output(attn);
    }

    // 8) O projection: [Nh*D, S] -> [H, S]
    struct ggml_tensor * out = dit_ggml_linear(ctx, ly->sa_o_proj, attn);
    return out;
}

// Build MLP sub-graph: SwiGLU
// norm_ffn: [H, S] pre-normalized + AdaLN-modulated hidden state
// Returns: output [H, S]
static struct ggml_tensor * dit_ggml_build_mlp(
        struct ggml_context * ctx,
        DiTGGML * m,
        DiTGGMLLayer * ly,
        struct ggml_tensor * norm_ffn,
        int S) {

    // Gate + Up projections: [H, S] -> [I, S]
    struct ggml_tensor * gate = dit_ggml_linear(ctx, ly->gate_proj, norm_ffn);
    struct ggml_tensor * up   = dit_ggml_linear(ctx, ly->up_proj, norm_ffn);

    // SwiGLU: silu(gate) * up
    struct ggml_tensor * ff = ggml_swiglu_split(ctx, gate, up);

    // Down projection: [I, S] -> [H, S]
    return dit_ggml_linear(ctx, ly->down_proj, ff);
}

// Build cross-attention sub-graph for a single layer.
// norm_ca: [H, S] pre-normalized hidden state (Q source)
// enc:     [H, enc_S] condition-embedded encoder states (K/V source)
// Returns: output [H, S] (NOT added to residual yet)
static struct ggml_tensor * dit_ggml_build_cross_attn(
        struct ggml_context * ctx,
        DiTGGML * m,
        DiTGGMLLayer * ly,
        struct ggml_tensor * norm_ca,    // [H, S]
        struct ggml_tensor * enc,        // [H, enc_S]
        struct ggml_tensor * positions,  // unused, kept for consistency
        int S, int enc_S) {

    DiTGGMLConfig & c = m->cfg;
    int D   = c.head_dim;
    int Nh  = c.n_heads;
    int Nkv = c.n_kv_heads;

    (void)positions;  // cross-attn has no RoPE

    // Q from hidden, K/V from encoder
    struct ggml_tensor * q = dit_ggml_linear(ctx, ly->ca_q_proj, norm_ca);  // [Nh*D, S]
    struct ggml_tensor * k = dit_ggml_linear(ctx, ly->ca_k_proj, enc);      // [Nkv*D, enc_S]
    struct ggml_tensor * v = dit_ggml_linear(ctx, ly->ca_v_proj, enc);      // [Nkv*D, enc_S]

    // reshape to [D, seq, heads]
    q = ggml_reshape_3d(ctx, q, D, Nh, S);
    q = ggml_permute(ctx, q, 0, 2, 1, 3);   // [D, S, Nh]

    k = ggml_reshape_3d(ctx, k, D, Nkv, enc_S);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);   // [D, enc_S, Nkv]

    v = ggml_reshape_3d(ctx, v, D, Nkv, enc_S);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);   // [D, enc_S, Nkv]

    // QK-norm (per head)
    q = ggml_rms_norm(ctx, q, c.rms_norm_eps);
    q = ggml_mul(ctx, q, dit_ggml_f32(ctx, ly->ca_q_norm));
    k = ggml_rms_norm(ctx, k, c.rms_norm_eps);
    k = ggml_mul(ctx, k, dit_ggml_f32(ctx, ly->ca_k_norm));

    // no RoPE for cross-attention
    // no mask (attend to all encoder positions)
    float scale = 1.0f / sqrtf((float)D);
    struct ggml_tensor * attn = ggml_flash_attn_ext(ctx, q, k, v, NULL, scale, 0.0f, 0.0f);

    // flash_attn_ext returns [D, Nh, S] (permuted), reshape directly to [H, S]
    attn = ggml_reshape_2d(ctx, attn, Nh * D, S);

    // O projection
    return dit_ggml_linear(ctx, ly->ca_o_proj, attn);
}

// Build one full DiT layer (AdaLN + self-attn + cross-attn + FFN + gated residuals)
// hidden: [H, S], tproj: [6H] (combined timestep projection)
// enc: [H, enc_S] (condition-embedded encoder states, or NULL to skip cross-attn)
// sw_mask: [S, S] sliding window mask (or NULL for full attention)
// Returns: updated hidden [H, S]
static struct ggml_tensor * dit_ggml_build_layer(
        struct ggml_context * ctx,
        DiTGGML * m,
        int layer_idx,
        struct ggml_tensor * hidden,     // [H, S]
        struct ggml_tensor * tproj,      // [6H] f32 combined temb projection
        struct ggml_tensor * enc,        // [H, enc_S] or NULL
        struct ggml_tensor * positions,  // [S] int32
        struct ggml_tensor * sw_mask,    // [S, S] or NULL
        int S, int enc_S) {

    DiTGGMLConfig & c = m->cfg;
    DiTGGMLLayer * ly = &m->layers[layer_idx];
    int H = c.hidden_size;

    // AdaLN: scale_shift_table [6, H] + tproj [6H] -> 6 vectors of [H]
    // scale_shift_table is stored as bf16, cast to f32 for arithmetic
    struct ggml_tensor * ss = ly->scale_shift_table;
    if (ss->type != GGML_TYPE_F32) {
        ss = ggml_cast(ctx, ss, GGML_TYPE_F32);
    }
    // flatten [H, 6] -> [6H] (ggml ne[0]=H, ne[1]=6, contiguous = 6H floats)
    struct ggml_tensor * ss_flat = ggml_reshape_1d(ctx, ss, 6 * H);
    struct ggml_tensor * adaln = ggml_add(ctx, ss_flat, tproj);  // [6H] f32

    // extract 6 modulation vectors [H] each
    size_t Hb = H * sizeof(float);
    struct ggml_tensor * shift_sa  = ggml_view_1d(ctx, adaln, H, 0 * Hb);
    struct ggml_tensor * scale_sa  = ggml_view_1d(ctx, adaln, H, 1 * Hb);
    struct ggml_tensor * gate_sa   = ggml_view_1d(ctx, adaln, H, 2 * Hb);
    struct ggml_tensor * shift_ffn = ggml_view_1d(ctx, adaln, H, 3 * Hb);
    struct ggml_tensor * scale_ffn = ggml_view_1d(ctx, adaln, H, 4 * Hb);
    struct ggml_tensor * gate_ffn  = ggml_view_1d(ctx, adaln, H, 5 * Hb);

    // Self-attention with AdaLN + gated residual
    struct ggml_tensor * residual = hidden;
    struct ggml_tensor * norm_sa = dit_ggml_rms_norm_weighted(ctx, hidden, ly->self_attn_norm, c.rms_norm_eps);
    norm_sa = dit_ggml_adaln(ctx, norm_sa, scale_sa, shift_sa);

    if (layer_idx == 0) {
        ggml_set_name(norm_sa, "layer0_sa_input");
        ggml_set_output(norm_sa);
    }

    // select mask: even layers use sliding window, odd layers use full attention
    struct ggml_tensor * mask = (ly->layer_type == 0) ? sw_mask : NULL;
    struct ggml_tensor * sa_out = dit_ggml_build_self_attn(ctx, m, ly, norm_sa, positions, mask, S, layer_idx);

    if (layer_idx == 0) {
        ggml_set_name(sa_out, "layer0_sa_output");
        ggml_set_output(sa_out);
    }

    hidden = dit_ggml_gated_add(ctx, residual, sa_out, gate_sa);

    if (layer_idx == 0) {
        ggml_set_name(hidden, "layer0_after_self_attn");
        ggml_set_output(hidden);
    }

    // Cross-attention (no gate, simple residual add)
    if (enc) {
        struct ggml_tensor * norm_ca = dit_ggml_rms_norm_weighted(ctx, hidden, ly->cross_attn_norm, c.rms_norm_eps);
        struct ggml_tensor * ca_out = dit_ggml_build_cross_attn(ctx, m, ly, norm_ca, enc, positions, S, enc_S);
        hidden = ggml_add(ctx, hidden, ca_out);
    }

    if (layer_idx == 0) {
        ggml_set_name(hidden, "layer0_after_cross_attn");
        ggml_set_output(hidden);
    }

    // FFN with AdaLN + gated residual
    residual = hidden;
    struct ggml_tensor * norm_ffn = dit_ggml_rms_norm_weighted(ctx, hidden, ly->mlp_norm, c.rms_norm_eps);
    norm_ffn = dit_ggml_adaln(ctx, norm_ffn, scale_ffn, shift_ffn);
    struct ggml_tensor * ffn_out = dit_ggml_build_mlp(ctx, m, ly, norm_ffn, S);
    hidden = dit_ggml_gated_add(ctx, residual, ffn_out, gate_ffn);

    return hidden;
}

// Build the full DiT forward graph (all layers).
// Returns the final output tensor [out_channels, T] (velocity prediction).
//
// Graph inputs (set via ggml_backend_tensor_set after alloc):
//   "input_latents"   [in_channels, T]  concat(context_latents, xt), set by caller
//   "enc_hidden"      [H, enc_S]        text encoder hidden states
//   "t"               [1] f32           flow matching timestep
//   "t_r"             [1] f32           reference timestep (= t_curr per step)
//   "positions"       [S] i32           position indices 0..S-1
//   "sw_mask"         [S, S] f32        sliding window mask (if S > window)
//
// Graph outputs:
//   "velocity"        [out_channels, T]  predicted flow velocity
static struct ggml_cgraph * dit_ggml_build_graph(
        DiTGGML * m,
        struct ggml_context * ctx,
        int T,                  // temporal length (before patching)
        int enc_S,              // encoder sequence length
        struct ggml_tensor ** p_input,      // [out] input tensor to fill
        struct ggml_tensor ** p_output,     // [out] output tensor to read
        struct ggml_tensor ** p_tproj = nullptr,  // [out] tproj tensor (for injection)
        bool inject_tproj = false) {        // use external tproj input

    DiTGGMLConfig & c = m->cfg;
    int S = T / c.patch_size;   // sequence length after patching
    int H = c.hidden_size;
    int P = c.patch_size;

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);

    // Inputs

    // Concatenated latent: [in_channels, T] = concat(context_128ch, xt)
    struct ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.in_channels, T);
    ggml_set_name(input, "input_latents");
    ggml_set_input(input);
    *p_input = input;

    // Encoder hidden states: [H, enc_S]
    struct ggml_tensor * enc_hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, enc_S);
    ggml_set_name(enc_hidden, "enc_hidden");
    ggml_set_input(enc_hidden);

    // Timesteps: scalars
    struct ggml_tensor * t_val = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_name(t_val, "t");
    ggml_set_input(t_val);

    struct ggml_tensor * tr_val = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_name(tr_val, "t_r");
    ggml_set_input(tr_val);

    // Position indices for RoPE: [0, 1, ..., S-1]
    struct ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // ggml pitfall: flash_attn_ext reads mask as fp16! Sliding window mask: [S, S] fp16
    struct ggml_tensor * sw_mask = NULL;
    if (c.sliding_window > 0 && S > c.sliding_window) {
        sw_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, S, S);
        ggml_set_name(sw_mask, "sw_mask");
        ggml_set_input(sw_mask);
    }

    // 1) Timestep embeddings
    struct ggml_tensor * tproj;
    struct ggml_tensor * temb;

    if (inject_tproj) {
        // External tproj input (from CUDA dump)
        tproj = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 6 * H);
        ggml_set_name(tproj, "tproj");
        ggml_set_input(tproj);

        // Still need temb for the final norm. Use a zero placeholder
        // (temb only affects final_norm which we can compare separately)
        temb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        ggml_set_name(temb, "temb");
        ggml_set_input(temb);
    } else {
        struct ggml_tensor * tproj_t;
        struct ggml_tensor * temb_t = dit_ggml_build_temb(ctx, &m->time_embed, t_val, &tproj_t, "_t");
        ggml_set_name(temb_t, "temb_t");
        ggml_set_output(temb_t);

        struct ggml_tensor * tproj_r;
        // CUDA passes (t - t_r) to time_embed_r, not t_r directly
        // In turbo mode t = t_r, so input is 0
        struct ggml_tensor * t_diff = ggml_sub(ctx, t_val, tr_val);
        struct ggml_tensor * temb_r = dit_ggml_build_temb(ctx, &m->time_embed_r, t_diff, &tproj_r, "_r");
        ggml_set_name(temb_r, "temb_r");
        ggml_set_output(temb_r);

        // combine: temb = temb_t + temb_r [H], tproj = tproj_t + tproj_r [6H]
        temb  = ggml_add(ctx, temb_t, temb_r);
        ggml_set_name(temb, "temb");
        ggml_set_output(temb);
        tproj = ggml_add(ctx, tproj_t, tproj_r);
        ggml_set_name(tproj, "tproj");
        ggml_set_output(tproj);
    }
    if (p_tproj) *p_tproj = tproj;

    // 2) proj_in: patchify + linear
    struct ggml_tensor * patched = ggml_reshape_2d(ctx, input, c.in_channels * P, S);

    // Weight from safetensors [H, in_ch, P] -> ggml [P, in_ch, H]
    // Patched data is p-major: [frame0_all_ch, frame1_all_ch, ...]
    // Must permute weight to match: [P, in_ch, H] -> [in_ch, P, H] -> flatten -> [in_ch*P, H]
    struct ggml_tensor * proj_w_perm = ggml_permute(ctx, m->proj_in_w, 1, 0, 2, 3);
    struct ggml_tensor * proj_w_cont = ggml_cont(ctx, proj_w_perm);
    struct ggml_tensor * proj_w_2d = ggml_reshape_2d(ctx, proj_w_cont,
                                                      c.in_channels * P, H);
    struct ggml_tensor * hidden = dit_ggml_linear_bias(ctx, proj_w_2d, m->proj_in_b, patched);
    ggml_set_name(hidden, "hidden_after_proj_in");
    ggml_set_output(hidden);

    // 3) Condition embedder: project encoder hidden states
    struct ggml_tensor * enc = dit_ggml_linear_bias(ctx, m->cond_emb_w, m->cond_emb_b, enc_hidden);
    ggml_set_name(enc, "enc_after_cond_emb");
    ggml_set_output(enc);

    // 4) Transformer layers
    for (int i = 0; i < c.n_layers; i++) {
        hidden = dit_ggml_build_layer(ctx, m, i, hidden, tproj, enc, positions, sw_mask, S, enc_S);
        if (i == 0) {
            ggml_set_name(hidden, "hidden_after_layer0");
            ggml_set_output(hidden);
        }
    }

    // 5) Output: AdaLN + proj_out
    // out_scale_shift: [H, 2] -> cast to f32 if bf16, flatten to [2H]
    struct ggml_tensor * oss = m->out_scale_shift;
    if (oss->type != GGML_TYPE_F32) {
        oss = ggml_cast(ctx, oss, GGML_TYPE_F32);
    }
    struct ggml_tensor * oss_flat = ggml_reshape_1d(ctx, oss, 2 * H);

    size_t Hb = H * sizeof(float);
    struct ggml_tensor * out_shift = ggml_view_1d(ctx, oss_flat, H, 0);
    struct ggml_tensor * out_scale = ggml_view_1d(ctx, oss_flat, H, Hb);
    out_shift = ggml_add(ctx, out_shift, temb);
    out_scale = ggml_add(ctx, out_scale, temb);

    struct ggml_tensor * norm_out = dit_ggml_rms_norm_weighted(ctx, hidden, m->norm_out, c.rms_norm_eps);
    norm_out = dit_ggml_adaln(ctx, norm_out, out_scale, out_shift);

    // proj_out: ConvTranspose1d(hidden, out_ch, k=P, s=P)
    // Weight from safetensors [H, out_ch, P] -> ggml [P, out_ch, H]
    // Permute to [out_ch, P, H] -> flatten -> [out_ch*P, H] -> transpose -> [H, out_ch*P]
    // This gives p-major output matching reshape_2d(output, out_ch, T) unpatchify
    struct ggml_tensor * proj_out_perm = ggml_permute(ctx, m->proj_out_w, 1, 0, 2, 3);
    struct ggml_tensor * proj_out_cont = ggml_cont(ctx, proj_out_perm);
    struct ggml_tensor * proj_out_w_2d = ggml_reshape_2d(ctx, proj_out_cont,
                                                          c.out_channels * P, H);
    proj_out_w_2d = ggml_cont(ctx, ggml_transpose(ctx, proj_out_w_2d));
    struct ggml_tensor * output = dit_ggml_linear_bias(ctx, proj_out_w_2d, m->proj_out_b, norm_out);
    output = ggml_reshape_2d(ctx, output, c.out_channels, T);

    ggml_set_name(output, "velocity");
    ggml_set_output(output);
    *p_output = output;

    ggml_build_forward_expand(gf, output);

    return gf;
}

// Flow matching generation loop
// Runs N euler steps to denoise latents.
//
// noise:            [out_ch, T]  initial gaussian noise (or zeros for test)
// context_latents:  [ctx_ch, T]  context (128 channels), constant across steps
// enc_hidden:       [H, enc_S]   encoder output (text conditioning)
// schedule:         array of N timestep values (e.g. {1.0, 0.875, ..., 0.125})
// output:           [out_ch, T]  generated latent (caller-allocated)
static void dit_ggml_generate(
        DiTGGML * model,
        const float * noise,
        const float * context_latents,
        const float * enc_hidden_data,
        int enc_S,
        int T,
        int num_steps,
        const float * schedule,
        float * output,
        const DebugDumper * dbg = nullptr,
        const float * inject_tproj_data = nullptr,  // [6*H] f32 from CUDA dump
        const float * inject_temb_data  = nullptr) { // [H] f32 from CUDA dump

    DiTGGMLConfig & c = model->cfg;
    int Oc    = c.out_channels;      // 64
    int ctx_ch = c.in_channels - Oc; // 128
    int in_ch = c.in_channels;       // 192
    int S     = T / c.patch_size;
    int n     = T * Oc;              // output element count
    int H     = c.hidden_size;

    bool do_inject_tproj = (inject_tproj_data != nullptr);
    if (do_inject_tproj) {
        fprintf(stderr, "[DiT] tproj injection ENABLED (bypassing timestep MLP)\n");
    }

    // Build graph once (shapes are constant across steps)
    size_t ctx_size = ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false);
    std::vector<uint8_t> ctx_buf(ctx_size);

    struct ggml_init_params gparams = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ ctx_buf.data(),
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(gparams);

    struct ggml_tensor * t_input  = NULL;
    struct ggml_tensor * t_output = NULL;
    struct ggml_tensor * t_tproj  = NULL;
    struct ggml_cgraph * gf = dit_ggml_build_graph(model, ctx, T, enc_S,
                                                    &t_input, &t_output,
                                                    do_inject_tproj ? &t_tproj : nullptr,
                                                    do_inject_tproj);

    fprintf(stderr, "[DiT] Graph: %d nodes\n", ggml_graph_n_nodes(gf));

    // Allocate compute buffers
    ggml_backend_sched_reset(model->sched);
    if (!ggml_backend_sched_alloc_graph(model->sched, gf)) {
        fprintf(stderr, "FATAL: failed to allocate graph\n");
        ggml_free(ctx);
        return;
    }

    // Set static inputs (constant across all steps)

    // Encoder hidden states: [H, enc_S]
    struct ggml_tensor * t_enc = ggml_graph_get_tensor(gf, "enc_hidden");
    ggml_backend_tensor_set(t_enc, enc_hidden_data, 0, c.hidden_size * enc_S * sizeof(float));

    // t_r is set per-step in the loop (= t_curr, same as CUDA reference)
    struct ggml_tensor * t_tr = do_inject_tproj ? nullptr : ggml_graph_get_tensor(gf, "t_r");

    // Positions: [0, 1, ..., S-1]
    struct ggml_tensor * t_pos = ggml_graph_get_tensor(gf, "positions");
    {
        std::vector<int32_t> pos(S);
        for (int i = 0; i < S; i++) pos[i] = i;
        ggml_backend_tensor_set(t_pos, pos.data(), 0, S * sizeof(int32_t));
    }

    // Sliding window mask: [S, S] fp16
    struct ggml_tensor * t_mask = ggml_graph_get_tensor(gf, "sw_mask");
    if (t_mask) {
        int win = c.sliding_window;
        std::vector<uint16_t> mask_data(S * S);
        for (int qi = 0; qi < S; qi++)
            for (int ki = 0; ki < S; ki++) {
                int dist = (qi > ki) ? (qi - ki) : (ki - qi);
                float v = (dist <= win) ? 0.0f : -INFINITY;
                mask_data[ki * S + qi] = ggml_fp32_to_fp16(v);
            }
        ggml_backend_tensor_set(t_mask, mask_data.data(), 0, S * S * sizeof(uint16_t));
    }

    // Inject tproj/temb from CUDA dump (if provided)
    if (do_inject_tproj && t_tproj) {
        struct ggml_tensor * t_tp = ggml_graph_get_tensor(gf, "tproj");
        if (t_tp) {
            ggml_backend_tensor_set(t_tp, inject_tproj_data, 0, 6 * H * sizeof(float));
            fprintf(stderr, "[inject] tproj: %d floats set\n", 6 * H);
        }
        struct ggml_tensor * t_te = ggml_graph_get_tensor(gf, "temb");
        if (t_te && inject_temb_data) {
            ggml_backend_tensor_set(t_te, inject_temb_data, 0, H * sizeof(float));
            fprintf(stderr, "[inject] temb: %d floats set\n", H);
        } else if (t_te) {
            // Zero out temb if not provided (it's only used in final norm)
            std::vector<float> zeros(H, 0.0f);
            ggml_backend_tensor_set(t_te, zeros.data(), 0, H * sizeof(float));
            fprintf(stderr, "[inject] temb: zeroed (no data provided)\n");
        }
    }

    // Prepare host buffers
    // xt: evolves across steps (starts as noise)
    std::vector<float> xt(noise, noise + n);
    std::vector<float> vt(n);

    // input_buf: concat(context_latents, xt) per time frame
    // ggml pitfall: tensor [ne0, ne1] has ne0 contiguous in memory.
    //   [in_ch, T] => ne0=in_ch, ne1=T => element (c, t) = data[t * in_ch + c]
    // channel concat on dim 0: input(c, t) = context(c, t) if c < ctx_ch
    //                           input(c, t) = xt(c - ctx_ch, t) if c >= ctx_ch
    std::vector<float> input_buf(in_ch * T);

    struct ggml_tensor * t_t = do_inject_tproj ? nullptr : ggml_graph_get_tensor(gf, "t");

    // Flow matching loop
    for (int step = 0; step < num_steps; step++) {
        float t_curr = schedule[step];

        // set timestep
        if (t_t) {
            ggml_backend_tensor_set(t_t, &t_curr, 0, sizeof(float));
        }
        if (t_tr) {
            ggml_backend_tensor_set(t_tr, &t_curr, 0, sizeof(float));
        }

        // build concat input: [in_ch, T]
        for (int t = 0; t < T; t++) {
            // context channels first
            memcpy(&input_buf[t * in_ch],
                   &context_latents[t * ctx_ch],
                   ctx_ch * sizeof(float));
            // then xt channels
            memcpy(&input_buf[t * in_ch + ctx_ch],
                   &xt[t * Oc],
                   Oc * sizeof(float));
        }
        ggml_backend_tensor_set(t_input, input_buf.data(), 0, in_ch * T * sizeof(float));

        // compute forward pass
        ggml_backend_sched_graph_compute(model->sched, gf);

        // dump intermediate tensors on step 0
        if (step == 0 && dbg && dbg->enabled) {
            auto dump_named = [&](const char *name) {
                struct ggml_tensor * t = ggml_graph_get_tensor(gf, name);
                if (t) {
                    std::vector<float> buf(ggml_nelements(t));
                    ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
                    if (t->ne[1] <= 1) {
                        debug_dump_1d(dbg, name, buf.data(), (int)t->ne[0]);
                    } else if (ggml_n_dims(t) <= 2) {
                        debug_dump_2d(dbg, name, buf.data(), (int)t->ne[0], (int)t->ne[1]);
                    } else {
                        // 3D+: flatten higher dims into ne[1]
                        int64_t flat = ggml_nelements(t) / t->ne[0];
                        debug_dump_2d(dbg, name, buf.data(), (int)t->ne[0], (int)flat);
                    }
                }
            };
            dump_named("tproj");
            dump_named("temb");
            dump_named("temb_t");
            dump_named("temb_r");
            dump_named("sinusoidal_t");
            dump_named("sinusoidal_r");
            dump_named("temb_lin1_t");
            dump_named("temb_lin1_r");
            dump_named("hidden_after_proj_in");
            dump_named("enc_after_cond_emb");
            dump_named("layer0_sa_input");
            dump_named("layer0_q_after_rope");
            dump_named("layer0_k_after_rope");
            dump_named("layer0_sa_output");
            dump_named("layer0_attn_out");
            dump_named("layer0_after_self_attn");
            dump_named("layer0_after_cross_attn");
            dump_named("hidden_after_layer0");
        }

        // read velocity output: [Oc, T]
        ggml_backend_tensor_get(t_output, vt.data(), 0, n * sizeof(float));

        // debug dump vt
        if (dbg && dbg->enabled) {
            char name[64];
            snprintf(name, sizeof(name), "dit_step%d_vt", step);
            debug_dump_2d(dbg, name, vt.data(), Oc, T);
        }

        // euler step
        if (step == num_steps - 1) {
            // final: x0 = zt - vt * t
            for (int i = 0; i < n; i++)
                output[i] = xt[i] - vt[i] * t_curr;
        } else {
            // intermediate: xt -= vt * dt
            float dt = t_curr - schedule[step + 1];
            for (int i = 0; i < n; i++)
                xt[i] -= vt[i] * dt;
        }

        // debug dump xt/x0 after step
        if (dbg && dbg->enabled) {
            char name[64];
            if (step == num_steps - 1) {
                snprintf(name, sizeof(name), "dit_x0");
                debug_dump_2d(dbg, name, output, Oc, T);
            } else {
                snprintf(name, sizeof(name), "dit_step%d_xt", step);
                debug_dump_2d(dbg, name, xt.data(), Oc, T);
            }
        }

        fprintf(stderr, "[DiT] step %d/%d t=%.3f\n", step + 1, num_steps, t_curr);
    }

    ggml_free(ctx);
}

// Free

static void dit_ggml_free(DiTGGML * m) {
    if (m->sched) ggml_backend_sched_free(m->sched);
    if (m->backend && m->backend != m->cpu_backend) ggml_backend_free(m->backend);
    if (m->cpu_backend) ggml_backend_free(m->cpu_backend);
    sf_weight_ctx_free(&m->wctx);
    memset(m, 0, sizeof(*m));
}
