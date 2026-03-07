#pragma once
// dit-graph.h: DiT compute graph construction (ggml)
//
// Graph builders: timestep embedding, self-attention, cross-attention,
// MLP, per-layer composition, and full N-step diffusion graph.
//
// ggml tensor layout reminder:
//   [S, H] in math = ne[0]=H, ne[1]=S in ggml
//   [Nh, S, D] in math = ne[0]=D, ne[1]=S, ne[2]=Nh in ggml

#include "dit.h"

#include <cmath>

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
        struct ggml_tensor * x,         // [H, S]
        struct ggml_tensor * weight,    // [H]
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

// Linear with optional LoRA: out = W@x + scale * (B@(A@x)). lora_a/lora_b may be NULL.
static struct ggml_tensor * dit_ggml_linear_lora(
        struct ggml_context * ctx,
        struct ggml_tensor * weight,
        struct ggml_tensor * lora_a,   // [in, r]
        struct ggml_tensor * lora_b,   // [r, out]
        float lora_scale,
        struct ggml_tensor * input) {
    struct ggml_tensor * out = ggml_mul_mat(ctx, weight, input);
    if (lora_a && lora_b && lora_scale != 0.0f) {
        struct ggml_tensor * ax = ggml_mul_mat(ctx, lora_a, input);
        struct ggml_tensor * bax = ggml_mul_mat(ctx, lora_b, ax);
        out = ggml_add(ctx, out, ggml_scale(ctx, bax, lora_scale));
    }
    return out;
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
        struct ggml_tensor * shift,
        struct ggml_tensor * one) {
    // norm * (1 + scale) + shift
    // one is [1] = 1.0, broadcasts to [H]; avoids expensive [H,S,N] add
    struct ggml_tensor * one_plus_s = ggml_add(ctx, scale, one);   // [H] + [1] -> [H]
    struct ggml_tensor * scaled = ggml_mul(ctx, norm, one_plus_s); // [H,S,N]
    return ggml_add(ctx, scaled, shift);                           // [H,S,N]
}

// Helper: Gated residual
// out = residual + x * gate
// residual: [H, S], x: [H, S], gate: [H]
// NOTE: no sigmoid, gate is a raw scaling factor (matches Python reference)
static struct ggml_tensor * dit_ggml_gated_add(
        struct ggml_context * ctx,
        struct ggml_tensor * residual,
        struct ggml_tensor * x,
        struct ggml_tensor * gate) {
    struct ggml_tensor * gated = ggml_mul(ctx, x, gate); // broadcast [H] over [H,S]
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

    // scale timestep by 1000 (diffusion convention, matches Python)
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

// F32 manual attention (fallback when flash_attn_ext is not available or imprecise).
// Q: [D, S, Nh], K: [D, S_kv, Nkv], V: [D, S_kv, Nkv]
// mask: [S_kv, S] F16 or NULL, scale: 1/sqrt(D)
// Returns: [D, Nh, S] (same layout as flash_attn_ext output)
static struct ggml_tensor * dit_attn_f32(
        struct ggml_context * ctx,
        struct ggml_tensor * q,
        struct ggml_tensor * k,
        struct ggml_tensor * v,
        struct ggml_tensor * mask,
        float scale) {
    struct ggml_tensor * scores = ggml_mul_mat(ctx, k, q);
    scores = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f);
    struct ggml_tensor * vt = ggml_cont(ctx, ggml_transpose(ctx, v));
    struct ggml_tensor * out = ggml_mul_mat(ctx, vt, scores);
    return ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
}

// Build self-attention sub-graph for a single layer.
// norm_sa: [H, S, N] pre-normalized + AdaLN-modulated hidden state
// Returns: output [H, S, N] (self-attention output, NOT added to residual yet)
static struct ggml_tensor * dit_ggml_build_self_attn(
        struct ggml_context * ctx,
        DiTGGML * m,
        DiTGGMLLayer * ly,
        struct ggml_tensor * norm_sa,   // [H, S, N] pre-normalized + AdaLN-modulated
        struct ggml_tensor * positions, // [S*N] int32 position indices for RoPE
        struct ggml_tensor * mask,      // [S, S] or NULL (sliding window mask)
        int S, int N, int layer_idx = -1) {

    DiTGGMLConfig & c = m->cfg;
    int D  = c.head_dim;
    int Nh = c.n_heads;
    int Nkv = c.n_kv_heads;

    // 1) QKV projections (full fused, QK partial, separate)
    struct ggml_tensor * q, * k, * v;
    int q_dim  = Nh * D;
    int kv_dim = Nkv * D;
    float lora_scale = m->lora_scale;
    if (ly->sa_qkv) {
        struct ggml_tensor * qkv = dit_ggml_linear(ctx, ly->sa_qkv, norm_sa);
        q = ggml_cont(ctx, ggml_view_3d(ctx, qkv, q_dim, S, N, qkv->nb[1], qkv->nb[2], 0));
        k = ggml_cont(ctx, ggml_view_3d(ctx, qkv, kv_dim, S, N, qkv->nb[1], qkv->nb[2], (size_t)q_dim * qkv->nb[0]));
        v = ggml_cont(ctx, ggml_view_3d(ctx, qkv, kv_dim, S, N, qkv->nb[1], qkv->nb[2], (size_t)(q_dim + kv_dim) * qkv->nb[0]));
        // LoRA on fused path: add scale * (B @ (A @ x)) per projection when adapters are loaded
        if (lora_scale != 0.0f) {
            if (ly->lora_sa_q_a && ly->lora_sa_q_b)
                q = ggml_add(ctx, q, ggml_scale(ctx, ggml_mul_mat(ctx, ly->lora_sa_q_b, ggml_mul_mat(ctx, ly->lora_sa_q_a, norm_sa)), lora_scale));
            if (ly->lora_sa_k_a && ly->lora_sa_k_b)
                k = ggml_add(ctx, k, ggml_scale(ctx, ggml_mul_mat(ctx, ly->lora_sa_k_b, ggml_mul_mat(ctx, ly->lora_sa_k_a, norm_sa)), lora_scale));
            if (ly->lora_sa_v_a && ly->lora_sa_v_b)
                v = ggml_add(ctx, v, ggml_scale(ctx, ggml_mul_mat(ctx, ly->lora_sa_v_b, ggml_mul_mat(ctx, ly->lora_sa_v_a, norm_sa)), lora_scale));
        }
    } else if (ly->sa_qk) {
        struct ggml_tensor * qk = dit_ggml_linear(ctx, ly->sa_qk, norm_sa);
        q = ggml_cont(ctx, ggml_view_3d(ctx, qk, q_dim, S, N, qk->nb[1], qk->nb[2], 0));
        k = ggml_cont(ctx, ggml_view_3d(ctx, qk, kv_dim, S, N, qk->nb[1], qk->nb[2], (size_t)q_dim * qk->nb[0]));
        if (lora_scale != 0.0f) {
            if (ly->lora_sa_q_a && ly->lora_sa_q_b)
                q = ggml_add(ctx, q, ggml_scale(ctx, ggml_mul_mat(ctx, ly->lora_sa_q_b, ggml_mul_mat(ctx, ly->lora_sa_q_a, norm_sa)), lora_scale));
            if (ly->lora_sa_k_a && ly->lora_sa_k_b)
                k = ggml_add(ctx, k, ggml_scale(ctx, ggml_mul_mat(ctx, ly->lora_sa_k_b, ggml_mul_mat(ctx, ly->lora_sa_k_a, norm_sa)), lora_scale));
        }
        v = dit_ggml_linear_lora(ctx, ly->sa_v_proj, ly->lora_sa_v_a, ly->lora_sa_v_b, lora_scale, norm_sa);
    } else {
        q = dit_ggml_linear_lora(ctx, ly->sa_q_proj, ly->lora_sa_q_a, ly->lora_sa_q_b, lora_scale, norm_sa);
        k = dit_ggml_linear_lora(ctx, ly->sa_k_proj, ly->lora_sa_k_a, ly->lora_sa_k_b, lora_scale, norm_sa);
        v = dit_ggml_linear_lora(ctx, ly->sa_v_proj, ly->lora_sa_v_a, ly->lora_sa_v_b, lora_scale, norm_sa);
    }

    // 2) Reshape to heads: [Nh*D, S, N] -> [D, Nh, S, N]
    //    Rope merges S*N then restores 4D. Permute to flash_attn layout after rope.
    q = ggml_reshape_4d(ctx, q, D, Nh, S, N);
    k = ggml_reshape_4d(ctx, k, D, Nkv, S, N);
    v = ggml_reshape_4d(ctx, v, D, Nkv, S, N);

    // 4) QK-Norm: per-head RMSNorm on D dimension
    //    [D, Nh, S] rms_norm operates on ne[0]=D
    q = ggml_rms_norm(ctx, q, c.rms_norm_eps);
    q = ggml_mul(ctx, q, dit_ggml_f32(ctx, ly->sa_q_norm));
    k = ggml_rms_norm(ctx, k, c.rms_norm_eps);
    k = ggml_mul(ctx, k, dit_ggml_f32(ctx, ly->sa_k_norm));

    // 5) RoPE (bidirectional, sequential positions)
    //    ggml_rope_ext asserts ne[2] == positions.ne[0].
    //    With batch N>1, positions has S*N elements (repeated [0..S-1] per batch).
    //    Merge S and N before rope, then restore 4D after.
    q = ggml_reshape_3d(ctx, q, D, Nh, S * N);
    k = ggml_reshape_3d(ctx, k, D, Nkv, S * N);
    q = ggml_rope_ext(ctx, q, positions, NULL,
                       D, 2 /*mode=NEOX*/, 0 /*n_ctx_orig*/,
                       c.rope_theta, 1.0f /*freq_scale*/,
                       0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, positions, NULL,
                       D, 2, 0,
                       c.rope_theta, 1.0f,
                       0.0f, 1.0f, 0.0f, 0.0f);
    q = ggml_reshape_4d(ctx, q, D, Nh, S, N);
    k = ggml_reshape_4d(ctx, k, D, Nkv, S, N);

    if (layer_idx == 0) {
        ggml_set_name(q, "layer0_q_after_rope");
        ggml_set_output(q);
        ggml_set_name(k, "layer0_k_after_rope");
        ggml_set_output(k);
    }

    // 6) Permute for flash_attn_ext: [D, Nh, S, N] -> [D, S, Nh, N]
    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    // 7) Attention (flash on GPU, F32 manual on CPU)
    //    Q[D, S, Nh, N], K[D, S, Nkv, N], V[D, S, Nkv, N]
    float scale = 1.0f / sqrtf((float)D);
    struct ggml_tensor * attn = m->use_flash_attn
        ? ggml_flash_attn_ext(ctx, q, k, v, mask, scale, 0.0f, 0.0f)
        : dit_attn_f32(ctx, q, k, v, mask, scale);

    // Both return [D, Nh, S, N]
    // Reshape: [D, Nh, S, N] -> [D*Nh, S, N] = [H, S, N]
    attn = ggml_reshape_3d(ctx, attn, Nh * D, S, N);

    if (layer_idx == 0) {
        ggml_set_name(attn, "layer0_attn_out");
        ggml_set_output(attn);
    }

    // 8) O projection: [Nh*D, S, N] -> [H, S, N]
    struct ggml_tensor * out = dit_ggml_linear_lora(ctx, ly->sa_o_proj, ly->lora_sa_o_a, ly->lora_sa_o_b, m->lora_scale, attn);
    return out;
}

// Build MLP sub-graph: SwiGLU
// norm_ffn: [H, S, N] pre-normalized + AdaLN-modulated hidden state
// Returns: output [H, S, N]
static struct ggml_tensor * dit_ggml_build_mlp(
        struct ggml_context * ctx,
        DiTGGML * m,
        DiTGGMLLayer * ly,
        struct ggml_tensor * norm_ffn,
        int S) {

    DiTGGMLConfig & c = m->cfg;
    int I = c.intermediate_size;
    int N = (int)norm_ffn->ne[2];
    float lora_scale = m->lora_scale;
    struct ggml_tensor * ff;
    if (ly->gate_up) {
        // Fused: single matmul [H, 2*I] x [H, S, N] -> [2*I, S, N], then swiglu splits ne[0]
        struct ggml_tensor * gu = dit_ggml_linear(ctx, ly->gate_up, norm_ffn);
        if (lora_scale != 0.0f && ((ly->lora_gate_a && ly->lora_gate_b) || (ly->lora_up_a && ly->lora_up_b))) {
            struct ggml_tensor * gate = ggml_cont(ctx, ggml_view_3d(ctx, gu, I, S, N, gu->nb[1], gu->nb[2], 0));
            struct ggml_tensor * up   = ggml_cont(ctx, ggml_view_3d(ctx, gu, I, S, N, gu->nb[1], gu->nb[2], (size_t)I * gu->nb[0]));
            if (ly->lora_gate_a && ly->lora_gate_b)
                gate = ggml_add(ctx, gate, ggml_scale(ctx, ggml_mul_mat(ctx, ly->lora_gate_b, ggml_mul_mat(ctx, ly->lora_gate_a, norm_ffn)), lora_scale));
            if (ly->lora_up_a && ly->lora_up_b)
                up   = ggml_add(ctx, up, ggml_scale(ctx, ggml_mul_mat(ctx, ly->lora_up_b, ggml_mul_mat(ctx, ly->lora_up_a, norm_ffn)), lora_scale));
            ff = ggml_swiglu_split(ctx, gate, up);
        } else {
            ff = ggml_swiglu(ctx, gu);
        }
    } else {
        // Separate: two matmuls + split swiglu (with optional LoRA)
        struct ggml_tensor * gate = dit_ggml_linear_lora(ctx, ly->gate_proj, ly->lora_gate_a, ly->lora_gate_b, lora_scale, norm_ffn);
        struct ggml_tensor * up   = dit_ggml_linear_lora(ctx, ly->up_proj, ly->lora_up_a, ly->lora_up_b, lora_scale, norm_ffn);
        ff = ggml_swiglu_split(ctx, gate, up);
    }

    // Down projection: [I, S] -> [H, S]
    return dit_ggml_linear_lora(ctx, ly->down_proj, ly->lora_down_a, ly->lora_down_b, lora_scale, ff);
}

// Build cross-attention sub-graph for a single layer.
// norm_ca: [H, S, N] pre-normalized hidden state (Q source)
// enc:     [H, enc_S, N] condition-embedded encoder states (K/V source)
// Returns: output [H, S, N] (NOT added to residual yet)
static struct ggml_tensor * dit_ggml_build_cross_attn(
        struct ggml_context * ctx,
        DiTGGML * m,
        DiTGGMLLayer * ly,
        struct ggml_tensor * norm_ca,   // [H, S, N]
        struct ggml_tensor * enc,       // [H, enc_S, N]
        struct ggml_tensor * positions, // unused, kept for consistency
        int S, int enc_S, int N) {

    DiTGGMLConfig & c = m->cfg;
    int D   = c.head_dim;
    int Nh  = c.n_heads;
    int Nkv = c.n_kv_heads;

    (void)positions;  // cross-attn has no RoPE

    // Q from hidden, KV from encoder (full fused, Q+KV partial, separate)
    int q_dim  = Nh * D;
    int kv_dim = Nkv * D;
    float lora_scale = m->lora_scale;
    struct ggml_tensor * q, * k, * v;
    if (ly->ca_qkv) {
        // Full QKV fused: split Q from hidden, KV from enc via weight views
        struct ggml_tensor * w_q  = ggml_view_2d(ctx, ly->ca_qkv, ly->ca_qkv->ne[0], q_dim,
                                                  ly->ca_qkv->nb[1], 0);
        struct ggml_tensor * w_kv = ggml_view_2d(ctx, ly->ca_qkv, ly->ca_qkv->ne[0], 2 * kv_dim,
                                                  ly->ca_qkv->nb[1], (size_t)q_dim * ly->ca_qkv->nb[1]);
        q = ggml_mul_mat(ctx, w_q, norm_ca);
        struct ggml_tensor * kv = ggml_mul_mat(ctx, w_kv, enc);
        k = ggml_cont(ctx, ggml_view_3d(ctx, kv, kv_dim, enc_S, N, kv->nb[1], kv->nb[2], 0));
        v = ggml_cont(ctx, ggml_view_3d(ctx, kv, kv_dim, enc_S, N, kv->nb[1], kv->nb[2], (size_t)kv_dim * kv->nb[0]));
        // LoRA on fused path: add scale * (B @ (A @ x)) for Q (from norm_ca), K/V (from enc)
        if (lora_scale != 0.0f) {
            if (ly->lora_ca_q_a && ly->lora_ca_q_b)
                q = ggml_add(ctx, q, ggml_scale(ctx, ggml_mul_mat(ctx, ly->lora_ca_q_b, ggml_mul_mat(ctx, ly->lora_ca_q_a, norm_ca)), lora_scale));
            if (ly->lora_ca_k_a && ly->lora_ca_k_b)
                k = ggml_add(ctx, k, ggml_scale(ctx, ggml_mul_mat(ctx, ly->lora_ca_k_b, ggml_mul_mat(ctx, ly->lora_ca_k_a, enc)), lora_scale));
            if (ly->lora_ca_v_a && ly->lora_ca_v_b)
                v = ggml_add(ctx, v, ggml_scale(ctx, ggml_mul_mat(ctx, ly->lora_ca_v_b, ggml_mul_mat(ctx, ly->lora_ca_v_a, enc)), lora_scale));
        }
    } else if (ly->ca_kv) {
        // Q separate, K+V fused
        q = dit_ggml_linear_lora(ctx, ly->ca_q_proj, ly->lora_ca_q_a, ly->lora_ca_q_b, lora_scale, norm_ca);
        struct ggml_tensor * kv = ggml_mul_mat(ctx, ly->ca_kv, enc);
        k = ggml_cont(ctx, ggml_view_3d(ctx, kv, kv_dim, enc_S, N, kv->nb[1], kv->nb[2], 0));
        v = ggml_cont(ctx, ggml_view_3d(ctx, kv, kv_dim, enc_S, N, kv->nb[1], kv->nb[2], (size_t)kv_dim * kv->nb[0]));
        if (lora_scale != 0.0f) {
            if (ly->lora_ca_k_a && ly->lora_ca_k_b)
                k = ggml_add(ctx, k, ggml_scale(ctx, ggml_mul_mat(ctx, ly->lora_ca_k_b, ggml_mul_mat(ctx, ly->lora_ca_k_a, enc)), lora_scale));
            if (ly->lora_ca_v_a && ly->lora_ca_v_b)
                v = ggml_add(ctx, v, ggml_scale(ctx, ggml_mul_mat(ctx, ly->lora_ca_v_b, ggml_mul_mat(ctx, ly->lora_ca_v_a, enc)), lora_scale));
        }
    } else {
        q = dit_ggml_linear_lora(ctx, ly->ca_q_proj, ly->lora_ca_q_a, ly->lora_ca_q_b, m->lora_scale, norm_ca);
        k = dit_ggml_linear_lora(ctx, ly->ca_k_proj, ly->lora_ca_k_a, ly->lora_ca_k_b, m->lora_scale, enc);
        v = dit_ggml_linear_lora(ctx, ly->ca_v_proj, ly->lora_ca_v_a, ly->lora_ca_v_b, m->lora_scale, enc);
    }

    // reshape to [D, heads, seq, N] then permute to [D, seq, heads, N]
    q = ggml_reshape_4d(ctx, q, D, Nh, S, N);
    q = ggml_permute(ctx, q, 0, 2, 1, 3);   // [D, S, Nh, N]

    k = ggml_reshape_4d(ctx, k, D, Nkv, enc_S, N);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);   // [D, enc_S, Nkv, N]

    v = ggml_reshape_4d(ctx, v, D, Nkv, enc_S, N);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);   // [D, enc_S, Nkv, N]

    // QK-norm (per head)
    q = ggml_rms_norm(ctx, q, c.rms_norm_eps);
    q = ggml_mul(ctx, q, dit_ggml_f32(ctx, ly->ca_q_norm));
    k = ggml_rms_norm(ctx, k, c.rms_norm_eps);
    k = ggml_mul(ctx, k, dit_ggml_f32(ctx, ly->ca_k_norm));

    // no RoPE for cross-attention
    // no mask (attend to all encoder positions)
    float scale = 1.0f / sqrtf((float)D);
    struct ggml_tensor * attn = m->use_flash_attn
        ? ggml_flash_attn_ext(ctx, q, k, v, NULL, scale, 0.0f, 0.0f)
        : dit_attn_f32(ctx, q, k, v, NULL, scale);

    // Attention output: [D, Nh, S, N], reshape to [H, S, N]
    attn = ggml_reshape_3d(ctx, attn, Nh * D, S, N);

    // O projection
    return dit_ggml_linear_lora(ctx, ly->ca_o_proj, ly->lora_ca_o_a, ly->lora_ca_o_b, m->lora_scale, attn);
}

// Build one full DiT layer (AdaLN + self-attn + cross-attn + FFN + gated residuals)
// hidden: [H, S, N], tproj: [6H] (combined timestep projection)
// enc: [H, enc_S, N] (condition-embedded encoder states, or NULL to skip cross-attn)
// sw_mask: [S, S] sliding window mask (or NULL for full attention)
// Returns: updated hidden [H, S, N]
static struct ggml_tensor * dit_ggml_build_layer(
        struct ggml_context * ctx,
        DiTGGML * m,
        int layer_idx,
        struct ggml_tensor * hidden,    // [H, S, N]
        struct ggml_tensor * tproj,     // [6H] f32 combined temb projection
        struct ggml_tensor * enc,       // [H, enc_S, N] or NULL
        struct ggml_tensor * positions, // [S] int32
        struct ggml_tensor * sw_mask,   // [S, S] or NULL
        int S, int enc_S, int N) {

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
    norm_sa = dit_ggml_adaln(ctx, norm_sa, scale_sa, shift_sa, m->scalar_one);

    if (layer_idx == 0) {
        ggml_set_name(norm_sa, "layer0_sa_input");
        ggml_set_output(norm_sa);
    }

    // select mask: even layers use sliding window, odd layers use full attention
    struct ggml_tensor * mask = (ly->layer_type == 0) ? sw_mask : NULL;
    struct ggml_tensor * sa_out = dit_ggml_build_self_attn(ctx, m, ly, norm_sa, positions, mask, S, N, layer_idx);

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
        struct ggml_tensor * ca_out = dit_ggml_build_cross_attn(ctx, m, ly, norm_ca, enc, positions, S, enc_S, N);
        hidden = ggml_add(ctx, hidden, ca_out);
    }

    if (layer_idx == 0) {
        ggml_set_name(hidden, "layer0_after_cross_attn");
        ggml_set_output(hidden);
    }

    // FFN with AdaLN + gated residual
    residual = hidden;
    struct ggml_tensor * norm_ffn = dit_ggml_rms_norm_weighted(ctx, hidden, ly->mlp_norm, c.rms_norm_eps);
    norm_ffn = dit_ggml_adaln(ctx, norm_ffn, scale_ffn, shift_ffn, m->scalar_one);
    struct ggml_tensor * ffn_out = dit_ggml_build_mlp(ctx, m, ly, norm_ffn, S);
    hidden = dit_ggml_gated_add(ctx, residual, ffn_out, gate_ffn);

    return hidden;
}

// Build the full DiT forward graph (all layers).
// Returns the final output tensor (velocity prediction).
// N = batch size (number of samples to denoise in parallel).
//
// Graph inputs (ggml [ne0, ne1, ne2] notation):
//   "input_latents"   [in_channels, T, N]  concat(context_latents, xt) per sample
//   "enc_hidden"      [H, enc_S, N]        text encoder hidden states (N copies)
//   "t"               [1] f32              flow matching timestep (shared)
//   "t_r"             [1] f32              reference timestep (shared)
//   "positions"       [S*N] i32            position indices 0..S-1 repeated N times
//   "sw_mask"         [S, S, 1, N] f16     sliding window mask (N identical copies)
//
// Graph outputs:
//   "velocity"        [out_channels, T, N]  predicted flow velocity
static struct ggml_cgraph * dit_ggml_build_graph(
        DiTGGML * m,
        struct ggml_context * ctx,
        int T,                 // temporal length (before patching)
        int enc_S,             // encoder sequence length
        int N,                 // batch size
        struct ggml_tensor ** p_input,     // [out] input tensor to fill
        struct ggml_tensor ** p_output) {  // [out] output tensor to read

    DiTGGMLConfig & c = m->cfg;
    int S = T / c.patch_size;  // sequence length after patching
    int H = c.hidden_size;
    int P = c.patch_size;

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);

    // Inputs

    // Concatenated latent: [in_channels, T, N] per sample
    struct ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.in_channels, T, N);
    ggml_set_name(input, "input_latents");
    ggml_set_input(input);
    *p_input = input;

    // Encoder hidden states: [H, enc_S, N]
    struct ggml_tensor * enc_hidden = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, H, enc_S, N);
    ggml_set_name(enc_hidden, "enc_hidden");
    ggml_set_input(enc_hidden);

    // Timesteps: scalars
    struct ggml_tensor * t_val = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_name(t_val, "t");
    ggml_set_input(t_val);

    struct ggml_tensor * tr_val = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_name(tr_val, "t_r");
    ggml_set_input(tr_val);

    // Position indices for RoPE: [N*S] with values [0..S-1] repeated N times.
    // The CUDA rope kernel indexes positions by channel_x = row / ne1 which
    // linearizes (ne2, ne3) = (S, N). Batch b reads pos[b*S + s], so we must
    // repeat the sequence for each batch element.
    struct ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S * N);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // ggml pitfall: flash_attn_ext reads mask as fp16!
    // Must be 4D [S, S, 1, N] not 2D [S, S]: the CUDA flash_attn_mask_to_KV_max
    // optimization kernel offsets the mask pointer by sequence*nb[3] per batch element.
    // With 2D mask (ne[3]=1), batch 1+ reads out of bounds. Replicate mask N times.
    struct ggml_tensor * sw_mask = NULL;
    if (c.sliding_window > 0 && S > c.sliding_window) {
        sw_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, S, S, 1, N);
        ggml_set_name(sw_mask, "sw_mask");
        ggml_set_input(sw_mask);
    }

    // 1) Timestep embeddings
    struct ggml_tensor * tproj;
    struct ggml_tensor * temb;

    {
        struct ggml_tensor * tproj_t;
        struct ggml_tensor * temb_t = dit_ggml_build_temb(ctx, &m->time_embed, t_val, &tproj_t, "_t");
        ggml_set_name(temb_t, "temb_t");
        ggml_set_output(temb_t);

        struct ggml_tensor * tproj_r;
        // Python passes (t - t_r) to time_embed_r, not t_r directly
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

    // 2) proj_in: patchify + linear (weight pre-permuted at load time)
    ggml_set_name(input, "proj_in_input");
    ggml_set_output(input);
    struct ggml_tensor * patched = ggml_reshape_3d(ctx, input, c.in_channels * P, S, N);
    struct ggml_tensor * hidden = dit_ggml_linear_bias(ctx, m->proj_in_w, m->proj_in_b, patched);
    ggml_set_name(hidden, "hidden_after_proj_in");
    ggml_set_output(hidden);

    // 3) Condition embedder: project encoder hidden states
    struct ggml_tensor * enc = dit_ggml_linear_bias(ctx, m->cond_emb_w, m->cond_emb_b, enc_hidden);
    ggml_set_name(enc, "enc_after_cond_emb");
    ggml_set_output(enc);

    // 4) Transformer layers
    for (int i = 0; i < c.n_layers; i++) {
        hidden = dit_ggml_build_layer(ctx, m, i, hidden, tproj, enc, positions, sw_mask, S, enc_S, N);
        // Debug dumps at key layers: 0, 6, 12, 18, 23
        if (i == 0 || i == 6 || i == 12 || i == 18 || i == c.n_layers - 1) {
            char lname[64];
            snprintf(lname, sizeof(lname), "hidden_after_layer%d", i);
            ggml_set_name(hidden, lname);
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
    norm_out = dit_ggml_adaln(ctx, norm_out, out_scale, out_shift, m->scalar_one);

    // proj_out: weight pre-permuted+transposed at load time to [H, out_ch*P] F32
    struct ggml_tensor * output = dit_ggml_linear_bias(ctx, m->proj_out_w, m->proj_out_b, norm_out);
    output = ggml_reshape_3d(ctx, output, c.out_channels, T, N);

    ggml_set_name(output, "velocity");
    ggml_set_output(output);
    *p_output = output;

    ggml_build_forward_expand(gf, output);

    return gf;
}
