#pragma once
// dit.h: ACE-Step DiT (Diffusion Transformer) via ggml compute graph
// Ported from Python ACE-Step-1.5 reference. Same weights, loaded from GGUF.
//
// Architecture: 24-layer transformer with AdaLN, GQA self-attn + cross-attn, SwiGLU MLP.
// Flow matching: 8 Euler steps (turbo schedule).
//
// ggml ops used: rms_norm, mul_mat, rope_ext, flash_attn_ext, swiglu_split,
//                conv_transpose_1d, add, mul, scale, view, reshape, permute.

#include "backend.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf-weights.h"
#include "lora-merge.h"

#include <cstdio>
#include <cstdlib>
#include <string>

// Config (mirrors dit.cuh DiTConfig)
struct DiTGGMLConfig {
    int   hidden_size       = 2048;
    int   intermediate_size = 6144;
    int   n_heads           = 16;
    int   n_kv_heads        = 8;
    int   head_dim          = 128;
    int   n_layers          = 24;
    int   in_channels       = 192;  // after context concat
    int   out_channels      = 64;   // audio_acoustic_hidden_dim
    int   patch_size        = 2;
    int   sliding_window    = 128;
    float rope_theta        = 1000000.0f;
    float rms_norm_eps      = 1e-6f;
};

// Layer weights
struct DiTGGMLTembWeights {
    struct ggml_tensor * linear_1_w;   // [256, hidden]
    struct ggml_tensor * linear_1_b;   // [hidden]
    struct ggml_tensor * linear_2_w;   // [hidden, hidden]
    struct ggml_tensor * linear_2_b;   // [hidden]
    struct ggml_tensor * time_proj_w;  // [hidden, 6*hidden]
    struct ggml_tensor * time_proj_b;  // [6*hidden]
};

struct DiTGGMLLayer {
    // Self-attention
    struct ggml_tensor * self_attn_norm;  // [hidden]
    struct ggml_tensor * sa_qkv;          // [hidden, (Nh+2*Nkv)*D] full fused (or NULL)
    struct ggml_tensor * sa_qk;           // [hidden, (Nh+Nkv)*D] partial QK fused (or NULL)
    struct ggml_tensor * sa_q_proj;       // separate fallback (NULL when any fusion active)
    struct ggml_tensor * sa_k_proj;
    struct ggml_tensor * sa_v_proj;
    struct ggml_tensor * sa_q_norm;  // [head_dim]
    struct ggml_tensor * sa_k_norm;  // [head_dim]
    struct ggml_tensor * sa_o_proj;  // [n_heads*head_dim, hidden]

    // Cross-attention
    struct ggml_tensor * cross_attn_norm;  // [hidden]
    struct ggml_tensor * ca_qkv;           // [hidden, (Nh+2*Nkv)*D] full fused (or NULL)
    struct ggml_tensor * ca_q_proj;        // separate (always for cross-attn with mixed types)
    struct ggml_tensor * ca_kv;            // [hidden, 2*Nkv*D] fused KV (or NULL)
    struct ggml_tensor * ca_k_proj;
    struct ggml_tensor * ca_v_proj;
    struct ggml_tensor * ca_q_norm;  // [head_dim]
    struct ggml_tensor * ca_k_norm;  // [head_dim]
    struct ggml_tensor * ca_o_proj;  // [n_heads*head_dim, hidden]

    // MLP
    struct ggml_tensor * mlp_norm;   // [hidden]
    struct ggml_tensor * gate_up;    // [hidden, 2*intermediate] fused (or NULL)
    struct ggml_tensor * gate_proj;  // [hidden, intermediate] (fallback if types differ)
    struct ggml_tensor * up_proj;    // [hidden, intermediate] (fallback if types differ)
    struct ggml_tensor * down_proj;  // [intermediate, hidden]

    // AdaLN scale-shift table: [6*hidden] (6 rows of [hidden])
    struct ggml_tensor * scale_shift_table;  // [hidden, 6] in ggml layout

    int layer_type;                          // 0=sliding, 1=full
};

// Full model
#define DIT_GGML_MAX_LAYERS 32

struct DiTGGML {
    DiTGGMLConfig cfg;

    // Timestep embeddings
    DiTGGMLTembWeights time_embed;
    DiTGGMLTembWeights time_embed_r;

    // proj_in: Conv1d(in_channels, hidden, kernel=2, stride=2)
    struct ggml_tensor * proj_in_w;  // [in_ch*P, H] pre-permuted F32
    struct ggml_tensor * proj_in_b;  // [hidden]

    // condition_embedder: Linear(hidden, hidden)
    struct ggml_tensor * cond_emb_w;  // [hidden, hidden]
    struct ggml_tensor * cond_emb_b;  // [hidden]

    // Layers
    DiTGGMLLayer layers[DIT_GGML_MAX_LAYERS];

    // Output
    struct ggml_tensor * norm_out;         // [hidden]
    struct ggml_tensor * out_scale_shift;  // [hidden, 2] in ggml layout
    struct ggml_tensor * proj_out_w;       // [H, out_ch*P] pre-permuted+transposed F32
    struct ggml_tensor * proj_out_b;       // [out_channels]

    // CFG (classifier-free guidance, used by base/sft models)
    struct ggml_tensor * null_condition_emb;  // [hidden] or NULL if not present

    // Backend
    ggml_backend_t       backend;
    ggml_backend_t       cpu_backend;
    ggml_backend_sched_t sched;
    bool                 use_flash_attn;

    // Weight storage
    WeightCtx wctx;

    // Pre-allocated constant for AdaLN (1+scale) fusion
    struct ggml_tensor * scalar_one;  // [1] = 1.0f, broadcast in ggml_add
};

// Load timestep embedding weights
static void dit_ggml_load_temb(DiTGGMLTembWeights * w,
                               WeightCtx *          wctx,
                               const GGUFModel &    gf,
                               const std::string &  prefix) {
    w->linear_1_w  = gf_load_tensor(wctx, gf, prefix + ".linear_1.weight");
    w->linear_1_b  = gf_load_tensor_f32(wctx, gf, prefix + ".linear_1.bias");
    w->linear_2_w  = gf_load_tensor(wctx, gf, prefix + ".linear_2.weight");
    w->linear_2_b  = gf_load_tensor_f32(wctx, gf, prefix + ".linear_2.bias");
    w->time_proj_w = gf_load_tensor(wctx, gf, prefix + ".time_proj.weight");
    w->time_proj_b = gf_load_tensor_f32(wctx, gf, prefix + ".time_proj.bias");
}

// Load proj_in weight: GGUF [H, in_ch, P] -> pre-permuted 2D [in_ch*P, H] F32
// Eliminates runtime permute+cont in the compute graph.
static struct ggml_tensor * dit_load_proj_in_w(WeightCtx *         wctx,
                                               const GGUFModel &   gf,
                                               const std::string & name,
                                               int                 H,
                                               int                 in_ch,
                                               int                 P) {
    int64_t idx = gguf_find_tensor(gf.gguf, name.c_str());
    if (idx < 0) {
        fprintf(stderr, "[GGUF] FATAL: tensor '%s' not found\n", name.c_str());
        exit(1);
    }
    struct ggml_tensor * src = ggml_get_tensor(gf.meta, name.c_str());
    if (!src) {
        fprintf(stderr, "[GGUF] FATAL: meta tensor '%s' not found\n", name.c_str());
        exit(1);
    }
    size_t       offset = gguf_get_tensor_offset(gf.gguf, idx);
    const void * raw    = gf.mapping + gf.data_offset + offset;

    struct ggml_tensor * dst = ggml_new_tensor_2d(wctx->ctx, GGML_TYPE_F32, in_ch * P, H);
    ggml_set_name(dst, name.c_str());

    size_t n = (size_t) in_ch * P * H;
    wctx->staging.emplace_back(n);
    auto & buf = wctx->staging.back();

    // src ggml [P, in_ch, H]: elem(p, ic, h) = raw[h*P*in_ch + ic*P + p]
    // dst ggml [in_ch*P, H]:  elem(j, h)     = buf[h*in_ch*P + j]  where j = p*in_ch + ic
    auto cvt = [&](auto read_fn) {
        for (int h = 0; h < H; h++) {
            for (int ic = 0; ic < in_ch; ic++) {
                for (int p = 0; p < P; p++) {
                    buf[h * in_ch * P + p * in_ch + ic] = read_fn(h * P * in_ch + ic * P + p);
                }
            }
        }
    };
    if (src->type == GGML_TYPE_BF16) {
        const uint16_t * s = (const uint16_t *) raw;
        cvt([&](int i) { return ggml_bf16_to_fp32(*(const ggml_bf16_t *) &s[i]); });
    } else if (src->type == GGML_TYPE_F16) {
        const ggml_fp16_t * s = (const ggml_fp16_t *) raw;
        cvt([&](int i) { return ggml_fp16_to_fp32(s[i]); });
    } else if (src->type == GGML_TYPE_F32) {
        const float * s = (const float *) raw;
        cvt([&](int i) { return s[i]; });
    } else {
        fprintf(stderr, "[GGUF] FATAL: unsupported type %d for '%s' in proj_in pre-permute\n", src->type, name.c_str());
        exit(1);
    }
    wctx->pending.push_back({ dst, buf.data(), n * sizeof(float), 0 });
    return dst;
}

// Load proj_out weight: GGUF [H, out_ch, P] -> pre-permuted+transposed 2D [H, out_ch*P] F32
// Eliminates runtime permute+cont+transpose+cont in the compute graph.
static struct ggml_tensor * dit_load_proj_out_w(WeightCtx *         wctx,
                                                const GGUFModel &   gf,
                                                const std::string & name,
                                                int                 H,
                                                int                 out_ch,
                                                int                 P) {
    int64_t idx = gguf_find_tensor(gf.gguf, name.c_str());
    if (idx < 0) {
        fprintf(stderr, "[GGUF] FATAL: tensor '%s' not found\n", name.c_str());
        exit(1);
    }
    struct ggml_tensor * src = ggml_get_tensor(gf.meta, name.c_str());
    if (!src) {
        fprintf(stderr, "[GGUF] FATAL: meta tensor '%s' not found\n", name.c_str());
        exit(1);
    }
    size_t       offset = gguf_get_tensor_offset(gf.gguf, idx);
    const void * raw    = gf.mapping + gf.data_offset + offset;

    struct ggml_tensor * dst = ggml_new_tensor_2d(wctx->ctx, GGML_TYPE_F32, H, out_ch * P);
    ggml_set_name(dst, name.c_str());

    size_t n = (size_t) out_ch * P * H;
    wctx->staging.emplace_back(n);
    auto & buf = wctx->staging.back();

    // src ggml [P, out_ch, H]: elem(p, oc, h) = raw[h*P*out_ch + oc*P + p]
    // dst ggml [H, out_ch*P]:  elem(h, j)     = buf[j*H + h]  where j = p*out_ch + oc
    auto cvt = [&](auto read_fn) {
        for (int h = 0; h < H; h++) {
            for (int oc = 0; oc < out_ch; oc++) {
                for (int p = 0; p < P; p++) {
                    buf[(p * out_ch + oc) * H + h] = read_fn(h * P * out_ch + oc * P + p);
                }
            }
        }
    };
    if (src->type == GGML_TYPE_BF16) {
        const uint16_t * s = (const uint16_t *) raw;
        cvt([&](int i) { return ggml_bf16_to_fp32(*(const ggml_bf16_t *) &s[i]); });
    } else if (src->type == GGML_TYPE_F16) {
        const ggml_fp16_t * s = (const ggml_fp16_t *) raw;
        cvt([&](int i) { return ggml_fp16_to_fp32(s[i]); });
    } else if (src->type == GGML_TYPE_F32) {
        const float * s = (const float *) raw;
        cvt([&](int i) { return s[i]; });
    } else {
        fprintf(stderr, "[GGUF] FATAL: unsupported type %d for '%s' in proj_out pre-permute\n", src->type,
                name.c_str());
        exit(1);
    }
    wctx->pending.push_back({ dst, buf.data(), n * sizeof(float), 0 });
    return dst;
}

// Load full DiT model from GGUF
static bool dit_ggml_load(DiTGGML *     m,
                          const char *  gguf_path,
                          DiTGGMLConfig cfg,
                          const char *  lora_path  = nullptr,
                          float         lora_scale = 1.0f) {
    m->cfg = cfg;

    GGUFModel gf;
    if (!gf_load(&gf, gguf_path)) {
        fprintf(stderr, "[Load] FATAL: cannot load %s\n", gguf_path);
        return false;
    }

    // Count tensors: temb(6*2) + proj_in(2) + cond_emb(2) + layers(19*24) + output(4) + null_cond(1) + scalar_one(1) = 476
    int n_tensors = 6 * 2 + 2 + 2 + 19 * cfg.n_layers + 4 + 1 + 1;
    wctx_init(&m->wctx, n_tensors);

    // Timestep embeddings
    dit_ggml_load_temb(&m->time_embed, &m->wctx, gf, "decoder.time_embed");
    dit_ggml_load_temb(&m->time_embed_r, &m->wctx, gf, "decoder.time_embed_r");

    // proj_in: Conv1d weight [hidden, in_ch, patch_size]
    // Pre-permuted to 2D [in_ch*P, H] F32 at load time
    m->proj_in_w =
        dit_load_proj_in_w(&m->wctx, gf, "decoder.proj_in.1.weight", cfg.hidden_size, cfg.in_channels, cfg.patch_size);
    m->proj_in_b = gf_load_tensor_f32(&m->wctx, gf, "decoder.proj_in.1.bias");

    // condition_embedder
    m->cond_emb_w = gf_load_tensor(&m->wctx, gf, "decoder.condition_embedder.weight");
    m->cond_emb_b = gf_load_tensor_f32(&m->wctx, gf, "decoder.condition_embedder.bias");

    // Layers
    for (int i = 0; i < cfg.n_layers; i++) {
        char prefix[128];
        snprintf(prefix, sizeof(prefix), "decoder.layers.%d", i);
        std::string    p(prefix);
        DiTGGMLLayer & ly = m->layers[i];

        // Self-attention: try full QKV, partial QK, separate
        ly.self_attn_norm = gf_load_tensor_f32(&m->wctx, gf, p + ".self_attn_norm.weight");
        ly.sa_qkv = gf_load_qkv_fused(&m->wctx, gf, p + ".self_attn.q_proj.weight", p + ".self_attn.k_proj.weight",
                                      p + ".self_attn.v_proj.weight");
        if (!ly.sa_qkv) {
            // Try Q+K fusion (same input, often same type in K-quants)
            ly.sa_qk = gf_load_pair_fused(&m->wctx, gf, p + ".self_attn.q_proj.weight", p + ".self_attn.k_proj.weight");
            if (ly.sa_qk) {
                ly.sa_v_proj = gf_load_tensor(&m->wctx, gf, p + ".self_attn.v_proj.weight");
                if (i == 0) {
                    fprintf(stderr, "[DiT] Self-attn: Q+K fused, V separate\n");
                }
            } else {
                ly.sa_q_proj = gf_load_tensor(&m->wctx, gf, p + ".self_attn.q_proj.weight");
                ly.sa_k_proj = gf_load_tensor(&m->wctx, gf, p + ".self_attn.k_proj.weight");
                ly.sa_v_proj = gf_load_tensor(&m->wctx, gf, p + ".self_attn.v_proj.weight");
                if (i == 0) {
                    fprintf(stderr, "[DiT] Self-attn: all separate (3 types differ)\n");
                }
            }
        } else {
            if (i == 0) {
                fprintf(stderr, "[DiT] Self-attn: Q+K+V fused\n");
            }
        }
        ly.sa_q_norm = gf_load_tensor_f32(&m->wctx, gf, p + ".self_attn.q_norm.weight");
        ly.sa_k_norm = gf_load_tensor_f32(&m->wctx, gf, p + ".self_attn.k_norm.weight");
        ly.sa_o_proj = gf_load_tensor(&m->wctx, gf, p + ".self_attn.o_proj.weight");

        // Cross-attention: try full QKV, K+V fused, separate
        ly.cross_attn_norm = gf_load_tensor_f32(&m->wctx, gf, p + ".cross_attn_norm.weight");
        ly.ca_qkv = gf_load_qkv_fused(&m->wctx, gf, p + ".cross_attn.q_proj.weight", p + ".cross_attn.k_proj.weight",
                                      p + ".cross_attn.v_proj.weight");
        if (!ly.ca_qkv) {
            ly.ca_q_proj = gf_load_tensor(&m->wctx, gf, p + ".cross_attn.q_proj.weight");
            // Try K+V fusion (same input enc, may share type)
            ly.ca_kv =
                gf_load_pair_fused(&m->wctx, gf, p + ".cross_attn.k_proj.weight", p + ".cross_attn.v_proj.weight");
            if (ly.ca_kv) {
                if (i == 0) {
                    fprintf(stderr, "[DiT] Cross-attn: Q separate, K+V fused\n");
                }
            } else {
                ly.ca_k_proj = gf_load_tensor(&m->wctx, gf, p + ".cross_attn.k_proj.weight");
                ly.ca_v_proj = gf_load_tensor(&m->wctx, gf, p + ".cross_attn.v_proj.weight");
                if (i == 0) {
                    fprintf(stderr, "[DiT] Cross-attn: all separate\n");
                }
            }
        } else {
            if (i == 0) {
                fprintf(stderr, "[DiT] Cross-attn: Q+K+V fused\n");
            }
        }
        ly.ca_q_norm = gf_load_tensor_f32(&m->wctx, gf, p + ".cross_attn.q_norm.weight");
        ly.ca_k_norm = gf_load_tensor_f32(&m->wctx, gf, p + ".cross_attn.k_norm.weight");
        ly.ca_o_proj = gf_load_tensor(&m->wctx, gf, p + ".cross_attn.o_proj.weight");

        // MLP: try gate+up fusion (same input, same pattern as QKV)
        ly.mlp_norm = gf_load_tensor_f32(&m->wctx, gf, p + ".mlp_norm.weight");
        ly.gate_up  = gf_load_pair_fused(&m->wctx, gf, p + ".mlp.gate_proj.weight", p + ".mlp.up_proj.weight");
        if (ly.gate_up) {
            if (i == 0) {
                fprintf(stderr, "[DiT] MLP: gate+up fused\n");
            }
        } else {
            ly.gate_proj = gf_load_tensor(&m->wctx, gf, p + ".mlp.gate_proj.weight");
            ly.up_proj   = gf_load_tensor(&m->wctx, gf, p + ".mlp.up_proj.weight");
            if (i == 0) {
                fprintf(stderr, "[DiT] MLP: gate+up separate (types differ)\n");
            }
        }
        ly.down_proj = gf_load_tensor(&m->wctx, gf, p + ".mlp.down_proj.weight");

        // AdaLN scale_shift_table [1, 6, hidden] in GGUF
        ly.scale_shift_table = gf_load_tensor_f32(&m->wctx, gf, p + ".scale_shift_table");

        ly.layer_type = (i % 2 == 0) ? 0 : 1;  // 0=sliding, 1=full
    }

    // Output
    m->norm_out        = gf_load_tensor_f32(&m->wctx, gf, "decoder.norm_out.weight");
    m->out_scale_shift = gf_load_tensor_f32(&m->wctx, gf, "decoder.scale_shift_table");
    m->proj_out_w = dit_load_proj_out_w(&m->wctx, gf, "decoder.proj_out.1.weight", cfg.hidden_size, cfg.out_channels,
                                        cfg.patch_size);
    m->proj_out_b = gf_load_tensor_f32(&m->wctx, gf, "decoder.proj_out.1.bias");

    // Null condition embedding for CFG (base/sft models; turbo has it but unused at inference)
    m->null_condition_emb = gf_try_load_tensor(&m->wctx, gf, "null_condition_emb");
    if (m->null_condition_emb) {
        fprintf(stderr, "[Load] null_condition_emb found (CFG available)\n");
    }

    // Scalar constant for AdaLN (1+scale) fusion
    static const float one_val = 1.0f;
    m->scalar_one              = ggml_new_tensor_1d(m->wctx.ctx, GGML_TYPE_F32, 1);
    m->wctx.pending.push_back({ m->scalar_one, &one_val, sizeof(float), 0 });

    // Merge LoRA deltas into projection weights (before GPU upload and QKV fusion)
    if (lora_path) {
        lora_merge(&m->wctx, gf, lora_path, lora_scale);
    }

    // Allocate backend buffer and copy weights
    if (!wctx_alloc(&m->wctx, m->backend)) {
        gf_close(&gf);
        return false;
    }
    gf_close(&gf);

    fprintf(stderr, "[Load] DiT: %d layers, H=%d, Nh=%d/%d, D=%d\n", cfg.n_layers, cfg.hidden_size, cfg.n_heads,
            cfg.n_kv_heads, cfg.head_dim);
    return true;
}

// Backend init
static void dit_ggml_init_backend(DiTGGML * m) {
    BackendPair bp    = backend_init("DiT");
    m->backend        = bp.backend;
    m->cpu_backend    = bp.cpu_backend;
    m->sched          = backend_sched_new(bp, 8192);
    // flash_attn_ext accumulates in F16 on CPU, causing audible drift over
    // 24 layers x 8 steps. Use F32 manual attention on CPU instead.
    m->use_flash_attn = (bp.backend != bp.cpu_backend);
}

static void dit_ggml_free(DiTGGML * m) {
    if (m->sched) {
        ggml_backend_sched_free(m->sched);
    }
    backend_release(m->backend, m->cpu_backend);
    wctx_free(&m->wctx);
    *m = {};
}
