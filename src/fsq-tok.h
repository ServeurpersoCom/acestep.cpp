// fsq-tok.h: FSQ tokenizer (VAE latents -> FSQ codes) via ggml
//
// Converts VAE latent frames [T_25Hz, 64] to FSQ integer codes [T_5Hz].
// Pool 5 25Hz frames into 1 5Hz code via attention pooling.
//
// Architecture per group (5 frames -> 1 code):
//   audio_acoustic_proj(64->2048) -> embed_tokens(2048->2048)
//   -> prepend CLS token -> 2L Qwen3 encoder -> take CLS output
//   -> RMSNorm -> project_in(2048->6) -> FSQ quantize -> integer index
//
// Weights live in the DiT GGUF (prefix "tokenizer.")
// Output codes feed into detok_ggml_decode (fsq-detok.h) to get DiT context.

#pragma once
#include "fsq-detok.h"
#include "qwen3-enc.h"

struct TokGGML {
    // audio_acoustic_proj: Linear(64, 2048)
    struct ggml_tensor * proj_w;  // [64, 2048]
    struct ggml_tensor * proj_b;  // [2048]

    // attention_pooler
    struct ggml_tensor * embed_w;      // [2048, 2048]
    struct ggml_tensor * embed_b;      // [2048]
    struct ggml_tensor * special_tok;  // [2048] (CLS token)
    Qwen3Layer           layers[2];
    struct ggml_tensor * norm;         // [2048]

    // quantizer.project_in: Linear(2048, 6)
    struct ggml_tensor * fsq_in_w;  // [2048, 6]
    struct ggml_tensor * fsq_in_b;  // [6]

    Qwen3Config          cfg;
    ggml_backend_t       backend;
    ggml_backend_t       cpu_backend;
    ggml_backend_sched_t sched;
    bool                 use_flash_attn;
    WeightCtx            wctx;
};

// FSQ encode: 6 raw values from project_in -> flat integer index
// Matches Python vector_quantize_pytorch FSQ.symmetry_preserving_bound() + codes_to_indices()
// QL(x) = 2/(L-1) * floor((L-1)*(tanh(x)+1)/2 + 0.5) - 1
// code = floor((L-1) * (tanh(x) + 1) / 2 + 0.5)  ->  integer [0, L-1]
static int fsq_encode_index(const float * raw_vals) {
    int index = 0;
    int mult  = 1;
    for (int d = 0; d < FSQ_NDIMS; d++) {
        int   L    = FSQ_LEVELS[d];
        float t    = tanhf(raw_vals[d]);
        int   code = (int) floorf((float) (L - 1) * (t + 1.0f) / 2.0f + 0.5f);
        if (code < 0) {
            code = 0;
        }
        if (code >= L) {
            code = L - 1;
        }
        index += code * mult;
        mult *= L;
    }
    return index;
}

// Load tokenizer weights from DiT GGUF
static bool tok_ggml_load(TokGGML * m, const char * gguf_path, ggml_backend_t backend, ggml_backend_t cpu_backend) {
    m->backend        = backend;
    m->cpu_backend    = cpu_backend;
    m->use_flash_attn = true;

    // Same Qwen3 config as detokenizer (2 layers, H=2048)
    m->cfg.n_layers          = 2;
    m->cfg.hidden_size       = 2048;
    m->cfg.n_heads           = 16;
    m->cfg.n_kv_heads        = 8;
    m->cfg.head_dim          = 128;
    m->cfg.intermediate_size = 6144;
    m->cfg.rms_norm_eps      = 1e-6f;
    m->cfg.rope_theta        = 1000000.0f;
    m->cfg.is_causal         = false;

    GGUFModel gf;
    if (!gf_load(&gf, gguf_path)) {
        fprintf(stderr, "[Tok] FATAL: cannot load %s\n", gguf_path);
        return false;
    }

    // proj(2) + embed(2) + special(1) + 2 layers x 11(22) + norm(1) + fsq_in(2) = 30
    wctx_init(&m->wctx, 30);

    m->proj_w      = gf_load_tensor(&m->wctx, gf, "tokenizer.audio_acoustic_proj.weight");
    m->proj_b      = gf_load_tensor(&m->wctx, gf, "tokenizer.audio_acoustic_proj.bias");
    m->embed_w     = gf_load_tensor(&m->wctx, gf, "tokenizer.attention_pooler.embed_tokens.weight");
    m->embed_b     = gf_load_tensor(&m->wctx, gf, "tokenizer.attention_pooler.embed_tokens.bias");
    m->special_tok = gf_load_tensor(&m->wctx, gf, "tokenizer.attention_pooler.special_token");
    m->norm        = gf_load_tensor(&m->wctx, gf, "tokenizer.attention_pooler.norm.weight");
    m->fsq_in_w    = gf_load_tensor(&m->wctx, gf, "tokenizer.quantizer.project_in.weight");
    m->fsq_in_b    = gf_load_tensor(&m->wctx, gf, "tokenizer.quantizer.project_in.bias");

    for (int i = 0; i < 2; i++) {
        char prefix[128];
        snprintf(prefix, sizeof(prefix), "tokenizer.attention_pooler.layers.%d", i);
        qwen3_load_layer(&m->wctx, gf, &m->layers[i], prefix);
    }

    if (!wctx_alloc(&m->wctx, backend)) {
        gf_close(&gf);
        return false;
    }
    gf_close(&gf);

    // Scheduler
    ggml_backend_t backends[2] = { backend, cpu_backend };
    int            n           = (backend == cpu_backend) ? 1 : 2;
    m->sched                   = ggml_backend_sched_new(backends, NULL, n, 4096, false, true);
    if (!m->sched) {
        fprintf(stderr, "[Tok] FATAL: failed to create scheduler\n");
        return false;
    }

    fprintf(stderr, "[Tok] Loaded: 2 layers, H=%d, pool_window=5\n", m->cfg.hidden_size);
    return true;
}

// Tokenize VAE latents to FSQ codes
// latents: [T_25Hz, 64] time-major (from VAE encoder)
// codes_out: caller-allocated, at least (T_25Hz + 4) / 5 ints
// silence_latent: [15000, 64] time-major (for padding to multiple of 5)
// Returns T_5Hz (number of codes) or -1 on error
static int tok_ggml_encode(TokGGML *     m,
                           const float * latents,
                           int           T_25Hz,
                           int *         codes_out,
                           const float * silence_latent) {
    int P = 5;                   // pool_window_size
    int S = 6;                   // sequence length per group (1 CLS + 5 patches)
    int H = m->cfg.hidden_size;  // 2048

    // Pad T_25Hz to multiple of 5
    int pad      = (P - (T_25Hz % P)) % P;
    int T_padded = T_25Hz + pad;
    int T_5Hz    = T_padded / P;

    // Build padded input
    std::vector<float> input((size_t) T_padded * 64);
    memcpy(input.data(), latents, (size_t) T_25Hz * 64 * sizeof(float));
    if (pad > 0) {
        memcpy(input.data() + (size_t) T_25Hz * 64, silence_latent, (size_t) pad * 64 * sizeof(float));
    }

    // Build graph (one group: input [64, 5] -> output [6])
    size_t                  ctx_size = ggml_tensor_overhead() * 256 + ggml_graph_overhead_custom(4096, false);
    uint8_t *               ctx_buf  = (uint8_t *) malloc(ctx_size);
    struct ggml_init_params gparams  = { ctx_size, ctx_buf, true };
    struct ggml_context *   ctx      = ggml_init(gparams);

    // Input: 5 VAE latent frames [64, 5]
    struct ggml_tensor * tok_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, P);
    ggml_set_name(tok_in, "tok_in");
    ggml_set_input(tok_in);

    // audio_acoustic_proj: [64, 5] -> [2048, 5]
    struct ggml_tensor * projected = ggml_mul_mat(ctx, m->proj_w, tok_in);
    projected                      = ggml_add(ctx, projected, qwen3_f32(ctx, ggml_reshape_2d(ctx, m->proj_b, H, 1)));

    // embed_tokens: [2048, 5] -> [2048, 5]
    struct ggml_tensor * embedded = ggml_mul_mat(ctx, m->embed_w, projected);
    embedded                      = ggml_add(ctx, embedded, qwen3_f32(ctx, ggml_reshape_2d(ctx, m->embed_b, H, 1)));

    // Prepend CLS special_token: [2048, 1] ++ [2048, 5] -> [2048, 6]
    struct ggml_tensor * cls    = ggml_reshape_2d(ctx, qwen3_f32(ctx, m->special_tok), H, 1);
    struct ggml_tensor * hidden = ggml_concat(ctx, cls, embedded, 1);

    // Position indices for RoPE: [0, 1, 2, 3, 4, 5]
    struct ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S);
    ggml_set_name(positions, "tok_pos");
    ggml_set_input(positions);

    // 2L Qwen3 encoder + RMSNorm (non-causal, no mask at S=6)
    hidden = qwen3_build_layers(ctx, m->cfg, m->layers, m->norm, hidden, positions, NULL, S, m->use_flash_attn);
    // hidden: [2048, 6]

    // Take CLS token (column 0): [2048, 1]
    struct ggml_tensor * cls_out = ggml_view_2d(ctx, hidden, H, 1, hidden->nb[1], 0);

    // project_in: [2048, 1] -> [6, 1]
    struct ggml_tensor * fsq_vals = ggml_mul_mat(ctx, m->fsq_in_w, cls_out);
    fsq_vals = ggml_add(ctx, fsq_vals, qwen3_f32(ctx, ggml_reshape_2d(ctx, m->fsq_in_b, FSQ_NDIMS, 1)));
    ggml_set_name(fsq_vals, "tok_out");
    ggml_set_output(fsq_vals);

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
    ggml_build_forward_expand(gf, fsq_vals);

    if (!ggml_backend_sched_alloc_graph(m->sched, gf)) {
        fprintf(stderr, "[Tok] FATAL: graph alloc failed\n");
        ggml_free(ctx);
        free(ctx_buf);
        return -1;
    }

    // Set positions once: [0, 1, 2, 3, 4, 5]
    int32_t pos_data[6] = { 0, 1, 2, 3, 4, 5 };

    struct ggml_tensor * t_in  = ggml_graph_get_tensor(gf, "tok_in");
    struct ggml_tensor * t_out = ggml_graph_get_tensor(gf, "tok_out");
    struct ggml_tensor * t_pos = ggml_graph_get_tensor(gf, "tok_pos");

    // Loop over T_5Hz groups
    float fsq_buf[FSQ_NDIMS];
    for (int g = 0; g < T_5Hz; g++) {
        // Re-set positions each iteration (allocator may share buffer)
        ggml_backend_tensor_set(t_pos, pos_data, 0, S * sizeof(int32_t));

        // Upload 5 frames: input[g*5 .. g*5+4], each 64 floats
        // ggml layout [64, 5]: 5 columns of 64 elements, contiguous per column
        // Our data is [T, 64] time-major: frame t at input[t*64]
        // ggml ne[0]=64 is contiguous, so data[col * 64 + row] = data[t * 64 + c]
        // This matches! Just upload 5*64 contiguous floats.
        ggml_backend_tensor_set(t_in, input.data() + (size_t) g * P * 64, 0, (size_t) P * 64 * sizeof(float));

        ggml_backend_sched_graph_compute(m->sched, gf);

        // Read 6 FSQ values, encode to integer index
        ggml_backend_tensor_get(t_out, fsq_buf, 0, FSQ_NDIMS * sizeof(float));
        codes_out[g] = fsq_encode_index(fsq_buf);
    }

    fprintf(stderr, "[Tok] Tokenized: %d frames -> %d codes (%.1fs @ 5Hz)\n", T_25Hz, T_5Hz, (float) T_5Hz / 5.0f);

    ggml_backend_sched_reset(m->sched);
    ggml_free(ctx);
    free(ctx_buf);
    return T_5Hz;
}

// Free
static void tok_ggml_free(TokGGML * m) {
    if (m->sched) {
        ggml_backend_sched_free(m->sched);
    }
    wctx_free(&m->wctx);
    *m = {};
}
