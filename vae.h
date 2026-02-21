// vae.h: AutoencoderOobleck decoder (audio VAE) via ggml
//
// Architecture: conv1(64->2048,k=7) -> 5xblock(snake+convT+3xresunit) -> snake+conv2(128->2,k=7)
// ResUnit(ch, dil): skip=x -> snake->conv(k=7,dil)->snake->conv(k=1)->+skip
// Snake: x + sin^2(e^a * x) / e^b
// Weight norm fused at load: w = g*v/||v||
// Upsample: 10x6x4x4x2 = 1920x

#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf_weights.h"
#include "backend.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// Structs
struct VAEResUnit {
    struct ggml_tensor * s1a, * s1b;   // snake1 exp(alpha), exp(beta) [1, C]
    struct ggml_tensor * c1w, * c1b;   // conv1 fused [7, C, C], bias [C]
    struct ggml_tensor * s2a, * s2b;   // snake2
    struct ggml_tensor * c2w, * c2b;   // conv2 fused [1, C, C], bias [C]
    int dilation;
};

struct VAEBlock {
    struct ggml_tensor * sa, * sb;     // snake exp(a/b) [1, in_ch]
    struct ggml_tensor * ctw, * ctb;   // conv_transpose fused [K, out_ch, in_ch], bias [out_ch]
    int in_ch, out_ch, stride, kernel;
    VAEResUnit ru[3];
};

struct VAEGGML {
    struct ggml_tensor * c1w, * c1b;   // conv1 [7, 64, 2048], bias [2048]
    VAEBlock blk[5];
    struct ggml_tensor * sa, * sb;     // final snake [1, 128]
    struct ggml_tensor * c2w;          // conv2 [7, 128, 2] (no bias)

    ggml_backend_t backend;
    ggml_backend_t cpu_backend;
    ggml_backend_sched_t sched;
    ggml_backend_buffer_t buf;
    struct ggml_context * weight_ctx;  // holds weight tensor metadata
};

// Load helpers
// Fuse weight_norm: w = g*v/||v||, write f32 into pre-allocated ggml_tensor
// Works for both Conv1d [OC,IC,K] and ConvTranspose1d [IC,OC,K]:
// weight_norm normalizes over dim=0 (shape[0]), regardless of semantics.
static void vae_fuse_wn(struct ggml_tensor * dst, const GGUFModel & gf, const std::string & pfx) {
    struct ggml_tensor * mv = ggml_get_tensor(gf.meta, (pfx + ".weight_v").c_str());
    const uint16_t * g = (const uint16_t *)gf_get_data(gf, (pfx + ".weight_g").c_str());
    const uint16_t * v = (const uint16_t *)gf_get_data(gf, (pfx + ".weight_v").c_str());
    // PyTorch dim0 is ggml ne[n_dims-1]
    int n_dims = ggml_n_dims(mv);
    int dim0 = (int)mv->ne[n_dims - 1];
    int fan = (int)(ggml_nelements(mv) / dim0);
    std::vector<float> w(dim0 * fan);
    for (int d = 0; d < dim0; d++) {
        float gv = ggml_bf16_to_fp32(*(const ggml_bf16_t *)&g[d]);
        float nsq = 0;
        for (int i = 0; i < fan; i++) {
            float vv = ggml_bf16_to_fp32(*(const ggml_bf16_t *)&v[d * fan + i]);
            nsq += vv * vv;
        }
        float s = gv / (sqrtf(nsq) + 1e-12f);
        for (int i = 0; i < fan; i++) {
            float vv = ggml_bf16_to_fp32(*(const ggml_bf16_t *)&v[d * fan + i]);
            w[d * fan + i] = vv * s;
        }
    }
    if (dst->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> w16(w.size());
        ggml_fp32_to_fp16_row(w.data(), w16.data(), (int)w.size());
        ggml_backend_tensor_set(dst, w16.data(), 0, w16.size() * sizeof(ggml_fp16_t));
    } else {
        ggml_backend_tensor_set(dst, w.data(), 0, w.size() * sizeof(float));
    }
}

// Load bf16 snake param [1,C,1] -> exp -> f32 [1, C]
static void vae_load_snake(struct ggml_tensor * dst, const GGUFModel & gf, const std::string & name) {
    struct ggml_tensor * mt = ggml_get_tensor(gf.meta, name.c_str());
    int C = (int)mt->ne[1];  // PyTorch [1,C,1] -> ggml ne=[1,C,1], middle dim
    const uint16_t * raw = (const uint16_t *)gf_get_data(gf, name.c_str());
    std::vector<float> d(C);
    for (int i = 0; i < C; i++)
        d[i] = expf(ggml_bf16_to_fp32(*(const ggml_bf16_t *)&raw[i]));
    ggml_backend_tensor_set(dst, d.data(), 0, C * sizeof(float));
}

// Load bf16 snake param [1,C,1] -> 1/exp -> f32 [1, C] (reciprocal for mul fusion)
static void vae_load_snake_inv(struct ggml_tensor * dst, const GGUFModel & gf, const std::string & name) {
    struct ggml_tensor * mt = ggml_get_tensor(gf.meta, name.c_str());
    int C = (int)mt->ne[1];
    const uint16_t * raw = (const uint16_t *)gf_get_data(gf, name.c_str());
    std::vector<float> d(C);
    for (int i = 0; i < C; i++)
        d[i] = 1.0f / expf(ggml_bf16_to_fp32(*(const ggml_bf16_t *)&raw[i]));
    ggml_backend_tensor_set(dst, d.data(), 0, C * sizeof(float));
}

// Load bf16 bias [C] -> f32
static void vae_load_bias(struct ggml_tensor * dst, const GGUFModel & gf, const std::string & name) {
    struct ggml_tensor * mt = ggml_get_tensor(gf.meta, name.c_str());
    int C = (int)mt->ne[0];  // 1D: ne[0] = C
    const uint16_t * raw = (const uint16_t *)gf_get_data(gf, name.c_str());
    std::vector<float> d(C);
    for (int i = 0; i < C; i++)
        d[i] = ggml_bf16_to_fp32(*(const ggml_bf16_t *)&raw[i]);
    ggml_backend_tensor_set(dst, d.data(), 0, C * sizeof(float));
}

// Load model
static void vae_ggml_load(VAEGGML * m, const char * path) {
    GGUFModel gf = {};
    if (!gf_load(&gf, path)) {
        fprintf(stderr, "[VAE] FATAL: cannot load %s\n", path);
        exit(1);
    }

    static const int strides[]   = {10, 6, 4, 4, 2};
    static const int in_ch[]     = {2048, 1024, 512, 256, 128};
    static const int out_ch[]    = {1024, 512, 256, 128, 128};
    static const int dilations[] = {1, 3, 9};

    // Phase 1: create tensor metadata (no_alloc context)
    size_t ctx_size = ggml_tensor_overhead() * 200;
    struct ggml_init_params p = { ctx_size, NULL, true };
    m->weight_ctx = ggml_init(p);
    struct ggml_context * ctx = m->weight_ctx;

    m->c1w = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 7, 64, 2048);
    m->c1b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2048);

    for (int i = 0; i < 5; i++) {
        VAEBlock & b = m->blk[i];
        b.in_ch  = in_ch[i];
        b.out_ch = out_ch[i];
        b.stride = strides[i];
        b.kernel = strides[i] * 2;
        int C    = out_ch[i];
        b.sa  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, in_ch[i]);
        b.sb  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, in_ch[i]);
        b.ctw = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, b.kernel, out_ch[i], in_ch[i]);
        b.ctb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_ch[i]);
        for (int r = 0; r < 3; r++) {
            VAEResUnit & ru = b.ru[r];
            ru.dilation = dilations[r];
            ru.s1a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, C);
            ru.s1b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, C);
            ru.c1w = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 7, C, C);
            ru.c1b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
            ru.s2a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, C);
            ru.s2b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, C);
            ru.c2w = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 1, C, C);
            ru.c2b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
        }
    }
    m->sa  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 128);
    m->sb  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 128);
    m->c2w = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 7, 128, 2);

    // Phase 2: allocate backend buffer on GPU (im2col grid Y fix enables long-sequence conv1d)
    BackendPair bp = backend_init("VAE");
    m->backend = bp.backend;
    m->cpu_backend = bp.cpu_backend;
    m->sched = backend_sched_new(bp, 8192);
    m->buf = ggml_backend_alloc_ctx_tensors(ctx, m->backend);
    fprintf(stderr, "[VAE] Backend: %s, Weight buffer: %.1f MB\n",
            ggml_backend_name(m->backend),
            (float)ggml_backend_buffer_get_size(m->buf) / (1024 * 1024));

    // Phase 3: load & fuse weights
    vae_fuse_wn(m->c1w, gf, "decoder.conv1");
    vae_load_bias(m->c1b, gf, "decoder.conv1.bias");

    for (int i = 0; i < 5; i++) {
        VAEBlock & b = m->blk[i];
        std::string blk_pfx = "decoder.block." + std::to_string(i);
        vae_load_snake(b.sa, gf, blk_pfx + ".snake1.alpha");
        vae_load_snake_inv(b.sb, gf, blk_pfx + ".snake1.beta");
        vae_fuse_wn(b.ctw, gf, blk_pfx + ".conv_t1");
        vae_load_bias(b.ctb, gf, blk_pfx + ".conv_t1.bias");
        for (int r = 0; r < 3; r++) {
            VAEResUnit & ru = b.ru[r];
            std::string rp = blk_pfx + ".res_unit" + std::to_string(r + 1);
            vae_load_snake(ru.s1a, gf, rp + ".snake1.alpha");
            vae_load_snake_inv(ru.s1b, gf, rp + ".snake1.beta");
            vae_fuse_wn(ru.c1w, gf, rp + ".conv1");
            vae_load_bias(ru.c1b, gf, rp + ".conv1.bias");
            vae_load_snake(ru.s2a, gf, rp + ".snake2.alpha");
            vae_load_snake_inv(ru.s2b, gf, rp + ".snake2.beta");
            vae_fuse_wn(ru.c2w, gf, rp + ".conv2");
            vae_load_bias(ru.c2b, gf, rp + ".conv2.bias");
        }
    }
    vae_load_snake(m->sa, gf, "decoder.snake1.alpha");
    vae_load_snake_inv(m->sb, gf, "decoder.snake1.beta");
    vae_fuse_wn(m->c2w, gf, "decoder.conv2");

    fprintf(stderr, "[VAE] Loaded: 5 blocks, upsample=1920x\n");
    gf_close(&gf);
}

// Graph building
// Snake activation: x + sin^2(exp_a * x) * inv_b
// x: [T, C], exp_a: [1, C], inv_b: [1, C] (pre-computed 1/exp(b))
static struct ggml_tensor * vae_snake(
        struct ggml_context * ctx,
        struct ggml_tensor * x,
        struct ggml_tensor * exp_a,
        struct ggml_tensor * inv_b) {
    struct ggml_tensor * ax = ggml_mul(ctx, x, exp_a);      // [T, C] (broadcast 1->T)
    struct ggml_tensor * s  = ggml_sin(ctx, ax);             // sin(e^a * x)
    struct ggml_tensor * s2 = ggml_sqr(ctx, s);              // sin^2
    struct ggml_tensor * d  = ggml_mul(ctx, s2, inv_b);      // * 1/e^b
    return ggml_add(ctx, x, d);                              // x + ...
}

// Conv1d + bias: data [T, IC] -> [T_out, OC]
static struct ggml_tensor * vae_conv1d(
        struct ggml_context * ctx,
        struct ggml_tensor * w,    // [K, IC, OC] (F16, pre-cast at load)
        struct ggml_tensor * b,    // [OC] or NULL
        struct ggml_tensor * x,    // [T, IC]
        int stride, int padding, int dilation) {
    struct ggml_tensor * y = ggml_conv_1d(ctx, w, x, stride, padding, dilation);
    // ggml_conv_1d returns [OL, OC, N=1], squeeze to 2d
    y = ggml_reshape_2d(ctx, y, y->ne[0], y->ne[1]);
    if (b) {
        // bias [OC] -> [1, OC] for broadcast over OL dimension
        struct ggml_tensor * b2d = ggml_reshape_2d(ctx, b, 1, b->ne[0]);
        y = ggml_add(ctx, y, b2d);
    }
    return y;
}

// ConvTranspose1d + bias
static struct ggml_tensor * vae_conv_t1d(
        struct ggml_context * ctx,
        struct ggml_tensor * w,    // [K, out_ch, in_ch]
        struct ggml_tensor * b,    // [out_ch] or NULL
        struct ggml_tensor * x,    // [T, in_ch]
        int stride, int padding) {
    // ggml_conv_transpose_1d asserts p0==0, so we crop manually
    struct ggml_tensor * y = ggml_conv_transpose_1d(ctx, w, x, stride, 0, 1);
    // y is 4d: [OL, OC, 1, 1], squeeze to 2d
    y = ggml_reshape_2d(ctx, y, y->ne[0], y->ne[1]);
    // Crop padding from both sides: [OL, OC] -> [OL - 2*pad, OC]
    if (padding > 0) {
        y = ggml_view_2d(ctx, y, y->ne[0] - 2 * padding, y->ne[1],
                         y->nb[1], padding * sizeof(float));
    }
    if (b) {
        struct ggml_tensor * b2d = ggml_reshape_2d(ctx, b, 1, b->ne[0]);
        y = ggml_add(ctx, y, b2d);
    }
    return y;
}

// ResUnit forward
static struct ggml_tensor * vae_res_unit(
        struct ggml_context * ctx,
        VAEResUnit * ru,
        struct ggml_tensor * x) {   // [T, C]
    struct ggml_tensor * skip = x;

    // snake1 -> dilated conv(k=7) -> snake2 -> conv(k=1)
    int pad = 3 * ru->dilation;  // (k-1)*dil/2 = 3*dil
    x = vae_snake(ctx, x, ru->s1a, ru->s1b);
    x = vae_conv1d(ctx, ru->c1w, ru->c1b, x, 1, pad, ru->dilation);
    x = vae_snake(ctx, x, ru->s2a, ru->s2b);
    x = vae_conv1d(ctx, ru->c2w, ru->c2b, x, 1, 0, 1);

    return ggml_add(ctx, skip, x);
}

// Build full VAE decode graph
// latent: [T_latent, 64] -> audio: [T_audio, 2]
static struct ggml_tensor * vae_ggml_build_graph(
        struct ggml_context * ctx,
        VAEGGML * m,
        struct ggml_tensor * latent) {   // [T, 64] input

    // conv1: [T, 64] -> [T, 2048]
    struct ggml_tensor * x = vae_conv1d(ctx, m->c1w, m->c1b, latent, 1, 3, 1);

    // 5 decoder blocks
    for (int i = 0; i < 5; i++) {
        VAEBlock & b = m->blk[i];
        // snake -> conv_transpose (upsample)
        x = vae_snake(ctx, x, b.sa, b.sb);
        int pad = (b.kernel - b.stride) / 2;
        x = vae_conv_t1d(ctx, b.ctw, b.ctb, x, b.stride, pad);
        // 3 res units
        for (int r = 0; r < 3; r++)
            x = vae_res_unit(ctx, &b.ru[r], x);
    }

    // Final: snake -> conv2(128->2, k=7, pad=3)
    x = vae_snake(ctx, x, m->sa, m->sb);
    x = vae_conv1d(ctx, m->c2w, NULL, x, 1, 3, 1);

    return x;  // [T_audio, 2]
}

// Decode API
// Decode latent [T_latent, out_ch] -> audio [2, T_audio]
// Input layout matches DiT output: [T, Oc] time-major (Oc contiguous per frame).
// Returns T_audio (or -1 on error).
static int vae_ggml_decode(
        VAEGGML * m,
        const float * latent,   // [T_latent, 64] flat time-major (DiT output layout)
        int T_latent,
        float * audio_out,      // [2, T_audio] flat (caller allocs T_latent*1920*2 floats)
        int max_T_audio) {

    int T_audio = T_latent * 1920;
    if (T_audio > max_T_audio) {
        fprintf(stderr, "[VAE] T_audio %d exceeds max %d\n", T_audio, max_T_audio);
        return -1;
    }

    // Build graph context (large: VAE has many ops with big intermediate tensors)
    size_t ctx_size = ggml_tensor_overhead() * 1024 + ggml_graph_overhead_custom(8192, false);
    std::vector<uint8_t> ctx_buf(ctx_size);
    struct ggml_init_params p = { ctx_size, ctx_buf.data(), true };
    struct ggml_context * ctx = ggml_init(p);

    // Input: [T_latent, 64] for ggml conv_1d (ne[0]=T_latent contiguous)
    struct ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_latent, 64);
    ggml_set_name(input, "vae_input");
    ggml_set_input(input);

    // Build graph
    struct ggml_tensor * output = vae_ggml_build_graph(ctx, m, input);
    ggml_set_name(output, "vae_output");
    ggml_set_output(output);

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(gf, output);
    fprintf(stderr, "[VAE] Graph: %d nodes, output ne=[%lld, %lld]\n",
            ggml_graph_n_nodes(gf), (long long)output->ne[0], (long long)output->ne[1]);

    // Allocate compute graph on persistent scheduler
    if (!ggml_backend_sched_alloc_graph(m->sched, gf)) {
        fprintf(stderr, "[VAE] FATAL: graph alloc failed\n");
        ggml_free(ctx);
        return -1;
    }

    // Set input: transpose [T, 64] time-major -> ggml ne=[T, 64] channel-major
    // DiT flat: frame t at offset t*64, element (t,c) = latent[t*64 + c]
    // ggml [T, 64]: ne[0]=T contiguous, element (t,c) = data[c*T + t]
    std::vector<float> transposed(64 * T_latent);
    for (int t = 0; t < T_latent; t++)
        for (int c = 0; c < 64; c++)
            transposed[c * T_latent + t] = latent[t * 64 + c];
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "vae_input"),
                            transposed.data(), 0, transposed.size() * sizeof(float));

    // Compute
    ggml_backend_sched_graph_compute(m->sched, gf);

    // Read output: ggml [T_audio, 2] ne[0]=T_audio -> element (t,c) = data[c*T_audio + t]
    // Target: [2, T_audio] flat = channel 0 then channel 1
    int T_out = (int)output->ne[0];
    std::vector<float> raw(T_out * 2);
    ggml_backend_tensor_get(output, raw.data(), 0, raw.size() * sizeof(float));
    // raw is already [c=0: T_out floats, c=1: T_out floats] due to ggml layout
    memcpy(audio_out, raw.data(), T_out * 2 * sizeof(float));

    fprintf(stderr, "[VAE] Decoded: T_latent=%d -> T_audio=%d (%.2fs @ 48kHz)\n",
            T_latent, T_out, (float)T_out / 48000.0f);

    ggml_backend_sched_reset(m->sched);
    ggml_free(ctx);
    return T_out;
}

// Tiled decode: overlap-discard chunking for bounded VRAM usage.
// Matches Python handler.py tiled_decode / _tiled_decode_gpu:
//   stride = chunk_size - 2*overlap
//   For each tile: decode latent window with overlap context, trim to core, concatenate.
// Default chunk=256, overlap=64 matches Python handler.py fallback defaults.
// Python auto-tunes chunk by VRAM: >=24GB->512, >=16GB->384, >=12GB->256, <12GB->128.
// Returns T_audio (total samples per channel) or -1 on error.
static int vae_ggml_decode_tiled(
        VAEGGML * m,
        const float * latent,       // [T_latent, 64] flat time-major (DiT output layout)
        int T_latent,
        float * audio_out,          // [2, T_audio] flat (caller allocs)
        int max_T_audio,
        int chunk_size = 256,
        int overlap = 64) {

    // Ensure positive stride (matches Python effective_overlap reduction)
    while (chunk_size - 2 * overlap <= 0 && overlap > 0)
        overlap /= 2;

    // Short sequence: decode directly
    if (T_latent <= chunk_size)
        return vae_ggml_decode(m, latent, T_latent, audio_out, max_T_audio);

    int stride = chunk_size - 2 * overlap;
    int num_steps = (T_latent + stride - 1) / stride;

    fprintf(stderr, "[VAE] Tiled decode: %d tiles (chunk=%d, overlap=%d, stride=%d)\n",
            num_steps, chunk_size, overlap, stride);

    // Temp buffer for one tile output (max tile = chunk_size + 2*overlap latent frames)
    int max_win = chunk_size + 2 * overlap;
    int max_tile_audio = max_win * 1920 + 1920;  // margin for conv padding
    std::vector<float> tile_audio(2 * max_tile_audio);
    // Temp latent slice [win_len, 64] time-major
    std::vector<float> tile_latent(64 * max_win);

    float upsample_factor = 0.0f;
    int audio_write_pos = 0;

    for (int i = 0; i < num_steps; i++) {
        // Core range in latent frames (the part we keep)
        int core_start = i * stride;
        int core_end = core_start + stride;
        if (core_end > T_latent) core_end = T_latent;

        // Window range with overlap context
        int win_start = core_start - overlap;
        if (win_start < 0) win_start = 0;
        int win_end = core_end + overlap;
        if (win_end > T_latent) win_end = T_latent;
        int win_len = win_end - win_start;

        // Extract latent window: [T_latent, 64] -> [win_len, 64] time-major
        // Element (t, c) in full = latent[t * 64 + c]
        for (int t = 0; t < win_len; t++)
            for (int c = 0; c < 64; c++)
                tile_latent[t * 64 + c] = latent[(win_start + t) * 64 + c];

        // Decode tile
        int tile_T = vae_ggml_decode(m, tile_latent.data(), win_len,
                                     tile_audio.data(), max_tile_audio);
        if (tile_T < 0) {
            fprintf(stderr, "[VAE] FATAL: tile %d decode failed\n", i);
            return -1;
        }

        // Determine upsample factor from first tile
        if (i == 0) {
            upsample_factor = (float)tile_T / (float)win_len;
            fprintf(stderr, "[VAE] Upsample factor: %.2f (expected ~1920)\n", upsample_factor);
        }

        // Compute trim in audio samples (matches Python int(round(...)))
        int added_start = core_start - win_start;
        int trim_start = (int)roundf((float)added_start * upsample_factor);
        int added_end = win_end - core_end;
        int trim_end = (int)roundf((float)added_end * upsample_factor);

        int end_idx = (trim_end > 0) ? (tile_T - trim_end) : tile_T;
        int core_len = end_idx - trim_start;
        if (core_len <= 0) continue;

        // Check output bounds
        if (audio_write_pos + core_len > max_T_audio) {
            fprintf(stderr, "[VAE] FATAL: tiled output exceeds max_T_audio\n");
            return -1;
        }

        // Append core audio: tile_audio is [2, tile_T] = ch0[0..tile_T-1] ch1[tile_T..2*tile_T-1]
        memcpy(audio_out + audio_write_pos,
               tile_audio.data() + trim_start,
               core_len * sizeof(float));
        memcpy(audio_out + max_T_audio + audio_write_pos,
               tile_audio.data() + tile_T + trim_start,
               core_len * sizeof(float));
        audio_write_pos += core_len;
    }

    // Compact ch1 from offset max_T_audio to offset audio_write_pos
    memmove(audio_out + audio_write_pos,
            audio_out + max_T_audio,
            audio_write_pos * sizeof(float));

    fprintf(stderr, "[VAE] Tiled decode done: %d tiles -> T_audio=%d (%.2fs @ 48kHz)\n",
            num_steps, audio_write_pos, (float)audio_write_pos / 48000.0f);

    return audio_write_pos;
}

// Free
static void vae_ggml_free(VAEGGML * m) {
    if (m->sched) ggml_backend_sched_free(m->sched);
    if (m->buf) ggml_backend_buffer_free(m->buf);
    if (m->weight_ctx) ggml_free(m->weight_ctx);
    if (m->backend && m->backend != m->cpu_backend) ggml_backend_free(m->backend);
    if (m->cpu_backend) ggml_backend_free(m->cpu_backend);
    *m = {};
}
