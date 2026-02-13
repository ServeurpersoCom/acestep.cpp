#pragma once
// VAE Oobleck Decoder: latents [64, T] -> audio [2, T*1920] at 48kHz
// Conv1d + ConvTranspose1d + Snake activation + weight norm fusion
//
// Requires: safetensors.h included before this file
// Standalone from transformer stack (no GEMM, pure kernels)

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <cuda_bf16.h>

using bf16 = __nv_bfloat16;

// Weight norm fusion: w = g * v / ||v||  (per output channel)
// g: [out_ch, 1, 1], v: [out_ch, in_ch, k]
// Output: fused [out_ch, in_ch, k] on GPU (bf16)
static bf16 *fuse_weight_norm(const bf16 *g_gpu, const bf16 *v_gpu,
                               int out_ch, int in_ch, int k) {
    int fan = in_ch * k;
    // Download to CPU for fusion
    std::vector<uint16_t> g_cpu(out_ch), v_cpu(out_ch * fan);
    cudaMemcpy(g_cpu.data(), g_gpu, out_ch * sizeof(bf16), cudaMemcpyDeviceToHost);
    cudaMemcpy(v_cpu.data(), v_gpu, (size_t)out_ch * fan * sizeof(bf16), cudaMemcpyDeviceToHost);

    std::vector<uint16_t> w_cpu(out_ch * fan);
    for (int o = 0; o < out_ch; o++) {
        float gv = __bfloat162float(*(bf16 *)&g_cpu[o]);
        // Compute ||v[o]||
        float norm_sq = 0.0f;
        for (int i = 0; i < fan; i++) {
            float vv = __bfloat162float(*(bf16 *)&v_cpu[o * fan + i]);
            norm_sq += vv * vv;
        }
        float scale = gv / (sqrtf(norm_sq) + 1e-12f);
        for (int i = 0; i < fan; i++) {
            float vv = __bfloat162float(*(bf16 *)&v_cpu[o * fan + i]);
            *(bf16 *)&w_cpu[o * fan + i] = __float2bfloat16(vv * scale);
        }
    }

    bf16 *w_gpu;
    cudaMalloc(&w_gpu, (size_t)out_ch * fan * sizeof(bf16));
    cudaMemcpy(w_gpu, w_cpu.data(), (size_t)out_ch * fan * sizeof(bf16), cudaMemcpyDefault);
    return w_gpu;
}

// Snake activation: x + (1/exp(beta)) * sin(exp(alpha)*x)^2
// Data format: [C, T], alpha/beta: [C] (squeezed from [1,C,1])
// In-place on x
__global__ void snake_kernel(bf16 *x, const bf16 *alpha, const bf16 *beta,
                              int C, int T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * T) return;
    int c = idx / T;

    float a = expf(__bfloat162float(alpha[c]));
    float b = expf(__bfloat162float(beta[c]));
    float xv = __bfloat162float(x[idx]);
    float s = sinf(a * xv);
    xv = xv + s * s / (b + 1e-9f);
    x[idx] = __float2bfloat16(xv);
}

static void snake_act(bf16 *x, const bf16 *alpha, const bf16 *beta, int C, int T) {
    int n = C * T;
    snake_kernel<<<(n + 255) / 256, 256>>>(x, alpha, beta, C, T);
}

// Conv1d: y[o, t] = bias[o] + sum_c sum_k w[o, c, k] * x[c, t*stride + k*dilation - pad]
// x: [in_ch, T_in], w: [out_ch, in_ch, kernel], bias: [out_ch] or NULL
// y: [out_ch, T_out]
__global__ void conv1d_kernel(bf16 *y, const bf16 *x, const bf16 *w, const bf16 *bias,
                               int in_ch, int out_ch, int kernel, int T_in, int T_out,
                               int stride, int dilation, int padding) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_ch * T_out) return;
    int o = idx / T_out;
    int t = idx % T_out;

    float acc = (bias != nullptr) ? __bfloat162float(bias[o]) : 0.0f;
    const bf16 *wo = w + (int64_t)o * in_ch * kernel;

    for (int c = 0; c < in_ch; c++) {
        for (int k = 0; k < kernel; k++) {
            int pos = t * stride + k * dilation - padding;
            if (pos >= 0 && pos < T_in) {
                acc += __bfloat162float(wo[c * kernel + k]) *
                       __bfloat162float(x[c * T_in + pos]);
            }
        }
    }
    y[idx] = __float2bfloat16(acc);
}

static void conv1d(bf16 *y, const bf16 *x, const bf16 *w, const bf16 *bias,
                   int in_ch, int out_ch, int kernel, int T_in,
                   int stride, int dilation, int padding) {
    int T_out = (T_in + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
    int n = out_ch * T_out;
    conv1d_kernel<<<(n + 255) / 256, 256>>>(
        y, x, w, bias, in_ch, out_ch, kernel, T_in, T_out, stride, dilation, padding);
}

static int conv1d_outlen(int T_in, int kernel, int stride, int dilation, int padding) {
    return (T_in + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
}

// ConvTranspose1d: scatter-accumulate
// x: [in_ch, T_in], w: [in_ch, out_ch, kernel]
// y: [out_ch, T_out] where T_out = (T_in-1)*stride + kernel - 2*padding
__global__ void conv_transpose1d_kernel(bf16 *y, const bf16 *x, const bf16 *w, const bf16 *bias,
                                         int in_ch, int out_ch, int kernel, int T_in, int T_out,
                                         int stride, int padding) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_ch * T_out) return;
    int o = idx / T_out;
    int t_out = idx % T_out;

    float acc = (bias != nullptr) ? __bfloat162float(bias[o]) : 0.0f;

    for (int c = 0; c < in_ch; c++) {
        for (int k = 0; k < kernel; k++) {
            int t_adj = t_out + padding - k;
            if (t_adj >= 0 && t_adj % stride == 0) {
                int t_in = t_adj / stride;
                if (t_in >= 0 && t_in < T_in) {
                    // weight layout: [in_ch, out_ch, kernel]
                    acc += __bfloat162float(x[c * T_in + t_in]) *
                           __bfloat162float(w[(int64_t)c * out_ch * kernel + o * kernel + k]);
                }
            }
        }
    }
    y[idx] = __float2bfloat16(acc);
}

static int conv_t1d_outlen(int T_in, int kernel, int stride, int padding) {
    return (T_in - 1) * stride + kernel - 2 * padding;
}

static void conv_transpose1d(bf16 *y, const bf16 *x, const bf16 *w, const bf16 *bias,
                              int in_ch, int out_ch, int kernel, int T_in,
                              int stride, int padding) {
    int T_out = conv_t1d_outlen(T_in, kernel, stride, padding);
    int n = out_ch * T_out;
    conv_transpose1d_kernel<<<(n + 255) / 256, 256>>>(
        y, x, w, bias, in_ch, out_ch, kernel, T_in, T_out, stride, padding);
}

// ResidualUnit: snake1 -> conv1(k=7, dilated) -> snake2 -> conv2(k=1) + skip
struct VAEResUnit {
    bf16 *snake1_a, *snake1_b;  // [ch]
    bf16 *conv1_w;              // [ch, ch, 7] (fused weight norm)
    bf16 *conv1_b;              // [ch]
    bf16 *snake2_a, *snake2_b;  // [ch]
    bf16 *conv2_w;              // [ch, ch, 1] (fused weight norm)
    bf16 *conv2_b;              // [ch]
    int dilation;               // 1, 3, or 9
};

// Decoder block: snake -> conv_transpose -> 3 ResUnits
struct VAEDecoderBlock {
    bf16 *snake_a, *snake_b;    // [in_ch]
    bf16 *conv_t_w;             // [in_ch, out_ch, 2*stride] (fused)
    bf16 *conv_t_b;             // [out_ch]
    int in_ch, out_ch, stride;
    VAEResUnit res[3];          // dilation 1, 3, 9
};

// Full VAE Decoder
struct VAEDecoder {
    // Initial conv: [64, T] -> [2048, T]
    bf16 *conv1_w;              // [2048, 64, 7] (fused)
    bf16 *conv1_b;              // [2048]

    // 5 decoder blocks
    VAEDecoderBlock blocks[5];

    // Final: snake -> conv2
    bf16 *snake_a, *snake_b;    // [128]
    bf16 *conv2_w;              // [2, 128, 7] (fused)
    // conv2 has no bias

    // Ping-pong buffers
    bf16 *buf_a, *buf_b;
    size_t buf_size;            // bytes per buffer
};

// Load helpers (self-contained, no dependency on dit.cuh)
static bf16 *vae_upload(SafeTensors &st, const std::string &name) {
    auto it = st.tensors.find(name);
    if (it == st.tensors.end()) {
        fprintf(stderr, "FATAL: VAE tensor '%s' not found\n", name.c_str());
        exit(1);
    }
    bf16 *d;
    cudaError_t e = cudaMalloc(&d, it->second.nbytes);
    if (e != cudaSuccess) {
        cudaGetLastError();  // clear stale error
        e = cudaMallocManaged(&d, it->second.nbytes);
        if (e != cudaSuccess) {
            fprintf(stderr, "FATAL: alloc failed for '%s'\n", name.c_str());
            exit(1);
        }
    }
    cudaMemcpy(d, it->second.data, it->second.nbytes, cudaMemcpyDefault);
    return d;
}

static void load_res_unit(VAEResUnit &ru, SafeTensors &st, const std::string &prefix,
                          int ch, int dilation) {
    ru.dilation = dilation;
    ru.snake1_a = vae_upload(st, prefix + ".snake1.alpha");
    ru.snake1_b = vae_upload(st, prefix + ".snake1.beta");
    ru.snake2_a = vae_upload(st, prefix + ".snake2.alpha");
    ru.snake2_b = vae_upload(st, prefix + ".snake2.beta");
    ru.conv1_b  = vae_upload(st, prefix + ".conv1.bias");
    ru.conv2_b  = vae_upload(st, prefix + ".conv2.bias");

    // Fuse weight norm
    bf16 *g1 = vae_upload(st, prefix + ".conv1.weight_g");
    bf16 *v1 = vae_upload(st, prefix + ".conv1.weight_v");
    ru.conv1_w = fuse_weight_norm(g1, v1, ch, ch, 7);
    cudaFree(g1); cudaFree(v1);

    bf16 *g2 = vae_upload(st, prefix + ".conv2.weight_g");
    bf16 *v2 = vae_upload(st, prefix + ".conv2.weight_v");
    ru.conv2_w = fuse_weight_norm(g2, v2, ch, ch, 1);
    cudaFree(g2); cudaFree(v2);
}

static void load_decoder_block(VAEDecoderBlock &blk, SafeTensors &st, const std::string &prefix,
                                int in_ch, int out_ch, int stride) {
    blk.in_ch = in_ch;
    blk.out_ch = out_ch;
    blk.stride = stride;

    blk.snake_a = vae_upload(st, prefix + ".snake1.alpha");
    blk.snake_b = vae_upload(st, prefix + ".snake1.beta");
    blk.conv_t_b = vae_upload(st, prefix + ".conv_t1.bias");

    // Fuse conv_transpose weight norm
    bf16 *g = vae_upload(st, prefix + ".conv_t1.weight_g");
    bf16 *v = vae_upload(st, prefix + ".conv_t1.weight_v");
    blk.conv_t_w = fuse_weight_norm(g, v, in_ch, out_ch, 2 * stride);
    cudaFree(g); cudaFree(v);

    // 3 ResUnits with dilations 1, 3, 9
    int dilations[3] = {1, 3, 9};
    for (int i = 0; i < 3; i++) {
        char name[32]; snprintf(name, sizeof(name), ".res_unit%d", i + 1);
        load_res_unit(blk.res[i], st, prefix + name, out_ch, dilations[i]);
    }
}

// Load full VAE decoder
static void load_vae_decoder(VAEDecoder *vae, const char *model_dir, int max_T_latent) {
    SafeTensors st;
    if (!safe_load(st, model_dir)) {
        fprintf(stderr, "FATAL: cannot load VAE from %s\n", model_dir);
        exit(1);
    }
    fprintf(stderr, "[VAE] Loading from %s (%zu tensors)...\n", model_dir, st.tensors.size());

    // conv1: Conv1d(64, 2048, k=7, pad=3)
    vae->conv1_b = vae_upload(st, "decoder.conv1.bias");
    {
        bf16 *g = vae_upload(st, "decoder.conv1.weight_g");
        bf16 *v = vae_upload(st, "decoder.conv1.weight_v");
        vae->conv1_w = fuse_weight_norm(g, v, 2048, 64, 7);
        cudaFree(g); cudaFree(v);
    }

    // 5 blocks: channel_multiples reversed = [16,8,4,2,1]*128
    // strides reversed = [10, 6, 4, 4, 2]
    int channels[] = {2048, 1024, 512, 256, 128};
    int strides[] = {10, 6, 4, 4, 2};
    int out_channels[] = {1024, 512, 256, 128, 128};
    for (int i = 0; i < 5; i++) {
        char prefix[64]; snprintf(prefix, sizeof(prefix), "decoder.block.%d", i);
        load_decoder_block(vae->blocks[i], st, prefix, channels[i], out_channels[i], strides[i]);
    }

    // Final snake + conv2
    vae->snake_a = vae_upload(st, "decoder.snake1.alpha");
    vae->snake_b = vae_upload(st, "decoder.snake1.beta");
    {
        bf16 *g = vae_upload(st, "decoder.conv2.weight_g");
        bf16 *v = vae_upload(st, "decoder.conv2.weight_v");
        vae->conv2_w = fuse_weight_norm(g, v, 2, 128, 7);
        cudaFree(g); cudaFree(v);
    }

    // Allocate ping-pong buffers (max intermediate size)
    // After conv1: [2048, T], after block0: [1024, T*10], ...
    // Max is block4 output: [128, T*1920] but we need max of in/out per step
    // Largest: [2048, max_T_latent] or [128, max_T_latent*1920]
    size_t max_elems = 0;
    int T = max_T_latent;
    // conv1 output
    max_elems = (size_t)2048 * T;
    // After each block
    for (int i = 0; i < 5; i++) {
        T *= strides[i];
        size_t e = (size_t)out_channels[i] * T;
        if (e > max_elems) max_elems = e;
    }
    // Final output: [2, T]
    size_t e = (size_t)2 * T;
    if (e > max_elems) max_elems = e;

    // vae_res_forward needs scratch = 3*ch*T per res unit
    // Allocate 3x max_elems so either buffer can serve as scratch
    vae->buf_size = 3 * max_elems * sizeof(bf16);
    cudaMalloc(&vae->buf_a, vae->buf_size);
    cudaMalloc(&vae->buf_b, vae->buf_size);

    fprintf(stderr, "[VAE] Loaded. 5 blocks, upsample=1920x, buf=%.1fMB each (3x for scratch)\n",
            (float)vae->buf_size / 1e6f);
}

// Residual add kernel with crop: out[c, t] = in_skip[c, t+crop] + in_res[c, t]
__global__ void residual_add_crop_kernel(bf16 *out, const bf16 *skip, const bf16 *res,
                                          int C, int T_out, int T_skip, int crop) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * T_out) return;
    int c = idx / T_out;
    int t = idx % T_out;
    float s = __bfloat162float(skip[c * T_skip + t + crop]);
    float r = __bfloat162float(res[c * T_out + t]);
    out[idx] = __float2bfloat16(s + r);
}

// Rewrite res_unit forward properly
static int vae_res_forward(VAEResUnit &ru, bf16 *data, bf16 *scratch, int ch, int T) {
    // data [ch, T] = input (will be overwritten with output)
    // scratch = large enough for intermediates

    int pad1 = ((7 - 1) * ru.dilation) / 2;
    int T_after_conv1 = conv1d_outlen(T, 7, 1, ru.dilation, pad1);
    int T_after_conv2 = T_after_conv1;  // k=1, no change
    int T_out = T_after_conv2;

    // scratch layout: [0: ch*T for snake copy] [ch*T: ch*T_after_conv1 for conv1 out]
    bf16 *s1_buf = scratch;                                  // [ch, T]
    bf16 *c1_buf = scratch + (int64_t)ch * T;                // [ch, T_after_conv1]
    bf16 *c2_buf = scratch + (int64_t)ch * T + (int64_t)ch * T_after_conv1;  // [ch, T_out]

    // snake1(data) -> s1_buf
    cudaMemcpyAsync(s1_buf, data, (size_t)ch * T * sizeof(bf16), cudaMemcpyDeviceToDevice);
    snake_act(s1_buf, ru.snake1_a, ru.snake1_b, ch, T);

    // conv1(s1_buf) -> c1_buf
    conv1d(c1_buf, s1_buf, ru.conv1_w, ru.conv1_b, ch, ch, 7, T, 1, ru.dilation, pad1);

    // snake2(c1_buf) -> in-place
    snake_act(c1_buf, ru.snake2_a, ru.snake2_b, ch, T_after_conv1);

    // conv2(c1_buf) -> c2_buf
    conv1d(c2_buf, c1_buf, ru.conv2_w, ru.conv2_b, ch, ch, 1, T_after_conv1, 1, 1, 0);

    // Residual: data = data[crop:] + c2_buf
    int crop = (T - T_out) / 2;
    int n = ch * T_out;
    residual_add_crop_kernel<<<(n + 255) / 256, 256>>>(data, data, c2_buf, ch, T_out, T, crop);

    return T_out;
}

// Forward decoder block: snake -> conv_t -> 3 ResUnits
static int vae_block_forward(VAEDecoderBlock &blk, bf16 *in, bf16 *out, bf16 *scratch,
                              int T_in) {
    int in_ch = blk.in_ch;
    int out_ch = blk.out_ch;
    int s = blk.stride;
    int pad = (s + 1) / 2;  // ceil(stride/2)

    // snake(in) -> in-place
    snake_act(in, blk.snake_a, blk.snake_b, in_ch, T_in);

    // conv_transpose(in) -> out
    int T_out = conv_t1d_outlen(T_in, 2 * s, s, pad);
    conv_transpose1d(out, in, blk.conv_t_w, blk.conv_t_b, in_ch, out_ch, 2 * s, T_in, s, pad);

    // 3 ResUnits (in-place on out)
    int T = T_out;
    for (int i = 0; i < 3; i++) {
        T = vae_res_forward(blk.res[i], out, scratch, out_ch, T);
    }
    return T;
}

// Full VAE decode: latents [64, T_latent] -> audio [2, T_audio]
// T_audio = T_latent * 1920
// output should be pre-allocated: [2, T_latent * 1920]
static int vae_decode(VAEDecoder *vae, bf16 *latents, int T_latent, bf16 *output) {
    bf16 *cur = vae->buf_a;
    bf16 *nxt = vae->buf_b;

    // conv1: [64, T] -> [2048, T]  (k=7, pad=3)
    int T = conv1d_outlen(T_latent, 7, 1, 1, 3);
    conv1d(cur, latents, vae->conv1_w, vae->conv1_b, 64, 2048, 7, T_latent, 1, 1, 3);

    // 5 blocks
    for (int i = 0; i < 5; i++) {
        // Use scratch from the other buffer's unused portion
        T = vae_block_forward(vae->blocks[i], cur, nxt, cur, T);
        // Swap
        bf16 *tmp = cur; cur = nxt; nxt = tmp;
    }
    // cur now has [128, T_audio]

    // Final: snake -> conv2 (128 -> 2, k=7, pad=3)
    snake_act(cur, vae->snake_a, vae->snake_b, 128, T);
    int T_final = conv1d_outlen(T, 7, 1, 1, 3);
    conv1d(output, cur, vae->conv2_w, nullptr, 128, 2, 7, T, 1, 1, 3);

    return T_final;
}

// Tiled VAE decode
// Matches Python handler.py tiled_decode / _tiled_decode_offload_cpu:
//   stride = chunk_size - 2*overlap
//   For each tile: extract latent window with overlap, decode, trim to core, scatter.
// Python defaults: chunk_size=512, overlap=64 (VAE_DECODE_MAX_CHUNK_SIZE).
//
// latents: [64, T_latent]  (channel-first, GPU)
// output:  [2, >=T_latent*1920]  (channel-first, GPU, pre-allocated)
// Returns actual number of audio samples per channel.
static int vae_decode_tiled(VAEDecoder *vae, bf16 *latents, int T_latent, bf16 *output,
                            int chunk_size = 512, int overlap = 64) {
    // Ensure positive stride (matches Python effective_overlap reduction)
    while (chunk_size - 2 * overlap <= 0 && overlap > 0)
        overlap /= 2;

    // Short track: decode directly, no tiling needed
    if (T_latent <= chunk_size)
        return vae_decode(vae, latents, T_latent, output);

    int stride = chunk_size - 2 * overlap;
    int num_steps = (T_latent + stride - 1) / stride;  // ceil(T / stride)

    // Allocate temp buffers for chunk latent slice and chunk audio output
    int max_win = chunk_size + 2 * overlap;
    if (max_win > T_latent) max_win = T_latent;
    bf16 *chunk_lat;  // [64, max_win]
    cudaMalloc(&chunk_lat, (size_t)64 * max_win * sizeof(bf16));
    int max_chunk_audio = max_win * 1920 + 1920;  // generous margin for conv padding
    bf16 *chunk_aud;  // [2, max_chunk_audio]
    cudaMalloc(&chunk_aud, (size_t)2 * max_chunk_audio * sizeof(bf16));

    float upsample_factor = 0.0f;
    int audio_write_pos = 0;

    // Append buffers per channel (avoids needing to know total_audio upfront)
    // Generous alloc: T_latent * 1920 is the theoretical max (exact upsample)
    int max_audio = T_latent * 1920 + 4096;  // small margin for conv padding
    bf16 *tmp_ch0, *tmp_ch1;
    cudaMalloc(&tmp_ch0, (size_t)max_audio * sizeof(bf16));
    cudaMalloc(&tmp_ch1, (size_t)max_audio * sizeof(bf16));

    fprintf(stderr, "[VAE] Tiled decode: %d chunks (chunk=%d, overlap=%d, stride=%d)\n",
            num_steps, chunk_size, overlap, stride);

    for (int i = 0; i < num_steps; i++) {
        // Core range in latent frames
        int core_start = i * stride;
        int core_end   = core_start + stride;
        if (core_end > T_latent) core_end = T_latent;

        // Window range with overlap (matches Python max(0,...) min(T,...))
        int win_start = core_start - overlap;
        if (win_start < 0) win_start = 0;
        int win_end = core_end + overlap;
        if (win_end > T_latent) win_end = T_latent;
        int win_len = win_end - win_start;

        // Extract latent window [64, win_len] from [64, T_latent] via strided copy
        cudaMemcpy2D(
            chunk_lat,                                          // dst
            (size_t)win_len * sizeof(bf16),                     // dst pitch
            latents + win_start,                                // src (offset in time)
            (size_t)T_latent * sizeof(bf16),                    // src pitch
            (size_t)win_len * sizeof(bf16),                     // width (bytes)
            64,                                                 // height (channels)
            cudaMemcpyDeviceToDevice
        );

        // Decode chunk
        int chunk_T_audio = vae_decode(vae, chunk_lat, win_len, chunk_aud);

        // First chunk: determine upsample_factor (matches Python)
        if (i == 0) {
            upsample_factor = (float)chunk_T_audio / (float)win_len;
            fprintf(stderr, "[VAE] Upsample factor: %.2f (expected ~1920)\n", upsample_factor);
        }

        // Compute trim in audio samples (matches Python int(round(...)))
        int added_start = core_start - win_start;
        int trim_start  = (int)roundf(added_start * upsample_factor);
        int added_end   = win_end - core_end;
        int trim_end    = (int)roundf(added_end * upsample_factor);

        int end_idx  = (trim_end > 0) ? (chunk_T_audio - trim_end) : chunk_T_audio;
        int core_len = end_idx - trim_start;
        if (core_len <= 0) continue;

        // Append core audio into per-channel temp buffers (no pitch assumption)
        // chunk_aud layout: [ch=0: 0..chunk_T_audio-1] [ch=1: chunk_T_audio..2*chunk_T_audio-1]
        cudaMemcpyAsync(
            tmp_ch0 + audio_write_pos,
            chunk_aud + trim_start,
            (size_t)core_len * sizeof(bf16),
            cudaMemcpyDeviceToDevice
        );
        cudaMemcpyAsync(
            tmp_ch1 + audio_write_pos,
            chunk_aud + chunk_T_audio + trim_start,
            (size_t)core_len * sizeof(bf16),
            cudaMemcpyDeviceToDevice
        );
        audio_write_pos += core_len;
    }

    // audio_write_pos is now the exact total, copy into output [2, audio_write_pos]
    cudaMemcpyAsync(output,
                    tmp_ch0,
                    (size_t)audio_write_pos * sizeof(bf16),
                    cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(output + audio_write_pos,
                    tmp_ch1,
                    (size_t)audio_write_pos * sizeof(bf16),
                    cudaMemcpyDeviceToDevice);

    cudaDeviceSynchronize();
    cudaFree(chunk_lat);
    cudaFree(chunk_aud);
    cudaFree(tmp_ch0);
    cudaFree(tmp_ch1);

    return audio_write_pos;
}
