// dit.cuh: ACE-Step DiT (Diffusion Transformer) CUDA implementation
// Batch bidirectional GQA attention, AdaLN, cross-attention, flow matching
// Reuses: kernels.cuh (RMSNorm, SwiGLU, QK-Norm), safetensors.h (loader)
#pragma once

#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include "../debug.h"

using bf16 = __nv_bfloat16;

// DiT Config (from acestep-v15-turbo/config.json)
struct DiTConfig {
    int hidden_size;        // 2048
    int intermediate_size;  // 6144
    int n_heads;            // 16
    int n_kv_heads;         // 8
    int head_dim;           // 128
    int n_layers;           // 24
    int in_channels;        // 192  (input channels to proj_in, after context concat)
    int out_channels;       // 64   (audio_acoustic_hidden_dim)
    int patch_size;         // 2
    int sliding_window;     // 128
    float rope_theta;       // 1000000
    float rms_norm_eps;     // 1e-6
    int max_seq_len;        // max T after patching (runtime)
};

static DiTConfig dit_default_config() {
    return {
        .hidden_size       = 2048,
        .intermediate_size = 6144,
        .n_heads           = 16,
        .n_kv_heads        = 8,
        .head_dim          = 128,
        .n_layers          = 24,
        .in_channels       = 192,
        .out_channels      = 64,
        .patch_size         = 2,
        .sliding_window    = 128,
        .rope_theta        = 1000000.0f,
        .rms_norm_eps      = 1e-6f,
        .max_seq_len       = 4096,
    };
}

// Batch linear: Y[S,out] = X[S,in] @ W[out,in]^T  (+ optional bias)
// Row-major layout. Uses cuBLAS with swap trick.
static void linear_batch(bf16 *Y, const bf16 *X, const bf16 *W, const bf16 *bias,
                         int S, int out_dim, int in_dim, cublasHandle_t handle) {
    float alpha = 1.0f, beta = 0.0f;
    // Row-major: C = A @ B  <->  cuBLAS(B^T @ A^T) with swap
    // A=X [S, in_dim], B=W^T [in_dim, out_dim], C=Y [S, out_dim]
    // cuBLAS: m=out_dim, n=S, k=in_dim
    cublasGemmEx(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        out_dim, S, in_dim,
        &alpha,
        W, CUDA_R_16BF, in_dim,       // W [out,in] row-major -> cuBLAS col-major [in,out]
        X, CUDA_R_16BF, in_dim,       // X [S,in] row-major -> cuBLAS col-major [in,S]
        &beta,
        Y, CUDA_R_16BF, out_dim,      // Y [S,out] row-major -> cuBLAS col-major [out,S]
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    // Add bias if present: Y[s,j] += bias[j]
    // Done by a simple kernel below
}

// Batch linear no-transpose: Y[S,out] = X[S,in] @ W[in,out]
// Used for ConvTranspose1d where weight is stored as [in, out]
static void linear_batch_nt(bf16 *Y, const bf16 *X, const bf16 *W,
                            int S, int out_dim, int in_dim, cublasHandle_t handle) {
    float alpha = 1.0f, beta = 0.0f;
    // Row-major Y = X @ W: cuBLAS(W^T_col @ X^T_col)
    // W [in,out] row-major = [out,in] col-major. OP_N -> [out,in], m=out, k=in
    // X [S,in] row-major = [in,S] col-major. OP_N -> [in,S], k=in, n=S
    cublasGemmEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        out_dim, S, in_dim,
        &alpha,
        W, CUDA_R_16BF, out_dim,       // W [in,out] row-major -> [out,in] col-major
        X, CUDA_R_16BF, in_dim,        // X [S,in] row-major -> [in,S] col-major
        &beta,
        Y, CUDA_R_16BF, out_dim,       // Y [S,out] row-major -> [out,S] col-major
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// Batch linear with f32 output (for attention scores)
// Batched GEMM for attention: scores = Q @ K^T  (all bf16, f32 accumulate)
// Q: [B*H, S, D], K: [B*H, S, D]  -> scores: [B*H, S, S]
static void batch_qk_gemm(float *scores, const bf16 *Q, const bf16 *K,
                           int S, int D, int batch, cublasHandle_t handle) {
    float alpha = 1.0f, beta = 0.0f;
    // Row-major: scores[b] = Q[b] @ K[b]^T -> [S,S]
    // cuBLAS col-major: scores_col = K_col @ Q_col^T
    // K row-major [S,D] -> col-major [D,S]. With OP_T -> [S,D], m=S, k=D
    // Q row-major [S,D] -> col-major [D,S]. With OP_N -> [D,S], k=D, n=S
    cublasGemmStridedBatchedEx(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        S, S, D,
        &alpha,
        K, CUDA_R_16BF, D, (int64_t)S * D,      // A = K
        Q, CUDA_R_16BF, D, (int64_t)S * D,      // B = Q
        &beta,
        scores, CUDA_R_32F, S, (int64_t)S * S,  // C = scores
        batch,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// f32 -> bf16 conversion kernel
__global__ void f32_to_bf16_kernel(bf16 *out, const float *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(in[i]);
}

// scores @ V -> output: [B*H, S, S] x [B*H, S, D] -> [B*H, S, D]
// scores in f32, V in bf16, output in bf16
// scratch: pre-allocated bf16 buffer of size >= batch*S*S for scores conversion
static void batch_sv_gemm(bf16 *output, const float *scores, const bf16 *V,
                           int S, int D, int batch, cublasHandle_t handle,
                           bf16 *scratch) {
    int n = batch * S * S;
    f32_to_bf16_kernel<<<(n + 255) / 256, 256>>>(scratch, scores, n);

    float alpha = 1.0f, beta = 0.0f;
    cublasGemmStridedBatchedEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        D, S, S,
        &alpha,
        V, CUDA_R_16BF, D, (int64_t)S * D,
        scratch, CUDA_R_16BF, S, (int64_t)S * S,
        &beta,
        output, CUDA_R_16BF, D, (int64_t)S * D,
        batch,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// CUDA Kernels, DiT specific

// Add bias to batch: Y[s,j] += bias[j], Y is [S, D]
__global__ void add_bias_2d_kernel(bf16 *Y, const bf16 *bias, int S, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= S * D) return;
    int j = idx % D;
    float val = __bfloat162float(Y[idx]) + __bfloat162float(bias[j]);
    Y[idx] = __float2bfloat16(val);
}

static void add_bias_2d(bf16 *Y, const bf16 *bias, int S, int D) {
    int n = S * D;
    add_bias_2d_kernel<<<(n + 255) / 256, 256>>>(Y, bias, S, D);
}

// RMSNorm for 2D: x[S, D], weight[D], out[S, D]
// Each block handles one row (one position)
__global__ void rmsnorm_2d_kernel(bf16 *out, const bf16 *x, const bf16 *w, int D, float eps) {
    int s = blockIdx.x;
    const bf16 *row = x + (int64_t)s * D;
    bf16 *orow = out + (int64_t)s * D;

    // Compute sum of squares
    extern __shared__ float smem[];
    float local_ss = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float v = __bfloat162float(row[d]);
        local_ss += v * v;
    }
    smem[threadIdx.x] = local_ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    float rms = rsqrtf(smem[0] / D + eps);

    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float v = __bfloat162float(row[d]) * rms * __bfloat162float(w[d]);
        orow[d] = __float2bfloat16(v);
    }
}

static void rmsnorm_2d(bf16 *out, const bf16 *x, const bf16 *w, int S, int D, float eps) {
    int threads = (D < 256) ? D : 256;
    rmsnorm_2d_kernel<<<S, threads, threads * sizeof(float)>>>(out, x, w, D, eps);
}

// AdaLN modulate: out[s,d] = norm[s,d] * (1 + scale[d]) + shift[d]
// norm: [S, D], scale: [D], shift: [D], out: [S, D]
__global__ void adaln_modulate_kernel(bf16 *out, const bf16 *norm, const bf16 *scale, const bf16 *shift, int S, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= S * D) return;
    int d = idx % D;
    float n = __bfloat162float(norm[idx]);
    float sc = __bfloat162float(scale[d]);
    float sh = __bfloat162float(shift[d]);
    out[idx] = __float2bfloat16(n * (1.0f + sc) + sh);
}

static void adaln_modulate(bf16 *out, const bf16 *norm, const bf16 *scale, const bf16 *shift, int S, int D) {
    int n = S * D;
    adaln_modulate_kernel<<<(n + 255) / 256, 256>>>(out, norm, scale, shift, S, D);
}

// Gated residual: x[i] = x[i] + y[i] * gate[d]  (gate broadcast over S)
__global__ void gated_add_kernel(bf16 *x, const bf16 *y, const bf16 *gate, int S, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= S * D) return;
    int d = idx % D;
    float xv = __bfloat162float(x[idx]);
    float yv = __bfloat162float(y[idx]);
    float gv = __bfloat162float(gate[d]);
    x[idx] = __float2bfloat16(xv + yv * gv);
}

static void gated_add(bf16 *x, const bf16 *y, const bf16 *gate, int S, int D) {
    int n = S * D;
    gated_add_kernel<<<(n + 255) / 256, 256>>>(x, y, gate, S, D);
}

// Simple residual add: x[i] += y[i]
__global__ void add_2d_kernel(bf16 *x, const bf16 *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = __float2bfloat16(__bfloat162float(x[i]) + __bfloat162float(y[i]));
}

static void add_2d(bf16 *x, const bf16 *y, int n) {
    add_2d_kernel<<<(n + 255) / 256, 256>>>(x, y, n);
}

// Copy bf16
static void copy_2d(bf16 *dst, const bf16 *src, int n) {
    cudaMemcpyAsync(dst, src, n * sizeof(bf16), cudaMemcpyDeviceToDevice);
}

// GQA expand: repeat KV heads to match Q heads
// in:  [n_kv_heads, S, D]  -> out: [n_heads, S, D]  (n_heads/n_kv_heads copies)
__global__ void gqa_expand_kernel(bf16 *out, const bf16 *in, int n_heads, int n_kv_heads, int S, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_heads * S * D;
    if (idx >= total) return;
    int h = idx / (S * D);
    int rem = idx % (S * D);
    int kv_h = h * n_kv_heads / n_heads;
    out[idx] = in[kv_h * S * D + rem];
}

static void gqa_expand(bf16 *out, const bf16 *in, int n_heads, int n_kv_heads, int S, int D) {
    int n = n_heads * S * D;
    gqa_expand_kernel<<<(n + 255) / 256, 256>>>(out, in, n_heads, n_kv_heads, S, D);
}

// Batch RoPE: apply rotary embeddings to Q[n_heads, S, D] and K[n_kv_heads, S, D]
// half-split: first D/2 dims get cos/sin, second D/2 get cos/sin with swap
// pos_ids: [S] position indices
__global__ void rope_batch_kernel(bf16 *q, bf16 *k,
                                  int n_q_heads, int n_kv_heads,
                                  int S, int head_dim,
                                  float theta_base) {
    // Grid: one block per (head, position) pair
    // Handles both Q and K heads
    int block_id = blockIdx.x;
    int total_heads = n_q_heads + n_kv_heads;
    int h = block_id / S;
    int s = block_id % S;
    if (h >= total_heads) return;

    int half = head_dim / 2;
    bool is_q = (h < n_q_heads);
    bf16 *ptr;
    if (is_q)
        ptr = q + (int64_t)h * S * head_dim + (int64_t)s * head_dim;
    else
        ptr = k + (int64_t)(h - n_q_heads) * S * head_dim + (int64_t)s * head_dim;

    int pos = s;  // position = sequence index (DiT uses sequential positions)

    for (int d = threadIdx.x; d < half; d += blockDim.x) {
        float freq = 1.0f / powf(theta_base, (float)(2 * d) / head_dim);
        float angle = pos * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);

        float v0 = __bfloat162float(ptr[d]);
        float v1 = __bfloat162float(ptr[d + half]);

        ptr[d]        = __float2bfloat16(v0 * cos_a - v1 * sin_a);
        ptr[d + half] = __float2bfloat16(v1 * cos_a + v0 * sin_a);
    }
}

static void rope_batch(bf16 *q, bf16 *k, int n_q_heads, int n_kv_heads,
                       int S, int head_dim, float theta) {
    int total_heads = n_q_heads + n_kv_heads;
    int blocks = total_heads * S;
    int threads = (head_dim / 2 < 128) ? (head_dim / 2) : 128;
    rope_batch_kernel<<<blocks, threads>>>(q, k, n_q_heads, n_kv_heads, S, head_dim, theta);
}

// QK-Norm for 2D: normalize each head vector independently
// x: [n_heads, S, head_dim], weight: [head_dim]
// Each block handles one (head, position)
__global__ void qknorm_2d_kernel(bf16 *x, const bf16 *w, int n_heads, int S, int D) {
    int block_id = blockIdx.x;
    int h = block_id / S;
    int s = block_id % S;
    if (h >= n_heads) return;
    bf16 *vec = x + (int64_t)h * S * D + (int64_t)s * D;

    extern __shared__ float smem[];
    float local_ss = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float v = __bfloat162float(vec[d]);
        local_ss += v * v;
    }
    smem[threadIdx.x] = local_ss;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }
    float rms = rsqrtf(smem[0] / D + 1e-6f);

    for (int d = threadIdx.x; d < D; d += blockDim.x)
        vec[d] = __float2bfloat16(__bfloat162float(vec[d]) * rms * __bfloat162float(w[d]));
}

static void qknorm_2d(bf16 *x, const bf16 *w, int n_heads, int S, int D) {
    int blocks = n_heads * S;
    int threads = (D < 128) ? D : 128;
    qknorm_2d_kernel<<<blocks, threads, threads * sizeof(float)>>>(x, w, n_heads, S, D);
}

// SwiGLU for 2D: gate[S,D] = silu(gate[S,D]) * up[S,D]
__global__ void silu_mul_2d_kernel(bf16 *gate, const bf16 *up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = __bfloat162float(gate[i]);
    float u = __bfloat162float(up[i]);
    float silu_g = g / (1.0f + expf(-g));
    gate[i] = __float2bfloat16(silu_g * u);
}

static void silu_mul_2d(bf16 *gate, const bf16 *up, int S, int D) {
    int n = S * D;
    silu_mul_2d_kernel<<<(n + 255) / 256, 256>>>(gate, up, n);
}

// Row-wise softmax on f32 attention scores [batch, S_q, S_kv]
// With optional sliding window mask (window=0 means full attention)
// Each block handles one row
__global__ void softmax_attn_kernel(float *scores, int S_q, int S_kv, int window, float scale) {
    int row_idx = blockIdx.x;   // which (batch, s_q) pair
    int s_q = row_idx % S_q;
    float *row = scores + (int64_t)row_idx * S_kv;

    extern __shared__ float smem[];

    // 1) Apply scale + sliding window mask, find max
    float local_max = -1e30f;
    for (int j = threadIdx.x; j < S_kv; j += blockDim.x) {
        float v = row[j] * scale;
        // Sliding window mask (bidirectional): |s_q - j| > window -> mask out
        // Python: torch.abs(diff) <= sliding_window
        if (window > 0) {
            int d = (s_q > j) ? (s_q - j) : (j - s_q);
            if (d > window) v = -1e30f;
        }        row[j] = v;
        local_max = fmaxf(local_max, v);
    }
    smem[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        __syncthreads();
    }
    float max_val = smem[0];

    // 2) Exp + sum
    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < S_kv; j += blockDim.x) {
        float v = expf(row[j] - max_val);
        row[j] = v;
        local_sum += v;
    }
    smem[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / (smem[0] + 1e-9f);

    // 3) Normalize
    for (int j = threadIdx.x; j < S_kv; j += blockDim.x)
        row[j] *= inv_sum;
}

// scale is 1/sqrt(head_dim), window=0 for full attention
static void softmax_attn(float *scores, int batch, int S_q, int S_kv, float scale, int window) {
    int rows = batch * S_q;
    int threads = (S_kv < 256) ? S_kv : 256;
    // Round threads to next power of 2 for reduction
    int t = 1; while (t < threads) t <<= 1; threads = t;
    if (threads > 256) threads = 256;
    softmax_attn_kernel<<<rows, threads, threads * sizeof(float)>>>(scores, S_q, S_kv, window, scale);
}

// Sinusoidal timestep embedding: t[B] -> emb[B, dim]
// emb[b, d] = cos/sin(t[b] * scale * freq[d])
__global__ void sinusoidal_emb_kernel(bf16 *emb, const float *t, int B, int dim, float scale, float max_period) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * dim;
    if (idx >= total) return;
    int b = idx / dim;
    int d = idx % dim;
    int half = dim / 2;

    float tv = t[b] * scale;
    float freq = expf(-(logf(max_period)) * (float)(d % half) / (float)half);
    float angle = tv * freq;
    float val = (d < half) ? cosf(angle) : sinf(angle);
    emb[idx] = __float2bfloat16(val);
}

static void sinusoidal_emb(bf16 *emb, const float *t, int B, int dim, float scale, float max_period) {
    int n = B * dim;
    sinusoidal_emb_kernel<<<(n + 255) / 256, 256>>>(emb, t, B, dim, scale, max_period);
}

// SiLU activation: x = x * sigmoid(x)
__global__ void silu_kernel(bf16 *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = __bfloat162float(x[i]);
    x[i] = __float2bfloat16(v / (1.0f + expf(-v)));
}

static void silu_inplace(bf16 *x, int n) {
    silu_kernel<<<(n + 255) / 256, 256>>>(x, n);
}

// Deinterleave for ConvTranspose1d output (after permuted weight GEMM):
// in: [T/2, kernel*out_ch] where layout is [k0_c0..k0_cN, k1_c0..k1_cN]
// out: [T, out_ch]  where out[2s+k, c] = in[s, k*out_ch + c]
__global__ void deinterleave_kernel(bf16 *out, const bf16 *in, int T_half, int out_ch, int kernel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int T = T_half * kernel;
    if (idx >= T * out_ch) return;
    int t = idx / out_ch;
    int c = idx % out_ch;
    int s = t / kernel;
    int k = t % kernel;
    out[idx] = in[(int64_t)s * kernel * out_ch + k * out_ch + c];
}

static void deinterleave(bf16 *out, const bf16 *in, int T_half, int out_ch, int kernel) {
    int n = T_half * kernel * out_ch;
    deinterleave_kernel<<<(n + 255) / 256, 256>>>(out, in, T_half, out_ch, kernel);
}

// Reshape Q/K from [S, n_heads*head_dim] (row-major) to [n_heads, S, head_dim]
// This is needed for batched attention GEMM
// In: contiguous [S, H*D], Out: contiguous [H, S, D]
__global__ void reshape_to_heads_kernel(bf16 *out, const bf16 *in, int S, int H, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = S * H * D;
    if (idx >= total) return;
    // out index: [h, s, d]
    int h = idx / (S * D);
    int rem = idx % (S * D);
    int s = rem / D;
    int d = rem % D;
    // in index: [s, h*D + d]
    out[idx] = in[(int64_t)s * H * D + h * D + d];
}

static void reshape_to_heads(bf16 *out, const bf16 *in, int S, int H, int D) {
    int n = S * H * D;
    reshape_to_heads_kernel<<<(n + 255) / 256, 256>>>(out, in, S, H, D);
}

// Inverse: [H, S, D] -> [S, H*D]
__global__ void reshape_from_heads_kernel(bf16 *out, const bf16 *in, int S, int H, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = S * H * D;
    if (idx >= total) return;
    // in index: [h, s, d]
    int h = idx / (S * D);
    int rem = idx % (S * D);
    int s = rem / D;
    int d = rem % D;
    // out index: [s, h*D + d]
    out[(int64_t)s * H * D + h * D + d] = in[idx];
}

static void reshape_from_heads(bf16 *out, const bf16 *in, int S, int H, int D) {
    int n = S * H * D;
    reshape_from_heads_kernel<<<(n + 255) / 256, 256>>>(out, in, S, H, D);
}

// DiT Layer weights
struct DiTLayer {
    // Self-attention
    bf16 *self_attn_norm;     // [hidden_size]
    bf16 *sa_q_proj;          // [n_heads*head_dim, hidden_size]
    bf16 *sa_k_proj;          // [n_kv_heads*head_dim, hidden_size]
    bf16 *sa_v_proj;          // [n_kv_heads*head_dim, hidden_size]
    bf16 *sa_q_norm;          // [head_dim]
    bf16 *sa_k_norm;          // [head_dim]
    bf16 *sa_o_proj;          // [hidden_size, n_heads*head_dim]

    // Cross-attention
    bf16 *cross_attn_norm;    // [hidden_size]
    bf16 *ca_q_proj;          // [n_heads*head_dim, hidden_size]
    bf16 *ca_k_proj;          // [n_kv_heads*head_dim, hidden_size]
    bf16 *ca_v_proj;          // [n_kv_heads*head_dim, hidden_size]
    bf16 *ca_q_norm;          // [head_dim]
    bf16 *ca_k_norm;          // [head_dim]
    bf16 *ca_o_proj;          // [hidden_size, n_heads*head_dim]

    // MLP
    bf16 *mlp_norm;           // [hidden_size]
    bf16 *gate_proj;          // [intermediate_size, hidden_size]
    bf16 *up_proj;            // [intermediate_size, hidden_size]
    bf16 *down_proj;          // [hidden_size, intermediate_size]

    // AdaLN scale-shift table: [1, 6, hidden_size] stored as [6, hidden_size]
    bf16 *scale_shift_table;

    // Layer type: 0=sliding, 1=full
    int layer_type;
};

// TimestepEmbedding weights
struct TimestepEmbWeights {
    bf16 *linear_1_w;     // [hidden, 256]
    bf16 *linear_1_b;     // [hidden]
    bf16 *linear_2_w;     // [hidden, hidden]
    bf16 *linear_2_b;     // [hidden]
    bf16 *time_proj_w;    // [hidden*6, hidden]
    bf16 *time_proj_b;    // [hidden*6]
};

// Full DiT model
#define DIT_MAX_LAYERS 32

struct DiTModel {
    DiTConfig cfg;

    // Timestep embeddings (two: for t and t-r)
    TimestepEmbWeights time_embed;
    TimestepEmbWeights time_embed_r;

    // proj_in: Conv1d(in_channels, hidden, kernel=2, stride=2) + bias
    // Stored as [hidden, in_channels*2] for GEMM
    bf16 *proj_in_w;      // [hidden_size, in_channels * patch_size]
    bf16 *proj_in_b;      // [hidden_size]

    // condition_embedder: Linear(hidden, hidden, bias=True)
    bf16 *cond_emb_w;     // [hidden_size, hidden_size]
    bf16 *cond_emb_b;     // [hidden_size]

    // Transformer layers
    DiTLayer layers[DIT_MAX_LAYERS];

    // Output: norm + scale_shift + proj_out (ConvTranspose1d)
    bf16 *norm_out;         // [hidden_size]
    bf16 *out_scale_shift;  // [2, hidden_size], scale_shift_table for output
    bf16 *proj_out_w;       // [hidden_size, out_channels * patch_size]
    bf16 *proj_out_b;       // [out_channels]

    // RoPE (not stored, computed on the fly)

    // Buffers
    // Scratch for forward pass (B=1, S = max_seq_len)
    bf16 *buf_hidden;       // [S, hidden_size]
    bf16 *buf_residual;     // [S, hidden_size]
    bf16 *buf_norm;         // [S, hidden_size]
    bf16 *buf_q;            // [n_heads, S, head_dim]  (after reshape to heads)
    bf16 *buf_k;            // [n_kv_heads, S, head_dim]
    bf16 *buf_v;            // [n_kv_heads, S, head_dim]
    bf16 *buf_k_exp;        // [n_heads, S, head_dim]  (GQA expanded)
    bf16 *buf_v_exp;        // [n_heads, S, head_dim]
    float *buf_scores;      // [n_heads, S, S]          (f32 for softmax)
    bf16  *buf_scores_bf16; // [n_heads, S, max(S,enc_S)] scratch for SV GEMM
    bf16 *buf_attn_out;     // [S, n_heads * head_dim]  (after reshape from heads)
    bf16 *buf_gate;         // [S, intermediate_size]
    bf16 *buf_up;           // [S, intermediate_size]
    bf16 *buf_proj_tmp;     // [max(S,enc_S), max(hidden,out_ch*P)] temp for projections
    bf16 *buf_adaln;        // [6, hidden_size], per-layer AdaLN params (safe from aliasing)
    bf16 *buf_vt;           // [S*patch_size, out_channels] velocity output for flow matching
    bf16 *buf_concat;       // [S*patch_size, in_channels] concat(context_latents, xt)

    // Timestep embedding scratch
    bf16 *buf_temb;         // [hidden_size]  (combined temb)
    bf16 *buf_temb_r;       // [hidden_size]  (temb_r out, avoids aliasing with buf_temb_tmp)
    bf16 *buf_tproj;        // [6, hidden_size] (combined timestep_proj)
    bf16 *buf_sinusoidal;   // [256] sinusoidal
    bf16 *buf_temb_tmp;     // [hidden_size] temp for linear_2 inside forward_temb
    float *buf_t_f32;       // [1] timestep scalar on GPU

    // Cross-attention KV cache: [n_layers][2][enc_S][n_kv_heads*head_dim]
    bf16 *cross_kv_cache;
    int cross_kv_enc_S;     // encoder sequence length (set at first use)
    bool cross_kv_valid;    // true after first flow step computes KV
    int max_enc_S;          // max encoder seq len for buffer sizing

    // Encoder hidden states buffer (projected)
    bf16 *buf_enc_hidden;   // [enc_S, hidden_size]

    cublasHandle_t cublas;
};

// Weight loading helpers

// Upload bf16 tensor to GPU (uses managed memory to handle VRAM overflow)
static bf16 *upload_tensor(const SafeTensor &t, size_t expected_elements = 0) {
    size_t nelems = t.nbytes / 2;  // bf16 = 2 bytes
    if (expected_elements > 0 && nelems != expected_elements) {
        fprintf(stderr, "FATAL: tensor size mismatch: got %zu, expected %zu\n", nelems, expected_elements);
        exit(1);
    }
    bf16 *d;
    cudaError_t e = cudaMalloc(&d, t.nbytes);
    if (e != cudaSuccess) {
        cudaGetLastError();  // clear stale error from failed cudaMalloc
        e = cudaMallocManaged(&d, t.nbytes);
        if (e != cudaSuccess) {
            fprintf(stderr, "FATAL: cudaMallocManaged failed (%zu bytes): %s\n", t.nbytes, cudaGetErrorString(e));
            exit(1);
        }
    }
    CUDA_CHECK(cudaMemcpy(d, t.data, t.nbytes, cudaMemcpyDefault));
    return d;
}

// Helper: try cudaMalloc, fall back to cudaMallocManaged
static bf16 *safe_bf16_malloc(size_t nbytes) {
    bf16 *d;
    cudaError_t e = cudaMalloc(&d, nbytes);
    if (e != cudaSuccess) {
        cudaGetLastError();  // clear error
        e = cudaMallocManaged(&d, nbytes);
        if (e != cudaSuccess) {
            fprintf(stderr, "FATAL: alloc failed (%zu bytes): %s\n", nbytes, cudaGetErrorString(e));
            exit(1);
        }
    }
    return d;
}

// Generic version for any pointer type
#define DIT_MALLOC(ptr, nbytes) do { \
    cudaError_t _e = cudaMalloc((void**)&(ptr), (nbytes)); \
    if (_e != cudaSuccess) { \
        cudaGetLastError(); \
        _e = cudaMallocManaged((void**)&(ptr), (nbytes)); \
        if (_e != cudaSuccess) { \
            fprintf(stderr, "FATAL: alloc failed (%zu bytes): %s\n", (size_t)(nbytes), cudaGetErrorString(_e)); \
            exit(1); \
        } \
    } \
} while(0)

// Upload Conv1d weight [out_ch, in_ch, kernel] -> permute to [out_ch, kernel, in_ch]
// So that GEMM can treat it as [out_ch, kernel*in_ch] matching input layout [kernel*in_ch]
static bf16 *upload_conv1d_weight(const SafeTensor &t, int out_ch, int in_ch, int kernel) {
    size_t nelems = (size_t)out_ch * in_ch * kernel;
    if (t.nbytes / 2 != nelems) {
        fprintf(stderr, "FATAL: conv1d weight size mismatch: got %zu, expected %zu\n", t.nbytes / 2, nelems);
        exit(1);
    }
    // Permute on CPU: [out_ch, in_ch, kernel] -> [out_ch, kernel, in_ch]
    const bf16 *src = (const bf16 *)t.data;
    std::vector<bf16> permuted(nelems);
    for (int o = 0; o < out_ch; o++)
        for (int k = 0; k < kernel; k++)
            for (int i = 0; i < in_ch; i++)
                permuted[o * kernel * in_ch + k * in_ch + i] = src[o * in_ch * kernel + i * kernel + k];
    bf16 *d;
    d = safe_bf16_malloc(nelems * sizeof(bf16));
    CUDA_CHECK(cudaMemcpy(d, permuted.data(), nelems * sizeof(bf16), cudaMemcpyDefault));
    fprintf(stderr, "[DiT] Permuted conv1d weight [%d, %d, %d] -> [%d, %d]\n", out_ch, in_ch, kernel, out_ch, kernel * in_ch);
    return d;
}

// Upload ConvTranspose1d weight [in_ch, out_ch, kernel] -> permute to [in_ch, kernel*out_ch]
// with layout [in_ch, k0_c0, k0_c1, ..., k1_c0, k1_c1, ...]
// Actually for deinterleave output: we want each input position to produce kernel output positions
// Weight stored as [in_ch, out_ch, kernel], we permute to [in_ch, kernel, out_ch] -> [in_ch, kernel*out_ch]
static bf16 *upload_conv_transpose1d_weight(const SafeTensor &t, int in_ch, int out_ch, int kernel) {
    size_t nelems = (size_t)in_ch * out_ch * kernel;
    if (t.nbytes / 2 != nelems) {
        fprintf(stderr, "FATAL: conv_transpose1d weight size mismatch: got %zu, expected %zu\n", t.nbytes / 2, nelems);
        exit(1);
    }
    // Permute on CPU: [in_ch, out_ch, kernel] -> [in_ch, kernel, out_ch]
    const bf16 *src = (const bf16 *)t.data;
    std::vector<bf16> permuted(nelems);
    for (int i = 0; i < in_ch; i++)
        for (int k = 0; k < kernel; k++)
            for (int o = 0; o < out_ch; o++)
                permuted[i * kernel * out_ch + k * out_ch + o] = src[i * out_ch * kernel + o * kernel + k];
    bf16 *d;
    d = safe_bf16_malloc(nelems * sizeof(bf16));
    CUDA_CHECK(cudaMemcpy(d, permuted.data(), nelems * sizeof(bf16), cudaMemcpyDefault));
    fprintf(stderr, "[DiT] Permuted conv_transpose1d weight [%d, %d, %d] -> [%d, %d]\n", in_ch, out_ch, kernel, in_ch, kernel * out_ch);
    return d;
}

// Try to load tensor, return nullptr if not found
static bf16 *try_upload(const SafeTensors &st, const std::string &name) {
    auto it = st.tensors.find(name);
    if (it == st.tensors.end()) return nullptr;
    return upload_tensor(it->second);
}

static bf16 *must_upload(const SafeTensors &st, const std::string &name) {
    bf16 *p = try_upload(st, name);
    if (!p) { fprintf(stderr, "FATAL: tensor '%s' not found\n", name.c_str()); exit(1); }
    return p;
}

// Load TimestepEmbedding weights
static void load_temb_weights(TimestepEmbWeights &w, const SafeTensors &st, const std::string &prefix) {
    w.linear_1_w  = must_upload(st, prefix + ".linear_1.weight");
    w.linear_1_b  = must_upload(st, prefix + ".linear_1.bias");
    w.linear_2_w  = must_upload(st, prefix + ".linear_2.weight");
    w.linear_2_b  = must_upload(st, prefix + ".linear_2.bias");
    w.time_proj_w = must_upload(st, prefix + ".time_proj.weight");
    w.time_proj_b = must_upload(st, prefix + ".time_proj.bias");
}

// Load full DiT model
static void load_dit_model(DiTModel *m, const char *model_dir, DiTConfig cfg) {
    m->cfg = cfg;

    SafeTensors st;
    if (!safe_load(st, model_dir)) {
        fprintf(stderr, "FATAL: cannot load safetensors from %s\n", model_dir);
        exit(1);
    }

    fprintf(stderr, "[DiT] Loading weights from %s (%zu tensors)...\n", model_dir, st.tensors.size());

    // Timestep embeddings
    load_temb_weights(m->time_embed, st, "decoder.time_embed");
    load_temb_weights(m->time_embed_r, st, "decoder.time_embed_r");

    // proj_in: Conv1d(in_channels, hidden, kernel=patch_size, stride=patch_size)
    // Weight [hidden, in_channels, patch_size] -> permuted to [hidden, patch_size*in_channels]
    {
        const SafeTensor &t = safe_get(st, "decoder.proj_in.1.weight");
        m->proj_in_w = upload_conv1d_weight(t, cfg.hidden_size, cfg.in_channels, cfg.patch_size);
    }
    m->proj_in_b = must_upload(st, "decoder.proj_in.1.bias");

    // condition_embedder
    m->cond_emb_w = must_upload(st, "decoder.condition_embedder.weight");
    m->cond_emb_b = must_upload(st, "decoder.condition_embedder.bias");

    // Layers
    for (int i = 0; i < cfg.n_layers; i++) {
        char prefix[128];
        DiTLayer &ly = m->layers[i];

        snprintf(prefix, sizeof(prefix), "decoder.layers.%d", i);
        std::string p(prefix);

        // Self-attention
        ly.self_attn_norm = must_upload(st, p + ".self_attn_norm.weight");
        ly.sa_q_proj      = must_upload(st, p + ".self_attn.q_proj.weight");
        ly.sa_k_proj      = must_upload(st, p + ".self_attn.k_proj.weight");
        ly.sa_v_proj      = must_upload(st, p + ".self_attn.v_proj.weight");
        ly.sa_q_norm      = must_upload(st, p + ".self_attn.q_norm.weight");
        ly.sa_k_norm      = must_upload(st, p + ".self_attn.k_norm.weight");
        ly.sa_o_proj      = must_upload(st, p + ".self_attn.o_proj.weight");

        // Cross-attention
        ly.cross_attn_norm = must_upload(st, p + ".cross_attn_norm.weight");
        ly.ca_q_proj       = must_upload(st, p + ".cross_attn.q_proj.weight");
        ly.ca_k_proj       = must_upload(st, p + ".cross_attn.k_proj.weight");
        ly.ca_v_proj       = must_upload(st, p + ".cross_attn.v_proj.weight");
        ly.ca_q_norm       = must_upload(st, p + ".cross_attn.q_norm.weight");
        ly.ca_k_norm       = must_upload(st, p + ".cross_attn.k_norm.weight");
        ly.ca_o_proj       = must_upload(st, p + ".cross_attn.o_proj.weight");

        // MLP
        ly.mlp_norm  = must_upload(st, p + ".mlp_norm.weight");
        ly.gate_proj = must_upload(st, p + ".mlp.gate_proj.weight");
        ly.up_proj   = must_upload(st, p + ".mlp.up_proj.weight");
        ly.down_proj = must_upload(st, p + ".mlp.down_proj.weight");

        // AdaLN scale-shift table [1, 6, hidden_size], stored flat as [6*hidden_size]
        ly.scale_shift_table = must_upload(st, p + ".scale_shift_table");

        // Layer type
        ly.layer_type = (i % 2 == 0) ? 0 : 1;  // 0=sliding, 1=full
    }

    // Output
    m->norm_out       = must_upload(st, "decoder.norm_out.weight");
    m->out_scale_shift = must_upload(st, "decoder.scale_shift_table");  // [1, 2, hidden_size]
    // proj_out: ConvTranspose1d(hidden, out_channels, kernel=patch_size, stride=patch_size)
    // Weight [hidden, out_channels, patch_size] -> permuted to [hidden, patch_size*out_channels]
    {
        const SafeTensor &t = safe_get(st, "decoder.proj_out.1.weight");
        m->proj_out_w = upload_conv_transpose1d_weight(t, cfg.hidden_size, cfg.out_channels, cfg.patch_size);
    }
    m->proj_out_b     = must_upload(st, "decoder.proj_out.1.bias");

    int S = cfg.max_seq_len;
    int H = cfg.hidden_size;
    int I = cfg.intermediate_size;
    int Nh = cfg.n_heads;
    int Nkv = cfg.n_kv_heads;
    int D = cfg.head_dim;
    int Oc = cfg.out_channels;
    int P = cfg.patch_size;

    // Conservative max encoder seq len (lyric + timbre + text tokens)
    // Will be reallocated if enc_S > this at runtime
    int max_eS = 256;
    m->max_enc_S = max_eS;
    int Smax = (S > max_eS) ? S : max_eS;  // max of self-attn and cross-attn seq lens

    // Allocate scratch buffers
    DIT_MALLOC(m->buf_hidden,       (int64_t)S * H * sizeof(bf16));
    DIT_MALLOC(m->buf_residual,     (int64_t)S * H * sizeof(bf16));
    DIT_MALLOC(m->buf_norm,         (int64_t)S * H * sizeof(bf16));
    DIT_MALLOC(m->buf_q,            (int64_t)Nh * S * D * sizeof(bf16));
    DIT_MALLOC(m->buf_k,            (int64_t)Nkv * Smax * D * sizeof(bf16));
    DIT_MALLOC(m->buf_v,            (int64_t)Nkv * Smax * D * sizeof(bf16));
    DIT_MALLOC(m->buf_k_exp,        (int64_t)Nh * Smax * D * sizeof(bf16));
    DIT_MALLOC(m->buf_v_exp,        (int64_t)Nh * Smax * D * sizeof(bf16));
    DIT_MALLOC(m->buf_scores,       (int64_t)Nh * S * Smax * sizeof(float));
    DIT_MALLOC(m->buf_scores_bf16,  (int64_t)Nh * S * Smax * sizeof(bf16));
    DIT_MALLOC(m->buf_attn_out,     (int64_t)S * Nh * D * sizeof(bf16));
    DIT_MALLOC(m->buf_gate,         (int64_t)S * I * sizeof(bf16));
    DIT_MALLOC(m->buf_up,           (int64_t)S * I * sizeof(bf16));
    DIT_MALLOC(m->buf_proj_tmp,     (int64_t)Smax * H * sizeof(bf16));
    DIT_MALLOC(m->buf_adaln,        6 * H * sizeof(bf16));
    DIT_MALLOC(m->buf_vt,           (int64_t)S * P * Oc * sizeof(bf16));
    DIT_MALLOC(m->buf_concat,       (int64_t)S * P * cfg.in_channels * sizeof(bf16));

    // Timestep embedding buffers
    DIT_MALLOC(m->buf_temb,       H * sizeof(bf16));
    DIT_MALLOC(m->buf_temb_r,     H * sizeof(bf16));
    DIT_MALLOC(m->buf_tproj,      6 * H * sizeof(bf16));
    DIT_MALLOC(m->buf_sinusoidal, 256 * sizeof(bf16));
    DIT_MALLOC(m->buf_temb_tmp,   H * sizeof(bf16));
    DIT_MALLOC(m->buf_t_f32, sizeof(float));

    // Cross-attention KV cache (allocated lazily)
    m->cross_kv_cache = nullptr;
    m->cross_kv_enc_S = 0;
    m->cross_kv_valid = false;

    m->buf_enc_hidden = nullptr;

    cublasCreate(&m->cublas);
    cublasSetMathMode(m->cublas, CUBLAS_DEFAULT_MATH);

    fprintf(stderr, "[DiT] Model loaded. %d layers, buffers for S=%d\n", cfg.n_layers, S);
}

// TimestepEmbedding forward
// Input: t scalar (f32 on GPU)
// Output: temb [hidden_size], timestep_proj [6, hidden_size]
static void forward_temb(DiTModel *m, const TimestepEmbWeights &w,
                         const float *t_gpu, bf16 *out_temb, bf16 *out_proj,
                         const DebugDumper *dbg = nullptr, const char *suffix = "") {
    int H = m->cfg.hidden_size;

    // 1) Sinusoidal embedding: t -> [256]
    sinusoidal_emb(m->buf_sinusoidal, t_gpu, 1, 256, 1000.0f, 10000.0f);
    if (dbg && dbg->enabled) {
        char name[64];
        snprintf(name, sizeof(name), "sinusoidal%s", suffix);
        debug_dump_bf16_2d(dbg, name, m->buf_sinusoidal, 1, 256);
    }

    // 2) linear_1: [256] -> [H]
    linear_batch(out_temb, m->buf_sinusoidal, w.linear_1_w, nullptr, 1, H, 256, m->cublas);
    add_bias_2d(out_temb, w.linear_1_b, 1, H);
    if (dbg && dbg->enabled) {
        char name[64];
        snprintf(name, sizeof(name), "temb_lin1%s", suffix);
        debug_dump_bf16_2d(dbg, name, out_temb, 1, H);
    }

    // 3) SiLU
    silu_inplace(out_temb, H);

    // 4) linear_2: [H] -> [H]
    linear_batch(m->buf_temb_tmp, out_temb, w.linear_2_w, nullptr, 1, H, H, m->cublas);
    add_bias_2d(m->buf_temb_tmp, w.linear_2_b, 1, H);
    copy_2d(out_temb, m->buf_temb_tmp, H);

    // 5) SiLU + time_proj: [H] -> [6*H]
    copy_2d(m->buf_temb_tmp, out_temb, H);
    silu_inplace(m->buf_temb_tmp, H);
    linear_batch(out_proj, m->buf_temb_tmp, w.time_proj_w, nullptr, 1, 6 * H, H, m->cublas);
    add_bias_2d(out_proj, w.time_proj_b, 1, 6 * H);
    // out_proj is now [6*H], viewed as [6, H] (timestep_proj)
}

// DiT self-attention forward (bidirectional, batch)
static void dit_self_attention(DiTModel *m, DiTLayer &ly, int S, int window,
                               const DebugDumper *dbg = nullptr, int layer = -1, int step = -1) {
    DiTConfig &c = m->cfg;
    int H = c.hidden_size;
    int Nh = c.n_heads;
    int Nkv = c.n_kv_heads;
    int D = c.head_dim;

    // Q, K, V projections: buf_norm [S, H] -> q/k/v
    // Q: [S, Nh*D], K: [S, Nkv*D], V: [S, Nkv*D]
    linear_batch(m->buf_attn_out, m->buf_norm, ly.sa_q_proj, nullptr, S, Nh * D, H, m->cublas);   // reuse buf_attn_out for Q temp
    linear_batch(m->buf_proj_tmp, m->buf_norm, ly.sa_k_proj, nullptr, S, Nkv * D, H, m->cublas);  // K temp in proj_tmp
    bf16 *k_flat = m->buf_proj_tmp;  // [S, Nkv*D]

    // V projection to buf_v (reshaped later)
    // We need a separate buffer for V flat, use buf_gate temporarily (large enough: S*I >> S*Nkv*D)
    bf16 *v_flat = m->buf_gate;
    linear_batch(v_flat, m->buf_norm, ly.sa_v_proj, nullptr, S, Nkv * D, H, m->cublas);

    // Reshape to heads: [S, H*D] -> [H, S, D]
    reshape_to_heads(m->buf_q, m->buf_attn_out, S, Nh, D);    // Q: [Nh, S, D]
    reshape_to_heads(m->buf_k, k_flat, S, Nkv, D);             // K: [Nkv, S, D]
    reshape_to_heads(m->buf_v, v_flat, S, Nkv, D);             // V: [Nkv, S, D]

    // QK-Norm
    qknorm_2d(m->buf_q, ly.sa_q_norm, Nh, S, D);
    qknorm_2d(m->buf_k, ly.sa_k_norm, Nkv, S, D);

    // RoPE (bidirectional, sequential positions)
    rope_batch(m->buf_q, m->buf_k, Nh, Nkv, S, D, c.rope_theta);

    // Debug: dump Q and K after QK-Norm + RoPE
    if (layer == 0 && step == 0 && dbg && dbg->enabled) {
        debug_dump_bf16_2d(dbg, "layer0_q_after_rope", m->buf_q, Nh * S, D);
        debug_dump_bf16_2d(dbg, "layer0_k_after_rope", m->buf_k, Nkv * S, D);
    }

    // GQA expand K, V: [Nkv, S, D] -> [Nh, S, D]
    gqa_expand(m->buf_k_exp, m->buf_k, Nh, Nkv, S, D);
    gqa_expand(m->buf_v_exp, m->buf_v, Nh, Nkv, S, D);

    // Q @ K^T -> scores [Nh, S, S] (f32)
    batch_qk_gemm(m->buf_scores, m->buf_q, m->buf_k_exp, S, D, Nh, m->cublas);

    // Softmax with optional sliding window mask
    float scale = 1.0f / sqrtf((float)D);
    int win = (window > 0) ? window : 0;
    softmax_attn(m->buf_scores, Nh, S, S, scale, win);

    // scores @ V -> attn_output [Nh, S, D] (bf16)
    batch_sv_gemm(m->buf_q, m->buf_scores, m->buf_v_exp, S, D, Nh, m->cublas, m->buf_scores_bf16);  // reuse buf_q for output

    // Reshape back: [Nh, S, D] -> [S, Nh*D]
    reshape_from_heads(m->buf_attn_out, m->buf_q, S, Nh, D);

    // Debug: attention output before O-proj
    if (layer == 0 && step == 0 && dbg && dbg->enabled)
        debug_dump_bf16_2d(dbg, "layer0_attn_out", m->buf_attn_out, S, Nh * D);

    // O projection: [S, Nh*D] -> [S, H]
    linear_batch(m->buf_norm, m->buf_attn_out, ly.sa_o_proj, nullptr, S, H, Nh * D, m->cublas);  // reuse buf_norm
    // buf_norm now holds self-attention output [S, H]
}

// DiT cross-attention forward
static void dit_cross_attention(DiTModel *m, DiTLayer &ly, int layer_idx, int S) {
    DiTConfig &c = m->cfg;
    int H = c.hidden_size;
    int Nh = c.n_heads;
    int Nkv = c.n_kv_heads;
    int D = c.head_dim;
    int enc_S = m->cross_kv_enc_S;
    int kv_dim = Nkv * D;

    // Q from current hidden state
    // buf_norm already has norm(hidden) after cross_attn_norm
    linear_batch(m->buf_attn_out, m->buf_norm, ly.ca_q_proj, nullptr, S, Nh * D, H, m->cublas);
    reshape_to_heads(m->buf_q, m->buf_attn_out, S, Nh, D);
    qknorm_2d(m->buf_q, ly.ca_q_norm, Nh, S, D);
    // No RoPE for cross-attention

    // K, V from encoder (cached or computed)
    bf16 *cached_k = m->cross_kv_cache + (int64_t)layer_idx * 2 * enc_S * kv_dim;
    bf16 *cached_v = cached_k + (int64_t)enc_S * kv_dim;

    if (!m->cross_kv_valid) {
        // First flow step: compute K, V from encoder hidden states
        // K: [enc_S, Nkv*D]
        linear_batch(m->buf_proj_tmp, m->buf_enc_hidden, ly.ca_k_proj, nullptr,
                     enc_S, kv_dim, H, m->cublas);
        // Reshape, QK-norm, then store
        reshape_to_heads(m->buf_k, m->buf_proj_tmp, enc_S, Nkv, D);
        qknorm_2d(m->buf_k, ly.ca_k_norm, Nkv, enc_S, D);
        cudaMemcpyAsync(cached_k, m->buf_k, (int64_t)Nkv * enc_S * D * sizeof(bf16), cudaMemcpyDeviceToDevice);

        // V: [enc_S, Nkv*D]
        bf16 *v_flat = m->buf_gate;  // temp
        linear_batch(v_flat, m->buf_enc_hidden, ly.ca_v_proj, nullptr,
                     enc_S, kv_dim, H, m->cublas);
        reshape_to_heads(m->buf_v, v_flat, enc_S, Nkv, D);
        cudaMemcpyAsync(cached_v, m->buf_v, (int64_t)Nkv * enc_S * D * sizeof(bf16), cudaMemcpyDeviceToDevice);
    }

    // GQA expand K, V for cross-attention
    // K: [Nkv, enc_S, D] -> [Nh, enc_S, D]
    gqa_expand(m->buf_k_exp, cached_k, Nh, Nkv, enc_S, D);
    gqa_expand(m->buf_v_exp, cached_v, Nh, Nkv, enc_S, D);

    // Q @ K^T -> scores [Nh, S, enc_S] (f32)
    // Need different GEMM: Q [Nh, S, D] x K^T [Nh, D, enc_S] -> [Nh, S, enc_S]
    {
        float alpha = 1.0f, beta = 0.0f;
        cublasGemmStridedBatchedEx(m->cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            enc_S, S, D,
            &alpha,
            m->buf_k_exp, CUDA_R_16BF, D, (int64_t)enc_S * D,
            m->buf_q, CUDA_R_16BF, D, (int64_t)S * D,
            &beta,
            m->buf_scores, CUDA_R_32F, enc_S, (int64_t)S * enc_S,
            Nh,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }

    // Softmax (full attention, no window)
    float scale = 1.0f / sqrtf((float)D);
    softmax_attn(m->buf_scores, Nh, S, enc_S, scale, 0);

    // scores @ V -> [Nh, S, D]
    {
        // Convert scores f32 -> bf16 for GEMM compatibility
        int n_scores = Nh * S * enc_S;
        f32_to_bf16_kernel<<<(n_scores + 255) / 256, 256>>>(m->buf_scores_bf16, m->buf_scores, n_scores);

        float alpha = 1.0f, beta = 0.0f;
        cublasGemmStridedBatchedEx(m->cublas,
            CUBLAS_OP_N, CUBLAS_OP_N,
            D, S, enc_S,
            &alpha,
            m->buf_v_exp, CUDA_R_16BF, D, (int64_t)enc_S * D,
            m->buf_scores_bf16, CUDA_R_16BF, enc_S, (int64_t)S * enc_S,
            &beta,
            m->buf_q, CUDA_R_16BF, D, (int64_t)S * D,  // reuse buf_q
            Nh,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }

    // Reshape back and O projection
    reshape_from_heads(m->buf_attn_out, m->buf_q, S, Nh, D);
    linear_batch(m->buf_norm, m->buf_attn_out, ly.ca_o_proj, nullptr, S, H, Nh * D, m->cublas);
    // buf_norm now holds cross-attention output [S, H]
}

// Concatenate two tensors along last dim: [T, a_ch] + [T, b_ch] -> [T, a_ch+b_ch]
__global__ void concat_channels_kernel(bf16 *dst, const bf16 *a, const bf16 *b,
                                       int T, int a_ch, int b_ch, int total_ch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * total_ch) return;
    int t = i / total_ch;
    int ch = i % total_ch;
    dst[i] = (ch < a_ch) ? a[t * a_ch + ch] : b[t * b_ch + (ch - a_ch)];
}

// Full DiT forward pass
// Input:  xt [T, in_channels/2], noisy latent (actually out_channels = 64)
//         context_latents [T, in_channels/2], context (silence + chunk masks)
//         t_scalar, timestep value (f32)
//         encoder_hidden_states [enc_S, hidden_size], condition embeddings
//         enc_S, encoder sequence length
// Output: velocity [T, out_channels], predicted flow velocity
static void forward_dit(DiTModel *m, bf16 *xt, bf16 *context_latents,
                        float t_value, float t_r_value,
                        bf16 *encoder_hidden_states, int enc_S,
                        bf16 *output, int T,
                        const DebugDumper *dbg = nullptr, int step = -1) {
    DiTConfig &c = m->cfg;
    int H = c.hidden_size;
    int P = c.patch_size;
    int S = T / P;  // sequence length after patching (T must be divisible by P)

    if (T % P != 0) {
        fprintf(stderr, "FATAL: T=%d not divisible by patch_size=%d\n", T, P);
        exit(1);
    }

    // Allocate cross-KV cache on first call
    if (m->cross_kv_cache == nullptr || enc_S != m->cross_kv_enc_S) {
        if (m->cross_kv_cache) cudaFree(m->cross_kv_cache);
        if (m->buf_enc_hidden) cudaFree(m->buf_enc_hidden);
        int kv_dim = c.n_kv_heads * c.head_dim;
        DIT_MALLOC(m->cross_kv_cache, (int64_t)c.n_layers * 2 * enc_S * kv_dim * sizeof(bf16));
        DIT_MALLOC(m->buf_enc_hidden, (int64_t)enc_S * H * sizeof(bf16));
        m->cross_kv_enc_S = enc_S;
        m->cross_kv_valid = false;
    }

    // 1) Timestep embeddings
    // Upload t to GPU
    float t_val = t_value;
    CUDA_CHECK(cudaMemcpy(m->buf_t_f32, &t_val, sizeof(float), cudaMemcpyDefault));
    bf16 *temb_t = m->buf_temb;         // [H]
    bf16 *tproj_t = m->buf_tproj;       // [6*H]
    const DebugDumper *step0_dbg = (step == 0) ? dbg : nullptr;
    forward_temb(m, m->time_embed, m->buf_t_f32, temb_t, tproj_t, step0_dbg, "_t");

    // t_r embedding (t - t_r, but in turbo mode t_r = t, so t - t_r = 0)
    float tr_val = t_value - t_r_value;
    CUDA_CHECK(cudaMemcpy(m->buf_t_f32, &tr_val, sizeof(float), cudaMemcpyDefault));
    bf16 *temb_r = m->buf_temb_r;      // [H] (dedicated buffer, avoids aliasing with buf_temb_tmp)
    // Need second tproj buffer, reuse buf_sinusoidal area? No, too small.
    // Use part of buf_gate for tproj_r (6*H bf16 = 6*2048*2 = 24KB, buf_gate = S*I*2 >> that)
    bf16 *tproj_r = m->buf_gate;        // [6*H] temp
    forward_temb(m, m->time_embed_r, m->buf_t_f32, temb_r, tproj_r, step0_dbg, "_r");

    // Dump individual temb_t/temb_r before combining
    if (step == 0 && dbg && dbg->enabled) {
        debug_dump_bf16_2d(dbg, "temb_t", temb_t, 1, H);
        debug_dump_bf16_2d(dbg, "temb_r", temb_r, 1, H);
    }

    // Combine: temb = temb_t + temb_r, tproj = tproj_t + tproj_r
    add_2d(temb_t, temb_r, H);
    add_2d(tproj_t, tproj_r, 6 * H);
    // temb_t = combined temb [H], tproj_t = combined tproj [6, H]

    if (step == 0 && dbg && dbg->enabled) {
        debug_dump_bf16_2d(dbg, "tproj", tproj_t, 6, H);
        debug_dump_bf16_2d(dbg, "temb", temb_t, 1, H);
    }

    // 2) Concatenate context_latents [T, ctx_ch] and xt [T, out_ch] -> [T, in_ch]
    // Python: hidden_states = torch.cat([context_latents, hidden_states], dim=-1)
    {
        int ctx_ch = c.in_channels - c.out_channels;  // 192 - 64 = 128
        int n = T * c.in_channels;
        concat_channels_kernel<<<(n + 255) / 256, 256>>>(
            m->buf_concat, context_latents, xt, T, ctx_ch, c.out_channels, c.in_channels);
    }

    // proj_in: treat Conv1d(in_ch, hidden, k=2, s=2) as linear on patches
    // Input [T, in_channels] viewed as [T/2, in_channels*2] (naturally contiguous)
    // Weight [hidden, in_channels, 2] viewed as [hidden, in_channels*2]
    int in_dim = c.in_channels * P;
    linear_batch(m->buf_hidden, m->buf_concat, m->proj_in_w, nullptr, S, H, in_dim, m->cublas);
    add_bias_2d(m->buf_hidden, m->proj_in_b, S, H);
    // buf_hidden: [S, H]
    if (step == 0 && dbg && dbg->enabled) {
        debug_dump_bf16_2d(dbg, "hidden_after_proj_in", m->buf_hidden, S, H);
        // Print tproj first4 values (bf16 on GPU)
        {
            float tmp[4];
            std::vector<float> buf(6 * H);
            // Convert tproj bf16 -> f32 to host
            bf16 tmp16[4];
            cudaMemcpy(tmp16, tproj_t, 4 * sizeof(bf16), cudaMemcpyDeviceToHost);
            for (int i = 0; i < 4; i++) tmp[i] = __bfloat162float(tmp16[i]);
            fprintf(stderr, "[Debug] tproj first4: %.6f %.6f %.6f %.6f\n", tmp[0], tmp[1], tmp[2], tmp[3]);
            cudaMemcpy(tmp16, temb_t, 4 * sizeof(bf16), cudaMemcpyDeviceToHost);
            for (int i = 0; i < 4; i++) tmp[i] = __bfloat162float(tmp16[i]);
            fprintf(stderr, "[Debug] temb first4: %.6f %.6f %.6f %.6f\n", tmp[0], tmp[1], tmp[2], tmp[3]);
        }
    }

    // 3) Project encoder hidden states
    if (!m->cross_kv_valid) {
        linear_batch(m->buf_enc_hidden, encoder_hidden_states, m->cond_emb_w, nullptr,
                     enc_S, H, H, m->cublas);
        add_bias_2d(m->buf_enc_hidden, m->cond_emb_b, enc_S, H);
        if (step == 0 && dbg && dbg->enabled)
            debug_dump_bf16_2d(dbg, "enc_after_cond_emb", m->buf_enc_hidden, enc_S, H);
    }

    // 4) Process through DiT layers
    for (int l = 0; l < c.n_layers; l++) {
        DiTLayer &ly = m->layers[l];
        int window = (ly.layer_type == 0) ? c.sliding_window : 0;

        // Extract AdaLN parameters from scale_shift_table + tproj
        // scale_shift_table: [6, H], tproj: [6, H]
        // sum -> [6, H], then chunk into 6 vectors of [H]
        bf16 *adaln = m->buf_adaln;  // dedicated [6*H] buffer, safe from aliasing
        copy_2d(adaln, ly.scale_shift_table, 6 * H);
        add_2d(adaln, tproj_t, 6 * H);

        bf16 *shift_msa   = adaln + 0 * H;
        bf16 *scale_msa   = adaln + 1 * H;
        bf16 *gate_msa    = adaln + 2 * H;
        bf16 *c_shift_msa = adaln + 3 * H;
        bf16 *c_scale_msa = adaln + 4 * H;
        bf16 *c_gate_msa  = adaln + 5 * H;

        // Step 1: Self-attention with AdaLN
        // Save residual
        copy_2d(m->buf_residual, m->buf_hidden, S * H);

        // RMSNorm + AdaLN modulate: buf_norm = norm(hidden) * (1 + scale) + shift
        rmsnorm_2d(m->buf_norm, m->buf_hidden, ly.self_attn_norm, S, H, c.rms_norm_eps);
        adaln_modulate(m->buf_norm, m->buf_norm, scale_msa, shift_msa, S, H);

        if (l == 0 && step == 0 && dbg && dbg->enabled)
            debug_dump_bf16_2d(dbg, "layer0_sa_input", m->buf_norm, S, H);

        // Self-attention (writes result to buf_norm)
        dit_self_attention(m, ly, S, window, dbg, l, step);

        if (l == 0 && step == 0 && dbg && dbg->enabled)
            debug_dump_bf16_2d(dbg, "layer0_sa_output", m->buf_norm, S, H);

        // Gated residual: hidden = residual + attn_output * gate_msa
        copy_2d(m->buf_hidden, m->buf_residual, S * H);
        gated_add(m->buf_hidden, m->buf_norm, gate_msa, S, H);

        if (l == 0 && step == 0 && dbg && dbg->enabled)
            debug_dump_bf16_2d(dbg, "layer0_after_self_attn", m->buf_hidden, S, H);

        // Step 2: Cross-attention (always, every layer)
        rmsnorm_2d(m->buf_norm, m->buf_hidden, ly.cross_attn_norm, S, H, c.rms_norm_eps);
        dit_cross_attention(m, ly, l, S);
        // Residual add (no gate for cross-attention)
        add_2d(m->buf_hidden, m->buf_norm, S * H);

        if (l == 0 && step == 0 && dbg && dbg->enabled)
            debug_dump_bf16_2d(dbg, "layer0_after_cross_attn", m->buf_hidden, S, H);

        // Step 3: MLP with AdaLN
        // c_shift_msa, c_scale_msa, c_gate_msa still valid in buf_adaln (dedicated buffer)

        copy_2d(m->buf_residual, m->buf_hidden, S * H);
        rmsnorm_2d(m->buf_norm, m->buf_hidden, ly.mlp_norm, S, H, c.rms_norm_eps);
        adaln_modulate(m->buf_norm, m->buf_norm, c_scale_msa, c_shift_msa, S, H);

        // MLP: SwiGLU
        linear_batch(m->buf_gate, m->buf_norm, ly.gate_proj, nullptr, S, c.intermediate_size, H, m->cublas);
        linear_batch(m->buf_up,   m->buf_norm, ly.up_proj,   nullptr, S, c.intermediate_size, H, m->cublas);
        silu_mul_2d(m->buf_gate, m->buf_up, S, c.intermediate_size);
        linear_batch(m->buf_norm, m->buf_gate, ly.down_proj, nullptr, S, H, c.intermediate_size, m->cublas);

        // Gated residual: hidden = residual + mlp_output * c_gate
        copy_2d(m->buf_hidden, m->buf_residual, S * H);
        gated_add(m->buf_hidden, m->buf_norm, c_gate_msa, S, H);

        if (l == 0 && step == 0 && dbg && dbg->enabled)
            debug_dump_bf16_2d(dbg, "hidden_after_layer0", m->buf_hidden, S, H);
    }

    // Mark cross-KV cache as valid after first complete forward pass
    m->cross_kv_valid = true;

    // 5) Output: AdaLN + proj_out
    // scale_shift_table for output: [2, H], shift, scale
    // temb [H] -> unsqueeze to [1, H], add to scale_shift_table [2, H]
    bf16 *out_ss = m->buf_proj_tmp;  // [2*H]
    copy_2d(out_ss, m->out_scale_shift, 2 * H);
    // Add temb to each of the 2 rows
    add_2d(out_ss, temb_t, H);           // shift += temb
    add_2d(out_ss + H, temb_t, H);       // scale += temb

    bf16 *out_shift = out_ss;
    bf16 *out_scale = out_ss + H;

    rmsnorm_2d(m->buf_norm, m->buf_hidden, m->norm_out, S, H, c.rms_norm_eps);
    adaln_modulate(m->buf_norm, m->buf_norm, out_scale, out_shift, S, H);

    // proj_out: ConvTranspose1d(hidden, out_channels, k=P, s=P)
    // Weight after permute: [hidden, P*out_channels] = [H, P*Oc]
    // Linear: [S, H] @ W[H, P*Oc] -> [S, P*Oc]  (no transpose, W is [in, out])
    int out_dim = c.out_channels * P;
    linear_batch_nt(m->buf_proj_tmp, m->buf_norm, m->proj_out_w, S, out_dim, H, m->cublas);
    deinterleave(output, m->buf_proj_tmp, S, c.out_channels, P);
    add_bias_2d(output, m->proj_out_b, T, c.out_channels);
}

// Invalidate cross-attention KV cache (call between flow steps
// that change encoder hidden states, e.g. cover->non-cover switch)
static void dit_invalidate_cross_kv(DiTModel *m) {
    m->cross_kv_valid = false;
}

// Flow matching: single Euler ODE step
// xt_next = xt - vt * dt
__global__ void euler_step_kernel(bf16 *xt, const bf16 *vt, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = __bfloat162float(xt[i]);
    float v = __bfloat162float(vt[i]);
    xt[i] = __float2bfloat16(x - v * dt);
}

static void euler_step(bf16 *xt, const bf16 *vt, float dt, int n) {
    euler_step_kernel<<<(n + 255) / 256, 256>>>(xt, vt, dt, n);
}

// Get x0 from noise: x0 = zt - vt * t
__global__ void get_x0_kernel(bf16 *x0, const bf16 *zt, const bf16 *vt, float t, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float z = __bfloat162float(zt[i]);
    float v = __bfloat162float(vt[i]);
    x0[i] = __float2bfloat16(z - v * t);
}

static void get_x0_from_noise(bf16 *x0, const bf16 *zt, const bf16 *vt, float t, int n) {
    get_x0_kernel<<<(n + 255) / 256, 256>>>(x0, zt, vt, t, n);
}

// Full flow matching generation loop (8 steps turbo)
static void dit_generate(DiTModel *m,
                         bf16 *noise,              // [T, out_channels] initial noise
                         bf16 *context_latents,    // [T, in_channels - out_channels] context (128ch)
                         bf16 *encoder_hidden_states, // [enc_S, hidden_size]
                         int enc_S, int T,
                         const float *t_schedule,  // [num_steps] descending timesteps
                         int num_steps,
                         bf16 *output,             // [T, out_channels]
                         const DebugDumper *dbg = nullptr) {
    int out_ch = m->cfg.out_channels;
    int n = T * out_ch;

    // xt starts as noise
    bf16 *xt = noise;  // in-place modification

    // Velocity output buffer
    bf16 *vt = m->buf_vt;  // [T, out_channels]

    dit_invalidate_cross_kv(m);

    for (int step = 0; step < num_steps; step++) {
        float t_curr = t_schedule[step];

        forward_dit(m, xt, context_latents, t_curr, t_curr,
                    encoder_hidden_states, enc_S, vt, T, dbg, step);

        // debug dump vt
        if (dbg && dbg->enabled) {
            char name[64];
            snprintf(name, sizeof(name), "dit_step%d_vt", step);
            debug_dump_bf16_2d(dbg, name, vt, T, out_ch);
        }

        if (step == num_steps - 1) {
            // Final step: x0 = zt - vt * t
            get_x0_from_noise(output, xt, vt, t_curr, n);

            if (dbg && dbg->enabled) {
                debug_dump_bf16_2d(dbg, "dit_x0", output, T, out_ch);
            }
        } else {
            // Euler step: xt = xt - vt * dt
            float dt = t_curr - t_schedule[step + 1];
            euler_step(xt, vt, dt, n);

            if (dbg && dbg->enabled) {
                char name[64];
                snprintf(name, sizeof(name), "dit_step%d_xt", step);
                debug_dump_bf16_2d(dbg, name, xt, T, out_ch);
            }
        }
    }
}


