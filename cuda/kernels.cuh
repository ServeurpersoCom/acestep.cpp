#pragma once
// CUDA kernels for Qwen3 transformer inference (bf16 storage, fp32 compute)
// Designed for reuse across LM, DiT, and encoder models.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#define CUDA_CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

typedef __nv_bfloat16 bf16;

// Embedding lookup: out[dim] = table[id * dim .. (id+1)*dim - 1]
__global__ void embed_lookup_kernel(bf16 *out, const bf16 *table, int id, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) out[i] = table[(int64_t)id * dim + i];
}

static void embed_lookup(bf16 *out, const bf16 *table, int id, int dim) {
    embed_lookup_kernel<<<(dim + 255) / 256, 256>>>(out, table, id, dim);
}

// RMSNorm: out[i] = (x[i] / rms) * w[i]
// Single block, threads reduce to compute variance
__global__ void rmsnorm_kernel(bf16 *out, const bf16 *x, const bf16 *w, int n, float eps) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = tid; i < n; i += nthreads) {
        float v = __bfloat162float(x[i]);
        sum_sq += v * v;
    }
    sdata[tid] = sum_sq;
    __syncthreads();

    // Tree reduction
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float rms_inv = rsqrtf(sdata[0] / (float)n + eps);

    // Normalize and scale
    for (int i = tid; i < n; i += nthreads)
        out[i] = __float2bfloat16(__bfloat162float(x[i]) * rms_inv * __bfloat162float(w[i]));
}

static void rmsnorm(bf16 *out, const bf16 *x, const bf16 *w, int n, float eps) {
    int threads = (n < 1024) ? n : 1024;
    rmsnorm_kernel<<<1, threads, threads * sizeof(float)>>>(out, x, w, n, eps);
}

// Per-head RMSNorm for QK-norm: operates on [n_heads, head_dim]
// One block per head
__global__ void head_rmsnorm_kernel(bf16 *out, const bf16 *x, const bf16 *w,
                                     int n_heads, int head_dim, float eps) {
    int h = blockIdx.x;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    const bf16 *xh = x + h * head_dim;
    bf16 *oh = out + h * head_dim;

    extern __shared__ float sdata[];
    float sum_sq = 0.0f;
    for (int i = tid; i < head_dim; i += nthreads) {
        float v = __bfloat162float(xh[i]);
        sum_sq += v * v;
    }
    sdata[tid] = sum_sq;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float rms_inv = rsqrtf(sdata[0] / (float)head_dim + eps);

    for (int i = tid; i < head_dim; i += nthreads)
        oh[i] = __float2bfloat16(__bfloat162float(xh[i]) * rms_inv * __bfloat162float(w[i]));
}

static void head_rmsnorm(bf16 *out, const bf16 *x, const bf16 *w,
                          int n_heads, int head_dim, float eps) {
    int threads = (head_dim < 256) ? head_dim : 256;
    head_rmsnorm_kernel<<<n_heads, threads, threads * sizeof(float)>>>(
        out, x, w, n_heads, head_dim, eps);
}

// RoPE: apply rotary position embeddings to Q[n_heads, head_dim]
// and K[n_kv_heads, head_dim] in-place
// Qwen3 uses "half-split" RoPE: pairs are (d, d+half) not (2d, 2d+1)
// rotate_half(x) = cat(-x[half:], x[:half])
// x' = x * cos + rotate_half(x) * sin
__global__ void rope_kernel(bf16 *q, bf16 *k, int pos,
                            int n_q_heads, int n_kv_heads, int head_dim, float theta) {
    int h = blockIdx.x;  // head index
    int d = threadIdx.x;  // dim index (0 .. head_dim/2 - 1)
    if (d >= head_dim / 2) return;

    int half = head_dim / 2;
    float freq = 1.0f / powf(theta, (float)(2 * d) / (float)head_dim);
    float angle = (float)pos * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    // Apply to Q heads: q'[d] = q[d]*cos - q[d+half]*sin
    //                   q'[d+half] = q[d+half]*cos + q[d]*sin
    if (h < n_q_heads) {
        int base = h * head_dim;
        float first  = __bfloat162float(q[base + d]);
        float second = __bfloat162float(q[base + d + half]);
        q[base + d]        = __float2bfloat16(first * cos_a - second * sin_a);
        q[base + d + half] = __float2bfloat16(second * cos_a + first * sin_a);
    }

    // Apply to K heads
    if (h < n_kv_heads) {
        int base = h * head_dim;
        float first  = __bfloat162float(k[base + d]);
        float second = __bfloat162float(k[base + d + half]);
        k[base + d]        = __float2bfloat16(first * cos_a - second * sin_a);
        k[base + d + half] = __float2bfloat16(second * cos_a + first * sin_a);
    }
}

static void rope(bf16 *q, bf16 *k, int pos,
                 int n_q_heads, int n_kv_heads, int head_dim, float theta) {
    int max_heads = (n_q_heads > n_kv_heads) ? n_q_heads : n_kv_heads;
    rope_kernel<<<max_heads, head_dim / 2>>>(q, k, pos, n_q_heads, n_kv_heads, head_dim, theta);
}

// SiLU * mul: out[i] = (gate[i] * sigmoid(gate[i])) * up[i]
// Used for SwiGLU FFN
__global__ void silu_mul_kernel(bf16 *out, const bf16 *gate, const bf16 *up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = __bfloat162float(gate[i]);
        float u = __bfloat162float(up[i]);
        float silu = g / (1.0f + expf(-g));
        out[i] = __float2bfloat16(silu * u);
    }
}

static void silu_mul(bf16 *out, const bf16 *gate, const bf16 *up, int n) {
    silu_mul_kernel<<<(n + 255) / 256, 256>>>(out, gate, up, n);
}

// Element-wise add: out[i] = a[i] + b[i]
__global__ void add_kernel(bf16 *out, const bf16 *a, const bf16 *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = __float2bfloat16(__bfloat162float(a[i]) + __bfloat162float(b[i]));
}

static void add_inplace(bf16 *a, const bf16 *b, int n) {
    add_kernel<<<(n + 255) / 256, 256>>>(a, a, b, n);
}

// Copy kernel: dst = src
__global__ void copy_kernel(bf16 *dst, const bf16 *src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

static void copy_bf16(bf16 *dst, const bf16 *src, int n) {
    copy_kernel<<<(n + 255) / 256, 256>>>(dst, src, n);
}

// KV cache write: store K and V vectors at given position
// K_cache/V_cache layout: [max_seq_len, n_kv_heads * head_dim]
__global__ void kv_store_kernel(bf16 *cache, const bf16 *vec, int pos, int kv_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kv_dim) cache[pos * kv_dim + i] = vec[i];
}

static void kv_store(bf16 *cache, const bf16 *vec, int pos, int kv_dim) {
    kv_store_kernel<<<(kv_dim + 255) / 256, 256>>>(cache, vec, pos, kv_dim);
}

// Decode GQA attention: single query token attending to all cached KV
// One block per Q head, computes scores->softmax->weighted sum
// Uses shared memory for attention scores (limited to MAX_SEQ shared mem)
#define ATTN_THREADS 256

__global__ void decode_attention_kernel(
    bf16 *output,            // [n_q_heads * head_dim]
    const bf16 *Q,           // [n_q_heads * head_dim]
    const bf16 *K_cache,     // [max_seq * kv_dim]
    const bf16 *V_cache,     // [max_seq * kv_dim]
    int seq_len, int n_q_heads, int n_kv_heads, int head_dim, int kv_dim, float scale)
{
    int h = blockIdx.x;
    int tid = threadIdx.x;
    int kv_h = h * n_kv_heads / n_q_heads;

    extern __shared__ float smem[];
    float *scores = smem;                    // [seq_len] (reused)
    float *reduce_buf = smem + seq_len;      // [ATTN_THREADS]

    const bf16 *q = Q + h * head_dim;

    // 1) Compute Q*K scores
    float local_max = -1e30f;
    for (int j = tid; j < seq_len; j += ATTN_THREADS) {
        const bf16 *k = K_cache + (int64_t)j * kv_dim + kv_h * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++)
            dot += __bfloat162float(q[d]) * __bfloat162float(k[d]);
        scores[j] = dot * scale;
        local_max = fmaxf(local_max, scores[j]);
    }

    // Reduce max across threads
    reduce_buf[tid] = local_max;
    __syncthreads();
    for (int s = ATTN_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) reduce_buf[tid] = fmaxf(reduce_buf[tid], reduce_buf[tid + s]);
        __syncthreads();
    }
    float max_val = reduce_buf[0];

    // 2) Exp and sum (softmax numerator + denominator)
    float local_sum = 0.0f;
    for (int j = tid; j < seq_len; j += ATTN_THREADS) {
        scores[j] = expf(scores[j] - max_val);
        local_sum += scores[j];
    }
    reduce_buf[tid] = local_sum;
    __syncthreads();
    for (int s = ATTN_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) reduce_buf[tid] += reduce_buf[tid + s];
        __syncthreads();
    }
    float sum_val = reduce_buf[0];

    // Normalize scores
    float inv_sum = 1.0f / (sum_val + 1e-9f);
    for (int j = tid; j < seq_len; j += ATTN_THREADS)
        scores[j] *= inv_sum;
    __syncthreads();

    // 3) Weighted sum: output = scores @ V
    for (int d = tid; d < head_dim; d += ATTN_THREADS) {
        float acc = 0.0f;
        for (int j = 0; j < seq_len; j++)
            acc += scores[j] * __bfloat162float(V_cache[(int64_t)j * kv_dim + kv_h * head_dim + d]);
        output[h * head_dim + d] = __float2bfloat16(acc);
    }
}

static void decode_attention(bf16 *output, const bf16 *Q,
                             const bf16 *K_cache, const bf16 *V_cache,
                             int seq_len, int n_q_heads, int n_kv_heads,
                             int head_dim, float scale) {
    int kv_dim = n_kv_heads * head_dim;
    // Shared memory: scores[seq_len] + reduce_buf[ATTN_THREADS]
    size_t smem = (seq_len + ATTN_THREADS) * sizeof(float);
    decode_attention_kernel<<<n_q_heads, ATTN_THREADS, smem>>>(
        output, Q, K_cache, V_cache, seq_len, n_q_heads, n_kv_heads, head_dim, kv_dim, scale);
}


