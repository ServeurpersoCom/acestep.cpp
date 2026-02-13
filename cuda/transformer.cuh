#pragma once
// Generic bidirectional transformer encoder block
// Reused by: LyricEncoder(8L), TimbreEncoder(4L), AttentionPooler(2L), Detokenizer(2L)
//
// Requires: kernels.cuh + dit.cuh included before this file
//   (uses linear_batch, reshape_to/from_heads, qknorm_2d, rope_batch,
//    gqa_expand, batch_qk_gemm, softmax_attn, batch_sv_gemm,
//    rmsnorm_2d, silu_mul_2d, copy_2d, add_2d)

// Encoder layer weights (11 tensors per layer)
// Same structure as AceStepEncoderLayer in Python:
//   input_layernorm -> self_attn -> residual -> post_attention_layernorm -> mlp -> residual
struct EncLayerWeights {
    // Pre-attention norm
    bf16 *input_layernorm;     // [H]

    // Self-attention (GQA: 16 Q heads, 8 KV heads, head_dim=128)
    bf16 *q_proj;              // [Nh*D, H]
    bf16 *k_proj;              // [Nkv*D, H]
    bf16 *v_proj;              // [Nkv*D, H]
    bf16 *q_norm;              // [D]
    bf16 *k_norm;              // [D]
    bf16 *o_proj;              // [H, Nh*D]

    // Post-attention norm
    bf16 *post_attn_layernorm; // [H]

    // MLP (SwiGLU)
    bf16 *gate_proj;           // [I, H]
    bf16 *up_proj;             // [I, H]
    bf16 *down_proj;           // [H, I]

    int layer_type;            // 0=sliding, 1=full
};

// Encoder config (shared dimensions for all encoder components)
struct EncConfig {
    int hidden_size;        // 2048
    int intermediate_size;  // 6144
    int n_heads;            // 16
    int n_kv_heads;         // 8
    int head_dim;           // 128
    int n_layers;           // varies: 8, 4, or 2
    int sliding_window;     // 128
    float rope_theta;       // 1000000
    float rms_norm_eps;     // 1e-6
    int max_seq_len;        // max S for buffer sizing
    bool is_causal;         // true for Qwen3-Embedding text encoder
};

static EncConfig enc_default_config(int n_layers, int max_seq_len) {
    return {
        .hidden_size       = 2048,
        .intermediate_size = 6144,
        .n_heads           = 16,
        .n_kv_heads        = 8,
        .head_dim          = 128,
        .n_layers          = n_layers,
        .sliding_window    = 128,
        .rope_theta        = 1000000.0f,
        .rms_norm_eps      = 1e-6f,
        .max_seq_len       = max_seq_len,
        .is_causal         = false,
    };
}

// Causal softmax: same as softmax_attn but masks j > s_q
// Combined with optional sliding window
__global__ void softmax_attn_causal_kernel(float *scores, int S_q, int S_kv,
                                           int window, float scale) {
    int row_idx = blockIdx.x;
    int s_q = row_idx % S_q;
    float *row = scores + (int64_t)row_idx * S_kv;

    extern __shared__ float smem[];

    float local_max = -1e30f;
    for (int j = threadIdx.x; j < S_kv; j += blockDim.x) {
        float v = row[j] * scale;
        // Causal mask: can only attend to positions <= current
        if (j > s_q) v = -1e30f;
        // Optional sliding window on top of causal
        if (window > 0 && j < s_q - window) v = -1e30f;
        row[j] = v;
        local_max = fmaxf(local_max, v);
    }
    smem[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        __syncthreads();
    }
    float max_val = smem[0];

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

    for (int j = threadIdx.x; j < S_kv; j += blockDim.x)
        row[j] *= inv_sum;
}

static void softmax_attn_causal(float *scores, int batch, int S_q, int S_kv,
                                float scale, int window) {
    int rows = batch * S_q;
    int threads = (S_kv < 256) ? S_kv : 256;
    int t = 1; while (t < threads) t <<= 1; threads = t;
    if (threads > 256) threads = 256;
    softmax_attn_causal_kernel<<<rows, threads, threads * sizeof(float)>>>(
        scores, S_q, S_kv, window, scale);
}

// Encoder model: weights + scratch buffers
#define ENC_MAX_LAYERS 32

struct EncModel {
    EncConfig cfg;
    EncLayerWeights layers[ENC_MAX_LAYERS];

    // Scratch buffers (allocated once for max_seq_len)
    bf16 *buf_hidden;    // [S, H], current hidden states
    bf16 *buf_norm;      // [S, H], after layernorm
    bf16 *buf_residual;  // [S, H]
    bf16 *buf_q;         // [Nh, S, D]
    bf16 *buf_k;         // [Nkv, S, D]
    bf16 *buf_v;         // [Nkv, S, D]
    bf16 *buf_k_exp;     // [Nh, S, D], GQA expanded K
    bf16 *buf_v_exp;     // [Nh, S, D], GQA expanded V
    float *buf_scores;   // [Nh, S, S], attention scores (f32)
    bf16 *buf_attn_out;  // [S, Nh*D]
    bf16 *buf_gate;      // [S, I]
    bf16 *buf_up;        // [S, I]
    bf16 *buf_tmp;       // [S, max(H, Nkv*D)], temp for projections
    bf16 *buf_scores_bf16; // [Nh, S, S], bf16 scratch for SV GEMM

    cublasHandle_t cublas;
};

// Allocate scratch buffers
static void enc_alloc_buffers(EncModel *m) {
    EncConfig &c = m->cfg;
    int S = c.max_seq_len;
    int H = c.hidden_size;
    int I = c.intermediate_size;
    int Nh = c.n_heads;
    int Nkv = c.n_kv_heads;
    int D = c.head_dim;

    auto alloc = [](bf16 **p, size_t bytes) {
        cudaMalloc(p, bytes);
    };
    auto alloc_f32 = [](float **p, size_t bytes) {
        cudaMalloc(p, bytes);
    };

    alloc(&m->buf_hidden,   (size_t)S * H * sizeof(bf16));
    alloc(&m->buf_norm,     (size_t)S * H * sizeof(bf16));
    alloc(&m->buf_residual, (size_t)S * H * sizeof(bf16));
    alloc(&m->buf_q,        (size_t)Nh * S * D * sizeof(bf16));
    alloc(&m->buf_k,        (size_t)Nkv * S * D * sizeof(bf16));
    alloc(&m->buf_v,        (size_t)Nkv * S * D * sizeof(bf16));
    alloc(&m->buf_k_exp,    (size_t)Nh * S * D * sizeof(bf16));
    alloc(&m->buf_v_exp,    (size_t)Nh * S * D * sizeof(bf16));
    alloc_f32(&m->buf_scores, (size_t)Nh * S * S * sizeof(float));
    alloc(&m->buf_attn_out, (size_t)S * Nh * D * sizeof(bf16));
    alloc(&m->buf_gate,     (size_t)S * I * sizeof(bf16));
    alloc(&m->buf_up,       (size_t)S * I * sizeof(bf16));
    alloc(&m->buf_tmp,      (size_t)S * H * sizeof(bf16));
    alloc(&m->buf_scores_bf16, (size_t)Nh * S * S * sizeof(bf16));

    fprintf(stderr, "[TransformerBlock] Buffers allocated for S=%d, H=%d, %d layers\n", S, H, c.n_layers);
}

// Load encoder layer weights from safetensors
// prefix: e.g. "encoder.lyric_encoder.layers.0"
static void enc_load_layer(EncLayerWeights &ly, SafeTensors &st,
                           const std::string &prefix, int layer_idx) {
    std::string p = prefix;

    ly.input_layernorm     = must_upload(st, p + ".input_layernorm.weight");
    ly.q_proj              = must_upload(st, p + ".self_attn.q_proj.weight");
    ly.k_proj              = must_upload(st, p + ".self_attn.k_proj.weight");
    ly.v_proj              = must_upload(st, p + ".self_attn.v_proj.weight");
    ly.q_norm              = must_upload(st, p + ".self_attn.q_norm.weight");
    ly.k_norm              = must_upload(st, p + ".self_attn.k_norm.weight");
    ly.o_proj              = must_upload(st, p + ".self_attn.o_proj.weight");
    ly.post_attn_layernorm = must_upload(st, p + ".post_attention_layernorm.weight");
    ly.gate_proj           = must_upload(st, p + ".mlp.gate_proj.weight");
    ly.up_proj             = must_upload(st, p + ".mlp.up_proj.weight");
    ly.down_proj           = must_upload(st, p + ".mlp.down_proj.weight");

    // Alternating sliding/full attention: even=sliding, odd=full
    ly.layer_type = (layer_idx % 2 == 0) ? 0 : 1;
}

// Bidirectional self-attention forward (no causal mask)
// Input:  buf_norm [S, H] (already normalized)
// Output: buf_norm [S, H] (attention output after O proj)
static void enc_self_attention(EncModel *m, EncLayerWeights &ly, int S) {
    EncConfig &c = m->cfg;
    int H = c.hidden_size;
    int Nh = c.n_heads;
    int Nkv = c.n_kv_heads;
    int D = c.head_dim;

    // QKV projections
    linear_batch(m->buf_attn_out, m->buf_norm, ly.q_proj, nullptr, S, Nh * D, H, m->cublas);
    linear_batch(m->buf_tmp,      m->buf_norm, ly.k_proj, nullptr, S, Nkv * D, H, m->cublas);
    bf16 *v_flat = m->buf_gate;  // temp (S*I >> S*Nkv*D)
    linear_batch(v_flat,          m->buf_norm, ly.v_proj, nullptr, S, Nkv * D, H, m->cublas);

    // Reshape [S, N*D] -> [N, S, D]
    reshape_to_heads(m->buf_q, m->buf_attn_out, S, Nh, D);
    reshape_to_heads(m->buf_k, m->buf_tmp, S, Nkv, D);
    reshape_to_heads(m->buf_v, v_flat, S, Nkv, D);

    // QK-Norm
    qknorm_2d(m->buf_q, ly.q_norm, Nh, S, D);
    qknorm_2d(m->buf_k, ly.k_norm, Nkv, S, D);

    // RoPE (bidirectional, sequential positions)
    rope_batch(m->buf_q, m->buf_k, Nh, Nkv, S, D, c.rope_theta);

    // GQA expand: [Nkv, S, D] -> [Nh, S, D]
    gqa_expand(m->buf_k_exp, m->buf_k, Nh, Nkv, S, D);
    gqa_expand(m->buf_v_exp, m->buf_v, Nh, Nkv, S, D);

    // Q @ K^T -> scores [Nh, S, S]
    batch_qk_gemm(m->buf_scores, m->buf_q, m->buf_k_exp, S, D, Nh, m->cublas);

    // Softmax (bidirectional: window=0 means full, >0 means sliding but NON-CAUSAL)
    // Or causal if config says so
    float scale = 1.0f / sqrtf((float)D);
    int window = (ly.layer_type == 0) ? c.sliding_window : 0;
    if (c.is_causal)
        softmax_attn_causal(m->buf_scores, Nh, S, S, scale, window);
    else
        softmax_attn(m->buf_scores, Nh, S, S, scale, window);

    // scores @ V -> [Nh, S, D]
    batch_sv_gemm(m->buf_q, m->buf_scores, m->buf_v_exp, S, D, Nh, m->cublas, m->buf_scores_bf16);

    // Reshape back [Nh, S, D] -> [S, Nh*D]
    reshape_from_heads(m->buf_attn_out, m->buf_q, S, Nh, D);

    // O projection -> buf_norm [S, H]
    linear_batch(m->buf_norm, m->buf_attn_out, ly.o_proj, nullptr, S, H, Nh * D, m->cublas);
}

// Forward one encoder layer
// hidden [S, H] in buf_hidden -> updated in-place
static void enc_forward_layer(EncModel *m, EncLayerWeights &ly, int S) {
    EncConfig &c = m->cfg;
    int H = c.hidden_size;

    // Self-attention block
    copy_2d(m->buf_residual, m->buf_hidden, S * H);
    rmsnorm_2d(m->buf_norm, m->buf_hidden, ly.input_layernorm, S, H, c.rms_norm_eps);
    enc_self_attention(m, ly, S);
    // Residual: hidden = residual + attn_output
    copy_2d(m->buf_hidden, m->buf_residual, S * H);
    add_2d(m->buf_hidden, m->buf_norm, S * H);

    // MLP block
    copy_2d(m->buf_residual, m->buf_hidden, S * H);
    rmsnorm_2d(m->buf_norm, m->buf_hidden, ly.post_attn_layernorm, S, H, c.rms_norm_eps);
    // SwiGLU: gate = silu(x @ gate_proj) * (x @ up_proj), out = gate @ down_proj
    linear_batch(m->buf_gate, m->buf_norm, ly.gate_proj, nullptr, S, c.intermediate_size, H, m->cublas);
    linear_batch(m->buf_up,   m->buf_norm, ly.up_proj,   nullptr, S, c.intermediate_size, H, m->cublas);
    silu_mul_2d(m->buf_gate, m->buf_up, S, c.intermediate_size);
    linear_batch(m->buf_norm, m->buf_gate, ly.down_proj, nullptr, S, H, c.intermediate_size, m->cublas);
    // Residual: hidden = residual + mlp_output
    copy_2d(m->buf_hidden, m->buf_residual, S * H);
    add_2d(m->buf_hidden, m->buf_norm, S * H);
}

// Forward full encoder with external final norm weight
// (callers provide the norm weight since it's stored outside the layers)
// Input:  buf_hidden [S, H]
// Output: buf_hidden [S, H]
static void enc_forward_with_norm(EncModel *m, int S, bf16 *final_norm_w) {
    for (int l = 0; l < m->cfg.n_layers; l++) {
        enc_forward_layer(m, m->layers[l], S);
    }
    // Final RMSNorm (in-place)
    rmsnorm_2d(m->buf_norm, m->buf_hidden, final_norm_w, S, m->cfg.hidden_size, m->cfg.rms_norm_eps);
    copy_2d(m->buf_hidden, m->buf_norm, S * m->cfg.hidden_size);
}

// Convenience: load N encoder layers from safetensors
// base_prefix: e.g. "encoder.lyric_encoder.layers"
static void enc_load_layers(EncModel *m, SafeTensors &st,
                            const char *base_prefix) {
    for (int i = 0; i < m->cfg.n_layers; i++) {
        char prefix[256];
        snprintf(prefix, sizeof(prefix), "%s.%d", base_prefix, i);
        enc_load_layer(m->layers[i], st, prefix, i);
    }
    fprintf(stderr, "[TransformerBlock] Loaded %d layers from %s.*\n", m->cfg.n_layers, base_prefix);
}

// Embed + project input: [S, in_dim] -> buf_hidden [S, H]
// For embed_tokens layers with weight [H, in_dim] + bias [H]
static void enc_embed_input(EncModel *m, bf16 *input, bf16 *embed_w, bf16 *embed_b,
                            int S, int in_dim) {
    linear_batch(m->buf_hidden, input, embed_w, nullptr, S, m->cfg.hidden_size, in_dim, m->cublas);
    add_bias_2d(m->buf_hidden, embed_b, S, m->cfg.hidden_size);
}
