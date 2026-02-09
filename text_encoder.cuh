#pragma once
// ACE-Step Text Encoder (text_encoder.cuh)
// Qwen3-Embedding-0.6B: 28L causal transformer
// Config: hidden=1024, inter=3072, heads=16/8, head_dim=128
// Also provides lyric embedding (just embed_tokens, no forward)
//
// Requires: kernels.cuh, dit.cuh, transformer.cuh

// Token embedding lookup kernel
// in: token_ids [S] (int32), embed [vocab_size, H]
// out: output [S, H]
__global__ void embed_tokens_kernel(bf16 *output, const int *token_ids,
                                    const bf16 *embed, int S, int H) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= S * H) return;
    int s = idx / H;
    int d = idx % H;
    int token = token_ids[s];
    output[idx] = embed[(int64_t)token * H + d];
}

static void embed_tokens_lookup(bf16 *output, const int *token_ids_gpu,
                                const bf16 *embed, int S, int H) {
    int n = S * H;
    embed_tokens_kernel<<<(n + 255) / 256, 256>>>(output, token_ids_gpu, embed, S, H);
}

// Text Encoder model
struct TextEncoder {
    EncModel enc;            // 28 layers causal transformer
    bf16 *embed_tokens_w;    // [151669, 1024]
    bf16 *norm_w;            // [1024]
    int vocab_size;          // 151669
    int hidden_size;         // 1024

    // GPU buffer for token IDs
    int *token_ids_gpu;      // [max_seq_len]
};

static EncConfig text_encoder_config(int max_seq_len) {
    return {
        .hidden_size       = 1024,
        .intermediate_size = 3072,
        .n_heads           = 16,
        .n_kv_heads        = 8,
        .head_dim          = 128,
        .n_layers          = 28,
        .sliding_window    = 0,       // no sliding window
        .rope_theta        = 1000000.0f,
        .rms_norm_eps      = 1e-6f,
        .max_seq_len       = max_seq_len,
        .is_causal         = true,    // causal attention
    };
}

static void text_encoder_load(TextEncoder *te, const char *model_dir, int max_seq_len) {
    te->enc.cfg = text_encoder_config(max_seq_len);
    te->vocab_size = 151669;
    te->hidden_size = 1024;
    cublasCreate(&te->enc.cublas);
    enc_alloc_buffers(&te->enc);

    SafeTensors st;
    if (!safe_load(st, model_dir)) {
        fprintf(stderr, "FATAL: cannot load text encoder from %s\n", model_dir);
        exit(1);
    }
    fprintf(stderr, "[TextEncoder] Loading from %s (%zu tensors)...\n", model_dir, st.tensors.size());

    // Layers: "layers.0", "layers.1", ...
    enc_load_layers(&te->enc, st, "layers");

    // Embedding table: [vocab_size, 1024]
    te->embed_tokens_w = must_upload(st, "embed_tokens.weight");

    // Final norm
    te->norm_w = must_upload(st, "norm.weight");

    // Token ID buffer on GPU
    cudaMalloc(&te->token_ids_gpu, max_seq_len * sizeof(int));

    // st destructor will munmap files (weights already copied to GPU)
    fprintf(stderr, "[TextEncoder] Loaded (28 layers, hidden=%d, vocab=%d, max_seq=%d)\n",
            te->hidden_size, te->vocab_size, max_seq_len);
}

// Full text encoder forward
// Input: token_ids [S] (int32 on CPU), S = sequence length
// Output: te->enc.buf_hidden [S, 1024]
static void text_encoder_forward(TextEncoder *te, const int *token_ids_cpu, int S) {
    int H = te->hidden_size;

    // Upload token IDs to GPU
    cudaMemcpy(te->token_ids_gpu, token_ids_cpu, S * sizeof(int), cudaMemcpyHostToDevice);

    // Token embedding lookup: [S] -> [S, 1024]
    embed_tokens_lookup(te->enc.buf_hidden, te->token_ids_gpu, te->embed_tokens_w, S, H);

    // 28 transformer layers (causal) + final norm
    enc_forward_with_norm(&te->enc, S, te->norm_w);

    // Output in te->enc.buf_hidden [S, 1024]
}

// Lyric embedding: just embed_tokens lookup, no transformer forward
// Input: token_ids [S] (int32 on CPU)
// Output: dst [S, 1024]
static void lyric_embed(TextEncoder *te, const int *token_ids_cpu, int S, bf16 *dst) {
    // Upload token IDs
    cudaMemcpy(te->token_ids_gpu, token_ids_cpu, S * sizeof(int), cudaMemcpyHostToDevice);

    // Embedding lookup -> dst [S, 1024]
    embed_tokens_lookup(dst, te->token_ids_gpu, te->embed_tokens_w, S, te->hidden_size);
}
