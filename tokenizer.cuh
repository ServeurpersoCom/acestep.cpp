#pragma once
// Tokenizer (25Hz -> 5Hz discrete) + Detokenizer (5Hz -> 25Hz continuous)
// Requires: kernels.cuh, dit.cuh, transformer.cuh included before this file

// FSQ (Finite Scalar Quantization)
// levels = [8,8,8,5,5,5], project_in [2048->6], project_out [6->2048]
struct FSQ {
    bf16 *project_in_w;    // [6, 2048]
    bf16 *project_in_b;    // [6]
    bf16 *project_out_w;   // [2048, 6]
    bf16 *project_out_b;   // [2048]

    // Scratch (on GPU)
    bf16 *buf_6;           // [max_T, 6], projected low-dim
    bf16 *buf_quant;       // [max_T, 6], quantized low-dim
};

// FSQ levels
static const int FSQ_NDIMS = 6;
static const int FSQ_LEVELS[FSQ_NDIMS] = {8, 8, 8, 5, 5, 5};
// Total codebook size = 8*8*8*5*5*5 = 64000

// FSQ decode indices: indices [N] -> quantized [N, 6]
__global__ void fsq_decode_indices_kernel(bf16 *quant, const int *indices, int N,
                                           int d0, int d1, int d2, int d3, int d4, int d5) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    int levels[FSQ_NDIMS] = {d0, d1, d2, d3, d4, d5};
    int idx = indices[n];
    bf16 *row = quant + n * FSQ_NDIMS;

    int stride = 1;
    for (int d = 0; d < FSQ_NDIMS; d++) {
        int L = levels[d];
        int level_idx = (idx / stride) % L;
        float half_L = (float)(L - 1) / 2.0f;
        row[d] = __float2bfloat16((float)level_idx / half_L - 1.0f);
        stride *= L;
    }
}

static void fsq_decode_indices(bf16 *quant, const int *indices, int N) {
    fsq_decode_indices_kernel<<<(N + 255) / 256, 256>>>(
        quant, indices, N, FSQ_LEVELS[0], FSQ_LEVELS[1], FSQ_LEVELS[2],
        FSQ_LEVELS[3], FSQ_LEVELS[4], FSQ_LEVELS[5]);
}

// Detokenizer: 5Hz discrete tokens -> 25Hz continuous
struct Detokenizer {
    EncModel detok_enc;
    bf16 *embed_w;             // [2048, 2048]
    bf16 *embed_b;             // [2048]
    bf16 *norm_w;              // [2048]
    bf16 *special_tokens;      // [5, 2048] (positional)
    bf16 *proj_out_w;          // [64, 2048]
    bf16 *proj_out_b;          // [64]

    // Scratch
    bf16 *buf_embedded;        // [5, 2048], per-group
    bf16 *buf_out;             // [max_T_25Hz, 64], final output

    int pool_window;           // 5
    int max_T_5Hz;

    cublasHandle_t cublas;
};

// Load Detokenizer
static void load_detokenizer(Detokenizer *detok, SafeTensors &st, int max_T_5Hz,
                             cublasHandle_t cublas) {
    int H = 2048;
    detok->pool_window = 5;
    detok->max_T_5Hz = max_T_5Hz;
    detok->cublas = cublas;

    detok->embed_w = must_upload(st, "detokenizer.embed_tokens.weight");
    detok->embed_b = must_upload(st, "detokenizer.embed_tokens.bias");
    detok->norm_w  = must_upload(st, "detokenizer.norm.weight");
    detok->special_tokens = must_upload(st, "detokenizer.special_tokens");
    detok->proj_out_w = must_upload(st, "detokenizer.proj_out.weight");
    detok->proj_out_b = must_upload(st, "detokenizer.proj_out.bias");

    detok->detok_enc.cfg = enc_default_config(2, 5);  // 2 layers, max S=5
    detok->detok_enc.cublas = cublas;
    enc_alloc_buffers(&detok->detok_enc);
    enc_load_layers(&detok->detok_enc, st, "detokenizer.layers");

    cudaMalloc(&detok->buf_embedded, (size_t)5 * H * sizeof(bf16));
    int max_T_25Hz = max_T_5Hz * 5;
    cudaMalloc(&detok->buf_out, (size_t)max_T_25Hz * 64 * sizeof(bf16));

    fprintf(stderr, "[Detokenizer] Loaded: 2L, S=5, proj_out->64, max_T_5Hz=%d\n", max_T_5Hz);
}

// Kernel: add special_tokens positional to broadcast-expanded embedding
// embedded [1, H] broadcasted -> [P, H], plus special_tokens [P, H]
// Result: dst[p] = embedded + special_tokens[p]
__global__ void broadcast_add_special_kernel(bf16 *dst, const bf16 *embedded,
                                              const bf16 *special_tokens, int P, int H) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = P * H;
    if (idx >= total) return;
    int p = idx / H;
    int h = idx % H;
    float e = __bfloat162float(embedded[h]);  // same for all p
    float s = __bfloat162float(special_tokens[p * H + h]);
    dst[idx] = __float2bfloat16(e + s);
}

// FSQ get_output_from_indices: indices [T_5Hz] -> quantized [T_5Hz, 2048]
// (Used for cover mode with LM-generated audio codes)
static void fsq_indices_to_quantized(FSQ *fsq, const int *indices, int T_5Hz,
                                      bf16 *output, cublasHandle_t cublas) {
    // Decode indices -> [T_5Hz, 6]
    fsq_decode_indices(fsq->buf_quant, indices, T_5Hz);
    // project_out: [T_5Hz, 6] -> [T_5Hz, 2048]
    linear_batch(output, fsq->buf_quant, fsq->project_out_w, nullptr,
                 T_5Hz, 2048, FSQ_NDIMS, cublas);
    add_bias_2d(output, fsq->project_out_b, T_5Hz, 2048);
}

// Detokenizer forward: [T_5Hz, 2048] -> [T_25Hz, 64]
static void detokenizer_forward(Detokenizer *detok, bf16 *quantized_5Hz, int T_5Hz,
                                 bf16 *output_25Hz) {
    int H = 2048;
    int P = detok->pool_window;  // 5

    for (int g = 0; g < T_5Hz; g++) {
        bf16 *token_in = quantized_5Hz + (int64_t)g * H;  // [H]

        // embed_tokens: [1, 2048] -> [1, 2048]
        bf16 *tmp_embed = detok->detok_enc.buf_norm;  // [1, H] temp
        linear_batch(tmp_embed, token_in, detok->embed_w, nullptr,
                     1, H, H, detok->cublas);
        add_bias_2d(tmp_embed, detok->embed_b, 1, H);

        // Broadcast add: embedded[H] + special_tokens[5, H] -> [5, H]
        int total = P * H;
        broadcast_add_special_kernel<<<(total + 255) / 256, 256>>>(
            detok->buf_embedded, tmp_embed, detok->special_tokens, P, H);

        // Copy to encoder input and run 2L
        cudaMemcpyAsync(detok->detok_enc.buf_hidden, detok->buf_embedded,
                        (size_t)P * H * sizeof(bf16), cudaMemcpyDeviceToDevice);
        enc_forward_with_norm(&detok->detok_enc, P, detok->norm_w);

        // proj_out: [5, 2048] -> [5, 64]
        bf16 *proj_dst = output_25Hz + (int64_t)g * P * 64;
        linear_batch(proj_dst,
                     detok->detok_enc.buf_hidden,
                     detok->proj_out_w, nullptr,
                     P, 64, H, detok->cublas);
        add_bias_2d(proj_dst, detok->proj_out_b, P, 64);
    }

}
