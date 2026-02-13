#pragma once
// ConditionEncoder: combines text, lyrics, timbre into encoder_hidden_states
// B=1 inference only. No batch packing complexity.
//
// Requires: kernels.cuh, dit.cuh, transformer.cuh included before this file

// ConditionEncoder weights + sub-models
struct ConditionEncoder {
    // Lyric encoder: 8 layers bidirectional
    EncModel lyric_enc;
    bf16 *lyric_embed_w;       // [2048, 1024]
    bf16 *lyric_embed_b;       // [2048]
    bf16 *lyric_norm_w;        // [2048]

    // Timbre encoder: 4 layers bidirectional
    EncModel timbre_enc;
    bf16 *timbre_embed_w;      // [2048, 64]
    bf16 *timbre_embed_b;      // [2048]
    bf16 *timbre_norm_w;       // [2048]
    bf16 *timbre_special_token; // [2048] (loaded but NOT used, commented out in Python)

    // Text projector: linear [1024 -> 2048], no bias
    bf16 *text_proj_w;         // [2048, 1024]

    // Null condition embedding (for CFG)
    bf16 *null_condition_emb;  // [2048]

    // Output buffer
    bf16 *encoder_hidden_states; // [max_enc_S, 2048]
    int enc_seq_len;             // actual S after packing
    int max_enc_S;               // max allocated

    // Temp buffers for text projection
    bf16 *buf_text_proj;       // [max_text_S, 2048]

    cublasHandle_t cublas;
};

// Load condition encoder from safetensors
static void load_condition_encoder(ConditionEncoder *ce, SafeTensors &st,
                                   int max_lyric_S, int max_timbre_S, int max_text_S,
                                   cublasHandle_t cublas) {
    int H = 2048;
    ce->cublas = cublas;
    ce->max_enc_S = max_lyric_S + max_timbre_S + max_text_S + 16; // headroom

    // Lyric encoder
    ce->lyric_embed_w = must_upload(st, "encoder.lyric_encoder.embed_tokens.weight");
    ce->lyric_embed_b = must_upload(st, "encoder.lyric_encoder.embed_tokens.bias");
    ce->lyric_norm_w  = must_upload(st, "encoder.lyric_encoder.norm.weight");

    ce->lyric_enc.cfg = enc_default_config(8, max_lyric_S);
    ce->lyric_enc.cublas = cublas;
    enc_alloc_buffers(&ce->lyric_enc);
    enc_load_layers(&ce->lyric_enc, st, "encoder.lyric_encoder.layers");

    // Timbre encoder
    ce->timbre_embed_w = must_upload(st, "encoder.timbre_encoder.embed_tokens.weight");
    ce->timbre_embed_b = must_upload(st, "encoder.timbre_encoder.embed_tokens.bias");
    ce->timbre_norm_w  = must_upload(st, "encoder.timbre_encoder.norm.weight");
    ce->timbre_special_token = must_upload(st, "encoder.timbre_encoder.special_token");

    ce->timbre_enc.cfg = enc_default_config(4, max_timbre_S);
    ce->timbre_enc.cublas = cublas;
    enc_alloc_buffers(&ce->timbre_enc);
    enc_load_layers(&ce->timbre_enc, st, "encoder.timbre_encoder.layers");

    // Text projector
    ce->text_proj_w = must_upload(st, "encoder.text_projector.weight");

    // Null condition
    ce->null_condition_emb = must_upload(st, "null_condition_emb");

    // Output + temp buffers
    cudaMalloc(&ce->encoder_hidden_states, (size_t)ce->max_enc_S * H * sizeof(bf16));
    cudaMalloc(&ce->buf_text_proj, (size_t)max_text_S * H * sizeof(bf16));

    ce->enc_seq_len = 0;

    fprintf(stderr, "[CondEnc] Loaded: lyric(8L, max_S=%d), timbre(4L, max_S=%d), text_proj, max_enc_S=%d\n",
            max_lyric_S, max_timbre_S, ce->max_enc_S);
}

// Forward: produce encoder_hidden_states [S_total, 2048]
//
// Inputs (all device pointers):
//   text_hidden    [S_text, 1024] : from TextEncoder (Qwen3-Embedding)
//   lyric_hidden   [S_lyric, 1024]: lyric text embeddings (from same tokenizer)
//   timbre_feats   [S_ref, 64]    : reference audio acoustic features
//                                     (NULL if no reference audio)
//
// Output:
//   ce->encoder_hidden_states [S_total, 2048]
//   ce->enc_seq_len = S_total
static void condition_encoder_forward(ConditionEncoder *ce,
                                      bf16 *text_hidden, int S_text,
                                      bf16 *lyric_hidden, int S_lyric,
                                      bf16 *timbre_feats, int S_ref) {
    int H = 2048;

    // 1) Text projection: [S_text, 1024] -> [S_text, 2048]
    linear_batch(ce->buf_text_proj, text_hidden, ce->text_proj_w, nullptr,
                 S_text, H, 1024, ce->cublas);

    // 2) Lyric encoder: embed [S_lyric, 1024] -> [S_lyric, 2048], then 8L
    enc_embed_input(&ce->lyric_enc, lyric_hidden, ce->lyric_embed_w, ce->lyric_embed_b,
                    S_lyric, 1024);
    enc_forward_with_norm(&ce->lyric_enc, S_lyric, ce->lyric_norm_w);

    // 3) Timbre encoder: embed [S_ref, 64] -> [S_ref, 2048], then 4L
    int S_timbre_out = 0;
    if (timbre_feats && S_ref > 0) {
        enc_embed_input(&ce->timbre_enc, timbre_feats, ce->timbre_embed_w, ce->timbre_embed_b,
                        S_ref, 64);
        enc_forward_with_norm(&ce->timbre_enc, S_ref, ce->timbre_norm_w);
        // Take first position [0, :] as timbre embedding -> 1 token
        // Result at ce->timbre_enc.buf_hidden[0:2048]
        S_timbre_out = 1;
    }

    // 4) Pack sequences: cat(lyric, timbre, text) -> encoder_hidden_states
    // For B=1, all tokens are valid, just concatenate
    int offset = 0;
    bf16 *out = ce->encoder_hidden_states;

    // Lyric: [S_lyric, H]
    if (S_lyric > 0) {
        cudaMemcpyAsync(out + (int64_t)offset * H,
                        ce->lyric_enc.buf_hidden,
                        (size_t)S_lyric * H * sizeof(bf16),
                        cudaMemcpyDeviceToDevice);
        offset += S_lyric;
    }

    // Timbre: [1, H] (first position only)
    if (S_timbre_out > 0) {
        cudaMemcpyAsync(out + (int64_t)offset * H,
                        ce->timbre_enc.buf_hidden,
                        (size_t)H * sizeof(bf16),
                        cudaMemcpyDeviceToDevice);
        offset += 1;
    }

    // Text: [S_text, H]
    if (S_text > 0) {
        cudaMemcpyAsync(out + (int64_t)offset * H,
                        ce->buf_text_proj,
                        (size_t)S_text * H * sizeof(bf16),
                        cudaMemcpyDeviceToDevice);
        offset += S_text;
    }

    ce->enc_seq_len = offset;

}
