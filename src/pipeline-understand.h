#pragma once
// pipeline-understand.h: ACE-Step reverse pipeline (audio -> metadata)
//
// A lightweight context bound to a ModelStore. Each generate call acquires
// the VAE encoder, FSQ tokenizer and LM from the store in sequence, with
// RAII release between stages. Audio -> latents -> codes -> LM -> metadata.

#include "request.h"

struct AceUnderstand;
struct ModelStore;

struct AceUnderstandParams {
    const char * model_path;   // LM GGUF (required, unless dump_dir set for tok-only mode)
    const char * dit_path;     // DiT GGUF (required for audio input, has FSQ codebook)
    const char * vae_path;     // VAE GGUF (required for audio input, has encoder)
    const char * dump_dir;     // dump tok_latents + tok_codes (NULL = disabled)
    int          max_seq;      // KV cache length (default: 8192)
    int          max_batch;    // must match ace_lm max_batch so the LM ModelKey is identical
                               // and the two pipelines share one LM instance (default: 1)
    bool         use_fsm;      // constrained decoding (default: true)
    bool         use_fa;       // flash attention (default: true)
    int          vae_chunk;    // latent frames per tile (default: 256)
    int          vae_overlap;  // overlap frames per side (default: 64)
};

void ace_understand_default_params(AceUnderstandParams * p);

// Build a lightweight understand context bound to a ModelStore. Validates
// paths and builds ModelKeys. GPU modules are acquired per request, never
// owned by the context. NULL on invalid input.
AceUnderstand * ace_understand_load(ModelStore * store, const AceUnderstandParams * params);

// Run the understand pipeline.
// src_audio: interleaved stereo 48kHz [L0,R0,L1,R1,...], or NULL for codes-only mode.
// src_len: samples per channel (0 if no audio).
// req: sampling params (temperature, top_p, top_k, seed). In codes-only mode,
//      req->audio_codes must be filled.
// out: filled with caption, lyrics, metadata, audio_codes, DiT defaults.
// cancel/cancel_data: abort callback, polled between tokens. NULL = never cancel.
// Returns 0 on success, -1 on error or cancellation.
int ace_understand_generate(AceUnderstand *    ctx,
                            const float *      src_audio,
                            int                src_len,
                            const AceRequest * req,
                            AceRequest *       out,
                            bool (*cancel)(void *) = nullptr,
                            void * cancel_data     = nullptr);

void ace_understand_free(AceUnderstand * ctx);
