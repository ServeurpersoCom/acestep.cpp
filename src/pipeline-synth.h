#pragma once
//
// pipeline-synth.h - ACE-Step synthesis pipeline
//
// Loads DiT + TextEncoder + CondEncoder + VAE once, then generates audio
// from enriched requests (output of pipeline-lm or pre-filled JSON).
//

#include "request.h"

#include <cstdlib>

struct AceSynth;

struct AceSynthParams {
    const char * text_encoder_path;  // Qwen3 text encoder GGUF (required)
    const char * dit_path;           // DiT GGUF (required)
    const char * vae_path;           // VAE GGUF (NULL = no audio decode, latent only)
    const char * lora_path;          // LoRA adapter path (NULL = no lora)
    float        lora_scale;         // 1.0
    bool         use_fa;             // flash attention (default: true)
    int          vae_chunk;          // latent frames per tile (default: 256)
    int          vae_overlap;        // overlap frames per side (default: 64)
    const char * dump_dir;           // intermediate tensor dump dir (NULL = disabled)
};

// Output audio buffer. Caller must free with ace_audio_free().
struct AceAudio {
    float * samples;      // planar stereo [L0..LN, R0..RN]
    int     n_samples;    // per channel
    int     sample_rate;  // always 48000
};

void ace_synth_default_params(AceSynthParams * p);

// Load all models. NULL on failure.
AceSynth * ace_synth_load(const AceSynthParams * params);

// Generate audio from request.
// src_audio: interleaved stereo 48kHz (for cover/lego mode), NULL for text2music.
// src_len: samples per channel.
// batch_n: number of variations (1..9).
// out[batch_n] allocated by caller, filled with audio buffers.
// Returns 0 on success.
int ace_synth_generate(AceSynth *         ctx,
                       const AceRequest * req,
                       const float *      src_audio,
                       int                src_len,
                       int                batch_n,
                       AceAudio *         out);

void ace_audio_free(AceAudio * audio);
void ace_synth_free(AceSynth * ctx);
