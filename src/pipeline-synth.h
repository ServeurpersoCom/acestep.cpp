#pragma once
// pipeline-synth.h: ACE-Step synthesis pipeline
//
// A lightweight context bound to a ModelStore. The pipeline holds no GPU
// modules: each op (text encoding, cond encoding, FSQ roundtrip, DiT denoise,
// VAE decode) acquires its own modules from the store with RAII release.
// STRICT policy keeps at most one module resident at a time. NEVER policy
// caches everything across calls. DiT weight swap between phases is an
// invisible consequence of the store, not an orchestration concern anymore.

#include "request.h"

#include <cstdlib>

struct AceSynth;
struct AceSynthJob;
struct ModelStore;

struct AceSynthParams {
    const char * text_encoder_path;  // Qwen3 text encoder GGUF (required)
    const char * dit_path;           // DiT GGUF (required)
    const char * vae_path;           // VAE GGUF (required)
    const char * adapter_path;       // adapter safetensors or directory (NULL to disable)
    float        adapter_scale;      // user scale multiplier
    bool         use_fa;             // flash attention
    bool         clamp_fp16;         // clamp hidden states to FP16 range
    bool         use_batch_cfg;      // batch cond+uncond in one DiT forward
    int          vae_chunk;          // latent frames per tile
    int          vae_overlap;        // overlap frames per side
    const char * dump_dir;           // intermediate tensor dump dir (NULL = disabled)
};

// Output audio buffer. Caller must free with ace_audio_free().
struct AceAudio {
    float * samples;      // planar stereo [L0..LN, R0..RN]
    int     n_samples;    // per channel
    int     sample_rate;  // always 48000
};

void ace_synth_default_params(AceSynthParams * p);

// Build a lightweight synth context bound to a ModelStore. Reads DiT metadata
// (config, silence, null_cond, is_turbo) through the store so text encoding
// and T resolution can run before the DiT itself is ever loaded. All GPU
// modules are acquired per op, never owned by the context. NULL on failure.
AceSynth * ace_synth_load(ModelStore * store, const AceSynthParams * params);

// Phase 1: encode sources, build context, run all DiT denoising steps.
// Modules are acquired as needed: VAE encoder for source and timbre, FSQ
// tokenizer and detokenizer for cover-mode roundtrip, text encoder + cond
// encoder for prompt embedding, DiT for denoising. In STRICT each module is
// released before the next acquires, so VRAM hosts one at a time.
// Produces a job that carries DiT latents in RAM for later VAE decoding.
// reqs[batch_n]: each request has its own caption, lyrics, metadata, audio_codes, and seed.
//   The first request (reqs[0]) is used for shared params (mode, duration, DiT settings).
//   seed must be resolved (non-negative) before calling this function.
// src_audio / ref_audio: interleaved stereo 48kHz buffers, NULL when not applicable.
// src_latents / ref_latents: pre-encoded latents [T_latent * 64] f32 alternative
//   to the matching audio buffer. When non-NULL, the corresponding VAE encoder
//   pass is skipped. Mutually exclusive per side: when both audio and latents
//   are provided for the same side, latents win and audio is ignored.
// batch_n: number of requests (1..9).
// cancel/cancel_data: abort callback, polled between DiT steps. NULL = never cancel.
// Returns NULL on error or cancellation.
AceSynthJob * ace_synth_job_run_dit(AceSynth *         ctx,
                                    const AceRequest * reqs,
                                    const float *      src_audio,
                                    int                src_len,
                                    const float *      src_latents,
                                    int                src_T_latent,
                                    const float *      ref_audio,
                                    int                ref_len,
                                    const float *      ref_latents,
                                    int                ref_T_latent,
                                    int                batch_n,
                                    bool (*cancel)(void *) = nullptr,
                                    void * cancel_data     = nullptr);

// Access the cover latents captured during Phase 1, regardless of how they
// were obtained (VAE-encoded from src_audio or fed directly via src_latents).
// Returns the [T_latent * 64] f32 buffer owned by the job, or NULL if the
// task did not produce cover latents (e.g. text2music). T_latent is written
// to *T_out when the pointer is non-NULL.
const float * ace_synth_job_get_latents(const AceSynthJob * job, int * T_out);

// Phase 2: VAE decode and waveform splice. Acquires the VAE decoder and FSQ
// detokenizer from the store; in STRICT this evicts the DiT from phase 1
// transparently.
// splice_src / splice_len: interleaved stereo source reused for repaint/lego wave splicing.
//   Pass NULL when the job did not carry a source audio.
// out[batch_n] allocated by caller, filled with audio buffers.
// Returns 0 on success, -1 on error or cancellation.
int ace_synth_job_run_vae(AceSynth *    ctx,
                          AceSynthJob * job,
                          const float * splice_src,
                          int           splice_len,
                          AceAudio *    out,
                          bool (*cancel)(void *) = nullptr,
                          void * cancel_data     = nullptr);

void ace_synth_job_free(AceSynthJob * job);

void ace_audio_free(AceAudio * audio);

void ace_synth_free(AceSynth * ctx);
