#pragma once
// pipeline-synth.h: ACE-Step synthesis pipeline
//
// A lightweight context bound to a ModelStore. The pipeline holds no GPU
// modules: each op (text encoding, cond encoding, FSQ roundtrip, DiT denoise,
// VAE decode) acquires its own modules from the store with RAII release.
// STRICT policy keeps at most one module resident at a time. NEVER policy
// caches everything across calls. DiT weight swap between phases is an
// invisible consequence of the store, not an orchestration concern anymore.

#include "dtw-score.h"
#include "request.h"

#include <cstdlib>
#include <vector>

struct AceSynth;
struct AceSynthJob;
struct ModelStore;

struct AceSynthParams {
    const char * text_encoder_path;  // Qwen3 text encoder GGUF (required)
    const char * dit_path;           // DiT GGUF (required)
    const char * vae_path;           // VAE GGUF (required except for score-only contexts)
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

// Score-only variant: text encoder and DiT are required, but VAE is not.
AceSynth * ace_synth_load_score(ModelStore * store, const AceSynthParams * params);

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

// Access the post-DiT denoised latent for one track of the job, owned by the
// job and valid until ace_synth_job_free. Layout is flat [T_latent * 64] f32
// time-major, identical to the /vae wire format: feeding it to /vae decode
// reproduces the track's output audio. *T_out is set to T_latent.
const float * ace_synth_job_get_latent(const AceSynthJob * job, int track_idx, int * T_out);

// Phase 2: latent splice (for repaint/lego) + VAE decode. Acquires the VAE
// decoder from the store; in STRICT this evicts the DiT from phase 1
// transparently. The splice happens in latent space using s.cover_latents
// captured during phase 1, no source audio is needed.
// out[batch_n] allocated by caller, filled with audio buffers.
// Returns 0 on success, -1 on error or cancellation.
int ace_synth_job_run_vae(AceSynth *    ctx,
                          AceSynthJob * job,
                          AceAudio *    out,
                          bool (*cancel)(void *) = nullptr,
                          void * cancel_data     = nullptr);

void ace_synth_job_free(AceSynthJob * job);

void ace_audio_free(AceAudio * audio);

void ace_synth_free(AceSynth * ctx);

// Lyric scoring against generated DiT latents. Runs the Python reference's
// pure-noise and regressed-latent attention passes, then computes LM and DiT
// alignment metrics on the pure lyric token range.
//
// pred_latents: [batch_n, pred_T_latent, 64] generated latent tensor.
// Each request seed must be resolved (non-negative) before calling.
// Returns one LyricScoreComparison per batch item in out_scores.
// Returns 0 on success, -1 on error.
int ace_synth_score(AceSynth *                          ctx,
                    const AceRequest *                  reqs,
                    int                                 batch_n,
                    const float *                       pred_latents,
                    int                                 pred_T_latent,
                    std::vector<LyricScoreComparison> & out_scores);
