#pragma once
// pipeline-synth-impl.h: private state of the synthesis pipeline
//
// Holds the types that the public API exposes as opaque handles (AceSynth,
// AceSynthJob) plus SynthState, the transient state that flows between the
// primitive ops during a single job.
//
// Included by the two implementation files:
//   pipeline-synth.cpp      orchestrator: load, phases, free
//   pipeline-synth-ops.cpp  primitives:   encode, context, noise, dit, vae

#include "debug.h"
#include "model-store.h"
#include "pipeline-synth.h"
#include "request.h"
#include "timer.h"

#include <string>
#include <vector>

// AceSynth is a thin handle over a ModelStore. It carries the ModelKeys the
// ops use to acquire modules and a pointer to the DiT metadata owned by the
// store (silence_latent, null_cond, config, is_turbo). No GPU module is ever
// owned here: each op performs its own require/release against the store.
struct AceSynth {
    ModelStore *   store;
    AceSynthParams params;

    // CPU metadata pointer. Owned by the store, valid for the store lifetime
    // (always longer than AceSynth). Gives ops access to silence_full,
    // null_cond_cpu, is_turbo and the DiT config without loading the DiT.
    const DiTMeta * meta;

    // Derived constants mirrored for inline use in ops.
    int Oc;      // out_channels (64)
    int ctx_ch;  // in_channels - Oc (128)

    // ModelKeys for the seven GPU modules the pipeline touches.
    ModelKey text_enc_key;   // Qwen3 text encoder, from text_encoder_path
    ModelKey cond_enc_key;   // condition encoder, from dit_path
    ModelKey fsq_tok_key;    // FSQ tokenizer, from dit_path
    ModelKey fsq_detok_key;  // FSQ detokenizer, from dit_path
    ModelKey dit_key;        // DiT, from dit_path (+ adapter_path, adapter_scale)
    ModelKey vae_enc_key;    // VAE encoder, from vae_path
    ModelKey vae_dec_key;    // VAE decoder, from vae_path
};

// Transient state for a single job, shared by reference across the primitive
// ops that make up phase 1 and phase 2.
struct SynthState {
    // model dimensions (from ctx)
    int Oc;
    int ctx_ch;

    // src audio encoding
    bool               have_cover;
    std::vector<float> cover_latents;        // [T_cover * 64] time-major VAE latents
    std::vector<float> noise_blend_latents;  // clean VAE latents for cover_noise_strength blending
    int                T_cover;

    // outpainting: silence padding for repaint beyond source bounds
    float              left_pad_sec;  // seconds of silence prepended (coordinate shift)
    std::vector<float> padded_src;    // interleaved stereo buffer with silence padding

    // mode flags
    std::string task;
    bool        is_repaint;
    bool        is_lego_region;
    float       rs;
    float       re;
    bool        use_source_context;
    bool        have_codes;
    int         max_codes_len;

    // shared params (from reqs[0])
    AceRequest rr;
    float      duration;
    int        num_steps;
    float      guidance_scale;
    float      shift;

    // diffusion schedule
    std::vector<float> schedule;

    // SDE mode: inject fresh noise at each denoising step (vs ODE pure integration)
    bool use_sde;

    // per-batch seeds (for reproducible SDE re-noising: seed + step offset)
    std::vector<int64_t> seeds;

    // latent dimensions
    int T;
    int S;
    int enc_S;

    // region
    int repaint_t0;
    int repaint_t1;

    // repaint quality (resolved from repaint_strength via _resolve_repaint_config)
    float repaint_injection_ratio;
    int   repaint_crossfade_frames;
    float repaint_wav_cf_sec;

    // DiT instruction
    std::string instruction_str;     // main DiT instruction (set by orchestrator)
    std::string nc_instruction_str;  // non-cover pass instruction (for cover_steps < 1.0 switching)

    // conditioning tensors
    std::vector<float> timbre_feats;
    int                S_ref_timbre;  // actual timbre sequence length (from VAE encode)

    // text encoding (per-batch)
    std::vector<std::vector<float>> per_enc;
    std::vector<int>                per_enc_S;
    std::vector<std::vector<float>> per_enc_nc;
    std::vector<int>                per_enc_S_nc;
    bool                            need_enc_switch;

    // stacked encoder hidden states
    int                max_enc_S;
    std::vector<float> enc_hidden;
    std::vector<float> enc_hidden_nc;
    std::vector<int>   per_enc_S_nc_final;
    std::vector<float> null_cond_vec;

    // DiT context
    std::vector<float> context;
    std::vector<float> context_silence;
    int                cover_steps;

    // noise + output
    std::vector<float> noise;
    std::vector<float> output;
    std::vector<float> repaint_src;
    std::vector<int>   per_S;

    // debug / timing
    DebugDumper dbg;
    Timer       timer;
};

// Job bridges phase 1 (DiT) and phase 2 (VAE). The latents live in state.output
// as planar [batch_n, Oc, T] f32 until ace_synth_job_run_vae consumes them.
struct AceSynthJob {
    SynthState state;
    int        batch_n;
};
