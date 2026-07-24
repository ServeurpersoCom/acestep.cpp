#pragma once
// pipeline-synth-ops.h: primitive operations of the synthesis pipeline
//
// Each op is a single step of phase 1 or phase 2 (VAE encode, FSQ roundtrip,
// text encoding, context build, noise init, DiT generate, VAE decode). The
// orchestrator in pipeline-synth.cpp calls them in sequence. SynthState and
// AceSynth are defined in pipeline-synth-impl.h.

#include "pipeline-synth.h"

#include <vector>

struct AceSynth;
struct SynthState;

// Phase 1 primitives.

// Encode src_audio into cover_latents. Sets s.have_cover, s.cover_latents, s.T_cover.
int ops_encode_src(const AceSynth * ctx,
                   const float *    src_audio,
                   int              src_len,
                   const float *    src_latents,
                   int              src_T_latent,
                   SynthState &     s);

// FSQ roundtrip on cover_latents (cover mode only).
void ops_fsq_roundtrip(const AceSynth * ctx, SynthState & s);

// Resolve shared DiT params (steps, guidance, shift) and scan audio_codes.
int ops_resolve_params(const AceSynth * ctx, const AceRequest * reqs, int batch_n, SynthState & s);

// Build flow-matching timestep schedule.
void ops_build_schedule(SynthState & s);

// Resolve T (latent frame count) and S (patch count).
int ops_resolve_T(const AceSynth * ctx, SynthState & s);

// Encode timbre from ref_audio via VAE. Sets s.timbre_feats and s.S_ref_timbre.
void ops_encode_timbre(const AceSynth * ctx,
                       const float *    ref_audio,
                       int              ref_len,
                       const float *    ref_latents,
                       int              ref_T_latent,
                       SynthState &     s);

// Per-batch text + lyric encoding (main pass + optional non-cover pass).
// Stacks results into s.enc_hidden / s.enc_hidden_nc.
int ops_encode_text(const AceSynth * ctx, const AceRequest * reqs, int batch_n, SynthState & s);

// Build DiT context tensor [batch_n, T, ctx_ch] = src_latents(64) | mask(64).
int ops_build_context(const AceSynth * ctx, const AceRequest * reqs, int batch_n, SynthState & s);

// Build silence context + cover_steps for audio_cover_strength switching.
void ops_build_context_silence(const AceSynth * ctx, int batch_n, SynthState & s);

// Init noise tensor (Philox) + cover noise blend.
void ops_init_noise(const AceSynth * ctx, const AceRequest * reqs, int batch_n, SynthState & s);

// Run the DiT denoising loop.
int ops_dit_generate(const AceSynth * ctx, int batch_n, SynthState & s, bool (*cancel)(void *), void * cancel_data);

// Phase 2 primitive.

// Latent splice for repaint/lego (kept generated frames inside [t0, t1),
// source latents elsewhere) followed by VAE decode for every batch item.
// Returns 0 on success, -1 on error/cancel.
int ops_vae_decode(const AceSynth * ctx,
                   int              batch_n,
                   AceAudio *       out,
                   SynthState &     s,
                   bool (*cancel)(void *),
                   void * cancel_data);

// Scoring primitive: DiT cross-attention capture for the Python reference's
// pure-noise and regressed-latent states. pred_latents is the generated DiT
// output [batch_n, T, 64].
//
// Requires ops_encode_text (for s.per_lyric_ids) and ops_build_context +
// ops_init_noise (for noise + context latents) to have been run first.
//
// out_scores: filled with one LM/DiT score comparison per batch item.
// cancel/cancel_data: abort callback, polled before and between DiT forwards.
// Returns 0 on success, -1 on error or cancellation.
int ops_score_forward(const AceSynth *                    ctx,
                      int                                 batch_n,
                      const float *                       pred_latents,
                      std::vector<LyricScoreComparison> & out_scores,
                      SynthState &                        s,
                      bool (*cancel)(void *),
                      void * cancel_data);
