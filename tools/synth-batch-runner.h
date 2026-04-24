#pragma once
// synth-batch-runner.h: two-phase orchestration shared by the synth binaries
//
// Phase 1 (all groups) runs ace_synth_job_run_dit. Each call acquires the DiT
// from the store for the duration of its denoising loop and releases it on
// scope exit. Under EVICT_STRICT this means the VAE encoder, text encoder,
// cond encoder and DiT never coexist in VRAM; under EVICT_NEVER they all
// accumulate across calls. Phase 2 (all jobs) runs ace_synth_job_run_vae,
// which acquires the VAE decoder on entry and releases it on exit.

#include "pipeline-synth.h"

#include <cstdio>
#include <vector>

// Run a batch of request groups through the two synthesis phases.
//
// groups[g][i]: request i of group g. All requests in a group must share
//   the same T (same audio_codes or same duration), which the ops assume
//   when they stack per-batch tensors for a single DiT forward.
//   seed must be resolved (non-negative) on every request.
// src_audio / ref_audio: interleaved stereo 48kHz buffers, NULL when not applicable.
// src_latents / ref_latents: pre-encoded latents [T_latent * 64] f32 alternative
//   to the matching audio buffer. When non-NULL, the corresponding VAE encoder
//   pass is skipped for every group. The same buffers are shared across groups,
//   matching how src_audio and ref_audio are shared today.
// audio_out[sum_g(groups[g].size())]: pre-allocated slots filled by phase 2.
//   On error, slots completed before the failure keep their audio; the rest
//   are left at {NULL, 0, 0}. Caller owns ace_audio_free.
// latent_out / T_latent_out: optional capture of the cover latents produced
//   (or ingested) by the first group. All groups within one call share the
//   same source so a single latent buffer is sufficient. Pass NULL to skip
//   the capture. On success and when the source path produced latents,
//   *latent_out is filled (move-assigned) and *T_latent_out is set; otherwise
//   *latent_out is left empty and *T_latent_out is set to 0.
// Returns 0 on success, -1 on any error or cancellation.
static int synth_batch_run(AceSynth *                             ctx,
                           std::vector<std::vector<AceRequest>> & groups,
                           const float *                          src_audio,
                           int                                    src_len,
                           const float *                          src_latents,
                           int                                    src_T_latent,
                           const float *                          ref_audio,
                           int                                    ref_len,
                           const float *                          ref_latents,
                           int                                    ref_T_latent,
                           AceAudio *                             audio_out,
                           std::vector<float> *                   latent_out   = nullptr,
                           int *                                  T_latent_out = nullptr,
                           bool (*cancel)(void *)                              = nullptr,
                           void * cancel_data                                  = nullptr) {
    const int                  n_groups = (int) groups.size();
    std::vector<AceSynthJob *> jobs(n_groups, nullptr);
    std::vector<int>           audio_off(n_groups, 0);

    if (latent_out) {
        latent_out->clear();
    }
    if (T_latent_out) {
        *T_latent_out = 0;
    }

    // Phase 1: denoising loop for each group. The DiT is acquired and released
    // by ops_dit_generate inside ace_synth_job_run_dit.
    int off = 0;
    for (int g = 0; g < n_groups; g++) {
        const int gn = (int) groups[g].size();
        jobs[g] = ace_synth_job_run_dit(ctx, groups[g].data(), src_audio, src_len, src_latents, src_T_latent, ref_audio,
                                        ref_len, ref_latents, ref_T_latent, gn, cancel, cancel_data);
        if (!jobs[g]) {
            for (int j = 0; j < g; j++) {
                ace_synth_job_free(jobs[j]);
            }
            return -1;
        }
        audio_off[g] = off;
        off += gn;
    }

    // Capture cover latents from the first group: source is shared across
    // groups so one buffer covers them all. Skipped when the caller did not
    // request capture or when the task did not produce cover latents.
    if (latent_out && n_groups > 0) {
        int           captured_T = 0;
        const float * captured   = ace_synth_job_get_latents(jobs[0], &captured_T);
        if (captured && captured_T > 0) {
            latent_out->assign(captured, captured + (size_t) captured_T * 64);
            if (T_latent_out) {
                *T_latent_out = captured_T;
            }
        }
    }

    // Phase 2: VAE decode for each job. The decoder is acquired and released
    // by ops_vae_decode_and_splice inside ace_synth_job_run_vae.
    for (int g = 0; g < n_groups; g++) {
        const int rc =
            ace_synth_job_run_vae(ctx, jobs[g], src_audio, src_len, audio_out + audio_off[g], cancel, cancel_data);
        ace_synth_job_free(jobs[g]);
        jobs[g] = nullptr;
        if (rc != 0) {
            for (int j = g + 1; j < n_groups; j++) {
                ace_synth_job_free(jobs[j]);
            }
            return -1;
        }
    }

    return 0;
}
