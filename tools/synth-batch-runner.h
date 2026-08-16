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
#include <string>
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
// latents_out: optional capture of one post-DiT latent per generated track,
//   indexed identically to audio_out. Each entry is [T_track * 64] f32 time-major,
//   T_track = entry.size() / 64. Pass NULL to skip the capture.
// lyric_timings_out: optional JSON timing payloads, indexed identically to
//   audio_out. Entries are empty unless request.return_lyric_timing is true.
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
                           std::vector<std::vector<float>> *      latents_out = nullptr,
                           std::vector<std::string> *             lyric_timings_out = nullptr,
                           bool (*cancel)(void *)                             = nullptr,
                           void * cancel_data                                 = nullptr) {
    const int                  n_groups = (int) groups.size();
    std::vector<AceSynthJob *> jobs(n_groups, nullptr);
    std::vector<int>           audio_off(n_groups, 0);

    if (latents_out) {
        latents_out->clear();
    }
    if (lyric_timings_out) {
        lyric_timings_out->clear();
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

    // Capture one post-DiT latent per track, time-major [T*64], indexed to
    // match audio_out. Latents live in jobs[g]->state.output until run_vae
    // frees the job; capture happens before phase 2.
    if (latents_out) {
        latents_out->resize((size_t) off);
        for (int g = 0; g < n_groups; g++) {
            const int gn = (int) groups[g].size();
            for (int i = 0; i < gn; i++) {
                int           T   = 0;
                const float * src = ace_synth_job_get_latent(jobs[g], i, &T);
                (*latents_out)[audio_off[g] + i].assign(src, src + (size_t) T * 64);
            }
        }
    }
    if (lyric_timings_out) {
        lyric_timings_out->resize((size_t) off);
        for (int g = 0; g < n_groups; g++) {
            const int gn = (int) groups[g].size();
            for (int i = 0; i < gn; i++) {
                const char * json = ace_synth_job_get_lyric_timing_json(jobs[g], i);
                if (json) {
                    (*lyric_timings_out)[audio_off[g] + i] = json;
                }
            }
        }
    }

    // Phase 2: latent splice + VAE decode for each job. The decoder is
    // acquired and released by ops_vae_decode inside ace_synth_job_run_vae.
    for (int g = 0; g < n_groups; g++) {
        const int rc = ace_synth_job_run_vae(ctx, jobs[g], audio_out + audio_off[g], cancel, cancel_data);
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
