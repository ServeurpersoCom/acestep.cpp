// pipeline-synth.cpp: ACE-Step synthesis pipeline implementation
//
// Thin orchestrator over a ModelStore. Holds no GPU module, no CPU-cached
// DiT state of its own: the store exposes DiTMeta (silence, null_cond, cfg,
// is_turbo) and each op acquires the GPU modules it needs on the fly.

#include "pipeline-synth.h"

#include "pipeline-synth-impl.h"
#include "pipeline-synth-ops.h"
#include "task-types.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

void ace_synth_default_params(AceSynthParams * p) {
    p->text_encoder_path = NULL;
    p->dit_path          = NULL;
    p->vae_path          = NULL;
    p->adapter_path      = NULL;
    p->adapter_scale     = 1.0f;
    p->use_fa            = true;
    p->clamp_fp16        = false;
    p->use_batch_cfg     = true;
    p->vae_chunk         = 256;
    p->vae_overlap       = 64;
    p->dump_dir          = NULL;
}

AceSynth * ace_synth_load(ModelStore * store, const AceSynthParams * params) {
    if (!store || !params) {
        fprintf(stderr, "[Synth-Load] ERROR: store and params are required\n");
        return NULL;
    }
    if (!params->dit_path) {
        fprintf(stderr, "[Synth-Load] ERROR: dit_path is NULL\n");
        return NULL;
    }
    if (!params->text_encoder_path) {
        fprintf(stderr, "[Synth-Load] ERROR: text_encoder_path is NULL\n");
        return NULL;
    }
    if (!params->vae_path) {
        fprintf(stderr, "[Synth-Load] ERROR: vae_path is NULL\n");
        return NULL;
    }

    AceSynth * ctx = new AceSynth();
    ctx->store     = store;
    ctx->params    = *params;

    // DiTMeta: config + silence_latent + null_condition_emb + is_turbo,
    // fetched once, valid for the store lifetime. Avoids loading the DiT
    // itself just to read a few CPU-side tensors.
    ctx->meta = store_dit_meta(store, params->dit_path);
    if (!ctx->meta) {
        fprintf(stderr, "[Synth-Load] FATAL: DiT metadata unavailable for %s\n", params->dit_path);
        delete ctx;
        return NULL;
    }
    ctx->Oc     = ctx->meta->cfg.out_channels;           // 64
    ctx->ctx_ch = ctx->meta->cfg.in_channels - ctx->Oc;  // 128

    // ModelKeys. Each path identifies its GGUF; adapter info rides with the
    // DiT key because two DiTs with different adapters are distinct modules.
    ctx->text_enc_key.kind = MODEL_TEXT_ENC;
    ctx->text_enc_key.path = params->text_encoder_path;

    ctx->cond_enc_key.kind = MODEL_COND_ENC;
    ctx->cond_enc_key.path = params->dit_path;

    ctx->fsq_tok_key.kind = MODEL_FSQ_TOK;
    ctx->fsq_tok_key.path = params->dit_path;

    ctx->fsq_detok_key.kind = MODEL_FSQ_DETOK;
    ctx->fsq_detok_key.path = params->dit_path;

    ctx->dit_key.kind          = MODEL_DIT;
    ctx->dit_key.path          = params->dit_path;
    ctx->dit_key.adapter_path  = params->adapter_path ? params->adapter_path : "";
    ctx->dit_key.adapter_scale = params->adapter_scale;

    ctx->vae_enc_key.kind = MODEL_VAE_ENC;
    ctx->vae_enc_key.path = params->vae_path;

    ctx->vae_dec_key.kind = MODEL_VAE_DEC;
    ctx->vae_dec_key.path = params->vae_path;

    fprintf(stderr, "[Synth-Load] Ready: turbo=%s, fa=%s, batch_cfg=%s\n", ctx->meta->is_turbo ? "yes" : "no",
            params->use_fa ? "yes" : "no", params->use_batch_cfg ? "yes" : "no");
    if (params->clamp_fp16) {
        fprintf(stderr, "[Synth-Load] FP16 clamp enabled\n");
    }
    if (params->adapter_path) {
        fprintf(stderr, "[Synth-Load] Adapter: %s (scale=%.2f)\n", params->adapter_path, params->adapter_scale);
    }

    return ctx;
}

// Phase 1: everything up to and including DiT generate. The produced job
// carries the latent output, the resolved state, and any silence-padded source
// buffer needed later by the wave splice in phase 2.
AceSynthJob * ace_synth_job_run_dit(AceSynth *         ctx,
                                    const AceRequest * reqs,
                                    const float *      src_audio,
                                    int                src_len,
                                    const float *      ref_audio,
                                    int                ref_len,
                                    int                batch_n,
                                    bool (*cancel)(void *),
                                    void * cancel_data) {
    if (!ctx || !reqs || batch_n < 1 || batch_n > 9) {
        return NULL;
    }

    AceSynthJob * job = new AceSynthJob();
    job->batch_n      = batch_n;

    SynthState & s = job->state;
    s.Oc           = ctx->Oc;
    s.ctx_ch       = ctx->ctx_ch;
    s.left_pad_sec = 0.0f;
    debug_init(&s.dbg, ctx->params.dump_dir);

    // Outpainting: pad src_audio with silence when region coordinates extend
    // beyond source bounds. Padding happens before VAE encode so T_cover
    // reflects the padded duration.
    const float * enc_audio = src_audio;
    int           enc_len   = src_len;
    {
        const std::string & tp = reqs[0].task_type;
        bool                is_region_task =
            (tp == TASK_REPAINT) || (tp == TASK_LEGO && reqs[0].repainting_end > reqs[0].repainting_start);
        if (is_region_task && src_audio && src_len > 0) {
            float src_dur  = (float) src_len / 48000.0f;
            float rs_raw   = reqs[0].repainting_start;
            float re_raw   = reqs[0].repainting_end;
            float end_time = (re_raw < 0.0f) ? src_dur : re_raw;
            float lpad     = (rs_raw < 0.0f) ? -rs_raw : 0.0f;
            float rpad     = (end_time > src_dur) ? end_time - src_dur : 0.0f;

            if (lpad > 0.0f || rpad > 0.0f) {
                int lpad_s       = (int) (lpad * 48000.0f);
                int rpad_s       = (int) (rpad * 48000.0f);
                int padded_total = src_len + lpad_s + rpad_s;

                s.padded_src.resize((size_t) padded_total * 2);
                memset(s.padded_src.data(), 0, s.padded_src.size() * sizeof(float));
                memcpy(s.padded_src.data() + (size_t) lpad_s * 2, src_audio, (size_t) src_len * 2 * sizeof(float));

                s.left_pad_sec = lpad;
                enc_audio      = s.padded_src.data();
                enc_len        = padded_total;

                fprintf(stderr, "[Outpaint] pad left=%.1fs right=%.1fs total=%.1fs\n", lpad, rpad,
                        (float) padded_total / 48000.0f);
            }
        }
    }

    // VAE encode source audio (possibly padded for outpainting).
    // ops_encode_src loads a local VAE encoder, encodes, and frees it:
    // no VAE encoder stays resident while the DiT runs.
    if (ops_encode_src(ctx, enc_audio, enc_len, s) != 0) {
        delete job;
        return NULL;
    }

    // Shared request, mode flags, use_source_context.
    // Per-batch: caption, lyrics, metadata, audio_codes, and seed come from reqs[b].
    // seed must be resolved (non-negative) before calling this function.
    s.rr                 = reqs[0];
    // task_type is the master. Empty = text2music.
    s.task               = s.rr.task_type.empty() ? std::string(TASK_TEXT2MUSIC) : s.rr.task_type;
    // only repaint uses region masking. complete uses full src context (Python behavior).
    s.is_repaint         = (s.task == TASK_REPAINT);
    // lego with valid rs/re = region-constrained: generate only in zone, full audio context
    s.is_lego_region     = (s.task == TASK_LEGO && s.rr.repainting_end > s.rr.repainting_start);
    s.rs                 = s.rr.repainting_start;
    s.re                 = s.rr.repainting_end;
    // use_source_context must be set before ops_resolve_T which uses it for T.
    // Mode routing refines it per task; this first pass covers all source-context tasks.
    s.use_source_context = (s.task == TASK_COVER || s.task == TASK_COVER_NOFSQ || s.task == TASK_REPAINT ||
                            s.task == TASK_LEGO || s.task == TASK_EXTRACT || s.task == TASK_COMPLETE);
    // non-cover encoding pass (for audio_cover_strength < 1.0 switching) always uses text2music.
    s.nc_instruction_str = DIT_INSTR_TEXT2MUSIC;

    // Resolve DiT params (steps, guidance, shift) and scan audio codes
    if (ops_resolve_params(ctx, reqs, batch_n, s) != 0) {
        delete job;
        return NULL;
    }

    // Promote text2music to cover when codes present (Python _resolve_generate_music_task).
    // DiT was trained with the cover instruction for codes guided generation.
    if (s.task == TASK_TEXT2MUSIC && s.have_codes) {
        s.task               = std::string(TASK_COVER);
        s.use_source_context = true;
    }

    // Timestep schedule
    ops_build_schedule(s);

    // SDE mode: "sde" = stochastic re-noising at each step
    s.use_sde = (s.rr.infer_method == "sde");

    // Resolve latent frame count T
    if (ops_resolve_T(ctx, s) != 0) {
        delete job;
        return NULL;
    }

    // Resolve repaint quality params from repaint_strength.
    // Python: _resolve_repaint_config("balanced", strength).
    // 0.0 = aggressive, 0.5 = balanced (default), 1.0 = conservative.
    {
        float rs = s.rr.repaint_strength;
        if (rs < 0.0f) {
            rs = 0.0f;
        }
        if (rs > 1.0f) {
            rs = 1.0f;
        }
        float inv                  = 1.0f - rs;
        s.repaint_injection_ratio  = inv;
        s.repaint_crossfade_frames = (int) (25.0f * inv + 0.5f);
        s.repaint_wav_cf_sec       = 0.05f * inv;
        if (s.is_repaint || s.is_lego_region) {
            fprintf(stderr, "[Synth-Run] repaint_strength=%.2f -> injection=%.2f, crossfade=%d frames, wav_cf=%.0fms\n",
                    rs, s.repaint_injection_ratio, s.repaint_crossfade_frames, s.repaint_wav_cf_sec * 1000.0f);
        }
    }

    // Mode routing: per-task instruction, use_source_context, validation.
    //    All task knowledge lives here. Adding a mode = adding one branch.
    //    Track name is UPPERCASE in instructions (matches Python task_utils.py).
    {
        std::string track_upper = s.rr.track;
        for (char & ch : track_upper) {
            ch = (char) toupper((unsigned char) ch);
        }
        if (s.task == TASK_TEXT2MUSIC) {
            // silence context, no pre-encoding work needed
            s.use_source_context = false;
            s.instruction_str    = DIT_INSTR_TEXT2MUSIC;
        } else if (s.task == TASK_COVER) {
            s.use_source_context  = true;
            s.instruction_str     = DIT_INSTR_COVER;
            // save clean VAE latents before FSQ degrades them.
            // the FSQ roundtrip only affects context conditioning;
            // cover_noise_strength blending needs the original clean latents.
            s.noise_blend_latents = s.cover_latents;
            ops_fsq_roundtrip(ctx, s);  // FSQ degrades source latents, DiT diverges from original
        } else if (s.task == TASK_COVER_NOFSQ) {
            // cover without FSQ roundtrip: DiT works on clean VAE latents at 25Hz.
            // produces remixes that stay close to the source structure and timbre.
            // pass ref_audio = src_audio for best results.
            s.use_source_context = true;
            s.instruction_str    = DIT_INSTR_COVER;
        } else if (s.task == TASK_REPAINT) {
            s.use_source_context = true;
            s.instruction_str    = DIT_INSTR_REPAINT;
        } else if (s.task == TASK_LEGO) {
            s.use_source_context      = true;
            s.rr.audio_cover_strength = 1.0f;  // all DiT steps hear the backing track
            s.instruction_str         = dit_instr_lego(track_upper);
            validate_track_names(s.rr.track, "Lego");
            fprintf(stderr, "[Synth-Run] task=%s\n", s.task.c_str());
            if (ctx->meta->is_turbo) {
                fprintf(stderr, "[Synth-Run] WARNING: lego requires base model, turbo output incoherent\n");
            }
        } else if (s.task == TASK_EXTRACT) {
            s.use_source_context      = true;
            s.rr.audio_cover_strength = 1.0f;  // DiT sees the full mix
            s.instruction_str         = dit_instr_extract(track_upper);
            validate_track_names(s.rr.track, "Extract");
            fprintf(stderr, "[Synth-Run] task=%s\n", s.task.c_str());
            if (ctx->meta->is_turbo) {
                fprintf(stderr, "[Synth-Run] WARNING: extract requires base model, turbo output incoherent\n");
            }
        } else if (s.task == TASK_COMPLETE) {
            s.use_source_context      = true;
            s.rr.audio_cover_strength = 1.0f;  // DiT sees the full isolated stem
            s.instruction_str         = dit_instr_complete(track_upper);
            validate_track_names(s.rr.track, "Complete");
            fprintf(stderr, "[Synth-Run] task=%s\n", s.task.c_str());
            if (ctx->meta->is_turbo) {
                fprintf(stderr, "[Synth-Run] WARNING: complete requires base model, turbo output incoherent\n");
            }
        }
        // validation: tasks that need source audio or codes
        if (s.use_source_context && !s.have_cover && !s.have_codes) {
            fprintf(stderr, "[Synth-Run] ERROR: task '%s' requires source audio or audio codes\n", s.task.c_str());
            delete job;
            return NULL;
        }
    }

    // Region coordinate adjustment (repaint and lego_region share the same path).
    // Shift rs/re into the padded reference frame. When no padding was applied,
    // left_pad_sec is 0 and the arithmetic is a no-op.
    // adjusted_start = repainting_start + left_padding_duration
    if (s.is_repaint || s.is_lego_region) {
        s.rs += s.left_pad_sec;
        if (s.re < 0.0f) {
            // sentinel: outpaint (start < 0) ends at source start, inpaint ends at source end
            s.re = (s.rr.repainting_start < 0.0f) ? s.left_pad_sec : (float) src_len / 48000.0f + s.left_pad_sec;
        } else {
            s.re += s.left_pad_sec;
        }
        if (s.re <= s.rs) {
            fprintf(stderr, "[Region] ERROR: end (%.1f) <= start (%.1f)\n", s.re, s.rs);
            delete job;
            return NULL;
        }
        fprintf(stderr, "[Region] %.1fs..%.1fs (canvas=%.1fs)\n", s.rs, s.re, (float) s.T_cover * 1920.0f / 48000.0f);
    }

    // Encode timbre from ref_audio (independent of task).
    // ops_encode_timbre loads a local VAE encoder, encodes, and frees it.
    ops_encode_timbre(ctx, ref_audio, ref_len, s);

    // Per-batch text + lyric encoding (main + optional non-cover pass)
    if (ops_encode_text(ctx, reqs, batch_n, s) != 0) {
        delete job;
        return NULL;
    }

    // Build DiT context [batch_n, T, ctx_ch] = src(64) | mask(64)
    if (ops_build_context(ctx, reqs, batch_n, s) != 0) {
        delete job;
        return NULL;
    }

    // Silence context for audio_cover_strength switching (cover only)
    ops_build_context_silence(ctx, batch_n, s);

    // Noise tensor (Philox), cover noise blend, per_S, repaint_src buffer
    ops_init_noise_and_repaint(ctx, reqs, batch_n, s);

    // DiT denoising loop. ops_dit_generate acquires the DiT from the store for
    // the duration of the loop and releases it on scope exit. Latents land in
    // s.output and stay in job memory until phase 2 feeds them to the VAE.
    if (ops_dit_generate(ctx, batch_n, s, cancel, cancel_data) != 0) {
        delete job;
        return NULL;
    }

    return job;
}

// Phase 2: VAE decode all batch items and apply waveform splice for
// repaint/lego regions. Picks the padded source when the job carried one.
int ace_synth_job_run_vae(AceSynth *    ctx,
                          AceSynthJob * job,
                          const float * splice_src,
                          int           splice_len,
                          AceAudio *    out,
                          bool (*cancel)(void *),
                          void * cancel_data) {
    if (!ctx || !job || !out) {
        return -1;
    }

    // Outpainting: splice uses the padded source (silence at extended boundaries).
    const float * sp_audio = job->state.padded_src.empty() ? splice_src : job->state.padded_src.data();
    int           sp_len   = job->state.padded_src.empty() ? splice_len : (int) (job->state.padded_src.size() / 2);

    return ops_vae_decode_and_splice(ctx, job->batch_n, out, job->state, sp_audio, sp_len, cancel, cancel_data);
}

int ace_synth_job_batch_n(const AceSynthJob * job) {
    return job ? job->batch_n : 0;
}

void ace_synth_job_free(AceSynthJob * job) {
    delete job;
}

void ace_audio_free(AceAudio * audio) {
    if (audio && audio->samples) {
        free(audio->samples);
        audio->samples   = NULL;
        audio->n_samples = 0;
    }
}

void ace_synth_free(AceSynth * ctx) {
    if (!ctx) {
        return;
    }
    delete ctx;
}
