// pipeline-synth.cpp: ACE-Step synthesis pipeline implementation
//
// Thin orchestrator over a ModelStore. Holds no GPU module, no CPU-cached
// DiT state of its own: the store exposes DiTMeta (silence, null_cond, cfg,
// is_turbo) and each op acquires the GPU modules it needs on the fly.
//
// One function per task. Each task reads its inputs, poses its flags on
// SynthState, then calls the ops in a linear sequence. The dispatcher at the
// bottom picks the right task function from reqs[0].task_type.

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
    p->vae_chunk         = 1024;
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

// Allocate job and init the SynthState fields every task poses the same way.
static AceSynthJob * alloc_job(AceSynth * ctx, const AceRequest * reqs, int batch_n) {
    AceSynthJob * job = new AceSynthJob();
    job->batch_n      = batch_n;
    SynthState & s    = job->state;
    s.Oc              = ctx->Oc;
    s.ctx_ch          = ctx->ctx_ch;
    s.left_pad_sec    = 0.0f;
    s.rr              = reqs[0];
    s.rs              = s.rr.repainting_start;
    s.re              = s.rr.repainting_end;
    s.is_repaint      = false;
    s.is_lego_region  = false;
    s.have_cover      = false;
    s.T_cover         = 0;
    debug_init(&s.dbg, ctx->params.dump_dir);
    return job;
}

// Outpainting: pad src_audio with silence when the region extends beyond
// source bounds. Returns the (possibly padded) buffer and length via out
// Pad source for outpainting when the region extends beyond the source bounds.
// Audio path: zero-pads s.padded_src in samples, the VAE encoder will turn
// silence audio into ~silence latents.
// Latent path: pads s.cover_latents directly with the precomputed silence_full
// latent (the canonical VAE encoding of audio silence). No VAE encode needed.
// Sets s.left_pad_sec for adjust_region_coords.
static void apply_outpainting_padding(const AceSynth *   ctx,
                                      const AceRequest & r,
                                      const float *      src_audio,
                                      int                src_len,
                                      const float *      src_latents,
                                      int                src_T_latent,
                                      SynthState &       s,
                                      const float *&     enc_audio,
                                      int &              enc_len,
                                      const float *&     enc_latents,
                                      int &              enc_T_latent) {
    enc_audio    = src_audio;
    enc_len      = src_len;
    enc_latents  = src_latents;
    enc_T_latent = src_T_latent;

    float src_dur =
        (src_latents && src_T_latent > 0) ? (float) src_T_latent * 1920.0f / 48000.0f : (float) src_len / 48000.0f;
    float rs_raw   = r.repainting_start;
    float re_raw   = r.repainting_end;
    float end_time = (re_raw < 0.0f) ? src_dur : re_raw;
    float lpad     = (rs_raw < 0.0f) ? -rs_raw : 0.0f;
    float rpad     = (end_time > src_dur) ? end_time - src_dur : 0.0f;
    s.left_pad_sec = lpad;

    if (src_latents && src_T_latent > 0) {
        if (lpad <= 0.0f && rpad <= 0.0f) {
            return;
        }
        // Latent pad: round to 25Hz frames, splice silence_full at the edges.
        int           lpad_T  = (int) (lpad * 25.0f + 0.5f);
        int           rpad_T  = (int) (rpad * 25.0f + 0.5f);
        int           total_T = lpad_T + src_T_latent + rpad_T;
        const float * sil     = ctx->meta->silence_full.data();
        s.padded_latents.resize((size_t) total_T * 64);
        memcpy(s.padded_latents.data(), sil, (size_t) lpad_T * 64 * sizeof(float));
        memcpy(s.padded_latents.data() + (size_t) lpad_T * 64, src_latents, (size_t) src_T_latent * 64 * sizeof(float));
        memcpy(s.padded_latents.data() + (size_t) (lpad_T + src_T_latent) * 64, sil,
               (size_t) rpad_T * 64 * sizeof(float));
        enc_latents  = s.padded_latents.data();
        enc_T_latent = total_T;
        fprintf(stderr, "[Outpaint] latent pad left=%.1fs (%d) right=%.1fs (%d) total=%.1fs\n", lpad, lpad_T, rpad,
                rpad_T, (float) total_T * 1920.0f / 48000.0f);
        return;
    }

    // Audio path: always populate s.padded_src so the post-decode waveform
    // splice always has the source PCM to read from, with or without outpaint.
    int lpad_s       = (int) (lpad * 48000.0f);
    int rpad_s       = (int) (rpad * 48000.0f);
    int padded_total = src_len + lpad_s + rpad_s;
    s.padded_src.resize((size_t) padded_total * 2);
    if (lpad_s > 0 || rpad_s > 0) {
        memset(s.padded_src.data(), 0, s.padded_src.size() * sizeof(float));
    }
    memcpy(s.padded_src.data() + (size_t) lpad_s * 2, src_audio, (size_t) src_len * 2 * sizeof(float));
    enc_audio = s.padded_src.data();
    enc_len   = padded_total;
    if (lpad > 0.0f || rpad > 0.0f) {
        fprintf(stderr, "[Outpaint] audio pad left=%.1fs right=%.1fs total=%.1fs\n", lpad, rpad,
                (float) padded_total / 48000.0f);
    }
}

// Shift region coords into the padded reference frame, resolve sentinel end
// (-1) to either left pad boundary (outpaint) or source end (inpaint).
// src_dur is the unpadded source duration in seconds, agnostic of audio vs
// latent input. Returns false when the resolved range is empty or inverted.
static bool adjust_region_coords(SynthState & s, float src_dur) {
    s.rs += s.left_pad_sec;
    if (s.re < 0.0f) {
        s.re = (s.rr.repainting_start < 0.0f) ? s.left_pad_sec : src_dur + s.left_pad_sec;
    } else {
        s.re += s.left_pad_sec;
    }
    if (s.re <= s.rs) {
        fprintf(stderr, "[Region] ERROR: end (%.1f) <= start (%.1f)\n", s.re, s.rs);
        return false;
    }
    fprintf(stderr, "[Region] %.1fs..%.1fs (canvas=%.1fs)\n", s.rs, s.re, (float) s.T_cover * 1920.0f / 48000.0f);
    return true;
}

// Uppercase track name for instruction templates, warn on unknown names.
static std::string prepare_track(const std::string & track, const char * label) {
    std::string upper = track;
    for (char & ch : upper) {
        ch = (char) toupper((unsigned char) ch);
    }
    validate_track_names(track, label);
    return upper;
}

// Warn when a stem task runs on a turbo model: the training objective does
// not cover stem isolation, output degrades to incoherent noise.
static void warn_if_turbo_stem(const AceSynth * ctx, const char * task_name) {
    if (ctx->meta->is_turbo) {
        fprintf(stderr, "[Synth-Run] WARNING: %s requires base model, turbo output incoherent\n", task_name);
    }
}

// Pin VAE-Enc across two back-to-back encodes (source then timbre) so STRICT
// does not unload and reload the 160 MB weights between them. On return the
// pin is released, VAE-Enc can be evicted by the next require. The pin is
// only justified when at least one side actually goes through the encoder:
// when both src and ref arrive as pre-encoded latents, no pin is needed.
static bool pinned_encode_src_and_timbre(AceSynth *    ctx,
                                         const float * src_audio,
                                         int           src_len,
                                         const float * src_latents,
                                         int           src_T_latent,
                                         const float * ref_audio,
                                         int           ref_len,
                                         const float * ref_latents,
                                         int           ref_T_latent,
                                         SynthState &  s) {
    bool have_src_audio   = src_audio && src_len > 0;
    bool have_src_latents = src_latents && src_T_latent > 0;
    bool have_src         = have_src_audio || have_src_latents;
    bool have_ref_audio   = ref_audio && ref_len > 0;
    bool have_ref_latents = ref_latents && ref_T_latent > 0;
    bool have_ref         = have_ref_audio || have_ref_latents;
    if (!have_src && !have_ref) {
        // Neither encode touches the GPU: timbre takes the silence path.
        ops_encode_timbre(ctx, NULL, 0, NULL, 0, s);
        return true;
    }
    bool        need_vae_pin = (have_src_audio || have_ref_audio);
    ModelHandle vae_pin(ctx->store, need_vae_pin ? store_require_vae_enc(ctx->store, ctx->vae_enc_key) : NULL);
    if (need_vae_pin && !vae_pin.ptr) {
        fprintf(stderr, "[Pipeline-Synth] FATAL: store_require_vae_enc (pin) failed\n");
        return false;
    }
    if (have_src) {
        if (ops_encode_src(ctx, src_audio, src_len, src_latents, src_T_latent, s) != 0) {
            return false;
        }
    }
    ops_encode_timbre(ctx, ref_audio, ref_len, ref_latents, ref_T_latent, s);
    return true;
}

// Common tail every task ends with once its inputs are encoded and flags are
// posed: resolve params, resolve T, build schedule, encode text, build
// context, init noise, run DiT. Returns 0 on success, -1 on error/cancel.
static int run_tail(AceSynth *         ctx,
                    const AceRequest * reqs,
                    int                batch_n,
                    SynthState &       s,
                    bool (*cancel)(void *),
                    void * cancel_data) {
    if (ops_resolve_params(ctx, reqs, batch_n, s) != 0) {
        return -1;
    }
    if (ops_resolve_T(ctx, s) != 0) {
        return -1;
    }
    ops_build_schedule(s);
    if (ops_encode_text(ctx, reqs, batch_n, s) != 0) {
        return -1;
    }
    if (ops_build_context(ctx, reqs, batch_n, s) != 0) {
        return -1;
    }
    ops_build_context_silence(ctx, batch_n, s);
    ops_init_noise(ctx, reqs, batch_n, s);
    if (ops_dit_generate(ctx, batch_n, s, cancel, cancel_data) != 0) {
        return -1;
    }
    return 0;
}

// text2music: pure generation. No src audio. Optional timbre reference.
// Audio codes from LM (or absent) condition the DiT context.
static AceSynthJob * run_text2music(AceSynth *         ctx,
                                    const AceRequest * reqs,
                                    const float *      ref_audio,
                                    int                ref_len,
                                    const float *      ref_latents,
                                    int                ref_T_latent,
                                    int                batch_n,
                                    bool (*cancel)(void *),
                                    void * cancel_data) {
    AceSynthJob * job    = alloc_job(ctx, reqs, batch_n);
    SynthState &  s      = job->state;
    // audio_codes from the LM produce a latent context; empty codes fall back
    // to silence (DiT-only). The DiT was trained with the cover instruction on
    // latent context, so the two flags cascade from the same condition.
    s.use_source_context = !reqs[0].audio_codes.empty();
    s.instruction_str    = s.use_source_context ? DIT_INSTR_COVER : DIT_INSTR_TEXT2MUSIC;

    if (!pinned_encode_src_and_timbre(ctx, NULL, 0, NULL, 0, ref_audio, ref_len, ref_latents, ref_T_latent, s)) {
        delete job;
        return NULL;
    }
    if (run_tail(ctx, reqs, batch_n, s, cancel, cancel_data) != 0) {
        delete job;
        return NULL;
    }
    return job;
}

// cover: src audio recomposed with FSQ roundtrip degrading the context, so
// the DiT diverges from the original while staying thematically aligned.
static AceSynthJob * run_cover(AceSynth *         ctx,
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
                               bool (*cancel)(void *),
                               void * cancel_data) {
    bool have_src = (src_audio && src_len > 0) || (src_latents && src_T_latent > 0);
    if (!have_src) {
        fprintf(stderr, "[Synth-Run] ERROR: task 'cover' requires source audio or latents\n");
        return NULL;
    }
    AceSynthJob * job    = alloc_job(ctx, reqs, batch_n);
    SynthState &  s      = job->state;
    s.use_source_context = true;
    s.instruction_str    = DIT_INSTR_COVER;

    if (!pinned_encode_src_and_timbre(ctx, src_audio, src_len, src_latents, src_T_latent, ref_audio, ref_len,
                                      ref_latents, ref_T_latent, s)) {
        delete job;
        return NULL;
    }

    // Snapshot clean VAE latents before the FSQ roundtrip degrades them.
    // cover_noise_strength blending needs the clean copy.
    s.noise_blend_latents = s.cover_latents;
    ops_fsq_roundtrip(ctx, s);

    if (run_tail(ctx, reqs, batch_n, s, cancel, cancel_data) != 0) {
        delete job;
        return NULL;
    }
    return job;
}

// cover-nofsq: cover without the FSQ roundtrip. DiT works on clean 25Hz VAE
// latents, output stays close to source structure and timbre. Pass
// ref_audio = src_audio for best results.
static AceSynthJob * run_cover_nofsq(AceSynth *         ctx,
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
                                     bool (*cancel)(void *),
                                     void * cancel_data) {
    bool have_src = (src_audio && src_len > 0) || (src_latents && src_T_latent > 0);
    if (!have_src) {
        fprintf(stderr, "[Synth-Run] ERROR: task 'cover-nofsq' requires source audio or latents\n");
        return NULL;
    }
    AceSynthJob * job    = alloc_job(ctx, reqs, batch_n);
    SynthState &  s      = job->state;
    s.use_source_context = true;
    s.instruction_str    = DIT_INSTR_COVER;

    if (!pinned_encode_src_and_timbre(ctx, src_audio, src_len, src_latents, src_T_latent, ref_audio, ref_len,
                                      ref_latents, ref_T_latent, s)) {
        delete job;
        return NULL;
    }
    if (run_tail(ctx, reqs, batch_n, s, cancel, cancel_data) != 0) {
        delete job;
        return NULL;
    }
    return job;
}

// repaint: region-bounded inpaint/outpaint. Source padded with silence at the
// edges when the region extends beyond bounds, audio-side or latent-side
// depending on what the caller provided. DiT regenerates inside the region,
// phase 2 splices the result with the source latents and decodes once.
static AceSynthJob * run_repaint(AceSynth *         ctx,
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
                                 bool (*cancel)(void *),
                                 void * cancel_data) {
    bool have_audio   = src_audio && src_len > 0;
    bool have_latents = src_latents && src_T_latent > 0;
    if (!have_audio && !have_latents) {
        fprintf(stderr, "[Synth-Run] ERROR: task 'repaint' requires source audio or src_latents\n");
        return NULL;
    }
    AceSynthJob * job    = alloc_job(ctx, reqs, batch_n);
    SynthState &  s      = job->state;
    s.is_repaint         = true;
    s.use_source_context = true;
    s.instruction_str    = DIT_INSTR_REPAINT;

    const float * enc_audio    = NULL;
    int           enc_len      = 0;
    const float * enc_latents  = NULL;
    int           enc_T_latent = 0;
    apply_outpainting_padding(ctx, reqs[0], src_audio, src_len, src_latents, src_T_latent, s, enc_audio, enc_len,
                              enc_latents, enc_T_latent);

    if (!pinned_encode_src_and_timbre(ctx, enc_audio, enc_len, enc_latents, enc_T_latent, ref_audio, ref_len,
                                      ref_latents, ref_T_latent, s)) {
        delete job;
        return NULL;
    }
    float src_dur = have_latents ? (float) src_T_latent * 1920.0f / 48000.0f : (float) src_len / 48000.0f;
    if (!adjust_region_coords(s, src_dur)) {
        delete job;
        return NULL;
    }
    if (run_tail(ctx, reqs, batch_n, s, cancel, cancel_data) != 0) {
        delete job;
        return NULL;
    }
    return job;
}

// lego: stem generation. With valid rs/re: region-constrained, DiT generates
// only in the zone with full audio context. Without: whole-song generation.
// audio_cover_strength forced to 1.0 so all DiT steps hear the backing track.
// Both src_audio and src_latents are accepted; in region mode the source is
// padded with silence at the edges when the region extends beyond bounds.
static AceSynthJob * run_lego(AceSynth *         ctx,
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
                              bool (*cancel)(void *),
                              void * cancel_data) {
    bool have_audio   = src_audio && src_len > 0;
    bool have_latents = src_latents && src_T_latent > 0;
    if (!have_audio && !have_latents) {
        fprintf(stderr, "[Synth-Run] ERROR: task 'lego' requires source audio or src_latents\n");
        return NULL;
    }
    AceSynthJob * job       = alloc_job(ctx, reqs, batch_n);
    SynthState &  s         = job->state;
    s.is_lego_region        = (s.rr.repainting_end > s.rr.repainting_start);
    s.use_source_context    = true;
    std::string track_upper = prepare_track(s.rr.track, "Lego");
    s.instruction_str       = dit_instr_lego(track_upper);
    fprintf(stderr, "[Synth-Run] task=%s\n", reqs[0].task_type.c_str());
    warn_if_turbo_stem(ctx, "lego");

    const float * enc_audio    = src_audio;
    int           enc_len      = src_len;
    const float * enc_latents  = src_latents;
    int           enc_T_latent = src_T_latent;
    if (s.is_lego_region) {
        apply_outpainting_padding(ctx, reqs[0], src_audio, src_len, src_latents, src_T_latent, s, enc_audio, enc_len,
                                  enc_latents, enc_T_latent);
    }

    if (!pinned_encode_src_and_timbre(ctx, enc_audio, enc_len, enc_latents, enc_T_latent, ref_audio, ref_len,
                                      ref_latents, ref_T_latent, s)) {
        delete job;
        return NULL;
    }
    float src_dur = have_latents ? (float) src_T_latent * 1920.0f / 48000.0f : (float) src_len / 48000.0f;
    if (s.is_lego_region && !adjust_region_coords(s, src_dur)) {
        delete job;
        return NULL;
    }
    if (run_tail(ctx, reqs, batch_n, s, cancel, cancel_data) != 0) {
        delete job;
        return NULL;
    }
    return job;
}

// extract: stem isolation from a full mix.
static AceSynthJob * run_extract(AceSynth *         ctx,
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
                                 bool (*cancel)(void *),
                                 void * cancel_data) {
    bool have_src = (src_audio && src_len > 0) || (src_latents && src_T_latent > 0);
    if (!have_src) {
        fprintf(stderr, "[Synth-Run] ERROR: task 'extract' requires source audio or latents\n");
        return NULL;
    }
    AceSynthJob * job       = alloc_job(ctx, reqs, batch_n);
    SynthState &  s         = job->state;
    s.use_source_context    = true;
    std::string track_upper = prepare_track(s.rr.track, "Extract");
    s.instruction_str       = dit_instr_extract(track_upper);
    fprintf(stderr, "[Synth-Run] task=%s\n", reqs[0].task_type.c_str());
    warn_if_turbo_stem(ctx, "extract");

    if (!pinned_encode_src_and_timbre(ctx, src_audio, src_len, src_latents, src_T_latent, ref_audio, ref_len,
                                      ref_latents, ref_T_latent, s)) {
        delete job;
        return NULL;
    }
    if (run_tail(ctx, reqs, batch_n, s, cancel, cancel_data) != 0) {
        delete job;
        return NULL;
    }
    return job;
}

// complete: extend an isolated stem with more content.
static AceSynthJob * run_complete(AceSynth *         ctx,
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
                                  bool (*cancel)(void *),
                                  void * cancel_data) {
    bool have_src = (src_audio && src_len > 0) || (src_latents && src_T_latent > 0);
    if (!have_src) {
        fprintf(stderr, "[Synth-Run] ERROR: task 'complete' requires source audio or latents\n");
        return NULL;
    }
    AceSynthJob * job       = alloc_job(ctx, reqs, batch_n);
    SynthState &  s         = job->state;
    s.use_source_context    = true;
    std::string track_upper = prepare_track(s.rr.track, "Complete");
    s.instruction_str       = dit_instr_complete(track_upper);
    fprintf(stderr, "[Synth-Run] task=%s\n", reqs[0].task_type.c_str());
    warn_if_turbo_stem(ctx, "complete");

    if (!pinned_encode_src_and_timbre(ctx, src_audio, src_len, src_latents, src_T_latent, ref_audio, ref_len,
                                      ref_latents, ref_T_latent, s)) {
        delete job;
        return NULL;
    }
    if (run_tail(ctx, reqs, batch_n, s, cancel, cancel_data) != 0) {
        delete job;
        return NULL;
    }
    return job;
}

// Phase 1 entry point. Dispatches on reqs[0].task_type to the right task
// function. task_type is always set: request_init defaults it to text2music,
// the JSON parser ignores empty strings.
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
                                    bool (*cancel)(void *),
                                    void * cancel_data) {
    if (!ctx || !reqs || batch_n < 1 || batch_n > 9) {
        return NULL;
    }
    const std::string & task = reqs[0].task_type;
    if (task == TASK_TEXT2MUSIC) {
        return run_text2music(ctx, reqs, ref_audio, ref_len, ref_latents, ref_T_latent, batch_n, cancel, cancel_data);
    }
    if (task == TASK_COVER) {
        return run_cover(ctx, reqs, src_audio, src_len, src_latents, src_T_latent, ref_audio, ref_len, ref_latents,
                         ref_T_latent, batch_n, cancel, cancel_data);
    }
    if (task == TASK_COVER_NOFSQ) {
        return run_cover_nofsq(ctx, reqs, src_audio, src_len, src_latents, src_T_latent, ref_audio, ref_len,
                               ref_latents, ref_T_latent, batch_n, cancel, cancel_data);
    }
    if (task == TASK_REPAINT) {
        return run_repaint(ctx, reqs, src_audio, src_len, src_latents, src_T_latent, ref_audio, ref_len, ref_latents,
                           ref_T_latent, batch_n, cancel, cancel_data);
    }
    if (task == TASK_LEGO) {
        return run_lego(ctx, reqs, src_audio, src_len, src_latents, src_T_latent, ref_audio, ref_len, ref_latents,
                        ref_T_latent, batch_n, cancel, cancel_data);
    }
    if (task == TASK_EXTRACT) {
        return run_extract(ctx, reqs, src_audio, src_len, src_latents, src_T_latent, ref_audio, ref_len, ref_latents,
                           ref_T_latent, batch_n, cancel, cancel_data);
    }
    if (task == TASK_COMPLETE) {
        return run_complete(ctx, reqs, src_audio, src_len, src_latents, src_T_latent, ref_audio, ref_len, ref_latents,
                            ref_T_latent, batch_n, cancel, cancel_data);
    }
    fprintf(stderr, "[Synth-Run] ERROR: unknown task_type '%s'\n", task.c_str());
    return NULL;
}

// Pointer to the post-DiT latent for one track in s.output, time-major
// [T*64] f32. Lives until ace_synth_job_free.
const float * ace_synth_job_get_latent(const AceSynthJob * job, int track_idx, int * T_out) {
    const SynthState & s = job->state;
    *T_out               = s.T;
    return s.output.data() + (size_t) track_idx * s.T * s.Oc;
}

// Phase 2: latent splice (for repaint/lego) + VAE decode for every batch item.
int ace_synth_job_run_vae(AceSynth *    ctx,
                          AceSynthJob * job,
                          AceAudio *    out,
                          bool (*cancel)(void *),
                          void * cancel_data) {
    if (!ctx || !job || !out) {
        return -1;
    }
    return ops_vae_decode(ctx, job->batch_n, out, job->state, cancel, cancel_data);
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
