// pipeline-synth.cpp: ACE-Step synthesis pipeline implementation
//
// Wraps DiT + TextEncoder + CondEncoder + VAE for audio generation.

#include "pipeline-synth.h"

#include "bpe.h"
#include "cond-enc.h"
#include "debug.h"
#include "dit-sampler.h"
#include "dit.h"
#include "fsq-detok.h"
#include "fsq-tok.h"
#include "gguf-weights.h"
#include "philox.h"
#include "pipeline-synth-ops.h"
#include "qwen3-enc.h"
#include "request.h"
#include "task-types.h"
#include "timer.h"
#include "vae-enc.h"
#include "vae.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct AceSynth {
    // Models (loaded once)
    DiTGGML       dit;
    DiTGGMLConfig dit_cfg;
    Qwen3GGML     text_enc;
    CondGGML      cond_enc;
    VAEGGML       vae;
    DetokGGML     detok;
    TokGGML       tok;
    BPETokenizer  bpe;

    // Metadata from DiT GGUF
    bool               is_turbo;
    std::vector<float> silence_full;  // [15000, 64] f32

    // Config
    AceSynthParams params;
    bool           have_vae;
    bool           have_detok;
    bool           have_tok;

    // Derived constants
    int Oc;      // out_channels (64)
    int ctx_ch;  // in_channels - Oc (128)
};

void ace_synth_default_params(AceSynthParams * p) {
    p->text_encoder_path = NULL;
    p->dit_path          = NULL;
    p->vae_path          = NULL;
    p->lora_path         = NULL;
    p->lora_scale        = 1.0f;
    p->use_fa            = true;
    p->clamp_fp16        = false;
    p->vae_chunk         = 256;
    p->vae_overlap       = 64;
    p->dump_dir          = NULL;
}

AceSynth * ace_synth_load(const AceSynthParams * params) {
    if (!params->dit_path) {
        fprintf(stderr, "[Ace-Synth] ERROR: dit_path is NULL\n");
        return NULL;
    }
    if (!params->text_encoder_path) {
        fprintf(stderr, "[Ace-Synth] ERROR: text_encoder_path is NULL\n");
        return NULL;
    }

    AceSynth * ctx  = new AceSynth();
    ctx->params     = *params;
    ctx->have_vae   = false;
    ctx->have_detok = false;
    ctx->have_tok   = false;

    Timer timer;

    // Load DiT model (once for all requests)
    ctx->dit = {};
    dit_ggml_init_backend(&ctx->dit);
    if (!params->use_fa) {
        ctx->dit.use_flash_attn = false;
    }
    fprintf(stderr, "[Load] Backend init: %.1f ms\n", timer.ms());

    timer.reset();
    if (!dit_ggml_load(&ctx->dit, params->dit_path, ctx->dit_cfg, params->lora_path, params->lora_scale)) {
        fprintf(stderr, "[DiT] FATAL: failed to load model\n");
        delete ctx;
        return NULL;
    }
    fprintf(stderr, "[Load] DiT weight load: %.1f ms\n", timer.ms());

    ctx->Oc     = ctx->dit_cfg.out_channels;           // 64
    ctx->ctx_ch = ctx->dit_cfg.in_channels - ctx->Oc;  // 128

    // Read DiT GGUF metadata + silence_latent tensor (once)
    ctx->is_turbo = false;
    {
        GGUFModel gf = {};
        if (gf_load(&gf, params->dit_path)) {
            ctx->is_turbo        = gf_get_bool(gf, "acestep.is_turbo");
            const void * sl_data = gf_get_data(gf, "silence_latent");
            if (sl_data) {
                ctx->silence_full.resize(15000 * 64);
                memcpy(ctx->silence_full.data(), sl_data, 15000 * 64 * sizeof(float));
                fprintf(stderr, "[Load] silence_latent: [15000, 64] from GGUF\n");
            } else {
                fprintf(stderr, "[DiT] FATAL: silence_latent tensor not found in %s\n", params->dit_path);
                gf_close(&gf);
                dit_ggml_free(&ctx->dit);
                delete ctx;
                return NULL;
            }
            gf_close(&gf);
        } else {
            fprintf(stderr, "[DiT] FATAL: cannot reopen %s for metadata\n", params->dit_path);
            dit_ggml_free(&ctx->dit);
            delete ctx;
            return NULL;
        }
    }

    // Load VAE model (once for all requests)
    ctx->vae = {};
    if (params->vae_path) {
        timer.reset();
        vae_ggml_load(&ctx->vae, params->vae_path);
        fprintf(stderr, "[Load] VAE weights: %.1f ms\n", timer.ms());
        ctx->have_vae = true;
    }

    // BPE tokenizer
    timer.reset();
    if (!load_bpe_from_gguf(&ctx->bpe, params->text_encoder_path)) {
        fprintf(stderr, "[BPE] FATAL: failed to load tokenizer from %s\n", params->text_encoder_path);
        dit_ggml_free(&ctx->dit);
        if (ctx->have_vae) {
            vae_ggml_free(&ctx->vae);
        }
        delete ctx;
        return NULL;
    }
    fprintf(stderr, "[Load] BPE tokenizer: %.1f ms\n", timer.ms());

    // Text encoder forward (caption only)
    timer.reset();
    ctx->text_enc = {};
    qwen3_init_backend(&ctx->text_enc);
    if (!params->use_fa) {
        ctx->text_enc.use_flash_attn = false;
    }
    if (!qwen3_load_text_encoder(&ctx->text_enc, params->text_encoder_path)) {
        fprintf(stderr, "[TextEncoder] FATAL: failed to load\n");
        dit_ggml_free(&ctx->dit);
        if (ctx->have_vae) {
            vae_ggml_free(&ctx->vae);
        }
        delete ctx;
        return NULL;
    }
    fprintf(stderr, "[Load] TextEncoder: %.1f ms\n", timer.ms());

    // Condition encoder forward
    timer.reset();
    ctx->cond_enc = {};
    cond_ggml_init_backend(&ctx->cond_enc);
    if (!params->use_fa) {
        ctx->cond_enc.use_flash_attn = false;
    }
    ctx->cond_enc.clamp_fp16 = params->clamp_fp16;
    if (!cond_ggml_load(&ctx->cond_enc, params->dit_path)) {
        fprintf(stderr, "[CondEncoder] FATAL: failed to load\n");
        qwen3_free(&ctx->text_enc);
        dit_ggml_free(&ctx->dit);
        if (ctx->have_vae) {
            vae_ggml_free(&ctx->vae);
        }
        delete ctx;
        return NULL;
    }
    fprintf(stderr, "[Load] ConditionEncoder: %.1f ms\n", timer.ms());

    // Detokenizer (for audio_codes mode, weights in DiT GGUF)
    timer.reset();
    ctx->detok = {};
    if (detok_ggml_load(&ctx->detok, params->dit_path, ctx->dit.backend, ctx->dit.cpu_backend)) {
        if (!params->use_fa) {
            ctx->detok.use_flash_attn = false;
        }
        ctx->have_detok = true;
        fprintf(stderr, "[Load] Detokenizer: %.1f ms\n", timer.ms());
    }

    // Tokenizer (for FSQ roundtrip in cover mode, weights in DiT GGUF)
    timer.reset();
    ctx->tok = {};
    if (tok_ggml_load(&ctx->tok, params->dit_path, ctx->dit.backend, ctx->dit.cpu_backend)) {
        ctx->have_tok = true;
        fprintf(stderr, "[Load] Tokenizer: %.1f ms\n", timer.ms());
    }

    fprintf(stderr, "[Ace-Synth] All models loaded, turbo=%s\n", ctx->is_turbo ? "yes" : "no");
    if (!params->use_fa) {
        fprintf(stderr, "[Ace-Synth] flash attention disabled\n");
    }
    if (params->clamp_fp16) {
        fprintf(stderr, "[Ace-Synth] FP16 clamp enabled\n");
    }

    return ctx;
}

// ace_synth_generate: thin orchestrator
//
// Calls ops_ primitives in sequence. Mode routing is one if/else-if block.
// Adding a new mode = adding one branch here.
int ace_synth_generate(AceSynth *         ctx,
                       const AceRequest * reqs,
                       const float *      src_audio,
                       int                src_len,
                       const float *      ref_audio,
                       int                ref_len,
                       int                batch_n,
                       AceAudio *         out,
                       bool (*cancel)(void *),
                       void * cancel_data) {
    if (!ctx || !reqs || !out || batch_n < 1 || batch_n > 9) {
        return -1;
    }

    SynthState s;
    s.Oc     = ctx->Oc;
    s.ctx_ch = ctx->ctx_ch;
    debug_init(&s.dbg, ctx->params.dump_dir);

    // VAE encode source audio
    if (ops_encode_src(ctx, src_audio, src_len, s) != 0) {
        return -1;
    }

    // Shared request, mode flags, use_source_context
    // Shared params from first request (mode, duration, DiT settings).
    // Per-batch: caption, lyrics, metadata, audio_codes, and seed come from reqs[b].
    // seed must be resolved (non-negative) before calling this function.
    s.rr         = reqs[0];
    // task_type is the master. Empty = text2music.
    s.task       = s.rr.task_type.empty() ? std::string(TASK_TEXT2MUSIC) : s.rr.task_type;
    // only repaint uses region masking. complete uses full src context (Python behavior).
    s.is_repaint = (s.task == TASK_REPAINT);
    // lego with valid rs/re = region-constrained: generate only in zone, full audio context
    s.is_lego_region =
        (s.task == TASK_LEGO && s.rr.repainting_start >= 0.0f && s.rr.repainting_end > s.rr.repainting_start);
    s.rs                 = s.rr.repainting_start;
    s.re                 = s.rr.repainting_end;
    // use_source_context must be set before ops_resolve_T which uses it for T.
    // Mode routing refines it per task; this first pass covers all source-context tasks.
    s.use_source_context = (s.task == TASK_COVER || s.task == TASK_REPAINT || s.task == TASK_LEGO ||
                            s.task == TASK_EXTRACT || s.task == TASK_COMPLETE);
    // non-cover encoding pass (for audio_cover_strength < 1.0 switching) always uses text2music.
    s.nc_instruction_str = DIT_INSTR_TEXT2MUSIC;

    // Resolve DiT params (steps, guidance, shift) and scan audio codes
    if (ops_resolve_params(ctx, reqs, batch_n, s) != 0) {
        return -1;
    }

    // Promote text2music to cover when codes present (Python _resolve_generate_music_task)
    // DiT was trained with the cover instruction for codes guided generation.
    if (s.task == TASK_TEXT2MUSIC && s.have_codes) {
        s.task               = std::string(TASK_COVER);
        s.use_source_context = true;
    }

    // Timestep schedule
    ops_build_schedule(s);

    // Resolve latent frame count T
    if (ops_resolve_T(ctx, s) != 0) {
        return -1;
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
            s.use_source_context = true;
            s.instruction_str    = DIT_INSTR_COVER;
            ops_fsq_roundtrip(ctx, s);  // lossy FSQ: creative freedom + rhythmic sync
        } else if (s.task == TASK_REPAINT) {
            s.use_source_context = true;
            s.instruction_str    = DIT_INSTR_REPAINT;
            if (ops_clamp_region(s) != 0) {
                return -1;
            }
        } else if (s.task == TASK_LEGO) {
            s.use_source_context      = true;
            s.rr.audio_cover_strength = 1.0f;  // all DiT steps hear the backing track
            s.instruction_str         = dit_instr_lego(track_upper);
            if (!s.rr.track.empty()) {
                bool valid = false;
                for (int k = 0; k < TRACK_NAMES_COUNT; k++) {
                    if (s.rr.track == TRACK_NAMES[k]) {
                        valid = true;
                        break;
                    }
                }
                if (!valid) {
                    fprintf(stderr, "[%s] WARNING: '%s' is not a standard track name\n", s.task.c_str(),
                            s.rr.track.c_str());
                }
            }
            fprintf(stderr, "[Pipeline] task=%s\n", s.task.c_str());
            if (ctx->is_turbo) {
                fprintf(stderr, "[Pipeline] WARNING: lego requires base model, turbo output incoherent\n");
            }
            if (s.is_lego_region) {
                if (ops_clamp_region(s) != 0) {
                    return -1;
                }
            }
        } else if (s.task == TASK_EXTRACT) {
            s.use_source_context      = true;
            s.rr.audio_cover_strength = 1.0f;  // DiT sees the full mix
            s.instruction_str         = dit_instr_extract(track_upper);
            if (!s.rr.track.empty()) {
                bool valid = false;
                for (int k = 0; k < TRACK_NAMES_COUNT; k++) {
                    if (s.rr.track == TRACK_NAMES[k]) {
                        valid = true;
                        break;
                    }
                }
                if (!valid) {
                    fprintf(stderr, "[%s] WARNING: '%s' is not a standard track name\n", s.task.c_str(),
                            s.rr.track.c_str());
                }
            }
            fprintf(stderr, "[Pipeline] task=%s\n", s.task.c_str());
            if (ctx->is_turbo) {
                fprintf(stderr, "[Pipeline] WARNING: extract requires base model, turbo output incoherent\n");
            }
        } else if (s.task == TASK_COMPLETE) {
            s.use_source_context      = true;
            s.rr.audio_cover_strength = 1.0f;  // DiT sees the full isolated stem
            s.instruction_str         = dit_instr_complete(track_upper);
            if (!s.rr.track.empty()) {
                bool valid = false;
                for (int k = 0; k < TRACK_NAMES_COUNT; k++) {
                    if (s.rr.track == TRACK_NAMES[k]) {
                        valid = true;
                        break;
                    }
                }
                if (!valid) {
                    fprintf(stderr, "[%s] WARNING: '%s' is not a standard track name\n", s.task.c_str(),
                            s.rr.track.c_str());
                }
            }
            fprintf(stderr, "[Pipeline] task=%s\n", s.task.c_str());
            if (ctx->is_turbo) {
                fprintf(stderr, "[Pipeline] WARNING: complete requires base model, turbo output incoherent\n");
            }
        }
        // validation: tasks that need source audio or codes
        if (s.use_source_context && !s.have_cover && !s.have_codes) {
            fprintf(stderr, "[%s] ERROR: requires source audio or audio codes\n", s.task.c_str());
            return -1;
        }
    }

    // Resolve repaint quality params from repaint_strength.
    // Python: _resolve_repaint_config("balanced", strength).
    // 0.0 = aggressive, 0.5 = balanced (default), 1.0 = conservative.
    {
        float rs              = s.rr.repaint_strength;
        if (rs < 0.0f) { rs = 0.0f; }
        if (rs > 1.0f) { rs = 1.0f; }
        float inv             = 1.0f - rs;
        s.repaint_injection_ratio  = inv;
        s.repaint_crossfade_frames = (int) (25.0f * inv + 0.5f);
        s.repaint_wav_cf_sec       = 0.05f * inv;
    }

    // Encode timbre from ref_audio (independent of task)
    ops_encode_timbre(ctx, ref_audio, ref_len, s);

    // Per-batch text + lyric encoding (main + optional non-cover pass)
    if (ops_encode_text(ctx, reqs, batch_n, s) != 0) {
        return -1;
    }

    // Build DiT context [batch_n, T, ctx_ch] = src(64) | mask(64)
    if (ops_build_context(ctx, reqs, batch_n, s) != 0) {
        return -1;
    }

    // Silence context for audio_cover_strength switching (cover only)
    ops_build_context_silence(ctx, batch_n, s);

    // Noise tensor (Philox), cover noise blend, per_S, repaint_src buffer
    ops_init_noise_and_repaint(ctx, reqs, batch_n, s);

    // DiT denoising loop
    if (ops_dit_generate(ctx, batch_n, s, cancel, cancel_data) != 0) {
        return -1;
    }

    // VAE decode and waveform splice
    if (ops_vae_decode_and_splice(ctx, batch_n, out, s, src_audio, src_len, cancel, cancel_data) != 0) {
        return -1;
    }

    return 0;
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
    if (ctx->have_detok) {
        detok_ggml_free(&ctx->detok);
    }
    if (ctx->have_tok) {
        tok_ggml_free(&ctx->tok);
    }
    if (ctx->have_vae) {
        vae_ggml_free(&ctx->vae);
    }
    cond_ggml_free(&ctx->cond_enc);
    qwen3_free(&ctx->text_enc);
    dit_ggml_free(&ctx->dit);
    delete ctx;
}
