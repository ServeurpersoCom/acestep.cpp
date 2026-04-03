// pipeline-synth.cpp: ACE-Step synthesis pipeline implementation
//
// Wraps DiT + TextEncoder + CondEncoder + VAE for audio generation.

#include "pipeline-synth-ops.h"

#include "bpe.h"
#include "cond-enc.h"
#include "debug.h"
#include "dit-sampler.h"
#include "dit.h"
#include "fsq-detok.h"
#include "fsq-tok.h"
#include "gguf-weights.h"
#include "philox.h"
#include "pipeline-synth.h"
#include "qwen3-enc.h"
#include "request.h"
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

// static helpers
static const int FRAMES_PER_SECOND = 25;

static std::vector<int> parse_codes_string(const std::string & s) {
    std::vector<int> codes;
    if (s.empty()) {
        return codes;
    }
    const char * p = s.c_str();
    while (*p) {
        while (*p == ',' || *p == ' ') {
            p++;
        }
        if (!*p) {
            break;
        }
        codes.push_back(atoi(p));
        while (*p && *p != ',') {
            p++;
        }
    }
    return codes;
}

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

// ops_encode_src
int ops_encode_src(AceSynth * ctx, const float * src_audio, int src_len, SynthState & s) {
    // Cover mode: load VAE encoder and encode source audio
    s.have_cover = false;
    s.T_cover    = 0;
    if (src_audio && src_len > 0) {
        s.timer.reset();
        int T_audio = src_len;

        VAEEncoder vae_enc = {};
        vae_enc_load(&vae_enc, ctx->params.vae_path);
        int max_T_lat = (T_audio / 1920) + 64;
        s.cover_latents.resize(max_T_lat * 64);

        s.T_cover = vae_enc_encode_tiled(&vae_enc, src_audio, T_audio, s.cover_latents.data(), max_T_lat,
                                         ctx->params.vae_chunk, ctx->params.vae_overlap);
        vae_enc_free(&vae_enc);
        if (s.T_cover < 0) {
            fprintf(stderr, "[VAE-Enc] FATAL: encode failed\n");
            return -1;
        }
        s.cover_latents.resize(s.T_cover * 64);
        fprintf(stderr, "[Cover] Encoded: T_cover=%d (%.2fs), %.1f ms\n", s.T_cover,
                (float) s.T_cover * 1920.0f / 48000.0f, s.timer.ms());

        s.have_cover = true;
    }

    return 0;
}

// ops_fsq_roundtrip
void ops_fsq_roundtrip(AceSynth * ctx, SynthState & s) {
    // FSQ roundtrip for cover: tokenize (25Hz->5Hz) + detokenize (5Hz->25Hz).
    // The lossy FSQ bottleneck gives the DiT creative freedom to produce a harmonic
    // reinterpretation of the source rather than a rigid copy. The s.output stays
    // rhythmically and melodically synchronized with the original (playable in sync),
    // while the DiT works on its training-distribution latents instead of clean VAE s.output.
    // Other tasks (lego, extract, repaint, complete) use clean latents directly.
    if (s.have_cover && ctx->have_tok && ctx->have_detok) {
        s.timer.reset();
        int              T_5Hz = (s.T_cover + 4) / 5;
        std::vector<int> codes(T_5Hz);
        int              T_5Hz_actual =
            tok_ggml_encode(&ctx->tok, s.cover_latents.data(), s.T_cover, codes.data(), ctx->silence_full.data());
        if (T_5Hz_actual > 0) {
            int                T_25Hz_rt = T_5Hz_actual * 5;
            std::vector<float> rt_latents(T_25Hz_rt * 64);
            int                ret = detok_ggml_decode(&ctx->detok, codes.data(), T_5Hz_actual, rt_latents.data());
            if (ret >= 0) {
                int copy_T = T_25Hz_rt < s.T_cover ? T_25Hz_rt : s.T_cover;
                memcpy(s.cover_latents.data(), rt_latents.data(), (size_t) copy_T * 64 * sizeof(float));
                fprintf(stderr, "[Cover] FSQ roundtrip: %d->%d->%d frames, %.1f ms\n", s.T_cover, T_5Hz_actual, copy_T,
                        s.timer.ms());
            }
        }
    }
}

// ops_resolve_params
int ops_resolve_params(AceSynth * ctx, const AceRequest * reqs, int batch_n, SynthState & s) {
    // Extract shared params from first request
    s.duration = s.rr.duration > 0 ? s.rr.duration : 30.0f;

    // Resolve DiT sampling params: 0 = auto-detect from model type.
    // Turbo: 8 steps, guidance=1.0, s.shift=3.0
    // Base/SFT: 50 steps, guidance=1.0, s.shift=1.0
    s.num_steps      = s.rr.inference_steps;
    s.guidance_scale = s.rr.guidance_scale;
    s.shift          = s.rr.shift;

    if (s.num_steps <= 0) {
        s.num_steps = ctx->is_turbo ? 8 : 50;
    }
    if (s.num_steps > 100) {
        fprintf(stderr, "[Pipeline] WARNING: inference_steps %d clamped to 100\n", s.num_steps);
        s.num_steps = 100;
    }

    if (s.guidance_scale <= 0.0f) {
        s.guidance_scale = 1.0f;
    } else if (ctx->is_turbo && s.guidance_scale > 1.0f) {
        fprintf(stderr, "[Pipeline] WARNING: turbo model, forcing guidance_scale=1.0 (was %.1f)\n", s.guidance_scale);
        s.guidance_scale = 1.0f;
    }

    if (s.shift <= 0.0f) {
        s.shift = ctx->is_turbo ? 3.0f : 1.0f;
    }

    // Audio codes: scan all requests to determine s.T from the longest code set.
    // Per-batch codes are decoded in the s.context building loop below.
    // Shorter code sets are padded with silence, longer ones are never truncated.
    s.max_codes_len = 0;
    s.have_codes    = false;
    for (int b = 0; b < batch_n; b++) {
        std::vector<int> cb = parse_codes_string(reqs[b].audio_codes);
        if ((int) cb.size() > s.max_codes_len) {
            s.max_codes_len = (int) cb.size();
        }
        if (!cb.empty()) {
            s.have_codes = true;
        }
    }
    if (s.have_codes) {
        fprintf(stderr, "[Pipeline] max audio codes across batch: %d (%.1fs @ 5Hz)\n", s.max_codes_len,
                (float) s.max_codes_len / 5.0f);
    }
    if (s.have_codes && !ctx->have_detok) {
        fprintf(stderr, "[Detokenizer] FATAL: failed to load\n");
        return -1;
    }

    return 0;
}

// ops_build_schedule
void ops_build_schedule(SynthState & s) {
    // Build s.schedule: t_i = s.shift * t / (1 + (s.shift-1)*t) where t = 1 - i/steps
    s.schedule.resize(s.num_steps);
    for (int i = 0; i < s.num_steps; i++) {
        float t       = 1.0f - (float) i / (float) s.num_steps;
        s.schedule[i] = s.shift * t / (1.0f + (s.shift - 1.0f) * t);
    }
}

// ops_resolve_T
int ops_resolve_T(AceSynth * ctx, SynthState & s) {
    // s.T = number of 25Hz latent frames for DiT
    // Source tasks: from source audio. Codes: from code count. Else: from s.duration.
    if (s.use_source_context && s.have_cover) {
        s.T        = s.T_cover;
        // s.duration in metas must match actual source length, not JSON default
        s.duration = (float) s.T_cover / (float) FRAMES_PER_SECOND;
    } else if (s.have_codes) {
        s.T = s.max_codes_len * 5;
    } else if (s.use_source_context) {
        // source context requested but neither cover_latents nor codes available.
        // duration fallthrough would produce a meaningless T for source tasks.
        fprintf(stderr, "[Pipeline] FATAL: use_source_context but no cover_latents and no audio_codes\n");
        return -1;
    } else {
        s.T = (int) (s.duration * FRAMES_PER_SECOND);
    }
    s.T     = ((s.T + ctx->dit_cfg.patch_size - 1) / ctx->dit_cfg.patch_size) * ctx->dit_cfg.patch_size;
    s.S     = s.T / ctx->dit_cfg.patch_size;
    s.enc_S = 0;

    fprintf(stderr, "[Pipeline] T=%d, S=%d\n", s.T, s.S);
    fprintf(stderr, "[Pipeline] seed=%lld, steps=%d, guidance=%.1f, shift=%.1f, duration=%.1fs\n",
            (long long) s.rr.seed, s.num_steps, s.guidance_scale, s.shift, s.duration);

    if (s.T > 15000) {
        fprintf(stderr, "[Pipeline] ERROR: T=%d exceeds silence_latent max 15000, skipping\n", s.T);
        return -1;
    }

    return 0;
}

// ops_clamp_region
int ops_clamp_region(SynthState & s) {
    // Clamp rs/re to source duration. Caller ensures this is a region task.
    float src_dur = (float) s.T_cover * 1920.0f / 48000.0f;
    if (s.rs < 0.0f) {
        s.rs = 0.0f;
    }
    if (s.re < 0.0f) {
        s.re = src_dur;
    }
    if (s.rs > src_dur) {
        s.rs = src_dur;
    }
    if (s.re > src_dur) {
        s.re = src_dur;
    }
    if (s.re <= s.rs) {
        fprintf(stderr, "[Repaint] ERROR: repainting_end (%.1f) <= repainting_start (%.1f)\n", s.re, s.rs);
        return -1;
    }
    fprintf(stderr, "[%s] Region: %.1fs..%.1fs (src=%.1fs)\n", s.task.c_str(), s.rs, s.re, src_dur);
    return 0;
}

// ops_encode_timbre
void ops_encode_timbre(AceSynth * ctx, const float * ref_audio, int ref_len, SynthState & s) {
    // 2. Timbre features from ref_audio (independent of src_audio).
    // ref_audio = timbre reference, VAE-encoded to latents then first 750 frames used.
    // NULL = silence (no timbre conditioning).
    s.timbre_feats.resize(S_REF_TIMBRE * 64);
    if (ref_audio && ref_len > 0) {
        s.timer.reset();
        VAEEncoder ref_vae = {};
        vae_enc_load(&ref_vae, ctx->params.vae_path);
        int                max_T_ref = (ref_len / 1920) + 64;
        std::vector<float> ref_latents(max_T_ref * 64);
        int                T_ref = vae_enc_encode_tiled(&ref_vae, ref_audio, ref_len, ref_latents.data(), max_T_ref,
                                                        ctx->params.vae_chunk, ctx->params.vae_overlap);
        vae_enc_free(&ref_vae);
        if (T_ref < 0) {
            fprintf(stderr, "[Timbre] WARNING: ref_audio encode failed, using silence\n");
            memcpy(s.timbre_feats.data(), ctx->silence_full.data(), S_REF_TIMBRE * 64 * sizeof(float));
        } else {
            int copy_n = T_ref < S_REF_TIMBRE ? T_ref : S_REF_TIMBRE;
            memcpy(s.timbre_feats.data(), ref_latents.data(), (size_t) copy_n * 64 * sizeof(float));
            if (copy_n < S_REF_TIMBRE) {
                memcpy(s.timbre_feats.data() + (size_t) copy_n * 64, ctx->silence_full.data() + (size_t) copy_n * 64,
                       (size_t) (S_REF_TIMBRE - copy_n) * 64 * sizeof(float));
            }
            fprintf(stderr, "[Timbre] ref_audio: %d frames (%.1fs), %.1f ms\n", copy_n, (float) copy_n / 25.0f,
                    s.timer.ms());
        }
    } else {
        memcpy(s.timbre_feats.data(), ctx->silence_full.data(), S_REF_TIMBRE * 64 * sizeof(float));
    }
}

// ops_encode_text
int ops_encode_text(AceSynth * ctx, const AceRequest * reqs, int batch_n, SynthState & s) {
    // 3. Per-batch text encoding.
    // Each batch element gets its own caption, lyrics, and metadata encoded independently.
    // TextEncoder + CondEncoder run in series (cheap: ~13ms per element).
    // Results are padded to s.max_enc_S with null_cond and stacked for a single DiT batch pass.
    int H_text = ctx->text_enc.cfg.hidden_size;  // 1024
    int H_cond = ctx->dit.cfg.hidden_size;       // 2048

    // read null_condition_emb from GPU for padding shorter encodings
    s.null_cond_vec.resize(H_cond);
    if (ctx->dit.null_condition_emb) {
        int emb_n = (int) ggml_nelements(ctx->dit.null_condition_emb);
        if (ctx->dit.null_condition_emb->type == GGML_TYPE_BF16) {
            std::vector<uint16_t> bf16_buf(emb_n);
            ggml_backend_tensor_get(ctx->dit.null_condition_emb, bf16_buf.data(), 0, emb_n * sizeof(uint16_t));
            for (int i = 0; i < emb_n; i++) {
                uint32_t w = (uint32_t) bf16_buf[i] << 16;
                memcpy(&s.null_cond_vec[i], &w, 4);
            }
        } else {
            ggml_backend_tensor_get(ctx->dit.null_condition_emb, s.null_cond_vec.data(), 0, emb_n * sizeof(float));
        }
    }

    // instruction_str must be set by the orchestrator. Empty means unknown task or bug.
    if (s.instruction_str.empty()) {
        fprintf(stderr, "[Encode] FATAL: instruction_str is empty (unknown task or orchestrator bug)\n");
        return -1;
    }

    // encode each batch element independently
    s.per_enc.resize(batch_n);
    s.per_enc_S.resize(batch_n);

    for (int b = 0; b < batch_n; b++) {
        const AceRequest & rb = reqs[b];

        // per-batch metadata
        char bpm_b[16] = "N/A";
        if (rb.bpm > 0) {
            snprintf(bpm_b, sizeof(bpm_b), "%d", rb.bpm);
        }
        const char * keyscale_b = rb.keyscale.empty() ? "N/A" : rb.keyscale.c_str();
        const char * timesig_b  = rb.timesignature.empty() ? "N/A" : rb.timesignature.c_str();
        const char * language_b = rb.vocal_language.empty() ? "unknown" : rb.vocal_language.c_str();

        char metas_b[512];
        snprintf(metas_b, sizeof(metas_b), "- bpm: %s\n- timesignature: %s\n- keyscale: %s\n- duration: %d seconds\n",
                 bpm_b, timesig_b, keyscale_b, (int) s.duration);
        std::string text_str = std::string("# Instruction\n") + s.instruction_str + "\n\n" + "# Caption\n" +
                               rb.caption + "\n\n" + "# Metas\n" + metas_b + "<|endoftext|>\n";
        std::string lyric_str =
            std::string("# Languages\n") + language_b + "\n\n# Lyric\n" + rb.lyrics + "<|endoftext|>";

        // tokenize
        auto text_ids  = bpe_encode(&ctx->bpe, text_str.c_str(), true);
        auto lyric_ids = bpe_encode(&ctx->bpe, lyric_str.c_str(), true);
        int  S_text    = (int) text_ids.size();
        int  S_lyric   = (int) lyric_ids.size();

        // TextEncoder forward
        std::vector<float> text_hidden(H_text * S_text);
        qwen3_forward(&ctx->text_enc, text_ids.data(), S_text, text_hidden.data());

        // lyric embedding (vocab lookup)
        std::vector<float> lyric_embed(H_text * S_lyric);
        qwen3_embed_lookup(&ctx->text_enc, lyric_ids.data(), S_lyric, lyric_embed.data());

        // CondEncoder forward
        s.timer.reset();
        cond_ggml_forward(&ctx->cond_enc, text_hidden.data(), S_text, lyric_embed.data(), S_lyric,
                          s.timbre_feats.data(), S_REF_TIMBRE, s.per_enc[b], &s.per_enc_S[b]);
        fprintf(stderr, "[Encode Batch%d] %d+%d tokens -> enc_S=%d, %.1f ms\n", b, S_text, S_lyric, s.per_enc_S[b],
                s.timer.ms());

        if (b == 0) {
            debug_dump_2d(&s.dbg, "text_hidden", text_hidden.data(), S_text, H_text);
            debug_dump_2d(&s.dbg, "lyric_embed", lyric_embed.data(), S_lyric, H_text);
            debug_dump_2d(&s.dbg, "enc_hidden", s.per_enc[b].data(), s.per_enc_S[b], H_cond);
        }
    }

    // second encoding pass using s.nc_instruction_str (set by orchestrator).
    // used after s.cover_steps when audio_cover_strength < 1.0 (s.context switches to silence).
    s.need_enc_switch = s.use_source_context && !s.is_repaint && !s.is_lego_region && s.rr.audio_cover_strength < 1.0f;
    s.per_enc_nc.resize(batch_n);
    s.per_enc_S_nc.assign(batch_n, 0);

    if (s.need_enc_switch) {
        for (int b = 0; b < batch_n; b++) {
            const AceRequest & rb = reqs[b];

            char bpm_b[16] = "N/A";
            if (rb.bpm > 0) {
                snprintf(bpm_b, sizeof(bpm_b), "%d", rb.bpm);
            }
            const char * keyscale_b = rb.keyscale.empty() ? "N/A" : rb.keyscale.c_str();
            const char * timesig_b  = rb.timesignature.empty() ? "N/A" : rb.timesignature.c_str();
            const char * language_b = rb.vocal_language.empty() ? "unknown" : rb.vocal_language.c_str();

            char metas_b[512];
            snprintf(metas_b, sizeof(metas_b),
                     "- bpm: %s\n- timesignature: %s\n- keyscale: %s\n- duration: %d seconds\n", bpm_b, timesig_b,
                     keyscale_b, (int) s.duration);
            std::string text_str = std::string("# Instruction\n") + s.nc_instruction_str + "\n\n" + "# Caption\n" +
                                   rb.caption + "\n\n" + "# Metas\n" + metas_b + "<|endoftext|>\n";
            std::string lyric_str =
                std::string("# Languages\n") + language_b + "\n\n# Lyric\n" + rb.lyrics + "<|endoftext|>";

            auto text_ids  = bpe_encode(&ctx->bpe, text_str.c_str(), true);
            auto lyric_ids = bpe_encode(&ctx->bpe, lyric_str.c_str(), true);
            int  S_text    = (int) text_ids.size();
            int  S_lyric   = (int) lyric_ids.size();

            std::vector<float> text_hidden(H_text * S_text);
            qwen3_forward(&ctx->text_enc, text_ids.data(), S_text, text_hidden.data());

            std::vector<float> lyric_embed(H_text * S_lyric);
            qwen3_embed_lookup(&ctx->text_enc, lyric_ids.data(), S_lyric, lyric_embed.data());

            cond_ggml_forward(&ctx->cond_enc, text_hidden.data(), S_text, lyric_embed.data(), S_lyric,
                              s.timbre_feats.data(), S_REF_TIMBRE, s.per_enc_nc[b], &s.per_enc_S_nc[b]);
            fprintf(stderr, "[Encode Batch%d] non-cover: %d+%d tokens -> enc_S=%d\n", b, S_text, S_lyric,
                    s.per_enc_S_nc[b]);
        }
    }

    // find max s.enc_S across both encodings (cover + text2music),
    // pad shorter encodings with null_cond, stack into [H, s.max_enc_S, N]
    s.max_enc_S = 0;
    for (int b = 0; b < batch_n; b++) {
        if (s.per_enc_S[b] > s.max_enc_S) {
            s.max_enc_S = s.per_enc_S[b];
        }
        if (s.need_enc_switch && s.per_enc_S_nc[b] > s.max_enc_S) {
            s.max_enc_S = s.per_enc_S_nc[b];
        }
    }
    s.enc_S = s.max_enc_S;

    s.enc_hidden.resize(H_cond * s.max_enc_S * batch_n);
    for (int b = 0; b < batch_n; b++) {
        float * dst = s.enc_hidden.data() + b * s.max_enc_S * H_cond;
        memcpy(dst, s.per_enc[b].data(), (size_t) s.per_enc_S[b] * H_cond * sizeof(float));
        for (int si = s.per_enc_S[b]; si < s.max_enc_S; si++) {
            memcpy(dst + si * H_cond, s.null_cond_vec.data(), H_cond * sizeof(float));
        }
    }

    // pad and stack text2music encoding (same s.max_enc_S for graph compatibility)
    if (s.need_enc_switch) {
        s.enc_hidden_nc.resize(H_cond * s.max_enc_S * batch_n);
        s.per_enc_S_nc_final.resize(batch_n);
        for (int b = 0; b < batch_n; b++) {
            float * dst = s.enc_hidden_nc.data() + b * s.max_enc_S * H_cond;
            memcpy(dst, s.per_enc_nc[b].data(), (size_t) s.per_enc_S_nc[b] * H_cond * sizeof(float));
            for (int si = s.per_enc_S_nc[b]; si < s.max_enc_S; si++) {
                memcpy(dst + si * H_cond, s.null_cond_vec.data(), H_cond * sizeof(float));
            }
            s.per_enc_S_nc_final[b] = s.per_enc_S_nc[b];
        }
    }

    if (batch_n > 1) {
        fprintf(stderr, "[Encode] Per-batch encoding done: max_enc_S=%d\n", s.max_enc_S);
    }

    return 0;
}

// ops_build_context
int ops_build_context(AceSynth * ctx, const AceRequest * reqs, int batch_n, SynthState & s) {
    // Build s.context: [batch_n, s.T, s.ctx_ch] = src_latents[64] + chunk_mask[64]
    // Cover/Lego/Repaint: shared s.context replicated (s.cover_latents from src_audio).
    // Passthrough: per-batch detokenized FSQ codes + silence padding, mask = 1.0.
    // Text2music: silence only, mask = 1.0.
    s.repaint_t0 = 0, s.repaint_t1 = 0;
    if (s.is_repaint) {
        s.repaint_t0 = (int) (s.rs * 48000.0f / 1920.0f);
        s.repaint_t1 = (int) (s.re * 48000.0f / 1920.0f);
        if (s.repaint_t0 < 0) {
            s.repaint_t0 = 0;
        }
        if (s.repaint_t1 > s.T) {
            s.repaint_t1 = s.T;
        }
        if (s.repaint_t0 > s.T) {
            s.repaint_t0 = s.T;
        }
        fprintf(stderr, "[Repaint] Latent frames: [%d, %d) / %d\n", s.repaint_t0, s.repaint_t1, s.T);
    }

    s.context.resize(batch_n * s.T * s.ctx_ch);

    if (s.use_source_context && s.have_cover) {
        // Cover/Lego/Repaint: build once, replicate (s.cover_latents are shared)
        std::vector<float> context_single(s.T * s.ctx_ch);
        for (int t = 0; t < s.T; t++) {
            bool          in_region = (s.is_repaint || s.is_lego_region) && t >= s.repaint_t0 && t < s.repaint_t1;
            // repaint silences the zone (DiT generates fresh there).
            // lego keeps full cover everywhere (DiT hears the whole backing track).
            const float * src;
            if (s.is_repaint && in_region) {
                src = ctx->silence_full.data() + t * s.Oc;
            } else {
                src = (t < s.T_cover) ? s.cover_latents.data() + t * s.Oc : ctx->silence_full.data() + t * s.Oc;
            }
            // region tasks: explicit 0/1 mask. all others: 1.0 (training distribution).
            float mask_val;
            if (s.is_repaint || s.is_lego_region) {
                mask_val = in_region ? 1.0f : 0.0f;
            } else {
                mask_val = 1.0f;  // training distribution: only 0/1 seen during training
            }
            for (int c = 0; c < s.Oc; c++) {
                context_single[t * s.ctx_ch + c] = src[c];
            }
            for (int c = 0; c < s.Oc; c++) {
                context_single[t * s.ctx_ch + s.Oc + c] = mask_val;
            }
        }
        for (int b = 0; b < batch_n; b++) {
            memcpy(s.context.data() + b * s.T * s.ctx_ch, context_single.data(), s.T * s.ctx_ch * sizeof(float));
        }
    } else {
        // Per-batch context from audio_codes or silence (text2music).
        // use_source_context with neither cover nor codes is an invalid state:
        // the orchestrator promised source context but provided nothing to condition on.
        if (s.use_source_context && !s.have_codes) {
            fprintf(stderr, "[Context] FATAL: use_source_context but no cover_latents and no audio_codes\n");
            return -1;
        }

        // Text2music / codes passthrough: per-batch context with per-batch audio_codes
        for (int b = 0; b < batch_n; b++) {
            float * ctx_dst = s.context.data() + b * s.T * s.ctx_ch;

            // decode this batch item's audio codes (if any)
            int                decoded_T = 0;
            std::vector<float> decoded_latents;
            std::vector<int>   codes_b = parse_codes_string(reqs[b].audio_codes);
            if (!codes_b.empty()) {
                s.timer.reset();
                int T_5Hz        = (int) codes_b.size();
                int T_25Hz_codes = T_5Hz * 5;
                decoded_latents.resize(T_25Hz_codes * s.Oc);

                int ret = detok_ggml_decode(&ctx->detok, codes_b.data(), T_5Hz, decoded_latents.data());
                if (ret < 0) {
                    fprintf(stderr, "[Detokenizer Batch%d] FATAL: decode failed\n", b);
                    return -1;
                }
                fprintf(stderr, "[Context Batch%d] Detokenizer: %.1f ms, %d codes\n", b, s.timer.ms(), T_5Hz);

                decoded_T = T_25Hz_codes < s.T ? T_25Hz_codes : s.T;
                if (b == 0) {
                    debug_dump_2d(&s.dbg, "detok_output", decoded_latents.data(), T_25Hz_codes, s.Oc);
                }
            }

            // fill s.context: decoded latents then silence, mask = 1.0 (training distribution)
            for (int t = 0; t < s.T; t++) {
                const float * src = (t < decoded_T) ? decoded_latents.data() + t * s.Oc :
                                                      ctx->silence_full.data() + (t - decoded_T) * s.Oc;
                for (int c = 0; c < s.Oc; c++) {
                    ctx_dst[t * s.ctx_ch + c] = src[c];
                }
                for (int c = 0; c < s.Oc; c++) {
                    ctx_dst[t * s.ctx_ch + s.Oc + c] = 1.0f;
                }
            }
        }
    }

    return 0;
}

// ops_build_context_silence
void ops_build_context_silence(AceSynth * ctx, int batch_n, SynthState & s) {
    // Cover mode: build silence s.context for audio_cover_strength switching
    // When step >= s.cover_steps, DiT switches from cover s.context to silence s.context
    // Repaint/lego_region: mask handles region; s.context switch never applies
    s.cover_steps = -1;
    if (s.use_source_context && !s.is_repaint && !s.is_lego_region) {
        float cover_strength = s.rr.audio_cover_strength;
        if (cover_strength < 1.0f) {
            // Build silence s.context: all frames use silence_latent
            std::vector<float> silence_single(s.T * s.ctx_ch);
            for (int t = 0; t < s.T; t++) {
                const float * src = ctx->silence_full.data() + t * s.Oc;
                for (int c = 0; c < s.Oc; c++) {
                    silence_single[t * s.ctx_ch + c] = src[c];
                }
                for (int c = 0; c < s.Oc; c++) {
                    silence_single[t * s.ctx_ch + s.Oc + c] = 1.0f;
                }
            }
            s.context_silence.resize(batch_n * s.T * s.ctx_ch);
            for (int b = 0; b < batch_n; b++) {
                memcpy(s.context_silence.data() + b * s.T * s.ctx_ch, silence_single.data(),
                       s.T * s.ctx_ch * sizeof(float));
            }
            s.cover_steps = (int) ((float) s.num_steps * cover_strength);
            fprintf(stderr, "[Cover] audio_cover_strength=%.2f -> switch at step %d/%d\n", cover_strength,
                    s.cover_steps, s.num_steps);
        }
    }
}

// ops_init_noise_and_repaint
void ops_init_noise_and_repaint(AceSynth * ctx, const AceRequest * reqs, int batch_n, SynthState & s) {
    // Generate N s.noise samples (Philox4x32-10, matches torch.randn on CUDA with bf16).
    // Each batch item uses its own seed from the request.
    s.noise.resize(batch_n * s.Oc * s.T);
    for (int b = 0; b < batch_n; b++) {
        float * dst = s.noise.data() + b * s.Oc * s.T;
        philox_randn(reqs[b].seed, dst, s.Oc * s.T, /*bf16_round=*/true);
        fprintf(stderr, "[Context Batch%d] Philox noise seed=%lld, [%d, %d]\n", b, (long long) reqs[b].seed, s.T, s.Oc);
    }

    // cover_noise_strength: blend initial s.noise with source latents.
    // xt = nearest_t * s.noise + (1 - nearest_t) * s.cover_latents, then truncate s.schedule.
    if (s.use_source_context && s.have_cover && s.rr.cover_noise_strength > 0.0f) {
        float effective_noise_level = 1.0f - s.rr.cover_noise_strength;
        // find nearest timestep in s.schedule
        int   start_idx             = 0;
        float best_dist             = fabsf(s.schedule[0] - effective_noise_level);
        for (int i = 1; i < s.num_steps; i++) {
            float dist = fabsf(s.schedule[i] - effective_noise_level);
            if (dist < best_dist) {
                best_dist = dist;
                start_idx = i;
            }
        }
        float nearest_t = s.schedule[start_idx];
        // blend: xt = nearest_t * s.noise + (1 - nearest_t) * s.cover_latents
        for (int b = 0; b < batch_n; b++) {
            float * n = s.noise.data() + b * s.Oc * s.T;
            for (int t = 0; t < s.T; t++) {
                int           t_src = t < s.T_cover ? t : s.T_cover - 1;
                const float * src   = s.cover_latents.data() + t_src * s.Oc;
                for (int c = 0; c < s.Oc; c++) {
                    int idx = t * s.Oc + c;
                    n[idx]  = nearest_t * n[idx] + (1.0f - nearest_t) * src[c];
                }
            }
        }
        // truncate s.schedule
        s.schedule.erase(s.schedule.begin(), s.schedule.begin() + start_idx);
        s.num_steps = (int) s.schedule.size();
        // recalculate s.cover_steps with remaining steps
        if (s.cover_steps >= 0) {
            s.cover_steps = (int) ((float) s.num_steps * s.rr.audio_cover_strength);
        }
        fprintf(stderr, "[Cover] cover_noise_strength=%.2f -> noise_level=%.4f, nearest_t=%.4f, remaining_steps=%d\n",
                s.rr.cover_noise_strength, effective_noise_level, nearest_t, s.num_steps);
    }

    // DiT Generate
    s.output.resize(batch_n * s.Oc * s.T);

    // Per-batch sequence lengths for attention padding masks.
    // Within a synth_batch_size group, all elements share the same s.T (same codes),
    // so s.per_S[b] = s.S for all b. The s.per_enc_S[] array has real encoder lengths
    // from per-batch text encoding above.
    // These become meaningful when the server/CLI batches requests with different s.T.
    s.per_S.assign(batch_n, s.S);

    // Debug dumps (sample 0)
    debug_dump_2d(&s.dbg, "noise", s.noise.data(), s.T, s.Oc);
    debug_dump_2d(&s.dbg, "context", s.context.data(), s.T, s.ctx_ch);

    fprintf(stderr, "[DiT] Starting: T=%d, S=%d, enc_S=%d, steps=%d, batch=%d%s\n", s.T, s.S, s.enc_S, s.num_steps,
            batch_n, s.use_source_context ? " (cover)" : "");

    // repaint/lego-region injection buffer: full cover latents padded with silence.
    // used for step injection and boundary blend in both repaint and lego-region modes.
    if (s.is_repaint || s.is_lego_region) {
        s.repaint_src.resize(s.T * s.Oc);
        for (int t = 0; t < s.T; t++) {
            const float * src =
                (t < s.T_cover) ? s.cover_latents.data() + t * s.Oc : ctx->silence_full.data() + t * s.Oc;
            memcpy(s.repaint_src.data() + t * s.Oc, src, s.Oc * sizeof(float));
        }
    }
}

// ops_dit_generate
int ops_dit_generate(AceSynth * ctx, int batch_n, SynthState & s, bool (*cancel)(void *), void * cancel_data) {
    s.timer.reset();
    int dit_rc = dit_ggml_generate(
        &ctx->dit, s.noise.data(), s.context.data(), s.enc_hidden.data(), s.enc_S, s.T, batch_n, s.num_steps,
        s.schedule.data(), s.output.data(), s.guidance_scale, &s.dbg,
        s.context_silence.empty() ? nullptr : s.context_silence.data(), s.cover_steps, cancel, cancel_data,
        s.per_S.data(), s.per_enc_S.data(), s.enc_hidden_nc.empty() ? nullptr : s.enc_hidden_nc.data(),
        s.per_enc_S_nc_final.empty() ? nullptr : s.per_enc_S_nc_final.data(),
        s.repaint_src.empty() ? nullptr : s.repaint_src.data(), s.repaint_t0, s.repaint_t1,
        s.repaint_injection_ratio, s.repaint_crossfade_frames);
    if (dit_rc != 0) {
        return -1;
    }
    fprintf(stderr, "[DiT] Total generation: %.1f ms (%.1f ms/sample)\n", s.timer.ms(), s.timer.ms() / batch_n);

    debug_dump_2d(&s.dbg, "dit_output", s.output.data(), s.T, s.Oc);
    return 0;
}

// ops_vae_decode_and_splice
int ops_vae_decode_and_splice(AceSynth *    ctx,
                              int           batch_n,
                              AceAudio *    out,
                              SynthState &  s,
                              const float * src_audio,
                              int           src_len,
                              bool (*cancel)(void *),
                              void * cancel_data) {
    // VAE Decode
    if (!ctx->have_vae) {
        for (int b = 0; b < batch_n; b++) {
            out[b].samples     = NULL;
            out[b].n_samples   = 0;
            out[b].sample_rate = 48000;
        }
        return 0;
    }

    {
        int                T_latent    = s.T;
        int                T_audio_max = T_latent * 1920;
        std::vector<float> audio(2 * T_audio_max);

        for (int b = 0; b < batch_n; b++) {
            float * dit_out = s.output.data() + b * s.Oc * s.T;

            s.timer.reset();
            int T_audio = vae_ggml_decode_tiled(&ctx->vae, dit_out, T_latent, audio.data(), T_audio_max,
                                                ctx->params.vae_chunk, ctx->params.vae_overlap, cancel, cancel_data);
            if (T_audio < 0) {
                // check if this was a cancellation or a real error
                if (cancel && cancel(cancel_data)) {
                    fprintf(stderr, "[VAE Batch%d] Cancelled\n", b);
                    return -1;
                }
                fprintf(stderr, "[VAE Batch%d] ERROR: decode failed\n", b);
                out[b].samples     = NULL;
                out[b].n_samples   = 0;
                out[b].sample_rate = 48000;
                continue;
            }
            fprintf(stderr, "[VAE Batch%d] Decode: %.1f ms\n", b, s.timer.ms());

            if (b == 0) {
                debug_dump_2d(&s.dbg, "vae_audio", audio.data(), 2, T_audio);
            }

            // Copy to s.output buffer
            int n_total    = 2 * T_audio;
            out[b].samples = (float *) malloc((size_t) n_total * sizeof(float));
            memcpy(out[b].samples, audio.data(), (size_t) n_total * sizeof(float));
            out[b].n_samples   = T_audio;
            out[b].sample_rate = 48000;

            // Waveform splice: replace non-repaint regions with original source audio.
            // Python: apply_repaint_waveform_splice (when mode != aggressive)
            // mask[s] = 1.0 inside repaint region, 0.0 outside, linear ramp at edges.
            // result = mask * pred + (1-mask) * src  [planar stereo: L:s.T, R:s.T]
            bool have_repaint_region = s.is_repaint || s.is_lego_region;
            if (have_repaint_region && src_audio) {  // always splice (non-aggressive)
                int T_splice = out[b].n_samples < src_len ? out[b].n_samples : src_len;
                int start_s  = (int) (s.rs * 48000.0f);
                int end_s    = (int) (s.re * 48000.0f);
                start_s      = start_s < 0 ? 0 : (start_s > T_splice ? T_splice : start_s);
                end_s        = end_s < start_s ? start_s : (end_s > T_splice ? T_splice : end_s);
                // skip splice if region covers everything
                if (start_s > 0 || end_s < T_splice) {
                    int cf_s       = (int) (s.repaint_wav_cf_sec * 48000.0f);
                    int fade_start = start_s - cf_s > 0 ? start_s - cf_s : 0;
                    int fade_end   = end_s + cf_s < T_splice ? end_s + cf_s : T_splice;
                    for (int ch = 0; ch < 2; ch++) {
                        float * pred = out[b].samples + (size_t) ch * out[b].n_samples;
                        // src_audio is interleaved [L0,R0,L1,R1,...]: access via s*2+ch
                        for (int si = 0; si < fade_start; si++) {
                            pred[si] = src_audio[(size_t) si * 2 + ch];
                        }
                        for (int si = fade_start; si < start_s; si++) {
                            // left ramp: 0->1 toward repaint zone (excl endpoints)
                            int   rl  = start_s - fade_start;
                            float m   = (float) (si - fade_start + 1) / (float) (rl + 1);
                            float src = src_audio[(size_t) si * 2 + ch];
                            pred[si]  = m * pred[si] + (1.0f - m) * src;
                        }
                        // [start_s, end_s): keep generated s.output as-is (mask=1)
                        for (int si = end_s; si < fade_end; si++) {
                            // right ramp: 1->0 away from repaint zone (excl endpoints)
                            int   rl  = fade_end - end_s;
                            float m   = (float) (fade_end - si) / (float) (rl + 1);
                            float src = src_audio[(size_t) si * 2 + ch];
                            pred[si]  = m * pred[si] + (1.0f - m) * src;
                        }
                        for (int si = fade_end; si < T_splice; si++) {
                            pred[si] = src_audio[(size_t) si * 2 + ch];
                        }
                    }
                    fprintf(stderr, "[Splice Batch%d] wav splice %.1fs-%.1fs cf=%.0fms\n", b, s.rs, s.re,
                            s.repaint_wav_cf_sec * 1000.0f);
                }
            }
        }
    }
    return 0;
}
