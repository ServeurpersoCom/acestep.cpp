// pipeline-synth-ops.cpp: primitive operations of the synthesis pipeline
//
// Each op takes AceSynth (the pipeline context) and SynthState (the transient
// job state). See pipeline-synth-ops.h for the per-op contract and
// pipeline-synth-impl.h for the struct layouts.

#include "pipeline-synth-ops.h"

#include "dit-sampler.h"
#include "philox.h"
#include "pipeline-synth-impl.h"
#include "task-types.h"
#include "vae-enc.h"

#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <system_error>
#include <vector>

static const int FRAMES_PER_SECOND = 25;

// CSV list parser tolerant to any whitespace around commas. Locale-immune via
// std::from_chars (C++17 charconv, overloaded on the numeric type). Used for
// audio_codes (int) and custom_timesteps (float). Bails on first parse error
// or overflow, returning the values consumed so far.
template <typename T> static std::vector<T> parse_csv(const std::string & s) {
    std::vector<T> out;
    const char *   first = s.data();
    const char *   last  = first + s.size();
    while (first < last) {
        while (first < last && (*first == ',' || *first == ' ')) {
            first++;
        }
        if (first == last) {
            break;
        }
        T    v{};
        auto r = std::from_chars(first, last, v);
        if (r.ec != std::errc{}) {
            break;
        }
        out.push_back(v);
        first = r.ptr;
    }
    return out;
}

int ops_encode_src(const AceSynth * ctx,
                   const float *    src_audio,
                   int              src_len,
                   const float *    src_latents,
                   int              src_T_latent,
                   SynthState &     s) {
    // Cover mode: ingest source either as pre-encoded latents (zero VAE work)
    // or by acquiring the VAE encoder and running it on src_audio. When both
    // are provided latents win: they were either produced by a previous run
    // or supplied verbatim by the client and need no further processing.
    s.have_cover = false;
    s.T_cover    = 0;
    if (src_latents && src_T_latent > 0) {
        s.cover_latents.assign(src_latents, src_latents + (size_t) src_T_latent * 64);
        s.T_cover    = src_T_latent;
        s.have_cover = true;
        fprintf(stderr, "[Encode-Src] Latents in: T_cover=%d (%.2fs), VAE encode skipped\n", s.T_cover,
                (float) s.T_cover * 1920.0f / 48000.0f);
        return 0;
    }
    if (src_audio && src_len > 0) {
        s.timer.reset();
        int T_audio = src_len;

        VAEEncoder * vae_enc = store_require_vae_enc(ctx->store, ctx->vae_enc_key);
        if (!vae_enc) {
            fprintf(stderr, "[Encode-Src] FATAL: store_require_vae_enc failed\n");
            return -1;
        }
        ModelHandle vae_enc_guard(ctx->store, vae_enc);

        int max_T_lat = (T_audio / 1920) + 64;
        s.cover_latents.resize(max_T_lat * 64);

        s.T_cover = vae_enc_encode_tiled(vae_enc, src_audio, T_audio, s.cover_latents.data(), max_T_lat,
                                         ctx->params.vae_chunk, ctx->params.vae_overlap);
        if (s.T_cover < 0) {
            fprintf(stderr, "[Encode-Src] FATAL: encode failed\n");
            return -1;
        }
        s.cover_latents.resize(s.T_cover * 64);
        fprintf(stderr, "[Encode-Src] Encoded: T_cover=%d (%.2fs), %.1f ms\n", s.T_cover,
                (float) s.T_cover * 1920.0f / 48000.0f, s.timer.ms());

        s.have_cover = true;
    }

    return 0;
}

void ops_fsq_roundtrip(const AceSynth * ctx, SynthState & s) {
    // FSQ roundtrip for cover: tokenize (25Hz->5Hz) + detokenize (5Hz->25Hz).
    // The lossy 5:1 temporal compression destroys micro-timings, ornaments and
    // transients. The DiT receives degraded latents and diverges from the source,
    // producing a free reinterpretation rather than a close remix.
    // cover-nofsq skips this call and feeds clean 25Hz VAE latents directly,
    // producing remixes that stay close to the source.
    // Other tasks (lego, extract, repaint, complete) also use clean latents.
    if (!s.have_cover) {
        return;
    }
    s.timer.reset();
    int              T_5Hz = (s.T_cover + 4) / 5;
    std::vector<int> codes(T_5Hz);

    // Tokenizer scope: acquire, encode 25Hz latents to 5Hz codes, release.
    int T_5Hz_actual;
    {
        TokGGML * tok = store_require_fsq_tok(ctx->store, ctx->fsq_tok_key);
        if (!tok) {
            fprintf(stderr, "[FSQ-Roundtrip] FATAL: store_require_fsq_tok failed\n");
            return;
        }
        ModelHandle tok_guard(ctx->store, tok);
        if (!ctx->params.use_fa) {
            tok->use_flash_attn = false;
        }

        T_5Hz_actual =
            tok_ggml_encode(tok, s.cover_latents.data(), s.T_cover, codes.data(), ctx->meta->silence_full.data());
    }
    if (T_5Hz_actual <= 0) {
        return;
    }

    // Detokenizer scope: acquire, decode 5Hz codes back to 25Hz latents, release.
    int                T_25Hz_rt = T_5Hz_actual * 5;
    std::vector<float> rt_latents(T_25Hz_rt * 64);
    int                ret;
    {
        DetokGGML * detok = store_require_fsq_detok(ctx->store, ctx->fsq_detok_key);
        if (!detok) {
            fprintf(stderr, "[FSQ-Roundtrip] FATAL: store_require_fsq_detok failed\n");
            return;
        }
        ModelHandle detok_guard(ctx->store, detok);
        if (!ctx->params.use_fa) {
            detok->use_flash_attn = false;
        }

        ret = detok_ggml_decode(detok, codes.data(), T_5Hz_actual, rt_latents.data());
    }
    if (ret < 0) {
        return;
    }
    int copy_T = T_25Hz_rt < s.T_cover ? T_25Hz_rt : s.T_cover;
    memcpy(s.cover_latents.data(), rt_latents.data(), (size_t) copy_T * 64 * sizeof(float));
    fprintf(stderr, "[FSQ-Roundtrip] %d->%d->%d frames, %.1f ms\n", s.T_cover, T_5Hz_actual, copy_T, s.timer.ms());
}

int ops_resolve_params(const AceSynth * ctx, const AceRequest * reqs, int batch_n, SynthState & s) {
    // Extract shared params from first request
    s.duration = s.rr.duration > 0 ? s.rr.duration : 30.0f;

    // Resolve DiT sampling params: 0 = auto-detect from model type.
    // Turbo: 8 steps, guidance=1.0, s.shift=3.0
    // Base/SFT: 50 steps, guidance=1.0, s.shift=1.0
    s.num_steps      = s.rr.inference_steps;
    s.guidance_scale = s.rr.guidance_scale;
    s.shift          = s.rr.shift;

    if (s.num_steps <= 0) {
        s.num_steps = ctx->meta->is_turbo ? 8 : 50;
    }
    if (s.num_steps > 100) {
        fprintf(stderr, "[Resolve-Params] WARNING: inference_steps %d clamped to 100\n", s.num_steps);
        s.num_steps = 100;
    }

    if (s.guidance_scale <= 0.0f) {
        s.guidance_scale = 1.0f;
    } else if (ctx->meta->is_turbo && s.guidance_scale > 1.0f) {
        fprintf(stderr, "[Resolve-Params] WARNING: turbo model, forcing guidance_scale=1.0 (was %.1f)\n",
                s.guidance_scale);
        s.guidance_scale = 1.0f;
    }

    if (s.shift <= 0.0f) {
        s.shift = ctx->meta->is_turbo ? 3.0f : 1.0f;
    }

    // Audio codes: parse once per request and stash in s.per_codes so
    // ops_build_context does not have to re-parse. Also records the longest
    // set (drives s.T) and whether any batch item carries codes at all.
    // Shorter code sets are padded with silence, longer ones are never truncated.
    s.per_codes.assign(batch_n, {});
    s.max_codes_len = 0;
    s.have_codes    = false;
    for (int b = 0; b < batch_n; b++) {
        s.per_codes[b] = parse_csv<int>(reqs[b].audio_codes);
        int sz         = (int) s.per_codes[b].size();
        if (sz > s.max_codes_len) {
            s.max_codes_len = sz;
        }
        if (sz > 0) {
            s.have_codes = true;
        }
    }
    if (s.have_codes) {
        fprintf(stderr, "[Resolve-Params] max audio codes across batch: %d (%.1fs @ 5Hz)\n", s.max_codes_len,
                (float) s.max_codes_len / 5.0f);
    }

    return 0;
}

void ops_build_schedule(SynthState & s) {
    // Custom timesteps override: CSV floats like
    // "0.97,0.76,0.615,0.5,0.395,0.28,0.18,0.085,0". Last value is the x0
    // endpoint handled implicitly by the sampler, so we drop it and take
    // schedule = first N-1 entries, num_steps = N-1.
    if (!s.rr.custom_timesteps.empty()) {
        std::vector<float> ts = parse_csv<float>(s.rr.custom_timesteps);
        if (ts.size() >= 2) {
            s.num_steps = (int) ts.size() - 1;
            s.schedule.assign(ts.begin(), ts.end() - 1);
            fprintf(stderr, "[Build-Schedule] Custom timesteps: %d steps\n", s.num_steps);
            return;
        }
        fprintf(stderr, "[Build-Schedule] WARN: custom_timesteps needs >= 2 values, falling back to shift\n");
    }
    // Default: t_i = shift * t / (1 + (shift-1)*t) with t = 1 - i/steps
    s.schedule.resize(s.num_steps);
    for (int i = 0; i < s.num_steps; i++) {
        float t       = 1.0f - (float) i / (float) s.num_steps;
        s.schedule[i] = s.shift * t / (1.0f + (s.shift - 1.0f) * t);
    }
}

int ops_resolve_T(const AceSynth * ctx, SynthState & s) {
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
        fprintf(stderr, "[Resolve-T] FATAL: use_source_context but no cover_latents and no audio_codes\n");
        return -1;
    } else {
        s.T = (int) (s.duration * FRAMES_PER_SECOND);
    }
    s.T     = ((s.T + ctx->meta->cfg.patch_size - 1) / ctx->meta->cfg.patch_size) * ctx->meta->cfg.patch_size;
    s.S     = s.T / ctx->meta->cfg.patch_size;
    s.enc_S = 0;

    fprintf(stderr, "[Resolve-T] T=%d, S=%d\n", s.T, s.S);
    fprintf(stderr, "[Resolve-T] seed=%lld, steps=%d, guidance=%.1f, shift=%.1f, duration=%.1fs\n",
            (long long) s.rr.seed, s.num_steps, s.guidance_scale, s.shift, s.duration);

    if (s.T > 15000) {
        fprintf(stderr, "[Resolve-T] ERROR: T=%d exceeds silence_latent max 15000, skipping\n", s.T);
        return -1;
    }

    return 0;
}

void ops_encode_timbre(const AceSynth * ctx,
                       const float *    ref_audio,
                       int              ref_len,
                       const float *    ref_latents,
                       int              ref_T_latent,
                       SynthState &     s) {
    // Timbre features from ref_audio or ref_latents (independent of src).
    // Two paths converge into s.timbre_feats: pre-encoded latents skip the
    // VAE encoder entirely, raw audio takes the encoder path. Latents win
    // when both are set. Without either input the timbre falls back to a
    // single silence frame, disabling timbre conditioning.
    if (ref_latents && ref_T_latent > 0) {
        s.S_ref_timbre = ref_T_latent;
        s.timbre_feats.assign(ref_latents, ref_latents + (size_t) ref_T_latent * 64);
        fprintf(stderr, "[Encode-Timbre] Latents in: %d frames (%.1fs), VAE encode skipped\n", ref_T_latent,
                (float) ref_T_latent / 25.0f);
        return;
    }
    if (ref_audio && ref_len > 0) {
        s.timer.reset();
        VAEEncoder * ref_vae = store_require_vae_enc(ctx->store, ctx->vae_enc_key);
        if (!ref_vae) {
            fprintf(stderr, "[Encode-Timbre] WARNING: store_require_vae_enc failed, using silence\n");
            s.S_ref_timbre = 1;
            s.timbre_feats.assign(ctx->meta->silence_full.data(), ctx->meta->silence_full.data() + 64);
            return;
        }
        ModelHandle ref_vae_guard(ctx->store, ref_vae);

        int                max_T_ref = (ref_len / 1920) + 64;
        std::vector<float> ref_lat_buf(max_T_ref * 64);
        int                T_ref = vae_enc_encode_tiled(ref_vae, ref_audio, ref_len, ref_lat_buf.data(), max_T_ref,
                                                        ctx->params.vae_chunk, ctx->params.vae_overlap);
        if (T_ref < 0) {
            fprintf(stderr, "[Encode-Timbre] WARNING: ref_audio encode failed, using silence\n");
            s.S_ref_timbre = 1;
            s.timbre_feats.assign(ctx->meta->silence_full.data(), ctx->meta->silence_full.data() + 64);
        } else {
            s.S_ref_timbre = T_ref;
            s.timbre_feats.assign(ref_lat_buf.data(), ref_lat_buf.data() + (size_t) T_ref * 64);
            fprintf(stderr, "[Encode-Timbre] ref_audio: %d frames (%.1fs), %.1f ms\n", T_ref, (float) T_ref / 25.0f,
                    s.timer.ms());
        }
    } else {
        s.S_ref_timbre = 1;
        s.timbre_feats.assign(ctx->meta->silence_full.data(), ctx->meta->silence_full.data() + 64);
    }
}

// Per-batch CPU-resident forward from the text encoder.
// Lives in a local array between the text_enc and cond_enc phases so the
// two GPU modules never need to coexist under EVICT_STRICT.
struct TextEncForward {
    std::vector<float> text_hidden;  // [S_text * H_text] f32
    std::vector<float> lyric_embed;  // [S_lyric * H_text] f32
    int                S_text;
    int                S_lyric;
};

// Build the text/lyric prompt pair that feeds the text encoder for one batch
// element. instruction is the DiT instruction header (main or non-cover).
static void build_prompt_strings(const AceRequest &  rb,
                                 const std::string & instruction,
                                 float               duration,
                                 std::string &       text_out,
                                 std::string &       lyric_out) {
    char bpm_b[16] = "N/A";
    if (rb.bpm > 0) {
        snprintf(bpm_b, sizeof(bpm_b), "%d", rb.bpm);
    }
    const char * keyscale_b = rb.keyscale.empty() ? "N/A" : rb.keyscale.c_str();
    const char * timesig_b  = rb.timesignature.empty() ? "N/A" : rb.timesignature.c_str();
    const char * language_b = rb.vocal_language.empty() ? "unknown" : rb.vocal_language.c_str();

    char metas_b[512];
    snprintf(metas_b, sizeof(metas_b), "- bpm: %s\n- timesignature: %s\n- keyscale: %s\n- duration: %d seconds\n",
             bpm_b, timesig_b, keyscale_b, (int) duration);
    text_out = std::string("# Instruction\n") + instruction + "\n\n" + "# Caption\n" + rb.caption + "\n\n" +
               "# Metas\n" + metas_b + "<|endoftext|>\n";
    lyric_out = std::string("# Languages\n") + language_b + "\n\n# Lyric\n" + rb.lyrics + "<|endoftext|>";
}

int ops_encode_text(const AceSynth * ctx, const AceRequest * reqs, int batch_n, SynthState & s) {
    // Per-batch text encoding in two GPU phases to keep EVICT_STRICT at one
    // module resident at a time:
    //   Phase A: acquire text_enc, run all qwen3 forwards (main and optional
    //            non-cover) into CPU-resident TextEncForward, release.
    //   Phase B: acquire cond_enc, run all cond_ggml forwards consuming those
    //            cached hidden states into s.per_enc / s.per_enc_nc, release.
    // Intermediate CPU footprint peaks at roughly batch_n * 2 * S * H_text * 4
    // bytes (a few MB), negligible next to the GPU modules.
    if (s.instruction_str.empty()) {
        fprintf(stderr, "[Encode-Text] FATAL: instruction_str is empty (unknown task or orchestrator bug)\n");
        return -1;
    }

    s.need_enc_switch = s.use_source_context && !s.is_repaint && !s.is_lego_region && s.rr.audio_cover_strength < 1.0f;

    BPETokenizer * bpe = store_bpe(ctx->store, ctx->params.text_encoder_path);
    if (!bpe) {
        fprintf(stderr, "[Encode-Text] FATAL: store_bpe failed\n");
        return -1;
    }

    std::vector<TextEncForward> main_fwd(batch_n);
    std::vector<TextEncForward> nc_fwd(s.need_enc_switch ? batch_n : 0);
    int                         H_text = 0;

    // Phase A: text encoder.
    {
        Qwen3GGML * te = store_require_text_enc(ctx->store, ctx->text_enc_key);
        if (!te) {
            fprintf(stderr, "[Encode-Text] FATAL: store_require_text_enc failed\n");
            return -1;
        }
        ModelHandle te_guard(ctx->store, te);
        if (!ctx->params.use_fa) {
            te->use_flash_attn = false;
        }

        H_text = te->cfg.hidden_size;

        for (int b = 0; b < batch_n; b++) {
            std::string text_str;
            std::string lyric_str;
            build_prompt_strings(reqs[b], s.instruction_str, s.duration, text_str, lyric_str);

            auto text_ids  = bpe_encode(bpe, text_str.c_str(), true);
            auto lyric_ids = bpe_encode(bpe, lyric_str.c_str(), true);
            int  S_text    = (int) text_ids.size();
            int  S_lyric   = (int) lyric_ids.size();

            main_fwd[b].S_text  = S_text;
            main_fwd[b].S_lyric = S_lyric;
            main_fwd[b].text_hidden.resize((size_t) H_text * S_text);
            qwen3_forward(te, text_ids.data(), S_text, main_fwd[b].text_hidden.data());
            main_fwd[b].lyric_embed.resize((size_t) H_text * S_lyric);
            qwen3_embed_lookup(te, lyric_ids.data(), S_lyric, main_fwd[b].lyric_embed.data());
        }

        if (s.need_enc_switch) {
            for (int b = 0; b < batch_n; b++) {
                std::string text_str;
                std::string lyric_str;
                build_prompt_strings(reqs[b], DIT_INSTR_TEXT2MUSIC, s.duration, text_str, lyric_str);

                auto text_ids  = bpe_encode(bpe, text_str.c_str(), true);
                auto lyric_ids = bpe_encode(bpe, lyric_str.c_str(), true);
                int  S_text    = (int) text_ids.size();
                int  S_lyric   = (int) lyric_ids.size();

                nc_fwd[b].S_text  = S_text;
                nc_fwd[b].S_lyric = S_lyric;
                nc_fwd[b].text_hidden.resize((size_t) H_text * S_text);
                qwen3_forward(te, text_ids.data(), S_text, nc_fwd[b].text_hidden.data());
                nc_fwd[b].lyric_embed.resize((size_t) H_text * S_lyric);
                qwen3_embed_lookup(te, lyric_ids.data(), S_lyric, nc_fwd[b].lyric_embed.data());
            }
        }

        // Debug dump of sample 0 while text_hidden and lyric_embed are live.
        debug_dump_2d(&s.dbg, "text_hidden", main_fwd[0].text_hidden.data(), main_fwd[0].S_text, H_text);
        debug_dump_2d(&s.dbg, "lyric_embed", main_fwd[0].lyric_embed.data(), main_fwd[0].S_lyric, H_text);
    }

    // Phase B: condition encoder.
    int H_cond = 0;
    s.per_enc.resize(batch_n);
    s.per_enc_S.resize(batch_n);
    s.per_enc_nc.resize(batch_n);
    s.per_enc_S_nc.assign(batch_n, 0);
    {
        CondGGML * ce = store_require_cond_enc(ctx->store, ctx->cond_enc_key);
        if (!ce) {
            fprintf(stderr, "[Encode-Text] FATAL: store_require_cond_enc failed\n");
            return -1;
        }
        ModelHandle ce_guard(ctx->store, ce);
        if (!ctx->params.use_fa) {
            ce->use_flash_attn = false;
        }
        ce->clamp_fp16 = ctx->params.clamp_fp16;

        H_cond = ce->lyric_cfg.hidden_size;

        // null_condition_emb lives on the DiTMeta. Empty when the model has none.
        s.null_cond_vec.resize(H_cond);
        if (!ctx->meta->null_cond_cpu.empty()) {
            memcpy(s.null_cond_vec.data(), ctx->meta->null_cond_cpu.data(), H_cond * sizeof(float));
        }

        for (int b = 0; b < batch_n; b++) {
            s.timer.reset();
            cond_ggml_forward(ce, main_fwd[b].text_hidden.data(), main_fwd[b].S_text, main_fwd[b].lyric_embed.data(),
                              main_fwd[b].S_lyric, s.timbre_feats.data(), s.S_ref_timbre, s.per_enc[b],
                              &s.per_enc_S[b]);
            fprintf(stderr, "[Encode-Text Batch%d] %d+%d tokens -> enc_S=%d, %.1f ms\n", b, main_fwd[b].S_text,
                    main_fwd[b].S_lyric, s.per_enc_S[b], s.timer.ms());
        }
        debug_dump_2d(&s.dbg, "enc_hidden", s.per_enc[0].data(), s.per_enc_S[0], H_cond);

        if (s.need_enc_switch) {
            for (int b = 0; b < batch_n; b++) {
                cond_ggml_forward(ce, nc_fwd[b].text_hidden.data(), nc_fwd[b].S_text, nc_fwd[b].lyric_embed.data(),
                                  nc_fwd[b].S_lyric, s.timbre_feats.data(), s.S_ref_timbre, s.per_enc_nc[b],
                                  &s.per_enc_S_nc[b]);
                fprintf(stderr, "[Encode-Text Batch%d] non-cover: %d+%d tokens -> enc_S=%d\n", b, nc_fwd[b].S_text,
                        nc_fwd[b].S_lyric, s.per_enc_S_nc[b]);
            }
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

    s.enc_hidden.resize((size_t) H_cond * s.max_enc_S * batch_n);
    for (int b = 0; b < batch_n; b++) {
        float * dst = s.enc_hidden.data() + (size_t) b * s.max_enc_S * H_cond;
        memcpy(dst, s.per_enc[b].data(), (size_t) s.per_enc_S[b] * H_cond * sizeof(float));
        for (int si = s.per_enc_S[b]; si < s.max_enc_S; si++) {
            memcpy(dst + si * H_cond, s.null_cond_vec.data(), H_cond * sizeof(float));
        }
    }

    // pad and stack text2music encoding (same s.max_enc_S for graph compatibility)
    if (s.need_enc_switch) {
        s.enc_hidden_nc.resize((size_t) H_cond * s.max_enc_S * batch_n);
        s.per_enc_S_nc_final.resize(batch_n);
        for (int b = 0; b < batch_n; b++) {
            float * dst = s.enc_hidden_nc.data() + (size_t) b * s.max_enc_S * H_cond;
            memcpy(dst, s.per_enc_nc[b].data(), (size_t) s.per_enc_S_nc[b] * H_cond * sizeof(float));
            for (int si = s.per_enc_S_nc[b]; si < s.max_enc_S; si++) {
                memcpy(dst + si * H_cond, s.null_cond_vec.data(), H_cond * sizeof(float));
            }
            s.per_enc_S_nc_final[b] = s.per_enc_S_nc[b];
        }
    }

    if (batch_n > 1) {
        fprintf(stderr, "[Encode-Text] Per-batch encoding done: max_enc_S=%d\n", s.max_enc_S);
    }

    return 0;
}

int ops_build_context(const AceSynth * ctx, const AceRequest * reqs, int batch_n, SynthState & s) {
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
        fprintf(stderr, "[Build-Context] Latent frames: [%d, %d) / %d\n", s.repaint_t0, s.repaint_t1, s.T);
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
                src = ctx->meta->silence_full.data() + t * s.Oc;
            } else {
                src = (t < s.T_cover) ? s.cover_latents.data() + t * s.Oc : ctx->meta->silence_full.data() + t * s.Oc;
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
            fprintf(stderr, "[Build-Context] FATAL: use_source_context but no cover_latents and no audio_codes\n");
            return -1;
        }

        // Decode batch items with audio codes through the FSQ detokenizer, cached
        // in CPU buffers before the fill loop so the detokenizer is held for the
        // shortest possible window under STRICT.
        std::vector<std::vector<float>> decoded_per_b(batch_n);
        std::vector<int>                decoded_T_per_b(batch_n, 0);

        // s.have_codes is already posed by ops_resolve_params over the same batch.
        bool any_codes = s.have_codes;

        if (any_codes) {
            DetokGGML * detok = store_require_fsq_detok(ctx->store, ctx->fsq_detok_key);
            if (!detok) {
                fprintf(stderr, "[Build-Context] FATAL: store_require_fsq_detok failed\n");
                return -1;
            }
            ModelHandle detok_guard(ctx->store, detok);
            if (!ctx->params.use_fa) {
                detok->use_flash_attn = false;
            }

            for (int b = 0; b < batch_n; b++) {
                const std::vector<int> & codes_b = s.per_codes[b];
                if (codes_b.empty()) {
                    continue;
                }
                s.timer.reset();
                int T_5Hz        = (int) codes_b.size();
                int T_25Hz_codes = T_5Hz * 5;
                decoded_per_b[b].resize((size_t) T_25Hz_codes * s.Oc);

                int ret = detok_ggml_decode(detok, codes_b.data(), T_5Hz, decoded_per_b[b].data());
                if (ret < 0) {
                    fprintf(stderr, "[Build-Context Batch%d] FATAL: detokenizer decode failed\n", b);
                    return -1;
                }
                fprintf(stderr, "[Build-Context Batch%d] Detokenizer: %.1f ms, %d codes\n", b, s.timer.ms(), T_5Hz);

                decoded_T_per_b[b] = T_25Hz_codes < s.T ? T_25Hz_codes : s.T;
                if (b == 0) {
                    debug_dump_2d(&s.dbg, "detok_output", decoded_per_b[b].data(), T_25Hz_codes, s.Oc);
                }
            }
        }

        // Fill s.context: decoded latents then silence, mask = 1.0 (training distribution).
        // The detokenizer is already released at this point; the CPU buffers in
        // decoded_per_b carry everything we need.
        for (int b = 0; b < batch_n; b++) {
            float *       ctx_dst   = s.context.data() + b * s.T * s.ctx_ch;
            const float * decoded   = decoded_per_b[b].data();
            int           decoded_T = decoded_T_per_b[b];

            for (int t = 0; t < s.T; t++) {
                const float * src =
                    (t < decoded_T) ? decoded + t * s.Oc : ctx->meta->silence_full.data() + (t - decoded_T) * s.Oc;
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

void ops_build_context_silence(const AceSynth * ctx, int batch_n, SynthState & s) {
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
                const float * src = ctx->meta->silence_full.data() + t * s.Oc;
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
            fprintf(stderr, "[Context-Silence] audio_cover_strength=%.2f -> switch at step %d/%d\n", cover_strength,
                    s.cover_steps, s.num_steps);
        }
    }
}

void ops_init_noise(const AceSynth * ctx, const AceRequest * reqs, int batch_n, SynthState & s) {
    // Generate N s.noise samples (Philox4x32-10, matches torch.randn on CUDA with bf16).
    // Each batch item uses its own seed from the request.
    s.noise.resize(batch_n * s.Oc * s.T);
    s.seeds.resize(batch_n);
    for (int b = 0; b < batch_n; b++) {
        float * dst = s.noise.data() + b * s.Oc * s.T;
        s.seeds[b]  = reqs[b].seed;
        philox_randn(reqs[b].seed, dst, s.Oc * s.T, /*bf16_round=*/true);
        fprintf(stderr, "[Init-Noise Batch%d] Philox noise seed=%lld, [%d, %d]%s\n", b, (long long) reqs[b].seed, s.T,
                s.Oc, s.use_sde ? " (SDE)" : "");
    }

    // cover_noise_strength: blend initial noise with clean source latents.
    // xt = nearest_t * noise + (1 - nearest_t) * clean_latents, then truncate schedule.
    // the FSQ roundtrip degrades cover_latents for context conditioning, but noise
    // blending needs the original clean VAE latents. noise_blend_latents holds the
    // clean copy when FSQ was applied; otherwise fall back to cover_latents (already clean).
    if (s.use_source_context && s.have_cover && s.rr.cover_noise_strength > 0.0f) {
        const std::vector<float> & blend_src = s.noise_blend_latents.empty() ? s.cover_latents : s.noise_blend_latents;
        float                      effective_noise_level = 1.0f - s.rr.cover_noise_strength;
        // find nearest timestep in s.schedule
        int                        start_idx             = 0;
        float                      best_dist             = fabsf(s.schedule[0] - effective_noise_level);
        for (int i = 1; i < s.num_steps; i++) {
            float dist = fabsf(s.schedule[i] - effective_noise_level);
            if (dist < best_dist) {
                best_dist = dist;
                start_idx = i;
            }
        }
        float nearest_t = s.schedule[start_idx];
        // blend: xt = nearest_t * s.noise + (1 - nearest_t) * clean_latents
        for (int b = 0; b < batch_n; b++) {
            float * n = s.noise.data() + b * s.Oc * s.T;
            for (int t = 0; t < s.T; t++) {
                int           t_src = t < s.T_cover ? t : s.T_cover - 1;
                const float * src   = blend_src.data() + t_src * s.Oc;
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
        fprintf(stderr,
                "[Init-Noise] cover_noise_strength=%.2f -> noise_level=%.4f, nearest_t=%.4f, remaining_steps=%d\n",
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

    fprintf(stderr, "[Init-Noise] Starting: T=%d, S=%d, enc_S=%d, steps=%d, batch=%d%s\n", s.T, s.S, s.enc_S,
            s.num_steps, batch_n, s.use_source_context ? " (cover)" : "");
}

int ops_dit_generate(const AceSynth * ctx, int batch_n, SynthState & s, bool (*cancel)(void *), void * cancel_data) {
    DiTGGML * dit = store_require_dit(ctx->store, ctx->dit_key);
    if (!dit) {
        fprintf(stderr, "[DiT-Generate] FATAL: store_require_dit failed\n");
        return -1;
    }
    ModelHandle dit_guard(ctx->store, dit);
    if (!ctx->params.use_fa) {
        dit->use_flash_attn = false;
    }

    s.timer.reset();
    int dit_rc = dit_ggml_generate(
        dit, s.noise.data(), s.context.data(), s.enc_hidden.data(), s.enc_S, s.T, batch_n, s.num_steps,
        s.schedule.data(), s.output.data(), s.guidance_scale, &s.dbg,
        s.context_silence.empty() ? nullptr : s.context_silence.data(), s.cover_steps, cancel, cancel_data,
        s.per_S.data(), s.per_enc_S.data(), s.enc_hidden_nc.empty() ? nullptr : s.enc_hidden_nc.data(),
        s.per_enc_S_nc_final.empty() ? nullptr : s.per_enc_S_nc_final.data(), s.use_sde, s.seeds.data(),
        ctx->params.use_batch_cfg, s.rr.dcw_scaler, s.rr.dcw_high_scaler, s.rr.dcw_mode.c_str());
    if (dit_rc != 0) {
        return -1;
    }
    fprintf(stderr, "[DiT-Generate] Total: %.1f ms (%.1f ms/sample)\n", s.timer.ms(), s.timer.ms() / batch_n);

    // Latent post-processing before VAE decode: pred = pred * rescale + shift.
    // Skipped at defaults (1.0 / 0.0).
    if (s.rr.latent_rescale != 1.0f || s.rr.latent_shift != 0.0f) {
        fprintf(stderr, "[DiT-Generate] Latent post: shift=%.3f rescale=%.3f\n", s.rr.latent_shift,
                s.rr.latent_rescale);
        const int n = (int) s.output.size();
        for (int i = 0; i < n; i++) {
            s.output[i] = s.output[i] * s.rr.latent_rescale + s.rr.latent_shift;
        }
    }

    debug_dump_2d(&s.dbg, "dit_output", s.output.data(), s.T, s.Oc);
    return 0;
}

int ops_vae_decode_and_splice(const AceSynth * ctx,
                              int              batch_n,
                              AceAudio *       out,
                              SynthState &     s,
                              const float *    src_audio,
                              int              src_len,
                              bool (*cancel)(void *),
                              void * cancel_data) {
    VAEGGML * vae = store_require_vae_dec(ctx->store, ctx->vae_dec_key);
    if (!vae) {
        fprintf(stderr, "[VAE-Decode] FATAL: store_require_vae_dec failed\n");
        return -1;
    }
    ModelHandle vae_guard(ctx->store, vae);

    int                T_latent    = s.T;
    int                T_audio_max = T_latent * 1920;
    std::vector<float> audio(2 * T_audio_max);

    for (int b = 0; b < batch_n; b++) {
        float * dit_out = s.output.data() + b * s.Oc * s.T;

        s.timer.reset();
        int T_audio = vae_ggml_decode_tiled(vae, dit_out, T_latent, audio.data(), T_audio_max, ctx->params.vae_chunk,
                                            ctx->params.vae_overlap, cancel, cancel_data);
        if (T_audio < 0) {
            // check if this was a cancellation or a real error
            if (cancel && cancel(cancel_data)) {
                fprintf(stderr, "[VAE-Decode Batch%d] Cancelled\n", b);
                return -1;
            }
            fprintf(stderr, "[VAE-Decode Batch%d] ERROR: decode failed\n", b);
            out[b].samples     = NULL;
            out[b].n_samples   = 0;
            out[b].sample_rate = 48000;
            continue;
        }
        fprintf(stderr, "[VAE-Decode Batch%d] Decode: %.1f ms\n", b, s.timer.ms());

        if (b == 0) {
            debug_dump_2d(&s.dbg, "vae_audio", audio.data(), 2, T_audio);
        }

        // Copy to s.output buffer
        int n_total    = 2 * T_audio;
        out[b].samples = (float *) malloc((size_t) n_total * sizeof(float));
        if (!out[b].samples) {
            fprintf(stderr, "[VAE-Decode Batch%d] ERROR: OOM allocating output (%d samples)\n", b, n_total);
            out[b].n_samples   = 0;
            out[b].sample_rate = 48000;
            continue;
        }
        memcpy(out[b].samples, audio.data(), (size_t) n_total * sizeof(float));
        out[b].n_samples   = T_audio;
        out[b].sample_rate = 48000;

        // Waveform splice: replace non-repaint regions with original source audio.
        // mask[s] = 1.0 inside repaint region, 0.0 outside, linear ramp at edges.
        // result = mask * pred + (1.0 - mask) * src  [planar stereo: L:s.T, R:s.T]
        // 10 ms crossfade kills the click at the splice joints, inaudible.
        bool have_repaint_region = s.is_repaint || s.is_lego_region;
        if (have_repaint_region && src_audio) {
            int T_splice = out[b].n_samples < src_len ? out[b].n_samples : src_len;
            int start_s  = (int) (s.rs * 48000.0f);
            int end_s    = (int) (s.re * 48000.0f);
            start_s      = start_s < 0 ? 0 : (start_s > T_splice ? T_splice : start_s);
            end_s        = end_s < start_s ? start_s : (end_s > T_splice ? T_splice : end_s);
            // skip splice if region covers everything
            if (start_s > 0 || end_s < T_splice) {
                int cf_s       = 480;  // 10 ms at 48 kHz
                int fade_start = start_s - cf_s > 0 ? start_s - cf_s : 0;
                int fade_end   = end_s + cf_s < T_splice ? end_s + cf_s : T_splice;
                for (int ch = 0; ch < 2; ch++) {
                    float * pred = out[b].samples + (size_t) ch * out[b].n_samples;
                    // src_audio is interleaved [L0,R0,L1,R1,...]: access via s*2+ch
                    for (int si = 0; si < fade_start; si++) {
                        pred[si] = src_audio[(size_t) si * 2 + ch];
                    }
                    for (int si = fade_start; si < start_s; si++) {
                        // left ramp: 0 to 1 toward repaint zone (excl endpoints)
                        int   rl  = start_s - fade_start;
                        float m   = (float) (si - fade_start + 1) / (float) (rl + 1);
                        float src = src_audio[(size_t) si * 2 + ch];
                        pred[si]  = m * pred[si] + (1.0f - m) * src;
                    }
                    // [start_s, end_s): keep generated s.output as is (mask=1)
                    for (int si = end_s; si < fade_end; si++) {
                        // right ramp: 1 to 0 away from repaint zone (excl endpoints)
                        int   rl  = fade_end - end_s;
                        float m   = (float) (fade_end - si) / (float) (rl + 1);
                        float src = src_audio[(size_t) si * 2 + ch];
                        pred[si]  = m * pred[si] + (1.0f - m) * src;
                    }
                    for (int si = fade_end; si < T_splice; si++) {
                        pred[si] = src_audio[(size_t) si * 2 + ch];
                    }
                }
                fprintf(stderr, "[WAV-Splice Batch%d] wav splice %.1fs..%.1fs cf=10ms\n", b, s.rs, s.re);
            }
        }
    }
    return 0;
}
