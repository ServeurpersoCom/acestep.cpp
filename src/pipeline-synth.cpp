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
#include "gguf-weights.h"
#include "philox.h"
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
    BPETokenizer  bpe;

    // Metadata from DiT GGUF
    bool               is_turbo;
    std::vector<float> silence_full;  // [15000, 64] f32

    // Config
    AceSynthParams params;
    bool           have_vae;
    bool           have_detok;

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

    // 1. Load BPE tokenizer
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

    // 4. Text encoder forward (caption only)
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

    // 6. Condition encoder forward
    timer.reset();
    ctx->cond_enc = {};
    cond_ggml_init_backend(&ctx->cond_enc);
    if (!params->use_fa) {
        ctx->cond_enc.use_flash_attn = false;
    }
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

    fprintf(stderr, "[Ace-Synth] All models loaded, turbo=%s\n", ctx->is_turbo ? "yes" : "no");

    return ctx;
}

int ace_synth_generate(AceSynth *         ctx,
                       const AceRequest * req,
                       const float *      src_audio,
                       int                src_len,
                       int                batch_n,
                       AceAudio *         out) {
    if (!ctx || !req || !out || batch_n < 1 || batch_n > 9) {
        return -1;
    }

    int Oc     = ctx->Oc;
    int ctx_ch = ctx->ctx_ch;

    Timer timer;

    DebugDumper dbg;
    debug_init(&dbg, ctx->params.dump_dir);

    // Cover mode: load VAE encoder and encode source audio
    bool               have_cover = false;
    std::vector<float> cover_latents;  // [T_cover, 64] time-major
    int                T_cover = 0;
    if (src_audio && src_len > 0) {
        if (!ctx->params.vae_path) {
            fprintf(stderr, "[Cover] ERROR: --src-audio requires --vae\n");
            return -1;
        }
        timer.reset();
        int T_audio = src_len;

        VAEEncoder vae_enc = {};
        vae_enc_load(&vae_enc, ctx->params.vae_path);
        int max_T_lat = (T_audio / 1920) + 64;
        cover_latents.resize(max_T_lat * 64);

        T_cover = vae_enc_encode_tiled(&vae_enc, src_audio, T_audio, cover_latents.data(), max_T_lat,
                                       ctx->params.vae_chunk, ctx->params.vae_overlap);
        vae_enc_free(&vae_enc);
        if (T_cover < 0) {
            fprintf(stderr, "[VAE-Enc] FATAL: encode failed\n");
            return -1;
        }
        cover_latents.resize(T_cover * 64);
        fprintf(stderr, "[Cover] Encoded: T_cover=%d (%.2fs), %.1f ms\n", T_cover, (float) T_cover * 1920.0f / 48000.0f,
                timer.ms());
        have_cover = true;
    }

    // Work on a mutable copy
    AceRequest rr = *req;

    if (rr.caption.empty() && rr.lego.empty()) {
        fprintf(stderr, "[Request] ERROR: caption is empty, skipping\n");
        return -1;
    }

    // Lego mode validation (base model only, requires --src-audio)
    bool is_lego = !rr.lego.empty();
    if (is_lego) {
        if (!have_cover) {
            fprintf(stderr, "[Lego] ERROR: lego requires --src-audio\n");
            return -1;
        }
        if (ctx->is_turbo) {
            fprintf(stderr, "[Lego] ERROR: lego requires the base DiT model (turbo detected)\n");
            return -1;
        }
        // Reference project: TRACK_NAMES (constants.py)
        static const char * allowed[] = {
            "vocals",     "backing_vocals", "drums", "bass", "guitar", "keyboard",
            "percussion", "strings",        "synth", "fx",   "brass",  "woodwinds",
        };
        bool valid = false;
        for (int k = 0; k < 12; k++) {
            if (rr.lego == allowed[k]) {
                valid = true;
                break;
            }
        }
        if (!valid) {
            fprintf(stderr, "[Lego] ERROR: '%s' is not a valid track name\n", rr.lego.c_str());
            fprintf(stderr,
                    "  Valid: vocals, backing_vocals, drums, bass, guitar, keyboard,\n"
                    "         percussion, strings, synth, fx, brass, woodwinds\n");
            return -1;
        }
    }

    // Extract params
    const char * caption     = rr.caption.c_str();
    const char * lyrics      = rr.lyrics.c_str();
    char         bpm_str[16] = "N/A";
    if (rr.bpm > 0) {
        snprintf(bpm_str, sizeof(bpm_str), "%d", rr.bpm);
    }
    const char * bpm            = bpm_str;
    const char * keyscale       = rr.keyscale.empty() ? "N/A" : rr.keyscale.c_str();
    const char * timesig        = rr.timesignature.empty() ? "N/A" : rr.timesignature.c_str();
    const char * language       = rr.vocal_language.empty() ? "unknown" : rr.vocal_language.c_str();
    float        duration       = rr.duration > 0 ? rr.duration : 30.0f;
    long long    seed           = rr.seed;
    int          num_steps      = rr.inference_steps > 0 ? rr.inference_steps : 8;
    float        guidance_scale = rr.guidance_scale;
    float        shift          = rr.shift > 0 ? rr.shift : 1.0f;

    if (guidance_scale <= 0.0f) {
        guidance_scale = 1.0f;
    } else if (ctx->is_turbo && guidance_scale > 1.0f) {
        fprintf(stderr, "[Pipeline] WARNING: turbo model, forcing guidance_scale=1.0 (was %.1f)\n", guidance_scale);
        guidance_scale = 1.0f;
    }

    if (seed < 0) {
        std::random_device rd;
        seed = (long long) rd() << 32 | rd();
        if (seed < 0) {
            seed = -seed;
        }
    }

    // Audio codes from request JSON (passthrough mode only, NOT cover)
    std::vector<int> codes_vec = parse_codes_string(rr.audio_codes);
    if (!codes_vec.empty()) {
        fprintf(stderr, "[Pipeline] %zu audio codes (%.1fs @ 5Hz)\n", codes_vec.size(),
                (float) codes_vec.size() / 5.0f);
    }
    if (!codes_vec.empty() && !ctx->have_detok) {
        fprintf(stderr, "[Detokenizer] FATAL: failed to load\n");
        return -1;
    }

    // Build schedule: t_i = shift * t / (1 + (shift-1)*t) where t = 1 - i/steps
    std::vector<float> schedule(num_steps);
    for (int i = 0; i < num_steps; i++) {
        float t     = 1.0f - (float) i / (float) num_steps;
        schedule[i] = shift * t / (1.0f + (shift - 1.0f) * t);
    }

    // T = number of 25Hz latent frames for DiT
    // Cover: from source audio. Codes: from code count. Else: from duration.
    int T;
    if (have_cover) {
        T        = T_cover;
        // duration in metas must match actual source length, not JSON default
        duration = (float) T_cover / (float) FRAMES_PER_SECOND;
    } else if (!codes_vec.empty()) {
        T = (int) codes_vec.size() * 5;
    } else {
        T = (int) (duration * FRAMES_PER_SECOND);
    }
    T         = ((T + ctx->dit_cfg.patch_size - 1) / ctx->dit_cfg.patch_size) * ctx->dit_cfg.patch_size;
    int S     = T / ctx->dit_cfg.patch_size;
    int enc_S = 0;

    fprintf(stderr, "[Pipeline] T=%d, S=%d\n", T, S);
    fprintf(stderr, "[Pipeline] seed=%lld, steps=%d, guidance=%.1f, shift=%.1f, duration=%.1fs\n", seed, num_steps,
            guidance_scale, shift, duration);

    if (T > 15000) {
        fprintf(stderr, "[Pipeline] ERROR: T=%d exceeds silence_latent max 15000, skipping\n", T);
        return -1;
    }

    // Repaint mode: resolve start/end, requires --src-audio
    // Both -1 = inactive. One or both >= 0 activates repaint.
    bool  is_repaint = false;
    float rs         = rr.repainting_start;
    float re         = rr.repainting_end;
    if (rs >= 0.0f || re >= 0.0f) {
        if (!have_cover) {
            fprintf(stderr, "[Repaint] ERROR: repainting_start/end require --src-audio\n");
            return -1;
        }
        float src_dur = (float) T_cover * 1920.0f / 48000.0f;
        if (rs < 0.0f) {
            rs = 0.0f;
        }
        if (re < 0.0f) {
            re = src_dur;
        }
        if (rs > src_dur) {
            rs = src_dur;
        }
        if (re > src_dur) {
            re = src_dur;
        }
        if (re > rs) {
            is_repaint = true;
            fprintf(stderr, "[Repaint] region: %.1fs - %.1fs (src=%.1fs)\n", rs, re, src_dur);
        } else {
            fprintf(stderr, "[Repaint] ERROR: repainting_end (%.1f) <= repainting_start (%.1f)\n", re, rs);
            return -1;
        }
    }

    // 2. Build formatted prompts
    // Reference project instruction templates (constants.py TASK_INSTRUCTIONS):
    //   text2music = "Fill the audio semantic mask..."
    //   cover      = "Generate audio semantic tokens..."
    //   repaint    = "Repaint the mask area..."
    //   lego       = "Generate the {TRACK_NAME} track based on the audio context:"
    // Auto-switches to cover when audio_codes are present
    bool        is_cover = have_cover || !codes_vec.empty();
    std::string instruction_str;
    if (is_lego) {
        // Lego mode: force audio_cover_strength=1.0 so all DiT steps see the source audio
        rr.audio_cover_strength = 1.0f;
        fprintf(stderr, "[Lego] track=%s, cover path, strength=1.0\n", rr.lego.c_str());
        // Reference project (task_utils.py:86): track name is UPPERCASE
        std::string track_upper = rr.lego;
        for (char & c : track_upper) {
            c = (char) toupper((unsigned char) c);
        }
        instruction_str = "Generate the " + track_upper + " track based on the audio context:";
    } else if (is_repaint) {
        instruction_str = "Repaint the mask area based on the given conditions:";
    } else if (is_cover) {
        instruction_str = "Generate audio semantic tokens based on the given conditions:";
    } else {
        instruction_str = "Fill the audio semantic mask based on the given conditions:";
    }

    char metas[512];
    snprintf(metas, sizeof(metas), "- bpm: %s\n- timesignature: %s\n- keyscale: %s\n- duration: %d seconds\n", bpm,
             timesig, keyscale, (int) duration);
    std::string text_str = std::string("# Instruction\n") + instruction_str + "\n\n" + "# Caption\n" + caption +
                           "\n\n" + "# Metas\n" + metas + "<|endoftext|>\n";

    std::string lyric_str = std::string("# Languages\n") + language + "\n\n# Lyric\n" + lyrics + "<|endoftext|>";

    // 3. Tokenize
    auto text_ids  = bpe_encode(&ctx->bpe, text_str.c_str(), true);
    auto lyric_ids = bpe_encode(&ctx->bpe, lyric_str.c_str(), true);
    int  S_text    = (int) text_ids.size();
    int  S_lyric   = (int) lyric_ids.size();
    fprintf(stderr, "[Pipeline] caption: %d tokens, lyrics: %d tokens\n", S_text, S_lyric);

    // 4. Text encoder forward (caption only)
    int                H_text = ctx->text_enc.cfg.hidden_size;  // 1024
    std::vector<float> text_hidden(H_text * S_text);

    timer.reset();
    qwen3_forward(&ctx->text_enc, text_ids.data(), S_text, text_hidden.data());
    fprintf(stderr, "[Encode] TextEncoder (%d tokens): %.1f ms\n", S_text, timer.ms());
    debug_dump_2d(&dbg, "text_hidden", text_hidden.data(), S_text, H_text);

    // 5. Lyric embedding (vocab lookup via text encoder)
    timer.reset();
    std::vector<float> lyric_embed(H_text * S_lyric);
    qwen3_embed_lookup(&ctx->text_enc, lyric_ids.data(), S_lyric, lyric_embed.data());
    fprintf(stderr, "[Encode] Lyric vocab lookup (%d tokens): %.1f ms\n", S_lyric, timer.ms());
    debug_dump_2d(&dbg, "lyric_embed", lyric_embed.data(), S_lyric, H_text);

    // Timbre input: source latents when available, silence otherwise
    const int          S_ref = 750;
    std::vector<float> timbre_feats(S_ref * 64);
    if (have_cover) {
        int copy_n = T_cover < S_ref ? T_cover : S_ref;
        memcpy(timbre_feats.data(), cover_latents.data(), (size_t) copy_n * 64 * sizeof(float));
        if (copy_n < S_ref) {
            memcpy(timbre_feats.data() + (size_t) copy_n * 64, ctx->silence_full.data() + (size_t) copy_n * 64,
                   (size_t) (S_ref - copy_n) * 64 * sizeof(float));
        }
        fprintf(stderr, "[Timbre] Using source latents (%d frames, %.1fs)\n", copy_n, (float) copy_n / 25.0f);
    } else {
        memcpy(timbre_feats.data(), ctx->silence_full.data(), S_ref * 64 * sizeof(float));
    }

    timer.reset();
    std::vector<float> enc_hidden;
    cond_ggml_forward(&ctx->cond_enc, text_hidden.data(), S_text, lyric_embed.data(), S_lyric, timbre_feats.data(),
                      S_ref, enc_hidden, &enc_S);
    fprintf(stderr, "[Encode] ConditionEncoder: %.1f ms, enc_S=%d\n", timer.ms(), enc_S);

    debug_dump_2d(&dbg, "enc_hidden", enc_hidden.data(), enc_S, 2048);

    // Decode audio codes if provided (passthrough mode only, NOT cover)
    int                decoded_T = 0;
    std::vector<float> decoded_latents;
    if (!have_cover && !codes_vec.empty()) {
        timer.reset();
        int T_5Hz        = (int) codes_vec.size();
        int T_25Hz_codes = T_5Hz * 5;
        decoded_latents.resize(T_25Hz_codes * Oc);

        int ret = detok_ggml_decode(&ctx->detok, codes_vec.data(), T_5Hz, decoded_latents.data());
        if (ret < 0) {
            fprintf(stderr, "[Detokenizer] FATAL: decode failed\n");
            return -1;
        }
        fprintf(stderr, "[Context] Detokenizer: %.1f ms\n", timer.ms());

        decoded_T = T_25Hz_codes < T ? T_25Hz_codes : T;
        debug_dump_2d(&dbg, "detok_output", decoded_latents.data(), T_25Hz_codes, Oc);
    }

    // Build context: [T, ctx_ch] = src_latents[64] + chunk_mask[64]
    // Cover/Lego: src = cover_latents, mask = 1.0 everywhere
    // Repaint:   src = silence in region / cover outside, mask = 1.0 in region / 0.0 outside
    // Passthrough: detokenized FSQ codes + silence padding, mask = 1.0
    // Text2music: silence only, mask = 1.0
    int repaint_t0 = 0, repaint_t1 = 0;
    if (is_repaint) {
        repaint_t0 = (int) (rs * 48000.0f / 1920.0f);  // sec -> latent frames (25 Hz)
        repaint_t1 = (int) (re * 48000.0f / 1920.0f);
        if (repaint_t0 < 0) {
            repaint_t0 = 0;
        }
        if (repaint_t1 > T) {
            repaint_t1 = T;
        }
        if (repaint_t0 > T) {
            repaint_t0 = T;
        }
        fprintf(stderr, "[Repaint] latent frames: [%d, %d) / %d\n", repaint_t0, repaint_t1, T);
    }
    std::vector<float> context_single(T * ctx_ch);
    if (have_cover) {
        for (int t = 0; t < T; t++) {
            bool          in_region = is_repaint && t >= repaint_t0 && t < repaint_t1;
            // src: silence in repaint region, cover_latents outside
            const float * src       = in_region ?
                                          ctx->silence_full.data() + t * Oc :
                                          ((t < T_cover) ? cover_latents.data() + t * Oc : ctx->silence_full.data() + t * Oc);
            float         mask_val  = is_repaint ? (in_region ? 1.0f : 0.0f) : 1.0f;
            for (int c = 0; c < Oc; c++) {
                context_single[t * ctx_ch + c] = src[c];
            }
            for (int c = 0; c < Oc; c++) {
                context_single[t * ctx_ch + Oc + c] = mask_val;
            }
        }
    } else {
        for (int t = 0; t < T; t++) {
            const float * src =
                (t < decoded_T) ? decoded_latents.data() + t * Oc : ctx->silence_full.data() + (t - decoded_T) * Oc;
            for (int c = 0; c < Oc; c++) {
                context_single[t * ctx_ch + c] = src[c];
            }
            for (int c = 0; c < Oc; c++) {
                context_single[t * ctx_ch + Oc + c] = 1.0f;
            }
        }
    }

    // Replicate context for N batch samples (all identical)
    std::vector<float> context(batch_n * T * ctx_ch);
    for (int b = 0; b < batch_n; b++) {
        memcpy(context.data() + b * T * ctx_ch, context_single.data(), T * ctx_ch * sizeof(float));
    }

    // Cover mode: build silence context for audio_cover_strength switching
    // When step >= cover_steps, DiT switches from cover context to silence context
    // Repaint mode: mask handles region selection, no context switching needed
    std::vector<float> context_silence;
    int                cover_steps = -1;
    if (have_cover && !is_repaint) {
        float cover_strength = rr.audio_cover_strength;
        if (cover_strength < 1.0f) {
            // Build silence context: all frames use silence_latent
            std::vector<float> silence_single(T * ctx_ch);
            for (int t = 0; t < T; t++) {
                const float * src = ctx->silence_full.data() + t * Oc;
                for (int c = 0; c < Oc; c++) {
                    silence_single[t * ctx_ch + c] = src[c];
                }
                for (int c = 0; c < Oc; c++) {
                    silence_single[t * ctx_ch + Oc + c] = 1.0f;
                }
            }
            context_silence.resize(batch_n * T * ctx_ch);
            for (int b = 0; b < batch_n; b++) {
                memcpy(context_silence.data() + b * T * ctx_ch, silence_single.data(), T * ctx_ch * sizeof(float));
            }
            cover_steps = (int) ((float) num_steps * cover_strength);
            fprintf(stderr, "[Cover] audio_cover_strength=%.2f -> switch at step %d/%d\n", cover_strength, cover_steps,
                    num_steps);
        }
    }

    // Generate N noise samples (Philox4x32-10, matches torch.randn on CUDA with bf16)
    std::vector<float> noise(batch_n * Oc * T);
    for (int b = 0; b < batch_n; b++) {
        float * dst = noise.data() + b * Oc * T;
        philox_randn(seed + b, dst, Oc * T, /*bf16_round=*/true);
        fprintf(stderr, "[Context Batch%d] Philox noise seed=%lld, [%d, %d]\n", b, seed + b, T, Oc);
    }

    // DiT Generate
    std::vector<float> output(batch_n * Oc * T);

    // Debug dumps (sample 0)
    debug_dump_2d(&dbg, "noise", noise.data(), T, Oc);
    debug_dump_2d(&dbg, "context", context.data(), T, ctx_ch);

    fprintf(stderr, "[DiT] Starting: T=%d, S=%d, enc_S=%d, steps=%d, batch=%d%s\n", T, S, enc_S, num_steps, batch_n,
            have_cover ? " (cover)" : "");

    timer.reset();
    dit_ggml_generate(&ctx->dit, noise.data(), context.data(), enc_hidden.data(), enc_S, T, batch_n, num_steps,
                      schedule.data(), output.data(), guidance_scale, &dbg,
                      context_silence.empty() ? nullptr : context_silence.data(), cover_steps);
    fprintf(stderr, "[DiT] Total generation: %.1f ms (%.1f ms/sample)\n", timer.ms(), timer.ms() / batch_n);

    debug_dump_2d(&dbg, "dit_output", output.data(), T, Oc);

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
        int                T_latent    = T;
        int                T_audio_max = T_latent * 1920;
        std::vector<float> audio(2 * T_audio_max);

        for (int b = 0; b < batch_n; b++) {
            float * dit_out = output.data() + b * Oc * T;

            timer.reset();
            int T_audio = vae_ggml_decode_tiled(&ctx->vae, dit_out, T_latent, audio.data(), T_audio_max,
                                                ctx->params.vae_chunk, ctx->params.vae_overlap);
            if (T_audio < 0) {
                fprintf(stderr, "[VAE Batch%d] ERROR: decode failed\n", b);
                out[b].samples     = NULL;
                out[b].n_samples   = 0;
                out[b].sample_rate = 48000;
                continue;
            }
            fprintf(stderr, "[VAE Batch%d] Decode: %.1f ms\n", b, timer.ms());

            if (b == 0) {
                debug_dump_2d(&dbg, "vae_audio", audio.data(), 2, T_audio);
            }

            // Copy to output buffer
            int n_total    = 2 * T_audio;
            out[b].samples = (float *) malloc((size_t) n_total * sizeof(float));
            memcpy(out[b].samples, audio.data(), (size_t) n_total * sizeof(float));
            out[b].n_samples   = T_audio;
            out[b].sample_rate = 48000;
        }
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
    if (ctx->have_vae) {
        vae_ggml_free(&ctx->vae);
    }
    cond_ggml_free(&ctx->cond_enc);
    qwen3_free(&ctx->text_enc);
    dit_ggml_free(&ctx->dit);
    delete ctx;
}
