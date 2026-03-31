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

    fprintf(stderr, "[Ace-Synth] All models loaded, turbo=%s\n", ctx->is_turbo ? "yes" : "no");
    if (!params->use_fa) {
        fprintf(stderr, "[Ace-Synth] flash attention disabled\n");
    }
    if (params->clamp_fp16) {
        fprintf(stderr, "[Ace-Synth] FP16 clamp enabled\n");
    }

    return ctx;
}

int ace_synth_generate(AceSynth *         ctx,
                       const AceRequest * reqs,
                       const float *      src_audio,
                       int                src_len,
                       int                batch_n,
                       AceAudio *         out,
                       bool (*cancel)(void *),
                       void * cancel_data) {
    if (!ctx || !reqs || !out || batch_n < 1 || batch_n > 9) {
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

    // Shared params from first request (mode, duration, DiT settings).
    // Per-batch: caption, lyrics, metadata, audio_codes, and seed come from reqs[b].
    // seed must be resolved (non-negative) before calling this function.
    AceRequest rr = reqs[0];

    // task_type is the master. Empty = text2music.
    std::string task = rr.task_type;
    if (task.empty()) {
        task = TASK_TEXT2MUSIC;
    }

    // repaint and complete use a binary mask on the source latents
    bool  is_repaint = (task == TASK_REPAINT || task == TASK_COMPLETE);
    float rs         = rr.repainting_start;
    float re         = rr.repainting_end;

    // use_source_context: task requires source latents in DiT context.
    // Separates "audio physically present" (have_cover) from "DiT should see source".
    // text2music ignores src_audio even if provided (uses it for timbre only).
    bool use_source_context = (task == TASK_COVER || task == TASK_REPAINT || task == TASK_LEGO ||
                               task == TASK_EXTRACT || task == TASK_COMPLETE);

    // validation: tasks that need source audio
    if (task == TASK_COVER || task == TASK_REPAINT || task == TASK_LEGO || task == TASK_EXTRACT ||
        task == TASK_COMPLETE) {
        if (!have_cover) {
            fprintf(stderr, "[%s] ERROR: requires source audio\n", task.c_str());
            return -1;
        }
    }

    // track name validation for lego/extract/complete
    if (task == TASK_LEGO || task == TASK_EXTRACT || task == TASK_COMPLETE) {
        if (!rr.track.empty()) {
            bool valid = false;
            for (int k = 0; k < TRACK_NAMES_COUNT; k++) {
                if (rr.track == TRACK_NAMES[k]) {
                    valid = true;
                    break;
                }
            }
            if (!valid) {
                fprintf(stderr, "[%s] WARNING: '%s' is not a standard track name\n", task.c_str(), rr.track.c_str());
            }
        }
    }
    fprintf(stderr, "[Pipeline] task=%s\n", task.c_str());

    // Extract shared params from first request
    float duration = rr.duration > 0 ? rr.duration : 30.0f;

    // Resolve DiT sampling params: 0 = auto-detect from model type.
    // Turbo: 8 steps, guidance=1.0, shift=3.0
    // Base/SFT: 50 steps, guidance=1.0, shift=1.0
    int   num_steps      = rr.inference_steps;
    float guidance_scale = rr.guidance_scale;
    float shift          = rr.shift;

    if (num_steps <= 0) {
        num_steps = ctx->is_turbo ? 8 : 50;
    }
    if (num_steps > 100) {
        fprintf(stderr, "[Pipeline] WARNING: inference_steps %d clamped to 100\n", num_steps);
        num_steps = 100;
    }

    if (guidance_scale <= 0.0f) {
        guidance_scale = 1.0f;
    } else if (ctx->is_turbo && guidance_scale > 1.0f) {
        fprintf(stderr, "[Pipeline] WARNING: turbo model, forcing guidance_scale=1.0 (was %.1f)\n", guidance_scale);
        guidance_scale = 1.0f;
    }

    if (shift <= 0.0f) {
        shift = ctx->is_turbo ? 3.0f : 1.0f;
    }

    // Audio codes: scan all requests to determine T from the longest code set.
    // Per-batch codes are decoded in the context building loop below.
    // Shorter code sets are padded with silence, longer ones are never truncated.
    int  max_codes_len = 0;
    bool have_codes    = false;
    for (int b = 0; b < batch_n; b++) {
        std::vector<int> cb = parse_codes_string(reqs[b].audio_codes);
        if ((int) cb.size() > max_codes_len) {
            max_codes_len = (int) cb.size();
        }
        if (!cb.empty()) {
            have_codes = true;
        }
    }
    if (have_codes) {
        fprintf(stderr, "[Pipeline] max audio codes across batch: %d (%.1fs @ 5Hz)\n", max_codes_len,
                (float) max_codes_len / 5.0f);
    }
    if (have_codes && !ctx->have_detok) {
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
    // Source tasks: from source audio. Codes: from code count. Else: from duration.
    int T;
    if (use_source_context && have_cover) {
        T        = T_cover;
        // duration in metas must match actual source length, not JSON default
        duration = (float) T_cover / (float) FRAMES_PER_SECOND;
    } else if (have_codes) {
        T = max_codes_len * 5;
    } else {
        T = (int) (duration * FRAMES_PER_SECOND);
    }
    T         = ((T + ctx->dit_cfg.patch_size - 1) / ctx->dit_cfg.patch_size) * ctx->dit_cfg.patch_size;
    int S     = T / ctx->dit_cfg.patch_size;
    int enc_S = 0;

    fprintf(stderr, "[Pipeline] T=%d, S=%d\n", T, S);
    fprintf(stderr, "[Pipeline] seed=%lld, steps=%d, guidance=%.1f, shift=%.1f, duration=%.1fs\n", (long long) rr.seed,
            num_steps, guidance_scale, shift, duration);

    if (T > 15000) {
        fprintf(stderr, "[Pipeline] ERROR: T=%d exceeds silence_latent max 15000, skipping\n", T);
        return -1;
    }

    // Repaint region: clamp start/end to source duration.
    if (is_repaint) {
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
        if (re <= rs) {
            fprintf(stderr, "[Repaint] ERROR: repainting_end (%.1f) <= repainting_start (%.1f)\n", re, rs);
            return -1;
        }
        fprintf(stderr, "[Repaint] Region: %.1fs - %.1fs (src=%.1fs)\n", rs, re, src_dur);
    }

    // DiT instruction from task_type (drives cross-attention behavior).
    // Track name is UPPERCASE in the instruction (matches Python task_utils.py).
    std::string track_upper = rr.track;
    for (char & c : track_upper) {
        c = (char) toupper((unsigned char) c);
    }

    std::string instruction_str;
    if (task == TASK_LEGO) {
        instruction_str = dit_instr_lego(track_upper);
    } else if (task == TASK_EXTRACT) {
        instruction_str = dit_instr_extract(track_upper);
    } else if (task == TASK_COMPLETE) {
        instruction_str = dit_instr_complete(track_upper);
    } else if (task == TASK_REPAINT) {
        instruction_str = DIT_INSTR_REPAINT;
    } else if (task == TASK_COVER || have_codes) {
        // cover instruction when task is cover, or text2music with LM-generated codes
        // (DiT sees decoded latents in context and was trained with this instruction).
        instruction_str = DIT_INSTR_COVER;
    } else {
        instruction_str = DIT_INSTR_TEXT2MUSIC;
    }

    // lego/extract/complete: force strength=1.0 (all DiT steps see source audio)
    if (task == TASK_LEGO || task == TASK_EXTRACT || task == TASK_COMPLETE) {
        rr.audio_cover_strength = 1.0f;
    }

    // 2. Timbre features (shared: same reference audio or silence for all batch elements)
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

    // 3. Per-batch text encoding.
    // Each batch element gets its own caption, lyrics, and metadata encoded independently.
    // TextEncoder + CondEncoder run in series (cheap: ~13ms per element).
    // Results are padded to max_enc_S with null_cond and stacked for a single DiT batch pass.
    int H_text = ctx->text_enc.cfg.hidden_size;  // 1024
    int H_cond = ctx->dit.cfg.hidden_size;       // 2048

    // read null_condition_emb from GPU for padding shorter encodings
    std::vector<float> null_cond_vec(H_cond);
    if (ctx->dit.null_condition_emb) {
        int emb_n = (int) ggml_nelements(ctx->dit.null_condition_emb);
        if (ctx->dit.null_condition_emb->type == GGML_TYPE_BF16) {
            std::vector<uint16_t> bf16_buf(emb_n);
            ggml_backend_tensor_get(ctx->dit.null_condition_emb, bf16_buf.data(), 0, emb_n * sizeof(uint16_t));
            for (int i = 0; i < emb_n; i++) {
                uint32_t w = (uint32_t) bf16_buf[i] << 16;
                memcpy(&null_cond_vec[i], &w, 4);
            }
        } else {
            ggml_backend_tensor_get(ctx->dit.null_condition_emb, null_cond_vec.data(), 0, emb_n * sizeof(float));
        }
    }

    // encode each batch element independently
    std::vector<std::vector<float>> per_enc(batch_n);
    std::vector<int>                per_enc_S(batch_n);

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
                 bpm_b, timesig_b, keyscale_b, (int) duration);
        std::string text_str = std::string("# Instruction\n") + instruction_str + "\n\n" + "# Caption\n" + rb.caption +
                               "\n\n" + "# Metas\n" + metas_b + "<|endoftext|>\n";
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
        timer.reset();
        cond_ggml_forward(&ctx->cond_enc, text_hidden.data(), S_text, lyric_embed.data(), S_lyric, timbre_feats.data(),
                          S_ref, per_enc[b], &per_enc_S[b]);
        fprintf(stderr, "[Encode Batch%d] %d+%d tokens -> enc_S=%d, %.1f ms\n", b, S_text, S_lyric, per_enc_S[b],
                timer.ms());

        if (b == 0) {
            debug_dump_2d(&dbg, "text_hidden", text_hidden.data(), S_text, H_text);
            debug_dump_2d(&dbg, "lyric_embed", lyric_embed.data(), S_lyric, H_text);
            debug_dump_2d(&dbg, "enc_hidden", per_enc[b].data(), per_enc_S[b], H_cond);
        }
    }

    // non-cover encoding: re-encode text with text2music instruction for post-switch phase.
    // same lyrics/timbre, only the instruction changes (cover -> text2music).
    bool need_enc_switch = use_source_context && !is_repaint && rr.audio_cover_strength < 1.0f;
    std::vector<std::vector<float>> per_enc_nc(batch_n);
    std::vector<int>                per_enc_S_nc(batch_n, 0);

    if (need_enc_switch) {
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
                     keyscale_b, (int) duration);
            std::string text_str = std::string("# Instruction\n") + DIT_INSTR_TEXT2MUSIC + "\n\n" + "# Caption\n" +
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
                              timbre_feats.data(), S_ref, per_enc_nc[b], &per_enc_S_nc[b]);
            fprintf(stderr, "[Encode Batch%d] non-cover: %d+%d tokens -> enc_S=%d\n", b, S_text, S_lyric,
                    per_enc_S_nc[b]);
        }
    }

    // find max enc_S across both cover and non-cover encodings,
    // pad shorter encodings with null_cond, stack into [H, max_enc_S, N]
    int max_enc_S = 0;
    for (int b = 0; b < batch_n; b++) {
        if (per_enc_S[b] > max_enc_S) {
            max_enc_S = per_enc_S[b];
        }
        if (need_enc_switch && per_enc_S_nc[b] > max_enc_S) {
            max_enc_S = per_enc_S_nc[b];
        }
    }
    enc_S = max_enc_S;

    std::vector<float> enc_hidden(H_cond * max_enc_S * batch_n);
    for (int b = 0; b < batch_n; b++) {
        float * dst = enc_hidden.data() + b * max_enc_S * H_cond;
        memcpy(dst, per_enc[b].data(), (size_t) per_enc_S[b] * H_cond * sizeof(float));
        for (int s = per_enc_S[b]; s < max_enc_S; s++) {
            memcpy(dst + s * H_cond, null_cond_vec.data(), H_cond * sizeof(float));
        }
    }

    // pad and stack non-cover encoding (same max_enc_S for graph compatibility)
    std::vector<float> enc_hidden_nc;
    std::vector<int>   per_enc_S_nc_final;
    if (need_enc_switch) {
        enc_hidden_nc.resize(H_cond * max_enc_S * batch_n);
        per_enc_S_nc_final.resize(batch_n);
        for (int b = 0; b < batch_n; b++) {
            float * dst = enc_hidden_nc.data() + b * max_enc_S * H_cond;
            memcpy(dst, per_enc_nc[b].data(), (size_t) per_enc_S_nc[b] * H_cond * sizeof(float));
            for (int s = per_enc_S_nc[b]; s < max_enc_S; s++) {
                memcpy(dst + s * H_cond, null_cond_vec.data(), H_cond * sizeof(float));
            }
            per_enc_S_nc_final[b] = per_enc_S_nc[b];
        }
    }

    if (batch_n > 1) {
        fprintf(stderr, "[Encode] Per-batch encoding done: max_enc_S=%d\n", max_enc_S);
    }

    // Build context: [batch_n, T, ctx_ch] = src_latents[64] + chunk_mask[64]
    // Cover/Lego/Repaint: shared context replicated (cover_latents from src_audio).
    // Passthrough: per-batch detokenized FSQ codes + silence padding, mask = 1.0.
    // Text2music: silence only, mask = 1.0.
    int repaint_t0 = 0, repaint_t1 = 0;
    if (is_repaint) {
        repaint_t0 = (int) (rs * 48000.0f / 1920.0f);
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
        fprintf(stderr, "[Repaint] Latent frames: [%d, %d) / %d\n", repaint_t0, repaint_t1, T);
    }

    std::vector<float> context(batch_n * T * ctx_ch);

    if (use_source_context && have_cover) {
        // Cover/Lego/Repaint: build once, replicate (cover_latents are shared)
        std::vector<float> context_single(T * ctx_ch);
        for (int t = 0; t < T; t++) {
            bool          in_region = is_repaint && t >= repaint_t0 && t < repaint_t1;
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
        for (int b = 0; b < batch_n; b++) {
            memcpy(context.data() + b * T * ctx_ch, context_single.data(), T * ctx_ch * sizeof(float));
        }
    } else {
        // Text2music / Passthrough: per-batch context with per-batch audio_codes
        for (int b = 0; b < batch_n; b++) {
            float * ctx_dst = context.data() + b * T * ctx_ch;

            // decode this batch item's audio codes (if any)
            int                decoded_T = 0;
            std::vector<float> decoded_latents;
            std::vector<int>   codes_b = parse_codes_string(reqs[b].audio_codes);
            if (!codes_b.empty()) {
                timer.reset();
                int T_5Hz        = (int) codes_b.size();
                int T_25Hz_codes = T_5Hz * 5;
                decoded_latents.resize(T_25Hz_codes * Oc);

                int ret = detok_ggml_decode(&ctx->detok, codes_b.data(), T_5Hz, decoded_latents.data());
                if (ret < 0) {
                    fprintf(stderr, "[Detokenizer Batch%d] FATAL: decode failed\n", b);
                    return -1;
                }
                fprintf(stderr, "[Context Batch%d] Detokenizer: %.1f ms, %d codes\n", b, timer.ms(), T_5Hz);

                decoded_T = T_25Hz_codes < T ? T_25Hz_codes : T;
                if (b == 0) {
                    debug_dump_2d(&dbg, "detok_output", decoded_latents.data(), T_25Hz_codes, Oc);
                }
            }

            // fill context: decoded latents then silence, mask = 1.0
            for (int t = 0; t < T; t++) {
                const float * src =
                    (t < decoded_T) ? decoded_latents.data() + t * Oc : ctx->silence_full.data() + (t - decoded_T) * Oc;
                for (int c = 0; c < Oc; c++) {
                    ctx_dst[t * ctx_ch + c] = src[c];
                }
                for (int c = 0; c < Oc; c++) {
                    ctx_dst[t * ctx_ch + Oc + c] = 1.0f;
                }
            }
        }
    }

    // Cover mode: build silence context for audio_cover_strength switching
    // When step >= cover_steps, DiT switches from cover context to silence context
    // Repaint mode: mask handles region selection, no context switching needed
    std::vector<float> context_silence;
    int                cover_steps = -1;
    if (use_source_context && !is_repaint) {
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

    // Generate N noise samples (Philox4x32-10, matches torch.randn on CUDA with bf16).
    // Each batch item uses its own seed from the request.
    std::vector<float> noise(batch_n * Oc * T);
    for (int b = 0; b < batch_n; b++) {
        float * dst = noise.data() + b * Oc * T;
        philox_randn(reqs[b].seed, dst, Oc * T, /*bf16_round=*/true);
        fprintf(stderr, "[Context Batch%d] Philox noise seed=%lld, [%d, %d]\n", b, (long long) reqs[b].seed, T, Oc);
    }

    // cover_noise_strength: start diffusion closer to source instead of pure noise.
    // xt = nearest_t * noise + (1 - nearest_t) * cover_latents, then truncate schedule.
    if (use_source_context && have_cover && rr.cover_noise_strength > 0.0f) {
        float effective_noise_level = 1.0f - rr.cover_noise_strength;
        // find nearest timestep in schedule
        int   start_idx             = 0;
        float best_dist             = fabsf(schedule[0] - effective_noise_level);
        for (int i = 1; i < num_steps; i++) {
            float dist = fabsf(schedule[i] - effective_noise_level);
            if (dist < best_dist) {
                best_dist = dist;
                start_idx = i;
            }
        }
        float nearest_t = schedule[start_idx];
        // blend: xt = nearest_t * noise + (1 - nearest_t) * cover_latents
        for (int b = 0; b < batch_n; b++) {
            float * n = noise.data() + b * Oc * T;
            for (int t = 0; t < T; t++) {
                int           t_src = t < T_cover ? t : T_cover - 1;
                const float * src   = cover_latents.data() + t_src * Oc;
                for (int c = 0; c < Oc; c++) {
                    int idx = t * Oc + c;
                    n[idx]  = nearest_t * n[idx] + (1.0f - nearest_t) * src[c];
                }
            }
        }
        // truncate schedule
        schedule.erase(schedule.begin(), schedule.begin() + start_idx);
        num_steps = (int) schedule.size();
        // recalculate cover_steps with remaining steps
        if (cover_steps >= 0) {
            cover_steps = (int) ((float) num_steps * rr.audio_cover_strength);
        }
        fprintf(stderr, "[Cover] cover_noise_strength=%.2f -> noise_level=%.4f, nearest_t=%.4f, remaining_steps=%d\n",
                rr.cover_noise_strength, effective_noise_level, nearest_t, num_steps);
    }

    // DiT Generate
    std::vector<float> output(batch_n * Oc * T);

    // Per-batch sequence lengths for attention padding masks.
    // Within a synth_batch_size group, all elements share the same T (same codes),
    // so per_S[b] = S for all b. The per_enc_S[] array has real encoder lengths
    // from per-batch text encoding above.
    // These become meaningful when the server/CLI batches requests with different T.
    std::vector<int> per_S(batch_n, S);

    // Debug dumps (sample 0)
    debug_dump_2d(&dbg, "noise", noise.data(), T, Oc);
    debug_dump_2d(&dbg, "context", context.data(), T, ctx_ch);

    fprintf(stderr, "[DiT] Starting: T=%d, S=%d, enc_S=%d, steps=%d, batch=%d%s\n", T, S, enc_S, num_steps, batch_n,
            use_source_context ? " (cover)" : "");

    // repaint injection buffer: cover_latents padded to T with silence.
    // T may exceed T_cover due to patch_size rounding; frames beyond T_cover use silence.
    std::vector<float> repaint_src;
    if (is_repaint) {
        repaint_src.resize(T * Oc);
        for (int t = 0; t < T; t++) {
            const float * src = (t < T_cover) ? cover_latents.data() + t * Oc : ctx->silence_full.data() + t * Oc;
            memcpy(repaint_src.data() + t * Oc, src, Oc * sizeof(float));
        }
    }

    timer.reset();
    int dit_rc = dit_ggml_generate(
        &ctx->dit, noise.data(), context.data(), enc_hidden.data(), enc_S, T, batch_n, num_steps, schedule.data(),
        output.data(), guidance_scale, &dbg, context_silence.empty() ? nullptr : context_silence.data(), cover_steps,
        cancel, cancel_data, per_S.data(), per_enc_S.data(), enc_hidden_nc.empty() ? nullptr : enc_hidden_nc.data(),
        per_enc_S_nc_final.empty() ? nullptr : per_enc_S_nc_final.data(),
        repaint_src.empty() ? nullptr : repaint_src.data(), repaint_t0, repaint_t1);
    if (dit_rc != 0) {
        return -1;
    }
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
