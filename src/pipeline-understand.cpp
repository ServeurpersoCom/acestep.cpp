// pipeline-understand.cpp - ACE-Step reverse pipeline (audio -> metadata)
//
// Wraps VAE encoder, FSQ tokenizer, and Qwen3 LM for audio understanding:
// audio -> latents -> codes -> LM understand -> metadata + lyrics + caption.

#include "pipeline-understand.h"

#include "backend.h"
#include "bpe.h"
#include "debug.h"
#include "fsq-tok.h"
#include "gguf-weights.h"
#include "metadata-fsm.h"
#include "prompt.h"
#include "qwen3-lm.h"
#include "sampling.h"
#include "timer.h"
#include "vae-enc.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

// Parse comma-separated codes string "3101,11837,27514,..." into vector
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

struct AceUnderstand {
    AceUnderstandParams params;

    // VAE encoder (for audio input mode)
    bool       have_vae_enc;
    VAEEncoder vae_enc;

    // FSQ tokenizer (for audio input mode)
    bool           have_fsq;
    TokGGML        fsq;
    ggml_backend_t fsq_backend;

    // silence latent from DiT GGUF (needed for FSQ tokenizer padding)
    std::vector<float> silence;

    // LM (owned when loaded from model_path, shared when passed via params)
    bool           have_lm;
    bool           owns_lm;        // true = we loaded it, false = shared from pipeline-lm
    BPETokenizer   bpe_storage;    // only used when owns_lm
    Qwen3LM        model_storage;  // only used when owns_lm
    Qwen3LM *      model;          // active pointer (to model_storage or shared)
    BPETokenizer * bpe;            // active pointer (to bpe_storage or shared)
    MetadataFSM    fsm_template;

    // DiT model type (for setting inference defaults in output)
    bool is_turbo;

    double load_ms;
};

void ace_understand_default_params(AceUnderstandParams * p) {
    p->model_path   = NULL;
    p->dit_path     = NULL;
    p->vae_path     = NULL;
    p->dump_dir     = NULL;
    p->max_seq      = 8192;
    p->use_fsm      = true;
    p->use_fa       = true;
    p->vae_chunk    = 256;
    p->vae_overlap  = 64;
    p->shared_model = NULL;
    p->shared_bpe   = NULL;
}

AceUnderstand * ace_understand_load(const AceUnderstandParams * params) {
    if (!params->model_path && !params->shared_model && !params->dump_dir) {
        fprintf(stderr, "[Understand] ERROR: model_path required (or shared_model, or dump_dir for tok-only)\n");
        return NULL;
    }

    Timer  t_load;
    auto * ctx        = new AceUnderstand();
    ctx->params       = *params;
    ctx->have_vae_enc = false;
    ctx->have_fsq     = false;
    ctx->have_lm      = false;
    ctx->owns_lm      = false;
    ctx->model        = nullptr;
    ctx->bpe          = nullptr;
    ctx->fsq_backend  = nullptr;
    ctx->is_turbo     = false;

    // Load VAE encoder (for audio encoding)
    if (params->vae_path) {
        fprintf(stderr, "[VAE-Enc] Loading %s...\n", params->vae_path);
        ctx->vae_enc = {};
        vae_enc_load(&ctx->vae_enc, params->vae_path);
        ctx->have_vae_enc = true;
    }

    // Load FSQ tokenizer + silence_latent from DiT GGUF
    // Tokenizer weights live in the DiT GGUF (prefix "tokenizer.")
    if (params->dit_path) {
        // Load silence_latent from DiT GGUF (needed for FSQ tokenizer padding)
        GGUFModel gf = {};
        if (!gf_load(&gf, params->dit_path)) {
            fprintf(stderr, "[DiT] FATAL: cannot open %s\n", params->dit_path);
            delete ctx;
            return NULL;
        }
        const void * sl = gf_get_data(gf, "silence_latent");
        if (!sl) {
            fprintf(stderr, "[DiT] FATAL: silence_latent not found in %s\n", params->dit_path);
            gf_close(&gf);
            delete ctx;
            return NULL;
        }
        ctx->silence.resize(15000 * 64);
        memcpy(ctx->silence.data(), sl, 15000 * 64 * sizeof(float));

        // detect turbo vs SFT/base
        ctx->is_turbo = gf_get_bool(gf, "acestep.is_turbo");
        gf_close(&gf);

        // FSQ tokenizer (CPU backend, weights in DiT GGUF)
        ctx->fsq_backend = cpu_backend_new(backend_cpu_n_threads());
        if (!ctx->fsq_backend) {
            fprintf(stderr, "[Tok] FATAL: failed to init CPU backend\n");
            delete ctx;
            return NULL;
        }
        ctx->fsq = {};
        if (!tok_ggml_load(&ctx->fsq, params->dit_path, ctx->fsq_backend, ctx->fsq_backend)) {
            fprintf(stderr, "[Tok] FATAL: load failed\n");
            ggml_backend_free(ctx->fsq_backend);
            delete ctx;
            return NULL;
        }
        ctx->have_fsq = true;
    }

    // Step 2: LM + BPE
    // Two modes: shared (pointers from pipeline-lm) or owned (load from GGUF).
    // The server passes shared pointers to avoid a second ~5GB copy of Qwen3.
    // The CLI binary always loads its own copy (shared_model = NULL).
    if (params->shared_model && params->shared_bpe) {
        // shared: just grab pointers, we don't own the memory
        ctx->model   = params->shared_model;
        ctx->bpe     = params->shared_bpe;
        ctx->owns_lm = false;
        ctx->have_lm = true;
        fprintf(stderr, "[Load] LM: shared from pipeline-lm\n");

        // FSM for constrained CoT metadata decoding
        if (params->use_fsm) {
            ctx->fsm_template.init(*ctx->bpe, ctx->model->cfg.vocab_size);
        }
    } else if (params->model_path) {
        // owned: load our own copy from GGUF (standalone CLI path)
        if (!load_bpe_from_gguf(&ctx->bpe_storage, params->model_path)) {
            delete ctx;
            return NULL;
        }

        if (!qw3lm_load(&ctx->model_storage, params->model_path, params->max_seq, 1)) {
            delete ctx;
            return NULL;
        }
        ctx->model_storage.use_flash_attn = params->use_fa;
        fprintf(stderr, "[Load] LM: %.0fms\n", t_load.ms());

        ctx->model   = &ctx->model_storage;
        ctx->bpe     = &ctx->bpe_storage;
        ctx->owns_lm = true;
        ctx->have_lm = true;

        // FSM for constrained CoT metadata decoding
        if (params->use_fsm) {
            ctx->fsm_template.init(*ctx->bpe, ctx->model->cfg.vocab_size);
        }
    }

    ctx->load_ms = t_load.ms();
    fprintf(stderr, "[Understand] Loaded in %.0fms\n", ctx->load_ms);
    return ctx;
}

int ace_understand_generate(AceUnderstand *    ctx,
                            const float *      src_audio,
                            int                src_len,
                            const AceRequest * req,
                            AceRequest *       out,
                            bool (*cancel)(void *),
                            void * cancel_data) {
    if (!ctx || !req || !out) {
        return -1;
    }

    Timer t_total;

    // Resolve seed (same as ace-lm)
    long long seed = req->seed;
    if (seed < 0) {
        std::random_device rd;
        seed = (int64_t) rd() << 32 | rd();
        if (seed < 0) {
            seed = -seed;
        }
    }

    // Generation params from request
    float temperature = req->lm_temperature;
    float top_p       = req->lm_top_p;
    int   top_k       = req->lm_top_k;

    std::vector<int> codes;

    // Step 1: get audio codes
    // --src-audio: full pipeline (VAE encode + FSQ tokenize)
    // --request without --src-audio: parse audio_codes from JSON
    // --request + --src-audio: audio from file, params from JSON
    if (src_audio && src_len > 0) {
        if (!ctx->have_vae_enc || !ctx->have_fsq) {
            fprintf(stderr, "[Understand] ERROR: audio input requires VAE + DiT models\n");
            return -1;
        }

        // VAE encode: audio -> latents [T_25Hz, 64]
        Timer              t_vae;
        int                max_T_lat = (src_len / 1920) + 64;
        std::vector<float> latents((size_t) max_T_lat * 64);

        int T_25Hz = vae_enc_encode_tiled(&ctx->vae_enc, src_audio, src_len, latents.data(), max_T_lat,
                                          ctx->params.vae_chunk, ctx->params.vae_overlap);
        if (T_25Hz < 0) {
            fprintf(stderr, "[VAE] FATAL: encode failed\n");
            return -1;
        }
        fprintf(stderr, "[VAE] Encoded: %d latent frames (%.2fs), %.0fms\n", T_25Hz,
                (float) T_25Hz * 1920.0f / 48000.0f, t_vae.ms());

        // FSQ tokenize: latents [T_25Hz, 64] -> codes [T_5Hz]
        Timer t_tok;
        int   max_codes = (T_25Hz + 4) / 5;
        codes.resize(max_codes);
        int T_5Hz = tok_ggml_encode(&ctx->fsq, latents.data(), T_25Hz, codes.data(), ctx->silence.data());
        if (T_5Hz < 0) {
            fprintf(stderr, "[Tok] FATAL: tokenize failed\n");
            return -1;
        }
        codes.resize(T_5Hz);
        fprintf(stderr, "[Tok] %d codes (%.2fs @ 5Hz), %.0fms\n", T_5Hz, (float) T_5Hz / 5.0f, t_tok.ms());

        // --dump: save latents and codes for test-tok-cossim.py
        if (ctx->params.dump_dir) {
            DebugDumper dbg;
            debug_init(&dbg, ctx->params.dump_dir);
            debug_dump_2d(&dbg, "tok_latents", latents.data(), T_25Hz, 64);
            char cpath[1024];
            snprintf(cpath, sizeof(cpath), "%s/tok_codes.bin", ctx->params.dump_dir);
            FILE * fc = fopen(cpath, "wb");
            if (fc) {
                fwrite(codes.data(), sizeof(int), (size_t) T_5Hz, fc);
                fclose(fc);
                fprintf(stderr, "[Debug] tok_codes: [%d] int32\n", T_5Hz);
            }
        }
    } else {
        // Codes from JSON: parse audio_codes string "3101,11837,..."
        codes = parse_codes_string(req->audio_codes);
        if (codes.empty()) {
            fprintf(stderr, "[Understand] ERROR: no audio and no audio_codes\n");
            return -1;
        }
        fprintf(stderr, "[Request] %zu codes from request\n", codes.size());
    }

    // --dump without --model: tok-only mode, skip LM
    if (!ctx->have_lm) {
        request_init(out);
        std::string codes_str;
        for (size_t i = 0; i < codes.size(); i++) {
            if (i > 0) {
                codes_str += ',';
            }
            codes_str += std::to_string(codes[i]);
        }
        out->audio_codes = codes_str;
        fprintf(stderr, "[Understand] Dump-only mode, %zu codes\n", codes.size());
        return 0;
    }

    int V = ctx->model->cfg.vocab_size;

    // FSM for constrained CoT metadata decoding
    bool        use_fsm = ctx->params.use_fsm;
    MetadataFSM fsm;
    if (use_fsm) {
        fsm = ctx->fsm_template;
    }

    // Step 3: build understand prompt
    // System: understand instruction
    // User: raw audio code tokens (not BPE text)
    // The LM sees the codes and generates metadata + lyrics
    std::vector<int> prompt = build_understand_prompt(*ctx->bpe, codes.data(), (int) codes.size());
    fprintf(stderr, "[Prompt] %zu tokens (%zu codes + framing)\n", prompt.size(), codes.size());

    // Step 4: prefill
    Timer              t_gen;
    std::vector<float> logits(V);
    qw3lm_reset_kv(ctx->model, 0);
    qw3lm_forward(ctx->model, prompt.data(), (int) prompt.size(), 0, logits.data());
    fprintf(stderr, "[Prefill] %.0fms, %zu tokens, seed=%lld\n", t_gen.ms(), prompt.size(), seed);

    // Step 5: autoregressive decode
    // No CFG, no batch. Single sequence, stop at <|im_end|>.
    // FSM constrains the CoT metadata block (<think>...</think>).
    // After </think>, generate free-form lyrics with audio codes blocked.
    std::mt19937     rng((uint32_t) seed);
    std::vector<int> gen_tokens;
    bool             past_think = false;
    int              max_tokens = 4096;

    for (int step = 0; step < max_tokens; step++) {
        if (cancel && cancel(cancel_data)) {
            fprintf(stderr, "[Understand] Cancelled at step %d\n", step);
            return -1;
        }

        // After </think>: block audio codes so the LM only generates text
        if (past_think) {
            for (int i = AUDIO_CODE_BASE; i < V; i++) {
                logits[i] = -INFINITY;
            }
        }

        // FSM mask (only active during CoT metadata phase)
        if (use_fsm && fsm.enabled && !past_think) {
            fsm.apply_mask(logits.data());
        }

        int tok = sample_top_k_p(logits.data(), V, temperature, top_p, top_k, rng);

        if (tok == TOKEN_IM_END) {
            break;
        }

        // Track FSM state
        if (use_fsm && fsm.enabled && !past_think) {
            fsm.update(tok);
        }

        if (tok == TOKEN_THINK_END) {
            past_think = true;
        }

        gen_tokens.push_back(tok);

        // Next token forward
        qw3lm_forward(ctx->model, &tok, 1, 0, logits.data());
    }

    fprintf(stderr, "[Decode] %zu tokens, %.0fms (%.1f tok/s)\n", gen_tokens.size(), t_gen.ms(),
            (float) gen_tokens.size() / (t_gen.ms() / 1000.0f));

    // Step 6: decode tokens to text, parse CoT metadata + lyrics
    std::string text   = bpe_decode(*ctx->bpe, gen_tokens);
    AcePrompt   parsed = {};
    parse_cot_and_lyrics(text, &parsed);

    if (parsed.bpm > 0) {
        fprintf(stderr, "[Result] bpm: %d\n", parsed.bpm);
    }
    if (parsed.duration > 0) {
        fprintf(stderr, "[Result] duration: %.0fs\n", parsed.duration);
    }
    if (!parsed.keyscale.empty()) {
        fprintf(stderr, "[Result] keyscale: %s\n", parsed.keyscale.c_str());
    }
    if (!parsed.timesignature.empty()) {
        fprintf(stderr, "[Result] timesig: %s\n", parsed.timesignature.c_str());
    }
    if (!parsed.vocal_language.empty()) {
        fprintf(stderr, "[Result] language: %s\n", parsed.vocal_language.c_str());
    }
    if (!parsed.caption.empty()) {
        fprintf(stderr, "[Result] caption: %.80s%s\n", parsed.caption.c_str(), parsed.caption.size() > 80 ? "..." : "");
    }
    if (!parsed.lyrics.empty()) {
        fprintf(stderr, "[Result] lyrics: %zu chars\n", parsed.lyrics.size());
    }

    // Step 7: write output JSON (reusable as ace-synth input with codes)
    request_init(out);
    out->caption        = parsed.caption;
    out->lyrics         = parsed.lyrics;
    out->bpm            = parsed.bpm;
    out->duration       = parsed.duration;
    out->keyscale       = parsed.keyscale;
    out->timesignature  = parsed.timesignature;
    out->vocal_language = parsed.vocal_language;
    out->seed           = seed;

    // Build audio_codes string from recovered codes (comma-separated)
    std::string codes_str;
    for (size_t i = 0; i < codes.size(); i++) {
        if (i > 0) {
            codes_str += ',';
        }
        codes_str += std::to_string(codes[i]);
    }
    out->audio_codes = codes_str;
    fprintf(stderr, "[Result] audio_codes: %zu codes\n", codes.size());

    // Set DiT defaults from model type (turbo vs SFT/base)
    if (ctx->is_turbo) {
        out->inference_steps = 8;
        out->shift           = 3.0f;
        out->guidance_scale  = 1.0f;
    } else {
        out->inference_steps = 50;
        out->shift           = 1.0f;
        out->guidance_scale  = 1.0f;
    }

    fprintf(stderr, "[Understand] Load %.0f | Total %.0fms | seed=%lld\n", ctx->load_ms, t_total.ms(), seed);
    return 0;
}

void ace_understand_free(AceUnderstand * ctx) {
    if (!ctx) {
        return;
    }
    // only free the LM if we loaded it ourselves (CLI path).
    // in server mode the LM is owned by pipeline-lm.
    if (ctx->have_lm && ctx->owns_lm) {
        qw3lm_free(&ctx->model_storage);
    }
    if (ctx->have_fsq) {
        tok_ggml_free(&ctx->fsq);
        ggml_backend_free(ctx->fsq_backend);
    }
    if (ctx->have_vae_enc) {
        vae_enc_free(&ctx->vae_enc);
    }
    delete ctx;
}
