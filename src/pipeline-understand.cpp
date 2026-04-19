// pipeline-understand.cpp: ACE-Step reverse pipeline (audio -> metadata)
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
#include "model-store.h"
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
    ModelStore *        store;
    AceUnderstandParams params;

    // Feature flags derived from provided paths.
    bool have_audio;  // vae_path + dit_path present: full audio-in mode
    bool have_lm;     // model_path present: run the understand LM

    ModelKey lm_key;
    ModelKey vae_enc_key;
    ModelKey fsq_tok_key;
};

void ace_understand_default_params(AceUnderstandParams * p) {
    p->model_path  = NULL;
    p->dit_path    = NULL;
    p->vae_path    = NULL;
    p->dump_dir    = NULL;
    p->max_seq     = 8192;
    p->max_batch   = 1;
    p->use_fsm     = true;
    p->use_fa      = true;
    p->vae_chunk   = 256;
    p->vae_overlap = 64;
}

AceUnderstand * ace_understand_load(ModelStore * store, const AceUnderstandParams * params) {
    if (!store || !params) {
        fprintf(stderr, "[Understand-Load] ERROR: store and params are required\n");
        return NULL;
    }
    if (!params->model_path && !params->dump_dir) {
        fprintf(stderr, "[Understand-Load] ERROR: model_path required (or dump_dir for tok-only)\n");
        return NULL;
    }

    auto * ctx      = new AceUnderstand();
    ctx->store      = store;
    ctx->params     = *params;
    ctx->have_audio = params->vae_path && params->dit_path;
    ctx->have_lm    = params->model_path != NULL;

    if (ctx->have_audio) {
        ctx->vae_enc_key.kind = MODEL_VAE_ENC;
        ctx->vae_enc_key.path = params->vae_path;

        ctx->fsq_tok_key.kind = MODEL_FSQ_TOK;
        ctx->fsq_tok_key.path = params->dit_path;
    }

    if (ctx->have_lm) {
        // LM key MUST stay identical to the one ace_lm builds, so the store
        // returns the same Qwen3LM instance to both pipelines. Drift here
        // silently doubles VRAM under --keep-loaded. See VRAM policy doctrine
        // in model-store.h.
        ctx->lm_key.kind      = MODEL_LM;
        ctx->lm_key.path      = params->model_path;
        ctx->lm_key.max_seq   = params->max_seq;
        ctx->lm_key.n_kv_sets = 2 * params->max_batch;
    }

    fprintf(stderr, "[Understand-Load] Ready: audio=%s, lm=%s, fa=%s, fsm=%s\n", ctx->have_audio ? "yes" : "no",
            ctx->have_lm ? "yes" : "no", params->use_fa ? "yes" : "no", params->use_fsm ? "yes" : "no");
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

    // LM RNG seed: always random (mt19937 uses 32 bits)
    std::random_device rd;
    uint32_t           seed = rd();

    // Generation params from request
    float temperature = req->lm_temperature;
    float top_p       = req->lm_top_p;
    int   top_k       = req->lm_top_k;

    std::vector<int> codes;

    // Step 1: get audio codes
    // src_audio provided: full pipeline (VAE encode + FSQ tokenize)
    // no src_audio, audio_codes in request: parse from JSON
    // src_audio + request: audio from caller, params from JSON
    if (src_audio && src_len > 0) {
        if (!ctx->have_audio) {
            fprintf(stderr, "[Understand] ERROR: audio input requires VAE + DiT models\n");
            return -1;
        }

        // VAE encode: audio -> latents [T_25Hz, 64]. VAE-Enc lives only for
        // this step: acquire, encode, release so the store can free it before
        // the tokenizer comes in (STRICT) or keep it resident (NEVER).
        Timer              t_vae;
        int                max_T_lat = (src_len / 1920) + 64;
        std::vector<float> latents((size_t) max_T_lat * 64);

        VAEEncoder * vae_enc = store_require_vae_enc(ctx->store, ctx->vae_enc_key);
        if (!vae_enc) {
            fprintf(stderr, "[Understand-VAE] FATAL: store_require_vae_enc failed\n");
            return -1;
        }
        int T_25Hz;
        {
            ModelHandle vae_guard(ctx->store, vae_enc);
            T_25Hz = vae_enc_encode_tiled(vae_enc, src_audio, src_len, latents.data(), max_T_lat, ctx->params.vae_chunk,
                                          ctx->params.vae_overlap);
        }
        if (T_25Hz < 0) {
            fprintf(stderr, "[Understand-VAE] FATAL: encode failed\n");
            return -1;
        }
        fprintf(stderr, "[Understand-VAE] Encoded: %d latent frames (%.2fs), %.0fms\n", T_25Hz,
                (float) T_25Hz * 1920.0f / 48000.0f, t_vae.ms());

        // FSQ tokenize: latents [T_25Hz, 64] -> codes [T_5Hz].
        // silence comes from the store's CPU cache of the DiT GGUF.
        const float * silence = store_silence(ctx->store, ctx->params.dit_path);
        if (!silence) {
            fprintf(stderr, "[Understand-Tok] FATAL: silence_latent unavailable\n");
            return -1;
        }

        Timer t_tok;
        int   max_codes = (T_25Hz + 4) / 5;
        codes.resize(max_codes);
        TokGGML * fsq = store_require_fsq_tok(ctx->store, ctx->fsq_tok_key);
        if (!fsq) {
            fprintf(stderr, "[Understand-Tok] FATAL: store_require_fsq_tok failed\n");
            return -1;
        }
        int T_5Hz;
        {
            ModelHandle fsq_guard(ctx->store, fsq);
            T_5Hz = tok_ggml_encode(fsq, latents.data(), T_25Hz, codes.data(), silence);
        }
        if (T_5Hz < 0) {
            fprintf(stderr, "[Understand-Tok] FATAL: tokenize failed\n");
            return -1;
        }
        codes.resize(T_5Hz);
        fprintf(stderr, "[Understand-Tok] %d codes (%.2fs @ 5Hz), %.0fms\n", T_5Hz, (float) T_5Hz / 5.0f, t_tok.ms());

        // dump: save latents and codes for test-tok-cossim.py
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
                fprintf(stderr, "[Understand-Debug] tok_codes: [%d] int32\n", T_5Hz);
            }
        }
    } else {
        // Codes from JSON: parse audio_codes string "3101,11837,..."
        codes = parse_codes_string(req->audio_codes);
        if (codes.empty()) {
            fprintf(stderr, "[Understand] ERROR: no audio and no audio_codes\n");
            return -1;
        }
        fprintf(stderr, "[Understand] %zu codes from request\n", codes.size());
    }

    // dump-only mode (no LM loaded): return codes and exit
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

    // Step 2: acquire the LM for prefill + autoregressive decode.
    // BPE and FSM template come from the store's CPU-side accessors; the FSM
    // must be copied before mutation since the template is shared.
    Qwen3LM * model = store_require_lm(ctx->store, ctx->lm_key);
    if (!model) {
        fprintf(stderr, "[Understand] FATAL: store_require_lm failed\n");
        return -1;
    }
    ModelHandle lm_guard(ctx->store, model);

    if (!ctx->params.use_fa) {
        model->use_flash_attn = false;
    }
    // Master never set clamp_fp16 on the understand LM. Since the store now
    // shares the LM with pipeline-lm, we must reset it explicitly: without
    // this, ace-lm running with --clamp-fp16 would leak its clamp state into
    // the next understand call on the same process.
    model->clamp_fp16 = false;

    BPETokenizer * bpe = store_bpe(ctx->store, ctx->params.model_path);
    if (!bpe) {
        fprintf(stderr, "[Understand] FATAL: store_bpe failed\n");
        return -1;
    }

    MetadataFSM * fsm_template = nullptr;
    if (ctx->params.use_fsm) {
        fsm_template = store_fsm(ctx->store, ctx->params.model_path, model->cfg.vocab_size);
        if (!fsm_template) {
            fprintf(stderr, "[Understand] FATAL: store_fsm failed\n");
            return -1;
        }
    }

    int V = model->cfg.vocab_size;

    // Local mutable FSM for this call. A copy is mandatory: apply_mask and
    // update mutate state that must not bleed across requests.
    bool        use_fsm = ctx->params.use_fsm;
    MetadataFSM fsm;
    if (fsm_template) {
        fsm = *fsm_template;
    }

    // Step 3: build understand prompt
    // System: understand instruction
    // User: raw audio code tokens (not BPE text)
    // The LM sees the codes and generates metadata + lyrics
    std::vector<int> prompt = build_understand_prompt(*bpe, codes.data(), (int) codes.size());
    fprintf(stderr, "[Understand-Prompt] %zu tokens (%zu codes + framing)\n", prompt.size(), codes.size());

    // Step 4: prefill
    Timer              t_gen;
    std::vector<float> logits(V);
    qw3lm_reset_kv(model, 0);
    qw3lm_forward(model, prompt.data(), (int) prompt.size(), 0, logits.data());
    fprintf(stderr, "[Understand-Prefill] %.0fms, %zu tokens, seed=%u\n", t_gen.ms(), prompt.size(), seed);

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
        qw3lm_forward(model, &tok, 1, 0, logits.data());
    }

    fprintf(stderr, "[Understand-Decode] %zu tokens, %.0fms (%.1f tok/s)\n", gen_tokens.size(), t_gen.ms(),
            (float) gen_tokens.size() / (t_gen.ms() / 1000.0f));

    // Step 6: decode tokens to text, parse CoT metadata + lyrics
    std::string text   = bpe_decode(*bpe, gen_tokens);
    AcePrompt   parsed = {};
    parse_cot_and_lyrics(text, &parsed);

    if (parsed.bpm > 0) {
        fprintf(stderr, "[Understand-Result] bpm: %d\n", parsed.bpm);
    }
    if (parsed.duration > 0) {
        fprintf(stderr, "[Understand-Result] duration: %.0fs\n", parsed.duration);
    }
    if (!parsed.keyscale.empty()) {
        fprintf(stderr, "[Understand-Result] keyscale: %s\n", parsed.keyscale.c_str());
    }
    if (!parsed.timesignature.empty()) {
        fprintf(stderr, "[Understand-Result] timesig: %s\n", parsed.timesignature.c_str());
    }
    if (!parsed.vocal_language.empty()) {
        fprintf(stderr, "[Understand-Result] language: %s\n", parsed.vocal_language.c_str());
    }
    if (!parsed.caption.empty()) {
        fprintf(stderr, "[Understand-Result] caption: %.80s%s\n", parsed.caption.c_str(),
                parsed.caption.size() > 80 ? "..." : "");
    }
    if (!parsed.lyrics.empty()) {
        fprintf(stderr, "[Understand-Result] lyrics: %zu chars\n", parsed.lyrics.size());
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
    out->seed           = req->seed;

    // Build audio_codes string from recovered codes (comma-separated)
    std::string codes_str;
    for (size_t i = 0; i < codes.size(); i++) {
        if (i > 0) {
            codes_str += ',';
        }
        codes_str += std::to_string(codes[i]);
    }
    out->audio_codes = codes_str;
    fprintf(stderr, "[Understand-Result] audio_codes: %zu codes\n", codes.size());

    // DiT sampling params: 0 = auto-detect from the DiT model at synth time.
    // understand does not know which DiT the user will synthesize with.
    out->inference_steps = 0;
    out->shift           = 0.0f;
    out->guidance_scale  = 0.0f;

    fprintf(stderr, "[Understand] Total %.0fms | seed=%u\n", t_total.ms(), seed);
    return 0;
}

void ace_understand_free(AceUnderstand * ctx) {
    if (!ctx) {
        return;
    }
    delete ctx;
}

const ModelKey * ace_understand_lm_key(const AceUnderstand * ctx) {
    return ctx && ctx->have_lm ? &ctx->lm_key : nullptr;
}
