// ace-understand.cpp: audio understanding via ggml (reverse pipeline)
//
// Audio -> VAE encode -> FSQ tokenize -> LM understand -> metadata + lyrics
// Or:  audio_codes from JSON -> LM understand -> metadata + lyrics
//
// Output: request JSON with metadata + lyrics, reusable as ace-qwen3 input.
//
// Usage: ./ace-understand --src-audio <wav> --model <gguf> --dit <gguf> --vae <gguf>
//        ./ace-understand --request <json> --model <gguf>
// See --help for full option list.

#include "audio-io.h"
#include "bpe.h"
#include "fsq-tok.h"
#include "gguf-weights.h"
#include "metadata-fsm.h"
#include "prompt.h"
#include "qwen3-lm.h"
#include "request.h"
#include "timer.h"
#include "vae-enc.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

// Sampling: temperature -> top_k -> top_p -> softmax -> multinomial
// Same as ace-qwen3.cpp. Greedy when temperature <= 0.
static int sample_top_k_p(float * logits, int V, float temperature, float top_p, int top_k, std::mt19937 & rng) {
    if (temperature <= 0.0f) {
        return (int) (std::max_element(logits, logits + V) - logits);
    }

    struct TokenProb {
        int   id;
        float prob;
    };

    static thread_local std::vector<float>     tmp_buf;
    static thread_local std::vector<TokenProb> sorted_buf;
    static thread_local std::vector<float>     probs_buf;

    float inv_temp = 1.0f / temperature;
    for (int i = 0; i < V; i++) {
        logits[i] *= inv_temp;
    }

    // top_k pre-filter (or effective_k=256 when top_k=0)
    int effective_k = (top_k > 0) ? top_k : 256;
    if (effective_k < V) {
        tmp_buf.resize(V);
        memcpy(tmp_buf.data(), logits, V * sizeof(float));
        std::nth_element(tmp_buf.begin(), tmp_buf.begin() + (effective_k - 1), tmp_buf.end(), std::greater<float>());
        float threshold = tmp_buf[effective_k - 1];
        for (int i = 0; i < V; i++) {
            if (logits[i] < threshold) {
                logits[i] = -INFINITY;
            }
        }
    }

    // top_p nucleus filter
    if (top_p > 0.0f && top_p < 1.0f) {
        sorted_buf.clear();
        for (int i = 0; i < V; i++) {
            if (logits[i] > -1e30f) {
                sorted_buf.push_back({ i, logits[i] });
            }
        }
        int K = (int) sorted_buf.size();
        if (K > 0) {
            std::sort(sorted_buf.begin(), sorted_buf.end(),
                      [](const TokenProb & a, const TokenProb & b) { return a.prob > b.prob; });
            float max_val = sorted_buf[0].prob;
            float sum     = 0.0f;
            probs_buf.resize(K);
            for (int i = 0; i < K; i++) {
                probs_buf[i] = expf(sorted_buf[i].prob - max_val);
                sum += probs_buf[i];
            }
            float inv = 1.0f / sum;
            float cum = 0.0f;
            for (int i = 0; i < K; i++) {
                if (i > 0 && cum >= top_p) {
                    logits[sorted_buf[i].id] = -INFINITY;
                }
                cum += probs_buf[i] * inv;
            }
        }
    }

    // softmax -> multinomial
    float max_val = -INFINITY;
    for (int i = 0; i < V; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
        }
    }
    float sum = 0.0f;
    for (int i = 0; i < V; i++) {
        logits[i] = expf(logits[i] - max_val);
        sum += logits[i];
    }
    std::uniform_real_distribution<float> dist(0.0f, sum);
    float                                 r   = dist(rng);
    float                                 acc = 0.0f;
    for (int i = 0; i < V; i++) {
        acc += logits[i];
        if (acc >= r) {
            return i;
        }
    }
    return 0;
}

// BPE decode (token IDs -> text). Same as ace-qwen3.cpp.
// Skips audio code tokens, im_start/end. Expands think tags to text.
static std::string bpe_decode(const BPETokenizer & bpe, const std::vector<int> & ids) {
    static std::unordered_map<int, uint8_t> byte_dec;
    static bool                             init = false;
    if (!init) {
        for (int b = 0; b < 256; b++) {
            int adv;
            int cp       = utf8_codepoint(bpe.byte2str[b].c_str(), &adv);
            byte_dec[cp] = (uint8_t) b;
        }
        init = true;
    }

    std::string result;
    for (int id : ids) {
        if (id == TOKEN_THINK) {
            result += "<think>";
            continue;
        }
        if (id == TOKEN_THINK_END) {
            result += "</think>";
            continue;
        }
        if (id == TOKEN_IM_START || id == TOKEN_IM_END) {
            continue;
        }
        if (id >= AUDIO_CODE_BASE) {
            continue;
        }
        if (id < 0 || id >= (int) bpe.id_to_str.size()) {
            continue;
        }
        const std::string & s = bpe.id_to_str[id];
        if (s.empty()) {
            continue;
        }
        const char * p = s.c_str();
        while (*p) {
            int  adv;
            int  cp = utf8_codepoint(p, &adv);
            auto it = byte_dec.find(cp);
            if (it != byte_dec.end()) {
                result += (char) it->second;
            }
            p += adv;
        }
    }
    return result;
}

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

static void usage(const char * prog) {
    fprintf(stderr,
            "Usage: %s [--src-audio <file> --dit <gguf> --vae <gguf> | --request <json>] --model <gguf>\n"
            "\n"
            "Audio input (full pipeline):\n"
            "  --src-audio <file>      Source audio (WAV or MP3, any sample rate)\n"
            "  --dit <gguf>            DiT GGUF (for FSQ tokenizer weights + silence_latent)\n"
            "  --vae <gguf>            VAE GGUF (for audio encoding)\n"
            "\n"
            "Code input (skip VAE + tokenizer):\n"
            "  --request <json>        Request JSON with audio_codes field\n"
            "\n"
            "Required:\n"
            "  --model <gguf>          5Hz LM GGUF (same model as ace-qwen3)\n"
            "\n"
            "Output:\n"
            "  -o <json>               Output JSON (default: stdout summary)\n"
            "\n"
            "Sampling params (seed, lm_temperature, lm_top_p, lm_top_k) come from the\n"
            "request JSON. Without --request, understand defaults apply (temperature=0.3).\n"
            "\n"
            "VAE tiling:\n"
            "  --vae-chunk <N>         Latent frames per tile (default: 256)\n"
            "  --vae-overlap <N>       Overlap frames per side (default: 64)\n"
            "\n"
            "Debug:\n"
            "  --max-seq <N>           KV cache size (default: 8192)\n"
            "  --no-fsm                Disable FSM constrained decoding\n"
            "  --no-fa                 Disable flash attention\n",
            prog);
}

int main(int argc, char ** argv) {
    const char * src_audio_path = nullptr;
    const char * dit_gguf       = nullptr;
    const char * vae_gguf       = nullptr;
    const char * request_path   = nullptr;
    const char * model_path     = nullptr;
    const char * output_path    = nullptr;
    int          max_seq        = 8192;
    int          vae_chunk      = 256;
    int          vae_overlap    = 64;
    bool         use_fsm        = true;
    bool         use_fa         = true;

    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--src-audio") && i + 1 < argc) {
            src_audio_path = argv[++i];
        } else if (!strcmp(argv[i], "--dit") && i + 1 < argc) {
            dit_gguf = argv[++i];
        } else if (!strcmp(argv[i], "--vae") && i + 1 < argc) {
            vae_gguf = argv[++i];
        } else if (!strcmp(argv[i], "--request") && i + 1 < argc) {
            request_path = argv[++i];
        } else if (!strcmp(argv[i], "--model") && i + 1 < argc) {
            model_path = argv[++i];
        } else if (!strcmp(argv[i], "-o") && i + 1 < argc) {
            output_path = argv[++i];
        } else if (!strcmp(argv[i], "--max-seq") && i + 1 < argc) {
            max_seq = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--vae-chunk") && i + 1 < argc) {
            vae_chunk = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--vae-overlap") && i + 1 < argc) {
            vae_overlap = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--no-fsm")) {
            use_fsm = false;
        } else if (!strcmp(argv[i], "--no-fa")) {
            use_fa = false;
        } else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (!model_path) {
        fprintf(stderr, "[CLI] ERROR: --model required\n");
        usage(argv[0]);
        return 1;
    }
    if (!src_audio_path && !request_path) {
        fprintf(stderr, "[CLI] ERROR: --src-audio or --request required\n");
        usage(argv[0]);
        return 1;
    }
    if (src_audio_path && (!dit_gguf || !vae_gguf)) {
        fprintf(stderr, "[CLI] ERROR: --src-audio requires --dit and --vae\n");
        return 1;
    }

    // Parse request JSON (if provided). Sampling params come from JSON.
    // When no JSON, understand defaults apply (temperature=0.3 for transcription).
    AceRequest req;
    request_init(&req);
    req.lm_temperature = 0.3f;  // understand default: lower than generation
    if (request_path) {
        if (!request_parse(&req, request_path)) {
            return 1;
        }
    }

    // Resolve seed (same as ace-qwen3)
    long long seed = req.seed;
    if (seed < 0) {
        std::random_device rd;
        seed = (int64_t) rd() << 32 | rd();
        if (seed < 0) {
            seed = -seed;
        }
    }

    // Generation params from request
    float temperature = req.lm_temperature;
    float top_p       = req.lm_top_p;
    int   top_k       = req.lm_top_k;

    Timer            t_total;
    std::vector<int> codes;

    // Step 1: get audio codes
    // --src-audio: full pipeline (VAE encode + FSQ tokenize)
    // --request without --src-audio: parse audio_codes from JSON
    // --request + --src-audio: audio from file, params from JSON
    if (src_audio_path) {
        fprintf(stderr, "[Understand] Source: %s\n", src_audio_path);

        // Read and resample audio to 48kHz stereo
        int     T_audio = 0;
        float * planar  = audio_read_48k(src_audio_path, &T_audio);
        if (!planar) {
            fprintf(stderr, "[Audio] FATAL: cannot read %s\n", src_audio_path);
            return 1;
        }
        fprintf(stderr, "[Audio] %.2fs @ 48kHz\n", (float) T_audio / 48000.0f);

        // VAE expects interleaved [L0,R0,L1,R1,...], convert from planar
        float * wav_data = (float *) malloc((size_t) T_audio * 2 * sizeof(float));
        for (int t = 0; t < T_audio; t++) {
            wav_data[t * 2 + 0] = planar[t];
            wav_data[t * 2 + 1] = planar[T_audio + t];
        }
        free(planar);

        // VAE encode: audio -> latents [T_25Hz, 64]
        Timer      t_vae;
        VAEEncoder vae_enc = {};
        vae_enc_load(&vae_enc, vae_gguf);
        int                max_T_lat = (T_audio / 1920) + 64;
        std::vector<float> latents((size_t) max_T_lat * 64);

        int T_25Hz =
            vae_enc_encode_tiled(&vae_enc, wav_data, T_audio, latents.data(), max_T_lat, vae_chunk, vae_overlap);
        free(wav_data);
        if (T_25Hz < 0) {
            fprintf(stderr, "[VAE] FATAL: encode failed\n");
            vae_enc_free(&vae_enc);
            return 1;
        }
        vae_enc_free(&vae_enc);
        fprintf(stderr, "[VAE] Encoded: %d latent frames (%.2fs), %.0fms\n", T_25Hz,
                (float) T_25Hz * 1920.0f / 48000.0f, t_vae.ms());

        // Load silence_latent from DiT GGUF (needed for FSQ tokenizer padding)
        std::vector<float> silence(15000 * 64);
        {
            GGUFModel gf = {};
            if (!gf_load(&gf, dit_gguf)) {
                fprintf(stderr, "[DiT] FATAL: cannot open %s\n", dit_gguf);
                return 1;
            }
            const void * sl = gf_get_data(gf, "silence_latent");
            if (!sl) {
                fprintf(stderr, "[DiT] FATAL: silence_latent not found in %s\n", dit_gguf);
                gf_close(&gf);
                return 1;
            }
            memcpy(silence.data(), sl, 15000 * 64 * sizeof(float));
            gf_close(&gf);
        }

        // FSQ tokenize: latents [T_25Hz, 64] -> codes [T_5Hz]
        // Tokenizer weights live in the DiT GGUF (prefix "tokenizer.")
        Timer          t_tok;
        TokGGML        tok    = {};
        ggml_backend_dev_t dev_cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        ggml_backend_t be_tok = NULL;
        if (dev_cpu) {
            be_tok = ggml_backend_dev_init(dev_cpu, NULL);
        }
        if (!be_tok) {
            be_tok = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
        }
        if (!be_tok) {
            fprintf(stderr, "[Tok] FATAL: failed to init CPU backend\n");
            return 1;
        }
        if (!tok_ggml_load(&tok, dit_gguf, be_tok, be_tok)) {
            fprintf(stderr, "[Tok] FATAL: load failed\n");
            ggml_backend_free(be_tok);
            return 1;
        }

        int max_codes = (T_25Hz + 4) / 5;
        codes.resize(max_codes);
        int T_5Hz = tok_ggml_encode(&tok, latents.data(), T_25Hz, codes.data(), silence.data());
        tok_ggml_free(&tok);
        ggml_backend_free(be_tok);
        if (T_5Hz < 0) {
            fprintf(stderr, "[Tok] FATAL: tokenize failed\n");
            return 1;
        }
        codes.resize(T_5Hz);
        fprintf(stderr, "[Tok] %d codes (%.2fs @ 5Hz), %.0fms\n", T_5Hz, (float) T_5Hz / 5.0f, t_tok.ms());

    } else {
        // Codes from JSON: parse audio_codes string "3101,11837,..."
        if (req.audio_codes.empty()) {
            fprintf(stderr, "[Request] ERROR: audio_codes is empty in %s\n", request_path);
            return 1;
        }
        codes = parse_codes_string(req.audio_codes);
        fprintf(stderr, "[Request] %zu codes from %s\n", codes.size(), request_path);
    }

    if (codes.empty()) {
        fprintf(stderr, "[Understand] ERROR: no audio codes to process\n");
        return 1;
    }

    // Step 2: load BPE tokenizer + LM
    BPETokenizer bpe;
    if (!load_bpe_from_gguf(&bpe, model_path)) {
        return 1;
    }

    Timer   t_load;
    Qwen3LM model;
    if (!qw3lm_load(&model, model_path, max_seq, 1)) {
        return 1;
    }
    model.use_flash_attn = use_fa;
    double load_ms       = t_load.ms();
    fprintf(stderr, "[Load] LM: %.0fms\n", load_ms);

    int V = model.cfg.vocab_size;

    // FSM for constrained CoT metadata decoding
    MetadataFSM fsm;
    if (use_fsm) {
        fsm.init(bpe, V);
    }

    // Step 3: build understand prompt
    // System: understand instruction
    // User: raw audio code tokens (not BPE text)
    // The LM sees the codes and generates metadata + lyrics
    auto prompt = build_understand_prompt(bpe, codes.data(), (int) codes.size());
    fprintf(stderr, "[Prompt] %zu tokens (%zu codes + framing)\n", prompt.size(), codes.size());

    // Step 4: prefill
    Timer              t_gen;
    std::vector<float> logits(V);
    qw3lm_forward(&model, prompt.data(), (int) prompt.size(), 0, logits.data());
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
        qw3lm_forward(&model, &tok, 1, 0, logits.data());
    }

    fprintf(stderr, "[Decode] %zu tokens, %.0fms (%.1f tok/s)\n", gen_tokens.size(), t_gen.ms(),
            (float) gen_tokens.size() / (t_gen.ms() / 1000.0f));

    qw3lm_free(&model);

    // Step 6: decode tokens to text, parse CoT metadata + lyrics
    std::string text = bpe_decode(bpe, gen_tokens);

    AcePrompt parsed = {};
    parse_cot_and_lyrics(text, &parsed);

    fprintf(stderr, "\n[Result]\n");
    if (parsed.bpm > 0) {
        fprintf(stderr, "  bpm: %d\n", parsed.bpm);
    }
    if (parsed.duration > 0) {
        fprintf(stderr, "  duration: %.0fs\n", parsed.duration);
    }
    if (!parsed.keyscale.empty()) {
        fprintf(stderr, "  keyscale: %s\n", parsed.keyscale.c_str());
    }
    if (!parsed.timesignature.empty()) {
        fprintf(stderr, "  timesig: %s\n", parsed.timesignature.c_str());
    }
    if (!parsed.vocal_language.empty()) {
        fprintf(stderr, "  language: %s\n", parsed.vocal_language.c_str());
    }
    if (!parsed.caption.empty()) {
        fprintf(stderr, "  caption: %.80s%s\n", parsed.caption.c_str(), parsed.caption.size() > 80 ? "..." : "");
    }
    if (!parsed.lyrics.empty()) {
        fprintf(stderr, "  lyrics: %zu chars\n", parsed.lyrics.size());
    }

    // Step 7: write output JSON (reusable as dit-vae input with codes)
    // Build audio_codes string from recovered codes (comma-separated)
    std::string codes_str;
    for (size_t i = 0; i < codes.size(); i++) {
        if (i > 0) {
            codes_str += ',';
        }
        codes_str += std::to_string(codes[i]);
    }
    fprintf(stderr, "  audio_codes: %zu codes\n", codes.size());

    if (output_path) {
        AceRequest out;
        request_init(&out);
        out.caption        = parsed.caption;
        out.lyrics         = parsed.lyrics;
        out.bpm            = parsed.bpm;
        out.duration       = parsed.duration;
        out.keyscale       = parsed.keyscale;
        out.timesignature  = parsed.timesignature;
        out.vocal_language = parsed.vocal_language;
        out.audio_codes    = codes_str;
        request_write(&out, output_path);
    }

    fprintf(stderr, "\n[Understand] Load %.0f | Total %.0fms | seed=%lld\n", load_ms, t_total.ms(), seed);
    return 0;
}
