// dit.cpp: ACEStep music generation via ggml (dit-vae binary)
//
// Usage: ./dit-vae [options]
// See --help for full option list.

#include "bpe.h"
#include "cond-enc.h"
#include "debug.h"
#include "dit-sampler.h"
#include "fsq-detok.h"
#include "philox.h"
#include "qwen3-enc.h"
#include "request.h"
#include "timer.h"
#include "vae-enc.h"
#include "vae.h"
#include "wav.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static void print_usage(const char * prog) {
    fprintf(stderr,
            "Usage: %s --request <json...> --text-encoder <gguf> --dit <gguf> --vae <gguf> [options]\n\n"
            "Required:\n"
            "  --request <json...>     One or more request JSONs (from ace-qwen3 --request)\n"
            "  --text-encoder <gguf>   Text encoder GGUF file\n"
            "  --dit <gguf>            DiT GGUF file\n"
            "  --vae <gguf>            VAE GGUF file\n\n"
            "Reference audio:\n"
            "  --src-audio <wav>       Source audio (48kHz stereo WAV)\n\n"
            "Batch:\n"
            "  --batch <N>             DiT variations per request (default: 1, max 9)\n\n"
            "Output naming: input.json -> input0.wav, input1.wav, ... (last digit = batch index)\n\n"
            "VAE tiling (memory control):\n"
            "  --vae-chunk <N>         Latent frames per tile (default: 256)\n"
            "  --vae-overlap <N>       Overlap frames per side (default: 64)\n\n"
            "Debug:\n"
            "  --no-fa                 Disable flash attention\n"
            "  --dump <dir>            Dump intermediate tensors\n",
            prog);
}

// Parse comma-separated codes string into vector
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

int main(int argc, char ** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::vector<const char *> request_paths;
    const char *              text_enc_gguf  = NULL;
    const char *              dit_gguf       = NULL;
    const char *              vae_gguf       = NULL;
    const char *              src_audio_path = NULL;
    const char *              dump_dir       = NULL;
    bool                      use_fa         = true;
    int                       batch_n        = 1;
    int                       vae_chunk      = 256;
    int                       vae_overlap    = 64;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--request") == 0) {
            // Collect all following non-option args
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                request_paths.push_back(argv[++i]);
            }
        } else if (strcmp(argv[i], "--text-encoder") == 0 && i + 1 < argc) {
            text_enc_gguf = argv[++i];
        } else if (strcmp(argv[i], "--dit") == 0 && i + 1 < argc) {
            dit_gguf = argv[++i];
        } else if (strcmp(argv[i], "--vae") == 0 && i + 1 < argc) {
            vae_gguf = argv[++i];
        } else if (strcmp(argv[i], "--src-audio") == 0 && i + 1 < argc) {
            src_audio_path = argv[++i];
        } else if (strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
            dump_dir = argv[++i];
        } else if (strcmp(argv[i], "--no-fa") == 0) {
            use_fa = false;
        } else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
            batch_n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--vae-chunk") == 0 && i + 1 < argc) {
            vae_chunk = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--vae-overlap") == 0 && i + 1 < argc) {
            vae_overlap = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (request_paths.empty()) {
        fprintf(stderr, "ERROR: --request required\n");
        print_usage(argv[0]);
        return 1;
    }
    if (batch_n < 1 || batch_n > 9) {
        fprintf(stderr, "ERROR: --batch must be 1..9\n");
        return 1;
    }
    if (!dit_gguf) {
        fprintf(stderr, "ERROR: --dit required\n");
        print_usage(argv[0]);
        return 1;
    }
    if (!text_enc_gguf) {
        fprintf(stderr, "ERROR: --text-encoder required\n");
        print_usage(argv[0]);
        return 1;
    }

    const int FRAMES_PER_SECOND = 25;

    DebugDumper dbg;
    debug_init(&dbg, dump_dir);

    Timer         timer;
    DiTGGMLConfig cfg;
    DiTGGML       model = {};

    // Load DiT model (once for all requests)
    dit_ggml_init_backend(&model);
    if (!use_fa) {
        model.use_flash_attn = false;
    }
    fprintf(stderr, "[Load] Backend init: %.1f ms\n", timer.ms());

    timer.reset();
    if (!dit_ggml_load(&model, dit_gguf, cfg)) {
        fprintf(stderr, "FATAL: failed to load DiT model\n");
        return 1;
    }
    fprintf(stderr, "[Load] DiT weight load: %.1f ms\n", timer.ms());

    // Read DiT GGUF metadata + silence_latent tensor (once)
    bool               is_turbo = false;
    std::vector<float> silence_full;  // [15000, 64] f32
    {
        GGUFModel gf = {};
        if (gf_load(&gf, dit_gguf)) {
            is_turbo             = gf_get_bool(gf, "acestep.is_turbo");
            const void * sl_data = gf_get_data(gf, "silence_latent");
            if (sl_data) {
                silence_full.resize(15000 * 64);
                memcpy(silence_full.data(), sl_data, 15000 * 64 * sizeof(float));
                fprintf(stderr, "[Load] silence_latent: [15000, 64] from GGUF\n");
            } else {
                fprintf(stderr, "FATAL: silence_latent tensor not found in %s\n", dit_gguf);
                gf_close(&gf);
                dit_ggml_free(&model);
                return 1;
            }
            gf_close(&gf);
        } else {
            fprintf(stderr, "FATAL: cannot reopen %s for metadata\n", dit_gguf);
            dit_ggml_free(&model);
            return 1;
        }
    }

    int Oc     = cfg.out_channels;      // 64
    int ctx_ch = cfg.in_channels - Oc;  // 128

    // Load VAE model (once for all requests)
    VAEGGML vae      = {};
    bool    have_vae = false;
    if (vae_gguf) {
        timer.reset();
        vae_ggml_load(&vae, vae_gguf);
        fprintf(stderr, "[Load] VAE weights: %.1f ms\n", timer.ms());
        have_vae = true;
    }

    // Cover mode: load VAE encoder and encode source audio
    VAEEncoder         vae_enc    = {};
    bool               have_cover = false;
    std::vector<float> cover_latents;  // [T_cover, 64] time-major
    int                T_cover = 0;
    if (src_audio_path) {
        if (!vae_gguf) {
            fprintf(stderr, "ERROR: --src-audio requires --vae\n");
            return 1;
        }
        timer.reset();
        int     T_audio = 0, wav_sr = 0;
        float * wav_data = read_wav(src_audio_path, &T_audio, &wav_sr);
        if (!wav_data) {
            fprintf(stderr, "FATAL: cannot read --src-audio %s\n", src_audio_path);
            return 1;
        }
        if (wav_sr != 48000) {
            fprintf(stderr, "[WARN] src_audio is %d Hz, VAE expects 48000. Resample with ffmpeg first.\n", wav_sr);
        }
        fprintf(stderr, "[Cover] Source audio: %.2fs\n", (float) T_audio / (float) (wav_sr > 0 ? wav_sr : 48000));

        vae_enc_load(&vae_enc, vae_gguf);
        int max_T_lat = (T_audio / 1920) + 64;
        cover_latents.resize(max_T_lat * 64);

        T_cover =
            vae_enc_encode_tiled(&vae_enc, wav_data, T_audio, cover_latents.data(), max_T_lat, vae_chunk, vae_overlap);
        free(wav_data);
        if (T_cover < 0) {
            fprintf(stderr, "FATAL: VAE encode of src_audio failed\n");
            vae_enc_free(&vae_enc);
            return 1;
        }
        cover_latents.resize(T_cover * 64);
        fprintf(stderr, "[Cover] Encoded: T_cover=%d (%.2fs), %.1f ms\n", T_cover, (float) T_cover * 1920.0f / 48000.0f,
                timer.ms());
        have_cover = true;
    }

    // Process each request
    for (int ri = 0; ri < (int) request_paths.size(); ri++) {
        const char * rpath = request_paths[ri];
        fprintf(stderr, "[Request %d/%d] %s (batch=%d)\n", ri + 1, (int) request_paths.size(), rpath, batch_n);

        // Compute output basename: strip .json suffix
        std::string basename(rpath);
        {
            size_t dot = basename.rfind(".json");
            if (dot != std::string::npos) {
                basename = basename.substr(0, dot);
            }
        }

        // Parse request JSON
        AceRequest req;
        request_init(&req);
        if (!request_parse(&req, rpath)) {
            fprintf(stderr, "ERROR: failed to parse %s, skipping\n", rpath);
            continue;
        }
        if (req.caption.empty()) {
            fprintf(stderr, "ERROR: caption is empty in %s, skipping\n", rpath);
            continue;
        }

        // Extract params
        const char * caption     = req.caption.c_str();
        const char * lyrics      = req.lyrics.c_str();
        char         bpm_str[16] = "N/A";
        if (req.bpm > 0) {
            snprintf(bpm_str, sizeof(bpm_str), "%d", req.bpm);
        }
        const char * bpm            = bpm_str;
        const char * keyscale       = req.keyscale.empty() ? "N/A" : req.keyscale.c_str();
        const char * timesig        = req.timesignature.empty() ? "N/A" : req.timesignature.c_str();
        const char * language       = req.vocal_language.empty() ? "unknown" : req.vocal_language.c_str();
        float        duration       = req.duration > 0 ? req.duration : 30.0f;
        long long    seed           = req.seed;
        int          num_steps      = req.inference_steps > 0 ? req.inference_steps : 8;
        float        guidance_scale = req.guidance_scale;
        float        shift          = req.shift > 0 ? req.shift : 1.0f;

        if (guidance_scale <= 0.0f) {
            guidance_scale = is_turbo ? 1.0f : 7.0f;
        } else if (is_turbo && guidance_scale > 1.0f) {
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
        std::vector<int> codes_vec = parse_codes_string(req.audio_codes);
        if (!codes_vec.empty()) {
            fprintf(stderr, "[Pipeline] %zu audio codes (%.1fs @ 5Hz)\n", codes_vec.size(),
                    (float) codes_vec.size() / 5.0f);
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
        T         = ((T + cfg.patch_size - 1) / cfg.patch_size) * cfg.patch_size;
        int S     = T / cfg.patch_size;
        int enc_S = 0;

        fprintf(stderr, "[Pipeline] T=%d, S=%d\n", T, S);
        fprintf(stderr, "[Pipeline] seed=%lld, steps=%d, guidance=%.1f, shift=%.1f, duration=%.1fs\n", seed, num_steps,
                guidance_scale, shift, duration);

        if (T > 15000) {
            fprintf(stderr, "ERROR: T=%d exceeds silence_latent max 15000, skipping\n", T);
            continue;
        }

        // Text encoding
        // 1. Load BPE tokenizer
        timer.reset();
        BPETokenizer tok;
        if (!load_bpe_from_gguf(&tok, text_enc_gguf)) {
            fprintf(stderr, "FATAL: failed to load tokenizer from %s\n", text_enc_gguf);
            dit_ggml_free(&model);
            if (have_vae) {
                vae_ggml_free(&vae);
            }
            return 1;
        }
        fprintf(stderr, "[Load] BPE tokenizer: %.1f ms\n", timer.ms());

        // 2. Build formatted prompts
        // Reference project uses opposite-sounding instructions (constants.py):
        //   text2music = "Fill the audio semantic mask..."
        //   cover      = "Generate audio semantic tokens..."
        // Auto-switches to cover when audio_codes are present
        bool         is_cover    = have_cover || !codes_vec.empty();
        const char * instruction = is_cover ? "Generate audio semantic tokens based on the given conditions:" :
                                              "Fill the audio semantic mask based on the given conditions:";
        char         metas[512];
        snprintf(metas, sizeof(metas), "- bpm: %s\n- timesignature: %s\n- keyscale: %s\n- duration: %d seconds\n", bpm,
                 timesig, keyscale, (int) duration);
        std::string text_str = std::string("# Instruction\n") + instruction + "\n\n" + "# Caption\n" + caption +
                               "\n\n" + "# Metas\n" + metas + "<|endoftext|>\n";

        std::string lyric_str = std::string("# Languages\n") + language + "\n\n# Lyric\n" + lyrics + "<|endoftext|>";

        // 3. Tokenize
        auto text_ids  = bpe_encode(&tok, text_str.c_str(), true);
        auto lyric_ids = bpe_encode(&tok, lyric_str.c_str(), true);
        int  S_text    = (int) text_ids.size();
        int  S_lyric   = (int) lyric_ids.size();
        fprintf(stderr, "[Pipeline] caption: %d tokens, lyrics: %d tokens\n", S_text, S_lyric);

        // 4. Text encoder forward (caption only)
        timer.reset();
        Qwen3GGML text_enc = {};
        qwen3_init_backend(&text_enc);
        if (!use_fa) {
            text_enc.use_flash_attn = false;
        }
        if (!qwen3_load_text_encoder(&text_enc, text_enc_gguf)) {
            fprintf(stderr, "FATAL: failed to load text encoder\n");
            dit_ggml_free(&model);
            if (have_vae) {
                vae_ggml_free(&vae);
            }
            return 1;
        }
        fprintf(stderr, "[Load] TextEncoder: %.1f ms\n", timer.ms());

        int                H_text = text_enc.cfg.hidden_size;  // 1024
        std::vector<float> text_hidden(H_text * S_text);

        timer.reset();
        qwen3_forward(&text_enc, text_ids.data(), S_text, text_hidden.data());
        fprintf(stderr, "[Encode] TextEncoder (%d tokens): %.1f ms\n", S_text, timer.ms());
        debug_dump_2d(&dbg, "text_hidden", text_hidden.data(), S_text, H_text);

        // 5. Lyric embedding (vocab lookup via text encoder)
        timer.reset();
        std::vector<float> lyric_embed(H_text * S_lyric);
        qwen3_embed_lookup(&text_enc, lyric_ids.data(), S_lyric, lyric_embed.data());
        fprintf(stderr, "[Encode] Lyric vocab lookup (%d tokens): %.1f ms\n", S_lyric, timer.ms());
        debug_dump_2d(&dbg, "lyric_embed", lyric_embed.data(), S_lyric, H_text);

        // 6. Condition encoder forward
        timer.reset();
        CondGGML cond = {};
        cond_ggml_init_backend(&cond);
        if (!use_fa) {
            cond.use_flash_attn = false;
        }
        if (!cond_ggml_load(&cond, dit_gguf)) {
            fprintf(stderr, "FATAL: failed to load condition encoder\n");
            dit_ggml_free(&model);
            if (have_vae) {
                vae_ggml_free(&vae);
            }
            return 1;
        }
        fprintf(stderr, "[Load] ConditionEncoder: %.1f ms\n", timer.ms());

        // Silence feats for timbre input: first 750 frames (30s @ 25Hz)
        const int          S_ref = 750;
        std::vector<float> silence_feats(S_ref * 64);
        memcpy(silence_feats.data(), silence_full.data(), S_ref * 64 * sizeof(float));

        timer.reset();
        std::vector<float> enc_hidden;
        cond_ggml_forward(&cond, text_hidden.data(), S_text, lyric_embed.data(), S_lyric, silence_feats.data(), S_ref,
                          enc_hidden, &enc_S);
        fprintf(stderr, "[Encode] ConditionEncoder: %.1f ms, enc_S=%d\n", timer.ms(), enc_S);

        qwen3_free(&text_enc);
        cond_ggml_free(&cond);

        debug_dump_2d(&dbg, "enc_hidden", enc_hidden.data(), enc_S, 2048);

        // Decode audio codes if provided (passthrough mode only, NOT cover)
        int                decoded_T = 0;
        std::vector<float> decoded_latents;
        if (!have_cover && !codes_vec.empty()) {
            timer.reset();
            DetokGGML detok = {};
            if (!detok_ggml_load(&detok, dit_gguf, model.backend, model.cpu_backend)) {
                fprintf(stderr, "FATAL: failed to load detokenizer\n");
                dit_ggml_free(&model);
                if (have_vae) {
                    vae_ggml_free(&vae);
                }
                return 1;
            }
            if (!use_fa) {
                detok.use_flash_attn = false;
            }
            fprintf(stderr, "[Load] Detokenizer: %.1f ms\n", timer.ms());

            int T_5Hz        = (int) codes_vec.size();
            int T_25Hz_codes = T_5Hz * 5;
            decoded_latents.resize(T_25Hz_codes * Oc);

            timer.reset();
            int ret = detok_ggml_decode(&detok, codes_vec.data(), T_5Hz, decoded_latents.data());
            if (ret < 0) {
                fprintf(stderr, "FATAL: detokenizer decode failed\n");
                dit_ggml_free(&model);
                if (have_vae) {
                    vae_ggml_free(&vae);
                }
                return 1;
            }
            fprintf(stderr, "[Context] Detokenizer: %.1f ms\n", timer.ms());

            decoded_T = T_25Hz_codes < T ? T_25Hz_codes : T;
            debug_dump_2d(&dbg, "detok_output", decoded_latents.data(), T_25Hz_codes, Oc);
            detok_ggml_free(&detok);
        }

        // Build context: [T, ctx_ch] = src_latents[64] + mask_ones[64]
        // Cover: VAE latents directly (matching Python: is_covers=False, raw latents as context)
        // Passthrough: detokenized FSQ codes + silence padding
        // Text2music: silence only
        std::vector<float> context_single(T * ctx_ch);
        if (have_cover) {
            for (int t = 0; t < T; t++) {
                const float * src = (t < T_cover) ? cover_latents.data() + t * Oc : silence_full.data() + t * Oc;
                for (int c = 0; c < Oc; c++) {
                    context_single[t * ctx_ch + c] = src[c];
                }
                for (int c = 0; c < Oc; c++) {
                    context_single[t * ctx_ch + Oc + c] = 1.0f;
                }
            }
        } else {
            for (int t = 0; t < T; t++) {
                const float * src =
                    (t < decoded_T) ? decoded_latents.data() + t * Oc : silence_full.data() + (t - decoded_T) * Oc;
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
        std::vector<float> context_silence;
        int                cover_steps = -1;
        if (have_cover) {
            float cover_strength = req.audio_cover_strength;
            if (cover_strength < 1.0f) {
                // Build silence context: all frames use silence_latent
                std::vector<float> silence_single(T * ctx_ch);
                for (int t = 0; t < T; t++) {
                    const float * src = silence_full.data() + t * Oc;
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
                fprintf(stderr, "[Cover] audio_cover_strength=%.2f -> switch at step %d/%d\n", cover_strength,
                        cover_steps, num_steps);
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
        dit_ggml_generate(&model, noise.data(), context.data(), enc_hidden.data(), enc_S, T, batch_n, num_steps,
                          schedule.data(), output.data(), guidance_scale, &dbg,
                          context_silence.empty() ? nullptr : context_silence.data(), cover_steps);
        fprintf(stderr, "[DiT] Total generation: %.1f ms (%.1f ms/sample)\n", timer.ms(), timer.ms() / batch_n);

        debug_dump_2d(&dbg, "dit_output", output.data(), T, Oc);

        // VAE Decode + Write WAVs
        if (have_vae) {
            int                T_latent    = T;
            int                T_audio_max = T_latent * 1920;
            std::vector<float> audio(2 * T_audio_max);

            for (int b = 0; b < batch_n; b++) {
                float * dit_out = output.data() + b * Oc * T;

                timer.reset();
                int T_audio =
                    vae_ggml_decode_tiled(&vae, dit_out, T_latent, audio.data(), T_audio_max, vae_chunk, vae_overlap);
                if (T_audio < 0) {
                    fprintf(stderr, "[VAE Batch%d] ERROR: decode failed\n", b);
                    continue;
                }
                fprintf(stderr, "[VAE Batch%d] Decode: %.1f ms\n", b, timer.ms());

                // Peak normalization to -1.0 dB
                {
                    float peak      = 0.0f;
                    int   n_samples = 2 * T_audio;
                    for (int i = 0; i < n_samples; i++) {
                        float a = audio[i] < 0 ? -audio[i] : audio[i];
                        if (a > peak) {
                            peak = a;
                        }
                    }
                    if (peak > 1e-6f) {
                        const float target_amp = powf(10.0f, -1.0f / 20.0f);
                        float       gain       = target_amp / peak;
                        for (int i = 0; i < n_samples; i++) {
                            audio[i] *= gain;
                        }
                    }
                }

                // Write WAV: basename + batch_index + .wav
                char wav_path[1024];
                snprintf(wav_path, sizeof(wav_path), "%s%d.wav", basename.c_str(), b);

                if (b == 0) {
                    debug_dump_2d(&dbg, "vae_audio", audio.data(), 2, T_audio);
                }

                if (write_wav(wav_path, audio.data(), T_audio, 48000)) {
                    fprintf(stderr, "[VAE Batch%d] Wrote %s: %d samples (%.2fs @ 48kHz stereo)\n", b, wav_path, T_audio,
                            (float) T_audio / 48000.0f);
                } else {
                    fprintf(stderr, "[VAE Batch%d] FATAL: failed to write %s\n", b, wav_path);
                }
            }
        }

        fprintf(stderr, "[Request %d/%d] Done\n", ri + 1, (int) request_paths.size());
    }

    if (have_cover) {
        vae_enc_free(&vae_enc);
    }
    if (have_vae) {
        vae_ggml_free(&vae);
    }
    dit_ggml_free(&model);
    fprintf(stderr, "[Pipeline] All done\n");
    return 0;
}
