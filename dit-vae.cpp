// dit.cpp: ACEStep music generation via ggml (dit-vae binary)
//
// Usage: ./dit-vae [options]
// See --help for full option list.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <random>

#include "ggml.h"
#include "ggml-backend.h"
#include "dit.h"
#include "vae.h"
#include "qwen3.h"
#include "tokenizer.h"
#include "cond.h"
#include "bpe.h"
#include "debug.h"
#include "request.h"

struct Timer {
    std::chrono::steady_clock::time_point t;
    Timer() : t(std::chrono::steady_clock::now()) {}
    double ms() const {
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t).count();
    }
    void reset() { t = std::chrono::steady_clock::now(); }
};

// Minimal WAV writer (16-bit PCM stereo)
static bool write_wav(const char * path, const float * audio, int T_audio, int sr) {
    FILE * f = fopen(path, "wb");
    if (!f) return false;
    int n_channels = 2;
    int bits = 16;
    int byte_rate = sr * n_channels * (bits / 8);
    int block_align = n_channels * (bits / 8);
    int data_size = T_audio * n_channels * (bits / 8);
    int file_size = 36 + data_size;
    fwrite("RIFF", 1, 4, f);
    fwrite(&file_size, 4, 1, f);
    fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f);
    int fmt_size = 16; fwrite(&fmt_size, 4, 1, f);
    short audio_fmt = 1; fwrite(&audio_fmt, 2, 1, f);
    short nc = (short)n_channels; fwrite(&nc, 2, 1, f);
    fwrite(&sr, 4, 1, f);
    fwrite(&byte_rate, 4, 1, f);
    short ba = (short)block_align; fwrite(&ba, 2, 1, f);
    short bp = (short)bits; fwrite(&bp, 2, 1, f);
    fwrite("data", 1, 4, f);
    fwrite(&data_size, 4, 1, f);
    for (int t = 0; t < T_audio; t++) {
        for (int c = 0; c < 2; c++) {
            float s = audio[c * T_audio + t];
            s = s < -1.0f ? -1.0f : (s > 1.0f ? 1.0f : s);
            short v = (short)(s * 32767.0f);
            fwrite(&v, 2, 1, f);
        }
    }
    fclose(f);
    return true;
}

static void print_usage(const char * prog) {
    fprintf(stderr,
        "Usage: %s --request <json...> --text-encoder <gguf> --dit <gguf> --vae <gguf> [options]\n\n"
        "Required:\n"
        "  --request <json...>     One or more request JSONs (from ace-qwen3 --request)\n"
        "  --text-encoder <gguf>   Text encoder GGUF file\n"
        "  --dit <gguf>            DiT GGUF file (from convert.py)\n"
        "  --vae <gguf>            VAE GGUF file\n\n"
        "Batch:\n"
        "  --batch <n>             DiT variations per request (default: 1, max 9)\n\n"
        "Audio:\n"
        "  --noise-file <path>     Load noise from bf16 file (Philox RNG dump, batch=1 only)\n\n"
        "VAE tiling (memory control):\n"
        "  --vae-chunk <n>         Latent frames per tile (default: 256)\n"
        "  --vae-overlap <n>       Overlap frames per side (default: 64)\n\n"
        "Output naming: input.json -> input0.wav, input1.wav, ... (last digit = batch index)\n\n"
        "Debug:\n"
        "  --dump <dir>            Dump intermediate tensors\n", prog);
}

// Parse comma-separated codes string into vector
static std::vector<int> parse_codes_string(const std::string & s) {
    std::vector<int> codes;
    if (s.empty()) return codes;
    const char * p = s.c_str();
    while (*p) {
        while (*p == ',' || *p == ' ') p++;
        if (!*p) break;
        codes.push_back(atoi(p));
        while (*p && *p != ',') p++;
    }
    return codes;
}

int main(int argc, char ** argv) {
    if (argc < 2) { print_usage(argv[0]); return 1; }

    std::vector<const char *> request_paths;
    const char * text_enc_gguf = NULL;
    const char * dit_gguf      = NULL;
    const char * vae_gguf       = NULL;
    const char * dump_dir      = NULL;
    const char * noise_file    = NULL;
    int batch_n                = 1;
    int vae_chunk              = 256;
    int vae_overlap            = 64;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--request") == 0) {
            // Collect all following non-option args
            while (i+1 < argc && argv[i+1][0] != '-')
                request_paths.push_back(argv[++i]);
        }
        else if (strcmp(argv[i], "--text-encoder") == 0 && i+1 < argc) text_enc_gguf = argv[++i];
        else if (strcmp(argv[i], "--dit") == 0 && i+1 < argc) dit_gguf = argv[++i];
        else if (strcmp(argv[i], "--vae") == 0 && i+1 < argc) vae_gguf = argv[++i];
        else if (strcmp(argv[i], "--noise-file") == 0 && i+1 < argc) noise_file = argv[++i];
        else if (strcmp(argv[i], "--dump") == 0 && i+1 < argc) dump_dir = argv[++i];
        else if (strcmp(argv[i], "--batch") == 0 && i+1 < argc) batch_n = atoi(argv[++i]);
        else if (strcmp(argv[i], "--vae-chunk") == 0 && i+1 < argc) vae_chunk = atoi(argv[++i]);
        else if (strcmp(argv[i], "--vae-overlap") == 0 && i+1 < argc) vae_overlap = atoi(argv[++i]);
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]); return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]); return 1;
        }
    }

    if (request_paths.empty()) {
        fprintf(stderr, "ERROR: --request required\n");
        print_usage(argv[0]); return 1;
    }
    if (batch_n < 1 || batch_n > 9) {
        fprintf(stderr, "ERROR: --batch must be 1..9\n"); return 1;
    }
    if (!dit_gguf) {
        fprintf(stderr, "ERROR: --dit required\n");
        print_usage(argv[0]); return 1;
    }
    if (!text_enc_gguf) {
        fprintf(stderr, "ERROR: --text-encoder required\n");
        print_usage(argv[0]); return 1;
    }


    const int FRAMES_PER_SECOND = 25;

    DebugDumper dbg;
    debug_init(&dbg, dump_dir);

    Timer timer;
    DiTGGMLConfig cfg;
    DiTGGML model = {};

    // Load DiT model (once for all requests)
    dit_ggml_init_backend(&model);
    fprintf(stderr, "[Load] Backend init: %.1f ms\n", timer.ms());

    timer.reset();
    if (!dit_ggml_load(&model, dit_gguf, cfg)) {
        fprintf(stderr, "FATAL: failed to load DiT model\n");
        return 1;
    }
    fprintf(stderr, "[Load] DiT weight load: %.1f ms\n", timer.ms());

    // Read DiT GGUF metadata + silence_latent tensor (once)
    bool is_turbo = false;
    std::vector<float> silence_full;  // [15000, 64] f32
    {
        GGUFModel gf = {};
        if (gf_load(&gf, dit_gguf)) {
            is_turbo = gf_get_bool(gf, "acestep.is_turbo");
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

    int Oc = cfg.out_channels;          // 64
    int ctx_ch = cfg.in_channels - Oc;  // 128

    // Load VAE model (once for all requests)
    VAEGGML vae = {};
    bool have_vae = false;
    if (vae_gguf) {
        timer.reset();
        vae_ggml_load(&vae, vae_gguf);
        fprintf(stderr, "[Load] VAE weights: %.1f ms\n", timer.ms());
        have_vae = true;
    }

    // Process each request
    for (int ri = 0; ri < (int)request_paths.size(); ri++) {
        const char * rpath = request_paths[ri];
        fprintf(stderr, "[Request %d/%d] %s (batch=%d)\n",
                ri + 1, (int)request_paths.size(), rpath, batch_n);

        // Compute output basename: strip .json suffix
        std::string basename(rpath);
        {
            size_t dot = basename.rfind(".json");
            if (dot != std::string::npos)
                basename = basename.substr(0, dot);
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
        const char * caption  = req.caption.c_str();
        const char * lyrics   = req.lyrics.empty() ? "[Instrumental]" : req.lyrics.c_str();
        char bpm_str[16] = "N/A";
        if (req.bpm > 0) snprintf(bpm_str, sizeof(bpm_str), "%d", req.bpm);
        const char * bpm      = bpm_str;
        const char * keyscale = req.keyscale.empty() ? "N/A" : req.keyscale.c_str();
        const char * timesig  = req.timesignature.empty() ? "N/A" : req.timesignature.c_str();
        const char * language = req.vocal_language.empty() ? "en" : req.vocal_language.c_str();
        float duration        = req.duration > 0 ? req.duration : 120.0f;
        int seed              = req.seed;
        int num_steps         = req.inference_steps > 0 ? req.inference_steps : 8;
        float guidance_scale  = req.guidance_scale > 0 ? req.guidance_scale : 7.0f;
        float shift           = req.shift > 0 ? req.shift : 1.0f;

        if (is_turbo && guidance_scale > 1.0f) {
            fprintf(stderr, "[Pipeline] WARNING: turbo model, forcing guidance_scale=1.0 (was %.1f)\n",
                    guidance_scale);
            guidance_scale = 1.0f;
        }

        if (seed < 0) {
            std::random_device rd;
            seed = (int)(rd() & 0x7FFFFFFF);
        }
        fprintf(stderr, "[Pipeline] seed=%d, steps=%d, guidance=%.1f, shift=%.1f, duration=%.1fs\n",
                seed, num_steps, guidance_scale, shift, duration);

        // Parse audio codes from request
        std::vector<int> codes_vec = parse_codes_string(req.audio_codes);
        if (!codes_vec.empty())
            fprintf(stderr, "[Pipeline] %zu audio codes (%.1fs @ 5Hz)\n",
                    codes_vec.size(), (float)codes_vec.size() / 5.0f);

        // Build schedule: t_i = shift * t / (1 + (shift-1)*t) where t = 1 - i/steps
        std::vector<float> schedule(num_steps);
        for (int i = 0; i < num_steps; i++) {
            float t = 1.0f - (float)i / (float)num_steps;
            schedule[i] = shift * t / (1.0f + (shift - 1.0f) * t);
        }

        int T = (int)(duration * FRAMES_PER_SECOND);
        T = ((T + cfg.patch_size - 1) / cfg.patch_size) * cfg.patch_size;
        int S = T / cfg.patch_size;
        int enc_S = 0;

        fprintf(stderr, "[Pipeline] T=%d, S=%d\n", T, S);

        if (T > 15000) {
            fprintf(stderr, "ERROR: T=%d exceeds silence_latent max 15000, skipping\n", T);
            continue;
        }

        // --- Text encoding ---
        // 1. Load BPE tokenizer
        timer.reset();
        BPETokenizer tok;
        if (!load_bpe_from_gguf(&tok, text_enc_gguf)) {
            fprintf(stderr, "FATAL: failed to load tokenizer from %s\n", text_enc_gguf);
            dit_ggml_free(&model);
            if (have_vae) vae_ggml_free(&vae);
            return 1;
        }
        fprintf(stderr, "[Load] BPE tokenizer: %.1f ms\n", timer.ms());

        // 2. Build formatted prompts
        const char * instruction = "Generate audio semantic tokens based on the given conditions:";
        char metas[512];
        snprintf(metas, sizeof(metas),
                 "- bpm: %s\n- timesignature: %s\n- keyscale: %s\n- duration: %d seconds\n",
                 bpm, timesig, keyscale, (int)duration);
        std::string text_str = std::string("# Instruction\n")
            + instruction + "\n\n"
            + "# Caption\n" + caption + "\n\n"
            + "# Metas\n" + metas + "<|endoftext|>\n";

        bool instrumental = (strcmp(lyrics, "[Instrumental]") == 0 || strcmp(lyrics, "[instrumental]") == 0);
        std::string lyric_str = std::string("# Languages\n") + language + "\n\n# Lyric\n"
            + (instrumental ? "[Instrumental]" : lyrics) + "<|endoftext|>";

        // 3. Tokenize
        auto text_ids  = bpe_encode(&tok, text_str.c_str(), true);
        auto lyric_ids = bpe_encode(&tok, lyric_str.c_str(), true);
        int S_text  = (int)text_ids.size();
        int S_lyric = (int)lyric_ids.size();
        fprintf(stderr, "[Pipeline] caption: %d tokens, lyrics: %d tokens\n", S_text, S_lyric);

        // 4. Text encoder forward (caption only)
        timer.reset();
        Qwen3GGML text_enc = {};
        qwen3_init_backend(&text_enc);
        if (!qwen3_load_text_encoder(&text_enc, text_enc_gguf)) {
            fprintf(stderr, "FATAL: failed to load text encoder\n");
            dit_ggml_free(&model);
            if (have_vae) vae_ggml_free(&vae);
            return 1;
        }
        fprintf(stderr, "[Load] TextEncoder: %.1f ms\n", timer.ms());

        int H_text = text_enc.cfg.hidden_size;  // 1024
        std::vector<float> text_hidden(H_text * S_text);

        timer.reset();
        qwen3_forward(&text_enc, text_ids.data(), S_text, text_hidden.data());
        fprintf(stderr, "[Encode] TextEncoder (%d tokens): %.1f ms\n", S_text, timer.ms());
        debug_dump_2d(&dbg, "text_hidden", text_hidden.data(), S_text, H_text);

        // 5. Lyric embedding (CPU vocab lookup from text encoder embed table)
        timer.reset();
        std::vector<float> lyric_embed(H_text * S_lyric);
        {
            GGUFModel gf_te = {};
            if (!gf_load(&gf_te, text_enc_gguf)) {
                fprintf(stderr, "FATAL: cannot reopen text encoder GGUF for lyric embed\n");
                dit_ggml_free(&model);
                if (have_vae) vae_ggml_free(&vae);
                return 1;
            }
            const void * embed_data = gf_get_data(gf_te, "embed_tokens.weight");
            if (!embed_data) {
                fprintf(stderr, "FATAL: embed_tokens.weight not found\n");
                gf_close(&gf_te);
                dit_ggml_free(&model);
                if (have_vae) vae_ggml_free(&vae);
                return 1;
            }
            qwen3_cpu_embed_lookup(embed_data, H_text,
                                    lyric_ids.data(), S_lyric,
                                    lyric_embed.data());
            gf_close(&gf_te);
        }
        fprintf(stderr, "[Encode] Lyric vocab lookup (%d tokens): %.1f ms\n", S_lyric, timer.ms());
        debug_dump_2d(&dbg, "lyric_embed", lyric_embed.data(), S_lyric, H_text);

        // 6. Condition encoder forward
        timer.reset();
        CondGGML cond = {};
        cond_ggml_init_backend(&cond);
        if (!cond_ggml_load(&cond, dit_gguf)) {
            fprintf(stderr, "FATAL: failed to load condition encoder\n");
            dit_ggml_free(&model);
            if (have_vae) vae_ggml_free(&vae);
            return 1;
        }
        fprintf(stderr, "[Load] ConditionEncoder: %.1f ms\n", timer.ms());

        // Silence feats for timbre input: first 750 frames (30s @ 25Hz)
        const int S_ref = 750;
        std::vector<float> silence_feats(S_ref * 64);
        memcpy(silence_feats.data(), silence_full.data(), S_ref * 64 * sizeof(float));

        timer.reset();
        std::vector<float> enc_hidden;
        cond_ggml_forward(&cond, text_hidden.data(), S_text,
                           lyric_embed.data(), S_lyric,
                           silence_feats.data(), S_ref,
                           enc_hidden, &enc_S);
        fprintf(stderr, "[Encode] ConditionEncoder: %.1f ms, enc_S=%d\n", timer.ms(), enc_S);

        qwen3_free(&text_enc);
        cond_ggml_free(&cond);

        debug_dump_2d(&dbg, "enc_hidden", enc_hidden.data(), enc_S, 2048);

        // --- Context building ---
        // Silence latent for this T
        std::vector<float> silence(Oc * T);
        memcpy(silence.data(), silence_full.data(), (size_t)(Oc * T) * sizeof(float));

        // Decode audio codes if provided
        int decoded_T = 0;
        std::vector<float> decoded_latents;
        if (!codes_vec.empty()) {
            timer.reset();
            DetokGGML detok = {};
            if (!detok_ggml_load(&detok, dit_gguf, model.backend, model.cpu_backend)) {
                fprintf(stderr, "FATAL: failed to load detokenizer\n");
                dit_ggml_free(&model);
                if (have_vae) vae_ggml_free(&vae);
                return 1;
            }
            fprintf(stderr, "[Load] Detokenizer: %.1f ms\n", timer.ms());

            int T_5Hz = (int)codes_vec.size();
            int T_25Hz_codes = T_5Hz * 5;
            decoded_latents.resize(T_25Hz_codes * Oc);

            timer.reset();
            int ret = detok_ggml_decode(&detok, codes_vec.data(), T_5Hz, decoded_latents.data());
            if (ret < 0) {
                fprintf(stderr, "FATAL: detokenizer decode failed\n");
                dit_ggml_free(&model);
                if (have_vae) vae_ggml_free(&vae);
                return 1;
            }
            fprintf(stderr, "[Context] Detokenizer: %.1f ms\n", timer.ms());

            decoded_T = T_25Hz_codes < T ? T_25Hz_codes : T;
            debug_dump_2d(&dbg, "detok_output", decoded_latents.data(), T_25Hz_codes, Oc);
            detok_ggml_free(&detok);
        }

        // Build single context: [T, ctx_ch] = src_latents[64] + mask_ones[64]
        std::vector<float> context_single(T * ctx_ch);
        for (int t = 0; t < T; t++) {
            const float * src = (t < decoded_T)
                ? decoded_latents.data() + t * Oc
                : silence.data() + t * Oc;
            for (int c = 0; c < Oc; c++)
                context_single[t * ctx_ch + c] = src[c];
            for (int c = 0; c < Oc; c++)
                context_single[t * ctx_ch + Oc + c] = 1.0f;
        }

        // Replicate context for N batch samples (all identical)
        std::vector<float> context(batch_n * T * ctx_ch);
        for (int b = 0; b < batch_n; b++)
            memcpy(context.data() + b * T * ctx_ch, context_single.data(),
                   T * ctx_ch * sizeof(float));

        // --- Generate N noise samples ---
        std::vector<float> noise(batch_n * Oc * T);

        if (noise_file && batch_n == 1) {
            // Load pre-generated Philox noise from bf16 file (batch=1 only)
            FILE * nf = fopen(noise_file, "rb");
            if (!nf) { fprintf(stderr, "FATAL: cannot open noise file %s\n", noise_file); return 1; }
            fseek(nf, 0, SEEK_END);
            int total_bf16 = (int)(ftell(nf) / sizeof(uint16_t));
            fseek(nf, 0, SEEK_SET);
            int T_file = total_bf16 / Oc;
            if (T_file < T) {
                fprintf(stderr, "FATAL: noise file too short: T_file=%d < T=%d\n", T_file, T);
                fclose(nf); return 1;
            }
            std::vector<uint16_t> bf16_all(Oc * T);
            fread(bf16_all.data(), sizeof(uint16_t), Oc * T, nf);
            fclose(nf);
            for (int i = 0; i < Oc * T; i++) {
                uint32_t w = (uint32_t)bf16_all[i] << 16;
                float v; memcpy(&v, &w, 4);
                noise[i] = v;
            }
            fprintf(stderr, "[Context] loaded noise from %s: [%d, %d] bf16\n",
                    noise_file, T, Oc);
        } else {
            // Generate N noise samples with seeds: seed, seed+1, ..., seed+N-1
            for (int b = 0; b < batch_n; b++) {
                std::mt19937 rng(seed + b);
                std::normal_distribution<float> normal(0.0f, 1.0f);
                float * dst = noise.data() + b * Oc * T;
                for (int i = 0; i < Oc * T; i++)
                    dst[i] = normal(rng);
                fprintf(stderr, "[Context Batch%d] noise seed=%d\n", b, seed + b);
            }
        }

        // --- DiT Generate ---
        std::vector<float> output(batch_n * Oc * T);

        // Debug dumps (sample 0)
        debug_dump_2d(&dbg, "noise", noise.data(), T, Oc);
        debug_dump_2d(&dbg, "context", context.data(), T, ctx_ch);

        fprintf(stderr, "[DiT] Starting: T=%d, S=%d, enc_S=%d, steps=%d, batch=%d\n",
                T, S, enc_S, num_steps, batch_n);

        timer.reset();
        dit_ggml_generate(&model, noise.data(), context.data(), enc_hidden.data(),
                          enc_S, T, batch_n, num_steps, schedule.data(), output.data(),
                          guidance_scale, &dbg);
        fprintf(stderr, "[DiT] Total generation: %.1f ms (%.1f ms/sample)\n",
                timer.ms(), timer.ms() / batch_n);

        debug_dump_2d(&dbg, "dit_output", output.data(), T, Oc);

        // --- VAE Decode + Write WAVs ---
        if (have_vae) {
            int T_latent = T;
            int T_audio_max = T_latent * 1920;
            std::vector<float> audio(2 * T_audio_max);

            for (int b = 0; b < batch_n; b++) {
                float * dit_out = output.data() + b * Oc * T;

                timer.reset();
                int T_audio = vae_ggml_decode_tiled(&vae, dit_out, T_latent, audio.data(),
                                                     T_audio_max, vae_chunk, vae_overlap);
                if (T_audio < 0) {
                    fprintf(stderr, "[VAE Batch%d] ERROR: decode failed\n", b);
                    continue;
                }
                fprintf(stderr, "[VAE Batch%d] Decode: %.1f ms\n", b, timer.ms());

                // Peak normalization to -1.0 dB
                {
                    float peak = 0.0f;
                    int n_samples = 2 * T_audio;
                    for (int i = 0; i < n_samples; i++) {
                        float a = audio[i] < 0 ? -audio[i] : audio[i];
                        if (a > peak) peak = a;
                    }
                    if (peak > 1e-6f) {
                        const float target_amp = powf(10.0f, -1.0f / 20.0f);
                        float gain = target_amp / peak;
                        for (int i = 0; i < n_samples; i++)
                            audio[i] *= gain;
                    }
                }

                // Write WAV: basename + batch_index + .wav
                char wav_path[1024];
                snprintf(wav_path, sizeof(wav_path), "%s%d.wav", basename.c_str(), b);

                if (b == 0) debug_dump_2d(&dbg, "vae_audio", audio.data(), 2, T_audio);

                if (write_wav(wav_path, audio.data(), T_audio, 48000)) {
                    fprintf(stderr, "[VAE Batch%d] Wrote %s: %d samples (%.2fs @ 48kHz stereo)\n",
                            b, wav_path, T_audio, (float)T_audio / 48000.0f);
                } else {
                    fprintf(stderr, "[VAE Batch%d] FATAL: failed to write %s\n", b, wav_path);
                }
            }
        }

        fprintf(stderr, "[Request %d/%d] Done\n", ri + 1, (int)request_paths.size());
    }

    if (have_vae) vae_ggml_free(&vae);
    dit_ggml_free(&model);
    fprintf(stderr, "[Pipeline] All done\n");
    return 0;
}
