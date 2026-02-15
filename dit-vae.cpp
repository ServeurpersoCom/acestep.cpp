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
    short nc = n_channels; fwrite(&nc, 2, 1, f);
    fwrite(&sr, 4, 1, f);
    fwrite(&byte_rate, 4, 1, f);
    short ba = block_align; fwrite(&ba, 2, 1, f);
    short bp = bits; fwrite(&bp, 2, 1, f);
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
    fprintf(stderr, "Usage: %s --request <json> --dit <dir> --text-encoder <dir> --vae <dir> [options]\n\n", prog);
    fprintf(stderr, "Required:\n");
    fprintf(stderr, "  --request <json>        Request JSON (from ace-qwen3 --request)\n");
    fprintf(stderr, "  --text-encoder <dir>    Qwen3-Embedding-0.6B directory\n");
    fprintf(stderr, "  --dit <dir>             DiT model directory (e.g. acestep-v15-turbo)\n");
    fprintf(stderr, "  --vae <dir>             VAE directory\n\n");
    fprintf(stderr, "Audio:\n");
    fprintf(stderr, "  --noise-file <path>     Load noise from bf16 file (Philox RNG dump)\n");
    fprintf(stderr, "  --output <path>         Output WAV (default: output.wav)\n\n");
    fprintf(stderr, "VAE tiling (memory control):\n");
    fprintf(stderr, "  --vae-chunk <n>         Latent frames per tile (default: 256)\n");
    fprintf(stderr, "  --vae-overlap <n>       Overlap frames per side (default: 64)\n\n");
    fprintf(stderr, "Debug:\n");
    fprintf(stderr, "  --dump <dir>            Dump intermediate tensors\n");
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

    const char * request_path = NULL;
    const char * text_enc_dir = NULL;
    const char * dit_dir      = NULL;
    const char * vae_dir      = NULL;
    const char * wav_path     = "output.wav";
    const char * dump_dir     = NULL;
    const char * noise_file   = NULL;
    int vae_chunk             = 256;
    int vae_overlap           = 64;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--request") == 0 && i+1 < argc) request_path = argv[++i];
        else if (strcmp(argv[i], "--text-encoder") == 0 && i+1 < argc) text_enc_dir = argv[++i];
        else if (strcmp(argv[i], "--dit") == 0 && i+1 < argc) dit_dir = argv[++i];
        else if (strcmp(argv[i], "--vae") == 0 && i+1 < argc) vae_dir = argv[++i];
        else if (strcmp(argv[i], "--noise-file") == 0 && i+1 < argc) noise_file = argv[++i];
        else if (strcmp(argv[i], "--dump") == 0 && i+1 < argc) dump_dir = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i+1 < argc) wav_path = argv[++i];
        else if (strcmp(argv[i], "--vae-chunk") == 0 && i+1 < argc) vae_chunk = atoi(argv[++i]);
        else if (strcmp(argv[i], "--vae-overlap") == 0 && i+1 < argc) vae_overlap = atoi(argv[++i]);
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]); return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]); return 1;
        }
    }

    if (!request_path) {
        fprintf(stderr, "ERROR: --request required\n");
        print_usage(argv[0]); return 1;
    }
    if (!dit_dir) {
        fprintf(stderr, "ERROR: --dit required\n");
        print_usage(argv[0]); return 1;
    }
    if (!text_enc_dir) {
        fprintf(stderr, "ERROR: --text-encoder required\n");
        print_usage(argv[0]); return 1;
    }

    // Read request JSON
    AceRequest req;
    if (!request_parse(&req, request_path)) return 1;
    request_dump(&req, stderr);

    int num_steps        = req.inference_steps;
    float shift          = req.shift;
    float guidance_scale = req.guidance_scale;

    if (req.caption.empty()) {
        fprintf(stderr, "ERROR: caption is empty in %s\n", request_path);
        return 1;
    }

    // Extract params from request
    const char * caption  = req.caption.c_str();
    const char * lyrics   = req.lyrics.empty() ? "[Instrumental]" : req.lyrics.c_str();
    char bpm_str[16] = "N/A";
    if (req.bpm > 0) snprintf(bpm_str, sizeof(bpm_str), "%d", req.bpm);
    const char * bpm = bpm_str;
    const char * keyscale = req.keyscale.empty() ? "N/A" : req.keyscale.c_str();
    const char * timesig  = req.timesignature.empty() ? "N/A" : req.timesignature.c_str();
    const char * language = req.vocal_language.empty() ? "en" : req.vocal_language.c_str();
    float duration        = req.duration > 0 ? req.duration : 120.0f;
    int seed              = req.seed;

    // Parse audio codes from request
    std::vector<int> codes_vec = parse_codes_string(req.audio_codes);
    if (!codes_vec.empty())
        fprintf(stderr, "[Pipeline] %zu audio codes from request (%.1fs @ 5Hz)\n",
                codes_vec.size(), codes_vec.size() / 5.0f);
    else
        fprintf(stderr, "[Pipeline] No codes, text-to-music from noise\n");

    int T = 0;

    if (seed < 0) {
        std::random_device rd;
        seed = (int)(rd() & 0x7FFFFFFF);
    }

    const int FRAMES_PER_SECOND = 25;  // ACEStep latent frame rate

    DebugDumper dbg;
    debug_init(&dbg, dump_dir);

    DiTGGMLConfig cfg;
    DiTGGML model = {};
    Timer timer;

    // Init DiT backend + load
    dit_ggml_init_backend(&model);
    fprintf(stderr, "[Load] Backend init: %.1f ms\n", timer.ms());

    timer.reset();
    if (!dit_ggml_load(&model, dit_dir, cfg)) {
        fprintf(stderr, "FATAL: failed to load DiT model\n");
        return 1;
    }
    fprintf(stderr, "[Load] DiT weight load: %.1f ms\n", timer.ms());

    int S = 0, enc_S = 0;
    int Oc = cfg.out_channels;      // 64
    int ctx_ch = cfg.in_channels - Oc;  // 128

    // Build schedule: t_i = shift * t / (1 + (shift-1)*t) where t = 1 - i/steps
    std::vector<float> schedule(num_steps);
    for (int i = 0; i < num_steps; i++) {
        float t = 1.0f - (float)i / num_steps;
        schedule[i] = shift * t / (1.0f + (shift - 1.0f) * t);
    }
    fprintf(stderr, "[Pipeline] shift=%.1f steps=%d: ", shift, num_steps);
    for (int i = 0; i < num_steps; i++) fprintf(stderr, "%.4f ", schedule[i]);
    fprintf(stderr, "\n");

    std::vector<float> noise;
    std::vector<float> context;
    std::vector<float> enc_hidden;

    // Full pipeline (text -> cond -> DiT -> VAE -> WAV)
    if (text_enc_dir) {
        T = (int)(duration * FRAMES_PER_SECOND);
        // Round up to patch_size multiple
        T = ((T + cfg.patch_size - 1) / cfg.patch_size) * cfg.patch_size;
        S = T / cfg.patch_size;
        fprintf(stderr, "[Pipeline] duration=%.1fs, T=%d, S=%d\n", duration, T, S);

        // 1. Load BPE tokenizer
        timer.reset();
        BPETokenizer tok;
        if (!load_bpe_tokenizer(&tok, text_enc_dir)) {
            fprintf(stderr, "FATAL: failed to load tokenizer from %s\n", text_enc_dir);
            return 1;
        }
        fprintf(stderr, "[Load] BPE tokenizer: %.1f ms\n", timer.ms());

        // 2. Build formatted prompts (match Python build_text_prompt/build_lyric_prompt)
        // Python always uses "Fill the audio semantic mask..." for text2music (even with LM codes)
        const char * instruction = "Fill the audio semantic mask based on the given conditions:";
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

        // 3. Text encoder forward
        timer.reset();
        Qwen3GGML text_enc = {};
        qwen3_init_backend(&text_enc);
        if (!qwen3_load_text_encoder(&text_enc, text_enc_dir)) {
            fprintf(stderr, "FATAL: failed to load text encoder\n");
            return 1;
        }
        fprintf(stderr, "[Load] TextEncoder: %.1f ms\n", timer.ms());

        int H_text = text_enc.cfg.hidden_size;  // 1024
        std::vector<float> text_hidden(H_text * S_text);

        timer.reset();
        qwen3_forward(&text_enc, text_ids.data(), S_text, text_hidden.data());
        fprintf(stderr, "[Encode] TextEncoder (%d tokens): %.1f ms\n", S_text, timer.ms());
        debug_dump_2d(&dbg, "text_hidden", text_hidden.data(), S_text, H_text);

        // 4. Lyric embedding (CPU vocab lookup)
        // Use text encoder's embed table (mmapped bf16) for lyric token embedding
        timer.reset();
        std::vector<float> lyric_embed(H_text * S_lyric);

        // Get embed_tokens data pointer from text encoder's safetensors
        // The embed table is bf16 [vocab, 1024] in PyTorch layout (1024 contiguous per token)
        {
            SafeTensors st_te;
            if (!safe_load(st_te, text_enc_dir)) {
                fprintf(stderr, "FATAL: cannot reload text encoder safetensors for lyric embed\n");
                return 1;
            }
            auto it = st_te.tensors.find("embed_tokens.weight");
            if (it == st_te.tensors.end()) {
                fprintf(stderr, "FATAL: embed_tokens.weight not found\n");
                return 1;
            }
            qwen3_cpu_embed_lookup(it->second.data, H_text,
                                    lyric_ids.data(), S_lyric,
                                    lyric_embed.data());
            // st_te goes out of scope, munmap happens. lyric_embed is on CPU, so OK.
        }
        fprintf(stderr, "[Encode] Lyric vocab lookup (%d tokens): %.1f ms\n", S_lyric, timer.ms());
        debug_dump_2d(&dbg, "lyric_embed", lyric_embed.data(), S_lyric, H_text);

        // 5. Condition encoder forward
        timer.reset();
        CondGGML cond = {};
        cond_ggml_init_backend(&cond);
        if (!cond_ggml_load(&cond, dit_dir)) {
            fprintf(stderr, "FATAL: failed to load condition encoder\n");
            return 1;
        }
        fprintf(stderr, "[Load] ConditionEncoder: %.1f ms\n", timer.ms());

        // Load silence_latent.bin as timbre input (same as Python reference)
        // Format: raw f32 [15000, 64], we use first 750 frames (30s @ 25Hz)
        const int S_ref = 750;
        std::vector<float> silence_feats(S_ref * 64);
        {
            std::string path = std::string(dit_dir) + "/silence_latent.bin";
            FILE * f = fopen(path.c_str(), "rb");
            if (!f) { fprintf(stderr, "FATAL: cannot open %s\n", path.c_str()); return 1; }
            if (fread(silence_feats.data(), sizeof(float), S_ref * 64, f) != (size_t)(S_ref * 64)) {
                fprintf(stderr, "FATAL: silence_latent.bin too short\n"); fclose(f); return 1;
            }
            fclose(f);
        }

        timer.reset();
        cond_ggml_forward(&cond, text_hidden.data(), S_text,
                           lyric_embed.data(), S_lyric,
                           silence_feats.data(), S_ref,
                           enc_hidden, &enc_S);
        fprintf(stderr, "[Encode] ConditionEncoder: %.1f ms\n", timer.ms());
        debug_dump_2d(&dbg, "enc_hidden", enc_hidden.data(), enc_S, 2048);

        // Free text encoder and cond encoder (weights no longer needed)
        qwen3_free(&text_enc);
        cond_ggml_free(&cond);

        // 6. Generate context_latents and noise
        // context = cat(src_latents[T,64], chunk_masks[T,64]) = [T, 128]
        // src_latents: decoded audio codes (if present) or silence encoding
        std::mt19937 rng(seed);
        std::normal_distribution<float> normal(0.0f, 1.0f);

        context.resize(ctx_ch * T);
        noise.resize(Oc * T);

        // Load silence_latent.bin [15000, 64] f32 (always needed for padding)
        std::vector<float> silence(Oc * T);
        {
            std::string slp = std::string(dit_dir) + "/silence_latent.bin";
            FILE * f = fopen(slp.c_str(), "rb");
            if (!f) { fprintf(stderr, "FATAL: cannot open %s\n", slp.c_str()); return 1; }
            fseek(f, 0, SEEK_END);
            int silence_T = (int)(ftell(f) / (Oc * sizeof(float)));
            fseek(f, 0, SEEK_SET);
            if (silence_T < T) {
                fprintf(stderr, "FATAL: silence_latent too short: %d < %d\n", silence_T, T);
                fclose(f); return 1;
            }
            fread(silence.data(), sizeof(float), Oc * T, f);
            fclose(f);
            fprintf(stderr, "[Context] loaded silence_latent: [%d, %d] (using first %d frames)\n",
                    silence_T, Oc, T);
        }

        // Decode audio codes if provided, otherwise use silence for all frames
        int decoded_T = 0;
        std::vector<float> decoded_latents;

        if (!codes_vec.empty()) {
            timer.reset();
            DetokGGML detok = {};
            if (!detok_ggml_load(&detok, dit_dir, model.backend, model.cpu_backend)) {
                fprintf(stderr, "FATAL: failed to load detokenizer\n"); return 1;
            }
            fprintf(stderr, "[Load] Detokenizer: %.1f ms\n", timer.ms());

            int T_5Hz = (int)codes_vec.size();
            int T_25Hz_codes = T_5Hz * 5;
            decoded_latents.resize(T_25Hz_codes * Oc);

            timer.reset();
            int ret = detok_ggml_decode(&detok, codes_vec.data(), T_5Hz, decoded_latents.data());
            if (ret < 0) { fprintf(stderr, "FATAL: detokenizer decode failed\n"); return 1; }
            fprintf(stderr, "[Context] Detokenizer decode: %.1f ms\n", timer.ms());

            decoded_T = T_25Hz_codes < T ? T_25Hz_codes : T;
            debug_dump_2d(&dbg, "detok_output", decoded_latents.data(), T_25Hz_codes, Oc);
            detok_ggml_free(&detok);
        }

        // Build context: src_latents[64] + mask_ones[64] per frame
        for (int t = 0; t < T; t++) {
            const float * src = (t < decoded_T)
                ? decoded_latents.data() + t * Oc
                : silence.data() + t * Oc;
            for (int c = 0; c < Oc; c++)
                context[t * ctx_ch + c] = src[c];
            for (int c = 0; c < Oc; c++)
                context[t * ctx_ch + Oc + c] = 1.0f;
        }
        if (decoded_T > 0)
            fprintf(stderr, "[Context] %d decoded + %d silence frames\n",
                    decoded_T, T - decoded_T);

        // Initial noise for flow matching
        if (noise_file) {
            // Load pre-generated Philox noise from bf16 file.
            // File layout: [T, C=64] time-major (matches prepare_noise: torch.randn([1, T, C])).
            // C++ layout: noise[t * Oc + c] = time-major (same order).
            // Read linearly: file[t * Oc + c] -> noise[t * Oc + c].
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
            fprintf(stderr, "[Context] loaded noise from %s: [%d, %d] bf16 (time-major, T_file=%d)\n",
                    noise_file, T, Oc, T_file);
        } else {
            for (int i = 0; i < Oc * T; i++)
                noise[i] = normal(rng);
        }

        fprintf(stderr, "[Context] context: [%d, %d], noise: [%d, %d], enc: [%d, 2048]\n",
                T, ctx_ch, T, Oc, enc_S);
        debug_dump_2d(&dbg, "noise", noise.data(), T, Oc);
        debug_dump_2d(&dbg, "context", context.data(), T, ctx_ch);
    }

    // DiT Generate
    std::vector<float> output(Oc * T);

    fprintf(stderr, "[DiT] Starting flow matching: T=%d, S=%d, enc_S=%d, steps=%d\n",
            T, S, enc_S, num_steps);

    timer.reset();
    dit_ggml_generate(&model, noise.data(), context.data(), enc_hidden.data(),
                      enc_S, T, num_steps, schedule.data(), output.data(),
                      guidance_scale, &dbg);
    fprintf(stderr, "[DiT] Total generation: %.1f ms\n", timer.ms());

    // Output stats
    {
        int n_output = Oc * T;
        fprintf(stderr, "[Output] shape: [%d, %d], first 8 values:", T, Oc);
        for (int i = 0; i < 8 && i < n_output; i++)
            fprintf(stderr, " %.6f", output[i]);
        fprintf(stderr, "\n");

        float sum = 0, sum_sq = 0;
        for (int i = 0; i < n_output; i++) {
            sum += output[i];
            sum_sq += output[i] * output[i];
        }
        float mean = sum / n_output;
        float var = sum_sq / n_output - mean * mean;
        fprintf(stderr, "[Output] mean=%.6f, std=%.6f\n", mean, sqrtf(var > 0 ? var : 0));

        debug_dump_2d(&dbg, "dit_output", output.data(), T, Oc);
    }

    // VAE Decode
    if (vae_dir) {
        fprintf(stderr, "[VAE] Decoding...\n");
        VAEGGML vae = {};

        timer.reset();
        vae_ggml_load(&vae, vae_dir);
        fprintf(stderr, "[Load] VAE weights: %.1f ms\n", timer.ms());

        int T_latent = T;
        int T_audio_max = T_latent * 1920;
        std::vector<float> audio(2 * T_audio_max);

        timer.reset();
        int T_audio = vae_ggml_decode_tiled(&vae, output.data(), T_latent, audio.data(), T_audio_max,
                                            vae_chunk, vae_overlap);
        if (T_audio < 0) {
            fprintf(stderr, "FATAL: VAE decode failed\n");
            vae_ggml_free(&vae);
            dit_ggml_free(&model);
            return 1;
        }
        fprintf(stderr, "[VAE] Decode: %.1f ms\n", timer.ms());

        // Anti-clipping normalization (matches Python handler.py line 2106-2109)
        {
            float peak = 0.0f;
            int n_samples = 2 * T_audio;  // stereo
            for (int i = 0; i < n_samples; i++) {
                float a = audio[i] < 0 ? -audio[i] : audio[i];
                if (a > peak) peak = a;
            }
            if (peak > 1.0f) {
                float inv_peak = 1.0f / peak;
                for (int i = 0; i < n_samples; i++)
                    audio[i] *= inv_peak;
                fprintf(stderr, "[VAE] Anti-clip normalize: peak=%.3f -> 1.0\n", peak);
            }
        }

        if (write_wav(wav_path, audio.data(), T_audio, 48000)) {
            fprintf(stderr, "[VAE] Wrote %s: %d samples (%.2fs @ 48kHz stereo)\n",
                    wav_path, T_audio, (float)T_audio / 48000.0f);
        } else {
            fprintf(stderr, "FATAL: failed to write %s\n", wav_path);
        }

        debug_dump_2d(&dbg, "vae_audio", audio.data(), 2, T_audio);

        vae_ggml_free(&vae);
    }

    dit_ggml_free(&model);
    fprintf(stderr, "[Pipeline] OK\n");
    return 0;
}
