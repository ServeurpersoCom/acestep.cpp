// ace-synth.cpp: ACE-Step synthesis CLI
// Thin wrapper: parses args, calls pipeline-synth, writes output files.

#include "audio-io.h"
#include "pipeline-synth.h"
#include "request.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void print_usage(const char * prog) {
    fprintf(stderr,
            "Usage: %s --request <json...> --text-encoder <gguf> --dit <gguf> --vae <gguf> [options]\n\n"
            "Required:\n"
            "  --request <json...>     One or more request JSONs (from ace-lm --request)\n"
            "  --text-encoder <gguf>   Text encoder GGUF file\n"
            "  --dit <gguf>            DiT GGUF file\n"
            "  --vae <gguf>            VAE GGUF file\n\n"
            "Reference audio:\n"
            "  --src-audio <file>      Source audio (WAV or MP3, any sample rate)\n\n"
            "LoRA:\n"
            "  --lora <path>           LoRA safetensors file or directory\n"
            "  --lora-scale <float>    LoRA scaling factor (default: 1.0)\n\n"
            "Batch:\n"
            "  --batch <N>             DiT variations per request (default: 1, max 9)\n\n"
            "Output:\n"
            "  Default: MP3 at 128 kbps. input.json -> input0.mp3, input1.mp3, ...\n"
            "  --mp3-bitrate <kbps>    MP3 bitrate (default: 128)\n"
            "  --wav                   Output WAV instead of MP3\n\n"
            "VAE tiling (memory control):\n"
            "  --vae-chunk <N>         Latent frames per tile (default: 256)\n"
            "  --vae-overlap <N>       Overlap frames per side (default: 64)\n\n"
            "Debug:\n"
            "  --no-fa                 Disable flash attention\n"
            "  --dump <dir>            Dump intermediate tensors\n",
            prog);
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
    const char *              lora_path      = NULL;
    float                     lora_scale     = 1.0f;
    bool                      use_fa         = true;
    int                       batch_n        = 1;
    int                       vae_chunk      = 256;
    int                       vae_overlap    = 64;
    bool                      output_wav     = false;  // default MP3, --wav forces WAV
    int                       mp3_kbps       = 128;

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
        } else if (strcmp(argv[i], "--lora") == 0 && i + 1 < argc) {
            lora_path = argv[++i];
        } else if (strcmp(argv[i], "--lora-scale") == 0 && i + 1 < argc) {
            lora_scale = (float) atof(argv[++i]);
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
        } else if (strcmp(argv[i], "--wav") == 0) {
            output_wav = true;
        } else if (strcmp(argv[i], "--mp3-bitrate") == 0 && i + 1 < argc) {
            mp3_kbps = atoi(argv[++i]);
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
        fprintf(stderr, "[CLI] ERROR: --request required\n");
        print_usage(argv[0]);
        return 1;
    }
    if (batch_n < 1 || batch_n > 9) {
        fprintf(stderr, "[CLI] ERROR: --batch must be 1..9\n");
        return 1;
    }
    if (!dit_gguf) {
        fprintf(stderr, "[CLI] ERROR: --dit required\n");
        print_usage(argv[0]);
        return 1;
    }
    if (!text_enc_gguf) {
        fprintf(stderr, "[CLI] ERROR: --text-encoder required\n");
        print_usage(argv[0]);
        return 1;
    }

    // Load models
    AceSynthParams params;
    ace_synth_default_params(&params);
    params.text_encoder_path = text_enc_gguf;
    params.dit_path          = dit_gguf;
    params.vae_path          = vae_gguf;
    params.lora_path         = lora_path;
    params.lora_scale        = lora_scale;
    params.use_fa            = use_fa;
    params.vae_chunk         = vae_chunk;
    params.vae_overlap       = vae_overlap;
    params.dump_dir          = dump_dir;

    AceSynth * ctx = ace_synth_load(&params);
    if (!ctx) {
        return 1;
    }

    // Read source audio (cover/lego mode)
    float * src_interleaved = NULL;
    int     src_len         = 0;
    if (src_audio_path) {
        if (!vae_gguf) {
            fprintf(stderr, "[Cover] ERROR: --src-audio requires --vae\n");
            ace_synth_free(ctx);
            return 1;
        }
        int     T_audio = 0;
        float * planar  = audio_read_48k(src_audio_path, &T_audio);
        if (!planar) {
            fprintf(stderr, "[Cover] FATAL: cannot read --src-audio %s\n", src_audio_path);
            ace_synth_free(ctx);
            return 1;
        }
        fprintf(stderr, "[Cover] Source audio: %.2fs @ 48kHz\n", (float) T_audio / 48000.0f);

        // VAE expects interleaved [L0,R0,L1,R1,...], convert from planar
        src_interleaved = (float *) malloc((size_t) T_audio * 2 * sizeof(float));
        for (int t = 0; t < T_audio; t++) {
            src_interleaved[t * 2 + 0] = planar[t];
            src_interleaved[t * 2 + 1] = planar[T_audio + t];
        }
        free(planar);
        src_len = T_audio;
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
            fprintf(stderr, "[Request] ERROR: failed to parse %s, skipping\n", rpath);
            continue;
        }
        request_dump(&req, stderr);
        if (req.caption.empty() && req.lego.empty()) {
            fprintf(stderr, "[Request] ERROR: caption is empty in %s, skipping\n", rpath);
            continue;
        }

        // Generate
        std::vector<AceAudio> audio(batch_n);
        if (ace_synth_generate(ctx, &req, src_interleaved, src_len, batch_n, audio.data()) != 0) {
            fprintf(stderr, "[Request] ERROR: generation failed for %s\n", rpath);
            continue;
        }

        // Write output files
        for (int b = 0; b < batch_n; b++) {
            if (!audio[b].samples) {
                continue;
            }

            // Write output: basename + batch_index + extension
            const char * ext = output_wav ? ".wav" : ".mp3";
            char         out_path[1024];
            snprintf(out_path, sizeof(out_path), "%s%d%s", basename.c_str(), b, ext);

            if (!audio_write(out_path, audio[b].samples, audio[b].n_samples, 48000, mp3_kbps)) {
                fprintf(stderr, "[VAE Batch%d] FATAL: failed to write %s\n", b, out_path);
            }
            ace_audio_free(&audio[b]);
        }

        fprintf(stderr, "[Request %d/%d] Done\n", ri + 1, (int) request_paths.size());
    }

    free(src_interleaved);
    ace_synth_free(ctx);
    fprintf(stderr, "[Pipeline] All done\n");
    return 0;
}
