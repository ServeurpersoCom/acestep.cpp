// ace-synth.cpp: ACE-Step synthesis CLI
// Thin wrapper: parses args, calls pipeline-synth, writes output files.

#include "audio-io.h"
#include "pipeline-synth.h"
#include "request.h"
#include "task-types.h"
#include "version.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void usage(const char * prog) {
    fprintf(stderr, "acestep.cpp %s\n\n", ACE_VERSION);
    fprintf(stderr,
            "Usage: %s --request <json...> --embedding <gguf> --dit <gguf> --vae <gguf> [options]\n\n"
            "Required:\n"
            "  --request <json...>     One or more request JSONs (from ace-lm --request)\n"
            "  --embedding <gguf>      Embedding GGUF file\n"
            "  --dit <gguf>            DiT GGUF file\n"
            "  --vae <gguf>            VAE GGUF file\n\n"
            "Audio:\n"
            "  --src-audio <file>      Source audio (WAV or MP3)\n"
            "  --ref-audio <file>      Timbre reference audio (WAV or MP3)\n\n"
            "LoRA:\n"
            "  --lora <path>           LoRA safetensors file or directory\n"
            "  --lora-scale <float>    LoRA scaling factor (default: 1.0)\n\n"
            "Output:\n"
            "  Default: MP3 at 128 kbps. input.json -> input0.mp3, input1.mp3, ...\n"
            "  --mp3-bitrate <kbps>    MP3 bitrate (default: 128)\n"
            "  --wav                   Output WAV instead of MP3\n\n"
            "  --wav-format <fmt>      WAV audio format (default: wav16)\n"
            "                            Requires use of --wav\n"
            "                            Supported values: wav, wav16, wav24, wav32\n"
            "                              wav/wav16: 16-bit signed-integer PCM audio\n"
            "                              wav24: 24-bit signed-integer PCM audio\n"
            "                              wav32: 32-bit IEEE floating-point PCM audio\n"
            "                                (wav32 disables normalization & peak clip)\n\n"
            "Memory control:\n"
            "  --vae-chunk <N>         Latent frames per tile (default: 256)\n"
            "  --vae-overlap <N>       Overlap frames per side (default: 64)\n\n"
            "Debug:\n"
            "  --no-fa                 Disable flash attention\n"
            "  --no-batch-cfg          Split DiT CFG into two separate forwards\n"
            "  --clamp-fp16            Clamp hidden states to FP16 range\n"
            "  --dump <dir>            Dump intermediate tensors\n",
            prog);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    std::vector<const char *> request_paths;
    const char *              text_enc_gguf  = NULL;
    const char *              dit_gguf       = NULL;
    const char *              vae_gguf       = NULL;
    const char *              src_audio_path = NULL;
    const char *              ref_audio_path = NULL;
    const char *              dump_dir       = NULL;
    const char *              lora_path      = NULL;
    float                     lora_scale     = 1.0f;
    bool                      use_fa         = true;
    bool                      use_batch_cfg  = true;
    bool                      clamp_fp16     = false;
    int                       vae_chunk      = 256;
    int                       vae_overlap    = 64;
    bool                      output_wav     = false;  // default MP3, --wav forces WAV
    const char *              wav_format_str = nullptr;
    WavFormat                 wav_format     = WAV_FORMAT_S16;
    int                       mp3_kbps       = 128;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--request")) {
            // Collect all following non-option args
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                request_paths.push_back(argv[++i]);
            }
        } else if (!strcmp(argv[i], "--embedding") && i + 1 < argc) {
            text_enc_gguf = argv[++i];
        } else if (!strcmp(argv[i], "--dit") && i + 1 < argc) {
            dit_gguf = argv[++i];
        } else if (!strcmp(argv[i], "--vae") && i + 1 < argc) {
            vae_gguf = argv[++i];
        } else if (!strcmp(argv[i], "--src-audio") && i + 1 < argc) {
            src_audio_path = argv[++i];
        } else if (!strcmp(argv[i], "--ref-audio") && i + 1 < argc) {
            ref_audio_path = argv[++i];
        } else if (!strcmp(argv[i], "--lora") && i + 1 < argc) {
            lora_path = argv[++i];
        } else if (!strcmp(argv[i], "--lora-scale") && i + 1 < argc) {
            lora_scale = (float) atof(argv[++i]);
        } else if (!strcmp(argv[i], "--dump") && i + 1 < argc) {
            dump_dir = argv[++i];
        } else if (!strcmp(argv[i], "--no-fa")) {
            use_fa = false;
        } else if (!strcmp(argv[i], "--no-batch-cfg")) {
            use_batch_cfg = false;
        } else if (!strcmp(argv[i], "--clamp-fp16")) {
            clamp_fp16 = true;
        } else if (!strcmp(argv[i], "--vae-chunk") && i + 1 < argc) {
            vae_chunk = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--vae-overlap") && i + 1 < argc) {
            vae_overlap = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--wav")) {
            output_wav = true;
        } else if (!strcmp(argv[i], "--wav-format") && i + 1 < argc) {
            wav_format_str = argv[++i];
        } else if (!strcmp(argv[i], "--mp3-bitrate") && i + 1 < argc) {
            mp3_kbps = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (request_paths.empty()) {
        fprintf(stderr, "[CLI] ERROR: --request required\n");
        usage(argv[0]);
        return 1;
    }
    if (!dit_gguf) {
        fprintf(stderr, "[CLI] ERROR: --dit required\n");
        usage(argv[0]);
        return 1;
    }
    if (!text_enc_gguf) {
        fprintf(stderr, "[CLI] ERROR: --embedding required\n");
        usage(argv[0]);
        return 1;
    }
    if (!output_wav && wav_format_str != nullptr) {
        fprintf(stderr, "[CLI] ERROR: --wav-format requires usage of --wav\n");
        usage(argv[0]);
        return 1;
    }
    if (!parse_optional_wav_format(wav_format_str, wav_format)) {
        fprintf(stderr, "[CLI] ERROR: --wav-format requires a supported value\n");
        usage(argv[0]);
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
    params.use_batch_cfg     = use_batch_cfg;
    params.clamp_fp16        = clamp_fp16;
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
            fprintf(stderr, "[Ace-Synth] ERROR: --src-audio requires --vae\n");
            ace_synth_free(ctx);
            return 1;
        }
        int     T_audio = 0;
        float * planar  = audio_read_48k(src_audio_path, &T_audio);
        if (!planar) {
            fprintf(stderr, "[Ace-Synth] FATAL: cannot read --src-audio %s\n", src_audio_path);
            ace_synth_free(ctx);
            return 1;
        }
        fprintf(stderr, "[Ace-Synth] Source audio: %.2fs @ 48kHz\n", (float) T_audio / 48000.0f);

        src_interleaved = audio_planar_to_interleaved(planar, T_audio);
        free(planar);
        src_len = T_audio;
    }

    // Read reference audio (timbre conditioning)
    float * ref_interleaved = NULL;
    int     ref_len         = 0;
    if (ref_audio_path) {
        if (!vae_gguf) {
            fprintf(stderr, "[Ace-Synth] ERROR: --ref-audio requires --vae\n");
            free(src_interleaved);
            free(ref_interleaved);
            ace_synth_free(ctx);
            return 1;
        }
        int     T_audio = 0;
        float * planar  = audio_read_48k(ref_audio_path, &T_audio);
        if (!planar) {
            fprintf(stderr, "[Ace-Synth] FATAL: cannot read --ref-audio %s\n", ref_audio_path);
            free(src_interleaved);
            free(ref_interleaved);
            ace_synth_free(ctx);
            return 1;
        }
        fprintf(stderr, "[Ace-Synth] Reference audio: %.2fs @ 48kHz\n", (float) T_audio / 48000.0f);
        ref_interleaved = audio_planar_to_interleaved(planar, T_audio);
        free(planar);
        ref_len = T_audio;
    }

    // Parse all requests
    int                      batch_n = (int) request_paths.size();
    std::vector<AceRequest>  reqs(batch_n);
    std::vector<std::string> basenames(batch_n);
    for (int ri = 0; ri < batch_n; ri++) {
        const char * rpath = request_paths[ri];
        request_init(&reqs[ri]);
        if (!request_parse(&reqs[ri], rpath)) {
            fprintf(stderr, "[Ace-Synth] FATAL: failed to parse %s\n", rpath);
            ace_synth_free(ctx);
            free(src_interleaved);
            free(ref_interleaved);
            return 1;
        }
        request_dump(&reqs[ri], stderr);
        if (reqs[ri].caption.empty() && reqs[ri].task_type != TASK_LEGO && reqs[ri].task_type != TASK_EXTRACT) {
            fprintf(stderr, "[Ace-Synth] FATAL: caption is empty in %s\n", rpath);
            ace_synth_free(ctx);
            free(src_interleaved);
            free(ref_interleaved);
            return 1;
        }
        // output basename: strip .json suffix
        basenames[ri] = rpath;
        size_t dot    = basenames[ri].rfind(".json");
        if (dot != std::string::npos) {
            basenames[ri] = basenames[ri].substr(0, dot);
        }
    }
    fprintf(stderr, "[Ace-Synth] Batch: %d request(s)\n", batch_n);

    // process each request as a separate group (same codes = same T per group).
    // synth_batch_size variations within a group share the same T -> true GPU batch.
    // different requests can have different T -> separate pipeline calls.
    std::vector<AceAudio>    all_audio;
    std::vector<std::string> all_basenames;
    std::vector<int>         all_synth_indices;

    for (int ri = 0; ri < batch_n; ri++) {
        int sbs = reqs[ri].synth_batch_size;
        if (sbs < 1) {
            sbs = 1;
        }

        // resolve seed once per original request
        request_resolve_seed(&reqs[ri]);
        long long base_seed = reqs[ri].seed;

        // build group: N copies with consecutive seeds
        std::vector<AceRequest> group(sbs);
        for (int i = 0; i < sbs; i++) {
            group[i]      = reqs[ri];
            group[i].seed = base_seed + i;
        }

        if (batch_n > 1 || sbs > 1) {
            fprintf(stderr, "[Ace-Synth] Group %d: %d track(s)\n", ri, sbs);
        }

        std::vector<AceAudio> group_audio(sbs);
        if (ace_synth_generate(ctx, group.data(), src_interleaved, src_len, ref_interleaved, ref_len, sbs,
                               group_audio.data()) != 0) {
            fprintf(stderr, "[Ace-Synth] ERROR: generation failed for group %d\n", ri);
            for (auto & a : all_audio) {
                ace_audio_free(&a);
            }
            for (auto & a : group_audio) {
                ace_audio_free(&a);
            }
            free(src_interleaved);
            free(ref_interleaved);
            ace_synth_free(ctx);
            return 1;
        }

        for (int i = 0; i < sbs; i++) {
            all_audio.push_back(group_audio[i]);
            all_basenames.push_back(basenames[ri]);
            all_synth_indices.push_back(i);
        }
    }

    // Write output files
    for (int b = 0; b < (int) all_audio.size(); b++) {
        if (!all_audio[b].samples) {
            continue;
        }
        const char * ext = output_wav ? ".wav" : ".mp3";
        char         out_path[1024];
        snprintf(out_path, sizeof(out_path), "%s%d%s", all_basenames[b].c_str(), all_synth_indices[b], ext);
        if (!audio_write(out_path, all_audio[b].samples, all_audio[b].n_samples, 48000, mp3_kbps, wav_format)) {
            fprintf(stderr, "[Ace-Synth Batch%d] FATAL: failed to write %s\n", b, out_path);
        }
        ace_audio_free(&all_audio[b]);
    }

    free(src_interleaved);
    free(ref_interleaved);
    ace_synth_free(ctx);
    fprintf(stderr, "[Ace-Synth] All done\n");
    return 0;
}
