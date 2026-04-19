// ace-understand.cpp: audio understanding CLI (thin wrapper)
//
// Audio -> VAE encode -> FSQ tokenize -> LM understand -> metadata + lyrics
// Or:  audio_codes from JSON -> LM understand -> metadata + lyrics
//
// Output: request JSON with metadata + lyrics, reusable as ace-lm input.
//
// Usage: ./ace-understand --src-audio <wav> --lm <gguf> --dit <gguf> --vae <gguf>
//        ./ace-understand --request <json> --lm <gguf>
// See --help for full option list.

#include "audio-io.h"
#include "model-store.h"
#include "pipeline-understand.h"
#include "request.h"
#include "version.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static void usage(const char * prog) {
    fprintf(stderr, "acestep.cpp %s\n\n", ACE_VERSION);
    fprintf(stderr,
            "Usage: %s [--src-audio <file> --dit <gguf> --vae <gguf> | --request <json>] --lm <gguf>\n"
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
            "  --lm <gguf>             5Hz LM GGUF file\n"
            "\n"
            "Output:\n"
            "  -o <json>               Output JSON (default: stdout summary)\n"
            "\n"
            "Sampling params (lm_temperature, lm_top_p, lm_top_k) come from the\n"
            "request JSON. Without --request, understand defaults apply\n"
            "(temperature=0.3, top_p disabled).\n"
            "\n"
            "Memory control:\n"
            "  --vae-chunk <N>         Latent frames per tile (default: 256)\n"
            "  --vae-overlap <N>       Overlap frames per side (default: 64)\n"
            "\n"
            "Debug:\n"
            "  --max-seq <N>           KV cache size (default: 8192)\n"
            "  --no-fsm                Disable FSM constrained decoding\n"
            "  --no-fa                 Disable flash attention\n"
            "  --dump <dir>            Dump tok_latents + tok_codes (skip LM)\n",
            prog);
}

int main(int argc, char ** argv) {
    const char * src_audio_path = NULL;
    const char * dit_gguf       = NULL;
    const char * vae_gguf       = NULL;
    const char * request_path   = NULL;
    const char * model_path     = NULL;
    const char * output_path    = NULL;
    const char * dump_dir       = NULL;

    AceUnderstandParams params;
    ace_understand_default_params(&params);

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
        } else if (!strcmp(argv[i], "--lm") && i + 1 < argc) {
            model_path = argv[++i];
        } else if (!strcmp(argv[i], "-o") && i + 1 < argc) {
            output_path = argv[++i];
        } else if (!strcmp(argv[i], "--dump") && i + 1 < argc) {
            dump_dir = argv[++i];
        } else if (!strcmp(argv[i], "--max-seq") && i + 1 < argc) {
            params.max_seq = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--vae-chunk") && i + 1 < argc) {
            params.vae_chunk = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--vae-overlap") && i + 1 < argc) {
            params.vae_overlap = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--no-fsm")) {
            params.use_fsm = false;
        } else if (!strcmp(argv[i], "--no-fa")) {
            params.use_fa = false;
        } else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (!model_path && !dump_dir) {
        fprintf(stderr, "[CLI] ERROR: --lm required (or use --dump for tok-only)\n");
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

    // set model paths
    params.model_path = model_path;
    params.dit_path   = dit_gguf;
    params.vae_path   = vae_gguf;
    params.dump_dir   = dump_dir;

    // load pipeline
    ModelStore *    store = store_create(EVICT_STRICT);
    AceUnderstand * ctx   = ace_understand_load(store, &params);
    if (!ctx) {
        store_free(store);
        return 1;
    }

    // Parse request JSON (if provided). Sampling params come from JSON.
    // When no JSON, understand defaults apply (temperature=0.3 for transcription).
    AceRequest req;
    request_init(&req);
    req.lm_temperature = 0.3f;  // understand default: lower than generation
    req.lm_top_p       = 1.0f;  // understand default: no nucleus sampling
    if (request_path) {
        if (!request_parse(&req, request_path)) {
            ace_understand_free(ctx);
            store_free(store);
            return 1;
        }
        request_dump(&req, stderr);
    }

    // Read and resample audio to 48kHz stereo
    float * src_interleaved = NULL;
    int     src_len         = 0;
    if (src_audio_path) {
        int     T_audio = 0;
        float * planar  = audio_read_48k(src_audio_path, &T_audio);
        if (!planar) {
            fprintf(stderr, "[Ace-Understand] FATAL: cannot read %s\n", src_audio_path);
            ace_understand_free(ctx);
            store_free(store);
            return 1;
        }
        fprintf(stderr, "[Ace-Understand] %.2fs @ 48kHz\n", (float) T_audio / 48000.0f);

        src_interleaved = audio_planar_to_interleaved(planar, T_audio);
        free(planar);
        src_len = T_audio;
    }

    // run understand pipeline
    AceRequest out;
    int        rc = ace_understand_generate(ctx, src_interleaved, src_len, &req, &out, NULL, NULL);
    free(src_interleaved);
    ace_understand_free(ctx);
    store_free(store);

    if (rc != 0) {
        return 1;
    }

    // write output JSON
    if (output_path) {
        request_write(&out, output_path);
    }
    return 0;
}
