// ace-understand.cpp: audio understanding CLI (thin wrapper)
//
// Audio -> VAE encode -> FSQ tokenize -> LM understand -> metadata + lyrics
//
// Output: request JSON with metadata + lyrics, reusable as ace-lm or ace-synth input.

#include "audio-io.h"
#include "model-registry.h"
#include "model-store.h"
#include "pipeline-understand.h"
#include "request.h"
#include "version.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static void usage(const char * prog) {
    AceUnderstandParams d;
    ace_understand_default_params(&d);

    fprintf(stderr, "acestep.cpp %s\n\n", ACE_VERSION);
    fprintf(stderr,
            "Usage: %s --models <dir> --src-audio <file> [--request <json>] [options]\n"
            "\n"
            "Required:\n"
            "  --models <dir>          Directory of GGUF model files\n"
            "  --src-audio <file>      Source audio (WAV or MP3, any sample rate)\n"
            "\n"
            "Optional:\n"
            "  --request <json>        Request JSON carrying model selection and\n"
            "                          sampling params (lm_model, synth_model,\n"
            "                          lm_temperature, lm_top_p, lm_top_k)\n"
            "\n"
            "When no --request is given, understand defaults apply\n"
            "(temperature 0.3, top_p disabled).\n"
            "\n"
            "Output:\n"
            "  -o <json>               Output JSON (default: stdout summary)\n"
            "\n"
            "Memory control:\n"
            "  --vae-chunk <N>         Latent frames per tile (default: %d)\n"
            "  --vae-overlap <N>       Overlap frames per side (default: %d)\n"
            "\n"
            "Debug:\n"
            "  --max-seq <N>           KV cache size (default: %d)\n"
            "  --no-fsm                Disable FSM constrained decoding\n"
            "  --no-fa                 Disable flash attention\n"
            "  --dump <dir>            Dump tok_latents + tok_codes (skip LM)\n",
            prog, d.vae_chunk, d.vae_overlap, d.max_seq);
}

int main(int argc, char ** argv) {
    const char * models_dir     = NULL;
    const char * src_audio_path = NULL;
    const char * request_path   = NULL;
    const char * output_path    = NULL;
    const char * dump_dir       = NULL;

    AceUnderstandParams params;
    ace_understand_default_params(&params);

    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--models") && i + 1 < argc) {
            models_dir = argv[++i];
        } else if (!strcmp(argv[i], "--src-audio") && i + 1 < argc) {
            src_audio_path = argv[++i];
        } else if (!strcmp(argv[i], "--request") && i + 1 < argc) {
            request_path = argv[++i];
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

    if (!models_dir) {
        fprintf(stderr, "[CLI] ERROR: --models required\n");
        usage(argv[0]);
        return 1;
    }
    if (!src_audio_path) {
        fprintf(stderr, "[CLI] ERROR: --src-audio required\n");
        usage(argv[0]);
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
            return 1;
        }
        request_dump(&req, stderr);
    }

    // Scan the registry and resolve model paths. Empty lm_model / synth_model
    // fall to the first entry of their bucket, matching server default behavior.
    ModelRegistry registry;
    if (!registry_scan(&registry, models_dir)) {
        fprintf(stderr, "[Ace-Understand] FATAL: cannot scan --models %s\n", models_dir);
        return 1;
    }

    if (registry.dit.empty() || registry.vae.empty()) {
        fprintf(stderr, "[Ace-Understand] FATAL: understand pipeline needs DiT and VAE models under %s\n", models_dir);
        return 1;
    }
    if (!dump_dir && registry.lm.empty()) {
        fprintf(stderr, "[Ace-Understand] FATAL: understand pipeline needs an LM model under %s\n", models_dir);
        return 1;
    }

    const ModelEntry * lm_entry =
        dump_dir ? NULL : (req.lm_model.empty() ? &registry.lm[0] : registry_find(registry.lm, req.lm_model.c_str()));
    if (!dump_dir && !lm_entry) {
        fprintf(stderr, "[Ace-Understand] FATAL: lm_model '%s' not found in registry\n", req.lm_model.c_str());
        return 1;
    }
    const ModelEntry * dit_entry =
        req.synth_model.empty() ? &registry.dit[0] : registry_find(registry.dit, req.synth_model.c_str());
    if (!dit_entry) {
        fprintf(stderr, "[Ace-Understand] FATAL: synth_model '%s' not found in registry\n", req.synth_model.c_str());
        return 1;
    }
    const ModelEntry * vae_entry = req.vae.empty() ? &registry.vae[0] : registry_find(registry.vae, req.vae.c_str());
    if (!vae_entry) {
        fprintf(stderr, "[Ace-Understand] FATAL: vae '%s' not found in registry\n", req.vae.c_str());
        return 1;
    }

    params.model_path = lm_entry ? lm_entry->path.c_str() : NULL;
    params.dit_path   = dit_entry->path.c_str();
    params.vae_path   = vae_entry->path.c_str();
    params.dump_dir   = dump_dir;

    // load pipeline
    ModelStore *    store = store_create(EVICT_STRICT);
    AceUnderstand * ctx   = ace_understand_load(store, &params);
    if (!ctx) {
        store_free(store);
        return 1;
    }

    // Read and resample audio to 48kHz stereo
    int     T_audio = 0;
    float * planar  = audio_read_48k(src_audio_path, &T_audio);
    if (!planar) {
        fprintf(stderr, "[Ace-Understand] FATAL: cannot read %s\n", src_audio_path);
        ace_understand_free(ctx);
        store_free(store);
        return 1;
    }
    fprintf(stderr, "[Ace-Understand] %.2fs @ 48kHz\n", (float) T_audio / 48000.0f);

    float * src_interleaved = audio_planar_to_interleaved(planar, T_audio);
    free(planar);
    int src_len = T_audio;

    // run understand pipeline. CLI feeds raw audio: NULL/0 for latents,
    // NULL/NULL for capture, NULL/NULL for cancel. The server reuses the
    // same entry point with the full latent IO surface.
    request_resolve_lm_seed(&req);
    AceRequest out;
    int rc = ace_understand_generate(ctx, src_interleaved, src_len, nullptr, 0, &req, &out, nullptr, nullptr, nullptr,
                                     nullptr);
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
