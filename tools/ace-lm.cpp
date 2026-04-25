// ace-lm.cpp: ACE-Step LLM CLI
// Thin wrapper: parses args, scans the model registry, calls pipeline-lm,
// writes output files. The model to use comes from request.lm_model, the
// registry resolves it to a GGUF path under --models <dir>.

#include "model-registry.h"
#include "model-store.h"
#include "pipeline-lm.h"
#include "request.h"
#include "task-types.h"
#include "version.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void usage(const char * prog) {
    AceLmParams d;
    ace_lm_default_params(&d);

    fprintf(stderr, "acestep.cpp %s\n\n", ACE_VERSION);
    fprintf(stderr,
            "Usage: %s --models <dir> --request <json> [options]\n"
            "\n"
            "Required:\n"
            "  --models <dir>         Directory of GGUF model files\n"
            "  --request <json>       Input request JSON (carries lm_model)\n"
            "\n"
            "Debug:\n"
            "  --max-seq <N>          KV cache size (default: %d)\n"
            "  --no-fsm               Disable FSM constrained decoding\n"
            "  --no-fa                Disable flash attention\n"
            "  --no-batch-cfg         Split CFG into two separate forwards\n"
            "  --clamp-fp16           Clamp hidden states to FP16 range\n"
            "  --dump-logits <path>   Dump prefill logits (binary f32)\n"
            "  --dump-tokens <path>   Dump prompt token IDs (CSV)\n",
            prog, d.max_seq);
}

int main(int argc, char ** argv) {
    AceLmParams params;
    ace_lm_default_params(&params);

    const char * models_dir   = NULL;
    const char * request_path = NULL;
    const char * dump_logits  = NULL;
    const char * dump_tokens  = NULL;

    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--models") && i + 1 < argc) {
            models_dir = argv[++i];
        } else if (!strcmp(argv[i], "--request") && i + 1 < argc) {
            request_path = argv[++i];
        } else if (!strcmp(argv[i], "--max-seq") && i + 1 < argc) {
            params.max_seq = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--no-fsm")) {
            params.use_fsm = false;
        } else if (!strcmp(argv[i], "--no-fa")) {
            params.use_fa = false;
        } else if (!strcmp(argv[i], "--no-batch-cfg")) {
            params.use_batch_cfg = false;
        } else if (!strcmp(argv[i], "--clamp-fp16")) {
            params.clamp_fp16 = true;
        } else if (!strcmp(argv[i], "--dump-logits") && i + 1 < argc) {
            dump_logits = argv[++i];
        } else if (!strcmp(argv[i], "--dump-tokens") && i + 1 < argc) {
            dump_tokens = argv[++i];
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
    if (!request_path) {
        fprintf(stderr, "[CLI] ERROR: --request required\n");
        usage(argv[0]);
        return 1;
    }

    // Parse input request
    AceRequest req;
    if (!request_parse(&req, request_path)) {
        return 1;
    }
    request_dump(&req, stderr);

    // Scan the registry and resolve lm_model. Missing or empty lm_model falls
    // to the first LM in the registry, matching server default behavior.
    ModelRegistry registry;
    if (!registry_scan(&registry, models_dir)) {
        fprintf(stderr, "[Ace-LM] FATAL: cannot scan --models %s\n", models_dir);
        return 1;
    }
    if (registry.lm.empty()) {
        fprintf(stderr, "[Ace-LM] FATAL: no LM models found under %s\n", models_dir);
        return 1;
    }
    const ModelEntry * lm_entry =
        req.lm_model.empty() ? &registry.lm[0] : registry_find(registry.lm, req.lm_model.c_str());
    if (!lm_entry) {
        fprintf(stderr, "[Ace-LM] FATAL: lm_model '%s' not found in registry\n", req.lm_model.c_str());
        return 1;
    }
    params.model_path = lm_entry->path.c_str();

    // lm_batch_size from JSON (clamped to 1..9)
    int lm_batch_size = req.lm_batch_size;
    if (lm_batch_size < 1) {
        lm_batch_size = 1;
    } else if (lm_batch_size > 9) {
        fprintf(stderr, "[Ace-LM] WARNING: lm_batch_size %d clamped to 9\n", lm_batch_size);
        lm_batch_size = 9;
    }

    // Resolve lm_mode string to integer mode used by ace_lm_generate.
    int mode;
    if (req.lm_mode == LM_MODE_NAME_GENERATE) {
        mode = LM_MODE_GENERATE;
    } else if (req.lm_mode == LM_MODE_NAME_INSPIRE) {
        mode = LM_MODE_INSPIRE;
    } else if (req.lm_mode == LM_MODE_NAME_FORMAT) {
        mode = LM_MODE_FORMAT;
    } else {
        fprintf(stderr, "[Ace-LM] FATAL: invalid lm_mode '%s' (use: generate, inspire, format)\n", req.lm_mode.c_str());
        return 1;
    }

    // Load model (KV cache sized for request batch)
    params.max_batch   = lm_batch_size;
    ModelStore * store = store_create(EVICT_STRICT);
    AceLm *      ctx   = ace_lm_load(store, &params);
    if (!ctx) {
        store_free(store);
        return 1;
    }

    // Generate
    request_resolve_lm_seed(&req);
    std::vector<AceRequest> out(lm_batch_size);
    if (ace_lm_generate(ctx, &req, lm_batch_size, out.data(), dump_logits, dump_tokens, NULL, NULL, mode) != 0) {
        ace_lm_free(ctx);
        store_free(store);
        return 1;
    }

    // Write output files: request.json -> request0.json, request1.json, ...
    std::string base(request_path);
    std::string ext = ".json";
    size_t      dot = base.rfind('.');
    if (dot != std::string::npos) {
        ext  = base.substr(dot);
        base = base.substr(0, dot);
    }
    for (int b = 0; b < lm_batch_size; b++) {
        char path[512];
        snprintf(path, sizeof(path), "%s%d%s", base.c_str(), b, ext.c_str());
        request_write(&out[b], path);
    }

    ace_lm_free(ctx);
    store_free(store);
    return 0;
}
