// ace-lm.cpp: ACE-Step LLM CLI
// Thin wrapper: parses args, calls pipeline-lm, writes output files.

#include "pipeline-lm.h"
#include "request.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void usage(const char * prog) {
    fprintf(stderr,
            "Usage: %s --request <json> --model <gguf> [options]\n"
            "\n"
            "Required:\n"
            "  --request <json>       Input request JSON\n"
            "  --model <gguf>         Model GGUF file\n"
            "\n"
            "Batch:\n"
            "  --batch <N>            Batch N sequences (default: 1)\n"
            "\n"
            "Output naming: input.json -> input0.json, input1.json, ... (last digit = batch index)\n"
            "\n"
            "Debug:\n"
            "  --max-seq <N>          KV cache size (default: 8192)\n"
            "  --no-fsm               Disable FSM constrained decoding\n"
            "  --no-fa                Disable flash attention\n"
            "  --dump-logits <path>   Dump prefill logits (binary f32)\n"
            "  --dump-tokens <path>   Dump prompt token IDs (CSV)\n",
            prog);
}

int main(int argc, char ** argv) {
    AceLmParams params;
    ace_lm_default_params(&params);

    const char * request_path = NULL;
    const char * dump_logits  = NULL;
    const char * dump_tokens  = NULL;
    int          batch_size   = 1;

    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) {
            params.model_path = argv[++i];
        } else if (!strcmp(argv[i], "--request") && i + 1 < argc) {
            request_path = argv[++i];
        } else if (!strcmp(argv[i], "--max-seq") && i + 1 < argc) {
            params.max_seq = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--batch") && i + 1 < argc) {
            batch_size = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--no-fsm")) {
            params.use_fsm = false;
        } else if (!strcmp(argv[i], "--no-fa")) {
            params.use_fa = false;
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

    if (!params.model_path) {
        fprintf(stderr, "[CLI] ERROR: --model required\n");
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

    // Load model
    params.max_batch = batch_size;
    AceLm * ctx      = ace_lm_load(&params);
    if (!ctx) {
        return 1;
    }

    // Generate
    std::vector<AceRequest> out(batch_size);
    if (ace_lm_generate(ctx, &req, batch_size, out.data(), dump_logits, dump_tokens) != 0) {
        ace_lm_free(ctx);
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
    for (int b = 0; b < batch_size; b++) {
        char path[512];
        snprintf(path, sizeof(path), "%s%d%s", base.c_str(), b, ext.c_str());
        request_write(&out[b], path);
    }

    ace_lm_free(ctx);
    return 0;
}
