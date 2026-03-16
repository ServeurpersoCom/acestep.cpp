#pragma once
//
// pipeline-lm.h - ACE-Step LM pipeline
//
// Loads Qwen3 causal LM once, then enriches requests:
// text caption -> metadata + lyrics + audio codes.
//

#include "request.h"

struct AceLm;

struct AceLmParams {
    const char * model_path;  // LM GGUF (required)
    int          max_seq;     // KV cache length (default: 8192)
    int          max_batch;   // max batch_size for generate (default: 4)
    bool         use_fsm;     // constrained decoding (default: true)
    bool         use_fa;      // flash attention (default: true)
};

void ace_lm_default_params(AceLmParams * p);

// Load model, tokenizer, FSM. NULL on failure.
AceLm * ace_lm_load(const AceLmParams * params);

// Enrich request with metadata, lyrics, audio codes.
// out[batch_size] allocated by caller, filled with enriched copies of req.
// dump_logits/dump_tokens: debug output paths (NULL to disable).
// Returns 0 on success.
int ace_lm_generate(AceLm *            ctx,
                    const AceRequest * req,
                    int                batch_size,
                    AceRequest *       out,
                    const char *       dump_logits,
                    const char *       dump_tokens);

void ace_lm_free(AceLm * ctx);
