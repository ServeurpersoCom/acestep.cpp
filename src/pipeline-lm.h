#pragma once
// pipeline-lm.h: ACE-Step LM pipeline
//
// A lightweight context bound to a ModelStore. On each generate call the
// pipeline acquires the Qwen3 LM, BPE tokenizer and FSM template from the
// store and releases the LM through RAII before returning.

#include "request.h"
#include "task-types.h"

struct AceLm;
struct ModelStore;

struct AceLmParams {
    const char * model_path;     // LM GGUF (required)
    int          max_seq;        // KV cache length (default: 8192)
    int          max_batch;      // max lm_batch_size for generate (default: 4)
    bool         use_fsm;        // constrained decoding (default: true)
    bool         use_fa;         // flash attention (default: true)
    bool         use_batch_cfg;  // batch cond+uncond in one forward (default: true)
    bool         clamp_fp16;     // clamp hidden states to FP16 range (default: false)
};

void ace_lm_default_params(AceLmParams * p);

// Build a lightweight LM context bound to a ModelStore. The GPU model is
// acquired per request, never owned by the context. NULL on invalid input.
AceLm * ace_lm_load(ModelStore * store, const AceLmParams * params);

// Enrich request with metadata, lyrics, audio codes.
// out[lm_batch_size] allocated by caller, filled with enriched copies of req.
// mode: LM_MODE_GENERATE (full), LM_MODE_INSPIRE (no codes), LM_MODE_FORMAT (no codes).
// dump_logits/dump_tokens: debug output paths (NULL to disable).
// cancel/cancel_data: abort callback, polled between tokens. NULL = never cancel.
// Returns 0 on success, -1 on error or cancellation.
int ace_lm_generate(AceLm *            ctx,
                    const AceRequest * req,
                    int                lm_batch_size,
                    AceRequest *       out,
                    const char *       dump_logits,
                    const char *       dump_tokens,
                    bool (*cancel)(void *) = nullptr,
                    void * cancel_data     = nullptr,
                    int    mode            = LM_MODE_GENERATE);

void ace_lm_free(AceLm * ctx);
