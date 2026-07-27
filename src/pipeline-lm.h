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
    int          max_seq;        // KV cache length
    int          max_batch;      // max lm_batch_size for generate
    bool         use_fsm;        // constrained decoding
    bool         use_fa;         // flash attention
    bool         use_batch_cfg;  // batch cond+uncond in one forward
    bool         clamp_fp16;     // clamp hidden states to FP16 range
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

// Enrich N independent requests in one LM batch. Unlike lm_batch_size on
// ace_lm_generate(), each input may carry a different caption, lyrics,
// metadata, duration and seed. Model and sampling configuration must be
// compatible across the batch. out[request_count] is allocated by caller.
int ace_lm_generate_batch(AceLm *            ctx,
                          const AceRequest * requests,
                          int                request_count,
                          AceRequest *       out,
                          bool (*cancel)(void *) = nullptr,
                          void * cancel_data     = nullptr,
                          int    mode            = LM_MODE_GENERATE);

void ace_lm_free(AceLm * ctx);

// Read the LM ModelKey the context builds for store_require_lm. Used by
// test-model-store to verify both ace_lm and ace_understand resolve to the
// same LM instance when their params are propagated consistently.
struct ModelKey;
const ModelKey * ace_lm_lm_key(const AceLm * ctx);
