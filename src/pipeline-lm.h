#pragma once
// pipeline-lm.h: ACE-Step LM pipeline
//
// A lightweight context bound to a ModelStore. On each generate call the
// pipeline acquires the Qwen3 LM, BPE tokenizer and FSM template from the
// store and releases the LM through RAII before returning.

#include "progress.h"
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

// Enrich n_req requests with metadata, lyrics, audio codes in one GPU batch.
// Requests may differ in caption, lyrics, metadata, duration and seeds, but
// must agree on the flags steering the batch: audio_codes present, lyrics
// present, metadata complete, use_cot_caption. Sampling configuration comes
// from reqs[0]. seed and lm_seed are resolved by the caller and pass through
// to the matching output. Identical prompts prefill once and share KV copies,
// so replicating one request with different seeds keeps the old fast path.
// out[n_req] allocated by caller, filled with enriched copies of each request.
// mode: LM_MODE_GENERATE (full), LM_MODE_INSPIRE (no codes), LM_MODE_FORMAT (no codes).
// dump_logits/dump_tokens: debug output paths (NULL to disable).
// progress: abort + progress callback, polled between tokens (ACE_STAGE_LM).
// Default {} = never cancel, no progress.
// Returns 0 on success, -1 on error or cancellation.
int ace_lm_generate(AceLm *            ctx,
                    const AceRequest * reqs,
                    int                n_req,
                    AceRequest *       out,
                    const char *       dump_logits,
                    const char *       dump_tokens,
                    AceProgress        progress = {},
                    int                mode     = LM_MODE_GENERATE);

void ace_lm_free(AceLm * ctx);

// Read the LM ModelKey the context builds for store_require_lm. Used by
// test-model-store to verify both ace_lm and ace_understand resolve to the
// same LM instance when their params are propagated consistently.
struct ModelKey;
const ModelKey * ace_lm_lm_key(const AceLm * ctx);
