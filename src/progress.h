#pragma once
// Progress + cancel callback for the generation pipelines.
//
// This widens the old `bool (*cancel)(void *)` + `void * cancel_data` pair (#18):
// the same callback now reports where the run is (stage, step, total) AND cancels
// by returning true. Every pipeline that took the cancel pair takes an
// AceProgress by value instead, defaulted to {} (no callback = no progress, never
// cancel), so callers that passed nothing are unchanged.

enum AceStage {
    ACE_STAGE_LM,          // LM planning + code generation
    ACE_STAGE_DIT,         // DiT flow-matching sampler
    ACE_STAGE_VAE_ENCODE,  // VAE tiled encode (reference audio -> latents)
    ACE_STAGE_VAE_DECODE,  // VAE tiled decode (latents -> audio)
    ACE_STAGE_MP3,         // MP3 output encode
};

// fn is called at each pipeline poll with the current (stage, step, total) and
// returns true to cancel the run. step is 0-based; total is the loop's iteration
// count. The loop's first iteration (step 0) reports before the first unit of
// work, so a UI can size a segmented bar. fn == nullptr means no progress and
// never cancel.
//
// total is an upper bound, not a promise: an LM stage reports total =
// max_new_tokens / max_tokens but stops early once every sequence hits EOS
// (usually far below total), so an LM bar should treat it as a ceiling and expect
// the stage to end before reaching it.
//
// A call with step < 0 is a bare cancel poll (see ace_cancelled), NOT a progress
// report: a progress-recording callback should honour its cancel return but skip
// it for sizing/advancing. It is used before a loop (to honour a cancel that
// landed before step 0 -- an empty loop therefore emits no step-0 report, only
// this poll), where there is no meaningful iteration index (MP3 encode), and in
// the VAE error-vs-cancel disambiguation -- so those never inject bogus steps
// into a stage's stream.
//
// Cancel must be level-triggered: once the run is cancelled, fn must keep
// returning true (the server reads a std::atomic<bool> flag, which is). The
// disambiguation re-polls rely on that.
//
// A stage may run more than once in a single request, each run re-emitting its
// step-0 sizing report followed by 0..N -- so a consumer should treat step 0 as
// the start of a (repeatable) pass, not assume a stage appears once. VAE encode
// runs for the source and the timbre reference in Cover tasks; VAE decode runs
// once per batch item; and DiT and LM repeat once per group in batched synthesis.
//
// Thread-safety: fn is always called from a single thread. The pipelines poll it
// sequentially, and MP3 encode -- though fork-join -- polls cancel only from its
// single-threaded post-join point, never from a worker. So fn need not be
// thread-safe.
struct AceProgress {
    bool (*fn)(void * data, AceStage stage, int step, int total) = nullptr;
    void * data                                                  = nullptr;
};

// The single place the pipelines poll for progress: reports (stage, step, total)
// and returns true if the callback asked to cancel.
static inline bool ace_progress(const AceProgress & p, AceStage stage, int step, int total) {
    return p.fn && p.fn(p.data, stage, step, total);
}

// A cancel poll with nothing to report -- for sites with no iteration index (MP3
// encode) or that re-check cancel after a failing return (VAE encode/decode) to
// tell a cancel from an error. Passes step/total = -1 so a progress consumer can
// tell it from a real report; returns true if cancelled.
static inline bool ace_cancelled(const AceProgress & p, AceStage stage) {
    return p.fn && p.fn(p.data, stage, -1, -1);
}
