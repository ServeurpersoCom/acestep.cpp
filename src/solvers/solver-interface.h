#pragma once
// Universal solver contract for ACE Step flow matching.
//
// A solver is a pure function of (xt, vt, t_curr, t_prev, n, state, model_fn, vt_buf).
// The sampler loop resolves the solver by name and calls it once per step.
// xt is modified in place to xt_next.
//
// Single evaluation solvers (Euler, SDE, DPM++ 3M, STORK4) ignore model_fn.
// Multi evaluation solvers (Heun, RK4, ...) added later use model_fn to re evaluate
// the DiT at intermediate timesteps.

#include <cstdint>
#include <functional>
#include <vector>

// Callback that evaluates the DiT model at (xt_in, t_val) and writes the
// CFG processed velocity into the sampler vt buffer (vt_buf, passed to step_fn).
// Empty for 1 NFE solvers.
using SolverModelFn = std::function<void(const float *, float)>;

// Persistent solver state, lives across steps within a single generation.
// Only the fields required by registered solvers are present.
struct SolverState {
    int step_index = 0;

    // DPM++ 3M: velocity history over the last two steps.
    std::vector<float> prev_vt;
    std::vector<float> prev_prev_vt;

    // STORK4: velocity history with associated step sizes (last 3 records).
    struct VelocityRecord {
        std::vector<float> vt;
        float              dt;
    };

    std::vector<VelocityRecord> velocity_history;
    int                         stork_substeps = 10;

    // SDE: per batch seeds for Philox renoising.
    const int64_t * seeds   = nullptr;
    int             batch_n = 1;
    int             n_per   = 0;
};

// xt:       [n] current latent, modified in place to xt_next.
// vt:       [n] velocity at (xt, t_curr), already CFG/APG processed.
// t_curr:   current timestep (1.0 = pure noise).
// t_prev:   next timestep (0.0 = clean signal).
// n:        total elements across all batches (batch_n * T * Oc).
// state:    mutable solver state, persists across steps.
// model_fn: callback for re evaluation, may be empty for 1 NFE solvers.
// vt_buf:   [n] sampler vt buffer, model_fn writes results here.
using SolverStepFn = void (*)(float *       xt,
                              const float * vt,
                              float         t_curr,
                              float         t_prev,
                              int           n,
                              SolverState & state,
                              SolverModelFn model_fn,
                              float *       vt_buf);
