#pragma once
// SDE Ancestral solver, 1 NFE plus Philox renoise per step.
// Predicts x0 then renoises with fresh Philox samples:
//   x0     = xt - vt * t_curr
//   x_next = t_prev * noise + (1.0 - t_prev) * x0
//
// Per batch seeds in SolverState drive bit for bit reproducible trajectories
// (Philox seeded with seed_b + step + 1). The caller must populate state.seeds,
// state.batch_n and state.n_per before invoking this solver, otherwise the
// call is a programming error.

#include "../philox.h"
#include "solver-interface.h"

#include <vector>

static void solver_sde_step(float *       xt,
                            const float * vt,
                            float         t_curr,
                            float         t_prev,
                            int           n,
                            SolverState & state,
                            SolverModelFn /*model_fn*/,
                            float * /*vt_buf*/) {
    int batch_n = state.batch_n;
    int n_per   = state.n_per;

    for (int b = 0; b < batch_n; b++) {
        // Fresh noise per batch item, seeded with (seed_b + step + 1).
        std::vector<float> fresh(n_per);
        philox_randn(state.seeds[b] + state.step_index + 1, fresh.data(), n_per, true);

        for (int i = 0; i < n_per; i++) {
            int   idx = b * n_per + i;
            float x0  = xt[idx] - vt[idx] * t_curr;
            xt[idx]   = t_prev * fresh[i] + (1.0f - t_prev) * x0;
        }
    }
}
