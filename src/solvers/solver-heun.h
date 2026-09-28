#pragma once
// solver-heun.h: Heun's method (2nd order, 2 NFE per step)
//
// Predictor-corrector:
//   1. Euler predictor:  x_pred = x_t - v_t * dt
//   2. Evaluate:         v_pred = model(x_pred, t_prev)
//   3. Corrector:        x_next = x_t - 0.5 * (v_t + v_pred) * dt
//
// Matches Python solvers.py heun_step()

#include "solver-interface.h"

static void solver_heun_step(float *       xt,
                             const float * vt,
                             float         t_curr,
                             float         t_prev,
                             int           n,
                             SolverState & state,
                             SolverModelFn model_fn,
                             float *       vt_buf) {
    float dt = t_curr - t_prev;

    if (!model_fn) {
        // Fallback to Euler if no model callback
        for (int i = 0; i < n; i++) {
            xt[i] -= vt[i] * dt;
        }
        return;
    }

    // Ensure scratch buffer is allocated
    if ((int) state.xt_scratch.size() < n) {
        state.xt_scratch.resize(n);
    }
    float * x_pred = state.xt_scratch.data();

    // Save k1 - vt and vt_buf alias the same memory, so model_fn will
    // overwrite k1 when it writes k2.  We must preserve it for the average.
    if ((int) state.prev_vt.size() < n) {
        state.prev_vt.resize(n);
    }
    memcpy(state.prev_vt.data(), vt, n * sizeof(float));

    // Predictor: Euler step
    for (int i = 0; i < n; i++) {
        x_pred[i] = xt[i] - vt[i] * dt;
    }

    // Evaluate velocity at predicted point - result goes into vt_buf
    model_fn(x_pred, t_prev);

    // Corrector: average of k1 and k2
    for (int i = 0; i < n; i++) {
        xt[i] -= 0.5f * (state.prev_vt[i] + vt_buf[i]) * dt;
    }
}
