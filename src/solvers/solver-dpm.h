#pragma once
// DPM++ 3M solver, third order Adams Bashforth, 1 NFE per step.
//
// Uses two previous velocities for the third order correction.
// Bootstrap: step 0 = Euler, step 1 = AB2, step 2+ = AB3.
//
// Coefficients:
//   AB3: v_eff = (23/12) * v_curr + (16/12) * (-v_prev) + (5/12) * v_prev_prev
//   AB2: v_eff = 1.5    * v_curr + 0.5    * (-v_prev)

#include "solver-interface.h"

#include <cstring>

static void solver_dpm3m_step(float *       xt,
                              const float * vt,
                              float         t_curr,
                              float         t_prev,
                              int           n,
                              SolverState & state,
                              SolverModelFn /*model_fn*/,
                              float * /*vt_buf*/) {
    float dt = t_curr - t_prev;

    if (!state.prev_vt.empty() && !state.prev_prev_vt.empty()) {
        // Third order step using two stored velocities.
        const float c0 = 23.0f / 12.0f;
        const float c1 = 16.0f / 12.0f;
        const float c2 = 5.0f / 12.0f;
        for (int i = 0; i < n; i++) {
            float v_eff = c0 * vt[i] - c1 * state.prev_vt[i] + c2 * state.prev_prev_vt[i];
            xt[i] -= v_eff * dt;
        }
    } else if (!state.prev_vt.empty()) {
        // Second order step using one stored velocity.
        for (int i = 0; i < n; i++) {
            float v_eff = 1.5f * vt[i] - 0.5f * state.prev_vt[i];
            xt[i] -= v_eff * dt;
        }
    } else {
        // First step bootstrap with Euler.
        for (int i = 0; i < n; i++) {
            xt[i] -= vt[i] * dt;
        }
    }

    // Shift history: prev_prev <- prev, prev <- current.
    state.prev_prev_vt = std::move(state.prev_vt);
    state.prev_vt.assign(vt, vt + n);
}
