#pragma once
// DPM++ multistep solvers, Adams Bashforth family, 1 NFE per step.
//
// Velocity history from previous steps gives the higher order correction
// without extra model evaluations. Bootstrap with Euler on the first step.
//
// Coefficients:
//   2M:          v_eff = 1.5 * v_curr - 0.5 * v_prev
//   3M:          v_eff = (23/12) * v_curr - (16/12) * v_prev + (5/12) * v_prev_prev
//   2M adaptive: v_eff = (1 + r/2) * v_curr - (r/2) * v_prev, r = dt / dt_prev,
//                AB2 corrected for non uniform schedules (r = 1 gives 2M).

#include "solver-interface.h"

#include <cstring>

static void solver_dpm2m_step(float *       xt,
                              const float * vt,
                              float         t_curr,
                              float         t_prev,
                              int           n,
                              SolverState & state,
                              SolverModelFn /*model_fn*/,
                              float * /*vt_buf*/) {
    float dt = t_curr - t_prev;

    if (!state.prev_vt.empty()) {
        for (int i = 0; i < n; i++) {
            float v_eff = 1.5f * vt[i] - 0.5f * state.prev_vt[i];
            xt[i] -= v_eff * dt;
        }
    } else {
        for (int i = 0; i < n; i++) {
            xt[i] -= vt[i] * dt;
        }
    }

    state.prev_vt.assign(vt, vt + n);
}

static void solver_dpm2m_ada_step(float *       xt,
                                  const float * vt,
                                  float         t_curr,
                                  float         t_prev,
                                  int           n,
                                  SolverState & state,
                                  SolverModelFn /*model_fn*/,
                                  float * /*vt_buf*/) {
    float dt = t_curr - t_prev;

    if (!state.prev_vt.empty() && state.prev_dt > 0.0f) {
        float r  = dt / state.prev_dt;
        float c1 = 1.0f + r / 2.0f;
        float c0 = r / 2.0f;
        for (int i = 0; i < n; i++) {
            float v_eff = c1 * vt[i] - c0 * state.prev_vt[i];
            xt[i] -= v_eff * dt;
        }
    } else {
        for (int i = 0; i < n; i++) {
            xt[i] -= vt[i] * dt;
        }
    }

    state.prev_vt.assign(vt, vt + n);
    state.prev_dt = dt;
}

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
