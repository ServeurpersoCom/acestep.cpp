#pragma once
// solver-rk4.h: Classic 4th-order Runge-Kutta (4 NFE per step)
//
// Extracted from the original dit-sampler.h RK4 branch.
// Uses 3 extra model evaluations at intermediate points:
//   k1 = v(xt, t_curr)          [already provided as vt]
//   k2 = v(xt - 0.5*dt*k1, t_mid)
//   k3 = v(xt - 0.5*dt*k2, t_mid)
//   k4 = v(xt - dt*k3, t_prev)
//   xt_next = xt - (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
//
// Matches Python solvers.py rk4_step()

#include "solver-interface.h"

static void solver_rk4_step(float *       xt,
                            const float * vt,
                            float         t_curr,
                            float         t_prev,
                            int           n,
                            SolverState & state,
                            SolverModelFn model_fn,
                            float *       vt_buf) {
    if (!model_fn) {
        // Fallback to Euler
        float dt = t_curr - t_prev;
        for (int i = 0; i < n; i++) {
            xt[i] -= vt[i] * dt;
        }
        return;
    }

    float dt    = t_curr - t_prev;
    float t_mid = (t_curr + t_prev) / 2.0f;

    // Ensure scratch buffer
    if ((int) state.xt_scratch.size() < n) {
        state.xt_scratch.resize(n);
    }
    float * xt_tmp = state.xt_scratch.data();

    // k1 = vt (already computed by sampler)
    // Save k1 since vt_buf will be overwritten
    std::vector<float> k1(vt, vt + n);

    // k2 = model(xt - 0.5*dt*k1, t_mid)
    for (int i = 0; i < n; i++) {
        xt_tmp[i] = xt[i] - 0.5f * dt * k1[i];
    }
    model_fn(xt_tmp, t_mid);
    std::vector<float> k2(vt_buf, vt_buf + n);

    // k3 = model(xt - 0.5*dt*k2, t_mid)
    for (int i = 0; i < n; i++) {
        xt_tmp[i] = xt[i] - 0.5f * dt * k2[i];
    }
    model_fn(xt_tmp, t_mid);
    std::vector<float> k3(vt_buf, vt_buf + n);

    // k4 = model(xt - dt*k3, t_prev)
    for (int i = 0; i < n; i++) {
        xt_tmp[i] = xt[i] - dt * k3[i];
    }
    model_fn(xt_tmp, t_prev);
    // k4 is now in vt_buf

    // Weighted average: xt -= (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    float w = dt / 6.0f;
    for (int i = 0; i < n; i++) {
        xt[i] -= w * (k1[i] + 2.0f * k2[i] + 2.0f * k3[i] + vt_buf[i]);
    }
}
