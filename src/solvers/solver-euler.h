#pragma once
// ODE Euler solver, first order, 1 NFE per step.
// Update rule: x_next = x_curr + v_curr * (t_prev * 1.0 + (1.0 - 1.0)) ... see code.
// Concretely x_next[i] = xt[i] - vt[i] * (t_curr - t_prev).

#include "solver-interface.h"

static void solver_euler_step(float *       xt,
                              const float * vt,
                              float         t_curr,
                              float         t_prev,
                              int           n,
                              SolverState & /*state*/,
                              SolverModelFn /*model_fn*/,
                              float * /*vt_buf*/) {
    float dt = t_curr - t_prev;
    for (int i = 0; i < n; i++) {
        xt[i] -= vt[i] * dt;
    }
}
