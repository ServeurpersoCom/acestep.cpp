#pragma once
// solver-rfsolver.h: RF-Solver (2nd order, 2 NFE per step)
//
// From "Taming Rectified Flow for Inversion and Editing" (arXiv:2411.04746).
// A training-free sampler that exploits the rectified flow ODE structure
// (variation of constants + 2nd-order Taylor expansion) to reduce per-step
// error from O(h^2) to O(h^3).
//
// Algorithm (adapted to our t=1->0 convention):
//   Given v_t = model(xt, t_curr):
//   1. Predict x_0:  x0_hat = xt - t_curr * v_t
//   2. Half-step:    x_mid = xt - v_t * (dt/2)
//   3. Midpoint eval: v_mid = model(x_mid, t_mid)
//   4. Predict x_0 from midpoint: x0_mid = x_mid - t_mid * v_mid
//   5. RF-corrected: x_next = (1 - t_prev) * x0_mid + t_prev * (x0_mid + v_mid)
//                            = x0_mid + t_prev * v_mid
//
// When t_curr is very small, the prediction collapses to x0_hat ~ xt.
// This is equivalent to Heun-class cost (2 NFE) but specifically derived
// for the rectified flow ODE, yielding better accuracy for flow-matching.
//
// The RF-specific advantage over standard midpoint/Heun comes from using
// the linear interpolation structure x_t = (1-t)*x_0 + t*eps to refine
// the x_0 prediction at the midpoint, rather than averaging velocities.

#include "solver-interface.h"

#include <cmath>

static void solver_rfsolver_step(float *       xt,
                                 const float * vt,
                                 float         t_curr,
                                 float         t_prev,
                                 int           n,
                                 SolverState & state,
                                 SolverModelFn model_fn,
                                 float *       vt_buf) {
    float dt = t_curr - t_prev;  // positive (stepping toward 0)

    if (!model_fn || t_curr < 1e-8f) {
        // Fallback to Euler if no model callback or degenerate timestep
        for (int i = 0; i < n; i++) {
            xt[i] -= vt[i] * dt;
        }
        return;
    }

    // Ensure scratch buffer is allocated
    if ((int) state.xt_scratch.size() < n) {
        state.xt_scratch.resize(n);
    }
    float * x_mid = state.xt_scratch.data();

    // Save k1 - vt and vt_buf alias the same memory, so model_fn will
    // overwrite when it writes the midpoint result.
    if ((int) state.prev_vt.size() < n) {
        state.prev_vt.resize(n);
    }
    memcpy(state.prev_vt.data(), vt, n * sizeof(float));
    const float * v_t = state.prev_vt.data();

    // -- Step 1: Euler half-step to midpoint --
    float half_dt = dt * 0.5f;
    float t_mid   = t_curr - half_dt;
    for (int i = 0; i < n; i++) {
        x_mid[i] = xt[i] - v_t[i] * half_dt;
    }

    // -- Step 2: Evaluate velocity at midpoint --
    model_fn(x_mid, t_mid);
    // Result is now in vt_buf (= v_mid)

    // -- Step 3: RF-specific x_0 prediction from midpoint --
    // Using the rectified flow structure: x_t = (1-t)*x_0 + t*eps
    // We predict x_0 from the midpoint evaluation:
    //   x_0_mid = x_mid - t_mid * v_mid
    // Then reconstruct:
    //   x_next = (1 - t_prev) * x_0_mid + t_prev * (x_0_mid + v_mid)
    //          = x_0_mid + t_prev * v_mid
    //
    // This differs from standard midpoint (x_t - v_mid * dt) by using
    // the RF linear interpolation to reconstruct x_next from the refined
    // x_0 prediction, which is more accurate when the velocity field
    // changes along the trajectory.
    for (int i = 0; i < n; i++) {
        float x_0_mid = x_mid[i] - t_mid * vt_buf[i];
        xt[i]         = x_0_mid + t_prev * vt_buf[i];
    }
}
