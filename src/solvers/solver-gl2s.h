#pragma once
// solver-gl2s.h: Gauss-Legendre 2-stage implicit Runge-Kutta (4th order)
//
// The highest-order A-stable symplectic IRK method with 2 stages.
// Uses fixed-point iteration to solve the implicit system at each step.
//
// Butcher tableau:
//   c1 = 1/2 - sqrt(3)/6    a11 = 1/4              a12 = 1/4 - sqrt(3)/6
//   c2 = 1/2 + sqrt(3)/6    a21 = 1/4 + sqrt(3)/6  a22 = 1/4
//                            b1  = 1/2              b2  = 1/2
//
// Fixed-point iteration (3 rounds, 6 NFE/step):
//   Init k1 = k2 = vt (reuse first model eval)
//   For each iteration:
//     x1 = xt - dt*(a11*k1 + a12*k2)
//     x2 = xt - dt*(a21*k1 + a22*k2)
//     k1 = model(x1, t - c1*dt)
//     k2 = model(x2, t - c2*dt)
//   Update: xt_next = xt - dt * 0.5 * (k1 + k2)
//
// Properties: A-stable, symplectic, 4th order convergence.
// Cost: 6 NFE per step (3 iterations x 2 evaluations).
// With 8 steps: 48 NFE total, comparable to RK4 at 12 steps (48 NFE).

#include "solver-interface.h"

#include <cmath>
#include <cstring>

// Butcher tableau constants (computed at compile time)
static constexpr float GL2S_SQRT3_6 = 0.28867513459481287f;  // sqrt(3)/6
static constexpr float GL2S_C1      = 0.5f - GL2S_SQRT3_6;   // ~ 0.2113
static constexpr float GL2S_C2      = 0.5f + GL2S_SQRT3_6;   // ~ 0.7887
static constexpr float GL2S_A11     = 0.25f;
static constexpr float GL2S_A12     = 0.25f - GL2S_SQRT3_6;  // ~ -0.0387
static constexpr float GL2S_A21     = 0.25f + GL2S_SQRT3_6;  // ~ 0.5387
static constexpr float GL2S_A22     = 0.25f;

// Number of fixed-point iterations. 3 is standard for smooth velocity fields.
// More iterations trade NFE for convergence; 2 is often sufficient for diffusion.
static constexpr int GL2S_ITERATIONS = 3;

static void solver_gl2s_step(float *       xt,
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

    // Ensure scratch buffers are allocated
    if ((int) state.xt_scratch.size() < n) {
        state.xt_scratch.resize(n);
    }

    // Local buffers for the two stages
    // k1, k2: stage velocities.  x1, x2: stage states.
    // We reuse xt_scratch for x1, and allocate the rest.
    float * x1 = state.xt_scratch.data();

    // prev_vt and prev_prev_vt are reused as k1 and k2 storage
    // (no conflict - GL2s is not stateful across steps like DPM++)
    if ((int) state.prev_vt.size() < n) {
        state.prev_vt.resize(n);
    }
    if ((int) state.prev_prev_vt.size() < n) {
        state.prev_prev_vt.resize(n);
    }
    float * k1 = state.prev_vt.data();
    float * k2 = state.prev_prev_vt.data();

    // Allocate x2 buffer (k1/k2 occupy prev_vt/prev_prev_vt)
    // Use a static thread-local to avoid per-step allocation churn
    static thread_local std::vector<float> x2_buf;
    if ((int) x2_buf.size() < n) {
        x2_buf.resize(n);
    }
    float * x2 = x2_buf.data();

    // Initialize: k1 = k2 = vt (reuse the first model evaluation)
    memcpy(k1, vt, n * sizeof(float));
    memcpy(k2, vt, n * sizeof(float));

    // Collocation time points
    float t1 = t_curr - GL2S_C1 * dt;
    float t2 = t_curr - GL2S_C2 * dt;

    // Fixed-point iteration
    for (int iter = 0; iter < GL2S_ITERATIONS; iter++) {
        // Stage 1 state: x1 = xt - dt*(a11*k1 + a12*k2)
        for (int i = 0; i < n; i++) {
            x1[i] = xt[i] - dt * (GL2S_A11 * k1[i] + GL2S_A12 * k2[i]);
        }
        // Stage 2 state: x2 = xt - dt*(a21*k1 + a22*k2)
        for (int i = 0; i < n; i++) {
            x2[i] = xt[i] - dt * (GL2S_A21 * k1[i] + GL2S_A22 * k2[i]);
        }

        // Evaluate k1 = v(x1, t1)
        model_fn(x1, t1);
        memcpy(k1, vt_buf, n * sizeof(float));

        // Evaluate k2 = v(x2, t2)
        model_fn(x2, t2);
        memcpy(k2, vt_buf, n * sizeof(float));
    }

    // Update: xt_next = xt - dt * 0.5 * (k1 + k2)
    for (int i = 0; i < n; i++) {
        xt[i] -= dt * 0.5f * (k1[i] + k2[i]);
    }
}
