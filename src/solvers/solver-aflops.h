#pragma once
// solver-aflops.h: A-FloPS (Adaptive Flow Path Sampler)
//
// From "A-FloPS: Accelerating Diffusion Models via Adaptive Flow Path Sampler"
// (arXiv:2509.00036). Training-free accelerated sampler that decomposes the
// velocity field into a linear drift (solved exactly) and a smooth residual
// (approximated via high-order interpolation).
//
// Adapted for ACE-Step flow matching where x_t = (1-t)*x_0 + t*eps,
// so v_t = dx/dt = eps - x_0.  Since ACE-Step is natively flow-matching,
// the diffusion->FM reparameterisation is a no-op; only the adaptive
// velocity decomposition is applied.
//
// Key idea:
//   v_t = -x_t / (1-t) + w_t        [decompose into linear drift + residual]
//
//   The linear ODE dx/dt = -x/(1-t) is solved exactly:
//     x_linear(t_prev) = ((1-t_prev)/(1-t_curr)) * x_t
//
//   The residual w_t is assumed smooth (slowly varying), so its integral
//   contribution is approximated:
//     delta = -(1-t_prev) * int[t_curr->t_prev] w_s/(1-s) ds
//       ~ -(1-t_prev) * w_t * ln((1-t_prev)/(1-t_curr))   [1st order]
//
// Two variants:
//   aflops   - 1 NFE per step, stateful 2nd-order using previous residual
//   aflops2  - 2 NFE per step, midpoint-corrected (no state needed)
//
// The multistep variant (aflops) stores the previous residual velocity w_{t-1}
// for Adams-Bashforth-like 2nd-order correction of the residual integral.

#include "solver-interface.h"

#include <algorithm>
#include <cmath>
#include <cstring>

// -- Helpers ------------------------------------------------------------------

// The drift -x/(1-t) is singular at t = 1, where every ACE Step schedule
// starts: below this alpha the exponential step is replaced by a plain one.
static const float AFLOPS_MIN_ALPHA = 1e-3f;

// Clamp sigma to avoid division by zero at boundaries
static inline float aflops_clamp_alpha(float t) {
    // alpha = 1 - t.  Clamp to [eps, 1-eps]
    float a = 1.0f - t;
    return (std::max)(1e-6f, (std::min)(a, 1.0f - 1e-6f));
}

// -- A-FloPS 1-NFE (stateful, multistep) --------------------------------------
//
// Step 1: Euler on residual (no history)
// Step 2+: 2nd-order AB-like correction using stored previous w
//
// The exponential integrator update formula:
//   x_{t_prev} = alpha_prev/alpha_curr * x_t - alpha_prev * w_eff * ln(alpha_prev/alpha_curr)
//
// where alpha = 1-t and w_eff is either:
//   w (1st order, step 1)  or  interpolated from w and prev_w (step 2+)

static void solver_aflops_step(float *       xt,
                               const float * vt,
                               float         t_curr,
                               float         t_prev,
                               int           n,
                               SolverState & state,
                               SolverModelFn /*model_fn*/,
                               float * /*vt_buf*/) {
    float alpha_curr = aflops_clamp_alpha(t_curr);
    float alpha_prev = aflops_clamp_alpha(t_prev);

    if (alpha_curr < AFLOPS_MIN_ALPHA) {
        float dt = t_curr - t_prev;
        for (int i = 0; i < n; i++) {
            xt[i] -= vt[i] * dt;
        }
        return;
    }

    // Ratio of (1-t) values:  > 1 because t_prev < t_curr
    float alpha_ratio = alpha_prev / alpha_curr;
    // Integration coefficient (positive since alpha_prev > alpha_curr)
    float log_ratio   = logf(alpha_ratio);

    // -- 1. Compute residual velocity w = v + x/(1-t) --------------------
    // w_t is what remains after removing the linear drift -x/(1-t)
    // We store w in prev_w for next step's 2nd-order correction
    std::vector<float> w(n);
    float              inv_alpha = 1.0f / alpha_curr;
    for (int i = 0; i < n; i++) {
        w[i] = vt[i] + xt[i] * inv_alpha;
    }

    // -- 2. Compute effective w for the residual integral -----------------
    bool has_prev = !state.aflops_prev_w.empty();

    if (has_prev) {
        // 2nd-order: Adams-Bashforth-like weighted average of current and
        // previous residual velocities.  This extrapolates w to the midpoint
        // of the current interval, reducing truncation error from O(h^2) to O(h^3).
        //
        // For uniform steps this is (3/2)*w - (1/2)*w_prev, but we use
        // step-ratio correction for non-uniform schedules.
        float dt      = t_curr - t_prev;                                // current step size
        float dt_prev = state.aflops_prev_t - state.aflops_prev_t_dst;  // previous step size
        float r       = (dt_prev > 1e-8f) ? dt / dt_prev : 1.0f;
        float c1      = 1.0f + 0.5f * r;                                // weight for current w
        float c0      = 0.5f * r;                                       // weight for previous w (subtracted)

        for (int i = 0; i < n; i++) {
            float w_eff = c1 * w[i] - c0 * state.aflops_prev_w[i];
            xt[i]       = alpha_ratio * xt[i] - alpha_prev * w_eff * log_ratio;
        }
    } else {
        // 1st order: constant-w assumption (exponential Euler)
        for (int i = 0; i < n; i++) {
            xt[i] = alpha_ratio * xt[i] - alpha_prev * w[i] * log_ratio;
        }
    }

    // -- 3. Store residual for next step ----------------------------------
    state.aflops_prev_w     = std::move(w);
    state.aflops_prev_t     = t_curr;
    state.aflops_prev_t_dst = t_prev;
}

// -- A-FloPS 2-NFE (midpoint-corrected, stateless) ----------------------------
//
// Uses a midpoint evaluation to get a 2nd-order accurate estimate of the
// residual integral without needing history from previous steps.
//
// Algorithm:
//   1. Save v_curr, Euler half-step to midpoint (stable at all alpha)
//   2. Evaluate model at midpoint -> get v_mid
//   3. Compute w_mid from v_mid and x_mid
//   4. Full exponential integrator step using w_mid
//
// NOTE: The half-step uses plain Euler rather than the exponential integrator.
// The exponential form has an O(dt^2/alpha^2) correction that explodes near t=1
// where alpha = 1-t ~ 0, producing an x_mid far outside the model's expected
// input distribution and garbling the midpoint velocity evaluation.

static void solver_aflops2_step(float *       xt,
                                const float * vt,
                                float         t_curr,
                                float         t_prev,
                                int           n,
                                SolverState & state,
                                SolverModelFn model_fn,
                                float *       vt_buf) {
    if (!model_fn || t_curr < 1e-8f) {
        // Fallback to Euler if no model callback or degenerate timestep
        float dt = t_curr - t_prev;
        for (int i = 0; i < n; i++) {
            xt[i] -= vt[i] * dt;
        }
        return;
    }

    float alpha_curr = aflops_clamp_alpha(t_curr);
    float alpha_prev = aflops_clamp_alpha(t_prev);

    // -- 1. Euler half-step to midpoint -----------------------------------
    // Save vt before model_fn overwrites vt_buf (they alias)
    if ((int) state.prev_vt.size() < n) {
        state.prev_vt.resize(n);
    }
    memcpy(state.prev_vt.data(), vt, n * sizeof(float));
    const float * v_curr = state.prev_vt.data();

    float dt        = t_curr - t_prev;
    float half_dt   = dt * 0.5f;
    float t_mid     = t_curr - half_dt;
    float alpha_mid = aflops_clamp_alpha(t_mid);

    // Ensure scratch buffer for midpoint
    if ((int) state.xt_scratch.size() < n) {
        state.xt_scratch.resize(n);
    }
    float * x_mid = state.xt_scratch.data();

    for (int i = 0; i < n; i++) {
        x_mid[i] = xt[i] - v_curr[i] * half_dt;
    }

    // -- 2. Evaluate model at midpoint ------------------------------------
    model_fn(x_mid, t_mid);
    // vt_buf now contains v_mid

    // -- 3. Compute w_mid from midpoint evaluation ------------------------
    float inv_alpha_mid = 1.0f / alpha_mid;
    // Stack-avoid: reuse w storage through prev_vt (no longer needed)
    for (int i = 0; i < n; i++) {
        // w_mid = v_mid + x_mid / (1 - t_mid)
        x_mid[i] = vt_buf[i] + x_mid[i] * inv_alpha_mid;  // reuse x_mid as w_mid
    }
    const float * w_mid = x_mid;                          // x_mid buffer now holds w_mid

    // Near t = 1 the exponential step is singular: take the midpoint step
    // with v_mid instead, still second order.
    if (alpha_curr < AFLOPS_MIN_ALPHA) {
        for (int i = 0; i < n; i++) {
            float v_mid = w_mid[i] - (xt[i] - v_curr[i] * half_dt) * inv_alpha_mid;
            xt[i] -= v_mid * dt;
        }
        return;
    }

    // -- 4. Full step using midpoint residual (2nd-order accurate) --------
    float alpha_ratio = alpha_prev / alpha_curr;
    float log_ratio   = logf(alpha_ratio);

    for (int i = 0; i < n; i++) {
        xt[i] = alpha_ratio * xt[i] - alpha_prev * w_mid[i] * log_ratio;
    }
}
