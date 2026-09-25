#pragma once
// solver-unipc.h: UniPC (Unified Predictor-Corrector) solver
//
// Implements the B(h)1 variant from:
//   Zhao et al., "UniPC: A Unified Predictor-Corrector Framework for
//   Fast Sampling of Diffusion Models" (arXiv:2302.04867, NeurIPS 2023)
//
// Adapted for ACE-Step flow matching where alpha_t = 1-t, sigma_t = t.
// Works in log-SNR (lambda) space: lambda(t) = log((1-t)/t).
//
// Two variants:
//   unipc   - Predictor + Corrector (2 NFE/step after warmup; 1 NFE at final step)
//   unipc_p - Predictor only        (1 NFE/step, stateful multistep like DPM++)
//
// The solver stores data predictions D_i = x_i - t_i * v_i (i.e. x0_hat)
// from previous steps, using polynomial interpolation in lambda-space
// to construct higher-order updates without extra model evaluations.
// The corrector does one additional model_fn call per step.

#include "solver-interface.h"

#include <algorithm>
#include <cmath>

// -- Flow matching helpers ------------------------------------------------

// log-SNR for flow matching: lambda(t) = log((1-t)/t)
static inline float unipc_lambda(float t) {
    t = (std::max)(t, 1e-7f);
    t = (std::min)(t, 1.0f - 1e-7f);
    return logf((1.0f - t) / t);
}

// expm1(x) = exp(x) - 1, numerically stable for small x
static inline float unipc_expm1(float x) {
    return expm1f(x);
}

// -- Small linear system solvers (Cramer's rule) --------------------------

// Solve 1x1: R[0]*x = b[0]
static inline void unipc_solve_1x1(const float * R, const float * b, float * x) {
    x[0] = (fabsf(R[0]) > 1e-12f) ? b[0] / R[0] : 0.0f;
}

// Solve 2x2: R*x = b  where R is row-major [r00 r01; r10 r11]
static inline void unipc_solve_2x2(const float * R, const float * b, float * x) {
    float det = R[0] * R[3] - R[1] * R[2];
    if (fabsf(det) < 1e-12f) {
        x[0] = x[1] = 0.0f;
        return;
    }
    float inv = 1.0f / det;
    x[0]      = (R[3] * b[0] - R[1] * b[1]) * inv;
    x[1]      = (R[0] * b[1] - R[2] * b[0]) * inv;
}

// Solve 3x3: R*x = b  where R is row-major
static inline void unipc_solve_3x3(const float * R, const float * b, float * x) {
    // Cramer's rule
    float a00 = R[0], a01 = R[1], a02 = R[2];
    float a10 = R[3], a11 = R[4], a12 = R[5];
    float a20 = R[6], a21 = R[7], a22 = R[8];

    float det = a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20);

    if (fabsf(det) < 1e-12f) {
        x[0] = x[1] = x[2] = 0.0f;
        return;
    }
    float inv = 1.0f / det;
    x[0] = ((a11 * a22 - a12 * a21) * b[0] + (a02 * a21 - a01 * a22) * b[1] + (a01 * a12 - a02 * a11) * b[2]) * inv;
    x[1] = ((a12 * a20 - a10 * a22) * b[0] + (a00 * a22 - a02 * a20) * b[1] + (a02 * a10 - a00 * a12) * b[2]) * inv;
    x[2] = ((a10 * a21 - a11 * a20) * b[0] + (a01 * a20 - a00 * a21) * b[1] + (a00 * a11 - a01 * a10) * b[2]) * inv;
}

// General dispatch for small systems
static inline void unipc_solve(int K, const float * R, const float * b, float * x) {
    switch (K) {
        case 1:
            unipc_solve_1x1(R, b, x);
            break;
        case 2:
            unipc_solve_2x2(R, b, x);
            break;
        case 3:
            unipc_solve_3x3(R, b, x);
            break;
        default:  // Should never happen (max order 3)
            for (int i = 0; i < K; i++) {
                x[i] = 0.0f;
            }
            break;
    }
}

// -- UniPC core update (B(h)1 variant, data_prediction mode) --------------
//
// This implements multistep_uni_pc_bh_update from the reference code.
//
// Parameters:
//   xt:           [n] current latent (modified in-place to result)
//   vt:           [n] velocity at (xt, t_curr)
//   t_curr:       current timestep (high noise side)
//   t_next:       target timestep (low noise side)
//   n:            total element count
//   state:        solver state with unipc history
//   model_fn:     callback for corrector evaluation
//   vt_buf:       [n] buffer for model_fn output (aliases vt)
//   use_corrector: whether to apply the corrector step
static void unipc_bh1_update(float *       xt,
                             const float * vt,
                             float         t_curr,
                             float         t_next,
                             int           n,
                             SolverState & state,
                             SolverModelFn model_fn,
                             float *       vt_buf,
                             bool          use_corrector) {
    // -- 1. Compute data prediction D_n = x0_hat ----------------------
    // For flow matching: x0 = xt - t * vt
    std::vector<float> D_n(n);
    for (int i = 0; i < n; i++) {
        D_n[i] = xt[i] - t_curr * vt[i];
    }

    // -- 2. Lambda (log-SNR) values -----------------------------------
    float lambda_curr = unipc_lambda(t_curr);
    float lambda_next = unipc_lambda(t_next);
    float h           = lambda_next - lambda_curr;  // negative (moving from noise to clean)

    // Flow matching: alpha = 1-t, sigma = t
    float alpha_next = 1.0f - t_next;
    float sigma_next = t_next;
    float sigma_curr = t_curr;

    // -- 3. Determine effective order from available history ----------
    int max_order = 2;                                 // UniPC default order
    int avail     = (int) state.unipc_history.size();
    int order     = (std::min)(max_order, avail + 1);  // +1 counts current D_n

    // -- 4. Compute B(h), h_phi_k coefficients -----------------------
    // For data_prediction: hh = -h (positive since h is negative for our direction)
    float hh  = -h;                   // positive value
    float B_h = hh;                   // B(h)1 variant: B(h) = h

    float h_phi_1 = unipc_expm1(hh);  // exp(hh) - 1

    // Compute h_phi_k sequence: h_phi_1, h_phi_2, h_phi_3, ...
    // h_phi_{k+1} = h_phi_k / hh - 1/k!
    float h_phi_k_vals[4];  // max order 3 + 1
    h_phi_k_vals[0]   = h_phi_1;
    float factorial_i = 1.0f;
    float h_phi_k     = h_phi_1;
    for (int k = 1; k <= order; k++) {
        h_phi_k         = h_phi_k / hh - 1.0f / factorial_i;
        h_phi_k_vals[k] = h_phi_k;
        factorial_i *= (float) (k + 1);
    }

    // -- 5. Build R matrix and b vector for the linear system ---------
    // rks[i] = ratio of lambda differences for each history point
    // R[i][j] = rks[j]^i  (Vandermonde-like)
    // b[i] = h_phi_{i+2} * (i+1)! / B_h
    int   K = order;         // system dimension
    float rks[3];            // max 3 history ratios
    int   n_D1 = order - 1;  // number of difference terms from history

    // Compute rks from history (newest to oldest)
    for (int i = 0; i < n_D1; i++) {
        int   hist_idx    = avail - 1 - i;  // newest first
        float lambda_hist = unipc_lambda(state.unipc_history[hist_idx].t);
        rks[i]            = (lambda_hist - lambda_curr) / h;
    }
    rks[n_D1] = 1.0f;  // for the corrector/predictor end term

    // Build R matrix (row-major, K x K)
    float R_mat[9] = { 0 };  // max 3x3
    for (int row = 0; row < K; row++) {
        for (int col = 0; col < K; col++) {
            R_mat[row * K + col] = powf(rks[col], (float) row);
        }
    }

    // Build b vector
    float b_vec[3] = { 0 };
    factorial_i    = 1.0f;
    h_phi_k        = h_phi_1;
    for (int i = 0; i < K; i++) {
        h_phi_k  = h_phi_k / hh - 1.0f / factorial_i;
        b_vec[i] = h_phi_k * factorial_i / B_h;
        factorial_i *= (float) (i + 2);
    }

    // -- 6. Compute D1 differences from history -----------------------
    // D1[i][j] = (D_hist[i] - D_n) / rk[i]  for each element j
    // We store these flat: d1_data[i * n + j]
    std::vector<float> d1_data;
    if (n_D1 > 0) {
        d1_data.resize(n_D1 * n);
        for (int i = 0; i < n_D1; i++) {
            int           hist_idx = avail - 1 - i;
            const float * D_hist   = state.unipc_history[hist_idx].model_output.data();
            float         rk_inv   = (fabsf(rks[i]) > 1e-12f) ? 1.0f / rks[i] : 0.0f;
            for (int j = 0; j < n; j++) {
                d1_data[i * n + j] = (D_hist[j] - D_n[j]) * rk_inv;
            }
        }
    }

    // -- 7. Compute base term x_t_ ------------------------------------
    // data_prediction: x_t_ = (sigma_next/sigma_curr) * xt - alpha_next * h_phi_1 * D_n
    float sigma_ratio = (fabsf(sigma_curr) > 1e-7f) ? sigma_next / sigma_curr : 0.0f;

    // Ensure scratch buffer
    if ((int) state.xt_scratch.size() < n) {
        state.xt_scratch.resize(n);
    }
    float * x_t_ = state.xt_scratch.data();  // base term

    for (int i = 0; i < n; i++) {
        x_t_[i] = sigma_ratio * xt[i] - alpha_next * h_phi_1 * D_n[i];
    }

    // -- 8. Predictor -------------------------------------------------
    // Solve for predictor coefficients using history only (K-1 system)
    bool use_predictor = (n_D1 > 0);
    if (use_predictor) {
        float rhos_p[3] = { 0 };
        if (order == 2) {
            // Hardcoded from reference: rhos_p = [0.5]
            rhos_p[0] = 0.5f;
        } else if (n_D1 > 0) {
            // Solve (K-1) x (K-1) system: R[:-1, :-1] * rhos_p = b[:-1]
            int   Kp     = K - 1;
            float R_p[4] = { 0 };  // max 2x2
            for (int row = 0; row < Kp; row++) {
                for (int col = 0; col < Kp; col++) {
                    R_p[row * Kp + col] = R_mat[row * K + col];
                }
            }
            unipc_solve(Kp, R_p, b_vec, rhos_p);
        }

        // Apply predictor: x_t = x_t_ - alpha_next * B_h * sum(rhos_p[k] * D1[k])
        for (int i = 0; i < n; i++) {
            float pred_res = 0.0f;
            for (int k = 0; k < n_D1; k++) {
                pred_res += rhos_p[k] * d1_data[k * n + i];
            }
            xt[i] = x_t_[i] - alpha_next * B_h * pred_res;
        }
    } else {
        // No history: first-order (just the base term)
        memcpy(xt, x_t_, n * sizeof(float));
    }

    // -- 9. Corrector (optional, costs 1 model_fn call) ---------------
    if (use_corrector && model_fn) {
        // Save predicted x0 before model_fn overwrites vt_buf
        // (vt_buf aliases vt - model_fn writes new velocity there)

        // Evaluate model at predicted point
        model_fn(xt, t_next);
        // Now vt_buf contains v(x_predicted, t_next)

        // Compute corrector's data prediction: D_corr = x_t - t_next * v_corr
        // Note: xt has been modified by predictor, and vt_buf has new velocity
        std::vector<float> D_corr_diff(n);  // D1_t = D_corr - D_n
        for (int i = 0; i < n; i++) {
            float D_corr   = xt[i] - t_next * vt_buf[i];
            D_corr_diff[i] = D_corr - D_n[i];
        }

        // Solve full K x K system for corrector coefficients
        float rhos_c[3] = { 0 };
        if (order == 1) {
            // Hardcoded from reference: rhos_c = [0.5]
            rhos_c[0] = 0.5f;
        } else {
            unipc_solve(K, R_mat, b_vec, rhos_c);
        }

        // Apply corrector:
        // x_t = x_t_ - alpha_next * B_h * (sum(rhos_c[:-1] * D1s) + rhos_c[-1] * D1_t)
        for (int i = 0; i < n; i++) {
            float corr_res = 0.0f;
            for (int k = 0; k < n_D1; k++) {
                corr_res += rhos_c[k] * d1_data[k * n + i];
            }
            corr_res += rhos_c[K - 1] * D_corr_diff[i];
            xt[i] = x_t_[i] - alpha_next * B_h * corr_res;
        }
    }

    // -- 10. Update history -------------------------------------------
    // Store D_n in history (the data prediction at this step)
    SolverState::UniPCRecord rec;
    rec.model_output = std::move(D_n);
    rec.t            = t_curr;

    state.unipc_history.push_back(std::move(rec));

    // Keep only the last (max_order - 1) entries
    while ((int) state.unipc_history.size() > max_order) {
        state.unipc_history.erase(state.unipc_history.begin());
    }
}

// -- Public solver step functions -----------------------------------------

// UniPC with corrector (2 NFE per step: 1 from sampler + 1 corrector)
// Falls back to Euler on the first step, then ramps up order.
static void solver_unipc_step(float *       xt,
                              const float * vt,
                              float         t_curr,
                              float         t_prev,
                              int           n,
                              SolverState & state,
                              SolverModelFn model_fn,
                              float *       vt_buf) {
    unipc_bh1_update(xt, vt, t_curr, t_prev, n, state, model_fn, vt_buf,
                     /*use_corrector=*/true);
}

// UniPC predictor only (1 NFE per step, stateful multistep)
// More efficient than the corrected variant, quality between Euler and UniPC.
static void solver_unipc_p_step(float *       xt,
                                const float * vt,
                                float         t_curr,
                                float         t_prev,
                                int           n,
                                SolverState & state,
                                SolverModelFn /*model_fn*/,
                                float * /*vt_buf*/) {
    unipc_bh1_update(xt, vt, t_curr, t_prev, n, state, nullptr, nullptr,
                     /*use_corrector=*/false);
}
