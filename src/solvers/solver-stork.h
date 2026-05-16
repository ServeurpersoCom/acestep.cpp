#pragma once
// STORK4 solver for ACE Step flow matching, 1 NFE per step.
//
// STORK = Stabilized Taylor Orthogonal Runge Kutta.
// Algorithm: Tan et al, 2025, arXiv:2505.24210.
// C++ port: scragnog (https://github.com/scragnog/HOT-Step-CPP).
//
// Method: 4th order ROCK4 Chebyshev sub stepping. Velocity derivatives are
// approximated from history via finite differences, then cheap arithmetic
// sub steps absorb stiffness without extra model evaluations.
//
// Adaptive: if a sub stepping attempt produces NaN or Inf, the sub step
// count is halved. Falls back to plain Euler as a last resort.

#include "solver-interface.h"
#include "solver-stork4-constants.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

// Detect any non finite element in a buffer.
static bool _stork_has_nan_inf(const float * data, int n) {
    for (int i = 0; i < n; i++) {
        if (!std::isfinite(data[i])) {
            return true;
        }
    }
    return false;
}

// RMS norm of an array, double precision accumulation.
static double _stork_rms(const float * data, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += (double) data[i] * (double) data[i];
    }
    return sqrt(sum / (double) n);
}

// Taylor expansion v(t + diff) approx v + diff * dv + 0.5 * diff^2 * d2v.
// Writes result into out[n]. v, dv, d2v are all [n].
static void _stork_taylor_approx(float *       out,
                                 int           order,
                                 float         diff_val,
                                 const float * v,
                                 const float * dv,
                                 const float * d2v,
                                 int           n) {
    if (order >= 1 && dv) {
        if (order >= 2 && d2v) {
            float half_diff2 = 0.5f * diff_val * diff_val;
            for (int i = 0; i < n; i++) {
                out[i] = v[i] + diff_val * dv[i] + half_diff2 * d2v[i];
            }
        } else {
            for (int i = 0; i < n; i++) {
                out[i] = v[i] + diff_val * dv[i];
            }
        }
    } else {
        memcpy(out, v, n * sizeof(float));
    }
}

// Estimate first and second velocity derivatives from history via finite
// differences. Returns the derivative order actually computed (0, 1, or 2).
// dv and d2v are caller allocated, [n] each, written when the corresponding
// order is available and stable.
static int _stork_compute_derivatives(const float * vt, int n, const SolverState & state, float * dv, float * d2v) {
    const auto & hist = state.velocity_history;
    if (hist.empty()) {
        return 0;
    }

    // First derivative: forward difference on the most recent step.
    const float * v_prev = hist.back().vt.data();
    float         h1     = hist.back().dt;
    if (h1 == 0.0f) {
        return 0;
    }

    for (int i = 0; i < n; i++) {
        dv[i] = (v_prev[i] - vt[i]) / h1;
    }

    // Stability guard: drop the derivative when noise dominates.
    double vt_rms = _stork_rms(vt, n);
    double dv_rms = _stork_rms(dv, n);
    if (vt_rms > 0.0 && dv_rms * fabs(h1) > 5.0 * vt_rms) {
        return 0;
    }

    if ((int) hist.size() < 2) {
        return 1;
    }

    // Second derivative: three point formula over the last two step sizes.
    const float * v_prev2 = hist[hist.size() - 2].vt.data();
    float         h2      = hist[hist.size() - 2].dt;
    if (h2 == 0.0f) {
        return 1;
    }

    double h1d   = (double) h1;
    double h2d   = (double) h2;
    double denom = h1d * h2d * (h1d + h2d);
    if (fabs(denom) < 1e-30) {
        return 1;
    }

    double coeff = 2.0 / denom;
    for (int i = 0; i < n; i++) {
        d2v[i] =
            (float) (coeff * ((double) v_prev2[i] * h1d - (double) v_prev[i] * (h1d + h2d) + (double) vt[i] * h2d));
    }

    // Stability guard for the second derivative.
    double d2v_rms = _stork_rms(d2v, n);
    if (vt_rms > 0.0 && d2v_rms * h1 * h1 > 5.0 * vt_rms) {
        return 1;
    }

    return 2;
}

// Append the latest velocity record and trim history to at most 3 entries.
static void _stork_update_history(SolverState & state, const float * vt, int n, float dt) {
    SolverState::VelocityRecord rec;
    rec.vt.assign(vt, vt + n);
    rec.dt = dt;
    state.velocity_history.push_back(std::move(rec));
    if ((int) state.velocity_history.size() > 3) {
        state.velocity_history.erase(state.velocity_history.begin());
    }
}

// Map a requested sub step count to the closest precomputed ROCK4 degree.
static void _rock4_mdegr(int s, int & mdeg, int & mz, int & mr) {
    int mp1 = 1;
    mdeg    = s;
    mz      = 0;

    for (int i = 0; i < STORK4_MS_LEN; i++) {
        if (STORK4_MS[i] >= s) {
            mdeg = STORK4_MS[i];
            mz   = i;
            mr   = mp1 - 1;
            return;
        }
        mp1 += STORK4_MS[i] * 2 - 1;
    }
    // Fallback to the largest tabulated degree.
    mz   = STORK4_MS_LEN - 1;
    mdeg = STORK4_MS[mz];
    mr   = mp1 - 1;
}

// Run a full ROCK4 sub stepping pass: Chebyshev recurrence followed by the
// 4 stage finishing procedure. Returns true when the result is finite.
static bool _rock4_substep(float *       xt_out,
                           const float * xt,
                           const float * vt,
                           int           s,
                           float         t_curr,
                           float         t_prev,
                           int           deriv_order,
                           const float * dv,
                           const float * d2v,
                           int           n) {
    // Clamp s to the maximum tabulated ROCK4 degree.
    int max_rock4 = STORK4_MS[STORK4_MS_LEN - 1];
    if (s > max_rock4) {
        s = max_rock4;
    }

    int mdeg, mz, mr;
    _rock4_mdegr(s, mdeg, mz, mr);

    float dt = t_curr - t_prev;

    std::vector<float> Y_j_2(xt, xt + n);
    std::vector<float> Y_j_1(xt, xt + n);
    std::vector<float> Y_j(n);
    std::vector<float> vel_approx(n);

    // Scalar timestamps tracked through the recurrence (current, previous, prev prev).
    float ci1 = t_curr;
    float ci2 = t_curr;
    float ci3 = t_curr;

    // Phase 1: Chebyshev recurrence over mdeg sub steps.
    for (int j = 1; j <= mdeg; j++) {
        if (j == 1) {
            float temp1_val = -dt * (float) STORK4_RECF[mr];
            ci1             = t_curr + temp1_val;
            ci2             = ci1;
            for (int i = 0; i < n; i++) {
                Y_j_1[i] = xt[i] + temp1_val * vt[i];
            }
        } else {
            float diff_val = ci1 - t_curr;
            _stork_taylor_approx(vel_approx.data(), deriv_order, diff_val, vt, dv, d2v, n);

            float temp1_val = -dt * (float) STORK4_RECF[mr + 2 * (j - 2) + 1];
            float temp3_val = -(float) STORK4_RECF[mr + 2 * (j - 2) + 2];
            float temp2_val = 1.0f - temp3_val;

            float ci1_new = temp1_val + temp2_val * ci2 + temp3_val * ci3;

            for (int i = 0; i < n; i++) {
                Y_j[i] = temp1_val * vel_approx[i] + temp2_val * Y_j_1[i] + temp3_val * Y_j_2[i];
            }

            memcpy(Y_j_2.data(), Y_j_1.data(), n * sizeof(float));
            memcpy(Y_j_1.data(), Y_j.data(), n * sizeof(float));
            ci3 = ci2;
            ci2 = ci1_new;
            ci1 = ci1_new;
        }
    }

    // Phase 2: 4 stage ROCK4 finishing procedure.
    // Y_base is the recurrence output, equal to Y_j_1 after the loop.
    float * Y_base = Y_j_1.data();

    std::vector<float> F1(n), F2(n), F3(n), F4(n);
    std::vector<float> Y_finish(n);

    // Stage 1.
    float fpa0  = -dt * (float) STORK4_FPA[mz][0];
    float diff1 = ci1 - t_curr;
    _stork_taylor_approx(F1.data(), deriv_order, diff1, vt, dv, d2v, n);
    for (int i = 0; i < n; i++) {
        Y_finish[i] = Y_base[i] + fpa0 * F1[i];
    }

    // Stage 2.
    float ci2_f = ci1 + fpa0;
    float fpa1  = -dt * (float) STORK4_FPA[mz][1];
    float fpa2  = -dt * (float) STORK4_FPA[mz][2];
    float diff2 = ci2_f - t_curr;
    _stork_taylor_approx(F2.data(), deriv_order, diff2, vt, dv, d2v, n);
    for (int i = 0; i < n; i++) {
        Y_finish[i] = Y_base[i] + fpa1 * F1[i] + fpa2 * F2[i];
    }

    // Stage 3.
    ci2_f       = ci1 + fpa1 + fpa2;
    float fpa3  = -dt * (float) STORK4_FPA[mz][3];
    float fpa4  = -dt * (float) STORK4_FPA[mz][4];
    float fpa5  = -dt * (float) STORK4_FPA[mz][5];
    float diff3 = ci2_f - t_curr;
    _stork_taylor_approx(F3.data(), deriv_order, diff3, vt, dv, d2v, n);

    // Stage 4 (final blend).
    ci2_f       = ci1 + fpa3 + fpa4 + fpa5;
    float fpb0  = -dt * (float) STORK4_FPB[mz][0];
    float fpb1  = -dt * (float) STORK4_FPB[mz][1];
    float fpb2  = -dt * (float) STORK4_FPB[mz][2];
    float fpb3  = -dt * (float) STORK4_FPB[mz][3];
    float diff4 = ci2_f - t_curr;
    _stork_taylor_approx(F4.data(), deriv_order, diff4, vt, dv, d2v, n);

    for (int i = 0; i < n; i++) {
        xt_out[i] = Y_base[i] + fpb0 * F1[i] + fpb1 * F2[i] + fpb2 * F3[i] + fpb3 * F4[i];
    }

    return !_stork_has_nan_inf(xt_out, n);
}

static void solver_stork4_step(float *       xt,
                               const float * vt,
                               float         t_curr,
                               float         t_prev,
                               int           n,
                               SolverState & state,
                               SolverModelFn /*model_fn*/,
                               float * /*vt_buf*/) {
    float dt = t_curr - t_prev;

    // Bootstrap: step 0 runs plain Euler (no history yet for derivatives).
    if (state.step_index == 0) {
        for (int i = 0; i < n; i++) {
            xt[i] -= vt[i] * dt;
        }
        _stork_update_history(state, vt, n, dt);
        state.step_index = 1;
        return;
    }

    // Estimate velocity derivatives from history.
    std::vector<float> dv(n), d2v(n);
    int                deriv_order = _stork_compute_derivatives(vt, n, state, dv.data(), d2v.data());

    // Adaptive sub stepping: try state.stork_substeps, halve on NaN until 2.
    int                s = state.stork_substeps >= 2 ? state.stork_substeps : 2;
    std::vector<float> xt_next(n);
    bool               success = false;

    while (s >= 2) {
        if (_rock4_substep(xt_next.data(), xt, vt, s, t_curr, t_prev, deriv_order, dv.data(), d2v.data(), n)) {
            success = true;
            break;
        }
        s /= 2;
    }

    if (success) {
        memcpy(xt, xt_next.data(), n * sizeof(float));
    } else {
        // Last resort: plain Euler.
        for (int i = 0; i < n; i++) {
            xt[i] -= vt[i] * dt;
        }
        fprintf(stderr, "[STORK4] step %d t=%.4f: all sub steps NaN, Euler fallback\n", state.step_index, t_curr);
    }

    _stork_update_history(state, vt, n, dt);
    state.step_index++;
}
