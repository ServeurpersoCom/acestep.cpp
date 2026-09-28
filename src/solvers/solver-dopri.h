#pragma once
// solver-dopri.h: Higher-order explicit Runge-Kutta ODE solvers
//
// Three solvers using Butcher tableaux:
//   - RK5:    Dormand-Prince 5th order, fixed step (6 NFE)
//   - DOPRI5: Dormand-Prince 5(4) adaptive with sub-stepping (7+ NFE)
//   - DOP853: Dormand-Prince 8th order, fixed step (13 NFE)
//
// Matches Python dopri_solvers.py

#include "solver-interface.h"

#include <algorithm>
#include <cstdio>

// ===========================================================================
// DOPRI5 Butcher tableau - Dormand-Prince 5th order (7 stages, FSAL)
// Verified from JAX (google/jax) and diffrax (patrick-kidger/diffrax)
// ===========================================================================

static const double DOPRI5_C[6] = { 1.0 / 5, 3.0 / 10, 4.0 / 5, 8.0 / 9, 1.0, 1.0 };

// Lower-triangular A matrix (row per stage 2..7, variable columns)
static const double DOPRI5_A2[1] = { 1.0 / 5 };
static const double DOPRI5_A3[2] = { 3.0 / 40, 9.0 / 40 };
static const double DOPRI5_A4[3] = { 44.0 / 45, -56.0 / 15, 32.0 / 9 };
static const double DOPRI5_A5[4] = { 19372.0 / 6561, -25360.0 / 2187, 64448.0 / 6561, -212.0 / 729 };
static const double DOPRI5_A6[5] = { 9017.0 / 3168, -355.0 / 33, 46732.0 / 5247, 49.0 / 176, -5103.0 / 18656 };
static const double DOPRI5_A7[6] = { 35.0 / 384, 0, 500.0 / 1113, 125.0 / 192, -2187.0 / 6784, 11.0 / 84 };

static const double * DOPRI5_A_ROWS[6] = { DOPRI5_A2, DOPRI5_A3, DOPRI5_A4, DOPRI5_A5, DOPRI5_A6, DOPRI5_A7 };
static const int      DOPRI5_A_COLS[6] = { 1, 2, 3, 4, 5, 6 };

static const double DOPRI5_B[7] = { 35.0 / 384, 0, 500.0 / 1113, 125.0 / 192, -2187.0 / 6784, 11.0 / 84, 0 };

// Error coefficients: b_5th - b_4th (for adaptive stepping)
static const double DOPRI5_E[7] = {
    35.0 / 384 - 1951.0 / 21600,       // k1
    0,                                 // k2
    500.0 / 1113 - 22642.0 / 50085,    // k3
    125.0 / 192 - 451.0 / 720,         // k4
    -2187.0 / 6784 + 12231.0 / 42400,  // k5
    11.0 / 84 - 649.0 / 6300,          // k6
    -1.0 / 60.0,                       // k7
};

// ===========================================================================
// DOP853 Butcher tableau - Dormand-Prince 8th order (13 stages)
// ===========================================================================

static const double DOP853_C[12] = {
    1.0 / 18,   1.0 / 12,
    1.0 / 8,    5.0 / 16,
    3.0 / 8,    59.0 / 400,
    93.0 / 200, 5490023248.0 / 9719169821.0,
    13.0 / 20,  1201146811.0 / 1299019798.0,
    1.0,        1.0,
};

// A matrix rows for stages 2..13 (12 rows)
static const double DOP853_A2[1] = { 1.0 / 18 };
static const double DOP853_A3[2] = { 1.0 / 48, 1.0 / 16 };
static const double DOP853_A4[3] = { 1.0 / 32, 0, 3.0 / 32 };
static const double DOP853_A5[4] = { 5.0 / 16, 0, -75.0 / 64, 75.0 / 64 };
static const double DOP853_A6[5] = { 3.0 / 80, 0, 0, 3.0 / 16, 3.0 / 20 };
static const double DOP853_A7[6] = { 29443841.0 / 614563906, 0, 0, 77736538.0 / 692538347, -28693883.0 / 1125000000,
                                     23124283.0 / 1800000000 };
static const double DOP853_A8[7] = {
    16016141.0 / 946692911,   0, 0, 61564180.0 / 158732637, 22789713.0 / 633445777, 545815736.0 / 2771057229,
    -180193667.0 / 1043307555
};
static const double DOP853_A9[8]   = { 39632708.0 / 573591083,
                                       0,
                                       0,
                                       -433636366.0 / 683701615,
                                       -421739975.0 / 2616292301,
                                       100302831.0 / 723423059,
                                       790204164.0 / 839813087,
                                       800635310.0 / 3783071287 };
static const double DOP853_A10[9]  = { 246121993.0 / 1340847787,
                                       0,
                                       0,
                                       -37695042795.0 / 15268766246,
                                       -309121744.0 / 1061227803,
                                       -12992083.0 / 490766935,
                                       6005943493.0 / 2108947869,
                                       393006217.0 / 1396673457,
                                       123872331.0 / 1001029789 };
static const double DOP853_A11[10] = { -1028468189.0 / 846180014,
                                       0,
                                       0,
                                       8478235783.0 / 508512852,
                                       1311729495.0 / 1432422823,
                                       -10304129995.0 / 1701304382,
                                       -48777925059.0 / 3047939560,
                                       15336726248.0 / 1032824649,
                                       -45442868181.0 / 3398467696,
                                       3065993473.0 / 597172653 };
static const double DOP853_A12[11] = { 185892177.0 / 718116043,
                                       0,
                                       0,
                                       -3185094517.0 / 667107341,
                                       -477755414.0 / 1098053517,
                                       -703635378.0 / 230739211,
                                       5731566787.0 / 1027545527,
                                       5232866602.0 / 850066563,
                                       -4093664535.0 / 808688257,
                                       3962137247.0 / 1805957418,
                                       65686358.0 / 487910083 };
static const double DOP853_A13[12] = { 403863854.0 / 491063109,
                                       0,
                                       0,
                                       -5068492393.0 / 434740067,
                                       -411421997.0 / 543043805,
                                       652783627.0 / 914296604,
                                       11173962825.0 / 925320556,
                                       -13158990841.0 / 6184727034,
                                       3936647629.0 / 1978049680,
                                       -160528059.0 / 685178525,
                                       248638103.0 / 1413531060,
                                       0 };

static const double * DOP853_A_ROWS[12] = { DOP853_A2, DOP853_A3, DOP853_A4,  DOP853_A5,  DOP853_A6,  DOP853_A7,
                                            DOP853_A8, DOP853_A9, DOP853_A10, DOP853_A11, DOP853_A12, DOP853_A13 };
static const int      DOP853_A_COLS[12] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };

static const double DOP853_B[13] = {
    14005451.0 / 335480064,      // k1
    0,                           // k2
    0,                           // k3
    0,                           // k4
    0,                           // k5
    -59238493.0 / 1068277825,    // k6
    181606767.0 / 758867731,     // k7
    561292985.0 / 797845732,     // k8
    -1041891430.0 / 1371343529,  // k9
    760417239.0 / 1151165299,    // k10
    118820643.0 / 751138087,     // k11
    -528747749.0 / 2220607170,   // k12
    1.0 / 4,                     // k13
};

// ===========================================================================
// Generic explicit Runge-Kutta step (shared by all three solvers)
// ===========================================================================

// Performs a generic ERK step using a Butcher tableau.
// Integrates dx/dt = -v(x,t) from t_curr to t_curr - dt.
//
// xt:        [n] current state
// vt:        [n] velocity at (xt, t_curr), i.e. k1
// t_curr:    current timestep
// dt:        step size (positive; t_curr - t_prev)
// a_rows:    lower-triangular A coefficient rows for stages 2..N
// a_cols:    number of columns in each A row
// b_sol:     solution weights for all stages
// c_nodes:   time-fraction nodes for stages 2..N
// num_extra: number of extra stages (rows in A / entries in c_nodes)
// n:         total elements
// model_fn:  model callback
// vt_buf:    [n] velocity output buffer (written by model_fn)
// xt_out:    [n] output buffer for x_next
// ks:        array of n_stages vectors, each [n], to store stage velocities
//            ks[0] is pre-filled with k1 (=vt)
static void _erk_step(const float *                     xt,
                      const float *                     vt,
                      float                             t_curr,
                      float                             dt,
                      const double * const              a_rows[],
                      const int                         a_cols[],
                      const double                      b_sol[],
                      const double                      c_nodes[],
                      int                               num_extra,
                      int                               n,
                      SolverModelFn                     model_fn,
                      float *                           vt_buf,
                      float *                           xt_out,
                      std::vector<std::vector<float>> & ks) {
    // ks[0] = k1 (already filled by caller)
    std::vector<float> x_tmp(n);

    // Compute intermediate stages
    for (int s = 0; s < num_extra; s++) {
        const double * a_row = a_rows[s];
        int            ncols = a_cols[s];
        double         ci    = c_nodes[s];

        // combo = sum(a[s][j] * ks[j] for j in 0..ncols-1)
        // x_tmp = xt - dt * combo
        for (int i = 0; i < n; i++) {
            double combo = 0.0;
            for (int j = 0; j < ncols; j++) {
                if (a_row[j] != 0.0) {
                    combo += a_row[j] * (double) ks[j][i];
                }
            }
            x_tmp[i] = (float) ((double) xt[i] - (double) dt * combo);
        }

        float t_stage = t_curr - (float) (ci * dt);
        model_fn(x_tmp.data(), t_stage);

        // Save stage velocity
        ks[s + 1].assign(vt_buf, vt_buf + n);
    }

    // Combine stages with solution weights: xt_out = xt - dt * sum(b[j] * ks[j])
    int total_stages = num_extra + 1;
    for (int i = 0; i < n; i++) {
        double sol = 0.0;
        for (int j = 0; j < total_stages; j++) {
            if (b_sol[j] != 0.0 && j < (int) ks.size()) {
                sol += b_sol[j] * (double) ks[j][i];
            }
        }
        xt_out[i] = (float) ((double) xt[i] - (double) dt * sol);
    }
}

// ===========================================================================
// RK5 - Dormand-Prince 5th order, fixed step (6 NFE)
// ===========================================================================

static void solver_rk5_step(float *       xt,
                            const float * vt,
                            float         t_curr,
                            float         t_prev,
                            int           n,
                            SolverState & /*state*/,
                            SolverModelFn model_fn,
                            float *       vt_buf) {
    if (!model_fn) {
        float dt = t_curr - t_prev;
        for (int i = 0; i < n; i++) {
            xt[i] -= vt[i] * dt;
        }
        return;
    }

    float dt = t_curr - t_prev;

    // 6 stages: k1 (=vt) + 5 extra
    std::vector<std::vector<float>> ks(6);
    ks[0].assign(vt, vt + n);
    for (int s = 1; s < 6; s++) {
        ks[s].resize(n);
    }

    std::vector<float> xt_next(n);
    _erk_step(xt, vt, t_curr, dt, DOPRI5_A_ROWS, DOPRI5_A_COLS, DOPRI5_B, DOPRI5_C, 5, n, model_fn, vt_buf,
              xt_next.data(), ks);

    memcpy(xt, xt_next.data(), n * sizeof(float));
}

// ===========================================================================
// DOPRI5 - Adaptive Dormand-Prince 5(4) with error-driven sub-stepping
// ===========================================================================

static void solver_dopri5_step(float *       xt,
                               const float * vt,
                               float         t_curr,
                               float         t_prev,
                               int           n,
                               SolverState & /*state*/,
                               SolverModelFn model_fn,
                               float *       vt_buf) {
    if (!model_fn) {
        float dt = t_curr - t_prev;
        for (int i = 0; i < n; i++) {
            xt[i] -= vt[i] * dt;
        }
        return;
    }

    const double atol    = 1e-3;
    const double rtol    = 1e-2;
    const int    max_sub = 8;
    const double safety  = 0.9;

    double t     = (double) t_curr;
    double t_end = (double) t_prev;
    double h     = t - t_end;

    // Working copies
    std::vector<float> x_cur(xt, xt + n);
    std::vector<float> v_cur(vt, vt + n);
    std::vector<float> x_next(n);

    // Stage storage: 7 stages for full DOPRI5 (including FSAL)
    std::vector<std::vector<float>> ks(7);

    int sub = 0;
    while (sub < max_sub && (t - t_end) > 1e-10) {
        h = (std::min)(h, t - t_end);  // don't overshoot

        // Initialize stages
        ks[0].assign(v_cur.begin(), v_cur.end());
        for (int s = 1; s < 7; s++) {
            ks[s].resize(n);
        }

        // Full DOPRI5 step (all 6 A rows -> stages k1..k7)
        _erk_step(x_cur.data(), v_cur.data(), (float) t, (float) h, DOPRI5_A_ROWS, DOPRI5_A_COLS, DOPRI5_B, DOPRI5_C, 6,
                  n, model_fn, vt_buf, x_next.data(), ks);

        // Error estimate: e = dt * sum(DOPRI5_E[j] * ks[j])
        double err_sq_sum = 0.0;
        for (int i = 0; i < n; i++) {
            double err_i = 0.0;
            for (int j = 0; j < 7; j++) {
                if (DOPRI5_E[j] != 0.0 && j < (int) ks.size()) {
                    err_i += DOPRI5_E[j] * (double) ks[j][i];
                }
            }
            err_i *= h;

            double scale = atol + rtol * (std::max)(std::abs((double) x_cur[i]), std::abs((double) x_next[i]));
            double ratio = err_i / scale;
            err_sq_sum += ratio * ratio;
        }
        double err_norm = sqrt(err_sq_sum / (double) n);

        if (err_norm <= 1.0) {
            // Accept step
            t -= h;
            x_cur = x_next;
            v_cur = ks[6];  // FSAL: k7 becomes k1 of next sub-step

            // Grow step size (capped at 5x)
            if (err_norm > 1e-10) {
                h *= (std::min)(5.0, safety * pow(err_norm, -0.2));
            } else {
                h *= 5.0;
            }
        } else {
            // Reject: shrink (floored at 0.2x)
            h *= (std::max)(0.2, safety * pow(err_norm, -0.2));
        }

        sub++;
    }

    memcpy(xt, x_cur.data(), n * sizeof(float));
}

// ===========================================================================
// DOP853 - Dormand-Prince 8th order, fixed step (13 NFE)
// ===========================================================================

static void solver_dop853_step(float *       xt,
                               const float * vt,
                               float         t_curr,
                               float         t_prev,
                               int           n,
                               SolverState & /*state*/,
                               SolverModelFn model_fn,
                               float *       vt_buf) {
    if (!model_fn) {
        float dt = t_curr - t_prev;
        for (int i = 0; i < n; i++) {
            xt[i] -= vt[i] * dt;
        }
        return;
    }

    float dt = t_curr - t_prev;

    // 13 stages: k1 (=vt) + 12 extra
    std::vector<std::vector<float>> ks(13);
    ks[0].assign(vt, vt + n);
    for (int s = 1; s < 13; s++) {
        ks[s].resize(n);
    }

    std::vector<float> xt_next(n);
    _erk_step(xt, vt, t_curr, dt, DOP853_A_ROWS, DOP853_A_COLS, DOP853_B, DOP853_C, 12, n, model_fn, vt_buf,
              xt_next.data(), ks);

    memcpy(xt, xt_next.data(), n * sizeof(float));
}
