#pragma once
// solver-jkass.h: JKASS solvers for ACE-Step flow matching
//
// Port from jeankassio/JK-AceStep-Nodes, adapted to solver interface.
//
// Two variants:
//   - jkass_quality: Heun with derivative averaging (2 NFE)
//   - jkass_fast: Euler with momentum + frequency damping + temporal smoothing (1 NFE)
//
// Matches Python jkass_solvers.py jkass_quality_step(), jkass_fast_step()

#include "solver-interface.h"

#include <algorithm>

// -- Helper: frequency damping ------------------------------------------------
// Exponential decay across the last axis (channels/frequency bins).
// Layout: xt is [T * Oc] flat. Oc is the "frequency" dimension.
// damping > 0 => stronger attenuation of higher-index bins.
static void _jkass_apply_frequency_damping(float * data, int T, int Oc, float damping) {
    if (damping <= 0.0f) {
        return;
    }

    // Pre-compute decay weights: exp(-damping * (c/Oc)^2)
    std::vector<float> freq_mult(Oc);
    for (int c = 0; c < Oc; c++) {
        float freq   = (float) c / (float) (Oc - 1);
        freq_mult[c] = expf(-damping * freq * freq);
    }

    for (int t = 0; t < T; t++) {
        for (int c = 0; c < Oc; c++) {
            data[t * Oc + c] *= freq_mult[c];
        }
    }
}

// -- Helper: temporal smoothing -----------------------------------------------
// Tiny 1D depthwise convolution [0.25, 0.5, 0.25] across the time axis.
// Layout: data is [T, Oc]. Operates independently per channel.
// strength: 0.0 = none, 1.0 = full smoothing.
static void _jkass_apply_temporal_smoothing(float * data, int T, int Oc, float strength) {
    if (strength <= 0.0f || T < 3) {
        return;
    }

    // Work on a copy to avoid reading modified values
    std::vector<float> smoothed(T * Oc);

    for (int c = 0; c < Oc; c++) {
        // Reflect padding: edge values are reflected
        for (int t = 0; t < T; t++) {
            int t_prev = (t > 0) ? t - 1 : 1;          // reflect at 0
            int t_next = (t < T - 1) ? t + 1 : T - 2;  // reflect at T-1

            float v_prev = data[t_prev * Oc + c];
            float v_curr = data[t * Oc + c];
            float v_next = data[t_next * Oc + c];

            smoothed[t * Oc + c] = 0.25f * v_prev + 0.5f * v_curr + 0.25f * v_next;
        }
    }

    // Blend: (1 - strength) * original + strength * smoothed
    for (int i = 0; i < T * Oc; i++) {
        data[i] = (1.0f - strength) * data[i] + strength * smoothed[i];
    }
}

// -- JKASS Quality (2 NFE) ----------------------------------------------------
// Heun with derivative averaging - identical to solver_heun_step but included
// here for semantic clarity in the solver registry.
static void solver_jkass_quality_step(float *       xt,
                                      const float * vt,
                                      float         t_curr,
                                      float         t_prev,
                                      int           n,
                                      SolverState & state,
                                      SolverModelFn model_fn,
                                      float *       vt_buf) {
    float dt = t_curr - t_prev;

    if (!model_fn || t_prev <= 0.0f) {
        // Fallback to Euler
        for (int i = 0; i < n; i++) {
            xt[i] -= vt[i] * dt;
        }
        return;
    }

    // Ensure scratch
    if ((int) state.xt_scratch.size() < n) {
        state.xt_scratch.resize(n);
    }
    float * x_pred = state.xt_scratch.data();

    // Save k1 - vt and vt_buf alias the same memory, so model_fn will
    // overwrite k1 when it writes k2.
    if ((int) state.prev_vt.size() < n) {
        state.prev_vt.resize(n);
    }
    memcpy(state.prev_vt.data(), vt, n * sizeof(float));

    // Euler predictor
    for (int i = 0; i < n; i++) {
        x_pred[i] = xt[i] - vt[i] * dt;
    }

    // Second evaluation at predicted point
    model_fn(x_pred, t_prev);

    // Average the two derivatives (Heun's method)
    for (int i = 0; i < n; i++) {
        xt[i] -= 0.5f * (state.prev_vt[i] + vt_buf[i]) * dt;
    }
}

// -- JKASS Fast (1 NFE) ------------------------------------------------------
// Euler with momentum + frequency damping + temporal smoothing.
// Uses SolverState fields: beat_stability, frequency_damping, temporal_smoothing, prev_delta.
//
// The caller must know T and Oc independently for the smoothing helpers.
// We infer them from state: n = batch_n * T * Oc, n_per = T * Oc.
// For the smoothing, we process per batch item since the layout is [B, T, Oc].
static void solver_jkass_fast_step(float *       xt,
                                   const float * vt,
                                   float         t_curr,
                                   float         t_prev,
                                   int           n,
                                   SolverState & state,
                                   SolverModelFn /*model_fn*/,
                                   float * /*vt_buf*/) {
    float dt = t_curr - t_prev;

    // Working copy of velocity delta (we may modify it)
    std::vector<float> delta(vt, vt + n);

    // Beat stability: momentum blend with previous step's delta
    if (!state.prev_delta.empty() && state.beat_stability > 0.0f) {
        float bs = state.beat_stability;
        for (int i = 0; i < n; i++) {
            delta[i] = (1.0f - bs) * delta[i] + bs * state.prev_delta[i];
        }
    }
    state.prev_delta.assign(delta.begin(), delta.end());

    // Frequency damping: per batch item
    if (state.frequency_damping > 0.0f && state.n_per > 0) {
        int batch_n = state.batch_n;
        int n_per   = state.n_per;
        // We need Oc to decompose n_per = T * Oc. Oc is always 64 for ACE-Step.
        int Oc      = 64;
        int T       = n_per / Oc;
        for (int b = 0; b < batch_n; b++) {
            _jkass_apply_frequency_damping(delta.data() + b * n_per, T, Oc, state.frequency_damping);
        }
    }

    // Temporal smoothing: per batch item
    if (state.temporal_smoothing > 0.0f && state.n_per > 0) {
        int batch_n = state.batch_n;
        int n_per   = state.n_per;
        int Oc      = 64;
        int T       = n_per / Oc;
        for (int b = 0; b < batch_n; b++) {
            _jkass_apply_temporal_smoothing(delta.data() + b * n_per, T, Oc, state.temporal_smoothing);
        }
    }

    // Euler step with modified velocity delta
    for (int i = 0; i < n; i++) {
        xt[i] -= delta[i] * dt;
    }
}
