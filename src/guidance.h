#pragma once
// guidance.h: classifier free guidance modes for the DiT sampler.
//
// Every mode combines the conditional and unconditional velocities through
// APG (momentum, per channel norm clipping, projection orthogonal to the
// conditional prediction) and differs in the scale it hands APG or in what
// it does around it:
//
//   apg            APG at the requested scale (Python ACE Step 1.5 default).
//   cfg_pp         CFG++: scale reduced by dt / t for large steps.
//   dynamic_cfg    scale decays with sqrt(cos) over the steps.
//   rescaled_cfg   APG output rescaled to the conditional std, phi blend.
//   cfg_zero_star  CFG-Zero*: zero velocity for the first steps, then APG
//                  (Fan et al. 2025, arXiv:2503.18886).
//   smc_cfg        sliding mode control on the cond/uncond error on top of
//                  APG (Han et al. 2025, arXiv:2603.03281).
//   cfg_mp         APG, then manifold projection after each solver step
//                  with extra model evaluations (Su et al. 2025,
//                  arXiv:2601.21892). The projection lives in the sampler.
//   adg            Angle based Dynamic Guidance, no APG: the angle between
//                  the conditional and unconditional x0 estimates is scaled
//                  and clipped to pi/6 (ACE-Step 1.5 adg_forward, base model).
//
// cfg_interval_start / cfg_interval_end restrict any mode to the steps whose
// t_curr falls inside them; outside it the conditional prediction is used.
//
// C++ ports of the HOT-Step guidance plugins (https://github.com/scragnog/HOT-Step-CPP).

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif

// APG (Adaptive Projected Guidance) for DiT CFG
// Matches Python ACE-Step-1.5 acestep/models/base/apg_guidance.py
struct APGMomentumBuffer {
    double              momentum;
    std::vector<double> running_average;
    bool                initialized;

    APGMomentumBuffer(double m = -0.75) : momentum(m), initialized(false) {}

    void update(const double * values, int n) {
        if (!initialized) {
            running_average.assign(values, values + n);
            initialized = true;
        } else {
            for (int i = 0; i < n; i++) {
                running_average[i] = values[i] + momentum * running_average[i];
            }
        }
    }
};

// project(v0, v1, dims=[1]): decompose v0 into parallel + orthogonal w.r.t. v1
// All math in double precision matching Python .double() calls.
// Layout: memory [T, Oc] time-major (ggml ne=[Oc, T]).
// Python dims=[1] on [B,T,C] = normalize/project per channel over T dimension.
// In memory [T, Oc] layout: for each channel c, operate over all T time frames.
static void apg_project(const double * v0, const double * v1, double * out_par, double * out_orth, int Oc, int T) {
    for (int c = 0; c < Oc; c++) {
        double norm2 = 0.0;
        for (int t = 0; t < T; t++) {
            norm2 += v1[t * Oc + c] * v1[t * Oc + c];
        }
        double inv_norm = (norm2 > 1e-60) ? (1.0 / sqrt(norm2)) : 0.0;

        double dot = 0.0;
        for (int t = 0; t < T; t++) {
            dot += v0[t * Oc + c] * (v1[t * Oc + c] * inv_norm);
        }

        for (int t = 0; t < T; t++) {
            int    idx    = t * Oc + c;
            double v1n    = v1[idx] * inv_norm;
            out_par[idx]  = dot * v1n;
            out_orth[idx] = v0[idx] - out_par[idx];
        }
    }
}

// APG forward matching Python apg_forward() exactly:
//   1. diff = cond - uncond
//   2. momentum.update(diff); diff = running_average
//   3. norm clip: per-channel L2 over T (dims=[1]), clip to norm_threshold=2.5
//   4. project(diff, pred_COND) -> (parallel, orthogonal)
//   5. result = pred_cond + (scale - 1) * orthogonal
// Internal computation in double precision (Python uses .double()).
static void apg_forward(const float *       pred_cond,
                        const float *       pred_uncond,
                        float               guidance_scale,
                        APGMomentumBuffer & mbuf,
                        float *             result,
                        int                 Oc,
                        int                 T,
                        float               norm_threshold = 2.5f) {
    int n = Oc * T;

    // 1. diff = cond - uncond (promote to double)
    std::vector<double> diff(n);
    for (int i = 0; i < n; i++) {
        diff[i] = (double) pred_cond[i] - (double) pred_uncond[i];
    }

    // 2. momentum update, then use smoothed diff
    mbuf.update(diff.data(), n);
    memcpy(diff.data(), mbuf.running_average.data(), n * sizeof(double));

    // 3. norm clipping: per-channel L2 over T (dims=[1]), clip to threshold
    if (norm_threshold > 0.0f) {
        for (int c = 0; c < Oc; c++) {
            double norm2 = 0.0;
            for (int t = 0; t < T; t++) {
                norm2 += diff[t * Oc + c] * diff[t * Oc + c];
            }
            double norm = sqrt(norm2 > 0.0 ? norm2 : 0.0);
            double s    = (norm > 1e-60) ? fmin(1.0, (double) norm_threshold / norm) : 1.0;
            if (s < 1.0) {
                for (int t = 0; t < T; t++) {
                    diff[t * Oc + c] *= s;
                }
            }
        }
    }

    // 4. project(diff, pred_COND) -> orthogonal component (double precision)
    std::vector<double> pred_cond_d(n), par(n), orth(n);
    for (int i = 0; i < n; i++) {
        pred_cond_d[i] = (double) pred_cond[i];
    }
    apg_project(diff.data(), pred_cond_d.data(), par.data(), orth.data(), Oc, T);

    // 5. result = pred_cond + (scale - 1) * orthogonal (back to float)
    double w = (double) guidance_scale - 1.0;
    for (int i = 0; i < n; i++) {
        result[i] = (float) ((double) pred_cond[i] + w * orth[i]);
    }
}

struct GuidanceParams {
    std::string mode            = "apg";
    float       apg_momentum    = 0.75f;  // running average decay, applied as -momentum
    float       norm_threshold  = 2.5f;   // APG per channel norm clip, 0 disables
    int         zero_init_steps = 1;      // cfg_zero_star
    float       smc_lambda      = 0.5f;   // smc_cfg sliding surface slope
    float       smc_k           = 0.1f;   // smc_cfg switching gain
    int         mp_iterations   = 1;      // cfg_mp fixed point iterations per step
    float       interval_start  = 0.0f;   // CFG only for t_curr in [start, end]
    float       interval_end    = 1.0f;
};

// Per sample state, lives across the steps of one generation.
struct GuidanceState {
    APGMomentumBuffer  mbuf;
    std::vector<float> prev_error;  // smc_cfg

    explicit GuidanceState(double momentum = -0.75) : mbuf(momentum) {}
};

struct GuidanceStep {
    int   step_idx;
    int   total_steps;
    float dt;  // t_curr - t_next
    float t_curr;
};

static bool guidance_known(const std::string & mode) {
    return mode == "apg" || mode == "cfg_pp" || mode == "dynamic_cfg" || mode == "rescaled_cfg" ||
           mode == "cfg_zero_star" || mode == "smc_cfg" || mode == "cfg_mp" || mode == "adg";
}

// ADG (ACE-Step 1.5 adg_forward), per frame of Oc channels: rotate the
// conditional x0 estimate away from the unconditional one by the scaled,
// clipped angle between them. xt is the current latent [T, Oc], t = t_curr.
static void guidance_adg(const float * xt,
                         const float * pred_cond,
                         const float * pred_uncond,
                         float         scale,
                         float         t,
                         float *       result,
                         int           Oc,
                         int           T) {
    const double        w     = (scale - 1.0 > 0.0 ? scale - 1.0 : 0.0) + 1e-3;
    const double        clip  = 3.14 / 6.0;
    const double        sigma = t > 1e-6f ? t : 1e-6;
    std::vector<double> lt(Oc), lu(Oc), perp(Oc);
    for (int f = 0; f < T; f++) {
        const float * x      = xt + (size_t) f * Oc;
        const float * vc     = pred_cond + (size_t) f * Oc;
        const float * vu     = pred_uncond + (size_t) f * Oc;
        double        dot_tu = 0.0, nt = 0.0, nu = 0.0;
        for (int c = 0; c < Oc; c++) {
            lt[c] = x[c] - sigma * vc[c];
            lu[c] = x[c] - sigma * vu[c];
            dot_tu += lt[c] * lu[c];
            nt += lt[c] * lt[c];
            nu += lu[c] * lu[c];
        }
        double cos_theta = dot_tu / (sqrt(nt) * sqrt(nu) + 1e-12);
        cos_theta        = fmin(fmax(cos_theta, -1.0 + 1e-6), 1.0 - 1e-6);
        double theta     = acos(cos_theta);
        double theta_new = fmin(fmax(w * theta, -clip), clip);
        // perpendicular part of (lt - lu) relative to lu
        double dot_du    = 0.0;
        for (int c = 0; c < Oc; c++) {
            dot_du += (lt[c] - lu[c]) * lu[c];
        }
        double k = dot_du / (nu + 1e-8);
        for (int c = 0; c < Oc; c++) {
            perp[c] = (lt[c] - lu[c]) - k * lu[c];
        }
        double  s_theta = sin(theta);
        double  p_gain  = s_theta > 1e-3 ? sin(theta_new) / s_theta : w;
        double  v_gain  = cos(theta_new);
        float * out     = result + (size_t) f * Oc;
        for (int c = 0; c < Oc; c++) {
            double latent_new = v_gain * lt[c] + p_gain * perp[c];
            out[c]            = (float) ((x[c] - latent_new) / sigma);
        }
    }
}

// Combine one sample's predictions [T, Oc] into the guided velocity. xt is
// the sample's current latent, read by adg.
static void guidance_apply(const GuidanceParams & p,
                           const GuidanceStep &   st,
                           const float *          xt,
                           const float *          pred_cond,
                           const float *          pred_uncond,
                           float                  scale,
                           GuidanceState &        gs,
                           float *                result,
                           int                    Oc,
                           int                    T) {
    const int n = Oc * T;

    if (st.t_curr < p.interval_start || st.t_curr > p.interval_end) {
        memcpy(result, pred_cond, (size_t) n * sizeof(float));
        return;
    }
    if (p.mode == "adg") {
        guidance_adg(xt, pred_cond, pred_uncond, scale, st.t_curr, result, Oc, T);
        return;
    }

    if (p.mode == "cfg_zero_star" && st.step_idx < p.zero_init_steps) {
        memset(result, 0, (size_t) n * sizeof(float));
        return;
    }

    float effective = scale;
    if (p.mode == "cfg_pp" && st.t_curr > 1e-6f) {
        effective = 1.0f + (scale - 1.0f) * (fabsf(st.dt) / st.t_curr);
    } else if (p.mode == "dynamic_cfg") {
        float progress = (float) st.step_idx / fmaxf((float) (st.total_steps - 1), 1.0f);
        // cos(pi/2) rounds to a tiny negative in float, clamp before the sqrt
        float decay    = sqrtf(fmaxf(cosf((float) M_PI / 2.0f * progress), 0.0f));
        effective      = 1.0f + (scale - 1.0f) * decay;
    }

    apg_forward(pred_cond, pred_uncond, effective, gs.mbuf, result, Oc, T, p.norm_threshold);

    if (p.mode == "rescaled_cfg") {
        float  phi = scale > 4.0f ? 0.95f : 0.7f;
        double sc = 0.0, sc2 = 0.0, sg = 0.0, sg2 = 0.0;
        for (int i = 0; i < n; i++) {
            double c = pred_cond[i], g = result[i];
            sc += c;
            sc2 += c * c;
            sg += g;
            sg2 += g * g;
        }
        double mc = sc / n, mg = sg / n;
        double vc = sc2 / n - mc * mc, vg = sg2 / n - mg * mg;
        double std_c  = vc > 0.0 ? sqrt(vc) : 1e-5;
        double std_g  = vg > 0.0 ? sqrt(vg) : 1e-5;
        double factor = std_c / (std_g + 1e-5);
        for (int i = 0; i < n; i++) {
            float rescaled = (float) (result[i] * factor);
            result[i]      = phi * rescaled + (1.0f - phi) * result[i];
        }
    } else if (p.mode == "smc_cfg") {
        // e = cond - uncond, surface s = de/dt + lambda * e, correction -k * sign(s)
        std::vector<float> error(n);
        for (int i = 0; i < n; i++) {
            error[i] = pred_cond[i] - pred_uncond[i];
        }
        if (st.step_idx > 0 && (int) gs.prev_error.size() == n) {
            float inv_dt = 1.0f / fmaxf(fabsf(st.dt), 1e-8f);
            for (int i = 0; i < n; i++) {
                float e_dot = (error[i] - gs.prev_error[i]) * inv_dt;
                float s     = e_dot + p.smc_lambda * error[i];
                float sign  = s > 0.0f ? 1.0f : (s < 0.0f ? -1.0f : 0.0f);
                result[i] += scale * (-p.smc_k * sign);
            }
        }
        gs.prev_error = std::move(error);
    }
}
