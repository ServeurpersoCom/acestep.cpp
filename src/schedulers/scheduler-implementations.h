#pragma once
// scheduler-implementations.h: all scheduler implementations
//
// Each scheduler fills output[num_steps] with descending timesteps.
// The shift warp is applied at the end. No trailing 0.
//
// Matches Python acestep/core/generation/schedulers.py

#include "scheduler-interface.h"

#include <algorithm>
#include <cmath>

#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif

// ===========================================================================
// Linear (uniform) - the original ACE-Step default
// ===========================================================================
static void scheduler_linear(float * output, int num_steps, float shift) {
    for (int i = 0; i < num_steps; i++) {
        output[i] = 1.0f - (float) i / (float) num_steps;
    }
    scheduler_apply_shift(output, num_steps, shift);
}

// ===========================================================================
// DDIM Uniform - log-SNR uniform (S-shaped distribution)
// ===========================================================================
static void scheduler_ddim_uniform(float * output, int num_steps, float shift) {
    // Logit bounds: t=0.9986 -> logit~6.57, t=0.0014 -> logit~-6.57
    const float t_max     = 0.9986f;
    const float t_min     = 0.0014f;
    const float logit_max = logf(t_max / (1.0f - t_max));
    const float logit_min = logf(t_min / (1.0f - t_min));

    for (int i = 0; i < num_steps; i++) {
        float frac    = (float) i / (float) num_steps;
        float logit_t = logit_max + (logit_min - logit_max) * frac;
        output[i]     = 1.0f / (1.0f + expf(-logit_t));  // sigmoid
    }
    scheduler_clamp(output, num_steps);
    scheduler_apply_shift(output, num_steps, shift);
}

// ===========================================================================
// SGM Uniform (Karras) - uniform in sigma^(1/rho) space with rho=7
// ===========================================================================
static void scheduler_sgm_uniform(float * output, int num_steps, float shift) {
    const float t_max     = 0.999f;
    const float t_min     = 0.001f;
    const float sigma_max = t_max / (1.0f - t_max);  // ~999
    const float sigma_min = t_min / (1.0f - t_min);  // ~0.001
    const float rho       = 7.0f;

    const float inv_rho = 1.0f / rho;
    const float s_max   = powf(sigma_max, inv_rho);
    const float s_min   = powf(sigma_min, inv_rho);

    for (int i = 0; i < num_steps; i++) {
        float frac  = (float) i / (float) num_steps;
        float sigma = powf(s_max + frac * (s_min - s_max), rho);
        output[i]   = sigma / (1.0f + sigma);
    }
    scheduler_clamp(output, num_steps);
    scheduler_apply_shift(output, num_steps, shift);
}

// ===========================================================================
// Bong Tangent - tangent-based, concentrates at high noise
// ===========================================================================
static void scheduler_bong_tangent(float * output, int num_steps, float shift) {
    const float scale = 1.5f;

    for (int i = 0; i < num_steps; i++) {
        float frac    = ((float) i + 0.5f) / (float) num_steps;
        float angle   = frac * (float) M_PI / 2.0f;
        float tan_val = tanf(angle);
        output[i]     = 1.0f - (2.0f / (float) M_PI) * atanf(tan_val * scale);
    }

    // Sort descending and clamp
    std::sort(output, output + num_steps, std::greater<float>());
    scheduler_clamp(output, num_steps);
    scheduler_apply_shift(output, num_steps, shift);
}

// ===========================================================================
// Linear-Quadratic - linear start, quadratic finish
// ===========================================================================
static void scheduler_linear_quadratic(float * output, int num_steps, float shift) {
    const float crossover = 0.5f;
    int         n_linear  = (std::max)((int) (num_steps * crossover), 1);
    int         n_quad    = num_steps - n_linear;

    float t_cross = 1.0f - crossover;  // 0.5

    // Linear region: 1.0 -> crossover point
    for (int i = 0; i < n_linear; i++) {
        output[i] = 1.0f - (float) i * crossover / (float) n_linear;
    }

    // Quadratic region: crossover -> 0
    for (int i = 0; i < n_quad; i++) {
        float frac           = (float) (i + 1) / (float) n_quad;
        output[n_linear + i] = t_cross * (1.0f - frac * frac);
    }

    scheduler_clamp(output, num_steps);
    scheduler_apply_shift(output, num_steps, shift);
}

// ===========================================================================
// Cosine - half-cosine curve (balanced S-shape)
// ===========================================================================
static void scheduler_cosine(float * output, int num_steps, float shift) {
    for (int i = 0; i < num_steps; i++) {
        float frac = (float) i / (float) num_steps;
        output[i]  = 0.5f * (1.0f + cosf((float) M_PI * frac));
    }
    scheduler_clamp(output, num_steps);
    scheduler_apply_shift(output, num_steps, shift);
}

// ===========================================================================
// Power - t = (1 - i/N)^p, default p=2
// ===========================================================================
static void scheduler_power_exp(float * output, int num_steps, float shift, float p) {
    for (int i = 0; i < num_steps; i++) {
        float frac = (float) i / (float) num_steps;
        output[i]  = powf(1.0f - frac, p);
    }
    scheduler_clamp(output, num_steps);
    scheduler_apply_shift(output, num_steps, shift);
}

// Exponent 2: front-loaded, more steps at high noise.
static void scheduler_power(float * output, int num_steps, float shift) {
    scheduler_power_exp(output, num_steps, shift, 2.0f);
}

// ===========================================================================
// Beta distribution helpers - for beta57 and generic beta schedules
//
// We implement the regularized incomplete beta function Ix(a,b) and its
// inverse via Newton's method, avoiding scipy dependency.
// ===========================================================================

// Log-gamma function (Stirling approximation + Lanczos for small values)
static double _lgamma_approx(double x) {
#if defined(_MSC_VER) || defined(__GNUC__)
    return lgamma(x);
#else
    // Fallback Stirling
    if (x <= 0.0) {
        return 0.0;
    }
    return (x - 0.5) * log(x) - x + 0.5 * log(2.0 * M_PI) + 1.0 / (12.0 * x);
#endif
}

// Log of the beta function: B(a,b) = Gamma(a)*Gamma(b)/Gamma(a+b)
static double _lbeta(double a, double b) {
    return _lgamma_approx(a) + _lgamma_approx(b) - _lgamma_approx(a + b);
}

// Regularized incomplete beta function Ix(a,b) via continued fraction.
// Clean Numerical-Recipes-style implementation (modified Lentz's method).
// Matches scipy.special.betainc to ~14 digits.

static double _betainc_cf(double a, double b, double x) {
    if (x <= 0.0) {
        return 0.0;
    }
    if (x >= 1.0) {
        return 1.0;
    }

    // Symmetry for convergence
    if (x > (a + 1.0) / (a + b + 2.0)) {
        return 1.0 - _betainc_cf(b, a, 1.0 - x);
    }

    // ln(x^a * (1-x)^b / B(a,b))
    double ln_pre = a * log(x) + b * log(1.0 - x) - _lbeta(a, b);

    // Evaluate continued fraction using modified Lentz's method
    double qab = a + b;
    double qap = a + 1.0;
    double qam = a - 1.0;

    double c = 1.0;
    double d = 1.0 - qab * x / qap;
    if (fabs(d) < 1e-30) {
        d = 1e-30;
    }
    d        = 1.0 / d;
    double h = d;

    for (int m = 1; m <= 200; m++) {
        int m2 = 2 * m;

        // Even numerator
        double aa = (double) m * (b - (double) m) * x / ((qam + (double) m2) * (a + (double) m2));
        d         = 1.0 + aa * d;
        if (fabs(d) < 1e-30) {
            d = 1e-30;
        }
        c = 1.0 + aa / c;
        if (fabs(c) < 1e-30) {
            c = 1e-30;
        }
        d = 1.0 / d;
        h *= d * c;

        // Odd numerator
        aa = -((a + (double) m) * (qab + (double) m) * x) / ((a + (double) m2) * (qap + (double) m2));
        d  = 1.0 + aa * d;
        if (fabs(d) < 1e-30) {
            d = 1e-30;
        }
        c = 1.0 + aa / c;
        if (fabs(c) < 1e-30) {
            c = 1e-30;
        }
        d          = 1.0 / d;
        double del = d * c;
        h *= del;

        if (fabs(del - 1.0) < 3e-14) {
            break;
        }
    }

    return exp(ln_pre) * h / a;
}

// Beta distribution PDF: f(x; a, b) = x^(a-1) * (1-x)^(b-1) / B(a,b)
static double _beta_pdf(double x, double a, double b) {
    if (x <= 0.0 || x >= 1.0) {
        return 0.0;
    }
    return exp((a - 1.0) * log(x) + (b - 1.0) * log(1.0 - x) - _lbeta(a, b));
}

// Inverse CDF of Beta distribution via Newton's method.
// Finds x such that Ix(a,b) = p.
static double _beta_ppf(double p, double a, double b) {
    if (p <= 0.0) {
        return 0.0;
    }
    if (p >= 1.0) {
        return 1.0;
    }

    // Initial guess using approximation
    double x = 0.5;

    // Better initial guess: use normal approximation for the mean
    double mu    = a / (a + b);
    double var   = a * b / ((a + b) * (a + b) * (a + b + 1.0));
    double sigma = sqrt(var);
    // Crude: use mu as start, nudge by p
    x            = mu + sigma * (2.0 * p - 1.0);
    if (x < 0.001) {
        x = 0.001;
    }
    if (x > 0.999) {
        x = 0.999;
    }

    // Newton-Raphson iteration
    for (int iter = 0; iter < 50; iter++) {
        double F = _betainc_cf(a, b, x) - p;
        double f = _beta_pdf(x, a, b);
        if (fabs(f) < 1e-30) {
            break;
        }

        double dx = -F / f;
        x += dx;

        // Keep in bounds
        if (x < 1e-10) {
            x = 1e-10;
        }
        if (x > 1.0 - 1e-10) {
            x = 1.0 - 1e-10;
        }

        if (fabs(dx) < 1e-12) {
            break;
        }
    }
    return x;
}

// ===========================================================================
// Beta57 - Beta(0.5, 0.7) distribution schedule
// ===========================================================================
static void scheduler_beta57(float * output, int num_steps, float shift) {
    const double alpha = 0.5;
    const double beta  = 0.7;

    for (int i = 0; i < num_steps; i++) {
        double u  = ((double) i + 0.5) / (double) num_steps;
        double t  = 1.0 - _beta_ppf(u, alpha, beta);
        output[i] = (float) t;
    }

    // Sort descending and clamp
    std::sort(output, output + num_steps, std::greater<float>());
    scheduler_clamp(output, num_steps);
    scheduler_apply_shift(output, num_steps, shift);
}

// ===========================================================================
// Beta (custom) - parameterized Beta(alpha, beta) distribution schedule
// Called from ops_build_schedule when scheduler string is "beta:<a>:<b>"
// ===========================================================================
static void scheduler_beta_custom(float * output, int num_steps, float shift, double alpha, double beta) {
    for (int i = 0; i < num_steps; i++) {
        double u  = ((double) i + 0.5) / (double) num_steps;
        double t  = 1.0 - _beta_ppf(u, alpha, beta);
        output[i] = (float) t;
    }
    std::sort(output, output + num_steps, std::greater<float>());
    scheduler_clamp(output, num_steps);
    scheduler_apply_shift(output, num_steps, shift);
}
