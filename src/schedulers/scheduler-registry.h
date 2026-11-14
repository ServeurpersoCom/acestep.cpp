#pragma once
// Compile time registry mapping scheduler names to timestep spacing functions.
//
// A scheduler fills schedule[num_steps] with descending timesteps in (0, 1],
// then applies the shift warp. "linear" is the original ACE Step schedule.
//
// Parameterized names:
//   power:<p>                    t = (1 - i/N)^p
//   beta:<a>:<b>                 Beta(a, b) quantiles
//   composite:<A>+<B>:<x>:<s>    blend of A and B around split s (0..1) with
//                                a crossover zone of width x (0..1)

#include "scheduler-implementations.h"
#include "scheduler-interface.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct SchedulerInfo {
    const char * name;
    const char * display_name;
    SchedulerFn  fn;
};

static const SchedulerInfo SCHEDULER_REGISTRY[] = {
    { "linear",           "Linear",           scheduler_linear           },
    { "ddim_uniform",     "DDIM Uniform",     scheduler_ddim_uniform     },
    { "sgm_uniform",      "SGM Uniform",      scheduler_sgm_uniform      },
    { "bong_tangent",     "Tangent",          scheduler_bong_tangent     },
    { "linear_quadratic", "Linear Quadratic", scheduler_linear_quadratic },
    { "cosine",           "Cosine",           scheduler_cosine           },
    { "power",            "Power",            scheduler_power            },
    { "beta57",           "Beta 57",          scheduler_beta57           },
};

static const int SCHEDULER_REGISTRY_SIZE = (int) (sizeof(SCHEDULER_REGISTRY) / sizeof(SCHEDULER_REGISTRY[0]));

// Look up a plain scheduler name. "karras" is an alias of sgm_uniform.
static const SchedulerInfo * scheduler_lookup(const char * name) {
    if (!name || !name[0]) {
        return &SCHEDULER_REGISTRY[0];
    }
    if (strcmp(name, "karras") == 0) {
        name = "sgm_uniform";
    }
    for (int i = 0; i < SCHEDULER_REGISTRY_SIZE; i++) {
        if (strcmp(SCHEDULER_REGISTRY[i].name, name) == 0) {
            return &SCHEDULER_REGISTRY[i];
        }
    }
    return nullptr;
}

// Fill out[num_steps] for a scheduler spec, plain or parameterized.
// Returns false when the spec names no known scheduler.
static bool scheduler_build(const std::string & spec, float * out, int num_steps, float shift) {
    if (spec.rfind("power:", 0) == 0) {
        float p = (float) atof(spec.c_str() + 6);
        scheduler_power_exp(out, num_steps, shift, p > 0.0f ? p : 2.0f);
        return true;
    }
    if (spec.rfind("beta:", 0) == 0) {
        const char * a_str = spec.c_str() + 5;
        const char * colon = strchr(a_str, ':');
        double       a     = atof(a_str);
        double       b     = colon ? atof(colon + 1) : 0.7;
        if (a <= 0.0 || b <= 0.0) {
            return false;
        }
        scheduler_beta_custom(out, num_steps, shift, a, b);
        return true;
    }
    if (spec.rfind("composite:", 0) == 0) {
        std::string body = spec.substr(10);
        size_t      plus = body.find('+');
        if (plus == std::string::npos) {
            return false;
        }
        std::string name_a = body.substr(0, plus);
        std::string rest   = body.substr(plus + 1);
        size_t      c1     = rest.find(':');
        std::string name_b = rest.substr(0, c1);
        float       cross  = 0.0f;
        float       split  = 0.5f;
        if (c1 != std::string::npos) {
            cross     = (float) atof(rest.c_str() + c1 + 1);
            size_t c2 = rest.find(':', c1 + 1);
            if (c2 != std::string::npos) {
                split = (float) atof(rest.c_str() + c2 + 1);
            }
        }
        cross = cross < 0.0f ? 0.0f : (cross > 1.0f ? 1.0f : cross);
        split = split < 0.0f ? 0.0f : (split > 1.0f ? 1.0f : split);

        std::vector<float> va(num_steps), vb(num_steps);
        if (!scheduler_build(name_a, va.data(), num_steps, shift) ||
            !scheduler_build(name_b, vb.data(), num_steps, shift)) {
            return false;
        }
        float lo = split - cross * 0.5f;
        float hi = split + cross * 0.5f;
        for (int i = 0; i < num_steps; i++) {
            float frac = (float) i / (float) num_steps;
            float w;
            if (cross < 1e-6f || frac <= lo) {
                w = frac < split ? 0.0f : 1.0f;
            } else if (frac >= hi) {
                w = 1.0f;
            } else {
                w = (frac - lo) / (hi - lo);
            }
            out[i] = (1.0f - w) * va[i] + w * vb[i];
        }
        // keep the schedule monotonically descending across the blend
        for (int i = 1; i < num_steps; i++) {
            if (out[i] > out[i - 1]) {
                out[i] = out[i - 1];
            }
        }
        scheduler_clamp(out, num_steps);
        return true;
    }
    const SchedulerInfo * info = scheduler_lookup(spec.c_str());
    if (!info) {
        return false;
    }
    info->fn(out, num_steps, shift);
    return true;
}
