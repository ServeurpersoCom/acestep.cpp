#pragma once
// Compile time registry mapping solver names to step functions.
//
// To add a solver later: include its header and append one row to
// SOLVER_REGISTRY. Nothing else changes in the sampler.
//
// Usage:
//   const SolverInfo * info = solver_lookup("stork4");
//   info->step_fn(xt, vt, t_curr, t_prev, n, state, model_fn, vt_buf);

#include "solver-aflops.h"
#include "solver-dopri.h"
#include "solver-dpm.h"
#include "solver-euler.h"
#include "solver-gl2s.h"
#include "solver-heun.h"
#include "solver-interface.h"
#include "solver-jkass.h"
#include "solver-rfsolver.h"
#include "solver-rk4.h"
#include "solver-sde.h"
#include "solver-stork.h"
#include "solver-unipc.h"

#include <cstring>

struct SolverInfo {
    const char * name;           // internal identifier, lowercase.
    const char * display_name;   // human readable name for logs and UI.
    SolverStepFn step_fn;
    int          nfe;            // model evaluations per step, 0 when adaptive.
    int          order;          // ODE integration order.
    bool         is_stateful;    // true when the solver maintains history.
    bool         injects_noise;  // true when the solver re-injects fresh noise at each step.
};

static const SolverInfo SOLVER_REGISTRY[] = {
    // single evaluation
    { "euler",         "ODE Euler",             solver_euler_step,         1,  1, false, false },
    { "sde",           "SDE Ancestral",         solver_sde_step,           1,  1, false, true  },
    { "dpm2m",         "DPM++ 2M",              solver_dpm2m_step,         1,  2, true,  false },
    { "dpm2m_ada",     "DPM++ 2M Adaptive",     solver_dpm2m_ada_step,     1,  2, true,  false },
    { "dpm3m",         "DPM++ 3M",              solver_dpm3m_step,         1,  3, true,  false },
    { "stork2",        "STORK 2",               solver_stork2_step,        1,  2, true,  false },
    { "stork4",        "STORK 4",               solver_stork4_step,        1,  4, true,  false },
    { "unipc_p",       "UniPC Predictor",       solver_unipc_p_step,       1,  2, true,  false },
    { "aflops",        "A-FloPS",               solver_aflops_step,        1,  2, true,  false },
    { "jkass_fast",    "JKASS Fast",            solver_jkass_fast_step,    1,  1, true,  false },
    // multi evaluation, model_fn re evaluates the DiT inside the step
    { "heun",          "Heun",                  solver_heun_step,          2,  2, false, false },
    { "rfsolver",      "RF-Solver",             solver_rfsolver_step,      2,  2, false, false },
    { "unipc",         "UniPC",                 solver_unipc_step,         2,  2, true,  false },
    { "aflops2",       "A-FloPS Midpoint",      solver_aflops2_step,       2,  2, false, false },
    { "jkass_quality", "JKASS Quality",         solver_jkass_quality_step, 2,  2, false, false },
    { "rk4",           "RK4",                   solver_rk4_step,           4,  4, false, false },
    { "rk5",           "RK5",                   solver_rk5_step,           6,  5, false, false },
    { "gl2s",          "Gauss-Legendre 2s",     solver_gl2s_step,          6,  4, false, false },
    { "dopri5",        "Dormand-Prince 5",      solver_dopri5_step,        0,  5, false, false },
    { "dop853",        "Dormand-Prince 8(5,3)", solver_dop853_step,        13, 8, false, false },
};

static const int SOLVER_REGISTRY_SIZE = (int) (sizeof(SOLVER_REGISTRY) / sizeof(SOLVER_REGISTRY[0]));

// Look up a solver by name. Returns nullptr when the name is unknown.
static const SolverInfo * solver_lookup(const char * name) {
    if (!name || !name[0]) {
        return &SOLVER_REGISTRY[0];
    }

    for (int i = 0; i < SOLVER_REGISTRY_SIZE; i++) {
        if (strcmp(SOLVER_REGISTRY[i].name, name) == 0) {
            return &SOLVER_REGISTRY[i];
        }
    }
    return nullptr;
}
