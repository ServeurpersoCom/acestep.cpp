#pragma once
// Compile time registry mapping solver names to step functions.
//
// To add a solver later: include its header and append one row to
// SOLVER_REGISTRY. Nothing else changes in the sampler.
//
// Usage:
//   const SolverInfo * info = solver_lookup("stork4");
//   info->step_fn(xt, vt, t_curr, t_prev, n, state, model_fn, vt_buf);

#include "solver-dpm.h"
#include "solver-euler.h"
#include "solver-interface.h"
#include "solver-sde.h"
#include "solver-stork.h"

#include <cstring>

struct SolverInfo {
    const char * name;           // internal identifier, lowercase.
    const char * display_name;   // human readable name for logs and UI.
    SolverStepFn step_fn;
    int          nfe;            // model evaluations per step.
    int          order;          // ODE integration order.
    bool         is_stateful;    // true when the solver maintains history.
    bool         injects_noise;  // true when the solver re-injects fresh noise at each step.
};

static const SolverInfo SOLVER_REGISTRY[] = {
    { "euler",  "ODE Euler",     solver_euler_step,  1, 1, false, false },
    { "sde",    "SDE Ancestral", solver_sde_step,    1, 1, false, true  },
    { "dpm3m",  "DPM++ 3M",      solver_dpm3m_step,  1, 3, true,  false },
    { "stork4", "STORK 4",       solver_stork4_step, 1, 4, true,  false },
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
