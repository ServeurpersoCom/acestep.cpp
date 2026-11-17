#pragma once
// scheduler-interface.h: universal scheduler contract for ACE-Step flow matching
//
// A scheduler defines HOW timesteps are spaced between t=1 (noise) and t=0 (clean).
// This is independent of the solver (which defines how each step is computed).
//
// Scheduler interface:
//   schedule_fn(output, num_steps, shift)
//
// - output[num_steps]: filled with descending timesteps in (0, 1]
// - num_steps: number of diffusion steps
// - shift: timestep shift factor (shift=1 is identity)

// Scheduler step function signature.
// Fills output[num_steps] with descending timesteps, then applies the shift warp.
using SchedulerFn = void (*)(float * output, int num_steps, float shift);

// Apply the standard shift warp: t' = shift*t / (1 + (shift-1)*t)
// Modifies values in-place. When shift == 1.0 this is identity.
static inline void scheduler_apply_shift(float * ts, int n, float shift) {
    if (shift == 1.0f) {
        return;
    }
    for (int i = 0; i < n; i++) {
        float t = ts[i];
        ts[i]   = shift * t / (1.0f + (shift - 1.0f) * t);
    }
}

// Clamp all values to [1e-6, 1.0]
static inline void scheduler_clamp(float * ts, int n) {
    for (int i = 0; i < n; i++) {
        if (ts[i] < 1e-6f) {
            ts[i] = 1e-6f;
        }
        if (ts[i] > 1.0f) {
            ts[i] = 1.0f;
        }
    }
}
