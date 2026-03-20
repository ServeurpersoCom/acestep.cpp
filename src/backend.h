#pragma once
// backend.h: shared GGML backend initialization
//
// All modules use the same pattern: load all backends, pick best GPU,
// keep CPU as fallback. This avoids duplicating init logic across
// qwen3.h, qwen3-lm.h, cond.h, dit.h, vae.h.

#include "ggml-backend.h"
#ifdef ACESTEP_HAVE_CUDA
// Query compute capability without pulling in cuda_runtime.h.
// cudaDeviceGetAttribute takes an int enum value; we pass the raw constants.
extern "C" int cudaDeviceGetAttribute(int *, int, int);
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

struct BackendPair {
    ggml_backend_t backend;
    ggml_backend_t cpu_backend;
    int            gpu_cc;  // CUDA compute capability (e.g. 720 for sm_72), 0 if not CUDA
};

// Cached backend state (shared across all modules in the same binary)
static BackendPair g_backend_cache = {};
static int         g_backend_refs  = 0;

// Physical core count heuristic (logical / 2 for HT/SMT).
// Used for GGML CPU thread count: GEMM shares SIMD units across hyperthreads,
// so one thread per physical core is optimal.
static int backend_cpu_n_threads(void) {
    int n = (int) std::thread::hardware_concurrency() / 2;
    return n > 0 ? n : 1;
}

// Standalone CPU backend via Registry API (DL-safe, no ggml-cpu.h needed).
// Sets thread count via proc address since ggml_backend_cpu_device_init_backend
// ignores its params string and always defaults to GGML_DEFAULT_N_THREADS (4).
// Returns NULL on failure.
static ggml_backend_t cpu_backend_new(int n_threads) {
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t     cpu     = NULL;
    if (cpu_dev) {
        cpu = ggml_backend_dev_init(cpu_dev, NULL);
    }
    if (!cpu) {
        cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    }
    if (!cpu) {
        return NULL;
    }

    ggml_backend_dev_t dev = ggml_backend_get_device(cpu);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : NULL;
    if (reg) {
        auto set_fn =
            (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (set_fn) {
            set_fn(cpu, n_threads);
        }
    }
    return cpu;
}

// Initialize backends: load all available (CUDA, Metal, Vulkan...),
// pick the best one, keep CPU as fallback.
// label: log prefix, e.g. "DiT", "VAE", "LM"
// Subsequent calls reuse the same backend (single VMM pool).
static BackendPair backend_init(const char * label) {
    if (g_backend_refs > 0) {
        g_backend_refs++;
        fprintf(stderr, "[Load] %s backend: %s (shared)\n", label, ggml_backend_name(g_backend_cache.backend));
        return g_backend_cache;
    }

    ggml_backend_load_all();
    BackendPair bp = {};

    // GGML_BACKEND env var: force a specific device instead of auto-best.
    // Device names: CUDA0, Vulkan0, CPU, BLAS (see ggml_backend_dev_name).
    const char * force_backend = std::getenv("GGML_BACKEND");
    if (force_backend) {
        bp.backend = ggml_backend_init_by_name(force_backend, nullptr);
        if (!bp.backend) {
            fprintf(stderr, "[Load] FATAL: GGML_BACKEND=%s not found. Available:", force_backend);
            for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                fprintf(stderr, " %s", ggml_backend_dev_name(ggml_backend_dev_get(i)));
            }
            fprintf(stderr, "\n");
            exit(1);
        }
    } else {
        bp.backend = ggml_backend_init_best();
    }
    if (!bp.backend) {
        fprintf(stderr, "[Load] FATAL: no backend available\n");
        exit(1);
    }
    bool best_is_cpu = (strcmp(ggml_backend_name(bp.backend), "CPU") == 0);
    int  n_threads   = backend_cpu_n_threads();
    if (best_is_cpu) {
        ggml_backend_free(bp.backend);
        bp.backend     = cpu_backend_new(n_threads);
        bp.cpu_backend = bp.backend;
    } else {
        bp.cpu_backend = cpu_backend_new(n_threads);
    }
    if (!bp.cpu_backend) {
        fprintf(stderr, "[Load] FATAL: failed to init CPU backend\n");
        exit(1);
    }
    fprintf(stderr, "[Load] %s backend: %s (CPU threads: %d)\n", label, ggml_backend_name(bp.backend), n_threads);

    bp.gpu_cc = 0;
#ifdef ACESTEP_HAVE_CUDA
    if (!best_is_cpu) {
        int major = 0, minor = 0;
        cudaDeviceGetAttribute(&major, 75, 0);  // cudaDevAttrComputeCapabilityMajor
        cudaDeviceGetAttribute(&minor, 76, 0);  // cudaDevAttrComputeCapabilityMinor
        bp.gpu_cc = major * 100 + minor * 10;
    }
#endif

    g_backend_cache = bp;
    g_backend_refs  = 1;
    return bp;
}

// Release a backend reference. Frees GPU + CPU backends when refcount hits 0.
static void backend_release(ggml_backend_t backend, ggml_backend_t cpu_backend) {
    if (g_backend_refs <= 0) {
        return;
    }
    g_backend_refs--;
    if (g_backend_refs == 0) {
        if (backend && backend != cpu_backend) {
            ggml_backend_free(backend);
        }
        if (cpu_backend) {
            ggml_backend_free(cpu_backend);
        }
        g_backend_cache = {};
    }
}

// Create a scheduler from a backend pair.
// max_nodes: graph size hint (4096 for small models, 8192 for large)
static ggml_backend_sched_t backend_sched_new(BackendPair bp, int max_nodes) {
    ggml_backend_t       backends[2] = { bp.backend, bp.cpu_backend };
    int                  n           = (bp.backend == bp.cpu_backend) ? 1 : 2;
    ggml_backend_sched_t sched       = ggml_backend_sched_new(backends, NULL, n, max_nodes, false, true);
    if (!sched) {
        fprintf(stderr, "[Load] FATAL: failed to create scheduler\n");
        exit(1);
    }
    return sched;
}
