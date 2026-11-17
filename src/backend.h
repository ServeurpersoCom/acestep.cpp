#pragma once
// backend.h: shared GGML backend initialization
//
// All modules use the same pattern: load all backends, pick best GPU,
// keep CPU as fallback. This avoids duplicating init logic across
// qwen3.h, qwen3-lm.h, cond.h, dit.h, vae.h.

#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

struct BackendPair {
    ggml_backend_t backend;
    ggml_backend_t cpu_backend;
    bool           has_gpu;
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

// Collapse exact consecutive duplicate ggml log lines and report the total
// count when the run ends. Debug lines are dropped, with the continuation
// lines that follow them: the CUDA backend writes "CUDA Graph id N reused" at
// that level on every graph launch, a different id each time, hundreds of
// lines a job.
static void acestep_ggml_log(enum ggml_log_level level, const char * text, void * user_data) {
    (void) user_data;
    static char                last[256]  = { 0 };
    static int                 count      = 0;
    static enum ggml_log_level last_level = GGML_LOG_LEVEL_NONE;

    if (level == GGML_LOG_LEVEL_CONT) {
        level = last_level;
    } else {
        last_level = level;
    }
    if (level == GGML_LOG_LEVEL_DEBUG) {
        return;
    }

    if (count > 0 && strcmp(text, last) == 0) {
        count++;
        return;
    }

    if (count > 1) {
        fprintf(stderr, "[Dedup] Previous line repeated %d times total\n", count);
    }

    fputs(text, stderr);
    strncpy(last, text, sizeof(last) - 1);
    last[sizeof(last) - 1] = 0;
    count                  = 1;
    fflush(stderr);
}

// Device named by --device, NULL for the best available one. Inline so
// every translation unit of the binary sees the one main sets.
inline const char * g_backend_device = NULL;

// Install the log callback and load every backend, once per process.
static void backend_load(void) {
    static const bool loaded = [] {
        ggml_log_set(acestep_ggml_log, nullptr);
        ggml_backend_load_all();
        return true;
    }();
    (void) loaded;
}

// Select the device of every later backend_init by name (CUDA0, Vulkan0,
// MTL0, CPU...). Returns false and lists the available devices when the
// name matches none.
static bool backend_set_device(const char * name) {
    backend_load();
    if (!ggml_backend_dev_by_name(name)) {
        fprintf(stderr, "--device %s not found. Available:", name);
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            fprintf(stderr, " %s", ggml_backend_dev_name(ggml_backend_dev_get(i)));
        }
        fprintf(stderr, "\n");
        return false;
    }
    g_backend_device = name;
    return true;
}

// Initialize backends: the device named by --device or the best available
// one (CUDA, Metal, Vulkan...), keep CPU as fallback.
// label: log prefix, e.g. "DiT", "VAE", "LM"
// Subsequent calls reuse the same backend (single VMM pool).
static BackendPair backend_init(const char * label) {
    backend_load();

    if (g_backend_refs > 0) {
        g_backend_refs++;
        fprintf(stderr, "[Load] %s backend: %s (shared)\n", label, ggml_backend_name(g_backend_cache.backend));
        return g_backend_cache;
    }

    BackendPair bp = {};
    bp.backend     = g_backend_device ? ggml_backend_init_by_name(g_backend_device, nullptr) : ggml_backend_init_best();
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
    bp.has_gpu = !best_is_cpu;
    fprintf(stderr, "[Load] %s backend: %s (CPU threads: %d)\n", label, ggml_backend_name(bp.backend), n_threads);

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
// When a GPU is present, use its host buffer type for the CPU backend.
// Pinned memory lets the scheduler keep more ops on GPU instead of
// falling back to CPU with plain malloc.
static ggml_backend_sched_t backend_sched_new(BackendPair bp, int max_nodes) {
    ggml_backend_t             backends[2] = { bp.backend, bp.cpu_backend };
    ggml_backend_buffer_type_t bufts[2]    = { NULL, NULL };
    int                        n           = (bp.backend == bp.cpu_backend) ? 1 : 2;

    bufts[0] = ggml_backend_get_default_buffer_type(bp.backend);
    if (n == 2) {
        ggml_backend_dev_t         gpu_dev   = ggml_backend_get_device(bp.backend);
        ggml_backend_buffer_type_t host_buft = gpu_dev ? ggml_backend_dev_host_buffer_type(gpu_dev) : NULL;
        bufts[1] = host_buft ? host_buft : ggml_backend_get_default_buffer_type(bp.cpu_backend);
    }

    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, n, max_nodes, false, true);
    if (!sched) {
        fprintf(stderr, "[Load] FATAL: failed to create scheduler\n");
        exit(1);
    }
    return sched;
}
