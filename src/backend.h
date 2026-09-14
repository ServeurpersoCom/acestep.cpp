#pragma once
// backend.h: shared GGML backend initialization
//
// All modules use the same pattern: load all backends, pick best GPU,
// keep CPU as fallback. This avoids duplicating init logic across
// qwen3.h, qwen3-lm.h, cond.h, dit.h, vae.h.

#include "backend-config.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#ifdef __APPLE__
#    include <sys/sysctl.h>
#endif

struct BackendPair {
    ggml_backend_t backend;
    ggml_backend_t cpu_backend;
    bool           has_gpu;
};

// Cached backend state, shared across every model that loads within one binary.
// static-in-header (internal linkage) means one copy per TU, not one program-wide
// like ace_backend_config() -- and that is fine only because exactly one TU per
// binary ever reaches backend_init(): in the engine every load goes through
// model-store.cpp's store_require_*, and the neural-codec tool is a standalone TU
// that loads its own VAE. If a second TU in the same binary ever called a loader
// directly, its loads would get a separate cache and refcount and not share --
// make this a shared singleton (ace_backend_config-style) if that day comes.
static BackendPair g_backend_cache = {};
static int         g_backend_refs  = 0;

// The auto GGML CPU thread count: one thread per useful physical core. GEMM
// shares SIMD units across hyperthreads, so one-per-physical is optimal.
static int backend_cpu_auto_threads(void) {
#ifdef __APPLE__
    // Apple silicon has no SMT and asymmetric cores, so logical/2 is the wrong
    // count -- it halves as if for hyperthreads. hw.perflevel0 is the
    // performance-core cluster, the cores GEMM should run on: the E-cores are
    // much slower and just drag a shared graph. That is 12 P-cores on an M3 Max
    // (12P+4E, where logical/2 gave 8) and 2 on an iPhone 15 Pro (2P+4E, where
    // logical/2 gave 3) -- fewer than before on a 2P device, but the fast cores
    // only, which is the one-thread-per-useful-physical-core this function wants.
    // (physicalcpu == logicalcpu here anyway, no SMT; physicalcpu just reads as
    // the honest intent.) The perflevel sysctls exist only on Apple silicon: on an
    // Intel Mac this query fails and control falls through to the logical/2 below,
    // which is the right answer there because Intel does have SMT.
    int    perf = 0;
    size_t sz   = sizeof(perf);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &perf, &sz, nullptr, 0) == 0 && perf > 0) {
        return perf;
    }
#endif
    // Non-Apple: x86 with SMT is the target, where logical / 2 approximates the
    // physical core count. (A non-SMT non-Apple host -- e.g. ARM64 Linux -- would
    // be undercounted, but that is not a platform this engine ships on.)
    int n = (int) std::thread::hardware_concurrency() / 2;
    return n > 0 ? n : 1;
}

// GGML CPU thread count: an embedder override (ace_backend_configure), otherwise
// the auto physical-core count. ace_resolve_threads() (backend-config.h) holds the
// shared policy: an explicit override is trusted -- an embedder may deliberately
// spend all cores including the E-cores -- and only clamped to the logical CPU count
// (ace_logical_cpus(), the same source the MP3 path uses) so an absurd value cannot
// ask GGML to spawn that many threads. The one-thread-per-physical-core reasoning is
// the default's (backend_cpu_auto_threads); an explicit request overrides it.
static int backend_cpu_n_threads(void) {
    return ace_resolve_threads(ace_backend_config().n_threads, ace_logical_cpus(), backend_cpu_auto_threads());
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
// count when the run ends (tames the CUDA graph capture "reused" flood).
static void acestep_ggml_log(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    static char last[256] = { 0 };
    static int  count     = 0;

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

// Initialize backends: load all available (CUDA, Metal, Vulkan...),
// pick the best one, keep CPU as fallback.
// label: log prefix, e.g. "DiT", "VAE", "LM"
// Subsequent calls reuse the same backend (single VMM pool).
static BackendPair backend_init(const char * label) {
    static bool log_installed = false;
    if (!log_installed) {
        ggml_log_set(acestep_ggml_log, nullptr);
        log_installed = true;
    }

    if (g_backend_refs > 0) {
        g_backend_refs++;
        fprintf(stderr, "[Load] %s backend: %s (shared)\n", label, ggml_backend_name(g_backend_cache.backend));
        return g_backend_cache;
    }

    ggml_backend_load_all();
    BackendPair bp = {};

    // Device selection: an explicit ace_backend_configure() wins, then the
    // GGML_BACKEND env var (the CLI fallback), then auto-best below.
    // Device names: CUDA0, Vulkan0, CPU, BLAS (see ggml_backend_dev_name).
    // An *empty* value from either source (config device "" or GGML_BACKEND="")
    // reads as unset and falls through to auto-best -- the `force_backend[0]`
    // guard -- by design; only a non-empty, unknown name is a hard error.
    const std::string & cfg_device    = ace_backend_config().device;
    const bool          from_config   = !cfg_device.empty();
    const char *        force_backend = from_config ? cfg_device.c_str() : std::getenv("GGML_BACKEND");
    if (force_backend && force_backend[0]) {
        bp.backend = ggml_backend_init_by_name(force_backend, nullptr);
        if (!bp.backend) {
            std::string avail;
            for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                avail += std::string(" ") + ggml_backend_dev_name(ggml_backend_dev_get(i));
            }
            fprintf(stderr, "[Load] FATAL: backend '%s' (from %s) not found. Available:%s\n", force_backend,
                    from_config ? "ace_backend_configure" : "GGML_BACKEND", avail.c_str());
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
