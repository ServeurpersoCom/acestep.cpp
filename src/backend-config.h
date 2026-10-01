#pragma once
// Explicit CPU thread count and compute-device selection for embedders (#19).
//
// An app calls ace_backend_configure() once, before its first model load, instead
// of relying on the GGML_BACKEND environment variable -- an app does not configure
// itself with getenv. The env var stays as a fallback for the CLIs: when nothing
// has been configured, backend init still honours GGML_BACKEND, then auto-picks
// the best device.
//
// All the setters below are setup-time only and unsynchronized: the engine reads
// this config (holding a reference to the device string) at the first model load,
// so set it before any generate/load call and never from another thread while one
// is running. It is set-once configuration, not a live control surface.
//
// This header is deliberately free of any ggml dependency so an embedder can
// include just it to configure the engine; backend.h reads the config. Like the
// rest of the engine's API it is C++ (not extern "C"): a Swift/Obj-C app calls it
// from its C++ or Objective-C++ bridge (the same shim it already needs for
// ace_synth_* etc.), not from Swift directly.
#include <string>
#include <thread>

#ifdef __APPLE__
#    include <sys/sysctl.h>
#endif

struct AceBackendConfig {
    std::string device;         // backend name (e.g. "Metal", "CUDA0", "CPU"); "" = auto
    int         n_threads = 0;  // CPU worker threads; <= 0 = auto
};

// The one config instance, shared across translation units that link into the
// same image: a function-local static inside an inline (external-linkage) function
// has one instance per linked image, so the app's ace_backend_configure() and the
// engine's backend init read the same object even in different TUs. acestep-core
// links statically (build-ios-libs.sh, the CMake STATIC lib), so that image is the
// whole app -- one instance. If the engine were ever built as a separate
// dylib/framework, this static could duplicate across the boundary; move it to a
// .cpp (one external definition) then.
inline AceBackendConfig & ace_backend_config() {
    static AceBackendConfig cfg;
    return cfg;
}

// Set the compute device: backend name, or NULL/"" for auto (falls back to the
// GGML_BACKEND env var, then the best available device).
inline void ace_backend_set_device(const char * device) {
    ace_backend_config().device = device ? device : "";
}

// Set the CPU worker thread count. This one value governs both workloads: the GGML
// inference threads and the MP3 export encoder threads. A positive count caps both
// (an embedder bounding CPU for battery/thermals means the whole pipeline). <= 0
// means auto, and the auto default differs by workload because their optimal thread
// policies differ: GGML uses the performance-core count on Apple silicon
// (hardware_concurrency()/2 elsewhere) since GEMM shares SIMD units across
// hyperthreads, while MP3 -- ALU-bound with a small working set -- uses all logical
// cores. There is deliberately one knob, not one per workload.
inline void ace_backend_set_threads(int n_threads) {
    ace_backend_config().n_threads = n_threads;
}

// Set BOTH device and thread count at once -- a convenience for the common
// configure-once-at-startup call. It sets the full config, so passing e.g.
// (nullptr, 8) also resets the device to auto; to change only one field later use
// the single-field setters above. Not thread-safe: call during setup, before any
// generate call.
inline void ace_backend_configure(const char * device, int n_threads) {
    ace_backend_set_device(device);
    ace_backend_set_threads(n_threads);
}

// The logical CPU count: std::thread::hardware_concurrency(), with an Apple sysctl
// fallback (hw.logicalcpu) for the rare case it returns 0, so a caller that uses the
// logical count as an auto default or a clamp ceiling gets a real number instead of
// collapsing to 1. Always >= 1. Shared (ggml-free) so backend.h and audio-io.h both
// source their logical-CPU reference here rather than each calling the raw query.
inline int ace_logical_cpus() {
    int n = (int) std::thread::hardware_concurrency();
    if (n > 0) {
        return n;
    }
#ifdef __APPLE__
    int    logical = 0;
    size_t sz      = sizeof(logical);
    if (sysctlbyname("hw.logicalcpu", &logical, &sz, nullptr, 0) == 0 && logical > 0) {
        return logical;
    }
#endif
    return 1;
}

// Resolve a CPU worker-thread count from an embedder's configured value. One
// definition, shared by the GGML backend (backend.h) and the MP3 encoder
// (audio-io.h), so the clamp rule stays in one place and cannot drift between them.
//   configured <= 0 -> auto: return auto_threads (each caller's own default policy)
//   configured  > 0 -> honour it, clamped to a ceiling so a bogus value (100000)
//                      cannot spawn that many threads. The ceiling is the logical
//                      CPU count when known (hw), else auto_threads -- so the clamp
//                      still bites when hardware_concurrency() reports 0.
// auto_threads is the caller's own auto default (the P-core count for GGML GEMM, all
// logical cores for the ALU-bound MP3 path); it is forced to >= 1, so the result is
// always >= 1. Pure arithmetic -- no ggml, no platform detection -- so it belongs
// with the rest of this ggml-free config header.
inline int ace_resolve_threads(int configured, int hw, int auto_threads) {
    if (auto_threads < 1) {
        auto_threads = 1;
    }
    if (configured <= 0) {
        return auto_threads;
    }
    const int ceiling = hw > 0 ? hw : auto_threads;
    return configured > ceiling ? ceiling : configured;
}
