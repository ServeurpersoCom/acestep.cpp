#pragma once
// safetensors.h: Safetensors mmap parser + ggml tensor loader.
// Loads BF16/F16/F32 weights directly into ggml backend buffers.
// No GGUF conversion. Zero-copy mmap where possible.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>

// Safetensors mmap parser (no ggml deps)
enum SafeDType { SF_BF16 = 0, SF_F16 = 1, SF_F32 = 2 };

struct SafeTensor {
    SafeDType dtype;
    std::vector<int64_t> shape;
    const void *data;      // mmap'd pointer to raw data
    size_t nbytes;
};

struct SafeFile {
    int fd;
    void *mapping;
    size_t file_size;
};

struct SafeTensors {
    std::unordered_map<std::string, SafeTensor> tensors;
    std::vector<SafeFile> files;  // keep mmaps alive

    ~SafeTensors() {
        for (auto &f : files) {
            if (f.mapping) munmap(f.mapping, f.file_size);
            if (f.fd >= 0) close(f.fd);
        }
    }
};

// Minimal JSON field extractor (no deps, works for safetensors headers)
static inline std::string json_str(const char *p, const char *key) {
    char needle[256];
    snprintf(needle, sizeof(needle), "\"%s\":", key);
    const char *f = strstr(p, needle);
    if (!f) return "";
    f += strlen(needle);
    while (*f == ' ' || *f == '"') f++;
    const char *e = f;
    while (*e && *e != '"' && *e != ',' && *e != '}') e++;
    return std::string(f, e);
}

static inline int64_t json_int(const char *p, const char *key) {
    std::string s = json_str(p, key);
    return s.empty() ? -1 : strtoll(s.c_str(), nullptr, 10);
}

// Parse shape array: "shape":[4096,2560]
static inline std::vector<int64_t> json_shape(const char *entry) {
    std::vector<int64_t> shape;
    const char *f = strstr(entry, "\"shape\":");
    if (!f) return shape;
    f = strchr(f, '[');
    if (!f) return shape;
    f++;
    while (*f && *f != ']') {
        while (*f == ' ' || *f == ',') f++;
        if (*f == ']') break;
        shape.push_back(strtoll(f, nullptr, 10));
        while (*f && *f != ',' && *f != ']') f++;
    }
    return shape;
}

// Parse data_offsets: "data_offsets":[0,20971520]
static inline void json_offsets(const char *entry, size_t &start, size_t &end) {
    const char *f = strstr(entry, "\"data_offsets\":");
    if (!f) { start = end = 0; return; }
    f = strchr(f, '[');
    if (!f) { start = end = 0; return; }
    f++;
    start = strtoull(f, nullptr, 10);
    f = strchr(f, ',');
    if (!f) { end = start; return; }
    f++;
    end = strtoull(f, nullptr, 10);
}

static inline SafeDType parse_dtype(const char *entry) {
    if (strstr(entry, "BF16") || strstr(entry, "bf16")) return SF_BF16;
    if (strstr(entry, "F16") || strstr(entry, "f16"))   return SF_F16;
    return SF_F32;
}

// Load a single .safetensors file
static bool safe_load_file(SafeTensors &st, const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "Cannot open %s\n", path); return false; }

    struct stat sb;
    fstat(fd, &sb);
    size_t file_size = sb.st_size;

    void *mapping = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapping == MAP_FAILED) { close(fd); fprintf(stderr, "mmap failed %s\n", path); return false; }

    st.files.push_back({fd, mapping, file_size});

    const uint8_t *base = (const uint8_t *)mapping;
    uint64_t header_len = *(const uint64_t *)base;
    const char *header = (const char *)(base + 8);
    const uint8_t *data_base = base + 8 + header_len;

    // Parse each tensor entry from the header JSON
    // Format: "tensor.name": {"dtype":"BF16","shape":[...],"data_offsets":[...]}
    const char *p = header;
    const char *header_end = header + header_len;
    while (p < header_end) {
        const char *q = strchr(p, '"');
        if (!q || q >= header_end) break;
        q++;
        const char *name_end = strchr(q, '"');
        if (!name_end || name_end >= header_end) break;
        std::string name(q, name_end);
        p = name_end + 1;

        const char *brace = strchr(p, '{');
        if (!brace || brace >= header_end) break;
        const char *brace_end = strchr(brace, '}');
        if (!brace_end || brace_end >= header_end) break;

        std::string entry(brace, brace_end + 1);
        p = brace_end + 1;

        if (name == "__metadata__") continue;
        if (entry.find("data_offsets") == std::string::npos) continue;

        SafeTensor t;
        t.dtype = parse_dtype(entry.c_str());
        t.shape = json_shape(entry.c_str());
        size_t off_start, off_end;
        json_offsets(entry.c_str(), off_start, off_end);
        t.data = data_base + off_start;
        t.nbytes = off_end - off_start;
        st.tensors[name] = t;
    }
    fprintf(stderr, "[Safetensors] %s: %zu tensors\n", path, st.tensors.size());
    return true;
}

// Load model directory (handles single file or sharded model-0000X-of-0000Y.safetensors)
static bool safe_load(SafeTensors &st, const char *dir_or_file) {
    struct stat sb;
    if (stat(dir_or_file, &sb) != 0) return false;

    if (S_ISREG(sb.st_mode)) {
        return safe_load_file(st, dir_or_file);
    }

    // Directory: load all .safetensors files sorted
    std::vector<std::string> files;
    DIR *d = opendir(dir_or_file);
    if (!d) return false;
    struct dirent *ent;
    while ((ent = readdir(d))) {
        std::string name = ent->d_name;
        if (name.size() > 12 && name.substr(name.size() - 12) == ".safetensors") {
            files.push_back(std::string(dir_or_file) + "/" + name);
        }
    }
    closedir(d);
    std::sort(files.begin(), files.end());

    for (auto &f : files) {
        if (!safe_load_file(st, f.c_str())) return false;
    }
    return !st.tensors.empty();
}

// Get tensor or die
static const SafeTensor &safe_get(const SafeTensors &st, const std::string &name) {
    auto it = st.tensors.find(name);
    if (it == st.tensors.end()) {
        fprintf(stderr, "FATAL: tensor '%s' not found\n", name.c_str());
        exit(1);
    }
    return it->second;
}

// ggml tensor loader
static ggml_type sf_dtype_to_ggml(SafeDType dt) {
    switch (dt) {
        case SF_BF16: return GGML_TYPE_BF16;
        case SF_F16:  return GGML_TYPE_F16;
        case SF_F32:  return GGML_TYPE_F32;
    }
    return GGML_TYPE_F32;
}

// Weight loading context.
// Manages a ggml_context for weight tensors + their backend buffer.
// Usage:
//   SFWeightCtx wctx;
//   sf_weight_ctx_init(&wctx, n_tensors);
//   ggml_tensor* w = sf_load_tensor(&wctx, st, "layer.0.weight");
//   sf_weight_ctx_alloc(&wctx, backend);
//
// After alloc, all tensors live in the backend buffer (GPU/CPU).
struct SFWeightCtx {
    struct ggml_context * ctx;
    ggml_backend_buffer_t buffer;

    struct PendingCopy {
        struct ggml_tensor * tensor;
        const void * src;
        size_t nbytes;
    };
    std::vector<PendingCopy> pending;
};

static void sf_weight_ctx_init(SFWeightCtx * wctx, int n_tensors) {
    size_t ctx_size = (size_t)n_tensors * ggml_tensor_overhead() + 1024;

    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    wctx->ctx = ggml_init(params);
    wctx->buffer = NULL;
    wctx->pending.clear();
    wctx->pending.reserve(n_tensors);
}

// Load a single tensor from safetensors into the weight context.
// Returns the ggml_tensor (not yet backed by memory, call sf_weight_ctx_alloc after).
static struct ggml_tensor * sf_load_tensor(
        SFWeightCtx * wctx,
        const SafeTensors & st,
        const std::string & name,
        const int64_t * shape_override = nullptr,
        int n_dims_override = 0) {

    auto it = st.tensors.find(name);
    if (it == st.tensors.end()) {
        fprintf(stderr, "[Safetensors] FATAL: tensor '%s' not found\n", name.c_str());
        exit(1);
    }
    const SafeTensor & t = it->second;

    ggml_type type = sf_dtype_to_ggml(t.dtype);

    // ggml uses ne[0]=innermost (column-major metadata)
    // safetensors shape is [dim0, dim1, ...] in row-major (PyTorch convention)
    // For a 2D weight [out, in]: ggml ne[0]=in, ne[1]=out
    int n_dims;
    int64_t ne[4] = {1, 1, 1, 1};

    if (shape_override && n_dims_override > 0) {
        n_dims = n_dims_override;
        for (int i = 0; i < n_dims; i++) ne[i] = shape_override[i];
    } else {
        n_dims = (int)t.shape.size();
        // Reverse: PyTorch [d0, d1, ...] -> ggml [d_last, ..., d0]
        for (int i = 0; i < n_dims && i < 4; i++) {
            ne[i] = t.shape[n_dims - 1 - i];
        }
    }

    struct ggml_tensor * tensor = ggml_new_tensor(wctx->ctx, type, n_dims, ne);
    ggml_set_name(tensor, name.c_str());

    wctx->pending.push_back({tensor, t.data, t.nbytes});

    return tensor;
}

// Try to load, returns nullptr if not found (no exit)
static struct ggml_tensor * sf_try_load_tensor(
        SFWeightCtx * wctx,
        const SafeTensors & st,
        const std::string & name) {
    auto it = st.tensors.find(name);
    if (it == st.tensors.end()) return nullptr;
    return sf_load_tensor(wctx, st, name);
}

// Allocate backend buffer and copy all pending tensor data.
// Call this ONCE after all sf_load_tensor calls.
static bool sf_weight_ctx_alloc(SFWeightCtx * wctx, ggml_backend_t backend) {
    wctx->buffer = ggml_backend_alloc_ctx_tensors(wctx->ctx, backend);
    if (!wctx->buffer) {
        fprintf(stderr, "[Safetensors] FATAL: failed to allocate backend buffer\n");
        return false;
    }

    size_t total = 0;
    for (auto & pc : wctx->pending) {
        ggml_backend_tensor_set(pc.tensor, pc.src, 0, pc.nbytes);
        total += pc.nbytes;
    }
    fprintf(stderr, "[Safetensors] Loaded %zu tensors, %.1f MB into backend\n",
            wctx->pending.size(), (float)total / (1024 * 1024));

    wctx->pending.clear();
    return true;
}

static void sf_weight_ctx_free(SFWeightCtx * wctx) {
    if (wctx->buffer) ggml_backend_buffer_free(wctx->buffer);
    if (wctx->ctx) ggml_free(wctx->ctx);
    wctx->buffer = NULL;
    wctx->ctx = NULL;
}
