#pragma once
// Minimal safetensors loader via mmap. Zero-copy, zero-dep.
// Supports multi-file models (model-00001-of-00002.safetensors etc.)
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>

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
        // Find next tensor name (skip __metadata__)
        const char *q = strchr(p, '"');
        if (!q || q >= header_end) break;
        q++;
        const char *name_end = strchr(q, '"');
        if (!name_end || name_end >= header_end) break;
        std::string name(q, name_end);
        p = name_end + 1;

        // Skip to the value object
        const char *brace = strchr(p, '{');
        if (!brace || brace >= header_end) break;
        const char *brace_end = strchr(brace, '}');
        if (!brace_end || brace_end >= header_end) break;

        // Extract a null-terminated copy of this entry for parsing
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
