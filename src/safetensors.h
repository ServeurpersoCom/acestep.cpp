#pragma once
// safetensors.h: minimal reader for LoRA adapter_model.safetensors
//
// Format: 8-byte header length (LE uint64), then JSON header, then raw tensor data.
// We only parse keys that look like "*lora_A*" / "*lora_B*" and extract shape + data_offsets.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>

struct SafeTensorInfo {
    std::string dtype;       // "F32", "F16", "BF16"
    int64_t shape[2];        // [dim0, dim1] from JSON
    int n_dims;
    uint64_t data_start;     // byte offset in file (after header)
    uint64_t data_end;
};

// Open file, read header, parse tensor metadata for LoRA tensors.
// Returns number of LoRA tensors found; fills *out with tensor name -> info.
// Caller must fclose(fp) and free the map; file position is left at start of data section.
static int safetensors_parse_lora(FILE * fp, std::unordered_map<std::string, SafeTensorInfo> * out) {
    out->clear();
    uint64_t header_len = 0;
    uint8_t h8[8];
    if (fread(h8, 1, 8, fp) != 8) return 0;
    header_len = (uint64_t)h8[0] | ((uint64_t)h8[1] << 8) | ((uint64_t)h8[2] << 16) | ((uint64_t)h8[3] << 24)
        | ((uint64_t)h8[4] << 32) | ((uint64_t)h8[5] << 40) | ((uint64_t)h8[6] << 48) | ((uint64_t)h8[7] << 56);
    if (header_len == 0 || header_len > 10 * 1024 * 1024) return 0;  // cap 10MB header
    std::vector<char> buf(header_len + 1);
    if (fread(buf.data(), 1, header_len, fp) != header_len) return 0;
    buf[header_len] = '\0';
    const char * json = buf.data();

    // Find each key that contains "lora_A" or "lora_B"
    const char * p = json;
    int count = 0;
    while ((p = strstr(p, "\"")) != nullptr) {
        const char * key_start = p + 1;
        p = strchr(key_start, '"');
        if (!p) break;
        std::string key(key_start, (size_t)(p - key_start));
        p++;
        if (key.find("lora_A") == std::string::npos && key.find("lora_B") == std::string::npos) {
            continue;
        }
        // Find the value object for this key: skip ":
        while (*p && (*p == ' ' || *p == ':')) p++;
        if (*p != '{') continue;
        const char * obj = p;
        SafeTensorInfo info = {};
        info.shape[0] = info.shape[1] = 1;
        info.n_dims = 0;
        // "shape":[n,m] or [n]
        const char * sh = strstr(obj, "\"shape\"");
        if (sh) {
            const char * br = strchr(sh, '[');
            if (br) {
                long long a = 0, b = 0;
                int n = sscanf(br, "[%lld,%lld]", &a, &b);
                if (n >= 1) { info.shape[0] = (int64_t)a; info.n_dims = 1; }
                if (n >= 2) { info.shape[1] = (int64_t)b; info.n_dims = 2; }
            }
        }
        const char * dt = strstr(obj, "\"dtype\"");
        if (dt) {
            const char * q = strchr(dt, '"');
            if (q) q = strchr(q + 1, '"');
            if (q) {
                const char * start = q + 1;
                const char * end = strchr(start, '"');
                if (end) info.dtype = std::string(start, end - start);
            }
        }
        const char * off = strstr(obj, "\"data_offsets\"");
        if (off) {
            const char * br = strchr(off, '[');
            if (br) {
                uint64_t s = 0, e = 0;
                if (sscanf(br, "[%llu,%llu]", (unsigned long long*)&s, (unsigned long long*)&e) == 2) {
                    info.data_start = s;
                    info.data_end = e;
                }
            }
        }
        if (info.dtype.empty() || info.n_dims == 0) continue;
        (*out)[key] = info;
        count++;
    }
    return count;
}

// Read raw tensor data from file. File must be positioned at start of data section
// (i.e. after the 8-byte header length + header bytes).
// data_offset in the JSON is relative to the start of the data section.
static bool safetensors_read_tensor_data(FILE * fp, uint64_t data_section_start,
        uint64_t tensor_start, uint64_t tensor_end, void * out_buf) {
    uint64_t off = data_section_start + tensor_start;
    uint64_t nbytes = tensor_end - tensor_start;
    if (fseek(fp, (long)off, SEEK_SET) != 0) return false;
    if (fread(out_buf, 1, nbytes, fp) != nbytes) return false;
    return true;
}
