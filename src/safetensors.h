#pragma once
// safetensors.h: minimal read only safetensors parser
//
// Format: 8 byte LE header length, JSON header, raw tensor data.
// We mmap the file, parse the header, and expose tensor entries
// with name, dtype, shape, and a pointer to the raw data.
//
// Only handles the flat safetensors JSON structure.
// Not a general purpose JSON parser.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#    include <windows.h>
#else
#    include <fcntl.h>
#    include <sys/mman.h>
#    include <sys/stat.h>
#    include <unistd.h>
#endif

struct STEntry {
    std::string name;
    std::string dtype;  // "F32", "BF16", "F16"
    int64_t     shape[4];
    int         n_dims;
    size_t      data_start;  // byte offset from data section
    size_t      data_end;
};

struct STFile {
    uint8_t *            mapping;
    size_t               file_size;
    size_t               data_offset;  // 8 + header_len (start of tensor data)
    std::vector<STEntry> entries;
#ifdef _WIN32
    HANDLE fh, mh;
#else
    int fd;
#endif
};

// JSON helpers (just enough for safetensors headers)

static size_t st_ws(const char * s, size_t p, size_t len) {
    while (p < len && (s[p] == ' ' || s[p] == '\t' || s[p] == '\n' || s[p] == '\r')) {
        p++;
    }
    return p;
}

// Read a quoted string starting at s[p] == '"'. Returns position after closing quote.
static size_t st_str(const char * s, size_t p, size_t len, std::string & out) {
    p++;  // skip opening "
    size_t start = p;
    while (p < len && s[p] != '"') {
        if (s[p] == '\\') {
            p++;  // skip escaped char
        }
        p++;
    }
    out.assign(s + start, p - start);
    return p + 1;  // skip closing "
}

// Skip any JSON value (string, object, array, number, bool, null).
// Handles nested structures by counting depth.
static size_t st_skip(const char * s, size_t p, size_t len) {
    p = st_ws(s, p, len);
    if (p >= len) {
        return p;
    }
    if (s[p] == '"') {
        p++;
        while (p < len && s[p] != '"') {
            if (s[p] == '\\') {
                p++;
            }
            p++;
        }
        return p + 1;
    }
    if (s[p] == '{' || s[p] == '[') {
        char open = s[p], close = (open == '{') ? '}' : ']';
        int  depth = 1;
        p++;
        while (p < len && depth > 0) {
            if (s[p] == '"') {
                p++;
                while (p < len && s[p] != '"') {
                    if (s[p] == '\\') {
                        p++;
                    }
                    p++;
                }
            } else if (s[p] == open) {
                depth++;
            } else if (s[p] == close) {
                depth--;
            }
            p++;
        }
        return p;
    }
    // number, bool, null: scan until delimiter
    while (p < len && s[p] != ',' && s[p] != '}' && s[p] != ']') {
        p++;
    }
    return p;
}

// Parse the safetensors JSON header into st->entries
static bool st_parse(STFile * st, const char * hdr, size_t len) {
    size_t p = st_ws(hdr, 0, len);
    if (p >= len || hdr[p] != '{') {
        return false;
    }
    p++;

    while (p < len) {
        p = st_ws(hdr, p, len);
        if (p >= len || hdr[p] == '}') {
            break;
        }
        if (hdr[p] == ',') {
            p++;
            continue;
        }

        // top level key
        std::string key;
        p = st_str(hdr, p, len, key);
        p = st_ws(hdr, p, len);
        p++;  // skip ':'
        p = st_ws(hdr, p, len);

        // __metadata__ is not a tensor, skip it
        if (key == "__metadata__") {
            p = st_skip(hdr, p, len);
            continue;
        }

        // parse tensor entry: {"dtype":"...","shape":[...],"data_offsets":[s,e]}
        if (hdr[p] != '{') {
            return false;
        }
        p++;

        STEntry e = {};
        e.name    = key;
        while (p < len) {
            p = st_ws(hdr, p, len);
            if (p >= len || hdr[p] == '}') {
                p++;
                break;
            }
            if (hdr[p] == ',') {
                p++;
                continue;
            }

            std::string field;
            p = st_str(hdr, p, len, field);
            p = st_ws(hdr, p, len);
            p++;  // skip ':'
            p = st_ws(hdr, p, len);

            if (field == "dtype") {
                p = st_str(hdr, p, len, e.dtype);
            } else if (field == "shape") {
                // array of ints
                p++;  // skip '['
                e.n_dims = 0;
                while (p < len && hdr[p] != ']') {
                    p = st_ws(hdr, p, len);
                    if (hdr[p] == ',') {
                        p++;
                        continue;
                    }
                    if (hdr[p] == ']') {
                        break;
                    }
                    char * end;
                    e.shape[e.n_dims++] = strtoll(hdr + p, &end, 10);
                    p                   = (size_t) (end - hdr);
                    if (e.n_dims >= 4) {
                        break;
                    }
                }
                if (p < len && hdr[p] == ']') {
                    p++;
                }
            } else if (field == "data_offsets") {
                // [start, end]
                p++;  // skip '['
                p = st_ws(hdr, p, len);
                char * end;
                e.data_start = (size_t) strtoull(hdr + p, &end, 10);
                p            = (size_t) (end - hdr);
                p            = st_ws(hdr, p, len);
                if (hdr[p] == ',') {
                    p++;
                }
                p          = st_ws(hdr, p, len);
                e.data_end = (size_t) strtoull(hdr + p, &end, 10);
                p          = (size_t) (end - hdr);
                p          = st_ws(hdr, p, len);
                if (p < len && hdr[p] == ']') {
                    p++;
                }
            } else {
                p = st_skip(hdr, p, len);
            }
        }

        st->entries.push_back(e);
    }
    return true;
}

static void st_close(STFile * st);

static bool st_open(STFile * st, const char * path) {
    *st = {};

#ifdef _WIN32
    st->fh = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (st->fh == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "[Safetensors] cannot open %s\n", path);
        return false;
    }
    LARGE_INTEGER li;
    GetFileSizeEx(st->fh, &li);
    st->file_size = (size_t) li.QuadPart;
    st->mh        = CreateFileMappingA(st->fh, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!st->mh) {
        CloseHandle(st->fh);
        return false;
    }
    st->mapping = (uint8_t *) MapViewOfFile(st->mh, FILE_MAP_READ, 0, 0, 0);
    if (!st->mapping) {
        CloseHandle(st->mh);
        CloseHandle(st->fh);
        return false;
    }
#else
    st->fd = open(path, O_RDONLY);
    if (st->fd < 0) {
        fprintf(stderr, "[Safetensors] cannot open %s\n", path);
        return false;
    }
    struct stat sb;
    fstat(st->fd, &sb);
    st->file_size = (size_t) sb.st_size;
    st->mapping   = (uint8_t *) mmap(NULL, st->file_size, PROT_READ, MAP_PRIVATE, st->fd, 0);
    if (st->mapping == MAP_FAILED) {
        close(st->fd);
        st->mapping = NULL;
        fprintf(stderr, "[Safetensors] mmap failed %s\n", path);
        return false;
    }
#endif

    // first 8 bytes: LE u64 header length
    if (st->file_size < 8) {
        fprintf(stderr, "[Safetensors] file too small %s\n", path);
        st_close(st);
        return false;
    }
    uint64_t hdr_len;
    memcpy(&hdr_len, st->mapping, 8);
    st->data_offset = 8 + (size_t) hdr_len;

    if (st->data_offset > st->file_size) {
        fprintf(stderr, "[Safetensors] header overflows file %s\n", path);
        st_close(st);
        return false;
    }

    // parse JSON header
    if (!st_parse(st, (const char *) st->mapping + 8, (size_t) hdr_len)) {
        fprintf(stderr, "[Safetensors] failed to parse header %s\n", path);
        st_close(st);
        return false;
    }

    fprintf(stderr, "[Safetensors] %s: %zu tensors\n", path, st->entries.size());
    return true;
}

static void st_close(STFile * st) {
#ifdef _WIN32
    if (st->mapping) {
        UnmapViewOfFile(st->mapping);
    }
    if (st->mh) {
        CloseHandle(st->mh);
    }
    if (st->fh && st->fh != INVALID_HANDLE_VALUE) {
        CloseHandle(st->fh);
    }
#else
    if (st->mapping) {
        munmap(st->mapping, st->file_size);
    }
    if (st->fd >= 0) {
        close(st->fd);
    }
#endif
    *st = {};
}

// Get raw data pointer for a tensor entry
static inline const void * st_data(const STFile & st, const STEntry & e) {
    return st.mapping + st.data_offset + e.data_start;
}
