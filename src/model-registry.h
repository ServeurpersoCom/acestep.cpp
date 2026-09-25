#pragma once
// model-registry.h: scan directories for GGUF models and adapters.
//
// Reads only GGUF headers (no weight data) to classify each file by its
// general.architecture KV into lm/dit/text-enc/vae buckets.
// Adapter entries are .safetensors files or PEFT directories.
//
// Usage:
//   ModelRegistry reg;
//   registry_scan(&reg, "./models");
//   registry_scan_adapters(&reg, "./adapters");
//   const ModelEntry * dit = registry_find(reg.dit, "acestep-v15-turbo-Q8_0.gguf");

#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
#else
#    include <dirent.h>
#    include <sys/stat.h>
#endif

struct ModelEntry {
    std::string name;  // filename (e.g. "acestep-v15-turbo-Q8_0.gguf")
    std::string path;  // full path
};

struct AdapterEntry {
    std::string name;  // filename or directory name (e.g. "singer.safetensors" or "my-adapter")
    std::string path;  // full path (file or PEFT directory)
};

struct ModelRegistry {
    std::vector<ModelEntry>   lm;
    std::vector<ModelEntry>   dit;
    std::vector<ModelEntry>   text_enc;
    std::vector<ModelEntry>   vae;
    std::vector<AdapterEntry> adapters;
};

// find an entry by name in a bucket. returns NULL if not found.
static const ModelEntry * registry_find(const std::vector<ModelEntry> & bucket, const char * name) {
    for (const auto & e : bucket) {
        if (e.name == name) {
            return &e;
        }
    }
    return nullptr;
}

// find an adapter entry by name. returns NULL if not found.
static const AdapterEntry * registry_find_adapter(const ModelRegistry & reg, const char * name) {
    for (const auto & e : reg.adapters) {
        if (e.name == name) {
            return &e;
        }
    }
    return nullptr;
}

// classify a GGUF file by reading its header.
// returns: "lm", "dit", "text-enc", "vae", or "" if unrecognized.
static std::string registry_classify_gguf(const char * path) {
    struct gguf_init_params params = { true, nullptr };
    struct gguf_context *   ctx    = gguf_init_from_file(path, params);
    if (!ctx) {
        return "";
    }

    std::string arch;
    int64_t     idx = gguf_find_key(ctx, "general.architecture");
    if (idx >= 0) {
        arch = gguf_get_val_str(ctx, idx);
    }
    gguf_free(ctx);

    // map GGUF architecture string to bucket name
    if (arch == "acestep-lm") {
        return "LM";
    }
    if (arch == "acestep-dit") {
        return "DiT";
    }
    if (arch == "acestep-text-enc") {
        return "Text-Enc";
    }
    if (arch == "acestep-vae") {
        return "VAE";
    }
    return "";
}

// check if a string ends with a suffix
static bool str_ends_with(const std::string & s, const char * suffix) {
    size_t slen = strlen(suffix);
    return s.size() >= slen && s.compare(s.size() - slen, slen, suffix) == 0;
}

#ifdef _WIN32

// scan a directory for files matching a pattern (Windows)
static void registry_list_dir(const char * dir, std::vector<std::string> * names) {
    std::string      pattern = std::string(dir) + "\\*";
    WIN32_FIND_DATAA fd;
    HANDLE           h = FindFirstFileA(pattern.c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) {
        return;
    }
    do {
        if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
            names->push_back(fd.cFileName);
        }
    } while (FindNextFileA(h, &fd));
    FindClose(h);
}

// list subdirectories (Windows)
static void registry_list_subdirs(const char * dir, std::vector<std::string> * names) {
    std::string      pattern = std::string(dir) + "\\*";
    WIN32_FIND_DATAA fd;
    HANDLE           h = FindFirstFileA(pattern.c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) {
        return;
    }
    do {
        if ((fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) && strcmp(fd.cFileName, ".") != 0 &&
            strcmp(fd.cFileName, "..") != 0) {
            names->push_back(fd.cFileName);
        }
    } while (FindNextFileA(h, &fd));
    FindClose(h);
}

static bool registry_is_file(const char * path) {
    DWORD attr = GetFileAttributesA(path);
    return attr != INVALID_FILE_ATTRIBUTES && !(attr & FILE_ATTRIBUTE_DIRECTORY);
}

#else

// scan a directory for files (POSIX)
static void registry_list_dir(const char * dir, std::vector<std::string> * names) {
    DIR * d = opendir(dir);
    if (!d) {
        return;
    }
    struct dirent * entry;
    while ((entry = readdir(d)) != nullptr) {
        // skip directories
        std::string full = std::string(dir) + "/" + entry->d_name;
        struct stat sb;
        if (stat(full.c_str(), &sb) == 0 && S_ISREG(sb.st_mode)) {
            names->push_back(entry->d_name);
        }
    }
    closedir(d);
}

// list subdirectories (POSIX)
static void registry_list_subdirs(const char * dir, std::vector<std::string> * names) {
    DIR * d = opendir(dir);
    if (!d) {
        return;
    }
    struct dirent * entry;
    while ((entry = readdir(d)) != nullptr) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        std::string full = std::string(dir) + "/" + entry->d_name;
        struct stat sb;
        if (stat(full.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode)) {
            names->push_back(entry->d_name);
        }
    }
    closedir(d);
}

static bool registry_is_file(const char * path) {
    struct stat sb;
    return stat(path, &sb) == 0 && S_ISREG(sb.st_mode);
}

#endif

// path separator
#ifdef _WIN32
#    define REGISTRY_SEP "\\"
#else
#    define REGISTRY_SEP "/"
#endif

// scan a directory for .gguf files, classify each by architecture.
// returns true if at least one model was found.
static bool registry_scan(ModelRegistry * reg, const char * models_dir) {
    std::vector<std::string> files;
    registry_list_dir(models_dir, &files);
    std::sort(files.begin(), files.end());

    int count = 0;
    for (const auto & fname : files) {
        if (!str_ends_with(fname, ".gguf")) {
            continue;
        }

        std::string full = std::string(models_dir) + REGISTRY_SEP + fname;
        std::string type = registry_classify_gguf(full.c_str());
        if (type.empty()) {
            fprintf(stderr, "[Registry] WARNING: skipping %s (unknown architecture)\n", fname.c_str());
            continue;
        }

        ModelEntry entry = { fname, full };
        if (type == "LM") {
            reg->lm.push_back(entry);
        } else if (type == "DiT") {
            reg->dit.push_back(entry);
        } else if (type == "Text-Enc") {
            reg->text_enc.push_back(entry);
        } else if (type == "VAE") {
            reg->vae.push_back(entry);
        }

        fprintf(stderr, "[Registry] %s -> %s\n", fname.c_str(), type.c_str());
        count++;
    }

    return count > 0;
}

// scan a directory for adapters.
// - .safetensors files: ComfyUI single-file format (alpha baked in)
// - subdirectories containing adapter_model.safetensors: PEFT format
// - subdirectories containing lokr_weights.safetensors: LoKr (ace-train)
// returns true if at least one adapter was found.
static bool registry_scan_adapters(ModelRegistry * reg, const char * adapters_dir) {
    int count = 0;

    // single .safetensors files
    std::vector<std::string> files;
    registry_list_dir(adapters_dir, &files);
    std::sort(files.begin(), files.end());
    for (const auto & fname : files) {
        if (!str_ends_with(fname, ".safetensors")) {
            continue;
        }
        std::string full = std::string(adapters_dir) + REGISTRY_SEP + fname;
        reg->adapters.push_back({ fname, full });
        fprintf(stderr, "[Registry] Adapter: %s (ComfyUI)\n", fname.c_str());
        count++;
    }

    // PEFT directories (contain adapter_model.safetensors)
    std::vector<std::string> subdirs;
    registry_list_subdirs(adapters_dir, &subdirs);
    std::sort(subdirs.begin(), subdirs.end());
    for (const auto & dname : subdirs) {
        std::string adapter =
            std::string(adapters_dir) + REGISTRY_SEP + dname + REGISTRY_SEP + "adapter_model.safetensors";
        std::string lokr = std::string(adapters_dir) + REGISTRY_SEP + dname + REGISTRY_SEP + "lokr_weights.safetensors";
        bool        is_peft = registry_is_file(adapter.c_str());
        bool        is_lokr = !is_peft && registry_is_file(lokr.c_str());
        std::string dir     = std::string(adapters_dir) + REGISTRY_SEP + dname;
        // a folder holding one weights file under its own name, as most
        // published adapters ship (name.safetensors + adapter_config.json)
        std::string single;
        if (!is_peft && !is_lokr) {
            std::vector<std::string> inner;
            registry_list_dir(dir.c_str(), &inner);
            int n = 0;
            for (const auto & f : inner) {
                if (str_ends_with(f, ".safetensors")) {
                    single = f;
                    n++;
                }
            }
            if (n != 1) {
                single.clear();
            }
        }
        if (is_peft || is_lokr || !single.empty()) {
            reg->adapters.push_back({ dname, dir });
            fprintf(stderr, "[Registry] Adapter: %s (%s)\n", dname.c_str(),
                    is_peft ? "PEFT" : (is_lokr ? "LoKr" : single.c_str()));
            count++;
        }
    }

    return count > 0;
}
