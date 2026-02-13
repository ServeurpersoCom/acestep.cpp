// compare.cpp: compare tensor dumps between CUDA and ggml
// Usage: ./compare <cuda_dump_dir> <ggml_dump_dir>
// Reads all .bin files in both directories, matches by name, computes cosine sim.

#include "debug.h"
#include <dirent.h>
#include <algorithm>

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <dir_a> <dir_b>\n", argv[0]);
        fprintf(stderr, "Compares tensor dumps from two backends.\n");
        fprintf(stderr, "Example: %s /tmp/cuda_dump /tmp/ggml_dump\n", argv[0]);
        return 1;
    }

    const char *dir_a = argv[1];
    const char *dir_b = argv[2];

    // Collect .bin files from dir_a
    std::vector<std::string> names;
    DIR *d = opendir(dir_a);
    if (!d) { fprintf(stderr, "Cannot open %s\n", dir_a); return 1; }
    struct dirent *ent;
    while ((ent = readdir(d))) {
        std::string name = ent->d_name;
        if (name.size() > 4 && name.substr(name.size() - 4) == ".bin") {
            names.push_back(name.substr(0, name.size() - 4));
        }
    }
    closedir(d);
    std::sort(names.begin(), names.end());

    if (names.empty()) {
        fprintf(stderr, "No .bin files found in %s\n", dir_a);
        return 1;
    }

    fprintf(stderr, "Comparing %zu tensors: %s vs %s\n\n", names.size(), dir_a, dir_b);
    fprintf(stderr, "%-30s %10s %10s %12s %12s\n",
            "TENSOR", "SHAPE_A", "SHAPE_B", "COSINE", "MAX_ERR");
    fprintf(stderr, "%-30s %10s %10s %12s %12s\n",
            "......", ".......", ".......", "......", ".......");

    int n_pass = 0, n_fail = 0, n_skip = 0;

    for (auto &name : names) {
        char path_a[1024], path_b[1024];
        snprintf(path_a, sizeof(path_a), "%s/%s.bin", dir_a, name.c_str());
        snprintf(path_b, sizeof(path_b), "%s/%s.bin", dir_b, name.c_str());

        std::vector<int> shape_a, shape_b;
        auto data_a = debug_load(path_a, shape_a);
        auto data_b = debug_load(path_b, shape_b);

        if (data_b.empty()) {
            fprintf(stderr, "%-30s %10s %10s %12s %12s  SKIP (not in B)\n",
                    name.c_str(), "", "", "", "");
            n_skip++;
            continue;
        }

        // Format shapes
        char sa[64] = "", sb[64] = "";
        for (int i = 0; i < (int)shape_a.size(); i++)
            snprintf(sa + strlen(sa), sizeof(sa) - strlen(sa), "%s%d",
                     i ? "x" : "", shape_a[i]);
        for (int i = 0; i < (int)shape_b.size(); i++)
            snprintf(sb + strlen(sb), sizeof(sb) - strlen(sb), "%s%d",
                     i ? "x" : "", shape_b[i]);

        if (data_a.size() != data_b.size()) {
            fprintf(stderr, "%-30s %10s %10s %12s %12s  FAIL (size mismatch)\n",
                    name.c_str(), sa, sb, "", "");
            n_fail++;
            continue;
        }

        int n = (int)data_a.size();
        double cos = debug_cosine_sim(data_a.data(), data_b.data(), n);
        double maxe = debug_max_abs_err(data_a.data(), data_b.data(), n);
        double meane = debug_mean_abs_err(data_a.data(), data_b.data(), n);

        const char *status = (cos > 0.999) ? "PASS" : (cos > 0.99) ? "WARN" : "FAIL";
        if (cos > 0.999) n_pass++; else n_fail++;

        fprintf(stderr, "%-30s %10s %10s %12.6f %12.6f  %s (mean=%.6f)\n",
                name.c_str(), sa, sb, cos, maxe, status, meane);
    }

    fprintf(stderr, "\n--- Summary: %d PASS, %d FAIL, %d SKIP ---\n",
            n_pass, n_fail, n_skip);

    return n_fail > 0 ? 1 : 0;
}
