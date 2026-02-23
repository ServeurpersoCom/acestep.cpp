// test-philox.cpp
// Dump Philox noise to binary file for comparison with PyTorch CUDA.
// Build: make (see Makefile)
// Usage: ./test-philox [seed] [count] [output.f32]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "philox.h"

int main(int argc, char ** argv) {
    if (argc > 1 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)) {
        fprintf(stderr, "usage: %s [seed=42] [count=48000] [output=philox-noise.f32]\n", argv[0]);
        return 0;
    }
    int64_t seed = argc > 1 ? atoll(argv[1]) : 42;
    int count    = argc > 2 ? atoi(argv[2])  : 48000;
    const char * path = argc > 3 ? argv[3] : "philox-noise.f32";

    float * out = new float[count];
    philox_randn(seed, out, count, true);

    FILE * f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 1; }
    fwrite(out, sizeof(float), count, f);
    fclose(f);
    delete[] out;

    fprintf(stderr, "wrote %d floats (seed=%lld) to %s\n", count, (long long)seed, path);
    return 0;
}
