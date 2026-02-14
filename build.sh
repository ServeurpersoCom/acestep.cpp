#!/bin/bash
set -e

BUILD_DIR="build"
CMAKE_ARGS=""

# Auto-detect backends
# Check standard CUDA paths if nvcc not in PATH
NVCC_PATH=""
if command -v nvcc &>/dev/null; then
    NVCC_PATH=$(command -v nvcc)
elif [ -x /usr/local/cuda/bin/nvcc ]; then
    NVCC_PATH=/usr/local/cuda/bin/nvcc
elif [ -x /opt/cuda/bin/nvcc ]; then
    NVCC_PATH=/opt/cuda/bin/nvcc
fi

if [ "$(uname)" = "Darwin" ]; then
    echo "[Build] macOS detected: enabling Metal + Accelerate BLAS"
    CMAKE_ARGS="-DGGML_METAL=ON -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Apple"
elif [ -n "$NVCC_PATH" ]; then
    echo "[Build] CUDA detected: $NVCC_PATH"
    CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=$NVCC_PATH"
    # Also enable BLAS if available (OpenBLAS, MKL, etc.)
    if pkg-config --exists openblas 2>/dev/null || [ -f /usr/include/cblas.h ]; then
        echo "[Build] BLAS detected: enabling"
        CMAKE_ARGS="$CMAKE_ARGS -DGGML_BLAS=ON"
    fi
elif command -v vulkaninfo &>/dev/null; then
    echo "[Build] Vulkan detected: enabling"
    CMAKE_ARGS="-DGGML_VULKAN=ON"
    if pkg-config --exists openblas 2>/dev/null || [ -f /usr/include/cblas.h ]; then
        echo "[Build] BLAS detected: enabling"
        CMAKE_ARGS="$CMAKE_ARGS -DGGML_BLAS=ON"
    fi
else
    echo "[Build] CPU only"
    if pkg-config --exists openblas 2>/dev/null || [ -f /usr/include/cblas.h ]; then
        echo "[Build] BLAS detected: enabling"
        CMAKE_ARGS="-DGGML_BLAS=ON"
    fi
fi

# Allow override: ./build.sh -DGGML_VULKAN=ON -DGGML_BLAS=OFF ...
CMAKE_ARGS="$CMAKE_ARGS $@"

echo "[Build] cmake args: $CMAKE_ARGS"

rm -rf "$BUILD_DIR"
mkdir "$BUILD_DIR"
cd "$BUILD_DIR"
cmake .. $CMAKE_ARGS
cmake --build . --config Release -j "$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
