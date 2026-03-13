#!/bin/bash

rm -rf build
mkdir build
cd build

cmake .. -DGGML_BLAS=ON -DBLAS_INCLUDE_DIRS=$PREFIX/include/openblas
cmake --build . --config Release -j "$(nproc)"
