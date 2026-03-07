cmake_minimum_required(VERSION 3.14)
project(acestep-ggml LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Suppress MSVC fopen/sprintf deprecation warnings
if(MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

# DiT tensor names can exceed default GGML_MAX_NAME of 64
add_compile_definitions(GGML_MAX_NAME=128)

# CUDA architectures: cover Turing to Blackwell for distributed binaries.
# Users can override with -DCMAKE_CUDA_ARCHITECTURES=native for local builds.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
        set(CMAKE_CUDA_ARCHITECTURES "75-virtual;80-virtual;86-real;89-real;120a-real;121a-real")
    else()
        set(CMAKE_CUDA_ARCHITECTURES "75-virtual;80-virtual;86-real;89-real")
    endif()
endif()

# ggml as subdirectory, inherits GGML_CUDA, GGML_METAL, etc. from cmake flags
add_subdirectory(ggml)

# Shared compile options and ggml linkage
macro(link_ggml_backends target)
    target_include_directories(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_include_directories(${target} SYSTEM PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
    )
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /wd4100 /wd4505)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wshadow -Wconversion
                              -Wno-unused-parameter -Wno-unused-function -Wno-sign-conversion)
    endif()
    target_link_libraries(${target} PRIVATE ggml ggml-base ggml-cpu)
    foreach(backend blas cuda metal vulkan)
        if(TARGET ggml-${backend})
            target_link_libraries(${target} PRIVATE ggml-${backend})
            string(TOUPPER ${backend} BACKEND_UPPER)
            target_compile_definitions(${target} PRIVATE ACESTEP_HAVE_${BACKEND_UPPER})
            if(backend STREQUAL "cuda")
                find_package(CUDAToolkit QUIET)
                if(CUDAToolkit_FOUND)
                    target_link_libraries(${target} PRIVATE CUDA::cudart)
                endif()
            endif()
        endif()
    endforeach()
endmacro()

# Core library (shared between binaries)
add_library(acestep-core STATIC
    src/request.cpp
)
link_ggml_backends(acestep-core)

# dit-vae: full pipeline (text-enc + cond + dit + vae + wav) + LoRA support
add_executable(dit-vae tools/dit-vae.cpp src/dit-lora.cpp)
target_link_libraries(dit-vae PRIVATE acestep-core)
link_ggml_backends(dit-vae)

# ace-qwen3: LLM inference (CoT + audio codes)
add_executable(ace-qwen3 tools/ace-qwen3.cpp)
target_link_libraries(ace-qwen3 PRIVATE acestep-core)
link_ggml_backends(ace-qwen3)

# quantize: GGUF requantizer (BF16 -> K-quants)
add_executable(quantize tools/quantize.cpp)
link_ggml_backends(quantize)

# neural-codec: Oobleck VAE neural audio codec (encode/decode WAV <-> latent)
add_executable(neural-codec tools/neural-codec.cpp)
link_ggml_backends(neural-codec)
