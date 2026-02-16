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

# ggml as subdirectory, inherits GGML_CUDA, GGML_METAL, etc. from cmake flags
add_subdirectory(ggml)

# Macro: link ggml backends + strict warnings (scoped to our targets, not ggml)
macro(link_ggml_backends target)
    target_include_directories(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
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
        endif()
    endforeach()
endmacro()

# dit-vae: full pipeline (text-enc + cond + dit + vae + wav)
add_executable(dit-vae dit-vae.cpp request.cpp)
link_ggml_backends(dit-vae)

# ace-qwen3: LLM inference (CoT + audio codes)
add_executable(ace-qwen3 ace-qwen3.cpp request.cpp)
link_ggml_backends(ace-qwen3)
