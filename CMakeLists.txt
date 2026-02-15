cmake_minimum_required(VERSION 3.14)
project(acestep-ggml LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ggml as subdirectory, inherits GGML_CUDA, GGML_METAL, etc. from cmake flags
add_subdirectory(ggml)

# Macro: link ggml backends to a target (avoids repeating per-executable)
macro(link_ggml_backends target)
    target_include_directories(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
    )
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

# pt2bin: convert silence_latent.pt -> .bin (raw float32)
add_executable(pt2bin pt2bin.cpp)
