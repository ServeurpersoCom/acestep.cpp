cmake_minimum_required(VERSION 3.14)
project(acestep-ggml LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ggml as subdirectory, inherits GGML_CUDA, GGML_METAL, etc. from cmake flags
add_subdirectory(ggml)

# dit-vae: full pipeline (text-enc + cond + dit + vae + wav)
add_executable(dit-vae dit-vae.cpp)

target_include_directories(dit-vae PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
)

target_link_libraries(dit-vae PRIVATE ggml ggml-base ggml-cpu)

if(TARGET ggml-blas)
    target_link_libraries(dit-vae PRIVATE ggml-blas)
endif()

if(TARGET ggml-cuda)
    target_link_libraries(dit-vae PRIVATE ggml-cuda)
    target_compile_definitions(dit-vae PRIVATE ACESTEP_HAVE_CUDA)
endif()

if(TARGET ggml-metal)
    target_link_libraries(dit-vae PRIVATE ggml-metal)
    target_compile_definitions(dit-vae PRIVATE ACESTEP_HAVE_METAL)
endif()

if(TARGET ggml-vulkan)
    target_link_libraries(dit-vae PRIVATE ggml-vulkan)
    target_compile_definitions(dit-vae PRIVATE ACESTEP_HAVE_VULKAN)
endif()

# ace-qwen3: LLM inference (CoT + audio codes)
add_executable(ace-qwen3 ace-qwen3.cpp)

target_include_directories(ace-qwen3 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
)

target_link_libraries(ace-qwen3 PRIVATE ggml ggml-base ggml-cpu)

if(TARGET ggml-blas)
    target_link_libraries(ace-qwen3 PRIVATE ggml-blas)
endif()

if(TARGET ggml-cuda)
    target_link_libraries(ace-qwen3 PRIVATE ggml-cuda)
    target_compile_definitions(ace-qwen3 PRIVATE ACESTEP_HAVE_CUDA)
endif()

if(TARGET ggml-metal)
    target_link_libraries(ace-qwen3 PRIVATE ggml-metal)
    target_compile_definitions(ace-qwen3 PRIVATE ACESTEP_HAVE_METAL)
endif()

if(TARGET ggml-vulkan)
    target_link_libraries(ace-qwen3 PRIVATE ggml-vulkan)
    target_compile_definitions(ace-qwen3 PRIVATE ACESTEP_HAVE_VULKAN)
endif()

# compare: cosine similarity between tensor dumps (debug tool)
add_executable(compare compare.cpp)
target_include_directories(compare PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
