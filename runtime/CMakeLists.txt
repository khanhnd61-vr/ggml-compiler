cmake_minimum_required(VERSION 3.22)
project(ggmlc_rt VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ggml comes from a pinned llama.cpp release tarball. The source tree is shared by every
# build directory (CPU, CUDA, ...) so it is downloaded once; each build keeps its own
# binary dir. Override GGMLC_DEPS_DIR to keep it elsewhere.
set(GGMLC_LLAMA_TAG "b11223" CACHE STRING "llama.cpp tag providing ggml")
set(GGMLC_DEPS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.deps" CACHE PATH "Where fetched sources live")

# ggml is linked statically into libggmlc_rt so the Python bindings load a single library.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
# llama.cpp's top-level CMake turns tinyBLAS (llamafile sgemm) on; ggml built on its own leaves it
# off, and its CPU f32/f16/bf16 matmuls then fall back to plain dot products (bf16: 156 vs 275
# GFLOP/s on an i5-12400F). vla.cpp builds with it on.
set(GGML_LLAMAFILE_DEFAULT ON)
# Likewise CUDA graphs: llama.cpp turns them on, standalone ggml leaves them off, and every graph
# then launches its kernels one by one (Qwen3-0.6B decode: 3.3 ms instead of ~2.7 ms a token).
set(GGML_CUDA_GRAPHS_DEFAULT ON)

include(FetchContent)
FetchContent_Declare(ggml
    URL https://github.com/ggml-org/llama.cpp/archive/refs/tags/${GGMLC_LLAMA_TAG}.tar.gz
    SOURCE_DIR ${GGMLC_DEPS_DIR}/llama.cpp-${GGMLC_LLAMA_TAG}
    SOURCE_SUBDIR ggml
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(ggml)

add_library(ggmlc_rt SHARED
    src/ggmlc_rt.cpp
    src/program.cpp
    src/ops.cpp
)
target_include_directories(ggmlc_rt
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${ggml_SOURCE_DIR}/ggml/include  # generated builders call ggml through libggmlc_rt
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${ggml_SOURCE_DIR}/vendor   # nlohmann/json.hpp ships in the llama.cpp tarball
)
target_link_libraries(ggmlc_rt PRIVATE ggml)
target_compile_definitions(ggmlc_rt PRIVATE GGMLC_RT_BUILD GGMLC_LLAMA_TAG="${GGMLC_LLAMA_TAG}")
if(GGML_CUDA)
    target_compile_definitions(ggmlc_rt PRIVATE GGMLC_HAVE_CUDA)
endif()
set_target_properties(ggmlc_rt PROPERTIES C_VISIBILITY_PRESET hidden CXX_VISIBILITY_PRESET hidden)

# Plain C on purpose: keeps ggmlc_rt.h honest as a C header.
add_executable(ggmlc-info tools/ggmlc_info.c)
target_link_libraries(ggmlc-info PRIVATE ggmlc_rt)
