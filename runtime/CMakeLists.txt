cmake_minimum_required(VERSION 3.22)
# The version lives in ggml_compiler/__init__.py; the runtime reports the same one.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/../ggml_compiler/__init__.py" _version_line REGEX "^__version__ = ")
string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" GGML_COMPILER_VERSION "${_version_line}")
project(ggml_compiler_rt VERSION ${GGML_COMPILER_VERSION} LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ggml comes from a pinned llama.cpp release tarball. The source tree is shared by every
# build directory (CPU, CUDA, ...) so it is downloaded once; each build keeps its own
# binary dir. Override GGML_COMPILER_DEPS_DIR to keep it elsewhere.
set(GGML_COMPILER_LLAMA_TAG "b11223" CACHE STRING "llama.cpp tag providing ggml")
set(GGML_COMPILER_DEPS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.deps" CACHE PATH "Where fetched sources live")

# ggml is linked statically into libggml_compiler_rt so the Python bindings load a single library.
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
    URL https://github.com/ggml-org/llama.cpp/archive/refs/tags/${GGML_COMPILER_LLAMA_TAG}.tar.gz
    SOURCE_DIR ${GGML_COMPILER_DEPS_DIR}/llama.cpp-${GGML_COMPILER_LLAMA_TAG}
    SOURCE_SUBDIR ggml
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(ggml)

add_library(ggml_compiler_rt SHARED
    src/ggml_compiler_rt.cpp
    src/program.cpp
    src/ops.cpp
)
target_include_directories(ggml_compiler_rt
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${ggml_SOURCE_DIR}/ggml/include  # generated builders call ggml through libggml_compiler_rt
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${ggml_SOURCE_DIR}/vendor   # nlohmann/json.hpp ships in the llama.cpp tarball
)
target_link_libraries(ggml_compiler_rt PRIVATE ggml)
target_compile_definitions(ggml_compiler_rt PRIVATE GGML_COMPILER_RT_BUILD
                           GGML_COMPILER_LLAMA_TAG="${GGML_COMPILER_LLAMA_TAG}"
                           GGML_COMPILER_VERSION="${PROJECT_VERSION}")
if(GGML_CUDA)
    target_compile_definitions(ggml_compiler_rt PRIVATE GGML_COMPILER_HAVE_CUDA)
endif()
set_target_properties(ggml_compiler_rt PROPERTIES C_VISIBILITY_PRESET hidden CXX_VISIBILITY_PRESET hidden)

# Plain C on purpose: keeps ggml_compiler_rt.h honest as a C header.
add_executable(ggml-compiler-info tools/ggml_compiler_info.c)
target_link_libraries(ggml-compiler-info PRIVATE ggml_compiler_rt)

# A wheel (pip / scikit-build-core) carries the library and the headers generated C++ needs: the
# runtime's and ggml's. Only this component is installed; ggml's own install rules are not.
if(SKBUILD)
    install(TARGETS ggml_compiler_rt
            LIBRARY DESTINATION ggml_compiler/lib COMPONENT wheel
            RUNTIME DESTINATION ggml_compiler/lib COMPONENT wheel)
    install(DIRECTORY include/ ${ggml_SOURCE_DIR}/ggml/include/ DESTINATION ggml_compiler/include COMPONENT wheel)
endif()
