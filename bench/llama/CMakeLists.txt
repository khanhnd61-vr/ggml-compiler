# llama-logits: dumps llama.cpp's logits for a token sequence, for ggml-compiler's KL / top-1 comparison.
# Links an existing llama.cpp build (it is never rebuilt here):
#   cmake -S bench/llama -B DIR -DLLAMA_SRC=.deps/llama.cpp-b11223 -DLLAMA_BUILD=/path/to/llama.cpp/build
cmake_minimum_required(VERSION 3.22)
project(ggml_compiler_llama_bench CXX)
set(CMAKE_CXX_STANDARD 17)
set(LLAMA_SRC "" CACHE PATH "llama.cpp source tree")
set(LLAMA_BUILD "" CACHE PATH "llama.cpp build directory (holds bin/libllama.so)")
if(NOT LLAMA_SRC OR NOT LLAMA_BUILD)
  message(FATAL_ERROR "set -DLLAMA_SRC and -DLLAMA_BUILD")
endif()
add_executable(llama-logits llama_logits.cpp)
target_include_directories(llama-logits PRIVATE ${LLAMA_SRC}/include ${LLAMA_SRC}/ggml/include)
target_link_libraries(llama-logits PRIVATE ${LLAMA_BUILD}/bin/libllama.so ${LLAMA_BUILD}/bin/libggml.so
                      ${LLAMA_BUILD}/bin/libggml-base.so)
set_target_properties(llama-logits PROPERTIES BUILD_RPATH ${LLAMA_BUILD}/bin)

# mtmd-logits: the same for an image prompt, from transformers' preprocessed pixels. It compiles
# llama.cpp's clip sources (same tree) into the tool: the f32-pixel entry points it needs are
# internal to libmtmd.
file(GLOB MTMD_MODELS ${LLAMA_SRC}/tools/mtmd/models/*.cpp)
add_executable(mtmd-logits mtmd_logits.cpp ${LLAMA_SRC}/tools/mtmd/clip.cpp ${MTMD_MODELS})
target_include_directories(mtmd-logits PRIVATE ${LLAMA_SRC}/tools/mtmd ${LLAMA_SRC}/include ${LLAMA_SRC}/ggml/include)
target_link_libraries(mtmd-logits PRIVATE ${LLAMA_BUILD}/bin/libllama.so ${LLAMA_BUILD}/bin/libggml.so
                      ${LLAMA_BUILD}/bin/libggml-base.so)
set_target_properties(mtmd-logits PROPERTIES BUILD_RPATH ${LLAMA_BUILD}/bin)
