/* ggml-compiler-info: load a ggml-compiler GGUF and print its inputs and outputs. */
#include <stdio.h>
#include <string.h>

#include "ggml_compiler_rt.h"

static const char * dtype_name(ggml_compiler_dtype d) {
    switch (d) {
        case GGML_COMPILER_DTYPE_F32: return "f32";
        case GGML_COMPILER_DTYPE_F16: return "f16";
        case GGML_COMPILER_DTYPE_I32: return "i32";
    }
    return "?";
}

static void print_info(const char * kind, const ggml_compiler_tensor_info * info) {
    printf("  %-6s %-24s %s [", kind, info->name, dtype_name(info->dtype));
    for (int d = 0; d < info->ndim; d++) {
        printf(d ? ", %lld" : "%lld", (long long) info->shape[d]);
    }
    printf("]\n");
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s MODEL.gguf [cpu|cuda|auto]\n", argv[0]);
        return 2;
    }
    ggml_compiler_options opt = ggml_compiler_default_options();
    if (argc > 2) {
        opt.backend = strcmp(argv[2], "cpu") == 0 ? GGML_COMPILER_BACKEND_CPU
                    : strcmp(argv[2], "cuda") == 0 ? GGML_COMPILER_BACKEND_CUDA
                                                   : GGML_COMPILER_BACKEND_AUTO;
    }
    ggml_compiler_model * m = ggml_compiler_load(argv[1], &opt);
    if (!m) {
        fprintf(stderr, "error: %s\n", ggml_compiler_last_error());
        return 1;
    }
    printf("%s on %s\n", ggml_compiler_version(), ggml_compiler_backend_name(m));
    ggml_compiler_tensor_info info;
    for (int i = 0; i < ggml_compiler_n_inputs(m); i++) {
        if (ggml_compiler_input_info(m, i, &info) == 0) print_info("input", &info);
    }
    for (int i = 0; i < ggml_compiler_n_outputs(m); i++) {
        if (ggml_compiler_output_info(m, i, &info) == 0) print_info("output", &info);
    }
    ggml_compiler_free(m);
    return 0;
}
