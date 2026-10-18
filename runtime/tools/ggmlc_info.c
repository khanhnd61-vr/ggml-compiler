/* ggmlc-info: load a ggmlc GGUF and print its inputs and outputs. */
#include <stdio.h>
#include <string.h>

#include "ggmlc_rt.h"

static const char * dtype_name(ggmlc_dtype d) {
    switch (d) {
        case GGMLC_DTYPE_F32: return "f32";
        case GGMLC_DTYPE_F16: return "f16";
        case GGMLC_DTYPE_I32: return "i32";
    }
    return "?";
}

static void print_info(const char * kind, const ggmlc_tensor_info * info) {
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
    ggmlc_options opt = ggmlc_default_options();
    if (argc > 2) {
        opt.backend = strcmp(argv[2], "cpu") == 0 ? GGMLC_BACKEND_CPU
                    : strcmp(argv[2], "cuda") == 0 ? GGMLC_BACKEND_CUDA
                                                   : GGMLC_BACKEND_AUTO;
    }
    ggmlc_model * m = ggmlc_load(argv[1], &opt);
    if (!m) {
        fprintf(stderr, "error: %s\n", ggmlc_last_error());
        return 1;
    }
    printf("%s on %s\n", ggmlc_version(), ggmlc_backend_name(m));
    ggmlc_tensor_info info;
    for (int i = 0; i < ggmlc_n_inputs(m); i++) {
        if (ggmlc_input_info(m, i, &info) == 0) print_info("input", &info);
    }
    for (int i = 0; i < ggmlc_n_outputs(m); i++) {
        if (ggmlc_output_info(m, i, &info) == 0) print_info("output", &info);
    }
    ggmlc_free(m);
    return 0;
}
