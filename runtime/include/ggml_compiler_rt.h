/*
 * ggml_compiler_rt.h - C API of the ggml-compiler runtime.
 *
 * A model is a GGUF written by `ggml-compiler compile`: weights plus a serialized program
 * (stage graphs and a driver). The runtime rebuilds each stage as a ggml graph once at
 * load, keeps every tensor that crosses a stage boundary on the backend, and runs the
 * driver on ggml_compiler_run().
 *
 * Every function that returns int returns 0 on success and -1 on failure; functions that
 * return a pointer return NULL on failure. After a failure, ggml_compiler_last_error()
 * describes it. Shapes are reported in PyTorch order (outermost dimension first).
 *
 * Symbolic dims: a program may have symbols (a sequence length, a KV length) that set some
 * tensor sizes at run time, each within a compiled range. Bind them with
 * ggml_compiler_set_symbol, or let ggml_compiler_set_input_shape solve them from an input's
 * shape. Sizes, infos and nbytes follow the current binding; it starts with every symbol at its
 * max. A run with a new binding first rebuilds the graphs of the stages that have symbolic
 * shapes.
 */
#ifndef GGML_COMPILER_RT_H
#define GGML_COMPILER_RT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#    ifdef GGML_COMPILER_RT_BUILD
#        define GGML_COMPILER_API __declspec(dllexport)
#    else
#        define GGML_COMPILER_API __declspec(dllimport)
#    endif
#else
#    define GGML_COMPILER_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_COMPILER_MAX_RANK 8

typedef struct ggml_compiler_model ggml_compiler_model;

typedef enum {
    GGML_COMPILER_BACKEND_CPU  = 0,
    GGML_COMPILER_BACKEND_CUDA = 1,
    GGML_COMPILER_BACKEND_AUTO = 2, /* first GPU if one is available, else CPU */
} ggml_compiler_backend;

typedef enum {
    GGML_COMPILER_DTYPE_F32 = 0,
    GGML_COMPILER_DTYPE_F16 = 1,
    GGML_COMPILER_DTYPE_I32 = 2,
} ggml_compiler_dtype;

typedef struct {
    ggml_compiler_backend backend;
    int                   device;             /* device index within the backend */
    int                   n_threads;          /* CPU threads; <= 0 means all hardware threads */
    bool                  allow_cpu_fallback; /* run nodes the backend lacks on the CPU instead of failing */
} ggml_compiler_options;

typedef struct {
    const char *        name;  /* valid while the model is alive */
    ggml_compiler_dtype dtype;
    int                 ndim;
    int64_t             shape[GGML_COMPILER_MAX_RANK];
    size_t              nbytes;
} ggml_compiler_tensor_info;

typedef struct {
    double ms_total; /* wall time of the last ggml_compiler_run */
    int    n_calls;  /* stage calls executed by the last ggml_compiler_run */
} ggml_compiler_stats;

GGML_COMPILER_API const char *          ggml_compiler_version(void);
GGML_COMPILER_API const char *          ggml_compiler_last_error(void);
GGML_COMPILER_API ggml_compiler_options ggml_compiler_default_options(void);
/* True if this build can run on the backend and at least one device for it is present. */
GGML_COMPILER_API bool                  ggml_compiler_has_backend(ggml_compiler_backend backend);

/* Encode f32 rows in a ggml type ("q8_0", "q4_K", "iq4_nl", ...), exactly as ggml stores it.
   Used by the compiler. Returns the bytes written, or 0 on error (unknown type, n_per_row not a
   multiple of the type's block size, dst_size too small). */
GGML_COMPILER_API size_t ggml_compiler_quantize(const char * type, const float * src, int64_t nrows,
                                                int64_t n_per_row, void * dst, size_t dst_size);
/* Bytes one row of n elements takes in that type, or 0 if the type is unknown. */
GGML_COMPILER_API size_t ggml_compiler_row_size(const char * type, int64_t n);

GGML_COMPILER_API ggml_compiler_model * ggml_compiler_load(const char * path, const ggml_compiler_options * opt);
/* A string from the GGUF's metadata ("ggml_compiler.meta": the program's own JSON metadata, such
   as an LLM's EOS ids), or NULL if the key is absent. Valid while the model is alive. */
GGML_COMPILER_API const char * ggml_compiler_metadata(const ggml_compiler_model * m, const char * key);
GGML_COMPILER_API void         ggml_compiler_free(ggml_compiler_model * m);
GGML_COMPILER_API const char * ggml_compiler_backend_name(const ggml_compiler_model * m);

/* Entry points, e in [0, ggml_compiler_n_entries): ways to run the program (an LLM's prefill and
   decode). The first is the default, which ggml_compiler_n_inputs / ggml_compiler_input_info /
   ggml_compiler_run use. Inputs and outputs are named across all entries: ggml_compiler_set_input
   and ggml_compiler_get_output take any of them. */
GGML_COMPILER_API int          ggml_compiler_n_entries(const ggml_compiler_model * m);
GGML_COMPILER_API const char * ggml_compiler_entry_name(const ggml_compiler_model * m, int e);
GGML_COMPILER_API int          ggml_compiler_entry_n_inputs(const ggml_compiler_model * m, int e);
GGML_COMPILER_API int          ggml_compiler_entry_n_outputs(const ggml_compiler_model * m, int e);
GGML_COMPILER_API int          ggml_compiler_entry_input_info(const ggml_compiler_model * m, int e, int i,
                                                              ggml_compiler_tensor_info * info);
GGML_COMPILER_API int          ggml_compiler_entry_output_info(const ggml_compiler_model * m, int e, int i,
                                                               ggml_compiler_tensor_info * info);
/* Run one entry point by name; its inputs must have been set. */
GGML_COMPILER_API int          ggml_compiler_run_entry(ggml_compiler_model * m, const char * entry);
/* Zero the program's state (a KV cache): the values that persist across runs. Load does it too. */
GGML_COMPILER_API int          ggml_compiler_reset(ggml_compiler_model * m);

GGML_COMPILER_API int ggml_compiler_n_inputs(const ggml_compiler_model * m);
GGML_COMPILER_API int ggml_compiler_n_outputs(const ggml_compiler_model * m);
GGML_COMPILER_API int ggml_compiler_input_info(const ggml_compiler_model * m, int i, ggml_compiler_tensor_info * info);
GGML_COMPILER_API int ggml_compiler_output_info(const ggml_compiler_model * m, int i, ggml_compiler_tensor_info * info);

/* Symbols, i in [0, ggml_compiler_n_symbols). Any out pointer may be NULL. */
GGML_COMPILER_API int ggml_compiler_n_symbols(const ggml_compiler_model * m);
GGML_COMPILER_API int ggml_compiler_symbol_info(const ggml_compiler_model * m, int i, const char ** name,
                                                int64_t * min, int64_t * max, int64_t * value);
GGML_COMPILER_API int ggml_compiler_set_symbol(ggml_compiler_model * m, const char * name, int64_t value);
/* Bind the symbols an input's shape determines; the shape must be one the program accepts. */
GGML_COMPILER_API int ggml_compiler_set_input_shape(ggml_compiler_model * m, const char * name, int ndim,
                                                    const int64_t * shape);

/* nbytes must equal the tensor's size for the current binding exactly. */
GGML_COMPILER_API int ggml_compiler_set_input(ggml_compiler_model * m, const char * name, const void * data,
                                              size_t nbytes);
GGML_COMPILER_API int ggml_compiler_run(ggml_compiler_model * m);
GGML_COMPILER_API int ggml_compiler_get_output(ggml_compiler_model * m, const char * name, void * dst, size_t nbytes);

GGML_COMPILER_API int ggml_compiler_last_stats(const ggml_compiler_model * m, ggml_compiler_stats * st);
/* Per-stage timings of the last run, i in [0, ggml_compiler_n_stages). */
GGML_COMPILER_API int ggml_compiler_n_stages(const ggml_compiler_model * m);
GGML_COMPILER_API int ggml_compiler_stage_stats(const ggml_compiler_model * m, int i, const char ** name, double * ms,
                                                int * n_calls);

/* --- Compiled stage builders (used by C++ generated with `ggml-compiler compile --emit cpp`) ----
 * A builder replaces the runtime's own graph construction for one stage: it receives the
 * stage's parameters, looks weights up by GGUF name, builds the ggml graph and names its outputs.
 * Everything else (weights, driver, buffers, the C API) stays the runtime's. */
struct ggml_context;
struct ggml_tensor;
typedef struct ggml_compiler_stage_build ggml_compiler_stage_build;
typedef int (*ggml_compiler_build_fn)(ggml_compiler_stage_build * b); /* 0 on success */

typedef struct {
    const char *           stage;
    ggml_compiler_build_fn build;
} ggml_compiler_builder;

GGML_COMPILER_API struct ggml_context * ggml_compiler_build_ctx(ggml_compiler_stage_build * b);
GGML_COMPILER_API bool                  ggml_compiler_build_gpu(ggml_compiler_stage_build * b);
/* Whether the backend being built for can run node t (as ggml_backend_supports_op). */
GGML_COMPILER_API bool                  ggml_compiler_build_supports(ggml_compiler_stage_build * b,
                                                                     const struct ggml_tensor * t);
GGML_COMPILER_API struct ggml_tensor *  ggml_compiler_build_param(ggml_compiler_stage_build * b, int i);
GGML_COMPILER_API struct ggml_tensor *  ggml_compiler_build_weight(ggml_compiler_stage_build * b,
                                                                   const char * gguf_name);
/* The value of symbol i for the graph being built. */
GGML_COMPILER_API int64_t               ggml_compiler_build_symbol(ggml_compiler_stage_build * b, int i);
GGML_COMPILER_API void                  ggml_compiler_build_output(ggml_compiler_stage_build * b, int i,
                                                                   struct ggml_tensor * t);

/* ggml_compiler_load, with the listed stages built by compiled builders. The GGUF's program must
   have those stages with the same parameters and outputs. */
GGML_COMPILER_API ggml_compiler_model * ggml_compiler_load_with(const char * path, const ggml_compiler_options * opt,
                                                                const ggml_compiler_builder * builders,
                                                                int n_builders);

#ifdef __cplusplus
}
#endif

#endif /* GGML_COMPILER_RT_H */
