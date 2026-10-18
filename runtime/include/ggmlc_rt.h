/*
 * ggmlc_rt.h - C API of the ggmlc runtime.
 *
 * A model is a GGUF written by `ggmlc compile`: weights plus a serialized program
 * (stage graphs and a driver). The runtime rebuilds each stage as a ggml graph once at
 * load, keeps every tensor that crosses a stage boundary on the backend, and runs the
 * driver on ggmlc_run().
 *
 * Every function that returns int returns 0 on success and -1 on failure; functions that
 * return a pointer return NULL on failure. After a failure, ggmlc_last_error() describes it.
 * Shapes are reported in PyTorch order (outermost dimension first).
 *
 * Symbolic dims: a program may have symbols (a sequence length, a KV length) that set some
 * tensor sizes at run time, each within a compiled range. Bind them with ggmlc_set_symbol, or
 * let ggmlc_set_input_shape solve them from an input's shape. Sizes, infos and nbytes follow the
 * current binding; it starts with every symbol at its max. A run with a new binding first
 * rebuilds the graphs of the stages that have symbolic shapes.
 */
#ifndef GGMLC_RT_H
#define GGMLC_RT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#    ifdef GGMLC_RT_BUILD
#        define GGMLC_API __declspec(dllexport)
#    else
#        define GGMLC_API __declspec(dllimport)
#    endif
#else
#    define GGMLC_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define GGMLC_MAX_RANK 8

typedef struct ggmlc_model ggmlc_model;

typedef enum {
    GGMLC_BACKEND_CPU  = 0,
    GGMLC_BACKEND_CUDA = 1,
    GGMLC_BACKEND_AUTO = 2, /* first GPU if one is available, else CPU */
} ggmlc_backend;

typedef enum {
    GGMLC_DTYPE_F32 = 0,
    GGMLC_DTYPE_F16 = 1,
    GGMLC_DTYPE_I32 = 2,
} ggmlc_dtype;

typedef struct {
    ggmlc_backend backend;
    int           device;             /* device index within the backend */
    int           n_threads;          /* CPU threads; <= 0 means all hardware threads */
    bool          allow_cpu_fallback; /* run nodes the backend lacks on the CPU instead of failing */
} ggmlc_options;

typedef struct {
    const char * name;  /* valid while the model is alive */
    ggmlc_dtype  dtype;
    int          ndim;
    int64_t      shape[GGMLC_MAX_RANK];
    size_t       nbytes;
} ggmlc_tensor_info;

typedef struct {
    double ms_total; /* wall time of the last ggmlc_run */
    int    n_calls;  /* stage calls executed by the last ggmlc_run */
} ggmlc_stats;

GGMLC_API const char *  ggmlc_version(void);
GGMLC_API const char *  ggmlc_last_error(void);
GGMLC_API ggmlc_options ggmlc_default_options(void);
/* True if this build can run on the backend and at least one device for it is present. */
GGMLC_API bool          ggmlc_has_backend(ggmlc_backend backend);

/* Encode f32 rows in a ggml type ("q8_0", "q4_K", "iq4_nl", ...), exactly as ggml stores it.
   Used by the compiler. Returns the bytes written, or 0 on error (unknown type, n_per_row not a
   multiple of the type's block size, dst_size too small). */
GGMLC_API size_t ggmlc_quantize(const char * type, const float * src, int64_t nrows, int64_t n_per_row, void * dst,
                                size_t dst_size);
/* Bytes one row of n elements takes in that type, or 0 if the type is unknown. */
GGMLC_API size_t ggmlc_row_size(const char * type, int64_t n);

GGMLC_API ggmlc_model * ggmlc_load(const char * path, const ggmlc_options * opt);
/* A string from the GGUF's metadata ("ggmlc.meta": the program's own JSON metadata, such as an
   LLM's EOS ids), or NULL if the key is absent. Valid while the model is alive. */
GGMLC_API const char *  ggmlc_metadata(const ggmlc_model * m, const char * key);
GGMLC_API void          ggmlc_free(ggmlc_model * m);
GGMLC_API const char *  ggmlc_backend_name(const ggmlc_model * m);

/* Entry points, e in [0, ggmlc_n_entries): ways to run the program (an LLM's prefill and decode).
   The first is the default, which ggmlc_n_inputs / ggmlc_input_info / ggmlc_run use. Inputs and
   outputs are named across all entries: ggmlc_set_input and ggmlc_get_output take any of them. */
GGMLC_API int          ggmlc_n_entries(const ggmlc_model * m);
GGMLC_API const char * ggmlc_entry_name(const ggmlc_model * m, int e);
GGMLC_API int          ggmlc_entry_n_inputs(const ggmlc_model * m, int e);
GGMLC_API int          ggmlc_entry_n_outputs(const ggmlc_model * m, int e);
GGMLC_API int          ggmlc_entry_input_info(const ggmlc_model * m, int e, int i, ggmlc_tensor_info * info);
GGMLC_API int          ggmlc_entry_output_info(const ggmlc_model * m, int e, int i, ggmlc_tensor_info * info);
/* Run one entry point by name; its inputs must have been set. */
GGMLC_API int          ggmlc_run_entry(ggmlc_model * m, const char * entry);
/* Zero the program's state (a KV cache): the values that persist across runs. Load does it too. */
GGMLC_API int          ggmlc_reset(ggmlc_model * m);

GGMLC_API int ggmlc_n_inputs(const ggmlc_model * m);
GGMLC_API int ggmlc_n_outputs(const ggmlc_model * m);
GGMLC_API int ggmlc_input_info(const ggmlc_model * m, int i, ggmlc_tensor_info * info);
GGMLC_API int ggmlc_output_info(const ggmlc_model * m, int i, ggmlc_tensor_info * info);

/* Symbols, i in [0, ggmlc_n_symbols). Any out pointer may be NULL. */
GGMLC_API int ggmlc_n_symbols(const ggmlc_model * m);
GGMLC_API int ggmlc_symbol_info(const ggmlc_model * m, int i, const char ** name, int64_t * min, int64_t * max,
                                int64_t * value);
GGMLC_API int ggmlc_set_symbol(ggmlc_model * m, const char * name, int64_t value);
/* Bind the symbols an input's shape determines; the shape must be one the program accepts. */
GGMLC_API int ggmlc_set_input_shape(ggmlc_model * m, const char * name, int ndim, const int64_t * shape);

/* nbytes must equal the tensor's size for the current binding exactly. */
GGMLC_API int ggmlc_set_input(ggmlc_model * m, const char * name, const void * data, size_t nbytes);
GGMLC_API int ggmlc_run(ggmlc_model * m);
GGMLC_API int ggmlc_get_output(ggmlc_model * m, const char * name, void * dst, size_t nbytes);

GGMLC_API int ggmlc_last_stats(const ggmlc_model * m, ggmlc_stats * st);
/* Per-stage timings of the last run, i in [0, ggmlc_n_stages). */
GGMLC_API int ggmlc_n_stages(const ggmlc_model * m);
GGMLC_API int ggmlc_stage_stats(const ggmlc_model * m, int i, const char ** name, double * ms, int * n_calls);

/* --- Compiled stage builders (used by C++ generated with `ggmlc compile --emit cpp`) ----------
 * A builder replaces the runtime's own graph construction for one stage: it receives the
 * stage's parameters, looks weights up by GGUF name, builds the ggml graph and names its outputs.
 * Everything else (weights, driver, buffers, the C API) stays the runtime's. */
struct ggml_context;
struct ggml_tensor;
typedef struct ggmlc_stage_build ggmlc_stage_build;
typedef int (*ggmlc_build_fn)(ggmlc_stage_build * b); /* 0 on success */

typedef struct {
    const char *   stage;
    ggmlc_build_fn build;
} ggmlc_builder;

GGMLC_API struct ggml_context * ggmlc_build_ctx(ggmlc_stage_build * b);
GGMLC_API bool                  ggmlc_build_gpu(ggmlc_stage_build * b);
GGMLC_API struct ggml_tensor *  ggmlc_build_param(ggmlc_stage_build * b, int i);
GGMLC_API struct ggml_tensor *  ggmlc_build_weight(ggmlc_stage_build * b, const char * gguf_name);
/* The value of symbol i for the graph being built. */
GGMLC_API int64_t               ggmlc_build_symbol(ggmlc_stage_build * b, int i);
GGMLC_API void                  ggmlc_build_output(ggmlc_stage_build * b, int i, struct ggml_tensor * t);

/* ggmlc_load, with the listed stages built by compiled builders. The GGUF's program must have
   those stages with the same parameters and outputs. */
GGMLC_API ggmlc_model * ggmlc_load_with(const char * path, const ggmlc_options * opt, const ggmlc_builder * builders,
                                        int n_builders);

#ifdef __cplusplus
}
#endif

#endif /* GGMLC_RT_H */
