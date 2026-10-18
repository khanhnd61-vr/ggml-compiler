"""Emit a compiled program as readable C++: one ggml graph builder per stage.

The builders are generated from the same program JSON the runtime interprets, and make the same
ggml calls in the same order (multi-call ops go through runtime/include/ggmlc_ops.h, which the
runtime's dispatcher also uses). Loaded with `<name>_load`, they replace the runtime's graph
construction and nothing else, so outputs match the runtime bit for bit on the same backend and
thread count. `<name>-check` verifies exactly that.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

_GGML_TYPES = {"f32": "GGML_TYPE_F32", "f16": "GGML_TYPE_F16", "bf16": "GGML_TYPE_BF16", "i32": "GGML_TYPE_I32"}
_UNARY = {"relu", "gelu", "gelu_erf", "silu", "tanh", "sigmoid", "exp", "log", "neg", "abs", "sqr", "sqrt", "sin",
          "cos", "softplus", "cont", "mean", "cumsum", "step", "sum_rows", "argmax"}
_BINARY = {"add", "sub", "mul", "div", "get_rows"}


def _f(v: float) -> str:
    """A C++ float literal that parses to the same f32 the runtime reads from the JSON (9
    significant digits round-trip any f32). Parameters are finite: the emitter clamps infinities."""
    x = float(np.float32(v))
    if not np.isfinite(x):
        raise ValueError(f"non-finite op parameter {v}")
    text = f"{x:.9g}"
    return text + ("f" if any(c in text for c in ".e") else ".0f")


def _ident(name: str, taken: set[str]) -> str:
    base = re.sub(r"[^0-9A-Za-z_]", "_", name).strip("_") or "t"
    if base[0].isdigit():
        base = "w_" + base
    out, k = base, 2
    while out in taken:
        out = f"{base}_{k}"
        k += 1
    taken.add(out)
    return out


def _dim(j: Any, syms: dict[str, str]) -> str:
    """A dim from the program JSON (see ggmlc/ir/sym.py) as a C++ int64_t expression."""
    if isinstance(j, int):
        return str(j)
    if isinstance(j, str):
        return syms[j]
    op, *args = j
    xs = [_dim(a, syms) for a in args]
    if op in ("+", "*"):
        return "(" + f" {op} ".join(xs) + ")"
    fn = {"//": "floordiv", "%": "mod", "max": "max", "min": "min"}[op]
    out = xs[0]
    for x in xs[1:]:
        out = f"ggmlc::ops::{fn}({out}, {x})"
    return out


def _call(n: dict[str, Any], a: list[str], syms: dict[str, str]) -> str:
    op, p = n["op"], n["p"]
    ne = [_dim(d, syms) for d in n["ne"]]
    if op in _BINARY:
        return f"ggml_{op}(ctx, {a[0]}, {a[1]})"
    if op == "mul_mat":  # f32 accumulation (Stage(matmul_acc="f32")) needs the helper; plain ggml otherwise
        return f"ggmlc::ops::mul_mat(ctx, {a[0]}, {a[1]}, true)" if p.get("acc") else f"ggml_mul_mat(ctx, {a[0]}, {a[1]})"
    if op in _UNARY:
        return f"ggml_{op}(ctx, {a[0]})"
    if op == "gelu_tanh":
        return f"ggmlc::ops::gelu_tanh(ctx, {a[0]}, gpu)"
    if op == "scale":
        return f"ggmlc::ops::scale(ctx, {a[0]}, {_f(p['s'])}, {_f(p.get('b', 0.0))})"
    if op == "clamp":
        return f"ggml_clamp(ctx, {a[0]}, {_f(p['min'])}, {_f(p['max'])})"
    if op in ("norm", "rms_norm"):
        return f"ggml_{op}(ctx, {a[0]}, {_f(p['eps'])})"
    if op == "soft_max":
        mask = a[1] if len(a) > 1 else "nullptr"
        return f"ggml_soft_max_ext(ctx, {a[0]}, {mask}, {_f(p.get('scale', 1.0))}, {_f(p.get('max_bias', 0.0))})"
    if op == "reshape":
        return f"ggmlc::ops::reshape(ctx, {a[0]}, {ne[0]}, {ne[1]}, {ne[2]}, {ne[3]})"
    if op == "permute":
        ax = p["axes"]
        return f"ggml_permute(ctx, {a[0]}, {ax[0]}, {ax[1]}, {ax[2]}, {ax[3]})"
    if op == "repeat":
        return f"ggml_repeat_4d(ctx, {a[0]}, {ne[0]}, {ne[1]}, {ne[2]}, {ne[3]})"
    if op == "concat":
        return f"ggml_concat(ctx, {a[0]}, {a[1]}, {p['dim']})"
    if op == "view":
        return f"ggmlc::ops::slice(ctx, {a[0]}, {p['dim']}, {_dim(p['start'], syms)}, {ne[0]}, {ne[1]}, {ne[2]}, {ne[3]})"
    if op == "arange":
        return (f"ggml_arange(ctx, (float) {_dim(p['start'], syms)}, (float) {_dim(p['stop'], syms)}, "
                f"{_f(p['step'])})")
    if op == "swiglu":
        return f"ggml_swiglu_split(ctx, {a[0]}, {a[1]})"
    if op == "rope":
        ff = a[2] if len(a) > 2 else "nullptr"
        return (f"ggml_rope_ext(ctx, {a[0]}, {a[1]}, {ff}, {p['n_dims']}, {p['mode']}, 0, {_f(p['freq_base'])}, "
                f"{_f(p['freq_scale'])}, 0.0f, {_f(p['attn_factor'])}, 32.0f, 1.0f)")
    if op == "set_rows":
        return f"ggml_set_rows(ctx, {a[0]}, {a[1]}, {a[2]})"
    if op == "cast":
        return f"ggml_cast(ctx, {a[0]}, {_GGML_TYPES[n['type']]})"
    if op == "im2col":
        return (f"ggml_im2col(ctx, {a[0]}, {a[1]}, {p['s0']}, {p['s1']}, {p['p0']}, {p['p1']}, {p['d0']}, {p['d1']}, "
                "true, GGML_TYPE_F32)")
    if op == "conv_2d":
        bias = a[2] if len(a) > 2 else "nullptr"
        return (f"ggmlc::ops::conv_2d(ctx, {a[0]}, {a[1]}, {bias}, {p['s0']}, {p['s1']}, {p['p0']}, {p['p1']}, "
                f"{p['d0']}, {p['d1']}, gpu)")
    if op == "pool_2d":
        kind = "GGML_OP_POOL_MAX" if p["op"] == "max" else "GGML_OP_POOL_AVG"
        return (f"ggml_pool_2d(ctx, {a[0]}, {kind}, {p['k0']}, {p['k1']}, {p['s0']}, {p['s1']}, {_f(p['p0'])}, "
                f"{_f(p['p1'])})")
    if op == "flash_attn":
        mask = a[3] if len(a) > 3 else "nullptr"
        return f"ggmlc::ops::flash_attn(ctx, {a[0]}, {a[1]}, {a[2]}, {mask}, {_f(p['scale'])})"
    raise ValueError(f"no C++ emission for op {op!r}")


def _used_symbols(j: Any) -> set[str]:
    """Symbols one dim (an int, a symbol name, or [op, operand...]) reads."""
    if isinstance(j, str):
        return {j}
    if isinstance(j, list) and j:
        return set().union(*(_used_symbols(x) for x in j[1:]))
    return set()


def _stage_cpp(stage: dict[str, Any], fqns: dict[str, str], symbols: list[dict[str, Any]]) -> tuple[str, str]:
    fn = _ident(f"build_{stage['name']}", set())
    taken = {"ctx", "gpu", "B"}
    names: dict[str, str] = {}
    lines = [f"// Stage {stage['name']!r}: {len(stage['nodes'])} nodes.",
             f"int {fn}(ggmlc_stage_build * B) {{",
             "    ggml_context * ctx = ggmlc_build_ctx(B);",
             "    const bool     gpu = ggmlc_build_gpu(B);",
             "    (void) gpu;"]
    used: set[str] = set()
    for n in stage["nodes"]:
        for d in [*n["ne"], n["p"].get("start", 0), n["p"].get("stop", 0)]:
            used |= _used_symbols(d)
    syms: dict[str, str] = {}
    for i, sy in enumerate(symbols):
        if sy["name"] in used:
            syms[sy["name"]] = v = _ident(sy["name"], taken)
            lines.append(f"    const int64_t  {v} = ggmlc_build_symbol(B, {i});  // in [{sy['min']}, {sy['max']}]")
    lines += ["", "    // Parameters"]
    for i, prm in enumerate(stage["params"]):
        names[prm["name"]] = v = _ident(f"in{i}", taken)
        lines.append(f"    ggml_tensor * {v} = ggmlc_build_param(B, {i});  // {prm['type']} {prm['shape']}")
    leaves = []
    for n in stage["nodes"]:
        for src in n["src"]:
            if (src.startswith("w.") or src.startswith("c.")) and src not in names and src not in leaves:
                leaves.append(src)
    if leaves:
        lines += ["", "    // Weights and constants, by GGUF name"]
    for src in leaves:
        label = fqns.get(src, f"const_{src[2:]}")
        names[src] = v = _ident(label, taken)
        lines.append(f'    ggml_tensor * {v} = ggmlc_build_weight(B, "{src}");')
    lines += ["", "    // Graph"]
    for n in stage["nodes"]:
        names[n["id"]] = v = _ident(f"t{n['id'][1:]}", taken)
        where = "  ".join(x for x in (n["op"], n.get("scope", ""), n.get("loc", "")) if x)
        lines.append(f"    ggml_tensor * {v} = {_call(n, [names[s] for s in n['src']], syms)};  // {where}")
    lines += [""] + [f"    ggmlc_build_output(B, {i}, {names[o]});" for i, o in enumerate(stage["outputs"])]
    lines += ["    return 0;", "}"]
    return fn, "\n".join(lines)


def write_cpp(program: dict[str, Any], fqns: dict[str, str], out_dir: Path, name: str, ggmlc_root: Path,
              example_binding: dict[str, int] | None = None) -> list[Path]:
    """Write <name>.h, <name>.cpp, main.cpp and CMakeLists.txt into out_dir. `example_binding`
    is the symbols' values for example_inputs.bin."""
    out_dir.mkdir(parents=True, exist_ok=True)
    builders, bodies = [], []
    symbols = program.get("symbols", [])
    bind_lines = "".join(f'    if (ggmlc_set_symbol(ref, "{k}", {v}) || ggmlc_set_symbol(gen, "{k}", {v})) die("ggmlc_set_symbol");\n'
                         for k, v in (example_binding or {}).items())
    for st in program["stages"]:
        fn, body = _stage_cpp(st, fqns, symbols)
        builders.append((st["name"], fn))
        bodies.append(body)

    header = f'''// {name}.h - generated by ggmlc. Graph builders for {name}, compiled in.
#pragma once

#include "ggmlc_rt.h"

#ifdef __cplusplus
extern "C" {{
#endif

/* Load the model's GGUF (written by the same `ggmlc compile`) with the stage graphs built by the
   code in {name}.cpp instead of the runtime's interpreter. Use the rest of ggmlc_rt.h as usual. */
ggmlc_model * {name}_load(const char * gguf_path, const ggmlc_options * opt);

#ifdef __cplusplus
}}
#endif
'''
    table = "\n".join(f'    {{"{s}", {fn}}},' for s, fn in builders)
    source = f'''// {name}.cpp - generated by ggmlc: one ggml graph builder per stage.
//
// These builders make the same ggml calls, in the same order, as the runtime's interpreter, so
// the outputs are bit-identical. Edit them to fuse or reorder by hand; then run {name}-check to see
// how far the results move.
#include "{name}.h"

#include <cmath>

#include "ggml.h"
#include "ggmlc_ops.h"

namespace {{

{chr(10).join(chr(10) + b for b in bodies)}

const ggmlc_builder kBuilders[] = {{
{table}
}};

}}  // namespace

ggmlc_model * {name}_load(const char * gguf_path, const ggmlc_options * opt) {{
    return ggmlc_load_with(gguf_path, opt, kBuilders, (int) (sizeof(kBuilders) / sizeof(kBuilders[0])));
}}
'''
    check = f'''// {name}-check MODEL.gguf INPUTS.bin [cpu|cuda|auto] [threads]
//
// Runs the model twice from the same GGUF and inputs: with the runtime interpreting the program,
// and with the builders compiled from {name}.cpp. Prints, per output, whether they agree bit for
// bit and the largest difference, and both timings. Exit status 0 only if every output agrees.
// INPUTS.bin holds the inputs' raw bytes back to back, in input order (`ggmlc compile` writes
// example_inputs.bin next to the GGUF).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "{name}.h"

static void die(const char * what) {{
    std::fprintf(stderr, "error: %s: %s\\n", what, ggmlc_last_error());
    std::exit(2);
}}

int main(int argc, char ** argv) {{
    if (argc < 3) {{
        std::fprintf(stderr, "usage: %s MODEL.gguf INPUTS.bin [cpu|cuda|auto] [threads]\\n", argv[0]);
        return 2;
    }}
    ggmlc_options opt = ggmlc_default_options();
    if (argc > 3) {{
        opt.backend = !std::strcmp(argv[3], "cpu") ? GGMLC_BACKEND_CPU
                    : !std::strcmp(argv[3], "cuda") ? GGMLC_BACKEND_CUDA : GGMLC_BACKEND_AUTO;
    }}
    if (argc > 4) opt.n_threads = std::atoi(argv[4]);

    ggmlc_model * ref = ggmlc_load(argv[1], &opt);
    if (!ref) die("ggmlc_load");
    ggmlc_model * gen = {name}_load(argv[1], &opt);
    if (!gen) die("{name}_load");
{bind_lines}
    std::ifstream in(argv[2], std::ios::binary);
    std::vector<std::string> fed;  // every input once, in entry order: how INPUTS.bin is laid out
    bool all_same = true;
    double ms_ref = 0.0, ms_gen = 0.0;
    for (int e = 0; e < ggmlc_n_entries(ref); e++) {{
        for (int i = 0; i < ggmlc_entry_n_inputs(ref, e); i++) {{
            ggmlc_tensor_info info;
            ggmlc_entry_input_info(ref, e, i, &info);
            if (std::find(fed.begin(), fed.end(), info.name) != fed.end()) continue;
            fed.push_back(info.name);
            std::vector<char> buf(info.nbytes);
            if (!in.read(buf.data(), (std::streamsize) buf.size())) {{
                std::fprintf(stderr, "error: %s ends before input %s\\n", argv[2], info.name);
                return 2;
            }}
            if (ggmlc_set_input(ref, info.name, buf.data(), buf.size()) || ggmlc_set_input(gen, info.name, buf.data(), buf.size())) {{
                die("ggmlc_set_input");
            }}
        }}
        const char * entry = ggmlc_entry_name(ref, e);
        if (ggmlc_run_entry(ref, entry)) die("ggmlc_run (runtime)");
        if (ggmlc_run_entry(gen, entry)) die("ggmlc_run ({name})");
        ggmlc_stats sr, sg;
        ggmlc_last_stats(ref, &sr);
        ggmlc_last_stats(gen, &sg);
        ms_ref += sr.ms_total;
        ms_gen += sg.ms_total;
        for (int i = 0; i < ggmlc_entry_n_outputs(ref, e); i++) {{
            ggmlc_tensor_info info;
            ggmlc_entry_output_info(ref, e, i, &info);
            std::vector<char> a(info.nbytes), b(info.nbytes);
            ggmlc_get_output(ref, info.name, a.data(), a.size());
            ggmlc_get_output(gen, info.name, b.data(), b.size());
            const bool same = std::memcmp(a.data(), b.data(), a.size()) == 0;
            double     diff = 0.0;
            if (info.dtype == GGMLC_DTYPE_F32) {{
                const float * x = (const float *) a.data();
                const float * y = (const float *) b.data();
                for (size_t k = 0; k < a.size() / sizeof(float); k++) diff = std::max(diff, (double) std::fabs(x[k] - y[k]));
            }}
            std::printf("%-32s %s  max|d| %g\\n", info.name, same ? "bit-identical" : "DIFFERENT", diff);
            all_same = all_same && same;
        }}
    }}
    std::printf("runtime %.3f ms, {name} %.3f ms (%s)\\n", ms_ref, ms_gen, ggmlc_backend_name(gen));
    ggmlc_free(gen);
    ggmlc_free(ref);
    return all_same ? 0 : 1;
}}
'''
    cmake = f'''cmake_minimum_required(VERSION 3.22)
project({name} LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The ggmlc checkout that generated this project; its runtime/ is built here unless a prebuilt
# libggmlc_rt is given (GGMLC_RUNTIME_LIB plus GGMLC_GGML_INCLUDE for ggml's headers).
set(GGMLC_ROOT "{ggmlc_root.as_posix()}" CACHE PATH "ggml-compiler checkout")
set(GGMLC_RUNTIME_LIB "" CACHE FILEPATH "prebuilt libggmlc_rt (optional)")
set(GGMLC_GGML_INCLUDE "${{GGMLC_ROOT}}/.deps/llama.cpp-b11223/ggml/include" CACHE PATH "ggml headers")

if(GGMLC_RUNTIME_LIB)
    add_library(ggmlc_rt SHARED IMPORTED)
    set_target_properties(ggmlc_rt PROPERTIES
        IMPORTED_LOCATION "${{GGMLC_RUNTIME_LIB}}"
        INTERFACE_INCLUDE_DIRECTORIES "${{GGMLC_ROOT}}/runtime/include;${{GGMLC_GGML_INCLUDE}}")
else()
    add_subdirectory("${{GGMLC_ROOT}}/runtime" ggmlc_rt)
endif()

add_library({name} STATIC {name}.cpp)
set_target_properties({name} PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories({name} PUBLIC ${{CMAKE_CURRENT_SOURCE_DIR}})
target_link_libraries({name} PUBLIC ggmlc_rt)

add_executable({name}-check main.cpp)
target_link_libraries({name}-check PRIVATE {name})
'''
    files = {f"{name}.h": header, f"{name}.cpp": source, "main.cpp": check, "CMakeLists.txt": cmake}
    paths = []
    for fname, text in files.items():
        path = out_dir / fname
        path.write_text(text)
        paths.append(path)
    return paths
