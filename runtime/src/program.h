// Parsed form of the `ggml_compiler.program` JSON stored in a ggml-compiler GGUF.
//
// Everything here is already in ggml terms: `ne` is innermost-first and op parameters are
// ggml axes. The Python emitter (ggml_compiler/emit/gguf_runtime.py) does the PyTorch-to-ggml
// translation; the runtime only checks and dispatches.
//
// A dim may be symbolic (format 3): an expression over the program's symbols. Specs keep the
// expressions, and ProgramSpec::bind writes their values for one binding into the plain `ne`,
// `shape` and parameter fields, which is all graph building reads.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ggml.h"
#include "nlohmann/json.hpp"

namespace ggml_compiler {

constexpr int kFormatVersion    = 3;  // 2: loop slices; 3: symbolic dims
constexpr int kMinFormatVersion = 1;

// An integer expression over symbols: a constant, symbol k, or an operation on sub-expressions.
struct Expr {
    enum class Kind { Const, Sym, Add, Mul, FloorDiv, Mod, Max, Min };
    Kind              kind = Kind::Const;
    int64_t           value = 0;  // Const: the value; Sym: the symbol index
    std::vector<Expr> args;

    bool    is_const() const { return kind == Kind::Const; }
    int64_t eval(const std::vector<int64_t> & binding) const;
    void    symbols(std::vector<bool> & used) const;  // marks the symbols it reads
};

struct Symbol {
    std::string name;
    int64_t     min = 1;
    int64_t     max = 1;
};

struct TensorSpec {
    std::string            name;
    ggml_type              type = GGML_TYPE_F32;
    std::array<int64_t, 4> ne{1, 1, 1, 1};  // for the bound symbols
    std::vector<int64_t>   shape;           // PyTorch order, as the user sees it; for the bound symbols
    std::array<Expr, 4>    ne_x;
    std::vector<Expr>      shape_x;
};

struct NodeSpec {
    std::string              id;   // "%N"
    std::string              op;
    std::vector<std::string> src;  // node ids, stage params "a.N", or GGUF tensors "w.N"/"c.N"
    ggml_type                type = GGML_TYPE_F32;
    std::array<int64_t, 4>   ne{1, 1, 1, 1};  // for the bound symbols
    nlohmann::json           p;    // op parameters, symbolic ones written for the bound symbols
    std::string              loc;  // source location, for error messages
    std::array<Expr, 4>                      ne_x;
    std::vector<std::pair<std::string, Expr>> p_x;  // symbolic parameters
};

struct StageSpec {
    std::string              name;
    std::vector<TensorSpec>  params;   // named "a.0", "a.1", ...
    std::vector<std::string> outputs;  // ids of the values the stage returns
    std::vector<NodeSpec>    nodes;    // topological order
    std::vector<bool>        uses;     // per symbol: does any shape in the stage read it
};

struct DriverOp {
    enum class Kind { Call, Loop };
    Kind kind = Kind::Call;

    // Call: run one stage. args/out name driver values.
    std::string              stage;
    std::vector<std::string> args;
    std::vector<std::string> out;
    int                      call_site = -1;  // set by the model at load

    // Loop: copy each carry's initial value, then n times: upload this iteration's scalars, copy
    // row i of each slice source into its loop value, set the counters, bind the symbols, fill the
    // masks, run the body, collect, and stop early if an `until` value is one of its ids.
    int                                                     n = 0;
    int                                                     n_symbol = -1;  // n is this symbol's value
    std::vector<std::pair<std::string, std::string>>        counters;  // (loop value, i32 start value)
    std::vector<std::pair<int, Expr>>                       bind;      // (symbol, over symbols + counters)
    std::vector<std::pair<std::string, int>>                masks;     // (loop value, counter index)
    std::vector<std::pair<std::string, std::vector<int>>>   until;     // (loop value, stop ids)
    std::vector<std::pair<std::string, std::string>>        collect;   // (driver value, loop value)
    std::vector<std::pair<std::string, std::string>>        carry;    // (loop value, initial value)
    std::vector<std::pair<std::string, std::vector<float>>> scalars;  // (value, one entry per iteration)
    std::vector<std::pair<std::string, std::string>>        slices;   // (loop value, source [n, ...])
    std::vector<DriverOp>                                   body;
    std::vector<std::vector<ggml_tensor *>>                 slice_rows;  // per slice, n views; set at load
};

// One way to run the program (format 3 may have several; the first is the default).
struct EntrySpec {
    std::string              name;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<DriverOp>    driver;
};

struct ProgramSpec {
    int                      format_version = 0;
    std::vector<Symbol>      symbols;
    std::vector<TensorSpec>  values;   // every driver value (program I/O and inter-stage tensors)
    std::vector<StageSpec>   stages;
    std::vector<EntrySpec>   entries;
    std::vector<std::string> state;    // values that persist across runs

    // Evaluate every symbolic dim and parameter for `binding` (one value per symbol, in range).
    void bind(const std::vector<int64_t> & binding);
};

// ne and shape of a tensor spec for a binding, without touching the spec.
std::array<int64_t, 4> eval_ne(const TensorSpec & t, const std::vector<int64_t> & binding);
std::vector<int64_t>   eval_shape(const TensorSpec & t, const std::vector<int64_t> & binding);

// Throws std::runtime_error with a readable message on any schema problem.
ProgramSpec parse_program(const std::string & json_text);

ggml_type parse_type(const std::string & name);

}  // namespace ggml_compiler
