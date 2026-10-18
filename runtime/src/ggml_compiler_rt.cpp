// ggml-compiler runtime: loads a ggml-compiler GGUF, builds its stage graphs, runs the driver.
//
// Memory layout:
// - weights and constants live in one backend buffer, filled from the GGUF;
// - every driver value (program inputs, outputs, tensors passed between stages) lives in a
//   second persistent backend buffer, so nothing returns to the host between stage calls;
// - each driver call site owns a ggml graph and a gallocr compute buffer.
// A stage reads its arguments straight from the driver value tensors and writes its results
// back with ggml_cpy nodes appended after all other nodes, so a call may safely read and
// write the same value (a loop carry).
//
// Symbolic dims: a value whose shape depends on a symbol gets a buffer for its largest size and,
// per binding, a view with that binding's shape. When the binding changes, the next run rebuilds
// the views and the graphs of the call sites (their gallocr buffers are reused, growing only if
// needed). Graphs are first built with every symbol at its max, so that is the reservation.
#include "ggml_compiler_rt.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#include "ops.h"
#include "program.h"

using namespace ggml_compiler;

namespace {

thread_local std::string g_error;

void set_error(const std::string & msg) {
    g_error = msg;
}

void log_callback(ggml_log_level level, const char * text, void *) {
    static const bool verbose = [] {
        const char * v = std::getenv("GGML_COMPILER_LOG");
        return v && std::strcmp(v, "debug") == 0;
    }();
    if (verbose || level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) {
        std::fputs(text, stderr);
    }
}

// ggml logs at INFO from the first device query on (the CUDA banner, for one). Every entry point
// that may touch ggml installs the filter first.
void init_logging() {
    static const bool once = [] {
        ggml_log_set(log_callback, nullptr);
        return true;
    }();
    (void) once;
}

struct CallSite {
    const StageSpec *          stage = nullptr;
    int                        stage_index = 0;
    ggml_context *             ctx    = nullptr;
    ggml_cgraph *              gf     = nullptr;
    ggml_gallocr_t             galloc = nullptr;
    bool                       stale  = false;  // the binding changed: rebuild before the next call
};

struct StageStat {
    double ms    = 0.0;
    int    calls = 0;
};

// GGML_COMPILER_PROFILE=<file>: run every graph one node at a time, synchronizing after each, and append
// "stage<TAB>node<TAB>op<TAB>ms" lines to <file> after each ggml_compiler_run. Slow; for finding hot spots.
const char * profile_path() {
    static const char * p = std::getenv("GGML_COMPILER_PROFILE");
    return p && *p ? p : nullptr;
}

struct FileCloser {
    void operator()(FILE * f) const { std::fclose(f); }
};

ggml_backend_t init_backend(const ggml_compiler_options & o) {
    auto init_cpu = [&]() {
        ggml_backend_t b = ggml_backend_cpu_init();
        if (!b) {
            throw std::runtime_error("failed to initialize the CPU backend");
        }
        const int n = o.n_threads > 0 ? o.n_threads : (int) std::max(1u, std::thread::hardware_concurrency());
        ggml_backend_cpu_set_n_threads(b, n);
        return b;
    };
    if (o.backend == GGML_COMPILER_BACKEND_CPU) {
        return init_cpu();
    }

    std::vector<ggml_backend_dev_t> gpus;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        const std::string reg = ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));
        if (o.backend == GGML_COMPILER_BACKEND_CUDA && reg != "CUDA") {
            continue;
        }
        gpus.push_back(dev);
    }
    if (gpus.empty()) {
        if (o.backend == GGML_COMPILER_BACKEND_AUTO) {
            return init_cpu();
        }
        throw std::runtime_error("no CUDA device available (is the runtime built with -DGGML_CUDA=ON?)");
    }
    if (o.device < 0 || (size_t) o.device >= gpus.size()) {
        throw std::runtime_error("device index " + std::to_string(o.device) + " out of range; " +
                                 std::to_string(gpus.size()) + " GPU device(s) found");
    }
    ggml_backend_t b = ggml_backend_dev_init(gpus[o.device], nullptr);
    if (!b) {
        throw std::runtime_error(std::string("failed to initialize ") + ggml_backend_dev_name(gpus[o.device]));
    }
    return b;
}

}  // namespace

struct ggml_compiler_stage_build {
    ggml_context *              ctx = nullptr;
    bool                        gpu = false;
    ggml_backend_t              backend = nullptr;
    std::vector<ggml_tensor *>  params;
    std::vector<ggml_tensor *>  outputs;
    ggml_context *              wctx = nullptr;
    const std::vector<int64_t> * binding = nullptr;
    std::string                 error;
};

struct ggml_compiler_model {
    ProgramSpec           prog;
    std::unordered_map<std::string, ggml_compiler_build_fn> builders;
    ggml_backend_t        backend = nullptr;
    gguf_context *        gguf    = nullptr;
    ggml_context *        wctx    = nullptr;  // weights and constants
    ggml_backend_buffer_t wbuf    = nullptr;
    ggml_context *        vctx    = nullptr;  // driver values
    ggml_backend_buffer_t vbuf    = nullptr;
    ggml_context *        bctx    = nullptr;  // per binding: views of symbolic values (no data of their own)
    std::vector<ggml_context *> view_ctxs;  // per binding: row views of loop slice sources

    std::vector<int64_t> binding;  // one value per symbol; the next run uses it
    std::vector<int64_t> built;    // the binding the views and graphs are built for

    std::unordered_map<std::string, ggml_tensor *>      base;    // driver value storage
    std::unordered_map<std::string, ggml_tensor *>      values;  // driver values as the bound graphs see them
    std::unordered_map<std::string, const TensorSpec *> value_specs;
    struct Entry {
        const EntrySpec *               spec = nullptr;
        std::vector<const TensorSpec *> inputs, outputs;
    };
    std::vector<Entry>                    entries;    // the first is the default
    std::unordered_map<std::string, bool> input_set;  // by input name, over every entry
    std::vector<CallSite>                               calls;
    std::vector<StageStat>                              stage_stats;
    ggml_compiler_stats                                         last{};

    ~ggml_compiler_model() {
        for (auto & c : calls) {
            if (c.galloc) ggml_gallocr_free(c.galloc);
            if (c.ctx) ggml_free(c.ctx);
        }
        for (auto * c : view_ctxs) ggml_free(c);
        if (bctx) ggml_free(bctx);
        if (vbuf) ggml_backend_buffer_free(vbuf);
        if (vctx) ggml_free(vctx);
        if (wbuf) ggml_backend_buffer_free(wbuf);
        if (wctx) ggml_free(wctx);
        if (gguf) gguf_free(gguf);
        if (backend) ggml_backend_free(backend);
    }

    void load(const char * path, const ggml_compiler_options & opt);

    // Make the views and graphs match `binding` (no-op when they already do).
    void rebind();
    // Index of symbol `name`, or -1.
    int symbol_index(const char * name) const {
        for (size_t i = 0; i < prog.symbols.size(); i++) {
            if (prog.symbols[i].name == name) return (int) i;
        }
        return -1;
    }
    size_t nbytes_of(const TensorSpec & v, const std::vector<int64_t> & b) const {
        const auto ne = eval_ne(v, b);
        return ggml_row_size(v.type, ne[0]) * ne[1] * ne[2] * ne[3];
    }

    void run_ops(const std::vector<DriverOp> & ops);
    void write_profile();

  private:
    struct NodeTime {
        std::string stage, node, op;
        double      ms = 0.0;
    };
    std::vector<NodeTime> profile;

    void profile_call(CallSite & c) {
        using clock = std::chrono::steady_clock;
        ggml_init_params ip{};
        ip.mem_size     = ggml_graph_overhead_custom(4, false) + 1024;
        ip.no_alloc     = true;
        ggml_context * pc = ggml_init(ip);
        ggml_cgraph *  one = ggml_new_graph_custom(pc, 4, false);
        for (int i = 0; i < ggml_graph_n_nodes(c.gf); i++) {
            ggml_tensor * t = ggml_graph_node(c.gf, i);
            ggml_graph_clear(one);
            ggml_graph_add_node(one, t);
            const auto t0 = clock::now();
            if (ggml_backend_graph_compute(backend, one) != GGML_STATUS_SUCCESS) {
                ggml_free(pc);
                throw std::runtime_error("profiling: node failed");
            }
            ggml_backend_synchronize(backend);
            profile.push_back({c.stage->name, t->name, ggml_op_desc(t),
                               std::chrono::duration<double, std::milli>(clock::now() - t0).count()});
        }
        ggml_free(pc);
    }

    void load_weights(const char * path);
    void alloc_values();
    void bind_values();
    void build_ops(std::vector<DriverOp> & ops);
    void build_call(DriverOp & d);
    void build_site(const DriverOp & d);
};

void ggml_compiler_model::load(const char * path, const ggml_compiler_options & opt) {
    if (opt.allow_cpu_fallback) {
        throw std::runtime_error("allow_cpu_fallback is not implemented yet");
    }

    gguf_init_params gp{};
    gp.no_alloc = true;
    gp.ctx      = &wctx;
    gguf        = gguf_init_from_file(path, gp);
    if (!gguf) {
        throw std::runtime_error(std::string("cannot read GGUF '") + path + "'");
    }
    const int64_t key = gguf_find_key(gguf, "ggml_compiler.program");
    if (key < 0) {
        throw std::runtime_error(std::string("'") + path +
                                 "' has no ggml_compiler.program; it was not written by ggml-compiler compile");
    }
    prog = parse_program(gguf_get_val_str(gguf, key));

    backend = init_backend(opt);
    load_weights(path);
    for (const auto & sy : prog.symbols) binding.push_back(sy.max);
    prog.bind(binding);
    alloc_values();
    for (const auto & s : prog.stages) {
        (void) s;
        stage_stats.emplace_back();
    }
    bind_values();
    for (auto & e : prog.entries) build_ops(e.driver);
    built = binding;
}

void ggml_compiler_model::rebind() {
    if (binding == built) return;
    prog.bind(binding);
    for (auto * c : view_ctxs) ggml_free(c);
    view_ctxs.clear();
    bind_values();
    for (auto & e : prog.entries) build_ops(e.driver);
    built = binding;
}

void ggml_compiler_model::build_ops(std::vector<DriverOp> & ops) {
    for (auto & d : ops) {
        if (d.kind == DriverOp::Kind::Call) {
            build_call(d);
            continue;
        }
        auto check_value = [&](const std::string & name) {
            if (!values.count(name)) throw std::runtime_error("loop uses undeclared value '" + name + "'");
            return values.at(name);
        };
        for (const auto & [dst, src] : d.carry) {
            ggml_tensor * a = check_value(dst);
            ggml_tensor * b = check_value(src);
            if (a->type != b->type || !ggml_are_same_shape(a, b)) {
                throw std::runtime_error("loop carry '" + dst + "' does not match its initial value '" + src + "'");
            }
        }
        for (const auto & sc : d.scalars) {
            if (check_value(sc.first)->type != GGML_TYPE_F32) {
                throw std::runtime_error("loop scalar '" + sc.first + "' must be f32");
            }
        }
        if (!d.slices.empty()) {
            ggml_init_params ip = { ggml_tensor_overhead() * d.slices.size() * d.n, nullptr, true };
            ggml_context *   vc = ggml_init(ip);
            view_ctxs.push_back(vc);
            d.slice_rows.clear();
            for (const auto & [dst, src] : d.slices) {
                ggml_tensor * a = check_value(dst);
                ggml_tensor * b = check_value(src);
                // The source is [n, ...dst] in PyTorch order: dst's elements, n times, outermost.
                if (a->type != b->type || ggml_nelements(b) != ggml_nelements(a) * d.n || !ggml_is_contiguous(b)) {
                    throw std::runtime_error("loop slice '" + dst + "' does not match its source '" + src + "'");
                }
                std::vector<ggml_tensor *> rows;
                for (int i = 0; i < d.n; i++) {
                    ggml_tensor * v = ggml_view_4d(vc, b, a->ne[0], a->ne[1], a->ne[2], a->ne[3], a->nb[1], a->nb[2],
                                                   a->nb[3], (size_t) i * ggml_nbytes(a));
                    if (ggml_backend_view_init(v) != GGML_STATUS_SUCCESS) {
                        throw std::runtime_error("cannot view loop slice source '" + src + "'");
                    }
                    rows.push_back(v);
                }
                d.slice_rows.push_back(std::move(rows));
            }
        }
        build_ops(d.body);
    }
}

void ggml_compiler_model::write_profile() {
    if (!profile_path() || profile.empty()) {
        return;
    }
    if (FILE * f = std::fopen(profile_path(), "a")) {
        for (const auto & p : profile) {
            std::fprintf(f, "%s\t%s\t%s\t%.6f\n", p.stage.c_str(), p.node.c_str(), p.op.c_str(), p.ms);
        }
        std::fclose(f);
    }
    profile.clear();
}

void ggml_compiler_model::run_ops(const std::vector<DriverOp> & ops) {
    using clock = std::chrono::steady_clock;
    for (const auto & d : ops) {
        if (d.kind == DriverOp::Kind::Call) {
            if (calls[d.call_site].stale) build_site(d);
            CallSite &  c      = calls[d.call_site];
            const auto  ts     = clock::now();
            if (profile_path()) {
                profile_call(c);
            }
            ggml_status status = profile_path() ? GGML_STATUS_SUCCESS : ggml_backend_graph_compute(backend, c.gf);
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("stage '" + c.stage->name + "' failed: " + ggml_status_to_string(status));
            }
            auto & st = stage_stats[c.stage_index];
            st.ms += std::chrono::duration<double, std::milli>(clock::now() - ts).count();
            st.calls++;
            last.n_calls++;
            continue;
        }
        for (const auto & [dst, src] : d.carry) {
            ggml_backend_tensor_copy(values.at(src), values.at(dst));
        }
        const int            n = d.n_symbol >= 0 ? (int) binding[d.n_symbol] : d.n;
        std::vector<int32_t> starts;
        for (const auto & [name, src] : d.counters) {
            int32_t v = 0;
            ggml_backend_tensor_get(values.at(src), &v, 0, sizeof v);
            starts.push_back(v);
        }
        for (const auto & [dst, src] : d.collect) {
            ggml_tensor * t = base.at(dst);
            ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));
        }
        std::vector<float>   fill;
        std::vector<int64_t> ext;
        std::vector<char>    row;
        for (int i = 0; i < n; i++) {
            for (const auto & [name, table] : d.scalars) {
                ggml_tensor * t = values.at(name);
                fill.assign(ggml_nelements(t), table[i]);
                ggml_backend_tensor_set(t, fill.data(), 0, ggml_nbytes(t));
            }
            for (size_t k = 0; k < d.slices.size(); k++) {
                ggml_backend_tensor_copy(d.slice_rows[k][i], values.at(d.slices[k].first));
            }
            for (size_t k = 0; k < d.counters.size(); k++) {
                const int32_t v = starts[k] + i;
                ggml_backend_tensor_set(values.at(d.counters[k].first), &v, 0, sizeof v);
            }
            if (!d.bind.empty()) {
                ext = binding;
                for (int32_t v : starts) ext.push_back(v + i);
                for (const auto & [k, e] : d.bind) {
                    const int64_t v = e.eval(ext);
                    if (v < prog.symbols[k].min || v > prog.symbols[k].max) {
                        throw std::runtime_error("loop binds " + prog.symbols[k].name + " = " + std::to_string(v) +
                                                 ", outside its range");
                    }
                    binding[k] = v;
                }
                rebind();  // a no-op unless a symbol changed
            }
            for (const auto & [name, k] : d.masks) {
                const TensorSpec & v   = *value_specs.at(name);
                const int64_t      pos = starts[k] + i;
                const int64_t      len = v.ne[0];
                fill.resize((size_t) (v.ne[0] * v.ne[1] * v.ne[2] * v.ne[3]));
                for (size_t j = 0; j < fill.size(); j++) fill[j] = (int64_t) (j % len) <= pos ? 0.0f : -INFINITY;
                ggml_backend_tensor_set(base.at(name), fill.data(), 0, fill.size() * sizeof(float));
            }
            run_ops(d.body);
            for (const auto & [dst, src] : d.collect) {
                const size_t nb = nbytes_of(*value_specs.at(src), binding);
                row.resize(nb);
                ggml_backend_tensor_get(base.at(src), row.data(), 0, nb);
                ggml_backend_tensor_set(base.at(dst), row.data(), (size_t) i * nb, nb);
            }
            bool stop = false;
            for (const auto & [name, ids] : d.until) {
                int32_t v = 0;
                ggml_backend_tensor_get(base.at(name), &v, 0, sizeof v);
                stop = stop || std::find(ids.begin(), ids.end(), v) != ids.end();
            }
            if (stop) break;
        }
    }
}

void ggml_compiler_model::load_weights(const char * path) {
    if (ggml_get_first_tensor(wctx) == nullptr) {
        return;
    }
    wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
    if (!wbuf) {
        throw std::runtime_error("cannot allocate weight buffer");
    }
    std::unique_ptr<FILE, FileCloser> f(std::fopen(path, "rb"));
    if (!f) {
        throw std::runtime_error(std::string("cannot open '") + path + "'");
    }
    const size_t         data_off = gguf_get_data_offset(gguf);
    const bool           host     = ggml_backend_buffer_is_host(wbuf);
    std::vector<uint8_t> staging;
    for (int64_t i = 0; i < gguf_get_n_tensors(gguf); i++) {
        const char *  name = gguf_get_tensor_name(gguf, i);
        ggml_tensor * t    = ggml_get_tensor(wctx, name);
        const size_t  n    = ggml_nbytes(t);
        if (std::fseek(f.get(), (long) (data_off + gguf_get_tensor_offset(gguf, i)), SEEK_SET) != 0) {
            throw std::runtime_error(std::string("seek failed for tensor ") + name);
        }
        void * dst = t->data;
        if (!host) {
            staging.resize(n);
            dst = staging.data();
        }
        if (std::fread(dst, 1, n, f.get()) != n) {
            throw std::runtime_error(std::string("short read for tensor ") + name);
        }
        if (!host) {
            ggml_backend_tensor_set(t, staging.data(), 0, n);
        }
    }
}

static bool symbolic(const TensorSpec & v) {
    for (const auto & d : v.ne_x) {
        if (!d.is_const()) return true;
    }
    return false;
}

void ggml_compiler_model::alloc_values() {
    ggml_init_params ip{};
    ip.mem_size   = ggml_tensor_overhead() * (prog.values.size() + 1);
    ip.no_alloc   = true;
    vctx          = ggml_init(ip);
    for (const auto & v : prog.values) {
        ggml_tensor * t = nullptr;
        if (!symbolic(v)) {
            t = ggml_new_tensor_4d(vctx, v.type, v.ne[0], v.ne[1], v.ne[2], v.ne[3]);
        } else {
            // Room for the largest size over the corners of its symbols' ranges (shapes are
            // monotone in each symbol, so a corner holds the maximum).
            std::vector<bool> used(prog.symbols.size(), false);
            for (const auto & d : v.ne_x) d.symbols(used);
            std::vector<size_t> idx;
            for (size_t k = 0; k < used.size(); k++) {
                if (used[k]) idx.push_back(k);
            }
            int64_t most = 0;
            std::vector<int64_t> b = binding;
            for (size_t mask = 0; mask < ((size_t) 1 << idx.size()); mask++) {
                for (size_t k = 0; k < idx.size(); k++) {
                    b[idx[k]] = (mask >> k) & 1 ? prog.symbols[idx[k]].max : prog.symbols[idx[k]].min;
                }
                const auto ne = eval_ne(v, b);
                most = std::max(most, ne[0] * ne[1] * ne[2] * ne[3]);
            }
            t = ggml_new_tensor_1d(vctx, v.type, most);
        }
        ggml_set_name(t, v.name.c_str());
        base[v.name]        = t;
        values[v.name]      = t;
        value_specs[v.name] = &v;
    }
    if (!prog.values.empty()) {
        vbuf = ggml_backend_alloc_ctx_tensors(vctx, backend);
        if (!vbuf) {
            throw std::runtime_error("cannot allocate driver value buffer");
        }
    }
    if (vbuf) ggml_backend_buffer_clear(vbuf, 0);  // state starts as zeros (and no stale NaNs anywhere)
    for (const auto & e : prog.entries) {
        Entry en;
        en.spec = &e;
        for (const auto & name : e.inputs) {
            if (!value_specs.count(name)) throw std::runtime_error("input '" + name + "' is not a declared value");
            en.inputs.push_back(value_specs[name]);
            input_set[name] = false;
        }
        for (const auto & name : e.outputs) {
            if (!value_specs.count(name)) throw std::runtime_error("output '" + name + "' is not a declared value");
            en.outputs.push_back(value_specs[name]);
        }
        entries.push_back(std::move(en));
    }
    for (const auto & name : prog.state) {
        if (!value_specs.count(name)) throw std::runtime_error("state '" + name + "' is not a declared value");
    }
}

void ggml_compiler_model::bind_values() {
    if (bctx) {
        ggml_free(bctx);
        bctx = nullptr;
    }
    size_t n = 0;
    for (const auto & v : prog.values) n += symbolic(v);
    if (n == 0) return;
    ggml_init_params ip{ggml_tensor_overhead() * (n + 1), nullptr, true};
    bctx = ggml_init(ip);
    for (const auto & v : prog.values) {
        if (!symbolic(v)) continue;
        const size_t  nb1 = ggml_row_size(v.type, v.ne[0]), nb2 = nb1 * v.ne[1], nb3 = nb2 * v.ne[2];
        ggml_tensor * t   = ggml_view_4d(bctx, base.at(v.name), v.ne[0], v.ne[1], v.ne[2], v.ne[3], nb1, nb2, nb3, 0);
        ggml_set_name(t, v.name.c_str());
        if (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("cannot view driver value '" + v.name + "'");
        }
        values[v.name] = t;
    }
}

void ggml_compiler_model::build_call(DriverOp & d) {
    if (d.call_site >= 0) {
        // Rebinding: a stage with only static shapes keeps its graph. Any other is rebuilt, since
        // the views of symbolic values it reads were rebuilt too.
        // The graph is rebuilt when the site next runs, so an entry point never rebuilds the
        // stages of another.
        const StageSpec & st = *calls[d.call_site].stage;
        if (std::find(st.uses.begin(), st.uses.end(), true) == st.uses.end()) return;
        CallSite & old = calls[d.call_site];
        ggml_free(old.ctx);
        old.ctx   = nullptr;
        old.gf    = nullptr;
        old.stale = true;
        return;
    } else {
        CallSite c;
        for (size_t i = 0; i < prog.stages.size(); i++) {
            if (prog.stages[i].name == d.stage) {
                c.stage       = &prog.stages[i];
                c.stage_index = (int) i;
            }
        }
        if (!c.stage) {
            throw std::runtime_error("driver calls unknown stage '" + d.stage + "'");
        }
        d.call_site = (int) calls.size();
        calls.push_back(c);  // owned by the model from here on, so a throw below cannot leak it
    }
    build_site(d);
}

void ggml_compiler_model::build_site(const DriverOp & d) {
    const StageSpec & s = *calls[d.call_site].stage;
    calls[d.call_site].stale = false;
    if (d.args.size() != s.params.size() || d.out.size() != s.outputs.size()) {
        throw std::runtime_error("call to stage '" + s.name + "' has the wrong number of arguments or results");
    }

    auto value = [&](const std::string & name) {
        auto it = values.find(name);
        if (it == values.end()) throw std::runtime_error("driver value '" + name + "' is not declared");
        return it->second;
    };

    // One IR node may expand to several ggml tensors (conv_2d: im2col, reshapes, matmul, permute,
    // cont, bias add), so size for kMaxExpansion per node. Leaves: parameters and every weight the
    // stage might read. Running out would make ggml_new_tensor return NULL in release builds.
    constexpr size_t kMaxExpansion = 8;
    const size_t     n_tensors      = kMaxExpansion * s.nodes.size() + 2 * s.outputs.size() + 16;
    const size_t     graph_size     = n_tensors + s.params.size() + (size_t) gguf_get_n_tensors(gguf) + 64;
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * n_tensors + ggml_graph_overhead_custom(graph_size, false);
    ip.no_alloc = true;
    CallSite & cs = calls[d.call_site];
    cs.ctx        = ggml_init(ip);

    const bool gpu = ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_GPU;
    std::unordered_map<std::string, ggml_tensor *> env;
    for (size_t i = 0; i < s.params.size(); i++) {
        ggml_tensor *      arg = value(d.args[i]);
        const TensorSpec & p   = s.params[i];
        for (int k = 0; k < 4; k++) {
            if (arg->ne[k] != p.ne[k] || arg->type != p.type) {
                throw std::runtime_error("stage '" + s.name + "' parameter " + p.name + " does not match value '" +
                                         d.args[i] + "'");
            }
        }
        env[p.name] = arg;
    }

    std::vector<ggml_tensor *> results;
    auto builder = builders.find(s.name);
    if (builder != builders.end()) {
        // A compiled builder makes the graph; check what it returns against the program.
        ggml_compiler_stage_build b;
        b.ctx     = cs.ctx;
        b.gpu     = gpu;
        b.backend = backend;
        b.wctx    = wctx;
        b.binding = &binding;
        for (size_t i = 0; i < s.params.size(); i++) b.params.push_back(env.at(s.params[i].name));
        b.outputs.assign(s.outputs.size(), nullptr);
        if (builder->second(&b) != 0 || !b.error.empty()) {
            throw std::runtime_error("builder for stage '" + s.name + "' failed: " + b.error);
        }
        std::unordered_map<std::string, const NodeSpec *> spec;
        for (const auto & n : s.nodes) spec[n.id] = &n;
        for (size_t i = 0; i < s.outputs.size(); i++) {
            ggml_tensor * t = b.outputs[i];
            if (!t) throw std::runtime_error("builder for stage '" + s.name + "' left output " + std::to_string(i) + " unset");
            auto it = spec.find(s.outputs[i]);
            if (it != spec.end() && (t->type != it->second->type || t->ne[0] != it->second->ne[0] ||
                                     t->ne[1] != it->second->ne[1] || t->ne[2] != it->second->ne[2] ||
                                     t->ne[3] != it->second->ne[3])) {
                throw std::runtime_error("builder for stage '" + s.name + "' output " + std::to_string(i) +
                                         " does not match the program");
            }
            results.push_back(t);
        }
    } else {
        std::vector<ggml_tensor *> src;
        for (const auto & n : s.nodes) {
            src.clear();
            for (const auto & name : n.src) {
                auto it = env.find(name);
                if (it != env.end()) {
                    src.push_back(it->second);
                    continue;
                }
                ggml_tensor * w = wctx ? ggml_get_tensor(wctx, name.c_str()) : nullptr;
                if (!w) {
                    throw std::runtime_error("stage '" + s.name + "' node " + n.id + " reads unknown tensor '" + name + "'");
                }
                src.push_back(w);
            }
            env[n.id] = emit_node(cs.ctx, n, src, backend);
        }
        for (const auto & id : s.outputs) {
            auto it = env.find(id);
            if (it == env.end()) throw std::runtime_error("stage '" + s.name + "' returns unknown value '" + id + "'");
            results.push_back(it->second);
        }
    }

    cs.gf = ggml_new_graph_custom(cs.ctx, graph_size, false);
    for (ggml_tensor * t : results) {
        ggml_build_forward_expand(cs.gf, t);
    }
    // Copies into driver values go last, after every read of the stage's arguments.
    for (size_t i = 0; i < results.size(); i++) {
        ggml_tensor * dst = value(d.out[i]);
        ggml_tensor * res = results[i];
        if (res == dst) continue;
        // Written in place: set_rows into a state value the stage read (a KV cache) returns a view
        // of that value's own memory. Nothing to copy.
        if (res->data && res->data == dst->data && res->type == dst->type && ggml_are_same_shape(res, dst) &&
            ggml_is_contiguous(res)) {
            continue;
        }
        if (ggml_nelements(res) != ggml_nelements(dst) || res->type != dst->type) {
            throw std::runtime_error("stage '" + s.name + "' result " + std::to_string(i) + " does not match value '" +
                                     d.out[i] + "'");
        }
        ggml_build_forward_expand(cs.gf, ggml_cpy(cs.ctx, res, dst));
    }

    std::string unsupported;
    for (int i = 0; i < ggml_graph_n_nodes(cs.gf); i++) {
        ggml_tensor * t = ggml_graph_node(cs.gf, i);
        if (!ggml_backend_supports_op(backend, t)) {
            unsupported += std::string("\n  ") + t->name + " (" + ggml_op_desc(t) + ")";
        }
    }
    if (!unsupported.empty()) {
        throw std::runtime_error(std::string("backend ") + ggml_backend_name(backend) + " cannot run these nodes of stage '" +
                                 s.name + "':" + unsupported);
    }

    if (!cs.galloc) cs.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(cs.galloc, cs.gf)) {
        throw std::runtime_error("cannot allocate compute buffer for stage '" + s.name + "'");
    }
}

// ---------------------------------------------------------------------------------------------
// C API

namespace {

int fill_info(const ggml_compiler_model * m, const TensorSpec & v, const std::vector<int64_t> & b,
              ggml_compiler_tensor_info * info) {
    if (!info) {
        set_error("info is NULL");
        return -1;
    }
    const auto shape = eval_shape(v, b);
    switch (v.type) {
        case GGML_TYPE_F32: info->dtype = GGML_COMPILER_DTYPE_F32; break;
        case GGML_TYPE_F16: info->dtype = GGML_COMPILER_DTYPE_F16; break;
        case GGML_TYPE_I32: info->dtype = GGML_COMPILER_DTYPE_I32; break;
        default: set_error("value '" + v.name + "' has a type the C API cannot describe"); return -1;
    }
    if (shape.size() > GGML_COMPILER_MAX_RANK) {
        set_error("value '" + v.name + "' has rank above GGML_COMPILER_MAX_RANK");
        return -1;
    }
    info->name = v.name.c_str();
    info->ndim = (int) shape.size();
    for (size_t i = 0; i < shape.size(); i++) {
        info->shape[i] = shape[i];
    }
    info->nbytes = m->nbytes_of(v, b);
    return 0;
}

// Solve the symbols an input's shape fixes: a dim with one unknown symbol is solved by search
// over the symbol's range (dims are monotone in it), then every dim is checked.
void solve_shape(ggml_compiler_model * m, const TensorSpec & v, const std::vector<int64_t> & shape) {
    if (shape.size() != v.shape_x.size()) {
        throw std::runtime_error("input '" + v.name + "' has rank " + std::to_string(v.shape_x.size()) + ", got " +
                                 std::to_string(shape.size()));
    }
    std::vector<int64_t> b = m->binding;
    std::vector<bool>    known(b.size(), false);
    for (int pass = 0; pass < 2; pass++) {
        for (size_t i = 0; i < shape.size(); i++) {
            std::vector<bool> used(b.size(), false);
            v.shape_x[i].symbols(used);
            int unknown = -1, n_unknown = 0;
            for (size_t k = 0; k < used.size(); k++) {
                if (used[k] && !known[k]) {
                    unknown = (int) k;
                    n_unknown++;
                }
            }
            if (n_unknown != 1) continue;
            const Symbol & sy = m->prog.symbols[unknown];
            int64_t lo = sy.min, hi = sy.max;
            while (lo < hi) {  // smallest value whose dim reaches shape[i]
                const int64_t mid = lo + (hi - lo) / 2;
                b[unknown]        = mid;
                if (v.shape_x[i].eval(b) < shape[i]) lo = mid + 1; else hi = mid;
            }
            b[unknown] = lo;
            if (v.shape_x[i].eval(b) != shape[i]) {
                throw std::runtime_error("input '" + v.name + "': no " + sy.name + " in [" + std::to_string(sy.min) + ", " +
                                         std::to_string(sy.max) + "] gives dim " + std::to_string(i) + " = " +
                                         std::to_string(shape[i]));
            }
            known[unknown] = true;
        }
    }
    const auto got = eval_shape(v, b);
    if (got != shape) {
        std::string want, have;
        for (auto d : shape) want += (want.empty() ? "" : ", ") + std::to_string(d);
        for (auto d : got) have += (have.empty() ? "" : ", ") + std::to_string(d);
        throw std::runtime_error("input '" + v.name + "' cannot have shape [" + want + "] (with the other symbols as bound, it is [" + have + "])");
    }
    m->binding = b;
}

template <typename F> auto guarded(F && f, decltype(f()) on_error) -> decltype(f()) {
    try {
        return f();
    } catch (const std::exception & e) {
        set_error(e.what());
    } catch (...) {
        set_error("unknown error");
    }
    return on_error;
}

}  // namespace

extern "C" {

const char * ggml_compiler_version(void) {
    return "ggml_compiler_rt " GGML_COMPILER_VERSION " (ggml " GGML_COMPILER_LLAMA_TAG ")";
}

const char * ggml_compiler_last_error(void) {
    return g_error.c_str();
}

bool ggml_compiler_has_backend(ggml_compiler_backend backend) {
    init_logging();
    if (backend == GGML_COMPILER_BACKEND_CPU || backend == GGML_COMPILER_BACKEND_AUTO) {
        return true;
    }
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU &&
            std::string(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev))) == "CUDA") {
            return true;
        }
    }
    return false;
}

size_t ggml_compiler_row_size(const char * type, int64_t n) {
    try {
        return ggml_row_size(parse_type(type), n);
    } catch (const std::exception & e) {
        set_error(e.what());
        return 0;
    }
}

size_t ggml_compiler_quantize(const char * type, const float * src, int64_t nrows, int64_t n_per_row, void * dst,
                              size_t dst_size) {
    try {
        const ggml_type t = parse_type(type);
        if (n_per_row % ggml_blck_size(t) != 0) {
            throw std::runtime_error(std::string("row length ") + std::to_string(n_per_row) + " is not a multiple of " +
                                     type + "'s block size " + std::to_string(ggml_blck_size(t)));
        }
        if (dst_size < ggml_row_size(t, n_per_row) * nrows) {
            throw std::runtime_error("destination buffer too small");
        }
        ggml_quantize_init(t);
        return ggml_quantize_chunk(t, src, dst, 0, nrows, n_per_row, nullptr);
    } catch (const std::exception & e) {
        set_error(e.what());
        return 0;
    }
}

ggml_compiler_options ggml_compiler_default_options(void) {
    ggml_compiler_options o{};
    o.backend            = GGML_COMPILER_BACKEND_AUTO;
    o.device             = 0;
    o.n_threads          = 0;
    o.allow_cpu_fallback = false;
    return o;
}

ggml_compiler_model * ggml_compiler_load(const char * path, const ggml_compiler_options * opt) {
    return ggml_compiler_load_with(path, opt, nullptr, 0);
}

ggml_compiler_model * ggml_compiler_load_with(const char * path, const ggml_compiler_options * opt,
                                              const ggml_compiler_builder * builders, int n_builders) {
    init_logging();
    const ggml_compiler_options o = opt ? *opt : ggml_compiler_default_options();
    return guarded(
        [&]() -> ggml_compiler_model * {
            if (!path) throw std::runtime_error("path is NULL");
            auto m = std::make_unique<ggml_compiler_model>();
            for (int i = 0; i < n_builders; i++) m->builders[builders[i].stage] = builders[i].build;
            m->load(path, o);
            for (const auto & [name, fn] : m->builders) {
                bool found = false;
                for (const auto & st : m->prog.stages) found = found || st.name == name;
                if (!found) throw std::runtime_error("builder for stage '" + name + "', which the program does not have");
            }
            return m.release();
        },
        nullptr);
}

ggml_context * ggml_compiler_build_ctx(ggml_compiler_stage_build * b) {
    return b->ctx;
}

bool ggml_compiler_build_gpu(ggml_compiler_stage_build * b) {
    return b->gpu;
}

bool ggml_compiler_build_supports(ggml_compiler_stage_build * b, const ggml_tensor * t) {
    return !b->backend || ggml_backend_supports_op(b->backend, t);
}

ggml_tensor * ggml_compiler_build_param(ggml_compiler_stage_build * b, int i) {
    if (i < 0 || i >= (int) b->params.size()) {
        b->error = "parameter index " + std::to_string(i) + " out of range";
        return nullptr;
    }
    return b->params[i];
}

ggml_tensor * ggml_compiler_build_weight(ggml_compiler_stage_build * b, const char * name) {
    ggml_tensor * t = b->wctx ? ggml_get_tensor(b->wctx, name) : nullptr;
    if (!t) b->error = std::string("no tensor '") + name + "' in the GGUF";
    return t;
}

int64_t ggml_compiler_build_symbol(ggml_compiler_stage_build * b, int i) {
    if (!b->binding || i < 0 || i >= (int) b->binding->size()) {
        b->error = "symbol index " + std::to_string(i) + " out of range";
        return 1;
    }
    return (*b->binding)[i];
}

void ggml_compiler_build_output(ggml_compiler_stage_build * b, int i, ggml_tensor * t) {
    if (i < 0 || i >= (int) b->outputs.size()) {
        b->error = "output index " + std::to_string(i) + " out of range";
        return;
    }
    b->outputs[i] = t;
}

const char * ggml_compiler_metadata(const ggml_compiler_model * m, const char * key) {
    if (!m || !key) return nullptr;
    const int64_t k = gguf_find_key(m->gguf, key);
    if (k < 0 || gguf_get_kv_type(m->gguf, k) != GGUF_TYPE_STRING) return nullptr;
    return gguf_get_val_str(m->gguf, k);
}

void ggml_compiler_free(ggml_compiler_model * m) {
    delete m;
}

const char * ggml_compiler_backend_name(const ggml_compiler_model * m) {
    return m ? ggml_backend_name(m->backend) : nullptr;
}

int ggml_compiler_n_entries(const ggml_compiler_model * m) {
    return m ? (int) m->entries.size() : -1;
}

const char * ggml_compiler_entry_name(const ggml_compiler_model * m, int e) {
    if (!m || e < 0 || e >= (int) m->entries.size()) {
        set_error("entry index out of range");
        return nullptr;
    }
    return m->entries[e].spec->name.c_str();
}

int ggml_compiler_entry_n_inputs(const ggml_compiler_model * m, int e) {
    return m && e >= 0 && e < (int) m->entries.size() ? (int) m->entries[e].inputs.size() : -1;
}

int ggml_compiler_entry_n_outputs(const ggml_compiler_model * m, int e) {
    return m && e >= 0 && e < (int) m->entries.size() ? (int) m->entries[e].outputs.size() : -1;
}

int ggml_compiler_entry_input_info(const ggml_compiler_model * m, int e, int i, ggml_compiler_tensor_info * info) {
    if (!m || e < 0 || e >= (int) m->entries.size() || i < 0 || i >= (int) m->entries[e].inputs.size()) {
        set_error("input index out of range");
        return -1;
    }
    return fill_info(m, *m->entries[e].inputs[i], m->binding, info);
}

int ggml_compiler_entry_output_info(const ggml_compiler_model * m, int e, int i, ggml_compiler_tensor_info * info) {
    if (!m || e < 0 || e >= (int) m->entries.size() || i < 0 || i >= (int) m->entries[e].outputs.size()) {
        set_error("output index out of range");
        return -1;
    }
    return fill_info(m, *m->entries[e].outputs[i], m->binding, info);
}

int ggml_compiler_n_inputs(const ggml_compiler_model * m) {
    return ggml_compiler_entry_n_inputs(m, 0);
}

int ggml_compiler_n_outputs(const ggml_compiler_model * m) {
    return ggml_compiler_entry_n_outputs(m, 0);
}

int ggml_compiler_input_info(const ggml_compiler_model * m, int i, ggml_compiler_tensor_info * info) {
    return ggml_compiler_entry_input_info(m, 0, i, info);
}

int ggml_compiler_output_info(const ggml_compiler_model * m, int i, ggml_compiler_tensor_info * info) {
    return ggml_compiler_entry_output_info(m, 0, i, info);
}

int ggml_compiler_reset(ggml_compiler_model * m) {
    return guarded(
        [&]() {
            if (!m) throw std::runtime_error("model is NULL");
            for (const auto & name : m->prog.state) {
                ggml_tensor * t = m->base.at(name);
                ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));
            }
            return 0;
        },
        -1);
}

int ggml_compiler_n_symbols(const ggml_compiler_model * m) {
    return m ? (int) m->prog.symbols.size() : -1;
}

int ggml_compiler_symbol_info(const ggml_compiler_model * m, int i, const char ** name, int64_t * min, int64_t * max,
                              int64_t * value) {
    if (!m || i < 0 || i >= (int) m->prog.symbols.size()) {
        set_error("symbol index out of range");
        return -1;
    }
    const Symbol & sy = m->prog.symbols[i];
    if (name) *name = sy.name.c_str();
    if (min) *min = sy.min;
    if (max) *max = sy.max;
    if (value) *value = m->binding[i];
    return 0;
}

int ggml_compiler_set_symbol(ggml_compiler_model * m, const char * name, int64_t value) {
    return guarded(
        [&]() {
            if (!m || !name) throw std::runtime_error("NULL argument");
            const int i = m->symbol_index(name);
            if (i < 0) throw std::runtime_error("no symbol named '" + std::string(name) + "'");
            const Symbol & sy = m->prog.symbols[i];
            if (value < sy.min || value > sy.max) {
                throw std::runtime_error("symbol '" + sy.name + "' = " + std::to_string(value) + " is outside [" +
                                         std::to_string(sy.min) + ", " + std::to_string(sy.max) + "]");
            }
            m->binding[i] = value;
            return 0;
        },
        -1);
}

int ggml_compiler_set_input_shape(ggml_compiler_model * m, const char * name, int ndim, const int64_t * shape) {
    return guarded(
        [&]() {
            if (!m || !name || (ndim > 0 && !shape)) throw std::runtime_error("NULL argument");
            if (!m->input_set.count(name)) throw std::runtime_error("no input named '" + std::string(name) + "'");
            solve_shape(m, *m->value_specs.at(name), std::vector<int64_t>(shape, shape + ndim));
            return 0;
        },
        -1);
}

int ggml_compiler_set_input(ggml_compiler_model * m, const char * name, const void * data, size_t nbytes) {
    return guarded(
        [&]() {
            if (!m || !name || !data) throw std::runtime_error("NULL argument");
            if (!m->input_set.count(name)) throw std::runtime_error("no input named '" + std::string(name) + "'");
            const size_t want = m->nbytes_of(*m->value_specs.at(name), m->binding);
            if (nbytes != want) {
                throw std::runtime_error("input '" + std::string(name) + "' needs " + std::to_string(want) +
                                         " bytes, got " + std::to_string(nbytes));
            }
            ggml_backend_tensor_set(m->base.at(name), data, 0, nbytes);
            m->input_set[name] = true;
            return 0;
        },
        -1);
}

int ggml_compiler_run(ggml_compiler_model * m) {
    return m ? ggml_compiler_run_entry(m, m->entries[0].spec->name.c_str()) : (set_error("model is NULL"), -1);
}

int ggml_compiler_run_entry(ggml_compiler_model * m, const char * entry) {
    return guarded(
        [&]() {
            if (!m || !entry) throw std::runtime_error("NULL argument");
            const ggml_compiler_model::Entry * e = nullptr;
            for (const auto & en : m->entries) {
                if (en.spec->name == entry) e = &en;
            }
            if (!e) throw std::runtime_error("no entry point named '" + std::string(entry) + "'");
            for (const TensorSpec * v : e->inputs) {
                if (!m->input_set.at(v->name)) throw std::runtime_error("input '" + v->name + "' was never set");
            }
            m->rebind();
            for (auto & st : m->stage_stats) st = StageStat{};
            m->last.n_calls = 0;
            using clock     = std::chrono::steady_clock;
            const auto t0   = clock::now();
            m->run_ops(e->spec->driver);
            m->last.ms_total = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
            m->write_profile();
            return 0;
        },
        -1);
}

int ggml_compiler_get_output(ggml_compiler_model * m, const char * name, void * dst, size_t nbytes) {
    return guarded(
        [&]() {
            if (!m || !name || !dst) throw std::runtime_error("NULL argument");
            for (const auto & e : m->entries) {
                for (const TensorSpec * v : e.outputs) {
                    if (v->name != name) continue;
                    const size_t have = m->nbytes_of(*v, m->built);
                    if (nbytes != have) {
                        throw std::runtime_error("output '" + std::string(name) + "' has " + std::to_string(have) +
                                                 " bytes, buffer has " + std::to_string(nbytes));
                    }
                    ggml_backend_tensor_get(m->base.at(name), dst, 0, nbytes);
                    return 0;
                }
            }
            throw std::runtime_error("no output named '" + std::string(name) + "'");
        },
        -1);
}

int ggml_compiler_last_stats(const ggml_compiler_model * m, ggml_compiler_stats * st) {
    if (!m || !st) {
        set_error("NULL argument");
        return -1;
    }
    *st = m->last;
    return 0;
}

int ggml_compiler_n_stages(const ggml_compiler_model * m) {
    return m ? (int) m->prog.stages.size() : -1;
}

int ggml_compiler_stage_stats(const ggml_compiler_model * m, int i, const char ** name, double * ms, int * n_calls) {
    if (!m || i < 0 || i >= (int) m->prog.stages.size()) {
        set_error("stage index out of range");
        return -1;
    }
    if (name) *name = m->prog.stages[i].name.c_str();
    if (ms) *ms = m->stage_stats[i].ms;
    if (n_calls) *n_calls = m->stage_stats[i].calls;
    return 0;
}

}  // extern "C"
