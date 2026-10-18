#include "program.h"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <stdexcept>

using nlohmann::json;

namespace ggmlc {

ggml_type parse_type(const std::string & name) {
    for (int t = 0; t < GGML_TYPE_COUNT; t++) {
        const char * tn = ggml_type_name((ggml_type) t);
        if (tn && name == tn) {
            return (ggml_type) t;
        }
    }
    throw std::runtime_error("unknown tensor type '" + name + "'");
}

int64_t Expr::eval(const std::vector<int64_t> & b) const {
    switch (kind) {
        case Kind::Const: return value;
        case Kind::Sym: return b.at((size_t) value);
        default: break;
    }
    int64_t acc = args.at(0).eval(b);
    for (size_t i = 1; i < args.size(); i++) {
        const int64_t x = args[i].eval(b);
        switch (kind) {
            case Kind::Add: acc += x; break;
            case Kind::Mul: acc *= x; break;
            case Kind::FloorDiv: {
                if (x == 0) throw std::runtime_error("division by zero in a symbolic dim");
                const int64_t q = acc / x;
                acc = (acc % x != 0 && ((acc < 0) != (x < 0))) ? q - 1 : q;
                break;
            }
            case Kind::Mod: {
                if (x == 0) throw std::runtime_error("modulo by zero in a symbolic dim");
                const int64_t r = acc % x;
                acc = (r != 0 && ((r < 0) != (x < 0))) ? r + x : r;
                break;
            }
            case Kind::Max: acc = std::max(acc, x); break;
            case Kind::Min: acc = std::min(acc, x); break;
            default: break;
        }
    }
    return acc;
}

void Expr::symbols(std::vector<bool> & used) const {
    if (kind == Kind::Sym) used.at((size_t) value) = true;
    for (const auto & a : args) a.symbols(used);
}

// A dim in the JSON: an integer, a symbol name, or [op, operand...] (see ggmlc/ir/sym.py).
static Expr parse_dim(const json & j, const std::vector<Symbol> & symbols) {
    Expr e;
    if (j.is_number_integer()) {
        e.value = j.get<int64_t>();
        return e;
    }
    if (j.is_string()) {
        const auto name = j.get<std::string>();
        for (size_t i = 0; i < symbols.size(); i++) {
            if (symbols[i].name == name) {
                e.kind  = Expr::Kind::Sym;
                e.value = (int64_t) i;
                return e;
            }
        }
        throw std::runtime_error("dim refers to unknown symbol '" + name + "'");
    }
    if (!j.is_array() || j.size() < 2 || !j[0].is_string()) {
        throw std::runtime_error("malformed dim " + j.dump());
    }
    const auto op = j[0].get<std::string>();
    if (op == "+") e.kind = Expr::Kind::Add;
    else if (op == "*") e.kind = Expr::Kind::Mul;
    else if (op == "//") e.kind = Expr::Kind::FloorDiv;
    else if (op == "%") e.kind = Expr::Kind::Mod;
    else if (op == "max") e.kind = Expr::Kind::Max;
    else if (op == "min") e.kind = Expr::Kind::Min;
    else throw std::runtime_error("unknown dim operator '" + op + "'");
    for (size_t i = 1; i < j.size(); i++) e.args.push_back(parse_dim(j[i], symbols));
    return e;
}

static std::array<Expr, 4> parse_ne(const json & j, const std::vector<Symbol> & symbols) {
    if (!j.is_array() || j.size() != 4) {
        throw std::runtime_error("'ne' must be an array of 4 dims");
    }
    std::array<Expr, 4> ne{};
    for (int i = 0; i < 4; i++) {
        ne[i] = parse_dim(j[i], symbols);
    }
    return ne;
}

static TensorSpec parse_tensor(const json & j, const std::vector<Symbol> & symbols) {
    TensorSpec t;
    t.name = j.at("name").get<std::string>();
    t.type = parse_type(j.at("type").get<std::string>());
    t.ne_x = parse_ne(j.at("ne"), symbols);
    for (const auto & d : j.at("shape")) t.shape_x.push_back(parse_dim(d, symbols));
    return t;
}

std::array<int64_t, 4> eval_ne(const TensorSpec & t, const std::vector<int64_t> & b) {
    std::array<int64_t, 4> ne{};
    for (int i = 0; i < 4; i++) ne[i] = t.ne_x[i].eval(b);
    return ne;
}

std::vector<int64_t> eval_shape(const TensorSpec & t, const std::vector<int64_t> & b) {
    std::vector<int64_t> shape;
    for (const auto & d : t.shape_x) shape.push_back(d.eval(b));
    return shape;
}

static void bind_tensor(TensorSpec & t, const std::vector<int64_t> & b) {
    t.ne    = eval_ne(t, b);
    t.shape = eval_shape(t, b);
    for (int64_t d : t.ne) {
        if (d < 1) throw std::runtime_error("value '" + t.name + "' gets a dim of " + std::to_string(d));
    }
}

void ProgramSpec::bind(const std::vector<int64_t> & b) {
    if (b.size() != symbols.size()) throw std::runtime_error("binding has the wrong number of symbols");
    for (size_t i = 0; i < symbols.size(); i++) {
        if (b[i] < symbols[i].min || b[i] > symbols[i].max) {
            throw std::runtime_error("symbol '" + symbols[i].name + "' = " + std::to_string(b[i]) + " is outside [" +
                                     std::to_string(symbols[i].min) + ", " + std::to_string(symbols[i].max) + "]");
        }
    }
    for (auto & v : values) bind_tensor(v, b);
    for (auto & s : stages) {
        for (auto & p : s.params) bind_tensor(p, b);
        for (auto & n : s.nodes) {
            for (int i = 0; i < 4; i++) n.ne[i] = n.ne_x[i].eval(b);
            for (const auto & [key, e] : n.p_x) n.p[key] = e.eval(b);
        }
    }
}

static Expr parse_dim(const json & j, const std::vector<Symbol> & symbols);

static DriverOp parse_driver_op(const json & jd, const std::vector<Symbol> & symbols) {
    DriverOp    d;
    const auto  kind = jd.at("op").get<std::string>();
    if (kind == "call") {
        d.kind  = DriverOp::Kind::Call;
        d.stage = jd.at("stage").get<std::string>();
        d.args  = jd.at("args").get<std::vector<std::string>>();
        d.out   = jd.at("out").get<std::vector<std::string>>();
    } else if (kind == "loop") {
        d.kind = DriverOp::Kind::Loop;
        if (jd.at("n").is_string()) {  // a trip count bound at run time
            const auto name = jd.at("n").get<std::string>();
            for (size_t i = 0; i < symbols.size(); i++) {
                if (symbols[i].name == name) d.n_symbol = (int) i;
            }
            if (d.n_symbol < 0) throw std::runtime_error("loop count refers to unknown symbol '" + name + "'");
            d.n = (int) symbols[d.n_symbol].max;
        } else {
            d.n = jd.at("n").get<int>();
        }
        // Counters are extra symbols for the bind expressions, after the program's own.
        std::vector<Symbol> ext = symbols;
        for (const auto & c : jd.value("counters", json::array())) {
            d.counters.emplace_back(c.at(0).get<std::string>(), c.at(1).get<std::string>());
            ext.push_back({c.at(0).get<std::string>(), 0, INT64_MAX});
        }
        for (const auto & b : jd.value("bind", json::array())) {
            const auto name = b.at(0).get<std::string>();
            int k = -1;
            for (size_t i = 0; i < symbols.size(); i++) {
                if (symbols[i].name == name) k = (int) i;
            }
            if (k < 0) throw std::runtime_error("loop binds unknown symbol '" + name + "'");
            d.bind.emplace_back(k, parse_dim(b.at(1), ext));
        }
        for (const auto & m : jd.value("masks", json::array())) {
            const auto counter = m.at(1).get<std::string>();
            int k = -1;
            for (size_t i = 0; i < d.counters.size(); i++) {
                if (d.counters[i].first == counter) k = (int) i;
            }
            if (k < 0) throw std::runtime_error("loop mask uses unknown counter '" + counter + "'");
            d.masks.emplace_back(m.at(0).get<std::string>(), k);
        }
        for (const auto & u : jd.value("until", json::array())) {
            d.until.emplace_back(u.at(0).get<std::string>(), u.at(1).get<std::vector<int>>());
        }
        for (const auto & c : jd.value("collect", json::array())) {
            d.collect.emplace_back(c.at(0).get<std::string>(), c.at(1).get<std::string>());
        }
        for (const auto & c : jd.at("carry")) {
            d.carry.emplace_back(c.at(0).get<std::string>(), c.at(1).get<std::string>());
        }
        for (const auto & sc : jd.at("scalars")) {
            auto values = sc.at("values").get<std::vector<float>>();
            if ((int) values.size() != d.n) {
                throw std::runtime_error("loop scalar has " + std::to_string(values.size()) + " entries for " +
                                         std::to_string(d.n) + " iterations");
            }
            d.scalars.emplace_back(sc.at("value").get<std::string>(), std::move(values));
        }
        for (const auto & sl : jd.value("slices", json::array())) {  // format 2
            d.slices.emplace_back(sl.at(0).get<std::string>(), sl.at(1).get<std::string>());
        }
        for (const auto & b : jd.at("body")) {
            d.body.push_back(parse_driver_op(b, symbols));
        }
    } else {
        throw std::runtime_error("driver op '" + kind + "' is not supported by this runtime");
    }
    return d;
}

ProgramSpec parse_program(const std::string & json_text) {
    json j;
    try {
        j = json::parse(json_text);
    } catch (const std::exception & e) {
        throw std::runtime_error(std::string("program JSON does not parse: ") + e.what());
    }

    ProgramSpec prog;
    try {
        prog.format_version = j.at("format_version").get<int>();
        if (prog.format_version < kMinFormatVersion || prog.format_version > kFormatVersion) {
            throw std::runtime_error("program format_version " + std::to_string(prog.format_version) +
                                     " is not supported (this runtime reads " + std::to_string(kMinFormatVersion) +
                                     " to " + std::to_string(kFormatVersion) + ")");
        }
        for (const auto & js : j.value("symbols", json::array())) {  // format 3
            prog.symbols.push_back({js.at("name").get<std::string>(), js.at("min").get<int64_t>(),
                                    js.at("max").get<int64_t>()});
        }
        for (const auto & v : j.at("values")) {
            prog.values.push_back(parse_tensor(v, prog.symbols));
        }

        for (const auto & js : j.at("stages")) {
            StageSpec s;
            s.name = js.at("name").get<std::string>();
            s.uses.assign(prog.symbols.size(), false);
            for (const auto & p : js.at("params")) {
                s.params.push_back(parse_tensor(p, prog.symbols));
                for (const auto & d : s.params.back().ne_x) d.symbols(s.uses);
            }
            s.outputs = js.at("outputs").get<std::vector<std::string>>();
            for (const auto & jn : js.at("nodes")) {
                NodeSpec n;
                n.id   = jn.at("id").get<std::string>();
                n.op   = jn.at("op").get<std::string>();
                n.src  = jn.at("src").get<std::vector<std::string>>();
                n.type = parse_type(jn.at("type").get<std::string>());
                n.ne_x = parse_ne(jn.at("ne"), prog.symbols);
                n.p    = jn.value("p", json::object());
                n.loc  = jn.value("loc", std::string());
                for (const auto & d : n.ne_x) d.symbols(s.uses);
                // Symbolic parameters: a view's start, an arange's start and stop.
                for (const char * key : {"start", "stop"}) {
                    if (n.p.contains(key) && !n.p[key].is_number()) {
                        n.p_x.emplace_back(key, parse_dim(n.p[key], prog.symbols));
                        n.p_x.back().second.symbols(s.uses);
                    }
                }
                s.nodes.push_back(std::move(n));
            }
            prog.stages.push_back(std::move(s));
        }

        auto parse_entry = [&prog](const json & je, const std::string & name) {
            EntrySpec e;
            e.name    = name;
            e.inputs  = je.at("inputs").get<std::vector<std::string>>();
            e.outputs = je.at("outputs").get<std::vector<std::string>>();
            for (const auto & jd : je.at("driver")) e.driver.push_back(parse_driver_op(jd, prog.symbols));
            return e;
        };
        if (j.contains("entries")) {  // format 3
            for (const auto & je : j.at("entries")) prog.entries.push_back(parse_entry(je, je.at("name").get<std::string>()));
        } else {
            prog.entries.push_back(parse_entry(j, "main"));
        }
        if (prog.entries.empty()) throw std::runtime_error("program has no entry point");
        prog.state = j.value("state", std::vector<std::string>{});
        std::vector<int64_t> lo;
        for (const auto & sy : prog.symbols) lo.push_back(sy.min);
        prog.bind(lo);
    } catch (const json::exception & e) {
        throw std::runtime_error(std::string("malformed program: ") + e.what());
    }
    return prog;
}

}  // namespace ggmlc
