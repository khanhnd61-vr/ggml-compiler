#include "ops.h"

#include "ggml_compiler_ops.h"

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace ggml_compiler {

namespace {

thread_local bool           g_gpu     = false;    // the backend the graph is being built for is a GPU
thread_local ggml_backend_t g_backend = nullptr;  // and that backend, for ops that ask what it can run

std::string shape_str(const int64_t * ne) {
    std::ostringstream os;
    os << "[" << ne[0] << ", " << ne[1] << ", " << ne[2] << ", " << ne[3] << "]";
    return os.str();
}

[[noreturn]] void fail(const NodeSpec & n, const std::string & msg) {
    std::string where = n.id + " (" + n.op + ")";
    if (!n.loc.empty()) {
        where += " at " + n.loc;
    }
    throw std::runtime_error(where + ": " + msg);
}

void need_src(const NodeSpec & n, const std::vector<ggml_tensor *> & src, size_t lo, size_t hi) {
    if (src.size() < lo || src.size() > hi) {
        fail(n, "expected " + std::to_string(lo) + (lo == hi ? "" : "-" + std::to_string(hi)) + " inputs, got " +
                    std::to_string(src.size()));
    }
}

float pf(const NodeSpec & n, const char * key) {
    if (!n.p.contains(key)) {
        fail(n, std::string("missing parameter '") + key + "'");
    }
    return n.p.at(key).get<float>();
}

int pi(const NodeSpec & n, const char * key) {
    if (!n.p.contains(key)) {
        fail(n, std::string("missing parameter '") + key + "'");
    }
    return n.p.at(key).get<int>();
}

using Emit = std::function<ggml_tensor *(ggml_context *, const NodeSpec &, const std::vector<ggml_tensor *> &)>;

Emit binary(ggml_tensor * (*fn)(ggml_context *, ggml_tensor *, ggml_tensor *)) {
    return [fn](ggml_context * ctx, const NodeSpec & n, const std::vector<ggml_tensor *> & s) {
        need_src(n, s, 2, 2);
        if (!ggml_can_repeat(s[1], s[0])) {
            fail(n, "cannot broadcast " + shape_str(s[1]->ne) + " into " + shape_str(s[0]->ne));
        }
        return fn(ctx, s[0], s[1]);
    };
}

Emit unary(ggml_tensor * (*fn)(ggml_context *, ggml_tensor *)) {
    return [fn](ggml_context * ctx, const NodeSpec & n, const std::vector<ggml_tensor *> & s) {
        need_src(n, s, 1, 1);
        return fn(ctx, s[0]);
    };
}

const std::unordered_map<std::string, Emit> & op_table() {
    static const std::unordered_map<std::string, Emit> table = {
        {"add", binary(ggml_add)},
        {"sub", binary(ggml_sub)},
        {"mul", binary(ggml_mul)},
        {"div", binary(ggml_div)},

        {"relu", unary(ggml_relu)},
        {"gelu", unary(ggml_gelu)},
        {"gelu_erf", unary(ggml_gelu_erf)},
        {"silu", unary(ggml_silu)},
        {"tanh", unary(ggml_tanh)},
        {"sigmoid", unary(ggml_sigmoid)},
        {"exp", unary(ggml_exp)},
        {"log", unary(ggml_log)},
        {"neg", unary(ggml_neg)},
        {"abs", unary(ggml_abs)},
        {"sqr", unary(ggml_sqr)},
        {"sqrt", unary(ggml_sqrt)},
        {"sin", unary(ggml_sin)},
        {"cos", unary(ggml_cos)},
        {"softplus", unary(ggml_softplus)},
        {"gelu_tanh",
         [](ggml_context * ctx, const NodeSpec & n, const std::vector<ggml_tensor *> & s) {
             need_src(n, s, 1, 1);
             return ops::gelu_tanh(ctx, s[0], g_gpu);
         }},
        {"cont", unary(ggml_cont)},
        {"mean", unary(ggml_mean)},
        {"cumsum", unary(ggml_cumsum)},
        {"step", unary(ggml_step)},
        {"sum_rows", unary(ggml_sum_rows)},

        {"mul_mat",
         [](ggml_context * ctx, const NodeSpec & n, const std::vector<ggml_tensor *> & s) {
             need_src(n, s, 2, 2);
             ggml_tensor * a = s[0];
             ggml_tensor * b = s[1];
             if (a->ne[0] != b->ne[0] || b->ne[2] % a->ne[2] != 0 || b->ne[3] % a->ne[3] != 0) {
                 fail(n, "cannot multiply " + shape_str(a->ne) + " by " + shape_str(b->ne));
             }
             if (ggml_is_transposed(a)) {
                 fail(n, "weight operand is transposed; it needs a cont first");
             }
             return ggml_compiler::ops::mul_mat(ctx, a, b, n.p.contains("acc") && pi(n, "acc") != 0);
         }},

        {"scale",
         [](ggml_context * ctx, const NodeSpec & n, const std::vector<ggml_tensor *> & s) {
             need_src(n, s, 1, 1);
             const ggml_tensor * x = s[0];  // ggml_scale's GGML_ASSERT(ggml_is_padded_1d(a))
             if (x->nb[0] != ggml_type_size(x->type) || x->nb[2] != x->nb[1] * x->ne[1] ||
                 x->nb[3] != x->nb[2] * x->ne[2]) {
                 fail(n, "input " + shape_str(s[0]->ne) + " is a strided view; it needs a cont first");
             }
             return ops::scale(ctx, s[0], pf(n, "s"), n.p.value("b", 0.0f));
         }},

        {"clamp",
         [](ggml_context * ctx, const NodeSpec & n, const std::vector<ggml_tensor *> & s) {
             need_src(n, s, 1, 1);
             return ggml_clamp(ctx, s[0], pf(n, "min"), pf(n, "max"));
         }},

        {"norm",
         [](ggml_context * ctx, const NodeSpec & n, const std::vector<ggml_tensor *> & s) {
             need_src(n, s, 1, 1);
             return ggml_norm(ctx, s[0], pf(n, "eps"));
         }},

        {"rms_norm",
         [](ggml_context * ctx, const NodeSpec & n, const std::vector<ggml_tensor *> & s) {
             need_src(n, s, 1, 1);
             return ggml_rms_norm(ctx, s[0], pf(n, "eps"));
         }},

        {"soft_max",
         [](ggml_context * ctx, const NodeSpec & n, const std::vector<ggml_tensor *> & s) {
             need_src(n, s, 1, 2);
             ggml_tensor * mask = s.size() > 1 ? s[1] : nullptr;
             return ggml_soft_max_ext(ctx, s[0], mask, n.p.value("scale", 1.0f), n.p.value("max_bias", 0.0f));
         }},

        {"reshape",
         [](ggml_context * ctx, const NodeSpec & n, const std::vector<ggml_tensor *> & s) {
             need_src(n, s, 1, 1);
             if (ggml_nelements(s[0]) != n.ne[0] * n.ne[1] * n.ne[2] * n.ne[3]) {
                 fail(n, "cannot reshape " + shape_str(s[0]->ne) + " to " + shape_str(n.ne.data()));
             }
             ggml_tensor * y = ops::reshape(ctx, s[0], n.ne[0], n.ne[1], n.ne[2], n.ne[3]);
             if (!y) {
                 fail(n, "input is a strided view and the reshape does more than add or drop size-1 dims; "
                         "it needs a cont first");
             }
             return y;
         }},

        {"permute",
         [](ggml_context * ctx, const NodeSpec & n, const std::vector<ggml_tensor *> & s) {
             need_src(n, s, 1, 1);
             const auto axes = n.p.at("axes").get<std::vector<int>>();
             if (axes.size() != 4) {
                 fail(n, "'axes' must have 4 entries");
             }
             return ggml_permute(ctx, s[0], axes[0], axes[1], axes[2], axes[3]);
         }},

        {"repeat",
         [](ggml_context * ctx, const NodeSpec & n, const std::vector<ggml_tensor *> & s) {
             need_src(n, s, 1, 1);
             for (int i = 0; i < 4; i++) {
                 if (n.ne[i] % s[0]->ne[i] != 0) {
                     fail(n, "cannot repeat " + shape_str(s[0]->ne) + " to " + shape_str(n.ne.data()));
                 }
             }
             return ggml_repeat_4d(ctx, s[0], n.ne[0], n.ne[1], n.ne[2], n.ne[3]);
         }},

        {"concat",
         [](ggml_context * ctx, const NodeSpec & n, const std::vector<ggml_tensor *> & s) {
             need_src(n, s, 2, 2);
             const int dim = pi(n, "dim");
             for (int i = 0; i < 4; i++) {
                 if (i != dim && s[0]->ne[i] != s[1]->ne[i]) {
                     fail(n, "cannot concat " + shape_str(s[0]->ne) + " and " + shape_str(s[1]->ne) +
                                 " along ggml dim " + std::to_string(dim));
                 }
             }
             return ggml_concat(ctx, s[0], s[1], dim);
         }},

        {"view",
         // A slice along one ggml dim with step 1: ne comes from the node, strides from the input.
         [](ggml_context * ctx, const NodeSpec & n, const std::vector<ggml_tensor *> & s) {
             need_src(n, s, 1, 1);
             const int     dim   = pi(n, "dim");
             const int64_t start = n.p.at("start").get<int64_t>();
             if (dim < 0 || dim > 3 || start < 0 || start + n.ne[dim] > s[0]->ne[dim]) {
                 fail(n, "slice out of range of " + shape_str(s[0]->ne));
             }
             return ops::slice(ctx, s[0], dim, start, n.ne[0], n.ne[1], n.ne[2], n.ne[3]);
         }},

        {"flash_attn",
         // q [D, Sq, H, B] f32, k/v [D, Sk, Hkv, B] f16, mask [Sk, >=Sq, ...] f16 -> [Dv, H, Sq, B]
         [](ggml_context * ctx, const NodeSpec & n, const std::vector<ggml_tensor *> & s) {
             need_src(n, s, 3, 4);
             ggml_tensor *q = s[0], *k = s[1], *v = s[2];
             ggml_tensor * mask = s.size() > 3 ? s[3] : nullptr;
             if (k->ne[0] != q->ne[0] || k->ne[1] != v->ne[1] || q->ne[3] != k->ne[3] || q->ne[3] != v->ne[3] ||
                 q->ne[2] % k->ne[2] != 0) {
                 fail(n, "q " + shape_str(q->ne) + ", k " + shape_str(k->ne) + ", v " + shape_str(v->ne) +
                             " do not line up");
             }
             if (mask && (mask->type != GGML_TYPE_F16 || !ggml_is_contiguous(mask) || q->ne[2] % mask->ne[2] != 0)) {
                 fail(n, "mask must be a contiguous f16 tensor broadcastable over heads");
             }
             return ggml_compiler::ops::flash_attn(ctx, q, k, v, mask, pf(n, "scale"), [&n](ggml_tensor * t) {
                 if (!g_backend || ggml_backend_supports_op(g_backend, t)) {
                     return true;
                 }
                 const char * log = std::getenv("GGML_COMPILER_LOG");
                 if (log && std::string(log) == "debug") {
                     std::fprintf(stderr, "node %s: %s cannot run flash attention with head size %lld; running it as "
                                  "matmul, softmax, matmul\n", n.id.c_str(), ggml_backend_name(g_backend),
                                  (long long) t->src[0]->ne[0]);
                 }
                 return false;
             });
         }},

        {"arange",
         [](ggml_context * ctx, const NodeSpec & n, const std::vector<ggml_tensor *> & s) {
             need_src(n, s, 0, 0);
             return ggml_arange(ctx, pf(n, "start"), pf(n, "stop"), pf(n, "step"));
         }},

        {"swiglu",
         [](ggml_context * ctx, const NodeSpec & n, const std::vector<ggml_tensor *> & s) {
             need_src(n, s, 2, 2);
             if (!ggml_are_same_shape(s[0], s[1]) || !ggml_is_contiguous(s[0]) || !ggml_is_contiguous(s[1])) {
                 fail(n, "swiglu needs a gate and an up of one shape, both contiguous");
             }
             return ggml_swiglu_split(ctx, s[0], s[1]);
         }},

        {"set_rows",
         // dst [C, R, ...], rows [C, n, ...] f32/f16, indices [n] i32: dst's rows written in place.
         [](ggml_context * ctx, const NodeSpec & n, const std::vector<ggml_tensor *> & s) {
             need_src(n, s, 3, 3);
             ggml_tensor *dst = s[0], *src = s[1], *idx = s[2];
             if (dst->ne[0] != src->ne[0] || dst->ne[2] != src->ne[2] || dst->ne[3] != src->ne[3] ||
                 src->ne[1] != idx->ne[0] || idx->ne[1] != 1 || idx->type != GGML_TYPE_I32 ||
                 (src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_F16)) {
                 fail(n, "cannot write rows " + shape_str(src->ne) + " at indices " + shape_str(idx->ne) + " into " +
                             shape_str(dst->ne));
             }
             if (dst->nb[0] != ggml_type_size(dst->type) || src->nb[0] != ggml_type_size(src->type)) {
                 fail(n, "set_rows needs rows that are contiguous");
             }
             return ggml_set_rows(ctx, dst, src, idx);
         }},

        {"rope",
         // x [D, H, S, B] (may be strided), positions [S] i32 [, freq_factors [n_dims/2] f32]
         [](ggml_context * ctx, const NodeSpec & n, const std::vector<ggml_tensor *> & s) {
             need_src(n, s, 2, 3);
             ggml_tensor *x = s[0], *pos = s[1];
             ggml_tensor * ff = s.size() > 2 ? s[2] : nullptr;
             const int     n_dims = pi(n, "n_dims");
             if (x->type != GGML_TYPE_F32 || x->nb[0] != ggml_type_size(x->type) || pos->type != GGML_TYPE_I32 ||
                 !ggml_is_vector(pos) || pos->ne[0] != x->ne[2] || n_dims % 2 || n_dims > x->ne[0] ||
                 (ff && (ff->type != GGML_TYPE_F32 || ff->ne[0] < n_dims / 2))) {
                 fail(n, "cannot rotate " + shape_str(x->ne) + " at positions " + shape_str(pos->ne));
             }
             return ggml_rope_ext(ctx, x, pos, ff, n_dims, pi(n, "mode"), 0, pf(n, "freq_base"), pf(n, "freq_scale"),
                                  0.0f, pf(n, "attn_factor"), 32.0f, 1.0f);
         }},

        {"argmax",
         [](ggml_context * ctx, const NodeSpec & n, const std::vector<ggml_tensor *> & s) {
             need_src(n, s, 1, 1);
             if (!ggml_is_matrix(s[0])) {
                 fail(n, "argmax needs a matrix, got " + shape_str(s[0]->ne));
             }
             return ggml_argmax(ctx, s[0]);
         }},

        {"get_rows",
         [](ggml_context * ctx, const NodeSpec & n, const std::vector<ggml_tensor *> & s) {
             need_src(n, s, 2, 2);
             if (s[1]->type != GGML_TYPE_I32) {
                 fail(n, "row indices must be i32");
             }
             return ggml_get_rows(ctx, s[0], s[1]);
         }},

        {"im2col",
         // Always F32: ggml_conv_2d would build the patch buffer in F16.
         [](ggml_context * ctx, const NodeSpec & n, const std::vector<ggml_tensor *> & s) {
             need_src(n, s, 2, 2);
             if (s[0]->ne[2] != s[1]->ne[2]) {
                 fail(n, "kernel " + shape_str(s[0]->ne) + " does not match input channels " + shape_str(s[1]->ne));
             }
             return ggml_im2col(ctx, s[0], s[1], pi(n, "s0"), pi(n, "s1"), pi(n, "p0"), pi(n, "p1"), pi(n, "d0"),
                                pi(n, "d1"), true, GGML_TYPE_F32);
         }},

        {"conv_2d",
         [](ggml_context * ctx, const NodeSpec & n, const std::vector<ggml_tensor *> & s) {
             need_src(n, s, 2, 3);
             ggml_tensor *w = s[0], *x = s[1];
             if (w->ne[2] != x->ne[2]) {
                 fail(n, "kernel " + shape_str(w->ne) + " does not match input channels " + shape_str(x->ne));
             }
             return ops::conv_2d(ctx, w, x, s.size() > 2 ? s[2] : nullptr, pi(n, "s0"), pi(n, "s1"), pi(n, "p0"),
                                 pi(n, "p1"), pi(n, "d0"), pi(n, "d1"), g_gpu);
         }},

        {"pool_2d",
         [](ggml_context * ctx, const NodeSpec & n, const std::vector<ggml_tensor *> & s) {
             need_src(n, s, 1, 1);
             const std::string op = n.p.at("op").get<std::string>();
             if (op != "max" && op != "avg") {
                 fail(n, "pool op must be max or avg");
             }
             return ggml_pool_2d(ctx, s[0], op == "max" ? GGML_OP_POOL_MAX : GGML_OP_POOL_AVG, pi(n, "k0"),
                                 pi(n, "k1"), pi(n, "s0"), pi(n, "s1"), pf(n, "p0"), pf(n, "p1"));
         }},

        {"cast",
         [](ggml_context * ctx, const NodeSpec & n, const std::vector<ggml_tensor *> & s) {
             need_src(n, s, 1, 1);
             return ggml_cast(ctx, s[0], n.type);
         }},
    };
    return table;
}

}  // namespace

ggml_tensor * emit_node(ggml_context * ctx, const NodeSpec & n, const std::vector<ggml_tensor *> & src,
                        ggml_backend_t backend) {
    g_backend          = backend;
    g_gpu              = backend && ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_GPU;
    const auto & table = op_table();
    auto         it    = table.find(n.op);
    if (it == table.end()) {
        fail(n, "op is not supported by this runtime");
    }
    ggml_tensor * t = it->second(ctx, n, src);
    if (t == nullptr) {
        fail(n, "ggml returned no tensor (graph context full?)");
    }
    if (t->type != n.type) {
        fail(n, std::string("produced type ") + ggml_type_name(t->type) + ", program expects " + ggml_type_name(n.type));
    }
    for (int i = 0; i < 4; i++) {
        if (t->ne[i] != n.ne[i]) {
            fail(n, "produced shape " + shape_str(t->ne) + ", program expects " + shape_str(n.ne.data()));
        }
    }
    ggml_set_name(t, n.id.c_str());
    return t;
}

}  // namespace ggml_compiler
