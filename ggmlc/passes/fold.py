"""Algebraic clean-up and compile-time evaluation of weight-only nodes.

simplify: always: permute(permute(x)) -> one permute; identity permutes and reshapes vanish; a
          permute that only moves size-1 dims is a reshape; reshape(reshape(x)) -> one reshape.
          Flag simplify: x + 0, x * 1 and all-zero softmax masks vanish, and a slice of a concat
          reads its part. fuse_rms_norm: RMSNorm written out (x * rsqrt(mean(x^2) + eps)) is
          one rms_norm. flatten_matmul: a 2-D weight times a batched activation becomes one
          2-D matmul. shared_kv_heads: K/V heads repeated for grouped-query attention are
          broadcast by the matmul instead.
fold:     always: a view of a weight above 4-D is folded (the GGUF cannot hold it).
          fold_batch_norm: a per-channel affine after a conv goes into its kernel and bias.
          fold_constants: a node reading only weights and constants is computed here, with the
          op's PyTorch reference, into a derived weight (or a constant). Cheap views (reshape,
          view) are left as ggml views, and nothing is folded into a tensor larger than its inputs.
"""

from __future__ import annotations

import torch

from ..ir import Builder, Const, Node, Stage, Value, Weight, sym
from ..ir.ops import OPS
from ..ir.program import ProgramIR
from ..options import Options, resolve
from . import rewrite


def _all(v: Value, value: float) -> bool:
    return isinstance(v, Const) and bool((v.data == value).all())


def _shared_heads(b: Builder, a: Value, other: Value) -> Value | None:
    """mul_mat's first operand when it is grouped-query attention's repeat_kv: K or V heads
    repeated r times ([KV, 1, S, D] -> [KV, r, S, D] -> [KV * r, S, D]), possibly transposed in its
    last two dims. ggml's mul_mat broadcasts a batch dim the same way (b head i reads a head
    i // r), so the [KV, S, D] original serves without the copy."""
    v, perm = a, None
    if isinstance(v, Node) and v.op == "permute":
        perm = list(v.attrs["perm"])
        if perm[:-2] != list(range(len(perm) - 2)) or perm[-2:] != [len(perm) - 1, len(perm) - 2]:
            return None
        v = v.inputs[0]
    base = v.shape  # a before its transpose: [..., H, S, D]
    while isinstance(v, Node) and v.op == "reshape":
        v = v.inputs[0]
    if not (isinstance(v, Node) and v.op == "repeat") or len(base) < 3:
        return None
    x = v.inputs[0]
    rep = v.shape
    src = (1,) * (len(rep) - x.rank) + x.shape
    diff = [j for j in range(len(rep)) if src[j] != rep[j]]
    if len(diff) != 1 or src[diff[0]] != 1 or diff[0] != len(rep) - 3 or rep[-2:] != base[-2:]:
        return None
    r, heads = rep[diff[0]], base[-3]
    kv = 1
    for d in rep[:-3]:
        kv *= d
    if kv * r != heads or any(d != 1 for d in base[:-3]) or other.rank < 3 or other.shape[-3] != heads:
        return None
    y = b.reshape(x, shape=[*base[:-3], kv, *base[-2:]])
    return b.permute(y, perm=perm) if perm else y


def _algebra(b: Builder, n: Node) -> Value | None:
    """Identities with constant operands: x + 0, x - 0, x * 1, x / 1, softmax with a zero
    mask; and a slice that lies inside one part of a concat."""
    x = n.inputs[0] if n.inputs else None
    if n.op in ("add", "sub") and _all(n.inputs[1], 0.0) or n.op in ("mul", "div") and _all(n.inputs[1], 1.0):
        return x
    if n.op == "add" and _all(x, 0.0) and n.inputs[1].shape == n.shape:
        return n.inputs[1]
    if n.op == "soft_max" and len(n.inputs) == 2 and _all(n.inputs[1], 0.0):
        return b.soft_max(x, **n.attrs)
    if n.op == "view" and isinstance(x, Node) and x.op == "concat" and x.attrs["dim"] == n.attrs["dim"]:
        # A slice that lies inside one part of a concat reads that part directly (this is what
        # functionalized `res[..., :h] = a; res[..., h:] = b` turns into).
        d, start, length = n.attrs["dim"], n.attrs["start"], n.attrs["length"]
        a, c = x.inputs
        if sym.definitely(start + length <= a.shape[d]):
            return a if (start, length) == (0, a.shape[d]) else b.view(a, dim=d, start=start, length=length)
        if sym.definitely(start >= a.shape[d]):
            s0 = start - a.shape[d]
            return c if (s0, length) == (0, c.shape[d]) else b.view(c, dim=d, start=s0, length=length)
    return None


def _ones(v: Value) -> bool:
    while isinstance(v, Node) and v.op in ("repeat", "reshape"):
        v = v.inputs[0]
    return _all(v, 1.0)


def _rms_norm(b: Builder, n: Node) -> Value | None:
    """x * (1 / sqrt(mean(x^2) + eps)), or x / sqrt(mean(x^2) + eps): rms_norm(x, eps)."""
    pairs = []
    if n.op == "mul":
        for x, d in ((n.inputs[0], n.inputs[1]), (n.inputs[1], n.inputs[0])):
            if isinstance(d, Node) and d.op == "div" and _ones(d.inputs[0]):
                pairs.append((x, d.inputs[1]))
    elif n.op == "div":
        pairs.append((n.inputs[0], n.inputs[1]))
    for x, sq in pairs:
        if not (isinstance(sq, Node) and sq.op == "sqrt"):
            continue
        e = sq.inputs[0]
        if not (isinstance(e, Node) and e.op == "scale" and e.attrs.get("s", 1.0) == 1.0):
            continue
        m = e.inputs[0]
        if not (isinstance(m, Node) and m.op == "mean" and isinstance(m.inputs[0], Node)):
            continue
        q = m.inputs[0]
        if q.op == "sqr" and q.inputs[0] is x and n.shape == x.shape and x.rank >= 1:
            return b.rms_norm(x, eps=float(e.attrs.get("b", 0.0)))
    return None


def simplify(stage: Stage, options: Options | None = None) -> None:
    options = options or resolve()
    algebra, flatten, shared_heads = ("simplify" in options, "flatten_matmul" in options,
                                      "shared_kv_heads" in options)
    rms = "fuse_rms_norm" in options
    glu = "fuse_glu" in options

    def fix(b: Builder, n: Node) -> Value | None:
        x = n.inputs[0] if n.inputs else None
        if glu and n.op == "mul" and n.inputs[0].shape == n.inputs[1].shape:
            for g, u in ((n.inputs[0], n.inputs[1]), (n.inputs[1], n.inputs[0])):
                if isinstance(g, Node) and g.op == "silu":
                    return b.swiglu(g.inputs[0], u)
        if rms and n.op in ("mul", "div"):
            r = _rms_norm(b, n)
            if r is not None:
                return r
        if n.op == "mul_mat" and shared_heads:
            shared = _shared_heads(b, x, n.inputs[1])
            if shared is not None:
                return b.mul_mat(shared, n.inputs[1])
        if algebra:
            r = _algebra(b, n)
            if r is not None:
                return r
        if flatten and n.op == "mul_mat" and n.inputs[0].rank == 2 and n.inputs[1].rank > 2:
            # A 2-D weight times [..., M, K]: one [rows, K] GEMM, not a batch of small ones. This
            # matters when the batch dims are not the outermost (seq-first [S, 1, D] in ACT is
            # 602 one-row matmuls to ggml).
            xb = n.inputs[1]
            rows = xb.type.numel() // xb.shape[-1]
            y = b.mul_mat(x, b.reshape(xb, shape=[rows, xb.shape[-1]]))
            return b.reshape(y, shape=list(n.shape))
        if n.op == "permute":
            perm = list(n.attrs["perm"])
            if isinstance(x, Node) and x.op == "permute":
                inner = list(x.attrs["perm"])
                perm = [inner[p] for p in perm]
                x = x.inputs[0]
            if perm == list(range(len(perm))):
                return x
            if [p for p in perm if x.shape[p] != 1] == sorted(p for p in perm if x.shape[p] != 1):
                # Only size-1 dims move: the same bytes, so a reshape. As a permute, ggml would see
                # swapped strides (ggml_is_transposed) and refuse it as a matmul operand.
                return b.reshape(x, shape=list(n.shape))
            if x is not n.inputs[0]:
                return b.permute(x, perm=perm)
        if n.op == "reshape":
            while isinstance(x, Node) and x.op == "reshape":
                x = x.inputs[0]
            if x.shape == n.shape:
                return x
            if x is not n.inputs[0]:
                return b.reshape(x, shape=list(n.shape))
        return None

    rewrite(stage, fix)


_KEEP = {"reshape", "view"}


def _per_channel(c: Value, conv: Node) -> torch.Tensor | None:
    """c as a [OC] vector if it is a constant that varies only along conv's channel dim."""
    oc = conv.shape[1]
    if not isinstance(c, Const) or c.type.numel() != oc:
        return None
    shape = (1,) * (4 - c.rank) + c.shape
    return c.data.reshape(oc) if shape[1] == oc else None


def fold(prog: ProgramIR, options: Options | None = None) -> None:
    options = options or resolve()
    constants = "fold_constants" in options

    def derived(name: str, t: torch.Tensor) -> Weight:
        t = t.detach().to(torch.float32).contiguous()
        base, k = name, 1
        while name in prog.weights:
            w = prog.weights[name]
            if w.data.shape == t.shape and torch.equal(w.data, t):
                return w
            k += 1
            name = f"{base}#{k}"
        prog.weights[name] = w = Weight(t, name)
        return w

    for stage in prog.stages.values():
        users = stage.users()

        def affine_into_conv(b: Builder, n: Node) -> Value | None:
            """conv(x) * c + d, with per-channel constants c and d (frozen or inference-mode batch
            norm): fold c and d into the conv's kernel and bias."""
            if n.op not in ("mul", "add"):
                return None
            conv, c = n.inputs
            if not (isinstance(conv, Node) and conv.op == "conv_2d" and isinstance(conv.inputs[0], Weight)):
                return None
            vec = _per_channel(c, conv)
            if vec is None or len(users.get(id(conv), [])) != 1 or any(o is conv for o in stage.outputs):
                return None
            w, x = conv.inputs[0], conv.inputs[1]
            bias = conv.inputs[2].data if len(conv.inputs) > 2 else torch.zeros(conv.shape[1])
            if n.op == "mul":
                w = derived(f"{w.fqn}:scaled", w.data * vec.reshape(-1, 1, 1, 1))
                bias = bias * vec
            else:
                bias = bias + vec
            new = b.conv_2d(w, x, derived(f"{w.fqn.split(':')[0]}:bias", bias), **conv.attrs)
            users[id(new)] = users.get(id(n), [])
            return new

        if "fold_batch_norm" in options:
            rewrite(stage, affine_into_conv)

        memo: dict[int, torch.Tensor] = {}

        def leaves_of(v: Value) -> list[Value] | None:
            """The weights/constants v is a view of, or None if v depends on anything at runtime."""
            if isinstance(v, (Weight, Const)):
                return [v]
            if isinstance(v, Node) and v.op in _KEEP:
                parts = [leaves_of(x) for x in v.inputs]
                return None if any(p is None for p in parts) else [x for p in parts for x in p]
            return None

        def value_of(v: Value) -> torch.Tensor:
            if isinstance(v, (Weight, Const)):
                return v.data
            if id(v) not in memo:
                memo[id(v)] = OPS[v.op].ref([value_of(x) for x in v.inputs], v.attrs)
            return memo[id(v)]

        def fix(b: Builder, n: Node) -> Value | None:
            # Fold a non-view node whose inputs are weights or constants, possibly behind views
            # (views alone stay free ggml views). A view of a weight above 4-D is folded too: the
            # GGUF cannot hold the weight itself (a Conv3d kernel read as a matmul).
            if not n.inputs:
                return None
            if n.op in _KEEP:
                x = n.inputs[0]
                if isinstance(x, (Weight, Const)) and x.rank > 4 and n.rank <= 4:
                    data = OPS[n.op].ref([x.data], n.attrs)
                    return derived(f"{x.fqn}:{n.op}", data) if isinstance(x, Weight) else b.const(data)
                return None
            if not constants or not n.type.is_static or sym.free(n.attrs):
                return None
            sources = [leaves_of(v) for v in n.inputs]
            if any(src is None for src in sources):
                return None
            leaves = {id(x): x for src in sources for x in src}
            if n.type.numel() > sum(x.type.numel() for x in leaves.values()):
                return None
            data = OPS[n.op].ref([value_of(v) for v in n.inputs], n.attrs)
            ws = [x for x in leaves.values() if isinstance(x, Weight)]
            return derived(f"{ws[0].fqn}:{n.op}", data) if ws else b.const(data)

        rewrite(stage, fix)
