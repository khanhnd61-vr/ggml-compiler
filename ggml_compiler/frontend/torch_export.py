"""PyTorch frontend: torch.export each stage, then lower its ATen graph to the IR.

- Parameters become Weights keyed by their FQN on the program's root module and shared by
  every stage (deduplicated by tensor identity, so tied weights are stored once).
- Buffers and lifted constants are compile-time constants: any node whose inputs are all
  constant is evaluated with PyTorch here (RoPE tables, position embeddings, masks).
- Ops with fused ggml lowerings are kept whole by removing them from the decomposition table.
- Missing lowerings are collected over the whole graph and reported together.
"""

from __future__ import annotations

import contextlib
import math
import operator
import os
import re
import warnings
from dataclasses import dataclass
from typing import Any

import sympy
import torch
import torch.fx
from torch.utils import _pytree as pytree

from .. import api
from ..ir import Builder, Stage, TensorType, Value, Weight, sym
from ..ir.program import CallOp, CompileError, Entry, LoopOp, ProgramIR
from ..lower import aten as _aten_lowerings  # noqa: F401  (registers the built-in lowerings)
from ..lower.registry import LOWERINGS, op_key

aten = torch.ops.aten

PRESERVE = [
    aten.linear.default,
    aten.layer_norm.default,
    aten.scaled_dot_product_attention.default,
]

_FLOAT_DTYPES = (torch.float32, torch.float16, torch.bfloat16, torch.float64)


class _StageModule(torch.nn.Module):
    """Wraps a stage function so torch.export sees the root module's parameters."""

    def __init__(self, root: torch.nn.Module | None, fn: Any) -> None:
        super().__init__()
        if root is not None:
            self.root = root
        object.__setattr__(self, "_fn", fn)  # not registered: its parameters belong to root

    def forward(self, *args: Any) -> Any:
        return self._fn(*args)


class WeightTable:
    """The program's weights, deduplicated by the identity of their source tensors.

    copy=True makes every f32 weight a copy instead of a view of the model's parameter. ModelOpt
    import needs that: it restores the original weights in place once the stages are exported.
    Otherwise views keep a multi-GB model from being held twice."""

    def __init__(self, copy: bool = False) -> None:
        self.copy = copy
        self.by_fqn: dict[str, Weight] = {}
        # Storage address -> (source tensor, weight). Holding the source keeps the address from
        # being reused by a later tensor (the ONNX frontend's initializers are temporaries).
        self._by_storage: dict[tuple, tuple[torch.Tensor, Weight]] = {}

    def get(self, fqn: str, t: torch.Tensor) -> Weight:
        if fqn in self.by_fqn:
            return self.by_fqn[fqn]
        key = (t.untyped_storage().data_ptr(), t.storage_offset(), tuple(t.shape), tuple(t.stride()), t.dtype)
        hit = self._by_storage.get(key)
        w = hit[1] if hit is not None else None
        if w is None:
            data = t.detach()
            if data.dtype in _FLOAT_DTYPES and data.dtype != torch.float32:
                data = data.to(torch.float32)  # a new tensor already
            elif self.copy:
                data = data.clone(memory_format=torch.contiguous_format)
            w = Weight(data.contiguous(), fqn)
            self._by_storage[key] = (t, w)
        self.by_fqn[fqn] = w
        return w

    def derived(self, name: str, t: torch.Tensor) -> Weight:
        """A weight computed at compile time from other weights. Same name and data: reused."""
        t = t.detach().to(torch.float32).contiguous()
        base, k = name, 1
        while name in self.by_fqn:
            w = self.by_fqn[name]
            if w.data.shape == t.shape and torch.equal(w.data, t):
                return w
            k += 1
            name = f"{base}#{k}"
        w = Weight(t, name)
        self.by_fqn[name] = w
        return w


# Ops on weights that ggml does for free as views: keep them in the graph instead of storing a
# copy of the weight.
_VIEW_OPS = {
    "aten.view.default", "aten.reshape.default", "aten._unsafe_view.default", "aten.unsqueeze.default",
    "aten.squeeze.dim", "aten.squeeze.dims", "aten.squeeze.default", "aten.flatten.using_ints",
    "aten.slice.Tensor", "aten.select.int", "aten.split.Tensor", "aten.split_with_sizes.default",
    "aten.unbind.int", "aten.alias.default", "aten.detach.default", "aten.clone.default",
}


# Ops that read only their tensor arguments' shape and dtype, which are static: always constant.
_SHAPE_ONLY = {
    "aten.full_like.default", "aten.zeros_like.default", "aten.ones_like.default", "aten.empty_like.default",
    "aten.new_zeros.default", "aten.new_ones.default", "aten.new_full.default", "aten.new_empty.default",
    "aten.sym_size.int",
}


def _fold_shape_only(node: torch.fx.Node, args: Any, kwargs: Any) -> Any:
    dummy = lambda x: torch.zeros(x.shape, dtype=torch.float32) if isinstance(x, Value) else x  # noqa: E731
    real_args, real_kwargs = pytree.tree_map(dummy, (args, kwargs))
    with torch.no_grad():
        out = node.target(*real_args, **real_kwargs)
    if node.target in (aten.empty_like.default, aten.new_empty.default):
        out = torch.zeros_like(out)  # empty means "any values"; make them deterministic
    return out


def _fold_weights(key: str, node: torch.fx.Node, args: Any, kwargs: Any, weights: WeightTable) -> Weight | None:
    """Evaluate an op whose only runtime inputs are weights, if that does not grow storage."""
    if key in _VIEW_OPS:
        return None
    ins = [leaf for leaf in pytree.tree_leaves((args, kwargs)) if isinstance(leaf, Weight)]
    real_args, real_kwargs = pytree.tree_map(lambda x: x.data if isinstance(x, Weight) else x, (args, kwargs))
    with torch.no_grad():
        out = node.target(*real_args, **real_kwargs)
    if not isinstance(out, torch.Tensor) or not out.is_floating_point():
        return None
    if out.numel() > sum(w.data.numel() for w in ins):
        return None
    return weights.derived(f"{ins[0].fqn}:{key.split('.')[1]}", out)


@dataclass
class Reference:
    """Example inputs and PyTorch outputs, keyed by flattened driver value name."""

    inputs: dict[str, torch.Tensor]
    outputs: dict[str, torch.Tensor]


# --- names and locations -------------------------------------------------------------------


def _leaf_name(prefix: str, path: tuple) -> str:
    parts = [prefix]
    for k in path:
        if isinstance(k, pytree.SequenceKey):
            parts.append(str(k.idx))
        elif isinstance(k, pytree.MappingKey):
            parts.append(str(k.key))
        elif isinstance(k, pytree.GetAttrKey):
            parts.append(k.name)
        else:
            parts.append(str(k))
    return ".".join(parts)


def flatten_named(prefix: str, tree: Any) -> list[tuple[str, Any]]:
    leaves, _ = pytree.tree_flatten_with_path(tree)
    return [(_leaf_name(prefix, path), leaf) for path, leaf in leaves]


_FRAME = re.compile(r'File "([^"]+)", line (\d+), in (\S+)')


def _loc(node: torch.fx.Node) -> str:
    frames = _FRAME.findall(node.meta.get("stack_trace") or "")
    if not frames:
        return ""
    f, line, fn = frames[-1]
    return f"{os.path.basename(f)}:{line} in {fn}"


def _scope(node: torch.fx.Node, prefix: str) -> str:
    """The innermost PyTorch module the node runs in, as a path on the root module."""
    stack = node.meta.get("nn_module_stack") or {}
    if not stack:
        return ""
    path = list(stack.values())[-1][0]
    path = re.sub(r"^L\['self'\]\.?", "", path)
    path = re.sub(r"\[(\d+)\]", r".\1", path)
    return path[len(prefix):] if path.startswith(prefix) else path


def _io_type(name: str, t: Any, dims: "_Dims | None" = None, state: bool = False) -> TensorType:
    """Runtime type of a driver value: integers become i32 (token ids, indices), everything else
    f32 (bool masks as 0/1). State may also be f16 (a KV cache in half precision)."""
    if not isinstance(t, torch.Tensor):
        raise CompileError(f"driver value {name} is a {type(t).__name__}, not a tensor")
    if t.dtype.is_complex:
        raise CompileError(f"driver value {name} is complex")
    integer = not t.dtype.is_floating_point and t.dtype != torch.bool
    shape = dims.shape(t, name) if dims is not None else tuple(t.shape)
    if state and t.dtype == torch.float16:
        return TensorType(shape, "f16")
    return TensorType(shape, "i32" if integer else "f32")


def runtime_tensor(t: torch.Tensor) -> torch.Tensor:
    """A tensor as the runtime stores it (see _io_type)."""
    integer = not t.dtype.is_floating_point and t.dtype != torch.bool
    return t.detach().to(torch.int32 if integer else torch.float32)


# --- symbolic dims --------------------------------------------------------------------------


def _has_sym(x: Any) -> bool:
    return any(isinstance(leaf, sympy.Basic) for leaf in pytree.tree_leaves(x))


# Python-level arithmetic on sizes that torch.export records as graph nodes.
_SYM_OPS: dict[Any, Any] = {
    operator.add: operator.add, operator.sub: operator.sub, operator.mul: operator.mul,
    operator.floordiv: operator.floordiv, operator.mod: operator.mod, operator.neg: operator.neg,
    operator.eq: sympy.Eq, operator.ne: sympy.Ne, operator.lt: sympy.Lt, operator.le: sympy.Le,
    operator.gt: sympy.Gt, operator.ge: sympy.Ge,
    torch.sym_max: sym.smax, torch.sym_min: sym.smin, torch.sym_int: lambda x: x,
    math.ceil: sympy.ceiling, math.floor: sympy.floor,
}


class _Dims:
    """torch.export's shape symbols (s11, s66) -> the program's named symbols."""

    def __init__(self) -> None:
        self.map: dict[sympy.Symbol, sympy.Symbol] = {}

    def __call__(self, d: Any, what: str = "") -> sym.Dim:
        if isinstance(d, torch.SymInt):
            e = d.node.expr
            unknown = e.free_symbols - self.map.keys()
            if unknown:
                raise CompileError(f"{what or 'a value'} has a size ({e}) that depends on data or on a dim not "
                                   f"declared in dynamic_shapes")
            return sym.norm(e.xreplace(self.map))
        return int(d)

    def shape(self, t: torch.Tensor, what: str = "") -> tuple[sym.Dim, ...]:
        return tuple(self(d, what) for d in t.shape)


def _walk_spec(example: Any, spec: Any):
    """(example leaf, its dynamic spec or None) in pytree leaf order."""
    if isinstance(example, torch.Tensor) or spec is None:
        for leaf in pytree.tree_leaves(example):
            yield leaf, spec if isinstance(example, torch.Tensor) else None
    elif isinstance(example, (list, tuple)):
        if not isinstance(spec, (list, tuple)) or len(spec) != len(example):
            raise CompileError(f"dynamic_shapes entry {spec!r} does not match an input of {len(example)} items")
        for e, sp in zip(example, spec):
            yield from _walk_spec(e, sp)
    elif isinstance(example, dict):
        for k in example:
            yield from _walk_spec(example[k], spec.get(k))
    else:
        yield example, None


def _mirror_spec(example: Any, spec: Any) -> Any:
    """torch.export wants a spec shaped like its input: a None for a whole pytree becomes one None
    per leaf."""
    if isinstance(example, torch.Tensor):
        return spec
    if spec is None:
        return pytree.tree_map(lambda _: None, example)
    if isinstance(example, (list, tuple)) and isinstance(spec, (list, tuple)):
        return type(spec)(_mirror_spec(e, sp) for e, sp in zip(example, spec))
    if isinstance(example, dict) and isinstance(spec, dict):
        return {k: _mirror_spec(example[k], spec.get(k)) for k in example}
    return spec


def _define_symbols(name: str, ep: Any, stage: api.Stage, inputs: list[torch.fx.Node], dims: _Dims) -> None:
    """Name torch's symbols after the user's Dims, with torch's ranges for them."""
    if stage.dynamic_shapes is None:
        return
    pairs = list(_walk_spec(list(stage.example_inputs), list(stage.dynamic_shapes)))
    if len(pairs) != len(inputs):
        raise CompileError(f"stage {name}: dynamic_shapes covers {len(pairs)} inputs, the graph has {len(inputs)}")
    for i, ((_, spec), node) in enumerate(zip(pairs, inputs)):
        for d, dim in (spec or {}).items():
            v = node.meta["val"].shape[d]
            if not isinstance(v, torch.SymInt):
                raise CompileError(f"stage {name}: dim {d} of input {i} ({node.name}) was declared dynamic but "
                                   f"torch.export specialized it to {v}")
            free = list(v.node.expr.free_symbols)
            if len(free) != 1:
                raise CompileError(f"stage {name}: dim {d} of input {i} is {v.node.expr}, not a function of one Dim")
            root = getattr(dim, "root", dim)
            rng = ep.range_constraints[free[0]]
            try:
                lo, hi = int(rng.lower), int(rng.upper)
            except (TypeError, ValueError, AttributeError, OverflowError):
                lo, hi = 0, None
            if hi is None or hi > 2**31:
                raise CompileError(f"stage {name}: Dim {root.__name__} needs a max")
            dims.map[free[0]] = sym.define(root.__name__, lo, hi)


# --- stage export --------------------------------------------------------------------------


def _decomp_table():
    """Core ATen decompositions, minus ops we lower whole: the fused ones in PRESERVE and any op
    with a registered lowering (so a user lowering for, say, mish sees mish)."""
    table = torch.export.default_decompositions()
    for op in list(table.keys()):
        if op in PRESERVE or op_key(op) in LOWERINGS:
            del table[op]
    return table


class _Missing:
    pass


_MISSING = _Missing()


def _contains(x: Any, cls: type) -> bool:
    return any(isinstance(leaf, cls) for leaf in pytree.tree_leaves(x))


def _unaliased(example_inputs: tuple[Any, ...]) -> tuple[Any, ...]:
    """Copy tensors that appear more than once among the example inputs. torch.export traces one
    tensor passed as two inputs as a single input: the graph would read only the first (six loop
    scalars sharing one zeros(1) example all became the first scalar)."""
    seen: set[int] = set()

    def fresh(x: Any) -> Any:
        if isinstance(x, torch.Tensor):
            if id(x) in seen:
                return x.clone()
            seen.add(id(x))
        return x

    return tuple(pytree.tree_map(fresh, list(example_inputs)))


def export_stage(name: str, stage: api.Stage, root: torch.nn.Module | None, weights: WeightTable) -> Stage:
    fn = stage.fn
    if isinstance(fn, torch.nn.Module) and (root is None or fn is root):
        module, prefix = fn, ""
    else:
        module, prefix = _StageModule(root, fn), "root."

    with torch.no_grad(), warnings.catch_warnings():
        # torch 2.11's own pytree code trips its LeafSpec deprecation inside export.
        warnings.filterwarnings("ignore", message=r".*isinstance\(treespec, LeafSpec\)")
        dynamic = None
        if stage.dynamic_shapes is not None:
            spec = tuple(_mirror_spec(e, sp) for e, sp in zip(stage.example_inputs, stage.dynamic_shapes))
            dynamic = {"args": spec} if prefix else spec
        ep = torch.export.export(module, _unaliased(stage.example_inputs), dynamic_shapes=dynamic)
        ep = ep.run_decompositions(_decomp_table())

    for spec in ep.graph_signature.output_specs:
        if spec.kind.name != "USER_OUTPUT":
            raise CompileError(f"stage {name} mutates state ({spec.kind.name} {spec.target}); stages must be pure")
    specs = {s.arg.name: s for s in ep.graph_signature.input_specs}
    dims = _Dims()
    _define_symbols(name, ep, stage, [n for n in ep.graph.nodes if n.op == "placeholder"
                                      and specs[n.name].kind.name == "USER_INPUT"], dims)

    out = Stage(name)
    b = Builder(out)
    env: dict[torch.fx.Node, Any] = {}
    missing: dict[str, list[str]] = {}

    for node in ep.graph.nodes:
        if node.op == "placeholder":
            spec = specs[node.name]
            kind = spec.kind.name
            if kind == "USER_INPUT":
                val = node.meta["val"]
                env[node] = out.add_param(_io_type(node.name, val, dims, state=val.dtype == torch.float16))
                if node.meta["val"].dtype == torch.bool:
                    b.bools.add(id(env[node]))
            elif kind == "PARAMETER":
                fqn = spec.target[len(prefix):] if spec.target.startswith(prefix) else spec.target
                env[node] = weights.get(fqn, ep.state_dict[spec.target])
            elif kind in ("BUFFER", "CONSTANT_TENSOR"):
                t = ep.state_dict[spec.target] if spec.target in ep.state_dict else ep.constants[spec.target]
                env[node] = t.detach()
            else:
                raise CompileError(f"stage {name}: unsupported input kind {kind} ({node.name})")
        elif node.op == "call_function":
            args, kwargs = torch.fx.node.map_arg((node.args, node.kwargs), lambda n: env[n])
            symbolic = _has_sym((args, kwargs))
            key = op_key(node.target)
            if _contains((args, kwargs), _Missing):
                env[node] = _MISSING
            elif key == "aten.sym_size.int":
                x, d = args
                env[node] = x.shape[d] if isinstance(x, Value) else int(x.shape[d])
            elif symbolic and node.target in _SYM_OPS:
                r = _SYM_OPS[node.target](*args)
                env[node] = sym.norm(r) if isinstance(r, (int, sympy.Expr)) else r
            elif not symbolic and not _contains((args, kwargs), Value):
                with torch.no_grad():
                    env[node] = node.target(*args, **kwargs)  # constant folding
                if key.startswith(("aten.empty", "aten.new_empty")):
                    env[node] = torch.zeros_like(env[node])  # never bake uninitialized memory in
            else:
                if key in _SHAPE_ONLY:
                    leaves = pytree.tree_leaves((args, kwargs))
                    if symbolic or not all(v.type.is_static for v in leaves if isinstance(v, Value)):
                        raise CompileError(f"stage {name}: {key} of a symbolic size is not supported (at {_loc(node)})")
                    env[node] = _fold_shape_only(node, args, kwargs)
                    continue
                vals = [leaf for leaf in pytree.tree_leaves((args, kwargs)) if isinstance(leaf, Value)]
                if vals and not symbolic and all(isinstance(v, Weight) for v in vals):
                    folded = _fold_weights(key, node, args, kwargs, weights)
                    if folded is not None:
                        env[node] = folded
                        continue
                lower = LOWERINGS.get(key)
                if lower is None:
                    missing.setdefault(key, []).append(_loc(node))
                    env[node] = _MISSING
                    continue
                b.loc, b.scope = _loc(node), _scope(node, prefix)
                val = node.meta.get("val")
                b.out_type = TensorType(dims.shape(val, key), "f32") if isinstance(val, torch.Tensor) else None
                res = lower(b, *args, **kwargs)
                if b.out_type is not None and isinstance(res, Value) and res.shape != b.out_type.shape:
                    raise CompileError(f"lowering of {key} produced shape {res.shape}, PyTorch traced "
                                       f"{b.out_type.shape} (at {b.loc})")
                env[node] = res
                if isinstance(res, Value) and isinstance(val, torch.Tensor) and val.dtype == torch.bool:
                    b.bools.add(id(res))
        elif node.op == "output":
            if missing:
                lines = [f"stage {name!r}: {len(missing)} op(s) have no ggml lowering"]
                for key, locs in missing.items():
                    lines.append(f"  {key:<40} {locs[0]}" + (f" (+{len(locs) - 1} more)" if len(locs) > 1 else ""))
                lines.append("Register a lowering with @ggml_compiler.lowering, or change the model.")
                raise CompileError("\n".join(lines))
            results = pytree.tree_leaves(torch.fx.node.map_arg(node.args[0], lambda n: env[n]))
            b.loc = ""
            out.outputs = [r if isinstance(r, Value) else b.const(r) for r in results]
        else:
            raise CompileError(f"stage {name}: unexpected graph node {node.op} {node.target}")
    return out


# --- program -------------------------------------------------------------------------------


def _all_calls(body: list[api.Call | api.Loop]) -> list[api.Call]:
    return [c for op in body for c in (op.body if isinstance(op, api.Loop) else [op])]


def _example_for(program: api.Program, value: str) -> Any:
    """Example of a driver value: a driver's own example if it has one, else the stage's example
    input in the position the value feeds. A loop carry's initial value takes the example of the
    loop value it starts; a slice source, n copies of its loop value's example."""
    drivers = list(program.entries.values())
    for d in drivers:
        if value in d.examples:
            return d.examples[value]
    for d in drivers:
        for c in _all_calls(d.body):
            if value in c.args:
                return program.stages[c.stage].example_inputs[c.args.index(value)]
    for d in drivers:
        for op in d.body:
            if isinstance(op, api.Loop):
                for dst, src in op.carry.items():
                    if src == value:
                        return _example_for(program, dst)
                for dst, src in op.slices.items():
                    if src == value:
                        return torch.stack([_example_for(program, dst)] * op.n)
    raise CompileError(f"driver value {value!r} is never passed to a stage")


def _symbol_example(program: api.Program, name: str) -> int:
    if name not in program.symbols:
        raise CompileError(f"loop count {name!r} is not a symbol of the program (Program(symbols=...))")
    return program.symbols[name][2]


def export_program(program: api.Program, export_context: Any = None) -> tuple[ProgramIR, Reference]:
    """Export every stage. `export_context`, if given, is entered after the PyTorch reference run
    and around the stage exports (ModelOpt import swaps in fake-quant weights there).

    The reference run calls every entry once, in order, carrying the state from one to the next.
    With several entries, reference outputs are keyed "<entry>/<value>"."""
    entries = program.entries
    for d in entries.values():
        for c in _all_calls(d.body):
            if c.stage not in program.stages:
                raise CompileError(f"driver calls unknown stage {c.stage!r}")

    # Run the drivers in PyTorch: reference outputs, and the pytree shape of every value.
    trees: dict[str, Any] = {k: pytree.tree_map(lambda t: t.clone() if isinstance(t, torch.Tensor) else t, v)
                             for k, v in program.state.items()}

    def named(names: list[str]) -> dict[str, torch.Tensor]:
        return {k: runtime_tensor(v) for n in names for k, v in flatten_named(n, trees[n])}

    def run(c: api.Call) -> None:
        res = program.stages[c.stage].fn(*[trees[a] for a in c.args])
        if len(c.out) == 1:
            trees[c.out[0]] = res
        elif isinstance(res, (tuple, list)) and len(res) == len(c.out):
            trees.update(zip(c.out, res))
        else:
            raise CompileError(f"stage {c.stage} does not return {len(c.out)} values")

    ref_in: dict[str, torch.Tensor] = {}
    ref_out: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for ename, d in entries.items():
            for name in d.inputs:
                trees[name] = _example_for(program, name)
            ref_in.update(named(d.inputs))
            for op in d.body:
                if isinstance(op, api.Call):
                    run(op)
                    continue
                n = op.n if isinstance(op.n, int) else _symbol_example(program, op.n)
                for dst, src in op.carry.items():
                    trees[dst] = trees[src]
                for name, table in op.scalars.items():
                    if len(table) != n:
                        raise CompileError(f"loop scalar {name} has {len(table)} entries for {op.n} iterations")
                for dst, src in op.slices.items():
                    if not isinstance(trees[src], torch.Tensor) or trees[src].shape[:1] != (n,):
                        raise CompileError(f"loop slice source {src} must be one tensor with a leading dim of {op.n}")
                starts = {c: int(trees[src].reshape(-1)[0]) for c, src in op.counters.items()}
                rows: dict[str, list[torch.Tensor]] = {dst: [] for dst in op.collect}
                for i in range(n):
                    for name, table in op.scalars.items():
                        trees[name] = torch.full_like(_example_for(program, name), table[i])
                    for dst, src in op.slices.items():
                        trees[dst] = trees[src][i]
                    for c, s0 in starts.items():
                        trees[c] = torch.tensor([s0 + i])  # int64, as PyTorch indexes; the runtime keeps i32
                    for m, c in op.masks.items():
                        shape = _example_for(program, m).shape
                        row = torch.where(torch.arange(shape[-1]) <= starts[c] + i, 0.0, float("-inf"))
                        trees[m] = row.expand(shape).contiguous()
                    for c in op.body:
                        run(c)
                    for dst, src in op.collect.items():
                        rows[dst].append(trees[src].detach().clone())
                    if any(int(trees[v].reshape(-1)[0]) in ids for v, ids in op.until.items()):
                        break
                for dst, src in op.collect.items():
                    out = torch.zeros((n, *rows[dst][0].shape), dtype=rows[dst][0].dtype)
                    out[: len(rows[dst])] = torch.stack(rows[dst])
                    trees[dst] = out
            outs = named(d.outputs)
            ref_out.update({f"{ename}/{k}" if len(entries) > 1 else k: v for k, v in outs.items()})

    prog = ProgramIR(entries={})
    weights = WeightTable(copy=export_context is not None)
    sym.RANGES.clear()  # symbols are named per program
    for name, (lo, hi, _) in program.symbols.items():
        sym.define(name, lo, hi)
    with export_context if export_context is not None else contextlib.nullcontext():
        for name, stage in program.stages.items():
            prog.stages[name] = export_stage(name, stage, program.root, weights)
    prog.weights = weights.by_fqn
    prog.symbols = dict(sym.RANGES)

    # A driver value's type comes from the stage signature it meets first (that keeps symbolic
    # dims); the reference run only has the example's sizes.
    def names_of(n: str) -> list[str]:
        return [leaf_name for leaf_name, _ in flatten_named(n, trees[n])]

    sig: dict[str, TensorType] = {}
    for d in entries.values():
        for c in _all_calls(d.body):
            st = prog.stages[c.stage]
            for leaf_name, p in zip([x for a in c.args for x in names_of(a)], st.params):
                sig.setdefault(leaf_name, p.type)
            for leaf_name, v in zip([x for o in c.out for x in names_of(o)], st.outputs):
                sig.setdefault(leaf_name, v.type)

    def leaves(names: list[str]) -> list[str]:
        out = []
        for n in names:
            for leaf_name, leaf in flatten_named(n, trees[n]):
                t = _io_type(leaf_name, leaf, state=n in program.state)
                st = sig.get(leaf_name)
                prog.values[leaf_name] = st if st is not None and st.dtype == t.dtype else t
                out.append(leaf_name)
        return out

    prog.state = leaves(list(program.state))
    for ename, d in entries.items():
        entry = prog.entries[ename] = Entry(inputs=leaves(d.inputs))
        for op in d.body:
            if isinstance(op, api.Call):
                entry.driver.append(CallOp(op.stage, leaves(op.args), leaves(op.out)))
                continue
            body = [CallOp(c.stage, leaves(c.args), leaves(c.out)) for c in op.body]
            carry = []
            for dst, src in op.carry.items():
                carry += list(zip(leaves([dst]), leaves([src])))
            scalars = []
            for name, table in op.scalars.items():
                names = leaves([name])
                if names != [name]:
                    raise CompileError(f"loop scalar {name} must be a single tensor")
                scalars.append((name, [float(v) for v in table]))
            slices = []
            for dst, src in op.slices.items():
                names = leaves([dst])
                if names != [dst]:
                    raise CompileError(f"loop slice {dst} must be a single tensor")
                slices.append((dst, leaves([src])[0]))
            single = lambda v: leaves([v])[0]  # noqa: E731  (counters, masks, ... are single tensors)
            counters = [(single(c), single(src)) for c, src in op.counters.items()]
            bind = []
            for name, fn in op.bind.items():
                if name not in sym.RANGES:
                    raise CompileError(f"loop binds unknown symbol {name!r}")
                expr = fn(**{c: sym.sympy.Symbol(c, integer=True, nonnegative=True) for c in op.counters})
                bind.append((name, sym.to_json(sym.norm(expr))))
            masks = [(single(m), c) for m, c in op.masks.items()]
            until = [(single(v), list(ids)) for v, ids in op.until.items()]
            collect = [(dst, single(src)) for dst, src in op.collect.items()]
            n = op.n if isinstance(op.n, int) else sym.symbol(op.n)
            for dst, src in collect:  # n rows of the loop value: symbolic when n is
                prog.values[dst] = TensorType((n, *prog.values[src].shape), prog.values[src].dtype)
            entry.driver.append(LoopOp(op.n, body, carry, scalars, slices, counters, bind, masks, until, collect))
        collected = {dst: prog.values[dst] for op in entry.driver if isinstance(op, LoopOp) for dst, _ in op.collect}
        entry.outputs = leaves(d.outputs)
        prog.values.update(collected)
    prog.check()
    return prog, Reference(ref_in, ref_out)
