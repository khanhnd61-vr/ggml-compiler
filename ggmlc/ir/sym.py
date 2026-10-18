"""Symbolic dimensions: sizes known only at run time, within a range fixed at compile time.

A dim is an int or a sympy expression over named positive integer symbols (`seq`, `n_kv`). The
program records each symbol's range; the runtime binds every symbol for a call, evaluates the
expressions and builds the stage graphs for that binding.

Python's `==` / `!=` on sympy expressions are structural, which is what shape checks want: `s` is
never "equal to 1" or to `s + 1`. Order comparisons need the ranges: `definitely` decides a
relation for every binding in range, by evaluating it at the corners of the symbols' ranges
(exact for the multilinear expressions shapes are made of).
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Iterable, Union

import sympy

Dim = Union[int, sympy.Expr]

# name -> (min, max) for every symbol defined by the compile in progress.
RANGES: dict[str, tuple[int, int]] = {}


def symbol(name: str) -> sympy.Symbol:
    return sympy.Symbol(name, integer=True, positive=True)


def define(name: str, lo: int, hi: int) -> sympy.Symbol:
    if not 1 <= lo <= hi:
        raise ValueError(f"symbol {name}: bad range [{lo}, {hi}]")
    if name in RANGES and RANGES[name] != (lo, hi):
        raise ValueError(f"symbol {name} defined twice, as {RANGES[name]} and {(lo, hi)}")
    RANGES[name] = (lo, hi)
    return symbol(name)


def norm(d: Any) -> Dim:
    """An int, or a sympy expression in canonical (expanded) form."""
    if isinstance(d, bool):
        raise TypeError("a dim cannot be a bool")
    if isinstance(d, int):
        return d
    if isinstance(d, sympy.Basic):
        if d.is_Integer:
            return int(d)
        d = sympy.expand(d)
        return int(d) if d.is_Integer else d
    return int(d)  # numpy and torch integers


def is_static(d: Any) -> bool:
    return isinstance(d, int)


def static(dims: Iterable[Any]) -> bool:
    return all(isinstance(d, int) for d in dims)


def names(d: Any) -> set[str]:
    return {s.name for s in d.free_symbols} if isinstance(d, sympy.Basic) else set()


def free(x: Any) -> set[str]:
    """Every symbol in a dim, a shape, or an attribute value (lists and dicts are walked)."""
    if isinstance(x, sympy.Basic):
        return names(x)
    if isinstance(x, (list, tuple)):
        return set().union(*(free(v) for v in x)) if x else set()
    if isinstance(x, dict):
        return set().union(*(free(v) for v in x.values())) if x else set()
    return set()


def ev(d: Any, binding: dict[str, int]) -> Any:
    """d with every symbol replaced by its bound value. Lists, tuples and dicts are walked;
    anything that is not a dim is returned as is."""
    if isinstance(d, sympy.Basic):
        v = d.subs({symbol(k): binding[k] for k in names(d)})
        if not v.is_Integer:
            raise ValueError(f"{d} does not evaluate to an integer for {binding}")
        return int(v)
    if isinstance(d, list):
        return [ev(v, binding) for v in d]
    if isinstance(d, tuple):
        return tuple(ev(v, binding) for v in d)
    if isinstance(d, dict):
        return {k: ev(v, binding) for k, v in d.items()}
    return d


def _corners(syms: set[str]) -> list[dict[str, int]]:
    missing = [s for s in syms if s not in RANGES]
    if missing:
        raise ValueError(f"symbol(s) {', '.join(sorted(missing))} have no range")
    names_ = sorted(syms)
    return [dict(zip(names_, c)) for c in itertools.product(*(RANGES[s] for s in names_))]


def bounds(d: Any) -> tuple[int, int]:
    """The smallest and largest value d takes over the symbols' ranges."""
    if isinstance(d, int):
        return d, d
    vals = [ev(d, c) for c in _corners(names(d))]
    return min(vals), max(vals)


def upper(d: Any) -> int:
    return bounds(d)[1]


def definitely(rel: Any) -> bool:
    """True if the relation holds for every binding in range. Plain bools pass through."""
    if isinstance(rel, (bool, sympy.logic.boolalg.BooleanAtom)):
        return bool(rel)
    syms = free(rel)
    if not syms:
        return bool(rel)
    return all(bool(rel.subs({symbol(k): v for k, v in c.items()})) for c in _corners(syms))


def divides(a: Any, b: Any) -> bool:
    """a divides b for every binding (structurally: b / a simplifies to an integer expression)."""
    if isinstance(a, int) and isinstance(b, int):
        return a != 0 and b % a == 0
    q = sympy.simplify(sympy.sympify(b) / sympy.sympify(a))
    return bool(q.is_integer)


def prod(dims: Iterable[Any]) -> Dim:
    return norm(math.prod(dims)) if dims else 1


# --- serialization --------------------------------------------------------------------------
# A dim in the program JSON: an int, a symbol name, or [op, operand...] with op one of
# "+", "*", "//" (floor division), "%", "max", "min". The runtime and generated C++ evaluate
# the same trees.


def to_json(d: Any) -> Any:
    if isinstance(d, int):
        return d
    if not isinstance(d, sympy.Basic):
        raise TypeError(f"not a dim: {d!r}")
    if d.is_Integer:
        return int(d)
    if d.is_Symbol:
        return d.name
    name = type(d).__name__
    if d.is_Add:
        return ["+", *(to_json(a) for a in d.args)]
    if d.is_Mul:
        num, den = sympy.fraction(d)
        if den != 1:
            raise ValueError(f"{d} is not an integer expression")
        return ["*", *(to_json(a) for a in d.args)]
    if name in ("FloorDiv", "CleanDiv", "IntTrueDiv"):
        return ["//", to_json(d.args[0]), to_json(d.args[1])]
    if name in ("Mod", "PythonMod"):
        return ["%", to_json(d.args[0]), to_json(d.args[1])]
    if name == "floor":
        num, den = sympy.fraction(sympy.together(d.args[0]))
        return ["//", to_json(norm(num)), to_json(norm(den))]
    if name == "ceiling":
        num, den = sympy.fraction(sympy.together(d.args[0]))
        return ["//", to_json(norm(num + den - 1)), to_json(norm(den))]
    if name in ("Max", "Min"):
        return [name.lower(), *(to_json(a) for a in d.args)]
    if d.is_Pow and d.args[1].is_Integer and int(d.args[1]) > 0:
        return ["*", *([to_json(d.args[0])] * int(d.args[1]))]
    raise ValueError(f"cannot serialize dim {d} ({name})")


def from_json(j: Any) -> Dim:
    if isinstance(j, int):
        return j
    if isinstance(j, str):
        return symbol(j)
    op, *args = j
    xs = [sympy.sympify(from_json(a)) for a in args]
    if op == "+":
        return norm(sympy.Add(*xs))
    if op == "*":
        return norm(sympy.Mul(*xs))
    if op == "//":
        return norm(sympy.floor(xs[0] / xs[1]))
    if op == "%":
        return norm(sympy.Mod(xs[0], xs[1]))
    if op in ("max", "min"):
        return norm((sympy.Max if op == "max" else sympy.Min)(*xs))
    raise ValueError(f"unknown dim op {op!r}")


def smin(a: Any, b: Any) -> Dim:
    if definitely(a <= b):
        return a
    if definitely(b <= a):
        return b
    return norm(sympy.Min(a, b))


def smax(a: Any, b: Any) -> Dim:
    if definitely(a >= b):
        return a
    if definitely(b >= a):
        return b
    return norm(sympy.Max(a, b))


def broadcast(a: tuple, b: tuple) -> tuple:
    """PyTorch broadcasting of two shapes whose dims may be symbolic."""
    n = max(len(a), len(b))
    a, b = (1,) * (n - len(a)) + tuple(a), (1,) * (n - len(b)) + tuple(b)
    out = []
    for x, y in zip(a, b):
        if x == y or y == 1:
            out.append(x)
        elif x == 1:
            out.append(y)
        else:
            raise ValueError(f"shapes {a} and {b} do not broadcast")
    return tuple(out)


def solve(pairs: Iterable[tuple[tuple, tuple]]) -> dict[str, int]:
    """Symbol values that make each symbolic shape equal its concrete one: [(dims, sizes), ...].
    Every symbol must appear alone (times a constant, plus a constant) in some dim, or in a dim
    whose other symbols are already solved."""
    binding: dict[str, int] = {}
    pending = [(d, int(v)) for dims, sizes in pairs if len(dims) == len(sizes)
               for d, v in zip(dims, sizes) if not is_static(d)]
    progress = True
    while progress:
        progress = False
        for d, v in pending:
            left = names(d) - binding.keys()
            if len(left) != 1:
                continue
            (name,) = left
            sol = sympy.solve(sympy.Eq(d.subs({symbol(k): x for k, x in binding.items()}), v), symbol(name))
            sol = [x for x in sol if x.is_Integer]
            if not sol:
                raise ValueError(f"no integer {name} gives {d} = {v}")
            binding[name] = int(sol[0])
            progress = True
    for d, v in pending:
        if names(d) - binding.keys():
            raise ValueError(f"cannot bind the symbols of {d} from the shapes")
        if ev(d, binding) != v:
            raise ValueError(f"dim {d} is {ev(d, binding)} for {binding}, the shape has {v}")
    for name, v in binding.items():
        lo, hi = RANGES.get(name, (v, v))
        if not lo <= v <= hi:
            raise ValueError(f"{name} = {v} is outside its compiled range [{lo}, {hi}]")
    return binding
