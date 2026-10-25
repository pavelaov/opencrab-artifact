"""Polynomial approximation contracts for the secure non-linear operators.

Every clipped activation and every normalised reciprocal in the gadget layer
evaluates a polynomial on a normalised argument u inside a declared range (the
approximation contract). This module computes the coefficients
deterministically by least squares on Chebyshev nodes (a near-minimax fit),
reports the maximum absolute error of each fit on a dense grid, and exposes
the contracts that `NumericContract` names and the planner consults.

Conventions: for a symmetric range [-R, R] the variable is u = x / R; for an
affine range [lo, hi] it is u = 2 (x - lo) / (hi - lo) - 1. Coefficients are
ascending (c[0] + c[1] u + ...). An even fit stores coefficients in u^2
(`even_in_u`); an odd fit stores the full ascending list with zeros at even
indices (`odd_in_u`), so Horner evaluation never needs a special case.

Named contracts (NC-v2): gelu_clip12 (erf-GELU, R=5, degree 12), silu_clip12
(R=8, degree 12), tanh_clip (R=4, degree 9, odd), recip_norm (degree 5 on
[0.5, 1), affine), rsqrt_norm (degree 5 on [0.25, 1), affine).

What is tested (tests/test_approx.py): gelu_clip12 max_abs_err < 5e-3 (the
contract records 4.84e-3), every named contract reproduces its function on a
dense grid within its recorded error, odd/even structure, determinism of two
independent builds, and refusal of unknown names.

Design limits: errors are measured on a 20001-point grid, not proven by a
minimax certificate; the fixed-point evaluation inside the gadget adds its own
truncation error that the gadget layer accounts for separately. Error messages
start with the stable codes APPROX_PARAM and APPROX_UNKNOWN_CONTRACT.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Callable, Dict, List

import numpy as np
from scipy.special import erf


def _gelu(x):
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _recip(v):
    return 1.0 / v


def _rsqrt(v):
    return 1.0 / np.sqrt(v)


@dataclass
class PolyContract:
    """A fitted polynomial with its validity range and measured error."""

    name: str
    coeffs: List[float]          # ascending powers of the fitted variable
    lo: float                    # valid input range [lo, hi] in x units
    hi: float
    scale: float                 # u = x / scale when not affine
    even_in_u: bool              # coefficients are in u^2
    max_abs_err: float           # on a dense grid inside [lo, hi]
    degree: int
    affine: bool = False         # u = 2 (x - lo) / (hi - lo) - 1
    odd_in_u: bool = False       # even-index coefficients are exactly zero

    def to_dict(self) -> dict:
        return asdict(self)

    def normalise(self, x):
        """Map x in [lo, hi] to the polynomial variable u."""
        x = np.asarray(x, dtype=np.float64)
        if self.affine:
            return 2.0 * (x - self.lo) / (self.hi - self.lo) - 1.0
        return x / self.scale

    def evaluate(self, x):
        """Plaintext reference evaluation of the fitted polynomial at x (float64)."""
        u = self.normalise(x)
        var = u * u if self.even_in_u else u
        return horner(self.coeffs, var)


def horner(coeffs: List[float], v):
    """Ascending-coefficient Horner evaluation (reference for the secure poly_horner)."""
    v = np.asarray(v, dtype=np.float64)
    acc = np.zeros_like(v) + float(coeffs[-1])
    for c in reversed(coeffs[:-1]):
        acc = acc * v + float(c)
    return acc


def _cheb_nodes(n: int) -> np.ndarray:
    k = np.arange(n)
    return np.cos((2 * k + 1) * np.pi / (2 * n))


def fit(name: str, f: Callable, lo: float, hi: float, degree: int,
        even: bool = False, affine: bool = False, odd: bool = False,
        nodes: int = 4000, grid: int = 20001) -> PolyContract:
    """Least-squares fit of f on [lo, hi] in the normalised variable.

    Non-affine fits require a symmetric range [-R, R] or a one-sided range
    [0, hi]; any other range must be fitted with affine=True.
    """
    if even and odd:
        raise ValueError("APPROX_PARAM: even and odd are mutually exclusive")
    if not isinstance(degree, int) or degree < 0:
        raise ValueError(f"APPROX_PARAM: degree must be a non-negative int, got {degree!r}")
    if odd and degree % 2 == 0:
        raise ValueError(f"APPROX_PARAM: an odd fit needs an odd degree, got {degree}")
    if not (hi > lo):
        raise ValueError(f"APPROX_PARAM: need hi > lo, got lo={lo}, hi={hi}")
    if nodes < degree + 1:
        raise ValueError("APPROX_PARAM: fewer Chebyshev nodes than coefficients")
    lo = float(lo)
    hi = float(hi)
    if affine:
        u = _cheb_nodes(nodes)
        x = lo + (u + 1.0) * 0.5 * (hi - lo)
        scale = 1.0
    else:
        scale = max(abs(lo), abs(hi))
        if lo >= 0.0:
            u = (_cheb_nodes(nodes) + 1.0) * 0.5 * (hi / scale)
        elif math.isclose(-lo, hi):
            u = _cheb_nodes(nodes)
        else:
            raise ValueError("APPROX_PARAM: non-affine fits need a symmetric or one-sided range")
        x = u * scale
    y = np.asarray(f(x), dtype=np.float64)
    if even:
        powers = list(range(0, degree + 1, 2))
        var = u * u
        V = np.vander(var, len(powers), increasing=True)
    elif odd:
        powers = list(range(1, degree + 1, 2))
        V = np.stack([u ** p for p in powers], axis=1)
    else:
        powers = list(range(0, degree + 1))
        V = np.vander(u, degree + 1, increasing=True)
    c, *_ = np.linalg.lstsq(V, y, rcond=None)
    if even:
        coeffs = [float(v) for v in c]
    elif odd:
        full = [0.0] * (degree + 1)
        for p, v in zip(powers, c):
            full[p] = float(v)
        coeffs = full
    else:
        coeffs = [float(v) for v in c]
    pc = PolyContract(name, coeffs, lo, hi, float(scale), bool(even), 0.0, int(degree),
                      affine=bool(affine), odd_in_u=bool(odd))
    xg = np.linspace(lo, hi, grid)
    err = float(np.max(np.abs(pc.evaluate(xg) - np.asarray(f(xg), dtype=np.float64))))
    pc.max_abs_err = err
    return pc


_SPECS: Dict[str, dict] = {
    "gelu_clip12": dict(f=_gelu, lo=-5.0, hi=5.0, degree=12),
    "silu_clip12": dict(f=_silu, lo=-8.0, hi=8.0, degree=12),
    "tanh_clip": dict(f=np.tanh, lo=-4.0, hi=4.0, degree=9, odd=True),
    "recip_norm": dict(f=_recip, lo=0.5, hi=1.0, degree=5, affine=True),
    "rsqrt_norm": dict(f=_rsqrt, lo=0.25, hi=1.0, degree=5, affine=True),
}

CONTRACT_NAMES = tuple(_SPECS.keys())


@lru_cache(maxsize=None)
def build_table() -> Dict[str, PolyContract]:
    """All named contracts (deterministic: rebuilding yields identical numbers)."""
    return {name: fit(name, **spec) for name, spec in _SPECS.items()}


def contract(name: str) -> PolyContract:
    """Named contract: gelu_clip12, silu_clip12, tanh_clip, recip_norm, rsqrt_norm."""
    table = build_table()
    if name not in table:
        raise ValueError(f"APPROX_UNKNOWN_CONTRACT: {name!r}; known: {', '.join(CONTRACT_NAMES)}")
    return table[name]


def contract_table_json() -> dict:
    """Serialisable view of every contract (name -> dict including max_abs_err)."""
    return {k: v.to_dict() for k, v in build_table().items()}


def dump_table(path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(contract_table_json(), fh, indent=1, sort_keys=True)


if __name__ == "__main__":  # pragma: no cover
    for k, v in build_table().items():
        print(f"{k:12s} deg={v.degree:2d} range=[{v.lo:g},{v.hi:g}] max_abs_err={v.max_abs_err:.3e}")
