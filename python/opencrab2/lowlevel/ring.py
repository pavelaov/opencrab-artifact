"""Z_{2^64} ring arithmetic and fixed-point encoding (paper, Section 2.1, transformer operators and fixed point).

All ring elements are ``numpy.uint64`` arrays interpreted as two's complement
integers modulo 2^64. This module is the single source of truth for
``encode`` / ``decode`` / ``to_signed`` so that every protocol module agrees
on the numeric semantics of ``NumericContract`` (``highlevel/contract.py``).

Generic small rings Z_{2^k} (k <= 64) are supported by ``reduce_k`` and
``to_signed_k``; the faithful truncation and ring-to-field gates use them so
that exhaustive tests at k <= 8 and the native k = 64 path share one code
path.

What is tested (tests/test_ring.py): wrap-around of add/sub/mul/neg, signed
views, msb, arithmetic shift, encode saturation (raises ``ContractViolation``,
no clamp), decode round trip, ``ring_matmul`` against a Python-int reference
and the integer path against the 16-bit-limb BLAS path. Design limit: the
limb matmul requires the shared dimension of each chunk to stay below 2^21 so
that float64 partial sums are exact; the function chunks automatically.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..errors import ContractViolation

MASK64: np.uint64 = np.uint64(0xFFFFFFFFFFFFFFFF)
_ONE = np.uint64(1)
_ZERO = np.uint64(0)
_SIGN_BIT = np.uint64(1) << np.uint64(63)

# Below this many multiply-adds the direct (wrapping) numpy integer matmul is
# used; above it the 16-bit limb float64 BLAS path.
_SMALL_MATMUL_WORK = 1 << 18
_LIMB_CHUNK_K = 1 << 20   # 2^20 * 2^32 < 2^53: float64 partial sums stay exact


def _u(a: Any) -> np.ndarray:
    return np.asarray(a, dtype=np.uint64)


def add(a: Any, b: Any) -> np.ndarray:
    """(a + b) mod 2^64."""
    with np.errstate(over="ignore"):
        return (_u(a) + _u(b)).astype(np.uint64)


def sub(a: Any, b: Any) -> np.ndarray:
    """(a - b) mod 2^64."""
    with np.errstate(over="ignore"):
        return (_u(a) - _u(b)).astype(np.uint64)


def mul(a: Any, b: Any) -> np.ndarray:
    """(a * b) mod 2^64 (numpy uint64 multiply wraps exactly)."""
    with np.errstate(over="ignore"):
        return (_u(a) * _u(b)).astype(np.uint64)


def neg(a: Any) -> np.ndarray:
    """(-a) mod 2^64."""
    with np.errstate(over="ignore"):
        return (_ZERO - _u(a)).astype(np.uint64)


def to_signed(v: Any) -> np.ndarray:
    """Reinterpret uint64 bits as int64 (two's complement view)."""
    return np.ascontiguousarray(_u(v)).view(np.int64)


def from_signed(v: Any) -> np.ndarray:
    """Reinterpret int64 as uint64 (two's complement view)."""
    return np.ascontiguousarray(np.asarray(v, dtype=np.int64)).view(np.uint64)


def msb(v: Any) -> np.ndarray:
    """Most significant bit of each element as uint64 in {0, 1}."""
    return ((_u(v) & _SIGN_BIT) >> np.uint64(63)).astype(np.uint64)


def arith_shift_right(v: Any, bits: int) -> np.ndarray:
    """Plaintext helper: floor(signed(v) / 2^bits) as a ring element."""
    bits = int(bits)
    if bits < 0 or bits > 63:
        raise ContractViolation(f"RING_SHIFT: bits must be in [0, 63], got {bits}")
    return from_signed(to_signed(v) >> np.int64(bits))


def encode(x: Any, contract: Any) -> np.ndarray:
    """rint(x * 2^f) mod 2^64; raises ContractViolation if |rint(x*2^f)| >= 2^(l-1).

    ``contract`` needs ``frac_bits`` and ``logical_bits`` attributes
    (``opencrab2.highlevel.contract.NumericContract``). Non-finite inputs are
    refused as well: there is no silent saturation anywhere.
    """
    f = int(contract.frac_bits)
    l = int(contract.logical_bits)
    xf = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(xf)):
        raise ContractViolation("CONTRACT_VIOLATION: encode received a non-finite value")
    scaled = np.rint(xf * float(2 ** f))
    bound = float(2 ** (l - 1))
    if np.any(np.abs(scaled) >= bound):
        worst = float(np.max(np.abs(scaled)))
        raise ContractViolation(
            f"CONTRACT_VIOLATION: |rint(x*2^{f})| = {worst:.0f} >= 2^{l - 1}; saturation is refused")
    return from_signed(scaled.astype(np.int64))


def decode(v: Any, contract: Any) -> np.ndarray:
    """signed(v) / 2^f as float64."""
    f = int(contract.frac_bits)
    return to_signed(v).astype(np.float64) / float(2 ** f)


def rand_u64(shape: Any, rng: Any) -> np.ndarray:
    """Uniform ring elements from ``rng.integers`` (``csprng.SecureRandom`` for every secret; numpy
    generators only in deterministic fixtures of insecure dealers)."""
    return rng.integers(0, 1 << 64, size=shape, dtype=np.uint64)


# ---------------------------------------------------------------------------
# Generic Z_{2^k} helpers (k <= 64) shared by trunc.py and ring_to_field.py
# ---------------------------------------------------------------------------
def check_k(k: int) -> int:
    k = int(k)
    if k < 1 or k > 64:
        raise ContractViolation(f"RING_BITS: k must be in [1, 64], got {k}")
    return k


def mask_k(k: int) -> np.uint64:
    """Bit mask 2^k - 1 as uint64 (all ones for k = 64)."""
    k = check_k(k)
    return MASK64 if k == 64 else np.uint64((1 << k) - 1)


def reduce_k(v: Any, k: int) -> np.ndarray:
    """v mod 2^k on uint64 arrays."""
    k = check_k(k)
    v = _u(v)
    return v if k == 64 else (v & mask_k(k)).astype(np.uint64)


def to_signed_k(v: Any, k: int) -> np.ndarray:
    """Representative of v mod 2^k in [-2^(k-1), 2^(k-1)) as int64 (k <= 64)."""
    k = check_k(k)
    if k == 64:
        return to_signed(v)
    v = reduce_k(v, k).astype(np.int64)
    half = np.int64(1 << (k - 1))
    return np.where(v >= half, v - np.int64(1 << k), v).astype(np.int64)


def from_signed_k(v: Any, k: int) -> np.ndarray:
    """Signed int64 values to their residue in [0, 2^k) as uint64."""
    return reduce_k(from_signed(np.asarray(v, dtype=np.int64)), k)


# ---------------------------------------------------------------------------
# Exact matrix multiplication mod 2^64
# ---------------------------------------------------------------------------
def _limbs16(A: np.ndarray) -> list:
    out = []
    for i in range(4):
        out.append(((A >> np.uint64(16 * i)) & np.uint64(0xFFFF)).astype(np.float64))
    return out


def _ring_matmul_limbs(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """16-bit limb decomposition, float64 BLAS products, exact recombination mod 2^64."""
    m, k = A.shape
    n = B.shape[1]
    out = np.zeros((m, n), np.uint64)
    al = _limbs16(A)
    bl = _limbs16(B)
    with np.errstate(over="ignore"):
        for i in range(4):
            for j in range(4 - i):
                acc = np.zeros((m, n), np.uint64)
                for c0 in range(0, k, _LIMB_CHUNK_K):
                    c1 = min(k, c0 + _LIMB_CHUNK_K)
                    prod = al[i][:, c0:c1] @ bl[j][c0:c1, :]
                    acc += prod.astype(np.uint64)
                out += acc << np.uint64(16 * (i + j))
    return out


def ring_matmul(A: Any, B: Any) -> np.ndarray:
    """A @ B exactly modulo 2^64 for uint64 matrices (BLAS limb path when large)."""
    A = _u(A)
    B = _u(B)
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[0]:
        raise ContractViolation(
            f"RING_SHAPE: ring_matmul needs (m,k) @ (k,n), got {A.shape} @ {B.shape}")
    m, k = A.shape
    n = B.shape[1]
    if k == 0:
        return np.zeros((m, n), np.uint64)
    if m * k * n <= _SMALL_MATMUL_WORK:
        with np.errstate(over="ignore"):
            return (A @ B).astype(np.uint64)
    return _ring_matmul_limbs(np.ascontiguousarray(A), np.ascontiguousarray(B))


__all__ = [
    "MASK64", "add", "sub", "mul", "neg", "to_signed", "from_signed", "msb",
    "arith_shift_right", "encode", "decode", "rand_u64", "ring_matmul",
    "check_k", "mask_k", "reduce_k", "to_signed_k", "from_signed_k",
]
