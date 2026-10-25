"""Additive and XOR secret sharing (paper, Section 2.2, sharing and products).

``share`` / ``reconstruct`` work over Z_{2^64}; ``share_mod`` / ``reconstruct_mod``
over Z_t for an odd modulus t < 2^62 (values held as uint64 below t);
``share_bits`` / ``reconstruct_bits`` are XOR shares of bits held as uint8.
``share_k`` / ``reconstruct_k`` are the Z_{2^k} variants used by the
small-ring exhaustive tests of trunc.py and ring_to_field.py.

What is tested (tests/test_sharing.py): reconstruction identities, uniformity
sanity of the first share (not a statistical proof), value-range refusal.
Design limit: the randomness source is the caller's ``rng`` (anything with a
numpy-style ``integers`` method). Shares of secrets must be drawn from
``csprng.SecureRandom``; a numpy Generator is a statistical generator whose
state is recoverable from outputs and is for deterministic fixtures only.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from ..errors import ContractViolation
from . import ring


def share(x: Any, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform additive shares (s0, s1) with s0 + s1 = x mod 2^64."""
    x = np.asarray(x, dtype=np.uint64)
    s0 = ring.rand_u64(x.shape, rng)
    s1 = ring.sub(x, s0)
    return s0, s1


def reconstruct(s0: Any, s1: Any) -> np.ndarray:
    """s0 + s1 mod 2^64."""
    return ring.add(s0, s1)


def share_k(x: Any, k: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform additive shares over Z_{2^k} (k <= 64), held as uint64 below 2^k."""
    x = ring.reduce_k(x, k)
    s0 = ring.reduce_k(ring.rand_u64(x.shape, rng), k)
    s1 = ring.reduce_k(ring.sub(x, s0), k)
    return s0, s1


def reconstruct_k(s0: Any, s1: Any, k: int) -> np.ndarray:
    """(s0 + s1) mod 2^k."""
    return ring.reduce_k(ring.add(s0, s1), k)


def check_modulus(t: int) -> int:
    """Validate an odd modulus 3 <= t < 2^62 (primality is checked by ring_to_field)."""
    t = int(t)
    if t < 3 or t >= (1 << 62) or t % 2 == 0:
        raise ContractViolation(f"SHARING_MODULUS: t must be odd with 3 <= t < 2^62, got {t}")
    return t


def share_mod(x: Any, t: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform additive shares over Z_t of values x (uint64, all < t)."""
    t = check_modulus(t)
    x = np.asarray(x, dtype=np.uint64)
    if np.any(x >= np.uint64(t)):
        raise ContractViolation("SHARING_RANGE: share_mod input must be below t")
    s0 = rng.integers(0, t, size=x.shape, dtype=np.uint64)
    tt = np.uint64(t)
    # x + t - s0 < 2^63, no wrap
    s1 = ((x + tt) - s0) % tt
    return s0, s1.astype(np.uint64)


def reconstruct_mod(s0: Any, s1: Any, t: int) -> np.ndarray:
    """(s0 + s1) mod t for shares below t."""
    t = check_modulus(t)
    tt = np.uint64(t)
    a = np.asarray(s0, dtype=np.uint64)
    b = np.asarray(s1, dtype=np.uint64)
    return ((a % tt) + (b % tt)) % tt


def share_bits(b: Any, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """XOR shares of bits (uint8 arrays with values in {0,1})."""
    b = np.asarray(b).astype(np.uint8) & np.uint8(1)
    s0 = rng.integers(0, 2, size=b.shape, dtype=np.uint8)
    s1 = (b ^ s0).astype(np.uint8)
    return s0, s1


def reconstruct_bits(s0: Any, s1: Any) -> np.ndarray:
    """XOR reconstruction of bit shares."""
    return (np.asarray(s0, dtype=np.uint8) ^ np.asarray(s1, dtype=np.uint8)) & np.uint8(1)


__all__ = [
    "share", "reconstruct", "share_k", "reconstruct_k", "check_modulus",
    "share_mod", "reconstruct_mod", "share_bits", "reconstruct_bits",
]
