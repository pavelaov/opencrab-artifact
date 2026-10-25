"""A64 -> Z_t resharing with the signed carry (paper, Section 5.2, RingToField).

Inputs: shares (a0, a1) of u over q = 2^k (k = 64 natively; k <= 64 for the
exhaustive small-ring tests); odd prime t < 2^62. Output: fresh shares over
Z_t of z = x_signed mod t with x_signed = u - 1{u >= q/2} q, the
representative in [-q/2, q/2).

Why a protocol is needed: reducing both shares mod t and adding leaves
the unwanted term (w + b) q mod t, where w is the wrap of a0 + a1 over q and
b the sign bit; the condition |x_signed| < t/2 does not remove it.
Counterexample: q = 256, t = 17, a0 = 255, a1 = 2 gives u = 1 and z = 1, the
naive share-wise sum gives 2.

Dealer per element: uniform r in Z_q; Z_q shares of r; Z_t shares of
(r mod t); DCF key A: alpha = r, n = k, payload group Z_t, beta = 1; DCF key
C: alpha = (r + 2^(k-1)) mod q, n = k, payload Z_t, beta = 1; Z_t shares of
the constant c = 1{r > (r + 2^(k-1)) mod q} (the 1{p > q'} of the DReLU
identity).

Online (one round):

    1. open y = u + [r] mod q
    2. [w]_t  = DCF_A(y)                         (= 1{y < r}, the wrap of y - r)
    3. [nn]_t = DCF_C(y) - DCF_A(y) + [c]_t      (= 1{u < q/2}, the DReLU identity with n = k)
       [b]_t  = 1 - [nn]_t                       (MSB of u; P0 adds the 1)
    4. [u mod t]_t = (y mod t) 1{party 0} - [r mod t]_t + (q mod t) [w]_t
    5. [z]_t = [u mod t]_t - (q mod t) [b]_t      (mod t)

Nothing but y is communicated; w and b stay shared. Z_t scalar products use
a double-and-add ladder so that every intermediate fits in uint64 for
t < 2^62.

Serialization: 48-byte header ``OC2R2F01``, the three share arrays, then the
two Z_t DCF key batches in the binding DCF wire layout
(``DCFKeyBatch.to_bytes``). ``nbytes()`` is
the charged material size (3272 bytes per element); ``wire_nbytes()`` is the
exact serialized length. A buffer whose embedded keys use the legacy DCF
layout (``OC2DCF01``) is refused with ``DCF_WIRE_LEGACY_CONVERT``: Z_t keys
in that layout were produced with 64-bit Convert words and
would not evaluate correctly under the binding rule.

Statistical term (paper, Section 6, loss). The Z_t
value words of the two DCF keys are 128-bit PRG blocks reduced mod t, each at
most t / 2^128 from uniform (a reduced 64-bit word would be t / 2^64 away,
about 2^-28 for the admitted t). A key pair of n levels consumes
``2 (2 n + 1)`` such words (two per level and one leaf word per party), an
element consumes two key pairs; ``r2f_statistical_log2_eps`` returns
log2(n_word t / 2^128) for a batch, the term the session budget charges
(about 2^-83 per element at t = 68720050177).

What is tested (tests/test_ring_to_field.py): exhaustive small rings (k <= 8,
t in {3,5,7,11,13,17}, all share pairs, chosen and random masks) against
``signed_mod``, the counterexamples of the share-wise sum, random 64-bit with
t = 68720050177 and with a 62-bit prime (exercises the exact 128-bit
reduction), serialization round trip, byte formulas, refusal of legacy
material, the statistical term, modulus validation. Design limit: the
admitted t is about 2^36; the bound above is a union bound over the
pseudo-random words of the keys, not a proof of the DCF itself.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..errors import ContractViolation
from . import ring
from .dcf import DCFKeyBatch, concat_keys, eval_dcf_batch, gen_dcf_batch, key_bytes_per_element
from .dcf import is_probable_prime as _dcf_is_probable_prime
from .sharing import share_k, share_mod

_HDR = struct.Struct("<8sBBHIQQQQ")   # magic, k, party, reserved, flags, M, t, len_A, len_C
_MAGIC = b"OC2R2F01"
HEADER_BYTES = _HDR.size

DEFAULT_T = 68720050177   # the admitted BFV plaintext modulus (odd prime, about 2^36)


# ---------------------------------------------------------------------------
# Modulus validation and Z_t helpers
# ---------------------------------------------------------------------------
def _is_probable_prime(n: int) -> bool:
    """Deterministic Miller-Rabin (dcf.is_probable_prime; one implementation for both checks)."""
    return _dcf_is_probable_prime(int(n))


def check_field_modulus(t: int) -> int:
    """Validate an odd prime 3 <= t < 2^62."""
    t = int(t)
    if t < 3 or t >= (1 << 62) or t % 2 == 0 or not _is_probable_prime(t):
        raise ContractViolation(f"R2F_MODULUS: t must be an odd prime with 3 <= t < 2^62, got {t}")
    return t


def mulmod_scalar(c: int, v, t: int) -> np.ndarray:
    """(c * v) mod t for a public scalar c and uint64 array v with values < t (t < 2^62).

    Double-and-add ladder over the bits of c: every intermediate is below 2t
    < 2^63, so no uint64 overflow occurs for any admitted t.
    """
    t = int(t)
    tt = np.uint64(t)
    v = (np.asarray(v, dtype=np.uint64) % tt).astype(np.uint64)
    c = int(c) % t
    acc = np.zeros_like(v)
    base = v
    while c:
        if c & 1:
            acc = (acc + base) % tt
        c >>= 1
        if c:
            base = (base + base) % tt
    return acc.astype(np.uint64)


def _tadd(a, b, t: int) -> np.ndarray:
    tt = np.uint64(t)
    return ((np.asarray(a, np.uint64) + np.asarray(b, np.uint64)) % tt).astype(np.uint64)


def _tsub(a, b, t: int) -> np.ndarray:
    tt = np.uint64(t)
    return ((np.asarray(a, np.uint64) + tt - np.asarray(b, np.uint64)) % tt).astype(np.uint64)


def signed_mod(u_values, t: int, k: int = 64) -> np.ndarray:
    """Plaintext reference: (representative of u in [-2^(k-1), 2^(k-1))) mod t, as uint64 below t."""
    t = int(t)
    if t < 2:
        raise ContractViolation(f"R2F_MODULUS: t must be >= 2, got {t}")
    s = ring.to_signed_k(np.asarray(u_values, dtype=np.uint64), k)
    return np.mod(s, np.int64(t)).astype(np.uint64)


def naive_sharewise_mod(a0, a1, t: int) -> np.ndarray:
    """The INCORRECT share-wise reduction (a0 mod t + a1 mod t) mod t, kept as the test baseline."""
    tt = np.uint64(int(t))
    return ((np.asarray(a0, np.uint64) % tt + np.asarray(a1, np.uint64) % tt) % tt).astype(np.uint64)


# ---------------------------------------------------------------------------
# Material
# ---------------------------------------------------------------------------
@dataclass
class R2FMaterial:
    """Per-party material for M ring-to-field conversions to Z_t."""

    r_share: np.ndarray        # (M,) Z_q share of r
    r_mod_t_share: np.ndarray  # (M,) Z_t share of r mod t
    key_A: DCFKeyBatch         # alpha = r, payload Z_t, beta = 1
    key_C: DCFKeyBatch         # alpha = (r + 2^(k-1)) mod q, payload Z_t, beta = 1
    c_share: np.ndarray        # (M,) Z_t share of 1{r > (r + 2^(k-1)) mod q}
    t: int
    party: int

    @property
    def M(self) -> int:
        return int(self.r_share.shape[0])

    @property
    def k(self) -> int:
        return int(self.key_A.n_bits)

    def nbytes(self) -> int:
        """Charged material size (declared size formula; what estimates and the cost model use)."""
        return self.M * (2 * key_bytes_per_element(self.k, 1) + 24)

    def wire_nbytes(self) -> int:
        """Exact length of ``to_bytes()`` (container header, shares, two keys in the binding layout)."""
        return HEADER_BYTES + 24 * self.M + self.key_A.wire_nbytes() + self.key_C.wire_nbytes()

    def slice(self, lo: int, hi: int) -> "R2FMaterial":
        return R2FMaterial(self.r_share[lo:hi], self.r_mod_t_share[lo:hi], self.key_A.slice(lo, hi),
                           self.key_C.slice(lo, hi), self.c_share[lo:hi], self.t, self.party)

    def to_bytes(self) -> bytes:
        ka = self.key_A.to_bytes()
        kc = self.key_C.to_bytes()
        hdr = _HDR.pack(_MAGIC, self.k, self.party, 0, 0, self.M, int(self.t), len(ka), len(kc))
        return b"".join([
            hdr,
            np.ascontiguousarray(self.r_share, dtype="<u8").tobytes(),
            np.ascontiguousarray(self.r_mod_t_share, dtype="<u8").tobytes(),
            np.ascontiguousarray(self.c_share, dtype="<u8").tobytes(),
            ka, kc,
        ])

    @staticmethod
    def from_bytes(b: bytes) -> "R2FMaterial":
        b = bytes(b)
        if len(b) < HEADER_BYTES:
            raise ContractViolation("R2F_WIRE: buffer shorter than the header")
        magic, k, party, _res, flags, M, t, lka, lkc = _HDR.unpack_from(b, 0)
        if magic != _MAGIC or flags != 0:
            raise ContractViolation("R2F_WIRE: bad magic or flags")
        t = check_field_modulus(t)
        ring.check_k(k)
        expect = HEADER_BYTES + 24 * M + lka + lkc
        if len(b) != expect:
            raise ContractViolation(f"R2F_WIRE: expected {expect} bytes, got {len(b)}")
        off = HEADER_BYTES
        r_share = np.frombuffer(b, dtype="<u8", count=M, offset=off).astype(np.uint64); off += 8 * M
        r_mod_t = np.frombuffer(b, dtype="<u8", count=M, offset=off).astype(np.uint64); off += 8 * M
        c_share = np.frombuffer(b, dtype="<u8", count=M, offset=off).astype(np.uint64); off += 8 * M
        key_A = DCFKeyBatch.from_bytes(b[off:off + lka]); off += lka
        key_C = DCFKeyBatch.from_bytes(b[off:off + lkc]); off += lkc
        for key in (key_A, key_C):
            if key.n_bits != k or key.party != party or key.M != M or key.P != 1 or key.modulus != t:
                raise ContractViolation("R2F_WIRE: embedded keys disagree with the header")
        if np.any(r_mod_t >= np.uint64(t)) or np.any(c_share >= np.uint64(t)):
            raise ContractViolation("R2F_WIRE: Z_t shares out of range")
        return R2FMaterial(r_share, r_mod_t, key_A, key_C, c_share, t, int(party))


def concat_r2f(a: R2FMaterial, b: R2FMaterial) -> R2FMaterial:
    if a.t != b.t or a.party != b.party or a.k != b.k:
        raise ContractViolation("R2F_CONCAT: materials differ in t, k or party")
    return R2FMaterial(np.concatenate([a.r_share, b.r_share]), np.concatenate([a.r_mod_t_share, b.r_mod_t_share]),
                       concat_keys(a.key_A, b.key_A), concat_keys(a.key_C, b.key_C),
                       np.concatenate([a.c_share, b.c_share]), a.t, a.party)


def r2f_bytes_per_element(t: int) -> int:
    """Native (k = 64) material per element: two Z_t DCF keys (64, 1) plus three shares (3272 bytes)."""
    check_field_modulus(t)
    return 2 * key_bytes_per_element(64, 1) + 24


def r2f_zt_words_per_element(k: int = 64) -> int:
    """Number of Z_t Convert words behind one element: 2 key pairs x 2 parties x (2 per level + 1 leaf)."""
    k = ring.check_k(k)
    return 4 * (2 * k + 1)


def r2f_statistical_log2_eps(t: int, count: int = 1, k: int = 64) -> float:
    """log2 of the statistical term ``n_word * t / 2^128`` of ``count`` conversions (paper, Section 6, loss).

    Union bound over the 128-bit PRG blocks reduced mod t that enter the two
    DCF key pairs of each element. This is the quantity a ``SessionBudget``
    is charged for the Z_t keys a session consumes.
    """
    t = check_field_modulus(t)
    count = int(count)
    if count < 1:
        raise ContractViolation(f"R2F_PARAMS: count must be >= 1, got {count}")
    return math.log2(count * r2f_zt_words_per_element(k)) + math.log2(t) - 128.0


# ---------------------------------------------------------------------------
# Dealer
# ---------------------------------------------------------------------------
def r2f_material_from_mask(r, t: int, rng: np.random.Generator, k: int = 64) -> Tuple[R2FMaterial, R2FMaterial]:
    """Dealer material for chosen masks r (M,) over Z_{2^k} (tests pass chosen masks)."""
    t = check_field_modulus(t)
    k = ring.check_k(k)
    if k < 2:
        raise ContractViolation(f"R2F_RING: k must be >= 2, got {k}")
    r = ring.reduce_k(np.asarray(r, dtype=np.uint64).reshape(-1), k)
    M = r.shape[0]
    if M == 0:
        raise ContractViolation("R2F_SHAPE: empty batch")
    half = np.uint64(1 << (k - 1))
    r_c = ring.reduce_k(ring.add(r, half), k)
    one = np.ones((M, 1), np.uint64)
    kA0, kA1 = gen_dcf_batch(r, one, k, rng, modulus=t)
    kC0, kC1 = gen_dcf_batch(r_c, one, k, rng, modulus=t)
    c = (r > r_c).astype(np.uint64)
    r0, r1 = share_k(r, k, rng)
    rt0, rt1 = share_mod(r % np.uint64(t), t, rng)
    c0, c1 = share_mod(c, t, rng)
    return (R2FMaterial(r0, rt0, kA0, kC0, c0, t, 0),
            R2FMaterial(r1, rt1, kA1, kC1, c1, t, 1))


def r2f_material_k(M: int, t: int, rng: np.random.Generator, k: int = 64) -> Tuple[R2FMaterial, R2FMaterial]:
    """Dealer material for M elements over Z_{2^k} with uniform masks."""
    k = ring.check_k(k)
    r = ring.reduce_k(ring.rand_u64((int(M),), rng), k)
    return r2f_material_from_mask(r, t, rng, k)


def r2f_material(M: int, t: int, rng: np.random.Generator) -> Tuple[R2FMaterial, R2FMaterial]:
    """Dealer material for M native (k = 64) conversions to Z_t."""
    return r2f_material_k(M, t, rng, 64)


# ---------------------------------------------------------------------------
# Online
# ---------------------------------------------------------------------------
def r2f_open_share(u_share, mat: R2FMaterial, party: int) -> np.ndarray:
    """This party's message for the single open: u + [r] mod 2^k."""
    if party != mat.party:
        raise ContractViolation(f"R2F_PARTY: material belongs to party {mat.party}, caller is {party}")
    return ring.reduce_k(ring.add(np.asarray(u_share, np.uint64), mat.r_share), mat.k)


def r2f_eval(mat: R2FMaterial, y, party: int) -> np.ndarray:
    """Steps 2 to 5 on the public y: this party's Z_t share of x_signed mod t (uint64 below t)."""
    if party != mat.party:
        raise ContractViolation(f"R2F_PARTY: material belongs to party {mat.party}, caller is {party}")
    k, t = mat.k, int(mat.t)
    y = ring.reduce_k(np.asarray(y, dtype=np.uint64).reshape(-1), k)
    if y.shape[0] != mat.M:
        raise ContractViolation(f"R2F_SHAPE: y has {y.shape[0]} elements, material has {mat.M}")
    tt = np.uint64(t)
    w = eval_dcf_batch(mat.key_A, y)[:, 0]              # Z_t share of 1{y < r}
    dc = eval_dcf_batch(mat.key_C, y)[:, 0]             # Z_t share of 1{y < r + 2^(k-1) mod q}
    nn = _tadd(_tsub(dc, w, t), mat.c_share, t)         # Z_t share of 1{u < q/2}
    b = _tsub(np.zeros_like(nn), nn, t)                 # -nn
    if party == 0:
        b = _tadd(b, np.uint64(1), t)                   # 1 - nn
    q_mod_t = pow(2, k, t)
    umt = _tadd(_tsub(np.zeros_like(w), mat.r_mod_t_share, t), mulmod_scalar(q_mod_t, w, t), t)
    if party == 0:
        umt = _tadd(umt, y % tt, t)
    z = _tsub(umt, mulmod_scalar(q_mod_t, b, t), t)
    return z.astype(np.uint64)


__all__ = [
    "R2FMaterial", "r2f_material", "r2f_material_k", "r2f_material_from_mask", "r2f_open_share",
    "r2f_eval", "signed_mod", "naive_sharewise_mod", "r2f_bytes_per_element", "mulmod_scalar",
    "check_field_modulus", "concat_r2f", "DEFAULT_T", "HEADER_BYTES",
    "r2f_zt_words_per_element", "r2f_statistical_log2_eps",
]
