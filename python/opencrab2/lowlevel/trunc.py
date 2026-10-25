"""Faithful truncation gate with the signed offset (paper, Section 5.1, faithful truncation).

Parameters: ring q = 2^k (k = 64 natively, k <= 64 for the exhaustive
small-ring tests), logical width L, shift f with 1 <= f <= L-1 <= k-1, input
x in [-2^(L-1), 2^(L-1)) (signed), a = 2^(L-1).

Dealer per element: uniform r in Z_q; shares [r]; shares [floor(r / 2^f)];
DCF key A (wrap): alpha = r, n_bits = k, payload 1; DCF key B (borrow):
alpha = r mod 2^f, n_bits = f, payload 1.

Online (one round):

    1. P0 adds the public offset: x' = x + a (P0 only). Both: open y = x' + [r] mod q.
    2. Locally: [w] = DCF_A(y) (= 1{y < r}), [b] = DCF_B(y mod 2^f) (= 1{y mod 2^f < r mod 2^f}).
    3. [z] = floor(y / 2^f) 1{party 0} - [floor(r / 2^f)] - [b] + 2^(k-f) [w] - 2^(L-1-f) 1{party 0}  (mod q)
    4. [z] are shares of floor(x / 2^f) with signed semantics.

Correctness: with z0 = x + a in [0, 2^L), y = z0 + r - w q and the low-f-bit
subtraction borrow b, floor(z0 / 2^f) = floor(y / 2^f) - floor(r / 2^f) - b +
w 2^(k-f); a is divisible by 2^f so subtracting a / 2^f = 2^(L-1-f) gives the
signed floor. The last term is easy to drop by mistake; the regression
(k=8, L=6, f=3, x=-13, r=37) returns -2 here and 2 without it.

Privacy: the only message is y = x' + r with uniform r, independent of x; w
and b stay shared. The DCF keys are one-time objects (ledger enforced). Both
keys carry a single payload word over Z_{2^64}, so the multi-payload repair
of the comparison keys (paper, Section 5.3, a finding on comparison keys) does
not change them.

Serialization: 40-byte header ``OC2TRN01``, the two share arrays, then the
two DCF key batches in the DCF wire layout (``DCFKeyBatch.to_bytes``; keys of
the earlier layout ``OC2DCF01`` still parse because P = 1). ``nbytes()`` is
the charged material size (2064 bytes per element at f = 16);
``wire_nbytes()`` is the exact serialized length, larger because the DCF
layout stores every control bit as a u64 word.

``trunc_local`` (SecureML local probabilistic truncation) is kept for
contract-bounded products. Its error is not a rounding error: with uniform
shares a truncated value x is off by at most one unit with probability 1 and
by a full wrap of 2^(64-f) ring units with probability |x|/2^64. Per session
this adds up. A 4-layer GPT-2 session of 16 prompt and 8 generated tokens
executes 5.49 million local truncations and a 12-layer session 15.6 million;
summing |x|/2^64 over the truncations whose result can reach a released token
(the GELU polynomial products and the squarings of the softmax exponential
dominate) gives a wrong token sequence in about 0.8 percent (4 layers) and
2.4 percent (12 layers) of sessions. A wrap in the LM head is harmless
because the argmax compares modulo 2^32. The wrap is silent: no party can
detect it, and a wrap before a cached key or value corrupts every later
token. The contract value ``trunc_products = "faithful"`` would route these
products through the gate above at one extra round and 2064 bytes of material
per truncated element; the measured sessions run ``local_prob``.

What is tested (tests/test_trunc.py): the regression value, exhaustive
agreement of the gate (real DCF keys, real shares) with floor(x / 2^f) and
with ``reference_trunc_small`` for every (k <= 8, L, f, x, r); 10^5 random
64-bit elements across magnitudes 2^20..2^39, boundary values and sign
changes; serialization round trip; byte formula (2064 bytes at f = 16);
``trunc_local`` on contract-bounded inputs (no wrap expected in 10^5 draws
below 2^39).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..errors import ContractViolation
from . import ring
from .dcf import DCFKeyBatch, concat_keys, eval_dcf_batch, gen_dcf_batch, key_bytes_per_element
from .sharing import share_k

_HDR = struct.Struct("<8sBBBBIQQQ")   # magic, f, L, k, party, flags, M, len_wrap, len_borrow
_MAGIC = b"OC2TRN01"
HEADER_BYTES = _HDR.size


def check_trunc_params(f: int, L: int, k: int = 64) -> Tuple[int, int, int]:
    """Validate 1 <= f <= L-1 <= k-1 (k <= 64)."""
    k = ring.check_k(k)
    f = int(f)
    L = int(L)
    if not (1 <= f <= L - 1 <= k - 1):
        raise ContractViolation(f"TRUNC_PARAMS: need 1 <= f <= L-1 <= k-1, got f={f}, L={L}, k={k}")
    return f, L, k


@dataclass
class TruncMaterial:
    """Per-party material for M faithful truncations by f bits at logical width L."""

    r_share: np.ndarray        # (M,) share of the mask r over Z_q
    r_high_share: np.ndarray   # (M,) share of floor(r / 2^f) over Z_q
    key_wrap: DCFKeyBatch      # alpha = r, n_bits = k, payload 1
    key_borrow: DCFKeyBatch    # alpha = r mod 2^f, n_bits = f, payload 1
    f: int
    L: int
    party: int

    @property
    def M(self) -> int:
        return int(self.r_share.shape[0])

    @property
    def k(self) -> int:
        """Ring width (bits), carried by the wrap key's domain."""
        return int(self.key_wrap.n_bits)

    def nbytes(self) -> int:
        """Charged material size (declared size formula; what estimates and the cost model use)."""
        return self.M * (key_bytes_per_element(self.k, 1) + key_bytes_per_element(self.f, 1) + 16)

    def wire_nbytes(self) -> int:
        """Exact length of ``to_bytes()`` (container header, shares, two keys in the binding layout)."""
        return HEADER_BYTES + 16 * self.M + self.key_wrap.wire_nbytes() + self.key_borrow.wire_nbytes()

    def slice(self, lo: int, hi: int) -> "TruncMaterial":
        return TruncMaterial(self.r_share[lo:hi], self.r_high_share[lo:hi], self.key_wrap.slice(lo, hi),
                             self.key_borrow.slice(lo, hi), self.f, self.L, self.party)

    def to_bytes(self) -> bytes:
        kw = self.key_wrap.to_bytes()
        kb = self.key_borrow.to_bytes()
        hdr = _HDR.pack(_MAGIC, self.f, self.L, self.k, self.party, 0, self.M, len(kw), len(kb))
        return b"".join([
            hdr,
            np.ascontiguousarray(self.r_share, dtype="<u8").tobytes(),
            np.ascontiguousarray(self.r_high_share, dtype="<u8").tobytes(),
            kw, kb,
        ])

    @staticmethod
    def from_bytes(b: bytes) -> "TruncMaterial":
        b = bytes(b)
        if len(b) < HEADER_BYTES:
            raise ContractViolation("TRUNC_WIRE: buffer shorter than the header")
        magic, f, L, k, party, flags, M, lkw, lkb = _HDR.unpack_from(b, 0)
        if magic != _MAGIC or flags != 0:
            raise ContractViolation("TRUNC_WIRE: bad magic or flags")
        check_trunc_params(f, L, k)
        expect = HEADER_BYTES + 16 * M + lkw + lkb
        if len(b) != expect:
            raise ContractViolation(f"TRUNC_WIRE: expected {expect} bytes, got {len(b)}")
        off = HEADER_BYTES
        r_share = np.frombuffer(b, dtype="<u8", count=M, offset=off).astype(np.uint64); off += 8 * M
        r_high = np.frombuffer(b, dtype="<u8", count=M, offset=off).astype(np.uint64); off += 8 * M
        key_wrap = DCFKeyBatch.from_bytes(b[off:off + lkw]); off += lkw
        key_borrow = DCFKeyBatch.from_bytes(b[off:off + lkb]); off += lkb
        if key_wrap.n_bits != k or key_borrow.n_bits != f or key_wrap.M != M or key_borrow.M != M:
            raise ContractViolation("TRUNC_WIRE: embedded keys disagree with the header")
        if key_wrap.party != party or key_borrow.party != party or key_wrap.P != 1 or key_borrow.P != 1:
            raise ContractViolation("TRUNC_WIRE: embedded keys disagree with the header")
        return TruncMaterial(r_share, r_high, key_wrap, key_borrow, int(f), int(L), int(party))


def concat_trunc(a: TruncMaterial, b: TruncMaterial) -> TruncMaterial:
    if a.f != b.f or a.L != b.L or a.party != b.party or a.k != b.k:
        raise ContractViolation("TRUNC_CONCAT: materials differ in f, L, k or party")
    return TruncMaterial(np.concatenate([a.r_share, b.r_share]),
                         np.concatenate([a.r_high_share, b.r_high_share]),
                         concat_keys(a.key_wrap, b.key_wrap), concat_keys(a.key_borrow, b.key_borrow),
                         a.f, a.L, a.party)


def trunc_bytes_per_element(f: int) -> int:
    """Native (k = 64) material per element: key(64,1) + key(f,1) + 16 (2064 bytes at f = 16)."""
    return key_bytes_per_element(64, 1) + key_bytes_per_element(int(f), 1) + 16


# ---------------------------------------------------------------------------
# Dealer
# ---------------------------------------------------------------------------
def trunc_material_from_mask(r, f: int, L: int, rng: np.random.Generator,
                             k: int = 64) -> Tuple[TruncMaterial, TruncMaterial]:
    """Dealer material for the given masks r (M,) over Z_{2^k} (tests pass chosen masks)."""
    f, L, k = check_trunc_params(f, L, k)
    r = ring.reduce_k(np.asarray(r, dtype=np.uint64).reshape(-1), k)
    M = r.shape[0]
    if M == 0:
        raise ContractViolation("TRUNC_SHAPE: empty batch")
    r_hi = (r >> np.uint64(f)).astype(np.uint64)
    r_lo = ring.reduce_k(r, f)
    one = np.ones((M, 1), np.uint64)
    kw0, kw1 = gen_dcf_batch(r, one, k, rng)
    kb0, kb1 = gen_dcf_batch(r_lo, one, f, rng)
    r0, r1 = share_k(r, k, rng)
    h0, h1 = share_k(r_hi, k, rng)
    return (TruncMaterial(r0, h0, kw0, kb0, f, L, 0),
            TruncMaterial(r1, h1, kw1, kb1, f, L, 1))


def trunc_material_k(M: int, f: int, L: int, rng: np.random.Generator,
                     k: int = 64) -> Tuple[TruncMaterial, TruncMaterial]:
    """Dealer material for M elements over Z_{2^k} with uniform masks."""
    f, L, k = check_trunc_params(f, L, k)
    r = ring.reduce_k(ring.rand_u64((int(M),), rng), k)
    return trunc_material_from_mask(r, f, L, rng, k)


def trunc_material(M: int, f: int, L: int, rng: np.random.Generator) -> Tuple[TruncMaterial, TruncMaterial]:
    """Dealer material for M native (k = 64) faithful truncations."""
    return trunc_material_k(M, f, L, rng, 64)


# ---------------------------------------------------------------------------
# Online
# ---------------------------------------------------------------------------
def trunc_offset(x_share, party: int, L: int, k: int = 64) -> np.ndarray:
    """Step 1: P0 adds the public offset a = 2^(L-1); P1 returns its share unchanged (mod 2^k)."""
    k = ring.check_k(k)
    L = int(L)
    if not (2 <= L <= k):
        raise ContractViolation(f"TRUNC_PARAMS: need 2 <= L <= k, got L={L}, k={k}")
    x = ring.reduce_k(np.asarray(x_share, dtype=np.uint64), k)
    if party == 0:
        return ring.reduce_k(ring.add(x, np.uint64(1 << (L - 1))), k)
    if party == 1:
        return x
    raise ContractViolation(f"TRUNC_PARTY: party must be 0 or 1, got {party}")


def trunc_open_share(x_share, mat: TruncMaterial, party: int) -> np.ndarray:
    """This party's message for the single open: x' + [r] mod 2^k (x' includes P0's offset)."""
    if party != mat.party:
        raise ContractViolation(f"TRUNC_PARTY: material belongs to party {mat.party}, caller is {party}")
    xp = trunc_offset(x_share, party, mat.L, mat.k)
    return ring.reduce_k(ring.add(xp, mat.r_share), mat.k)


def trunc_faithful_eval(mat: TruncMaterial, y, party: int) -> np.ndarray:
    """Steps 2 and 3 on the public y: this party's share of floor(x / 2^f) (mod 2^k).

    Includes the -2^(L-1-f) term on P0.
    """
    if party != mat.party:
        raise ContractViolation(f"TRUNC_PARTY: material belongs to party {mat.party}, caller is {party}")
    k, f, L = mat.k, mat.f, mat.L
    y = ring.reduce_k(np.asarray(y, dtype=np.uint64).reshape(-1), k)
    if y.shape[0] != mat.M:
        raise ContractViolation(f"TRUNC_SHAPE: y has {y.shape[0]} elements, material has {mat.M}")
    w = eval_dcf_batch(mat.key_wrap, y)[:, 0]                       # share of 1{y < r}
    b = eval_dcf_batch(mat.key_borrow, ring.reduce_k(y, f))[:, 0]    # share of 1{y mod 2^f < r mod 2^f}
    out = ring.sub(ring.sub(ring.mul(w, np.uint64(1 << (k - f))), mat.r_high_share), b)
    if party == 0:
        out = ring.add(out, y >> np.uint64(f))
        out = ring.sub(out, np.uint64(1 << (L - 1 - f)))
    return ring.reduce_k(out, k)


def trunc_local(share, f: int, party: int) -> np.ndarray:
    """SecureML local probabilistic truncation by f bits (k = 64 only, zero rounds).

    P0 shifts its share arithmetically; P1 negates, shifts, negates. The
    reconstruction is floor(x / 2^f) up to one unit except with probability
    |x|/2^64, when it is off by a full 2^(64-f) wrap (module docstring: about
    0.8 percent of 4-layer and 2.4 percent of 12-layer sessions of 8 tokens).
    Admitted only for contract-bounded products (``trunc_products =
    local_prob``); the LayerNorm square uses the faithful gate.
    """
    f = int(f)
    if f < 0 or f > 63:
        raise ContractViolation(f"TRUNC_PARAMS: trunc_local needs 0 <= f <= 63, got {f}")
    s = np.asarray(share, dtype=np.uint64)
    if f == 0:
        return s
    if party == 0:
        return ring.arith_shift_right(s, f)
    if party == 1:
        return ring.neg(ring.arith_shift_right(ring.neg(s), f))
    raise ContractViolation(f"TRUNC_PARTY: party must be 0 or 1, got {party}")


# ---------------------------------------------------------------------------
# Pure-integer reference
# ---------------------------------------------------------------------------
def reference_trunc_small(k: int, L: int, f: int, x: int, r: int) -> int:
    """Pure-int evaluation of the gate formula; returns the signed result in [-2^(k-1), 2^(k-1)).

    Used by the exhaustive tests as an independent oracle of the gate algebra
    (paper, Section 5.1, TruncFaithful; the DCF outputs are replaced by the
    comparisons they encode).
    """
    f, L, k = check_trunc_params(f, L, k)
    q = 1 << k
    a = 1 << (L - 1)
    x = int(x)
    r = int(r) % q
    if not (-a <= x < a):
        raise ContractViolation(f"TRUNC_RANGE: x = {x} outside [-2^{L - 1}, 2^{L - 1})")
    y = (x + a + r) % q
    w = 1 if y < r else 0
    b = 1 if (y % (1 << f)) < (r % (1 << f)) else 0
    z = (y >> f) - (r >> f) - b + (1 << (k - f)) * w - (1 << (L - 1 - f))
    z %= q
    return z - q if z >= q // 2 else z


def reference_trunc_small_no_offset_term(k: int, L: int, f: int, x: int, r: int) -> int:
    """The printed (incorrect) formula without the -2^(L-1-f) term, kept for the regression test."""
    f, L, k = check_trunc_params(f, L, k)
    q = 1 << k
    a = 1 << (L - 1)
    y = (int(x) + a + int(r)) % q
    r = int(r) % q
    w = 1 if y < r else 0
    b = 1 if (y % (1 << f)) < (r % (1 << f)) else 0
    z = ((y >> f) - (r >> f) - b + (1 << (k - f)) * w) % q
    return z - q if z >= q // 2 else z


__all__ = [
    "TruncMaterial", "trunc_material", "trunc_material_k", "trunc_material_from_mask",
    "trunc_offset", "trunc_open_share", "trunc_faithful_eval", "trunc_local",
    "trunc_bytes_per_element", "reference_trunc_small", "reference_trunc_small_no_offset_term",
    "check_trunc_params", "concat_trunc", "HEADER_BYTES",
]
