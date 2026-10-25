"""One-round DReLU / ReLU from two DCF keys (paper, Section 5.3, one-round DReLU).

With a public masked value y = x + r (mod 2^64), an n-bit comparison ring,
p = r mod 2^n and q = (r + 2^(n-1)) mod 2^n, for |x| < 2^(n-1):

    1{x >= 0} = DCF_q(y_n) - DCF_p(y_n) + 1{p > q},     y_n = y mod 2^n.

Proof sketch (tested, not only asserted): 1{x >= 0} = 1{(y_n - p) mod 2^n <
2^(n-1)} = 1{y_n in the cyclic interval [p, q)} and the indicator of a cyclic
interval is 1{y < q} - 1{y < p} + 1{p > q}. With payload (1, r) the same two
evaluations give shares of r * 1{x >= 0}, hence ReLU(x) = y*[b] - [r b] in
one round. The dealer-known constants 1{p > q} and 1{p > q} r are shipped as
additive shares; revealing 1{p > q} would leak one bit of r.

Material per element: two DCF keys (n bits, P in {1, 2}) plus three ring
shares (r, 1{p>q}, 1{p>q} r). Serialization: 40-byte header ``OC2DRL01``,
three raw little-endian share arrays, then the two DCF key batches in the
binding DCF wire layout (``DCFKeyBatch.to_bytes``). ``nbytes()``
is the charged material size (declared size formula,
2200 bytes per element at n = 32 with product); ``wire_nbytes()`` is the
exact serialized length, which is larger because the binding DCF layout
stores every control bit as a u64 word.

Comparison-key repair (paper, Section 5.3, a finding on comparison keys). The payload (1, r) key is
exactly the object the pre-repair rule broke: with one PRG value word
shared by both payload components, the difference of the two public
correction words of a level revealed every bit of alpha = r mod 2^n and the
value r - 1, hence the mask r and the comparison input x = y - r, to each
party from its own key. ``gen_dcf_batch`` now gives every payload component
its own Convert words (``prg.expand_convert``); the regression
``tests/test_dcf_component_independence.py`` recovers the mask from 2000 of
2000 keys of the old rule and from 0 of 2000 keys of this module. A buffer
whose embedded P = 2 keys use the legacy DCF layout (``OC2DCF01``) is refused
(``DCF_WIRE_LEGACY_CONVERT``): such keys were generated with the old rule.

What is tested (tests/test_drelu.py): exhaustive x at n = 8 against the sign
function and ReLU, random at n in {16, 32, 64}, serialization round trip,
byte formulas, refusal of legacy product material, out-of-domain inputs are
documented (not detected: the gate returns garbage when |x| >= 2^(n-1); the
numeric contract bounds x).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..errors import ContractViolation
from . import ring
from .dcf import DCFKeyBatch, concat_keys, eval_dcf_batch, gen_dcf_batch, key_bytes_per_element
from .sharing import share

_HDR = struct.Struct("<8sBBHIQQQ")   # magic, n_bits, party, with_product, flags, M, len_key_p, len_key_q
_MAGIC = b"OC2DRL01"
HEADER_BYTES = _HDR.size


@dataclass
class DReLUMaterial:
    """Per-party material for a batch of M one-round sign / ReLU gates."""

    r_share: np.ndarray      # (M,) share of the 64-bit mask r
    key_p: DCFKeyBatch       # alpha = r mod 2^n, payload (1[, r])
    key_q: DCFKeyBatch       # alpha = (r + 2^(n-1)) mod 2^n, payload (1[, r])
    c_b: np.ndarray          # (M,) share of 1{p > q}
    c_rb: np.ndarray         # (M,) share of 1{p > q} r (zeros without product payload)
    n_bits: int
    party: int

    @property
    def M(self) -> int:
        return int(self.r_share.shape[0])

    @property
    def with_product(self) -> bool:
        return self.key_p.P >= 2

    def nbytes(self) -> int:
        """Charged material size (declared size formula; what estimates and the cost model use)."""
        return self.M * drelu_bytes_per_element(self.n_bits, self.with_product)

    def wire_nbytes(self) -> int:
        """Exact length of ``to_bytes()`` (container header, shares, two keys in the binding layout)."""
        return HEADER_BYTES + 24 * self.M + self.key_p.wire_nbytes() + self.key_q.wire_nbytes()

    def slice(self, lo: int, hi: int) -> "DReLUMaterial":
        return DReLUMaterial(self.r_share[lo:hi], self.key_p.slice(lo, hi), self.key_q.slice(lo, hi),
                             self.c_b[lo:hi], self.c_rb[lo:hi], self.n_bits, self.party)

    def to_bytes(self) -> bytes:
        kp = self.key_p.to_bytes()
        kq = self.key_q.to_bytes()
        hdr = _HDR.pack(_MAGIC, self.n_bits, self.party, 1 if self.with_product else 0, 0,
                        self.M, len(kp), len(kq))
        return b"".join([
            hdr,
            np.ascontiguousarray(self.r_share, dtype="<u8").tobytes(),
            np.ascontiguousarray(self.c_b, dtype="<u8").tobytes(),
            np.ascontiguousarray(self.c_rb, dtype="<u8").tobytes(),
            kp, kq,
        ])

    @staticmethod
    def from_bytes(b: bytes) -> "DReLUMaterial":
        b = bytes(b)
        if len(b) < HEADER_BYTES:
            raise ContractViolation("DRELU_WIRE: buffer shorter than the header")
        magic, n_bits, party, wp, flags, M, lkp, lkq = _HDR.unpack_from(b, 0)
        if magic != _MAGIC or flags != 0:
            raise ContractViolation("DRELU_WIRE: bad magic or flags")
        expect = HEADER_BYTES + 24 * M + lkp + lkq
        if len(b) != expect:
            raise ContractViolation(f"DRELU_WIRE: expected {expect} bytes, got {len(b)}")
        off = HEADER_BYTES
        r_share = np.frombuffer(b, dtype="<u8", count=M, offset=off).astype(np.uint64); off += 8 * M
        c_b = np.frombuffer(b, dtype="<u8", count=M, offset=off).astype(np.uint64); off += 8 * M
        c_rb = np.frombuffer(b, dtype="<u8", count=M, offset=off).astype(np.uint64); off += 8 * M
        key_p = DCFKeyBatch.from_bytes(b[off:off + lkp]); off += lkp
        key_q = DCFKeyBatch.from_bytes(b[off:off + lkq]); off += lkq
        if key_p.n_bits != n_bits or key_q.n_bits != n_bits or key_p.party != party or key_q.party != party:
            raise ContractViolation("DRELU_WIRE: embedded keys disagree with the header")
        if (key_p.P >= 2) != bool(wp) or key_p.P != key_q.P or key_p.M != M or key_q.M != M:
            raise ContractViolation("DRELU_WIRE: embedded key shape disagrees with the header")
        return DReLUMaterial(r_share, key_p, key_q, c_b, c_rb, int(n_bits), int(party))


def concat_drelu(a: DReLUMaterial, b: DReLUMaterial) -> DReLUMaterial:
    if a.n_bits != b.n_bits or a.party != b.party or a.with_product != b.with_product:
        raise ContractViolation("DRELU_CONCAT: materials differ in n_bits, party or payload")
    return DReLUMaterial(np.concatenate([a.r_share, b.r_share]), concat_keys(a.key_p, b.key_p),
                         concat_keys(a.key_q, b.key_q), np.concatenate([a.c_b, b.c_b]),
                         np.concatenate([a.c_rb, b.c_rb]), a.n_bits, a.party)


def drelu_bytes_per_element(n_bits: int, with_product: bool = True) -> int:
    """Two DCF keys (P = 2 with product, else 1) plus three ring shares."""
    P = 2 if with_product else 1
    return 2 * key_bytes_per_element(n_bits, P) + 24


def drelu_material_from_mask(r, n_bits: int, rng: np.random.Generator,
                             with_product: bool = True) -> Tuple[DReLUMaterial, DReLUMaterial]:
    """Dealer: material for the given 64-bit masks r (M,) (tests pass chosen masks)."""
    n_bits = int(n_bits)
    if n_bits < 2 or n_bits > 64:
        raise ContractViolation(f"DRELU_N_BITS: n_bits must be in [2, 64], got {n_bits}")
    r = np.asarray(r, dtype=np.uint64).reshape(-1)
    M = r.shape[0]
    if M == 0:
        raise ContractViolation("DRELU_SHAPE: empty batch")
    p = ring.reduce_k(r, n_bits)
    half = np.uint64(1 << (n_bits - 1))
    q = ring.reduce_k(ring.add(p, half), n_bits)
    if with_product:
        beta = np.stack([np.ones(M, np.uint64), r], axis=1)
    else:
        beta = np.ones((M, 1), np.uint64)
    kp0, kp1 = gen_dcf_batch(p, beta, n_bits, rng)
    kq0, kq1 = gen_dcf_batch(q, beta, n_bits, rng)
    wrap = (p > q).astype(np.uint64)
    wrap_r = ring.mul(wrap, r)
    r0, r1 = share(r, rng)
    cb0, cb1 = share(wrap, rng)
    if with_product:
        crb0, crb1 = share(wrap_r, rng)
    else:
        crb0 = np.zeros(M, np.uint64)
        crb1 = np.zeros(M, np.uint64)
    return (DReLUMaterial(r0, kp0, kq0, cb0, crb0, n_bits, 0),
            DReLUMaterial(r1, kp1, kq1, cb1, crb1, n_bits, 1))


def drelu_material(M: int, n_bits: int, rng: np.random.Generator,
                   with_product: bool = True) -> Tuple[DReLUMaterial, DReLUMaterial]:
    """Dealer: material for M one-round sign (and ReLU) gates with uniform masks."""
    r = ring.rand_u64((int(M),), rng)
    return drelu_material_from_mask(r, n_bits, rng, with_product)


def drelu_mask_share(mat: DReLUMaterial) -> np.ndarray:
    """This party's share of r, to be added to its share of x before the single open."""
    return mat.r_share


def drelu_eval(mat: DReLUMaterial, y) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """From the public y = x + r (M,) return shares of (1{x >= 0}, r 1{x >= 0}).

    The second element is None when the material carries no product payload.
    """
    y = np.asarray(y, dtype=np.uint64).reshape(-1)
    if y.shape[0] != mat.M:
        raise ContractViolation(f"DRELU_SHAPE: y has {y.shape[0]} elements, material has {mat.M}")
    yn = ring.reduce_k(y, mat.n_bits)
    oq = eval_dcf_batch(mat.key_q, yn)
    op = eval_dcf_batch(mat.key_p, yn)
    res = ring.sub(oq, op)
    b = ring.add(res[:, 0], mat.c_b)
    if res.shape[1] > 1:
        rb = ring.add(res[:, 1], mat.c_rb)
        return b, rb
    return b, None


def relu_from_drelu(y, b_share, rb_share) -> np.ndarray:
    """ReLU(x) share = y*[b] - [r b] for the public y = x + r (mod 2^64)."""
    return ring.sub(ring.mul(y, b_share), rb_share)


__all__ = [
    "DReLUMaterial", "drelu_material", "drelu_material_from_mask", "drelu_eval",
    "relu_from_drelu", "drelu_bytes_per_element", "drelu_mask_share", "concat_drelu",
    "HEADER_BYTES",
]
