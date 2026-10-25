"""Batched GGM Distributed Comparison Function (paper, Section 2.2, DCF comparison).

GGM key batches on the fixed-key MMO PRG (``opencrab2.lowlevel.prg``) with a
payload group option and the component-specific Convert (paper, Section 5.3,
a finding on comparison keys; Appendix D, Disclosure).
For M independent keys this module realizes

    f^{<}_{alpha[m], beta[m]}(x) = beta[m]  if  x < alpha[m]  else  0

over an n_bits-bit input domain (1 <= n_bits <= 64), with the payload group

* ``modulus=None``: Z_{2^64}^P, P payload words, numpy uint64 wrap;
* ``modulus=t`` (odd, 3 <= t < 2^62): Z_t^P. Payload words are reduced mod
  t, PRG value words are 128-bit blocks reduced mod t (exact arithmetic),
  every correction word is computed mod t, the output is mod t and party 1
  negates mod t.

The construction is the DCF of Boyle, Chandran, Gilboa, Gupta, Ishai, Kumar
and Rathee (Eurocrypt 2021, Fig. 1) with the product-group Convert: one GGM level per input bit, a seed
correction word, P value correction words and two control-bit corrections
per level, and a final value correction word.

Convert rule (binding in both languages; the words
come from ``prg.expand_convert`` / ``prg.leaf_words``). With
B_j(s) = AES_K(s xor c_j) xor s every payload component c reads its OWN
pseudo-random words:

    Z_{2^64}:  c = 0 uses block 2 (vL = LE64 bytes 0..7, vR = LE64 bytes 8..15),
               c >= 1 uses block 3 + c. P = 1 keys are bit-identical to the
               keys of the pre-repair rule.
    Z_t:       vL_c = LE128(B_{4+2c}(s)) mod t, vR_c = LE128(B_{5+2c}(s)) mod t.
    leaf:      the final correction word of component c uses vL_c of the leaf seed.

Gen and Eval use the component-specific words everywhere a value word enters
(V_CW, the V_alpha update, the running output sum, the final correction
word). The pre-repair rule (of the superseded engine) reused one value
word for all components; the public correction words then satisfied
``cw_values[i][1] - cw_values[i][0] = +-(a_i - a_{i-1}) (beta_1 - beta_0)``,
so a party read every bit of alpha and beta_1 - beta_0 from its own key (for
DReLU material: the mask r, hence the comparison input).

Evaluation expands only the blocks it needs: per key and level the seed
block of the branch selected by the public input bit, the control block and
ONE value block per payload component (in Z_t mode the block 4 + 2c + x_i),
i.e. 2 + P AES blocks, vectorized over the batch; large batches are split
over a small thread pool. Control-bit dependent corrections are applied with
masks, never with branches.

Sizes. ``key_bytes_per_element`` is the declared key size
``16 + n_bits*(16 + 8P + 1) + 8P`` (two control bits of a level in one byte;
paper, Section 5.3, online communication);
this is the quantity the cost model and the material estimates charge and
what ``nbytes()`` reports. The binding wire layout (identical to the
C++ ``DcfKeyBatch::to_bytes`` and checked by the cross-language vectors) stores every control bit as one u64 word:

    u8 version = 2, u8 party, u8 n_bits, u8 P, u64 LE M, u64 LE modulus (0 = none),
    seed[M][2], cw_seeds[n][2][M], cw_values[n][P][M], cw_tL[n][M], cw_tR[n][M],
    cw_final[P][M], all u64 LE,

so ``len(to_bytes()) = 20 + 8 M (2 + 4n + nP + P)`` (``wire_nbytes()``), which
is larger than ``nbytes()`` by 15 bytes per level and key. ``from_bytes``
reads this layout and still recognizes the legacy layout (magic
``OC2DCF01``, 32-byte header, packed control bits): such a buffer parses when
P = 1 with the Z_{2^64} payload group (those keys are bit-identical under
both Convert rules) and is refused with ``DCF_WIRE_LEGACY_CONVERT`` otherwise,
because a P >= 2 or Z_t key in that layout was produced with the
pre-repair Convert (it leaks, and it would evaluate to garbage here).
Message sizes depend only on (M, n_bits, P), never on alpha or beta.

What is tested (tests/test_dcf.py, tests/test_dcf_component_independence.py):
exhaustive correctness for n_bits <= 8 in both payload groups and P in
{1, 2, 3}, random correctness at 32 and 64 bits, agreement with an
independent pure-integer evaluation of the Convert rule, the four
KEY_COMPONENT_INDEPENDENCE checks (the blind mask recovery succeeds on 0 of
2000 P = 2 DReLU keys and on 2000 of 2000 keys of the pre-repair rule), a
P = 1 vector pinned before the repair, serialization round trips,
both wire layouts, slicing and concatenation, the byte formulas, a
performance smoke (2^16 keys at 32 bits under 3 s). What is NOT proven by
the tests: they show that the specific correction-word relation is gone and
that components use disjoint PRG blocks; pseudo-randomness of a key rests on
the cited construction and on the PRG in the ideal-cipher model. Design
limits: a Z_t value word is at most t / 2^128 from uniform (below 2^-66 for
every admitted t; the session budget charges n_word * t / 2^128); keys are
one-time objects and the ledger (not this module) enforces single use.
"""

from __future__ import annotations

import functools
import os
import struct
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..errors import ContractViolation
from . import prg as _prg

_ZERO = np.uint64(0)
_ONE = np.uint64(1)
_SH56 = np.uint64(56)
_TW3 = _prg.tweak_word(_prg.TWEAK_CONTROL)

MAX_MODULUS = 1 << 62
MAX_PAYLOADS = _prg.MAX_PAYLOADS

EVAL_CHUNK = 1 << 15
# Both parties of an in-process harness evaluate concurrently, so use at most
# half the cores (numpy and OpenSSL release the GIL on large buffers).
EVAL_THREADS = max(1, min(4, (os.cpu_count() or 2) // 2))

WIRE_VERSION = 2
_HDR = struct.Struct("<BBBBQQ")            # version, party, n_bits, P, M, modulus (0 = none)
HEADER_BYTES = _HDR.size                    # 20, binding layout
_LEGACY_HDR = struct.Struct("<8sBBHQQI")    # magic, n_bits, party, P, M, modulus (0 = none), flags
_LEGACY_MAGIC = b"OC2DCF01"
LEGACY_HEADER_BYTES = _LEGACY_HDR.size      # 32, legacy layout (read only)
_MAX_WIRE_M = 1 << 40


# ---------------------------------------------------------------------------
# Parameter validation and group helpers
# ---------------------------------------------------------------------------
def check_n_bits(n_bits: int) -> int:
    n_bits = int(n_bits)
    if n_bits < 1 or n_bits > 64:
        raise ContractViolation(f"DCF_N_BITS: n_bits must be in [1, 64], got {n_bits}")
    return n_bits


@functools.lru_cache(maxsize=256)
def is_probable_prime(n: int) -> bool:
    """Deterministic Miller-Rabin for n < 3.3e24 (bases 2..41); exact for every admitted modulus."""
    n = int(n)
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)
    for p in small:
        if n % p == 0:
            return n == p
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in small:
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True


def check_modulus(modulus: Optional[int]) -> Optional[int]:
    """None (payload group Z_{2^64}) or an odd PRIME 3 <= t < 2^62 (paper, Section 5.2), as in the C++ check."""
    if modulus is None:
        return None
    t = int(modulus)
    if t < 3 or t >= MAX_MODULUS or t % 2 == 0 or not is_probable_prime(t):
        raise ContractViolation(f"DCF_MODULUS: modulus must be an odd prime with 3 <= t < 2^62, got {t}")
    return t


def check_payloads(P: int) -> int:
    P = int(P)
    if P < 1 or P > MAX_PAYLOADS:
        raise ContractViolation(f"DCF_PAYLOADS: P must be in [1, {MAX_PAYLOADS}], got {P}")
    return P


def _domain_mask(n_bits: int) -> Optional[np.uint64]:
    return None if n_bits == 64 else np.uint64((1 << n_bits) - 1)


def _conv(v: np.ndarray, t: Optional[int]) -> np.ndarray:
    """Reduce payload words into the group (identity for Z_{2^64})."""
    if t is None:
        return v
    return v % np.uint64(t)


def _gadd(a: np.ndarray, b: np.ndarray, t: Optional[int]) -> np.ndarray:
    if t is None:
        return (a + b).astype(np.uint64)
    return ((a + b) % np.uint64(t)).astype(np.uint64)


def _gsub(a: np.ndarray, b: np.ndarray, t: Optional[int]) -> np.ndarray:
    if t is None:
        return (a - b).astype(np.uint64)
    tt = np.uint64(t)
    return ((a + tt - b) % tt).astype(np.uint64)


def _gneg(v: np.ndarray, t: Optional[int]) -> np.ndarray:
    if t is None:
        return (_ZERO - v).astype(np.uint64)
    tt = np.uint64(t)
    return ((tt - v) % tt).astype(np.uint64)


def _neg_if(mask: np.ndarray, v: np.ndarray, t: Optional[int]) -> np.ndarray:
    """-v where mask == 1 else v (group negation), broadcasting (M,) over (M,P)."""
    neg = _gneg(v, t)
    m = mask.astype(bool)
    if v.ndim == 2:
        m = m[:, None]
    return np.where(m, neg, v).astype(np.uint64)


def key_bytes_per_element(n_bits: int, payloads: int) -> int:
    """Key size of one DCF key (declared size formula, the quantity the cost model charges)."""
    return 16 + int(n_bits) * (16 + 8 * int(payloads) + 1) + 8 * int(payloads)


def key_wire_bytes_per_element(n_bits: int, payloads: int) -> int:
    """Bytes one key occupies in the binding wire layout (every control bit is a u64 word)."""
    n, P = int(n_bits), int(payloads)
    return 8 * (2 + 4 * n + n * P + P)


def wire_bytes(M: int, n_bits: int, payloads: int) -> int:
    """Exact length of ``DCFKeyBatch.to_bytes()`` for a batch of M keys."""
    return HEADER_BYTES + int(M) * key_wire_bytes_per_element(n_bits, payloads)


# ---------------------------------------------------------------------------
# Key batch
# ---------------------------------------------------------------------------
@dataclass
class DCFKeyBatch:
    """One party's batch of M DCF keys (n_bits levels, P payload words).

    Arrays are word-major per level so that per-level masking is contiguous:
    seed (M,2); cw_seeds (n,2,M); cw_values (n,P,M); cw_tL, cw_tR (n,M);
    cw_final (P,M); all uint64. ``modulus`` None means Z_{2^64}.
    """

    seed: np.ndarray
    cw_seeds: np.ndarray
    cw_values: np.ndarray
    cw_tL: np.ndarray
    cw_tR: np.ndarray
    cw_final: np.ndarray
    n_bits: int
    party: int
    modulus: Optional[int] = None

    @property
    def M(self) -> int:
        return int(self.seed.shape[0])

    @property
    def P(self) -> int:
        return int(self.cw_values.shape[1])

    def nbytes(self) -> int:
        """Key material size by the declared size formula (what the cost model charges)."""
        return self.M * key_bytes_per_element(self.n_bits, self.P)

    def wire_nbytes(self) -> int:
        """Exact length of ``to_bytes()`` (binding layout, header included)."""
        return wire_bytes(self.M, self.n_bits, self.P)

    def slice(self, lo: int, hi: int) -> "DCFKeyBatch":
        return DCFKeyBatch(self.seed[lo:hi], self.cw_seeds[:, :, lo:hi], self.cw_values[:, :, lo:hi],
                           self.cw_tL[:, lo:hi], self.cw_tR[:, lo:hi], self.cw_final[:, lo:hi],
                           self.n_bits, self.party, self.modulus)

    def validate(self) -> None:
        n, M, P = self.n_bits, self.M, self.P
        check_n_bits(n)
        check_modulus(self.modulus)
        check_payloads(P)
        if self.party not in (0, 1):
            raise ContractViolation(f"DCF_PARTY: party must be 0 or 1, got {self.party}")
        shapes = {
            "seed": ((M, 2), self.seed.shape),
            "cw_seeds": ((n, 2, M), self.cw_seeds.shape),
            "cw_values": ((n, P, M), self.cw_values.shape),
            "cw_tL": ((n, M), self.cw_tL.shape),
            "cw_tR": ((n, M), self.cw_tR.shape),
            "cw_final": ((P, M), self.cw_final.shape),
        }
        for name, (want, got) in shapes.items():
            if tuple(want) != tuple(got):
                raise ContractViolation(f"DCF_SHAPE: {name} has shape {got}, expected {want}")

    def to_bytes(self) -> bytes:
        """Binding wire layout: 20-byte header, then the six arrays as u64 LE in C order."""
        self.validate()
        hdr = _HDR.pack(WIRE_VERSION, self.party, self.n_bits, self.P, self.M,
                        0 if self.modulus is None else int(self.modulus))
        parts = [
            hdr,
            np.ascontiguousarray(self.seed, dtype="<u8").tobytes(),
            np.ascontiguousarray(self.cw_seeds, dtype="<u8").tobytes(),
            np.ascontiguousarray(self.cw_values, dtype="<u8").tobytes(),
            np.ascontiguousarray(self.cw_tL & _ONE, dtype="<u8").tobytes(),
            np.ascontiguousarray(self.cw_tR & _ONE, dtype="<u8").tobytes(),
            np.ascontiguousarray(self.cw_final, dtype="<u8").tobytes(),
        ]
        return b"".join(parts)

    @staticmethod
    def from_bytes(b: bytes) -> "DCFKeyBatch":
        """Parse the binding layout (first byte 2) or the legacy layout (magic ``OC2DCF01``)."""
        b = bytes(b)
        if len(b) >= 8 and b[:8] == _LEGACY_MAGIC:
            return _from_bytes_legacy(b)
        if len(b) < HEADER_BYTES:
            raise ContractViolation("DCF_WIRE: buffer shorter than the header")
        version, party, n_bits, P, M, modulus = _HDR.unpack_from(b, 0)
        if version != WIRE_VERSION:
            raise ContractViolation(f"DCF_WIRE: unknown layout (first byte {version}, expected {WIRE_VERSION})")
        if party not in (0, 1):
            raise ContractViolation(f"DCF_WIRE: bad party {party}")
        try:
            check_n_bits(n_bits)
            mod = None if modulus == 0 else check_modulus(modulus)
        except ContractViolation as exc:
            raise ContractViolation(f"DCF_WIRE: bad header ({exc.code})") from None
        if P < 1:
            raise ContractViolation("DCF_WIRE: P must be >= 1")
        if M > _MAX_WIRE_M:
            raise ContractViolation("DCF_WIRE: batch size exceeds the limit")
        expect = wire_bytes(M, n_bits, P)
        if len(b) != expect:
            raise ContractViolation(f"DCF_WIRE: expected {expect} bytes, got {len(b)}")
        off = HEADER_BYTES

        def take_u64(count: int, shape):
            nonlocal off
            arr = np.frombuffer(b, dtype="<u8", count=count, offset=off).astype(np.uint64).reshape(shape)
            off += 8 * count
            return arr

        seed = take_u64(2 * M, (M, 2))
        cw_seeds = take_u64(n_bits * 2 * M, (n_bits, 2, M))
        cw_values = take_u64(n_bits * P * M, (n_bits, P, M))
        cw_tL = take_u64(n_bits * M, (n_bits, M))
        cw_tR = take_u64(n_bits * M, (n_bits, M))
        cw_final = take_u64(P * M, (P, M))
        if M and (np.any(cw_tL > _ONE) or np.any(cw_tR > _ONE)):
            raise ContractViolation("DCF_WIRE: control-bit words must be 0 or 1")
        if mod is not None and M and (np.any(cw_values >= np.uint64(mod)) or np.any(cw_final >= np.uint64(mod))):
            raise ContractViolation("DCF_WIRE: Z_t correction words out of range")
        key = DCFKeyBatch(seed, cw_seeds, cw_values, cw_tL, cw_tR, cw_final, int(n_bits), int(party), mod)
        key.validate()
        return key


def _from_bytes_legacy(b: bytes) -> DCFKeyBatch:
    """First v2 layout (``OC2DCF01``): parses for P = 1 over Z_{2^64}, refused otherwise."""
    if len(b) < LEGACY_HEADER_BYTES:
        raise ContractViolation("DCF_WIRE: buffer shorter than the OC2DCF01 header")
    magic, n_bits, party, P, M, modulus, flags = _LEGACY_HDR.unpack_from(b, 0)
    if flags != 0:
        raise ContractViolation(f"DCF_WIRE: unsupported flags {flags}")
    try:
        check_n_bits(n_bits)
        if modulus != 0:
            check_modulus(modulus)
    except ContractViolation as exc:
        raise ContractViolation(f"DCF_WIRE: bad header ({exc.code})") from None
    if party not in (0, 1):
        raise ContractViolation(f"DCF_WIRE: bad party {party}")
    if P < 1:
        raise ContractViolation("DCF_WIRE: P must be >= 1")
    if M > _MAX_WIRE_M:
        raise ContractViolation("DCF_WIRE: batch size exceeds the limit")
    expect = LEGACY_HEADER_BYTES + M * key_bytes_per_element(n_bits, P)
    if len(b) != expect:
        raise ContractViolation(f"DCF_WIRE: expected {expect} bytes, got {len(b)}")
    if P != 1 or modulus != 0:
        raise ContractViolation(
            "DCF_WIRE_LEGACY_CONVERT: an OC2DCF01 key with P >= 2 or a Z_t payload was generated with the "
            "single-word Convert rule of the superseded engine and is refused")
    off = LEGACY_HEADER_BYTES

    def take_u64(count: int, shape):
        nonlocal off
        arr = np.frombuffer(b, dtype="<u8", count=count, offset=off).astype(np.uint64).reshape(shape)
        off += 8 * count
        return arr

    seed = take_u64(2 * M, (M, 2))
    cw_seeds = take_u64(n_bits * 2 * M, (n_bits, 2, M))
    cw_values = take_u64(n_bits * P * M, (n_bits, P, M))
    tpack = np.frombuffer(b, dtype=np.uint8, count=n_bits * M, offset=off).reshape(n_bits, M)
    off += n_bits * M
    if M and np.any(tpack > np.uint8(3)):
        raise ContractViolation("DCF_WIRE: packed control bits out of range")
    cw_tL = (tpack & np.uint8(1)).astype(np.uint64)
    cw_tR = ((tpack >> np.uint8(1)) & np.uint8(1)).astype(np.uint64)
    cw_final = take_u64(P * M, (P, M))
    key = DCFKeyBatch(seed, cw_seeds, cw_values, cw_tL, cw_tR, cw_final, int(n_bits), int(party), None)
    key.validate()
    return key


def concat_keys(a: DCFKeyBatch, b: DCFKeyBatch) -> DCFKeyBatch:
    """Concatenate two key batches with equal (n_bits, party, P, modulus) along M."""
    if a.n_bits != b.n_bits or a.party != b.party or a.P != b.P or a.modulus != b.modulus:
        raise ContractViolation("DCF_CONCAT: key batches differ in n_bits, party, P or modulus")
    return DCFKeyBatch(np.concatenate([a.seed, b.seed]),
                       np.concatenate([a.cw_seeds, b.cw_seeds], axis=2),
                       np.concatenate([a.cw_values, b.cw_values], axis=2),
                       np.concatenate([a.cw_tL, b.cw_tL], axis=1),
                       np.concatenate([a.cw_tR, b.cw_tR], axis=1),
                       np.concatenate([a.cw_final, b.cw_final], axis=1),
                       a.n_bits, a.party, a.modulus)


# ---------------------------------------------------------------------------
# Key generation (dealer side)
# ---------------------------------------------------------------------------
def gen_dcf_batch(alpha, beta, n_bits: int, rng: np.random.Generator,
                  modulus: Optional[int] = None) -> Tuple[DCFKeyBatch, DCFKeyBatch]:
    """Generate M DCF key pairs for f^{<}_{alpha[m], beta[m]} over n_bits bits.

    alpha: (M,) uint64, reduced mod 2^n_bits; beta: (M,) or (M,P) uint64,
    reduced mod ``modulus`` when a modulus is given. Every payload component
    uses its own Convert words (module docstring); for P = 1 over Z_{2^64}
    the keys are bit-identical to the keys of the pre-repair rule.
    """
    n_bits = check_n_bits(n_bits)
    t = check_modulus(modulus)
    alpha = np.asarray(alpha, dtype=np.uint64).reshape(-1)
    dm = _domain_mask(n_bits)
    if dm is not None:
        alpha = (alpha & dm).astype(np.uint64)
    beta = np.asarray(beta, dtype=np.uint64)
    if beta.ndim == 1:
        beta = beta[:, None]
    if beta.ndim != 2 or beta.shape[0] != alpha.shape[0]:
        raise ContractViolation(
            f"DCF_SHAPE: beta must be (M,) or (M,P) with M = {alpha.shape[0]}, got {beta.shape}")
    beta = _conv(beta, t)
    M, P = beta.shape
    if M == 0:
        raise ContractViolation("DCF_SHAPE: empty batch")       # the code of the C++ generator
    check_payloads(P)
    red = None if t is None else _prg.Mod128(t)

    s = [_prg.random_seeds(M, rng), _prg.random_seeds(M, rng)]
    s0_init, s1_init = s[0].copy(), s[1].copy()
    tb = [np.zeros(M, np.uint64), np.ones(M, np.uint64)]
    v_alpha = np.zeros((M, P), np.uint64)

    cw_seeds = np.zeros((n_bits, 2, M), np.uint64)
    cw_values = np.zeros((n_bits, P, M), np.uint64)
    cw_tL = np.zeros((n_bits, M), np.uint64)
    cw_tR = np.zeros((n_bits, M), np.uint64)

    with np.errstate(over="ignore"):
        for i in range(n_bits):
            a_i = ((alpha >> np.uint64(n_bits - 1 - i)) & _ONE).astype(np.uint64)
            keep_L = (a_i == 0)
            kl = keep_L[:, None]
            # vL*, vR*: (M, P) group elements, one independent word per payload component
            sL0, sR0, vL0, vR0, tL0, tR0 = _prg.expand_convert(s[0], P, t, red)
            sL1, sR1, vL1, vR1, tL1, tR1 = _prg.expand_convert(s[1], P, t, red)
            s_lose0 = np.where(kl, sR0, sL0)
            s_lose1 = np.where(kl, sR1, sL1)
            v_lose0 = np.where(kl, vR0, vL0)
            v_lose1 = np.where(kl, vR1, vL1)
            v_keep0 = np.where(kl, vL0, vR0)
            v_keep1 = np.where(kl, vL1, vR1)
            s_cw = (s_lose0 ^ s_lose1).astype(np.uint64)
            # the payload enters on the Lose branch exactly when Keep == Right (a_i == 1)
            add_beta = np.where(kl, _ZERO, beta).astype(np.uint64)
            sign1 = tb[1]
            diff = _gsub(v_lose1, v_lose0, t)
            v_cw = _gadd(_gsub(diff, v_alpha, t), add_beta, t)
            v_cw = _neg_if(sign1, v_cw, t)
            v_alpha = _gadd(_gadd(_gsub(v_alpha, v_keep1, t), v_keep0, t),
                            _neg_if(sign1, v_cw, t), t)
            t_cw_L = (tL0 ^ tL1 ^ a_i ^ _ONE).astype(np.uint64)
            t_cw_R = (tR0 ^ tR1 ^ a_i).astype(np.uint64)
            cw_seeds[i] = s_cw.T
            cw_values[i] = v_cw.T
            cw_tL[i] = t_cw_L
            cw_tR[i] = t_cw_R
            t_cw_keep = np.where(keep_L, t_cw_L, t_cw_R)
            for b_, (sLb, sRb, tLb, tRb) in enumerate(((sL0, sR0, tL0, tR0), (sL1, sR1, tL1, tR1))):
                s_keep = np.where(kl, sLb, sRb)
                t_keep = np.where(keep_L, tLb, tRb)
                apply = (tb[b_] == 1)
                s_keep = np.where(apply[:, None], s_keep ^ s_cw, s_keep)
                t_keep = np.where(apply, t_keep ^ t_cw_keep, t_keep)
                s[b_] = np.ascontiguousarray(s_keep, dtype=np.uint64)
                tb[b_] = t_keep.astype(np.uint64)
        vfin0 = _prg.leaf_words(s[0], P, t, red)       # (M, P): vL_c of the leaf seed
        vfin1 = _prg.leaf_words(s[1], P, t, red)
        cw_final = _gsub(_gsub(vfin1, vfin0, t), v_alpha, t)
        cw_final = np.ascontiguousarray(_neg_if(tb[1], cw_final, t).T)

    k0 = DCFKeyBatch(s0_init, cw_seeds, cw_values, cw_tL, cw_tR, cw_final, n_bits, 0, t)
    k1 = DCFKeyBatch(s1_init, cw_seeds, cw_values, cw_tL, cw_tR, cw_final, n_bits, 1, t)
    return k0, k1


# ---------------------------------------------------------------------------
# Evaluation (party side)
# ---------------------------------------------------------------------------
def _eval_dcf_chunk(key: DCFKeyBatch, x: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Evaluate keys [lo, hi) on x[lo:hi]; returns (P, hi-lo) uint64 output shares.

    Blocks per key and level: [branch seed (tweak x_i), control (tweak 3),
    one value block per component]. Z_{2^64}: the value block of component c
    is block 2 (c = 0) or 3 + c and the input bit selects its lo or hi word.
    Z_t: the value block of component c is block 4 + 2c + x_i, reduced mod t
    as a 128-bit integer.
    """
    n = key.n_bits
    t = key.modulus
    M = hi - lo
    P = key.P
    x = x[lo:hi]
    seed = key.seed[lo:hi]
    s0 = np.ascontiguousarray(seed[:, 0])      # seed word 0 (bytes 0..7)
    s1 = np.ascontiguousarray(seed[:, 1])      # seed word 1 (bytes 8..15, carries the tweak)
    tbit = np.full(M, key.party, np.uint64)
    out = np.zeros((P, M), np.uint64)
    if t is None:
        red = None
        tt = None
        tw_val = [_prg.tweak_word(_prg.ring_value_tweak(c)) for c in range(P)]
        tw_xor = None
    else:
        red = _prg.Mod128(t)
        tt = np.uint64(t)
        tw_val = [_prg.tweak_word(_prg.zt_value_tweak(c, 0)) for c in range(P)]
        tw_xor = [tw_val[c] ^ _prg.tweak_word(_prg.zt_value_tweak(c, 1)) for c in range(P)]
    X = np.empty((M, 2 + P, 2), np.uint64)
    buf = bytearray(X.nbytes + 16)
    with np.errstate(over="ignore"):
        for i in range(n):
            x_i = (x >> np.uint64(n - 1 - i)) & _ONE
            mx = _ZERO - x_i                               # all ones where x_i == 1
            X[:, :, 0] = s0[:, None]
            np.bitwise_xor(s1, x_i << _SH56, out=X[:, 0, 1])   # seed block of branch x_i (tweak 0 or 1)
            np.bitwise_xor(s1, _TW3, out=X[:, 1, 1])           # control block (tweak 3)
            if t is None:
                for c in range(P):
                    np.bitwise_xor(s1, tw_val[c], out=X[:, 2 + c, 1])
            else:
                for c in range(P):
                    np.bitwise_xor(s1, tw_val[c] ^ (tw_xor[c] & mx), out=X[:, 2 + c, 1])
            Y = _prg.aes_fixed_bulk(X, buf)                    # (M, 2 + P, 2) view of buf
            mt = _ZERO - tbit                                  # all ones where t == 1
            cw = key.cw_seeds[i, :, lo:hi]                     # (2,M)
            s0n = Y[:, 0, 0] ^ s0 ^ (cw[0] & mt)
            s1n = Y[:, 0, 1] ^ s1 ^ (cw[1] & mt)
            w3 = Y[:, 1, 0] ^ s0
            tL = w3 & _ONE
            tR = (w3 >> np.uint64(8)) & _ONE
            tn = tL ^ ((tL ^ tR) & x_i)
            cwL = key.cw_tL[i, lo:hi]
            cw_t = cwL ^ ((cwL ^ key.cw_tR[i, lo:hi]) & x_i)
            tn ^= cw_t & tbit
            cwv = key.cw_values[i, :, lo:hi]                   # (P,M)
            if t is None:
                for c in range(P):
                    vL = Y[:, 2 + c, 0] ^ s0
                    vR = Y[:, 2 + c, 1] ^ s1
                    out[c] += (vL ^ ((vL ^ vR) & mx)) + (cwv[c] & mt)
            else:
                for c in range(P):
                    v = red.reduce(Y[:, 2 + c, 0] ^ s0, Y[:, 2 + c, 1] ^ s1)
                    out[c] = (out[c] + v + (cwv[c] & mt)) % tt
            s0, s1, tbit = s0n, s1n, tn
        # leaf Convert: vL_c of the leaf seed, one block per component
        Xf = np.empty((M, P, 2), np.uint64)
        Xf[:, :, 0] = s0[:, None]
        for c in range(P):
            np.bitwise_xor(s1, tw_val[c], out=Xf[:, c, 1])
        Yf = _prg.aes_fixed_bulk(Xf)
        mt = _ZERO - tbit
        cwf = key.cw_final[:, lo:hi]
        if t is None:
            for c in range(P):
                out[c] += (Yf[:, c, 0] ^ s0) + (cwf[c] & mt)
            if key.party == 1:
                out = (_ZERO - out).astype(np.uint64)
        else:
            for c in range(P):
                fin = red.reduce(Yf[:, c, 0] ^ s0, Yf[:, c, 1] ^ s1)
                out[c] = (out[c] + fin + (cwf[c] & mt)) % tt
            if key.party == 1:
                out = ((tt - out) % tt).astype(np.uint64)
    return out


def eval_dcf_batch(key: DCFKeyBatch, x) -> np.ndarray:
    """Evaluate the M keys on M public inputs x (M,) -> (M,P) output shares.

    Output shares are in Z_{2^64} (modulus None) or below t (modulus t).
    Per level only the branch selected by the input bit is expanded. Work is
    vectorized over the batch; batches above EVAL_CHUNK are split over threads.
    """
    x = np.asarray(x, dtype=np.uint64).reshape(-1)
    M = key.M
    if x.shape[0] != M:
        raise ContractViolation(f"DCF_SHAPE: x has {x.shape[0]} elements, key batch has {M}")
    check_payloads(key.P)
    dm = _domain_mask(key.n_bits)
    if dm is not None:
        x = (x & dm).astype(np.uint64)
    if M == 0:
        return np.zeros((0, key.P), np.uint64)
    if M <= EVAL_CHUNK or EVAL_THREADS <= 1:
        return np.ascontiguousarray(_eval_dcf_chunk(key, x, 0, M).T)
    from concurrent.futures import ThreadPoolExecutor
    nchunks = min(EVAL_THREADS, max(1, -(-M // EVAL_CHUNK)))
    bounds = np.linspace(0, M, nchunks + 1).astype(int)
    with ThreadPoolExecutor(max_workers=nchunks) as ex:
        parts = list(ex.map(lambda b: _eval_dcf_chunk(key, x, int(b[0]), int(b[1])),
                            zip(bounds[:-1], bounds[1:])))
    return np.ascontiguousarray(np.concatenate(parts, axis=1).T)


def reconstruct_dcf(out0: np.ndarray, out1: np.ndarray, modulus: Optional[int]) -> np.ndarray:
    """Plaintext reconstruction of two output share arrays (test helper)."""
    with np.errstate(over="ignore"):
        if modulus is None:
            return (np.asarray(out0, np.uint64) + np.asarray(out1, np.uint64)).astype(np.uint64)
        tt = np.uint64(int(modulus))
        return ((np.asarray(out0, np.uint64) + np.asarray(out1, np.uint64)) % tt).astype(np.uint64)


__all__ = [
    "DCFKeyBatch", "gen_dcf_batch", "eval_dcf_batch", "key_bytes_per_element",
    "key_wire_bytes_per_element", "wire_bytes", "concat_keys", "reconstruct_dcf",
    "check_n_bits", "check_modulus", "check_payloads",
    "HEADER_BYTES", "LEGACY_HEADER_BYTES", "WIRE_VERSION", "EVAL_CHUNK", "EVAL_THREADS",
    "MAX_MODULUS", "MAX_PAYLOADS",
]
