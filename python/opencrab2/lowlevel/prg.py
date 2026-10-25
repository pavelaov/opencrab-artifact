"""Fixed-key Matyas-Meyer-Oseas PRG for the GGM tree.

    B_j(s) = AES_K(s xor c_j) xor s,   K = 2b7e151628aed2a6abf7158809cf4f3c,
    c_j = the 16-byte big-endian integer j.

Base layout (tweaks 0..3). The 64-byte expansion ``out = concat_j B_j(s)`` of
a 16-byte seed is split as

    sL = out[0:16], sR = out[16:32], vL = out[32:40] (LE u64), vR = out[40:48],
    tL = out[48] & 1, tR = out[49] & 1.

Component-specific Convert (paper, Section 5.3, a finding on comparison
keys; binding in both languages). Every payload component c of a DCF key
reads its OWN pseudo-random words:

    seeds             sL = B_0(s), sR = B_1(s)
    control bits      tL = B_3(s)[0] & 1, tR = B_3(s)[1] & 1
    Z_{2^64}, c = 0   vL_0 = LE64(B_2(s)[0:8]),     vR_0 = LE64(B_2(s)[8:16])      (the base layout)
    Z_{2^64}, c >= 1  vL_c = LE64(B_{3+c}(s)[0:8]), vR_c = LE64(B_{3+c}(s)[8:16])  (tweaks 4, 5, ...)
    Z_t, any c        vL_c = LE128(B_{4+2c}(s)) mod t, vR_c = LE128(B_{5+2c}(s)) mod t
                      (block 2 is unused in Z_t mode)
    leaf convert      the final correction word of component c uses vL_c of the leaf seed

Why separate words: the superseded engine reused ONE value word for every
payload component. The public correction words of a key with payload
(beta_0, beta_1) then satisfied ``cw_values[i][1] - cw_values[i][0] =
+-(a_i - a_{i-1}) (beta_1 - beta_0)``, so a party read every bit of alpha and
the value beta_1 - beta_0 from its own key (for DReLU material: the mask r,
hence the comparison input). Why 128-bit blocks for Z_t: a 64-bit word
reduced mod t is t / 2^64 away from uniform (2^-28 for t = 68720050177); the 128-bit block
reduced mod t is at most t / 2^128 away (below 2^-66 for every admitted
t < 2^62). The reduction is exact integer arithmetic (``Mod128``: limb
splitting sized from the bit length of t, no 64-bit overflow).

Blocks are handled as two little-endian uint64 words (lo = bytes 0..7, hi =
bytes 8..15), the view numpy takes of an (M, 2) uint64 array; the tweak c_j
therefore only touches the hi word (byte 15 carries j & 0xff, byte 14 the
next byte, and so on: ``tweak_word``).

Security of this PRG holds in the ideal-cipher model with a public fixed key;
it is the standard choice of FSS libraries because it allows hardware-batched
expansion. Distinct tweaks give distinct cipher inputs, so the blocks of one
seed are independent pseudo-random strings in that model. The AES backend is
the ``cryptography`` package (OpenSSL, zero-copy ``update_into``) with a
fallback to pycryptodomex.

Dealer randomness: ``os_random_bytes`` wraps ``secrets.token_bytes``;
``SecureRandom`` / ``secure_rng`` (module ``csprng``, re-exported here) is the
ChaCha20 stream every secret is drawn from. ``generator_from_bytes`` seeds a
numpy ``Generator`` (PCG64) and is kept for deterministic fixtures of insecure
dealers only: a statistical generator must never produce secrets (paper,
Appendix D, Disclosure).
``aes_ctr_keystream`` is provided for callers that need an AES-128-CTR stream
under a per-bundle key.

What is tested (tests/test_prg.py): the cross-language vector
``expand(seed = 00..0f)`` against an independent AES implementation,
determinism, the byte layout above, single MMO blocks for one-byte and
multi-byte tweaks against a byte-level computation, the Convert words of
every component against that byte-level computation, the exact 128-bit
reduction against Python integers for t from 2 up to 62 bits, and that
``aes_ecb_encrypt`` matches the FIPS-197 Appendix C.1 vector. Design limit:
the fixed key is public by design; the PRG is not a general-purpose CSPRNG
and must only be used inside the GGM tree.
"""

from __future__ import annotations

import secrets
import struct
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from ..errors import ContractViolation
from .csprng import SecureRandom, secure_rng

FIXED_KEY: bytes = bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")
LAMBDA_BYTES = 16

# Tweak block j as two little-endian uint64 words: the 16-byte big-endian
# integer j has its only non-zero byte at index 15, i.e. word 1 equals j << 56.
TWEAKS64: np.ndarray = np.array([[0, j << 56] for j in range(4)], dtype=np.uint64)

# Tweak indices of the component-specific Convert.
TWEAK_SEED_L = 0
TWEAK_SEED_R = 1
TWEAK_VALUE0 = 2      # Z_{2^64} component 0 (vL = lo word, vR = hi word); unused in Z_t mode
TWEAK_CONTROL = 3
MAX_PAYLOADS = 255    # the key wire format carries P in one byte
MAX_FIELD_MODULUS = 1 << 62

try:  # OpenSSL-backed AES (fast bulk ECB, zero-copy update_into)
    from cryptography.hazmat.primitives.ciphers import Cipher as _CCipher
    from cryptography.hazmat.primitives.ciphers import algorithms as _calg
    from cryptography.hazmat.primitives.ciphers import modes as _cmodes

    _HAVE_CRYPTOGRAPHY = True
except Exception:  # pragma: no cover
    _HAVE_CRYPTOGRAPHY = False

try:
    from Cryptodome.Cipher import AES as _PyCAES

    _HAVE_PYCRYPTODOME = True
except Exception:  # pragma: no cover
    _HAVE_PYCRYPTODOME = False

if not _HAVE_CRYPTOGRAPHY and not _HAVE_PYCRYPTODOME:  # pragma: no cover
    raise ImportError("PRG_BACKEND: neither cryptography nor pycryptodomex is installed")

if _HAVE_CRYPTOGRAPHY:
    _FIXED_CIPHER = _CCipher(_calg.AES(FIXED_KEY), _cmodes.ECB())
else:  # pragma: no cover
    _FIXED_CIPHER = None
    _FIXED_PYC = _PyCAES.new(FIXED_KEY, _PyCAES.MODE_ECB)


def aes_ecb_encrypt(key16: bytes, data: bytes) -> bytes:
    """Raw AES-128-ECB encryption of a multiple of 16 bytes (no padding)."""
    if len(key16) != 16:
        raise ContractViolation(f"PRG_KEY: AES-128 key must be 16 bytes, got {len(key16)}")
    if len(data) % 16 != 0:
        raise ContractViolation(f"PRG_BLOCK: data length {len(data)} is not a multiple of 16")
    if _HAVE_CRYPTOGRAPHY:
        enc = _CCipher(_calg.AES(bytes(key16)), _cmodes.ECB()).encryptor()
        return enc.update(bytes(data)) + enc.finalize()
    return _PyCAES.new(bytes(key16), _PyCAES.MODE_ECB).encrypt(bytes(data))  # pragma: no cover


def aes_fixed_bulk(X: np.ndarray, buf: Optional[bytearray] = None) -> np.ndarray:
    """AES_K-ECB (fixed key) of a contiguous uint64 array viewed as 16-byte blocks.

    ``X`` must have an even number of uint64 words. ``buf`` (>= X.nbytes + 16
    bytes) may be passed to avoid re-allocating the output buffer in per-level
    loops; the returned array views it, so callers must consume it before the
    next call that reuses ``buf``.
    """
    X = np.ascontiguousarray(X, dtype=np.uint64)
    nbytes = X.nbytes
    if nbytes % 16 != 0:
        raise ContractViolation("PRG_BLOCK: array byte length is not a multiple of 16")
    if nbytes == 0:
        # an empty batch: nothing to encrypt (memoryview cannot cast a zero-sized array: TypeError before)
        return np.zeros(X.shape, dtype=np.uint64)
    if _HAVE_CRYPTOGRAPHY:
        enc = _FIXED_CIPHER.encryptor()
        if buf is None or len(buf) < nbytes + 16:
            buf = bytearray(nbytes + 16)   # cryptography requires block slack in update_into
        enc.update_into(memoryview(X).cast("B"), buf)
        return np.frombuffer(buf, dtype=np.uint64, count=nbytes // 8).reshape(X.shape)
    out = _FIXED_PYC.encrypt(X.tobytes())  # pragma: no cover
    return np.frombuffer(out, dtype=np.uint64).reshape(X.shape)  # pragma: no cover


def expand(seeds: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Expand M 16-byte seeds (M,2) uint64 into (sL, sR, vL, vR, tL, tR).

    sL, sR: (M,2) uint64; vL, vR: (M,) uint64; tL, tR: (M,) uint64 in {0,1}.
    One AES-ECB call over 4M blocks. Layout as in the module docstring.
    """
    seeds = np.asarray(seeds, dtype=np.uint64)
    if seeds.ndim != 2 or seeds.shape[1] != 2:
        raise ContractViolation(f"PRG_SEEDS: seeds must have shape (M, 2), got {seeds.shape}")
    X = np.ascontiguousarray(seeds[:, None, :] ^ TWEAKS64[None, :, :])
    Y = aes_fixed_bulk(X)
    Y = Y ^ seeds[:, None, :]
    sL = np.ascontiguousarray(Y[:, 0, :])
    sR = np.ascontiguousarray(Y[:, 1, :])
    vL = np.ascontiguousarray(Y[:, 2, 0])
    vR = np.ascontiguousarray(Y[:, 2, 1])
    w3 = Y[:, 3, 0]
    tL = (w3 & np.uint64(1)).astype(np.uint64)                       # out[48] & 1
    tR = ((w3 >> np.uint64(8)) & np.uint64(1)).astype(np.uint64)     # out[49] & 1
    return sL, sR, vL, vR, tL, tR


# ---------------------------------------------------------------------------
# Component-specific Convert
# ---------------------------------------------------------------------------
def tweak_word(j: int) -> np.uint64:
    """High little-endian word of the tweak block c_j (the low word is zero for j < 2^64).

    c_j is the 16-byte big-endian integer j, so byte 15 holds j & 0xff, byte
    14 the next byte, and so on; read as the little-endian word of bytes
    8..15 this is the byte-swapped j. For j < 256 it equals ``j << 56``.
    """
    j = int(j)
    if j < 0 or j >= (1 << 64):
        raise ContractViolation(f"PRG_TWEAK: tweak index must be in [0, 2^64), got {j}")
    return np.uint64(int.from_bytes(j.to_bytes(8, "big"), "little"))


def check_payloads(P: int) -> int:
    P = int(P)
    if P < 1 or P > MAX_PAYLOADS:
        raise ContractViolation(f"PRG_PAYLOADS: P must be in [1, {MAX_PAYLOADS}], got {P}")
    return P


def ring_value_tweak(c: int) -> int:
    """Tweak of the Z_{2^64} value block of component c: 2 for c = 0, 3 + c for c >= 1."""
    c = int(c)
    if c < 0 or c >= MAX_PAYLOADS:
        raise ContractViolation(f"PRG_PAYLOADS: component index out of range: {c}")
    return TWEAK_VALUE0 if c == 0 else 3 + c


def zt_value_tweak(c: int, side: int) -> int:
    """Tweak of the Z_t value block of component c: 4 + 2c for side 0 (L), 5 + 2c for side 1 (R)."""
    c = int(c)
    if c < 0 or c >= MAX_PAYLOADS or side not in (0, 1):
        raise ContractViolation(f"PRG_PAYLOADS: component or side out of range: c={c}, side={side}")
    return 4 + 2 * c + int(side)


def convert_tweaks(P: int, modulus: Optional[int] = None) -> List[int]:
    """The tweaks a full expansion of one seed needs, in block order.

    Z_{2^64}: [0, 1, 2, 3, 4, ..., 2 + P] (4 + (P - 1) blocks). Z_t: [0, 1, 3,
    4, ..., 3 + 2P] (3 + 2P blocks; block 2 is unused).
    """
    P = check_payloads(P)
    if modulus is None:
        return [0, 1, 2, 3] + [3 + c for c in range(1, P)]
    return [0, 1, 3] + [4 + j for j in range(2 * P)]


def mmo_blocks(seeds: np.ndarray, tweaks: Sequence[int]) -> np.ndarray:
    """B_j(s) for every seed (M, 2) and every tweak j in ``tweaks``: uint64 array (M, len(tweaks), 2).

    One AES-ECB call over M * len(tweaks) blocks.
    """
    seeds = np.asarray(seeds, dtype=np.uint64)
    if seeds.ndim != 2 or seeds.shape[1] != 2:
        raise ContractViolation(f"PRG_SEEDS: seeds must have shape (M, 2), got {seeds.shape}")
    tw = np.zeros((len(tweaks), 2), np.uint64)
    for k, j in enumerate(tweaks):
        tw[k, 1] = tweak_word(j)
    X = np.ascontiguousarray(seeds[:, None, :] ^ tw[None, :, :])
    return aes_fixed_bulk(X) ^ seeds[:, None, :]


def mmo_bytes(seed16: bytes, tweak: int) -> bytes:
    """One raw MMO block B_tweak(seed) as 16 bytes (cross-language vector helper)."""
    if len(seed16) != 16:
        raise ContractViolation(f"PRG_SEEDS: seed must be 16 bytes, got {len(seed16)}")
    s = np.frombuffer(bytes(seed16), dtype=np.uint64).reshape(1, 2)
    return np.ascontiguousarray(mmo_blocks(s, [tweak])[0, 0]).tobytes()


class Mod128:
    """Exact reduction of 128-bit little-endian blocks mod t (2 <= t < 2^62), vectorized.

    ``reduce(lo, hi)`` returns ``(lo + 2^64 hi) mod t`` for uint64 arrays.
    With b = bit_length(t) the high word is first reduced below t < 2^b and
    then split into limbs of w = 64 - b bits; limb k is multiplied by the
    precomputed constant ``2^(64 + w k) mod t`` (below 2^b), so every product
    is below 2^(w + b) = 2^64 and every partial sum is below 2t < 2^63: no
    step can overflow a uint64. One limb suffices for t < 2^32, two for the
    admitted t = 68720050177 (37 bits), 31 for a 62-bit t.
    """

    __slots__ = ("t", "tt", "w", "mask", "consts", "c64")

    def __init__(self, t: int):
        t = int(t)
        if t < 2 or t >= MAX_FIELD_MODULUS:
            raise ContractViolation(f"PRG_MODULUS: modulus must satisfy 2 <= t < 2^62, got {t}")
        self.t = t
        self.tt = np.uint64(t)
        b = t.bit_length()
        self.w = 64 - b
        self.mask = np.uint64((1 << self.w) - 1)
        self.c64 = (1 << 64) % t
        nlimbs = -(-b // self.w)
        self.consts = [np.uint64((self.c64 << (self.w * k)) % t) for k in range(nlimbs)]

    def mul_c64(self, h: np.ndarray) -> np.ndarray:
        """(h * 2^64) mod t for a uint64 array h with values below t."""
        tt = self.tt
        if len(self.consts) == 1:
            return (h * self.consts[0]) % tt
        acc = ((h & self.mask) * self.consts[0]) % tt
        for k in range(1, len(self.consts)):
            limb = (h >> np.uint64(self.w * k)) & self.mask
            acc = (acc + (limb * self.consts[k]) % tt) % tt
        return acc

    def reduce(self, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """(lo + 2^64 hi) mod t, exact, as uint64 below t."""
        tt = self.tt
        lo = np.asarray(lo, dtype=np.uint64)
        hi = np.asarray(hi, dtype=np.uint64)
        with np.errstate(over="ignore"):
            return ((lo % tt + self.mul_c64(hi % tt)) % tt).astype(np.uint64)


def reduce128_mod(lo, hi, t: int) -> np.ndarray:
    """LE128 block (lo word, hi word) reduced mod t, exact (see :class:`Mod128`)."""
    return Mod128(t).reduce(lo, hi)


def _reducer(modulus: Optional[int], reducer: Optional[Mod128]) -> Optional[Mod128]:
    if modulus is None:
        return None
    if reducer is not None:
        if reducer.t != int(modulus):
            raise ContractViolation("PRG_MODULUS: reducer does not match the modulus")
        return reducer
    return Mod128(int(modulus))


def expand_convert(seeds: np.ndarray, P: int, modulus: Optional[int] = None,
                   reducer: Optional[Mod128] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                             np.ndarray, np.ndarray]:
    """Full GGM expansion with the component-specific Convert: (sL, sR, vL, vR, tL, tR).

    sL, sR: (M, 2) uint64; vL, vR: (M, P) uint64 group elements (Z_{2^64}
    words, or values below t when ``modulus`` is given); tL, tR: (M,) uint64
    in {0, 1}. Component c reads block 2 (c = 0) or 3 + c (c >= 1) in
    Z_{2^64} mode and the 128-bit blocks 4 + 2c, 5 + 2c reduced mod t in Z_t
    mode. For P = 1 without modulus this equals :func:`expand`.
    """
    P = check_payloads(P)
    red = _reducer(modulus, reducer)
    B = mmo_blocks(seeds, convert_tweaks(P, modulus))
    M = B.shape[0]
    sL = np.ascontiguousarray(B[:, 0, :])
    sR = np.ascontiguousarray(B[:, 1, :])
    vL = np.empty((M, P), np.uint64)
    vR = np.empty((M, P), np.uint64)
    if red is None:
        w3 = B[:, 3, 0]
        vL[:, 0] = B[:, 2, 0]
        vR[:, 0] = B[:, 2, 1]
        for c in range(1, P):
            vL[:, c] = B[:, 3 + c, 0]
            vR[:, c] = B[:, 3 + c, 1]
    else:
        w3 = B[:, 2, 0]                      # tweak 3 sits at block index 2 of [0, 1, 3, 4, ...]
        for c in range(P):
            vL[:, c] = red.reduce(B[:, 3 + 2 * c, 0], B[:, 3 + 2 * c, 1])
            vR[:, c] = red.reduce(B[:, 4 + 2 * c, 0], B[:, 4 + 2 * c, 1])
    tL = (w3 & np.uint64(1)).astype(np.uint64)                       # B_3(s)[0] & 1
    tR = ((w3 >> np.uint64(8)) & np.uint64(1)).astype(np.uint64)     # B_3(s)[1] & 1
    return sL, sR, vL, vR, tL, tR


def leaf_words(seeds: np.ndarray, P: int, modulus: Optional[int] = None,
               reducer: Optional[Mod128] = None) -> np.ndarray:
    """Leaf Convert: vL_c of every leaf seed, (M, P) uint64 (one AES block per component)."""
    P = check_payloads(P)
    red = _reducer(modulus, reducer)
    if red is None:
        B = mmo_blocks(seeds, [ring_value_tweak(c) for c in range(P)])
        return np.ascontiguousarray(B[:, :, 0])
    B = mmo_blocks(seeds, [zt_value_tweak(c, 0) for c in range(P)])
    out = np.empty((B.shape[0], P), np.uint64)
    for c in range(P):
        out[:, c] = red.reduce(B[:, c, 0], B[:, c, 1])
    return out


def expand_bytes(seed16: bytes) -> bytes:
    """The raw 64-byte expansion of one seed (cross-language test vector helper)."""
    if len(seed16) != 16:
        raise ContractViolation(f"PRG_SEEDS: seed must be 16 bytes, got {len(seed16)}")
    s = np.frombuffer(bytes(seed16), dtype=np.uint64).reshape(1, 2)
    X = np.ascontiguousarray(s[:, None, :] ^ TWEAKS64[None, :, :])
    Y = aes_fixed_bulk(X) ^ s[:, None, :]
    return np.ascontiguousarray(Y).tobytes()


def seeds_from_bytes(raw: bytes) -> np.ndarray:
    """(M,2) uint64 seeds from 16*M raw bytes."""
    if len(raw) % 16 != 0:
        raise ContractViolation("PRG_SEEDS: raw seed bytes must be a multiple of 16")
    return np.frombuffer(bytes(raw), dtype=np.uint64).reshape(-1, 2).copy()


def random_seeds(M: int, rng: Any) -> np.ndarray:
    """M uniform 16-byte seeds as (M,2) uint64 from ``rng.integers`` (SecureRandom in production)."""
    return rng.integers(0, 1 << 64, size=(int(M), 2), dtype=np.uint64)


# ---------------------------------------------------------------------------
# Dealer randomness
# ---------------------------------------------------------------------------
def os_random_bytes(n: int) -> bytes:
    """CSPRNG bytes from the operating system (secrets.token_bytes)."""
    return secrets.token_bytes(int(n))


def generator_from_bytes(seed32: bytes) -> np.random.Generator:
    """numpy Generator (PCG64) seeded from 32 bytes. Deterministic fixtures of insecure dealers only.

    Design limit: PCG64 is a statistical generator. A party
    receives raw outputs of the dealer's generator inside its bundle, and the
    state of this generator can be recovered from outputs, so secrecy of the
    seed does NOT protect the material. A ``Dealer`` built on this generator
    labels every bundle insecure; secrets come from ``SecureRandom``.
    """
    seed32 = bytes(seed32)
    if len(seed32) != 32:
        raise ContractViolation(f"PRG_SEED32: expected 32 seed bytes, got {len(seed32)}")
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(int.from_bytes(seed32, "little"))))


def aes_ctr_keystream(key16: bytes, nonce8: bytes, nbytes: int, counter0: int = 0) -> bytes:
    """AES-128-CTR keystream: blocks AES_key(nonce8 || BE64(counter)) for counter0, counter0+1, ...

    A CSPRNG stream for callers that must not use numpy's statistical
    generator (for example when stretching a 128-bit bundle key).
    """
    if len(nonce8) != 8:
        raise ContractViolation(f"PRG_NONCE: nonce must be 8 bytes, got {len(nonce8)}")
    nblocks = (int(nbytes) + 15) // 16
    counters = np.arange(counter0, counter0 + nblocks, dtype=np.uint64)
    blocks = bytearray()
    nonce = bytes(nonce8)
    for c in counters.tolist():
        blocks += nonce + struct.pack(">Q", c)
    return aes_ecb_encrypt(key16, bytes(blocks))[: int(nbytes)]


__all__ = [
    "FIXED_KEY", "LAMBDA_BYTES", "TWEAKS64", "aes_ecb_encrypt", "aes_fixed_bulk",
    "expand", "expand_bytes", "seeds_from_bytes", "random_seeds",
    "os_random_bytes", "generator_from_bytes", "aes_ctr_keystream", "SecureRandom", "secure_rng",
    "TWEAK_SEED_L", "TWEAK_SEED_R", "TWEAK_VALUE0", "TWEAK_CONTROL", "MAX_PAYLOADS", "MAX_FIELD_MODULUS",
    "tweak_word", "check_payloads", "ring_value_tweak", "zt_value_tweak", "convert_tweaks",
    "mmo_blocks", "mmo_bytes", "Mod128", "reduce128_mod", "expand_convert", "leaf_words",
]
