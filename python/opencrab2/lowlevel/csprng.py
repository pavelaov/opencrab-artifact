"""Cryptographic randomness for secret values.

``SecureRandom`` is a ChaCha20 (RFC 7539) keystream generator keyed with 32
bytes. It replaces the numpy ``Generator`` (PCG64) wherever a secret is drawn:
dealer masks, Beaver triples, DCF root seeds, input-sharing masks of a party,
toy BFV keys and encryption randomness. The reason is not statistical quality.
A recipient's bundle is a window of raw generator outputs, and PCG64 is a
linear congruential generator with an output permutation whose state can be
recovered from outputs (Bouillaguet, Martinez, Sauvage, ToSC 2020(3)); one
recovery predicts every mask of the dealer, forwards and backwards. The
secrecy of the seed is therefore NOT sufficient for a statistical generator.
For ChaCha20 the outputs are pseudo-random under the standard assumption on
the cipher, so a window of outputs says nothing about the rest of the stream.

API: the library only ever calls ``rng.integers(low, high, size, dtype)``, so
this class offers that one numpy-compatible method (plus ``bytes``). Bounds
that are not a power of two use rejection sampling on the smallest covering
bit mask, so every value of [low, high) is exactly equally likely (no modulo
bias). The number of keystream bytes drawn by a rejection loop depends only on
the public bound and on discarded candidates, never on a value that is kept.

``SecureRandom.from_os()`` keys the stream from ``secrets.token_bytes`` and
sets ``os_seeded``; the key is not stored outside the cipher object. A stream
built from a caller-supplied key (``SecureRandom(key)``, ``from_seed``) is
reproducible by that caller and has ``os_seeded == False``: the dealer labels
its bundles insecure (same rule as the C++ ``Rng::from_seed``).

Tested (tests/test_csprng.py): the RFC 7539 Sec. 2.4.2 keystream, determinism
of keyed streams, exact range for power-of-two and other bounds, signed
ranges, dtype handling, that no numpy BitGenerator is involved. Design limit:
this is a stream generator, not a DRBG with reseeding or forward secrecy
against a later compromise of the process memory.
"""
from __future__ import annotations

import hashlib
import secrets
import threading
from typing import Any, Optional, Tuple

import numpy as np

from ..errors import ContractViolation

try:  # pragma: no cover - exercised implicitly
    from cryptography.hazmat.primitives.ciphers import Cipher as _Cipher
    from cryptography.hazmat.primitives.ciphers import algorithms as _alg
    _HAVE_CRYPTOGRAPHY = True
except Exception:  # pragma: no cover
    _HAVE_CRYPTOGRAPHY = False

_CHUNK = 1 << 20
_ZERO = bytes(_CHUNK)


class SecureRandom:
    """ChaCha20 keystream generator with a numpy-compatible ``integers`` method."""

    __slots__ = ("_enc", "_lock", "os_seeded", "bytes_drawn")

    def __init__(self, key32: bytes, nonce12: bytes = bytes(12), *, _os_seeded: bool = False) -> None:
        key32 = bytes(key32)
        nonce12 = bytes(nonce12)
        if len(key32) != 32:
            raise ContractViolation(f"CSPRNG_KEY: expected a 32-byte key, got {len(key32)} bytes")
        if len(nonce12) != 12:
            raise ContractViolation(f"CSPRNG_NONCE: expected a 12-byte nonce, got {len(nonce12)} bytes")
        if _HAVE_CRYPTOGRAPHY:
            # cryptography takes the 16-byte (counter LE32 || nonce) form of RFC 7539; counter starts at 0
            self._enc = _Cipher(_alg.ChaCha20(key32, bytes(4) + nonce12), mode=None).encryptor()
        else:  # pragma: no cover
            from Cryptodome.Cipher import ChaCha20 as _CC
            self._enc = _CC.new(key=key32, nonce=nonce12)
        self._lock = threading.Lock()
        self.os_seeded = bool(_os_seeded)
        self.bytes_drawn = 0

    # -- constructors ------------------------------------------------------------------
    @classmethod
    def from_os(cls) -> "SecureRandom":
        """Stream keyed with 32 bytes of the operating system CSPRNG (production)."""
        return cls(secrets.token_bytes(32), _os_seeded=True)

    @classmethod
    def from_seed(cls, seed: int) -> "SecureRandom":
        """Deterministic stream for tests: key = SHA-256 of the decimal seed. Never eligible."""
        return cls(hashlib.sha256(b"opencrab2-csprng-test-seed:" + str(int(seed)).encode("ascii")).digest())

    # -- raw stream --------------------------------------------------------------------
    def bytes(self, n: int) -> bytes:
        """The next n keystream bytes."""
        n = int(n)
        if n < 0:
            raise ContractViolation("CSPRNG_PARAMS: byte count must be non-negative")
        with self._lock:
            self.bytes_drawn += n
            if n <= _CHUNK:
                return self._stream(n)
            parts = []
            left = n
            while left > 0:
                take = min(left, _CHUNK)
                parts.append(self._stream(take))
                left -= take
            return b"".join(parts)

    def _stream(self, n: int) -> bytes:
        if _HAVE_CRYPTOGRAPHY:
            return self._enc.update(_ZERO[:n])
        return self._enc.encrypt(_ZERO[:n])  # pragma: no cover

    def _words(self, count: int) -> np.ndarray:
        return np.frombuffer(self.bytes(8 * int(count)), dtype="<u8").astype(np.uint64)

    # -- numpy-compatible sampling -----------------------------------------------------
    def integers(self, low: Any, high: Any = None, size: Any = None, dtype: Any = np.int64,
                 endpoint: bool = False) -> Any:
        """Uniform integers in [low, high) (or [0, low) when high is None), like numpy's Generator.

        Supports every span up to 2^64 and the integer dtypes; refuses a range
        that does not fit the dtype (CSPRNG_PARAMS) instead of wrapping.
        """
        if high is None:
            low, high = 0, low
        lo = int(low)
        hi = int(high) + (1 if endpoint else 0)
        span = hi - lo
        dt = np.dtype(dtype)
        if dt.kind not in "iub":
            raise ContractViolation(f"CSPRNG_PARAMS: unsupported dtype {dt}")
        if span <= 0 or span > (1 << 64):
            raise ContractViolation("CSPRNG_PARAMS: need low < high and a span of at most 2^64")
        if dt.kind == "b":
            dmin, dmax = 0, 1
        else:
            info = np.iinfo(dt)
            dmin, dmax = int(info.min), int(info.max)
        if lo < dmin or hi - 1 > dmax:
            raise ContractViolation(f"CSPRNG_PARAMS: range [{lo}, {hi}) does not fit dtype {dt}")
        if size is None:
            shape: Tuple[int, ...] = ()
        elif isinstance(size, (int, np.integer)):
            shape = (int(size),)
        else:
            shape = tuple(int(s) for s in size)
        if any(s < 0 for s in shape):
            raise ContractViolation("CSPRNG_PARAMS: negative dimension")
        count = 1
        for s in shape:
            count *= s
        v = self._below(span, count)
        # v < span as uint64; add low in wrapping uint64 arithmetic, then reinterpret
        if lo != 0:
            v = v + np.uint64(lo % (1 << 64))
        if dt.kind == "i":
            out = v.view(np.int64).astype(dt, copy=False)
        elif dt.kind == "b":
            out = v.astype(np.bool_)
        else:
            out = v.astype(dt, copy=False)
        out = out.reshape(shape)
        if size is None:
            return out[()]
        return out

    def _below(self, span: int, count: int) -> np.ndarray:
        """count uniform uint64 values below span (1 <= span <= 2^64), exact."""
        if count == 0:
            return np.zeros(0, dtype=np.uint64)
        if span == (1 << 64):
            return self._words(count)
        if span == 1:
            return np.zeros(count, dtype=np.uint64)
        bits = (span - 1).bit_length()
        mask = np.uint64((1 << bits) - 1)
        if span == (1 << bits):
            return self._words(count) & mask
        out = np.empty(count, dtype=np.uint64)
        filled = 0
        bound = np.uint64(span)
        while filled < count:
            need = count - filled
            # acceptance probability is above one half; draw a margin to finish in few passes
            cand = self._words(need + (need >> 1) + 16) & mask
            good = cand[cand < bound]
            take = min(need, int(good.shape[0]))
            out[filled:filled + take] = good[:take]
            filled += take
        return out


def secure_rng() -> SecureRandom:
    """A fresh OS-keyed ``SecureRandom`` (the production default of Dealer and Party)."""
    return SecureRandom.from_os()


def is_statistical_generator(rng: Any) -> bool:
    """True for numpy generators (and anything exposing a numpy BitGenerator): never for secrets."""
    return isinstance(rng, (np.random.Generator, np.random.RandomState)) or hasattr(rng, "bit_generator")


def rng_is_os_seeded(rng: Optional[Any]) -> bool:
    """True only for a ``SecureRandom`` keyed from the OS."""
    return isinstance(rng, SecureRandom) and rng.os_seeded


__all__ = ["SecureRandom", "secure_rng", "is_statistical_generator", "rng_is_os_seeded"]
