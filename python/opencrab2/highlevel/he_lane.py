"""HE lane primitives (paper, Section 5.2, ExactA2H): exact BFV A2H and the disabled CKKS H2A.

Component status (module constants, mirrored in the registry):

* ``BFVExactA2H`` / ``a2h_exact``: ``ADMITTED`` at primitive level. The
  conversion first reshares the A64 value into Z_t with the one-round
  ring-to-field gate (``Party.ring_to_field``, paper, Section 5.2, RingToField:
  the signed representative x_signed = u - 1{u >= 2^63} 2^64 is reduced mod t
  inside the secure computation, wrap and sign bits stay shared). P1 then
  BFV-encrypts its Z_t share (TenSEAL, batched ``bfv_vector``, plaintext modulus
  t = 68720050177, N = 8192) and sends the ciphertext to P0, who adds its Z_t
  share as a plaintext. Because BFV plaintext addition is exact modulo t,
  Dec(ct) = x_signed mod t exactly, for every share decomposition including
  the wrapping pattern a0 = 2^64 - 1, a1 = 2 on which the share-wise reduction
  fails. The ciphertext is
  held by P0 only; the keyholder P1 never sees it. There is no admitted
  ciphertext matvec on real parameters, so the layer lane status is
  ``UNAVAILABLE``.
* ``CKKSFastH2A``: ``UNCERTIFIED_DISABLED``. It is the fast CKKS prototype
  (client-key CKKS linear layer plus mask-and-decrypt H2A) kept so that
  DIAGNOSTIC measurements remain reproducible. Its construction refuses to run
  unless ``diagnostic=True`` is passed, and records produced with it can never
  be security eligible (the keyholder sees the evaluated ciphertext; the
  per-slot flooding bound is not an end-to-end certificate; paper, Section
  5.2, reference bridge).

What is tested (tests/test_he_lane.py): Dec(a2h_exact(share)) == x_signed mod t
on random shares including negative values, values near +-2^39 and the
wrapping pattern above; the naive sharewise reduction is shown to fail on the same
inputs; chunking over several ciphertexts; the CKKS guard raises without
``diagnostic=True`` and the diagnostic path reproduces W x within CKKS
precision. Design limits: TenSEAL ciphertext sizes depend only on the public
parameters and the number of chunks (public shape); the BFV context uses
TenSEAL's default coefficient modulus for N = 8192 (about 218 bits), which is
a 128-bit parameter set according to the SEAL defaults, not an estimate we
re-ran here. Error codes: HE_BACKEND, HE_PARTY, HE_SHAPE, HE_RANGE,
NOT_ADMITTED_COMPONENT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np

from ..errors import AdmissionError, ContractViolation
from ..lowlevel import ring

try:  # TenSEAL is optional at import time; every entry point checks _HAVE_TENSEAL
    import tenseal as ts

    _HAVE_TENSEAL = True
except Exception:  # pragma: no cover
    ts = None
    _HAVE_TENSEAL = False

BFV_PLAIN_MODULUS = 68720050177      # prime, 37 bits, = 1 mod 16384 (SEAL batching at N = 8192)
BFV_POLY_DEGREE = 8192
STATUS_BFV_EXACT_A2H = "ADMITTED"            # primitive level only
STATUS_BFV_LAYER_LANE = "UNAVAILABLE"        # no admitted matvec on real parameters
STATUS_CKKS_FAST_H2A = "UNCERTIFIED_DISABLED"
STATUS = {"BFVExactA2H": STATUS_BFV_EXACT_A2H, "bfv_layer_lane": STATUS_BFV_LAYER_LANE,
          "CKKSFastH2A": STATUS_CKKS_FAST_H2A}


def _require_tenseal() -> None:
    if not _HAVE_TENSEAL:
        raise ContractViolation("HE_BACKEND: tenseal is not installed")


def _party_index(party: Any) -> int:
    p = getattr(party, "party", party)
    if int(p) not in (0, 1):
        raise ContractViolation(f"HE_PARTY: party must be 0 or 1, got {party!r}")
    return int(p)


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    # deterministic Miller-Rabin for n < 3.3e24
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


# ---------------------------------------------------------------------------
# BFV exact A2H
# ---------------------------------------------------------------------------
class BFVExactA2H:
    """Per-party TenSEAL BFV context for the exact A2H primitive.

    ``party`` 1 (the client) generates the key pair and holds the secret key;
    ``party`` 0 (the server) holds only the public context received through
    ``load_public``. ``encrypt_share`` (P1) and ``add_share`` (P0) operate on
    at most ``slots`` (= N) Z_t values per ciphertext; ``a2h_exact`` chunks
    longer vectors. ``decrypt`` exists for tests only (the protocol never
    decrypts the A2H output).
    """

    def __init__(self, party: Any, poly_modulus_degree: int = BFV_POLY_DEGREE,
                 plain_modulus: int = BFV_PLAIN_MODULUS, public_context_bytes: Optional[bytes] = None) -> None:
        _require_tenseal()
        self.party = _party_index(party)
        self.N = int(poly_modulus_degree)
        self.t = int(plain_modulus)
        if self.N & (self.N - 1) or self.N < 1024:
            raise ContractViolation(f"HE_PARAMS: poly_modulus_degree must be a power of two >= 1024, got {self.N}")
        if not _is_prime(self.t) or (self.t - 1) % (2 * self.N) != 0:
            raise ContractViolation(
                f"HE_PARAMS: plain modulus must be a prime with t = 1 mod 2N for batching, got {self.t}")
        self.counters = {"ciphertexts": 0, "ct_bytes": 0, "plain_adds": 0}
        if self.party == 1:
            self.ctx = ts.context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=self.N, plain_modulus=self.t)
        else:
            self.ctx = None
            if public_context_bytes is not None:
                self.load_public(public_context_bytes)

    @property
    def slots(self) -> int:
        return self.N

    def public_bytes(self) -> bytes:
        """Serialised public context (public key only, no secret key, no Galois keys)."""
        if self.party != 1:
            raise ContractViolation("HE_PARTY: only the keyholder P1 exports the public context")
        return self.ctx.serialize(save_public_key=True, save_secret_key=False,
                                  save_galois_keys=False, save_relin_keys=False)

    def load_public(self, ctx_pub_bytes: bytes) -> None:
        if self.party != 0:
            raise ContractViolation("HE_PARTY: only P0 loads a public context")
        self.ctx = ts.context_from(bytes(ctx_pub_bytes))
        if self.ctx.is_private():
            raise ContractViolation("HE_PARTY: received context carries a secret key; refused")

    def _check_vec(self, z: Any, name: str) -> List[int]:
        z = np.asarray(z, dtype=np.uint64).reshape(-1)
        if z.size > self.N:
            raise ContractViolation(f"HE_SHAPE: {name} has {z.size} values, at most N={self.N} per ciphertext")
        if z.size and int(z.max()) >= self.t:
            raise ContractViolation(f"HE_RANGE: {name} must hold Z_t values below t={self.t}")
        return [int(v) for v in z.tolist()]

    def encrypt_share(self, z1_t: Any) -> bytes:
        """P1: Enc_pk(z1) as a batched BFV vector (values below t)."""
        if self.party != 1:
            raise ContractViolation("HE_PARTY: encrypt_share runs on the keyholder P1")
        vals = self._check_vec(z1_t, "z1")
        ct = ts.bfv_vector(self.ctx, vals)
        data = ct.serialize()
        self.counters["ciphertexts"] += 1
        self.counters["ct_bytes"] += len(data)
        return data

    def add_share(self, ctx_pub_bytes: Optional[bytes], ct_bytes: bytes, z0_t: Any) -> bytes:
        """P0: ct + z0 as plaintext (exact mod t). ``ctx_pub_bytes`` may be None once loaded."""
        if self.party != 0:
            raise ContractViolation("HE_PARTY: add_share runs on the server P0")
        if ctx_pub_bytes is not None:
            self.load_public(ctx_pub_bytes)
        if self.ctx is None:
            raise ContractViolation("HE_PARTY: P0 has no public context yet")
        vals = self._check_vec(z0_t, "z0")
        ct = ts.bfv_vector_from(self.ctx, bytes(ct_bytes))
        if ct.size() != len(vals):
            raise ContractViolation(f"HE_SHAPE: ciphertext holds {ct.size()} slots, share has {len(vals)}")
        out = ct + vals
        data = out.serialize()
        self.counters["plain_adds"] += 1
        self.counters["ct_bytes"] += len(data)
        return data

    def decrypt(self, ct_bytes: bytes) -> np.ndarray:
        """TEST ONLY: decrypt a ciphertext under P1's key; values in [0, t) as uint64."""
        if self.party != 1:
            raise ContractViolation("HE_PARTY: only the keyholder decrypts (tests only)")
        ct = ts.bfv_vector_from(self.ctx, bytes(ct_bytes))
        raw = ct.decrypt()
        t = self.t
        return np.array([int(v) % t for v in raw], dtype=np.uint64)


def naive_sharewise_reduction(a0: Any, a1: Any, t: int) -> np.ndarray:
    """The REJECTED share-wise reduction (a0 mod t + a1 mod t) mod t (test oracle of the defect)."""
    t = int(t)
    a0 = np.asarray(a0, np.uint64).reshape(-1)
    a1 = np.asarray(a1, np.uint64).reshape(-1)
    return np.array([(int(x) % t + int(y) % t) % t for x, y in zip(a0.tolist(), a1.tolist())], dtype=np.uint64)


def signed_mod_t(u: Any, t: int) -> np.ndarray:
    """Plaintext reference: x_signed mod t for ring elements u (uint64)."""
    t = int(t)
    s = ring.to_signed(np.asarray(u, np.uint64).reshape(-1))
    return np.array([int(v) % t for v in s.tolist()], dtype=np.uint64)


def _get_bfv(party: Any, t: int, N: int) -> BFVExactA2H:
    """Per-party cached BFV context; the public context travels P1 -> P0 once (setup)."""
    key = f"bfv:{t}:{N}"
    store = getattr(party, "_he", None)
    if store is None:
        store = {}
        setattr(party, "_he", store)
    he = store.get(key)
    if he is None:
        he = BFVExactA2H(party.party, poly_modulus_degree=N, plain_modulus=t)
        with party.track("a2h_setup", (N,)):
            if party.party == 1:
                party.ch.send(he.public_bytes())
            else:
                he.load_public(party.ch.recv())
            party._count_round()
        store[key] = he
    return he


def a2h_exact(party: Any, share: Any, t: int = BFV_PLAIN_MODULUS, N: int = BFV_POLY_DEGREE):
    """A64 share -> BFV ciphertext of x_signed mod t, held by P0 (returns None on P1).

    Rounds: 1 (ring_to_field open) + 1 message P1 -> P0 (ciphertext(s)).
    Vectors longer than N slots are sent as ceil(M / N) ciphertexts in one
    frame; the frame layout depends only on M (public shape). Returns bytes
    for a single ciphertext, a list of bytes for several.
    """
    share = np.asarray(share, np.uint64)
    flat = share.reshape(-1)
    M = int(flat.size)
    if M == 0:
        raise ContractViolation("HE_SHAPE: empty share")
    n_ct = int(math.ceil(M / N))
    with party.track("a2h_exact", share.shape):
        z = party.ring_to_field(flat, t)
        he = _get_bfv(party, int(t), int(N))
        if party.party == 1:
            frame = {"count": n_ct}
            for i in range(n_ct):
                frame[f"ct{i}"] = he.encrypt_share(z[i * N:(i + 1) * N])
            party.ch.send(frame)
            party._count_round()
            return None
        frame = party.ch.recv()
        party._count_round()
        if int(frame["count"]) != n_ct:
            raise ContractViolation(f"HE_SHAPE: peer sent {frame['count']} ciphertexts, expected {n_ct}")
        outs = [he.add_share(None, frame[f"ct{i}"], z[i * N:(i + 1) * N]) for i in range(n_ct)]
        return outs[0] if n_ct == 1 else outs


# ---------------------------------------------------------------------------
# CKKS fast H2A (prototype, UNCERTIFIED_DISABLED)
# ---------------------------------------------------------------------------
@dataclass
class CKKSProfile:
    poly_modulus_degree: int = 8192
    coeff_mod_bit_sizes: Tuple[int, ...] = (60, 40, 40, 60)
    global_scale_bits: int = 40


@dataclass
class HECounters:
    ct_pl_mul: int = 0
    rotations: int = 0
    ciphertexts_sent: int = 0
    ct_bytes: int = 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class ConversionCert:
    """Parameter record of a DIAGNOSTIC conversion (not a security certificate)."""
    direction: str
    n: int
    log_scale: int
    mask_bits: int
    flooding_bits: int
    status: str = STATUS_CKKS_FAST_H2A

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class CKKSFastH2A:
    """Fast CKKS lane: client-key linear layer plus mask-and-decrypt H2A.

    STATUS = UNCERTIFIED_DISABLED. The constructor raises
    ``AdmissionError(NOT_ADMITTED_COMPONENT)`` unless ``diagnostic=True``;
    every output of a diagnostic instance is labelled DIAGNOSTIC and can never
    enter an eligible record. The measurements it supports are diagnostic
    comparisons, not admitted protocol steps.
    """

    STATUS = STATUS_CKKS_FAST_H2A

    def __init__(self, party: Any, diagnostic: bool = False, profile: Optional[CKKSProfile] = None,
                 public_context_bytes: Optional[bytes] = None) -> None:
        if not diagnostic:
            raise AdmissionError(
                "NOT_ADMITTED_COMPONENT: CKKSFastH2A is UNCERTIFIED_DISABLED; "
                "pass diagnostic=True for DIAGNOSTIC measurements only (never eligible)")
        _require_tenseal()
        self.party = _party_index(party)
        self.profile = profile or CKKSProfile()
        self.counters = HECounters()
        self.evidence_status = "DIAGNOSTIC"
        if self.party == 1:
            self.ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=self.profile.poly_modulus_degree,
                                  coeff_mod_bit_sizes=list(self.profile.coeff_mod_bit_sizes))
            self.ctx.global_scale = 2 ** self.profile.global_scale_bits
            self.ctx.generate_galois_keys()
        else:
            self.ctx = None
            if public_context_bytes is not None:
                self.load_public(public_context_bytes)

    def public_bytes(self) -> bytes:
        if self.party != 1:
            raise ContractViolation("HE_PARTY: only the keyholder exports the public context")
        return self.ctx.serialize(save_public_key=True, save_secret_key=False,
                                  save_galois_keys=True, save_relin_keys=True)

    def load_public(self, ctx_pub_bytes: bytes) -> None:
        if self.party != 0:
            raise ContractViolation("HE_PARTY: only P0 loads a public context")
        self.ctx = ts.context_from(bytes(ctx_pub_bytes))
        if self.ctx.is_private():
            raise ContractViolation("HE_PARTY: received context carries a secret key; refused")

    # client side
    def encrypt(self, x_real: Any) -> bytes:
        if self.party != 1:
            raise ContractViolation("HE_PARTY: encrypt runs on the keyholder P1")
        vec = ts.ckks_vector(self.ctx, np.asarray(x_real, np.float64).reshape(-1).tolist())
        self.counters.ciphertexts_sent += 1
        data = vec.serialize()
        self.counters.ct_bytes += len(data)
        return data

    def decrypt(self, ct_bytes: bytes) -> np.ndarray:
        if self.party != 1:
            raise ContractViolation("HE_PARTY: only the keyholder decrypts")
        return np.array(ts.ckks_vector_from(self.ctx, bytes(ct_bytes)).decrypt(), dtype=np.float64)

    def h2a_client(self, masked_ct_bytes: bytes, contract: Any) -> np.ndarray:
        """Client share = rint(Dec(masked ct) 2^f) as a ring element (DIAGNOSTIC).

        The keyholder sees the evaluated ciphertext, and the masked value lies
        far outside the logical range on purpose (the mask floods it), so this
        uses an unchecked encoding: the decoded double cannot represent the
        low bits of a 60-bit integer exactly, which is one of the
        reasons the route is not admitted.
        """
        m_real = self.decrypt(masked_ct_bytes)
        scaled = np.rint(m_real * float(2 ** int(contract.frac_bits)))
        if not np.all(np.isfinite(scaled)) or np.any(np.abs(scaled) >= 2.0 ** 63):
            raise ContractViolation("HE_RANGE: decoded masked value does not fit a 64-bit ring element")
        return ring.from_signed(scaled.astype(np.int64))

    # server side
    def linear(self, ct_bytes: bytes, W: Any, b: Optional[Any] = None) -> bytes:
        """y = W x (+ b) homomorphically; W (d_out, d_in) plaintext on P0."""
        if self.party != 0 or self.ctx is None:
            raise ContractViolation("HE_PARTY: linear runs on P0 with a loaded public context")
        ct = ts.ckks_vector_from(self.ctx, bytes(ct_bytes))
        W = np.asarray(W, np.float64)
        out = ct.matmul(W.T.tolist())
        self.counters.ct_pl_mul += 1
        self.counters.rotations += 2 * int(math.ceil(math.sqrt(max(W.shape[1], 1))))
        if b is not None:
            out = out + np.asarray(b, np.float64).reshape(-1).tolist()
        data = out.serialize()
        self.counters.ct_bytes += len(data)
        return data

    def h2a_server(self, ct_bytes: bytes, length: int, rng: np.random.Generator, contract: Any,
                   flooding_bits: int = 20) -> Tuple[bytes, np.ndarray, ConversionCert]:
        """Mask the ciphertext with a wide random r; server share = -r (DIAGNOSTIC)."""
        if self.party != 0 or self.ctx is None:
            raise ContractViolation("HE_PARTY: h2a_server runs on P0 with a loaded public context")
        ct = ts.ckks_vector_from(self.ctx, bytes(ct_bytes))
        mask_bits = int(contract.logical_bits)
        hi = 1 << (mask_bits + int(flooding_bits))
        r_int = rng.integers(-hi, hi, size=int(length), dtype=np.int64)
        r_real = r_int.astype(np.float64) / float(2 ** int(contract.frac_bits))
        masked = ct + r_real.tolist()
        data = masked.serialize()
        self.counters.ct_bytes += len(data)
        server_share = ring.neg(ring.from_signed(r_int))
        cert = ConversionCert("H2A", self.profile.poly_modulus_degree, self.profile.global_scale_bits,
                              mask_bits, int(flooding_bits))
        return data, server_share, cert


__all__ = ["BFVExactA2H", "a2h_exact", "CKKSFastH2A", "CKKSProfile", "ConversionCert", "HECounters",
           "BFV_PLAIN_MODULUS", "BFV_POLY_DEGREE", "STATUS", "STATUS_BFV_EXACT_A2H", "STATUS_BFV_LAYER_LANE",
           "STATUS_CKKS_FAST_H2A", "naive_sharewise_reduction", "signed_mod_t"]
