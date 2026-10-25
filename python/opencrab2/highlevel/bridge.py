"""Secure toy-BFV bridge: A2H_toy, Eval_toy, H2A_ref (paper, Appendix B, toy scheme and H2A-ref).

Reference design with secure internal decryption over the power-of-two toy
scheme of ``opencrab2.lowlevel.toy_bfv`` (status TEST_ONLY). Roles: P0 holds
the evaluated ciphertext and the plaintext weights W; P1 holds the toy secret
key s. Nothing but the three protocol messages below is communicated.

* ``a2h_toy(party, [x]_A64, bfv, pk)``: both parties reduce their A64 share
  modulo t = 2^log_t locally (t divides 2^64, so the reduction commutes with
  the sharing); P1 encrypts its reduced share under its own key and sends
  (c0, c1) to P0 (one message, 2 * 8 * N bytes); P0 adds Delta * (x0 mod t)
  into c0. Result: an encryption of x mod t held by P0 only.
* ``eval_toy(bfv, ct, W)``: P0's plaintext-weight coefficient-packed matvec
  (``ToyBFV.matvec_pt``); the noise bound is checked by the toy module.
* ``h2a_ref(party, bfv, ct | None, sk | None, positions)``: secure internal
  decryption. The product c1 * s of P0's private polynomial with P1's private
  polynomial is a bilinear map, so [c1] = (c1, 0) and [s] = (0, s) are valid
  additive sharings and the negacyclic product is computed with ONE Beaver
  round over the Karatsuba leaf products (3^log2(N) elementwise products,
  2 * 8 * 3^log2(N) bytes per party; 19683 products at N = 512 against N^2 =
  262144 for the schoolbook outer product). P0 adds c0 and Delta/2 locally.
  The faithful truncation gate (``Party.trunc_faithful`` with f = 64 - log_t
  and L = 64) then yields shares of floor((c0 + c1 s + Delta/2) / 2^f) in the
  signed semantics, which equals the plaintext coefficient modulo t
  (2^(64-f) = t). The coefficients at ``positions`` are the A64 shares of y.
  Rounds: 1 (Beaver open) + 1 (truncation open) = 2. The keyholder P1 never
  sees ct, c1 s or the noise; P0 never sees s.

What is tested (tests/test_bridge.py): shared Karatsuba product against
``toy_bfv.negacyclic_mul`` on random polynomials; end-to-end share ->
a2h_toy -> eval_toy(W) -> h2a_ref -> reconstruct == W x mod t at (N = 128,
d_in = d_out = 8) and (N = 512, d_in = 64, d_out = 8), 200 trials each, with
the noise bound respected; the truncation material must carry L = 64. Design
limits: the gate with L = 64 interprets the pre-rounding value as a signed
64-bit integer, so the reconstructed shares equal the plaintext modulo t
(signed representative), not necessarily the value in [0, t); the test
reduces modulo t. Real-parameter (SEAL) H2A_ref is NOT executed; any cost
estimate for it is derived from these toy measurements and must be red in
the paper. Error codes: BRIDGE_PARTY, BRIDGE_SHAPE, BRIDGE_PARAMS.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from ..errors import BridgeError
from ..lowlevel import ring
from ..lowlevel.toy_bfv import ToyBFV

BRIDGE_L = 64   # logical width handed to the faithful gate (full ring; see docstring)


def _u(a: Any) -> np.ndarray:
    return np.asarray(a, dtype=np.uint64)


def karatsuba_leaf_operands(sa: np.ndarray, sb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split two shared length-N polynomials into the 3^log2(N) Karatsuba leaf factors.

    Level by level, every polynomial of length n becomes three polynomials of
    length n/2: [lo, hi, lo + hi]; rows are ordered (all lo, all hi, all mid)
    so that ``karatsuba_recombine`` can undo the split.
    """
    A = _u(sa).reshape(1, -1)
    B = _u(sb).reshape(1, -1)
    N = A.shape[1]
    if N == 0 or N & (N - 1):
        raise BridgeError(f"BRIDGE_SHAPE: polynomial length must be a power of two, got {N}")
    if B.shape != A.shape:
        raise BridgeError(f"BRIDGE_SHAPE: operand lengths differ ({A.shape[1]} vs {B.shape[1]})")
    n = N
    while n > 1:
        h = n // 2
        A = np.concatenate([A[:, :h], A[:, h:], ring.add(A[:, :h], A[:, h:])], axis=0)
        B = np.concatenate([B[:, :h], B[:, h:], ring.add(B[:, :h], B[:, h:])], axis=0)
        n = h
    return A[:, 0], B[:, 0]


def karatsuba_recombine(leaf_products: np.ndarray, N: int) -> np.ndarray:
    """Recombine the 3^log2(N) leaf products into the negacyclic product mod (X^N + 1), mod 2^64."""
    P = _u(leaf_products).reshape(-1, 1)
    P = np.concatenate([P, np.zeros_like(P)], axis=1)          # length-1 products stored as length 2
    while P.shape[0] > 1:
        Bn = P.shape[0] // 3
        Lc = P.shape[1]
        lo, hi, mid = P[:Bn], P[Bn:2 * Bn], P[2 * Bn:]
        res = np.zeros((Bn, 2 * Lc), np.uint64)
        res[:, :Lc] = ring.add(res[:, :Lc], lo)
        res[:, Lc:] = ring.add(res[:, Lc:], hi)
        h = Lc // 2
        res[:, h:h + Lc] = ring.add(res[:, h:h + Lc], ring.sub(ring.sub(mid, lo), hi))
        P = res
    full = P[0]
    if full.shape[0] != 2 * N:
        raise BridgeError(f"BRIDGE_SHAPE: recombined length {full.shape[0]} differs from 2N = {2 * N}")
    return ring.sub(full[:N], full[N:])


def negacyclic_mul_shared(party: Any, sa: np.ndarray, sb: np.ndarray) -> np.ndarray:
    """Shares of a * b mod (X^N + 1) from shares of a and b: one Beaver round (Karatsuba leaves)."""
    sa = _u(sa).reshape(-1)
    sb = _u(sb).reshape(-1)
    N = int(sa.shape[0])
    with party.track("negacyclic_mul_shared", (N,)):
        la, lb = karatsuba_leaf_operands(sa, sb)
        prod = party.beaver_mul(la, lb)
        return karatsuba_recombine(prod, N)


def a2h_toy(party: Any, share: Any, bfv: ToyBFV, pk: Any):
    """A64 share -> toy ciphertext of x mod t held by P0 (returns None on P1). One message P1 -> P0."""
    share = _u(share).reshape(-1)
    N = bfv.params.N
    if share.shape[0] > N:
        raise BridgeError(f"BRIDGE_SHAPE: share length {share.shape[0]} exceeds N = {N}")
    x_t = bfv.encode_vector(share)                      # local reduction mod t (t | 2^64)
    with party.track("a2h_toy", (int(share.shape[0]), N)):
        if party.party == 1:
            if pk is None:
                raise BridgeError("BRIDGE_PARTY: P1 needs its public key to encrypt")
            c0, c1 = bfv.encrypt(pk, x_t, party.rng)
            party.ch.send(np.stack([c0, c1]))
            party._count_round()
            return None
        ct = _u(party.ch.recv())
        party._count_round()
        if ct.shape != (2, N):
            raise BridgeError(f"BRIDGE_SHAPE: expected a (2, {N}) ciphertext frame, got {ct.shape}")
        return bfv.add_pt((ct[0].copy(), ct[1].copy()), x_t)


def eval_toy(bfv: ToyBFV, ct: Any, W: Any):
    """P0: plaintext-weight matvec on the toy ciphertext; returns (ct_y, positions)."""
    return bfv.matvec_pt(ct, W)


def h2a_ref(party: Any, bfv: ToyBFV, ct_or_None: Any, sk_or_None: Any, positions: Any) -> np.ndarray:
    """Secure internal decryption (see module docstring); returns A64 shares of y at ``positions``."""
    N = bfv.params.N
    shift = bfv.params.shift
    positions = np.asarray(positions, dtype=np.int64).reshape(-1)
    if positions.size and (positions.min() < 0 or positions.max() >= N):
        raise BridgeError(f"BRIDGE_SHAPE: positions must lie in [0, {N})")
    with party.track("h2a_ref", (N, int(positions.size))):
        if party.party == 0:
            if ct_or_None is None:
                raise BridgeError("BRIDGE_PARTY: P0 must supply the evaluated ciphertext")
            if sk_or_None is not None:
                raise BridgeError("BRIDGE_PARTY: P0 must not hold the secret key")
            c0 = _u(ct_or_None[0]).reshape(-1)
            c1 = _u(ct_or_None[1]).reshape(-1)
            if c0.shape[0] != N or c1.shape[0] != N:
                raise BridgeError(f"BRIDGE_SHAPE: ciphertext components must have length N = {N}")
            sa, sb = c1, np.zeros(N, np.uint64)
        else:
            if ct_or_None is not None:
                raise BridgeError("BRIDGE_PARTY: the keyholder P1 must never receive the evaluated ciphertext")
            if sk_or_None is None:
                raise BridgeError("BRIDGE_PARTY: P1 must supply its secret key share")
            s = _u(sk_or_None).reshape(-1)
            if s.shape[0] != N:
                raise BridgeError(f"BRIDGE_SHAPE: secret key must have length N = {N}")
            sa, sb = np.zeros(N, np.uint64), s
        prod = negacyclic_mul_shared(party, sa, sb)                       # round 1
        if party.party == 0:
            prod = ring.add(prod, ring.add(c0, np.uint64(bfv.params.delta // 2)))
        z = party.trunc_faithful(prod, f=shift, L=BRIDGE_L)               # round 2
        return z[positions]


__all__ = ["a2h_toy", "eval_toy", "h2a_ref", "negacyclic_mul_shared", "karatsuba_leaf_operands",
           "karatsuba_recombine", "BRIDGE_L"]
