"""Secure toy-BFV bridge tests: shared Karatsuba product, A2H_toy, Eval_toy, H2A_ref end to end.

Uses the in-process harness of tests/test_party.py. The truncation material
for H2A_ref is generated for (f = 64 - log_t, L = 64) by the harness dealer on
demand. 1000 random trials in total (800 at N = 128, 200 at N = 512), noise bound
respected.
"""

from __future__ import annotations

import numpy as np
import pytest

from opencrab2.errors import BridgeError
from opencrab2.highlevel.bridge import (BRIDGE_L, a2h_toy, eval_toy, h2a_ref, karatsuba_leaf_operands,
                                        karatsuba_recombine, negacyclic_mul_shared)
from opencrab2.highlevel.party import Party
from opencrab2.lowlevel import ring, sharing
from opencrab2.lowlevel.toy_bfv import ToyBFV, ToyParams, negacyclic_mul
from tests.test_party import TwoPartyHarness


@pytest.mark.parametrize("N", [8, 64, 128, 512])
def test_karatsuba_plaintext_identity(N):
    rng = np.random.default_rng(N)
    a = ring.rand_u64((N,), rng)
    b = ring.rand_u64((N,), rng)
    la, lb = karatsuba_leaf_operands(a, b)
    assert la.shape == (3 ** int(np.log2(N)),)
    prod = ring.mul(la, lb)
    got = karatsuba_recombine(prod, N)
    assert np.array_equal(got, negacyclic_mul(a, b))


def test_karatsuba_rejects_bad_lengths():
    with pytest.raises(BridgeError):
        karatsuba_leaf_operands(np.zeros(6, np.uint64), np.zeros(6, np.uint64))
    with pytest.raises(BridgeError):
        karatsuba_leaf_operands(np.zeros(8, np.uint64), np.zeros(4, np.uint64))


def test_negacyclic_mul_shared_one_round():
    h = TwoPartyHarness(seed=60)
    try:
        N = 128
        rng = np.random.default_rng(1)
        a = ring.rand_u64((N,), rng)
        b = ring.rand_u64((N,), rng)
        a0, a1 = sharing.share(a, rng)
        b0, b1 = sharing.share(b, rng)
        o0, o1 = h.run(lambda p: negacyclic_mul_shared(p, a0, b0), lambda p: negacyclic_mul_shared(p, a1, b1))
        assert np.array_equal(ring.add(o0, o1), negacyclic_mul(a, b))
        st = [s for s in h.parties[0].stats if s["kind"] == "negacyclic_mul_shared"][0]
        assert st["rounds"] == 1
        leaves = 3 ** 7
        # one open of 2 * leaves words per party (frame overhead excluded)
        assert st["bytes"] >= 2 * 8 * 2 * leaves and st["bytes"] < 2 * 8 * 2 * leaves + 200
    finally:
        h.close()


def _trial(h: TwoPartyHarness, bfv: ToyBFV, rng: np.random.Generator, d_in: int, d_out: int, w_max: int, pk, sk):
    t = bfv.params.t
    x = rng.integers(0, t, size=d_in, dtype=np.uint64)
    W = rng.integers(-w_max, w_max + 1, size=(d_out, d_in)).astype(np.int64)
    x0 = ring.rand_u64((d_in,), rng)
    x1 = ring.sub(x, x0)

    def p0(p: Party):
        ct = a2h_toy(p, x0, bfv, None)
        ct_y, positions = eval_toy(bfv, ct, W)
        return h2a_ref(p, bfv, ct_y, None, positions), positions

    def p1(p: Party):
        a2h_toy(p, x1, bfv, pk)
        positions = np.arange(1, d_out + 1, dtype=np.int64) * d_in - 1    # public layout
        return h2a_ref(p, bfv, None, sk, positions), positions

    (y0, pos0), (y1, pos1) = h.run(p0, p1)
    assert np.array_equal(pos0, pos1)
    y = ring.add(y0, y1)
    got = (ring.to_signed(y) % np.int64(t)).astype(np.uint64)
    want = bfv.matvec_reference(W, x)
    return got, want


@pytest.mark.parametrize("N,d_in,d_out,log_t,w_max,trials", [(128, 8, 8, 16, 100, 800), (512, 64, 8, 16, 20, 200)])
def test_bridge_end_to_end(N, d_in, d_out, log_t, w_max, trials):
    params = ToyParams(N=N, log_t=log_t)
    bfv = ToyBFV(params)
    assert bfv.noise_budget_ok(d_in, w_max)
    rng = np.random.default_rng(1000 + N)
    sk, pk = bfv.keygen(rng)
    h = TwoPartyHarness(seed=70 + N)
    try:
        for _ in range(trials):
            got, want = _trial(h, bfv, rng, d_in, d_out, w_max, pk, sk)
            assert np.array_equal(got, want)
        st0 = h.parties[0].stats
        h2a = [s for s in st0 if s["kind"] == "h2a_ref"]
        assert h2a and all(s["rounds"] == 2 for s in h2a)
        a2h = [s for s in st0 if s["kind"] == "a2h_toy"]
        assert a2h and all(s["rounds"] == 1 for s in a2h)
        # the faithful gate ran with L = 64 and f = 64 - log_t (the dealer refuses other (f, L) pairs)
        assert BRIDGE_L == 64
    finally:
        h.close()


def test_bridge_party_role_checks():
    params = ToyParams(N=64, log_t=8)
    bfv = ToyBFV(params)
    rng = np.random.default_rng(5)
    sk, pk = bfv.keygen(rng)
    h = TwoPartyHarness(seed=80)
    try:
        ct = bfv.encrypt(pk, bfv.encode_vector(np.zeros(4, np.uint64)), rng)
        pos = np.arange(1, 3) * 2 - 1

        def fn(p: Party):
            if p.party == 0:
                with pytest.raises(BridgeError):
                    h2a_ref(p, bfv, None, None, pos)           # P0 without the ciphertext
                with pytest.raises(BridgeError):
                    h2a_ref(p, bfv, ct, sk, pos)               # P0 must not hold the key
            else:
                with pytest.raises(BridgeError):
                    h2a_ref(p, bfv, ct, sk, pos)               # keyholder must never see the ciphertext
                with pytest.raises(BridgeError):
                    h2a_ref(p, bfv, None, None, pos)           # P1 without its key
            return True

        assert h.run_same(fn) == (True, True)
    finally:
        h.close()


def test_party_toy_bridge_method():
    params = ToyParams(N=128, log_t=16)
    bfv = ToyBFV(params)
    rng = np.random.default_rng(6)
    sk, pk = bfv.keygen(rng)
    W = rng.integers(-50, 51, size=(4, 8)).astype(np.int64)
    x = rng.integers(0, params.t, size=8, dtype=np.uint64)
    x0 = ring.rand_u64((8,), rng)
    x1 = ring.sub(x, x0)
    h = TwoPartyHarness(seed=81)
    try:
        y0, y1 = h.run(lambda p: p.toy_bridge(x0, bfv, pk, None, W), lambda p: p.toy_bridge(x1, bfv, pk, sk, W))
        got = (ring.to_signed(ring.add(y0, y1)) % np.int64(params.t)).astype(np.uint64)
        assert np.array_equal(got, bfv.matvec_reference(W, x))
    finally:
        h.close()
