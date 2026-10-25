"""HE lane tests: exact BFV A2H (TenSEAL) and the UNCERTIFIED_DISABLED CKKS guard.

TenSEAL is a pinned requirement (requirements.txt), imported directly, so a
missing or broken installation fails here. Every protocol run uses the
in-process harness of tests/test_party.py; the ciphertext produced on P0 is
decrypted in the TEST by P1's context (the protocol itself never decrypts).
"""

from __future__ import annotations

import numpy as np
import pytest
import tenseal as ts

from opencrab2.errors import AdmissionError, ContractViolation
from opencrab2.highlevel import he_lane
from opencrab2.highlevel.he_lane import (BFV_PLAIN_MODULUS, BFVExactA2H, CKKSFastH2A, a2h_exact,
                                         naive_sharewise_reduction, signed_mod_t)
from opencrab2.lowlevel import ring
from tests.test_party import CONTRACT, TwoPartyHarness, dec, enc

T = BFV_PLAIN_MODULUS


def test_status_constants():
    assert he_lane.STATUS["BFVExactA2H"] == "ADMITTED"
    assert he_lane.STATUS["bfv_layer_lane"] == "UNAVAILABLE"
    assert he_lane.STATUS["CKKSFastH2A"] == "UNCERTIFIED_DISABLED"
    assert CKKSFastH2A.STATUS == "UNCERTIFIED_DISABLED"


def test_bfv_primitive_roundtrip_and_party_checks():
    client = BFVExactA2H(1)
    server = BFVExactA2H(0, public_context_bytes=client.public_bytes())
    rng = np.random.default_rng(0)
    z1 = rng.integers(0, T, size=64, dtype=np.uint64)
    z0 = rng.integers(0, T, size=64, dtype=np.uint64)
    ct = server.add_share(None, client.encrypt_share(z1), z0)
    got = client.decrypt(ct)
    assert np.array_equal(got, (z0 + z1) % np.uint64(T))
    with pytest.raises(ContractViolation):
        server.encrypt_share(z1)             # P0 has no secret key
    with pytest.raises(ContractViolation):
        client.add_share(None, ct, z0)       # keyholder never adds the server share
    with pytest.raises(ContractViolation):
        server.decrypt(ct)
    with pytest.raises(ContractViolation):
        client.encrypt_share(np.array([T], np.uint64))   # out of Z_t range
    with pytest.raises(ContractViolation):
        client.encrypt_share(np.zeros(8193, np.uint64))  # more than N slots
    with pytest.raises(ContractViolation):
        BFVExactA2H(1, plain_modulus=65521)             # prime but not 1 mod 2N at N = 8192
    with pytest.raises(ContractViolation):
        BFVExactA2H(1, plain_modulus=65536)             # not prime
    with pytest.raises(ContractViolation):
        BFVExactA2H(0, public_context_bytes=client.ctx.serialize(save_secret_key=True))


def _run_a2h(h: TwoPartyHarness, s0: np.ndarray, s1: np.ndarray):
    ct0, ct1 = h.run(lambda p: a2h_exact(p, s0, T), lambda p: a2h_exact(p, s1, T))
    assert ct1 is None                       # P1 never receives the ciphertext
    he1 = h.parties[1]._he[f"bfv:{T}:8192"]
    cts = ct0 if isinstance(ct0, list) else [ct0]
    return np.concatenate([he1.decrypt(c) for c in cts])


def test_a2h_exact_random_negative_and_boundary_values():
    h = TwoPartyHarness(seed=50)
    try:
        rng = np.random.default_rng(1)
        vals = np.concatenate([
            rng.integers(-(2 ** 39), 2 ** 39, size=200),
            np.array([0, 1, -1, 2 ** 39 - 1, -(2 ** 39), 2 ** 39 - 2, -(2 ** 39) + 1, 12345, -12345]),
            rng.integers(-(2 ** 20), 2 ** 20, size=50),
        ]).astype(np.int64)
        x = ring.from_signed(vals)
        s0 = ring.rand_u64((x.size,), rng)
        s1 = ring.sub(x, s0)
        got = _run_a2h(h, s0, s1)
        want = signed_mod_t(x, T)
        assert np.array_equal(got, want)
        # the naive sharewise reduction (the rejected pattern) disagrees on wrapped and negative shares
        naive = naive_sharewise_reduction(s0, s1, T)
        assert np.any(naive != want)
        st = [s for s in h.parties[0].stats if s["kind"] == "a2h_exact"][0]
        # first call: ring_to_field open + one-time public-context setup + ciphertext message
        assert st["rounds"] == 3 and st["bytes"] > 0 and st["material_bytes"] > 0
        assert any(s["kind"] == "a2h_setup" for s in h.parties[0].all_stats)
        got2 = _run_a2h(h, s0[:5], s1[:5])
        assert np.array_equal(got2, want[:5])
        st2 = [s for s in h.parties[0].stats if s["kind"] == "a2h_exact"][-1]
        assert st2["rounds"] == 2          # steady state: 1 (ring_to_field) + 1 (ciphertext P1 -> P0)
    finally:
        h.close()


def test_a2h_exact_audit_counterexample_pattern():
    """q = 2^64 analogue of (q=256, t=17, a0=255, a1=2): u = 1 so z must be 1; naive gives something else."""
    h = TwoPartyHarness(seed=51)
    try:
        s0 = np.array([2 ** 64 - 1, 2 ** 63, 2 ** 63 + 5], np.uint64)
        s1 = np.array([2, 2 ** 63, 2 ** 63 - 7], np.uint64)   # u = 1, 0, -2
        got = _run_a2h(h, s0, s1)
        assert got.tolist() == [1, 0, (-2) % T]
        naive = naive_sharewise_reduction(s0, s1, T)
        assert naive.tolist() != got.tolist()
    finally:
        h.close()


def test_a2h_exact_chunks_over_several_ciphertexts():
    h = TwoPartyHarness(seed=52)
    try:
        rng = np.random.default_rng(2)
        M = 8192 + 300
        vals = rng.integers(-(2 ** 30), 2 ** 30, size=M).astype(np.int64)
        x = ring.from_signed(vals)
        s0 = ring.rand_u64((M,), rng)
        s1 = ring.sub(x, s0)
        ct0, ct1 = h.run(lambda p: a2h_exact(p, s0, T), lambda p: a2h_exact(p, s1, T))
        assert isinstance(ct0, list) and len(ct0) == 2 and ct1 is None
        he1 = h.parties[1]._he[f"bfv:{T}:8192"]
        got = np.concatenate([he1.decrypt(c) for c in ct0])
        assert np.array_equal(got, signed_mod_t(x, T))
    finally:
        h.close()


def test_party_method_delegates():
    h = TwoPartyHarness(seed=53)
    try:
        x = enc(np.array([1.5, -2.0]))
        s0 = np.array([7, 11], np.uint64)
        s1 = ring.sub(x, s0)
        ct0, ct1 = h.run(lambda p: p.a2h_exact(s0, T), lambda p: p.a2h_exact(s1, T))
        assert ct1 is None and isinstance(ct0, (bytes, bytearray))
        he1 = h.parties[1]._he[f"bfv:{T}:8192"]
        assert np.array_equal(he1.decrypt(ct0), signed_mod_t(x, T))
    finally:
        h.close()


# ---------------------------------------------------------------------------
# CKKS fast H2A: disabled unless diagnostic
# ---------------------------------------------------------------------------
def test_ckks_guard_refuses_without_diagnostic():
    with pytest.raises(AdmissionError) as ei:
        CKKSFastH2A(1)
    assert str(ei.value).startswith("NOT_ADMITTED_COMPONENT:")
    with pytest.raises(AdmissionError):
        CKKSFastH2A(0, diagnostic=False)


def test_ckks_diagnostic_path_reproduces_linear_layer():
    client = CKKSFastH2A(1, diagnostic=True)
    server = CKKSFastH2A(0, diagnostic=True, public_context_bytes=client.public_bytes())
    assert client.evidence_status == "DIAGNOSTIC" and server.evidence_status == "DIAGNOSTIC"
    rng = np.random.default_rng(3)
    W = rng.uniform(-1, 1, size=(16, 32))
    b = rng.uniform(-1, 1, size=16)
    x = rng.uniform(-2, 2, size=32)
    ct = client.encrypt(x)
    ct_y = server.linear(ct, W, b)
    # with a small mask the CKKS decode is accurate; this is a functionality check only
    masked, server_share, cert = server.h2a_server(ct_y, 16, rng, CONTRACT, flooding_bits=0)
    client_share = client.h2a_client(masked, CONTRACT)
    y = dec(ring.add(server_share, client_share))
    assert np.max(np.abs(y - (W @ x + b))) < 1e-2
    assert cert.status == "UNCERTIFIED_DISABLED" and cert.direction == "H2A"
    assert server.counters.ct_pl_mul == 1
    # with 20 flooding bits the masked value has about 60 bits and a double cannot carry its
    # low bits, so the reconstruction degrades; this documents the defect, it is not a pass criterion
    masked2, share2, _ = server.h2a_server(ct_y, 16, rng, CONTRACT, flooding_bits=20)
    y2 = dec(ring.add(share2, client.h2a_client(masked2, CONTRACT)))
    err_flooded = float(np.max(np.abs(y2 - (W @ x + b))))
    assert np.isfinite(err_flooded)
