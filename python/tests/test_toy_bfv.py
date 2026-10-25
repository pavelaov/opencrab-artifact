"""Tests for opencrab2.lowlevel.toy_bfv.

Covers: negacyclic multiplication against a pure-integer reference, 1000
encrypt/decrypt round trips at every admitted N, homomorphic ct+ct, ct+pt and
ct*pt, coefficient-packed matvec at (N=128, d_in=d_out=8) and
(N=512, d_in=64, d_out=8), the noise bound refusing a violating parameter
set, the A2H_toy local-reduction pattern, the decrypt_arith_steps contract the
secure bridge reproduces, and parameter validation.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest


def _ensure_errors_module() -> None:
    """Thin test-only fallback: used when opencrab2.errors is absent."""
    try:
        import opencrab2.errors  # noqa: F401
    except ImportError:  # pragma: no cover
        mod = types.ModuleType("opencrab2.errors")

        class OpenCrabError(Exception):
            pass

        class BridgeError(OpenCrabError):
            pass

        mod.OpenCrabError = OpenCrabError
        mod.BridgeError = BridgeError
        sys.modules["opencrab2.errors"] = mod


_ensure_errors_module()

from opencrab2.errors import BridgeError  # noqa: E402
from opencrab2.lowlevel import toy_bfv as tb  # noqa: E402

LOG_T = 16


def _rng(seed: int = 2027) -> np.random.Generator:
    return np.random.default_rng(seed)


def _rand_pt(bfv: tb.ToyBFV, rng) -> np.ndarray:
    return rng.integers(0, bfv.params.t, bfv.params.N, dtype=np.uint64)


# ---------------------------------------------------------------------------
# ring arithmetic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N", [64, 128, 512])
def test_negacyclic_mul_matches_integer_reference(N):
    rng = _rng(N)
    for _ in range(3):
        a = rng.integers(0, 1 << 64, N, dtype=np.uint64)
        b = rng.integers(0, 1 << 64, N, dtype=np.uint64)
        assert np.array_equal(tb.negacyclic_mul(a, b), tb.negacyclic_mul_reference(a, b))


def test_negacyclic_wrap_sign():
    # X^(N-1) * X = X^N = -1 mod (X^N + 1): coefficient 0 becomes 2^64 - 1
    N = 64
    a = np.zeros(N, dtype=np.uint64)
    a[N - 1] = 1
    b = np.zeros(N, dtype=np.uint64)
    b[1] = 1
    out = tb.negacyclic_mul(a, b)
    assert int(out[0]) == (1 << 64) - 1
    assert int(out[1:].sum()) == 0


def test_negacyclic_rejects_bad_types():
    with pytest.raises(BridgeError, match="^TOY_BFV_TYPE"):
        tb.negacyclic_mul(np.zeros(64, dtype=np.int64), np.zeros(64, dtype=np.uint64))
    with pytest.raises(BridgeError, match="^TOY_BFV_SHAPE"):
        tb.negacyclic_mul(np.zeros(64, dtype=np.uint64), np.zeros(128, dtype=np.uint64))


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------

def test_params_properties_and_validation():
    p = tb.ToyParams(128, 16)
    assert p.t == 1 << 16
    assert p.delta == 1 << 48
    assert p.shift == 48
    assert p.B_e == 8
    with pytest.raises(BridgeError, match="^TOY_BFV_PARAM"):
        tb.ToyParams(100, 16)
    with pytest.raises(BridgeError, match="^TOY_BFV_PARAM"):
        tb.ToyParams(128, 33)
    with pytest.raises(BridgeError, match="^TOY_BFV_PARAM"):
        tb.ToyParams(128, 0)
    with pytest.raises(BridgeError, match="^TOY_BFV_PARAM"):
        tb.ToyParams(128, 16, B_e=0)


# ---------------------------------------------------------------------------
# encrypt / decrypt
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N", tb.ALLOWED_N)
def test_1000_round_trips(N):
    rng = _rng(N)
    bfv = tb.ToyBFV(tb.ToyParams(N, LOG_T))
    sk, pk = bfv.keygen(rng)
    assert sk.dtype == np.uint64 and sk.shape == (N,)
    assert set(np.unique(sk.view(np.int64)).tolist()) <= {-1, 0, 1}
    bound = bfv.fresh_noise_bound()
    for _ in range(1000):
        m = _rand_pt(bfv, rng)
        ct = bfv.encrypt(pk, m, rng)
        steps = bfv.decrypt_arith_steps(sk, ct)
        assert np.array_equal(steps["plaintext"], m)
        assert int(np.abs(steps["noise_signed"]).max()) <= bound
    assert np.array_equal(bfv.decrypt(sk, bfv.encrypt(pk, m, rng)), m)


def test_encrypt_rejects_out_of_range_plaintext():
    rng = _rng(5)
    bfv = tb.ToyBFV(tb.ToyParams(64, 8))
    sk, pk = bfv.keygen(rng)
    m = np.zeros(64, dtype=np.uint64)
    m[3] = 256
    with pytest.raises(BridgeError, match="^TOY_BFV_PLAINTEXT_RANGE"):
        bfv.encrypt(pk, m, rng)
    with pytest.raises(BridgeError, match="^TOY_BFV_SHAPE"):
        bfv.encrypt(pk, np.zeros(32, dtype=np.uint64), rng)


def test_decrypt_arith_steps_contract():
    """The bridge recomputes c1*s on shares, adds c0 and Delta/2, truncates by shift bits."""
    rng = _rng(7)
    bfv = tb.ToyBFV(tb.ToyParams(128, 12))
    sk, pk = bfv.keygen(rng)
    m = _rand_pt(bfv, rng)
    c0, c1 = bfv.encrypt(pk, m, rng)
    st = bfv.decrypt_arith_steps(sk, (c0, c1))
    assert st["shift"] == 64 - 12
    assert st["half_delta"] == bfv.params.delta // 2
    with np.errstate(over="ignore"):
        assert np.array_equal(st["phase"], c0 + tb.negacyclic_mul(c1, sk))
        assert np.array_equal(st["pre_round"], st["phase"] + np.uint64(st["half_delta"]))
    assert np.array_equal(st["plaintext"], st["pre_round"] >> np.uint64(st["shift"]))
    assert np.array_equal(st["plaintext"], m)
    # additive-share emulation of the arithmetic chain (not the secure protocol):
    # a random split of pre_round, recombined mod 2^64, truncates to the same plaintext.
    r = rng.integers(0, 1 << 64, 128, dtype=np.uint64)
    with np.errstate(over="ignore"):
        s0 = st["pre_round"] - r
        s1 = r
        recomb = s0 + s1
    assert np.array_equal(recomb >> np.uint64(st["shift"]), m)


# ---------------------------------------------------------------------------
# homomorphic operations
# ---------------------------------------------------------------------------

def test_add_ct_and_add_pt():
    rng = _rng(11)
    bfv = tb.ToyBFV(tb.ToyParams(128, LOG_T))
    sk, pk = bfv.keygen(rng)
    t = bfv.params.t
    for _ in range(50):
        m1, m2, m3 = _rand_pt(bfv, rng), _rand_pt(bfv, rng), _rand_pt(bfv, rng)
        ct = bfv.add_pt(bfv.add_ct(bfv.encrypt(pk, m1, rng), bfv.encrypt(pk, m2, rng)), m3)
        expect = ((m1.astype(object) + m2.astype(object) + m3.astype(object)) % t).astype(np.uint64)
        assert np.array_equal(bfv.decrypt(sk, ct), expect)


def test_mul_pt_matches_plaintext_negacyclic_product_mod_t():
    rng = _rng(13)
    bfv = tb.ToyBFV(tb.ToyParams(128, LOG_T))
    sk, pk = bfv.keygen(rng)
    t = bfv.params.t
    for _ in range(20):
        m = _rand_pt(bfv, rng)
        p_signed = rng.integers(-3, 4, 128, dtype=np.int64)        # small plaintext polynomial
        p = p_signed.view(np.uint64)
        ct = bfv.mul_pt(bfv.encrypt(pk, m, rng), p)
        # reference: negacyclic product over the integers, reduced mod t
        ref = tb.negacyclic_mul_reference(m, p)                     # mod 2^64, then mod t (t | 2^64)
        ref_t = (ref & np.uint64(t - 1))
        assert np.array_equal(bfv.decrypt(sk, ct), ref_t)
    assert bfv.ct_pt_products == 20


# ---------------------------------------------------------------------------
# packed matvec
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N,d_in,d_out", [(128, 8, 8), (512, 64, 8)])
def test_matvec_equals_W_x_mod_t(N, d_in, d_out):
    rng = _rng(N + d_in)
    bfv = tb.ToyBFV(tb.ToyParams(N, LOG_T))
    sk, pk = bfv.keygen(rng)
    t = bfv.params.t
    for trial in range(10):
        W = rng.integers(0, t, (d_out, d_in))
        x = rng.integers(0, t, d_in)
        ct_x = bfv.encrypt(pk, bfv.encode_vector(x), rng)
        ct_y, positions = bfv.matvec_pt(ct_x, W)
        assert positions.shape == (d_out,)
        assert positions.tolist() == [(k + 1) * d_in - 1 for k in range(d_out)]
        y = bfv.decrypt(sk, ct_y)[positions]
        assert np.array_equal(y, bfv.matvec_reference(W, x))
    assert bfv.ct_pt_products == 10            # one ct-pt product per matvec at these shapes


def test_matvec_with_signed_weights_and_1000_trials_small():
    """1000 random (W, x) at N=64, d_in=4, d_out=4 with signed W entries (noise stays within bound)."""
    rng = _rng(99)
    bfv = tb.ToyBFV(tb.ToyParams(64, 12))
    sk, pk = bfv.keygen(rng)
    t = bfv.params.t
    for _ in range(1000):
        W = rng.integers(-t // 2, t // 2, (4, 4))
        x = rng.integers(0, t, 4)
        ct_y, pos = bfv.matvec_pt(bfv.encrypt(pk, bfv.encode_vector(x), rng), W)
        assert np.array_equal(bfv.decrypt(sk, ct_y)[pos], bfv.matvec_reference(W, x))


def test_matvec_blocks_for_large_d_out():
    rng = _rng(21)
    bfv = tb.ToyBFV(tb.ToyParams(128, LOG_T))
    sk, pk = bfv.keygen(rng)
    t = bfv.params.t
    d_in, d_out = 32, 10          # rows_per_ct = 4, so 3 blocks
    W = rng.integers(0, t, (d_out, d_in))
    x = rng.integers(0, t, d_in)
    ct_x = bfv.encrypt(pk, bfv.encode_vector(x), rng)
    with pytest.raises(BridgeError, match="^TOY_BFV_SHAPE"):
        bfv.matvec_pt(ct_x, W)
    blocks = bfv.matvec_pt_blocks(ct_x, W)
    assert len(blocks) == 3
    y = np.zeros(d_out, dtype=np.uint64)
    for ct_y, pos, lo, hi in blocks:
        y[lo:hi] = bfv.decrypt(sk, ct_y)[pos]
    assert np.array_equal(y, bfv.matvec_reference(W, x))
    assert bfv.ct_pt_products == 3


def test_noise_budget_rejects_violating_parameters():
    bad = tb.ToyBFV(tb.ToyParams(512, 32))
    assert bad.noise_budget_ok(64, 1 << 31) is False
    assert bad.noise_bound(64, 1 << 31) >= bad.params.delta // 2
    good = tb.ToyBFV(tb.ToyParams(512, LOG_T))
    assert good.noise_budget_ok(64, 1 << 15) is True
    assert good.noise_bound(64, 1 << 15) < good.params.delta // 2
    rng = _rng(3)
    sk, pk = bad.keygen(rng)
    W = np.full((8, 64), (1 << 31) - 1, dtype=np.int64)
    x = np.zeros(64, dtype=np.int64)
    ct_x = bad.encrypt(pk, bad.encode_vector(x), rng)
    with pytest.raises(BridgeError, match="^TOY_BFV_NOISE_BUDGET"):
        bad.matvec_pt(ct_x, W)
    assert bad.ct_pt_products == 0            # refused before any product


def test_encode_vector_reduces_mod_t_and_checks_length():
    bfv = tb.ToyBFV(tb.ToyParams(64, 8))
    poly = bfv.encode_vector(np.array([-1, 256, 5], dtype=np.int64))
    assert poly.dtype == np.uint64 and poly.shape == (64,)
    assert poly[:3].tolist() == [255, 0, 5]
    assert int(poly[3:].sum()) == 0
    with pytest.raises(BridgeError, match="^TOY_BFV_SHAPE"):
        bfv.encode_vector(np.zeros(65, dtype=np.int64))


# ---------------------------------------------------------------------------
# A2H_toy local pattern
# ---------------------------------------------------------------------------

def test_a2h_toy_local_reduction_pattern():
    """P1 encrypts its share mod t, P0 adds Delta * (own share mod t): decrypts to x mod t."""
    rng = _rng(31)
    bfv = tb.ToyBFV(tb.ToyParams(128, LOG_T))
    sk, pk = bfv.keygen(rng)
    t = bfv.params.t
    d = 16
    for _ in range(50):
        x = rng.integers(0, 1 << 64, d, dtype=np.uint64)                    # A64 value
        x0 = rng.integers(0, 1 << 64, d, dtype=np.uint64)
        with np.errstate(over="ignore"):
            x1 = x - x0                                                      # shares: x0 + x1 = x mod 2^64
        ct = tb.a2h_toy_local(bfv, pk, x1, x0, rng)
        dec = bfv.decrypt(sk, ct)
        assert np.array_equal(dec[:d], x & np.uint64(t - 1))
        assert int(dec[d:].sum()) == 0
