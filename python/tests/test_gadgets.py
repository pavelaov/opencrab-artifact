"""Gadget tests: every nonlinear gadget against a float reference, k-ary tournaments exact across k.

Tolerances: gelu 5e-3 abs, softmax 1e-3 abs,
layernorm 2e-3 abs, exp 4e-3 relative on [-5, 0]. Max / argmax are exact
integer computations and must be bit-identical for k in {2, 4, 8} and equal
to numpy (lowest index on ties). All runs use the in-process harness of
tests/test_party.py (INSECURE_TEST_ONLY_SHARED_SEED, DIAGNOSTIC).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import erf

from opencrab2.errors import ContractViolation
from opencrab2.highlevel import approx
from opencrab2.highlevel.party import Party
from opencrab2.lowlevel import ring
from tests.test_party import CONTRACT, F, TwoPartyHarness, dec, enc, run_shared, shares_of


def _gelu(x):
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _rounds(h: TwoPartyHarness, kind: str) -> list:
    return [s["rounds"] for s in h.parties[0].stats if s["kind"] == kind]


# ---------------------------------------------------------------------------
# comparisons and selects
# ---------------------------------------------------------------------------
def test_drelu_relu_compare_select():
    h = TwoPartyHarness(seed=20)
    try:
        rng = np.random.default_rng(0)
        # edge values stay inside the DCF bound after the public shift by 2.5 (|x - 2.5| < 2^15)
        x = np.concatenate([rng.uniform(-3000, 3000, size=200), np.array([0.0, -1e-4, 1e-4, 32764.0, -32765.0])])
        xq = dec(enc(x))
        b = run_shared(h, enc(x), lambda p, s: p.drelu(s))
        assert np.array_equal(b, (xq >= 0).astype(np.uint64))
        r = run_shared(h, enc(x), lambda p, s: p.relu(s))
        assert np.array_equal(dec(r), np.maximum(xq, 0.0))
        ge = run_shared(h, enc(x), lambda p, s: p.compare_public(s, 2.5, "ge"))
        lt = run_shared(h, enc(x), lambda p, s: p.compare_public(s, 2.5, "lt"))
        assert np.array_equal(ge, (xq >= 2.5).astype(np.uint64))
        assert np.array_equal(lt, (xq < 2.5).astype(np.uint64))
        a = rng.uniform(-10, 10, size=x.shape)
        bs = shares_of(enc(a), 5)
        xs = shares_of(enc(x), 6)
        bits = shares_of((xq >= 0).astype(np.uint64), 7)

        def fn(p: Party):
            return p.select(bits[p.party], xs[p.party], bs[p.party])

        o0, o1 = h.run_same(fn)
        want = np.where(xq >= 0, xq, dec(enc(a)))
        assert np.array_equal(dec(ring.add(o0, o1)), want)
        assert _rounds(h, "drelu") and all(r == 1 for r in _rounds(h, "drelu"))
        assert all(r == 1 for r in _rounds(h, "relu")) and all(r == 1 for r in _rounds(h, "select"))
    finally:
        h.close()


# ---------------------------------------------------------------------------
# polynomial activations
# ---------------------------------------------------------------------------
def test_gelu_silu_tanh_within_contract():
    h = TwoPartyHarness(seed=21)
    try:
        rng = np.random.default_rng(1)
        x = np.concatenate([rng.uniform(-8, 8, size=600), np.linspace(-5.2, 5.2, 100), rng.uniform(-200, 200, size=50)])
        xq = dec(enc(x))
        g = dec(run_shared(h, enc(x), lambda p, s: p.gelu(s)))
        assert np.max(np.abs(g - _gelu(xq))) < 5e-3
        s_ = dec(run_shared(h, enc(x), lambda p, s: p.silu(s)))
        c = approx.contract("silu_clip12")
        assert np.max(np.abs(s_ - _silu(xq))) < c.max_abs_err + 1e-3
        xt = rng.uniform(-6, 6, size=300)
        t = dec(run_shared(h, enc(xt), lambda p, s: p.tanh(s)))
        ct = approx.contract("tanh_clip")
        assert np.max(np.abs(t - np.tanh(dec(enc(xt))))) < ct.max_abs_err + 2e-3
        deg = approx.contract("gelu_clip12").degree
        assert _rounds(h, "gelu_clip12") == [1 + deg + 1]
    finally:
        h.close()


def test_poly_horner_matches_float_horner():
    h = TwoPartyHarness(seed=22)
    try:
        coeffs = [0.5, -1.25, 0.75, 0.125, -0.0625]
        u = np.linspace(-1, 1, 257)
        out = dec(run_shared(h, enc(u), lambda p, s: p.poly_horner(s, coeffs)))
        assert np.max(np.abs(out - approx.horner(coeffs, dec(enc(u))))) < 8.0 / 2 ** F
    finally:
        h.close()


# ---------------------------------------------------------------------------
# exp / recip / rsqrt
# ---------------------------------------------------------------------------
def test_exp_neg_relative_error():
    h = TwoPartyHarness(seed=23)
    try:
        z = np.concatenate([np.linspace(-5, 0, 400), np.linspace(-16, -5, 100), np.array([-30.0, -100.0, -1000.0])])
        zq = dec(enc(z))
        out = dec(run_shared(h, enc(z), lambda p, s: p.exp_neg(s)))
        ref = np.exp(np.maximum(zq, -16.0))
        core = zq >= -5
        assert np.max(np.abs(out[core] - ref[core]) / ref[core]) < 4e-3
        assert np.max(np.abs(out - ref)) < 1e-3
        assert np.all(out >= -2.0 / 2 ** F)
        assert _rounds(h, "exp_neg") == [3 + CONTRACT.exp_r]
    finally:
        h.close()


def test_recip_and_rsqrt_ranges():
    h = TwoPartyHarness(seed=24)
    try:
        rng = np.random.default_rng(2)
        S = np.concatenate([np.array([1.0, 1.5, 2.0, 3.99, 4.0, 4095.0, 2047.5]), rng.uniform(1.0, 4096.0, size=200)])
        for e in (0, 10):
            out = run_shared(h, enc(S), lambda p, s: p.recip(s, extra_bits=e))
            got = ring.to_signed(out).astype(np.float64) / 2.0 ** (F + e)
            rel = np.abs(got - 1.0 / dec(enc(S))) * dec(enc(S))
            assert np.max(rel) < 2e-3 + 2.0 ** (F - F - e) * 4096 / 2 ** F, e
        assert _rounds(h, "recip") == [12, 12]
        V = np.concatenate([np.array([4.0 ** -7 + 1e-6, 0.001, 0.25, 1.0, 1.5, 100.0, 30000.0]),
                            np.exp(rng.uniform(np.log(1e-4), np.log(3e4), size=200))])
        for e in (0, 4):
            out = run_shared(h, enc(V), lambda p, s: p.rsqrt(s, extra_bits=e))
            got = ring.to_signed(out).astype(np.float64) / 2.0 ** (F + e)
            ref = 1.0 / np.sqrt(dec(enc(V)))
            rel = np.abs(got - ref) / ref
            assert np.max(rel) < 3e-3, (e, np.max(rel))
        assert _rounds(h, "rsqrt") == [14, 14]
    finally:
        h.close()


def test_rsqrt_normalisation_product_stays_inside_the_local_truncation_bound():
    # Paper, Appendix D, Disclosure ii. The selector s2 used to sit at scale 2^(2f): V * s2 was a 2^48
    # product and the local truncation returned a row off by 2^32 with probability about 2^-16.6 (42 of
    # 2^22 measured).
    # With e2 = rsqrt_scale_bits the product is below 2^34: zero gross errors on 2^22 elements
    # (expected count 2^-7 under the bound, and the run is deterministic).
    from opencrab2.highlevel.contract import SMOLLM_CONTRACT
    from opencrab2.highlevel.party import rsqrt_product_bits, rsqrt_scale_bits
    from opencrab2.lowlevel import sharing
    j0, m = int(CONTRACT.rsqrt_j0), int(CONTRACT.rsqrt_m)
    e2 = rsqrt_scale_bits(CONTRACT)
    assert e2 == 2 and rsqrt_product_bits(CONTRACT) == 34
    assert rsqrt_product_bits(SMOLLM_CONTRACT) <= 40 and rsqrt_product_bits(CONTRACT) <= 40
    # every selector value 4^-(j0+1+k) is an exact integer at scale 2^(f+e2)
    for k in range(0, m + 1):
        val = 4.0 ** (-(j0 + 1) - k) * float(1 << (F + e2))
        assert val == int(val) and val >= 1.0
    N = 1 << 22
    rng = np.random.default_rng(11)
    V = np.exp2(rng.uniform(2 * (j0 + 1), 14.0, size=N))
    V_enc = enc(V)
    k = np.clip(np.floor(np.log2(V) / 2.0).astype(np.int64) - j0, 0, m)
    s2_enc = ring.from_signed(np.rint(4.0 ** (-(j0 + 1) - k) * float(1 << (F + e2))).astype(np.int64))
    prod = ring.to_signed(ring.mul(V_enc, s2_enc))
    assert int(prod.min()) > 0 and int(prod.max()) <= (1 << 34)
    exact = np.floor_divide(prod, np.int64(1) << np.int64(F + e2))
    h = TwoPartyHarness(seed=3)
    try:
        h.auto[0].BEAVER_CHUNK = h.auto[1].BEAVER_CHUNK = 1 << 20
        xs = sharing.share(V_enc, np.random.default_rng(5))
        ss = sharing.share(s2_enc, np.random.default_rng(6))
        o0, o1 = h.run(lambda p: p.mul_fxp(xs[0], ss[0], F + e2), lambda p: p.mul_fxp(xs[1], ss[1], F + e2))
        got = ring.to_signed(ring.add(o0, o1))
        assert int((np.abs(got - exact) > 2).sum()) == 0
    finally:
        h.close()


# ---------------------------------------------------------------------------
# softmax / layernorm
# ---------------------------------------------------------------------------
def _softmax_ref(x, mask=None):
    x = np.array(x, np.float64)
    if mask is not None:
        x = np.where(mask, x, -np.inf)
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


@pytest.mark.parametrize("k_max", [2, 4])
def test_softmax_rows_with_and_without_mask(k_max):
    h = TwoPartyHarness(seed=25 + k_max)
    try:
        rng = np.random.default_rng(3)
        x = rng.uniform(-8, 8, size=(3, 5, 24))
        xq = dec(enc(x))
        out = dec(run_shared(h, enc(x), lambda p, s: p.softmax_rows(s, None, k_max=k_max)))
        assert np.max(np.abs(out - _softmax_ref(xq))) < 1e-3
        mask = np.tril(np.ones((5, 24), bool), k=19)[None]
        out_m = dec(run_shared(h, enc(x), lambda p, s: p.softmax_rows(s, mask, k_max=k_max)))
        ref_m = _softmax_ref(xq, np.broadcast_to(mask, xq.shape))
        assert np.max(np.abs(out_m - ref_m)) < 1e-3
        assert np.all(out_m[~np.broadcast_to(mask, xq.shape)] == 0.0)
        assert np.allclose(out.sum(axis=-1), 1.0, atol=5e-3)
    finally:
        h.close()


def test_layernorm_rows_with_affine_and_large_residual():
    h = TwoPartyHarness(seed=28)
    try:
        rng = np.random.default_rng(4)
        d = 64
        x = rng.normal(0, 3, size=(6, d))
        x[0, 3] = 500.0          # massive activation: residual square 2^50 ring units
        # small-variance row (var about 0.09 = 5900 ulps at f = 16); a variance of 1e-3 would carry a
        # 1.7 percent quantisation error before rsqrt, which is the documented f = 16 precision limit
        x[1, :] = x[1, :] * 0.1
        gamma = rng.uniform(0.5, 1.5, size=d)
        beta = rng.uniform(-0.5, 0.5, size=d)
        gs = shares_of(enc(gamma), 8)
        bs = shares_of(enc(beta), 9)
        xs = shares_of(enc(x), 10)

        def fn(p: Party):
            return p.layernorm_rows(xs[p.party], gs[p.party], bs[p.party], eps=1e-5)

        o0, o1 = h.run_same(fn)
        xq = dec(enc(x))
        mu = xq.mean(axis=-1, keepdims=True)
        var = ((xq - mu) ** 2).mean(axis=-1, keepdims=True)
        ref = (xq - mu) / np.sqrt(var + 1e-5) * dec(enc(gamma)) + dec(enc(beta))
        got = dec(ring.add(o0, o1))
        assert np.max(np.abs(got - ref)) < 2e-3
        kinds = [s["kind"] for s in h.parties[0].all_stats]
        assert "trunc_faithful" in kinds
        assert _rounds(h, "layernorm_rows") == [18]
        rms = dec(run_shared(h, enc(x[2:4]), lambda p, s: p.rmsnorm_rows(s, None, eps=1e-6)))
        rref = xq[2:4] / np.sqrt((xq[2:4] ** 2).mean(axis=-1, keepdims=True) + 1e-6)
        assert np.max(np.abs(rms - rref)) < 2e-3
    finally:
        h.close()


# ---------------------------------------------------------------------------
# k-ary tournaments
# ---------------------------------------------------------------------------
def _argmax_lowest(vals: np.ndarray, valid: np.ndarray) -> np.ndarray:
    masked = np.where(valid, vals, -np.inf)
    return np.argmax(masked, axis=-1)      # numpy returns the first (lowest) index on ties


def _rounds_expected(n: int, k: int) -> int:
    r = 0
    w = n
    while w > 1:
        w = -(-w // k)
        r += 1
    return 2 * r


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 9, 17, 50, 64, 65])
def test_max_and_argmax_exact_across_k(n):
    rng = np.random.default_rng(100 + n)
    R = 6
    # integers with many ties; magnitudes inside the k = 8 node bound (|diff| < 2^11 real)
    vals = rng.integers(-600, 600, size=(R, n)).astype(np.float64)
    if n >= 3:
        vals[0, :] = 7.0                 # all ties: lowest index must win
        vals[1, 1] = vals[1, 2] = vals[1].max() + 1.0
    valid_full = np.ones((R, n), bool)
    mask = rng.random((R, n)) < 0.7
    mask[:, 0] = True                     # every row keeps at least one valid entry
    outs = {}
    for k in (2, 4, 8):
        h = TwoPartyHarness(seed=300 + 10 * n + k)
        try:
            xs = shares_of(enc(vals), 42)

            def fn(p: Party):
                m = p.max_rows(xs[p.party], None, k=k)
                a = p.argmax_row(xs[p.party], k=k, release=False)
                mm = p.max_rows(xs[p.party], mask, k=k)
                am = p.argmax_row(xs[p.party], k=k, release=False, mask=mask)
                return m, a, mm, am

            (m0, a0, mm0, am0), (m1, a1, mm1, am1) = h.run_same(fn)
            m = dec(ring.add(m0, m1))
            a = ring.add(a0, a1)
            mm = dec(ring.add(mm0, mm1))
            am = ring.add(am0, am1)
            assert np.array_equal(m, vals.max(axis=-1)), k
            assert np.array_equal(a, _argmax_lowest(vals, valid_full).astype(np.uint64)), k
            assert np.array_equal(mm, np.where(mask, vals, -np.inf).max(axis=-1)), k
            assert np.array_equal(am, _argmax_lowest(vals, mask).astype(np.uint64)), k
            outs[k] = (m0 + m1, a0 + a1, mm0 + mm1, am0 + am1)
            if n > 1:
                assert _rounds(h, f"max_rows_k{k}")[0] == _rounds_expected(n, k), (n, k)
                assert _rounds(h, f"argmax_k{k}")[0] == _rounds_expected(n, k), (n, k)
            else:
                assert _rounds(h, f"max_rows_k{k}")[0] == 0
        finally:
            h.close()
    for k in (4, 8):
        for i in range(4):
            assert np.array_equal(outs[k][i], outs[2][i]), (k, i)


def test_argmax_release_goes_to_p1_only():
    h = TwoPartyHarness(seed=31, release=("token", 1))
    try:
        vals = np.array([1.0, 5.0, 5.0, -2.0, 3.0])
        xs = shares_of(enc(vals), 1)
        o0, o1 = h.run_same(lambda p: p.argmax_row(xs[p.party], k=4))
        assert o0 is None and int(o1) == 1
        assert h.parties[0].release.count == 1 and h.parties[1].release.count == 1
        assert h.parties[1].release_log[-1]["kind"] == "token"
    finally:
        h.close()


def test_tournament_rejects_bad_k_and_empty_rows():
    h = TwoPartyHarness(seed=32)
    try:
        def fn(p: Party):
            with pytest.raises(ContractViolation):
                p.argmax_row(np.zeros(4, np.uint64), k=3, release=False)
            with pytest.raises(ContractViolation):
                p.max_rows(np.zeros((1, 4), np.uint64), mask=np.zeros((1, 4), bool), k=2)
            with pytest.raises(ContractViolation):
                p.max_rows(np.zeros((1, 0), np.uint64), k=2)
            return True
        assert h.run_same(fn) == (True, True)
    finally:
        h.close()


def test_softmax_k4_equals_k2_bitwise_on_max_path():
    """The max shift is exact for every k, so the softmax outputs are bit-identical across k_max."""
    rng = np.random.default_rng(9)
    x = rng.uniform(-6, 6, size=(2, 3, 13))
    outs = []
    for k in (2, 4):
        h = TwoPartyHarness(seed=40 + k)
        try:
            outs.append(run_shared(h, enc(x), lambda p, s: p.softmax_rows(s, None, k_max=k)))
        finally:
            h.close()
    # the exp/recip chain is deterministic given identical max shares, up to the local
    # probabilistic truncation which depends on the (seeded) shares: the harness seeds
    # differ per k, so compare values rather than bits here
    assert np.max(np.abs(dec(outs[0]) - dec(outs[1]))) < 3.0 / 2 ** F
