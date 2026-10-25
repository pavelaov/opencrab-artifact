"""Tests for opencrab2.highlevel.approx (polynomial approximation contracts)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import erf

from opencrab2.highlevel import approx


def _gelu(x):
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))


def test_named_contracts_exist_with_declared_shapes():
    gelu = approx.contract("gelu_clip12")
    assert (gelu.lo, gelu.hi, gelu.degree, gelu.scale) == (-5.0, 5.0, 12, 5.0)
    assert len(gelu.coeffs) == 13 and not gelu.even_in_u and not gelu.odd_in_u and not gelu.affine
    silu = approx.contract("silu_clip12")
    assert (silu.lo, silu.hi, silu.degree, silu.scale) == (-8.0, 8.0, 12, 8.0)
    tanh = approx.contract("tanh_clip")
    assert (tanh.lo, tanh.hi, tanh.degree, tanh.scale) == (-4.0, 4.0, 9, 4.0)
    assert tanh.odd_in_u and len(tanh.coeffs) == 10
    assert all(tanh.coeffs[i] == 0.0 for i in range(0, 10, 2))
    recip = approx.contract("recip_norm")
    assert (recip.lo, recip.hi, recip.degree) == (0.5, 1.0, 5) and recip.affine
    rsqrt = approx.contract("rsqrt_norm")
    assert (rsqrt.lo, rsqrt.hi, rsqrt.degree) == (0.25, 1.0, 5) and rsqrt.affine
    assert set(approx.CONTRACT_NAMES) == {"gelu_clip12", "silu_clip12", "tanh_clip", "recip_norm", "rsqrt_norm"}


def test_gelu_clip12_max_abs_err_below_5e_3():
    gelu = approx.contract("gelu_clip12")
    assert gelu.max_abs_err < 5e-3
    assert gelu.max_abs_err == pytest.approx(4.84e-3, rel=0.02)   # the value the numeric contract records
    xg = np.linspace(-5.0, 5.0, 100_001)
    err = np.max(np.abs(gelu.evaluate(xg) - _gelu(xg)))
    assert err <= gelu.max_abs_err * 1.01 + 1e-12


@pytest.mark.parametrize("name,f,bound", [
    ("silu_clip12", lambda x: x / (1.0 + np.exp(-x)), 1.5e-2),
    ("tanh_clip", np.tanh, 3e-2),
    ("recip_norm", lambda v: 1.0 / v, 2e-4),
    ("rsqrt_norm", lambda v: 1.0 / np.sqrt(v), 2e-3),
])
def test_contracts_reproduce_their_functions(name, f, bound):
    pc = approx.contract(name)
    assert pc.max_abs_err < bound
    xg = np.linspace(pc.lo, pc.hi, 50_001)
    err = float(np.max(np.abs(pc.evaluate(xg) - f(xg))))
    assert err <= pc.max_abs_err * 1.01 + 1e-12


def test_odd_fit_is_antisymmetric():
    tanh = approx.contract("tanh_clip")
    x = np.linspace(0.0, 4.0, 1001)
    assert np.allclose(tanh.evaluate(-x), -tanh.evaluate(x), atol=1e-15)


def test_deterministic_rebuild():
    a = approx.fit("g", _gelu, -5.0, 5.0, 12)
    b = approx.fit("g", _gelu, -5.0, 5.0, 12)
    assert a.coeffs == b.coeffs and a.max_abs_err == b.max_abs_err
    assert a.coeffs == approx.contract("gelu_clip12").coeffs
    table = approx.contract_table_json()
    assert set(table) == set(approx.CONTRACT_NAMES)
    assert table["gelu_clip12"]["max_abs_err"] == a.max_abs_err


def test_even_fit_in_u_squared():
    pc = approx.fit("cosh", np.cosh, -1.0, 1.0, 6, even=True)
    assert pc.even_in_u and len(pc.coeffs) == 4
    assert pc.max_abs_err < 1e-6
    assert np.allclose(pc.evaluate(np.array([0.3])), np.cosh(0.3), atol=1e-5)


def test_horner_matches_polyval():
    coeffs = [1.0, -2.0, 0.5, 3.0]
    u = np.linspace(-1, 1, 11)
    assert np.allclose(approx.horner(coeffs, u), np.polyval(coeffs[::-1], u))


def test_fit_parameter_validation_and_unknown_name():
    with pytest.raises(ValueError, match="^APPROX_UNKNOWN_CONTRACT"):
        approx.contract("gelu_new")
    with pytest.raises(ValueError, match="^APPROX_PARAM"):
        approx.fit("x", np.tanh, -4.0, 4.0, 8, odd=True)
    with pytest.raises(ValueError, match="^APPROX_PARAM"):
        approx.fit("x", np.tanh, -4.0, 4.0, 8, even=True, odd=True)
    with pytest.raises(ValueError, match="^APPROX_PARAM"):
        approx.fit("x", np.tanh, -3.0, 5.0, 8)           # asymmetric range needs affine=True
    with pytest.raises(ValueError, match="^APPROX_PARAM"):
        approx.fit("x", np.tanh, 1.0, 1.0, 8)
