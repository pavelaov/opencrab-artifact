"""Tests for opencrab2.lowlevel.ring and opencrab2.errors."""

from __future__ import annotations

import numpy as np
import pytest

from opencrab2 import errors
from opencrab2.lowlevel import ring

try:  # fall back to a stand-in when the contract module is absent
    from opencrab2.highlevel.contract import DEFAULT_CONTRACT as _CONTRACT
except Exception:  # pragma: no cover
    class _Contract:
        frac_bits = 16
        logical_bits = 40

    _CONTRACT = _Contract()


def test_add_sub_mul_neg_wrap():
    a = np.array([0xFFFFFFFFFFFFFFFF, 1, 2 ** 63], np.uint64)
    b = np.array([1, 0xFFFFFFFFFFFFFFFF, 2 ** 63], np.uint64)
    assert ring.add(a, b).tolist() == [0, 0, 0]
    assert ring.sub(a, b).tolist() == [2 ** 64 - 2, 2, 0]
    assert ring.mul(a, b).tolist() == [2 ** 64 - 1, 2 ** 64 - 1, 0]
    assert ring.neg(np.array([1, 0], np.uint64)).tolist() == [2 ** 64 - 1, 0]
    assert ring.mul(np.uint64(2 ** 32 + 1), np.uint64(2 ** 32 - 1)).tolist() == 2 ** 64 - 1


def test_to_signed_from_signed_msb():
    v = np.array([0, 1, 2 ** 64 - 1, 2 ** 63], np.uint64)
    assert ring.to_signed(v).tolist() == [0, 1, -1, -2 ** 63]
    assert ring.from_signed(np.array([-1, -2 ** 63, 5], np.int64)).tolist() == [2 ** 64 - 1, 2 ** 63, 5]
    assert ring.msb(v).tolist() == [0, 0, 1, 1]
    assert ring.msb(np.uint64(2 ** 63 - 1)) == 0


def test_arith_shift_right():
    v = ring.from_signed(np.array([-13, 13, -1, 2 ** 40 + 7], np.int64))
    assert ring.to_signed(ring.arith_shift_right(v, 3)).tolist() == [-2, 1, -1, 2 ** 37]
    with pytest.raises(errors.ContractViolation):
        ring.arith_shift_right(v, 64)


def test_encode_decode_roundtrip_and_saturation():
    x = np.array([0.0, 1.5, -2.25, 1234.5678, -0.00001])
    v = ring.encode(x, _CONTRACT)
    assert v.dtype == np.uint64
    back = ring.decode(v, _CONTRACT)
    assert np.max(np.abs(back - x)) <= 2.0 ** -17
    bound = 2.0 ** (_CONTRACT.logical_bits - 1 - _CONTRACT.frac_bits)
    ring.encode(np.array([bound - 2.0 ** -_CONTRACT.frac_bits]), _CONTRACT)
    with pytest.raises(errors.ContractViolation) as ei:
        ring.encode(np.array([bound]), _CONTRACT)
    assert str(ei.value).startswith("CONTRACT_VIOLATION:")
    with pytest.raises(errors.ContractViolation):
        ring.encode(np.array([-bound - 1.0]), _CONTRACT)
    with pytest.raises(errors.ContractViolation):
        ring.encode(np.array([np.nan]), _CONTRACT)


def test_rand_u64():
    rng = np.random.default_rng(0)
    v = ring.rand_u64((1000,), rng)
    assert v.dtype == np.uint64 and v.shape == (1000,)
    assert np.any(v >= np.uint64(2 ** 63)) and np.any(v < np.uint64(2 ** 63))
    assert ring.rand_u64((3, 4), rng).shape == (3, 4)


def _int_matmul_mod(A, B):
    m, k = A.shape
    n = B.shape[1]
    out = np.zeros((m, n), np.uint64)
    Al = A.tolist()
    Bl = B.tolist()
    for i in range(m):
        for j in range(n):
            s = 0
            for p in range(k):
                s += Al[i][p] * Bl[p][j]
            out[i, j] = s % (1 << 64)
    return out


def test_ring_matmul_small_matches_python_ints():
    rng = np.random.default_rng(1)
    A = ring.rand_u64((5, 7), rng)
    B = ring.rand_u64((7, 3), rng)
    assert np.array_equal(ring.ring_matmul(A, B), _int_matmul_mod(A, B))


def test_ring_matmul_limb_path_matches_integer_path():
    rng = np.random.default_rng(2)
    A = ring.rand_u64((40, 300), rng)
    B = ring.rand_u64((300, 30), rng)
    direct = (A @ B).astype(np.uint64)
    limbs = ring._ring_matmul_limbs(A, B)
    assert np.array_equal(direct, limbs)
    assert np.array_equal(ring.ring_matmul(A, B), direct)
    big_A = ring.rand_u64((64, 1100), rng)
    big_B = ring.rand_u64((1100, 40), rng)
    with np.errstate(over="ignore"):
        assert np.array_equal(ring.ring_matmul(big_A, big_B), (big_A @ big_B).astype(np.uint64))
    with pytest.raises(errors.ContractViolation):
        ring.ring_matmul(A, A)


def test_small_ring_helpers():
    assert ring.mask_k(8) == np.uint64(255) and ring.mask_k(64) == ring.MASK64
    assert ring.reduce_k(np.array([257, 2 ** 64 - 1], np.uint64), 8).tolist() == [1, 255]
    assert ring.to_signed_k(np.array([255, 127, 128], np.uint64), 8).tolist() == [-1, 127, -128]
    assert ring.from_signed_k(np.array([-1, -128, 5], np.int64), 8).tolist() == [255, 128, 5]
    assert ring.to_signed_k(np.array([2 ** 64 - 1], np.uint64), 64).tolist() == [-1]
    with pytest.raises(errors.ContractViolation):
        ring.check_k(65)
    with pytest.raises(errors.ContractViolation):
        ring.check_k(0)


def test_error_codes():
    e = errors.ContractViolation("value out of range")
    assert str(e).startswith("CONTRACT_VIOLATION:") and e.code == "CONTRACT_VIOLATION"
    e2 = errors.ContractViolation("TRUNC_PARAMS: bad f")
    assert e2.code == "TRUNC_PARAMS" and str(e2) == "TRUNC_PARAMS: bad f"
    e3 = errors.AdmissionError("no such component", code="NOT_ADMITTED_COMPONENT")
    assert e3.code == "NOT_ADMITTED_COMPONENT" and str(e3).startswith("NOT_ADMITTED_COMPONENT:")
    assert issubclass(errors.MaterialReuse, errors.MaterialError)
    assert issubclass(errors.MaterialStall, errors.OpenCrabError)
    for cls in (errors.UnauthorizedRelease, errors.ChannelError, errors.TelemetryError,
                errors.LedgerError, errors.BridgeError, errors.MaterialVersionMismatch):
        assert str(cls("x")).split(":")[0].isupper()
