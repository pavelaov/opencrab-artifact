"""Tests for opencrab2.lowlevel.trunc (faithful truncation gate; paper, Section 5.1, TruncFaithful)."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from opencrab2 import errors
from opencrab2.lowlevel import dcf, ring, sharing, trunc


def _legacy_key_bytes(key) -> bytes:
    """A DCF key batch in the legacy wire layout (``OC2DCF01``, packed control bits)."""
    hdr = struct.pack("<8sBBHQQI", b"OC2DCF01", key.n_bits, key.party, key.P, key.M,
                      0 if key.modulus is None else int(key.modulus), 0)
    tpack = ((key.cw_tL & np.uint64(1)) | ((key.cw_tR & np.uint64(1)) << np.uint64(1))).astype(np.uint8)
    return b"".join([hdr, key.seed.astype("<u8").tobytes(), key.cw_seeds.astype("<u8").tobytes(),
                     key.cw_values.astype("<u8").tobytes(), tpack.tobytes(), key.cw_final.astype("<u8").tobytes()])


def _gate(m0, m1, x_signed, k):
    """Two-party emulation over Z_{2^k}: share x, offset, open y, evaluate, reconstruct (signed)."""
    rng = np.random.default_rng(12345)
    x = ring.from_signed_k(np.asarray(x_signed, np.int64), k)
    x0, x1 = sharing.share_k(x, k, rng)
    y = sharing.reconstruct_k(trunc.trunc_open_share(x0, m0, 0), trunc.trunc_open_share(x1, m1, 1), k)
    z = sharing.reconstruct_k(trunc.trunc_faithful_eval(m0, y, 0), trunc.trunc_faithful_eval(m1, y, 1), k)
    return ring.to_signed_k(z, k)


def test_regression_k8_L6_f3_x_minus13_r37():
    assert trunc.reference_trunc_small(8, 6, 3, -13, 37) == -2
    assert trunc.reference_trunc_small_no_offset_term(8, 6, 3, -13, 37) == 2
    rng = np.random.default_rng(0)
    m0, m1 = trunc.trunc_material_from_mask(np.array([37], np.uint64), 3, 6, rng, k=8)
    assert _gate(m0, m1, [-13], 8).tolist() == [-2]


def test_exhaustive_small_rings():
    """Every (k <= 10, L, f, x, r) through the real DCF gate (as in the C++ test); reference
    oracle on k <= 6."""
    rng = np.random.default_rng(1)
    total = 0
    for k in range(2, 11):
        for L in range(2, k + 1):
            for f in range(1, L):
                q = 1 << k
                xs = np.arange(-(1 << (L - 1)), 1 << (L - 1), dtype=np.int64)
                rs = np.arange(q, dtype=np.uint64)
                X = np.repeat(xs, q)
                R = np.tile(rs, xs.shape[0])
                m0, m1 = trunc.trunc_material_from_mask(R, f, L, rng, k)
                got = _gate(m0, m1, X, k)
                assert np.array_equal(got, X >> f), (k, L, f)
                if k <= 6:
                    ref = np.array([trunc.reference_trunc_small(k, L, f, int(x), int(r)) for x, r in zip(X, R)],
                                   np.int64)
                    assert np.array_equal(ref, X >> f), (k, L, f)
                total += X.shape[0]
    assert total == 21_445_744          # sum over k = 2..10 of 2^k * sum_{L=2..k} (L - 1) 2^L


def test_random_64bit_magnitudes_and_boundaries():
    rng = np.random.default_rng(2)
    L, f = 40, 16
    mags = list(range(20, 40))
    per = 1_000_000 // len(mags)        # 10^6 elements across magnitudes 2^20..2^(L-1)
    parts = []
    for m in mags:
        mag = rng.integers(1 << (m - 1), 1 << m, size=per, dtype=np.int64)
        sign = rng.integers(0, 2, size=per) * 2 - 1
        parts.append(mag * sign)
    X = np.concatenate(parts)
    big = (1 << (L - 1)) - 1
    boundaries = np.array([0, 1, -1, big, -big, -(1 << (L - 1)), (1 << f) - 1, -(1 << f) + 1, 1 << f, -(1 << f),
                           (1 << f) + 1, -(1 << f) - 1, 3 * (1 << f) - 1, -3 * (1 << f) + 1], np.int64)
    X = np.concatenate([X, boundaries])
    assert X.shape[0] >= 1_000_000
    m0, m1 = trunc.trunc_material(X.shape[0], f, L, rng)
    assert m0.k == 64
    got = _gate(m0, m1, X, 64)
    assert np.array_equal(got, X >> f)


@pytest.mark.parametrize("L,f", [(40, 1), (40, 39), (64, 16), (64, 63), (17, 16), (2, 1)])
def test_random_64bit_other_widths(L, f):
    rng = np.random.default_rng(100 + L + f)
    M = 3000
    lo = -(1 << (L - 1))
    hi = (1 << (L - 1))
    X = rng.integers(lo, hi, size=M, dtype=np.int64)
    X[:4] = [lo, hi - 1, 0, -1]
    m0, m1 = trunc.trunc_material(M, f, L, rng)
    got = _gate(m0, m1, X, 64)
    assert np.array_equal(got, X >> f)


def test_offset_and_party_semantics():
    rng = np.random.default_rng(3)
    x = ring.rand_u64((5,), rng)
    assert np.array_equal(trunc.trunc_offset(x, 1, 40), x)
    assert np.array_equal(trunc.trunc_offset(x, 0, 40), ring.add(x, np.uint64(1 << 39)))
    m0, m1 = trunc.trunc_material(5, 16, 40, rng)
    with pytest.raises(errors.ContractViolation):
        trunc.trunc_faithful_eval(m0, x, 1)
    with pytest.raises(errors.ContractViolation):
        trunc.trunc_open_share(x, m1, 0)
    with pytest.raises(errors.ContractViolation):
        trunc.trunc_faithful_eval(m0, x[:3], 0)
    with pytest.raises(errors.ContractViolation):
        trunc.trunc_offset(x, 2, 40)


def test_parameter_validation():
    rng = np.random.default_rng(4)
    for f, L, k in ((0, 40, 64), (40, 40, 64), (16, 65, 64), (3, 9, 8), (1, 1, 8)):
        with pytest.raises(errors.ContractViolation):
            trunc.check_trunc_params(f, L, k)
        with pytest.raises(errors.ContractViolation):
            trunc.trunc_material_k(2, f, L, rng, k)
    with pytest.raises(errors.ContractViolation):
        trunc.reference_trunc_small(8, 6, 3, 32, 0)


def test_bytes_formula_and_serialization():
    rng = np.random.default_rng(5)
    M = 11
    m0, m1 = trunc.trunc_material(M, 16, 40, rng)
    assert trunc.trunc_bytes_per_element(16) == 2064
    assert m0.nbytes() == M * 2064
    b = m1.to_bytes()
    mm = trunc.TruncMaterial.from_bytes(b)
    assert mm.M == M and mm.f == 16 and mm.L == 40 and mm.k == 64 and mm.party == 1
    assert np.array_equal(mm.r_share, m1.r_share) and np.array_equal(mm.r_high_share, m1.r_high_share)
    X = rng.integers(-(1 << 39), 1 << 39, size=M, dtype=np.int64)
    assert np.array_equal(_gate(m0, mm, X, 64), X >> 16)
    with pytest.raises(errors.ContractViolation):
        trunc.TruncMaterial.from_bytes(b[:-1])
    m0b, _ = trunc.trunc_material(M, 16, 40, rng)
    assert len(m0b.to_bytes()) == len(m0.to_bytes())
    # container header OC2TRN01 (40 bytes), two share arrays, two keys in the binding DCF layout
    assert trunc.HEADER_BYTES == 40 and len(b) == m1.wire_nbytes()
    assert len(b) == 40 + 16 * M + dcf.wire_bytes(M, 64, 1) + dcf.wire_bytes(M, 16, 1)
    magic, f, L, k, party, flags, MM, lkw, lkb = struct.unpack_from("<8sBBBBIQQQ", b, 0)
    assert (magic, f, L, k, party, flags, MM) == (b"OC2TRN01", 16, 40, 64, 1, 0, M)
    assert (lkw, lkb) == (dcf.wire_bytes(M, 64, 1), dcf.wire_bytes(M, 16, 1))
    off = 40 + 16 * M
    assert b[off:off + lkw] == m1.key_wrap.to_bytes() and b[off + lkw:] == m1.key_borrow.to_bytes()
    # a bundle of the first layout (embedded OC2DCF01 keys, P = 1) still parses and gives the same gate
    old = struct.pack("<8sBBBBIQQQ", b"OC2TRN01", 16, 40, 64, 1, 0, M,
                      len(_legacy_key_bytes(m1.key_wrap)), len(_legacy_key_bytes(m1.key_borrow)))
    old += m1.r_share.astype("<u8").tobytes() + m1.r_high_share.astype("<u8").tobytes()
    old += _legacy_key_bytes(m1.key_wrap) + _legacy_key_bytes(m1.key_borrow)
    mo = trunc.TruncMaterial.from_bytes(old)
    assert np.array_equal(_gate(m0, mo, X, 64), X >> 16) and mo.to_bytes() == b
    # an embedded key that disagrees with the container header is refused
    swapped = b[:off] + m0.key_wrap.to_bytes() + b[off + lkw:]
    with pytest.raises(errors.ContractViolation):
        trunc.TruncMaterial.from_bytes(swapped)
    # slicing and concatenation keep the gate exact
    a0, a1 = m0.slice(0, 4), m1.slice(0, 4)
    b0, b1 = m0.slice(4, 11), m1.slice(4, 11)
    assert np.array_equal(_gate(a0, a1, X[:4], 64), X[:4] >> 16)
    c0, c1 = trunc.concat_trunc(a0, b0), trunc.concat_trunc(a1, b1)
    assert np.array_equal(_gate(c0, c1, X, 64), X >> 16)


def test_trunc_local_contract_bounded():
    rng = np.random.default_rng(6)
    M = 100_000
    X = rng.integers(-(1 << 39), 1 << 39, size=M, dtype=np.int64)
    x0, x1 = sharing.share(ring.from_signed(X), rng)
    z = ring.to_signed(sharing.reconstruct(trunc.trunc_local(x0, 16, 0), trunc.trunc_local(x1, 16, 1)))
    err = np.abs(z - (X >> 16))
    # wrap probability |x|/2^64 < 2^-25 per element: none expected in 10^5 draws
    assert int(np.max(err)) <= 1
    assert np.array_equal(trunc.trunc_local(x0, 0, 0), x0)
    with pytest.raises(errors.ContractViolation):
        trunc.trunc_local(x0, 64, 0)
