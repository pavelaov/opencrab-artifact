"""Tests for opencrab2.lowlevel.drelu (one-round DReLU / ReLU)."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from opencrab2 import errors
from opencrab2.lowlevel import dcf, drelu, ring, sharing


def _run(m0, m1, x_signed):
    """Two-party emulation: share x, open y = x + r, evaluate both sides, reconstruct."""
    rng = np.random.default_rng(99)
    x = ring.from_signed(np.asarray(x_signed, np.int64))
    x0, x1 = sharing.share(x, rng)
    y = sharing.reconstruct(ring.add(x0, drelu.drelu_mask_share(m0)), ring.add(x1, drelu.drelu_mask_share(m1)))
    b0, rb0 = drelu.drelu_eval(m0, y)
    b1, rb1 = drelu.drelu_eval(m1, y)
    b = sharing.reconstruct(b0, b1)
    rb = None if rb0 is None else sharing.reconstruct(rb0, rb1)
    relu = None if rb0 is None else sharing.reconstruct(drelu.relu_from_drelu(y, b0, rb0),
                                                        drelu.relu_from_drelu(y, b1, rb1))
    return y, b, rb, relu


def test_exhaustive_n8_sign_and_relu():
    rng = np.random.default_rng(0)
    n = 8
    xs = np.arange(-128, 128, dtype=np.int64)
    reps = 64
    X = np.tile(xs, reps)
    r = ring.rand_u64((X.shape[0],), rng)
    # include masks whose low n bits sit on both sides of the wrap boundary
    r[:256] = (r[:256] & ~np.uint64(255)) | np.arange(256, dtype=np.uint64)
    m0, m1 = drelu.drelu_material_from_mask(r, n, rng, with_product=True)
    y, b, rb, relu = _run(m0, m1, X)
    assert np.array_equal(b, (X >= 0).astype(np.uint64))
    assert np.array_equal(rb, ring.mul(r, (X >= 0).astype(np.uint64)))
    assert np.array_equal(ring.to_signed(relu), np.maximum(X, 0))


@pytest.mark.parametrize("n", [16, 32, 64])
def test_random_wide(n):
    rng = np.random.default_rng(n)
    M = 2048
    lo = -(1 << (n - 1)) if n < 64 else -(1 << 62)
    hi = (1 << (n - 1)) if n < 64 else (1 << 62)
    X = rng.integers(lo, hi, size=M, dtype=np.int64)
    X[:8] = [0, -1, 1, lo, hi - 1, lo + 1, -2, 2]
    m0, m1 = drelu.drelu_material(M, n, rng, with_product=True)
    assert m0.M == M and m1.party == 1
    y, b, rb, relu = _run(m0, m1, X)
    assert np.array_equal(b, (X >= 0).astype(np.uint64))
    assert np.array_equal(ring.to_signed(relu), np.maximum(X, 0))


def test_without_product_payload():
    rng = np.random.default_rng(5)
    M = 500
    X = rng.integers(-(1 << 30), 1 << 30, size=M, dtype=np.int64)
    m0, m1 = drelu.drelu_material(M, 32, rng, with_product=False)
    assert m0.key_p.P == 1 and not m0.with_product
    assert np.all(m0.c_rb == 0) and np.all(m1.c_rb == 0)
    y, b, rb, relu = _run(m0, m1, X)
    assert rb is None and relu is None
    assert np.array_equal(b, (X >= 0).astype(np.uint64))


def test_bytes_formula_and_serialization():
    rng = np.random.default_rng(6)
    M = 23
    m0, m1 = drelu.drelu_material(M, 32, rng, with_product=True)
    assert drelu.drelu_bytes_per_element(32, True) == 2 * 1088 + 24
    assert drelu.drelu_bytes_per_element(32, False) == 2 * (16 + 32 * 25 + 8) + 24
    assert m0.nbytes() == M * drelu.drelu_bytes_per_element(32, True)
    b = m1.to_bytes()
    mm = drelu.DReLUMaterial.from_bytes(b)
    assert mm.M == M and mm.party == 1 and mm.n_bits == 32 and mm.with_product
    assert np.array_equal(mm.r_share, m1.r_share) and np.array_equal(mm.c_b, m1.c_b) and np.array_equal(mm.c_rb, m1.c_rb)
    X = rng.integers(-(1 << 30), 1 << 30, size=M, dtype=np.int64)
    y, b_ref, rb_ref, _ = _run(m0, m1, X)
    y2, b2, rb2, _ = _run(m0, mm, X)
    assert np.array_equal(b_ref, b2) and np.array_equal(rb_ref, rb2)
    with pytest.raises(errors.ContractViolation):
        drelu.DReLUMaterial.from_bytes(b[:-3])
    # wire size does not depend on the masks
    m0b, _ = drelu.drelu_material(M, 32, rng, with_product=True)
    assert len(m0b.to_bytes()) == len(m0.to_bytes())
    # container header OC2DRL01 (40 bytes), three share arrays, two keys in the binding DCF layout
    b = m1.to_bytes()
    assert drelu.HEADER_BYTES == 40 and len(b) == m1.wire_nbytes()
    assert len(b) == 40 + 24 * M + 2 * dcf.wire_bytes(M, 32, 2)
    magic, n_bits, party, wp, flags, MM, lkp, lkq = struct.unpack_from("<8sBBHIQQQ", b, 0)
    assert (magic, n_bits, party, wp, flags, MM) == (b"OC2DRL01", 32, 1, 1, 0, M)
    assert lkp == lkq == dcf.wire_bytes(M, 32, 2)
    off = 40 + 24 * M
    assert b[off:off + lkp] == m1.key_p.to_bytes() and b[off + lkp:] == m1.key_q.to_bytes()
    assert b[off] == 2                                   # first byte of an embedded key: wire version 2


def _legacy_key_bytes(key) -> bytes:
    hdr = struct.pack("<8sBBHQQI", b"OC2DCF01", key.n_bits, key.party, key.P, key.M, 0, 0)
    tpack = ((key.cw_tL & np.uint64(1)) | ((key.cw_tR & np.uint64(1)) << np.uint64(1))).astype(np.uint8)
    return b"".join([hdr, key.seed.astype("<u8").tobytes(), key.cw_seeds.astype("<u8").tobytes(),
                     key.cw_values.astype("<u8").tobytes(), tpack.tobytes(), key.cw_final.astype("<u8").tobytes()])


def _legacy_container(mat) -> bytes:
    kp, kq = _legacy_key_bytes(mat.key_p), _legacy_key_bytes(mat.key_q)
    hdr = struct.pack("<8sBBHIQQQ", b"OC2DRL01", mat.n_bits, mat.party, 1 if mat.with_product else 0, 0,
                      mat.M, len(kp), len(kq))
    return b"".join([hdr, mat.r_share.astype("<u8").tobytes(), mat.c_b.astype("<u8").tobytes(),
                     mat.c_rb.astype("<u8").tobytes(), kp, kq])


def test_old_bundles_parse_without_product_and_fail_cleanly_with_product():
    rng = np.random.default_rng(8)
    M = 12
    X = rng.integers(-(1 << 30), 1 << 30, size=M, dtype=np.int64)
    # P = 1 material of the legacy layout is unaffected by the comparison-key repair and still loads
    m0, m1 = drelu.drelu_material(M, 32, rng, with_product=False)
    old = drelu.DReLUMaterial.from_bytes(_legacy_container(m1))
    assert old.M == M and not old.with_product
    _, b, _, _ = _run(m0, old, X)
    assert np.array_equal(b, (X >= 0).astype(np.uint64))
    assert old.to_bytes() == m1.to_bytes()
    # payload (1, r) material of the first layout was generated with the leaking Convert rule: refused
    p0, _p1 = drelu.drelu_material(M, 32, rng, with_product=True)
    with pytest.raises(errors.ContractViolation) as ei:
        drelu.DReLUMaterial.from_bytes(_legacy_container(p0))
    assert ei.value.code == "DCF_WIRE_LEGACY_CONVERT"


def test_slice_concat_and_validation():
    rng = np.random.default_rng(7)
    M = 60
    X = rng.integers(-1000, 1000, size=M, dtype=np.int64)
    m0, m1 = drelu.drelu_material(M, 16, rng)
    a0, a1 = m0.slice(0, 25), m1.slice(0, 25)
    b0, b1 = m0.slice(25, 60), m1.slice(25, 60)
    _, ba, _, _ = _run(a0, a1, X[:25])
    _, bb, _, _ = _run(b0, b1, X[25:])
    assert np.array_equal(np.concatenate([ba, bb]), (X >= 0).astype(np.uint64))
    c0 = drelu.concat_drelu(a0, b0)
    c1 = drelu.concat_drelu(a1, b1)
    _, bc, _, _ = _run(c0, c1, X)
    assert np.array_equal(bc, (X >= 0).astype(np.uint64))
    with pytest.raises(errors.ContractViolation):
        drelu.concat_drelu(a0, b1)
    with pytest.raises(errors.ContractViolation):
        drelu.drelu_eval(m0, np.zeros(3, np.uint64))
    with pytest.raises(errors.ContractViolation):
        drelu.drelu_material(4, 1, rng)
