"""Tests for opencrab2.lowlevel.ring_to_field (A64 -> Z_t with the signed carry; paper, Section 5.2, RingToField)."""

from __future__ import annotations

import math
import struct

import numpy as np
import pytest

from opencrab2 import errors
from opencrab2.lowlevel import dcf, ring, ring_to_field as r2f, sharing

T_BFV = 68720050177
T_62 = (1 << 62) - 57          # largest admitted width: exercises the 31-limb exact 128-bit reduction
SMALL_T = (3, 5, 7, 11, 13, 17)


def _legacy_key_bytes(key) -> bytes:
    """A DCF key batch in the legacy wire layout (``OC2DCF01``, packed control bits)."""
    hdr = struct.pack("<8sBBHQQI", b"OC2DCF01", key.n_bits, key.party, key.P, key.M,
                      0 if key.modulus is None else int(key.modulus), 0)
    tpack = ((key.cw_tL & np.uint64(1)) | ((key.cw_tR & np.uint64(1)) << np.uint64(1))).astype(np.uint8)
    return b"".join([hdr, key.seed.astype("<u8").tobytes(), key.cw_seeds.astype("<u8").tobytes(),
                     key.cw_values.astype("<u8").tobytes(), tpack.tobytes(), key.cw_final.astype("<u8").tobytes()])


def _convert(m0, m1, a0, a1, k):
    """Two-party emulation: open y = u + r, evaluate both sides, reconstruct the Z_t value."""
    a0 = np.asarray(a0, np.uint64).reshape(-1)
    a1 = np.asarray(a1, np.uint64).reshape(-1)
    y = sharing.reconstruct_k(r2f.r2f_open_share(a0, m0, 0), r2f.r2f_open_share(a1, m1, 1), k)
    z0 = r2f.r2f_eval(m0, y, 0)
    z1 = r2f.r2f_eval(m1, y, 1)
    assert np.all(z0 < np.uint64(m0.t)) and np.all(z1 < np.uint64(m1.t))
    return sharing.reconstruct_mod(z0, z1, m0.t)


def test_audit_counterexamples_q256_t17():
    rng = np.random.default_rng(0)
    k, t = 8, 17
    cases = [((255, 2), 1, 2), ((128, 129), 1, 2), ((100, 100), 12, 13)]
    for (a0, a1), expect, naive in cases:
        u = (a0 + a1) % 256
        assert int(r2f.signed_mod(np.array([u], np.uint64), t, k)[0]) == expect
        assert int(r2f.naive_sharewise_mod(a0, a1, t)) == naive
        for r in (0, 1, 37, 127, 128, 200, 255):
            m0, m1 = r2f.r2f_material_from_mask(np.array([r], np.uint64), t, rng, k)
            assert _convert(m0, m1, [a0], [a1], k).tolist() == [expect], (a0, a1, r)


@pytest.mark.parametrize("t", SMALL_T)
def test_exhaustive_small_rings(t):
    rng = np.random.default_rng(t)
    for k in range(2, 9):
        q = 1 << k
        A0 = np.repeat(np.arange(q, dtype=np.uint64), q)
        A1 = np.tile(np.arange(q, dtype=np.uint64), q)
        U = ring.reduce_k(ring.add(A0, A1), k)
        expect = r2f.signed_mod(U, t, k)
        if k <= 4:   # all masks as well
            A0 = np.repeat(A0, q)
            A1 = np.repeat(A1, q)
            R = np.tile(np.arange(q, dtype=np.uint64), q * q)
            expect = np.repeat(expect, q)
        else:
            R = ring.reduce_k(ring.rand_u64(A0.shape, rng), k)
        m0, m1 = r2f.r2f_material_from_mask(R, t, rng, k)
        got = _convert(m0, m1, A0, A1, k)
        assert np.array_equal(got, expect), (k, t)


def test_random_64bit_admitted_modulus():
    rng = np.random.default_rng(1)
    M = 8192
    U = ring.rand_u64((M,), rng)
    U[:6] = [0, 1, 2 ** 64 - 1, 2 ** 63, 2 ** 63 - 1, 2 ** 63 + 1]
    a0, a1 = sharing.share(U, rng)
    m0, m1 = r2f.r2f_material(M, T_BFV, rng)
    assert m0.k == 64 and m0.t == T_BFV
    got = _convert(m0, m1, a0, a1, 64)
    assert np.array_equal(got, r2f.signed_mod(U, T_BFV))
    naive = r2f.naive_sharewise_mod(a0, a1, T_BFV)
    assert np.sum(naive != got) > M // 4     # the share-wise reduction is wrong on a large fraction


def test_random_64bit_small_modulus_and_contract_values():
    rng = np.random.default_rng(2)
    M = 4096
    X = rng.integers(-(1 << 39), 1 << 39, size=M, dtype=np.int64)
    U = ring.from_signed(X)
    a0, a1 = sharing.share(U, rng)
    for t in (17, 68720050177):
        m0, m1 = r2f.r2f_material(M, t, rng)
        got = _convert(m0, m1, a0, a1, 64)
        assert np.array_equal(got.astype(np.int64), np.mod(X, np.int64(t)))


def test_random_64bit_62_bit_modulus_uses_exact_128_bit_words():
    # t close to 2^62: a 64-bit word times (2^64 mod t) would overflow; the keys must still be exact
    rng = np.random.default_rng(7)
    M = 1024
    U = ring.rand_u64((M,), rng)
    U[:6] = [0, 1, 2 ** 64 - 1, 2 ** 63, 2 ** 63 - 1, 2 ** 63 + 1]
    a0, a1 = sharing.share(U, rng)
    m0, m1 = r2f.r2f_material(M, T_62, rng)
    got = _convert(m0, m1, a0, a1, 64)
    ref = np.array([((int(u) - (1 << 64)) if int(u) >= (1 << 63) else int(u)) % T_62 for u in U.tolist()], np.uint64)
    assert np.array_equal(got, ref)


def test_statistical_term_of_the_field_keys():
    # n_word * t / 2^128 with n_word = 2 key pairs x 2 parties x (2 per level + 1 leaf) (paper, Section 6, loss)
    assert r2f.r2f_zt_words_per_element() == 4 * (2 * 64 + 1) == 516
    assert r2f.r2f_zt_words_per_element(8) == 68
    e1 = r2f.r2f_statistical_log2_eps(T_BFV)
    assert math.isclose(e1, math.log2(516 * T_BFV) - 128.0, rel_tol=0, abs_tol=1e-9)
    assert -83.5 < e1 < -82.5
    assert math.isclose(r2f.r2f_statistical_log2_eps(T_BFV, 1 << 20), e1 + 20.0, abs_tol=1e-9)
    assert r2f.r2f_statistical_log2_eps(T_BFV, 1 << 18) < -64.0          # 2^18 conversions fit kappa_sys = 64
    assert r2f.r2f_statistical_log2_eps(T_62) < -56.9                    # 62-bit t: about 2^-57 per element
    # a reduced 64-bit word (the pre-repair rule) is t / 2^64 per word away, far above 2^-64
    assert math.log2(T_BFV) - 64.0 > -28.1
    with pytest.raises(errors.ContractViolation):
        r2f.r2f_statistical_log2_eps(T_BFV, 0)
    with pytest.raises(errors.ContractViolation):
        r2f.r2f_statistical_log2_eps(16)


def test_signed_mod_reference():
    assert r2f.signed_mod(np.array([255, 2, 128, 127], np.uint64), 17, 8).tolist() == [16, 2, 8, 8]
    assert r2f.signed_mod(np.array([2 ** 64 - 1], np.uint64), 7).tolist() == [6]
    with pytest.raises(errors.ContractViolation):
        r2f.signed_mod(np.array([1], np.uint64), 1)


def test_mulmod_scalar_against_python_ints():
    rng = np.random.default_rng(3)
    for t in (3, 17, T_BFV, (1 << 62) - 57):
        v = rng.integers(0, t, size=200, dtype=np.uint64)
        for c in (0, 1, 2, t - 1, int(rng.integers(0, t)), pow(2, 64, t)):
            got = r2f.mulmod_scalar(c, v, t)
            ref = np.array([(c * int(x)) % t for x in v.tolist()], np.uint64)
            assert np.array_equal(got, ref)


def test_modulus_validation():
    rng = np.random.default_rng(4)
    for bad in (2, 4, 15, 1 << 62, 9, 21, 68720050179 if not r2f._is_probable_prime(68720050179) else 25):
        with pytest.raises(errors.ContractViolation):
            r2f.check_field_modulus(bad)
        with pytest.raises(errors.ContractViolation):
            r2f.r2f_material(2, bad, rng)
    for good in (3, 5, 7, 11, 13, 17, T_BFV, (1 << 62) - 57):
        assert r2f.check_field_modulus(good) == good
    assert r2f._is_probable_prime((1 << 62) - 57)


def test_bytes_formula_and_serialization():
    rng = np.random.default_rng(5)
    M = 13
    m0, m1 = r2f.r2f_material(M, T_BFV, rng)
    assert r2f.r2f_bytes_per_element(T_BFV) == 2 * 1624 + 24
    assert m0.nbytes() == M * r2f.r2f_bytes_per_element(T_BFV)
    b = m1.to_bytes()
    mm = r2f.R2FMaterial.from_bytes(b)
    assert mm.M == M and mm.t == T_BFV and mm.k == 64 and mm.party == 1
    for name in ("r_share", "r_mod_t_share", "c_share"):
        assert np.array_equal(getattr(mm, name), getattr(m1, name))
    U = ring.rand_u64((M,), rng)
    a0, a1 = sharing.share(U, rng)
    assert np.array_equal(_convert(m0, mm, a0, a1, 64), r2f.signed_mod(U, T_BFV))
    with pytest.raises(errors.ContractViolation):
        r2f.R2FMaterial.from_bytes(b[:-1])
    m0b, _ = r2f.r2f_material(M, T_BFV, rng)
    assert len(m0b.to_bytes()) == len(m0.to_bytes())
    # container header OC2R2F01 (48 bytes), three share arrays, two Z_t keys in the binding DCF layout
    assert r2f.HEADER_BYTES == 48 and len(b) == m1.wire_nbytes()
    assert len(b) == 48 + 24 * M + 2 * dcf.wire_bytes(M, 64, 1)
    magic, k, party, _res, flags, MM, t, lka, lkc = struct.unpack_from("<8sBBHIQQQQ", b, 0)
    assert (magic, k, party, flags, MM, t) == (b"OC2R2F01", 64, 1, 0, M, T_BFV)
    assert lka == lkc == dcf.wire_bytes(M, 64, 1)
    off = 48 + 24 * M
    assert b[off:off + lka] == m1.key_A.to_bytes() and b[off + lka:] == m1.key_C.to_bytes()
    assert struct.unpack_from("<BBBBQQ", b, off) == (2, 1, 64, 1, M, T_BFV)
    # a bundle of the legacy layout carries Z_t keys made with 64-bit Convert words: refused
    old = struct.pack("<8sBBHIQQQQ", b"OC2R2F01", 64, 1, 0, 0, M, T_BFV,
                      len(_legacy_key_bytes(m1.key_A)), len(_legacy_key_bytes(m1.key_C)))
    old += b"".join(getattr(m1, n).astype("<u8").tobytes() for n in ("r_share", "r_mod_t_share", "c_share"))
    old += _legacy_key_bytes(m1.key_A) + _legacy_key_bytes(m1.key_C)
    with pytest.raises(errors.ContractViolation) as ei:
        r2f.R2FMaterial.from_bytes(old)
    assert ei.value.code == "DCF_WIRE_LEGACY_CONVERT"
    s0, s1 = m0.slice(0, 5), m1.slice(0, 5)
    assert np.array_equal(_convert(s0, s1, a0[:5], a1[:5], 64), r2f.signed_mod(U[:5], T_BFV))
    c0 = r2f.concat_r2f(s0, m0.slice(5, 13))
    c1 = r2f.concat_r2f(s1, m1.slice(5, 13))
    assert np.array_equal(_convert(c0, c1, a0, a1, 64), r2f.signed_mod(U, T_BFV))


def test_party_and_shape_validation():
    rng = np.random.default_rng(6)
    m0, m1 = r2f.r2f_material(4, 17, rng)
    y = np.zeros(4, np.uint64)
    with pytest.raises(errors.ContractViolation):
        r2f.r2f_eval(m0, y, 1)
    with pytest.raises(errors.ContractViolation):
        r2f.r2f_open_share(y, m1, 0)
    with pytest.raises(errors.ContractViolation):
        r2f.r2f_eval(m0, y[:2], 0)
    with pytest.raises(errors.ContractViolation):
        r2f.concat_r2f(m0, m1)
