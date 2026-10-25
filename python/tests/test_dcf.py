"""Tests for opencrab2.lowlevel.dcf (batched GGM DCF, both payload groups)."""

from __future__ import annotations

import struct
import time

import numpy as np
import pytest

from opencrab2 import errors
from opencrab2.lowlevel import dcf

T_BFV = 68720050177


def _expected(alpha, x, beta, modulus):
    exp = np.where((x < alpha)[:, None], beta, np.uint64(0)).astype(np.uint64)
    if modulus is not None:
        exp = exp % np.uint64(modulus)
    return exp


def _both(k0, k1, x):
    return dcf.reconstruct_dcf(dcf.eval_dcf_batch(k0, x), dcf.eval_dcf_batch(k1, x), k0.modulus)


def legacy_oc2dcf01_bytes(key) -> bytes:
    """The legacy wire layout (32-byte ``OC2DCF01`` header, control bits packed one byte per level and key)."""
    hdr = struct.pack("<8sBBHQQI", b"OC2DCF01", key.n_bits, key.party, key.P, key.M,
                      0 if key.modulus is None else int(key.modulus), 0)
    tpack = ((key.cw_tL & np.uint64(1)) | ((key.cw_tR & np.uint64(1)) << np.uint64(1))).astype(np.uint8)
    return b"".join([hdr, key.seed.astype("<u8").tobytes(), key.cw_seeds.astype("<u8").tobytes(),
                     key.cw_values.astype("<u8").tobytes(), tpack.tobytes(), key.cw_final.astype("<u8").tobytes()])


@pytest.mark.parametrize("P", [1, 2, 3])
@pytest.mark.parametrize("modulus", [None, 3, 17, T_BFV])
@pytest.mark.parametrize("n_bits", [1, 2, 3, 5, 8])
def test_exhaustive_small_domains(n_bits, modulus, P):
    rng = np.random.default_rng(100 + n_bits)
    N = 1 << n_bits
    alpha = np.repeat(np.arange(N, dtype=np.uint64), N)
    x = np.tile(np.arange(N, dtype=np.uint64), N)
    M = alpha.shape[0]
    beta = rng.integers(0, 1 << 64, size=(M, P), dtype=np.uint64)
    k0, k1 = dcf.gen_dcf_batch(alpha, beta, n_bits, rng, modulus=modulus)
    out0 = dcf.eval_dcf_batch(k0, x)
    out1 = dcf.eval_dcf_batch(k1, x)
    assert out0.shape == (M, P) and out0.dtype == np.uint64
    if modulus is not None:
        assert np.all(out0 < np.uint64(modulus)) and np.all(out1 < np.uint64(modulus))
    got = dcf.reconstruct_dcf(out0, out1, modulus)
    assert np.array_equal(got, _expected(alpha, x, beta, modulus))


@pytest.mark.parametrize("modulus", [None, 17, T_BFV, (1 << 62) - 57])
@pytest.mark.parametrize("n_bits", [32, 64])
def test_random_wide_domains(n_bits, modulus):
    rng = np.random.default_rng(200 + n_bits)
    M = 4096
    hi = 1 << n_bits
    alpha = rng.integers(0, hi, size=M, dtype=np.uint64)
    # mix of far, near and equal inputs
    x = rng.integers(0, hi, size=M, dtype=np.uint64)
    x[: M // 4] = alpha[: M // 4]
    with np.errstate(over="ignore"):
        x[M // 4: M // 2] = (alpha[M // 4: M // 2] - np.uint64(1)) & (np.uint64(hi - 1) if n_bits < 64 else np.uint64(2 ** 64 - 1))
        x[M // 2: 3 * M // 4] = (alpha[M // 2: 3 * M // 4] + np.uint64(1)) & (np.uint64(hi - 1) if n_bits < 64 else np.uint64(2 ** 64 - 1))
    beta = rng.integers(0, 1 << 64, size=(M, 3), dtype=np.uint64)
    k0, k1 = dcf.gen_dcf_batch(alpha, beta, n_bits, rng, modulus=modulus)
    assert np.array_equal(_both(k0, k1, x), _expected(alpha, x, beta, modulus))


def test_single_payload_vector_input():
    rng = np.random.default_rng(7)
    M = 512
    alpha = rng.integers(0, 1 << 16, size=M, dtype=np.uint64)
    x = rng.integers(0, 1 << 16, size=M, dtype=np.uint64)
    beta = np.ones(M, np.uint64)
    k0, k1 = dcf.gen_dcf_batch(alpha, beta, 16, rng)
    assert k0.P == 1
    assert np.array_equal(_both(k0, k1, x)[:, 0], (x < alpha).astype(np.uint64))


def test_key_bytes_formula_and_serialization_roundtrip():
    rng = np.random.default_rng(8)
    for modulus in (None, T_BFV):
        for n_bits, P in ((32, 2), (64, 1), (16, 3), (5, 1)):
            M = 37
            alpha = rng.integers(0, 1 << min(n_bits, 63), size=M, dtype=np.uint64)
            beta = rng.integers(0, 1 << 64, size=(M, P), dtype=np.uint64)
            k0, k1 = dcf.gen_dcf_batch(alpha, beta, n_bits, rng, modulus=modulus)
            assert k0.nbytes() == M * dcf.key_bytes_per_element(n_bits, P)
            b = k1.to_bytes()
            # binding layout: 20-byte header, six arrays as u64 LE
            assert len(b) == k1.wire_nbytes() == dcf.wire_bytes(M, n_bits, P)
            assert len(b) == 20 + 8 * (2 * M + 2 * n_bits * M + n_bits * P * M + 2 * n_bits * M + P * M)
            assert len(b) == dcf.HEADER_BYTES + M * dcf.key_wire_bytes_per_element(n_bits, P)
            assert struct.unpack_from("<BBBBQQ", b, 0) == (2, 1, n_bits, P, M, 0 if modulus is None else modulus)
            off = 20
            for name in ("seed", "cw_seeds", "cw_values", "cw_tL", "cw_tR", "cw_final"):
                arr = getattr(k1, name)
                assert b[off:off + 8 * arr.size] == np.ascontiguousarray(arr, dtype="<u8").tobytes(), name
                off += 8 * arr.size
            assert off == len(b)
            kk = dcf.DCFKeyBatch.from_bytes(b)
            assert kk.n_bits == n_bits and kk.party == 1 and kk.M == M and kk.P == P and kk.modulus == modulus
            for name in ("seed", "cw_seeds", "cw_values", "cw_tL", "cw_tR", "cw_final"):
                assert np.array_equal(getattr(kk, name), getattr(k1, name)), name
            x = rng.integers(0, 1 << min(n_bits, 63), size=M, dtype=np.uint64)
            assert np.array_equal(dcf.eval_dcf_batch(kk, x), dcf.eval_dcf_batch(k1, x))
            assert kk.to_bytes() == b
    assert dcf.HEADER_BYTES == 20 and dcf.LEGACY_HEADER_BYTES == 32 and dcf.WIRE_VERSION == 2
    assert dcf.key_bytes_per_element(32, 2) == 1088
    assert dcf.key_bytes_per_element(64, 1) == 1624
    assert dcf.key_bytes_per_element(16, 1) == 424
    # the binding layout spends one u64 per control bit: 15 bytes per level more than the charged formula
    for n_bits, P in ((32, 2), (64, 1), (16, 1)):
        assert dcf.key_wire_bytes_per_element(n_bits, P) == dcf.key_bytes_per_element(n_bits, P) + 15 * n_bits
    assert dcf.key_wire_bytes_per_element(32, 2) == 1568


def test_wire_size_independent_of_secrets():
    rng = np.random.default_rng(9)
    M = 16
    sizes = set()
    for _ in range(3):
        alpha = rng.integers(0, 1 << 32, size=M, dtype=np.uint64)
        beta = rng.integers(0, 1 << 64, size=(M, 2), dtype=np.uint64)
        k0, k1 = dcf.gen_dcf_batch(alpha, beta, 32, rng)
        sizes.add(len(k0.to_bytes()))
        sizes.add(len(k1.to_bytes()))
    assert len(sizes) == 1


def test_from_bytes_rejects_corruption():
    rng = np.random.default_rng(10)
    k0, _ = dcf.gen_dcf_batch(np.arange(4, dtype=np.uint64), np.ones((4, 1), np.uint64), 8, rng)
    b = k0.to_bytes()
    for bad in (b[:-1], b + b"\x00", b[:10], b"", b"\x01" + b[1:], b"\x03" + b[1:], b"XXXXXXXX" + b[8:],
                b[:1] + b"\x02" + b[2:],            # party 2
                b[:2] + b"\x00" + b[3:],            # n_bits 0
                b[:2] + b"\x41" + b[3:],            # n_bits 65
                b[:3] + b"\x00" + b[4:],            # P 0
                b[:3] + b"\x02" + b[4:],            # P 2 with the length of P 1
                b[:12] + struct.pack("<Q", 16) + b[20:],          # even modulus
                b[:12] + struct.pack("<Q", 1 << 62) + b[20:],     # modulus too large
                b[:4] + struct.pack("<Q", 1 << 50) + b[12:]):     # absurd batch size
        with pytest.raises(errors.ContractViolation) as ei:
            dcf.DCFKeyBatch.from_bytes(bad)
        assert ei.value.code == "DCF_WIRE"
    # a control-bit word that is not 0 or 1 is refused
    off_tL = 20 + 8 * (2 * 4 + 8 * 2 * 4 + 8 * 1 * 4)
    bad = bytearray(b)
    bad[off_tL] = 2
    with pytest.raises(errors.ContractViolation) as ei:
        dcf.DCFKeyBatch.from_bytes(bytes(bad))
    assert ei.value.code == "DCF_WIRE"
    # Z_t correction words must be below t
    z0, _ = dcf.gen_dcf_batch(np.arange(4, dtype=np.uint64), np.ones((4, 1), np.uint64), 8, rng, modulus=17)
    zb = bytearray(z0.to_bytes())
    zb[-8:] = struct.pack("<Q", 17)
    with pytest.raises(errors.ContractViolation) as ei:
        dcf.DCFKeyBatch.from_bytes(bytes(zb))
    assert ei.value.code == "DCF_WIRE"


def test_legacy_oc2dcf01_layout_parses_or_fails_cleanly():
    rng = np.random.default_rng(15)
    M = 9
    alpha = rng.integers(0, 1 << 16, size=M, dtype=np.uint64)
    x = rng.integers(0, 1 << 16, size=M, dtype=np.uint64)
    # P = 1 over Z_{2^64}: bit-identical under both Convert rules, so an old buffer still parses and evaluates
    k0, k1 = dcf.gen_dcf_batch(alpha, np.ones((M, 1), np.uint64), 16, rng)
    old = legacy_oc2dcf01_bytes(k1)
    assert len(old) == 32 + k1.nbytes()
    kk = dcf.DCFKeyBatch.from_bytes(old)
    for name in ("seed", "cw_seeds", "cw_values", "cw_tL", "cw_tR", "cw_final"):
        assert np.array_equal(getattr(kk, name), getattr(k1, name)), name
    assert kk.party == 1 and kk.modulus is None
    assert np.array_equal(dcf.eval_dcf_batch(kk, x), dcf.eval_dcf_batch(k1, x))
    assert kk.to_bytes() == k1.to_bytes()               # re-serialization always writes the binding layout
    # P >= 2 or Z_t keys in that layout used the pre-repair Convert rule: refused with a stable code
    p2, _ = dcf.gen_dcf_batch(alpha, np.ones((M, 2), np.uint64), 16, rng)
    zt, _ = dcf.gen_dcf_batch(alpha, np.ones((M, 1), np.uint64), 16, rng, modulus=T_BFV)
    for key in (p2, zt):
        with pytest.raises(errors.ContractViolation) as ei:
            dcf.DCFKeyBatch.from_bytes(legacy_oc2dcf01_bytes(key))
        assert ei.value.code == "DCF_WIRE_LEGACY_CONVERT"
    # malformed legacy buffers fail closed as well
    for bad in (old[:-1], old[:20], old[:28] + b"\x01\x00\x00\x00" + old[32:]):
        with pytest.raises(errors.ContractViolation) as ei:
            dcf.DCFKeyBatch.from_bytes(bad)
        assert ei.value.code == "DCF_WIRE"


def test_slice_and_concat():
    rng = np.random.default_rng(11)
    M = 100
    alpha = rng.integers(0, 1 << 20, size=M, dtype=np.uint64)
    beta = rng.integers(0, 1 << 64, size=(M, 2), dtype=np.uint64)
    x = rng.integers(0, 1 << 20, size=M, dtype=np.uint64)
    k0, k1 = dcf.gen_dcf_batch(alpha, beta, 20, rng, modulus=17)
    full = dcf.eval_dcf_batch(k0, x)
    a = k0.slice(0, 40)
    b = k0.slice(40, 100)
    assert a.M == 40 and b.M == 60
    assert np.array_equal(dcf.eval_dcf_batch(a, x[:40]), full[:40])
    cat = dcf.concat_keys(a, b)
    assert cat.M == M and np.array_equal(dcf.eval_dcf_batch(cat, x), full)
    with pytest.raises(errors.ContractViolation):
        dcf.concat_keys(a, k1.slice(40, 100))
    other, _ = dcf.gen_dcf_batch(alpha[:40], beta[:40], 20, rng)
    with pytest.raises(errors.ContractViolation):
        dcf.concat_keys(a, other)


def test_parameter_validation():
    rng = np.random.default_rng(12)
    alpha = np.arange(4, dtype=np.uint64)
    beta = np.ones((4, 1), np.uint64)
    with pytest.raises(errors.ContractViolation):
        dcf.gen_dcf_batch(alpha, beta, 0, rng)
    with pytest.raises(errors.ContractViolation):
        dcf.gen_dcf_batch(alpha, beta, 65, rng)
    with pytest.raises(errors.ContractViolation):
        dcf.gen_dcf_batch(alpha, beta, 8, rng, modulus=16)
    with pytest.raises(errors.ContractViolation):
        dcf.gen_dcf_batch(alpha, beta, 8, rng, modulus=1 << 62)
    # an odd PRIME t (paper, Section 5.2; the C++ check and ring_to_field refuse composites too)
    for composite in (9, 15, 561, 206160150531, (1 << 62) - 1, 4611686018427387845):
        assert not dcf.is_probable_prime(composite)
        with pytest.raises(errors.ContractViolation) as ei:
            dcf.gen_dcf_batch(alpha, beta, 8, rng, modulus=composite)
        assert ei.value.code == "DCF_MODULUS"
    for prime in (3, 5, 17, 68720050177, (1 << 61) - 1, (1 << 62) - 57):
        assert dcf.check_modulus(prime) == prime
    assert dcf.check_modulus(None) is None
    with pytest.raises(errors.ContractViolation):
        dcf.gen_dcf_batch(alpha, np.ones((3, 1), np.uint64), 8, rng)
    with pytest.raises(errors.ContractViolation):
        dcf.gen_dcf_batch(alpha, np.ones((4, 256), np.uint64), 8, rng)      # P is one byte on the wire
    k0, _ = dcf.gen_dcf_batch(alpha, beta, 8, rng)
    with pytest.raises(errors.ContractViolation):
        dcf.eval_dcf_batch(k0, np.arange(3, dtype=np.uint64))


def test_eval_threads_chunking_matches_single_chunk(monkeypatch):
    rng = np.random.default_rng(13)
    M = 3000
    alpha = rng.integers(0, 1 << 12, size=M, dtype=np.uint64)
    x = rng.integers(0, 1 << 12, size=M, dtype=np.uint64)
    beta = rng.integers(0, 1 << 64, size=(M, 2), dtype=np.uint64)
    k0, k1 = dcf.gen_dcf_batch(alpha, beta, 12, rng)
    single = dcf.eval_dcf_batch(k0, x)
    monkeypatch.setattr(dcf, "EVAL_CHUNK", 500)
    monkeypatch.setattr(dcf, "EVAL_THREADS", 3)
    assert np.array_equal(dcf.eval_dcf_batch(k0, x), single)
    assert np.array_equal(_both(k0, k1, x), _expected(alpha, x, beta, None))


def test_performance_smoke_eval_2_16_keys_32_bits():
    rng = np.random.default_rng(14)
    M = 1 << 16
    alpha = rng.integers(0, 1 << 32, size=M, dtype=np.uint64)
    x = rng.integers(0, 1 << 32, size=M, dtype=np.uint64)
    beta = np.ones((M, 1), np.uint64)
    k0, k1 = dcf.gen_dcf_batch(alpha, beta, 32, rng)
    t0 = time.perf_counter()
    out0 = dcf.eval_dcf_batch(k0, x)
    elapsed = time.perf_counter() - t0
    out1 = dcf.eval_dcf_batch(k1, x)
    assert np.array_equal(dcf.reconstruct_dcf(out0, out1, None)[:, 0], (x < alpha).astype(np.uint64))
    assert elapsed < 3.0, f"eval of 2^16 32-bit keys took {elapsed:.2f} s"


def test_many_payload_components_use_multi_byte_tweaks():
    # P = 130 in Z_t mode reaches tweak 5 + 2 * 129 = 263 > 255 (two tweak bytes)
    rng = np.random.default_rng(16)
    M = 6
    alpha = rng.integers(0, 1 << 8, size=M, dtype=np.uint64)
    x = np.array([0, 255, int(alpha[2]), max(int(alpha[3]) - 1, 0), 17, 200], dtype=np.uint64)
    for modulus, P in ((T_BFV, 130), (None, 255)):
        beta = rng.integers(0, 1 << 64, size=(M, P), dtype=np.uint64)
        k0, k1 = dcf.gen_dcf_batch(alpha, beta, 8, rng, modulus=modulus)
        assert np.array_equal(_both(k0, k1, x), _expected(alpha, x, beta, modulus))
        kk = dcf.DCFKeyBatch.from_bytes(k0.to_bytes())
        assert kk.P == P and np.array_equal(dcf.eval_dcf_batch(kk, x), dcf.eval_dcf_batch(k0, x))


def test_empty_batches_fail_closed_with_the_codes_of_the_cpp_generators():
    # M = 0 used to escape as "TypeError: memoryview: cannot cast view with zeros in shape or strides"
    from opencrab2.lowlevel import drelu, prg, ring_to_field, trunc
    rng = np.random.default_rng(0)
    cases = [(lambda: dcf.gen_dcf_batch(np.zeros(0, np.uint64), np.zeros((0, 1), np.uint64), 8, rng), "DCF_SHAPE"),
             (lambda: dcf.gen_dcf_batch(np.zeros(0, np.uint64), np.zeros((0, 2), np.uint64), 32, rng, modulus=17), "DCF_SHAPE"),
             (lambda: drelu.drelu_material(0, 32, rng), "DRELU_SHAPE"),
             (lambda: drelu.drelu_material(0, 32, rng, with_product=False), "DRELU_SHAPE"),
             (lambda: trunc.trunc_material(0, 16, 40, rng), "TRUNC_SHAPE"),
             (lambda: ring_to_field.r2f_material(0, 17, rng), "R2F_SHAPE")]
    for fn, code in cases:
        with pytest.raises(errors.ContractViolation) as ei:
            fn()
        assert ei.value.code == code
    # the PRG helper itself answers an empty batch with an empty result
    assert prg.aes_fixed_bulk(np.zeros((0, 2), np.uint64)).shape == (0, 2)
