"""Tests for opencrab2.lowlevel.prg (fixed-key MMO PRG and AES backend)."""

from __future__ import annotations

import numpy as np
import pytest

from opencrab2 import errors
from opencrab2.lowlevel import prg

# Cross-language test vector: the 64-byte expansion of seed 00..0f.
EXPAND_VECTOR_00_0F = (
    "50ff65cf9d6834b1d2003de297a2e26f922a73060b96decbfe08492b03d6868e"
    "5e1b4a1ba11e6a59b159221f73334c70b8b54cb235923debc21bd439a15689f8"
)


def test_fips197_appendix_c1_vector():
    key = bytes.fromhex("000102030405060708090a0b0c0d0e0f")
    pt = bytes.fromhex("00112233445566778899aabbccddeeff")
    assert prg.aes_ecb_encrypt(key, pt).hex() == "69c4e0d86a7b0430d8cdb78070b4c55a"
    with pytest.raises(errors.ContractViolation):
        prg.aes_ecb_encrypt(key[:8], pt)
    with pytest.raises(errors.ContractViolation):
        prg.aes_ecb_encrypt(key, pt[:5])


def test_sp800_38a_ecb_vector_with_fixed_key():
    # NIST SP 800-38A F.1.1 uses exactly the fixed key 2b7e1516...
    pt = np.frombuffer(bytes.fromhex("6bc1bee22e409f96e93d7e117393172a"), np.uint64).reshape(1, 2)
    ct = prg.aes_fixed_bulk(pt)
    assert ct.tobytes().hex() == "3ad77bb40d7a3660a89ecaf32466ef97"
    assert prg.FIXED_KEY.hex() == "2b7e151628aed2a6abf7158809cf4f3c"


def test_expand_byte_layout_matches_expand_bytes():
    seed = bytes(range(16))
    out = prg.expand_bytes(seed)
    assert len(out) == 64 and out.hex() == EXPAND_VECTOR_00_0F
    s = np.frombuffer(seed, np.uint64).reshape(1, 2)
    sL, sR, vL, vR, tL, tR = prg.expand(s)
    assert sL.tobytes() == out[0:16]
    assert sR.tobytes() == out[16:32]
    assert int(vL[0]) == int.from_bytes(out[32:40], "little")
    assert int(vR[0]) == int.from_bytes(out[40:48], "little")
    assert int(tL[0]) == out[48] & 1
    assert int(tR[0]) == out[49] & 1


def test_expand_matches_independent_aes():
    Cryptodome = pytest.importorskip("Cryptodome.Cipher")
    AES = Cryptodome.AES
    seed = bytes(range(16))
    out = b""
    for j in range(4):
        tweak = j.to_bytes(16, "big")
        x = bytes(a ^ b for a, b in zip(seed, tweak))
        y = AES.new(prg.FIXED_KEY, AES.MODE_ECB).encrypt(x)
        out += bytes(a ^ b for a, b in zip(y, seed))
    assert out.hex() == EXPAND_VECTOR_00_0F
    rng = np.random.default_rng(5)
    seeds = prg.random_seeds(7, rng)
    sL, sR, vL, vR, tL, tR = prg.expand(seeds)
    for m in range(7):
        ref = b""
        s = seeds[m].tobytes()
        for j in range(4):
            tweak = j.to_bytes(16, "big")
            y = AES.new(prg.FIXED_KEY, AES.MODE_ECB).encrypt(bytes(a ^ b for a, b in zip(s, tweak)))
            ref += bytes(a ^ b for a, b in zip(y, s))
        assert sL[m].tobytes() == ref[0:16] and sR[m].tobytes() == ref[16:32]
        assert int(vL[m]) == int.from_bytes(ref[32:40], "little")
        assert int(vR[m]) == int.from_bytes(ref[40:48], "little")
        assert int(tL[m]) == ref[48] & 1 and int(tR[m]) == ref[49] & 1


def test_expand_is_deterministic_and_batch_independent():
    rng = np.random.default_rng(3)
    seeds = prg.random_seeds(50, rng)
    a = prg.expand(seeds)
    b = prg.expand(seeds)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)
    single = prg.expand(seeds[10:11])
    for x, y in zip(a, single):
        assert np.array_equal(x[10:11], y)
    assert a[0].shape == (50, 2) and a[2].shape == (50,) and set(np.unique(a[4]).tolist()) <= {0, 1}
    with pytest.raises(errors.ContractViolation):
        prg.expand(np.zeros((3, 3), np.uint64))


def test_seeds_from_bytes_roundtrip():
    raw = bytes(range(32))
    s = prg.seeds_from_bytes(raw)
    assert s.shape == (2, 2) and s.tobytes() == raw
    with pytest.raises(errors.ContractViolation):
        prg.seeds_from_bytes(raw[:20])


def test_generator_from_bytes_and_os_random():
    seed = bytes(range(32))
    g1 = prg.generator_from_bytes(seed)
    g2 = prg.generator_from_bytes(seed)
    assert g1.integers(0, 1 << 62, size=5).tolist() == g2.integers(0, 1 << 62, size=5).tolist()
    with pytest.raises(errors.ContractViolation):
        prg.generator_from_bytes(seed[:16])
    a = prg.os_random_bytes(32)
    b = prg.os_random_bytes(32)
    assert len(a) == 32 and a != b


# ---------------------------------------------------------------------------
# Component-specific Convert
# ---------------------------------------------------------------------------
def _block(seed: bytes, j: int) -> bytes:
    """Byte-level B_j(s) = AES_K(s xor c_j) xor s with c_j the 16-byte big-endian integer j."""
    tweak = j.to_bytes(16, "big")
    y = prg.aes_ecb_encrypt(prg.FIXED_KEY, bytes(a ^ b for a, b in zip(seed, tweak)))
    return bytes(a ^ b for a, b in zip(y, seed))


def test_tweak_word_and_mmo_blocks_match_byte_level():
    assert int(prg.tweak_word(0)) == 0 and int(prg.tweak_word(3)) == 3 << 56
    assert int(prg.tweak_word(255)) == 255 << 56
    assert int(prg.tweak_word(513)) == (0x01 << 56) | (0x02 << 48)      # bytes 14, 15 = 02 01
    for j in range(4):
        assert int(prg.tweak_word(j)) == int(prg.TWEAKS64[j, 1])
    with pytest.raises(errors.ContractViolation):
        prg.tweak_word(-1)
    with pytest.raises(errors.ContractViolation):
        prg.tweak_word(1 << 64)
    rng = np.random.default_rng(21)
    seeds = prg.random_seeds(4, rng)
    tweaks = [0, 1, 2, 3, 4, 5, 9, 255, 256, 513, 70000]
    B = prg.mmo_blocks(seeds, tweaks)
    assert B.shape == (4, len(tweaks), 2) and B.dtype == np.uint64
    for m in range(4):
        s = seeds[m].tobytes()
        for k, j in enumerate(tweaks):
            assert B[m, k].tobytes() == _block(s, j), (m, j)
    seed = bytes(range(16))
    assert prg.mmo_bytes(seed, 300) == _block(seed, 300)
    assert b"".join(prg.mmo_bytes(seed, j) for j in range(4)).hex() == EXPAND_VECTOR_00_0F
    with pytest.raises(errors.ContractViolation):
        prg.mmo_bytes(seed[:8], 0)
    with pytest.raises(errors.ContractViolation):
        prg.mmo_blocks(np.zeros((2, 3), np.uint64), [0])


def test_convert_tweak_tables():
    assert prg.convert_tweaks(1) == [0, 1, 2, 3]
    assert prg.convert_tweaks(3) == [0, 1, 2, 3, 4, 5]
    assert prg.convert_tweaks(1, 17) == [0, 1, 3, 4, 5]
    assert prg.convert_tweaks(2, 17) == [0, 1, 3, 4, 5, 6, 7]
    assert [prg.ring_value_tweak(c) for c in range(4)] == [2, 4, 5, 6]
    assert [prg.zt_value_tweak(c, s) for c in range(3) for s in (0, 1)] == [4, 5, 6, 7, 8, 9]
    # no block is shared between two components, nor with the seed and control blocks
    for P in (1, 2, 3, 255):
        ring_blocks = [prg.ring_value_tweak(c) for c in range(P)]
        zt_blocks = [prg.zt_value_tweak(c, s) for c in range(P) for s in (0, 1)]
        assert len(set(ring_blocks)) == P and not set(ring_blocks) & {0, 1, 3}
        assert len(set(zt_blocks)) == 2 * P and not set(zt_blocks) & {0, 1, 2, 3}
    for bad in (0, 256):
        with pytest.raises(errors.ContractViolation):
            prg.convert_tweaks(bad)
    with pytest.raises(errors.ContractViolation):
        prg.zt_value_tweak(0, 2)


@pytest.mark.parametrize("modulus", [None, 17, 68720050177, (1 << 62) - 57])
@pytest.mark.parametrize("P", [1, 2, 3])
def test_expand_convert_uses_component_specific_blocks(P, modulus):
    rng = np.random.default_rng(22)
    seeds = prg.random_seeds(6, rng)
    sL, sR, vL, vR, tL, tR = prg.expand_convert(seeds, P, modulus)
    leaf = prg.leaf_words(seeds, P, modulus)
    assert vL.shape == (6, P) and vR.shape == (6, P) and leaf.shape == (6, P)
    for m in range(6):
        s = seeds[m].tobytes()
        assert sL[m].tobytes() == _block(s, 0) and sR[m].tobytes() == _block(s, 1)
        assert int(tL[m]) == _block(s, 3)[0] & 1 and int(tR[m]) == _block(s, 3)[1] & 1
        for c in range(P):
            if modulus is None:
                blk = _block(s, 2 if c == 0 else 3 + c)
                eL = int.from_bytes(blk[0:8], "little")
                eR = int.from_bytes(blk[8:16], "little")
            else:
                eL = int.from_bytes(_block(s, 4 + 2 * c), "little") % modulus
                eR = int.from_bytes(_block(s, 5 + 2 * c), "little") % modulus
            assert int(vL[m, c]) == eL and int(vR[m, c]) == eR, (m, c)
            assert int(leaf[m, c]) == eL
    if P > 1 and (modulus is None or modulus > (1 << 30)):
        # the words of two components are different pseudo-random values (a tiny field collides by chance)
        assert not np.any(vL[:, 0] == vL[:, 1]) and not np.any(vR[:, 0] == vR[:, 1])


def test_expand_convert_p1_ring_equals_base_layout():
    rng = np.random.default_rng(23)
    seeds = prg.random_seeds(40, rng)
    base = prg.expand(seeds)
    conv = prg.expand_convert(seeds, 1)
    for a, b in zip(base, conv):
        assert np.array_equal(a, b.reshape(a.shape))
    assert np.array_equal(prg.leaf_words(seeds, 1)[:, 0], base[2])


def test_mod128_is_exact_for_every_modulus_width():
    import random
    rnd = random.Random(24)
    edge_lo = [0, 2 ** 64 - 1, 2 ** 64 - 1, 0, 1]
    edge_hi = [0, 2 ** 64 - 1, 0, 2 ** 64 - 1, 1]
    for t in (2, 3, 17, (1 << 31) - 1, (1 << 32) - 5, (1 << 32) + 15, 68720050177, (1 << 48) - 59,
              (1 << 61) - 1, (1 << 62) - 57, (1 << 62) - 1):
        red = prg.Mod128(t)
        lo = np.array([rnd.getrandbits(64) for _ in range(500)] + edge_lo, dtype=np.uint64)
        hi = np.array([rnd.getrandbits(64) for _ in range(500)] + edge_hi, dtype=np.uint64)
        got = red.reduce(lo, hi)
        assert got.dtype == np.uint64
        assert got.tolist() == [(int(a) + (int(b) << 64)) % t for a, b in zip(lo.tolist(), hi.tolist())], t
        assert np.array_equal(prg.reduce128_mod(lo, hi, t), got)
        # every limb constant times the largest limb stays below 2^64 (no overflow by construction)
        assert all(int(c) * ((1 << red.w) - 1) < (1 << 64) for c in red.consts)
    assert len(prg.Mod128(17).consts) == 1 and len(prg.Mod128(68720050177).consts) == 2
    # a 64-bit word alone is NOT what the Z_t rule uses: the high word matters
    red = prg.Mod128(68720050177)
    lo = np.array([5], np.uint64)
    assert int(red.reduce(lo, np.array([1], np.uint64))[0]) == (5 + (1 << 64)) % 68720050177 != 5
    for bad in (0, 1, 1 << 62):
        with pytest.raises(errors.ContractViolation):
            prg.Mod128(bad)
    with pytest.raises(errors.ContractViolation):
        prg.expand_convert(np.zeros((1, 2), np.uint64), 1, 17, prg.Mod128(19))


def test_aes_ctr_keystream():
    key = bytes(range(16))
    nonce = b"\x00" * 8
    ks = prg.aes_ctr_keystream(key, nonce, 40)
    assert len(ks) == 40
    assert ks == prg.aes_ctr_keystream(key, nonce, 40)
    ks1 = prg.aes_ctr_keystream(key, nonce, 16, counter0=1)
    assert ks1 == ks[16:32]
    # block 0 is AES(nonce || 0) under the key
    assert ks[:16] == prg.aes_ecb_encrypt(key, nonce + (0).to_bytes(8, "big"))
    with pytest.raises(errors.ContractViolation):
        prg.aes_ctr_keystream(key, b"\x00" * 7, 16)
