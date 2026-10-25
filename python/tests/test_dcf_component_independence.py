"""KEY_COMPONENT_INDEPENDENCE regression (paper, Section 5.3, a finding on comparison keys).

The pre-repair rule (of the superseded engine) derived ONE pseudo-random value
word per GGM level and reused it for every payload component. For a key with
payload (beta_0, beta_1) the public correction words then satisfied

    cw_values[i][1] - cw_values[i][0] = +-(a_i - a_{i-1}) (beta_1 - beta_0),      a_{-1} = 0,

so one party read every bit of alpha (from the pattern of non-zero
differences) and the value beta_1 - beta_0 (from the first non-zero
difference) out of its OWN key. For DReLU material (beta = (1, r),
alpha = r mod 2^n) that is the mask r, hence the comparison input x = y - r.

The four mandatory checks:

1. the blind mask recovery succeeds on 0 of 2000 P = 2 DReLU keys;
2. no level of any of the 2000 keys has a correction-word difference equal to
   +-(beta_1 - beta_0);
3. the functional tests (exhaustive n <= 8, random 32 and 64 bits) pass for
   P in {1, 2, 3} and both payload groups;
4. P = 1 keys and outputs are bit-identical to the pre-repair rule
   (pinned vectors captured before the repair).

Positive control: the attack code is run against the pre-repair rule
(restored by patching the two Convert entry points of ``prg`` with the shared
word) and must succeed on 2000 of 2000 keys, and that emulation is itself
pinned to the digest of a P = 2 key generated before the repair.
A test that only ever saw the repaired keys could pass with a broken attack.

What these tests do NOT prove: that a key is pseudo-random. They show that
the specific relation is gone, that every component reads the PRG blocks the
Convert rule of ``lowlevel/prg.py`` names (independent pure-integer evaluation), and that the
construction stays correct. Pseudo-randomness rests on the cited DCF
construction with a PRG whose blocks are independent in the ideal-cipher
model.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from opencrab2.lowlevel import dcf, drelu, prg, ring

T_BFV = 68720050177
T_62 = (1 << 62) - 57
MASK64 = (1 << 64) - 1
TRIALS = 2000


# ---------------------------------------------------------------------------
# The attack (uses ONE party's key and nothing else)
# ---------------------------------------------------------------------------
def cw_difference(key) -> np.ndarray:
    """(n, M) differences cw_values[i][1] - cw_values[i][0] in the payload group of the key."""
    cv = key.cw_values
    with np.errstate(over="ignore"):
        if key.modulus is None:
            return (cv[:, 1, :] - cv[:, 0, :]).astype(np.uint64)
        t = np.uint64(key.modulus)
        return ((cv[:, 1, :] + t - cv[:, 0, :]) % t).astype(np.uint64)


def blind_mask_candidates(key) -> list:
    """Per key: the candidate set {1 - d, 1 + d} from the first level with a non-zero difference d."""
    diff = cw_difference(key)
    out = []
    for m in range(key.M):
        nz = np.nonzero(diff[:, m])[0]
        if nz.size == 0:
            out.append(set())
            continue
        d = int(diff[nz[0], m])
        out.append({(1 - d) & MASK64, (1 + d) & MASK64})
    return out


def blind_alpha(key) -> np.ndarray:
    """Per key: alpha read from the non-zero pattern, a_i = a_{i-1} xor [d_i != 0]."""
    nzbit = (cw_difference(key) != 0).astype(np.uint64)            # (n, M)
    bits = np.bitwise_xor.accumulate(nzbit, axis=0)                # a_i
    alpha = np.zeros(key.M, np.uint64)
    for i in range(key.n_bits):
        alpha = (alpha << np.uint64(1)) | bits[i]
    return alpha


def run_attack(mats, r, n_bits):
    """Counts over both parties and both keys: (mask recoveries, alpha recoveries, +-(beta_1 - beta_0) hits)."""
    mask_hits = alpha_hits = relation_hits = 0
    low = np.uint64((1 << n_bits) - 1) if n_bits < 64 else np.uint64(MASK64)
    half = np.uint64(1 << (n_bits - 1))
    with np.errstate(over="ignore"):
        p = r & low
        q = (p + half) & low
        bd = (r - np.uint64(1)).astype(np.uint64)                  # beta_1 - beta_0 = r - 1
        nbd = (np.uint64(0) - bd).astype(np.uint64)
    for mat in mats:
        for key, alpha in ((mat.key_p, p), (mat.key_q, q)):
            cands = blind_mask_candidates(key)
            mask_hits += sum(1 for m in range(key.M) if int(r[m]) in cands[m])
            alpha_hits += int(np.sum(blind_alpha(key) == alpha))
            diff = cw_difference(key)
            relation_hits += int(np.sum(((diff == bd[None, :]) | (diff == nbd[None, :])).any(axis=0)))
    return mask_hits, alpha_hits, relation_hits


# ---------------------------------------------------------------------------
# The pre-repair rule (positive control only; never used by the library)
# ---------------------------------------------------------------------------
def _legacy_expand_convert(seeds, P, modulus=None, reducer=None):
    sL, sR, vL, vR, tL, tR = prg.expand(seeds)
    if modulus is not None:
        vL = vL % np.uint64(modulus)
        vR = vR % np.uint64(modulus)
    return sL, sR, np.repeat(vL[:, None], P, axis=1), np.repeat(vR[:, None], P, axis=1), tL, tR


def _legacy_leaf_words(seeds, P, modulus=None, reducer=None):
    v = prg.expand(seeds)[2]
    if modulus is not None:
        v = v % np.uint64(modulus)
    return np.repeat(v[:, None], P, axis=1)


@pytest.fixture
def legacy_rule(monkeypatch):
    """Key generation with one shared value word per level (the broken rule)."""
    monkeypatch.setattr(prg, "expand_convert", _legacy_expand_convert)
    monkeypatch.setattr(prg, "leaf_words", _legacy_leaf_words)


def _digest(key) -> str:
    h = hashlib.sha256()
    for a in (key.seed, key.cw_seeds, key.cw_values, key.cw_tL, key.cw_tR, key.cw_final):
        h.update(np.ascontiguousarray(a, dtype="<u8").tobytes())
    return h.hexdigest()


# Captured before the repair with the inputs of _pin_inputs().
PIN_P1_DIGEST = ("cf0a26df5a5d6afe7af7bf8c274472dd07def1c4e56ba507abb888653969433c",
                 "5516826a5bc49b33c7c7b582e7158d3f42dc358532c3a9478b47588fa74c6ab8")
PIN_P1_SEED0 = [[18016930633132456890, 7013373421822782593], [17030886991259909300, 4827373169039523470],
                [5886301771240251012, 2178398751884474106], [4459800937283539335, 5875913863038956187]]
PIN_P1_CW_VALUES_LEVEL0 = [196601703534063792, 5536011061736846777, 5064682825080955167, 4423154416225875008]
PIN_P1_CW_FINAL = [3067029119738886321, 9645907433095303324, 14173216187687519402, 9224061025943197225]
PIN_P1_OUT0 = [17433861454814908251, 11487833931436836938, 6953698266051400916, 9328115912465367524]
PIN_P1_OUT1 = [1012882618894643372, 6958910142272714678, 11493045807658150700, 9118628161367640881]
PIN_P1_N32 = {
    "alpha": [4115325593, 2173385197, 3254087975],
    "beta": [9443182695686696458, 17933673198744876582, 11342960890024317363],
    "x": [2427050077, 2978039228, 1623000348],
    "digest": ("5111bd137edc749398ba2b1a42a172e0d833645497c6fc88804af42bc401bfb6",
               "dffc9528551d5a79b8d029092455af1af9c0290ff874b500a12cc1b3f01b3bfc"),
    "out0": [7810790838893730299, 2019422375999096033, 16528420452057219129],
    "out1": [1632391856792966159, 16427321697710455583, 13261284511676649850],
}
# Digests of a P = 2 key and of a Z_t key made before the repair: these MUST differ now.
PIN_OLD_P2_DIGEST0 = "c419269731e0b7e5d42335a042220673d097714b1281eabdb224d1ff406ffaba"
PIN_OLD_ZT_DIGEST0 = "374af50d70493e20ff060b87dc8b5337c05a225977d88ab7ca5b2c6645445c83"


def _pin_inputs():
    alpha = np.array([0x5A, 0xC3, 0x00, 0xFF], dtype=np.uint64)
    beta = np.array([[7], [0xFFFFFFFFFFFFFFF0], [1], [123456789]], dtype=np.uint64)
    x = np.array([0x59, 0xC3, 0x00, 0xFE], dtype=np.uint64)
    return alpha, beta, x


# ---------------------------------------------------------------------------
# Positive control: the attack breaks the pre-repair rule completely
# ---------------------------------------------------------------------------
def test_positive_control_attack_recovers_every_mask_under_the_old_rule(legacy_rule):
    alpha, beta, _x = _pin_inputs()
    with np.errstate(over="ignore"):
        beta2 = np.concatenate([beta, beta + np.uint64(5)], axis=1)
    old_p2, _ = dcf.gen_dcf_batch(alpha, beta2, 8, np.random.default_rng(1234))
    assert _digest(old_p2) == PIN_OLD_P2_DIGEST0          # the emulation IS the pre-repair rule
    old_zt, _ = dcf.gen_dcf_batch(alpha, np.ones((4, 1), np.uint64), 8, np.random.default_rng(1234), modulus=T_BFV)
    assert _digest(old_zt) == PIN_OLD_ZT_DIGEST0

    rng = np.random.default_rng(20261010)
    r = ring.rand_u64((TRIALS,), rng)
    mats = drelu.drelu_material_from_mask(r, 32, rng, with_product=True)
    mask_hits, alpha_hits, relation_hits = run_attack(mats, r, 32)
    assert mask_hits == 4 * TRIALS          # both parties, both keys: 2000 of 2000 each
    assert alpha_hits == 4 * TRIALS         # every bit of alpha as well
    assert relation_hits == 4 * TRIALS


# ---------------------------------------------------------------------------
# Check 1 and 2: the repaired keys
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_bits", [32, 64])
def test_blind_mask_recovery_fails_on_2000_drelu_keys(n_bits):
    rng = np.random.default_rng(20261010 + n_bits)
    r = ring.rand_u64((TRIALS,), rng)
    mats = drelu.drelu_material_from_mask(r, n_bits, rng, with_product=True)
    assert mats[0].key_p.P == 2 and mats[0].M == TRIALS
    mask_hits, alpha_hits, relation_hits = run_attack(mats, r, n_bits)
    assert mask_hits == 0           # check 1: r is recovered in 0 trials (each party, each key)
    assert relation_hits == 0       # check 2: no level difference equals +-(beta_1 - beta_0)
    assert alpha_hits == 0          # the non-zero pattern no longer spells alpha
    # the gate still computes the sign and r * sign
    x = rng.integers(-(1 << 30), 1 << 30, size=TRIALS, dtype=np.int64)
    with np.errstate(over="ignore"):
        y = (ring.from_signed(x) + r).astype(np.uint64)
    b0, rb0 = drelu.drelu_eval(mats[0], y)
    b1, rb1 = drelu.drelu_eval(mats[1], y)
    nonneg = (x >= 0).astype(np.uint64)
    assert np.array_equal(ring.add(b0, b1), nonneg)
    assert np.array_equal(ring.add(rb0, rb1), ring.mul(r, nonneg))


def test_correction_word_difference_is_not_the_payload_difference_at_any_level():
    # generic payloads (not only the DReLU shape) in both groups, P = 2 and P = 3
    rng = np.random.default_rng(31)
    for modulus in (None, T_BFV):
        for P in (2, 3):
            alpha = rng.integers(0, 1 << 32, size=TRIALS, dtype=np.uint64)
            beta = rng.integers(0, 1 << 64, size=(TRIALS, P), dtype=np.uint64)
            k0, _k1 = dcf.gen_dcf_batch(alpha, beta, 32, rng, modulus=modulus)
            bg = beta if modulus is None else beta % np.uint64(modulus)
            cv = k0.cw_values
            for c in range(1, P):
                with np.errstate(over="ignore"):
                    if modulus is None:
                        diff = (cv[:, c, :] - cv[:, 0, :]).astype(np.uint64)
                        bd = (bg[:, c] - bg[:, 0]).astype(np.uint64)
                        nbd = (np.uint64(0) - bd).astype(np.uint64)
                    else:
                        t = np.uint64(modulus)
                        diff = ((cv[:, c, :] + t - cv[:, 0, :]) % t).astype(np.uint64)
                        bd = ((bg[:, c] + t - bg[:, 0]) % t).astype(np.uint64)
                        nbd = ((t - bd) % t).astype(np.uint64)
                hits = int(np.sum((diff == bd[None, :]) | (diff == nbd[None, :])))
                assert hits == 0, (modulus, P, c)
                # under the old rule every level without a bit change gave a zero difference
                assert int(np.sum(diff == 0)) == 0, (modulus, P, c)


def test_old_rule_relation_in_the_field_group_too(legacy_rule):
    # positive control for the Z_t group: the shared word leaks there as well
    rng = np.random.default_rng(32)
    M = 200
    alpha = rng.integers(0, 1 << 16, size=M, dtype=np.uint64)
    beta = rng.integers(0, T_BFV, size=(M, 2), dtype=np.uint64)
    k0, _ = dcf.gen_dcf_batch(alpha, beta, 16, rng, modulus=T_BFV)
    assert np.array_equal(blind_alpha(k0), alpha)


# ---------------------------------------------------------------------------
# Check 3: functional tests for P in {1, 2, 3}, both payload groups
# ---------------------------------------------------------------------------
def _expected(alpha, x, beta, modulus):
    exp = np.where((x < alpha)[:, None], beta, np.uint64(0)).astype(np.uint64)
    return exp if modulus is None else exp % np.uint64(modulus)


@pytest.mark.parametrize("modulus", [None, 17, T_BFV, T_62])
@pytest.mark.parametrize("P", [1, 2, 3])
def test_functional_exhaustive_up_to_8_bits(P, modulus):
    for n_bits in range(1, 9):
        rng = np.random.default_rng(1000 + 10 * n_bits + P)
        N = 1 << n_bits
        alpha = np.repeat(np.arange(N, dtype=np.uint64), N)
        x = np.tile(np.arange(N, dtype=np.uint64), N)
        beta = rng.integers(0, 1 << 64, size=(alpha.shape[0], P), dtype=np.uint64)
        k0, k1 = dcf.gen_dcf_batch(alpha, beta, n_bits, rng, modulus=modulus)
        o0, o1 = dcf.eval_dcf_batch(k0, x), dcf.eval_dcf_batch(k1, x)
        if modulus is not None:
            assert np.all(o0 < np.uint64(modulus)) and np.all(o1 < np.uint64(modulus))
        assert np.array_equal(dcf.reconstruct_dcf(o0, o1, modulus), _expected(alpha, x, beta, modulus)), n_bits


@pytest.mark.parametrize("modulus", [None, 17, T_BFV, T_62])
@pytest.mark.parametrize("n_bits", [32, 64])
@pytest.mark.parametrize("P", [1, 2, 3])
def test_functional_random_32_and_64_bits(P, n_bits, modulus):
    rng = np.random.default_rng(2000 + n_bits + P)
    M = 2048
    hi = 1 << n_bits
    alpha = rng.integers(0, hi, size=M, dtype=np.uint64)
    x = rng.integers(0, hi, size=M, dtype=np.uint64)
    x[: M // 4] = alpha[: M // 4]
    with np.errstate(over="ignore"):
        x[M // 4: M // 2] = alpha[M // 4: M // 2] - np.uint64(1)
        x[M // 2: 3 * M // 4] = alpha[M // 2: 3 * M // 4] + np.uint64(1)
    if n_bits < 64:
        x &= np.uint64(hi - 1)
    beta = rng.integers(0, 1 << 64, size=(M, P), dtype=np.uint64)
    k0, k1 = dcf.gen_dcf_batch(alpha, beta, n_bits, rng, modulus=modulus)
    got = dcf.reconstruct_dcf(dcf.eval_dcf_batch(k0, x), dcf.eval_dcf_batch(k1, x), modulus)
    assert np.array_equal(got, _expected(alpha, x, beta, modulus))


# ---------------------------------------------------------------------------
# Check 4: P = 1 keys and outputs are bit-identical to the pre-repair rule
# ---------------------------------------------------------------------------
def test_p1_keys_and_outputs_are_bit_identical_to_the_first_implementation():
    alpha, beta, x = _pin_inputs()
    k0, k1 = dcf.gen_dcf_batch(alpha, beta, 8, np.random.default_rng(1234))
    assert (_digest(k0), _digest(k1)) == PIN_P1_DIGEST
    assert k0.seed.tolist() == PIN_P1_SEED0
    assert k0.cw_values[0, 0].tolist() == PIN_P1_CW_VALUES_LEVEL0
    assert k0.cw_final[0].tolist() == PIN_P1_CW_FINAL
    assert k0.cw_tL[0].tolist() == [1, 1, 1, 1] and k0.cw_tR[7].tolist() == [1, 1, 0, 0]
    assert dcf.eval_dcf_batch(k0, x)[:, 0].tolist() == PIN_P1_OUT0
    assert dcf.eval_dcf_batch(k1, x)[:, 0].tolist() == PIN_P1_OUT1
    assert dcf.reconstruct_dcf(dcf.eval_dcf_batch(k0, x), dcf.eval_dcf_batch(k1, x), None)[:, 0].tolist() == \
        [7, 0, 0, 123456789]

    rng = np.random.default_rng(99)
    alpha32 = rng.integers(0, 1 << 32, size=3, dtype=np.uint64)
    beta32 = rng.integers(0, 1 << 64, size=(3, 1), dtype=np.uint64)
    a0, a1 = dcf.gen_dcf_batch(alpha32, beta32, 32, rng)
    x32 = rng.integers(0, 1 << 32, size=3, dtype=np.uint64)
    assert alpha32.tolist() == PIN_P1_N32["alpha"] and beta32[:, 0].tolist() == PIN_P1_N32["beta"]
    assert x32.tolist() == PIN_P1_N32["x"]
    assert (_digest(a0), _digest(a1)) == PIN_P1_N32["digest"]
    assert dcf.eval_dcf_batch(a0, x32)[:, 0].tolist() == PIN_P1_N32["out0"]
    assert dcf.eval_dcf_batch(a1, x32)[:, 0].tolist() == PIN_P1_N32["out1"]


def test_p2_and_field_keys_differ_from_the_first_implementation():
    alpha, beta, _x = _pin_inputs()
    with np.errstate(over="ignore"):
        beta2 = np.concatenate([beta, beta + np.uint64(5)], axis=1)
    p2, _ = dcf.gen_dcf_batch(alpha, beta2, 8, np.random.default_rng(1234))
    assert _digest(p2) != PIN_OLD_P2_DIGEST0
    # component 0 of a P = 2 key still follows the P = 1 words; only the other components moved
    p1, _ = dcf.gen_dcf_batch(alpha, beta, 8, np.random.default_rng(1234))
    assert np.array_equal(p2.seed, p1.seed) and np.array_equal(p2.cw_seeds, p1.cw_seeds)
    assert np.array_equal(p2.cw_values[:, 0, :], p1.cw_values[:, 0, :])
    assert np.array_equal(p2.cw_final[0], p1.cw_final[0])
    zt, _ = dcf.gen_dcf_batch(alpha, np.ones((4, 1), np.uint64), 8, np.random.default_rng(1234), modulus=T_BFV)
    assert _digest(zt) != PIN_OLD_ZT_DIGEST0         # Z_t: 128-bit words, also for P = 1


# ---------------------------------------------------------------------------
# The Convert rule itself: an independent pure-integer evaluation of the rule in lowlevel/prg.py
# ---------------------------------------------------------------------------
_FIXED_KEY = bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")


def _B(seed: bytes, j: int) -> bytes:
    """B_j(s) = AES_K(s xor c_j) xor s, c_j the 16-byte big-endian integer j (byte arithmetic only)."""
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    x = bytes(a ^ b for a, b in zip(seed, j.to_bytes(16, "big")))
    enc = Cipher(algorithms.AES(_FIXED_KEY), modes.ECB()).encryptor()
    y = enc.update(x) + enc.finalize()
    return bytes(a ^ b for a, b in zip(y, seed))


def _convert_word(seed: bytes, c: int, side: int, modulus) -> int:
    if modulus is None:
        blk = _B(seed, 2 if c == 0 else 3 + c)
        return int.from_bytes(blk[8 * side: 8 * side + 8], "little")
    return int.from_bytes(_B(seed, 4 + 2 * c + side), "little") % modulus


def reference_eval(key, m: int, x: int) -> list:
    """Evaluate key m on x with Python integers, following the Convert rule of lowlevel/dcf.py line by line."""
    n, P, modulus = key.n_bits, key.P, key.modulus
    group = (1 << 64) if modulus is None else modulus
    s = key.seed[m].astype("<u8").tobytes()
    t = key.party
    out = [0] * P
    for i in range(n):
        xi = (x >> (n - 1 - i)) & 1
        for c in range(P):
            out[c] = (out[c] + _convert_word(s, c, xi, modulus) + t * int(key.cw_values[i, c, m])) % group
        ctl = _B(s, 3)
        nxt = _B(s, xi)
        tn = ctl[xi] & 1
        if t:
            cw = key.cw_seeds[i, :, m].astype("<u8").tobytes()
            nxt = bytes(a ^ b for a, b in zip(nxt, cw))
            tn ^= int(key.cw_tR[i, m] if xi else key.cw_tL[i, m])
        s, t = nxt, tn
    for c in range(P):
        out[c] = (out[c] + _convert_word(s, c, 0, modulus) + t * int(key.cw_final[c, m])) % group
    if key.party == 1:
        out = [(-v) % group for v in out]
    return out


@pytest.mark.parametrize("modulus", [None, 17, T_BFV, T_62])
@pytest.mark.parametrize("P", [1, 2, 3])
def test_eval_matches_the_pure_integer_reference_of_the_convert_rule(P, modulus):
    rng = np.random.default_rng(40 + P)
    M = 6
    for n_bits in (5, 32, 64):
        alpha = rng.integers(0, 1 << min(n_bits, 63), size=M, dtype=np.uint64)
        x = rng.integers(0, 1 << min(n_bits, 63), size=M, dtype=np.uint64)
        x[0] = alpha[0]
        beta = rng.integers(0, 1 << 64, size=(M, P), dtype=np.uint64)
        for key in dcf.gen_dcf_batch(alpha, beta, n_bits, rng, modulus=modulus):
            got = dcf.eval_dcf_batch(key, x)
            for m in range(M):
                assert got[m].tolist() == reference_eval(key, m, int(x[m])), (n_bits, key.party, m)
