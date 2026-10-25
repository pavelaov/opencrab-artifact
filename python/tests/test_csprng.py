"""csprng.SecureRandom (paper, Appendix D, Disclosure): ChaCha20 stream, exact ranges, and the rule that no secret of an
eligible dealer or of a default party comes from a numpy BitGenerator."""
import inspect

import numpy as np
import pytest

from opencrab2 import errors
from opencrab2.highlevel.contract import NumericContract
from opencrab2.highlevel.dealer import MODE_ISOLATED, Dealer
from opencrab2.highlevel.party import Party
from opencrab2.lowlevel import csprng, prg, ring, sharing
from opencrab2.lowlevel.csprng import SecureRandom

C = NumericContract()


def test_rfc7539_keystream_vector():
    # RFC 7539 Sec. 2.4.2: key 00..1f, nonce 00 00 00 00 00 00 00 4a 00 00 00 00, block counter 1
    r = SecureRandom(bytes(range(32)), bytes([0, 0, 0, 0, 0, 0, 0, 0x4A, 0, 0, 0, 0]))
    r.bytes(64)                                     # block 0
    assert r.bytes(16).hex() == "224f51f3401bd9e12fde276fb8631ded"
    with pytest.raises(errors.ContractViolation) as ei:
        SecureRandom(bytes(16))
    assert ei.value.code == "CSPRNG_KEY"


def test_determinism_and_os_flag():
    a = SecureRandom.from_seed(7)
    b = SecureRandom.from_seed(7)
    c = SecureRandom.from_seed(8)
    xa = a.integers(0, 1 << 64, size=(4, 3), dtype=np.uint64)
    assert np.array_equal(xa, b.integers(0, 1 << 64, size=(4, 3), dtype=np.uint64))
    assert not np.array_equal(xa, c.integers(0, 1 << 64, size=(4, 3), dtype=np.uint64))
    assert a.os_seeded is False and SecureRandom.from_os().os_seeded is True
    o1 = SecureRandom.from_os().integers(0, 1 << 64, size=8, dtype=np.uint64)
    o2 = csprng.secure_rng().integers(0, 1 << 64, size=8, dtype=np.uint64)
    assert not np.array_equal(o1, o2)
    assert csprng.rng_is_os_seeded(a) is False and csprng.rng_is_os_seeded(np.random.default_rng(1)) is False
    assert csprng.is_statistical_generator(np.random.default_rng(1)) and not csprng.is_statistical_generator(a)
    assert not hasattr(a, "bit_generator")


@pytest.mark.parametrize("low,high,dtype", [
    (0, 1 << 64, np.uint64), (0, 2, np.uint8), (0, 3, np.uint64), (0, 68720050177, np.uint64),
    (-1, 2, np.int64), (-8, 9, np.int64), (-(1 << 60), 1 << 60, np.int64), (0, (1 << 62) - 57, np.uint64),
    (5, 6, np.int64), (-(1 << 63), 1 << 63, np.int64), (0, 1 << 32, np.uint32),
])
def test_integers_range_shape_dtype(low, high, dtype):
    r = SecureRandom.from_seed(1)
    v = r.integers(low, high, size=(50, 40), dtype=dtype)
    assert v.shape == (50, 40) and v.dtype == np.dtype(dtype)
    assert int(v.min()) >= low and int(v.max()) < high
    if high - low <= 17:
        assert set(np.unique(v).tolist()) == set(range(low, high))      # every value of a small range occurs
    assert r.integers(low, high, size=0, dtype=dtype).shape == (0,)
    s = r.integers(low, high, dtype=dtype)
    assert np.ndim(s) == 0 and low <= int(s) < high


def test_rejection_sampling_is_unbiased_on_a_small_bound():
    r = SecureRandom.from_seed(2)
    v = r.integers(0, 3, size=300000, dtype=np.uint64)
    counts = np.bincount(v.astype(np.int64), minlength=3)
    assert np.all(np.abs(counts - 100000) < 1500)            # about 5.8 sigma
    assert r.integers(7, size=10).dtype == np.int64 and r.integers(0, 4, size=3, endpoint=True).max() <= 4


def test_bad_parameters_fail_closed():
    r = SecureRandom.from_seed(3)
    for args, kw in (((5, 5), {}), ((0, (1 << 64) + 1), {"dtype": np.uint64}), ((-1, 2), {"dtype": np.uint64}),
                     ((0, 300), {"dtype": np.uint8}), ((0, 2), {"dtype": np.float64}), ((0, 2), {"size": (-1,)})):
        with pytest.raises(errors.ContractViolation) as ei:
            r.integers(*args, **kw)
        assert ei.value.code == "CSPRNG_PARAMS"
    with pytest.raises(errors.ContractViolation):
        r.bytes(-1)


def test_library_helpers_accept_the_stream():
    r = SecureRandom.from_seed(4)
    x = ring.rand_u64((5,), r)
    s0, s1 = sharing.share(x, r)
    assert np.array_equal(sharing.reconstruct(s0, s1), x)
    b0, b1 = sharing.share_bits(np.array([0, 1, 1], np.uint8), r)
    assert sharing.reconstruct_bits(b0, b1).tolist() == [0, 1, 1]
    z0, z1 = sharing.share_mod(np.array([1, 2, 16], np.uint64), 17, r)
    assert sharing.reconstruct_mod(z0, z1, 17).tolist() == [1, 2, 16]
    assert prg.random_seeds(3, r).shape == (3, 2)
    assert prg.SecureRandom is SecureRandom


def test_eligible_dealer_and_default_party_do_not_use_a_numpy_bit_generator():
    # regression: a recipient bundle is a window of raw generator outputs, so a statistical generator leaks
    d = Dealer(C, None, MODE_ISOLATED)
    assert d.insecure is False
    assert isinstance(d.rng, SecureRandom) and d.rng.os_seeded
    assert not isinstance(d.rng, np.random.Generator) and not hasattr(d.rng, "bit_generator")
    (h0, p0, h1, p1), = d.make_beaver((64,), 1)
    assert h0["insecure"] is False and p0 != p1
    # every dealer whose source is a numpy generator is labelled insecure
    assert Dealer(C, np.random.default_rng(0), MODE_ISOLATED).insecure is True
    p = Party(0, None, None, C, None)
    assert isinstance(p.rng, SecureRandom) and p.rng.os_seeded and not hasattr(p.rng, "bit_generator")


def test_session_runner_party_source():
    from opencrab2.highlevel import session
    src = inspect.getsource(session.SessionRunner._make_party)
    assert "SecureRandom.from_os()" in src and "token_bytes" not in src
