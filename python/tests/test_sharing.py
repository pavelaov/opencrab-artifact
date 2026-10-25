"""Tests for opencrab2.lowlevel.sharing."""

from __future__ import annotations

import numpy as np
import pytest

from opencrab2 import errors
from opencrab2.lowlevel import ring, sharing


def test_share_reconstruct_z64():
    rng = np.random.default_rng(0)
    x = ring.rand_u64((1000,), rng)
    s0, s1 = sharing.share(x, rng)
    assert s0.dtype == np.uint64 and s1.dtype == np.uint64
    assert np.array_equal(sharing.reconstruct(s0, s1), x)
    assert np.sum(s0 == x) < 5          # shares are not the value
    assert sharing.share(np.uint64(7), rng)[0].shape == ()


def test_share_reconstruct_zk():
    rng = np.random.default_rng(1)
    for k in (1, 3, 8, 63, 64):
        x = ring.reduce_k(ring.rand_u64((500,), rng), k)
        s0, s1 = sharing.share_k(x, k, rng)
        assert np.all(s0 <= ring.mask_k(k)) and np.all(s1 <= ring.mask_k(k))
        assert np.array_equal(sharing.reconstruct_k(s0, s1, k), x)


def test_share_reconstruct_mod():
    rng = np.random.default_rng(2)
    for t in (3, 17, 68720050177, (1 << 62) - 57):
        x = rng.integers(0, t, size=2000, dtype=np.uint64)
        s0, s1 = sharing.share_mod(x, t, rng)
        assert np.all(s0 < np.uint64(t)) and np.all(s1 < np.uint64(t))
        assert np.array_equal(sharing.reconstruct_mod(s0, s1, t), x)
        if t > 3:
            assert np.sum(s0 == x) < 2000 // 2
    with pytest.raises(errors.ContractViolation):
        sharing.share_mod(np.array([17], np.uint64), 17, rng)
    with pytest.raises(errors.ContractViolation):
        sharing.share_mod(np.array([1], np.uint64), 16, rng)
    with pytest.raises(errors.ContractViolation):
        sharing.check_modulus(1 << 62)


def test_share_bits():
    rng = np.random.default_rng(3)
    b = rng.integers(0, 2, size=5000, dtype=np.uint8)
    s0, s1 = sharing.share_bits(b, rng)
    assert s0.dtype == np.uint8 and set(np.unique(s0).tolist()) <= {0, 1}
    assert np.array_equal(sharing.reconstruct_bits(s0, s1), b)
    assert 2000 < int(np.sum(s0)) < 3000
