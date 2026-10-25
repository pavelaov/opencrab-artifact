"""Tests for opencrab2.lowlevel.interval_mask."""

from __future__ import annotations

import math

import pytest

from opencrab2.lowlevel import interval_mask as im


def test_v05_toy_case():
    # r uniform on [-64, 64): M = 128; x0 = -8, x1 = 7: |x0 - x1| = 15; TV = 15 / 128
    assert im.tv_interval_shift_bounds(-64, 64, -8, 7) == 0.1171875
    assert im.tv_interval_shift(15, 128) == 0.1171875
    assert im.tv_interval_shift(-15, 128) == 0.1171875


def test_tv_saturates_and_zero():
    assert im.tv_interval_shift(0, 128) == 0.0
    assert im.tv_interval_shift(128, 128) == 1.0
    assert im.tv_interval_shift(10_000, 128) == 1.0
    assert im.tv_interval_shift(1, 1 << 40) == 2.0 ** -40


def test_tv_parameter_validation():
    with pytest.raises(ValueError, match="^INTERVAL_MASK_PARAM"):
        im.tv_interval_shift(1, 0)
    with pytest.raises(ValueError, match="^INTERVAL_MASK_PARAM"):
        im.tv_interval_shift(1.5, 10)
    with pytest.raises(ValueError, match="^INTERVAL_MASK_PARAM"):
        im.tv_interval_shift_bounds(64, -64, 0, 1)


def test_union_bound_arithmetic_8192_slots():
    # 8192 slots x 2^-20 = 2^-7 (an upper bound, not measured leakage)
    assert im.union_bound_log2(-20, 8192) == -7.0
    assert im.union_bound_log2(-20.0, 1) == -20.0


def test_budget_refuses_2_pow_minus_7_and_charges_nothing():
    b = im.SessionBudget(kappa_sys=64)
    assert b.charge(-20, calls=8192, label="ckks_mask") is False
    assert b.spent == 0
    assert b.remaining_log2() == -64.0
    assert b.refusals == 1
    assert b.charges == []


def test_budget_exact_exhaustion_and_refusal():
    b = im.SessionBudget(kappa_sys=64)
    assert b.charge(-65) is True
    assert b.remaining_log2() == -65.0
    assert b.charge(-65) is True              # exactly fills 2^-64 (allowed: total <= 2^-kappa)
    assert b.remaining_log2() == -math.inf
    assert b.charge(-200) is False            # nothing fits any more
    assert b.spent_log2() == -64.0
    cert = b.certificate()
    assert cert["kappa_sys"] == 64
    assert len(cert["charges"]) == 2 and cert["refusals"] == 1


def test_budget_many_tiny_terms_are_not_rounded_away():
    b = im.SessionBudget(kappa_sys=64)
    assert b.charge(-65) is True              # half the budget
    for _ in range(1023):
        assert b.charge(-75) is True          # 1023 * 2^-75 < 2^-65
    assert b.charge(-75) is True              # exactly 1024 * 2^-75 = 2^-65: fills the budget
    assert b.remaining_log2() == -math.inf
    assert b.charge(-120) is False


def test_budget_with_calls_and_would_fit():
    b = im.SessionBudget(kappa_sys=40)
    assert b.would_fit(-50, calls=1024) is True        # 2^-40 exactly
    assert b.would_fit(-50, calls=1025) is False
    assert b.charge(-50, calls=512) is True
    assert b.remaining_log2() == -41.0


def test_budget_non_integer_log2_is_conservative():
    b = im.SessionBudget(kappa_sys=64)
    assert b.charge(-64.5) is True             # 2^-64.5 is about 0.71 of the 2^-64 budget
    assert b.remaining_log2() < -65.0          # remaining 2^-64 - 2^-64.5 is about 0.29 * 2^-64, below 2^-65
    assert b.charge(-64.5) is False            # a second 2^-64.5 term would exceed (2 * 2^-64.5 = 2^-63.5 > 2^-64)


def test_budget_parameter_validation():
    with pytest.raises(ValueError, match="^INTERVAL_MASK_PARAM"):
        im.SessionBudget(0)
    b = im.SessionBudget()
    with pytest.raises(ValueError, match="^INTERVAL_MASK_PARAM"):
        b.charge(float("nan"))
    with pytest.raises(ValueError, match="^INTERVAL_MASK_PARAM"):
        b.charge(-20, calls=0)
