"""Tests for opencrab2.highlevel.statistics."""

from __future__ import annotations

import numpy as np
import pytest

from opencrab2.highlevel import statistics as st


def test_zero_failure_upper_bound_known_values():
    assert st.zero_failure_upper_bound(10) == pytest.approx(1 - 0.05 ** 0.1, rel=0, abs=1e-15)
    assert st.zero_failure_upper_bound(1) == pytest.approx(0.95)
    assert st.zero_failure_upper_bound(48) == pytest.approx(1 - 0.05 ** (1 / 48))
    assert st.zero_failure_upper_bound(10, alpha=0.1) == pytest.approx(1 - 0.1 ** 0.1)
    # rule of three: 1 - 0.05^(1/n) ~ 3/n for large n
    assert st.zero_failure_upper_bound(1000) == pytest.approx(3.0 / 1000, rel=0.01)
    with pytest.raises(ValueError, match="^STATISTICS_PARAM"):
        st.zero_failure_upper_bound(0)
    with pytest.raises(ValueError, match="^STATISTICS_PARAM"):
        st.zero_failure_upper_bound(10, alpha=1.0)


def test_paired_bootstrap_ci_contains_mean_and_is_deterministic():
    rng = np.random.default_rng(123)
    diffs = rng.normal(0.8, 0.5, size=40)
    mean, lo, hi = st.paired_bootstrap_ci(diffs, n_boot=5000, alpha=0.05, seed=7)
    assert mean == pytest.approx(diffs.mean())
    assert lo <= mean <= hi
    assert lo > 0.0                                   # clearly positive effect at this sample
    assert (mean, lo, hi) == st.paired_bootstrap_ci(diffs, n_boot=5000, alpha=0.05, seed=7)
    other = st.paired_bootstrap_ci(diffs, n_boot=5000, alpha=0.05, seed=8)
    assert other[0] == mean and (other[1], other[2]) != (lo, hi)
    # a wider interval at a smaller alpha
    _, lo99, hi99 = st.paired_bootstrap_ci(diffs, n_boot=5000, alpha=0.01, seed=7)
    assert lo99 <= lo and hi99 >= hi


def test_paired_bootstrap_ci_degenerate_and_validation():
    assert st.paired_bootstrap_ci([2.0, 2.0, 2.0], n_boot=100) == (2.0, 2.0, 2.0)
    with pytest.raises(ValueError, match="^STATISTICS_EMPTY"):
        st.paired_bootstrap_ci([])
    with pytest.raises(ValueError, match="^STATISTICS_NONFINITE"):
        st.paired_bootstrap_ci([1.0, float("nan")])
    with pytest.raises(ValueError, match="^STATISTICS_PARAM"):
        st.paired_bootstrap_ci([1.0, 2.0], n_boot=0)
    with pytest.raises(ValueError, match="^STATISTICS_PARAM"):
        st.paired_bootstrap_ci([1.0, 2.0], alpha=0.0)


def test_ratio_ci_contains_point_ratio():
    rng = np.random.default_rng(5)
    a = rng.normal(10.0, 1.0, size=30)
    b = rng.normal(5.0, 0.5, size=30)
    ratio, lo, hi = st.ratio_ci(a, b, n_boot=4000, seed=1)
    assert ratio == pytest.approx(a.mean() / b.mean())
    assert lo <= ratio <= hi
    assert 1.5 < lo and hi < 2.6
    assert (ratio, lo, hi) == st.ratio_ci(a, b, n_boot=4000, seed=1)
    # unequal lengths fall back to independent resampling
    r2, lo2, hi2 = st.ratio_ci(a, b[:20], n_boot=2000, seed=1)
    assert lo2 <= r2 <= hi2
    with pytest.raises(ValueError, match="^STATISTICS_PARAM"):
        st.ratio_ci(a, b[:20], paired=True)
    with pytest.raises(ValueError, match="^STATISTICS_RATIO_DENOMINATOR"):
        st.ratio_ci([1.0, 2.0], [0.0, 0.0])


def test_summarize_known_values():
    s = st.summarize([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert s["n"] == 10
    assert s["mean"] == 5.5 and s["median"] == 5.5
    assert s["min"] == 1.0 and s["max"] == 10.0
    assert s["p95"] == pytest.approx(9.55)
    assert s["std"] == pytest.approx(np.std(np.arange(1, 11), ddof=1))
    assert st.summary is st.summarize
    assert st.summarize([4.0])["std"] == 0.0
    with pytest.raises(ValueError, match="^STATISTICS_EMPTY"):
        st.summarize([])


def test_errors_are_typed_and_keep_their_codes():
    from opencrab2.errors import OpenCrabError
    from opencrab2.highlevel import statistics as st
    with pytest.raises(st.StatisticsError) as ei:
        st.zero_failure_upper_bound(0)
    assert isinstance(ei.value, OpenCrabError) and isinstance(ei.value, ValueError) and ei.value.code == "STATISTICS_PARAM"
    with pytest.raises(OpenCrabError) as ei:
        st.paired_bootstrap_ci([], n_boot=10, alpha=0.05, seed=1)
    assert ei.value.code == "STATISTICS_EMPTY"
