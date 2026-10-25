"""Statistics for the evaluation (paper, Section 7.1, statistics).

Provides the paired percentile bootstrap for mean differences and for ratios
of means, the one-sided upper confidence bound on a failure probability after
n failure-free Bernoulli trials (1 - alpha^(1/n)), and a descriptive summary
(n, mean, median, p95, min, max, std). Every bootstrap is seeded through
`numpy.random.default_rng(seed)`, so a report regenerated from the same raw
records is byte-identical.

What is tested (tests/test_statistics.py): zero_failure_upper_bound(10) equals
1 - 0.05 ** 0.1; the bootstrap interval contains the sample mean; identical
seeds give identical intervals; ratio_ci contains the point ratio; summarize
on a known list; parameter validation.

Design limits: the percentile bootstrap assumes the resampled units are
independent (resample traces or sessions, never correlated tokens); tails such
as p95 need enough samples to be more than descriptive; the zero-failure bound
is a bound on a Bernoulli parameter, not a claim about correlated decisions.
Error messages start with the stable codes STATISTICS_PARAM,
STATISTICS_EMPTY, STATISTICS_NONFINITE, STATISTICS_RATIO_DENOMINATOR.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from ..errors import OpenCrabError


class StatisticsError(OpenCrabError, ValueError):
    """A statistics input outside its domain. Typed (an OpenCrabError with a stable ``.code``) and still a
    ValueError, so callers that caught ValueError keep working."""

    default_code = "STATISTICS_ERROR"

_CHUNK = 2000


def _as_vector(values, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise StatisticsError(f"STATISTICS_EMPTY: {name} has no values")
    if not np.all(np.isfinite(arr)):
        raise StatisticsError(f"STATISTICS_NONFINITE: {name} contains NaN or infinity")
    return arr


def _check_boot(n_boot: int, alpha: float) -> None:
    if not isinstance(n_boot, (int, np.integer)) or isinstance(n_boot, bool) or n_boot < 1:
        raise StatisticsError(f"STATISTICS_PARAM: n_boot must be a positive int, got {n_boot!r}")
    if not (0.0 < float(alpha) < 1.0):
        raise StatisticsError(f"STATISTICS_PARAM: alpha must lie in (0, 1), got {alpha!r}")


def _bootstrap_indices(rng: np.random.Generator, n: int, count: int) -> np.ndarray:
    return rng.integers(0, n, size=(count, n))


def _percentile_interval(samples: np.ndarray, alpha: float) -> Tuple[float, float]:
    lo, hi = np.percentile(samples, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(lo), float(hi)


def bootstrap_mean_samples(x: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """n_boot resampled means of x, computed in chunks to bound memory."""
    n = x.shape[0]
    out = np.empty(n_boot, dtype=np.float64)
    for start in range(0, n_boot, _CHUNK):
        m = min(_CHUNK, n_boot - start)
        idx = _bootstrap_indices(rng, n, m)
        out[start:start + m] = x[idx].mean(axis=1)
    return out


def paired_bootstrap_ci(diffs, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float, float]:
    """(mean, lo, hi): percentile bootstrap interval of the mean of paired differences."""
    _check_boot(n_boot, alpha)
    x = _as_vector(diffs, "diffs")
    rng = np.random.default_rng(seed)
    samples = bootstrap_mean_samples(x, n_boot, rng)
    lo, hi = _percentile_interval(samples, alpha)
    return float(x.mean()), lo, hi


def ratio_ci(a, b, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0,
             paired: Optional[bool] = None) -> Tuple[float, float, float]:
    """(ratio, lo, hi) for mean(a) / mean(b).

    Paired resampling (same index for a and b) when the lengths agree and
    `paired` is not False; otherwise independent two-sample resampling. All b
    must be non-zero in every resample (timings are positive), else the call
    raises STATISTICS_RATIO_DENOMINATOR.
    """
    _check_boot(n_boot, alpha)
    xa = _as_vector(a, "a")
    xb = _as_vector(b, "b")
    if paired is None:
        paired = xa.shape[0] == xb.shape[0]
    if paired and xa.shape[0] != xb.shape[0]:
        raise StatisticsError("STATISTICS_PARAM: paired ratio needs equal lengths")
    if xb.mean() == 0.0:
        raise StatisticsError("STATISTICS_RATIO_DENOMINATOR: mean(b) is zero")
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot, dtype=np.float64)
    for start in range(0, n_boot, _CHUNK):
        m = min(_CHUNK, n_boot - start)
        if paired:
            idx = _bootstrap_indices(rng, xa.shape[0], m)
            num = xa[idx].mean(axis=1)
            den = xb[idx].mean(axis=1)
        else:
            num = xa[_bootstrap_indices(rng, xa.shape[0], m)].mean(axis=1)
            den = xb[_bootstrap_indices(rng, xb.shape[0], m)].mean(axis=1)
        if np.any(den == 0.0):
            raise StatisticsError("STATISTICS_RATIO_DENOMINATOR: a resampled mean of b is zero")
        out[start:start + m] = num / den
    lo, hi = _percentile_interval(out, alpha)
    return float(xa.mean() / xb.mean()), lo, hi


def zero_failure_upper_bound(n: int, alpha: float = 0.05) -> float:
    """One-sided (1 - alpha) upper bound on p after n independent failure-free trials: 1 - alpha ** (1/n)."""
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 1:
        raise StatisticsError(f"STATISTICS_PARAM: n must be a positive int, got {n!r}")
    if not (0.0 < float(alpha) < 1.0):
        raise StatisticsError(f"STATISTICS_PARAM: alpha must lie in (0, 1), got {alpha!r}")
    return 1.0 - float(alpha) ** (1.0 / n)


def summarize(values) -> Dict[str, float]:
    """Descriptive summary: n, mean, median, p95, min, max, std (sample std, 0.0 for n = 1)."""
    x = _as_vector(values, "values")
    n = int(x.shape[0])
    return {
        "n": n,
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95.0)),
        "min": float(x.min()),
        "max": float(x.max()),
        "std": float(x.std(ddof=1)) if n > 1 else 0.0,
    }


summary = summarize
