"""Interval-mask distances and the per-session statistical budget.

For r uniform on a consecutive integer interval of length M, the total
variation distance between x0 + r and x1 + r is exactly

    TV(x0 + r, x1 + r) = min(1, |x0 - x1| / M).

This file provides that exact distance, the union-bound arithmetic that turns
a per-call term 2^-kappa into a session total, and `SessionBudget`, which
tracks sum_i eps_i against a pre-declared system target 2^-kappa_sys and
refuses an admission that would exceed it. A caller passes the budget to
``registry.admit``, which consults it whenever a route declares a statistical
term and refuses such a route when no budget is given (admission never lowers
a parameter to buy speed). The session paths pass none, so every route they
admit declares no statistical term; the budget is exercised by the tests.

What is tested (tests/test_interval_mask.py): the toy case lo=-64, hi=64,
x0=-8, x1=7 gives TV = 15/128 = 0.1171875; the union bound 8192 x 2^-20 =
2^-7; a budget of kappa_sys=64 refuses the 2^-7 charge and leaves the spent
total untouched; exact exhaustion at 2^-64 and refusal afterwards.

Design limits: the TV formula covers the raw integer masking term only, not a
keyholder's full evaluated-ciphertext view; a value that passes the
budget is an upper bound on a statistical term, not a measured leakage. Sums
are kept as exact rationals so that many tiny terms cannot be rounded away.
Error messages start with the stable code INTERVAL_MASK_PARAM.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import List, Optional

_CONSERVATIVE_SLACK = Fraction(1, 1 << 50)   # applied to non-integer log2 terms


def tv_interval_shift(delta: int, M: int) -> float:
    """Exact TV distance min(1, |delta| / M) between x+r and (x+delta)+r for r uniform on M consecutive integers."""
    if not isinstance(M, int) or isinstance(M, bool) or M <= 0:
        raise ValueError(f"INTERVAL_MASK_PARAM: interval length M must be a positive int, got {M!r}")
    if not isinstance(delta, int) or isinstance(delta, bool):
        raise ValueError(f"INTERVAL_MASK_PARAM: delta must be an int, got {type(delta).__name__}")
    d = abs(delta)
    if d >= M:
        return 1.0
    return float(Fraction(d, M))


def tv_interval_shift_bounds(lo: int, hi: int, x0: int, x1: int) -> float:
    """TV for r uniform on the half-open integer interval [lo, hi) (M = hi - lo) and inputs x0, x1."""
    for name, v in (("lo", lo), ("hi", hi), ("x0", x0), ("x1", x1)):
        if not isinstance(v, int) or isinstance(v, bool):
            raise ValueError(f"INTERVAL_MASK_PARAM: {name} must be an int, got {type(v).__name__}")
    if hi <= lo:
        raise ValueError(f"INTERVAL_MASK_PARAM: need hi > lo, got lo={lo}, hi={hi}")
    return tv_interval_shift(x1 - x0, hi - lo)


def union_bound_log2(log2_eps: float, calls: int) -> float:
    """log2 of the union bound calls * 2^log2_eps (e.g. -20 with 8192 calls gives -7)."""
    _check_calls(calls)
    _check_log2(log2_eps)
    return float(log2_eps) + math.log2(calls)


def _check_calls(calls: int) -> None:
    if not isinstance(calls, int) or isinstance(calls, bool) or calls < 1:
        raise ValueError(f"INTERVAL_MASK_PARAM: calls must be a positive int, got {calls!r}")


def _check_log2(log2_eps: float) -> None:
    if isinstance(log2_eps, bool) or not isinstance(log2_eps, (int, float)):
        raise ValueError(f"INTERVAL_MASK_PARAM: log2_eps must be a number, got {type(log2_eps).__name__}")
    if not math.isfinite(log2_eps):
        raise ValueError(f"INTERVAL_MASK_PARAM: log2_eps must be finite, got {log2_eps!r}")


def _eps_fraction(log2_eps: float, calls: int) -> Fraction:
    """Exact rational calls * 2^log2_eps; non-integer exponents are rounded up by a 2^-50 relative slack."""
    if float(log2_eps).is_integer():
        e = int(log2_eps)
        base = Fraction(1 << e, 1) if e >= 0 else Fraction(1, 1 << (-e))
    else:
        base = Fraction(2.0 ** float(log2_eps)) * (1 + _CONSERVATIVE_SLACK)
    return base * calls


def _log2_fraction(x: Fraction) -> float:
    if x <= 0:
        return -math.inf
    return math.log2(x.numerator) - math.log2(x.denominator)


class SessionBudget:
    """Tracks sum_i eps_i against the declared target 2^-kappa_sys (default 64).

    `charge(log2_eps, calls)` adds calls * 2^log2_eps if the new total stays
    at or below 2^-kappa_sys and returns True; otherwise returns False and
    charges nothing. `remaining_log2()` is log2 of the unspent budget (-inf
    when exhausted). `certificate()` lists every accepted charge with its
    label so that an admission record can carry complete call counts.
    """

    def __init__(self, kappa_sys: int = 64):
        if not isinstance(kappa_sys, int) or isinstance(kappa_sys, bool) or kappa_sys < 1:
            raise ValueError(f"INTERVAL_MASK_PARAM: kappa_sys must be a positive int, got {kappa_sys!r}")
        self.kappa_sys = kappa_sys
        self._limit = Fraction(1, 1 << kappa_sys)
        self._spent = Fraction(0)
        self.charges: List[dict] = []
        self.refusals = 0

    @property
    def limit(self) -> Fraction:
        return self._limit

    @property
    def spent(self) -> Fraction:
        return self._spent

    def would_fit(self, log2_eps: float, calls: int = 1) -> bool:
        _check_log2(log2_eps)
        _check_calls(calls)
        return self._spent + _eps_fraction(log2_eps, calls) <= self._limit

    def charge(self, log2_eps: float, calls: int = 1, label: Optional[str] = None) -> bool:
        """Charge calls * 2^log2_eps; False (and nothing charged) if the total would exceed 2^-kappa_sys."""
        _check_log2(log2_eps)
        _check_calls(calls)
        eps = _eps_fraction(log2_eps, calls)
        if self._spent + eps > self._limit:
            self.refusals += 1
            return False
        self._spent += eps
        self.charges.append({
            "label": label or "",
            "log2_eps": float(log2_eps),
            "calls": int(calls),
            "log2_total": _log2_fraction(eps),
        })
        return True

    def remaining_log2(self) -> float:
        """log2 of the unspent budget; -inf when the budget is exactly exhausted."""
        return _log2_fraction(self._limit - self._spent)

    def spent_log2(self) -> float:
        return _log2_fraction(self._spent)

    def certificate(self) -> dict:
        """Serialisable summary: target, spent, remaining, accepted charges, refusals."""
        return {
            "kappa_sys": self.kappa_sys,
            "spent_log2": self.spent_log2(),
            "remaining_log2": self.remaining_log2(),
            "charges": list(self.charges),
            "refusals": self.refusals,
        }
