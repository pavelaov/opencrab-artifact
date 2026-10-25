"""State-aware runtime controller and baseline policies (paper, Section 4.2, objective and decision).

Objective (myopic).  For every candidate p of the current state S_t:

    Qhat_t(p | S_t) = c_install(p | S_t)
                    + sum_{j=0}^{H-1} s_j * E[T_exec(p, t+j) | public history]
                    + c_wait(p | S_t)

H is the remaining declared budget (public), s_j the regime survival estimate
of a geometric dwell model fitted on the observed telemetry frames only
(no future access): with n_changes regime changes over n_steps observed
transitions and a prior mean dwell D, hazard h = (n_changes + 1) / (n_steps + D)
and s_j = (1 - h)^j.  ``stationary_survival = True`` (or D = inf) declares the
explicit fixed-capacity assumption s_j = 1.  ``c_wait`` is the declared
``c_wait_s`` of the candidate (0 unless the caller prices a material wait);
an infeasible candidate (material or memory) is excluded rather than priced.

Step by step.  E[T_exec(p, t+j)] is entry j of the candidate's
``exec_s_by_epoch``: the remaining epochs have different public shapes (the
prefill processes the prompt, a decode step one row), so the sum is taken
over the scheduled steps and never as H times the current epoch.  A
candidate without the lists (no schedule was given) is priced on one shape,
``exec_s_per_epoch`` for every step.  The baselines that count the remaining
horizon (B0, B2, B3, B5, B6) use the same per-step sum without the weights
(:func:`horizon_exec`); B1 is per epoch by definition.

Uncertainty.  ``q_hat`` returns the variance over the horizon,

    var = sigma_install^2 + A2 + epoch_corr * (A1^2 - A2),
    A1 = sum_j s_j sigma_j,  A2 = sum_j (s_j sigma_j)^2

with sigma_j entry j of ``exec_sigma_by_epoch`` (variances summed over the
horizon, the sigma of one epoch is never reused as the total).  On one shape
this is sigma_exec^2 * (S2 + epoch_corr * (S1^2 - S2)) with S1 = sum_j s_j and
S2 = sum_j s_j^2.  ``epoch_corr`` in [0, 1] is the declared share of exec
residual that is fully correlated across epochs: 0 gives independent epochs,
1 gives one common residual.  ``U_diff(p) = sqrt(var(keep) + var(p))`` is
the standard deviation of the cumulative difference Qhat(keep) - Qhat(p) with
keep and p treated as independent.

Decision.  ``lower_bound_gain(p) = Qhat(keep) - Qhat(p) - z * U_diff(p)``;
``p* = argmax_p lower_bound_gain(p)``; switch iff ``p* != keep``,
``lower_bound_gain(p*) > hysteresis_s`` and the incumbent has met
``min_residency`` epochs.  Residency counter rule (identical to the C++
controller): the counter is the number of epochs
the incumbent will have executed when the next decision is taken; a switch
sets it to 1, every kept epoch adds 1, and a switch is held while the counter
is below ``min_residency``.  A plan chosen at epoch t0 can therefore be
replaced at the first epoch t >= t0 + min_residency.  Already paid
sensing/planning is sunk for the decision; the session totals charge it.

Conditional guarantee.  If the true costs of every candidate lie in
the declared intervals and install/state costs are fully included, a
positive lower-bound gain beats keep over the comparison horizon.  No
unconditional no-harm claim is made: every :class:`Decision` lists its
assumptions.

Planner hook.  :class:`ControllerPolicy` (B4 and its ablations)
accepts an optional planner (``ExactAssign``, ``ILPAdapter``,
``GreedyPerOperator``).  The planner then chooses the assignment of the epoch
on a chain built from the candidates (:func:`candidates_chain`: one operator
per plan axis with the survival-weighted execution cost of each axis value and
the install objects it needs), and the decision rule decides between keep and that
assignment.  The planner x controller factorial therefore covers B4 as well
as B2 (:class:`PlannerPlusB2Policy`).

Cross-language vectors.  :meth:`StateAwareController.decide_with_survival`
takes the survival vector explicitly; with it the Python controller
reproduces ``cpp/tests/vectors/controller_vectors.json`` and
``cpp_controller_vectors.json`` (tests/test_cross_language_controller.py).

What is tested (tests/test_controller.py): the complete objective against
steady-state-first (keep 10/epoch, A 1/epoch with install 100, B 4/epoch
with install 1, H = 5: B is chosen with Q = 50, 105, 21, while the
steady-state-first rule keeps); a stationary environment (a cold incumbent
installs once when the horizon amortizes the install and never switches
again); exhaustive brute force over 3 candidates x 5 epochs x 20 random
cost tables (B4 with zero uncertainty and hysteresis equals the optimal
switching sequence; B2 equals B4); the conditional guarantee (assumptions
non-empty on every decision); the residency rule for
min_residency 0 to 3, hysteresis and uncertainty ablations; epoch_corr and
stationary_survival; the planner hook (exact planners equal the plain
controller without uncertainty, the greedy planner can differ); the dwell
model never reads future frames; the per-step objective (mean, variance,
the one-shape fallback, B2, B3 and the oracle on a schedule) and, with the
calibrated cost model, the first-step choice of B2, B4 and B4_no_install over
sensed bandwidths from 0.3 to 100 Gbit/s.
Design limits: the survival model is geometric (memoryless), residuals of
different candidates are treated as independent, and the objective omits the
expected-future-state term (the objective is myopic).  The C++ dwell estimator
is the same (the cross-language vectors check that both fit the stored
survival vector from the telemetry), and C++ keeps an infeasible incumbent
where this controller switches to the best feasible plan (KEEP_INFEASIBLE).
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .candidates import Candidate, SessionState, exec_cost, normalize_plan, plan_key

ASSUMPTIONS_B4: Tuple[str, ...] = (
    "costs within declared intervals",
    "regime survival model: geometric hazard fitted on observed frames only",
    "install cost candidate-specific and fully included",
    "sensing and planning cost sunk for the decision, charged in session totals",
    "public telemetry only: shapes, length buckets, probes, declared budget",
    "myopic objective: expected-future-state term omitted",
    "residuals of candidates independent across epochs and candidates",
)


RESIDENCY_CAP = 1 << 20      # the residency counter saturates here (same constant as the C++ controller)


@dataclass
class ControllerConfig:
    """Controller parameters; names, defaults and semantics identical to the C++ ``ControllerConfig``."""
    hysteresis_s: float = 0.5
    z: float = 1.0                    # confidence multiplier on U_diff (enters the lower bound only)
    min_residency: int = 1            # epochs a plan executes (switching epoch included) before another switch
    dwell_prior: float = 5.0          # prior mean dwell in epochs (inf = stationary)
    env_change_rel: float = 0.25      # relative change that declares a new regime
    ignore_install: bool = False      # ablation: c_install = 0 in the decision (the session still pays)
    use_uncertainty: bool = True      # ablation: U_diff = 0
    epoch_corr: float = 0.0           # share of fully correlated exec residual across epochs (0 = independent)
    stationary_survival: bool = False  # explicit fixed-capacity assumption s_j = 1

    def validate(self) -> "ControllerConfig":
        """Fail closed on parameters outside their domain (code CONTROLLER_PARAM)."""
        for name in ("hysteresis_s", "z", "env_change_rel", "epoch_corr"):
            v = float(getattr(self, name))
            if not math.isfinite(v) or v < 0.0:
                raise ValueError(f"CONTROLLER_PARAM: {name} must be finite and >= 0, got {v!r}")
        if float(self.epoch_corr) > 1.0:
            raise ValueError(f"CONTROLLER_PARAM: epoch_corr must lie in [0, 1], got {self.epoch_corr!r}")
        if int(self.min_residency) < 0:
            raise ValueError(f"CONTROLLER_PARAM: min_residency must be >= 0, got {self.min_residency!r}")
        return self

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Decision:
    chosen: dict
    switched: bool
    q_keep: float
    q_best: float
    lower_bound_gain: float
    u_diff: float
    reason: str
    candidates: List[dict]
    assumptions: List[str]

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# dwell model
# ---------------------------------------------------------------------------
class DwellModel:
    """Geometric regime-dwell model fitted on observed public frames (causal)."""

    def __init__(self, prior_mean_dwell: float = 5.0, change_rel: float = 0.25):
        if not (prior_mean_dwell > 0):
            raise ValueError(f"CONTROLLER_PARAM: dwell_prior must be positive, got {prior_mean_dwell!r}")
        self.prior = float(prior_mean_dwell)
        self.change_rel = float(change_rel)
        self.n_steps = 0
        self.n_changes = 0
        self.cur_dwell = 0
        self.dwells: List[int] = []
        self._last: Optional[Tuple[float, float]] = None

    def observe(self, bandwidth_bps: float, rtt_s: float) -> bool:
        """Feed one frame; returns True when a regime change was declared."""
        bw, rtt = float(bandwidth_bps), float(rtt_s)
        if self._last is None:
            self._last = (bw, rtt)
            self.cur_dwell = 1
            return False
        lbw, lrtt = self._last
        rel = max(abs(bw - lbw) / max(abs(lbw), 1e-9), abs(rtt - lrtt) / max(abs(lrtt), 1e-12))
        self.n_steps += 1
        self._last = (bw, rtt)
        if rel > self.change_rel:
            self.n_changes += 1
            self.dwells.append(self.cur_dwell)
            self.cur_dwell = 1
            return True
        self.cur_dwell += 1
        return False

    def hazard(self) -> float:
        if math.isinf(self.prior):
            return 0.0
        return (self.n_changes + 1.0) / (self.n_steps + self.prior)

    def survival(self, H: int) -> List[float]:
        h = min(1.0, max(0.0, self.hazard()))
        return [(1.0 - h) ** j for j in range(max(0, int(H)))]


# ---------------------------------------------------------------------------
# objective
# ---------------------------------------------------------------------------
def q_hat(candidate: Candidate, state: SessionState, survival: Sequence[float],
          include_install: bool = True, epoch_corr: float = 0.0) -> Tuple[float, float]:
    """(mean, variance) of Qhat over the horizon given the survival weights s_j.

    With per-step costs on the candidate: mean = c_install (when included) +
    sum_j s_j * exec_j + c_wait and variance = A2 + epoch_corr * (A1^2 - A2) +
    sigma_install^2 (when included), with a_j = s_j * sigma_j, A1 = sum a_j and
    A2 = sum a_j^2; exec_j and sigma_j are entry j of ``exec_s_by_epoch`` and
    ``exec_sigma_by_epoch``, and a step beyond the lists takes their last
    entry.  Without the lists (one shape): mean = c_install + S1 * exec +
    c_wait and variance = sigma_exec^2 * (S2 + epoch_corr * (S1^2 - S2)) +
    sigma_install^2, with S1 = sum s_j and S2 = sum s_j^2.  All sums are plain
    left to right accumulations, so that the C++ ``q_hat`` computes the same
    figures on the same inputs.
    """
    install = float(candidate.c_install_s) if include_install else 0.0
    wait = float(getattr(candidate, "c_wait_s", 0.0))
    by_s = getattr(candidate, "exec_s_by_epoch", None)
    if by_s:
        by_g = candidate.exec_sigma_by_epoch
        if len(by_g) != len(by_s):
            raise ValueError(f"CANDIDATE_SCHEDULE: {len(by_s)} step costs but {len(by_g)} step sigmas")
        last = len(by_s) - 1
        ex = a1 = a2 = 0.0
        for j, v in enumerate(survival):
            v = float(v)
            k = j if j < last else last
            ex += v * float(by_s[k])
            a = v * float(by_g[k])
            a1 += a
            a2 += a * a
        mean = install + ex + wait
        var = a2 + float(epoch_corr) * (a1 * a1 - a2)
    else:
        s1 = 0.0
        s2 = 0.0
        for v in survival:
            v = float(v)
            s1 += v
            s2 += v * v
        mean = install + s1 * float(candidate.exec_s_per_epoch) + wait
        sig2 = float(candidate.exec_sigma) * float(candidate.exec_sigma)
        var = sig2 * (s2 + float(epoch_corr) * (s1 * s1 - s2))
    if include_install:
        var += float(candidate.c_install_sigma) * float(candidate.c_install_sigma)
    return float(mean), float(var)


def horizon_exec(candidate: Candidate, H: int) -> float:
    """Execution time of the next ``H`` steps without survival weights (the baselines' horizon term).

    The per-step sum of ``exec_s_by_epoch`` when the candidate carries it (a
    step beyond the list takes its last entry), else ``H * exec_s_per_epoch``.
    """
    by_s = getattr(candidate, "exec_s_by_epoch", None)
    if not by_s:
        return H * candidate.exec_s_per_epoch
    last = len(by_s) - 1
    total = 0.0
    for j in range(int(H)):
        total += float(by_s[j if j < last else last])
    return total


def _keep_of(state: SessionState, candidates: Sequence[Candidate]) -> Candidate:
    for c in candidates:
        if c.is_keep:
            return c
    for c in candidates:
        if c.plan == state.incumbent:
            return c
    raise ValueError("CANDIDATES_NO_KEEP: the candidate list must contain the incumbent (keep)")


class StateAwareController:
    """The state-aware controller; deterministic given identical frames and candidates on both parties."""

    def __init__(self, cfg: Optional[ControllerConfig] = None):
        self.cfg = (cfg or ControllerConfig()).validate()
        prior = math.inf if self.cfg.stationary_survival else self.cfg.dwell_prior
        self.dwell = DwellModel(prior, self.cfg.env_change_rel)
        self.epochs_since_switch = RESIDENCY_CAP
        self.frames_seen = 0
        self.history: List[Decision] = []

    def observe(self, frame: Any) -> None:
        """Feed a telemetry frame (TelemetryFrame or mapping with bandwidth_bps and rtt_s)."""
        if isinstance(frame, Mapping):
            bw, rtt = frame["bandwidth_bps"], frame["rtt_s"]
        else:
            bw, rtt = frame.bandwidth_bps, frame.rtt_s
        self.dwell.observe(bw, rtt)
        self.frames_seen += 1

    def stationary(self) -> bool:
        """True when the explicit fixed-capacity assumption s_j = 1 is in force."""
        return bool(self.cfg.stationary_survival) or math.isinf(float(self.cfg.dwell_prior))

    def survival(self, H: int) -> List[float]:
        if self.cfg.stationary_survival:
            return [1.0] * max(0, int(H))
        return self.dwell.survival(H)

    def notify_epoch(self, switched: bool) -> None:
        """Residency counter: epochs the incumbent will have executed at the next decision.

        A switch sets it to 1 (the switching epoch counts), every kept epoch
        adds 1; a plan chosen at epoch t0 is replaceable at the first epoch
        t >= t0 + min_residency (the same counter rule as the C++ controller).
        """
        self.epochs_since_switch = 1 if switched else min(self.epochs_since_switch + 1, RESIDENCY_CAP)

    def reset_residency(self) -> None:
        self.epochs_since_switch = RESIDENCY_CAP

    def assumptions(self) -> List[str]:
        """The assumption set under which the lower-bound-gain argument holds for this configuration."""
        cfg = self.cfg
        out: List[str] = []
        for a in ASSUMPTIONS_B4:
            if a.startswith("regime survival model") and self.stationary():
                out.append("regime survival model: stationary, explicit fixed-capacity assumption s_j = 1")
            elif a.startswith("install cost") and cfg.ignore_install:
                out.append("install cost ignored in the decision (ablation); the session still pays the real install")
            elif a.startswith("residuals of candidates") and float(cfg.epoch_corr) > 0.0:
                out.append("exec residuals of one candidate correlated across epochs with the declared share "
                           f"epoch_corr = {float(cfg.epoch_corr)!r}; candidates independent")
            else:
                out.append(a)
        if not cfg.use_uncertainty:
            out.append("uncertainty ignored (ablation): U_diff = 0")
        return out

    def decide(self, state: SessionState, candidates: Sequence[Candidate], epoch: int) -> Decision:
        """One decision per epoch from the fitted survival model; updates the residency counter."""
        H = max(0, int(state.remaining_epochs))
        return self.decide_with_survival(state, candidates, epoch, self.survival(H))

    def decide_with_survival(self, state: SessionState, candidates: Sequence[Candidate], epoch: int,
                             survival: Sequence[float]) -> Decision:
        """The decision rule with an explicit survival vector (cross-language vectors, planner hook, tests).

        ``survival[j]`` weights epoch t + j; its length is the horizon.  The
        residency counter is updated exactly as in :meth:`decide`.
        """
        cfg = self.cfg
        s = [float(v) for v in survival]
        keep = _keep_of(state, candidates)
        inc = not cfg.ignore_install
        corr = float(cfg.epoch_corr)
        q_keep, v_keep = q_hat(keep, state, s, include_install=inc, epoch_corr=corr)
        records: List[dict] = []
        # p* = argmax over every candidate of (Qhat(keep) - Qhat(p) - z U_diff(p)); keep scores 0
        best: Candidate = keep
        best_lbg = 0.0
        best_q = q_keep
        best_u = 0.0
        any_challenger = False
        best_feasible_q: Optional[Tuple[float, Candidate]] = None
        for c in candidates:
            if c is keep:
                records.append({"plan": dict(c.plan), "is_keep": True, "feasible": c.feasible, "reason": c.reason,
                                "q_mean": q_keep, "q_var": v_keep, "gain": 0.0, "lower_bound_gain": 0.0, "u_diff": 0.0,
                                "install_objects": list(c.install_objects), "c_install_s": c.c_install_s})
                continue
            q_p, v_p = q_hat(c, state, s, include_install=inc, epoch_corr=corr)
            u = math.sqrt(max(v_keep + v_p, 0.0)) if cfg.use_uncertainty else 0.0
            gain = q_keep - q_p
            lbg = gain - cfg.z * u
            records.append({"plan": dict(c.plan), "is_keep": False, "feasible": c.feasible, "reason": c.reason,
                            "q_mean": q_p, "q_var": v_p, "gain": gain, "lower_bound_gain": lbg, "u_diff": u,
                            "install_objects": list(c.install_objects), "c_install_s": c.c_install_s})
            if not c.feasible:
                continue
            any_challenger = True
            if best_feasible_q is None or q_p < best_feasible_q[0]:
                best_feasible_q = (q_p, c)
            if lbg > best_lbg:
                best, best_lbg, best_q, best_u = c, lbg, q_p, u
        switched = False
        chosen = keep.plan
        if not keep.feasible:
            if best_feasible_q is None:
                reason = "NO_FEASIBLE_CANDIDATE: keep infeasible and no feasible challenger"
            else:
                chosen, switched = best_feasible_q[1].plan, True
                best_q = best_feasible_q[0]
                best_lbg = q_keep - best_q
                reason = f"KEEP_INFEASIBLE: {keep.reason}"
        elif not any_challenger:
            reason = "NO_CHALLENGER: no feasible candidate other than keep"
        elif best is keep:
            reason = "KEEP_IS_ARGMAX: no challenger has a positive lower-bound gain"
        elif self.epochs_since_switch < cfg.min_residency:
            reason = f"MIN_RESIDENCY_HOLD: {self.epochs_since_switch} < {cfg.min_residency}"
        elif best_lbg > cfg.hysteresis_s:
            chosen, switched = best.plan, True
            reason = "LOWER_BOUND_GAIN_EXCEEDS_HYSTERESIS"
        else:
            reason = "GAIN_BELOW_THRESHOLD: lower-bound gain does not exceed hysteresis"
        self.notify_epoch(switched)
        dec = Decision(chosen=dict(chosen), switched=switched, q_keep=float(q_keep), q_best=float(best_q),
                       lower_bound_gain=float(best_lbg if math.isfinite(best_lbg) else 0.0), u_diff=float(best_u),
                       reason=reason, candidates=records, assumptions=self.assumptions())
        self.history.append(dec)
        return dec


# ---------------------------------------------------------------------------
# policies (same engine, catalog and calibration; only the rule differs)
# ---------------------------------------------------------------------------
class Policy:
    """``decide(state, candidates, env, epoch) -> Decision``; ``observe(frame)`` feeds telemetry."""
    name: str = "policy"

    def decide(self, state: SessionState, candidates: Sequence[Candidate], env: Any, epoch: int) -> Decision:
        raise NotImplementedError

    def observe(self, frame: Any) -> None:
        return None


def _records(candidates: Sequence[Candidate], H: int, include_install: bool, per_step: bool = True) -> List[dict]:
    out = []
    for c in candidates:
        q = (horizon_exec(c, H) if per_step else H * c.exec_s_per_epoch) + (c.c_install_s if include_install else 0.0)
        out.append({"plan": dict(c.plan), "is_keep": c.is_keep, "feasible": c.feasible, "reason": c.reason,
                    "q_mean": q, "q_var": 0.0, "install_objects": list(c.install_objects),
                    "c_install_s": c.c_install_s})
    return out


class FixedPolicy(Policy):
    """B0: a fixed plan for the whole session (installs it at the first epoch if needed)."""

    def __init__(self, plan: Mapping[str, Any], name: str = "B0_fixed"):
        self.plan = dict(plan)
        self.name = name

    def decide(self, state, candidates, env, epoch) -> Decision:
        keep = _keep_of(state, candidates)
        H = max(0, int(state.remaining_epochs))
        target = next((c for c in candidates if c.plan == self.plan), None)
        if target is None:
            raise ValueError(f"POLICY_PLAN_UNKNOWN: fixed plan {self.plan} is not among the candidates")
        switched = target is not keep and target.plan != keep.plan
        q_keep = horizon_exec(keep, H) + keep.c_install_s
        q_best = horizon_exec(target, H) + target.c_install_s
        return Decision(dict(target.plan), switched, q_keep, q_best, q_keep - q_best, 0.0,
                        "FIXED_PLAN", _records(candidates, H, True),
                        ["fixed plan chosen before the session; no adaptation"])


class NaivePolicy(Policy):
    """B1: per-epoch cheapest execution, install ignored; ``delay`` k gives delayed:k.

    The rule and every reported figure use the current epoch's cost alone
    (``exec_s_per_epoch``, extrapolated as H times that value).
    """

    def __init__(self, delay: int = 0, name: str = "B1_naive"):
        self.delay = int(delay)
        self.name = name
        self._streak_key: Optional[str] = None
        self._streak = 0

    def decide(self, state, candidates, env, epoch) -> Decision:
        keep = _keep_of(state, candidates)
        H = max(0, int(state.remaining_epochs))
        feas = [c for c in candidates if c.feasible] or [keep]
        best = min(feas, key=lambda c: c.exec_s_per_epoch)
        key = plan_key(best.plan)
        self._streak = self._streak + 1 if key == self._streak_key else 1
        self._streak_key = key
        q_keep = H * keep.exec_s_per_epoch
        q_best = H * best.exec_s_per_epoch
        assumptions = ["install cost ignored", "per-epoch cheapest execution", "no uncertainty, no hysteresis"]
        if best is keep or best.plan == keep.plan:
            return Decision(dict(keep.plan), False, q_keep, q_best, 0.0, 0.0, "ALREADY_CHEAPEST",
                            _records(candidates, H, False, False), assumptions)
        if self._streak <= self.delay:
            return Decision(dict(keep.plan), False, q_keep, q_best, q_keep - q_best, 0.0,
                            f"DELAYED_HOLD: {self._streak}/{self.delay}", _records(candidates, H, False, False),
                            assumptions + [f"delayed replan k={self.delay}"])
        return Decision(dict(best.plan), True, q_keep, q_best, q_keep - q_best, 0.0,
                        "NAIVE_PER_EPOCH_BEST" if self.delay == 0 else f"DELAYED_REPLAN: k={self.delay}",
                        _records(candidates, H, False, False), assumptions)


class RemainingPlusInstallPolicy(Policy):
    """B2: minimise the execution time of the remaining H steps plus the candidate-specific install.

    The horizon term is :func:`horizon_exec` (the per-step sum, or H * exec on
    one shape); no uncertainty, no hysteresis, no survival weights.
    """

    name = "B2_remaining_plus_install"

    def decide(self, state, candidates, env, epoch) -> Decision:
        keep = _keep_of(state, candidates)
        H = max(0, int(state.remaining_epochs))
        q_keep = horizon_exec(keep, H) + keep.c_install_s
        best, best_q = keep, q_keep
        for c in candidates:
            if c is keep or not c.feasible:
                continue
            q = horizon_exec(c, H) + c.c_install_s
            if q < best_q:
                best, best_q = c, q
        switched = best is not keep
        return Decision(dict(best.plan), switched, q_keep, best_q, q_keep - best_q, 0.0,
                        "REMAINING_PLUS_INSTALL_MIN" if switched else "KEEP_IS_MIN",
                        _records(candidates, H, True),
                        ["costs within declared intervals", "fixed remaining horizon H, no survival weights",
                         "install cost candidate-specific", "no uncertainty, no hysteresis"])


class PlannerPlusB2Policy(Policy):
    """B3: a static planner picks the per-epoch optimal assignment, B2 control decides the switch.

    The planner is ExactAssign (or ILPAdapter) over a chain with one operator
    per plan axis whose realization costs are the axis-separable execution
    costs of the candidates; because the per-epoch cost of a plan is the sum
    of its axis costs, the planner's optimum equals the candidate with the
    least ``exec_s_per_epoch`` (verified in tests).  The controller then
    applies the B2 rule between keep and that single challenger, with the
    horizon term of B2 (:func:`horizon_exec`); the static assignment itself
    stays the optimum of the current epoch.
    """

    name = "B3_planner_plus_B2"

    def __init__(self, planner: Any = None):
        self.planner = planner

    def _planner_choice(self, candidates: Sequence[Candidate]) -> Candidate:
        feas = [c for c in candidates if c.feasible]
        if not feas:
            return _keep_of_list(candidates)
        if self.planner is None:
            return min(feas, key=lambda c: c.exec_s_per_epoch)
        chain = [{"id": "plan", "realizations": [
            {"name": plan_key(c.plan), "in_domain": "A64", "out_domain": "A64", "cost": c.exec_s_per_epoch}
            for c in feas]}]
        assignment, _ = self.planner.plan(chain, {"start_domain": "A64", "conversions": {}}, None, None, None)
        chosen = assignment.get("plan")
        for c in feas:
            if plan_key(c.plan) == chosen:
                return c
        return min(feas, key=lambda c: c.exec_s_per_epoch)

    def decide(self, state, candidates, env, epoch) -> Decision:
        keep = _keep_of(state, candidates)
        H = max(0, int(state.remaining_epochs))
        target = self._planner_choice(candidates)
        q_keep = horizon_exec(keep, H) + keep.c_install_s
        q_t = horizon_exec(target, H) + target.c_install_s
        switched = target is not keep and target.plan != keep.plan and q_t < q_keep
        chosen = target if switched else keep
        return Decision(dict(chosen.plan), switched, q_keep, q_t if switched else q_keep, q_keep - q_t, 0.0,
                        "PLANNER_ASSIGNMENT_B2_SWITCH" if switched else "PLANNER_ASSIGNMENT_B2_KEEP",
                        _records(candidates, H, True),
                        ["static planner assignment on per-epoch cost", "B2 control: remaining execution + install",
                         "no uncertainty, no hysteresis"])


def _keep_of_list(candidates: Sequence[Candidate]) -> Candidate:
    for c in candidates:
        if c.is_keep:
            return c
    return candidates[0]


# ---------------------------------------------------------------------------
# planner hook: chain built from the candidate list
# ---------------------------------------------------------------------------
def _horizon_cost(c: Candidate, horizon: Any) -> float:
    """Execution over the horizon plus the waiting cost: Qhat without the install term.

    ``horizon`` is the survival vector (the per-step sum of :func:`q_hat`) or
    a number, which prices that many steps on the current epoch's shape.
    """
    if isinstance(horizon, (int, float)):
        return float(horizon) * float(c.exec_s_per_epoch) + float(getattr(c, "c_wait_s", 0.0))
    return q_hat(c, None, horizon, include_install=False)[0]


def _object_costs(cands: Sequence[Candidate]) -> Optional[Dict[str, float]]:
    """One install cost per object, or None when the candidates disagree or do not itemize."""
    costs: Dict[str, float] = {}
    for c in cands:
        objs = list(c.install_objects)
        by_obj = dict(getattr(c, "install_cost_by_object", {}) or {})
        if objs and not by_obj:
            if len(objs) != 1:
                return None
            by_obj = {objs[0]: float(c.c_install_s)}
        if set(by_obj) != set(objs):
            return None
        for o, v in by_obj.items():
            v = float(v)
            if o in costs and not math.isclose(costs[o], v, rel_tol=1e-9, abs_tol=1e-12):
                return None
            costs.setdefault(o, v)
    return costs


def _single_chain(cands: Sequence[Candidate], horizon: Any, include_install: bool) -> Tuple[list, dict, Callable]:
    feas = [c for c in cands if c.feasible]
    by_name = {plan_key(c.plan): dict(c.plan) for c in feas}
    chain = [{"id": "plan", "realizations": [
        {"name": plan_key(c.plan), "in_domain": "A64", "out_domain": "A64",
         "cost": _horizon_cost(c, horizon) + (float(c.c_install_s) if include_install else 0.0)} for c in feas]}]
    return chain, {"start_domain": "A64", "conversions": {}}, lambda assignment: dict(by_name[assignment["plan"]])


def _axis_chain(cands: Sequence[Candidate], horizon: Any, include_install: bool
                ) -> Optional[Tuple[list, dict, Callable]]:
    """One operator per plan axis, or None when the candidates are not an additive full product."""
    axes = list(cands[0].plan.keys())
    if any(list(c.plan.keys()) != axes and set(c.plan.keys()) != set(axes) for c in cands):
        return None
    values: Dict[str, list] = {a: [] for a in axes}
    for c in cands:
        for a in axes:
            if c.plan[a] not in values[a]:
                values[a].append(c.plan[a])
    n_prod = 1
    for a in axes:
        n_prod *= len(values[a])
    keys = {tuple(c.plan[a] for a in axes) for c in cands}
    if len(cands) != n_prod or len(keys) != n_prod:
        return None
    if set(keys) != set(itertools.product(*[values[a] for a in axes])):
        return None
    # additive fit of the horizon cost: y(p) = sum over axes of w[axis][value]
    offsets: Dict[str, int] = {}
    col = 0
    for a in axes:
        offsets[a] = col
        col += len(values[a])
    X = np.zeros((len(cands), col), dtype=np.float64)
    y = np.zeros(len(cands), dtype=np.float64)
    for i, c in enumerate(cands):
        for a in axes:
            X[i, offsets[a] + values[a].index(c.plan[a])] = 1.0
        y[i] = _horizon_cost(c, horizon)
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    scale = max(1.0, float(np.max(np.abs(y))))
    if float(np.max(np.abs(X @ w - y))) > 1e-8 * scale:
        return None
    # install objects: attached to an axis value when every candidate with that value needs the object
    attached: Dict[Tuple[str, Any], List[str]] = {(a, v): [] for a in axes for v in values[a]}
    costs: Dict[str, float] = {}
    if include_install:
        got = _object_costs(cands)
        if got is None:
            return None
        costs = got
        for o in costs:
            for a in axes:
                for v in values[a]:
                    if all(o in c.install_objects for c in cands if c.plan[a] == v):
                        attached[(a, v)].append(o)
        for c in cands:
            union: set = set()
            for a in axes:
                union.update(attached[(a, c.plan[a])])
            if union != set(c.install_objects):
                return None
            total = sum(costs[o] for o in union)
            if not math.isclose(total, float(c.c_install_s), rel_tol=1e-6, abs_tol=1e-9):
                return None
    chain = []
    name_to_value: Dict[str, Dict[str, Any]] = {}
    const = 0.0
    shifted: Dict[Tuple[str, Any], float] = {}
    for a in axes:
        ws = [float(w[offsets[a] + j]) for j in range(len(values[a]))]
        lo = min(ws)
        const += lo
        for v, wv in zip(values[a], ws):
            shifted[(a, v)] = wv - lo
    for ai, a in enumerate(axes):
        reals = []
        name_to_value[a] = {}
        for j, v in enumerate(values[a]):
            name = f"{j}:{v}"
            name_to_value[a][name] = v
            reals.append({"name": name, "in_domain": "A64", "out_domain": "A64",
                          "cost": shifted[(a, v)] + (const if ai == 0 else 0.0),
                          "objects": [{"id": o, "install_cost": costs[o]} for o in attached[(a, v)]]})
        chain.append({"id": str(a), "realizations": reals})
    return (chain, {"start_domain": "A64", "conversions": {}},
            lambda assignment: {a: name_to_value[a][assignment[str(a)]] for a in axes})


def candidates_chain(state: SessionState, candidates: Sequence[Candidate], horizon: Any,
                     include_install: bool = True) -> Tuple[list, dict, Callable]:
    """Planner chain, catalog and assignment decoder built from the candidate list alone.

    ``horizon`` is the survival vector, which gives the horizon cost
    ``sum_j s_j * exec_j + c_wait`` of :func:`q_hat`, or a number, which gives
    ``horizon * exec + c_wait`` on the current epoch's shape.  When every
    candidate is feasible, the candidates form the full product of
    their axis values, the horizon cost is additive
    over the axes (least-squares residual below 1e-8 relative) and every
    install object can be attached to axis values (an object is attached to
    an axis value when every candidate with that value needs it; the union
    over the values of a plan must reproduce its install objects and install
    cost), the chain has one operator per plan axis with ``objects`` priced
    once by the planner.  Otherwise the chain is the single operator ``plan``
    whose realizations are the feasible candidates with their full cost; every
    planner is exact on it.  Costs are seconds over the horizon.
    """
    cands = list(candidates)
    h = float(horizon) if isinstance(horizon, (int, float)) else [float(v) for v in horizon]
    if cands and all(c.feasible for c in cands):
        per_axis = _axis_chain(cands, h, bool(include_install))
        if per_axis is not None:
            return per_axis
    return _single_chain(cands, h, bool(include_install))


def planner_assignment(planner: Any, state: SessionState, candidates: Sequence[Candidate], horizon: Any,
                       env: Any = None, include_install: bool = True,
                       chain_builder: Optional[Callable] = None) -> Tuple[Candidate, dict]:
    """(candidate assigned by the planner for this epoch, record of how it was obtained).

    ``chain_builder(state, candidates, horizon, include_install)`` returns
    ``(chain, catalog, decode)`` with ``decode(assignment) -> plan``; the
    default is :func:`candidates_chain`, and ``horizon`` is the survival
    vector or a number as described there.  Keep is returned when no candidate
    is feasible, when the planner finds no assignment or when the assignment
    is not a feasible candidate (the record names the fallback code).
    """
    keep = _keep_of(state, candidates)
    info: Dict[str, Any] = {"planner": str(getattr(planner, "name", type(planner).__name__)), "chain_ops": 0,
                            "cost": None, "plan": dict(keep.plan), "fallback": ""}
    if not any(c.feasible for c in candidates):
        info["fallback"] = "NO_FEASIBLE_CANDIDATE"
        return keep, info
    builder = chain_builder if chain_builder is not None else candidates_chain
    chain, pcat, decode = builder(state, candidates, horizon, include_install)
    info["chain_ops"] = len(chain)
    assignment, cost = planner.plan(chain, pcat, env, None, state)
    if not assignment or not math.isfinite(float(cost)):
        info["fallback"] = "PLANNER_NO_ASSIGNMENT"
        return keep, info
    plan = decode(assignment)
    pick = next((c for c in candidates if c.plan == plan), None)
    if pick is None or not pick.feasible:
        info["fallback"] = "ASSIGNMENT_NOT_FEASIBLE"
        return keep, info
    info["cost"] = float(cost)
    info["plan"] = dict(pick.plan)
    return pick, info


class ControllerPolicy(Policy):
    """B4 and its ablations: the StateAwareController behind the Policy interface.

    With ``planner`` (ExactAssign, ILPAdapter or GreedyPerOperator) the
    assignment of the epoch comes from the planner: it minimises the
    survival-weighted horizon cost plus missing-object installs on the chain
    of :func:`candidates_chain` (or of ``chain_builder``), and the decision rule
    then decides between keep and that single assignment with its own
    uncertainty, hysteresis and residency terms.  Without a planner the
    controller ranks every candidate itself.  ``planner_log`` keeps one
    record per decision (planner name, chain size, planner cost, plan,
    fallback code).
    """

    def __init__(self, cfg: ControllerConfig, name: str = "B4_opencrab", planner: Any = None,
                 chain_builder: Optional[Callable] = None):
        self.ctrl = StateAwareController(cfg)
        self.name = name
        self.planner = planner
        self.chain_builder = chain_builder
        self.planner_log: List[dict] = []

    def observe(self, frame: Any) -> None:
        self.ctrl.observe(frame)

    def decide(self, state, candidates, env, epoch) -> Decision:
        if self.planner is None:
            return self.ctrl.decide(state, candidates, epoch)
        keep = _keep_of(state, candidates)
        s = self.ctrl.survival(max(0, int(state.remaining_epochs)))
        pick, info = planner_assignment(self.planner, state, candidates, s, env,
                                        not self.ctrl.cfg.ignore_install, self.chain_builder)
        self.planner_log.append(info)
        restricted = [keep] if (pick is keep or pick.plan == keep.plan) else [keep, pick]
        dec = self.ctrl.decide_with_survival(state, restricted, epoch, s)
        for rec in dec.candidates:
            rec["planner_pick"] = bool(rec["plan"] == pick.plan)
        dec.assumptions = list(dec.assumptions) + [
            f"assignment of the epoch chosen by planner {info['planner']} on a chain of {info['chain_ops']} "
            "operator(s) built from the candidates; the decision rule decides between keep and that assignment"]
        return dec


class OraclePolicy(Policy):
    """B5: future-aware oracle; reads the exogenous trace, pays the real install.

    ``trace`` maps epoch -> Env (list or dict).  ``costmodel`` prices every
    candidate at every future epoch on the public shape of that epoch:
    ``schedule`` lists the shapes of the remaining epochs (entry j for epoch
    t + j; the runner refreshes the attribute before each decision, and a
    step beyond the list takes its last entry).  Without a schedule every
    step has ``shapes`` and ``kv_growth`` adds one key per epoch to T.  Not
    executable without the trace (an upper bound on what any causal policy
    can achieve, not a baseline system).
    """

    name = "B5_oracle"

    def __init__(self, trace: Any, costmodel: Any, shapes: Mapping[str, Any], kv_growth: bool = True,
                 schedule: Optional[Sequence[Mapping[str, Any]]] = None):
        self.trace = trace
        self.costmodel = costmodel
        self.shapes = dict(shapes)
        self.kv_growth = bool(kv_growth)
        self.schedule = list(schedule) if schedule else None

    def _env_at(self, j: int) -> Any:
        if isinstance(self.trace, Mapping):
            keys = sorted(self.trace)
            return self.trace[min(keys, key=lambda k: (k > j, abs(k - j)))]
        return self.trace[min(j, len(self.trace) - 1)]

    def decide(self, state, candidates, env, epoch) -> Decision:
        keep = _keep_of(state, candidates)
        H = max(0, int(state.remaining_epochs))
        sched = self.schedule
        totals: Dict[str, float] = {}
        for c in candidates:
            if not c.feasible and c is not keep:
                continue
            tot = c.c_install_s
            for j in range(H):
                if sched:
                    sh = sched[min(j, len(sched) - 1)]
                else:
                    sh = dict(self.shapes)
                    if self.kv_growth:
                        sh["T"] = int(sh["T"]) + j
                tot += exec_cost(c.plan, sh, self.costmodel, self._env_at(epoch + j))[0]
            totals[plan_key(c.plan)] = tot
        q_keep = totals[plan_key(keep.plan)]
        best = keep
        for c in candidates:
            k = plan_key(c.plan)
            if k in totals and totals[k] < totals[plan_key(best.plan)]:
                best = c
        switched = best is not keep
        return Decision(dict(best.plan), switched, q_keep, totals[plan_key(best.plan)],
                        q_keep - totals[plan_key(best.plan)], 0.0,
                        "ORACLE_FUTURE_TRACE", _records(candidates, H, True),
                        ["oracle: reads the future trace", "pays the real install", "not executable causally"])


class ZeroOverheadOracle(Policy):
    """B6: per-epoch cheapest plan with install and switching cost set to zero (lower bound only).

    The choice is the cheapest plan of the current epoch (a switch is free at
    every epoch); the reported horizon figures are :func:`horizon_exec`.
    """

    name = "B6_zero_overhead"

    def decide(self, state, candidates, env, epoch) -> Decision:
        keep = _keep_of(state, candidates)
        H = max(0, int(state.remaining_epochs))
        feas = [c for c in candidates if c.feasible] or [keep]
        best = min(feas, key=lambda c: c.exec_s_per_epoch)
        switched = best is not keep and best.plan != keep.plan
        q_keep, q_best = horizon_exec(keep, H), horizon_exec(best, H)
        return Decision(dict(best.plan), switched, q_keep, q_best, q_keep - q_best, 0.0, "ZERO_OVERHEAD_ORACLE",
                        _records(candidates, H, False),
                        ["oracle: install and switching cost set to zero", "not executable"])


POLICY_NAMES: Tuple[str, ...] = (
    "B0_fixed:<plan-json>", "B1_naive", "B2_remaining_plus_install", "B3_planner_plus_B2",
    "B4_opencrab", "B4_no_install", "B4_no_hyst", "B4_no_unc", "delayed:<k>", "B5_oracle", "B6_zero_overhead",
)


def make_policy(name: str, **kw: Any) -> Policy:
    """Policy factory.

    ``kw``: ControllerConfig overrides for the B4 variants plus the optional
    ``planner`` (ExactAssign, ILPAdapter, GreedyPerOperator) and
    ``chain_builder`` of :class:`ControllerPolicy`; ``planner`` for B3;
    ``plan`` for B0; ``trace``, ``costmodel``, ``shapes`` and the optional
    ``schedule`` for B5.
    """
    if name.startswith("B0_fixed:"):
        plan = json.loads(name[len("B0_fixed:"):])
        return FixedPolicy(plan, name)
    if name in ("B0_fixed", "B0_fixed_best_hindsight"):
        if "plan" not in kw:
            raise ValueError(f"POLICY_PARAM: {name} needs plan=<dict> (best hindsight plan is chosen by the runner)")
        return FixedPolicy(kw["plan"], name)
    if name == "B1_naive":
        return NaivePolicy(0, name)
    if name.startswith("delayed:"):
        return NaivePolicy(int(name.split(":", 1)[1]), name)
    if name == "B2_remaining_plus_install":
        return RemainingPlusInstallPolicy()
    if name == "B3_planner_plus_B2":
        return PlannerPlusB2Policy(kw.get("planner"))
    if name in ("B4_opencrab", "B4_no_install", "B4_no_hyst", "B4_no_unc"):
        overrides = {k: v for k, v in kw.items() if k in ControllerConfig.__dataclass_fields__}
        if name == "B4_no_install":
            overrides["ignore_install"] = True
        if name == "B4_no_hyst":
            overrides.setdefault("hysteresis_s", 0.0)
            overrides.setdefault("min_residency", 0)
        if name == "B4_no_unc":
            overrides["use_uncertainty"] = False
        return ControllerPolicy(ControllerConfig(**overrides), name, planner=kw.get("planner"),
                                chain_builder=kw.get("chain_builder"))
    if name == "B5_oracle":
        for k in ("trace", "costmodel", "shapes"):
            if k not in kw:
                raise ValueError(f"POLICY_PARAM: B5_oracle needs {k}=...")
        return OraclePolicy(kw["trace"], kw["costmodel"], kw["shapes"], kw.get("kv_growth", True), kw.get("schedule"))
    if name == "B6_zero_overhead":
        return ZeroOverheadOracle()
    raise ValueError(f"POLICY_UNKNOWN: {name!r}; known: {list(POLICY_NAMES)}")


__all__ = [
    "ASSUMPTIONS_B4", "RESIDENCY_CAP", "ControllerConfig", "Decision", "DwellModel", "q_hat", "horizon_exec",
    "StateAwareController",
    "candidates_chain", "planner_assignment",
    "Policy", "FixedPolicy", "NaivePolicy", "RemainingPlusInstallPolicy", "PlannerPlusB2Policy",
    "ControllerPolicy", "OraclePolicy", "ZeroOverheadOracle", "POLICY_NAMES", "make_policy",
]
