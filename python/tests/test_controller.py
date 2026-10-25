"""Tests for opencrab2.highlevel.controller: the complete objective against steady-state-first, a stationary
environment, the listed assumptions, exhaustive brute force, policies, dwell model, the per-step objective
and, with the calibrated cost model, the first-step choice over sensed bandwidths."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest

from opencrab2.highlevel import candidates as cd
from opencrab2.highlevel import controller as ct
from opencrab2.highlevel import costmodel as cm

PLAN_KEEP = {"embedding": "generic", "linear": "generic", "argmax_k": 2, "max_k": 2}
PLAN_A = {"embedding": "wmask", "linear": "wmask", "argmax_k": 2, "max_k": 2}
PLAN_B = {"embedding": "generic", "linear": "generic", "argmax_k": 8, "max_k": 4}
ENV = cm.Env(100e6, 0.01)


def cand(plan, exec_s, install_s=0.0, exec_sigma=0.0, install_sigma=0.0, is_keep=False, feasible=True,
         objects=None):
    return cd.Candidate(plan=dict(plan), install_objects=list(objects or ([] if install_s == 0 else ["obj"])),
                        c_install_s=float(install_s), c_install_sigma=float(install_sigma),
                        exec_s_per_epoch=float(exec_s), exec_sigma=float(exec_sigma), material_bytes_per_epoch=0,
                        mem_peak_bytes=0, feasible=feasible, reason="OK" if feasible else "INSUFFICIENT_MATERIAL: test",
                        is_keep=is_keep)


def state(incumbent=PLAN_KEEP, remaining=5):
    return cd.SessionState(dict(incumbent), {}, {}, 32, remaining, "s", "gpt2@v1", "nc")


def cfg_exact(**kw):
    base = dict(hysteresis_s=0.0, z=1.0, min_residency=0, dwell_prior=math.inf, env_change_rel=0.25)
    base.update(kw)
    return ct.ControllerConfig(**base)


def frame(bw, rtt, epoch=0):
    return {"bandwidth_bps": bw, "rtt_s": rtt, "epoch": epoch}


# ---------------------------------------------------------------------------
# the complete objective beats steady-state-first
# ---------------------------------------------------------------------------
def install_example_candidates():
    return [cand(PLAN_KEEP, 10.0, 0.0, is_keep=True), cand(PLAN_A, 1.0, 100.0), cand(PLAN_B, 4.0, 1.0)]


def old_rule(cands, H):
    """The original rule: pick the steady-state cheapest challenger, then test its install against the saving."""
    keep = cands[0]
    challenger = min(cands[1:], key=lambda c: c.exec_s_per_epoch)
    saving = H * (keep.exec_s_per_epoch - challenger.exec_s_per_epoch)
    return challenger.plan if saving > challenger.c_install_s else keep.plan


def test_install_example_complete_objective_selects_b_old_rule_keeps():
    cands = install_example_candidates()
    ctrl = ct.StateAwareController(cfg_exact())
    dec = ctrl.decide(state(remaining=5), cands, epoch=0)
    assert dec.switched and dec.chosen == PLAN_B
    assert math.isclose(dec.q_keep, 50.0)
    q = {cd.plan_key(r["plan"]): r["q_mean"] for r in dec.candidates}
    assert math.isclose(q[cd.plan_key(PLAN_A)], 105.0)
    assert math.isclose(q[cd.plan_key(PLAN_B)], 21.0)
    assert math.isclose(dec.q_best, 21.0) and math.isclose(dec.lower_bound_gain, 29.0)
    assert dec.u_diff == 0.0
    assert old_rule(cands, 5) == PLAN_KEEP          # steady-state-first stays with keep (50 < 105)


def test_install_example_holds_under_default_survival_prior():
    dec = ct.StateAwareController(ct.ControllerConfig(hysteresis_s=0.0, min_residency=0)).decide(
        state(remaining=5), install_example_candidates(), 0)
    assert dec.chosen == PLAN_B and dec.switched
    assert dec.q_keep < 50.0                         # survival weights discount the horizon


# ---------------------------------------------------------------------------
# stationary environment, cold start installs once
# ---------------------------------------------------------------------------
def simulate_stationary(H, install, exec_keep=10.0, exec_w=2.0, cfg=None):
    ctrl = ct.StateAwareController(cfg or cfg_exact())
    incumbent = dict(PLAN_KEEP)
    installed = set()
    switches = []
    for epoch in range(H):
        ctrl.observe(frame(100e6, 0.02, epoch))
        st = state(incumbent, remaining=H - epoch)
        keep_is_w = incumbent == PLAN_A
        c_keep = cand(incumbent, exec_w if keep_is_w else exec_keep, 0.0, is_keep=True)
        other_plan = PLAN_KEEP if keep_is_w else PLAN_A
        other_install = 0.0 if (other_plan == PLAN_KEEP or "A" in installed) else install
        c_other = cand(other_plan, exec_keep if keep_is_w else exec_w, other_install)
        dec = ctrl.decide(st, [c_keep, c_other], epoch)
        if dec.switched:
            switches.append(epoch)
            incumbent = dict(dec.chosen)
            if incumbent == PLAN_A:
                installed.add("A")
    return switches


def test_stationary_cold_incumbent_installs_once_when_amortised():
    assert simulate_stationary(H=8, install=30.0) == [0]          # 80 vs 30 + 16: install at epoch 0, then stay
    assert simulate_stationary(H=2, install=30.0) == []           # 20 vs 34: never worth it


def test_stationary_already_optimal_incumbent_never_switches():
    ctrl = ct.StateAwareController(cfg_exact())
    for epoch in range(6):
        ctrl.observe(frame(100e6, 0.02, epoch))
        dec = ctrl.decide(state(PLAN_A, 6 - epoch), [cand(PLAN_A, 2.0, 0.0, is_keep=True), cand(PLAN_KEEP, 10.0, 0.0)], epoch)
        assert not dec.switched and dec.reason.startswith("KEEP_IS_ARGMAX")
        assert dec.chosen == PLAN_A and math.isclose(dec.q_best, dec.q_keep)


# ---------------------------------------------------------------------------
# exhaustive brute force: 3 candidates x 5 epochs x 20 random cost tables
# ---------------------------------------------------------------------------
def sequence_cost(seq, exec_s, install_s):
    """Total cost of a switching sequence; an install is paid once per plan when first entered."""
    total, installed = 0.0, {0}
    for p in seq:
        if p not in installed:
            total += install_s[p]
            installed.add(p)
        total += exec_s[p]
    return total


@pytest.mark.parametrize("seed", range(20))
def test_b4_zero_uncertainty_matches_exhaustive_optimum(seed):
    rng = np.random.default_rng(seed)
    H = 5
    exec_s = rng.uniform(1.0, 10.0, size=3)
    install_s = np.array([0.0, rng.uniform(0.0, 30.0), rng.uniform(0.0, 30.0)])
    plans = [PLAN_KEEP, PLAN_A, PLAN_B]
    cands = [cand(plans[i], exec_s[i], install_s[i], is_keep=(i == 0)) for i in range(3)]
    best_seq = min(itertools.product(range(3), repeat=H), key=lambda s: sequence_cost(s, exec_s, install_s))
    best_total = sequence_cost(best_seq, exec_s, install_s)
    dec = ct.StateAwareController(cfg_exact()).decide(state(remaining=H), cands, 0)
    chosen = plans.index(dec.chosen)
    assert math.isclose(install_s[chosen] + H * exec_s[chosen], best_total, rel_tol=1e-12)
    assert math.isclose(dec.q_best, best_total, rel_tol=1e-12)
    # B2 equals B4 when uncertainty and hysteresis are off
    b2 = ct.make_policy("B2_remaining_plus_install").decide(state(remaining=H), cands, ENV, 0)
    assert b2.chosen == dec.chosen and b2.switched == dec.switched
    assert math.isclose(b2.q_best, dec.q_best, rel_tol=1e-12)
    b4u = ct.make_policy("B4_no_unc", hysteresis_s=0.0, min_residency=0, dwell_prior=math.inf).decide(
        state(remaining=H), cands, ENV, 0)
    assert b4u.chosen == dec.chosen


# ---------------------------------------------------------------------------
# uncertainty, hysteresis, residency
# ---------------------------------------------------------------------------
def test_uncertainty_sums_variance_over_horizon():
    keep = cand(PLAN_KEEP, 10.0, 0.0, exec_sigma=1.0, is_keep=True)
    other = cand(PLAN_A, 9.0, 0.0, exec_sigma=2.0, install_sigma=3.0)
    s = [1.0] * 5
    m_k, v_k = ct.q_hat(keep, state(), s)
    m_o, v_o = ct.q_hat(other, state(), s)
    assert math.isclose(m_k, 50.0) and math.isclose(v_k, 5.0)
    assert math.isclose(m_o, 45.0) and math.isclose(v_o, 5 * 4.0 + 9.0)
    dec = ct.StateAwareController(cfg_exact(z=1.0)).decide(state(), [keep, other], 0)
    rec = next(r for r in dec.candidates if not r["is_keep"])
    assert math.isclose(rec["u_diff"], math.sqrt(5.0 + 29.0))
    assert math.isclose(rec["lower_bound_gain"], 5.0 - math.sqrt(34.0))
    assert not dec.switched and dec.reason.startswith("KEEP_IS_ARGMAX")   # 5 - 5.83 < 0: uncertain gain, keep
    assert dec.lower_bound_gain == 0.0 and dec.u_diff == 0.0 and math.isclose(dec.q_best, dec.q_keep)
    # with a larger expected gain the challenger is p* and the decision carries its uncertainty
    other2 = cand(PLAN_A, 2.0, 0.0, exec_sigma=2.0, install_sigma=3.0)
    dec2 = ct.StateAwareController(cfg_exact(z=1.0)).decide(state(), [keep, other2], 0)
    assert dec2.switched and math.isclose(dec2.u_diff, math.sqrt(34.0))
    assert math.isclose(dec2.lower_bound_gain, 40.0 - math.sqrt(34.0))
    dec2 = ct.StateAwareController(cfg_exact(use_uncertainty=False)).decide(state(), [keep, other], 0)
    assert dec2.switched and dec2.u_diff == 0.0
    # survival weights enter the variance squared
    m_s, v_s = ct.q_hat(other, state(), [1.0, 0.5], include_install=False)
    assert math.isclose(m_s, 1.5 * 9.0) and math.isclose(v_s, (1.0 + 0.25) * 4.0)


def test_hysteresis_blocks_small_gains():
    keep = cand(PLAN_KEEP, 10.0, 0.0, is_keep=True)
    other = cand(PLAN_A, 9.9, 0.1)                           # gain 5 * 0.1 - 0.1 = 0.4
    assert not ct.StateAwareController(cfg_exact(hysteresis_s=0.5)).decide(state(), [keep, other], 0).switched
    assert ct.StateAwareController(cfg_exact(hysteresis_s=0.0)).decide(state(), [keep, other], 0).switched


def test_min_residency_holds_after_a_switch():
    """Counter rule shared with the C++ controller: replaceable at the first epoch t >= t0 + min_residency."""
    ctrl = ct.StateAwareController(cfg_exact(min_residency=2))
    assert ctrl.epochs_since_switch == ct.RESIDENCY_CAP
    first = ctrl.decide(state(remaining=5), [cand(PLAN_KEEP, 10.0, is_keep=True), cand(PLAN_A, 1.0, 1.0)], 0)
    assert first.switched and ctrl.epochs_since_switch == 1      # the switching epoch counts as one
    second = ctrl.decide(state(PLAN_A, 4), [cand(PLAN_A, 1.0, is_keep=True), cand(PLAN_B, 0.1, 0.0)], 1)
    assert not second.switched and second.reason.startswith("MIN_RESIDENCY_HOLD")
    assert ctrl.epochs_since_switch == 2
    third = ctrl.decide(state(PLAN_A, 3), [cand(PLAN_A, 1.0, is_keep=True), cand(PLAN_B, 0.1, 0.0)], 2)
    assert third.switched                                         # epoch 2 = t0 + min_residency
    assert ctrl.epochs_since_switch == 1


@pytest.mark.parametrize("min_residency,expected", [(0, [0, 1, 2, 3, 4, 5]), (1, [0, 1, 2, 3, 4, 5]),
                                                    (2, [0, 2, 4]), (3, [0, 3])])
def test_min_residency_counter_rule(min_residency, expected):
    plans = [PLAN_KEEP, PLAN_A, PLAN_B]
    ctrl = ct.StateAwareController(cfg_exact(min_residency=min_residency))
    cur = 0
    switches = []
    for epoch in range(6):
        nxt = (cur + 1) % 3
        dec = ctrl.decide(state(plans[cur], 10 - epoch),
                          [cand(plans[cur], 5.0, is_keep=True), cand(plans[nxt], 1.0, 0.0)], epoch)
        if dec.switched:
            switches.append(epoch)
            cur = nxt
    assert switches == expected


def test_default_min_residency_allows_a_switch_at_the_next_epoch():
    ctrl = ct.StateAwareController(ct.ControllerConfig(hysteresis_s=0.0, stationary_survival=True))
    assert ctrl.cfg.min_residency == 1
    a = ctrl.decide(state(remaining=5), [cand(PLAN_KEEP, 10.0, is_keep=True), cand(PLAN_A, 1.0, 1.0)], 0)
    b = ctrl.decide(state(PLAN_A, 4), [cand(PLAN_A, 1.0, is_keep=True), cand(PLAN_B, 0.1, 0.0)], 1)
    assert a.switched and b.switched


def test_ignore_install_ablation_changes_decision():
    cands = [cand(PLAN_KEEP, 10.0, is_keep=True), cand(PLAN_A, 9.0, 100.0)]
    with_install = ct.make_policy("B4_opencrab", hysteresis_s=0.0, min_residency=0, dwell_prior=math.inf)
    without = ct.make_policy("B4_no_install", hysteresis_s=0.0, min_residency=0, dwell_prior=math.inf)
    assert not with_install.decide(state(), cands, ENV, 0).switched
    assert without.decide(state(), cands, ENV, 0).switched


def test_infeasible_keep_switches_to_best_feasible():
    cands = [cand(PLAN_KEEP, 1.0, is_keep=True, feasible=False), cand(PLAN_A, 5.0, 10.0), cand(PLAN_B, 6.0, 0.0)]
    dec = ct.StateAwareController(cfg_exact()).decide(state(), cands, 0)
    assert dec.switched and dec.chosen == PLAN_B and dec.reason.startswith("KEEP_INFEASIBLE")
    only_keep = [cand(PLAN_KEEP, 1.0, is_keep=True, feasible=False), cand(PLAN_A, 5.0, feasible=False)]
    dec2 = ct.StateAwareController(cfg_exact()).decide(state(), only_keep, 0)
    assert not dec2.switched and dec2.reason.startswith("NO_FEASIBLE_CANDIDATE")
    infeasible_challenger = [cand(PLAN_KEEP, 10.0, is_keep=True), cand(PLAN_A, 1.0, feasible=False)]
    dec3 = ct.StateAwareController(cfg_exact()).decide(state(), infeasible_challenger, 0)
    assert not dec3.switched and dec3.reason.startswith("NO_CHALLENGER")


# ---------------------------------------------------------------------------
# dwell model (fitted on observed frames only)
# ---------------------------------------------------------------------------
def test_dwell_model_is_causal_and_fits_observed_changes():
    dm = ct.DwellModel(prior_mean_dwell=5.0, change_rel=0.25)
    assert math.isclose(dm.hazard(), 0.2)
    assert dm.survival(3) == [1.0, 0.8, 0.8 ** 2]
    for _ in range(10):
        dm.observe(100e6, 0.02)                              # ten frames are nine observed transitions
    assert dm.n_changes == 0 and dm.n_steps == 9 and math.isclose(dm.hazard(), 1.0 / (9 + 5.0))
    changed = dm.observe(10e6, 0.02)                         # ten-fold bandwidth drop: a regime change
    assert changed and dm.n_changes == 1 and dm.dwells == [10]
    h_after = dm.hazard()
    assert math.isclose(h_after, 2.0 / (10 + 5.0)) and h_after > 1.0 / 14.0
    s = dm.survival(5)
    assert all(s[i] > s[i + 1] for i in range(4)) and s[0] == 1.0
    assert ct.DwellModel(math.inf).survival(4) == [1.0, 1.0, 1.0, 1.0]
    with pytest.raises(ValueError, match="^CONTROLLER_PARAM"):
        ct.DwellModel(0.0)


def test_controller_observe_accepts_frames_and_shrinks_horizon():
    ctrl = ct.StateAwareController(ct.ControllerConfig(hysteresis_s=0.0, min_residency=0))
    for e in range(6):
        ctrl.observe(frame(100e6 if e % 2 == 0 else 10e6, 0.02, e))   # a change every epoch
    s = ctrl.survival(5)
    assert sum(s) < 5.0 and s[1] < 0.5
    cands = [cand(PLAN_KEEP, 10.0, is_keep=True), cand(PLAN_A, 1.0, 30.0)]
    dec = ctrl.decide(state(remaining=5), cands, 6)
    assert dec.q_keep < 50.0 and not dec.switched              # frequent changes make the install unattractive


# ---------------------------------------------------------------------------
# listed assumptions (no unconditional no-harm claim) and the policy family
# ---------------------------------------------------------------------------
def all_policies():
    return [ct.make_policy(n) for n in ("B1_naive", "B2_remaining_plus_install", "B3_planner_plus_B2",
                                          "B4_opencrab", "B4_no_install", "B4_no_hyst", "B4_no_unc", "delayed:2",
                                          "B6_zero_overhead")] + [ct.make_policy("B0_fixed:" + json.dumps(PLAN_A))]


def test_every_decision_carries_assumptions():
    cands = install_example_candidates()
    for pol in all_policies():
        dec = pol.decide(state(), cands, ENV, 0)
        assert isinstance(dec.assumptions, list) and len(dec.assumptions) > 0, pol.name
        assert isinstance(dec, ct.Decision) and len(dec.candidates) == 3
    b4 = ct.make_policy("B4_opencrab").decide(state(), cands, ENV, 0)
    assert "costs within declared intervals" in b4.assumptions
    assert any("survival" in a for a in b4.assumptions)
    assert "unconditional" not in " ".join(b4.assumptions)
    d = b4.to_dict()
    assert json.dumps(d)


def test_policy_rules_differ_as_specified():
    cands = install_example_candidates()
    st = state(remaining=5)
    b1 = ct.make_policy("B1_naive").decide(st, cands, ENV, 0)
    assert b1.switched and b1.chosen == PLAN_A                 # install ignored: A is the per-epoch cheapest
    b2 = ct.make_policy("B2_remaining_plus_install").decide(st, cands, ENV, 0)
    assert b2.chosen == PLAN_B and math.isclose(b2.q_best, 21.0)
    b3 = ct.make_policy("B3_planner_plus_B2").decide(st, cands, ENV, 0)
    assert not b3.switched and b3.chosen == PLAN_KEEP          # planner picks A (per epoch), B2 refuses 105 > 50
    b6 = ct.make_policy("B6_zero_overhead").decide(st, cands, ENV, 0)
    assert b6.chosen == PLAN_A and math.isclose(b6.q_best, 5.0)
    fixed = ct.make_policy("B0_fixed:" + json.dumps(PLAN_B)).decide(st, cands, ENV, 0)
    assert fixed.chosen == PLAN_B and fixed.switched
    delayed = ct.make_policy("delayed:2")
    assert not delayed.decide(st, cands, ENV, 0).switched
    assert not delayed.decide(st, cands, ENV, 1).switched
    assert delayed.decide(st, cands, ENV, 2).switched
    with pytest.raises(ValueError, match="^POLICY_UNKNOWN"):
        ct.make_policy("B9")
    with pytest.raises(ValueError, match="^POLICY_PARAM"):
        ct.make_policy("B5_oracle")


def test_b3_planner_adapter_matches_min_exec_candidate():
    from opencrab2.highlevel.planner import ExactAssign
    cands = install_example_candidates()
    pol = ct.PlannerPlusB2Policy(planner=ExactAssign())
    assert pol._planner_choice(cands).plan == PLAN_A
    dec = pol.decide(state(remaining=5), cands, ENV, 0)
    assert dec.chosen == PLAN_KEEP


def test_b5_oracle_reads_future_trace_and_pays_install():
    shapes = cd.gpt2_shapes(n_layers=1, n=1, T=8)
    cat = cd.gpt2_catalog(shapes)
    model = cm.FittedCostModel(label="MODELED")
    for comp in cm.COMPONENTS:
        model.fits[comp] = cm.ComponentFit(1e-9, 0.0, 0.001, 0.0, 1, "MODELED", "toy")
    st = cd.SessionState(dict(PLAN_KEEP), {}, {}, 8, 4, "s", "gpt2@v1", "nc")
    now = cm.Env(1e13, 1e-7)        # transport negligible now: the toy compute is lane independent, installs cost extra
    cands = cd.enumerate_candidates(st, cat, model, now, shapes)
    # the future is a very slow link: byte-heavy generic plans become expensive, wmask plans win
    trace = [now] + [cm.Env(1e6, 0.001)] * 10
    oracle = ct.make_policy("B5_oracle", trace=trace, costmodel=model, shapes=shapes)
    dec = oracle.decide(st, cands, now, 0)
    assert dec.chosen["linear"] == "wmask" and dec.switched
    causal = ct.make_policy("B4_opencrab", hysteresis_s=0.0, min_residency=0, dwell_prior=math.inf).decide(st, cands, now, 0)
    assert causal.chosen["linear"] == "generic"                # a causal policy cannot see the slowdown
    assert any("oracle" in a for a in dec.assumptions)



# ---------------------------------------------------------------------------
# epoch_corr, stationary_survival, c_wait (C++ defaults and semantics)
# ---------------------------------------------------------------------------
def test_config_defaults_and_validation():
    cfg = ct.ControllerConfig()
    assert cfg.epoch_corr == 0.0 and cfg.stationary_survival is False
    assert list(ct.ControllerConfig.__dataclass_fields__) == [
        "hysteresis_s", "z", "min_residency", "dwell_prior", "env_change_rel", "ignore_install", "use_uncertainty",
        "epoch_corr", "stationary_survival"]
    for bad in (dict(epoch_corr=-0.1), dict(epoch_corr=1.5), dict(hysteresis_s=-1.0), dict(z=-1.0),
                dict(min_residency=-1), dict(env_change_rel=float("nan"))):
        with pytest.raises(ValueError, match="^CONTROLLER_PARAM"):
            ct.StateAwareController(ct.ControllerConfig(**bad))
    assert json.dumps(cfg.to_dict())


def test_epoch_corr_interpolates_between_independent_and_common_residual():
    c = cand(PLAN_A, 9.0, 0.0, exec_sigma=2.0, install_sigma=3.0)
    s = [1.0, 0.5, 0.25]
    s1, s2 = 1.75, 1.0 + 0.25 + 0.0625
    m0, v0 = ct.q_hat(c, state(), s, epoch_corr=0.0)
    m1, v1 = ct.q_hat(c, state(), s, epoch_corr=1.0)
    mh, vh = ct.q_hat(c, state(), s, epoch_corr=0.5)
    assert m0 == m1 == mh == s1 * 9.0
    assert math.isclose(v0, 4.0 * s2 + 9.0)                    # independent epochs: sum of squares
    assert math.isclose(v1, 4.0 * s1 * s1 + 9.0)               # one common residual: square of the sum
    assert math.isclose(vh, 4.0 * (s2 + 0.5 * (s1 * s1 - s2)) + 9.0)
    assert v0 < vh < v1
    # the controller uses the declared share: a gain that clears independent noise is held under correlation
    keep = cand(PLAN_KEEP, 10.0, 0.0, exec_sigma=1.0, is_keep=True)
    other = cand(PLAN_A, 9.0, 0.0, exec_sigma=1.0)
    indep = ct.StateAwareController(cfg_exact(epoch_corr=0.0)).decide(state(remaining=9), [keep, other], 0)
    corr = ct.StateAwareController(cfg_exact(epoch_corr=1.0)).decide(state(remaining=9), [keep, other], 0)
    assert indep.switched and math.isclose(indep.u_diff, math.sqrt(18.0))        # gain 9 > sqrt(9 + 9)
    assert not corr.switched                                                      # gain 9 < sqrt(81 + 81)
    rec = next(r for r in corr.candidates if not r["is_keep"])
    assert math.isclose(rec["u_diff"], math.sqrt(162.0)) and math.isclose(rec["gain"], 9.0)
    assert any("epoch_corr" in a for a in corr.assumptions)


def test_stationary_survival_is_the_fixed_capacity_assumption():
    cands = install_example_candidates()
    moving = ct.StateAwareController(ct.ControllerConfig(hysteresis_s=0.0, min_residency=0))
    fixed = ct.StateAwareController(ct.ControllerConfig(hysteresis_s=0.0, min_residency=0, stationary_survival=True))
    for e in range(6):
        fr = frame(100e6 if e % 2 == 0 else 10e6, 0.02, e)      # a regime change at every frame
        moving.observe(fr)
        fixed.observe(fr)
    assert fixed.survival(5) == [1.0] * 5 and sum(moving.survival(5)) < 5.0
    d_fixed = fixed.decide(state(remaining=5), cands, 6)
    assert math.isclose(d_fixed.q_keep, 50.0) and d_fixed.chosen == PLAN_B
    assert any("stationary" in a for a in d_fixed.assumptions)
    assert not any("stationary" in a for a in moving.decide(state(remaining=5), cands, 6).assumptions)
    # stationary_survival does not read dwell_prior at all
    assert ct.StateAwareController(ct.ControllerConfig(stationary_survival=True, dwell_prior=0.0)).survival(3) == [1.0] * 3


def test_decide_with_survival_equals_decide_and_prices_wait():
    cands = install_example_candidates()
    a = ct.StateAwareController(cfg_exact()).decide(state(remaining=5), cands, 0)
    b = ct.StateAwareController(cfg_exact()).decide_with_survival(state(remaining=5), cands, 0, [1.0] * 5)
    assert a.to_dict() == b.to_dict()
    half = ct.StateAwareController(cfg_exact()).decide_with_survival(state(remaining=5), cands, 0, [1.0, 0.5])
    assert math.isclose(half.q_keep, 15.0)
    waiting = install_example_candidates()
    waiting[2].c_wait_s = 40.0                                   # B now costs 21 + 40 = 61 > 50
    dec = ct.StateAwareController(cfg_exact()).decide(state(remaining=5), waiting, 0)
    assert not dec.switched
    q = {cd.plan_key(r["plan"]): r["q_mean"] for r in dec.candidates}
    assert math.isclose(q[cd.plan_key(PLAN_B)], 61.0)


def test_ablation_assumptions_describe_the_configuration():
    cands = install_example_candidates()
    no_inst = ct.make_policy("B4_no_install").decide(state(), cands, ENV, 0)
    assert any("install cost ignored" in a for a in no_inst.assumptions)
    assert "install cost candidate-specific and fully included" not in no_inst.assumptions
    no_unc = ct.make_policy("B4_no_unc").decide(state(), cands, ENV, 0)
    assert any("uncertainty ignored" in a for a in no_unc.assumptions)
    full = ct.make_policy("B4_opencrab").decide(state(), cands, ENV, 0)
    assert list(full.assumptions) == list(ct.ASSUMPTIONS_B4)


# ---------------------------------------------------------------------------
# planner hook of B4: planner x controller factorial
# ---------------------------------------------------------------------------
def _planners():
    from opencrab2.highlevel.planner import ExactAssign, GreedyPerOperator, ILPAdapter
    return {"ExactAssign": ExactAssign, "ILPAdapter": ILPAdapter, "GreedyPerOperator": GreedyPerOperator}


def _toy_model():
    model = cm.FittedCostModel(label="MODELED")
    for comp in cm.COMPONENTS:
        model.fits[comp] = cm.ComponentFit(1e-9, 0.0, 0.001, 0.0005, 1, "MODELED", "toy")
    return model


def _gpt2_candidates(env, remaining=6, installed=None):
    shapes = cd.gpt2_shapes(n_layers=2, n=1, T=16)
    cat = cd.gpt2_catalog(shapes)
    st = cd.SessionState(dict(PLAN_KEEP), dict(installed or {}), {}, 16, remaining, "s", "gpt2@v1", "nc")
    return st, cd.enumerate_candidates(st, cat, _toy_model(), env, shapes)


@pytest.mark.parametrize("planner_name", ["ExactAssign", "ILPAdapter", "GreedyPerOperator"])
@pytest.mark.parametrize("env", [cm.Env(100e6, 0.01), cm.Env(2e6, 0.0005), cm.Env(1e9, 0.05)])
def test_b4_with_planner_equals_plain_b4_without_uncertainty(planner_name, env):
    """On the GPT-2 catalog the candidates are an additive product: every planner returns argmin Qhat."""
    st, cands = _gpt2_candidates(env)
    kw = dict(hysteresis_s=0.0, min_residency=0, stationary_survival=True)
    plain = ct.make_policy("B4_no_unc", **kw).decide(st, cands, env, 0)
    pol = ct.make_policy("B4_no_unc", planner=_planners()[planner_name](), **kw)
    dec = pol.decide(st, cands, env, 0)
    assert isinstance(pol, ct.ControllerPolicy) and pol.planner is not None
    info = pol.planner_log[-1]
    assert info["planner"] == planner_name and info["chain_ops"] == 4 and info["fallback"] == ""
    best = min(cands, key=lambda c: st.remaining_epochs * c.exec_s_per_epoch + c.c_install_s)
    assert info["plan"] == best.plan
    assert math.isclose(info["cost"], st.remaining_epochs * best.exec_s_per_epoch + best.c_install_s, rel_tol=1e-6)
    assert dec.chosen == plain.chosen and dec.switched == plain.switched
    assert math.isclose(dec.q_keep, plain.q_keep, rel_tol=1e-12)
    assert math.isclose(dec.q_best, plain.q_best, rel_tol=1e-12)
    assert len(dec.candidates) <= 2 and sum(1 for r in dec.candidates if r["planner_pick"]) == 1
    assert any(planner_name in a for a in dec.assumptions)


def _shared_object_candidates():
    """Two axes whose better values share one install object: a greedy per-operator planner refuses both."""
    def mk(a, b, horizon_cost, keep=False):
        needs = ["S"] if "y" in (a, b) else []
        return cd.Candidate(plan={"a": a, "b": b}, install_objects=needs, c_install_s=9.0 if needs else 0.0,
                            c_install_sigma=0.0, exec_s_per_epoch=horizon_cost, exec_sigma=0.0,
                            material_bytes_per_epoch=0, mem_peak_bytes=0, feasible=True, reason="OK", is_keep=keep,
                            install_cost_by_object={"S": 9.0} if needs else {})
    cands = [mk("x", "x", 20.0, True), mk("y", "x", 14.0), mk("x", "y", 14.0), mk("y", "y", 8.0)]
    st = cd.SessionState({"a": "x", "b": "x"}, {}, {}, 32, 1, "s", "m", "h")
    return st, cands


def test_planner_hook_exact_planners_switch_where_greedy_keeps():
    st, cands = _shared_object_candidates()
    kw = dict(hysteresis_s=0.0, min_residency=0, stationary_survival=True)
    P = _planners()
    chain, pcat, decode = ct.candidates_chain(st, cands, 1.0, True)
    assert [op["id"] for op in chain] == ["a", "b"]
    from opencrab2.highlevel.planner import brute_force
    bf_assign, bf_cost = brute_force(chain, pcat, None, None, st)
    assert decode(bf_assign) == {"a": "y", "b": "y"} and math.isclose(bf_cost, 17.0)
    for name in ("ExactAssign", "ILPAdapter"):
        pol = ct.make_policy("B4_opencrab", planner=P[name](), **kw)
        dec = pol.decide(st, cands, ENV, 0)
        assert dec.switched and dec.chosen == {"a": "y", "b": "y"}, name
        assert math.isclose(dec.q_keep, 20.0) and math.isclose(dec.q_best, 17.0)
        assert math.isclose(pol.planner_log[-1]["cost"], 17.0)
    greedy = ct.make_policy("B4_opencrab", planner=P["GreedyPerOperator"](), **kw)
    dg = greedy.decide(st, cands, ENV, 0)
    assert not dg.switched and dg.chosen == {"a": "x", "b": "x"}          # 14 + 6 = 20: no gain over keep
    assert greedy.planner_log[-1]["plan"] == {"a": "x", "b": "x"}
    # without a planner the controller ranks every candidate itself and finds the joint assignment
    plain = ct.make_policy("B4_opencrab", **kw).decide(st, cands, ENV, 0)
    assert plain.switched and plain.chosen == {"a": "y", "b": "y"}


def test_planner_hook_falls_back_to_the_candidate_chain_when_not_a_product():
    cands = install_example_candidates()                                    # three plans: not a full product of axis values
    kw = dict(hysteresis_s=0.0, min_residency=0, stationary_survival=True)
    for name, cls in _planners().items():
        pol = ct.make_policy("B4_opencrab", planner=cls(), **kw)
        dec = pol.decide(state(remaining=5), cands, ENV, 0)
        assert pol.planner_log[-1]["chain_ops"] == 1, name
        assert dec.switched and dec.chosen == PLAN_B and math.isclose(dec.q_best, 21.0), name
    # an infeasible candidate is never assigned: the chain holds feasible candidates only
    blocked = [cand(PLAN_KEEP, 10.0, is_keep=True), cand(PLAN_A, 1.0, 1.0, feasible=False), cand(PLAN_B, 4.0, 1.0)]
    pol = ct.make_policy("B4_opencrab", planner=_planners()["ExactAssign"](), **kw)
    dec = pol.decide(state(remaining=5), blocked, ENV, 0)
    assert dec.chosen == PLAN_B and pol.planner_log[-1]["plan"] == PLAN_B
    none = [cand(PLAN_KEEP, 10.0, is_keep=True, feasible=False), cand(PLAN_A, 1.0, feasible=False)]
    pol2 = ct.make_policy("B4_opencrab", planner=_planners()["ExactAssign"](), **kw)
    d2 = pol2.decide(state(remaining=5), none, ENV, 0)
    assert not d2.switched and pol2.planner_log[-1]["fallback"] == "NO_FEASIBLE_CANDIDATE"


def test_planner_controller_factorial_covers_b2_and_b4():
    """Every (planner, control) cell is constructible through make_policy and returns a Decision."""
    cands = install_example_candidates()
    cells = {}
    for name, cls in _planners().items():
        b2 = ct.make_policy("B3_planner_plus_B2", planner=cls())
        b4 = ct.make_policy("B4_opencrab", planner=cls(), hysteresis_s=0.0, min_residency=0, stationary_survival=True)
        cells[(name, "B2")] = b2.decide(state(remaining=5), cands, ENV, 0)
        cells[(name, "B4")] = b4.decide(state(remaining=5), cands, ENV, 0)
    assert len(cells) == 6 and all(isinstance(d, ct.Decision) for d in cells.values())
    assert all(cells[(n, "B4")].chosen == PLAN_B for n in _planners())
    for abl in ("B4_no_install", "B4_no_hyst", "B4_no_unc"):
        pol = ct.make_policy(abl, planner=_planners()["ILPAdapter"]())
        assert pol.planner is not None and pol.name == abl
        assert isinstance(pol.decide(state(remaining=5), cands, ENV, 0), ct.Decision)
    assert ct.make_policy("B4_opencrab").planner is None
    # the ablation without install cost hands the planner a chain without install objects
    no_inst = ct.make_policy("B4_no_install", planner=_planners()["ExactAssign"](), hysteresis_s=0.0, min_residency=0,
                             stationary_survival=True)
    d = no_inst.decide(state(remaining=5), cands, ENV, 0)
    assert d.chosen == PLAN_A                                    # 5 x 1 < 5 x 4 once the install of 100 is ignored


def test_candidates_chain_is_additive_and_carries_install_objects():
    env = cm.Env(100e6, 0.01)
    st, cands = _gpt2_candidates(env, remaining=4)
    chain, pcat, decode = ct.candidates_chain(st, cands, 4.0, True)
    assert [op["id"] for op in chain] == ["embedding", "linear", "argmax_k", "max_k"]
    assert [len(op["realizations"]) for op in chain] == [2, 2, 3, 2]
    objs = {(op["id"], r["name"].split(":", 1)[1]): sorted(o["id"] for o in r["objects"])
            for op in chain for r in op["realizations"]}
    assert "wte" in objs[("embedding", "wmask")] and "wte" not in objs[("embedding", "generic")]
    assert any(o.startswith("h0.") for o in objs[("linear", "wmask")])
    assert not any(o.startswith("h0.") for o in objs[("linear", "generic")])
    assert all("lm_head" in v for v in objs.values())            # the common object is priced once by the planner
    assert all(r["cost"] >= 0.0 for op in chain for r in op["realizations"])
    from opencrab2.highlevel.planner import ExactAssign, brute_force
    for c in cands[:6]:
        names = {}
        for op in chain:
            names[op["id"]] = next(r["name"] for r in op["realizations"] if r["name"].split(":", 1)[1] == str(c.plan[op["id"]]))
        assert decode(names) == c.plan
    a_ex, c_ex = ExactAssign().plan(chain, pcat, env, None, st)
    a_bf, c_bf = brute_force(chain, pcat, env, None, st)
    best = min(cands, key=lambda c: 4.0 * c.exec_s_per_epoch + c.c_install_s)
    assert decode(a_ex) == decode(a_bf) == best.plan
    assert math.isclose(c_ex, c_bf, rel_tol=1e-9) and math.isclose(c_ex, 4.0 * best.exec_s_per_epoch + best.c_install_s, rel_tol=1e-6)
    # an infeasible candidate switches the builder to the single-operator chain of feasible candidates
    cands[3].feasible = False
    chain1, _, decode1 = ct.candidates_chain(st, cands, 4.0, True)
    assert len(chain1) == 1 and len(chain1[0]["realizations"]) == len(cands) - 1


# ---------------------------------------------------------------------------
# per-step objective (paper, Section 4.2, objective): E[T_exec(p, t + j)] of every remaining step
# ---------------------------------------------------------------------------
def cand_steps(plan, by_s, by_g=None, install_s=0.0, install_sigma=0.0, is_keep=False, feasible=True):
    """A candidate with per-step costs; entry 0 is the current epoch (``exec_s_per_epoch``)."""
    by_g = list(by_g) if by_g is not None else [0.0] * len(by_s)
    c = cand(plan, by_s[0], install_s, by_g[0], install_sigma, is_keep=is_keep, feasible=feasible)
    c.exec_s_by_epoch = [float(v) for v in by_s]
    c.exec_sigma_by_epoch = [float(v) for v in by_g]
    return c


def test_q_hat_sums_the_scheduled_steps():
    c = cand_steps(PLAN_A, [16.0, 7.0, 7.0, 7.0], [0.6, 0.3, 0.3, 0.3], install_s=5.0, install_sigma=0.2)
    s = [1.0, 0.8, 0.64, 0.512]
    m, v = ct.q_hat(c, state(), s)
    assert math.isclose(m, 5.0 + 16.0 + 7.0 * (0.8 + 0.64 + 0.512))
    a = [0.6, 0.8 * 0.3, 0.64 * 0.3, 0.512 * 0.3]
    assert math.isclose(v, sum(x * x for x in a) + 0.04)
    # epoch_corr mixes in the square of the sum, as on one shape
    m1, v1 = ct.q_hat(c, state(), s, epoch_corr=1.0)
    assert m1 == m and math.isclose(v1, sum(a) ** 2 + 0.04)
    _, vh = ct.q_hat(c, state(), s, epoch_corr=0.5)
    assert math.isclose(vh, 0.5 * (v - 0.04) + 0.5 * (v1 - 0.04) + 0.04)
    m0, v0 = ct.q_hat(c, state(), s, include_install=False)
    assert math.isclose(m0, m - 5.0) and math.isclose(v0, v - 0.04)
    # the one-shape form is the special case of equal entries
    flat = cand_steps(PLAN_A, [7.0] * 4, [0.3] * 4, install_s=5.0, install_sigma=0.2)
    plain = cand(PLAN_A, 7.0, 5.0, 0.3, 0.2)
    for corr in (0.0, 0.5, 1.0):
        mf, vf = ct.q_hat(flat, state(), s, epoch_corr=corr)
        mp, vp = ct.q_hat(plain, state(), s, epoch_corr=corr)
        assert math.isclose(mf, mp, rel_tol=1e-12) and math.isclose(vf, vp, rel_tol=1e-12)
    # a horizon longer than the lists repeats their last entry; a shorter one ignores the surplus
    short = cand_steps(PLAN_A, [16.0, 7.0])
    assert math.isclose(ct.q_hat(short, state(), [1.0] * 5)[0], 16.0 + 4 * 7.0)
    long_ = cand_steps(PLAN_A, [16.0, 7.0, 1.0, 1.0])
    assert math.isclose(ct.q_hat(long_, state(), [1.0, 1.0])[0], 23.0)
    with pytest.raises(ValueError, match="^CANDIDATE_SCHEDULE"):
        ct.q_hat(cand_steps(PLAN_A, [1.0, 2.0], [0.1]), state(), [1.0, 1.0])


def test_horizon_exec_and_the_baselines_sum_the_steps():
    keep = cand_steps(PLAN_KEEP, [16.0, 13.0, 13.0, 13.0], is_keep=True)     # generic: cheaper prefill, dear decode
    w = cand_steps(PLAN_A, [18.0, 7.0, 7.0, 7.0], install_s=6.0)            # weight masks: dearer prefill, cheap decode
    assert math.isclose(ct.horizon_exec(keep, 4), 55.0) and math.isclose(ct.horizon_exec(w, 4), 39.0)
    assert math.isclose(ct.horizon_exec(w, 6), 53.0)                         # beyond the list: the last entry
    assert math.isclose(ct.horizon_exec(cand(PLAN_A, 7.0), 6), 42.0)         # one shape: H * exec
    st = state(remaining=4)
    flat = [cand(PLAN_KEEP, 16.0, is_keep=True), cand(PLAN_A, 18.0, 6.0)]    # the same candidates on the first step alone
    b2 = ct.make_policy("B2_remaining_plus_install").decide(st, [keep, w], ENV, 0)
    assert b2.switched and b2.chosen == PLAN_A and math.isclose(b2.q_keep, 55.0) and math.isclose(b2.q_best, 45.0)
    assert not ct.make_policy("B2_remaining_plus_install").decide(st, flat, ENV, 0).switched   # 4 x 16 < 4 x 18 + 6
    q = {cd.plan_key(r["plan"]): r["q_mean"] for r in b2.candidates}
    assert math.isclose(q[cd.plan_key(PLAN_A)], 45.0) and math.isclose(q[cd.plan_key(PLAN_KEEP)], 55.0)
    kw = dict(hysteresis_s=0.0, min_residency=0, stationary_survival=True)
    b4 = ct.make_policy("B4_opencrab", **kw).decide(st, [keep, w], ENV, 0)
    assert b4.switched and math.isclose(b4.q_keep, 55.0) and math.isclose(b4.q_best, 45.0)
    assert not ct.make_policy("B4_opencrab", **kw).decide(st, flat, ENV, 0).switched
    fixed = ct.make_policy("B0_fixed:" + json.dumps(PLAN_A)).decide(st, [keep, w], ENV, 0)
    assert fixed.switched and math.isclose(fixed.q_best, 45.0) and math.isclose(fixed.q_keep, 55.0)
    # B1 compares the current epoch alone; B6 takes the cheapest current epoch and reports the per-step horizon
    b1 = ct.make_policy("B1_naive").decide(st, [keep, w], ENV, 0)
    assert not b1.switched and math.isclose(b1.q_keep, 64.0)
    b6 = ct.make_policy("B6_zero_overhead").decide(st, [keep, w], ENV, 0)
    assert not b6.switched and math.isclose(b6.q_keep, 55.0) and math.isclose(b6.q_best, 55.0)
    cheap_now = cand_steps(PLAN_B, [15.0, 14.0, 14.0, 14.0])
    b6 = ct.make_policy("B6_zero_overhead").decide(st, [keep, w, cheap_now], ENV, 0)
    assert b6.chosen == PLAN_B and math.isclose(b6.q_best, 57.0)


def test_b3_applies_the_b2_horizon_to_the_planner_pick():
    from opencrab2.highlevel.planner import ExactAssign
    keep = cand_steps(PLAN_KEEP, [10.0, 10.0, 10.0], is_keep=True)
    a = cand_steps(PLAN_A, [9.0, 1.0, 1.0], install_s=5.0)                  # the planner's pick: cheapest current epoch
    st = state(remaining=3)
    b3 = ct.PlannerPlusB2Policy(planner=ExactAssign()).decide(st, [keep, a], ENV, 0)
    assert b3.switched and b3.chosen == PLAN_A and math.isclose(b3.q_keep, 30.0) and math.isclose(b3.q_best, 16.0)
    flat = [cand(PLAN_KEEP, 10.0, is_keep=True), cand(PLAN_A, 9.0, 5.0)]
    assert not ct.PlannerPlusB2Policy(planner=ExactAssign()).decide(st, flat, ENV, 0).switched   # 27 + 5 > 30


def test_b5_oracle_prices_every_future_epoch_on_its_scheduled_shape():
    model = _toy_model()
    sched = [cd.gpt2_shapes(n_layers=1, n=4, T=8)] + [cd.gpt2_shapes(n_layers=1, n=1, T=8)] * 3
    cat = cd.gpt2_catalog(sched[0])
    st = cd.SessionState(dict(PLAN_KEEP), {}, {}, 8, 4, "s", "gpt2@v1", "nc")
    now, later = cm.Env(1e8, 0.005), cm.Env(1e6, 0.02)
    cands = cd.enumerate_candidates(st, cat, model, now, sched[0], schedule=sched)
    keep = cands[0]
    oracle = ct.make_policy("B5_oracle", trace=[now, later, later, later], costmodel=model, shapes=sched[0], schedule=sched)
    dec = oracle.decide(st, cands, now, 0)
    for c in cands:
        want = c.c_install_s + cd.exec_cost(c.plan, sched[0], model, now)[0] + \
            sum(cd.exec_cost(c.plan, sh, model, later)[0] for sh in sched[1:])
        if c.is_keep:
            assert math.isclose(dec.q_keep, want, rel_tol=1e-12)
        if c.plan == dec.chosen:
            assert math.isclose(dec.q_best, want, rel_tol=1e-12)
    # under a constant trace the oracle's total is the per-step sum of the candidate lists (B2's horizon term)
    const = ct.make_policy("B5_oracle", trace=[now] * 4, costmodel=model, shapes=sched[0], schedule=sched)
    d2 = const.decide(st, cands, now, 0)
    assert math.isclose(d2.q_keep, keep.c_install_s + sum(keep.exec_s_by_epoch), rel_tol=1e-12)
    recs = {cd.plan_key(r["plan"]): r["q_mean"] for r in d2.candidates}
    for c in cands:
        assert math.isclose(recs[cd.plan_key(c.plan)], c.c_install_s + ct.horizon_exec(c, 4), rel_tol=1e-12)
    # without a schedule the oracle grows the KV length by one key per epoch on the current shape
    grow = ct.make_policy("B5_oracle", trace=[now] * 4, costmodel=model, shapes=sched[0])
    d3 = grow.decide(st, cands, now, 0)
    want = keep.c_install_s + sum(cd.exec_cost(keep.plan, dict(sched[0], T=8 + j), model, now)[0] for j in range(4))
    assert math.isclose(d3.q_keep, want, rel_tol=1e-12) and d3.q_keep != d2.q_keep


@pytest.mark.parametrize("planner_name", ["ExactAssign", "ILPAdapter", "GreedyPerOperator"])
def test_planner_hook_prices_the_schedule(planner_name):
    """The planner chain carries the per-step horizon cost: every planner returns argmin of the per-step Qhat."""
    env = cm.Env(100e6, 0.01)
    sched = [cd.gpt2_shapes(n_layers=2, n=8, T=16)] + [cd.gpt2_shapes(n_layers=2, n=1, T=16)] * 5
    st = cd.SessionState(dict(PLAN_KEEP), {}, {}, 16, 6, "s", "gpt2@v1", "nc")
    cands = cd.enumerate_candidates(st, cd.gpt2_catalog(sched[0]), _toy_model(), env, sched[0], schedule=sched)
    kw = dict(hysteresis_s=0.0, min_residency=0, stationary_survival=True)
    plain = ct.make_policy("B4_no_unc", **kw).decide(st, cands, env, 0)
    pol = ct.make_policy("B4_no_unc", planner=_planners()[planner_name](), **kw)
    dec = pol.decide(st, cands, env, 0)
    best = min(cands, key=lambda c: ct.horizon_exec(c, 6) + c.c_install_s)
    info = pol.planner_log[-1]
    assert info["plan"] == best.plan and info["chain_ops"] == 4 and info["fallback"] == ""
    assert math.isclose(info["cost"], ct.horizon_exec(best, 6) + best.c_install_s, rel_tol=1e-6)
    assert dec.chosen == plain.chosen == best.plan and dec.switched == plain.switched
    assert math.isclose(dec.q_best, plain.q_best, rel_tol=1e-12) and math.isclose(dec.q_keep, plain.q_keep, rel_tol=1e-12)
    # the survival vector is the horizon of the chain: weights below one discount the later steps
    s = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]
    chain, _, decode = ct.candidates_chain(st, cands, s, True)
    chain_h, _, _ = ct.candidates_chain(st, cands, 6.0, True)
    assert [op["id"] for op in chain] == [op["id"] for op in chain_h] == ["embedding", "linear", "argmax_k", "max_k"]
    tot_s = sum(min(r["cost"] for r in op["realizations"]) for op in chain)
    tot_h = sum(min(r["cost"] for r in op["realizations"]) for op in chain_h)
    assert 0.0 < tot_s < tot_h


# ---------------------------------------------------------------------------
# the calibrated cost model: the first-step choice over sensed bandwidths (paper, Section 4.2)
# ---------------------------------------------------------------------------
def _fitted_model() -> cm.FittedCostModel:
    """The shipped calibration (results/calibration/fitted_model.json), or the package's calibration path."""
    shipped = Path(__file__).resolve().parents[2] / "results" / "calibration" / "fitted_model.json"
    for p in (shipped, cm.calibration_path()):
        if p.is_file():
            return cm.FittedCostModel.load(p)
    pytest.fail(f"CALIBRATION_MISSING: {shipped}")


@pytest.fixture(scope="module")
def fitted():
    return _fitted_model()


GPT2_BUCKET = 32            # the length bucket of the GPT-2 sessions: a 16-token prompt and 8 epochs stay in one bucket
GPT2_PROMPT = 16


def _gpt2_state(n_layers, warm, remaining):
    """Cold: all-generic incumbent, nothing installed. Warm: all-wmask incumbent with every object installed."""
    catalog = cd.gpt2_catalog(cd.gpt2_shapes(n_layers, GPT2_PROMPT, GPT2_BUCKET))
    version = {"model_version": "gpt2@test", "client_key_domain": "client-default"}
    installed = {o: dict(version) for o in catalog["objects"]} if warm else {}
    return cd.SessionState(dict(PLAN_A if warm else PLAN_KEEP), installed, {}, GPT2_BUCKET, remaining, "s", "gpt2@test", "nc")


def _gpt2_decision(policy, model, n_layers, warm, bw, rtt, t, n_epochs, schedule=True, memo=None):
    """The decision of epoch ``t`` of an ``n_epochs``-epoch session: the prefill at t = 0, one-row decode steps after."""
    env = cm.Env(float(bw), float(rtt) / 2.0, 1.0)
    rows = GPT2_PROMPT if t == 0 else 1
    sched = [cd.gpt2_shapes(n_layers, rows, GPT2_BUCKET)] + [cd.gpt2_shapes(n_layers, 1, GPT2_BUCKET)] * (n_epochs - t - 1)
    st = _gpt2_state(n_layers, warm, n_epochs - t)
    cands = cd.enumerate_candidates(st, cd.gpt2_catalog(sched[0]), model, env, sched[0],
                                    schedule=sched if schedule else None, memo=memo)
    pol = ct.make_policy(policy)
    pol.observe({"bandwidth_bps": float(bw), "rtt_s": float(rtt)})
    return pol.decide(st, cands, env, t)


SENSED_BANDWIDTHS = (0.3e9, 1e9, 2.78e9, 3.145e9, 5.4e9, 10e9, 23.5e9, 50e9, 100e9)


@pytest.mark.parametrize("policy", ["B2_remaining_plus_install", "B4_opencrab", "B4_no_install"])
@pytest.mark.parametrize("n_layers", [4, 12])
@pytest.mark.parametrize("warm", [False, True])
def test_calibrated_first_step_chooses_the_weight_masks_at_every_sensed_bandwidth(fitted, policy, n_layers, warm):
    """At the prefill the decode steps are priced as one-row steps, so the first-step plan is all-wmask whatever the probe read."""
    memo = {}
    for n_epochs in (4, 8):
        for rtt in (0.5e-3, 1.6e-3):
            for bw in SENSED_BANDWIDTHS:
                dec = _gpt2_decision(policy, fitted, n_layers, warm, bw, rtt, 0, n_epochs, memo=memo)
                assert dec.chosen == PLAN_A, (policy, n_layers, warm, n_epochs, rtt, bw, dec.chosen, dec.reason)
                assert dec.switched == (not warm), (policy, n_layers, warm, n_epochs, rtt, bw, dec.reason)


def test_calibrated_first_step_of_the_recorded_frames(fitted):
    """Three first-step frames of the first measured campaign (diag4 B4 seed 2, full12 B4 seed 2, warm diag4 B2 seed 1).

    Priced on the prefill shape alone the recorded values come back (Qhat(keep), Qhat(best)) and the plan without
    weight masks is kept or taken; priced step by step the weight masks are installed or kept.
    """
    frames = [("B4_opencrab", 4, False, 23510647043.330708, 0.001581299991812557, 71.20827529808291, 71.20827529808291),
              ("B4_opencrab", 12, False, 3145099192.1254463, 0.0012669000134337693, 162.9190412942429, 162.9190412942429),
              ("B2_remaining_plus_install", 4, True, 1e11, 0.001671799982432276, 143.0537398739729, 132.01342263733002)]
    for policy, n_layers, warm, bw, rtt, q_keep, q_best in frames:
        one_shape = _gpt2_decision(policy, fitted, n_layers, warm, bw, rtt, 0, 8, schedule=False)
        assert one_shape.chosen == PLAN_KEEP and one_shape.switched == warm, (policy, n_layers, one_shape.reason)
        assert math.isclose(one_shape.q_keep, q_keep, rel_tol=1e-6) and math.isclose(one_shape.q_best, q_best, rel_tol=1e-6)
        per_step = _gpt2_decision(policy, fitted, n_layers, warm, bw, rtt, 0, 8)
        assert per_step.chosen == PLAN_A and per_step.switched == (not warm), (policy, n_layers, per_step.reason)
        assert per_step.q_keep < q_keep                       # seven one-row steps cost less than seven prefills


@pytest.mark.parametrize("policy", ["B2_remaining_plus_install", "B4_opencrab", "B4_no_install", "B1_naive", "B3_planner_plus_B2"])
def test_calibrated_later_steps_decide_as_before(fitted, policy):
    """From the first decode step on every remaining epoch has the one-row shape: per-step pricing equals one-shape pricing."""
    for n_layers in (4, 12):
        for warm in (False, True):
            for t in (1, 4, 7):
                for bw in (0.5e9, 3e9, 100e9):
                    a = _gpt2_decision(policy, fitted, n_layers, warm, bw, 1.2e-3, t, 8, schedule=False)
                    b = _gpt2_decision(policy, fitted, n_layers, warm, bw, 1.2e-3, t, 8)
                    tag = (policy, n_layers, warm, t, bw)
                    assert a.chosen == b.chosen and a.switched == b.switched, tag
                    assert math.isclose(a.q_keep, b.q_keep, rel_tol=1e-9) and math.isclose(a.q_best, b.q_best, rel_tol=1e-9), tag
                    assert math.isclose(a.lower_bound_gain, b.lower_bound_gain, rel_tol=1e-9, abs_tol=1e-9), tag

