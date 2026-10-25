"""run_simulation tests: the quick campaign validates, is MODELED, covers every policy and writes the factorial table."""

from __future__ import annotations

import json

import pytest

import math

from opencrab2.experiments import run_simulation as rs
from opencrab2.experiments import schema, traces
from opencrab2.highlevel import candidates as cand
from opencrab2.highlevel import costmodel as cm


@pytest.fixture(scope="module")
def cmodel():
    return rs.CachedCostModel(cm.default_model())


def test_quick_campaign_records_validate_and_cover_policies(tmp_path):
    rc = rs.main(["--quick", "--workers", "2", "--out-root", str(tmp_path)])
    assert rc == 0
    recs = schema.read_jsonl(tmp_path / "raw" / "simulation.jsonl")      # strict: every record validates
    assert recs
    assert {r["status"] for r in recs} == {"MODELED"}
    assert all(r["security_eligible"] is False for r in recs)
    assert all(r["record_kind"] == "simulation" and r["workload_scope"] == "full-model" for r in recs)
    assert all(r["clock_basis"] == "modeled-epoch" for r in recs)
    policies = {r["policy_id"] for r in recs}
    assert set(rs.ADAPTIVE_POLICIES) <= policies
    assert "B0_fixed_best_hindsight" in policies
    assert any(p.startswith("B0_fixed:") for p in policies)
    assert {r["family"] for r in recs} == {"S0", "S3"}
    assert {r["repeat"] for r in recs} == {1, 2}
    assert all(r["tokens"] == 1 + r["T"] for r in recs if r["aborts"] == 0)
    assert all(r["T_cold_s"] >= r["T_warm_s"] >= 0 and r["T_cold_s"] >= r["T_online_s"] for r in recs)
    assert len({r["run_id"] for r in recs}) == len(recs)
    # the per-epoch payload is not written by default; the record names its file relative to the data root
    assert all("epochs" not in r["extra"] and r["extra"]["n_epochs_logged"] == r["tokens"] for r in recs)
    assert {r["raw_measurement_path"] for r in recs} == {"raw/simulation.jsonl"}
    fact = schema.read_jsonl(tmp_path / "raw" / "factorial.jsonl")
    assert fact and {r["record_kind"] for r in fact} == {"factorial"}
    assert {r["raw_measurement_path"] for r in fact} == {"raw/factorial.jsonl"} and all("epochs" not in r["extra"] for r in fact)
    assert {(r["extra"]["planner"], r["extra"]["control"]) for r in fact} == {(p, c) for p in rs.PLANNERS for c in rs.CONTROLS}
    tex = (tmp_path / "tables" / "factorial.tex").read_text(encoding="utf-8")
    assert "\\est{" in tex and "ExactAssign" in tex and "ILPAdapter" in tex and "\\bottomrule" in tex
    summary = json.loads((tmp_path / "processed" / "factorial_summary.json").read_text(encoding="utf-8"))
    assert set(summary["factorial"]) == {f"{p}x{c}" for p in rs.PLANNERS for c in rs.CONTROLS}
    # resumable: a second run adds nothing
    n = len(recs)
    rs.main(["--quick", "--workers", "1", "--out-root", str(tmp_path)])
    assert len(schema.read_jsonl(tmp_path / "raw" / "simulation.jsonl")) == n


def test_simulation_is_deterministic(cmodel):
    tr = traces.make_trace("S3", "two-changes", 2, rs.session_duration_estimate("S3", "two-changes", 8, cmodel))
    a = rs.simulate_session(tr, "B4_opencrab", 8, 2, cmodel)
    b = rs.simulate_session(tr, "B4_opencrab", 8, 2, cmodel)
    assert a.total_s == b.total_s and a.final_plan == b.final_plan and a.post_init_switches == b.post_init_switches
    assert a.tokens == 9 and a.decisions == 9


def test_cancel_without_resume_aborts_and_burns(cmodel):
    dur = rs.session_duration_estimate("S7", "cancel-no-resume", 8, cmodel)
    tr = traces.make_trace("S7", "cancel-no-resume", 2, dur)
    r = rs.simulate_session(tr, "B2_remaining_plus_install", 8, 2, cmodel)
    assert r.aborts == 1 and r.tokens < 9 and r.material_burned > 0
    rec = rs.session_record(tr, 8, 2, r, {}, "x")
    schema.validate(rec)
    assert rec["aborts"] == 1 and rec["status"] == "MODELED"
    assert "epochs" not in rec["extra"] and rec["extra"]["n_epochs_logged"] == len(r.epochs)
    full = rs.session_record(tr, 8, 2, r, {}, "x", keep_epochs=True)
    schema.validate(full)
    assert [e["epoch"] for e in full["extra"]["epochs"]] == [e.epoch for e in r.epochs]
    assert full["extra"]["epochs"][-1]["reason"] == "ABORTED_BY_CANCEL"
    # the two forms agree on every session total
    assert {k: v for k, v in full["extra"].items() if k != "epochs"} == rec["extra"]
    assert all(full[k] == rec[k] for k in schema.RECORD_FIELDS if k not in ("created_utc",))
    # run_cell accepts the flag as the eighth element of its argument tuple
    recs, facts = rs.run_cell(("S0", "stable-LAN", 1, 8, ("B4_opencrab",), (), False, True))
    assert recs and all("epochs" in x["extra"] for x in recs) and not facts


def test_cancel_with_resume_recovers(cmodel):
    dur = rs.session_duration_estimate("S7", "cancel-mid-resume", 8, cmodel)
    tr = traces.make_trace("S7", "cancel-mid-resume", 2, dur)
    r = rs.simulate_session(tr, "B4_opencrab", 8, 2, cmodel)
    assert r.aborts == 0 and r.tokens == 9 and r.recovery_s > 0 and r.material_burned > 0


def test_material_starvation_waits(cmodel):
    dur = rs.session_duration_estimate("S5", "refill-starved", 8, cmodel)
    tr = traces.make_trace("S5", "refill-starved", 2, dur)
    r = rs.simulate_session(tr, "B4_opencrab", 8, 2, cmodel)
    assert r.wait_s > 0 and r.material_generated > 0
    ample = traces.make_trace("S5", "refill-ample", 2, dur)
    r2 = rs.simulate_session(ample, "B4_opencrab", 8, 2, cmodel)
    assert r2.wait_s < r.wait_s


def test_warm_start_installs_nothing(cmodel):
    dur = rs.session_duration_estimate("S8", "warm-all", 8, cmodel)
    warm = rs.simulate_session(traces.make_trace("S8", "warm-all", 2, dur), "B2_remaining_plus_install", 8, 2, cmodel)
    cold = rs.simulate_session(traces.make_trace("S8", "cold", 2, dur), "B2_remaining_plus_install", 8, 2, cmodel)
    assert warm.install_s <= cold.install_s
    assert warm.total_s < cold.total_s or warm.install_s == 0.0


def test_zero_overhead_oracle_pays_only_execution(cmodel):
    """B6 is the lower bound: no sensing, install, switch waste or material wait; execution only (MODELED)."""
    dur = rs.session_duration_estimate("S5", "refill-starved", 8, cmodel)
    tr = traces.make_trace("S5", "refill-starved", 2, dur)
    b6 = rs.simulate_session(tr, "B6_zero_overhead", 8, 2, cmodel)
    assert b6.sensing_s == 0.0 and b6.install_s == 0.0 and b6.wait_s == 0.0 and b6.material_burned == 0
    assert b6.tokens == 9 and abs(b6.total_s - b6.exec_s) < 1e-9
    b2 = rs.simulate_session(tr, "B2_remaining_plus_install", 8, 2, cmodel)
    assert b2.sensing_s > 0.0 and b2.total_s > b2.exec_s
    cold = traces.make_trace("S8", "cold", 2, rs.session_duration_estimate("S8", "cold", 8, cmodel))
    assert rs.simulate_session(cold, "B6_zero_overhead", 8, 2, cmodel).install_s == 0.0
    assert rs.simulate_session(cold, "B5_oracle", 8, 2, cmodel).install_s > 0.0       # B5 pays the real install


def test_factorial_decider_reports_latency_and_gap(cmodel):
    dur = rs.session_duration_estimate("S0", "stable-WAN", 8, cmodel)
    tr = traces.make_trace("S0", "stable-WAN", 2, dur)
    r = rs.simulate_session(tr, "F:ILPAdapterxB4", 8, 2, cmodel, factorial=("ILPAdapter", "B4"))
    assert r.extra["planner"] == "ILPAdapter" and r.extra["control"] == "B4"
    assert r.extra["decisions"] == 9 and r.extra["solver_latency_mean_s"] >= 0.0
    assert r.extra["optimality_gap_mean_s"] == 0.0 and r.extra["identical_to_exact_rate"] == 1.0


def test_build_chain_sums_the_scheduled_steps(cmodel):
    """The factorial's planner chain prices each axis value over the remaining epochs on their own shapes."""
    T = 8
    sched = [rs._shapes_for(e, T, 0) for e in range(1 + T)]       # the prefill, then one-row decode steps
    catalog = cand.gpt2_catalog(sched[0])
    env = cm.Env(1e9, 0.0006)
    memo = {}
    chain_s, _ = rs.build_chain(catalog, sched[0], cmodel, env, 1 + T, schedule=sched, memo=memo)
    chain_h, _ = rs.build_chain(catalog, sched[0], cmodel, env, 1 + T)
    per_step = [rs.axis_costs(catalog["axes"], sh, cmodel, env) for sh in sched]
    for op_s, op_h in zip(chain_s, chain_h):
        axis = op_s["id"]
        for r_s, r_h in zip(op_s["realizations"], op_h["realizations"]):
            v = next(v for v in catalog["axes"][axis] if str(v) == r_s["name"])
            assert math.isclose(r_s["cost"], sum(p[axis][v] for p in per_step), rel_tol=1e-9)
            assert math.isclose(r_h["cost"], (1 + T) * per_step[0][axis][v], rel_tol=1e-9)
            assert r_s["objects"] == r_h["objects"]
    lin_s = {r["name"]: r["cost"] for r in chain_s[1]["realizations"]}
    lin_h = {r["name"]: r["cost"] for r in chain_h[1]["realizations"]}
    assert lin_s != lin_h                                           # the prefill shape is not the decode shape
    again, _ = rs.build_chain(catalog, sched[0], cmodel, env, 1 + T, schedule=sched, memo=memo)
    assert again == chain_s and any(k[0] == "axis" for k in memo)


def test_oracle_envs_follow_the_predicted_step_durations(cmodel):
    dur = rs.session_duration_estimate("S3", "two-changes", 8, cmodel)
    tr = traces.make_trace("S3", "two-changes", 2, dur)
    envs = rs._oracle_envs(tr, 1.5, [10.0, 5.0, 0.0], 3)
    assert sorted(envs) == [3, 4, 5]                               # keyed by the epoch number
    assert envs[3] == rs.env_from_state(traces.state_at(tr, 1.5))
    assert envs[4] == rs.env_from_state(traces.state_at(tr, 11.5))
    assert envs[5] == rs.env_from_state(traces.state_at(tr, 16.5))
    r = rs.simulate_session(tr, "B5_oracle", 8, 2, cmodel)
    assert r.tokens == 9 and r.decisions == 9
