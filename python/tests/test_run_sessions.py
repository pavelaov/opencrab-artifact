"""run_sessions, workloads and run_campaign: matrices, prediction, stamping, resumption, a two-worker tiny run."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from opencrab2.experiments import run_sessions as rs
from opencrab2.experiments import schema, workloads
from opencrab2.highlevel import costmodel as cm

PYDIR = Path(__file__).resolve().parents[1]


def test_workloads_tiny_model_and_reference_match_the_session_tests():
    from tests.test_session import PROMPT, make_tiny_weights, reference_tokens
    W, meta = workloads.make_tiny_weights(1)
    W2, meta2 = make_tiny_weights(1, d=64, vocab=256)
    assert meta == meta2 and all((W[k] == W2[k]).all() for k in W)
    ref = workloads.reference_greedy_tokens(W, meta, 1, PROMPT, 3)
    assert ref == reference_tokens(W2, meta2, 1, PROMPT, 3) and list(workloads.TINY_PROMPT) == list(PROMPT)
    margins = workloads.reference_margins(W, meta, 1, PROMPT, 3)
    assert len(margins) == 3 and all(m >= 0 for m in margins)
    with pytest.raises(ValueError):
        workloads.reference_greedy_tokens(W, meta, 1, [], 3)
    assert workloads.scope_for_layers(12) == "full-model" and workloads.scope_for_layers(4) == "prefix-4L"
    with pytest.raises(ValueError):
        workloads.scope_for_layers(7)


def test_diag4_matrix_cells_policies_and_order():
    cells = rs.build_matrix("diag4")
    assert len(cells) == 117 and len({c.run_id for c in cells}) == 117
    assert {c.n_layers for c in cells} == {4} and {c.T for c in cells} == {8} and {c.seed for c in cells} == {1, 2, 3}
    assert {c.scope for c in cells} == {"prefix-4L"}
    traces_ = {(c.family, c.variant) for c in cells}
    assert traces_ == {("S0", "stable-LAN"), ("S0", "stable-WAN"), ("S1", "ramp-WAN-c-to-LAN"), ("S2", "rtt-step-up"),
                       ("S3", "two-changes"), ("S8", "cold-LAN"), ("S8", "warm-all-LAN")}
    base = {"B0_fixed:wmask", "B0_fixed:generic", "B1_naive", "B2_remaining_plus_install", "B4_opencrab"}
    extra = {"B4_no_install", "delayed:2"}
    for fam, var in traces_:
        for seed in (1, 2, 3):
            pols = [c.policy for c in cells if (c.family, c.variant, c.seed) == (fam, var, seed)]
            assert set(pols) == (base | extra if fam in ("S1", "S3") else base) and len(pols) == len(set(pols))
    # the policy order is randomized per seed and deterministic
    orders = {seed: [c.policy for c in cells if (c.family, c.variant, c.seed) == ("S1", "ramp-WAN-c-to-LAN", seed)]
              for seed in (1, 2, 3)}
    assert len({tuple(o) for o in orders.values()}) > 1
    assert [c.run_id for c in rs.build_matrix("diag4")] == [c.run_id for c in cells]
    warm = [c for c in cells if c.warm]
    assert {c.variant for c in warm} == {"warm-all-LAN"} and len(warm) == 15
    # the S8 pair runs on the same LAN link
    from opencrab2.experiments import traces
    cold_t = traces.make_trace("S8", "cold-LAN", 1, 30.0)
    warm_t = traces.make_trace("S8", "warm-all-LAN", 1, 30.0)
    assert cold_t["points"] == warm_t["points"] and cold_t["points"][0][1:3] == list(traces.ANCHORS["LAN"])


def test_full12_and_smoke_matrices():
    cells = rs.build_matrix("full12")
    assert len(cells) == 18 and {c.n_layers for c in cells} == {12} and {c.scope for c in cells} == {"full-model"}
    assert {c.policy for c in cells} == {"B0_fixed:wmask", "B2_remaining_plus_install", "B4_opencrab"}
    assert {(c.family, c.variant) for c in cells} == {("S0", "stable-LAN"), ("S1", "ramp-WAN-c-to-LAN")}
    assert len(rs.build_matrix("smoke")) == 2
    with pytest.raises(ValueError):
        rs.build_matrix("nope")
    with pytest.raises(ValueError):
        rs.run_matrix("smoke", Path("."), 10.0, dry_run=True, workers=0)


def test_dry_run_prints_cells_with_predicted_durations(tmp_path, capsys):
    rc = rs.main(["--matrix", "full12", "--dry-run", "--workers", "3", "--out-root", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0 and out.count("predicted") >= 18 and "wall at 3 worker(s)" in out and "MODELED" in out
    assert "no completed session at this worker count yet" in out
    assert not (tmp_path / "raw").exists()                                    # a dry run writes nothing
    summary = rs.run_matrix("diag4", tmp_path, 10.0, dry_run=True, workers=4)
    assert len(summary["todo"]) == 117 and summary["predicted"]["status"] == "MODELED"
    by_id = {r["run_id"]: r for r in summary["todo"]}
    cold = by_id["session:diag4:S8:cold-LAN:B0_fixed:wmask:T8:L4:s1"]
    warm = by_id["session:diag4:S8:warm-all-LAN:B0_fixed:wmask:T8:L4:s1"]
    generic = by_id["session:diag4:S8:cold-LAN:B0_fixed:generic:T8:L4:s1"]
    assert cold["install_s"] > 0 and warm["install_s"] == 0.0 and warm["priming_s"] > 0 and cold["priming_s"] == 0.0
    assert warm["online_s"] == pytest.approx(cold["online_s"]) and generic["predicted_plan"]["linear"] == "generic"
    assert generic["install_s"] < cold["install_s"]                           # only the tied head is installed
    assert summary["predicted"]["wall_seconds_at_workers"] <= summary["predicted"]["sum_cell_seconds"]
    assert summary["predicted"]["wall_seconds_at_workers"] >= summary["predicted"]["sum_cell_seconds"] / 4.0 - 1e-6
    # a measured cell has ONE declared trace duration, whatever the policy predicts; the ramp spans it
    cell = rs.Cell("diag4", "S1", "ramp-WAN-c-to-LAN", "B4_opencrab", 1, 4, 8)
    tr = rs.cell_trace(cell, cm.default_model(), tmp_path / "traces")
    assert tr["duration_s"] == rs.TRACE_DURATION_S[("diag4", "S1", "ramp-WAN-c-to-LAN")] == 770.0
    assert rs.trace_duration_s(cell) == (770.0, rs.TRACE_RULE_FIXED)
    assert tr["points"][0][1] < tr["points"][-1][1] and tr["trace_id"].endswith(":d770")
    # the smoke matrix keeps the prediction rule
    smoke = rs.Cell("smoke", "S0", "stable-LAN", "B4_opencrab", 1, 1, 2)
    pred = rs.predict_cell(smoke, cm.default_model())
    dur, rule = rs.trace_duration_s(smoke, cm.default_model())
    assert rule == rs.TRACE_RULE_PREDICTED
    assert dur == pytest.approx(max(10.0, rs.TRACE_SAFETY * (pred["predicted_total_s"] - pred["priming_s"])))
    assert rs.cell_trace(smoke, cm.default_model(), tmp_path / "traces")["duration_s"] == pytest.approx(dur)


def test_every_measured_cell_shares_one_trace_across_its_policies(tmp_path):
    """All policies of one (matrix, trace cell, seed) run on the same trace: same id, same hash, declared duration."""
    for matrix in ("diag4", "full12"):
        cells = rs.build_matrix(matrix)
        groups = {}
        for c in cells:
            groups.setdefault((c.family, c.variant, c.seed), []).append(c)
        assert all((matrix, fam, var) in rs.TRACE_DURATION_S for fam, var, _s in groups)
        for (fam, var, seed), members in groups.items():
            trs = [rs.cell_trace(c, None, tmp_path / "traces") for c in members]
            assert len(members) >= 3 and len({(t["trace_id"], t["trace_hash"]) for t in trs}) == 1, (matrix, fam, var, seed)
            assert trs[0]["duration_s"] == rs.TRACE_DURATION_S[(matrix, fam, var)]
            assert trs[0]["trace_id"] == f"{fam}:{var}:seed{seed}:d{rs.TRACE_DURATION_S[(matrix, fam, var)]:g}"
    # the declared values: about 1.2 x the median session of the first campaign, rounded to 10 s; S2 and S3
    # use 440 s so that their change points fall inside the session (at most 330 s after its start)
    assert all(d > 0 and d % 10 == 0 for d in rs.TRACE_DURATION_S.values())
    assert rs.TRACE_DURATION_S[("diag4", "S2", "rtt-step-up")] == 440.0
    assert rs.TRACE_DURATION_S[("diag4", "S3", "two-changes")] == 440.0
    assert rs.TRACE_DURATION_S[("full12", "S1", "ramp-WAN-c-to-LAN")] == 1550.0
    s2 = rs.cell_trace(rs.Cell("diag4", "S2", "rtt-step-up", "B4_opencrab", 1, 4, 8), None, tmp_path / "traces")
    assert [p[0] for p in s2["points"]] == [0.0, 110.0, 220.0, 330.0]
    for seed in (1, 2, 3):
        s3 = rs.cell_trace(rs.Cell("diag4", "S3", "two-changes", "B4_opencrab", seed, 4, 8), None, tmp_path / "traces")
        assert len(s3["meta"]["change_times_s"]) == 2 and max(s3["meta"]["change_times_s"]) < 300.0
    # a stamped record says which rule produced its trace
    cell = rs.Cell("diag4", "S8", "warm-all-LAN", "B4_opencrab", 2, 4, 8)
    tr = rs.cell_trace(cell, None, tmp_path / "traces")
    raw = schema.new_record(status="MEASURED_NATIVE", record_kind="session", dealer_mode="ISOLATED_RECIPIENT_BUNDLES",
                            workload_scope="prefix-4L", clock_basis="wall-clock-exogenous", policy_id="B4_opencrab",
                            run_id="x", extra={"party": 1})
    rec = rs.stamp(raw, cell, tr, "raw.jsonl", {}, root=tmp_path)
    assert rec["extra"]["trace_duration_s"] == 430.0 and rec["extra"]["trace_duration_rule"] == rs.TRACE_RULE_FIXED
    assert rec["trace_id"] == tr["trace_id"] and rec["trace_hash"] == tr["trace_hash"]


def test_stamp_abort_and_resume_bookkeeping(tmp_path):
    cell = rs.Cell("diag4", "S0", "stable-LAN", "B0_fixed:wmask", 2, 4, 8)
    tr = {"trace_id": "t-1", "trace_hash": "h" * 64, "split": "test", "duration_s": 12.0}
    detail = tmp_path / "raw" / "sessions" / "abcd" / "sessions.whatever.p1.detail.jsonl"
    raw = schema.new_record(status="MEASURED_NATIVE", record_kind="session", dealer_mode="ISOLATED_RECIPIENT_BUNDLES",
                            workload_scope="prefix-4L", clock_basis="wall-clock-exogenous", policy_id="B0_fixed",
                            run_id="whatever", T_cold_s=3.0, T_warm_s=2.0, T_online_s=1.0, tokens=8, output_match=False,
                            raw_measurement_path=str(detail),
                            extra={"party": 1, "session_wall_s": 3.0, "generated_tokens": [1, 2, 3],
                                   "reference_tokens": [1, 7, 3], "first_mismatch_step": 1})
    raw["paths"] = {"record": str(detail.parent / "sessions.jsonl"), "detail": str(detail)}
    rec = rs.stamp(raw, cell, tr, "raw.jsonl", {"concurrent_workers": 4, "threads_per_process": 2, "ports": {"p0": 1},
                                                 "reference_margins": [2.0, 0.25, 1.0]}, root=tmp_path)
    schema.validate(rec)
    assert rec["run_id"] == cell.run_id + ":p1" and rec["policy_id"] == "B0_fixed:wmask"
    assert rec["extra"]["runner_policy_id"] == "B0_fixed" and rec["extra"]["matrix"] == "diag4"
    assert rec["extra"]["concurrent_workers"] == 4 and rec["extra"]["threads_per_process"] == 2
    # the client record keeps what a mismatch report needs: tokens, reference, first differing step and its margin
    assert rec["extra"]["generated_tokens"] == [1, 2, 3] and rec["extra"]["reference_tokens"] == [1, 7, 3]
    assert rec["extra"]["first_mismatch_step"] == 1 and rec["extra"]["first_mismatch_margin"] == 0.25
    # paths relative to the data root, forward slashes, never a host path
    assert rec["raw_measurement_path"] == "raw/sessions/abcd/sessions.whatever.p1.detail.jsonl"
    assert rec["extra"]["detail_path"] == rec["raw_measurement_path"]
    assert rec["extra"]["launcher"] == {"paths": {"record": "raw/sessions/abcd/sessions.jsonl",
                                                  "detail": "raw/sessions/abcd/sessions.whatever.p1.detail.jsonl"}}
    assert rec["extra"]["trace_duration_rule"] == rs.TRACE_RULE_FIXED
    assert rec["family"] == "S0" and rec["seed"] == 2 and rec["n_layers"] == 4 and rec["workload_scope"] == "prefix-4L"
    matched = rs.stamp(dict(raw, extra={"party": 1, "first_mismatch_step": None}), cell, tr, "raw.jsonl",
                       {"reference_margins": [2.0, 0.25, 1.0]}, root=tmp_path)
    assert matched["extra"]["first_mismatch_margin"] is None
    path = tmp_path / "raw" / "sessions.jsonl"
    schema.append_jsonl(path, rec)
    other = rs.Cell("diag4", "S0", "stable-LAN", "B1_naive", 2, 4, 8)
    ab = rs.aborted_record(other, tr, "ChannelError: LAUNCH_FAILED", str(path), {"concurrent_workers": 4,
                                                                                 "threads_per_process": 2}, root=tmp_path)
    schema.append_jsonl(path, ab)
    assert ab["aborts"] == 1 and ab["security_eligible"] is False and ab["extra"]["concurrent_workers"] == 4
    assert ab["raw_measurement_path"] == "raw/sessions.jsonl"
    # only the P1 record of `cell` is there: the cell is NOT done (its P0 record would be lost for good)
    assert rs.completed_cells(path) == {other.run_id}
    assert rs.completed_cells(path, retry_aborted=True) == set()
    rec0 = dict(rec, run_id=cell.run_id + ":p0", output_match=None, extra=dict(rec["extra"], party=0))
    schema.append_jsonl(path, rec0)
    assert rs.completed_cells(path) == {cell.run_id, other.run_id}
    assert rs.completed_cells(path, retry_aborted=True) == {cell.run_id}
    summary = rs.run_matrix("diag4", tmp_path, 10.0, dry_run=True, only=["S0:stable-LAN"])
    assert summary["cells"] == 15 and summary["already_done"] == 2
    summary = rs.run_matrix("diag4", tmp_path, 10.0, dry_run=True, only=["S0:stable-LAN"], retry_aborted=True)
    assert summary["already_done"] == 1


def test_resume_with_one_party_record_and_with_a_torn_last_line(tmp_path):
    # Regression: a file holding only "<run_id>:p0" counted the cell as done, so the client
    # record was never produced; a partial last line stopped every resume with SCHEMA_BAD_VALUE; and the next
    # append was glued onto the fragment.
    cell = rs.Cell("diag4", "S8", "cold-LAN", "B0_fixed", 1, 4, 8)
    path = tmp_path / "raw" / "sessions.jsonl"
    base = schema.new_record(status="MEASURED_NATIVE", record_kind="session", dealer_mode="ISOLATED_RECIPIENT_BUNDLES",
                             workload_scope="prefix-4L", clock_basis="wall-clock-exogenous", policy_id="B0_fixed",
                             run_id=cell.run_id + ":p0", tokens=8, extra={"party": 0, "matrix": "diag4"})
    schema.append_jsonl(path, base)
    assert rs.completed_cells(path) == set()
    todo = rs.run_matrix("diag4", tmp_path, 10.0, dry_run=True, only=["S8:cold-LAN"])
    assert todo["already_done"] == 0
    # a torn last line: skipped by the resume bookkeeping, refused by the strict reader
    with open(path, "ab") as fh:
        fh.write(b'{"run_id": "session:diag4:S8:cold-LAN:B0_fixed:generic:T8:L4:s1:p1", "sta')
    assert rs.completed_cells(path) == set()
    assert len(schema.read_jsonl(path, strict=False)) == 1
    with pytest.raises(ValueError, match="^SCHEMA_BAD_VALUE"):
        schema.read_jsonl(path, strict=True)
    # the next append starts on a new line: the record is intact, the fragment stays a line of its own
    p1 = dict(base, run_id=cell.run_id + ":p1", extra={"party": 1, "matrix": "diag4"})
    schema.append_jsonl(path, p1)
    raw = path.read_bytes().split(b"\n")
    assert raw[-1] == b"" and json.loads(raw[-2])["run_id"].endswith(":p1") and raw[1].endswith(b'"sta')
    # an unparsable line that is NOT the last one is corruption in both modes
    with pytest.raises(ValueError, match="^SCHEMA_BAD_VALUE"):
        schema.read_jsonl(path, strict=False)
    with pytest.raises(ValueError, match="^SCHEMA_BAD_VALUE"):
        rs.completed_cells(path)
    # without the fragment: both party records present, the cell is done
    clean = tmp_path / "raw2" / "sessions.jsonl"
    schema.append_jsonl(clean, base)
    schema.append_jsonl(clean, p1)
    assert rs.completed_cells(clean) == {cell.run_id}
    schema.locked_append_lines(tmp_path / "x.jsonl", ['{"a":1}'])
    with open(tmp_path / "x.jsonl", "ab") as fh:
        fh.write(b'{"b":{"c":3}')
    schema.locked_append_lines(tmp_path / "x.jsonl", ['{"d":4}'])
    nl = bytes([10])
    assert (tmp_path / "x.jsonl").read_bytes() == b'{"a":1}' + nl + b'{"b":{"c":3}' + nl + b'{"d":4}' + nl


def test_smoke_matrix_tiny_model_two_workers_end_to_end(tmp_path):
    """Two concurrent three-process sessions into one output file; then a rerun skips both cells."""
    summary = rs.run_matrix("smoke", tmp_path, 600.0, workers=2, threads_per_process=1, model="tiny")
    assert summary["run"] == 2 and summary["aborted"] == 0, summary
    recs = schema.read_jsonl(tmp_path / "raw" / "sessions.jsonl")
    assert len(recs) == 4 and sorted(r["run_id"][-3:] for r in recs) == [":p0", ":p0", ":p1", ":p1"]
    ports = set()
    for r in recs:
        e = r["extra"]
        assert r["status"] == "MEASURED_NATIVE" and r["security_eligible"] is True and r["aborts"] == 0
        assert e["concurrent_workers"] == 2 and e["threads_per_process"] == 1 and e["matrix"] == "smoke"
        assert r["output_match"] is (True if r["extra"]["party"] == 1 else None) and r["tokens"] == 2 and len(e["reference_margins"]) == 2
        if e["party"] == 1:
            # the client record reports a mismatch on its own: tokens, reference, first differing step
            assert len(e["generated_tokens"]) == 2 and e["generated_tokens"] == e["reference_tokens"]
            assert e["first_mismatch_step"] is None and e["first_mismatch_margin"] is None
        else:
            assert "generated_tokens" not in json.dumps(r)                   # the server never holds a token
        assert r["workload_scope"] == "prefix-1L" and e["model"] == "tiny" and e["party_rng"] == "os-entropy"
        # relative to the data root with forward slashes, never a host path
        assert r["raw_measurement_path"].startswith("raw/sessions/") and "\\" not in r["raw_measurement_path"]
        assert (tmp_path / r["raw_measurement_path"]).is_file()
        assert e["trace_duration_rule"] == rs.TRACE_RULE_PREDICTED
        ports.add((e["ports"]["dealer"], e["ports"]["p0"]))
    assert len(ports) == 2 and all(len(set(p)) == 2 for p in ports)           # each session had its own ports
    p1 = [r for r in recs if r["extra"]["party"] == 1][0]
    detail = [json.loads(l) for l in open(tmp_path / p1["raw_measurement_path"], encoding="utf-8")]
    assert detail[-1]["generated_tokens"] is not None and detail[-1]["output_match"] is True
    again = rs.run_matrix("smoke", tmp_path, 600.0, workers=2, threads_per_process=1, model="tiny")
    assert again["already_done"] == 2 and again["run"] == 0
    assert len(schema.read_jsonl(tmp_path / "raw" / "sessions.jsonl")) == 4
    obs = rs.observed_ratio(tmp_path / "raw" / "sessions.jsonl", cm.default_model(), None, 2)
    assert obs is not None and obs["n"] == 2 and obs["median"] > 0


def test_run_campaign_stages_logs_and_resume(tmp_path):
    script = str(PYDIR.parent / "scripts" / "run_campaign.py")
    env = dict(os.environ, PYTHONPATH=str(PYDIR))

    def call(*args):
        return subprocess.run([sys.executable, script, "--out-root", str(tmp_path)] + list(args), cwd=str(PYDIR), env=env,
                              capture_output=True, text=True, timeout=600)

    bad = call("--stage", "nope")
    assert bad.returncode == 2 and "CAMPAIGN_STAGE_UNKNOWN" in bad.stderr
    # the shipped results/ is never a target unless named itself
    inside = subprocess.run([sys.executable, script, "--out-root", str(PYDIR.parent / "results" / "raw"), "--stage", "tables",
                             "--dry-run"], cwd=str(PYDIR), env=env, capture_output=True, text=True, timeout=600)
    assert inside.returncode == 2 and "CAMPAIGN_OUT_ROOT" in inside.stderr
    dry = call("--stage", "all", "--dry-run", "--workers", "4")
    assert dry.returncode == 0, dry.stdout + dry.stderr
    for mod in ("run_primitives", "run_calibration", "run_simulation", "make_tables"):
        assert f"opencrab2.experiments.{mod}" in dry.stdout
    # without a C++ build the benchmark stage is skipped in a dry run (and named as such)
    rc = _campaign_module()
    assert ("run_cpp_bench.py" in dry.stdout) if rc.cpp_binaries_present() else ("cpp_bench: skipped, no C++ build" in dry.stdout)
    assert "run_web --model gpt2 --repeats 3 --trials 10 --threads-per-process 2" in dry.stdout
    # the paper's campaign (no --stage) does not repeat the calibration; --quick runs every stage
    camp = call("--dry-run")
    assert camp.returncode == 0 and "opencrab2.experiments.run_calibration" not in camp.stdout
    assert "opencrab2.experiments.run_simulation" in camp.stdout and "--matrix diag4 --workers 3 " in camp.stdout
    assert "opencrab2.experiments.run_calibration" in call("--dry-run", "--quick", "--stage", "all").stdout
    assert "--matrix diag4 --workers 4 --threads-per-process 2 --timeout 7200.0" in dry.stdout
    assert "--matrix full12 --workers 3" in dry.stdout
    assert "--skip S0:stable-WAN/B0_fixed:generic,S1/B0_fixed:generic" in dry.stdout and "--skip S1/B2_remaining_plus_install" in dry.stdout
    assert dry.stdout.count("predicted") > 100 and not (tmp_path / "raw" / "sessions.jsonl").exists()
    assert not (tmp_path / "logs" / "campaign_state.json").exists()            # a dry run records no stage as done
    quick = call("--stage", "diag4,full12,web", "--dry-run", "--quick")
    assert quick.returncode == 0 and "--matrix smoke --model tiny" in quick.stdout and "full12: skipped" in quick.stdout
    assert "run_web --model tiny --repeats 1" in quick.stdout
    run1 = call("--stage", "primitives,tables", "--quick")
    assert run1.returncode == 0, run1.stdout + run1.stderr
    state = json.loads((tmp_path / "logs" / "campaign_state.json").read_text(encoding="utf-8"))
    assert state["primitives"]["status"] == "done" and state["tables"]["status"] == "done"
    assert sys.executable not in json.dumps(state["primitives"]["signature"])       # resumable on another host
    assert os.path.isfile(state["primitives"]["log"]) and "exit 0" in open(state["primitives"]["log"], encoding="utf-8").read()
    assert (tmp_path / "raw" / "primitives.jsonl").is_file()
    assert all(r["raw_measurement_path"] == "raw/primitives.jsonl" for r in schema.read_jsonl(tmp_path / "raw" / "primitives.jsonl"))
    from opencrab2.experiments import make_tables as mt
    assert sorted(p.stem for p in (tmp_path / "tables").glob("*.tex")) == sorted(mt.TABLE_NAMES)
    n_before = len(schema.read_jsonl(tmp_path / "raw" / "primitives.jsonl"))
    run2 = call("--stage", "primitives,tables", "--quick")
    assert run2.returncode == 0 and "primitives: already done" in run2.stdout
    assert len(schema.read_jsonl(tmp_path / "raw" / "primitives.jsonl")) == n_before


def _campaign_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_campaign_under_test", PYDIR.parent / "scripts" / "run_campaign.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_campaign_root_child_environment_and_paper_defaults(tmp_path, monkeypatch):
    rc = _campaign_module()
    tree = PYDIR.parent
    # runs/<name> by default; the shipped results/ only when named itself, never a directory inside it
    assert rc.resolve_root(None, "quick") == (tree / "runs" / "quick").resolve()
    assert rc.resolve_root(None, "full") == (tree / "runs" / "full").resolve()
    assert rc.resolve_root(str(tree / "results"), "full") == (tree / "results").resolve()
    assert rc.resolve_root(str(tmp_path), "full") == tmp_path.resolve()
    with pytest.raises(ValueError, match="^CAMPAIGN_OUT_ROOT"):
        rc.resolve_root(str(tree / "results" / "tables"), "full")
    with pytest.raises(ValueError, match="^CAMPAIGN_NAME"):
        rc.resolve_root(None, "../results")
    # every child gets the root, the package path and a proxy bypass for loopback
    for var in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(var, "example.org")
    env = rc.child_env(tmp_path)
    assert env["OPENCRAB2_RESULTS_DIR"] == str(tmp_path) and "HF_HUB_OFFLINE" in env
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(tree / "python")
    for var in (("NO_PROXY",) if os.name == "nt" else ("NO_PROXY", "no_proxy")):
        assert {"example.org", "127.0.0.1", "localhost"} <= set(env[var].split(","))
    assert [k for k in env if k.upper() == "NO_PROXY"] == (["NO_PROXY"] if os.name == "nt" else ["NO_PROXY", "no_proxy"])
    # the flags of the paper's campaign are the defaults
    args = rc.build_parser().parse_args([])
    assert (args.workers, args.full12_workers, args.simulation_workers) == (3, 3, 16)
    assert rc.CAMPAIGN_STAGES == tuple(s for s in rc.STAGES if s != "calibration") and args.stage is None
    assert (args.threads_per_process, args.timeout) == (2, 7200.0)
    assert args.diag4_skip == "S0:stable-WAN/B0_fixed:generic,S1/B0_fixed:generic,S2/B0_fixed:generic,S3/B0_fixed:generic"
    assert args.full12_skip == "S1/B2_remaining_plus_install"
    assert (args.web_model, args.web_repeats, args.web_trials) == ("gpt2", 3, 10)
    web = rc.stage_command("web", args, tmp_path)
    assert web[web.index("--threads-per-process") + 1] == "2" and web[web.index("--model") + 1] == "gpt2"
    # the done check compares neither the interpreter nor the location of the tree or the root
    sig = " ".join(rc.signature(rc.stage_command("cpp_bench", args, tmp_path), tmp_path))
    assert sys.executable not in sig and str(tree) not in sig and str(tmp_path) not in sig


def test_tiny_prompt_has_a_clear_reference_margin_for_every_matrix_shape():
    # diag4 --model tiny (4 layers, 8 tokens): TINY_PROMPT has a first margin of 0.012 logit, below the
    # contract error, so the cell could not serve as a correctness check
    W4, meta4 = workloads.make_tiny_weights(4)
    assert min(workloads.reference_margins(W4, meta4, 4, workloads.TINY_PROMPT, 8)) < 0.05
    prompt, margin = workloads.tiny_prompt(4, 8)
    assert len(prompt) == len(workloads.TINY_PROMPT) and all(0 <= t < 256 for t in prompt)
    assert margin >= workloads.TINY_MIN_MARGIN
    assert min(workloads.reference_margins(W4, meta4, 4, prompt, 8)) == pytest.approx(margin)
    assert workloads.tiny_prompt(4, 8) == (prompt, margin)                       # deterministic
    # the one-layer smoke shape keeps the pinned prompt
    assert workloads.tiny_prompt(1, 2)[0] == tuple(workloads.TINY_PROMPT)
    cell = rs.Cell("diag4", "S8", "cold-LAN", "B0_fixed:generic", 1, 4, 8, "tiny")
    assert rs.cell_workload(cell)["prompt_ids"] == list(prompt)
    smoke = rs.Cell("smoke", "S0", "stable-LAN", "B4_opencrab", 1, 1, 2, "tiny")
    assert rs.cell_workload(smoke)["prompt_ids"] == list(workloads.TINY_PROMPT)
