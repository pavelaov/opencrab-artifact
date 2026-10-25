"""run_web tests: child environment with the thread cap, first-epoch and mismatch fields, record configuration,
relative paths; one tiny-model cold_warm stack end to end (three processes)."""

from __future__ import annotations

import json
import os
import pathlib

import pytest

from opencrab2.experiments import run_web, schema, workloads
from opencrab2.platform.launch import THREAD_ENV

PYDIR = pathlib.Path(__file__).resolve().parents[1]


def _epoch(epoch: int, plan: dict, switched: bool, **kw) -> dict:
    ev = {"_type": "epoch", "epoch": epoch, "gate": "prefill" if epoch == 0 else "decode", "plan": plan,
          "switched": switched, "decision_reason": "KEEP_IS_ARGMAX", "prepare_s": 8.4, "material_wait_s": 8.3,
          "install_s": 0.0, "exec_s": 12.2, "sensed_bw_bps": 3.9e9, "installed_now": [], "t": 1.0}
    ev.update(kw)
    return ev


def test_child_environment_carries_the_thread_cap():
    env = run_web._env(2)
    assert all(env[k] == "2" for k in THREAD_ENV) and env["HF_HUB_OFFLINE"] == "1"
    assert str(PYDIR) in env["PYTHONPATH"].split(os.pathsep)
    unset = run_web._env(None)
    assert all((k in unset) == (k in os.environ) for k in THREAD_ENV)       # no cap unless asked
    with pytest.raises(run_web.WebError, match="^WEB_PARAMS"):
        run_web._env(0)
    st = run_web.Stack(pathlib.Path("."), "npz:x.npz", 1, threads_per_process=3)
    assert st.threads_per_process == 3 and all(st.env()[k] == "3" for k in THREAD_ENV)
    assert run_web.Stack(pathlib.Path("."), "npz:x.npz", 1).threads_per_process == run_web.DEFAULT_THREADS_PER_PROCESS
    assert run_web.build_parser().get_default("threads_per_process") == run_web.DEFAULT_THREADS_PER_PROCESS == 2
    assert run_web.COLOCATED_PROCESSES == 3


def test_epoch0_summary_and_mismatch_helpers():
    wmask = {"embedding": "wmask", "linear": "wmask", "argmax_k": 2, "max_k": 2}
    generic = {"embedding": "generic", "linear": "generic", "argmax_k": 2, "max_k": 2}
    events = [{"_type": "status", "event": "handshake"}, _epoch(0, generic, True, decision_reason="LOWER_BOUND_GAIN"),
              {"_type": "token", "token_id": 5, "t": 2.0}, _epoch(1, wmask, True)]
    s = run_web.epoch0_summary(events)
    assert s["plan"] == generic and s["switched"] is True and s["decision_reason"] == "LOWER_BOUND_GAIN"
    assert s["material_wait_s"] == 8.3 and s["prepare_s"] == 8.4 and s["exec_s"] == 12.2 and s["install_s"] == 0.0
    assert s["plan_key"] == "argmax_k=2|embedding=generic|linear=generic|max_k=2" and s["sensed_bw_bps"] == 3.9e9
    assert set(s) == set(run_web.EPOCH0_KEYS) | {"plan_key"}
    empty = run_web.epoch0_summary([{"_type": "token", "token_id": 1}])
    assert all(v is None for v in empty.values())
    assert run_web._mismatch([1, 2, 3], [1, 2, 3], [0.5, 0.4, 0.3]) == (None, None)
    assert run_web._mismatch([1, 9, 3], [1, 2, 3], [0.5, 0.4, 0.3]) == (1, 0.4)
    assert run_web._mismatch([1, 2], [1, 2, 3], [0.5, 0.4, 0.3]) == (2, 0.3)          # a prefix: the missing step
    assert run_web._mismatch([1, 9], [1, 2, 3], None) == (1, None)
    assert run_web._mismatch(None, [1], [0.1]) == (None, None) and run_web._mismatch([1], None, None) == (None, None)


def test_web_record_carries_the_configuration_and_a_relative_path(tmp_path):
    cw = {"ttft_cold_s": 2.5, "ttft_warm_s": 1.5, "warm_verified": True, "completion": 1.0, "tokens": 4,
          "first_T_cold_s": 5.0, "second_T_warm_s": 3.0, "second_security_eligible": True, "aborted": False,
          "output_match": True, "second_epoch0_plan_key": "x", "second_epoch0_switched": False,
          "second_first_mismatch_step": None, "token_ids": [1, 2]}
    raw = tmp_path / "raw" / "web.jsonl"
    rec = run_web.web_record("cold_warm", 0, cw, 1, 4, str(raw), None, "tiny@abc", "B4_opencrab", "tiny",
                             threads_per_process=2, root=tmp_path)
    schema.validate(rec)
    assert rec["raw_measurement_path"] == "raw/web.jsonl" and rec["extra"]["threads_per_process"] == 2
    assert rec["extra"]["colocated_processes"] == 3 and "token_ids" not in rec["extra"]
    assert rec["extra"]["second_epoch0_switched"] is False and rec["extra"]["second_first_mismatch_step"] is None
    # the default (as the make_tables tests call it): the campaign default cap, a relative plain path
    rec2 = run_web.web_record("cold_warm", 0, cw, 1, 4, "x", None, "tiny@abc", "B4_opencrab", "tiny")
    assert rec2["extra"]["threads_per_process"] == run_web.DEFAULT_THREADS_PER_PROCESS and rec2["raw_measurement_path"] == "x"
    assert not pathlib.PureWindowsPath(rec["raw_measurement_path"]).is_absolute()
    assert "\\" not in rec["raw_measurement_path"]


def test_dry_run_reports_the_thread_cap(tmp_path, capsys):
    rc = run_web.main(["--dry-run", "--threads-per-process", "4", "--out-root", str(tmp_path)])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out["threads_per_process"] == 4 and out["colocated_processes"] == 3
    assert run_web.main(["--dry-run", "--threads-per-process", "0", "--out-root", str(tmp_path)]) == 2


def test_cold_warm_on_one_tiny_stack_end_to_end(tmp_path):
    """One stack (dealer, model server, sidecar), two generations: records carry the cap, the first-epoch
    path of both generations, the mismatch fields and relative paths; the second generation is warm."""
    pytest.importorskip("aiohttp")
    summary = run_web.run(["cold_warm"], "tiny", 2, 1, tmp_path, None, 1, "B4_opencrab", threads_per_process=1)
    assert summary["written"] == 1 and summary["threads_per_process"] == 1
    recs = schema.read_jsonl(tmp_path / "raw" / "web.jsonl")
    assert len(recs) == 1
    r = recs[0]
    e = r["extra"]
    assert r["aborts"] == 0, e.get("error")
    assert r["raw_measurement_path"] == "raw/web.jsonl" and r["status"] == "MEASURED_NATIVE"
    assert e["threads_per_process"] == 1 and e["colocated_processes"] == 3
    assert e["warm_verified"] is True and e["second_warm_start"] is True and r["T_warm_s"] == e["second_T_warm_s"]
    for label in ("first", "second"):
        assert isinstance(e[f"{label}_epoch0_plan"], dict) and isinstance(e[f"{label}_epoch0_switched"], bool)
        assert e[f"{label}_epoch0_exec_s"] > 0 and e[f"{label}_epoch0_material_wait_s"] >= 0
        assert e[f"{label}_epoch0_plan_key"] and e[f"{label}_epoch0_decision_reason"]
        assert e[f"{label}_first_mismatch_step"] is None and e[f"{label}_first_mismatch_margin"] is None
        assert set(e[f"{label}_epoch0_plan"]) == {"embedding", "linear", "argmax_k", "max_k"}
        assert isinstance(e[f"{label}_epoch0_installed_now"], list) and e[f"{label}_epoch0_install_s"] >= 0.0
        assert e[f"{label}_epoch0_sensed_bw_bps"] > 0
    # the record says which path each generation took (the tiny model's products are small, so the controller
    # may keep the generic plan on both; the warm request installs nothing it already holds)
    assert e["second_epoch0_installed_now"] == [] and r["output_match"] is True
    assert e["output_match_first"] is True and e["output_match_second"] is True
    work = pathlib.Path(summary["work"])
    assert (work / "tiny_cold_warm_r0" / "sidecar.log").is_file()
    assert "generated_tokens" not in json.dumps(r)
    # the tiny prompt of the workload has a clear margin at every step (the reference is computable)
    W, meta = workloads.make_tiny_weights(1)
    assert workloads.reference_greedy_tokens(W, meta, 1, list(workloads.TINY_PROMPT), 2)
