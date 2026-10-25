"""Three-process launch test: dealer service + P0 + P1 as OS processes.

The tiny random-weight model of tests/test_session.py is stored as an ``.npz``
and loaded by the P0 process; the dealer streams in ISOLATED_RECIPIENT_BUNDLES
mode; both evidence records must validate, be security eligible and share the
exogenous trace identity and clock origin.
"""

from __future__ import annotations

import json
import os

import pytest

from opencrab2.highlevel import session as S
from opencrab2.highlevel.contract import DEFAULT_CONTRACT
from opencrab2.highlevel.dealer import MODE_ISOLATED
from opencrab2.platform import launch as L
from opencrab2.platform import shaper as SH
from tests.test_session import PROMPT, make_tiny_weights, reference_tokens

C = DEFAULT_CONTRACT


def _records(path: str):
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    try:
        from opencrab2.experiments import schema          # type: ignore
        for r in recs:
            schema.validate(r)
    except ImportError:
        for r in recs:
            S.validate_record(r)
    return recs


def test_launch_two_party_three_processes(tmp_path):
    W, meta = make_tiny_weights(1, d=64, vocab=256)
    npz = S.save_model_npz(str(tmp_path / "tiny.npz"), W, meta)
    ref = reference_tokens(W, meta, 1, PROMPT, 3)
    trace = {"trace_id": "S0-stable-launch", "family": "S0", "split": "test", "seed": 4,
             "points": [(0.0, 5e8, 0.0002, 1.0, None), (60.0, 5e8, 0.0002, 1.0, None)]}
    cfg = S.SessionConfig(model=f"npz:{npz}", n_layers=1, prompt_ids=list(PROMPT), n_generate=3,
                          policy="B4_opencrab", trace_id=trace["trace_id"], seed=5, contract=C, length_bucket=8,
                          session_id="launch-1", reference_tokens=ref)
    res = L.launch_two_party(cfg, trace, str(tmp_path / "out"), timeout_s=500.0)
    assert sorted(res["summaries"]) == [0, 1] and not res["errors"]
    assert all(s["completed"] and not s["cancelled"] and s["epochs"] == 3 for s in res["summaries"].values())
    assert res["summaries"][0]["origin_wall_s"] == res["summaries"][1]["origin_wall_s"] == res["origin_wall_s"]
    assert res["summaries"][1]["generated_tokens"] == ref
    recs = _records(res["records"][0])
    assert len(recs) == 2
    for r in recs:
        assert r["dealer_mode"] == MODE_ISOLATED and r["status"] == "MEASURED_NATIVE"
        assert r["security_eligible"] is True and r["tokens"] == 3 and r["aborts"] == 0
        assert r["trace_id"] == trace["trace_id"] and r["trace_hash"] == SH.trace_hash(trace)
        assert r["T_online_s"] <= r["T_cold_s"] and r["material_consumed"] > 0
        assert r["output_match"] is (True if r["extra"]["party"] == 1 else None) and r["policy_id"] == "B4_opencrab"
        # the record names its detail file without a host path; the launcher result holds the real paths
        assert "\\" not in r["raw_measurement_path"] and not os.path.isabs(r["raw_measurement_path"])
        assert r["raw_measurement_path"] == os.path.basename(r["raw_measurement_path"])
        if r["extra"]["party"] == 1:
            assert r["extra"]["generated_tokens"] == ref and r["extra"]["reference_tokens"] == ref
            assert r["extra"]["first_mismatch_step"] is None
        else:
            assert "generated_tokens" not in r["extra"] and "reference_tokens" not in r["extra"]
    assert sorted(os.path.basename(p) for p in os.listdir(tmp_path / "out") if p.endswith(".detail.jsonl")) == \
        sorted(r["raw_measurement_path"] for r in recs)
    assert recs[0]["initial_state_hash"] == recs[1]["initial_state_hash"]
    # the per-recipient keys file never outlives the launch; the dealer log carries ids and sizes only
    assert not [p for p in os.listdir(tmp_path / "out") if p.startswith("dealer-keys")]
    log = open(res["dealer_log"], encoding="utf-8").read()
    assert "delivered recipient=0" in log and "delivered recipient=1" in log and "seed" not in log.lower()
    # two ledgers, no RESERVED entry left behind
    for p in (0, 1):
        lines = [json.loads(l) for l in open(tmp_path / "out" / f"launch-1-ledger{p}.jsonl", encoding="utf-8") if l.strip()]
        states = {}
        for rec in lines:
            if rec.get("kind") == "STATE":
                states[rec["material_id"]] = rec["state"]
        assert states and "RESERVED" not in set(states.values())


def test_launch_rejects_bad_config(tmp_path):
    from opencrab2.errors import ContractViolation
    cfg = S.SessionConfig(model="npz:missing.npz", n_layers=1, prompt_ids=[], n_generate=0, policy="B1_naive",
                          trace_id="t", seed=1, contract=C)
    with pytest.raises(ContractViolation):
        L.launch_two_party(cfg, None, str(tmp_path))


def test_launch_warm_prime_reference_and_thread_caps(tmp_path):
    """Priming session then a measured warm session on the same processes; reference computed by the launcher."""
    W, meta = make_tiny_weights(1, d=64, vocab=256)
    npz = S.save_model_npz(str(tmp_path / "tiny.npz"), W, meta)
    ref = reference_tokens(W, meta, 1, PROMPT, 2)
    cfg = S.SessionConfig(model=f"npz:{npz}", n_layers=1, prompt_ids=list(PROMPT), n_generate=2,
                          policy="B2_remaining_plus_install", trace_id="unshaped", seed=3, contract=C, length_bucket=8,
                          session_id="launch-warm", run_id="session:test:S8:warm:B2:s3")
    before = {k: os.environ.get(k) for k in L.THREAD_ENV}
    res = L.launch_two_party(cfg, None, str(tmp_path / "out"), timeout_s=500.0, threads_per_process=1, warm_prime=True)
    assert {k: os.environ.get(k) for k in L.THREAD_ENV} == before          # the launcher's own environment is restored
    assert cfg.reference_tokens == ref and len(res["reference_margins"]) == 2 and min(res["reference_margins"]) > 0
    assert res["warm_prime"] is True and res["threads_per_process"] == 1
    assert res["ports"]["dealer"] != res["ports"]["p0"] and res["ports"]["p0"] > 0
    assert res["summaries"][1]["generated_tokens"] == ref and res["summaries"][1]["output_match"] is True
    assert res["summaries"][0]["generated_tokens"] is None                  # P0 never holds the client's tokens
    assert res["summaries"][1]["priming"]["completed"] and "wte" in res["summaries"][1]["priming"]["installed"]
    recs = _records(res["records"][0])
    assert len(recs) == 2
    for r in recs:
        assert r["run_id"] == "session:test:S8:warm:B2:s3" and r["security_eligible"] is True and r["aborts"] == 0
        assert r["extra"]["warm_start"] is True and r["T_cold_s"] is None and r["T_warm_s"] > 0
        assert r["extra"]["T_warm_basis"] == "measured-warm-request" and r["initializations"] == 0
        assert r["output_match"] is (True if r["extra"]["party"] == 1 else None) and r["tokens"] == 2
        assert ("generated_tokens" in r["extra"]) is (r["extra"]["party"] == 1)     # P0 never holds a token
        assert os.path.isfile(tmp_path / "out" / r["raw_measurement_path"]) and ":" not in r["raw_measurement_path"]
    # the priming session is recorded apart from the measured log
    prim = [json.loads(l) for l in open(tmp_path / "out" / L.PRIMING_FILE, encoding="utf-8") if l.strip()]
    assert len(prim) == 2 and all(p["run_id"].endswith(":prime") and p["extra"]["warm_start"] is False for p in prim)
    text = open(res["records"][0], encoding="utf-8").read()
    assert ":prime" not in text
