"""Schema tests: required fields, defaults, eligibility consistency, JSONL round trip, tree hash, data root."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from opencrab2.experiments import run_primitives, schema


def test_validate_rejects_record_missing_status():
    rec = schema.new_record()
    del rec["status"]
    with pytest.raises(ValueError) as ei:
        schema.validate(rec)
    assert str(ei.value).startswith("SCHEMA_MISSING_FIELD")
    assert "status" in str(ei.value)


def test_new_record_defaults_are_safe_and_validate():
    rec = schema.new_record()
    schema.validate(rec)
    assert rec["schema"] == schema.SCHEMA_ID
    assert rec["status"] == "MODELED"
    assert rec["security_eligible"] is False
    assert rec["diagnostic_reveals"] == 0
    assert rec["dealer_mode"] == "NONE"
    assert rec["code_commit"].startswith("tree-")
    assert set(schema.RECORD_FIELDS) <= set(rec)
    assert rec["run_id"]


def test_every_required_field_is_checked():
    for f in schema.RECORD_FIELDS:
        rec = schema.new_record()
        del rec[f]
        with pytest.raises(ValueError):
            schema.validate(rec)


def test_eligibility_claims_are_checked():
    with pytest.raises(ValueError) as ei:
        schema.new_record(security_eligible=True)
    assert str(ei.value).startswith("SCHEMA_ELIGIBILITY")
    with pytest.raises(ValueError):
        schema.new_record(security_eligible=True, dealer_mode="ISOLATED_RECIPIENT_BUNDLES", status="MEASURED_NATIVE",
                          diagnostic_reveals=1)
    with pytest.raises(ValueError):
        schema.new_record(security_eligible=True, dealer_mode="INSECURE_TEST_ONLY_SHARED_SEED", status="MEASURED_NATIVE")
    with pytest.raises(ValueError):
        schema.new_record(security_eligible=True, dealer_mode="ISOLATED_RECIPIENT_BUNDLES", status="MODELED")
    ok = schema.new_record(security_eligible=True, dealer_mode="ISOLATED_RECIPIENT_BUNDLES", status="MEASURED_NATIVE")
    assert ok["security_eligible"] is True


@pytest.mark.parametrize("field,value,code", [
    ("status", "GUESSED", "SCHEMA_BAD_STATUS"),
    ("workload_scope", "whole", "SCHEMA_BAD_SCOPE"),
    ("dealer_mode", "shared", "SCHEMA_BAD_VALUE"),
    ("clock_basis", "epoch", "SCHEMA_BAD_VALUE"),
    ("bytes", -1, "SCHEMA_BAD_TYPE"),
    ("bytes", 1.5, "SCHEMA_BAD_TYPE"),
    ("T_cold_s", -2.0, "SCHEMA_BAD_TYPE"),
    ("T_cold_s", float("nan"), "SCHEMA_BAD_TYPE"),
    ("output_match", "yes", "SCHEMA_BAD_TYPE"),
    ("security_eligible", 1, "SCHEMA_BAD_TYPE"),
    ("run_id", "", "SCHEMA_BAD_VALUE"),
    ("record_kind", "guess", "SCHEMA_BAD_VALUE"),
])
def test_bad_values_are_rejected_with_codes(field, value, code):
    rec = schema.new_record()
    rec[field] = value
    with pytest.raises(ValueError) as ei:
        schema.validate(rec)
    assert str(ei.value).startswith(code), str(ei.value)


def test_unknown_field_is_refused():
    with pytest.raises(ValueError) as ei:
        schema.new_record(speedup=14.6)
    assert str(ei.value).startswith("SCHEMA_UNKNOWN_FIELD")
    rec = schema.new_record()
    rec["speedup"] = 14.6
    with pytest.raises(ValueError):
        schema.validate(rec)


def test_jsonl_round_trip_and_completed_ids(tmp_path):
    p = tmp_path / "raw" / "x.jsonl"
    a = schema.new_record(run_id="r1", status="MEASURED_NATIVE", T_cold_s=1.5, extra={"k": [1, 2]})
    b = schema.new_record(run_id="r2", status="HISTORICAL", tokens=3)
    schema.append_jsonl(p, a)
    schema.append_jsonl(p, b)
    back = schema.read_jsonl(p)
    assert back == [a, b]
    assert schema.completed_run_ids(p) == {"r1", "r2"}
    # append never overwrites
    schema.append_jsonl(p, schema.new_record(run_id="r3"))
    assert len(schema.read_jsonl(p)) == 3
    kinds = list(schema.iter_raw_records(tmp_path / "raw"))
    assert len(kinds) == 3 and kinds[0][0] == "x"
    assert b"\r" not in p.read_bytes()                                     # LF on every platform
    with open(p, "a", encoding="utf-8") as fh:
        fh.write("not json\n")
    with pytest.raises(ValueError):
        schema.read_jsonl(p)


def test_compressed_raw_file_is_read_when_the_plain_one_is_absent(tmp_path):
    import gzip
    raw = tmp_path / "raw"
    raw.mkdir()
    recs = [schema.new_record(run_id=f"r{i}", status="MODELED") for i in range(3)]
    with gzip.open(raw / "sim.jsonl.gz", "wt", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    assert schema.raw_file(raw / "sim.jsonl") == raw / "sim.jsonl.gz"
    assert schema.read_jsonl(raw / "sim.jsonl") == recs
    assert schema.read_jsonl(raw / "sim.jsonl.gz") == recs
    assert schema.completed_run_ids(raw / "sim.jsonl") == {"r0", "r1", "r2"}
    assert [stem for stem, _ in schema.iter_raw_records(raw)] == ["sim"] * 3
    assert schema.read_jsonl(raw / "none.jsonl") == [] and schema.raw_file(raw / "none.jsonl") is None
    # the plain file wins once it exists
    schema.append_jsonl(raw / "sim.jsonl", schema.new_record(run_id="plain"))
    assert schema.raw_file(raw / "sim.jsonl") == raw / "sim.jsonl"
    assert [r["run_id"] for r in schema.read_jsonl(raw / "sim.jsonl")] == ["plain"]
    assert sorted(stem for stem, _ in schema.iter_raw_records(raw)) == ["sim"]      # each name once


def test_relative_raw_path_never_returns_a_host_path(tmp_path):
    root = schema.experiments_root()
    assert schema.relative_raw_path(root / "raw" / "web.jsonl") == "raw/web.jsonl"
    deep = root / "raw" / "sessions" / "0123456789abcdef" / "sessions.x.p1.detail.jsonl"
    assert schema.relative_raw_path(deep) == "raw/sessions/0123456789abcdef/sessions.x.p1.detail.jsonl"
    # another data root: relative to it when asked, else cut at the raw directory
    other = tmp_path / "out" / "raw" / "calibration.jsonl"
    assert schema.relative_raw_path(other, tmp_path / "out") == "raw/calibration.jsonl"
    assert schema.relative_raw_path(other) == "raw/calibration.jsonl"
    assert schema.relative_raw_path(tmp_path / "out" / "calibration" / "fitted_model.json") == "calibration/fitted_model.json"
    # no anchor directory: the file name only; a relative input keeps its (posix) form
    assert schema.relative_raw_path(tmp_path / "records.jsonl") == "records.jsonl"
    assert schema.relative_raw_path("x/y.jsonl") == "x/y.jsonl"
    for p in (deep, other, tmp_path / "records.jsonl"):
        rel = schema.relative_raw_path(p)
        assert "\\" not in rel and not rel.startswith("/") and ":" not in rel.split("/")[0]


def test_data_root_is_the_shipped_results_unless_redirected(tmp_path, monkeypatch):
    tree = Path(schema.__file__).resolve().parents[3]
    monkeypatch.delenv(schema.RESULTS_DIR_ENV, raising=False)
    assert schema.RESULTS_DIR_ENV == "OPENCRAB2_RESULTS_DIR"
    assert schema.experiments_root() == schema.shipped_results_root() == tree / "results"
    assert (schema.shipped_results_root() / "calibration" / "formulas.json").is_file()
    # a rerun names another root; the shipped one stays where it is
    monkeypatch.setenv(schema.RESULTS_DIR_ENV, str(tmp_path / "rerun"))
    assert schema.experiments_root() == (tmp_path / "rerun").resolve()
    assert schema.shipped_results_root() == tree / "results"
    assert schema.relative_raw_path(tmp_path / "rerun" / "raw" / "web.jsonl") == "raw/web.jsonl"
    monkeypatch.setenv(schema.RESULTS_DIR_ENV, "  ")                     # blank: not a redirect
    assert schema.experiments_root() == schema.shipped_results_root()


def test_peak_rss_is_a_positive_byte_count():
    v = schema.peak_rss_bytes()
    assert v is None or (isinstance(v, int) and v > 1 << 20)


def test_append_rejects_invalid_record(tmp_path):
    rec = schema.new_record()
    rec["status"] = "GUESSED"
    with pytest.raises(ValueError):
        schema.append_jsonl(tmp_path / "bad.jsonl", rec)
    assert not (tmp_path / "bad.jsonl").exists()


def _append_worker(path: str, tag: int, count: int) -> None:
    from opencrab2.experiments import schema as sc
    for i in range(count):
        sc.locked_append_lines(path, [json.dumps({"tag": tag, "i": i})])


def test_locked_append_from_several_processes_loses_no_line(tmp_path):
    """Concurrent appends from spawned processes (as the two party processes do) all land as whole lines."""
    import multiprocessing
    path = str(tmp_path / "records.jsonl")
    ctx = multiprocessing.get_context("spawn")
    n_proc, count = 4, 25
    procs = [ctx.Process(target=_append_worker, args=(path, t, count)) for t in range(n_proc)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)
    assert all(p.exitcode == 0 for p in procs)
    lines = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    assert len(lines) == n_proc * count
    assert sorted((r["tag"], r["i"]) for r in lines) == [(t, i) for t in range(n_proc) for i in range(count)]
    assert not (tmp_path / "records.jsonl.lock").exists()


def test_tree_hash_is_stable_and_content_based(tmp_path):
    assert schema.tree_hash() == schema.tree_hash()
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    h1 = schema.tree_hash(tmp_path)
    (tmp_path / "a.py").write_text("x = 2\n", encoding="utf-8")
    h2 = schema.tree_hash(tmp_path)
    assert h1 != h2 and h1.startswith("tree-")


def test_stable_hash_is_canonical():
    assert schema.stable_hash({"b": 1, "a": [1, 2]}) == schema.stable_hash({"a": [1, 2], "b": 1})
    assert schema.stable_hash({"a": 1}) != schema.stable_hash({"a": 2})


def test_status_classes():
    assert schema.is_estimate("MODELED") and schema.is_estimate("ESTIMATED") and schema.is_estimate("EMULATED_NUMERIC")
    assert not schema.is_estimate("MEASURED_NATIVE") and not schema.is_estimate("HISTORICAL")
    assert schema.is_historical("HISTORICAL")


def test_primitive_record_builder_validates():
    timing = {"repeats": 3, "seconds": [0.1, 0.2, 0.3], "median_s": 0.2, "p95_s": 0.29, "min_s": 0.1, "mean_s": 0.2,
              "per_element_s": 1e-4, "material_bytes": 2048, "bytes": 16, "rounds": 1}
    rec = run_primitives.primitive_record("dcf_eval", 1024, timing, {"n_bits": 32})
    schema.validate(rec)
    assert rec["status"] == "MEASURED_NATIVE" and rec["record_kind"] == "primitive"
    assert rec["extra"]["component"] == "dcf_eval" and rec["material_consumed"] == 2048
    json.dumps(rec)
