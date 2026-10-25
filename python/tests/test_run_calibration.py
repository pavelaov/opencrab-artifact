"""run_calibration --fit-only: refits from the raw records (plain or gzip) and keeps the campaign block."""

from __future__ import annotations

import gzip
import json
import pathlib
import shutil

import pytest

from opencrab2.experiments import run_calibration as rc
from opencrab2.experiments import schema

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def _shipped_calibration():
    raw = schema.raw_file(RESULTS / "raw" / "calibration.jsonl")
    metrics = RESULTS / "calibration" / "calibration_metrics.json"
    if raw is None or not metrics.is_file():
        pytest.skip("the shipped calibration records are not in this tree")
    return raw, metrics


def test_previous_campaign_block_reads_only_a_campaign_dict(tmp_path):
    assert rc.previous_campaign_block(tmp_path) is None
    (tmp_path / "calibration").mkdir()
    (tmp_path / "calibration" / "calibration_metrics.json").write_text("not json", encoding="utf-8")
    assert rc.previous_campaign_block(tmp_path) is None
    (tmp_path / "calibration" / "calibration_metrics.json").write_text(json.dumps({"campaign": [1]}), encoding="utf-8")
    assert rc.previous_campaign_block(tmp_path) is None
    (tmp_path / "calibration" / "calibration_metrics.json").write_text(
        json.dumps({"campaign": {"campaign_id": "c1", "formulas_sha256": "ab"}}), encoding="utf-8")
    assert rc.previous_campaign_block(tmp_path) == {"campaign_id": "c1", "formulas_sha256": "ab"}


def test_fit_only_keeps_the_campaign_block_and_reads_compressed_records(tmp_path):
    raw, metrics = _shipped_calibration()
    (tmp_path / "raw").mkdir()
    (tmp_path / "calibration").mkdir()
    # the records as a compressed file only: the reader falls back to <name>.jsonl.gz
    with open(raw, "rb") as src, gzip.open(tmp_path / "raw" / "calibration.jsonl.gz", "wb") as dst:
        shutil.copyfileobj(src, dst)
    shutil.copy(metrics, tmp_path / "calibration" / "calibration_metrics.json")
    before = json.loads(metrics.read_text(encoding="utf-8"))
    summary = rc.refit_from_raw(tmp_path)
    after = json.loads((tmp_path / "calibration" / "calibration_metrics.json").read_text(encoding="utf-8"))
    assert summary["refit"] is True and summary["campaign"] == before["campaign"] == after["campaign"]
    assert after["campaign"]["formulas_sha256"] == before["campaign"]["formulas_sha256"]
    assert after["refit_utc"] and after["n_rows"] == before["n_rows"]
    assert after["fitted_components"] == before["fitted_components"]
    assert set(after["models"]) >= {"fitted_MEASURED_NATIVE", "handwritten_MODELED"}
    model = json.loads((tmp_path / "calibration" / "fitted_model.json").read_text(encoding="utf-8"))
    shipped = json.loads((RESULTS / "calibration" / "fitted_model.json").read_text(encoding="utf-8"))
    assert set(model["components"]) == set(shipped["components"])
    assert all(model["components"][k]["status"] == shipped["components"][k]["status"] for k in shipped["components"])
    assert b"\r" not in (tmp_path / "calibration" / "calibration_metrics.json").read_bytes()
    # without a previous metrics file the refit writes no campaign block (the tables then say stale)
    (tmp_path / "calibration" / "calibration_metrics.json").unlink()
    summary2 = rc.refit_from_raw(tmp_path)
    assert "campaign" not in summary2
