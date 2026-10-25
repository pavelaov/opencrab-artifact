"""run_primitives: one small benchmark end to end; the records validate and name their file relative to the data root."""

from __future__ import annotations

import pathlib

from opencrab2.experiments import run_primitives as rp
from opencrab2.experiments import schema


def test_small_run_writes_valid_records_with_a_relative_path(tmp_path):
    out = tmp_path / "out" / "raw" / "primitives.jsonl"
    recs = rp.run([64], 1, out, only=["dcf_eval"])
    assert {r["extra"]["component"] for r in recs} == {"dcf_gen", "dcf_eval"}
    for r in recs:
        schema.validate(r)
        assert r["status"] == "MEASURED_NATIVE" and r["record_kind"] == "primitive" and r["extra"]["batch"] == 64
        assert r["raw_measurement_path"] == "raw/primitives.jsonl"          # never a host path
    on_disk = schema.read_jsonl(out)
    assert [r["run_id"] for r in on_disk] == [r["run_id"] for r in recs]
    assert b"\r" not in out.read_bytes()                                     # LF on every platform
    assert not pathlib.PureWindowsPath(on_disk[0]["raw_measurement_path"]).is_absolute()


def test_parser_defaults():
    p = rp.build_parser()
    args = p.parse_args(["--quick", "--only", "drelu,trunc_faithful"])
    assert args.quick and args.only == "drelu,trunc_faithful" and args.out is None
    assert set(rp.PRIMITIVES) >= {"dcf_gen", "dcf_eval", "drelu", "trunc_faithful", "ring_to_field"}
