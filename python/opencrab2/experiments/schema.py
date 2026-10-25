"""Evidence record schema ``opencrab-native-evidence-v2`` (paper, Section 3.2, evidence classes).

Every raw record written by a runner in this package is one JSON line with
exactly the vocabulary below. ``RECORD_FIELDS`` lists the required fields;
:func:`validate` fails closed on a missing field, an unknown status, a wrong
type or an inconsistent eligibility claim; :func:`new_record` fills every
field with a safe default (``status="MODELED"``, ``security_eligible=False``,
``diagnostic_reveals=0``) so that a record can never claim more than the
runner stated explicitly.

Eligibility rule: ``security_eligible`` may be true only when
``dealer_mode == "ISOLATED_RECIPIENT_BUNDLES"``, ``diagnostic_reveals == 0``,
``status == "MEASURED_NATIVE"`` and ``aborts == 0``. The schema cannot check
that every used component was ADMITTED (that is the registry's job at run
time), so a record that passes :func:`validate` is well formed, not proven
eligible.

Per-kind payloads (calibration rows, simulation epoch summaries, web metrics)
live under the ``extra`` field so that the required vocabulary stays flat and
identical for every evidence class; ``make_tables.py`` reads the status of a
record, never of a cell, and wraps every non-measured value.

Data root. :func:`experiments_root` is ``<tree>/results`` (the shipped
measured outputs) unless ``OPENCRAB2_RESULTS_DIR`` names another directory;
a rerun sets that variable and every driver and child process writes there.
``raw_measurement_path`` names the file that holds the measurement behind a
record (a raw JSONL file or a per-session detail file) relative to the data
root with forward slashes (:func:`relative_raw_path`); a record never carries
an absolute host path. Raw files are read from ``<name>.jsonl`` or, when that
file is absent, from ``<name>.jsonl.gz``.

What is tested (tests/test_schema.py): a record without ``status`` is
rejected, every default of :func:`new_record` validates, an eligibility claim
with an insecure dealer or a diagnostic reveal is rejected, unknown statuses
and workload scopes are rejected, the JSONL append/read round trip preserves
the record, a compressed raw file is read when the plain one is absent, the
relative path helper never returns a host path, and ``code_commit`` is a
stable hash of the Python tree. Design limit: the tree hash replaces a git
commit because this tree is not a git repository; a reader must treat it as
a content hash of ``python/opencrab2``. Error messages start with the stable
codes ``SCHEMA_MISSING_FIELD``, ``SCHEMA_BAD_STATUS``, ``SCHEMA_BAD_SCOPE``,
``SCHEMA_BAD_TYPE``, ``SCHEMA_BAD_VALUE``, ``SCHEMA_ELIGIBILITY``,
``SCHEMA_UNKNOWN_FIELD``.
"""

from __future__ import annotations

import datetime as _dt
import gzip
import hashlib
import json
import math
import os
import secrets
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

SCHEMA_ID = "opencrab-native-evidence-v2"
PROFILE_ID = "SH2P-TrustedOffline-v1"

STATUSES: Tuple[str, ...] = (
    "MEASURED_NATIVE", "EMULATED_NUMERIC", "MODELED", "EXTERNAL_REPORTED", "DIAGNOSTIC", "HISTORICAL", "ESTIMATED",
)
# statuses whose values the paper must print in red (make_tables wraps them in \est{...})
ESTIMATE_STATUSES: Tuple[str, ...] = ("MODELED", "EMULATED_NUMERIC", "ESTIMATED")
HISTORICAL_STATUSES: Tuple[str, ...] = ("HISTORICAL",)
WORKLOAD_SCOPES: Tuple[str, ...] = ("full-model", "prefix-4L", "prefix-1L", "primitive", "web")
DEALER_MODES: Tuple[str, ...] = ("ISOLATED_RECIPIENT_BUNDLES", "INSECURE_TEST_ONLY_SHARED_SEED", "NONE")
CLOCK_BASES: Tuple[str, ...] = ("wall-clock-exogenous", "modeled-epoch", "process-perf-counter")
RECORD_KINDS: Tuple[str, ...] = ("session", "primitive", "calibration", "simulation", "factorial", "web")

# required fields (plus the schema tag)
RECORD_FIELDS: Tuple[str, ...] = (
    "schema", "status", "security_eligible", "code_commit", "profile_id", "numeric_contract_hash",
    "dealer_mode", "diagnostic_reveals", "workload_scope", "model_revision", "trace_id", "trace_hash",
    "clock_basis", "policy_id", "initial_state_hash", "run_id", "repeat", "initializations",
    "post_init_switches", "rejected_candidates", "aborts", "T_cold_s", "T_warm_s", "T_online_s", "bytes",
    "rounds", "material_generated", "material_transferred", "material_consumed", "material_burned",
    "peak_rss_client", "peak_rss_server", "tokens", "output_match", "raw_measurement_path",
)
# optional fields a runner may add; anything else is refused so that typos never hide a value
OPTIONAL_FIELDS: Tuple[str, ...] = (
    "record_kind", "seed", "family", "variant", "split", "T", "n_layers", "created_utc", "host", "label", "extra",
)

_INT_FIELDS = ("diagnostic_reveals", "repeat", "initializations", "post_init_switches", "rejected_candidates",
               "aborts", "bytes", "rounds", "material_generated", "material_transferred", "material_consumed",
               "material_burned", "tokens")
_OPT_FLOAT_FIELDS = ("T_cold_s", "T_warm_s", "T_online_s")
_OPT_INT_FIELDS = ("peak_rss_client", "peak_rss_server")
_STR_FIELDS = ("schema", "status", "code_commit", "profile_id", "numeric_contract_hash", "dealer_mode",
               "workload_scope", "model_revision", "trace_id", "trace_hash", "clock_basis", "policy_id",
               "initial_state_hash", "run_id", "raw_measurement_path")


# ---------------------------------------------------------------------------
# tree hash (stands in for the git commit of a tree that is not a repository)
# ---------------------------------------------------------------------------
_TREE_HASH: Optional[str] = None


def package_root() -> Path:
    """Directory of the ``opencrab2`` package."""
    return Path(__file__).resolve().parents[1]


RESULTS_DIR_ENV = "OPENCRAB2_RESULTS_DIR"


def shipped_results_root() -> Path:
    """``<tree>/results`` next to ``python/``: the measured outputs that ship with the code."""
    return Path(__file__).resolve().parents[3] / "results"


def experiments_root() -> Path:
    """The data root (raw, calibration, tables): ``$OPENCRAB2_RESULTS_DIR`` when set, else the shipped ``results/``.

    A rerun sets the variable (``scripts/run_campaign.py`` does so for every
    stage and every child process inherits it) so that the drivers write
    there and leave the shipped outputs untouched.
    """
    override = os.environ.get(RESULTS_DIR_ENV, "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return shipped_results_root()


RAW_PATH_ANCHORS: Tuple[str, ...] = ("raw", "calibration", "processed", "tables")


def relative_raw_path(path: Any, root: Optional[Any] = None) -> str:
    """``path`` as a record stores it: relative to the data root, forward slashes, never a host path.

    ``root`` defaults to :func:`experiments_root`. A path outside the root
    (a driver writing into another ``--out-root``) is cut at its deepest
    directory named like one of the root's subdirectories
    (``RAW_PATH_ANCHORS``: ``raw/sessions/<cell>/<file>`` stays recognisable);
    a path without such a component keeps its file name only. An already
    relative path is returned with forward slashes.
    """
    p = Path(str(path))
    if not p.is_absolute():
        return p.as_posix()
    base = Path(root) if root is not None else experiments_root()
    try:
        return p.resolve().relative_to(base.resolve()).as_posix()
    except (ValueError, OSError):
        pass
    parts = p.parts
    for i in range(len(parts) - 2, -1, -1):          # the deepest anchor directory above the file
        if parts[i] in RAW_PATH_ANCHORS:
            return Path(*parts[i:]).as_posix()
    return p.name


def tree_hash(root: Optional[Path] = None) -> str:
    """sha256 over the sorted relative paths and contents of every ``.py`` and ``.json`` file under ``root``."""
    global _TREE_HASH
    if root is None and _TREE_HASH is not None:
        return _TREE_HASH
    base = Path(root) if root is not None else package_root()
    h = hashlib.sha256()
    files = sorted(p for p in base.rglob("*") if p.suffix in (".py", ".json") and "__pycache__" not in p.parts)
    for p in files:
        rel = p.relative_to(base).as_posix().encode("utf-8")
        h.update(len(rel).to_bytes(4, "big"))
        h.update(rel)
        data = p.read_bytes()
        h.update(len(data).to_bytes(8, "big"))
        h.update(data)
    digest = "tree-" + h.hexdigest()
    if root is None:
        _TREE_HASH = digest
    return digest


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_contract_hash() -> str:
    """Hash of the default numeric contract when the contract module is importable, else an explicit marker."""
    try:
        from opencrab2.highlevel.contract import DEFAULT_CONTRACT
        return DEFAULT_CONTRACT.contract_hash()
    except Exception:  # pragma: no cover (the contract module is part of this tree)
        return "contract-unavailable"


def stable_hash(obj: Any) -> str:
    """sha256 of the canonical JSON (sorted keys, no spaces) of a JSON-able object."""
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# construction and validation
# ---------------------------------------------------------------------------
def new_record(**kw: Any) -> dict:
    """A complete record with safe defaults; keyword arguments override fields (unknown names are refused)."""
    rec: Dict[str, Any] = {
        "schema": SCHEMA_ID,
        "status": "MODELED",
        "security_eligible": False,
        "code_commit": tree_hash(),
        "profile_id": PROFILE_ID,
        "numeric_contract_hash": default_contract_hash(),
        "dealer_mode": "NONE",
        "diagnostic_reveals": 0,
        "workload_scope": "primitive",
        "model_revision": "n/a",
        "trace_id": "none",
        "trace_hash": "",
        "clock_basis": "modeled-epoch",
        "policy_id": "none",
        "initial_state_hash": "",
        "run_id": "run-" + secrets.token_hex(8),
        "repeat": 0,
        "initializations": 0,
        "post_init_switches": 0,
        "rejected_candidates": 0,
        "aborts": 0,
        "T_cold_s": None,
        "T_warm_s": None,
        "T_online_s": None,
        "bytes": 0,
        "rounds": 0,
        "material_generated": 0,
        "material_transferred": 0,
        "material_consumed": 0,
        "material_burned": 0,
        "peak_rss_client": None,
        "peak_rss_server": None,
        "tokens": 0,
        "output_match": None,
        "raw_measurement_path": "",
        "created_utc": utc_now(),
        "extra": {},
    }
    for k, v in kw.items():
        if k not in RECORD_FIELDS and k not in OPTIONAL_FIELDS:
            raise ValueError(f"SCHEMA_UNKNOWN_FIELD: {k!r} is not a record field")
        rec[k] = v
    validate(rec)
    return rec


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _finite_float(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def validate(record: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` (stable ``SCHEMA_*`` code) unless ``record`` conforms; return None otherwise."""
    if not isinstance(record, Mapping):
        raise ValueError("SCHEMA_BAD_TYPE: a record must be a mapping")
    missing = [f for f in RECORD_FIELDS if f not in record]
    if missing:
        raise ValueError(f"SCHEMA_MISSING_FIELD: record lacks {missing}")
    unknown = [k for k in record if k not in RECORD_FIELDS and k not in OPTIONAL_FIELDS]
    if unknown:
        raise ValueError(f"SCHEMA_UNKNOWN_FIELD: {unknown} are not record fields")
    if record["schema"] != SCHEMA_ID:
        raise ValueError(f"SCHEMA_BAD_VALUE: schema must be {SCHEMA_ID!r}, got {record['schema']!r}")
    status = record["status"]
    if status not in STATUSES:
        raise ValueError(f"SCHEMA_BAD_STATUS: {status!r} is not one of {list(STATUSES)}")
    if record["workload_scope"] not in WORKLOAD_SCOPES:
        raise ValueError(f"SCHEMA_BAD_SCOPE: {record['workload_scope']!r} is not one of {list(WORKLOAD_SCOPES)}")
    if record["dealer_mode"] not in DEALER_MODES:
        raise ValueError(f"SCHEMA_BAD_VALUE: dealer_mode {record['dealer_mode']!r} is not one of {list(DEALER_MODES)}")
    if record["clock_basis"] not in CLOCK_BASES:
        raise ValueError(f"SCHEMA_BAD_VALUE: clock_basis {record['clock_basis']!r} is not one of {list(CLOCK_BASES)}")
    for f in _STR_FIELDS:
        if not isinstance(record[f], str):
            raise ValueError(f"SCHEMA_BAD_TYPE: {f} must be a string")
    if not record["run_id"]:
        raise ValueError("SCHEMA_BAD_VALUE: run_id must be non-empty")
    if not isinstance(record["security_eligible"], bool):
        raise ValueError("SCHEMA_BAD_TYPE: security_eligible must be a bool")
    for f in _INT_FIELDS:
        if not _is_int(record[f]) or record[f] < 0:
            raise ValueError(f"SCHEMA_BAD_TYPE: {f} must be a non-negative int, got {record[f]!r}")
    for f in _OPT_FLOAT_FIELDS:
        v = record[f]
        if v is not None and (not _finite_float(v) or float(v) < 0.0):
            raise ValueError(f"SCHEMA_BAD_TYPE: {f} must be None or a finite non-negative number, got {v!r}")
    for f in _OPT_INT_FIELDS:
        v = record[f]
        if v is not None and (not _is_int(v) or v < 0):
            raise ValueError(f"SCHEMA_BAD_TYPE: {f} must be None or a non-negative int, got {v!r}")
    om = record["output_match"]
    if om is not None and not isinstance(om, bool):
        raise ValueError("SCHEMA_BAD_TYPE: output_match must be None or a bool")
    if "record_kind" in record and record["record_kind"] not in RECORD_KINDS:
        raise ValueError(f"SCHEMA_BAD_VALUE: record_kind {record['record_kind']!r} is not one of {list(RECORD_KINDS)}")
    if "extra" in record and not isinstance(record["extra"], Mapping):
        raise ValueError("SCHEMA_BAD_TYPE: extra must be a mapping")
    if record["security_eligible"]:
        if record["dealer_mode"] != "ISOLATED_RECIPIENT_BUNDLES":
            raise ValueError("SCHEMA_ELIGIBILITY: security_eligible requires dealer_mode ISOLATED_RECIPIENT_BUNDLES")
        if record["diagnostic_reveals"] != 0:
            raise ValueError("SCHEMA_ELIGIBILITY: security_eligible requires diagnostic_reveals == 0")
        if status != "MEASURED_NATIVE":
            raise ValueError("SCHEMA_ELIGIBILITY: security_eligible requires status MEASURED_NATIVE")
        if record["aborts"] != 0:
            raise ValueError("SCHEMA_ELIGIBILITY: security_eligible requires a completed run (aborts == 0)")
    try:
        json.dumps(record, allow_nan=False, default=_json_default)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"SCHEMA_BAD_VALUE: record is not JSON serialisable without NaN ({exc})") from None


def _json_default(o: Any) -> Any:
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:  # pragma: no cover
        pass
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    raise TypeError(f"SCHEMA_BAD_VALUE: cannot serialise {type(o).__name__}")


def is_estimate(status: str) -> bool:
    return status in ESTIMATE_STATUSES


def is_historical(status: str) -> bool:
    return status in HISTORICAL_STATUSES


# ---------------------------------------------------------------------------
# JSONL persistence (append only, never overwrite)
# ---------------------------------------------------------------------------
_LOCK_TIMEOUT_S = 30.0
_LOCK_STALE_S = 120.0


def locked_append_lines(path: Any, lines: Iterable[str]) -> None:
    """Append complete lines to ``path`` under a cross-process lock file (creates parents, never truncates).

    Two party processes finish a session within milliseconds of each other and
    both append their record to the same JSONL file. ``open(path, "a")`` on
    Windows is a seek-to-end followed by a write, two separate system calls, so
    a concurrent append from the other process can be lost (observed once in
    tests/test_launch.py: one of two records missing). The lock is a sibling
    file ``<path>.lock`` created with ``O_CREAT | O_EXCL`` (atomic on every
    platform); the holder appends, flushes, fsyncs and removes it. A lock older
    than ``_LOCK_STALE_S`` seconds is treated as left behind by a crashed
    process and removed. Waiting longer than ``_LOCK_TIMEOUT_S`` raises
    ``OSError`` with code ``RECORD_LOCK_TIMEOUT`` rather than writing unlocked.
    What is tested (tests/test_schema.py): concurrent appends from several
    processes all land as whole lines. Design limit: this protects appends of
    cooperating processes on one host; it is not a defence against a hostile
    writer.
    """
    import time as _time
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lock = Path(str(p) + ".lock")
    payload = "".join(l if l.endswith("\n") else l + "\n" for l in lines)
    deadline = _time.monotonic() + _LOCK_TIMEOUT_S
    while True:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except (FileExistsError, PermissionError):
            # Windows reports a lock file that another process holds open or has
            # just unlinked (delete pending) as EACCES rather than EEXIST; both
            # mean "contended", so wait and retry.
            try:
                if _time.time() - lock.stat().st_mtime > _LOCK_STALE_S:
                    lock.unlink()
                    continue
            except OSError:
                pass
            if _time.monotonic() > deadline:
                raise OSError(f"RECORD_LOCK_TIMEOUT: could not acquire {lock} within {_LOCK_TIMEOUT_S} s")
            _time.sleep(0.005)
    try:
        os.close(fd)
        # A file that does not end with a newline holds the fragment of an append that died. Start on a
        # new line, so that this record is never glued onto the fragment (the fragment stays one
        # unparsable line of its own; read_jsonl(strict=False) skips it only when it is the last line).
        try:
            with open(p, "rb") as rf:
                rf.seek(0, os.SEEK_END)
                if rf.tell() > 0:
                    rf.seek(-1, os.SEEK_END)
                    if rf.read(1) != b"\n":
                        payload = "\n" + payload
        except FileNotFoundError:
            pass
        with open(p, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
    finally:
        try:
            lock.unlink()
        except OSError:
            pass


def append_jsonl(path: Any, record: Mapping[str, Any]) -> None:
    """Validate and append one record as a JSON line (creates parents, never truncates, cross-process safe)."""
    validate(record)
    line = json.dumps(record, sort_keys=True, allow_nan=False, default=_json_default)
    locked_append_lines(path, [line])


def append_many(path: Any, records: Iterable[Mapping[str, Any]]) -> int:
    n = 0
    for r in records:
        append_jsonl(path, r)
        n += 1
    return n


def raw_file(path: Any) -> Optional[Path]:
    """``<name>.jsonl`` when it exists, else ``<name>.jsonl.gz`` when that exists, else None."""
    p = Path(path)
    if p.is_file():
        return p
    gz = p if p.suffix == ".gz" else Path(str(p) + ".gz")
    return gz if gz.is_file() else None


def _read_text_lines(p: Path) -> List[str]:
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8", errors="replace") as fh:   # type: ignore[call-overload]
        return list(fh)


def read_jsonl(path: Any, strict: bool = True) -> List[dict]:
    """Read every record of a JSONL file; ``strict`` validates each line and refuses malformed files.

    ``path`` names ``<name>.jsonl``; when that file is absent and ``<name>.jsonl.gz`` exists, the compressed
    file is read instead (the shipped modeled records are compressed). With ``strict=False`` an unparsable
    LAST line is skipped: it is the torn tail of an append that died (a resumable campaign must be able to
    read its own output after a crash). An unparsable line that is followed by another record is corruption
    in both modes (``SCHEMA_BAD_VALUE``).
    """
    out: List[dict] = []
    p = raw_file(path)
    if p is None:
        return out
    lines = [(ln, line.strip()) for ln, line in enumerate(_read_text_lines(p), 1)]
    lines = [(ln, line) for ln, line in lines if line]
    for i, (ln, line) in enumerate(lines):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            if not strict and i == len(lines) - 1:
                break                      # torn tail of a crashed append
            raise ValueError(f"SCHEMA_BAD_VALUE: {p} line {ln} is not JSON ({exc.msg})") from None
        if strict:
            validate(rec)
        out.append(rec)
    return out


def iter_raw_records(raw_dir: Any, strict: bool = True) -> Iterator[Tuple[str, dict]]:
    """Yield ``(file_stem, record)`` for every record under ``raw_dir/*.jsonl`` (or ``*.jsonl.gz`` when the
    plain file is absent) in sorted file order."""
    d = Path(raw_dir)
    if not d.is_dir():
        return
    names = sorted({p.name[:-3] if p.name.endswith(".gz") else p.name
                    for p in list(d.glob("*.jsonl")) + list(d.glob("*.jsonl.gz"))})
    for name in names:
        for rec in read_jsonl(d / name, strict=strict):
            yield Path(name).stem, rec


def completed_run_ids(path: Any) -> set:
    """run_ids already present in a JSONL file (for resumable campaigns)."""
    return {r["run_id"] for r in read_jsonl(path, strict=False) if isinstance(r, dict) and "run_id" in r}


def peak_rss_bytes() -> Optional[int]:
    """Peak resident set size of this process in bytes when the platform exposes it, else None.

    POSIX: the high-water mark ``ru_maxrss`` of ``resource.getrusage`` (kilobytes on Linux, bytes on macOS).
    Windows (no ``resource`` module): ``psutil`` ``peak_wset``. ``psutil``'s ``rss`` is the current size, not
    the peak, so it is never used as a stand-in.
    """
    try:
        import resource  # type: ignore
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return peak if sys.platform == "darwin" else peak * 1024
    except Exception:
        pass
    try:
        import psutil
        peak = getattr(psutil.Process(os.getpid()).memory_info(), "peak_wset", None)
        return int(peak) if peak is not None else None
    except Exception:
        return None


__all__ = [
    "SCHEMA_ID", "PROFILE_ID", "STATUSES", "ESTIMATE_STATUSES", "HISTORICAL_STATUSES", "WORKLOAD_SCOPES",
    "DEALER_MODES", "CLOCK_BASES", "RECORD_KINDS", "RECORD_FIELDS", "OPTIONAL_FIELDS", "RAW_PATH_ANCHORS",
    "package_root", "RESULTS_DIR_ENV", "shipped_results_root", "experiments_root", "relative_raw_path", "tree_hash",
    "utc_now", "default_contract_hash",
    "stable_hash", "new_record", "validate", "is_estimate", "is_historical", "append_jsonl", "append_many",
    "raw_file", "read_jsonl", "iter_raw_records", "completed_run_ids", "peak_rss_bytes",
]
