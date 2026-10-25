"""The paper tables from the raw evidence records: never from literals.

``make_all`` writes exactly eight files into ``<data root>/tables/`` (the
``--out-root``; default ``schema.experiments_root()``), each a bare
``tabular`` (booktabs rules, no ``table`` float and no caption: the paper
supplies those) that starts with comment lines naming the inputs it was
computed from, the marking rule, how many cells are measured and how many are
expected, and, per column, where an expected value comes from. Every number
the paper's prose cites is copied from a cell or from a computed line of these
header comments, so the headers also carry the derived figures honest prose
needs (listed per table below):

* ``catalog.tex``: component, kind, registry status, evidence class. Generated
  from ``registry.REGISTRY`` (not from raw records). The evidence class is the
  strongest record status a run that uses the component may carry: ADMITTED
  components can appear in MEASURED_NATIVE records, TEST_ONLY components only
  in DIAGNOSTIC records, components that are unavailable or disabled in none.
* ``blocks.tex``: one row per building block (DReLU with one and with two
  payloads, TruncFaithful, RingToField, ExactA2H encrypt-and-add, argmax over
  the GPT-2 vocabulary for k = 2, 4, 8, the toy bridge): online bytes per
  element and party, rounds, time per element in Python
  (``raw/primitives.jsonl``) and in C++ (``raw/cpp_bench.jsonl``). The C++ to
  Python ratio behind the expected cells uses only blocks measured in both
  languages with the same procedure and parameters; the header says what
  each toy bridge cell is (the Python steps at their N, the C++ local variant
  at its N) and that the row enters no ratio.
* ``costmodel.tex``: hold-out metrics of the calibration (MAPE, median, p95,
  Spearman, lane accuracy, plan accuracy, regret, informative pair, two-sigma
  coverage) for the hand-written, the fitted and the uncertainty-aware model
  (``calibration/calibration_metrics.json``). Every row of the file is the
  evaluation of that model on the same measured hold-out rows and is printed
  plain, the hand-written constants included (a measured evaluation of
  constants that were never fitted, not an expectation). A calibration that
  predates the current ``formulas.json`` is stale: its cells are expectations.
* ``dynamic.tex``: the measured diag4 campaign (``raw/sessions.jsonl``). Rows
  are trace cells; columns are the best fixed plan in hindsight (B0, seconds),
  then for B1, B2 and B4 the mean paired difference to B0 in percent with a
  95 % bootstrap interval, the number of paired sessions, and the
  post-initialization switches. A row label ends in ``*`` when only one fixed
  plan ran in the cell (the other was skipped by a declared reduction): B0 is
  then that plan, not a choice in hindsight. The header gives, for every
  printed difference, the Student t interval (n - 1 degrees of freedom) and
  whether the policy executed the identical plan sequence as B0 in every
  seed, plus the noise floor: the largest absolute paired difference among
  the identical-plan pairs of the matrix. After the seven diag4 rows come the
  full12 cells (label prefix ``12L``), computed by the same rules within the
  full12 matrix (B0, pairs and noise floor never cross matrices), with their
  own header lines (single-fixed-plan cells, Student t lines, noise floor); a
  policy that did not run in a full12 cell (not planned by the matrix, or
  without a session in a cell whose other policies ran) prints ``-``.
* ``dynamic_modeled.tex``: the same layout for the modeled campaign over all
  nine trace families (``raw/simulation.jsonl``); every value is ``\\est{}``.
  The header computes the derived figures the prose cites: the mean paired
  difference to B0 of ``B4_no_install`` and of ``delayed:2`` on S1 and on S3
  at 8 tokens.
* ``factorial.tex``: planner x controller (``raw/factorial.jsonl``).
* ``e2e.tex``: the full12 and diag4 anchors on the stable LAN: T_cold,
  T_warm, T_online, bytes, rounds, material, peak RSS, output match, n. The
  header names the concurrency level of each matrix.
* ``web.tex``: cold TTFT and warm TTFT as the mean with [min, max] over the
  repeats, inter-token latency, completion, cancel waste, resume recovery and
  the rejection counters per workload (``raw/web.jsonl``). Records are never
  pooled across thread caps: per workload the records of one cap are used
  (the latest cap; records without the field only when no capped record
  exists) and the header says which were left out. Per workload the header
  lists, per repeat, the first-epoch plan path the warm request took (plan,
  switched or kept, decision reason) from the fields the driver records.

The session tables (dynamic and e2e) state in their headers, per matrix, the
measured to predicted session time ratio of the campaign sessions only (median
and range, with the concurrency level named; the smoke sessions are pipeline
checks and excluded) and the output match: the matching of the completed
sessions per matrix and one line per mismatching session with its run id, the
first differing step and the reference margin at that step (or that the record
lacks those fields).

Status discipline. A value computed from ``MEASURED_NATIVE`` records is
printed plain; from ``MODELED``, ``EMULATED_NUMERIC`` or ``ESTIMATED`` records
it is wrapped in ``\\est{}``; ``HISTORICAL`` in ``\\hist{}``; ``DIAGNOSTIC`` in
``\\diag{}``; ``EXTERNAL_REPORTED`` in ``\\ext{}``. A value aggregated from
records of mixed status takes the weakest status. Cells hold only these four
macros and plain numbers. A derived warm time (cold minus setup and
installation) is not a measurement of a warm request and is wrapped in
``\\est{}``; only the wall time of a real warm request is printed plain. One
header line of every table states the marking rule: plain = measured by this
release; est = expected or derived; diag = measured with a test-only
component. ``security_eligible`` does not drive the marking: a record with
``security_eligible`` false that is a plain in-process timing (a primitive
benchmark, a rejection probe, a cancelled generation) stays plain, and the
blocks and web headers say so.

Expected values. A cell without a measured record is never left as text: it
prints the value the measurement is EXPECTED to give, wrapped in ``\\est{}``
(red in the paper), so that the operator can cross-check the record that
arrives later and notice a logging error. :class:`Expectations` computes them
and never reads a typed result:

* seconds: the fitted cost model through ``run_sessions.predict_cell``, the
  predictor of the dry run (online + sensing + installation + material
  generation; the warm variant without installation);
* bytes and rounds: the exact protocol formulas
  (``candidates.plan_bytes_rounds`` plus the installation messages of
  ``costmodel`` and the two plan-independent messages of an epoch, the
  client's one-hot input and the released token index);
* material: ``candidates.material_per_epoch`` plus the installation masks;
* paired differences and switches of the dynamic table: the MODELED campaign
  records of the same trace family, variant, T and layer count when
  ``raw/simulation.jsonl`` has them, otherwise the predictor (difference of
  the predicted plan to the cheaper fixed plan; switches from a walk of the
  remaining-cost rule along the trace);
* n, output match and the rejection counters: the planned counts of the
  matrices of ``run_sessions`` and of the defaults of ``run_web``;
* peak memory: ``candidates.mem_peak`` plus the measured process baseline
  (median of measured peak minus estimate over the completed sessions);
* Web rows: the predictor on an unshaped loopback link;
* cost-model rows: the previous (stale) calibration, the fitted row for the
  uncertainty-aware model, else a self-check of the two models on the declared
  formula shapes;
* C++ block times: the Python time times the median C++ to Python ratio of the
  blocks measured in both languages.

The header of every table states the rule per column and counts the expected
and the measured cells. A measured record always wins over an expectation.

Aggregation rules (fixed here, before any campaign record exists): sessions
are read from the client's (P1) record; an aborted session is never averaged
and is counted in the header comment; the session time is the measured wall
time of the session; pairs are formed within one seed and one concurrency
level (``extra.concurrent_workers``, ``extra.threads_per_process``); a matrix
with records of several levels uses the level of its latest record and the
header counts what was left out; the best fixed plan in hindsight is, per
trace cell, the fixed plan with the lowest mean session time over the paired
seeds; intervals are percentile bootstrap intervals of the mean paired
difference (10 000 resamples, seed 0); the Student t interval of the header
is mean +- t(0.975, n - 1) x sample standard deviation / sqrt(n).

Input files are read as ``<name>.jsonl`` or, when that is absent, as
``<name>.jsonl.gz``; tables and the manifest are written with LF line endings.
The only numeric literals in this file are formatting precisions, unit
conversions and structural indices; no result value is typed here. A table
this script generated under a name that is no longer in ``TABLE_NAMES`` is
moved to ``tables/obsolete/``.

Usage::

    python -m opencrab2.experiments.make_tables [--out-root DIR]

What is tested (tests/test_make_tables.py): every table is produced from an
empty tree with every data cell an ``\\est{}`` expectation (or the ``-`` of a
policy the full12 matrix does not plan) and without a placeholder text;
MODELED input yields ``\\est`` wrapped cells and MEASURED_NATIVE cells stay
plain; each file is a bare tabular with the header comments, the marking line
and the cell counts; a malformed raw line is skipped and counted; a compressed
raw file is read; files end lines with LF; the paired percent difference and
its interval are those of ``statistics.paired_bootstrap_ci``; the header
figures (ratio per matrix, output match lines, Student intervals,
identical-plan flags, noise floor, the derived modeled differences, the web
paths and cap grouping) are checked on synthetic records; the full12 rows of
the dynamic table are checked on a fixture with both matrices (rows after the
prefix rows, B0 and pairs within each matrix, hyphens, star cells, their own
Student t, identical-plan and noise-floor lines). Design limits: tables
summarise the records as they are; three seeds give wide intervals and the
table prints them as such; an expectation is a model output (MODELED), the
predictor prices neither process start nor the contention of concurrent
sessions, and the header of the session tables reports how far completed
sessions were from it.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import math
import re
import statistics as _stats
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import t as _student_t

from opencrab2.experiments import run_sessions as rs
from opencrab2.experiments import run_simulation as sim
from opencrab2.experiments import run_web, schema, traces
from opencrab2.highlevel import candidates as cands
from opencrab2.highlevel import costmodel as cm
from opencrab2.highlevel import planner as pl
from opencrab2.highlevel.contract import DEFAULT_CONTRACT
from opencrab2.highlevel.statistics import paired_bootstrap_ci, zero_failure_upper_bound

SIG = 3                      # significant digits of a formatted number
TABLE_NAMES: Tuple[str, ...] = ("catalog", "blocks", "costmodel", "dynamic", "dynamic_modeled", "factorial", "e2e", "web")
GENERATED_MARK = "generated by opencrab2.experiments.make_tables"
STATUS_RANK = {"MEASURED_NATIVE": 0, "EXTERNAL_REPORTED": 1, "HISTORICAL": 2, "DIAGNOSTIC": 3,
               "EMULATED_NUMERIC": 4, "ESTIMATED": 5, "MODELED": 6}
MIB = float(2 ** 20)
PERCENT = 100.0
MICRO = 1.0e6
PARTIES = 2
B0_BEST = "B0_fixed_best_hindsight"
ADAPTIVE = "B4_opencrab"     # the predictor prices every adaptive policy alike (one plan held for the session)
DYNAMIC_POLICIES: Tuple[Tuple[str, str], ...] = (("B1", "B1_naive"), ("B2", "B2_remaining_plus_install"),
                                                 ("B4", "B4_opencrab"))
DIAG4_CELLS: Tuple[Tuple[str, str, str], ...] = (
    ("S0", "stable-LAN", "S0 stable LAN"), ("S0", "stable-WAN", "S0 stable WAN"),
    ("S1", "ramp-WAN-c-to-LAN", "S1 ramp"), ("S2", "rtt-step-up", "S2 RTT only"),
    ("S3", "two-changes", "S3 two changes"), ("S8", "cold-LAN", "S8 cold"), ("S8", "warm-all-LAN", "S8 warm installed"))
# the full-model rows of the dynamic table, printed after the prefix rows; the label prefix 12L keeps them apart
FULL12_MATRIX = "full12"
FULL12_CELLS: Tuple[Tuple[str, str, str], ...] = (("S0", "stable-LAN", "12L S0 stable LAN"),
                                                 ("S1", "ramp-WAN-c-to-LAN", "12L S1 ramp"))
NOT_RUN = "-"                # a policy that did not run in a full12 cell (never the "--" of a missing number)
E2E_ROWS: Tuple[Tuple[str, str, str], ...] = (
    ("full12", "B0_fixed:wmask", "full 12L, B0 wmask"), ("full12", "B2_remaining_plus_install", "full 12L, B2"),
    ("full12", "B4_opencrab", "full 12L, B4"), ("diag4", "B0_fixed:wmask", "prefix 4L, B0 wmask"),
    ("diag4", "B2_remaining_plus_install", "prefix 4L, B2"), ("diag4", "B4_opencrab", "prefix 4L, B4"))
E2E_TRACE: Tuple[str, str] = ("S0", "stable-LAN")
WEB_WORKLOADS: Tuple[Tuple[str, str], ...] = (("tiny", "tiny"), ("gpt2-1L", "GPT-2 1L"), ("gpt2-4L", "GPT-2 4L"))
REJECTION_KINDS: Tuple[str, ...] = ("origin", "token", "version", "stale")
COSTMODEL_ROWS: Tuple[Tuple[str, str], ...] = (("handwritten", "hand-written"), ("fitted", "fitted"),
                                               ("uncertainty", "uncertainty-aware"))
COSTMODEL_METRICS: Tuple[str, ...] = ("n", "mape", "median_rel_err", "p95_rel_err", "spearman", "top1_lane_acc",
                                      "plan_identity", "realized_regret_s", "informative_pair_correct",
                                      "informative_pair_decisions", "coverage_2sigma")
PLATFORM_BENCHES: Tuple[str, ...] = ("wire_codec_roundtrip", "hmac_sha256", "ws_loopback_roundtrip",
                                     "dealer_beaver_pair_plus_combine")
FAMILIES: Tuple[str, ...] = ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8")
VOCAB = int(cm.GPT2_CARD["vocab"])
BLOCKS: Tuple[Tuple[str, str], ...] = (
    ("drelu_p1", "DReLU, one payload"), ("drelu_p2", "DReLU, two payloads"), ("trunc_faithful", "TruncFaithful"),
    ("ring_to_field", "RingToField"), ("a2h_exact", "ExactA2H encrypt-and-add"),
    ("argmax_k2", f"argmax over {VOCAB}, k = 2"), ("argmax_k4", f"argmax over {VOCAB}, k = 4"),
    ("argmax_k8", f"argmax over {VOCAB}, k = 8"), ("toy_bridge", "toy bridge H2A-ref"))
BLOCK_KEYS: Tuple[str, ...] = tuple(k for k, _ in BLOCKS)
A2H_STEPS: Tuple[str, ...] = ("bfv_encrypt_share", "bfv_add_share")     # the online encrypt-and-add of ExactA2H
TOY_SETUP_STEPS: Tuple[str, ...] = ("toy_keygen",)                       # one-time, not part of a conversion
TOY_ONLINE_STEPS = 3                                                     # encrypt, plaintext product, decrypt
# the toy bridge row is the local procedure (encrypt, plaintext product, decrypt) in both languages; the C++
# two-party reference (toy_bridge_h2a_ref) is another procedure and has no Python counterpart
TOY_LOCAL_COMPONENTS: Tuple[str, ...] = ("toy_bridge", "toy_bridge_local")
PARAM_LABELS: Tuple[str, ...] = ("step", "phase", "description", "realization")   # record parameters that name, not size
SOLVER_TIMINGS = 5                                                       # repetitions of the planner timing
PIPELINE_MATRICES: Tuple[str, ...] = ("smoke",)                          # pipeline checks: evidence for no hypothesis
ALPHA = 0.05                                                             # two-sided level of the Student t interval
DERIVED_POLICIES: Tuple[str, ...] = ("B4_no_install", "delayed:2")       # ablations the prose cites from the header
DERIVED_FAMILIES: Tuple[str, ...] = ("S1", "S3")
MARKING = ("marking: plain = measured by this release; est = expected or derived; diag = measured with a test-only "
           "component")
MARKING_EXTRA = {"HISTORICAL": "hist = historical record", "EXTERNAL_REPORTED": "ext = externally reported value"}
ELIGIBILITY_NOTE = ("security_eligible does not drive the marking: a record with security_eligible false that is a plain "
                    "in-process timing stays plain")


# ---------------------------------------------------------------------------
# cells and formatting
# ---------------------------------------------------------------------------
@dataclass
class Cell:
    text: str
    status: Optional[str] = None       # None = structural text (labels), never wrapped
    expected: bool = False             # True = no measured record yet: a computed expectation, printed in est


def expd(text: str) -> Cell:
    """An expected value (no measured record yet): wrapped in est and counted as expected in the header."""
    return Cell(text, "ESTIMATED", True)


def weakest(statuses: Iterable[str]) -> str:
    best = None
    for s in statuses:
        if best is None or STATUS_RANK.get(s, len(STATUS_RANK)) > STATUS_RANK.get(best, len(STATUS_RANK)):
            best = s
    return best or "MODELED"


def wrap(text: str, status: Optional[str]) -> str:
    if status is None:
        return text
    if schema.is_estimate(status):
        return "\\est{" + text + "}"
    if schema.is_historical(status):
        return "\\hist{" + text + "}"
    if status == "DIAGNOSTIC":
        return "\\diag{" + text + "}"
    if status == "EXTERNAL_REPORTED":
        return "\\ext{" + text + "}"
    return text


def fmt(v: Any) -> str:
    """Plain number text (no TeX macro): ints as digits, floats with SIG significant digits, missing as --."""
    if v is None:
        return "--"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    try:
        x = float(v)
    except (TypeError, ValueError):
        return tex_escape(str(v))
    if not math.isfinite(x):
        return "--"
    if x == 0.0:
        return "0"
    if abs(x) >= 1e5 or abs(x) < 1e-3:
        return f"{x:.{SIG - 1}e}".replace("e+0", "e").replace("e-0", "e-")
    if abs(x) >= 10 ** SIG:
        return str(int(float(f"{x:.{SIG}g}")))      # SIG significant digits, written out (1010, not 1.01e+03)
    return f"{x:.{SIG}g}"


def fmt_ci(mean: float, lo: float, hi: float) -> str:
    return f"{fmt(mean)} [{fmt(lo)}, {fmt(hi)}]"


def fmt_range(vals: Sequence[float]) -> str:
    """Mean with [min, max] over the values (a single value repeats itself as the range)."""
    xs = [float(v) for v in vals]
    return fmt_ci(float(np.mean(xs)), min(xs), max(xs))


def student_t_ci(diffs: Sequence[float], alpha: float = ALPHA) -> Optional[Tuple[float, float, float]]:
    """(mean, lo, hi): two-sided Student t interval of the mean, n - 1 degrees of freedom; None below two values."""
    x = np.asarray([float(v) for v in diffs], dtype=np.float64)
    n = int(x.shape[0])
    if n < 2:
        return None
    half = float(_student_t.ppf(1.0 - alpha / 2.0, n - 1)) * float(np.std(x, ddof=1)) / math.sqrt(n)
    mean = float(x.mean())
    return mean, mean - half, mean + half


def tex_escape(s: str) -> str:
    return (s.replace("\\", "/").replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")
            .replace("#", "\\#").replace("{", "(").replace("}", ")"))


def _cell_kind(c: Cell) -> Optional[str]:
    if c.expected:
        return "expected"
    if c.status is None:
        return None
    return "measured" if c.status == "MEASURED_NATIVE" else "other"


def _count(cells: Iterable[Cell]) -> Dict[str, int]:
    out = {"expected": 0, "measured": 0, "other": 0}
    for c in cells:
        k = _cell_kind(c)
        if k is not None:
            out[k] += 1
    return out


COUNT_RE = re.compile(r"^% cells: (\d+) expected .*?, (\d+) measured .*?, (\d+) from ", re.M)


def cell_counts(text: str) -> Dict[str, int]:
    """``{expected, measured, other}`` of a generated table, read back from its header comment."""
    m = COUNT_RE.search(text)
    if m is None:
        return {"expected": 0, "measured": 0, "other": 0}
    return {"expected": int(m.group(1)), "measured": int(m.group(2)), "other": int(m.group(3))}


def tabular(comment: Sequence[str], columns: Sequence[str], rows: Sequence[Sequence[Cell]],
            align: Optional[str] = None, sources: Optional[Mapping[str, str]] = None, count: bool = True) -> str:
    """A bare booktabs ``tabular`` preceded by the provenance comments (no float, no caption).

    After the caller's comment lines the header counts the expected, measured and other value cells of the
    table and of every value column; ``sources`` maps a column header to the rule its expected values follow;
    ``count=False`` is for a table of facts whose cells are labels (the catalog).
    """
    ncol = len(columns)
    al = align or ("l" + "r" * (ncol - 1))
    for r in rows:
        if len(r) != ncol:
            raise ValueError("TABLES_INTERNAL: a row does not match the column count")
    src = dict(sources or {})
    tot = _count(c for r in rows for c in r if count)      # count=False: a table of facts, without value cells
    out = [f"% {GENERATED_MARK}; do not edit by hand"]
    out += ["% " + str(c).replace("\n", " ") for c in comment]
    present = {c.status for r in rows for c in r if c.status is not None}
    out.append("% " + "; ".join([MARKING] + [MARKING_EXTRA[s] for s in MARKING_EXTRA if s in present]))
    out.append(f"% cells: {tot['expected']} expected (no measured record yet; computed, never typed; printed in est), "
               f"{tot['measured']} measured (printed plain), {tot['other']} from modeled, derived or diagnostic records")
    first: Dict[str, int] = {}
    for j, col in enumerate(columns):
        cnt = _count(r[j] for r in rows if count)
        if not sum(cnt.values()) and col not in src:
            continue
        line = f"% column {j + 1} ({col}): {cnt['expected']} expected, {cnt['measured']} measured, {cnt['other']} other"
        if col in src:
            rule = str(src[col]).replace("\n", " ")
            line += "; an expected value is " + (rule if rule not in first else f"computed by the rule of column {first[rule]}")
            first.setdefault(rule, j + 1)
        out.append(line)
    out += ["\\begin{tabular}{" + al + "}", "\\toprule", " & ".join(columns) + " \\\\", "\\midrule"]
    for r in rows:
        out.append(" & ".join(wrap(c.text, c.status) for c in r) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(out)


def write_table(out_dir: Path, name: str, text: str) -> Path:
    """Write ``<name>.tex`` with LF line endings on every platform."""
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{name}.tex"
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return p


class _Lazy:
    """A mapping computed on first access (expectations are priced only for the cells that need one)."""

    def __init__(self, fn: Callable[[], Mapping[str, Any]]):
        self._fn = fn
        self._val: Optional[Mapping[str, Any]] = None

    def __getitem__(self, key: str) -> Any:
        if self._val is None:
            self._val = self._fn()
        return self._val[key]


# ---------------------------------------------------------------------------
# tolerant input
# ---------------------------------------------------------------------------
@dataclass
class RawFile:
    """Records of one raw JSONL file that passed the shape check, plus what was skipped (never raises).

    ``path`` is the plain ``<name>.jsonl`` that was asked for; ``source`` the file actually read (the plain file,
    or ``<name>.jsonl.gz`` when only the compressed file exists).
    """
    path: Path
    records: List[dict]
    lines: int = 0
    invalid: int = 0
    present: bool = False
    source: Optional[Path] = None

    def describe(self, used: Optional[int] = None) -> str:
        if not self.present:
            return f"{self.path.name}: missing"
        s = f"{(self.source or self.path).name}: {len(self.records)} records"
        if self.invalid:
            s += f" ({self.invalid} malformed lines skipped)"
        if used is not None:
            s += f", {used} used"
        return s


def _shape_ok(rec: Any) -> bool:
    return (isinstance(rec, dict) and rec.get("schema") == schema.SCHEMA_ID and rec.get("status") in schema.STATUSES
            and isinstance(rec.get("extra", {}), dict) and isinstance(rec.get("run_id"), str)
            and all(k in rec for k in ("policy_id", "aborts", "bytes", "rounds", "T_cold_s", "T_warm_s", "T_online_s")))


def read_raw(path: Path, slim: Optional[Callable[[dict], dict]] = None) -> RawFile:
    """Read a raw file line by line; malformed lines and records outside the schema shape are counted, not raised.

    ``<name>.jsonl`` is read when it exists, else ``<name>.jsonl.gz`` (gzip); neither present = a missing file.
    """
    rf = RawFile(Path(path), [])
    try:
        src = schema.raw_file(rf.path)
        if src is None:
            return rf
        rf.present = True
        rf.source = src
        opener = gzip.open if src.suffix == ".gz" else open
        with opener(src, "rt", encoding="utf-8", errors="replace") as fh:   # type: ignore[call-overload]
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rf.lines += 1
                try:
                    rec = json.loads(line)
                except ValueError:
                    rf.invalid += 1
                    continue
                if not _shape_ok(rec):
                    rf.invalid += 1
                    continue
                rf.records.append(slim(rec) if slim is not None else rec)
    except (OSError, EOFError):        # unreadable file or a truncated gzip stream: what was read stays
        pass
    return rf


def _slim_simulation(rec: dict) -> dict:
    """Drop the per-epoch payload of a modeled record (the tables need the session totals only)."""
    e = rec.get("extra") or {}
    rec["extra"] = {k: v for k, v in e.items() if k not in ("epochs", "decisions")}
    return rec


def _num(v: Any) -> Optional[float]:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    x = float(v)
    return x if math.isfinite(x) else None


def _mean(vals: Sequence[Optional[float]]) -> Optional[float]:
    xs = [float(v) for v in vals if v is not None]
    return float(np.mean(xs)) if xs else None


def session_time(rec: Mapping[str, Any]) -> Optional[float]:
    """Measured (or modeled) wall time of a session: the runner's wall time, else T_cold, else T_warm."""
    e = rec.get("extra") or {}
    for v in (e.get("session_wall_s"), rec.get("T_cold_s"), rec.get("T_warm_s")):
        x = _num(v)
        if x is not None:
            return x
    return None


def _is_client_record(rec: Mapping[str, Any]) -> bool:
    party = (rec.get("extra") or {}).get("party")
    return party in (1, None)


def _completed(rec: Mapping[str, Any]) -> bool:
    e = rec.get("extra") or {}
    return int(rec.get("aborts", 0) or 0) == 0 and not e.get("abort_reason") and e.get("completed", True) is not False


def _latest_by(recs: Iterable[dict], key: Callable[[dict], Any]) -> Dict[Any, dict]:
    out: Dict[Any, dict] = {}
    for r in recs:                       # file order is append order: a later record replaces an earlier one
        out[key(r)] = r
    return out


def _plan_seq(rec: Mapping[str, Any]) -> Optional[Tuple[str, ...]]:
    """The plan keys the session executed, one per epoch (``extra.plans``, a list or its JSON text); None if absent."""
    v = (rec.get("extra") or {}).get("plans")
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except ValueError:
            return None
    if not isinstance(v, list):
        return None
    return tuple(str(x) for x in v)


def _pairs(pol_recs: Mapping[Any, dict], ref_recs: Mapping[Any, dict]) -> List[dict]:
    """The paired sessions of a policy against the reference (same seed and concurrency level).

    Each pair carries the percent difference of the session times, the two statuses, the policy's switch count
    and ``same_plans``: True/False when both records list the plan sequence they executed, else None.
    """
    out: List[dict] = []
    for k, r in pol_recs.items():
        ref = ref_recs.get(k)
        if ref is None:
            continue
        a, b = session_time(r), session_time(ref)
        if a is None or b is None or b <= 0.0:
            continue
        pa, pb = _plan_seq(r), _plan_seq(ref)
        out.append({"key": k, "diff": PERCENT * (a - b) / b, "statuses": [str(r["status"]), str(ref["status"])],
                    "switches": float(r.get("post_init_switches", 0) or 0),
                    "same_plans": (pa == pb) if (pa is not None and pb is not None) else None})
    return out


def _level_of(rec: Mapping[str, Any]) -> Tuple[Any, Any]:
    """(concurrent sessions, threads per process) of a record: the concurrency level that is never pooled."""
    e = rec.get("extra") or {}
    return e.get("concurrent_workers"), e.get("threads_per_process")


def _level_text(level: Tuple[Any, Any]) -> str:
    workers, threads = level
    w = f"{workers} concurrent sessions" if workers is not None else "concurrency level not recorded"
    t = f"{threads} threads per process" if threads is not None else "thread cap not recorded"
    return f"{w}, {t}"


def _single_level(recs: Sequence[dict]) -> Tuple[List[dict], Optional[Tuple[Any, Any]], int]:
    """(records of one concurrency level, that level, records left out): the level most records carry.

    Levels are never pooled: when a matrix holds records of several levels (a stray session or a partial rerun at
    another worker count), the level of the majority of its records is kept (the latest record's level on a tie)
    and the others are counted for the header.
    """
    if not recs:
        return [], None, 0
    counts: Dict[Tuple[Any, Any], int] = {}
    for r in recs:
        counts[_level_of(r)] = counts.get(_level_of(r), 0) + 1
    latest = {_level_of(r): i for i, r in enumerate(recs)}
    level = max(counts, key=lambda lv: (counts[lv], latest[lv]))
    kept = [r for r in recs if _level_of(r) == level]
    return kept, level, len(recs) - len(kept)


def _campaign(matrix: Any) -> bool:
    return matrix is not None and str(matrix) not in PIPELINE_MATRICES


def output_match_lines(records: Iterable[Mapping[str, Any]]) -> List[str]:
    """Header lines on the output match: per matrix the matching of the completed client sessions, then one line
    per mismatching session with its run id, first differing step and reference margin (or that the record lacks
    the two fields)."""
    client = [r for r in records if r.get("record_kind", "session") == "session" and _is_client_record(r)]
    done = [r for r in _latest_by(client, lambda r: r["run_id"]).values() if _completed(r)]
    by_matrix: Dict[str, List[dict]] = {}
    for r in done:
        by_matrix.setdefault(str((r.get("extra") or {}).get("matrix")), []).append(r)
    if not by_matrix:
        return ["output match: no completed session yet"]
    lines: List[str] = []
    for matrix in sorted(by_matrix, key=lambda m: (not _campaign(m), m)):
        recs = by_matrix[matrix]
        flags = [r.get("output_match") for r in recs]
        known = [f for f in flags if isinstance(f, bool)]
        s = (f"output match {matrix}" + ("" if _campaign(matrix) else " (pipeline check, evidence for no hypothesis)")
             + f": {sum(1 for f in known if f)} of {len(known)} completed sessions match the float reference")
        if len(known) < len(flags):
            s += f" ({len(flags) - len(known)} without an output_match flag)"
        lines.append(s)
        for r in recs:
            if r.get("output_match") is not False:
                continue
            e = r.get("extra") or {}
            step, margin = e.get("first_mismatch_step"), e.get("first_mismatch_margin")
            if isinstance(step, int) and not isinstance(step, bool):
                where = f"first differing step {step}, reference margin " + (fmt(margin) if _num(margin) is not None
                                                                              else "not in the record")
            else:
                where = "first differing step and reference margin not in the record (the fields first_mismatch_step " \
                        "and first_mismatch_margin are absent)"
            lines.append(f"mismatch {r['run_id']}: {where}")
    return lines


# ---------------------------------------------------------------------------
# expectations (MODELED; for the cells that have no measured record yet)
# ---------------------------------------------------------------------------
class _CachedModel:
    """``predict`` of a fitted model memoised on (component, shape, environment); what the predictor is handed."""

    def __init__(self, model: cm.FittedCostModel):
        self.model = model
        self.fits = model.fits
        self.label = model.label
        self._cache: Dict[tuple, Tuple[float, float]] = {}

    def predict(self, component: str, n: int, d_in: int, d_out: int, env: Any) -> Tuple[float, float]:
        e = cm.as_env(env)
        k = (component, n, d_in, d_out, e.bandwidth_bps, e.one_way_latency_s, e.cpu_share)
        v = self._cache.get(k)
        if v is None:
            v = self.model.predict(component, n, d_in, d_out, e)
            self._cache[k] = v
        return v

    def has(self, component: str) -> bool:
        return self.model.has(component)

    def status_of(self, component: str) -> str:
        return self.model.status_of(component)


_EXPECT_CACHE: Dict[tuple, Any] = {}     # process-wide: (model and rate token, quantity, cell) -> expectation


def _workload(cell: rs.Cell) -> Tuple[dict, Callable[..., dict], dict]:
    """(workload, shapes of an epoch, catalog) of a cell, with the epoch shapes of ``run_sessions.predict_cell``."""
    wl = rs.cell_workload(cell)
    n_prompt = len(wl["prompt_ids"])
    bucket = int(wl["length_bucket"])

    def shapes_for(epoch: int, rows: Optional[int] = None) -> dict:
        n = (n_prompt if epoch == 0 else 1) if rows is None else int(rows)
        kv = n_prompt + epoch
        kv = max(bucket, int(math.ceil(kv / bucket)) * bucket)
        return cands.gpt2_shapes(cell.n_layers, n, kv, **wl["dims"])

    return wl, shapes_for, cands.gpt2_catalog(shapes_for(0))


def _state_for(plan: Mapping[str, Any], bucket: int, epochs: int) -> cands.SessionState:
    return cands.SessionState(incumbent=dict(plan), installed={}, stock={}, kv_bucket=int(bucket),
                              remaining_epochs=int(epochs), session_id="expected", model_version="expected",
                              contract_hash=DEFAULT_CONTRACT.contract_hash())


class Expectations:
    """Computed expectations for cells without a measured record (module docstring). Every value is MODELED.

    The model is ``<out_root>/calibration/fitted_model.json`` when present, else ``costmodel.default_model()``
    (the shipped calibration of this tree, or the hand-written baseline); a component the model
    has no fit for is priced with the hand-written baseline. The dealer rate is the ``dcf_gen`` rate of
    ``raw/primitives.jsonl`` (``run_sessions.material_rate_bytes_per_s``); before that file exists it is the
    hand-written DCF level rate applied to the keys of one DReLU gate.
    """

    def __init__(self, out_root: Optional[Path] = None):
        self.out_root = Path(out_root) if out_root is not None else None
        model: Optional[cm.FittedCostModel] = None
        self.model_source = "costmodel.default_model()"
        if self.out_root is not None:
            p = self.out_root / "calibration" / "fitted_model.json"
            try:
                if p.is_file():
                    model = cm.FittedCostModel.load(p)
                    self.model_source = "calibration/fitted_model.json"
            except (OSError, ValueError, KeyError, TypeError):
                model = None
        if model is None:
            model = cm.default_model()
        hand = cm.handwritten_model()
        filled = sorted(set(hand.fits) - set(model.fits))
        self.model = cm.FittedCostModel(fits={**hand.fits, **model.fits}, rel_bw=model.rel_bw, rel_rtt=model.rel_rtt,
                                        label=model.label)
        if filled:
            self.model_source += f" (hand-written baseline for {', '.join(filled)})"
        self.cached = _CachedModel(self.model)
        rate: Optional[float] = None
        try:
            rate = rs.material_rate_bytes_per_s(self.out_root) if self.out_root is not None else None
        except (OSError, ValueError, TypeError, KeyError):
            rate = None
        self.rate_source = "dcf_gen rate of raw/primitives.jsonl"
        if not rate:
            # two DCF keys of n levels, each level expanded for both parties, at the hand-written level rate
            levels = PARTIES * PARTIES * cm.DCF_DOMAIN_BITS
            rate = cm.drelu_material_bytes() / (levels * cm.HANDWRITTEN["dcf_level_s"])
            self.rate_source = "hand-written DCF level rate (raw/primitives.jsonl has no dcf_gen record)"
        self.rate = float(rate)
        blob = json.dumps(self.model.to_json(), sort_keys=True) + repr(self.rate)
        self.token = hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def describe(self) -> str:
        return (f"expected values (est): cost model {self.model_source}, label {self.model.label}; dealer rate "
                f"{fmt(self.rate / MIB)} MiB/s per party from the {self.rate_source}")

    def _memo(self, key: tuple, fn: Callable[[], Any]) -> Any:
        k = (self.token,) + key
        if k not in _EXPECT_CACHE:
            _EXPECT_CACHE[k] = fn()
        return _EXPECT_CACHE[k]

    # -- one session ----------------------------------------------------------
    def session(self, family: str, variant: str, policy: str, n_layers: int, T: int, model: str = rs.MODEL) -> dict:
        """Expected totals of one session cell: seconds through ``predict_cell``, bytes, rounds and material from
        the exact formulas, the analytic memory peak, and the predicted plan."""
        kind = policy if policy in rs.FIXED_PLANS else ADAPTIVE
        return self._memo(("session", family, variant, kind, int(n_layers), int(T), model),
                          lambda: self._session(rs.Cell("expected", family, variant, kind, 0, int(n_layers), int(T), model)))

    def _session(self, cell: rs.Cell) -> dict:
        p = rs.predict_cell(cell, self.cached, self.rate)
        wl, shapes_for, catalog = _workload(cell)
        plan = dict(p["predicted_plan"])
        nbytes = rounds = 0
        per_epoch: List[int] = []
        for e in range(cell.T):
            sh = shapes_for(e)
            b, r = cands.plan_bytes_rounds(plan, sh)
            # plan-independent messages of an epoch (candidates docstring): the client's one-hot input shares
            # and the released token index, one frame and one round each
            nbytes += b + cm.ONEWAY_BYTES * sh["n"] * sh["vocab"] + cm.ONEWAY_BYTES + PARTIES * cm.FRAME_BYTES
            rounds += r + PARTIES
            per_epoch.append(int(sum(cands.material_per_epoch(plan, sh, DEFAULT_CONTRACT).values())))
        material = int(sum(per_epoch))
        if not cell.warm:
            for o in cands.plan_objects(plan, catalog):
                d_in, d_out = catalog["objects"][o]
                nbytes += cm.bytes_formula("install_wmask", 1, d_in, d_out)
                rounds += cm.rounds_formula("install_wmask", 1, d_in, d_out)
                material += cm.material_formula("install_wmask", 1, d_in, d_out, DEFAULT_CONTRACT)
        state = _state_for(plan, int(wl["length_bucket"]), cell.T)
        mem = cands.mem_peak(plan, shapes_for(0), catalog, state, max(per_epoch) if per_epoch else 0)
        work = float(p["online_s"]) + float(p["sensing_s"]) + float(p["material_s"] or 0.0)
        return {"plan": plan, "T_cold_s": work + float(p["install_s"]), "T_warm_s": work, "T_online_s": float(p["online_s"]),
                "total_s": float(p["predicted_total_s"]) - float(p["priming_s"]), "bytes": int(nbytes),
                "rounds": int(rounds), "material": material, "mem_est": int(mem)}

    def switches(self, family: str, variant: str, n_layers: int, T: int, seeds: Sequence[int],
                 model: str = rs.MODEL) -> float:
        """Expected post-initialization switches: mean over the planned seeds of a walk of the remaining-cost rule.

        At every epoch the environment is the trace state at the predicted elapsed time; the plan is replaced
        when another plan has a strictly lower cost of the remaining epochs plus the installation of the objects
        it still lacks. The trace lasts ``TRACE_SAFETY`` times the predicted session, as in ``cell_trace``.
        """
        return self._memo(("switches", family, variant, int(n_layers), int(T), tuple(int(s) for s in seeds), model),
                          lambda: self._switches(family, variant, int(n_layers), int(T), seeds, model))

    def _switches(self, family: str, variant: str, n_layers: int, T: int, seeds: Sequence[int], model: str) -> float:
        base = self.session(family, variant, ADAPTIVE, n_layers, T, model)
        cell = rs.Cell("expected", family, variant, ADAPTIVE, 0, n_layers, T, model)
        _wl, shapes_for, catalog = _workload(cell)
        axes = catalog["axes"]
        plans = [dict(zip(axes, combo)) for combo in itertools.product(*axes.values())]
        shapes = [shapes_for(e) for e in range(T)]
        mats = [sum(cands.material_per_epoch(base["plan"], sh, DEFAULT_CONTRACT).values()) for sh in shapes]
        memo: Dict[tuple, float] = {}

        def exec_s(plan: dict, e: int, env: cm.Env) -> float:
            k = (cands.plan_key(plan), shapes[e]["n"], shapes[e]["T"], env)
            if k not in memo:
                memo[k] = cands.exec_cost(plan, shapes[e], self.cached, env)[0]
            return memo[k]

        counts: List[float] = []
        for seed in seeds or (0,):
            tr = traces.make_trace(family, variant, int(seed), rs.TRACE_SAFETY * max(base["total_s"], traces.RAMP_STEP_S))
            installed = set(catalog["objects"]) if cell.warm else set()
            incumbent: Optional[dict] = None
            t = 0.0
            n_sw = 0
            for e in range(T):
                st = traces.state_at(tr, t)
                env = cm.Env(float(st["bandwidth_bps"]), float(st["one_way_delay_s"]), float(st.get("cpu_share", 1.0) or 1.0))

                def missing(plan: dict) -> List[str]:
                    return [o for o in cands.plan_objects(plan, catalog) if o not in installed]

                def remaining(plan: dict) -> float:
                    return (sum(exec_s(plan, j, env) for j in range(e, T))
                            + cands.install_cost(missing(plan), catalog, self.cached, env)[0])

                best = min(plans, key=remaining)
                if incumbent is None:
                    incumbent = best
                elif best != incumbent and remaining(best) < remaining(incumbent):
                    incumbent = best
                    n_sw += 1
                t += cands.install_cost(missing(incumbent), catalog, self.cached, env)[0]
                installed.update(cands.plan_objects(incumbent, catalog))
                t += exec_s(incumbent, e, env) + rs._probe_cost(env) + mats[e] / self.rate
            counts.append(float(n_sw))
        return float(np.mean(counts))

    # -- measured against expected (header information and the memory baseline) ---
    def _measured_sessions(self, records: Iterable[Mapping[str, Any]]) -> List[Tuple[Mapping[str, Any], dict]]:
        out = []
        for r in records:
            e = r.get("extra") or {}
            if e.get("party") != 1 or not _completed(r) or r.get("status") != "MEASURED_NATIVE":
                continue
            try:
                s = self.session(str(r["family"]), str(r["variant"]), str(r["policy_id"]), int(r["n_layers"]), int(r["T"]),
                                 str(e.get("model", rs.MODEL)))
            except (KeyError, TypeError, ValueError):
                continue
            out.append((r, s))
        return out

    def observed_note(self, records: Iterable[Mapping[str, Any]]) -> str:
        """One header sentence: how far the completed campaign sessions were from the predictor, per matrix.

        Campaign sessions only: the pipeline matrices (``PIPELINE_MATRICES``) are counted and excluded. Per matrix
        the median and the range of measured over predicted session seconds, with the concurrency level named,
        never pooled across matrices.
        """
        ratios: Dict[str, List[float]] = {}
        levels: Dict[str, set] = {}
        skipped = 0
        for r, s in self._measured_sessions(_latest_by(records, lambda x: x["run_id"]).values()):
            wall = session_time(r)
            if not wall or s["total_s"] <= 0.0:
                continue
            matrix = str((r.get("extra") or {}).get("matrix"))
            if not _campaign(matrix):
                skipped += 1
                continue
            ratios.setdefault(matrix, []).append(wall / s["total_s"])
            levels.setdefault(matrix, set()).add(_level_of(r))
        if not ratios:
            return "no completed campaign session yet to compare the predictor with"
        parts = []
        for matrix in sorted(ratios):
            xs = ratios[matrix]
            lv = " or ".join(_level_text(l) for l in sorted(levels[matrix], key=str))
            parts.append(f"{matrix} ({lv}): median {fmt(float(np.median(xs)))}, range {fmt(min(xs))} to {fmt(max(xs))}, "
                         f"n = {len(xs)}")
        return ("measured over predicted session seconds of the completed campaign sessions, per matrix: "
                + "; ".join(parts)
                + (f"; {skipped} completed {'session' if skipped == 1 else 'sessions'} of the pipeline matrices "
                   f"({', '.join(PIPELINE_MATRICES)}) excluded" if skipped else "")
                + "; the predictor prices neither process start nor the contention of concurrent sessions, so scale "
                  "an expected time by the factor of its matrix before comparing")

    def rss_baseline(self, records: Iterable[Mapping[str, Any]], model: str = rs.MODEL) -> Tuple[float, int]:
        """(bytes, n): median of measured peak RSS minus the analytic estimate over the completed sessions of
        ``model``; (0, 0) without such a record."""
        diffs = []
        for r, s in self._measured_sessions(_latest_by(records, lambda x: x["run_id"]).values()):
            if str((r.get("extra") or {}).get("model", rs.MODEL)) != model:
                continue
            peak = max(int(r.get("peak_rss_client") or 0), int(r.get("peak_rss_server") or 0))
            if peak > 0:
                diffs.append(max(0.0, float(peak - s["mem_est"])))
        return (float(np.median(diffs)), len(diffs)) if diffs else (0.0, 0)

    # -- web ---------------------------------------------------------------------
    def web(self, label: str) -> dict:
        """Expected Web metrics of a workload on the unshaped loopback link (seconds; bytes per party)."""
        return self._memo(("web", label), lambda: self._web(label))

    def _web(self, label: str) -> dict:
        model, _sep, tail = label.partition("-")
        n_layers = int(tail.rstrip("L")) if tail else 1
        T = int(run_web.build_parser().get_default("n_generate"))
        fam, var = E2E_TRACE
        cell = rs.Cell("expected", fam, var, run_web.DEFAULT_POLICY, 0, n_layers, T, model)
        plan = self.session(fam, var, run_web.DEFAULT_POLICY, n_layers, T, model)["plan"]
        wl, shapes_for, catalog = _workload(cell)
        env = cm.Env(0.0, 0.0, 1.0)                       # unshaped loopback, as the priming session of predict_cell
        prefill = cands.exec_cost(plan, shapes_for(0), self.cached, env)[0]
        decode = [cands.exec_cost(plan, shapes_for(e), self.cached, env)[0] for e in range(1, T)] or [prefill]
        install = cands.install_cost(cands.plan_objects(plan, catalog), catalog, self.cached, env)[0]
        setup = sum(cands.material_per_epoch(plan, shapes_for(0), DEFAULT_CONTRACT).values()) / self.rate
        resumed = shapes_for(1, rows=len(wl["prompt_ids"]) + 1)      # the prompt plus the token released before the cut
        return {"ttft_cold_s": setup + install + prefill, "ttft_warm_s": prefill, "itl_mean_s": float(np.mean(decode)),
                "completion": float(T) / float(T),
                "cancel_waste_bytes": float(sum(cands.material_per_epoch(plan, shapes_for(1), DEFAULT_CONTRACT).values())),
                "recovery_s": rs._probe_cost(env) + cands.exec_cost(plan, resumed, self.cached, env)[0]}

    # -- building blocks -----------------------------------------------------------
    def block_formula(self, key: str) -> Tuple[float, int]:
        """(online bytes per element and party, rounds) of a block from the exact formulas."""
        n = cm.BFV_N
        single = cm.rounds_formula("trunc_faithful", 1, 1, 1)        # one opening exchange
        if key in ("drelu_p1", "drelu_p2", "toy_bridge"):            # one opened ring element per element
            return cm.OPEN_BYTES / PARTIES, single
        if key == "a2h_exact":                                       # the ciphertext message alone (RingToField has its row)
            b = cm.payload_bytes("a2h_exact", 1, n, n) - cm.payload_bytes("ring_to_field", 1, n, n)
            return b / n / PARTIES, cm.rounds_formula("a2h_exact", 1, n, n) - cm.rounds_formula("ring_to_field", 1, n, n)
        if key.startswith("argmax_k"):
            return cm.payload_bytes(key, 1, VOCAB, 1) / VOCAB / PARTIES, cm.rounds_formula(key, 1, VOCAB, 1)
        return cm.payload_bytes(key, 1, n, n) / n / PARTIES, cm.rounds_formula(key, 1, n, n)

    def block_python_s(self, key: str) -> Tuple[float, str]:
        """(seconds per element, rule) expected of the Python block without a record of either language."""
        n = cm.BFV_N
        h = cm.HANDWRITTEN
        if key.startswith("argmax_k"):
            return self.model.compute_s(key, 1, VOCAB, 1) / VOCAB, "fitted model"
        if key in ("trunc_faithful", "ring_to_field"):
            return self.model.compute_s(key, 1, n, n) / n, "fitted model"
        if key == "a2h_exact":
            d = (self.model.compute_s("a2h_exact", 1, n, n) - self.model.compute_s("ring_to_field", 1, n, n)) / n
            return (d, "fitted model") if d > 0.0 else (h["bfv_ct_s"] / n, "hand-written")
        if key == "toy_bridge":
            return TOY_ONLINE_STEPS * h["bfv_ct_s"] / n, "hand-written"
        payloads = PARTIES if key == "drelu_p2" else 1
        levels = PARTIES * cm.DCF_DOMAIN_BITS * h["dcf_level_s"]     # two DCF evaluations of n levels
        return (levels * cm.dcf_key_bytes(cm.DCF_DOMAIN_BITS, payloads) / cm.dcf_key_bytes(cm.DCF_DOMAIN_BITS, 1),
                "hand-written")

    # -- cost model ------------------------------------------------------------------
    def costmodel_selfcheck(self) -> Dict[str, dict]:
        """Hold-out metrics expected without any calibration file: the two models against each other.

        Points: every declared GPT-2 formula shape on the four network anchors; the fitted prediction stands in for
        the measurement. Hand-written row: its error, rank correlation, lane and plan agreement and regret
        against the fitted model. Fitted (and uncertainty-aware) row: the relative error of a Gaussian residual
        with the fitted sigma (mean, 0.75 and 0.975 normal quantiles), full agreement with itself and the
        two-sigma mass of a Gaussian.
        """
        return self._memo(("selfcheck",), self._selfcheck)

    def _selfcheck(self) -> Dict[str, dict]:
        fitted, hand = self.cached, cm.handwritten_model()
        envs = [cm.Env(bw, dl, 1.0) for bw, dl in traces.ANCHORS.values()]
        small = {(c,) + tuple(shape) for c, shapes in cm.FORMULA_TEST_SHAPES.items() for shape in shapes}
        pts = [p for p in cm.formula_shapes() if p not in small]       # the GPT-2 shapes, not the tiny test shapes
        y, sg, hw, hs = [], [], [], []
        for env in envs:
            for c, n, d_in, d_out in pts:
                mu, s = fitted.predict(c, n, d_in, d_out, env)
                y.append(mu)
                sg.append(s)
                hw.append(cm.handwritten_predict(c, n, d_in, d_out, env))
                hs.append(hand.predict(c, n, d_in, d_out, env)[1])
        y, sg, hw, hs = (np.asarray(v, dtype=np.float64) for v in (y, sg, hw, hs))
        lanes = [(n, d_in, d_out) for c, n, d_in, d_out in pts if c == "linear_wmask"
                 and ("linear_generic", n, d_in, d_out) in pts]
        agree = decisions = informative = informative_ok = 0
        for env in envs:
            for n, d_in, d_out in lanes:
                fw, sw = fitted.predict("linear_wmask", n, d_in, d_out, env)
                fg, sgn = fitted.predict("linear_generic", n, d_in, d_out, env)
                same = (fw < fg) == (cm.handwritten_predict("linear_wmask", n, d_in, d_out, env)
                                     < cm.handwritten_predict("linear_generic", n, d_in, d_out, env))
                decisions += 1
                agree += int(same)
                if abs(fw - fg) > PARTIES * math.hypot(sw, sgn):       # lanes further apart than two sigma
                    informative += 1
                    informative_ok += int(same)
        cell = rs.Cell("expected", E2E_TRACE[0], E2E_TRACE[1], ADAPTIVE, 0, sim.N_LAYERS, min(sim.DEFAULT_TS))
        _wl, shapes_for, catalog = _workload(cell)
        axes = catalog["axes"]
        plans = [dict(zip(axes, combo)) for combo in itertools.product(*axes.values())]
        same_plan, regret = [], []
        for env in envs:
            cost_f = [cands.exec_cost(p, shapes_for(0), fitted, env)[0] for p in plans]
            cost_h = [cands.exec_cost(p, shapes_for(0), hand, env)[0] for p in plans]
            same_plan.append(float(int(np.argmin(cost_f)) == int(np.argmin(cost_h))))
            regret.append(cost_f[int(np.argmin(cost_h))] - min(cost_f))
        em = cm.error_metrics(y, hw)
        rel = sg / np.maximum(np.abs(y), np.finfo(np.float64).tiny)
        nd = _stats.NormalDist()
        hand_row = {"n": int(len(y)), "mape": em["mape"], "median_rel_err": em["median_rel_err"],
                    "p95_rel_err": em["p95_rel_err"], "spearman": em["spearman"],
                    "top1_lane_acc": agree / max(1, decisions), "plan_identity": float(np.mean(same_plan)),
                    "realized_regret_s": float(np.mean(regret)), "informative_pair_correct": informative_ok,
                    "informative_pair_decisions": informative,
                    "coverage_2sigma": float(np.mean(np.abs(hw - y) <= PARTIES * hs))}
        fit_row = {"n": int(len(y)), "mape": float(np.mean(rel)) * math.sqrt(PARTIES / math.pi),
                   "median_rel_err": float(np.median(rel)) * nd.inv_cdf(0.75),
                   "p95_rel_err": float(np.percentile(rel, 95)) * nd.inv_cdf(0.975), "spearman": cm.spearman(y, y),
                   "top1_lane_acc": decisions / max(1, decisions), "plan_identity": float(len(envs)) / len(envs),
                   "realized_regret_s": float(min(regret) - min(regret)), "informative_pair_correct": informative,
                   "informative_pair_decisions": informative,
                   "coverage_2sigma": nd.cdf(float(PARTIES)) - nd.cdf(-float(PARTIES))}
        return {"handwritten": hand_row, "fitted": fit_row, "uncertainty": dict(fit_row)}

    # -- modeled campaign and factorial ---------------------------------------------------
    def family(self, fam: str, T: int) -> dict:
        """Expected row of the modeled campaign for a family without records: the predictor over its variants."""
        return self._memo(("family", fam, int(T)), lambda: self._family(fam, int(T)))

    def _family(self, fam: str, T: int) -> dict:
        ss = [self.session(fam, v, ADAPTIVE, sim.N_LAYERS, T) for v in traces.variants(fam)]
        total = float(np.mean([s["total_s"] for s in ss]))
        # the best fixed plan in hindsight and the adaptive policies hold the same predicted plan: the
        # difference is the predictor's own, and the plans held within one session number one
        held = float(np.mean([len({cands.plan_key(s["plan"])}) for s in ss]))
        return {"b0_s": total, "pct": PERCENT * (total - total) / total, "switches": held - min(held, 1.0),
                "n": len(ss) * sim.DEFAULT_SEEDS}

    def factorial(self) -> dict:
        """Expected factorial row per planner: campaign-mean predicted seconds, and one timed solve of the chain."""
        return self._memo(("factorial",), self._factorial)

    def _factorial(self) -> dict:
        specs = [(f, v) for f in FAMILIES for v in traces.variants(f)]
        seconds = float(np.mean([self.session(f, v, ADAPTIVE, sim.N_LAYERS, T)["total_s"]
                                 for f, v in specs for T in sim.DEFAULT_TS]))
        fam, var = E2E_TRACE
        T0 = min(sim.DEFAULT_TS)
        cell = rs.Cell("expected", fam, var, ADAPTIVE, 0, sim.N_LAYERS, T0)
        wl, shapes_for, catalog = _workload(cell)
        env = rs._env_of(fam, var)
        plan = self.session(fam, var, ADAPTIVE, sim.N_LAYERS, T0)["plan"]
        state = _state_for(plan, int(wl["length_bucket"]), T0)
        chain, pcat = sim.build_chain(catalog, shapes_for(0), self.cached, env, T0)
        ex_assign, ex_cost = pl.ExactAssign().plan(chain, pcat, env, self.cached, state)
        out: Dict[str, Any] = {"seconds": seconds,
                               "n": len(specs) * len(sim.DEFAULT_TS) * sim.DEFAULT_SEEDS * len(sim.CONTROLS)}
        for name in sim.PLANNERS:
            try:
                planner = getattr(pl, name)()
                lat = []
                for _ in range(SOLVER_TIMINGS):
                    t0 = time.perf_counter()
                    assignment, cost = planner.plan(chain, pcat, env, self.cached, state)
                    lat.append(time.perf_counter() - t0)
                out[name] = {"solver_latency_mean_s": float(np.median(lat)),
                             "optimality_gap_mean_s": max(0.0, float(cost) - float(ex_cost)),
                             "identical_to_exact_rate": float(assignment == ex_assign)}
            except Exception:          # a planner whose solver is not installed: the exact planner stands in
                t0 = time.perf_counter()
                assignment, cost = pl.ExactAssign().plan(chain, pcat, env, self.cached, state)
                out[name] = {"solver_latency_mean_s": time.perf_counter() - t0,
                             "optimality_gap_mean_s": max(0.0, float(cost) - float(ex_cost)),
                             "identical_to_exact_rate": float(assignment == ex_assign)}
        return out


# ---------------------------------------------------------------------------
# catalog
# ---------------------------------------------------------------------------
def evidence_class(status: str) -> Cell:
    if status == "ADMITTED":
        return Cell("MEASURED\\_NATIVE", "MEASURED_NATIVE")
    if status == "TEST_ONLY":
        return Cell("DIAGNOSTIC", "DIAGNOSTIC")
    return Cell("none")


def catalog_table() -> str:
    cols = ["component", "kind", "status", "evidence class"]
    note = "this table lists registry facts, not experimental values: it has no expected cell"
    try:
        from opencrab2.highlevel import registry
        comps = list(registry.REGISTRY.values())
        version = str(getattr(registry, "REGISTRY_VERSION", "unversioned"))
    except Exception as exc:  # the registry is part of this tree; a broken import still yields a table
        return tabular([f"source: registry.REGISTRY could not be imported ({exc.__class__.__name__})", note],
                       cols, [[Cell("registry"), Cell("unavailable"), Cell("unavailable"), Cell("none")]], align="llll",
                       count=False)
    rows = [[Cell(tex_escape(c.name)), Cell(tex_escape(c.kind)), Cell(tex_escape(c.status)), evidence_class(c.status)]
            for c in comps]
    by_status: Dict[str, int] = {}
    for c in comps:
        by_status[c.status] = by_status.get(c.status, 0) + 1
    return tabular([f"source: opencrab2.highlevel.registry.REGISTRY version {version}, {len(comps)} components "
                    f"({', '.join(f'{k} {v}' for k, v in sorted(by_status.items()))}); no raw records are read",
                    "evidence class = strongest record status a run using the component may carry", note],
                   cols, rows, align="llll", count=False)


# ---------------------------------------------------------------------------
# blocks
# ---------------------------------------------------------------------------
def _block_of(e: Mapping[str, Any]) -> Optional[Tuple[str, str]]:
    """(block key, step) of a primitive record of either language; None for a record no row uses."""
    comp = str(e.get("component", ""))
    params = e.get("params") or {}
    step = str(params.get("step", "") or "")
    if not comp or e.get("layer") == "platform" or comp in PLATFORM_BENCHES or comp.endswith("_material"):
        return None
    if comp == "drelu":
        return ("drelu_p2" if params.get("with_product", True) else "drelu_p1"), ""
    if comp.startswith("toy_bridge"):
        if comp not in TOY_LOCAL_COMPONENTS:         # the two-party reference: another procedure, no Python counterpart
            return None
        return None if step in TOY_SETUP_STEPS else ("toy_bridge", step)
    if comp.startswith("a2h"):
        return ("a2h_exact", step) if (not step or step in A2H_STEPS) else None
    return (comp, "") if comp in BLOCK_KEYS else None


def _test_only_block(key: str) -> bool:
    return key.startswith("toy_bridge")


def _size_params(params: Mapping[str, Any]) -> Dict[str, Any]:
    """The parameters of a record that size the measurement (N, widths, bit lengths), without the labels."""
    return {str(k): v for k, v in (params or {}).items() if str(k) not in PARAM_LABELS}


def _params_text(params: Mapping[str, Any]) -> str:
    return ", ".join(f"{k} = {v}" for k, v in sorted(params.items())) or "no parameters recorded"


def _alike(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """Two records measure alike when every sizing parameter they both record has the same value."""
    return all(a[k] == b[k] for k in a if k in b)


def _block_entry(rec: Mapping[str, Any]) -> Optional[dict]:
    e = rec.get("extra") or {}
    batch = _num(e.get("batch"))
    per = _num(e.get("per_element_s"))
    if e.get("skipped") or batch is None or batch <= 0 or per is None:
        return None
    return {"batch": int(batch), "per_element_s": per, "bytes_per_element": float(rec.get("bytes", 0) or 0) / batch,
            "rounds": int(rec.get("rounds", 0) or 0), "status": str(rec["status"]),
            "component": str(e.get("component", "")), "params": _size_params(e.get("params") or {}),
            "eligible": rec.get("security_eligible")}


def blocks_table(py: RawFile, cpp: RawFile, expect: Optional[Expectations] = None) -> str:
    expect = expect or Expectations()
    cols = ["block", "online bytes / elem. / party", "rounds", "Python [us / elem.]", "C++ [us / elem.]"]
    steps: Dict[str, Dict[str, Dict[str, dict]]] = {}
    for lang, rf in (("py", py), ("cpp", cpp)):
        for rec in rf.records:
            where = _block_of(rec.get("extra") or {})
            ent = _block_entry(rec)
            if where is None or ent is None:
                continue
            if where[0].startswith("argmax_k") and ent["batch"] != VOCAB:
                # The row is the argmax over the whole vocabulary. A record of another size is a
                # different measurement and must not stand in for it (it did once, while the
                # benchmark was still writing its larger cases).
                continue
            slot = steps.setdefault(where[0], {}).setdefault(lang, {})
            cur = slot.get(where[1])
            if cur is None or ent["batch"] >= cur["batch"]:      # the largest batch, the latest record of it
                slot[where[1]] = ent
    used = {"py": 0, "cpp": 0}
    block: Dict[str, Dict[str, dict]] = {}
    for key, langs in steps.items():
        for lang, by_step in langs.items():
            ents = list(by_step.values())
            used[lang] += len(ents)
            status = weakest(e["status"] for e in ents)
            if _test_only_block(key):
                status = weakest([status, "DIAGNOSTIC"])
            params: Dict[str, Any] = {}
            for e in ents:
                params.update(e["params"])
            block.setdefault(key, {})[lang] = {
                "per_element_s": float(sum(e["per_element_s"] for e in ents)),          # the steps of one conversion
                "bytes_per_element": max(e["bytes_per_element"] for e in ents),         # one message, not one per step
                "rounds": max(e["rounds"] for e in ents), "status": status, "params": params,
                "batch": max(e["batch"] for e in ents), "components": sorted({e["component"] for e in ents}),
                "steps": [s for s in by_step if s], "eligible": sorted({str(e["eligible"]).lower() for e in ents})}
    # the C++ to Python ratio behind the expected cells: blocks measured in both languages, like with like (the
    # sizing parameters both records carry agree) and no test-only block (a diagnostic procedure, not a protocol one)
    in_ratio: List[str] = []
    left_out: List[str] = []
    ratios: List[float] = []
    for key, shown in BLOCKS:
        b = block.get(key, {})
        if "py" not in b or "cpp" not in b or b["py"]["per_element_s"] <= 0.0:
            continue
        if _test_only_block(key):
            left_out.append(f"{shown} (test-only procedure)")
        elif not _alike(b["py"]["params"], b["cpp"]["params"]):
            left_out.append(f"{shown} (parameters differ: Python {_params_text(b['py']['params'])}; C++ "
                            f"{_params_text(b['cpp']['params'])})")
        else:
            in_ratio.append(shown)
            ratios.append(b["cpp"]["per_element_s"] / b["py"]["per_element_s"])
    ratio = float(np.median(ratios)) if ratios else 1.0
    rows: List[List[Cell]] = []
    for key, shown in BLOCKS:
        p, c = block.get(key, {}).get("py"), block.get(key, {}).get("cpp")
        ref = p or c
        if ref is not None:
            wire = [Cell(fmt(ref["bytes_per_element"] / PARTIES), ref["status"]), Cell(fmt(ref["rounds"]), ref["status"])]
        else:
            b, r = expect.block_formula(key)
            wire = [expd(fmt(b)), expd(fmt(r))]
        if p is not None:
            py_s, py_cell = p["per_element_s"], Cell(fmt(p["per_element_s"] * MICRO), p["status"])
        else:
            py_s = c["per_element_s"] / ratio if c is not None and ratios else expect.block_python_s(key)[0]
            py_cell = expd(fmt(py_s * MICRO))
        cpp_cell = Cell(fmt(c["per_element_s"] * MICRO), c["status"]) if c is not None else expd(fmt(py_s * ratio * MICRO))
        rows.append([Cell(shown)] + wire + [py_cell, cpp_cell])
    if ratios:
        both = (f"{fmt(ratio)} (median over the {len(ratios)} blocks measured alike in both languages: "
                + ", ".join(in_ratio) + ")")
    else:
        both = f"{fmt(ratio)} (no block is measured alike in both languages yet: the Python expectation is repeated)"
    if left_out:
        both += "; measured in both languages but outside the ratio: " + "; ".join(left_out)
    toy = block.get("toy_bridge", {})
    toy_cells = []
    for lang, name in (("py", "Python"), ("cpp", "C++")):
        t = toy.get(lang)
        if t is None:
            toy_cells.append(f"{name} cell expected (no record)")
            continue
        what = (" + ".join(t["steps"]) + " of " if t["steps"] else "") + ", ".join(t["components"])
        toy_cells.append(f"{name} cell = {what} at {_params_text(t['params'])} (batch {t['batch']})")
    eligible = sorted({v for b in block.values() for lang in b.values() for v in lang["eligible"]})
    return tabular(
        ["inputs: raw/" + py.describe(used["py"]) + "; raw/" + cpp.describe(used["cpp"]),
         "one row per building block; per block and language the record with the largest batch (the latest one); "
         "bytes per element and party = bytes of the online phase over both directions / elements / 2; time per "
         "element in microseconds; ExactA2H = the steps " + " + ".join(A2H_STEPS) + "; toy bridge = every step but "
         + ", ".join(TOY_SETUP_STEPS) + " of the local procedure " + " or ".join(TOY_LOCAL_COMPONENTS)
         + " (the C++ two-party reference toy_bridge_h2a_ref is another procedure without a Python counterpart and is "
         "not printed), TEST_ONLY and printed as diagnostic; dcf_gen, dcf_eval, bfv_context and the material "
         "generators are setup or sub-steps and have no row",
         "toy bridge row: " + "; ".join(toy_cells) + "; the row enters no ratio",
         ELIGIBILITY_NOTE + f"; the primitive records carry security_eligible {' and '.join(eligible) or 'unset'}: "
         "in-process timings of the parties outside a session, printed plain",
         expect.describe(), "C++ to Python time ratio used for expectations: " + both],
        cols, rows,
        sources={cols[1]: "the exact formula of costmodel (opened payload over both directions / elements / 2; argmax "
                          f"on one row of {VOCAB} logits; ExactA2H = the ciphertext message alone; the toy bridge is "
                          "priced as one opened ring element)",
                 cols[2]: "the round formula of costmodel (tournament: 2 per level; a single opening exchange for "
                          "DReLU and the toy bridge; ExactA2H minus its RingToField round)",
                 cols[3]: "the fitted cost model's compute time per element for argmax, TruncFaithful, RingToField and "
                          "ExactA2H (minus RingToField); for a block the model has no component for, the C++ time "
                          "divided by the ratio above, else the hand-written DCF level and BFV rates",
                 cols[4]: "the Python time of the row (measured or expected) times the ratio above"})


# ---------------------------------------------------------------------------
# cost model
# ---------------------------------------------------------------------------
def split_model_status(name: str) -> Tuple[str, str]:
    """``<label>_<STATUS>`` -> (label, status); an unknown or missing suffix is MODELED, never measured."""
    for status in sorted(STATUS_RANK, key=len, reverse=True):
        suffix = "_" + status
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)], status
    return name, "MODELED"


def _sha256_file(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def costmodel_table(out_root: Path, expect: Optional[Expectations] = None) -> str:
    out_root = Path(out_root)
    expect = expect or Expectations(out_root)
    cols = ["model", "n", "MAPE", "median", "p95", "Spearman", "lane acc.", "plan acc.", "regret [s]", "informative pair",
            "2-sigma coverage"]
    path = out_root / "calibration" / "calibration_metrics.json"
    m: Dict[str, Any] = {}
    note = "calibration/calibration_metrics.json: missing"
    try:
        if path.is_file():
            with open(path, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                m = loaded
                note = (f"calibration/calibration_metrics.json: {m.get('n_train', '?')} train rows, "
                        f"{m.get('n_test', '?')} hold-out rows")
    except (OSError, ValueError):
        note = "calibration/calibration_metrics.json: unreadable"
    models = m.get("models") if isinstance(m.get("models"), dict) else {}
    formulas_now = _sha256_file(cm.formulas_path())         # the shipped file, whatever the data root
    formulas_then = (m.get("campaign") or {}).get("formulas_sha256") if isinstance(m.get("campaign"), dict) else None
    stale = bool(models) and (formulas_now is None or formulas_then != formulas_now)
    by_label = {split_model_status(k)[0]: (split_model_status(k)[1], v) for k, v in models.items() if isinstance(v, dict)}

    def test_split(label: str) -> dict:
        splits = by_label[label][1]
        return splits.get("test", {}) if isinstance(splits.get("test"), dict) else {}

    def row(shown: str, t: Mapping[str, Any], status: Optional[str], expected: bool, label: str) -> List[Cell]:
        if expected and any(_num(t.get(k)) is None for k in COSTMODEL_METRICS):     # absent, null or NaN in the file
            t = {**expect.costmodel_selfcheck()[label], **{k: v for k, v in t.items() if _num(v) is not None}}
        mk = expd if expected else (lambda s: Cell(s, status))
        inf = f"{int(t.get('informative_pair_correct', 0) or 0)}/{int(t.get('informative_pair_decisions', 0) or 0)}"
        return [Cell(shown), mk(fmt(t.get("n"))), mk(fmt(t.get("mape"))), mk(fmt(t.get("median_rel_err"))),
                mk(fmt(t.get("p95_rel_err"))), mk(fmt(t.get("spearman"))), mk(fmt(t.get("top1_lane_acc"))),
                mk(fmt(t.get("plan_identity"))), mk(fmt(t.get("realized_regret_s"))), mk(inf),
                mk(fmt(t.get("coverage_2sigma")))]

    rows: List[List[Cell]] = []
    origin: List[str] = []
    for label, shown in COSTMODEL_ROWS:
        if label in by_label:
            # every row of the file is the evaluation of that model on the same measured hold-out rows: a measured
            # evaluation, printed plain, whatever status label the file gives the model (the hand-written constants
            # were never fitted and carry MODELED there; their error on measured rows is still a measurement)
            rows.append(row(shown, test_split(label), "MEASURED_NATIVE", stale, label))
            origin.append(f"{shown}: " + ("the stale calibration" if stale else "measured evaluation on the hold-out rows")
                          + f" (model label in the file: {by_label[label][0]})")
        elif label == "uncertainty" and "fitted" in by_label:
            rows.append(row(shown, test_split("fitted"), None, True, label))
            origin.append(f"{shown}: expected, copied from the fitted row (it reads the fitted model; its regression "
                          "metrics equal the fitted model's by construction)")
        else:
            rows.append(row(shown, {}, None, True, label))
            origin.append(f"{shown}: expected, from the self-check of the two models")
    rule = m.get("split_rule") if isinstance(m.get("split_rule"), dict) else {}
    comment = ["inputs: " + note,
               f"hold-out declared before fitting: shapes {rule.get('holdout_shapes')}, network "
               f"{rule.get('holdout_network')}, n = {rule.get('holdout_n')}",
               "rows: " + "; ".join(origin),
               "the hand-written row is the measured evaluation of the hand-written constants on the same hold-out rows "
               "as the fitted row, not an expectation; a metric the file leaves empty (null or NaN) is undefined for "
               "that model and is printed as --; in a stale file it is filled from the self-check"]
    if stale:
        comment.append("STALE: this calibration does not carry the hash of the current calibration/formulas.json; every "
                       "cell is wrapped in est and counts as an expectation of the re-run of the calibration stage")
    source = ("the value of the previous (stale) calibration when the file has the row; for the uncertainty-aware model "
              "the fitted row; else, and for a metric the file leaves empty, the self-check of the hand-written and the "
              "fitted model on the declared formula shapes over the four network anchors (fitted prediction as the "
              "stand-in measurement; fitted row = relative error of a Gaussian residual with the fitted sigma)")
    return tabular(comment, cols, rows, sources={c: source for c in cols[1:]})


# ---------------------------------------------------------------------------
# dynamic tables (measured diag4 and modeled campaign)
# ---------------------------------------------------------------------------
def _pair_key(rec: Mapping[str, Any]) -> tuple:
    e = rec.get("extra") or {}
    return (rec.get("seed", rec.get("repeat")), e.get("concurrent_workers"), e.get("threads_per_process"))


def _pct_cell(pol_recs: Mapping[Any, dict], ref_recs: Mapping[Any, dict]) -> Tuple[Optional[Cell], int, List[float]]:
    """Mean paired difference to the reference in percent with the bootstrap interval; (cell, n pairs, switches)."""
    pairs = _pairs(pol_recs, ref_recs)
    if not pairs:
        return None, 0, []
    mean, lo, hi = paired_bootstrap_ci([p["diff"] for p in pairs])
    return (Cell(fmt_ci(mean, lo, hi), weakest(s for p in pairs for s in p["statuses"])), len(pairs),
            [p["switches"] for p in pairs])


def _pair_detail(pairs: Sequence[dict]) -> str:
    """Header text of one printed difference: the Student t interval and the identical-plan count over the pairs."""
    diffs = [p["diff"] for p in pairs]
    st = student_t_ci(diffs)
    text = (f"{fmt(st[0])}, Student t [{fmt(st[1])}, {fmt(st[2])}]" if st is not None
            else f"{fmt(float(np.mean(diffs)))}, no Student t interval at n = {len(diffs)}")
    flags = [p["same_plans"] for p in pairs]
    if all(f is None for f in flags):
        return text + ", plan sequences not in the records"
    known = [f for f in flags if f is not None]
    return text + f", identical plan sequence as B0 in {sum(1 for f in known if f)} of {len(known)} seeds"


def _dynamic_row(label: str, by_policy: Mapping[str, Mapping[Any, dict]], ref: Optional[Mapping[Any, dict]],
                 fallback: Any, not_run: Iterable[str] = ()) -> Tuple[List[Cell], Dict[str, List[dict]]]:
    """One row of the dynamic layout: label, B0 seconds, three percent cells, n, switches; plus, per policy with a
    measured difference, its pairs (for the header details).

    ``fallback`` maps ``b0`` and ``n`` to expected cells and ``pct`` and ``sw`` to per-policy expected cells and
    switch texts; it is consulted only for the cells the records do not fill. A policy in ``not_run`` did not run in
    the cell: its percent cell and its slot of the switch cell print ``NOT_RUN`` (no value, no expectation).
    """
    skip = set(not_run)
    cells = [Cell(label)]
    st_ref = weakest(str(r["status"]) for r in ref.values()) if ref else None
    cells.append(Cell(fmt(_mean([session_time(r) for r in ref.values()])), st_ref) if ref else fallback["b0"])
    n_pairs: List[int] = []
    sw_txt: List[str] = []
    sw_status: List[str] = []
    details: Dict[str, List[dict]] = {}
    for short, pol in DYNAMIC_POLICIES:
        if pol in skip:
            cells.append(Cell(NOT_RUN))
            sw_txt.append(NOT_RUN)
            continue
        recs = by_policy.get(pol, {})
        pairs = _pairs(recs, ref or {})
        cell, n, switches = _pct_cell(recs, ref or {})
        if cell is not None:
            cells.append(cell)
            n_pairs.append(n)
            sw_txt.append(fmt(float(np.mean(switches))))
            sw_status += [str(r["status"]) for r in recs.values()]
            details[short] = pairs
        else:
            cells.append(fallback["pct"][pol])
            sw_txt.append(fallback["sw"][pol])
    ran = sum(1 for _s, pol in DYNAMIC_POLICIES if pol not in skip)
    if not ran:                                         # no policy ran in the cell: nothing to count or to expect
        return cells + [Cell(NOT_RUN), Cell(" / ".join(sw_txt))], details
    cells.append(Cell(fmt(min(n_pairs)), st_ref) if n_pairs else fallback["n"])
    whole = len(n_pairs) == ran                         # a partly measured switch cell is an expectation as a whole
    cells.append(Cell(" / ".join(sw_txt), weakest(sw_status)) if whole else expd(" / ".join(sw_txt)))
    return cells, details


def _fixed_plan_name(policy: str) -> str:
    """'B0_fixed:wmask' -> 'all-wmask' (the fixed plan a B0 policy id names)."""
    _b0, _sep, tail = policy.partition(":")
    return f"all-{tail}" if tail else policy


DYNAMIC_COLUMNS = ["trace cell", "B0 best fixed [s]", "B1 diff. [\\%]", "B2 diff. [\\%]", "B4 diff. [\\%]", "n",
                   "switches B1 / B2 / B4"]


def _diag_fallback(expect: Expectations, matrix: str, fam: str, var: str, modeled: Sequence[dict]) -> dict:
    """Expected cells of one measured trace cell: modeled campaign records of the same cell when they exist,
    else the predictor."""
    m = rs.MATRICES[matrix]
    L, T, seeds = int(m["n_layers"]), int(m["T"][0]), list(m["seeds"])
    fixed = [p for p in m["policies"] if p in rs.FIXED_PLANS]
    b0 = min(expect.session(fam, var, p, L, T)["total_s"] for p in fixed)
    adaptive = expect.session(fam, var, ADAPTIVE, L, T)["total_s"]
    by_policy: Dict[str, Dict[Any, dict]] = {}
    for r in modeled:
        if r.get("family") == fam and r.get("variant") == var and r.get("T") == T and r.get("n_layers") == L:
            by_policy.setdefault(str(r["policy_id"]), {})[(r.get("variant"), r.get("repeat"))] = r
    ref = by_policy.get(B0_BEST, {})
    pct: Dict[str, Cell] = {}
    sw: Dict[str, str] = {}
    for _short, pol in DYNAMIC_POLICIES:
        cell, _n, switches = _pct_cell(by_policy.get(pol, {}), ref)
        if cell is not None:
            pct[pol], sw[pol] = expd(cell.text), fmt(float(np.mean(switches)))
        else:
            pct[pol] = expd(fmt(PERCENT * (adaptive - b0) / b0))
            sw[pol] = fmt(expect.switches(fam, var, L, T, seeds))
    return {"b0": expd(fmt(b0)), "pct": pct, "sw": sw, "n": expd(fmt(len(seeds)))}


def _measured_block(sessions: RawFile, matrix: str, cell_specs: Sequence[Tuple[str, str, str]], expect: Expectations,
                    modeled: Sequence[dict], mark_not_run: bool) -> dict:
    """Rows and header material of one measured matrix in the dynamic layout.

    B0, the pairs and the noise floor are formed within the matrix: records of another matrix never enter a row.
    With ``mark_not_run`` a policy that did not run in a cell prints ``NOT_RUN``: a policy the matrix does not plan,
    or a planned one without any session in a cell whose other policies have sessions; a cell without any record
    keeps the expectations of its planned policies. Returns the rows, the records used, the aborted and other-level
    counts, the concurrency level, the Student t detail lines, the single-fixed-plan cells (label, plan, why), the
    noise-floor differences and the not-run notes.
    """
    recs = [r for r in sessions.records if (r.get("extra") or {}).get("matrix") == matrix and _is_client_record(r)]
    recs = list(_latest_by(recs, lambda r: r["run_id"]).values())   # a rerun of a cell replaces its earlier record
    done = [r for r in recs if _completed(r)]
    aborted = len(recs) - len(done)
    done, level, other_level = _single_level(done)                     # one concurrency level, never pooled
    m = rs.MATRICES[matrix]
    rows: List[List[Cell]] = []
    used = 0
    details: List[str] = []
    only_fixed: List[Tuple[str, str, str]] = []
    noise: List[float] = []
    not_run_notes: List[str] = []
    for fam, var, label in cell_specs:
        cell_recs = [r for r in done if r.get("family") == fam and r.get("variant") == var]
        by_policy: Dict[str, Dict[Any, dict]] = {}
        for r in cell_recs:
            by_policy.setdefault(str(r["policy_id"]), {})[_pair_key(r)] = r
        planned = list(m["policies"]) + list(m["extra_policies"].get(fam, []))
        fixed = {p: v for p, v in by_policy.items() if p.startswith("B0_fixed")}
        ref = None
        if fixed:
            # best fixed plan in hindsight: lowest mean session time in this cell
            ref_pol = min(fixed, key=lambda p: _mean([session_time(r) for r in fixed[p].values()]) or math.inf)
            ref = fixed[ref_pol]
            if len(fixed) == 1:        # the other fixed plan was skipped: B0 is this plan, not a choice in hindsight
                label = label + "*"
                others = [_fixed_plan_name(p) for p in planned if p.startswith("B0_fixed") and p != ref_pol]
                why = ("the matrix plans no other fixed plan" if not others else
                       f"the matrix also plans {', '.join(others)}, with no completed session in the cell")
                only_fixed.append((label, _fixed_plan_name(ref_pol), why))
            # the noise floor: every policy session of the cell with the plan sequence of its seed's B0 session
            noise += [abs(p["diff"]) for pol, v in by_policy.items() if pol != ref_pol
                      for p in _pairs(v, ref) if p["same_plans"] is True]
        not_run: List[str] = []
        if mark_not_run:
            seen = {str(r["policy_id"]) for r in recs if r.get("family") == fam and r.get("variant") == var}
            for short, pol in DYNAMIC_POLICIES:
                if pol not in planned:
                    not_run.append(pol)
                    not_run_notes.append(f"{short} in {label} (not a policy of the matrix)")
                elif seen and pol not in seen:
                    not_run.append(pol)
                    not_run_notes.append(f"{short} in {label} (a policy of the matrix without a session in the cell)")
        used += sum(len(v) for v in by_policy.values())
        cells, det = _dynamic_row(label, by_policy, ref,
                                  _Lazy(lambda fam=fam, var=var: _diag_fallback(expect, matrix, fam, var, modeled)),
                                  not_run)
        rows.append(cells)
        if det:
            details.append(f"{label}: " + "; ".join(f"{short} {_pair_detail(det[short])}" for short in det))
    return {"matrix": matrix, "rows": rows, "used": used, "aborted": aborted, "level": level,
            "other_level": other_level, "details": details, "only_fixed": only_fixed, "noise": noise,
            "not_run": not_run_notes}


def _inputs_part(b: Mapping[str, Any]) -> str:
    """The inputs-line clause of one matrix: records used, sessions excluded, the concurrency level kept."""
    return (f"matrix {b['matrix']}, client (P1) records, {b['used']} used; {b['aborted']} aborted or incomplete sessions "
            "excluded"
            + (f"; concurrency level {_level_text(b['level'])}" if b["level"] is not None else "")
            + (f"; {b['other_level']} completed records of another concurrency level left out (levels are never pooled)"
               if b["other_level"] else ""))


def _student_intro(m: Mapping[str, Any], has_details: bool, prefix: str = "") -> str:
    return (f"{prefix}per printed difference, one line per cell: mean, Student t interval ({len(m['seeds']) - 1} degrees "
            f"of freedom at n = {len(m['seeds'])}) and whether the policy executed the identical plan sequence as "
            "B0 in every seed" + ("" if has_details else "; no measured difference yet"))


def _noise_line(noise: Sequence[float], prefix: str = "", scope: str = "the matrix") -> str:
    if noise:
        return (f"{prefix}noise floor: largest absolute paired difference among the {len(noise)} identical-plan pairs of "
                f"{scope} (a policy session with the plan sequence of its seed's B0 session): "
                f"{fmt(max(noise))} percent (median {fmt(float(np.median(noise)))})")
    return f"{prefix}noise floor: no identical-plan pair (plan sequences absent from the records or all different)"


def dynamic_table(sessions: RawFile, matrix: str = "diag4", sim_file: Optional[RawFile] = None,
                  expect: Optional[Expectations] = None) -> str:
    """The measured dynamic table: the rows of ``matrix`` (``DIAG4_CELLS``), then the rows of the full12 matrix
    (``FULL12_CELLS``, label prefix 12L) by the same rules within that matrix; a policy that did not run in a
    full-model cell prints ``NOT_RUN``."""
    expect = expect or Expectations()
    full_matrix = FULL12_MATRIX
    modeled = [r for r in (sim_file.records if sim_file is not None else [])
               if r.get("record_kind", "simulation") == "simulation" and _completed(r)]
    d = _measured_block(sessions, matrix, DIAG4_CELLS, expect, modeled, False)
    f = _measured_block(sessions, full_matrix, FULL12_CELLS, expect, modeled, True)
    rows = d["rows"] + f["rows"]
    m, fm = rs.MATRICES[matrix], rs.MATRICES[full_matrix]
    pct_rule = (f"the mean paired difference of the MODELED campaign records (raw/simulation.jsonl) of the same family, "
                f"variant, T = {m['T'][0]} and {m['n_layers']} layers, with their bootstrap interval, when such records "
                "exist; otherwise one number from the predictor, 100 x (predicted plan - cheaper fixed plan) / cheaper "
                "fixed plan, without an interval (the predictor holds one plan for the session and prices B1, B2 and B4 "
                "alike)")
    comment = ["inputs: raw/" + sessions.describe(d["used"] + f["used"]) + "; " + _inputs_part(d) + "; "
               + _inputs_part(f),
               "B0 = fixed plan with the lowest mean session time in the cell (hindsight); differences are paired "
               "within seed and concurrency level, mean percent with the 95 percent bootstrap interval"]
    if d["only_fixed"]:
        comment.append("* = only one fixed plan ran in the cell (the other was skipped by a declared matrix reduction), "
                       "so B0 is that plan and not the better of two: "
                       + "; ".join(f"{label} ({plan})" for label, plan, _why in d["only_fixed"]))
    comment.append(_student_intro(m, bool(d["details"])))
    comment += d["details"]
    comment.append(_noise_line(d["noise"]))
    full = (f"{full_matrix} rows (label prefix 12L; {fm['n_layers']} layers): B0 and the paired differences are formed "
            f"within the {full_matrix} matrix by the rules above, never across matrices, and an expected value follows "
            f"the rule of its column for the {full_matrix} matrix ({fm['n_layers']} layers, T = {fm['T'][0]})")
    if f["only_fixed"]:
        full += ("; * = only one fixed plan ran in the cell, so B0 is that plan and not the better of two: "
                 + "; ".join(f"{label} ({plan}; {why})" for label, plan, why in f["only_fixed"]))
    comment.append(full)
    if f["not_run"]:
        comment.append(f"{full_matrix} {NOT_RUN} = the policy has no session in the cell, printed without a value or an "
                       "expectation: " + "; ".join(f["not_run"]))
    comment.append(_student_intro(fm, bool(f["details"]), f"{full_matrix} "))
    comment += f["details"]
    comment.append(_noise_line(f["noise"], f"{full_matrix} ", f"the {full_matrix} matrix"))
    comment += output_match_lines(sessions.records)
    comment += [expect.describe(), expect.observed_note(sessions.records)]
    return tabular(
        comment,
        DYNAMIC_COLUMNS, rows,
        sources={DYNAMIC_COLUMNS[1]: "the predicted session seconds (run_sessions.predict_cell: online + sensing + "
                                     "installation + material generation; the warm cell without installation) of the "
                                     "cheaper of the fixed plans of the matrix",
                 DYNAMIC_COLUMNS[2]: pct_rule, DYNAMIC_COLUMNS[3]: pct_rule, DYNAMIC_COLUMNS[4]: pct_rule,
                 DYNAMIC_COLUMNS[5]: f"the planned number of paired sessions per cell (seeds {list(m['seeds'])})",
                 DYNAMIC_COLUMNS[6]: "the mean switch count of the matching MODELED campaign records when they exist; "
                                     "otherwise the mean over the planned seeds of a walk of the remaining-cost rule "
                                     "along the trace (a plan is replaced when another one has a strictly lower "
                                     "predicted cost of the remaining epochs plus missing installations); a cell with "
                                     "one policy still unmeasured is printed as an expectation as a whole"})


def dynamic_modeled_table(sim_file: RawFile, expect: Optional[Expectations] = None) -> str:
    expect = expect or Expectations()
    recs = [r for r in sim_file.records if r.get("record_kind", "simulation") == "simulation" and _completed(r)]
    rows: List[List[Cell]] = []
    used = 0
    Ts = sorted({r.get("T") for r in recs if isinstance(r.get("T"), int)}) or list(sim.DEFAULT_TS)

    def fallback(fam: str, T: int) -> dict:
        f = expect.family(fam, T)
        return {"b0": expd(fmt(f["b0_s"])), "n": expd(fmt(f["n"])),
                "pct": {pol: expd(fmt(f["pct"])) for _s, pol in DYNAMIC_POLICIES},
                "sw": {pol: fmt(f["switches"]) for _s, pol in DYNAMIC_POLICIES}}

    derived: List[str] = []
    for T in Ts:
        for fam in FAMILIES:
            fr = [r for r in recs if r.get("family") == fam and r.get("T") == T]
            by_policy: Dict[str, Dict[Any, dict]] = {}
            for r in fr:
                by_policy.setdefault(str(r["policy_id"]), {})[(r.get("variant"), r.get("repeat"))] = r
            ref = by_policy.get(B0_BEST)
            used += sum(len(v) for p, v in by_policy.items() if p == B0_BEST or p in dict((b, a) for a, b in DYNAMIC_POLICIES))
            row, _det = _dynamic_row(f"{fam}, T = {T}", by_policy, ref, _Lazy(lambda fam=fam, T=T: fallback(fam, int(T))))
            # the modeled campaign is never printed as a measurement, whatever the record says
            rows.append([c if (c.status is None or c.expected) else Cell(c.text, weakest([c.status, "MODELED"])) for c in row])
            if T == min(sim.DEFAULT_TS) and fam in DERIVED_FAMILIES:
                # derived figures the prose cites: the ablations against B0, same pairing as the printed columns
                for pol in DERIVED_POLICIES:
                    cell, n, _sw = _pct_cell(by_policy.get(pol, {}), ref or {})
                    derived.append(f"{pol} on {fam} at T = {T}: " + (f"{cell.text} (n = {n})" if cell is not None
                                                                      else "no campaign record"))
    held = ("the predictor's own figure for a family without campaign records: it holds one plan per session, the same "
            "for the best fixed plan in hindsight and for every adaptive policy, and prices neither material waits "
            "nor cancel and resume")
    return tabular(
        ["inputs: raw/" + sim_file.describe(used) + "; MODELED campaign (cost-model simulation), every value est",
         "B0 = best fixed plan in hindsight per (variant, seed); differences are paired per (variant, seed), "
         "mean percent with the 95 percent bootstrap interval",
         "derived (est, cited by the prose): mean paired difference to B0 in percent with the bootstrap interval of the "
         "ablations " + " and ".join(DERIVED_POLICIES) + ": "
         + ("; ".join(derived) if derived else f"no record of T = {min(sim.DEFAULT_TS)} yet"),
         "values of campaign records count as 'other' (MODELED records), not as expected: " + expect.describe()],
        ["trace family"] + DYNAMIC_COLUMNS[1:], rows,
        sources={DYNAMIC_COLUMNS[1]: f"the mean over the family's variants of the predicted session seconds "
                                     f"(run_sessions.predict_cell, {sim.N_LAYERS} layers, cheapest plan)",
                 DYNAMIC_COLUMNS[2]: held, DYNAMIC_COLUMNS[3]: held, DYNAMIC_COLUMNS[4]: held,
                 DYNAMIC_COLUMNS[5]: f"the planned pairs: variants of the family x {sim.DEFAULT_SEEDS} seeds",
                 DYNAMIC_COLUMNS[6]: held})


# ---------------------------------------------------------------------------
# factorial
# ---------------------------------------------------------------------------
def factorial_table(fac: RawFile, expect: Optional[Expectations] = None) -> str:
    expect = expect or Expectations()
    cols = ["planner", "B2 control [s]", "B4 control [s]", "solver [ms]", "opt. gap [s]", "same as exact", "n"]
    by_pc: Dict[Tuple[str, str], List[dict]] = {}
    for r in fac.records:
        e = r.get("extra") or {}
        if r.get("record_kind", "factorial") != "factorial" or "planner" not in e:
            continue
        by_pc.setdefault((str(e.get("planner")), str(e.get("control"))), []).append(r)
    planners = sorted({p for p, _c in by_pc}) or sorted(sim.PLANNERS)
    rows: List[List[Cell]] = []
    used = 0
    for planner in planners:
        ex_all = _Lazy(expect.factorial)
        ex_planner = _Lazy(lambda planner=planner: expect.factorial().get(planner) or expect.factorial()[sim.PLANNERS[0]])
        cells = [Cell(tex_escape(planner))]
        allr: List[dict] = []
        for control in ("B2", "B4"):
            rs_ = by_pc.get((planner, control), [])
            allr += rs_
            t = _mean([session_time(r) for r in rs_])
            cells.append(Cell(fmt(t), weakest(r["status"] for r in rs_)) if rs_ and t is not None
                         else expd(fmt(ex_all["seconds"])))
        used += len(allr)
        st = weakest(r["status"] for r in allr)

        def ex(name: str, scale: float = 1.0) -> Cell:
            v = _mean([_num((r.get("extra") or {}).get(name)) for r in allr])
            return Cell(fmt(v * scale), st) if v is not None else expd(fmt(ex_planner[name] * scale))

        cells += [ex("solver_latency_mean_s", 1e3), ex("optimality_gap_mean_s"), ex("identical_to_exact_rate"),
                  Cell(fmt(len(allr)), st) if allr else expd(fmt(ex_all["n"]))]
        rows.append(cells)
    secs = (f"the mean predicted session seconds (run_sessions.predict_cell, {sim.N_LAYERS} layers) over every trace "
            f"variant and T in {list(sim.DEFAULT_TS)}; the predictor does not separate planners or controllers")
    solve = ("taken from one solve of the four-operator chain of the stable-LAN prefill by that planner, run while this "
             "table is generated: its latency (median of repeated calls), its cost minus the exact planner's cost, "
             "and whether its assignment equals the exact one")
    return tabular(
        ["inputs: raw/" + fac.describe(used) + "; planner x controller on identical chains, catalog and calibration",
         "mean session time per controller; solver latency, optimality gap and identical-assignment rate pooled "
         "over both controllers",
         "values of campaign records count as 'other' (MODELED records), not as expected: " + expect.describe()],
        cols, rows,
        sources={cols[1]: secs, cols[2]: secs, cols[3]: solve, cols[4]: solve, cols[5]: solve,
                 cols[6]: f"the planned sessions per planner: trace variants x T x {sim.DEFAULT_SEEDS} seeds x "
                          f"{len(sim.CONTROLS)} controllers"})


def write_factorial_table(out_root: Path) -> Optional[Path]:
    """Rewrite ``tables/factorial.tex`` (called by run_simulation after a campaign)."""
    return write_table(Path(out_root) / "tables", "factorial",
                       factorial_table(read_raw(Path(out_root) / "raw" / "factorial.jsonl", _slim_simulation),
                                       Expectations(Path(out_root))))


# ---------------------------------------------------------------------------
# end to end anchors
# ---------------------------------------------------------------------------
def e2e_table(sessions: RawFile, expect: Optional[Expectations] = None) -> str:
    expect = expect or Expectations()
    cols = ["anchor", "$T_{cold}$ [s]", "$T_{warm}$ [s]", "$T_{online}$ [s]", "bytes [MiB]", "rounds", "material [MiB]",
            "peak RSS [MiB]", "output match", "n"]
    client = [r for r in sessions.records if _is_client_record(r)]
    client = list(_latest_by(client, lambda r: r["run_id"]).values())   # a rerun of a cell replaces its earlier record
    baseline = _Lazy(lambda: dict(zip(("bytes", "n"), expect.rss_baseline(sessions.records))))
    fam, var = E2E_TRACE
    # one concurrency level per matrix, never pooled: the level of the matrix's latest completed record
    levels: Dict[str, Tuple[List[dict], Optional[Tuple[Any, Any]], int]] = {}
    for matrix in sorted({mx for mx, _p, _l in E2E_ROWS}):
        levels[matrix] = _single_level([r for r in client if (r.get("extra") or {}).get("matrix") == matrix
                                        and _completed(r)])
    rows: List[List[Cell]] = []
    used = 0
    for matrix, policy, label in E2E_ROWS:
        m = rs.MATRICES[matrix]
        planned = len(m["seeds"])
        ex = _Lazy(lambda m=m, policy=policy: expect.session(fam, var, policy, int(m["n_layers"]), int(m["T"][0])))
        base = [r for r in levels[matrix][0] if r.get("policy_id") == policy]
        cold = list(_latest_by([r for r in base if r.get("family") == fam and r.get("variant") == var],
                               _pair_key).values())
        warm = [r for r in _latest_by([r for r in base if (r.get("extra") or {}).get("warm_start") is True], _pair_key).values()
                if (r.get("extra") or {}).get("T_warm_basis") == "measured-warm-request"]

        def rss_expected() -> Cell:
            return expd(fmt((ex["mem_est"] + baseline["bytes"]) / MIB))

        if not cold:
            tw = (Cell(fmt(_mean([_num(r.get("T_warm_s")) for r in warm])), weakest(str(r["status"]) for r in warm))
                  if warm and _mean([_num(r.get("T_warm_s")) for r in warm]) is not None else expd(fmt(ex["T_warm_s"])))
            used += len(warm)
            rows.append([Cell(label), expd(fmt(ex["T_cold_s"])), tw, expd(fmt(ex["T_online_s"])),
                         expd(fmt(ex["bytes"] / MIB)), expd(fmt(ex["rounds"])), expd(fmt(ex["material"] / MIB)),
                         rss_expected(), expd(f"{planned}/{planned}"), expd(fmt(planned))])
            continue
        used += len(cold) + len(warm)
        st = weakest(str(r["status"]) for r in cold)
        v_warm = _mean([_num(r.get("T_warm_s")) for r in warm]) if warm else None
        v_derived = _mean([_num(r.get("T_warm_s")) for r in cold])
        if v_warm is not None:       # the wall time of real warm requests (S8 warm cell of the same matrix and policy)
            tw = Cell(fmt(v_warm), weakest(str(r["status"]) for r in warm))
        elif v_derived is not None:  # cold minus setup and installation: derived, not a measured warm request
            tw = Cell(fmt(v_derived), weakest([st, "ESTIMATED"]))
        else:
            tw = expd(fmt(ex["T_warm_s"]))
        rss = [max(int(r.get("peak_rss_client") or 0), int(r.get("peak_rss_server") or 0)) for r in cold]
        match = [r.get("output_match") for r in cold if isinstance(r.get("output_match"), bool)]
        t_cold = _mean([_num(r.get("T_cold_s")) for r in cold])
        t_online = _mean([_num(r.get("T_online_s")) for r in cold])
        rows.append([Cell(label), Cell(fmt(t_cold), st) if t_cold is not None else expd(fmt(ex["T_cold_s"])), tw,
                     Cell(fmt(t_online), st) if t_online is not None else expd(fmt(ex["T_online_s"])),
                     Cell(fmt(_mean([float(r["bytes"]) for r in cold]) / MIB), st),
                     Cell(fmt(_mean([float(r["rounds"]) for r in cold])), st),
                     Cell(fmt(_mean([float(r.get("material_consumed", 0) or 0) for r in cold]) / MIB), st),
                     Cell(fmt(max(rss) / MIB), st) if rss and max(rss) > 0 else rss_expected(),
                     Cell(f"{sum(1 for x in match if x)}/{len(match)}", st) if match else expd(f"{planned}/{planned}"),
                     Cell(fmt(len(cold)), st)])
    wire = ("the exact formulas for the predicted plan: candidates.plan_bytes_rounds over the epochs, the installation "
            "messages of the plan's objects (costmodel install_wmask) and the two plan-independent messages of an epoch "
            "(the client's one-hot input shares and the released token index); softmax is priced without the public "
            "causal mask, so the prefill figure is an upper bound")
    conc = []
    for matrix, (_kept, level, other) in levels.items():
        if level is None:
            continue
        conc.append(f"{matrix} rows at {_level_text(level)}"
                    + (f" ({other} completed records of another level left out)" if other else ""))
    return tabular(
        ["inputs: raw/" + sessions.describe(used) + "; stable-LAN cell of each matrix, client (P1) records, means "
         "over seeds",
         "T_warm plain = wall time of real warm requests (S8 warm cell); T_warm in est next to measured cells = cold "
         "minus setup and installation (derived from the measured cold sessions, counted as 'other'); material = "
         "consumed bytes per party; peak RSS = larger of client and server",
         "concurrency: " + ("; ".join(conc) if conc else "no completed session yet")
         + "; one level per matrix, never pooled, so the full-model and the prefix rows are not compared with each other"]
        + output_match_lines(sessions.records)
        + [expect.describe(), expect.observed_note(sessions.records)],
        cols, rows,
        sources={cols[1]: "run_sessions.predict_cell of the row's cell: online + sensing + installation + material "
                          "generation (B0 = the fixed all-wmask plan; B2 and B4 = the cheapest plan of the catalog)",
                 cols[2]: "the same prediction without installation (online + sensing + material generation)",
                 cols[3]: "the predicted execution time of the plan over all epochs (online_s of predict_cell)",
                 cols[4]: wire, cols[5]: wire,
                 cols[6]: "candidates.material_per_epoch of the predicted plan over the epochs plus the installation "
                          "masks, per party (same upper bound on the prefill softmax)",
                 cols[7]: "candidates.mem_peak of the predicted plan (KV cache, installed masks, working set, one epoch "
                          "of staged material) plus the measured process baseline: the median of measured peak minus "
                          "that estimate over the completed GPT-2 sessions of any matrix, zero while there is none",
                 cols[8]: "the planned count, every planned session matching the reference continuation",
                 cols[9]: "the planned number of sessions per cell (the seeds of the matrix)"})


# ---------------------------------------------------------------------------
# web
# ---------------------------------------------------------------------------
def _cap_group(recs: Sequence[dict]) -> Tuple[List[dict], Optional[int], str]:
    """(records of one thread cap, that cap, header note) of a workload's records.

    Records of different thread caps are never pooled, nor records without the cap field with capped ones. The cap
    of the latest capped record is kept; records without the field are used only when no record carries a cap.
    """
    caps = [(r.get("extra") or {}).get("threads_per_process") for r in recs]
    capped = [c for c in caps if c is not None]
    if not capped:
        return list(recs), None, ("thread cap not in the records (they predate the threads_per_process field: uncapped "
                                  "processes)" if recs else "no record")
    cap = int(capped[-1])
    kept = [r for r, c in zip(recs, caps) if c == cap]
    left: Dict[str, int] = {}
    for c in caps:
        if c != cap:
            k = "without the field" if c is None else f"cap {c}"
            left[k] = left.get(k, 0) + 1
    procs = sorted({str((r.get("extra") or {}).get("colocated_processes")) for r in kept})
    note = f"thread cap {cap} per process ({', '.join(procs)} co-located processes)"
    if left:
        note += "; left out: " + ", ".join(f"{n} records {k}" for k, n in sorted(left.items()))
    return kept, cap, note


def _repeat_path(rec: Mapping[str, Any]) -> str:
    """One cold_warm repeat: its TTFTs and the first-epoch plan path of the warm request, from the driver's fields."""
    e = rec.get("extra") or {}
    s = f"repeat {rec.get('repeat')}: TTFT cold {fmt(_num(e.get('ttft_cold_s')))} s"
    if not e.get("warm_verified") or _num(e.get("ttft_warm_s")) is None:
        return s + ", second generation not a verified warm request"
    s += f", warm {fmt(_num(e.get('ttft_warm_s')))} s, warm request at epoch 0: "
    plan = e.get("second_epoch0_plan_key")
    if plan is None:
        return s + "plan path not in the record (the epoch0 fields are absent)"
    s += f"plan {plan}, " + ("switched" if e.get("second_epoch0_switched") else "kept")
    if e.get("second_epoch0_decision_reason") is not None:
        s += f" ({e.get('second_epoch0_decision_reason')})"
    return s


def web_table(web: RawFile, expect: Optional[Expectations] = None) -> str:
    expect = expect or Expectations()
    cols = ["workload", "TTFT cold [s]", "TTFT warm [s]", "inter-token [s]", "completion", "cancel waste [MiB]",
            "resume recovery [s]"] + [f"{k} rej." for k in REJECTION_KINDS] + ["n"]
    parser = run_web.build_parser()
    repeats, trials = int(parser.get_default("repeats")), int(parser.get_default("trials"))
    probes = repeats * trials
    recs = list(_latest_by([r for r in web.records if r.get("record_kind", "web") == "web"], lambda r: r["run_id"]).values())
    rows: List[List[Cell]] = []
    used = 0
    paths: List[str] = []
    for label, shown in WEB_WORKLOADS:
        mine, _cap, cap_note = _cap_group([r for r in recs if (r.get("extra") or {}).get("model_label") == label])
        ex = _Lazy(lambda label=label: expect.web(label))
        used += len(mine)

        def scen(name: str) -> List[dict]:
            return [r for r in mine if r.get("variant") == name and _completed(r)]

        def metric(rs_: Sequence[dict], key: str, scale: float = 1.0) -> Cell:
            v = _mean([_num((r.get("extra") or {}).get(key)) for r in rs_])
            return Cell(fmt(v * scale), weakest(str(r["status"]) for r in rs_)) if v is not None else expd(fmt(ex[key] * scale))

        def spread(rs_: Sequence[dict], key: str) -> Cell:
            """Mean with [min, max] over the repeats that carry the value; the expectation has no spread."""
            vals = [v for v in (_num((r.get("extra") or {}).get(key)) for r in rs_) if v is not None]
            return Cell(fmt_range(vals), weakest(str(r["status"]) for r in rs_)) if vals else expd(fmt(ex[key]))

        cw, cn, rc, rj = scen("cold_warm"), scen("cancel"), scen("reconnect"), scen("rejections")
        cells = [Cell(shown), spread(cw, "ttft_cold_s"), spread(cw, "ttft_warm_s"), metric(cw, "itl_mean_s"),
                 metric(cw + rc, "completion"), metric(cn, "cancel_waste_bytes", 1.0 / MIB), metric(rc, "recovery_s")]
        if mine:
            paths.append(f"{shown}: {cap_note}; " + ("; ".join(_repeat_path(r) for r in sorted(cw, key=lambda r: r["repeat"]))
                                                   if cw else "no completed cold_warm record"))
        for kind in REJECTION_KINDS:
            n_try = sum(int((r.get("extra") or {}).get(f"rejected_{kind}_trials", 0) or 0) for r in rj)
            wrong = sum(int((r.get("extra") or {}).get(f"rejected_{kind}_accepted_wrongly", 0) or 0) for r in rj)
            cells.append(Cell(f"{n_try - wrong}/{n_try}", weakest(str(r["status"]) for r in rj)) if n_try
                         else expd(f"{probes}/{probes}"))
        cells.append(Cell(fmt(len(cw)), weakest(str(r["status"]) for r in cw)) if cw else expd(fmt(repeats)))
        rows.append(cells)
    bounds = []
    for label, _shown in WEB_WORKLOADS:
        rj = [r for r in recs if (r.get("extra") or {}).get("model_label") == label and r.get("variant") == "rejections"]
        n_try = sum(int((r.get("extra") or {}).get(f"rejected_{k}_trials", 0) or 0) for r in rj for k in REJECTION_KINDS)
        wrong = sum(int((r.get("extra") or {}).get(f"rejected_{k}_accepted_wrongly", 0) or 0) for r in rj for k in REJECTION_KINDS)
        if n_try and wrong == 0:
            bounds.append(f"{label}: 0 wrong acceptances in {n_try} probes, one-sided 95 percent bound "
                          f"{fmt(zero_failure_upper_bound(n_try))}")
    base = (f"priced with the fitted cost model on the unshaped loopback link for the plan run_sessions.predict_cell "
            f"selects ({run_web.DEFAULT_POLICY}, {parser.get_default('n_generate')} tokens): ")
    rej = f"the planned count: {repeats} repeats x {trials} probes per kind, every probe rejected"
    sources = {cols[1]: base + "setup (material of the prefill at the dealer rate) + installation of the plan's objects "
                               "+ prefill",
               cols[2]: base + "the prefill alone",
               cols[3]: base + "the mean of the decode steps",
               cols[4]: "the planned tokens over the requested tokens",
               cols[5]: "the material of one decode epoch of that plan, per party (candidates.material_per_epoch): what "
                        "is prefetched when the cancel arrives",
               cols[6]: base + "one handshake probe + one prefill of the released prefix (the prompt plus the token "
                               "released before the cut)",
               cols[-1]: f"the planned cold_warm repeats ({repeats})"}
    sources.update({f"{k} rej.": rej for k in REJECTION_KINDS})
    return tabular(["inputs: raw/" + web.describe(used) + "; latest record per (workload, scenario, repeat); per workload "
                    "the records of one thread cap (the cap of the latest capped record; records without the field only "
                    "when no capped record exists), never pooled across caps",
                    "TTFT cold and TTFT warm = mean [min, max] over the cold_warm repeats; TTFT warm = second generation "
                    "on the same stack, reported only when the runner verified a warm start; rejection columns are "
                    "rejected/attempted; n = cold_warm repeats"
                    + ("; " + "; ".join(bounds) if bounds else ""),
                    "per workload, one line each: the thread cap of the records and the first-epoch plan path of every "
                    "warm request (the plan the second generation held at its epoch 0, whether it switched there, and "
                    "the decision reason)" + ("" if paths else "; no record yet")]
                   + paths
                   + [ELIGIBILITY_NOTE + "; cancel and rejections records carry security_eligible false (a cancelled "
                    "generation and refused requests are not completed eligible sessions): their counts and timings are "
                    "measured and printed plain",
                    expect.describe() + "; the predictor prices the secure computation only, not HTTP, the browser "
                    "sidecar or process start"], cols, rows, sources=sources)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def _retire_obsolete(tables_dir: Path) -> List[str]:
    """Move generated .tex files whose name is not in ``TABLE_NAMES`` (a retired table or macros file) out of the way."""
    moved: List[str] = []
    if not tables_dir.is_dir():
        return moved
    for p in sorted(tables_dir.glob("*.tex")):
        if p.stem in TABLE_NAMES:
            continue
        try:
            head = p.read_text(encoding="utf-8", errors="replace")[:200]
        except OSError:
            continue
        if "opencrab2.experiments.make_tables" in head:
            dest = tables_dir / "obsolete"
            dest.mkdir(exist_ok=True)
            try:
                p.replace(dest / p.name)
                moved.append(p.name)
            except OSError:
                pass
    return moved


def make_all(out_root: Path, reference: Optional[str] = None) -> dict:
    """Write the eight paper tables; returns the manifest (also written to ``tables/manifest.json``)."""
    out_root = Path(out_root)
    raw = out_root / "raw"
    tables_dir = out_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    retired = _retire_obsolete(tables_dir)
    files = {"primitives": read_raw(raw / "primitives.jsonl"), "cpp_bench": read_raw(raw / "cpp_bench.jsonl"),
             "sessions": read_raw(raw / "sessions.jsonl"), "simulation": read_raw(raw / "simulation.jsonl", _slim_simulation),
             "factorial": read_raw(raw / "factorial.jsonl", _slim_simulation), "web": read_raw(raw / "web.jsonl")}
    expect = Expectations(out_root)
    builders: Dict[str, Callable[[], str]] = {
        "catalog": catalog_table,
        "blocks": lambda: blocks_table(files["primitives"], files["cpp_bench"], expect),
        "costmodel": lambda: costmodel_table(out_root, expect),
        "dynamic": lambda: dynamic_table(files["sessions"], sim_file=files["simulation"], expect=expect),
        "dynamic_modeled": lambda: dynamic_modeled_table(files["simulation"], expect),
        "factorial": lambda: factorial_table(files["factorial"], expect),
        "e2e": lambda: e2e_table(files["sessions"], expect),
        "web": lambda: web_table(files["web"], expect),
    }
    written: List[str] = []
    errors: Dict[str, str] = {}
    cells: Dict[str, Dict[str, int]] = {}
    for name in TABLE_NAMES:
        try:
            text = builders[name]()
        except Exception as exc:  # a table is always produced: an unexpected input shape yields a one-row table
            errors[name] = f"{exc.__class__.__name__}: {exc}"
            text = tabular([f"inputs could not be aggregated ({exc.__class__.__name__}); see tables/manifest.json",
                            "the single row counts the raw records that were read; no experimental value is printed"],
                           ["table", "records read"], [[Cell(name), Cell(fmt(sum(len(v.records) for v in files.values())))]])
        write_table(tables_dir, name, text)
        written.append(name)
        cells[name] = cell_counts(text)
    manifest = {"tables": written, "errors": errors, "retired": retired, "cells": cells,
                "expectations": {"cost_model": expect.model_source, "model_label": expect.model.label,
                                 "dealer_rate_bytes_per_s": expect.rate, "dealer_rate_source": expect.rate_source},
                "inputs": {k: {"present": v.present, "records": len(v.records), "malformed": v.invalid} for k, v in files.items()},
                "records_by_status": {}, "generated_utc": schema.utc_now(), "code_commit": schema.tree_hash()}
    for v in files.values():
        for r in v.records:
            manifest["records_by_status"][r["status"]] = manifest["records_by_status"].get(r["status"], 0) + 1
    with open(tables_dir / "manifest.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="make_tables", description="the paper tables from raw/*.jsonl under the data root")
    p.add_argument("--out-root", default=None,
                   help="data root (default: OPENCRAB2_RESULTS_DIR, else the tree's results/); tables go to tables/")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    out_root = Path(args.out_root) if args.out_root else schema.experiments_root()
    manifest = make_all(out_root)
    sys.stdout.write(json.dumps(manifest, indent=1) + "\n")
    return 0 if not manifest["errors"] else 1


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main())


__all__ = ["Cell", "RawFile", "TABLE_NAMES", "MARKING", "Expectations", "expd", "wrap", "weakest", "fmt", "fmt_ci",
           "fmt_range", "student_t_ci", "tex_escape", "tabular", "cell_counts", "read_raw", "session_time",
           "output_match_lines", "catalog_table", "blocks_table", "costmodel_table", "dynamic_table",
           "dynamic_modeled_table", "factorial_table", "e2e_table", "web_table", "write_factorial_table", "make_all",
           "main"]
