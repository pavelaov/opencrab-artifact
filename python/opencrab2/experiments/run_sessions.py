"""Measured three-process session campaign driver (paper, Section 7.1, experimental details).

Drives ``opencrab2.platform.launch.launch_two_party`` (dealer + P0 + P1 as
three OS processes, wall-clock exogenous shaping) over a declared matrix and
appends the resulting ``MEASURED_NATIVE`` evidence records to ONE file,
``raw/sessions.jsonl`` under the data root (lock-protected appends, so
several worker processes can write to it).

Matrices (declared before the first session; paper, Section 7.1, experimental details):

* ``smoke``: GPT-2 1-layer prefix, 2 tokens, stable LAN, B4 and B2, seed 1.
  A pipeline check, not evidence for any claim.
* ``diag4``: GPT-2 4-layer prefix (``workload_scope = prefix-4L``), the pinned
  16-token prompt, T = 8. Trace cells: S0 stable LAN, S0 stable WAN
  (100 Mbit/s, 20 ms one way), S1 ramp from 20 Mbit/s to 1 Gbit/s over the
  session, S2 RTT only (0.25 ms to 100 ms one way at 100 Mbit/s), S3 two
  regime changes, S8 cold and S8 warm-installed on the same LAN. Policies on
  every cell: B0 fixed all-wmask plan, B0 fixed all-generic plan, B1, B2, B4;
  on the S1 and S3 cells additionally B4_no_install and delayed:2. Seeds 1,
  2, 3; the policy order is randomized per seed.
* ``full12``: full 12-layer GPT-2 (``workload_scope = full-model``), T = 8;
  cells S0 stable LAN and S1 ramp; policies B0 fixed all-wmask, B2, B4;
  seeds 1, 2, 3.

The S8 warm cell is a real warm request: the launcher runs an unshaped
priming session on the same three processes first (``warm_prime``), so the
measured session starts with every object installed; its ``T_warm_s`` is its
measured wall time and its ``T_cold_s`` is None.

Trace duration. Every measured cell of ``diag4`` and ``full12`` has ONE
declared trace duration (``TRACE_DURATION_S``, seconds, next to the matrix
definition), shared by all policies and seeds of the cell, so that the
policies of one seed run on the same trace points (same ``trace_id`` and
``trace_hash``) and the ramps and changes of the trace fall inside the
session whatever the policy. The constants are about 1.2 times the median
session length of the cell in the first measured campaign, rounded to 10 s,
except S2 and S3, whose change points scale with the duration: they use
440 s, so every delay step and regime change falls inside the session. The
``smoke`` matrix (and any cell without a declared duration) keeps the
prediction rule of :func:`cell_trace`: 1.2 times the predicted session.

Concurrency. ``--workers N`` runs N sessions side by side in a process pool;
each launched process gets ``OMP_NUM_THREADS``, ``OPENBLAS_NUM_THREADS`` and
``MKL_NUM_THREADS`` set from ``--threads-per-process`` (default 2). Every
listening socket of a session binds port 0, so concurrent sessions never
collide; the ports are stored in the record. Every record carries
``extra.concurrent_workers`` and ``extra.threads_per_process``: timings taken
at different concurrency levels must not be pooled, and paired comparisons
are valid only between sessions of the same level (the tables group by it).

Resumability. Every cell has a deterministic ``run_id``; a rerun skips the
run_ids already present in the output file (``--retry-aborted`` reruns the
cells whose only record is an abort; ``--redo`` ignores the file).

Output match. The launcher computes the greedy continuation of the float
reference (erf-GELU, ``experiments.workloads``) and hands it to P1 only; each
P1 record carries ``output_match``, the released tokens
(``extra.generated_tokens``), the reference tokens, the reference margins,
the first differing step (``extra.first_mismatch_step``, None when the
output matches) and the reference margin at that step
(``extra.first_mismatch_margin``), so that a mismatch can be reported from
the record alone. P0's record never holds a token. ``raw_measurement_path``
and the other file names in a record are relative to the data root.

``--dry-run`` prints the cell list with a predicted duration per cell from
the fitted cost model (online time of the predicted plan plus installation,
plus material generation at the rate measured by ``run_primitives`` when
that file exists) and the total at the chosen worker count, so the operator
can budget. The prediction is MODELED; it is never written as a record.

Usage::

    python -m opencrab2.experiments.run_sessions --matrix smoke|diag4|full12 [--workers N]
        [--threads-per-process 2] [--dry-run] [--timeout 7200] [--only LIST] [--model tiny]

What is tested (tests/test_run_sessions.py): matrix construction (cell counts,
per-seed policy order, the extra policies on S1 and S3 only), run_id
determinism, the dry-run prediction, one shared trace per (cell, seed) with
the declared duration, record stamping, resumption, and one tiny-model matrix
with two workers end to end. Design limit: timings are from loopback
processes with application-level shaping and are labelled ``clock_basis =
wall-clock-exogenous``; with ``--workers`` above 1 the sessions share the
host's cores and memory bandwidth.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import itertools
import json
import math
import multiprocessing
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from opencrab2.experiments import schema, traces, workloads
from opencrab2.experiments.workloads import MODEL_REPO, PINNED_PROMPT_IDS, PROMPT_TEXT
from opencrab2.highlevel import candidates as cands
from opencrab2.highlevel import costmodel as cm
from opencrab2.highlevel.contract import DEFAULT_CONTRACT

MODEL = "gpt2"
DEFAULT_TIMEOUT_S = 7200.0
DEFAULT_THREADS_PER_PROCESS = 2
PROCESSES_PER_SESSION = 3
PLAN_WMASK: Dict[str, Any] = {"embedding": "wmask", "linear": "wmask", "argmax_k": 2, "max_k": 2}
PLAN_GENERIC: Dict[str, Any] = {"embedding": "generic", "linear": "generic", "argmax_k": 2, "max_k": 2}
FIXED_PLANS: Dict[str, Dict[str, Any]] = {"B0_fixed:wmask": PLAN_WMASK, "B0_fixed:generic": PLAN_GENERIC}
DIAG_POLICIES: Tuple[str, ...] = ("B0_fixed:wmask", "B0_fixed:generic", "B1_naive", "B2_remaining_plus_install",
                                  "B4_opencrab")
DIAG_EXTRA_POLICIES: Tuple[str, ...] = ("B4_no_install", "delayed:2")
WARM_VARIANTS: Tuple[str, ...] = ("warm-all-LAN", "warm-all")
TRACE_SAFETY = 1.2
PROBE_SAMPLES = 3
PROBE_BURST_BYTES = 65536


@dataclass(frozen=True)
class Cell:
    matrix: str
    family: str
    variant: str
    policy: str
    seed: int
    n_layers: int
    T: int
    model: str = MODEL

    @property
    def run_id(self) -> str:
        return f"session:{self.matrix}:{self.family}:{self.variant}:{self.policy}:T{self.T}:L{self.n_layers}:s{self.seed}"

    @property
    def warm(self) -> bool:
        return self.family == "S8" and self.variant in WARM_VARIANTS

    @property
    def scope(self) -> str:
        if self.model != MODEL:
            return "prefix-1L"      # the tiny pipeline model is never reported as a model-scale result
        return workloads.scope_for_layers(self.n_layers)

    def to_dict(self) -> dict:
        d = {"matrix": self.matrix, "family": self.family, "variant": self.variant, "policy": self.policy,
             "seed": self.seed, "n_layers": self.n_layers, "T": self.T, "model": self.model}
        d["run_id"] = self.run_id
        d["warm"] = self.warm
        d["workload_scope"] = self.scope
        return d


MATRICES: Dict[str, dict] = {
    "smoke": {"traces": [("S0", "stable-LAN")], "policies": ["B4_opencrab", "B2_remaining_plus_install"],
              "extra_policies": {}, "seeds": [1], "n_layers": 1, "T": [2]},
    "diag4": {"traces": [("S0", "stable-LAN"), ("S0", "stable-WAN"), ("S1", "ramp-WAN-c-to-LAN"), ("S2", "rtt-step-up"),
                         ("S3", "two-changes"), ("S8", "cold-LAN"), ("S8", "warm-all-LAN")],
              "policies": list(DIAG_POLICIES),
              "extra_policies": {"S1": list(DIAG_EXTRA_POLICIES), "S3": list(DIAG_EXTRA_POLICIES)},
              "seeds": [1, 2, 3], "n_layers": 4, "T": [8]},
    "full12": {"traces": [("S0", "stable-LAN"), ("S1", "ramp-WAN-c-to-LAN")],
               "policies": ["B0_fixed:wmask", "B2_remaining_plus_install", "B4_opencrab"],
               "extra_policies": {}, "seeds": [1, 2, 3], "n_layers": 12, "T": [8]},
}
# Declared trace duration in seconds per measured cell, (matrix, family, variant): one trace per (cell, seed)
# for every policy. About 1.2 x the median session length of the cell in the first measured campaign, rounded
# to 10 s, except S2 and S3: their change points scale with the duration, and that rule pushed them past the
# end of the session (the pilot S2 sessions were long because they ran at 100 ms), so both use 440 s, which
# puts the S2 delay steps at 110, 220 and 330 s and the S3 regime changes of seeds 1 to 3 between 61 and
# 290 s, inside the shortest session of the cell. The smoke matrix has no entry and keeps the prediction rule.
TRACE_DURATION_S: Dict[Tuple[str, str, str], float] = {
    ("diag4", "S0", "stable-LAN"): 450.0,
    ("diag4", "S0", "stable-WAN"): 650.0,
    ("diag4", "S1", "ramp-WAN-c-to-LAN"): 770.0,
    ("diag4", "S2", "rtt-step-up"): 440.0,
    ("diag4", "S3", "two-changes"): 440.0,
    ("diag4", "S8", "cold-LAN"): 450.0,
    ("diag4", "S8", "warm-all-LAN"): 430.0,
    ("full12", "S0", "stable-LAN"): 410.0,
    ("full12", "S1", "ramp-WAN-c-to-LAN"): 1550.0,
}
TRACE_RULE_FIXED = "declared-per-cell"
TRACE_RULE_PREDICTED = "predicted-session-x1.2"


def build_matrix(name: str, model: str = MODEL) -> List[Cell]:
    """Cells of a matrix; within a (seed, trace cell) the policy order is a seeded permutation."""
    if name not in MATRICES:
        raise ValueError(f"SESSIONS_MATRIX_UNKNOWN: {name!r} is not one of {list(MATRICES)}")
    m = MATRICES[name]
    cells: List[Cell] = []
    for seed in m["seeds"]:
        for ti, (fam, var) in enumerate(m["traces"]):
            pols = list(m["policies"]) + list(m["extra_policies"].get(fam, []))
            rng = np.random.default_rng([int(seed), ti, len(pols)])
            order = [pols[i] for i in rng.permutation(len(pols))]
            for T in m["T"]:
                for pol in order:
                    cells.append(Cell(name, fam, var, pol, int(seed), int(m["n_layers"]), int(T), str(model)))
    return cells


def prompt_ids() -> List[int]:
    """GPT-2 ids of the public prompt from the local tokenizer when available, else the pinned list.

    The tokenizer is that of the pinned snapshot (``transformer.load_gpt2_tokenizer``); a tokenizer that is
    present but encodes the prompt differently is an error (``SESSIONS_PROMPT_MISMATCH``).
    """
    from opencrab2.errors import MaterialError
    try:
        from opencrab2.highlevel.transformer import load_gpt2_tokenizer
        ids = [int(v) for v in load_gpt2_tokenizer(MODEL_REPO).encode(PROMPT_TEXT)]     # the pinned revision
        if ids != list(PINNED_PROMPT_IDS):
            raise ValueError(f"SESSIONS_PROMPT_MISMATCH: tokenizer gave {ids}, pinned {list(PINNED_PROMPT_IDS)}")
        return ids
    except (ImportError, MaterialError):              # no transformers, or the snapshot is not in the local cache
        return list(PINNED_PROMPT_IDS)
    except (OSError, ValueError) as exc:
        if "SESSIONS_PROMPT_MISMATCH" in str(exc):
            raise
        return list(PINNED_PROMPT_IDS)


# ---------------------------------------------------------------------------
# workload of a cell (model spec, prompt, public dimensions)
# ---------------------------------------------------------------------------
def cell_workload(cell: Cell, work_dir: Optional[Path] = None) -> dict:
    """``{model, prompt_ids, length_bucket, dims}`` of a cell; the tiny model is written to ``work_dir`` on demand."""
    if cell.model == MODEL:
        return {"model": MODEL, "prompt_ids": list(PINNED_PROMPT_IDS), "length_bucket": workloads.GPT2_LENGTH_BUCKET,
                "dims": {"d_model": 768, "d_ff": 3072, "n_head": 12, "vocab": 50257}}
    if cell.model == "tiny":
        spec = "npz:<work>/tiny.npz"
        if work_dir is not None:
            from opencrab2.highlevel.session import save_model_npz
            path = Path(work_dir) / f"tiny-L{cell.n_layers}.npz"
            if not path.is_file():
                W, meta = workloads.make_tiny_weights(cell.n_layers)
                tmp = path.with_name(f"{path.stem}.{multiprocessing.current_process().pid}.npz")
                save_model_npz(str(tmp), W, meta)
                try:
                    tmp.replace(path)
                except OSError:
                    pass
            spec = f"npz:{path}"
        # a prompt whose reference margins leave room for the numeric contract (TINY_PROMPT for one layer)
        prompt, _margin = workloads.tiny_prompt(cell.n_layers, cell.T)
        return {"model": spec, "prompt_ids": list(prompt), "length_bucket": workloads.TINY_LENGTH_BUCKET,
                "dims": {"d_model": 64, "d_ff": 256, "n_head": 4, "vocab": 256}}
    raise ValueError(f"SESSIONS_MODEL_UNKNOWN: {cell.model!r} (use 'gpt2' or 'tiny')")


# ---------------------------------------------------------------------------
# prediction (MODELED; for the dry run and for the trace duration)
# ---------------------------------------------------------------------------
def _env_of(family: str, variant: str) -> cm.Env:
    probe = traces.make_trace(family, variant, 0, 1.0)
    st = traces.state_at(probe, 0.0)
    return cm.Env(float(st["bandwidth_bps"]), float(st["one_way_delay_s"]), float(st.get("cpu_share", 1.0) or 1.0))


def _probe_cost(env: cm.Env) -> float:
    bw = env.bandwidth_bps or math.inf
    return PROBE_SAMPLES * (2 * env.rtt_s + PROBE_BURST_BYTES * 8.0 / bw) + env.rtt_s


def material_rate_bytes_per_s(out_root: Optional[Path] = None) -> Optional[float]:
    """Dealer generation rate (bytes of one party's material per second) measured by run_primitives, or None."""
    root = Path(out_root) if out_root is not None else schema.experiments_root()
    rates: List[float] = []
    for r in schema.read_jsonl(root / "raw" / "primitives.jsonl", strict=False):
        e = r.get("extra") or {}
        if e.get("component") == "dcf_gen" and e.get("median_s") and e.get("material_bytes"):
            rates.append(float(e["material_bytes"]) / 2.0 / float(e["median_s"]))
    return float(np.median(rates)) if rates else None


def predict_cell(cell: Cell, model: Any, material_rate: Optional[float] = None) -> dict:
    """Predicted seconds of a cell under the fitted cost model at the trace's initial link state.

    ``online_s`` is the predicted execution of the plan the policy is expected
    to hold (the fixed plan for B0, the cheapest plan per epoch otherwise),
    ``install_s`` the installation of the objects that plan needs (zero on a
    warm cell), ``material_s`` the dealer's generation time at the measured
    rate (None when run_primitives has not run). MODELED.
    """
    wl = cell_workload(cell)
    dims = wl["dims"]
    env = _env_of(cell.family, cell.variant)
    n_prompt = len(wl["prompt_ids"])
    bucket = int(wl["length_bucket"])

    def shapes_for(epoch: int) -> dict:
        n = n_prompt if epoch == 0 else 1
        kv = n_prompt + epoch
        kv = max(bucket, int(math.ceil(kv / bucket)) * bucket)
        return cands.gpt2_shapes(cell.n_layers, n, kv, **dims)

    catalog = cands.gpt2_catalog(shapes_for(0))
    axes = catalog["axes"]
    all_plans = [dict(zip(axes, combo)) for combo in itertools.product(*axes.values())]
    fixed = FIXED_PLANS.get(cell.policy)
    online = sensing = 0.0
    material = 0
    used_objects: set = set()
    cheapest_total: Optional[Tuple[float, dict]] = None
    if fixed is None:
        for p in all_plans:
            tot = sum(cands.exec_cost(p, shapes_for(e), model, env)[0] for e in range(cell.T))
            objs = cands.plan_objects(p, catalog)
            tot += 0.0 if cell.warm else cands.install_cost(objs, catalog, model, env)[0]
            if cheapest_total is None or tot < cheapest_total[0]:
                cheapest_total = (tot, p)
    plan = dict(fixed) if fixed is not None else dict(cheapest_total[1])
    for e in range(cell.T):
        sh = shapes_for(e)
        online += cands.exec_cost(plan, sh, model, env)[0]
        sensing += _probe_cost(env)
        material += int(sum(cands.material_per_epoch(plan, sh, DEFAULT_CONTRACT).values()))
        used_objects.update(cands.plan_objects(plan, catalog))
    install = 0.0 if cell.warm else cands.install_cost(sorted(used_objects), catalog, model, env)[0]
    material_s = (material / material_rate) if material_rate else None
    total = online + sensing + install + (material_s or 0.0)
    priming = 0.0
    if cell.warm:
        sh0 = shapes_for(0)
        priming = (cands.exec_cost(PLAN_WMASK, sh0, model, cm.Env(0.0, 0.0, 1.0))[0]
                   + cands.install_cost(cands.plan_objects(PLAN_WMASK, catalog), catalog, model, cm.Env(0.0, 0.0, 1.0))[0])
    return {"run_id": cell.run_id, "predicted_plan": plan, "online_s": float(online), "sensing_s": float(sensing),
            "install_s": float(install), "material_bytes_per_party": int(material),
            "material_s": float(material_s) if material_s is not None else None,
            "priming_s": float(priming), "predicted_total_s": float(total + priming), "status": "MODELED"}


def trace_duration_s(cell: Cell, costmodel: Any = None, material_rate: Optional[float] = None) -> Tuple[float, str]:
    """(duration in seconds, rule) of a cell's trace.

    A cell listed in ``TRACE_DURATION_S`` uses its declared duration (the same
    for every policy and seed; ``TRACE_RULE_FIXED``). Any other cell (the
    smoke matrix) uses 1.2 x the predicted session without the priming, at
    least 10 s (``TRACE_RULE_PREDICTED``), which needs the cost model.
    """
    fixed = TRACE_DURATION_S.get((cell.matrix, cell.family, cell.variant))
    if fixed is not None:
        return float(fixed), TRACE_RULE_FIXED
    if costmodel is None:
        costmodel = cm.default_model()
    pred = predict_cell(cell, costmodel, material_rate)
    return max(10.0, TRACE_SAFETY * (pred["predicted_total_s"] - pred["priming_s"])), TRACE_RULE_PREDICTED


def cell_trace(cell: Cell, costmodel: Any, traces_dir: Path, material_rate: Optional[float] = None) -> dict:
    """The exogenous trace of a cell (:func:`trace_duration_s`), written to ``traces_dir`` as JSON."""
    duration, _rule = trace_duration_s(cell, costmodel, material_rate)
    tr = traces.make_trace(cell.family, cell.variant, cell.seed, duration)
    traces.write_trace(tr, traces_dir)
    return tr


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------
def _records_from_launch(result: Any) -> List[dict]:
    """The per-party record dicts of a launch result (``record0`` / ``record1``)."""
    recs: List[dict] = []
    if isinstance(result, Mapping):
        for k in ("record0", "record1"):
            v = result.get(k)
            if isinstance(v, Mapping) and "status" in v:
                recs.append(dict(v))
    return recs


def _relative_paths(obj: Any, root: Optional[Path]) -> Any:
    """Every string value of a (nested) mapping of file paths, relative to the data root."""
    if isinstance(obj, Mapping):
        return {k: _relative_paths(v, root) for k, v in obj.items()}
    if isinstance(obj, str) and obj:
        return schema.relative_raw_path(obj, root)
    return obj


def stamp(rec: dict, cell: Cell, tr: dict, raw_path: str, run_meta: Optional[Mapping[str, Any]] = None,
          root: Optional[Path] = None) -> dict:
    """Fill the matrix metadata into a launcher record without overriding what the launcher measured.

    Top-level fields outside the schema vocabulary (the launcher's own, such
    as ``paths``) are moved under ``extra["launcher"]``; the run_id gets a
    per-party suffix; ``run_meta`` (concurrency, threads, ports, reference
    margins) is merged into ``extra``. The record's ``raw_measurement_path``
    and the launcher's paths are stored relative to ``root`` (the data root;
    default :func:`schema.experiments_root`), never as host paths. The P1
    record keeps the tokens the runner wrote; when it names a first
    mismatching step, ``extra.first_mismatch_margin`` is the reference margin
    of that step (from ``run_meta["reference_margins"]``).
    """
    rec = dict(rec)
    extra = dict(rec.get("extra") or {})
    foreign = [k for k in rec if k not in schema.RECORD_FIELDS and k not in schema.OPTIONAL_FIELDS]
    if foreign:
        extra["launcher"] = {k: rec.pop(k) for k in foreign}
    if isinstance(extra.get("launcher"), Mapping) and isinstance(extra["launcher"].get("paths"), Mapping):
        extra["launcher"] = dict(extra["launcher"], paths=_relative_paths(extra["launcher"]["paths"], root))
    if rec.get("raw_measurement_path"):
        rec["raw_measurement_path"] = schema.relative_raw_path(str(rec["raw_measurement_path"]), root)
    party = extra.get("party")
    rec["schema"] = schema.SCHEMA_ID
    rec["run_id"] = cell.run_id + (f":p{int(party)}" if party in (0, 1) else "")
    rec["record_kind"] = "session"
    rec.setdefault("status", "MEASURED_NATIVE")
    rec.setdefault("security_eligible", False)
    extra["runner_policy_id"] = rec.get("policy_id")
    rec["policy_id"] = cell.policy
    rec["trace_id"] = tr["trace_id"]
    rec["trace_hash"] = tr["trace_hash"]
    rec["repeat"] = int(cell.seed)
    rec["seed"] = int(cell.seed)
    rec["family"] = cell.family
    rec["variant"] = cell.variant
    rec["split"] = tr["split"]
    rec["T"] = int(cell.T)
    rec["n_layers"] = int(cell.n_layers)
    rec["workload_scope"] = cell.scope
    rec.setdefault("clock_basis", "wall-clock-exogenous")
    fixed = (cell.matrix, cell.family, cell.variant) in TRACE_DURATION_S
    extra.update({"matrix": cell.matrix, "trace_duration_s": tr["duration_s"],
                  "trace_duration_rule": TRACE_RULE_FIXED if fixed else TRACE_RULE_PREDICTED, "model": cell.model,
                  "cell_warm": bool(cell.warm), "detail_path": rec.get("raw_measurement_path", "")})
    extra.update(dict(run_meta or {}))
    step = extra.get("first_mismatch_step")
    margins = extra.get("reference_margins")
    if isinstance(step, int) and not isinstance(step, bool) and isinstance(margins, list) and 0 <= step < len(margins):
        extra["first_mismatch_margin"] = float(margins[step])
    elif "first_mismatch_step" in extra:
        extra["first_mismatch_margin"] = None
    rec["extra"] = extra
    defaults = schema.new_record()
    for k in schema.RECORD_FIELDS:
        rec.setdefault(k, defaults[k])
    return rec


def aborted_record(cell: Cell, tr: Optional[dict], reason: str, raw_path: str,
                   run_meta: Optional[Mapping[str, Any]] = None, root: Optional[Path] = None) -> dict:
    extra = {"matrix": cell.matrix, "abort_reason": reason, "model": cell.model, "cell_warm": bool(cell.warm)}
    extra.update(dict(run_meta or {}))
    return schema.new_record(
        status="MEASURED_NATIVE", record_kind="session", security_eligible=False, dealer_mode="ISOLATED_RECIPIENT_BUNDLES",
        workload_scope=cell.scope, model_revision=MODEL_REPO if cell.model == MODEL else cell.model,
        trace_id=tr["trace_id"] if tr else "none", trace_hash=tr["trace_hash"] if tr else "",
        clock_basis="wall-clock-exogenous", policy_id=cell.policy, run_id=cell.run_id, repeat=int(cell.seed), aborts=1,
        raw_measurement_path=schema.relative_raw_path(raw_path, root), seed=int(cell.seed), family=cell.family,
        variant=cell.variant, split=tr["split"] if tr else "none", T=int(cell.T), n_layers=int(cell.n_layers), extra=extra)


def completed_cells(raw_path: Path, retry_aborted: bool = False) -> set:
    """Cells whose result is already in the output file.

    A cell is done when BOTH party records (``<run_id>:p0`` and ``<run_id>:p1``) are present, or when the
    runner wrote its abort record (the bare ``run_id``; skipped with ``retry_aborted``). A cell with only one
    party record is not done: the other process died before its append, and counting the cell as done would
    lose that record for good (the client record is the one ``make_tables`` reads). Such a cell is rerun; the
    tables keep the latest record per run_id. A torn last line of the file is skipped (``read_jsonl``).
    """
    parties: Dict[str, set] = {}
    done: set = set()
    for r in schema.read_jsonl(raw_path, strict=False):
        rid = r.get("run_id") if isinstance(r, dict) else None
        if not isinstance(rid, str):
            continue
        if retry_aborted and int(r.get("aborts", 0) or 0) > 0 and (r.get("extra") or {}).get("abort_reason"):
            continue
        if rid.endswith((":p0", ":p1")):
            parties.setdefault(rid[:-3], set()).add(rid[-2:])
        else:
            done.add(rid)                 # the abort record of the runner (or a record of an older layout)
    for base, seen in parties.items():
        if seen >= {"p0", "p1"}:
            done.add(base)
    return done


def observed_ratio(raw_path: Path, model: Any, material_rate: Optional[float], workers: int) -> Optional[dict]:
    """Median of measured session wall time over the prediction, from completed client records at this worker count.

    The fitted model prices the online gates and the installation; a measured
    session also waits for material, senses, plans and commits, and shares
    the host with the other workers. The ratio lets the dry run scale its
    prediction by what this host actually delivered. None without records.
    """
    ratios: List[float] = []
    for r in schema.read_jsonl(raw_path, strict=False):
        if not isinstance(r, dict):
            continue
        e = r.get("extra") or {}
        if e.get("party") != 1 or int(r.get("aborts", 0) or 0) or e.get("concurrent_workers") != int(workers):
            continue
        wall = e.get("session_wall_s")
        try:
            cell = Cell(str(e["matrix"]), str(r["family"]), str(r["variant"]), str(r["policy_id"]), int(r["seed"]),
                        int(r["n_layers"]), int(r["T"]), str(e.get("model", MODEL)))
            pr = predict_cell(cell, model, material_rate)
            pred = pr["predicted_total_s"] - pr["priming_s"]
        except (KeyError, TypeError, ValueError):
            continue
        if isinstance(wall, (int, float)) and wall > 0 and pred > 0:
            ratios.append(float(wall) / float(pred))
    if not ratios:
        return None
    return {"median": float(np.median(ratios)), "min": float(min(ratios)), "max": float(max(ratios)), "n": len(ratios)}


# ---------------------------------------------------------------------------
# one cell (runs in a worker process)
# ---------------------------------------------------------------------------
def run_cell(cell_d: Mapping[str, Any], out_root: str, timeout_s: float, workers: int, threads: int) -> dict:
    """Launch one cell, append its records, return a one-line summary. Never raises: a failure is an aborted record."""
    cell = Cell(**{k: cell_d[k] for k in ("matrix", "family", "variant", "policy", "seed", "n_layers", "T", "model")})
    root = Path(out_root)
    raw_path = root / "raw" / "sessions.jsonl"
    sess_dir = root / "raw" / "sessions"
    cell_dir = sess_dir / hashlib.sha256(cell.run_id.encode()).hexdigest()[:16]
    cell_dir.mkdir(parents=True, exist_ok=True)
    run_meta: Dict[str, Any] = {"concurrent_workers": int(workers), "threads_per_process": int(threads),
                                "processes_per_session": PROCESSES_PER_SESSION}
    t0 = time.perf_counter()
    tr: Optional[dict] = None
    try:
        from opencrab2.highlevel.session import SessionConfig
        from opencrab2.platform.launch import launch_two_party
        model = cm.default_model()
        tr = cell_trace(cell, model, root / "traces", material_rate_bytes_per_s(root))
        wl = cell_workload(cell, sess_dir)
        fixed = FIXED_PLANS.get(cell.policy)
        cfg = SessionConfig(model=wl["model"], n_layers=cell.n_layers, prompt_ids=list(wl["prompt_ids"]),
                            n_generate=cell.T, policy="B0_fixed" if fixed is not None else cell.policy,
                            fixed_plan=dict(fixed) if fixed is not None else None, trace_id=tr["trace_id"],
                            seed=cell.seed, contract=DEFAULT_CONTRACT, length_bucket=int(wl["length_bucket"]),
                            session_id="s-" + hashlib.sha256(cell.run_id.encode()).hexdigest()[:12],
                            run_id=cell.run_id, repeat=int(cell.seed), workload_scope=cell.scope,
                            setup_id="setup-" + hashlib.sha256(cell.run_id.encode()).hexdigest()[:8])
        result = launch_two_party(cfg, tr, str(cell_dir), timeout_s=float(timeout_s), threads_per_process=int(threads),
                                  warm_prime=cell.warm)
        recs = _records_from_launch(result)
        if len(recs) != 2:
            raise RuntimeError("SESSIONS_NO_RECORD: the launcher did not return one record per party")
        run_meta.update({"ports": dict(result.get("ports") or {}), "launch_wall_s": time.perf_counter() - t0,
                         "reference_margins": list(result.get("reference_margins") or []),
                         "warm_prime": bool(result.get("warm_prime")),
                         "priming": (result.get("summaries", {}).get(1, {}) or {}).get("priming")})
        for r in recs:
            schema.append_jsonl(raw_path, stamp(r, cell, tr, str(raw_path), run_meta, root))
        r1 = next((r for r in recs if (r.get("extra") or {}).get("party") == 1), recs[-1])
        wall = (r1.get("extra") or {}).get("session_wall_s")
        return {"run_id": cell.run_id, "ok": True, "seconds": time.perf_counter() - t0, "session_wall_s": wall,
                "output_match": r1.get("output_match"), "aborts": int(r1.get("aborts", 0))}
    except Exception as exc:  # the cell is reported by reason, never filled
        reason = f"{exc.__class__.__name__}: {exc}"
        try:
            with open(cell_dir / "abort.txt", "w", encoding="utf-8", newline="\n") as fh:
                fh.write(reason + "\n" + traceback.format_exc())
            run_meta["launch_wall_s"] = time.perf_counter() - t0
            schema.append_jsonl(raw_path, aborted_record(cell, tr, reason, str(raw_path), run_meta, root))
        except Exception as exc2:   # pragma: no cover (disk failure)
            reason += f" | record write failed: {exc2.__class__.__name__}"
        return {"run_id": cell.run_id, "ok": False, "seconds": time.perf_counter() - t0, "reason": reason}


# ---------------------------------------------------------------------------
# matrix driver
# ---------------------------------------------------------------------------
def _schedule_seconds(durations: Sequence[float], workers: int) -> float:
    """Wall time of running ``durations`` in order on ``workers`` slots (greedy list scheduling)."""
    slots = [0.0] * max(1, int(workers))
    for d in durations:
        i = int(np.argmin(slots))
        slots[i] += float(d)
    return float(max(slots)) if slots else 0.0


def run_matrix(name: str, out_root: Path, timeout_s: float, dry_run: bool = False, only: Optional[Sequence[str]] = None,
               redo: bool = False, workers: int = 1, threads_per_process: int = DEFAULT_THREADS_PER_PROCESS,
               model: str = MODEL, retry_aborted: bool = False, log: Any = None,
               skip: Optional[Sequence[str]] = None) -> dict:
    if int(workers) < 1 or int(threads_per_process) < 1:
        raise ValueError("SESSIONS_PARAMS: workers and threads_per_process must be >= 1")
    cells = build_matrix(name, model)
    if only:
        keep = set(only)
        cells = [c for c in cells if c.policy in keep or c.variant in keep or c.family in keep
                 or f"{c.family}:{c.variant}" in keep or f"s{c.seed}" in keep]
    if skip:
        # A skipped (cell, policy) pair is a declared reduction of the matrix (budget), never a
        # silent drop: the caller fixes the skip list before the run and the tables mark the
        # affected cells. Tokens: a policy, "family/policy", "family:variant/policy" or "s<seed>".
        drop = set(skip)
        cells = [c for c in cells if not (c.policy in drop or f"{c.family}/{c.policy}" in drop
                                          or f"{c.family}:{c.variant}/{c.policy}" in drop or f"s{c.seed}" in drop)]
    raw_path = out_root / "raw" / "sessions.jsonl"
    done = set() if redo else completed_cells(raw_path, retry_aborted)
    todo = [c for c in cells if c.run_id not in done]
    summary: Dict[str, Any] = {"matrix": name, "model": model, "cells": len(cells), "already_done": len(cells) - len(todo),
                               "run": 0, "aborted": 0, "dry_run": dry_run, "workers": int(workers),
                               "threads_per_process": int(threads_per_process), "output": str(raw_path)}
    out = log if log is not None else sys.stdout
    if dry_run:
        cmodel = cm.default_model()
        rate = material_rate_bytes_per_s(out_root)
        rows = []
        for c in todo:
            p = predict_cell(c, cmodel, rate)
            rows.append(dict(c.to_dict(), **{k: p[k] for k in ("predicted_total_s", "online_s", "install_s",
                                                                 "material_s", "priming_s", "predicted_plan")}))
        durs = [r["predicted_total_s"] for r in rows]
        obs = observed_ratio(raw_path, cmodel, rate, int(workers))
        summary["todo"] = rows
        summary["predicted"] = {
            "status": "MODELED", "cost_model": getattr(cmodel, "label", "fitted"),
            "material_rate_bytes_per_s": rate,
            "sum_cell_seconds": float(sum(durs)),
            "wall_seconds_at_workers": _schedule_seconds(durs, workers),
            "observed_over_predicted": obs,
            "wall_seconds_scaled_by_observed": (_schedule_seconds(durs, workers) * obs["median"]) if obs else None,
            "note": ("online + sensing + install from the fitted cost model at the trace's initial link state; "
                     "material generation at the dcf_gen rate of run_primitives when available; process start, "
                     "model loading and the reference computation are not included"),
        }
        return summary
    if not todo:
        return summary
    tasks = [(c.to_dict(), str(out_root), float(timeout_s), int(workers), int(threads_per_process)) for c in todo]

    def note(res: dict) -> None:
        summary["run" if res.get("ok") else "aborted"] += 1
        out.write(f"{res['run_id']}: {'ok' if res.get('ok') else 'ABORTED'} {res['seconds']:.1f} s"
                  + (f" match={res.get('output_match')}" if res.get("ok") else f" ({res.get('reason', '')[:200]})") + "\n")
        out.flush()

    if int(workers) == 1:
        for t in tasks:
            note(run_cell(*t))
    else:
        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=int(workers), mp_context=ctx) as pool:
            futs = [pool.submit(run_cell, *t) for t in tasks]
            for fut, t in zip(futs, tasks):
                try:
                    note(fut.result())
                except Exception as exc:  # a worker process died: the cell is reported, the pool goes on
                    rid = t[0]["run_id"]
                    note({"run_id": rid, "ok": False, "seconds": 0.0,
                          "reason": f"SESSIONS_WORKER_DIED: {exc.__class__.__name__}"})
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run_sessions", description="measured three-process session campaign")
    p.add_argument("--matrix", default="smoke", choices=sorted(MATRICES))
    p.add_argument("--workers", type=int, default=1, help="sessions run side by side (process pool)")
    p.add_argument("--threads-per-process", type=int, default=DEFAULT_THREADS_PER_PROCESS,
                   help="OMP/OPENBLAS/MKL thread cap of every launched process")
    p.add_argument("--dry-run", action="store_true", help="print the pending cells and predicted durations")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="per-session timeout in seconds")
    p.add_argument("--only", default=None, help="comma separated policies, variants, families, family:variant or s<seed>")
    p.add_argument("--skip", default=None,
                   help="comma separated exclusions: policy, family/policy, family:variant/policy or s<seed> "
                        "(a declared matrix reduction, fixed before the run)")
    p.add_argument("--redo", action="store_true", help="ignore completed run_ids")
    p.add_argument("--retry-aborted", action="store_true", help="rerun cells whose only record is an abort")
    p.add_argument("--model", default=MODEL, choices=[MODEL, "tiny"],
                   help="'tiny' swaps in the random byte-vocabulary model (pipeline test only)")
    p.add_argument("--out-root", default=None)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    out_root = Path(args.out_root) if args.out_root else schema.experiments_root()
    try:
        summary = run_matrix(args.matrix, out_root, args.timeout, args.dry_run, args.only.split(",") if args.only else None,
                             args.redo, args.workers, args.threads_per_process, args.model, args.retry_aborted,
                             skip=args.skip.split(",") if args.skip else None)
    except (ImportError, ValueError) as exc:
        sys.stderr.write(str(exc) + "\n")
        return 2
    if args.dry_run:
        for r in summary.get("todo", []):
            ms = r["material_s"]
            sys.stdout.write(f"{r['run_id']:<78s} predicted {r['predicted_total_s']:9.1f} s  (online {r['online_s']:.1f}, "
                             f"install {r['install_s']:.1f}, material {'n/a' if ms is None else format(ms, '.1f')}, "
                             f"priming {r['priming_s']:.1f})\n")
        pr = summary["predicted"]
        sys.stdout.write(f"cells pending {len(summary.get('todo', []))} of {summary['cells']}; predicted sum "
                         f"{pr['sum_cell_seconds'] / 3600.0:.2f} h; wall at {summary['workers']} worker(s) "
                         f"{pr['wall_seconds_at_workers'] / 3600.0:.2f} h (MODELED; {pr['note']})\n")
        obs = pr.get("observed_over_predicted")
        if obs:
            sys.stdout.write(f"completed sessions at this worker count ran {obs['median']:.2f}x the prediction (median of "
                             f"{obs['n']}, range {obs['min']:.2f} to {obs['max']:.2f}); scaled wall budget "
                             f"{pr['wall_seconds_scaled_by_observed'] / 3600.0:.2f} h\n")
        else:
            sys.stdout.write("no completed session at this worker count yet: run the smoke matrix with the same --workers "
                             "to learn the measured/predicted ratio of this host\n")
        return 0
    sys.stdout.write(json.dumps(summary, indent=1) + "\n")
    return 0 if summary["aborted"] == 0 else 1


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main())


__all__ = ["Cell", "MATRICES", "FIXED_PLANS", "TRACE_DURATION_S", "TRACE_RULE_FIXED", "TRACE_RULE_PREDICTED",
           "PROMPT_TEXT", "PINNED_PROMPT_IDS", "build_matrix", "prompt_ids", "cell_workload", "predict_cell",
           "trace_duration_s", "cell_trace", "stamp", "aborted_record", "completed_cells", "run_cell", "run_matrix",
           "material_rate_bytes_per_s", "main"]
