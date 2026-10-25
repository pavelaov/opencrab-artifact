"""Single entry point of the evaluation campaign: stages, logs, resumption.

Stages, in their dependency order::

    primitives    low-level gate microbenchmarks (Python)                -> raw/primitives.jsonl
    cpp_bench     C++ building blocks against the isolated C++ dealer    -> raw/cpp_bench.jsonl
    calibration   four-network grid with the pre-declared hold-out;      -> raw/calibration.jsonl,
                  fits and evaluates the cost model                         calibration/fitted_model.json, calibration_metrics.json
    simulation    MODELED campaign over the nine trace families          -> raw/simulation.jsonl, raw/factorial.jsonl
    diag4         measured GPT-2 4-layer matrix (105 sessions)           -> raw/sessions.jsonl
    full12        measured full GPT-2 matrix (15 sessions)               -> raw/sessions.jsonl
    web           Web service: TTFT, warm TTFT, cancel, resume           -> raw/web.jsonl
    tables        the eight paper tables                                 -> tables/*.tex

Usage (from the artifact root)::

    python scripts/run_campaign.py --quick                          # every stage with the tiny model (minutes) -> runs/quick
    python scripts/run_campaign.py                                  # the paper's campaign (several hours)      -> runs/full
    python scripts/run_campaign.py --stage diag4,full12 --dry-run   # cells and predicted hours
    python scripts/run_campaign.py --stage tables --out-root results   # regenerate results/tables from the shipped records

Without ``--stage`` the paper's campaign runs every stage except the
calibration (``CAMPAIGN_STAGES``): the shipped ``results/calibration`` prices
the simulation, the session controllers and the expected table cells, and the
calibration is not repeated. ``--stage all`` adds a fresh calibration (about
4.6 hours here), whose fitted model the later stages of that root then use.
With ``--quick`` the default is every stage, the calibration in its quick form
included.

The defaults are the flags of the declared final campaign: diag4 and full12
with 3 sessions side by side (the first campaign ran diag4 with 6), 2 BLAS
threads per process, a 7200 s session timeout, the declared reduction
``--skip`` of the generic fixed plan on the four non-anchor traces of diag4
and of the remaining-plus-install policy on the ramp trace of full12; the Web
stage on the tiny model and the GPT-2 prefixes with 3 repeats, 10 rejection
probes per kind and 2 threads per process; the simulation with 16 workers.
The reductions were declared before the first session (paper, Section 7.1,
experimental details), and the tables mark the cells they affect.

Output root. Every stage writes under one data root, ``runs/<name>`` below the
artifact root (``--name``; ``quick`` with ``--quick``, else ``full``), or the
directory given with ``--out-root``. The shipped ``results/`` is refused as a
target unless ``--out-root results`` (that directory itself) is given
explicitly. The root is exported to every child as ``OPENCRAB2_RESULTS_DIR``,
so the drivers, the cost model (``calibration/fitted_model.json`` of the root
once the calibration stage wrote it, else the shipped one) and every spawned
party or service process agree on it; ``formulas.json`` is always the shipped
file. Children also get ``NO_PROXY=127.0.0.1,localhost`` (loopback HTTP breaks
behind a proxy), ``HF_HUB_OFFLINE=1`` and the ``python/`` directory on
``PYTHONPATH``.

Every stage runs its runner in a child interpreter (the one running this
script), with stdout and stderr copied to ``<root>/logs/<stage>-<utc>.log``
and to the console. ``<root>/logs/campaign_state.json`` records per stage the
command, the log, the start and end times and the exit status.

Resumption. A stage whose last run finished with status 0 under the same
arguments (the interpreter and the root path are not part of the comparison)
is skipped (``--force`` reruns it). An interrupted or failed stage is run
again and continues where it stopped, because every runner is resumable on
its own: run_sessions and run_web skip the run_ids already in their output
file, run_simulation skips the records it already wrote, run_calibration
``--resume`` keeps the networks it finished. ``--retry-aborted`` makes the
session stages rerun cells whose only record is an abort.

Stale inputs are moved aside, never deleted and never mixed: a fresh
calibration moves the previous ``raw/calibration.jsonl`` to ``raw/stale/``;
the simulation stage moves ``raw/simulation.jsonl`` and ``raw/factorial.jsonl``
there when they were produced under another fitted model or another
``formulas.json`` than the current ones.

``--quick`` is the pipeline check: primitives, C++ benchmark, calibration and
simulation in their quick forms, the smoke session matrix on the tiny model in
the diag4 stage (full12 has no quick form and is skipped), the Web stage on
the tiny model with one repeat, then the tables; a missing C++ build skips
the benchmark stage in quick mode and fails it otherwise.

Exit status: 0 when every requested stage succeeded, 1 when a stage failed
(later stages are not started unless ``--keep-going``), 2 on a usage error.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
TREE = HERE.parent                                   # the artifact root (python/, cpp/, results/, scripts/)
PYDIR = TREE / "python"
RESULTS = TREE / "results"                           # the shipped outputs
SHIPPED_CALIBRATION = RESULTS / "calibration"
CPP_BIN = TREE / "cpp" / "build" / "bin"
EXE = ".exe" if os.name == "nt" else ""
BUILD_HINT = "build with: bash scripts/ubuntu.sh build (CMake) on Ubuntu, python cpp/build.py on Windows"

STAGES: Tuple[str, ...] = ("primitives", "cpp_bench", "calibration", "simulation", "diag4", "full12", "web", "tables")
# the paper's campaign: the calibration is not repeated, the shipped fitted model prices every later stage
CAMPAIGN_STAGES: Tuple[str, ...] = tuple(s for s in STAGES if s != "calibration")
STATE_FILE = "campaign_state.json"

# the campaign of the paper (declared final campaign)
DIAG4_WORKERS = 3
FULL12_WORKERS = 3
SIMULATION_WORKERS = 16
THREADS_PER_PROCESS = 2
SESSION_TIMEOUT_S = 7200.0
DIAG4_SKIP = "S0:stable-WAN/B0_fixed:generic,S1/B0_fixed:generic,S2/B0_fixed:generic,S3/B0_fixed:generic"
FULL12_SKIP = "S1/B2_remaining_plus_install"
WEB_MODEL = "gpt2"                                   # the tiny model and the GPT-2 1-layer and 4-layer prefixes
WEB_REPEATS = 3
WEB_TRIALS = 10

# the quick pipeline check
QUICK_SESSION_WORKERS = 2
QUICK_WEB_REPEATS = 1
QUICK_WEB_TRIALS = 2
QUICK_CALIBRATION_NETWORKS = "LAN"
QUICK_CALIBRATION_REPEATS = 1


def utc_stamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_root(out_root: Optional[str], name: str) -> Path:
    """The data root: ``--out-root`` or ``runs/<name>``; the shipped results/ only when named itself."""
    if out_root:
        root = Path(out_root).expanduser().resolve()
        results = RESULTS.resolve()
        if root != results and results in root.parents:
            raise ValueError(f"CAMPAIGN_OUT_ROOT: {root} lies inside the shipped results/; a rerun writes to runs/<name> "
                             f"(or to results/ itself with --out-root results)")
        return root
    if not name or name in (".", "..") or any(sep in name for sep in ("/", "\\")):
        raise ValueError(f"CAMPAIGN_NAME: --name must be a plain directory name, got {name!r}")
    return (TREE / "runs" / name).resolve()


def sha256_of(paths: List[Path]) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.name.encode("utf-8"))
        try:
            h.update(p.read_bytes())
        except OSError:
            h.update(b"<missing>")
    return h.hexdigest()


def fitted_model_file(root: Path) -> Path:
    """The calibration the drivers price with under this root (see costmodel.calibration_path)."""
    own = root / "calibration" / "fitted_model.json"
    return own if own.is_file() else SHIPPED_CALIBRATION / "fitted_model.json"


def load_state(logs: Path) -> dict:
    p = logs / STATE_FILE
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    except (OSError, ValueError):
        return {}


def save_state(logs: Path, state: dict) -> None:
    logs.mkdir(parents=True, exist_ok=True)
    tmp = logs / (STATE_FILE + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(state, fh, indent=1, sort_keys=True)
        fh.write("\n")
    os.replace(str(tmp), str(logs / STATE_FILE))


def move_aside(root: Path, names: List[str], reason: str, out) -> List[str]:
    """Move stale raw files to ``raw/stale/<utc>/`` (never deleted)."""
    moved: List[str] = []
    present = [root / "raw" / n for n in names if (root / "raw" / n).is_file()]
    if not present:
        return moved
    dest = root / "raw" / "stale" / utc_stamp()
    dest.mkdir(parents=True, exist_ok=True)
    for p in present:
        os.replace(str(p), str(dest / p.name))
        moved.append(str(dest / p.name))
    with open(dest / "REASON.txt", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(reason + "\n")
    out.write(f"[campaign] moved {len(moved)} stale file(s) to {dest} ({reason})\n")
    return moved


def cpp_binaries_present() -> bool:
    return all((CPP_BIN / (n + EXE)).is_file() for n in ("opencrab2_dealer", "opencrab2_bench"))


def stage_command(stage: str, args: argparse.Namespace, root: Path) -> Optional[List[str]]:
    """The child command line of a stage (module form, so the runner's own CLI is what runs); None = skipped."""
    py = [sys.executable, "-m"]
    out = ["--out-root", str(root)]
    quick = bool(args.quick)
    if stage == "primitives":
        return py + ["opencrab2.experiments.run_primitives", "--out", str(root / "raw")] + (["--quick"] if quick else [])
    if stage == "cpp_bench":
        return [sys.executable, str(HERE / "run_cpp_bench.py"), "--bin", str(CPP_BIN),
                "--out", str(root / "raw" / "cpp_bench.jsonl")] + (["--quick"] if quick else [])
    if stage == "calibration":
        cmd = py + ["opencrab2.experiments.run_calibration", "--resume"] + out
        if quick:
            cmd += ["--quick", "--networks", QUICK_CALIBRATION_NETWORKS, "--repeats", str(QUICK_CALIBRATION_REPEATS)]
        return cmd
    if stage == "simulation":
        return (py + ["opencrab2.experiments.run_simulation", "--workers", str(max(1, args.simulation_workers))] + out
                + (["--quick"] if quick else []))
    if stage in ("diag4", "full12"):
        if quick:
            if stage == "full12":
                return None
            cmd = py + ["opencrab2.experiments.run_sessions", "--matrix", "smoke", "--model", "tiny",
                        "--workers", str(QUICK_SESSION_WORKERS), "--threads-per-process", str(args.threads_per_process),
                        "--timeout", str(args.timeout)] + out
        else:
            workers = args.workers if stage == "diag4" else args.full12_workers
            skip = args.diag4_skip if stage == "diag4" else args.full12_skip
            cmd = py + ["opencrab2.experiments.run_sessions", "--matrix", stage, "--workers", str(max(1, workers)),
                        "--threads-per-process", str(args.threads_per_process), "--timeout", str(args.timeout)] + out
            if skip:
                cmd += ["--skip", skip]
        if args.retry_aborted:
            cmd.append("--retry-aborted")
        if args.dry_run:
            cmd.append("--dry-run")
        return cmd
    if stage == "web":
        model = "tiny" if quick else args.web_model
        repeats = QUICK_WEB_REPEATS if quick else args.web_repeats
        trials = QUICK_WEB_TRIALS if quick else args.web_trials
        return (py + ["opencrab2.experiments.run_web", "--model", model, "--repeats", str(repeats), "--trials", str(trials),
                      "--threads-per-process", str(args.threads_per_process)] + out + (["--dry-run"] if args.dry_run else []))
    if stage == "tables":
        return py + ["opencrab2.experiments.make_tables"] + out
    raise ValueError(f"CAMPAIGN_STAGE_UNKNOWN: {stage!r}")


def signature(cmd: List[str], root: Path) -> List[str]:
    """The arguments of a stage without the interpreter and with the paths of this tree abstracted."""
    sig: List[str] = []
    for a in cmd[1:]:
        a = a.replace(str(root), "<root>").replace(str(TREE), "<tree>")
        sig.append(a)
    return sig


def child_env(root: Path) -> Dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PYDIR) + os.pathsep + env.get("PYTHONPATH", "")
    env["OPENCRAB2_RESULTS_DIR"] = str(root)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    # loopback HTTP of the Web stage must not go through a proxy: both spellings list 127.0.0.1 and localhost
    # next to the entries already present (Windows has one case-insensitive variable)
    hosts: List[str] = []
    for var in ("NO_PROXY", "no_proxy"):
        for v in env.get(var, "").split(","):
            if v.strip() and v.strip() not in hosts:
                hosts.append(v.strip())
    hosts += [h for h in ("127.0.0.1", "localhost") if h not in hosts]
    for k in [k for k in env if k.upper() == "NO_PROXY"]:
        del env[k]
    for var in (("NO_PROXY",) if os.name == "nt" else ("NO_PROXY", "no_proxy")):
        env[var] = ",".join(hosts)
    return env


def prepare_stage(stage: str, root: Path, state: dict, out) -> dict:
    """Stage-specific bookkeeping before the run; returns fields to store in the stage state."""
    info: Dict[str, object] = {}
    if stage == "calibration":
        resume = root / "calibration" / "work" / "resume.json"
        unfinished = False
        try:
            unfinished = resume.is_file() and not json.loads(resume.read_text(encoding="utf-8")).get("finished", True)
        except (OSError, ValueError):
            unfinished = False
        if not unfinished:
            info["moved"] = move_aside(root, ["calibration.jsonl"], "a fresh calibration campaign starts; earlier rows "
                                       "were measured by an earlier code tree and must not be refitted together", out)
    if stage == "simulation":
        inputs = sha256_of([fitted_model_file(root), SHIPPED_CALIBRATION / "formulas.json"])
        info["inputs_sha256"] = inputs
        if (state.get("simulation") or {}).get("inputs_sha256") != inputs:
            info["moved"] = move_aside(root, ["simulation.jsonl", "factorial.jsonl"],
                                       "modeled records were produced under another fitted model or formulas file", out)
    return info


def run_stage(stage: str, args: argparse.Namespace, root: Path, state: dict, out) -> int:
    logs = root / "logs"
    cmd = stage_command(stage, args, root)
    if cmd is None:
        out.write(f"[campaign] {stage}: skipped (no quick form; the smoke matrix of the diag4 stage covers the session path)\n")
        return 0
    if stage == "cpp_bench" and not cpp_binaries_present():
        if args.quick or args.dry_run:
            out.write(f"[campaign] {stage}: skipped, no C++ build in {CPP_BIN} ({BUILD_HINT})\n")
            return 0
        out.write(f"[campaign] {stage}: CAMPAIGN_CPP_BUILD_MISSING: {CPP_BIN} holds no opencrab2_dealer and opencrab2_bench "
                  f"({BUILD_HINT})\n")
        return 1
    prev = state.get(stage) or {}
    sig = signature(cmd, root)
    if args.dry_run and stage not in ("diag4", "full12", "web"):
        out.write(f"[campaign] {stage}: would run {' '.join(cmd)}\n")
        return 0
    if (not args.force and not args.dry_run and prev.get("status") == "done" and prev.get("signature") == sig
            and stage != "tables"):
        out.write(f"[campaign] {stage}: already done ({prev.get('finished_utc')}); use --force to rerun\n")
        return 0
    logs.mkdir(parents=True, exist_ok=True)
    info = {} if args.dry_run else prepare_stage(stage, root, state, out)
    log_path = logs / f"{stage}-{utc_stamp()}{'-dryrun' if args.dry_run else ''}.log"
    entry = dict(info, status="running", command=cmd, signature=sig, log=str(log_path), started_utc=utc_stamp())
    if not args.dry_run:
        state[stage] = entry
        save_state(logs, state)
    out.write(f"[campaign] {stage}: {' '.join(cmd)}\n[campaign] {stage}: log {log_path}\n")
    out.flush()
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8", newline="\n") as log:
        log.write(f"# stage {stage} started {entry['started_utc']}\n# command: {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=str(TREE), env=child_env(root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                log.flush()
                out.write(line)
                out.flush()
            rc = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            rc = 130
        log.write(f"# stage {stage} exit {rc} after {time.time() - t0:.1f} s\n")
    if not args.dry_run:
        entry.update(status="done" if rc == 0 else "failed", returncode=rc, finished_utc=utc_stamp(),
                     elapsed_s=round(time.time() - t0, 1))
        state[stage] = entry
        save_state(logs, state)
    out.write(f"[campaign] {stage}: exit {rc} after {time.time() - t0:.1f} s\n")
    return rc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run_campaign", description="OpenCrab v2 evaluation campaign (staged, resumable)")
    p.add_argument("--stage", default=None,
                   help=f"comma separated subset of {','.join(STAGES)}; 'campaign' = every stage but the calibration "
                        f"(default), 'all' = every stage (default with --quick)")
    p.add_argument("--quick", action="store_true", help="pipeline check in minutes: quick stage forms and the tiny model")
    p.add_argument("--name", default=None, help="run name: the root is runs/<name> (default: quick with --quick, else full)")
    p.add_argument("--out-root", default=None, help="data root instead of runs/<name>; 'results' (itself) rewrites the shipped outputs")
    p.add_argument("--workers", type=int, default=DIAG4_WORKERS, help="sessions side by side in the diag4 stage")
    p.add_argument("--full12-workers", type=int, default=FULL12_WORKERS, help="sessions side by side in the full12 stage")
    p.add_argument("--simulation-workers", type=int, default=SIMULATION_WORKERS, help="process pool size of the simulation")
    p.add_argument("--threads-per-process", type=int, default=THREADS_PER_PROCESS,
                   help="OMP/OPENBLAS/MKL threads of every launched party or service process")
    p.add_argument("--timeout", type=float, default=SESSION_TIMEOUT_S, help="per-session timeout in seconds")
    p.add_argument("--diag4-skip", default=DIAG4_SKIP, help="declared matrix reduction of the diag4 stage ('' for none)")
    p.add_argument("--full12-skip", default=FULL12_SKIP, help="declared matrix reduction of the full12 stage ('' for none)")
    p.add_argument("--web-model", default=WEB_MODEL, choices=["tiny", "gpt2"],
                   help="'tiny' measures the tiny model only; 'gpt2' adds the 1-layer and 4-layer GPT-2 prefixes")
    p.add_argument("--web-repeats", type=int, default=WEB_REPEATS)
    p.add_argument("--web-trials", type=int, default=WEB_TRIALS, help="rejection probes per kind")
    p.add_argument("--retry-aborted", action="store_true", help="session stages rerun cells whose only record is an abort")
    p.add_argument("--force", action="store_true", help="rerun stages that are recorded as done")
    p.add_argument("--keep-going", action="store_true", help="start later stages after a failed one")
    p.add_argument("--dry-run", action="store_true", help="print commands; session stages print cells and predicted time")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    spec = (args.stage if args.stage is not None else ("all" if args.quick else "campaign")).strip()
    if spec == "all":
        wanted = list(STAGES)
    elif spec == "campaign":
        wanted = list(CAMPAIGN_STAGES)
    else:
        wanted = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in STAGES]
    if unknown or not wanted:
        sys.stderr.write(f"CAMPAIGN_STAGE_UNKNOWN: {unknown}; known {list(STAGES)}\n")
        return 2
    if min(args.workers, args.full12_workers, args.simulation_workers, args.threads_per_process) < 1:
        sys.stderr.write("CAMPAIGN_PARAMS: workers and threads must be >= 1\n")
        return 2
    try:
        root = resolve_root(args.out_root, args.name or ("quick" if args.quick else "full"))
    except ValueError as exc:
        sys.stderr.write(str(exc) + "\n")
        return 2
    if root == RESULTS.resolve():
        sys.stdout.write("[campaign] writing into the shipped results/ (given explicitly with --out-root)\n")
    sys.stdout.write(f"[campaign] data root {root}{' (quick)' if args.quick else ''}\n")
    state = load_state(root / "logs")
    order = [s for s in STAGES if s in wanted]
    failed: List[str] = []
    t0 = time.time()
    for stage in order:
        rc = run_stage(stage, args, root, state, sys.stdout)
        if rc != 0:
            failed.append(stage)
            if not args.keep_going:
                break
    if failed:
        sys.stdout.write(f"[campaign] failed stage(s): {failed}\n")
        return 1
    sys.stdout.write(f"[campaign] finished: {order} in {time.time() - t0:.1f} s; data root {root}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
