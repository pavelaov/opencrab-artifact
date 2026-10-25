"""Run the C++ building-block benchmark against an isolated C++ dealer process.

Starts ``opencrab2_dealer`` on an ephemeral loopback port with freshly generated recipient keys,
runs ``opencrab2_bench`` against it (every record is then MEASURED_NATIVE and carries
dealer_mode ISOLATED_RECIPIENT_BUNDLES), and writes ``raw/cpp_bench.jsonl`` under the data root
(``--out``; default ``$OPENCRAB2_RESULTS_DIR/raw/cpp_bench.jsonl``, else ``runs/cpp_bench/raw/cpp_bench.jsonl``
under the artifact root; the shipped ``results/`` is never the default target).

Usage (from the artifact root):

    python scripts/run_cpp_bench.py [--bin cpp/build/bin] [--out runs/x/raw/cpp_bench.jsonl]
                                    [--batches 1024,4096,16384,65536] [--repeats 7] [--quick]

``--quick`` is the pipeline check of ``run_campaign.py --quick``: one batch size, one repeat, the
smallest argmax shape. Design limit: both processes share one host; timings are loopback timings.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                       # the artifact root (python/, cpp/, results/, scripts/)

QUICK = {"batches": "1024", "repeats": "1", "argmax_n": "256", "argmax_k": "2", "argmax_repeats": "1", "bridge_n": "128"}


def default_out() -> Path:
    override = os.environ.get("OPENCRAB2_RESULTS_DIR", "").strip()
    root = Path(override).expanduser() if override else ROOT / "runs" / "cpp_bench"
    return root / "raw" / "cpp_bench.jsonl"


def bench_paths(out: Path):
    """(directory the bench runs in, output path relative to it): the data root and raw/<file> when the file
    lies in a directory named raw, else the file's own directory and its name."""
    out = out.resolve()
    run_dir = out.parent.parent if out.parent.name == "raw" else out.parent
    return run_dir, out.relative_to(run_dir).as_posix()


def exe(bindir: Path, name: str) -> str:
    p = bindir / (name + (".exe" if os.name == "nt" else ""))
    if not p.exists():
        raise SystemExit("CPP_BENCH_BINARY_MISSING: %s (build with: bash scripts/ubuntu.sh build (CMake) on Ubuntu, "
                         "python cpp/build.py on Windows)" % p)
    return str(p)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="run_cpp_bench", description="C++ building-block benchmark against the C++ dealer")
    ap.add_argument("--bin", default=str(ROOT / "cpp" / "build" / "bin"))
    ap.add_argument("--out", default=None, help="output file (default: raw/cpp_bench.jsonl under the data root)")
    ap.add_argument("--batches", default=None, help="comma separated batch sizes (default 1024,4096,16384,65536)")
    ap.add_argument("--repeats", default=None, help="timed repeats per shape (default 7)")
    ap.add_argument("--argmax-n", default=None, help="default 256,4096,50257")
    ap.add_argument("--argmax-k", default=None, help="default 2,4,8")
    ap.add_argument("--argmax-repeats", default=None, help="default 3")
    ap.add_argument("--bridge-n", default=None, help="default 128")
    ap.add_argument("--quick", action="store_true", help="one batch, one repeat, smallest argmax shape (pipeline check)")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    full = {"batches": "1024,4096,16384,65536", "repeats": "7", "argmax_n": "256,4096,50257", "argmax_k": "2,4,8",
            "argmax_repeats": "3", "bridge_n": "128"}
    base = QUICK if args.quick else full
    opt = {k: (getattr(args, k) if getattr(args, k) is not None else v) for k, v in base.items()}

    bindir = Path(args.bin)
    out = Path(args.out) if args.out else default_out()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():                                   # an earlier run of this root: kept next to the new file
        stale = out.with_suffix(".jsonl.prev")
        if stale.exists():
            stale.unlink()
        out.rename(stale)

    dealer_exe, bench_exe = exe(bindir, "opencrab2_dealer"), exe(bindir, "opencrab2_bench")
    tmp = Path(tempfile.mkdtemp(prefix="oc2_cpp_bench_"))
    keys = tmp / "keys.json"
    setup_id = "setup-bench-%d" % int(time.time())
    # Both dealer streams go to files. A pipe that nobody drains fills up after a few hundred
    # log lines and blocks the dealer, which the benchmark then sees as a dead connection.
    dealer_log = open(tmp / "dealer.log", "wb")
    dealer_out_path = tmp / "dealer.out"
    dealer_out = open(dealer_out_path, "wb")
    rc = 1
    dealer = subprocess.Popen(
        [dealer_exe, "--listen", "127.0.0.1:0", "--keys", str(keys), "--gen-keys",
         "--setup-id", setup_id, "--model-version", "bench-model", "--contract", "NC-v2-f16-l40"],
        stdout=dealer_out, stderr=dealer_log, cwd=str(ROOT))
    try:
        port = None
        deadline = time.time() + 30.0
        while time.time() < deadline and port is None:
            if dealer.poll() is not None:
                break
            time.sleep(0.1)
            try:
                text_all = dealer_out_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for text in text_all.splitlines():
                try:
                    d = json.loads(text.strip())
                except ValueError:
                    continue
                if isinstance(d, dict) and "port" in d:
                    port = int(d["port"])
                    break
        if port is None:
            raise SystemExit("CPP_BENCH_DEALER_START: the dealer did not announce a port (see %s)" % (tmp / "dealer.log"))
        # the bench runs in the data root and gets the output path relative to it, so that the records
        # name their file as raw/cpp_bench.jsonl and never carry a host path
        run_dir, rel_out = bench_paths(out)
        cmd = [bench_exe, "--out", rel_out, "--batches", opt["batches"], "--repeats", opt["repeats"],
               "--argmax-n", opt["argmax_n"], "--argmax-k", opt["argmax_k"], "--argmax-repeats", opt["argmax_repeats"],
               "--bridge-n", opt["bridge_n"],
               "--dealer", "127.0.0.1:%d" % port, "--keys", str(keys), "--setup-id", setup_id]
        print("CPP_BENCH_RUN:", " ".join(cmd), "(in %s)" % run_dir, flush=True)
        t0 = time.time()
        rc = subprocess.call(cmd, cwd=str(run_dir))
        print("CPP_BENCH_EXIT: %d after %.1f s" % (rc, time.time() - t0), flush=True)
    finally:
        dealer.terminate()
        try:
            dealer.wait(timeout=10)
        except subprocess.TimeoutExpired:
            dealer.kill()
        dealer_log.close()
        dealer_out.close()
        # the directory holds the dealer's recipient keys and logs: kept for inspection only after a failure
        if rc == 0:
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            print("CPP_BENCH_DEALER_FILES: kept in %s" % tmp, flush=True)

    n = 0
    statuses: dict = {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            n += 1
            statuses[rec.get("status")] = statuses.get(rec.get("status"), 0) + 1
    print("CPP_BENCH_RECORDS: %d %s -> %s" % (n, json.dumps(statuses, sort_keys=True), out), flush=True)
    return 0 if n and rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
