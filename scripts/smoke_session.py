"""Smoke test of the three-process session path (dealer + P0 + P1) through ``opencrab2.platform.launch``.

Runs one 1-layer loopback session with the isolated dealer in a separate
interpreter and the two parties as spawned processes, exactly as the
evaluation scripts do, then prints the two evidence records' key fields and
checks that both validate, are security eligible, and that P1's released tokens
match the float reference (erf-GELU) of the same functionality.

Default model: the random-weight tiny GPT-2 layout with a byte vocabulary of
``opencrab2.experiments.workloads`` (no download, about 20 s). ``--model gpt2``
uses the pinned local snapshot of ``openai-community/gpt2`` (1 layer, about one
minute, several GB of RAM).

Usage (from the artifact root):

    python scripts/smoke_session.py [--model tiny|gpt2] [--n-generate 2] [--out-dir DIR]

Exit status 0 on success, 1 on any check failure. This script is a smoke
path, not a benchmark: it writes into a temporary directory unless
``--out-dir`` is given and never appends to the shipped ``results/raw``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from typing import List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                                              # the artifact root
sys.path.insert(0, os.path.join(ROOT, "python"))

from opencrab2.experiments import schema                                   # noqa: E402
from opencrab2.experiments.workloads import TINY_PROMPT as _TINY_PROMPT   # noqa: E402
from opencrab2.experiments.workloads import make_tiny_weights             # noqa: E402
from opencrab2.highlevel import session as S                              # noqa: E402
from opencrab2.highlevel.contract import DEFAULT_CONTRACT                 # noqa: E402
from opencrab2.highlevel.dealer import MODE_ISOLATED                      # noqa: E402
from opencrab2.highlevel.transformer import reference_gpt2_logits         # noqa: E402
from opencrab2.platform import launch as L                                # noqa: E402

TINY_PROMPT = list(_TINY_PROMPT)                 # four byte-level tokens
GPT2_PROMPT = [464, 3139, 286, 4881]             # "The capital of France"

__all__ = ["TINY_PROMPT", "GPT2_PROMPT", "make_tiny_weights", "reference_tokens", "main"]


def reference_tokens(W: dict, meta: dict, n_layers: int, prompt: List[int], n_gen: int) -> List[int]:
    """Float greedy decode of the declared functionality (erf-GELU) for ``output_match``."""
    cur = list(prompt)
    out: List[int] = []
    for _ in range(n_gen):
        lg = reference_gpt2_logits(W, meta, n_layers, cur)[-1]
        t = int(np.argmax(lg))
        out.append(t)
        cur.append(t)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="smoke_session")
    ap.add_argument("--model", default="tiny", choices=["tiny", "gpt2"])
    ap.add_argument("--n-generate", type=int, default=2)
    ap.add_argument("--policy", default="B4_opencrab")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--timeout-s", type=float, default=900.0)
    args = ap.parse_args(argv)

    out_dir = args.out_dir or tempfile.mkdtemp(prefix="oc2-smoke-")
    os.makedirs(out_dir, exist_ok=True)
    n_layers = 1
    if args.model == "tiny":
        W, meta = make_tiny_weights(n_layers)
        npz = S.save_model_npz(os.path.join(out_dir, "tiny.npz"), W, meta)
        model = f"npz:{npz}"
        prompt = list(TINY_PROMPT)
        length_bucket = 8
    else:
        W, meta = S.load_model("gpt2", n_layers)
        model = "gpt2"
        prompt = list(GPT2_PROMPT)
        length_bucket = 16
    ref = reference_tokens(W, meta, n_layers, prompt, args.n_generate)
    trace = {"trace_id": "S0-stable-smoke", "family": "S0", "split": "test", "seed": 1,
             "points": [(0.0, 5e8, 0.0002, 1.0, None), (600.0, 5e8, 0.0002, 1.0, None)]}
    cfg = S.SessionConfig(model=model, n_layers=n_layers, prompt_ids=prompt, n_generate=args.n_generate,
                          policy=args.policy, trace_id=trace["trace_id"], seed=1, contract=DEFAULT_CONTRACT,
                          length_bucket=length_bucket, session_id="smoke-1", reference_tokens=ref)
    t0 = time.time()
    res = L.launch_two_party(cfg, trace, out_dir, timeout_s=args.timeout_s)
    wall = time.time() - t0

    ok = True
    recs = [json.loads(l) for l in open(res["records"][0], encoding="utf-8") if l.strip()]
    for r in recs:
        schema.validate(r)
        line = {k: r.get(k) for k in ("status", "security_eligible", "dealer_mode", "diagnostic_reveals",
                                      "workload_scope", "policy_id", "tokens", "T_cold_s", "T_online_s", "bytes",
                                      "rounds", "material_consumed", "material_burned", "output_match", "aborts")}
        print(json.dumps(line, sort_keys=True))
        ok &= r["status"] == "MEASURED_NATIVE" and r["security_eligible"] is True
        ok &= r["dealer_mode"] == MODE_ISOLATED and r["diagnostic_reveals"] == 0 and r["aborts"] == 0
    generated = res["summaries"][1].get("generated_tokens")
    print(json.dumps({"generated_tokens": generated, "reference_tokens": ref, "launch_wall_s": round(wall, 2),
                      "out_dir": out_dir, "records": res["records"]}))
    ok &= list(generated or []) == list(ref)
    ok &= not res["errors"]
    print("SMOKE_OK" if ok else "SMOKE_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
