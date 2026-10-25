#!/usr/bin/env python3
"""Live cross-language sessions over loopback: the dealer of one language serves the parties of the other.

    python cpp/tests/interop_cross_language.py [--bin cpp/build/bin]
        [--mode all|py-dealer|cpp-dealer] [--n-tokens 2] [--keep]

Modes:

  py-dealer   the Python dealer service (python -m opencrab2.platform.dealer_service, isolated mode,
              OS randomness) serves two C++ opencrab2_party processes running a tiny-decoder session
              (P0 holds a weights file written by "opencrab2_party --make-weights", P1 the prompt and
              the plaintext greedy reference of "opencrab2_party --reference").
  cpp-dealer  the C++ dealer (opencrab2_dealer, isolated mode, OS randomness) serves the Python
              tiny-model session: two Python party processes started by
              opencrab2.platform.launch.launch_two_party(..., external_dealer=...).

What crosses the language boundary in both modes, byte for byte: the dealer dialogue (paper,
Appendix B, DealerDelivery: HELLO, CHALLENGE, AUTH, REQ, BUNDLE + bytes frame over the typed wire
codec of opencrab2.highlevel.channel with the u64 BE length prefix), the HMAC tags over the canonical
header and the payload, the bundle headers (binding fields, shape_a / shape_b / shape spellings), and
every material payload: Beaver and select triples, matrix triples, weight masks and per-call masks,
vector shares, and the DCF-based DReLU (P = 1 and P = 2), truncation and ring_to_field material in
the binding key layout with the component-specific Convert words (paper, Section 5.3, a finding on
comparison keys). A session only produces the reference tokens when every gate evaluated the other
language's material correctly, so output_match is the end-to-end check.

What does NOT cross the boundary: the two parties of one session are always of the same language.
A C++ party cannot run a session with a Python party: the session layers differ from the first
message on.

Both evidence records of each session are validated with opencrab2.experiments.schema.validate.
Exit status 0 when every selected mode passed, 1 otherwise. Records go to a temporary directory
(removed afterwards unless --keep).
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
PYDIR = os.path.join(REPO, "python")
EXE = ".exe" if os.name == "nt" else ""
if PYDIR not in sys.path:
    sys.path.insert(0, PYDIR)

ISOLATED = "ISOLATED_RECIPIENT_BUNDLES"
CONTRACT = "NC-v2-f16-l40"


def exe(bindir: str, name: str) -> str:
    return os.path.join(bindir, name + EXE)


def default_bindir() -> str:
    return os.path.join(REPO, "cpp", "build", "bin")


def binaries_present(bindir: str) -> bool:
    return all(os.path.isfile(exe(bindir, n)) for n in ("opencrab2_dealer", "opencrab2_party"))


def _last_json(text: str) -> dict:
    for line in reversed(text.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError("INTEROP_NO_JSON: no JSON line in the tool output")


def _tail(path: str, n: int = 1500) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()[-n:]
    except OSError:
        return ""


def _records(path: str) -> list:
    from opencrab2.experiments import schema
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                schema.validate(rec)
                out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Python dealer, C++ parties
# ---------------------------------------------------------------------------
def run_py_dealer_cpp_parties(bindir: str, tmp: str, n_tokens: int = 2, timeout_s: float = 300.0) -> dict:
    """Returns {ok, reference, tokens, records, dealer_delivered, notes}."""
    setup_id = "setup-interop-pydealer"
    keys = os.path.join(tmp, "pydealer_keys.json")
    with open(keys, "w", encoding="utf-8") as fh:
        json.dump({"0": secrets.token_hex(32), "1": secrets.token_hex(32)}, fh)
    port_file = os.path.join(tmp, "pydealer.port")
    dealer_log = os.path.join(tmp, "pydealer.log")
    env = dict(os.environ)
    env["PYTHONPATH"] = PYDIR + os.pathsep + env.get("PYTHONPATH", "")
    procs = []
    result = {"ok": False, "notes": []}
    log = open(dealer_log, "w", encoding="utf-8")
    dealer = subprocess.Popen([sys.executable, "-m", "opencrab2.platform.dealer_service", "--listen", "127.0.0.1:0", "--keys", keys,
                               "--contract", CONTRACT, "--setup-id", setup_id, "--model-version", "tiny-decoder-v1",
                               "--mode", ISOLATED, "--port-file", port_file],
                              cwd=PYDIR, env=env, stdout=subprocess.DEVNULL, stderr=log)
    procs.append(dealer)
    try:
        deadline = time.time() + 60.0
        port = None
        while time.time() < deadline and dealer.poll() is None:
            if os.path.isfile(port_file):
                text = open(port_file, "r", encoding="utf-8").read().strip()
                if text:
                    port = int(text)
                    break
            time.sleep(0.05)
        if port is None:
            result["notes"].append("the Python dealer did not report a port: " + _tail(dealer_log))
            return result
        weights = os.path.join(tmp, "tiny_weights.bin")
        party = exe(bindir, "opencrab2_party")
        r = subprocess.run([party, "--make-weights", weights, "--seed", "2027", "--max-pos", "64"], capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            result["notes"].append("make-weights failed: " + r.stderr[-400:])
            return result
        prompt_ids = [3, 1, 4, 1]
        prompt = ",".join(str(t) for t in prompt_ids)
        r = subprocess.run([party, "--reference", "--weights", weights, "--prompt-ids", prompt, "--generate", str(n_tokens)],
                           capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            result["notes"].append("reference failed: " + r.stderr[-400:])
            return result
        reference = _last_json(r.stdout)["tokens"]
        out = os.path.join(tmp, "pydealer_evidence.jsonl")
        common = ["--dealer", "127.0.0.1:%d" % port, "--keys", keys, "--setup-id", setup_id, "--out", out, "--journal-dir", tmp,
                  "--generate", str(n_tokens), "--prompt-len", str(len(prompt_ids)), "--policy", "B4_opencrab"]
        err0 = open(os.path.join(tmp, "pydealer_p0.err"), "w", encoding="utf-8")
        err1 = open(os.path.join(tmp, "pydealer_p1.err"), "w", encoding="utf-8")
        p0 = subprocess.Popen([party, "--party", "0", "--listen", "127.0.0.1:0", "--weights", weights] + common,
                              stdout=subprocess.PIPE, stderr=err0, text=True)
        procs.append(p0)
        ready = json.loads(p0.stdout.readline())
        p1 = subprocess.Popen([party, "--party", "1", "--peer", "127.0.0.1:%d" % int(ready["port"]), "--prompt-ids", prompt,
                               "--reference-tokens", ",".join(str(t) for t in reference)] + common,
                              stdout=subprocess.DEVNULL, stderr=err1, text=True)
        procs.append(p1)
        try:
            c1 = p1.wait(timeout=timeout_s)
            p0.stdout.read()
            c0 = p0.wait(timeout=60.0)
        except subprocess.TimeoutExpired:
            result["notes"].append("a C++ party did not finish in time")
            return result
        finally:
            err0.close()
            err1.close()
        if c0 != 0 or c1 != 0:
            result["notes"].append("exit codes p0=%s p1=%s; p0: %s; p1: %s" % (c0, c1, _tail(os.path.join(tmp, "pydealer_p0.err"), 500),
                                                                             _tail(os.path.join(tmp, "pydealer_p1.err"), 500)))
            return result
        recs = _records(out)
        by_party = {r["extra"].get("party"): r for r in recs}
        log.flush()
        delivered = _tail(dealer_log, 1 << 22).count("delivered recipient=")
        ok = len(recs) == 2 and set(by_party) == {0, 1}
        if ok:
            r0, r1 = by_party[0], by_party[1]
            ok = ok and r1["output_match"] is True and r1["extra"]["tokens_list"] == reference
            ok = ok and r0["bytes"] == r1["bytes"] and r0["rounds"] == r1["rounds"]
            for rec in recs:
                ok = ok and rec["dealer_mode"] == ISOLATED and rec["aborts"] == 0 and rec["extra"].get("language") == "cpp"
                ok = ok and rec["security_eligible"] is True and rec["status"] == "MEASURED_NATIVE"
            ok = ok and delivered > 0
            result.update(reference=reference, tokens=r1["extra"]["tokens_list"], bytes=r1["bytes"], rounds=r1["rounds"],
                          material_consumed=r1["material_consumed"])
        result.update(ok=bool(ok), records=recs, dealer_delivered=delivered)
        return result
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
            try:
                p.wait(timeout=10)
            except Exception:
                pass
        log.close()


# ---------------------------------------------------------------------------
# C++ dealer, Python parties
# ---------------------------------------------------------------------------
def run_cpp_dealer_python_session(bindir: str, tmp: str, n_tokens: int = 2, timeout_s: float = 600.0) -> dict:
    """Returns {ok, reference, tokens, records, dealer_stats, notes}. Call from a __main__ guarded process (spawn)."""
    scripts = os.path.join(REPO, "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import smoke_session as sm
    from opencrab2.highlevel import session as S
    from opencrab2.highlevel.contract import DEFAULT_CONTRACT
    from opencrab2.platform import launch as L

    result = {"ok": False, "notes": []}
    W, meta = sm.make_tiny_weights(1)
    npz = S.save_model_npz(os.path.join(tmp, "tiny.npz"), W, meta)
    prompt = list(sm.TINY_PROMPT)
    reference = sm.reference_tokens(W, meta, 1, prompt, n_tokens)
    model_version = S.model_version_of(meta)
    setup_id = "setup-interop-cppdealer"
    keys = os.path.join(tmp, "cppdealer_keys.json")
    err_path = os.path.join(tmp, "cppdealer.err")
    err = open(err_path, "w", encoding="utf-8")
    dealer = subprocess.Popen([exe(bindir, "opencrab2_dealer"), "--listen", "127.0.0.1:0", "--keys", keys, "--gen-keys",
                               "--setup-id", setup_id, "--model-version", model_version, "--contract", DEFAULT_CONTRACT.name],
                              stdout=subprocess.PIPE, stderr=err, text=True)
    try:
        ready = json.loads(dealer.stdout.readline())
        if ready.get("mode") != ISOLATED or not str(ready.get("randomness", "")).startswith("os_csprng"):
            result["notes"].append("the C++ dealer is not isolated: %r" % (ready,))
            return result
        out_dir = os.path.join(tmp, "cppdealer_session")
        cfg = S.SessionConfig(model=f"npz:{npz}", n_layers=1, prompt_ids=prompt, n_generate=n_tokens, policy="B4_opencrab",
                              trace_id="unshaped", seed=1, contract=DEFAULT_CONTRACT, length_bucket=8,
                              session_id="interop-cppdealer-1", reference_tokens=reference, setup_id=setup_id,
                              model_version=model_version)
        try:
            res = L.launch_two_party(cfg, None, out_dir, timeout_s=timeout_s, external_dealer=("127.0.0.1", int(ready["port"]), keys))
        except Exception as exc:   # noqa: BLE001 (reported to the caller)
            result["notes"].append("launch failed: %s: %s" % (exc.__class__.__name__, exc))
            return result
        recs = _records(res["records"][0])
        tokens = list(res["summaries"][1].get("generated_tokens") or [])
        ok = len(recs) == 2 and tokens == list(reference)
        for rec in recs:
            ok = ok and rec["dealer_mode"] == ISOLATED and rec["aborts"] == 0
            # output_match stays with the client (P1): the record of P0 carries None
            ok = ok and rec["output_match"] is (True if rec.get("extra", {}).get("party") == 1 else None)
            ok = ok and rec["security_eligible"] is True and rec["status"] == "MEASURED_NATIVE"
        result.update(ok=bool(ok), reference=reference, tokens=tokens, records=recs)
        if recs:
            result.update(bytes=recs[0]["bytes"], rounds=recs[0]["rounds"], material_consumed=recs[0]["material_consumed"])
        return result
    finally:
        if dealer.poll() is None:
            dealer.kill()
        try:
            dealer.wait(timeout=10)
        except Exception:
            pass
        err.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", default=default_bindir())
    ap.add_argument("--mode", default="all", choices=["all", "py-dealer", "cpp-dealer"])
    ap.add_argument("--n-tokens", type=int, default=2)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args(argv)
    if not binaries_present(args.bin):
        print("INTEROP_BINARIES_MISSING: build first with: bash scripts/ubuntu.sh build on Ubuntu, python cpp/build.py on Windows (looked in %s)" % args.bin)
        return 1
    tmp = tempfile.mkdtemp(prefix="oc2_interop_")
    status = {}
    try:
        for mode, fn in (("py-dealer", run_py_dealer_cpp_parties), ("cpp-dealer", run_cpp_dealer_python_session)):
            if args.mode not in ("all", mode):
                continue
            t0 = time.time()
            res = fn(args.bin, tmp, args.n_tokens)
            status[mode] = bool(res["ok"])
            summary = {k: res.get(k) for k in ("reference", "tokens", "bytes", "rounds", "material_consumed", "dealer_delivered")
                       if res.get(k) is not None}
            print("[%s] %s in %.1f s %s" % (mode, "OK" if res["ok"] else "FAILED", time.time() - t0, json.dumps(summary, sort_keys=True)))
            for note in res["notes"]:
                print("[%s] note: %s" % (mode, note))
    finally:
        if args.keep:
            print("[interop] kept", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    print("[interop] SUMMARY: " + " ".join("%s=%s" % (m, "OK" if ok else "FAILED") for m, ok in status.items()))
    return 0 if status and all(status.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
