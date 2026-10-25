#!/usr/bin/env python3
"""Multi-process loopback runs of the C++ platform, checked against the Python evidence schema.

    python cpp/tests/smoke_three_process.py
        [--bin cpp/build/bin] [--mode all|web|party|cancel|bench] [--out FILE] [--n-tokens 3]
        [--prompt "secure inference"] [--policy B4_opencrab] [--model-seed 2026]
        [--trace-family S1 --trace-variant ramp-LAN-to-WAN-c --trace-seed 2 --trace-duration 6]
        [--verify-in-sidecar] [--handler tiny|demo] [--keep]

Modes (every mode starts its own isolated dealer process, opencrab2_dealer, on an ephemeral port):

  web     dealer + model server (P0) + sidecar (P1). One generation is POSTed to the sidecar, the
          job is followed through GET /v1/status, the web metrics block (TTFT, inter-token latency,
          completion, cancel waste, rejection counters) is checked, and the two evidence records
          are read back. The model is a weights file written by "opencrab2_party --make-weights"
          and given to the server only; this launcher, which holds the model and the prompt like
          an experimenter, computes the plaintext greedy reference with "opencrab2_party
          --reference" and compares it with the tokens P1 released (launcher_output_match). With
          --verify-in-sidecar the sidecar loads the weights itself (a test mode:
          the P1 record is then DIAGNOSTIC because the weights were disclosed).
  party   dealer + two opencrab2_party processes running the highlevel session loop over a TCP
          channel shaped by a trace of python -m opencrab2.experiments.traces (written here with
          traces.make_trace and traces.write_trace). P0 gets the weights file, P1 the prompt ids
          and the reference tokens.
  cancel  the same pair with --cancel-after-epoch 1 --resume: declared cancellation, then
          resumption under the same session id inside the same processes.
  bench   opencrab2_bench --print-schema is compared field by field with schema.py; a quick
          bench runs against the isolated dealer (every record MEASURED_NATIVE) and once more
          without a dealer (shared seed: every record DIAGNOSTIC).

Every evidence line written by a C++ process is validated with
opencrab2.experiments.schema.validate (the Python validator, imported from python/). Records go to
--out; without --out they go to a temporary file that is removed afterwards, so a smoke run never
adds lines to results/raw by accident.

Exit status 0 when every selected mode passed; 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
EXE = ".exe" if os.name == "nt" else ""
sys.path.insert(0, os.path.join(REPO, "python"))

from opencrab2.experiments import schema, traces  # noqa: E402

ISOLATED = "ISOLATED_RECIPIENT_BUNDLES"
SHARED_SEED = "INSECURE_TEST_ONLY_SHARED_SEED"


class Proc:
    """One child process. stdout is drained by a thread (the processes print JSON event lines and,
    at the end of a session, one large record line); stderr goes to a file in the temporary
    directory (the server logs one line per frame, which would fill an undrained pipe and block
    the process)."""

    def __init__(self, name: str, argv: list, cwd: str, tmp: str) -> None:
        self.name = name
        self.argv = argv
        self.err_path = os.path.join(tmp, name + ".stderr.log")
        self.err_file = open(self.err_path, "w", encoding="utf-8")
        self.p = subprocess.Popen(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=self.err_file, text=True,
                                  encoding="utf-8", errors="replace", bufsize=1)
        self.ready = None
        self.lines = []
        self.lock = threading.Lock()
        self.events = {}
        self.reader = threading.Thread(target=self._drain, daemon=True)
        self.reader.start()

    def _drain(self) -> None:
        for line in self.p.stdout:
            line = line.strip()
            if not line:
                continue
            with self.lock:
                self.lines.append(line)
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                ev = obj.get("event") if isinstance(obj, dict) else None
                if ev:
                    self.events.setdefault(ev, []).append(obj)

    def wait_ready(self, event: str, timeout_s: float = 60.0) -> dict:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            with self.lock:
                if self.events.get(event):
                    self.ready = self.events[event][0]
                    return self.ready
            if self.p.poll() is not None:
                raise SystemExit("%s exited early (code %s): %s" % (self.name, self.p.returncode, self.stderr_tail()))
            time.sleep(0.05)
        raise SystemExit("%s: no %s line within %.0f s" % (self.name, event, timeout_s))

    def wait_exit(self, timeout_s: float) -> int:
        try:
            self.p.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self.p.kill()
            self.p.wait()
            return -9
        self.reader.join(timeout=5.0)
        return self.p.returncode

    def stderr_tail(self, n: int = 2000) -> str:
        try:
            self.err_file.flush()
            with open(self.err_path, encoding="utf-8", errors="replace") as fh:
                return fh.read()[-n:]
        except OSError:
            return ""

    def stop(self, timeout_s: float = 20.0) -> tuple:
        """Returns (exit code or "terminated", stderr tail). The apps stop gracefully on SIGINT or
        SIGTERM (final counters line, exit 0); Python's terminate() is TerminateProcess on Windows,
        which skips the handlers, so a process that was still serving reports "terminated"."""
        terminated = False
        if self.p.poll() is None:
            self.p.terminate()
            terminated = True
        try:
            self.p.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self.p.kill()
            self.p.wait()
        self.reader.join(timeout=5.0)
        err = self.stderr_tail()
        self.err_file.close()
        code = self.p.returncode
        if terminated and (code != 0):
            code = "terminated"
        return code, err


def http(method: str, url: str, token: str, body: dict | None = None, timeout: float = 30.0) -> tuple:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", "Bearer " + token)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode(errors="replace")}


def exe(bindir: str, name: str) -> str:
    return os.path.join(bindir, name + EXE)


def run_tool(argv: list, timeout: float = 600.0) -> tuple:
    r = subprocess.run(argv, cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    return r.returncode, r.stdout, r.stderr


def read_new_records(path: str, n_before: int) -> list:
    """The lines appended to path since n_before, each validated with the Python schema."""
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    for line in lines[n_before:]:
        if not line.strip():
            continue
        rec = json.loads(line)
        schema.validate(rec)  # raises ValueError with a SCHEMA_* code on any non-conforming C++ line
        out.append(rec)
    return out


def count_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as fh:
        return sum(1 for _ in fh)


def describe(rec: dict) -> str:
    e = rec.get("extra", {})
    return ("party=%s status=%s eligible=%s policy=%s tokens=%s list=%s output_match=%s bytes=%s rounds=%s T_cold=%.3f "
            "T_online=%.3f switches=%s inits=%s aborts=%s burned=%s plan=%s trace=%s notes=%s" % (
                e.get("party"), rec.get("status"), rec.get("security_eligible"), rec.get("policy_id"), rec.get("tokens"),
                e.get("tokens_list"), rec.get("output_match"), rec.get("bytes"), rec.get("rounds"), rec.get("T_cold_s") or 0.0,
                rec.get("T_online_s") or 0.0, rec.get("post_init_switches"), rec.get("initializations"), rec.get("aborts"),
                rec.get("material_burned"), json.dumps(e.get("final_plan")), rec.get("trace_id"),
                json.dumps([n.split(":")[0] for n in e.get("eligibility_notes", [])])))


def start_dealer(bindir: str, tmp: str, setup_id: str, model_version: str, name: str = "dealer") -> tuple:
    keys = os.path.join(tmp, name + "_keys.json")
    dealer = Proc(name, [exe(bindir, "opencrab2_dealer"), "--listen", "127.0.0.1:0", "--keys", keys, "--gen-keys",
                         "--setup-id", setup_id, "--model-version", model_version, "--contract", "NC-v2-f16-l40"], REPO, tmp)
    d = dealer.wait_ready("dealer_ready")
    assert d["mode"] == ISOLATED, "the smoke dealer must be isolated"
    assert d["randomness"].startswith("os_csprng"), "the isolated dealer must draw from the OS CSPRNG"
    return dealer, keys, "127.0.0.1:%d" % d["port"], d


def make_model(bindir: str, tmp: str, seed: int, prompt_ids: list, n_tokens: int, max_pos: int) -> tuple:
    """Weights file for P0 and the plaintext greedy reference (launcher side: it holds model and prompt)."""
    weights = os.path.join(tmp, "tiny_weights_%d.bin" % seed)
    code, out, err = run_tool([exe(bindir, "opencrab2_party"), "--make-weights", weights, "--seed", str(seed), "--max-pos", str(max_pos)])
    assert code == 0, "make-weights failed: " + err
    made = json.loads(out.strip().splitlines()[-1])
    code, out, err = run_tool([exe(bindir, "opencrab2_party"), "--reference", "--weights", weights, "--prompt-ids",
                               ",".join(str(t) for t in prompt_ids), "--generate", str(n_tokens)])
    assert code == 0, "reference failed: " + err
    ref = json.loads(out.strip().splitlines()[-1])
    assert ref["model_digest"] == made["model_digest"]
    return weights, ref["tokens"], made["model_config"]


# ---------------------------------------------------------------------------
# mode web: dealer + server + sidecar
# ---------------------------------------------------------------------------
def mode_web(args, bindir: str, tmp: str, out_path: str) -> bool:
    procs = []
    ok = True
    t_start = time.time()
    model_version, setup_id = "tiny-decoder-v1", "setup-web-1"
    common = ["--model-version", model_version, "--setup-id", setup_id, "--contract", "NC-v2-f16-l40"]
    vocab = 32
    prompt_ids = [b % vocab for b in args.prompt.encode("utf-8")] or [0]  # the sidecar's public byte -> id rule
    try:
        dealer, keys, dealer_addr, d = start_dealer(bindir, tmp, setup_id, model_version, "web_dealer")
        procs.append(dealer)
        print("[web] dealer   ", json.dumps(d))
        token_file = os.path.join(tmp, "token.txt")
        reference = None
        server_argv = [exe(bindir, "opencrab2_server"), "--ws", "127.0.0.1:0", "--dealer", dealer_addr, "--keys", keys,
                       "--handler", args.handler] + common
        sidecar_extra = []
        if args.handler == "tiny":
            server_argv += ["--out", out_path, "--journal-dir", tmp]
            if args.verify_in_sidecar:
                weights = os.path.join(tmp, "tiny_weights_seeded.bin")
                server_argv += ["--tiny-seed", str(args.model_seed), "--weights-out", weights]
                sidecar_extra = ["--verify-weights", weights]
            else:
                weights, reference, model_cfg = make_model(bindir, tmp, args.model_seed, prompt_ids, args.n_tokens, 64)
                assert model_cfg["vocab"] == vocab
                server_argv += ["--weights", weights]
            if args.web_trace:
                server_argv += ["--trace", args.web_trace]
        else:
            server_argv += ["--demo-n", "64", "--demo-seed", "42"]
        server = Proc("server", server_argv, REPO, tmp)
        procs.append(server)
        s = server.wait_ready("server_ready")
        print("[web] server   ", json.dumps(s))

        sidecar_argv = [exe(bindir, "opencrab2_sidecar"), "--http", "127.0.0.1:0", "--server", "ws://127.0.0.1:%d/session" % s["port"],
                        "--dealer", dealer_addr, "--keys", keys, "--engine", args.handler, "--token-file", token_file] + common
        if args.handler == "tiny":
            sidecar_argv += ["--out", out_path, "--policy", args.policy, "--journal-dir", tmp] + sidecar_extra
        else:
            sidecar_argv += ["--demo-n", "64"]
        sidecar = Proc("sidecar", sidecar_argv, REPO, tmp)
        procs.append(sidecar)
        c = sidecar.wait_ready("sidecar_ready")
        print("[web] sidecar  ", json.dumps({k: v for k, v in c.items() if k != "launch_url"}))  # the URL carries the bearer token
        base = "http://127.0.0.1:%d" % c["port"]
        with open(token_file, encoding="utf-8") as fh:
            token = fh.read().strip()

        st, _ = http("GET", base + "/v1/status", "not-the-token")
        print("[web] GET /v1/status with a wrong token ->", st)
        ok = ok and st == 401

        n_before = count_lines(out_path)
        t_gen = time.time()
        st, job = http("POST", base + "/v1/generate", token, {"prompt": args.prompt, "n_tokens": args.n_tokens})
        print("[web] generate", st, json.dumps(job))
        assert st == 202, "generate not accepted"
        job_id = job["job_id"]
        full_status = {}
        mine = {}
        tokens_seen = 0
        while True:
            st, full_status = http("GET", base + "/v1/status", token)
            jobs = [j for j in full_status.get("jobs", []) if j.get("job_id") == job_id]
            if jobs and jobs[0].get("tokens_emitted", 0) != tokens_seen:
                tokens_seen = jobs[0]["tokens_emitted"]
                print("[web] tokens emitted: %d (%.2f s)" % (tokens_seen, time.time() - t_gen))
            if jobs and jobs[0].get("state") not in ("queued", "running"):
                mine = jobs[0]
                break
            if time.time() - t_gen > 900:
                raise SystemExit("job did not finish within 900 s")
            time.sleep(0.2)
        t_done = time.time() - t_gen
        print("[web] job", json.dumps(mine))
        ok = ok and mine.get("state") == "done"
        if mine.get("state") != "done":
            print("JOB_FAILED:", mine.get("error_code"))

        # web metrics (TTFT, inter-token latency, counters) in GET /v1/status
        m = full_status.get("metrics", {})
        print("[web] metrics", json.dumps(m))
        metrics_ok = (m.get("ttft_s", {}).get("n") == 1 and m["ttft_s"]["cold_s"] > 0 and m["ttft_s"]["mean"] > 0
                      and m.get("inter_token_s", {}).get("n") == max(0, args.n_tokens - 1)
                      and m.get("completion", {}).get("completed") == 1 and m["completion"]["rate"] == 1.0
                      and m.get("cancel_waste", {}).get("jobs") == 0
                      and m.get("rejections", {}).get("auth") == 1 and m["rejections"]["origin"] == 0
                      and m["rejections"]["version_mismatch"] == 0 and m["rejections"]["stale_session"] == 0)
        metrics_ok = metrics_ok and mine.get("completed") is True and mine.get("ttft_s", -1) > 0 and mine.get("cold") is True
        if args.n_tokens > 1:
            metrics_ok = metrics_ok and m["inter_token_s"]["mean"] > 0 and mine["inter_token_s"]["n"] == args.n_tokens - 1
        if args.handler == "tiny":
            jm = mine.get("metrics", {})
            metrics_ok = metrics_ok and jm.get("material_consumed_bytes", 0) > 0 and jm.get("T_online_s", 0) > 0
        if not metrics_ok:
            print("[web] METRICS CHECK FAILED")
        ok = ok and metrics_ok

        records = read_new_records(out_path, n_before) if args.handler == "tiny" else []
        for rec in records:
            print("[web] record", describe(rec))
        if args.handler == "tiny":
            ok = ok and len(records) == 2
            by_party = {r["extra"].get("party"): r for r in records}
            ok = ok and set(by_party) == {0, 1}
            for rec in records:
                e = rec["extra"]
                ok = ok and e.get("completed") is True and rec["aborts"] == 0 and e.get("failure", "") == ""
                ok = ok and e.get("language") == "cpp" and rec["dealer_mode"] == ISOLATED and rec["record_kind"] == "web"
                ok = ok and rec["code_commit"].startswith("cpp-tree-") and "seed" not in rec and "seed" not in e
                ok = ok and e.get("party_rng") == "os_csprng"
            if len(records) == 2:
                ok = ok and records[0]["bytes"] == records[1]["bytes"] and records[0]["rounds"] == records[1]["rounds"]
                ok = ok and records[0]["trace_hash"] == records[1]["trace_hash"]
            p1 = by_party.get(1, {})
            p1_tokens = p1.get("extra", {}).get("tokens_list", [])
            ok = ok and p1.get("tokens") == args.n_tokens and len(p1_tokens) == args.n_tokens
            if args.verify_in_sidecar:
                # the sidecar saw the weights: output_match is set and the record says DIAGNOSTIC
                ok = ok and p1.get("output_match") is True and p1.get("security_eligible") is False and p1.get("status") == "DIAGNOSTIC"
                ok = ok and by_party.get(0, {}).get("security_eligible") is False  # the model came from a public seed
            else:
                launcher_match = p1_tokens == reference
                print("[web] launcher_output_match=%s reference=%s released=%s" % (launcher_match, reference, p1_tokens))
                ok = ok and launcher_match
                for rec in records:  # deployment shape: private weights file, OS randomness, isolated dealer
                    if not (rec["security_eligible"] is True and rec["status"] == "MEASURED_NATIVE"):
                        print("[web] NOT ELIGIBLE:", rec["extra"].get("eligibility_notes"))
                        ok = False
        else:
            ok = ok and mine.get("tokens_emitted") == args.n_tokens

        if args.handler == "tiny":
            ok = web_cancel_then_warm(args, base, token, out_path) and ok
        print("[web] job %.2f s, total %.2f s, %s" % (t_done, time.time() - t_start, "OK" if ok else "FAILED"))
        return ok
    finally:
        stop_all(procs, "web")


def wait_job(base: str, token: str, job_id: str, until, timeout_s: float = 300.0) -> tuple:
    """Poll GET /v1/status until until(job) is true; returns (job, full status)."""
    t0 = time.time()
    while True:
        _, status = http("GET", base + "/v1/status", token)
        jobs = [j for j in status.get("jobs", []) if j.get("job_id") == job_id]
        if jobs and until(jobs[0]):
            return jobs[0], status
        if time.time() - t0 > timeout_s:
            raise SystemExit("job %s did not reach the expected state within %.0f s" % (job_id, timeout_s))
        time.sleep(0.05)


def web_cancel_then_warm(args, base: str, token: str, out_path: str) -> bool:
    """Second and third job on the same launch: a cancelled generation (cancel waste, declared
    cancellation inside the session loop) and a warm one (warm TTFT; the server still serves)."""
    ok = True
    n_before = count_lines(out_path)
    st, job = http("POST", base + "/v1/generate", token, {"prompt": args.prompt, "n_tokens": 8})
    assert st == 202, "second generate not accepted"
    wait_job(base, token, job["job_id"], lambda j: j.get("tokens_emitted", 0) >= 1 or j.get("state") not in ("queued", "running"))
    st, _ = http("POST", base + "/v1/cancel", token, {"job_id": job["job_id"]})
    ok = ok and st == 200
    cancelled, status = wait_job(base, token, job["job_id"], lambda j: j.get("state") not in ("queued", "running"))
    print("[web] cancelled job: state=%s tokens=%s cancel=%s metrics=%s" % (
        cancelled.get("state"), cancelled.get("tokens_emitted"), json.dumps(cancelled.get("cancel")), json.dumps(cancelled.get("metrics"))))
    ok = ok and cancelled.get("state") == "cancelled" and 1 <= cancelled.get("tokens_emitted", 0) < 8
    ok = ok and cancelled.get("completed") is False and cancelled["cancel"]["requested"] is True and cancelled["cancel"]["latency_s"] >= 0
    m = status.get("metrics", {})
    waste = m.get("cancel_waste", {})
    ok = ok and waste.get("jobs") == 1 and waste.get("latency_s_max", -1) >= 0 and waste.get("wall_s", 0) > 0
    ok = ok and m.get("completion", {}).get("cancelled") == 1 and m["completion"]["completed"] == 1 and m["completion"]["rate"] == 0.5
    # the two records of the cancelled session: declared cancellation, never eligible, nothing left RESERVED
    time.sleep(0.5)
    recs = read_new_records(out_path, n_before)
    for rec in recs:
        print("[web] cancelled record", describe(rec), "cancelled=%s reserved_after_run=%s" % (
            rec["extra"].get("cancelled"), rec["extra"].get("reserved_after_run")))
    ok = ok and len(recs) == 2
    for rec in recs:
        e = rec["extra"]
        ok = ok and e.get("cancelled") is True and e.get("completed") is False and rec["security_eligible"] is False
        ok = ok and e.get("reserved_after_run", 0) == 0
    # a third job on the same launch completes: warm TTFT is measured, the cancel did not poison the server
    st, job3 = http("POST", base + "/v1/generate", token, {"prompt": args.prompt, "n_tokens": 2})
    assert st == 202, "third generate not accepted"
    done, status = wait_job(base, token, job3["job_id"], lambda j: j.get("state") not in ("queued", "running"))
    m = status.get("metrics", {})
    print("[web] warm job: state=%s ttft=%.3f s; ttft_s=%s" % (done.get("state"), done.get("ttft_s", -1), json.dumps(m.get("ttft_s"))))
    ok = ok and done.get("state") == "done" and done.get("cold") is False and done.get("ttft_s", -1) > 0
    ok = ok and m["ttft_s"]["warm_n"] >= 1 and m["ttft_s"]["warm_mean_s"] > 0 and m["ttft_s"]["cold_s"] > 0
    ok = ok and m["completion"]["completed"] == 2 and m["completion"]["cancelled"] == 1
    if not ok:
        print("[web] CANCEL / WARM CHECK FAILED")
    return ok


def stop_all(procs: list, tag: str) -> None:
    for pr in reversed(procs):
        code, err = pr.stop()
        with pr.lock:
            tail = [l for l in pr.lines if l.startswith("{") and '"session_record"' not in l and '"party_record"' not in l]
        print("[%s] stop %s exit=%s last=%s" % (tag, pr.name, code, tail[-1][:300] if tail else ""))
        if code not in (0, None, "terminated"):
            print("[%s] stderr %s: %s" % (tag, pr.name, err[-1500:]))


# ---------------------------------------------------------------------------
# modes party and cancel: dealer + two opencrab2_party processes
# ---------------------------------------------------------------------------
def run_party_pair(args, bindir: str, tmp: str, out_path: str, tag: str, extra: list, trace_path: str | None,
                   n_tokens: int) -> tuple:
    """Returns (ok, records by party). P0 gets the weights file, P1 the prompt and the reference tokens."""
    procs = []
    setup_id = "setup-" + tag
    prompt_ids = [3, 1, 4, 1]
    try:
        dealer, keys, dealer_addr, d = start_dealer(bindir, tmp, setup_id, "tiny-decoder-v1", tag + "_dealer")
        procs.append(dealer)
        weights, reference, model_cfg = make_model(bindir, tmp, args.model_seed + 1, prompt_ids, n_tokens, 64)
        common = ["--dealer", dealer_addr, "--keys", keys, "--setup-id", setup_id, "--out", out_path, "--journal-dir", tmp,
                  "--generate", str(n_tokens), "--prompt-len", str(len(prompt_ids)), "--policy", args.policy] + extra
        if trace_path:
            common += ["--trace", trace_path]
        n_before = count_lines(out_path)
        t0 = time.time()
        p0 = Proc(tag + "_p0", [exe(bindir, "opencrab2_party"), "--party", "0", "--listen", "127.0.0.1:0", "--weights", weights] + common,
                  REPO, tmp)
        procs.append(p0)
        port = p0.wait_ready("party0_listening")["port"]
        p1 = Proc(tag + "_p1", [exe(bindir, "opencrab2_party"), "--party", "1", "--peer", "127.0.0.1:%d" % port, "--prompt-ids",
                                ",".join(str(t) for t in prompt_ids), "--reference-tokens", ",".join(str(t) for t in reference)]
                  + common, REPO, tmp)
        procs.append(p1)
        c1 = p1.wait_exit(900.0)
        c0 = p0.wait_exit(120.0)
        wall = time.time() - t0
        ok = c0 == 0 and c1 == 0
        if not ok:
            print("[%s] exit codes p0=%s p1=%s\n%s\n%s" % (tag, c0, c1, p0.stderr_tail(600), p1.stderr_tail(600)))
        records = read_new_records(out_path, n_before)
        by_party = {r["extra"].get("party"): r for r in records}
        for rec in records:
            print("[%s] record" % tag, describe(rec))
        ok = ok and len(records) == 2 and set(by_party) == {0, 1}
        if ok:
            ok = ok and by_party[0]["bytes"] == by_party[1]["bytes"] and by_party[0]["rounds"] == by_party[1]["rounds"]
            ok = ok and by_party[1]["output_match"] is True and by_party[1]["extra"]["tokens_list"] == reference
            for rec in records:
                e = rec["extra"]
                ok = ok and e.get("language") == "cpp" and e.get("implementation") == "opencrab2_party"
                ok = ok and rec["record_kind"] == "session" and rec["dealer_mode"] == ISOLATED and rec["aborts"] == 0
                ok = ok and e.get("completed") is True and e.get("party_rng") == "os_csprng" and e.get("model_source") == "file"
                ok = ok and rec["code_commit"].startswith("cpp-tree-")
        print("[%s] wall %.2f s, %s" % (tag, wall, "OK" if ok else "FAILED"))
        return ok, by_party
    finally:
        stop_all(procs, tag)


def mode_party(args, bindir: str, tmp: str, out_path: str) -> bool:
    # a trace of the experiments catalog, written with the Python generator and loaded by both parties
    tr = traces.make_trace(args.trace_family, args.trace_variant, args.trace_seed, args.trace_duration)
    trace_path = str(traces.write_trace(tr, os.path.join(tmp, "traces")))
    print("[party] trace %s hash=%s points=%d" % (tr["trace_id"], tr["trace_hash"][:16], len(tr["points"])))
    ok, by_party = run_party_pair(args, bindir, tmp, out_path, "party", [], trace_path, args.n_tokens)
    if not ok:
        return False
    for rec in by_party.values():
        e = rec["extra"]
        # the record names the Python trace: same id, same points hash, the family columns of the schema
        ok = ok and rec["trace_id"] == tr["trace_id"] and rec["trace_hash"] == tr["trace_hash"] and rec["family"] == tr["family"]
        ok = ok and rec["variant"] == tr["variant"] and rec["split"] == tr["split"] and e.get("trace_hash_verified") is True
        ok = ok and e.get("shaping") is True and (e.get("shaping_delay_s") or 0.0) > 0.0  # the link was really shaped
        if not (rec["security_eligible"] is True and rec["status"] == "MEASURED_NATIVE"):
            print("[party] NOT ELIGIBLE:", e.get("eligibility_notes"))
            ok = False
    # the shaped link as the controller saw it: public telemetry per epoch (bandwidth, RTT)
    frames = [(ep.get("epoch"), ep.get("frame", {}).get("bandwidth_bps"), ep.get("frame", {}).get("rtt_s"))
              for ep in by_party[1]["extra"].get("epochs", [])]
    print("[party] telemetry (epoch, bandwidth_bps, rtt_s):", json.dumps(frames))
    min_rtt = 2.0 * min(p[2] for p in tr["points"])
    ok = ok and len(frames) == args.n_tokens and all(f[2] is not None and f[2] >= 0.8 * min_rtt for f in frames)
    print("[party] %s" % ("OK" if ok else "FAILED"))
    return ok


def mode_cancel(args, bindir: str, tmp: str, out_path: str) -> bool:
    n_tokens = max(3, args.n_tokens)
    ok, by_party = run_party_pair(args, bindir, tmp, out_path, "cancel", ["--cancel-after-epoch", "1", "--resume"], None, n_tokens)
    if not ok:
        return False
    for rec in by_party.values():
        e = rec["extra"]
        c = e.get("cancel") or {}
        ok = ok and c.get("cancelled") is True and c.get("cancel_after_epoch") == 1 and e.get("resumed") is True
        ok = ok and e.get("reserved_after_run", 0) == 0  # no RESERVED id survives the cancellation
    print("[cancel] cancel summary:", json.dumps(by_party[1]["extra"].get("cancel")), "OK" if ok else "FAILED")
    return ok


# ---------------------------------------------------------------------------
# mode bench: schema parity and quick benches in both dealer modes
# ---------------------------------------------------------------------------
def mode_bench(args, bindir: str, tmp: str, out_path: str) -> bool:
    ok = True
    code, out, err = run_tool([exe(bindir, "opencrab2_bench"), "--print-schema"])
    assert code == 0, "print-schema failed: " + err
    cpp = json.loads(out.strip().splitlines()[-1])
    pairs = [("record_fields", schema.RECORD_FIELDS), ("optional_fields", schema.OPTIONAL_FIELDS), ("statuses", schema.STATUSES),
             ("workload_scopes", schema.WORKLOAD_SCOPES), ("dealer_modes", schema.DEALER_MODES), ("clock_bases", schema.CLOCK_BASES),
             ("record_kinds", schema.RECORD_KINDS)]
    for name, py in pairs:
        same = list(cpp[name]) == list(py)
        if not same:
            print("[bench] SCHEMA DRIFT in %s: cpp=%s python=%s" % (name, cpp[name], list(py)))
        ok = ok and same
    ok = ok and cpp["schema"] == schema.SCHEMA_ID and cpp["profile_id"] == schema.PROFILE_ID
    ok = ok and cpp["default_contract_hash"] == schema.default_contract_hash()
    print("[bench] schema vocabulary equals schema.py: %s (code_commit %s)" % (ok, cpp["code_commit"]))

    code, out, err = run_tool([exe(bindir, "opencrab2_bench"), "--help"])
    ok = ok and code == 0 and "usage:" in out  # --help prints the usage and runs nothing

    procs = []
    try:
        dealer, keys, dealer_addr, d = start_dealer(bindir, tmp, "setup-bench", "bench-model", "bench_dealer")
        procs.append(dealer)
        n_before = count_lines(out_path)
        code, out, err = run_tool([exe(bindir, "opencrab2_bench"), "--out", out_path, "--batches", "64", "--repeats", "2",
                                   "--argmax-n", "64,256", "--argmax-k", "2,4,8", "--argmax-repeats", "1", "--dealer", dealer_addr,
                                   "--keys", keys, "--setup-id", "setup-bench"])
        print("[bench] isolated run exit=%s %s" % (code, err.strip().splitlines()[-1] if err.strip() else ""))
        ok = ok and code == 0
        recs = read_new_records(out_path, n_before)
        comps = sorted({r["extra"]["component"] for r in recs})
        print("[bench] isolated: %d records, components %s" % (len(recs), comps))
        need = {"dcf_eval", "drelu_p1", "drelu_p2", "trunc_faithful", "ring_to_field", "argmax_k2", "argmax_k4", "argmax_k8",
                "toy_bridge_local", "toy_bridge_h2a_ref", "wire_codec_roundtrip"}
        missing = need - set(comps)
        if missing:
            print("[bench] MISSING components:", sorted(missing))
        ok = ok and not missing
        for r in recs:
            e = r["extra"]
            good = (r["status"] == "MEASURED_NATIVE" and r["dealer_mode"] == ISOLATED and r["security_eligible"] is False
                    and e["language"] == "cpp" and r["workload_scope"] == "primitive" and r["record_kind"] == "primitive"
                    and r["output_match"] is not False and r["clock_basis"] == "process-perf-counter")
            if not good:
                print("[bench] BAD RECORD:", e.get("component"), r["status"], r["dealer_mode"], r["output_match"])
            ok = ok and good
        for r in recs:
            e = r["extra"]
            if e["component"].startswith("argmax_k"):
                print("[bench] %s n=%d rounds=%d bytes=%d material=%d online=%.4f s match=%s" % (
                    e["component"], e["params"]["n"], r["rounds"], r["bytes"], r["material_consumed"], r["T_online_s"], r["output_match"]))
        # the same k-ary argmax releases the same index for every k: rounds shrink with k
        by = {(r["extra"]["component"], r["extra"]["params"]["n"]): r for r in recs if r["extra"]["component"].startswith("argmax_k")}
        if ("argmax_k2", 256) in by and ("argmax_k8", 256) in by:
            ok = ok and by[("argmax_k8", 256)]["rounds"] < by[("argmax_k2", 256)]["rounds"]
    finally:
        stop_all(procs, "bench")

    n_before = count_lines(out_path)
    code, out, err = run_tool([exe(bindir, "opencrab2_bench"), "--out", out_path, "--batches", "64", "--repeats", "1", "--only",
                               "drelu,trunc,argmax", "--argmax-n", "64", "--argmax-k", "4", "--argmax-repeats", "1"])
    ok = ok and code == 0
    recs = read_new_records(out_path, n_before)
    for r in recs:  # shared seed: DIAGNOSTIC, never eligible (the dealer did not run isolated)
        good = r["status"] == "DIAGNOSTIC" and r["dealer_mode"] == SHARED_SEED and r["security_eligible"] is False
        if not good:
            print("[bench] BAD SHARED-SEED RECORD:", r["extra"].get("component"), r["status"], r["dealer_mode"])
        ok = ok and good
    ok = ok and len(recs) >= 5
    print("[bench] shared-seed run: %d records, all DIAGNOSTIC: %s" % (len(recs), ok))
    print("[bench] %s" % ("OK" if ok else "FAILED"))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", default=os.path.join(REPO, "cpp", "build", "bin"))
    ap.add_argument("--mode", default="all", choices=["all", "web", "party", "cancel", "bench"])
    ap.add_argument("--out", default=None, help="evidence JSONL (default: a temporary file, removed afterwards)")
    ap.add_argument("--n-tokens", type=int, default=3)
    ap.add_argument("--prompt", default="secure inference")
    ap.add_argument("--policy", default="B4_opencrab")
    ap.add_argument("--model-seed", type=int, default=2026)
    ap.add_argument("--handler", default="tiny", choices=["tiny", "demo"])
    ap.add_argument("--verify-in-sidecar", action="store_true", help="test mode: the sidecar loads the weights (DIAGNOSTIC record)")
    ap.add_argument("--web-trace", default=None, help="trace file for the web mode (default: the server's stable link)")
    ap.add_argument("--trace-family", default="S1")
    ap.add_argument("--trace-variant", default="ramp-LAN-to-WAN-c")
    ap.add_argument("--trace-seed", type=int, default=2)
    ap.add_argument("--trace-duration", type=float, default=6.0)
    ap.add_argument("--keep", action="store_true", help="keep the temporary directory")
    args = ap.parse_args()

    bindir = os.path.abspath(args.bin)
    tmp = tempfile.mkdtemp(prefix="oc2_smoke_")
    out_path = os.path.abspath(args.out) if args.out else os.path.join(tmp, "evidence.jsonl")
    if args.out:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    modes = ["web", "party", "cancel", "bench"] if args.mode == "all" else [args.mode]
    results = {}
    t0 = time.time()
    try:
        for mode in modes:
            fn = {"web": mode_web, "party": mode_party, "cancel": mode_cancel, "bench": mode_bench}[mode]
            try:
                results[mode] = bool(fn(args, bindir, tmp, out_path))
            except (AssertionError, ValueError, KeyError, OSError, SystemExit) as exc:
                print("[%s] FAILED: %s: %s" % (mode, type(exc).__name__, exc))
                results[mode] = False
        n_lines = count_lines(out_path)
        print("[smoke] SUMMARY: %s, %d evidence lines validated against schema.py, %.1f s, out=%s" % (
            " ".join("%s=%s" % (m, "OK" if results[m] else "FAILED") for m in modes), n_lines, time.time() - t0,
            out_path if args.out else "(temporary)"))
        return 0 if all(results.values()) else 1
    finally:
        if not args.keep:
            for root, dirs, files in os.walk(tmp, topdown=False):
                for f in files:
                    try:
                        os.remove(os.path.join(root, f))
                    except OSError:
                        pass
                for dname in dirs:
                    try:
                        os.rmdir(os.path.join(root, dname))
                    except OSError:
                        pass
            try:
                os.rmdir(tmp)
            except OSError:
                pass
        else:
            print("[smoke] kept", tmp)


if __name__ == "__main__":
    sys.exit(main())
