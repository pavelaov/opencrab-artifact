"""Web driver: dealer, model server and sidecar as processes; TTFT, inter-token latency, cancel, resume, rejections.

Starts ``opencrab2.platform.dealer_service`` (isolated dealer),
``opencrab2.platform.model_server`` (P0, aiohttp WebSocket ``/session``) and
``opencrab2.platform.sidecar`` (P1, local HTTP on 127.0.0.1 with an Origin
allow-list and a per-launch bearer token) with ``subprocess`` and measures,
through the sidecar's public HTTP API only (``POST /v1/generate``,
``GET /v1/stream``, ``POST /v1/cancel``, ``POST /v1/resume``,
``GET /v1/status``), the metrics of the Web service (paper, Section 7.4, Web service):

* ``cold_warm``: two consecutive generations on one freshly started stack.
  TTFT (time to the first released token) of the first run is the cold
  figure. The model server and the sidecar keep the installed objects, the
  shared vectors and the ledger per ``(model_version, client_key_domain)``,
  so the second generation is a real warm request: its TTFT is reported as
  ``ttft_warm_s`` and its measured wall time as ``T_warm_s`` only when the
  runner's own ``done`` event says ``warm_start`` and its epoch events show
  no installation; otherwise the value is kept as ``ttft_second_s`` with a
  note and no warm figure is reported. Inter-token latency (mean, p95) and
  completion (released tokens divided by the declared budget) come from the
  warm run; ``output_match`` compares both runs' tokens with the float
  reference (erf-GELU) that this driver computes from the public test model;
  ``first_first_mismatch_step`` / ``second_first_mismatch_step`` name the
  first differing step (None on a match) and ``..._margin`` the reference
  margin of that step. For both generations the record keeps the path the
  request took at its first epoch (``first_epoch0_*`` and ``second_epoch0_*``:
  the plan, whether the controller switched, the decision reason, the
  material wait, the prepare, install and execution seconds and the sensed
  bandwidth), because a warm request that switches to the generic plan for
  its prefill and back costs more than one that keeps the installed masks.
* ``cancel``: ``/v1/cancel`` after the first token; the safe material waste
  is the ``material_burned`` figure of the session's ``done`` event (the
  ledger's BURNED bytes) together with the ledger counts (no RESERVED entry
  may remain) and the cancel latency (request to the epoch boundary).
* ``reconnect``: the client's declared disconnect (``/v1/cancel``, which closes
  the WebSocket) followed by ``/v1/resume`` on the same ``session_id``: a new
  WebSocket, RESUME handshake, warm resume of the runner on both sides. The
  recovery cost is the time from the resume request to the next released
  token; completion is the total of both parts over the budget.
* ``server_restart`` (not in the default set): the model server process is
  killed and restarted mid-generation, then ``/v1/resume`` is attempted. The
  current server keeps sessions in memory, so the expected outcome is a
  ``SESSION_UNKNOWN`` rejection; the record carries the code and whether the
  session recovered. This documents the design limit, it does not hide it.
* ``rejections``: for each kind ``trials`` probes whose wrong acceptance is a
  failure (the table reports rejected/attempted with the one-sided
  zero-failure bound): a request from a disallowed Origin to the sidecar
  (403 ``ORIGIN_REJECTED``), a stale bearer token (401 ``TOKEN_REJECTED``), a
  WebSocket HELLO with a stale ``model_version`` to the model server
  (``REJECT OBJECT_VERSION_MISMATCH``) and a RESUME of an unknown session
  (``REJECT SESSION_UNKNOWN``). The sidecar and server counters are read
  afterwards as a cross-check.

Every record is ``MEASURED_NATIVE`` with ``workload_scope = web`` and
``dealer_mode = ISOLATED_RECIPIENT_BUNDLES``; ``security_eligible`` is taken
from the session runner's own ``done`` event (never asserted by this driver)
and only when the scenario completed. The sidecar never forwards plaintext
prompts and this driver never reads payload bytes from the server logs. The
driver checks for the platform modules at start and fails closed with
``WEB_PLATFORM_MISSING`` (``--dry-run`` prints the planned command lines).

Process order. The model server starts first (it derives ``model_version``
from the weights it loads); the driver reads that version from the server's
``GET /v1/status`` and starts the dealer with the same declared version on a
pre-chosen loopback port, then the sidecar. All timestamps are wall-clock
(``time.time``) because the sidecar stamps every streamed event with its own
wall time ``t``, which the driver uses for TTFT and inter-token gaps so that
the driver's stream-connection latency is excluded.

Thread cap. The three processes share one host. Every child is started with
``OMP_NUM_THREADS``, ``OPENBLAS_NUM_THREADS`` and ``MKL_NUM_THREADS`` set to
``--threads-per-process`` (default 2, as in ``run_sessions``), through the
same variables the session launcher sets (``platform.launch.THREAD_ENV``).
Without the cap each party opens a BLAS pool of every logical processor, and
when both parties run the same large product at the same instant (the
16-row embedding product of a warm prefill under the weight-masked plan) the
two pools stall each other: that product took tens of seconds instead of
about one second in the first measured campaign. Every record carries
``extra.threads_per_process`` and ``extra.colocated_processes`` (3), so that
timings of different configurations are never pooled.

Workloads. The tiny byte-vocabulary model (``experiments.workloads``) is
always measured (label ``tiny``); with ``--model gpt2`` the GPT-2 1-layer and
4-layer prefixes are measured as well (labels ``gpt2-1L`` and ``gpt2-4L``,
the pinned 16-token prompt). Records are resumable: the run_id of a
(workload, scenario, repeat) is deterministic and a rerun skips what the
output file already holds (``--redo`` ignores it).

Usage::

    python -m opencrab2.experiments.run_web [--model tiny|gpt2] [--n-generate 4] [--repeats 3] [--trials 10]
        [--scenarios cold_warm,cancel,reconnect,rejections] [--trace S0:stable-WAN] [--threads-per-process 2]
        [--dry-run] [--redo] [--out-root DIR]

What is tested (tests/test_run_web.py and tests/test_make_tables.py): the
command lines are accepted by the platform parsers, the child environment
carries the thread cap, the first-epoch fields and the mismatch fields are
derived from the sidecar's events, records carry the configuration and
relative paths, and the web table wraps statuses correctly; the measured
path needs the platform processes (tiny model, one stack). Design limit:
loopback without shaping unless ``--trace`` names a trace family:variant,
whose JSON is handed to the server and the sidecar through ``--trace-json``;
timings are from one host.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from opencrab2.experiments import schema, traces, workloads
from opencrab2.highlevel.contract import DEFAULT_CONTRACT

PYTHON = sys.executable
PROMPT_IDS: Tuple[int, ...] = (1026, 373, 262, 1266, 286, 1661, 11, 340, 373, 262, 5290, 286, 1661, 11, 340, 373)
START_TIMEOUT_S = 240.0
STREAM_TIMEOUT_S = 3600.0
STATUS_POLL_S = 0.25
SCENARIOS: Tuple[str, ...] = ("cold_warm", "cancel", "reconnect", "rejections", "server_restart")
DEFAULT_SCENARIOS: Tuple[str, ...] = ("cold_warm", "cancel", "reconnect", "rejections")
REJECTION_KINDS: Tuple[str, ...] = ("origin", "token", "version", "stale")
REQUIRED_MODULES: Tuple[str, ...] = ("opencrab2.platform.dealer_service", "opencrab2.platform.model_server",
                                     "opencrab2.platform.sidecar")
DEFAULT_POLICY = "B4_opencrab"
DEFAULT_THREADS_PER_PROCESS = 2
COLOCATED_PROCESSES = 3          # dealer, model server and sidecar on one host
WORKLOAD_LABELS: Tuple[str, ...] = ("tiny", "gpt2-1L", "gpt2-4L")
DONE_KEYS: Tuple[str, ...] = ("T_cold_s", "T_warm_s", "T_online_s", "bytes", "rounds", "material_burned", "security_eligible",
                              "epochs_done", "ledger_counts", "warm_start", "T_warm_basis", "session_wall_s",
                              "runner_ttft_s", "install_s", "setup_s", "wasted_bytes", "burned_on_cancel",
                              "material_consumed", "peak_rss_client", "peak_rss_server")
# fields of the sidecar's epoch event kept for the first epoch of each generation of cold_warm
EPOCH0_KEYS: Tuple[str, ...] = ("plan", "switched", "decision_reason", "material_wait_s", "prepare_s", "install_s",
                                "exec_s", "sensed_bw_bps", "installed_now")
# the driver's loopback HTTP calls never go through a proxy (urlopen would send them to http_proxy when no_proxy
# does not list 127.0.0.1)
_LOOPBACK = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _pydir() -> str:
    return str(Path(__file__).resolve().parents[2])


def _env(threads_per_process: Optional[int] = None) -> dict:
    """Environment of a child process: the package on the path, offline weights, the BLAS thread cap
    (``platform.launch.thread_env``, the same variables the session launcher sets)."""
    from opencrab2.platform.launch import thread_env
    env = dict(os.environ)
    env["PYTHONPATH"] = _pydir() + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("HF_HUB_OFFLINE", "1")
    if threads_per_process is not None and int(threads_per_process) < 1:
        raise WebError("WEB_PARAMS: threads_per_process must be >= 1")
    env.update(thread_env(threads_per_process))
    return env


def _write_text(path: Path, text: str) -> None:
    """Write a small text file with LF line endings on every platform."""
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def platform_available() -> Tuple[bool, str]:
    missing = [m for m in REQUIRED_MODULES if importlib.util.find_spec(m) is None]
    if missing:
        return False, f"WEB_PLATFORM_MISSING: modules not found: {missing}"
    return True, "OK"


class WebError(RuntimeError):
    """Driver-side failure (stable code in the message)."""


# ---------------------------------------------------------------------------
# process management
# ---------------------------------------------------------------------------
class Stack:
    """The three platform processes with their ports, keys, token and logs.

    ``commands()`` returns the exact command lines of the platform CLIs
    (``--dealer-key-file`` hex files, ``--token-file``, ``--trace-json``,
    ``--records``); ports that are not known yet are the placeholders
    ``{dealer_port}`` and ``{server_port}``.
    """

    def __init__(self, work: Path, model: str, n_layers: int, trace: Optional[dict] = None,
                 model_version: Optional[str] = None, tokenizer: str = "bytes", length_bucket: int = 32,
                 ttl_s: float = 600.0, threads_per_process: Optional[int] = DEFAULT_THREADS_PER_PROCESS):
        self.work = Path(work)
        self.model = str(model)
        self.n_layers = int(n_layers)
        self.trace = trace
        self.model_version: Optional[str] = model_version
        self.tokenizer = str(tokenizer)
        self.length_bucket = int(length_bucket)
        self.ttl_s = float(ttl_s)
        self.threads_per_process = int(threads_per_process) if threads_per_process is not None else None
        self.procs: Dict[str, subprocess.Popen] = {}
        self.keys = {0: secrets.token_bytes(32), 1: secrets.token_bytes(32)}
        self.token: Optional[str] = None
        self.dealer_port: Optional[int] = None
        self.server_port: Optional[int] = None
        self.http_port: Optional[int] = None
        self._drains: List[threading.Thread] = []

    # -- command lines -----------------------------------------------------------------------
    @property
    def origin(self) -> str:
        return f"http://127.0.0.1:{self.http_port if self.http_port is not None else '{http_port}'}"

    def commands(self) -> Dict[str, List[str]]:
        w = self.work
        dealer_port = str(self.dealer_port) if self.dealer_port is not None else "{dealer_port}"
        server_port = str(self.server_port) if self.server_port is not None else "{server_port}"
        version = self.model_version if self.model_version is not None else "{model_version}"
        trace_args = ["--trace-json", str(w / "trace.json")] if self.trace is not None else []
        cmds = {
            "dealer": [PYTHON, "-m", "opencrab2.platform.dealer_service", "--listen", f"127.0.0.1:{dealer_port}",
                       "--keys", str(w / "keys.json"), "--contract", DEFAULT_CONTRACT.name, "--setup-id", "web",
                       "--model-version", version],
            "server": [PYTHON, "-m", "opencrab2.platform.model_server", "--ws", f"127.0.0.1:{server_port}",
                       "--dealer", f"127.0.0.1:{dealer_port}", "--dealer-key-file", str(w / "key0.hex"),
                       "--model", self.model, "--n-layers", str(self.n_layers), "--contract", DEFAULT_CONTRACT.name,
                       "--state-dir", str(w / "server"), "--ttl-s", str(self.ttl_s),
                       "--records", str(w / "server-records.jsonl")] + trace_args,
            "sidecar": [PYTHON, "-m", "opencrab2.platform.sidecar", "--http", "127.0.0.1:0",
                        "--server", f"ws://127.0.0.1:{server_port}/session", "--dealer", f"127.0.0.1:{dealer_port}",
                        "--dealer-key-file", str(w / "key1.hex"), "--n-layers", str(self.n_layers),
                        "--contract", DEFAULT_CONTRACT.name, "--tokenizer", self.tokenizer,
                        "--state-dir", str(w / "sidecar"), "--token-file", str(w / "token.txt"),
                        "--records", str(w / "client-records.jsonl"), "--length-bucket", str(self.length_bucket)]
                       + trace_args,
        }
        if self.model_version is not None:
            cmds["server"] += ["--model-version", self.model_version]
            cmds["sidecar"] += ["--model-version", self.model_version]
        return cmds

    # -- spawning ----------------------------------------------------------------------------
    def env(self) -> dict:
        """The environment every child of this stack starts with (``_env`` with the stack's thread cap)."""
        return _env(self.threads_per_process)

    def _spawn(self, name: str, cmd: List[str], want: Sequence[str], timeout_s: float = START_TIMEOUT_S) -> Dict[str, str]:
        """Start a service, wait for its announcement lines on stdout, then drain stdout into its log."""
        log_path = self.work / f"{name}.log"
        log = open(log_path, "a", encoding="utf-8", newline="\n")
        proc = subprocess.Popen(cmd, cwd=_pydir(), env=self.env(), stdout=subprocess.PIPE, stderr=log, text=True)
        self.procs[name] = proc
        got: Dict[str, str] = {}
        deadline = time.time() + timeout_s
        while time.time() < deadline and not all(k in got for k in want):
            line = proc.stdout.readline() if proc.stdout else ""
            if not line:
                if proc.poll() is not None:
                    break
                continue
            tok = line.split(maxsplit=1)
            if tok and tok[0] in want:
                got[tok[0]] = tok[1].strip() if len(tok) > 1 else ""
        if not all(k in got for k in want):
            proc.kill()
            log.close()
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-1500:]
            raise WebError(f"WEB_PROCESS_START: {name} did not announce {list(want)} within {timeout_s} s; log tail: {tail!r}")

        def drain() -> None:
            try:
                for ln in proc.stdout:          # type: ignore[union-attr]
                    log.write(ln)
                    log.flush()
            except Exception:
                pass

        t = threading.Thread(target=drain, name=f"drain-{name}", daemon=True)
        t.start()
        self._drains.append(t)
        return got

    def start(self) -> None:
        from opencrab2.platform.dealer_service import write_keys, write_private_text
        self.work.mkdir(parents=True, exist_ok=True)
        write_keys(str(self.work / "keys.json"), self.keys)              # the key files: mode 0600 on POSIX
        write_private_text(str(self.work / "key0.hex"), self.keys[0].hex())
        write_private_text(str(self.work / "key1.hex"), self.keys[1].hex())
        if self.trace is not None:
            _write_text(self.work / "trace.json", json.dumps(self.trace))
        # the dealer's port is chosen ahead so that the server can be started first
        self.dealer_port = free_port()
        self.server_port = 0
        got = self._spawn("server", self.commands()["server"], ["LISTENING"])
        self.server_port = int(got["LISTENING"].split()[1])
        st = self.wait_server_status()
        version = str(st.get("model_version", ""))
        if self.model_version is None:
            if not version:
                raise WebError("WEB_SERVER_VERSION: the model server reported no model_version")
            self.model_version = version
        elif version != self.model_version:
            raise WebError(f"WEB_SERVER_VERSION: server serves {version!r}, driver declared {self.model_version!r}")
        got = self._spawn("dealer", self.commands()["dealer"], ["LISTENING"])
        self.dealer_port = int(got["LISTENING"].split()[1])
        got = self._spawn("sidecar", self.commands()["sidecar"], ["LISTENING", "TOKEN"])
        self.http_port = int(got["LISTENING"].split()[1])
        self.token = got["TOKEN"].strip()
        self.wait_ready()

    def restart_server(self) -> float:
        """Kill and restart the model server on the same port (disconnect injection); returns the restart seconds."""
        t0 = time.time()
        p = self.procs.get("server")
        if p is not None:
            p.kill()
            p.wait(timeout=30)
        self._spawn("server", self.commands()["server"], ["LISTENING"])
        self.wait_server_status()
        return time.time() - t0

    def stop(self) -> None:
        for name in ("sidecar", "server", "dealer"):
            p = self.procs.get(name)
            if p is not None and p.poll() is None:
                p.kill()
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass

    # -- HTTP -------------------------------------------------------------------------------
    def _check_alive(self) -> None:
        for name, p in self.procs.items():
            if p.poll() is not None and name != "server_restart":
                raise WebError(f"WEB_PROCESS_EXIT: {name} exited with {p.returncode}; see {self.work / (name + '.log')}")

    def wait_server_status(self, timeout_s: float = START_TIMEOUT_S) -> dict:
        deadline = time.time() + timeout_s
        last = ""
        while time.time() < deadline:
            try:
                return self.server_status()
            except (urllib.error.URLError, ConnectionError, OSError, ValueError) as exc:
                last = f"{exc.__class__.__name__}: {exc}"
            p = self.procs.get("server")
            if p is not None and p.poll() is not None:
                raise WebError(f"WEB_PROCESS_EXIT: server exited with {p.returncode}; see {self.work / 'server.log'}")
            time.sleep(STATUS_POLL_S)
        raise WebError(f"WEB_STACK_TIMEOUT: server status not reachable after {timeout_s} s ({last})")

    def wait_ready(self, timeout_s: float = START_TIMEOUT_S) -> dict:
        deadline = time.time() + timeout_s
        last = ""
        while time.time() < deadline:
            self._check_alive()
            try:
                st = self.http("GET", "/v1/status")
                if isinstance(st, dict):
                    return st
            except (urllib.error.URLError, ConnectionError, OSError, ValueError) as exc:
                last = f"{exc.__class__.__name__}: {exc}"
            time.sleep(STATUS_POLL_S)
        raise WebError(f"WEB_STACK_TIMEOUT: sidecar not ready after {timeout_s} s ({last})")

    def http(self, method: str, path: str, body: Optional[dict] = None, origin: Optional[str] = None,
             token: Optional[str] = None, timeout: float = 30.0) -> Any:
        """One sidecar request with the bearer token and the own Origin unless overridden."""
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(f"http://127.0.0.1:{self.http_port}{path}", data=data, method=method)
        req.add_header("Content-Type", "application/json")
        req.add_header("Origin", origin if origin is not None else self.origin)
        req.add_header("Authorization", f"Bearer {token if token is not None else self.token}")
        with _LOOPBACK.open(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}

    def server_status(self) -> dict:
        req = urllib.request.Request(f"http://127.0.0.1:{self.server_port}/v1/status", method="GET")
        with _LOOPBACK.open(req, timeout=10.0) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def session_status(self, session_id: str) -> dict:
        st = self.http("GET", "/v1/status")
        return dict(st.get("sessions", {}).get(session_id, {}))

    def wait_session(self, session_id: str, statuses: Sequence[str], timeout_s: float = 600.0,
                     on_server: bool = False) -> dict:
        """Poll the sidecar (or the server) until the session's status is one of ``statuses``."""
        deadline = time.time() + timeout_s
        last: dict = {}
        while time.time() < deadline:
            if on_server:
                last = dict(self.server_status().get("sessions", {}).get(session_id, {}))
            else:
                last = self.session_status(session_id)
            if last.get("status") in statuses:
                return last
            time.sleep(STATUS_POLL_S)
        raise WebError(f"WEB_SESSION_WAIT: session {session_id} did not reach {list(statuses)} "
                       f"(last {last.get('status')!r}, error {last.get('error')!r})")

    def stream(self, session_id: str, since: int = 0, stop_after_tokens: Optional[int] = None,
               timeout_s: float = STREAM_TIMEOUT_S) -> List[dict]:
        """Consume the SSE feed from event index ``since``; returns the events (``_type``, ``_recv`` added).

        Stops at the ``end`` marker (after ``done`` or ``error``) or once
        ``stop_after_tokens`` token events were seen (the sidecar keeps the
        history, so a later call with ``since`` continues the same session).
        """
        req = urllib.request.Request(f"http://127.0.0.1:{self.http_port}/v1/stream?session_id={session_id}&since={since}")
        req.add_header("Origin", self.origin)
        req.add_header("Authorization", f"Bearer {self.token}")
        events: List[dict] = []
        n_tokens = 0
        deadline = time.time() + timeout_s
        with _LOOPBACK.open(req, timeout=120.0) as resp:
            typ, data = "message", None
            while time.time() < deadline:
                raw = resp.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if line.startswith(":"):
                    continue
                if line == "":
                    if typ == "end":
                        return events
                    if data is not None:
                        try:
                            ev = json.loads(data)
                        except json.JSONDecodeError:
                            ev = {"raw": data}
                        if isinstance(ev, dict):
                            ev["_type"] = typ
                            ev["_recv"] = time.time()
                            events.append(ev)
                            if typ == "token":
                                n_tokens += 1
                                if stop_after_tokens is not None and n_tokens >= stop_after_tokens:
                                    return events
                    typ, data = "message", None
                    continue
                if line.startswith("event:"):
                    typ = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    data = line[len("data:"):].strip()
        return events


# ---------------------------------------------------------------------------
# metrics helpers
# ---------------------------------------------------------------------------
def _ev_time(ev: dict) -> float:
    t = ev.get("t")
    return float(t) if isinstance(t, (int, float)) else float(ev.get("_recv", float("nan")))


def token_metrics(t_request: float, events: Sequence[dict], n_generate: int) -> dict:
    """TTFT, inter-token gaps and completion from the sidecar's wall-clock event stamps."""
    toks = [e for e in events if e.get("_type") == "token"]
    done = next((e for e in events if e.get("_type") == "done"), None)
    err = next((e for e in events if e.get("_type") == "error"), None)
    out: Dict[str, Any] = {"tokens": len(toks), "completion": min(1.0, len(toks) / max(1, n_generate)),
                           "status": (done or {}).get("status"), "error_code": (err or {}).get("code")}
    if toks:
        times = [_ev_time(e) for e in toks]
        out["ttft_s"] = times[0] - t_request
        gaps = [b - a for a, b in zip(times, times[1:])]
        out["itl_s"] = gaps
        out["itl_mean_s"] = float(np.mean(gaps)) if gaps else None
        out["itl_p95_s"] = float(np.percentile(gaps, 95)) if gaps else None
        out["token_ids"] = [int(e.get("token_id", -1)) for e in toks]
    epochs = [e for e in events if e.get("_type") == "epoch"]
    out["install_s_total"] = float(sum(float(e.get("install_s") or 0.0) for e in epochs))
    out["installed_now"] = [o for e in epochs for o in (e.get("installed_now") or [])]
    out["switches"] = int(sum(1 for e in epochs if e.get("switched")))
    if done is not None:
        for k in DONE_KEYS:
            if k in done:
                out[f"done_{k}"] = done[k]
    return out


def _match(token_ids: Optional[Sequence[int]], reference: Optional[Sequence[int]]) -> Optional[bool]:
    """True/False when both the released tokens and the reference are known, else None."""
    if reference is None or token_ids is None:
        return None
    return [int(t) for t in token_ids] == [int(t) for t in reference][:len(token_ids)] and len(token_ids) == len(reference)


def _mismatch(token_ids: Optional[Sequence[int]], reference: Optional[Sequence[int]],
              margins: Optional[Sequence[float]]) -> Tuple[Optional[int], Optional[float]]:
    """(first differing step, reference margin at that step); (None, None) on a match or without a reference."""
    if reference is None or token_ids is None:
        return None, None
    from opencrab2.highlevel.session import first_mismatch_step
    step = first_mismatch_step([int(t) for t in token_ids], [int(t) for t in reference])
    if step is None:
        return None, None
    margin = float(margins[step]) if margins is not None and 0 <= step < len(margins) else None
    return int(step), margin


def epoch0_summary(events: Sequence[dict]) -> Dict[str, Any]:
    """The path a generation took at its first epoch: ``EPOCH0_KEYS`` of the sidecar's epoch-0 event plus
    ``plan_key`` (the plan as one string); every value None when the stream holds no epoch event."""
    ev = next((e for e in events if e.get("_type") == "epoch" and e.get("epoch") == 0), None)
    out: Dict[str, Any] = {k: (ev.get(k) if ev is not None else None) for k in EPOCH0_KEYS}
    plan = out.get("plan")
    out["plan_key"] = ("|".join(f"{k}={plan[k]}" for k in sorted(plan)) if isinstance(plan, dict) and plan else None)
    return out


def _wait_server_idle(stack: "Stack", session_id: str) -> None:
    """The next request is admitted only when the server left the running state for ``session_id``."""
    stack.wait_session(session_id, ("done", "cancelled", "failed", "disconnected", "expired"), on_server=True)


def _generate(stack: Stack, prompt_ids: Sequence[int], n_generate: int, policy: str) -> Tuple[str, float]:
    t0 = time.time()
    reply = stack.http("POST", "/v1/generate", {"prompt_ids": [int(v) for v in prompt_ids], "n_generate": int(n_generate),
                                                "policy": policy})
    sid = str(reply.get("session_id", ""))
    if not sid:
        raise WebError(f"WEB_GENERATE: no session_id in the sidecar reply {reply!r}")
    return sid, t0


# ---------------------------------------------------------------------------
# scenarios
# ---------------------------------------------------------------------------
def scenario_cold_warm(stack: Stack, prompt_ids: Sequence[int], n_generate: int, policy: str,
                       reference: Optional[Sequence[int]] = None,
                       reference_margins: Optional[Sequence[float]] = None) -> dict:
    out: Dict[str, Any] = {}
    runs = []
    for label in ("first", "second"):
        sid, t0 = _generate(stack, prompt_ids, n_generate, policy)
        events = stack.stream(sid)
        m = token_metrics(t0, events, n_generate)
        m["session_id"] = sid
        runs.append(m)
        out[f"ttft_{label}_s"] = m.get("ttft_s")
        out[f"{label}_tokens"] = m["tokens"]
        out[f"{label}_completion"] = m["completion"]
        out[f"{label}_install_s"] = m["install_s_total"]
        out[f"{label}_status"] = m.get("status")
        if m.get("error_code"):
            out[f"{label}_error"] = m["error_code"]
        # the first epoch decides the path of the request: kept masks, or a switch to the generic plan and back
        for k, v in epoch0_summary(events).items():
            out[f"{label}_epoch0_{k}"] = v
        step, margin = _mismatch(m.get("token_ids"), reference, reference_margins)
        out[f"{label}_first_mismatch_step"] = step
        out[f"{label}_first_mismatch_margin"] = margin
        if m.get("status") is not None:
            _wait_server_idle(stack, sid)
    first, second = runs
    out["ttft_cold_s"] = first.get("ttft_s")
    out["ttft_second_s"] = second.get("ttft_s")
    # verified by the runner's own flags: it started from the stack, installed nothing and spent no install time
    warm_verified = (bool(second.get("done_warm_start")) and float(second.get("done_install_s") or 0.0) == 0.0
                     and not second["installed_now"] and second["completion"] >= 1.0)
    out["warm_verified"] = warm_verified
    out["ttft_warm_s"] = second.get("ttft_s") if warm_verified else None
    if not warm_verified:
        out["warm_note"] = "second session was not a verified warm request; its TTFT is kept as ttft_second_s only"
    out["itl_mean_s"] = second.get("itl_mean_s")
    out["itl_p95_s"] = second.get("itl_p95_s")
    out["itl_s"] = second.get("itl_s")
    out["itl_cold_mean_s"] = first.get("itl_mean_s")
    out["completion"] = second["completion"]
    out["tokens"] = second["tokens"]
    out["tokens_identical"] = first.get("token_ids") == second.get("token_ids") if first.get("token_ids") else None
    out["output_match_first"] = _match(first.get("token_ids"), reference)
    out["output_match_second"] = _match(second.get("token_ids"), reference)
    if out["output_match_first"] is not None and out["output_match_second"] is not None:
        out["output_match"] = bool(out["output_match_first"] and out["output_match_second"])
    for k in DONE_KEYS:
        if f"done_{k}" in first:
            out[f"first_{k}"] = first[f"done_{k}"]
        if f"done_{k}" in second:
            out[f"second_{k}"] = second[f"done_{k}"]
    out["aborted"] = not (first["completion"] >= 1.0 and second["completion"] >= 1.0)
    return out


def scenario_cancel(stack: Stack, prompt_ids: Sequence[int], n_generate: int, policy: str,
                    reference: Optional[Sequence[int]] = None,
                    reference_margins: Optional[Sequence[float]] = None) -> dict:
    sid, t0 = _generate(stack, prompt_ids, n_generate, policy)
    first = stack.stream(sid, stop_after_tokens=1)
    t_cancel = time.time()
    reply = stack.http("POST", "/v1/cancel", {"session_id": sid})
    rest = stack.stream(sid, since=len(first))
    events = list(first) + list(rest)
    m = token_metrics(t0, events, n_generate)
    done = next((e for e in events if e.get("_type") == "done"), None)
    st = stack.session_status(sid)
    counts = dict((done or {}).get("ledger_counts") or st.get("ledger_counts") or {})
    out = {"session_id": sid, "tokens_before_cancel": len([e for e in first if e.get("_type") == "token"]),
           "tokens_total": m["tokens"], "tokens": m["tokens"], "cancel_reply": reply,
           "cancel_effective": m.get("status") == "cancelled",
           "cancel_latency_s": (_ev_time(done) - t_cancel) if done is not None else None,
           "cancel_waste_bytes": (done or {}).get("material_burned"),
           "cancel_wasted_buffered_bytes": (done or {}).get("wasted_bytes"),
           "burned_reserved_ids": (done or {}).get("burned_on_cancel"), "ledger_counts_after_cancel": counts,
           "reserved_after_cancel": int(counts.get("RESERVED", 0)) if counts else None,
           "status_after_cancel": st.get("status"), "completion": m["completion"], "ttft_s": m.get("ttft_s"),
           "aborted": False}
    for k in ("bytes", "rounds", "epochs_done"):
        if f"done_{k}" in m:
            out[k] = m[f"done_{k}"]
    return out


def scenario_reconnect(stack: Stack, prompt_ids: Sequence[int], n_generate: int, policy: str,
                       reference: Optional[Sequence[int]] = None,
                       reference_margins: Optional[Sequence[float]] = None) -> dict:
    sid, t0 = _generate(stack, prompt_ids, n_generate, policy)
    first = stack.stream(sid, stop_after_tokens=1)
    t_cancel = time.time()
    stack.http("POST", "/v1/cancel", {"session_id": sid})
    part1 = stack.stream(sid, since=len(first))
    events1 = list(first) + list(part1)
    m1 = token_metrics(t0, events1, n_generate)
    # both sides must have reached the epoch boundary before the RESUME handshake is admitted
    stack.wait_session(sid, ("cancelled", "disconnected"))
    stack.wait_session(sid, ("cancelled", "disconnected"), on_server=True)
    t_resume = time.time()
    reply = stack.http("POST", "/v1/resume", {"session_id": sid})
    part2 = stack.stream(sid, since=len(events1))
    m2 = token_metrics(t_resume, part2, n_generate)
    toks2 = [e for e in part2 if e.get("_type") == "token"]
    total_tokens = m1["tokens"] + len(toks2)
    done2 = next((e for e in part2 if e.get("_type") == "done"), None)
    all_ids = list(m1.get("token_ids") or []) + [int(e.get("token_id", -1)) for e in toks2]
    step, margin = _mismatch(all_ids, reference, reference_margins) if total_tokens >= n_generate else (None, None)
    out = {"session_id": sid, "output_match": _match(all_ids, reference) if total_tokens >= n_generate else None,
           "first_mismatch_step": step, "first_mismatch_margin": margin,
           "disconnect_mode": "declared cancel (WebSocket closed) then /v1/resume on the same session_id",
           "tokens_before_disconnect": m1["tokens"], "tokens_after_resume": len(toks2), "tokens": total_tokens,
           "completion": min(1.0, total_tokens / max(1, n_generate)), "resume_reply": reply,
           "resume_epoch": reply.get("resume_epoch"), "down_s": t_resume - t_cancel,
           "recovery_s": (_ev_time(toks2[0]) - t_resume) if toks2 else None,
           "resume_total_s": (_ev_time(done2) - t_resume) if done2 is not None else None,
           "reconnects": 1, "material_burned_at_disconnect": m1.get("done_material_burned"),
           "material_burned_final": m2.get("done_material_burned"), "status_final": m2.get("status"),
           "error_after_resume": m2.get("error_code"), "security_eligible_final": m2.get("done_security_eligible"),
           "aborted": total_tokens < n_generate}
    for k in ("bytes", "rounds", "T_cold_s", "T_online_s", "epochs_done", "ledger_counts"):
        if f"done_{k}" in m2:
            out[f"final_{k}"] = m2[f"done_{k}"]
    out["bytes"] = m2.get("done_bytes", 0)
    out["rounds"] = m2.get("done_rounds", 0)
    return out


def scenario_server_restart(stack: Stack, prompt_ids: Sequence[int], n_generate: int, policy: str,
                            reference: Optional[Sequence[int]] = None,
                            reference_margins: Optional[Sequence[float]] = None) -> dict:
    sid, t0 = _generate(stack, prompt_ids, n_generate, policy)
    first = stack.stream(sid, stop_after_tokens=1)
    t_kill = time.time()
    restart_s = stack.restart_server()
    part1 = stack.stream(sid, since=len(first))
    events1 = list(first) + list(part1)
    m1 = token_metrics(t0, events1, n_generate)
    st = stack.wait_session(sid, ("disconnected", "failed", "cancelled", "done"), timeout_s=300.0)
    out: Dict[str, Any] = {"session_id": sid, "disconnect_mode": "model server process killed and restarted on the same port",
                           "server_restart_s": restart_s, "tokens_before_disconnect": m1["tokens"],
                           "status_after_disconnect": st.get("status"), "error_after_disconnect": m1.get("error_code"),
                           "recovered": False, "recovery_code": None, "tokens": m1["tokens"],
                           "completion": m1["completion"]}
    if st.get("status") in ("disconnected", "cancelled"):
        t_resume = time.time()
        try:
            reply = stack.http("POST", "/v1/resume", {"session_id": sid})
            out["resume_reply"] = reply
            part2 = stack.stream(sid, since=len(events1), timeout_s=600.0)
            m2 = token_metrics(t_resume, part2, n_generate)
            toks2 = [e for e in part2 if e.get("_type") == "token"]
            out["tokens"] = m1["tokens"] + len(toks2)
            out["completion"] = min(1.0, out["tokens"] / max(1, n_generate))
            out["recovered"] = bool(toks2) and m2.get("status") == "done"
            out["recovery_code"] = m2.get("error_code") or ("OK" if out["recovered"] else m2.get("status"))
            out["recovery_s"] = (_ev_time(toks2[0]) - t_resume) if toks2 else None
        except urllib.error.HTTPError as exc:
            out["recovery_code"] = f"HTTP {exc.code}"
    out["aborted"] = not out["recovered"]
    out["down_s"] = time.time() - t_kill
    return out


async def _ws_probe_async(url: str, frame: dict, origin: str, timeout: float) -> Tuple[bool, str]:
    import aiohttp
    try:
        async with aiohttp.ClientSession() as s:
            async with s.ws_connect(url, headers={"Origin": origin}, timeout=timeout) as ws:
                await ws.send_json(frame)
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=timeout)
                except asyncio.TimeoutError:
                    return False, "no reply within the timeout (not rejected)"
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        obj = json.loads(msg.data)
                    except ValueError:
                        return False, "non-JSON reply accepted"
                    if isinstance(obj, dict) and obj.get("type") == "REJECT":
                        return True, str(obj.get("code"))
                    return False, f"accepted: {str(obj.get('type') if isinstance(obj, dict) else obj)[:60]}"
                return True, f"closed: {msg.type.name}"
    except Exception as exc:  # the handshake itself refused
        return True, f"handshake refused: {exc.__class__.__name__}"


def ws_probe(url: str, frame: dict, origin: str, timeout: float = 15.0) -> Tuple[bool, str]:
    """(rejected, detail): True when the server answers REJECT, closes, or refuses the handshake."""
    return asyncio.run(_ws_probe_async(url, frame, origin, timeout))


def scenario_rejections(stack: Stack, trials: int) -> dict:
    out: Dict[str, Any] = {}
    ws_url = f"ws://127.0.0.1:{stack.server_port}/session"
    contract_hash = DEFAULT_CONTRACT.contract_hash()
    probes: Dict[str, Any] = {}

    def http_probe(**kw: Any) -> Tuple[bool, str]:
        try:
            stack.http("GET", "/v1/status", None, timeout=10.0, **kw)
            return False, "accepted"
        except urllib.error.HTTPError as exc:
            return True, f"HTTP {exc.code}"
        except (urllib.error.URLError, OSError) as exc:
            return True, exc.__class__.__name__

    probes["origin"] = lambda i: http_probe(origin="http://evil.example")
    probes["token"] = lambda i: http_probe(token="deadbeef" * 6)
    probes["version"] = lambda i: ws_probe(ws_url, {
        "type": "HELLO", "session_id": f"probe-version-{i}-{secrets.token_hex(4)}", "model_version": "stale-version-0",
        "numeric_contract_hash": contract_hash, "profile_id": schema.PROFILE_ID, "nonce": secrets.token_hex(8),
        "length_bucket": 32, "n_generate": 1, "prompt_len": 1, "n_layers": stack.n_layers}, stack.origin)
    probes["stale"] = lambda i: ws_probe(ws_url, {
        "type": "RESUME", "session_id": f"expired-{secrets.token_hex(6)}", "model_version": stack.model_version,
        "numeric_contract_hash": contract_hash, "profile_id": schema.PROFILE_ID, "nonce": secrets.token_hex(8)}, stack.origin)
    for kind in REJECTION_KINDS:
        rej = acc = 0
        details: List[str] = []
        for i in range(int(trials)):
            r, d = probes[kind](i)
            rej += int(r)
            acc += int(not r)
            details.append(d)
        out.update({f"rejected_{kind}_trials": int(trials), f"rejected_{kind}_rejected": rej,
                    f"rejected_{kind}_accepted_wrongly": acc, f"rejected_{kind}_detail": sorted(set(details))})
    side = stack.http("GET", "/v1/status").get("counters", {})
    serv = stack.server_status().get("counters", {})
    out["sidecar_counters"] = {k: int(v) for k, v in side.items() if k.startswith("rejected")}
    out["server_counters"] = {k: int(v) for k, v in serv.items() if k.startswith("rejected")}
    out["aborted"] = False
    return out


# ---------------------------------------------------------------------------
# records and driver
# ---------------------------------------------------------------------------
def web_run_id(label: str, scenario: str, n_layers: int, n_generate: int, repeat: int) -> str:
    """Deterministic id of one (workload, scenario, repeat) cell (resumable campaigns)."""
    return f"web:{label}:{scenario}:L{int(n_layers)}:T{int(n_generate)}:r{int(repeat)}"


def web_record(scenario: str, repeat: int, metrics: dict, n_layers: int, n_generate: int, raw_path: str,
               trace: Optional[dict], model_version: str, policy: str, model_label: str = "",
               threads_per_process: Optional[int] = DEFAULT_THREADS_PER_PROCESS, root: Optional[Path] = None) -> dict:
    """One MEASURED_NATIVE web record; eligibility only as declared by the runner's done event on a completed run.

    ``T_cold_s`` is the first (cold) generation's wall time as measured by the
    session runner; ``T_warm_s`` is the measured wall time of the verified
    warm request (``second_T_warm_s``) and None when the second generation
    was not a verified warm request. Token ids never enter the record (the
    first mismatching step and its reference margin do). ``extra`` carries the
    configuration of the stack (``threads_per_process`` of every child,
    ``colocated_processes``); ``raw_path`` is stored relative to ``root``.
    """
    m = {k: v for k, v in dict(metrics).items() if not str(k).startswith("token_ids")}
    aborted = bool(m.get("aborted", False)) or bool(m.get("error"))
    eligible_flag = m.get("second_security_eligible", m.get("security_eligible_final"))
    eligible = bool(eligible_flag) and not aborted and scenario in ("cold_warm", "reconnect")
    t_cold = m.get("first_T_cold_s")
    t_warm = None
    if m.get("warm_verified"):
        t_warm = m.get("second_T_warm_s", m.get("second_T_cold_s"))
    t_online = m.get("first_T_online_s")
    bytes_ = m.get("first_bytes", m.get("bytes", 0)) or 0
    rounds = m.get("first_rounds", m.get("rounds", 0)) or 0
    burned = m.get("material_burned_final", m.get("cancel_waste_bytes", m.get("first_material_burned", 0))) or 0
    om = m.get("output_match")
    label = model_label or "unlabelled"
    rss_c = m.get("first_peak_rss_client")
    rss_s = m.get("first_peak_rss_server")
    return schema.new_record(
        status="MEASURED_NATIVE", security_eligible=eligible, record_kind="web", workload_scope="web",
        dealer_mode="ISOLATED_RECIPIENT_BUNDLES", model_revision=model_version,
        trace_id=trace["trace_id"] if trace else "loopback-unshaped", trace_hash=trace["trace_hash"] if trace else "",
        clock_basis="wall-clock-exogenous" if trace else "process-perf-counter", policy_id=policy,
        run_id=web_run_id(label, scenario, n_layers, n_generate, repeat), repeat=int(repeat),
        aborts=1 if aborted else 0, post_init_switches=int(m.get("switches", 0) or 0),
        T_cold_s=float(t_cold) if isinstance(t_cold, (int, float)) and t_cold >= 0 else None,
        T_warm_s=float(t_warm) if isinstance(t_warm, (int, float)) and t_warm >= 0 else None,
        T_online_s=float(t_online) if isinstance(t_online, (int, float)) and t_online >= 0 else None,
        bytes=int(bytes_), rounds=int(rounds), material_burned=int(burned),
        material_consumed=int(m.get("first_material_consumed", 0) or 0),
        peak_rss_client=int(rss_c) if isinstance(rss_c, int) and not isinstance(rss_c, bool) else None,
        peak_rss_server=int(rss_s) if isinstance(rss_s, int) and not isinstance(rss_s, bool) else None,
        output_match=om if isinstance(om, bool) else None,
        tokens=int(m.get("tokens", 0) or 0), raw_measurement_path=schema.relative_raw_path(raw_path, root),
        variant=scenario, n_layers=int(n_layers), T=int(n_generate),
        extra=dict(m, scenario=scenario, model_label=label,
                   threads_per_process=int(threads_per_process) if threads_per_process is not None else None,
                   colocated_processes=COLOCATED_PROCESSES))


def workload_specs(model: str, work: Path) -> List[dict]:
    """The measured workloads: the tiny model always; the GPT-2 1-layer and 4-layer prefixes with ``model == 'gpt2'``."""
    from opencrab2.highlevel.session import save_model_npz
    work.mkdir(parents=True, exist_ok=True)
    W, meta = workloads.make_tiny_weights(1)
    npz = save_model_npz(str(work / "tiny.npz"), W, meta)
    specs = [{"label": "tiny", "model": f"npz:{npz}", "n_layers": 1, "prompt_ids": list(workloads.TINY_PROMPT),
              "length_bucket": workloads.TINY_LENGTH_BUCKET, "weights": (W, meta)}]
    if model == "gpt2":
        for n in (1, 4):
            specs.append({"label": f"gpt2-{n}L", "model": "gpt2", "n_layers": n, "prompt_ids": list(PROMPT_IDS),
                          "length_bucket": workloads.GPT2_LENGTH_BUCKET, "weights": None})
    elif model != "tiny":
        raise WebError(f"WEB_MODEL_UNKNOWN: {model!r} (use 'tiny' or 'gpt2')")
    return specs


def _reference(spec: dict, n_generate: int) -> Tuple[Optional[List[int]], Optional[List[float]]]:
    """(float reference continuation, its top-1 minus top-2 margins) of a workload (driver side, public test
    input); (None, None) when it cannot be computed."""
    try:
        if spec.get("weights") is not None:
            W, meta = spec["weights"]
        else:
            from opencrab2.highlevel.session import load_model
            W, meta = load_model(spec["model"], spec["n_layers"])
        toks = workloads.reference_greedy_tokens(W, meta, spec["n_layers"], spec["prompt_ids"], n_generate)
        margins = workloads.reference_margins(W, meta, spec["n_layers"], spec["prompt_ids"], n_generate)
        return toks, margins
    except Exception:
        return None, None


def run(scenarios: Sequence[str], model: str, n_generate: int, repeats: int, out_root: Path,
        trace_spec: Optional[Tuple[str, str]], trials: int, policy: str, redo: bool = False,
        log: Any = None, threads_per_process: int = DEFAULT_THREADS_PER_PROCESS) -> dict:
    ok, msg = platform_available()
    if not ok:
        raise ImportError(msg)
    if int(threads_per_process) < 1:
        raise WebError("WEB_PARAMS: threads_per_process must be >= 1")
    raw_path = out_root / "raw" / "web.jsonl"
    work = out_root / "raw" / "web_work" / time.strftime("%Y%m%dT%H%M%S")
    trace = traces.make_trace(trace_spec[0], trace_spec[1], 2, 600.0) if trace_spec else None
    done_ids = set() if redo else schema.completed_run_ids(raw_path)
    summary: Dict[str, Any] = {"scenarios": {}, "work": str(work), "model": model, "skipped": 0, "written": 0,
                               "threads_per_process": int(threads_per_process),
                               "colocated_processes": COLOCATED_PROCESSES}
    fns = {"cold_warm": scenario_cold_warm, "cancel": scenario_cancel, "reconnect": scenario_reconnect,
           "server_restart": scenario_server_restart}
    out = log if log is not None else sys.stdout
    for spec in workload_specs(model, work):
        label, n_layers = spec["label"], int(spec["n_layers"])
        reference: Optional[List[int]] = None
        margins: Optional[List[float]] = None
        ref_done = False
        for scen in scenarios:
            results = []
            for rep_i in range(int(repeats)):
                rid = web_run_id(label, scen, n_layers, n_generate, rep_i)
                if rid in done_ids:
                    summary["skipped"] += 1
                    continue
                if not ref_done and scen != "rejections":
                    reference, margins = _reference(spec, n_generate)
                    ref_done = True
                stack = Stack(work / f"{label}_{scen}_r{rep_i}", spec["model"], n_layers, trace,
                              length_bucket=int(spec["length_bucket"]), threads_per_process=int(threads_per_process))
                m: Dict[str, Any]
                t0 = time.time()
                try:
                    stack.start()
                    if scen == "rejections":
                        m = scenario_rejections(stack, trials)
                    elif scen in fns:
                        m = fns[scen](stack, spec["prompt_ids"], n_generate, policy, reference, margins)
                    else:
                        raise WebError(f"WEB_SCENARIO_UNKNOWN: {scen!r}")
                except Exception as exc:  # the cell is reported by reason, never filled
                    m = {"error": f"{exc.__class__.__name__}: {exc}", "completion": 0.0, "aborted": True}
                finally:
                    stack.stop()
                m["driver_wall_s"] = time.time() - t0
                version = stack.model_version or f"{label}"
                rec = web_record(scen, rep_i, m, n_layers, n_generate, str(raw_path), trace, version, policy, label,
                                 threads_per_process=int(threads_per_process), root=out_root)
                schema.append_jsonl(raw_path, rec)
                summary["written"] += 1
                results.append({k: v for k, v in m.items() if not str(k).endswith("_detail") and not isinstance(v, (list, dict))})
                out.write(f"{rec['run_id']}: {'ABORTED' if rec['aborts'] else 'ok'} {m['driver_wall_s']:.1f} s\n")
                out.flush()
            summary["scenarios"][f"{label}:{scen}"] = results
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run_web", description="Web platform measurements")
    p.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS), help=f"subset of {','.join(SCENARIOS)}")
    p.add_argument("--model", default="tiny", choices=["tiny", "gpt2"],
                   help="'tiny' measures the tiny model only; 'gpt2' adds the GPT-2 1-layer and 4-layer prefixes")
    p.add_argument("--n-generate", type=int, default=4)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--trials", type=int, default=10, help="rejection probes per kind")
    p.add_argument("--policy", default=DEFAULT_POLICY)
    p.add_argument("--threads-per-process", type=int, default=DEFAULT_THREADS_PER_PROCESS,
                   help="OMP/OPENBLAS/MKL thread cap of the dealer, the model server and the sidecar")
    p.add_argument("--trace", default=None, help="family:variant for shaped runs (default: unshaped loopback)")
    p.add_argument("--redo", action="store_true", help="ignore run_ids already present in raw/web.jsonl")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out-root", default=None)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    out_root = Path(args.out_root) if args.out_root else schema.experiments_root()
    scen = [s for s in args.scenarios.split(",") if s]
    unknown = [s for s in scen if s not in SCENARIOS]
    if unknown:
        sys.stderr.write(f"WEB_SCENARIO_UNKNOWN: {unknown}; known {list(SCENARIOS)}\n")
        return 2
    spec: Optional[Tuple[str, str]] = None
    if args.trace:
        parts = args.trace.split(":")
        if len(parts) != 2:
            sys.stderr.write("WEB_TRACE_SPEC: --trace expects family:variant\n")
            return 2
        spec = (parts[0], parts[1])
    if int(args.threads_per_process) < 1:
        sys.stderr.write("WEB_PARAMS: --threads-per-process must be >= 1\n")
        return 2
    if args.dry_run:
        st = Stack(out_root / "raw" / "web_work" / "dry", "npz:<work>/tiny.npz", 1,
                   length_bucket=workloads.TINY_LENGTH_BUCKET, threads_per_process=args.threads_per_process)
        ok, msg = platform_available()
        labels = ["tiny"] + (["gpt2-1L", "gpt2-4L"] if args.model == "gpt2" else [])
        sys.stdout.write(json.dumps({"platform_available": ok, "message": msg, "scenarios": scen, "workloads": labels,
                                     "threads_per_process": int(args.threads_per_process),
                                     "colocated_processes": COLOCATED_PROCESSES,
                                     "cells": [web_run_id(l, s, 4 if l.endswith("4L") else 1, args.n_generate, r)
                                               for l in labels for s in scen for r in range(args.repeats)],
                                     "commands": st.commands()}, indent=1) + "\n")
        return 0
    try:
        summary = run(scen, args.model, args.n_generate, args.repeats, out_root, spec, args.trials, args.policy,
                      args.redo, threads_per_process=args.threads_per_process)
    except (ImportError, WebError) as exc:
        sys.stderr.write(str(exc) + "\n")
        return 2
    sys.stdout.write(json.dumps(summary, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main())


__all__ = ["SCENARIOS", "DEFAULT_SCENARIOS", "REJECTION_KINDS", "REQUIRED_MODULES", "DEFAULT_THREADS_PER_PROCESS",
           "COLOCATED_PROCESSES", "EPOCH0_KEYS", "Stack", "WebError", "platform_available", "token_metrics",
           "epoch0_summary", "scenario_cold_warm", "scenario_cancel", "scenario_reconnect", "scenario_server_restart",
           "scenario_rejections", "ws_probe", "web_record", "web_run_id", "workload_specs", "WORKLOAD_LABELS", "run",
           "main"]
