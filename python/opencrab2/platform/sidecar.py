"""Client sidecar P1: local HTTP API for the browser page, MPC client over the WebSocket (paper, Section 3.2, design and threat model).

CLI::

    python -m opencrab2.platform.sidecar --http 127.0.0.1:8787 --server ws://127.0.0.1:8765/session \
        --dealer 127.0.0.1:PORT --dealer-key-file key1.hex --origin http://127.0.0.1:8787 \
        [--model-version V] [--n-layers 1] [--tokenizer bytes|gpt2] [--state-dir DIR] [--token-file PATH]
        [--client-key-domain NAME]

The sidecar is the trusted client: it runs on the user's machine, binds
to 127.0.0.1, serves the static browser page from ``browser_client/`` (no
third-party scripts, CSP ``default-src 'self'``) and exposes the API the page
calls. Every ``/v1/*`` request must carry the per-launch bearer token (printed
once as ``TOKEN <hex>`` and written to ``--token-file``) and, when a browser
sends one, an ``Origin`` from the allowlist; a request from another origin is
refused with 403 (``ORIGIN_REJECTED``), a missing or stale token with 401
(``TOKEN_REJECTED``). The prompt is tokenized locally and never leaves the
sidecar: the client's one-hot rows are its own share of the embedding input,
so the WebSocket carries only masked protocol frames and public metadata.

Endpoints: ``POST /v1/generate`` ``{prompt | prompt_ids, n_generate, policy}``
starts a session (HELLO over the WebSocket, binding checks, MPC client in a
thread) and returns ``{session_id, binding}``; ``POST /v1/cancel``
``{session_id}`` raises the declared cancel flag (burned RESERVED and buffered
material at the next epoch boundary); ``POST /v1/resume`` ``{session_id}``
reconnects and warm-resumes; ``GET /v1/stream?session_id=&since=`` is a
server-sent-events feed of released tokens and public status (``token``,
``epoch``, ``status``, ``done``, ``error``); ``GET /v1/status`` lists sessions
with ledger counts and the counters of rejected origins and tokens. An
``epoch`` event carries the public epoch record of the runner (plan, switch
flag, decision reason, sensing, planning, prepare, material wait, install,
commit and execution seconds, bytes, rounds, material, sensed link), so a
driver can tell which path a request took without the detail file.

Warm requests. The sidecar keeps one ``WarmStack`` per ``(model_version,
client_key_domain)`` (its ledger, dealer connection, mask shares, installed
offsets and vector shares) across the sessions of one launch. Every HELLO
carries the stack's public digest; when the model server answers
``HELLO_ACK.warm = true`` the session starts from that state (no installation,
no vector sharing; the ``done`` event then carries ``warm_start = true`` and
the measured ``T_warm_s``), otherwise a fresh stack is built on both sides.
A session that fails or disconnects inside a gate retires its stack.

What is tested (tests/test_platform.py): generation through the sidecar with
the token stream arriving, 403 on a wrong Origin, 401 on a stale token, cancel
leaving no RESERVED material and a resume completing the remaining tokens.
Design limits: the token is a per-launch capability for the local user (not a
user identity); the bytes tokenizer serves the tiny byte-vocabulary model and
the GPT-2 tokenizer needs ``transformers`` offline.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import hmac
import json
import os
import secrets
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from aiohttp import WSMsgType, web

from opencrab2.errors import ChannelError, ContractViolation, OpenCrabError
from opencrab2.highlevel.contract import NumericContract, contract_by_name
from opencrab2.highlevel.dealer import PROFILE_ID, DealerClient
from opencrab2.highlevel.ledger import MaterialLedger
from opencrab2.highlevel.session import SessionConfig, SessionDealer, SessionRunner, WarmStack, new_warm_stack
from opencrab2.platform.model_server import MAX_WS_MSG, WebSocketChannel, pump_websocket, read_key_file

BROWSER_DIR = Path(__file__).with_name("browser_client")
CSP = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:"


def _log(msg: str) -> None:
    sys.stderr.write(f"[opencrab2-sidecar] {msg}\n")
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# tokenizers (local only)
# ---------------------------------------------------------------------------
class BytesTokenizer:
    """UTF-8 bytes as token ids (the tiny model's vocabulary of 256)."""
    name = "bytes"

    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8"))

    def decode_token(self, token_id: int) -> str:
        return bytes([int(token_id)]).decode("utf-8", errors="replace") if 0 <= int(token_id) < 256 else ""


class GPT2Tokenizer:
    name = "gpt2"

    def __init__(self) -> None:
        from opencrab2.highlevel.transformer import load_gpt2_tokenizer
        self.tok = load_gpt2_tokenizer()                  # the pinned revision, local cache only

    def encode(self, text: str) -> List[int]:
        return [int(t) for t in self.tok.encode(text)]

    def decode_token(self, token_id: int) -> str:
        return self.tok.decode([int(token_id)])


def make_tokenizer(name: str) -> Any:
    if name == "bytes":
        return BytesTokenizer()
    if name == "gpt2":
        return GPT2Tokenizer()
    raise ContractViolation(f"SIDECAR_TOKENIZER: unknown tokenizer {name!r}")


# ---------------------------------------------------------------------------
# client session entry
# ---------------------------------------------------------------------------
class _ClientSession:
    def __init__(self, session_id: str, prompt_ids: List[int], n_generate: int, policy: str) -> None:
        self.session_id = session_id
        self.prompt_ids = list(prompt_ids)
        self.n_generate = int(n_generate)
        self.policy = policy
        self.status = "starting"
        self.error = ""
        self.binding: dict = {}
        self.history: List[dict] = []
        self.changed = asyncio.Event()
        self.cancel_event = threading.Event()
        self.runner: Optional[SessionRunner] = None
        self.ledger: Optional[MaterialLedger] = None
        self.dealer: Optional[SessionDealer] = None
        self.service: Any = None
        self.channel: Optional[WebSocketChannel] = None
        self.ws: Any = None
        self.http: Any = None
        self.thread: Optional[threading.Thread] = None
        self.created = time.time()
        self.tokens: List[int] = []
        self.record_path = ""
        self.stack: Optional[WarmStack] = None
        self.warm = False

    @property
    def finished(self) -> bool:
        return self.status in ("done", "cancelled", "rejected", "failed", "disconnected")

    def public(self) -> dict:
        counts = {}
        if self.ledger is not None:
            try:
                counts = self.ledger.counts()
            except Exception:
                counts = {}
        return {"session_id": self.session_id, "status": self.status, "error": self.error, "binding": self.binding,
                "tokens": len(self.tokens), "n_generate": self.n_generate, "prompt_len": len(self.prompt_ids),
                "warm": bool(self.warm),
                "epochs_done": self.runner.epochs_done if self.runner is not None else 0, "ledger_counts": counts,
                "events": len(self.history), "policy": self.policy,
                "channel": self.channel.counters() if self.channel is not None else None,
                "plan": dict(self.runner.plan) if self.runner is not None else None}


class Sidecar:
    """P1 service: browser-facing HTTP API plus the MPC client sessions."""

    def __init__(self, server_url: str, dealer_host: str, dealer_port: int, dealer_key: bytes, state_dir: str,
                 contract: NumericContract, token: str, origins: List[str], *, model_version: Optional[str] = None,
                 n_layers: int = 1, tokenizer: str = "bytes", trace: Any = None, records_path: Optional[str] = None,
                 length_bucket: int = 32, default_policy: str = "B4_opencrab", costmodel_json: Optional[dict] = None,
                 window: int = 64, client_key_domain: str = "default") -> None:
        self.server_url = str(server_url)
        self.dealer_host, self.dealer_port, self.dealer_key = dealer_host, int(dealer_port), bytes(dealer_key)
        self.state_dir = str(state_dir)
        os.makedirs(self.state_dir, exist_ok=True)
        self.contract = contract
        self.contract_hash = contract.contract_hash()
        self.token = str(token)
        self.origins = {o.rstrip("/") for o in origins}
        self.model_version = model_version
        self.n_layers = int(n_layers)
        self.tokenizer = make_tokenizer(tokenizer)
        self.trace = trace
        self.records_path = records_path or os.path.join(self.state_dir, "client-records.jsonl")
        self.length_bucket = int(length_bucket)
        self.default_policy = default_policy
        self.costmodel_json = costmodel_json
        self.window = int(window)
        self.sessions: Dict[str, _ClientSession] = {}
        self.client_key_domain = str(client_key_domain)
        self.stacks: Dict[Tuple[str, str], WarmStack] = {}
        self._stack_seq = 0
        self.counters: Dict[str, int] = collections.Counter()
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    # -- auth ----------------------------------------------------------------------------------------
    def _authorize(self, request: web.Request) -> Optional[web.Response]:
        origin = request.headers.get("Origin")
        if origin is not None:
            if origin.rstrip("/") not in self.origins:
                self.counters["rejected_origin"] += 1
                _log("rejected origin")
                return web.json_response({"code": "ORIGIN_REJECTED", "message": "origin not in the allowlist"}, status=403)
        else:
            site = request.headers.get("Sec-Fetch-Site")
            if site is not None and site not in ("same-origin", "none"):
                self.counters["rejected_origin"] += 1
                return web.json_response({"code": "ORIGIN_REJECTED", "message": "cross-site request"}, status=403)
        auth = request.headers.get("Authorization", "")
        ok = auth.startswith("Bearer ") and hmac.compare_digest(auth[len("Bearer "):].strip(), self.token)
        if not ok:
            self.counters["rejected_token"] += 1
            return web.json_response({"code": "TOKEN_REJECTED", "message": "missing or stale bearer token"}, status=401)
        return None

    # -- events --------------------------------------------------------------------------------------
    def _push(self, entry: _ClientSession, ev: dict) -> None:
        ev = dict(ev)
        ev.setdefault("t", time.time())
        entry.history.append(ev)
        if self.loop is not None:
            self.loop.call_soon_threadsafe(entry.changed.set)

    def _hooks(self, entry: _ClientSession) -> dict:
        def on_token(p: dict) -> None:
            tid = int(p["token_id"])
            entry.tokens.append(tid)
            self._push(entry, {"type": "token", "index": int(p["index"]), "token_id": tid,
                               "text": self.tokenizer.decode_token(tid), "epoch": int(p["epoch"])})

        def on_epoch(p: dict) -> None:
            keep = ("epoch", "gate", "plan", "switched", "aborted", "decision_reason", "sensing_s", "planning_s",
                    "prepare_s", "material_wait_s", "install_s", "commit_s", "exec_s", "bytes", "rounds",
                    "material_bytes_consumed", "wasted_bytes", "prefetched", "sensed_bw_bps", "sensed_rtt_s",
                    "installed_now", "kv_bucket")
            self._push(entry, {"type": "epoch", **{k: p.get(k) for k in keep}})

        def on_status(p: dict) -> None:
            self._push(entry, {"type": "status", **p})

        return {"token": on_token, "epoch": on_epoch, "status": on_status}

    # -- session start / resume --------------------------------------------------------------------------
    async def _connect(self, entry: _ClientSession, resume: bool) -> None:
        loop = asyncio.get_running_loop()
        http = aiohttp.ClientSession()
        entry.http = http
        try:
            ws = await http.ws_connect(self.server_url, max_msg_size=MAX_WS_MSG, heartbeat=30.0)
        except Exception as exc:
            entry.status = "failed"
            entry.error = f"SERVER_UNREACHABLE: {exc.__class__.__name__}"
            self._push(entry, {"type": "error", "code": "SERVER_UNREACHABLE"})
            await http.close()
            return
        entry.ws = ws
        model_version = self.model_version or (entry.binding.get("model_version") if resume else None)
        if model_version is None:
            known = sorted({k[0] for k, st in self.stacks.items() if st.usable})
            if len(known) == 1:
                model_version = known[0]
        if model_version is None:
            # learn the deployment's version from the dealer binding
            try:
                svc = await loop.run_in_executor(None, self._dealer_connect)
                model_version = str(svc.binding["model_version"])
                entry.service = svc
            except OpenCrabError as exc:
                entry.status = "failed"
                entry.error = str(exc)
                self._push(entry, {"type": "error", "code": exc.code})
                await ws.close()
                await http.close()
                return
        origin = time.time() + 0.5
        stack = self.stacks.get((str(model_version), self.client_key_domain))
        offer = stack.digest() if (stack is not None and stack.usable and not resume) else ""
        hello = {"type": "RESUME" if resume else "HELLO", "session_id": entry.session_id, "model_version": model_version,
                 "client_key_domain": self.client_key_domain, "warm_digest": offer,
                 "numeric_contract_hash": self.contract_hash, "profile_id": PROFILE_ID, "nonce": secrets.token_hex(16),
                 "length_bucket": self.length_bucket, "n_generate": entry.n_generate, "prompt_len": len(entry.prompt_ids),
                 "n_layers": self.n_layers, "policy": entry.policy, "trace_hash": "none", "origin_wall_s": origin,
                 "trace_id": "unshaped", "seed": 1}
        await ws.send_json(hello)
        try:
            msg = await asyncio.wait_for(ws.receive(), timeout=120.0)
            ack = json.loads(msg.data) if msg.type == WSMsgType.TEXT else {"type": "REJECT", "code": "SESSION_BAD_HELLO"}
        except (asyncio.TimeoutError, ValueError):
            ack = {"type": "REJECT", "code": "SERVER_TIMEOUT"}
        if ack.get("type") != "HELLO_ACK":
            code = str(ack.get("code", "SERVER_REJECTED"))
            entry.status = "rejected"
            entry.error = code
            self.counters[f"rejected_by_server_{code.lower()}"] += 1
            self._push(entry, {"type": "error", "code": code})
            _log(f"session={entry.session_id} rejected by the server code={code}")
            await ws.close()
            await http.close()
            return
        channel = WebSocketChannel(1, ws, loop, entry.ledger, None, window=self.window)
        entry.channel = channel
        if not resume:
            try:
                await loop.run_in_executor(None, self._setup_party, entry, ack, channel, hello)
            except OpenCrabError as exc:
                entry.status = "failed"
                entry.error = str(exc)
                self._push(entry, {"type": "error", "code": exc.code})
                await ws.close()
                await http.close()
                return
        else:
            channel.ledger = entry.ledger
            entry.cancel_event.clear()
            entry.runner.attach_channel(channel)
            entry.runner.cancel_event = entry.cancel_event
        entry.binding = {"session_id": entry.session_id, "setup_id": ack.get("setup_id"),
                         "model_version": ack.get("model_version"), "numeric_contract_hash": ack.get("numeric_contract_hash"),
                         "profile_id": ack.get("profile_id"), "dealer_mode": ack.get("dealer_mode"),
                         "server": self.server_url, "model_card": ack.get("model_card"), "resume_epoch": ack.get("resume_epoch")}
        entry.status = "running"
        self._push(entry, {"type": "status", "event": "connected", "binding": entry.binding, "resume": resume})
        entry.thread = threading.Thread(target=self._run_party, args=(entry, resume), name=f"client-{entry.session_id}",
                                        daemon=True)
        entry.thread.start()
        asyncio.create_task(self._pump(entry))

    def _dealer_connect(self):
        from opencrab2.platform.dealer_service import DealerServiceClient
        svc = DealerServiceClient(self.dealer_host, self.dealer_port, 1, self.dealer_key, self.contract_hash, retries=100)
        svc.connect()
        return svc

    def _setup_party(self, entry: _ClientSession, ack: dict, channel: WebSocketChannel, hello: dict) -> None:
        model_version = str(ack["model_version"])
        key = (model_version, self.client_key_domain)
        stack = self.stacks.get(key)
        offered = str(hello.get("warm_digest") or "")
        if bool(ack.get("warm")):
            # the server accepted the offered digest: this session must start from exactly that state
            if stack is None or not stack.usable or not offered or stack.digest() != offered:
                raise OpenCrabError("WARM_STACK_MISMATCH: the server announced a warm start the sidecar did not offer")
            svc = stack.service
            if entry.service is not None and entry.service is not svc:
                entry.service.close()
            self.counters["warm_sessions"] += 1
        else:
            svc = entry.service or self._dealer_connect()
            if str(svc.binding["model_version"]) != model_version:
                raise OpenCrabError(f"OBJECT_VERSION_MISMATCH: dealer declares {svc.binding['model_version']!r}, server {model_version!r}")
            self._stack_seq += 1
            sdir = os.path.join(self.state_dir, f"stack-{self._stack_seq:04d}")
            os.makedirs(sdir, exist_ok=True)
            stack = new_warm_stack(1, model_version, self.client_key_domain, self.n_layers,
                                   os.path.join(sdir, "ledger1.jsonl"), str(svc.binding["setup_id"]), self.contract,
                                   self.dealer_key, svc)
            self.stacks[key] = stack
            self.counters["stacks_created"] += 1
        entry.service = svc
        entry.stack = stack
        setup_id = str(stack.ledger.binding["setup_id"])
        entry.ledger = stack.ledger
        channel.ledger = stack.ledger
        cfg = SessionConfig(model="remote", n_layers=self.n_layers, prompt_ids=list(entry.prompt_ids),
                            n_generate=entry.n_generate, policy=entry.policy, trace_id="unshaped", seed=int(hello["seed"]),
                            contract=self.contract, length_bucket=self.length_bucket, session_id=entry.session_id,
                            setup_id=setup_id, model_version=model_version, cpu_load_party=None,
                            costmodel_json=self.costmodel_json, record_tokens=False,
                            client_key_domain=self.client_key_domain)
        entry.runner = SessionRunner(1, cfg, channel, stack.client, stack.ledger, None, self.trace, self.records_path,
                                     dealer_service=svc, cancel_event=entry.cancel_event,
                                     origin_wall_s=float(hello["origin_wall_s"]), meta=ack.get("model_card"),
                                     hooks=self._hooks(entry), start_cpu_load=False, warm=stack)
        entry.dealer = entry.runner.dealer
        entry.warm = bool(entry.runner.warm_start)

    async def _pump(self, entry: _ClientSession) -> None:
        async def on_control(ctl: dict) -> None:
            if ctl.get("type") == "PONG":
                return
            self.counters["server_control"] += 1
        try:
            await pump_websocket(entry.ws, entry.channel, on_control)
        finally:
            if entry.http is not None:
                try:
                    await entry.http.close()
                except Exception:
                    pass

    def _run_party(self, entry: _ClientSession, resume: bool) -> None:
        try:
            rec = entry.runner.resume() if resume else entry.runner.run()
            entry.record_path = rec["paths"]["record"]
            entry.status = "cancelled" if entry.runner.cancelled else "done"
            self.counters["completed" if entry.runner.completed else "cancelled"] += 1
            self._push(entry, {"type": "done", "status": entry.status, "tokens": len(entry.tokens),
                               "T_cold_s": rec["T_cold_s"], "T_warm_s": rec["T_warm_s"], "T_online_s": rec["T_online_s"],
                               "warm_start": bool(rec["extra"].get("warm_start")),
                               "T_warm_basis": rec["extra"].get("T_warm_basis"),
                               "session_wall_s": rec["extra"].get("session_wall_s"),
                               "runner_ttft_s": rec["extra"].get("ttft_s"),
                               "install_s": rec["extra"].get("install_s"), "setup_s": rec["extra"].get("setup_s"),
                               "wasted_bytes": rec["extra"].get("wasted_bytes"),
                               "burned_on_cancel": rec["extra"].get("burned_on_cancel"),
                               "material_consumed": rec["material_consumed"],
                               "peak_rss_client": rec["peak_rss_client"], "peak_rss_server": rec["peak_rss_server"],
                               "security_eligible": rec["security_eligible"], "bytes": rec["bytes"], "rounds": rec["rounds"],
                               "material_burned": rec["material_burned"], "epochs_done": entry.runner.epochs_done,
                               "ledger_counts": entry.ledger.counts() if entry.ledger else {}})
        except ChannelError as exc:
            entry.status = "disconnected"
            entry.error = str(exc)
            self.counters["disconnected"] += 1
            self._burn(entry, "DISCONNECT")
            self._push(entry, {"type": "error", "code": exc.code, "recoverable": True})
        except Exception as exc:   # noqa: BLE001
            entry.status = "failed"
            entry.error = f"{exc.__class__.__name__}: {exc}"
            self.counters["failed"] += 1
            self._burn(entry, "FAILURE")
            code = getattr(exc, "code", exc.__class__.__name__)
            self._push(entry, {"type": "error", "code": code})
        finally:
            _log(f"session={entry.session_id} status={entry.status} tokens={len(entry.tokens)}")
            if entry.channel is not None:
                entry.channel.close()

    @staticmethod
    def _burn(entry: _ClientSession, reason: str) -> None:
        if entry.stack is not None:
            entry.stack.usable = False      # never offered to a later session; this session's own resume still works
        try:
            if entry.ledger is not None:
                entry.ledger.burn_reserved(reason)
            if entry.dealer is not None:
                entry.dealer.discard(reason)
        except Exception:
            pass

    # -- HTTP handlers -----------------------------------------------------------------------------------
    async def index(self, request: web.Request) -> web.Response:
        body = (BROWSER_DIR / "index.html").read_text(encoding="utf-8")
        return web.Response(text=body, content_type="text/html", headers={"Content-Security-Policy": CSP,
                                                                           "Referrer-Policy": "no-referrer"})

    async def app_js(self, request: web.Request) -> web.Response:
        body = (BROWSER_DIR / "app.js").read_text(encoding="utf-8")
        return web.Response(text=body, content_type="application/javascript",
                            headers={"Content-Security-Policy": CSP, "Cache-Control": "no-store"})

    async def generate(self, request: web.Request) -> web.Response:
        denied = self._authorize(request)
        if denied is not None:
            return denied
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"code": "BAD_REQUEST", "message": "JSON body expected"}, status=400)
        if not isinstance(body, dict):
            return web.json_response({"code": "BAD_REQUEST"}, status=400)
        if isinstance(body.get("prompt_ids"), list):
            try:
                ids = [int(t) for t in body["prompt_ids"]]
            except (TypeError, ValueError):
                return web.json_response({"code": "BAD_REQUEST", "message": "prompt_ids must be ints"}, status=400)
        elif isinstance(body.get("prompt"), str) and body["prompt"]:
            ids = self.tokenizer.encode(body["prompt"])
        else:
            return web.json_response({"code": "BAD_REQUEST", "message": "prompt or prompt_ids required"}, status=400)
        if not ids:
            return web.json_response({"code": "BAD_REQUEST", "message": "empty prompt"}, status=400)
        n_generate = int(body.get("n_generate", 8))
        if n_generate < 1 or n_generate > 4096:
            return web.json_response({"code": "BAD_REQUEST", "message": "n_generate out of range"}, status=400)
        policy = str(body.get("policy", self.default_policy))
        sid = str(body.get("session_id") or ("web-" + secrets.token_hex(8)))
        if sid in self.sessions and not self.sessions[sid].finished:
            return web.json_response({"code": "SESSION_BUSY"}, status=409)
        entry = _ClientSession(sid, ids, n_generate, policy)
        self.sessions[sid] = entry
        self.counters["generate_requests"] += 1
        _log(f"generate session={sid} prompt_len={len(ids)} n_generate={n_generate} policy={policy}")
        asyncio.create_task(self._connect(entry, resume=False))
        return web.json_response({"session_id": sid, "prompt_len": len(ids), "n_generate": n_generate, "policy": policy})

    async def cancel(self, request: web.Request) -> web.Response:
        denied = self._authorize(request)
        if denied is not None:
            return denied
        body = await request.json()
        entry = self.sessions.get(str(body.get("session_id", "")))
        if entry is None:
            return web.json_response({"code": "SESSION_UNKNOWN"}, status=404)
        entry.cancel_event.set()
        if entry.runner is not None:
            entry.runner.cancel()
        if entry.ws is not None and not entry.ws.closed:
            try:
                await entry.ws.send_json({"type": "CANCEL", "session_id": entry.session_id})
            except Exception:
                pass
        self.counters["cancel_requests"] += 1
        self._push(entry, {"type": "status", "event": "cancel_requested"})
        return web.json_response({"session_id": entry.session_id, "status": entry.status, "cancel": True})

    async def resume(self, request: web.Request) -> web.Response:
        denied = self._authorize(request)
        if denied is not None:
            return denied
        body = await request.json()
        entry = self.sessions.get(str(body.get("session_id", "")))
        if entry is None:
            return web.json_response({"code": "SESSION_UNKNOWN"}, status=404)
        if entry.runner is None or entry.status not in ("cancelled", "disconnected"):
            return web.json_response({"code": "SESSION_NOT_RESUMABLE", "status": entry.status}, status=409)
        if entry.runner.completed:
            return web.json_response({"code": "SESSION_COMPLETED"}, status=409)
        entry.status = "resuming"
        self.counters["resume_requests"] += 1
        asyncio.create_task(self._connect(entry, resume=True))
        return web.json_response({"session_id": entry.session_id, "status": entry.status,
                                  "resume_epoch": entry.runner.epochs_done})

    async def stream(self, request: web.Request) -> web.StreamResponse:
        denied = self._authorize(request)
        if denied is not None:
            return denied
        entry = self.sessions.get(request.query.get("session_id", ""))
        if entry is None:
            return web.json_response({"code": "SESSION_UNKNOWN"}, status=404)
        try:
            idx = max(0, int(request.query.get("since", "0")))
        except ValueError:
            idx = 0
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-store",
                                           "X-Accel-Buffering": "no"})
        await resp.prepare(request)
        try:
            while True:
                while idx < len(entry.history):
                    ev = entry.history[idx]
                    data = json.dumps(ev, default=str)
                    await resp.write(f"event: {ev.get('type', 'message')}\nid: {idx}\ndata: {data}\n\n".encode("utf-8"))
                    idx += 1
                    if ev.get("type") in ("done", "error"):
                        await resp.write(b"event: end\ndata: {}\n\n")
                        return resp
                entry.changed.clear()
                try:
                    await asyncio.wait_for(entry.changed.wait(), timeout=15.0)
                except asyncio.TimeoutError:
                    await resp.write(b": keep-alive\n\n")
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        return resp

    async def status(self, request: web.Request) -> web.Response:
        denied = self._authorize(request)
        if denied is not None:
            return denied
        return web.json_response({"server": self.server_url, "model_version": self.model_version,
                                  "numeric_contract_hash": self.contract_hash, "profile_id": PROFILE_ID,
                                  "origins": sorted(self.origins), "tokenizer": self.tokenizer.name, "n_layers": self.n_layers,
                                  "counters": dict(self.counters), "client_key_domain": self.client_key_domain,
                                  "stacks": [{"client_key_domain": k[1], "sessions": int(v.sessions),
                                              "usable": bool(v.usable), "installed": sorted(v.installed)}
                                             for k, v in sorted(self.stacks.items())],
                                  "sessions": {sid: e.public() for sid, e in self.sessions.items()}})

    def app(self) -> web.Application:
        app = web.Application(client_max_size=1 << 20)
        app.router.add_get("/", self.index)
        app.router.add_get("/index.html", self.index)
        app.router.add_get("/app.js", self.app_js)
        app.router.add_post("/v1/generate", self.generate)
        app.router.add_post("/v1/cancel", self.cancel)
        app.router.add_post("/v1/resume", self.resume)
        app.router.add_get("/v1/stream", self.stream)
        app.router.add_get("/v1/status", self.status)
        return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_hostport(s: str) -> Tuple[str, int]:
    host, _, port = s.rpartition(":")
    if not host or not port.isdigit():
        raise argparse.ArgumentTypeError("SIDECAR_ARGS: expected host:port")
    return host, int(port)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="opencrab2-sidecar", description="OpenCrab v2 client sidecar (P1)")
    p.add_argument("--http", type=_parse_hostport, default=("127.0.0.1", 8787))
    p.add_argument("--server", required=True, help="model service WebSocket URL, e.g. ws://127.0.0.1:8765/session")
    p.add_argument("--dealer", type=_parse_hostport, required=True)
    p.add_argument("--dealer-key-file", required=True, help="file with the hex HMAC key of recipient 1")
    p.add_argument("--origin", action="append", default=None, help="allowed browser origin (repeatable); default: own origin")
    p.add_argument("--model-version", default=None)
    p.add_argument("--n-layers", type=int, default=1)
    p.add_argument("--contract", default="NC-v2-f16-l40")
    p.add_argument("--tokenizer", default="bytes", choices=["bytes", "gpt2"])
    p.add_argument("--state-dir", default=None, help="default: raw/sidecar-state under the data root")
    p.add_argument("--token-file", default=None)
    p.add_argument("--trace-json", default=None)
    p.add_argument("--records", default=None)
    p.add_argument("--length-bucket", type=int, default=32)
    p.add_argument("--port-file", default=None)
    p.add_argument("--client-key-domain", default="default",
                   help="key domain of this client; installed objects are reused only within one domain")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.http[0] not in ("127.0.0.1", "localhost", "::1"):
        _log("SIDECAR_BIND: the sidecar binds to loopback only")
        return 2
    try:
        contract = contract_by_name(args.contract)
        key = read_key_file(args.dealer_key_file)
        trace = None
        if args.trace_json:
            with open(args.trace_json, "r", encoding="utf-8") as fh:
                trace = json.load(fh)
    except (OpenCrabError, OSError, ValueError) as exc:
        _log(str(exc))
        return 2
    token = secrets.token_hex(24)
    if not args.state_dir:
        from opencrab2.experiments import schema
        args.state_dir = str(schema.experiments_root() / "raw" / "sidecar-state")

    async def _serve() -> None:
        runner_app = web.AppRunner(web.Application(), access_log=None)
        # bind first to learn the port, so the default origin can be computed
        sidecar = Sidecar(args.server, args.dealer[0], args.dealer[1], key, args.state_dir, contract, token,
                          args.origin or [], model_version=args.model_version, n_layers=args.n_layers,
                          tokenizer=args.tokenizer, trace=trace, records_path=args.records, length_bucket=args.length_bucket,
                          client_key_domain=args.client_key_domain)
        sidecar.loop = asyncio.get_running_loop()
        runner_app = web.AppRunner(sidecar.app(), access_log=None)
        await runner_app.setup()
        site = web.TCPSite(runner_app, args.http[0], args.http[1])
        await site.start()
        port = runner_app.addresses[0][1] if runner_app.addresses else args.http[1]
        own = f"http://{args.http[0]}:{port}"
        if not args.origin:
            sidecar.origins = {own}
        if args.token_file:
            from opencrab2.platform.dealer_service import write_private_text
            write_private_text(args.token_file, token)       # mode 0600 on POSIX
        if args.port_file:
            with open(args.port_file, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(str(port))
        sys.stdout.write(f"LISTENING {args.http[0]} {port}\nTOKEN {token}\nURL {own}/#token={token}\n")
        sys.stdout.flush()
        _log(f"server={args.server} origins={sorted(sidecar.origins)} tokenizer={args.tokenizer}")
        while True:
            await asyncio.sleep(3600)

    try:
        asyncio.run(_serve())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main())


__all__ = ["Sidecar", "BytesTokenizer", "GPT2Tokenizer", "make_tokenizer", "build_parser", "main", "CSP", "BROWSER_DIR"]
