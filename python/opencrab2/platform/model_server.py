"""Model service P0: aiohttp WebSocket endpoint ``/session`` and the WebSocket channel adapter (paper, Section 3.2, design and threat model).

CLI::

    python -m opencrab2.platform.model_server --ws 127.0.0.1:8765 --dealer 127.0.0.1:PORT \
        --dealer-key-file key0.hex --model npz:tiny.npz --n-layers 1 [--model-version V] \
        [--state-dir DIR] [--ttl-s 600] [--trace-json trace.json] [--records PATH]

Frames on the WebSocket:

* text frames are JSON control messages: ``HELLO{session_id, model_version,
  numeric_contract_hash, profile_id, nonce, length_bucket, n_generate,
  prompt_len, n_layers, policy, trace_hash, client_key_domain, warm_digest}``
  (the client's prompt never travels, only its public length bucket),
  ``RESUME{session_id, model_version, numeric_contract_hash, profile_id,
  nonce}``, ``CANCEL``, ``PING``; the server answers ``HELLO_ACK{session_id,
  setup_id, model_version, model_card, dealer_mode, ttl_s, server_nonce,
  resume_epoch, warm, client_key_domain}``, ``REJECT{code}``
  (``OBJECT_VERSION_MISMATCH``, ``CONTRACT_MISMATCH``, ``PROFILE_MISMATCH``,
  ``SESSION_EXPIRED``, ``SESSION_UNKNOWN``, ``SESSION_BUSY``, ``SESSION_BAD_HELLO``),
  ``PONG``.
* binary frames tunnel the typed MPC payloads of ``highlevel.channel`` (never
  pickle) through :class:`WebSocketChannel`: header ``kind(1) seq(8 BE)
  gate_len(2 BE) gate_id`` then the payload. ``kind`` 0x01 is DATA, 0x02 FLOW
  (``seq`` carries granted credits). The receiver keeps a bounded inbound queue
  of ``window`` frames and grants credits as the party consumes them
  (backpressure); a DATA frame whose seq was already consumed is a duplicate:
  it is never delivered again, the frames the receiver sent in response to the
  original are retransmitted byte-identical from the send log after the
  ledger's retransmission log confirmed the digest, and the
  duplicate is counted. A gap in the sequence closes the channel.

Sessions are bound to ``(session_id, setup_id, model_version, contract hash,
profile)`` and expire after ``ttl_s`` of inactivity; a CANCEL raises the
declared cancel flag that both parties exchange at the next epoch boundary; a
RESUME on a cancelled session re-attaches the runner (warm resume, KV cache
kept) after the same binding checks. Logs carry session ids, frame kinds,
sizes and codes, never payload bytes. One session runs at a time per server
(the weight-mask ids of the dealer are not namespaced by session; design
limit), a second HELLO is refused with ``SESSION_BUSY``.

Warm stacks. The server keeps one :class:`~opencrab2.highlevel.session.WarmStack`
per ``(model_version, client_key_domain)``: the stack ledger, the dealer
connection, the mask shares, the installed offsets E = W - B and the shares
of the server-private vectors survive a session that ended at a declared
boundary (completed or cancelled). A HELLO carries the client's public stack
digest; when it equals the server's digest for that key the new session is a
warm request (``HELLO_ACK.warm = true``: nothing is installed or shared
again), otherwise both sides start a fresh stack. A session that fails or
disconnects inside a gate makes its stack unusable for later sessions (its
own resume keeps working on it). The digest holds ids, versions and names
only.

What is tested (tests/test_platform.py, subprocess): a full tiny-model
generation through the sidecar, a HELLO with another model_version refused,
a duplicate DATA frame answered from the retransmission log without advancing
the protocol, cancel leaving no RESERVED material. Design limits: TLS
termination is left to a reverse proxy (the binding covers identity, not
transport encryption on loopback); the retransmission log is in memory and
bounded, the ledger keeps the digests.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import queue
import secrets
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from aiohttp import WSMsgType, web

from opencrab2.errors import ChannelError, OpenCrabError
from opencrab2.highlevel.channel import ChannelStats, WallClockShaper, decode_payload, encode_payload
from opencrab2.highlevel.contract import NumericContract, contract_by_name
from opencrab2.highlevel.dealer import PROFILE_ID, DealerClient
from opencrab2.highlevel.ledger import MaterialLedger, payload_digest
from opencrab2.highlevel.session import (SessionConfig, SessionDealer, SessionRunner, WarmStack, load_model,
                                         model_card, model_version_of, new_warm_stack)

KIND_DATA = 0x01
KIND_FLOW = 0x02
_HDR = struct.Struct(">BQH")
DEFAULT_WINDOW = 64
MAX_WS_MSG = 1 << 31
SEND_LOG_FRAMES = 1024
REJECT_CODES = ("OBJECT_VERSION_MISMATCH", "CONTRACT_MISMATCH", "PROFILE_MISMATCH", "SESSION_EXPIRED",
                "SESSION_UNKNOWN", "SESSION_BUSY", "SESSION_BAD_HELLO")


def _log(msg: str) -> None:
    sys.stderr.write(f"[opencrab2-server] {msg}\n")
    sys.stderr.flush()


def read_key_file(path: str) -> bytes:
    key = bytes.fromhex(Path(path).read_text(encoding="utf-8").strip())
    if len(key) < 16:
        raise ChannelError("SERVER_KEY: dealer key shorter than 16 bytes")
    return key


# ---------------------------------------------------------------------------
# WebSocket channel adapter (sync Channel interface over an asyncio WebSocket)
# ---------------------------------------------------------------------------
class _Closed:
    pass


_CLOSED = _Closed()


class WebSocketChannel:
    """``send/recv/exchange/stats`` of ``highlevel.channel.Channel`` over one aiohttp WebSocket.

    The protocol thread calls the synchronous methods; the asyncio loop thread
    feeds :meth:`on_binary` / :meth:`on_closed` from the socket reader.
    ``ledger`` (optional) records every sent ``(gate_id, seq, digest)`` and
    authorizes retransmissions. Byte counters follow the TCP channel
    convention (payload plus an 8-byte prefix equivalent) so that web and TCP
    records compare; the WebSocket header overhead is counted separately.
    """

    def __init__(self, party: int, ws: Any, loop: asyncio.AbstractEventLoop, ledger: Optional[MaterialLedger] = None,
                 shaper: Optional[WallClockShaper] = None, window: int = DEFAULT_WINDOW, recv_timeout_s: float = 900.0,
                 send_timeout_s: float = 300.0, log_sends: bool = True) -> None:
        if party not in (0, 1):
            raise ChannelError(f"CHANNEL_PARAMS: party must be 0 or 1, got {party!r}")
        self.party = int(party)
        self.ws = ws
        self.loop = loop
        self.ledger = ledger
        self.shaper = shaper
        self.stats = ChannelStats()
        self.gate_id = "session"
        self.window = int(window)
        self.recv_timeout_s = float(recv_timeout_s)
        self.send_timeout_s = float(send_timeout_s)
        self.log_sends = bool(log_sends)
        self._inbox: "queue.Queue[Any]" = queue.Queue()
        self._credit = threading.Semaphore(self.window)
        self._delivered_since_grant = 0
        self._next_seq = 0
        self._expected = 0
        self._last_peer_seq: Optional[int] = None
        self._sent: "collections.OrderedDict[int, Tuple[str, bytes]]" = collections.OrderedDict()
        self._resp_start: Dict[int, int] = {}
        self._resp_end: Dict[int, int] = {}
        self._link_free_at = 0.0
        self._closed = False
        self._error: Optional[str] = None
        self._lock = threading.Lock()
        self.duplicates = 0
        self.retransmitted = 0
        self.retransmit_refused = 0
        self.flow_grants = 0
        self.ws_overhead_bytes = 0

    # -- loop-thread side -------------------------------------------------------------------
    def on_binary(self, data: bytes) -> None:
        """Called by the socket reader for every binary frame."""
        if len(data) < _HDR.size:
            self.on_closed("CHANNEL_WS_MALFORMED: short frame")
            return
        kind, seq, glen = _HDR.unpack_from(data, 0)
        if len(data) < _HDR.size + glen:
            self.on_closed("CHANNEL_WS_MALFORMED: truncated gate id")
            return
        self.ws_overhead_bytes += _HDR.size + glen
        if kind == KIND_FLOW:
            for _ in range(int(seq)):
                self._credit.release()
            return
        if kind != KIND_DATA:
            self.on_closed(f"CHANNEL_WS_MALFORMED: unknown frame kind {kind}")
            return
        payload = data[_HDR.size + glen:]
        with self._lock:
            if seq < self._expected:
                self.duplicates += 1
                self.loop.create_task(self._replay_response(int(seq)))
                return
            if seq > self._expected:
                self.on_closed(f"CHANNEL_WS_GAP: expected seq {self._expected}, got {seq}")
                return
            if self._last_peer_seq is not None:
                self._resp_end[self._last_peer_seq] = self._next_seq
            self._resp_start[int(seq)] = self._next_seq
            self._last_peer_seq = int(seq)
            self._expected = int(seq) + 1
        self._inbox.put(bytes(payload))

    def on_closed(self, error: Optional[str] = None) -> None:
        if self._closed:
            return
        self._closed = True
        self._error = error
        self._inbox.put(_CLOSED)
        for _ in range(self.window):
            self._credit.release()

    async def _replay_response(self, peer_seq: int) -> None:
        with self._lock:
            start = self._resp_start.get(peer_seq)
            end = self._resp_end.get(peer_seq, self._next_seq)
            frames = []
            if start is not None:
                for s in range(start, end):
                    ent = self._sent.get(s)
                    if ent is not None:
                        frames.append((s, ent[0], ent[1]))
        for s, gate, frame in frames:
            if self.ledger is not None and self.log_sends:
                payload = frame[_HDR.size + len(gate.encode("utf-8")):]
                if not self.ledger.retransmit_allowed(gate, s, payload_digest(payload)):
                    self.retransmit_refused += 1
                    continue
            try:
                await self.ws.send_bytes(frame)
                self.retransmitted += 1
            except Exception:
                return

    # -- protocol-thread side ---------------------------------------------------------------
    def set_shaper(self, shaper: Optional[WallClockShaper]) -> None:
        self.shaper = shaper
        self._link_free_at = 0.0

    def _shape(self, nbytes: int) -> None:
        if self.shaper is None:
            return
        now = time.time()
        start = max(now, self._link_free_at)
        ser = self.shaper.serialization_time(nbytes, start)
        self._link_free_at = start + ser
        delay = (self._link_free_at - now) + self.shaper.state_at(self._link_free_at).one_way_delay_s
        if delay > 0.0:
            self.stats.shaping_delay_s += delay
            time.sleep(delay)

    def _submit(self, coro: Any) -> Any:
        fut = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return fut.result(timeout=self.send_timeout_s)

    def send_raw(self, payload: bytes) -> int:
        if self._closed:
            raise ChannelError(f"CHANNEL_CLOSED: websocket closed ({self._error or 'peer'})")
        gate = str(self.gate_id).encode("utf-8")
        with self._lock:
            seq = self._next_seq
            self._next_seq += 1
        frame = _HDR.pack(KIND_DATA, seq, len(gate)) + gate + bytes(payload)
        self._shape(8 + len(payload))
        if not self._credit.acquire(timeout=self.send_timeout_s):
            raise ChannelError("CHANNEL_FLOW_TIMEOUT: peer granted no credit")
        if self._closed:
            raise ChannelError(f"CHANNEL_CLOSED: websocket closed ({self._error or 'peer'})")
        if self.ledger is not None and self.log_sends:
            self.ledger.record_send(self.gate_id, seq, payload_digest(payload))
        with self._lock:
            self._sent[seq] = (str(self.gate_id), frame)
            while len(self._sent) > SEND_LOG_FRAMES:
                self._sent.popitem(last=False)
        try:
            self._submit(self.ws.send_bytes(frame))
        except Exception as exc:
            raise ChannelError(f"CHANNEL_CLOSED: websocket send failed ({exc.__class__.__name__})") from exc
        self.stats.bytes_sent += 8 + len(payload)
        self.stats.messages_sent += 1
        self.ws_overhead_bytes += len(frame) - len(payload)
        return 8 + len(payload)

    def _grant(self, n: int) -> None:
        frame = _HDR.pack(KIND_FLOW, int(n), 0)
        self.flow_grants += 1
        try:
            self._submit(self.ws.send_bytes(frame))
        except Exception:
            pass

    def recv_raw(self) -> bytes:
        try:
            item = self._inbox.get(timeout=self.recv_timeout_s)
        except queue.Empty:
            raise ChannelError("CHANNEL_RECV_TIMEOUT: no frame from the peer") from None
        if item is _CLOSED:
            self._inbox.put(_CLOSED)
            raise ChannelError(f"CHANNEL_CLOSED: websocket closed ({self._error or 'peer'})")
        payload = bytes(item)
        self.stats.bytes_recv += 8 + len(payload)
        self.stats.messages_recv += 1
        self._delivered_since_grant += 1
        if self._delivered_since_grant >= max(1, self.window // 2):
            n = self._delivered_since_grant
            self._delivered_since_grant = 0
            self._grant(n)
        return payload

    def send(self, obj: Any) -> None:
        self.send_raw(encode_payload(obj))

    def recv(self) -> Any:
        return decode_payload(self.recv_raw())

    def exchange(self, obj: Any) -> Any:
        self.stats.rounds += 1
        if self.party == 0:
            self.send(obj)
            return self.recv()
        other = self.recv()
        self.send(obj)
        return other

    def count_round(self, n: int = 1) -> None:
        self.stats.rounds += int(n)

    def close(self) -> None:
        if not self._closed:
            self.on_closed("closed locally")
            try:
                self._submit(self.ws.close())
            except Exception:
                pass

    @property
    def closed(self) -> bool:
        return self._closed

    def counters(self) -> dict:
        return {"duplicates": self.duplicates, "retransmitted": self.retransmitted,
                "retransmit_refused": self.retransmit_refused, "flow_grants": self.flow_grants,
                "ws_overhead_bytes": self.ws_overhead_bytes, "next_seq": self._next_seq, "expected": self._expected}


async def pump_websocket(ws: Any, channel: WebSocketChannel, on_control: Any = None) -> None:
    """Read every frame of ``ws`` into ``channel`` (binary) or ``on_control`` (text JSON) until it closes."""
    try:
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                channel.on_binary(msg.data)
            elif msg.type == WSMsgType.TEXT:
                try:
                    ctl = json.loads(msg.data)
                except ValueError:
                    ctl = {"type": "MALFORMED"}
                if on_control is not None:
                    await on_control(ctl)
            elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED, WSMsgType.ERROR):
                break
    finally:
        channel.on_closed("peer closed")


# ---------------------------------------------------------------------------
# session table
# ---------------------------------------------------------------------------
class _Session:
    def __init__(self, session_id: str, runner: SessionRunner, ledger: MaterialLedger, dealer: SessionDealer,
                 service: Any, cancel_event: threading.Event) -> None:
        self.session_id = session_id
        self.runner = runner
        self.ledger = ledger
        self.dealer = dealer
        self.service = service
        self.cancel_event = cancel_event
        self.created = time.time()
        self.last_active = self.created
        self.status = "running"
        self.thread: Optional[threading.Thread] = None
        self.channel: Optional[WebSocketChannel] = None
        self.error = ""
        self.record_path = ""
        self.stack: Optional[WarmStack] = None
        self.warm = False

    def expired(self, ttl_s: float) -> bool:
        return (time.time() - self.last_active) > ttl_s

    def public(self) -> dict:
        try:
            counts = self.ledger.counts()
        except Exception:
            counts = {}
        return {"session_id": self.session_id, "status": self.status, "epochs_done": self.runner.epochs_done,
                "warm": bool(self.warm),
                "tokens": len(self.runner.generated) if self.runner.party_id == 1 else None,
                "created": self.created, "last_active": self.last_active, "ledger_counts": counts,
                "plan": dict(self.runner.plan), "error": self.error,
                "channel": self.channel.counters() if self.channel is not None else None}


class ModelServer:
    """P0 service: one loaded model, one dealer, a session table, the WebSocket endpoint."""

    def __init__(self, model: str, n_layers: int, contract: NumericContract, dealer_host: str, dealer_port: int,
                 dealer_key: bytes, state_dir: str, *, model_version: Optional[str] = None, ttl_s: float = 600.0,
                 trace: Any = None, records_path: Optional[str] = None, costmodel_json: Optional[dict] = None,
                 window: int = DEFAULT_WINDOW) -> None:
        self.contract = contract
        self.contract_hash = contract.contract_hash()
        self.n_layers = int(n_layers)
        self.W, self.meta = load_model(model, self.n_layers)
        self.card = model_card(self.meta, self.n_layers)
        self.model_version = model_version or model_version_of(self.meta)
        self.dealer_host, self.dealer_port, self.dealer_key = dealer_host, int(dealer_port), bytes(dealer_key)
        self.state_dir = str(state_dir)
        os.makedirs(self.state_dir, exist_ok=True)
        self.ttl_s = float(ttl_s)
        self.trace = trace
        self.records_path = records_path or os.path.join(self.state_dir, "server-records.jsonl")
        self.costmodel_json = costmodel_json
        self.window = int(window)
        self.sessions: Dict[str, _Session] = {}
        self.stacks: Dict[Tuple[str, str], WarmStack] = {}
        self._stack_seq = 0
        self._lock = threading.Lock()
        self.counters: Dict[str, int] = collections.Counter()
        self.setup_id: Optional[str] = None
        self.dealer_mode: Optional[str] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    # -- helpers ------------------------------------------------------------------------------
    def _dealer_connection(self):
        from opencrab2.platform.dealer_service import DealerServiceClient
        svc = DealerServiceClient(self.dealer_host, self.dealer_port, 0, self.dealer_key, self.contract_hash, retries=100)
        binding = svc.connect()
        if str(binding["model_version"]) != self.model_version:
            svc.close()
            raise OpenCrabError(f"OBJECT_VERSION_MISMATCH: dealer declares model_version {binding['model_version']!r}, "
                                f"server serves {self.model_version!r}")
        self.setup_id = str(binding["setup_id"])
        self.dealer_mode = str(binding["dealer_mode"])
        return svc

    def _reject(self, hello: dict) -> Optional[str]:
        if not isinstance(hello, dict) or hello.get("type") not in ("HELLO", "RESUME"):
            return "SESSION_BAD_HELLO"
        if str(hello.get("profile_id")) != PROFILE_ID:
            return "PROFILE_MISMATCH"
        if str(hello.get("numeric_contract_hash")) != self.contract_hash:
            return "CONTRACT_MISMATCH"
        if str(hello.get("model_version")) != self.model_version:
            return "OBJECT_VERSION_MISMATCH"
        sid = str(hello.get("session_id", ""))
        if not sid:
            return "SESSION_BAD_HELLO"
        if hello["type"] == "HELLO":
            for k in ("n_generate", "prompt_len", "n_layers", "length_bucket"):
                v = hello.get(k)
                if not isinstance(v, int) or isinstance(v, bool) or v < 1:
                    return "SESSION_BAD_HELLO"
            if int(hello["n_layers"]) != self.n_layers:
                return "SESSION_BAD_HELLO"
            with self._lock:
                self._expire_locked()
                if sid in self.sessions and self.sessions[sid].status in ("running", "resuming"):
                    return "SESSION_BUSY"
                if any(s.status in ("running", "resuming") for s in self.sessions.values()):
                    return "SESSION_BUSY"
        else:
            with self._lock:
                s = self.sessions.get(sid)
                if s is None:
                    return "SESSION_UNKNOWN"
                if s.expired(self.ttl_s):
                    s.status = "expired"
                    return "SESSION_EXPIRED"
                if s.status not in ("cancelled", "disconnected"):
                    return "SESSION_BUSY"
                if any(o.status in ("running", "resuming") for o in self.sessions.values()):
                    return "SESSION_BUSY"
        return None

    def _expire_locked(self) -> None:
        for s in self.sessions.values():
            if s.status in ("cancelled", "disconnected") and s.expired(self.ttl_s):
                s.status = "expired"
                self.counters["expired"] += 1

    def _new_session(self, hello: dict, channel: WebSocketChannel, cancel_event: threading.Event) -> _Session:
        sid = str(hello["session_id"])
        domain = str(hello.get("client_key_domain") or "default")
        want = str(hello.get("warm_digest") or "")
        key = (self.model_version, domain)
        with self._lock:
            stack = self.stacks.get(key)
        warm = bool(stack is not None and stack.usable and want and stack.digest() == want)
        if not warm:
            # no common reusable state: both sides start a fresh stack (new ledger, new dealer connection, new masks)
            svc = self._dealer_connection()
            with self._lock:
                self._stack_seq += 1
                seq = self._stack_seq
            sdir = os.path.join(self.state_dir, f"stack-{seq:04d}")
            os.makedirs(sdir, exist_ok=True)
            stack = new_warm_stack(0, self.model_version, domain, self.n_layers, os.path.join(sdir, "ledger0.jsonl"),
                                   str(self.setup_id), self.contract, self.dealer_key, svc)
            with self._lock:
                self.stacks[key] = stack
            self.counters["stacks_created"] += 1
        else:
            self.counters["warm_sessions"] += 1
        cfg = SessionConfig(model="loaded", n_layers=self.n_layers, prompt_ids=[], n_generate=int(hello["n_generate"]),
                            policy=str(hello.get("policy", "B4_opencrab")), trace_id=str(hello.get("trace_id", "unshaped")),
                            seed=int(hello.get("seed", 1)), contract=self.contract, length_bucket=int(hello["length_bucket"]),
                            session_id=sid, setup_id=str(self.setup_id), model_version=self.model_version,
                            prompt_len=int(hello["prompt_len"]), cpu_load_party=None, costmodel_json=self.costmodel_json,
                            client_key_domain=domain)
        channel.ledger = stack.ledger
        origin = float(hello.get("origin_wall_s") or time.time())
        runner = SessionRunner(0, cfg, channel, stack.client, stack.ledger, None, self.trace, self.records_path,
                               dealer_service=stack.service, cancel_event=cancel_event, origin_wall_s=origin,
                               weights=self.W, meta=self.meta, start_cpu_load=False, warm=stack)
        s = _Session(sid, runner, stack.ledger, runner.dealer, stack.service, cancel_event)
        s.channel = channel
        s.stack = stack
        s.warm = bool(runner.warm_start)
        return s

    def _run_thread(self, s: _Session, resume: bool) -> None:
        try:
            rec = s.runner.resume() if resume else s.runner.run()
            s.record_path = rec["paths"]["record"]
            s.status = "cancelled" if s.runner.cancelled else "done"
            self.counters["completed" if s.runner.completed else "cancelled"] += 1
        except ChannelError as exc:
            s.status = "disconnected"
            s.error = str(exc)
            self.counters["disconnected"] += 1
            self._burn_uncertain(s, "DISCONNECT")
        except Exception as exc:   # noqa: BLE001
            s.status = "failed"
            s.error = f"{exc.__class__.__name__}: {exc}"
            self.counters["failed"] += 1
            self._burn_uncertain(s, "FAILURE")
        finally:
            s.last_active = time.time()
            _log(f"session={s.session_id} status={s.status} epochs={s.runner.epochs_done} code={s.error.split(':')[0] if s.error else 'OK'}")
            if s.channel is not None:
                s.channel.close()

    @staticmethod
    def _burn_uncertain(s: _Session, reason: str) -> None:
        if s.stack is not None:
            s.stack.usable = False          # never offered to a later session; this session's own resume still works
        try:
            s.ledger.burn_reserved(reason)
            s.dealer.discard(reason)
        except Exception:
            pass

    # -- aiohttp handlers ---------------------------------------------------------------------------
    async def ws_handler(self, request: web.Request) -> web.StreamResponse:
        ws = web.WebSocketResponse(max_msg_size=MAX_WS_MSG, heartbeat=30.0)
        await ws.prepare(request)
        loop = asyncio.get_running_loop()
        try:
            first = await asyncio.wait_for(ws.receive(), timeout=30.0)
        except asyncio.TimeoutError:
            await ws.close()
            return ws
        hello: Any = None
        if first.type == WSMsgType.TEXT:
            try:
                hello = json.loads(first.data)
            except ValueError:
                hello = None
        code = self._reject(hello)
        if code is not None:
            self.counters[f"rejected_{code.lower()}"] += 1
            _log(f"rejected code={code}")
            await ws.send_json({"type": "REJECT", "code": code})
            await ws.close()
            return ws
        sid = str(hello["session_id"])
        resume = hello["type"] == "RESUME"
        cancel_event = threading.Event()
        channel = WebSocketChannel(0, ws, loop, None, None, window=self.window)
        try:
            if resume:
                with self._lock:
                    s = self.sessions[sid]
                    s.status = "resuming"
                channel.ledger = s.ledger
                s.channel = channel
                s.cancel_event.clear()
                s.runner.attach_channel(channel)
                s.runner.cancel_event = s.cancel_event
                cancel_event = s.cancel_event
            else:
                s = await loop.run_in_executor(None, self._new_session, hello, channel, cancel_event)
                with self._lock:
                    self.sessions[sid] = s
        except OpenCrabError as exc:
            self.counters["rejected_setup"] += 1
            _log(f"session setup failed code={exc.code}")
            await ws.send_json({"type": "REJECT", "code": exc.code})
            await ws.close()
            return ws
        s.status = "running"
        s.last_active = time.time()
        await ws.send_json({"type": "HELLO_ACK", "session_id": sid, "setup_id": self.setup_id,
                            "model_version": self.model_version, "numeric_contract_hash": self.contract_hash,
                            "profile_id": PROFILE_ID, "model_card": self.card, "dealer_mode": self.dealer_mode,
                            "ttl_s": self.ttl_s, "server_nonce": secrets.token_hex(16),
                            "resume_epoch": int(s.runner.epochs_done), "resume": resume,
                            "warm": bool(s.warm) and not resume,
                            "client_key_domain": str(s.runner.cfg.client_key_domain)})
        _log(f"session={sid} {'resume' if resume else 'hello'} prompt_len={hello.get('prompt_len')} n_generate={hello.get('n_generate')}")
        s.thread = threading.Thread(target=self._run_thread, args=(s, resume), name=f"session-{sid}", daemon=True)
        s.thread.start()

        async def on_control(ctl: dict) -> None:
            kind = ctl.get("type")
            s.last_active = time.time()
            if kind == "CANCEL":
                cancel_event.set()
                s.runner.cancel()
                self.counters["cancel_requests"] += 1
                _log(f"session={sid} cancel requested")
            elif kind == "PING":
                await ws.send_json({"type": "PONG", "t": time.time()})
            else:
                self.counters["unknown_control"] += 1

        await pump_websocket(ws, channel, on_control)
        s.last_active = time.time()
        return ws

    async def status_handler(self, request: web.Request) -> web.Response:
        with self._lock:
            self._expire_locked()
            sessions = {sid: s.public() for sid, s in self.sessions.items()}
        return web.json_response({"model_version": self.model_version, "numeric_contract_hash": self.contract_hash,
                                  "profile_id": PROFILE_ID, "setup_id": self.setup_id, "dealer_mode": self.dealer_mode,
                                  "n_layers": self.n_layers, "ttl_s": self.ttl_s, "counters": dict(self.counters),
                                  "stacks": [{"client_key_domain": k[1], "sessions": int(v.sessions),
                                              "usable": bool(v.usable), "installed": sorted(v.installed)}
                                             for k, v in sorted(self.stacks.items())],
                                  "sessions": sessions, "model_card": self.card})

    def app(self) -> web.Application:
        app = web.Application(client_max_size=1 << 20)
        app.router.add_get("/session", self.ws_handler)
        app.router.add_get("/v1/status", self.status_handler)
        return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_hostport(s: str) -> Tuple[str, int]:
    host, _, port = s.rpartition(":")
    if not host or not port.isdigit():
        raise argparse.ArgumentTypeError("SERVER_ARGS: expected host:port")
    return host, int(port)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="opencrab2-server", description="OpenCrab v2 model service (P0)")
    p.add_argument("--ws", type=_parse_hostport, default=("127.0.0.1", 8765), help="WebSocket listen host:port (0 = ephemeral)")
    p.add_argument("--dealer", type=_parse_hostport, required=True, help="dealer service host:port")
    p.add_argument("--dealer-key-file", required=True, help="file with the hex HMAC key of recipient 0")
    p.add_argument("--model", default="gpt2", help="'gpt2' or 'npz:<path>'")
    p.add_argument("--n-layers", type=int, default=1)
    p.add_argument("--model-version", default=None)
    p.add_argument("--contract", default="NC-v2-f16-l40")
    p.add_argument("--state-dir", default=None, help="default: raw/server-state under the data root")
    p.add_argument("--ttl-s", type=float, default=600.0)
    p.add_argument("--trace-json", default=None)
    p.add_argument("--records", default=None)
    p.add_argument("--port-file", default=None)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.state_dir:
        from opencrab2.experiments import schema
        args.state_dir = str(schema.experiments_root() / "raw" / "server-state")
    try:
        contract = contract_by_name(args.contract)
        key = read_key_file(args.dealer_key_file)
        trace = None
        if args.trace_json:
            with open(args.trace_json, "r", encoding="utf-8") as fh:
                trace = json.load(fh)
        server = ModelServer(args.model, args.n_layers, contract, args.dealer[0], args.dealer[1], key, args.state_dir,
                             model_version=args.model_version, ttl_s=args.ttl_s, trace=trace, records_path=args.records)
    except (OpenCrabError, OSError, ValueError) as exc:
        _log(str(exc))
        return 2

    async def _serve() -> None:
        server.loop = asyncio.get_running_loop()
        runner = web.AppRunner(server.app(), access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, args.ws[0], args.ws[1])
        await site.start()
        port = runner.addresses[0][1] if runner.addresses else args.ws[1]
        if args.port_file:
            with open(args.port_file, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(str(port))
        sys.stdout.write(f"LISTENING {args.ws[0]} {port}\n")
        sys.stdout.flush()
        _log(f"model_version={server.model_version} n_layers={server.n_layers} contract={contract.name}")
        while True:
            await asyncio.sleep(3600)

    try:
        asyncio.run(_serve())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main())


__all__ = ["WebSocketChannel", "pump_websocket", "ModelServer", "KIND_DATA", "KIND_FLOW", "REJECT_CODES",
           "read_key_file", "build_parser", "main"]
