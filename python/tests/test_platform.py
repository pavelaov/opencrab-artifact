"""Web platform tests: dealer service, model server and sidecar as subprocesses.

The tiny byte-vocabulary model of tests/test_session.py is served by the model
server; the sidecar tokenizes the prompt locally (UTF-8 bytes) and drives the
MPC client over the WebSocket. Checks: the token stream arrives and equals the
float reference, a wrong Origin is refused with 403, a stale token with 401, a
HELLO with another model_version is refused with OBJECT_VERSION_MISMATCH, a
duplicate DATA frame is answered byte-identically from the retransmission log
without advancing the protocol, and a cancel mid-session leaves no RESERVED
material (then a resume completes the remaining tokens).
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import secrets
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import pytest

aiohttp = pytest.importorskip("aiohttp")

from opencrab2.highlevel import session as S                                    # noqa: E402
from opencrab2.highlevel.channel import decode_payload, encode_payload           # noqa: E402
from opencrab2.highlevel.contract import DEFAULT_CONTRACT                        # noqa: E402
from opencrab2.highlevel.dealer import MODE_ISOLATED, PROFILE_ID                 # noqa: E402
from opencrab2.platform.dealer_service import write_keys                         # noqa: E402
from opencrab2.platform.model_server import KIND_DATA, KIND_FLOW, _HDR           # noqa: E402
from tests.test_session import PROMPT, make_tiny_weights, reference_tokens       # noqa: E402

PYDIR = str(pathlib.Path(__file__).resolve().parents[1])
C = DEFAULT_CONTRACT
KEYS = {0: secrets.token_bytes(32), 1: secrets.token_bytes(32)}
PROMPT_TEXT = bytes(PROMPT).decode("utf-8")      # "Hell"


def _env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = PYDIR + os.pathsep + env.get("PYTHONPATH", "")
    env["HF_HUB_OFFLINE"] = "1"
    return env


def _start(cmd: List[str], want: List[str], log_path: pathlib.Path,
           timeout_s: float = 90.0) -> Tuple[subprocess.Popen, Dict[str, str]]:
    """Spawn a service; stderr goes to ``log_path`` (a PIPE nobody drains would block the child's logging)."""
    log = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=PYDIR, env=_env(), stdout=subprocess.PIPE, stderr=log, text=True)
    got: Dict[str, str] = {}
    deadline = time.time() + timeout_s
    while time.time() < deadline and not all(k in got for k in want):
        line = proc.stdout.readline()
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
        err = log_path.read_text(encoding="utf-8", errors="replace")
        pytest.fail(f"process {cmd[2]} did not start: {err[-2000:]}")
    return proc, got


class Platform:
    def __init__(self, tmp: pathlib.Path) -> None:
        self.tmp = tmp
        self.W, self.meta = make_tiny_weights(1, d=64, vocab=256)
        self.npz = S.save_model_npz(str(tmp / "tiny.npz"), self.W, self.meta)
        self.model_version = S.model_version_of(self.meta)
        self.ref = reference_tokens(self.W, self.meta, 1, PROMPT, 8)
        keys_path = str(tmp / "keys.json")
        write_keys(keys_path, KEYS)
        (tmp / "key0.hex").write_text(KEYS[0].hex(), encoding="utf-8")
        (tmp / "key1.hex").write_text(KEYS[1].hex(), encoding="utf-8")
        self.procs: List[subprocess.Popen] = []
        py = sys.executable
        p, got = _start([py, "-m", "opencrab2.platform.dealer_service", "--listen", "127.0.0.1:0", "--keys", keys_path,
                         "--contract", C.name, "--setup-id", "web-setup", "--model-version", self.model_version], ["LISTENING"],
                        tmp / "dealer.log")
        self.procs.append(p)
        self.dealer = got["LISTENING"].split()[1]
        p, got = _start([py, "-m", "opencrab2.platform.model_server", "--ws", "127.0.0.1:0", "--dealer", f"127.0.0.1:{self.dealer}",
                         "--dealer-key-file", str(tmp / "key0.hex"), "--model", f"npz:{self.npz}", "--n-layers", "1",
                         "--state-dir", str(tmp / "server"), "--ttl-s", "120"], ["LISTENING"], tmp / "server.log")
        self.procs.append(p)
        self.server_port = got["LISTENING"].split()[1]
        self.ws_url = f"ws://127.0.0.1:{self.server_port}/session"
        p, got = _start([py, "-m", "opencrab2.platform.sidecar", "--http", "127.0.0.1:0", "--server", self.ws_url,
                         "--dealer", f"127.0.0.1:{self.dealer}", "--dealer-key-file", str(tmp / "key1.hex"),
                         "--n-layers", "1", "--tokenizer", "bytes", "--state-dir", str(tmp / "sidecar"),
                         "--length-bucket", "8"], ["LISTENING", "TOKEN"], tmp / "sidecar.log")
        self.procs.append(p)
        self.sidecar_port = got["LISTENING"].split()[1]
        self.token = got["TOKEN"]
        self.origin = f"http://127.0.0.1:{self.sidecar_port}"
        self.base = self.origin

    def headers(self, token: Optional[str] = None, origin: Optional[str] = "own") -> dict:
        h = {"Authorization": f"Bearer {self.token if token is None else token}"}
        if origin == "own":
            h["Origin"] = self.origin
        elif origin is not None:
            h["Origin"] = origin
        return h

    def logs(self) -> Dict[str, str]:
        return {n: (self.tmp / f"{n}.log").read_text(encoding="utf-8", errors="replace")
                for n in ("dealer", "server", "sidecar")}

    def close(self) -> None:
        for p in reversed(self.procs):
            p.kill()
        for p in self.procs:
            try:
                p.communicate(timeout=15)
            except subprocess.TimeoutExpired:
                pass


@pytest.fixture(scope="module")
def plat(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("platform")
    p = Platform(tmp)
    yield p
    p.close()


def run(coro):
    return asyncio.run(coro)


async def _post(plat: Platform, path: str, body: dict, **hk) -> Tuple[int, Any]:
    async with aiohttp.ClientSession() as s:
        async with s.post(plat.base + path, json=body, headers=plat.headers(**hk)) as r:
            try:
                return r.status, await r.json()
            except Exception:
                return r.status, None


async def _get(plat: Platform, path: str, **hk) -> Tuple[int, Any]:
    async with aiohttp.ClientSession() as s:
        async with s.get(plat.base + path, headers=plat.headers(**hk)) as r:
            return r.status, await r.json()


async def _stream(plat: Platform, sid: str, since: int = 0, stop_on: Optional[str] = None, on_event=None,
                  timeout_s: float = 240.0) -> List[dict]:
    """Collect SSE events of a session until ``end`` (or until ``stop_on`` type arrives)."""
    events: List[dict] = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_s)) as s:
        async with s.get(f"{plat.base}/v1/stream?session_id={sid}&since={since}", headers=plat.headers()) as r:
            assert r.status == 200, await r.text()
            buf = ""
            async for chunk in r.content.iter_any():
                buf += chunk.decode("utf-8")
                while "\n\n" in buf:
                    block, buf = buf.split("\n\n", 1)
                    typ, data, eid = "message", None, None
                    for line in block.split("\n"):
                        if line.startswith("event: "):
                            typ = line[7:].strip()
                        elif line.startswith("data: "):
                            data = line[6:]
                        elif line.startswith("id: "):
                            eid = int(line[4:])
                    if typ == "end":
                        return events
                    if data is not None:
                        ev = json.loads(data)
                        ev["_type"] = typ
                        ev["_id"] = eid
                        events.append(ev)
                        if on_event is not None:
                            await on_event(ev)
                        if stop_on is not None and typ == stop_on:
                            return events
    return events


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
def test_generate_streams_tokens_through_the_sidecar(plat):
    status, body = run(_post(plat, "/v1/generate", {"prompt": PROMPT_TEXT, "n_generate": 3, "policy": "B2_remaining_plus_install"}))
    assert status == 200 and body["prompt_len"] == len(PROMPT)
    sid = body["session_id"]
    events = run(_stream(plat, sid))
    tokens = [e["token_id"] for e in events if e["_type"] == "token"]
    assert tokens == plat.ref[:3]
    done = [e for e in events if e["_type"] == "done"]
    assert done and done[0]["status"] == "done" and done[0]["tokens"] == 3 and done[0]["security_eligible"] is True
    bind = [e for e in events if e["_type"] == "status" and e.get("binding")]
    assert bind and bind[0]["binding"]["model_version"] == plat.model_version
    assert bind[0]["binding"]["dealer_mode"] == MODE_ISOLATED and bind[0]["binding"]["profile_id"] == PROFILE_ID
    epochs = [e for e in events if e["_type"] == "epoch"]
    assert len(epochs) == 3 and epochs[0]["gate"] == "prefill" and epochs[-1]["gate"] == "decode"
    st = run(_get(plat, "/v1/status"))[1]
    assert st["sessions"][sid]["status"] == "done" and st["sessions"][sid]["ledger_counts"]["RESERVED"] == 0
    # the server side agrees and never saw a plaintext prompt (it only knows the public length)
    async def server_status():
        async with aiohttp.ClientSession() as s:
            async with s.get(f"http://127.0.0.1:{plat.server_port}/v1/status") as r:
                return await r.json()
    sst = run(server_status())
    assert sst["sessions"][sid]["status"] == "done" and sst["model_version"] == plat.model_version
    assert sst["sessions"][sid]["ledger_counts"]["RESERVED"] == 0
    # the page is served same-origin without third-party scripts
    async def page():
        async with aiohttp.ClientSession() as s:
            async with s.get(plat.base + "/") as r:
                return r.status, r.headers.get("Content-Security-Policy", ""), await r.text()
    code, csp, html = run(page())
    assert code == 200 and "default-src 'self'" in csp and "http://" not in html.split("<script")[-1]
    assert 'src="app.js"' in html and "https://" not in html


def test_wrong_origin_403_and_stale_token_401(plat):
    status, body = run(_post(plat, "/v1/generate", {"prompt": "x", "n_generate": 1}, origin="http://evil.example"))
    assert status == 403 and body["code"] == "ORIGIN_REJECTED"
    status, body = run(_post(plat, "/v1/generate", {"prompt": "x", "n_generate": 1}, token="deadbeef" * 6))
    assert status == 401 and body["code"] == "TOKEN_REJECTED"
    status, body = run(_get(plat, "/v1/status", token=""))
    assert status == 401
    status, body = run(_get(plat, "/v1/status", origin="https://127.0.0.1:1"))
    assert status == 403
    st = run(_get(plat, "/v1/status"))[1]
    assert st["counters"]["rejected_origin"] >= 2 and st["counters"]["rejected_token"] >= 2


def _hello(plat: Platform, sid: str, **over) -> dict:
    h = {"type": "HELLO", "session_id": sid, "model_version": plat.model_version, "numeric_contract_hash": C.contract_hash(),
         "profile_id": PROFILE_ID, "nonce": secrets.token_hex(8), "length_bucket": 8, "n_generate": 2, "prompt_len": 4,
         "n_layers": 1, "policy": "B1_naive", "trace_hash": "none", "origin_wall_s": time.time(), "trace_id": "unshaped", "seed": 1}
    h.update(over)
    return h


def test_hello_with_wrong_model_version_is_refused(plat):
    async def go(over: dict) -> dict:
        async with aiohttp.ClientSession() as s:
            async with s.ws_connect(plat.ws_url) as ws:
                await ws.send_json(_hello(plat, "bad-" + secrets.token_hex(3), **over))
                msg = await asyncio.wait_for(ws.receive(), timeout=30)
                return json.loads(msg.data)
    r = run(go({"model_version": plat.model_version + "-other"}))
    assert r["type"] == "REJECT" and r["code"] == "OBJECT_VERSION_MISMATCH"
    r = run(go({"numeric_contract_hash": "00" * 32}))
    assert r["type"] == "REJECT" and r["code"] == "CONTRACT_MISMATCH"
    r = run(go({"profile_id": "ACTIVE-2PC"}))
    assert r["type"] == "REJECT" and r["code"] == "PROFILE_MISMATCH"
    r = run(go({"type": "RESUME", "session_id": "never-seen"}))
    assert r["type"] == "REJECT" and r["code"] == "SESSION_UNKNOWN"


def _frame(seq: int, payload: bytes, gate: bytes = b"session") -> bytes:
    return _HDR.pack(KIND_DATA, seq, len(gate)) + gate + payload


def _parse(data: bytes):
    kind, seq, glen = _HDR.unpack_from(data, 0)
    return kind, seq, data[_HDR.size + glen:]


def test_duplicate_frame_is_answered_from_the_retransmission_log(plat):
    sid = "dup-" + secrets.token_hex(3)

    async def go():
        async with aiohttp.ClientSession() as s:
            async with s.ws_connect(plat.ws_url, max_msg_size=1 << 30) as ws:
                await ws.send_json(_hello(plat, sid))
                ack = json.loads((await asyncio.wait_for(ws.receive(), timeout=60)).data)
                assert ack["type"] == "HELLO_ACK" and ack["model_version"] == plat.model_version

                async def next_data():
                    while True:
                        m = await asyncio.wait_for(ws.receive(), timeout=120)
                        assert m.type == aiohttp.WSMsgType.BINARY, m
                        kind, seq, payload = _parse(m.data)
                        if kind == KIND_FLOW:
                            continue
                        return seq, m.data, payload

                seq0, raw0, p0 = await next_data()
                assert seq0 == 0
                server_hello = decode_payload(p0)
                assert server_hello["party"] == 0 and server_hello["model_card"]["vocab"] == 256
                client_hello = dict(server_hello, party=1, model_card=None, nonce=secrets.token_hex(16))
                mine = _frame(0, encode_payload(client_hello))
                await ws.send_bytes(mine)
                # the server answers with its vector-share message and the first epoch's status frame
                first = [await next_data() for _ in range(2)]
                assert [f[0] for f in first] == [1, 2]
                assert decode_payload(first[1][2]) == {"cancel": False, "epoch": 0, "generation": 0}
                # byte-identical duplicate of our seq 0: replayed answers, protocol not advanced
                await ws.send_bytes(mine)
                again = [await next_data() for _ in range(2)]
                assert [f[1] for f in again] == [f[1] for f in first]
                with pytest.raises(asyncio.TimeoutError):      # nothing else arrives: the server waits for our seq 1
                    await asyncio.wait_for(ws.receive(), timeout=2.0)
                async with s.get(f"http://127.0.0.1:{plat.server_port}/v1/status") as r:
                    st = await r.json()
                ch = st["sessions"][sid]["channel"]
                assert ch["duplicates"] == 1 and ch["retransmitted"] == 2 and ch["retransmit_refused"] == 0
                assert ch["expected"] == 1 and ch["next_seq"] == 3
                await ws.close()
            # the server burns RESERVED material on the disconnect and frees the slot
            for _ in range(100):
                async with s.get(f"http://127.0.0.1:{plat.server_port}/v1/status") as r:
                    st = await r.json()
                if st["sessions"][sid]["status"] in ("disconnected", "failed"):
                    break
                await asyncio.sleep(0.2)
            assert st["sessions"][sid]["status"] == "disconnected"
            assert st["sessions"][sid]["ledger_counts"].get("RESERVED", 0) == 0
            assert st["counters"]["disconnected"] >= 1

    run(go())


def test_cancel_leaves_no_reserved_material_and_resume_completes(plat):
    status, body = run(_post(plat, "/v1/generate", {"prompt": PROMPT_TEXT, "n_generate": 6, "policy": "B4_opencrab"}))
    assert status == 200
    sid = body["session_id"]

    async def cancel_after_first_token(ev):
        if ev["_type"] == "token" and ev["index"] == 0:
            st, b = await _post(plat, "/v1/cancel", {"session_id": sid})
            assert st == 200 and b["cancel"] is True

    events = run(_stream(plat, sid, on_event=cancel_after_first_token))
    done = [e for e in events if e["_type"] == "done"]
    assert done and done[0]["status"] == "cancelled"
    n_tok = len([e for e in events if e["_type"] == "token"])
    assert 1 <= n_tok < 6
    st = run(_get(plat, "/v1/status"))[1]["sessions"][sid]
    assert st["status"] == "cancelled"
    assert st["ledger_counts"]["RESERVED"] == 0 and st["ledger_counts"]["AVAILABLE"] == 0 and st["ledger_counts"]["BURNED"] > 0
    cancelled_ev = [e for e in events if e["_type"] == "status" and e.get("event") == "cancelled"]
    assert cancelled_ev and cancelled_ev[0]["wasted_bytes"] > 0

    async def server_session():
        async with aiohttp.ClientSession() as s:
            async with s.get(f"http://127.0.0.1:{plat.server_port}/v1/status") as r:
                return (await r.json())["sessions"][sid]

    def server_session_settled(timeout_s: float = 15.0) -> dict:
        # The sidecar's done/cancelled event is emitted once P1 holds its result; P0's runner
        # still writes its evidence record (fsync) afterwards, so the server status lags by
        # milliseconds. Poll until it leaves "running" (bounded), then assert on the final state.
        deadline = time.time() + timeout_s
        srv = run(server_session())
        while srv["status"] == "running" and time.time() < deadline:
            time.sleep(0.05)
            srv = run(server_session())
        return srv
    srv = server_session_settled()
    assert srv["status"] == "cancelled" and srv["ledger_counts"]["RESERVED"] == 0 and srv["ledger_counts"]["BURNED"] > 0
    # resume under the same session id: the remaining tokens arrive and equal the float reference
    status, body = run(_post(plat, "/v1/resume", {"session_id": sid}))
    assert status == 200 and body["resume_epoch"] == n_tok
    events2 = run(_stream(plat, sid, since=len(events)))
    done2 = [e for e in events2 if e["_type"] == "done"]
    assert done2 and done2[0]["status"] == "done" and done2[0]["tokens"] == 6
    all_tokens = [e["token_id"] for e in events if e["_type"] == "token"] + [e["token_id"] for e in events2 if e["_type"] == "token"]
    assert all_tokens == plat.ref[:6]
    srv = server_session_settled()
    assert srv["status"] == "done" and srv["ledger_counts"]["RESERVED"] == 0


def test_websocket_channel_flow_control_ordering_and_gap():
    """In-process: the adapter satisfies send/recv/exchange/stats, a bounded window blocks the sender until FLOW
    credits arrive, frames stay ordered and byte-exact, and a sequence gap closes the receiver."""
    import threading

    import numpy as np

    from opencrab2.errors import ChannelError
    from opencrab2.platform.model_server import WebSocketChannel

    loop = asyncio.new_event_loop()
    th = threading.Thread(target=loop.run_forever, name="ws-loop", daemon=True)
    th.start()
    ends: Dict[str, WebSocketChannel] = {}

    class FakeWS:
        """Delivers every binary frame straight into the peer adapter (loop thread, like aiohttp's reader)."""

        def __init__(self, peer: str) -> None:
            self.peer = peer
            self.sent: List[bytes] = []
            self.closed = False

        async def send_bytes(self, data: bytes) -> None:
            self.sent.append(bytes(data))
            ends[self.peer].on_binary(bytes(data))

        async def close(self) -> None:
            self.closed = True

    try:
        ws_a, ws_b = FakeWS("B"), FakeWS("A")
        A = WebSocketChannel(0, ws_a, loop, window=4, send_timeout_s=10.0, recv_timeout_s=10.0)
        B = WebSocketChannel(1, ws_b, loop, window=4, send_timeout_s=10.0, recv_timeout_s=10.0)
        ends["A"], ends["B"] = A, B
        payloads = [{"i": i, "x": np.arange(i + 1, dtype=np.uint64)} for i in range(10)]
        done = threading.Event()

        def sender() -> None:
            for p in payloads:
                A.send(p)
            done.set()

        t = threading.Thread(target=sender, daemon=True)
        t.start()
        time.sleep(0.5)
        assert A.stats.messages_sent == 4 and not done.is_set()       # the window bounds the in-flight frames
        got = [B.recv() for _ in range(10)]
        assert done.wait(5.0)
        assert [g["i"] for g in got] == list(range(10))
        assert all(np.array_equal(g["x"], p["x"]) for g, p in zip(got, payloads))
        assert B.flow_grants >= 1 and A.stats.messages_sent == 10 == B.stats.messages_recv
        assert A.stats.bytes_sent == B.stats.bytes_recv > 0
        assert all(_parse(f)[0] == KIND_DATA for f in ws_a.sent) and any(_parse(f)[0] == KIND_FLOW for f in ws_b.sent)
        # exchange: P0 sends then receives, P1 receives then sends; one round on both sides
        res: Dict[str, Any] = {}
        ta = threading.Thread(target=lambda: res.__setitem__("A", A.exchange(b"from-A")))
        tb = threading.Thread(target=lambda: res.__setitem__("B", B.exchange(b"from-B")))
        ta.start()
        tb.start()
        ta.join(5.0)
        tb.join(5.0)
        assert res == {"A": b"from-B", "B": b"from-A"} and A.stats.rounds == 1 == B.stats.rounds
        # a gap in the peer's sequence closes the channel (fail closed, never reorder)
        B.on_binary(_frame(99, b"\x08"))
        with pytest.raises(ChannelError) as ei:
            B.recv()
        assert ei.value.code == "CHANNEL_CLOSED" and "CHANNEL_WS_GAP" in str(ei.value)
        with pytest.raises(ChannelError):
            B.send(b"late")
        assert B.counters()["expected"] == 11
    finally:
        loop.call_soon_threadsafe(loop.stop)
        th.join(5.0)
        loop.close()


def test_resume_after_ttl_expiry_is_refused_in_process(tmp_path):
    """HELLO/RESUME binding checks of the model server without sockets: TTL expiry, version mismatch, busy slot."""
    from opencrab2.platform.model_server import ModelServer, _Session

    W, meta = make_tiny_weights(1, d=64, vocab=256)
    npz = S.save_model_npz(str(tmp_path / "tiny.npz"), W, meta)
    srv = ModelServer(f"npz:{npz}", 1, C, "127.0.0.1", 1, b"k" * 32, str(tmp_path / "state"), ttl_s=0.05)
    base = {"model_version": srv.model_version, "numeric_contract_hash": C.contract_hash(), "profile_id": PROFILE_ID}
    old = _Session.__new__(_Session)
    old.session_id, old.status = "old", "cancelled"
    old.created = old.last_active = time.time() - 10.0
    srv.sessions["old"] = old
    assert srv._reject({"type": "RESUME", "session_id": "old", **base}) == "SESSION_EXPIRED"
    assert old.status == "expired"
    assert srv._reject({"type": "RESUME", "session_id": "old", **base}) == "SESSION_EXPIRED"
    assert srv._reject({"type": "RESUME", "session_id": "never", **base}) == "SESSION_UNKNOWN"
    hello = {"type": "HELLO", "session_id": "new", "n_generate": 2, "prompt_len": 4, "n_layers": 1, "length_bucket": 8, **base}
    assert srv._reject(hello) is None
    assert srv._reject(dict(hello, model_version=srv.model_version + "-x")) == "OBJECT_VERSION_MISMATCH"
    assert srv._reject(dict(hello, numeric_contract_hash="00" * 32)) == "CONTRACT_MISMATCH"
    assert srv._reject(dict(hello, profile_id="other")) == "PROFILE_MISMATCH"
    assert srv._reject(dict(hello, n_layers=2)) == "SESSION_BAD_HELLO"
    assert srv._reject(dict(hello, prompt_len=0)) == "SESSION_BAD_HELLO"
    assert srv._reject({"type": "PING"}) == "SESSION_BAD_HELLO"
    running = _Session.__new__(_Session)
    running.session_id, running.status = "run", "running"
    running.created = running.last_active = time.time()
    srv.sessions["run"] = running
    assert srv._reject(hello) == "SESSION_BUSY"                       # one session at a time per server
    # a cancelled session expires lazily (TTL) on the next HELLO and frees the slot
    running.status = "cancelled"
    running.last_active = time.time() - 10.0
    assert srv._reject(hello) is None and running.status == "expired" and srv.counters["expired"] >= 1


def test_logs_and_server_records_carry_no_prompt(plat):
    for p in plat.procs:
        assert p.poll() is None
    logs = plat.logs()
    for name, text in logs.items():
        assert PROMPT_TEXT not in text and "Hell" not in text, name
        assert "seed" not in text.lower(), name
    assert "delivered recipient=0" in logs["dealer"] and "delivered recipient=1" in logs["dealer"]
    assert "status=done" in logs["server"] and "rejected code=OBJECT_VERSION_MISMATCH" in logs["server"]
    # the server's evidence records never carry the client's tokens; the sidecar's do not either (record_tokens=False)
    for rel in (("server", "server-records.jsonl"), ("sidecar", "client-records.jsonl")):
        path = plat.tmp / rel[0] / rel[1]
        assert path.is_file(), path
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            assert "generated_tokens" not in rec.get("extra", {}) and rec["schema"] == "opencrab-native-evidence-v2"
            assert rec["dealer_mode"] == MODE_ISOLATED


def test_second_generate_on_the_same_stack_is_a_warm_request(plat):
    """Server and sidecar keep installed objects and the ledger per (model_version, client_key_domain)."""
    async def server_status():
        async with aiohttp.ClientSession() as s:
            async with s.get(f"http://127.0.0.1:{plat.server_port}/v1/status") as r:
                return await r.json()

    def generate(n: int):
        status, body = run(_post(plat, "/v1/generate", {"prompt": PROMPT_TEXT, "n_generate": n, "policy": "B4_opencrab"}))
        assert status == 200
        sid = body["session_id"]
        events = run(_stream(plat, sid))
        deadline = time.time() + 15.0
        while run(server_status())["sessions"][sid]["status"] == "running" and time.time() < deadline:
            time.sleep(0.05)
        return sid, events

    _sid_a, ev_a = generate(2)                       # leaves a usable stack whatever the earlier tests did
    done_a = [e for e in ev_a if e["_type"] == "done"]
    assert done_a and done_a[0]["status"] == "done"
    sid_b, ev_b = generate(3)
    done_b = [e for e in ev_b if e["_type"] == "done"]
    assert done_b and done_b[0]["status"] == "done" and done_b[0]["tokens"] == 3
    assert done_b[0]["warm_start"] is True and done_b[0]["T_warm_basis"] == "measured-warm-request"
    assert done_b[0]["T_cold_s"] is None and done_b[0]["T_warm_s"] > 0 and done_b[0]["install_s"] == 0.0
    assert done_b[0]["security_eligible"] is True
    epochs = [e for e in ev_b if e["_type"] == "epoch"]
    assert len(epochs) == 3 and all(not e["installed_now"] for e in epochs)
    assert [e["token_id"] for e in ev_b if e["_type"] == "token"] == plat.ref[:3]
    side = run(_get(plat, "/v1/status"))[1]
    assert side["sessions"][sid_b]["warm"] is True and side["counters"]["warm_sessions"] >= 1
    assert side["stacks"] and side["stacks"][-1]["usable"] is True and "lm_head" in side["stacks"][-1]["installed"]
    assert side["sessions"][sid_b]["ledger_counts"]["RESERVED"] == 0
    srv = run(server_status())
    assert srv["sessions"][sid_b]["warm"] is True and srv["counters"]["warm_sessions"] >= 1
    assert srv["stacks"][-1]["sessions"] == side["stacks"][-1]["sessions"] >= 2
