"""End-to-end tests for opencrab2.platform.dealer_service.

A dealer process is started as a subprocess and two authenticated client
connections provision the same request ids; the delivered halves are ingested
into ledger-backed DealerClients and the correlations are reconstructed. The
dialogue itself (client-first HELLO) is exercised frame by frame over a raw
channel.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import pathlib
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from opencrab2.errors import ChannelError, MaterialError
from opencrab2.highlevel.channel import dial_channel
from opencrab2.highlevel.contract import DEFAULT_CONTRACT
from opencrab2.highlevel.ledger import MaterialLedger
from opencrab2.highlevel import dealer as dealer_mod
from opencrab2.highlevel.dealer import (HEADER_KEYS, MODE_INSECURE, MODE_ISOLATED, Dealer, DealerClient,
                                        material_estimate, material_estimate_detail)
from opencrab2.lowlevel import drelu, hmac_auth, ring
from opencrab2.platform import dealer_service as svc
from opencrab2.platform.dealer_service import (DealerService, DealerServiceClient, auth_tag, load_keys, main,
                                               params_fingerprint, requests_from_detail, requests_from_estimate,
                                               write_keys)

PYDIR = str(pathlib.Path(__file__).resolve().parents[1])
C = DEFAULT_CONTRACT
CH = C.contract_hash()
KEYS = {0: bytes(range(1, 33)), 1: bytes(range(101, 133))}
PROFILE = "SH2P-TrustedOffline-v1"


def binding(setup_id="su", model_version="v0"):
    return {"session_id": "sess", "setup_id": setup_id, "model_version": model_version,
            "numeric_contract_hash": CH, "profile_id": PROFILE}


@pytest.fixture
def keys_file(tmp_path):
    p = tmp_path / "keys.json"
    write_keys(str(p), KEYS)
    return str(p)


@pytest.fixture
def clients(tmp_path):
    out = []
    for p in (0, 1):
        L = MaterialLedger(str(tmp_path / f"ledger{p}.jsonl"), binding())
        out.append(DealerClient(p, L, KEYS[p], C))
    yield out
    for c in out:
        c.ledger.close()


def _spawn(keys_file, *extra):
    env = dict(os.environ)
    env["PYTHONPATH"] = PYDIR + os.pathsep + env.get("PYTHONPATH", "")
    env.pop(dealer_mod.ENV_ALLOW_INSECURE, None)
    proc = subprocess.Popen(
        [sys.executable, "-m", "opencrab2.platform.dealer_service", "--listen", "127.0.0.1:0", "--keys", keys_file,
         "--contract", "NC-v2-f16-l40", "--setup-id", "su", "--model-version", "v0", *extra],
        cwd=PYDIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    line = proc.stdout.readline()
    if not line:
        err = proc.stderr.read()
        proc.kill()
        pytest.fail(f"dealer service did not start: {err}")
    tok = line.split()
    assert tok[0] == "LISTENING" and tok[1] == "127.0.0.1"
    return proc, int(tok[2])


@pytest.fixture
def service_proc(keys_file):
    proc, port = _spawn(keys_file)
    yield ("127.0.0.1", port, proc)
    proc.kill()
    try:
        proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        pass


@pytest.fixture
def os_service():
    """In-process service on an OS-seeded isolated dealer (the eligible configuration)."""
    dealer = Dealer(C, setup_id="su", model_version="v0")
    service = DealerService(dealer, KEYS, "127.0.0.1", 0)
    port = service.start()
    yield service, port
    service.shutdown()


def connect(host, port, recipient, key=None, expect_hash=CH, **kw):
    c = DealerServiceClient(host, port, recipient, KEYS[recipient] if key is None else key, expect_hash, retries=200, **kw)
    c.connect()
    return c


def raw_authenticated(port, recipient, setup_id="su"):
    """Speak the dialogue by hand up to AUTH_OK; returns (channel, challenge frame, auth_ok frame)."""
    ch = dial_channel("127.0.0.1", port, party=1, retries=200)
    ch.send({"type": "HELLO", "recipient": recipient, "setup_id": setup_id, "contract_hash": CH})
    chal = ch.recv()
    assert chal["type"] == "CHALLENGE"
    ch.send({"type": "AUTH", "tag": hmac.new(KEYS[recipient], chal["nonce"], hashlib.sha256).hexdigest()})
    ok = ch.recv()
    return ch, chal, ok


# --------------------------------------------------------------------------
# the dialogue, frame by frame
# --------------------------------------------------------------------------
def test_dialogue_frame_sequence_is_client_first(os_service, clients):
    service, port = os_service
    c0, c1 = clients
    ch = dial_channel("127.0.0.1", port, party=1, retries=200)
    try:
        # the dialogue is client-first: the server says nothing before HELLO
        ch.sock.settimeout(0.3)
        with pytest.raises(ChannelError):
            ch.recv()
        ch.sock.settimeout(None)
        # client: HELLO{recipient, setup_id, contract_hash}
        ch.send({"type": "HELLO", "recipient": 0, "setup_id": "su", "contract_hash": CH})
        # server: CHALLENGE{nonce}
        chal = ch.recv()
        assert chal["type"] == "CHALLENGE" and isinstance(chal["nonce"], bytes) and len(chal["nonce"]) == 32
        assert chal["dealer_mode"] == MODE_ISOLATED and chal["insecure"] is False
        assert chal["setup_id"] == "su" and chal["model_version"] == "v0"
        assert chal["numeric_contract_hash"] == CH and chal["profile_id"] == PROFILE
        # client: AUTH{tag = HMAC(key_recipient, nonce)}
        tag = hmac.new(KEYS[0], chal["nonce"], hashlib.sha256).hexdigest()
        assert tag == auth_tag(KEYS[0], chal["nonce"]) == auth_tag(KEYS[0], chal["nonce"], 0)
        ch.send({"type": "AUTH", "tag": tag})
        assert ch.recv() == {"type": "AUTH_OK", "recipient": 0}
        # client: REQ{material_type, params, count}; server: BUNDLE{header, tag} then ONE bytes frame
        ch.send({"type": "REQ", "material_type": "beaver", "params": {"shape": [4]}, "count": 1})
        bundle = ch.recv()
        assert set(bundle) == {"type", "header", "tag", "remaining"} and bundle["type"] == "BUNDLE"
        assert "payload" not in bundle and bundle["remaining"] == 0
        header, tg = bundle["header"], bundle["tag"]
        assert set(HEADER_KEYS) <= set(header) and header["parts"] == ["a", "b", "c"]
        assert header["recipient"] == 0 and header["material_type"] == "beaver" and header["gate_id"] == "*"
        assert header["dealer_mode"] == MODE_ISOLATED and header["insecure"] is False
        payload = ch.recv()
        assert isinstance(payload, bytes) and len(payload) == header["bytes"] == 96
        assert tg == hmac_auth.tag_with_payload(KEYS[0], header, payload)
        c0.ingest_frame(bundle, payload)
        # the peer's identical REQ (no request_id) receives the other half of the same pair
        ch1, _chal1, ok1 = raw_authenticated(port, 1)
        assert ok1 == {"type": "AUTH_OK", "recipient": 1}
        ch1.send({"type": "REQ", "material_type": "beaver", "params": {"shape": [4]}, "count": 1})
        b1 = ch1.recv()
        p1 = ch1.recv()
        assert b1["header"]["material_id"] == header["material_id"] and b1["header"]["recipient"] == 1
        c1.ingest_frame(b1, p1)
        x0, x1 = c0.take_beaver((4,)), c1.take_beaver((4,))
        assert np.array_equal(ring.mul(ring.add(x0.a, x1.a), ring.add(x0.b, x1.b)), ring.add(x0.c, x1.c))
        assert c0.security_eligible and c1.security_eligible
        # a request error is one ERROR frame and nothing else; the connection stays usable
        ch.send({"type": "REQ", "material_type": "teleport", "params": {}, "count": 1})
        err = ch.recv()
        assert err["type"] == "ERROR" and err["code"] == "DEALER_PARAMS" and isinstance(err["message"], str)
        ch.send({"type": "STATUS"})
        st = ch.recv()
        assert st["type"] == "STATUS" and st["bundles_sent"] == 2 and st["pending_halves"] == 0
        # an unknown frame type ends the session
        ch.send({"type": "PROVISION", "request_id": "old", "material_type": "beaver", "params": {"shape": [4]}})
        assert ch.recv()["code"] == "DEALER_BAD_MESSAGE"
        with pytest.raises(ChannelError):
            ch.recv()
        ch1.send({"type": "BYE"})
        ch1.close()
    finally:
        ch.close()


def test_handshake_refusals_frame_level(os_service):
    service, port = os_service

    def first_reply(*frames):
        ch = dial_channel("127.0.0.1", port, party=1, retries=200)
        try:
            out = None
            for f in frames:
                ch.send(f)
                out = ch.recv()
            with pytest.raises(ChannelError):      # every refusal closes the connection
                ch.recv()
            return out
        finally:
            ch.close()

    hello = {"type": "HELLO", "recipient": 0, "setup_id": "su", "contract_hash": CH}
    assert first_reply({"type": "REQ", "material_type": "beaver", "params": {"shape": [4]}, "count": 1})["code"] == "DEALER_BAD_MESSAGE"
    assert first_reply({"type": "AUTH", "tag": "00" * 32})["code"] == "DEALER_BAD_MESSAGE"
    assert first_reply(b"not a dict")["code"] == "DEALER_BAD_MESSAGE"
    for bad_recipient in (2, -1, "0", True, None):
        assert first_reply({**hello, "recipient": bad_recipient})["code"] == "DEALER_UNKNOWN_RECIPIENT"
    assert first_reply({**hello, "setup_id": "other"})["code"] == "SETUP_MISMATCH"
    assert first_reply({**hello, "contract_hash": "00" * 32})["code"] == "CONTRACT_MISMATCH"
    assert first_reply({**hello, "contract_hash": ""})["code"] == "CONTRACT_MISMATCH"      # an empty hash is a mismatch
    assert first_reply({"type": "HELLO", "recipient": 0, "setup_id": "su"})["code"] == "DEALER_BAD_MESSAGE"
    assert first_reply({**hello, "setup_id": 7})["code"] == "DEALER_BAD_MESSAGE"
    # the contract hash is checked before a challenge is issued: no nonce is revealed on a mismatch
    assert "nonce" not in first_reply({**hello, "contract_hash": "ff" * 32})
    # wrong tags: other key, garbage, missing, a REQ instead of AUTH
    assert first_reply(hello, {"type": "AUTH", "tag": "00" * 32})["code"] == "AUTH_FAILED"
    assert first_reply(hello, {"type": "AUTH", "tag": 5})["code"] == "AUTH_FAILED"
    assert first_reply(hello, {"type": "AUTH"})["code"] == "AUTH_FAILED"
    assert first_reply(hello, {"type": "REQ", "material_type": "beaver", "params": {}, "count": 1})["code"] == "DEALER_BAD_MESSAGE"
    # a tag computed with the nonce of ANOTHER connection is refused (fresh challenge per connection)
    ch_a = dial_channel("127.0.0.1", port, party=1, retries=200)
    ch_b = dial_channel("127.0.0.1", port, party=1, retries=200)
    try:
        ch_a.send(hello)
        ch_b.send(hello)
        na, nb = ch_a.recv()["nonce"], ch_b.recv()["nonce"]
        assert na != nb
        ch_b.send({"type": "AUTH", "tag": auth_tag(KEYS[0], na)})
        assert ch_b.recv()["code"] == "AUTH_FAILED"
        ch_a.send({"type": "AUTH", "tag": auth_tag(KEYS[0], na)})
        assert ch_a.recv()["type"] == "AUTH_OK"
    finally:
        ch_a.close()
        ch_b.close()
    assert service.auth_failures >= 9
    # an empty setup id means "not pinned": the client adopts the dealer's declared setup
    ch, chal, ok = raw_authenticated(port, 1, setup_id="")
    assert ok["type"] == "AUTH_OK" and chal["setup_id"] == "su"
    ch.close()


def test_client_checks_the_challenge_before_authenticating(os_service, monkeypatch):
    service, port = os_service
    # pinned setup id: accepted when equal, refused by the service when different
    s = connect("127.0.0.1", port, 0, setup_id="su")
    assert s.binding == {"setup_id": "su", "model_version": "v0", "numeric_contract_hash": CH,
                         "profile_id": PROFILE, "dealer_mode": MODE_ISOLATED}
    assert s.dealer_insecure is False
    s.close()
    with pytest.raises(MaterialError) as ei:
        connect("127.0.0.1", port, 0, setup_id="other")
    assert ei.value.code == "SETUP_MISMATCH"
    # a client without a contract hash cannot connect (fail closed)
    with pytest.raises(MaterialError) as ei:
        connect("127.0.0.1", port, 0, expect_hash=None)
    assert ei.value.code == "CONTRACT_MISMATCH"
    # a malformed challenge is never signed
    signed = []
    monkeypatch.setattr(svc, "auth_frame", lambda key, nonce: signed.append(nonce) or {"type": "AUTH", "tag": "00" * 32})
    monkeypatch.setattr(svc, "parse_challenge", dealer_mod.parse_challenge)
    lsock_port = _fake_server([{"type": "CHALLENGE", "nonce": b"short", "dealer_mode": MODE_ISOLATED}])
    with pytest.raises(MaterialError) as ei:
        connect("127.0.0.1", lsock_port, 0)
    assert ei.value.code == "DEALER_BAD_MESSAGE" and signed == []


def _fake_server(replies):
    """A one-connection scripted peer: after each received frame it sends the next scripted reply."""
    from opencrab2.highlevel.channel import accept_channel, open_listener
    lsock, port = open_listener("127.0.0.1", 0)

    def run():
        try:
            ch = accept_channel(lsock, 0, timeout_s=10.0)
        except ChannelError:
            return
        try:
            for r in replies:
                ch.recv()
                ch.send(r)
            time.sleep(0.2)
        except ChannelError:
            pass
        finally:
            ch.close()

    threading.Thread(target=run, daemon=True).start()
    return port


def test_client_refuses_insecure_dealers_and_bundles_by_default(monkeypatch):
    # what a platform party process sees (no in-process dealer, no environment switch)
    monkeypatch.setitem(dealer_mod._TEST_ONLY_STATE, "in_process_insecure_dealer", False)
    monkeypatch.delenv(dealer_mod.ENV_ALLOW_INSECURE, raising=False)
    nonce = bytes(range(32))
    chal = {"type": "CHALLENGE", "nonce": nonce, "setup_id": "su", "model_version": "v0",
            "numeric_contract_hash": CH, "profile_id": PROFILE}
    for declared in ({"dealer_mode": MODE_INSECURE, "insecure": True}, {"dealer_mode": MODE_INSECURE, "insecure": False},
                     {"dealer_mode": MODE_ISOLATED, "insecure": True}, {"dealer_mode": "", "insecure": False},
                     {"insecure": False}, {"dealer_mode": MODE_ISOLATED, "insecure": "no"}):
        port = _fake_server([{**chal, **declared}, {"type": "AUTH_OK", "recipient": 0}])
        with pytest.raises(MaterialError) as ei:
            connect("127.0.0.1", port, 0)
        assert ei.value.code == "DEALER_INSECURE_MODE"
    # the explicit test-only switch admits it and reports the mode
    port = _fake_server([{**chal, "dealer_mode": MODE_INSECURE, "insecure": True}, {"type": "AUTH_OK", "recipient": 0}])
    s = connect("127.0.0.1", port, 0, allow_insecure=True)
    assert s.binding["dealer_mode"] == MODE_INSECURE and s.dealer_insecure is True
    s._drop()
    # a dealer that advertises the eligible mode but streams a bundle labelled insecure is refused as well
    d = Dealer(C, np.random.default_rng(1), MODE_ISOLATED, setup_id="su", model_version="v0")      # seeded: insecure label
    monkeypatch.setitem(dealer_mod._TEST_ONLY_STATE, "in_process_insecure_dealer", False)
    (h0, p0, _, _), = d.make_beaver((2,), 1)
    hh, pp, tg = d.seal(0, h0, p0, KEYS[0])
    assert hh["insecure"] is True and hh["dealer_mode"] == MODE_ISOLATED
    port = _fake_server([{**chal, "dealer_mode": MODE_ISOLATED, "insecure": False}, {"type": "AUTH_OK", "recipient": 0},
                         {"type": "BUNDLE", "header": hh, "tag": tg, "remaining": 0}])
    s = connect("127.0.0.1", port, 0)
    with pytest.raises((MaterialError, ChannelError)) as ei:
        s.provision("r", "beaver", {"shape": [2]}, 1)
    assert ei.value.code in ("BUNDLE_INSECURE_MODE", "CHANNEL_CLOSED")
    # with the payload frame present the refusal is the insecure label
    lport = _fake_server_bundle(chal, hh, tg, pp)
    s = connect("127.0.0.1", lport, 0)
    with pytest.raises(MaterialError) as ei:
        s.provision("r", "beaver", {"shape": [2]}, 1)
    assert ei.value.code == "BUNDLE_INSECURE_MODE" and s.ch is None


def _fake_server_bundle(chal, header, tag, payload):
    from opencrab2.highlevel.channel import accept_channel, open_listener
    lsock, port = open_listener("127.0.0.1", 0)

    def run():
        try:
            ch = accept_channel(lsock, 0, timeout_s=10.0)
        except ChannelError:
            return
        try:
            ch.recv()
            ch.send({**chal, "dealer_mode": MODE_ISOLATED, "insecure": False})
            ch.recv()
            ch.send({"type": "AUTH_OK", "recipient": 0})
            ch.recv()
            ch.send({"type": "BUNDLE", "header": header, "tag": tag, "remaining": 0})
            ch.send(payload)
            time.sleep(0.2)
        except ChannelError:
            pass
        finally:
            ch.close()

    threading.Thread(target=run, daemon=True).start()
    return port


def test_multi_bundle_requests_drain_one_bundle_per_req(os_service, clients):
    service, port = os_service
    c0, c1 = clients
    ch, _chal, _ok = raw_authenticated(port, 0)
    try:
        req = {"type": "REQ", "material_type": "beaver", "params": {"shape": [8]}, "count": 3, "request_id": "multi"}
        ids = []
        for expect_remaining in (2, 1, 0):
            ch.send(req)
            b = ch.recv()
            p = ch.recv()
            assert b["type"] == "BUNDLE" and b["remaining"] == expect_remaining and isinstance(p, bytes)
            c0.ingest_frame(b, p)
            ids.append(b["header"]["material_id"])
        assert len(set(ids)) == 3
        ch.send(req)                                   # drained for this recipient
        assert ch.recv()["code"] == "DEALER_ALREADY_DELIVERED"
        ch.send({**req, "count": 2})                   # same id, other parameters
        assert ch.recv()["code"] == "DEALER_REQUEST_CONFLICT"
        assert service.pending_halves() == 3
        # the client loop fetches the peer's three halves with one call
        s1 = connect("127.0.0.1", port, 1)
        got = s1.provision("multi", "beaver", {"shape": [8]}, count=3)
        assert [h["material_id"] for h, _p, _t in got] == ids and s1.received_bundles == 3
        c1.ingest_many(got)
        s1.close()
        assert service.pending_halves() == 0
        with service._lock:
            assert service._requests == {} and service._fifo == {}
        x0, x1 = c0.take_beaver((24,)), c1.take_beaver((24,))
        assert np.array_equal(ring.mul(ring.add(x0.a, x1.a), ring.add(x0.b, x1.b)), ring.add(x0.c, x1.c))
        # bad REQ fields get stable codes and leave the connection usable
        for bad in ({"type": "REQ", "material_type": "beaver", "params": {"shape": [4]}, "count": 0},
                    {"type": "REQ", "material_type": "beaver", "params": {"shape": [4]}, "count": True},
                    {"type": "REQ", "material_type": "beaver", "params": [4], "count": 1},
                    {"type": "REQ", "material_type": 7, "params": {}, "count": 1},
                    {"type": "REQ", "material_type": "beaver", "params": {}, "count": 1},
                    {"type": "REQ", "material_type": "beaver", "params": {"shape": [4]}, "count": 1, "request_id": ""},
                    {"type": "REQ", "material_type": "beaver", "params": {"shape": [4]}, "count": 1, "gate_id": 3}):
            ch.send(bad)
            assert ch.recv()["code"] == "DEALER_BAD_REQUEST"
        ch.send({"type": "REQ", "material_type": "beaver", "params": {"shape": [2]}, "count": 1})
        assert ch.recv()["type"] == "BUNDLE" and isinstance(ch.recv(), bytes)
    finally:
        ch.close()


def test_pairing_without_request_id_is_first_come_first_served(os_service, clients):
    service, port = os_service
    c0, c1 = clients
    a, _, _ = raw_authenticated(port, 0)
    b, _, _ = raw_authenticated(port, 1)
    try:
        req = {"type": "REQ", "material_type": "drelu", "params": {"M": 4, "with_product": True}, "count": 1, "gate_id": "g"}
        got = {0: [], 1: []}
        for ch, r in ((a, 0), (a, 0), (b, 1), (b, 1)):        # recipient 0 asks twice before recipient 1
            ch.send(req)
            frame = ch.recv()
            payload = ch.recv()
            got[r].append(frame["header"]["material_id"])
            (c0 if r == 0 else c1).ingest_frame(frame, payload)
            assert frame["header"]["gate_id"] == "g" and frame["header"]["parts"] == ["drelu_material"]
        assert got[0] == got[1] and len(set(got[0])) == 2
        assert service.pending_halves() == 0
        d0, d1 = c0.take_drelu(8), c1.take_drelu(8)
        xr = np.array([-3.0, 2.0, 0.0, -0.5, 1.0, 5.0, -7.0, 0.25])
        y = ring.add(ring.encode(xr, C), ring.add(d0.r_share, d1.r_share))
        s0, rb0 = drelu.drelu_eval(d0, y)
        s1, rb1 = drelu.drelu_eval(d1, y)
        assert np.array_equal(ring.add(s0, s1), (xr >= 0).astype(np.uint64))
        # a different gate id is a different pairing key
        b.send({**req, "gate_id": "h"})
        other = b.recv()
        b.recv()
        assert other["header"]["material_id"] not in got[0] and service.pending_halves() == 1
    finally:
        a.close()
        b.close()


def test_params_fingerprint_and_wide_values(os_service, clients):
    assert params_fingerprint({"b": [1, 2], "a": "x"}) == params_fingerprint({"a": "x", "b": [1, 2]})
    assert params_fingerprint({"shape": [4]}) != params_fingerprint({"shape": [5]})
    v = np.array([1, 2 ** 64 - 1], np.uint64)
    assert params_fingerprint({"values": v}) == params_fingerprint({"values": v.copy()})
    assert params_fingerprint({"values": v}) != params_fingerprint({"values": v[::-1]})
    with pytest.raises(MaterialError):
        params_fingerprint({"x": float("nan")})
    # ring values above 2^63 do not fit the i64 int of the codec: the client sends a uint64 array
    service, port = os_service
    c0, c1 = clients
    values = [0, 1, 2 ** 63, 2 ** 64 - 1]
    # the eligible (isolated, OS-keyed) dealer never accepts a plaintext vector: stable code, connection usable
    s = connect("127.0.0.1", port, 0)
    with pytest.raises(MaterialError) as ei:
        s.provision_into(c0, "vec", "vector_share", {"name": "bias", "values": values})
    assert ei.value.code == "DEALER_PLAINTEXT_REFUSED"
    assert s.provision_into(c0, "bv", "beaver", {"shape": [4]}) == 1 and c0.security_eligible
    s.close()
    with pytest.raises(MaterialError) as ei:
        service.dealer.make("vector_share", {"name": "bias", "values": values})
    assert ei.value.code == "DEALER_PLAINTEXT_REFUSED"
    # the insecure unit-test dealer still serves the type (wide values travel as a uint64 array)
    insecure = DealerService(Dealer(C, np.random.default_rng(3), MODE_INSECURE, setup_id="su", model_version="v0"),
                             KEYS, "127.0.0.1", 0)
    iport = insecure.start()
    try:
        for party, c in ((0, c0), (1, c1)):
            c.allow_insecure = True
            s = connect("127.0.0.1", iport, party)
            assert s.provision_into(c, "vec", "vector_share", {"name": "bias", "values": values}) == 1
            s.close()
        assert ring.add(c0.take_vector_share("bias"), c1.take_vector_share("bias")).tolist() == values
        assert c0.security_eligible is False
    finally:
        insecure.shutdown()


# --------------------------------------------------------------------------
# subprocess end to end
# --------------------------------------------------------------------------
def test_two_parties_provision_the_same_request_ids(service_proc, clients):
    host, port, _proc = service_proc
    c0, c1 = clients
    s0 = connect(host, port, 0)
    s1 = connect(host, port, 1)
    try:
        assert s0.binding["setup_id"] == "su" and s0.binding["model_version"] == "v0"
        assert s0.binding["dealer_mode"] == MODE_ISOLATED == s1.binding["dealer_mode"]
        assert s0.binding["numeric_contract_hash"] == CH and s0.binding["profile_id"] == PROFILE
        assert s0.dealer_insecure is False
        # Beaver: P1 asks first, P0 later; both receive their own halves of the same pairs
        n1 = s1.provision_into(c1, "e0:beaver", "beaver", {"shape": [64]}, count=2, gate_id="mlp")
        n0 = s0.provision_into(c0, "e0:beaver", "beaver", {"shape": [64]}, count=2, gate_id="mlp")
        assert n0 == n1 == 2
        assert c0.ledger.ids_in_state("AVAILABLE") == c1.ledger.ids_in_state("AVAILABLE")
        b0 = c0.take_beaver((100,))
        b1 = c1.take_beaver((100,))
        assert np.array_equal(ring.mul(ring.add(b0.a, b1.a), ring.add(b0.b, b1.b)), ring.add(b0.c, b1.c))
        # DReLU
        s0.provision_into(c0, "e0:drelu", "drelu", {"M": 8, "with_product": True}, count=1)
        s1.provision_into(c1, "e0:drelu", "drelu", {"M": 8, "with_product": True}, count=1)
        d0 = c0.take_drelu(8)
        d1 = c1.take_drelu(8)
        xr = np.array([-3.0, 2.0, 0.0, -0.5, 1.0, 5.0, -7.0, 0.25])
        x = ring.encode(xr, C)
        y = ring.add(x, ring.add(d0.r_share, d1.r_share))
        s_0, _ = drelu.drelu_eval(d0, y)
        s_1, _ = drelu.drelu_eval(d1, y)
        assert np.array_equal(ring.add(s_0, s_1), (xr >= 0).astype(np.uint64))
        # matrix triple and wmask mask + calls
        for s, c in ((s0, c0), (s1, c1)):
            s.provision_into(c, "e0:mt", "matrix_triple", {"shape_a": [2, 3], "shape_b": [3, 2]}, count=1)
            s.provision_into(c, "e0:mask", "wmask_mask", {"mat_id": "fc", "shape": [3, 2]}, count=1)
            s.provision_into(c, "e0:calls", "wmask_call", {"mat_id": "fc", "n_rows": 2}, count=2)
            s.provision_into(c, "e0:trunc", "trunc_faithful", {"M": 3}, count=1)
            s.provision_into(c, "e0:r2f", "ring_to_field", {"M": 2}, count=1)
        t0 = c0.take_matrix_triple((2, 3), (3, 2))
        t1 = c1.take_matrix_triple((2, 3), (3, 2))
        assert np.array_equal(ring.ring_matmul(ring.add(t0.a, t1.a), ring.add(t0.b, t1.b)), ring.add(t0.c, t1.c))
        B = ring.add(c0.wmask_mask_share("fc"), c1.wmask_mask_share("fc"))
        w0 = c0.take_wmask_call("fc", 2)
        w1 = c1.take_wmask_call("fc", 2)
        assert np.array_equal(ring.ring_matmul(ring.add(w0.a, w1.a), B), ring.add(w0.c, w1.c))
        assert c0.stock()["trunc_faithful"] == (3, 3 * 2064) and c0.stock()["ring_to_field"] == (2, 2 * 3272)
        # status reflects production
        st = s0.status()
        assert st["type"] == "STATUS" and st["produced"]["beaver"][0] == 128 and st["dealer_mode"] == MODE_ISOLATED
        assert st["insecure"] is False and st["pending_halves"] == 0
        assert c0.security_eligible and c1.security_eligible
    finally:
        s0.close()
        s1.close()


def test_rejections_over_the_wire(service_proc, clients):
    host, port, _proc = service_proc
    c0, c1 = clients
    # wrong key
    with pytest.raises(MaterialError) as ei:
        connect(host, port, 0, key=b"\x99" * 32)
    assert ei.value.code == "AUTH_FAILED"
    # key of the other recipient
    with pytest.raises(MaterialError) as ei:
        connect(host, port, 0, key=KEYS[1])
    assert ei.value.code == "AUTH_FAILED"
    # wrong contract expectation
    with pytest.raises(MaterialError) as ei:
        connect(host, port, 0, expect_hash="00" * 32)
    assert ei.value.code == "CONTRACT_MISMATCH"
    # wrong setup
    with pytest.raises(MaterialError) as ei:
        connect(host, port, 0, setup_id="another-setup")
    assert ei.value.code == "SETUP_MISMATCH"
    s0 = connect(host, port, 0)
    s1 = connect(host, port, 1)
    try:
        bundles0 = s0.provision("r1", "beaver", {"shape": [4]}, count=1)
        assert len(bundles0) == 1 and bundles0[0][0]["recipient"] == 0
        # a bundle streamed to recipient 0 cannot be ingested by party 1 (foreign key, wrong recipient)
        with pytest.raises(MaterialError) as ei:
            c1.ingest(*bundles0[0])
        assert ei.value.code == "BUNDLE_BAD_TAG"
        c0.ingest(*bundles0[0])
        # duplicate delivery to the same recipient is refused
        with pytest.raises(MaterialError) as ei:
            s0.provision("r1", "beaver", {"shape": [4]}, count=1)
        assert ei.value.code == "DEALER_ALREADY_DELIVERED"
        # conflicting parameters under the same request id
        with pytest.raises(MaterialError) as ei:
            s1.provision("r1", "beaver", {"shape": [5]}, count=1)
        assert ei.value.code == "DEALER_REQUEST_CONFLICT"
        with pytest.raises(MaterialError) as ei:
            s1.provision("r1", "beaver", {"shape": [4]}, count=2)
        assert ei.value.code == "DEALER_REQUEST_CONFLICT"
        # the matching request succeeds and the pair reconstructs
        bundles1 = s1.provision("r1", "beaver", {"shape": [4]}, count=1)
        c1.ingest(*bundles1[0])
        assert bundles1[0][0]["material_id"] == bundles0[0][0]["material_id"]
        b0 = c0.take_beaver((4,))
        b1 = c1.take_beaver((4,))
        assert np.array_equal(ring.mul(ring.add(b0.a, b1.a), ring.add(b0.b, b1.b)), ring.add(b0.c, b1.c))
        # bad requests get stable codes and the connection stays usable
        with pytest.raises(MaterialError) as ei:
            s0.provision("r2", "teleport", {}, count=1)
        assert ei.value.code == "DEALER_PARAMS"
        with pytest.raises(MaterialError) as ei:
            s0.provision("r3", "beaver", {"shape": [0]}, count=1)
        assert ei.value.code == "DEALER_PARAMS"
        with pytest.raises(MaterialError) as ei:
            s0.provision("r4", "wmask_call", {"mat_id": "none", "n_rows": 1}, count=1)
        assert ei.value.code == "WMASK_MASK_MISSING"
        with pytest.raises(MaterialError) as ei:
            s0.provision("r5", "beaver", {"shape": [4]}, count=0)
        assert ei.value.code == "DEALER_BAD_REQUEST"
        with pytest.raises(MaterialError) as ei:
            s0.provision("r5", "drelu", {"with_product": True}, count=1)          # missing M
        assert ei.value.code == "DEALER_BAD_REQUEST"
        assert len(s0.provision("r6", "beaver", {"shape": [4]}, count=1)) == 1
    finally:
        s0.close()
        s1.close()


def test_service_logs_carry_no_payload_bytes(service_proc, clients):
    host, port, proc = service_proc
    s0 = connect(host, port, 0)
    bundles = s0.provision("x", "beaver", {"shape": [16]}, count=1)
    with pytest.raises(MaterialError):
        connect(host, port, 1, key=b"\x42" * 32)
    s0.close()
    proc.kill()
    _out, err = proc.communicate(timeout=10)
    assert "delivered recipient=0 request=x type=beaver bundles=1 bytes=384" in err
    assert "authenticated recipient=0" in err and "authentication failed" in err
    assert bundles[0][1].hex()[:32] not in err
    assert bundles[0][2] not in err                           # no bundle tag
    assert KEYS[0].hex() not in err and KEYS[1].hex() not in err and ("42" * 32) not in err
    assert "seed" not in err.lower() and "nonce" not in err.lower()


# --------------------------------------------------------------------------
# in-process service: estimate driven provisioning
# --------------------------------------------------------------------------
def test_in_process_service_with_estimate_requests(clients):
    dealer = Dealer(C, np.random.default_rng(42), setup_id="su", model_version="v0")
    service = DealerService(dealer, KEYS, "127.0.0.1", 0)
    port = service.start()
    c0, c1 = clients
    try:
        est = material_estimate("softmax_rows", (2, 8), C, k_max=2)
        detail = material_estimate_detail("softmax_rows", (2, 8), C, k_max=2)
        reqs = requests_from_detail(detail, "epoch3:softmax")
        assert {r["material_type"] for r in reqs} == set(est)
        assert len(reqs) == len(detail) == 4        # drelu P=2, drelu P=1, kary_select, beaver
        for party, c in ((0, c0), (1, c1)):
            s = connect("127.0.0.1", port, party)      # the seeded dealer lives in this process: harness detected
            assert s.dealer_insecure is True and s.binding["dealer_mode"] == MODE_ISOLATED
            for r in reqs:
                s.provision_into(c, r["request_id"], r["material_type"], r["params"], r["count"])
            s.close()
        st = c0.stock()
        for mt, (count, nbytes) in est.items():
            assert st[mt][0] == count and st[mt][1] == nbytes      # exact, byte for byte (charged sizes)
        assert c0.ledger.counts()["AVAILABLE"] == len(reqs)
        # a caller-seeded dealer never yields eligible material, whatever its mode says
        assert not c0.security_eligible and not c1.security_eligible
        # the aggregated path over-provisions DReLU with the product payload (superset), never under
        agg = requests_from_estimate(est, C, "epoch4:softmax")
        for party, c in ((0, c0), (1, c1)):
            s = connect("127.0.0.1", port, party)
            for r in agg:
                s.provision_into(c, r["request_id"], r["material_type"], r["params"], r["count"])
            s.close()
        st2 = c0.stock()
        for mt, (count, nbytes) in est.items():
            assert st2[mt][0] == 2 * count and st2[mt][1] >= 2 * nbytes
        assert service.connections_accepted == 4 and service.auth_failures == 0
        # the pending table is empty once both recipients fetched everything
        with service._lock:
            assert service._requests == {} and service._fifo == {}
        assert service.pending_halves() == 0
    finally:
        service.shutdown()


def test_in_process_service_concurrent_clients(clients):
    dealer = Dealer(C, np.random.default_rng(7), setup_id="su", model_version="v0")
    service = DealerService(dealer, KEYS, "127.0.0.1", 0)
    port = service.start()
    c0, c1 = clients
    errors = []

    def run(party, c):
        try:
            s = connect("127.0.0.1", port, party)
            for i in range(5):
                s.provision_into(c, f"q{i}", "drelu", {"M": 3}, 1)
            s.close()
        except Exception as exc:   # noqa: BLE001
            errors.append(exc)

    try:
        ts = [threading.Thread(target=run, args=(p, c)) for p, c in ((0, c0), (1, c1))]
        for t in ts:
            t.start()
        for t in ts:
            t.join(60)
        assert not errors
        assert c0.stock()["drelu"] == (15, 15 * 2200) == c1.stock()["drelu"]
        d0 = c0.take_drelu(15)
        d1 = c1.take_drelu(15)
        x = ring.encode(np.linspace(-4, 4, 15), C)
        y = ring.add(x, ring.add(d0.r_share, d1.r_share))
        s_0, _ = drelu.drelu_eval(d0, y)
        s_1, _ = drelu.drelu_eval(d1, y)
        assert np.array_equal(ring.add(s_0, s_1), (np.linspace(-4, 4, 15) >= 0).astype(np.uint64))
    finally:
        service.shutdown()


def test_refill_rate_paces_delivery(clients):
    dealer = Dealer(C, np.random.default_rng(11), setup_id="su", model_version="v0")
    with pytest.raises(MaterialError):
        DealerService(dealer, KEYS, "127.0.0.1", 0, refill_bytes_per_s=0.0)
    service = DealerService(dealer, KEYS, "127.0.0.1", 0, refill_bytes_per_s=200_000.0)   # 200 kB/s
    port = service.start()
    c0, _ = clients
    try:
        s0 = connect("127.0.0.1", port, 0)
        t0 = time.perf_counter()
        s0.provision_into(c0, "refill", "beaver", {"shape": [2000]}, count=1)     # 48 000 bytes: 0.24 s at 200 kB/s
        el = time.perf_counter() - t0
        assert el >= 0.9 * 0.24
        st = s0.status()
        assert st["refill_bytes_per_s"] == 200_000.0 and st["delivered_bytes"] == 48_000
        assert service.delivered_bytes == 48_000
        s0.close()
    finally:
        service.shutdown()
    fast = DealerService(dealer, KEYS, "127.0.0.1", 0)
    port = fast.start()
    try:
        s1 = connect("127.0.0.1", port, 1)
        t0 = time.perf_counter()
        s1.provision("refill", "beaver", {"shape": [2000]}, count=1)
        assert time.perf_counter() - t0 < 0.2
        assert s1.status()["refill_bytes_per_s"] is None
        s1.close()
    finally:
        fast.shutdown()


def test_handshake_timeout_and_pending_limit(clients):
    dealer = Dealer(C, np.random.default_rng(12), setup_id="su", model_version="v0")
    service = DealerService(dealer, KEYS, "127.0.0.1", 0, max_pending_requests=2, handshake_timeout_s=0.3)
    port = service.start()
    try:
        idle = dial_channel("127.0.0.1", port, party=1, retries=200)       # never sends HELLO
        time.sleep(0.8)
        with pytest.raises(ChannelError):
            idle.recv()                                                    # the service dropped it
        idle.close()
        s0 = connect("127.0.0.1", port, 0)
        s0.provision("a", "beaver", {"shape": [2]}, 1)
        s0.provision("b", "beaver", {"shape": [2]}, 1)
        with pytest.raises(MaterialError) as ei:
            s0.provision("c", "beaver", {"shape": [2]}, 1)                 # two requests wait for the peer
        assert ei.value.code == "DEALER_BUSY"
        s1 = connect("127.0.0.1", port, 1)
        s1.provision("a", "beaver", {"shape": [2]}, 1)
        assert len(s0.provision("c", "beaver", {"shape": [2]}, 1)) == 1
        s0.close()
        s1.close()
    finally:
        service.shutdown()


# --------------------------------------------------------------------------
# keys and CLI validation
# --------------------------------------------------------------------------
def test_load_keys_validation(tmp_path):
    p = tmp_path / "k.json"
    p.write_text(json.dumps({"0": "00" * 32}), encoding="utf-8")
    with pytest.raises(MaterialError) as ei:
        load_keys(str(p))
    assert ei.value.code == "DEALER_KEYS"
    p.write_text(json.dumps({"0": "00" * 32, "1": "00" * 32}), encoding="utf-8")
    with pytest.raises(MaterialError):
        load_keys(str(p))                       # identical keys
    p.write_text(json.dumps({"0": "00" * 8, "1": "11" * 32}), encoding="utf-8")
    with pytest.raises(MaterialError):
        load_keys(str(p))                       # too short
    p.write_text(json.dumps({"0": "zz", "1": "11" * 32}), encoding="utf-8")
    with pytest.raises(MaterialError):
        load_keys(str(p))                       # not hex
    write_keys(str(p), KEYS)
    assert load_keys(str(p)) == KEYS


def test_secret_files_are_private_and_lf(tmp_path):
    """Keys and tokens are written with mode 0600 on POSIX (also over an existing file) and LF endings."""
    p = tmp_path / "token.txt"
    p.write_text("old\n", encoding="utf-8")
    if hasattr(os, "chmod"):
        os.chmod(p, 0o644)
    svc.write_private_text(str(p), "a\nb")
    assert p.read_bytes() == b"a\nb"
    write_keys(str(tmp_path / "k.json"), KEYS)
    assert b"\r" not in (tmp_path / "k.json").read_bytes() and load_keys(str(tmp_path / "k.json")) == KEYS
    if os.name == "posix":
        assert (p.stat().st_mode & 0o777) == 0o600 and ((tmp_path / "k.json").stat().st_mode & 0o777) == 0o600
    with pytest.raises(OSError):
        svc.write_private_text(str(tmp_path / "missing" / "t.txt"), "x")


def test_cli_refuses_seed_outside_insecure_mode(keys_file, tmp_path):
    assert main(["--keys", keys_file, "--insecure-seed", "7"]) == 2
    assert main(["--keys", str(tmp_path / "missing.json")]) == 2
    assert main(["--keys", keys_file, "--contract", "NC-unknown"]) == 2
    with pytest.raises(SystemExit):
        main(["--keys", keys_file, "--refill-bytes-per-s", "-5"])
    with pytest.raises(SystemExit):
        main(["--keys", keys_file, "--listen", "nohostport"])


def test_cli_insecure_mode_marks_headers(keys_file, clients, monkeypatch):
    proc, port = _spawn(keys_file, "--mode", MODE_INSECURE, "--insecure-seed", "5")
    try:
        c0, _ = clients
        # a party process (no in-process dealer, no environment switch) refuses this dealer before authenticating
        monkeypatch.setitem(dealer_mod._TEST_ONLY_STATE, "in_process_insecure_dealer", False)
        monkeypatch.delenv(dealer_mod.ENV_ALLOW_INSECURE, raising=False)
        with pytest.raises(MaterialError) as ei:
            connect("127.0.0.1", port, 0)
        assert ei.value.code == "DEALER_INSECURE_MODE"
        # the environment switch (inherited by spawned party processes of a diagnostic run) admits it
        monkeypatch.setenv(dealer_mod.ENV_ALLOW_INSECURE, "1")
        s_env = connect("127.0.0.1", port, 1)
        assert s_env.binding["dealer_mode"] == MODE_INSECURE
        s_env.close()
        monkeypatch.delenv(dealer_mod.ENV_ALLOW_INSECURE)
        # the explicit test-only switch admits it and the material is labelled, never eligible
        s0 = connect("127.0.0.1", port, 0, allow_insecure=True)
        assert s0.binding["dealer_mode"] == MODE_INSECURE and s0.dealer_insecure is True
        b = s0.provision("r", "beaver", {"shape": [2]}, 1)
        s0.close()
        assert b[0][0]["insecure"] is True
        with pytest.raises(MaterialError) as ei:
            c0.ingest(*b[0])                                  # the default DealerClient refuses the bundle too
        assert ei.value.code == "BUNDLE_INSECURE_MODE"
        c0.allow_insecure = True
        c0.ingest(*b[0])
        assert not c0.security_eligible
    finally:
        proc.kill()
        proc.communicate(timeout=10)
