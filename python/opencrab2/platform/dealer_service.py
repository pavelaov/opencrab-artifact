"""Isolated dealer process (paper, Section 4.3 and Appendix B, DealerDelivery).

CLI::

    python -m opencrab2.platform.dealer_service --listen 127.0.0.1:0 --keys keys.json \
        --contract NC-v2-f16-l40 [--setup-id ID] [--model-version V] \
        [--mode ISOLATED_RECIPIENT_BUNDLES] [--port-file PATH]

The process owns the master randomness (``Dealer`` seeded from the OS CSPRNG;
the seed is never written, logged or exported) and serves two recipients over
TCP. Each connection is authenticated by HMAC over a fresh challenge with the
per-recipient key from the keys file (``{"0": "<hex>", "1": "<hex>"}``);
every bundle streamed on that connection is sealed for that recipient only.

Wire dialogue (the same in both implementations; typed frames of
``highlevel.channel``, never pickle; the client speaks first):

    client  HELLO{recipient, setup_id, contract_hash}
    server  CHALLENGE{nonce}                 nonce = 32 random bytes; the frame also carries the public
                                             binding: dealer_mode, insecure, model_version, profile_id,
                                             setup_id, numeric_contract_hash
    client  AUTH{tag}                        tag = hex(HMAC-SHA256(key_recipient, nonce))
    server  AUTH_OK{recipient}
    client  REQ{material_type, params, count}        optional keys: gate_id, request_id
    server  BUNDLE{header, tag}  followed by ONE bytes frame holding the payload
    server  ERROR{code, message}             instead of any reply, on any failure

Rules of the service:

* HELLO is checked before anything is revealed: an unknown recipient is
  ``DEALER_UNKNOWN_RECIPIENT``, a contract hash that differs from the
  dealer's is ``CONTRACT_MISMATCH`` (an empty hash is a mismatch), a setup id
  that differs is ``SETUP_MISMATCH``. An EMPTY setup id means the client does
  not pin a setup and adopts the one declared in CHALLENGE (the party-side
  callers of this package learn their setup id from the dealer; the ledger
  binding of every bundle is still enforced by ``DealerClient.ingest``). The
  C++ service refuses an empty setup id, so a cross-language client passes it.
* A wrong tag is ``AUTH_FAILED`` and closes the connection. Handshake frames
  must arrive within ``handshake_timeout_s``.
* One REQ yields exactly one BUNDLE and one bytes frame, or one ERROR. A
  request that stands for several bundle pairs (``count`` > 1 for Beaver-like
  material, matrix triples, wmask calls and DCF-based gates) is drained by
  repeating the same REQ; the BUNDLE frame carries ``remaining``, the number
  of bundles the request still holds for this recipient.
  ``DealerServiceClient.provision`` does that loop.
* Pairing. With ``request_id`` the first party to name a request triggers the
  generation of all its bundle PAIRS and each recipient receives only its own
  halves; the same id with other parameters is ``DEALER_REQUEST_CONFLICT`` and
  a recipient that already drained a request gets
  ``DEALER_ALREADY_DELIVERED``: material is never streamed twice. Without
  ``request_id`` (the C++ client) identical REQ frames of the two recipients
  are paired first come, first served.
* ``STATUS`` returns produced counts; ``BYE`` closes; any other frame type is
  ``DEALER_BAD_MESSAGE`` and closes the connection.

Party side (``DealerServiceClient``): the dealer mode advertised in CHALLENGE
is checked before the client authenticates, and a dealer that is not
``ISOLATED_RECIPIENT_BUNDLES`` or that labels itself insecure is refused with
``DEALER_INSECURE_MODE`` unless the test-only switch is on
(``highlevel.dealer.insecure_material_allowed``: explicit
``allow_insecure=True``, the environment variable
``OPENCRAB2_TEST_ALLOW_INSECURE_DEALER=1``, or an insecure dealer hosted by
this very process). Every received header is checked the same way
(``BUNDLE_INSECURE_MODE``). The model server, the sidecar and the launcher
construct this client with its defaults, so the platform parties fail closed.

Logs (stderr) carry recipient ids, request ids, material types, counts and
byte sizes only. The HMAC challenge is a shared-key integrity check for the
trusted-offline profile, not an active-security proof (hmac_auth docstring);
the nonce is fixed at 32 bytes so that an AUTH tag can never be replayed as a
bundle tag.

What is tested (tests/test_dealer_service.py): the exact frame sequence of the
dialogue over a raw channel; a subprocess service with two authenticated
client connections provisioning the same request id, the reconstruction of
the delivered Beaver and DReLU correlations from the two ledgers' clients,
rejection of a wrong key, of a wrong contract hash or setup id, of a
cross-recipient bundle, of a conflicting request id and of a duplicate
delivery; pairing without request ids; the refusal of an insecure dealer by a
default client; the refill pacing. Design limit: the service keeps
undelivered halves in memory until both recipients fetched them; there is no
on-disk spool (a restart regenerates fresh material and the parties' ledgers
burn what was reserved).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import signal
import socket
import sys
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from opencrab2.errors import ChannelError, ContractViolation, MaterialError, OpenCrabError
from opencrab2.highlevel.channel import Channel, dial_channel, open_listener
from opencrab2.highlevel.contract import NumericContract, contract_by_name
from opencrab2.highlevel.dealer import (AUTH_NONCE_BYTES, MODE_INSECURE, MODE_ISOLATED, MODES, PROFILE_ID, Dealer,
                                        DealerClient, auth_frame, dealer_auth_tag, hello_frame,
                                        insecure_material_allowed, parse_bundle_frame, parse_challenge,
                                        raise_if_error_frame, req_frame, verify_dealer_auth_tag)
from opencrab2.lowlevel import hmac_auth

MIN_KEY_BYTES = 16
MAX_BUNDLE_BYTES = (1 << 32) - 1          # the payload travels in one bytes frame (u32 length)
HANDSHAKE_TIMEOUT_S = 30.0
BINDING_KEYS = ("setup_id", "model_version", "numeric_contract_hash", "profile_id", "dealer_mode")


def _log(msg: str) -> None:
    sys.stderr.write(f"[opencrab2-dealer] {msg}\n")
    sys.stderr.flush()


def load_keys(path: str) -> Dict[int, bytes]:
    """Read ``{"0": hex, "1": hex}``; refuses short or missing keys."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise MaterialError("DEALER_KEYS: keys file must hold a JSON object")
    keys: Dict[int, bytes] = {}
    for r in (0, 1):
        v = raw.get(str(r))
        if not isinstance(v, str):
            raise MaterialError(f"DEALER_KEYS: missing key for recipient {r}")
        try:
            kb = bytes.fromhex(v)
        except ValueError:
            raise MaterialError(f"DEALER_KEYS: key for recipient {r} is not hex") from None
        if len(kb) < MIN_KEY_BYTES:
            raise MaterialError(f"DEALER_KEYS: key for recipient {r} shorter than {MIN_KEY_BYTES} bytes")
        keys[r] = kb
    if keys[0] == keys[1]:
        raise MaterialError("DEALER_KEYS: the two recipient keys must differ")
    return keys


def write_private_text(path: str, text: str) -> None:
    """Write a secret text file (recipient keys, a session token) readable by its owner only.

    On POSIX the file is created with mode 0600 and an existing file is reset to 0600 before the
    write; Windows keeps the ACL of the directory. LF line endings on every platform.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_BINARY", 0)
    fd = os.open(path, flags, 0o600)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
        fh = os.fdopen(fd, "w", encoding="utf-8", newline="\n")
    except BaseException:
        os.close(fd)
        raise
    with fh:
        fh.write(text)


def write_keys(path: str, keys: Dict[int, bytes]) -> None:
    write_private_text(path, json.dumps({str(r): k.hex() for r, k in keys.items()}))


def auth_tag(key: bytes, nonce: bytes, recipient: Optional[int] = None) -> str:
    """AUTH tag of the dialogue: hex(HMAC-SHA256(key_recipient, nonce)) over the 32 raw nonce bytes.

    ``recipient`` is accepted for source compatibility and ignored: the
    recipient is bound by its key (the two keys must differ, ``load_keys``).
    """
    return dealer_auth_tag(key, nonce)


def params_fingerprint(params: dict) -> str:
    """Canonical text of a REQ params dict (arrays replaced by digest, shape and dtype); pairing key only."""
    def norm(v: Any) -> Any:
        if isinstance(v, (np.ndarray, np.generic)):
            a = np.ascontiguousarray(v)
            return {"__ndarray__": hashlib.sha256(a.tobytes()).hexdigest(), "shape": list(a.shape), "dtype": a.dtype.str}
        if isinstance(v, (bytes, bytearray)):
            return {"__bytes__": hashlib.sha256(bytes(v)).hexdigest(), "len": len(v)}
        if isinstance(v, dict):
            return {str(k): norm(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [norm(x) for x in v]
        return v
    try:
        return hmac_auth.canonical_json(norm(params)).decode("utf-8")
    except (TypeError, ValueError):
        raise MaterialError("DEALER_BAD_REQUEST: params are not canonicalisable") from None


class _Request:
    __slots__ = ("material_type", "fingerprint", "count", "gate_id", "pairs", "next", "created")

    def __init__(self, material_type: str, fingerprint: str, count: int, gate_id: str, pairs: list):
        self.material_type = material_type
        self.fingerprint = fingerprint
        self.count = count
        self.gate_id = gate_id
        self.pairs = pairs
        self.next = {0: 0, 1: 0}          # index of the next undelivered pair per recipient
        self.created = time.time()

    def drained(self, recipient: int) -> bool:
        return self.next[recipient] >= len(self.pairs)


class DealerService:
    """TCP front end of one :class:`Dealer`; one thread per authenticated connection."""

    def __init__(self, dealer: Dealer, keys: Dict[int, bytes], host: str = "127.0.0.1", port: int = 0,
                 max_pending_requests: int = 4096, refill_bytes_per_s: Optional[float] = None,
                 handshake_timeout_s: float = HANDSHAKE_TIMEOUT_S):
        self.dealer = dealer
        self.keys = dict(keys)
        self.host = host
        self.port = int(port)
        self._lsock: Optional[socket.socket] = None
        self._requests: Dict[str, _Request] = {}                       # request_id -> pairs being drained
        self._fifo: Dict[Tuple[str, int], Deque[Tuple[dict, bytes]]] = {}   # (pairing key, recipient) -> waiting halves
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        self.max_pending_requests = int(max_pending_requests)
        self.handshake_timeout_s = float(handshake_timeout_s)
        self.connections_accepted = 0
        self.auth_failures = 0
        self.bundles_sent = 0
        # Refill model (trace family S5, declared refill rate): delivered bundle bytes never exceed
        # refill_bytes_per_s x elapsed wall time, shared across both recipients.
        if refill_bytes_per_s is not None and float(refill_bytes_per_s) <= 0.0:
            raise MaterialError("DEALER_PARAMS: refill_bytes_per_s must be > 0 when given")
        self.refill_bytes_per_s = None if refill_bytes_per_s is None else float(refill_bytes_per_s)
        self._refill_free_at = 0.0
        self._refill_lock = threading.Lock()
        self.delivered_bytes = 0

    def _pace(self, nbytes: int) -> None:
        """Token bucket on delivered bytes (zero burst depth); no-op without a refill rate."""
        with self._refill_lock:
            self.delivered_bytes += int(nbytes)
            if self.refill_bytes_per_s is None:
                return
            now = time.time()
            start = max(now, self._refill_free_at)
            self._refill_free_at = start + nbytes / self.refill_bytes_per_s
            wait = self._refill_free_at - now
        if wait > 0.0:
            time.sleep(wait)

    # -- lifecycle ---------------------------------------------------------------------
    def start(self) -> int:
        self._lsock, self.port = open_listener(self.host, self.port, backlog=8)
        self._lsock.settimeout(0.5)
        t = threading.Thread(target=self._accept_loop, name="dealer-accept", daemon=True)
        t.start()
        self._threads.append(t)
        return self.port

    def _accept_loop(self) -> None:
        assert self._lsock is not None
        while not self._stop.is_set():
            try:
                conn, _addr = self._lsock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            self.connections_accepted += 1
            th = threading.Thread(target=self._serve_connection, args=(conn,), daemon=True)
            th.start()
            self._threads.append(th)

    def shutdown(self) -> None:
        self._stop.set()
        if self._lsock is not None:
            try:
                self._lsock.close()
            except OSError:
                pass

    def serve_forever(self, poll_s: float = 0.2) -> None:
        while not self._stop.is_set():
            time.sleep(poll_s)

    def pending_halves(self) -> int:
        """Bundle halves generated and not yet fetched by their recipient."""
        with self._lock:
            n = sum(len(q) for q in self._fifo.values())
            n += sum((len(r.pairs) - r.next[0]) + (len(r.pairs) - r.next[1]) for r in self._requests.values())
            return n

    # -- per connection ------------------------------------------------------------------
    @staticmethod
    def _error(ch: Channel, code: str, message: str = "") -> None:
        try:
            ch.send({"type": "ERROR", "code": code, "message": message})
        except (ChannelError, OSError):
            pass

    def _serve_connection(self, conn: socket.socket) -> None:
        ch = Channel(conn, party=0)
        recipient: Optional[int] = None
        try:
            recipient = self._authenticate(ch)
            if recipient is None:
                return
            while not self._stop.is_set():
                msg = ch.recv()
                kind = msg.get("type") if isinstance(msg, dict) else None
                if kind == "BYE":
                    break
                if kind == "STATUS":
                    with self._lock:
                        pending = len(self._requests)
                    ch.send({"type": "STATUS", "produced": {k: list(v) for k, v in self.dealer.produced_summary().items()},
                             "pending_requests": pending, "pending_halves": self.pending_halves(),
                             "dealer_mode": self.dealer.mode, "insecure": bool(self.dealer.insecure),
                             "refill_bytes_per_s": self.refill_bytes_per_s, "delivered_bytes": self.delivered_bytes,
                             "bundles_sent": self.bundles_sent, "bytes_sent": self.delivered_bytes})
                    continue
                if kind == "REQ":
                    self._handle_req(ch, recipient, msg)
                    continue
                # anything else ends the session (fail closed, as the C++ service does)
                self._error(ch, "DEALER_BAD_MESSAGE", "expected REQ, STATUS or BYE")
                break
        except ChannelError:
            pass    # peer closed, timed out, or sent a malformed frame
        except Exception as exc:    # never let a handler thread die silently
            _log(f"connection error recipient={recipient} {exc.__class__.__name__}")
        finally:
            ch.close()

    def _authenticate(self, ch: Channel) -> Optional[int]:
        """HELLO / CHALLENGE / AUTH / AUTH_OK; returns the authenticated recipient or None."""
        try:
            ch.sock.settimeout(self.handshake_timeout_s)
        except OSError:
            pass
        try:
            hello = ch.recv()
        except ChannelError:
            return None
        if not isinstance(hello, dict) or hello.get("type") != "HELLO":
            self._error(ch, "DEALER_BAD_MESSAGE", "expected HELLO")
            return None
        recipient = hello.get("recipient")
        if isinstance(recipient, bool) or not isinstance(recipient, int) or recipient not in self.keys:
            self.auth_failures += 1
            _log("hello refused: unknown recipient")
            self._error(ch, "DEALER_UNKNOWN_RECIPIENT", "recipient must be 0 or 1")
            return None
        setup_id = hello.get("setup_id")
        contract_hash = hello.get("contract_hash")
        if not isinstance(setup_id, str) or not isinstance(contract_hash, str):
            self._error(ch, "DEALER_BAD_MESSAGE", "HELLO needs setup_id and contract_hash strings")
            return None
        if setup_id != "" and setup_id != self.dealer.setup_id:
            _log(f"hello refused recipient={recipient}: setup mismatch")
            self._error(ch, "SETUP_MISMATCH", "setup_id differs from the dealer's")
            return None
        if contract_hash != self.dealer.contract_hash:
            _log(f"hello refused recipient={recipient}: contract mismatch")
            self._error(ch, "CONTRACT_MISMATCH", "contract_hash differs from the dealer's numeric contract")
            return None
        nonce = secrets.token_bytes(AUTH_NONCE_BYTES)
        try:
            ch.send({"type": "CHALLENGE", "nonce": nonce, "dealer_mode": self.dealer.mode,
                     "insecure": bool(self.dealer.insecure), "model_version": self.dealer.model_version,
                     "profile_id": self.dealer.profile_id, "setup_id": self.dealer.setup_id,
                     "numeric_contract_hash": self.dealer.contract_hash})
            auth = ch.recv()
        except ChannelError:
            return None
        if not isinstance(auth, dict) or auth.get("type") != "AUTH":
            self._error(ch, "DEALER_BAD_MESSAGE", "expected AUTH")
            return None
        if not verify_dealer_auth_tag(self.keys[recipient], nonce, auth.get("tag")):
            self.auth_failures += 1
            _log("authentication failed")
            self._error(ch, "AUTH_FAILED", "challenge response rejected")
            return None
        try:
            ch.send({"type": "AUTH_OK", "recipient": recipient})
            ch.sock.settimeout(None)
        except (ChannelError, OSError):
            return None
        _log(f"authenticated recipient={recipient}")
        return recipient

    def _take_half(self, recipient: int, material_type: str, params: dict, count: int, gate_id: str,
                   request_id: Optional[str]) -> Tuple[dict, bytes, int]:
        """Next undelivered half of the request for ``recipient``: (header, payload, remaining). Lock held by caller."""
        fingerprint = params_fingerprint(params)

        def generate() -> list:
            if len(self._requests) + sum(1 for q in self._fifo.values() if q) >= self.max_pending_requests:
                raise MaterialError("DEALER_BUSY: too many pending requests")
            pairs = self.dealer.make(material_type, params, count, gate_id)
            if not pairs:
                raise MaterialError("DEALER_PARAMS: the request produced no bundle")
            if any(len(p[1]) > MAX_BUNDLE_BYTES or len(p[3]) > MAX_BUNDLE_BYTES for p in pairs):
                raise MaterialError("MATERIAL_TOO_LARGE: a bundle payload exceeds the bytes frame; split the request")
            return pairs

        def half(pair: tuple) -> Tuple[dict, bytes]:
            return (pair[0], pair[1]) if recipient == 0 else (pair[2], pair[3])

        if request_id is not None:
            req = self._requests.get(request_id)
            if req is None:
                req = _Request(material_type, fingerprint, count, gate_id, generate())
                self._requests[request_id] = req
            else:
                if req.material_type != material_type or req.count != count or req.fingerprint != fingerprint:
                    raise MaterialError(f"DEALER_REQUEST_CONFLICT: request {request_id!r} already exists with other parameters")
                if req.drained(recipient):
                    raise MaterialError(f"DEALER_ALREADY_DELIVERED: request {request_id!r} was already streamed to recipient {recipient}")
            idx = req.next[recipient]
            req.next[recipient] = idx + 1          # advance before streaming: a concurrent duplicate gets the next half or a refusal
            header, payload = half(req.pairs[idx])
            remaining = len(req.pairs) - req.next[recipient]
            if req.drained(0) and req.drained(1):
                del self._requests[request_id]
            return header, payload, remaining
        # no request id: identical REQ frames of the two recipients pair first come, first served
        key = hashlib.sha256("\x1f".join((material_type, fingerprint, str(count), gate_id)).encode("utf-8")).hexdigest()
        mine = self._fifo.get((key, recipient))
        if not mine:
            pairs = generate()
            mine = self._fifo.setdefault((key, recipient), deque())
            theirs = self._fifo.setdefault((key, 1 - recipient), deque())
            for pair in pairs:
                mine.append((pair[0], pair[1]) if recipient == 0 else (pair[2], pair[3]))
                theirs.append((pair[2], pair[3]) if recipient == 0 else (pair[0], pair[1]))
        header, payload = mine.popleft()
        remaining = len(mine)
        if not mine:
            del self._fifo[(key, recipient)]
        return header, payload, remaining

    def _handle_req(self, ch: Channel, recipient: int, msg: dict) -> None:
        try:
            material_type = msg.get("material_type")
            params = msg.get("params")
            count = msg.get("count", 1)
            gate_id = msg.get("gate_id", "*")
            request_id = msg.get("request_id")
            if not isinstance(material_type, str) or not material_type:
                raise MaterialError("DEALER_BAD_REQUEST: material_type must be a non-empty str")
            if params is None:
                params = {}
            if not isinstance(params, dict):
                raise MaterialError("DEALER_BAD_REQUEST: params must be a dict")
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise MaterialError("DEALER_BAD_REQUEST: count must be an int >= 1")
            if not isinstance(gate_id, str):
                raise MaterialError("DEALER_BAD_REQUEST: gate_id must be a str")
            if request_id is not None and (not isinstance(request_id, str) or not request_id):
                raise MaterialError("DEALER_BAD_REQUEST: request_id must be a non-empty str")
            with self._lock:
                header, payload, remaining = self._take_half(recipient, material_type, params, count, gate_id, request_id)
        except (MaterialError, ContractViolation, OpenCrabError) as exc:
            self._error(ch, exc.code, str(exc))
            return
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            self._error(ch, "DEALER_BAD_REQUEST", f"bad params ({exc.__class__.__name__})")
            return
        h, p, tag = self.dealer.seal(recipient, header, payload, self.keys[recipient])
        self._pace(len(p))
        ch.send({"type": "BUNDLE", "header": h, "tag": tag, "remaining": int(remaining)})
        ch.send(p)                                   # exactly one bytes frame per BUNDLE
        self.bundles_sent += 1
        _log(f"delivered recipient={recipient} request={request_id if request_id is not None else '-'} "
             f"type={material_type} bundles=1 bytes={len(p)}")


# --------------------------------------------------------------------------
# Client side
# --------------------------------------------------------------------------
class DealerServiceClient:
    """Party-side connection to the dealer service (authenticates as ``recipient``).

    ``expected_contract_hash`` is the party's numeric contract hash, sent in
    HELLO (the service refuses a mismatch). ``setup_id`` pins the setup when
    the party knows it; left None, the client adopts the setup the dealer
    declares (``binding["setup_id"]``). ``allow_insecure`` is the test-only
    switch of the dealer-mode check: None (default) refuses a dealer that is not
    ISOLATED_RECIPIENT_BUNDLES or that labels itself insecure unless
    ``insecure_material_allowed()`` says otherwise.
    """

    def __init__(self, host: str, port: int, recipient: int, hmac_key: bytes,
                 expected_contract_hash: Optional[str] = None, retries: int = 500, *,
                 setup_id: Optional[str] = None, allow_insecure: Optional[bool] = None):
        self.host = host
        self.port = int(port)
        self.recipient = int(recipient)
        self.key = bytes(hmac_key)
        self.expected_contract_hash = expected_contract_hash
        self.setup_id = None if setup_id is None else str(setup_id)
        self.allow_insecure = None if allow_insecure is None else bool(allow_insecure)
        self.retries = int(retries)
        self.ch: Optional[Channel] = None
        self.binding: Dict[str, Any] = {}
        self.dealer_insecure: Optional[bool] = None      # what CHALLENGE declared (None before connect)
        self.received_bytes = 0
        self.received_bundles = 0

    def _drop(self) -> None:
        """Close without a BYE (the dialogue is out of step or refused)."""
        if self.ch is not None:
            self.ch.close()
            self.ch = None

    def connect(self) -> Dict[str, Any]:
        """HELLO, CHALLENGE, AUTH, AUTH_OK; returns the public binding declared by the dealer.

        The binding keys are ``setup_id, model_version, numeric_contract_hash,
        profile_id, dealer_mode``. A refusal by the service raises
        ``MaterialError`` with the wire code (``AUTH_FAILED``,
        ``CONTRACT_MISMATCH``, ``SETUP_MISMATCH``,
        ``DEALER_UNKNOWN_RECIPIENT``); a dealer that is not isolated and
        OS-seeded raises ``DEALER_INSECURE_MODE`` BEFORE the client
        authenticates, unless the test-only switch is on.
        """
        self.ch = dial_channel(self.host, self.port, party=1, retries=self.retries)
        try:
            self.ch.send(hello_frame(self.recipient, self.setup_id, self.expected_contract_hash))
            chal = self.ch.recv()
            nonce = parse_challenge(chal)
            mode = chal.get("dealer_mode")
            insecure = chal.get("insecure", False) is not False or mode != MODE_ISOLATED
            if insecure and not insecure_material_allowed(self.allow_insecure):
                raise MaterialError(
                    "DEALER_INSECURE_MODE: the dealer is not an isolated OS-seeded dealer "
                    f"(dealer_mode {mode!r}); refused because the test-only switch is off")
            for key, mine, code in (("setup_id", self.setup_id, "SETUP_MISMATCH"),
                                    ("numeric_contract_hash", self.expected_contract_hash, "CONTRACT_MISMATCH")):
                theirs = chal.get(key)
                if mine is not None and theirs is not None and theirs != mine:
                    raise MaterialError(f"{code}: the dealer declares another {key}")
            self.ch.send(auth_frame(self.key, nonce))
            reply = self.ch.recv()
            raise_if_error_frame(reply, "the dealer refused the connection")
            if not isinstance(reply, dict) or reply.get("type") != "AUTH_OK":
                raise MaterialError("DEALER_BAD_MESSAGE: expected AUTH_OK")
            if "recipient" in reply and reply.get("recipient") != self.recipient:
                raise MaterialError("DEALER_BAD_MESSAGE: AUTH_OK names another recipient")
        except (MaterialError, ChannelError):
            self._drop()
            raise
        fallback = {"setup_id": self.setup_id or "", "model_version": "",
                    "numeric_contract_hash": self.expected_contract_hash or "", "profile_id": "", "dealer_mode": ""}
        self.binding = {k: (chal[k] if isinstance(chal.get(k), str) else fallback[k]) for k in BINDING_KEYS}
        self.dealer_insecure = bool(insecure)
        return dict(self.binding)

    def provision(self, request_id: str, material_type: str, params: Optional[dict] = None, count: int = 1,
                  gate_id: str = "*") -> List[Tuple[dict, bytes, str]]:
        """Request ``count`` bundles; returns this recipient's sealed halves.

        One REQ frame per bundle: the same REQ (same ``request_id``) is
        repeated while the BUNDLE frame reports ``remaining`` > 0.
        """
        if self.ch is None:
            raise MaterialError("DEALER_PROTOCOL: not connected")
        count = int(count)
        params = dict(params or {})
        if "values" in params and not isinstance(params["values"], np.ndarray):
            # ring values do not fit the i64 int of the codec: they travel as a uint64 array
            params["values"] = np.asarray(params["values"], dtype=np.uint64)
        req = req_frame(material_type, params, count, gate_id, None if request_id is None else str(request_id))
        limit = max(count, 1)
        out: List[Tuple[dict, bytes, str]] = []
        while True:
            self.ch.send(req)
            # an ERROR{code} reply raises its code here and leaves the connection usable (no bytes frame follows)
            header, tag, remaining = parse_bundle_frame(self.ch.recv())
            payload = self.ch.recv()
            if not isinstance(payload, (bytes, bytearray)):
                self._drop()
                raise MaterialError("DEALER_BAD_MESSAGE: a BUNDLE must be followed by one bytes frame")
            insecure = header.get("insecure") is not False or header.get("dealer_mode") != MODE_ISOLATED
            if insecure and not insecure_material_allowed(self.allow_insecure):
                self._drop()
                raise MaterialError("BUNDLE_INSECURE_MODE: the dealer streamed a bundle labelled insecure; "
                                    "refused because the test-only switch is off")
            out.append((header, bytes(payload), tag))
            self.received_bytes += len(payload)
            self.received_bundles += 1
            if remaining == 0:
                return out
            if len(out) >= limit:
                self._drop()
                raise MaterialError("DEALER_BAD_MESSAGE: the dealer announces more bundles than requested")

    def provision_into(self, client: DealerClient, request_id: str, material_type: str,
                       params: Optional[dict] = None, count: int = 1, gate_id: str = "*") -> int:
        """Provision and ingest into a :class:`DealerClient`; returns the number of bundles."""
        bundles = self.provision(request_id, material_type, params, count, gate_id)
        client.ingest_many(bundles)
        return len(bundles)

    def status(self) -> dict:
        if self.ch is None:
            raise MaterialError("DEALER_PROTOCOL: not connected")
        self.ch.send({"type": "STATUS"})
        msg = self.ch.recv()
        if not isinstance(msg, dict) or msg.get("type") != "STATUS":
            raise MaterialError("DEALER_PROTOCOL: expected STATUS")
        return msg

    def close(self) -> None:
        if self.ch is not None:
            try:
                self.ch.send({"type": "BYE"})
            except ChannelError:
                pass
            self.ch.close()
            self.ch = None


def requests_from_detail(detail: List[dict], request_prefix: str) -> List[dict]:
    """Translate ``material_estimate_detail`` entries into exact provisioning requests.

    One request per (material_type, params) entry for the shape-free types
    (Beaver-like, DReLU, faithful truncation, ring-to-field); matrix triples
    and wmask calls need shapes and are requested by the plan directly. The
    delivered stock then matches the detailed estimate byte for byte.
    """
    out = []
    for i, ent in enumerate(detail):
        mt = ent["material_type"]
        count = int(ent["count"])
        if count <= 0:
            continue
        rid = f"{request_prefix}:{mt}:{i}"
        params = dict(ent.get("params") or {})
        if mt in ("beaver", "kary_select"):
            out.append({"request_id": rid, "material_type": mt, "params": {"shape": [count]}, "count": 1})
        elif mt in ("drelu", "trunc_faithful", "ring_to_field"):
            params["M"] = count
            out.append({"request_id": rid, "material_type": mt, "params": params, "count": 1})
    return out


def requests_from_estimate(estimate: Dict[str, Tuple[int, int]], contract: NumericContract,
                           request_prefix: str, t: Optional[int] = None) -> List[dict]:
    """Translate an aggregated ``material_estimate`` dict into provisioning requests for the shape-free types.

    DReLU material is requested with the product payload (P=2), a strict
    superset of the bit-only gates, so the delivered bytes may exceed the
    estimate; use :func:`requests_from_detail` for exact provisioning.
    """
    out = []
    for mt, (count, _nbytes) in estimate.items():
        if count <= 0:
            continue
        rid = f"{request_prefix}:{mt}"
        if mt in ("beaver", "kary_select"):
            out.append({"request_id": rid, "material_type": mt, "params": {"shape": [int(count)]}, "count": 1})
        elif mt == "drelu":
            out.append({"request_id": rid, "material_type": mt,
                        "params": {"M": int(count), "n_bits": int(contract.dcf_domain_bits), "with_product": True}, "count": 1})
        elif mt == "trunc_faithful":
            out.append({"request_id": rid, "material_type": mt,
                        "params": {"M": int(count), "f": int(contract.frac_bits), "L": int(contract.logical_bits)}, "count": 1})
        elif mt == "ring_to_field":
            p = {"M": int(count)}
            if t is not None:
                p["t"] = int(t)
            out.append({"request_id": rid, "material_type": mt, "params": p, "count": 1})
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _parse_listen(s: str) -> Tuple[str, int]:
    host, _, port = s.rpartition(":")
    if not host or not port.isdigit():
        raise argparse.ArgumentTypeError("DEALER_ARGS: --listen must be host:port")
    return host, int(port)


def _positive_float(s: str) -> float:
    try:
        v = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError("DEALER_ARGS: expected a number") from None
    if v <= 0.0:
        raise argparse.ArgumentTypeError("DEALER_ARGS: expected a positive number")
    return v


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="opencrab2-dealer", description="Isolated OpenCrab v2 dealer service")
    p.add_argument("--listen", type=_parse_listen, default=("127.0.0.1", 0), help="host:port (port 0 = ephemeral)")
    p.add_argument("--keys", required=True, help="JSON file with per-recipient HMAC keys {\"0\": hex, \"1\": hex}")
    p.add_argument("--contract", default="NC-v2-f16-l40", help="named numeric contract")
    p.add_argument("--setup-id", default="setup-0")
    p.add_argument("--model-version", default="unversioned")
    p.add_argument("--profile-id", default=PROFILE_ID)
    p.add_argument("--mode", default=MODE_ISOLATED, choices=list(MODES))
    p.add_argument("--insecure-seed", type=int, default=None,
                   help="deterministic seed; admitted ONLY with --mode INSECURE_TEST_ONLY_SHARED_SEED")
    p.add_argument("--port-file", default=None, help="write the chosen port to this file after binding")
    p.add_argument("--refill-bytes-per-s", type=_positive_float, default=None,
                   help="refill model: cap on delivered material bytes per second (trace family S5)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        keys = load_keys(args.keys)
        contract = contract_by_name(args.contract)
    except OpenCrabError as exc:
        _log(str(exc))
        return 2
    except OSError as exc:
        _log(f"DEALER_KEYS: cannot read keys file ({exc.__class__.__name__})")
        return 2
    if args.insecure_seed is not None and args.mode != MODE_INSECURE:
        _log("DEALER_MODE: --insecure-seed is admitted only with --mode INSECURE_TEST_ONLY_SHARED_SEED")
        return 2
    rng = np.random.default_rng(args.insecure_seed) if args.insecure_seed is not None else None
    dealer = Dealer(contract, rng, args.mode, setup_id=args.setup_id, model_version=args.model_version,
                    profile_id=args.profile_id)
    service = DealerService(dealer, keys, args.listen[0], args.listen[1], refill_bytes_per_s=args.refill_bytes_per_s)
    port = service.start()
    if args.port_file:
        with open(args.port_file, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(str(port))
    sys.stdout.write(f"LISTENING {args.listen[0]} {port}\n")
    sys.stdout.flush()
    _log(f"mode={dealer.mode} insecure={dealer.insecure} contract={contract.name} setup_id={dealer.setup_id} "
         f"model_version={dealer.model_version} refill_bytes_per_s={service.refill_bytes_per_s}")

    def _stop(*_a: Any) -> None:
        service.shutdown()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _stop)
        except (ValueError, OSError):
            pass
    try:
        service.serve_forever()
    except KeyboardInterrupt:
        service.shutdown()
    return 0


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main())


__all__ = [
    "DealerService", "DealerServiceClient", "load_keys", "write_keys", "write_private_text", "auth_tag", "params_fingerprint",
    "requests_from_estimate", "requests_from_detail", "build_parser", "main",
    "MAX_BUNDLE_BYTES", "HANDSHAKE_TIMEOUT_S", "BINDING_KEYS",
]
