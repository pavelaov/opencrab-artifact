"""Durable material ledger (paper, Section 4.3, material lifecycle and commit).

Every unit of correlated randomness (a dealer bundle) is tracked by a
``material_id`` through the states ``AVAILABLE -> RESERVED -> SPENT``;
``BURNED`` is the terminal variant of SPENT reached on uncertainty
(disconnect, timeout, cancellation, or a restart that finds RESERVED
entries). The journal is an append-only JSONL file; every state change is
written and ``os.fsync``-ed BEFORE the in-memory state changes and before
the method returns, so that the first secret-dependent message that uses a
material id is only ever sent after its SPENT record is durable.

Journal line kinds:

* ``BINDING``: ``{kind, session_id, setup_id, model_version, numeric_contract_hash, profile_id, ts}``
  (first line; a reopen with a different tuple is refused).
* ``STATE``: ``{kind, material_id, state, session_id, epoch, gate_id, seq, ts}``
  plus ``material_type, nbytes, version`` on AVAILABLE and ``reason`` on BURNED.
* ``SEND``: ``{kind, gate_id, seq, digest, ts}`` (retransmission log).
* ``INSTALLED``: ``{kind, object_id, version, ts}`` (versioned installed objects).
* ``NOTE``: ``{kind, note, dropped_bytes, offset, ts}`` (audit note, no state; written
  when a torn tail was dropped: ``note`` is ``LEDGER_TORN_TAIL``).

Torn tail. Every append is one buffer of complete ``line + newline`` records
followed by fsync, so a crash can leave at most one incomplete record: bytes
after the last newline. Such a tail was never acknowledged to the caller. On
open and on ``recover`` it is dropped DURABLY (the file is truncated to the
last newline and fsync-ed) before the append descriptor is used again, and a
NOTE line records sizes only. Without the truncation the next record would be
glued onto the fragment: the record would be lost at the next open (a SEND
record lost means a different payload accepted for the same (gate, seq)) or
the journal would become unreadable. Any other unparsable line, including a
newline-terminated last line, is corruption: ``LEDGER_CORRUPT``, nothing is
changed on disk.

Invariants (all tested in tests/test_ledger.py, including crash injection at
every boundary by aborting and re-opening the journal, and in a subprocess):

1. A material_id enters SPENT at most once; a SPENT (or BURNED) id is never
   returned by ``reserve``.
2. ``reserve`` is atomic (all ids or none, one write); ``spend`` is durable
   before it returns; ``recover`` burns every RESERVED id.
3. A message for ``(gate_id, seq)`` may be retransmitted only byte-identical
   (``retransmit_allowed`` compares the payload digest); material of a prior
   seq is SPENT and can never be reserved for a new input.
4. The ledger is bound to ``(session_id, setup_id, model_version,
   numeric_contract_hash, profile_id)``; a bundle version with a different
   tuple is rejected; ``installed(object_id, version)`` returns False on any
   version mismatch.

Design limit: a local journal does not resist
a maliciously rolled-back host. For untrusted persistence an external
monotonic-state mechanism or a fresh session with discarded material is
required.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

from opencrab2.errors import LedgerError, MaterialReuse, MaterialVersionMismatch

AVAILABLE = "AVAILABLE"
RESERVED = "RESERVED"
SPENT = "SPENT"
BURNED = "BURNED"
STATES = (AVAILABLE, RESERVED, SPENT, BURNED)

BINDING_KEYS = ("session_id", "setup_id", "model_version", "numeric_contract_hash", "profile_id")
# Keys of a bundle ``version`` dict that must equal the session binding.
VERSION_KEYS = ("setup_id", "model_version", "numeric_contract_hash", "profile_id")


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def payload_digest(payload: bytes) -> str:
    """SHA-256 hex of message bytes, the key of the retransmission log."""
    return hashlib.sha256(bytes(payload)).hexdigest()


class _Entry:
    __slots__ = ("material_id", "material_type", "nbytes", "version", "state", "gate_id", "epoch", "reason")

    def __init__(self, material_id: str, material_type: str, nbytes: int, version: dict):
        self.material_id = material_id
        self.material_type = material_type
        self.nbytes = int(nbytes)
        self.version = dict(version)
        self.state = AVAILABLE
        self.gate_id: Optional[str] = None
        self.epoch: Optional[int] = None
        self.reason: Optional[str] = None


class MaterialLedger:
    """Append-only, fsync-ed material state machine bound to one session.

    ``session_binding`` must contain exactly the keys of :data:`BINDING_KEYS`.
    Opening an existing journal replays it; if RESERVED entries are found the
    ledger refuses state changes until :meth:`recover` has burned them
    (``LEDGER_RECOVERY_REQUIRED``): a restart with RESERVED entries burns them
    (paper, Section 4.3, material lifecycle and commit).
    """

    def __init__(self, journal_path: str, session_binding: dict, crash_hook: Any = None):
        self.journal_path = str(journal_path)
        self.binding = self._check_binding(session_binding)
        self._lock = threading.RLock()
        self._entries: Dict[str, _Entry] = {}
        self._send_log: Dict[Tuple[str, int], str] = {}
        self._installed: Dict[str, dict] = {}
        self._fd: Optional[int] = None
        self._needs_recovery = False
        self._closed = False
        # Test seam: ``crash_hook(boundary_name)`` is called right after each durable
        # write; tests raise from it to emulate a crash at that boundary.
        self._crash_hook = crash_hook
        self.torn_tail_bytes = 0          # bytes of incomplete trailing records dropped by this handle
        exists = os.path.exists(self.journal_path) and os.path.getsize(self.journal_path) > 0
        if exists:
            # replay (and drop a torn tail durably) BEFORE the append descriptor exists
            info = self._replay(check_binding=True)
            self._open_fd()
            self._note_torn_tail(info)
            self._needs_recovery = any(e.state == RESERVED for e in self._entries.values())
        else:
            self._open_fd()
            rec = {"kind": "BINDING", **self.binding}
            self._append_lines([rec], boundary="binding")

    # -- setup helpers ---------------------------------------------------------
    @staticmethod
    def _check_binding(b: Any) -> dict:
        if not isinstance(b, dict):
            raise LedgerError("LEDGER_BINDING: session_binding must be a dict")
        missing = [k for k in BINDING_KEYS if k not in b]
        if missing:
            raise LedgerError(f"LEDGER_BINDING: missing keys {missing}")
        out = {}
        for k in BINDING_KEYS:
            v = b[k]
            if not isinstance(v, str) or not v:
                raise LedgerError(f"LEDGER_BINDING: {k} must be a non-empty string")
            out[k] = v
        return out

    def _open_fd(self) -> None:
        d = os.path.dirname(os.path.abspath(self.journal_path))
        os.makedirs(d, exist_ok=True)
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
        if hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY
        self._fd = os.open(self.journal_path, flags, 0o600)

    def close(self) -> None:
        with self._lock:
            if self._fd is not None:
                try:
                    os.fsync(self._fd)
                except OSError:
                    pass
                os.close(self._fd)
                self._fd = None
            self._closed = True

    def __enter__(self) -> "MaterialLedger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- durable append ----------------------------------------------------------
    def _append_lines(self, records: Iterable[dict], boundary: str) -> None:
        """Write all records as one buffer, fsync, then call the crash hook."""
        if self._fd is None or self._closed:
            raise LedgerError("LEDGER_CLOSED: journal is closed")
        ts = time.time()
        buf = "".join(canonical_json({**r, "ts": ts}) + "\n" for r in records).encode("utf-8")
        view = memoryview(buf)
        while len(view):
            n = os.write(self._fd, view)
            view = view[n:]
        os.fsync(self._fd)
        if self._crash_hook is not None:
            self._crash_hook(boundary)

    # -- replay --------------------------------------------------------------------
    def _replay(self, check_binding: bool) -> dict:
        entries: Dict[str, _Entry] = {}
        send_log: Dict[Tuple[str, int], str] = {}
        installed: Dict[str, dict] = {}
        lines = 0
        truncated = 0
        binding_seen = False
        with open(self.journal_path, "rb") as fh:
            raw = fh.read()
        # Bytes after the last newline are an incomplete record (see the module docstring): never
        # parsed, never kept. Everything before must be complete, valid lines.
        complete_len = raw.rfind(b"\n") + 1
        torn_bytes = len(raw) - complete_len
        if torn_bytes:
            truncated = 1
        parts = raw[:complete_len].split(b"\n")
        for i, line in enumerate(parts):
            if not line:
                continue
            try:
                rec = json.loads(line.decode("utf-8"))
                if not isinstance(rec, dict):
                    raise ValueError("not an object")
            except (ValueError, UnicodeDecodeError):
                raise LedgerError(f"LEDGER_CORRUPT: unparsable journal line {i + 1}")
            lines += 1
            kind = rec.get("kind")
            if kind == "NOTE":
                continue
            if kind == "BINDING":
                if binding_seen:
                    raise LedgerError("LEDGER_CORRUPT: second BINDING record")
                binding_seen = True
                if check_binding:
                    for k in BINDING_KEYS:
                        if rec.get(k) != self.binding[k]:
                            raise LedgerError(
                                f"LEDGER_BINDING_MISMATCH: journal {k}={rec.get(k)!r} differs from "
                                f"session {self.binding[k]!r}")
            elif kind == "STATE":
                self._replay_state(entries, rec, i + 1)
            elif kind == "SEND":
                key = (str(rec["gate_id"]), int(rec["seq"]))
                prev = send_log.get(key)
                if prev is not None and prev != rec["digest"]:
                    raise LedgerError(f"LEDGER_CORRUPT: conflicting SEND records for {key}")
                send_log[key] = str(rec["digest"])
            elif kind == "INSTALLED":
                installed[str(rec["object_id"])] = dict(rec["version"])
            else:
                raise LedgerError(f"LEDGER_CORRUPT: unknown record kind {kind!r} at line {i + 1}")
        if not binding_seen:
            raise LedgerError("LEDGER_CORRUPT: journal has no BINDING record")
        if torn_bytes:
            # every complete line is valid: drop the fragment durably before anything is appended
            self._truncate_journal(complete_len)
        self._entries = entries
        self._send_log = send_log
        self._installed = installed
        return {"lines": lines, "truncated_tail": truncated, "torn_tail_bytes": torn_bytes, "offset": complete_len}

    def _truncate_journal(self, size: int) -> None:
        """Durable truncation to ``size`` bytes; the append descriptor is reopened afterwards."""
        had_fd = self._fd is not None
        if had_fd:
            os.close(self._fd)
            self._fd = None
        try:
            with open(self.journal_path, "rb+") as fh:
                fh.truncate(int(size))
                fh.flush()
                os.fsync(fh.fileno())
        except OSError as exc:
            raise LedgerError(f"LEDGER_TRUNCATE: cannot drop the torn tail ({exc.__class__.__name__})")
        if had_fd:
            self._open_fd()

    def _note_torn_tail(self, info: dict) -> None:
        """Journal the audit note of a dropped tail (sizes and offsets only, never the fragment)."""
        n = int(info.get("torn_tail_bytes", 0))
        if n <= 0:
            return
        self.torn_tail_bytes += n
        self._append_lines([{"kind": "NOTE", "note": "LEDGER_TORN_TAIL", "dropped_bytes": n,
                             "offset": int(info.get("offset", 0))}], boundary="torn_tail_note")

    @staticmethod
    def _replay_state(entries: Dict[str, _Entry], rec: dict, lineno: int) -> None:
        mid = str(rec["material_id"])
        st = rec["state"]
        if st not in STATES:
            raise LedgerError(f"LEDGER_CORRUPT: unknown state {st!r} at line {lineno}")
        e = entries.get(mid)
        if st == AVAILABLE:
            if e is not None:
                raise LedgerError(f"LEDGER_CORRUPT: duplicate AVAILABLE for {mid} at line {lineno}")
            entries[mid] = _Entry(mid, str(rec["material_type"]), int(rec["nbytes"]), dict(rec["version"]))
            return
        if e is None:
            raise LedgerError(f"LEDGER_CORRUPT: {st} before AVAILABLE for {mid} at line {lineno}")
        if st == RESERVED:
            if e.state != AVAILABLE:
                raise LedgerError(f"LEDGER_CORRUPT: RESERVED from {e.state} for {mid} at line {lineno}")
            e.gate_id = rec.get("gate_id")
            e.epoch = rec.get("epoch")
        elif st == SPENT:
            if e.state != RESERVED:
                raise LedgerError(f"LEDGER_CORRUPT: SPENT from {e.state} for {mid} at line {lineno}")
        elif st == BURNED:
            if e.state not in (AVAILABLE, RESERVED):
                raise LedgerError(f"LEDGER_CORRUPT: BURNED from {e.state} for {mid} at line {lineno}")
            e.reason = rec.get("reason")
        e.state = st

    def _state_record(self, e: _Entry, state: str, **extra: Any) -> dict:
        rec = {"kind": "STATE", "material_id": e.material_id, "state": state,
               "session_id": self.binding["session_id"], "epoch": e.epoch if state != AVAILABLE else None,
               "gate_id": e.gate_id if state != AVAILABLE else None, "seq": None}
        rec.update(extra)
        return rec

    def _guard(self) -> None:
        if self._closed:
            raise LedgerError("LEDGER_CLOSED: journal is closed")
        if self._needs_recovery:
            raise LedgerError("LEDGER_RECOVERY_REQUIRED: journal holds RESERVED entries; call recover() first")

    # -- public API ----------------------------------------------------------------
    def add_available(self, material_id: str, material_type: str, nbytes: int, version: dict) -> None:
        """Register a verified bundle; refuses duplicates and foreign versions."""
        with self._lock:
            self._guard()
            material_id = str(material_id)
            if not material_id:
                raise LedgerError("LEDGER_PARAMS: material_id must be non-empty")
            if material_id in self._entries:
                raise MaterialReuse(f"MATERIAL_REUSE: material_id {material_id!r} already known to the ledger")
            if not isinstance(version, dict):
                raise MaterialVersionMismatch("MATERIAL_VERSION_MISMATCH: version must be a dict")
            for k in VERSION_KEYS:
                if version.get(k) != self.binding[k]:
                    raise MaterialVersionMismatch(
                        f"MATERIAL_VERSION_MISMATCH: bundle {k}={version.get(k)!r} differs from "
                        f"session {self.binding[k]!r}")
            if int(nbytes) < 0:
                raise LedgerError("LEDGER_PARAMS: nbytes must be >= 0")
            e = _Entry(material_id, str(material_type), int(nbytes), version)
            rec = self._state_record(e, AVAILABLE, material_type=e.material_type, nbytes=e.nbytes,
                                     version=e.version)
            self._append_lines([rec], boundary="add_available")
            self._entries[material_id] = e

    def reserve(self, material_ids: List[str], gate_id: str, epoch: int) -> None:
        """Atomically move all ids AVAILABLE -> RESERVED for ``(gate_id, epoch)``."""
        with self._lock:
            self._guard()
            ids = [str(m) for m in material_ids]
            if len(set(ids)) != len(ids):
                raise LedgerError("LEDGER_PARAMS: duplicate material_id in reserve request")
            for mid in ids:
                e = self._entries.get(mid)
                if e is None:
                    raise LedgerError(f"LEDGER_UNKNOWN_MATERIAL: {mid!r}")
                if e.state in (SPENT, BURNED):
                    raise MaterialReuse(f"MATERIAL_REUSE: {mid!r} is {e.state} and can never be reserved again")
                if e.state != AVAILABLE:
                    raise LedgerError(f"LEDGER_RESERVE_UNAVAILABLE: {mid!r} is {e.state}")
            if not ids:
                return
            recs = []
            for mid in ids:
                e = self._entries[mid]
                recs.append({"kind": "STATE", "material_id": mid, "state": RESERVED,
                             "session_id": self.binding["session_id"], "epoch": int(epoch),
                             "gate_id": str(gate_id), "seq": None})
            self._append_lines(recs, boundary="reserve")
            for mid in ids:
                e = self._entries[mid]
                e.state = RESERVED
                e.gate_id = str(gate_id)
                e.epoch = int(epoch)

    def spend(self, material_ids: List[str]) -> None:
        """RESERVED -> SPENT, durable before return (call before the first message).

        A repeated id in the request counts once (one SPENT line per id: a second line would make the
        journal unreadable, invariant 1).
        """
        with self._lock:
            self._guard()
            ids = list(dict.fromkeys(str(m) for m in material_ids))
            for mid in ids:
                e = self._entries.get(mid)
                if e is None:
                    raise LedgerError(f"LEDGER_UNKNOWN_MATERIAL: {mid!r}")
                if e.state in (SPENT, BURNED):
                    raise MaterialReuse(f"MATERIAL_REUSE: {mid!r} is already {e.state}")
                if e.state != RESERVED:
                    raise LedgerError(f"LEDGER_SPEND_NOT_RESERVED: {mid!r} is {e.state}")
            if not ids:
                return
            recs = [self._state_record(self._entries[mid], SPENT) for mid in ids]
            self._append_lines(recs, boundary="spend")
            for mid in ids:
                self._entries[mid].state = SPENT

    def burn(self, material_ids: List[str], reason: str) -> None:
        """AVAILABLE/RESERVED -> BURNED (idempotent for SPENT/BURNED ids and for repeated ids)."""
        with self._lock:
            if self._closed:
                raise LedgerError("LEDGER_CLOSED: journal is closed")
            ids = list(dict.fromkeys(str(m) for m in material_ids))
            recs = []
            todo = []
            for mid in ids:
                e = self._entries.get(mid)
                if e is None:
                    raise LedgerError(f"LEDGER_UNKNOWN_MATERIAL: {mid!r}")
                if e.state in (SPENT, BURNED):
                    continue
                recs.append(self._state_record(e, BURNED, reason=str(reason)))
                todo.append(e)
            if not recs:
                return
            self._append_lines(recs, boundary="burn")
            for e in todo:
                e.state = BURNED
                e.reason = str(reason)

    def burn_reserved(self, reason: str) -> int:
        """Burn every RESERVED id (disconnect / cancel path); returns the count."""
        with self._lock:
            ids = [m for m, e in self._entries.items() if e.state == RESERVED]
            self.burn(ids, reason)
            return len(ids)

    def state(self, material_id: str) -> str:
        with self._lock:
            e = self._entries.get(str(material_id))
            if e is None:
                raise LedgerError(f"LEDGER_UNKNOWN_MATERIAL: {material_id!r}")
            return e.state

    def material_type(self, material_id: str) -> str:
        with self._lock:
            e = self._entries.get(str(material_id))
            if e is None:
                raise LedgerError(f"LEDGER_UNKNOWN_MATERIAL: {material_id!r}")
            return e.material_type

    def stock(self, material_type: str) -> Tuple[int, int]:
        """(count, bytes) of AVAILABLE material of ``material_type``."""
        with self._lock:
            cnt = 0
            nb = 0
            for e in self._entries.values():
                if e.state == AVAILABLE and e.material_type == material_type:
                    cnt += 1
                    nb += e.nbytes
            return cnt, nb

    def ids_in_state(self, state: str, material_type: Optional[str] = None) -> List[str]:
        with self._lock:
            return [m for m, e in self._entries.items()
                    if e.state == state and (material_type is None or e.material_type == material_type)]

    def counts(self) -> Dict[str, int]:
        with self._lock:
            out = {s: 0 for s in STATES}
            for e in self._entries.values():
                out[e.state] += 1
            return out

    def bytes_in_state(self, state: str) -> int:
        with self._lock:
            return sum(e.nbytes for e in self._entries.values() if e.state == state)

    @property
    def needs_recovery(self) -> bool:
        return self._needs_recovery

    def recover(self) -> dict:
        """Replay the journal from disk; RESERVED -> BURNED; returns counts."""
        with self._lock:
            if self._closed:
                raise LedgerError("LEDGER_CLOSED: journal is closed")
            info = self._replay(check_binding=True)
            self._note_torn_tail(info)
            reserved = [m for m, e in self._entries.items() if e.state == RESERVED]
            if reserved:
                recs = [self._state_record(self._entries[m], BURNED, reason="RECOVER") for m in reserved]
                self._append_lines(recs, boundary="recover")
                for m in reserved:
                    self._entries[m].state = BURNED
                    self._entries[m].reason = "RECOVER"
            self._needs_recovery = False
            c = self.counts()
            return {
                "available": c[AVAILABLE], "reserved_burned": len(reserved), "spent": c[SPENT],
                "burned": c[BURNED], "send_log_entries": len(self._send_log),
                "installed": len(self._installed), "lines": info["lines"],
                # a tail dropped when this handle opened the journal is reported here as well
                "truncated_tail": 1 if self.torn_tail_bytes > 0 else 0, "torn_tail_bytes": self.torn_tail_bytes,
            }

    # -- retransmission log ----------------------------------------------------------
    def record_send(self, gate_id: str, seq: int, payload_digest: str) -> None:
        """Log the digest of the message for ``(gate_id, seq)`` before sending it."""
        with self._lock:
            self._guard()
            key = (str(gate_id), int(seq))
            prev = self._send_log.get(key)
            if prev is not None:
                if prev == payload_digest:
                    return
                raise LedgerError(f"LEDGER_SEND_CONFLICT: (gate {key[0]!r}, seq {key[1]}) already logged with a different payload")
            rec = {"kind": "SEND", "gate_id": key[0], "seq": key[1], "digest": str(payload_digest)}
            self._append_lines([rec], boundary="record_send")
            self._send_log[key] = str(payload_digest)

    def retransmit_allowed(self, gate_id: str, seq: int, payload_digest: str) -> bool:
        """True only for a byte-identical replay of a logged ``(gate_id, seq)``."""
        with self._lock:
            prev = self._send_log.get((str(gate_id), int(seq)))
            return prev is not None and prev == str(payload_digest)

    def send_logged(self, gate_id: str, seq: int) -> bool:
        with self._lock:
            return (str(gate_id), int(seq)) in self._send_log

    # -- installed objects -------------------------------------------------------------
    def register_installed(self, object_id: str, version: dict) -> None:
        """Record an installed object (e.g. a wmask E = W - B) with its version tuple."""
        with self._lock:
            self._guard()
            if not isinstance(version, dict) or not version:
                raise LedgerError("LEDGER_PARAMS: installed version must be a non-empty dict")
            rec = {"kind": "INSTALLED", "object_id": str(object_id), "version": dict(version)}
            self._append_lines([rec], boundary="register_installed")
            self._installed[str(object_id)] = dict(version)

    def installed(self, object_id: str, version: dict) -> bool:
        """True iff ``object_id`` was registered with exactly this version dict."""
        with self._lock:
            cur = self._installed.get(str(object_id))
            if cur is None or not isinstance(version, dict):
                return False
            return canonical_json(cur) == canonical_json(version)

    def installed_objects(self) -> Dict[str, dict]:
        with self._lock:
            return {k: dict(v) for k, v in self._installed.items()}


__all__ = [
    "MaterialLedger", "AVAILABLE", "RESERVED", "SPENT", "BURNED", "STATES",
    "BINDING_KEYS", "VERSION_KEYS", "payload_digest", "canonical_json",
]
