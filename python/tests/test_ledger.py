"""Tests for opencrab2.highlevel.ledger: invariants 1-4 with crash injection."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from opencrab2.errors import LedgerError, MaterialReuse, MaterialVersionMismatch
from opencrab2.highlevel.ledger import (AVAILABLE, BURNED, RESERVED, SPENT, MaterialLedger, payload_digest)

PYDIR = str(pathlib.Path(__file__).resolve().parents[1])
CH = "9c" * 32

BINDING = {"session_id": "sess-1", "setup_id": "setup-A", "model_version": "gpt2-v0",
           "numeric_contract_hash": CH, "profile_id": "SH2P-TrustedOffline-v1"}
VERSION = {k: BINDING[k] for k in ("setup_id", "model_version", "numeric_contract_hash", "profile_id")}


class SimulatedCrash(Exception):
    pass


def fresh(tmp_path, name="j.jsonl", binding=BINDING, **kw):
    return MaterialLedger(str(tmp_path / name), binding, **kw)


def read_journal(path):
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# --------------------------------------------------------------------------
# binding and registration
# --------------------------------------------------------------------------
def test_binding_validation(tmp_path):
    bad = dict(BINDING)
    del bad["setup_id"]
    with pytest.raises(LedgerError) as ei:
        fresh(tmp_path, binding=bad)
    assert ei.value.code == "LEDGER_BINDING"
    with pytest.raises(LedgerError):
        fresh(tmp_path, "k.jsonl", binding={**BINDING, "session_id": ""})
    L = fresh(tmp_path)
    assert L.binding == BINDING
    recs = read_journal(L.journal_path)
    assert recs[0]["kind"] == "BINDING" and all(recs[0][k] == BINDING[k] for k in BINDING)
    L.close()


def test_add_available_rejects_foreign_version_and_duplicates(tmp_path):
    L = fresh(tmp_path)
    L.add_available("m1", "beaver", 24, VERSION)
    with pytest.raises(MaterialReuse) as ei:
        L.add_available("m1", "beaver", 24, VERSION)
    assert ei.value.code == "MATERIAL_REUSE"
    for k in VERSION:
        with pytest.raises(MaterialVersionMismatch) as ei:
            L.add_available("m2", "beaver", 24, {**VERSION, k: "other"})
        assert ei.value.code == "MATERIAL_VERSION_MISMATCH"
    with pytest.raises(MaterialVersionMismatch):
        L.add_available("m2", "beaver", 24, "not-a-dict")   # type: ignore[arg-type]
    assert L.state("m1") == AVAILABLE and L.stock("beaver") == (1, 24)
    with pytest.raises(LedgerError) as ei:
        L.state("zz")
    assert ei.value.code == "LEDGER_UNKNOWN_MATERIAL"
    L.close()


# --------------------------------------------------------------------------
# state machine (invariant 1 and 2)
# --------------------------------------------------------------------------
def test_state_machine_and_invariant_1(tmp_path):
    L = fresh(tmp_path)
    for i in range(4):
        L.add_available(f"m{i}", "drelu", 2200, VERSION)
    L.reserve(["m0", "m1"], "gate-a", 3)
    assert L.state("m0") == RESERVED and L.stock("drelu") == (2, 4400)
    with pytest.raises(LedgerError) as ei:
        L.spend(["m2"])                       # not reserved
    assert ei.value.code == "LEDGER_SPEND_NOT_RESERVED"
    L.spend(["m0", "m1"])
    assert L.state("m0") == SPENT == L.state("m1")
    with pytest.raises(MaterialReuse):
        L.spend(["m0"])                        # enters SPENT at most once
    with pytest.raises(MaterialReuse):
        L.reserve(["m0"], "gate-b", 4)         # a SPENT id is never returned
    with pytest.raises(LedgerError) as ei:
        L.reserve(["m2", "m2"], "g", 0)
    assert ei.value.code == "LEDGER_PARAMS"
    L.reserve(["m2"], "gate-b", 4)
    L.burn(["m2"], "DISCONNECT")
    assert L.state("m2") == BURNED
    with pytest.raises(MaterialReuse):
        L.reserve(["m2"], "gate-c", 5)
    L.burn(["m0", "m2"], "again")              # idempotent on terminal states
    assert L.state("m0") == SPENT and L.state("m2") == BURNED
    L.burn(["m3"], "CANCEL")                   # AVAILABLE -> BURNED directly
    assert L.state("m3") == BURNED
    assert L.counts() == {AVAILABLE: 0, RESERVED: 0, SPENT: 2, BURNED: 2}
    assert L.bytes_in_state(SPENT) == 4400
    L.close()


def test_reserve_is_atomic(tmp_path):
    L = fresh(tmp_path)
    for i in range(3):
        L.add_available(f"m{i}", "beaver", 24, VERSION)
    L.reserve(["m2"], "g", 0)
    L.spend(["m2"])
    n_lines = len(read_journal(L.journal_path))
    with pytest.raises(LedgerError):
        L.reserve(["m0", "m1", "unknown"], "g", 1)
    with pytest.raises(MaterialReuse):
        L.reserve(["m0", "m2"], "g", 1)       # one SPENT id poisons the whole request
    assert L.state("m0") == AVAILABLE and L.state("m1") == AVAILABLE
    assert len(read_journal(L.journal_path)) == n_lines   # nothing was written
    L.reserve(["m0", "m1"], "g", 1)
    assert L.ids_in_state(RESERVED) == ["m0", "m1"]
    L.close()


def test_concurrent_reservation_of_the_same_id(tmp_path):
    L = fresh(tmp_path)
    L.add_available("m", "beaver", 24, VERSION)
    barrier = threading.Barrier(8)

    def worker(_i):
        barrier.wait()
        try:
            L.reserve(["m"], "g", 0)
            return 1
        except LedgerError:
            return 0

    with ThreadPoolExecutor(8) as ex:
        wins = sum(ex.map(worker, range(8)))
    assert wins == 1 and L.state("m") == RESERVED
    L.close()


def test_journal_line_schema_and_fsync(tmp_path, monkeypatch):
    calls = {"n": 0}
    real = os.fsync

    def counting(fd):
        calls["n"] += 1
        return real(fd)

    monkeypatch.setattr(os, "fsync", counting)
    L = fresh(tmp_path)
    n0 = calls["n"]
    L.add_available("m", "beaver", 24, VERSION)
    L.reserve(["m"], "g", 2)
    L.spend(["m"])
    L.record_send("g", 0, "d" * 64)
    L.register_installed("obj", {"model_version": "gpt2-v0", "client_key_domain": "k1"})
    assert calls["n"] - n0 == 5                 # one fsync per state change
    recs = read_journal(L.journal_path)
    assert recs[0]["kind"] == "BINDING"
    state_lines = [r for r in recs if r["kind"] == "STATE"]
    assert [r["state"] for r in state_lines] == [AVAILABLE, RESERVED, SPENT]
    for r in state_lines:
        assert {"material_id", "state", "session_id", "epoch", "gate_id", "seq", "ts"} <= set(r)
        assert r["session_id"] == "sess-1"
    assert state_lines[1]["gate_id"] == "g" and state_lines[1]["epoch"] == 2
    assert state_lines[2]["gate_id"] == "g" and state_lines[2]["epoch"] == 2
    assert recs[-2]["kind"] == "SEND" and recs[-1]["kind"] == "INSTALLED"
    L.close()


# --------------------------------------------------------------------------
# recovery (invariant 2) and journal robustness
# --------------------------------------------------------------------------
def test_reopen_requires_recover_and_burns_reserved(tmp_path):
    L = fresh(tmp_path)
    for i in range(3):
        L.add_available(f"m{i}", "beaver", 24, VERSION)
    L.reserve(["m0"], "g", 0)
    L.spend(["m0"])
    L.reserve(["m1"], "g", 1)
    L.close()
    R = fresh(tmp_path)
    assert R.needs_recovery
    with pytest.raises(LedgerError) as ei:
        R.reserve(["m2"], "g", 2)
    assert ei.value.code == "LEDGER_RECOVERY_REQUIRED"
    with pytest.raises(LedgerError):
        R.add_available("m9", "beaver", 24, VERSION)
    info = R.recover()
    assert info["reserved_burned"] == 1 and info["spent"] == 1 and info["available"] == 1 and info["burned"] == 1
    assert R.state("m1") == BURNED and R.state("m0") == SPENT and R.state("m2") == AVAILABLE
    assert not R.needs_recovery
    R.reserve(["m2"], "g", 2)
    recs = read_journal(R.journal_path)
    burned = [r for r in recs if r.get("state") == BURNED]
    assert burned and burned[0]["reason"] == "RECOVER" and burned[0]["material_id"] == "m1"
    R.close()
    # a second recover on a clean journal is a no-op
    S = fresh(tmp_path)
    assert S.needs_recovery                     # m2 was left RESERVED
    assert S.recover()["reserved_burned"] == 1
    assert S.recover()["reserved_burned"] == 0
    S.close()


def test_reopen_with_different_binding_is_refused(tmp_path):
    L = fresh(tmp_path)
    L.close()
    for k in BINDING:
        with pytest.raises(LedgerError) as ei:
            fresh(tmp_path, binding={**BINDING, k: "other"})
        assert ei.value.code == "LEDGER_BINDING_MISMATCH"


def test_torn_tail_is_discarded_but_middle_corruption_is_refused(tmp_path):
    L = fresh(tmp_path)
    L.add_available("m0", "beaver", 24, VERSION)
    L.close()
    path = tmp_path / "j.jsonl"
    with open(path, "ab") as fh:
        fh.write(b'{"kind":"STATE","material_id":"m0","state":"RES')   # torn write, no newline
    size_before = os.path.getsize(path)
    R = fresh(tmp_path)
    # the fragment is dropped durably when the journal is opened, before anything can be appended
    assert R.torn_tail_bytes == 47 and R.state("m0") == AVAILABLE
    rec = R.recover()
    assert rec["truncated_tail"] == 1 and rec["torn_tail_bytes"] == 47 and R.state("m0") == AVAILABLE
    R.close()
    raw = open(path, "rb").read()
    assert raw.endswith(b"\n") and b'"state":"RES' not in raw and b"LEDGER_TORN_TAIL" in raw
    assert len(raw) > size_before - 47
    # an unparsable line that IS newline-terminated cannot come from a torn append: corruption, file untouched
    with open(path, "ab") as fh:
        fh.write(b'{"kind":"STATE","material_id":"m0","state":"RES\n')
    with open(path, "ab") as fh:
        fh.write(b'{"kind":"SEND","gate_id":"g","seq":0,"digest":"x","ts":0}\n')
    before = open(path, "rb").read()
    with pytest.raises(LedgerError) as ei:
        fresh(tmp_path)
    assert ei.value.code == "LEDGER_CORRUPT" and open(path, "rb").read() == before
    # also when it is the last line
    with open(path, "wb") as fh:
        fh.write(raw + b'{"kind":"STATE","material_id":"m0","state":"RES\n')
    with pytest.raises(LedgerError) as ei:
        fresh(tmp_path)
    assert ei.value.code == "LEDGER_CORRUPT"


def _torn(path, fragment=b'{"kind":"STATE","material_id":"zz","state":"RESERV'):
    with open(path, "ab") as fh:
        fh.write(fragment)                       # an append that died before its newline


@pytest.mark.parametrize("n_reserved", [1, 2])
def test_torn_tail_then_appended_records_survive_a_reopen(tmp_path, n_reserved):
    # Regression: the BURNED records of recover() used to be glued onto the fragment. With one record it
    # was lost at the next open (needs_recovery again); with two the journal became LEDGER_CORRUPT for good.
    path = tmp_path / "j.jsonl"
    L = fresh(tmp_path)
    ids = [f"m{i}" for i in range(n_reserved)]
    for m in ids:
        L.add_available(m, "beaver", 24, VERSION)
    L.reserve(ids, "g", 0)
    L.close()
    _torn(path)
    R = fresh(tmp_path)
    assert R.needs_recovery
    rec = R.recover()
    assert rec["reserved_burned"] == n_reserved and rec["truncated_tail"] == 1
    assert R.recover()["reserved_burned"] == 0            # a second recover on the same handle replays cleanly
    R.close()
    for line in open(path, "rb").read().split(b"\n"):
        if line:
            json.loads(line)                              # every line is one record
    S = fresh(tmp_path)
    assert not S.needs_recovery and all(S.state(m) == BURNED for m in ids)
    assert S.recover()["truncated_tail"] == 0
    S.close()


def test_torn_tail_then_record_send_keeps_the_send_log(tmp_path):
    # Regression (invariant 3): the SEND record glued onto a fragment was discarded at the next open, and
    # a DIFFERENT payload was then accepted for the same (gate, seq).
    path = tmp_path / "j.jsonl"
    L = fresh(tmp_path)
    L.add_available("m0", "beaver", 24, VERSION)
    L.close()
    _torn(path)
    R = fresh(tmp_path)
    R.record_send("g", 1, "aa")
    R.close()
    S = fresh(tmp_path)
    assert S.send_logged("g", 1) and S.retransmit_allowed("g", 1, "aa")
    with pytest.raises(LedgerError) as ei:
        S.record_send("g", 1, "bb")
    assert ei.value.code == "LEDGER_SEND_CONFLICT"
    S.close()


def test_torn_tail_then_full_lifecycle_reopens(tmp_path):
    path = tmp_path / "j.jsonl"
    L = fresh(tmp_path)
    L.close()
    _torn(path, b'{"kind":"STATE","material_id":"m9","state":"AVAILABLE","nbytes":24}')   # complete JSON, no newline
    R = fresh(tmp_path)
    with pytest.raises(LedgerError):
        R.state("m9")                                     # an unterminated record was never acknowledged
    R.add_available("m1", "beaver", 24, VERSION)
    R.reserve(["m1"], "g", 0)
    R.spend(["m1"])
    R.close()
    S = fresh(tmp_path)
    assert S.state("m1") == SPENT and S.torn_tail_bytes == 0
    S.close()


def test_spend_and_burn_with_repeated_ids_write_one_line_per_id(tmp_path):
    # Regression: spend(["a", "a"]) wrote two SPENT lines and the journal could not be reopened.
    path = tmp_path / "j.jsonl"
    L = fresh(tmp_path)
    for m in ("a", "b", "c"):
        L.add_available(m, "beaver", 10, VERSION)
    L.reserve(["a", "b"], "g", 0)
    L.spend(["a", "a"])
    L.burn(["b", "b", "c", "c", "b"], "x")
    assert L.state("a") == SPENT and L.state("b") == BURNED and L.state("c") == BURNED
    assert L.bytes_in_state(SPENT) == 10 and L.bytes_in_state(BURNED) == 20
    with pytest.raises(LedgerError) as ei:
        L.reserve(["a", "a"], "g", 1)
    assert ei.value.code == "LEDGER_PARAMS"
    L.close()
    text = open(path, "rb").read().decode()
    assert text.count('"state":"SPENT"') == 1 and text.count('"state":"BURNED"') == 2
    R = fresh(tmp_path)
    assert R.state("a") == SPENT and R.counts()[BURNED] == 2
    R.close()


def test_invalid_transition_in_journal_is_corruption(tmp_path):
    L = fresh(tmp_path)
    L.add_available("m0", "beaver", 24, VERSION)
    L.close()
    with open(tmp_path / "j.jsonl", "ab") as fh:
        fh.write(b'{"kind":"STATE","material_id":"m0","state":"SPENT","session_id":"sess-1","epoch":0,"gate_id":"g","seq":null,"ts":0}\n')
    with pytest.raises(LedgerError) as ei:
        fresh(tmp_path)
    assert ei.value.code == "LEDGER_CORRUPT"


# --------------------------------------------------------------------------
# retransmission (invariant 3) and installed objects (invariant 4)
# --------------------------------------------------------------------------
def test_retransmission_log(tmp_path):
    L = fresh(tmp_path)
    msg = b"\x01\x02\x03"
    d = payload_digest(msg)
    assert not L.retransmit_allowed("g", 0, d) and not L.send_logged("g", 0)
    L.record_send("g", 0, d)
    L.record_send("g", 0, d)                  # idempotent
    assert L.retransmit_allowed("g", 0, d)
    assert not L.retransmit_allowed("g", 0, payload_digest(b"\x01\x02\x04"))
    assert not L.retransmit_allowed("g", 1, d) and not L.retransmit_allowed("h", 0, d)
    with pytest.raises(LedgerError) as ei:
        L.record_send("g", 0, payload_digest(b"new input under the old mask"))
    assert ei.value.code == "LEDGER_SEND_CONFLICT"
    L.close()
    R = fresh(tmp_path)
    assert R.retransmit_allowed("g", 0, d) and not R.retransmit_allowed("g", 0, "0" * 64)
    R.close()


def test_installed_objects_version_binding(tmp_path):
    L = fresh(tmp_path)
    v0 = {"model_version": "gpt2-v0", "client_key_domain": "client-key-1"}
    assert not L.installed("lm_head", v0)
    L.register_installed("lm_head", v0)
    assert L.installed("lm_head", dict(v0))
    assert not L.installed("lm_head", {**v0, "model_version": "gpt2-v1"})
    assert not L.installed("lm_head", {**v0, "client_key_domain": "client-key-2"})
    assert not L.installed("lm_head", {"model_version": "gpt2-v0"})          # missing key is a mismatch
    assert not L.installed("lm_head", {**v0, "extra": 1})                   # extra key is a mismatch
    assert not L.installed("other", v0)
    with pytest.raises(LedgerError):
        L.register_installed("x", {})
    L.close()
    R = fresh(tmp_path)
    assert R.installed("lm_head", v0) and R.installed_objects() == {"lm_head": v0}
    R.close()


def test_cross_model_mask_reuse_regression(tmp_path):
    """Reusing the wmask B across W0 and W1 leaks E1 - E0 = W1 - W0; versioning prevents it."""
    rng = np.random.default_rng(1)
    W0 = rng.integers(0, 2 ** 64, size=(4, 6), dtype=np.uint64)
    W1 = rng.integers(0, 2 ** 64, size=(4, 6), dtype=np.uint64)
    B = rng.integers(0, 2 ** 64, size=(4, 6), dtype=np.uint64)
    with np.errstate(over="ignore"):
        E0 = W0 - B
        E1 = W1 - B                           # the forbidden reuse
        assert np.array_equal(E1 - E0, W1 - W0)   # the leak
        B1 = rng.integers(0, 2 ** 64, size=(4, 6), dtype=np.uint64)
        E1_fresh = W1 - B1
        assert not np.array_equal(E1_fresh - E0, W1 - W0)
    # ledger side: the installed object is bound to (model_version, client_key_domain)
    L = fresh(tmp_path)
    L.add_available("mask-v0", "wmask_mask", 8 * 24, VERSION)
    L.reserve(["mask-v0"], "install", 0)
    L.spend(["mask-v0"])
    L.register_installed("lm_head", {"model_version": "gpt2-v0", "client_key_domain": "ck"})
    # a new model version must not find the old object installed, nor reuse its mask
    assert not L.installed("lm_head", {"model_version": "gpt2-v1", "client_key_domain": "ck"})
    with pytest.raises(MaterialReuse):
        L.reserve(["mask-v0"], "install", 1)
    # and a bundle generated for the new model version is refused by a session bound to v0
    with pytest.raises(MaterialVersionMismatch):
        L.add_available("mask-v1", "wmask_mask", 8 * 24, {**VERSION, "model_version": "gpt2-v1"})
    L.close()


# --------------------------------------------------------------------------
# crash injection (in process, via the crash hook, and in a subprocess)
# --------------------------------------------------------------------------
BOUNDARIES = ["after_reserve", "after_spend_before_send", "after_send_before_ack", "after_commit"]


def scenario(L, boundary, crash):
    """Run the gate flow and crash (raise/exit) at ``boundary``."""
    for i in range(4):
        L.add_available(f"m{i}", "beaver", 24, VERSION)
    L.reserve(["m0", "m1"], "gate", 0)
    if boundary == "after_reserve":
        crash()
    L.spend(["m0", "m1"])                      # durable before the message
    if boundary == "after_spend_before_send":
        crash()
    L.record_send("gate", 0, "d" * 64)
    if boundary == "after_send_before_ack":
        crash()
    L.register_installed("lm_head", {"model_version": "gpt2-v0", "client_key_domain": "ck"})
    L.reserve(["m2"], "gate", 1)
    if boundary == "after_commit":
        crash()


def assert_invariants_after_recovery(path, boundary):
    R = MaterialLedger(path, BINDING)
    info = R.recover()
    assert R.counts()[RESERVED] == 0                                   # (2) RESERVED -> BURNED
    if boundary == "after_reserve":
        assert R.state("m0") == BURNED and R.state("m1") == BURNED and info["reserved_burned"] == 2
    else:
        assert R.state("m0") == SPENT and R.state("m1") == SPENT        # (1) SPENT stays SPENT
        with pytest.raises(MaterialReuse):
            R.reserve(["m0"], "gate", 7)                                 # (1) never returned
    assert R.state("m3") == AVAILABLE
    if boundary == "after_commit":
        assert R.state("m2") == BURNED and R.installed("lm_head", {"model_version": "gpt2-v0", "client_key_domain": "ck"})
    else:
        assert R.state("m2") == AVAILABLE and not R.installed("lm_head", {"model_version": "gpt2-v0", "client_key_domain": "ck"})
    sent = boundary in ("after_send_before_ack", "after_commit")
    assert R.retransmit_allowed("gate", 0, "d" * 64) is sent              # (3) byte-identical replay only
    assert not R.retransmit_allowed("gate", 0, "e" * 64)
    # (3) the SPENT record precedes the SEND record in the journal
    recs = read_journal(path)
    kinds = [(r["kind"], r.get("state")) for r in recs]
    if sent:
        assert kinds.index(("SEND", None)) > max(i for i, k in enumerate(kinds) if k == ("STATE", SPENT))
    # (4) binding persists: a foreign binding cannot open the journal
    R.close()
    with pytest.raises(LedgerError):
        MaterialLedger(path, {**BINDING, "model_version": "gpt2-v1"})


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_crash_injection_in_process(tmp_path, boundary):
    path = str(tmp_path / "crash.jsonl")

    def crash():
        raise SimulatedCrash(boundary)

    L = MaterialLedger(path, BINDING)
    with pytest.raises(SimulatedCrash):
        scenario(L, boundary, crash)
    L.close()                                  # the file content is already durable; this only frees the fd
    assert_invariants_after_recovery(path, boundary)


@pytest.mark.parametrize("hook_boundary", ["reserve", "spend", "record_send", "register_installed", "add_available"])
def test_crash_right_after_the_durable_write(tmp_path, hook_boundary):
    """Crash between the fsync and the in-memory update: the journal must still replay cleanly."""
    path = str(tmp_path / "hook.jsonl")

    def hook(b):
        if b == hook_boundary:
            raise SimulatedCrash(b)

    L = MaterialLedger(path, BINDING, crash_hook=hook)
    with pytest.raises(SimulatedCrash):
        scenario(L, "never", lambda: None)
    L.close()
    R = MaterialLedger(path, BINDING)
    R.recover()
    assert R.counts()[RESERVED] == 0
    for mid in ("m0", "m1", "m2", "m3"):
        if mid in R.ids_in_state(SPENT):
            with pytest.raises(MaterialReuse):
                R.reserve([mid], "g", 9)
    R.close()


CHILD = r"""
import os, sys
sys.path.insert(0, sys.argv[3])
from opencrab2.highlevel.ledger import MaterialLedger
path, boundary = sys.argv[1], sys.argv[2]
BINDING = {"session_id": "sess-1", "setup_id": "setup-A", "model_version": "gpt2-v0",
           "numeric_contract_hash": "%s", "profile_id": "SH2P-TrustedOffline-v1"}
VERSION = {k: BINDING[k] for k in ("setup_id", "model_version", "numeric_contract_hash", "profile_id")}
L = MaterialLedger(path, BINDING)
def crash():
    os._exit(3)
for i in range(4):
    L.add_available(f"m{i}", "beaver", 24, VERSION)
L.reserve(["m0", "m1"], "gate", 0)
if boundary == "after_reserve": crash()
L.spend(["m0", "m1"])
if boundary == "after_spend_before_send": crash()
L.record_send("gate", 0, "d" * 64)
if boundary == "after_send_before_ack": crash()
L.register_installed("lm_head", {"model_version": "gpt2-v0", "client_key_domain": "ck"})
L.reserve(["m2"], "gate", 1)
if boundary == "after_commit": crash()
""" % CH


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_crash_injection_subprocess(tmp_path, boundary):
    path = str(tmp_path / "sub.jsonl")
    proc = subprocess.run([sys.executable, "-c", CHILD, path, boundary, PYDIR], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 3, proc.stderr
    assert_invariants_after_recovery(path, boundary)


def test_closed_ledger_refuses(tmp_path):
    L = fresh(tmp_path)
    L.close()
    with pytest.raises(LedgerError) as ei:
        L.add_available("m", "beaver", 24, VERSION)
    assert ei.value.code == "LEDGER_CLOSED"
    with pytest.raises(LedgerError):
        L.recover()
