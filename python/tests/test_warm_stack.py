"""Warm stacks, per-session deltas, party randomness and token hygiene of the session runtime (in-process parties).

Two ``SessionRunner`` threads share one isolated in-process dealer service, as
in tests/test_session.py, but each party keeps a ``WarmStack`` across two
sessions: the second session must be a real warm request (nothing installed
or shared again, measured ``T_warm_s``, ``T_cold_s`` None), its counters must
be per-session deltas, and a one-sided stack must be refused.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, List, Optional

import pytest

from opencrab2.errors import AdmissionError, OpenCrabError
from opencrab2.highlevel import session as S
from opencrab2.highlevel.dealer import MODE_ISOLATED
from opencrab2.platform.dealer_service import DealerServiceClient
from tests.test_session import C, KEYS, N_GEN, PROMPT, LocalSession, make_tiny_weights, reference_tokens


@pytest.fixture(scope="module")
def tiny():
    W, meta = make_tiny_weights(1, d=64, vocab=256)
    return W, meta, reference_tokens(W, meta, 1, PROMPT, N_GEN)


def _stacks(ls: LocalSession, tmp: str, tag: str) -> List[S.WarmStack]:
    out = []
    for p in (0, 1):
        svc = DealerServiceClient("127.0.0.1", ls.port, p, KEYS[p], C.contract_hash(), retries=200)
        svc.connect()
        out.append(S.new_warm_stack(p, ls.model_version, "default", 1, os.path.join(tmp, f"{tag}-ledger{p}.jsonl"),
                                    "setup-t", C, KEYS[p], svc))
    return out


def _run(ls: LocalSession, stacks: List[Optional[S.WarmStack]], cfgs, out_path: str, chans=None, timeout_s: float = 240.0):
    chans = chans or ls.channels()
    out: List[Any] = [None, None]
    err: List[Optional[BaseException]] = [None, None]

    def worker(i: int) -> None:
        try:
            kw = {"dealer_service": stacks[i].service, "warm": stacks[i]}
            if i == 0:
                kw["weights"], kw["meta"] = ls.W, ls.meta
            out[i] = S.SessionRunner(i, cfgs[i], chans[i], stacks[i].client, stacks[i].ledger, None, None, out_path,
                                     **kw).run()
        except BaseException as exc:   # noqa: BLE001
            err[i] = exc
            try:
                chans[i].close()
            except Exception:
                pass

    ths = [threading.Thread(target=worker, args=(i,), daemon=True) for i in (0, 1)]
    for t in ths:
        t.start()
    for t in ths:
        t.join(timeout_s)
    for i in (0, 1):
        if err[i] is not None and isinstance(err[i], OpenCrabError) and not err[i].code.startswith("CHANNEL"):
            raise err[i]
    for i in (0, 1):
        if err[i] is not None:
            raise err[i]
    assert not any(t.is_alive() for t in ths)
    return out[0], out[1], chans


def test_second_session_on_a_stack_is_a_measured_warm_request(tiny, tmp_path):
    W, meta, ref = tiny
    ls = LocalSession(W, meta, session_id="unused", tmp=str(tmp_path))
    stacks = _stacks(ls, str(tmp_path), "a")
    try:
        out = str(tmp_path / "records.jsonl")
        assert stacks[0].digest() == "" and not stacks[0].warm

        def cfgs(sid: str, policy: str, **kw):
            c1 = ls.cfg(policy, session_id=sid, reference_tokens=ref, **kw)
            c0 = ls.cfg(policy, session_id=sid, prompt_ids=[], prompt_len=len(PROMPT), **kw)   # P0: public length only
            return [c0, c1]

        # session 1 (cold) installs every object under the fixed all-wmask plan
        a0, a1, chans = _run(ls, stacks, cfgs("warm-a", "B0_fixed", fixed_plan=dict(S.DEFAULT_FIXED_PLAN)), out)
        assert a1["output_match"] is True and a0["output_match"] is None      # P0 never receives the predicate
        assert a1["extra"]["warm_start"] is False and a1["T_cold_s"] is not None
        assert a1["extra"]["T_warm_basis"] == "derived-cold-minus-setup-install" and a1["T_warm_s"] <= a1["T_cold_s"]
        assert a1["initializations"] >= 1 and a1["security_eligible"] is True
        assert stacks[0].warm and stacks[1].warm and stacks[0].digest() == stacks[1].digest() != ""
        assert sorted(stacks[1].installed) == ["h0.fc", "h0.out", "h0.proj", "h0.qkv", "lm_head", "wte"]
        assert stacks[0].plan == S.DEFAULT_FIXED_PLAN
        # session 2 on the same stacks AND the same channel: a warm request with per-session counters
        assert a1["extra"]["switches"] == 1                                       # cold plan -> fixed plan at epoch 0
        b0, b1, _ = _run(ls, stacks, cfgs("warm-b", "B0_fixed", fixed_plan=dict(S.DEFAULT_FIXED_PLAN)), out, chans=chans)
        for r in (b0, b1):
            assert r["extra"]["warm_start"] is True and r["extra"]["T_warm_basis"] == "measured-warm-request"
            assert r["T_cold_s"] is None and r["T_warm_s"] == pytest.approx(r["extra"]["session_wall_s"])
            assert r["initializations"] == 0 and r["extra"]["install_s"] == 0.0
            assert all(not e["installed_now"] and not e["masks_delivered"] for e in r["detail"]["epochs"])
            assert r["extra"]["switches"] == 0                                    # starts from the stack's plan
            assert r["security_eligible"] is True and r["tokens"] == N_GEN and r["aborts"] == 0
            assert r["extra"]["warm_objects"] == 6 and r["extra"]["ttft_s"] > 0
        assert b1["output_match"] is True and b1["detail"]["generated_tokens"] == ref
        # deltas: the warm session reports its own bytes and material, not the stack's running totals
        assert 0 < b1["material_consumed"] < stacks[1].client.consumed_bytes
        assert b1["material_consumed"] + a1["material_consumed"] == stacks[1].client.consumed_bytes
        assert 0 < b1["bytes"] < chans[1].stats.bytes_sent + chans[1].stats.bytes_recv
        assert b1["bytes"] < a1["bytes"]                                           # no install, no vector sharing
        assert b1["extra"]["setup_s"] < a1["extra"]["setup_s"] + a1["extra"]["install_s"]
        assert stacks[1].sessions == 2 and stacks[1].ledger.counts()["RESERVED"] == 0
        # the client's record and detail file carry the tokens and the reference; the server's never do
        for line in open(out, encoding="utf-8"):
            rec = json.loads(line)
            if rec["extra"]["party"] == 1:
                assert rec["extra"]["generated_tokens"] == ref and rec["extra"]["reference_tokens"] == ref
                assert rec["extra"]["first_mismatch_step"] is None
            else:
                assert "generated_tokens" not in rec["extra"] and "reference_tokens" not in rec["extra"]
            assert rec["extra"]["party_rng"] == "os-entropy"
            assert not os.path.isabs(rec["raw_measurement_path"])
        d1 = [json.loads(l) for l in open(b1["paths"]["detail"], encoding="utf-8")]
        assert d1[-1]["kind"] == "summary" and d1[-1]["generated_tokens"] == ref and d1[-1]["output_match"] is True
        d0 = [json.loads(l) for l in open(b0["paths"]["detail"], encoding="utf-8")]
        assert d0[-1]["generated_tokens"] is None and d0[-1]["reference_tokens"] is None
        for ch in chans:
            ch.close()
    finally:
        for st in stacks:
            st.close()
        ls.close()


def test_one_sided_or_foreign_stack_is_refused(tiny, tmp_path):
    W, meta, ref = tiny
    ls = LocalSession(W, meta, session_id="unused", tmp=str(tmp_path))
    stacks = _stacks(ls, str(tmp_path), "b")
    fresh = _stacks(ls, str(tmp_path), "c")
    try:
        out = str(tmp_path / "records.jsonl")
        cfg = lambda sid: [ls.cfg("B0_fixed", session_id=sid, fixed_plan=dict(S.DEFAULT_FIXED_PLAN), n_generate=1),
                           ls.cfg("B0_fixed", session_id=sid, fixed_plan=dict(S.DEFAULT_FIXED_PLAN), n_generate=1)]
        _run(ls, stacks, cfg("one-a"), out)
        assert stacks[0].warm and not fresh[1].warm
        # P0 holds a warm stack, P1 a fresh one: the handshake refuses instead of mixing states
        with pytest.raises(AdmissionError) as ei:
            _run(ls, [stacks[0], fresh[1]], cfg("one-b"), out)
        assert ei.value.code == "SESSION_BINDING_MISMATCH"
        # a stack is bound to its (model_version, client_key_domain, n_layers) and to its own ledger
        bad = ls.cfg("B1_naive", session_id="one-c", client_key_domain="other")
        with pytest.raises(AdmissionError) as ei:
            S.SessionRunner(1, bad, None, stacks[1].client, stacks[1].ledger, None, None, out,
                            dealer_service=stacks[1].service, warm=stacks[1])
        assert ei.value.code == "OBJECT_VERSION_MISMATCH"
        with pytest.raises(AdmissionError) as ei:
            S.SessionRunner(1, ls.cfg("B1_naive", session_id="one-d"), None, stacks[1].client, fresh[1].ledger, None,
                            None, out, dealer_service=stacks[1].service, warm=stacks[1])
        assert ei.value.code == "WARM_STACK_BINDING"
        stacks[1].invalidate("TEST")
        assert not stacks[1].warm and stacks[1].digest() == ""
        with pytest.raises(AdmissionError) as ei:
            S.SessionRunner(1, ls.cfg("B1_naive", session_id="one-e"), None, stacks[1].client, stacks[1].ledger, None,
                            None, out, dealer_service=stacks[1].service, warm=stacks[1])
        assert ei.value.code == "WARM_STACK_UNUSABLE"
    finally:
        for st in stacks + fresh:
            st.close()
        ls.close()


def test_public_seed_party_rng_is_never_security_eligible(tiny, tmp_path):
    W, meta, ref = tiny
    ls = LocalSession(W, meta, session_id="rng-1", tmp=str(tmp_path))
    try:
        out = str(tmp_path / "records.jsonl")
        cfgs = [ls.cfg("B0_fixed", fixed_plan=dict(S.DEFAULT_FIXED_PLAN), n_generate=1, party_rng_from_os=False,
                       reference_tokens=ref[:1]) for _ in (0, 1)]
        r0, r1, _ = ls.run(cfgs, out)
        for r in (r0, r1):
            assert r["extra"]["party_rng"] == "public-seed" and r["security_eligible"] is False
            # one status rule for both languages: masks from the public seed are not the
            # secure configuration, so the record is DIAGNOSTIC
            assert r["status"] == "DIAGNOSTIC" and r["aborts"] == 0 and r["dealer_mode"] == MODE_ISOLATED
        assert r1["output_match"] is True
        assert S.SessionConfig(model="m", n_layers=1, prompt_ids=[1], n_generate=1, policy="B1_naive", trace_id="t",
                               seed=1, contract=C).party_rng_from_os is True
    finally:
        ls.close()


def test_detail_file_name_is_portable_for_run_ids_with_colons(tiny, tmp_path):
    W, meta, ref = tiny
    ls = LocalSession(W, meta, session_id="colon-1", tmp=str(tmp_path))
    try:
        out = str(tmp_path / "records.jsonl")
        rid = 'session:diag4:S0:stable-LAN:B0_fixed:{"a": 1}:T8:L4:s1'
        cfgs = [ls.cfg("B0_fixed", fixed_plan=dict(S.DEFAULT_FIXED_PLAN), n_generate=1, run_id=rid) for _ in (0, 1)]
        r0, r1, _ = ls.run(cfgs, out)
        name = os.path.basename(r1["paths"]["detail"])
        assert os.path.isfile(r1["paths"]["detail"]) and ":" not in name and "{" not in name and '"' not in name
        assert r1["raw_measurement_path"] == name                 # relative to the data root, never a host path
        assert r1["run_id"] == rid
    finally:
        ls.close()
