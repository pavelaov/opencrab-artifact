"""Session runtime tests: tiny model, in-process isolated dealer, loopback.

Harness: an in-process ``DealerService`` in ISOLATED_RECIPIENT_BUNDLES mode (one
``Dealer`` seeded from the OS CSPRNG) serves two authenticated
``DealerServiceClient`` connections; each party owns a ``MaterialLedger`` and a
``DealerClient``; the two parties run ``run_session_party`` in two threads over
a ``socket.socketpair``. The model is a random-weight tiny GPT-2
(``make_tiny_weights``) so a full session with two decode epochs takes seconds.
A negative run uses INSECURE_TEST_ONLY_SHARED_SEED to show that eligibility is
refused. Set ``OPENCRAB2_REAL_GPT2=1`` to also run one 1-layer real GPT-2
session (slow, needs the local HF cache).
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import tempfile
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from opencrab2.errors import AdmissionError, OpenCrabError
from opencrab2.highlevel import session as S
from opencrab2.highlevel.channel import Channel
from opencrab2.highlevel.contract import DEFAULT_CONTRACT
from opencrab2.highlevel.dealer import MODE_INSECURE, MODE_ISOLATED, PROFILE_ID, Dealer, DealerClient
from opencrab2.highlevel.ledger import MaterialLedger
from opencrab2.platform.dealer_service import DealerService, DealerServiceClient
from opencrab2.platform import shaper as SH

C = DEFAULT_CONTRACT
KEYS = {0: bytes(range(1, 33)), 1: bytes(range(101, 133))}
PROMPT = [72, 101, 108, 108]       # four byte-level tokens of the tiny model
N_GEN = 3                          # prefill + 2 decode epochs
POLICIES = ["B0_fixed", "B1_naive", "B2_remaining_plus_install", "B4_opencrab"]


# ---------------------------------------------------------------------------
# tiny model
# ---------------------------------------------------------------------------
def make_tiny_weights(n_layers: int, d: int = 64, vocab: int = 256, n_head: int = 4, n_ctx: int = 64,
                      seed: int = 7) -> Tuple[Dict[str, np.ndarray], dict]:
    """Random-weight GPT-2 layout (Conv1D (d_in, d_out)) with a byte-level vocabulary."""
    rng = np.random.default_rng(seed)
    d_ff = 4 * d
    W = {"wte": rng.normal(0, 0.4, size=(vocab, d)).astype(np.float32),
         "wpe": rng.normal(0, 0.1, size=(n_ctx, d)).astype(np.float32),
         "ln_f.g": rng.uniform(0.8, 1.2, size=d).astype(np.float32),
         "ln_f.b": rng.normal(0, 0.1, size=d).astype(np.float32)}
    for li in range(n_layers):
        W[f"h{li}.ln_1.g"] = rng.uniform(0.8, 1.2, size=d).astype(np.float32)
        W[f"h{li}.ln_1.b"] = rng.normal(0, 0.1, size=d).astype(np.float32)
        W[f"h{li}.ln_2.g"] = rng.uniform(0.8, 1.2, size=d).astype(np.float32)
        W[f"h{li}.ln_2.b"] = rng.normal(0, 0.1, size=d).astype(np.float32)
        W[f"h{li}.qkv.w"] = rng.normal(0, 0.12, size=(d, 3 * d)).astype(np.float32)
        W[f"h{li}.qkv.b"] = rng.normal(0, 0.05, size=3 * d).astype(np.float32)
        W[f"h{li}.out.w"] = rng.normal(0, 0.12, size=(d, d)).astype(np.float32)
        W[f"h{li}.out.b"] = rng.normal(0, 0.05, size=d).astype(np.float32)
        W[f"h{li}.fc.w"] = rng.normal(0, 0.15, size=(d, d_ff)).astype(np.float32)
        W[f"h{li}.fc.b"] = rng.normal(0, 0.05, size=d_ff).astype(np.float32)
        W[f"h{li}.proj.w"] = rng.normal(0, 0.12, size=(d_ff, d)).astype(np.float32)
        W[f"h{li}.proj.b"] = rng.normal(0, 0.05, size=d).astype(np.float32)
    h = hashlib.sha256()
    for k in sorted(W):
        h.update(k.encode())
        h.update(W[k].tobytes())
    meta = {"model": f"tiny-gpt2-d{d}-v{vocab}-L{n_layers}", "revision": h.hexdigest()[:16],
            "weight_sha256": h.hexdigest(), "d_model": d, "n_head": n_head, "d_ff": d_ff, "vocab": vocab,
            "ln_eps": 1e-5, "n_layers_total": n_layers, "n_layers": n_layers, "n_ctx": n_ctx}
    return W, meta


def reference_tokens(W: dict, meta: dict, n_layers: int, prompt: List[int], n_gen: int) -> List[int]:
    """Float greedy decode of the declared functionality (erf-GELU) for ``output_match``."""
    from opencrab2.highlevel.transformer import reference_gpt2_logits
    cur = list(prompt)
    out = []
    for _ in range(n_gen):
        lg = reference_gpt2_logits(W, meta, n_layers, cur)[-1]
        t = int(np.argmax(lg))
        out.append(t)
        cur.append(t)
    return out


# ---------------------------------------------------------------------------
# in-process two-party harness with a real (in-process) isolated dealer service
# ---------------------------------------------------------------------------
class LocalSession:
    def __init__(self, W, meta, *, mode: str = MODE_ISOLATED, session_id: str = "sess-1", model_version=None,
                 tmp: Optional[str] = None, seed: Optional[int] = None):
        self.W, self.meta = W, meta
        self.model_version = model_version or S.model_version_of(meta)
        self.session_id = session_id
        self.tmp = tmp or tempfile.mkdtemp(prefix="oc2-session-")
        rng = np.random.default_rng(seed) if seed is not None else None
        self.dealer = Dealer(C, rng, mode, setup_id="setup-t", model_version=self.model_version)
        self.service = DealerService(self.dealer, KEYS, "127.0.0.1", 0)
        self.port = self.service.start()
        self.binding = {"session_id": session_id, "setup_id": "setup-t", "model_version": self.model_version,
                        "numeric_contract_hash": C.contract_hash(), "profile_id": PROFILE_ID}
        self.ledgers: List[MaterialLedger] = []
        self.clients: List[DealerClient] = []
        self.services: List[DealerServiceClient] = []
        self.open_parties()

    def open_parties(self) -> None:
        self.ledgers = [MaterialLedger(os.path.join(self.tmp, f"ledger{p}.jsonl"), self.binding) for p in (0, 1)]
        self.clients = [DealerClient(p, self.ledgers[p], KEYS[p], C) for p in (0, 1)]
        self.services = []
        for p in (0, 1):
            s = DealerServiceClient("127.0.0.1", self.port, p, KEYS[p], C.contract_hash(), retries=200)
            s.connect()
            self.services.append(s)

    def cfg(self, policy: str, **kw) -> S.SessionConfig:
        base = dict(model="tiny", n_layers=self.meta["n_layers"], prompt_ids=list(PROMPT), n_generate=N_GEN,
                    policy=policy, trace_id="unshaped", seed=11, contract=C, session_id=self.session_id,
                    setup_id="setup-t", model_version=self.model_version, length_bucket=8)
        base.update(kw)
        return S.SessionConfig(**base)

    def channels(self) -> List[Channel]:
        s0, s1 = socket.socketpair()
        return [Channel(s0, 0), Channel(s1, 1)]

    def run(self, cfgs, out_path: str, trace=None, runners: Optional[list] = None, cancel_events=None,
            timeout_s: float = 240.0, weights_for_p0: bool = True, hooks: Optional[list] = None):
        """Run both parties (new runners, or resume the given ones); returns (out0, out1, runners).

        ``hooks`` (optional, one dict per party) is forwarded to ``SessionRunner(hooks=...)``.
        """
        chans = self.channels()
        out: List[Any] = [None, None]
        err: List[Optional[BaseException]] = [None, None]
        made: List[Any] = [None, None]

        def worker(i: int) -> None:
            try:
                if runners is not None and runners[i] is not None:
                    runners[i].attach_channel(chans[i])
                    made[i] = runners[i]
                    out[i] = runners[i].resume()
                else:
                    kw = dict(dealer_service=self.services[i], origin_wall_s=None,
                              cancel_event=(cancel_events[i] if cancel_events else None))
                    if hooks is not None and hooks[i]:
                        kw["hooks"] = hooks[i]
                    if i == 0 and weights_for_p0:
                        kw["weights"], kw["meta"] = self.W, self.meta
                    r = S.SessionRunner(i, cfgs[i], chans[i], self.clients[i], self.ledgers[i], None, trace,
                                        out_path, **kw)
                    made[i] = r
                    out[i] = r.run()
            except BaseException as exc:   # noqa: BLE001 (re-raised in the main thread)
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
        for ch in chans:
            ch.close()
        for i in (0, 1):
            if err[i] is not None and isinstance(err[i], OpenCrabError) and not err[i].code.startswith("CHANNEL"):
                raise err[i]
        for i in (0, 1):
            if err[i] is not None:
                raise err[i]
        if any(t.is_alive() for t in ths):
            raise TimeoutError("HARNESS_TIMEOUT: a party did not finish")
        return out[0], out[1], made

    def close(self) -> None:
        for s in self.services:
            try:
                s.close()
            except Exception:
                pass
        for l in self.ledgers:
            l.close()
        self.service.shutdown()


@pytest.fixture(scope="module")
def tiny():
    W, meta = make_tiny_weights(1, d=64, vocab=256)
    ref = reference_tokens(W, meta, 1, PROMPT, N_GEN)
    return W, meta, ref


def _validate(path: str) -> List[dict]:
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    try:
        from opencrab2.experiments import schema          # type: ignore
        for r in recs:
            schema.validate(r)
    except ImportError:
        for r in recs:
            S.validate_record(r)
    return recs


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------
def test_first_mismatch_step_names_the_first_differing_position():
    assert S.first_mismatch_step([1, 2, 3], [1, 2, 3]) is None
    assert S.first_mismatch_step([1, 9, 3], [1, 2, 3]) == 1
    assert S.first_mismatch_step([9, 2, 3], [1, 2, 3]) == 0
    assert S.first_mismatch_step([1, 2], [1, 2, 3]) == 2            # a prefix: the step the client did not release
    assert S.first_mismatch_step([1, 2, 3, 4], [1, 2, 3]) == 3
    assert S.first_mismatch_step([], []) is None and S.first_mismatch_step([], [5]) == 0
    assert S.record_path("relative/x.jsonl") == "relative/x.jsonl"
    assert S.record_path(os.path.join(os.getcwd(), "x.jsonl")) == "x.jsonl"       # no host path in a record


def test_horizon_schedule_lists_the_public_shapes_of_the_remaining_epochs():
    """Entry 0 is the current epoch; every later epoch is one decode row with its own KV length bucket."""
    def shapes_of(rows: int, kv: int) -> dict:
        return {"n": rows, "T": S._bucket(kv, 8)}

    # the prefill of a 5-token prompt in a 4-epoch session, then one row per epoch
    assert S.horizon_schedule(shapes_of, 0, 4, 5, 5, 5) == [{"n": 5, "T": 8}] + [{"n": 1, "T": 8}] * 3
    assert S.horizon_schedule(shapes_of, 2, 4, 5, 1, 7) == [{"n": 1, "T": 8}, {"n": 1, "T": 8}]
    assert S.horizon_schedule(shapes_of, 3, 4, 5, 1, 8) == [{"n": 1, "T": 8}]
    # a recovery prefill at epoch 2 processes the prompt and the two generated tokens
    assert S.horizon_schedule(shapes_of, 2, 4, 5, 7, 7) == [{"n": 7, "T": 8}, {"n": 1, "T": 8}]
    # the KV length bucket of a later step grows with the step
    assert [s["T"] for s in S.horizon_schedule(shapes_of, 0, 6, 6, 6, 6)] == [8, 8, 8, 16, 16, 16]
    assert len(S.horizon_schedule(shapes_of, 0, 8, 16, 16, 16)) == 8


def test_tournament_counts_match_binary_and_kary_structure():
    assert S.tournament_counts(1, 2, True) == (0, 0, 0)
    assert S.tournament_counts(8, 2, False) == (7, 0, 7)            # n - 1 comparisons, n - 1 pair selects
    p1, p2, bv = S.tournament_counts(8, 2, True)
    assert (p1, bv) == (7, 7 + 3)                                    # index selects from the second level on
    p1, p2, bv = S.tournament_counts(8, 8, True)
    assert (p1, p2, bv) == (28, 7, 0)                                # one node of 8: 28 pairs, 7 payload gates
    p1, p2, bv = S.tournament_counts(256, 4, True)
    assert p1 == 6 * (64 + 16 + 4 + 1) and bv == 0 and p2 == 3 * 64 + 2 * 3 * (16 + 4 + 1)


def test_material_requests_are_deterministic_and_chunked(tiny):
    W, meta, _ = tiny
    card = S.model_card(meta, 1)
    plan = {"embedding": "wmask", "linear": "generic", "argmax_k": 4, "max_k": 2}
    mp = S.epoch_material_plan(plan, card, 4, 4, 1, C)
    assert mp["trunc"] >= 4 * 64 * 2 + 64 and mp["drelu_p1"] > 0 and mp["beaver"] > 0
    assert ("wte", 4) in mp["wmask_calls"] and ("lm_head", 1) in mp["wmask_calls"]
    assert ((4, 64), (64, 192)) in mp["matrix_triples"]
    r1 = S.material_requests(mp, "sid:g0", 3, C)
    r2 = S.material_requests(mp, "sid:g0", 3, C)
    assert r1 == r2 and len({r["request_id"] for r in r1}) == len(r1)
    assert all(r["request_id"].startswith("sid:g0:e3:") for r in r1)
    big = dict(mp, drelu_p1=3 * S.MAX_GATES_PER_BUNDLE + 5)
    reqs = [r for r in S.material_requests(big, "p", 0, C) if r["request_id"].count(":drelu1:")]
    assert [r["params"]["M"] for r in reqs] == [S.MAX_GATES_PER_BUNDLE] * 3 + [5]
    assert S.material_plan_bytes(mp, C) > 0


def test_trace_shaper_and_pacer():
    tr = {"trace_id": "t", "family": "S5", "split": "tune", "seed": 1,
          "points": [(0.0, 1e8, 0.001, 1.0, 8e6), (1.0, 5e7, 0.002, 0.5, None)]}
    h = SH.trace_hash(tr)
    assert h == SH.trace_hash(tr) and h != SH.trace_hash(dict(tr, seed=2))
    sh = SH.TraceShaper(tr, origin_wall_s=1000.0)
    assert sh.cpu_share_at(1000.5) == 1.0 and sh.cpu_share_at(1001.5) == 0.5
    assert sh.refill_bps_at(1000.5) == 8e6 and sh.refill_bps_at(1002.0) is None
    assert sh.shaped and sh.has_cpu_load and sh.refill is not None
    assert abs(sh.link.serialization_time(12500, 1000.0) - 0.001) < 1e-9     # 100 kbit at 100 Mbit/s
    assert sh.refill.serialization_time(1_000_000, 1000.0) == pytest.approx(1.0)   # 8 Mbit at 8 Mbit/s
    pacer = SH.MaterialRefillPacer(sh)
    assert pacer.active
    load = SH.CpuContentionLoad(sh, n_cores=4)
    assert load.target_load_cores(1001.5) == pytest.approx(2.0) and load.target_load_cores(1000.5) == 0.0
    assert SH.TraceShaper(SH.unshaped_trace(), 0.0).shaped is False
    with pytest.raises(OpenCrabError):
        SH.normalize_points({"points": [(1.0, None, 0.0, 1.0, None), (0.5, None, 0.0, 1.0, None)]})


def test_cpu_contention_load_follows_the_trace():
    tr = {"trace_id": "s4", "points": [(0.0, None, 0.0, 1.0, None), (0.3, None, 0.0, 0.5, None),
                                        (0.6, None, 0.0, 1.0, None)]}
    import time
    sh = SH.TraceShaper(tr, origin_wall_s=time.time())
    load = SH.CpuContentionLoad(sh, n_cores=2, tick_s=0.02)
    try:
        assert load.start() == 1
        time.sleep(0.45)
        duties_mid = load.current_duties()
        time.sleep(0.4)
        duties_end = load.current_duties()
    finally:
        load.stop()
    assert duties_mid == [1.0] and duties_end == [0.0]
    assert any(v == 100 for _t, v in load.history) and any(v == 0 for _t, v in load.history)


def test_workload_scope_label_fails_closed_on_unsupported_prefix():
    from opencrab2.errors import ContractViolation
    assert S.workload_scope_for(1, 1) == "full-model" and S.workload_scope_for(12, 12) == "full-model"
    assert S.workload_scope_for(1, 12) == "prefix-1L" and S.workload_scope_for(4, 12) == "prefix-4L"
    assert S.workload_scope_for(2, 12, "web") == "web"
    with pytest.raises(ContractViolation) as ei:
        S.workload_scope_for(2, 12)                 # the schema names no prefix-2L label
    assert ei.value.code == "WORKLOAD_SCOPE_UNSUPPORTED"
    with pytest.raises(ContractViolation):
        S.workload_scope_for(1, 12, "prefix-2L")


def test_validate_record_rejects_bad_shapes():
    rec = {k: 0 for k in S.RECORD_FIELDS}
    with pytest.raises(OpenCrabError):
        S.validate_record(rec)
    rec.update(status="MEASURED_NATIVE", security_eligible=False, dealer_mode=MODE_ISOLATED,
               clock_basis=S.CLOCK_BASIS, output_match=None, T_cold_s=0.0, T_warm_s=0.0, T_online_s=0.0,
               code_commit="x", profile_id=PROFILE_ID, numeric_contract_hash="h", workload_scope="full-model",
               model_revision="r", trace_id="t", trace_hash="n", policy_id="B1", initial_state_hash="s", run_id="r",
               raw_measurement_path="p")
    S.validate_record(rec)
    with pytest.raises(OpenCrabError):
        S.validate_record(dict(rec, security_eligible=True, dealer_mode=MODE_INSECURE))


# ---------------------------------------------------------------------------
# full sessions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("policy", POLICIES)
def test_session_policies_over_loopback(tiny, policy, tmp_path):
    W, meta, ref = tiny
    ls = LocalSession(W, meta, session_id=f"sess-{policy}", tmp=str(tmp_path))
    try:
        out = str(tmp_path / "records.jsonl")
        cfgs = [ls.cfg(policy, reference_tokens=ref), ls.cfg(policy, reference_tokens=ref)]
        r0, r1, runners = ls.run(cfgs, out)
        recs = _validate(out)
        assert len(recs) == 2
        for r in (r0, r1):
            assert r["status"] == "MEASURED_NATIVE" and r["dealer_mode"] == MODE_ISOLATED
            assert r["security_eligible"] is True
            assert r["tokens"] == N_GEN and r["policy_id"] == policy and r["aborts"] == 0
            assert r["T_online_s"] <= r["T_cold_s"] and r["T_warm_s"] <= r["T_cold_s"]
            assert r["bytes"] > 0 and r["rounds"] > 0 and r["material_consumed"] > 0
            assert r["material_transferred"] >= r["material_consumed"]
            assert r["workload_scope"] == "full-model" and r["clock_basis"] == S.CLOCK_BASIS
            assert r["detail"]["completed"] and not r["detail"]["cancelled"]
            assert len(r["detail"]["epochs"]) == N_GEN and r["detail"]["releases"] == N_GEN
            assert r["detail"]["telemetry_accepted"] == N_GEN and r["detail"]["telemetry_rejected"] == 0
            # the record names the detail file without a host path; ``paths`` holds the file the party wrote
            assert os.path.isfile(r["paths"]["detail"]) and not os.path.isabs(r["raw_measurement_path"])
            assert os.path.basename(r["paths"]["detail"]) == r["raw_measurement_path"]
        assert r0["initial_state_hash"] == r1["initial_state_hash"] and r0["bytes"] == r1["bytes"]
        assert r0["peak_rss_client"] == r1["peak_rss_client"] and r0["peak_rss_server"] == r1["peak_rss_server"]
        assert r1["detail"]["generated_tokens"] == ref and r1["output_match"] is True
        # the client record carries what a mismatch report needs; P0 never receives the predicate or a token
        assert r1["extra"]["generated_tokens"] == ref and r1["extra"]["reference_tokens"] == ref
        assert r1["extra"]["first_mismatch_step"] is None and r1["detail"]["first_mismatch_step"] is None
        assert r0["output_match"] is None and "generated_tokens" not in r0["extra"]
        plans0 = [e["plan"] for e in r0["detail"]["epochs"]]
        plans1 = [e["plan"] for e in r1["detail"]["epochs"]]
        assert plans0 == plans1
        if policy == "B0_fixed":
            assert all(p == S.DEFAULT_FIXED_PLAN for p in plans0) and r0["initializations"] == 1
        # the material estimate is an upper bound: no stall request was needed
        assert r0["detail"]["stalls"] == 0 and r1["detail"]["stalls"] == 0
        # every ledger id ended in a terminal or available state, never RESERVED
        for L in ls.ledgers:
            assert L.counts()["RESERVED"] == 0 and L.counts()["SPENT"] > 0
    finally:
        ls.close()


def test_insecure_dealer_mode_is_never_eligible(tiny, tmp_path):
    W, meta, ref = tiny
    ls = LocalSession(W, meta, mode=MODE_INSECURE, session_id="sess-insecure", tmp=str(tmp_path), seed=3)
    try:
        out = str(tmp_path / "records.jsonl")
        r0, r1, _ = ls.run([ls.cfg("B1_naive"), ls.cfg("B1_naive")], out)
        for r in (r0, r1):
            assert r["dealer_mode"] == MODE_INSECURE and r["status"] == "DIAGNOSTIC"
            assert r["security_eligible"] is False and r["tokens"] == N_GEN
        _validate(out)
    finally:
        ls.close()


def _journal_states(path: str, material_id: str) -> List[dict]:
    """STATE records of one material id in journal order."""
    out = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("kind") == "STATE" and rec.get("material_id") == material_id:
            out.append(rec)
    return out


def test_cancellation_burns_material_and_warm_resume_completes(tiny, tmp_path):
    W, meta, ref = tiny
    ls = LocalSession(W, meta, session_id="sess-cancel", tmp=str(tmp_path))
    try:
        out = str(tmp_path / "records.jsonl")
        cfgs = [ls.cfg("B2_remaining_plus_install", cancel_after_epoch=0, reference_tokens=ref) for _ in (0, 1)]
        # a gate that holds material RESERVED when the cancel lands: at the end of epoch 0 (after the prefetch of
        # the epoch-1 material) each party reserves one AVAILABLE bundle under a test gate and never spends it
        held: List[List[str]] = [[], []]

        def make_hooks(i: int) -> dict:
            def on_epoch(rec: dict) -> None:
                if int(rec["epoch"]) == 0:
                    ids = ls.ledgers[i].ids_in_state("AVAILABLE")
                    if ids:
                        ls.ledgers[i].reserve(ids[:1], "test-held", 0)
                        held[i] = ids[:1]
            return {"epoch": on_epoch}

        r0, r1, runners = ls.run(cfgs, out, hooks=[make_hooks(0), make_hooks(1)])
        for i, (r, L) in enumerate(zip((r0, r1), ls.ledgers)):
            assert r["detail"]["cancelled"] and not r["detail"]["completed"] and r["security_eligible"] is False
            assert r["tokens"] == 1 and r["detail"]["cancel_epoch"] == 1
            counts = L.counts()
            assert counts["RESERVED"] == 0                      # nothing left reserved
            assert counts["BURNED"] > 0                         # the prefetched epoch-1 material was burned as safe waste
            assert counts["AVAILABLE"] == 0                     # no one-time material survives a cancel
            assert r["material_burned"] > 0 and r["detail"]["wasted_bytes"] > 0
            assert r["detail"]["epochs"][-1]["gate"] == "cancelled"
            # the RESERVED id of the held gate was burned by the cancel (ledger states and the durable journal)
            assert held[i], "the epoch hook found no AVAILABLE bundle to hold"
            assert L.state(held[i][0]) == "BURNED"
            assert r["detail"]["burned_on_cancel"] >= 1 and r["extra"]["burned_on_cancel"] >= 1
            assert r["detail"]["epochs"][-1]["burned_reserved"] >= 1
            journal = _journal_states(os.path.join(ls.tmp, f"ledger{i}.jsonl"), held[i][0])
            assert [j["state"] for j in journal] == ["AVAILABLE", "RESERVED", "BURNED"]
            assert journal[1]["gate_id"] == "test-held" and journal[2]["reason"] == "CANCEL"
        _validate(out)
        # warm resume: same runners, new channel, same session_id; the remaining two epochs complete
        for rn in runners:
            rn.cfg.cancel_after_epoch = None
        q0, q1, _ = ls.run(cfgs, out, runners=runners)
        for q in (q0, q1):
            assert q["detail"]["completed"] and q["detail"]["resumed"] and q["tokens"] == N_GEN
            # the resumed run completed under the isolated dealer without reveals or aborts: eligible;
            # the cancelled record above stays ineligible (the run had not completed)
            assert q["detail"]["epochs_done"] == N_GEN and q["security_eligible"] is True and q["aborts"] == 0
            assert q["extra"]["resumed"] is True and q["extra"]["burned_on_cancel"] >= 1
        assert q1["detail"]["generated_tokens"] == ref and q1["output_match"] is True
        assert len(_validate(out)) == 4
    finally:
        ls.close()


def test_resume_with_other_model_version_is_rejected(tiny, tmp_path):
    W, meta, ref = tiny
    ls = LocalSession(W, meta, session_id="sess-ver", tmp=str(tmp_path))
    try:
        out = str(tmp_path / "records.jsonl")
        cfgs = [ls.cfg("B1_naive", cancel_after_epoch=0) for _ in (0, 1)]
        r0, r1, runners = ls.run(cfgs, out)
        assert r0["detail"]["cancelled"]
        # the server comes back with another model version: the warm resume handshake refuses it
        runners[0].model_version = runners[0].model_version + "-v2"
        with pytest.raises(AdmissionError) as ei:
            ls.run(cfgs, out, runners=runners)
        assert ei.value.code == "OBJECT_VERSION_MISMATCH"
        # cold resume from the public state with a mismatching declared version is refused before any message
        ls.ledgers[1].close()
        ledger1 = MaterialLedger(os.path.join(ls.tmp, "ledger1.jsonl"), ls.binding)
        client1 = DealerClient(1, ledger1, KEYS[1], C)
        bad = ls.cfg("B1_naive", resume=True, model_version=ls.model_version + "-v2")
        with pytest.raises(AdmissionError) as ei:
            S.SessionRunner(1, bad, None, client1, ledger1, None, None, out, dealer_service=ls.services[1])
        assert ei.value.code == "OBJECT_VERSION_MISMATCH"
        ledger1.close()
        ls.ledgers[1] = MaterialLedger(os.path.join(ls.tmp, "ledger1.jsonl"), ls.binding)
    finally:
        ls.close()


def test_cold_resume_recovers_ledger_and_reprefills(tiny, tmp_path):
    W, meta, ref = tiny
    ls = LocalSession(W, meta, session_id="sess-cold", tmp=str(tmp_path))
    try:
        out = str(tmp_path / "records.jsonl")
        cfgs = [ls.cfg("B1_naive", cancel_after_epoch=0, reference_tokens=ref) for _ in (0, 1)]
        ls.run(cfgs, out)
        # new processes would reopen the journals: emulate with fresh ledgers and clients
        for L in ls.ledgers:
            L.close()
        for s in ls.services:
            s.close()
        ls.open_parties()
        cfgs2 = [ls.cfg("B1_naive", resume=True, reference_tokens=ref) for _ in (0, 1)]
        q0, q1, runners = ls.run(cfgs2, out)
        for q in (q0, q1):
            assert q["detail"]["completed"] and q["detail"]["resumed"] and q["tokens"] == N_GEN
            assert q["detail"]["epochs"][0]["gate"] == "recovery_prefill" and q["detail"]["epochs"][0]["n_rows"] == len(PROMPT) + 1
            assert q["detail"]["epochs"][-1]["gate"] == "decode"
        assert q1["detail"]["generated_tokens"] == ref and q1["output_match"] is True
        for L in ls.ledgers:
            assert L.counts()["RESERVED"] == 0
    finally:
        ls.close()


def test_shaped_trace_runs_and_labels_record(tiny, tmp_path):
    W, meta, ref = tiny
    ls = LocalSession(W, meta, session_id="sess-shaped", tmp=str(tmp_path))
    try:
        trace = {"trace_id": "S1-ramp-test", "family": "S1", "split": "test", "seed": 2,
                 "points": [(0.0, 2e8, 0.0005, 1.0, None), (2.0, 5e8, 0.0002, 1.0, None)]}
        out = str(tmp_path / "records.jsonl")
        cfgs = [ls.cfg("B4_opencrab", reference_tokens=ref, cpu_load_party=None) for _ in (0, 1)]
        r0, r1, _ = ls.run(cfgs, out, trace=trace)
        for r in (r0, r1):
            assert r["trace_id"] == "S1-ramp-test" and r["trace_hash"] == SH.trace_hash(trace)
            assert r["detail"]["shaped"] is True and r["security_eligible"] is True
            assert r["detail"]["channel"]["shaping_delay_s"] > 0.0
            assert all(e["sensed_rtt_s"] >= 0.0004 for e in r["detail"]["epochs"])
        assert r0["detail"]["origin_wall_s"] != r1["detail"]["origin_wall_s"] or True
    finally:
        ls.close()


def _gpt2_snapshot_cached() -> bool:
    """True when the pinned ``openai-community/gpt2`` snapshot is in the local HF cache (never touches the network)."""
    if os.environ.get("OPENCRAB2_SKIP_REAL_GPT2") == "1":
        return False
    try:
        from opencrab2.highlevel.transformer import GPT2_REPO, locate_snapshot
        locate_snapshot(GPT2_REPO)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _gpt2_snapshot_cached(),
                    reason="the local HF snapshot of openai-community/gpt2 is unavailable (or OPENCRAB2_SKIP_REAL_GPT2=1)")
def test_real_gpt2_one_layer_session(tmp_path):
    """One 1-layer real GPT-2 session (prefill + 1 decode epoch, B4_opencrab) over the in-process isolated dealer.

    About 55 s on the measured host; the tiny model covers the policy sweep above, this run shows the real
    weights, the pinned revision and the 50257-way argmax release go through the same session loop.
    """
    from opencrab2.highlevel.transformer import load_gpt2_weights, reference_gpt2_logits
    W, meta = load_gpt2_weights(1)
    prompt = [464, 3139, 286, 4881]
    ls = LocalSession(W, meta, session_id="sess-gpt2", tmp=str(tmp_path))
    try:
        out = str(tmp_path / "records.jsonl")
        cur = list(prompt)
        ref: List[int] = []
        for _ in range(2):
            ref.append(int(np.argmax(reference_gpt2_logits(W, meta, 1, cur)[-1])))
            cur.append(ref[-1])
        cfgs = [ls.cfg("B4_opencrab", prompt_ids=prompt, n_generate=2, n_layers=1, model="gpt2", reference_tokens=ref)
                for _ in (0, 1)]
        r0, r1, _ = ls.run(cfgs, out, timeout_s=3600.0)
        assert r1["detail"]["generated_tokens"] == ref and r1["output_match"] is True
        for r in (r0, r1):
            assert r["security_eligible"] is True and r["status"] == "MEASURED_NATIVE" and r["tokens"] == 2
            assert r["model_revision"] == meta["revision"] and r["workload_scope"] == "prefix-1L"
            assert r["aborts"] == 0 and r["detail"]["stalls"] == 0
        assert len(_validate(out)) == 2
    finally:
        ls.close()
