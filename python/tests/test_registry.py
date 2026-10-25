"""Tests for opencrab2.highlevel.registry: admission order, reason codes, proof ledger, default GPT-2 tape."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from opencrab2.errors import AdmissionError
from opencrab2.highlevel import candidates as cd
from opencrab2.highlevel import costmodel as cm
from opencrab2.highlevel import registry as rg
from opencrab2.highlevel.contract import DEFAULT_CONTRACT, SMOLLM_CONTRACT
from opencrab2.lowlevel.interval_mask import SessionBudget

PROFILE = rg.PROFILE_ID
ACTIVE = "Mal2P-active-v1"
PLAN = {"embedding": "wmask", "linear": "wmask", "argmax_k": 4, "max_k": 2}
VERSION = {"model_version": "gpt2@v1", "client_key_domain": "client-default"}


def state(installed=None, stock=None, **kw):
    return cd.SessionState(dict(PLAN), dict(installed or {}), dict(stock or {}), 32, 8, "sess", "gpt2@v1", "nc-hash", **kw)


REQUIRED_NAMES = ["linear_wmask", "linear_generic", "embedding_wmask", "embedding_generic", "matmul_shared", "drelu",
                  "relu", "trunc_faithful", "trunc_local", "softmax_rows_k2", "softmax_rows_k4", "layernorm_rows",
                  "gelu_clip12", "silu_clip12", "argmax_k2", "argmax_k4", "argmax_k8", "ring_to_field", "a2h_exact",
                  "he_linear_bfv_matvec", "toy_bridge", "ckks_fast_h2a", "release_final"]


def test_registry_contents_and_statuses():
    for name in REQUIRED_NAMES:
        assert name in rg.REGISTRY, name
    assert rg.REGISTRY["a2h_exact"].status == rg.STATUS_ADMITTED
    assert rg.REGISTRY["a2h_exact"].params["scope"] == "primitive"
    assert rg.REGISTRY["he_linear_bfv_matvec"].status == rg.STATUS_UNAVAILABLE
    assert rg.REGISTRY["he_linear_bfv_matvec"].entry == ""
    assert rg.REGISTRY["toy_bridge"].status == rg.STATUS_TEST_ONLY
    assert rg.REGISTRY["ckks_fast_h2a"].status == rg.STATUS_UNCERTIFIED_DISABLED
    assert all(c.status in rg.STATUSES for c in rg.REGISTRY.values())
    assert all(PROFILE in c.profiles for c in rg.REGISTRY.values())
    assert not any(ACTIVE in c.profiles for c in rg.REGISTRY.values())
    assert json.dumps(rg.export_registry())
    assert rg.REGISTRY["ring_to_field"].in_domain == "A64" and rg.REGISTRY["ring_to_field"].out_domain == "Zt"
    assert rg.REGISTRY["a2h_exact"].out_domain == "H_BFV"
    assert rg.REGISTRY["ckks_fast_h2a"].in_domain == "H_CKKS"


def test_proof_ledger_covers_every_admitted_component():
    ledger = rg.load_proof_ledger()
    required = ("component", "construction", "reference", "parameters", "corruption_model", "setup",
                "concurrency_scope", "failure_bound", "test_ids", "status")
    for comp in rg.REGISTRY.values():
        if comp.status != rg.STATUS_ADMITTED:
            continue
        assert comp.proof_ledger_id in ledger, comp.name
        entry = ledger[comp.proof_ledger_id]
        for field in required:
            assert field in entry, (comp.name, field)
        assert entry["component"] == comp.name
        assert entry["status"] == rg.STATUS_ADMITTED
        assert entry["concurrency_scope"] == "sequential composition"
        assert "passive" in entry["corruption_model"] or "dealer" in entry["corruption_model"]
        assert isinstance(entry["test_ids"], list) and entry["test_ids"]
    for name in ("he_linear_bfv_matvec", "ckks_fast_h2a"):
        assert rg.REGISTRY[name].proof_ledger_id == ""
    assert ledger["PL-toy_bridge"]["status"] == rg.STATUS_TEST_ONLY
    assert "SecureML" in ledger["PL-trunc_local"]["reference"]
    assert "Beaver" in ledger["PL-linear_generic"]["reference"]
    assert "Eurocrypt 2021" in ledger["PL-drelu"]["reference"]
    assert "Section 5.1" in ledger["PL-trunc_faithful"]["reference"]
    assert "Section 5.2" in ledger["PL-ring_to_field"]["reference"]
    # comparison-key repair: DReLU and every k-ary entry cite the product-group Convert and the regression file
    for lid in ("PL-drelu", "PL-relu", "PL-argmax_k2", "PL-argmax_k4", "PL-argmax_k8", "PL-softmax_rows_k2",
                "PL-softmax_rows_k4"):
        assert "product-group Convert" in ledger[lid]["reference"], lid
        assert "Boyle" in ledger[lid]["reference"], lid
        assert any(t.startswith("tests/test_dcf_component_independence.py") for t in ledger[lid]["test_ids"]), lid
    assert (Path(__file__).parent / "test_dcf_component_independence.py").is_file()
    # the ring_to_field failure bound is the Z_t word term
    assert "n_word * t / 2^128" in ledger["PL-ring_to_field"]["failure_bound"]
    assert ledger["PL-ring_to_field"]["parameters"]["zt_words_per_element"] == rg.ZT_WORDS_PER_R2F_ELEMENT == 516
    assert "n_word * t / 2^128" in rg.REGISTRY["ring_to_field"].failure_bound
    # the k-ary bound is an admission check, not an unchecked precondition
    for lid in ("PL-argmax_k4", "PL-argmax_k8", "PL-softmax_rows_k4"):
        assert "KARY_MAGNITUDE" in ledger[lid]["failure_bound"], lid
        assert "not checked" not in ledger[lid]["failure_bound"], lid
    assert ledger["PL-a2h_exact"]["parameters"]["ciphertext_bytes_measured"] == cm.BFV_CT_BYTES == 432404
    assert "16 (n V + V d)" in ledger["PL-embedding_generic"]["parameters"]["online_bytes"]
    assert "16 (n V + V d)" in rg.REGISTRY["embedding_generic"].params["online_bytes"]
    assert "Fan" in ledger["PL-a2h_exact"]["reference"] and "Brakerski" in ledger["PL-a2h_exact"]["reference"]
    text = Path(rg.LEDGER_PATH).read_text(encoding="utf-8")
    assert chr(0x2014) not in text and chr(0x2013) not in text


@pytest.mark.parametrize("plan", [
    {"embedding": "wmask", "linear": "wmask", "argmax_k": 2, "max_k": 2},
    {"embedding": "generic", "linear": "generic", "argmax_k": 8, "max_k": 4},
    {"embedding": "wmask", "linear": "generic", "argmax_k": 4, "max_k": 4},
])
def test_default_gpt2_tape_is_admitted(plan):
    tape = rg.default_gpt2_tape(plan, n_layers=2)
    ok, code, detail = rg.admit_explain(tape, PROFILE, state())
    assert ok and code == "OK", detail
    assert rg.admit(tape, PROFILE, state()) == (True, "OK")
    assert rg.admit(tape, PROFILE, None) == (True, "OK")
    kinds = {e["kind"] for e in tape}
    assert f"argmax_k{plan['argmax_k']}" in kinds and f"softmax_rows_k{plan['max_k']}" in kinds
    assert tape[-1]["kind"] == "release_final"


def test_active_profile_rejected():
    tape = rg.default_gpt2_tape(PLAN, 1)
    assert rg.admit(tape, ACTIVE, state()) == (False, "PROFILE_MISMATCH")
    assert rg.component_admissible("linear_wmask", ACTIVE) == (False, "PROFILE_MISMATCH")
    assert rg.component_admissible("linear_wmask", PROFILE) == (True, "OK")


def test_disabled_and_unavailable_components_rejected_in_any_plan():
    tape = rg.default_gpt2_tape(PLAN, 1)
    ckks = tape[:3] + [{"kind": "ckks_fast_h2a", "out": "z", "args": {"x": "x0"}}]
    assert rg.admit(ckks, PROFILE, state()) == (False, "NOT_ADMITTED_COMPONENT")
    bfv = tape[:3] + [{"kind": "a2h_exact", "out": "ct", "args": {"x": "x0"}, "shape": (1, 768, 1)},
                      {"kind": "he_linear_bfv_matvec", "out": "ct2", "args": {"x": "ct"}}]
    assert rg.admit(bfv, PROFILE, state()) == (False, "NOT_ADMITTED_COMPONENT")
    primitive_only = tape[:3] + [{"kind": "a2h_exact", "out": "ct", "args": {"x": "x0"}, "shape": (1, 768, 1)}]
    # a2h_exact consumes Z_t keys: its statistical term needs the session budget (paper, Section 6, loss)
    assert rg.admit(primitive_only, PROFILE, state(), budget=SessionBudget(64)) == (True, "OK")
    assert rg.admit(primitive_only, PROFILE, state()) == (False, "BUDGET_EXCEEDED")
    toy = tape[:3] + [{"kind": "toy_bridge", "out": "y", "args": {"x": "x0"}}]
    assert rg.admit(toy, PROFILE, state()) == (False, "NOT_ADMITTED_COMPONENT")
    unknown = tape[:3] + [{"kind": "he_linear", "out": "y", "args": {"x": "x0"}}]
    assert rg.admit(unknown, PROFILE, state()) == (False, "NOT_ADMITTED_COMPONENT")
    assert rg.admit([], PROFILE, state()) == (False, "NOT_ADMITTED_COMPONENT")
    # construction comes before profile in the admission order
    assert rg.admit(ckks, ACTIVE, state()) == (False, "NOT_ADMITTED_COMPONENT")


def test_installed_object_version_mismatch():
    tape = rg.default_gpt2_tape(PLAN, 1)
    stale = state(installed={"wte": {"model_version": "gpt2@v0", "client_key_domain": "client-default"}})
    assert rg.admit(tape, PROFILE, stale) == (False, "OBJECT_VERSION_MISMATCH")
    other_key = state(installed={"h0.qkv": {"model_version": "gpt2@v1", "client_key_domain": "another-client"}})
    assert rg.admit(tape, PROFILE, other_key) == (False, "OBJECT_VERSION_MISMATCH")
    current = state(installed={"wte": dict(VERSION), "h0.qkv": dict(VERSION)})
    assert rg.admit(tape, PROFILE, current) == (True, "OK")
    assert rg.admit(tape, PROFILE, state()) == (True, "OK")              # nothing installed yet: install later
    declared = tape[:3] + [{"kind": "linear_wmask", "out": "y", "args": {"x": "x0"}, "object": "lm_head",
                            "object_version": {"model_version": "gpt2@v9", "client_key_domain": "client-default"}}]
    assert rg.admit(declared, PROFILE, state(installed={"lm_head": dict(VERSION)})) == (False, "OBJECT_VERSION_MISMATCH")


def test_contract_mismatch():
    good = rg.default_gpt2_tape(PLAN, 1, contract_hash="nc-hash")
    bad = rg.default_gpt2_tape(PLAN, 1, contract_hash="other-hash")
    assert rg.admit(good, PROFILE, state()) == (True, "OK")
    assert rg.admit(bad, PROFILE, state()) == (False, "CONTRACT_MISMATCH")


def test_type_errors_surface_with_their_codes_after_profile():
    tape = rg.default_gpt2_tape(PLAN, 1)
    missing = [tape[0], {"kind": "embedding_generic", "out": "x0", "args": {"x": "tok"}, "shape": (1, 50257, 768)}]
    assert rg.admit(missing, PROFILE, state()) == (False, "MISSING_CONVERSION")
    assert rg.admit(missing, ACTIVE, state()) == (False, "PROFILE_MISMATCH")     # profile before types
    fed_back = tape + [{"kind": "layernorm_rows", "out": "again", "args": {"x": "token"}, "shape": (1, 768, 768)}]
    assert rg.admit(fed_back, PROFILE, state()) == (False, "MISSING_CONVERSION")   # a released token must be re-shared
    zt_into_ring = tape[:3] + [{"kind": "ring_to_field", "out": "z", "args": {"x": "x0"}, "shape": (1, 768, 1)},
                               {"kind": "layernorm_rows", "out": "y", "args": {"x": "z"}, "shape": (1, 768, 768)}]
    assert rg.admit(zt_into_ring, PROFILE, state()) == (False, "DOMAIN_MISMATCH")


def test_budget_checks():
    tape = rg.default_gpt2_tape(PLAN, 1)
    declared = list(tape)
    declared[3] = dict(declared[3], log2_eps=-7, calls=1)
    assert rg.admit(declared, PROFILE, state(), budget=None) == (False, "BUDGET_EXCEEDED")
    assert rg.admit(declared, PROFILE, state(), budget=SessionBudget(64)) == (False, "BUDGET_EXCEEDED")
    tiny = list(tape)
    tiny[3] = dict(tiny[3], log2_eps=-68, calls=4)                                   # 2^-66 per admitted plan
    b = SessionBudget(64)
    assert rg.admit(tiny, PROFILE, state(), budget=b) == (True, "OK")
    assert rg.admit(tiny, PROFILE, state(), budget=b) == (True, "OK")
    assert rg.admit(tiny, PROFILE, state(), budget=b) == (True, "OK")
    assert rg.admit(tiny, PROFILE, state(), budget=b) == (True, "OK")                # 4 x 2^-66 = 2^-64 exactly
    assert rg.admit(tiny, PROFILE, state(), budget=b) == (False, "BUDGET_EXCEEDED")  # the fifth would exceed 2^-64
    assert rg.admit(tape, PROFILE, state(), budget=SessionBudget(64)) == (True, "OK")   # no statistical term declared


def test_material_and_memory_resources():
    tape = rg.default_gpt2_tape(PLAN, 1)
    need = rg.tape_material(tape)
    assert need["drelu"] > 0 and need["wmask_mask"] > 0 and need["trunc_faithful"] > 0
    rich = state(stock={t: (1, b) for t, b in need.items()})
    assert rg.admit(tape, PROFILE, rich) == (True, "OK")
    poor = state(stock={t: (1, b) for t, b in need.items()})
    poor.stock["drelu"] = (1, need["drelu"] - 1)
    assert rg.admit(tape, PROFILE, poor) == (False, "INSUFFICIENT_MATERIAL")
    heavy = [dict(e, mem_bytes=1 << 30) for e in tape]
    assert rg.admit(heavy, PROFILE, state(mem_limit_bytes=1 << 31)) == (False, "MEMORY_INFEASIBLE")
    assert rg.admit(heavy, PROFILE, state(mem_limit_bytes=1 << 40)) == (True, "OK")
    assert rg.admit(heavy, PROFILE, state()) == (True, "OK")
    relu_entry = {"kind": "relu", "out": "r", "args": {"x": "x0"}, "shape": (2, 10, 1)}
    assert rg.entry_material(relu_entry)["drelu"] == 20 * 2200
    assert rg.entry_material({"kind": "release_final", "out": "t", "args": {"x": "x0"}}) == {}


def test_admission_error_type_and_codes_are_stable():
    assert set(rg.REASON_CODES) == {"NOT_ADMITTED_COMPONENT", "PROFILE_MISMATCH", "MISSING_CONVERSION", "DOMAIN_MISMATCH",
                                    "CONTRACT_MISMATCH", "BUDGET_EXCEEDED", "INSUFFICIENT_MATERIAL", "MEMORY_INFEASIBLE",
                                    "OBJECT_VERSION_MISMATCH"}
    err = AdmissionError("detail", code="PROFILE_MISMATCH")
    assert str(err).startswith("PROFILE_MISMATCH:") and err.code == "PROFILE_MISMATCH"
    src = Path(rg.__file__).read_text(encoding="utf-8")
    assert chr(0x2014) not in src and chr(0x2013) not in src



# ---------------------------------------------------------------------------
# declared magnitude bound of fused k-ary nodes (BUDGET_EXCEEDED, KARY_MAGNITUDE)
# ---------------------------------------------------------------------------
def _site(kind, row_len, **bounds):
    """A minimal typed tape with one k-ary site of the given row length."""
    tape = [{"kind": "input_client_plain", "out": "tok", "args": {}},
            {"kind": "share_input", "out": "x", "args": {"x": "tok"}, "shape": (1, row_len, 1)}]
    e = {"kind": kind, "out": "y", "args": {"x": "x"}, "shape": (1, row_len, row_len)}
    e.update(bounds)
    return tape + [e]


def test_kary_limits_from_the_contract():
    # n = 32: binary nodes keep the DCF bound, fused nodes lose 1 + ceil(log2 g) bits
    assert [rg.kary_limit_log2(32, g) for g in (2, 3, 4, 5, 8)] == [31, 28, 28, 27, 27]
    # the limit on the declared operand bound B (real units, |x_i - x_j| <= 2 B) at n = 32, f = 16
    assert rg.kary_bound_limit(2, 50257, 32, 16) == 2 ** 14
    assert rg.kary_bound_limit(4, 50257, 32, 16) == 2 ** 11
    assert rg.kary_bound_limit(8, 50257, 32, 16) == 2 ** 10
    # the node sizes follow the public row length: 8 items at k = 8 is one node of 8, 9 items adds a pass-through
    assert cm.tournament_node_sizes(8, 8) == (8,) and cm.tournament_node_sizes(9, 8) == (2, 8)
    assert cm.tournament_node_sizes(4, 8) == (4,) and rg.kary_bound_limit(8, 4, 32, 16) == 2 ** 11
    assert cm.tournament_node_sizes(2, 8) == (2,) and rg.kary_bound_limit(8, 2, 32, 16) == 2 ** 14
    assert cm.tournament_node_sizes(1, 4) == ()
    assert rg.DEFAULT_LOGITS_BOUND == 512.0 and rg.DEFAULT_SCORE_BOUND == 512.0
    # both shipped contracts have n = 32, f = 16: the defaults fit every k
    for c in (DEFAULT_CONTRACT, SMOLLM_CONTRACT):
        assert (c.dcf_domain_bits, c.frac_bits) == (32, 16)
        for kind in rg.KARY_KINDS:
            assert rg.kary_site_check({"kind": kind, "shape": (1, 50257, 1)}, c) == (True, "OK")


@pytest.mark.parametrize("kind,key,ok_bound,bad_bound", [
    ("argmax_k2", "logits_bound", 16383.0, 16384.0),
    ("argmax_k4", "logits_bound", 2047.0, 2048.0),
    ("argmax_k8", "logits_bound", 1023.0, 1024.0),
    ("softmax_rows_k2", "score_bound", 16383.0, 16384.0),
    ("softmax_rows_k4", "score_bound", 2047.0, 2048.0),
])
def test_kary_magnitude_admission_check(kind, key, ok_bound, bad_bound):
    good = _site(kind, 64, **{key: ok_bound})
    assert rg.admit(good, PROFILE, state()) == (True, "OK")
    bad = _site(kind, 64, **{key: bad_bound})
    ok, code, detail = rg.admit_explain(bad, PROFILE, state())
    assert (ok, code) == (False, "BUDGET_EXCEEDED") and detail.startswith("KARY_MAGNITUDE"), detail
    assert rg.admit(bad, PROFILE, state()) == (False, "BUDGET_EXCEEDED")
    assert rg.admit(bad, PROFILE, None) == (False, "BUDGET_EXCEEDED")       # no session needed: a tape property
    # the generic key is accepted for either site kind and wins over the specific one
    assert rg.admit(_site(kind, 64, operand_bound=bad_bound, **{key: ok_bound}), PROFILE, state()) == (False, "BUDGET_EXCEEDED")
    # the default declaration (512) is used when the entry declares nothing
    assert rg.admit(_site(kind, 64), PROFILE, state()) == (True, "OK")
    # invalid declarations fail closed
    for junk in (0.0, -1.0, float("inf"), float("nan"), "512", True):
        ok, code, detail = rg.admit_explain(_site(kind, 64, **{key: junk}), PROFILE, state())
        assert (ok, code) == (False, "BUDGET_EXCEEDED") and detail.startswith("KARY_MAGNITUDE"), junk


def test_kary_magnitude_depends_on_k_row_length_and_contract():
    # the same declared bound is fine at k = 2 and k = 4 and refused at k = 8
    for kind, want in (("argmax_k2", True), ("argmax_k4", True), ("argmax_k8", False)):
        assert rg.admit(_site(kind, 50257, logits_bound=1500.0), PROFILE, state())[0] is want, kind
    # a short row never builds a node of more than four members, so k = 8 gets the k = 4 limit there
    assert rg.admit(_site("argmax_k8", 4, logits_bound=1500.0), PROFILE, state()) == (True, "OK")
    assert rg.admit(_site("argmax_k8", 5, logits_bound=1500.0), PROFILE, state()) == (False, "BUDGET_EXCEEDED")
    # two items are one binary node for every k
    assert rg.admit(_site("argmax_k8", 2, logits_bound=10000.0), PROFILE, state()) == (True, "OK")
    # a contract with a narrower DCF domain tightens the limit: n = 28 gives 2^6 at k = 8
    import dataclasses
    narrow = dataclasses.replace(DEFAULT_CONTRACT, dcf_domain_bits=28, name="NC-test-n28")
    assert rg.kary_site_check({"kind": "argmax_k8", "shape": (1, 64, 1), "logits_bound": 63.0}, narrow) == (True, "OK")
    ok, detail = rg.kary_site_check({"kind": "argmax_k8", "shape": (1, 64, 1), "logits_bound": 64.0}, narrow)
    assert not ok and detail.startswith("KARY_MAGNITUDE") and "n = 28" in detail
    tape = _site("argmax_k8", 64, logits_bound=512.0)
    assert rg.admit(tape, PROFILE, state()) == (True, "OK")
    assert rg.admit(tape, PROFILE, state(), contract=narrow) == (False, "BUDGET_EXCEEDED")
    # the contract is resolved from the session hash when it is a registered one
    st = cd.SessionState(dict(PLAN), {}, {}, 32, 8, "sess", "gpt2@v1", SMOLLM_CONTRACT.contract_hash())
    assert rg.resolve_contract(st) is SMOLLM_CONTRACT
    assert rg.resolve_contract(state()) is DEFAULT_CONTRACT and rg.resolve_contract(None) is DEFAULT_CONTRACT
    assert rg.resolve_contract(st, narrow) is narrow
    # an argmax site without a declared shape cannot have its index range checked: refused for fused nodes
    no_shape = _site("argmax_k4", 64)
    del no_shape[-1]["shape"]
    ok, code, detail = rg.admit_explain(no_shape, PROFILE, state())
    assert (ok, code) == (False, "BUDGET_EXCEEDED") and "KARY_MAGNITUDE" in detail
    # the index path: a fused node compares index differences up to the row length
    tiny = cm.ProtocolParams(dcf_domain_bits=20, frac_bits=4)            # the check reads (n, f) only
    assert rg.kary_site_check({"kind": "argmax_k8", "shape": (1, 2 ** 15, 1), "logits_bound": 1.0}, tiny) == (True, "OK")
    ok, detail = rg.kary_site_check({"kind": "argmax_k8", "shape": (1, 2 ** 15 + 1, 1), "logits_bound": 1.0}, tiny)
    assert not ok and "index range" in detail


def test_default_gpt2_tape_declares_the_kary_bounds():
    tape = rg.default_gpt2_tape(PLAN, 2)
    sm = [e for e in tape if e["kind"].startswith("softmax_rows_k")]
    am = [e for e in tape if e["kind"].startswith("argmax_k")]
    assert len(sm) == 2 and len(am) == 1
    assert all(e["score_bound"] == rg.DEFAULT_SCORE_BOUND for e in sm) and am[0]["logits_bound"] == rg.DEFAULT_LOGITS_BOUND
    # a deployment that declares larger logits loses the 8-ary argmax but keeps k = 2 and k = 4
    shapes = dict(cd.gpt2_shapes(n_layers=2, n=1, T=32), logits_bound=1500.0)
    verdict = {}
    for k in (2, 4, 8):
        plan = dict(PLAN, argmax_k=k)
        verdict[k] = rg.admit_explain(rg.default_gpt2_tape(plan, 2, shapes), PROFILE, state())
    assert verdict[2][0] and verdict[4][0] and not verdict[8][0]
    assert verdict[8][1] == "BUDGET_EXCEEDED" and verdict[8][2].startswith("KARY_MAGNITUDE")
    # the same gate reaches the candidate enumeration through admit_fn: the 8-ary plans are infeasible
    model = cm.FittedCostModel(label="MODELED")
    for comp in cm.COMPONENTS:
        model.fits[comp] = cm.ComponentFit(1e-9, 0.0, 0.001, 0.0005, 1, "MODELED", "toy")
    cat = cd.gpt2_catalog(shapes)
    st = state()
    cands = cd.enumerate_candidates(st, cat, model, cm.Env(100e6, 0.01), shapes,
                                    admit_fn=lambda p: rg.admit(rg.default_gpt2_tape(p, 2, shapes), PROFILE, st))
    assert all(c.feasible == (c.plan["argmax_k"] != 8) for c in cands)
    assert all(c.reason == "BUDGET_EXCEEDED" for c in cands if c.plan["argmax_k"] == 8)
    # scores above the 4-ary limit remove max_k = 4 only
    shapes2 = dict(cd.gpt2_shapes(n_layers=2, n=1, T=32), score_bound=3000.0)
    for mk, want in ((2, True), (4, False)):
        assert rg.admit(rg.default_gpt2_tape(dict(PLAN, max_k=mk), 2, shapes2), PROFILE, state())[0] is want


def test_kary_check_comes_after_contract_and_before_resources():
    bad = rg.default_gpt2_tape(PLAN, 1, dict(cd.gpt2_shapes(n_layers=1), logits_bound=5000.0), contract_hash="other-hash")
    assert rg.admit(bad, PROFILE, state()) == (False, "CONTRACT_MISMATCH")           # contract first
    bad2 = rg.default_gpt2_tape(PLAN, 1, dict(cd.gpt2_shapes(n_layers=1), logits_bound=5000.0))
    poor = state(stock={"drelu": (1, 1)})
    assert rg.admit(bad2, PROFILE, poor) == (False, "BUDGET_EXCEEDED")               # numerics before resources
    assert rg.admit(bad2, ACTIVE, state()) == (False, "PROFILE_MISMATCH")            # profile before numerics


# ---------------------------------------------------------------------------
# Z_t key words are a statistical term of the session budget (paper, Section 6, loss)
# ---------------------------------------------------------------------------
def test_zt_statistical_term_is_charged_to_the_budget():
    t = 68720050177
    per_elem = rg.zt_log2_eps_per_element(t)
    assert math.isclose(per_elem, math.log2(516 * t) - 128.0) and -83.0 < per_elem < -82.9
    assert rg.REGISTRY["ring_to_field"].params["zt_words_per_element"] == 516
    head = rg.default_gpt2_tape(PLAN, 1)[:3]
    r2f = head + [{"kind": "ring_to_field", "out": "z", "args": {"x": "x0"}, "shape": (1, 768, 1)}]
    assert rg.admit(r2f, PROFILE, state()) == (False, "BUDGET_EXCEEDED")              # no budget: fail closed
    b = SessionBudget(64)
    assert rg.admit(r2f, PROFILE, state(), budget=b) == (True, "OK")
    charge = b.charges[-1]
    assert charge["calls"] == 768 and math.isclose(charge["log2_eps"], per_elem)
    assert math.isclose(b.spent_log2(), per_elem + math.log2(768), abs_tol=1e-6)
    # 2^19 elements exhaust kappa_sys = 64 (2^-82.99 x 2^19 > 2^-64): refused, nothing charged
    huge = head + [{"kind": "ring_to_field", "out": "z", "args": {"x": "x0"}, "shape": (1024, 1024, 1)}]
    b2 = SessionBudget(64)
    ok, code, detail = rg.admit_explain(huge, PROFILE, state(), budget=b2)
    assert (ok, code) == (False, "BUDGET_EXCEEDED") and b2.charges == []
    assert rg.admit(head + [{"kind": "ring_to_field", "out": "z", "args": {"x": "x0"}, "shape": (256, 1024, 1)}],
                    PROFILE, state(), budget=SessionBudget(64)) == (True, "OK")
    # the element count includes the declared repetition; an entry without a shape cannot be counted
    rep4 = head + [{"kind": "a2h_exact", "out": "ct", "args": {"x": "x0"}, "shape": (2, 100, 1), "count": 4}]
    b3 = SessionBudget(64)
    assert rg.admit(rep4, PROFILE, state(), budget=b3) == (True, "OK") and b3.charges[-1]["calls"] == 800
    no_shape = head + [{"kind": "ring_to_field", "out": "z", "args": {"x": "x0"}}]
    ok, code, detail = rg.admit_explain(no_shape, PROFILE, state(), budget=SessionBudget(64))
    assert (ok, code) == (False, "BUDGET_EXCEEDED") and detail.startswith("ZT_WORDS")
    # a larger admitted modulus costs proportionally more
    assert rg.zt_log2_eps_per_element(2 ** 61 - 1) > per_elem
    with pytest.raises(ValueError, match="^ZT_PARAM"):
        rg.zt_log2_eps_per_element(1)
    # the default GPT-2 tape has no Z_t key: it needs no budget
    assert rg.admit(rg.default_gpt2_tape(PLAN, 1), PROFILE, state()) == (True, "OK")

