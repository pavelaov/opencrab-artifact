"""Tests for opencrab2.highlevel.candidates (state, catalog, candidate enumeration, feasibility)."""

from __future__ import annotations

import itertools
import math

import pytest

from opencrab2.highlevel import candidates as cd
from opencrab2.highlevel import costmodel as cm

ENV = cm.Env(100e6, 0.01)


def toy_model() -> cm.FittedCostModel:
    """Deterministic coefficients for every component (no file dependency)."""
    m = cm.FittedCostModel(label="MODELED")
    for comp in cm.COMPONENTS:
        m.fits[comp] = cm.ComponentFit(1e-9, 0.0, 0.001, 0.0005, 1, "MODELED", "toy")
    return m


def state(incumbent=None, installed=None, stock=None, remaining=8, version="gpt2@v1", **kw) -> cd.SessionState:
    inc = incumbent or {"embedding": "generic", "linear": "generic", "argmax_k": 2, "max_k": 2}
    return cd.SessionState(inc, dict(installed or {}), dict(stock or {}), 32, remaining, "sess-1", version, "nc-hash", **kw)


@pytest.fixture
def setup():
    shapes = cd.gpt2_shapes(n_layers=2, n=1, T=32)
    return shapes, cd.gpt2_catalog(shapes), toy_model()


def test_keep_first_and_full_product(setup):
    shapes, cat, model = setup
    st = state()
    cands = cd.enumerate_candidates(st, cat, model, ENV, shapes)
    n_plans = 1
    for vals in cat["axes"].values():
        n_plans *= len(vals)
    assert len(cands) == n_plans == 24
    assert cands[0].is_keep and cands[0].plan == st.incumbent
    assert sum(1 for c in cands if c.is_keep) == 1
    keys = [cd.plan_key(c.plan) for c in cands]
    assert len(set(keys)) == len(keys)
    expected = {cd.plan_key(dict(zip(cat["axes"], combo))) for combo in itertools.product(*cat["axes"].values())}
    assert set(keys) == expected


def test_install_cost_is_candidate_specific(setup):
    shapes, cat, model = setup
    st = state()
    cands = {cd.plan_key(c.plan): c for c in cd.enumerate_candidates(st, cat, model, ENV, shapes)}
    keep = cands[cd.plan_key(st.incumbent)]
    assert keep.install_objects == ["lm_head"]                     # cold start: the LM head is always weight masked
    w_all = cands[cd.plan_key({"embedding": "wmask", "linear": "wmask", "argmax_k": 2, "max_k": 2})]
    w_emb = cands[cd.plan_key({"embedding": "wmask", "linear": "generic", "argmax_k": 2, "max_k": 2})]
    assert set(w_emb.install_objects) == {"lm_head", "wte"}
    assert set(w_all.install_objects) == {"lm_head", "wte"} | {f"h{li}.{s}" for li in range(2) for s in ("qkv", "out", "fc", "proj")}
    assert w_all.c_install_s > w_emb.c_install_s > keep.c_install_s > 0
    # the argmax arity never needs an object
    k8 = cands[cd.plan_key({"embedding": "generic", "linear": "generic", "argmax_k": 8, "max_k": 2})]
    assert k8.install_objects == ["lm_head"] and k8.c_install_s == keep.c_install_s
    assert k8.exec_s_per_epoch < keep.exec_s_per_epoch            # fewer rounds under RTT 20 ms


def test_installed_objects_at_current_version_are_free(setup):
    shapes, cat, model = setup
    st = state(installed={"wte": {"model_version": "gpt2@v1", "client_key_domain": "client-default"},
                          "lm_head": {"model_version": "gpt2@v1", "client_key_domain": "client-default"}})
    cands = {cd.plan_key(c.plan): c for c in cd.enumerate_candidates(st, cat, model, ENV, shapes)}
    w_emb = cands[cd.plan_key({"embedding": "wmask", "linear": "generic", "argmax_k": 2, "max_k": 2})]
    assert w_emb.install_objects == [] and w_emb.c_install_s == 0.0 and w_emb.c_install_sigma == 0.0


def test_version_mismatch_recharges_install(setup):
    shapes, cat, model = setup
    stale = state(installed={"wte": {"model_version": "gpt2@v0", "client_key_domain": "client-default"}})
    assert not stale.is_installed("wte")
    other_client = state(installed={"wte": {"model_version": "gpt2@v1", "client_key_domain": "other-client"}})
    assert not other_client.is_installed("wte")
    cands = {cd.plan_key(c.plan): c for c in cd.enumerate_candidates(stale, cat, model, ENV, shapes)}
    w_emb = cands[cd.plan_key({"embedding": "wmask", "linear": "generic", "argmax_k": 2, "max_k": 2})]
    assert "wte" in w_emb.install_objects


def test_material_feasibility_by_type(setup):
    shapes, cat, model = setup
    st = state()
    need = cd.material_per_epoch(st.incumbent, shapes)
    assert need["drelu"] > 0 and need["matrix_triple"] > 0 and need["trunc_faithful"] > 0
    rich = state(stock={t: (1, b * 2) for t, b in need.items()})
    poor = state(stock={t: (1, b * 2) for t, b in need.items()})
    poor.stock["drelu"] = (1, need["drelu"] // 2)
    assert cd.enumerate_candidates(rich, cat, model, ENV, shapes)[0].feasible
    keep_poor = cd.enumerate_candidates(poor, cat, model, ENV, shapes)[0]
    assert not keep_poor.feasible and keep_poor.reason.startswith("INSUFFICIENT_MATERIAL")
    # an empty stock dict declares no inventory constraint
    assert all(c.feasible for c in cd.enumerate_candidates(st, cat, model, ENV, shapes))


def test_memory_feasibility(setup):
    shapes, cat, model = setup
    st = state(mem_limit_bytes=1)
    cands = cd.enumerate_candidates(st, cat, model, ENV, shapes)
    assert all((not c.feasible) and c.reason.startswith("MEMORY_INFEASIBLE") for c in cands)
    big = state(mem_limit_bytes=1 << 40)
    assert all(c.feasible for c in cd.enumerate_candidates(big, cat, model, ENV, shapes))
    peaks = [c.mem_peak_bytes for c in cands]
    assert all(p > 0 for p in peaks)


def test_admit_hook_marks_rejected_candidates(setup):
    shapes, cat, model = setup
    st = state()

    def gate(plan):
        return (False, "NOT_ADMITTED_COMPONENT") if plan["linear"] == "generic" else (True, "OK")

    cands = cd.enumerate_candidates(st, cat, model, ENV, shapes, admit_fn=gate)
    assert cands[0].is_keep and not cands[0].feasible and cands[0].reason == "NOT_ADMITTED_COMPONENT"
    assert all(c.feasible == (c.plan["linear"] == "wmask") for c in cands)


def test_plan_calls_structure(setup):
    shapes, cat, model = setup
    plan = {"embedding": "wmask", "linear": "wmask", "argmax_k": 4, "max_k": 4}
    calls = cd.plan_calls(plan, shapes)
    comps = [c[0] for c in calls]
    assert comps[0] == "embedding_wmask"
    assert comps.count("linear_wmask") == 4 * shapes["n_layers"] + 1
    assert comps.count("softmax_rows_k4") == shapes["n_layers"]
    assert comps.count("argmax_k4") == 1 and "argmax_k2" not in comps
    assert comps.count("layernorm_rows") == 2 * shapes["n_layers"] + 1
    sm = [c for c in calls if c[0] == "softmax_rows_k4"][0]
    assert sm[1] == shapes["n_head"] * shapes["n"] and sm[2] == shapes["T"]
    mean, sigma = cd.exec_cost(plan, shapes, model, ENV)
    assert mean > 0 and sigma > 0


def test_normalize_plan_rejects_bad_values(setup):
    shapes, cat, model = setup
    with pytest.raises(ValueError, match="^PLAN_VALUE_INVALID"):
        cd.normalize_plan({"embedding": "wmask", "linear": "wmask", "argmax_k": 3, "max_k": 2}, cat)
    with pytest.raises(ValueError, match="^PLAN_AXIS_MISSING"):
        cd.normalize_plan({"embedding": "wmask"}, cat)
    assert cd.plan_key({"b": 1, "a": 2}) == "a=2|b=1"


def test_install_cost_is_itemized_per_object(setup):
    shapes, cat, model = setup
    st = state()
    for c in cd.enumerate_candidates(st, cat, model, ENV, shapes):
        assert set(c.install_cost_by_object) == set(c.install_objects)
        assert math.isclose(sum(c.install_cost_by_object.values()), c.c_install_s, rel_tol=1e-12, abs_tol=1e-15)
        assert c.c_wait_s == 0.0
        d = c.to_dict()
        assert d["install_cost_by_object"] == c.install_cost_by_object and d["c_wait_s"] == 0.0
    per_obj = cd.install_costs(["wte", "lm_head"], cat, model, ENV)
    assert set(per_obj) == {"wte", "lm_head"}
    mean, sigma = cd.install_cost(["wte", "lm_head"], cat, model, ENV)
    assert math.isclose(mean, per_obj["wte"][0] + per_obj["lm_head"][0])
    assert math.isclose(sigma, math.hypot(per_obj["wte"][1], per_obj["lm_head"][1]))
    # the same object costs the same in every candidate that needs it (the planner chain relies on it)
    cands = cd.enumerate_candidates(st, cat, model, ENV, shapes)
    wte = {c.install_cost_by_object["wte"] for c in cands if "wte" in c.install_objects}
    assert len(wte) == 1


def test_attention_products_are_one_batched_call(setup):
    """transformer.attention opens the stacked heads in one round: count products share one round."""
    shapes, cat, model = setup
    H = shapes["n_head"]
    assert cd.BATCHED_COMPONENTS == ("matmul_shared",)
    one, var1 = cd.call_cost("matmul_shared", 1, 64, 32, 1, model, ENV)
    batch, varH = cd.call_cost("matmul_shared", 1, 64, 32, H, model, ENV)
    rounds = cm.rounds_formula("matmul_shared", 1, 64, 32)
    assert rounds == 1
    assert math.isclose(batch, H * one - (H - 1) * rounds * ENV.rtt_s, rel_tol=1e-12)
    assert math.isclose(varH, H * var1)
    # an unbatched component is count independent calls
    two, _ = cd.call_cost("layernorm_rows", 1, 768, 768, 2, model, ENV)
    assert math.isclose(two, 2 * model.predict("layernorm_rows", 1, 768, 768, ENV)[0])
    # exec_cost is the sum of the entry costs, so one epoch saves 2 (H - 1) round trips per layer
    plan = {"embedding": "wmask", "linear": "wmask", "argmax_k": 2, "max_k": 2}
    naive = sum(count * model.predict(c, n, a, b, ENV)[0] for c, n, a, b, count in cd.plan_calls(plan, shapes))
    mean, _ = cd.exec_cost(plan, shapes, model, ENV)
    assert math.isclose(naive - mean, shapes["n_layers"] * 2 * (H - 1) * ENV.rtt_s, rel_tol=1e-9)
    # the correction is plan independent: plan differences are unchanged
    other = dict(plan, argmax_k=8, max_k=4)
    naive_o = sum(count * model.predict(c, n, a, b, ENV)[0] for c, n, a, b, count in cd.plan_calls(other, shapes))
    assert math.isclose(naive - naive_o, mean - cd.exec_cost(other, shapes, model, ENV)[0], rel_tol=1e-9, abs_tol=1e-12)


def test_plan_bytes_rounds_matches_the_component_formulas(setup):
    shapes, cat, model = setup
    H, L = shapes["n_head"], shapes["n_layers"]
    plan = {"embedding": "generic", "linear": "wmask", "argmax_k": 4, "max_k": 4}
    nbytes, rounds = cd.plan_bytes_rounds(plan, shapes)
    calls = cd.plan_calls(plan, shapes)
    naive_rounds = sum(count * cm.rounds_formula(c, n, a, b) for c, n, a, b, count in calls)
    naive_bytes = sum(count * cm.bytes_formula(c, n, a, b) for c, n, a, b, count in calls)
    assert rounds == naive_rounds - L * 2 * (H - 1)
    assert nbytes == naive_bytes - L * 2 * (H - 1) * 2 * cm.FRAME_BYTES
    # payload is untouched by the batching
    payload = sum(count * cm.payload_bytes(c, n, a, b) for c, n, a, b, count in calls)
    frames = nbytes - payload
    assert frames % cm.FRAME_BYTES == 0 and frames > 0
    # fewer rounds for the larger arity, at the price of more bytes (fused k-ary pricing)
    b2, r2 = cd.plan_bytes_rounds(dict(plan, argmax_k=2, max_k=2), shapes)
    b8, r8 = cd.plan_bytes_rounds(dict(plan, argmax_k=8, max_k=4), shapes)
    assert r8 < r2 and b8 > b2


def test_material_per_epoch_uses_the_fused_tournament_counts(setup):
    shapes, cat, model = setup
    V = shapes["vocab"]
    base = {"embedding": "wmask", "linear": "wmask", "argmax_k": 2, "max_k": 2}
    m2 = cd.material_per_epoch(base, shapes)
    m8 = cd.material_per_epoch(dict(base, argmax_k=8), shapes)
    _, p1_2, p2_2, bv_2 = cm.tournament_gates(V, 2, True)
    _, p1_8, p2_8, bv_8 = cm.tournament_gates(V, 8, True)
    assert m8["drelu"] - m2["drelu"] == (p1_8 - p1_2) * 1672 + (p2_8 - p2_2) * 2200
    assert m2["kary_select"] - m8.get("kary_select", 0) == (bv_2 - bv_8) * 24
    assert m8["drelu"] > m2["drelu"]                              # the 8-ary argmax spends more DReLU material


# ---------------------------------------------------------------------------
# horizon schedule: per-step costs (paper, Section 4.2, objective) and the pricing memo
# ---------------------------------------------------------------------------
class CountingModel:
    """Delegates to a cost model and counts its ``predict`` calls."""

    def __init__(self, model):
        self.model = model
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self.model, name)

    def predict(self, *args):
        self.calls += 1
        return self.model.predict(*args)


def test_schedule_prices_every_step_and_entry_zero_is_the_current_epoch(setup):
    shapes, cat, model = setup                                     # one decode row, KV bucket 32
    prefill = cd.gpt2_shapes(n_layers=2, n=16, T=32)
    cat_p = cd.gpt2_catalog(prefill)
    longer = cd.gpt2_shapes(n_layers=2, n=1, T=64)
    sched = [prefill, shapes, shapes, longer]
    st = state(remaining=4)
    cands = cd.enumerate_candidates(st, cat_p, model, ENV, prefill, schedule=sched)
    for c in cands:
        assert len(c.exec_s_by_epoch) == len(c.exec_sigma_by_epoch) == 4
        assert c.exec_s_by_epoch[0] == c.exec_s_per_epoch and c.exec_sigma_by_epoch[0] == c.exec_sigma
        for j, sh in enumerate(sched):
            mu, sg = cd.exec_cost(c.plan, sh, model, ENV)
            assert math.isclose(c.exec_s_by_epoch[j], mu, rel_tol=1e-12) and math.isclose(c.exec_sigma_by_epoch[j], sg, rel_tol=1e-12)
        assert c.exec_s_by_epoch[1] == c.exec_s_by_epoch[2]        # one shape, one price
        assert c.exec_s_by_epoch[0] > c.exec_s_by_epoch[1] and c.exec_s_by_epoch[3] > c.exec_s_by_epoch[1]
        d = c.to_dict()
        assert d["exec_s_by_epoch"] == c.exec_s_by_epoch and d["exec_sigma_by_epoch"] == c.exec_sigma_by_epoch
    # install, material and memory are those of the current epoch, exactly as without a schedule
    plain = cd.enumerate_candidates(st, cat_p, model, ENV, prefill)
    for a, b in zip(plain, cands):
        assert a.plan == b.plan and a.install_objects == b.install_objects and a.c_install_s == b.c_install_s
        assert a.material_by_type == b.material_by_type and a.mem_peak_bytes == b.mem_peak_bytes
        assert a.exec_s_per_epoch == b.exec_s_per_epoch and a.exec_sigma == b.exec_sigma
        assert a.exec_s_by_epoch == [] and a.exec_sigma_by_epoch == []
    assert cd.shape_key(prefill) != cd.shape_key(shapes) and cd.shape_key(shapes) == cd.shape_key(dict(shapes, mem_limit_bytes=1))
    with pytest.raises(ValueError, match="^SCHEDULE_FIRST_STEP"):
        cd.enumerate_candidates(st, cat_p, model, ENV, prefill, schedule=[shapes, shapes])


def test_memo_prices_each_shape_once_and_returns_the_unmemoised_values(setup):
    shapes, cat, model = setup
    prefill = cd.gpt2_shapes(n_layers=2, n=16, T=32)
    cat_p = cd.gpt2_catalog(prefill)
    sched = [prefill] + [shapes] * 7
    st = state(remaining=8)
    counting = CountingModel(model)
    plain = cd.enumerate_candidates(st, cat_p, counting, ENV, prefill, schedule=sched)
    n_plain, counting.calls = counting.calls, 0
    memo = {}
    first = cd.enumerate_candidates(st, cat_p, counting, ENV, prefill, schedule=sched, memo=memo)
    n_first, counting.calls = counting.calls, 0
    second = cd.enumerate_candidates(st, cat_p, counting, ENV, prefill, schedule=sched, memo=memo)
    n_second, counting.calls = counting.calls, 0
    assert [c.to_dict() for c in plain] == [c.to_dict() for c in first] == [c.to_dict() for c in second]
    assert 0 < n_second < n_first < n_plain                    # the second pass predicts the install costs only
    # the memo shares the entries of plan_calls between plans and prices the two shapes once each
    assert n_first < sum(len(cd.plan_calls(c.plan, prefill)) for c in plain)
    assert any(k[0] == "exec" for k in memo) and any(k[0] == "material" for k in memo)
    # another environment is priced afresh
    other = cd.enumerate_candidates(st, cat_p, counting, cm.Env(2e6, 0.0005), prefill, schedule=sched, memo=memo)
    assert counting.calls > n_second and other[0].exec_s_per_epoch != first[0].exec_s_per_epoch
    # exec_cost alone: with and without the memo the same (mean, sigma)
    plan = {"embedding": "wmask", "linear": "wmask", "argmax_k": 4, "max_k": 4}
    assert cd.exec_cost(plan, prefill, model, ENV, {}) == cd.exec_cost(plan, prefill, model, ENV)

