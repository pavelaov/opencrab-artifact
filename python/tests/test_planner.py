"""Tests for opencrab2.highlevel.planner: exactness against brute force, the level-aware state, admission gate."""

from __future__ import annotations

import math

import numpy as np
import pytest

from opencrab2.highlevel import candidates as cd
from opencrab2.highlevel import costmodel as cm
from opencrab2.highlevel import planner as pl

INF = float("inf")
DOMAINS = ("A64", "Zt", "H_BFV")


def random_chain(rng, n_ops, max_real=3, n_objects=3):
    chain = []
    for i in range(n_ops):
        reals = []
        for r in range(int(rng.integers(1, max_real + 1))):
            in_d = DOMAINS[int(rng.integers(0, 3))]
            out_d = DOMAINS[int(rng.integers(0, 3))]
            real = {"name": f"r{r}", "in_domain": in_d, "out_domain": out_d, "cost": float(rng.uniform(1.0, 10.0))}
            if rng.random() < 0.5:
                real["objects"] = [(f"o{int(rng.integers(0, n_objects))}", float(rng.uniform(0.0, 20.0)))]
            if out_d == "H_BFV":
                real["level_delta"] = -int(rng.integers(0, 3))
                if in_d == "H_BFV" and rng.random() < 0.5:
                    real["level_min"] = int(rng.integers(1, 3))
            reals.append(real)
        chain.append({"id": f"op{i}", "realizations": reals})
    return chain


def random_catalog(rng):
    conv = {}
    if rng.random() < 0.8:
        conv["A64->Zt"] = {"cost": float(rng.uniform(0.0, 5.0))}
    if rng.random() < 0.8:
        conv["Zt->H_BFV"] = {"cost": float(rng.uniform(0.0, 5.0))}
    if rng.random() < 0.5:
        conv["H_BFV->A64"] = {"cost": float(rng.uniform(0.0, 5.0))}
    if rng.random() < 0.5:
        conv["A64->H_BFV"] = {"cost": float(rng.uniform(0.0, 5.0))}
    cat = {"start_domain": "A64", "conversions": conv, "levels": {"H_BFV": 3}}
    if rng.random() < 0.3:
        cat["end_domain"] = "A64"
    return cat


@pytest.mark.parametrize("seed", range(60))
def test_exact_planners_match_brute_force(seed):
    rng = np.random.default_rng(seed)
    chain = random_chain(rng, int(rng.integers(1, 7)))
    cat = random_catalog(rng)
    bf_assign, bf_cost = pl.brute_force(chain, cat, None, None, None)
    ex = pl.ExactAssign()
    ex_assign, ex_cost = ex.plan(chain, cat, None, None, None)
    ilp = pl.ILPAdapter()
    ilp_assign, ilp_cost = ilp.plan(chain, cat, None, None, None)
    assert math.isclose(ex_cost, bf_cost, rel_tol=1e-12) or (ex_cost == INF and bf_cost == INF)
    assert math.isclose(ilp_cost, bf_cost, rel_tol=1e-12) or (ilp_cost == INF and bf_cost == INF)
    assert ilp.exact
    if bf_cost < INF:
        norm = pl.normalize_chain(chain, cat, None, None, None)
        for assign in (ex_assign, ilp_assign):
            choice = [next(k for k, r in enumerate(reals) if r.name == assign[op_id]) for op_id, reals in norm]
            assert math.isclose(pl.evaluate_path(norm, choice, cat), bf_cost, rel_tol=1e-12)
        g_assign, g_cost = pl.GreedyPerOperator().plan(chain, cat, None, None, None)
        assert g_cost >= bf_cost - 1e-12
    else:
        assert ex_assign == {} and ilp_assign == {}


# ---------------------------------------------------------------------------
# equal domains, unequal remaining levels
# ---------------------------------------------------------------------------
def typed_level_chain():
    return [
        {"id": "enter", "realizations": [
            {"name": "cheap_deep", "in_domain": "A64", "out_domain": "H_BFV", "cost": 1.0, "level_delta": -2},
            {"name": "careful_shallow", "in_domain": "A64", "out_domain": "H_BFV", "cost": 2.0, "level_delta": -1}]},
        {"id": "mul2", "realizations": [
            {"name": "he_depth2", "in_domain": "H_BFV", "out_domain": "H_BFV", "cost": 1.0, "level_min": 2, "level_delta": -2}]},
    ]


LEVEL_CATALOG = {"start_domain": "A64", "conversions": {}, "levels": {"H_BFV": 3}}


def domain_only_dp(chain, catalog):
    """Deliberately wrong baseline: Markov state = ending domain only (keeps the cheapest prefix per domain)."""
    norm = pl.normalize_chain(chain, catalog, None, None, None)
    frontier = {pl.start_state(catalog): (0.0, ())}
    for op_id, reals in norm:
        nxt = {}
        for st, (cost, path) in frontier.items():
            for r in reals:
                res = pl.step(st, r, catalog)
                if res is None:
                    continue
                ns, c = res
                key = ns.domain
                if key not in nxt or cost + c < nxt[key][0]:
                    nxt[key] = (cost + c, path + (r.name,), ns)
        frontier = {v[2]: (v[0], v[1]) for v in nxt.values()}
        if not frontier:
            return {}, INF
    best = min(frontier.values(), key=lambda t: t[0])
    return {op_id: name for (op_id, _), name in zip(norm, best[1])}, best[0]


def test_typed_state_domain_only_dp_misses_the_feasible_prefix():
    chain = typed_level_chain()
    wrong_assign, wrong_cost = domain_only_dp(chain, LEVEL_CATALOG)
    assert wrong_cost == INF and wrong_assign == {}
    ex_assign, ex_cost = pl.ExactAssign().plan(chain, LEVEL_CATALOG, None, None, None)
    assert ex_assign == {"enter": "careful_shallow", "mul2": "he_depth2"} and math.isclose(ex_cost, 3.0)
    bf_assign, bf_cost = pl.brute_force(chain, LEVEL_CATALOG, None, None, None)
    assert bf_assign == ex_assign and math.isclose(bf_cost, 3.0)
    ilp_assign, ilp_cost = pl.ILPAdapter().plan(chain, LEVEL_CATALOG, None, None, None)
    assert ilp_assign == ex_assign and math.isclose(ilp_cost, 3.0)
    g_assign, g_cost = pl.GreedyPerOperator().plan(chain, LEVEL_CATALOG, None, None, None)
    assert g_cost == INF                                        # greedy takes cheap_deep and gets stuck


def test_level_accounting_and_conversions():
    cat = {"start_domain": "A64", "conversions": {"A64->Zt": {"cost": 1.5}, "Zt->H_BFV": {"cost": 2.5}}, "levels": {"H_BFV": 2}}
    st = pl.start_state(cat)
    assert st.level is None and st.domain == "A64"
    r = pl.Realization("r", "Zt", "Zt", 1.0, (), None, 0, None)
    ns, c = pl.step(st, r, cat)
    assert ns.domain == "Zt" and math.isclose(c, 2.5) and ns.level is None
    r2 = pl.Realization("r2", "H_BFV", "H_BFV", 1.0, (), None, -1, None)
    ns2, c2 = pl.step(ns, r2, cat)
    assert ns2.level == 1 and math.isclose(c2, 3.5)
    assert pl.step(ns2, pl.Realization("r3", "H_BFV", "H_BFV", 1.0, (), 2, -1, None), cat) is None
    assert pl.step(ns2, pl.Realization("r4", "H_BFV", "H_BFV", 1.0, (), None, -2, None), cat) is None
    assert pl.step(ns2, pl.Realization("r5", "H_BFV", "A64", 1.0, (), None, 0, None), cat)[0].level is None
    assert pl.step(st, pl.Realization("r6", "H_CKKS", "A64", 1.0, (), None, 0, None), cat) is None


def test_install_paid_once_and_installed_objects_free():
    chain = [
        {"id": "a", "realizations": [{"name": "w", "in_domain": "A64", "out_domain": "A64", "cost": 1.0, "objects": [("W", 10.0)]},
                                     {"name": "g", "in_domain": "A64", "out_domain": "A64", "cost": 5.0}]},
        {"id": "b", "realizations": [{"name": "w", "in_domain": "A64", "out_domain": "A64", "cost": 1.0, "objects": [("W", 10.0)]},
                                     {"name": "g", "in_domain": "A64", "out_domain": "A64", "cost": 5.0}]},
        {"id": "c", "realizations": [{"name": "w", "in_domain": "A64", "out_domain": "A64", "cost": 1.0, "objects": [("W", 10.0)]},
                                     {"name": "g", "in_domain": "A64", "out_domain": "A64", "cost": 5.0}]},
    ]
    cat = {"start_domain": "A64", "conversions": {}}
    assign, cost = pl.ExactAssign().plan(chain, cat, None, None, None)
    assert assign == {"a": "w", "b": "w", "c": "w"} and math.isclose(cost, 13.0)   # one install for three uses
    st = cd.SessionState({}, {"W": {"model_version": "m", "client_key_domain": "client-default"}}, {}, 1, 1, "s", "m", "h")
    assign2, cost2 = pl.ExactAssign().plan(chain, cat, None, None, st)
    assert math.isclose(cost2, 3.0)
    stale = cd.SessionState({}, {"W": {"model_version": "old", "client_key_domain": "client-default"}}, {}, 1, 1, "s", "m", "h")
    assert math.isclose(pl.ExactAssign().plan(chain, cat, None, None, stale)[1], 13.0)
    assert math.isclose(pl.ILPAdapter().plan(chain, cat, None, None, st)[1], 3.0)


def test_shared_admission_gate_drops_non_admitted_realizations():
    chain = [{"id": "h2a", "realizations": [
        {"name": "fast", "component": "ckks_fast_h2a", "in_domain": "H_CKKS", "out_domain": "A64", "cost": 0.1},
        {"name": "unavailable", "component": "he_linear_bfv_matvec", "in_domain": "H_BFV", "out_domain": "H_BFV", "cost": 0.1},
        {"name": "toy", "component": "toy_bridge", "in_domain": "A64", "out_domain": "A64", "cost": 0.1},
        {"name": "mpc", "component": "linear_wmask", "in_domain": "A64", "out_domain": "A64", "cost": 5.0}]}]
    norm = pl.normalize_chain(chain, {"start_domain": "A64"}, None, None, None)
    assert [r.name for r in norm[0][1]] == ["mpc"]
    assign, cost = pl.ExactAssign().plan(chain, {"start_domain": "A64", "conversions": {}}, None, None, None)
    assert assign == {"h2a": "mpc"} and math.isclose(cost, 5.0)
    active = pl.normalize_chain(chain, {"start_domain": "A64"}, None, None, None, profile="Mal2P-active-v1")
    assert active[0][1] == []


def test_gpt2_linear_chain_with_real_components():
    shapes = cd.gpt2_shapes(n_layers=1, n=1, T=16)
    cat = cd.gpt2_catalog(shapes)
    model = cm.FittedCostModel(label="MODELED")
    for comp in cm.COMPONENTS:
        model.fits[comp] = cm.ComponentFit(1e-9, 0.0, 0.001, 0.0, 1, "MODELED", "toy")
    chain = pl.gpt2_linear_chain(shapes, cat)
    pcat = {"start_domain": "A64", "conversions": {}, "objects": cat["objects"]}
    env = cm.Env(100e6, 0.001)
    st = cd.SessionState({}, {}, {}, 16, 1, "s", "gpt2@v1", "h")
    bf = pl.brute_force(chain, pcat, env, model, st)
    ex = pl.ExactAssign().plan(chain, pcat, env, model, st)
    ilp = pl.ILPAdapter().plan(chain, pcat, env, model, st)
    assert math.isclose(ex[1], bf[1], rel_tol=1e-12) and math.isclose(ilp[1], bf[1], rel_tol=1e-12)
    assert ex[0] == bf[0] == ilp[0]
    # per-epoch planning without installed objects prefers the install-free generic lanes
    assert all(v == "generic" for k, v in ex[0].items() if k != "lm_head")
    all_installed = cd.SessionState({}, {o: {"model_version": "gpt2@v1", "client_key_domain": "client-default"} for o in cat["objects"]},
                                    {}, 16, 1, "s", "gpt2@v1", "h")
    ex2 = pl.ExactAssign().plan(chain, pcat, env, model, all_installed)
    assert all(v == "wmask" for v in ex2[0].values())
    assert ex2[1] < ex[1]


def test_ilp_node_budget_reports_inexact():
    rng = np.random.default_rng(3)
    chain = random_chain(rng, 6, max_real=3, n_objects=1)
    cat = {"start_domain": "A64", "conversions": {"A64->Zt": {"cost": 1.0}, "Zt->H_BFV": {"cost": 1.0},
                                                   "H_BFV->A64": {"cost": 1.0}, "A64->H_BFV": {"cost": 1.0}},
           "levels": {"H_BFV": 9}}
    ilp = pl.ILPAdapter(max_nodes=2)
    ilp.plan(chain, cat, None, None, None)
    assert not ilp.exact
    full = pl.ILPAdapter()
    full.plan(chain, cat, None, None, None)
    assert full.exact and full.nodes_visited >= 2


def test_pareto_prune_keeps_incomparable_states():
    a = pl.PlanState("H_BFV", frozenset(), 1)
    b = pl.PlanState("H_BFV", frozenset(), 2)
    c = pl.PlanState("H_BFV", frozenset({"o"}), 1)
    states = {a: (1.0, ("x",)), b: (2.0, ("y",)), c: (3.0, ("z",))}
    kept = pl.pareto_prune(states)
    assert set(kept) == {a, b, c}
    dominated = pl.PlanState("H_BFV", frozenset(), 1)
    states2 = {a: (1.0, ("x",)), pl.PlanState("H_BFV", frozenset(), 0): (1.5, ("w",))}
    assert set(pl.pareto_prune(states2)) == {a}
    assert dominated == a


def test_frontier_stays_within_the_stated_bound():
    """Complexity statement of the module: |F| <= |domains| * 2^|objects| * (L + 1) after every operator."""
    worst = 0
    for seed in range(60):
        rng = np.random.default_rng(1000 + seed)
        chain = random_chain(rng, int(rng.integers(1, 7)))
        cat = random_catalog(rng)
        ex = pl.ExactAssign()
        ex.plan(chain, cat, None, None, None)
        bound = pl.frontier_bound(pl.normalize_chain(chain, cat, None, None, None), cat)
        assert all(size <= bound for size in ex.frontier_sizes), (seed, ex.frontier_sizes, bound)
        worst = max([worst] + ex.frontier_sizes)
    assert worst >= 2                                           # the frontier really holds several states
    # three operators sharing one object over one domain, no levels: bound 1 * 2^1 * 1
    chain = [{"id": f"op{i}", "realizations": [
        {"name": "w", "in_domain": "A64", "out_domain": "A64", "cost": 1.0, "objects": [("W", 10.0)]},
        {"name": "g", "in_domain": "A64", "out_domain": "A64", "cost": 5.0}]} for i in range(3)]
    cat = {"start_domain": "A64", "conversions": {}}
    assert pl.frontier_bound(pl.normalize_chain(chain, cat, None, None, None), cat) == 2
    ex = pl.ExactAssign()
    ex.plan(chain, cat, None, None, None)
    assert max(ex.frontier_sizes) <= 2


def test_complexity_docstring_states_one_bound():
    doc = pl.__doc__
    assert doc.count("O(n * |F| * R)") == 1
    assert "|F| <= |domains| * 2^|objects| * (L + 1)" in doc
    assert "pairwise Pareto prune" in doc
    assert "O(R^n)" in doc                                      # the exhaustive planners keep their own statement

