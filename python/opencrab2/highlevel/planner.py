"""Assignment planners over a serialized operator chain (paper, Section 4.4, planner interface).

A chain is a list of operators ``{"id": str, "realizations": [r, ...]}``.  A
realization ``r`` has ``name``, ``in_domain``, ``out_domain`` and either a
precomputed ``cost`` (seconds) or ``component`` + ``shape`` (priced through
``costmodel.predict``), optional ``objects`` (installed objects it needs:
``{"id", "install_cost"}`` or ``{"id", "shape"}`` priced as install_wmask,
or a bare id looked up in ``catalog["objects"]``), and optional level
constraints ``level_min`` / ``level_delta`` for HE domains.  The catalog
declares ``start_domain`` (default A64), optional ``end_domain``,
``conversions`` (``"A64->Zt": {"cost": float}``) and ``levels`` (level
budget on entering an HE domain, e.g. ``{"H_BFV": 3}``).

Markov state of a path prefix: ``(ending domain, frozenset of installed
objects, level tag)``.  Recording only the domain is insufficient: two
prefixes in the same domain with different remaining levels are different
states, and the cheaper one may leave no feasible continuation.

Planners (all share the registry admission gate for realizations that name
a component):

* :class:`ExactAssign`: Pareto dynamic program over the explicit state.  A
  frontier state is pruned only when another state of the same domain has a
  superset of installed objects, at least the same level and no higher cost.
  Complexity (the one bound this module states): O(n * |F| * R) transitions
  for n operators with at most R realizations each, where the frontier F of
  one operator satisfies |F| <= |domains| * 2^|objects| * (L + 1) (L the
  largest level budget of the catalog), plus the pairwise Pareto prune, which
  performs at most |F|^2 dominance tests per operator, each a comparison of
  two object sets (O(n * |F|^2 * |objects|) over the chain).  The frontier is
  exponential in the number of installable objects in the worst case; the
  projection onto the objects a later operator still references and the prune
  keep it far smaller on the chains of this tree (``frontier_sizes`` records
  it).  Exact for a serialized chain, a fixed catalog and no global resource
  constraints beyond the state.
* :class:`ILPAdapter`: depth-first branch and bound over the same catalog
  (lower bound = cost so far + sum of the cheapest realization of every
  remaining operator, valid because conversion and install costs are
  non-negative).  A faithful re-implementation of the assignment formulation
  used by ILP planners (Silph/NetAdapt style), not a rerun of those systems.
  Exact; worst case O(R^n).
* :class:`GreedyPerOperator`: cheapest step from the current state, no
  lookahead.  Heuristic.
* :func:`brute_force`: enumerates every assignment (O(R^n)); the reference.

What is tested (tests/test_planner.py): ExactAssign and ILPAdapter equal
brute_force on random small chains (n <= 6, <= 3 realizations each,
conversions, shared install objects, levels); the frontier of every operator
stays within the stated bound; a chain with equal domains and unequal
remaining levels, where a domain-only DP (provided as a deliberately wrong
baseline in the test)
misses the only feasible prefix; the admission gate drops non-admitted
realizations; GreedyPerOperator is never cheaper than ExactAssign.
Design limit: candidate pruning before the chain is built (which plans
reach the planner) is an engineering approximation outside this file.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import registry as _registry

INF = float("inf")


@dataclass(frozen=True)
class PlanState:
    domain: str
    installed: FrozenSet[str]
    level: Optional[int]


@dataclass(frozen=True)
class Realization:
    name: str
    in_domain: str
    out_domain: str
    cost: float
    objects: Tuple[Tuple[str, float], ...]
    level_min: Optional[int]
    level_delta: int
    component: Optional[str]


def _object_cost(o: Any, catalog: Mapping[str, Any], costmodel: Any, env: Any) -> Tuple[str, float]:
    if isinstance(o, str):
        oid = o
        inst = dict(catalog.get("install_costs", {}))
        if oid in inst:
            return oid, float(inst[oid])
        shape = dict(catalog.get("objects", {})).get(oid)
        if shape is None or costmodel is None:
            raise ValueError(f"PLANNER_OBJECT: object {oid!r} has no install cost or shape in the catalog")
        return oid, float(costmodel.predict("install_wmask", 1, int(shape[0]), int(shape[1]), env)[0])
    if isinstance(o, (tuple, list)) and len(o) == 2:
        return str(o[0]), float(o[1])
    if isinstance(o, Mapping):
        oid = str(o["id"])
        if "install_cost" in o:
            return oid, float(o["install_cost"])
        if "shape" in o and costmodel is not None:
            s = o["shape"]
            return oid, float(costmodel.predict("install_wmask", 1, int(s[0]), int(s[1]), env)[0])
        return _object_cost(oid, catalog, costmodel, env)
    raise ValueError(f"PLANNER_OBJECT: cannot interpret object {o!r}")


def normalize_chain(chain: Sequence[Mapping[str, Any]], catalog: Mapping[str, Any], env: Any, costmodel: Any,
                    state: Any, profile: str = _registry.PROFILE_ID) -> List[Tuple[str, List[Realization]]]:
    """Price every realization and drop the ones the shared admission gate rejects."""
    installed_now: set = set()
    if state is not None and getattr(state, "installed", None):
        for oid in state.installed:
            if hasattr(state, "is_installed") and not state.is_installed(oid):
                continue
            installed_now.add(str(oid))
    out: List[Tuple[str, List[Realization]]] = []
    for i, op in enumerate(chain):
        op_id = str(op.get("id", f"op{i}"))
        reals: List[Realization] = []
        for r in op["realizations"]:
            comp = r.get("component")
            if comp is not None:
                ok, _ = _registry.component_admissible(str(comp), profile)
                if not ok:
                    continue
            if "cost" in r:
                cost = float(r["cost"])
            elif comp is not None and costmodel is not None:
                n, d_in, d_out = (int(v) for v in r["shape"])
                cost = float(costmodel.predict(str(comp), n, d_in, d_out, env)[0]) * float(r.get("count", 1))
            else:
                raise ValueError(f"PLANNER_COST: realization {r.get('name')!r} of {op_id!r} has neither cost nor component")
            objs = []
            for o in r.get("objects", []) or []:
                oid, c = _object_cost(o, catalog, costmodel, env)
                if oid not in installed_now:
                    objs.append((oid, c))
            reals.append(Realization(str(r["name"]), str(r["in_domain"]), str(r["out_domain"]), cost,
                                     tuple(objs), r.get("level_min"), int(r.get("level_delta", 0)),
                                     str(comp) if comp is not None else None))
        out.append((op_id, reals))
    return out


def start_state(catalog: Mapping[str, Any]) -> PlanState:
    dom = str(catalog.get("start_domain", "A64"))
    levels = dict(catalog.get("levels", {}))
    return PlanState(dom, frozenset(), int(levels[dom]) if dom in levels else None)


def _conversion(have: str, need: str, catalog: Mapping[str, Any]) -> float:
    if have == need:
        return 0.0
    conv = dict(catalog.get("conversions", {})).get(f"{have}->{need}")
    if conv is None:
        return INF
    if isinstance(conv, Mapping):
        return float(conv.get("cost", 0.0))
    return float(conv)


def step(st: PlanState, r: Realization, catalog: Mapping[str, Any]) -> Optional[Tuple[PlanState, float]]:
    """Apply one realization to a state: (new state, incremental cost) or None when infeasible."""
    levels = dict(catalog.get("levels", {}))
    conv = _conversion(st.domain, r.in_domain, catalog)
    if conv == INF:
        return None
    lvl = st.level
    if r.in_domain != st.domain:
        lvl = int(levels[r.in_domain]) if r.in_domain in levels else None
    if r.level_min is not None and (lvl is None or lvl < int(r.level_min)):
        return None
    if r.out_domain in levels:
        if lvl is None or r.out_domain != r.in_domain:
            lvl = int(levels[r.out_domain])
        lvl = lvl + r.level_delta
        if lvl < 0:
            return None
    else:
        lvl = None
    install = sum(c for oid, c in r.objects if oid not in st.installed)
    new_inst = st.installed | frozenset(oid for oid, _ in r.objects)
    return PlanState(r.out_domain, new_inst, lvl), conv + r.cost + install


def _end_ok(st: PlanState, catalog: Mapping[str, Any]) -> bool:
    end = catalog.get("end_domain")
    return end is None or st.domain == str(end)


def evaluate_path(norm: Sequence[Tuple[str, List[Realization]]], choice: Sequence[int],
                  catalog: Mapping[str, Any]) -> float:
    st = start_state(catalog)
    total = 0.0
    for (op_id, reals), ci in zip(norm, choice):
        res = step(st, reals[ci], catalog)
        if res is None:
            return INF
        st, c = res
        total += c
    return total if _end_ok(st, catalog) else INF


def _dominates(a: Tuple[PlanState, float], b: Tuple[PlanState, float]) -> bool:
    """True when state a is at least as good as b in every future-relevant respect and not identical."""
    sa, ca = a
    sb, cb = b
    if sa.domain != sb.domain or ca > cb:
        return False
    if not sa.installed >= sb.installed:
        return False
    if (sa.level is None) != (sb.level is None):
        return False
    if sa.level is not None and sa.level < sb.level:
        return False
    return ca < cb or sa.installed > sb.installed or (sa.level is not None and sa.level > sb.level)


def future_objects(norm: Sequence[Tuple[str, List[Realization]]]) -> List[FrozenSet[str]]:
    """future[i] = objects referenced by any realization of an operator after position i."""
    n = len(norm)
    out: List[FrozenSet[str]] = [frozenset()] * n
    acc: FrozenSet[str] = frozenset()
    for i in range(n - 1, -1, -1):
        out[i] = acc
        for r in norm[i][1]:
            acc = acc | frozenset(oid for oid, _ in r.objects)
    return out


def object_use_counts(norm: Sequence[Tuple[str, List[Realization]]]) -> Dict[str, int]:
    """Number of operators whose realizations reference each object."""
    counts: Dict[str, int] = {}
    for _, reals in norm:
        seen = set()
        for r in reals:
            for oid, _ in r.objects:
                seen.add(oid)
        for oid in seen:
            counts[oid] = counts.get(oid, 0) + 1
    return counts


def pareto_prune(states: Mapping[PlanState, Tuple[float, Tuple[str, ...]]]
                 ) -> Dict[PlanState, Tuple[float, Tuple[str, ...]]]:
    """Drop every state dominated by another state of the same domain (exact pruning)."""
    by_dom: Dict[str, List[Tuple[PlanState, float]]] = {}
    for s, (c, _) in states.items():
        by_dom.setdefault(s.domain, []).append((s, c))
    keep: Dict[PlanState, Tuple[float, Tuple[str, ...]]] = {}
    for dom, items in by_dom.items():
        items.sort(key=lambda t: t[1])
        for s1, c1 in items:
            dominated = False
            for s2, c2 in items:
                if c2 > c1:
                    break
                if s2 is not s1 and _dominates((s2, c2), (s1, c1)):
                    dominated = True
                    break
            if not dominated:
                keep[s1] = states[s1]
    return keep


def frontier_bound(norm: Sequence[Tuple[str, List[Realization]]], catalog: Mapping[str, Any]) -> int:
    """The stated bound |domains| * 2^|objects| * (L + 1) on the frontier of one operator.

    ``domains`` are the start domain and every output domain of the chain,
    ``objects`` the install objects any realization references, L the largest
    level budget of the catalog (0 when no domain carries levels).
    """
    domains = {str(catalog.get("start_domain", "A64"))}
    objects: set = set()
    for _, reals in norm:
        for r in reals:
            domains.add(r.out_domain)
            objects.update(oid for oid, _ in r.objects)
    levels = [int(v) for v in dict(catalog.get("levels", {})).values()]
    L = max(levels) if levels else 0
    return len(domains) * (1 << len(objects)) * (L + 1)


class ExactAssign:
    """Pareto dynamic program over (domain, installed objects, level); exact for a serialized chain.

    ``frontier_sizes`` holds the frontier size after every operator of the
    last ``plan`` call; each is at most :func:`frontier_bound`.
    """

    name = "ExactAssign"

    def __init__(self, profile: str = _registry.PROFILE_ID):
        self.profile = profile
        self.frontier_sizes: List[int] = []

    def plan(self, chain: Sequence[Mapping[str, Any]], catalog: Mapping[str, Any], env: Any, costmodel: Any,
             state: Any) -> Tuple[dict, float]:
        norm = normalize_chain(chain, catalog, env, costmodel, state, self.profile)
        future = future_objects(norm)
        frontier: Dict[PlanState, Tuple[float, Tuple[str, ...]]] = {start_state(catalog): (0.0, ())}
        self.frontier_sizes = []
        for i, (op_id, reals) in enumerate(norm):
            nxt: Dict[PlanState, Tuple[float, Tuple[str, ...]]] = {}
            for st, (cost, path) in frontier.items():
                for r in reals:
                    res = step(st, r, catalog)
                    if res is None:
                        continue
                    ns, c = res
                    # project the installed set onto the objects a later operator can still use:
                    # an object no later operator references cannot change any future cost
                    ns = PlanState(ns.domain, ns.installed & future[i], ns.level)
                    tot = cost + c
                    if ns not in nxt or tot < nxt[ns][0]:
                        nxt[ns] = (tot, path + (r.name,))
            frontier = pareto_prune(nxt)
            self.frontier_sizes.append(len(frontier))
            if not frontier:
                return {}, INF
        best = None
        for st, (cost, path) in frontier.items():
            if _end_ok(st, catalog) and (best is None or cost < best[0]):
                best = (cost, path)
        if best is None:
            return {}, INF
        return {op_id: name for (op_id, _), name in zip(norm, best[1])}, float(best[0])


class ILPAdapter:
    """Branch and bound over the same chain and catalog (exact assignment, ILP-style formulation)."""

    name = "ILPAdapter"

    def __init__(self, profile: str = _registry.PROFILE_ID, max_nodes: Optional[int] = None):
        self.profile = profile
        self.max_nodes = max_nodes
        self.nodes_visited = 0
        self.exact = True

    def plan(self, chain: Sequence[Mapping[str, Any]], catalog: Mapping[str, Any], env: Any, costmodel: Any,
             state: Any) -> Tuple[dict, float]:
        norm = normalize_chain(chain, catalog, env, costmodel, state, self.profile)
        n = len(norm)
        # lower bound per remaining operator: cheapest realization cost plus the install cost of the
        # objects only that operator references (an install shared by several operators is left out,
        # conversions are left out; both are non-negative, so the bound stays valid)
        uses = object_use_counts(norm)

        def lb(r: Realization) -> float:
            return r.cost + sum(c for oid, c in r.objects if uses.get(oid, 0) == 1)

        suffix_min = [0.0] * (n + 1)
        for i in range(n - 1, -1, -1):
            reals = norm[i][1]
            suffix_min[i] = suffix_min[i + 1] + (min(lb(r) for r in reals) if reals else INF)
        best_cost = INF
        best_path: Tuple[int, ...] = ()
        self.nodes_visited = 0
        self.exact = True
        greedy_assign, greedy_cost = GreedyPerOperator(self.profile).plan(chain, catalog, env, costmodel, state)
        if greedy_cost < INF:
            best_cost = greedy_cost
            best_path = tuple(next(i for i, r in enumerate(reals) if r.name == greedy_assign[op_id])
                              for op_id, reals in norm)

        def dfs(i: int, st: PlanState, acc: float, path: Tuple[int, ...]) -> None:
            nonlocal best_cost, best_path
            self.nodes_visited += 1
            if self.max_nodes is not None and self.nodes_visited > self.max_nodes:
                self.exact = False
                return
            if acc + suffix_min[i] >= best_cost:
                return
            if i == n:
                if _end_ok(st, catalog) and acc < best_cost:
                    best_cost, best_path = acc, path
                return
            reals = norm[i][1]
            order = sorted(range(len(reals)), key=lambda k: reals[k].cost)
            for k in order:
                res = step(st, reals[k], catalog)
                if res is None:
                    continue
                ns, c = res
                dfs(i + 1, ns, acc + c, path + (k,))

        dfs(0, start_state(catalog), 0.0, ())
        if best_cost == INF:
            return {}, INF
        return {op_id: reals[k].name for (op_id, reals), k in zip(norm, best_path)}, float(best_cost)


class GreedyPerOperator:
    """Cheapest incremental step per operator from the current state (heuristic, no lookahead)."""

    name = "GreedyPerOperator"

    def __init__(self, profile: str = _registry.PROFILE_ID):
        self.profile = profile

    def plan(self, chain: Sequence[Mapping[str, Any]], catalog: Mapping[str, Any], env: Any, costmodel: Any,
             state: Any) -> Tuple[dict, float]:
        norm = normalize_chain(chain, catalog, env, costmodel, state, self.profile)
        st = start_state(catalog)
        total = 0.0
        assign: dict = {}
        for op_id, reals in norm:
            best = None
            for r in reals:
                res = step(st, r, catalog)
                if res is None:
                    continue
                if best is None or res[1] < best[1]:
                    best = (res[0], res[1], r.name)
            if best is None:
                return {}, INF
            st, c, name = best
            total += c
            assign[op_id] = name
        if not _end_ok(st, catalog):
            return {}, INF
        return assign, float(total)


def brute_force(chain: Sequence[Mapping[str, Any]], catalog: Mapping[str, Any], env: Any, costmodel: Any,
                state: Any, profile: str = _registry.PROFILE_ID) -> Tuple[dict, float]:
    """Enumerate every assignment; the reference the exact planners are tested against."""
    norm = normalize_chain(chain, catalog, env, costmodel, state, profile)
    if any(len(reals) == 0 for _, reals in norm):
        return {}, INF
    best_cost = INF
    best_choice: Optional[Tuple[int, ...]] = None
    for choice in itertools.product(*[range(len(reals)) for _, reals in norm]):
        c = evaluate_path(norm, choice, catalog)
        if c < best_cost:
            best_cost, best_choice = c, choice
    if best_choice is None:
        return {}, INF
    return {op_id: reals[k].name for (op_id, reals), k in zip(norm, best_choice)}, float(best_cost)


def gpt2_linear_chain(shapes: Mapping[str, Any], catalog: Mapping[str, Any]) -> List[dict]:
    """The linear sites of one GPT-2 epoch as a chain with wmask / generic realizations (real components)."""
    n, d, f, V, L = (int(shapes["n"]), int(shapes["d_model"]), int(shapes["d_ff"]), int(shapes["vocab"]),
                     int(shapes["n_layers"]))
    chain: List[dict] = [{"id": "embedding", "realizations": [
        {"name": "wmask", "component": "embedding_wmask", "in_domain": "A64", "out_domain": "A64",
         "shape": (n, V, d), "objects": ["wte"]},
        {"name": "generic", "component": "embedding_generic", "in_domain": "A64", "out_domain": "A64",
         "shape": (n, V, d)}]}]
    for li in range(L):
        for site, shape in (("qkv", (n, d, 3 * d)), ("out", (n, d, d)), ("fc", (n, d, f)), ("proj", (n, f, d))):
            oid = f"h{li}.{site}"
            chain.append({"id": oid, "realizations": [
                {"name": "wmask", "component": "linear_wmask", "in_domain": "A64", "out_domain": "A64",
                 "shape": shape, "objects": [oid]},
                {"name": "generic", "component": "linear_generic", "in_domain": "A64", "out_domain": "A64",
                 "shape": shape}]})
    chain.append({"id": "lm_head", "realizations": [
        {"name": "wmask", "component": "linear_wmask", "in_domain": "A64", "out_domain": "A64",
         "shape": (1, d, V), "objects": ["lm_head"]}]})
    return chain


__all__ = ["INF", "PlanState", "Realization", "normalize_chain", "start_state", "step", "evaluate_path",
           "future_objects", "object_use_counts", "pareto_prune", "frontier_bound",
           "ExactAssign", "ILPAdapter", "GreedyPerOperator", "brute_force", "gpt2_linear_chain"]
