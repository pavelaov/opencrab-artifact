"""Session state and candidate enumeration (paper, Section 4.1, session state and candidates).

The controller never sees secrets: a :class:`SessionState` holds the
incumbent plan, the installed objects with their versions, the material stock
per type, the public KV length bucket, the remaining public token budget and
the session identifiers.  :func:`enumerate_candidates` turns the catalog of
plan axes into :class:`Candidate` records, ``keep`` first, each carrying the
candidate-specific incremental install cost ``c_install(p | S_t)`` (only the
objects that are not installed at the current version), the predicted
execution cost of one epoch with its uncertainty, the material need per
epoch by type, a memory peak estimate and the feasibility flags.

Horizon schedule.  The remaining epochs do not share one public shape: the
prefill processes the whole prompt, every later decode step one row.  Given
the ``schedule`` of public shapes of the remaining epochs (entry j for epoch
t + j), every candidate also carries ``exec_s_by_epoch`` and
``exec_sigma_by_epoch``, the predicted time and standard deviation of each
remaining step, which the controller sums step by step.  ``exec_s_per_epoch``
and ``exec_sigma`` stay the values of the current epoch (entry 0); the
per-epoch baseline, the feasibility checks and the records use them.  Steps
with one shape are priced once per plan.

Plan vocabulary (``SecureGPT2.set_plan``): ``{"embedding": "wmask" |
"generic", "linear": "wmask" | "generic", "argmax_k": 2 | 4 | 8, "max_k": 2 | 4}``.
The LM head is always weight masked and installed on first use (its install
is charged in the cold-start total).

Catalog format (:func:`gpt2_catalog`): ``axes`` (axis -> admissible values in
enumeration order), ``objects`` (object id -> (d_in, d_out)), ``requires``
(axis -> value -> object ids), ``common_objects`` (installed for every plan).

Per-epoch pricing (:func:`plan_calls`, :func:`exec_cost`).  Every entry is
``(component, n, d_in, d_out, count)`` and is priced with the component
formulas of ``costmodel`` (exact gate counts of ``party.Party``).  The two
attention products of a layer are ONE ``matmul_shared`` call on the stacked
heads in ``transformer.attention`` (one opening round for the whole batch):
an entry of a component in :data:`BATCHED_COMPONENTS` with count c is priced
as c products that share one round (:func:`call_cost`).  Not priced because
they are identical for every plan: the client's one-hot ``share_input`` (one
message of 8 n V bytes, one round) and the RELEASE_FINAL message of the token
index (8 bytes, one round).  A public causal mask (prefill) removes pairwise
comparisons of the softmax tournament, so the softmax entry is an upper bound
there and exact for a decode step.

What is tested (tests/test_candidates.py): keep is always first and equals
the incumbent; a plan whose objects are installed at the current version has
c_install 0 while a version mismatch re-charges the install; install cost is
candidate specific (only missing objects) and itemized per object; material
and memory feasibility flags; the enumeration covers the full product of axes
exactly once; the batched pricing of the attention products; the per-step
costs of a schedule (entry 0 equals the current epoch, one pricing per
distinct shape, the memo returns the unmemoised values).
Design limits: the memory peak is an analytic estimate (KV cache, installed
masks, working set, staged material), not a measured RSS; the material check
compares declared stock bytes per type, it does not reserve; the batched
entry still charges the fixed compute and the framing of one call per head.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from . import costmodel as cm

PLAN_AXES: Dict[str, Tuple[Any, ...]] = {
    "embedding": ("wmask", "generic"),
    "linear": ("wmask", "generic"),
    "argmax_k": (2, 4, 8),
    "max_k": (2, 4),
}
MATERIAL_TYPES: Tuple[str, ...] = ("beaver", "matrix_triple", "wmask_mask", "drelu",
                                   "trunc_faithful", "ring_to_field", "kary_select")
# components whose plan_calls entry with count c is ONE Party call on c stacked operands (one round)
BATCHED_COMPONENTS: Tuple[str, ...] = ("matmul_shared",)


@dataclass
class SessionState:
    """Public session state S_t.  ``stock`` maps material type -> (count, bytes) AVAILABLE;
    an empty dict declares that no inventory constraint is tracked."""
    incumbent: dict
    installed: Dict[str, dict]
    stock: Dict[str, Tuple[int, int]]
    kv_bucket: int
    remaining_epochs: int
    session_id: str
    model_version: str
    contract_hash: str
    client_key_domain: str = "client-default"
    mem_limit_bytes: Optional[int] = None

    def object_version(self) -> dict:
        return {"model_version": self.model_version, "client_key_domain": self.client_key_domain}

    def is_installed(self, object_id: str) -> bool:
        """True only when the object is present with exactly the current version tuple."""
        v = self.installed.get(object_id)
        if not isinstance(v, Mapping):
            return False
        want = self.object_version()
        return all(v.get(k) == want[k] for k in want)


@dataclass
class Candidate:
    plan: dict
    install_objects: List[str]
    c_install_s: float
    c_install_sigma: float
    exec_s_per_epoch: float
    exec_sigma: float
    material_bytes_per_epoch: int
    mem_peak_bytes: int
    feasible: bool
    reason: str
    material_by_type: Dict[str, int] = field(default_factory=dict)
    is_keep: bool = False
    c_wait_s: float = 0.0                                   # c_wait(p | S_t) of the objective
    install_cost_by_object: Dict[str, float] = field(default_factory=dict)   # itemized c_install_s
    exec_s_by_epoch: List[float] = field(default_factory=list)       # E[T_exec(p, t + j)] per remaining step j
    exec_sigma_by_epoch: List[float] = field(default_factory=list)   # its standard deviation; both empty = one shape

    def to_dict(self) -> dict:
        return {"plan": dict(self.plan), "install_objects": list(self.install_objects),
                "c_install_s": self.c_install_s, "c_install_sigma": self.c_install_sigma,
                "exec_s_per_epoch": self.exec_s_per_epoch, "exec_sigma": self.exec_sigma,
                "material_bytes_per_epoch": self.material_bytes_per_epoch,
                "mem_peak_bytes": self.mem_peak_bytes, "feasible": self.feasible,
                "reason": self.reason, "is_keep": self.is_keep, "c_wait_s": self.c_wait_s,
                "install_cost_by_object": dict(self.install_cost_by_object),
                "exec_s_by_epoch": list(self.exec_s_by_epoch),
                "exec_sigma_by_epoch": list(self.exec_sigma_by_epoch)}


# ---------------------------------------------------------------------------
# shapes and catalog
# ---------------------------------------------------------------------------
def gpt2_shapes(n_layers: int = 12, n: int = 1, T: int = 32, d_model: int = 768, d_ff: int = 3072,
                n_head: int = 12, vocab: int = 50257, mem_limit_bytes: Optional[int] = None) -> dict:
    """Public shapes of one epoch: n rows processed, T = KV length bucket."""
    return {"n_layers": int(n_layers), "n": int(n), "T": int(T), "d_model": int(d_model), "d_ff": int(d_ff),
            "n_head": int(n_head), "vocab": int(vocab), "mem_limit_bytes": mem_limit_bytes}


def shape_key(shapes: Mapping[str, Any]) -> Tuple[int, ...]:
    """The public shape of one epoch as a hashable key (everything :func:`plan_calls` reads)."""
    return tuple(int(shapes[k]) for k in ("n_layers", "n", "T", "d_model", "d_ff", "n_head", "vocab"))


def gpt2_catalog(shapes: Mapping[str, Any]) -> dict:
    """Plan axes, installable objects with shapes, and which axis value requires which objects."""
    d, f, V, L = int(shapes["d_model"]), int(shapes["d_ff"]), int(shapes["vocab"]), int(shapes["n_layers"])
    objects: Dict[str, Tuple[int, int]] = {"wte": (V, d), "lm_head": (d, V)}
    layer_mats: List[str] = []
    for li in range(L):
        for name, shp in (("qkv", (d, 3 * d)), ("out", (d, d)), ("fc", (d, f)), ("proj", (f, d))):
            oid = f"h{li}.{name}"
            objects[oid] = shp
            layer_mats.append(oid)
    return {
        "axes": {k: tuple(v) for k, v in PLAN_AXES.items()},
        "objects": objects,
        "requires": {"embedding": {"wmask": ["wte"], "generic": []},
                     "linear": {"wmask": layer_mats, "generic": []},
                     "argmax_k": {2: [], 4: [], 8: []},
                     "max_k": {2: [], 4: []}},
        "common_objects": ["lm_head"],
    }


def normalize_plan(plan: Mapping[str, Any], catalog: Mapping[str, Any]) -> dict:
    """Validate a plan against the catalog axes and return a canonical dict."""
    out: dict = {}
    for axis, values in catalog["axes"].items():
        if axis not in plan:
            raise ValueError(f"PLAN_AXIS_MISSING: plan lacks axis {axis!r}")
        v = plan[axis]
        if v not in values:
            raise ValueError(f"PLAN_VALUE_INVALID: axis {axis!r} value {v!r} not in {list(values)}")
        out[axis] = v
    return out


def plan_key(plan: Mapping[str, Any]) -> str:
    return "|".join(f"{k}={plan[k]}" for k in sorted(plan))


def plan_objects(plan: Mapping[str, Any], catalog: Mapping[str, Any]) -> List[str]:
    """Installed objects a plan needs (common objects first, then per axis value)."""
    objs: List[str] = list(catalog.get("common_objects", []))
    for axis, value in plan.items():
        objs.extend(catalog["requires"].get(axis, {}).get(value, []))
    seen: set = set()
    ordered: List[str] = []
    for o in objs:
        if o not in seen:
            seen.add(o)
            ordered.append(o)
    return ordered


def missing_objects(plan: Mapping[str, Any], state: SessionState, catalog: Mapping[str, Any]) -> List[str]:
    """Objects the plan needs that are not installed at the current (model_version, client_key_domain)."""
    return [o for o in plan_objects(plan, catalog) if not state.is_installed(o)]


# ---------------------------------------------------------------------------
# per-epoch component calls of a plan
# ---------------------------------------------------------------------------
def plan_calls(plan: Mapping[str, Any], shapes: Mapping[str, Any]) -> List[Tuple[str, int, int, int, int]]:
    """(component, n, d_in, d_out, count) of every gadget call in one epoch under ``plan``."""
    n, T = int(shapes["n"]), int(shapes["T"])
    d, f, V, L, H = (int(shapes["d_model"]), int(shapes["d_ff"]), int(shapes["vocab"]),
                     int(shapes["n_layers"]), int(shapes["n_head"]))
    dh = d // H
    lin = "linear_wmask" if plan["linear"] == "wmask" else "linear_generic"
    emb = "embedding_wmask" if plan["embedding"] == "wmask" else "embedding_generic"
    calls: List[Tuple[str, int, int, int, int]] = [(emb, n, V, d, 1)]
    for _ in range(L):
        calls.append(("layernorm_rows", n, d, d, 1))
        calls.append((lin, n, d, 3 * d, 1))
        calls.append(("matmul_shared", n, dh, T, H))                 # Q K^T per head
        calls.append((f"softmax_rows_k{int(plan['max_k'])}", H * n, T, T, 1))
        calls.append(("matmul_shared", n, T, dh, H))                 # P V per head
        calls.append((lin, n, d, d, 1))
        calls.append(("layernorm_rows", n, d, d, 1))
        calls.append((lin, n, d, f, 1))
        calls.append(("gelu_clip12", n, f, f, 1))
        calls.append((lin, n, f, d, 1))
    calls.append(("layernorm_rows", 1, d, d, 1))
    calls.append(("linear_wmask", 1, d, V, 1))                       # LM head on the last row
    calls.append((f"argmax_k{int(plan['argmax_k'])}", 1, V, 1, 1))
    return calls


def call_cost(comp: str, n: int, d_in: int, d_out: int, count: int, costmodel: Any, env: Any) -> Tuple[float, float]:
    """(mean_s, variance) of one :func:`plan_calls` entry.

    ``count`` independent calls cost ``count`` times the prediction.  For a
    component of :data:`BATCHED_COMPONENTS` the ``count`` products are one
    Party call on stacked operands: same payload, material and compute, but
    one opening round instead of ``count``, so ``(count - 1) * rounds * RTT``
    is taken back from the sum (``costmodel.predict`` charges ``rounds * RTT``
    per call under the same environment).
    """
    mu, sg = costmodel.predict(comp, n, d_in, d_out, env)
    mean = count * mu
    if count > 1 and comp in BATCHED_COMPONENTS:
        mean -= (count - 1) * cm.rounds_formula(comp, n, d_in, d_out) * cm.as_env(env).rtt_s
    return float(mean), float(count * sg * sg)


def exec_cost(plan: Mapping[str, Any], shapes: Mapping[str, Any], costmodel: Any, env: Any,
              memo: Optional[dict] = None) -> Tuple[float, float]:
    """(mean_s, sigma_s) of one epoch; component sigmas are combined as independent.

    ``memo`` is an optional dict owned by the caller.  It keeps the result per
    (plan, shape, environment) and the cost of every :func:`plan_calls` entry
    per environment, so that plans sharing an entry and steps sharing a shape
    are priced once.  One memo serves one cost model; the values are the ones
    computed without it.
    """
    if memo is None:
        mean = var = 0.0
        for comp, n, d_in, d_out, count in plan_calls(plan, shapes):
            mu, v = call_cost(comp, n, d_in, d_out, count, costmodel, env)
            mean += mu
            var += v
        return float(mean), float(math.sqrt(var))
    e = cm.as_env(env)
    ek = (e.bandwidth_bps, e.one_way_latency_s, e.cpu_share)
    key = ("exec", plan_key(plan), shape_key(shapes), ek)
    hit = memo.get(key)
    if hit is None:
        mean = var = 0.0
        for call in plan_calls(plan, shapes):
            ck = call + ek
            cv = memo.get(ck)
            if cv is None:
                cv = memo[ck] = call_cost(*call, costmodel, env)
            mean += cv[0]
            var += cv[1]
        hit = memo[key] = (float(mean), float(math.sqrt(var)))
    return hit


def plan_bytes_rounds(plan: Mapping[str, Any], shapes: Mapping[str, Any]) -> Tuple[int, int]:
    """(online bytes, rounds) of one epoch from the cost-model formulas, batched entries sharing one round."""
    nbytes = rounds = 0
    for comp, n, d_in, d_out, count in plan_calls(plan, shapes):
        b = cm.bytes_formula(comp, n, d_in, d_out)
        r = cm.rounds_formula(comp, n, d_in, d_out)
        if count > 1 and comp in BATCHED_COMPONENTS:
            frames = cm.FRAME_BYTES * cm.messages_formula(comp, n, d_in, d_out)
            nbytes += count * (b - frames) + frames
            rounds += r
        else:
            nbytes += count * b
            rounds += count * r
    return int(nbytes), int(rounds)


def install_costs(objects: Sequence[str], catalog: Mapping[str, Any], costmodel: Any,
                  env: Any) -> Dict[str, Tuple[float, float]]:
    """``{object id: (mean_s, sigma_s)}`` of installing each listed weight-mask object (install_wmask per matrix)."""
    out: Dict[str, Tuple[float, float]] = {}
    for o in objects:
        d_in, d_out = catalog["objects"][o]
        mu, sg = costmodel.predict("install_wmask", 1, d_in, d_out, env)
        out[str(o)] = (float(mu), float(sg))
    return out


def install_cost(objects: Sequence[str], catalog: Mapping[str, Any], costmodel: Any, env: Any) -> Tuple[float, float]:
    """(mean_s, sigma_s) of installing the listed weight-mask objects (install_wmask per matrix)."""
    mean = var = 0.0
    for mu, sg in install_costs(objects, catalog, costmodel, env).values():
        mean += mu
        var += sg * sg
    return float(mean), float(math.sqrt(var))


def material_per_epoch(plan: Mapping[str, Any], shapes: Mapping[str, Any], contract: Any = None) -> Dict[str, int]:
    """Per-party material bytes of one epoch by material type."""
    out: Dict[str, int] = {}
    for comp, n, d_in, d_out, count in plan_calls(plan, shapes):
        for t, b in cm.material_breakdown(comp, n, d_in, d_out, contract).items():
            out[t] = out.get(t, 0) + count * b
    return out


def mem_peak(plan: Mapping[str, Any], shapes: Mapping[str, Any], catalog: Mapping[str, Any],
             state: SessionState, material_bytes: int) -> int:
    """Analytic per-party peak: KV cache shares, installed masks (P0 holds E and B0), working set, staged material."""
    n, T = int(shapes["n"]), int(shapes["T"])
    d, f, L = int(shapes["d_model"]), int(shapes["d_ff"]), int(shapes["n_layers"])
    kv = 2 * L * max(T, int(state.kv_bucket)) * d * 8
    objs = 0
    for o in set(plan_objects(plan, catalog)) | set(state.installed):
        if o in catalog["objects"]:
            d_in, d_out = catalog["objects"][o]
            objs += 16 * d_in * d_out
    working = 8 * n * (4 * f + 3 * d) * 4
    return int(kv + objs + working + material_bytes)


# ---------------------------------------------------------------------------
# enumeration
# ---------------------------------------------------------------------------
def enumerate_candidates(state: SessionState, catalog: Mapping[str, Any], costmodel: Any, env: Any,
                         shapes: Mapping[str, Any], contract: Any = None,
                         admit_fn: Optional[Callable[[dict], Tuple[bool, str]]] = None,
                         schedule: Optional[Sequence[Mapping[str, Any]]] = None,
                         memo: Optional[dict] = None) -> List[Candidate]:
    """All plans of the catalog as candidates, ``keep`` (the incumbent) first.

    ``admit_fn(plan) -> (ok, code)`` is an optional shared admission gate
    (registry.admit over the plan's tape); a rejected plan is returned with
    ``feasible=False`` and the code as reason so that the decision record can
    list rejected candidates.  Keep is returned even when infeasible.

    ``schedule`` lists the public shapes of the remaining epochs, entry j for
    epoch t + j; entry 0 must be ``shapes`` (``SCHEDULE_FIRST_STEP``
    otherwise).  Every candidate then carries ``exec_s_by_epoch`` and
    ``exec_sigma_by_epoch`` with one entry per scheduled step, priced under
    ``env``; a shape that occurs in several steps is priced once per plan.
    Without a schedule both lists stay empty and the controller prices the
    whole horizon on ``shapes``.  Material, memory and admission are checked
    for the current epoch only.

    ``memo`` (see :func:`exec_cost`) may be shared by a caller that enumerates
    repeatedly under one cost model and one contract; it also keeps the
    material need per (plan, shape).
    """
    incumbent = normalize_plan(state.incumbent, catalog)
    axes = list(catalog["axes"].items())
    plans: List[dict] = [incumbent]
    for combo in itertools.product(*[vals for _, vals in axes]):
        p = {axis: v for (axis, _), v in zip(axes, combo)}
        if p != incumbent:
            plans.append(p)
    key0 = shape_key(shapes)
    steps = [(shape_key(sh), sh) for sh in schedule] if schedule is not None else []
    if steps and steps[0][0] != key0:
        raise ValueError(f"SCHEDULE_FIRST_STEP: the schedule starts with shape {steps[0][0]}, the current epoch has {key0}")
    out: List[Candidate] = []
    for i, p in enumerate(plans):
        need = missing_objects(p, state, catalog)
        per_obj = install_costs(need, catalog, costmodel, env)
        c_inst = s_var = 0.0
        for mu_o, sg_o in per_obj.values():
            c_inst += mu_o
            s_var += sg_o * sg_o
        s_inst = math.sqrt(s_var)
        ex, sx = exec_cost(p, shapes, costmodel, env, memo)
        priced = {key0: (ex, sx)}
        by_s: List[float] = []
        by_g: List[float] = []
        for sk, sh in steps:
            v = priced.get(sk)
            if v is None:
                v = priced[sk] = exec_cost(p, sh, costmodel, env, memo)
            by_s.append(v[0])
            by_g.append(v[1])
        if memo is None:
            mat = material_per_epoch(p, shapes, contract)
        else:
            mk = ("material", plan_key(p), key0)
            if mk not in memo:
                memo[mk] = material_per_epoch(p, shapes, contract)
            mat = dict(memo[mk])
        mat_total = int(sum(mat.values()))
        peak = mem_peak(p, shapes, catalog, state, mat_total)
        feasible, reason = True, "OK"
        if state.stock:
            for t, b in mat.items():
                have = int(state.stock.get(t, (0, 0))[1])
                if have < b:
                    feasible, reason = False, f"INSUFFICIENT_MATERIAL: {t} needs {b} B, stock {have} B"
                    break
        limit = state.mem_limit_bytes if state.mem_limit_bytes is not None else shapes.get("mem_limit_bytes")
        if feasible and limit is not None and peak > int(limit):
            feasible, reason = False, f"MEMORY_INFEASIBLE: peak {peak} B > limit {int(limit)} B"
        if feasible and admit_fn is not None:
            ok, code = admit_fn(p)
            if not ok:
                feasible, reason = False, str(code)
        out.append(Candidate(plan=p, install_objects=need, c_install_s=c_inst, c_install_sigma=s_inst,
                             exec_s_per_epoch=ex, exec_sigma=sx, material_bytes_per_epoch=mat_total,
                             mem_peak_bytes=peak, feasible=feasible, reason=reason, material_by_type=mat,
                             is_keep=(i == 0), install_cost_by_object={o: v[0] for o, v in per_obj.items()},
                             exec_s_by_epoch=by_s, exec_sigma_by_epoch=by_g))
    return out


__all__ = [
    "PLAN_AXES", "MATERIAL_TYPES", "SessionState", "Candidate", "gpt2_shapes", "shape_key", "gpt2_catalog",
    "normalize_plan", "plan_key", "plan_objects", "missing_objects", "plan_calls", "call_cost", "exec_cost",
    "plan_bytes_rounds", "install_costs", "install_cost", "material_per_epoch", "mem_peak", "enumerate_candidates",
    "BATCHED_COMPONENTS",
]
