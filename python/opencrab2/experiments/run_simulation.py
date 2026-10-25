"""MODELED session campaign: every policy on every trace with the fitted cost model as environment.

What this is. A closed-loop simulation of the session epoch loop (paper,
Section 4) in which the environment is the calibrated cost model: the execution
time of an epoch under plan p is the model mean under the TRUE trace state
(integrated over the trace segments the epoch spans) times a log-normal
noise factor with the fitted per-component sigma, and installs are priced by
the model the same way. Policies see only public telemetry (the trace state
at the probe instant, one epoch late relative to the truth inside the epoch)
and the same calibrated model. Every record carries ``status = MODELED``; a
MODELED number is never a measurement and the paper prints it in red.

Workload: GPT-2 with 12 layers, a 16-token prompt and T in {8, 32} greedy
decode steps; epoch 0 is the prefill (n = 16 rows), epoch j >= 1 one decode
step with KV length 16 + j (or the public length bucket of an S6 trace).
Every decision prices the remaining epochs step by step on these shapes (the
schedule passed to ``candidates.enumerate_candidates``), through the same
policy code as a measured session; one pricing memo per cell keeps the cost
of a (plan, shape, environment) and the material need of a (plan, shape), so
the per-step horizon does not multiply the run time.

Mechanisms simulated per trace family: S0-S3 link changes (telemetry lag
and fluid integration across changes), S4 public CPU share, S5 material
stock with refill (candidate feasibility, waits and burn on a switch), S6
length buckets with a per-party memory limit, S7 cancel / resume /
disconnect (the in-flight epoch's material is burned, the epoch is redone
after the resume, a session without a resume aborts), S8 initial installed
objects.

Policies (controller.make_policy): B0 fixed plans (and the derived
``B0_fixed_best_hindsight`` per trace, seed and T), B1_naive,
B2_remaining_plus_install, B3_planner_plus_B2, B4_opencrab, the ablations
B4_no_install / B4_no_hyst / B4_no_unc, delayed:2, B5_oracle (reads the
trace; the start of every remaining epoch is projected from the incumbent's
predicted per-step durations; pays the real install), B6_zero_overhead
(per-epoch cheapest plan under the
observed environment; the environment charges it no sensing, no install, no
plan-switch waste and no material wait, so its total is execution time only:
a lower bound, not an executable system). Policy order is randomized per
seed.

Planner x controller factorial: (ExactAssign, ILPAdapter, GreedyPerOperator)
x (B2, B4). The planner solves a four-operator chain (embedding, linear,
argmax, max) whose realizations are the axis values priced with the per-axis
execution cost summed over the remaining steps and the install objects they
need; the control rule then
decides between keep and the planner's assignment. Per decision the solver
latency, the optimality gap against ExactAssign and whether the assignment
equals ExactAssign's are recorded (identical assignments mean the
contribution is not assignment superiority).

Usage::

    python -m opencrab2.experiments.run_simulation [--quick] [--seeds 10] [--families S0,S1] [--T 8,32]
        [--workers N] [--keep-epochs] [--out-root DIR]

Outputs under the data root: raw/simulation.jsonl (one MODELED record per
session), raw/factorial.jsonl, processed/factorial_summary.json,
tables/factorial.tex (through make_tables). A record carries the session
totals; the per-epoch payload (``extra.epochs``, three quarters of the file
and read by no table) is written only with ``--keep-epochs``. The process
pool uses the ``spawn`` start method on every platform.

What is tested (tests/test_run_simulation.py): the quick campaign (2 seeds,
2 families) produces records that validate, carry status MODELED, cover
every policy, hold no per-epoch payload unless asked, and the factorial table
exists. Design limits: a fluid cost-model environment has no queueing or slow
start, telemetry is exact at the probe instant, and the noise model is the
fitted sigma (Gaussian on the log scale, clipped at three sigma).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import multiprocessing
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from opencrab2.experiments import schema, traces
from opencrab2.highlevel import candidates as cand
from opencrab2.highlevel import controller as ctl
from opencrab2.highlevel import costmodel as cm
from opencrab2.highlevel import planner as pl
from opencrab2.highlevel import registry
from opencrab2.highlevel.contract import DEFAULT_CONTRACT

MODEL_NAME = "openai-community/gpt2"
N_LAYERS = 12
PROMPT_TOKENS = 16
DEFAULT_TS: Tuple[int, ...] = (8, 32)
QUICK_TS: Tuple[int, ...] = (8,)
DEFAULT_SEEDS = 10
QUICK_SEEDS = 2
QUICK_FAMILIES: Tuple[str, ...] = ("S0", "S3")
TRACE_DURATION_S = 3600.0            # a long enough timeline for any T (the session ends when the tokens are out)
COLD_PLAN = {"embedding": "generic", "linear": "generic", "argmax_k": 2, "max_k": 2}
PROBE_BURST_BYTES = 65536
PROBE_SAMPLES = 3
RECONNECT_RTTS = 2
NOISE_CLIP_SIGMA = 3.0
DEFAULT_FIXED_PLANS: Tuple[dict, ...] = tuple(
    [{"embedding": e, "linear": l, "argmax_k": k, "max_k": 2} for e in ("wmask", "generic") for l in ("wmask", "generic")
     for k in (2, 4, 8)] + [{"embedding": "wmask", "linear": "wmask", "argmax_k": 2, "max_k": 4},
                            {"embedding": "wmask", "linear": "wmask", "argmax_k": 8, "max_k": 4}])
QUICK_FIXED_PLANS: Tuple[dict, ...] = ({"embedding": "wmask", "linear": "wmask", "argmax_k": 2, "max_k": 2},
                                       {"embedding": "generic", "linear": "generic", "argmax_k": 2, "max_k": 2})
ADAPTIVE_POLICIES: Tuple[str, ...] = ("B1_naive", "B2_remaining_plus_install", "B3_planner_plus_B2", "B4_opencrab",
                                      "B4_no_install", "B4_no_hyst", "B4_no_unc", "delayed:2", "B5_oracle",
                                      "B6_zero_overhead")
ZERO_OVERHEAD_POLICY = "B6_zero_overhead"
PLANNERS: Tuple[str, ...] = ("ExactAssign", "ILPAdapter", "GreedyPerOperator")
CONTROLS: Tuple[str, ...] = ("B2", "B4")
AXIS_ORDER: Tuple[str, ...] = ("embedding", "linear", "argmax_k", "max_k")
MEMO_MAX_ENTRIES = 200000            # the pricing memo drops its per-environment entries above this size


# ---------------------------------------------------------------------------
# cached cost model and environment
# ---------------------------------------------------------------------------
class CachedCostModel:
    """Wraps a FittedCostModel; ``predict`` is memoised on (component, shape, rounded env).

    ``memo()`` is the pricing memo of ``candidates.enumerate_candidates`` for
    this model and the default contract, shared by every session of a cell.
    """

    def __init__(self, model: cm.FittedCostModel):
        self.model = model
        self._cache: Dict[tuple, Tuple[float, float]] = {}
        self._memo: Dict[tuple, Any] = {}
        self.label = model.label
        self.fits = model.fits

    def memo(self) -> Dict[tuple, Any]:
        """The memo; once it is large, the entries of past environments go and the material needs stay."""
        if len(self._memo) > MEMO_MAX_ENTRIES:
            self._memo = {k: v for k, v in self._memo.items() if k[0] == "material"}
        return self._memo

    @staticmethod
    def _key(env: Any) -> tuple:
        e = cm.as_env(env)
        bw = None if not e.bandwidth_bps else float(f"{e.bandwidth_bps:.4g}")
        return (bw, float(f"{e.one_way_latency_s:.4g}"), float(f"{e.cpu_share:.3g}"))

    def predict(self, component: str, n: int, d_in: int, d_out: int, env: Any) -> Tuple[float, float]:
        k = (component, int(n), int(d_in), int(d_out)) + self._key(env)
        v = self._cache.get(k)
        if v is None:
            v = self.model.predict(component, n, d_in, d_out, env)
            self._cache[k] = v
        return v

    def has(self, component: str) -> bool:
        return self.model.has(component)

    def status_of(self, component: str) -> str:
        return self.model.status_of(component)


def env_from_state(st: Mapping[str, float]) -> cm.Env:
    return cm.Env(float(st["bandwidth_bps"]), float(st["one_way_delay_s"]), float(st.get("cpu_share", 1.0)))


def next_change_after(trace: Mapping[str, Any], t: float) -> float:
    for p in trace["points"]:
        if float(p[0]) > t:
            return float(p[0])
    return math.inf


# ---------------------------------------------------------------------------
# session simulation
# ---------------------------------------------------------------------------
@dataclass
class EpochLog:
    epoch: int
    t_start: float
    plan: dict
    switched: bool
    install_s: float
    exec_s: float
    sensing_s: float
    wait_s: float
    recovery_s: float
    reason: str
    lower_bound_gain: float
    q_keep: float
    q_best: float
    rejected: int
    burned_bytes: int


@dataclass
class SessionResult:
    policy_id: str
    total_s: float
    exec_s: float
    install_s: float
    initial_install_s: float
    sensing_s: float
    wait_s: float
    recovery_s: float
    tokens: int
    aborts: int
    initializations: int
    post_init_switches: int
    false_switches: int
    rejected_candidates: int
    bytes: int
    rounds: int
    material_consumed: int
    material_burned: int
    material_generated: int
    final_plan: dict
    epochs: List[EpochLog] = field(default_factory=list)
    decisions: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


def _noise_factor(rng: np.random.Generator, mean: float, sigma: float) -> float:
    if mean <= 0.0 or sigma <= 0.0:
        return 1.0
    rel = min(sigma / mean, 1.0)
    z = float(np.clip(rng.standard_normal(), -NOISE_CLIP_SIGMA, NOISE_CLIP_SIGMA))
    s = math.sqrt(math.log1p(rel * rel))
    return math.exp(s * z - 0.5 * s * s)


def _integrate(trace: Mapping[str, Any], t0: float, cost_of_env: Any, noise: float) -> float:
    """Wall time to finish one unit of work started at t0 when the cost under env e is cost_of_env(e) * noise."""
    t = t0
    work = 1.0
    guard = 0
    while work > 1e-12 and guard < 100000:
        guard += 1
        st = traces.state_at(trace, t)
        c = max(1e-9, float(cost_of_env(st)) * noise)
        seg_end = next_change_after(trace, t)
        dt = min(seg_end - t, work * c)
        work -= dt / c
        t += dt
    return t - t0


def _bytes_rounds(plan: dict, shapes: dict) -> Tuple[int, int]:
    """Online bytes and rounds of one epoch: the exact figure of ``candidates.plan_bytes_rounds`` (batched entries,
    such as the per-head attention products, share one round and one set of frames)."""
    b, r = cand.plan_bytes_rounds(plan, shapes)
    return int(b), int(r)


def _probe_cost(env: cm.Env) -> float:
    bw = env.bandwidth_bps or math.inf
    return PROBE_SAMPLES * (2 * env.rtt_s + PROBE_BURST_BYTES * 8.0 / bw) + env.rtt_s


def _shapes_for(epoch: int, T: int, bucket: int) -> dict:
    n = PROMPT_TOKENS if epoch == 0 else 1
    kv = PROMPT_TOKENS + epoch
    if bucket > 0:
        kv = max(bucket, int(math.ceil(kv / bucket)) * bucket)
    return cand.gpt2_shapes(N_LAYERS, n, kv)


class _Inventory:
    """Simulated material stock per type (S5): bytes available, proportional refill along the plan mix."""

    def __init__(self, initial_bytes: Optional[int], refill_bps: float, mix: Mapping[str, int]):
        self.enabled = initial_bytes is not None
        self.refill_bps = float(refill_bps)
        total = float(sum(mix.values())) or 1.0
        self.weights = {k: v / total for k, v in mix.items()}
        self.stock: Dict[str, float] = {k: float(initial_bytes or 0) * w for k, w in self.weights.items()}
        self.generated = 0

    def as_state(self) -> Dict[str, Tuple[int, int]]:
        if not self.enabled:
            return {}
        return {k: (0, int(v)) for k, v in self.stock.items()}

    def refill(self, dt: float, mix: Optional[Mapping[str, int]] = None) -> None:
        if not self.enabled or dt <= 0.0:
            return
        add = self.refill_bps * dt
        w = self.weights
        if mix:
            tot = float(sum(mix.values())) or 1.0
            w = {k: v / tot for k, v in mix.items()}
        for k, frac in w.items():
            self.stock[k] = self.stock.get(k, 0.0) + add * frac
        self.generated += int(add)

    def wait_for(self, need: Mapping[str, int]) -> float:
        """Seconds of refill needed before every type covers ``need`` (0 when already covered)."""
        if not self.enabled:
            return 0.0
        tot = float(sum(need.values())) or 1.0
        worst = 0.0
        for k, b in need.items():
            have = self.stock.get(k, 0.0)
            if have < b:
                rate = self.refill_bps * (b / tot)
                worst = max(worst, (b - have) / max(rate, 1e-9))
        return worst

    def consume(self, need: Mapping[str, int]) -> None:
        if not self.enabled:
            return
        for k, b in need.items():
            self.stock[k] = max(0.0, self.stock.get(k, 0.0) - b)


def make_policy(name: str, costmodel: Any, shapes: dict, trace: Mapping[str, Any], planner: Any = None) -> ctl.Policy:
    if name.startswith("B0_fixed:"):
        return ctl.make_policy(name)
    if name == "B5_oracle":
        return ctl.make_policy(name, trace={0: cm.Env(1e9, 0.0)}, costmodel=costmodel, shapes=shapes)
    if name == "B3_planner_plus_B2":
        return ctl.make_policy(name, planner=planner if planner is not None else pl.ExactAssign())
    return ctl.make_policy(name)


def _oracle_envs(trace: Mapping[str, Any], t: float, steps_s: Sequence[float], epoch: int) -> Dict[int, cm.Env]:
    """Trace states at the projected starts of the remaining epochs (the oracle reads the exogenous trace).

    ``steps_s[j]`` is the predicted duration of epoch ``epoch + j`` under the
    incumbent; the result is keyed by the epoch number, as ``OraclePolicy``
    reads it.
    """
    out = {}
    start = float(t)
    for j, step_s in enumerate(steps_s):
        out[int(epoch) + j] = env_from_state(traces.state_at(trace, start))
        start += max(float(step_s), 1e-3)
    return out


def session_duration_estimate(family: str, variant: str, T: int, costmodel: Any) -> float:
    """Trace duration for a cell: 1.2 x the predicted session of the cheapest plan plus a full install.

    Events (ramps, changes, cancels) are placed as fractions of the duration,
    so tying the duration to the predicted session length keeps them inside
    the session on fast and slow links alike; the cheapest plan (minimum over
    the catalog per epoch) is the floor any policy can approach, and the full
    weight-mask install is the largest one-time cost. Deterministic given the
    model and the trace's initial state.
    """
    probe = traces.make_trace(family, variant, 0, 1.0)
    env = env_from_state(traces.state_at(probe, 0.0))
    bucket = int(probe.get("length_bucket", 0) or 0)
    shapes0 = _shapes_for(0, T, bucket)
    catalog = cand.gpt2_catalog(shapes0)
    plans = [dict(zip(catalog["axes"], combo)) for combo in __import__("itertools").product(*catalog["axes"].values())]
    total = cand.install_cost(list(catalog["objects"]), catalog, costmodel, env)[0]
    for epoch in range(1 + int(T)):
        shapes = _shapes_for(epoch, T, bucket)
        total += min(cand.exec_cost(p, shapes, costmodel, env)[0] for p in plans) + _probe_cost(env)
    return max(10.0, 1.2 * total)


def _stable_seed(seed: int, policy_name: str) -> int:
    import hashlib
    h = hashlib.sha256(f"{int(seed)}|{policy_name}".encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big")


def simulate_session(trace: Mapping[str, Any], policy_name: str, T: int, seed: int, costmodel: CachedCostModel,
                     planner: Any = None, factorial: Optional[Tuple[str, str]] = None) -> SessionResult:
    """Run one session of 1 + T epochs under ``policy_name``; returns the SessionResult (MODELED)."""
    rng = np.random.default_rng(_stable_seed(seed, policy_name))
    contract_hash = DEFAULT_CONTRACT.contract_hash()
    bucket = int(trace.get("length_bucket", 0) or 0)
    shapes0 = _shapes_for(0, T, bucket)
    catalog = cand.gpt2_catalog(shapes0)
    version = {"model_version": f"{MODEL_NAME}@sim", "client_key_domain": "client-default"}
    installed = {o: dict(version) for o in trace.get("initial_installed", []) if o in catalog["objects"]}
    mem_limit = trace.get("mem_limit_bytes")
    inv_mix = cand.material_per_epoch(COLD_PLAN, _shapes_for(1, T, bucket), DEFAULT_CONTRACT)
    refill = float(traces.state_at(trace, 0.0)["stock_refill_bps"])
    inventory = _Inventory(trace.get("initial_stock_bytes"), refill, inv_mix)
    events = [(float(t), str(k)) for t, k in trace.get("events", [])]
    incumbent = dict(COLD_PLAN)
    zero_overhead = policy_name == ZERO_OVERHEAD_POLICY and factorial is None
    fact = _FactorialDecider(factorial[0], factorial[1]) if factorial else None
    policy = make_policy(policy_name, costmodel, shapes0, trace, planner) if fact is None else ctl.Policy()
    res = SessionResult(policy_name, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, dict(incumbent))
    t = 0.0
    tokens = 0
    epoch = 0
    switch_history: List[Tuple[int, str]] = []
    admitted_cache: Dict[str, Tuple[bool, str]] = {}
    total_epochs = 1 + int(T)
    epoch_shapes = [_shapes_for(e, T, bucket) for e in range(total_epochs)]      # the public shape of every epoch
    if mem_limit is not None:
        for sh in epoch_shapes:
            sh["mem_limit_bytes"] = int(mem_limit)
    while epoch < total_epochs:
        shapes = epoch_shapes[epoch]
        schedule = epoch_shapes[epoch:]
        # ---- sensing: public telemetry at the probe instant ---------------------------
        st_obs = traces.state_at(trace, t)
        env_obs = env_from_state(st_obs)
        sensing = 0.0 if zero_overhead else _probe_cost(env_obs)
        frame = {"bandwidth_bps": env_obs.bandwidth_bps, "rtt_s": env_obs.rtt_s}
        if fact is not None:
            fact.observe(frame)
        else:
            policy.observe(frame)
        t += sensing
        res.sensing_s += sensing
        inventory.refill(sensing)
        # ---- candidates ------------------------------------------------------------------
        state = cand.SessionState(dict(incumbent), dict(installed), inventory.as_state(), int(shapes["T"]),
                                  total_epochs - epoch, f"sim-{trace['trace_id']}-{seed}", version["model_version"],
                                  contract_hash, version["client_key_domain"], mem_limit)

        def admit_fn(plan: dict, _state=state, _shapes=shapes) -> Tuple[bool, str]:
            key = cand.plan_key(plan)
            if key not in admitted_cache:
                tape = registry.default_gpt2_tape(plan, N_LAYERS, _shapes, contract_hash)
                admitted_cache[key] = registry.admit(tape, registry.PROFILE_ID, _state)
            return admitted_cache[key]

        memo = costmodel.memo()
        cands = cand.enumerate_candidates(state, catalog, costmodel, env_obs, shapes, DEFAULT_CONTRACT, admit_fn,
                                          schedule=schedule, memo=memo)
        rejected = sum(1 for c in cands if not c.feasible)
        res.rejected_candidates += rejected
        # ---- decision --------------------------------------------------------------------
        if policy_name == "B5_oracle":
            policy = ctl.make_policy("B5_oracle", trace=_oracle_envs(trace, t, cands[0].exec_s_by_epoch, epoch),
                                     costmodel=costmodel, shapes=shapes, schedule=schedule)
        if fact is not None:
            decision = fact.decide(state, cands, env_obs, epoch, costmodel, catalog, schedule, memo)
        else:
            decision = policy.decide(state, cands, env_obs, epoch)
        res.decisions += 1
        chosen = dict(decision.chosen)
        chosen_c = next((c for c in cands if c.plan == chosen), cands[0])
        switched = chosen != incumbent
        # ---- material wait and burn on a switch -----------------------------------------
        need = chosen_c.material_by_type
        wait = 0.0 if zero_overhead else inventory.wait_for(need)
        burned = 0
        if switched and inventory.enabled and not zero_overhead:
            # staged one-time material of the incumbent for this epoch is burned (plan-switch waste)
            inc_need = cands[0].material_by_type
            burned = int(sum(min(inventory.stock.get(k, 0.0), v) for k, v in inc_need.items()))
            inventory.consume(inc_need)
            wait = inventory.wait_for(need)
        if wait > 0.0:
            inventory.refill(wait, need)
            t += wait
            res.wait_s += wait
        # ---- install (candidate-specific, real objects only; the zero-overhead oracle pays nothing) ----
        install_s = 0.0
        if chosen_c.install_objects:
            if not zero_overhead:
                def inst_cost(st_env: Mapping[str, float], _objs=list(chosen_c.install_objects)) -> float:
                    return cand.install_cost(_objs, catalog, costmodel, env_from_state(st_env))[0]
                noise = _noise_factor(rng, chosen_c.c_install_s, chosen_c.c_install_sigma)
                install_s = _integrate(trace, t, inst_cost, noise)
                t += install_s
                res.install_s += install_s
                if epoch == 0:
                    res.initial_install_s += install_s
                inventory.refill(install_s)
            for o in chosen_c.install_objects:
                installed[o] = dict(version)
        if epoch == 0 and (switched or chosen_c.install_objects):
            res.initializations += 1          # cold-start initialization (plan choice and/or first installs)
        if switched:
            if epoch > 0:
                res.post_init_switches += 1
            # a switch back to the plan left at most two epochs earlier counts as a false switch
            if len(switch_history) >= 1 and epoch - switch_history[-1][0] <= 2 and \
                    any(k == cand.plan_key(chosen) for _, k in switch_history[-2:-1]):
                res.false_switches += 1
            switch_history.append((epoch, cand.plan_key(chosen)))
        incumbent = chosen
        # ---- execution with S7 events --------------------------------------------------------
        def exec_cost(st_env: Mapping[str, float], _plan=dict(chosen), _shapes=dict(shapes)) -> float:
            return cand.exec_cost(_plan, _shapes, costmodel, env_from_state(st_env))[0]

        noise = _noise_factor(rng, chosen_c.exec_s_per_epoch, chosen_c.exec_sigma)
        dur = _integrate(trace, t, exec_cost, noise)
        recovery = 0.0
        aborted = False
        while True:
            # events inside this execution window, plus events missed while sensing or installing (they
            # strike the epoch at its start)
            ev = [e for e in events if e[0] < t + dur]
            if not ev:
                break
            t_ev, kind = ev[0]
            if kind not in ("cancel", "disconnect"):
                events = [e for e in events if e != ev[0]]
                continue
            # the in-flight epoch is lost: its material is burned, the clock moves to the event
            burned += int(sum(need.values()))
            events = [e for e in events if e != ev[0]]
            t = max(t_ev, t)
            resume = next((e for e in events if e[1] == "resume" and e[0] >= t_ev), None)
            if resume is None:
                aborted = True
                break
            events = [e for e in events if e != resume]
            env_r = env_from_state(traces.state_at(trace, resume[0]))
            rec_s = RECONNECT_RTTS * env_r.rtt_s
            recovery += (resume[0] - t_ev) + rec_s
            t = resume[0] + rec_s
            inventory.refill(resume[0] - t_ev)
            inventory.generated += int(sum(need.values()))      # fresh bundles after the resume
            dur = _integrate(trace, t, exec_cost, noise)
        if aborted:
            res.aborts += 1
            res.recovery_s += recovery
            res.material_burned += burned
            res.epochs.append(EpochLog(epoch, t, dict(chosen), switched, install_s, 0.0, sensing, wait, recovery,
                                       "ABORTED_BY_CANCEL", decision.lower_bound_gain, decision.q_keep, decision.q_best,
                                       rejected, burned))
            break
        t += dur
        inventory.refill(dur, need)
        inventory.consume(need)
        res.exec_s += dur
        res.recovery_s += recovery
        res.material_burned += burned
        res.material_consumed += int(sum(need.values()))
        inventory.generated += 0 if inventory.enabled else int(sum(need.values()))
        b, r = _bytes_rounds(chosen, shapes)
        res.bytes += b
        res.rounds += r
        tokens += 1
        res.epochs.append(EpochLog(epoch, t - dur, dict(chosen), switched, install_s, dur, sensing, wait, recovery,
                                   decision.reason, decision.lower_bound_gain, decision.q_keep, decision.q_best,
                                   rejected, burned))
        epoch += 1
    res.total_s = t
    res.tokens = tokens
    res.final_plan = dict(incumbent)
    res.material_generated = int(inventory.generated if inventory.enabled else res.material_consumed + res.material_burned)
    if fact is not None:
        res.extra.update(fact.summary())
    return res


# ---------------------------------------------------------------------------
# planner x controller factorial
# ---------------------------------------------------------------------------
def axis_costs(plan_axes: Mapping[str, Sequence[Any]], shapes: dict, costmodel: Any, env: Any) -> Dict[str, Dict[Any, float]]:
    """Per-axis per-epoch execution cost (separable: embedding call, layer linears, argmax, softmax)."""
    out: Dict[str, Dict[Any, float]] = {a: {} for a in AXIS_ORDER}
    base = dict(COLD_PLAN)
    for axis in AXIS_ORDER:
        for v in plan_axes[axis]:
            p = dict(base)
            p[axis] = v
            tot = 0.0
            for comp, n, d_in, d_out, count in cand.plan_calls(p, shapes):
                if axis == "embedding" and comp.startswith("embedding"):
                    tot += count * costmodel.predict(comp, n, d_in, d_out, env)[0]
                elif axis == "linear" and comp in ("linear_wmask", "linear_generic") and not (n == 1 and d_out == shapes["vocab"]):
                    tot += count * costmodel.predict(comp, n, d_in, d_out, env)[0]
                elif axis == "argmax_k" and comp.startswith("argmax_k"):
                    tot += count * costmodel.predict(comp, n, d_in, d_out, env)[0]
                elif axis == "max_k" and comp.startswith("softmax_rows_k"):
                    tot += count * costmodel.predict(comp, n, d_in, d_out, env)[0]
            out[axis][v] = tot
    return out


def build_chain(catalog: Mapping[str, Any], shapes: dict, costmodel: Any, env: Any, H: int,
                schedule: Optional[Sequence[Mapping[str, Any]]] = None, memo: Optional[dict] = None) -> Tuple[list, dict]:
    """Four-operator chain (one per plan axis) with install objects; costs are the remaining epochs of execution.

    With ``schedule`` (the public shapes of the remaining epochs) the cost of
    an axis value is its execution cost summed over the scheduled steps;
    without it, H times its cost on ``shapes``. ``memo`` keeps the per-axis
    costs of a (shape, environment).
    """
    e = cm.as_env(env)
    total: Dict[str, Dict[Any, float]] = {a: {v: 0.0 for v in catalog["axes"][a]} for a in AXIS_ORDER}
    for sh in (schedule if schedule is not None else [shapes]):
        key = ("axis", cand.shape_key(sh), (e.bandwidth_bps, e.one_way_latency_s, e.cpu_share))
        costs = memo.get(key) if memo is not None else None
        if costs is None:
            costs = axis_costs(catalog["axes"], sh, costmodel, env)
            if memo is not None:
                memo[key] = costs
        for axis in AXIS_ORDER:
            for v in catalog["axes"][axis]:
                total[axis][v] += costs[axis][v] if schedule is not None else H * costs[axis][v]
    chain = []
    for axis in AXIS_ORDER:
        reals = []
        for v in catalog["axes"][axis]:
            reals.append({"name": str(v), "in_domain": "A64", "out_domain": "A64", "cost": total[axis][v],
                          "objects": list(catalog["requires"][axis].get(v, []))})
        chain.append({"id": axis, "realizations": reals})
    pcat = {"start_domain": "A64", "conversions": {}, "objects": dict(catalog["objects"])}
    return chain, pcat


def _assignment_to_plan(assignment: Mapping[str, str], catalog: Mapping[str, Any]) -> dict:
    plan = {}
    for axis in AXIS_ORDER:
        name = assignment[axis]
        for v in catalog["axes"][axis]:
            if str(v) == str(name):
                plan[axis] = v
    return plan


class _FactorialDecider:
    """Planner P chooses the assignment of the chain; control rule C decides keep versus that assignment."""

    def __init__(self, planner_name: str, control: str):
        self.planner_name = planner_name
        self.control = control
        self.planner = {"ExactAssign": pl.ExactAssign(), "ILPAdapter": pl.ILPAdapter(),
                        "GreedyPerOperator": pl.GreedyPerOperator()}[planner_name]
        self.exact = pl.ExactAssign()
        self.ctrl = ctl.StateAwareController(ctl.ControllerConfig()) if control == "B4" else None
        self.latency: List[float] = []
        self.gap: List[float] = []
        self.identical = 0
        self.decisions = 0

    def observe(self, frame: Any) -> None:
        if self.ctrl is not None:
            self.ctrl.observe(frame)

    def decide(self, state: cand.SessionState, cands: Sequence[cand.Candidate], env: Any, epoch: int, costmodel: Any,
               catalog: Mapping[str, Any], schedule: Sequence[Mapping[str, Any]],
               memo: Optional[dict] = None) -> ctl.Decision:
        H = max(1, int(state.remaining_epochs))
        chain, pcat = build_chain(catalog, schedule[0], costmodel, env, H, schedule=schedule, memo=memo)
        t0 = time.perf_counter()
        assignment, cost = self.planner.plan(chain, pcat, env, costmodel, state)
        self.latency.append(time.perf_counter() - t0)
        ex_assign, ex_cost = self.exact.plan(chain, pcat, env, costmodel, state)
        self.gap.append(max(0.0, float(cost) - float(ex_cost)))
        self.identical += int(assignment == ex_assign)
        self.decisions += 1
        plan = _assignment_to_plan(assignment, catalog)
        keep = cands[0]
        pick = next((c for c in cands if c.plan == plan), None)
        if pick is None or not pick.feasible:
            pick = keep
        restricted = [keep] if pick is keep else [keep, pick]
        if self.ctrl is not None:
            return self.ctrl.decide(state, restricted, epoch)
        return ctl.RemainingPlusInstallPolicy().decide(state, restricted, env, epoch)

    def summary(self) -> dict:
        return {"planner": self.planner_name, "control": self.control, "decisions": self.decisions,
                "solver_latency_mean_s": float(np.mean(self.latency)) if self.latency else 0.0,
                "solver_latency_max_s": float(np.max(self.latency)) if self.latency else 0.0,
                "optimality_gap_mean_s": float(np.mean(self.gap)) if self.gap else 0.0,
                "identical_to_exact_rate": self.identical / self.decisions if self.decisions else float("nan")}


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------
def session_record(trace: Mapping[str, Any], T: int, seed: int, res: SessionResult, model_label: dict,
                   raw_path: str, kind: str = "simulation", keep_epochs: bool = False) -> dict:
    """One MODELED record of a simulated session; ``extra.epochs`` (the per-epoch log) only with ``keep_epochs``.

    ``raw_path`` is the record's ``raw_measurement_path`` relative to the data root (``raw/simulation.jsonl``).
    """
    extra = {
        "exec_s": res.exec_s, "install_s": res.install_s, "initial_install_s": res.initial_install_s,
        "sensing_s": res.sensing_s, "wait_s": res.wait_s, "recovery_s": res.recovery_s,
        "false_switches": res.false_switches, "decisions": res.decisions, "final_plan": res.final_plan,
        "n_epochs_logged": len(res.epochs), "model_status": model_label, "prompt_tokens": PROMPT_TOKENS,
        "n_layers": N_LAYERS, "trace_meta": dict(trace.get("meta", {})),
    }
    if keep_epochs:
        extra["epochs"] = [e.__dict__ for e in res.epochs]
    extra.update(res.extra)
    return schema.new_record(
        status="MODELED", record_kind=kind, workload_scope="full-model", clock_basis="modeled-epoch", dealer_mode="NONE",
        model_revision=f"{MODEL_NAME} (shapes only; MODELED)", trace_id=trace["trace_id"], trace_hash=trace["trace_hash"],
        policy_id=res.policy_id, initial_state_hash=schema.stable_hash({"installed": sorted(trace.get("initial_installed", [])),
                                                                         "plan": COLD_PLAN}),
        run_id=f"{kind}:{trace['trace_id']}:T{T}:{res.policy_id}", repeat=int(seed), initializations=res.initializations,
        post_init_switches=res.post_init_switches, rejected_candidates=res.rejected_candidates, aborts=res.aborts,
        T_cold_s=res.total_s, T_warm_s=max(0.0, res.total_s - res.initial_install_s), T_online_s=res.exec_s,
        bytes=res.bytes, rounds=res.rounds, material_generated=res.material_generated,
        material_transferred=res.material_generated, material_consumed=res.material_consumed,
        material_burned=res.material_burned, tokens=res.tokens, output_match=None,
        raw_measurement_path=schema.relative_raw_path(raw_path), seed=int(seed), family=trace["family"],
        variant=trace["variant"], split=trace["split"], T=int(T), n_layers=N_LAYERS, extra=extra)


def model_status_map(model: cm.FittedCostModel) -> dict:
    return {k: v.status for k, v in model.fits.items()}


# ---------------------------------------------------------------------------
# campaign
# ---------------------------------------------------------------------------
SIMULATION_RAW = "raw/simulation.jsonl"
FACTORIAL_RAW = "raw/factorial.jsonl"


def run_cell(args: Tuple[Any, ...]) -> Tuple[List[dict], List[dict]]:
    """One (family, variant, seed, T): every policy in a seed-randomized order plus the factorial; returns records.

    ``args`` is ``(family, variant, seed, T, policies, fixed_plans, with_factorial[, keep_epochs])``.
    """
    family, variant, seed, T, policies, fixed_plans, with_factorial = args[:7]
    keep_epochs = bool(args[7]) if len(args) > 7 else False
    model = cm.default_model()
    cmodel = CachedCostModel(model)
    status = model_status_map(model)
    duration = session_duration_estimate(family, variant, int(T), cmodel)
    trace = traces.make_trace(family, variant, int(seed), duration)
    order = list(policies) + [f"B0_fixed:{json.dumps(p, sort_keys=True)}" for p in fixed_plans]
    perm = np.random.default_rng(int(seed)).permutation(len(order))
    order = [order[i] for i in perm]
    recs: List[dict] = []
    fixed_results: List[Tuple[dict, SessionResult]] = []
    for name in order:
        res = simulate_session(trace, name, int(T), int(seed), cmodel)
        recs.append(session_record(trace, T, seed, res, status, SIMULATION_RAW, keep_epochs=keep_epochs))
        if name.startswith("B0_fixed:"):
            fixed_results.append((json.loads(name[len("B0_fixed:"):]), res))
    if fixed_results:
        best_plan, best = min(fixed_results, key=lambda pr: pr[1].total_s)
        derived = SessionResult(**{**best.__dict__, "policy_id": "B0_fixed_best_hindsight"})
        derived.extra = dict(best.extra, hindsight_plan=best_plan, fixed_plans_run=len(fixed_results))
        recs.append(session_record(trace, T, seed, derived, status, SIMULATION_RAW, keep_epochs=keep_epochs))
    fact_recs: List[dict] = []
    if with_factorial:
        for pn in PLANNERS:
            for ctrl_name in CONTROLS:
                res = simulate_session(trace, f"F:{pn}x{ctrl_name}", int(T), int(seed), cmodel, factorial=(pn, ctrl_name))
                fact_recs.append(session_record(trace, T, seed, res, status, FACTORIAL_RAW, kind="factorial",
                                                keep_epochs=keep_epochs))
    return recs, fact_recs


def factorial_summary(records: Sequence[Mapping[str, Any]]) -> dict:
    out: Dict[str, dict] = {}
    for r in records:
        e = r["extra"]
        key = f"{e['planner']}x{e['control']}"
        s = out.setdefault(key, {"planner": e["planner"], "control": e["control"], "n": 0, "total_s": [],
                                 "solver_latency_mean_s": [], "optimality_gap_mean_s": [], "identical_to_exact_rate": [],
                                 "post_init_switches": []})
        s["n"] += 1
        s["total_s"].append(float(r["T_cold_s"]))
        s["solver_latency_mean_s"].append(float(e["solver_latency_mean_s"]))
        s["optimality_gap_mean_s"].append(float(e["optimality_gap_mean_s"]))
        s["identical_to_exact_rate"].append(float(e["identical_to_exact_rate"]))
        s["post_init_switches"].append(int(r["post_init_switches"]))
    for s in out.values():
        for k in ("total_s", "solver_latency_mean_s", "optimality_gap_mean_s", "identical_to_exact_rate", "post_init_switches"):
            v = np.asarray(s[k], dtype=np.float64)
            s[k] = {"mean": float(np.nanmean(v)), "median": float(np.nanmedian(v)), "n": int(v.size)}
    return out


def run_campaign(families: Sequence[str], seeds: Sequence[int], Ts: Sequence[int], policies: Sequence[str],
                 fixed_plans: Sequence[dict], out_root: Path, workers: int, with_factorial: bool = True,
                 variants_limit: Optional[int] = None, keep_epochs: bool = False) -> dict:
    t0 = time.perf_counter()
    raw_dir = out_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cells: List[tuple] = []
    for fam in families:
        vs = list(traces.variants(fam))
        if variants_limit:
            vs = vs[:variants_limit]
        for v in vs:
            for s in seeds:
                for T in Ts:
                    cells.append((fam, v, int(s), int(T), tuple(policies), tuple(fixed_plans), with_factorial,
                                  bool(keep_epochs)))
    sim_path = raw_dir / "simulation.jsonl"
    fac_path = raw_dir / "factorial.jsonl"
    done = schema.completed_run_ids(sim_path)
    done_fact = schema.completed_run_ids(fac_path)       # a resumed campaign never appends a factorial record twice
    n_rec = n_fact = 0
    fact_records: List[dict] = []
    model = cm.default_model()

    def _emit(recs: List[dict], facts: List[dict]) -> None:
        nonlocal n_rec, n_fact
        for r in recs:
            if r["run_id"] + f":s{r['repeat']}" in done:
                continue
            r["run_id"] = r["run_id"] + f":s{r['repeat']}"
            schema.append_jsonl(sim_path, r)
            n_rec += 1
        for r in facts:
            if r["run_id"] + f":s{r['repeat']}" in done_fact:
                continue
            r["run_id"] = r["run_id"] + f":s{r['repeat']}"
            schema.append_jsonl(fac_path, r)
            fact_records.append(r)
            n_fact += 1

    if workers <= 1 or len(cells) == 1:
        for c in cells:
            recs, facts = run_cell(c)
            _emit(recs, facts)
    else:
        # spawn on every platform: the Linux default (fork) was never exercised on the measured host
        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as ex:
            for recs, facts in ex.map(run_cell, cells, chunksize=1):
                _emit(recs, facts)
    all_fact = [r for r in schema.read_jsonl(fac_path) if r.get("record_kind") == "factorial"]
    summary = {"cells": len(cells), "records": n_rec, "factorial_records": n_fact, "elapsed_s": time.perf_counter() - t0,
               "model_label": model.label, "model_status": model_status_map(model),
               "factorial": factorial_summary(all_fact), "policies": list(policies),
               "fixed_plans": [dict(p) for p in fixed_plans], "families": list(families), "seeds": list(seeds), "T": list(Ts),
               "keep_epochs": bool(keep_epochs)}
    proc = out_root / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    with open(proc / "factorial_summary.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True, default=str)
    try:
        from opencrab2.experiments import make_tables
        make_tables.write_factorial_table(out_root)
    except Exception as exc:  # the table step must never lose the records
        summary["table_error"] = f"{exc.__class__.__name__}: {exc}"
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run_simulation", description="MODELED session campaign over the trace matrix")
    p.add_argument("--quick", action="store_true", help="2 seeds, families S0 and S3 (first two variants), T = 8")
    p.add_argument("--seeds", type=int, default=None, help="number of seeds (1..N)")
    p.add_argument("--families", default=None, help="comma separated families (default: all S0..S8)")
    p.add_argument("--T", default=None, help="comma separated decode lengths (default 8,32)")
    p.add_argument("--policies", default=None, help="comma separated adaptive policies")
    p.add_argument("--fixed-plans", default=None, help="'all' for every plan, 'quick' for two, default ten")
    p.add_argument("--workers", type=int, default=None, help="process pool size (default: min(cpu, 16))")
    p.add_argument("--no-factorial", action="store_true")
    p.add_argument("--keep-epochs", action="store_true",
                   help="write the per-epoch log (extra.epochs) into every record; off by default (no table reads it)")
    p.add_argument("--out-root", default=None)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    out_root = Path(args.out_root) if args.out_root else schema.experiments_root()
    if args.quick:
        families = list(QUICK_FAMILIES)
        seeds = list(range(1, QUICK_SEEDS + 1))
        Ts = list(QUICK_TS)
        fixed = list(QUICK_FIXED_PLANS)
        variants_limit = 2
    else:
        families = list(traces.FAMILY_NAMES)
        seeds = list(range(1, DEFAULT_SEEDS + 1))
        Ts = list(DEFAULT_TS)
        fixed = list(DEFAULT_FIXED_PLANS)
        variants_limit = None
    if args.families:
        families = args.families.split(",")
    if args.seeds:
        seeds = list(range(1, int(args.seeds) + 1))
    if args.T:
        Ts = [int(v) for v in args.T.split(",")]
    policies = args.policies.split(",") if args.policies else list(ADAPTIVE_POLICIES)
    if args.fixed_plans == "all":
        fixed = [{"embedding": e, "linear": l, "argmax_k": k, "max_k": m} for e in ("wmask", "generic")
                 for l in ("wmask", "generic") for k in (2, 4, 8) for m in (2, 4)]
    elif args.fixed_plans == "quick":
        fixed = list(QUICK_FIXED_PLANS)
    workers = args.workers if args.workers is not None else max(1, min(16, (os.cpu_count() or 2)))
    summary = run_campaign(families, seeds, Ts, policies, fixed, out_root, workers, not args.no_factorial, variants_limit,
                           keep_epochs=bool(args.keep_epochs))
    sys.stdout.write(json.dumps({k: v for k, v in summary.items() if k not in ("factorial", "model_status")}, indent=1) + "\n")
    for k, v in summary["factorial"].items():
        sys.stdout.write(f"{k:>32s} total={v['total_s']['mean']:.2f}s latency={v['solver_latency_mean_s']['mean'] * 1e3:.2f}ms "
                         f"gap={v['optimality_gap_mean_s']['mean']:.4f}s identical={v['identical_to_exact_rate']['mean']:.3f}\n")
    return 0


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main())


__all__ = ["CachedCostModel", "SessionResult", "EpochLog", "simulate_session", "session_record", "run_cell",
           "run_campaign", "factorial_summary", "build_chain", "axis_costs", "ADAPTIVE_POLICIES", "PLANNERS",
           "CONTROLS", "DEFAULT_FIXED_PLANS", "QUICK_FIXED_PLANS", "COLD_PLAN", "SIMULATION_RAW", "FACTORIAL_RAW",
           "main"]
