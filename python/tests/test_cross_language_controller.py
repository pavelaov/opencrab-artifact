"""Cross-language controller parity.

Both controller vector files under ``cpp/tests/vectors`` are replayed with the
Python ``StateAwareController``:

* ``controller_vectors.json`` (written by ``cpp/tests/gen_vectors.py`` from the
  Python controller, replayed by the C++ ``test_vectors``);
* ``cpp_controller_vectors.json`` (written by the C++ ``test_decision --vectors``).

Every instance carries the configuration, the telemetry samples, the state,
the candidates, the survival vector and the decision.  The replay feeds the
telemetry to ``observe`` and calls ``decide_with_survival`` with the stored
survival vector.  Asserted for every instance: the chosen plan, the
switch flag and Qhat(keep); for a switching instance also Qhat of the chosen
plan, the lower-bound gain and U_diff; for a non-switching instance the
lower-bound gain and U_diff of the best feasible challenger, read from the
per-candidate records, against the C++ decision (which reports the best
challenger) and the decision fields against the Python file.  The reason code
is compared exactly for the Python file and by category for the C++ file.
The survival vectors themselves are reproduced from the telemetry by the
Python dwell model for both files (the two languages fit one model: hazard
(changes + 1) / (transitions + prior)), and for the C++ file also by the C++
formula restated here.

Per-step pricing.  A candidate may carry ``exec_s_by_epoch`` and
``exec_sigma_by_epoch`` (the predicted time and standard deviation of every
remaining step); the Python file's scheduled instances do, together with the
decision that one-shape pricing of the same candidates would have taken
(``single_shape``), so that a reader which ignores the lists is told apart.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from opencrab2.highlevel import candidates as cd
from opencrab2.highlevel import controller as ct

VECTORS = Path(__file__).resolve().parents[2] / "cpp" / "tests" / "vectors"
PY_FILE = VECTORS / "controller_vectors.json"
CPP_FILE = VECTORS / "cpp_controller_vectors.json"
REL = 1e-12
ABS = 1e-12


def _load(path: Path) -> dict:
    assert path.is_file(), f"VECTORS_MISSING: {path}"
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    assert isinstance(doc.get("instances"), list) and doc["instances"], f"VECTORS_EMPTY: {path}"
    return doc


def _close(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=REL, abs_tol=ABS)


def _config(d: dict) -> ct.ControllerConfig:
    known = set(ct.ControllerConfig.__dataclass_fields__)
    assert set(d) <= known, f"unknown controller config keys {sorted(set(d) - known)}"
    return ct.ControllerConfig(**d)


def _state(d: dict) -> cd.SessionState:
    return cd.SessionState(dict(d["incumbent"]), dict(d.get("installed", {})), dict(d.get("stock", {})),
                           int(d["kv_bucket"]), int(d["remaining_epochs"]), str(d["session_id"]),
                           str(d["model_version"]), str(d["contract_hash"]))


def _candidates(items: list, incumbent: dict) -> list:
    out = []
    for c in items:
        out.append(cd.Candidate(
            plan=dict(c["plan"]), install_objects=list(c["install_objects"]), c_install_s=float(c["c_install_s"]),
            c_install_sigma=float(c["c_install_sigma"]), exec_s_per_epoch=float(c["exec_s_per_epoch"]),
            exec_sigma=float(c["exec_sigma"]), material_bytes_per_epoch=int(c.get("material_bytes_per_epoch", 0)),
            mem_peak_bytes=int(c.get("mem_peak_bytes", 0)), feasible=bool(c["feasible"]), reason=str(c.get("reason", "")),
            material_by_type=dict(c.get("material_by_type", {})), is_keep=(dict(c["plan"]) == dict(incumbent)),
            c_wait_s=float(c.get("c_wait_s", 0.0)),
            install_cost_by_object={str(k): float(v) for k, v in dict(c.get("install_cost_by_object", {})).items()},
            exec_s_by_epoch=[float(v) for v in c.get("exec_s_by_epoch", [])],
            exec_sigma_by_epoch=[float(v) for v in c.get("exec_sigma_by_epoch", [])]))
    assert sum(1 for c in out if c.is_keep) == 1, "every instance carries the incumbent exactly once"
    return out


def _replay(inst: dict) -> tuple:
    cfg = _config(inst["config"])
    ctl = ct.StateAwareController(cfg)
    for frame in inst["telemetry"]:
        ctl.observe(frame)
    state = _state(inst["state"])
    cands = _candidates(inst["candidates"], state.incumbent)
    dec = ctl.decide_with_survival(state, cands, int(inst["epoch"]), [float(v) for v in inst["survival"]])
    return ctl, state, cands, dec


def _best_challenger(dec: ct.Decision):
    """Record of the feasible challenger with the largest lower-bound gain (first maximum), or None."""
    best = None
    for rec in dec.candidates:
        if rec["is_keep"] or not rec["feasible"]:
            continue
        if best is None or rec["lower_bound_gain"] > best["lower_bound_gain"]:
            best = rec
    return best


def cpp_survival(telemetry: list, cfg: dict, horizon: int) -> list:
    """The C++ dwell model, restated from cpp/src/highlevel/controller.cpp (DwellModel).

    A change is declared when bandwidth or RTT moves by more than
    env_change_rel relative to the previous sample; with n_c changes over n_s
    transitions (one per frame after the first) h = (n_c + 1) / (n_s +
    dwell_prior), clamped to [0, 1], and s_j = (1 - h)^j; all ones under
    stationary_survival or a non-positive prior.
    """
    prior = 0.0 if cfg.get("stationary_survival", False) else float(cfg["dwell_prior"])
    if prior <= 0.0:
        return [1.0] * horizon
    changes, transitions, last = 0, 0, None
    for f in telemetry:
        bw, rtt = float(f["bandwidth_bps"]), float(f["rtt_s"])
        if last is not None:
            transitions += 1
            rel = max(abs(bw - last[0]) / max(abs(last[0]), 1e-9), abs(rtt - last[1]) / max(abs(last[1]), 1e-12))
            if rel > float(cfg["env_change_rel"]):
                changes += 1
        last = (bw, rtt)
    h = min(1.0, max(0.0, (changes + 1.0) / (transitions + prior)))
    return [(1.0 - h) ** j for j in range(horizon)]


# ---------------------------------------------------------------------------
# controller_vectors.json (Python generated, replayed by C++)
# ---------------------------------------------------------------------------
def test_python_generated_vectors_are_reproduced():
    doc = _load(PY_FILE)
    assert doc["format"] == "opencrab2-controller-vectors-python-v1"
    switching = 0
    for inst in doc["instances"]:
        ctl, state, cands, dec = _replay(inst)
        want = inst["decision"]
        tag = f"instance {inst['id']}"
        assert dec.chosen == want["chosen"], tag
        assert dec.switched == want["switched"], tag
        assert _close(dec.q_keep, want["q_keep"]), tag
        assert _close(dec.q_best, want["q_best"]), tag
        assert _close(dec.lower_bound_gain, want["lower_bound_gain"]), tag
        assert _close(dec.u_diff, want["u_diff"]), tag
        assert dec.reason == want["reason"], tag
        switching += int(dec.switched)
        # the Python dwell model reproduces the stored survival vector from the telemetry
        own = ctl.survival(int(inst["state"]["remaining_epochs"]))
        assert len(own) == len(inst["survival"]), tag
        assert all(_close(a, b) for a, b in zip(own, inst["survival"])), tag
    assert switching == doc["instances_switching"] and switching > 0
    assert switching < len(doc["instances"])           # both outcomes are exercised


def test_python_generated_vectors_match_decide_from_telemetry():
    """decide() (survival fitted by the Python dwell model) equals the replay with the stored vector."""
    doc = _load(PY_FILE)
    for inst in doc["instances"]:
        cfg = _config(inst["config"])
        ctl = ct.StateAwareController(cfg)
        for frame in inst["telemetry"]:
            ctl.observe(frame)
        state = _state(inst["state"])
        dec = ctl.decide(state, _candidates(inst["candidates"], state.incumbent), int(inst["epoch"]))
        want = inst["decision"]
        assert dec.chosen == want["chosen"] and dec.switched == want["switched"], inst["id"]
        assert math.isclose(dec.q_keep, want["q_keep"], rel_tol=1e-9), inst["id"]


def test_python_generated_scheduled_instances_are_priced_step_by_step():
    """Instances with per-step costs: the lists decide, and dropping them gives the recorded one-shape decision."""
    doc = _load(PY_FILE)
    scheduled = [i for i in doc["instances"] if any(c.get("exec_s_by_epoch") for c in i["candidates"])]
    assert len(scheduled) == doc["instances_scheduled"] > 0
    assert all(not any(c.get("exec_s_by_epoch") for c in i["candidates"]) for i in doc["instances"][:30])
    differs = 0
    lengths = set()
    for inst in scheduled:
        H = int(inst["state"]["remaining_epochs"])
        for c in inst["candidates"]:
            assert len(c["exec_s_by_epoch"]) == len(c["exec_sigma_by_epoch"]) > 0, inst["id"]
            assert c["exec_s_by_epoch"][0] == c["exec_s_per_epoch"] and c["exec_sigma_by_epoch"][0] == c["exec_sigma"], inst["id"]
            lengths.add(len(c["exec_s_by_epoch"]) - H)
        ctl = ct.StateAwareController(_config(inst["config"]))
        for frame in inst["telemetry"]:
            ctl.observe(frame)
        state = _state(inst["state"])
        flat = []
        for c in inst["candidates"]:
            c = dict(c)
            c.pop("exec_s_by_epoch")
            c.pop("exec_sigma_by_epoch")
            flat.append(c)
        dec = ctl.decide_with_survival(state, _candidates(flat, state.incumbent), int(inst["epoch"]),
                                       [float(v) for v in inst["survival"]])
        want = inst["single_shape"]
        tag = f"instance {inst['id']} (one shape)"
        assert dec.chosen == want["chosen"] and dec.switched == want["switched"], tag
        assert _close(dec.q_keep, want["q_keep"]), tag
        assert want["differs"] == (want["chosen"] != inst["decision"]["chosen"] or want["switched"] != inst["decision"]["switched"]), tag
        differs += int(want["differs"])
    assert differs == doc["instances_scheduled_single_shape_differs"] > 0
    assert lengths == {-2, 0, 2}                       # lists shorter than, equal to and longer than the horizon


# ---------------------------------------------------------------------------
# cpp_controller_vectors.json (C++ generated, replayed by Python)
# ---------------------------------------------------------------------------
def test_cpp_generated_vectors_are_reproduced():
    doc = _load(CPP_FILE)
    assert doc["format"] == "opencrab2-controller-vectors-v1"
    switching = 0
    seen_reasons = set()
    for inst in doc["instances"]:
        cfg = inst["config"]
        for key in ("epoch_corr", "stationary_survival", "min_residency", "hysteresis_s", "z"):
            assert key in cfg, (inst["id"], key)
        ctl, state, cands, dec = _replay(inst)
        want = inst["decision"]
        tag = f"instance {inst['id']}"
        assert dec.chosen == want["chosen"], tag
        assert dec.switched == want["switched"], tag
        assert _close(dec.q_keep, want["q_keep"]), tag
        # one dwell model in both languages: the Python fit reproduces the C++ survival vector
        own = ctl.survival(len(inst["survival"]))
        assert len(own) == len(inst["survival"]) and all(_close(a, b) for a, b in zip(own, inst["survival"])), tag
        best = _best_challenger(dec)
        if dec.switched:
            switching += 1
            assert _close(dec.q_best, want["q_best"]), tag
            assert _close(dec.lower_bound_gain, want["lower_bound_gain"]), tag
            assert _close(dec.u_diff, want["u_diff"]), tag
            assert best is not None and best["plan"] == want["chosen"], tag
            category = "LOWER_BOUND_GAIN_EXCEEDS_HYSTERESIS"
            assert dec.reason == category, tag
        else:
            # C++ reports the best feasible challenger (0 when there is none) and Qhat(keep) as q_best
            assert _close(want["q_best"], want["q_keep"]), tag
            assert _close(dec.q_keep, want["q_best"]), tag
            if best is None:
                assert want["lower_bound_gain"] == 0.0 and want["u_diff"] == 0.0, tag
                category = "INCUMBENT_OPTIMAL"
            else:
                assert _close(best["lower_bound_gain"], want["lower_bound_gain"]), tag
                assert _close(best["u_diff"], want["u_diff"]), tag
                category = "GAIN_BELOW_UNCERTAINTY_OR_HYSTERESIS" if best["gain"] > 0.0 else "INCUMBENT_OPTIMAL"
        assert want["reason"] == category, tag
        seen_reasons.add(category)
        # the stored survival vector is the documented C++ dwell model on the stored telemetry
        H = len(inst["survival"])
        assert H == max(int(inst["state"]["remaining_epochs"]), 1), tag
        mine = cpp_survival(inst["telemetry"], cfg, H)
        assert all(_close(a, b) for a, b in zip(mine, inst["survival"])), tag
    assert 0 < switching < len(doc["instances"])
    assert "LOWER_BOUND_GAIN_EXCEEDS_HYSTERESIS" in seen_reasons and "INCUMBENT_OPTIMAL" in seen_reasons


def test_cpp_defaults_and_stationary_flag_match_python():
    doc = _load(CPP_FILE)
    default = ct.ControllerConfig()
    assert default.epoch_corr == 0.0 and default.stationary_survival is False
    assert default.hysteresis_s == 0.5 and default.z == 1.0 and default.min_residency == 1
    assert default.dwell_prior == 5.0 and default.env_change_rel == 0.25
    stationary = [i for i in doc["instances"] if i["config"]["stationary_survival"]]
    assert stationary, "the C++ vectors exercise stationary_survival"
    for inst in stationary:
        ctl = ct.StateAwareController(_config(inst["config"]))
        for frame in inst["telemetry"]:
            ctl.observe(frame)
        H = len(inst["survival"])
        assert ctl.survival(H) == [1.0] * H == [float(v) for v in inst["survival"]]
        # with the explicit fixed-capacity assumption the two dwell models coincide: decide() needs no vector
        state = _state(inst["state"])
        dec = ctl.decide(state, _candidates(inst["candidates"], state.incumbent), int(inst["epoch"]))
        assert dec.chosen == inst["decision"]["chosen"] and dec.switched == inst["decision"]["switched"]
        assert _close(dec.q_keep, inst["decision"]["q_keep"])
        assert any("stationary" in a for a in dec.assumptions)


@pytest.mark.parametrize("min_residency", [0, 1, 2, 3])
def test_residency_rule_is_the_cpp_rule(min_residency):
    """A plan chosen at epoch t0 is replaceable at the first epoch t >= t0 + min_residency (both languages)."""
    plans = [{"embedding": "generic", "linear": "generic", "argmax_k": k, "max_k": 2} for k in (2, 4, 8)]

    def cand(plan, exec_s, keep=False):
        return cd.Candidate(dict(plan), [], 0.0, 0.0, float(exec_s), 0.0, 0, 0, True, "", is_keep=keep)

    ctl = ct.StateAwareController(ct.ControllerConfig(hysteresis_s=0.0, min_residency=min_residency,
                                                      stationary_survival=True))
    assert ctl.epochs_since_switch == ct.RESIDENCY_CAP          # a fresh controller is never held
    incumbent = 0
    switched_at = []
    for epoch in range(6):
        nxt = (incumbent + 1) % 3
        state = cd.SessionState(dict(plans[incumbent]), {}, {}, 32, 10 - epoch, "s", "m", "h")
        # the next plan in the cycle is always strictly better than the incumbent
        dec = ctl.decide(state, [cand(plans[incumbent], 5.0, True), cand(plans[nxt], 1.0)], epoch)
        if dec.switched:
            switched_at.append(epoch)
            incumbent = nxt
        else:
            assert dec.reason.startswith("MIN_RESIDENCY_HOLD")
    step = max(1, min_residency)
    assert switched_at == list(range(0, 6, step))
