"""Exogenous environment traces for the evaluation (paper, Section 7.1, models and traces).

A trace is a continuous timeline in wall seconds since the shared origin of a
paired run: ``points`` is a sorted list ``[elapsed_s, bandwidth_bps,
one_way_delay_s, cpu_share, stock_refill_bps]`` and the state in force at
time ``t`` is the last point with ``elapsed_s <= t`` (piecewise constant;
ramps are sampled every ``RAMP_STEP_S`` seconds so that a transfer spanning a
change pays the integral of the rates). Every trace carries its seed,
the generator version, the split label (odd seed = ``tune``, even seed =
``test``, fixed before any run) and ``trace_hash = sha256`` of the canonical JSON
of the points. The four network anchors are LAN (1 Gbit/s, 0.25 ms one-way),
WAN (100 Mbit/s, 20 ms), WAN-c (20 Mbit/s, 40 ms) and WAN-mid (50 Mbit/s,
10 ms).

Families (each with at least three amplitude/duration variants):

* S0 stable links: one anchor for the whole duration.
* S1 bandwidth ramp: linear bandwidth ramp between two anchors over the whole
  duration at the start anchor's delay.
* S2 RTT-only variation: fixed bandwidth (100 Mbit/s), one-way delay walks
  through {0.25, 10, 40, 100} ms in equal segments (variant fixes the order).
* S3 multiple changes and spikes: 2 to 6 regime changes between anchors plus
  1 to 3 short spikes (0.5 to 3 s, bandwidth divided by 10, delay + 50 ms).
* S4 compute contention: public ``cpu_share`` in {1.0, 0.5, 0.25} during a
  contention episode on a fixed link.
* S5 material inventory: declared ``stock_refill_bps`` (bytes per second of
  dealer delivery) and ``initial_stock_bytes`` on a fixed link.
* S6 KV/memory growth: length bucket in {64, 256, 1024} with a declared
  per-party memory limit.
* S7 Web cancel/resume: ``events`` = ``[[elapsed_s, kind]]`` with kinds
  ``cancel``, ``resume``, ``disconnect``; the link is fixed.
* S8 installation state: identical link, different ``initial_installed``
  object lists (cold, warm embedding, warm all).

The variants are synthetic and labelled as such in ``provenance``; they are
controlled experiments, not captured mobile traces.

What is tested (tests/test_traces.py): determinism and hash stability for a
fixed seed, distinct hashes for distinct seeds, at least three variants per
family, sorted timelines inside the duration, the S1 monotone ramp, S2
varying only the delay inside the admitted set, S3 change and spike counts
inside the declared ranges, S4 cpu_share values, S7 events, S8 installed
lists, the split parity rule and compatibility with
``channel.WallClockShaper.from_points``. Design limit: the anchors are
declared link parameters, not measured capacities; the shaper applying them
is an application-level emulation, which the session layer labels
``clock_basis = wall-clock-exogenous``.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

GENERATOR_VERSION = "opencrab2-traces-v2.0"
RAMP_STEP_S = 1.0

# network anchors: name -> (bandwidth_bps, one_way_delay_s)
ANCHORS: Dict[str, Tuple[float, float]] = {
    "LAN": (1.0e9, 0.25e-3),
    "WAN": (100.0e6, 20.0e-3),
    "WAN-c": (20.0e6, 40.0e-3),
    "WAN-mid": (50.0e6, 10.0e-3),
}
ANCHOR_NAMES: Tuple[str, ...] = tuple(ANCHORS)
S2_DELAYS_MS: Tuple[float, ...] = (0.25, 10.0, 40.0, 100.0)
S2_BANDWIDTH_BPS = 100.0e6
S3_SPIKE_BW_DIVISOR = 10.0
S3_SPIKE_EXTRA_DELAY_S = 50.0e-3
S4_CPU_LEVELS: Tuple[float, ...] = (1.0, 0.5, 0.25)
S6_LENGTH_BUCKETS: Tuple[int, ...] = (64, 256, 1024)
GPT2_LAYER_OBJECTS: Tuple[str, ...] = tuple(f"h{li}.{s}" for li in range(12) for s in ("qkv", "out", "fc", "proj"))

# family -> variant -> parameters (frozen before any run)
FAMILIES: Dict[str, Dict[str, dict]] = {
    "S0": {
        "stable-LAN": {"anchor": "LAN"},
        "stable-WAN": {"anchor": "WAN"},
        "stable-WAN-c": {"anchor": "WAN-c"},
        "stable-WAN-mid": {"anchor": "WAN-mid"},
    },
    "S1": {
        "ramp-WAN-c-to-LAN": {"start": "WAN-c", "end": "LAN"},
        "ramp-LAN-to-WAN-c": {"start": "LAN", "end": "WAN-c"},
        "ramp-WAN-to-WAN-mid": {"start": "WAN", "end": "WAN-mid"},
        "ramp-WAN-mid-to-WAN": {"start": "WAN-mid", "end": "WAN"},
    },
    "S2": {
        "rtt-step-up": {"order": "up"},
        "rtt-step-down": {"order": "down"},
        "rtt-alternate": {"order": "alternate"},
        "rtt-shuffle": {"order": "shuffle"},
    },
    "S3": {
        "two-changes": {"n_changes": 2, "n_spikes": 1},
        "four-changes": {"n_changes": 4, "n_spikes": 2},
        "six-changes": {"n_changes": 6, "n_spikes": 3},
    },
    "S4": {
        "cpu-1.0": {"cpu_share": 1.0, "anchor": "WAN"},
        "cpu-0.5": {"cpu_share": 0.5, "anchor": "WAN"},
        "cpu-0.25": {"cpu_share": 0.25, "anchor": "WAN"},
    },
    # refill in bytes per second of delivered material; one GPT-2 decode epoch consumes about 2.9e8 bytes per
    # party under the default contract, so the three levels are below, near and above the per-epoch need
    "S5": {
        "refill-starved": {"stock_refill_bps": 2.0e6, "initial_stock_bytes": 256 * 2 ** 20, "anchor": "WAN"},
        "refill-low": {"stock_refill_bps": 1.0e7, "initial_stock_bytes": 1024 * 2 ** 20, "anchor": "WAN"},
        "refill-ample": {"stock_refill_bps": 5.0e7, "initial_stock_bytes": 4096 * 2 ** 20, "anchor": "WAN"},
    },
    "S6": {
        "len-64": {"length_bucket": 64, "mem_limit_bytes": 2 * 2 ** 30, "anchor": "WAN"},
        "len-256": {"length_bucket": 256, "mem_limit_bytes": 4 * 2 ** 30, "anchor": "WAN"},
        "len-1024": {"length_bucket": 1024, "mem_limit_bytes": 8 * 2 ** 30, "anchor": "WAN"},
    },
    "S7": {
        "cancel-early-resume": {"events": [[0.2, "cancel"], [0.3, "resume"]], "anchor": "WAN"},
        "cancel-mid-resume": {"events": [[0.5, "cancel"], [0.65, "resume"]], "anchor": "WAN"},
        "cancel-no-resume": {"events": [[0.5, "cancel"]], "anchor": "WAN"},
        "disconnect-reconnect": {"events": [[0.4, "disconnect"], [0.45, "resume"]], "anchor": "WAN"},
    },
    "S8": {
        "cold": {"initial_installed": [], "anchor": "WAN"},
        "warm-embedding": {"initial_installed": ["wte", "lm_head"], "anchor": "WAN"},
        "warm-all": {"initial_installed": ["wte", "lm_head"] + list(GPT2_LAYER_OBJECTS), "anchor": "WAN"},
        # the cold and warm pair of the measured diag4 matrix: the same link state on the LAN anchor
        "cold-LAN": {"initial_installed": [], "anchor": "LAN"},
        "warm-all-LAN": {"initial_installed": ["wte", "lm_head"] + list(GPT2_LAYER_OBJECTS), "anchor": "LAN"},
    },
}
FAMILY_NAMES: Tuple[str, ...] = tuple(FAMILIES)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _check(family: str, variant: str, seed: int, duration_s: float) -> dict:
    if family not in FAMILIES:
        raise ValueError(f"TRACE_UNKNOWN_FAMILY: {family!r} is not one of {list(FAMILIES)}")
    if variant not in FAMILIES[family]:
        raise ValueError(f"TRACE_UNKNOWN_VARIANT: {variant!r} is not a variant of {family} ({list(FAMILIES[family])})")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or int(seed) < 0:
        raise ValueError(f"TRACE_PARAM: seed must be a non-negative int, got {seed!r}")
    d = float(duration_s)
    if not math.isfinite(d) or d <= 0.0:
        raise ValueError(f"TRACE_PARAM: duration_s must be positive and finite, got {duration_s!r}")
    return dict(FAMILIES[family][variant])


def _rng(family: str, variant: str, seed: int) -> np.random.Generator:
    digest = hashlib.sha256(f"{GENERATOR_VERSION}|{family}|{variant}|{int(seed)}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def split_for_seed(seed: int) -> str:
    """Preregistered rule: odd seeds tune thresholds, even seeds are test traces."""
    return "tune" if int(seed) % 2 == 1 else "test"


def _pt(t: float, bw: float, delay: float, cpu: float = 1.0, refill: float = 0.0) -> List[float]:
    return [round(float(t), 6), float(bw), float(delay), float(cpu), float(refill)]


def points_hash(points: Sequence[Sequence[float]]) -> str:
    text = json.dumps([list(map(float, p)) for p in points], separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def trace_id_of(family: str, variant: str, seed: int, duration_s: float) -> str:
    return f"{family}:{variant}:seed{int(seed)}:d{float(duration_s):g}"


# ---------------------------------------------------------------------------
# family generators (each returns points and family-specific metadata)
# ---------------------------------------------------------------------------
def _gen_s0(p: dict, rng: np.random.Generator, D: float) -> Tuple[List[List[float]], dict]:
    bw, dl = ANCHORS[p["anchor"]]
    return [_pt(0.0, bw, dl)], {"anchor": p["anchor"]}


def _gen_s1(p: dict, rng: np.random.Generator, D: float) -> Tuple[List[List[float]], dict]:
    bw0, dl0 = ANCHORS[p["start"]]
    bw1, _dl1 = ANCHORS[p["end"]]
    n_steps = max(2, int(math.ceil(D / RAMP_STEP_S)))
    pts = []
    for i in range(n_steps):
        t = i * D / n_steps
        frac = t / D
        pts.append(_pt(t, bw0 + (bw1 - bw0) * frac, dl0))
    return pts, {"start": p["start"], "end": p["end"], "ramp": "linear", "step_s": D / n_steps}


def _gen_s2(p: dict, rng: np.random.Generator, D: float) -> Tuple[List[List[float]], dict]:
    delays = list(S2_DELAYS_MS)
    order = p["order"]
    if order == "up":
        seq = delays
    elif order == "down":
        seq = delays[::-1]
    elif order == "alternate":
        seq = [delays[0], delays[3], delays[1], delays[2]]
    else:
        seq = [delays[i] for i in rng.permutation(len(delays))]
    seg = D / len(seq)
    pts = [_pt(i * seg, S2_BANDWIDTH_BPS, ms * 1e-3) for i, ms in enumerate(seq)]
    return pts, {"bandwidth_bps": S2_BANDWIDTH_BPS, "delay_order_ms": seq, "segment_s": seg}


def _gen_s3(p: dict, rng: np.random.Generator, D: float) -> Tuple[List[List[float]], dict]:
    n_changes = int(p["n_changes"])
    n_spikes = int(p["n_spikes"])
    if not (2 <= n_changes <= 6) or not (1 <= n_spikes <= 3):
        raise ValueError("TRACE_PARAM: S3 needs 2 <= n_changes <= 6 and 1 <= n_spikes <= 3")
    names = list(ANCHOR_NAMES)
    cur = names[int(rng.integers(len(names)))]
    regimes = [cur]
    for _ in range(n_changes):
        choices = [a for a in names if a != cur]
        cur = choices[int(rng.integers(len(choices)))]
        regimes.append(cur)
    # change times: sorted uniform draws inside (0.1 D, 0.9 D), at least 2 s apart when possible
    times = np.sort(rng.uniform(0.1 * D, 0.9 * D, size=n_changes))
    segments: List[Tuple[float, float, float]] = []
    t_prev = 0.0
    for i, name in enumerate(regimes):
        t_end = float(times[i]) if i < n_changes else D
        bw, dl = ANCHORS[name]
        segments.append((t_prev, bw, dl))
        t_prev = t_end
    starts = [s[0] for s in segments]
    spikes = []
    for _ in range(n_spikes):
        dur = float(rng.uniform(0.5, 3.0))
        t0 = float(rng.uniform(0.05 * D, max(0.05 * D + 1e-3, 0.95 * D - dur)))
        spikes.append((t0, t0 + dur))
    spikes.sort()

    def state_at(t: float) -> Tuple[float, float]:
        idx = 0
        for i, s in enumerate(starts):
            if s <= t:
                idx = i
        return segments[idx][1], segments[idx][2]

    breakpoints = sorted(set([0.0] + starts + [b for sp in spikes for b in sp]))
    pts: List[List[float]] = []
    for t in breakpoints:
        if t >= D:
            continue
        bw, dl = state_at(t)
        in_spike = any(a <= t < b for a, b in spikes)
        if in_spike:
            bw, dl = bw / S3_SPIKE_BW_DIVISOR, dl + S3_SPIKE_EXTRA_DELAY_S
        pts.append(_pt(t, bw, dl))
    meta = {"regimes": regimes, "change_times_s": [float(x) for x in times],
            "spikes_s": [[float(a), float(b)] for a, b in spikes], "n_changes": n_changes, "n_spikes": n_spikes}
    return pts, meta


def _gen_s4(p: dict, rng: np.random.Generator, D: float) -> Tuple[List[List[float]], dict]:
    bw, dl = ANCHORS[p["anchor"]]
    level = float(p["cpu_share"])
    if level not in S4_CPU_LEVELS:
        raise ValueError(f"TRACE_PARAM: cpu_share must be one of {S4_CPU_LEVELS}")
    start = float(rng.uniform(0.2, 0.5)) * D
    length = float(rng.uniform(0.3, 0.5)) * D
    end = min(D, start + length)
    pts = [_pt(0.0, bw, dl, 1.0)]
    if level < 1.0:
        pts.append(_pt(start, bw, dl, level))
        if end < D:
            pts.append(_pt(end, bw, dl, 1.0))
    return pts, {"anchor": p["anchor"], "cpu_share": level, "episode_s": [start, end]}


def _gen_s5(p: dict, rng: np.random.Generator, D: float) -> Tuple[List[List[float]], dict]:
    bw, dl = ANCHORS[p["anchor"]]
    refill = float(p["stock_refill_bps"])
    return [_pt(0.0, bw, dl, 1.0, refill)], {"anchor": p["anchor"], "stock_refill_bps": refill,
                                               "initial_stock_bytes": int(p["initial_stock_bytes"])}


def _gen_s6(p: dict, rng: np.random.Generator, D: float) -> Tuple[List[List[float]], dict]:
    bw, dl = ANCHORS[p["anchor"]]
    lb = int(p["length_bucket"])
    if lb not in S6_LENGTH_BUCKETS:
        raise ValueError(f"TRACE_PARAM: length_bucket must be one of {S6_LENGTH_BUCKETS}")
    return [_pt(0.0, bw, dl)], {"anchor": p["anchor"], "length_bucket": lb, "mem_limit_bytes": int(p["mem_limit_bytes"])}


def _gen_s7(p: dict, rng: np.random.Generator, D: float) -> Tuple[List[List[float]], dict]:
    bw, dl = ANCHORS[p["anchor"]]
    jitter = float(rng.uniform(-0.02, 0.02)) * D
    events = [[round(max(0.0, min(D, float(frac) * D + jitter)), 6), str(kind)] for frac, kind in p["events"]]
    return [_pt(0.0, bw, dl)], {"anchor": p["anchor"], "events": events}


def _gen_s8(p: dict, rng: np.random.Generator, D: float) -> Tuple[List[List[float]], dict]:
    bw, dl = ANCHORS[p["anchor"]]
    return [_pt(0.0, bw, dl)], {"anchor": p["anchor"], "initial_installed": list(p["initial_installed"])}


_GENERATORS = {"S0": _gen_s0, "S1": _gen_s1, "S2": _gen_s2, "S3": _gen_s3, "S4": _gen_s4,
               "S5": _gen_s5, "S6": _gen_s6, "S7": _gen_s7, "S8": _gen_s8}


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def make_trace(family: str, variant: str, seed: int, duration_s: float) -> dict:
    """Deterministic trace ``{trace_id, trace_hash, family, variant, seed, split, duration_s, points, meta, ...}``.

    ``points`` rows are ``[elapsed_s, bandwidth_bps, one_way_delay_s, cpu_share, stock_refill_bps]``.
    """
    params = _check(family, variant, seed, duration_s)
    D = float(duration_s)
    rng = _rng(family, variant, int(seed))
    pts, meta = _GENERATORS[family](params, rng, D)
    pts = [p for p in pts if p[0] < D]
    pts.sort(key=lambda p: p[0])
    if not pts or pts[0][0] != 0.0:
        raise ValueError("TRACE_INTERNAL: a trace must start at elapsed 0")
    for a, b in zip(pts, pts[1:]):
        if b[0] < a[0]:
            raise ValueError("TRACE_INTERNAL: points must be sorted")
    return {
        "trace_id": trace_id_of(family, variant, int(seed), D),
        "trace_hash": points_hash(pts),
        "family": family,
        "variant": variant,
        "seed": int(seed),
        "split": split_for_seed(int(seed)),
        "duration_s": D,
        "generator_version": GENERATOR_VERSION,
        "provenance": {"kind": "synthetic", "generator": GENERATOR_VERSION, "seed": int(seed),
                       "anchors": {k: list(v) for k, v in ANCHORS.items()},
                       "note": "controlled synthetic trace; not a captured mobile trace"},
        "columns": ["elapsed_s", "bandwidth_bps", "one_way_delay_s", "cpu_share", "stock_refill_bps"],
        "points": pts,
        "meta": meta,
        "events": list(meta.get("events", [])),
        "initial_installed": list(meta.get("initial_installed", [])),
        "length_bucket": int(meta.get("length_bucket", 0)),
        "mem_limit_bytes": meta.get("mem_limit_bytes"),
        "initial_stock_bytes": meta.get("initial_stock_bytes"),
    }


def variants(family: str) -> Tuple[str, ...]:
    if family not in FAMILIES:
        raise ValueError(f"TRACE_UNKNOWN_FAMILY: {family!r}")
    return tuple(FAMILIES[family])


def all_specs(families: Optional[Sequence[str]] = None) -> List[Tuple[str, str]]:
    """Every (family, variant) pair in catalog order."""
    fams = list(families) if families is not None else list(FAMILY_NAMES)
    return [(f, v) for f in fams for v in variants(f)]


def state_at(trace: Mapping[str, Any], elapsed_s: float) -> dict:
    """Environment in force at ``elapsed_s``: bandwidth_bps, one_way_delay_s, cpu_share, stock_refill_bps."""
    pts = trace["points"]
    cur = pts[0]
    for p in pts:
        if p[0] <= elapsed_s:
            cur = p
        else:
            break
    return {"bandwidth_bps": float(cur[1]), "one_way_delay_s": float(cur[2]), "cpu_share": float(cur[3]),
            "stock_refill_bps": float(cur[4])}


def shaper_points(trace: Mapping[str, Any]) -> List[Tuple[float, float, float]]:
    """``(elapsed_s, bandwidth_bps, one_way_delay_s)`` tuples for ``WallClockShaper.from_points``."""
    return [(float(p[0]), float(p[1]), float(p[2])) for p in trace["points"]]


def events_between(trace: Mapping[str, Any], t0: float, t1: float) -> List[Tuple[float, str]]:
    """S7 events with ``t0 <= elapsed < t1`` in time order."""
    return [(float(t), str(k)) for t, k in trace.get("events", []) if t0 <= float(t) < t1]


def write_trace(trace: Mapping[str, Any], directory: Any) -> Path:
    """Write ``<directory>/<trace_id with ':' -> '_'>.json``; returns the path."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    p = d / (str(trace["trace_id"]).replace(":", "_") + ".json")
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(dict(trace), fh, indent=1, sort_keys=True)
    return p


def load_trace(path: Any) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        tr = json.load(fh)
    if points_hash(tr["points"]) != tr["trace_hash"]:
        raise ValueError(f"TRACE_HASH_MISMATCH: {path} points do not match trace_hash")
    return tr


def catalog(duration_s: float, seeds: Sequence[int], families: Optional[Sequence[str]] = None) -> List[dict]:
    """Every trace of the matrix (family x variant x seed) at one duration."""
    return [make_trace(f, v, s, duration_s) for f, v in all_specs(families) for s in seeds]


__all__ = [
    "GENERATOR_VERSION", "RAMP_STEP_S", "ANCHORS", "ANCHOR_NAMES", "S2_DELAYS_MS", "S2_BANDWIDTH_BPS",
    "S4_CPU_LEVELS", "S6_LENGTH_BUCKETS", "FAMILIES", "FAMILY_NAMES", "make_trace", "variants", "all_specs",
    "state_at", "shaper_points", "events_between", "write_trace", "load_trace", "catalog", "split_for_seed",
    "points_hash", "trace_id_of",
]
