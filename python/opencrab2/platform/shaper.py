"""Trace-driven shaping, public CPU contention and material refill pacing.

A trace (``opencrab2.experiments.traces.make_trace``) is a dict
``{trace_id, trace_hash, family, split, seed, points}`` whose ``points`` are
tuples ``(elapsed_s, bandwidth_bps, one_way_delay_s, cpu_share, stock_refill_bps)``
(dict points with the same keys are accepted). ``elapsed_s`` is measured from a
wall-clock origin shared by both parties, so events occur at identical elapsed
times for every policy and the controller never reads a future point.

Three exogenous quantities are derived from one trace:

* the link (``bandwidth_bps``, ``one_way_delay_s``) drives the
  :class:`opencrab2.highlevel.channel.WallClockShaper` of both channel ends
  (:func:`build_shaper`), an application-level emulation of a link, not a
  kernel-level replay; records produced with it are labelled
  ``clock_basis = wall-clock-exogenous`` by the session layer;
* ``cpu_share`` (fraction of the host left to the party, 1.0 = idle host) drives
  :class:`CpuContentionLoad`: a controller thread keeps ``(1 - cpu_share) *
  n_cores`` busy worker PROCESSES (one per loaded core, the fractional core
  duty-cycled) so that trace family S4 injects real, public CPU contention;
* ``stock_refill_bps`` (bit/s of material the dealer can deliver, None or 0 =
  unconstrained) drives :class:`MaterialRefillPacer`: the party waits before
  ingesting a bundle until the integral of the refill rate covers the bytes
  delivered so far (the same zero-burst token bucket as the link shaper), which
  is the declared inventory refill of trace family S5.

What is tested (tests/test_launch.py, tests/test_session.py): trace hash
determinism and sensitivity, point normalisation, the link shaper built from a
trace, cpu_share / refill lookups over time, the refill pacer's wait, and the
contention load raising and lowering its worker count along a trace. Design
limits: the busy workers load whole cores of the shared host (both parties and
the dealer feel them), the sleep-based link emulation cannot reproduce packet
loss or slow start, and the pacer meters delivered bytes, not dealer compute.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import threading
import time
from typing import Any, List, Optional, Tuple

from opencrab2.errors import ChannelError
from opencrab2.highlevel.channel import LinkState, WallClockShaper

TRACE_POINT_FIELDS: Tuple[str, ...] = ("elapsed_s", "bandwidth_bps", "one_way_delay_s", "cpu_share", "stock_refill_bps")
CLOCK_BASIS = "wall-clock-exogenous"


def _finite_or_none(v: Any) -> Optional[float]:
    if v is None:
        return None
    f = float(v)
    if not math.isfinite(f) or f <= 0.0:
        return None
    return f


def normalize_points(trace: Any) -> List[Tuple[float, Optional[float], float, float, Optional[float]]]:
    """Trace points as ``(elapsed_s, bandwidth_bps | None, one_way_delay_s, cpu_share, stock_refill_bps | None)``.

    Accepts a trace dict (``points`` key), a list of tuples (3 to 5 entries,
    missing entries default to cpu_share 1.0 and no refill limit) or a list of
    dicts keyed by :data:`TRACE_POINT_FIELDS`. Fails closed on unsorted or
    non-finite times and on cpu_share outside (0, 1].
    """
    pts = trace.get("points") if isinstance(trace, dict) else trace
    if pts is None:
        raise ChannelError("TRACE_PARAMS: trace has no points")
    out = []
    last = -math.inf
    for i, p in enumerate(pts):
        if isinstance(p, dict):
            vals = [p.get(k) for k in TRACE_POINT_FIELDS]
        else:
            vals = list(p) + [None] * (5 - len(p))
            if len(vals) > 5 or len(p) < 3:
                raise ChannelError(f"TRACE_PARAMS: point #{i} needs 3 to 5 entries, got {len(p)}")
        t = float(vals[0])
        if not math.isfinite(t) or t < last:
            raise ChannelError(f"TRACE_PARAMS: point #{i} elapsed_s {vals[0]!r} is not finite and sorted")
        last = t
        bw = _finite_or_none(vals[1])
        delay = float(vals[2]) if vals[2] is not None else 0.0
        if not math.isfinite(delay) or delay < 0.0:
            raise ChannelError(f"TRACE_PARAMS: point #{i} one_way_delay_s {vals[2]!r} invalid")
        cpu = 1.0 if vals[3] is None else float(vals[3])
        if not (0.0 < cpu <= 1.0):
            raise ChannelError(f"TRACE_PARAMS: point #{i} cpu_share {vals[3]!r} outside (0, 1]")
        refill = _finite_or_none(vals[4])
        out.append((t, bw, delay, cpu, refill))
    if not out:
        raise ChannelError("TRACE_PARAMS: trace must contain at least one point")
    return out


def trace_hash(trace: Any) -> str:
    """The trace's own ``trace_hash`` when ``experiments.traces`` computed one, else a SHA-256 over the canonical JSON
    of ``{trace_id, family, split, seed, points}``; ``"none"`` for an unshaped run."""
    if trace is None:
        return "none"
    if isinstance(trace, dict) and isinstance(trace.get("trace_hash"), str) and trace["trace_hash"]:
        return str(trace["trace_hash"])
    pts = normalize_points(trace)
    body = {"trace_id": str(trace.get("trace_id", "")) if isinstance(trace, dict) else "",
            "family": str(trace.get("family", "")) if isinstance(trace, dict) else "",
            "split": str(trace.get("split", "")) if isinstance(trace, dict) else "",
            "seed": trace.get("seed") if isinstance(trace, dict) else None,
            "points": [list(p) for p in pts]}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def unshaped_trace(trace_id: str = "unshaped", duration_s: float = 3600.0) -> dict:
    """The trivial exogenous trace: unlimited link, no delay, idle host, unconstrained refill."""
    return {"trace_id": trace_id, "family": "S0", "split": "test", "seed": 0,
            "points": [(0.0, None, 0.0, 1.0, None), (float(duration_s), None, 0.0, 1.0, None)]}


def build_shaper(trace: Any, origin_wall_s: float) -> WallClockShaper:
    """Link shaper (bandwidth and one-way delay) of ``trace`` with the shared wall-clock origin."""
    pts = normalize_points(trace)
    return WallClockShaper([(t, LinkState(bw, delay)) for t, bw, delay, _cpu, _refill in pts], float(origin_wall_s))


class TraceShaper:
    """All exogenous views of one trace: link shaper, cpu_share(t), refill(t) and the trace identity."""

    def __init__(self, trace: Any, origin_wall_s: float):
        self.points = normalize_points(trace)
        self.origin_wall_s = float(origin_wall_s)
        self.trace_id = str(trace.get("trace_id", "trace")) if isinstance(trace, dict) else "trace"
        self.trace_hash = trace_hash(trace)
        self.family = str(trace.get("family", "")) if isinstance(trace, dict) else ""
        self.link = WallClockShaper([(t, LinkState(bw, d)) for t, bw, d, _c, _r in self.points], self.origin_wall_s)
        self._starts = [p[0] for p in self.points]
        refill_pts = [(t, LinkState(r, 0.0)) for t, _bw, _d, _c, r in self.points]
        self.refill: Optional[WallClockShaper] = None
        if any(p[4] is not None for p in self.points):
            # a segment without a refill limit is unshaped (None bandwidth drains instantly)
            self.refill = WallClockShaper(refill_pts, self.origin_wall_s)

    def _index(self, wall_s: float) -> int:
        e = float(wall_s) - self.origin_wall_s
        idx = 0
        for i, s in enumerate(self._starts):
            if s <= e:
                idx = i
            else:
                break
        return idx

    def cpu_share_at(self, wall_s: float) -> float:
        return float(self.points[self._index(wall_s)][3])

    def refill_bps_at(self, wall_s: float) -> Optional[float]:
        return self.points[self._index(wall_s)][4]

    def link_state_at(self, wall_s: float) -> LinkState:
        return self.link.state_at(wall_s)

    @property
    def shaped(self) -> bool:
        """True when any segment limits the link, loads the host or limits the refill."""
        return any(bw is not None or d > 0.0 or c < 1.0 or r is not None for _t, bw, d, c, r in self.points)

    @property
    def has_cpu_load(self) -> bool:
        return any(c < 1.0 for _t, _bw, _d, c, _r in self.points)

    def describe(self) -> dict:
        return {"trace_id": self.trace_id, "trace_hash": self.trace_hash, "family": self.family,
                "origin_wall_s": self.origin_wall_s, "points": len(self.points), "shaped": self.shaped,
                "cpu_load": self.has_cpu_load, "refill_limited": self.refill is not None}


# ---------------------------------------------------------------------------
# material refill pacer (S5)
# ---------------------------------------------------------------------------
class MaterialRefillPacer:
    """Zero-burst token bucket on delivered material bytes, rate = stock_refill_bps of the trace.

    ``wait_for(nbytes)`` sleeps until the integral of the refill rate since the
    previous delivery covers ``nbytes``; it returns the seconds waited. With no
    refill limit in the trace the pacer is a no-op.
    """

    def __init__(self, shaper: Optional[TraceShaper]):
        self.shaper = shaper
        self._free_at = 0.0
        self.waited_s = 0.0
        self.delivered_bytes = 0
        self._lock = threading.Lock()

    @property
    def active(self) -> bool:
        return self.shaper is not None and self.shaper.refill is not None

    def wait_for(self, nbytes: int) -> float:
        nbytes = int(nbytes)
        with self._lock:
            self.delivered_bytes += nbytes
            if not self.active or nbytes <= 0:
                return 0.0
            now = time.time()
            start = max(now, self._free_at)
            ser = self.shaper.refill.serialization_time(nbytes, start)   # type: ignore[union-attr]
            self._free_at = start + ser
            wait = self._free_at - now
        if wait > 0.0:
            time.sleep(wait)
            with self._lock:
                self.waited_s += wait
            return wait
        return 0.0


# ---------------------------------------------------------------------------
# public CPU contention (S4)
# ---------------------------------------------------------------------------
def _busy_worker(stop_event: Any, duty: Any, slice_s: float = 0.01) -> None:   # pragma: no cover (child process)
    """Busy loop with a duty cycle read from the shared ``duty`` value (0 = sleep, 1 = fully busy)."""
    x = 1.0
    while not stop_event.is_set():
        d = float(duty.value)
        if d <= 0.0:
            time.sleep(slice_s)
            continue
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < slice_s * d:
            x = x * 1.0000001 + 1e-9
        rest = slice_s * (1.0 - d)
        if rest > 0.0:
            time.sleep(rest)


class CpuContentionLoad:
    """Keeps ``(1 - cpu_share(t)) * n_cores`` cores busy following the trace (public load, S4).

    One spawned worker process per core that is ever loaded; a controller
    thread updates the per-worker duty cycle every ``tick_s`` from the trace.
    ``start()`` is a no-op when the trace never loads the host.
    """

    def __init__(self, shaper: TraceShaper, n_cores: Optional[int] = None, tick_s: float = 0.05):
        self.shaper = shaper
        self.n_cores = int(n_cores) if n_cores is not None else max(1, int(os.cpu_count() or 1))
        self.tick_s = float(tick_s)
        self._procs: List[Any] = []
        self._duties: List[Any] = []
        self._stop_event: Any = None
        self._thread: Optional[threading.Thread] = None
        self._thread_stop = threading.Event()
        self.max_workers = 0
        self.history: List[Tuple[float, int]] = []     # (wall_s, loaded cores x 100)

    def target_load_cores(self, wall_s: float) -> float:
        return (1.0 - self.shaper.cpu_share_at(wall_s)) * self.n_cores

    def start(self) -> int:
        """Spawn the workers (returns their count; 0 when the trace never loads the host)."""
        if self._procs:
            return len(self._procs)
        peak = max((1.0 - c) * self.n_cores for _t, _bw, _d, c, _r in self.shaper.points)
        self.max_workers = int(math.ceil(peak - 1e-9)) if peak > 0 else 0
        if self.max_workers == 0:
            return 0
        ctx = multiprocessing.get_context("spawn")
        self._stop_event = ctx.Event()
        for _ in range(self.max_workers):
            duty = ctx.Value("d", 0.0)
            p = ctx.Process(target=_busy_worker, args=(self._stop_event, duty), daemon=True)
            p.start()
            self._procs.append(p)
            self._duties.append(duty)
        self._thread_stop.clear()
        self._thread = threading.Thread(target=self._control_loop, name="cpu-contention", daemon=True)
        self._thread.start()
        return self.max_workers

    def _apply(self, wall_s: float) -> None:
        load = max(0.0, min(float(self.max_workers), self.target_load_cores(wall_s)))
        full = int(math.floor(load + 1e-9))
        frac = load - full
        for i, duty in enumerate(self._duties):
            duty.value = 1.0 if i < full else (frac if i == full and frac > 1e-6 else 0.0)
        self.history.append((wall_s, int(round(load * 100))))

    def _control_loop(self) -> None:
        while not self._thread_stop.is_set():
            self._apply(time.time())
            self._thread_stop.wait(self.tick_s)

    def current_duties(self) -> List[float]:
        return [float(d.value) for d in self._duties]

    def stop(self) -> None:
        self._thread_stop.set()
        if self._thread is not None:
            self._thread.join(2.0)
            self._thread = None
        if self._stop_event is not None:
            self._stop_event.set()
        for p in self._procs:
            p.join(2.0)
            if p.is_alive():
                p.terminate()
        self._procs = []
        self._duties = []

    def __enter__(self) -> "CpuContentionLoad":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()


__all__ = [
    "TRACE_POINT_FIELDS", "CLOCK_BASIS", "normalize_points", "trace_hash", "unshaped_trace", "build_shaper",
    "TraceShaper", "MaterialRefillPacer", "CpuContentionLoad",
]
