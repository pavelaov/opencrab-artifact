"""Public telemetry frames and the fixed-size channel probe (paper, Section 4.2, objective and decision).

A :class:`TelemetryFrame` is one sensing record of the PUBLIC environment
(bandwidth, RTT, CPU load) at a fixed epoch. Frames are HMAC-SHA256 signed
over their canonical JSON (sorted keys, no spaces, the same canonical form as
``lowlevel.hmac_auth.canonical_json``) and carry a sequence number. Both
parties feed identical verified frames to identical deterministic controllers,
so a frame that is replayed, out of order, non-monotone in time, non-finite,
out of range or wrongly tagged is rejected (``TelemetryError`` with a stable
code) and never reaches a controller.

The probe is fixed-size by construction: ``samples`` times a ping/pong pair of
16-byte payloads followed by one ``burst``-byte transfer answered by a 16-byte
acknowledgement, at fixed epochs. Its message sizes depend on nothing secret.

Estimator (:func:`probe_estimate`). The round-trip time is the median of the
ping times. The bandwidth is one pooled figure per probe, ``burst * 8 /
(median(burst times) - min(ping times))`` with the difference floored at
``PROBE_MIN_NET_S``. The smallest ping is the least delayed observation of the
round trip, so one late ping (a scheduler or shaper tick) cannot cancel the
burst time of its own sample, which a median of per-sample ratios lets happen
when the burst time is close to the timer resolution. This narrows the upper
tail of the estimate; it is not a bound.

Integrity note (not a security proof): the HMAC tag binds a frame to the
shared telemetry key; it protects the two honest parties against a corrupted
transport, not against a corrupted party that holds the key.

What is tested (tests/test_telemetry.py): sign/verify round trip, rejection
codes for every malformed case, strict sequencing from the configured first
seq, dict round trip, probe over a loopback channel (fixed sizes, plausible
RTT with a shaped channel), the pooled estimator on fixed samples (one late
ping, the floor, the shared vector file), and the signing parity with
``hmac_auth.tag`` when that module is importable. Design limit: wall-clock
telemetry cannot be proven secret-independent by tests; the primary profile
declares fixed probes and public load as the allowed observations.
"""

from __future__ import annotations

import dataclasses
import hashlib
import hmac
import json
import math
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from opencrab2.errors import TelemetryError

PROBE_PING_BYTES = 16
PROBE_ACK_BYTES = 16
PROBE_MIN_NET_S = 1e-7      # floor of the burst time net of the round trip (seconds)


def canonical_json(obj: Any) -> bytes:
    """Canonical JSON bytes (sorted keys, no spaces, UTF-8, non-finite floats refused).

    Byte-identical to ``lowlevel.hmac_auth.canonical_json`` so that
    ``sign_frame`` equals ``hmac_auth.tag(key, frame.to_dict())`` for every
    frame, including non-ASCII ``source`` strings.
    """
    try:
        text = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except ValueError as exc:
        raise TelemetryError(f"TELEMETRY_NON_FINITE: frame is not canonicalisable ({exc})") from None
    return text.encode("utf-8")


@dataclass
class TelemetryFrame:
    """One public sensing record. ``seq`` increases by exactly one per session."""
    seq: int
    t: float
    bandwidth_bps: float
    rtt_s: float
    epoch: int
    cpu_load: float = 0.0
    source: str = "probe"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seq": int(self.seq), "t": float(self.t), "bandwidth_bps": float(self.bandwidth_bps),
            "rtt_s": float(self.rtt_s), "epoch": int(self.epoch), "cpu_load": float(self.cpu_load),
            "source": str(self.source),
        }

    @staticmethod
    def from_dict(d: Any) -> "TelemetryFrame":
        """Strictly typed decode of a frame received from the peer."""
        if not isinstance(d, dict):
            raise TelemetryError("TELEMETRY_MALFORMED: frame must be a dict")
        names = {f.name for f in dataclasses.fields(TelemetryFrame)}
        if set(d) != names:
            raise TelemetryError(f"TELEMETRY_MALFORMED: frame keys {sorted(d)} != {sorted(names)}")
        for k in ("seq", "epoch"):
            v = d[k]
            if isinstance(v, bool) or not isinstance(v, int):
                raise TelemetryError(f"TELEMETRY_MALFORMED: {k} must be an int")
        for k in ("t", "bandwidth_bps", "rtt_s", "cpu_load"):
            v = d[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise TelemetryError(f"TELEMETRY_MALFORMED: {k} must be a number")
        if not isinstance(d["source"], str):
            raise TelemetryError("TELEMETRY_MALFORMED: source must be a str")
        return TelemetryFrame(seq=int(d["seq"]), t=float(d["t"]), bandwidth_bps=float(d["bandwidth_bps"]),
                              rtt_s=float(d["rtt_s"]), epoch=int(d["epoch"]), cpu_load=float(d["cpu_load"]),
                              source=str(d["source"]))

    def canonical(self) -> bytes:
        return canonical_json(self.to_dict())

    def digest(self) -> str:
        """SHA-256 hex of the canonical frame (for decision digests and logs)."""
        return hashlib.sha256(self.canonical()).hexdigest()


def sign_frame(frame: TelemetryFrame, key: bytes) -> str:
    """HMAC-SHA256 hex tag over the canonical JSON of ``frame``."""
    if not isinstance(key, (bytes, bytearray)) or len(key) < 16:
        raise TelemetryError("TELEMETRY_KEY: key must be at least 16 bytes")
    return hmac.new(bytes(key), frame.canonical(), hashlib.sha256).hexdigest()


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


class TelemetryVerifier:
    """Accepts frames strictly in sequence; every rejection raises ``TelemetryError``.

    Rejection codes: ``TELEMETRY_BAD_TAG``, ``TELEMETRY_BAD_SEQ``,
    ``TELEMETRY_REPLAY`` (seq already consumed), ``TELEMETRY_OUT_OF_ORDER``
    (gap or wrong first seq), ``TELEMETRY_NON_MONOTONE_TIME``,
    ``TELEMETRY_NON_FINITE``, ``TELEMETRY_OUT_OF_RANGE``,
    ``TELEMETRY_BAD_EPOCH`` (epoch decreased), ``TELEMETRY_MALFORMED``.
    """

    def __init__(self, key: bytes, first_seq: int = 0, max_bandwidth_bps: float = 1e12,
                 max_rtt_s: float = 60.0, max_cpu_load: float = 1e6):
        if not isinstance(key, (bytes, bytearray)) or len(key) < 16:
            raise TelemetryError("TELEMETRY_KEY: key must be at least 16 bytes")
        self.key = bytes(key)
        self.first_seq = int(first_seq)
        self.max_bandwidth_bps = float(max_bandwidth_bps)
        self.max_rtt_s = float(max_rtt_s)
        self.max_cpu_load = float(max_cpu_load)
        self.last_seq: Optional[int] = None
        self.last_t: Optional[float] = None
        self.last_epoch: Optional[int] = None
        self.accepted: List[TelemetryFrame] = []
        self.rejected: List[Tuple[str, Any]] = []

    def _reject(self, code: str, detail: str, frame: Any) -> None:
        self.rejected.append((code, frame))
        raise TelemetryError(f"{code}: {detail}")

    def accept(self, frame: TelemetryFrame, tag: str) -> None:
        if not isinstance(frame, TelemetryFrame):
            self._reject("TELEMETRY_MALFORMED", "frame is not a TelemetryFrame", frame)
        if isinstance(frame.seq, bool) or not isinstance(frame.seq, int) or frame.seq < 0:
            self._reject("TELEMETRY_BAD_SEQ", f"seq {frame.seq!r} is not a non-negative int", frame)
        if isinstance(frame.epoch, bool) or not isinstance(frame.epoch, int) or frame.epoch < 0:
            self._reject("TELEMETRY_BAD_EPOCH", f"epoch {frame.epoch!r} is not a non-negative int", frame)
        for name in ("t", "bandwidth_bps", "rtt_s", "cpu_load"):
            v = getattr(frame, name)
            if not _finite(v) or float(v) < 0.0:
                self._reject("TELEMETRY_NON_FINITE", f"{name}={v!r} is not a finite non-negative number", frame)
        if not isinstance(frame.source, str) or not frame.source:
            self._reject("TELEMETRY_MALFORMED", "source must be a non-empty str", frame)
        # Tag check before any state-dependent check so that a forged frame can
        # never influence the sequencing state.
        if not isinstance(tag, str) or len(tag) != 64:
            self._reject("TELEMETRY_BAD_TAG", f"tag of seq {frame.seq} has the wrong form", frame)
        expected = sign_frame(frame, self.key)
        if not hmac.compare_digest(tag.encode("ascii", "replace"), expected.encode("ascii")):
            self._reject("TELEMETRY_BAD_TAG", f"frame seq={frame.seq} fails HMAC verification", frame)
        if self.last_seq is None:
            if frame.seq != self.first_seq:
                self._reject("TELEMETRY_OUT_OF_ORDER", f"first frame has seq {frame.seq}, expected {self.first_seq}", frame)
        else:
            if frame.seq <= self.last_seq:
                self._reject("TELEMETRY_REPLAY", f"seq {frame.seq} already consumed (last {self.last_seq})", frame)
            if frame.seq != self.last_seq + 1:
                self._reject("TELEMETRY_OUT_OF_ORDER", f"seq {frame.seq} after {self.last_seq} (gap)", frame)
        if self.last_t is not None and float(frame.t) < self.last_t:
            self._reject("TELEMETRY_NON_MONOTONE_TIME", f"t={frame.t} < previous {self.last_t}", frame)
        if self.last_epoch is not None and frame.epoch < self.last_epoch:
            self._reject("TELEMETRY_BAD_EPOCH", f"epoch {frame.epoch} < previous {self.last_epoch}", frame)
        if (float(frame.bandwidth_bps) > self.max_bandwidth_bps or float(frame.rtt_s) > self.max_rtt_s
                or float(frame.cpu_load) > self.max_cpu_load):
            self._reject("TELEMETRY_OUT_OF_RANGE",
                         f"bandwidth {frame.bandwidth_bps} / rtt {frame.rtt_s} / cpu {frame.cpu_load} outside the plausible range",
                         frame)
        self.last_seq = int(frame.seq)
        self.last_t = float(frame.t)
        self.last_epoch = int(frame.epoch)
        self.accepted.append(frame)


# --------------------------------------------------------------------------
# Fixed-size probe
# --------------------------------------------------------------------------
def probe_estimate(rtts: Sequence[float], elapsed: Sequence[float], burst: int) -> Tuple[float, float]:
    """``(bw_bps, rtt_s)`` of one probe from its samples.

    ``rtts[i]`` is the ping/pong time and ``elapsed[i]`` the burst/ack time of
    sample i. ``rtt_s`` is the median ping time. ``bw_bps`` is the pooled
    figure ``burst * 8 / max(median(elapsed) - min(rtts), PROBE_MIN_NET_S)``.
    The median of an even number of samples is the mean of the two middle ones.
    """
    if len(rtts) < 1 or len(rtts) != len(elapsed) or int(burst) < 1:
        raise TelemetryError("TELEMETRY_PROBE_PARAMS: one ping time per burst time, at least one sample, burst >= 1")
    net = max(statistics.median(elapsed) - min(rtts), PROBE_MIN_NET_S)
    return float(int(burst) * 8.0 / net), float(statistics.median(rtts))


def probe_channel(channel: Any, party: int, samples: int = 3, burst: int = 65536) -> Tuple[float, float]:
    """Fixed-size probe: P1 drives, P0 answers; returns ``(bw_bps, rtt_s)`` on P1, ``(nan, nan)`` on P0.

    Per sample: ``ping`` (16 bytes) / ``pong`` (16 bytes) for the RTT, then a
    ``burst``-byte payload / 16-byte ack for the goodput; the figures are
    :func:`probe_estimate` over the ``samples``. Every message size is fixed
    in advance; the probe consumes ``2 * samples`` rounds, which are
    accounted on the channel.
    """
    samples = int(samples)
    burst = int(burst)
    if samples < 1 or burst < 1:
        raise TelemetryError("TELEMETRY_PROBE_PARAMS: samples and burst must be >= 1")
    ping = b"\x00" * PROBE_PING_BYTES
    ack = b"\x01" * PROBE_ACK_BYTES
    payload = b"\x02" * burst
    rtts: List[float] = []
    els: List[float] = []
    for _ in range(samples):
        channel.count_round(2)
        if party == 1:
            t0 = time.perf_counter()
            channel.send(ping)
            channel.recv()
            rtt = time.perf_counter() - t0
            t0 = time.perf_counter()
            channel.send(payload)
            channel.recv()
            el = time.perf_counter() - t0
            rtts.append(rtt)
            els.append(el)
        else:
            channel.recv()
            channel.send(ping)
            channel.recv()
            channel.send(ack)
    if party == 1:
        return probe_estimate(rtts, els, burst)
    return float("nan"), float("nan")


def sense_and_exchange(channel: Any, party: int, key: bytes, seq: int, epoch: int, t: float,
                       verifier: TelemetryVerifier, samples: int = 3, burst: int = 65536,
                       max_bandwidth_bps: float = 1e11, cpu_load: float = 0.0) -> TelemetryFrame:
    """Probe, build a signed frame on P1, send it to P0, verify on both sides.

    Both parties return the same accepted frame (or raise ``TelemetryError``),
    so identical controllers observe identical inputs. Costs one extra round.
    """
    bw, rtt = probe_channel(channel, party, samples=samples, burst=burst)
    channel.count_round(1)
    if party == 1:
        frame = TelemetryFrame(seq=int(seq), t=float(t), bandwidth_bps=float(min(bw, max_bandwidth_bps)),
                               rtt_s=float(max(rtt, 1e-6)), epoch=int(epoch), cpu_load=float(cpu_load),
                               source="probe")
        tag = sign_frame(frame, key)
        channel.send({"frame": frame.to_dict(), "tag": tag})
    else:
        msg = channel.recv()
        if not isinstance(msg, dict) or "frame" not in msg or "tag" not in msg:
            raise TelemetryError("TELEMETRY_MALFORMED: telemetry message must carry frame and tag")
        frame = TelemetryFrame.from_dict(msg["frame"])
        tag = msg["tag"]
    verifier.accept(frame, tag)
    return frame


__all__ = [
    "TelemetryFrame", "sign_frame", "TelemetryVerifier", "probe_estimate", "probe_channel", "sense_and_exchange",
    "canonical_json", "PROBE_PING_BYTES", "PROBE_ACK_BYTES", "PROBE_MIN_NET_S",
]
