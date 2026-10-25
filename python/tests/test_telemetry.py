"""Tests for opencrab2.highlevel.telemetry: signed frames, the fixed-size probe and its pooled estimator."""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path

import pytest

from opencrab2.errors import TelemetryError
from opencrab2.highlevel.channel import LinkState, WallClockShaper, accept_channel, dial_channel, open_listener
from opencrab2.highlevel.telemetry import (PROBE_ACK_BYTES, PROBE_MIN_NET_S, PROBE_PING_BYTES, TelemetryFrame,
                                           TelemetryVerifier, probe_channel, probe_estimate, sense_and_exchange,
                                           sign_frame)

KEY = b"\x11" * 32
OTHER = b"\x22" * 32
VECTORS = Path(__file__).resolve().parents[2] / "cpp" / "tests" / "vectors" / "telemetry_vectors.json"


def frame(seq=0, t=0.0, bw=1e6, rtt=0.01, epoch=0, cpu=0.0, source="probe"):
    return TelemetryFrame(seq=seq, t=t, bandwidth_bps=bw, rtt_s=rtt, epoch=epoch, cpu_load=cpu, source=source)


def tcp_pair(shaper0=None, shaper1=None):
    lsock, port = open_listener("127.0.0.1", 0)
    box = {}

    def acc():
        box["ch"] = accept_channel(lsock, 0, shaper0, timeout_s=10.0)

    t = threading.Thread(target=acc, daemon=True)
    t.start()
    ch1 = dial_channel("127.0.0.1", port, 1, shaper1)
    t.join(10.0)
    return box["ch"], ch1


# --------------------------------------------------------------------------
# signing and verification
# --------------------------------------------------------------------------
def test_sign_verify_round_trip_and_tamper():
    f = frame()
    tag = sign_frame(f, KEY)
    assert len(tag) == 64 and tag == sign_frame(frame(), KEY)
    v = TelemetryVerifier(KEY)
    v.accept(f, tag)
    assert v.accepted == [f] and v.last_seq == 0
    v2 = TelemetryVerifier(KEY)
    with pytest.raises(TelemetryError) as ei:
        v2.accept(frame(bw=2e6), tag)
    assert ei.value.code == "TELEMETRY_BAD_TAG"
    with pytest.raises(TelemetryError) as ei:
        v2.accept(f, sign_frame(f, OTHER))
    assert ei.value.code == "TELEMETRY_BAD_TAG"
    with pytest.raises(TelemetryError) as ei:
        v2.accept(f, "zz")
    assert ei.value.code == "TELEMETRY_BAD_TAG"
    assert v2.last_seq is None and len(v2.rejected) == 3 and v2.rejected[0][0] == "TELEMETRY_BAD_TAG"
    with pytest.raises(TelemetryError):
        sign_frame(f, b"short")
    with pytest.raises(TelemetryError):
        TelemetryVerifier(b"short")


def test_sequencing_replay_gap_and_first_seq():
    v = TelemetryVerifier(KEY)
    with pytest.raises(TelemetryError) as ei:
        v.accept(frame(seq=1), sign_frame(frame(seq=1), KEY))
    assert ei.value.code == "TELEMETRY_OUT_OF_ORDER"
    v.accept(frame(seq=0), sign_frame(frame(seq=0), KEY))
    v.accept(frame(seq=1, t=1.0, epoch=1), sign_frame(frame(seq=1, t=1.0, epoch=1), KEY))
    with pytest.raises(TelemetryError) as ei:
        v.accept(frame(seq=1, t=1.0, epoch=1), sign_frame(frame(seq=1, t=1.0, epoch=1), KEY))
    assert ei.value.code == "TELEMETRY_REPLAY"
    with pytest.raises(TelemetryError) as ei:
        v.accept(frame(seq=0, t=2.0), sign_frame(frame(seq=0, t=2.0), KEY))
    assert ei.value.code == "TELEMETRY_REPLAY"
    with pytest.raises(TelemetryError) as ei:
        v.accept(frame(seq=3, t=2.0, epoch=2), sign_frame(frame(seq=3, t=2.0, epoch=2), KEY))
    assert ei.value.code == "TELEMETRY_OUT_OF_ORDER"
    assert v.last_seq == 1 and len(v.accepted) == 2
    v5 = TelemetryVerifier(KEY, first_seq=5)
    v5.accept(frame(seq=5), sign_frame(frame(seq=5), KEY))


def test_time_epoch_and_value_checks():
    v = TelemetryVerifier(KEY, max_bandwidth_bps=1e9, max_rtt_s=5.0)
    v.accept(frame(seq=0, t=10.0, epoch=2), sign_frame(frame(seq=0, t=10.0, epoch=2), KEY))
    for bad, code in [
        (frame(seq=1, t=9.0, epoch=2), "TELEMETRY_NON_MONOTONE_TIME"),
        (frame(seq=1, t=10.0, epoch=1), "TELEMETRY_BAD_EPOCH"),
        (frame(seq=1, t=10.0, epoch=2, bw=float("nan")), "TELEMETRY_NON_FINITE"),
        (frame(seq=1, t=10.0, epoch=2, rtt=float("inf")), "TELEMETRY_NON_FINITE"),
        (frame(seq=1, t=10.0, epoch=2, bw=-1.0), "TELEMETRY_NON_FINITE"),
        (frame(seq=1, t=10.0, epoch=2, cpu=float("nan")), "TELEMETRY_NON_FINITE"),
        (frame(seq=1, t=10.0, epoch=2, bw=2e9), "TELEMETRY_OUT_OF_RANGE"),
        (frame(seq=1, t=10.0, epoch=2, rtt=6.0), "TELEMETRY_OUT_OF_RANGE"),
        (frame(seq=1, t=10.0, epoch=2, source=""), "TELEMETRY_MALFORMED"),
    ]:
        with pytest.raises(TelemetryError) as ei:
            v.accept(bad, sign_frame(bad, KEY) if math.isfinite(bad.bandwidth_bps) and math.isfinite(bad.rtt_s) and math.isfinite(bad.cpu_load) else "0" * 64)
        assert ei.value.code == code, (bad, ei.value)
    assert v.last_seq == 0
    with pytest.raises(TelemetryError) as ei:
        v.accept(frame(seq=-1), "0" * 64)
    assert ei.value.code == "TELEMETRY_BAD_SEQ"
    with pytest.raises(TelemetryError) as ei:
        v.accept(frame(seq=1, t=10.0, epoch=-1), "0" * 64)
    assert ei.value.code == "TELEMETRY_BAD_EPOCH"
    with pytest.raises(TelemetryError) as ei:
        v.accept({"seq": 1}, "0" * 64)   # type: ignore[arg-type]
    assert ei.value.code == "TELEMETRY_MALFORMED"
    v.accept(frame(seq=1, t=10.0, epoch=2), sign_frame(frame(seq=1, t=10.0, epoch=2), KEY))


def test_dict_round_trip_is_strict():
    f = frame(seq=3, t=1.5, bw=2.5e6, rtt=0.02, epoch=3, cpu=0.3, source="probe")
    d = f.to_dict()
    assert TelemetryFrame.from_dict(d) == f
    assert sign_frame(TelemetryFrame.from_dict(d), KEY) == sign_frame(f, KEY)
    for bad in [dict(d, extra=1), {k: v for k, v in d.items() if k != "rtt_s"}, dict(d, seq="3"),
                dict(d, t="1.5"), dict(d, source=5), dict(d, epoch=True), "frame"]:
        with pytest.raises(TelemetryError) as ei:
            TelemetryFrame.from_dict(bad)
        assert ei.value.code == "TELEMETRY_MALFORMED"
    assert len(f.digest()) == 64


def test_parity_with_hmac_auth_tag():
    from opencrab2.lowlevel import hmac_auth
    for f in (frame(seq=2, t=3.25, bw=1e7, rtt=0.005, epoch=2),
              frame(seq=2, t=3.25, bw=1e7, rtt=0.005, epoch=2, source="sonde été 世界")):
        assert sign_frame(f, KEY) == hmac_auth.tag(KEY, f.to_dict())
        assert hmac_auth.verify(KEY, f.to_dict(), sign_frame(f, KEY))
    with pytest.raises(TelemetryError) as ei:
        sign_frame(frame(bw=float("nan")), KEY)
    assert ei.value.code == "TELEMETRY_NON_FINITE"


# --------------------------------------------------------------------------
# probe
# --------------------------------------------------------------------------
def test_probe_estimate_pools_the_samples():
    """bw = burst * 8 / (median(burst times) - min(ping times)); rtt = median(ping times)."""
    # a 1 Gbit/s link whose delays fall on a 0.5 ms timer grid: 64 KiB take 0.52 ms, the round trip 1.05 ms,
    # and two of the three pings carry one extra tick
    rtts = [1.05e-3, 1.56e-3, 1.58e-3]
    els = [1.60e-3, 1.59e-3, 1.585e-3]
    bw, rtt = probe_estimate(rtts, els, 65536)
    assert math.isclose(rtt, 1.56e-3) and math.isclose(bw, 65536 * 8 / (1.59e-3 - 1.05e-3))
    assert 0.9e9 < bw < 1.1e9
    # the median of the per-sample ratios would read tens of Gbit/s on the same messages
    ratios = sorted(65536 * 8 / max(e - r, PROBE_MIN_NET_S) for r, e in zip(rtts, els))
    assert ratios[1] > 10e9
    # the floor: a burst that returns no later than the shortest ping
    bw2, rtt2 = probe_estimate([1.0e-3, 1.1e-3, 1.2e-3], [0.9e-3, 0.95e-3, 1.0e-3], 65536)
    assert bw2 == 65536 * 8 / PROBE_MIN_NET_S and math.isclose(rtt2, 1.1e-3)
    # one sample, and an even number of samples (the mean of the two middle values)
    bw1, rtt1 = probe_estimate([0.020], [0.050], 4096)
    assert math.isclose(bw1, 4096 * 8 / 0.030) and rtt1 == 0.020
    bw3, rtt3 = probe_estimate([0.040, 0.042], [0.105, 0.109], 16384)
    assert math.isclose(rtt3, 0.041) and math.isclose(bw3, 16384 * 8 / (0.107 - 0.040))
    for bad in (([], [], 1024), ([0.01], [], 1024), ([0.01], [0.02], 0)):
        with pytest.raises(TelemetryError) as ei:
            probe_estimate(*bad)
        assert ei.value.code == "TELEMETRY_PROBE_PARAMS"


def test_probe_estimate_matches_the_shared_vector_file():
    """telemetry_vectors.json (written by cpp/tests/gen_vectors.py) is reproduced to relative 1e-12."""
    assert VECTORS.is_file(), f"VECTORS_MISSING: {VECTORS}"
    doc = json.loads(VECTORS.read_text(encoding="utf-8"))
    assert doc["format"] == "opencrab2-telemetry-vectors-v1" and doc["min_net_s"] == PROBE_MIN_NET_S
    cap = float(doc["max_bandwidth_bps"])
    names = set()
    for case in doc["cases"]:
        bw, rtt = probe_estimate(case["rtt_s"], case["elapsed_s"], case["burst"])
        assert math.isclose(bw, case["bandwidth_bps"], rel_tol=1e-12), case["name"]
        assert math.isclose(rtt, case["rtt_median_s"], rel_tol=1e-12), case["name"]
        assert math.isclose(min(bw, cap), case["frame_bandwidth_bps"], rel_tol=1e-12), case["name"]
        assert math.isclose(max(rtt, 1e-6), case["frame_rtt_s"], rel_tol=1e-12), case["name"]
        names.add(case["name"])
    assert {"one_tick_in_two_pings", "one_tick_in_one_ping", "floor", "above_the_cap", "single_sample", "even_two"} <= names
    assert sum(1 for c in doc["cases"] if c["frame_bandwidth_bps"] < c["bandwidth_bps"]) >= 2   # the cap is exercised


def test_probe_fixed_sizes_over_loopback():
    ch0, ch1 = tcp_pair()
    try:
        box = {}

        def p0():
            box["r0"] = probe_channel(ch0, 0, samples=3, burst=4096)

        t = threading.Thread(target=p0)
        t.start()
        bw, rtt = probe_channel(ch1, 1, samples=3, burst=4096)
        t.join(10)
        assert math.isnan(box["r0"][0]) and math.isnan(box["r0"][1])
        assert math.isfinite(bw) and bw > 0 and math.isfinite(rtt) and rtt > 0
        ping_frame = 8 + 1 + 4 + PROBE_PING_BYTES
        burst_frame = 8 + 1 + 4 + 4096
        ack_frame = 8 + 1 + 4 + PROBE_ACK_BYTES
        assert ch1.stats.bytes_sent == 3 * (ping_frame + burst_frame)
        assert ch0.stats.bytes_sent == 3 * (ping_frame + ack_frame)
        assert ch1.stats.messages_sent == 6 == ch0.stats.messages_recv
        assert ch0.stats.rounds == ch1.stats.rounds == 6
        with pytest.raises(TelemetryError):
            probe_channel(ch1, 1, samples=0)
    finally:
        ch0.close()
        ch1.close()


def test_probe_sees_shaped_rtt_and_bandwidth():
    origin = time.time()
    sh = WallClockShaper([(0.0, LinkState(2e6, 0.020))], origin)   # 2 Mbit/s, 20 ms one way, both directions
    ch0, ch1 = tcp_pair(shaper0=sh, shaper1=WallClockShaper([(0.0, LinkState(2e6, 0.020))], origin))
    try:
        t = threading.Thread(target=lambda: probe_channel(ch0, 0, samples=2, burst=16384))
        t.start()
        bw, rtt = probe_channel(ch1, 1, samples=2, burst=16384)
        t.join(20)
        assert rtt >= 0.036                                  # 2 x 20 ms minus scheduling slack
        assert 2e5 < bw < 2e6 * 1.5                          # not above the shaped rate by more than slack
    finally:
        ch0.close()
        ch1.close()


def test_sense_and_exchange_both_sides_identical_and_replay_refused():
    ch0, ch1 = tcp_pair()
    try:
        v0 = TelemetryVerifier(KEY)
        v1 = TelemetryVerifier(KEY)
        box = {}

        def p0():
            box["f0"] = sense_and_exchange(ch0, 0, KEY, seq=0, epoch=0, t=0.0, verifier=v0, samples=1, burst=1024)
            try:
                sense_and_exchange(ch0, 0, KEY, seq=0, epoch=1, t=1.0, verifier=v0, samples=1, burst=1024)
            except TelemetryError as exc:
                box["e0"] = exc.code

        t = threading.Thread(target=p0)
        t.start()
        f1 = sense_and_exchange(ch1, 1, KEY, seq=0, epoch=0, t=0.0, verifier=v1, samples=1, burst=1024)
        with pytest.raises(TelemetryError) as ei:
            sense_and_exchange(ch1, 1, KEY, seq=0, epoch=1, t=1.0, verifier=v1, samples=1, burst=1024)
        t.join(10)
        assert box["f0"] == f1 and f1.seq == 0 and f1.source == "probe"
        assert ei.value.code == "TELEMETRY_REPLAY" and box["e0"] == "TELEMETRY_REPLAY"
        assert v0.accepted == v1.accepted == [f1]
        assert ch0.stats.rounds == ch1.stats.rounds == 2 * (2 + 1)
    finally:
        ch0.close()
        ch1.close()


def test_sense_and_exchange_rejects_foreign_key_on_the_receiver():
    ch0, ch1 = tcp_pair()
    try:
        v0 = TelemetryVerifier(OTHER)
        v1 = TelemetryVerifier(KEY)
        box = {}

        def p0():
            try:
                sense_and_exchange(ch0, 0, OTHER, seq=0, epoch=0, t=0.0, verifier=v0, samples=1, burst=256)
            except TelemetryError as exc:
                box["e0"] = exc.code

        t = threading.Thread(target=p0)
        t.start()
        sense_and_exchange(ch1, 1, KEY, seq=0, epoch=0, t=0.0, verifier=v1, samples=1, burst=256)
        t.join(10)
        assert box["e0"] == "TELEMETRY_BAD_TAG" and v0.accepted == []
    finally:
        ch0.close()
        ch1.close()
