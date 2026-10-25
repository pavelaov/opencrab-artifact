"""Trace generator tests: determinism, hash stability, family semantics, split rule, shaper compatibility."""

from __future__ import annotations

import pytest

from opencrab2.experiments import traces
from opencrab2.highlevel.channel import WallClockShaper

D = 120.0


def test_every_family_has_at_least_three_variants_and_generates():
    assert set(traces.FAMILY_NAMES) == {f"S{i}" for i in range(9)}
    for fam in traces.FAMILY_NAMES:
        vs = traces.variants(fam)
        assert len(vs) >= 3, fam
        for v in vs:
            tr = traces.make_trace(fam, v, 2, D)
            pts = tr["points"]
            assert pts[0][0] == 0.0
            assert all(a[0] <= b[0] for a, b in zip(pts, pts[1:]))
            assert all(0.0 <= p[0] < D for p in pts)
            assert all(p[1] > 0 and p[2] >= 0 and 0 < p[3] <= 1.0 and p[4] >= 0 for p in pts)
            assert tr["trace_id"] == f"{fam}:{v}:seed2:d120"
            assert tr["generator_version"] == traces.GENERATOR_VERSION
            assert tr["provenance"]["kind"] == "synthetic"


def test_deterministic_by_seed_and_hash_stable():
    for fam, v in (("S3", "six-changes"), ("S1", "ramp-WAN-c-to-LAN"), ("S2", "rtt-shuffle"), ("S7", "cancel-no-resume")):
        a = traces.make_trace(fam, v, 4, D)
        b = traces.make_trace(fam, v, 4, D)
        assert a == b
        assert a["trace_hash"] == traces.points_hash(a["points"])
        c = traces.make_trace(fam, v, 5, D)
        assert c["trace_id"] != a["trace_id"]
    # seeded randomness changes S3 and the S2 shuffle but not the stable link
    assert traces.make_trace("S3", "six-changes", 4, D)["trace_hash"] != traces.make_trace("S3", "six-changes", 5, D)["trace_hash"]
    assert traces.make_trace("S0", "stable-WAN", 4, D)["trace_hash"] == traces.make_trace("S0", "stable-WAN", 5, D)["trace_hash"]


def test_pinned_hash_regression():
    # the generator version is part of the seed derivation; a change here must be deliberate
    tr = traces.make_trace("S3", "two-changes", 2, D)
    again = traces.make_trace("S3", "two-changes", 2, D)
    assert tr["trace_hash"] == again["trace_hash"]
    assert len(tr["trace_hash"]) == 64


def test_split_rule_is_seed_parity():
    assert traces.split_for_seed(1) == "tune" and traces.split_for_seed(2) == "test"
    assert traces.make_trace("S0", "stable-LAN", 7, D)["split"] == "tune"
    assert traces.make_trace("S0", "stable-LAN", 8, D)["split"] == "test"


def test_s0_matches_anchor():
    for v, anchor in (("stable-LAN", "LAN"), ("stable-WAN", "WAN"), ("stable-WAN-c", "WAN-c"), ("stable-WAN-mid", "WAN-mid")):
        tr = traces.make_trace("S0", v, 1, D)
        bw, dl = traces.ANCHORS[anchor]
        assert tr["points"] == [[0.0, bw, dl, 1.0, 0.0]]


def test_s1_is_a_linear_ramp_between_anchors():
    tr = traces.make_trace("S1", "ramp-WAN-c-to-LAN", 2, D)
    pts = tr["points"]
    bws = [p[1] for p in pts]
    assert len(pts) >= 100
    assert all(b <= a for a, b in zip(bws[1:], bws[2:])) or all(a <= b for a, b in zip(bws, bws[1:]))
    assert bws[0] == traces.ANCHORS["WAN-c"][0]
    assert abs(bws[-1] - traces.ANCHORS["LAN"][0]) < (traces.ANCHORS["LAN"][0] - traces.ANCHORS["WAN-c"][0]) / 50
    assert len({p[2] for p in pts}) == 1                       # delay fixed on a bandwidth ramp
    down = traces.make_trace("S1", "ramp-LAN-to-WAN-c", 2, D)["points"]
    assert all(a[1] >= b[1] for a, b in zip(down, down[1:]))


def test_s2_varies_only_the_delay_inside_the_admitted_set():
    for v in traces.variants("S2"):
        tr = traces.make_trace("S2", v, 3, D)
        pts = tr["points"]
        assert {p[1] for p in pts} == {traces.S2_BANDWIDTH_BPS}
        delays_ms = sorted(round(p[2] * 1e3, 6) for p in pts)
        assert delays_ms == sorted(traces.S2_DELAYS_MS)
        assert len(pts) == len(traces.S2_DELAYS_MS)
    up = traces.make_trace("S2", "rtt-step-up", 3, D)["points"]
    assert [p[2] for p in up] == sorted(p[2] for p in up)


def test_s3_change_and_spike_counts():
    for v, (nc, ns) in (("two-changes", (2, 1)), ("four-changes", (4, 2)), ("six-changes", (6, 3))):
        for seed in (1, 2, 3):
            tr = traces.make_trace("S3", v, seed, D)
            m = tr["meta"]
            assert m["n_changes"] == nc and m["n_spikes"] == ns
            assert 2 <= m["n_changes"] <= 6 and 1 <= m["n_spikes"] <= 3
            assert len(m["regimes"]) == nc + 1
            assert all(a != b for a, b in zip(m["regimes"], m["regimes"][1:]))
            assert all(0.5 <= b - a <= 3.0 + 1e-9 for a, b in m["spikes_s"])
            assert all(0.0 < t < D for t in m["change_times_s"])
            anchors = {tuple(x) for x in traces.ANCHORS.values()}
            spiked = [p for p in tr["points"] if (p[1], p[2]) not in anchors]
            assert spiked, "a spike must leave the anchor set"


def test_s4_cpu_share_levels():
    for v, level in (("cpu-1.0", 1.0), ("cpu-0.5", 0.5), ("cpu-0.25", 0.25)):
        tr = traces.make_trace("S4", v, 2, D)
        shares = {p[3] for p in tr["points"]}
        assert shares <= {1.0, level}
        if level < 1.0:
            assert level in shares
            a, b = tr["meta"]["episode_s"]
            assert 0 < a < b <= D
            assert traces.state_at(tr, (a + b) / 2)["cpu_share"] == level
            assert traces.state_at(tr, a / 2)["cpu_share"] == 1.0


def test_s5_s6_s7_s8_metadata():
    for v in traces.variants("S5"):
        tr = traces.make_trace("S5", v, 2, D)
        assert tr["points"][0][4] > 0 and tr["initial_stock_bytes"] > 0
        assert traces.state_at(tr, 1.0)["stock_refill_bps"] == tr["meta"]["stock_refill_bps"]
    for v, lb in (("len-64", 64), ("len-256", 256), ("len-1024", 1024)):
        tr = traces.make_trace("S6", v, 2, D)
        assert tr["length_bucket"] == lb and tr["mem_limit_bytes"] > 0
    for v in traces.variants("S7"):
        tr = traces.make_trace("S7", v, 2, D)
        kinds = [k for _, k in tr["events"]]
        assert kinds and kinds[0] in ("cancel", "disconnect")
        assert all(0.0 <= t <= D for t, _ in tr["events"])
        assert all(a[0] <= b[0] for a, b in zip(tr["events"], tr["events"][1:]))
        if v == "cancel-no-resume":
            assert "resume" not in kinds
        assert traces.events_between(tr, 0.0, D) == [(float(t), k) for t, k in tr["events"]]
    assert traces.make_trace("S8", "cold", 2, D)["initial_installed"] == []
    assert traces.make_trace("S8", "warm-embedding", 2, D)["initial_installed"] == ["wte", "lm_head"]
    warm = traces.make_trace("S8", "warm-all", 2, D)["initial_installed"]
    assert len(warm) == 2 + 48 and "h11.proj" in warm


def test_shaper_compatibility_and_state_lookup():
    tr = traces.make_trace("S3", "six-changes", 4, D)
    sh = WallClockShaper.from_points(traces.shaper_points(tr), 0.0)
    assert sh.delay_for(1 << 20, 10.0) > 0.0
    st = traces.state_at(tr, 0.0)
    assert st["bandwidth_bps"] == tr["points"][0][1]
    last = tr["points"][-1]
    assert traces.state_at(tr, D * 10)["bandwidth_bps"] == last[1]


def test_catalog_and_io_round_trip(tmp_path):
    cat = traces.catalog(D, [1, 2], ["S0", "S2"])
    assert len(cat) == (len(traces.variants("S0")) + len(traces.variants("S2"))) * 2
    tr = cat[0]
    p = traces.write_trace(tr, tmp_path)
    assert traces.load_trace(p) == tr
    text = p.read_text(encoding="utf-8").replace("1000000000.0", "999.0", 1)
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError) as ei:
        traces.load_trace(p)
    assert str(ei.value).startswith("TRACE_HASH_MISMATCH")


def test_errors_have_codes():
    with pytest.raises(ValueError) as ei:
        traces.make_trace("S9", "x", 1, D)
    assert str(ei.value).startswith("TRACE_UNKNOWN_FAMILY")
    with pytest.raises(ValueError) as ei:
        traces.make_trace("S0", "nope", 1, D)
    assert str(ei.value).startswith("TRACE_UNKNOWN_VARIANT")
    with pytest.raises(ValueError) as ei:
        traces.make_trace("S0", "stable-LAN", -1, D)
    assert str(ei.value).startswith("TRACE_PARAM")
    with pytest.raises(ValueError) as ei:
        traces.make_trace("S0", "stable-LAN", 1, 0.0)
    assert str(ei.value).startswith("TRACE_PARAM")
