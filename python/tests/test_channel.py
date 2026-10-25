"""Tests for opencrab2.highlevel.channel."""

from __future__ import annotations

import math
import socket
import threading
import time

import numpy as np
import pytest

from opencrab2.errors import ChannelError
from opencrab2.highlevel.channel import (Channel, ChannelStats, LinkState, WallClockShaper, accept_channel,
                                         decode_payload, dial_channel, encode_payload, frame_size,
                                         listen_channel, open_listener)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def tcp_pair(shaper0=None, shaper1=None):
    lsock, port = open_listener("127.0.0.1", 0)
    box = {}

    def acc():
        box["ch"] = accept_channel(lsock, 0, shaper0, timeout_s=10.0)

    t = threading.Thread(target=acc, daemon=True)
    t.start()
    ch1 = dial_channel("127.0.0.1", port, 1, shaper1)
    t.join(10.0)
    assert "ch" in box
    return box["ch"], ch1


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


# --------------------------------------------------------------------------
# codec
# --------------------------------------------------------------------------
# Binding byte vectors of the wire codec. The same
# values produce the same bytes in the C++ codec (cpp/src/highlevel/codec.cpp).
WIRE_VECTORS = [
    (None, "08"),
    (True, "0901"),
    (False, "0900"),
    (5, "040000000000000005"),
    (-1, "04ffffffffffffffff"),
    (2 ** 63 - 1, "047fffffffffffffff"),
    (-(2 ** 63), "048000000000000000"),
    (1.5, "053ff8000000000000"),
    ("ab", "06000000026162"),
    ("é", "0600000002c3a9"),
    (b"ab", "01000000026162"),
    (b"", "0100000000"),
    (np.array([1, 2], np.uint64), "02" "01" "01" "0000000000000002" "0100000000000000" "0200000000000000"),
    (np.array([-1], np.int64), "02" "02" "01" "0000000000000001" "ffffffffffffffff"),
    (np.array([1.5], np.float64), "02" "03" "01" "0000000000000001" "000000000000f83f"),
    (np.array([[1, 2, 3]], np.uint8), "02" "04" "02" "0000000000000001" "0000000000000003" "010203"),
    (np.array([-2], np.int32), "02" "05" "01" "0000000000000001" "feffffff"),
    (np.uint64(7), "02" "01" "00" "0700000000000000"),
    ({"k": None}, "03" "00000001" "0001" "6b" "08"),
    ([True, None], "07" "00000002" "0901" "08"),
    ({"type": "REQ", "count": 1}, "03" "00000002" "0004" "74797065" "06" "00000003" "524551"
                                  "0005" "636f756e74" "04" "0000000000000001"),
]


@pytest.mark.parametrize("obj,hexbytes", WIRE_VECTORS, ids=[str(i) for i in range(len(WIRE_VECTORS))])
def test_codec_binding_byte_vectors(obj, hexbytes):
    enc = encode_payload(obj)
    assert enc.hex() == hexbytes
    dec = decode_payload(bytes.fromhex(hexbytes))
    if isinstance(obj, (np.ndarray, np.generic)):
        assert isinstance(dec, np.ndarray) and dec.dtype == np.asarray(obj).dtype
        assert dec.shape == np.asarray(obj).shape and np.array_equal(dec, obj)
    else:
        assert dec == obj and type(dec) is type(obj)
    assert frame_size(obj) == 8 + len(enc)


def test_codec_none_and_bool_tags_are_not_swapped():
    # both codecs fix 0x08 = none, 0x09 = bool (an earlier codec had them the other way round)
    assert encode_payload(None) == b"\x08" and encode_payload(True) == b"\x09\x01"
    assert decode_payload(b"\x08") is None and decode_payload(b"\x09\x00") is False
    with pytest.raises(ChannelError):
        decode_payload(b"\x08\x01")          # the old encoding of True: trailing byte after a none
    with pytest.raises(ChannelError):
        decode_payload(b"\x09")              # the old encoding of None: a bool without its byte
    with pytest.raises(ChannelError):
        decode_payload(b"\x09\x02")          # a bool byte must be 0 or 1


def test_codec_round_trip_all_types():
    arr = np.arange(12, dtype=np.uint64).reshape(3, 4) * np.uint64(2 ** 60)
    obj = {
        "bytes": b"\x00\x01\xff", "u64": arr, "i64": np.array([-1, 2], dtype=np.int64),
        "f64": np.linspace(0, 1, 5), "u8": np.arange(4, dtype=np.uint8), "i32": np.array([[-7, 8]], np.int32),
        "int": -5, "max": 2 ** 63 - 1, "min": -(2 ** 63),
        "float": 1.5, "str": "héllo 世界", "list": [1, "x", None, True, 2.5, b"z"],
        "tuple": (1, 2), "none": None, "bool": True, "nested": {"d": {"e": [np.uint64(7)]}}, "empty": {},
        "scalar": np.float64(2.5), "zero_dim": np.zeros((0, 3), np.int32),
        "big_endian": np.arange(3, dtype=">u8"), "strided": np.arange(10, dtype=np.uint64)[::3],
    }
    d = decode_payload(encode_payload(obj))
    assert list(d) == list(obj)              # dict order is preserved on the wire
    assert d["bytes"] == b"\x00\x01\xff"
    assert np.array_equal(d["u64"], arr) and d["u64"].dtype == np.uint64
    assert np.array_equal(d["i64"], obj["i64"]) and d["i64"].dtype == np.int64
    assert np.array_equal(d["f64"], obj["f64"]) and d["f64"].dtype == np.float64
    assert d["u8"].dtype == np.uint8 and np.array_equal(d["i32"], obj["i32"]) and d["i32"].dtype == np.int32
    assert d["int"] == -5 and d["max"] == 2 ** 63 - 1 and d["min"] == -(2 ** 63)
    assert d["float"] == 1.5 and d["str"] == obj["str"]
    assert d["list"] == [1, "x", None, True, 2.5, b"z"]
    assert d["tuple"] == [1, 2]
    assert d["none"] is None and d["bool"] is True and isinstance(d["bool"], bool)
    assert d["nested"]["d"]["e"][0].shape == () and int(d["nested"]["d"]["e"][0]) == 7
    assert d["empty"] == {}
    assert d["scalar"].shape == () and float(d["scalar"]) == 2.5
    assert d["zero_dim"].shape == (0, 3) and d["zero_dim"].dtype == np.int32
    assert d["big_endian"].dtype == np.uint64 and d["big_endian"].tolist() == [0, 1, 2]
    assert d["strided"].tolist() == [0, 3, 6, 9]
    assert d["u64"].flags.writeable


def test_codec_non_finite_floats_and_ints_edge():
    d = decode_payload(encode_payload([float("nan"), float("inf"), 0, 2 ** 63 - 1, -(2 ** 63), 255, 256, -1, -128, -129]))
    assert math.isnan(d[0]) and d[1] == math.inf
    assert d[2:] == [0, 2 ** 63 - 1, -(2 ** 63), 255, 256, -1, -128, -129]


@pytest.mark.parametrize("bad", [2 ** 63, -(2 ** 63) - 1, 2 ** 64 - 1, 2 ** 70, -(2 ** 65), 2 ** 2100])
def test_codec_int_outside_i64_fails_closed(bad):
    # an int is exactly an i64 on the wire: wider values are refused, never truncated or widened
    for obj in (bad, [1, bad], {"k": {"deep": bad}}):
        with pytest.raises(ChannelError) as ei:
            encode_payload(obj)
        assert ei.value.code == "CODEC_INT_RANGE"
    # the two admitted carriers of a wide value: a uint64 array (when it fits) or a decimal string
    if 0 <= bad < 2 ** 64:
        assert int(decode_payload(encode_payload(np.uint64(bad)))) == bad
        assert decode_payload(encode_payload(np.array([bad], np.uint64))).tolist() == [bad]
    assert int(decode_payload(encode_payload(str(bad)))) == bad


def test_frame_size_depends_only_on_shape():
    a = np.zeros((7, 9), np.uint64)
    b = np.full((7, 9), 2 ** 64 - 1, np.uint64)
    # 8 prefix + tag + dtype code + ndim + two u64 dims + raw data
    assert frame_size(a) == frame_size(b) == 8 + 1 + 1 + 1 + 16 + 7 * 9 * 8
    assert frame_size({"x": a}) == frame_size({"x": b})
    assert frame_size(b"\x00" * 16) == 8 + 1 + 4 + 16
    assert frame_size(5) == frame_size(-(2 ** 63)) == 8 + 1 + 8          # an int never changes the frame size


@pytest.mark.parametrize("bad", [
    object(), {1: 2}, {"k": object()}, np.array([object()], dtype=object), set([1]),
    np.array([1 + 2j]),
    # dtypes outside the five wire dtypes are refused, never cast silently
    np.ones(3, np.float32), np.array([True, False]), np.arange(3, dtype=np.uint32), np.arange(3, dtype=np.int16),
    np.arange(3, dtype=np.int8), np.arange(3, dtype=np.uint16), np.bool_(True), np.float32(1.0),
    np.zeros((1,) * 17, np.uint64),
])
def test_codec_refuses_unsupported(bad):
    with pytest.raises(ChannelError) as ei:
        encode_payload(bad)
    assert ei.value.code in ("CHANNEL_CODEC_UNSUPPORTED", "CHANNEL_CODEC_BAD_KEY")


def test_codec_nesting_limit_is_symmetric():
    deep = cur = []
    for _ in range(63):
        nxt = []
        cur.append(nxt)
        cur = nxt
    assert decode_payload(encode_payload(deep)) == deep          # 64 nested lists are fine
    cur.append([])
    with pytest.raises(ChannelError) as ei:
        encode_payload(deep)                                      # 65 are refused by the encoder
    assert ei.value.code == "CHANNEL_CODEC_UNSUPPORTED"
    with pytest.raises(ChannelError) as ei:
        decode_payload(b"\x07\x00\x00\x00\x01" * 65 + b"\x08")    # and by the decoder
    assert ei.value.code == "CHANNEL_CODEC_MALFORMED"
    assert decode_payload(b"\x07\x00\x00\x00\x01" * 64 + b"\x08") is not None


def test_codec_malformed_inputs():
    good = encode_payload({"a": np.arange(3, dtype=np.uint64), "b": "x"})
    with pytest.raises(ChannelError) as ei:
        decode_payload(good[:-1])
    assert ei.value.code == "CHANNEL_CODEC_MALFORMED"
    with pytest.raises(ChannelError):
        decode_payload(good + b"\x00")
    with pytest.raises(ChannelError):
        decode_payload(b"\x7f")
    with pytest.raises(ChannelError):
        decode_payload(b"\x00")
    with pytest.raises(ChannelError):
        decode_payload(b"\x0a")
    with pytest.raises(ChannelError):
        decode_payload(b"")
    arr = encode_payload(np.arange(2, dtype=np.uint64))
    assert arr[:3] == b"\x02\x01\x01"
    # array header claiming a dtype code outside 1..5
    for code in (0, 6, 255):
        with pytest.raises(ChannelError) as ei:
            decode_payload(arr[:1] + bytes([code]) + arr[2:])
        assert ei.value.code == "CHANNEL_CODEC_MALFORMED"
    # a dimension that disagrees with the data that follows (too many and too few elements)
    for dim in (3, 1, 2 ** 63):
        bad = arr[:3] + dim.to_bytes(8, "big") + arr[11:]
        with pytest.raises(ChannelError) as ei:
            decode_payload(bad)
        assert ei.value.code == "CHANNEL_CODEC_MALFORMED"
    # too many dimensions, truncated int / float / bytes / str, invalid utf-8
    for bad in (b"\x02\x01\x11" + b"\x00" * 8 * 17, b"\x04" + b"\x00" * 7, b"\x05" + b"\x00" * 7,
                b"\x01\x00\x00\x00\x05abc", b"\x06\x00\x00\x00\x05abc", b"\x06\x00\x00\x00\x01\xff",
                b"\x03\x00\x00\x00\x01\x00\x01\xff\x08", b"\x03\x00\x00\x00\x02\x00\x01k\x08",
                b"\x07\x00\x00\x00\x02\x08"):
        with pytest.raises(ChannelError) as ei:
            decode_payload(bad)
        assert ei.value.code == "CHANNEL_CODEC_MALFORMED"


def test_codec_malformed_frames_shared_with_cpp():
    # The same frames are refused by the C++ decoder (cpp test_ledger, decoder limits); every refusal is a
    # typed ChannelError (a bare ValueError from numpy used to escape for the (0, 2^63) shape).
    import struct
    assert decode_payload(bytes([0x09, 0x00])) is False and decode_payload(bytes([0x09, 0x01])) is True

    def nd(ndim, dims):
        return bytes([0x02, 0x01, ndim]) + b"".join(struct.pack(">Q", d) for d in dims)

    bad = [bytes([0x09, 0x02]), bytes([0x09, 0xFF]), nd(17, [0] * 17), nd(255, [0] * 255), nd(2, [0, 1 << 63]),
           nd(2, [0, (1 << 64) - 1]), nd(3, [1 << 62, 1 << 62, 0]),
           bytes([0x06, 0, 0, 0, 2, 0xFF, 0xFE]), bytes([0x06, 0, 0, 0, 2, 0xC0, 0x80]), bytes([0x06, 0, 0, 0, 3, 0xED, 0xA0, 0x80]),
           bytes([0x06, 0, 0, 0, 4, 0xF4, 0x90, 0x80, 0x80]), bytes([0x06, 0, 0, 0, 1, 0xE2]),
           bytes([0x03, 0, 0, 0, 1, 0, 1, 0xFF, 0x08])]
    for frame in bad:
        with pytest.raises(ChannelError) as ei:
            decode_payload(frame)
        assert ei.value.code == "CHANNEL_CODEC_MALFORMED", frame.hex()
    assert decode_payload(nd(16, [0] * 16)).shape == (0,) * 16
    assert decode_payload(nd(2, [0, 1 << 40])).shape == (0, 1 << 40)
    ok = bytes([0x06, 0, 0, 0, 10, 0x61, 0xC3, 0xA9, 0xE2, 0x82, 0xAC, 0xF0, 0x9F, 0x98, 0x80])
    assert encode_payload(decode_payload(ok)) == ok


def test_codec_never_uses_pickle():
    import opencrab2.highlevel.channel as ch
    import inspect
    src = inspect.getsource(ch)
    assert "pickle" not in src.replace("never uses pickle", "").replace("never pickle", "").replace("NEVER uses pickle", "").replace("not pickle", "")


# --------------------------------------------------------------------------
# TCP channel
# --------------------------------------------------------------------------
def test_tcp_send_recv_and_exact_counters():
    ch0, ch1 = tcp_pair()
    try:
        arr = np.arange(1000, dtype=np.uint64)
        msg = {"arr": arr, "n": 3, "s": "ok", "b": b"\x01" * 100}
        box = {}

        def p1():
            box["got"] = ch1.recv()
            ch1.send(b"ack")

        t = threading.Thread(target=p1)
        t.start()
        ch0.send(msg)
        ack = ch0.recv()
        t.join(5)
        assert ack == b"ack"
        assert np.array_equal(box["got"]["arr"], arr) and box["got"]["n"] == 3
        assert ch0.stats.bytes_sent == frame_size(msg) == ch1.stats.bytes_recv
        assert ch1.stats.bytes_sent == frame_size(b"ack") == ch0.stats.bytes_recv
        assert ch0.stats.messages_sent == 1 and ch0.stats.messages_recv == 1
        assert ch1.stats.messages_sent == 1 and ch1.stats.messages_recv == 1
        assert ch0.stats.rounds == 0 and ch1.stats.rounds == 0
        snap = ch0.stats.snapshot()
        assert snap["bytes_total"] == snap["bytes_sent"] + snap["bytes_recv"]
    finally:
        ch0.close()
        ch1.close()


def test_exchange_counts_one_round_each_side():
    ch0, ch1 = tcp_pair()
    try:
        box = {}

        def p1():
            box["r"] = [ch1.exchange(np.uint64(i)) for i in range(5)]

        t = threading.Thread(target=p1)
        t.start()
        r0 = [ch0.exchange(np.uint64(10 + i)) for i in range(5)]
        t.join(5)
        assert [int(x) for x in r0] == list(range(5))
        assert [int(x) for x in box["r"]] == list(range(10, 15))
        assert ch0.stats.rounds == 5 and ch1.stats.rounds == 5
        assert ch0.stats.bytes_sent == ch1.stats.bytes_recv and ch1.stats.bytes_sent == ch0.stats.bytes_recv
        before = ch0.stats.copy()
        ch0.count_round(2)
        assert ch0.stats.diff(before)["rounds"] == 2
    finally:
        ch0.close()
        ch1.close()


def test_listen_channel_with_fixed_port_and_dial_retries():
    port = free_port()
    box = {}

    def srv():
        box["ch"], box["port"] = listen_channel("127.0.0.1", port, 0, timeout_s=10.0)

    t = threading.Thread(target=srv, daemon=True)
    # dial first: the dialer must retry until the listener is up
    t2 = threading.Timer(0.2, t.start)
    t2.start()
    ch1 = dial_channel("127.0.0.1", port, 1, retries=500)
    t.join(10)
    assert box["port"] == port
    ch0 = box["ch"]
    try:
        ch1.send(42)
        assert ch0.recv() == 42
    finally:
        ch0.close()
        ch1.close()


def test_recv_after_peer_close_raises_channel_closed():
    ch0, ch1 = tcp_pair()
    ch1.close()
    with pytest.raises(ChannelError) as ei:
        ch0.recv()
    assert ei.value.code == "CHANNEL_CLOSED"
    ch0.close()


def test_dial_failure_code():
    with pytest.raises(ChannelError) as ei:
        dial_channel("127.0.0.1", free_port(), 1, retries=2, retry_sleep_s=0.01)
    assert ei.value.code == "CHANNEL_CONNECT_FAILED"


def test_channel_party_validation():
    s = socket.socket()
    with pytest.raises(ChannelError):
        Channel(s, 2)
    s.close()


# --------------------------------------------------------------------------
# shaper (pure functions)
# --------------------------------------------------------------------------
def test_shaper_integral_across_segment_change():
    sh = WallClockShaper([(0.0, LinkState(8000.0, 0.01)), (1.0, LinkState(16000.0, 0.02))], origin_wall_s=100.0)
    # 1500 bytes = 12000 bits starting at elapsed 0.5: 0.5 s at 8 kbit/s drains 4000 bits,
    # the remaining 8000 bits take 0.5 s at 16 kbit/s: 1.0 s in total.
    assert sh.serialization_time(1500, 100.5) == pytest.approx(1.0)
    assert sh.delay_for(1500, 100.5) == pytest.approx(1.0 + 0.02)   # delay of the state at the end
    # entirely inside the first segment: 500 bytes = 4000 bits at 8 kbit/s = 0.5 s
    assert sh.delay_for(500, 100.0) == pytest.approx(0.5 + 0.01)
    # a transfer at a constant rate equals bits / rate, whatever the start
    assert sh.serialization_time(2000, 105.0) == pytest.approx(16000 / 16000.0)
    # before the origin the first state applies
    assert sh.state_at(50.0).bandwidth_bps == 8000.0
    assert sh.state_at(101.0).bandwidth_bps == 16000.0 and sh.state_at(100.999).bandwidth_bps == 8000.0
    assert sh.rtt_at(101.0) == pytest.approx(0.04)
    assert sh.delay_for(0, 100.0) == pytest.approx(0.01)


def test_shaper_monotone_in_bytes_and_additive():
    sh = WallClockShaper([(0.0, LinkState(1e6, 0.0)), (0.5, LinkState(2e5, 0.0)), (2.0, LinkState(5e6, 0.0))], 0.0)
    prev = 0.0
    for n in (0, 10, 1000, 10000, 100000, 1000000):
        cur = sh.serialization_time(n, 0.3)
        assert cur >= prev
        prev = cur
    # splitting a transfer at the virtual-clock boundary gives the same total
    whole = sh.serialization_time(200000, 0.3)
    first = sh.serialization_time(100000, 0.3)
    second = sh.serialization_time(100000, 0.3 + first)
    assert whole == pytest.approx(first + second, rel=1e-9)


def test_shaper_unlimited_and_down_segments():
    sh = WallClockShaper([(0.0, LinkState(None, 0.005))], 0.0)
    assert sh.serialization_time(10 ** 9, 0.0) == 0.0
    assert sh.delay_for(10 ** 9, 0.0) == pytest.approx(0.005)
    # bandwidth 0 has ONE meaning in both languages: unshaped. It used to mean a
    # blocked link in this class only, while the C++ shaper, platform.shaper and the cost model said unshaped.
    zero = WallClockShaper([(0.0, LinkState(0.0, 0.0)), (1.0, LinkState(8000.0, 0.0))], 0.0)
    assert zero.serialization_time(1000, 0.5) == 0.0
    assert WallClockShaper([(0.0, LinkState(0.0, 0.25))], 0.0).delay_for(10 ** 6, 0.0) == pytest.approx(0.25)
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ChannelError) as ei:
            LinkState(bad, 0.0)
        assert ei.value.code == "CHANNEL_SHAPER_PARAMS"
        with pytest.raises(ChannelError):
            LinkState(1e6, bad)


def test_shaper_rule_shared_with_cpp():
    # Slow links, an unshaped stretch and a zero delay; the same cases are in
    # cpp/tests/vectors/shaper_vectors.json and are checked by cpp test_vectors and cpp test_engine.
    # Delay = serialization + one-way delay of the state in force when the LAST byte leaves.
    pts = [(0.0, 8000.0, 0.010), (1.0, 80000.0, 0.050), (2.0, 0.0, 0.200), (3.0, 800.0, 0.0)]
    sh = WallClockShaper.from_points(pts, 0.0)
    assert sh.delay_for(1000, 0.0) == pytest.approx(1.050, abs=1e-12)
    assert sh.delay_for(1500, 0.0) == pytest.approx(1.100, abs=1e-12)
    assert sh.delay_for(20000, 0.5) == pytest.approx(1.700, abs=1e-12)     # C++ said 1.51, Python 97.5 before
    assert sh.delay_for(100, 2.5) == pytest.approx(0.200, abs=1e-12)       # C++ 0.2, Python 1.5 before
    import json
    import pathlib
    path = pathlib.Path(__file__).resolve().parents[2] / "cpp" / "tests" / "vectors" / "shaper_vectors.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert len(doc["cases"]) >= 40
    for case in doc["cases"]:
        s = WallClockShaper.from_points([tuple(p) for p in case["points"]], float(case["origin"]))
        assert s.delay_for(int(case["nbytes"]), float(case["wall_s"])) == pytest.approx(float(case["delay_s"]), rel=1e-12, abs=1e-12)
        assert s.serialization_time(int(case["nbytes"]), float(case["wall_s"])) == pytest.approx(
            float(case["serialization_s"]), rel=1e-12, abs=1e-12)


def test_shaper_validation_and_from_points():
    with pytest.raises(ChannelError):
        WallClockShaper([], 0.0)
    with pytest.raises(ChannelError):
        WallClockShaper([(1.0, LinkState(1.0, 0.0)), (0.0, LinkState(1.0, 0.0))], 0.0)
    with pytest.raises(ChannelError):
        LinkState(-1.0, 0.0)
    with pytest.raises(ChannelError):
        LinkState(1.0, float("nan"))
    sh = WallClockShaper.from_points([(0.0, 1e6, 0.001, 1.0, 0.0), (5.0, None, 0.002, 1.0, 0.0)], 10.0)
    assert sh.state_at(10.0).bandwidth_bps == 1e6 and sh.state_at(15.0).bandwidth_bps is None
    with pytest.raises(ChannelError):
        WallClockShaper.from_points([(0.0, 1e6)], 0.0)


def test_shaped_channel_timing_sanity():
    origin = time.time()
    # 1 Mbit/s and 10 ms one way on P0's sending side only
    sh = WallClockShaper([(0.0, LinkState(1e6, 0.010))], origin)
    ch0, ch1 = tcp_pair(shaper0=sh)
    try:
        payload = b"\x00" * 12500           # 100 000 bits: 0.1 s at 1 Mbit/s
        box = {}

        def p1():
            box["t"] = time.perf_counter()
            box["got"] = ch1.recv()
            box["t"] = time.perf_counter() - box["t"]

        t = threading.Thread(target=p1)
        t.start()
        t0 = time.perf_counter()
        ch0.send(payload)
        el = time.perf_counter() - t0
        t.join(5)
        assert box["got"] == payload
        assert el >= 0.9 * (0.1 + 0.010)
        assert ch0.stats.shaping_delay_s >= 0.9 * 0.11
        assert ch1.stats.shaping_delay_s == 0.0
        ch0.set_shaper(None)
        t1 = time.perf_counter()
        t = threading.Thread(target=lambda: ch1.recv())
        t.start()
        ch0.send(payload)
        t.join(5)
        assert time.perf_counter() - t1 < 0.09
    finally:
        ch0.close()
        ch1.close()


def test_stats_snapshot_fields():
    s = ChannelStats(1, 2, 3, 4, 5, 0.5)
    snap = s.snapshot()
    assert snap == {"bytes_sent": 1, "bytes_recv": 2, "bytes_total": 3, "messages_sent": 3, "messages_recv": 4,
                    "rounds": 5, "shaping_delay_s": 0.5}
