"""Tests for opencrab2.lowlevel.hmac_auth."""

from __future__ import annotations

import hashlib
import hmac
import json
import math

import pytest

from opencrab2.lowlevel import hmac_auth as ha

KEY = bytes(range(32))


def test_canonical_json_bytes():
    obj = {"b": 1, "a": [1, 2], "c": "é", "d": {"z": True, "y": None}}
    assert ha.canonical_json(obj) == '{"a":[1,2],"b":1,"c":"é","d":{"y":null,"z":true}}'.encode("utf-8")


def test_canonical_json_key_order_independent():
    a = {"profile_id": "SH2P-TrustedOffline-v1", "recipient": 1, "count": 8}
    b = {"count": 8, "recipient": 1, "profile_id": "SH2P-TrustedOffline-v1"}
    assert ha.canonical_json(a) == ha.canonical_json(b)
    assert ha.tag(KEY, a) == ha.tag(KEY, b)


def test_canonical_json_rejects_nan_and_bytes():
    with pytest.raises(ValueError, match="^HMAC_CANONICAL_JSON"):
        ha.canonical_json({"x": math.nan})
    with pytest.raises(TypeError, match="^HMAC_CANONICAL_JSON"):
        ha.canonical_json({"x": b"raw"})


def test_tag_matches_direct_hmac():
    header = {"gate_id": "L3.qkv", "material_id": "m-0001", "count": 4096, "bytes": 65536}
    expected = hmac.new(KEY, json.dumps(header, sort_keys=True, separators=(",", ":")).encode(), hashlib.sha256).hexdigest()
    t = ha.tag(KEY, header)
    assert t == expected
    assert len(t) == 64 and t == t.lower()
    assert ha.verify(KEY, header, t) is True
    assert ha.verify(KEY, header, t.upper()) is True          # hex case does not matter


def test_verify_rejects_tamper_wrong_key_and_malformed():
    header = {"recipient": 0, "material_id": "m-7"}
    t = ha.tag(KEY, header)
    assert ha.verify(KEY, {"recipient": 1, "material_id": "m-7"}, t) is False
    assert ha.verify(b"\x01" * 32, header, t) is False
    assert ha.verify(KEY, header, t[:-1]) is False
    assert ha.verify(KEY, header, t[:-1] + "g") is False
    assert ha.verify(KEY, header, "") is False
    assert ha.verify(KEY, header, None) is False
    assert ha.verify(KEY, header, t[:-1] + ("0" if t[-1] != "0" else "1")) is False


def test_payload_framing():
    header = {"material_type": "beaver", "count": 2}
    payload = b"\x00" * 48
    t = ha.tag_with_payload(KEY, header, payload)
    assert ha.verify_with_payload(KEY, header, payload, t) is True
    assert ha.verify_with_payload(KEY, header, payload + b"\x00", t) is False
    assert ha.verify_with_payload(KEY, header, b"\x01" + payload[1:], t) is False
    framed = ha.mac_input(header, payload)
    hdr = ha.canonical_json(header)
    assert framed == len(hdr).to_bytes(8, "big") + hdr + payload
    # the header/payload boundary is unambiguous: moving bytes across it changes the tag
    moved_header = {"material_type": "beaver", "count": 2, "x": ""}
    assert ha.tag_with_payload(KEY, moved_header, payload) != t
    with pytest.raises(TypeError, match="^HMAC_CANONICAL_JSON"):
        ha.mac_input(header, "not bytes")


def test_key_validation():
    with pytest.raises(ValueError, match="^HMAC_KEY"):
        ha.tag(b"", {"a": 1})
    with pytest.raises(TypeError, match="^HMAC_KEY"):
        ha.tag("string-key", {"a": 1})
    assert ha.tag(bytearray(KEY), {"a": 1}) == ha.tag(KEY, {"a": 1})


def test_tag_length_is_independent_of_message_content():
    short = ha.tag(KEY, {"a": 0})
    long_ = ha.tag(KEY, {"a": "x" * 10_000, "b": list(range(1000))})
    assert len(short) == len(long_) == 64
