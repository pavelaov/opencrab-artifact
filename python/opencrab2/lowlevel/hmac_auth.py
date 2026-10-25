"""HMAC-SHA256 tags over canonical JSON with constant-time verification.

Canonical JSON: sorted keys, separators without spaces, UTF-8 bytes,
non-finite floats rejected. `tag(key, obj)` returns the lowercase hex HMAC of
`canonical_json(obj)`; `verify` recomputes it and compares with
`hmac.compare_digest`. The `*_with_payload` variants authenticate a JSON
header together with an opaque byte payload (dealer bundles) through the
unambiguous framing `len(header) as 8-byte big-endian || header || payload`.

These are integrity tools for dealer bundle delivery and telemetry frames
under the SH2P-TrustedOffline-v1 profile. They are not security proofs and
they do not provide active security for the protocol (paper, Section 6, scope
of the theorem): a tag binds a
message to a key holder; it does not make a passive protocol robust against a
deviating party.

What is tested (tests/test_hmac_auth.py): canonical bytes for a fixed object,
key-order independence, agreement with a direct hashlib/hmac computation,
tamper and wrong-key rejection, malformed-hex rejection, payload framing,
rejection of NaN and of an empty key.

Design limits: float canonicalisation follows Python's shortest round-trip
repr, so cross-language headers should carry integers and strings only (the
C++ mirror emits identical bytes for those types). Error messages start with
the stable codes HMAC_KEY and HMAC_CANONICAL_JSON.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

HASH_NAME = "sha256"
TAG_HEX_LEN = 64


def _check_key(key) -> bytes:
    if not isinstance(key, (bytes, bytearray, memoryview)):
        raise TypeError(f"HMAC_KEY: key must be bytes, got {type(key).__name__}")
    key = bytes(key)
    if len(key) == 0:
        raise ValueError("HMAC_KEY: key must not be empty (32 random bytes recommended)")
    return key


def canonical_json(obj: Any) -> bytes:
    """Sorted keys, separators (',', ':'), UTF-8, non-finite floats rejected."""
    try:
        text = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"HMAC_CANONICAL_JSON: object is not canonicalisable: {exc}") from None
    return text.encode("utf-8")


def mac_input(obj: Any, payload: bytes) -> bytes:
    """Framing for header + payload: len(header) 8-byte big-endian || header || payload."""
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError(f"HMAC_CANONICAL_JSON: payload must be bytes, got {type(payload).__name__}")
    header = canonical_json(obj)
    return len(header).to_bytes(8, "big") + header + bytes(payload)


def tag(key: bytes, obj: Any) -> str:
    """Lowercase hex HMAC-SHA256 of canonical_json(obj)."""
    key = _check_key(key)
    return hmac.new(key, canonical_json(obj), hashlib.sha256).hexdigest()


def verify(key: bytes, obj: Any, tag_hex: str) -> bool:
    """Constant-time comparison of the recomputed tag with tag_hex; malformed input yields False."""
    key = _check_key(key)
    expected = hmac.new(key, canonical_json(obj), hashlib.sha256).digest()
    return _compare_hex(expected, tag_hex)


def tag_with_payload(key: bytes, obj: Any, payload: bytes) -> str:
    """Hex HMAC-SHA256 over mac_input(obj, payload)."""
    key = _check_key(key)
    return hmac.new(key, mac_input(obj, payload), hashlib.sha256).hexdigest()


def verify_with_payload(key: bytes, obj: Any, payload: bytes, tag_hex: str) -> bool:
    key = _check_key(key)
    expected = hmac.new(key, mac_input(obj, payload), hashlib.sha256).digest()
    return _compare_hex(expected, tag_hex)


def _compare_hex(expected: bytes, tag_hex) -> bool:
    if not isinstance(tag_hex, str) or len(tag_hex) != TAG_HEX_LEN:
        return False
    try:
        given = bytes.fromhex(tag_hex)
    except ValueError:
        return False
    return hmac.compare_digest(expected, given)
