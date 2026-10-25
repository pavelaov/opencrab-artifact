"""Isolated dealer, recipient-bound bundles and the party-side material client (paper, Appendix B, DealerDelivery).

Two roles live here:

* :class:`Dealer` (generation side; runs inside the dealer process of
  ``platform/dealer_service.py``). It owns the master randomness, produces
  every material type of ``MATERIAL_TYPES`` as a PAIR of recipient-bound bundles
  ``(header0, payload0, header1, payload1)`` and seals each bundle for its
  recipient with ``hmac_auth.tag_with_payload`` under that recipient's key.
  Payloads are raw little-endian uint64 arrays or the ``to_bytes`` forms of
  the lowlevel material dataclasses; pickle is never used.
* :class:`DealerClient` (party side). ``ingest`` verifies the tag, the
  recipient, the numeric contract hash and the material_id uniqueness, decodes
  the payload into a typed pool and registers it in the party's
  :class:`MaterialLedger`. Every ``take_*`` reserves and spends the touched
  bundle ids in the ledger BEFORE returning the material, so the SPENT record
  is durable before any secret-dependent message can use it (paper,
  Section 4.3, material lifecycle and commit).

Modes. ``ISOLATED_RECIPIENT_BUNDLES`` (default, the only eligible mode): the
caller is the dealer process, which routes bundle i only to recipient i.
``INSECURE_TEST_ONLY_SHARED_SEED``: unit tests only; both bundles are returned
to one caller and every header carries ``insecure: true``; a client that
ingests such a bundle records it and reports ``security_eligible == False``
(records are DIAGNOSTIC). No mode is a replay in which the parties
regenerate the dealer's randomness from a common seed: a party never receives
the master seed, the peer bundle, or a reconstructible mask.

Insecure labelling. The ``insecure`` flag of a header is not
derived from the mode alone: a ``Dealer`` built with a caller-supplied
``rng`` labels every bundle ``insecure: true`` whatever the mode, because
whoever seeded that generator can replay the dealer's randomness. Only a
dealer that seeds itself from the OS CSPRNG (``rng=None``), or a caller that
passes a ``csprng.SecureRandom.from_os()`` stream, produces eligible bundles.
A numpy Generator is never eligible (a bundle is a window of raw generator
outputs and the PCG64 state is recoverable from outputs; paper, Appendix D,
Disclosure).
On the receiving side insecure bundles are refused with
``BUNDLE_INSECURE_MODE`` unless the test-only switch is on
(``insecure_material_allowed``): an explicit ``allow_insecure=True``, the
environment variable ``OPENCRAB2_TEST_ALLOW_INSECURE_DEALER=1``, or the
fact that this very process hosts an insecure ``Dealer`` (an in-process unit
test harness, where the process already holds the dealer's randomness and a
refusal would protect nothing). A platform party process never builds a
``Dealer``, so it fails closed.

Bundle header (canonical JSON, all public): ``profile_id, setup_id,
model_version, numeric_contract_hash, recipient, gate_id, material_type,
material_id, count, bytes`` plus ``params`` (shape/n_bits/f/L/t/mat_id),
``parts`` (names of the payload parts in order),
``dealer_mode`` and ``insecure``. The payload of DCF-based material is the
``to_bytes()`` form of the lowlevel container named in ``parts``; its
embedded DCF keys use the binding wire layout (``DCFKeyBatch.to_bytes``), so
``header["bytes"]`` (the exact payload length) is larger than the charged
material size of ``material_estimate`` (the declared size formulas).

Dealer dialogue (the same frames in Python and C++). The frames a party sends and reads are
built and checked here (``hello_frame``, ``dealer_auth_tag``,
``auth_frame``, ``req_frame``, ``parse_challenge``, ``parse_bundle_frame``);
``platform/dealer_service.py`` owns the sockets:

    client  HELLO{recipient, setup_id, contract_hash}
    server  CHALLENGE{nonce}                         (32 random bytes)
    client  AUTH{tag = hex(HMAC-SHA256(key_recipient, nonce))}
    server  AUTH_OK{recipient}
    client  REQ{material_type, params, count}        (optional gate_id, request_id)
    server  BUNDLE{header, tag}  then one bytes frame with the payload
    server  ERROR{code}                              on any failure

What is tested (tests/test_dealer.py): every material type reconstructs its
correlation from the two payloads; sealing and ingestion; rejection of wrong
recipient, wrong key, tampered payload, wrong contract hash, duplicate id,
foreign version; the insecure labelling of seeded dealers and the fail-closed
default of the client; the dialogue frames; ledger state after takes; stalls;
partial bundle consumption; byte accounting versus ``material_estimate``; the
k-ary tournament formulas. Design limit: the ledger accounts material at
bundle granularity (a bundle is SPENT when first touched; its remainder is
held in memory and lost on a crash, which is the safe direction), so
``MaterialLedger.stock`` is a lower bound on exact element stock;
``DealerClient.stock`` is exact. The AUTH tag and the bundle tags are
shared-key integrity tools for the trusted-offline profile, not a proof of
active security.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import os
import secrets
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from opencrab2.errors import (ContractViolation, MaterialError, MaterialReuse, MaterialStall,
                              MaterialVersionMismatch)
from opencrab2.highlevel.contract import NumericContract
from opencrab2.highlevel.ledger import MaterialLedger
from opencrab2.lowlevel import hmac_auth, ring, sharing
from opencrab2.lowlevel.csprng import SecureRandom
from opencrab2.lowlevel.dcf import concat_keys, key_bytes_per_element
from opencrab2.lowlevel.drelu import DReLUMaterial, drelu_bytes_per_element, drelu_material
from opencrab2.lowlevel.ring_to_field import R2FMaterial, r2f_bytes_per_element, r2f_material
from opencrab2.lowlevel.trunc import TruncMaterial, trunc_bytes_per_element, trunc_material

MODE_ISOLATED = "ISOLATED_RECIPIENT_BUNDLES"
MODE_INSECURE = "INSECURE_TEST_ONLY_SHARED_SEED"
MODES = (MODE_ISOLATED, MODE_INSECURE)
PROFILE_ID = "SH2P-TrustedOffline-v1"
DEFAULT_T = 68720050177   # admitted BFV plaintext modulus (odd prime, about 2^36)

MATERIAL_TYPES = ("beaver", "matrix_triple", "wmask_mask", "wmask_call", "drelu",
                  "trunc_faithful", "ring_to_field", "vector_share", "kary_select")
HEADER_KEYS = ("profile_id", "setup_id", "model_version", "numeric_contract_hash", "recipient",
               "gate_id", "material_type", "material_id", "count", "bytes", "params",
               "dealer_mode", "insecure")
VERSION_KEYS = ("setup_id", "model_version", "numeric_contract_hash", "profile_id")
# Header field ``parts``: names of the payload parts in payload order. The names are
# the ones the C++ dealer service writes; DCF-based material is one lowlevel container per bundle.
PAYLOAD_PARTS: Dict[str, Tuple[str, ...]] = {
    "beaver": ("a", "b", "c"),
    "kary_select": ("a", "b", "c"),
    "matrix_triple": ("A", "B", "C"),
    "wmask_mask": ("B_share",),
    "wmask_call": ("A", "C"),
    "vector_share": ("share",),
    "drelu": ("drelu_material",),
    "trunc_faithful": ("trunc_material",),
    "ring_to_field": ("r2f_material",),
}

# --------------------------------------------------------------------------
# Test-only switch for insecure material
# --------------------------------------------------------------------------
ENV_ALLOW_INSECURE = "OPENCRAB2_TEST_ALLOW_INSECURE_DEALER"
_TEST_ONLY_STATE = {"in_process_insecure_dealer": False}


def insecure_material_allowed(explicit: Optional[bool] = None) -> bool:
    """Whether this process may accept bundles labelled insecure (test-only switch).

    ``explicit`` (the ``allow_insecure`` argument of a client) wins when it is
    not None. Otherwise the switch is on only when the environment variable
    ``OPENCRAB2_TEST_ALLOW_INSECURE_DEALER`` equals ``1`` or when this process
    itself hosts an insecure ``Dealer`` (in-process unit-test harness). A
    platform party process has neither, so it refuses.
    """
    if explicit is not None:
        return bool(explicit)
    return bool(_TEST_ONLY_STATE["in_process_insecure_dealer"]) or os.environ.get(ENV_ALLOW_INSECURE, "") == "1"


# --------------------------------------------------------------------------
# Dealer dialogue frames, party side
# --------------------------------------------------------------------------
AUTH_NONCE_BYTES = 32
_TAG_HEX_LEN = 64


def dealer_auth_tag(hmac_key: bytes, nonce: bytes) -> str:
    """AUTH tag: lowercase hex HMAC-SHA256(key_recipient, nonce) over the raw 32 nonce bytes.

    The nonce length is fixed: a bundle tag authenticates ``8-byte length ||
    canonical header || payload`` with a header of more than a hundred bytes,
    so a 32-byte challenge can never be mistaken for (or replayed as) a
    bundle. An integrity tool for the trusted-offline profile, not a proof.
    """
    if not isinstance(hmac_key, (bytes, bytearray, memoryview)) or len(bytes(hmac_key)) == 0:
        raise MaterialError("DEALER_KEYS: the recipient HMAC key must be non-empty bytes")
    if not isinstance(nonce, (bytes, bytearray, memoryview)) or len(bytes(nonce)) != AUTH_NONCE_BYTES:
        raise MaterialError(f"DEALER_BAD_MESSAGE: the challenge nonce must be {AUTH_NONCE_BYTES} bytes")
    return hmac.new(bytes(hmac_key), bytes(nonce), hashlib.sha256).hexdigest()


def verify_dealer_auth_tag(hmac_key: bytes, nonce: bytes, tag_hex: Any) -> bool:
    """Constant-time check of an AUTH tag; malformed input yields False."""
    if not isinstance(tag_hex, str) or len(tag_hex) != _TAG_HEX_LEN:
        return False
    try:
        given = bytes.fromhex(tag_hex)
    except ValueError:
        return False
    expected = hmac.new(bytes(hmac_key), bytes(nonce), hashlib.sha256).digest()
    return hmac.compare_digest(expected, given)


def hello_frame(recipient: int, setup_id: Optional[str], contract_hash: Optional[str]) -> dict:
    """First frame of the dialogue. An empty ``setup_id`` means "not pinned by the client"."""
    if recipient not in (0, 1):
        raise MaterialError(f"DEALER_PARAMS: recipient must be 0 or 1, got {recipient!r}")
    return {"type": "HELLO", "recipient": int(recipient), "setup_id": str(setup_id or ""),
            "contract_hash": str(contract_hash or "")}


def raise_if_error_frame(msg: Any, context: str = "the dealer refused the request") -> None:
    """Raise ``MaterialError`` with the stable code of an ERROR{code} frame; no-op for any other frame."""
    if isinstance(msg, dict) and msg.get("type") == "ERROR":
        detail = msg.get("message")
        detail = detail[:200] if isinstance(detail, str) and detail else context
        code = _wire_code(msg)
        if detail.startswith(code + ":"):
            detail = detail[len(code) + 1:].strip()
        raise MaterialError(f"{code}: {detail}")


def parse_challenge(msg: Any) -> bytes:
    """Validate a CHALLENGE frame and return its nonce; an ERROR frame raises its code."""
    raise_if_error_frame(msg, "the dealer refused the connection")
    if not isinstance(msg, dict) or msg.get("type") != "CHALLENGE":
        raise MaterialError("DEALER_BAD_MESSAGE: expected CHALLENGE")
    nonce = msg.get("nonce")
    if not isinstance(nonce, (bytes, bytearray)) or len(nonce) != AUTH_NONCE_BYTES:
        raise MaterialError(f"DEALER_BAD_MESSAGE: CHALLENGE must carry a {AUTH_NONCE_BYTES}-byte nonce")
    return bytes(nonce)


def auth_frame(hmac_key: bytes, nonce: bytes) -> dict:
    return {"type": "AUTH", "tag": dealer_auth_tag(hmac_key, nonce)}


def req_frame(material_type: str, params: Optional[dict], count: int, gate_id: Optional[str] = None,
              request_id: Optional[str] = None) -> dict:
    """REQ{material_type, params, count} with the optional ``gate_id`` and ``request_id`` keys.

    ``request_id`` names a request so that both recipients obtain the two
    halves of the SAME pairs regardless of arrival order; without it the
    service pairs identical REQ frames first come, first served.
    """
    count = int(count)
    if count < 1:
        raise MaterialError("DEALER_BAD_REQUEST: count must be >= 1")
    msg: Dict[str, Any] = {"type": "REQ", "material_type": str(material_type), "params": dict(params or {}),
                           "count": count}
    if gate_id is not None:
        msg["gate_id"] = str(gate_id)
    if request_id is not None:
        msg["request_id"] = str(request_id)
    return msg


def _wire_code(msg: dict) -> str:
    """The stable code of an ERROR frame; anything that is not an upper-case identifier is replaced."""
    code = msg.get("code")
    if isinstance(code, str) and code and all(ch.isupper() or ch.isdigit() or ch == "_" for ch in code) \
            and code[0].isupper():
        return code
    return "DEALER_ERROR"


def parse_bundle_frame(msg: Any) -> Tuple[dict, str, int]:
    """Validate a BUNDLE{header, tag} frame: returns (header, tag, remaining); ERROR raises its code.

    ``remaining`` (0 when absent) is the number of further bundles the same
    REQ still has for this recipient. Nothing here is trusted before
    ``DealerClient.ingest`` verified the tag.
    """
    raise_if_error_frame(msg)
    if not isinstance(msg, dict) or msg.get("type") != "BUNDLE":
        raise MaterialError("DEALER_BAD_MESSAGE: expected BUNDLE")
    header = msg.get("header")
    tag = msg.get("tag")
    if not isinstance(header, dict) or not isinstance(tag, str):
        raise MaterialError("DEALER_BAD_MESSAGE: BUNDLE must carry header (dict) and tag (str)")
    remaining = msg.get("remaining", 0)
    if isinstance(remaining, bool) or not isinstance(remaining, int) or remaining < 0:
        raise MaterialError("DEALER_BAD_MESSAGE: BUNDLE remaining must be a non-negative int")
    return header, tag, int(remaining)


# --------------------------------------------------------------------------
# Party-side material objects
# --------------------------------------------------------------------------
@dataclass
class BeaverShare:
    """This party's shares of elementwise triples a*b = c of ``shape``."""
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    shape: tuple

    @property
    def count(self) -> int:
        return int(self.a.size)

    def nbytes(self) -> int:
        return 24 * self.count


@dataclass
class MatrixTripleShare:
    """This party's shares of A (n,k), B (k,m), C = A @ B (n,m) over Z_{2^64}."""
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray

    def nbytes(self) -> int:
        return 8 * (self.a.size + self.b.size + self.c.size)


@dataclass
class WMaskCall:
    """Per-call fresh material of linear.wmask: A (n,d) and C = A @ B (n,m) shares."""
    a: np.ndarray
    c: np.ndarray
    mat_id: str

    def nbytes(self) -> int:
        return 8 * (self.a.size + self.c.size)


# --------------------------------------------------------------------------
# Raw array codec (little-endian uint64, shapes carried by the header)
# --------------------------------------------------------------------------
def _u64_bytes(*arrays: np.ndarray) -> bytes:
    return b"".join(np.ascontiguousarray(np.asarray(a, dtype=np.uint64), dtype="<u8").tobytes() for a in arrays)


def _u64_split(buf: bytes, shapes: Sequence[tuple]) -> List[np.ndarray]:
    out = []
    off = 0
    for shp in shapes:
        n = _numel(shp)
        end = off + 8 * n
        if end > len(buf):
            raise MaterialError("BUNDLE_PAYLOAD_MISMATCH: payload shorter than the declared shapes")
        arr = np.frombuffer(buf, dtype="<u8", count=n, offset=off).astype(np.uint64).reshape(tuple(shp))
        out.append(arr)
        off = end
    if off != len(buf):
        raise MaterialError("BUNDLE_PAYLOAD_MISMATCH: payload longer than the declared shapes")
    return out


def _numel(shape: Sequence[int]) -> int:
    n = 1
    for s in shape:
        s = int(s)
        if s < 0:
            raise ContractViolation(f"MATERIAL_SHAPE: negative dimension in {tuple(shape)}")
        n *= s
    return n


def _shape_tuple(shape: Any) -> tuple:
    if isinstance(shape, (int, np.integer)):
        return (int(shape),)
    try:
        return tuple(int(s) for s in shape)
    except TypeError:
        raise ContractViolation(f"MATERIAL_SHAPE: shape {shape!r} is not a sequence of ints") from None


def _check_mode(mode: str) -> str:
    if mode not in MODES:
        raise MaterialError(f"DEALER_MODE: unknown mode {mode!r}; admitted modes {MODES}")
    return mode


# --------------------------------------------------------------------------
# Material concatenation helpers (header fields + dcf.concat_keys)
# --------------------------------------------------------------------------
def _concat_drelu(a: DReLUMaterial, b: DReLUMaterial) -> DReLUMaterial:
    if a.n_bits != b.n_bits or a.party != b.party or a.key_p.P != b.key_p.P:
        raise MaterialError("MATERIAL_POOL: incompatible DReLU material concatenated")
    return DReLUMaterial(np.concatenate([a.r_share, b.r_share]), concat_keys(a.key_p, b.key_p),
                         concat_keys(a.key_q, b.key_q), np.concatenate([a.c_b, b.c_b]),
                         np.concatenate([a.c_rb, b.c_rb]), a.n_bits, a.party)


def _concat_trunc(a: TruncMaterial, b: TruncMaterial) -> TruncMaterial:
    if a.f != b.f or a.L != b.L or a.party != b.party:
        raise MaterialError("MATERIAL_POOL: incompatible truncation material concatenated")
    return TruncMaterial(np.concatenate([a.r_share, b.r_share]), np.concatenate([a.r_high_share, b.r_high_share]),
                         concat_keys(a.key_wrap, b.key_wrap), concat_keys(a.key_borrow, b.key_borrow),
                         a.f, a.L, a.party)


def _concat_r2f(a: R2FMaterial, b: R2FMaterial) -> R2FMaterial:
    if a.t != b.t or a.party != b.party:
        raise MaterialError("MATERIAL_POOL: incompatible ring-to-field material concatenated")
    return R2FMaterial(np.concatenate([a.r_share, b.r_share]), np.concatenate([a.r_mod_t_share, b.r_mod_t_share]),
                       concat_keys(a.key_A, b.key_A), concat_keys(a.key_C, b.key_C),
                       np.concatenate([a.c_share, b.c_share]), a.t, a.party)


# --------------------------------------------------------------------------
# Dealer (generation side)
# --------------------------------------------------------------------------
class Dealer:
    """Generates recipient-bound material pairs; lives in the dealer process.

    ``rng`` defaults to ``csprng.SecureRandom.from_os()``: a ChaCha20 stream
    keyed with 32 OS bytes. The key is never stored, logged or
    exported by this class. A recipient's bundle is a window of raw generator
    outputs, so the generator itself must be cryptographic: a numpy Generator
    (PCG64) is accepted for deterministic fixtures only and every bundle of
    such a dealer is labelled ``insecure: true`` whatever the mode and
    whatever ``rng_is_os_entropy`` says. A caller-supplied ``SecureRandom``
    is eligible only when it was keyed by ``SecureRandom.from_os()`` (its
    ``os_seeded`` flag); ``rng_is_os_entropy=True`` with any other generator
    is refused with ``DEALER_PARAMS``. ``setup_id`` and ``model_version`` are the public binding carried
    by every header.
    """

    def __init__(self, contract: NumericContract, rng: Optional[Any] = None,
                 mode: str = MODE_ISOLATED, *, setup_id: str = "setup-0",
                 model_version: str = "unversioned", profile_id: str = PROFILE_ID,
                 rng_is_os_entropy: bool = False):
        if not isinstance(contract, NumericContract):
            raise ContractViolation("DEALER_PARAMS: contract must be a NumericContract")
        self.contract = contract
        self.contract_hash = contract.contract_hash()
        self.mode = _check_mode(mode)
        if rng is None:
            rng = SecureRandom.from_os()
            self.rng_from_os_entropy = True
        elif isinstance(rng, SecureRandom):
            if rng_is_os_entropy and not rng.os_seeded:
                raise ContractViolation("DEALER_PARAMS: rng_is_os_entropy needs SecureRandom.from_os()")
            self.rng_from_os_entropy = bool(rng.os_seeded)
        else:
            if not isinstance(rng, np.random.Generator):
                raise ContractViolation("DEALER_PARAMS: rng must be a SecureRandom, a numpy Generator or None")
            if rng_is_os_entropy:
                # a statistical generator leaks its state through the bundles, whatever seeded it
                raise ContractViolation("DEALER_PARAMS: a numpy Generator is never eligible; "
                                        "pass rng=None or SecureRandom.from_os()")
            self.rng_from_os_entropy = False
        self.rng = rng
        # Insecure when the mode says so OR when the randomness is replayable by the caller.
        self._insecure = (self.mode == MODE_INSECURE) or not self.rng_from_os_entropy
        if self._insecure:
            _TEST_ONLY_STATE["in_process_insecure_dealer"] = True
        self.setup_id = str(setup_id)
        self.model_version = str(model_version)
        self.profile_id = str(profile_id)
        self._counter = 0
        self._wmask_B: Dict[str, np.ndarray] = {}      # dealer-private per-matrix masks
        self.produced: Dict[str, List[int]] = {t: [0, 0] for t in MATERIAL_TYPES}
        self._lock = threading.RLock()
        self._id_nonce = secrets.token_hex(4)

    # -- binding ---------------------------------------------------------------------
    def bind(self, setup_id: Optional[str] = None, model_version: Optional[str] = None) -> None:
        with self._lock:
            if setup_id is not None:
                self.setup_id = str(setup_id)
            if model_version is not None:
                self.model_version = str(model_version)

    @property
    def insecure(self) -> bool:
        """True for INSECURE_TEST_ONLY_SHARED_SEED and for any caller-supplied rng not declared OS entropy."""
        return self._insecure

    # -- header / pairing ---------------------------------------------------------------
    def _new_material_id(self, material_type: str) -> str:
        self._counter += 1
        return f"{material_type}-{self.setup_id}-{self._id_nonce}-{self._counter:08d}"

    def _header(self, recipient: int, material_type: str, material_id: str, count: int,
                nbytes: int, params: dict, gate_id: str) -> dict:
        return {
            "profile_id": self.profile_id,
            "setup_id": self.setup_id,
            "model_version": self.model_version,
            "numeric_contract_hash": self.contract_hash,
            "recipient": int(recipient),
            "gate_id": str(gate_id),
            "material_type": material_type,
            "material_id": material_id,
            "count": int(count),
            "bytes": int(nbytes),
            "params": params,
            "parts": list(PAYLOAD_PARTS[material_type]),
            "dealer_mode": self.mode,
            "insecure": self.insecure,
        }

    def _pair(self, material_type: str, payload0: bytes, payload1: bytes, count: int,
              params: dict, gate_id: str) -> Tuple[dict, bytes, dict, bytes]:
        mid = self._new_material_id(material_type)
        h0 = self._header(0, material_type, mid, count, len(payload0), params, gate_id)
        h1 = self._header(1, material_type, mid, count, len(payload1), params, gate_id)
        p = self.produced[material_type]
        p[0] += int(count)
        p[1] += len(payload0)
        return h0, payload0, h1, payload1

    # -- material types -------------------------------------------------------------------
    def make_beaver(self, shape: Any, count: int = 1, gate_id: str = "*",
                    material_type: str = "beaver") -> List[Tuple[dict, bytes, dict, bytes]]:
        """``count`` bundles, each holding elementwise triples of ``shape``."""
        if material_type not in ("beaver", "kary_select"):
            raise MaterialError(f"DEALER_PARAMS: Beaver-like type {material_type!r} unknown")
        shp = _shape_tuple(shape)
        n = _numel(shp)
        if n < 1 or int(count) < 1:
            raise ContractViolation("DEALER_PARAMS: Beaver shape must be non-empty and count >= 1")
        out = []
        with self._lock:
            for _ in range(int(count)):
                a = ring.rand_u64(shp, self.rng)
                b = ring.rand_u64(shp, self.rng)
                c = ring.mul(a, b)
                a0, a1 = sharing.share(a, self.rng)
                b0, b1 = sharing.share(b, self.rng)
                c0, c1 = sharing.share(c, self.rng)
                params = {"shape": list(shp), "elements": n}
                out.append(self._pair(material_type, _u64_bytes(a0, b0, c0), _u64_bytes(a1, b1, c1),
                                      n, params, gate_id))
        return out

    def make_kary_select(self, shape: Any, count: int = 1, gate_id: str = "*") -> List[Tuple[dict, bytes, dict, bytes]]:
        """Beaver triples reserved for the selects of the k-ary tournaments."""
        return self.make_beaver(shape, count, gate_id, material_type="kary_select")

    def make_matrix_triples(self, shape_a: Any, shape_b: Any, count: int = 1,
                            gate_id: str = "*") -> List[Tuple[dict, bytes, dict, bytes]]:
        """``count`` triples A (n,k), B (k,m), C = A @ B exactly mod 2^64."""
        sa = _shape_tuple(shape_a)
        sb = _shape_tuple(shape_b)
        if len(sa) != 2 or len(sb) != 2 or sa[1] != sb[0]:
            raise ContractViolation(f"DEALER_PARAMS: matrix triple needs (n,k) and (k,m), got {sa} and {sb}")
        if int(count) < 1:
            raise ContractViolation("DEALER_PARAMS: count must be >= 1")
        out = []
        with self._lock:
            for _ in range(int(count)):
                A = ring.rand_u64(sa, self.rng)
                B = ring.rand_u64(sb, self.rng)
                C = ring.ring_matmul(A, B)
                a0, a1 = sharing.share(A, self.rng)
                b0, b1 = sharing.share(B, self.rng)
                c0, c1 = sharing.share(C, self.rng)
                params = {"shape_a": list(sa), "shape_b": list(sb)}
                out.append(self._pair("matrix_triple", _u64_bytes(a0, b0, c0), _u64_bytes(a1, b1, c1),
                                      1, params, gate_id))
        return out

    def make_drelu(self, M: int, n_bits: Optional[int] = None, with_product: bool = True,
                   gate_id: str = "*") -> List[Tuple[dict, bytes, dict, bytes]]:
        """One bundle of M one-round DReLU/ReLU gates (two DCF keys + 3 shares each)."""
        n = int(self.contract.dcf_domain_bits if n_bits is None else n_bits)
        M = int(M)
        if M < 1:
            raise ContractViolation("DEALER_PARAMS: M must be >= 1")
        if not (2 <= n <= 64):
            raise ContractViolation(f"DEALER_PARAMS: n_bits must be in [2, 64], got {n}")
        with self._lock:
            m0, m1 = drelu_material(M, n, self.rng, with_product=bool(with_product))
            params = {"n_bits": n, "with_product": bool(with_product), "P": 2 if with_product else 1}
            return [self._pair("drelu", m0.to_bytes(), m1.to_bytes(), M, params, gate_id)]

    def make_trunc(self, M: int, f: Optional[int] = None, L: Optional[int] = None,
                   gate_id: str = "*") -> List[Tuple[dict, bytes, dict, bytes]]:
        """One bundle of M faithful truncation gates with the signed offset (paper, Section 5.1, TruncFaithful)."""
        f = int(self.contract.frac_bits if f is None else f)
        L = int(self.contract.logical_bits if L is None else L)
        M = int(M)
        if M < 1:
            raise ContractViolation("DEALER_PARAMS: M must be >= 1")
        if not (1 <= f <= L - 1 <= 63):
            raise ContractViolation(f"DEALER_PARAMS: need 1 <= f <= L-1 <= 63, got f={f}, L={L}")
        with self._lock:
            t0, t1 = trunc_material(M, f, L, self.rng)
            params = {"f": f, "L": L}
            return [self._pair("trunc_faithful", t0.to_bytes(), t1.to_bytes(), M, params, gate_id)]

    def make_r2f(self, M: int, t: int = DEFAULT_T, gate_id: str = "*") -> List[Tuple[dict, bytes, dict, bytes]]:
        """One bundle of M ring-to-field gates for the odd prime t (paper, Section 5.2, RingToField)."""
        M = int(M)
        if M < 1:
            raise ContractViolation("DEALER_PARAMS: M must be >= 1")
        with self._lock:
            r0, r1 = r2f_material(M, int(t), self.rng)
            params = {"t": int(t)}
            return [self._pair("ring_to_field", r0.to_bytes(), r1.to_bytes(), M, params, gate_id)]

    def make_wmask(self, mat_id: str, W_enc: Any, gate_id: str = "*") -> List[Tuple[dict, bytes, dict, bytes]]:
        """Per-matrix reusable mask B (d,m): shares to both parties, B kept dealer-private.

        ``W_enc`` is used for its SHAPE only (an array or a shape tuple); the
        dealer never reads the model weights. E = W - B is produced online by
        the one-round install protocol of the Party (P1 sends B1, P0 returns
        W - B0 - B1), which is why a fresh B is generated on every call: a
        new model version must never reuse B (one B under W0 and W1 would hand
        the client E1 - E0 = W1 - W0).
        """
        if isinstance(W_enc, np.ndarray):
            shp = tuple(int(s) for s in W_enc.shape)
        else:
            shp = _shape_tuple(W_enc)
        if len(shp) != 2 or _numel(shp) < 1:
            raise ContractViolation(f"DEALER_PARAMS: wmask matrix shape must be (d, m), got {shp}")
        mat_id = str(mat_id)
        with self._lock:
            B = ring.rand_u64(shp, self.rng)
            b0, b1 = sharing.share(B, self.rng)
            self._wmask_B[mat_id] = B
            params = {"mat_id": mat_id, "shape": list(shp), "model_version": self.model_version}
            return [self._pair("wmask_mask", _u64_bytes(b0), _u64_bytes(b1), 1, params, gate_id)]

    def make_wmask_calls(self, mat_id: str, n_rows: int, count: int = 1,
                         gate_id: str = "*") -> List[Tuple[dict, bytes, dict, bytes]]:
        """``count`` per-call bundles A (n_rows,d), C = A @ B (n_rows,m) for ``mat_id``."""
        mat_id = str(mat_id)
        n_rows = int(n_rows)
        if n_rows < 1 or int(count) < 1:
            raise ContractViolation("DEALER_PARAMS: n_rows and count must be >= 1")
        with self._lock:
            B = self._wmask_B.get(mat_id)
            if B is None:
                raise MaterialError(f"WMASK_MASK_MISSING: no mask registered for {mat_id!r}; call make_wmask first")
            d, m = B.shape
            out = []
            for _ in range(int(count)):
                A = ring.rand_u64((n_rows, d), self.rng)
                C = ring.ring_matmul(A, B)
                a0, a1 = sharing.share(A, self.rng)
                c0, c1 = sharing.share(C, self.rng)
                params = {"mat_id": mat_id, "n_rows": n_rows, "d": int(d), "m": int(m)}
                out.append(self._pair("wmask_call", _u64_bytes(a0, c0), _u64_bytes(a1, c1), 1, params, gate_id))
            return out

    def make_vector_share(self, name: str, v_enc: Any, gate_id: str = "*") -> List[Tuple[dict, bytes, dict, bytes]]:
        """Additive shares of an encoded vector handed to the dealer in the clear. INSECURE HARNESS ONLY.

        The dealer of profile SH2P-TrustedOffline-v1 never receives a model value: server-private
        vectors are shared online by P0 with ``Party.share_input``. An eligible dealer (isolated mode,
        OS-keyed CSPRNG) therefore refuses this request with ``DEALER_PLAINTEXT_REFUSED``; only a dealer
        that already labels its bundles insecure (unit-test harness) serves it.
        """
        if not self.insecure:
            raise MaterialError("DEALER_PLAINTEXT_REFUSED: vector_share hands a plaintext vector to the dealer; "
                                "an eligible dealer refuses it (share server-private vectors with share_input)")
        v = np.asarray(v_enc, dtype=np.uint64)
        if v.size < 1:
            raise ContractViolation("DEALER_PARAMS: vector must be non-empty")
        with self._lock:
            v0, v1 = sharing.share(v, self.rng)
            params = {"name": str(name), "shape": list(v.shape)}
            return [self._pair("vector_share", _u64_bytes(v0), _u64_bytes(v1), int(v.size), params, gate_id)]

    def make(self, material_type: str, params: dict, count: int = 1, gate_id: str = "*") -> List[Tuple[dict, bytes, dict, bytes]]:
        """Dispatch a provisioning request ``{material_type, params, count}`` (dealer service)."""
        params = dict(params or {})
        count = int(count)
        if material_type in ("beaver", "kary_select"):
            return self.make_beaver(params["shape"], count, gate_id, material_type=material_type)
        if material_type == "matrix_triple":
            return self.make_matrix_triples(params["shape_a"], params["shape_b"], count, gate_id)
        if material_type == "drelu":
            out = []
            for _ in range(count):
                out += self.make_drelu(params["M"], params.get("n_bits"), bool(params.get("with_product", True)), gate_id)
            return out
        if material_type == "trunc_faithful":
            out = []
            for _ in range(count):
                out += self.make_trunc(params["M"], params.get("f"), params.get("L"), gate_id)
            return out
        if material_type == "ring_to_field":
            out = []
            for _ in range(count):
                out += self.make_r2f(params["M"], int(params.get("t", DEFAULT_T)), gate_id)
            return out
        if material_type == "wmask_mask":
            return self.make_wmask(params["mat_id"], params["shape"], gate_id)
        if material_type == "wmask_call":
            return self.make_wmask_calls(params["mat_id"], params["n_rows"], count, gate_id)
        if material_type == "vector_share":
            return self.make_vector_share(params["name"], np.asarray(params["values"], dtype=np.uint64), gate_id)
        raise MaterialError(f"DEALER_PARAMS: unknown material type {material_type!r}")

    # -- sealing -----------------------------------------------------------------------------
    def seal(self, recipient: int, header: dict, payload: bytes, hmac_key: bytes) -> Tuple[dict, bytes, str]:
        """Authenticate ``(header, payload)`` for ``recipient`` under that recipient's key."""
        if int(header.get("recipient", -1)) != int(recipient):
            raise MaterialError(f"BUNDLE_WRONG_RECIPIENT: header is for {header.get('recipient')!r}, sealing for {recipient}")
        if int(header.get("bytes", -1)) != len(payload):
            raise MaterialError("BUNDLE_MALFORMED: header bytes does not match the payload length")
        tag = hmac_auth.tag_with_payload(hmac_key, header, payload)
        return header, bytes(payload), tag

    def produced_summary(self) -> Dict[str, Tuple[int, int]]:
        with self._lock:
            return {k: (v[0], v[1]) for k, v in self.produced.items() if v[0] or v[1]}


# --------------------------------------------------------------------------
# Party-side pools
# --------------------------------------------------------------------------
class _Chunk:
    __slots__ = ("material_id", "arrays", "offset", "size", "obj")

    def __init__(self, material_id: str, size: int, arrays: Optional[tuple] = None, obj: Any = None):
        self.material_id = material_id
        self.arrays = arrays        # tuple of flat arrays (Beaver-like)
        self.obj = obj              # sliceable material object (DReLU/trunc/r2f)
        self.offset = 0
        self.size = int(size)

    @property
    def remaining(self) -> int:
        return self.size - self.offset


class _SlicePool:
    """FIFO of sliceable chunks; a chunk's id is spent when first touched."""

    def __init__(self):
        self.chunks: Deque[_Chunk] = deque()
        self.total = 0

    def add(self, chunk: _Chunk) -> None:
        self.chunks.append(chunk)
        self.total += chunk.size

    def ids_for(self, n: int) -> List[str]:
        """Material ids of untouched chunks a take of ``n`` elements will touch."""
        if n > self.total:
            raise MaterialStall(f"MATERIAL_STALL: need {n} elements, have {self.total}")
        ids = []
        need = n
        for ch in self.chunks:
            if need <= 0:
                break
            if ch.offset == 0:
                ids.append(ch.material_id)
            need -= ch.remaining
        return ids

    def take_parts(self, n: int) -> List[Tuple[_Chunk, int, int]]:
        """Pop ``n`` elements as (chunk, lo, hi) parts; call after the ledger accepted."""
        if n > self.total:
            raise MaterialStall(f"MATERIAL_STALL: need {n} elements, have {self.total}")
        parts = []
        need = n
        while need > 0:
            ch = self.chunks[0]
            lo = ch.offset
            k = min(need, ch.remaining)
            hi = lo + k
            parts.append((ch, lo, hi))
            ch.offset = hi
            need -= k
            if ch.remaining == 0:
                self.chunks.popleft()
        self.total -= n
        return parts

    def untouched_ids(self) -> List[str]:
        return [ch.material_id for ch in self.chunks if ch.offset == 0]

    def clear(self) -> int:
        n = self.total
        self.chunks.clear()
        self.total = 0
        return n


class DealerClient:
    """Party-side view of its own bundles with ledger-backed one-time semantics.

    ``take_*`` methods reserve and spend the touched bundle ids under the
    current ``(gate_id, epoch)`` context (``set_context``) before returning the
    material, raise ``MaterialStall`` when the typed pool cannot serve the
    request, and never return material twice.
    """

    def __init__(self, party: int, ledger: MaterialLedger, hmac_key: bytes, contract: NumericContract,
                 *, allow_insecure: Optional[bool] = None, profile_id: str = PROFILE_ID):
        if party not in (0, 1):
            raise MaterialError(f"DEALER_PARAMS: party must be 0 or 1, got {party!r}")
        if not isinstance(contract, NumericContract):
            raise ContractViolation("DEALER_PARAMS: contract must be a NumericContract")
        self.party = int(party)
        self.ledger = ledger
        self.hmac_key = bytes(hmac_key)
        self.contract = contract
        self.contract_hash = contract.contract_hash()
        self.profile_id = str(profile_id)
        # None (default): insecure bundles are refused unless the test-only switch is on
        # (``insecure_material_allowed``, evaluated at ingest time). True / False force the answer.
        self.allow_insecure: Optional[bool] = None if allow_insecure is None else bool(allow_insecure)
        self.insecure_material_seen = False
        self.gate_id = "*"
        self.epoch = 0
        self._lock = threading.RLock()
        self._seen_ids: set = set()
        self._beaver = _SlicePool()
        self._kary = _SlicePool()
        self._drelu: Dict[Tuple[int, int], _SlicePool] = {}         # (n_bits, P)
        self._trunc: Dict[Tuple[int, int], _SlicePool] = {}         # (f, L)
        self._r2f: Dict[int, _SlicePool] = {}                       # t
        self._matrix: Dict[Tuple[tuple, tuple], Deque[Tuple[str, MatrixTripleShare]]] = {}
        self._wmask_calls: Dict[Tuple[str, int], Deque[Tuple[str, WMaskCall]]] = {}
        self._wmask_mask: Dict[str, Tuple[str, np.ndarray, dict]] = {}
        self._wmask_mask_use: Dict[str, dict] = {}     # installed-object version of the first access
        self._wmask_mask_spent: set = set()
        self._vectors: Dict[str, Tuple[str, np.ndarray]] = {}
        self._vectors_spent: set = set()
        self.ingested_bytes = 0
        self.ingested_bundles = 0
        self.consumed_bytes = 0
        self.stalls = 0
        self.wasted_bytes = 0

    # -- context ------------------------------------------------------------------------------
    def set_context(self, gate_id: str, epoch: int) -> None:
        """Gate/epoch under which the next takes are reserved in the ledger."""
        with self._lock:
            self.gate_id = str(gate_id)
            self.epoch = int(epoch)

    @property
    def security_eligible(self) -> bool:
        """False once any INSECURE_TEST_ONLY_SHARED_SEED bundle was ingested."""
        return not self.insecure_material_seen

    # -- ingest -------------------------------------------------------------------------------
    def ingest(self, header: dict, payload: bytes, tag: str) -> None:
        """Verify, decode and register one bundle; any failure leaves no trace."""
        if not isinstance(header, dict):
            raise MaterialError("BUNDLE_MALFORMED: header must be a dict")
        missing = [k for k in HEADER_KEYS if k not in header]
        if missing:
            raise MaterialError(f"BUNDLE_MALFORMED: header misses {missing}")
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise MaterialError("BUNDLE_MALFORMED: payload must be bytes")
        payload = bytes(payload)
        # 1. authenticity first: nothing below is trusted before the tag verifies
        if not hmac_auth.verify_with_payload(self.hmac_key, header, payload, tag):
            raise MaterialError("BUNDLE_BAD_TAG: HMAC verification failed (foreign key or tampered bundle)")
        # 2. recipient binding
        if int(header["recipient"]) != self.party:
            raise MaterialError(f"BUNDLE_WRONG_RECIPIENT: bundle for party {header['recipient']} ingested by party {self.party}")
        # 3. contract and profile binding
        if header["numeric_contract_hash"] != self.contract_hash:
            raise MaterialVersionMismatch("BUNDLE_CONTRACT_MISMATCH: bundle numeric_contract_hash differs from the session contract")
        if header["profile_id"] != self.profile_id:
            raise MaterialVersionMismatch(f"BUNDLE_PROFILE_MISMATCH: bundle profile {header['profile_id']!r} != {self.profile_id!r}")
        if int(header["bytes"]) != len(payload):
            raise MaterialError("BUNDLE_MALFORMED: header bytes does not match the payload length")
        mt = header["material_type"]
        if mt not in MATERIAL_TYPES:
            raise MaterialError(f"BUNDLE_MALFORMED: unknown material_type {mt!r}")
        mid = str(header["material_id"])
        # Anything that is not positively the eligible mode with insecure == False counts as insecure.
        insecure = header.get("insecure") is not False or header.get("dealer_mode") != MODE_ISOLATED
        if insecure and not insecure_material_allowed(self.allow_insecure):
            raise MaterialError("BUNDLE_INSECURE_MODE: bundle labelled insecure (INSECURE_TEST_ONLY_SHARED_SEED or a "
                                "caller-seeded dealer) refused; the test-only switch is off")
        parts = header.get("parts")
        if parts is not None and list(parts) != list(PAYLOAD_PARTS[mt]):
            raise MaterialError(f"BUNDLE_MALFORMED: parts {parts!r} do not describe a {mt} payload")
        params = header["params"] if isinstance(header["params"], dict) else {}
        with self._lock:
            # 4. uniqueness
            if mid in self._seen_ids:
                raise MaterialReuse(f"BUNDLE_DUPLICATE_MATERIAL_ID: {mid!r} was ingested before")
            # 5. decode (before touching the ledger, so a bad payload leaves no record)
            decoded = self._decode(mt, params, payload, int(header["count"]))
            # 6. ledger version binding (setup_id, model_version, contract hash, profile)
            version = {k: header[k] for k in VERSION_KEYS}
            self.ledger.add_available(mid, mt, len(payload), version)
            # 7. store
            self._seen_ids.add(mid)
            self._store(mt, mid, params, decoded, version)
            if insecure:
                self.insecure_material_seen = True
            self.ingested_bytes += len(payload)
            self.ingested_bundles += 1

    def ingest_many(self, bundles: Sequence[Tuple[dict, bytes, str]]) -> int:
        for h, p, t in bundles:
            self.ingest(h, p, t)
        return len(bundles)

    def ingest_frame(self, bundle_msg: Any, payload: Any) -> None:
        """Ingest one ``BUNDLE{header, tag}`` frame and the ``bytes`` frame that follows it."""
        header, tag, _remaining = parse_bundle_frame(bundle_msg)
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise MaterialError("DEALER_BAD_MESSAGE: a BUNDLE must be followed by one bytes frame")
        self.ingest(header, bytes(payload), tag)

    def _decode(self, mt: str, params: dict, payload: bytes, count: int) -> Any:
        try:
            if mt in ("beaver", "kary_select"):
                shp = _shape_tuple(params["shape"])
                n = _numel(shp)
                if n != count:
                    raise MaterialError("BUNDLE_PAYLOAD_MISMATCH: count != elements of shape")
                a, b, c = _u64_split(payload, [(n,), (n,), (n,)])
                return (a, b, c)
            if mt == "matrix_triple":
                sa = _shape_tuple(params["shape_a"])
                sb = _shape_tuple(params["shape_b"])
                if len(sa) != 2 or len(sb) != 2 or sa[1] != sb[0]:
                    raise MaterialError("BUNDLE_PAYLOAD_MISMATCH: bad matrix triple shapes")
                a, b, c = _u64_split(payload, [sa, sb, (sa[0], sb[1])])
                return MatrixTripleShare(a, b, c)
            if mt == "wmask_mask":
                shp = _shape_tuple(params["shape"])
                (b,) = _u64_split(payload, [shp])
                return b
            if mt == "wmask_call":
                n, d, m = int(params["n_rows"]), int(params["d"]), int(params["m"])
                a, c = _u64_split(payload, [(n, d), (n, m)])
                return WMaskCall(a, c, str(params["mat_id"]))
            if mt == "vector_share":
                shp = _shape_tuple(params["shape"])
                (v,) = _u64_split(payload, [shp])
                return v
            if mt == "drelu":
                mat = DReLUMaterial.from_bytes(payload)
                P = 2 if bool(params.get("with_product", True)) else 1
                if mat.party != self.party or mat.M != count or mat.n_bits != int(params["n_bits"]) or mat.key_p.P != P:
                    raise MaterialError("BUNDLE_PAYLOAD_MISMATCH: DReLU material disagrees with its header")
                return mat
            if mt == "trunc_faithful":
                mat = TruncMaterial.from_bytes(payload)
                if mat.party != self.party or mat.M != count or mat.f != int(params["f"]) or mat.L != int(params["L"]):
                    raise MaterialError("BUNDLE_PAYLOAD_MISMATCH: truncation material disagrees with its header")
                return mat
            if mt == "ring_to_field":
                mat = R2FMaterial.from_bytes(payload)
                if mat.party != self.party or mat.M != count or int(mat.t) != int(params["t"]):
                    raise MaterialError("BUNDLE_PAYLOAD_MISMATCH: ring-to-field material disagrees with its header")
                return mat
        except (KeyError, TypeError, ValueError) as exc:
            raise MaterialError(f"BUNDLE_PAYLOAD_MISMATCH: cannot decode {mt} bundle ({exc.__class__.__name__})") from exc
        raise MaterialError(f"BUNDLE_MALFORMED: unknown material_type {mt!r}")

    def _store(self, mt: str, mid: str, params: dict, decoded: Any, version: dict) -> None:
        if mt == "beaver":
            a, b, c = decoded
            self._beaver.add(_Chunk(mid, a.size, arrays=(a, b, c)))
        elif mt == "kary_select":
            a, b, c = decoded
            self._kary.add(_Chunk(mid, a.size, arrays=(a, b, c)))
        elif mt == "matrix_triple":
            key = (tuple(decoded.a.shape), tuple(decoded.b.shape))
            self._matrix.setdefault(key, deque()).append((mid, decoded))
        elif mt == "wmask_mask":
            mat_id = str(params["mat_id"])
            if mat_id in self._wmask_mask:
                raise MaterialReuse(f"WMASK_MASK_DUPLICATE: a mask for {mat_id!r} is already installed; "
                                    "a new model version needs a new mat_id")
            self._wmask_mask[mat_id] = (mid, decoded, dict(version))
        elif mt == "wmask_call":
            self._wmask_calls.setdefault((decoded.mat_id, int(decoded.a.shape[0])), deque()).append((mid, decoded))
        elif mt == "vector_share":
            self._vectors[str(params["name"])] = (mid, decoded)
        elif mt == "drelu":
            self._drelu.setdefault((decoded.n_bits, decoded.key_p.P), _SlicePool()).add(_Chunk(mid, decoded.M, obj=decoded))
        elif mt == "trunc_faithful":
            self._trunc.setdefault((decoded.f, decoded.L), _SlicePool()).add(_Chunk(mid, decoded.M, obj=decoded))
        elif mt == "ring_to_field":
            self._r2f.setdefault(int(decoded.t), _SlicePool()).add(_Chunk(mid, decoded.M, obj=decoded))

    # -- ledger coupling -------------------------------------------------------------------------
    def _spend_ids(self, ids: List[str]) -> None:
        if not ids:
            return
        self.ledger.reserve(ids, self.gate_id, self.epoch)
        self.ledger.spend(ids)

    def _take_sliced(self, pool: Optional[_SlicePool], n: int, what: str) -> List[Tuple[_Chunk, int, int]]:
        n = int(n)
        if n < 1:
            raise ContractViolation(f"DEALER_PARAMS: take count must be >= 1 for {what}")
        if pool is None or pool.total < n:
            self.stalls += 1
            have = 0 if pool is None else pool.total
            raise MaterialStall(f"MATERIAL_STALL: {what}: need {n}, have {have}")
        ids = pool.ids_for(n)
        self._spend_ids(ids)
        return pool.take_parts(n)

    # -- takes -----------------------------------------------------------------------------------
    def _take_beaver_like(self, pool: _SlicePool, shape: Any, what: str) -> BeaverShare:
        shp = _shape_tuple(shape)
        n = _numel(shp)
        with self._lock:
            parts = self._take_sliced(pool, n, what)
            if len(parts) == 1:
                ch, lo, hi = parts[0]
                a, b, c = (arr[lo:hi] for arr in ch.arrays)
            else:
                a = np.concatenate([ch.arrays[0][lo:hi] for ch, lo, hi in parts])
                b = np.concatenate([ch.arrays[1][lo:hi] for ch, lo, hi in parts])
                c = np.concatenate([ch.arrays[2][lo:hi] for ch, lo, hi in parts])
            self.consumed_bytes += 24 * n
            return BeaverShare(a.reshape(shp).copy(), b.reshape(shp).copy(), c.reshape(shp).copy(), shp)

    def take_beaver(self, shape: Any) -> BeaverShare:
        return self._take_beaver_like(self._beaver, shape, "beaver")

    def take_kary_select(self, shape: Any) -> BeaverShare:
        return self._take_beaver_like(self._kary, shape, "kary_select")

    def take_matrix_triple(self, shape_a: Any, shape_b: Any) -> MatrixTripleShare:
        key = (_shape_tuple(shape_a), _shape_tuple(shape_b))
        with self._lock:
            q = self._matrix.get(key)
            if not q:
                self.stalls += 1
                raise MaterialStall(f"MATERIAL_STALL: matrix_triple {key} exhausted")
            mid, obj = q[0]
            self._spend_ids([mid])
            q.popleft()
            self.consumed_bytes += obj.nbytes()
            return obj

    def take_wmask_call(self, mat_id: str, n_rows: int) -> WMaskCall:
        key = (str(mat_id), int(n_rows))
        with self._lock:
            q = self._wmask_calls.get(key)
            if not q:
                self.stalls += 1
                raise MaterialStall(f"MATERIAL_STALL: wmask_call for {key[0]!r} with {key[1]} rows exhausted")
            mid, obj = q[0]
            self._spend_ids([mid])
            q.popleft()
            self.consumed_bytes += obj.nbytes()
            return obj

    def wmask_mask_share(self, mat_id: str, for_version: Optional[dict] = None) -> np.ndarray:
        """This party's share of the per-matrix mask B; spent in the ledger on first access.

        ``for_version`` is the installed-object version the caller will use the mask for
        (``{model_version, client_key_domain}``). The first access that names a version binds the mask
        to it; a later access for another version, or for a model_version other than the one the
        bundle header carries, is refused with ``WMASK_MASK_VERSION_MISMATCH`` (one mask, one W).
        """
        mat_id = str(mat_id)
        with self._lock:
            ent = self._wmask_mask.get(mat_id)
            if ent is None:
                raise MaterialStall(f"WMASK_MASK_MISSING: no mask bundle for {mat_id!r}")
            mid, b, _version = ent
            if for_version is not None:
                fv = {str(k): str(v) for k, v in dict(for_version).items()}
                if "model_version" in fv and "model_version" in _version \
                        and fv["model_version"] != str(_version["model_version"]):
                    raise MaterialVersionMismatch(
                        f"WMASK_MASK_VERSION_MISMATCH: mask bundle of {mat_id!r} is bound to another model_version")
                first = self._wmask_mask_use.get(mat_id)
                if first is not None and first != fv:
                    raise MaterialVersionMismatch(
                        f"WMASK_MASK_VERSION_MISMATCH: mask of {mat_id!r} already served another version")
                self._wmask_mask_use[mat_id] = fv
            if mid not in self._wmask_mask_spent:
                self._spend_ids([mid])
                self._wmask_mask_spent.add(mid)
                self.consumed_bytes += 8 * b.size
            return b

    def wmask_mask_version(self, mat_id: str) -> dict:
        """Version tuple the mask bundle was bound to (for installed-object records)."""
        with self._lock:
            ent = self._wmask_mask.get(str(mat_id))
            if ent is None:
                raise MaterialStall(f"WMASK_MASK_MISSING: no mask bundle for {mat_id!r}")
            return dict(ent[2])

    def _take_typed(self, pool: Optional[_SlicePool], count: int, what: str, concat) -> Any:
        with self._lock:
            parts = self._take_sliced(pool, count, what)
            out = None
            for ch, lo, hi in parts:
                piece = ch.obj.slice(lo, hi)
                out = piece if out is None else concat(out, piece)
            self.consumed_bytes += out.nbytes()
            return out

    def take_drelu(self, count: int, with_product: bool = True, n_bits: Optional[int] = None) -> DReLUMaterial:
        """DReLU material; a bit-only request (P=1) is served from product-payload (P=2) stock when its own pool is short.

        A P=2 bundle is a strict superset of a P=1 bundle (callers ignore the
        second payload word), so the fallback never changes the protocol; it
        only spends more bytes than the estimate predicted.
        """
        n = int(self.contract.dcf_domain_bits if n_bits is None else n_bits)
        P = 2 if with_product else 1
        with self._lock:
            pool = self._drelu.get((n, P))
            if P == 1 and (pool is None or pool.total < int(count)):
                alt = self._drelu.get((n, 2))
                if alt is not None and alt.total >= int(count):
                    pool = alt
            return self._take_typed(pool, count, f"drelu(n_bits={n}, P={P})", _concat_drelu)

    def take_trunc(self, count: int, f: Optional[int] = None, L: Optional[int] = None) -> TruncMaterial:
        f = int(self.contract.frac_bits if f is None else f)
        L = int(self.contract.logical_bits if L is None else L)
        return self._take_typed(self._trunc.get((f, L)), count, f"trunc_faithful(f={f}, L={L})", _concat_trunc)

    def take_r2f(self, count: int, t: int = DEFAULT_T) -> R2FMaterial:
        return self._take_typed(self._r2f.get(int(t)), count, f"ring_to_field(t={t})", _concat_r2f)

    def take_vector_share(self, name: str) -> np.ndarray:
        """Share of a server-private vector; spent on first take, reusable afterwards (it is a fixed value, not a mask)."""
        name = str(name)
        with self._lock:
            ent = self._vectors.get(name)
            if ent is None:
                raise MaterialStall(f"VECTOR_SHARE_MISSING: no vector share named {name!r}")
            mid, v = ent
            if mid not in self._vectors_spent:
                self._spend_ids([mid])
                self._vectors_spent.add(mid)
                self.consumed_bytes += 8 * v.size
            return v

    # -- stock and waste -------------------------------------------------------------------------------
    def stock(self) -> Dict[str, Tuple[int, int]]:
        """Exact remaining material per type as (count, bytes); count is in elements or objects."""
        with self._lock:
            out: Dict[str, Tuple[int, int]] = {}
            out["beaver"] = (self._beaver.total, 24 * self._beaver.total)
            out["kary_select"] = (self._kary.total, 24 * self._kary.total)
            cnt = sum(p.total for p in self._drelu.values())
            nb = sum(p.total * drelu_bytes_per_element(k[0], k[1] == 2) for k, p in self._drelu.items())
            out["drelu"] = (cnt, nb)
            cnt = sum(p.total for p in self._trunc.values())
            nb = sum(p.total * trunc_bytes_per_element(k[0]) for k, p in self._trunc.items())
            out["trunc_faithful"] = (cnt, nb)
            cnt = sum(p.total for p in self._r2f.values())
            nb = sum(p.total * r2f_bytes_per_element(t) for t, p in self._r2f.items())
            out["ring_to_field"] = (cnt, nb)
            cnt = sum(len(q) for q in self._matrix.values())
            nb = sum(o.nbytes() for q in self._matrix.values() for _, o in q)
            out["matrix_triple"] = (cnt, nb)
            cnt = sum(len(q) for q in self._wmask_calls.values())
            nb = sum(o.nbytes() for q in self._wmask_calls.values() for _, o in q)
            out["wmask_call"] = (cnt, nb)
            masks = [(mid, b) for mid, b, _ in self._wmask_mask.values() if mid not in self._wmask_mask_spent]
            out["wmask_mask"] = (len(masks), sum(8 * b.size for _, b in masks))
            vecs = [(mid, v) for mid, v in self._vectors.values() if mid not in self._vectors_spent]
            out["vector_share"] = (len(vecs), sum(8 * v.size for _, v in vecs))
            return out

    def discard(self, material_type: Optional[str] = None, reason: str = "PLAN_SWITCH") -> int:
        """Drop buffered one-time material (plan switch waste); untouched bundles are BURNED in the ledger.

        Returns the number of bytes discarded. Masks and vector shares are kept
        (they are installed objects, not per-call material).
        """
        with self._lock:
            pools: List[Tuple[str, _SlicePool]] = []
            if material_type in (None, "beaver"):
                pools.append(("beaver", self._beaver))
            if material_type in (None, "kary_select"):
                pools.append(("kary_select", self._kary))
            if material_type in (None, "drelu"):
                pools += [("drelu", p) for p in self._drelu.values()]
            if material_type in (None, "trunc_faithful"):
                pools += [("trunc_faithful", p) for p in self._trunc.values()]
            if material_type in (None, "ring_to_field"):
                pools += [("ring_to_field", p) for p in self._r2f.values()]
            before = self.stock()
            burn_ids: List[str] = []
            for _, p in pools:
                burn_ids += p.untouched_ids()
                p.clear()
            if material_type in (None, "matrix_triple"):
                for q in self._matrix.values():
                    burn_ids += [mid for mid, _ in q]
                    q.clear()
            if material_type in (None, "wmask_call"):
                for q in self._wmask_calls.values():
                    burn_ids += [mid for mid, _ in q]
                    q.clear()
            if burn_ids:
                self.ledger.burn(burn_ids, reason)
            after = self.stock()
            wasted = sum(before[k][1] - after[k][1] for k in before)
            self.wasted_bytes += wasted
            return wasted


# --------------------------------------------------------------------------
# Planner-facing material estimate (per-element material formulas)
# --------------------------------------------------------------------------
def tournament_shape(n: int, k: int) -> Tuple[int, int]:
    """(rounds, comparisons) of a k-ary max/argmax tournament over n candidates.

    Rounds = ceil(log_k n) for k in {2, 4, 8}; every node of g <= k
    candidates spends g - 1 comparisons, so the total is n - 1 whatever k is
    (each comparison eliminates exactly one candidate). Ties go to the lowest
    index in every realization (contract ``argmax_tie``).
    """
    n = int(n)
    k = int(k)
    if n < 1:
        raise ContractViolation(f"TOURNAMENT_PARAMS: n must be >= 1, got {n}")
    if k not in (2, 4, 8):
        raise ContractViolation(f"TOURNAMENT_PARAMS: k must be in {{2, 4, 8}}, got {k}")
    rounds = 0
    w = n
    while w > 1:
        w = -(-w // k)
        rounds += 1
    return rounds, n - 1


def tournament_material(n: int, k: int, with_index: bool) -> Tuple[int, int, int]:
    """(DReLU P=1, DReLU P=2, Beaver select) gates of the fused k-ary tournament ``party.Party._tournament`` over n items.

    Prices exactly what the Python realization issues (the C++ estimate is
    identical): per node of g members, g(g-1)/2 pairwise comparisons (DReLU without
    payload) in the first round; in the second round a binary node spends one
    Beaver select for the value plus one for the index once indices are secret
    (every level after the first), a node with g >= 3 members spends g - 1
    payload DReLUs (P = 2) for the value and the same again for a secret index.
    ``tournament_shape`` keeps the declared accounting (n - 1 eliminations) used
    for rounds; this function is what ``material_estimate`` charges. Tested in
    tests/test_dealer.py against ``session.tournament_counts`` and the k = 2 closed form.
    """
    n, k = int(n), int(k)
    if n < 1:
        raise ContractViolation(f"TOURNAMENT_PARAMS: n must be >= 1, got {n}")
    if k not in (2, 4, 8):
        raise ContractViolation(f"TOURNAMENT_PARAMS: k must be in {{2, 4, 8}}, got {k}")
    p1 = p2 = bv = 0
    w = n
    idx_public = True
    while w > 1:
        groups = [(s, min(k, w - s)) for s in range(0, w, k)]
        for _s, g in groups:
            p1 += g * (g - 1) // 2
            extra = 1 if (with_index and not idx_public) else 0
            if g == 2:
                bv += 1 + extra
            elif g >= 3:
                p2 += (g - 1) * (1 + extra)
        w = len(groups)
        idx_public = False
    return p1, p2, bv


def _degree_from_name(name: str, default: int) -> int:
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else default


def _bytes_of(material_type: str, count: int, contract: NumericContract, **p: Any) -> int:
    if material_type in ("beaver", "kary_select"):
        return 24 * count
    if material_type in ("matrix_triple", "wmask_call", "wmask_mask", "vector_share"):
        return 8 * count
    if material_type == "drelu":
        return drelu_bytes_per_element(int(p.get("n_bits", contract.dcf_domain_bits)), bool(p.get("with_product", True))) * count
    if material_type == "trunc_faithful":
        return trunc_bytes_per_element(int(p.get("f", contract.frac_bits))) * count
    if material_type == "ring_to_field":
        return r2f_bytes_per_element(int(p.get("t", DEFAULT_T))) * count
    raise MaterialError(f"MATERIAL_ESTIMATE_UNKNOWN_TYPE: {material_type!r}")


def _params_key(material_type: str, contract: NumericContract, p: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    """The dealer-facing parameters that distinguish pools of one material type."""
    if material_type == "drelu":
        return (("n_bits", int(p.get("n_bits", contract.dcf_domain_bits))), ("with_product", bool(p.get("with_product", True))))
    if material_type == "trunc_faithful":
        return (("f", int(p.get("f", contract.frac_bits))), ("L", int(p.get("L", contract.logical_bits))))
    if material_type == "ring_to_field":
        return (("t", int(p.get("t", DEFAULT_T))),)
    return ()


def _acc(est: Dict[Tuple[str, tuple], List[int]], material_type: str, count: int, contract: NumericContract,
         **p: Any) -> None:
    if count <= 0:
        return
    e = est.setdefault((material_type, _params_key(material_type, contract, p)), [0, 0])
    e[0] += int(count)
    e[1] += _bytes_of(material_type, int(count), contract, **p)


def _rows_and_n(shape: Any) -> Tuple[int, int]:
    shp = _shape_tuple(shape)
    if len(shp) < 1:
        raise ContractViolation("MATERIAL_ESTIMATE_SHAPE: row ops need shape (..., n)")
    return _numel(shp[:-1]), int(shp[-1])


def _estimate_into(est: Dict[Tuple[str, tuple], List[int]], op_kind: str, shape: Any, contract: NumericContract,
                   kw: dict) -> None:
    n_bits = int(kw.get("n_bits", contract.dcf_domain_bits))
    if op_kind in ("beaver_mul", "mul", "square", "select", "mul_fxp"):
        _acc(est, "beaver", _numel(_shape_tuple(shape)), contract)
    elif op_kind == "poly_horner":
        _acc(est, "beaver", int(kw.get("degree", 1)) * _numel(_shape_tuple(shape)), contract)
    elif op_kind == "matmul_shared":
        sa, sb = shape
        sa = _shape_tuple(sa)
        sb = _shape_tuple(sb)
        n, k, m = sa[-2], sa[-1], sb[-1]
        batch = _numel(sa[:-2])
        _acc(est, "matrix_triple", batch * (n * k + k * m + n * m), contract)
    elif op_kind in ("linear_generic", "embedding_generic"):
        n, d, m = _shape_tuple(shape)
        _acc(est, "matrix_triple", n * d + d * m + n * m, contract)
    elif op_kind in ("linear_wmask", "embedding_wmask"):
        n, d, m = _shape_tuple(shape)
        _acc(est, "wmask_call", n * d + n * m, contract)
    elif op_kind == "install_wmask":
        d, m = _shape_tuple(shape)
        _acc(est, "wmask_mask", d * m, contract)
    elif op_kind in ("drelu", "compare_public"):
        wp = bool(kw.get("with_product", False))
        _acc(est, "drelu", _numel(_shape_tuple(shape)), contract, n_bits=n_bits, with_product=wp)
    elif op_kind == "relu":
        _acc(est, "drelu", _numel(_shape_tuple(shape)), contract, n_bits=n_bits, with_product=True)
    elif op_kind == "trunc_faithful":
        _acc(est, "trunc_faithful", _numel(_shape_tuple(shape)), contract, f=int(kw.get("f", contract.frac_bits)))
    elif op_kind in ("ring_to_field", "a2h_exact"):
        _acc(est, "ring_to_field", _numel(_shape_tuple(shape)), contract, t=int(kw.get("t", DEFAULT_T)))
    elif op_kind in ("max_rows", "argmax_k", "argmax_row"):
        R, n = _rows_and_n(shape)
        with_index = op_kind in ("argmax_k", "argmax_row")
        p1, p2, bv = tournament_material(n, int(kw.get("k", 2)), with_index)
        _acc(est, "drelu", R * p1, contract, n_bits=n_bits, with_product=False)   # pairwise comparisons (P = 1)
        _acc(est, "drelu", R * p2, contract, n_bits=n_bits, with_product=True)    # fused node selects (payload)
        _acc(est, "kary_select", R * bv, contract)                                # Beaver selects of binary nodes
    elif op_kind == "exp_neg":
        N = _numel(_shape_tuple(shape))
        r = int(kw.get("r", contract.exp_r))
        order = int(kw.get("order", contract.exp_order))
        _acc(est, "drelu", N, contract, n_bits=n_bits, with_product=True)      # clamp at -Z
        _acc(est, "beaver", (order + r) * N, contract)                          # base square(s) + r squarings
    elif op_kind == "recip":
        N = _numel(_shape_tuple(shape))
        K = int(kw.get("K", contract.recip_K))
        deg = int(kw.get("degree", contract.recip_degree))
        newton = int(kw.get("newton", contract.recip_newton))
        _acc(est, "drelu", K * N, contract, n_bits=n_bits, with_product=False)
        _acc(est, "beaver", (1 + deg + 2 * newton + 1) * N, contract)
    elif op_kind == "rsqrt":
        N = _numel(_shape_tuple(shape))
        m = int(kw.get("m", contract.rsqrt_m))
        deg = int(kw.get("degree", contract.rsqrt_degree))
        newton = int(kw.get("newton", contract.rsqrt_newton))
        _acc(est, "drelu", m * N, contract, n_bits=n_bits, with_product=False)
        _acc(est, "beaver", (1 + deg + 3 * newton + 1) * N, contract)
    elif op_kind == "softmax_rows":
        R, n = _rows_and_n(shape)
        _estimate_into(est, "max_rows", shape, contract, {"k": kw.get("k_max", kw.get("k", 2)), "n_bits": n_bits})
        _estimate_into(est, "exp_neg", (R, n), contract, kw)
        _estimate_into(est, "recip", (R,), contract, kw)
        _acc(est, "beaver", R * n, contract)                                     # t * inv
    elif op_kind in ("layernorm_rows", "rmsnorm_rows"):
        R, d = _rows_and_n(shape)
        _estimate_into(est, "rsqrt", (R,), contract, kw)
        _acc(est, "beaver", 3 * R * d, contract)                                 # dx*dx, dx*rs, y*gamma
        faithful = kw.get("exact_square", contract.trunc_square_ln == "faithful")
        if faithful:
            _acc(est, "trunc_faithful", R * d, contract, f=int(kw.get("f", contract.frac_bits)))
    elif op_kind in ("gelu", "silu", "gelu_clip", "silu_clip"):
        N = _numel(_shape_tuple(shape))
        name = contract.gelu if op_kind.startswith("gelu") else contract.silu
        deg = int(kw.get("degree", _degree_from_name(name, 12)))
        _acc(est, "drelu", 2 * N, contract, n_bits=n_bits, with_product=False)
        _acc(est, "beaver", (deg + 2) * N, contract)
    elif op_kind in ("tanh", "tanh_clip"):
        N = _numel(_shape_tuple(shape))
        deg = int(kw.get("degree", 9))
        _acc(est, "drelu", 2 * N, contract, n_bits=n_bits, with_product=False)
        _acc(est, "beaver", (deg + 1) * N, contract)
    else:
        raise MaterialError(f"MATERIAL_ESTIMATE_UNKNOWN_OP: {op_kind!r}")


def material_estimate(op_kind: str, shape: Any, contract: NumericContract, **kw: Any) -> Dict[str, Tuple[int, int]]:
    """Per-party material one op consumes: ``{material_type: (count, bytes)}``.

    Shape conventions: elementwise ops take the tensor shape; ``matmul_shared``
    takes ``(shape_a, shape_b)``; ``linear_*`` / ``embedding_*`` take
    ``(n, d_in, d_out)``; ``install_wmask`` takes ``(d, m)``; row ops
    (``softmax_rows``, ``layernorm_rows``, ``max_rows``, ``argmax_k``) take
    ``(..., n)``. Keyword knobs: ``k`` (tournament arity), ``n_bits``, ``f``,
    ``t``, ``degree``, ``exact_square``. Counts are elements for Beaver-like
    and matrix material and gates for DReLU/trunc/r2f material; bytes use the
    lowlevel per-element formulas (DReLU 2200 B at n=32 with product, faithful
    truncation 2064 B at f=16, ring-to-field 3272 B).
    """
    if not isinstance(contract, NumericContract):
        raise ContractViolation("MATERIAL_ESTIMATE_PARAMS: contract must be a NumericContract")
    est: Dict[Tuple[str, tuple], List[int]] = {}
    _estimate_into(est, str(op_kind), shape, contract, dict(kw))
    out: Dict[str, List[int]] = {}
    for (mt, _params), (count, nbytes) in est.items():
        o = out.setdefault(mt, [0, 0])
        o[0] += count
        o[1] += nbytes
    return {k: (v[0], v[1]) for k, v in out.items()}


def material_estimate_detail(op_kind: str, shape: Any, contract: NumericContract, **kw: Any) -> List[Dict[str, Any]]:
    """Like :func:`material_estimate` but split by dealer parameters.

    Returns ``[{material_type, params, count, bytes}, ...]`` where ``params``
    is exactly what a PROVISION request needs (``n_bits``/``with_product`` for
    DReLU, ``f``/``L`` for faithful truncation, ``t`` for ring-to-field, empty
    otherwise). Summing the entries of one type gives ``material_estimate``.
    """
    if not isinstance(contract, NumericContract):
        raise ContractViolation("MATERIAL_ESTIMATE_PARAMS: contract must be a NumericContract")
    est: Dict[Tuple[str, tuple], List[int]] = {}
    _estimate_into(est, str(op_kind), shape, contract, dict(kw))
    return [{"material_type": mt, "params": dict(params), "count": v[0], "bytes": v[1]}
            for (mt, params), v in est.items()]


def merge_estimates(*ests: Dict[str, Tuple[int, int]]) -> Dict[str, Tuple[int, int]]:
    out: Dict[str, List[int]] = {}
    for e in ests:
        for k, (c, b) in e.items():
            o = out.setdefault(k, [0, 0])
            o[0] += int(c)
            o[1] += int(b)
    return {k: (v[0], v[1]) for k, v in out.items()}


def estimate_total_bytes(est: Dict[str, Tuple[int, int]]) -> int:
    return int(sum(b for _, b in est.values()))


__all__ = [
    "MODE_ISOLATED", "MODE_INSECURE", "MODES", "PROFILE_ID", "DEFAULT_T", "MATERIAL_TYPES", "HEADER_KEYS",
    "PAYLOAD_PARTS", "ENV_ALLOW_INSECURE", "insecure_material_allowed", "AUTH_NONCE_BYTES",
    "dealer_auth_tag", "verify_dealer_auth_tag", "hello_frame", "parse_challenge", "auth_frame", "req_frame",
    "parse_bundle_frame", "raise_if_error_frame",
    "BeaverShare", "MatrixTripleShare", "WMaskCall", "Dealer", "DealerClient",
    "material_estimate", "material_estimate_detail", "merge_estimates", "estimate_total_bytes", "tournament_shape",
    "tournament_material",
]
