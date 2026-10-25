"""TCP channel between the two online parties.

Wire format: an 8-byte big-endian length prefix followed by a typed payload.
The payload codec is implemented with ``struct`` and raw array bytes and
NEVER uses pickle: a frame crosses a trust boundary, and the receiver only
ever allocates what the declared dtype/shape/length imply. The codec is the
wire codec of both languages, byte-identical to the C++ ``encode_value`` / ``decode_value``:

    tag 0x01 bytes   : u32 BE length, raw bytes
    tag 0x02 ndarray : u8 dtype (1 = uint64, 2 = int64, 3 = float64, 4 = uint8, 5 = int32), u8 ndim,
                       ndim x u64 BE dims, raw little-endian element data (C order)
    tag 0x03 dict    : u32 BE count, then count x (u16 BE key length, utf-8 key, typed value)
    tag 0x04 int     : i64 BE
    tag 0x05 float   : f64 BE (IEEE 754)
    tag 0x06 str     : u32 BE length, utf-8
    tag 0x07 list    : u32 BE count, then count x typed value (tuples are sent as lists)
    tag 0x08 none    : no body
    tag 0x09 bool    : u8 (0 or 1)

Fail-closed rules of the encoder: a Python ``int`` outside the signed 64-bit
range is refused with ``CODEC_INT_RANGE`` (a wider value travels as a uint64
array or as a decimal string, never truncated); ``bytes`` or ``str`` of
2^32 bytes or more are refused with ``CODEC_LENGTH_RANGE``; an array whose
dtype is not one of the five wire dtypes is refused with
``CHANNEL_CODEC_UNSUPPORTED`` (no silent cast); anything else is refused
with ``CHANNEL_CODEC_UNSUPPORTED``. A numpy scalar is sent as a 0-dimensional
array of its dtype. The decoder refuses unknown tags, unknown dtype codes,
truncated or trailing bytes, a bool byte other than 0 or 1 and nesting
deeper than 64 with ``CHANNEL_CODEC_MALFORMED``.

Counters are exact: ``bytes_sent`` / ``bytes_recv`` count the full frame
(prefix + payload), ``exchange`` counts exactly one round. Message sizes
depend only on the public shapes and dtypes of the values, never on secret
contents (a secret-dependent branch must therefore never change the shape of
what it sends; this module cannot enforce that on its own).

Shaping: :class:`WallClockShaper` applies a token bucket whose bandwidth and
one-way delay are read from an exogenous trace indexed by elapsed wall time
since a shared origin. A transfer spanning a state change pays the integral
of the rates across the segments it covers (``delay_for`` walks the trace
segments). Shaping is applied at the sender by sleeping before the write,
which is an application-level emulation of a link, not a kernel-level replay:
results produced with it are labelled as such by the session layer.

What is tested (tests/test_channel.py): pinned byte vectors of every tag (the
same bytes the C++ codec emits), codec round trips for every type, the
integer and length range refusals, refusal of unsupported types and dtypes
and of malformed frames, exact counters over a
loopback TCP pair, round counting of ``exchange``, the shaper's integral
across segment changes (pure function tests) and a timing sanity check.
Design limit: the shaper serializes messages of one channel through a single
virtual link; a second channel between the same hosts is shaped
independently (no shared queue across channels).
"""

from __future__ import annotations

import math
import socket
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from opencrab2.errors import ChannelError

# --------------------------------------------------------------------------
# Typed payload codec (struct based, never pickle)
# --------------------------------------------------------------------------
_TAG_BYTES = 0x01
_TAG_NDARRAY = 0x02
_TAG_DICT = 0x03
_TAG_INT = 0x04
_TAG_FLOAT = 0x05
_TAG_STR = 0x06
_TAG_LIST = 0x07
_TAG_NONE = 0x08
_TAG_BOOL = 0x09

# Wire dtype codes of tag 0x02 (as in the C++ codec): code -> little-endian numpy dtype string, and back.
_DTYPE_BY_CODE = {1: "<u8", 2: "<i8", 3: "<f8", 4: "|u1", 5: "<i4"}
_CODE_BY_DTYPE = {v: k for k, v in _DTYPE_BY_CODE.items()}
_MAX_NDIM = 16
_MAX_DIM = 1 << 40         # per dimension, the element cap of the C++ decoder (CODEC_SHAPE)
_MAX_FRAME_BYTES = 1 << 40        # 1 TiB: refuses absurd length prefixes
_MAX_DICT_ITEMS = 1 << 20
_MAX_LIST_ITEMS = 1 << 24
_MAX_DEPTH = 64                   # nesting limit of dict/list values (same as the C++ decoder)
_MAX_U32 = (1 << 32) - 1
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1

_U8 = struct.Struct(">B")
_U16 = struct.Struct(">H")
_U32 = struct.Struct(">I")
_U64 = struct.Struct(">Q")
_I64 = struct.Struct(">q")
_F64 = struct.Struct(">d")


def _encode_into(out: List[bytes], obj: Any, depth: int = 0) -> None:
    if obj is None:
        out.append(_U8.pack(_TAG_NONE))
    elif isinstance(obj, bool):
        out.append(_U8.pack(_TAG_BOOL) + _U8.pack(1 if obj else 0))
    elif isinstance(obj, (bytes, bytearray, memoryview)):
        b = bytes(obj)
        if len(b) > _MAX_U32:
            raise ChannelError(f"CODEC_LENGTH_RANGE: bytes value of {len(b)} bytes exceeds the u32 length field")
        out.append(_U8.pack(_TAG_BYTES) + _U32.pack(len(b)))
        out.append(b)
    elif isinstance(obj, np.ndarray) or isinstance(obj, np.generic):
        arr = np.asarray(obj)
        if arr.dtype.hasobject:
            raise ChannelError("CHANNEL_CODEC_UNSUPPORTED: object arrays cannot be sent")
        # Normalize to the little-endian layout the wire carries.
        dt = arr.dtype
        if dt.byteorder == ">" or (dt.byteorder == "=" and not _little_endian_host()):
            arr = arr.astype(dt.newbyteorder("<"))
        code = _CODE_BY_DTYPE.get(arr.dtype.str)
        if code is None:
            raise ChannelError(f"CHANNEL_CODEC_UNSUPPORTED: dtype {arr.dtype.str!r} is not a wire dtype "
                               "(uint64, int64, float64, uint8, int32)")
        if arr.ndim > _MAX_NDIM:
            raise ChannelError(f"CHANNEL_CODEC_UNSUPPORTED: ndim {arr.ndim} exceeds {_MAX_NDIM}")
        raw = np.ascontiguousarray(arr).tobytes(order="C")
        hdr = (_U8.pack(_TAG_NDARRAY) + _U8.pack(code) + _U8.pack(arr.ndim)
               + b"".join(_U64.pack(int(s)) for s in arr.shape))
        out.append(hdr)
        out.append(raw)
    elif depth >= _MAX_DEPTH and isinstance(obj, (dict, list, tuple)):
        raise ChannelError(f"CHANNEL_CODEC_UNSUPPORTED: nesting deeper than {_MAX_DEPTH}")
    elif isinstance(obj, dict):
        if len(obj) > _MAX_DICT_ITEMS:
            raise ChannelError("CHANNEL_CODEC_UNSUPPORTED: dict too large")
        out.append(_U8.pack(_TAG_DICT) + _U32.pack(len(obj)))
        for k, v in obj.items():
            if not isinstance(k, str):
                raise ChannelError(f"CHANNEL_CODEC_BAD_KEY: dict keys must be str, got {type(k).__name__}")
            kb = k.encode("utf-8")
            if len(kb) > 0xFFFF:
                raise ChannelError("CHANNEL_CODEC_BAD_KEY: key longer than 65535 bytes")
            out.append(_U16.pack(len(kb)) + kb)
            _encode_into(out, v, depth + 1)
    elif isinstance(obj, int):
        if obj < INT64_MIN or obj > INT64_MAX:
            raise ChannelError(f"CODEC_INT_RANGE: int of {obj.bit_length()} bits does not fit the i64 wire field; "
                               "send a uint64 array or a decimal string instead")
        out.append(_U8.pack(_TAG_INT) + _I64.pack(obj))
    elif isinstance(obj, float):
        out.append(_U8.pack(_TAG_FLOAT) + _F64.pack(obj))
    elif isinstance(obj, str):
        try:
            sb = obj.encode("utf-8")
        except UnicodeEncodeError:
            raise ChannelError("CHANNEL_CODEC_UNSUPPORTED: str is not encodable as utf-8") from None
        if len(sb) > _MAX_U32:
            raise ChannelError(f"CODEC_LENGTH_RANGE: str of {len(sb)} bytes exceeds the u32 length field")
        out.append(_U8.pack(_TAG_STR) + _U32.pack(len(sb)))
        out.append(sb)
    elif isinstance(obj, (list, tuple)):
        if len(obj) > _MAX_LIST_ITEMS:
            raise ChannelError("CHANNEL_CODEC_UNSUPPORTED: list too large")
        out.append(_U8.pack(_TAG_LIST) + _U32.pack(len(obj)))
        for v in obj:
            _encode_into(out, v, depth + 1)
    else:
        raise ChannelError(f"CHANNEL_CODEC_UNSUPPORTED: cannot encode {type(obj).__name__}")


def _little_endian_host() -> bool:
    return np.dtype("=u8").str == "<u8"


def encode_payload(obj: Any) -> bytes:
    """Serialize ``obj`` to the typed wire format (never pickle)."""
    out: List[bytes] = []
    _encode_into(out, obj)
    return b"".join(out)


class _Reader:
    __slots__ = ("buf", "pos", "n")

    def __init__(self, buf: bytes):
        self.buf = buf
        self.pos = 0
        self.n = len(buf)

    def take(self, k: int) -> bytes:
        if k < 0 or self.pos + k > self.n:
            raise ChannelError(f"CHANNEL_CODEC_MALFORMED: truncated payload (need {k} at {self.pos}/{self.n})")
        b = self.buf[self.pos:self.pos + k]
        self.pos += k
        return b

    def u8(self) -> int:
        return _U8.unpack(self.take(1))[0]

    def u16(self) -> int:
        return _U16.unpack(self.take(2))[0]

    def u32(self) -> int:
        return _U32.unpack(self.take(4))[0]

    def u64(self) -> int:
        return _U64.unpack(self.take(8))[0]


def _decode_one(r: _Reader, depth: int = 0) -> Any:
    tag = r.u8()
    if tag in (_TAG_DICT, _TAG_LIST) and depth >= _MAX_DEPTH:
        raise ChannelError("CHANNEL_CODEC_MALFORMED: nesting too deep")
    if tag == _TAG_NONE:
        return None
    if tag == _TAG_BOOL:
        v = r.u8()
        if v not in (0, 1):
            raise ChannelError("CHANNEL_CODEC_MALFORMED: bool byte not in {0,1}")
        return bool(v)
    if tag == _TAG_BYTES:
        n = r.u32()
        return bytes(r.take(n))
    if tag == _TAG_NDARRAY:
        code = r.u8()
        dtstr = _DTYPE_BY_CODE.get(code)
        if dtstr is None:
            raise ChannelError(f"CHANNEL_CODEC_MALFORMED: unknown dtype code {code}")
        ndim = r.u8()
        if ndim > _MAX_NDIM:
            raise ChannelError("CHANNEL_CODEC_MALFORMED: ndim too large")
        shape = tuple(r.u64() for _ in range(ndim))
        if any(s > _MAX_DIM for s in shape):
            # also for an array without elements: (0, 2^63) is a shape neither side can hold, and numpy
            # would answer with a bare ValueError that callers catching OpenCrabError miss
            raise ChannelError("CHANNEL_CODEC_MALFORMED: array dimension above 2^40")
        dt = np.dtype(dtstr)
        numel = 1
        for s in shape:
            numel *= s
        # Python integers: the product cannot wrap; take() refuses more bytes than the frame holds
        # before anything is allocated.
        raw = r.take(numel * dt.itemsize)
        try:
            arr = np.frombuffer(raw, dtype=dt, count=numel).reshape(shape)
        except (ValueError, TypeError, MemoryError) as exc:
            raise ChannelError(f"CHANNEL_CODEC_MALFORMED: array shape refused ({exc.__class__.__name__})") from None
        # Own the memory (frombuffer views are read-only and alias the frame).
        return arr.copy()
    if tag == _TAG_DICT:
        cnt = r.u32()
        if cnt > _MAX_DICT_ITEMS:
            raise ChannelError("CHANNEL_CODEC_MALFORMED: dict count too large")
        d: Dict[str, Any] = {}
        for _ in range(cnt):
            kl = r.u16()
            try:
                k = r.take(kl).decode("utf-8")
            except UnicodeDecodeError:
                raise ChannelError("CHANNEL_CODEC_MALFORMED: dict key is not utf-8") from None
            d[k] = _decode_one(r, depth + 1)
        return d
    if tag == _TAG_INT:
        return _I64.unpack(r.take(8))[0]
    if tag == _TAG_FLOAT:
        return _F64.unpack(r.take(8))[0]
    if tag == _TAG_STR:
        n = r.u32()
        try:
            return r.take(n).decode("utf-8")
        except UnicodeDecodeError:
            raise ChannelError("CHANNEL_CODEC_MALFORMED: str is not utf-8") from None
    if tag == _TAG_LIST:
        cnt = r.u32()
        if cnt > _MAX_LIST_ITEMS:
            raise ChannelError("CHANNEL_CODEC_MALFORMED: list count too large")
        return [_decode_one(r, depth + 1) for _ in range(cnt)]
    raise ChannelError(f"CHANNEL_CODEC_MALFORMED: unknown tag 0x{tag:02x}")


def decode_payload(buf: bytes) -> Any:
    """Inverse of :func:`encode_payload`; refuses trailing bytes and malformed input."""
    r = _Reader(bytes(buf))
    obj = _decode_one(r)
    if r.pos != r.n:
        raise ChannelError(f"CHANNEL_CODEC_MALFORMED: {r.n - r.pos} trailing bytes")
    return obj


def frame_size(obj: Any) -> int:
    """Exact on-wire size (prefix + payload) of ``obj``; depends only on public shapes."""
    return 8 + len(encode_payload(obj))


# --------------------------------------------------------------------------
# Statistics and link state
# --------------------------------------------------------------------------
@dataclass
class ChannelStats:
    bytes_sent: int = 0
    bytes_recv: int = 0
    messages_sent: int = 0
    messages_recv: int = 0
    rounds: int = 0
    shaping_delay_s: float = 0.0

    def snapshot(self) -> dict:
        return {
            "bytes_sent": int(self.bytes_sent),
            "bytes_recv": int(self.bytes_recv),
            "bytes_total": int(self.bytes_sent + self.bytes_recv),
            "messages_sent": int(self.messages_sent),
            "messages_recv": int(self.messages_recv),
            "rounds": int(self.rounds),
            "shaping_delay_s": float(self.shaping_delay_s),
        }

    def diff(self, earlier: "ChannelStats") -> dict:
        """Counters accumulated since ``earlier`` (a copy taken before a gadget)."""
        return {
            "bytes_sent": self.bytes_sent - earlier.bytes_sent,
            "bytes_recv": self.bytes_recv - earlier.bytes_recv,
            "messages_sent": self.messages_sent - earlier.messages_sent,
            "messages_recv": self.messages_recv - earlier.messages_recv,
            "rounds": self.rounds - earlier.rounds,
            "shaping_delay_s": self.shaping_delay_s - earlier.shaping_delay_s,
        }

    def copy(self) -> "ChannelStats":
        return ChannelStats(self.bytes_sent, self.bytes_recv, self.messages_sent,
                            self.messages_recv, self.rounds, self.shaping_delay_s)


@dataclass
class LinkState:
    """Link parameters in force during one trace segment.

    ``bandwidth_bps`` is in bits per second; ``None`` or ``0`` means unshaped
    (infinite): one meaning in both languages, in ``platform.shaper`` and in the
    cost model. A trace cannot express an outage.
    ``one_way_delay_s`` is the propagation delay added to every message.
    """
    bandwidth_bps: Optional[float]
    one_way_delay_s: float = 0.0

    def __post_init__(self) -> None:
        if self.bandwidth_bps is not None:
            b = float(self.bandwidth_bps)
            if not math.isfinite(b) or b < 0:
                raise ChannelError(f"CHANNEL_SHAPER_PARAMS: bandwidth_bps must be finite and >= 0, got {b}")
            self.bandwidth_bps = b
        d = float(self.one_way_delay_s)
        if not math.isfinite(d) or d < 0:
            raise ChannelError(f"CHANNEL_SHAPER_PARAMS: one_way_delay_s must be finite and >= 0, got {d}")
        self.one_way_delay_s = d


class WallClockShaper:
    """Token bucket driven by an exogenous trace indexed by wall time.

    ``trace`` is a list of ``(elapsed_s, LinkState)`` sorted by ``elapsed_s``;
    segment i is in force from ``trace[i][0]`` until ``trace[i+1][0]`` (the
    last one forever). Before the first point the first state applies. The
    bucket has zero burst depth, i.e. it is a virtual-clock rate limiter: the
    serialization time of ``nbytes`` starting at wall time ``w`` is the
    smallest ``T`` with ``integral_{w}^{w+T} bandwidth(t) dt = 8 * nbytes``.
    A change of bandwidth in the middle of a transfer therefore applies to
    the bytes not yet drained (a change applies in the middle of a transfer).

    The shaper is a pure function of ``(nbytes, wall_s)``; queueing of
    consecutive messages on one channel is tracked by :class:`Channel`.
    Both parties must construct it from the same trace and the same
    ``origin_wall_s`` so that the public environment is identical for both.
    """

    def __init__(self, trace: List[Tuple[float, LinkState]], origin_wall_s: float):
        if not trace:
            raise ChannelError("CHANNEL_SHAPER_PARAMS: trace must contain at least one segment")
        pts: List[Tuple[float, LinkState]] = []
        last = -math.inf
        for i, item in enumerate(trace):
            try:
                t, st = item
            except (TypeError, ValueError):
                raise ChannelError("CHANNEL_SHAPER_PARAMS: trace items must be (elapsed_s, LinkState)") from None
            t = float(t)
            if not math.isfinite(t):
                raise ChannelError("CHANNEL_SHAPER_PARAMS: elapsed_s must be finite")
            if t < last:
                raise ChannelError("CHANNEL_SHAPER_PARAMS: trace must be sorted by elapsed_s")
            if not isinstance(st, LinkState):
                raise ChannelError("CHANNEL_SHAPER_PARAMS: trace states must be LinkState")
            last = t
            pts.append((t, st))
        self.trace = pts
        self.origin_wall_s = float(origin_wall_s)
        self._starts = [t for t, _ in pts]

    @staticmethod
    def from_points(points: List[tuple], origin_wall_s: float) -> "WallClockShaper":
        """Build from ``opencrab2.experiments.traces`` points ``(elapsed_s, bandwidth_bps, one_way_delay_s, ...)``."""
        tr = []
        for p in points:
            if len(p) < 3:
                raise ChannelError("CHANNEL_SHAPER_PARAMS: trace points need (elapsed_s, bandwidth_bps, one_way_delay_s)")
            bw = p[1]
            tr.append((float(p[0]), LinkState(None if bw is None else float(bw), float(p[2]))))
        return WallClockShaper(tr, origin_wall_s)

    # -- trace lookup ----------------------------------------------------------
    def _segment_index(self, elapsed_s: float) -> int:
        # last index with start <= elapsed_s; 0 when before the first point
        lo, hi = 0, len(self._starts) - 1
        if elapsed_s < self._starts[0]:
            return 0
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._starts[mid] <= elapsed_s:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def state_at(self, wall_s: float) -> LinkState:
        """Link state in force at absolute wall time ``wall_s``."""
        return self.trace[self._segment_index(float(wall_s) - self.origin_wall_s)][1]

    def elapsed(self, wall_s: float) -> float:
        return float(wall_s) - self.origin_wall_s

    def serialization_time(self, nbytes: int, wall_s: float) -> float:
        """Seconds to drain ``nbytes`` through the bucket starting at ``wall_s``.

        Walks the trace segments: each segment contributes ``bandwidth *
        remaining_segment_time`` bits until the message is drained. A segment
        with bandwidth ``None`` or ``0`` is unshaped: whatever is left of the
        transfer when it reaches that segment leaves at once (the rule of the
        C++ shaper, of ``platform.shaper.normalize_points`` and of the cost
        model; a bandwidth of 0 used to mean a blocked link here only).
        """
        nbytes = int(nbytes)
        if nbytes < 0:
            raise ChannelError("CHANNEL_SHAPER_PARAMS: nbytes must be >= 0")
        remaining_bits = 8.0 * nbytes
        if remaining_bits == 0.0:
            return 0.0
        t = float(wall_s) - self.origin_wall_s
        idx = self._segment_index(t)
        total = 0.0
        n = len(self.trace)
        while remaining_bits > 0.0:
            bw = self.trace[idx][1].bandwidth_bps
            seg_end = self.trace[idx + 1][0] if idx + 1 < n else math.inf
            if bw is None or bw <= 0.0:
                return total                      # unshaped segment drains instantly
            if seg_end == math.inf:
                return total + remaining_bits / bw
            avail_s = max(seg_end - t, 0.0)
            bits_in_seg = bw * avail_s
            if bits_in_seg >= remaining_bits:
                return total + remaining_bits / bw
            total += avail_s
            remaining_bits -= bits_in_seg
            t = seg_end
            idx += 1
        return total

    def delay_for(self, nbytes: int, wall_s: float) -> float:
        """Arrival delay of a message of ``nbytes`` sent at ``wall_s``.

        Serialization across the segments (token bucket integral) plus the
        one-way propagation delay of the state in force when the last byte
        leaves the bucket.
        """
        ser = self.serialization_time(nbytes, wall_s)
        return ser + self.state_at(float(wall_s) + ser).one_way_delay_s

    def rtt_at(self, wall_s: float) -> float:
        """2 x one-way delay of the state in force at ``wall_s`` (for probes/tests)."""
        return 2.0 * self.state_at(wall_s).one_way_delay_s


# --------------------------------------------------------------------------
# Channel
# --------------------------------------------------------------------------
class Channel:
    """A connected TCP endpoint with exact accounting and optional shaping.

    ``party`` fixes the ordering of :meth:`exchange` (P0 sends first, P1
    receives first) so that one socket pair never deadlocks.
    """

    def __init__(self, sock: socket.socket, party: int, shaper: Optional[WallClockShaper] = None):
        if party not in (0, 1):
            raise ChannelError(f"CHANNEL_PARAMS: party must be 0 or 1, got {party!r}")
        self.sock = sock
        self.party = int(party)
        self.stats = ChannelStats()
        self.shaper: Optional[WallClockShaper] = shaper
        self._link_free_at = 0.0           # wall time when the virtual link is idle again
        self._closed = False
        try:
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        try:
            self.sock.settimeout(None)
        except OSError:
            pass

    # -- shaping ---------------------------------------------------------------
    def set_shaper(self, shaper: Optional[WallClockShaper]) -> None:
        self.shaper = shaper
        self._link_free_at = 0.0

    def _shape(self, nbytes: int) -> None:
        if self.shaper is None:
            return
        now = time.time()
        start = max(now, self._link_free_at)
        ser = self.shaper.serialization_time(nbytes, start)
        self._link_free_at = start + ser
        delay = (self._link_free_at - now) + self.shaper.state_at(self._link_free_at).one_way_delay_s
        if delay > 0.0:
            self.stats.shaping_delay_s += delay
            time.sleep(delay)

    # -- framed IO -------------------------------------------------------------
    def _sendall(self, data: bytes) -> None:
        try:
            self.sock.sendall(data)
        except OSError as exc:
            raise ChannelError(f"CHANNEL_CLOSED: send failed ({exc.__class__.__name__})") from exc

    def _recvall(self, n: int) -> bytes:
        chunks: List[bytes] = []
        got = 0
        while got < n:
            try:
                chunk = self.sock.recv(min(n - got, 1 << 20))
            except OSError as exc:
                raise ChannelError(f"CHANNEL_CLOSED: recv failed ({exc.__class__.__name__})") from exc
            if not chunk:
                raise ChannelError("CHANNEL_CLOSED: peer closed the connection during recv")
            chunks.append(chunk)
            got += len(chunk)
        return b"".join(chunks) if len(chunks) != 1 else chunks[0]

    def send_raw(self, payload: bytes) -> int:
        """Send already-encoded payload bytes as one frame; returns the frame size."""
        frame = _U64.pack(len(payload)) + payload
        self._shape(len(frame))
        self._sendall(frame)
        self.stats.bytes_sent += len(frame)
        self.stats.messages_sent += 1
        return len(frame)

    def recv_raw(self) -> bytes:
        """Receive one frame and return its payload bytes (undecoded)."""
        header = self._recvall(8)
        (length,) = _U64.unpack(header)
        if length > _MAX_FRAME_BYTES:
            raise ChannelError(f"CHANNEL_FRAME_TOO_LARGE: length prefix {length} exceeds the limit")
        payload = self._recvall(length)
        self.stats.bytes_recv += 8 + length
        self.stats.messages_recv += 1
        return payload

    def send(self, obj: Any) -> None:
        """Encode and send one message (bytes, ndarray, dict, int in i64, float, str, list, bool, None)."""
        self.send_raw(encode_payload(obj))

    def recv(self) -> Any:
        """Receive and decode one message."""
        return decode_payload(self.recv_raw())

    def exchange(self, obj: Any) -> Any:
        """One communication round: P0 sends then receives, P1 receives then sends."""
        self.stats.rounds += 1
        if self.party == 0:
            self.send(obj)
            return self.recv()
        other = self.recv()
        self.send(obj)
        return other

    def count_round(self, n: int = 1) -> None:
        """Account rounds for protocols that use send/recv directly (one-directional opens)."""
        self.stats.rounds += int(n)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass

    @property
    def closed(self) -> bool:
        return self._closed

    def __enter__(self) -> "Channel":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# --------------------------------------------------------------------------
# Connection helpers
# --------------------------------------------------------------------------
def open_listener(host: str, port: int, backlog: int = 1) -> Tuple[socket.socket, int]:
    """Bind and listen; returns ``(listening_socket, chosen_port)`` (port 0 picks a free one)."""
    lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        lsock.bind((host, int(port)))
        lsock.listen(backlog)
    except OSError as exc:
        lsock.close()
        raise ChannelError(f"CHANNEL_BIND_FAILED: {host}:{port} ({exc.__class__.__name__})") from exc
    return lsock, int(lsock.getsockname()[1])


def accept_channel(lsock: socket.socket, party: int, shaper: Optional[WallClockShaper] = None,
                   timeout_s: Optional[float] = None, close_listener: bool = True) -> Channel:
    """Accept one connection on ``lsock`` and wrap it in a :class:`Channel`."""
    try:
        lsock.settimeout(timeout_s)
        conn, _ = lsock.accept()
    except OSError as exc:
        raise ChannelError(f"CHANNEL_ACCEPT_FAILED: {exc.__class__.__name__}") from exc
    finally:
        if close_listener:
            lsock.close()
    conn.settimeout(None)
    return Channel(conn, party, shaper)


def listen_channel(host: str, port: int, party: int, shaper: Optional[WallClockShaper] = None,
                   timeout_s: Optional[float] = None) -> Tuple[Channel, int]:
    """Listen on ``host:port``, accept exactly one peer; returns ``(channel, chosen_port)``.

    With ``port=0`` the chosen port is only known after the peer connected,
    so a launcher that needs the port beforehand uses :func:`open_listener`
    followed by :func:`accept_channel`.
    """
    lsock, chosen = open_listener(host, port)
    ch = accept_channel(lsock, party, shaper, timeout_s=timeout_s)
    return ch, chosen


def dial_channel(host: str, port: int, party: int, shaper: Optional[WallClockShaper] = None,
                 retries: int = 500, retry_sleep_s: float = 0.02,
                 connect_timeout_s: float = 30.0) -> Channel:
    """Connect to a listening peer, retrying while it is not up yet."""
    last: Optional[BaseException] = None
    for _ in range(max(1, int(retries))):
        try:
            s = socket.create_connection((host, int(port)), timeout=connect_timeout_s)
            s.settimeout(None)
            return Channel(s, party, shaper)
        except OSError as exc:
            last = exc
            time.sleep(retry_sleep_s)
    raise ChannelError(f"CHANNEL_CONNECT_FAILED: could not connect to {host}:{port} ({last!r})")


__all__ = [
    "ChannelStats", "LinkState", "WallClockShaper", "Channel",
    "encode_payload", "decode_payload", "frame_size", "INT64_MIN", "INT64_MAX",
    "open_listener", "accept_channel", "listen_channel", "dial_channel",
]
