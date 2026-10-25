"""Two-party gadget layer: ``Party`` and ``ReleasePolicy`` (paper, Section 5.4, operators composed from the blocks).

Both parties run the same code; ``party`` (0 = model owner P0, 1 = client P1)
selects the asymmetric steps. Every online message goes through the
``Channel`` of ``opencrab2.highlevel.channel`` (exact byte and round counters)
and every piece of correlated randomness comes from the party's
``DealerClient`` (``take_*``). Before the first message that uses a material
object the party moves its ids through the ``MaterialLedger``
(``AVAILABLE -> RESERVED -> SPENT``, durable before use; paper, Section 4.3,
material lifecycle and commit); objects that
carry no id (a dealer client that spends at take time) are consumed as is.

Gadgets (all vectorised over arbitrary numpy shapes, row-wise on the last axis
where relevant): sharing and masked opening, add / sub / public add / public
real multiply (frexp mantissa with pre-shift), Beaver multiply, fixed-point
multiply with SecureML local truncation, the faithful truncation gate
(``trunc_offset`` then one open then ``trunc_faithful_eval``), matrix triples,
``linear_generic`` / ``linear_wmask`` with the real one-round ``install_wmask``,
one-round DReLU / ReLU, public comparison, select, k-ary max / argmax
tournaments (k in {2, 4, 8}, 2 * ceil(log_k n) rounds, bit-identical output for
every k, lowest index on ties), exp on (-inf, 0], normalised reciprocal and
inverse square root, softmax, LayerNorm / RMSNorm with the faithful square,
clipped polynomial GELU / SiLU / tanh, Horner, and the ring-to-field resharing.

Release of model-derived values: ``reveal_final`` is the only method that
sends a model-derived value to one party in the clear. It goes through
``ReleasePolicy`` (recipient P1, kind in {token, class}, max_count). There is
no ``reveal_hidden`` and no layerwise diagnostic. ``reveal`` opens a value to
both parties and is used only on uniformly masked quantities (y = x + r).

Per-gadget statistics: ``Party.stats`` holds one dict per top-level gadget
call ``{kind, seconds, bytes, rounds, material_bytes, shape}``;
``Party.all_stats`` also holds nested calls.

k-ary tournament (design): a level groups the surviving candidates into nodes
of k. Round 1 opens the masked differences of every pair inside a node (DReLU,
k(k-1)/2 comparisons per node; pairs with a public-masked member are decided
publicly and need no comparison). Locally every member i gets its rank
r_i = #{j : i beats j}, a permutation because ties are broken by the lower
index. Round 2 selects: a node of size 2 uses one Beaver select; a node of size
g >= 3 uses one DReLU with product payload per member i >= 1 on
u_i = (r_i - (g-1)) M + (x_i - x_0) + M/2 with M = 2^(n-1-ceil(log2 g)), so that
1{u_i >= 0} = 1{r_i = g-1} = w_i and ReLU(u_i) - (M/2) w_i = w_i (x_i - x_0);
the node maximum is x_0 + sum_i w_i (x_i - x_0) and the node argmax is
idx_0 + sum_i w_i (idx_i - idx_0) (a second DReLU batch when the indices are
shared, a local sum at the first level where they are public). Both rounds
of a level are single openings, so a level costs exactly two rounds. Design
limit: inside a node of size g >= 3 the compared differences must satisfy
|x_i - x_j| < M/2 = 2^(n-2-ceil(log2 g)) ring units (2^12 real units for g in
{3, 4}, 2^11 for g in {5..8} at n = 32, f = 16); binary nodes keep the full
DCF bound 2^(n-1). Outputs are exact integers and therefore bit-identical for
every k on inputs inside these bounds (tested in tests/test_gadgets.py).

What is tested (tests/test_party.py, tests/test_gadgets.py): every gadget
against a float reference in an in-process two-party harness (gelu 5e-3 abs,
softmax 1e-3, layernorm 2e-3, exp 4e-3 rel), the faithful square on large
residuals, k-ary max / argmax exactness across k and against numpy argmax
with lowest-index ties, the release policy (count, kind, recipient),
install_wmask versioning, ledger reserve/spend ordering. Design limits: the
SecureML truncation of a contract-bounded product x wraps with probability
|x|/2^64 per element (about 0.8 percent of 4-layer and 2.4 percent of 12-layer
sessions of 8 tokens release a wrong token sequence, see lowlevel/trunc.py);
the DCF bound |x| < 2^(n-1) is a precondition the
caller establishes through the numeric contract, it is not checked at run
time (a secret-dependent check would leak).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..errors import (ContractViolation, MaterialError, MaterialReuse, MaterialVersionMismatch,
                      UnauthorizedRelease)
from ..lowlevel import ring
from ..lowlevel.csprng import SecureRandom
from ..lowlevel.drelu import drelu_eval, relu_from_drelu
from ..lowlevel.ring_to_field import r2f_eval
from ..lowlevel.trunc import trunc_faithful_eval, trunc_local as _trunc_local_ll, trunc_offset
from . import approx

RELEASE_KINDS = ("token", "class")
ARGMAX_K = (2, 4, 8)
MAX_K = (2, 4, 8)

_U64 = np.uint64
_ONE = np.uint64(1)
_ZERO = np.uint64(0)


def _u(a: Any) -> np.ndarray:
    return np.asarray(a, dtype=np.uint64)


def _const_u64(value: int) -> np.uint64:
    """Python int (any sign) to its residue mod 2^64 as np.uint64."""
    return np.uint64(int(value) % (1 << 64))


def _ids_of(obj: Any) -> List[str]:
    """Material ids carried by a dealer object (``material_ids`` or ``material_id``)."""
    ids = getattr(obj, "material_ids", None)
    if ids is None:
        mid = getattr(obj, "material_id", None)
        ids = [mid] if mid else []
    return [str(i) for i in ids]


def _nbytes_of(obj: Any) -> int:
    """Material volume of a dealer object in bytes (per party)."""
    nb = getattr(obj, "nbytes", None)
    if callable(nb):
        try:
            return int(nb())
        except TypeError:
            pass
    total = 0
    for name in ("a", "b", "c"):
        arr = getattr(obj, name, None)
        if isinstance(arr, np.ndarray):
            total += int(arr.nbytes)
    return total


def _bmm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Exact ring matmul over the last two axes with broadcast leading dims."""
    A = _u(A)
    B = _u(B)
    if A.ndim == 2 and B.ndim == 2:
        return ring.ring_matmul(A, B)
    lead = np.broadcast_shapes(A.shape[:-2], B.shape[:-2])
    A2 = np.broadcast_to(A, lead + A.shape[-2:])
    B2 = np.broadcast_to(B, lead + B.shape[-2:])
    out = np.empty(lead + (A.shape[-2], B.shape[-1]), np.uint64)
    for idx in np.ndindex(*lead):
        out[idx] = ring.ring_matmul(A2[idx], B2[idx])
    return out


# ---------------------------------------------------------------------------
# Release policy
# ---------------------------------------------------------------------------
class ReleasePolicy:
    """Declared output release of a session: recipient, kind, and count.

    The admitted profile SH2P-TrustedOffline-v1 releases generated tokens or
    the classification only to the client P1, so ``recipient`` must be 1.
    ``authorize(kind)`` is called by both parties for every release so that
    their counters stay in step; it raises ``UnauthorizedRelease`` on a kind
    mismatch or when ``max_count`` releases were already authorised. Tested in
    tests/test_party.py. Design limit: the policy is a local object; the
    session layer binds it to the HELLO frame of the web protocol.
    """

    def __init__(self, recipient: int, kind: str, max_count: int) -> None:
        if int(recipient) != 1:
            raise ContractViolation(
                f"RELEASE_RECIPIENT: the admitted profile releases outputs only to P1, got {recipient}")
        if kind not in RELEASE_KINDS:
            raise ContractViolation(f"RELEASE_KIND: kind must be one of {RELEASE_KINDS}, got {kind!r}")
        if int(max_count) < 0:
            raise ContractViolation(f"RELEASE_COUNT: max_count must be >= 0, got {max_count}")
        self.recipient = int(recipient)
        self.kind = str(kind)
        self.max_count = int(max_count)
        self.count = 0

    def authorize(self, kind: str) -> None:
        if kind != self.kind:
            raise UnauthorizedRelease(
                f"UNAUTHORIZED_RELEASE: release kind {kind!r} not declared (declared {self.kind!r})")
        if self.count >= self.max_count:
            raise UnauthorizedRelease(
                f"RELEASE_BUDGET_EXCEEDED: {self.max_count} releases of kind {self.kind!r} already authorised")
        self.count += 1

    def remaining(self) -> int:
        return self.max_count - self.count

    def to_dict(self) -> dict:
        return {"recipient": self.recipient, "kind": self.kind, "max_count": self.max_count,
                "count": self.count}


# ---------------------------------------------------------------------------
# Statistics tracking
# ---------------------------------------------------------------------------
@dataclass
class GadgetStats:
    kind: str
    seconds: float
    bytes: int
    rounds: int
    material_bytes: int
    shape: tuple = ()
    depth: int = 0

    def as_dict(self) -> dict:
        return {"kind": self.kind, "seconds": self.seconds, "bytes": self.bytes,
                "rounds": self.rounds, "material_bytes": self.material_bytes,
                "shape": list(self.shape), "depth": self.depth}


class _Track:
    """Context manager recording one statistics entry on the party."""

    def __init__(self, party: "Party", kind: str, shape: Any = ()) -> None:
        self.p = party
        self.kind = kind
        try:
            self.shape = tuple(int(v) for v in np.shape(shape)) if not isinstance(shape, tuple) else tuple(shape)
        except Exception:
            self.shape = ()

    def __enter__(self) -> "_Track":
        st = self.p.ch.stats
        self.t0 = time.perf_counter()
        self.b0 = int(st.bytes_sent) + int(st.bytes_recv)
        self.r0 = int(st.rounds)
        self.m0 = self.p._material_bytes
        self.depth = self.p._depth
        self.p._depth += 1
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.p._depth -= 1
        st = self.p.ch.stats
        rec = {"kind": self.kind, "seconds": time.perf_counter() - self.t0,
               "bytes": int(st.bytes_sent) + int(st.bytes_recv) - self.b0,
               "rounds": int(st.rounds) - self.r0,
               "material_bytes": self.p._material_bytes - self.m0,
               "shape": list(self.shape), "depth": self.depth}
        if self.depth == 0:
            self.p.stats.append(rec)
        self.p.all_stats.append(rec)
        return False


def rsqrt_scale_bits(contract: Any) -> int:
    """Extra scale bits e2 of the rsqrt selector s2 (held at scale 2^(f + e2)); see Party.rsqrt.

    e2 = max(0, 2 (j0 + 1 + m) - f): the smallest scale at which every value 4^-(j0+1+k), k <= m,
    is an integer. The normalisation product V * s2 is then below 2^(2f + e2) ring units, and
    ``rsqrt_product_bits`` must stay at or below 40 so that the wrap probability |x|/2^64 of trunc_local
    stays below 2^-23 per element here (checked by the registry and by tests/test_gadgets.py).
    """
    f = int(contract.frac_bits)
    return max(0, 2 * (int(contract.rsqrt_j0) + 1 + int(contract.rsqrt_m)) - f)


def rsqrt_product_bits(contract: Any) -> int:
    """Bit length bound l_x of the rsqrt normalisation product V * s2 (v < 1 at scale 2^(2f + e2))."""
    return 2 * int(contract.frac_bits) + rsqrt_scale_bits(contract)


# ---------------------------------------------------------------------------
# Party
# ---------------------------------------------------------------------------
class Party:
    """One semi-honest online party (see module docstring)."""

    def __init__(self, party: int, channel: Any, dealer: Any, contract: Any,
                 release: Optional[ReleasePolicy], rng: Optional[Any] = None,
                 model_version: str = "unversioned", client_key_domain: str = "default") -> None:
        if int(party) not in (0, 1):
            raise ContractViolation(f"PARTY_ID: party must be 0 or 1, got {party}")
        self.party = int(party)
        self.ch = channel
        self.dealer = dealer
        self.contract = contract
        self.release = release
        # share_input sends value minus mask and the peer sees a window of raw outputs: the default is
        # a ChaCha20 stream keyed from the OS, never a statistical generator
        self.rng = rng if rng is not None else SecureRandom.from_os()
        self.ledger = getattr(dealer, "ledger", None)
        self.model_version = str(model_version)
        self.client_key_domain = str(client_key_domain)
        self.gate_id = "gate0"
        self.epoch = 0
        self.stats: List[dict] = []
        self.all_stats: List[dict] = []
        self._depth = 0
        self._material_bytes = 0
        self._vectors: Dict[str, Optional[np.ndarray]] = {}
        self._wmask_b: Dict[str, np.ndarray] = {}
        self._wmask_bound: Dict[str, dict] = {}       # version of the first install served by each mask
        self._installed: Dict[str, dict] = {}
        self.install_log: List[dict] = []
        self.release_log: List[dict] = []
        self._he: Dict[str, Any] = {}

    # -- bookkeeping ---------------------------------------------------------
    def track(self, kind: str, shape: Any = ()) -> _Track:
        return _Track(self, kind, shape)

    def set_gate(self, gate_id: str, epoch: Optional[int] = None) -> None:
        """Name the gate (and epoch) under which the next material is reserved."""
        self.gate_id = str(gate_id)
        if epoch is not None:
            self.epoch = int(epoch)

    @property
    def f(self) -> int:
        return int(self.contract.frac_bits)

    @property
    def material_bytes(self) -> int:
        return self._material_bytes

    def enc(self, x: Any) -> np.ndarray:
        return ring.encode(np.asarray(x, np.float64), self.contract)

    def dec(self, v: Any) -> np.ndarray:
        return ring.decode(v, self.contract)

    def const(self, value: float, shape: Any = ()) -> np.ndarray:
        """Share of a public fixed-point constant (P0 holds it, P1 holds 0)."""
        if self.party == 0:
            return self.enc(np.full(shape, float(value)))
        return np.zeros(shape, np.uint64)

    def const_ring(self, value: Any) -> np.ndarray:
        """Share of a public ring array (P0 holds it, P1 holds 0)."""
        v = _u(value)
        return v.copy() if self.party == 0 else np.zeros(v.shape, np.uint64)

    def _consume(self, material: Any) -> int:
        """Ledger transition for a material object; returns its byte volume.

        The admitted ``DealerClient`` reserves and spends bundle ids inside
        ``take_*`` (durable before the material is returned), so its objects
        carry no ids and only the byte accounting happens here. For a client
        that hands out objects with ``material_ids`` / ``material_id``, ids in
        AVAILABLE are reserved under the current gate and every RESERVED id is
        spent before the first message; BURNED ids are refused.
        """
        if material is None:
            return 0
        nbytes = _nbytes_of(material)
        self._material_bytes += nbytes
        ids = _ids_of(material)
        if ids and self.ledger is not None:
            states = {i: self.ledger.state(i) for i in ids}
            burned = [i for i, s in states.items() if s == "BURNED"]
            if burned:
                raise MaterialReuse(f"MATERIAL_REUSE: burned material presented: {burned[:3]}")
            avail = [i for i, s in states.items() if s == "AVAILABLE"]
            if avail:
                self.ledger.reserve(avail, self.gate_id, self.epoch)
            to_spend = [i for i in ids if self.ledger.state(i) == "RESERVED"]
            if to_spend:
                self.ledger.spend(to_spend)
        return nbytes

    # -- sharing / opening -----------------------------------------------------
    def _count_round(self) -> None:
        """Account one round for a one-directional message (send/recv outside ``exchange``)."""
        cr = getattr(self.ch, "count_round", None)
        if callable(cr):
            cr(1)
        else:  # pragma: no cover
            self.ch.stats.rounds = int(self.ch.stats.rounds) + 1

    def share_input(self, value_enc: Optional[np.ndarray], owner: int, shape: Any) -> np.ndarray:
        """Owner secret-shares ``value_enc`` (one message owner -> peer); the peer passes None."""
        shape = tuple(shape) if shape is not None else ()
        if self.party == int(owner):
            value = _u(value_enc)
            if value.shape != shape:
                raise ContractViolation(f"SHARE_SHAPE: value shape {value.shape} differs from declared {shape}")
            mask = ring.rand_u64(shape, self.rng)
            self.ch.send(ring.sub(value, mask))
            self._count_round()
            return mask
        other = _u(self.ch.recv())
        self._count_round()
        if other.shape != shape:
            raise ContractViolation(f"SHARE_SHAPE: received shape {other.shape} differs from declared {shape}")
        return other

    def _open(self, share: np.ndarray) -> np.ndarray:
        """One exchange round: both parties learn share0 + share1."""
        share = _u(share)
        other = _u(self.ch.exchange(share))
        if other.shape != share.shape:
            raise ContractViolation(f"OPEN_SHAPE: peer sent {other.shape}, expected {share.shape}")
        return ring.add(share, other)

    def reveal(self, share: np.ndarray) -> np.ndarray:
        """Open a value to BOTH parties. Only for uniformly masked quantities."""
        return self._open(share)

    def reveal_final(self, share: np.ndarray, kind: str) -> Optional[np.ndarray]:
        """Release a model-derived value to the declared recipient (P1) via ReleasePolicy."""
        if self.release is None:
            raise UnauthorizedRelease("UNAUTHORIZED_RELEASE: no ReleasePolicy declared for this party")
        self.release.authorize(kind)
        share = _u(share)
        self.release_log.append({"kind": kind, "shape": list(share.shape), "count": self.release.count})
        if self.party == self.release.recipient:
            other = _u(self.ch.recv())
            self._count_round()
            return ring.add(share, other)
        self.ch.send(share)
        self._count_round()
        return None

    # -- local linear algebra ---------------------------------------------------
    def add(self, a: Any, b: Any) -> np.ndarray:
        return ring.add(a, b)

    def sub(self, a: Any, b: Any) -> np.ndarray:
        return ring.sub(a, b)

    def add_public(self, share: Any, pub: Any) -> np.ndarray:
        """Add a public ring constant (P0 folds it in)."""
        if self.party == 0:
            return ring.add(share, pub)
        return np.array(share, np.uint64, copy=True)

    def add_public_real(self, share: Any, value: Any) -> np.ndarray:
        share = _u(share)
        return self.add_public(share, self.enc(np.broadcast_to(np.asarray(value, np.float64), share.shape)))

    def mul_public(self, share: Any, pub: Any) -> np.ndarray:
        """Multiply by a public ring integer (exact, no truncation)."""
        return ring.mul(share, pub)

    def mul_public_real(self, share: Any, value: float) -> np.ndarray:
        """Multiply a fixed-point share by a public real constant.

        Integers are exact. Otherwise ``value = m * 2^e`` (frexp) and the
        mantissa is encoded at full f-bit precision; small constants
        (e <= -5) pre-shift the share by ``-e - 4`` bits so the product keeps
        full precision while staying bounded. Error: relative at most 2^-f
        from the mantissa rounding (|x c| 2^-f absolute) plus 2 ulp of
        truncation (tested in tests/test_party.py).
        """
        value = float(value)
        share = _u(share)
        if value.is_integer():
            return ring.mul(share, _const_u64(int(value)))
        f = self.f
        m, e = math.frexp(value)
        pre = max(0, -e - 4)
        shift = f - e - pre
        if shift < 0:
            return ring.mul(share, _const_u64(int(round(value))))
        s = self.trunc_local(share, pre) if pre else share
        return self.trunc_local(ring.mul(s, self.enc(np.array(m))), shift)

    def matmul_public(self, share: Any, pub: Any, pub_on_left: bool = False) -> np.ndarray:
        s = _u(share)
        p = _u(pub)
        return _bmm(p, s) if pub_on_left else _bmm(s, p)

    # -- truncation --------------------------------------------------------------
    def trunc_local(self, share: Any, f: Optional[int] = None) -> np.ndarray:
        """SecureML local probabilistic truncation by ``f`` bits (0 rounds).

        Correct up to 1 ulp except for a full wrap of 2^(64-f) ring units with
        probability |x|/2^64 (paper, Section 5, secure building blocks); used only on contract-bounded
        products.
        """
        f = self.f if f is None else int(f)
        share = _u(share)
        if f == 0:
            return share
        return _trunc_local_ll(share, f, self.party)

    def trunc_faithful(self, share: Any, f: Optional[int] = None, L: Optional[int] = None) -> np.ndarray:
        """Exact floor(x / 2^f) via the faithful gate (one round, zero failure).

        ``L`` defaults to the contract's logical width; the gate is exact for
        x + 2^(L-1) in [0, 2^64), i.e. x in [-2^(L-1), 2^64 - 2^(L-1)), which
        covers every non-negative square as well as the symmetric logical
        range. The material must have been generated for the same (f, L).
        """
        f = self.f if f is None else int(f)
        L = int(self.contract.logical_bits) if L is None else int(L)
        if not (1 <= f <= L - 1 <= 63):
            raise ContractViolation(f"TRUNC_PARAMS: need 1 <= f <= L-1 <= 63, got f={f}, L={L}")
        x = _u(share)
        flat = x.reshape(-1)
        with self.track("trunc_faithful", x.shape):
            mat = self.dealer.take_trunc(int(flat.size), f, L=L)
            if int(getattr(mat, "f", f)) != f or int(getattr(mat, "L", L)) != L:
                raise MaterialError(
                    f"MATERIAL_PARAMS: trunc material is for (f={getattr(mat, 'f', None)}, "
                    f"L={getattr(mat, 'L', None)}), requested (f={f}, L={L})")
            self._consume(mat)
            xo = trunc_offset(flat, self.party, L)
            y = self._open(ring.add(xo, _u(mat.r_share).reshape(-1)))
            out = _u(trunc_faithful_eval(mat, y, self.party))
            return out.reshape(x.shape)

    # -- Beaver products -----------------------------------------------------------
    def _beaver_begin(self, sx: np.ndarray, sy: np.ndarray, triple: Any = None) -> Tuple[np.ndarray, Any, tuple]:
        sx = _u(sx)
        sy = _u(sy)
        shape = np.broadcast_shapes(sx.shape, sy.shape)
        sx = np.broadcast_to(sx, shape)
        sy = np.broadcast_to(sy, shape)
        if triple is None:
            triple = self.dealer.take_beaver(shape)
        self._consume(triple)
        a = _u(triple.a).reshape(shape)
        b = _u(triple.b).reshape(shape)
        masked = np.stack([ring.sub(sx, a), ring.sub(sy, b)])
        return masked, (a, b, _u(triple.c).reshape(shape)), shape

    def _beaver_end(self, abc: tuple, de: np.ndarray) -> np.ndarray:
        a, b, c = abc
        d, e = de[0], de[1]
        z = ring.add(c, ring.add(ring.mul(d, b), ring.mul(e, a)))
        if self.party == 0:
            z = ring.add(z, ring.mul(d, e))
        return z

    def beaver_mul(self, sx: Any, sy: Any, triple: Any = None) -> np.ndarray:
        """Shares of x*y (ring product, 1 round, no truncation)."""
        masked, abc, shape = self._beaver_begin(sx, sy, triple)
        de = self._open(masked)
        return self._beaver_end(abc, de)

    def mul_fxp(self, sx: Any, sy: Any, trunc_bits: Optional[int] = None) -> np.ndarray:
        """Fixed-point product: Beaver + local truncation (contract-bounded operands)."""
        return self.trunc_local(self.beaver_mul(sx, sy), trunc_bits)

    def square_fxp(self, sx: Any, exact: bool = False) -> np.ndarray:
        """x*x re-scaled to f bits; ``exact`` routes the truncation through the faithful gate."""
        if exact:
            return self.trunc_faithful(self.beaver_mul(sx, sx))
        return self.mul_fxp(sx, sx)

    # -- matrix products -----------------------------------------------------------
    def matmul_shared(self, sX: Any, sY: Any, trunc: bool = True, triple: Any = None) -> np.ndarray:
        """Shared x shared product over the last two axes (batched), 1 round.

        Opens X - A and Y - B: 8 (numel(X) + numel(Y)) bytes per direction.
        """
        sX = _u(sX)
        sY = _u(sY)
        if sX.ndim < 2 or sY.ndim < 2 or sX.shape[-1] != sY.shape[-2]:
            raise ContractViolation(f"MATMUL_SHAPE: {sX.shape} @ {sY.shape}")
        n, kk = int(sX.shape[-2]), int(sX.shape[-1])
        m = int(sY.shape[-1])
        lead = np.broadcast_shapes(sX.shape[:-2], sY.shape[:-2])
        Xb = np.broadcast_to(sX, lead + (n, kk)).reshape(-1, n, kk)
        Yb = np.broadcast_to(sY, lead + (kk, m)).reshape(-1, kk, m)
        batch = Xb.shape[0]
        with self.track("matmul_shared", (sX.shape, sY.shape)):
            if triple is None:
                triples = [self.dealer.take_matrix_triple((n, kk), (kk, m)) for _ in range(batch)]
            else:
                triples = list(triple) if isinstance(triple, (list, tuple)) else [triple]
                if len(triples) != batch:
                    raise MaterialError(f"MATERIAL_PARAMS: need {batch} matrix triples, got {len(triples)}")
            for t in triples:
                self._consume(t)
            A = np.stack([_u(t.a).reshape(n, kk) for t in triples])
            B = np.stack([_u(t.b).reshape(kk, m) for t in triples])
            C = np.stack([_u(t.c).reshape(n, m) for t in triples])
            # one open for every batch element: 8 (numel(X) + numel(Y)) bytes per direction
            de = self._open(np.concatenate([ring.sub(Xb, A).reshape(-1), ring.sub(Yb, B).reshape(-1)]))
            D = de[:Xb.size].reshape(Xb.shape)
            E = de[Xb.size:].reshape(Yb.shape)
            z = ring.add(C, ring.add(_bmm(D, B), _bmm(A, E)))
            if self.party == 0:
                z = ring.add(z, _bmm(D, E))
            z = z.reshape(lead + (n, m))
            return self.trunc_local(z) if trunc else z

    def vector_share(self, name: str, optional: bool = False) -> Optional[np.ndarray]:
        """This party's share of a server-private vector or matrix (cached).

        The share comes from ``share_input`` (P0 shares its vector online; the session layer stores it
        under ``name``). A dealer-provided ``vector_share`` bundle means the dealer saw the plaintext
        vector: it is accepted only from a dealer whose material is already labelled insecure (the unit
        test harness) and refused with ``VECTOR_SHARE_NOT_PRESHARED`` otherwise.
        """
        if name in self._vectors:
            v = self._vectors[name]
        else:
            try:
                v = _u(self.dealer.take_vector_share(name))
            except (MaterialError, KeyError, AttributeError) as exc:
                if not optional:
                    raise MaterialError(f"MATERIAL_MISSING_VECTOR: no vector share named {name!r} ({exc})")
                v = None
            if v is not None and not self._dealer_is_insecure():
                raise MaterialError(f"VECTOR_SHARE_NOT_PRESHARED: {name!r} was not shared with share_input and an "
                                    "eligible dealer must never hold a plaintext vector")
            self._vectors[name] = v
        if v is None and not optional:
            raise MaterialError(f"MATERIAL_MISSING_VECTOR: no vector share named {name!r}")
        return v

    def _dealer_is_insecure(self) -> bool:
        """True iff the material client reports that insecure (test-only) bundles were ingested."""
        for obj in (self.dealer, getattr(self.dealer, "client", None)):
            if obj is None:
                continue
            se = getattr(obj, "security_eligible", None)
            if se is None:
                continue
            return not bool(se() if callable(se) else se)
        return False                               # unknown front end: treated as eligible (fail closed)

    def linear_generic(self, sX: Any, mat_id: str, trunc: bool = True) -> np.ndarray:
        """Y = X @ W (+ b) with secret-shared W via a generic matrix triple (1 round)."""
        sX = _u(sX)
        W = self.vector_share(mat_id)
        with self.track("linear_generic", (sX.shape[0], W.shape[0], W.shape[1])):
            y = self.matmul_shared(sX, W, trunc=trunc)
            b = self.vector_share(mat_id + ".b", optional=True)
            if b is not None:
                y = ring.add(y, np.broadcast_to(b, y.shape))
            return y

    def installed(self, mat_id: str) -> bool:
        """True iff ``mat_id`` is installed under the current (model_version, client_key_domain)."""
        rec = self._installed.get(mat_id)
        if rec is None:
            return False
        if rec["version"] != self._version():
            return False
        if self.ledger is not None and hasattr(self.ledger, "installed"):
            try:
                return bool(self.ledger.installed(self._object_id(mat_id), rec["version"]))
            except Exception:  # pragma: no cover
                return False
        return True

    def installed_objects(self) -> Dict[str, dict]:
        return {k: dict(v["version"]) for k, v in self._installed.items()}

    def _version(self) -> dict:
        return {"model_version": self.model_version, "client_key_domain": self.client_key_domain}

    @staticmethod
    def _object_id(mat_id: str) -> str:
        return f"wmask_install({mat_id})"

    def install_wmask(self, mat_id: str, W_enc_or_None: Optional[np.ndarray]) -> dict:
        """Real one-round install of the public offset E = W - B for ``mat_id``.

        P1 sends its mask share B1; P0 computes E = W - B0 - B1 and returns E.
        Both record the installed object versioned by (model_version,
        client_key_domain) in the ledger; E is never reused across versions.
        Returns {mat_id, bytes, seconds, rounds, version, skipped}.
        """
        version = self._version()
        if self.installed(mat_id):
            return {"mat_id": mat_id, "bytes": 0, "seconds": 0.0, "rounds": 0, "version": version,
                    "skipped": True}
        # One mask serves one version (paper, Section 4.3, ledger): E1 = W1 - B and E2 = W2 - B under one B give
        # the client W2 - W1 exactly. Every check below runs before the first message of the install.
        bound = self._wmask_bound.get(mat_id)
        if bound is not None and bound != version:
            self._wmask_b.pop(mat_id, None)
            raise MaterialVersionMismatch(
                f"WMASK_MASK_VERSION_MISMATCH: the mask of {mat_id!r} already served another "
                "(model_version, client_key_domain); a new version needs a fresh mask bundle under a new mat_id")
        if mat_id not in self._wmask_b:
            try:
                share = self.dealer.wmask_mask_share(mat_id, for_version=dict(version))
            except TypeError:                      # a dealer front end without the version argument
                share = self.dealer.wmask_mask_share(mat_id)
            self._wmask_b[mat_id] = _u(share)
        bundle_version = None
        ver_fn = getattr(self.dealer, "wmask_mask_version", None)
        if ver_fn is not None:
            try:
                bundle_version = ver_fn(mat_id)
            except MaterialError:
                bundle_version = None              # warm stack: the share was carried over, the bundle is gone
        if bundle_version is not None and "model_version" in bundle_version \
                and str(bundle_version["model_version"]) != self.model_version:
            self._wmask_b.pop(mat_id, None)
            raise MaterialVersionMismatch(
                f"WMASK_MASK_VERSION_MISMATCH: the mask bundle of {mat_id!r} is bound to another model_version "
                "than this party; request a fresh mask bundle")
        self._wmask_bound[mat_id] = dict(version)
        B = self._wmask_b[mat_id]
        st = self.ch.stats
        b0 = int(st.bytes_sent) + int(st.bytes_recv)
        t0 = time.perf_counter()
        with self.track("install_wmask", B.shape):
            if self.party == 1:
                if W_enc_or_None is not None:
                    raise ContractViolation("INSTALL_OWNER: the client holds no plaintext weights")
                self.ch.send(B)
                E = _u(self.ch.recv())
                self._count_round()
            else:
                if W_enc_or_None is None:
                    raise ContractViolation("INSTALL_OWNER: P0 must supply the encoded weight matrix")
                W = _u(W_enc_or_None)
                if W.shape != B.shape:
                    raise ContractViolation(f"INSTALL_SHAPE: W {W.shape} differs from mask {B.shape}")
                B1 = _u(self.ch.recv())
                if B1.shape != B.shape:
                    raise ContractViolation(f"INSTALL_SHAPE: peer mask share {B1.shape} differs from {B.shape}")
                E = ring.sub(ring.sub(W, B), B1)
                self.ch.send(E)
                self._count_round()
            if E.shape != B.shape:
                raise ContractViolation(f"INSTALL_SHAPE: E {E.shape} differs from mask {B.shape}")
        self._installed[mat_id] = {"E": E, "version": dict(version)}
        if self.ledger is not None and hasattr(self.ledger, "register_installed"):
            self.ledger.register_installed(self._object_id(mat_id), dict(version))
        rec = {"mat_id": mat_id, "bytes": int(st.bytes_sent) + int(st.bytes_recv) - b0,
               "seconds": time.perf_counter() - t0, "rounds": 1, "version": dict(version),
               "skipped": False, "shape": list(B.shape)}
        self.install_log.append(rec)
        return rec

    def linear_wmask(self, sX: Any, mat_id: str, trunc: bool = True) -> np.ndarray:
        """Y = X @ W (+ b) with weight-masked material: opens X - A only (8 n d bytes).

        Z = C + D @ [B] + [A] @ E (+ D @ E on P0), D = X - A public.
        Requires ``installed(mat_id)`` (raises MaterialError otherwise).
        """
        sX = _u(sX)
        if not self.installed(mat_id):
            raise MaterialError(f"OBJECT_NOT_INSTALLED: wmask linear on {mat_id!r} before its install")
        E = self._installed[mat_id]["E"]
        B = self._wmask_b[mat_id]
        n = int(sX.shape[0])
        with self.track("linear_wmask", (n, B.shape[0], B.shape[1])):
            call = self.dealer.take_wmask_call(mat_id, n)
            self._consume(call)
            A = _u(call.a).reshape(sX.shape)
            C = _u(call.c).reshape(n, B.shape[1])
            D = self._open(ring.sub(sX, A))
            z = ring.add(C, ring.add(_bmm(D, B), _bmm(A, E)))
            if self.party == 0:
                z = ring.add(z, _bmm(D, E))
            if trunc:
                z = self.trunc_local(z)
            b = self.vector_share(mat_id + ".b", optional=True)
            if b is not None:
                z = ring.add(z, np.broadcast_to(b, z.shape))
            return z

    # -- FSS comparison layer -------------------------------------------------------
    def _drelu_begin(self, x_flat: np.ndarray, with_product: bool) -> Tuple[np.ndarray, Any]:
        mat = self.dealer.take_drelu(int(x_flat.size), with_product=with_product)
        self._consume(mat)
        return ring.add(x_flat, _u(mat.r_share).reshape(-1)), mat

    @staticmethod
    def _drelu_end(mat: Any, y: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        b, rb = drelu_eval(mat, y)
        return _u(b), (None if rb is None else _u(rb))

    def drelu(self, x: Any) -> np.ndarray:
        """Shares of 1{x >= 0} (one round). Requires |x| < 2^(n-1) ring units."""
        x = _u(x)
        with self.track("drelu", x.shape):
            flat = x.reshape(-1)
            if flat.size == 0:
                return np.zeros(x.shape, np.uint64)
            masked, mat = self._drelu_begin(flat, False)
            y = self._open(masked)
            b, _ = self._drelu_end(mat, y)
            return b.reshape(x.shape)

    def relu(self, x: Any) -> np.ndarray:
        """ReLU(x) = y [b] - [r b] in the same round as DReLU."""
        x = _u(x)
        with self.track("relu", x.shape):
            flat = x.reshape(-1)
            if flat.size == 0:
                return np.zeros(x.shape, np.uint64)
            masked, mat = self._drelu_begin(flat, True)
            y = self._open(masked)
            b, rb = self._drelu_end(mat, y)
            return _u(relu_from_drelu(y, b, rb)).reshape(x.shape)

    def compare_public(self, x: Any, c: Any, op: str = "ge") -> np.ndarray:
        """1{x >= c} ('ge') or 1{x < c} ('lt') for a public real c (1 round)."""
        x = _u(x)
        with self.track("compare_public", x.shape):
            xc = self.add_public_real(x, -np.asarray(c, np.float64))
            b = self.drelu(xc)
            if op == "ge":
                return b
            if op == "lt":
                return self.add_public(ring.neg(b), _ONE)
            raise ContractViolation(f"COMPARE_OP: op must be 'ge' or 'lt', got {op!r}")

    def select(self, bit: Any, a: Any, b: Any, triple: Any = None) -> np.ndarray:
        """b + bit (a - b): arithmetic multiplexer (1 round, exact)."""
        with self.track("select", np.shape(a)):
            diff = ring.sub(a, b)
            return ring.add(b, self.beaver_mul(bit, diff, triple))

    # -- k-ary tournaments ------------------------------------------------------------
    @staticmethod
    def _check_k(k: int, allowed: tuple, what: str) -> int:
        k = int(k)
        if k not in allowed:
            raise ContractViolation(f"{what}_K: k must be one of {allowed}, got {k}")
        return k

    def _tournament(self, x: np.ndarray, mask: Optional[np.ndarray], k: int,
                    with_index: bool) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """k-ary max (and argmax) over the last axis; see the module docstring."""
        x = _u(x)
        lead = x.shape[:-1]
        n = int(x.shape[-1])
        if n == 0:
            raise ContractViolation("MAX_ROWS_EMPTY: cannot take the maximum of an empty row")
        vals = x.reshape(-1, n)
        R = vals.shape[0]
        if mask is None:
            valid = np.ones((R, n), bool)
        else:
            valid = np.broadcast_to(np.asarray(mask, bool), x.shape).reshape(R, n).copy()
            if not bool(valid.any(axis=1).all()):
                raise ContractViolation("MAX_ROWS_EMPTY: every row needs at least one valid entry")
        nb = int(self.contract.dcf_domain_bits)
        idx: Optional[np.ndarray] = None
        idx_public = True
        if with_index:
            idx = self.const_ring(np.broadcast_to(np.arange(n, dtype=np.uint64), (R, n)))
        while vals.shape[1] > 1:
            w = vals.shape[1]
            groups = [(s, min(k, w - s)) for s in range(0, w, k)]
            # ---- round 1: all pairs inside every node ------------------------------
            gi_l: List[int] = []
            gj_l: List[int] = []
            for s, g in groups:
                for i in range(g):
                    for j in range(i + 1, g):
                        gi_l.append(s + i)
                        gj_l.append(s + j)
            gi = np.asarray(gi_l, dtype=np.int64)
            gj = np.asarray(gj_l, dtype=np.int64)
            P = int(gi.size)
            c = np.zeros((R, P), np.uint64)
            if P:
                vi = valid[:, gi]
                vj = valid[:, gj]
                vp = vi & vj
                diff = ring.sub(vals[:, gi], vals[:, gj])
                flat = diff[vp]
                if flat.size:
                    c[vp] = self.drelu(flat)
                override = (vi | ~vj) & ~vp
                if self.party == 0:
                    c[override] = _ONE
            rank = np.zeros((R, w), np.uint64)
            with np.errstate(over="ignore"):
                np.add.at(rank, (slice(None), gi), c)
                lose = ring.neg(c) if self.party == 1 else ring.sub(np.ones_like(c), c)
                np.add.at(rank, (slice(None), gj), lose)
            # ---- round 2: fused selection ---------------------------------------------
            opens: List[np.ndarray] = []
            plan: List[tuple] = []
            for s, g in groups:
                if g == 1:
                    plan.append(("pass", s, g))
                elif g == 2:
                    plan.append(("pair", s, g))
                else:
                    plan.append(("node", s, g))
            # locate the pair bit of each binary node: pairs are enumerated in node order
            pair_pos: Dict[int, int] = {}
            pos = 0
            for s, g in groups:
                if g == 2:
                    pair_pos[s] = pos
                pos += g * (g - 1) // 2
            requests: List[tuple] = []
            for kind, s, g in plan:
                if kind == "pair":
                    bit = c[:, pair_pos[s]]
                    dv = ring.sub(vals[:, s], vals[:, s + 1])
                    masked, abc, _ = self._beaver_begin(bit, dv)
                    opens.append(masked.reshape(-1))
                    req = {"kind": "pair", "s": s, "val": abc, "bit": bit}
                    if with_index and not idx_public:
                        di = ring.sub(idx[:, s], idx[:, s + 1])
                        masked_i, abc_i, _ = self._beaver_begin(bit, di)
                        opens.append(masked_i.reshape(-1))
                        req["idx"] = abc_i
                    requests.append(req)
                elif kind == "node":
                    M = 1 << (nb - 1 - int(math.ceil(math.log2(g))))
                    Mu = _const_u64(M)
                    offset = _const_u64(-(g - 1) * M + M // 2)
                    members = list(range(1, g))
                    cols = [s + i for i in members]
                    d = ring.sub(vals[:, cols], vals[:, [s]])
                    d[~valid[:, cols]] = _ZERO
                    u = ring.add(ring.mul(rank[:, cols], Mu), d)
                    if self.party == 0:
                        u = ring.add(u, offset)
                    masked_u, mat_u = self._drelu_begin(u.reshape(-1), True)
                    opens.append(masked_u)
                    req = {"kind": "node", "s": s, "g": g, "M": M, "mat": mat_u, "cols": cols}
                    if with_index and not idx_public:
                        di = ring.sub(idx[:, cols], idx[:, [s]])
                        v = ring.add(ring.mul(rank[:, cols], Mu), di)
                        if self.party == 0:
                            v = ring.add(v, offset)
                        masked_v, mat_v = self._drelu_begin(v.reshape(-1), True)
                        opens.append(masked_v)
                        req["mat_idx"] = mat_v
                    requests.append(req)
            opened = self._open(np.concatenate(opens)) if opens else np.zeros(0, np.uint64)
            cursor = 0
            new_vals = np.empty((R, len(groups)), np.uint64)
            new_idx = np.empty((R, len(groups)), np.uint64) if with_index else None
            new_valid = np.empty((R, len(groups)), bool)
            qi = 0
            for gidx, (kind, s, g) in enumerate(plan):
                new_valid[:, gidx] = valid[:, s:s + g].any(axis=1)
                if kind == "pass":
                    new_vals[:, gidx] = vals[:, s]
                    if with_index:
                        new_idx[:, gidx] = idx[:, s]
                    continue
                req = requests[qi]
                qi += 1
                if kind == "pair":
                    de = opened[cursor:cursor + 2 * R].reshape(2, R)
                    cursor += 2 * R
                    new_vals[:, gidx] = ring.add(vals[:, s + 1], self._beaver_end(req["val"], de))
                    if with_index:
                        if idx_public:
                            # idx_0 - idx_1 = -1 publicly: idx = (s + 1) - bit
                            new_idx[:, gidx] = ring.sub(self.const_ring(np.full(R, s + 1, np.uint64)), req["bit"])
                        else:
                            de_i = opened[cursor:cursor + 2 * R].reshape(2, R)
                            cursor += 2 * R
                            new_idx[:, gidx] = ring.add(idx[:, s + 1], self._beaver_end(req["idx"], de_i))
                else:
                    cols = req["cols"]
                    m = len(cols)
                    half = _const_u64(req["M"] // 2)
                    y = opened[cursor:cursor + R * m]
                    cursor += R * m
                    b, rb = self._drelu_end(req["mat"], y)
                    relu = _u(relu_from_drelu(y, b, rb)).reshape(R, m)
                    bm = b.reshape(R, m)
                    contrib = ring.sub(relu, ring.mul(bm, half))
                    new_vals[:, gidx] = ring.add(vals[:, s], contrib.sum(axis=1, dtype=np.uint64))
                    if with_index:
                        if idx_public:
                            offs = np.asarray([i for i in range(1, g)], dtype=np.uint64)
                            acc = ring.mul(bm, offs[None, :]).sum(axis=1, dtype=np.uint64)
                            new_idx[:, gidx] = ring.add(self.const_ring(np.full(R, s, np.uint64)), acc)
                        else:
                            yv = opened[cursor:cursor + R * m]
                            cursor += R * m
                            bv, rbv = self._drelu_end(req["mat_idx"], yv)
                            relu_v = _u(relu_from_drelu(yv, bv, rbv)).reshape(R, m)
                            contrib_i = ring.sub(relu_v, ring.mul(bv.reshape(R, m), half))
                            new_idx[:, gidx] = ring.add(idx[:, s], contrib_i.sum(axis=1, dtype=np.uint64))
            vals, valid = new_vals, new_valid
            if with_index:
                idx = new_idx
                idx_public = False
        out_val = vals[:, 0].reshape(lead)
        out_idx = idx[:, 0].reshape(lead) if with_index else None
        return out_val, out_idx

    def max_rows(self, x: Any, mask: Optional[np.ndarray] = None, k: int = 2) -> np.ndarray:
        """Row-wise maximum over the last axis by a k-ary tournament (2 ceil(log_k n) rounds).

        ``mask`` (public bool, True = valid) excludes positions; every row needs
        at least one valid entry. Output is bit-identical for every k.
        """
        k = self._check_k(k, MAX_K, "MAX_ROWS")
        x = _u(x)
        with self.track(f"max_rows_k{k}", x.shape):
            val, _ = self._tournament(x, mask, k, False)
            return val

    def argmax_row(self, x: Any, k: int = 2, kind: Optional[str] = None,
                   release: bool = True, mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Argmax over the last axis (lowest index on ties), k-ary tournament.

        With ``release=True`` (default) the index shares are released to P1 only
        through ``reveal_final`` under the policy kind (default: the declared
        kind); P0 receives None. With ``release=False`` the index SHARE is
        returned to both parties and nothing is revealed.
        """
        k = self._check_k(k, ARGMAX_K, "ARGMAX")
        x = _u(x)
        with self.track(f"argmax_k{k}", x.shape):
            _, idx = self._tournament(x, mask, k, True)
            if not release:
                return idx
            kind = self.release.kind if (kind is None and self.release is not None) else kind
            return self.reveal_final(idx, kind if kind is not None else "token")

    # -- polynomials -------------------------------------------------------------------
    def poly_horner(self, u: Any, coeffs: Sequence[float]) -> np.ndarray:
        """p(u) with ascending public coefficients; one Beaver + truncation per degree."""
        u = _u(u)
        acc = self.const(coeffs[-1], u.shape)
        for cf in reversed(list(coeffs)[:-1]):
            acc = self.mul_fxp(acc, u)
            acc = self.add_public_real(acc, cf)
        return acc

    def poly_contract(self, x: Any, name: str) -> np.ndarray:
        """Evaluate a named approximation contract on x (normalisation + Horner)."""
        c = approx.contract(name)
        x = _u(x)
        if getattr(c, "affine", False):
            kf = 2.0 / (c.hi - c.lo)
            u = self.mul_public_real(x, kf)
            u = self.add_public_real(u, -(2.0 * c.lo / (c.hi - c.lo) + 1.0))
        else:
            u = self.mul_public_real(x, 1.0 / c.scale) if c.scale != 1.0 else x
        if c.even_in_u:
            u = self.square_fxp(u)
        return self.poly_horner(u, c.coeffs)

    # -- clipped polynomial activation ------------------------------------------------
    def clip_poly_act(self, x: Any, name: str) -> np.ndarray:
        """GELU / SiLU / tanh: FSS clip outside [-R, R], polynomial inside.

        Rounds: 1 (two DReLUs batched) + degree (Horner) + 1 (products batched).
        Requires |x| + R < 2^(n-1-f) real units.
        """
        x = _u(x)
        c = approx.contract(name)
        R = float(c.hi)
        with self.track(name, x.shape):
            both = self.drelu(np.stack([self.add_public_real(x, R), self.add_public_real(x, -R)]))
            b_lo, b_hi = both[0], both[1]
            u = self.mul_public_real(x, 1.0 / R)
            p = self.poly_horner(u, c.coeffs)
            w = ring.sub(b_lo, b_hi)
            if name.startswith("tanh"):
                wp = self.beaver_mul(w, p)
                one = self.enc(np.array(1.0))
                act = ring.add(wp, ring.mul(ring.add(b_hi, b_lo), one))
                return self.add_public_real(act, -1.0)
            prod = self.beaver_mul(np.stack([b_hi, w]), np.stack([x, p]))
            return ring.add(prod[0], prod[1])

    def gelu(self, x: Any) -> np.ndarray:
        return self.clip_poly_act(x, str(getattr(self.contract, "gelu", "gelu_clip12")))

    def silu(self, x: Any) -> np.ndarray:
        return self.clip_poly_act(x, str(getattr(self.contract, "silu", "silu_clip12")))

    def tanh(self, x: Any) -> np.ndarray:
        return self.clip_poly_act(x, "tanh_clip")

    # -- exp on (-inf, 0] ----------------------------------------------------------------
    def exp_neg(self, z: Any, order: Optional[int] = None) -> np.ndarray:
        """exp(z) for z <= 0: clamp at -Z, base 1 + w + w^2/2 (order 2), r squarings.

        The chain runs at scale 2^(f+e) and drops the e extra bits at the end.
        Rounds 3 + r = 10 at the default contract. Requires |z + Z| < 2^(n-1-f).
        """
        z = _u(z)
        Z = float(self.contract.exp_Z)
        r = int(self.contract.exp_r)
        e = int(self.contract.exp_extra_bits)
        f = self.f
        order = int(self.contract.exp_order) if order is None else int(order)
        if r < e:
            raise ContractViolation(f"EXP_PARAMS: need r >= extra bits, got r={r}, e={e}")
        with self.track("exp_neg", z.shape):
            zZ = self.add_public_real(z, Z)
            b = self.drelu(zZ)
            zc = self.add_public_real(self.beaver_mul(b, zZ), -Z)
            w = self.trunc_local(zc, r - e)
            t = self.add_public(w, _const_u64(1 << (f + e)))
            if order >= 2:
                t = ring.add(t, self.trunc_local(self.beaver_mul(w, w), f + e + 1))
            for _ in range(r):
                t = self.trunc_local(self.beaver_mul(t, t), f + e)
            return self.trunc_local(t, e)

    # -- normalised reciprocal / rsqrt -----------------------------------------------------
    def _threshold_bits(self, S: np.ndarray, thresholds: Sequence[float]) -> List[Optional[np.ndarray]]:
        """Batched DReLU(S - T_j); thresholds beyond the DCF bound are answered 0 (contract)."""
        bound = 2.0 ** (int(self.contract.dcf_domain_bits) - 1 - self.f)
        live = [j for j, T in enumerate(thresholds) if T < bound]
        out: List[Optional[np.ndarray]] = [None] * len(thresholds)
        if not live:
            return out
        stack = np.stack([self.add_public_real(S, -thresholds[j]) for j in live])
        bits = self.drelu(stack)
        for i, j in enumerate(live):
            out[j] = bits[i]
        return out

    def recip(self, S: Any, extra_bits: int = 0) -> np.ndarray:
        """1/S for S in [1, 2^(K+1)); output at scale 2^(f + extra_bits). Rounds 12."""
        S = _u(S)
        K = int(self.contract.recip_K)
        newton = int(getattr(self.contract, "recip_newton", 2))
        with self.track("recip", S.shape):
            bits = self._threshold_bits(S, [2.0 ** j for j in range(1, K + 1)])
            s = self.const(0.5, S.shape)
            for j, b in enumerate(bits, start=1):
                if b is not None:
                    s = ring.sub(s, ring.mul(b, self.enc(np.array(2.0 ** -(j + 1)))))
            v = self.mul_fxp(S, s)
            q = self.poly_contract(v, "recip_norm")
            for _ in range(newton):
                vq = self.mul_fxp(v, q)
                q = self.mul_fxp(q, self.add_public_real(ring.neg(vq), 2.0))
            return self.mul_fxp(q, s, self.f - int(extra_bits))

    def rsqrt(self, V: Any, extra_bits: int = 0) -> np.ndarray:
        """1/sqrt(V) for V in [4^j0, 4^(j0+m+1)) within the DCF bound; scale 2^(f + extra_bits). Rounds 14."""
        V = _u(V)
        j0 = int(self.contract.rsqrt_j0)
        m = int(self.contract.rsqrt_m)
        newton = int(getattr(self.contract, "rsqrt_newton", 2))
        f = self.f
        with self.track("rsqrt", V.shape):
            bits = self._threshold_bits(V, [4.0 ** (j0 + i) for i in range(1, m + 1)])
            # s2 = 4^-(j0+1+k) is an exact integer at scale 2^(f+e2) as soon as f + e2 >= 2 (j0+1+m).
            # The smallest such e2 keeps the Beaver product V * s2 = v * 2^(2f+e2) (v < 1) inside the
            # declared bound of the local truncation: 2^34 ring units under NC-v2, failure <= 2^-29 per
            # element. (e2 = f made it a 2^48 product: a wrong row with probability 2^-16.6.)
            e2 = rsqrt_scale_bits(self.contract)
            sc2 = float(1 << (f + e2))
            s2 = (np.full(V.shape, _const_u64(int(round(4.0 ** -(j0 + 1) * sc2))), np.uint64)
                  if self.party == 0 else np.zeros(V.shape, np.uint64))
            s1 = self.const(2.0 ** -(j0 + 1), V.shape)
            for i, cbit in enumerate(bits, start=1):
                if cbit is None:
                    continue
                k2 = int(round(4.0 ** -(j0 + 1) * 3.0 * 4.0 ** -i * sc2))
                k1 = int(round(2.0 ** -(j0 + 1) * 2.0 ** -i * (1 << f)))
                s2 = ring.sub(s2, ring.mul(cbit, _const_u64(k2)))
                s1 = ring.sub(s1, ring.mul(cbit, _const_u64(k1)))
            v = self.mul_fxp(V, s2, f + e2)
            q = self.poly_contract(v, "rsqrt_norm")
            for _ in range(newton):
                vq = self.mul_fxp(v, q)
                vqq = self.mul_fxp(vq, q)
                h = self.add_public_real(ring.neg(self.trunc_local(vqq, 1)), 1.5)
                q = self.mul_fxp(q, h)
            return self.mul_fxp(q, s1, f - int(extra_bits))

    # -- softmax / layernorm -------------------------------------------------------------------
    def softmax_rows(self, x: Any, mask: Optional[np.ndarray] = None, k_max: int = 2) -> np.ndarray:
        """Row softmax over the last axis with a public boolean mask (True = attend).

        Masked positions get exactly 0. Requires max - min of each row below
        2^(n-1-f) - Z (binary max) or the k-ary node bound (k_max > 2).
        """
        x = _u(x)
        n = int(x.shape[-1])
        with self.track(f"softmax_rows_k{int(k_max)}", x.shape):
            if mask is not None:
                mask = np.broadcast_to(np.asarray(mask, bool), x.shape)
            m = self.max_rows(x, mask, k=k_max)
            z = ring.sub(x, m[..., None])
            if mask is not None:
                z = np.array(z, copy=True)
                z[~mask] = self.enc(np.array(-float(self.contract.exp_Z))) if self.party == 0 else _ZERO
            t = self.exp_neg(z)
            if mask is not None:
                t = np.array(t, copy=True)
                t[~mask] = _ZERO
            S = t.sum(axis=-1, dtype=np.uint64)
            e = min(int(math.ceil(math.log2(max(n, 2)))), 10)
            inv = self.recip(S, extra_bits=e)
            return self.mul_fxp(t, inv[..., None], self.f + e)

    def layernorm_rows(self, x: Any, gamma: Optional[np.ndarray] = None, beta: Optional[np.ndarray] = None,
                       eps: float = 1e-5, rms: bool = False) -> np.ndarray:
        """LayerNorm (or RMSNorm) over the last axis; gamma / beta are SHARES (None = identity).

        The square dx*dx is truncated by the faithful gate (contract
        trunc_square_ln = faithful): the squared residual of a massive
        activation can exceed 2^50 ring units where local truncation fails
        with probability about 2^-13 per element. Rounds: 1 (square) + 1
        (faithful gate) + 14 (rsqrt) + 1 (dx rs) + 1 (gamma) = 18.
        """
        x = _u(x)
        d = int(x.shape[-1])
        extra = int(self.contract.layernorm_extra_bits)
        kind = "rmsnorm_rows" if rms else "layernorm_rows"
        with self.track(kind, x.shape):
            if rms:
                dx = x
            else:
                mu = self.mul_public_real(x.sum(axis=-1, dtype=np.uint64), 1.0 / d)
                dx = ring.sub(x, mu[..., None])
            sq = self.square_fxp(dx, exact=(str(getattr(self.contract, "trunc_square_ln", "faithful")) == "faithful"))
            var = self.mul_public_real(sq.sum(axis=-1, dtype=np.uint64), 1.0 / d)
            var = self.add_public_real(var, float(eps))
            rs = self.rsqrt(var, extra_bits=extra)
            y = self.mul_fxp(dx, rs[..., None], self.f + extra)
            if gamma is not None:
                y = self.mul_fxp(y, np.broadcast_to(_u(gamma), y.shape))
            if beta is not None:
                y = ring.add(y, np.broadcast_to(_u(beta), y.shape))
            return y

    def rmsnorm_rows(self, x: Any, gamma: Optional[np.ndarray] = None, eps: float = 1e-6) -> np.ndarray:
        return self.layernorm_rows(x, gamma, None, eps, rms=True)

    # -- conversions ----------------------------------------------------------------------------
    def ring_to_field(self, share: Any, t: int) -> np.ndarray:
        """A64 -> Z_t resharing of the signed representative (one round; paper, Section 5.2, RingToField).

        Opens y = u + r only; the wrap and sign bits stay shared. Output: this
        party's Z_t share (uint64 below t) of x_signed mod t.
        """
        share = _u(share)
        t = int(t)
        flat = share.reshape(-1)
        with self.track("ring_to_field", share.shape):
            mat = self.dealer.take_r2f(int(flat.size), t)
            if int(getattr(mat, "t", t)) != t:
                raise MaterialError(f"MATERIAL_PARAMS: r2f material is for t={getattr(mat, 't', None)}, requested {t}")
            self._consume(mat)
            y = self._open(ring.add(flat, _u(mat.r_share).reshape(-1)))
            z = _u(r2f_eval(mat, y, self.party))
            return z.reshape(share.shape)

    def a2h_exact(self, share: Any, t: int = 68720050177):
        """TenSEAL BFV exact A2H (ring_to_field, then P1 encrypts, P0 adds); see he_lane.a2h_exact."""
        from .he_lane import a2h_exact as _a2h
        return _a2h(self, share, t)

    def toy_bridge(self, share: Any, bfv: Any, pk: Any, sk: Any, W: Any) -> np.ndarray:
        """share -> a2h_toy -> eval_toy(W) -> h2a_ref -> A64 shares of W x mod t (paper, Appendix B, H2A-ref)."""
        from .bridge import a2h_toy, eval_toy, h2a_ref
        ct = a2h_toy(self, share, bfv, pk)
        positions = None
        if self.party == 0:
            ct, positions = eval_toy(bfv, ct, W)
        d_out = int(np.asarray(W).shape[0])
        d_in = int(np.asarray(W).shape[1])
        if positions is None:
            positions = np.arange(1, d_out + 1, dtype=np.int64) * d_in - 1
        return h2a_ref(self, bfv, ct if self.party == 0 else None, sk if self.party == 1 else None, positions)


__all__ = ["Party", "ReleasePolicy", "GadgetStats", "RELEASE_KINDS", "ARGMAX_K", "MAX_K"]
