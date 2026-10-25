"""Calibrated per-component cost model (paper, Section 4.2, objective; Section 7.3, accuracy of the cost model).

Model. For every component c and public shape (n, d_in, d_out) the predicted
wall time of one call under an environment ``env`` is

    t = (a_c * MACs + k_c * bytes + b_c) / cpu_share + bytes * 8 / BW + rounds * RTT

where ``bytes`` and ``rounds`` come from the deterministic protocol formulas
of this file (:func:`bytes_formula`, :func:`rounds_formula`) and
``(a_c, k_c, b_c)`` are fitted by non-negative least squares on calibration
rows (:meth:`FittedCostModel.fit`).  The uncertainty of a prediction is

    sigma = sqrt(sigma_c^2 / cpu_share^2 + (bytes*8/BW * rel_bw)^2 + (rounds*RTT * rel_rtt)^2)

with ``sigma_c`` the repeat spread of the compute residual on the training
rows and ``rel_bw``, ``rel_rtt`` the declared relative uncertainty of the
bandwidth and RTT estimates.

Shape conventions (every formula uses the same ones):

* ``linear_wmask``, ``linear_generic``, ``matmul_shared``: X (n x d_in) times
  W (d_in x d_out).
* ``embedding_wmask``, ``embedding_generic``: n one-hot rows, d_in = V
  (vocabulary), d_out = d_model.
* ``install_wmask``: one matrix d_in x d_out (n is ignored).
* ``gelu_clip12``, ``trunc_faithful``, ``ring_to_field``, ``a2h_exact``:
  elementwise on n * d_in elements (d_out ignored).
* ``softmax_rows_k2``, ``softmax_rows_k4``: n rows of length d_in (keys).
* ``layernorm_rows``: n rows of width d_in, with the affine gamma product.
* ``argmax_k2``, ``argmax_k4``, ``argmax_k8``: n rows of d_in logits.

What the formulas price.  Every
formula is derived from :func:`gate_counts`, which counts exactly the gates
that ``party.Party`` issues for one call of the component:

* one opened ring element costs 16 bytes (8 from each party); a Beaver product
  opens two masked elements per element (32 bytes) and consumes one 24-byte
  triple; a DReLU opens one element and consumes two DCF keys plus three ring
  shares (1672 bytes without payload, 2200 bytes with the product payload at
  n = 32); the faithful truncation and the ring-to-field gate open one element
  each (material 2064 and 3272 bytes).
* linear: ``linear_wmask`` opens X - A (16 n d_in bytes); ``linear_generic``,
  ``embedding_generic`` and ``matmul_shared`` open X - A and W - B
  (16 (n d_in + d_in d_out) bytes, both directions for every element);
  ``install_wmask`` is one message B1 up and one message E down
  (16 d_in d_out bytes in total); all take one round.
* ``gelu_clip12``: 2 DReLUs and degree + 2 Beaver elements per element,
  rounds 1 + degree + 1.
* ``exp_neg``: 1 DReLU and 2 + r Beaver elements per element, rounds 3 + r.
  ``recip``: one DReLU per live threshold and 1 + degree + 2 newton + 1 Beaver
  elements per row.  ``rsqrt``: one DReLU per live threshold and
  1 + degree + 3 newton + 1 Beaver elements per row.  A threshold is live when
  it lies below the DCF bound 2^(n-1-f) real units: all K = 11 thresholds of
  recip, 14 of the m = 15 thresholds of rsqrt under the default contract.
* ``layernorm_rows``: three Beaver products and one faithful truncation per
  element (the square is a full Beaver product, two openings), plus rsqrt per
  row; rounds 1 + 1 + 14 + 1 + 1 = 18.
* fused k-ary tournament (``argmax_k2/k4/k8`` with index, the row maximum of
  ``softmax_rows_k2/k4`` without): per level the surviving w items are grouped
  into nodes of at most k.  Round 1: g (g - 1) / 2 pairwise DReLUs without
  payload per node of g members.  Round 2: a node of 2 members spends one
  Beaver select for the value (plus one for the index once indices are shared,
  that is on every level after the first); a node of g >= 3 members spends
  g - 1 DReLUs with the product payload for the value (and g - 1 more for a
  shared index).  Both rounds are single openings: 2 rounds per level,
  2 ceil(log_k V) in total.  :func:`tournament_gates` returns the counts; they
  equal ``dealer.tournament_material`` and ``session.tournament_counts``.
  The formulas price the unmasked row; a public mask only removes pairwise
  comparisons (and the comparison round of a level it decides completely), so
  the figures are exact without a mask and upper bounds with one.
* ``argmax_k*`` is the tournament alone (``argmax_row(release=False)``); the
  RELEASE_FINAL message of the index share (8 bytes per row, one message, one
  round) is a separate step and is not part of the component.
* ``a2h_exact``: ring_to_field (one round, 16 bytes per element) and one
  message P1 -> P0 carrying ceil(N / 8192) BFV ciphertexts; the ciphertext
  size is MEASURED (:data:`BFV_CT_BYTES`, :func:`measure_bfv_ct_bytes`).

Byte accounting.  :func:`payload_bytes` is the opened payload over both
directions (raw ring elements and ciphertext bytes).  Framing rule: every
message is one frame with an 8-byte length prefix and a typed-value header;
:data:`FRAME_BYTES` = 19 is the framing of a frame that carries one 1-D uint64
array under the wire codec of ``channel.py`` (8-byte prefix, tag, dtype, ndim
and one u64 dimension); every further array dimension adds 8 bytes.
``bytes_formula = payload_bytes + FRAME_BYTES * messages_formula``.  Message
sizes depend on public shapes only.

Shared formula file.  :func:`write_formulas_json` writes
``results/calibration/formulas.json``: the constants, the measured BFV
ciphertext size with its provenance, the tournament gate counts and, for the
declared list :func:`formula_shapes` (GPT-2 shapes and the small shapes of
the measured test), payload bytes, messages, bytes, rounds, gate counts and
material.  The C++ cost model is checked against this file.

Evidence status. Every :class:`ComponentFit` carries ``status``:
``MEASURED_NATIVE`` (fitted on rows measured by this tree) or ``MODELED``
(derived from another fit or hand set).  The paper marks any MODELED figure
as an estimate.  :func:`default_model` returns the fitted model of
:func:`calibration_path` (the shipped ``results/calibration/fitted_model.json``,
or the copy of a rerun root named by ``OPENCRAB2_RESULTS_DIR``), else the
hand-written MODELED baseline.

What is tested (tests/test_costmodel.py): every component is run once per
declared small shape in the in-process two-party harness and the measured
channel bytes minus the measured framing, the message count, the rounds and
the consumed material equal the formulas (a2h_exact: exact on the
ring-to-field part, within the measured spread on the ciphertext); the
tournament counts equal the dealer and session functions; the NNLS fit
recovers planted coefficients from synthetic rows with zero residual; predict
returns a finite (mean, sigma) that moves with bandwidth and RTT; JSON round
trip; formulas.json is reproducible; the default model follows the
calibration file of the data root; error metrics on known vectors.  Design
limits: the fit is affine per component (no cross terms),
the transport model is a fluid bandwidth plus a per-round RTT (no slow
start), the framing term is the nominal 1-D frame figure, and the TenSEAL
ciphertext serialization is compressed, so its size varies by about 0.05
percent around the measured value.
"""

from __future__ import annotations

import json
import math
import os
import time as _time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..errors import ContractViolation

# ---------------------------------------------------------------------------
# component vocabulary
# ---------------------------------------------------------------------------
COMPONENTS: Tuple[str, ...] = (
    "linear_wmask", "linear_generic", "embedding_wmask", "embedding_generic",
    "install_wmask", "gelu_clip12", "softmax_rows_k2", "softmax_rows_k4",
    "layernorm_rows", "argmax_k2", "argmax_k4", "argmax_k8", "trunc_faithful",
    "ring_to_field", "a2h_exact", "matmul_shared",
)
LINEAR_COMPONENTS: Tuple[str, ...] = ("linear_wmask", "linear_generic", "embedding_wmask",
                                      "embedding_generic", "matmul_shared")
ELEMENTWISE_COMPONENTS: Tuple[str, ...] = ("gelu_clip12", "trunc_faithful", "ring_to_field", "a2h_exact")
ROW_COMPONENTS: Tuple[str, ...] = ("softmax_rows_k2", "softmax_rows_k4", "layernorm_rows",
                                   "argmax_k2", "argmax_k4", "argmax_k8")
KARY_COMPONENTS: Tuple[str, ...] = ("softmax_rows_k2", "softmax_rows_k4", "argmax_k2", "argmax_k4", "argmax_k8")
TOURNAMENT_K: Tuple[int, ...] = (2, 4, 8)


# protocol constants (paper, Section 5, building blocks)
# framing of one frame that carries a 1-D uint64 array under the wire codec of channel.py (identical in
# Python and C++): 8-byte length prefix + tag (1) + dtype (1) + ndim (1) + one u64 dimension (8)
FRAME_BYTES = 19
FRAME_DIM_BYTES = 8          # every further array dimension adds one u64 to the header
OPEN_BYTES = 16              # one opened ring element: 8 bytes from each party
BEAVER_OPEN_BYTES = 32       # one Beaver product opens two masked elements
SELECT_BYTES = 32            # one select is one Beaver product of a bit and a difference
ONEWAY_BYTES = 8             # one ring element sent in one direction only (install)
BEAVER_BYTES = 24            # material of one Beaver element (a, b, c shares)
BFV_N = 8192                 # a2h_exact batching slots
BFV_T = 68720050177          # admitted BFV plaintext modulus
# MEASURED: len(tenseal.bfv_vector(ctx, 8192 values).serialize()) at N = 8192, t = 68720050177 with the
# TenSEAL default coefficient modulus (2026-10-10, tenseal 0.3.17).  The serialization is compressed:
# a re-measurement over 400 ciphertexts gave mean 432453, min 432278, max 432653 (std 64), see
# measure_bfv_ct_bytes and the provenance block of formulas.json.
BFV_CT_BYTES = 432404
BFV_CT_REL_TOL = 1.0e-3      # accepted relative deviation of a re-measured mean from BFV_CT_BYTES
DCF_DOMAIN_BITS = 32         # comparisons use n = 32 (contract dcf_domain_bits)
RING_BITS = 64
FRAC_BITS = 16
GELU_DEGREE = 12
EXP_ROUNDS = 10              # order-2 base, r = 7 squarings (contract exp)
RECIP_ROUNDS = 12            # K = 11 thresholds, degree 5, 2 Newton steps
RSQRT_ROUNDS = 14            # m = 15 thresholds, degree 5, 2 Newton steps
LAYERNORM_ROUNDS = 18        # square (1) + faithful gate (1) + rsqrt (14) + dx*rs (1) + gamma (1)
RECIP_K = 11
RSQRT_M = 15
RECIP_LIVE = 11              # thresholds 2^1 .. 2^11 all lie below the DCF bound 2^15
RSQRT_LIVE = 14              # thresholds 4^-6 .. 4^7 lie below 2^15; 4^8 does not and is answered 0
EXP_PRODUCTS = 9             # clamp select + base square + 7 squarings
RECIP_PRODUCTS = 11          # 1 normalisation + 5 Horner + 4 Newton + 1 rescale
RSQRT_PRODUCTS = 13          # 1 + 5 + 6 + 1
GELU_PRODUCTS = GELU_DEGREE + 2   # Horner + two blend products
GELU_COMPARISONS = 2         # clip at -R and +R


@dataclass(frozen=True)
class Env:
    """Public environment estimate: bandwidth (bit/s, None or 0 = unlimited), one-way delay, CPU share."""
    bandwidth_bps: float
    one_way_latency_s: float
    cpu_share: float = 1.0

    @property
    def rtt_s(self) -> float:
        return 2.0 * float(self.one_way_latency_s)


def as_env(env: Any) -> Env:
    """Accept an Env, an object with bandwidth_bps and rtt_s or one_way_latency_s, or a dict."""
    if isinstance(env, Env):
        return env
    if isinstance(env, Mapping):
        bw = env.get("bandwidth_bps")
        cpu = float(env.get("cpu_share", 1.0))
        if "one_way_latency_s" in env:
            return Env(bw, float(env["one_way_latency_s"]), cpu)
        return Env(bw, float(env["rtt_s"]) / 2.0, cpu)
    cpu = float(getattr(env, "cpu_share", 1.0))
    if hasattr(env, "one_way_latency_s"):
        return Env(getattr(env, "bandwidth_bps", None), float(env.one_way_latency_s), cpu)
    if hasattr(env, "rtt_s"):
        return Env(getattr(env, "bandwidth_bps", None), float(env.rtt_s) / 2.0, cpu)
    raise TypeError(f"ENV_TYPE: cannot interpret environment {env!r}")


class EnvError(ContractViolation, ValueError):
    """An environment estimate outside its domain (code ENV_PARAMS; the C++ model returns the same code)."""

    default_code = "ENV_PARAMS"


def check_env(env: Any) -> Env:
    """``as_env`` plus validation: cpu_share finite and > 0, latency finite and >= 0, bandwidth None or finite
    and >= 0 (None or 0 = unlimited). A value outside the domain is refused with ``ENV_PARAMS``; it used to be
    repaired silently (cpu_share <= 0 priced as 1.0) or to propagate as NaN (bandwidth NaN gave (nan, nan))."""
    e = as_env(env)
    try:
        cpu = float(e.cpu_share)
        lat = float(e.one_way_latency_s)
        bw = None if e.bandwidth_bps is None else float(e.bandwidth_bps)
    except (TypeError, ValueError):
        raise EnvError("ENV_PARAMS: environment fields must be numbers") from None
    if not math.isfinite(cpu) or cpu <= 0.0:
        raise EnvError(f"ENV_PARAMS: cpu_share must be finite and > 0, got {e.cpu_share!r}")
    if not math.isfinite(lat) or lat < 0.0:
        raise EnvError(f"ENV_PARAMS: one_way_latency_s must be finite and >= 0, got {e.one_way_latency_s!r}")
    if bw is not None and (not math.isfinite(bw) or bw < 0.0):
        raise EnvError(f"ENV_PARAMS: bandwidth_bps must be None or finite and >= 0, got {e.bandwidth_bps!r}")
    return e


def _check_component(component: str) -> None:
    if component not in COMPONENTS:
        raise ValueError(f"UNKNOWN_COMPONENT: {component!r} is not one of {list(COMPONENTS)}")


def _pos(name: str, v: int) -> int:
    v = int(v)
    if v < 1:
        raise ValueError(f"SHAPE_PARAM: {name} must be >= 1, got {v}")
    return v


# ---------------------------------------------------------------------------
# protocol parameters read from the numeric contract
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProtocolParams:
    """The contract fields the gate counts depend on (defaults: NC-v2-f16-l40)."""
    frac_bits: int = FRAC_BITS
    dcf_domain_bits: int = DCF_DOMAIN_BITS
    ring_bits: int = RING_BITS
    exp_r: int = 7
    exp_order: int = 2
    recip_K: int = RECIP_K
    recip_degree: int = 5
    recip_newton: int = 2
    rsqrt_m: int = RSQRT_M
    rsqrt_j0: int = -7
    rsqrt_degree: int = 5
    rsqrt_newton: int = 2
    gelu_degree: int = GELU_DEGREE
    faithful_square: bool = True

    # -- derived counts (per element or per row) ---------------------------------
    @property
    def compare_bound_log2(self) -> int:
        """log2 of the DCF bound in real units: thresholds at or above 2^(n-1-f) are not compared."""
        return self.dcf_domain_bits - 1 - self.frac_bits

    @property
    def recip_live(self) -> int:
        """Thresholds 2^j, j = 1..K, below the DCF bound (``Party._threshold_bits``)."""
        return sum(1 for j in range(1, self.recip_K + 1) if j < self.compare_bound_log2)

    @property
    def rsqrt_live(self) -> int:
        """Thresholds 4^(j0+i), i = 1..m, below the DCF bound."""
        return sum(1 for i in range(1, self.rsqrt_m + 1) if 2 * (self.rsqrt_j0 + i) < self.compare_bound_log2)

    @property
    def exp_products(self) -> int:
        return 1 + (1 if self.exp_order >= 2 else 0) + self.exp_r

    @property
    def exp_rounds(self) -> int:
        return 1 + self.exp_products

    @property
    def recip_products(self) -> int:
        return 1 + self.recip_degree + 2 * self.recip_newton + 1

    @property
    def recip_rounds(self) -> int:
        return (1 if self.recip_live else 0) + self.recip_products

    @property
    def rsqrt_products(self) -> int:
        return 1 + self.rsqrt_degree + 3 * self.rsqrt_newton + 1

    @property
    def rsqrt_rounds(self) -> int:
        return (1 if self.rsqrt_live else 0) + self.rsqrt_products

    @property
    def gelu_products(self) -> int:
        return self.gelu_degree + 2

    @property
    def gelu_rounds(self) -> int:
        return 1 + self.gelu_degree + 1


def _degree_from_name(name: Any, default: int) -> int:
    digits = "".join(ch for ch in str(name) if ch.isdigit())
    return int(digits) if digits else int(default)


def protocol_params(contract: Any = None) -> ProtocolParams:
    """ProtocolParams of a NumericContract (None = the default contract constants)."""
    if contract is None:
        return ProtocolParams()
    if isinstance(contract, ProtocolParams):
        return contract
    d = ProtocolParams()

    def g(name: str, default: int) -> int:
        return int(getattr(contract, name, default))

    return ProtocolParams(
        frac_bits=g("frac_bits", d.frac_bits), dcf_domain_bits=g("dcf_domain_bits", d.dcf_domain_bits),
        ring_bits=g("ring_bits", d.ring_bits), exp_r=g("exp_r", d.exp_r), exp_order=g("exp_order", d.exp_order),
        recip_K=g("recip_K", d.recip_K), recip_degree=g("recip_degree", d.recip_degree),
        recip_newton=g("recip_newton", d.recip_newton), rsqrt_m=g("rsqrt_m", d.rsqrt_m),
        rsqrt_j0=g("rsqrt_j0", d.rsqrt_j0), rsqrt_degree=g("rsqrt_degree", d.rsqrt_degree),
        rsqrt_newton=g("rsqrt_newton", d.rsqrt_newton),
        gelu_degree=_degree_from_name(getattr(contract, "gelu", "gelu_clip12"), d.gelu_degree),
        faithful_square=str(getattr(contract, "trunc_square_ln", "faithful")) == "faithful")


# ---------------------------------------------------------------------------
# tournament geometry (shared by argmax_k and the softmax max)
# ---------------------------------------------------------------------------
def kary_tournament(V: int, k: int) -> Tuple[int, int]:
    """(levels, eliminations) of a k-ary max tournament over V elements.

    Levels = ceil(log_k V) (0 for V = 1); every element but the winner is
    eliminated exactly once, so the second figure is V - 1 for every k.  This
    is the declared accounting used for rounds (2 per level).  The gates the
    fused realization issues are NOT V - 1 per kind: see :func:`tournament_gates`.
    """
    V = _pos("V", V)
    k = int(k)
    if k < 2:
        raise ValueError(f"SHAPE_PARAM: tournament arity k must be >= 2, got {k}")
    levels, elim, w = 0, 0, V
    while w > 1:
        nodes = (w + k - 1) // k
        elim += w - nodes
        w = nodes
        levels += 1
    return levels, elim


def tournament_gates(V: int, k: int, with_index: bool) -> Tuple[int, int, int, int]:
    """(levels, DReLU without payload, DReLU with payload, Beaver selects) of one fused k-ary tournament row.

    Exactly what ``party.Party._tournament`` issues on an unmasked row of V
    items (module docstring).  The last three figures equal
    ``dealer.tournament_material(V, k, with_index)`` and
    ``session.tournament_counts`` (tested).  Closed form per level with w
    survivors: w // k nodes of k members and one node of w % k members.
    """
    V = _pos("V", V)
    k = int(k)
    if k < 2:
        raise ValueError(f"SHAPE_PARAM: tournament arity k must be >= 2, got {k}")
    levels = p1 = p2 = bv = 0
    w = V
    idx_public = True
    while w > 1:
        extra = 1 if (with_index and not idx_public) else 0
        for g, count in ((k, w // k), (w % k, 1)):
            if g < 2 or count < 1:
                continue
            p1 += count * (g * (g - 1) // 2)
            if g == 2:
                bv += count * (1 + extra)
            else:
                p2 += count * (g - 1) * (1 + extra)
        w = (w + k - 1) // k
        idx_public = False
        levels += 1
    return levels, p1, p2, bv


def tournament_node_sizes(V: int, k: int) -> Tuple[int, ...]:
    """Sorted distinct node sizes g >= 2 that occur in a k-ary tournament over V items (public shape)."""
    V = _pos("V", V)
    k = int(k)
    if k < 2:
        raise ValueError(f"SHAPE_PARAM: tournament arity k must be >= 2, got {k}")
    sizes = set()
    w = V
    while w > 1:
        if w // k:
            sizes.add(k)
        if w % k >= 2:
            sizes.add(w % k)
        w = (w + k - 1) // k
    return tuple(sorted(sizes))


def _k_of(component: str) -> int:
    return int(component.rsplit("_k", 1)[1])


# ---------------------------------------------------------------------------
# gate counts: the single source of every structural formula
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GateCounts:
    """Gates, openings, rounds and messages of ONE call of a component (the two parties are symmetric)."""
    drelu_p1: int = 0        # DReLU gates without payload (comparison bit only)
    drelu_p2: int = 0        # DReLU gates with the product payload (fused node selects)
    beaver: int = 0          # Beaver product elements from the general pool
    kary_select: int = 0     # Beaver select elements of binary tournament nodes
    trunc: int = 0           # faithful truncation gates
    r2f: int = 0             # ring-to-field gates
    matrix_triple: int = 0   # matrix triple elements (A, B, C)
    wmask_call: int = 0      # per-call weight-mask elements (A, C)
    wmask_mask: int = 0      # reusable mask elements (B), consumed by the install
    open_elems: int = 0      # other ring elements opened by both parties (linear openings)
    oneway_elems: int = 0    # ring elements sent in one direction only (install: B1 up, E down)
    bfv_ct: int = 0          # BFV ciphertexts sent P1 -> P0
    rounds: int = 0
    messages: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {k: int(v) for k, v in asdict(self).items()}


def gate_counts(component: str, n: int, d_in: int, d_out: int, contract: Any = None) -> GateCounts:
    """Exact gate, opening, round and message counts of one call (module docstring)."""
    _check_component(component)
    n, d_in = _pos("n", n), _pos("d_in", d_in)
    P = protocol_params(contract)
    if component in ("linear_wmask", "embedding_wmask"):
        d_out = _pos("d_out", d_out)
        return GateCounts(wmask_call=n * d_in + n * d_out, open_elems=n * d_in, rounds=1, messages=2)
    if component in ("linear_generic", "embedding_generic", "matmul_shared"):
        d_out = _pos("d_out", d_out)
        return GateCounts(matrix_triple=n * d_in + d_in * d_out + n * d_out, open_elems=n * d_in + d_in * d_out,
                          rounds=1, messages=2)
    if component == "install_wmask":
        d_out = _pos("d_out", d_out)
        return GateCounts(wmask_mask=d_in * d_out, oneway_elems=2 * d_in * d_out, rounds=1, messages=2)
    N = n * d_in
    if component == "trunc_faithful":
        return GateCounts(trunc=N, rounds=1, messages=2)
    if component == "ring_to_field":
        return GateCounts(r2f=N, rounds=1, messages=2)
    if component == "a2h_exact":
        # ring_to_field opening (one exchange), then ONE message P1 -> P0 with every ciphertext
        return GateCounts(r2f=N, bfv_ct=(N + BFV_N - 1) // BFV_N, rounds=2, messages=3)
    if component == "gelu_clip12":
        r = P.gelu_rounds
        return GateCounts(drelu_p1=GELU_COMPARISONS * N, beaver=P.gelu_products * N, rounds=r, messages=2 * r)
    if component == "layernorm_rows":
        r = 1 + (1 if P.faithful_square else 0) + P.rsqrt_rounds + 1 + 1
        return GateCounts(drelu_p1=P.rsqrt_live * n, beaver=3 * N + P.rsqrt_products * n,
                          trunc=N if P.faithful_square else 0, rounds=r, messages=2 * r)
    if component.startswith("softmax_rows_k"):
        levels, p1, p2, bv = tournament_gates(d_in, _k_of(component), False)
        r = 2 * levels + P.exp_rounds + P.recip_rounds + 1
        return GateCounts(drelu_p1=n * p1 + N + P.recip_live * n, drelu_p2=n * p2, kary_select=n * bv,
                          beaver=P.exp_products * N + P.recip_products * n + N, rounds=r, messages=2 * r)
    if component.startswith("argmax_k"):
        levels, p1, p2, bv = tournament_gates(d_in, _k_of(component), True)
        return GateCounts(drelu_p1=n * p1, drelu_p2=n * p2, kary_select=n * bv, rounds=2 * levels, messages=4 * levels)
    raise ValueError(f"UNKNOWN_COMPONENT: {component!r}")


# ---------------------------------------------------------------------------
# deterministic structural formulas
# ---------------------------------------------------------------------------
def elements(component: str, n: int, d_in: int, d_out: int) -> int:
    """Element count the compute term scales with for non-linear components."""
    _check_component(component)
    n, d_in = _pos("n", n), _pos("d_in", d_in)
    if component in ELEMENTWISE_COMPONENTS or component in ROW_COMPONENTS:
        return n * d_in
    if component == "install_wmask":
        return d_in * _pos("d_out", d_out)
    return n * d_in


def macs(component: str, n: int, d_in: int, d_out: int) -> int:
    """Work unit of the compute term: MACs for linear components, elements otherwise."""
    _check_component(component)
    if component in LINEAR_COMPONENTS:
        return _pos("n", n) * _pos("d_in", d_in) * _pos("d_out", d_out)
    if component == "install_wmask":
        return _pos("d_in", d_in) * _pos("d_out", d_out)
    return elements(component, n, d_in, d_out)


def rounds_formula(component: str, n: int, d_in: int, d_out: int) -> int:
    """Online rounds of one call (shape-deterministic; 0 for an argmax over a single item)."""
    return int(gate_counts(component, n, d_in, d_out).rounds)


def messages_formula(component: str, n: int, d_in: int, d_out: int) -> int:
    """Frames of one call over both directions (two per exchange round; a2h_exact: two plus the ciphertext frame)."""
    return int(gate_counts(component, n, d_in, d_out).messages)


def _payload_of(g: GateCounts, ct_bytes: int) -> int:
    return int(OPEN_BYTES * (g.drelu_p1 + g.drelu_p2 + g.trunc + g.r2f + g.open_elems)
               + BEAVER_OPEN_BYTES * (g.beaver + g.kary_select)
               + ONEWAY_BYTES * g.oneway_elems
               + int(ct_bytes) * g.bfv_ct)


def payload_bytes(component: str, n: int, d_in: int, d_out: int) -> int:
    """Opened payload bytes of one call, both directions, without framing (exact, see gate_counts)."""
    g = gate_counts(component, n, d_in, d_out)
    return _payload_of(g, bfv_ct_bytes() if g.bfv_ct else 0)


def bytes_formula(component: str, n: int, d_in: int, d_out: int) -> int:
    """Total online bytes of one call: payload plus FRAME_BYTES of framing per message."""
    g = gate_counts(component, n, d_in, d_out)
    return int(_payload_of(g, bfv_ct_bytes() if g.bfv_ct else 0) + FRAME_BYTES * g.messages)


# material sizes per element and party (paper, Section 5, secure building blocks)
def dcf_key_bytes(n_bits: int, payloads: int) -> int:
    return 16 + int(n_bits) * (16 + 8 * int(payloads) + 1) + 8 * int(payloads)


def drelu_material_bytes(n_bits: int = DCF_DOMAIN_BITS, payloads: int = 1) -> int:
    """Two DCF keys plus three ring shares (r, 1{p>q}, 1{p>q} r): 1672 bytes (P = 1), 2200 bytes (P = 2) at n = 32."""
    return 2 * dcf_key_bytes(n_bits, payloads) + 3 * 8


def trunc_material_bytes(f: int) -> int:
    """r share, floor(r/2^f) share, 64-bit wrap key, f-bit borrow key: 2064 bytes at f = 16."""
    return 16 + dcf_key_bytes(RING_BITS, 1) + dcf_key_bytes(int(f), 1)


def r2f_material_bytes() -> int:
    """r share, r mod t share, c share, two 64-bit keys with Z_t payload: 3272 bytes."""
    return 3 * 8 + 2 * dcf_key_bytes(RING_BITS, 1)


def _contract_bits(contract: Any) -> Tuple[int, int]:
    P = protocol_params(contract)
    return P.frac_bits, P.dcf_domain_bits


def _material_of(g: GateCounts, P: ProtocolParams) -> Dict[str, int]:
    out: Dict[str, int] = {}
    drelu = (g.drelu_p1 * drelu_material_bytes(P.dcf_domain_bits, 1)
             + g.drelu_p2 * drelu_material_bytes(P.dcf_domain_bits, 2))
    if drelu:
        out["drelu"] = drelu
    if g.beaver:
        out["beaver"] = BEAVER_BYTES * g.beaver
    if g.kary_select:
        out["kary_select"] = BEAVER_BYTES * g.kary_select
    if g.trunc:
        out["trunc_faithful"] = trunc_material_bytes(P.frac_bits) * g.trunc
    if g.r2f:
        out["ring_to_field"] = r2f_material_bytes() * g.r2f
    if g.matrix_triple:
        out["matrix_triple"] = 8 * g.matrix_triple
    if g.wmask_call or g.wmask_mask:
        out["wmask_mask"] = 8 * (g.wmask_call + g.wmask_mask)
    return {k: int(v) for k, v in out.items()}


def material_breakdown(component: str, n: int, d_in: int, d_out: int,
                       contract: Any = None) -> Dict[str, int]:
    """Per-party material bytes of one call, by material type (cost-model vocabulary).

    ``drelu`` sums the gates without payload and with the product payload;
    ``kary_select`` is the Beaver material of the binary-node selects (the
    party draws it from the Beaver pool); ``wmask_mask`` covers the per-call
    (A, C = A B) shares of the weight-masked linear and the reusable mask B of
    an install.  Types with zero need are omitted.
    """
    return _material_of(gate_counts(component, n, d_in, d_out, contract), protocol_params(contract))


def material_formula(component: str, n: int, d_in: int, d_out: int, contract: Any = None) -> int:
    """Per-party material bytes consumed by one call (sum over material types)."""
    return int(sum(material_breakdown(component, n, d_in, d_out, contract).values()))


def transport_s(nbytes: float, rounds: float, env: Env) -> float:
    t = 0.0
    if env.bandwidth_bps:
        t += float(nbytes) * 8.0 / float(env.bandwidth_bps)
    t += float(rounds) * env.rtt_s
    return t


# ---------------------------------------------------------------------------
# shared formula file (results/calibration/formulas.json)
# ---------------------------------------------------------------------------
FORMULAS_SCHEMA = "opencrab2-formulas-v1"
GPT2_CARD = {"d_model": 768, "d_ff": 3072, "n_head": 12, "vocab": 50257}

# three or more small shapes per component: the measured test of tests/test_costmodel.py runs each of them
# once in the two-party harness and compares the channel counters with the formulas
FORMULA_TEST_SHAPES: Dict[str, Tuple[Tuple[int, int, int], ...]] = {
    "linear_wmask": ((1, 8, 3), (2, 5, 7), (3, 6, 6)),
    "linear_generic": ((1, 8, 3), (2, 5, 7), (3, 6, 6)),
    "matmul_shared": ((1, 8, 3), (2, 5, 7), (3, 6, 6)),
    "embedding_wmask": ((1, 11, 4), (2, 9, 3), (3, 16, 5)),
    "embedding_generic": ((1, 11, 4), (2, 9, 3), (3, 16, 5)),
    "install_wmask": ((1, 8, 3), (1, 5, 7), (1, 16, 16)),
    "gelu_clip12": ((1, 8, 8), (2, 5, 5), (3, 16, 16)),
    "softmax_rows_k2": ((1, 8, 8), (2, 5, 5), (3, 17, 17), (2, 1, 1)),
    "softmax_rows_k4": ((1, 8, 8), (2, 5, 5), (3, 17, 17), (2, 1, 1)),
    "layernorm_rows": ((1, 8, 8), (2, 5, 5), (3, 16, 16)),
    "argmax_k2": ((1, 8, 1), (2, 5, 1), (3, 17, 1), (2, 70, 1), (1, 2, 1)),
    "argmax_k4": ((1, 8, 1), (2, 5, 1), (3, 17, 1), (2, 70, 1), (1, 2, 1)),
    "argmax_k8": ((1, 8, 1), (2, 5, 1), (3, 17, 1), (2, 70, 1), (1, 2, 1)),
    "trunc_faithful": ((1, 8, 8), (2, 5, 5), (3, 16, 16)),
    "ring_to_field": ((1, 8, 8), (2, 5, 5), (3, 16, 16)),
    "a2h_exact": ((1, 8, 8), (2, 5, 5), (1, 8193, 8193)),
}


def formula_shapes() -> List[Tuple[str, int, int, int]]:
    """Declared (component, n, d_in, d_out) list of formulas.json: GPT-2 shapes, then the small test shapes.

    GPT-2 (d_model 768, d_ff 3072, 12 heads, vocabulary 50257): n = 1 (decode
    step) and n = 16 (prefill) rows; KV lengths 16, 17, 24, 32 and 48.
    """
    d, f, H, V = GPT2_CARD["d_model"], GPT2_CARD["d_ff"], GPT2_CARD["n_head"], GPT2_CARD["vocab"]
    dh = d // H
    rows = (1, 16)
    kv = (16, 17, 24, 32, 48)
    out: List[Tuple[str, int, int, int]] = []
    for n in rows:
        out.append(("embedding_wmask", n, V, d))
        out.append(("embedding_generic", n, V, d))
        for comp in ("linear_wmask", "linear_generic"):
            for d_in, d_out in ((d, 3 * d), (d, d), (d, f), (f, d)):
                out.append((comp, n, d_in, d_out))
        out.append(("layernorm_rows", n, d, d))
        out.append(("gelu_clip12", n, f, f))
        for comp in ("trunc_faithful", "ring_to_field", "a2h_exact"):
            out.append((comp, n, d, d))
        for T in kv:
            out.append(("matmul_shared", n, dh, T))
            out.append(("matmul_shared", n, T, dh))
            out.append(("softmax_rows_k2", H * n, T, T))
            out.append(("softmax_rows_k4", H * n, T, T))
    out.append(("linear_wmask", 1, d, V))                                 # LM head on the last row
    for d_in, d_out in ((V, d), (d, V), (d, 3 * d), (d, d), (d, f), (f, d)):
        out.append(("install_wmask", 1, d_in, d_out))
    for comp in ("argmax_k2", "argmax_k4", "argmax_k8"):
        out.append((comp, 1, V, 1))
        out.append((comp, 1, 1000, 1))
    for comp in ("trunc_faithful", "ring_to_field", "a2h_exact"):
        out.append((comp, 1, BFV_N, BFV_N))
    for comp in COMPONENTS:
        for n, d_in, d_out in FORMULA_TEST_SHAPES[comp]:
            out.append((comp, n, d_in, d_out))
    seen = set()
    uniq: List[Tuple[str, int, int, int]] = []
    for item in out:
        if item not in seen:
            seen.add(item)
            uniq.append(item)
    return uniq


def shipped_calibration_dir() -> Path:
    """``<tree>/results/calibration``: the calibration that ships with the code."""
    return Path(__file__).resolve().parents[3] / "results" / "calibration"


def formulas_path() -> Path:
    """results/calibration/formulas.json of this tree (the shipped file, never a rerun's copy).

    The formulas are a pure function of this module; the file is shared with
    the C++ cost model and its hash is recorded by the calibration, so every
    reader uses the one shipped copy.
    """
    return shipped_calibration_dir() / "formulas.json"


_FORMULAS_CACHE: Optional[dict] = None
_BFV_CT_CACHE: Optional[int] = None


def load_formulas(path: Any = None) -> dict:
    """Parse formulas.json (cached for the default path).

    Raises ``FileNotFoundError`` (``FORMULAS_JSON_MISSING``) when the file is
    absent and ``ValueError`` (``FORMULAS_JSON_INVALID``) when it does not
    carry the expected schema; nothing is approximated silently.
    """
    global _FORMULAS_CACHE
    if path is None and _FORMULAS_CACHE is not None:
        return _FORMULAS_CACHE
    p = Path(path) if path is not None else formulas_path()
    if not p.is_file():
        raise FileNotFoundError(f"FORMULAS_JSON_MISSING: {p}")
    try:
        with open(p, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as exc:
        raise ValueError(f"FORMULAS_JSON_INVALID: {p} is unreadable ({exc.__class__.__name__})") from exc
    if not isinstance(doc, dict) or doc.get("schema") != FORMULAS_SCHEMA:
        raise ValueError(f"FORMULAS_JSON_INVALID: {p} does not carry schema {FORMULAS_SCHEMA!r}")
    ct = doc.get("bfv", {}).get("ct_bytes") if isinstance(doc.get("bfv"), dict) else None
    if isinstance(ct, bool) or not isinstance(ct, int) or ct <= 0:
        raise ValueError(f"FORMULAS_JSON_INVALID: {p} has no positive integer bfv.ct_bytes")
    if not isinstance(doc.get("shapes"), list):
        raise ValueError(f"FORMULAS_JSON_INVALID: {p} has no shapes list")
    if path is None:
        _FORMULAS_CACHE = doc
    return doc


def bfv_ct_bytes() -> int:
    """Bytes charged per BFV ciphertext: the measured value of formulas.json, else :data:`BFV_CT_BYTES`.

    Both cost models load the figure from the shared file; the constant of
    this module is the same measured value and is used when the results tree
    is not present.  An unreadable file fails closed.
    """
    global _BFV_CT_CACHE
    if _BFV_CT_CACHE is None:
        try:
            _BFV_CT_CACHE = int(load_formulas()["bfv"]["ct_bytes"])
        except FileNotFoundError:
            _BFV_CT_CACHE = int(BFV_CT_BYTES)
    return _BFV_CT_CACHE


def reset_formulas_cache() -> None:
    """Forget the cached formulas.json (call after the file was rewritten)."""
    global _FORMULAS_CACHE, _BFV_CT_CACHE
    _FORMULAS_CACHE = None
    _BFV_CT_CACHE = None


def measure_bfv_ct_bytes(samples: int = 32, seed: int = 0) -> dict:
    """Serialize ``samples`` fresh BFV ciphertexts with TenSEAL and return the size statistics.

    Parameters: N = 8192, plaintext modulus t = 68720050177, TenSEAL default
    coefficient modulus, 8192 uniformly random slot values per ciphertext
    (the values are public test data).  The serialized size is not constant
    (the SEAL serializer compresses), so mean, min, max and standard deviation
    are reported.  Raises ``RuntimeError`` (``HE_BACKEND``) without TenSEAL.
    """
    samples = int(samples)
    if samples < 1:
        raise ValueError(f"SHAPE_PARAM: samples must be >= 1, got {samples}")
    try:
        import tenseal as ts
    except Exception as exc:
        raise RuntimeError(f"HE_BACKEND: tenseal is not importable ({exc.__class__.__name__})") from exc
    ctx = ts.context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=BFV_N, plain_modulus=BFV_T)
    rng = np.random.default_rng(int(seed))
    sizes = []
    for _ in range(samples):
        vals = [int(v) for v in rng.integers(0, BFV_T, size=BFV_N)]
        sizes.append(len(ts.bfv_vector(ctx, vals).serialize()))
    arr = np.asarray(sizes, dtype=np.float64)
    return {"samples": samples, "mean": float(arr.mean()), "min": int(arr.min()), "max": int(arr.max()),
            "std": float(arr.std()), "tenseal_version": str(getattr(ts, "__version__", "unknown")),
            "N": BFV_N, "t": BFV_T, "slots": BFV_N}


def _shape_record(component: str, n: int, d_in: int, d_out: int, P: ProtocolParams, ct_bytes: int) -> dict:
    g = gate_counts(component, n, d_in, d_out, P)
    payload = _payload_of(g, ct_bytes if g.bfv_ct else 0)
    mat = _material_of(g, P)
    return {"component": component, "n": int(n), "d_in": int(d_in), "d_out": int(d_out),
            "macs": int(macs(component, n, d_in, d_out)), "payload_bytes": int(payload),
            "messages": int(g.messages), "bytes": int(payload + FRAME_BYTES * g.messages), "rounds": int(g.rounds),
            "material_bytes": int(sum(mat.values())), "material": mat, "gates": g.to_dict()}


def formulas_document(measured: Optional[Mapping[str, Any]] = None) -> dict:
    """The formulas.json document for the default contract (pure function of this module and ``measured``)."""
    P = ProtocolParams()
    ct = int(BFV_CT_BYTES)
    prov: Dict[str, Any] = {
        "library": "TenSEAL (Microsoft SEAL BFV), batched bfv_vector",
        "method": "len(tenseal.bfv_vector(context, 8192 values below t).serialize())",
        "parameters": "poly_modulus_degree 8192, plain_modulus 68720050177, TenSEAL default coefficient modulus",
        "declared_value": ct,
        "declared_source": "one serialization measured on 2026-10-10 with tenseal 0.3.17 "
                           "(CROSS_LANGUAGE_DECISIONS item 6); reference re-measurement over 400 ciphertexts: "
                           "mean 432453, min 432278, max 432653, std 64",
        "size_is_constant": False,
        "note": "the SEAL serializer compresses the ciphertext, so the size varies with the ciphertext bytes "
                "(about 0.05 percent); the formulas charge the declared measured value",
    }
    if measured is not None:
        prov["remeasured"] = {k: measured[k] for k in ("samples", "mean", "min", "max", "std", "tenseal_version")
                              if k in measured}
        prov["confirmed_this_run"] = True
    else:
        prov["confirmed_this_run"] = False
    try:
        from .contract import DEFAULT_CONTRACT
        chash, cname = DEFAULT_CONTRACT.contract_hash(), str(DEFAULT_CONTRACT.name)
    except Exception:  # pragma: no cover
        chash, cname = "", "NC-v2-f16-l40"
    tournament = []
    for V in (1, 2, 3, 4, 5, 8, 9, 16, 17, 24, 32, 48, 64, 65, 70, 1000, 50257):
        for k in TOURNAMENT_K:
            for with_index in (False, True):
                levels, p1, p2, bv = tournament_gates(V, k, with_index)
                tournament.append({"n": V, "k": k, "with_index": with_index, "levels": levels, "rounds": 2 * levels,
                                   "drelu_p1": p1, "drelu_p2": p2, "beaver_select": bv,
                                   "payload_bytes": OPEN_BYTES * (p1 + p2) + SELECT_BYTES * bv})
    return {
        "schema": FORMULAS_SCHEMA,
        "generator": "opencrab2.highlevel.costmodel.write_formulas_json",
        "contract": {"name": cname, "hash": chash, "params": asdict(P),
                     "derived": {"recip_live": P.recip_live, "rsqrt_live": P.rsqrt_live,
                                 "exp_products": P.exp_products, "exp_rounds": P.exp_rounds,
                                 "recip_products": P.recip_products, "recip_rounds": P.recip_rounds,
                                 "rsqrt_products": P.rsqrt_products, "rsqrt_rounds": P.rsqrt_rounds,
                                 "gelu_products": P.gelu_products, "gelu_rounds": P.gelu_rounds}},
        "conventions": {
            "shape": "linear, embedding, matmul: X (n x d_in) W (d_in x d_out); install: one d_in x d_out matrix; "
                     "gelu, trunc_faithful, ring_to_field, a2h_exact: n * d_in elements; softmax, layernorm, "
                     "argmax: n rows of length d_in",
            "payload_bytes": "raw opened bytes over both directions: 16 per opened ring element (DReLU, faithful "
                             "truncation, ring-to-field and linear openings), 32 per Beaver element, 8 per "
                             "one-way ring element, ct_bytes per BFV ciphertext",
            "bytes": "payload_bytes + frame_bytes * messages",
            "material": "per-party bytes by material type; drelu = drelu_p1 * drelu_p1_bytes + drelu_p2 * "
                        "drelu_p2_bytes; kary_select and beaver 24 per element; matrix_triple and wmask_mask 8 "
                        "per element",
            "tournament": "fused k-ary tournament of party.Party._tournament: per node of g members g(g-1)/2 "
                          "DReLUs without payload; a node of 2 members one Beaver select per output value; a "
                          "node of g >= 3 members g-1 payload DReLUs per output value; the index is an output "
                          "value on every level after the first; 2 rounds per level",
            "argmax": "argmax_k* is the tournament with index, release excluded (RELEASE_FINAL adds one "
                      "message of 8 bytes per row and one round)",
            "mask": "formulas price the unmasked row; a public mask only removes pairwise comparisons",
        },
        "framing": {"frame_bytes": int(FRAME_BYTES), "dim_bytes": FRAME_DIM_BYTES,
                    "rule": "every message is one frame: an 8-byte length prefix plus a typed-value header; "
                            "frame_bytes is the framing of a frame carrying one 1-D uint64 array under the "
                            "binding wire codec of API.md (8 prefix + tag + dtype + ndim + one u64 dimension); "
                            "every further array dimension adds dim_bytes; payload is the raw little-endian "
                            "element data"},
        "constants": {"OPEN_BYTES": OPEN_BYTES, "BEAVER_OPEN_BYTES": BEAVER_OPEN_BYTES, "SELECT_BYTES": SELECT_BYTES,
                      "ONEWAY_BYTES": ONEWAY_BYTES, "BEAVER_BYTES": BEAVER_BYTES,
                      "drelu_p1_bytes": drelu_material_bytes(P.dcf_domain_bits, 1),
                      "drelu_p2_bytes": drelu_material_bytes(P.dcf_domain_bits, 2),
                      "trunc_faithful_bytes": trunc_material_bytes(P.frac_bits),
                      "ring_to_field_bytes": r2f_material_bytes(), "matrix_triple_elem_bytes": 8,
                      "wmask_elem_bytes": 8, "BFV_N": BFV_N},
        "bfv": {"N": BFV_N, "t": BFV_T, "slots_per_ciphertext": BFV_N, "ct_bytes": ct, "status": "MEASURED_NATIVE",
                "relative_tolerance": BFV_CT_REL_TOL, "provenance": prov},
        "tournament": tournament,
        "shapes": [_shape_record(c, n, d_in, d_out, P, ct) for c, n, d_in, d_out in formula_shapes()],
    }


def write_formulas_json(path: Any = None, measure: bool = True, samples: int = 32) -> dict:
    """Write the shared formula file and return the document.

    With ``measure=True`` and TenSEAL importable the BFV ciphertext size is
    re-measured (:func:`measure_bfv_ct_bytes`); the call fails closed with
    ``BFV_CT_BYTES_UNCONFIRMED`` when the re-measured mean deviates from
    :data:`BFV_CT_BYTES` by more than :data:`BFV_CT_REL_TOL`.  Without
    TenSEAL (or with ``measure=False``) the declared measured value is
    written and the provenance says that this run did not confirm it.  The
    file is replaced atomically.
    """
    measured: Optional[dict] = None
    if measure:
        try:
            measured = measure_bfv_ct_bytes(samples)
        except RuntimeError:
            measured = None
        if measured is not None and abs(measured["mean"] - BFV_CT_BYTES) > BFV_CT_REL_TOL * BFV_CT_BYTES:
            raise ValueError(f"BFV_CT_BYTES_UNCONFIRMED: re-measured mean {measured['mean']:.1f} B deviates from the "
                             f"declared {BFV_CT_BYTES} B by more than {BFV_CT_REL_TOL:g} relative")
    doc = formulas_document(measured)
    p = Path(path) if path is not None else formulas_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
        fh.write("\n")
    last: Optional[BaseException] = None
    for _ in range(40):
        try:
            os.replace(tmp, p)
            last = None
            break
        except PermissionError as exc:  # a concurrent reader holds the file open (Windows)
            last = exc
            _time.sleep(0.05)
    if last is not None:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise last
    if p.resolve() == formulas_path().resolve():
        reset_formulas_cache()
    return doc


# ---------------------------------------------------------------------------
# fitted model
# ---------------------------------------------------------------------------
@dataclass
class ComponentFit:
    compute_per_mac_s: float = 0.0
    compute_per_byte_s: float = 0.0
    compute_fixed_s: float = 0.0
    sigma_compute_s: float = 0.0
    n_train: int = 0
    status: str = "MEASURED_NATIVE"       # MEASURED_NATIVE | HISTORICAL | MODELED
    source: str = ""

    def to_dict(self) -> dict:
        return {"compute_per_mac_s": self.compute_per_mac_s, "compute_per_byte_s": self.compute_per_byte_s,
                "compute_fixed_s": self.compute_fixed_s, "sigma_compute_s": self.sigma_compute_s,
                "n_train": self.n_train, "status": self.status, "source": self.source}

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "ComponentFit":
        return ComponentFit(float(d.get("compute_per_mac_s", 0.0)), float(d.get("compute_per_byte_s", 0.0)),
                            float(d.get("compute_fixed_s", 0.0)), float(d.get("sigma_compute_s", 0.0)),
                            int(d.get("n_train", 0)), str(d.get("status", "MEASURED_NATIVE")),
                            str(d.get("source", "")))

    def compute_s(self, n_macs: float, nbytes: float) -> float:
        return self.compute_per_mac_s * float(n_macs) + self.compute_per_byte_s * float(nbytes) + self.compute_fixed_s


def _nnls(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-negative least squares (scipy when available, projected LS otherwise)."""
    try:
        from scipy.optimize import nnls
        coef, _ = nnls(X, y)
        return np.asarray(coef, dtype=np.float64)
    except Exception:  # pragma: no cover
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        return np.maximum(np.asarray(coef, dtype=np.float64), 0.0)


@dataclass
class FittedCostModel:
    """Per-component affine compute fit plus the deterministic transport model."""
    fits: Dict[str, ComponentFit] = field(default_factory=dict)
    rel_bw: float = 0.10
    rel_rtt: float = 0.10
    label: str = "MEASURED_NATIVE"

    # -- fitting ----------------------------------------------------------
    @staticmethod
    def fit(rows: Iterable[Mapping[str, Any]], rel_bw: float = 0.10, rel_rtt: float = 0.10,
            status: str = "MEASURED_NATIVE", source: str = "") -> "FittedCostModel":
        """Fit ``compute = a*MACs + k*bytes + b`` per component by NNLS.

        ``rows``: dicts with component, n, d_in, d_out, bandwidth_bps,
        one_way_latency_s, measured_s, and optionally bytes, rounds (channel
        counters; the formulas are used when absent) and cpu_share.  The
        transport part is subtracted with the row's own bytes/rounds before
        fitting.  ``sigma_compute_s`` is the RMS over configurations of the
        repeat standard deviation of the compute residual; when no repeats
        exist the residual standard deviation of the fit is used.
        """
        by_comp: Dict[str, List[Mapping[str, Any]]] = {}
        for r in rows:
            by_comp.setdefault(str(r["component"]), []).append(r)
        model = FittedCostModel(rel_bw=float(rel_bw), rel_rtt=float(rel_rtt), label=status)
        for comp, rs in by_comp.items():
            _check_component(comp)
            ys: List[float] = []
            Xs: List[List[float]] = []
            groups: Dict[tuple, List[float]] = {}
            for r in rs:
                n, d_in, d_out = int(r["n"]), int(r["d_in"]), int(r.get("d_out", 1) or 1)
                env = Env(r.get("bandwidth_bps"), float(r["one_way_latency_s"]), float(r.get("cpu_share", 1.0)))
                nbytes = float(r["bytes"]) if r.get("bytes") is not None else float(bytes_formula(comp, n, d_in, d_out))
                rounds = float(r["rounds"]) if r.get("rounds") is not None else float(rounds_formula(comp, n, d_in, d_out))
                y = (float(r["measured_s"]) - transport_s(nbytes, rounds, env)) * env.cpu_share
                ys.append(y)
                Xs.append([float(macs(comp, n, d_in, d_out)), nbytes, 1.0])
                groups.setdefault((n, d_in, d_out, env.bandwidth_bps, env.one_way_latency_s, env.cpu_share), []).append(y)
            X = np.asarray(Xs, dtype=np.float64)
            yv = np.asarray(ys, dtype=np.float64)
            scale = np.maximum(X.max(axis=0), 1e-300)
            coef = _nnls(X / scale, yv) / scale
            stds = [float(np.std(v, ddof=1)) for v in groups.values() if len(v) > 1]
            if stds:
                sigma = float(np.sqrt(np.mean(np.square(stds))))
            else:
                resid = yv - X @ coef
                dof = max(1, len(yv) - 3)
                sigma = float(np.sqrt(np.sum(resid ** 2) / dof)) if len(yv) > 3 else float(np.sqrt(np.mean(resid ** 2)))
            model.fits[comp] = ComponentFit(float(coef[0]), float(coef[1]), float(coef[2]), sigma,
                                            len(rs), status, source)
        return model

    # -- prediction -------------------------------------------------------
    def has(self, component: str) -> bool:
        return component in self.fits

    def compute_s(self, component: str, n: int, d_in: int, d_out: int, nbytes: Optional[float] = None) -> float:
        f = self.fits.get(component)
        if f is None:
            raise ValueError(f"NO_FIT: component {component!r} has no fitted coefficients")
        if nbytes is None:
            nbytes = bytes_formula(component, n, d_in, d_out)
        return f.compute_s(macs(component, n, d_in, d_out), nbytes)

    def predict(self, component: str, n: int, d_in: int, d_out: int, env: Any) -> Tuple[float, float]:
        """(mean_s, sigma_s) of one call of ``component`` on (n, d_in, d_out) under ``env``."""
        _check_component(component)
        f = self.fits.get(component)
        if f is None:
            raise ValueError(f"NO_FIT: component {component!r} has no fitted coefficients")
        e = check_env(env)                      # ENV_PARAMS for cpu_share <= 0 or a non-finite field
        cpu = float(e.cpu_share)
        nbytes = bytes_formula(component, n, d_in, d_out)
        rounds = rounds_formula(component, n, d_in, d_out)
        comm = nbytes * 8.0 / float(e.bandwidth_bps) if e.bandwidth_bps else 0.0
        rt = rounds * e.rtt_s
        mean = self.compute_s(component, n, d_in, d_out, nbytes) / cpu + comm + rt
        s_c, s_b, s_r = f.sigma_compute_s / cpu, comm * self.rel_bw, rt * self.rel_rtt
        sigma = math.sqrt(s_c * s_c + s_b * s_b + s_r * s_r)     # products, not pow: bit-exact on every libm
        return float(mean), float(sigma)

    def status_of(self, component: str) -> str:
        f = self.fits.get(component)
        return f.status if f is not None else "MISSING"

    def perturbed(self, factors: Mapping[str, Mapping[str, float]]) -> "FittedCostModel":
        """Copy with compute coefficients multiplied (sensitivity studies)."""
        m = FittedCostModel(rel_bw=self.rel_bw, rel_rtt=self.rel_rtt, label=self.label)
        for comp, f in self.fits.items():
            d = f.to_dict()
            for k, v in factors.get(comp, {}).items():
                d[k] = float(d[k]) * float(v)
            m.fits[comp] = ComponentFit.from_dict(d)
        return m

    # -- persistence --------------------------------------------------------
    def to_json(self) -> dict:
        return {"rel_bw": self.rel_bw, "rel_rtt": self.rel_rtt, "label": self.label,
                "components": {k: v.to_dict() for k, v in self.fits.items()}}

    @staticmethod
    def from_json(d: Mapping[str, Any]) -> "FittedCostModel":
        m = FittedCostModel(rel_bw=float(d.get("rel_bw", 0.10)), rel_rtt=float(d.get("rel_rtt", 0.10)),
                            label=str(d.get("label", "MEASURED_NATIVE")))
        for k, v in dict(d.get("components", {})).items():
            m.fits[str(k)] = ComponentFit.from_dict(v)
        return m

    @staticmethod
    def load(path: Any) -> "FittedCostModel":
        with open(path, "r", encoding="utf-8") as fh:
            return FittedCostModel.from_json(json.load(fh))

    def save(self, path: Any) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(self.to_json(), fh, indent=1, sort_keys=True)


# ---------------------------------------------------------------------------
# hand-set baseline (MODELED), used only as the comparison point of the calibration study
# ---------------------------------------------------------------------------
HANDWRITTEN = {
    "mac_s": 1.0e-8,            # uint64 matmul throughput of about 100 M MAC/s single thread
    "elem_linear_s": 2.0e-8,    # share handling per opened element
    "dcf_level_s": 5.0e-8,      # one GGM level of one DCF evaluation
    "beaver_elem_s": 3.0e-7,    # one Beaver product per element incl. serialisation
    "bfv_ct_s": 0.02,           # one BFV encrypt + plaintext add at N = 8192
    "fixed_s": 1.0e-3,          # per call overhead
}


def handwritten_predict(component: str, n: int, d_in: int, d_out: int, env: Any) -> float:
    """Hand-set analytic estimate (status MODELED) of one call; no fitted coefficient is used.

    Linear components and installs: MAC throughput plus share handling.  Every
    other component is priced from :func:`gate_counts`: two DCF evaluations of
    n levels per DReLU (three half-evaluations more for the payload word are
    ignored), 64 + f levels per faithful truncation, 2 x 64 levels per
    ring-to-field gate, one Beaver element cost per product element and one
    BFV encryption per ciphertext.
    """
    _check_component(component)
    e = as_env(env)
    H = HANDWRITTEN
    N = elements(component, n, d_in, d_out)
    if component in LINEAR_COMPONENTS or component == "install_wmask":
        compute = H["mac_s"] * macs(component, n, d_in, d_out) + H["elem_linear_s"] * N
    else:
        g = gate_counts(component, n, d_in, d_out)
        levels = (g.drelu_p1 + g.drelu_p2) * 2 * DCF_DOMAIN_BITS + g.trunc * (RING_BITS + FRAC_BITS) + g.r2f * 2 * RING_BITS
        compute = (levels * H["dcf_level_s"] + (g.beaver + g.kary_select) * H["beaver_elem_s"]
                   + g.bfv_ct * H["bfv_ct_s"])
    cpu = float(check_env(e).cpu_share)
    return float((compute + H["fixed_s"]) / cpu
                 + transport_s(bytes_formula(component, n, d_in, d_out), rounds_formula(component, n, d_in, d_out), e))


def handwritten_model() -> FittedCostModel:
    """A FittedCostModel whose coefficients reproduce :func:`handwritten_predict` at a reference shape (MODELED)."""
    m = FittedCostModel(label="MODELED")
    env0 = Env(None, 0.0, 1.0)
    for comp in COMPONENTS:
        n, d_in, d_out = (1, 768, 768)
        if comp.startswith("argmax") or comp.startswith("embedding"):
            d_in = 50257
        if comp.startswith("softmax"):
            d_in = 32
        total = handwritten_predict(comp, n, d_in, d_out, env0) - HANDWRITTEN["fixed_s"]
        per_mac = total / max(1, macs(comp, n, d_in, d_out))
        m.fits[comp] = ComponentFit(per_mac, 0.0, HANDWRITTEN["fixed_s"], 0.25 * total, 0, "MODELED",
                                    "handwritten constants at reference shape")
    return m


def calibration_path() -> Path:
    """calibration/fitted_model.json: the rerun root's copy when ``OPENCRAB2_RESULTS_DIR`` names one that holds the
    file, else the shipped results/calibration/fitted_model.json of this tree."""
    override = os.environ.get("OPENCRAB2_RESULTS_DIR", "").strip()
    if override:
        p = Path(override).expanduser() / "calibration" / "fitted_model.json"
        if p.is_file():
            return p
    return shipped_calibration_dir() / "fitted_model.json"


_DEFAULT: Optional[FittedCostModel] = None


def default_model() -> FittedCostModel:
    """The calibrated model (:func:`calibration_path`) when present, else the MODELED hand-written baseline."""
    global _DEFAULT
    if _DEFAULT is None:
        p = calibration_path()
        _DEFAULT = FittedCostModel.load(p) if p.is_file() else handwritten_model()
    return _DEFAULT


# ---------------------------------------------------------------------------
# evaluation metrics
# ---------------------------------------------------------------------------
def _rankdata(v: np.ndarray) -> np.ndarray:
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v), dtype=np.float64)
    sv = v[order]
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2:
        return float("nan")
    ra, rb = _rankdata(a), _rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def kendall(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = len(a)
    if n < 2:
        return float("nan")
    conc = disc = 0
    for i in range(n):
        s = np.sign(a[i] - a[i + 1:]) * np.sign(b[i] - b[i + 1:])
        conc += int(np.sum(s > 0))
        disc += int(np.sum(s < 0))
    return float((conc - disc) / (n * (n - 1) / 2))


def error_metrics(measured: Sequence[float], predicted: Sequence[float]) -> dict:
    """mape, median and p95 relative error, Spearman and Kendall rank correlation, n."""
    m = np.asarray(measured, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    if m.shape != p.shape:
        raise ValueError(f"METRIC_SHAPE: measured {m.shape} and predicted {p.shape} differ")
    rel = np.abs(p - m) / np.maximum(np.abs(m), 1e-12)
    return {"mape": float(np.mean(rel)) if len(rel) else float("nan"),
            "median_rel_err": float(np.median(rel)) if len(rel) else float("nan"),
            "p95_rel_err": float(np.percentile(rel, 95)) if len(rel) else float("nan"),
            "spearman": spearman(m, p), "kendall": kendall(m, p), "n": int(len(m))}


__all__ = [
    "COMPONENTS", "LINEAR_COMPONENTS", "ELEMENTWISE_COMPONENTS", "ROW_COMPONENTS", "KARY_COMPONENTS", "TOURNAMENT_K",
    "FRAME_BYTES", "FRAME_DIM_BYTES", "OPEN_BYTES", "BEAVER_OPEN_BYTES", "SELECT_BYTES", "ONEWAY_BYTES",
    "BEAVER_BYTES", "BFV_N", "BFV_T", "BFV_CT_BYTES", "BFV_CT_REL_TOL",
    "Env", "as_env", "ProtocolParams", "protocol_params", "kary_tournament", "tournament_gates",
    "tournament_node_sizes", "GateCounts", "gate_counts", "elements", "macs", "rounds_formula", "messages_formula",
    "payload_bytes", "bytes_formula", "dcf_key_bytes", "drelu_material_bytes", "trunc_material_bytes",
    "r2f_material_bytes", "material_breakdown", "material_formula", "transport_s",
    "FORMULAS_SCHEMA", "FORMULA_TEST_SHAPES", "formula_shapes", "formulas_path", "load_formulas", "bfv_ct_bytes",
    "reset_formulas_cache", "measure_bfv_ct_bytes", "formulas_document", "write_formulas_json",
    "ComponentFit", "FittedCostModel", "handwritten_predict", "handwritten_model", "HANDWRITTEN",
    "shipped_calibration_dir", "calibration_path", "default_model",
    "spearman", "kendall", "error_metrics",
]
