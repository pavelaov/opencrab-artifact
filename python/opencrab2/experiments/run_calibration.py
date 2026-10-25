"""Shaped two-process calibration of the per-component cost model (paper, Section 7.3, accuracy of the cost model).

Measures every component of ``costmodel.COMPONENTS`` on a pre-declared shape
grid under the four network anchors (LAN, WAN, WAN-c, WAN-mid) with two
party processes and an isolated dealer process, fits the affine compute
model by NNLS on the TRAIN split, evaluates it on the pre-declared HOLD-OUT,
and writes ``calibration/fitted_model.json`` under the data root plus raw records.

Pre-declared hold-out (fixed here before any measurement): a configuration
is TEST when its shape is 1024x1024 or
3072x768, or its network is WAN-mid, or its batch is n = 32; everything else
is TRAIN. Crossover-near cases (k-ary argmax under several RTTs, generic
versus masked linear lanes at several bandwidths) are inside the grid.

Process layout. The parent starts ``opencrab2.platform.dealer_service`` in
``ISOLATED_RECIPIENT_BUNDLES`` mode with fresh per-recipient HMAC keys, then,
per network, two worker processes (this module with ``--worker``). P0 listens
on loopback, P1 dials; both wrap the socket in a ``Channel`` with a
``WallClockShaper`` built from the network anchor (identical origin). Material
comes only from the dealer service through ``DealerServiceClient`` into a
ledger-backed ``DealerClient``; a thin wrapper records the exact material a
warm-up call consumed and provisions precisely that amount before every
timed repeat, so the timed interval contains no dealer traffic (a residual
stall is counted, timed and subtracted, and the row is flagged). Each timed
call is preceded by a one-message barrier. Per call the row keeps both
parties' wall seconds (``measured_s`` is their maximum), the exact channel
byte and round counters and the material bytes consumed.

Realization notes. ``linear_generic`` and ``embedding_generic`` are measured
as ``Party.matmul_shared`` on a pre-shared weight share (the online protocol
of ``Party.linear_generic`` is exactly this matrix-triple product; the bias
add is local), because a calibration has no model owner whose private weight
vector the dealer would share. ``a2h_exact`` is measured only when TenSEAL is
importable; otherwise it keeps its MODELED prior in the written model.

Metrics (hold-out and train): MAPE, median and p95 relative error, Spearman
and Kendall rank correlation, interval coverage at two sigma, lane top-1
accuracy and realized regret per lane group (linear wmask/generic, embedding
wmask/generic, argmax k2/k4/k8, softmax k2/k4), the informative-pair
diagnostic (accuracy restricted to lane groups whose measured winner changes
somewhere in the split, with its decision count as denominator, next to the
dominated groups), threshold-near decisions (relative gap below 10 %), and
full-plan identity per (n, network). The comparison baseline is the hand-set
MODELED model.

Usage::

    python -m opencrab2.experiments.run_calibration --quick            # small grid, under 10 minutes
    python -m opencrab2.experiments.run_calibration [--networks LAN,WAN] [--repeats 3] [--components a,b]
    python -m opencrab2.experiments.run_calibration --fit-only          # refit from raw/calibration.jsonl

Outputs under the data root: calibration/fitted_model.json,
calibration_metrics.json, grid.json; raw/calibration.jsonl (one
MEASURED_NATIVE record per timed call); processed/calibration_rows.csv.
``--fit-only`` refits from the raw records (``raw/calibration.jsonl`` or its
``.gz`` form) and rewrites the model and the metrics; it keeps the
``campaign`` block of an existing calibration_metrics.json (campaign id,
host, the hash of the formulas file the campaign priced with), because a
refit measures nothing new and the tables read that block to tell a current
calibration from a stale one.

What is tested (tests/test_run_calibration.py): ``--fit-only`` on a copy of
the shipped records keeps the campaign block and reproduces the fitted
components; the quick campaign is run manually (its runtime is recorded in
calibration_metrics.json); the pure functions (``build_grid``, ``split_of``,
``fit_rows``, ``evaluate_models``) are deterministic given the rows. Design
limits: loopback shaping is an application-level link emulation, timings are
single-thread CPython on one host, and the fitted model is affine per
component (no cross terms).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import secrets
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from opencrab2.errors import MaterialError, MaterialStall, OpenCrabError
from opencrab2.experiments import schema
from opencrab2.experiments.traces import ANCHORS
from opencrab2.highlevel import costmodel as cm
from opencrab2.highlevel.channel import WallClockShaper, accept_channel, dial_channel, open_listener
from opencrab2.highlevel.contract import DEFAULT_CONTRACT, NumericContract
from opencrab2.highlevel.dealer import DealerClient, MODE_ISOLATED, PROFILE_ID
from opencrab2.highlevel.ledger import MaterialLedger
from opencrab2.lowlevel import ring

PYTHON = sys.executable
NETWORKS: Tuple[str, ...] = ("LAN", "WAN", "WAN-c", "WAN-mid")
HOLDOUT_SHAPES = {(1024, 1024), (3072, 768)}
HOLDOUT_NETWORK = "WAN-mid"
HOLDOUT_N = 32
T_FIELD = 68720050177
DATA_SEED = 4242
D_MODEL = 768
VOCAB = 50257
NEAR_THRESHOLD_REL = 0.10
COVERAGE_Z = 2.0

LINEAR_SHAPES: Tuple[Tuple[int, int], ...] = ((256, 256), (512, 512), (768, 768), (768, 2304), (768, 3072),
                                              (3072, 768), (1024, 1024))
LINEAR_NS: Tuple[int, ...] = (1, 8, 32)
EMBED_VOCABS: Tuple[int, ...] = (4096, 16384, VOCAB)
ELEM_DS: Tuple[int, ...] = (768, 3072)
SOFTMAX_ROWS: Tuple[int, ...] = (12, 96, 384)
SOFTMAX_KEYS: Tuple[int, ...] = (8, 32, 128)
ARGMAX_VOCABS: Tuple[int, ...] = (4096, 16384, VOCAB)

QUICK_LINEAR = (((256, 256), (1, 8, 32)), ((768, 768), (1, 8)), ((1024, 1024), (1, 8)))
QUICK_EMBED_VOCABS = (4096,)
QUICK_EMBED_NS = (1, 8)
QUICK_ELEM = ((256, (1, 8)), (768, (1, 8)))
QUICK_SOFTMAX = ((12, (8, 32)), (96, (8, 32)))
QUICK_LN_NS = (1, 8)
QUICK_ARGMAX_VOCABS = (4096, 16384)
QUICK_NETWORKS = ("LAN", "WAN-mid")
QUICK_REPEATS = 2
DEFAULT_REPEATS = 3

LANE_GROUPS: Dict[str, Tuple[str, ...]] = {
    "linear": ("linear_wmask", "linear_generic"),
    "embedding": ("embedding_wmask", "embedding_generic"),
    "argmax": ("argmax_k2", "argmax_k4", "argmax_k8"),
    "softmax": ("softmax_rows_k2", "softmax_rows_k4"),
}


# ---------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    component: str
    n: int
    d_in: int
    d_out: int

    @property
    def cfg_id(self) -> str:
        return f"{self.component}:{self.n}x{self.d_in}x{self.d_out}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cfg_id"] = self.cfg_id
        return d


def split_of(component: str, n: int, d_in: int, d_out: int, network: str) -> str:
    """Pre-declared hold-out rule (shape, network or batch extrapolation)."""
    if (int(d_in), int(d_out)) in HOLDOUT_SHAPES or network == HOLDOUT_NETWORK or int(n) == HOLDOUT_N:
        return "test"
    return "train"


def build_grid(quick: bool = False, components: Optional[Sequence[str]] = None) -> List[Config]:
    """The declared configuration grid (component, n, d_in, d_out), independent of the network."""
    out: List[Config] = []
    seen = set()

    def add(comp: str, n: int, d_in: int, d_out: int) -> None:
        c = Config(comp, int(n), int(d_in), int(d_out))
        if c not in seen:
            seen.add(c)
            out.append(c)

    if quick:
        for (d, m), ns in QUICK_LINEAR:
            add("install_wmask", 1, d, m)
            for n in ns:
                for comp in ("linear_wmask", "linear_generic", "matmul_shared"):
                    add(comp, n, d, m)
        for V in QUICK_EMBED_VOCABS:
            add("install_wmask", 1, V, D_MODEL)
            for n in QUICK_EMBED_NS:
                add("embedding_wmask", n, V, D_MODEL)
                add("embedding_generic", n, V, D_MODEL)
        for d, ns in QUICK_ELEM:
            for n in ns:
                for comp in cm.ELEMENTWISE_COMPONENTS:
                    add(comp, n, d, d)
        for rows, keys in QUICK_SOFTMAX:
            for k in keys:
                add("softmax_rows_k2", rows, k, k)
                add("softmax_rows_k4", rows, k, k)
        for n in QUICK_LN_NS:
            add("layernorm_rows", n, D_MODEL, D_MODEL)
        for V in QUICK_ARGMAX_VOCABS:
            for comp in ("argmax_k2", "argmax_k4", "argmax_k8"):
                add(comp, 1, V, 1)
    else:
        for d, m in LINEAR_SHAPES:
            add("install_wmask", 1, d, m)
            for n in LINEAR_NS:
                for comp in ("linear_wmask", "linear_generic", "matmul_shared"):
                    add(comp, n, d, m)
        for V in EMBED_VOCABS:
            add("install_wmask", 1, V, D_MODEL)
            for n in LINEAR_NS:
                add("embedding_wmask", n, V, D_MODEL)
                if V < VOCAB or n <= 8:        # the W - B opening dominates and is independent of n
                    add("embedding_generic", n, V, D_MODEL)
        for d in ELEM_DS:
            for n in LINEAR_NS:
                for comp in cm.ELEMENTWISE_COMPONENTS:
                    add(comp, n, d, d)
        for rows in SOFTMAX_ROWS:
            for k in SOFTMAX_KEYS:
                add("softmax_rows_k2", rows, k, k)
                add("softmax_rows_k4", rows, k, k)
        for n in LINEAR_NS:
            add("layernorm_rows", n, D_MODEL, D_MODEL)
        for V in ARGMAX_VOCABS:
            for comp in ("argmax_k2", "argmax_k4", "argmax_k8"):
                add(comp, 1, V, 1)
    if components:
        keep = set(components)
        out = [c for c in out if c.component in keep]
    return out


# ---------------------------------------------------------------------------
# worker side: service-backed material client
# ---------------------------------------------------------------------------
STALL_CHUNK: Dict[str, int] = {"drelu": 4096, "trunc_faithful": 1024, "ring_to_field": 1024,
                               "beaver": 1 << 16, "kary_select": 1 << 16}
WARMUP_FACTOR_DEFAULT = 2.0
WARMUP_FACTOR_KARY_DRELU = 8.0     # the k-ary realization spends more DReLU than the planner estimate (k >= 4)


def aggregate_takes(takes: Sequence[Tuple[str, dict, int]]) -> List[Tuple[str, dict, int]]:
    """Merge the takes of one call into one bundle request per (material type, dealer parameters).

    Shape-free types (Beaver-like, DReLU, faithful truncation, ring-to-field)
    are summed into one request each; matrix triples, wmask calls and masks
    are objects and stay one request per take. The DealerClient slices every
    take from the merged bundle, so a bundle id is spent once per call (one
    durable ledger write) instead of once per node of a tournament.
    """
    merged: Dict[tuple, List[Any]] = {}
    order: List[tuple] = []
    out: List[Tuple[str, dict, int]] = []
    for mt, params, count in takes:
        if mt in ("beaver", "kary_select"):
            key = (mt,)
            if key not in merged:
                merged[key] = [mt, {"shape": [0]}, 1]
                order.append(key)
            merged[key][1]["shape"][0] += int(params["shape"][0])
        elif mt in ("drelu", "trunc_faithful", "ring_to_field"):
            p = {k: v for k, v in params.items() if k != "M"}
            key = (mt,) + tuple(sorted(p.items()))
            if key not in merged:
                merged[key] = [mt, dict(p, M=0), 1]
                order.append(key)
            merged[key][1]["M"] += int(params["M"])
        else:
            out.append((mt, dict(params), int(count)))
    return [(merged[k][0], merged[k][1], merged[k][2]) for k in order] + out


class ServiceBackedClient:
    """DealerClient facade that provisions from the dealer service on a stall and records every take.

    Both parties run identical gadget code, so their take sequences and
    request ids coincide; the dealer service pairs the two requests and
    streams each party only its own half. ``record`` collects
    ``(material_type, params, count)`` of the takes of one call so that the
    same volume can be provisioned ahead of the timed repeats
    (:func:`aggregate_takes`). A stall provisions a generous chunk
    (``STALL_CHUNK``) so that a warm-up never issues thousands of tiny requests.
    """

    def __init__(self, client: DealerClient, service: Any, mask_shapes: Dict[str, Tuple[int, int]]):
        self.client = client
        self.svc = service
        self.party = client.party
        self.ledger = client.ledger
        self.contract = client.contract
        self.mask_shapes = mask_shapes
        self.scope = "init"
        self._req = 0
        self.stalls = 0
        self.provision_s = 0.0
        self.provisioned_bytes = 0
        self.record: Optional[List[Tuple[str, dict, int]]] = None

    def set_scope(self, scope: str) -> None:
        self.scope = str(scope)
        self._req = 0

    def _rid(self, mt: str) -> str:
        self._req += 1
        return f"{self.scope}:{mt}:{self._req}"

    def provision(self, mt: str, params: dict, count: int = 1) -> None:
        t0 = time.perf_counter()
        before = self.client.ingested_bytes
        self.svc.provision_into(self.client, self._rid(mt), mt, params, count, gate_id=self.scope)
        self.provisioned_bytes += self.client.ingested_bytes - before
        self.provision_s += time.perf_counter() - t0

    def provision_many(self, takes: Sequence[Tuple[str, dict, int]]) -> None:
        for mt, params, count in aggregate_takes(takes):
            if mt == "wmask_mask":
                continue
            self.provision(mt, params, count)

    def _retry(self, fn: Callable[[], Any], mt: str, params: dict, count: int = 1) -> Any:
        if self.record is not None:
            self.record.append((mt, dict(params), int(count)))
        try:
            return fn()
        except MaterialStall:
            self.stalls += 1
            p = dict(params)
            if mt in ("beaver", "kary_select"):
                p["shape"] = [max(int(p["shape"][0]), STALL_CHUNK[mt])]
            elif mt in STALL_CHUNK:
                p["M"] = max(int(p["M"]), STALL_CHUNK[mt])
            self.provision(mt, p, count)
            return fn()

    @staticmethod
    def _numel(shape: Any) -> int:
        shp = tuple(int(s) for s in (shape if isinstance(shape, (tuple, list)) else (shape,)))
        n = 1
        for s in shp:
            n *= s
        return int(n)

    def take_beaver(self, shape):
        n = self._numel(shape)
        return self._retry(lambda: self.client.take_beaver(shape), "beaver", {"shape": [n]})

    def take_kary_select(self, shape):
        n = self._numel(shape)
        return self._retry(lambda: self.client.take_kary_select(shape), "kary_select", {"shape": [n]})

    def take_matrix_triple(self, shape_a, shape_b):
        sa, sb = [int(s) for s in shape_a], [int(s) for s in shape_b]
        return self._retry(lambda: self.client.take_matrix_triple(tuple(sa), tuple(sb)), "matrix_triple",
                           {"shape_a": sa, "shape_b": sb})

    def take_wmask_call(self, mat_id, n_rows):
        return self._retry(lambda: self.client.take_wmask_call(mat_id, int(n_rows)), "wmask_call",
                           {"mat_id": str(mat_id), "n_rows": int(n_rows)})

    def wmask_mask_share(self, mat_id, for_version=None):
        if mat_id not in self.mask_shapes:
            raise MaterialError(f"CALIBRATION_MASK_SHAPE: unknown wmask shape for {mat_id!r}")
        shp = [int(s) for s in self.mask_shapes[mat_id]]
        return self._retry(lambda: self.client.wmask_mask_share(mat_id, for_version=for_version), "wmask_mask",
                           {"mat_id": str(mat_id), "shape": shp})

    def wmask_mask_version(self, mat_id):
        return self.client.wmask_mask_version(mat_id)

    def take_drelu(self, count, with_product=True, n_bits=None):
        nb = int(self.contract.dcf_domain_bits if n_bits is None else n_bits)
        return self._retry(lambda: self.client.take_drelu(int(count), with_product=with_product, n_bits=n_bits),
                           "drelu", {"M": int(count), "n_bits": nb, "with_product": bool(with_product)})

    def take_trunc(self, count, f=None, L=None):
        ff = int(self.contract.frac_bits if f is None else f)
        LL = int(self.contract.logical_bits if L is None else L)
        return self._retry(lambda: self.client.take_trunc(int(count), f, L), "trunc_faithful",
                           {"M": int(count), "f": ff, "L": LL})

    def take_r2f(self, count, t):
        return self._retry(lambda: self.client.take_r2f(int(count), int(t)), "ring_to_field",
                           {"M": int(count), "t": int(t)})

    def take_vector_share(self, name):
        raise MaterialError(f"MATERIAL_MISSING_VECTOR: calibration serves no private vector ({name!r})")

    def stock(self):
        return self.client.stock()

    def set_context(self, gate_id: str, epoch: int) -> None:
        self.client.set_context(gate_id, epoch)


# ---------------------------------------------------------------------------
# worker side: inputs and gadget calls
# ---------------------------------------------------------------------------
def _enc(x: np.ndarray, contract: NumericContract) -> np.ndarray:
    return ring.encode(np.asarray(x, np.float64), contract)


def _shares(value: np.ndarray, party: int, seed: int) -> np.ndarray:
    """Deterministic additive sharing of a public calibration input (P1 holds r, P0 holds value - r)."""
    r = ring.rand_u64(np.shape(value), np.random.default_rng(seed))
    return r if party == 1 else ring.sub(np.asarray(value, np.uint64), r)


def inputs_for(cfg: Config, party: int, contract: NumericContract, seed: int = DATA_SEED) -> Dict[str, np.ndarray]:
    """Shares of public calibration inputs (data-oblivious protocols; values stay inside the contract)."""
    h = int.from_bytes(hashlib.sha256(f"{seed}|{cfg.cfg_id}".encode()).digest()[:4], "big")
    rng = np.random.default_rng(h)
    comp, n, d_in, d_out = cfg.component, cfg.n, cfg.d_in, cfg.d_out
    out: Dict[str, np.ndarray] = {}
    if comp in ("linear_wmask", "linear_generic", "matmul_shared"):
        X = _enc(rng.uniform(-1.0, 1.0, size=(n, d_in)), contract)
        W = _enc(rng.uniform(-0.05, 0.05, size=(d_in, d_out)), contract)
        out["X"] = _shares(X, party, h + 1)
        out["W"] = _shares(W, party, h + 2)
        out["W_enc"] = W
    elif comp in ("embedding_wmask", "embedding_generic"):
        ids = rng.integers(0, d_in, size=n)
        X = np.zeros((n, d_in), np.uint64)
        X[np.arange(n), ids] = np.uint64(1)                       # one-hot in ring units (no scale)
        W = _enc(rng.uniform(-0.1, 0.1, size=(d_in, d_out)), contract)
        out["X"] = X if party == 1 else np.zeros_like(X)         # the client's one-hot is its own share
        out["W"] = _shares(W, party, h + 2)
        out["W_enc"] = W
    elif comp == "install_wmask":
        out["W_enc"] = _enc(rng.uniform(-0.1, 0.1, size=(d_in, d_out)), contract)
    elif comp == "gelu_clip12":
        out["X"] = _shares(_enc(rng.uniform(-6.0, 6.0, size=(n, d_in)), contract), party, h + 1)
    elif comp == "trunc_faithful":
        out["X"] = _shares(_enc(rng.uniform(-1000.0, 1000.0, size=(n, d_in)), contract), party, h + 1)
    elif comp in ("ring_to_field", "a2h_exact"):
        out["X"] = _shares(_enc(rng.uniform(-1000.0, 1000.0, size=(n, d_in)), contract), party, h + 1)
    elif comp.startswith("softmax_rows_k"):
        out["X"] = _shares(_enc(rng.uniform(-4.0, 4.0, size=(n, d_in)), contract), party, h + 1)
    elif comp == "layernorm_rows":
        out["X"] = _shares(_enc(rng.normal(0.0, 1.0, size=(n, d_in)), contract), party, h + 1)
        # the gain is a shared server vector in the transformer (18 rounds, as the formula prices it)
        out["G"] = _shares(_enc(rng.uniform(0.8, 1.2, size=(d_in,)), contract), party, h + 3)
    elif comp.startswith("argmax_k"):
        out["X"] = _shares(_enc(rng.uniform(-40.0, 40.0, size=(d_in,)), contract), party, h + 1)
    else:
        raise ValueError(f"CALIBRATION_COMPONENT: no input generator for {comp!r}")
    return out


def mat_id_for(cfg: Config, repeat: Optional[int] = None) -> str:
    base = f"cal:{cfg.d_in}x{cfg.d_out}"
    return base if repeat is None else f"{base}:install{repeat}"


def estimated_takes(cfg: Config, contract: NumericContract, mat_id: str) -> List[Tuple[str, dict, int]]:
    """Planner-side material estimate of one call, scaled by the warm-up factors (surplus is harmless)."""
    from opencrab2.highlevel.dealer import material_estimate_detail
    comp, n, d_in, d_out = cfg.component, cfg.n, cfg.d_in, cfg.d_out
    if comp in ("linear_wmask", "embedding_wmask"):
        return [("wmask_call", {"mat_id": mat_id, "n_rows": n}, 1)]
    if comp in ("linear_generic", "matmul_shared", "embedding_generic"):
        return [("matrix_triple", {"shape_a": [n, d_in], "shape_b": [d_in, d_out]}, 1)]
    if comp == "install_wmask":
        return []
    kary = False
    if comp == "gelu_clip12":
        op, shape, kw = "gelu", (n, d_in), {}
    elif comp == "trunc_faithful":
        op, shape, kw = "trunc_faithful", (n, d_in), {}
    elif comp in ("ring_to_field", "a2h_exact"):
        op, shape, kw = "ring_to_field", (n, d_in), {"t": T_FIELD}
    elif comp.startswith("softmax_rows_k"):
        k = int(comp.rsplit("_k", 1)[1])
        op, shape, kw = "softmax_rows", (n, d_in), {"k_max": k}
        kary = k > 2
    elif comp == "layernorm_rows":
        op, shape, kw = "layernorm_rows", (n, d_in), {}
    elif comp.startswith("argmax_k"):
        k = int(comp.rsplit("_k", 1)[1])
        op, shape, kw = "argmax_k", (1, d_in), {"k": k}
        kary = k > 2
    else:
        raise ValueError(f"CALIBRATION_COMPONENT: no estimate mapping for {comp!r}")
    takes: List[Tuple[str, dict, int]] = []
    for ent in material_estimate_detail(op, shape, contract, **kw):
        mt = ent["material_type"]
        count = int(ent["count"])
        if count <= 0:
            continue
        factor = WARMUP_FACTOR_KARY_DRELU if (kary and mt == "drelu") else WARMUP_FACTOR_DEFAULT
        c = int(math.ceil(count * factor))
        params = dict(ent.get("params") or {})
        if mt in ("beaver", "kary_select"):
            takes.append((mt, {"shape": [c]}, 1))
        elif mt in ("drelu", "trunc_faithful", "ring_to_field"):
            params["M"] = c
            takes.append((mt, params, 1))
    return takes


def run_gadget(party: Any, cfg: Config, data: Dict[str, np.ndarray], mat_id: str) -> dict:
    """One call of the component; returns {seconds, bytes, rounds, material_bytes} from the party counters."""
    comp = cfg.component
    kind = comp
    if comp == "install_wmask":
        rec = party.install_wmask(mat_id, data["W_enc"] if party.party == 0 else None)
        if rec.get("skipped"):
            raise MaterialError("CALIBRATION_INSTALL_SKIPPED: install repeat hit an installed object")
        return {"seconds": float(rec["seconds"]), "bytes": int(rec["bytes"]), "rounds": int(rec["rounds"]),
                "material_bytes": int(8 * cfg.d_in * cfg.d_out)}
    n0 = len(party.stats)
    if comp == "linear_wmask":
        party.linear_wmask(data["X"], mat_id, trunc=True)
    elif comp == "embedding_wmask":
        party.linear_wmask(data["X"], mat_id, trunc=False)
        kind = "linear_wmask"
    elif comp in ("linear_generic", "matmul_shared"):
        party.matmul_shared(data["X"], data["W"], trunc=True)
        kind = "matmul_shared"
    elif comp == "embedding_generic":
        party.matmul_shared(data["X"], data["W"], trunc=False)
        kind = "matmul_shared"
    elif comp == "gelu_clip12":
        party.gelu(data["X"])
    elif comp == "trunc_faithful":
        party.trunc_faithful(data["X"])
    elif comp == "ring_to_field":
        party.ring_to_field(data["X"], T_FIELD)
    elif comp == "a2h_exact":
        from opencrab2.highlevel.he_lane import a2h_exact
        a2h_exact(party, data["X"], T_FIELD)
    elif comp.startswith("softmax_rows_k"):
        party.softmax_rows(data["X"], k_max=int(comp.rsplit("_k", 1)[1]))
    elif comp == "layernorm_rows":
        party.layernorm_rows(data["X"], gamma=data["G"])
    elif comp.startswith("argmax_k"):
        party.argmax_row(data["X"], k=int(comp.rsplit("_k", 1)[1]), release=False)
    else:
        raise ValueError(f"CALIBRATION_COMPONENT: cannot run {comp!r}")
    recs = [s for s in party.stats[n0:] if s["kind"] == kind]
    if not recs:
        raise MaterialError(f"CALIBRATION_STATS: no statistics entry of kind {kind!r} after the call")
    s = recs[-1]
    return {"seconds": float(s["seconds"]), "bytes": int(s["bytes"]), "rounds": int(s["rounds"]),
            "material_bytes": int(s["material_bytes"])}


def worker_main(party: int, config_path: str) -> int:
    """Entry point of one party process (see module docstring for the protocol)."""
    from opencrab2.highlevel.party import Party
    from opencrab2.platform.dealer_service import DealerServiceClient

    with open(config_path, "r", encoding="utf-8") as fh:
        wc = json.load(fh)
    contract = DEFAULT_CONTRACT
    network = wc["network"]
    bw, delay = float(wc["bandwidth_bps"]), float(wc["one_way_delay_s"])
    origin = float(wc["origin_wall_s"])
    shaper = WallClockShaper.from_points([(0.0, bw, delay)], origin)
    binding = {"session_id": f"calibration-{network}", "setup_id": wc["setup_id"], "model_version": "calibration",
               "numeric_contract_hash": contract.contract_hash(), "profile_id": PROFILE_ID}
    ledger = MaterialLedger(os.path.join(wc["work_dir"], f"ledger_{network}_p{party}.jsonl"), binding)
    key = bytes.fromhex(wc["key_hex"])
    client = DealerClient(party, ledger, key, contract, allow_insecure=False)
    svc = DealerServiceClient(wc["dealer_host"], int(wc["dealer_port"]), party, key, contract.contract_hash())
    svc.connect()
    if svc.binding.get("dealer_mode") != MODE_ISOLATED:
        raise MaterialError("CALIBRATION_DEALER_MODE: the dealer service is not in ISOLATED_RECIPIENT_BUNDLES mode")
    port_file = wc["port_file"]
    if party == 0:
        lsock, port = open_listener("127.0.0.1", 0)
        with open(port_file, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(str(port))
        ch = accept_channel(lsock, 0, shaper, timeout_s=120.0)
    else:
        deadline = time.time() + 120.0
        port = None
        while time.time() < deadline:
            if os.path.exists(port_file) and os.path.getsize(port_file) > 0:
                with open(port_file, "r", encoding="utf-8") as fh:
                    port = int(fh.read().strip())
                break
            time.sleep(0.05)
        if port is None:
            raise MaterialError("CALIBRATION_PEER: P0 did not publish its port")
        ch = dial_channel("127.0.0.1", port, 1, shaper)
    mask_shapes: Dict[str, Tuple[int, int]] = {}
    sbc = ServiceBackedClient(client, svc, mask_shapes)
    p = Party(party, ch, sbc, contract, None, np.random.default_rng(1000 + party), model_version="calibration")
    repeats = int(wc["repeats"])
    rows_path = wc["rows_path"]
    configs = [Config(c["component"], int(c["n"]), int(c["d_in"]), int(c["d_out"])) for c in wc["configs"]]
    installed_shapes: set = set()
    t_start = time.perf_counter()
    try:
        for ci, cfg in enumerate(configs):
            data = inputs_for(cfg, party, contract)
            mat_id = mat_id_for(cfg)
            # one-time preparation outside every timed interval
            if cfg.component in ("linear_wmask", "embedding_wmask"):
                mask_shapes[mat_id] = (cfg.d_in, cfg.d_out)
                if (cfg.d_in, cfg.d_out) not in installed_shapes:
                    sbc.set_scope(f"{cfg.cfg_id}:prep")
                    W_enc = data["W_enc"] if party == 0 else None
                    p.install_wmask(mat_id, W_enc)
                    installed_shapes.add((cfg.d_in, cfg.d_out))
            if cfg.component == "a2h_exact":
                import importlib.util
                if importlib.util.find_spec("tenseal") is None:
                    _append_row(rows_path, {"cfg_id": cfg.cfg_id, "component": cfg.component, "n": cfg.n,
                                            "d_in": cfg.d_in, "d_out": cfg.d_out, "network": network, "party": party,
                                            "skipped": True, "reason": "tenseal unavailable"})
                    ch.exchange(b"S")
                    continue
            # warm-up (records the exact material consumption of one call); the planner estimate is
            # provisioned first so that the warm-up itself rarely stalls
            sbc.set_scope(f"{cfg.cfg_id}:est")
            sbc.provision_many(estimated_takes(cfg, contract, mat_id))
            sbc.set_scope(f"{cfg.cfg_id}:warm")
            sbc.record = []
            if cfg.component == "install_wmask":
                wm = mat_id_for(cfg, -1)
                mask_shapes[wm] = (cfg.d_in, cfg.d_out)
                run_gadget(p, cfg, data, wm)
            else:
                run_gadget(p, cfg, data, mat_id)
            takes = list(sbc.record)
            sbc.record = None
            for rep in range(repeats):
                sbc.set_scope(f"{cfg.cfg_id}:r{rep}:pre")
                sbc.provision_many(takes)            # one bundle per material type (installs use a fresh mat_id below)
                call_mat = mat_id
                if cfg.component == "install_wmask":
                    call_mat = mat_id_for(cfg, rep)
                    mask_shapes[call_mat] = (cfg.d_in, cfg.d_out)
                    sbc.provision("wmask_mask", {"mat_id": call_mat, "shape": [cfg.d_in, cfg.d_out]}, 1)
                sbc.set_scope(f"{cfg.cfg_id}:r{rep}:call")
                stalls0, prov0 = sbc.stalls, sbc.provision_s
                ch.exchange(b"B")                     # barrier: both timers start together
                t0 = time.perf_counter()
                st = run_gadget(p, cfg, data, call_mat)
                wall = time.perf_counter() - t0
                inside = sbc.provision_s - prov0
                row = {"cfg_id": cfg.cfg_id, "component": cfg.component, "n": cfg.n, "d_in": cfg.d_in,
                       "d_out": cfg.d_out, "network": network, "bandwidth_bps": bw, "one_way_delay_s": delay,
                       "one_way_latency_s": delay, "cpu_share": 1.0, "repeat": rep, "party": party,
                       "blas_threads": BLAS_THREADS,
                       "split": split_of(cfg.component, cfg.n, cfg.d_in, cfg.d_out, network),
                       "wall_s": wall, "gadget_s": st["seconds"], "measured_party_s": max(0.0, st["seconds"] - inside),
                       "bytes": st["bytes"], "rounds": st["rounds"], "material_bytes": st["material_bytes"],
                       "stalls": sbc.stalls - stalls0, "provision_inside_s": inside, "skipped": False}
                _append_row(rows_path, row)
            sys.stderr.write(f"[calib p{party} {network}] {ci + 1}/{len(configs)} {cfg.cfg_id} done "
                             f"({time.perf_counter() - t_start:.1f} s)\n")
        ch.exchange(b"D")
    finally:
        try:
            svc.close()
        except Exception:
            pass
        ch.close()
        ledger.close()
    return 0


def _append_row(path: str, row: dict) -> None:
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def _write_json(path: Any, obj: Any, **kw: Any) -> None:
    """Write a JSON document with LF line endings on every platform."""
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, **kw)


# ---------------------------------------------------------------------------
# parent side: processes
# ---------------------------------------------------------------------------
def _pydir() -> str:
    return str(Path(__file__).resolve().parents[2])


# BLAS threads per process: three processes (dealer, P0, P1) share the host, so each gets a quarter of the
# logical processors (at most 8); without the cap the parties oversubscribe the CPU and GEMM-heavy
# calls measure several times slower than in a single process (observed on the first smoke run).
BLAS_THREADS = max(1, min(8, (os.cpu_count() or 4) // 4))


def _env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = _pydir() + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("HF_HUB_OFFLINE", "1")
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[var] = str(BLAS_THREADS)
    return env


def start_dealer(keys_file: str, setup_id: str, log_path: str) -> Tuple[subprocess.Popen, str, int]:
    """Start the isolated dealer service; returns (process, host, port)."""
    log = open(log_path, "a", encoding="utf-8", newline="\n")
    proc = subprocess.Popen([PYTHON, "-m", "opencrab2.platform.dealer_service", "--listen", "127.0.0.1:0",
                             "--keys", keys_file, "--contract", DEFAULT_CONTRACT.name, "--setup-id", setup_id,
                             "--model-version", "calibration", "--mode", MODE_ISOLATED],
                            cwd=_pydir(), env=_env(), stdout=subprocess.PIPE, stderr=log, text=True)
    line = proc.stdout.readline() if proc.stdout else ""
    if not line.startswith("LISTENING"):
        proc.kill()
        raise MaterialError(f"CALIBRATION_DEALER_START: dealer service did not start ({line.strip()!r}); see {log_path}")
    tok = line.split()
    return proc, tok[1], int(tok[2])


def run_network(network: str, configs: Sequence[Config], repeats: int, work: Path, dealer: Tuple[str, int],
                keys: Dict[int, bytes], setup_id: str, timeout_s: float) -> List[dict]:
    bw, delay = ANCHORS[network]
    rows_paths = [str(work / f"rows_{network}_p{p}.jsonl") for p in (0, 1)]
    port_file = str(work / f"port_{network}.txt")
    for f in rows_paths + [port_file]:
        if os.path.exists(f):
            os.remove(f)
    procs = []
    origin = time.time()
    for party in (0, 1):
        cfg = {"network": network, "bandwidth_bps": bw, "one_way_delay_s": delay, "origin_wall_s": origin,
               "dealer_host": dealer[0], "dealer_port": dealer[1], "key_hex": keys[party].hex(),
               "configs": [c.to_dict() for c in configs], "repeats": int(repeats), "rows_path": rows_paths[party],
               "port_file": port_file, "setup_id": setup_id, "work_dir": str(work)}
        cpath = work / f"worker_{network}_p{party}.json"
        _write_json(cpath, cfg)
        log = open(work / f"worker_{network}_p{party}.log", "a", encoding="utf-8", newline="\n")
        procs.append(subprocess.Popen([PYTHON, "-m", "opencrab2.experiments.run_calibration", "--worker", str(party),
                                       "--config", str(cpath)], cwd=_pydir(), env=_env(), stdout=log, stderr=log))
    deadline = time.time() + timeout_s
    while time.time() < deadline and any(p.poll() is None for p in procs):
        time.sleep(0.2)
    failed = []
    for party, p in enumerate(procs):
        if p.poll() is None:
            p.kill()
            failed.append(f"P{party} timed out")
        elif p.returncode != 0:
            failed.append(f"P{party} exit {p.returncode}")
    if failed:
        raise MaterialError(f"CALIBRATION_WORKER_FAILED: {network}: {'; '.join(failed)}; see {work}")
    by_party: List[Dict[Tuple[str, int], dict]] = [{}, {}]
    for party in (0, 1):
        with open(rows_paths[party], "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    if r.get("skipped"):
                        by_party[party][(r["cfg_id"], -1)] = r
                    else:
                        by_party[party][(r["cfg_id"], int(r["repeat"]))] = r
    rows: List[dict] = []
    for key, r0 in sorted(by_party[0].items()):
        r1 = by_party[1].get(key)
        if r1 is None:
            continue
        row = dict(r0)
        row.pop("party", None)
        if r0.get("skipped"):
            rows.append(row)
            continue
        row["measured_p0_s"] = float(r0["measured_party_s"])
        row["measured_p1_s"] = float(r1["measured_party_s"])
        row["measured_s"] = max(row["measured_p0_s"], row["measured_p1_s"])
        row["bytes_p1"] = int(r1["bytes"])
        row["stalls"] = int(r0["stalls"]) + int(r1["stalls"])
        row["provision_inside_s"] = float(r0["provision_inside_s"]) + float(r1["provision_inside_s"])
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# fitting and evaluation (pure functions over rows)
# ---------------------------------------------------------------------------
def fit_rows(rows: Sequence[Mapping[str, Any]], status: str = "MEASURED_NATIVE",
             source: str = "run_calibration") -> cm.FittedCostModel:
    train = [r for r in rows if r.get("split") == "train" and not r.get("skipped")]
    if not train:
        raise ValueError("CALIBRATION_NO_TRAIN_ROWS: nothing to fit")
    return cm.FittedCostModel.fit(train, status=status, source=source)


def complete_model(fitted: cm.FittedCostModel) -> Tuple[cm.FittedCostModel, Dict[str, str]]:
    """Fill components without a fresh fit from the hand-written MODELED baseline (status kept)."""
    hand = cm.handwritten_model()
    filled: Dict[str, str] = {}
    out = cm.FittedCostModel(fits=dict(fitted.fits), rel_bw=fitted.rel_bw, rel_rtt=fitted.rel_rtt, label=fitted.label)
    for comp in cm.COMPONENTS:
        if comp not in out.fits:
            out.fits[comp] = hand.fits[comp]
        filled[comp] = out.fits[comp].status
    return out, filled


def _predict(model: Any, r: Mapping[str, Any]) -> Tuple[float, float]:
    env = cm.Env(r["bandwidth_bps"], r["one_way_delay_s"], float(r.get("cpu_share", 1.0)))
    if model == "handwritten":
        return cm.handwritten_predict(r["component"], r["n"], r["d_in"], r["d_out"], env), float("nan")
    if not model.has(r["component"]):
        return float("nan"), float("nan")
    return model.predict(r["component"], r["n"], r["d_in"], r["d_out"], env)


class UncertaintyAwareModel:
    """The fitted model read the way the B4 controller reads it: a lane is chosen by its pessimistic cost.

    ``predict`` returns the fitted (mean, sigma) unchanged, so the regression
    metrics and the two-sigma coverage are those of the fitted model;
    ``decision_cost`` is ``mean + z sigma`` and is what the lane and plan
    metrics compare (``z = UNCERTAINTY_Z``, declared here before any fit).
    """

    def __init__(self, base: Any, z: float = 1.0) -> None:
        self.base = base
        self.z = float(z)

    def has(self, component: str) -> bool:
        return self.base.has(component)

    def predict(self, component: str, n: int, d_in: int, d_out: int, env: Any) -> Tuple[float, float]:
        return self.base.predict(component, n, d_in, d_out, env)

    def decision_cost(self, component: str, n: int, d_in: int, d_out: int, env: Any) -> float:
        mu, sg = self.base.predict(component, n, d_in, d_out, env)
        return float(mu) + self.z * (float(sg) if math.isfinite(float(sg)) else 0.0)


UNCERTAINTY_Z = 1.0


def _decision_cost(model: Any, r: Mapping[str, Any]) -> float:
    """What a model compares when it picks a lane: the mean, or the pessimistic cost of an uncertainty-aware model."""
    if hasattr(model, "decision_cost"):
        if not model.has(r["component"]):
            return float("nan")
        env = cm.Env(r["bandwidth_bps"], r["one_way_delay_s"], float(r.get("cpu_share", 1.0)))
        return model.decision_cost(r["component"], r["n"], r["d_in"], r["d_out"], env)
    return _predict(model, r)[0]


def _group_key(r: Mapping[str, Any]) -> tuple:
    return (int(r["n"]), int(r["d_in"]), int(r["d_out"]), str(r["network"]))


def lane_metrics(rows: Sequence[Mapping[str, Any]], model: Any) -> dict:
    """Top-1 lane accuracy, regret, informative versus dominated groups, near-threshold decisions, plan identity."""
    meas: Dict[tuple, Dict[str, List[float]]] = {}
    for r in rows:
        if r.get("skipped"):
            continue
        meas.setdefault(_group_key(r), {}).setdefault(r["component"], []).append(float(r["measured_s"]))
    preds_cache: Dict[Tuple[tuple, str], float] = {}

    def pred(key: tuple, comp: str, r: Mapping[str, Any]) -> float:
        ck = (key, comp)
        if ck not in preds_cache:
            preds_cache[ck] = _decision_cost(model, r)
        return preds_cache[ck]

    rows_by_key: Dict[Tuple[tuple, str], Mapping[str, Any]] = {}
    for r in rows:
        if not r.get("skipped"):
            rows_by_key[(_group_key(r), r["component"])] = r
    groups_out: Dict[str, dict] = {}
    total = top1 = 0
    regret = 0.0
    informative_total = informative_ok = dominated_total = dominated_ok = 0
    near_total = near_ok = far_total = far_ok = 0
    for gname, members in LANE_GROUPS.items():
        decisions = []
        for key, comps in meas.items():
            present = [c for c in members if c in comps]
            if len(present) < 2:
                continue
            m = {c: float(np.mean(comps[c])) for c in present}
            p = {c: pred(key, c, rows_by_key[(key, c)]) for c in present}
            if any(not math.isfinite(v) for v in p.values()):
                continue
            bm = min(m, key=m.get)
            bp = min(p, key=p.get)
            vals = sorted(m.values())
            gap = (vals[1] - vals[0]) / max(vals[0], 1e-12)
            decisions.append({"key": list(key), "measured_best": bm, "predicted_best": bp, "ok": bm == bp,
                              "regret_s": m[bp] - m[bm], "rel_gap": gap})
        winners = {d["measured_best"] for d in decisions}
        informative = len(winners) > 1
        ok = sum(1 for d in decisions if d["ok"])
        total += len(decisions)
        top1 += ok
        regret += sum(d["regret_s"] for d in decisions)
        if informative:
            informative_total += len(decisions)
            informative_ok += ok
        else:
            dominated_total += len(decisions)
            dominated_ok += ok
        for d in decisions:
            if d["rel_gap"] < NEAR_THRESHOLD_REL:
                near_total += 1
                near_ok += int(d["ok"])
            else:
                far_total += 1
                far_ok += int(d["ok"])
        groups_out[gname] = {"decisions": len(decisions), "top1": ok, "informative": informative,
                             "measured_winners": sorted(winners), "regret_s": sum(d["regret_s"] for d in decisions),
                             "accuracy": ok / len(decisions) if decisions else float("nan")}
    # full-plan identity per (n, network): the plan is the tuple of lane winners summed over the grid shapes
    plan_total = plan_ok = 0
    by_nn: Dict[Tuple[int, str], Dict[str, Dict[str, float]]] = {}
    for key, comps in meas.items():
        n, d_in, d_out, net = key
        for gname, members in LANE_GROUPS.items():
            present = [c for c in members if c in comps]
            if len(present) != len(members):
                continue
            slot = by_nn.setdefault((n, net), {}).setdefault(gname, {"measured": {}, "predicted": {}})
            for c in present:
                slot["measured"][c] = slot["measured"].get(c, 0.0) + float(np.mean(comps[c]))
                slot["predicted"][c] = slot["predicted"].get(c, 0.0) + pred(key, c, rows_by_key[(key, c)])
    for (n, net), groups in by_nn.items():
        if len(groups) < 2:
            continue
        mp = tuple(min(g["measured"], key=g["measured"].get) for g in groups.values())
        pp = tuple(min(g["predicted"], key=g["predicted"].get) for g in groups.values())
        plan_total += 1
        plan_ok += int(mp == pp)
    return {
        "top1_lane_acc": top1 / total if total else float("nan"), "lane_decisions": total, "realized_regret_s": regret,
        "informative_pair_acc": informative_ok / informative_total if informative_total else float("nan"),
        "informative_pair_decisions": informative_total, "informative_pair_correct": informative_ok,
        "dominated_pair_acc": dominated_ok / dominated_total if dominated_total else float("nan"),
        "dominated_pair_decisions": dominated_total,
        "near_threshold_acc": near_ok / near_total if near_total else float("nan"), "near_threshold_decisions": near_total,
        "far_threshold_acc": far_ok / far_total if far_total else float("nan"), "far_threshold_decisions": far_total,
        "plan_identity": plan_ok / plan_total if plan_total else float("nan"), "plans": plan_total,
        "groups": groups_out,
    }


def evaluate_models(rows: Sequence[Mapping[str, Any]], models: Mapping[str, Any]) -> dict:
    """Per model and split: error metrics, two-sigma coverage, lane metrics."""
    out: Dict[str, Any] = {}
    for name, model in models.items():
        out[name] = {}
        for split in ("train", "test"):
            rs = [r for r in rows if r.get("split") == split and not r.get("skipped")]
            meas, pred, cover = [], [], []
            for r in rs:
                mu, sg = _predict(model, r)
                if not math.isfinite(mu):
                    continue
                meas.append(float(r["measured_s"]))
                pred.append(float(mu))
                if math.isfinite(sg):
                    cover.append(abs(float(r["measured_s"]) - mu) <= COVERAGE_Z * sg)
            res = cm.error_metrics(meas, pred) if meas else {"mape": float("nan"), "median_rel_err": float("nan"),
                                                               "p95_rel_err": float("nan"), "spearman": float("nan"),
                                                               "kendall": float("nan"), "n": 0}
            res["coverage_2sigma"] = float(np.mean(cover)) if cover else float("nan")
            res.update(lane_metrics(rs, model))
            out[name][split] = res
    return out


def rows_to_records(rows: Sequence[Mapping[str, Any]], raw_path: str, campaign_id: str) -> List[dict]:
    recs = []
    for r in rows:
        if r.get("skipped"):
            continue
        recs.append(schema.new_record(
            status="MEASURED_NATIVE", record_kind="calibration", workload_scope="primitive",
            clock_basis="wall-clock-exogenous", dealer_mode=MODE_ISOLATED, security_eligible=False,
            trace_id=f"anchor:{r['network']}", trace_hash=schema.stable_hash(list(ANCHORS[r["network"]])),
            policy_id="none", run_id=f"calib:{campaign_id}:{r['network']}:{r['cfg_id']}:r{r['repeat']}",
            repeat=int(r["repeat"]), T_online_s=float(r["measured_s"]), bytes=int(r["bytes"]), rounds=int(r["rounds"]),
            material_consumed=int(r["material_bytes"]), raw_measurement_path=raw_path, label=r["component"],
            split=r["split"], extra=dict(r)))
    return recs


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def fit_and_write(rows: List[dict], out_root: Path, meta: dict) -> dict:
    """Fit on TRAIN, evaluate, complete the model and write every output; returns the metrics dict."""
    fitted = fit_rows(rows)
    model, filled = complete_model(fitted)
    models: Dict[str, Any] = {"handwritten_MODELED": "handwritten", "fitted_MEASURED_NATIVE": fitted,
                              "uncertainty_MEASURED_NATIVE": UncertaintyAwareModel(fitted, UNCERTAINTY_Z)}
    metrics = evaluate_models(rows, models)
    calib = out_root / "calibration"
    calib.mkdir(parents=True, exist_ok=True)
    model.save(calib / "fitted_model.json")
    summary = {
        "campaign": meta, "split_rule": {"holdout_shapes": sorted(map(list, HOLDOUT_SHAPES)),
                                         "holdout_network": HOLDOUT_NETWORK, "holdout_n": HOLDOUT_N},
        "n_rows": len([r for r in rows if not r.get("skipped")]),
        "n_train": len([r for r in rows if r.get("split") == "train" and not r.get("skipped")]),
        "n_test": len([r for r in rows if r.get("split") == "test" and not r.get("skipped")]),
        "rows_with_stalls": len([r for r in rows if int(r.get("stalls", 0)) > 0]),
        "skipped": [r for r in rows if r.get("skipped")],
        "component_status": filled,
        "fitted_components": sorted(fitted.fits),
        "sigma_compute_s": {k: v.sigma_compute_s for k, v in fitted.fits.items()},
        "models": metrics,
        "informative_pair_note": "accuracy restricted to lane groups whose measured winner changes inside the split; "
                                 "dominated groups (one winner everywhere) are reported separately",
    }
    _write_json(calib / "calibration_metrics.json", summary, indent=1, sort_keys=True, default=_json_default)
    write_csv(out_root / "processed" / "calibration_rows.csv", rows)
    raw_path = out_root / "raw" / "calibration.jsonl"
    for rec in rows_to_records(rows, schema.relative_raw_path(raw_path, out_root), meta.get("campaign_id", "")):
        schema.append_jsonl(raw_path, rec)
    return summary


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    raise TypeError(f"not serialisable: {type(o).__name__}")


# ---------------------------------------------------------------------------
# campaign
# ---------------------------------------------------------------------------
def run_campaign(quick: bool, networks: Sequence[str], repeats: int, out_root: Path,
                 components: Optional[Sequence[str]] = None, timeout_s: Optional[float] = None,
                 resume: bool = False) -> dict:
    """Measure the grid on every network, fit, evaluate, write. ``resume`` continues the last unfinished campaign
    with the same declared grid: networks whose merged rows are on disk are not measured again."""
    t_start = time.perf_counter()
    grid = build_grid(quick, components)
    decl = schema.stable_hash({"quick": bool(quick), "networks": list(networks), "repeats": int(repeats),
                               "configs": [c.to_dict() for c in grid], "code": "calibration-v2-gamma"})
    state_path = out_root / "calibration" / "work" / "resume.json"
    campaign_id = time.strftime("%Y%m%dT%H%M%S")
    if resume and state_path.is_file():
        try:
            st = json.loads(state_path.read_text(encoding="utf-8"))
            if st.get("declaration") == decl and not st.get("finished"):
                campaign_id = str(st["campaign_id"])
        except (ValueError, KeyError):
            pass
    work = out_root / "calibration" / "work" / campaign_id
    work.mkdir(parents=True, exist_ok=True)
    _write_json(state_path, {"campaign_id": campaign_id, "declaration": decl, "finished": False})
    _write_json(out_root / "calibration" / "grid.json",
                {"quick": quick, "networks": list(networks), "repeats": repeats,
                 "configs": [c.to_dict() for c in grid],
                 "split_rule": {"holdout_shapes": sorted(map(list, HOLDOUT_SHAPES)),
                                "holdout_network": HOLDOUT_NETWORK, "holdout_n": HOLDOUT_N}}, indent=1)
    keys = {0: secrets.token_bytes(32), 1: secrets.token_bytes(32)}
    keys_file = str(work / "keys.json")
    from opencrab2.platform.dealer_service import write_keys
    write_keys(keys_file, keys)
    setup_id = f"calib-{campaign_id}"
    dealer_proc, host, port = start_dealer(keys_file, setup_id, str(work / "dealer.log"))
    rows: List[dict] = []
    per_network_s: Dict[str, float] = {}
    try:
        for net in networks:
            if net not in ANCHORS:
                raise ValueError(f"CALIBRATION_NETWORK: unknown network {net!r}; known {list(ANCHORS)}")
            t0 = time.perf_counter()
            to = timeout_s if timeout_s is not None else (1200.0 if quick else 4 * 3600.0)
            done_path = work / f"rows_done_{net}.json"
            if resume and done_path.is_file():
                with open(done_path, "r", encoding="utf-8") as fh:
                    rows += json.load(fh)
                per_network_s[net] = 0.0
                sys.stdout.write(f"network {net}: resumed from {done_path}\n")
                continue
            net_rows = run_network(net, grid, repeats, work, (host, port), keys, setup_id, to)
            _write_json(done_path, net_rows)
            rows += net_rows
            per_network_s[net] = time.perf_counter() - t0
            sys.stdout.write(f"network {net}: {per_network_s[net]:.1f} s, {len(rows)} rows so far\n")
            sys.stdout.flush()
    finally:
        dealer_proc.kill()
        try:
            dealer_proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            pass
    _write_json(work / "rows_merged.json", rows)
    meta = {"campaign_id": campaign_id, "quick": quick, "networks": list(networks), "repeats": repeats,
            "per_network_s": per_network_s, "dealer_mode": MODE_ISOLATED, "host": _host_info(),
            "code_commit": schema.tree_hash(), "contract": DEFAULT_CONTRACT.name,
            "formulas_sha256": _formulas_sha256(),
            "elapsed_s": time.perf_counter() - t_start}
    summary = fit_and_write(rows, out_root, meta)
    summary["uncertainty_z"] = UNCERTAINTY_Z
    _write_json(state_path, {"campaign_id": campaign_id, "declaration": decl, "finished": True})
    summary["campaign"]["elapsed_s"] = time.perf_counter() - t_start
    _write_json(out_root / "calibration" / "calibration_metrics.json", summary, indent=1, sort_keys=True,
                default=_json_default)
    return summary


def _formulas_sha256() -> Optional[str]:
    """Hash of the shared formulas file the campaign priced with (make_tables marks older calibrations stale)."""
    try:
        return hashlib.sha256(Path(cm.formulas_path()).read_bytes()).hexdigest()
    except OSError:
        return None


def _host_info() -> dict:
    import platform
    return {"platform": platform.platform(), "processor": platform.processor(), "python": sys.version.split()[0],
            "cpu_count": os.cpu_count()}


def previous_campaign_block(out_root: Path) -> Optional[dict]:
    """The ``campaign`` block of an existing ``calibration/calibration_metrics.json`` (None when there is none)."""
    path = Path(out_root) / "calibration" / "calibration_metrics.json"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            previous = json.load(fh)
    except (OSError, ValueError):
        return None
    block = previous.get("campaign") if isinstance(previous, dict) else None
    return dict(block) if isinstance(block, dict) else None


def refit_from_raw(out_root: Path) -> dict:
    """Refit the model from the raw calibration records (``raw/calibration.jsonl`` or its ``.gz`` form).

    A refit measures nothing: the ``campaign`` block of the existing metrics
    file (campaign id, host, the hash of the formulas file the campaign priced
    with) is carried over unchanged, with ``refit`` set and the refit time
    stamped. Without an existing block the file has no campaign block and the
    tables treat the calibration as stale, which is the honest statement.
    """
    raw = schema.read_jsonl(out_root / "raw" / "calibration.jsonl")
    rows = [dict(r["extra"]) for r in raw if r.get("record_kind") == "calibration"]
    if not rows:
        raise ValueError("CALIBRATION_NO_ROWS: raw/calibration.jsonl holds no calibration records")
    campaign = previous_campaign_block(out_root)
    fitted = fit_rows(rows)
    model, filled = complete_model(fitted)
    models: Dict[str, Any] = {"handwritten_MODELED": "handwritten", "fitted_MEASURED_NATIVE": fitted,
                              "uncertainty_MEASURED_NATIVE": UncertaintyAwareModel(fitted, UNCERTAINTY_Z)}
    metrics = evaluate_models(rows, models)
    model.save(out_root / "calibration" / "fitted_model.json")
    summary = {"refit": True, "refit_utc": schema.utc_now(), "n_rows": len(rows), "component_status": filled,
               "models": metrics,
               "split_rule": {"holdout_shapes": sorted(map(list, HOLDOUT_SHAPES)),
                              "holdout_network": HOLDOUT_NETWORK, "holdout_n": HOLDOUT_N},
               "n_train": len([r for r in rows if r.get("split") == "train" and not r.get("skipped")]),
               "n_test": len([r for r in rows if r.get("split") == "test" and not r.get("skipped")]),
               "fitted_components": sorted(fitted.fits),
               "sigma_compute_s": {k: v.sigma_compute_s for k, v in fitted.fits.items()},
               "uncertainty_z": UNCERTAINTY_Z}
    if campaign is not None:
        summary["campaign"] = campaign
    _write_json(out_root / "calibration" / "calibration_metrics.json", summary, indent=1, sort_keys=True,
                default=_json_default)
    return summary


def _print_summary(summary: dict) -> None:
    for name, splits in summary.get("models", {}).items():
        t = splits.get("test", {})
        sys.stdout.write(f"{name:>26s} test: n={t.get('n')} mape={t.get('mape'):.3f} median={t.get('median_rel_err'):.3f} "
                         f"p95={t.get('p95_rel_err'):.3f} rho={t.get('spearman'):.3f} lane={t.get('top1_lane_acc'):.3f} "
                         f"informative={t.get('informative_pair_correct')}/{t.get('informative_pair_decisions')} "
                         f"coverage={t.get('coverage_2sigma'):.3f}\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run_calibration", description="shaped two-process cost-model calibration")
    p.add_argument("--quick", action="store_true", help="small grid, two networks, two repeats (under 10 minutes)")
    p.add_argument("--networks", default=None, help="comma separated subset of LAN,WAN,WAN-c,WAN-mid")
    p.add_argument("--repeats", type=int, default=None)
    p.add_argument("--components", default=None, help="comma separated subset of costmodel.COMPONENTS")
    p.add_argument("--out-root", default=None, help="data root (default: OPENCRAB2_RESULTS_DIR, else the tree's results/)")
    p.add_argument("--timeout", type=float, default=None, help="per-network worker timeout in seconds")
    p.add_argument("--fit-only", action="store_true",
                   help="refit from raw/calibration.jsonl (keeps the campaign block of the existing metrics file)")
    p.add_argument("--resume", action="store_true",
                   help="continue the last unfinished campaign with the same declared grid (finished networks are kept)")
    p.add_argument("--worker", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--config", default=None, help=argparse.SUPPRESS)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.worker is not None:
        try:
            return worker_main(int(args.worker), args.config)
        except (OpenCrabError, OSError, ValueError) as exc:
            sys.stderr.write(f"[calib worker {args.worker}] {exc}\n")
            return 3
    out_root = Path(args.out_root) if args.out_root else schema.experiments_root()
    if args.fit_only:
        summary = refit_from_raw(out_root)
        _print_summary(summary)
        return 0
    networks = tuple(args.networks.split(",")) if args.networks else (QUICK_NETWORKS if args.quick else NETWORKS)
    repeats = args.repeats if args.repeats is not None else (QUICK_REPEATS if args.quick else DEFAULT_REPEATS)
    comps = args.components.split(",") if args.components else None
    summary = run_campaign(args.quick, networks, repeats, out_root, comps, args.timeout, args.resume)
    _print_summary(summary)
    sys.stdout.write(f"elapsed {summary['campaign']['elapsed_s']:.1f} s; model -> {out_root / 'calibration' / 'fitted_model.json'}\n")
    return 0


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main())


__all__ = ["Config", "NETWORKS", "HOLDOUT_SHAPES", "HOLDOUT_NETWORK", "HOLDOUT_N", "LANE_GROUPS", "split_of",
           "build_grid", "ServiceBackedClient", "inputs_for", "run_gadget", "worker_main", "run_network",
           "fit_rows", "complete_model", "lane_metrics", "evaluate_models", "rows_to_records", "fit_and_write",
           "run_campaign", "previous_campaign_block", "refit_from_raw", "UncertaintyAwareModel", "UNCERTAINTY_Z",
           "main"]
