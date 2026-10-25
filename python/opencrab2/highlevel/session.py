"""Session runtime: the epoch loop and the evidence record (paper, Section 4, state-aware session control).

Both parties run :func:`run_session_party` symmetrically (``party`` selects the
asymmetric steps) over one :class:`~opencrab2.highlevel.channel.Channel` (TCP
or the WebSocket adapter of the platform). A session generates ``n_generate``
tokens of a GPT-2 style model in ``n_generate`` epochs: epoch 0 is the prefill
of the client's private prompt (the first token is released at its end), every
later epoch is one greedy decode step. Per epoch, in lock step:

1. **Cancellation status** (public, declared): both parties exchange their
   cancel flags; if either is raised the epoch is not started, RESERVED
   material is burned, buffered one-time material is burned as safe waste,
   and the session stops (it may be resumed later under the same session_id).
2. **Sense**: the fixed-size probe of ``telemetry.sense_and_exchange`` (3 x
   16-byte ping/pong plus one 64 KiB burst) at a fixed point of every epoch;
   P1 signs the frame, both verify it, both feed it to identical controllers.
3. **Candidates and decision**: ``candidates.enumerate_candidates`` over the
   public state S_t (incumbent, installed objects with versions, material stock,
   KV length bucket, remaining budget) with the shared admission gate
   ``registry.admit`` on every plan's tape; the policy (B0 .. B6 and the B4
   ablations of ``controller.make_policy``) decides. The remaining epochs are
   priced step by step on their own public shapes (:func:`horizon_schedule`:
   the prompt rows for a prefill or recovery prefill, one row for each later
   decode step, each with its KV length bucket).
4. **Two-phase commit**: phase A exchanges the SHA-256 digest of (plan,
   installed object versions, target objects, epoch, session binding); on a
   mismatch the epoch is aborted and the old plan is kept (its digest must then
   agree). Phase B prepares resources: the parties request the epoch's material
   from the isolated dealer by type and count (deterministic request ids, each
   party receives only its own half), request the mask bundles of objects to
   install, and share server-private vectors a generic realization needs; both
   exchange a readiness flag and fall back to the old plan when either is not
   ready. Phase C installs the missing weight-masked objects through the real
   one-round ``Party.install_wmask`` (which registers the versioned object in
   the ledger) and exchanges the digest of the installed set; a mismatch there
   is fatal because public installation records cannot be rolled back.
5. **Execute** the declared gate (prefill or one decode step) under the
   committed plan; the token is released to P1 only through ``ReleasePolicy``.
6. **Prefetch**: the next epoch's material for the current plan is requested
   ahead of time, so that a later switch wastes it (counted as plan-switch
   waste, burned in the ledger) and so that the inventory state S_{t+1} is real.

Material requests are upper bounds derived from the gadget realizations of
``party.py`` (:func:`epoch_material_plan`); a shortfall during execution is
served by a symmetric stall request (counted). Server-private vectors (biases,
LayerNorm parameters, and the full matrices of a generic realization) are
shared dealer-free: P0 secret-shares them to P1 in one message per group, so
no weight ever reaches the dealer.

Resumption: the public state (epochs done, plan, installed objects, counts) is
written after every epoch. A warm resume (runner alive, as in the Web sidecar)
re-attaches a channel and continues at the next epoch; a cold resume (new
process) reloads the public state, recovers the ledger (RESERVED -> BURNED,
in-memory material gone -> AVAILABLE ids burned), re-runs the prefill over the
prompt plus the tokens generated so far as the recovery gate, then continues.
Either way the handshake binds (session_id, setup_id, model_version, contract
hash, profile, shapes, policy, trace hash); a model_version mismatch is
refused with ``OBJECT_VERSION_MISMATCH``.

Timing boundaries: ``T_cold`` is the wall time from the first protocol
message of the handshake to the end of the last gate and includes key/setup,
material generation and delivery waits, installation, online execution,
sensing, planning, commit and service overhead; ``T_warm`` subtracts setup
and installation (the declared reusable state: installed objects, shared
vectors, keys); ``T_online`` is the sum of the gate wall times only. Excluded
from every figure: process start, weight loading from disk, journal opening,
the final summary exchange and the record write (listed in the detail file as
``timing_exclusions``).

Warm stacks (item (c) of the campaign plan): a :class:`WarmStack` holds what
one party keeps between sessions of one deployment per ``(model_version,
client_key_domain, n_layers)``: the ledger (bound to the stack id), the dealer
connection and ``DealerClient``, the per-matrix mask shares, the installed
offsets E = W - B with their versions, the shares of the server-private
vectors and matrices, and the last committed plan. A runner given ``warm=``
starts from that state: nothing is installed or shared again, the first plan
is the stack's last plan, and the record's ``T_warm_s`` is the measured wall
time of the warm request (``extra.T_warm_basis = "measured-warm-request"``;
``T_cold_s`` is then None because no cold time was measured). Both parties
bind the stack's public digest (object ids, versions, mask material ids,
vector names, plan) in the handshake; a difference is refused with
``SESSION_BINDING_MISMATCH``, so a one-sided stack can never be used. Every
counter of the record is a per-session delta, so sessions that share a
channel, a ledger or a dealer connection do not inherit each other's totals.

Party randomness: the masks a party draws (``share_input`` sends value minus
mask) come from the OS CSPRNG by default (``SessionConfig.party_rng_from_os``).
With the switch off the masks derive from the public ``seed``, a peer that
knows the seed recovers the inputs, and the record is not security evidence:
``security_eligible`` is False and ``extra.party_rng`` says ``public-seed``.

Released tokens stay with the client. The record carries ``tokens`` (a count);
the P1 record additionally carries, when ``SessionConfig.record_tokens`` is
set, ``extra.generated_tokens``, ``extra.reference_tokens`` and
``extra.first_mismatch_step`` (the first step whose released token differs
from the reference, None when they agree or no reference is known), so that
an output mismatch can be reported from the record alone. ``output_match``
(released tokens equal to the client's reference) is computed by P1 from its
own view and written to the P1 record only; it is never sent to P0, whose
record carries ``output_match: null`` and no token. ``raw_measurement_path``
names the party's detail file relative to the data root (never a host path).

What is tested (tests/test_session.py): a tiny random-weight model over
loopback with two decode epochs for B0_fixed, B1_naive,
B2_remaining_plus_install and B4_opencrab against an in-process isolated
dealer service; the record validates; ``security_eligible`` is True only in
ISOLATED_RECIPIENT_BUNDLES mode; cancellation burns the buffered material
and leaves no RESERVED id; a resume with a different model_version is refused;
a warm resume completes the remaining epochs; the material estimates are upper
bounds of the consumption. Design limits: the material estimate is a bound,
not the exact consumption (the stall path is the safety net); the public
state file is local and does not resist a rolled-back host; one dealer
service serves one session at a time because weight-mask ids are not
namespaced by session.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..errors import (AdmissionError, ChannelError, ContractViolation, MaterialError, MaterialStall,
                      OpenCrabError)
from . import candidates as cands
from . import controller as ctrl
from . import registry, telemetry
from . import transformer as tf
from .contract import NumericContract, canonical_json_bytes
from .costmodel import Env, FittedCostModel, default_model
from .dealer import MODE_INSECURE, MODE_ISOLATED, PROFILE_ID, DealerClient
from .ledger import MaterialLedger
from .party import Party, ReleasePolicy
from ..lowlevel.csprng import SecureRandom

RECORD_SCHEMA = "opencrab-native-evidence-v2"
RECORD_FIELDS: Tuple[str, ...] = (
    "schema", "status", "security_eligible", "code_commit", "profile_id", "numeric_contract_hash", "dealer_mode",
    "diagnostic_reveals", "workload_scope", "model_revision", "trace_id", "trace_hash", "clock_basis",
    "policy_id", "initial_state_hash", "run_id", "repeat", "initializations", "post_init_switches",
    "rejected_candidates", "aborts", "T_cold_s", "T_warm_s", "T_online_s", "bytes", "rounds",
    "material_generated", "material_transferred", "material_consumed", "material_burned",
    "peak_rss_client", "peak_rss_server", "tokens", "output_match", "raw_measurement_path",
)
CLOCK_BASIS = "wall-clock-exogenous"
COLD_PLAN: Dict[str, Any] = {"embedding": "generic", "linear": "generic", "argmax_k": 2, "max_k": 2}
DEFAULT_FIXED_PLAN: Dict[str, Any] = {"embedding": "wmask", "linear": "wmask", "argmax_k": 2, "max_k": 2}
TIMING_EXCLUSIONS: Tuple[str, ...] = (
    "process start and interpreter import", "weight loading from disk", "ledger journal opening and recovery",
    "final public summary exchange", "evidence record and detail file writes",
    "closing sync after the record write", "dealer status query before the handshake",
)
STATE_SUFFIX = ".state.json"
MAX_GATES_PER_BUNDLE = 1 << 16
MAX_BEAVER_PER_BUNDLE = 1 << 20

_HANDSHAKE_KEYS = ("session_id", "setup_id", "model_version", "numeric_contract_hash", "profile_id", "n_layers",
                   "prompt_len", "n_generate", "length_bucket", "policy", "trace_hash", "resume", "resume_epoch")


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass
class SessionConfig:
    """Public session parameters. Fields after ``length_bucket`` are optional extensions."""
    model: str
    n_layers: int
    prompt_ids: List[int]
    n_generate: int
    policy: str
    trace_id: str
    seed: int
    contract: NumericContract
    profile_id: str = PROFILE_ID
    length_bucket: int = 32
    session_id: str = ""
    setup_id: str = "setup-0"
    model_version: str = ""
    client_key_domain: str = "default"
    initial_plan: Optional[dict] = None
    fixed_plan: Optional[dict] = None
    controller_kw: Dict[str, Any] = field(default_factory=dict)
    prompt_len: Optional[int] = None
    reference_tokens: Optional[List[int]] = None
    record_tokens: bool = True
    cancel_after_epoch: Optional[int] = None
    resume: bool = False
    state_path: Optional[str] = None
    material_wait_budget_s: float = 60.0
    cpu_load_party: Optional[int] = 0
    repeat: int = 0
    run_id: str = ""
    workload_scope: Optional[str] = None
    probe_samples: int = 3
    probe_burst: int = 65536
    costmodel_json: Optional[dict] = None
    telemetry_max_bandwidth_bps: float = 1e11
    ln_prescale_bits: int = 3
    party_rng_from_os: bool = True

    def effective_prompt_len(self) -> int:
        n = int(self.prompt_len) if self.prompt_len is not None else len(self.prompt_ids or [])
        if n < 1:
            raise ContractViolation("SESSION_PARAMS: the prompt must hold at least one token")
        return n

    def validate(self) -> None:
        if int(self.n_generate) < 1:
            raise ContractViolation("SESSION_PARAMS: n_generate must be >= 1")
        if int(self.n_layers) < 1:
            raise ContractViolation("SESSION_PARAMS: n_layers must be >= 1")
        if int(self.length_bucket) < 1:
            raise ContractViolation("SESSION_PARAMS: length_bucket must be >= 1")
        if not isinstance(self.contract, NumericContract):
            raise ContractViolation("SESSION_PARAMS: contract must be a NumericContract")
        self.effective_prompt_len()


# ---------------------------------------------------------------------------
# model loading (P0 loads weights; the model card is public)
# ---------------------------------------------------------------------------
def save_model_npz(path: str, W: Dict[str, np.ndarray], meta: dict) -> str:
    """Store a (tiny or exported) model as ``.npz`` with the meta dict as a JSON string under ``__meta__``."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in W.items()}
    arrays["__meta__"] = np.frombuffer(json.dumps(meta, sort_keys=True).encode("utf-8"), dtype=np.uint8)
    np.savez(str(p), **arrays)
    return str(p)


def load_model(model: str, n_layers: int) -> Tuple[Dict[str, np.ndarray], dict]:
    """``gpt2`` / ``openai-community/gpt2`` from the pinned local cache, or ``npz:<path>`` from :func:`save_model_npz`."""
    n_layers = int(n_layers)
    if model.startswith("npz:"):
        path = model[len("npz:"):]
        if not os.path.isfile(path):
            raise MaterialError(f"MODEL_FILE: {path!r} does not exist")
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(bytes(z["__meta__"]).decode("utf-8"))
            W = {k: np.asarray(z[k], dtype=np.float32) for k in z.files if k != "__meta__"}
        total = int(meta.get("n_layers_total", meta.get("n_layers", n_layers)))
        if n_layers > total:
            raise ContractViolation(f"MODEL_LAYERS: n_layers {n_layers} exceeds the {total} layers of {path!r}")
        meta = dict(meta)
        meta["n_layers"] = n_layers
        return W, meta
    if model in ("gpt2", tf.GPT2_REPO):
        return tf.load_gpt2_weights(n_layers, tf.GPT2_REPO)
    raise ContractViolation(f"MODEL_SPEC: unknown model {model!r} (use 'gpt2' or 'npz:<path>')")


def model_version_of(meta: dict) -> str:
    return f"{meta.get('model', 'model')}@{meta.get('revision', 'unpinned')}"


def workload_scope_for(n_layers: int, n_layers_total: int, override: Optional[str] = None) -> str:
    """Record label of the run: ``full-model`` when every layer runs, else ``prefix-<n>L``.

    The GPT-2 prefixes are the 1-layer and 4-layer runs of the evaluation (paper,
    Section 7.1, models and traces).
    The evidence schema admits a fixed vocabulary (``full-model``, ``prefix-4L``,
    ``prefix-1L``, ``primitive``, ``web``); a label outside it is refused here,
    before any material is consumed, with ``WORKLOAD_SCOPE_UNSUPPORTED`` rather
    than at record-write time after the whole session ran. ``override``
    (``SessionConfig.workload_scope``) must itself be an admitted label.
    """
    n, total = int(n_layers), int(n_layers_total)
    scope = str(override) if override else ("full-model" if n >= total else f"prefix-{n}L")
    allowed: Optional[Tuple[str, ...]] = None
    try:
        from ..experiments import schema as schema_mod   # type: ignore
        allowed = tuple(schema_mod.WORKLOAD_SCOPES)
    except Exception:
        allowed = None
    if allowed is not None and scope not in allowed:
        raise ContractViolation(f"WORKLOAD_SCOPE_UNSUPPORTED: {scope!r} is not one of {list(allowed)}; run a prefix the "
                                f"schema names or set SessionConfig.workload_scope to an admitted label")
    return scope


def model_card(meta: dict, n_layers: int) -> dict:
    """Public architecture description sent by P0 in the handshake (no weights)."""
    return {"model": str(meta.get("model", "model")), "revision": str(meta.get("revision", "unpinned")),
            "weight_sha256": str(meta.get("weight_sha256", "")), "d_model": int(meta["d_model"]),
            "n_head": int(meta["n_head"]), "d_ff": int(meta["d_ff"]), "vocab": int(meta["vocab"]),
            "ln_eps": float(meta.get("ln_eps", 1e-5)), "n_layers": int(n_layers),
            "n_layers_total": int(meta.get("n_layers_total", n_layers)), "n_ctx": int(meta.get("n_ctx", 1024)),
            "activation_contract": "erf_gelu"}


_TREE_HASH: Optional[str] = None


def code_tree_hash() -> str:
    """SHA-256 over the sorted relative paths and contents of every ``opencrab2/**/*.py`` (no git here)."""
    global _TREE_HASH
    if _TREE_HASH is None:
        root = Path(__file__).resolve().parents[1]
        h = hashlib.sha256()
        for p in sorted(root.rglob("*.py")):
            rel = p.relative_to(root).as_posix()
            h.update(rel.encode("utf-8"))
            h.update(b"\0")
            h.update(p.read_bytes())
            h.update(b"\0")
        _TREE_HASH = "tree-" + h.hexdigest()
    return _TREE_HASH


def peak_rss_bytes() -> int:
    """Peak resident set size of this process in bytes (0 when the platform exposes none).

    POSIX: ``ru_maxrss`` of ``resource.getrusage`` (kilobytes on Linux, bytes on macOS); Windows (no
    ``resource`` module): ``psutil`` ``peak_wset``. ``psutil``'s ``rss`` is the current size and is never used.
    """
    try:
        import resource
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return peak if sys.platform == "darwin" else peak * 1024
    except Exception:
        pass
    try:
        import psutil
        peak = getattr(psutil.Process().memory_info(), "peak_wset", None)
        return int(peak) if peak is not None else 0
    except Exception:
        return 0


def record_path(path: Any) -> str:
    """A file path as a record stores it: relative to the data root with forward slashes, never a host path."""
    try:
        from ..experiments.schema import relative_raw_path   # type: ignore
    except Exception:   # pragma: no cover (schema module absent)
        return Path(str(path)).name
    return relative_raw_path(path)


# ---------------------------------------------------------------------------
# public shapes, installable objects, server-private vectors
# ---------------------------------------------------------------------------
def object_shapes(card: dict, n_layers: int) -> Dict[str, Tuple[int, int]]:
    """Shapes of every weight-masked object (dealer mask shape = Party install shape)."""
    d, f, V = int(card["d_model"]), int(card["d_ff"]), int(card["vocab"])
    out: Dict[str, Tuple[int, int]] = {"wte": (V, d), "lm_head": (d, V)}
    for li in range(int(n_layers)):
        out[f"h{li}.qkv"] = (d, 3 * d)
        out[f"h{li}.out"] = (d, d)
        out[f"h{li}.fc"] = (d, f)
        out[f"h{li}.proj"] = (f, d)
    return out


def generic_objects(plan: dict, n_layers: int) -> List[str]:
    """Server-private matrices a plan reads through ``linear_generic`` (shared dealer-free)."""
    objs: List[str] = []
    if plan.get("embedding") == "generic":
        objs.append("wte")
    if plan.get("linear") == "generic":
        for li in range(int(n_layers)):
            objs.extend(f"h{li}.{s}" for s in tf.LINEAR_SITES)
    return objs


def server_vector_specs(card: dict, n_layers: int) -> List[Tuple[str, Tuple[int, ...]]]:
    """Names and public shapes of the LayerNorm parameters and biases shared at setup."""
    d, f = int(card["d_model"]), int(card["d_ff"])
    specs: List[Tuple[str, Tuple[int, ...]]] = []
    for li in range(int(n_layers)):
        for n in ("ln_1.g", "ln_1.b", "ln_2.g", "ln_2.b"):
            specs.append((f"h{li}.{n}", (d,)))
        specs.append((f"h{li}.qkv.b", (3 * d,)))
        specs.append((f"h{li}.out.b", (d,)))
        specs.append((f"h{li}.fc.b", (f,)))
        specs.append((f"h{li}.proj.b", (d,)))
    specs.append(("ln_f.g", (d,)))
    specs.append(("ln_f.b", (d,)))
    return specs


def _weight_for(W: Dict[str, np.ndarray], name: str) -> np.ndarray:
    if name in W:
        return W[name]
    if name + ".w" in W:
        return W[name + ".w"]
    raise MaterialError(f"MODEL_KEY: {name!r} missing from the weights")


def share_server_vectors(party: Party, specs: Sequence[Tuple[str, Tuple[int, ...]]],
                         W: Optional[Dict[str, np.ndarray]]) -> int:
    """P0 secret-shares the listed encoded vectors to P1 in ONE message; both store their shares.

    The shares land in the Party's vector cache (``Party.vector_share`` reads it
    before asking the dealer), so no server-private value ever reaches the
    dealer. Returns the number of ring elements shared.
    """
    total = int(sum(int(np.prod(s)) for _, s in specs))
    if total == 0:
        return 0
    if party.party == 0:
        if W is None:
            raise ContractViolation("MODEL_OWNER: P0 must hold the weights to share server vectors")
        flat = np.concatenate([party.enc(_weight_for(W, n)).reshape(-1) for n, _ in specs]).astype(np.uint64)
        share = party.share_input(flat, owner=0, shape=(total,))
    else:
        if W is not None:
            raise ContractViolation("MODEL_OWNER: the client must not hold plaintext weights")
        share = party.share_input(None, owner=0, shape=(total,))
    off = 0
    for name, shp in specs:
        n = int(np.prod(shp))
        party._vectors[name] = np.ascontiguousarray(share[off:off + n].reshape(tuple(shp)))
        off += n
    return total


# ---------------------------------------------------------------------------
# material planning (upper bounds of the party.py realizations)
# ---------------------------------------------------------------------------
def tournament_counts(n: int, k: int, with_index: bool) -> Tuple[int, int, int]:
    """(DReLU P=1, DReLU P=2, Beaver) gates of the k-ary tournament of ``party.Party._tournament`` over n items."""
    n, k = int(n), int(k)
    if n < 1:
        raise ContractViolation("TOURNAMENT_PARAMS: n must be >= 1")
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


def epoch_material_plan(plan: dict, card: dict, n_rows: int, T: int, n_layers: int,
                        contract: NumericContract) -> Dict[str, Any]:
    """Per-party material one epoch consumes under ``plan`` (upper bounds, by dealer type).

    ``n_rows`` rows are processed with a KV length of ``T`` after the epoch.
    Returns ``{beaver, drelu_p1, drelu_p2, trunc, matrix_triples: {(sa, sb): count},
    wmask_calls: {(mat_id, n_rows): count}}``. Constants mirror the gadgets of
    party.py: LayerNorm 3 R d + 14 R Beaver, R d faithful truncations and m R
    thresholds; exp 9 N Beaver and N comparisons; recip 12 R Beaver and K R
    comparisons; GELU 14 N Beaver and 2 N comparisons; tournaments exact by
    :func:`tournament_counts`.
    """
    d, f, H, V = int(card["d_model"]), int(card["d_ff"]), int(card["n_head"]), int(card["vocab"])
    dh = d // H
    n, T = int(n_rows), int(T)
    L = int(n_layers)
    beaver = p1 = p2 = trunc = 0
    mts: Dict[Tuple[tuple, tuple], int] = {}
    wcs: Dict[Tuple[str, int], int] = {}

    def mt(sa: tuple, sb: tuple, c: int) -> None:
        mts[(sa, sb)] = mts.get((sa, sb), 0) + int(c)

    def wc(mat: str, rows: int, c: int = 1) -> None:
        wcs[(mat, int(rows))] = wcs.get((mat, int(rows)), 0) + int(c)

    def lin(mat: str, d_in: int, d_out: int, rows: int) -> None:
        if plan["linear"] == "wmask":
            wc(mat, rows)
        else:
            mt((rows, d_in), (d_in, d_out), 1)

    def layernorm(rows: int) -> None:
        nonlocal beaver, p1, trunc
        beaver += 3 * rows * d + 14 * rows
        trunc += rows * d
        p1 += int(contract.rsqrt_m) * rows

    if plan["embedding"] == "wmask":
        wc("wte", n)
    else:
        mt((n, V), (V, d), 1)
    R = H * n
    for li in range(L):
        layernorm(n)
        lin(f"h{li}.qkv", d, 3 * d, n)
        mt((n, dh), (dh, T), H)
        a1, a2, ab = tournament_counts(T, int(plan["max_k"]), False)
        p1 += R * a1 + R * T + int(contract.recip_K) * R
        p2 += R * a2
        beaver += R * ab + 9 * R * T + 12 * R + R * T
        mt((n, T), (T, dh), H)
        lin(f"h{li}.out", d, d, n)
        layernorm(n)
        lin(f"h{li}.fc", d, f, n)
        p1 += 2 * n * f
        beaver += 14 * n * f
        lin(f"h{li}.proj", f, d, n)
    layernorm(1)
    wc("lm_head", 1)
    g1, g2, gb = tournament_counts(V, int(plan["argmax_k"]), True)
    p1 += g1
    p2 += g2
    beaver += gb
    # margins against rounding of the live-threshold counts and the public-mask pruning
    beaver = beaver + 64 + beaver // 50
    p1 = p1 + 16 + p1 // 50
    trunc = trunc + 8
    return {"beaver": int(beaver), "drelu_p1": int(p1), "drelu_p2": int(p2), "trunc": int(trunc),
            "matrix_triples": mts, "wmask_calls": wcs}


def material_requests(mplan: Dict[str, Any], prefix: str, epoch: int, contract: NumericContract,
                      gate_id: Optional[str] = None, attempt: int = 0) -> List[dict]:
    """PROVISION requests (deterministic ids, so both parties pair at the dealer) for one epoch.

    ``attempt`` distinguishes a retry after an aborted prepare phase from the
    first request set of the epoch (the dealer refuses a reused id with other
    parameters, so every attempt gets its own id space).
    """
    gate = gate_id if gate_id is not None else f"e{int(epoch)}"
    reqs: List[dict] = []

    def rid(kind: str, i: int) -> str:
        return f"{prefix}:e{int(epoch)}:a{int(attempt)}:{kind}:{i}"

    def chunks(total: int, cap: int) -> List[int]:
        out = []
        while total > 0:
            k = min(total, cap)
            out.append(k)
            total -= k
        return out

    for i, m in enumerate(chunks(int(mplan.get("beaver", 0)), MAX_BEAVER_PER_BUNDLE)):
        reqs.append({"request_id": rid("beaver", i), "material_type": "beaver", "params": {"shape": [m]},
                     "count": 1, "gate_id": gate})
    for i, m in enumerate(chunks(int(mplan.get("drelu_p1", 0)), MAX_GATES_PER_BUNDLE)):
        reqs.append({"request_id": rid("drelu1", i), "material_type": "drelu",
                     "params": {"M": m, "n_bits": int(contract.dcf_domain_bits), "with_product": False},
                     "count": 1, "gate_id": gate})
    for i, m in enumerate(chunks(int(mplan.get("drelu_p2", 0)), MAX_GATES_PER_BUNDLE)):
        reqs.append({"request_id": rid("drelu2", i), "material_type": "drelu",
                     "params": {"M": m, "n_bits": int(contract.dcf_domain_bits), "with_product": True},
                     "count": 1, "gate_id": gate})
    for i, m in enumerate(chunks(int(mplan.get("trunc", 0)), MAX_GATES_PER_BUNDLE)):
        reqs.append({"request_id": rid("trunc", i), "material_type": "trunc_faithful",
                     "params": {"M": m, "f": int(contract.frac_bits), "L": int(contract.logical_bits)},
                     "count": 1, "gate_id": gate})
    for i, ((sa, sb), c) in enumerate(sorted(mplan.get("matrix_triples", {}).items())):
        reqs.append({"request_id": rid("mt", i), "material_type": "matrix_triple",
                     "params": {"shape_a": [int(v) for v in sa], "shape_b": [int(v) for v in sb]},
                     "count": int(c), "gate_id": gate})
    for i, ((mat, rows), c) in enumerate(sorted(mplan.get("wmask_calls", {}).items())):
        reqs.append({"request_id": rid("wc", i), "material_type": "wmask_call",
                     "params": {"mat_id": str(mat), "n_rows": int(rows)}, "count": int(c), "gate_id": gate})
    return reqs


def material_plan_bytes(mplan: Dict[str, Any], contract: NumericContract) -> int:
    """Per-party bytes of a material plan (lowlevel per-element formulas)."""
    from ..lowlevel.drelu import drelu_bytes_per_element
    from ..lowlevel.trunc import trunc_bytes_per_element
    nb = int(contract.dcf_domain_bits)
    total = 24 * int(mplan.get("beaver", 0))
    total += drelu_bytes_per_element(nb, False) * int(mplan.get("drelu_p1", 0))
    total += drelu_bytes_per_element(nb, True) * int(mplan.get("drelu_p2", 0))
    total += trunc_bytes_per_element(int(contract.frac_bits)) * int(mplan.get("trunc", 0))
    for (sa, sb), c in mplan.get("matrix_triples", {}).items():
        total += 8 * c * (sa[0] * sa[1] + sb[0] * sb[1] + sa[0] * sb[1])
    return int(total)


# ---------------------------------------------------------------------------
# dealer wrapper: provisioning ahead of each epoch plus a symmetric stall path
# ---------------------------------------------------------------------------
class SessionDealer:
    """Party-side material access of one session: a ``DealerClient`` fed by the isolated dealer service.

    ``provision(requests)`` streams the requested bundles (each party receives
    its own half of the same pairs under the same request ids) through the
    optional refill pacer into the client and its ledger. Every ``take_*`` of
    the Party API is forwarded; on ``MaterialStall`` the missing amount is
    requested under a deterministic stall id (consumption is shape
    deterministic, so both parties stall at the same gate) and the take is
    retried once. Without a service a stall propagates.
    """

    STALL_BEAVER = 1 << 12
    STALL_GATES = 256

    def __init__(self, client: DealerClient, service: Any, prefix: str, pacer: Any = None,
                 shapes: Optional[Dict[str, Tuple[int, int]]] = None) -> None:
        self.client = client
        self.service = service
        self.prefix = str(prefix)
        self.pacer = pacer
        self.shapes: Dict[str, Tuple[int, int]] = dict(shapes or {})
        self.ledger = client.ledger
        self.party = client.party
        self.contract = client.contract
        self._lock = threading.RLock()
        self._stall_counter = 0
        self._masks: set = set()
        self.stalls = 0
        self.provision_s = 0.0
        self.provisioned_bytes = 0
        self.provisioned_bundles = 0
        self.requests_sent = 0

    # -- context / status ------------------------------------------------------------
    def set_context(self, gate_id: str, epoch: int) -> None:
        self.client.set_context(gate_id, epoch)

    @property
    def security_eligible(self) -> bool:
        return bool(self.client.security_eligible)

    @property
    def dealer_mode(self) -> str:
        if self.service is not None and isinstance(getattr(self.service, "binding", None), dict):
            mode = self.service.binding.get("dealer_mode")
            if mode:
                return str(mode)
        return MODE_ISOLATED if self.client.security_eligible else MODE_INSECURE

    def stock(self) -> Dict[str, Tuple[int, int]]:
        return self.client.stock()

    def discard(self, reason: str = "PLAN_SWITCH") -> int:
        return int(self.client.discard(None, reason))

    # -- provisioning ---------------------------------------------------------------------
    def provision(self, requests: Sequence[dict]) -> dict:
        """Request and ingest every bundle of ``requests``; returns ``{bundles, bytes, seconds}``."""
        if not requests:
            return {"bundles": 0, "bytes": 0, "seconds": 0.0}
        if self.service is None:
            raise MaterialError("DEALER_SERVICE_MISSING: no dealer service to request material from")
        t0 = time.perf_counter()
        nb = 0
        n = 0
        with self._lock:
            for r in requests:
                bundles = self.service.provision(str(r["request_id"]), str(r["material_type"]),
                                                dict(r.get("params") or {}), int(r.get("count", 1)),
                                                str(r.get("gate_id", "*")))
                self.requests_sent += 1
                for header, payload, tag in bundles:
                    if self.pacer is not None:
                        self.pacer.wait_for(len(payload))
                    self.client.ingest(header, payload, tag)
                    nb += len(payload)
                    n += 1
        el = time.perf_counter() - t0
        self.provision_s += el
        self.provisioned_bytes += nb
        self.provisioned_bundles += n
        return {"bundles": n, "bytes": nb, "seconds": el}

    def ensure_mask(self, mat_id: str, generation: int = 0) -> bool:
        """Request the per-matrix mask bundle of ``mat_id`` once per client; True when newly delivered."""
        mat_id = str(mat_id)
        with self._lock:
            if mat_id in self._masks:
                return False
            if mat_id not in self.shapes:
                raise MaterialError(f"WMASK_SHAPE_UNKNOWN: no public shape for object {mat_id!r}")
            d, m = self.shapes[mat_id]
            self.provision([{"request_id": f"{self.prefix}:mask:{mat_id}:{int(generation)}",
                             "material_type": "wmask_mask", "params": {"mat_id": mat_id, "shape": [int(d), int(m)]},
                             "count": 1, "gate_id": "install"}])
            self._masks.add(mat_id)
            return True

    def _stall(self, fn: Callable[[], Any], material_type: str, params: dict, what: str) -> Any:
        try:
            return fn()
        except MaterialStall:
            if self.service is None:
                raise
        with self._lock:
            self.stalls += 1
            self._stall_counter += 1
            rid = f"{self.prefix}:stall:{self._stall_counter}"
            self.provision([{"request_id": rid, "material_type": material_type, "params": params, "count": 1,
                             "gate_id": f"stall:{what}"}])
        return fn()

    # -- Party API -------------------------------------------------------------------------
    @staticmethod
    def _numel(shape: Any) -> int:
        shp = (shape,) if isinstance(shape, (int, np.integer)) else tuple(shape)
        return int(np.prod([int(s) for s in shp])) if shp else 1

    def take_beaver(self, shape: Any):
        n = self._numel(shape)
        return self._stall(lambda: self.client.take_beaver(shape), "beaver",
                           {"shape": [max(n, self.STALL_BEAVER)]}, "beaver")

    def take_kary_select(self, shape: Any):
        n = self._numel(shape)
        return self._stall(lambda: self.client.take_kary_select(shape), "kary_select",
                           {"shape": [max(n, self.STALL_BEAVER)]}, "kary_select")

    def take_matrix_triple(self, shape_a: Any, shape_b: Any):
        sa = [int(v) for v in shape_a]
        sb = [int(v) for v in shape_b]
        return self._stall(lambda: self.client.take_matrix_triple(shape_a, shape_b), "matrix_triple",
                           {"shape_a": sa, "shape_b": sb}, "matrix_triple")

    def take_wmask_call(self, mat_id: str, n_rows: int):
        self.ensure_mask(mat_id)
        return self._stall(lambda: self.client.take_wmask_call(mat_id, n_rows), "wmask_call",
                           {"mat_id": str(mat_id), "n_rows": int(n_rows)}, "wmask_call")

    def wmask_mask_share(self, mat_id: str, for_version: Optional[dict] = None) -> np.ndarray:
        self.ensure_mask(mat_id)
        return self.client.wmask_mask_share(mat_id, for_version=for_version)

    def wmask_mask_version(self, mat_id: str) -> dict:
        return self.client.wmask_mask_version(mat_id)

    def take_drelu(self, count: int, with_product: bool = True, n_bits: Optional[int] = None):
        n = int(self.contract.dcf_domain_bits if n_bits is None else n_bits)
        return self._stall(lambda: self.client.take_drelu(count, with_product=with_product, n_bits=n_bits), "drelu",
                           {"M": max(int(count), self.STALL_GATES), "n_bits": n, "with_product": bool(with_product)},
                           "drelu")

    def take_trunc(self, count: int, f: Optional[int] = None, L: Optional[int] = None):
        ff = int(self.contract.frac_bits if f is None else f)
        LL = int(self.contract.logical_bits if L is None else L)
        return self._stall(lambda: self.client.take_trunc(count, f, L), "trunc_faithful",
                           {"M": max(int(count), self.STALL_GATES), "f": ff, "L": LL}, "trunc_faithful")

    def take_r2f(self, count: int, t: int):
        return self._stall(lambda: self.client.take_r2f(count, t), "ring_to_field",
                           {"M": max(int(count), 64), "t": int(t)}, "ring_to_field")

    def take_vector_share(self, name: str) -> np.ndarray:
        return self.client.take_vector_share(name)


# ---------------------------------------------------------------------------
# warm stacks: what one party keeps between sessions of one deployment
# ---------------------------------------------------------------------------
class WarmStack:
    """Reusable state of ONE party for one ``(model_version, client_key_domain, n_layers)``.

    Created by the platform (model server, sidecar, launcher) with a ledger
    bound to ``stack_id`` and one authenticated dealer connection; filled by
    :meth:`SessionRunner.finish` after a session that ended at a declared
    boundary (completed or cancelled). ``usable`` is cleared by
    :meth:`invalidate` when a session on the stack failed in the middle of a
    gate; such a stack is never offered again (the platform builds a new one).
    The digest covers public data only (ids, versions, names, the plan).
    """

    def __init__(self, party: int, model_version: str, client_key_domain: str, n_layers: int, ledger: MaterialLedger,
                 client: DealerClient, service: Any, stack_id: Optional[str] = None) -> None:
        self.party = int(party)
        self.model_version = str(model_version)
        self.client_key_domain = str(client_key_domain)
        self.n_layers = int(n_layers)
        self.ledger = ledger
        self.client = client
        self.service = service
        self.stack_id = str(stack_id) if stack_id else "stack-" + secrets.token_hex(8)
        if str(ledger.binding.get("session_id")) != self.stack_id:
            raise AdmissionError("WARM_STACK_BINDING: the stack ledger must be bound to the stack id")
        self.masks: set = set()
        self.wmask_b: Dict[str, np.ndarray] = {}
        self.installed: Dict[str, dict] = {}
        self.vectors: Dict[str, Optional[np.ndarray]] = {}
        self.generic_shared: set = set()
        self.plan: Optional[Dict[str, Any]] = None
        self.sessions = 0
        self.usable = True

    @property
    def key(self) -> Tuple[str, str, int]:
        return (self.model_version, self.client_key_domain, self.n_layers)

    @property
    def warm(self) -> bool:
        """True once a session left reusable state behind."""
        return bool(self.usable and self.sessions > 0)

    def mask_ids(self) -> Dict[str, str]:
        held = getattr(self.client, "_wmask_mask", {}) or {}
        return {str(m): str(held[m][0]) for m in sorted(self.masks) if m in held}

    def digest(self) -> str:
        """Public digest both parties must agree on before a warm start ("" for a stack without state)."""
        if not self.warm:
            return ""
        return _digest({"model_version": self.model_version, "client_key_domain": self.client_key_domain,
                        "n_layers": self.n_layers, "sessions": int(self.sessions),
                        "installed": {k: dict(v["version"]) for k, v in sorted(self.installed.items())},
                        "mask_ids": self.mask_ids(),
                        "vectors": sorted(k for k, v in self.vectors.items() if v is not None),
                        "generic_shared": sorted(self.generic_shared), "plan": dict(self.plan or {})})

    def invalidate(self, reason: str = "FAILURE") -> None:
        """A session failed inside a gate: burn what is uncertain and never offer this stack again."""
        self.usable = False
        try:
            self.ledger.burn_reserved(reason)
            self.client.discard(None, reason)
        except Exception:
            pass

    def close(self) -> None:
        for obj in (self.service, self.ledger):
            try:
                if obj is not None:
                    obj.close()
            except Exception:
                pass


def new_warm_stack(party: int, model_version: str, client_key_domain: str, n_layers: int, journal_path: str,
                   setup_id: str, contract: NumericContract, hmac_key: bytes, service: Any,
                   profile_id: str = PROFILE_ID, allow_insecure: Optional[bool] = None) -> WarmStack:
    """Open the stack ledger (bound to a fresh stack id) and the ``DealerClient`` of one party."""
    stack_id = "stack-" + secrets.token_hex(8)
    binding = {"session_id": stack_id, "setup_id": str(setup_id), "model_version": str(model_version),
               "numeric_contract_hash": contract.contract_hash(), "profile_id": str(profile_id)}
    ledger = MaterialLedger(journal_path, binding)
    client = DealerClient(int(party), ledger, hmac_key, contract, allow_insecure=allow_insecure)
    return WarmStack(party, model_version, client_key_domain, n_layers, ledger, client, service, stack_id)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _digest(obj: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(obj)).hexdigest()


def _bucket(T: int, bucket: int) -> int:
    b = max(1, int(bucket))
    return int(math.ceil(int(T) / b) * b)


def horizon_schedule(shapes_of: Callable[[int, int], dict], t: int, n_epochs: int, prompt_len: int,
                     n_rows: int, T: int) -> List[dict]:
    """Public shapes of the remaining epochs ``t .. n_epochs - 1`` (paper, Section 4.2, objective).

    Epoch ``t`` processes ``n_rows`` rows with KV length ``T`` (the prefill, a
    recovery prefill or one decode step); every later epoch ``u`` is one
    decode step with KV length ``prompt_len + u``. ``shapes_of(rows, kv)``
    gives the cost-model shapes on the session's length bucket.
    """
    return [shapes_of(n_rows, T)] + [shapes_of(1, int(prompt_len) + u) for u in range(int(t) + 1, int(n_epochs))]


def _check_equal(a: Any, b: Any, what: str, code: str) -> None:
    if a != b:
        raise AdmissionError(f"{code}: {what} differs between the parties ({a!r} vs {b!r})")


def first_mismatch_step(generated: Sequence[int], reference: Sequence[int]) -> Optional[int]:
    """First index at which two token sequences differ (the shorter one's length when one is a prefix of the
    other), None when they are equal."""
    g = [int(t) for t in generated]
    r = [int(t) for t in reference]
    for i, (a, b) in enumerate(zip(g, r)):
        if a != b:
            return i
    return None if len(g) == len(r) else min(len(g), len(r))


def _atomic_write_json(path: str, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(obj, sort_keys=True))
    os.replace(str(tmp), str(p))


def _append_lines(path: Any, lines: List[str]) -> None:
    """Cross-process safe append of whole lines (``experiments.schema.locked_append_lines`` when present)."""
    try:
        from ..experiments.schema import locked_append_lines   # type: ignore
    except Exception:   # pragma: no cover (schema module absent)
        locked_append_lines = None
    if locked_append_lines is not None:
        locked_append_lines(path, lines)
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8", newline="\n") as fh:
        fh.write("".join(l if l.endswith("\n") else l + "\n" for l in lines))
        fh.flush()
        os.fsync(fh.fileno())


def validate_record(record: dict) -> None:
    """Shape check of an evidence record (used when ``experiments.schema`` is absent)."""
    if not isinstance(record, dict):
        raise ContractViolation("RECORD_SHAPE: record must be a dict")
    missing = [k for k in RECORD_FIELDS if k not in record]
    if missing:
        raise ContractViolation(f"RECORD_SHAPE: missing fields {missing}")
    if record["status"] not in ("MEASURED_NATIVE", "EMULATED_NUMERIC", "MODELED", "EXTERNAL_REPORTED",
                                "DIAGNOSTIC", "HISTORICAL"):
        raise ContractViolation(f"RECORD_SHAPE: unknown status {record['status']!r}")
    if not isinstance(record["security_eligible"], bool):
        raise ContractViolation("RECORD_SHAPE: security_eligible must be a bool")
    if record["dealer_mode"] not in (MODE_ISOLATED, MODE_INSECURE):
        raise ContractViolation(f"RECORD_SHAPE: unknown dealer_mode {record['dealer_mode']!r}")
    if record["security_eligible"] and (record["dealer_mode"] != MODE_ISOLATED or int(record["diagnostic_reveals"]) != 0):
        raise ContractViolation("RECORD_SHAPE: an eligible record needs the isolated dealer and no diagnostic reveals")
    for k in ("T_cold_s", "T_warm_s", "T_online_s"):
        v = record[k]
        if v is None and k == "T_cold_s":
            continue        # a warm request measures no cold time
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or v < 0:
            raise ContractViolation(f"RECORD_SHAPE: {k} must be a finite non-negative number")
    for k in ("bytes", "rounds", "material_generated", "material_transferred", "material_consumed",
              "material_burned", "initializations", "post_init_switches", "rejected_candidates", "aborts",
              "diagnostic_reveals", "repeat", "tokens"):
        if not isinstance(record[k], int) or record[k] < 0:
            raise ContractViolation(f"RECORD_SHAPE: {k} must be a non-negative int")
    if record["clock_basis"] != CLOCK_BASIS:
        raise ContractViolation(f"RECORD_SHAPE: clock_basis must be {CLOCK_BASIS!r}")
    if record["output_match"] not in (True, False, None):
        raise ContractViolation("RECORD_SHAPE: output_match must be a bool or None")


# ---------------------------------------------------------------------------
# the session runner
# ---------------------------------------------------------------------------
class SessionRunner:
    """One party's view of a session (see the module docstring for the epoch protocol)."""

    def __init__(self, party: int, cfg: SessionConfig, channel: Any, dealer: Any, ledger: MaterialLedger,
                 costmodel: Any, trace: Any, out_path: str, *, dealer_service: Any = None,
                 cancel_event: Optional[threading.Event] = None, origin_wall_s: Optional[float] = None,
                 weights: Optional[Dict[str, np.ndarray]] = None, meta: Optional[dict] = None,
                 hooks: Optional[Dict[str, Callable[..., Any]]] = None, start_cpu_load: bool = True,
                 warm: Optional[WarmStack] = None) -> None:
        from ..platform import shaper as _shaper   # platform layer, imported here to keep the module import light

        cfg.validate()
        self.party_id = int(party)
        if self.party_id not in (0, 1):
            raise ContractViolation(f"PARTY_ID: party must be 0 or 1, got {party}")
        self.cfg = cfg
        self.ch = channel
        self.ledger = ledger
        self.out_path = str(out_path)
        self.hooks = dict(hooks or {})
        self.cancel_event = cancel_event if cancel_event is not None else threading.Event()
        self.contract: NumericContract = cfg.contract
        self.contract_hash = self.contract.contract_hash()
        self.n_epochs = int(cfg.n_generate)
        self.prompt_len = cfg.effective_prompt_len()
        self.session_id = cfg.session_id or f"sess-{uuid.uuid4().hex[:12]}"
        self.run_id = cfg.run_id or f"{self.session_id}-r{int(cfg.repeat)}"
        self.generation = 0
        self.warm = warm
        if warm is not None:
            if warm.party != self.party_id or not warm.usable:
                raise AdmissionError("WARM_STACK_UNUSABLE: the stack belongs to another party or was invalidated")
            if cfg.resume:
                raise AdmissionError("WARM_STACK_RESUME: a cold resume cannot start from a warm stack")
            if ledger is not warm.ledger:
                raise AdmissionError("WARM_STACK_BINDING: the session must use the stack ledger")
        self.warm_start = bool(warm is not None and warm.warm)
        self.warm_digest = warm.digest() if warm is not None else ""
        # cost model shared by both parties (identical coefficients)
        if costmodel is None:
            costmodel = FittedCostModel.from_json(cfg.costmodel_json) if cfg.costmodel_json else default_model()
        self.costmodel = costmodel
        # exogenous trace
        self.trace = trace
        self.origin_wall_s = float(origin_wall_s) if origin_wall_s is not None else time.time()
        self.shaper = _shaper.TraceShaper(trace, self.origin_wall_s) if trace is not None else None
        self.trace_hash = self.shaper.trace_hash if self.shaper is not None else "none"
        self.trace_id = self.shaper.trace_id if self.shaper is not None else str(cfg.trace_id)
        if self.shaper is not None and hasattr(channel, "set_shaper"):
            channel.set_shaper(self.shaper.link if self.shaper.shaped else None)
        self.pacer = _shaper.MaterialRefillPacer(self.shaper)
        self._name_transport_gate(channel)
        self.cpu_load = None
        if (self.shaper is not None and start_cpu_load and cfg.cpu_load_party is not None
                and int(cfg.cpu_load_party) == self.party_id and self.shaper.has_cpu_load):
            self.cpu_load = _shaper.CpuContentionLoad(self.shaper)
        # weights (P0) and the public model card
        self.W: Optional[Dict[str, np.ndarray]] = None
        self.meta: Optional[dict] = None
        self.card: Optional[dict] = None
        if self.party_id == 0:
            if weights is None or meta is None:
                weights, meta = load_model(cfg.model, cfg.n_layers)
            self.W, self.meta = weights, dict(meta)
            self.card = model_card(self.meta, cfg.n_layers)
            self.model_version = cfg.model_version or model_version_of(self.meta)
        else:
            if weights is not None:
                raise ContractViolation("MODEL_OWNER: the client must not hold plaintext weights")
            if meta is not None:
                self.card = model_card(dict(meta), cfg.n_layers)
            self.model_version = cfg.model_version or str(ledger.binding.get("model_version", ""))
        if not self.model_version:
            raise ContractViolation("SESSION_PARAMS: model_version is required on the client")
        self.binding = {"session_id": self.session_id, "setup_id": str(cfg.setup_id), "model_version": self.model_version,
                        "numeric_contract_hash": self.contract_hash, "profile_id": str(cfg.profile_id)}
        if warm is not None and (warm.model_version != self.model_version
                                 or warm.client_key_domain != str(cfg.client_key_domain)
                                 or warm.n_layers != int(cfg.n_layers)):
            raise AdmissionError("OBJECT_VERSION_MISMATCH: the warm stack was built for another (model_version, "
                                 "client_key_domain, n_layers)")
        # the ledger of a warm stack outlives its sessions and is bound to the stack id instead of a session id
        ledger_binding = dict(self.binding, session_id=warm.stack_id) if warm is not None else self.binding
        for k in ("session_id", "setup_id", "model_version", "numeric_contract_hash", "profile_id"):
            if str(ledger.binding.get(k)) != ledger_binding[k]:
                raise AdmissionError(
                    f"OBJECT_VERSION_MISMATCH: ledger binding {k}={ledger.binding.get(k)!r} differs from the session "
                    f"{ledger_binding[k]!r}")
        # dealer access
        self.object_shapes: Dict[str, Tuple[int, int]] = object_shapes(self.card, cfg.n_layers) if self.card else {}
        if isinstance(dealer, SessionDealer):
            self.dealer = dealer
            if dealer_service is not None:
                self.dealer.service = dealer_service
            self.dealer.pacer = self.pacer
            self.dealer.shapes.update(self.object_shapes)
        else:
            self.dealer = SessionDealer(dealer, dealer_service, f"{self.session_id}:g{self.generation}", self.pacer,
                                        self.object_shapes)
        if warm is not None:
            if self.dealer.client is not warm.client:
                raise AdmissionError("WARM_STACK_BINDING: the session must use the stack's dealer client")
            self.dealer._masks = set(warm.masks) if self.warm_start else set()
        # per-session baselines: a stack, a reused channel or a long-lived dealer carry earlier sessions' totals
        self._base_ingested = int(self.dealer.client.ingested_bytes)
        self._base_consumed = int(self.dealer.client.consumed_bytes)
        self._base_burned = int(ledger.bytes_in_state("BURNED"))
        self._base_generated: Optional[int] = None
        self._ch_base: Any = None
        self._ch_carry_bytes = 0
        self._ch_carry_rounds = 0
        self._warm_saved = False
        # public state
        self.state_path = cfg.state_path or (str(ledger.journal_path) + STATE_SUFFIX)
        self.epochs_done = 0
        self.generated: List[int] = []
        self.plan: Dict[str, Any] = dict(cfg.initial_plan or (warm.plan if self.warm_start and warm.plan else None)
                                         or COLD_PLAN)
        self.kv_lost = False
        self.cancelled = False
        self.cancel_epoch: Optional[int] = None
        self.completed = False
        self.resumed = False
        self.aborts = 0
        self.rejected_candidates = 0
        self.initializations = 0
        self.post_init_switches = 0
        self.switches = 0
        self.wasted_bytes = 0
        self.burned_on_cancel = 0
        self.stall_requests = 0
        self.epochs: List[dict] = []
        self.decisions: List[dict] = []
        self.setup_s = 0.0
        self.install_s_total = 0.0
        self.generic_share_s = 0.0
        self.exec_s_total = 0.0
        self.t_start: Optional[float] = None
        self.t_end: Optional[float] = None
        self.ttft_s: Optional[float] = None
        self.telemetry_seq = 0
        self.predelivered: Optional[Tuple[str, int]] = None   # (plan key, epoch) of prefetched material
        self.initial_state_hash = ""
        self.workload_scope: Optional[str] = None
        self.policy_id = str(cfg.policy)
        self.party: Optional[Party] = None
        self.model: Optional[tf.SecureGPT2] = None
        self.policy: Optional[ctrl.Policy] = None
        self.verifier: Optional[telemetry.TelemetryVerifier] = None
        self.telemetry_key: Optional[bytes] = None
        self.record: Optional[dict] = None
        self.detail_path = ""
        self.generic_shared: set = set()
        self._stats_mark = 0
        self._lock = threading.RLock()
        if cfg.resume:
            self._load_public_state()

    # -- public state ----------------------------------------------------------------------
    def _public_state(self) -> dict:
        st = {"session_id": self.session_id, "setup_id": str(self.cfg.setup_id), "model_version": self.model_version,
              "numeric_contract_hash": self.contract_hash, "profile_id": str(self.cfg.profile_id),
              "epochs_done": int(self.epochs_done), "plan": dict(self.plan),
              "installed": self.party.installed_objects() if self.party is not None else {},
              "generated_count": len(self.generated), "prompt_len": self.prompt_len,
              "n_generate": self.n_epochs, "generation": int(self.generation), "policy": self.policy_id,
              "cancelled": bool(self.cancelled), "completed": bool(self.completed)}
        if self.party_id == 1:
            st["generated_tokens"] = [int(t) for t in self.generated]
        return st

    def _save_public_state(self) -> None:
        _atomic_write_json(self.state_path, self._public_state())

    def _load_public_state(self) -> None:
        """Cold resume: reload the public state, verify the binding, recover the ledger."""
        if not os.path.isfile(self.state_path):
            raise AdmissionError(f"SESSION_STATE_MISSING: no public state at {self.state_path!r} to resume from")
        st = json.loads(Path(self.state_path).read_text(encoding="utf-8"))
        if str(st.get("session_id")) != self.session_id:
            raise AdmissionError(f"SESSION_BINDING_MISMATCH: state session {st.get('session_id')!r} != {self.session_id!r}")
        if str(st.get("model_version")) != self.model_version:
            raise AdmissionError(
                f"OBJECT_VERSION_MISMATCH: resumed session was bound to model_version {st.get('model_version')!r}, "
                f"this party declares {self.model_version!r}")
        if str(st.get("numeric_contract_hash")) != self.contract_hash:
            raise AdmissionError("CONTRACT_MISMATCH: resumed session used another numeric contract")
        if int(st.get("n_generate", self.n_epochs)) != self.n_epochs or int(st.get("prompt_len", self.prompt_len)) != self.prompt_len:
            raise AdmissionError("SESSION_BINDING_MISMATCH: resumed session declares other public shapes")
        self.epochs_done = int(st.get("epochs_done", 0))
        self.plan = dict(st.get("plan", self.plan))
        self.generation = int(st.get("generation", 0)) + 1
        if self.party_id == 1:
            self.generated = [int(t) for t in st.get("generated_tokens", [])]
        self.resumed = True
        self.kv_lost = self.epochs_done > 0
        if self.ledger.needs_recovery:
            self.ledger.recover()
        # the material bytes of a previous process are gone: their ids can never be used again
        leftover = self.ledger.ids_in_state("AVAILABLE")
        if leftover:
            self.ledger.burn(leftover, "RESUME_COLD")
        self.dealer.prefix = f"{self.session_id}:g{self.generation}"

    # -- setup ---------------------------------------------------------------------------------
    def _make_party(self) -> None:
        released = len(self.generated)
        remaining = max(0, self.n_epochs - released)
        pol = ReleasePolicy(1, "token", remaining)
        if self.cfg.party_rng_from_os:
            # share_input sends value minus mask: the mask must be unpredictable to the peer (OS entropy)
            rng = SecureRandom.from_os()
        else:
            rng = np.random.default_rng(int(self.cfg.seed) * 7919 + self.party_id + 1)   # diagnostic: public seed
        self.party = Party(self.party_id, self.ch, self.dealer, self.contract, pol, rng,
                           model_version=self.model_version, client_key_domain=str(self.cfg.client_key_domain))
        if self.warm_start:
            w = self.warm
            self.party._wmask_b.update(w.wmask_b)
            self.party._installed.update({k: {"E": v["E"], "version": dict(v["version"])}
                                          for k, v in w.installed.items()})
            self.party._vectors.update(w.vectors)
        self.model = tf.SecureGPT2(self.party, self.W, self.card, int(self.cfg.n_layers), dict(self.plan),
                                   ln_prescale_bits=int(self.cfg.ln_prescale_bits))
        self.generic_shared = set(self.warm.generic_shared) if self.warm_start else set()
        self._stats_mark = 0

    def _make_policy(self, shapes: dict) -> ctrl.Policy:
        name = str(self.cfg.policy)
        kw = dict(self.cfg.controller_kw or {})
        if name in ("B0_fixed", "B0_fixed_best_hindsight"):
            return ctrl.make_policy(name, plan=dict(self.cfg.fixed_plan or DEFAULT_FIXED_PLAN))
        if name == "B5_oracle":
            envs: List[Env] = []
            if self.shaper is not None:
                for _t, bw, delay, cpu, _r in self.shaper.points:
                    envs.append(Env(bw if bw is not None else 0.0, float(delay), float(cpu)))
            else:
                envs.append(Env(0.0, 0.0, 1.0))
            return ctrl.make_policy(name, trace=envs, costmodel=self.costmodel, shapes=shapes)
        return ctrl.make_policy(name, **kw)

    def _hello(self) -> dict:
        return {"session_id": self.session_id, "setup_id": str(self.cfg.setup_id), "model_version": self.model_version,
                "numeric_contract_hash": self.contract_hash, "profile_id": str(self.cfg.profile_id),
                "n_layers": int(self.cfg.n_layers), "prompt_len": int(self.prompt_len),
                "n_generate": int(self.n_epochs), "length_bucket": int(self.cfg.length_bucket),
                "policy": self.policy_id, "trace_hash": self.trace_hash, "resume": bool(self.resumed),
                "resume_epoch": int(self.epochs_done), "nonce": secrets.token_hex(16),
                "model_card": self.card if self.party_id == 0 else None, "party": self.party_id,
                "kv_lost": bool(self.kv_lost), "warm_digest": str(self.warm_digest),
                "client_key_domain": str(self.cfg.client_key_domain)}

    def handshake(self) -> dict:
        """Exchange and verify the public session binding; derive the telemetry key; share server vectors."""
        if self._base_generated is None:
            self._base_generated = self._dealer_generated()
        if self._ch_base is None:
            self._ch_base = self.ch.stats.copy()
        t0 = time.perf_counter()
        if self.t_start is None:
            self.t_start = t0
        mine = self._hello()
        theirs = self.ch.exchange(mine)
        if not isinstance(theirs, dict) or int(theirs.get("party", -1)) != 1 - self.party_id:
            raise ChannelError("SESSION_HANDSHAKE: malformed hello from the peer")
        _check_equal(theirs.get("model_version"), mine["model_version"], "model_version", "OBJECT_VERSION_MISMATCH")
        _check_equal(theirs.get("numeric_contract_hash"), mine["numeric_contract_hash"], "numeric_contract_hash",
                     "CONTRACT_MISMATCH")
        _check_equal(theirs.get("profile_id"), mine["profile_id"], "profile_id", "PROFILE_MISMATCH")
        for k in _HANDSHAKE_KEYS:
            if k in ("model_version", "numeric_contract_hash", "profile_id"):
                continue
            _check_equal(theirs.get(k), mine[k], k, "SESSION_BINDING_MISMATCH")
        if self.party is None:
            # a warm start is admitted only when both parties hold the same stack state (public digest)
            if str(theirs.get("warm_digest", "")) != mine["warm_digest"]:
                raise AdmissionError("SESSION_BINDING_MISMATCH: the parties hold different warm stack states")
            if str(theirs.get("client_key_domain", mine["client_key_domain"])) != mine["client_key_domain"]:
                raise AdmissionError("SESSION_BINDING_MISMATCH: client_key_domain differs between the parties")
        if self.party_id == 1:
            card = theirs.get("model_card")
            if not isinstance(card, dict):
                raise ChannelError("SESSION_HANDSHAKE: the server sent no model card")
            if self.card is not None and {k: self.card[k] for k in ("d_model", "n_head", "d_ff", "vocab")} != \
                    {k: card.get(k) for k in ("d_model", "n_head", "d_ff", "vocab")}:
                raise AdmissionError("SESSION_BINDING_MISMATCH: the server's model card differs from the client's")
            self.card = dict(card)
            self.object_shapes = object_shapes(self.card, self.cfg.n_layers)
            self.dealer.shapes.update(self.object_shapes)
        # the record label is fixed here, before any material is consumed (fails closed on an unsupported prefix)
        self.workload_scope = workload_scope_for(self.cfg.n_layers, self.card.get("n_layers_total", self.cfg.n_layers),
                                                 self.cfg.workload_scope)
        kv_lost = bool(mine["kv_lost"]) or bool(theirs.get("kv_lost", False))
        self.kv_lost = kv_lost
        nonces = sorted([str(mine["nonce"]), str(theirs.get("nonce", ""))]) if self.party_id == 0 else \
            sorted([str(theirs.get("nonce", "")), str(mine["nonce"])])
        self.telemetry_key = hashlib.sha256(("opencrab2-telemetry-v1|" + self.session_id + "|" + "|".join(nonces))
                                            .encode("utf-8")).digest()
        self.verifier = telemetry.TelemetryVerifier(self.telemetry_key, first_seq=self.telemetry_seq,
                                                    max_bandwidth_bps=1e12, max_rtt_s=60.0)
        if self.party is None:
            self._make_party()
            specs = [sp for sp in server_vector_specs(self.card, self.cfg.n_layers)
                     if self.party._vectors.get(sp[0]) is None]
            share_server_vectors(self.party, specs, self.W)
        else:
            self.party.ch = self.ch
        self.setup_s += time.perf_counter() - t0
        if self.initial_state_hash == "":
            self.initial_state_hash = _digest({"plan": self.plan, "installed": self.party.installed_objects(),
                                               "epoch": self.epochs_done, "binding": self.binding,
                                               "prompt_len": self.prompt_len, "n_generate": self.n_epochs})
        self._emit("status", {"event": "handshake", "session_id": self.session_id, "resume": self.resumed,
                              "epoch": self.epochs_done})
        return {"session_id": self.session_id, "binding": dict(self.binding), "model_card": dict(self.card),
                "resume": self.resumed, "epoch": self.epochs_done}

    def _name_transport_gate(self, channel: Any) -> None:
        """A transport that logs its frames in the ledger (the WebSocket adapter) gets one gate namespace per
        connection generation, so a reconnection never collides with the (gate_id, seq) entries of the previous one."""
        if channel is not None and hasattr(channel, "gate_id"):
            try:
                tag = f"{self.session_id}-" if getattr(self, "warm", None) is not None else ""
                channel.gate_id = f"ws-{tag}g{int(self.generation)}"
            except Exception:
                pass

    def attach_channel(self, channel: Any) -> None:
        """Warm resume: continue the same runner over a new channel."""
        if self._ch_base is not None and self.ch is not None and channel is not self.ch:
            d = self.ch.stats.diff(self._ch_base)
            self._ch_carry_bytes += int(d["bytes_sent"] + d["bytes_recv"])
            self._ch_carry_rounds += int(d["rounds"])
            self._ch_base = channel.stats.copy()
        self.ch = channel
        if self.shaper is not None and hasattr(channel, "set_shaper"):
            channel.set_shaper(self.shaper.link if self.shaper.shaped else None)
        if self.party is not None:
            self.party.ch = channel
        self.resumed = True
        self.cancelled = False
        self.cancel_event.clear()
        self.predelivered = None
        self.generation += 1
        self.dealer.prefix = f"{self.session_id}:g{self.generation}"
        self._name_transport_gate(channel)

    def cancel(self) -> None:
        self.cancel_event.set()

    # -- epoch helpers ----------------------------------------------------------------------------
    def _emit(self, kind: str, payload: Any) -> None:
        fn = self.hooks.get(kind)
        if fn is not None:
            try:
                fn(payload)
            except Exception:   # a hook must never break the protocol
                pass

    def _rows_and_T(self, t: int) -> Tuple[int, int, str]:
        """(rows processed, KV length after the epoch, gate kind) of epoch ``t``."""
        if t == 0:
            return self.prompt_len, self.prompt_len, "prefill"
        if self.kv_lost:
            n = self.prompt_len + len(self.generated) if self.party_id == 1 else self.prompt_len + t
            return n, n, "recovery_prefill"
        return 1, self.prompt_len + t, "decode"

    def _shapes(self, n_rows: int, T: int) -> dict:
        return cands.gpt2_shapes(int(self.cfg.n_layers), n=int(n_rows), T=_bucket(T, self.cfg.length_bucket),
                                 d_model=int(self.card["d_model"]), d_ff=int(self.card["d_ff"]),
                                 n_head=int(self.card["n_head"]), vocab=int(self.card["vocab"]))

    def _stock_for_state(self) -> Dict[str, Tuple[int, int]]:
        """Material stock in the cost-model vocabulary plus the declared refill allowance (S5)."""
        refill = self.shaper.refill_bps_at(time.time()) if self.shaper is not None else None
        raw = self.dealer.stock()
        bv = raw.get("beaver", (0, 0))
        ks = raw.get("kary_select", (0, 0))
        wc = raw.get("wmask_call", (0, 0))
        st: Dict[str, Tuple[int, int]] = {
            "beaver": (bv[0] + ks[0], bv[1] + ks[1]), "kary_select": (bv[0] + ks[0], bv[1] + ks[1]),
            "matrix_triple": tuple(raw.get("matrix_triple", (0, 0))), "wmask_mask": (wc[0], wc[1]),
            "drelu": tuple(raw.get("drelu", (0, 0))), "trunc_faithful": tuple(raw.get("trunc_faithful", (0, 0))),
            "ring_to_field": tuple(raw.get("ring_to_field", (0, 0))),
        }
        if refill is None:
            return {}
        allowance = int(refill / 8.0 * float(self.cfg.material_wait_budget_s))
        return {k: (int(c), int(b) + allowance) for k, (c, b) in st.items()}

    def _session_state(self, t: int, n_rows: int, T: int) -> cands.SessionState:
        return cands.SessionState(incumbent=dict(self.plan), installed=self.party.installed_objects(),
                                  stock=self._stock_for_state(), kv_bucket=_bucket(T, self.cfg.length_bucket),
                                  remaining_epochs=self.n_epochs - t, session_id=self.session_id,
                                  model_version=self.model_version, contract_hash=self.contract_hash,
                                  client_key_domain=str(self.cfg.client_key_domain))

    def _total_material_check(self, cl: List[cands.Candidate], state: cands.SessionState) -> None:
        """Across types, the sum of a candidate's need must fit the summed stock plus one allowance."""
        if not state.stock:
            return
        refill = self.shaper.refill_bps_at(time.time()) if self.shaper is not None else None
        allowance = int(refill / 8.0 * float(self.cfg.material_wait_budget_s)) if refill else 0
        raw_total = sum(int(b) for _c, b in self.dealer.stock().values())
        for c in cl:
            if c.feasible and c.material_bytes_per_epoch > raw_total + allowance:
                c.feasible = False
                c.reason = (f"INSUFFICIENT_MATERIAL: epoch needs {c.material_bytes_per_epoch} B, stock {raw_total} B "
                            f"plus refill allowance {allowance} B")

    def _plan_digest(self, plan: dict, t: int, catalog: dict) -> str:
        return _digest({"plan": plan, "epoch": int(t), "binding": self.binding,
                        "installed": self.party.installed_objects(),
                        "target_objects": cands.plan_objects(plan, catalog),
                        "object_version": {"model_version": self.model_version,
                                           "client_key_domain": str(self.cfg.client_key_domain)}})

    def _prepare(self, plan: dict, t: int, n_rows: int, T: int, catalog: dict,
                 attempt: int = 0) -> Tuple[bool, dict]:
        """Phase B: mask bundles of the plan's objects, material of the epoch, generic vectors.

        The per-matrix masks come first because the dealer derives every
        ``wmask_call`` bundle from the mask it holds for that matrix.
        """
        info = {"material_bytes": 0, "material_s": 0.0, "masks": [], "generic_shared": [], "wasted_bytes": 0,
                "prefetched": False, "error": ""}
        key = cands.plan_key(plan)
        try:
            for obj in cands.plan_objects(plan, catalog):
                if self.dealer.ensure_mask(obj, self.generation):
                    info["masks"].append(obj)
            if self.predelivered == (key, t):
                info["prefetched"] = True
            else:
                if self.predelivered is not None and self.predelivered[1] == t:
                    w = self.dealer.discard("PLAN_SWITCH")
                    info["wasted_bytes"] = w
                    self.wasted_bytes += w
                mplan = epoch_material_plan(plan, self.card, n_rows, T, int(self.cfg.n_layers), self.contract)
                res = self.dealer.provision(material_requests(mplan, self.dealer.prefix, t, self.contract,
                                                              attempt=attempt))
                info["material_bytes"] = int(res["bytes"])
                info["material_s"] = float(res["seconds"])
            self.predelivered = None
            # generic realizations read the full matrices as shares (dealer-free, one message per group)
            need = [o for o in generic_objects(plan, int(self.cfg.n_layers)) if o not in self.generic_shared]
            if need:
                t0 = time.perf_counter()
                specs = [(o, tuple(self.object_shapes[o])) for o in need]
                share_server_vectors(self.party, specs, self.W)
                self.generic_shared.update(need)
                self.generic_share_s += time.perf_counter() - t0
                info["generic_shared"] = need
            return True, info
        except (MaterialError, OpenCrabError) as exc:
            if isinstance(exc, ChannelError):
                raise
            info["error"] = str(exc)
            return False, info

    def _cancel_now(self, t: int) -> None:
        """Declared cancellation between gates: burn RESERVED ids and the buffered one-time material."""
        burned = self.ledger.burn_reserved("CANCEL")
        wasted = self.dealer.discard("CANCEL")
        self.burned_on_cancel += int(burned)
        self.wasted_bytes += int(wasted)
        self.cancelled = True
        self.cancel_epoch = int(t)
        self.predelivered = None
        self.t_end = time.perf_counter()
        self.epochs.append({"epoch": int(t), "gate": "cancelled", "burned_reserved": int(burned),
                            "wasted_bytes": int(wasted), "t_s": self.t_end - self.t_start})
        self._save_public_state()
        self._emit("status", {"event": "cancelled", "epoch": int(t), "burned_reserved": int(burned),
                              "wasted_bytes": int(wasted)})

    # -- one epoch ---------------------------------------------------------------------------------
    def run_epoch(self, t: int) -> str:
        """Run epoch ``t``; returns ``"ok"`` or ``"cancelled"``."""
        p = self.party
        ch = self.ch
        st0 = ch.stats.copy()
        mat0 = p.material_bytes
        t_e0 = time.perf_counter()
        # 0. declared cancellation status (public)
        mine_cancel = bool(self.cancel_event.is_set())
        status = ch.exchange({"cancel": mine_cancel, "epoch": int(t), "generation": int(self.generation)})
        if not isinstance(status, dict) or int(status.get("epoch", -1)) != int(t):
            raise ChannelError(f"SESSION_LOCKSTEP: the peer is at epoch {status.get('epoch') if isinstance(status, dict) else '?'}, we are at {t}")
        if mine_cancel or bool(status.get("cancel", False)):
            self._cancel_now(t)
            return "cancelled"
        n_rows, T, gate = self._rows_and_T(t)
        shapes = self._shapes(n_rows, T)
        schedule = horizon_schedule(self._shapes, t, self.n_epochs, self.prompt_len, n_rows, T)
        catalog = cands.gpt2_catalog(shapes)
        gate_id = f"e{t}"
        p.set_gate(gate_id, t)
        self.dealer.set_context(gate_id, t)
        # 1. sense (fixed-size probe at a fixed point of the epoch)
        now = time.time()
        cpu_share = self.shaper.cpu_share_at(now) if self.shaper is not None else 1.0
        frame = telemetry.sense_and_exchange(ch, self.party_id, self.telemetry_key, self.telemetry_seq, t,
                                             max(0.0, now - self.origin_wall_s), self.verifier,
                                             samples=int(self.cfg.probe_samples),
                                             burst=int(self.cfg.probe_burst),
                                             max_bandwidth_bps=float(self.cfg.telemetry_max_bandwidth_bps),
                                             cpu_load=float(1.0 - cpu_share))
        self.telemetry_seq += 1
        env = Env(float(frame.bandwidth_bps), float(frame.rtt_s) / 2.0, max(0.05, 1.0 - float(frame.cpu_load)))
        t_sensed = time.perf_counter()
        # 2. candidates and decision (identical on both sides)
        if self.policy is None:
            self.policy = self._make_policy(shapes)
        if isinstance(self.policy, ctrl.OraclePolicy):
            self.policy.schedule = schedule
        self.policy.observe(frame)
        state = self._session_state(t, n_rows, T)

        def admit_fn(plan: dict) -> Tuple[bool, str]:
            tape = registry.default_gpt2_tape(plan, int(self.cfg.n_layers), shapes, self.contract_hash)
            return registry.admit(tape, str(self.cfg.profile_id), state)

        # the horizon is priced step by step: the prefill and the decode steps have different public shapes
        cl = cands.enumerate_candidates(state, catalog, self.costmodel, env, shapes, self.contract, admit_fn,
                                        schedule=schedule)
        self._total_material_check(cl, state)
        decision = self.policy.decide(state, cl, env, t)
        rejected = sum(1 for c in cl if not c.feasible)
        self.rejected_candidates += rejected
        new_plan = cands.normalize_plan(decision.chosen, catalog)
        t_planned = time.perf_counter()
        # 3. two-phase commit: A agree on the plan, B prepare resources, C install and commit
        aborted = False
        d_mine = self._plan_digest(new_plan, t, catalog)
        d_theirs = ch.exchange({"phase": "A", "digest": d_mine})
        if not isinstance(d_theirs, dict) or d_theirs.get("digest") != d_mine:
            aborted = True
            self.aborts += 1
            new_plan = dict(self.plan)
            d_mine = self._plan_digest(new_plan, t, catalog)
            d_theirs = ch.exchange({"phase": "A2", "digest": d_mine})
            if not isinstance(d_theirs, dict) or d_theirs.get("digest") != d_mine:
                raise ChannelError("SESSION_COMMIT_DIVERGENCE: the parties disagree on the incumbent plan")
        ready, prep = self._prepare(new_plan, t, n_rows, T, catalog)
        ans = ch.exchange({"phase": "B", "ready": bool(ready)})
        both_ready = bool(ready) and isinstance(ans, dict) and bool(ans.get("ready", False))
        if not both_ready:
            self.aborts += 1
            aborted = True
            if new_plan == self.plan:
                raise MaterialError(f"SESSION_MATERIAL_UNAVAILABLE: epoch {t} cannot be prepared ({prep.get('error', '')})")
            new_plan = dict(self.plan)
            self.wasted_bytes += self.dealer.discard("PREPARE_FAILED")   # half-delivered material of the abandoned plan
            ready, prep2 = self._prepare(new_plan, t, n_rows, T, catalog, attempt=1)
            ans = ch.exchange({"phase": "B2", "ready": bool(ready)})
            if not (bool(ready) and isinstance(ans, dict) and bool(ans.get("ready", False))):
                raise MaterialError(f"SESSION_MATERIAL_UNAVAILABLE: epoch {t} cannot be prepared ({prep2.get('error', '')})")
            prep = prep2
        t_prepared = time.perf_counter()
        inst = self.model.install_for_plan(new_plan)
        t_installed = time.perf_counter()
        installed_digest = _digest({"installed": self.party.installed_objects(), "plan": new_plan, "epoch": int(t)})
        c_ans = ch.exchange({"phase": "C", "digest": installed_digest})
        if not isinstance(c_ans, dict) or c_ans.get("digest") != installed_digest:
            raise ChannelError("SESSION_COMMIT_DIVERGENCE: installed object sets differ after installation")
        switched = new_plan != self.plan
        if switched:
            self.switches += 1
            if t > 0:
                self.post_init_switches += 1
        if inst["objects"]:
            self.initializations += 1
        self.install_s_total += float(inst["seconds"])
        self.plan = dict(new_plan)
        self.model.set_plan(self.plan)
        t_committed = time.perf_counter()
        # 4. the declared gate under the committed plan
        t_x0 = time.perf_counter()
        if gate == "prefill":
            ids = list(self.cfg.prompt_ids) if self.party_id == 1 else None
            tok = self.model.prefill(ids, self.prompt_len)
        elif gate == "recovery_prefill":
            ids = (list(self.cfg.prompt_ids) + list(self.generated)) if self.party_id == 1 else None
            tok = self.model.prefill(ids, n_rows)
            self.kv_lost = False
        else:
            last = int(self.generated[-1]) if self.party_id == 1 else None
            tok = self.model.decode_step(last, self.prompt_len + t - 1)
        t_x1 = time.perf_counter()
        self.exec_s_total += t_x1 - t_x0
        if self.ttft_s is None and self.t_start is not None:
            self.ttft_s = t_x1 - self.t_start       # first released token of this session (both parties clock it)
        if tok is not None:
            self.generated.append(int(tok))
            self._emit("token", {"epoch": int(t), "index": len(self.generated) - 1, "token_id": int(tok)})
        self.epochs_done = t + 1
        self.t_end = t_x1
        # 5. prefetch the next epoch's material under the current plan (a later switch wastes it)
        t_pf0 = time.perf_counter()
        prefetch_bytes = 0
        if t + 1 < self.n_epochs and not self.cancel_event.is_set():
            n_next, T_next = 1, self.prompt_len + t + 1
            mplan = epoch_material_plan(self.plan, self.card, n_next, T_next, int(self.cfg.n_layers), self.contract)
            try:
                res = self.dealer.provision(material_requests(mplan, self.dealer.prefix, t + 1, self.contract))
                prefetch_bytes = int(res["bytes"])
                self.predelivered = (cands.plan_key(self.plan), t + 1)
            except MaterialError as exc:
                self.predelivered = None
                self._emit("status", {"event": "prefetch_failed", "epoch": t + 1, "error": str(exc)})
        t_pf1 = time.perf_counter()
        if self.cfg.cancel_after_epoch is not None and int(self.cfg.cancel_after_epoch) == t:
            self.cancel_event.set()
        # 6. epoch record
        d = ch.stats.diff(st0)
        rec = {
            "epoch": int(t), "gate": gate, "n_rows": int(n_rows), "T": int(T), "kv_bucket": _bucket(T, self.cfg.length_bucket),
            "plan": dict(self.plan), "switched": bool(switched), "aborted": bool(aborted),
            "decision_reason": decision.reason, "q_keep": decision.q_keep, "q_best": decision.q_best,
            "lower_bound_gain": decision.lower_bound_gain, "u_diff": decision.u_diff,
            "rejected_candidates": int(rejected), "n_candidates": len(cl),
            "sensed_bw_bps": frame.bandwidth_bps, "sensed_rtt_s": frame.rtt_s, "cpu_load": frame.cpu_load,
            "sensing_s": t_sensed - t_e0, "planning_s": t_planned - t_sensed, "prepare_s": t_prepared - t_planned,
            "install_s": t_installed - t_prepared, "commit_s": t_committed - t_installed, "exec_s": t_x1 - t_x0,
            "prefetch_s": t_pf1 - t_pf0, "epoch_s": t_pf1 - t_e0, "t_start_s": t_e0 - self.t_start,
            "installed_now": list(inst["objects"]), "install_bytes": int(inst["bytes"]),
            "material_bytes_consumed": int(p.material_bytes - mat0), "material_bytes_delivered": int(prep["material_bytes"]),
            "material_wait_s": float(prep["material_s"]), "prefetch_bytes": int(prefetch_bytes),
            "wasted_bytes": int(prep["wasted_bytes"]), "prefetched": bool(prep["prefetched"]),
            "masks_delivered": list(prep["masks"]), "generic_shared": list(prep["generic_shared"]),
            "bytes": int(d["bytes_sent"] + d["bytes_recv"]), "rounds": int(d["rounds"]),
            "shaping_delay_s": float(d["shaping_delay_s"]), "stalls": int(self.dealer.stalls),
            "released": tok is not None, "ops": self._ops_delta(),
        }
        self.epochs.append(rec)
        self.decisions.append({"epoch": int(t), "decision": decision.to_dict()})
        self._save_public_state()
        self._emit("epoch", rec)
        return "ok"

    def _ops_delta(self) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        stats = self.party.stats
        for s in stats[self._stats_mark:]:
            d = out.setdefault(s["kind"], {"calls": 0, "seconds": 0.0, "bytes": 0, "rounds": 0, "material_bytes": 0})
            d["calls"] += 1
            d["seconds"] += float(s["seconds"])
            d["bytes"] += int(s["bytes"])
            d["rounds"] += int(s["rounds"])
            d["material_bytes"] += int(s["material_bytes"])
        self._stats_mark = len(stats)
        return out

    # -- whole session -------------------------------------------------------------------------------
    def run(self) -> dict:
        """Handshake (if not done), run the remaining epochs, write and return the evidence record."""
        if self.party is None or self.verifier is None or self.resumed and self.telemetry_key is None:
            self.handshake()
        # the exogenous clock starts at the shared origin: both parties wait for it (bounded) before the first epoch
        lead = self.origin_wall_s - time.time()
        if 0.0 < lead <= 30.0:
            time.sleep(lead)
        if self.cpu_load is not None and not self.cpu_load._procs:
            self.cpu_load.start()
        try:
            for t in range(self.epochs_done, self.n_epochs):
                if self.run_epoch(t) == "cancelled":
                    break
            else:
                self.completed = True
                if self.t_end is None:
                    self.t_end = time.perf_counter()
        finally:
            if self.cpu_load is not None and (self.completed or self.cancelled):
                self.cpu_load.stop()
        return self.finish()

    def resume(self) -> dict:
        """Warm resume on the channel attached with :meth:`attach_channel`: handshake again and continue."""
        if self.completed:
            raise AdmissionError("SESSION_COMPLETED: nothing left to resume")
        self.handshake()
        return self.run()

    def finish(self) -> dict:
        """Final public summary exchange, record assembly, record write."""
        if self.t_end is None:
            self.t_end = time.perf_counter()
        if self.warm is not None and (self.completed or self.cancelled):
            self._save_warm_state()
        # The summary carries public values only. output_match is a predicate over the tokens that
        # RELEASE_FINAL gave to P1 alone (does the output equal a value P1 chose?); it never crosses
        # the channel: P1 writes it to its own record, the record of P0 carries None.
        summary_mine = {"peak_rss": int(peak_rss_bytes()), "tokens": len(self.generated) if self.party_id == 1 else None,
                        "completed": bool(self.completed), "cancelled": bool(self.cancelled)}
        try:
            theirs = self.ch.exchange(summary_mine)
            if not isinstance(theirs, dict):
                theirs = {}
        except ChannelError:
            theirs = {}
        rss_client = summary_mine["peak_rss"] if self.party_id == 1 else int(theirs.get("peak_rss", 0) or 0)
        rss_server = summary_mine["peak_rss"] if self.party_id == 0 else int(theirs.get("peak_rss", 0) or 0)
        tokens = len(self.generated) if self.party_id == 1 else int(theirs.get("tokens") or 0)
        output_match = self._output_match() if self.party_id == 1 else None
        self.record = self._build_record(rss_client, rss_server, tokens, output_match)
        self._write_record(self.record)
        # closing sync (outside every figure): P1 returns only after P0 has written its record, so a service that
        # starts the next session right after the client's last event never finds the server still busy
        try:
            if self.party_id == 0:
                self.ch.send({"fin": True})
            else:
                self.ch.recv()
        except ChannelError:
            pass
        self._emit("status", {"event": "finished", "completed": self.completed, "cancelled": self.cancelled,
                              "record_path": self.out_path})
        return dict(self.record, detail=self._detail(), paths={"record": self.out_path, "detail": self.detail_path,
                                                               "state": self.state_path})

    def _save_warm_state(self) -> None:
        """Leave the reusable state on the stack; one-time material never crosses a session boundary."""
        w = self.warm
        try:
            self.wasted_bytes += int(self.dealer.discard("SESSION_END"))
        except Exception:
            w.usable = False
            return
        if self.party is not None:
            w.wmask_b = dict(self.party._wmask_b)
            w.installed = {k: {"E": v["E"], "version": dict(v["version"])} for k, v in self.party._installed.items()}
            w.vectors = {k: v for k, v in self.party._vectors.items() if v is not None}
            w.generic_shared = set(self.generic_shared)
            w.masks = set(self.dealer._masks)
            w.plan = dict(self.plan)
            if not self._warm_saved:
                w.sessions += 1
                self._warm_saved = True

    def _dealer_generated(self) -> int:
        """Bytes the dealer produced so far for both recipients (0 when the service does not report it)."""
        try:
            if self.dealer.service is not None and hasattr(self.dealer.service, "status"):
                stt = self.dealer.service.status()
                return 2 * int(sum(int(v[1]) for v in (stt.get("produced") or {}).values()))
        except Exception:
            pass
        return 0

    def _output_match(self) -> Optional[bool]:
        ref = self.cfg.reference_tokens
        if ref is None or self.party_id != 1:
            return None
        return [int(t) for t in self.generated] == [int(t) for t in ref][:len(self.generated)] and \
            len(self.generated) == len(ref)

    def _first_mismatch_step(self) -> Optional[int]:
        """Index of the first released token that differs from the reference (P1 only).

        When every common position agrees but one sequence is longer, the index is the first position
        held by one side only. None when the sequences are equal or no reference is known.
        """
        ref = self.cfg.reference_tokens
        if ref is None or self.party_id != 1:
            return None
        return first_mismatch_step([int(t) for t in self.generated], [int(t) for t in ref])

    def _executed_plans_admitted(self) -> bool:
        plans = {cands.plan_key(e["plan"]): e["plan"] for e in self.epochs if "plan" in e}
        shapes = self._shapes(1, self.prompt_len + 1)
        for plan in plans.values():
            tape = registry.default_gpt2_tape(plan, int(self.cfg.n_layers), shapes, self.contract_hash)
            ok, _code = registry.admit(tape, str(self.cfg.profile_id), None)
            if not ok:
                return False
        return True

    def _build_record(self, rss_client: int, rss_server: int, tokens: int, output_match: Any) -> dict:
        st = self.ch.stats
        dealer_mode = self.dealer.dealer_mode
        eligible = (dealer_mode == MODE_ISOLATED and self.dealer.security_eligible and self.completed
                    and not self.cancelled and self.aborts == 0 and self._executed_plans_admitted()
                    and bool(self.cfg.party_rng_from_os))
        ingested = int(self.dealer.client.ingested_bytes) - self._base_ingested
        consumed = int(self.dealer.client.consumed_bytes) - self._base_consumed
        burned = int(self.ledger.bytes_in_state("BURNED")) - self._base_burned
        generated = max(0, self._dealer_generated() - int(self._base_generated or 0))
        if generated == 0:
            generated = ingested
        wall = max(0.0, float(self.t_end - self.t_start)) if self.t_start is not None else 0.0
        if self.warm_start:
            # a warm request: its wall time IS the warm figure; no cold time was measured in this session
            T_cold: Optional[float] = None
            T_warm = wall
        else:
            T_cold = wall
            T_warm = max(0.0, wall - self.setup_s - self.install_s_total - self.generic_share_s)
        if self._ch_base is not None:
            chd = st.diff(self._ch_base)
        else:
            chd = {"bytes_sent": st.bytes_sent, "bytes_recv": st.bytes_recv, "rounds": st.rounds}
        ch_bytes = int(chd["bytes_sent"] + chd["bytes_recv"]) + int(self._ch_carry_bytes)
        ch_rounds = int(chd["rounds"]) + int(self._ch_carry_rounds)
        total = int(self.card.get("n_layers_total", self.cfg.n_layers)) if self.card else int(self.cfg.n_layers)
        scope = self.workload_scope or workload_scope_for(self.cfg.n_layers, total, self.cfg.workload_scope)
        self.detail_path = self._detail_path()
        extra = {
            "session_id": self.session_id, "party": self.party_id, "model_version": self.model_version,
            "epochs_done": int(self.epochs_done), "completed": bool(self.completed), "cancelled": bool(self.cancelled),
            "resumed": bool(self.resumed), "switches": int(self.switches), "wasted_bytes": int(self.wasted_bytes),
            "burned_on_cancel": int(self.burned_on_cancel), "stalls": int(self.dealer.stalls),
            "setup_s": float(self.setup_s), "install_s": float(self.install_s_total),
            "generic_share_s": float(self.generic_share_s), "provision_s": float(self.dealer.provision_s),
            "prompt_len": int(self.prompt_len), "length_bucket": int(self.cfg.length_bucket),
            "shaped": bool(self.shaper.shaped) if self.shaper is not None else False,
            "timing_exclusions": list(TIMING_EXCLUSIONS),
            "plans": [cands.plan_key(e["plan"]) for e in self.epochs if "plan" in e],
            "session_wall_s": float(wall), "warm_start": bool(self.warm_start),
            "T_warm_basis": "measured-warm-request" if self.warm_start else "derived-cold-minus-setup-install",
            "warm_stack": self.warm is not None,
            "warm_objects": len(self.warm.installed) if (self.warm is not None and self.warm_start) else 0,
            "party_rng": "os-entropy" if self.cfg.party_rng_from_os else "public-seed",
            "client_key_domain": str(self.cfg.client_key_domain),
            "ttft_s": float(self.ttft_s) if self.ttft_s is not None else None,
        }
        if self.party_id == 1 and self.cfg.record_tokens:
            # the client's own record says what it released and what the reference was, so a mismatch can
            # be reported without the detail file; P0 never holds these values
            ref = self.cfg.reference_tokens
            extra["generated_tokens"] = [int(t) for t in self.generated]
            extra["reference_tokens"] = [int(t) for t in ref] if ref is not None else None
            extra["first_mismatch_step"] = self._first_mismatch_step()
        rec = {
            "schema": RECORD_SCHEMA,
            # one rule for both languages: DIAGNOSTIC when the configuration is not the
            # secure one (insecure dealer material, party masks from the public seed); a cancelled or
            # aborted run of a secure configuration stays MEASURED_NATIVE and is simply not eligible
            "status": ("MEASURED_NATIVE" if (dealer_mode == MODE_ISOLATED and bool(self.cfg.party_rng_from_os))
                       else "DIAGNOSTIC"),
            "security_eligible": bool(eligible),
            "code_commit": code_tree_hash(),
            "profile_id": str(self.cfg.profile_id),
            "numeric_contract_hash": self.contract_hash,
            "dealer_mode": dealer_mode,
            "diagnostic_reveals": 0,
            "workload_scope": scope,
            "model_revision": str(self.card.get("revision", "unpinned")) if self.card else "unknown",
            "trace_id": self.trace_id,
            "trace_hash": self.trace_hash,
            "clock_basis": CLOCK_BASIS,
            "policy_id": self.policy_id,
            "initial_state_hash": self.initial_state_hash,
            "run_id": self.run_id,
            "repeat": int(self.cfg.repeat),
            "initializations": int(self.initializations),
            "post_init_switches": int(self.post_init_switches),
            "rejected_candidates": int(self.rejected_candidates),
            "aborts": int(self.aborts),
            "T_cold_s": T_cold,
            "T_warm_s": T_warm,
            "T_online_s": float(self.exec_s_total),
            "bytes": int(ch_bytes),
            "rounds": int(ch_rounds),
            "material_generated": int(max(0, generated)),
            "material_transferred": int(max(0, ingested)),
            "material_consumed": int(max(0, consumed)),
            "material_burned": int(max(0, burned)),
            "peak_rss_client": int(rss_client),
            "peak_rss_server": int(rss_server),
            "tokens": int(tokens),
            "output_match": output_match if output_match in (True, False) else None,
            "raw_measurement_path": record_path(self.detail_path),
            "record_kind": "session",
            "seed": int(self.cfg.seed),
            "n_layers": int(self.cfg.n_layers),
            "T": int(self.n_epochs),
            "extra": extra,
        }
        return rec

    def _detail_path(self) -> str:
        p = Path(self.out_path)
        # run ids carry ':' and '{' (matrix cells, fixed plans): keep the file name portable
        safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(self.run_id))[:120]
        return str(p.with_name(f"{p.stem}.{safe}.p{self.party_id}.detail.jsonl"))

    def _detail(self) -> dict:
        return {
            "session_id": self.session_id, "party": self.party_id, "model_version": self.model_version,
            "model_card": self.card, "epochs": self.epochs, "decisions": self.decisions,
            "completed": self.completed, "cancelled": self.cancelled, "cancel_epoch": self.cancel_epoch,
            "resumed": self.resumed, "generation": self.generation, "epochs_done": self.epochs_done,
            "setup_s": self.setup_s, "install_s": self.install_s_total, "generic_share_s": self.generic_share_s,
            "exec_s": self.exec_s_total, "wasted_bytes": self.wasted_bytes, "burned_on_cancel": self.burned_on_cancel,
            "stalls": self.dealer.stalls, "provision_s": self.dealer.provision_s,
            "provisioned_bytes": self.dealer.provisioned_bytes, "requests_sent": self.dealer.requests_sent,
            "ledger_counts": self.ledger.counts(), "telemetry_accepted": len(self.verifier.accepted) if self.verifier else 0,
            "telemetry_rejected": len(self.verifier.rejected) if self.verifier else 0,
            "releases": len(self.party.release_log) if self.party is not None else 0,
            "generated_tokens": [int(t) for t in self.generated] if (self.party_id == 1 and self.cfg.record_tokens) else None,
            "shaped": bool(self.shaper.shaped) if self.shaper is not None else False,
            "trace": self.shaper.describe() if self.shaper is not None else None,
            "origin_wall_s": self.origin_wall_s, "cpu_load_workers": self.cpu_load.max_workers if self.cpu_load else 0,
            "warm_start": bool(self.warm_start), "ttft_s": self.ttft_s,
            "output_match": self._output_match() if self.party_id == 1 else None,
            "first_mismatch_step": self._first_mismatch_step(),
            "reference_tokens": ([int(t) for t in self.cfg.reference_tokens]
                                 if (self.party_id == 1 and self.cfg.record_tokens
                                     and self.cfg.reference_tokens is not None) else None),
            "timing_exclusions": list(TIMING_EXCLUSIONS), "switches": self.switches,
            "channel": self.ch.stats.snapshot(),
        }

    def _write_record(self, rec: dict) -> None:
        schema = None
        try:
            from ..experiments import schema as schema_mod   # type: ignore
            schema = schema_mod
        except Exception:
            schema = None
        if schema is not None and hasattr(schema, "new_record"):
            rec = schema.new_record(**rec)      # fills created_utc, validates, refuses unknown fields
            self.record = rec
        else:
            validate_record(rec)
        # Both party processes append to the same files within milliseconds of
        # each other; a plain open(..., "a") can lose one line on Windows, so the
        # append goes through the lock-protected writer (schema.locked_append_lines).
        _append_lines(self.out_path, [json.dumps(rec, sort_keys=True)])
        detail = self._detail()
        lines = [json.dumps({"kind": "epoch", "run_id": self.run_id, "party": self.party_id, **e}, sort_keys=True)
                 for e in self.epochs]
        lines.append(json.dumps({"kind": "summary", "run_id": self.run_id,
                                 **{k: v for k, v in detail.items() if k != "epochs"}}, sort_keys=True, default=str))
        _append_lines(self.detail_path, lines)


# ---------------------------------------------------------------------------
# API entry point
# ---------------------------------------------------------------------------
def run_session_party(party: int, cfg: SessionConfig, channel: Any, dealer: Any, ledger: MaterialLedger,
                      costmodel: Any, trace: Any, out_path: str, **kw: Any) -> dict:
    """Run one party of a full session; writes one evidence record line to ``out_path``.

    ``dealer`` is the party's ``DealerClient`` (or a prepared ``SessionDealer``);
    ``kw`` may carry ``dealer_service`` (the authenticated dealer connection),
    ``cancel_event``, ``origin_wall_s`` (shared shaper origin), ``weights`` /
    ``meta`` (P0 only; otherwise ``cfg.model`` is loaded), ``hooks``
    (``token`` / ``epoch`` / ``status`` callables), ``start_cpu_load`` and
    ``warm`` (the party's :class:`WarmStack`; ``dealer`` and ``ledger`` must then be the stack's).
    Returns the record plus ``detail`` and ``paths``.
    """
    runner = SessionRunner(int(party), cfg, channel, dealer, ledger, costmodel, trace, out_path, **kw)
    return runner.run()


__all__ = [
    "RECORD_SCHEMA", "RECORD_FIELDS", "CLOCK_BASIS", "COLD_PLAN", "DEFAULT_FIXED_PLAN", "TIMING_EXCLUSIONS",
    "SessionConfig", "save_model_npz", "load_model", "model_version_of", "workload_scope_for", "model_card", "code_tree_hash",
    "peak_rss_bytes", "record_path", "first_mismatch_step", "object_shapes", "generic_objects", "server_vector_specs",
    "share_server_vectors",
    "tournament_counts", "epoch_material_plan", "material_requests", "material_plan_bytes", "SessionDealer",
    "validate_record", "SessionRunner", "run_session_party", "WarmStack", "new_warm_stack",
]
