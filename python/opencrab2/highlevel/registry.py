"""Component registry and admission gate.

Every realization is a :class:`Component` with a status (``ADMITTED``,
``UNCERTIFIED_DISABLED``, ``UNAVAILABLE``, ``TEST_ONLY``), the profiles it is
valid under, typed input/output domains, a construction reference, its
parameters, a proof-ledger id, a failure bound and its material type.
``Instantiated`` is the outcome of the admission checks, never a table's
self-certification.

Admission order, applied by :func:`admit` to a plan tape:

1. construction/evidence: every kind is a registered component with status
   ADMITTED and a proof-ledger entry (``NOT_ADMITTED_COMPONENT``);
2. profile: the requested profile is listed by every component
   (``PROFILE_MISMATCH``; an active profile never falls through to a passive
   realization);
3. types/numerics/budgets: typecheck.check_tape (``MISSING_CONVERSION``,
   ``DOMAIN_MISMATCH``), declared contract hash equals the session contract
   (``CONTRACT_MISMATCH``), the declared magnitude bound of every k-ary site
   fits the fused-node bound (``BUDGET_EXCEEDED`` with detail
   ``KARY_MAGNITUDE``, see below), route-declared statistical terms fit the
   session budget (``BUDGET_EXCEEDED``; a declared term without a budget fails
   closed);
4. resources: installed objects referenced by the tape are either absent (to
   be installed transactionally) or present at exactly the session version
   (``OBJECT_VERSION_MISMATCH``); declared material need fits the tracked
   stock (``INSUFFICIENT_MATERIAL``); declared memory fits the limit
   (``MEMORY_INFEASIBLE``);
5. cost: never a rejection reason; ranking belongs to the controller.

``admit`` returns ``(ok, code)`` with the bare code; :func:`admit_explain`
adds the detail string.

k-ary magnitude check (paper, Section 5.3, equivalence and admission).  A fused tournament node
of g >= 3 members is correct only when the compared differences inside the
node stay below 2^(n-2-ceil(log2 g)) ring units; a node of two members and
every pairwise comparison need the DCF bound 2^(n-1).  The check is a
property of the contract (n = dcf_domain_bits, f = frac_bits) and of the
declared public bound of the site, never of secret values: a tape entry of
kind ``argmax_k*`` declares ``logits_bound`` and an entry of kind
``softmax_rows_k*`` declares ``score_bound`` (real units, |x| <= bound for
every operand of the site; ``operand_bound`` is accepted for either).  The
defaults are :data:`DEFAULT_LOGITS_BOUND` = 512 and
:data:`DEFAULT_SCORE_BOUND` = 512: declarations of the default GPT-2 tape,
not measured properties of a model; a deployment whose logits or scores can
exceed them must declare its own bound.  With |x| <= B the largest difference
is 2 B, so the site is admitted iff 2 B 2^f < 2^(n-2-ceil(log2 g)) for the
largest node size g that the row length and k produce (and the index of an
argmax row fits the same bound).  At n = 32, f = 16 this means B < 2048 for
k = 4, B < 1024 for k = 8 and B < 16384 for k = 2.

Z_t statistical term (paper, Section 6, loss).  ``ring_to_field`` and ``a2h_exact`` consume
DCF keys with Z_t payload whose value words are 128-bit blocks reduced mod t:
statistical distance at most ``n_word * t / 2^128`` per element with
``n_word`` = :data:`ZT_WORDS_PER_R2F_ELEMENT` = 516.  Admission charges this
term per element against the session budget (2^-82.99 per element at
t = 68720050177) and fails closed without a budget.  Statuses of note: ``a2h_exact`` is ADMITTED at the
primitive level only (nothing admitted consumes H_BFV, so no layer lane
exists); ``he_linear_bfv_matvec`` is UNAVAILABLE on real parameters;
``toy_bridge`` is TEST_ONLY; ``ckks_fast_h2a`` is UNCERTIFIED_DISABLED.

What is tested (tests/test_registry.py): the default GPT-2 tape is admitted
under SH2P-TrustedOffline-v1; an active profile is rejected with
PROFILE_MISMATCH; tapes using ckks_fast_h2a or he_linear_bfv_matvec are
rejected with NOT_ADMITTED_COMPONENT; an installed object at another model
version is rejected with OBJECT_VERSION_MISMATCH while a missing object is
accepted; a contract hash mismatch, a budget overrun, insufficient stock and
a memory overrun produce their codes; the k-ary magnitude check at the
default and at violating declared bounds for every k; the Z_t budget charge;
every ADMITTED component has a ledger entry with the required fields.  Design limit: the gate checks declarations
(hashes, versions, stock figures); it does not measure the running process.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

from opencrab2.errors import AdmissionError

from . import costmodel as cm
from . import typecheck

if TYPE_CHECKING:  # pragma: no cover
    from .candidates import SessionState

PROFILE_ID = "SH2P-TrustedOffline-v1"
PROFILES_SH: Tuple[str, ...] = (PROFILE_ID,)

STATUS_ADMITTED = "ADMITTED"
STATUS_UNCERTIFIED_DISABLED = "UNCERTIFIED_DISABLED"
STATUS_UNAVAILABLE = "UNAVAILABLE"
STATUS_TEST_ONLY = "TEST_ONLY"
STATUSES: Tuple[str, ...] = (STATUS_ADMITTED, STATUS_UNCERTIFIED_DISABLED, STATUS_UNAVAILABLE, STATUS_TEST_ONLY)

REASON_CODES: Tuple[str, ...] = (
    "NOT_ADMITTED_COMPONENT", "PROFILE_MISMATCH", "MISSING_CONVERSION", "DOMAIN_MISMATCH",
    "CONTRACT_MISMATCH", "BUDGET_EXCEEDED", "INSUFFICIENT_MATERIAL", "MEMORY_INFEASIBLE",
    "OBJECT_VERSION_MISMATCH",
)

LEDGER_PATH = Path(__file__).with_name("proof_ledger.json")
REGISTRY_VERSION = "opencrab2-registry-v3"

# Declared public operand bounds of the k-ary sites (real units), checked at admission.  These are
# declarations carried by the plan tape, not measured properties of a model.
DEFAULT_LOGITS_BOUND = 512.0     # |logit| <= 2^9 at an argmax site
DEFAULT_SCORE_BOUND = 512.0      # |attention score| <= 2^9 at a softmax site
KARY_DETAIL = "KARY_MAGNITUDE"
KARY_KINDS: Dict[str, Tuple[str, int]] = {
    "argmax_k2": ("logits_bound", 2), "argmax_k4": ("logits_bound", 4), "argmax_k8": ("logits_bound", 8),
    "softmax_rows_k2": ("score_bound", 2), "softmax_rows_k4": ("score_bound", 4),
}
# Z_t PRG value words Gen derives for one ring_to_field element (one 128-bit word per payload component and
# level, see lowlevel/prg.py; paper, Section 6, loss): two DCF key pairs over
# n = 64 levels with one payload component; per pair two words (left, right) for each party seed at every
# level plus one leaf word per party: 2 * (4 * 64 + 2).
ZT_WORDS_PER_R2F_ELEMENT = 2 * (4 * 64 + 2)
ZT_DETAIL = "ZT_WORDS"


@dataclass(frozen=True)
class Component:
    name: str
    kind: str                       # linear | nonlinear | conversion | share-op | release | bridge
    status: str
    entry: str                      # dotted path of the executable (empty when UNAVAILABLE)
    profiles: Tuple[str, ...]
    in_domain: str                  # "A64" or "A64|Zt" (alternatives separated by |)
    out_domain: str
    construction_ref: str
    params: Dict[str, Any] = field(default_factory=dict)
    proof_ledger_id: str = ""
    failure_bound: str = ""
    material_type: str = "none"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["profiles"] = list(self.profiles)
        return d


_P = "opencrab2.highlevel.party.Party."
_SECUREML = "SecureML local truncation (Mohassel and Zhang, IEEE S&P 2017), Thm. 1"
_DCF = ("DCF of Boyle, Chandran, Gilboa, Gupta, Ishai, Kumar, Rathee (Eurocrypt 2021), GGM-tree batch with the "
        "product-group Convert: component-specific PRG value words (paper, Section 5.3, a finding on comparison keys)")
_KARY_FB = ("0 (exact) given compared differences below 2^(n-2-ceil(log2 g)) ring units inside every node of "
            "g >= 3 members and below 2^(n-1) elsewhere; checked at admission from the contract and the declared "
            "public operand bound (BUDGET_EXCEEDED, KARY_MAGNITUDE)")
_ZT_FB = ("exact reconstruction; statistical distance of the Z_t key words at most n_word * t / 2^128 per element, "
          "n_word = 516 (2^-82.99 at t = 68720050177), charged to the session budget at admission")
_BEAVER = "Beaver circuit randomization (CRYPTO 1991) with a trusted dealer"
_DRELU_FB = "0 (exact) given |x| < 2^(n-1) ring units, n = 32 (contract dcf_domain_bits)"
_TRUNC_LOCAL_FB = ("a locally truncated value x is off by a full wrap of 2^(64-s) ring units with probability |x|/2^64 "
                   "(SecureML local truncation); a 4-layer session of 16 prompt and 8 generated tokens executes 5.49 million "
                   "local truncations and a 12-layer session 15.6 million, and the sum over the truncations that can reach a "
                   "released token gives a wrong token sequence in about 0.8 percent (4 layers) and 2.4 percent (12 layers) "
                   "of sessions; a wrap in the LM head is harmless because the argmax compares modulo 2^32")

_COMPONENTS: Tuple[Component, ...] = (
    Component("share_input", "share-op", STATUS_ADMITTED, _P + "share_input", PROFILES_SH,
              "plain_client|plain_server", "A64", "additive secret sharing over Z_2^64 (information-theoretic)",
              {"ring_bits": 64}, "PL-share_input", "0", "none"),
    Component("linear_wmask", "linear", STATUS_ADMITTED, _P + "linear_wmask", PROFILES_SH, "A64", "A64",
              _BEAVER + "; weight-masked variant: reusable mask B per matrix, E = W - B revealed once at install, "
              "fresh (A, C = A B) per call",
              {"requires_installed": True, "trunc": "local_prob", "online_bytes": "16 n d_in + framing", "rounds": 1},
              "PL-linear_wmask", _TRUNC_LOCAL_FB, "wmask_mask"),
    Component("linear_generic", "linear", STATUS_ADMITTED, _P + "linear_generic", PROFILES_SH, "A64", "A64",
              _BEAVER + "; matrix triple (A, B, C = A B) per call, W shared in",
              {"trunc": "local_prob", "online_bytes": "16 (n d_in + d_in d_out) + framing", "rounds": 1},
              "PL-linear_generic", _TRUNC_LOCAL_FB, "matrix_triple"),
    Component("embedding_wmask", "linear", STATUS_ADMITTED, _P + "linear_wmask", PROFILES_SH, "A64", "A64",
              "linear_wmask on the client's shared one-hot (n = 1, d_in = V), no truncation",
              {"requires_installed": True, "trunc": "none", "rounds": 1},
              "PL-embedding_wmask", "0 (no truncation)", "wmask_mask"),
    Component("embedding_generic", "linear", STATUS_ADMITTED, _P + "linear_generic", PROFILES_SH, "A64", "A64",
              "linear_generic on the client's shared one-hot (n = 1, d_in = V), no truncation",
              {"trunc": "none", "online_bytes": "16 (n V + V d) + framing", "rounds": 1},
              "PL-embedding_generic", "0 (no truncation)", "matrix_triple"),
    Component("matmul_shared", "linear", STATUS_ADMITTED, _P + "matmul_shared", PROFILES_SH, "A64", "A64",
              _BEAVER + "; matrix triple for shared x shared products",
              {"trunc": "local_prob", "rounds": 1}, "PL-matmul_shared", _TRUNC_LOCAL_FB, "matrix_triple"),
    Component("drelu", "nonlinear", STATUS_ADMITTED, _P + "drelu", PROFILES_SH, "A64", "A64",
              "one-round DReLU from two DCF keys (paper, Section 5.3): 1{x >= 0} = DCF_q(y) - DCF_p(y) + 1{p > q}; " + _DCF,
              {"n_bits": 32, "payloads": 1, "rounds": 1}, "PL-drelu", _DRELU_FB, "drelu"),
    Component("relu", "nonlinear", STATUS_ADMITTED, _P + "relu", PROFILES_SH, "A64", "A64",
              "drelu with payload (1, r): ReLU(x) = y [b] - [r b] in the same round; the two "
              "payload components use independent Convert words; " + _DCF,
              {"n_bits": 32, "payloads": 2, "rounds": 1}, "PL-relu", _DRELU_FB, "drelu"),
    Component("trunc_faithful", "share-op", STATUS_ADMITTED, _P + "trunc_faithful", PROFILES_SH, "A64", "A64",
              "faithful truncation gate with the signed offset (paper, Section 5.1): open y = x + a + r, "
              "wrap and borrow bits from two DCF keys, minus 2^(L-1-f)",
              {"f": 16, "L": 40, "rounds": 1, "material_bytes_per_element": 2064},
              "PL-trunc_faithful", "0 (exact) for |x| < 2^(L-1)", "trunc_faithful"),
    Component("trunc_local", "share-op", STATUS_ADMITTED, _P + "trunc_local", PROFILES_SH, "A64", "A64",
              _SECUREML + " (0 rounds); admitted only for contract-bounded products",
              {"f": 16, "rounds": 0, "requires": "|x| < 2^(l-1), l = 40"},
              "PL-trunc_local", _TRUNC_LOCAL_FB, "none"),
    Component("softmax_rows_k2", "nonlinear", STATUS_ADMITTED, _P + "softmax_rows", PROFILES_SH, "A64", "A64",
              "binary max tournament (DReLU + select), exp_neg (order-2 base, 7 squarings), recip (K = 11, degree 5, "
              "2 Newton), product; public causal mask",
              {"k_max": 2, "exp_rounds": 10, "recip_rounds": 12, "score_bound_default": DEFAULT_SCORE_BOUND},
              "PL-softmax_rows_k2",
              "max: " + _KARY_FB + "; approximation error per polynomial contract", "drelu"),
    Component("softmax_rows_k4", "nonlinear", STATUS_ADMITTED, _P + "softmax_rows", PROFILES_SH, "A64", "A64",
              "as softmax_rows_k2 with a fused 4-ary max tournament (2 ceil(log_4 T) rounds; per node of g members "
              "g(g-1)/2 pairwise DReLUs, then one Beaver select (g = 2) or g-1 payload DReLUs (g >= 3)); " + _DCF,
              {"k_max": 4, "exp_rounds": 10, "recip_rounds": 12, "score_bound_default": DEFAULT_SCORE_BOUND},
              "PL-softmax_rows_k4",
              "max: " + _KARY_FB + "; approximation error per polynomial contract", "drelu"),
    Component("layernorm_rows", "nonlinear", STATUS_ADMITTED, _P + "layernorm_rows", PROFILES_SH, "A64", "A64",
              "mean (local), square via the faithful gate, rsqrt (j0 = -7, m = 15, degree 5, 2 Newton), products "
              "(trunc_faithful for dx*dx)",
              {"rounds": 18, "square": "faithful", "extra_bits": 4}, "PL-layernorm_rows",
              "square exact; rsqrt normalisation product V * s2 has l_x = 2f + e2 = 34 bits (s2 at scale 2^(f+e2), "
              "e2 = max(0, 2 (j0+1+m) - f)): <= 2^-29 per row; other products: " + _TRUNC_LOCAL_FB,
              "trunc_faithful"),
    Component("gelu_clip12", "nonlinear", STATUS_ADMITTED, _P + "gelu", PROFILES_SH, "A64", "A64",
              "clip at |x| <= 5 with two DReLU, degree-12 Chebyshev least-squares erf-GELU polynomial by Horner, blend",
              {"R": 5, "degree": 12, "max_abs_err": 4.84e-3, "rounds": 14, "semantics": "erf_gelu"},
              "PL-gelu_clip12", "approximation max_abs_err 4.84e-3 (contract), not a failure probability", "drelu"),
    Component("silu_clip12", "nonlinear", STATUS_ADMITTED, _P + "silu", PROFILES_SH, "A64", "A64",
              "clip at |x| <= 8 with two DReLU, degree-12 polynomial by Horner, blend",
              {"R": 8, "degree": 12, "rounds": 14}, "PL-silu_clip12",
              "approximation error per polynomial contract silu_clip12", "drelu"),
    Component("argmax_k2", "nonlinear", STATUS_ADMITTED, _P + "argmax_row", PROFILES_SH, "A64", "A64",
              "binary tournament of DReLU + (value, index) selects; lowest index wins ties; index shares released "
              "only through release_final",
              {"k": 2, "tie": "lowest_index", "logits_bound_default": DEFAULT_LOGITS_BOUND}, "PL-argmax_k2",
              _KARY_FB, "drelu"),
    Component("argmax_k4", "nonlinear", STATUS_ADMITTED, _P + "argmax_row", PROFILES_SH, "A64", "A64",
              "fused 4-ary tournament (per node of g members g(g-1)/2 pairwise DReLUs, then one Beaver select per "
              "output value (g = 2) or g-1 payload DReLUs per output value (g >= 3); 2 rounds per level), identical "
              "output to argmax_k2 (lowest index on ties); " + _DCF,
              {"k": 4, "tie": "lowest_index", "logits_bound_default": DEFAULT_LOGITS_BOUND}, "PL-argmax_k4",
              _KARY_FB, "drelu"),
    Component("argmax_k8", "nonlinear", STATUS_ADMITTED, _P + "argmax_row", PROFILES_SH, "A64", "A64",
              "fused 8-ary tournament (per node of g members g(g-1)/2 pairwise DReLUs, then one Beaver select per "
              "output value (g = 2) or g-1 payload DReLUs per output value (g >= 3); 2 rounds per level), identical "
              "output to argmax_k2 (lowest index on ties); " + _DCF,
              {"k": 8, "tie": "lowest_index", "logits_bound_default": DEFAULT_LOGITS_BOUND}, "PL-argmax_k8",
              _KARY_FB, "drelu"),
    Component("ring_to_field", "conversion", STATUS_ADMITTED, _P + "ring_to_field", PROFILES_SH, "A64", "Zt",
              "A64 -> Zt resharing with signed carry (paper, Section 5.2): open y = u + r, wrap and sign bits "
              "from two 64-bit DCF keys with Z_t payload (128-bit PRG blocks reduced mod t), "
              "z = x_signed mod t; w and b stay shared",
              {"t": 68720050177, "rounds": 1, "material_bytes_per_element": 3272,
               "zt_words_per_element": ZT_WORDS_PER_R2F_ELEMENT}, "PL-ring_to_field", _ZT_FB,
              "ring_to_field"),
    Component("a2h_exact", "conversion", STATUS_ADMITTED, "opencrab2.highlevel.he_lane.a2h_exact", PROFILES_SH,
              "A64|Zt", "H_BFV",
              "ring_to_field then BFV (Brakerski 2012; Fan and Vercauteren 2012) via SEAL/TenSEAL: P1 encrypts z1 "
              "batched with plaintext modulus t, P0 adds z0 as plaintext; ciphertext held by P0 only",
              {"scheme": "BFV", "N": 8192, "t": 68720050177, "scope": "primitive", "rounds": 2,
               "ciphertext_bytes_measured": cm.BFV_CT_BYTES, "zt_words_per_element": ZT_WORDS_PER_R2F_ELEMENT},
              "PL-a2h_exact", "0 (exact) within the BFV noise budget; ring_to_field part: " + _ZT_FB,
              "ring_to_field"),
    Component("he_linear_bfv_matvec", "linear", STATUS_UNAVAILABLE, "", PROFILES_SH, "H_BFV", "H_BFV",
              "BFV plaintext-weight matrix-vector product on real parameters: no executable (TenSEAL exposes no "
              "BFV matmul or rotations)", {"N": 8192}, "", "n/a (no executable)", "none"),
    Component("toy_bridge", "bridge", STATUS_TEST_ONLY, "opencrab2.highlevel.bridge.h2a_ref", PROFILES_SH, "A64", "A64",
              "power-of-two toy BFV reference bridge A2H_toy -> matvec_pt -> H2A_ref with secure internal decryption "
              "(paper, Appendix B, H2A-ref); reference design, N <= 512", {"N": [64, 128, 256, 512], "rounds": 2},
              "PL-toy_bridge", "0 within the checked noise bound", "beaver"),
    Component("ckks_fast_h2a", "conversion", STATUS_UNCERTIFIED_DISABLED, "opencrab2.highlevel.he_lane.CKKSFastH2A",
              PROFILES_SH, "H_CKKS", "A64",
              "flooded masked decryption to the keyholder; per-slot interval mask argument only: joint-view "
              "simulation, noise tails and precision not certified; measured as DIAGNOSTIC only",
              {"kappa_mask_per_slot": 20, "certificate": "not emitted"}, "", "uncertified", "none"),
    Component("release_final", "release", STATUS_ADMITTED, _P + "reveal_final", PROFILES_SH, "A64", "plain_client",
              "RELEASE_FINAL through ReleasePolicy: declared recipient P1, type and count; P0 sends its share",
              {"recipient": 1}, "PL-release_final", "n/a", "none"),
)

REGISTRY: Dict[str, Component] = {c.name: c for c in _COMPONENTS}


def export_registry() -> List[dict]:
    """Serialisable registry (one dict per component)."""
    return [c.to_dict() for c in _COMPONENTS]


_LEDGER_CACHE: Optional[dict] = None


def load_proof_ledger(path: Any = None) -> dict:
    """``{ledger id: entry}`` from proof_ledger.json (cached for the default path)."""
    global _LEDGER_CACHE
    p = Path(path) if path is not None else LEDGER_PATH
    if path is None and _LEDGER_CACHE is not None:
        return _LEDGER_CACHE
    with open(p, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    entries = {str(k): v for k, v in dict(doc.get("entries", {})).items()}
    if path is None:
        _LEDGER_CACHE = entries
    return entries


def component_admissible(name: str, profile: str) -> Tuple[bool, str]:
    """Steps 1 and 2 for one component name: (ok, code)."""
    comp = REGISTRY.get(name)
    if comp is None:
        return False, "NOT_ADMITTED_COMPONENT"
    if comp.status != STATUS_ADMITTED:
        return False, "NOT_ADMITTED_COMPONENT"
    if comp.proof_ledger_id not in load_proof_ledger():
        return False, "NOT_ADMITTED_COMPONENT"
    if profile not in comp.profiles:
        return False, "PROFILE_MISMATCH"
    return True, "OK"


# material need of a tape entry (per party, by type) -------------------------
_GELU_LIKE = {"gelu_clip12", "silu_clip12"}


def entry_material(entry: Mapping[str, Any], contract: Any = None) -> Dict[str, int]:
    kind = str(entry["kind"])
    shape = entry.get("shape")
    if shape is None:
        return {}
    n, d_in, d_out = (int(shape[0]), int(shape[1]), int(shape[2]) if len(shape) > 2 else 1)
    count = int(entry.get("count", 1))
    if kind in cm.COMPONENTS:
        comp = kind
    elif kind in _GELU_LIKE:
        comp = "gelu_clip12"
    elif kind in ("drelu", "relu"):
        p = 2 if kind == "relu" else 1
        _, nb = cm._contract_bits(contract)
        return {"drelu": count * cm.drelu_material_bytes(nb, p) * n * d_in}
    else:
        return {}
    return {t: count * b for t, b in cm.material_breakdown(comp, n, d_in, d_out, contract).items()}


def tape_material(tape: Sequence[Mapping[str, Any]], contract: Any = None) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for e in tape:
        for t, b in entry_material(e, contract).items():
            out[t] = out.get(t, 0) + int(b)
    return out


# numeric checks of step 3 ------------------------------------------------------
_CONTRACT_BY_HASH: Optional[Dict[str, Any]] = None


def _known_contracts() -> Dict[str, Any]:
    global _CONTRACT_BY_HASH
    if _CONTRACT_BY_HASH is None:
        from .contract import DEFAULT_CONTRACT, SMOLLM_CONTRACT
        _CONTRACT_BY_HASH = {c.contract_hash(): c for c in (DEFAULT_CONTRACT, SMOLLM_CONTRACT)}
    return _CONTRACT_BY_HASH


def resolve_contract(state: Optional["SessionState"], contract: Any = None) -> Any:
    """The numeric contract the numeric checks use.

    An explicit ``contract`` wins; otherwise the session contract hash is
    looked up among the contracts of ``contract.py``; a session without a
    registered hash (or no session) is checked against the default contract.
    A declared hash that differs from the session hash is CONTRACT_MISMATCH
    before this function is reached.
    """
    if contract is not None:
        return contract
    known = _known_contracts()
    if state is not None and str(getattr(state, "contract_hash", "")) in known:
        return known[str(state.contract_hash)]
    from .contract import DEFAULT_CONTRACT
    return DEFAULT_CONTRACT


def kary_limit_log2(n_bits: int, g: int) -> int:
    """log2 of the exclusive bound (ring units) on a compared difference inside a tournament node of g members.

    g = 2: the DCF bound 2^(n-1).  g >= 3: the fused-node bound
    2^(n-2-ceil(log2 g)) of ``party.Party._tournament`` (M / 2 with
    M = 2^(n-1-ceil(log2 g))).
    """
    g = int(g)
    if g < 2:
        raise ValueError(f"KARY_PARAM: node size must be >= 2, got {g}")
    if g == 2:
        return int(n_bits) - 1
    return int(n_bits) - 2 - (g - 1).bit_length()


def kary_bound_limit(k: int, row_len: Optional[int], n_bits: int, frac_bits: int) -> Fraction:
    """Exclusive limit on the declared operand bound B (real units) of a k-ary site: admitted iff B < limit."""
    sizes = cm.tournament_node_sizes(int(row_len), int(k)) if row_len is not None else (int(k),)
    if not sizes:
        return Fraction(2) ** (int(n_bits) - 2 - int(frac_bits))        # a single item: nothing is compared
    worst = min(kary_limit_log2(n_bits, g) for g in sizes)
    # |x_i - x_j| <= 2 B real = 2 B 2^f ring units must stay below 2^worst
    return Fraction(2) ** (worst - 1 - int(frac_bits))


def kary_site_check(entry: Mapping[str, Any], contract: Any = None) -> Tuple[bool, str]:
    """(ok, detail) of the k-ary magnitude check for one tape entry; non k-ary kinds pass.

    The detail string starts with ``KARY_MAGNITUDE`` on refusal.  Only public
    quantities enter: the kind (k), the declared shape (row length), the
    declared bound and the contract (n, f).
    """
    kind = str(entry.get("kind", ""))
    if kind not in KARY_KINDS:
        return True, "OK"
    bound_key, k = KARY_KINDS[kind]
    P = cm.protocol_params(contract)
    n_bits, f = int(P.dcf_domain_bits), int(P.frac_bits)
    shape = entry.get("shape")
    row_len = int(shape[1]) if shape is not None and len(shape) > 1 else None
    declared = entry.get("operand_bound", entry.get(bound_key))
    if declared is None:
        declared = DEFAULT_LOGITS_BOUND if bound_key == "logits_bound" else DEFAULT_SCORE_BOUND
    if isinstance(declared, bool) or not isinstance(declared, (int, float)) or not math.isfinite(declared) or declared <= 0:
        return False, f"{KARY_DETAIL}: {kind} declares an invalid {bound_key} {declared!r} (positive finite real required)"
    if row_len is not None and row_len < 1:
        return False, f"{KARY_DETAIL}: {kind} declares an invalid row length {row_len}"
    limit = kary_bound_limit(k, row_len, n_bits, f)
    if not Fraction(declared) < limit:
        sizes = cm.tournament_node_sizes(row_len, k) if row_len is not None else (k,)
        return False, (f"{KARY_DETAIL}: {kind} with {bound_key} {float(declared):g} over a row of "
                       f"{row_len if row_len is not None else 'undeclared'} items (node sizes {list(sizes)}) needs "
                       f"{bound_key} < {float(limit):g} at n = {n_bits}, f = {f}")
    if bound_key == "logits_bound":
        # the index path of a fused node compares index differences below the row length (ring integers)
        sizes = cm.tournament_node_sizes(row_len, k) if row_len is not None else (k,)
        big = [g for g in sizes if g >= 3]
        if big:
            if row_len is None:
                return False, f"{KARY_DETAIL}: {kind} declares no shape, the index bound cannot be checked"
            worst = min(kary_limit_log2(n_bits, g) for g in big)
            if not (row_len - 1) < (1 << worst):
                return False, (f"{KARY_DETAIL}: {kind} index range {row_len - 1} reaches the fused-node bound "
                               f"2^{worst} at n = {n_bits}")
    return True, "OK"


def zt_log2_eps_per_element(t: int, words: int = ZT_WORDS_PER_R2F_ELEMENT) -> float:
    """log2 of the statistical term of one ring_to_field element: words * t / 2^128 (paper, Section 6, loss)."""
    t, words = int(t), int(words)
    if t < 3 or words < 1:
        raise ValueError(f"ZT_PARAM: need t >= 3 and words >= 1, got t={t}, words={words}")
    return math.log2(words * t) - 128.0


# admission ------------------------------------------------------------------
def admit_explain(plan_tape: Sequence[Mapping[str, Any]], profile: str, state: Optional["SessionState"],
                  budget: Any = None, contract: Any = None) -> Tuple[bool, str, str]:
    """(ok, code, detail) following the admission order construction -> profile -> types -> resources -> cost."""
    tape = list(plan_tape or [])
    if not tape:
        return False, "NOT_ADMITTED_COMPONENT", "empty plan tape"
    try:
        ledger = load_proof_ledger()
    except (OSError, ValueError) as exc:
        return False, "NOT_ADMITTED_COMPONENT", f"proof ledger unreadable: {exc}"
    # 1. construction / evidence
    for i, e in enumerate(tape):
        kind = str(e.get("kind", ""))
        if kind in typecheck.INPUT_KINDS:
            continue
        comp = REGISTRY.get(kind)
        if comp is None:
            return False, "NOT_ADMITTED_COMPONENT", f"entry #{i}: {kind!r} is not a registered component"
        if comp.status != STATUS_ADMITTED:
            return False, "NOT_ADMITTED_COMPONENT", f"entry #{i}: {kind!r} has status {comp.status}"
        led = ledger.get(comp.proof_ledger_id)
        if led is None:
            return False, "NOT_ADMITTED_COMPONENT", f"entry #{i}: {kind!r} has no proof ledger entry"
        if str(led.get("status", STATUS_ADMITTED)) != STATUS_ADMITTED:
            return False, "NOT_ADMITTED_COMPONENT", f"entry #{i}: ledger status of {kind!r} is {led.get('status')}"
    # 2. profile
    for i, e in enumerate(tape):
        kind = str(e.get("kind", ""))
        if kind in typecheck.INPUT_KINDS:
            continue
        comp = REGISTRY[kind]
        if profile not in comp.profiles:
            return False, "PROFILE_MISMATCH", (f"entry #{i}: {kind!r} is admitted under {list(comp.profiles)}, "
                                               f"not {profile!r}")
    # 3. types, numerics, budgets
    try:
        typecheck.check_tape(tape)
    except AdmissionError as exc:
        return False, exc.code, str(exc)
    if state is not None:
        for i, e in enumerate(tape):
            h = e.get("contract_hash")
            if h is not None and str(h) != str(state.contract_hash):
                return False, "CONTRACT_MISMATCH", f"entry #{i} declares contract {h!r}, session has {state.contract_hash!r}"
    numeric_contract = resolve_contract(state, contract)
    for i, e in enumerate(tape):
        ok_k, detail_k = kary_site_check(e, numeric_contract)
        if not ok_k:
            return False, "BUDGET_EXCEEDED", f"{detail_k} (entry #{i})"
    charges: List[Tuple[float, int, str]] = []
    for i, e in enumerate(tape):
        kind = str(e.get("kind", ""))
        term = e.get("log2_eps")
        calls = int(e.get("calls", 1))
        if term is None and kind in REGISTRY:
            params = REGISTRY[kind].params
            term = params.get("log2_eps_per_call")
            if term is None and "zt_words_per_element" in params:
                shape = e.get("shape")
                if shape is None or len(shape) < 2:
                    return False, "BUDGET_EXCEEDED", (f"{ZT_DETAIL}: entry #{i} {kind!r} declares no shape, its "
                                                      "Z_t statistical term cannot be counted")
                term = zt_log2_eps_per_element(int(e.get("t", params["t"])), int(params["zt_words_per_element"]))
                calls = int(shape[0]) * int(shape[1]) * int(e.get("count", 1))
                if calls < 1:
                    return False, "BUDGET_EXCEEDED", f"{ZT_DETAIL}: entry #{i} {kind!r} declares an empty shape"
        if term is not None:
            charges.append((float(term), calls, f"#{i}:{kind}"))
    if charges:
        if budget is None:
            return False, "BUDGET_EXCEEDED", "a route declares a statistical term but no session budget was provided"
        fits = getattr(budget, "would_fit", None)
        if fits is not None:
            for term, calls, label in charges:
                if not fits(term, calls):
                    return False, "BUDGET_EXCEEDED", f"{label}: {calls} x 2^{term} exceeds the remaining budget"
    # 4. resources
    if state is not None:
        want = state.object_version()
        for i, e in enumerate(tape):
            obj = e.get("object")
            if obj is None:
                continue
            have = state.installed.get(str(obj))
            if have is None:
                continue                      # not installed yet: installed transactionally before execution
            if not state.is_installed(str(obj)):
                return False, "OBJECT_VERSION_MISMATCH", (f"entry #{i}: object {obj!r} installed at {dict(have)}, "
                                                          f"session requires {want}")
            declared = e.get("object_version")
            if declared is not None and any(dict(declared).get(k) != want[k] for k in want):
                return False, "OBJECT_VERSION_MISMATCH", f"entry #{i}: declared version {declared} differs from {want}"
        if state.stock:
            need = tape_material(tape, contract)
            for t, b in need.items():
                have_b = int(state.stock.get(t, (0, 0))[1])
                if have_b < b:
                    return False, "INSUFFICIENT_MATERIAL", f"material {t}: need {b} B, stock {have_b} B"
        limit = getattr(state, "mem_limit_bytes", None)
        if limit is not None:
            mem = sum(int(e.get("mem_bytes", 0)) for e in tape)
            if mem > int(limit):
                return False, "MEMORY_INFEASIBLE", f"declared peak {mem} B exceeds limit {int(limit)} B"
    # budgets are charged only once every check has passed
    for term, calls, label in charges:
        if not budget.charge(term, calls):
            return False, "BUDGET_EXCEEDED", f"{label}: charge of {calls} x 2^{term} refused"
    # 5. cost: ranking only
    return True, "OK", f"{len(tape)} entries admitted under {profile}"


def admit(plan_tape: Sequence[Mapping[str, Any]], profile: str, state: Optional["SessionState"],
          budget: Any = None, contract: Any = None) -> Tuple[bool, str]:
    """``(ok, reason_code)``; the code is one of REASON_CODES or "OK".

    ``contract`` (optional) is the NumericContract of the numeric checks; by
    default it is resolved from the session contract hash (:func:`resolve_contract`).
    """
    ok, code, _ = admit_explain(plan_tape, profile, state, budget, contract)
    return ok, code


# default GPT-2 plan tape ------------------------------------------------------
def default_gpt2_tape(plan: Mapping[str, Any], n_layers: int = 12, shapes: Optional[Mapping[str, Any]] = None,
                      contract_hash: Optional[str] = None) -> List[dict]:
    """One decode epoch of SecureGPT2 under ``plan`` as an admission tape (shapes and objects declared).

    Every softmax entry declares ``score_bound`` and the argmax entry declares
    ``logits_bound`` (public operand bounds of the k-ary magnitude check); the values come
    from ``shapes["score_bound"]`` / ``shapes["logits_bound"]`` when present
    and are the documented defaults otherwise.
    """
    sh = dict(shapes) if shapes is not None else {}
    score_bound = sh.get("score_bound")
    score_bound = DEFAULT_SCORE_BOUND if score_bound is None else score_bound
    logits_bound = sh.get("logits_bound")
    logits_bound = DEFAULT_LOGITS_BOUND if logits_bound is None else logits_bound
    n = int(sh.get("n", 1))
    T = int(sh.get("T", 32))
    d = int(sh.get("d_model", 768))
    f = int(sh.get("d_ff", 3072))
    H = int(sh.get("n_head", 12))
    V = int(sh.get("vocab", 50257))
    dh = d // H
    lin = "linear_wmask" if plan.get("linear", "wmask") == "wmask" else "linear_generic"
    emb = "embedding_wmask" if plan.get("embedding", "wmask") == "wmask" else "embedding_generic"
    k_arg = int(plan.get("argmax_k", 2))
    k_max = int(plan.get("max_k", 2))
    first: dict = {"kind": "input_client_plain", "out": "tok", "args": {}}
    if contract_hash is not None:
        first["contract_hash"] = contract_hash
    tape: List[dict] = [first, {"kind": "share_input", "out": "onehot", "args": {"x": "tok"}, "shape": (n, V, 1)}]
    e_emb: dict = {"kind": emb, "out": "x0", "args": {"x": "onehot"}, "shape": (n, V, d)}
    if emb == "embedding_wmask":
        e_emb["object"] = "wte"
    tape.append(e_emb)
    cur = "x0"

    def linear(out: str, x: str, shape: Tuple[int, int, int], obj: str) -> dict:
        e: dict = {"kind": lin, "out": out, "args": {"x": x}, "shape": shape}
        if lin == "linear_wmask":
            e["object"] = obj
        return e

    for li in range(int(n_layers)):
        tape.append({"kind": "layernorm_rows", "out": f"ln1_{li}", "args": {"x": cur}, "shape": (n, d, d)})
        tape.append(linear(f"qkv_{li}", f"ln1_{li}", (n, d, 3 * d), f"h{li}.qkv"))
        tape.append({"kind": "matmul_shared", "out": f"s_{li}", "args": {"x": f"qkv_{li}", "y": f"qkv_{li}"},
                     "shape": (n, dh, T), "count": H})
        tape.append({"kind": f"softmax_rows_k{k_max}", "out": f"p_{li}", "args": {"x": f"s_{li}"}, "shape": (H * n, T, T),
                     "score_bound": score_bound})
        tape.append({"kind": "matmul_shared", "out": f"ctx_{li}", "args": {"x": f"p_{li}", "y": f"qkv_{li}"},
                     "shape": (n, T, dh), "count": H})
        tape.append(linear(f"o_{li}", f"ctx_{li}", (n, d, d), f"h{li}.out"))
        tape.append({"kind": "layernorm_rows", "out": f"ln2_{li}", "args": {"x": f"o_{li}"}, "shape": (n, d, d)})
        tape.append(linear(f"fc_{li}", f"ln2_{li}", (n, d, f), f"h{li}.fc"))
        tape.append({"kind": "gelu_clip12", "out": f"a_{li}", "args": {"x": f"fc_{li}"}, "shape": (n, f, f)})
        tape.append(linear(f"y_{li}", f"a_{li}", (n, f, d), f"h{li}.proj"))
        cur = f"y_{li}"
    tape.append({"kind": "layernorm_rows", "out": "lnf", "args": {"x": cur}, "shape": (1, d, d)})
    tape.append({"kind": "linear_wmask", "out": "logits", "args": {"x": "lnf"}, "shape": (1, d, V), "object": "lm_head"})
    tape.append({"kind": f"argmax_k{k_arg}", "out": "idx", "args": {"x": "logits"}, "shape": (1, V, 1),
                 "logits_bound": logits_bound})
    tape.append({"kind": "release_final", "out": "token", "args": {"x": "idx"}, "recipient": 1, "count": 1})
    return tape


__all__ = [
    "PROFILE_ID", "PROFILES_SH", "STATUS_ADMITTED", "STATUS_UNCERTIFIED_DISABLED", "STATUS_UNAVAILABLE",
    "STATUS_TEST_ONLY", "STATUSES", "REASON_CODES", "LEDGER_PATH", "REGISTRY_VERSION", "Component", "REGISTRY",
    "export_registry", "load_proof_ledger", "component_admissible", "entry_material", "tape_material",
    "admit_explain", "admit", "default_gpt2_tape",
    "DEFAULT_LOGITS_BOUND", "DEFAULT_SCORE_BOUND", "KARY_DETAIL", "KARY_KINDS", "ZT_WORDS_PER_R2F_ELEMENT",
    "ZT_DETAIL", "resolve_contract", "kary_limit_log2", "kary_bound_limit", "kary_site_check",
    "zt_log2_eps_per_element",
]
