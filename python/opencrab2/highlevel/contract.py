"""Numeric contract (paper, Section 2.1, transformer operators and fixed point).

A :class:`NumericContract` is the versioned description of the discrete
fixed-point functionality every realization must implement: ring
width, fixed-point scale, logical range, comparison domain, truncation modes,
activation and transcendental gadget parameters, and the tie rule of argmax.
It is a frozen dataclass; its identity is ``contract_hash()``, the SHA-256 of
the canonical JSON (sorted keys, no spaces) of all fields. Dealer bundles,
ledgers, evidence records and telemetry all carry that hash so that results
produced under different contracts are never merged.

What is tested (tests/test_contract.py): field defaults, hash determinism and
sensitivity to every field, the two named contracts, ``to_json`` round trip,
validation refusals, and that the SmolLM contract differs from the default
only in the documented fields. Design limit: the hash binds the parameter
set, not the correctness of any gadget that implements it; that is the job of
the proof ledger (``highlevel/proof_ledger.json``, checked by ``registry``).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict

from opencrab2.errors import ContractViolation


def canonical_json_bytes(obj: Any) -> bytes:
    """Canonical JSON: sorted keys, no spaces, UTF-8 (the same form as ``lowlevel.hmac_auth.canonical_json``)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class NumericContract:
    """Versioned fixed-point functionality (the same field list in Python and C++).

    Field semantics:

    * ``ring_bits``: 64, Z_{2^64} as numpy uint64 two's complement.
    * ``frac_bits``: f, scale 2^f.
    * ``logical_bits``: l, every encoded value satisfies |v| < 2^(l-1) ring units.
    * ``dcf_domain_bits``: n, compared values satisfy |x| < 2^(n-1) ring units.
    * ``gelu`` / ``silu``: names of the clipped polynomial contracts (approx.py).
    * ``exp_*``: second-order base, r squarings, clamp Z, extra scale bits.
    * ``recip_*``: K thresholds, polynomial degree, Newton steps.
    * ``rsqrt_*``: j0, m thresholds, degree, Newton steps.
    * ``layernorm_extra_bits``: extra output bits kept by LayerNorm/RMSNorm.
    * ``argmax_tie``: ``lowest_index`` for every k-ary realization.
    * ``trunc_products``: ``local_prob`` (SecureML local truncation of
      contract-bounded Beaver products).
    * ``trunc_square_ln``: ``faithful`` (the LayerNorm square uses the faithful gate).
    * ``activation_semantics``: ``erf_gelu`` (GPT-2 ``gelu_new`` is NOT used; disclosed).
    """

    ring_bits: int = 64
    frac_bits: int = 16
    logical_bits: int = 40
    dcf_domain_bits: int = 32
    name: str = "NC-v2-f16-l40"
    gelu: str = "gelu_clip12"
    silu: str = "silu_clip12"
    exp_order: int = 2
    exp_r: int = 7
    exp_Z: int = 16
    exp_extra_bits: int = 4
    recip_K: int = 11
    recip_degree: int = 5
    recip_newton: int = 2
    rsqrt_j0: int = -7
    rsqrt_m: int = 15
    rsqrt_degree: int = 5
    rsqrt_newton: int = 2
    layernorm_extra_bits: int = 4
    argmax_tie: str = "lowest_index"
    trunc_products: str = "local_prob"
    trunc_square_ln: str = "faithful"
    activation_semantics: str = "erf_gelu"

    def __post_init__(self) -> None:
        # Fail closed on a parameter set that no gadget could implement.
        if self.ring_bits != 64:
            raise ContractViolation(f"CONTRACT_PARAMS: ring_bits must be 64, got {self.ring_bits}")
        if not (1 <= self.frac_bits <= 62):
            raise ContractViolation(f"CONTRACT_PARAMS: frac_bits must be in [1, 62], got {self.frac_bits}")
        if not (self.frac_bits + 1 <= self.logical_bits <= 63):
            raise ContractViolation(
                f"CONTRACT_PARAMS: logical_bits must satisfy frac_bits+1 <= l <= 63, got {self.logical_bits}")
        if not (2 <= self.dcf_domain_bits <= 64):
            raise ContractViolation(
                f"CONTRACT_PARAMS: dcf_domain_bits must be in [2, 64], got {self.dcf_domain_bits}")
        if self.exp_order not in (1, 2):
            raise ContractViolation(f"CONTRACT_PARAMS: exp_order must be 1 or 2, got {self.exp_order}")
        if self.exp_r < 1 or self.exp_r > self.frac_bits:
            raise ContractViolation(
                f"CONTRACT_PARAMS: exp_r must satisfy 1 <= r <= frac_bits, got {self.exp_r}")
        for fld in ("exp_Z", "exp_extra_bits", "recip_K", "recip_degree", "recip_newton",
                    "rsqrt_m", "rsqrt_degree", "rsqrt_newton", "layernorm_extra_bits"):
            v = getattr(self, fld)
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                raise ContractViolation(f"CONTRACT_PARAMS: {fld} must be a non-negative int, got {v!r}")
        if self.argmax_tie != "lowest_index":
            raise ContractViolation(f"CONTRACT_PARAMS: argmax_tie must be 'lowest_index', got {self.argmax_tie!r}")
        if self.trunc_products not in ("local_prob", "faithful"):
            raise ContractViolation(f"CONTRACT_PARAMS: trunc_products {self.trunc_products!r} unknown")
        if self.trunc_square_ln not in ("local_prob", "faithful"):
            raise ContractViolation(f"CONTRACT_PARAMS: trunc_square_ln {self.trunc_square_ln!r} unknown")
        if self.activation_semantics != "erf_gelu":
            raise ContractViolation(
                f"CONTRACT_PARAMS: activation_semantics must be 'erf_gelu', got {self.activation_semantics!r}")
        if not isinstance(self.name, str) or not self.name:
            raise ContractViolation("CONTRACT_PARAMS: name must be a non-empty string")

    # -- derived quantities --------------------------------------------------
    @property
    def scale(self) -> int:
        """Fixed-point scale 2^f (exact Python int)."""
        return 1 << self.frac_bits

    @property
    def logical_bound(self) -> int:
        """2^(l-1): encoded values satisfy |v| < logical_bound."""
        return 1 << (self.logical_bits - 1)

    @property
    def compare_bound_real(self) -> float:
        """Largest |x_real| admissible for a DCF comparison: 2^(n-1-f)."""
        return float(2 ** (self.dcf_domain_bits - 1 - self.frac_bits))

    def softmax_extra_bits(self, n: int) -> int:
        """min(ceil(log2 n), 10) extra output bits of the softmax reciprocal."""
        n = int(n)
        if n < 1:
            raise ContractViolation(f"CONTRACT_PARAMS: softmax row length must be >= 1, got {n}")
        return min((n - 1).bit_length(), 10)

    # -- identity --------------------------------------------------------------
    def to_json(self) -> Dict[str, Any]:
        """All fields as a plain dict (what the hash is computed over)."""
        return dataclasses.asdict(self)

    def contract_hash(self) -> str:
        """SHA-256 hex of the canonical JSON of all fields."""
        return hashlib.sha256(canonical_json_bytes(self.to_json())).hexdigest()

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "NumericContract":
        """Inverse of :meth:`to_json`; unknown or missing keys are refused."""
        if not isinstance(d, dict):
            raise ContractViolation("CONTRACT_JSON: expected a dict")
        names = {f.name for f in dataclasses.fields(NumericContract)}
        extra = set(d) - names
        missing = names - set(d)
        if extra or missing:
            raise ContractViolation(
                f"CONTRACT_JSON: field set mismatch (extra={sorted(extra)}, missing={sorted(missing)})")
        return NumericContract(**d)

    def replace(self, **changes: Any) -> "NumericContract":
        """A new contract with some fields changed (hash changes accordingly)."""
        return dataclasses.replace(self, **changes)


DEFAULT_CONTRACT: NumericContract = NumericContract()

# Model-specific extension for SmolLM2: l = 46, rsqrt m = 16, separate hash.
# The two-stage scaling v = (V*s1)*s1 of the rsqrt gadget is selected by the
# gadget from ``name``; results under this contract are never merged with the
# default contract.
SMOLLM_CONTRACT: NumericContract = NumericContract(
    logical_bits=46, rsqrt_m=16, name="NC-v2-f16-l46-smollm")

CONTRACTS: Dict[str, NumericContract] = {
    DEFAULT_CONTRACT.name: DEFAULT_CONTRACT,
    SMOLLM_CONTRACT.name: SMOLLM_CONTRACT,
}


def contract_by_name(name: str) -> NumericContract:
    """Look up one of the named contracts (used by CLIs); unknown names fail closed."""
    try:
        return CONTRACTS[name]
    except KeyError:
        raise ContractViolation(
            f"CONTRACT_UNKNOWN: no contract named {name!r}; known: {sorted(CONTRACTS)}") from None


__all__ = [
    "NumericContract", "DEFAULT_CONTRACT", "SMOLLM_CONTRACT", "CONTRACTS",
    "contract_by_name", "canonical_json_bytes",
]
