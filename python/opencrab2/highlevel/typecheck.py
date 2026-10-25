"""Domain typing of an op tape.

Domains (one meaning each): ``plain_client``, ``plain_server``, ``A64``
(additive shares over Z_2^64), ``B`` (XOR shares of bits), ``Zt`` (additive
shares over Z_t), ``H_BFV`` (BFV ciphertext under the client key, held by the
server), ``H_CKKS`` (CKKS ciphertext).  ``H_BFV`` and ``H_CKKS`` are distinct
types and nothing casts between them.

A tape is a list of entries ``{"kind": <component or input marker>, "out":
<name>, "args": {"x": <name>, "y": <name>, "inputs": [...]}}``.  Input
markers ``input_client_plain`` / ``input_server_plain`` create values; every
other kind must be a registry component whose ``in_domain`` accepts the
domain of each operand and whose ``out_domain`` types the result.
:func:`infer_domains` returns ``{name: domain}``; :func:`check_tape` raises
``AdmissionError`` with the stable codes:

* ``NOT_ADMITTED_COMPONENT``: the kind is not a registered component.
* ``MISSING_CONVERSION``: the operand domain could be repaired by one
  registered conversion (plain -> A64 by share_input, A64 -> Zt by
  ring_to_field, A64 or Zt -> H_BFV by a2h_exact) that the tape omits.
* ``DOMAIN_MISMATCH``: no conversion exists (for example a revealed plaintext
  fed back into a secure gadget, an H_CKKS value fed to a BFV consumer, or an
  operand no earlier entry produced).

The registry status of a component is NOT checked here (that is admission
step 1 in registry.admit); typing is step 3 of the admission order.

What is tested (tests/test_typecheck.py): the default GPT-2 tape types
end to end with the expected output domains; a missing share_input and a
missing ring_to_field raise MISSING_CONVERSION; an H_CKKS operand on a BFV
consumer, a plaintext fed back, and an undefined operand raise
DOMAIN_MISMATCH; an unknown kind raises NOT_ADMITTED_COMPONENT; every error
message starts with its code.  Design limit: typing is per value name; a
tape that reuses a name shadows the earlier value (the later definition
wins), as in a straight-line program.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from opencrab2.errors import AdmissionError

DOMAINS: Tuple[str, ...] = ("plain_client", "plain_server", "A64", "B", "Zt", "H_BFV", "H_CKKS")
INPUT_KINDS: Dict[str, str] = {"input_client_plain": "plain_client", "input_server_plain": "plain_server"}
# (have, need) -> conversion component that repairs it
CONVERSIONS: Dict[Tuple[str, str], str] = {
    ("plain_client", "A64"): "share_input",
    ("plain_server", "A64"): "share_input",
    ("A64", "Zt"): "ring_to_field",
    ("A64", "H_BFV"): "a2h_exact",
    ("Zt", "H_BFV"): "a2h_exact",
}


def _fields(op: Any, i: int) -> Tuple[str, str, Dict[str, Any]]:
    if not isinstance(op, Mapping):
        raise AdmissionError(f"tape entry #{i} is not a mapping: {type(op).__name__}", code="DOMAIN_MISMATCH")
    if "kind" not in op or "out" not in op:
        raise AdmissionError(f"tape entry #{i} lacks 'kind' or 'out'", code="DOMAIN_MISMATCH")
    return str(op["kind"]), str(op["out"]), dict(op.get("args", {}) or {})


def _operands(args: Mapping[str, Any]) -> List[str]:
    names: List[str] = []
    for key in ("x", "y"):
        if args.get(key) is not None:
            names.append(str(args[key]))
    for extra in args.get("inputs", []) or []:
        names.append(str(extra))
    return names


def split_domains(spec: str) -> Tuple[str, ...]:
    """``"A64|Zt"`` -> ("A64", "Zt")."""
    return tuple(s.strip() for s in str(spec).split("|") if s.strip())


def op_signature(kind: str) -> Tuple[Optional[Tuple[str, ...]], str]:
    """(accepted input domains or None for a creating op, output domain) of a tape kind."""
    if kind in INPUT_KINDS:
        return None, INPUT_KINDS[kind]
    from . import registry as _registry       # lazy: registry imports this module at load time
    comp = _registry.REGISTRY.get(kind)
    if comp is None:
        raise AdmissionError(f"tape kind {kind!r} is not a registered component", code="NOT_ADMITTED_COMPONENT")
    return split_domains(comp.in_domain), str(comp.out_domain)


def infer_domains(tape: Iterable[Any]) -> Dict[str, str]:
    """Walk the tape and return ``{name: domain}``; raise AdmissionError on the first ill-typed operand."""
    domains: Dict[str, str] = {}
    for i, op in enumerate(tape):
        kind, out, args = _fields(op, i)
        accepted, produces = op_signature(kind)
        if accepted is not None:
            names = _operands(args)
            if not names:
                raise AdmissionError(f"entry #{i} {kind!r} names no operand", code="DOMAIN_MISMATCH")
            for name in names:
                have = domains.get(name)
                if have is None:
                    raise AdmissionError(f"entry #{i} {kind!r} consumes {name!r} which no earlier entry produced",
                                         code="DOMAIN_MISMATCH")
                if have in accepted:
                    continue
                for need in accepted:
                    conv = CONVERSIONS.get((have, need))
                    if conv is not None:
                        raise AdmissionError(
                            f"entry #{i} {kind!r} consumes {name!r} in domain {have!r} but requires {need!r}; "
                            f"insert {conv!r} before it", code="MISSING_CONVERSION")
                raise AdmissionError(
                    f"entry #{i} {kind!r} consumes {name!r} in domain {have!r} but requires one of {list(accepted)}",
                    code="DOMAIN_MISMATCH")
        if produces not in DOMAINS:
            raise AdmissionError(f"entry #{i} {kind!r} produces unknown domain {produces!r}", code="DOMAIN_MISMATCH")
        domains[out] = produces
    return domains


def check_tape(tape: Iterable[Any]) -> None:
    """Fail closed: raise AdmissionError (codes above) unless every operand is in an accepted domain."""
    infer_domains(list(tape))


def tape_feasible(tape: Iterable[Any]) -> Tuple[bool, str]:
    """Planner-facing form: ``(True, "OK")`` or ``(False, code)``."""
    try:
        check_tape(tape)
        return True, "OK"
    except AdmissionError as exc:
        return False, exc.code


__all__ = ["DOMAINS", "INPUT_KINDS", "CONVERSIONS", "split_domains", "op_signature", "infer_domains",
           "check_tape", "tape_feasible"]
