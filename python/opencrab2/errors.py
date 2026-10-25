"""Typed exceptions for opencrab2.

Every exception message starts with a stable upper-case code followed by a
colon, for example ``"CONTRACT_VIOLATION: value out of range"``. The code is
also available as the ``.code`` attribute. A caller may pass an explicit code
(``ContractViolation("value out of range", code="TRUNC_PARAMS")``) or a
message that already starts with a code (``ContractViolation("TRUNC_PARAMS: f
must satisfy 1 <= f <= L-1")``); in both cases the resulting message starts
with the code. When neither is given the class default code is prepended.

What is tested: code extraction, default codes, prefix invariant
(tests/test_ring.py). Design limit: codes are plain strings, not an enum, so
that groups writing other modules can introduce new codes without editing
this file.
"""

from __future__ import annotations

import re
from typing import Optional

_CODE_RE = re.compile(r"^([A-Z][A-Z0-9_]*):")


class OpenCrabError(Exception):
    """Base class. ``.code`` is the stable upper-case code of the message."""

    default_code: str = "OPENCRAB_ERROR"

    def __init__(self, message: str = "", code: Optional[str] = None) -> None:
        message = str(message)
        m = _CODE_RE.match(message)
        if code is not None:
            if not _CODE_RE.match(code + ":"):
                raise ValueError("BAD_ERROR_CODE: error codes are upper-case identifiers")
            if m is None or m.group(1) != code:
                message = f"{code}: {message}" if message else f"{code}:"
        elif m is not None:
            code = m.group(1)
        else:
            code = self.default_code
            message = f"{code}: {message}" if message else f"{code}:"
        self.code: str = code
        super().__init__(message)

    def __str__(self) -> str:
        return self.args[0] if self.args else f"{self.code}:"


class ContractViolation(OpenCrabError):
    """Numeric contract or parameter domain violated (no silent clamp)."""

    default_code = "CONTRACT_VIOLATION"


class MaterialError(OpenCrabError):
    """Correlated-randomness material problem (base of the three below)."""

    default_code = "MATERIAL_ERROR"


class MaterialReuse(MaterialError):
    """A one-time material id was presented twice."""

    default_code = "MATERIAL_REUSE"


class MaterialStall(MaterialError):
    """Material inventory exhausted for the requested type."""

    default_code = "MATERIAL_STALL"


class MaterialVersionMismatch(MaterialError):
    """Bundle binding tuple differs from the session binding."""

    default_code = "MATERIAL_VERSION_MISMATCH"


class AdmissionError(OpenCrabError):
    """Registry refused a plan; ``.code`` is one of the admission reason codes."""

    default_code = "NOT_ADMITTED_COMPONENT"


class UnauthorizedRelease(OpenCrabError):
    """A reveal to a party not authorized by the ReleasePolicy."""

    default_code = "UNAUTHORIZED_RELEASE"


class ChannelError(OpenCrabError):
    """Transport failure or malformed frame."""

    default_code = "CHANNEL_ERROR"


class TelemetryError(OpenCrabError):
    """Rejected telemetry frame (bad tag, replay, out of order)."""

    default_code = "TELEMETRY_ERROR"


class LedgerError(OpenCrabError):
    """Material ledger invariant violated or journal unreadable."""

    default_code = "LEDGER_ERROR"


class BridgeError(OpenCrabError):
    """HE bridge parameter or noise-bound failure."""

    default_code = "BRIDGE_ERROR"


__all__ = [
    "OpenCrabError",
    "ContractViolation",
    "MaterialError",
    "MaterialReuse",
    "MaterialStall",
    "MaterialVersionMismatch",
    "AdmissionError",
    "UnauthorizedRelease",
    "ChannelError",
    "TelemetryError",
    "LedgerError",
    "BridgeError",
]
