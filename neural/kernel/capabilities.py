"""Machine-readable stability contract for Neural package capabilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum


class CapabilityStatus(str, Enum):
    """Compatibility level for one public capability."""

    STABLE = "stable"
    EXPERIMENTAL = "experimental"
    DEPRECATED = "deprecated"


@dataclass(frozen=True, slots=True)
class Capability:
    """One capability and its supported installation path."""

    name: str
    module: str
    status: CapabilityStatus
    extra: str | None
    summary: str
    replacement: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        """Return a JSON-compatible representation."""
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


CAPABILITIES: tuple[Capability, ...] = (
    Capability(
        name="kernel.normalization",
        module="neural.kernel",
        status=CapabilityStatus.STABLE,
        extra=None,
        summary="JSON-compatible market, quote, order, position, and policy types.",
    ),
    Capability(
        name="kernel.replay",
        module="neural.kernel",
        status=CapabilityStatus.STABLE,
        extra=None,
        summary="Dependency-free deterministic fixture replay and digest.",
    ),
    Capability(
        name="cli.diagnostics",
        module="neural.cli",
        status=CapabilityStatus.STABLE,
        extra=None,
        summary="Machine-readable doctor, capabilities, and replay demo commands.",
    ),
    Capability(
        name="auth.kalshi",
        module="neural.auth",
        status=CapabilityStatus.EXPERIMENTAL,
        extra="trading",
        summary="Kalshi authentication and signed HTTP clients.",
    ),
    Capability(
        name="data_collection",
        module="neural.data_collection",
        status=CapabilityStatus.EXPERIMENTAL,
        extra="analysis",
        summary="Provider-specific market collection and tabular normalization.",
    ),
    Capability(
        name="trading.paper",
        module="neural.trading",
        status=CapabilityStatus.EXPERIMENTAL,
        extra="trading",
        summary="Paper portfolio and venue adapter utilities.",
    ),
    Capability(
        name="analysis",
        module="neural.analysis",
        status=CapabilityStatus.EXPERIMENTAL,
        extra="analysis",
        summary="Strategy, risk-sizing, and backtesting utilities.",
    ),
    Capability(
        name="analysis.sentiment",
        module="neural.analysis.sentiment",
        status=CapabilityStatus.DEPRECATED,
        extra="sentiment",
        summary="Legacy sentiment adapters retained for compatibility.",
        replacement="Evidence-backed analysis in the Vaticor product workflow.",
    ),
    Capability(
        name="deployment",
        module="neural.deployment",
        status=CapabilityStatus.DEPRECATED,
        extra="deployment",
        summary="Legacy generic deployment helpers retained for compatibility.",
        replacement="A separately owned Vaticor runtime provider.",
    ),
    Capability(
        name="trading.fix",
        module="neural.trading.fix",
        status=CapabilityStatus.DEPRECATED,
        extra="fix",
        summary="Experimental FIX helpers retained for compatibility.",
        replacement="Kalshi conformance through the supported adapter boundary.",
    ),
)

_CAPABILITIES_BY_NAME = {capability.name: capability for capability in CAPABILITIES}


def capability_matrix() -> list[dict[str, str | None]]:
    """Return the ordered public capability matrix."""
    return [capability.as_dict() for capability in CAPABILITIES]


def get_capability(name: str) -> Capability:
    """Return one named capability or raise a clear lookup error."""
    try:
        return _CAPABILITIES_BY_NAME[name]
    except KeyError as exc:
        raise KeyError(f"Unknown Neural capability: {name}") from exc
