"""Dependency-free deterministic replay primitives."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

DEMO_REPLAY_DIGEST = "569f87947428d4d093425134181b754ca974675aa7c74fc2cb73a8e193f5b4e6"


def _decimal(value: Decimal | float | int | str, *, field: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{field} must be a finite decimal") from exc
    if not parsed.is_finite():
        raise ValueError(f"{field} must be a finite decimal")
    return parsed


def _price(value: Decimal | float | int | str, *, field: str) -> Decimal:
    parsed = _decimal(value, field=field)
    if parsed < 0 or parsed > 1:
        raise ValueError(f"{field} must be between 0 and 1")
    return parsed


def _timestamp(value: str) -> str:
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError("observed_at must be an ISO 8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("observed_at must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _decimal_text(value: Decimal) -> str:
    if value.is_zero():
        return "0"
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


@dataclass(frozen=True, slots=True)
class ReplayEvent:
    """One normalized, immutable market observation."""

    observed_at: str
    market_id: str
    yes_price: Decimal
    no_price: Decimal
    source: str = "fixture"
    volume: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        object.__setattr__(self, "observed_at", _timestamp(self.observed_at))
        object.__setattr__(self, "market_id", self.market_id.strip())
        object.__setattr__(self, "source", self.source.strip())
        object.__setattr__(self, "yes_price", _price(self.yes_price, field="yes_price"))
        object.__setattr__(self, "no_price", _price(self.no_price, field="no_price"))
        object.__setattr__(self, "volume", _decimal(self.volume, field="volume"))
        if not self.market_id:
            raise ValueError("market_id must not be empty")
        if not self.source:
            raise ValueError("source must not be empty")
        if self.volume < 0:
            raise ValueError("volume must be non-negative")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ReplayEvent:
        """Build an event from a JSON-compatible mapping."""
        return cls(
            observed_at=str(value["observed_at"]),
            market_id=str(value["market_id"]),
            yes_price=_price(value["yes_price"], field="yes_price"),
            no_price=_price(value["no_price"], field="no_price"),
            source=str(value.get("source", "fixture")),
            volume=_decimal(value.get("volume", 0), field="volume"),
        )

    def as_dict(self) -> dict[str, str]:
        """Return the canonical JSON-compatible event."""
        return {
            "market_id": self.market_id,
            "no_price": _decimal_text(self.no_price),
            "observed_at": self.observed_at,
            "source": self.source,
            "volume": _decimal_text(self.volume),
            "yes_price": _decimal_text(self.yes_price),
        }


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """Canonical replay output bound to a SHA-256 digest."""

    events: tuple[ReplayEvent, ...]
    snapshot: tuple[ReplayEvent, ...]
    digest: str

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible replay result."""
        return {
            "digest": self.digest,
            "event_count": len(self.events),
            "market_count": len(self.snapshot),
            "snapshot": [event.as_dict() for event in self.snapshot],
        }


def _event_sort_key(event: ReplayEvent) -> tuple[datetime, str, str, str, str, str]:
    payload = event.as_dict()
    return (
        datetime.fromisoformat(payload["observed_at"].replace("Z", "+00:00")),
        payload["source"],
        payload["market_id"],
        payload["yes_price"],
        payload["no_price"],
        payload["volume"],
    )


def replay(events: Iterable[ReplayEvent | Mapping[str, Any]]) -> ReplayResult:
    """Replay observations into a deterministic final market snapshot."""
    normalized = tuple(
        sorted(
            (
                event if isinstance(event, ReplayEvent) else ReplayEvent.from_mapping(event)
                for event in events
            ),
            key=_event_sort_key,
        )
    )
    if not normalized:
        raise ValueError("replay requires at least one event")

    latest: dict[tuple[str, str], ReplayEvent] = {}
    for event in normalized:
        latest[(event.source, event.market_id)] = event
    snapshot = tuple(sorted(latest.values(), key=_event_sort_key))

    payload = {
        "events": [event.as_dict() for event in normalized],
        "snapshot": [event.as_dict() for event in snapshot],
    }
    canonical = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return ReplayResult(
        events=normalized,
        snapshot=snapshot,
        digest=hashlib.sha256(canonical).hexdigest(),
    )


def demo_events() -> tuple[ReplayEvent, ...]:
    """Return a small deterministic fixture with two markets and two rounds."""
    return (
        ReplayEvent(
            observed_at="2026-07-29T12:00:00Z",
            source="kalshi-fixture",
            market_id="KX-DEMO-YES",
            yes_price=Decimal("0.45"),
            no_price=Decimal("0.55"),
            volume=Decimal("10"),
        ),
        ReplayEvent(
            observed_at="2026-07-29T12:01:00Z",
            source="kalshi-fixture",
            market_id="KX-DEMO-YES",
            yes_price=Decimal("0.52"),
            no_price=Decimal("0.48"),
            volume=Decimal("14"),
        ),
        ReplayEvent(
            observed_at="2026-07-29T12:00:00Z",
            source="kalshi-fixture",
            market_id="KX-DEMO-NO",
            yes_price=Decimal("0.61"),
            no_price=Decimal("0.39"),
            volume=Decimal("7"),
        ),
        ReplayEvent(
            observed_at="2026-07-29T12:01:00Z",
            source="kalshi-fixture",
            market_id="KX-DEMO-NO",
            yes_price=Decimal("0.58"),
            no_price=Decimal("0.42"),
            volume=Decimal("9"),
        ),
    )


def run_demo_replay() -> ReplayResult:
    """Run the built-in deterministic replay fixture."""
    result = replay(demo_events())
    if result.digest != DEMO_REPLAY_DIGEST:
        raise RuntimeError("built-in replay fixture digest changed")
    return result
