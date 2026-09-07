"""Versioned, side-effect-free price-rule specifications (NRCL-86).

Validation is not approval, venue compatibility, or permission to trade.
This module intentionally does not change neural.contracts v1 or kernel replay.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, fields
from decimal import Decimal, localcontext
from typing import Any

STRATEGY_VERSION = "1.0.0"
_DECIMAL_PATTERN = r"(?:0|[1-9][0-9]{0,17})(?:\.[0-9]{1,18})?"
_MARKET_PATTERN = r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}"
_DECIMAL_FIELDS = ("entry_price", "exit_price", "quantity", "max_position", "max_exposure_usd")


def _decimal(value: Any, name: str) -> Decimal:
    if not isinstance(value, (str, Decimal)):
        raise ValueError(f"{name}: use a decimal string or Decimal, never float/int/bool")
    if isinstance(value, Decimal):
        if not value.is_finite() or value.is_signed():
            raise ValueError(f"{name}: expected a finite nonnegative decimal")
        # Bound expansion before formatting values such as Decimal('1e999999').
        if value.adjusted() > 17 or value.as_tuple().exponent < -18:
            raise ValueError(f"{name}: maximum 18 integer and 18 fractional digits")
        value = format(value, "f")
    if re.fullmatch(_DECIMAL_PATTERN, value) is None:
        raise ValueError(f"{name}: expected plain decimal, at most 18 digits on each side")
    return Decimal(value)


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


@dataclass(frozen=True)
class StrategySpec:
    """Buy the selected outcome at ask <= entry; exit at bid >= exit.

    A runner may buy the fixed quantity while below both risk caps, and sell
    up to that quantity from existing holdings. No shorting, model calls,
    automatic venue selection, or authorization is represented here.
    """

    venue: str
    market_id: str
    outcome: str
    entry_price: Decimal
    exit_price: Decimal
    quantity: Decimal
    max_position: Decimal
    max_exposure_usd: Decimal
    schema_version: str = STRATEGY_VERSION
    kind: str = "price_rule"

    def __post_init__(self) -> None:
        if self.schema_version != STRATEGY_VERSION or self.kind != "price_rule":
            raise ValueError("unsupported strategy version or kind")
        if self.venue not in ("kalshi", "polymarket_us"):
            raise ValueError("venue must be kalshi or polymarket_us")
        if self.outcome not in ("yes", "no"):
            raise ValueError("outcome must be yes or no")
        if not isinstance(self.market_id, str) or not re.fullmatch(_MARKET_PATTERN, self.market_id):
            raise ValueError("market_id must be a nonempty native market identifier")
        for name in _DECIMAL_FIELDS:
            object.__setattr__(self, name, _decimal(getattr(self, name), name))
        if not Decimal(0) < self.entry_price < self.exit_price < Decimal(1):
            raise ValueError("prices must satisfy 0 < entry_price < exit_price < 1")
        if not Decimal(0) < self.quantity <= self.max_position:
            raise ValueError("quantity must be positive and no greater than max_position")
        # Multiplication must not depend on a caller's ambient decimal context.
        with localcontext() as context:
            context.prec = 80
            minimum_exposure = self.quantity * self.entry_price
        if self.max_exposure_usd < minimum_exposure:
            raise ValueError("max_exposure_usd cannot cover one entry order before fees")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> StrategySpec:
        """Validate wire input, including required fields and strict string decimals."""
        expected = {field.name for field in fields(cls)}
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("strategy must contain exactly the documented required fields")
        if any(not isinstance(value, str) for value in payload.values()):
            raise ValueError("all strategy wire fields must be strings")
        return cls(**payload)

    @classmethod
    def from_json(cls, text: str) -> StrategySpec:
        def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate strategy field: {key}")
                result[key] = value
            return result

        return cls.from_dict(json.loads(text, object_pairs_hook=unique_pairs))

    def to_dict(self) -> dict[str, str]:
        return {
            field.name: (
                _decimal_text(getattr(self, field.name))
                if field.name in _DECIMAL_FIELDS
                else getattr(self, field.name)
            )
            for field in fields(self)
        }

    def to_json(self) -> str:
        """Canonical UTF-8 JSON: sorted keys, normalized decimals, no whitespace."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def version_id(self) -> str:
        """Content identity, not a signature or evidence of human approval."""
        return "sha256:" + hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


def strategy_schema() -> dict[str, Any]:
    """Export the wire-shape schema; StrategySpec also enforces cross-field rules."""
    properties = {
        "schema_version": {"const": STRATEGY_VERSION},
        "kind": {"const": "price_rule"},
        "venue": {"enum": ["kalshi", "polymarket_us"]},
        "market_id": {"type": "string", "pattern": "^" + _MARKET_PATTERN + "$(?![\\s\\S])"},
        "outcome": {"enum": ["yes", "no"]},
        **{
            name: {"type": "string", "pattern": "^" + _DECIMAL_PATTERN + "$(?![\\s\\S])"}
            for name in _DECIMAL_FIELDS
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Neural StrategySpec " + STRATEGY_VERSION,
        "description": "Wire shape only; validate cross-field rules with StrategySpec before use.",
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }
