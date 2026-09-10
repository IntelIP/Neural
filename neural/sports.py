"""Versioned MLB winner propositions and evidence-aware rule comparison (NRCL-99).

Pure data: no discovery, fuzzy team matching, settlement, or trading authority.
The existing NormalizedMarket.metadata can carry SportsMarket.to_dict().
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, fields
from datetime import date, datetime
from ipaddress import IPv6Address
from typing import Any, Literal
from urllib.parse import urlsplit

SPORTS_VERSION = "1.0.0"
RULE_FIELDS = (
    "winner",
    "extra_innings",
    "forfeit",
    "postponement",
    "cancellation",
    "shortened_game",
    "settlement_source",
    "exceptional_payout",
    "venue_change",
    "replay",
)
Status = Literal["compatible", "different", "unknown"]


def _text(value: Any, name: str, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or not value or value != value.strip() or len(value) > 2048:
        raise ValueError(f"{name}: expected nonempty text without surrounding whitespace")
    if any(ord(char) < 32 for char in value):
        raise ValueError(f"{name}: control characters are forbidden")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name}: text must be valid UTF-8") from exc


def _wire(cls: Any, payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != {field.name for field in fields(cls)}:
        raise ValueError(f"{cls.__name__}: expected exactly the documented fields")
    return dict(payload)


@dataclass(frozen=True)
class RuleEvidence:
    """A reviewed policy identifier, with retained source provenance.

    Values are semantic identifiers assigned by the source reviewer, not raw
    prose or an automatic interpretation. Equal strings alone do not prove law.
    """

    name: str
    value: str
    source_url: str
    source_sha256: str | None
    retrieved_at: str
    scope: str

    def __post_init__(self) -> None:
        if self.name not in RULE_FIELDS:
            raise ValueError("unsupported settlement rule field")
        for name in ("value", "source_url", "retrieved_at"):
            _text(getattr(self, name), name)
        url = urlsplit(self.source_url)
        if (
            url.scheme != "https"
            or not url.hostname
            or url.username is not None
            or url.password is not None
        ):
            raise ValueError("rule source must be an HTTPS URL without credentials")
        try:
            port = url.port  # Reject nonnumeric and out-of-range ports.
            if url.netloc.endswith(":") or (port is not None and port < 1):
                raise ValueError("empty or unusable port")
            host = url.hostname.encode("idna").decode("ascii")
            if url.netloc.startswith("["):
                IPv6Address(host)
            else:
                host = host.removesuffix(".")
                label = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
                if len(host) > 253 or re.fullmatch(rf"{label}(?:\.{label})*", host) is None:
                    raise ValueError("malformed hostname")
        except (ValueError, UnicodeError) as exc:
            raise ValueError("rule source must have a valid hostname and port") from exc
        if self.source_sha256 is not None and (
            not isinstance(self.source_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.source_sha256) is None
        ):
            raise ValueError("source_sha256 must be a lowercase SHA-256 or null")
        try:
            observed = datetime.fromisoformat(self.retrieved_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("retrieved_at must be an ISO timestamp with timezone") from exc
        if observed.tzinfo is None or observed.utcoffset() is None:
            raise ValueError("retrieved_at must be timezone-aware")
        if self.scope not in ("listed_contract", "series", "guidance", "fixture"):
            raise ValueError("unsupported rule source scope")


@dataclass(frozen=True)
class SettlementRules:
    terms: tuple[RuleEvidence, ...] = ()
    complete: bool = False

    def __post_init__(self) -> None:
        if type(self.complete) is not bool:
            raise ValueError("complete must be a boolean")
        if not isinstance(self.terms, tuple) or any(
            not isinstance(term, RuleEvidence) for term in self.terms
        ):
            raise ValueError("terms must be a tuple of RuleEvidence")
        if len({term.name for term in self.terms}) != len(self.terms):
            raise ValueError("duplicate settlement rule field")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SettlementRules:
        wire = _wire(cls, payload)
        if not isinstance(wire["terms"], list):
            raise ValueError("terms must be an array")
        wire["terms"] = tuple(RuleEvidence(**_wire(RuleEvidence, item)) for item in wire["terms"])
        return cls(**wire)


@dataclass(frozen=True)
class SportsMarket:
    """One selected team-wins outcome; canonical mapping is caller-owned.

    game_date is the original official local schedule date, not a UTC date
    derived from a timestamp. A postponed game retains its canonical identity.
    Unknown canonical fields remain null. Raw venue identifiers are preserved.
    """

    venue: str
    event_id: str
    market_id: str
    outcome_id: str
    raw_home_team_id: str | None
    raw_away_team_id: str | None
    canonical_event_id: str | None
    home_team_id: str | None
    away_team_id: str | None
    outcome_team_id: str | None
    game_date: str | None
    game_number: int | None
    period: str | None
    rules: SettlementRules
    schema_version: str = SPORTS_VERSION
    sport: str = "baseball"
    league: str = "mlb"
    market_type: str = "winner"

    def __post_init__(self) -> None:
        if self.schema_version != SPORTS_VERSION:
            raise ValueError("unsupported sports contract version")
        if (self.sport, self.league, self.market_type) != ("baseball", "mlb", "winner"):
            raise ValueError("v1 supports MLB winner propositions only")
        if self.venue not in ("kalshi", "polymarket_us", "novig"):
            raise ValueError("unsupported U.S. venue")
        for name in ("event_id", "market_id", "outcome_id"):
            _text(getattr(self, name), name)
        for name in (
            "raw_home_team_id",
            "raw_away_team_id",
            "canonical_event_id",
            "home_team_id",
            "away_team_id",
            "outcome_team_id",
            "period",
        ):
            _text(getattr(self, name), name, optional=True)
        if self.home_team_id is not None and self.home_team_id == self.away_team_id:
            raise ValueError("home and away teams must differ")
        if self.outcome_team_id is not None and self.outcome_team_id not in (
            self.home_team_id,
            self.away_team_id,
        ):
            raise ValueError("selected outcome must identify a mapped participant")
        if self.game_number is not None and (
            type(self.game_number) is not int or self.game_number not in (1, 2)
        ):
            raise ValueError("game_number must be 1, 2, or null; never infer missing game 1")
        if self.game_date is not None:
            if not isinstance(self.game_date, str) or not re.fullmatch(
                r"\d{4}-\d{2}-\d{2}", self.game_date
            ):
                raise ValueError("game_date must be an ISO local date or null")
            date.fromisoformat(self.game_date)
        if not isinstance(self.rules, SettlementRules):
            raise ValueError("rules must be SettlementRules")
        if any(term.scope == "fixture" for term in self.rules.terms):
            for name in ("event_id", "market_id", "outcome_id", "canonical_event_id"):
                value = getattr(self, name)
                if value is not None and not value.startswith("fixture:"):
                    raise ValueError(f"{name}: fixture rules require fixture-prefixed identities")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SportsMarket:
        wire = _wire(cls, payload)
        wire["rules"] = SettlementRules.from_dict(wire["rules"])
        return cls(**wire)

    def to_dict(self) -> dict[str, Any]:
        wire = asdict(self)
        wire["rules"]["terms"] = [asdict(term) for term in self.rules.terms]
        return wire


@dataclass(frozen=True)
class SportsComparison:
    proposition: Status
    settlement: Status
    differences: tuple[str, ...]
    unknowns: tuple[str, ...]

    @property
    def status(self) -> Status:
        if "different" in (self.proposition, self.settlement):
            return "different"
        if "unknown" in (self.proposition, self.settlement):
            return "unknown"
        return "compatible"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "proposition": self.proposition,
            "settlement": self.settlement,
            "differences": list(self.differences),
            "unknowns": list(self.unknowns),
        }


def compare_sports_markets(left: SportsMarket, right: SportsMarket) -> SportsComparison:
    """Compare the reviewed v1 dimensions, never infer missing facts as equal.

    Settlement is compared only after canonical full-game propositions match.
    A known policy difference wins over other unknown rule fields. Compatibility
    requires complete reviewer coverage, listed-contract sources and snapshots.
    Synthetic fixtures can compare with fixtures, never real contract sources.
    """
    differences: list[str] = []
    unknowns: list[str] = []
    for name in (
        "canonical_event_id",
        "home_team_id",
        "away_team_id",
        "outcome_team_id",
        "game_date",
        "game_number",
        "period",
    ):
        a, b = getattr(left, name), getattr(right, name)
        if a is None or b is None:
            unknowns.append(name)
        elif a != b:
            differences.append(name)
    if any(market.period not in (None, "full_game") for market in (left, right)):
        differences.append("scope:full_game_only")
    proposition: Status = "different" if differences else "unknown" if unknowns else "compatible"
    if proposition != "compatible":
        return SportsComparison(
            proposition, "unknown", tuple(differences), tuple(unknowns + ["rules:not_compared"])
        )

    left_rules = {term.name: term for term in left.rules.terms}
    right_rules = {term.name: term for term in right.rules.terms}
    for name in RULE_FIELDS:
        a_rule, b_rule = left_rules.get(name), right_rules.get(name)
        if a_rule is None or b_rule is None:
            unknowns.append("rules:" + name)
            continue
        if "fixture" in (a_rule.scope, b_rule.scope) and a_rule.scope != b_rule.scope:
            unknowns.append("rules:" + name + ":mixed_fixture")
            continue
        if a_rule.value != b_rule.value:
            differences.append("rules:" + name)
        if any(
            rule.scope not in ("listed_contract", "fixture") or rule.source_sha256 is None
            for rule in (a_rule, b_rule)
        ):
            unknowns.append("rules:" + name + ":provenance")
    if not left.rules.complete or not right.rules.complete:
        unknowns.append("rules:incomplete_review")
    settlement: Status = "different" if differences else "unknown" if unknowns else "compatible"
    return SportsComparison(proposition, settlement, tuple(differences), tuple(unknowns))
