"""NRCL-99: identities, rule provenance and false-equivalence boundaries."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from neural.sports import RuleEvidence, SettlementRules, SportsMarket, compare_sports_markets

FIXTURE = json.loads(
    (Path(__file__).parents[2] / "examples/sports-matching-fixtures.json").read_text()
)


def market(name="synthetic_kalshi"):
    return SportsMarket.from_dict(FIXTURE["markets"][name])


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda case: case["name"])
def test_customer_comparison_cases(case):
    left = market(case["left"])
    right = SportsMarket.from_dict({**FIXTURE["markets"][case["right"]], **case["right_overrides"]})
    result = compare_sports_markets(left, right)
    assert result.status == case["expected"]
    assert result.proposition == case["expected_proposition"]
    if case["expected_field"]:
        assert case["expected_field"] in (*result.differences, *result.unknowns)
    assert compare_sports_markets(right, left) == result
    if result.proposition != "compatible":
        assert result.settlement == "unknown"
        assert not any(field.startswith("rules:") for field in result.differences)


def test_wire_round_trip_retains_raw_ids_and_rule_provenance():
    for payload in FIXTURE["markets"].values():
        parsed = SportsMarket.from_dict(payload)
        assert parsed.to_dict() == payload
        assert SportsMarket.from_dict(json.loads(json.dumps(parsed.to_dict()))) == parsed
    source = FIXTURE["synthetic_rule_source"]
    assert hashlib.sha256(source["canonical_json"].encode()).hexdigest() == source["sha256"]
    assert {term.source_sha256 for term in market().rules.terms} == {source["sha256"]}


@pytest.mark.parametrize(
    "field",
    [
        "canonical_event_id",
        "home_team_id",
        "away_team_id",
        "outcome_team_id",
        "game_date",
        "game_number",
        "period",
    ],
)
def test_missing_identity_on_both_sides_is_unknown(field):
    original = market()
    changes = {field: None}
    if field == "home_team_id":
        changes["outcome_team_id"] = None
    missing = replace(original, **changes)
    result = compare_sports_markets(missing, missing)
    assert result.status == "unknown"
    assert field in result.unknowns


def test_outcome_and_event_identity_cannot_match_by_team_names_only():
    original = market()
    for changed in (
        replace(original, outcome_team_id=original.away_team_id),
        replace(original, canonical_event_id="fixture:other-event"),
        replace(original, game_date="2026-09-11"),
        replace(original, period="first_5_innings"),
    ):
        assert compare_sports_markets(original, changed).status == "different"
    partial = replace(original, period="first_5_innings")
    assert compare_sports_markets(partial, partial).status == "different"


@pytest.mark.parametrize("scope", ["series", "guidance"])
def test_matching_policy_values_with_partial_source_scope_stay_unknown(scope):
    original = market()
    rules = SettlementRules(
        tuple(replace(term, scope=scope) for term in original.rules.terms), True
    )
    candidate = replace(original, rules=rules)
    assert compare_sports_markets(candidate, candidate).status == "unknown"


def test_missing_rule_hash_and_incomplete_review_cannot_claim_compatibility():
    original = market()
    variants = [
        SettlementRules(original.rules.terms, False),
        SettlementRules(original.rules.terms[:-1], True),
        SettlementRules(
            tuple(replace(term, source_sha256=None) for term in original.rules.terms), True
        ),
    ]
    for rules in variants:
        candidate = replace(original, rules=rules)
        assert compare_sports_markets(candidate, candidate).status == "unknown"
    real = replace(
        original,
        rules=SettlementRules(
            tuple(replace(term, scope="listed_contract") for term in original.rules.terms), True
        ),
    )
    assert compare_sports_markets(original, real).status == "unknown"


@pytest.mark.parametrize("field", ["event_id", "market_id", "outcome_id", "canonical_event_id"])
def test_fixture_rules_cannot_be_copied_onto_real_market_identities(field):
    payload = market().to_dict()
    payload[field] = "mlb:real-market-identity"
    with pytest.raises(ValueError, match="fixture-prefixed"):
        SportsMarket.from_dict(payload)


@pytest.mark.parametrize(
    "source_url",
    [
        "https://exa mple.com/path",
        "https://./x",
        "https://example.com:notaport/path",
        "https://example.com:65536/path",
        "https://example.com:/path",
        "https://-example.com/path",
        "https://example..com/path",
        "https://@example.com/path",
        "https://[v1.foo]/rules",
    ],
)
def test_malformed_source_authorities_are_rejected(source_url):
    with pytest.raises(ValueError):
        replace(market().rules.terms[0], source_url=source_url)


@pytest.mark.parametrize(
    "source_url",
    [
        "https://example.com:443/path",
        "https://example.com./path",
        "https://[2001:db8::1]/path",
        "https://bücher.example/path",
    ],
)
def test_valid_source_authorities_remain_supported(source_url):
    assert replace(market().rules.terms[0], source_url=source_url).source_url == source_url


def test_lone_surrogates_cannot_escape_the_wire_boundary():
    for field in (
        "event_id",
        "market_id",
        "outcome_id",
        "raw_home_team_id",
        "raw_away_team_id",
        "canonical_event_id",
        "home_team_id",
        "away_team_id",
        "outcome_team_id",
        "period",
    ):
        for surrogate in ("\ud800", "\udfff"):
            with pytest.raises(ValueError, match="UTF-8"):
                replace(market(), **{field: "fixture:" + surrogate})
    for field in ("value", "source_url", "retrieved_at"):
        with pytest.raises(ValueError, match="UTF-8"):
            replace(market().rules.terms[0], **{field: "source\ud800"})


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "2.0.0"),
        ("league", "nba"),
        ("venue", "polymarket"),
        ("market_type", "spread"),
        ("game_number", True),
        ("game_number", 0),
        ("game_number", "1"),
        ("game_date", "2026-09-10T00:00:00Z"),
        ("game_date", "2026-02-30"),
        ("outcome_team_id", "mlb:other"),
        ("market_id", ""),
        ("canonical_event_id", " event "),
    ],
)
def test_invalid_wire_identity_fails(field, value):
    payload = market().to_dict()
    payload[field] = value
    with pytest.raises(ValueError):
        SportsMarket.from_dict(payload)


def test_strict_nested_wire_and_rule_evidence():
    payload = market().to_dict()
    with pytest.raises(ValueError):
        SportsMarket.from_dict({**payload, "unknown": True})
    with pytest.raises(ValueError):
        SettlementRules.from_dict({"terms": [], "complete": "true"})
    with pytest.raises(ValueError):
        SettlementRules((market().rules.terms[0], market().rules.terms[0]), True)
    evidence = payload["rules"]["terms"][0]
    for change in (
        {"name": "unmodeled_rule"},
        {"source_sha256": "not-a-hash"},
        {"retrieved_at": "2026-09-10T00:00:00"},
        {"source_url": "http://example.com"},
        {"source_url": "https://user:password@example.com"},
        {"value": ""},
    ):
        with pytest.raises(ValueError):
            RuleEvidence(**{**evidence, **change})
    payload["rules"]["terms"][0]["extra"] = "rejected"
    with pytest.raises(ValueError):
        SportsMarket.from_dict(payload)
