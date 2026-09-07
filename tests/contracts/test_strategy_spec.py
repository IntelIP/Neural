"""NRCL-86: portable strategy identity, strict validation, and schema parity."""

import json
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal, localcontext
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

from neural.strategy import StrategySpec, strategy_schema


@pytest.fixture
def payload():
    return json.loads((Path(__file__).parents[2] / "examples/strategy-price-rule.json").read_text())


def test_round_trip_schema_and_identity(payload):
    spec = StrategySpec.from_dict(payload)
    assert spec.entry_price == Decimal("0.4")
    assert spec.quantity == Decimal("2.5")
    assert StrategySpec.from_json(spec.to_json()) == spec
    assert spec.version_id.startswith("sha256:")
    assert len(spec.version_id) == 71
    equivalent = dict(reversed(list(payload.items())))
    equivalent.update(entry_price="0.4000", quantity="2.500")
    assert StrategySpec.from_dict(equivalent).version_id == spec.version_id
    schema = strategy_schema()
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)
    Draft202012Validator(schema).validate(spec.to_dict())
    with pytest.raises(FrozenInstanceError):
        spec.quantity = Decimal("3")
    exported = spec.to_dict()
    exported["quantity"] = "3"
    assert spec.quantity == Decimal("2.5")


@pytest.mark.parametrize(
    "field,value",
    [
        ("venue", "polymarket_us"),
        ("market_id", "OTHER-MARKET"),
        ("outcome", "no"),
        ("entry_price", "0.39"),
        ("exit_price", "0.56"),
        ("quantity", "3"),
        ("max_position", "21"),
        ("max_exposure_usd", "11"),
    ],
)
def test_any_rule_or_instrument_change_changes_identity(payload, field, value):
    original = StrategySpec.from_dict(payload)
    changed = StrategySpec.from_dict({**payload, field: value})
    assert original.version_id != changed.version_id


@pytest.mark.parametrize(
    "value",
    [
        0.4,
        1,
        True,
        None,
        "NaN",
        "Infinity",
        "-0",
        "-1",
        "1e-2",
        " 0.4",
        "0.4\n",
        "00.4",
        ".4",
        "0.1234567890123456789",
        "1000000000000000000",
    ],
)
def test_invalid_decimal_wire_values(payload, value):
    candidate = {**payload, "entry_price": value}
    with pytest.raises(ValueError):
        StrategySpec.from_dict(candidate)
    with pytest.raises(ValidationError):
        Draft202012Validator(strategy_schema()).validate(candidate)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "2.0.0"),
        ("kind", "python"),
        ("venue", "polymarket"),
        ("outcome", "long"),
        ("market_id", ""),
        ("market_id", "market\n"),
        ("market_id", "x" * 257),
        ("entry_price", "0"),
        ("exit_price", "1"),
        ("entry_price", "0.55"),
        ("entry_price", "0.6"),
        ("quantity", "0"),
        ("quantity", "21"),
        ("max_exposure_usd", "0"),
        ("max_exposure_usd", "0.99"),
    ],
)
def test_invalid_or_contradictory_rules(payload, field, value):
    with pytest.raises(ValueError):
        StrategySpec.from_dict({**payload, field: value})


def test_missing_extra_and_duplicate_fields(payload):
    with pytest.raises(ValueError):
        StrategySpec.from_dict({**payload, "live_enabled": True})
    with pytest.raises(ValueError):
        StrategySpec.from_dict({k: v for k, v in payload.items() if k != "schema_version"})
    with pytest.raises(ValueError):
        StrategySpec.from_json("[]")
    with pytest.raises(ValueError, match="duplicate"):
        StrategySpec.from_json(json.dumps(payload)[:-1] + ', "quantity": "3"}')


def test_direct_constructor_and_replace_validate(payload):
    spec = StrategySpec.from_dict(payload)
    assert replace(spec, quantity=Decimal("3")).quantity == Decimal("3")
    for value in [0.4, True, Decimal("NaN"), Decimal("-0"), Decimal("1e999999")]:
        with pytest.raises(ValueError):
            replace(spec, entry_price=value)
    with pytest.raises(ValueError):
        replace(spec, quantity=Decimal("100"))


def test_identity_and_limit_check_independent_of_decimal_context(payload):
    payload.update(
        entry_price="0.123456789012345678", quantity="3", max_exposure_usd="0.370370367037037033"
    )
    # Exact cost is 0.370370367037037034, not a rounded approximation.
    with localcontext() as context:
        context.prec = 2
        with pytest.raises(ValueError, match="cannot cover"):
            StrategySpec.from_dict(payload)
        payload["max_exposure_usd"] = "0.370370367037037034"
        low_precision = StrategySpec.from_dict(payload).version_id
    assert StrategySpec.from_dict(payload).version_id == low_precision


def test_schema_export_is_detached():
    first = strategy_schema()
    first["properties"].clear()
    assert "venue" in strategy_schema()["properties"]
