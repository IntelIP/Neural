from __future__ import annotations

from decimal import Decimal

import pytest

from neural.kernel import (
    DEMO_REPLAY_DIGEST,
    ReplayEvent,
    demo_events,
    replay,
    run_demo_replay,
)


def test_demo_replay_is_deterministic_across_input_order() -> None:
    expected = run_demo_replay()
    reversed_result = replay(reversed(demo_events()))

    assert expected.digest == reversed_result.digest
    assert expected.digest == DEMO_REPLAY_DIGEST
    assert expected.as_dict() == reversed_result.as_dict()
    assert len(expected.digest) == 64
    assert expected.as_dict()["event_count"] == 4
    assert expected.as_dict()["market_count"] == 2


def test_replay_snapshot_keeps_latest_market_observation() -> None:
    result = run_demo_replay()
    snapshot = {event.market_id: event for event in result.snapshot}

    assert snapshot["KX-DEMO-YES"].yes_price == Decimal("0.52")
    assert snapshot["KX-DEMO-NO"].no_price == Decimal("0.42")


def test_replay_normalizes_timestamps_to_utc() -> None:
    event = ReplayEvent.from_mapping(
        {
            "observed_at": "2026-07-29T13:00:00+01:00",
            "market_id": "KX-DEMO",
            "yes_price": "0.5",
            "no_price": "0.5",
        }
    )

    assert event.observed_at == "2026-07-29T12:00:00Z"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("yes_price", "1.01", "between 0 and 1"),
        ("no_price", "-0.01", "between 0 and 1"),
        ("volume", "-1", "non-negative"),
        ("observed_at", "2026-07-29T12:00:00", "include a timezone"),
    ],
)
def test_replay_event_rejects_invalid_input(field: str, value: str, message: str) -> None:
    payload = {
        "observed_at": "2026-07-29T12:00:00Z",
        "market_id": "KX-DEMO",
        "yes_price": "0.5",
        "no_price": "0.5",
        "volume": "1",
    }
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        ReplayEvent.from_mapping(payload)


def test_empty_replay_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one event"):
        replay([])
