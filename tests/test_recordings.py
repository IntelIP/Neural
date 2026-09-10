"""Synthetic cross-venue replay; no credentials, data capture or market calls."""

import json
from dataclasses import replace
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from neural.paper import simulate_recording
from neural.paper_worker import PaperJobs
from neural.recordings import describe_recording, replay_book_recording
from neural.sports import SportsMarket, compare_sports_markets
from neural.strategy import StrategySpec

FIXTURES = Path(__file__).parents[1] / "examples" / "recordings"


def inputs(venue="polymarket-us"):
    stem = FIXTURES / f"synthetic-{venue}"
    spec = StrategySpec.from_json(Path(str(stem) + "-strategy.json").read_text())
    return spec, Path(str(stem) + ".jsonl")


def run(spec, path):
    return simulate_recording(spec, path, initial_cash="10", fee_per_contract="0.01")


def changed_recording(tmp_path, change):
    spec, fixture = inputs()
    rows = [json.loads(line) for line in fixture.read_text().splitlines()]
    change(rows)
    path = tmp_path / "modified.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return spec, path


def test_same_strategy_logic_uses_full_decimal_depth_across_venues():
    kalshi_spec, kalshi_path = inputs("kalshi")
    poly_spec, poly_path = inputs()
    assert replace(kalshi_spec, venue=poly_spec.venue, market_id=poly_spec.market_id) == poly_spec
    kalshi, poly = run(kalshi_spec, kalshi_path), run(poly_spec, poly_path)
    with localcontext() as ctx:
        ctx.prec = 2
        assert run(poly_spec, poly_path) == poly
    assert kalshi["trace"] == poly["trace"]
    assert kalshi["cash"] == poly["cash"] == "10.52"
    assert kalshi["realized_pnl"] == poly["realized_pnl"] == "0.52"
    assert poly["trace"][2]["levels"] == [
        {"price": "0.4", "quantity": "0.75"},
        {"price": "0.42", "quantity": "1.25"},
    ]
    assert poly["trace"][2]["source_at"] == "2026-09-10T18:00:01+00:00"
    assert poly["market_compatibility"]["status"] == "unknown"
    comparison = compare_sports_markets(
        SportsMarket.from_dict(kalshi["sports_market"]),
        SportsMarket.from_dict(poly["sports_market"]),
    )
    assert comparison.status == "compatible"  # Synthetic rules only.
    assert kalshi["recording_digest"] != poly["recording_digest"]
    assert kalshi["result_id"] != poly["result_id"]


def test_summary_preserves_native_identity_and_counts():
    spec, path = inputs()
    summary = describe_recording(path)
    assert summary["venue"] == spec.venue
    assert summary["market_id"] == spec.market_id
    assert summary["sports_market"]["event_id"] == "fixture:polymarket_us:event-1"
    assert summary["provenance"] == "synthetic"
    assert summary["event_count"] == 6
    assert summary["book_count"] == 4
    assert summary["reset_count"] == 2
    assert summary["disconnects"] == 1
    assert summary["reconnects"] == 0
    assert summary["start_at"] == "2026-09-10T18:00:00+00:00"
    assert summary["end_at"] == "2026-09-10T18:00:05+00:00"
    with pytest.raises(ValueError, match="max_events"):
        describe_recording(path, max_events=5)


@pytest.mark.parametrize("outcome", ["yes", "no"])
def test_selected_outcome_book_is_explicit(tmp_path, outcome):
    spec, path = changed_recording(tmp_path, lambda rows: rows[0].update(outcome=outcome))
    assert run(replace(spec, outcome=outcome), path)["cash"] == "10.52"
    with pytest.raises(ValueError, match="outcome"):
        run(replace(spec, outcome="no" if outcome == "yes" else "yes"), path)


@pytest.mark.parametrize(
    "change,reason",
    [
        (lambda rows: rows[2].update(asks=[]), "both nonempty"),
        (lambda rows: rows[2].update(bids=[]), "both nonempty"),
        (lambda rows: rows[2].pop("asks"), "shape"),
        (lambda rows: rows[2].update(quality="bbo_only"), "full_depth"),
        (lambda rows: rows[2].update(bids=[["0.6", "2"]]), "crossed"),
        (lambda rows: rows[2].update(source_at="2026-09-10T17:59:00Z"), "stale"),
        (lambda rows: rows[2].update(source_at="2026-09-10T18:01:00Z"), "future"),
        (lambda rows: rows[2].update(source_at=None), "required"),
        (lambda rows: rows[2].update(source_at="2026-09-10T18:00:00"), "timezone"),
        (lambda rows: rows[3].update(source_at="2026-09-10T17:59:59Z"), "regress"),
        (lambda rows: rows[3].update(received_at="2026-09-10T17:59:59Z"), "regress"),
        (lambda rows: rows[3].update(sequence=3), "sequence"),
        (lambda rows: rows[2].update(sequence=True), "sequence"),
        (lambda rows: rows[2].update(asks=[[0.4, "3"]]), "decimal"),
        (lambda rows: rows[2].update(asks=[["0.4", "0"]]), "quantity"),
        (lambda rows: rows[2].update(asks=[["1.1", "3"]]), "price"),
        (lambda rows: rows[2].update(asks=[["0.4", "1"], ["0.4", "2"]]), "unique"),
        (lambda rows: rows[2].update(asks=[["0.5", "1"], ["0.4", "2"]]), "order"),
        (lambda rows: rows[0].update(venue="polymarket"), "venue"),
        (lambda rows: rows[0].update(market_id="other"), "sports market"),
        (lambda rows: rows[0].update(provenance="live"), "synthetic"),
        (lambda rows: rows.pop(), "terminal reset"),
        (lambda rows: rows.pop(1), "outside recording session"),
    ],
)
def test_quality_faults_fail_before_returning_a_report(tmp_path, change, reason):
    spec, path = changed_recording(tmp_path, change)
    with pytest.raises(ValueError, match=reason):
        run(spec, path)
    with pytest.raises(ValueError):
        describe_recording(path)


def test_wrong_venue_rejected_even_when_market_ids_match():
    spec, path = inputs()
    with pytest.raises(ValueError, match="venue/market"):
        run(replace(spec, venue="kalshi"), path)


def test_truncated_or_duplicate_tail_fails_closed(tmp_path):
    spec, fixture = inputs()
    path = tmp_path / "tail.jsonl"
    for tail in ('{"kind":"reset","kind":"book"}\n', "bad\n"):
        path.write_text(fixture.read_text() + tail)
        with pytest.raises(ValueError):
            run(spec, path)
    path.write_text(fixture.read_text().rstrip("\n"))
    with pytest.raises(ValueError, match="truncated"):
        run(spec, path)


def test_depth_does_not_round_trip_through_float(tmp_path):
    spec, path = changed_recording(
        tmp_path,
        lambda rows: rows[2].update(asks=[["0.400000000000000001", "3.000000000000000001"]]),
    )
    event = list(replay_book_recording(path))[1]
    assert event.update.book.asks(spec.outcome)[0].price == Decimal("0.400000000000000001")
    assert event.update.book.asks(spec.outcome)[0].quantity == Decimal("3.000000000000000001")


def test_polymarket_jobs_snapshot_and_recover_without_venue_specific_worker(tmp_path):
    spec, fixture = inputs()
    path = tmp_path / "recording.jsonl"
    path.write_bytes(fixture.read_bytes())
    database = tmp_path / "jobs.sqlite3"
    jobs = PaperJobs(database)
    identity = jobs.submit(spec, path, initial_cash="10", fee_per_contract="0.01")
    assert jobs.submit(spec, path, initial_cash="10.0", fee_per_contract="0.010") == identity
    path.write_text("malformed replacement\n")
    completed = PaperJobs(database).run_next()
    assert completed["status"] == "completed"
    assert completed["result"] == run(spec, fixture)
    assert PaperJobs(database).inspect(identity) == completed
    bad_identity = jobs.submit(spec, path, initial_cash="10", fee_per_contract="0.01")
    assert jobs.run_next()["status"] == "failed"
    assert PaperJobs(database).inspect(bad_identity)["result"] is None
    assert PaperJobs(database).run_next() is None
