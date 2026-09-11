"""Synthetic cross-venue replay; no credentials, data capture or market calls."""

import builtins
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


def changed_recording(tmp_path, change, venue="polymarket-us"):
    spec, fixture = inputs(venue)
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
    assert poly["trace"][2]["source_at"] == "2026-09-10T18:00:02+00:00"
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


def test_no_strategy_cannot_reuse_yes_sports_proposition():
    spec, path = inputs()
    assert run(spec, path)["sports_market"]["outcome_team_id"] == "mlb:atl"
    with pytest.raises(ValueError, match="outcome"):
        run(replace(spec, outcome="no"), path)


def test_no_header_cannot_label_opposite_trade_as_yes_team(tmp_path):
    spec, path = changed_recording(tmp_path, lambda rows: rows[0].update(outcome="no"))
    with pytest.raises(ValueError, match="YES team-wins outcome only"):
        run(replace(spec, outcome="no"), path)
    with pytest.raises(ValueError, match="YES team-wins outcome only"):
        describe_recording(path)


@pytest.mark.parametrize("venue", ["kalshi", "polymarket-us"])
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
def test_quality_faults_fail_before_returning_a_report(tmp_path, change, reason, venue):
    spec, path = changed_recording(tmp_path, change, venue)
    with pytest.raises(ValueError, match=reason):
        run(spec, path)
    with pytest.raises(ValueError):
        describe_recording(path)


@pytest.mark.parametrize("venue", ["kalshi", "polymarket-us"])
@pytest.mark.parametrize("side", ["buy", "sell"])
@pytest.mark.parametrize("depth", ["1.99", "2.00"])
def test_fill_or_kill_at_exact_depth_boundary(tmp_path, venue, side, depth):
    def change(rows):
        index, ladder, price = (3, "asks", "0.42") if side == "buy" else (5, "bids", "0.7")
        rows[index][ladder] = [[price, depth]]

    spec, path = changed_recording(tmp_path, change, venue)
    result = run(spec, path)
    row = result["trace"][2 if side == "buy" else 4]
    if depth == "1.99":
        assert row["action"] == "cancel"
        assert row["reason"] == "insufficient_executable_depth"
        assert result["position"] == ("0" if side == "buy" else "2")
        assert result["cash"] == ("10" if side == "buy" else "9.155")
        assert result["realized_pnl"] == "0"
    else:
        assert row["action"] == "fill"
        assert row["levels"] == [{"price": "0.42" if side == "buy" else "0.7", "quantity": "2"}]
        assert row["fees"] == "0.02"


@pytest.mark.parametrize("venue", ["kalshi", "polymarket-us"])
@pytest.mark.parametrize("cash", ["0.919999999999999999", "0.92"])
def test_reserved_cash_boundary_includes_fees(venue, cash):
    spec, path = inputs(venue)
    result = simulate_recording(spec, path, initial_cash=cash, fee_per_contract="0.01")
    if cash == "0.92":
        assert result["trace"][1]["action"] == "intent"
        assert result["trace"][1]["reserved_cash"] == "0.92"
        assert result["realized_pnl"] == "0.52"
    else:
        assert result["trace"][1]["reason"] == "insufficient_cash"
        assert result["cash"] == cash
        assert result["position"] == "0"
        assert not any(row["action"] == "fill" for row in result["trace"])


@pytest.mark.parametrize("venue", ["kalshi", "polymarket-us"])
@pytest.mark.parametrize("source", ["2026-09-10T17:59:31Z", "2026-09-10T17:59:30.999999Z"])
def test_source_age_boundary_to_microsecond(tmp_path, venue, source):
    spec, path = changed_recording(tmp_path, lambda rows: rows[2].update(source_at=source), venue)
    if source.endswith("31Z"):
        assert run(spec, path)["cash"] == "10.52"
    else:
        with pytest.raises(ValueError, match="stale"):
            run(spec, path)


@pytest.mark.parametrize("venue", ["novig", "polymarket", "unknown"])
def test_unsupported_venue_cannot_create_strategy_or_recording(tmp_path, venue):
    spec, _ = inputs()
    with pytest.raises(ValueError, match="venue"):
        replace(spec, venue=venue)
    _, path = changed_recording(tmp_path, lambda rows: rows[0].update(venue=venue))
    with pytest.raises(ValueError, match="venue"):
        describe_recording(path)


@pytest.mark.parametrize("venue", ["kalshi", "polymarket-us"])
@pytest.mark.parametrize("cash", ["3.439999999999999999", "3.44"])
def test_exit_fees_never_make_cash_negative(venue, cash):
    spec, path = inputs(venue)
    spec = replace(spec, max_exposure_usd="3")
    result = simulate_recording(spec, path, initial_cash=cash, fee_per_contract="1")
    if cash == "3.44":
        assert result["trace"][4]["action"] == "fill"
        assert result["cash"] == result["position"] == "0"
        assert result["realized_pnl"] == "-3.44"
    else:
        assert result["trace"][4]["reason"] == "insufficient_cash_for_fees"
        assert result["cash"] == "0.614999999999999999"
        assert result["position"] == "2"
        assert result["realized_pnl"] == "0"


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


@pytest.mark.parametrize("describe", [False, True])
@pytest.mark.parametrize("replace_after_open", [1, 2])
def test_atomic_path_replace_cannot_mix_report_metadata_and_books(
    tmp_path, monkeypatch, describe, replace_after_open
):
    spec, fixture = inputs()
    path = tmp_path / "recording.jsonl"
    replacement = tmp_path / "replacement.jsonl"
    path.write_bytes(fixture.read_bytes())
    rows = [json.loads(line) for line in fixture.read_text().splitlines()]
    rows[0]["sports_market"]["outcome_team_id"] = "mlb:tb"
    rows[2]["asks"] = rows[3]["asks"] = [["0.44", "3"]]
    replacement.write_text("".join(json.dumps(row) + "\n" for row in rows))
    consume = describe_recording if describe else lambda path: run(spec, path)
    expected = consume(path)
    original_open = builtins.open
    opens = 0

    def replace_on_open(file, *args, **kwargs):
        nonlocal opens
        source = original_open(file, *args, **kwargs)
        if file == path:
            opens += 1
            if opens == replace_after_open:
                replacement.replace(path)
        return source

    monkeypatch.setattr(builtins, "open", replace_on_open)
    if replace_after_open == 1:
        with pytest.raises(ValueError, match="metadata changed"):
            consume(path)
    else:
        # The replay already opened A: its accepted metadata and every book
        # still belong to A, even though the path now resolves to B.
        assert consume(path) == expected


def test_replay_header_and_rows_share_one_open_file(tmp_path, monkeypatch):
    _, fixture = inputs()
    path = tmp_path / "recording.jsonl"
    replacement = tmp_path / "replacement.jsonl"
    path.write_bytes(fixture.read_bytes())
    rows = [json.loads(line) for line in fixture.read_text().splitlines()]
    rows[0]["sports_market"]["outcome_team_id"] = "mlb:tb"
    rows[2]["asks"] = [["0.44", "3"]]
    replacement.write_text("".join(json.dumps(row) + "\n" for row in rows))
    expected = list(replay_book_recording(path))
    original_open = builtins.open

    def replace_on_open(file, *args, **kwargs):
        source = original_open(file, *args, **kwargs)
        if file == path and replacement.exists():
            replacement.replace(path)
        return source

    monkeypatch.setattr(builtins, "open", replace_on_open)
    assert list(replay_book_recording(path)) == expected


@pytest.mark.parametrize("side", ["buy", "sell"])
@pytest.mark.parametrize("equal", [False, True])
def test_delayed_source_cannot_fill_before_or_at_intent_time(tmp_path, side, equal):
    def delay(rows):
        signal, fill, second = (2, 3, 0) if side == "buy" else (4, 5, 2)
        rows[signal]["source_at"] = f"2026-09-10T18:00:{second:02d}Z"
        timestamp = f"{second + 1:02d}" if equal else f"{second:02d}.500000"
        rows[fill]["source_at"] = f"2026-09-10T18:00:{timestamp}Z"

    spec, path = changed_recording(tmp_path, delay)
    result = run(spec, path)
    row = result["trace"][2 if side == "buy" else 4]
    assert row["action"] == "cancel"
    assert row["reason"] == "source_not_after_intent"
    assert result["position"] == ("0" if side == "buy" else "2")
    assert result["realized_pnl"] == "0"
