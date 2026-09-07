"""Synthetic offline fixtures; never connect to a venue."""

import json
import subprocess
import sys
from dataclasses import replace
from decimal import localcontext

import pytest

from neural.paper import simulate_recording
from neural.strategy import StrategySpec


@pytest.fixture
def spec():
    return StrategySpec("kalshi", "KX-EXAMPLE", "yes", "0.45", "0.65", "2", "2", "1")


def recording(tmp_path, books=None, *, times=None, outcome="yes"):
    books = books or [
        ("0.30", "0.60", "3"),
        ("0.30", "0.60", "3"),
        ("0.70", "0.20", "3"),
        ("0.70", "0.20", "3"),
    ]
    records = []

    def add(kind, second, **extra):
        records.append(
            dict(
                version="kalshi-book/1",
                ticker="KX-EXAMPLE",
                kind=kind,
                received_at=f"2026-09-07T00:00:{second:02d}+00:00",
                **extra,
            )
        )

    add("reset", 0, reason="connecting")
    for index, (bid, opposite, quantity) in enumerate(books, 1):
        yes, no = (bid, opposite) if outcome == "yes" else (opposite, bid)
        add(
            "frame",
            times[index - 1] if times else index,
            frame={
                "type": "orderbook_snapshot",
                "sid": 7,
                "seq": index,
                "msg": {
                    "market_ticker": "KX-EXAMPLE",
                    "yes_dollars_fp": [[yes, quantity]],
                    "no_dollars_fp": [[no, quantity]],
                },
            },
        )
    add("reset", 59, reason="disconnected")
    path = tmp_path / "books.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    return path


def run(spec, path, **kwargs):
    return simulate_recording(
        spec,
        path,
        initial_cash=kwargs.pop("initial_cash", "10"),
        fee_per_contract=kwargs.pop("fee_per_contract", "0.01"),
        **kwargs,
    )


@pytest.mark.parametrize("outcome", ["yes", "no"])
def test_round_trip_exact_and_deterministic(tmp_path, spec, outcome):
    path = recording(tmp_path, outcome=outcome)
    spec = replace(spec, outcome=outcome)
    result = run(spec, path)
    with localcontext() as ctx:
        ctx.prec = 2
        assert run(spec, path) == result
    assert result["cash"] == "10.56"
    assert result["realized_pnl"] == "0.56"
    assert result["position"] == result["acquisition_cost"] == "0"
    assert [row["action"] for row in result["trace"]] == [
        "reset",
        "intent",
        "fill",
        "intent",
        "fill",
        "reset",
    ]
    assert result["trace"][2]["fees"] == "0.02"


@pytest.mark.parametrize(
    "cash,fee,reason", [("0.91", "0.01", "insufficient_cash"), ("10", "0.1", "max_exposure_usd")]
)
def test_entry_budget_includes_reserved_fees(tmp_path, spec, cash, fee, reason):
    result = run(spec, recording(tmp_path), initial_cash=cash, fee_per_contract=fee)
    assert result["trace"][1]["reason"] == reason
    assert result["position"] == "0"
    assert not any(row["action"] == "fill" for row in result["trace"])


def test_insufficient_depth_is_fok(tmp_path, spec):
    path = recording(tmp_path, [("0.3", "0.6", "3"), ("0.3", "0.6", "1")])
    result = run(spec, path)
    assert result["trace"][2]["reason"] == "insufficient_executable_depth"
    assert result["cash"] == "10"


def test_expiry_and_terminal_cancel(tmp_path, spec):
    result = run(spec, recording(tmp_path, times=[1, 40, 41, 42]))
    assert result["trace"][2]["reason"] == "order_expired"
    result = run(spec, recording(tmp_path, [("0.3", "0.6", "3")]))
    assert result["trace"][-1]["action"] == "cancel"
    assert result["cash"] == "10"


def test_open_position_remains_unrealized(tmp_path, spec):
    result = run(spec, recording(tmp_path, [("0.3", "0.6", "3")] * 2))
    assert result["position"] == "2"
    assert result["acquisition_cost"] == "0.82"
    assert result["realized_pnl"] == "0"


@pytest.mark.parametrize("fault", ["tail", "ticker", "clock", "crossed", "bound"])
def test_bad_input_never_returns_success(tmp_path, spec, fault):
    path = recording(tmp_path)
    kwargs = {}
    if fault == "tail":
        path.write_text(path.read_text() + "bad\n")
    elif fault == "ticker":
        spec = replace(spec, market_id="OTHER")
    elif fault == "clock":
        path = recording(tmp_path, times=[2, 1, 3, 4])
    elif fault == "crossed":
        path = recording(tmp_path, [("0.7", "0.6", "3")])
    else:
        kwargs["max_events"] = 2
    with pytest.raises(ValueError):
        run(spec, path, **kwargs)


@pytest.mark.parametrize(
    "changes,kwargs",
    [
        ({"venue": "polymarket_us"}, {}),
        ({"quantity": "0.001"}, {}),
        ({}, {"fee_per_contract": "NaN"}),
        ({}, {"initial_cash": 10.0}),
        ({}, {"max_events": True}),
        ({}, {"max_order_age_seconds": 0}),
    ],
)
def test_reject_invalid_model_inputs(tmp_path, spec, changes, kwargs):
    with pytest.raises(ValueError):
        run(replace(spec, **changes), recording(tmp_path), **kwargs)


def test_cli_success_and_bad_tail_no_stdout(tmp_path, spec):
    strategy = tmp_path / "strategy.json"
    strategy.write_text(spec.to_json())
    path = recording(tmp_path)
    command = [
        sys.executable,
        "-m",
        "neural.paper",
        "--strategy",
        str(strategy),
        "--recording",
        str(path),
        "--cash",
        "10",
        "--fee-per-contract",
        "0.01",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["cash"] == "10.56"
    path.write_text(path.read_text() + "bad\n")
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert result.stdout == ""


def test_reconnect_cancels_intent_and_preserves_holdings(tmp_path, spec):
    path = recording(tmp_path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]

    def reset(reason, at):
        return dict(rows[0], reason=reason, received_at=at)

    # Entry fills, exit intent is invalidated, reconnect then needs a fresh intent.
    rows[4:4] = [
        reset("disconnected", rows[3]["received_at"]),
        reset("connecting", rows[3]["received_at"]),
    ]
    extra = json.loads(json.dumps(rows[-2]))
    extra["frame"]["seq"] += 1
    extra["received_at"] = "2026-09-07T00:00:05+00:00"
    rows.insert(-1, extra)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = run(spec, path)
    assert [r["action"] for r in result["trace"]] == [
        "reset",
        "intent",
        "fill",
        "intent",
        "cancel",
        "reset",
        "intent",
        "fill",
        "reset",
    ]
    assert result["cash"] == "10.56"


def test_depth_walk_and_no_second_cycle(tmp_path, spec):
    path = recording(tmp_path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[2]["frame"]["msg"]["no_dollars_fp"] = [["0.6", "1"], ["0.56", "2"]]
    extra = json.loads(json.dumps(rows[1]))
    extra["frame"]["seq"] = 5
    extra["received_at"] = "2026-09-07T00:00:05+00:00"
    rows.insert(-1, extra)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = run(spec, path)
    assert result["trace"][2]["notional"] == "0.84"
    assert len(result["trace"][2]["levels"]) == 2
    assert result["cash"] == "10.52"
    assert result["trace"][-2]["action"] == "hold"
