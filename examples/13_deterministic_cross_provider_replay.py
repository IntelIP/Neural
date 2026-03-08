"""
Deterministic cross-provider normalization and analysis replay.

This example is the complement to the live public bridge example. Instead of
depending on live market movement, it replays fixed provider-shaped records so
the output is reproducible and the analysis stack produces non-trivial signals.

It demonstrates:
1. Provider-specific raw pulls with different field names.
2. One shared normalization schema across Kalshi and Polymarket US.
3. One provider-agnostic event stream built from repeated polling rounds.
4. One analysis pass using the same strategy and risk sizing methods.

Examples:
    uv run python examples/13_deterministic_cross_provider_replay.py
    uv run python examples/13_deterministic_cross_provider_replay.py --output-dir data/deterministic_cross_provider
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

# Allow running the example directly from the repository root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neural.analysis import edge_proportional, fixed_percentage, kelly_criterion
from neural.analysis.strategies import MeanReversionStrategy

ANALYSIS_CAPITAL = 1000.0


RAW_KALSHI_ROUNDS: list[list[dict[str, Any]]] = [
    [
        {
            "ticker": "KXNBAGAME-26MAR10CHINYK-NYK",
            "title": "Chicago at New York Winner?",
            "status": "open",
            "home_team": "New York Knicks",
            "away_team": "Chicago Bulls",
            "game_date": "2026-03-10T23:00:00Z",
            "yes_ask": 55,
            "no_ask": 45,
            "volume": 100,
        }
    ],
    [
        {
            "ticker": "KXNBAGAME-26MAR10CHINYK-NYK",
            "title": "Chicago at New York Winner?",
            "status": "open",
            "home_team": "New York Knicks",
            "away_team": "Chicago Bulls",
            "game_date": "2026-03-10T23:00:00Z",
            "yes_ask": 54,
            "no_ask": 46,
            "volume": 110,
        }
    ],
    [
        {
            "ticker": "KXNBAGAME-26MAR10CHINYK-NYK",
            "title": "Chicago at New York Winner?",
            "status": "open",
            "home_team": "New York Knicks",
            "away_team": "Chicago Bulls",
            "game_date": "2026-03-10T23:00:00Z",
            "yes_ask": 40,
            "no_ask": 60,
            "volume": 180,
        }
    ],
]


RAW_POLYMARKET_ROUNDS: list[list[dict[str, Any]]] = [
    [
        {
            "id": "pm-bos-lal-moneyline",
            "slug": "lakers-celtics-moneyline",
            "question": "Will the Boston Celtics beat the Los Angeles Lakers?",
            "status": "open",
            "homeTeam": "Boston Celtics",
            "awayTeam": "Los Angeles Lakers",
            "gameStartTime": "2026-03-11T00:30:00Z",
            "marketType": "moneyline",
            "yes_price": 0.42,
            "no_price": 0.58,
            "last_price": 0.42,
            "volume": 90,
        },
        {
            "id": "pm-den-dal-moneyline",
            "slug": "mavericks-nuggets-moneyline",
            "question": "Will the Denver Nuggets beat the Dallas Mavericks?",
            "status": "open",
            "homeTeam": "Denver Nuggets",
            "awayTeam": "Dallas Mavericks",
            "gameStartTime": "2026-03-11T02:00:00Z",
            "marketType": "moneyline",
            "yes_price": 0.50,
            "no_price": 0.50,
            "last_price": 0.50,
            "volume": 70,
        },
    ],
    [
        {
            "id": "pm-bos-lal-moneyline",
            "slug": "lakers-celtics-moneyline",
            "question": "Will the Boston Celtics beat the Los Angeles Lakers?",
            "status": "open",
            "homeTeam": "Boston Celtics",
            "awayTeam": "Los Angeles Lakers",
            "gameStartTime": "2026-03-11T00:30:00Z",
            "marketType": "moneyline",
            "yes_price": 0.43,
            "no_price": 0.57,
            "last_price": 0.43,
            "volume": 95,
        },
        {
            "id": "pm-den-dal-moneyline",
            "slug": "mavericks-nuggets-moneyline",
            "question": "Will the Denver Nuggets beat the Dallas Mavericks?",
            "status": "open",
            "homeTeam": "Denver Nuggets",
            "awayTeam": "Dallas Mavericks",
            "gameStartTime": "2026-03-11T02:00:00Z",
            "marketType": "moneyline",
            "yes_price": 0.51,
            "no_price": 0.49,
            "last_price": 0.51,
            "volume": 72,
        },
    ],
    [
        {
            "id": "pm-bos-lal-moneyline",
            "slug": "lakers-celtics-moneyline",
            "question": "Will the Boston Celtics beat the Los Angeles Lakers?",
            "status": "open",
            "homeTeam": "Boston Celtics",
            "awayTeam": "Los Angeles Lakers",
            "gameStartTime": "2026-03-11T00:30:00Z",
            "marketType": "moneyline",
            "yes_price": 0.64,
            "no_price": 0.36,
            "last_price": 0.64,
            "volume": 200,
        },
        {
            "id": "pm-den-dal-moneyline",
            "slug": "mavericks-nuggets-moneyline",
            "question": "Will the Denver Nuggets beat the Dallas Mavericks?",
            "status": "open",
            "homeTeam": "Denver Nuggets",
            "awayTeam": "Dallas Mavericks",
            "gameStartTime": "2026-03-11T02:00:00Z",
            "marketType": "moneyline",
            "yes_price": 0.505,
            "no_price": 0.495,
            "last_price": 0.505,
            "volume": 74,
        },
    ],
]


ROUND_TIMESTAMPS = [
    pd.Timestamp("2026-03-08T12:00:00Z"),
    pd.Timestamp("2026-03-08T12:01:00Z"),
    pd.Timestamp("2026-03-08T12:02:00Z"),
]


def _normalize_kalshi_round(records: list[dict[str, Any]], stream_timestamp: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for record in records:
        yes_price = float(record["yes_ask"]) / 100.0
        no_price = float(record["no_ask"]) / 100.0
        rows.append(
            {
                "exchange": "kalshi",
                "market_id": record["ticker"],
                "ticker": record["ticker"],
                "title": record["title"],
                "status": record["status"],
                "home_team": record["home_team"],
                "away_team": record["away_team"],
                "game_date": pd.Timestamp(record["game_date"]),
                "market_type": "moneyline",
                "yes_price": yes_price,
                "no_price": no_price,
                "last_price": yes_price,
                "volume": float(record["volume"]),
                "stream_timestamp": stream_timestamp,
            }
        )
    return pd.DataFrame.from_records(rows)


def _normalize_polymarket_round(
    records: list[dict[str, Any]], stream_timestamp: pd.Timestamp
) -> pd.DataFrame:
    rows = []
    for record in records:
        rows.append(
            {
                "exchange": "polymarket_us",
                "market_id": record["id"],
                "ticker": record["slug"],
                "title": record["question"],
                "status": record["status"],
                "home_team": record["homeTeam"],
                "away_team": record["awayTeam"],
                "game_date": pd.Timestamp(record["gameStartTime"]),
                "market_type": record["marketType"],
                "yes_price": float(record["yes_price"]),
                "no_price": float(record["no_price"]),
                "last_price": float(record["last_price"]),
                "volume": float(record["volume"]),
                "stream_timestamp": stream_timestamp,
            }
        )
    return pd.DataFrame.from_records(rows)


def build_replay_stream() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for round_index, stream_timestamp in enumerate(ROUND_TIMESTAMPS):
        kalshi_frame = _normalize_kalshi_round(RAW_KALSHI_ROUNDS[round_index], stream_timestamp)
        polymarket_frame = _normalize_polymarket_round(
            RAW_POLYMARKET_ROUNDS[round_index], stream_timestamp
        )
        round_frame = pd.concat([kalshi_frame, polymarket_frame], ignore_index=True)
        round_frame["poll_round"] = round_index + 1
        frames.append(round_frame)

    stream = pd.concat(frames, ignore_index=True)
    stream["stream_timestamp"] = pd.to_datetime(stream["stream_timestamp"], utc=True)
    stream["game_date"] = pd.to_datetime(stream["game_date"], utc=True)
    return stream


def latest_snapshot(stream: pd.DataFrame) -> pd.DataFrame:
    if stream.empty:
        return stream
    latest_round = int(stream["poll_round"].max())
    snapshot = stream[stream["poll_round"] == latest_round].copy()
    return snapshot.reset_index(drop=True)


def analyze_standardized_stream(stream: pd.DataFrame) -> pd.DataFrame:
    if stream.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for (exchange, market_id), market_events in stream.groupby(["exchange", "market_id"], sort=False):
        market_events = market_events.sort_values("stream_timestamp").reset_index(drop=True)
        market_data = market_events[["ticker", "yes_price", "no_price", "volume"]].rename(
            columns={"yes_price": "yes_ask", "no_price": "no_ask"}
        )

        strategy = MeanReversionStrategy(
            name=f"{exchange}:{market_id}",
            divergence_threshold=0.03,
            lookback_periods=2,
            use_sportsbook=False,
            initial_capital=ANALYSIS_CAPITAL,
            max_position_size=0.05,
        )

        signal = strategy.hold(str(market_id))
        for end_idx in range(1, len(market_data) + 1):
            signal = strategy.analyze(market_data.iloc[:end_idx])

        latest_yes = float(market_data.iloc[-1]["yes_ask"])
        rolling_mean = float(market_data["yes_ask"].mean())
        edge = abs(latest_yes - rolling_mean)

        rows.append(
            {
                "exchange": exchange,
                "market_id": market_id,
                "ticker": market_events.iloc[-1]["ticker"],
                "title": market_events.iloc[-1]["title"],
                "signal": signal.signal_type.value,
                "signal_confidence": signal.confidence,
                "recommended_fraction": signal.recommended_size,
                "latest_yes_price": latest_yes,
                "rolling_mean_yes_price": rolling_mean,
                "edge_vs_mean": edge,
                "kelly_fraction": kelly_criterion(edge=edge, odds=1.0),
                "edge_contracts": edge_proportional(edge=edge, capital=ANALYSIS_CAPITAL),
                "fixed_contracts": fixed_percentage(capital=ANALYSIS_CAPITAL, percentage=0.02),
                "entry_price": (signal.metadata or {}).get("entry_price"),
                "target_price": (signal.metadata or {}).get("target_price"),
                "fair_value": (signal.metadata or {}).get("fair_value"),
            }
        )

    return pd.DataFrame.from_records(rows).sort_values(
        ["exchange", "edge_vs_mean"], ascending=[True, False]
    ).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay deterministic cross-provider market pulls through one normalization and analysis layer."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "deterministic_cross_provider",
        help="Directory for replay snapshot, stream, and analysis CSVs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    stream = build_replay_stream()
    snapshot = latest_snapshot(stream)
    analysis = analyze_standardized_stream(stream)

    date_stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    snapshot_path = output_dir / f"deterministic_snapshot_{date_stamp}.csv"
    stream_path = output_dir / f"deterministic_stream_{date_stamp}.csv"
    analysis_path = output_dir / f"deterministic_analysis_{date_stamp}.csv"

    snapshot.to_csv(snapshot_path, index=False)
    stream.to_csv(stream_path, index=False)
    analysis.to_csv(analysis_path, index=False)

    print(f"Wrote deterministic snapshot to {snapshot_path.as_posix()}")
    print(f"Wrote deterministic stream to {stream_path.as_posix()}")
    print(f"Wrote deterministic analysis to {analysis_path.as_posix()}")

    print("\nLatest Standardized Snapshot:\n")
    print(
        snapshot[
            [
                "exchange",
                "ticker",
                "title",
                "home_team",
                "away_team",
                "game_date",
                "yes_price",
                "no_price",
                "last_price",
                "volume",
            ]
        ].to_string(index=False)
    )

    print("\nDeterministic Replay Stream:\n")
    print(
        stream[
            [
                "poll_round",
                "exchange",
                "ticker",
                "stream_timestamp",
                "yes_price",
                "no_price",
                "volume",
            ]
        ].to_string(index=False)
    )

    print("\nAnalysis Output:\n")
    print(
        analysis[
            [
                "exchange",
                "ticker",
                "signal",
                "signal_confidence",
                "recommended_fraction",
                "latest_yes_price",
                "rolling_mean_yes_price",
                "edge_vs_mean",
                "kelly_fraction",
                "edge_contracts",
                "fixed_contracts",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
