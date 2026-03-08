"""
Cross-provider public market normalization and analysis demo.

This example shows how the SDK can:
1. Pull public NBA markets from Kalshi and Polymarket US.
2. Standardize them into one shared tabular schema.
3. Poll both providers to create a simple provider-agnostic event stream.
4. Run the same analysis stack over both feeds.

The goal is not to prove alpha. It demonstrates that external provider data can
flow through one normalization layer and then into one analysis layer.

Examples:
    uv run python examples/12_cross_provider_analysis_bridge.py
    uv run python examples/12_cross_provider_analysis_bridge.py --limit 3 --rounds 3 --interval 1.0
    uv run python examples/12_cross_provider_analysis_bridge.py --output-dir data/cross_provider_analysis
"""

from __future__ import annotations

import argparse
import asyncio
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
from neural.data_collection import PolymarketUSMarketsSource, get_nba_games

ANALYSIS_CAPITAL = 1000.0


def _coerce_probability(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric > 1:
        numeric /= 100.0
    if 0 <= numeric <= 1:
        return numeric
    return None


def _snapshot_now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(UTC))


async def fetch_kalshi_nba(limit: int) -> pd.DataFrame:
    df = await get_nba_games(status="open", limit=max(limit, 1), use_authenticated=False)
    if df.empty:
        return pd.DataFrame()

    df = df.copy().head(limit)
    df["exchange"] = "kalshi"
    df["market_id"] = df["ticker"].astype(str)
    df["yes_price"] = df["yes_ask"].apply(_coerce_probability)
    df["no_price"] = df["no_ask"].apply(_coerce_probability)
    df["last_price"] = df["yes_price"]
    df["market_type"] = "moneyline"
    df["stream_timestamp"] = _snapshot_now()
    return df[
        [
            "exchange",
            "market_id",
            "ticker",
            "title",
            "status",
            "home_team",
            "away_team",
            "game_date",
            "market_type",
            "yes_price",
            "no_price",
            "last_price",
            "volume",
            "stream_timestamp",
        ]
    ].reset_index(drop=True)


def fetch_polymarket_nba(limit: int) -> pd.DataFrame:
    source = PolymarketUSMarketsSource(config={"sport": "nba", "limit": limit, "sports_only": True})
    try:
        df = source.get_markets_df()
        if df.empty:
            return df

        df = df.copy().head(limit)
        yes_prices = []
        no_prices = []
        last_prices = []
        volumes = []
        for market_id in df["market_id"]:
            quote = source.adapter.get_quote(str(market_id))
            yes_prices.append(quote.yes_ask if quote.yes_ask is not None else quote.yes_bid)
            no_prices.append(quote.no_ask if quote.no_ask is not None else quote.no_bid)
            last_prices.append(quote.last_price)
            volumes.append(quote.volume)

        df["exchange"] = "polymarket_us"
        df["yes_price"] = yes_prices
        df["no_price"] = no_prices
        df["last_price"] = last_prices
        df["volume"] = volumes
        df["stream_timestamp"] = _snapshot_now()
        return df[
            [
                "exchange",
                "market_id",
                "ticker",
                "title",
                "status",
                "home_team",
                "away_team",
                "game_date",
                "market_type",
                "yes_price",
                "no_price",
                "last_price",
                "volume",
                "stream_timestamp",
            ]
        ].reset_index(drop=True)
    finally:
        source.adapter.close()


async def fetch_standardized_snapshots(limit: int) -> pd.DataFrame:
    kalshi_df, polymarket_df = await asyncio.gather(fetch_kalshi_nba(limit), asyncio.to_thread(fetch_polymarket_nba, limit))
    frames = [df for df in (kalshi_df, polymarket_df) if not df.empty]
    if not frames:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for frame in frames:
        records.extend(frame.to_dict("records"))

    combined = pd.DataFrame.from_records(records)
    combined["game_date"] = pd.to_datetime(combined["game_date"], errors="coerce", utc=True)
    for column in ("yes_price", "no_price", "last_price", "volume"):
        combined[column] = pd.to_numeric(combined[column], errors="coerce")
    return combined


def _event_frame_from_snapshot(snapshot: pd.DataFrame) -> pd.DataFrame:
    if snapshot.empty:
        return snapshot

    events = snapshot.copy()
    events["yes_ask"] = events["yes_price"].fillna(events["last_price"])
    events["no_ask"] = events["no_price"]
    missing_no = events["no_ask"].isna() & events["yes_ask"].notna()
    events.loc[missing_no, "no_ask"] = 1.0 - events.loc[missing_no, "yes_ask"]
    events = events.dropna(subset=["yes_ask", "no_ask"])
    return events[
        [
            "exchange",
            "market_id",
            "ticker",
            "title",
            "stream_timestamp",
            "yes_ask",
            "no_ask",
            "last_price",
            "volume",
        ]
    ].reset_index(drop=True)


async def build_polling_stream(limit: int, rounds: int, interval: float) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for round_index in range(rounds):
        snapshot = await fetch_standardized_snapshots(limit)
        if not snapshot.empty:
            snapshot = snapshot.copy()
            snapshot["poll_round"] = round_index + 1
            frames.append(_event_frame_from_snapshot(snapshot))
        if round_index < rounds - 1 and interval > 0:
            await asyncio.sleep(interval)

    if not frames:
        return pd.DataFrame()
    stream = pd.concat(frames, ignore_index=True)
    stream["stream_timestamp"] = pd.to_datetime(stream["stream_timestamp"], errors="coerce")
    return stream


def analyze_standardized_stream(stream: pd.DataFrame) -> pd.DataFrame:
    if stream.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for (exchange, market_id), market_events in stream.groupby(["exchange", "market_id"], sort=False):
        market_events = market_events.sort_values("stream_timestamp").reset_index(drop=True)
        market_data = market_events[["ticker", "yes_ask", "no_ask", "volume"]].copy()

        strategy = MeanReversionStrategy(
            name=f"{exchange}:{market_id}",
            divergence_threshold=0.001,
            lookback_periods=1,
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
                "latest_yes_price": latest_yes,
                "rolling_mean_yes_price": rolling_mean,
                "edge_vs_mean": edge,
                "signal": signal.signal_type.value,
                "signal_confidence": signal.confidence,
                "recommended_size": signal.recommended_size,
                "kelly_fraction": kelly_criterion(edge=edge, odds=1.0),
                "edge_contracts": edge_proportional(edge=edge, capital=ANALYSIS_CAPITAL),
                "fixed_contracts": fixed_percentage(capital=ANALYSIS_CAPITAL, percentage=0.02),
            }
        )

    result = pd.DataFrame(rows)
    return result.sort_values(["exchange", "edge_vs_mean"], ascending=[True, False]).reset_index(drop=True)


async def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot = await fetch_standardized_snapshots(args.limit)
    stream = await build_polling_stream(args.limit, args.rounds, args.interval)
    analysis = analyze_standardized_stream(stream)

    date_stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    snapshot_path = output_dir / f"cross_provider_snapshot_{date_stamp}.csv"
    stream_path = output_dir / f"cross_provider_stream_{date_stamp}.csv"
    analysis_path = output_dir / f"cross_provider_analysis_{date_stamp}.csv"

    snapshot.to_csv(snapshot_path, index=False)
    stream.to_csv(stream_path, index=False)
    analysis.to_csv(analysis_path, index=False)

    print(f"Wrote standardized snapshot to {snapshot_path.as_posix()}")
    print(f"Wrote polling stream to {stream_path.as_posix()}")
    print(f"Wrote analysis output to {analysis_path.as_posix()}")

    if not snapshot.empty:
        print("\nStandardized Snapshot Preview:\n")
        print(
            snapshot[
                [
                    column
                    for column in (
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
                    )
                    if column in snapshot.columns
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

    if not stream.empty:
        print("\nStandardized Stream Preview:\n")
        print(stream.head(10).to_string(index=False))

    if not analysis.empty:
        print("\nAnalysis Preview:\n")
        print(analysis.head(10).to_string(index=False))
    else:
        print("\nAnalysis Preview:\n")
        print("No analyzable stream rows were produced.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull public NBA data from multiple providers, normalize it, and run one analysis pass."
    )
    parser.add_argument("--limit", type=int, default=3, help="Markets per provider to include, default: 3")
    parser.add_argument("--rounds", type=int, default=2, help="Polling rounds for the standardized stream, default: 2")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between polling rounds, default: 1.0")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "cross_provider_analysis",
        help="Directory for normalized snapshot, stream, and analysis CSVs",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main())
