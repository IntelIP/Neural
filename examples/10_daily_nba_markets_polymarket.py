"""
Daily Polymarket US NBA market snapshot example.

Fetches current Polymarket US NBA markets with the Neural SDK and writes
the snapshot to a dated CSV file for downstream reporting or automation.

Public market reads do not require credentials. If Polymarket US API
credentials are configured locally, the same SDK adapter can also be used for
authenticated account endpoints elsewhere in the SDK.

Examples:
    uv run python examples/10_daily_nba_markets_polymarket.py
    uv run python examples/10_daily_nba_markets_polymarket.py --sport nba --status open --limit 200
    uv run python examples/10_daily_nba_markets_polymarket.py --output-dir data/polymarket_nba_snapshots
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running the example directly from the repository root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neural.data_collection import PolymarketUSMarketsSource


def _hydrate_quotes(source: PolymarketUSMarketsSource, markets):
    if markets.empty or "market_id" not in markets.columns:
        return markets

    markets = markets.copy()
    last_prices = []
    volumes = []
    yes_prices = []
    no_prices = []
    for market_id in markets["market_id"]:
        quote = source.adapter.get_quote(str(market_id))
        yes_prices.append(quote.yes_ask if quote.yes_ask is not None else quote.yes_bid)
        no_prices.append(quote.no_ask if quote.no_ask is not None else quote.no_bid)
        last_prices.append(quote.last_price)
        volumes.append(quote.volume)

    markets["yes_price"] = yes_prices
    markets["no_price"] = no_prices
    markets["last_price"] = last_prices
    markets["volume"] = volumes
    return markets


def fetch_daily_polymarket_markets(
    *,
    sport: str,
    status: str | None,
    limit: int,
    output_dir: Path,
    sports_only: bool,
) -> Path:
    source = PolymarketUSMarketsSource(
        config={
            "sport": sport,
            "limit": limit,
            "sports_only": sports_only,
        }
    )
    try:
        markets = source.get_markets_df()
        markets = _hydrate_quotes(source, markets)
    finally:
        source.adapter.close()

    if status and not markets.empty and "status" in markets.columns:
        status_mask = markets["status"].astype(str).str.lower() == status.lower()
        markets = markets[status_mask].reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    output_path = output_dir / f"polymarket_{sport.lower()}_markets_{snapshot_date}.csv"
    markets.to_csv(output_path, index=False)

    print(
        f"Wrote {len(markets)} Polymarket {sport.upper()} markets to {output_path.as_posix()}"
    )
    if not markets.empty:
        preview_columns = [
            column
            for column in (
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
                "sport",
                "category",
            )
            if column in markets.columns
        ]
        if preview_columns:
            print()
            print(markets[preview_columns].head(10).to_string(index=False))

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch and store a daily Polymarket US sports market snapshot."
    )
    parser.add_argument("--sport", default="nba", help="Sport/league filter, default: nba")
    parser.add_argument(
        "--status",
        default="open",
        help="Optional status filter applied after fetch, default: open",
    )
    parser.add_argument("--limit", type=int, default=200, help="Maximum markets to fetch, default: 200")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "polymarket_nba_snapshots",
        help="Directory for dated CSV snapshots, default: data/polymarket_nba_snapshots",
    )
    parser.add_argument(
        "--all-markets",
        action="store_true",
        help="Use the generic markets endpoint instead of the sports-only endpoint",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fetch_daily_polymarket_markets(
        sport=args.sport,
        status=args.status,
        limit=args.limit,
        output_dir=args.output_dir,
        sports_only=not args.all_markets,
    )


if __name__ == "__main__":
    main()
