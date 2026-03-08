"""
Daily NBA market snapshot example.

Fetches current Kalshi NBA game markets with the Neural SDK and writes
the snapshot to a dated CSV file for downstream reporting or automation.

Examples:
    python examples/09_daily_nba_markets.py
    python examples/09_daily_nba_markets.py --status open --limit 200
    python examples/09_daily_nba_markets.py --output-dir data/nba_snapshots
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running the example directly from the repository root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neural.data_collection import get_nba_games


async def fetch_daily_nba_markets(
    *,
    status: str,
    limit: int,
    output_dir: Path,
    use_authenticated: bool,
) -> Path:
    markets = await get_nba_games(
        status=status,
        limit=limit,
        use_authenticated=use_authenticated,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    output_path = output_dir / f"nba_markets_{snapshot_date}.csv"
    markets.to_csv(output_path, index=False)

    today_markets = 0
    if not markets.empty and "game_date" in markets.columns:
        today = datetime.now().date()
        today_markets = int(markets["game_date"].dt.date.eq(today).sum())

    print(f"Wrote {len(markets)} NBA markets to {output_path.as_posix()}")
    if "game_date" in markets.columns:
        print(f"Markets dated for today: {today_markets}")
    if not markets.empty:
        preview_columns = [
            column
            for column in ("ticker", "title", "home_team", "away_team", "game_date", "yes_ask", "volume")
            if column in markets.columns
        ]
        if preview_columns:
            print()
            print(markets[preview_columns].head(10).to_string(index=False))

    return output_path



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and store a daily NBA market snapshot.")
    parser.add_argument("--status", default="open", help="Kalshi market status filter, default: open")
    parser.add_argument("--limit", type=int, default=200, help="Maximum markets to fetch, default: 200")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "nba_snapshots",
        help="Directory for dated CSV snapshots, default: data/nba_snapshots",
    )
    parser.add_argument(
        "--authenticated",
        action="store_true",
        help="Use authenticated Kalshi API access instead of the public market endpoint",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    asyncio.run(
        fetch_daily_nba_markets(
            status=args.status,
            limit=args.limit,
            output_dir=args.output_dir,
            use_authenticated=args.authenticated,
        )
    )


if __name__ == "__main__":
    main()
