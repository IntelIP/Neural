#!/usr/bin/env python3
"""End-to-end quickstart script for the Neural SDK.

Fetches live markets, applies a toy signal, and routes simulated orders
through the paper trading client so you can inspect the workflow before
hooking up live execution.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pandas as pd

from neural.auth.http_client import KalshiHTTPClient
from neural.trading.paper_client import PaperTradingClient

SERIES_TICKER = "KXNFLGAME"
FETCH_LIMIT = 20
ORDER_SIZE = 25


@dataclass
class MarketPick:
    ticker: str
    title: str
    yes_ask: float
    no_ask: float
    volume: int


def fetch_markets(series_ticker: str = SERIES_TICKER, limit: int = FETCH_LIMIT) -> pd.DataFrame:
    """Pull the latest markets for a Kalshi series using the authenticated REST client."""
    client = KalshiHTTPClient()
    try:
        payload = client.get(
            "/markets",
            params={
                "series_ticker": series_ticker,
                "status": "open",
                "limit": limit,
            },
        )
    finally:
        client.close()

    markets = payload.get("markets", [])
    if not markets:
        raise RuntimeError("No markets returned; verify credentials and series ticker")

    df = pd.DataFrame(markets)
    expected = {"ticker", "title", "yes_ask", "no_ask", "volume"}
    missing = expected - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing expected keys from API response: {missing}")
    return df


def choose_candidates(df: pd.DataFrame, top_n: int = 3) -> list[MarketPick]:
    """Pick a few markets with the tightest YES/NO spread as a toy "edge" signal."""
    df = df.copy()
    df["yes_spread"] = df["yes_ask"] - df.get("yes_bid", 0)
    narrowed = df.sort_values(["yes_spread", "volume"], ascending=[True, False]).head(top_n)

    picks: list[MarketPick] = []
    for _, row in narrowed.iterrows():
        picks.append(
            MarketPick(
                ticker=row["ticker"],
                title=row["title"],
                yes_ask=row["yes_ask"] / 100,
                no_ask=row["no_ask"] / 100,
                volume=int(row.get("volume", 0)),
            )
        )
    return picks


async def simulate_orders(picks: list[MarketPick]) -> None:
    paper = PaperTradingClient(
        initial_capital=10_000,
        commission_per_trade=0.00,
        slippage_pct=0.001,
        save_trades=False,
    )

    for pick in picks:
        result = await paper.place_order(
            market_id=pick.ticker,
            side="yes",
            quantity=ORDER_SIZE,
            order_type="market",
            market_name=pick.title,
            strategy="QuickStart",
            confidence=0.6,
        )
        status = "FILLED" if result.get("success") else "FAILED"
        price = result.get("filled_price", 0)
        print(
            f"[{status}] {pick.ticker} — {pick.title}\n"
            f"    yes_ask=${pick.yes_ask:.2f} | no_ask=${pick.no_ask:.2f} | volume={pick.volume:,}\n"
            f"    filled_price=${price:.2f}\n"
        )

    summary = paper.portfolio.get_portfolio_metrics()
    print("Portfolio snapshot:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")


async def main() -> None:
    print("📡 Fetching markets...")
    df = fetch_markets()
    print(df[["ticker", "title", "yes_ask", "no_ask", "volume"]].head())

    print("\n🎯 Selecting candidates")
    picks = choose_candidates(df)
    for pick in picks:
        print(f"  - {pick.ticker} | {pick.title} | YES ${pick.yes_ask:.2f} | NO ${pick.no_ask:.2f}")

    print("\n🧪 Simulating trades in paper account\n")
    await simulate_orders(picks)


if __name__ == "__main__":
    asyncio.run(main())
