#!/usr/bin/env python3
"""
Poll REST API for 15 seconds and return pricing data
"""

import asyncio
from datetime import datetime

from neural.trading.rest_streaming import stream_via_rest


async def poll_seahawks_cardinals():
    """Poll market data for 15 seconds"""

    # Market tickers
    sea_ticker = "KXNFLGAME-25SEP25SEAARI-SEA"
    ari_ticker = "KXNFLGAME-25SEP25SEAARI-ARI"

    print("🏈 Polling Seahawks vs Cardinals Markets")
    print("=" * 60)
    print("Duration: 15 seconds")
    print("Poll interval: 1 second")
    print("\nMarkets:")
    print(f"  • {sea_ticker}")
    print(f"  • {ari_ticker}")
    print("\n" + "-" * 60)

    # Collect data
    df = await stream_via_rest(
        tickers=[sea_ticker, ari_ticker],
        duration_seconds=15,
        poll_interval=1.0,
        on_update=lambda snapshot: print(
            f"[{datetime.now().strftime('%H:%M:%S')}] "
            f"{'SEA' if 'SEA' in snapshot.ticker else 'ARI'}: "
            f"${snapshot.yes_mid:.3f} ({snapshot.implied_probability:.1f}%) "
            f"Spread: ${snapshot.yes_spread:.3f}"
        ),
    )

    return df


# Run the polling
print("\n🚀 Starting REST API polling...\n")
df = asyncio.run(poll_seahawks_cardinals())

print("\n" + "=" * 60)
print("📊 PRICING DATA SUMMARY")
print("=" * 60)

if not df.empty:
    # Analyze Seattle data
    sea_data = df[df["ticker"].str.contains("SEA")]
    if not sea_data.empty:
        print("\n🏈 Seattle Seahawks:")
        print(f"  Data points: {len(sea_data)}")
        start_price = sea_data.iloc[0]["yes_mid"]
        start_prob = sea_data.iloc[0]["implied_prob"]
        print(f"  Starting price: ${start_price:.3f} ({start_prob:.1f}%)")

        end_price = sea_data.iloc[-1]["yes_mid"]
        end_prob = sea_data.iloc[-1]["implied_prob"]
        print(f"  Ending price: ${end_price:.3f} ({end_prob:.1f}%)")
        print(f"  Min price: ${sea_data['yes_mid'].min():.3f}")
        print(f"  Max price: ${sea_data['yes_mid'].max():.3f}")
        print(f"  Avg spread: ${sea_data['yes_spread'].mean():.3f}")

        price_change = sea_data.iloc[-1]["yes_mid"] - sea_data.iloc[0]["yes_mid"]
        if abs(price_change) > 0.001:
            direction = "📈" if price_change > 0 else "📉"
            movement_cents = price_change * 100
        print(f"  Movement: {direction} ${abs(price_change):.3f} ({movement_cents:+.1f}¢)")

    # Analyze Arizona data
    ari_data = df[df["ticker"].str.contains("ARI")]
    if not ari_data.empty:
        print("\n🏈 Arizona Cardinals:")
        print(f"  Data points: {len(ari_data)}")
        ari_start_price = ari_data.iloc[0]["yes_mid"]
        ari_start_prob = ari_data.iloc[0]["implied_prob"]
        print(f"  Starting price: ${ari_start_price:.3f} ({ari_start_prob:.1f}%)")

        ari_end_price = ari_data.iloc[-1]["yes_mid"]
        ari_end_prob = ari_data.iloc[-1]["implied_prob"]
        print(f"  Ending price: ${ari_end_price:.3f} ({ari_end_prob:.1f}%)")
        print(f"  Min price: ${ari_data['yes_mid'].min():.3f}")
        print(f"  Max price: ${ari_data['yes_mid'].max():.3f}")
        print(f"  Avg spread: ${ari_data['yes_spread'].mean():.3f}")

        price_change = ari_data.iloc[-1]["yes_mid"] - ari_data.iloc[0]["yes_mid"]
        if abs(price_change) > 0.001:
            direction = "📈" if price_change > 0 else "📉"
            movement_cents = price_change * 100
        print(f"  Movement: {direction} ${abs(price_change):.3f} ({movement_cents:+.1f}¢)")

    # Combined analysis
    if not sea_data.empty and not ari_data.empty:
        print("\n📊 Combined Analysis:")
        sea_prob = sea_data.iloc[-1]["implied_prob"]
        ari_prob = ari_data.iloc[-1]["implied_prob"]
        total = sea_prob + ari_prob

        print(f"  Total probability: {total:.1f}%")
        if total < 98:
            arbitrage_profit = 100 - total
            print(f"  💰 ARBITRAGE OPPORTUNITY: {arbitrage_profit:.1f}% profit potential")
        elif total > 102:
            print(f"  ⚠️ OVERPRICED: Total exceeds 100% by {total - 100:.1f}%")

        # Volume comparison
        sea_vol = sea_data.iloc[-1]["volume"]
        ari_vol = ari_data.iloc[-1]["volume"]
        print("\n📈 Volume:")
        print(f"  Seattle: {sea_vol:,} contracts")
        print(f"  Arizona: {ari_vol:,} contracts")
        print(f"  Total: {sea_vol + ari_vol:,} contracts")

    # Save to CSV
    filename = f"pricing_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False)
    print(f"\n💾 Full data saved to: {filename}")

    # Show raw data sample
    print("\n📋 Raw Data Sample (last 3 updates per team):")
    print("-" * 60)
    for ticker in df["ticker"].unique():
        ticker_data = df[df["ticker"] == ticker].tail(3)
        team = "Seattle" if "SEA" in ticker else "Arizona"
        print(f"\n{team}:")
        for _, row in ticker_data.iterrows():
            print(
                f"  [{row['timestamp']}] ${row['yes_mid']:.3f} "
                f"(Bid: ${row['yes_bid']:.3f}, Ask: ${row['yes_ask']:.3f})"
            )

else:
    print("❌ No data collected")

print("\n" + "=" * 60)
