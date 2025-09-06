#!/usr/bin/env python3
"""
The Odds API Demo

Demonstrates comprehensive sports betting odds retrieval and analysis.
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neural_sdk.data_sources.odds.rest_adapter import OddsAPIAdapter
from neural_sdk.data_sources.odds.models import Sport, Market


async def demo_basic_odds():
    """Demonstrate basic odds retrieval."""
    print("\n" + "="*60)
    print("📊 BASIC ODDS RETRIEVAL")
    print("="*60)
    
    # Initialize adapter
    adapter = OddsAPIAdapter()
    await adapter.connect()
    
    # Get NFL odds
    print("\n🏈 Fetching NFL odds...")
    nfl_odds = await adapter.get_nfl_odds()
    
    if nfl_odds:
        print(f"Found {len(nfl_odds)} NFL games with odds")
        
        # Display first game
        game = nfl_odds[0]
        print(f"\n📌 {game.away_team} @ {game.home_team}")
        print(f"   Start: {game.commence_time.strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"   Bookmakers: {len(game.bookmakers)}")
        
        # Show consensus odds
        consensus = game.get_consensus_moneyline()
        print(f"\n   Consensus Moneyline:")
        for team, odds in consensus.items():
            print(f"     {team}: {odds:+.0f}")
        
        # Show best odds
        best = game.get_best_odds("h2h")
        print(f"\n   Best Moneyline Odds:")
        for team, details in best.items():
            print(f"     {team}: {details['price']:+.0f} ({details['bookmaker']})")
    else:
        print("No NFL games found")
    
    # Show API usage
    stats = adapter.get_usage_stats()
    print(f"\n📈 API Usage:")
    print(f"   Requests Used: {stats['requests_used']}")
    print(f"   Requests Remaining: {stats['requests_remaining']}")
    
    await adapter.disconnect()


async def demo_arbitrage_finder():
    """Find arbitrage opportunities."""
    print("\n" + "="*60)
    print("💰 ARBITRAGE OPPORTUNITY FINDER")
    print("="*60)
    
    adapter = OddsAPIAdapter()
    await adapter.connect()
    
    # Check multiple sports
    sports = [Sport.NFL.value, Sport.NBA.value, Sport.NCAAF.value]
    
    for sport in sports:
        sport_name = sport.split("_")[-1].upper()
        print(f"\n🔍 Checking {sport_name} for arbitrage...")
        
        opportunities = await adapter.find_arbitrage_opportunities(sport)
        
        if opportunities:
            print(f"   ✅ Found {len(opportunities)} arbitrage opportunities!")
            
            for opp in opportunities[:3]:  # Show first 3
                arb = opp["arbitrage"]
                print(f"\n   📌 {opp['game']}")
                print(f"      Profit Margin: {arb['profit_margin']:.2f}%")
                print(f"      Bet 1: {arb['home_bet']['team']} @ {arb['home_bet']['odds']:+.0f}")
                print(f"             at {arb['home_bet']['bookmaker']}")
                print(f"             Stake: {arb['home_bet']['stake_percentage']:.1%}")
                print(f"      Bet 2: {arb['away_bet']['team']} @ {arb['away_bet']['odds']:+.0f}")
                print(f"             at {arb['away_bet']['bookmaker']}")
                print(f"             Stake: {arb['away_bet']['stake_percentage']:.1%}")
        else:
            print(f"   ❌ No arbitrage opportunities found")
    
    await adapter.disconnect()


async def demo_best_odds_comparison():
    """Compare odds across bookmakers."""
    print("\n" + "="*60)
    print("🎯 BEST ODDS COMPARISON")
    print("="*60)
    
    adapter = OddsAPIAdapter()
    await adapter.connect()
    
    # Get NFL games
    print("\n🏈 Analyzing NFL odds across bookmakers...")
    best_odds = await adapter.get_best_odds(Sport.NFL.value, "spreads")
    
    if best_odds:
        print(f"Found best spreads for {len(best_odds)} games")
        
        # Show first 3 games
        for game_id, details in list(best_odds.items())[:3]:
            print(f"\n📌 {details['game']}")
            print(f"   Start: {details['commence_time'].strftime('%m/%d %H:%M')}")
            print("   Best Spreads:")
            
            for team, odds_info in details["best_odds"].items():
                if odds_info["point"]:
                    print(f"     {team}: {odds_info['point']:+.1f} @ {odds_info['price']:+.0f}")
                    print(f"              ({odds_info['bookmaker']})")
    
    await adapter.disconnect()


async def demo_line_movement():
    """Track line movements (simulated)."""
    print("\n" + "="*60)
    print("📈 LINE MOVEMENT TRACKING")
    print("="*60)
    
    adapter = OddsAPIAdapter()
    await adapter.connect()
    
    print("\n⏱️ Monitoring line movements...")
    print("(This would normally run for hours, showing 1 check)")
    
    # Get initial odds
    nfl_odds = await adapter.get_nfl_odds()
    
    if nfl_odds:
        game = nfl_odds[0]
        print(f"\n📌 Tracking: {game.away_team} @ {game.home_team}")
        
        # Track initial lines
        for bookmaker in game.bookmakers[:3]:  # First 3 bookmakers
            moneyline = bookmaker.get_market("h2h")
            if moneyline:
                for outcome in moneyline.outcomes:
                    adapter.track_line_movement(
                        game.id,
                        bookmaker.title,
                        "h2h",
                        outcome.name,
                        outcome.price
                    )
        
        # Show tracked movements
        print("\n   Initial Lines Tracked:")
        for key, movement in list(adapter.line_movements.items())[:3]:
            parts = key.split(":")
            print(f"     {parts[1]} - {parts[3]}: {movement.movements[-1]['price']:+.0f}")
            print(f"     Trend: {movement.get_trend()}")
    
    await adapter.disconnect()


async def demo_multi_sport_overview():
    """Get overview of multiple sports."""
    print("\n" + "="*60)
    print("🏆 MULTI-SPORT ODDS OVERVIEW")
    print("="*60)
    
    adapter = OddsAPIAdapter()
    await adapter.connect()
    
    # Get available sports
    print("\n📋 Available Sports:")
    sports = await adapter.get_sports()
    
    active_sports = [s for s in sports if s.get("active")]
    print(f"Found {len(active_sports)} active sports")
    
    # Check games for each major sport
    major_sports = [
        ("NFL", Sport.NFL.value),
        ("NBA", Sport.NBA.value),
        ("NCAAF", Sport.NCAAF.value),
        ("MLB", Sport.MLB.value),
        ("NHL", Sport.NHL.value)
    ]
    
    for sport_name, sport_key in major_sports:
        # Check if sport is in season
        if not any(s["key"] == sport_key for s in active_sports):
            continue
            
        odds = await adapter.get_odds(sport_key, markets="h2h")
        
        if odds:
            print(f"\n{sport_name}: {len(odds)} games")
            
            # Show next game
            next_game = min(odds, key=lambda x: x.commence_time)
            print(f"  Next: {next_game.away_team} @ {next_game.home_team}")
            print(f"        {next_game.commence_time.strftime('%m/%d %H:%M UTC')}")
            
            # Average number of bookmakers
            avg_bookmakers = sum(len(g.bookmakers) for g in odds) / len(odds)
            print(f"  Avg Bookmakers: {avg_bookmakers:.1f}")
    
    await adapter.disconnect()


async def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("🎰 THE ODDS API - COMPREHENSIVE DEMO")
    print("="*60)
    print("Demonstrating sports betting odds retrieval and analysis")
    print(f"API Key: {'Set' if os.getenv('ODDS_API_KEY') else 'Using default'}")
    
    # Run demos
    await demo_basic_odds()
    await demo_best_odds_comparison()
    await demo_arbitrage_finder()
    await demo_line_movement()
    await demo_multi_sport_overview()
    
    print("\n" + "="*60)
    print("✅ ODDS API DEMO COMPLETE")
    print("="*60)
    print("\nKey Features Demonstrated:")
    print("  ✓ Real-time odds from multiple bookmakers")
    print("  ✓ Best odds comparison across books")
    print("  ✓ Arbitrage opportunity detection")
    print("  ✓ Line movement tracking")
    print("  ✓ Multi-sport support")
    print("  ✓ API quota management with caching")


if __name__ == "__main__":
    asyncio.run(main())