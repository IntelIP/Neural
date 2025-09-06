#!/usr/bin/env python3
"""
REST Data Sources Demo

Demonstrates the unified REST API data source framework with
Kalshi, ESPN, and Weather integrations.
"""

import asyncio
import logging
from datetime import datetime
import os
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from neural_sdk.data_sources.kalshi.rest_adapter import KalshiRESTAdapter
from neural_sdk.data_sources.espn.rest_adapter import ESPNRESTAdapter
from neural_sdk.data_sources.weather.rest_adapter import WeatherRESTAdapter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demo_kalshi_rest():
    """Demonstrate Kalshi REST API adapter."""
    print("\n" + "="*60)
    print("🎲 KALSHI REST API DEMO")
    print("="*60)
    
    try:
        # Initialize Kalshi adapter
        kalshi = KalshiRESTAdapter()
        
        # Connect
        await kalshi.connect()
        
        # Get NFL markets
        print("\n📊 Fetching NFL markets...")
        nfl_markets = await kalshi.get_nfl_markets()
        
        if "data" in nfl_markets and "markets" in nfl_markets["data"]:
            markets = nfl_markets["data"]["markets"][:5]  # Show first 5
            print(f"Found {len(nfl_markets['data']['markets'])} NFL markets")
            
            for market in markets:
                print(f"\n  • {market.get('title', 'Unknown')}")
                print(f"    Ticker: {market.get('ticker')}")
                print(f"    Yes Price: ${market.get('yes_bid', 0):.2f}")
                print(f"    Status: {market.get('status')}")
        else:
            print("No NFL markets found")
        
        # Get stats
        stats = kalshi.get_stats()
        print(f"\n📈 Kalshi Stats:")
        print(f"  • Total Requests: {stats['total_requests']}")
        print(f"  • Cache Hit Rate: {stats['cache_hit_rate']:.1%}")
        print(f"  • Average Latency: {stats['average_latency']:.3f}s")
        
        # Disconnect
        await kalshi.disconnect()
        
    except Exception as e:
        logger.error(f"Kalshi demo error: {e}")
        print(f"❌ Kalshi demo failed: {e}")


async def demo_espn_rest():
    """Demonstrate ESPN REST API adapter."""
    print("\n" + "="*60)
    print("🏈 ESPN REST API DEMO")
    print("="*60)
    
    try:
        # Initialize ESPN adapter
        espn = ESPNRESTAdapter()
        
        # Connect
        await espn.connect()
        
        # Get NFL scoreboard
        print("\n📋 Fetching NFL scoreboard...")
        scoreboard = await espn.get_nfl_games()
        
        if "games" in scoreboard:
            games = scoreboard["games"][:3]  # Show first 3
            print(f"Found {len(scoreboard['games'])} NFL games")
            
            for game in games:
                print(f"\n  • {game.get('name', 'Unknown')}")
                print(f"    Status: {game.get('status')}")
                
                home = game.get("home_team", {})
                away = game.get("away_team", {})
                
                if home and away:
                    print(f"    {away.get('name', 'Away')}: {away.get('score', 0)}")
                    print(f"    {home.get('name', 'Home')}: {home.get('score', 0)}")
        else:
            print("No games found")
        
        # Get stats
        stats = espn.get_stats()
        print(f"\n📈 ESPN Stats:")
        print(f"  • Total Requests: {stats['total_requests']}")
        print(f"  • Cache Hit Rate: {stats['cache_hit_rate']:.1%}")
        
        # Disconnect
        await espn.disconnect()
        
    except Exception as e:
        logger.error(f"ESPN demo error: {e}")
        print(f"❌ ESPN demo failed: {e}")


async def demo_weather_rest():
    """Demonstrate Weather REST API adapter."""
    print("\n" + "="*60)
    print("🌤️ WEATHER REST API DEMO")
    print("="*60)
    
    try:
        # Initialize Weather adapter
        weather = WeatherRESTAdapter()
        
        # Check if API key is configured
        if not weather.api_key:
            print("⚠️ No OpenWeatherMap API key configured")
            print("Set OPENWEATHER_API_KEY environment variable to enable weather features")
            return
        
        # Connect
        await weather.connect()
        
        # Get weather for NFL stadiums
        print("\n🏟️ Fetching weather for outdoor NFL stadiums...")
        
        # Demo with a few stadiums
        stadiums = ["Lambeau Field", "Arrowhead Stadium", "Soldier Field"]
        
        for stadium_name in stadiums:
            print(f"\n  • {stadium_name}")
            
            result = await weather.get_nfl_stadium_weather(stadium_name)
            
            if "parsed" in result:
                w = result["parsed"]
                print(f"    Team: {result.get('team', 'Unknown')}")
                print(f"    Temperature: {w['temperature']['actual']:.1f}°F")
                print(f"    Wind: {w['wind']['speed']:.1f} mph")
                print(f"    Conditions: {w['conditions']['description']}")
                
                if "impact" in result:
                    impact = result["impact"]
                    print(f"    Impact: {impact['summary']}")
                    
                    # Show betting considerations
                    if impact.get("betting_considerations", {}).get("recommendations"):
                        print("    Betting notes:")
                        for rec in impact["betting_considerations"]["recommendations"]:
                            print(f"      - {rec}")
            else:
                print(f"    ❌ Could not get weather data")
        
        # Get stats
        stats = weather.get_stats()
        print(f"\n📈 Weather Stats:")
        print(f"  • Total Requests: {stats['total_requests']}")
        print(f"  • Cache Hit Rate: {stats['cache_hit_rate']:.1%}")
        
        # Disconnect
        await weather.disconnect()
        
    except Exception as e:
        logger.error(f"Weather demo error: {e}")
        print(f"❌ Weather demo failed: {e}")


async def demo_unified_data():
    """Demonstrate unified data fetching from multiple sources."""
    print("\n" + "="*60)
    print("🔄 UNIFIED DATA FETCHING DEMO")
    print("="*60)
    
    try:
        # Initialize all adapters
        kalshi = KalshiRESTAdapter()
        espn = ESPNRESTAdapter()
        weather = WeatherRESTAdapter()
        
        # Connect all
        await asyncio.gather(
            kalshi.connect(),
            espn.connect(),
            weather.connect() if weather.api_key else asyncio.sleep(0)
        )
        
        print("\n📊 Fetching data from all sources concurrently...")
        
        # Fetch from all sources in parallel
        results = await asyncio.gather(
            kalshi.get_nfl_markets(),
            espn.get_nfl_games(),
            weather.get_nfl_stadium_weather("Lambeau Field") if weather.api_key else asyncio.sleep(0),
            return_exceptions=True
        )
        
        kalshi_data, espn_data, weather_data = results
        
        # Process results
        print("\n✅ Data fetched from all sources:")
        
        if not isinstance(kalshi_data, Exception) and "data" in kalshi_data:
            print(f"  • Kalshi: {len(kalshi_data['data'].get('markets', []))} markets")
        
        if not isinstance(espn_data, Exception) and "games" in espn_data:
            print(f"  • ESPN: {len(espn_data['games'])} games")
        
        if weather.api_key and not isinstance(weather_data, Exception) and "parsed" in weather_data:
            w = weather_data["parsed"]
            print(f"  • Weather: {w['temperature']['actual']:.1f}°F at Lambeau Field")
        
        # Show combined stats
        print("\n📈 Combined Statistics:")
        total_requests = 0
        total_cache_hits = 0
        
        for adapter in [kalshi, espn, weather]:
            stats = adapter.get_stats()
            total_requests += stats['total_requests']
            total_cache_hits += stats['cache_hits']
        
        print(f"  • Total API Requests: {total_requests}")
        print(f"  • Total Cache Hits: {total_cache_hits}")
        print(f"  • Overall Cache Rate: {total_cache_hits/total_requests:.1%}" if total_requests > 0 else "N/A")
        
        # Disconnect all
        await asyncio.gather(
            kalshi.disconnect(),
            espn.disconnect(),
            weather.disconnect() if weather.api_key else asyncio.sleep(0)
        )
        
    except Exception as e:
        logger.error(f"Unified demo error: {e}")
        print(f"❌ Unified demo failed: {e}")


async def main():
    """Run all REST data source demos."""
    print("\n" + "="*60)
    print("🚀 NEURAL SDK - REST DATA SOURCES DEMO")
    print("="*60)
    print("\nThis demo shows the unified REST API framework with:")
    print("  • Kalshi market data")
    print("  • ESPN sports data")
    print("  • Weather impact analysis")
    print("  • Unified data fetching")
    
    # Run individual demos
    await demo_kalshi_rest()
    await demo_espn_rest()
    await demo_weather_rest()
    
    # Run unified demo
    await demo_unified_data()
    
    print("\n" + "="*60)
    print("✅ REST DATA SOURCES DEMO COMPLETE")
    print("="*60)
    print("\nKey Features Demonstrated:")
    print("  ✓ Unified REST interface for all sources")
    print("  ✓ Automatic authentication handling")
    print("  ✓ Rate limiting and caching")
    print("  ✓ Concurrent data fetching")
    print("  ✓ Weather impact analysis for betting")
    print("  ✓ Error handling and retries")


if __name__ == "__main__":
    asyncio.run(main())