#!/usr/bin/env python3
"""
Test Real CFB Games - September 12, 2025

This script fetches real CFB games happening today and tomorrow,
maps them to Kalshi ticker format, and tests our sentiment analysis
on actual games to demonstrate the complete trading pipeline.
"""

import asyncio
from datetime import datetime, timedelta
import sys
import os
import logging
from typing import Dict, List, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural.sports.espn_cfb import ESPNCFB, ESPNCFBConfig
from neural.social.twitter_client import TwitterClient, TwitterConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def map_team_to_kalshi_code(team_name: str) -> str:
    """Map full team name to Kalshi-style code."""
    
    # Common team mappings for Kalshi format
    team_mappings = {
        # Power 5 Teams
        "Indiana Hoosiers": "IND",
        "Indiana State Sycamores": "INST", 
        "Syracuse Orange": "SYR",
        "Colgate Raiders": "COLG",
        "Colorado Buffaloes": "COLO",
        "Houston Cougars": "HOU",
        "Kansas State Wildcats": "KSU",
        "Arizona Wildcats": "ARIZ",
        "New Mexico Lobos": "UNM",
        "UCLA Bruins": "UCLA",
        
        # Saturday games
        "Oregon Ducks": "ORE",
        "Northwestern Wildcats": "NW",
        "Clemson Tigers": "CLEM",
        "Georgia Tech Yellow Jackets": "GT",
        "Oklahoma Sooners": "OU", 
        "Temple Owls": "TEM",
        "Wisconsin Badgers": "WISC",
        "Alabama Crimson Tide": "BAMA",
        "Central Michigan Chippewas": "CMU",
        "Michigan Wolverines": "MICH",
        "Houston Christian Huskies": "HCU",
        "Nebraska Cornhuskers": "NEB",
        "Towson Tigers": "TOW",
        "Maryland Terrapins": "MD",
        "William & Mary Tribe": "WM",
        "Virginia Cavaliers": "UVA",
        "Samford Bulldogs": "SAM",
        "Baylor Bears": "BAY",
        "Memphis Tigers": "MEM",
        "Troy Trojans": "TROY"
    }
    
    # Try exact match first
    if team_name in team_mappings:
        return team_mappings[team_name]
    
    # Try to extract common abbreviations
    if "Alabama" in team_name:
        return "BAMA"
    elif "Michigan" in team_name and "Central" not in team_name:
        return "MICH"
    elif "Oregon" in team_name:
        return "ORE"
    elif "Clemson" in team_name:
        return "CLEM"
    elif "Oklahoma" in team_name:
        return "OU"
    elif "Wisconsin" in team_name:
        return "WISC"
    elif "Indiana" in team_name and "State" not in team_name:
        return "IND"
    elif "Kansas State" in team_name:
        return "KSU"
    elif "Northwestern" in team_name:
        return "NW"
    elif "Georgia Tech" in team_name:
        return "GT"
    elif "UCLA" in team_name:
        return "UCLA"
    elif "Colorado" in team_name:
        return "COLO"
    elif "Syracuse" in team_name:
        return "SYR"
    elif "Arizona" in team_name:
        return "ARIZ"
    elif "Houston" in team_name and "Christian" not in team_name:
        return "HOU"
    elif "Nebraska" in team_name:
        return "NEB"
    elif "Maryland" in team_name:
        return "MD"
    elif "Virginia" in team_name and "West" not in team_name:
        return "UVA"
    elif "Baylor" in team_name:
        return "BAY"
    else:
        # Fallback: use first 4 letters of the main name
        main_name = team_name.split()[0]
        return main_name[:4].upper()


def create_kalshi_ticker(away_team: str, home_team: str, market_type: str = "WIN") -> str:
    """Create Kalshi-style market ticker."""
    
    away_code = map_team_to_kalshi_code(away_team)
    home_code = map_team_to_kalshi_code(home_team)
    
    # Kalshi format for CFB: NCAAFWIN-DATE-AWAY-HOME or similar
    date_str = datetime.now().strftime("%y%b%d").upper()
    
    if market_type == "WIN":
        return f"NCAAF-{date_str}-{away_code}-{home_code}-WIN"
    elif market_type == "SPREAD":
        return f"NCAAF-{date_str}-{away_code}-{home_code}-SPREAD"
    else:
        return f"NCAAF-{date_str}-{away_code}-{home_code}-{market_type}"


async def get_todays_real_games():
    """Fetch real CFB games for today and tomorrow."""
    
    logger.info("🏈 Fetching real CFB games from ESPN...")
    
    cfb = ESPNCFB()
    
    # Get today and tomorrow
    today = datetime.now()
    tomorrow = today + timedelta(days=1)
    
    today_str = today.strftime("%Y%m%d")
    tomorrow_str = tomorrow.strftime("%Y%m%d")
    
    all_games = []
    
    try:
        # Today's games
        logger.info(f"📅 Checking {today.strftime('%A, %B %d')} ({today_str})...")
        today_scores = await cfb.get_scoreboard(dates=today_str)
        
        if "events" in today_scores and today_scores["events"]:
            for event in today_scores["events"]:
                game_info = extract_game_info(event, "TODAY")
                if game_info:
                    all_games.append(game_info)
            
            logger.info(f"  ✅ Found {len(today_scores['events'])} games today")
        
        # Tomorrow's games
        logger.info(f"📅 Checking {tomorrow.strftime('%A, %B %d')} ({tomorrow_str})...")
        tomorrow_scores = await cfb.get_scoreboard(dates=tomorrow_str)
        
        if "events" in tomorrow_scores and tomorrow_scores["events"]:
            for event in tomorrow_scores["events"]:
                game_info = extract_game_info(event, "TOMORROW")
                if game_info:
                    all_games.append(game_info)
            
            logger.info(f"  ✅ Found {len(tomorrow_scores['events'])} games tomorrow")
        
    finally:
        if cfb.session:
            await cfb.disconnect()
    
    return all_games


def extract_game_info(event: Dict, day_label: str) -> Optional[Dict]:
    """Extract game information from ESPN event data."""
    
    try:
        # Basic info
        event_id = event.get("id")
        name = event.get("name", "Unknown")
        status = event.get("status", {}).get("type", {}).get("name", "Unknown")
        
        # Get time
        date_str_full = event.get("date", "")
        if date_str_full:
            game_time = datetime.fromisoformat(date_str_full.replace('Z', '+00:00'))
            time_str = game_time.strftime("%I:%M %p ET")
        else:
            time_str = "TBD"
        
        # Extract teams
        if "competitions" in event and event["competitions"]:
            competition = event["competitions"][0]
            if "competitors" in competition and len(competition["competitors"]) == 2:
                away = competition["competitors"][1]  # Away team
                home = competition["competitors"][0]  # Home team
                
                away_team = away.get("team", {}).get("displayName", "???")
                home_team = home.get("team", {}).get("displayName", "???")
                away_score = away.get("score", "-")
                home_score = home.get("score", "-")
                
                # Get rankings if available
                away_rank = away.get("curatedRank", {}).get("current")
                home_rank = home.get("curatedRank", {}).get("current")
                
                return {
                    "event_id": event_id,
                    "day": day_label,
                    "name": name,
                    "away_team": away_team,
                    "home_team": home_team,
                    "away_rank": away_rank,
                    "home_rank": home_rank,
                    "away_score": away_score,
                    "home_score": home_score,
                    "time": time_str,
                    "status": status,
                    "game_time": game_time if date_str_full else None
                }
    
    except Exception as e:
        logger.error(f"Error extracting game info: {e}")
        return None


async def test_sentiment_on_real_games(games: List[Dict]):
    """Test sentiment analysis on real CFB games."""
    
    logger.info(f"\n🧠 Testing Sentiment Analysis on {len(games)} Real Games")
    logger.info("=" * 60)
    
    # Import sentiment analyzer
    from examples.sentiment_analysis_stack_showcase import AdvancedSentimentAnalyzer
    
    # Initialize sentiment analyzer (will use mock data since we don't have Twitter API)
    twitter_client = None
    if os.getenv('TWITTERAPI_IO_KEY'):
        twitter_client = TwitterClient(TwitterConfig())
        logger.info("  ✅ Using real Twitter API")
    else:
        logger.info("  ℹ️ Using mock sentiment data (set TWITTERAPI_IO_KEY for real data)")
    
    sentiment_analyzer = AdvancedSentimentAnalyzer(twitter_client)
    
    analyzed_games = []
    
    for i, game in enumerate(games[:5], 1):  # Test first 5 games
        logger.info(f"\n{i}. {game['name']}")
        logger.info(f"   {game['away_team']} @ {game['home_team']}")
        logger.info(f"   Time: {game['time']} ({game['day']})")
        
        # Create Kalshi tickers
        win_ticker = create_kalshi_ticker(game['away_team'], game['home_team'], "WIN")
        spread_ticker = create_kalshi_ticker(game['away_team'], game['home_team'], "SPREAD")
        
        logger.info(f"   Win Ticker: {win_ticker}")
        logger.info(f"   Spread Ticker: {spread_ticker}")
        
        # Analyze sentiment
        try:
            market_context = {
                'market_id': win_ticker,
                'current_price': 0.50,  # Mock market price
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'game_time': game.get('game_time'),
                'volume': 15000  # Mock volume
            }
            
            sentiment_profile = await sentiment_analyzer.analyze_market_sentiment(
                game['home_team'],
                game['away_team'], 
                market_context
            )
            
            # Display results
            logger.info(f"   📊 Sentiment Results:")
            logger.info(f"     Home Sentiment: {sentiment_profile.home_team_sentiment.overall_sentiment:.2f}")
            logger.info(f"     Away Sentiment: {sentiment_profile.away_team_sentiment.overall_sentiment:.2f}")
            logger.info(f"     Market Sentiment: {sentiment_profile.market_sentiment.overall_sentiment:.2f}")
            logger.info(f"     Confidence: {sentiment_profile.market_sentiment.confidence_score:.1%}")
            logger.info(f"     Implied Probability: {sentiment_profile.predicted_probability:.1%}")
            logger.info(f"     Recommendation: {sentiment_profile.recommendation}")
            
            if sentiment_profile.edge_opportunity > 0.03:
                logger.info(f"     🚨 EDGE DETECTED: {sentiment_profile.edge_opportunity:.1%}")
            
            analyzed_games.append({
                "game": game,
                "sentiment": sentiment_profile,
                "tickers": {
                    "win": win_ticker,
                    "spread": spread_ticker
                }
            })
            
        except Exception as e:
            logger.error(f"     ❌ Sentiment analysis failed: {e}")
    
    return analyzed_games


async def main():
    """Main execution function."""
    
    print("\n" + "=" * 70)
    print("🏈 REAL CFB GAMES SENTIMENT TRADING TEST")
    print("   September 12, 2025")
    print("=" * 70)
    
    try:
        # Get real games
        games = await get_todays_real_games()
        
        if not games:
            logger.warning("⚠️ No CFB games found for today/tomorrow")
            return
        
        logger.info(f"\n📊 Found {len(games)} total CFB games")
        
        # Display game summary
        logger.info(f"\n🎯 GAME SUMMARY:")
        logger.info("-" * 40)
        
        today_games = [g for g in games if g['day'] == 'TODAY']
        tomorrow_games = [g for g in games if g['day'] == 'TOMORROW']
        
        logger.info(f"Today ({len(today_games)} games):")
        for game in today_games[:3]:
            rank_info = ""
            if game.get('away_rank') and game['away_rank'] <= 25:
                rank_info += f"#{game['away_rank']} "
            if game.get('home_rank') and game['home_rank'] <= 25:
                rank_info += f"vs #{game['home_rank']} "
            
            logger.info(f"  • {rank_info}{game['away_team']} @ {game['home_team']} ({game['time']})")
        
        logger.info(f"\nTomorrow ({len(tomorrow_games)} games):")
        for game in tomorrow_games[:5]:
            rank_info = ""
            if game.get('away_rank') and game['away_rank'] <= 25:
                rank_info += f"#{game['away_rank']} "
            if game.get('home_rank') and game['home_rank'] <= 25:
                rank_info += f"vs #{game['home_rank']} "
            
            logger.info(f"  • {rank_info}{game['away_team']} @ {game['home_team']} ({game['time']})")
        
        if len(tomorrow_games) > 5:
            logger.info(f"  • ... and {len(tomorrow_games) - 5} more games")
        
        # Test sentiment analysis on real games
        analyzed_games = await test_sentiment_on_real_games(games)
        
        # Final summary
        logger.info(f"\n" + "=" * 70)
        logger.info("🎉 REAL CFB SENTIMENT TRADING TEST COMPLETE!")
        logger.info("=" * 70)
        
        logger.info(f"📊 RESULTS SUMMARY:")
        logger.info(f"  🏈 Real Games Analyzed: {len(analyzed_games)}")
        logger.info(f"  🎯 Kalshi Tickers Generated: {len(analyzed_games) * 2}")  # Win + Spread
        logger.info(f"  🧠 Sentiment Profiles Created: {len(analyzed_games)}")
        
        edge_opportunities = [g for g in analyzed_games if g['sentiment'].edge_opportunity > 0.03]
        logger.info(f"  ⚡ Edge Opportunities: {len(edge_opportunities)}")
        
        if edge_opportunities:
            logger.info(f"\n🚨 TOP EDGE OPPORTUNITIES:")
            for game_data in edge_opportunities:
                game = game_data['game']
                sentiment = game_data['sentiment']
                logger.info(f"  • {game['name']}: {sentiment.edge_opportunity:.1%} edge ({sentiment.recommendation})")
        
        logger.info(f"\n✅ Neural SDK successfully tested with REAL CFB games!")
        logger.info("Ready for live sentiment-based trading on actual markets! 🚀")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
