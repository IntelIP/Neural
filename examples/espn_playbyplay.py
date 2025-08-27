"""
Example: ESPN Play-by-Play Data Streaming
Demonstrates how to stream and analyze play-by-play data
"""

import asyncio
import logging
from data_pipeline.data_sources.espn import ESPNClient, ESPNStreamAdapter, EventType
from data_pipeline.utils import setup_logging


async def fetch_game_data():
    """Example of fetching play-by-play data for a specific game"""
    setup_logging(level="INFO")
    logger = logging.getLogger(__name__)
    
    client = ESPNClient()
    
    try:
        # Get live NFL games
        logger.info("Fetching live NFL games...")
        live_games = await client.get_live_games("nfl")
        
        if not live_games:
            logger.info("No live games found. Fetching recent games...")
            scoreboard = await client.get_scoreboard("nfl")
            games = scoreboard.get('events', [])[:3]  # Get first 3 games
            
            for game in games:
                game_id = game['id']
                logger.info(f"\nGame: {game.get('name', 'Unknown')}")
                
                # Get play-by-play
                pbp = await client.get_play_by_play(game_id)
                
                logger.info(f"  Total Plays: {pbp['total_plays']}")
                logger.info(f"  Scoring Plays: {len(pbp['scoring_plays'])}")
                logger.info(f"  High Impact Plays: {len(pbp['high_impact_plays'])}")
                
                # Show some high-impact plays
                for play in pbp['high_impact_plays'][:3]:
                    logger.info(f"    → {play.text} (Impact: {play.impact_score:.2f})")
        else:
            # Analyze live game
            game = live_games[0]
            game_id = game['id']
            logger.info(f"\nLive Game: {game['name']}")
            logger.info(f"  Status: Q{game['period']} - {game['clock']}")
            
            # Get detailed data
            pbp = await client.get_play_by_play(game_id)
            game_state = pbp['game_state']
            
            logger.info("\nGame State:")
            logger.info(f"  Score: {game_state.home_team} {game_state.home_score} - {game_state.away_team} {game_state.away_score}")
            logger.info(f"  Win Probability: {game_state.home_team} {game_state.home_win_probability:.1f}%")
            logger.info(f"  Situation: {game_state.down} & {game_state.distance} at {game_state.yard_line}")
            
            # Get recent high-impact plays
            logger.info("\nRecent High-Impact Plays:")
            for play in pbp['high_impact_plays'][-5:]:
                logger.info(f"  {play.quarter}Q {play.clock}: {play.text[:100]}...")
                logger.info(f"    Events: {[e.value for e in play.events]}")
                logger.info(f"    Impact Score: {play.impact_score:.2f}")
            
            # Get injuries
            injuries = await client.get_injuries(game_id)
            if injuries:
                logger.info("\nInjuries (Market Impact):")
                for injury in injuries:
                    logger.info(f"  {injury['player']} ({injury['team']}): {injury['status']}")
    
    finally:
        await client.close()


async def stream_live_games():
    """Example of streaming live game events"""
    setup_logging(level="INFO")
    logger = logging.getLogger(__name__)
    
    # Event handler
    async def handle_game_event(event):
        # Only log high-impact events
        if event.impact_score > 0.5:
            logger.info("\n🎯 Game Event Detected!")
            logger.info(f"  Type: {event.type.value}")
            logger.info(f"  Description: {event.description}")
            logger.info(f"  Impact Score: {event.impact_score:.2f}")
            logger.info(f"  Game State: {event.game_state.home_team} {event.game_state.home_score} - {event.game_state.away_team} {event.game_state.away_score}")
            logger.info(f"  Win Prob: {event.game_state.home_team} {event.game_state.home_win_probability:.1f}%")
            
            # Special alerts for critical events
            if event.type == EventType.TOUCHDOWN:
                logger.info("  💥 TOUCHDOWN! Major market impact expected!")
            elif event.type == EventType.INTERCEPTION:
                logger.info("  🔄 INTERCEPTION! Momentum shift detected!")
            elif event.type == EventType.INJURY:
                logger.info("  🚑 INJURY! Check for market overreaction!")
    
    # Create stream adapter
    stream = ESPNStreamAdapter(on_event=handle_game_event)
    
    try:
        # Start streaming
        await stream.start()
        
        # Monitor all NFL games
        logger.info("Starting NFL game monitoring...")
        await stream.monitor_sport("nfl")
        
        # Run for 5 minutes
        logger.info("Streaming live events for 5 minutes...")
        await asyncio.sleep(300)
        
    finally:
        await stream.stop()


async def analyze_game_momentum():
    """Example of analyzing game momentum and critical situations"""
    setup_logging(level="INFO")
    logger = logging.getLogger(__name__)
    
    client = ESPNClient()
    
    try:
        # Get a game
        scoreboard = await client.get_scoreboard("nfl")
        games = scoreboard.get('events', [])
        
        if not games:
            logger.info("No games available")
            return
        
        game = games[0]
        game_id = game['id']
        logger.info(f"Analyzing: {game.get('name', 'Unknown')}")
        
        # Get win probability history
        win_prob_history = await client.get_win_probability_history(game_id)
        
        if win_prob_history:
            logger.info("\nWin Probability Swings:")
            
            # Find biggest swings
            biggest_swing = 0
            swing_play = None
            
            for i in range(1, len(win_prob_history)):
                prev = win_prob_history[i-1]
                curr = win_prob_history[i]
                
                swing = abs(curr['home_win_percentage'] - prev['home_win_percentage'])
                if swing > biggest_swing:
                    biggest_swing = swing
                    swing_play = curr
            
            if swing_play:
                logger.info(f"  Biggest Swing: {biggest_swing:.1f}%")
                logger.info(f"  Play: {swing_play['play_text'][:100]}...")
            
            # Current probability
            if win_prob_history:
                current = win_prob_history[-1]
                logger.info("\nCurrent Win Probability:")
                logger.info(f"  Home: {current['home_win_percentage']:.1f}%")
                logger.info(f"  Away: {current['away_win_percentage']:.1f}%")
        
        # Get game leaders
        leaders = await client.get_game_leaders(game_id)
        
        if leaders:
            logger.info("\nGame Leaders (Key Players):")
            for category, players in leaders.items():
                if players:
                    leader = players[0]
                    logger.info(f"  {category}: {leader['player']} - {leader['displayValue']}")
        
        # Get scoring plays
        scoring_plays = await client.get_scoring_plays(game_id)
        
        if scoring_plays:
            logger.info("\nScoring Summary:")
            for play in scoring_plays[-5:]:  # Last 5 scores
                logger.info(f"  Q{play['quarter']} {play['clock']}: {play['text'][:80]}...")
                logger.info(f"    Score after: {play['home_score']}-{play['away_score']}")
    
    finally:
        await client.close()


async def main():
    """Run all examples"""
    print("=" * 60)
    print("ESPN Play-by-Play Examples")
    print("=" * 60)
    
    print("\n1. Fetching Game Data...")
    print("-" * 40)
    await fetch_game_data()
    
    print("\n2. Analyzing Game Momentum...")
    print("-" * 40)
    await analyze_game_momentum()
    
    print("\n3. Streaming Live Events (30 seconds demo)...")
    print("-" * 40)
    
    # Create a shorter streaming demo
    stream = ESPNStreamAdapter()
    await stream.start()
    await stream.monitor_sport("nfl")
    await asyncio.sleep(30)  # 30 second demo
    await stream.stop()
    
    print("\n✅ Examples completed!")


if __name__ == "__main__":
    asyncio.run(main())