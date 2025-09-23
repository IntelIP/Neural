#!/usr/bin/env python3
"""
ESPN Sports Data Example

This example demonstrates how to use the Neural SDK to fetch sports data
from ESPN's API for NFL, College Football, and NBA.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any

from neural.sports import ESPNNFL, ESPNCFB, ESPNNBA
from neural.data_collection import DataPipeline, TransformStage, create_logger


# Configure logging
logger = create_logger("espn_example", level="INFO")


class ScoreEnrichmentStage(TransformStage):
    """Transform stage to enrich score data with additional context."""
    
    async def process(self, data: Any) -> Any:
        """Add win probability and game status context."""
        if not isinstance(data, dict) or "events" not in data:
            return data
        
        for event in data.get("events", []):
            # Add game context
            if "competitions" in event and event["competitions"]:
                competition = event["competitions"][0]
                
                # Calculate score differential
                if "competitors" in competition and len(competition["competitors"]) == 2:
                    team1 = competition["competitors"][0]
                    team2 = competition["competitors"][1]
                    
                    score1 = int(team1.get("score", 0))
                    score2 = int(team2.get("score", 0))
                    
                    event["score_differential"] = abs(score1 - score2)
                    event["is_close_game"] = event["score_differential"] <= 7
                    
                    # Determine leader
                    if score1 > score2:
                        event["leader"] = team1["team"]["displayName"]
                    elif score2 > score1:
                        event["leader"] = team2["team"]["displayName"]
                    else:
                        event["leader"] = "Tied"
        
        return data


async def nfl_example():
    """Demonstrate NFL data fetching."""
    logger.info("=" * 50)
    logger.info("NFL Example")
    logger.info("=" * 50)
    
    # Initialize NFL client
    nfl = ESPNNFL()
    
    # Connect to the data source
    await nfl.connect()
    
    try:
        # Get current week's scoreboard
        logger.info("Fetching NFL scoreboard...")
        scores = await nfl.get_scoreboard()
        
        if "events" in scores:
            logger.info(f"Found {len(scores['events'])} NFL games")
            
            for event in scores["events"][:3]:  # Show first 3 games
                name = event.get("name", "Unknown")
                status = event.get("status", {}).get("type", {}).get("name", "Unknown")
                logger.info(f"  - {name}: {status}")
        
        # Get specific team roster
        logger.info("\nFetching Green Bay Packers roster...")
        roster = await nfl.get_team_roster("GB")
        
        if "athletes" in roster:
            logger.info(f"Packers roster has {len(roster['athletes'])} players")
        
        # Get team schedule
        logger.info("\nFetching team schedule...")
        schedule = await nfl.get_schedule("GB")
        
        if "events" in schedule:
            logger.info(f"Found {len(schedule['events'])} games in schedule")
        
        # Get NFL standings
        logger.info("\nFetching NFL standings...")
        standings = await nfl.get_standings()
        
        if "children" in standings:
            for conference in standings["children"][:2]:
                logger.info(f"  Conference: {conference.get('name', 'Unknown')}")
        
    finally:
        await nfl.disconnect()


async def cfb_example():
    """Demonstrate College Football data fetching."""
    logger.info("\n" + "=" * 50)
    logger.info("College Football Example")
    logger.info("=" * 50)
    
    # Initialize CFB client
    cfb = ESPNCFB()
    
    await cfb.connect()
    
    try:
        # Get scoreboard for SEC games
        logger.info("Fetching SEC games...")
        scores = await cfb.get_scoreboard(conference="SEC")
        
        if "events" in scores:
            logger.info(f"Found {len(scores['events'])} SEC games")
            
            for event in scores["events"][:3]:  # Show first 3 games
                name = event.get("name", "Unknown")
                logger.info(f"  - {name}")
        
        # Get rankings
        logger.info("\nFetching CFB rankings...")
        rankings = await cfb.get_rankings()
        
        if "rankings" in rankings:
            for poll in rankings["rankings"]:
                poll_name = poll.get("name", "Unknown Poll")
                logger.info(f"\n{poll_name}:")
                
                if "ranks" in poll:
                    for rank in poll["ranks"][:5]:  # Top 5
                        team = rank.get("team", {}).get("name", "Unknown")
                        position = rank.get("rank", "?")
                        logger.info(f"  #{position}: {team}")
        
        # Get conferences
        logger.info("\nAvailable conferences:")
        conferences = await cfb.get_conferences()
        
        for conf in conferences.get("conferences", [])[:5]:
            logger.info(f"  - {conf['name']} (ID: {conf['id']})")
        
    finally:
        await cfb.disconnect()


async def nba_example():
    """Demonstrate NBA data fetching."""
    logger.info("\n" + "=" * 50)
    logger.info("NBA Example")
    logger.info("=" * 50)
    
    # Initialize NBA client
    nba = ESPNNBA()
    
    await nba.connect()
    
    try:
        # Get today's games
        logger.info("Fetching today's NBA games...")
        today = datetime.now().strftime("%Y%m%d")
        scores = await nba.get_scoreboard(dates=today)
        
        if "events" in scores:
            logger.info(f"Found {len(scores['events'])} NBA games today")
            
            for event in scores["events"][:5]:  # Show first 5 games
                name = event.get("shortName", "Unknown")
                status = event.get("status", {}).get("type", {}).get("name", "Unknown")
                logger.info(f"  - {name}: {status}")
        
        # Get standings
        logger.info("\nFetching NBA standings...")
        standings = await nba.get_standings()
        
        if "children" in standings:
            for conference in standings["children"]:
                conf_name = conference.get("name", "Unknown")
                logger.info(f"\n{conf_name} Conference:")
                
                # Show top teams
                for division in conference.get("children", []):
                    div_name = division.get("name", "Unknown")
                    logger.info(f"  {div_name}:")
                    
                    entries = division.get("standings", {}).get("entries", [])
                    for entry in entries[:3]:  # Top 3 teams
                        team = entry.get("team", {}).get("displayName", "Unknown")
                        stats = entry.get("stats", [])
                        
                        # Extract wins and losses
                        wins = losses = 0
                        for stat in stats:
                            if stat.get("name") == "wins":
                                wins = stat.get("value", 0)
                            elif stat.get("name") == "losses":
                                losses = stat.get("value", 0)
                        
                        logger.info(f"    - {team}: {wins}-{losses}")
        
        # Get specific team info
        logger.info("\nFetching Lakers information...")
        lakers = await nba.get_team("LAL")
        
        if "team" in lakers:
            team_info = lakers["team"]
            logger.info(f"  Name: {team_info.get('displayName', 'Unknown')}")
            logger.info(f"  Location: {team_info.get('location', 'Unknown')}")
            logger.info(f"  Abbreviation: {team_info.get('abbreviation', 'Unknown')}")
        
    finally:
        await nba.disconnect()


async def pipeline_example():
    """Demonstrate using multiple sports sources in a pipeline."""
    logger.info("\n" + "=" * 50)
    logger.info("Data Pipeline Example")
    logger.info("=" * 50)
    
    # Create data pipeline
    pipeline = DataPipeline()
    
    # Initialize sports clients
    nfl = ESPNNFL()
    cfb = ESPNCFB()
    nba = ESPNNBA()
    
    # Add sources to pipeline
    await pipeline.add_source("nfl", nfl)
    await pipeline.add_source("cfb", cfb)
    await pipeline.add_source("nba", nba)
    
    # Add enrichment stage
    pipeline.add_stage(ScoreEnrichmentStage())
    
    # Define consumer to process all sports data
    async def process_scores(data: Dict[str, Any]):
        """Process enriched score data."""
        if "events" in data:
            for event in data["events"]:
                if event.get("is_close_game"):
                    name = event.get("shortName", "Unknown")
                    leader = event.get("leader", "Unknown")
                    diff = event.get("score_differential", 0)
                    logger.info(f"Close game alert: {name} - {leader} leads by {diff}")
    
    # Add consumer
    await pipeline.add_consumer(process_scores)
    
    # Connect all sources
    await nfl.connect()
    await cfb.connect()
    await nba.connect()
    
    try:
        # Start pipeline
        await pipeline.start()
        
        # Fetch data from all sources
        logger.info("Fetching data from all sports...")
        
        # Trigger data fetches
        nfl_scores = await nfl.get_scoreboard()
        cfb_scores = await cfb.get_scoreboard()
        nba_scores = await nba.get_scoreboard()
        
        # Process through pipeline
        await pipeline.process("nfl", nfl_scores)
        await pipeline.process("cfb", cfb_scores)
        await pipeline.process("nba", nba_scores)
        
        # Get pipeline stats
        stats = pipeline.get_stats()
        logger.info(f"\nPipeline processed {stats.get('messages_processed', 0)} messages")
        
    finally:
        # Clean shutdown
        await pipeline.stop()
        await nfl.disconnect()
        await cfb.disconnect()
        await nba.disconnect()


async def main():
    """Main example function."""
    logger.info("ESPN Sports Data Example using Neural SDK")
    logger.info("=========================================\n")
    
    try:
        # Run individual sport examples
        await nfl_example()
        await cfb_example()
        await nba_example()
        
        # Run pipeline example
        await pipeline_example()
        
        logger.info("\n" + "=" * 50)
        logger.info("Example completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in example: {e}")
        raise


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())