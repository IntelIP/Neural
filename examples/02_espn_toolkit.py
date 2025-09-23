"""
ESPN Toolkit using the neural data collection module.

This example demonstrates how to create custom data sources for ESPN APIs
to gather games, scores, news, and real-time updates for analysis.
"""

import sys
import os
from typing import Dict, Any, Optional

# Add the neural package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neural.data_collection import RestApiSource, DataTransformer, register_source


# Custom ESPN data sources
@register_source()
class ESPNNFLScoreboard(RestApiSource):
    """Real-time NFL scoreboard data."""

    def __init__(self, interval: float = 30.0, dates: Optional[str] = None):
        params = {}
        if dates:
            params["dates"] = dates
        super().__init__(
            name="espn_nfl_scoreboard",
            url="http://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard",
            params=params,
            interval=interval  # Configurable polling interval
        )


@register_source()
class ESPNCollegeFootballScoreboard(RestApiSource):
    """College football scoreboard with games and scores."""

    def __init__(self, groups: str = "80"):  # FBS by default
        super().__init__(
            name="espn_college_football_scoreboard",
            url="http://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard",
            params={"groups": groups},
            interval=60.0
        )


@register_source()
class ESPNNFLNews(RestApiSource):
    """NFL news and analytics from ESPN."""

    def __init__(self):
        super().__init__(
            name="espn_nfl_news",
            url="http://site.api.espn.com/apis/site/v2/sports/football/nfl/news",
            interval=300.0  # News updates every 5 minutes
        )


@register_source()
class ESPNNBAScoreboard(RestApiSource):
    """NBA real-time scoreboard."""

    def __init__(self):
        super().__init__(
            name="espn_nba_scoreboard",
            url="http://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
            interval=30.0
        )


@register_source()
class ESPNCollegeFootballRankings(RestApiSource):
    """College football rankings and analytics."""

    def __init__(self):
        super().__init__(
            name="espn_college_football_rankings",
            url="http://site.api.espn.com/apis/site/v2/sports/football/college-football/rankings",
            interval=3600.0  # Rankings update hourly
        )


@register_source()
class ESPNGameSummary(RestApiSource):
    """Real-time game summary with play-by-play data."""

    def __init__(self, game_id: str, sport: str = "football/nfl", interval: float = 10.0):
        super().__init__(
            name=f"espn_game_summary_{game_id}",
            url=f"http://site.api.espn.com/apis/site/v2/sports/{sport}/summary",
            params={"event": game_id},
            interval=interval  # Poll every 10 seconds for real-time updates
        )


# Custom transformers for ESPN data
espn_scoreboard_transformer = DataTransformer([
    DataTransformer.flatten_keys,  # Flatten nested structures
    lambda data: {k: v for k, v in data.items() if k in ['events', 'leagues', 'season']},  # Filter relevant fields
])

espn_news_transformer = DataTransformer([
    lambda data: {k: v for k, v in data.items() if k in ['articles', 'header']},
])

espn_rankings_transformer = DataTransformer([
    DataTransformer.flatten_keys,
])

espn_game_summary_transformer = DataTransformer([
    lambda data: {k: v for k, v in data.items() if k in ['header', 'drives', 'scoringPlays', 'pickcenter']},  # Focus on game details and plays
    DataTransformer.flatten_keys,
])


# Register transformers
from neural.data_collection import registry
registry.transformers["espn_nfl_scoreboard"] = espn_scoreboard_transformer
registry.transformers["espn_college_football_scoreboard"] = espn_scoreboard_transformer
registry.transformers["espn_nba_scoreboard"] = espn_scoreboard_transformer
registry.transformers["espn_nfl_news"] = espn_news_transformer
registry.transformers["espn_college_football_rankings"] = espn_rankings_transformer
registry.transformers["espn_game_summary"] = espn_game_summary_transformer


async def collect_nfl_data(interval: float = 30.0):
    """Collect NFL scoreboard data."""
    source = ESPNNFLScoreboard(interval=interval)
    transformer = registry.get_transformer("espn_nfl_scoreboard")

    async with source:
        async for raw_data in source.collect():
            transformed = transformer.transform(raw_data)
            print(f"NFL Data: {len(transformed.get('events', []))} games")
            # Process for analysis
            break  # Just one sample


async def find_ravens_lions_game(interval: float = 5.0):
    """Find Ravens vs Lions game on 9/22."""
    source = ESPNNFLScoreboard(interval=interval)
    transformer = registry.get_transformer("espn_nfl_scoreboard")

    async with source:
        async for raw_data in source.collect():
            transformed = transformer.transform(raw_data)
            events = transformed.get('events', [])
            for event in events:
                competitors = event.get('competitions', [{}])[0].get('competitors', [])
                if len(competitors) == 2:
                    team1 = competitors[0].get('team', {}).get('name', '')
                    team2 = competitors[1].get('team', {}).get('name', '')
                    if ('Ravens' in team1 and 'Lions' in team2) or ('Lions' in team1 and 'Ravens' in team2):
                        print(f"Found Ravens vs Lions game: {event}")
                        return event
            print(f"No Ravens vs Lions game found in {len(events)} events")
            break


async def collect_college_football_data():
    """Collect college football data."""
    source = ESPNCollegeFootballScoreboard()
    transformer = registry.get_transformer("espn_college_football_scoreboard")

    async with source:
        async for raw_data in source.collect():
            transformed = transformer.transform(raw_data)
            print(f"College Football: {len(transformed.get('events', []))} games")
            break


async def collect_nba_realtime():
    """Collect real-time NBA scores."""
    source = ESPNNBAScoreboard()
    transformer = registry.get_transformer("espn_nba_scoreboard")

    async with source:
        count = 0
        async for raw_data in source.collect():
            transformed = transformer.transform(raw_data)
            print(f"NBA Real-time: {len(transformed.get('events', []))} games")
            count += 1
            if count >= 3:  # Collect a few updates
                break


async def collect_news_analytics():
    """Collect ESPN news for analytics."""
    source = ESPNNFLNews()
    transformer = registry.get_transformer("espn_nfl_news")

    async with source:
        async for raw_data in source.collect():
            transformed = transformer.transform(raw_data)
            articles = transformed.get('articles', [])
            print(f"NFL News: {len(articles)} articles")
            if articles:
                print(f"Latest: {articles[0].get('headline', 'N/A')}")
            break


async def collect_ravens_lions_play_by_play(game_id: str = "401671000", interval: float = 10.0):
    """Collect real-time play-by-play for Ravens vs Lions."""
    source = ESPNGameSummary(game_id=game_id, interval=interval)
    transformer = registry.get_transformer("espn_game_summary")

    async with source:
        async for raw_data in source.collect():
            transformed = transformer.transform(raw_data)
            drives = transformed.get('drives', [])
            print(f"Game Summary: {len(drives)} drives")
            if drives:
                # Show latest drive plays
                latest_drive = drives[-1]
                plays = latest_drive.get('plays', [])
                print(f"Latest Drive: {len(plays)} plays")
                for play in plays[-3:]:  # Last 3 plays
                    print(f"- {play.get('text', 'N/A')}")
            break  # Just one sample


async def collect_past_game_play_by_play(dates: str = "20240915-20240921"):
    """Collect play-by-play for a past NFL game."""
    # Get past scoreboard
    scoreboard_source = ESPNNFLScoreboard(dates=dates, interval=60.0)
    transformer = registry.get_transformer("espn_nfl_scoreboard")

    game_id = None
    async with scoreboard_source:
        async for raw_data in scoreboard_source.collect():
            transformed = transformer.transform(raw_data)
            events = transformed.get('events', [])
            # Pick the first completed game
            for event in events:
                status = event.get('status', {}).get('type', {}).get('completed', False)
                if status:
                    game_id = event.get('id')
                    print(f"Found past game: {event.get('shortName', 'N/A')} (ID: {game_id})")
                    break
            break

    if game_id:
        # Now get play-by-play for that game
        summary_source = ESPNGameSummary(game_id=game_id, interval=60.0)
        summary_transformer = registry.get_transformer("espn_game_summary")

        async with summary_source:
            async for raw_data in summary_source.collect():
                transformed = summary_transformer.transform(raw_data)
                drives = transformed.get('drives', [])
                print(f"Past Game Play-by-Play: {len(drives)} drives")
                total_plays = sum(len(drive.get('plays', [])) for drive in drives)
                print(f"Total Plays: {total_plays}")
                if drives:
                    # Show first drive's plays as example
                    first_drive = drives[0]
                    plays = first_drive.get('plays', [])
                    print(f"First Drive Plays ({len(plays)}):")
                    for play in plays[:5]:  # First 5 plays
                        print(f"- {play.get('text', 'N/A')}")
                break


async def collect_chiefs_giants_play_by_play(game_id: str = "401772920"):
    """Collect play-by-play for Chiefs vs Giants using known game ID."""
    print(f"Fetching play-by-play for game ID: {game_id}")

    summary_source = ESPNGameSummary(game_id=game_id, interval=60.0)
    summary_transformer = registry.get_transformer("espn_game_summary")

    async with summary_source:
        async for raw_data in summary_source.collect():
            transformed = summary_transformer.transform(raw_data)
            drives = transformed.get('drives', [])
            print(f"Game Play-by-Play: {len(drives)} drives")
            total_plays = sum(len(drive.get('plays', [])) for drive in drives)
            print(f"Total Plays: {total_plays}")
            if drives:
                # Show scoring plays
                scoring_plays = []
                for drive in drives:
                    for play in drive.get('plays', []):
                        if 'field goal' in play.get('text', '').lower() or 'touchdown' in play.get('text', '').lower():
                            scoring_plays.append(play.get('text', 'N/A'))
                print("Scoring Plays:")
                for play in scoring_plays:
                    print(f"- {play}")
                # Show final score from header
                header = transformed.get('header', {})
                if 'competitions' in header:
                    comp = header['competitions'][0]
                    home = comp.get('competitors', [])[0]
                    away = comp.get('competitors', [])[1]
                    home_score = home.get('score', 'N/A')
                    away_score = away.get('score', 'N/A')
                    home_name = home.get('team', {}).get('name', 'Home')
                    away_name = away.get('team', {}).get('name', 'Away')
                    print(f"Final Score: {away_name} {away_score}, {home_name} {home_score}")
            else:
                print("No drives available")
            break


async def main():
    """Run ESPN toolkit examples."""
    print("=== ESPN Data Collection Toolkit Demo ===\n")

    print("1. Finding Ravens vs Lions game (polling every 5 seconds)...")
    try:
        game = await find_ravens_lions_game(interval=5.0)
        if game:
            print("Game details:")
            print(f"- ID: {game.get('id')}")
            print(f"- Date: {game.get('date')}")
            print(f"- Status: {game.get('status', {}).get('type', {}).get('description')}")
            competitors = game.get('competitions', [{}])[0].get('competitors', [])
            for comp in competitors:
                team = comp.get('team', {})
                score = comp.get('score', 'N/A')
                print(f"- {team.get('name')} ({team.get('abbreviation')}): {score}")
    except Exception as e:
        print(f"Game search failed: {e}")

    print("\n2. Collecting NFL Scoreboard (30s interval)...")
    try:
        await collect_nfl_data(interval=30.0)
    except Exception as e:
        print(f"NFL collection failed: {e}")

    print("\n3. Collecting College Football...")
    try:
        await collect_college_football_data()
    except Exception as e:
        print(f"College Football collection failed: {e}")

    print("\n4. Collecting Real-time NBA Scores...")
    try:
        await collect_nba_realtime()
    except Exception as e:
        print(f"NBA collection failed: {e}")

    print("\n5. Collecting ESPN News/Analytics...")
    try:
        await collect_news_analytics()
    except Exception as e:
        print(f"News collection failed: {e}")

    print("\n6. Collecting Ravens vs Lions Play-by-Play...")
    try:
        await collect_ravens_lions_play_by_play(interval=10.0)
    except Exception as e:
        print(f"Play-by-play collection failed: {e}")

    print("\n7. Collecting Past Game Play-by-Play...")
    try:
        await collect_past_game_play_by_play()
    except Exception as e:
        print(f"Past game collection failed: {e}")

    print("\n8. Collecting Chiefs vs Giants Play-by-Play (9/21/25)...")
    try:
        await collect_chiefs_giants_play_by_play("20250921")
    except Exception as e:
        print(f"Chiefs vs Giants collection failed: {e}")

    print("\n=== Toolkit Usage ===")
    print("To use in your algorithms:")
    print("- Instantiate sources: source = ESPNNFLScoreboard(interval=5.0)")
    print("- For specific date: source = ESPNNFLScoreboard(dates='20250921-20250921')")
    print("- For play-by-play: source = ESPNGameSummary(game_id='GAME_ID', interval=10.0)")
    print("- Use async context: async with source:")
    print("- Collect data: async for data in source.collect():")
    print("- Transform: transformer.transform(data)")
    print("- Integrate into your analysis pipelines")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())