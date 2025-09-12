#!/usr/bin/env python3
"""
Check NFL and CFB games for today, tomorrow, and Sunday.
"""

import asyncio
from datetime import datetime, timedelta
import json

from neural.sports import ESPNNFL, ESPNCFB


async def check_games():
    """Check games for specified dates."""
    
    # Calculate dates
    today = datetime.now()
    tomorrow = today + timedelta(days=1)
    
    # Find next Sunday
    days_until_sunday = (6 - today.weekday()) % 7
    if days_until_sunday == 0 and today.hour >= 20:  # If it's Sunday evening, get next Sunday
        days_until_sunday = 7
    sunday = today + timedelta(days=days_until_sunday)
    
    # Format dates for ESPN API
    today_str = today.strftime("%Y%m%d")
    tomorrow_str = tomorrow.strftime("%Y%m%d")
    sunday_str = sunday.strftime("%Y%m%d")
    
    print("=" * 60)
    print(f"Checking games for:")
    print(f"  Today:    {today.strftime('%A, %B %d, %Y')} ({today_str})")
    print(f"  Tomorrow: {tomorrow.strftime('%A, %B %d, %Y')} ({tomorrow_str})")
    print(f"  Sunday:   {sunday.strftime('%A, %B %d, %Y')} ({sunday_str})")
    print("=" * 60)
    
    # Initialize clients (no need to connect - auto-connects on first request)
    nfl = ESPNNFL()
    cfb = ESPNCFB()
    
    try:
        # Check NFL games
        print("\n🏈 NFL GAMES")
        print("-" * 40)
        
        for date_label, date_str in [("TODAY", today_str), ("TOMORROW", tomorrow_str), ("SUNDAY", sunday_str)]:
            print(f"\n{date_label}:")
            try:
                scores = await nfl.get_scoreboard(dates=date_str)
                
                if "events" in scores and scores["events"]:
                    for event in scores["events"]:
                        # Extract game info
                        name = event.get("name", "Unknown")
                        status = event.get("status", {}).get("type", {}).get("name", "Unknown")
                        
                        # Get time
                        date_str_full = event.get("date", "")
                        if date_str_full:
                            game_time = datetime.fromisoformat(date_str_full.replace('Z', '+00:00'))
                            time_str = game_time.strftime("%I:%M %p ET")
                        else:
                            time_str = "TBD"
                        
                        # Get scores if game is in progress or completed
                        if "competitions" in event and event["competitions"]:
                            competition = event["competitions"][0]
                            if "competitors" in competition and len(competition["competitors"]) == 2:
                                away = competition["competitors"][1]  # Away team is usually index 1
                                home = competition["competitors"][0]  # Home team is usually index 0
                                
                                away_team = away.get("team", {}).get("abbreviation", "???")
                                home_team = home.get("team", {}).get("abbreviation", "???")
                                away_score = away.get("score", "-")
                                home_score = home.get("score", "-")
                                
                                if status in ["STATUS_IN_PROGRESS", "STATUS_HALFTIME", "STATUS_END_PERIOD"]:
                                    print(f"  {away_team} {away_score} @ {home_team} {home_score} - {status}")
                                elif status == "STATUS_FINAL":
                                    print(f"  {away_team} {away_score} @ {home_team} {home_score} - FINAL")
                                else:
                                    print(f"  {away_team} @ {home_team} - {time_str}")
                        else:
                            print(f"  {name} - {time_str}")
                else:
                    print(f"  No games scheduled")
                    
            except Exception as e:
                print(f"  Error fetching data: {e}")
        
        # Check CFB games
        print("\n\n🏈 COLLEGE FOOTBALL GAMES")
        print("-" * 40)
        
        for date_label, date_str in [("TODAY", today_str), ("TOMORROW", tomorrow_str), ("SUNDAY", sunday_str)]:
            print(f"\n{date_label}:")
            try:
                # Get all FBS games
                scores = await cfb.get_scoreboard(dates=date_str)
                
                if "events" in scores and scores["events"]:
                    # Group games by status
                    in_progress = []
                    scheduled = []
                    final = []
                    
                    for event in scores["events"]:
                        # Extract game info
                        name = event.get("name", "Unknown")
                        status = event.get("status", {}).get("type", {}).get("name", "Unknown")
                        
                        # Get time
                        date_str_full = event.get("date", "")
                        if date_str_full:
                            game_time = datetime.fromisoformat(date_str_full.replace('Z', '+00:00'))
                            time_str = game_time.strftime("%I:%M %p ET")
                        else:
                            time_str = "TBD"
                        
                        # Get scores and teams
                        if "competitions" in event and event["competitions"]:
                            competition = event["competitions"][0]
                            if "competitors" in competition and len(competition["competitors"]) == 2:
                                away = competition["competitors"][1]
                                home = competition["competitors"][0]
                                
                                away_team = away.get("team", {}).get("displayName", "???")
                                home_team = home.get("team", {}).get("displayName", "???")
                                away_score = away.get("score", "-")
                                home_score = home.get("score", "-")
                                
                                # Get rankings if available
                                away_rank = away.get("curatedRank", {}).get("current")
                                home_rank = home.get("curatedRank", {}).get("current")
                                
                                if away_rank and away_rank <= 25:
                                    away_team = f"#{away_rank} {away_team}"
                                if home_rank and home_rank <= 25:
                                    home_team = f"#{home_rank} {home_team}"
                                
                                game_info = {
                                    "away": away_team,
                                    "home": home_team,
                                    "away_score": away_score,
                                    "home_score": home_score,
                                    "time": time_str,
                                    "status": status
                                }
                                
                                if status in ["STATUS_IN_PROGRESS", "STATUS_HALFTIME", "STATUS_END_PERIOD"]:
                                    in_progress.append(game_info)
                                elif status == "STATUS_FINAL":
                                    final.append(game_info)
                                else:
                                    scheduled.append(game_info)
                    
                    # Display games by category
                    if in_progress:
                        print("  LIVE:")
                        for game in in_progress:
                            print(f"    {game['away']} {game['away_score']} @ {game['home']} {game['home_score']}")
                    
                    if scheduled:
                        print(f"  SCHEDULED ({len(scheduled)} games):")
                        for game in scheduled[:10]:  # Show first 10
                            print(f"    {game['away']} @ {game['home']} - {game['time']}")
                        if len(scheduled) > 10:
                            print(f"    ... and {len(scheduled) - 10} more games")
                    
                    if final:
                        print(f"  FINAL ({len(final)} games):")
                        for game in final[:5]:  # Show first 5
                            print(f"    {game['away']} {game['away_score']} @ {game['home']} {game['home_score']}")
                        if len(final) > 5:
                            print(f"    ... and {len(final) - 5} more games")
                else:
                    print(f"  No games scheduled")
                    
            except Exception as e:
                print(f"  Error fetching data: {e}")
        
    finally:
        # Cleanup connections if they were created
        if nfl.session:
            await nfl.disconnect()
        if cfb.session:
            await cfb.disconnect()
    
    print("\n" + "=" * 60)
    print("Game check complete!")


if __name__ == "__main__":
    asyncio.run(check_games())