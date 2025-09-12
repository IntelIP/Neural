#!/usr/bin/env python3
"""
Example: Collect NFL-related tweets using Twitter API.

This script demonstrates how to use the TwitterClient to collect
tweets about NFL teams, games, and players.
"""

import asyncio
from datetime import datetime
import json

from neural.social import TwitterClient, TwitterConfig


async def collect_nfl_tweets():
    """Collect various NFL-related tweets."""
    
    # Initialize Twitter client with custom config
    config = TwitterConfig(
        cache_tweets_ttl=30,  # Short cache for real-time data
        rate_limit_requests=50.0  # Conservative rate limit
    )
    twitter = TwitterClient(config)
    
    print("=" * 60)
    print("NFL Twitter Data Collection")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        # 1. Search for game-specific tweets
        print("\n1. Searching for Packers vs Commanders tweets...")
        game_tweets = await twitter.search_tweets(
            query="Packers Commanders",
            limit=10,
            search_type="Latest"
        )
        
        if "tweets" in game_tweets:
            print(f"   Found {len(game_tweets['tweets'])} tweets")
            for i, tweet in enumerate(game_tweets["tweets"][:3], 1):
                print(f"   Tweet {i}: {tweet.get('text', '')[:100]}...")
        
        # 2. Get NFL official account tweets
        print("\n2. Getting recent tweets from @NFL...")
        nfl_tweets = await twitter.get_user_tweets(
            username="NFL",
            limit=5
        )
        
        if "tweets" in nfl_tweets:
            print(f"   Found {len(nfl_tweets['tweets'])} tweets")
            for tweet in nfl_tweets["tweets"][:2]:
                print(f"   - {tweet.get('text', '')[:100]}...")
        
        # 3. Search for injury news
        print("\n3. Searching for NFL injury updates...")
        injury_tweets = await twitter.search_tweets(
            query="NFL injury report",
            limit=10,
            search_type="Latest"
        )
        
        if "tweets" in injury_tweets:
            print(f"   Found {len(injury_tweets['tweets'])} injury-related tweets")
        
        # 4. Get tweets from ESPN NFL account
        print("\n4. Getting ESPN NFL tweets...")
        espn_tweets = await twitter.get_user_tweets(
            username="ESPNNFL",
            limit=5
        )
        
        if "tweets" in espn_tweets:
            print(f"   Found {len(espn_tweets['tweets'])} ESPN NFL tweets")
        
        # 5. Search for specific team hashtags
        teams = ["#GoPackGo", "#HTTC", "#DallasCowboys", "#ChiefsKingdom"]
        print("\n5. Searching team hashtags...")
        
        for hashtag in teams:
            team_tweets = await twitter.search_tweets(
                query=hashtag,
                limit=5,
                search_type="Top"
            )
            
            if "tweets" in team_tweets:
                print(f"   {hashtag}: {len(team_tweets['tweets'])} tweets found")
        
        # 6. Get trending topics in USA
        print("\n6. Getting trending topics in USA...")
        trends = await twitter.get_trends(woeid=23424977)  # USA WOEID
        
        if "trends" in trends:
            sports_trends = [
                trend for trend in trends["trends"][:10]
                if any(keyword in trend.get("name", "").lower() 
                      for keyword in ["nfl", "football", "game", "team"])
            ]
            
            if sports_trends:
                print("   Sports-related trends:")
                for trend in sports_trends[:5]:
                    print(f"   - {trend.get('name')}")
            else:
                print("   No sports trends in top 10")
        
        # 7. Display API usage costs
        print("\n" + "=" * 60)
        print("API Usage Summary:")
        costs = twitter.get_api_costs()
        print(f"   Total Requests: {costs['requests_count']}")
        print(f"   Tweets Fetched: {costs['tweets_fetched']}")
        print(f"   Users Fetched: {costs['users_fetched']}")
        print(f"   Estimated Cost: ${costs['total_cost']:.6f}")
        print(f"   Avg Cost/Request: ${costs['average_cost_per_request']:.6f}")
        
        # 8. Example of paginated collection
        print("\n7. Collecting paginated data (multiple pages)...")
        all_tweets = await twitter.collect_paginated_data(
            method="search_tweets",
            max_pages=3,
            query="#NFL",
            limit=20,
            search_type="Latest"
        )
        print(f"   Collected {len(all_tweets)} total tweets across multiple pages")
        
        # Save sample data to file
        sample_data = {
            "collection_time": datetime.now().isoformat(),
            "game_tweets_sample": game_tweets.get("tweets", [])[:2] if "tweets" in game_tweets else [],
            "nfl_account_tweets": nfl_tweets.get("tweets", [])[:2] if "tweets" in nfl_tweets else [],
            "api_costs": costs
        }
        
        with open("nfl_twitter_sample.json", "w") as f:
            json.dump(sample_data, f, indent=2, default=str)
        print("\n✓ Sample data saved to nfl_twitter_sample.json")
        
    except Exception as e:
        print(f"\n❌ Error collecting data: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if twitter.session:
            await twitter.disconnect()
    
    print("\n" + "=" * 60)
    print("Collection complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(collect_nfl_tweets())