#!/usr/bin/env python3
"""
Example: Monitor real-time tweets for a specific game.

This script demonstrates continuous monitoring of tweets
for a specific NFL game, useful for sentiment tracking.
"""

import asyncio
from datetime import datetime
import json
import time

from neural.social import TwitterClient, TwitterConfig


async def monitor_game(team1: str, team2: str, duration_minutes: int = 5):
    """
    Monitor tweets about a specific game.
    
    Args:
        team1: First team name
        team2: Second team name
        duration_minutes: How long to monitor
    """
    
    # Initialize client
    config = TwitterConfig(
        cache_tweets_ttl=10,  # Very short cache for real-time
        rate_limit_requests=20.0  # Limit requests to avoid rate limiting
    )
    twitter = TwitterClient(config)
    
    print("=" * 60)
    print(f"Game Tweet Monitor: {team1} vs {team2}")
    print(f"Duration: {duration_minutes} minutes")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Data collection
    all_tweets = []
    tweet_ids = set()  # Track unique tweets
    
    # Monitoring parameters
    end_time = time.time() + (duration_minutes * 60)
    interval = 30  # Seconds between checks
    iteration = 0
    
    try:
        while time.time() < end_time:
            iteration += 1
            current_time = datetime.now().strftime("%H:%M:%S")
            
            print(f"\n[{current_time}] Check #{iteration}")
            
            # Search for game-related tweets
            query = f"{team1} {team2} OR #{team1} #{team2}"
            
            try:
                # Get latest tweets
                result = await twitter.search_tweets(
                    query=query,
                    limit=20,
                    search_type="Latest"
                )
                
                if "tweets" in result:
                    new_tweets = []
                    
                    for tweet in result["tweets"]:
                        tweet_id = tweet.get("id")
                        
                        # Only process new tweets
                        if tweet_id and tweet_id not in tweet_ids:
                            tweet_ids.add(tweet_id)
                            new_tweets.append(tweet)
                            
                            # Extract key info
                            text = tweet.get("text", "")[:100]
                            user = tweet.get("user", {}).get("screen_name", "unknown")
                            
                            print(f"  NEW: @{user}: {text}...")
                    
                    if new_tweets:
                        all_tweets.extend(new_tweets)
                        print(f"  → Added {len(new_tweets)} new tweets")
                    else:
                        print(f"  → No new tweets")
                    
                    # Show cumulative stats
                    print(f"  Total unique tweets: {len(tweet_ids)}")
                
            except Exception as e:
                print(f"  ⚠ Error in iteration {iteration}: {e}")
            
            # Show API costs periodically
            if iteration % 5 == 0:
                costs = twitter.get_api_costs()
                print(f"\n  💰 API Costs: ${costs['total_cost']:.6f} ({costs['requests_count']} requests)")
            
            # Wait before next check
            if time.time() < end_time:
                print(f"\n  Waiting {interval} seconds...")
                await asyncio.sleep(interval)
        
        # Final summary
        print("\n" + "=" * 60)
        print("MONITORING COMPLETE")
        print("=" * 60)
        print(f"Total tweets collected: {len(all_tweets)}")
        print(f"Unique tweets: {len(tweet_ids)}")
        print(f"Iterations: {iteration}")
        
        # Analyze tweet frequency
        if all_tweets:
            # Count tweets by user
            user_counts = {}
            for tweet in all_tweets:
                user = tweet.get("user", {}).get("screen_name", "unknown")
                user_counts[user] = user_counts.get(user, 0) + 1
            
            # Top tweeters
            top_users = sorted(user_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            print("\nTop Tweeters:")
            for user, count in top_users:
                print(f"  @{user}: {count} tweets")
            
            # Popular words (simple analysis)
            word_counts = {}
            for tweet in all_tweets:
                text = tweet.get("text", "").lower()
                words = text.split()
                for word in words:
                    if len(word) > 4 and not word.startswith("http"):
                        word_counts[word] = word_counts.get(word, 0) + 1
            
            top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            print("\nTop Words:")
            for word, count in top_words[:5]:
                print(f"  {word}: {count} occurrences")
        
        # Save data
        output_file = f"game_monitor_{team1}_{team2}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        monitor_data = {
            "game": f"{team1} vs {team2}",
            "start_time": datetime.now().isoformat(),
            "duration_minutes": duration_minutes,
            "total_tweets": len(all_tweets),
            "unique_tweets": len(tweet_ids),
            "iterations": iteration,
            "tweets_sample": all_tweets[:10],  # Save first 10 as sample
            "api_costs": twitter.get_api_costs()
        }
        
        with open(output_file, "w") as f:
            json.dump(monitor_data, f, indent=2, default=str)
        
        print(f"\n✓ Data saved to {output_file}")
        
        # Final API costs
        costs = twitter.get_api_costs()
        print(f"\nFinal API Costs:")
        print(f"  Total: ${costs['total_cost']:.6f}")
        print(f"  Requests: {costs['requests_count']}")
        print(f"  Avg per request: ${costs['average_cost_per_request']:.6f}")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Monitoring interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if twitter.session:
            await twitter.disconnect()


async def main():
    """Main entry point."""
    # Example: Monitor Packers vs Commanders
    await monitor_game("Packers", "Commanders", duration_minutes=2)


if __name__ == "__main__":
    asyncio.run(main())