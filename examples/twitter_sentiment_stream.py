"""
Example: Twitter Sentiment Streaming
Real-time sentiment analysis for game monitoring
"""

import asyncio
import logging
import os

from neural_sdk.data_pipeline.data_sources.twitter import (
    FilterManager,
    SentimentAnalyzer,
    TwitterStreamAdapter,
)
from neural_sdk.data_pipeline.utils import setup_logging


async def basic_sentiment_stream():
    """Basic example of streaming Twitter sentiment"""
    setup_logging(level="INFO")
    logger = logging.getLogger(__name__)

    # Check for API key
    if not (os.getenv("TWITTERAPI_KEY") or os.getenv("TWITTER_BEARER_TOKEN")):
        logger.error(
            "Twitter API key not set. For this integration, add TWITTERAPI_KEY (TwitterAPI.io) "
            "or set TWITTER_BEARER_TOKEN if you intend to reuse it as the API key."
        )
        return

    # Create stream adapter
    stream = TwitterStreamAdapter()

    # Define callbacks
    async def handle_sentiment_event(event):
        logger.info("=" * 60)
        logger.info("SENTIMENT UPDATE")
        logger.info(
            f"  Period: {event.start_time.strftime('%H:%M:%S')} - {event.end_time.strftime('%H:%M:%S')}"
        )
        logger.info(
            f"  Tweets: {event.tweet_count} from {event.unique_authors} authors"
        )
        logger.info(f"  Sentiment: {event.avg_sentiment_score:+.2f}")
        logger.info(f"  Velocity: {event.tweets_per_minute:.1f} tweets/min")
        logger.info(f"  Impact: {event.market_impact.value}")

        if event.trending_keywords:
            logger.info(f"  Trending: {', '.join(event.trending_keywords[:5])}")

        # Show sentiment distribution
        dist = event.sentiment_distribution
        total = sum(dist.values())
        if total > 0:
            logger.info(
                f"  Distribution: "
                f"Positive {dist.get('positive', 0)/total*100:.0f}% | "
                f"Neutral {dist.get('neutral', 0)/total*100:.0f}% | "
                f"Negative {dist.get('negative', 0)/total*100:.0f}%"
            )

    async def handle_sentiment_shift(shift):
        logger.warning("🔄 SENTIMENT SHIFT DETECTED!")
        logger.warning(f"  Type: {shift['type']}")
        logger.warning(f"  Magnitude: {shift['magnitude']:.2f}")
        logger.warning(
            f"  From: {shift['from_sentiment']:+.2f} To: {shift['to_sentiment']:+.2f}"
        )
        logger.warning(f"  Tweet surge: {shift.get('tweet_surge', 1):.1f}x normal")

    async def handle_high_impact(tweet, sentiment):
        logger.info("💥 HIGH IMPACT TWEET")
        logger.info(
            f"  Author: @{tweet.author.username} ({tweet.author.followers_count:,} followers)"
        )
        logger.info(f"  Text: {tweet.text[:200]}...")
        logger.info(
            f"  Sentiment: {sentiment.score:+.2f} ({sentiment.sentiment_label})"
        )
        if sentiment.market_keywords:
            logger.info(f"  Market Keywords: {sentiment.market_keywords}")
        logger.info(f"  Engagement: {tweet.metrics.engagement_score:.2f}")

    # Set callbacks
    stream.on_sentiment_event = handle_sentiment_event
    stream.on_sentiment_shift = handle_sentiment_shift
    stream.on_high_impact_tweet = handle_high_impact

    try:
        # Start streaming
        await stream.start()

        # Monitor a game (example teams)
        await stream.monitor_game(
            game_id="GAME123",
            home_team="Chiefs",
            away_team="Bills",
            sport="nfl",
            players=["Patrick Mahomes", "Josh Allen", "Travis Kelce", "Stefon Diggs"],
        )

        logger.info("Streaming Twitter sentiment for Chiefs vs Bills...")
        logger.info("Monitoring key players and team accounts...")

        # Run for 5 minutes
        await asyncio.sleep(300)

        # Get final stats
        stats = stream.get_stats()
        logger.info("\n" + "=" * 60)
        logger.info("STREAM STATISTICS")
        logger.info(f"  Total tweets: {stats['tweets_received']}")
        logger.info(f"  Rate: {stats.get('tweets_per_minute', 0):.1f} tweets/min")
        logger.info(f"  Current sentiment: {stats.get('current_sentiment', 0):+.2f}")

    finally:
        await stream.stop()


async def game_sentiment_tracking():
    """Track sentiment throughout a game"""
    setup_logging(level="INFO")
    logger = logging.getLogger(__name__)

    if not (os.getenv("TWITTERAPI_KEY") or os.getenv("TWITTER_BEARER_TOKEN")):
        logger.error("Twitter API key not set (TWITTERAPI_KEY or TWITTER_BEARER_TOKEN)")
        return

    stream = TwitterStreamAdapter()

    # Track key moments
    key_moments = []

    async def handle_sentiment_event(event):
        # Track significant moments
        if event.market_impact.value in ["high", "critical"]:
            key_moments.append(
                {
                    "time": event.end_time,
                    "sentiment": event.avg_sentiment_score,
                    "impact": event.market_impact.value,
                    "tweets": event.tweet_count,
                }
            )

            logger.info(
                f"📍 Key Moment: Sentiment {event.avg_sentiment_score:+.2f}, "
                f"Impact: {event.market_impact.value}"
            )

    stream.on_sentiment_event = handle_sentiment_event

    try:
        await stream.start()

        # Set up game monitoring
        game_id = "NFL_2024_WK1_KC_BUF"
        await stream.monitor_game(
            game_id=game_id, home_team="Chiefs", away_team="Bills", sport="nfl"
        )

        logger.info("Tracking game sentiment...")

        # Simulate game quarters (30 seconds each for demo)
        quarters = ["Q1", "Q2", "Halftime", "Q3", "Q4"]

        for quarter in quarters:
            logger.info(f"\n{'='*40}")
            logger.info(f"📢 {quarter}")
            await asyncio.sleep(30)

            # Get current sentiment
            current = stream.get_current_sentiment()
            trend = stream.get_sentiment_trend(periods=3)

            if current is not None:
                logger.info(f"Current sentiment: {current:+.2f}")
                if len(trend) > 1:
                    direction = "📈" if trend[-1] > trend[0] else "📉"
                    logger.info(f"Trend: {direction} {trend}")

        # Get game summary
        summary = stream.get_game_summary(game_id)
        if summary:
            logger.info("\n" + "=" * 60)
            logger.info("GAME SENTIMENT SUMMARY")
            logger.info(f"  Total tweets: {summary.total_tweets}")
            logger.info(f"  Unique authors: {summary.unique_authors}")

            # Show momentum shifts
            shifts = summary.calculate_momentum_shifts()
            if shifts:
                logger.info("\nMomentum Shifts:")
                for shift in shifts:
                    logger.info(
                        f"  {shift['time'].strftime('%H:%M:%S')}: "
                        f"{shift['from_sentiment']:+.2f} → {shift['to_sentiment']:+.2f}"
                    )

            # Show key moments
            if key_moments:
                logger.info("\nKey Moments:")
                for moment in key_moments:
                    logger.info(
                        f"  {moment['time'].strftime('%H:%M:%S')}: "
                        f"Sentiment {moment['sentiment']:+.2f}, "
                        f"Impact: {moment['impact']}"
                    )

    finally:
        await stream.stop()


async def player_injury_monitoring():
    """Monitor for player injury news"""
    setup_logging(level="INFO")
    logger = logging.getLogger(__name__)

    if not (os.getenv("TWITTERAPI_KEY") or os.getenv("TWITTER_BEARER_TOKEN")):
        logger.error("Twitter API key not set (TWITTERAPI_KEY or TWITTER_BEARER_TOKEN)")
        return

    # Set up filter manager
    filter_manager = FilterManager()

    # Create player-specific filter
    key_players = [
        "Patrick Mahomes",
        "Josh Allen",
        "Justin Jefferson",
        "Tyreek Hill",
        "Nick Chubb",
    ]

    logger.info("Setting up injury monitoring for key players...")

    for player in key_players:
        rule = await filter_manager.create_player_filter(
            player_name=player,
            keywords=["injury", "injured", "questionable", "doubtful", "out", "return"],
        )
        logger.info(f"  Monitoring: {player} (Rule: {rule.tag})")

    # Create sentiment analyzer
    analyzer = SentimentAnalyzer()

    # Set up WebSocket client
    from neural_sdk.data_pipeline.data_sources.twitter.client import AsyncTwitterWebSocketClient

    client = AsyncTwitterWebSocketClient()

    injury_alerts = []

    def handle_tweet(tweet):
        # Check for injury keywords
        injury_keywords = ["injury", "injured", "out", "questionable", "doubtful"]
        text_lower = tweet.text.lower()

        if any(kw in text_lower for kw in injury_keywords):
            # Analyze sentiment
            sentiment = analyzer.analyze_tweet(tweet)

            # High-impact injury news
            if tweet.author.verified or tweet.author.followers_count > 10000:
                alert = {
                    "player": next(
                        (p for p in key_players if p.lower() in text_lower), "Unknown"
                    ),
                    "author": tweet.author.username,
                    "text": tweet.text[:200],
                    "sentiment": sentiment.score,
                    "impact": "HIGH" if tweet.author.verified else "MEDIUM",
                    "time": tweet.created_at,
                }

                injury_alerts.append(alert)

                logger.warning("🚨 INJURY ALERT!")
                logger.warning(f"  Player: {alert['player']}")
                logger.warning(f"  Source: @{alert['author']}")
                logger.warning(f"  Message: {alert['text']}...")
                logger.warning(f"  Impact: {alert['impact']}")

    client.on_tweet(handle_tweet)

    try:
        await client.connect()

        logger.info("Monitoring for injury news...")

        # Run for 3 minutes
        await asyncio.sleep(180)

        # Summary
        if injury_alerts:
            logger.info("\n" + "=" * 60)
            logger.info("INJURY ALERTS SUMMARY")
            for alert in injury_alerts:
                logger.info(
                    f"\n{alert['time'].strftime('%H:%M:%S')} - {alert['player']}"
                )
                logger.info(f"  Source: @{alert['author']} ({alert['impact']})")
                logger.info(f"  Sentiment: {alert['sentiment']:+.2f}")
        else:
            logger.info("\nNo injury alerts detected")

    finally:
        await client.disconnect()

        # Clean up old filter rules
        await filter_manager.cleanup_old_rules(hours=1)


async def main():
    """Run Twitter sentiment examples"""
    print("=" * 60)
    print("Twitter Sentiment Streaming Examples")
    print("=" * 60)

    print("\nSelect an example:")
    print("1. Basic sentiment streaming")
    print("2. Game sentiment tracking")
    print("3. Player injury monitoring")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == "1":
        await basic_sentiment_stream()
    elif choice == "2":
        await game_sentiment_tracking()
    elif choice == "3":
        await player_injury_monitoring()
    else:
        print("Running basic sentiment streaming...")
        await basic_sentiment_stream()


if __name__ == "__main__":
    # Note: Set TWITTERAPI_KEY environment variable before running
    # export TWITTERAPI_KEY="your_api_key_here"
    asyncio.run(main())
