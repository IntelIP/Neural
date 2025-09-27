"""
Data Aggregation Pipeline for Sentiment-Based Trading

This module orchestrates multiple data sources (Twitter, ESPN, Kalshi) and
provides synchronized, real-time data streams for sentiment analysis and trading.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, AsyncGenerator, Any, Set, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging

from .base import DataSource
from .twitter_source import TwitterAPISource, create_twitter_source
from .espn_enhanced import ESPNGameCastSource, create_gamecast_source
from .kalshi_api_source import KalshiAPISource
from ..analysis.sentiment import SentimentAnalyzer, GameSentimentTracker, create_sentiment_analyzer


@dataclass
class DataPoint:
    """Unified data point from any source."""
    source: str
    timestamp: datetime
    data: Dict[str, Any]
    game_id: Optional[str] = None
    teams: Optional[List[str]] = None


@dataclass
class AggregatedData:
    """Aggregated data from multiple sources."""
    timestamp: datetime
    game_id: str
    teams: List[str]
    twitter_data: Optional[Dict[str, Any]] = None
    espn_data: Optional[Dict[str, Any]] = None
    kalshi_data: Optional[Dict[str, Any]] = None
    sentiment_metrics: Optional[Dict[str, Any]] = None
    trading_signals: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceConfig:
    """Configuration for a data source."""
    enabled: bool = True
    poll_interval: float = 30.0
    buffer_size: int = 100
    timeout: float = 10.0
    retry_attempts: int = 3
    config: Dict[str, Any] = field(default_factory=dict)


class DataBuffer:
    """Thread-safe data buffer with time-based expiration."""

    def __init__(self, max_size: int = 1000, max_age_minutes: int = 60):
        self.max_size = max_size
        self.max_age = timedelta(minutes=max_age_minutes)
        self.buffer: deque = deque(maxlen=max_size)
        self._lock = asyncio.Lock()

    async def add(self, data_point: DataPoint):
        """Add a data point to the buffer."""
        async with self._lock:
            self.buffer.append(data_point)

    async def get_recent(self, minutes: int = 5) -> List[DataPoint]:
        """Get data points from the last N minutes."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)

        async with self._lock:
            return [
                dp for dp in self.buffer
                if dp.timestamp >= cutoff_time
            ]

    async def get_by_source(self, source: str, minutes: int = 5) -> List[DataPoint]:
        """Get data points from a specific source."""
        recent_data = await self.get_recent(minutes)
        return [dp for dp in recent_data if dp.source == source]

    async def cleanup_old_data(self):
        """Remove expired data points."""
        cutoff_time = datetime.now() - self.max_age

        async with self._lock:
            # Convert to list to avoid modifying deque during iteration
            current_data = list(self.buffer)
            self.buffer.clear()

            for dp in current_data:
                if dp.timestamp >= cutoff_time:
                    self.buffer.append(dp)


class MultiSourceAggregator:
    """
    Aggregates data from multiple sources with intelligent synchronization.

    Handles rate limiting, error recovery, and provides unified data streams
    for sentiment analysis and trading algorithms.
    """

    def __init__(
        self,
        game_id: str,
        teams: List[str],
        twitter_config: Optional[SourceConfig] = None,
        espn_config: Optional[SourceConfig] = None,
        kalshi_config: Optional[SourceConfig] = None,
        sentiment_analyzer: Optional[SentimentAnalyzer] = None
    ):
        self.game_id = game_id
        self.teams = teams

        # Source configurations
        self.twitter_config = twitter_config or SourceConfig(poll_interval=30.0)
        self.espn_config = espn_config or SourceConfig(poll_interval=5.0)
        self.kalshi_config = kalshi_config or SourceConfig(poll_interval=10.0)

        # Data sources
        self.twitter_source: Optional[TwitterAPISource] = None
        self.espn_source: Optional[ESPNGameCastSource] = None
        self.kalshi_source: Optional[KalshiAPISource] = None

        # Data management
        self.data_buffer = DataBuffer(max_size=5000, max_age_minutes=120)
        self.sentiment_tracker = GameSentimentTracker(
            game_id=game_id,
            teams=teams,
            sentiment_analyzer=sentiment_analyzer or create_sentiment_analyzer()
        )

        # State management
        self._running = False
        self._tasks: List[asyncio.Task] = []
        self.logger = logging.getLogger(f"aggregator_{game_id}")

        # Event handlers
        self.data_handlers: List[Callable[[AggregatedData], None]] = []

    async def initialize(self, **source_kwargs):
        """Initialize all data sources."""
        try:
            # Initialize Twitter source
            if self.twitter_config.enabled:
                twitter_api_key = source_kwargs.get('twitter_api_key')
                if twitter_api_key:
                    self.twitter_source = create_twitter_source(
                        api_key=twitter_api_key,
                        teams=self.teams,
                        poll_interval=self.twitter_config.poll_interval
                    )
                    await self.twitter_source.connect()
                    self.logger.info("Twitter source initialized")

            # Initialize ESPN source
            if self.espn_config.enabled:
                self.espn_source = create_gamecast_source(
                    game_id=self.game_id,
                    poll_interval=self.espn_config.poll_interval,
                    enhanced_sentiment=True
                )
                await self.espn_source.connect()
                self.logger.info("ESPN source initialized")

            # Initialize Kalshi source
            if self.kalshi_config.enabled:
                kalshi_config = source_kwargs.get('kalshi_config', {})
                if kalshi_config:
                    self.kalshi_source = KalshiAPISource(kalshi_config)
                    await self.kalshi_source.connect()
                    self.logger.info("Kalshi source initialized")

        except Exception as e:
            self.logger.error(f"Error initializing sources: {e}")
            raise

    async def start(self, **source_kwargs):
        """Start data collection from all sources."""
        if self._running:
            return

        await self.initialize(**source_kwargs)
        self._running = True

        # Start data collection tasks
        if self.twitter_source:
            task = asyncio.create_task(self._collect_twitter_data())
            self._tasks.append(task)

        if self.espn_source:
            task = asyncio.create_task(self._collect_espn_data())
            self._tasks.append(task)

        if self.kalshi_source:
            task = asyncio.create_task(self._collect_kalshi_data())
            self._tasks.append(task)

        # Start aggregation task
        aggregation_task = asyncio.create_task(self._aggregate_data())
        self._tasks.append(aggregation_task)

        # Start cleanup task
        cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._tasks.append(cleanup_task)

        self.logger.info("Data aggregation started")

    async def stop(self):
        """Stop all data collection."""
        self._running = False

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self._tasks, return_exceptions=True)

        # Disconnect sources
        if self.twitter_source:
            await self.twitter_source.disconnect()
        if self.espn_source:
            await self.espn_source.disconnect()
        if self.kalshi_source:
            await self.kalshi_source.disconnect()

        self.logger.info("Data aggregation stopped")

    async def _collect_twitter_data(self):
        """Collect Twitter data continuously."""
        while self._running:
            try:
                async for tweet_batch in self.twitter_source.collect():
                    if not self._running:
                        break

                    # Process tweets if they're in the expected format
                    tweets = tweet_batch if isinstance(tweet_batch, list) else [tweet_batch]

                    data_point = DataPoint(
                        source="twitter",
                        timestamp=datetime.now(),
                        data={"tweets": tweets, "count": len(tweets)},
                        game_id=self.game_id,
                        teams=self.teams
                    )

                    await self.data_buffer.add(data_point)

                    # Update sentiment tracker
                    self.sentiment_tracker.add_twitter_data(tweets)

            except Exception as e:
                self.logger.error(f"Twitter collection error: {e}")
                await asyncio.sleep(60)  # Wait longer on error

    async def _collect_espn_data(self):
        """Collect ESPN data continuously."""
        while self._running:
            try:
                async for espn_data in self.espn_source.collect():
                    if not self._running:
                        break

                    data_point = DataPoint(
                        source="espn",
                        timestamp=datetime.now(),
                        data=espn_data,
                        game_id=self.game_id,
                        teams=self.teams
                    )

                    await self.data_buffer.add(data_point)

                    # Update sentiment tracker
                    self.sentiment_tracker.add_espn_data(espn_data)

            except Exception as e:
                self.logger.error(f"ESPN collection error: {e}")
                await asyncio.sleep(60)

    async def _collect_kalshi_data(self):
        """Collect Kalshi market data continuously."""
        while self._running:
            try:
                async for market_data in self.kalshi_source.collect():
                    if not self._running:
                        break

                    data_point = DataPoint(
                        source="kalshi",
                        timestamp=datetime.now(),
                        data=market_data,
                        game_id=self.game_id,
                        teams=self.teams
                    )

                    await self.data_buffer.add(data_point)

            except Exception as e:
                self.logger.error(f"Kalshi collection error: {e}")
                await asyncio.sleep(60)

    async def _aggregate_data(self):
        """Aggregate data from all sources periodically."""
        while self._running:
            try:
                # Get recent data from all sources
                twitter_data = await self.data_buffer.get_by_source("twitter", minutes=2)
                espn_data = await self.data_buffer.get_by_source("espn", minutes=2)
                kalshi_data = await self.data_buffer.get_by_source("kalshi", minutes=2)

                # Get current sentiment metrics
                sentiment_metrics = self.sentiment_tracker.get_current_sentiment()

                # Create aggregated data point
                aggregated = AggregatedData(
                    timestamp=datetime.now(),
                    game_id=self.game_id,
                    teams=self.teams,
                    twitter_data=self._summarize_twitter_data(twitter_data),
                    espn_data=self._get_latest_espn_data(espn_data),
                    kalshi_data=self._get_latest_kalshi_data(kalshi_data),
                    sentiment_metrics=sentiment_metrics,
                    metadata={
                        'twitter_points': len(twitter_data),
                        'espn_points': len(espn_data),
                        'kalshi_points': len(kalshi_data),
                        'signal_strength': self.sentiment_tracker.get_trading_signal_strength()
                    }
                )

                # Notify handlers
                for handler in self.data_handlers:
                    try:
                        handler(aggregated)
                    except Exception as e:
                        self.logger.error(f"Handler error: {e}")

                await asyncio.sleep(10)  # Aggregate every 10 seconds

            except Exception as e:
                self.logger.error(f"Aggregation error: {e}")
                await asyncio.sleep(30)

    async def _cleanup_loop(self):
        """Periodic cleanup of old data."""
        while self._running:
            try:
                await self.data_buffer.cleanup_old_data()
                await asyncio.sleep(300)  # Cleanup every 5 minutes
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(300)

    def _summarize_twitter_data(self, twitter_points: List[DataPoint]) -> Optional[Dict[str, Any]]:
        """Summarize recent Twitter data."""
        if not twitter_points:
            return None

        all_tweets = []
        for point in twitter_points:
            tweets = point.data.get("tweets", [])
            all_tweets.extend(tweets)

        if not all_tweets:
            return None

        return {
            'tweet_count': len(all_tweets),
            'latest_timestamp': max(point.timestamp for point in twitter_points),
            'total_engagement': sum(
                tweet.get('metrics', {}).get('like_count', 0) +
                tweet.get('metrics', {}).get('retweet_count', 0)
                for tweet in all_tweets
            ),
            'sample_tweets': all_tweets[:5]  # Store sample for analysis
        }

    def _get_latest_espn_data(self, espn_points: List[DataPoint]) -> Optional[Dict[str, Any]]:
        """Get the latest ESPN data."""
        if not espn_points:
            return None

        latest_point = max(espn_points, key=lambda p: p.timestamp)
        return latest_point.data

    def _get_latest_kalshi_data(self, kalshi_points: List[DataPoint]) -> Optional[Dict[str, Any]]:
        """Get the latest Kalshi market data."""
        if not kalshi_points:
            return None

        latest_point = max(kalshi_points, key=lambda p: p.timestamp)
        return latest_point.data

    def add_data_handler(self, handler: Callable[[AggregatedData], None]):
        """Add a handler for aggregated data."""
        self.data_handlers.append(handler)

    async def get_current_state(self) -> Dict[str, Any]:
        """Get current aggregation state."""
        recent_twitter = await self.data_buffer.get_by_source("twitter", minutes=5)
        recent_espn = await self.data_buffer.get_by_source("espn", minutes=5)
        recent_kalshi = await self.data_buffer.get_by_source("kalshi", minutes=5)

        return {
            'running': self._running,
            'game_id': self.game_id,
            'teams': self.teams,
            'data_points': {
                'twitter': len(recent_twitter),
                'espn': len(recent_espn),
                'kalshi': len(recent_kalshi)
            },
            'sentiment_metrics': self.sentiment_tracker.get_current_sentiment(),
            'signal_strength': self.sentiment_tracker.get_trading_signal_strength(),
            'buffer_size': len(self.data_buffer.buffer)
        }

    async def stream_data(self) -> AsyncGenerator[AggregatedData, None]:
        """Stream aggregated data continuously."""
        last_aggregation_time = datetime.now()

        while self._running:
            try:
                # Check if enough time has passed for new aggregation
                if datetime.now() - last_aggregation_time >= timedelta(seconds=15):
                    # Get recent data from all sources
                    twitter_data = await self.data_buffer.get_by_source("twitter", minutes=3)
                    espn_data = await self.data_buffer.get_by_source("espn", minutes=3)
                    kalshi_data = await self.data_buffer.get_by_source("kalshi", minutes=3)

                    # Only yield if we have some data
                    if twitter_data or espn_data or kalshi_data:
                        sentiment_metrics = self.sentiment_tracker.get_current_sentiment()

                        aggregated = AggregatedData(
                            timestamp=datetime.now(),
                            game_id=self.game_id,
                            teams=self.teams,
                            twitter_data=self._summarize_twitter_data(twitter_data),
                            espn_data=self._get_latest_espn_data(espn_data),
                            kalshi_data=self._get_latest_kalshi_data(kalshi_data),
                            sentiment_metrics=sentiment_metrics,
                            metadata={
                                'twitter_points': len(twitter_data),
                                'espn_points': len(espn_data),
                                'kalshi_points': len(kalshi_data),
                                'signal_strength': self.sentiment_tracker.get_trading_signal_strength()
                            }
                        )

                        yield aggregated
                        last_aggregation_time = datetime.now()

                await asyncio.sleep(5)  # Check every 5 seconds

            except Exception as e:
                self.logger.error(f"Stream error: {e}")
                await asyncio.sleep(10)


# Factory function for easy setup
def create_aggregator(
    game_id: str,
    teams: List[str],
    twitter_enabled: bool = True,
    espn_enabled: bool = True,
    kalshi_enabled: bool = True,
    **kwargs
) -> MultiSourceAggregator:
    """
    Create a data aggregator with specified configuration.

    Args:
        game_id: Unique game identifier
        teams: List of team names
        twitter_enabled: Enable Twitter data collection
        espn_enabled: Enable ESPN data collection
        kalshi_enabled: Enable Kalshi data collection
        **kwargs: Additional configuration options

    Returns:
        Configured MultiSourceAggregator
    """
    twitter_config = SourceConfig(
        enabled=twitter_enabled,
        poll_interval=kwargs.get('twitter_interval', 30.0)
    )

    espn_config = SourceConfig(
        enabled=espn_enabled,
        poll_interval=kwargs.get('espn_interval', 5.0)
    )

    kalshi_config = SourceConfig(
        enabled=kalshi_enabled,
        poll_interval=kwargs.get('kalshi_interval', 10.0)
    )

    return MultiSourceAggregator(
        game_id=game_id,
        teams=teams,
        twitter_config=twitter_config,
        espn_config=espn_config,
        kalshi_config=kalshi_config,
        sentiment_analyzer=kwargs.get('sentiment_analyzer')
    )


# Example usage
if __name__ == "__main__":
    async def example():
        # Create aggregator for Ravens vs Lions game
        aggregator = create_aggregator(
            game_id="401547439",
            teams=["Baltimore Ravens", "Detroit Lions"],
            twitter_enabled=True,
            espn_enabled=True,
            kalshi_enabled=True
        )

        # Add a data handler
        def handle_data(data: AggregatedData):
            print(f"Aggregated data at {data.timestamp}")
            print(f"Sentiment: {data.sentiment_metrics}")

        aggregator.add_data_handler(handle_data)

        # Start aggregation (would need API keys in real usage)
        try:
            await aggregator.start(
                twitter_api_key="your_twitter_key",
                kalshi_config={"api_key": "your_kalshi_key"}
            )

            # Let it run for a bit
            await asyncio.sleep(60)

        finally:
            await aggregator.stop()

    # Note: This example won't run without proper API keys
    # asyncio.run(example())