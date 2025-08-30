"""
Data Aggregator
Bridge between StreamManager and legacy code
DEPRECATED: Use kalshi_web_infra.stream_manager.StreamManager directly
"""

import asyncio
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from enum import Enum

# Import the new StreamManager
from data_pipeline.orchestration.unified_stream_manager import (
    StreamManager,
    MarketContext,
    UnifiedEvent,
    EventType as StreamEventType,
    DataSource
)

logger = logging.getLogger(__name__)
logger.warning("data_aggregator.py is deprecated. Use StreamManager directly.")


class DataSource(Enum):
    """Data source types."""
    KALSHI = "kalshi"
    ESPN = "espn"
    TWITTER = "twitter"


@dataclass
class MarketContext:
    """Aggregated context for a market."""
    market_ticker: str
    last_update: datetime = field(default_factory=datetime.now)
    
    # Kalshi data
    yes_bid: Optional[float] = None
    yes_ask: Optional[float] = None
    volume_24h: Optional[float] = None
    orderbook_pressure: Optional[float] = None  # Buy vs sell pressure
    
    # ESPN data
    game_id: Optional[str] = None
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    score: Optional[Dict[str, int]] = None
    win_probability: Optional[Dict[str, float]] = None
    recent_events: List[Dict[str, Any]] = field(default_factory=list)
    
    # Twitter sentiment
    sentiment_score: float = 0.0
    sentiment_velocity: float = 0.0  # Rate of change
    tweet_volume: int = 0
    high_impact_tweets: List[Dict[str, Any]] = field(default_factory=list)
    
    # Computed metrics
    sentiment_price_divergence: Optional[float] = None
    opportunity_score: Optional[float] = None
    
    def update_kalshi(self, data: Dict[str, Any]):
        """Update Kalshi market data."""
        self.yes_bid = data.get("yes_bid", self.yes_bid)
        self.yes_ask = data.get("yes_ask", self.yes_ask)
        self.volume_24h = data.get("volume_24h", self.volume_24h)
        self.last_update = datetime.now()
        
        # Calculate orderbook pressure
        if "yes_levels" in data and "no_levels" in data:
            yes_volume = sum(l.get("quantity", 0) for l in data["yes_levels"][:5])
            no_volume = sum(l.get("quantity", 0) for l in data["no_levels"][:5])
            if yes_volume + no_volume > 0:
                self.orderbook_pressure = (yes_volume - no_volume) / (yes_volume + no_volume)
    
    def update_espn(self, data: Dict[str, Any]):
        """Update ESPN game data."""
        event_type = data.get("type")
        
        if event_type == ESPNEventType.SCORE_UPDATE.value:
            self.score = data.get("score")
        elif event_type == ESPNEventType.WIN_PROBABILITY.value:
            self.win_probability = data.get("win_probability")
        elif event_type in [ESPNEventType.INJURY.value, ESPNEventType.BIG_PLAY.value]:
            self.recent_events.append(data)
            # Keep only last 10 events
            self.recent_events = self.recent_events[-10:]
        
        self.last_update = datetime.now()
    
    def update_twitter(self, tweet: Dict[str, Any]):
        """Update Twitter sentiment data."""
        self.tweet_volume += 1
        
        # Update sentiment score (weighted average)
        new_sentiment = tweet.get("sentiment_score", 0)
        self.sentiment_score = (self.sentiment_score * 0.9 + new_sentiment * 0.1)
        
        # Track high-impact tweets
        if tweet.get("impact") == "HIGH":
            self.high_impact_tweets.append(tweet)
            self.high_impact_tweets = self.high_impact_tweets[-5:]  # Keep last 5
        
        # Calculate sentiment velocity
        # (This would need historical data in production)
        self.sentiment_velocity = new_sentiment - self.sentiment_score
        
        self.last_update = datetime.now()
    
    def calculate_opportunity(self):
        """Calculate trading opportunity score."""
        if self.yes_bid is None or self.yes_ask is None:
            return
        
        # Current market price (midpoint)
        market_price = (self.yes_bid + self.yes_ask) / 2
        
        # Expected price based on sentiment
        if self.sentiment_score != 0:
            sentiment_implied_price = 0.5 + (self.sentiment_score * 0.3)
            sentiment_implied_price = max(0.01, min(0.99, sentiment_implied_price))
            
            # Divergence between sentiment and market
            self.sentiment_price_divergence = sentiment_implied_price - market_price
        
        # Win probability divergence (if ESPN data available)
        win_prob_divergence = 0
        if self.win_probability and self.home_team:
            espn_implied_price = self.win_probability.get("home", 50) / 100
            win_prob_divergence = espn_implied_price - market_price
        
        # Opportunity score factors:
        # 1. Sentiment-price divergence
        # 2. Win probability divergence
        # 3. Orderbook pressure
        # 4. Recent high-impact events
        
        factors = []
        
        if self.sentiment_price_divergence:
            factors.append(abs(self.sentiment_price_divergence) * 2)
        
        if win_prob_divergence != 0:
            factors.append(abs(win_prob_divergence) * 3)
        
        if self.orderbook_pressure is not None:
            factors.append(abs(self.orderbook_pressure))
        
        if len(self.recent_events) > 0:
            # Recent events increase opportunity
            factors.append(0.2 * len(self.recent_events))
        
        if len(self.high_impact_tweets) > 0:
            factors.append(0.3 * len(self.high_impact_tweets))
        
        # Calculate final score (0-1 scale)
        if factors:
            self.opportunity_score = min(1.0, sum(factors) / len(factors))
        else:
            self.opportunity_score = 0.0


class DataAggregator:
    """
    DEPRECATED: Wrapper around StreamManager for backward compatibility.
    Use StreamManager directly for new code.
    """
    
    def __init__(
        self,
        kalshi_api_key: Optional[str] = None,
        twitter_api_key: Optional[str] = None,
        redis_url: str = "redis://localhost:6379"
    ):
        """
        Initialize data aggregator (deprecated wrapper).
        
        Args:
            kalshi_api_key: Kalshi API key (unused - uses env vars)
            twitter_api_key: TwitterAPI.io API key (unused - uses env vars)
            redis_url: Redis connection URL (unused)
        """
        logger.warning("DataAggregator is deprecated. Migrate to StreamManager.")
        
        # Use StreamManager internally
        self.stream_manager = StreamManager()
        
        # Legacy compatibility
        self.markets = self.stream_manager.markets
        
        # Callbacks mapping
        self.callbacks: Dict[str, List[Callable]] = {
            "market_update": [],
            "opportunity_alert": [],
            "sentiment_shift": []
        }
        
        # State
        self.is_running = False
    
    async def initialize(self):
        """Initialize all data sources."""
        logger.info("Initializing data aggregator...")
        
        # Initialize Kalshi
        if self.kalshi_api_key:
            self.kalshi_client = KalshiWebSocket(
                api_key=self.kalshi_api_key,
                on_message=self._handle_kalshi_message,
                on_error=self._handle_error
            )
        
        # Initialize ESPN
        self.espn_adapter = ESPNWebSocketAdapter(
            on_message=self._handle_espn_message,
            on_error=self._handle_error,
            poll_interval=10
        )
        
        # Initialize Twitter
        if self.twitter_api_key:
            self.twitter_client = TwitterWebSocket(
                api_key=self.twitter_api_key
            )
        
        # Initialize Redis
        self.redis_pubsub = RedisPubSub(redis_url=self.redis_url)
        
        logger.info("Data aggregator initialized")
    
    async def start(self):
        """Start all data sources."""
        if self.is_running:
            logger.warning("Aggregator already running")
            return
        
        self.is_running = True
        logger.info("Starting data aggregator...")
        
        # Connect to data sources
        if self.kalshi_client:
            await self.kalshi_client.connect()
        
        await self.espn_adapter.connect()
        
        if self.twitter_client:
            await self.twitter_client.connect()
        
        await self.redis_pubsub.connect()
        
        # Start aggregation loop
        self.tasks.append(
            asyncio.create_task(self._aggregation_loop())
        )
        
        # Start Twitter streaming if available
        if self.twitter_client:
            self.tasks.append(
                asyncio.create_task(self._stream_twitter())
            )
        
        logger.info("Data aggregator started")
    
    async def track_market(
        self,
        market_ticker: str,
        game_id: Optional[str] = None,
        teams: Optional[List[str]] = None
    ):
        """
        Start tracking a specific market.
        
        Args:
            market_ticker: Kalshi market ticker
            game_id: ESPN game ID
            teams: Teams to track on Twitter
        """
        logger.info(f"Tracking market: {market_ticker}")
        
        # Create market context
        if market_ticker not in self.markets:
            self.markets[market_ticker] = MarketContext(market_ticker=market_ticker)
        
        context = self.markets[market_ticker]
        context.game_id = game_id
        
        # Subscribe to Kalshi channels
        if self.kalshi_client:
            await self.kalshi_client.subscribe(
                channel=KalshiChannel.TICKER,
                market_tickers=[market_ticker]
            )
            
            await self.kalshi_client.subscribe(
                channel=KalshiChannel.ORDERBOOK_DELTA,
                market_tickers=[market_ticker]
            )
        
        # Subscribe to ESPN game
        if game_id and self.espn_adapter:
            await self.espn_adapter.subscribe_game(game_id)
        
        # Set up Twitter filters
        if teams and self.twitter_client:
            await self.twitter_client.setup_filters(teams=teams)
    
    async def _handle_kalshi_message(self, msg: Dict[str, Any]):
        """Handle Kalshi WebSocket messages."""
        market_ticker = msg.get("market_ticker")
        if not market_ticker:
            return
        
        # Get or create market context
        if market_ticker not in self.markets:
            self.markets[market_ticker] = MarketContext(market_ticker=market_ticker)
        
        context = self.markets[market_ticker]
        context.update_kalshi(msg)
        
        # Calculate opportunity
        context.calculate_opportunity()
        
        # Notify callbacks
        await self._notify_market_update(context)
        
        # Check for opportunities
        if context.opportunity_score and context.opportunity_score > 0.7:
            await self._notify_opportunity(context)
        
        # Publish to Redis
        await self._publish_to_redis(context)
    
    async def _handle_espn_message(self, msg: Dict[str, Any]):
        """Handle ESPN adapter messages."""
        game_id = msg.get("game_id")
        
        # Find market for this game
        for context in self.markets.values():
            if context.game_id == game_id:
                context.update_espn(msg)
                context.calculate_opportunity()
                
                await self._notify_market_update(context)
                
                # Check for high-impact events
                if msg.get("type") in [ESPNEventType.INJURY.value, ESPNEventType.BIG_PLAY.value]:
                    await self._notify_opportunity(context)
                
                break
    
    async def _stream_twitter(self):
        """Stream and process Twitter data."""
        if not self.twitter_client:
            return
        
        try:
            async for tweet in self.twitter_client.stream_tweets():
                # Match tweet to markets based on content
                for context in self.markets.values():
                    # Simple matching - in production would be more sophisticated
                    if context.home_team and context.away_team:
                        text = tweet.get("text", "").lower()
                        if (context.home_team.lower() in text or 
                            context.away_team.lower() in text):
                            
                            context.update_twitter(tweet)
                            context.calculate_opportunity()
                            
                            # Check for sentiment shifts
                            if abs(context.sentiment_velocity) > 0.2:
                                await self._notify_sentiment_shift(context)
                            
                            await self._notify_market_update(context)
                            
        except Exception as e:
            logger.error(f"Twitter streaming error: {e}")
    
    async def _aggregation_loop(self):
        """Main aggregation loop."""
        while self.is_running:
            try:
                # Periodic opportunity calculation
                for context in self.markets.values():
                    context.calculate_opportunity()
                    
                    # Check for stale data
                    if (datetime.now() - context.last_update).seconds > 60:
                        logger.warning(f"Stale data for {context.market_ticker}")
                
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Aggregation loop error: {e}")
                await asyncio.sleep(10)
    
    async def _publish_to_redis(self, context: MarketContext):
        """Publish aggregated data to Redis."""
        if not self.redis_pubsub:
            return
        
        # Publish market data
        await self.redis_pubsub.publish(
            channel=Channel.MARKET_DATA,
            message_type=MessageType.MARKET_OPPORTUNITY,
            data={
                "market_ticker": context.market_ticker,
                "yes_bid": context.yes_bid,
                "yes_ask": context.yes_ask,
                "opportunity_score": context.opportunity_score,
                "sentiment_divergence": context.sentiment_price_divergence,
                "orderbook_pressure": context.orderbook_pressure
            },
            sender="data_aggregator"
        )
    
    async def _notify_market_update(self, context: MarketContext):
        """Notify callbacks of market update."""
        for callback in self.callbacks["market_update"]:
            try:
                await callback(context)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    async def _notify_opportunity(self, context: MarketContext):
        """Notify callbacks of trading opportunity."""
        for callback in self.callbacks["opportunity_alert"]:
            try:
                await callback(context)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    async def _notify_sentiment_shift(self, context: MarketContext):
        """Notify callbacks of sentiment shift."""
        for callback in self.callbacks["sentiment_shift"]:
            try:
                await callback(context)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    async def _handle_error(self, error: Any):
        """Handle errors from data sources."""
        logger.error(f"Data source error: {error}")
    
    def register_callback(self, event_type: str, callback: Callable):
        """Register a callback for specific events."""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
    
    async def stop(self):
        """Stop all data sources."""
        self.is_running = False
        
        # Cancel tasks
        for task in self.tasks:
            task.cancel()
        
        # Close connections
        if self.kalshi_client:
            await self.kalshi_client.close()
        
        if self.espn_adapter:
            await self.espn_adapter.close()
        
        if self.twitter_client:
            await self.twitter_client.close()
        
        if self.redis_pubsub:
            await self.redis_pubsub.close()
        
        logger.info("Data aggregator stopped")
    
    def get_market_context(self, market_ticker: str) -> Optional[MarketContext]:
        """Get context for a specific market."""
        return self.markets.get(market_ticker)
    
    def get_top_opportunities(self, limit: int = 5) -> List[MarketContext]:
        """Get markets with highest opportunity scores."""
        sorted_markets = sorted(
            self.markets.values(),
            key=lambda m: m.opportunity_score or 0,
            reverse=True
        )
        return sorted_markets[:limit]


# Example usage
async def example_usage():
    """Example of using the data aggregator."""
    import os
    
    # Callback for opportunities
    async def on_opportunity(context: MarketContext):
        print(f"\n🎯 OPPORTUNITY ALERT: {context.market_ticker}")
        print(f"  Opportunity Score: {context.opportunity_score:.2f}")
        print(f"  Yes Bid/Ask: ${context.yes_bid:.2f}/${context.yes_ask:.2f}")
        print(f"  Sentiment: {context.sentiment_score:.2f}")
        if context.sentiment_price_divergence:
            print(f"  Divergence: {context.sentiment_price_divergence:.2f}")
    
    # Initialize aggregator
    aggregator = DataAggregator(
        kalshi_api_key=os.getenv("KALSHI_API_KEY"),
        twitter_api_key=os.getenv("TWITTERAPI_KEY")
    )
    
    # Register callbacks
    aggregator.register_callback("opportunity_alert", on_opportunity)
    
    # Initialize and start
    await aggregator.initialize()
    await aggregator.start()
    
    # Track a market
    await aggregator.track_market(
        market_ticker="SUPERBOWL-2025",
        game_id="401547435",
        teams=["Chiefs", "Bills"]
    )
    
    # Run for a while
    await asyncio.sleep(60)
    
    # Get top opportunities
    top = aggregator.get_top_opportunities()
    print("\nTop Opportunities:")
    for context in top:
        print(f"  {context.market_ticker}: {context.opportunity_score:.2f}")
    
    # Stop
    await aggregator.stop()


if __name__ == "__main__":
    asyncio.run(example_usage())