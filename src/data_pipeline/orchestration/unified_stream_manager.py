"""
Stream Manager Service
Unified streaming manager for all data sources - Agentuity compliant
Single source of truth for WebSocket connections and data correlation
"""

import os
import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import aiohttp

# Import our existing infrastructure
from src.data_pipeline.streaming.websocket import KalshiWebSocket as KalshiWebSocketClient
from src.data_pipeline.data_sources.espn.stream import ESPNStreamAdapter
from src.data_pipeline.data_sources.twitter.stream import TwitterStreamAdapter
from src.data_pipeline.data_sources.twitter.sentiment import MarketImpact

# Import new infrastructure components
from src.data_pipeline.event_buffer import EventBuffer, BufferedEvent, Priority
from src.data_pipeline.state_manager import get_state_manager, ComputationCache
from src.data_pipeline.window_aggregator import WindowAggregator

# Import resilience components
from src.data_pipeline.reliability.resilience_coordinator import (
    get_resilience_coordinator, ServicePriority, DegradationLevel
)
from src.data_pipeline.reliability.health_monitor import get_health_monitor, websocket_health_check

# Import rate limiting and backpressure components
from src.data_pipeline.reliability.rate_limiter import HierarchicalRateLimiter
from src.data_pipeline.reliability.backpressure_manager import BackpressureController, PressureLevel

# Import Redis publisher for agent communication
from src.data_pipeline.orchestration.redis_event_publisher import RedisPublisher, RedisStreamBridge

logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Data source types"""
    KALSHI = "kalshi"
    ESPN = "espn"
    TWITTER = "twitter"


class EventType(Enum):
    """Unified event types"""
    # Market events
    PRICE_UPDATE = "price_update"
    ORDERBOOK_UPDATE = "orderbook_update"
    TRADE_EXECUTED = "trade_executed"
    
    # Game events
    SCORE_UPDATE = "score_update"
    WIN_PROBABILITY = "win_probability"
    INJURY_ALERT = "injury_alert"
    BIG_PLAY = "big_play"
    GAME_STATUS = "game_status"
    
    # Sentiment events
    SENTIMENT_UPDATE = "sentiment_update"
    SENTIMENT_SHIFT = "sentiment_shift"
    HIGH_IMPACT_TWEET = "high_impact_tweet"
    
    # Correlation events
    MARKET_OPPORTUNITY = "market_opportunity"
    DIVERGENCE_DETECTED = "divergence_detected"
    RISK_ALERT = "risk_alert"


@dataclass
class UnifiedEvent:
    """Unified event structure for all data sources"""
    type: EventType
    source: DataSource
    market_ticker: Optional[str] = None
    game_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    impact_score: float = 0.0  # 0-1 scale
    requires_action: bool = False


@dataclass
class MarketContext:
    """Real-time market context with all correlated data"""
    market_ticker: str
    last_update: datetime = field(default_factory=datetime.now)
    
    # Kalshi data
    yes_price: Optional[float] = None
    no_price: Optional[float] = None
    yes_bid: Optional[float] = None
    yes_ask: Optional[float] = None
    volume_24h: Optional[float] = None
    orderbook_depth: Optional[Dict[str, Any]] = None
    
    # ESPN data
    game_id: Optional[str] = None
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    home_score: int = 0
    away_score: int = 0
    quarter: int = 0
    time_remaining: Optional[str] = None
    win_probability: Optional[Dict[str, float]] = None
    recent_plays: List[str] = field(default_factory=list)
    
    # Twitter sentiment
    sentiment_score: float = 0.0  # -1 to 1
    sentiment_velocity: float = 0.0  # Rate of change
    tweet_volume: int = 0
    high_impact_tweets: List[Dict[str, Any]] = field(default_factory=list)
    
    # Computed metrics
    price_sentiment_divergence: float = 0.0
    opportunity_score: float = 0.0
    risk_level: str = "low"  # low, medium, high, critical


class StreamManager:
    """
    Unified Stream Manager - Single source of truth for all streaming data
    
    Manages all WebSocket connections, correlates data, and publishes to agents
    via Agentuity messaging system.
    """
    
    def __init__(self, agent_context=None):
        """
        Initialize Stream Manager
        
        Args:
            agent_context: Agentuity AgentContext for messaging
        """
        self.agent_context = agent_context
        
        # Data source clients
        self.kalshi_client: Optional[KalshiWebSocketClient] = None
        self.espn_adapter: Optional[ESPNStreamAdapter] = None
        self.twitter_adapter: Optional[TwitterStreamAdapter] = None
        
        # Market contexts
        self.markets: Dict[str, MarketContext] = {}
        
        # Event subscribers (for non-Agentuity use)
        self.event_handlers: Dict[EventType, List[Callable]] = {}
        
        # State
        self.is_running = False
        self.tasks: List[asyncio.Task] = []
        
        # Statistics
        self.events_processed = 0
        self.last_event_time = None
        
        # New infrastructure components
        self.event_buffer = EventBuffer(size=10000)
        self.state_manager = get_state_manager(hot_cache_size=100)
        self.window_aggregator = WindowAggregator()
        self.computation_cache = ComputationCache(self.state_manager)
        
        # Resilience components
        self.resilience_coordinator = get_resilience_coordinator(agent_context)
        self.health_monitor = get_health_monitor()
        
        # Register emergency stop handler
        self.resilience_coordinator.emergency_stop_handler = self._handle_emergency_stop
        
        # Rate limiting and backpressure
        self.throttle_manager = HierarchicalRateLimiter(
            global_limit=1000,  # 1000 events/second globally
            window=1.0
        )
        
        self.backpressure_controller = BackpressureController()
        self.backpressure_controller.on_pressure_change = self._handle_pressure_change
        self.backpressure_controller.on_throttle_change = self._handle_throttle_change
        
        # Redis publisher for agent communication
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis_publisher = RedisPublisher(redis_url=redis_url)
        self.redis_bridge = RedisStreamBridge(self, self.redis_publisher)
    
    async def initialize(self):
        """Initialize all streaming clients"""
        logger.info("Initializing Stream Manager...")
        
        # Initialize Kalshi WebSocket
        kalshi_key_id = os.getenv("KALSHI_API_KEY_ID")
        kalshi_private_key = os.getenv("KALSHI_PRIVATE_KEY")
        
        if kalshi_key_id and kalshi_private_key:
            from src.data_pipeline.data_sources.kalshi.auth import KalshiAuth
            from src.data_pipeline.config.settings import KalshiConfig
            
            # Get environment settings
            environment = os.getenv("KALSHI_ENVIRONMENT", "demo")
            api_base_url = f"https://trading-api.kalshi.com/trade-api/v2" if environment == "prod" else "https://demo-api.kalshi.co/trade-api/v2"
            ws_url = f"wss://trading-api.kalshi.com/trade-api/ws/v2" if environment == "prod" else "wss://demo-api.kalshi.co/trade-api/ws/v2"
            
            config = KalshiConfig(
                api_key_id=kalshi_key_id,
                private_key=kalshi_private_key,
                environment=environment,
                api_base_url=api_base_url,
                ws_url=ws_url
            )
            auth = KalshiAuth(config)
            
            # Import custom message handler
            from src.data_pipeline.message_handler import StreamManagerMessageHandler
            
            # Create message handler with callbacks
            message_handler = StreamManagerMessageHandler(
                on_message=self._handle_kalshi_message,
                on_error=self._handle_error
            )
            
            self.kalshi_client = KalshiWebSocketClient(
                config=config,
                auth=auth,
                message_handler=message_handler,
                enable_circuit_breaker=True,
                auto_reconnect=True
            )
            
            # Register Kalshi service with resilience coordinator
            self.resilience_coordinator.register_service(
                name="kalshi_websocket",
                priority=ServicePriority.CRITICAL,
                failure_threshold=3,
                timeout=30.0,
                health_check=lambda: websocket_health_check(self.kalshi_client)
            )
            
            logger.info("Kalshi WebSocket client initialized with circuit breaker")
        else:
            logger.warning("Kalshi credentials not found")
        
        # Initialize ESPN adapter
        self.espn_adapter = ESPNStreamAdapter(
            on_event=self._handle_espn_event,
            on_error=self._handle_error,
            enable_circuit_breaker=True
        )
        
        # Register ESPN service with resilience coordinator
        self.resilience_coordinator.register_service(
            name="espn_api",
            priority=ServicePriority.MEDIUM,
            failure_threshold=5,
            timeout=10.0,
            health_check=self.espn_adapter.health_check if self.espn_adapter else None,
            dependencies=[]
        )
        
        logger.info("ESPN stream adapter initialized with circuit breaker")
        
        # Initialize Twitter adapter
        twitter_key = os.getenv("TWITTERAPI_KEY")
        if twitter_key:
            self.twitter_adapter = TwitterStreamAdapter()
            
            # Set callbacks
            self.twitter_adapter.on_sentiment_event = self._handle_twitter_sentiment
            self.twitter_adapter.on_sentiment_shift = self._handle_sentiment_shift
            self.twitter_adapter.on_high_impact_tweet = self._handle_high_impact_tweet
            
            # Register Twitter service with resilience coordinator
            self.resilience_coordinator.register_service(
                name="twitter_stream",
                priority=ServicePriority.LOW,
                failure_threshold=10,
                timeout=15.0,
                health_check=None,  # Twitter adapter doesn't have health check yet
                dependencies=[]
            )
            
            logger.info("Twitter stream adapter initialized with circuit breaker")
        else:
            logger.warning("Twitter API key not found")
        
        # Register service rate limits
        self.throttle_manager.register_service("kalshi", limit=500, window=1.0)
        self.throttle_manager.register_service("espn", limit=100, window=60.0)
        self.throttle_manager.register_service("twitter", limit=200, window=1.0)
        
        # Register backpressure sources
        self.backpressure_controller.register_source("event_buffer", queue_capacity=10000)
        self.backpressure_controller.register_source("kalshi_ws", queue_capacity=5000)
        self.backpressure_controller.register_source("espn_queue", queue_capacity=1000)
        
        logger.info("Stream Manager initialized successfully")
    
    async def start(self):
        """Start all streaming services"""
        if self.is_running:
            logger.warning("Stream Manager already running")
            return
        
        self.is_running = True
        logger.info("Starting Stream Manager...")
        
        # Start Kalshi WebSocket
        if self.kalshi_client:
            await self.kalshi_client.connect()
        
        # Start ESPN adapter
        if self.espn_adapter:
            await self.espn_adapter.start()
        
        # Start Twitter adapter
        if self.twitter_adapter:
            await self.twitter_adapter.start()
        
        # Start correlation engine
        self.tasks.append(
            asyncio.create_task(self._correlation_engine())
        )
        
        # Start health monitor
        self.tasks.append(
            asyncio.create_task(self._health_monitor())
        )
        
        # Start event buffer processor
        self.tasks.append(
            asyncio.create_task(self._process_event_buffer())
        )
        
        # Start resilience monitoring
        await self.resilience_coordinator.start_monitoring(interval=30.0)
        await self.health_monitor.start()
        
        # Start backpressure monitoring
        await self.backpressure_controller.start_monitoring()
        
        # Start Redis bridge
        await self.redis_bridge.start()
        
        # Register health probes
        for service_name in ["kalshi_websocket", "espn_api", "twitter_stream"]:
            if service_name in self.resilience_coordinator.services:
                config = self.resilience_coordinator.services[service_name]
                self.health_monitor.register_probe(
                    service_name=service_name,
                    check_func=config.health_check,
                    interval=30.0
                )
        
        logger.info("Stream Manager started with resilience monitoring")
    
    async def track_market(
        self,
        market_ticker: str,
        game_id: Optional[str] = None,
        home_team: Optional[str] = None,
        away_team: Optional[str] = None,
        sport: str = "nfl"
    ):
        """
        Track a specific market with all data sources
        
        Args:
            market_ticker: Kalshi market ticker
            game_id: ESPN game ID
            home_team: Home team name
            away_team: Away team name
            sport: Sport type
        """
        logger.info(f"Tracking market: {market_ticker}")
        
        # Create market context
        if market_ticker not in self.markets:
            self.markets[market_ticker] = MarketContext(
                market_ticker=market_ticker,
                game_id=game_id,
                home_team=home_team,
                away_team=away_team
            )
        
        # Subscribe to Kalshi market
        if self.kalshi_client:
            await self.kalshi_client.subscribe_ticker(market_ticker)
            await self.kalshi_client.subscribe_orderbook(market_ticker)
        
        # Monitor ESPN game
        if self.espn_adapter and game_id:
            await self.espn_adapter.monitor_game(game_id, sport)
        
        # Setup Twitter monitoring
        if self.twitter_adapter and home_team and away_team:
            await self.twitter_adapter.monitor_game(
                game_id=game_id or f"{home_team}_vs_{away_team}",
                home_team=home_team,
                away_team=away_team,
                sport=sport
            )
        
        # Emit tracking event
        await self._emit_event(UnifiedEvent(
            type=EventType.MARKET_OPPORTUNITY,
            source=DataSource.KALSHI,
            market_ticker=market_ticker,
            game_id=game_id,
            data={
                "action": "tracking_started",
                "home_team": home_team,
                "away_team": away_team
            }
        ))
    
    async def _handle_kalshi_message(self, message: Dict[str, Any]):
        """Process Kalshi WebSocket messages"""
        try:
            msg_type = message.get("type")
            market_ticker = message.get("market_ticker")
            
            if not market_ticker or market_ticker not in self.markets:
                return
            
            context = self.markets[market_ticker]
            
            if msg_type == "ticker":
                # Update prices
                context.yes_price = message.get("yes_price")
                context.no_price = message.get("no_price")
                context.yes_bid = message.get("yes_bid")
                context.yes_ask = message.get("yes_ask")
                context.volume_24h = message.get("volume")
                
                await self._emit_event(UnifiedEvent(
                    type=EventType.PRICE_UPDATE,
                    source=DataSource.KALSHI,
                    market_ticker=market_ticker,
                    data=message
                ))
            
            elif msg_type in ["orderbook_snapshot", "orderbook_delta"]:
                # Update orderbook
                context.orderbook_depth = message.get("orderbook", {})
                
                await self._emit_event(UnifiedEvent(
                    type=EventType.ORDERBOOK_UPDATE,
                    source=DataSource.KALSHI,
                    market_ticker=market_ticker,
                    data=message
                ))
            
            context.last_update = datetime.now()
            
        except Exception as e:
            logger.error(f"Error handling Kalshi message: {e}")
    
    async def _handle_espn_event(self, event):
        """Process ESPN game events"""
        try:
            game_id = event.game_id
            
            # Find market for this game
            market_ticker = None
            for ticker, context in self.markets.items():
                if context.game_id == game_id:
                    market_ticker = ticker
                    break
            
            if not market_ticker:
                return
            
            context = self.markets[market_ticker]
            game_state = event.game_state
            
            # Update game state
            if game_state:
                context.home_score = game_state.home_score
                context.away_score = game_state.away_score
                context.quarter = game_state.quarter
                context.time_remaining = game_state.clock
                context.win_probability = {
                    "home": game_state.home_win_probability,
                    "away": 100 - game_state.home_win_probability
                }
            
            # Map ESPN event types to unified types
            event_map = {
                "TOUCHDOWN": EventType.BIG_PLAY,
                "FIELD_GOAL_MADE": EventType.SCORE_UPDATE,
                "INJURY": EventType.INJURY_ALERT,
                "TURNOVER": EventType.BIG_PLAY
            }
            
            unified_type = event_map.get(event.type.value, EventType.GAME_STATUS)
            
            await self._emit_event(UnifiedEvent(
                type=unified_type,
                source=DataSource.ESPN,
                market_ticker=market_ticker,
                game_id=game_id,
                data={
                    "description": event.description,
                    "game_state": game_state.__dict__ if game_state else {}
                },
                impact_score=event.impact_score,
                requires_action=event.impact_score > 0.7
            ))
            
            context.last_update = datetime.now()
            
        except Exception as e:
            logger.error(f"Error handling ESPN event: {e}")
    
    async def _handle_twitter_sentiment(self, event):
        """Process Twitter sentiment events"""
        try:
            # Match to markets based on teams/keywords
            for market_ticker, context in self.markets.items():
                if not context.home_team or not context.away_team:
                    continue
                
                # Simple matching - could be more sophisticated
                if (context.home_team in str(event) or 
                    context.away_team in str(event)):
                    
                    # Update sentiment
                    context.sentiment_score = event.avg_sentiment_score
                    context.sentiment_velocity = event.sentiment_velocity
                    context.tweet_volume = event.tweet_count
                    
                    await self._emit_event(UnifiedEvent(
                        type=EventType.SENTIMENT_UPDATE,
                        source=DataSource.TWITTER,
                        market_ticker=market_ticker,
                        data={
                            "sentiment": event.avg_sentiment_score,
                            "velocity": event.sentiment_velocity,
                            "volume": event.tweet_count,
                            "impact": event.market_impact.value
                        },
                        impact_score=0.5 if event.market_impact == MarketImpact.HIGH else 0.3
                    ))
                    
                    context.last_update = datetime.now()
                    break
                    
        except Exception as e:
            logger.error(f"Error handling Twitter sentiment: {e}")
    
    async def _handle_sentiment_shift(self, shift):
        """Process sentiment shift alerts"""
        try:
            await self._emit_event(UnifiedEvent(
                type=EventType.SENTIMENT_SHIFT,
                source=DataSource.TWITTER,
                data=shift,
                impact_score=0.8,
                requires_action=True
            ))
        except Exception as e:
            logger.error(f"Error handling sentiment shift: {e}")
    
    async def _handle_high_impact_tweet(self, tweet, sentiment):
        """Process high-impact tweets"""
        try:
            # Find relevant market
            for market_ticker, context in self.markets.items():
                if context.home_team and context.away_team:
                    text = tweet.text.lower()
                    if (context.home_team.lower() in text or
                        context.away_team.lower() in text):
                        
                        # Add to high-impact list
                        context.high_impact_tweets.append({
                            "author": tweet.author.username,
                            "text": tweet.text[:200],
                            "sentiment": sentiment.score,
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        # Keep only last 5
                        context.high_impact_tweets = context.high_impact_tweets[-5:]
                        
                        await self._emit_event(UnifiedEvent(
                            type=EventType.HIGH_IMPACT_TWEET,
                            source=DataSource.TWITTER,
                            market_ticker=market_ticker,
                            data={
                                "author": tweet.author.username,
                                "text": tweet.text,
                                "sentiment": sentiment.score,
                                "keywords": sentiment.market_keywords
                            },
                            impact_score=0.9,
                            requires_action=True
                        ))
                        break
                        
        except Exception as e:
            logger.error(f"Error handling high-impact tweet: {e}")
    
    async def _correlation_engine(self):
        """
        Correlation engine - Identifies opportunities and divergences
        Runs every 5 seconds to analyze cross-source patterns
        """
        while self.is_running:
            try:
                for market_ticker, context in self.markets.items():
                    # Skip if missing data
                    if not context.yes_price or context.sentiment_score == 0:
                        continue
                    
                    # Calculate divergence
                    market_implied_prob = context.yes_price
                    sentiment_implied_prob = 0.5 + (context.sentiment_score * 0.3)
                    sentiment_implied_prob = max(0.01, min(0.99, sentiment_implied_prob))
                    
                    divergence = abs(sentiment_implied_prob - market_implied_prob)
                    context.price_sentiment_divergence = divergence
                    
                    # ESPN win probability divergence
                    if context.win_probability and context.home_team:
                        espn_prob = context.win_probability.get("home", 50) / 100
                        espn_divergence = abs(espn_prob - market_implied_prob)
                        
                        # Triple divergence check
                        if divergence > 0.15 and espn_divergence > 0.15:
                            # All three sources disagree significantly
                            context.opportunity_score = min(
                                (divergence + espn_divergence) / 2,
                                1.0
                            )
                            
                            await self._emit_event(UnifiedEvent(
                                type=EventType.DIVERGENCE_DETECTED,
                                source=DataSource.KALSHI,
                                market_ticker=market_ticker,
                                data={
                                    "market_price": market_implied_prob,
                                    "sentiment_implied": sentiment_implied_prob,
                                    "espn_implied": espn_prob,
                                    "divergence": divergence,
                                    "opportunity_score": context.opportunity_score
                                },
                                impact_score=context.opportunity_score,
                                requires_action=context.opportunity_score > 0.7
                            ))
                    
                    # Calculate risk level
                    if context.opportunity_score > 0.8:
                        context.risk_level = "high"
                    elif context.opportunity_score > 0.6:
                        context.risk_level = "medium"
                    else:
                        context.risk_level = "low"
                
                await asyncio.sleep(5)  # Run every 5 seconds
                
            except Exception as e:
                logger.error(f"Correlation engine error: {e}")
                await asyncio.sleep(10)
    
    async def _health_monitor(self):
        """Monitor health of all connections"""
        while self.is_running:
            try:
                # Check for stale data
                now = datetime.now()
                for market_ticker, context in self.markets.items():
                    staleness = (now - context.last_update).seconds
                    
                    if staleness > 60:
                        logger.warning(f"Stale data for {market_ticker}: {staleness}s")
                        
                        if staleness > 300:  # 5 minutes
                            await self._emit_event(UnifiedEvent(
                                type=EventType.RISK_ALERT,
                                source=DataSource.KALSHI,
                                market_ticker=market_ticker,
                                data={
                                    "alert": "stale_data",
                                    "staleness_seconds": staleness
                                },
                                impact_score=0.6,
                                requires_action=True
                            ))
                
                # Log statistics
                if self.last_event_time:
                    time_since_last = (now - self.last_event_time).seconds
                    if time_since_last > 30:
                        logger.info(f"No events for {time_since_last}s")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(60)
    
    async def _emit_event(self, event: UnifiedEvent):
        """
        Emit event to EventBuffer and DataCoordinator
        
        Args:
            event: Unified event to emit
        """
        # Apply rate limiting
        source = event.source.value if hasattr(event.source, 'value') else str(event.source)
        try:
            await self.throttle_manager.acquire(source)
        except Exception as e:
            logger.warning(f"Rate limited for source {source}: {e}")
            return
        
        # Check backpressure
        if not self.backpressure_controller.should_accept("event_buffer", priority=5):
            logger.warning(f"Event dropped due to backpressure: {event.type}")
            return
        
        self.events_processed += 1
        self.last_event_time = datetime.now()
        
        # Determine priority based on event characteristics
        priority = Priority.NORMAL
        if event.requires_action:
            priority = Priority.CRITICAL
        elif event.impact_score > 0.7:
            priority = Priority.HIGH
        elif event.impact_score < 0.3:
            priority = Priority.LOW
        
        # Create buffered event
        buffered_event = BufferedEvent(
            event_type=event.type.value if hasattr(event.type, 'value') else str(event.type),
            data={
                "market_ticker": event.market_ticker,
                "game_id": event.game_id,
                "source": event.source.value if hasattr(event.source, 'value') else str(event.source),
                "impact_score": event.impact_score,
                "requires_action": event.requires_action,
                **event.data
            },
            priority=priority,
            timestamp=event.timestamp,
            destinations=["DataCoordinator", "RiskManager"] if event.requires_action else ["DataCoordinator"]
        )
        
        # Write to event buffer
        success = await self.event_buffer.write(buffered_event)
        if not success:
            logger.warning(f"Failed to buffer event: {event.type}")
        
        # Publish to Redis for agent consumption
        if self.redis_bridge and self.redis_bridge.is_running:
            try:
                # Route to appropriate Redis channel based on event type
                if event.type == EventType.PRICE_UPDATE:
                    await self.redis_bridge.handle_market_update(event.data)
                elif event.type == EventType.TRADE_EXECUTED:
                    await self.redis_bridge.handle_trade(event.data)
                elif event.type == EventType.ORDERBOOK_UPDATE:
                    await self.redis_bridge.handle_orderbook(event.data)
                elif event.source == DataSource.ESPN:
                    await self.redis_bridge.handle_espn_update(event.data)
                
                # Twitter sentiment routes
                if event.source == DataSource.TWITTER or event.type in (
                    EventType.SENTIMENT_UPDATE,
                    EventType.SENTIMENT_SHIFT,
                    EventType.HIGH_IMPACT_TWEET
                ):
                    await self.redis_bridge.handle_twitter_sentiment({
                        "type": event.type.value if hasattr(event.type, 'value') else str(event.type),
                        **event.data
                    })
                    
                # For high-impact events, publish as signals
                if event.impact_score > 0.7:
                    signal = {
                        "market_ticker": event.market_ticker,
                        "action": "analyze",
                        "impact_score": event.impact_score,
                        "source": event.source.value if hasattr(event.source, 'value') else str(event.source),
                        "data": event.data
                    }
                    await self.redis_bridge.publish_signal(signal)
            except Exception as e:
                logger.error(f"Failed to publish to Redis: {e}")
        
        # Add to window aggregator for time-based analysis
        if event.market_ticker:
            window_results = await self.window_aggregator.add_event(
                value=event.impact_score,
                metadata={
                    "type": event.type.value if hasattr(event.type, 'value') else str(event.type),
                    "market": event.market_ticker
                }
            )
            
            # Check for anomalies in window data
            if self.window_aggregator.detect_anomaly("5m"):
                logger.warning(f"Anomaly detected in {event.market_ticker} data stream")
                # Create risk alert event
                risk_event = BufferedEvent(
                    event_type="ANOMALY_DETECTED",
                    data={
                        "market_ticker": event.market_ticker,
                        "window_stats": self.window_aggregator.get_aggregates("5m"),
                        "velocity": self.window_aggregator.calculate_velocity("5m")
                    },
                    priority=Priority.HIGH,
                    destinations=["RiskManager"]
                )
                await self.event_buffer.write(risk_event)
        
        # For critical events, also send directly to DataCoordinator
        if priority == Priority.CRITICAL:
            event_data = {
                "command": "stream_event",
                "type": event.type.value if hasattr(event.type, 'value') else str(event.type),
                "market_ticker": event.market_ticker,
                "data": event.data,
                "impact_score": event.impact_score,
                "requires_action": event.requires_action,
                "timestamp": event.timestamp.isoformat()
            }
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "http://localhost:3500/agents/DataCoordinator",
                        json=event_data,
                        headers={"Content-Type": "application/json"}
                    ) as resp:
                        if resp.status == 200:
                            logger.info(f"Critical event sent to DataCoordinator: {event.type}")
                        else:
                            logger.warning(f"Failed to send critical event: {resp.status}")
            except Exception as e:
                logger.error(f"Error sending critical event: {e}")
        
        # Call local event handlers (for backward compatibility)
        if event.type in self.event_handlers:
            for handler in self.event_handlers[event.type]:
                try:
                    await handler(event)
                except Exception as e:
                    logger.error(f"Event handler error: {e}")
    
    async def _handle_error(self, error: Dict[str, Any]):
        """Handle errors from data sources"""
        logger.error(f"Stream error: {error}")
        
        # Send error to DataCoordinator for risk monitoring
        try:
            async with aiohttp.ClientSession() as session:
                await session.post(
                    "http://localhost:3500/agents/DataCoordinator",
                    json={
                        "command": "stream_event",
                        "type": "system_error",
                        "data": error,
                        "requires_action": True
                    },
                    headers={"Content-Type": "application/json"}
                )
        except Exception as e:
            logger.error(f"Could not report error to DataCoordinator: {e}")
    
    def register_handler(self, event_type: EventType, handler: Callable):
        """
        Register local event handler
        
        Args:
            event_type: Type of event to handle
            handler: Async callback function
        """
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def get_market_context(self, market_ticker: str) -> Optional[MarketContext]:
        """Get current context for a market"""
        return self.markets.get(market_ticker)
    
    def get_all_contexts(self) -> Dict[str, MarketContext]:
        """Get all market contexts"""
        return self.markets.copy()
    
    async def _handle_pressure_change(self, old_level: PressureLevel, new_level: PressureLevel):
        """Handle backpressure level changes"""
        logger.info(f"Backpressure changed: {old_level.name} -> {new_level.name}")
        
        # Notify agents if critical
        if new_level == PressureLevel.CRITICAL:
            try:
                async with aiohttp.ClientSession() as session:
                    await session.post(
                        "http://localhost:3500/agents/DataCoordinator",
                        json={
                            "command": "backpressure_alert",
                            "level": new_level.name,
                            "timestamp": datetime.now().isoformat()
                        }
                    )
            except Exception as e:
                logger.error(f"Failed to notify backpressure: {e}")
    
    async def _handle_throttle_change(self, throttle_factors: Dict[str, float]):
        """Handle throttle factor changes"""
        logger.info(f"Throttle factors updated: {throttle_factors}")
        
        # Could adjust polling rates, batch sizes, etc. based on throttle factors
        
    async def _process_event_buffer(self):
        """Process events from the event buffer"""
        logger.info("Starting event buffer processor")
        
        while self.is_running:
            try:
                # Update backpressure metrics
                buffer_stats = self.event_buffer.get_stats()
                self.backpressure_controller.update_metrics(
                    source="event_buffer",
                    queue_depth=buffer_stats.get("current_size", 0),
                    latency_ms=buffer_stats.get("avg_processing_time_ms", 0)
                )
                
                # Process events from buffer
                events = await self.event_buffer.read_blocking(
                    consumer_id="stream_manager",
                    batch_size=10,
                    timeout=0.1
                )
                
                for event in events:
                    # Route events to destinations
                    for destination in event.destinations:
                        if destination in ["DataCoordinator", "RiskManager"]:
                            # Send to agent via HTTP
                            try:
                                async with aiohttp.ClientSession() as session:
                                    await session.post(
                                        f"http://localhost:3500/agents/{destination}",
                                        json={
                                            "command": "buffered_event",
                                            "event": {
                                                "type": event.event_type,
                                                "data": event.data,
                                                "priority": event.priority.name,
                                                "timestamp": event.timestamp.isoformat()
                                            }
                                        },
                                        headers={"Content-Type": "application/json"}
                                    )
                            except Exception as e:
                                logger.error(f"Failed to route event to {destination}: {e}")
                
                # Small yield to prevent CPU hogging
                await asyncio.sleep(0)
                
            except Exception as e:
                logger.error(f"Event buffer processor error: {e}")
                await asyncio.sleep(1)
    
    async def _handle_emergency_stop(self):
        """Handle emergency stop from resilience coordinator"""
        logger.critical("EMERGENCY STOP TRIGGERED - Shutting down all trading operations")
        
        # Stop accepting new events
        self.is_running = False
        
        # Send emergency notification to agents
        try:
            async with aiohttp.ClientSession() as session:
                for agent in ["DataCoordinator", "TradeExecutor", "RiskManager"]:
                    await session.post(
                        f"http://localhost:3500/agents/{agent}",
                        json={
                            "command": "emergency_stop",
                            "reason": "Circuit breaker emergency",
                            "timestamp": datetime.now().isoformat()
                        },
                        headers={"Content-Type": "application/json"}
                    )
        except Exception as e:
            logger.error(f"Failed to notify agents of emergency stop: {e}")
        
        # Stop all data streams
        await self.stop()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive stream manager statistics"""
        return {
            "is_running": self.is_running,
            "markets_tracked": len(self.markets),
            "events_processed": self.events_processed,
            "last_event_time": self.last_event_time.isoformat() if self.last_event_time else None,
            "data_sources": {
                "kalshi": self.kalshi_client is not None,
                "espn": self.espn_adapter is not None,
                "twitter": self.twitter_adapter is not None
            },
            "buffer_stats": self.event_buffer.get_stats(),
            "cache_stats": self.state_manager.get_cache_stats(),
            "window_stats": self.window_aggregator.get_stats(),
            "resilience": {
                "coordinator_status": self.resilience_coordinator.get_status(),
                "health_monitor_status": self.health_monitor.get_status(),
                "degradation_level": self.resilience_coordinator.degradation_level.name
            },
            "rate_limiting": self.throttle_manager.get_stats(),
            "backpressure": self.backpressure_controller.get_status()
        }
    
    async def stop(self):
        """Stop all streaming services"""
        self.is_running = False
        
        # Stop Redis bridge
        await self.redis_bridge.stop()
        
        # Stop resilience monitoring
        await self.resilience_coordinator.stop_monitoring()
        await self.health_monitor.stop()
        
        # Cancel tasks
        for task in self.tasks:
            task.cancel()
        
        # Stop data sources
        if self.kalshi_client:
            await self.kalshi_client.disconnect()
        
        if self.espn_adapter:
            await self.espn_adapter.stop()
        
        if self.twitter_adapter:
            await self.twitter_adapter.stop()
        
        logger.info(f"Stream Manager stopped. Events processed: {self.events_processed}")


# Standalone usage (for testing without Agentuity)
async def example_usage():
    """Example of using StreamManager standalone"""
    
    # Event handler
    async def handle_opportunity(event: UnifiedEvent):
        if event.type == EventType.DIVERGENCE_DETECTED:
            print(f"🎯 OPPORTUNITY: {event.data}")
    
    # Initialize
    manager = StreamManager()
    manager.register_handler(EventType.DIVERGENCE_DETECTED, handle_opportunity)
    
    await manager.initialize()
    await manager.start()
    
    # Track a market
    await manager.track_market(
        market_ticker="SUPERBOWL-2025",
        game_id="401547435",
        home_team="Chiefs",
        away_team="Bills",
        sport="nfl"
    )
    
    # Run for a while
    await asyncio.sleep(300)
    
    # Get statistics
    stats = manager.get_statistics()
    print(f"Statistics: {stats}")
    
    await manager.stop()


if __name__ == "__main__":
    asyncio.run(example_usage())
