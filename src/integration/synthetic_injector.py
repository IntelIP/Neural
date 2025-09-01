"""
Synthetic Data Injector

Injects synthetic training data into Redis channels with realistic timing,
sequencing, and market dynamics for agent training.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import logging
import json
import redis.asyncio as redis
import random
import numpy as np

from ..sdk.core.base_adapter import StandardizedEvent
from ..synthetic_data.generators.game_engine import SyntheticGame
from ..synthetic_data.generators.market_simulator import TradingScenario, MarketEvent


class EventTiming(Enum):
    """Event timing strategies"""
    REALTIME = "realtime"  # Real-world timing
    ACCELERATED = "accelerated"  # Faster than realtime
    BURST = "burst"  # All at once
    ADAPTIVE = "adaptive"  # Adjust based on agent response


@dataclass
class InjectionConfig:
    """Configuration for data injection"""
    timing_mode: EventTiming = EventTiming.ACCELERATED
    acceleration_factor: float = 10.0  # 10x speed
    
    # Realistic delays (in seconds)
    min_event_delay: float = 0.05  # 50ms minimum
    max_event_delay: float = 5.0  # 5s maximum
    market_update_frequency: float = 0.5  # Market updates every 500ms
    
    # Noise and realism
    add_timing_jitter: bool = True
    jitter_factor: float = 0.2  # ±20% timing variation
    add_missing_data: bool = True  # Simulate data gaps
    missing_data_probability: float = 0.02  # 2% chance of missing data
    
    # Market dynamics
    add_market_noise: bool = True
    price_noise_stddev: float = 0.01  # 1% price noise
    volume_multiplier_range: Tuple[float, float] = (0.8, 1.2)
    
    # Redis configuration
    redis_url: str = "redis://localhost:6379"
    channel_prefix: str = ""  # Empty for production channels, "training:" for isolated
    batch_size: int = 100  # Events to buffer before publishing
    
    # Performance
    max_concurrent_injections: int = 10
    enable_backpressure: bool = True
    backpressure_threshold: int = 1000  # Pause if this many events pending


@dataclass
class InjectionMetrics:
    """Metrics for injection performance"""
    events_injected: int = 0
    events_failed: int = 0
    total_latency: float = 0.0
    min_latency: float = float('inf')
    max_latency: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    
    @property
    def avg_latency(self) -> float:
        return self.total_latency / self.events_injected if self.events_injected > 0 else 0.0
    
    @property
    def success_rate(self) -> float:
        total = self.events_injected + self.events_failed
        return self.events_injected / total if total > 0 else 0.0
    
    @property
    def events_per_second(self) -> float:
        elapsed = (datetime.now() - self.start_time).total_seconds()
        return self.events_injected / elapsed if elapsed > 0 else 0.0


class SyntheticDataInjector:
    """
    Injects synthetic data into Redis channels for agent training.
    
    Handles timing, sequencing, and realistic market dynamics to create
    believable training scenarios.
    """
    
    def __init__(self, config: InjectionConfig = None):
        self.config = config or InjectionConfig()
        self.logger = logging.getLogger(__name__)
        
        # Redis connections
        self.redis_client: Optional[redis.Redis] = None
        self.publisher: Optional[redis.Redis] = None
        
        # Injection state
        self.is_injecting = False
        self.injection_tasks: List[asyncio.Task] = []
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.pending_events = 0
        
        # Metrics
        self.metrics = InjectionMetrics()
        
        # Channel mappings
        self.channel_map = {
            "game_event": "espn:games",
            "market_update": "kalshi:markets",
            "trade": "kalshi:trades",
            "signal": "kalshi:signals",
            "sentiment": "twitter:sentiment",
            "orderbook": "kalshi:orderbook"
        }
        
        # Market state tracking (for realistic updates)
        self.market_states: Dict[str, Dict[str, float]] = {}
    
    async def initialize(self) -> None:
        """Initialize the injector"""
        try:
            self.logger.info("Initializing Synthetic Data Injector")
            
            # Connect to Redis
            self.redis_client = redis.from_url(self.config.redis_url)
            self.publisher = redis.from_url(self.config.redis_url)
            
            # Test connection
            await self.redis_client.ping()
            
            # Start background publisher
            self.injection_tasks.append(
                asyncio.create_task(self._publisher_loop())
            )
            
            self.logger.info("Synthetic Data Injector initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize injector: {e}")
            raise
    
    async def inject_game_scenario(
        self,
        game: SyntheticGame,
        market_ticker: Optional[str] = None,
        start_immediately: bool = True
    ) -> str:
        """
        Inject a complete game scenario with synchronized market updates.
        
        Args:
            game: Synthetic game to inject
            market_ticker: Associated market ticker for price updates
            start_immediately: Whether to start injection immediately
            
        Returns:
            Injection ID for tracking
        """
        try:
            injection_id = f"game_{game.game_id}_{datetime.now().timestamp()}"
            
            # Convert game to events
            events = self._game_to_events(game, market_ticker)
            
            # Calculate timing
            event_timings = self._calculate_event_timings(events)
            
            # Create injection task
            if start_immediately:
                task = asyncio.create_task(
                    self._inject_event_sequence(injection_id, events, event_timings)
                )
                self.injection_tasks.append(task)
            else:
                # Queue for later injection
                for event, timing in zip(events, event_timings):
                    await self.event_queue.put((event, timing))
            
            self.logger.info(f"Injecting game scenario {injection_id} with {len(events)} events")
            return injection_id
            
        except Exception as e:
            self.logger.error(f"Failed to inject game scenario: {e}")
            raise
    
    async def inject_trading_scenario(
        self,
        scenario: TradingScenario,
        include_orderbook: bool = True
    ) -> str:
        """
        Inject a trading scenario with market events and price updates.
        
        Args:
            scenario: Trading scenario to inject
            include_orderbook: Whether to include orderbook updates
            
        Returns:
            Injection ID for tracking
        """
        try:
            injection_id = f"trading_{scenario.scenario_id}_{datetime.now().timestamp()}"
            
            # Convert scenario to events
            events = self._trading_scenario_to_events(scenario, include_orderbook)
            
            # Calculate timing with market update frequency
            event_timings = self._calculate_market_timings(events)
            
            # Start injection
            task = asyncio.create_task(
                self._inject_event_sequence(injection_id, events, event_timings)
            )
            self.injection_tasks.append(task)
            
            self.logger.info(f"Injecting trading scenario {injection_id} with {len(events)} events")
            return injection_id
            
        except Exception as e:
            self.logger.error(f"Failed to inject trading scenario: {e}")
            raise
    
    def _game_to_events(self, game: SyntheticGame, market_ticker: Optional[str]) -> List[Dict[str, Any]]:
        """Convert a synthetic game to injectable events"""
        events = []
        
        for play in game.plays:
            # Game event
            event = {
                "type": "game_event",
                "event_id": f"play_{play.play_id}",
                "game_id": game.game_id,
                "timestamp": play.timestamp,
                "data": {
                    "play_type": play.play_type,
                    "description": play.description,
                    "team_possession": play.team_possession,
                    "score_home": play.score_home,
                    "score_away": play.score_away,
                    "quarter": play.quarter,
                    "time_remaining": play.time_remaining,
                    "yards_gained": getattr(play, 'yards_gained', 0),
                    "down": getattr(play, 'down', None),
                    "yards_to_go": getattr(play, 'yards_to_go', None)
                }
            }
            events.append(event)
            
            # Add correlated market update if ticker provided
            if market_ticker and self._is_significant_play(play):
                market_event = self._generate_market_reaction(play, market_ticker)
                events.append(market_event)
        
        return events
    
    def _is_significant_play(self, play) -> bool:
        """Determine if a play should trigger market movement"""
        significant_types = ["touchdown", "field_goal", "interception", "fumble", "injury"]
        return any(sig in play.play_type.lower() for sig in significant_types)
    
    def _generate_market_reaction(self, play, market_ticker: str) -> Dict[str, Any]:
        """Generate a market price reaction to a game event"""
        # Get current market state or initialize
        if market_ticker not in self.market_states:
            self.market_states[market_ticker] = {
                "yes_price": 0.5,
                "no_price": 0.5,
                "volume": 1000
            }
        
        current_state = self.market_states[market_ticker]
        
        # Calculate price impact based on play type
        impact = 0.0
        if "touchdown" in play.play_type.lower():
            # Touchdown by home team increases yes price
            # Assuming home team is the team we're tracking for "YES" price
            if hasattr(play, 'team_possession'):
                # If possession matches home indication, positive impact
                impact = random.uniform(0.02, 0.05) if "home" in play.team_possession.lower() else random.uniform(-0.05, -0.02)
            else:
                impact = random.uniform(-0.02, 0.02)  # Random if unknown
        elif "field_goal" in play.play_type.lower():
            if hasattr(play, 'team_possession'):
                impact = random.uniform(-0.01, 0.01) if "home" in play.team_possession.lower() else random.uniform(-0.02, 0.0)
            else:
                impact = random.uniform(-0.015, 0.005)
        elif "interception" in play.play_type.lower() or "fumble" in play.play_type.lower():
            if hasattr(play, 'team_possession'):
                impact = random.uniform(-0.03, -0.01) if "home" in play.team_possession.lower() else random.uniform(0.01, 0.03)
            else:
                impact = random.uniform(-0.01, 0.01)
        
        # Apply impact with noise
        if self.config.add_market_noise:
            noise = np.random.normal(0, self.config.price_noise_stddev)
            impact += noise
        
        # Update prices
        new_yes_price = max(0.01, min(0.99, current_state["yes_price"] + impact))
        new_no_price = 1.0 - new_yes_price
        
        # Update volume
        volume_multiplier = random.uniform(*self.config.volume_multiplier_range)
        new_volume = current_state["volume"] * volume_multiplier
        
        # Store new state
        self.market_states[market_ticker] = {
            "yes_price": new_yes_price,
            "no_price": new_no_price,
            "volume": new_volume
        }
        
        return {
            "type": "market_update",
            "event_id": f"market_{play.play_id}",
            "market_ticker": market_ticker,
            "timestamp": play.timestamp + timedelta(seconds=random.uniform(0.1, 0.5)),
            "data": {
                "yes_price": new_yes_price,
                "no_price": new_no_price,
                "yes_bid": new_yes_price - 0.01,
                "yes_ask": new_yes_price + 0.01,
                "no_bid": new_no_price - 0.01,
                "no_ask": new_no_price + 0.01,
                "volume": new_volume,
                "trigger_event": play.play_id
            }
        }
    
    def _trading_scenario_to_events(self, scenario: TradingScenario, include_orderbook: bool) -> List[Dict[str, Any]]:
        """Convert trading scenario to injectable events"""
        events = []
        
        for market_event in scenario.events:
            # Market price update
            event = {
                "type": "market_update",
                "event_id": market_event.event_id,
                "market_ticker": scenario.market_ticker,
                "timestamp": market_event.timestamp,
                "data": {
                    "event_type": market_event.event_type,
                    "yes_price": market_event.price,
                    "no_price": 1.0 - market_event.price,
                    "volume": market_event.volume,
                    "price_change": market_event.price_change,
                    "information_content": market_event.information_value
                }
            }
            events.append(event)
            
            # Add orderbook update if requested
            if include_orderbook and random.random() < 0.3:  # 30% chance
                orderbook_event = self._generate_orderbook_update(scenario.market_ticker, market_event)
                events.append(orderbook_event)
        
        return events
    
    def _generate_orderbook_update(self, market_ticker: str, market_event: MarketEvent) -> Dict[str, Any]:
        """Generate a realistic orderbook update"""
        base_price = market_event.price
        
        # Generate bid/ask levels
        levels = 5
        bids = []
        asks = []
        
        for i in range(levels):
            spread = 0.01 * (i + 1)
            bid_price = base_price - spread
            ask_price = base_price + spread
            
            # Volume decreases with distance from mid
            volume = market_event.volume * (1.0 / (i + 1))
            
            if bid_price > 0:
                bids.append({
                    "price": bid_price,
                    "quantity": int(volume * random.uniform(0.8, 1.2))
                })
            
            if ask_price < 1:
                asks.append({
                    "price": ask_price,
                    "quantity": int(volume * random.uniform(0.8, 1.2))
                })
        
        return {
            "type": "orderbook",
            "event_id": f"orderbook_{market_event.event_id}",
            "market_ticker": market_ticker,
            "timestamp": market_event.timestamp + timedelta(milliseconds=50),
            "data": {
                "bids": bids,
                "asks": asks,
                "mid_price": base_price,
                "spread": asks[0]["price"] - bids[0]["price"] if bids and asks else 0.02
            }
        }
    
    def _calculate_event_timings(self, events: List[Dict[str, Any]]) -> List[float]:
        """Calculate realistic timing delays for events"""
        timings = []
        
        for i, event in enumerate(events):
            if self.config.timing_mode == EventTiming.BURST:
                delay = 0.0  # No delay
            elif self.config.timing_mode == EventTiming.REALTIME:
                # Calculate actual time difference
                if i > 0:
                    prev_timestamp = events[i-1].get("timestamp", datetime.now())
                    curr_timestamp = event.get("timestamp", datetime.now())
                    
                    if isinstance(prev_timestamp, str):
                        prev_timestamp = datetime.fromisoformat(prev_timestamp)
                    if isinstance(curr_timestamp, str):
                        curr_timestamp = datetime.fromisoformat(curr_timestamp)
                    
                    delay = (curr_timestamp - prev_timestamp).total_seconds()
                else:
                    delay = 0.0
            elif self.config.timing_mode == EventTiming.ACCELERATED:
                # Base delay with acceleration
                base_delay = random.uniform(self.config.min_event_delay, self.config.max_event_delay)
                delay = base_delay / self.config.acceleration_factor
            else:  # ADAPTIVE
                # Adjust based on event type and importance
                if event["type"] == "market_update":
                    delay = self.config.market_update_frequency / self.config.acceleration_factor
                else:
                    delay = random.uniform(0.1, 1.0) / self.config.acceleration_factor
            
            # Add jitter if configured
            if self.config.add_timing_jitter:
                jitter = random.gauss(0, delay * self.config.jitter_factor) if delay > 0 else 0
                delay = max(0.001, delay + jitter)  # Ensure positive
            
            timings.append(delay)
        
        return timings
    
    def _calculate_market_timings(self, events: List[Dict[str, Any]]) -> List[float]:
        """Calculate timing specifically for market events"""
        timings = []
        
        for event in events:
            if event["type"] == "market_update":
                # Regular market updates
                delay = self.config.market_update_frequency / self.config.acceleration_factor
            elif event["type"] == "orderbook":
                # Orderbook updates are more frequent
                delay = (self.config.market_update_frequency * 0.5) / self.config.acceleration_factor
            else:
                # Other events
                delay = random.uniform(0.5, 2.0) / self.config.acceleration_factor
            
            # Add realistic variation
            if self.config.add_timing_jitter:
                delay *= random.uniform(0.8, 1.2)
            
            timings.append(delay)
        
        return timings
    
    async def _inject_event_sequence(
        self,
        injection_id: str,
        events: List[Dict[str, Any]],
        timings: List[float]
    ) -> None:
        """Inject a sequence of events with specified timing"""
        try:
            self.is_injecting = True
            
            for event, delay in zip(events, timings):
                # Check backpressure
                if self.config.enable_backpressure and self.pending_events > self.config.backpressure_threshold:
                    await self._wait_for_backpressure()
                
                # Wait for timing
                if delay > 0:
                    await asyncio.sleep(delay)
                
                # Simulate missing data if configured
                if self.config.add_missing_data and random.random() < self.config.missing_data_probability:
                    self.logger.debug(f"Simulating missing data for event {event.get('event_id')}")
                    continue
                
                # Queue event for publishing
                await self.event_queue.put((event, injection_id))
                self.pending_events += 1
            
            self.logger.info(f"Completed injection sequence {injection_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to inject event sequence: {e}")
            self.metrics.events_failed += len(events)
        finally:
            self.is_injecting = False
    
    async def _wait_for_backpressure(self) -> None:
        """Wait for backpressure to clear"""
        self.logger.debug("Backpressure detected, waiting...")
        while self.pending_events > self.config.backpressure_threshold * 0.5:
            await asyncio.sleep(0.1)
    
    async def _publisher_loop(self) -> None:
        """Background loop for publishing queued events"""
        batch = []
        
        while True:
            try:
                # Get event from queue with timeout
                try:
                    event, injection_id = await asyncio.wait_for(
                        self.event_queue.get(), timeout=1.0
                    )
                    batch.append((event, injection_id))
                except asyncio.TimeoutError:
                    # Flush batch if we have events
                    if batch:
                        await self._publish_batch(batch)
                        batch = []
                    continue
                
                # Publish batch if full
                if len(batch) >= self.config.batch_size:
                    await self._publish_batch(batch)
                    batch = []
                    
            except Exception as e:
                self.logger.error(f"Publisher loop error: {e}")
                await asyncio.sleep(1)
    
    async def _publish_batch(self, batch: List[Tuple[Dict[str, Any], str]]) -> None:
        """Publish a batch of events to Redis"""
        try:
            start_time = datetime.now()
            
            for event, injection_id in batch:
                channel = self._get_channel_for_event(event)
                
                # Add channel prefix if configured
                if self.config.channel_prefix:
                    channel = f"{self.config.channel_prefix}{channel}"
                
                # Prepare message
                message = {
                    "timestamp": event["timestamp"].isoformat() if hasattr(event["timestamp"], 'isoformat') else str(event["timestamp"]),
                    "source": "synthetic_injector",
                    "injection_id": injection_id,
                    "type": event["type"],
                    "data": event["data"]
                }
                
                # Add event-specific fields
                if "event_id" in event:
                    message["event_id"] = event["event_id"]
                if "game_id" in event:
                    message["game_id"] = event["game_id"]
                if "market_ticker" in event:
                    message["market_ticker"] = event["market_ticker"]
                
                # Publish to Redis
                await self.publisher.publish(channel, json.dumps(message))
                
                # Update metrics
                self.metrics.events_injected += 1
                self.pending_events -= 1
            
            # Update latency metrics
            latency = (datetime.now() - start_time).total_seconds()
            self.metrics.total_latency += latency
            self.metrics.min_latency = min(self.metrics.min_latency, latency)
            self.metrics.max_latency = max(self.metrics.max_latency, latency)
            
        except Exception as e:
            self.logger.error(f"Failed to publish batch: {e}")
            self.metrics.events_failed += len(batch)
            self.pending_events -= len(batch)
    
    def _get_channel_for_event(self, event: Dict[str, Any]) -> str:
        """Get the appropriate Redis channel for an event type"""
        event_type = event.get("type", "unknown")
        return self.channel_map.get(event_type, "kalshi:unknown")
    
    async def inject_burst(self, events: List[StandardizedEvent]) -> str:
        """
        Inject a burst of events as quickly as possible.
        
        Args:
            events: List of standardized events to inject
            
        Returns:
            Injection ID
        """
        try:
            injection_id = f"burst_{datetime.now().timestamp()}"
            
            # Convert to injectable format
            injectable_events = []
            for event in events:
                injectable = {
                    "type": self._event_type_to_channel_type(event.event_type),
                    "event_id": event.event_id,
                    "timestamp": event.timestamp,
                    "data": {
                        "description": event.description,
                        "metadata": event.metadata
                    }
                }
                
                if event.game_id:
                    injectable["game_id"] = event.game_id
                
                injectable_events.append(injectable)
            
            # Queue all events immediately
            for event in injectable_events:
                await self.event_queue.put((event, injection_id))
                self.pending_events += 1
            
            self.logger.info(f"Queued burst injection {injection_id} with {len(events)} events")
            return injection_id
            
        except Exception as e:
            self.logger.error(f"Failed to inject burst: {e}")
            raise
    
    def _event_type_to_channel_type(self, event_type: str) -> str:
        """Convert StandardizedEvent type to channel type"""
        mapping = {
            "score_update": "game_event",
            "big_play": "game_event",
            "market_update": "market_update",
            "trade": "trade",
            "signal": "signal"
        }
        return mapping.get(event_type, "game_event")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get injection metrics"""
        return {
            "events_injected": self.metrics.events_injected,
            "events_failed": self.metrics.events_failed,
            "success_rate": f"{self.metrics.success_rate:.2%}",
            "avg_latency_ms": self.metrics.avg_latency * 1000,
            "min_latency_ms": self.metrics.min_latency * 1000 if self.metrics.min_latency != float('inf') else 0,
            "max_latency_ms": self.metrics.max_latency * 1000,
            "events_per_second": self.metrics.events_per_second,
            "pending_events": self.pending_events,
            "is_injecting": self.is_injecting
        }
    
    async def shutdown(self) -> None:
        """Shutdown the injector"""
        try:
            self.logger.info("Shutting down Synthetic Data Injector")
            
            # Cancel injection tasks
            for task in self.injection_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            if self.injection_tasks:
                await asyncio.gather(*self.injection_tasks, return_exceptions=True)
            
            # Close Redis connections
            if self.redis_client:
                await self.redis_client.close()
            if self.publisher:
                await self.publisher.close()
            
            self.logger.info(f"Injector shutdown complete. Final metrics: {self.get_metrics()}")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")