"""
Unified Stream Manager

Coordinates multiple WebSocket data sources for real-time trading.
Provides event correlation and unified data access.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import statistics

from ..kalshi.websocket_adapter import KalshiWebSocketAdapter, KalshiChannel
from ..odds.rest_adapter import OddsAPIAdapter

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Unified event types across all sources."""
    # Market events
    PRICE_UPDATE = "price_update"
    ORDERBOOK_UPDATE = "orderbook_update"
    TRADE_EXECUTED = "trade_executed"
    
    # Odds events
    ODDS_UPDATE = "odds_update"
    LINE_MOVEMENT = "line_movement"
    
    # Correlation events
    ARBITRAGE_OPPORTUNITY = "arbitrage_opportunity"
    DIVERGENCE_DETECTED = "divergence_detected"
    SIGNAL_GENERATED = "signal_generated"


@dataclass
class UnifiedMarketData:
    """Unified market data from all sources."""
    ticker: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Kalshi data
    kalshi_yes_price: Optional[float] = None
    kalshi_no_price: Optional[float] = None
    kalshi_yes_bid: Optional[float] = None
    kalshi_yes_ask: Optional[float] = None
    kalshi_no_bid: Optional[float] = None
    kalshi_no_ask: Optional[float] = None
    kalshi_volume: Optional[float] = None
    kalshi_open_interest: Optional[float] = None
    
    # Odds data
    odds_consensus_home: Optional[float] = None
    odds_consensus_away: Optional[float] = None
    odds_best_home: Optional[Dict[str, Any]] = None
    odds_best_away: Optional[Dict[str, Any]] = None
    odds_spread: Optional[float] = None
    odds_total: Optional[float] = None
    
    # Computed metrics
    kalshi_spread: Optional[float] = None
    odds_implied_prob_home: Optional[float] = None
    odds_implied_prob_away: Optional[float] = None
    divergence_score: Optional[float] = None
    arbitrage_exists: bool = False
    
    def compute_metrics(self):
        """Compute derived metrics."""
        # Kalshi spread
        if self.kalshi_yes_ask and self.kalshi_yes_bid:
            self.kalshi_spread = self.kalshi_yes_ask - self.kalshi_yes_bid
        
        # Odds implied probabilities
        if self.odds_consensus_home:
            self.odds_implied_prob_home = self._american_to_probability(self.odds_consensus_home)
        if self.odds_consensus_away:
            self.odds_implied_prob_away = self._american_to_probability(self.odds_consensus_away)
        
        # Divergence score
        if self.kalshi_yes_price and self.odds_implied_prob_home:
            self.divergence_score = abs(self.kalshi_yes_price - self.odds_implied_prob_home)
        
        # Check for arbitrage
        if self.divergence_score and self.divergence_score > 0.05:  # 5% divergence
            self.arbitrage_exists = True
    
    def _american_to_probability(self, odds: float) -> float:
        """Convert American odds to probability."""
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)


@dataclass
class StreamConfig:
    """Configuration for unified stream manager."""
    enable_kalshi: bool = True
    enable_odds_polling: bool = True
    odds_poll_interval: int = 30  # seconds
    correlation_window: int = 5  # seconds
    divergence_threshold: float = 0.05  # 5%
    max_tracked_markets: int = 100


class UnifiedStreamManager:
    """
    Manages multiple data streams for real-time trading.
    
    Features:
    - Coordinates Kalshi WebSocket and Odds API
    - Correlates events across sources
    - Detects arbitrage opportunities
    - Generates unified trading signals
    """
    
    def __init__(self, config: StreamConfig = None):
        """
        Initialize unified stream manager.
        
        Args:
            config: Stream configuration
        """
        self.config = config or StreamConfig()
        
        # Data sources
        self.kalshi_ws: Optional[KalshiWebSocketAdapter] = None
        self.odds_api: Optional[OddsAPIAdapter] = None
        
        # Market data storage
        self._market_data: Dict[str, UnifiedMarketData] = {}
        self._market_history: Dict[str, List[UnifiedMarketData]] = defaultdict(list)
        
        # Event handlers
        self._event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Background tasks
        self._tasks: List[asyncio.Task] = []
        
        # Tracking
        self._tracked_markets: Set[str] = set()
        self._game_mappings: Dict[str, str] = {}  # Map Kalshi ticker to game ID
        
        logger.info("Unified stream manager initialized")
    
    async def start(self):
        """Start all data streams."""
        logger.info("Starting unified stream manager")
        
        # Initialize Kalshi WebSocket
        if self.config.enable_kalshi:
            await self._start_kalshi_stream()
        
        # Initialize Odds polling
        if self.config.enable_odds_polling:
            await self._start_odds_polling()
        
        # Start correlation engine
        task = asyncio.create_task(self._correlation_loop())
        self._tasks.append(task)
        
        logger.info("Unified stream manager started")
    
    async def stop(self):
        """Stop all data streams."""
        logger.info("Stopping unified stream manager")
        
        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
        
        # Disconnect sources
        if self.kalshi_ws:
            await self.kalshi_ws.disconnect()
        
        logger.info("Unified stream manager stopped")
    
    # Data source initialization
    
    async def _start_kalshi_stream(self):
        """Start Kalshi WebSocket stream."""
        self.kalshi_ws = KalshiWebSocketAdapter()
        
        # Register event handlers
        self.kalshi_ws.on("ticker", self._handle_kalshi_ticker)
        self.kalshi_ws.on("orderbook", self._handle_kalshi_orderbook)
        self.kalshi_ws.on("trade", self._handle_kalshi_trade)
        self.kalshi_ws.on("connected", self._handle_kalshi_connected)
        self.kalshi_ws.on("disconnected", self._handle_kalshi_disconnected)
        
        # Connect
        if await self.kalshi_ws.connect():
            logger.info("Kalshi WebSocket connected")
        else:
            logger.error("Failed to connect Kalshi WebSocket")
    
    async def _start_odds_polling(self):
        """Start Odds API polling."""
        self.odds_api = OddsAPIAdapter()
        
        # Start polling task
        task = asyncio.create_task(self._odds_polling_loop())
        self._tasks.append(task)
        
        logger.info("Odds API polling started")
    
    # Market tracking
    
    async def track_market(
        self,
        kalshi_ticker: str,
        game_id: Optional[str] = None,
        channels: Optional[List[KalshiChannel]] = None
    ):
        """
        Track a market across all sources.
        
        Args:
            kalshi_ticker: Kalshi market ticker
            game_id: Optional game ID for odds correlation
            channels: Kalshi channels to subscribe
        """
        if len(self._tracked_markets) >= self.config.max_tracked_markets:
            logger.warning("Max tracked markets reached")
            return
        
        # Track market
        self._tracked_markets.add(kalshi_ticker)
        if game_id:
            self._game_mappings[kalshi_ticker] = game_id
        
        # Initialize market data
        if kalshi_ticker not in self._market_data:
            self._market_data[kalshi_ticker] = UnifiedMarketData(ticker=kalshi_ticker)
        
        # Subscribe to Kalshi
        if self.kalshi_ws and self.kalshi_ws.is_connected():
            await self.kalshi_ws.subscribe_market(kalshi_ticker, channels)
        
        logger.info(f"Tracking market: {kalshi_ticker}")
    
    async def untrack_market(self, kalshi_ticker: str):
        """Stop tracking a market."""
        if kalshi_ticker not in self._tracked_markets:
            return
        
        # Unsubscribe from Kalshi
        if self.kalshi_ws and self.kalshi_ws.is_connected():
            await self.kalshi_ws.unsubscribe_market(kalshi_ticker)
        
        # Remove tracking
        self._tracked_markets.discard(kalshi_ticker)
        self._game_mappings.pop(kalshi_ticker, None)
        
        logger.info(f"Untracked market: {kalshi_ticker}")
    
    # Kalshi event handlers
    
    async def _handle_kalshi_ticker(self, data: Dict[str, Any]):
        """Handle Kalshi ticker update."""
        ticker = data.get("market_ticker")
        if ticker not in self._tracked_markets:
            return
        
        # Update market data
        market = self._market_data.get(ticker)
        if not market:
            market = UnifiedMarketData(ticker=ticker)
            self._market_data[ticker] = market
        
        market.kalshi_yes_price = data.get("yes_price")
        market.kalshi_no_price = data.get("no_price")
        market.kalshi_yes_bid = data.get("yes_bid")
        market.kalshi_yes_ask = data.get("yes_ask")
        market.kalshi_no_bid = data.get("no_bid")
        market.kalshi_no_ask = data.get("no_ask")
        market.kalshi_volume = data.get("volume")
        market.kalshi_open_interest = data.get("open_interest")
        market.timestamp = datetime.utcnow()
        
        # Compute metrics
        market.compute_metrics()
        
        # Store history
        self._market_history[ticker].append(market)
        if len(self._market_history[ticker]) > 1000:
            self._market_history[ticker].pop(0)
        
        # Emit event
        await self._emit_event(EventType.PRICE_UPDATE, {
            "ticker": ticker,
            "data": market
        })
    
    async def _handle_kalshi_orderbook(self, data: Dict[str, Any]):
        """Handle Kalshi orderbook update."""
        ticker = data.get("market_ticker")
        
        await self._emit_event(EventType.ORDERBOOK_UPDATE, {
            "ticker": ticker,
            "data": data
        })
    
    async def _handle_kalshi_trade(self, data: Dict[str, Any]):
        """Handle Kalshi trade execution."""
        ticker = data.get("market_ticker")
        
        await self._emit_event(EventType.TRADE_EXECUTED, {
            "ticker": ticker,
            "data": data
        })
    
    async def _handle_kalshi_connected(self, data: Dict[str, Any]):
        """Handle Kalshi connection."""
        logger.info("Kalshi WebSocket connected, resubscribing to markets")
        
        # Resubscribe to tracked markets
        for ticker in self._tracked_markets:
            if self.kalshi_ws:
                await self.kalshi_ws.subscribe_market(ticker)
    
    async def _handle_kalshi_disconnected(self, data: Dict[str, Any]):
        """Handle Kalshi disconnection."""
        logger.warning("Kalshi WebSocket disconnected")
    
    # Odds polling
    
    async def _odds_polling_loop(self):
        """Poll Odds API for updates."""
        while True:
            try:
                await self._fetch_odds_updates()
                await asyncio.sleep(self.config.odds_poll_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Odds polling error: {e}")
                await asyncio.sleep(self.config.odds_poll_interval)
    
    async def _fetch_odds_updates(self):
        """Fetch latest odds for tracked games."""
        if not self.odds_api:
            return
        
        # Get unique game IDs
        game_ids = set(self._game_mappings.values())
        if not game_ids:
            return
        
        # Fetch NFL odds (could be expanded to other sports)
        try:
            odds_list = await self.odds_api.get_nfl_odds()
            
            for game in odds_list:
                game_id = game.id
                
                # Find corresponding Kalshi ticker
                kalshi_ticker = None
                for ticker, gid in self._game_mappings.items():
                    if gid == game_id:
                        kalshi_ticker = ticker
                        break
                
                if not kalshi_ticker:
                    continue
                
                # Update market data
                market = self._market_data.get(kalshi_ticker)
                if not market:
                    continue
                
                # Get consensus odds
                consensus = game.get_consensus_moneyline()
                market.odds_consensus_home = consensus.get(game.home_team)
                market.odds_consensus_away = consensus.get(game.away_team)
                
                # Get best odds
                best_odds = game.get_best_odds("h2h")
                if game.home_team in best_odds:
                    market.odds_best_home = best_odds[game.home_team]
                if game.away_team in best_odds:
                    market.odds_best_away = best_odds[game.away_team]
                
                # Get spread and total
                spread_odds = game.get_best_odds("spreads")
                if spread_odds:
                    first_team = list(spread_odds.keys())[0]
                    market.odds_spread = spread_odds[first_team].get("point")
                
                total_odds = game.get_best_odds("totals")
                if total_odds and "Over" in total_odds:
                    market.odds_total = total_odds["Over"].get("point")
                
                # Compute metrics
                market.compute_metrics()
                
                # Emit event
                await self._emit_event(EventType.ODDS_UPDATE, {
                    "ticker": kalshi_ticker,
                    "data": market
                })
                
                # Check for line movement
                await self._check_line_movement(kalshi_ticker, market)
                
        except Exception as e:
            logger.error(f"Error fetching odds: {e}")
    
    async def _check_line_movement(self, ticker: str, current: UnifiedMarketData):
        """Check for significant line movements."""
        history = self._market_history.get(ticker, [])
        if len(history) < 2:
            return
        
        # Get previous data
        previous = history[-2]
        
        # Check for significant changes
        if previous.odds_consensus_home and current.odds_consensus_home:
            change = abs(current.odds_consensus_home - previous.odds_consensus_home)
            if change > 10:  # 10 point line movement
                await self._emit_event(EventType.LINE_MOVEMENT, {
                    "ticker": ticker,
                    "previous": previous.odds_consensus_home,
                    "current": current.odds_consensus_home,
                    "change": change
                })
    
    # Correlation engine
    
    async def _correlation_loop(self):
        """Correlate data across sources."""
        while True:
            try:
                await self._correlate_markets()
                await asyncio.sleep(self.config.correlation_window)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Correlation error: {e}")
                await asyncio.sleep(self.config.correlation_window)
    
    async def _correlate_markets(self):
        """Correlate market data across sources."""
        for ticker, market in self._market_data.items():
            if not market.kalshi_yes_price or not market.odds_consensus_home:
                continue
            
            # Check for divergence
            if market.divergence_score and market.divergence_score > self.config.divergence_threshold:
                await self._emit_event(EventType.DIVERGENCE_DETECTED, {
                    "ticker": ticker,
                    "kalshi_price": market.kalshi_yes_price,
                    "odds_implied_prob": market.odds_implied_prob_home,
                    "divergence": market.divergence_score
                })
            
            # Check for arbitrage
            if market.arbitrage_exists:
                await self._emit_event(EventType.ARBITRAGE_OPPORTUNITY, {
                    "ticker": ticker,
                    "data": market
                })
    
    # Event handling
    
    def on(self, event: EventType, handler: Callable):
        """
        Register event handler.
        
        Args:
            event: Event type
            handler: Event handler function
        """
        self._event_handlers[event].append(handler)
    
    async def _emit_event(self, event: EventType, data: Any):
        """
        Emit event to registered handlers.
        
        Args:
            event: Event type
            data: Event data
        """
        for handler in self._event_handlers.get(event, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                logger.error(f"Event handler error: {e}")
    
    # Data access
    
    def get_market_data(self, ticker: str) -> Optional[UnifiedMarketData]:
        """Get current market data."""
        return self._market_data.get(ticker)
    
    def get_market_history(
        self,
        ticker: str,
        limit: Optional[int] = None
    ) -> List[UnifiedMarketData]:
        """Get market history."""
        history = self._market_history.get(ticker, [])
        if limit:
            return history[-limit:]
        return history
    
    def get_tracked_markets(self) -> List[str]:
        """Get list of tracked markets."""
        return list(self._tracked_markets)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get stream manager statistics."""
        stats = {
            "tracked_markets": len(self._tracked_markets),
            "markets_with_data": len(self._market_data),
            "total_history_points": sum(len(h) for h in self._market_history.values())
        }
        
        if self.kalshi_ws:
            stats["kalshi"] = self.kalshi_ws.get_stats()
        
        if self.odds_api:
            stats["odds_api"] = self.odds_api.get_usage_stats()
        
        return stats
    
    def calculate_volatility(self, ticker: str, window: int = 20) -> Optional[float]:
        """
        Calculate price volatility for a market.
        
        Args:
            ticker: Market ticker
            window: Number of data points to consider
            
        Returns:
            Volatility as standard deviation
        """
        history = self.get_market_history(ticker, limit=window)
        if len(history) < 2:
            return None
        
        prices = [h.kalshi_yes_price for h in history if h.kalshi_yes_price]
        if len(prices) < 2:
            return None
        
        return statistics.stdev(prices)