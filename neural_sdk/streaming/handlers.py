"""
Message Handlers for WebSocket Streaming

Provides specialized handlers for different message types.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class MessageHandler(ABC):
    """Abstract base class for message handlers."""
    
    @abstractmethod
    async def handle(self, message: Dict[str, Any]) -> Any:
        """Handle incoming message."""
        pass
        
    def validate_message(self, message: Dict[str, Any]) -> bool:
        """Validate message format."""
        return isinstance(message, dict) and 'type' in message


class OrderbookHandler(MessageHandler):
    """
    Handler for orderbook updates.
    
    Maintains orderbook state and processes delta updates.
    """
    
    def __init__(self):
        """Initialize orderbook handler."""
        self.orderbooks = {}
        self.last_update = {}
        
    async def handle(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle orderbook update message.
        
        Args:
            message: Orderbook update message
            
        Returns:
            Processed orderbook state
        """
        if not self.validate_message(message):
            logger.warning(f"Invalid orderbook message: {message}")
            return {}
            
        ticker = message.get('ticker', 'UNKNOWN')
        update_type = message.get('update_type', 'snapshot')
        
        if ticker not in self.orderbooks:
            self.orderbooks[ticker] = {
                'bids': {},
                'asks': {},
                'last_update': None
            }
            
        orderbook = self.orderbooks[ticker]
        
        if update_type == 'snapshot':
            # Full orderbook snapshot
            orderbook['bids'] = self._process_levels(message.get('bids', []))
            orderbook['asks'] = self._process_levels(message.get('asks', []))
        elif update_type == 'delta':
            # Incremental update
            self._apply_deltas(orderbook['bids'], message.get('bid_deltas', []))
            self._apply_deltas(orderbook['asks'], message.get('ask_deltas', []))
            
        orderbook['last_update'] = datetime.now()
        self.last_update[ticker] = orderbook['last_update']
        
        return {
            'ticker': ticker,
            'bids': list(orderbook['bids'].items()),
            'asks': list(orderbook['asks'].items()),
            'spread': self._calculate_spread(orderbook),
            'mid_price': self._calculate_mid_price(orderbook)
        }
        
    def _process_levels(self, levels: List) -> Dict[float, float]:
        """Process price levels into orderbook format."""
        processed = {}
        for level in levels:
            if isinstance(level, (list, tuple)) and len(level) >= 2:
                price, size = float(level[0]), float(level[1])
                if size > 0:
                    processed[price] = size
        return processed
        
    def _apply_deltas(self, book_side: Dict[float, float], deltas: List):
        """Apply delta updates to orderbook side."""
        for delta in deltas:
            if isinstance(delta, (list, tuple)) and len(delta) >= 2:
                price, size = float(delta[0]), float(delta[1])
                if size > 0:
                    book_side[price] = size
                elif price in book_side:
                    del book_side[price]
                    
    def _calculate_spread(self, orderbook: Dict) -> Optional[float]:
        """Calculate bid-ask spread."""
        if orderbook['bids'] and orderbook['asks']:
            best_bid = max(orderbook['bids'].keys())
            best_ask = min(orderbook['asks'].keys())
            return best_ask - best_bid
        return None
        
    def _calculate_mid_price(self, orderbook: Dict) -> Optional[float]:
        """Calculate mid price."""
        if orderbook['bids'] and orderbook['asks']:
            best_bid = max(orderbook['bids'].keys())
            best_ask = min(orderbook['asks'].keys())
            return (best_bid + best_ask) / 2
        return None


class TickerHandler(MessageHandler):
    """
    Handler for ticker/price updates.
    
    Tracks price movements and calculates metrics.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize ticker handler.
        
        Args:
            window_size: Size of price history window
        """
        self.price_history = {}
        self.window_size = window_size
        self.metrics = {}
        
    async def handle(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle ticker update message.
        
        Args:
            message: Ticker update message
            
        Returns:
            Price metrics and analysis
        """
        if not self.validate_message(message):
            logger.warning(f"Invalid ticker message: {message}")
            return {}
            
        ticker = message.get('ticker', 'UNKNOWN')
        price = float(message.get('price', 0))
        volume = float(message.get('volume', 0))
        timestamp = message.get('timestamp', datetime.now())
        
        # Initialize history if needed
        if ticker not in self.price_history:
            self.price_history[ticker] = []
            self.metrics[ticker] = {}
            
        # Add to history
        self.price_history[ticker].append({
            'price': price,
            'volume': volume,
            'timestamp': timestamp
        })
        
        # Maintain window size
        if len(self.price_history[ticker]) > self.window_size:
            self.price_history[ticker].pop(0)
            
        # Calculate metrics
        metrics = self._calculate_metrics(ticker)
        self.metrics[ticker] = metrics
        
        return {
            'ticker': ticker,
            'price': price,
            'volume': volume,
            'metrics': metrics
        }
        
    def _calculate_metrics(self, ticker: str) -> Dict[str, Any]:
        """Calculate price metrics."""
        history = self.price_history[ticker]
        if not history:
            return {}
            
        prices = [h['price'] for h in history]
        
        metrics = {
            'current': prices[-1],
            'high': max(prices),
            'low': min(prices),
            'avg': sum(prices) / len(prices),
            'change': prices[-1] - prices[0] if len(prices) > 1 else 0,
            'change_pct': ((prices[-1] / prices[0]) - 1) * 100 if len(prices) > 1 and prices[0] != 0 else 0
        }
        
        # Add volatility if enough data
        if len(prices) > 10:
            avg = metrics['avg']
            variance = sum((p - avg) ** 2 for p in prices) / len(prices)
            metrics['volatility'] = variance ** 0.5
            
        return metrics


class TradeHandler(MessageHandler):
    """
    Handler for executed trades.
    
    Tracks trade flow and calculates trade metrics.
    """
    
    def __init__(self):
        """Initialize trade handler."""
        self.trades = {}
        self.trade_stats = {}
        
    async def handle(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle trade execution message.
        
        Args:
            message: Trade execution message
            
        Returns:
            Trade analysis and statistics
        """
        if not self.validate_message(message):
            logger.warning(f"Invalid trade message: {message}")
            return {}
            
        ticker = message.get('ticker', 'UNKNOWN')
        trade = {
            'price': float(message.get('price', 0)),
            'size': float(message.get('size', 0)),
            'side': message.get('side', 'unknown'),
            'timestamp': message.get('timestamp', datetime.now()),
            'trade_id': message.get('trade_id', '')
        }
        
        # Store trade
        if ticker not in self.trades:
            self.trades[ticker] = []
            self.trade_stats[ticker] = {
                'buy_volume': 0,
                'sell_volume': 0,
                'total_volume': 0,
                'vwap': 0,
                'trade_count': 0
            }
            
        self.trades[ticker].append(trade)
        
        # Update statistics
        stats = self.trade_stats[ticker]
        stats['trade_count'] += 1
        stats['total_volume'] += trade['size']
        
        if trade['side'] == 'buy':
            stats['buy_volume'] += trade['size']
        elif trade['side'] == 'sell':
            stats['sell_volume'] += trade['size']
            
        # Calculate VWAP
        total_value = sum(t['price'] * t['size'] for t in self.trades[ticker])
        total_size = sum(t['size'] for t in self.trades[ticker])
        if total_size > 0:
            stats['vwap'] = total_value / total_size
            
        # Calculate buy/sell pressure
        if stats['total_volume'] > 0:
            stats['buy_pressure'] = stats['buy_volume'] / stats['total_volume']
            stats['sell_pressure'] = stats['sell_volume'] / stats['total_volume']
            
        return {
            'ticker': ticker,
            'trade': trade,
            'stats': stats
        }