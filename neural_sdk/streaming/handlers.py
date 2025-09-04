"""
Neural SDK Stream Event Handlers

Event handler utilities and decorators for WebSocket streaming.
"""

import logging
from typing import Any, Callable, Dict, List
from enum import Enum

logger = logging.getLogger(__name__)


class StreamEventType(Enum):
    """Types of streaming events."""
    MARKET_DATA = "market_data"
    TRADE = "trade"
    CONNECTION = "connection"
    ERROR = "error"
    PRICE_ALERT = "price_alert"
    VOLUME_ALERT = "volume_alert"


class StreamEventHandler:
    """
    Event handler registry for streaming events.
    
    Provides a centralized way to manage and dispatch streaming events
    with filtering, priority, and error handling capabilities.
    """
    
    def __init__(self):
        """Initialize event handler registry."""
        self.handlers: Dict[StreamEventType, List[Dict]] = {
            event_type: [] for event_type in StreamEventType
        }
        self.global_handlers: List[Callable] = []
        
        # Event filtering
        self.market_filters: Dict[str, List[Callable]] = {}  # ticker -> handlers
        self.team_filters: Dict[str, List[Callable]] = {}    # team -> handlers
        
        # Statistics
        self.event_count: Dict[StreamEventType, int] = {
            event_type: 0 for event_type in StreamEventType
        }
    
    def register_handler(
        self,
        event_type: StreamEventType,
        handler: Callable,
        priority: int = 0,
        filters: Dict[str, Any] = None
    ) -> None:
        """
        Register an event handler.
        
        Args:
            event_type: Type of event to handle
            handler: Handler function
            priority: Handler priority (higher = earlier execution)
            filters: Event filters (e.g., {'ticker': 'NFL-*'})
        """
        handler_info = {
            'handler': handler,
            'priority': priority,
            'filters': filters or {},
            'name': handler.__name__
        }
        
        self.handlers[event_type].append(handler_info)
        
        # Sort by priority (descending)
        self.handlers[event_type].sort(key=lambda x: x['priority'], reverse=True)
        
        logger.info(f"Registered {event_type.value} handler: {handler.__name__}")
    
    def register_market_handler(self, ticker: str, handler: Callable) -> None:
        """
        Register handler for specific market ticker.
        
        Args:
            ticker: Market ticker (supports wildcards like 'NFL-*')
            handler: Handler function
        """
        if ticker not in self.market_filters:
            self.market_filters[ticker] = []
        
        self.market_filters[ticker].append(handler)
        logger.info(f"Registered market handler for {ticker}: {handler.__name__}")
    
    def register_team_handler(self, team_code: str, handler: Callable) -> None:
        """
        Register handler for specific team.
        
        Args:
            team_code: Team code (e.g., 'PHI', 'KC')
            handler: Handler function
        """
        team_code = team_code.upper()
        if team_code not in self.team_filters:
            self.team_filters[team_code] = []
        
        self.team_filters[team_code].append(handler)
        logger.info(f"Registered team handler for {team_code}: {handler.__name__}")
    
    def register_global_handler(self, handler: Callable) -> None:
        """
        Register global handler that receives all events.
        
        Args:
            handler: Handler function
        """
        self.global_handlers.append(handler)
        logger.info(f"Registered global handler: {handler.__name__}")
    
    async def dispatch_event(
        self,
        event_type: StreamEventType,
        event_data: Dict[str, Any]
    ) -> None:
        """
        Dispatch event to registered handlers.
        
        Args:
            event_type: Type of event
            event_data: Event data dictionary
        """
        self.event_count[event_type] += 1
        
        # Dispatch to global handlers first
        for handler in self.global_handlers:
            try:
                await self._call_handler(handler, event_type, event_data)
            except Exception as e:
                logger.error(f"Error in global handler {handler.__name__}: {e}")
        
        # Dispatch to specific event type handlers
        for handler_info in self.handlers[event_type]:
            if self._should_handle_event(handler_info, event_data):
                try:
                    await self._call_handler(
                        handler_info['handler'],
                        event_type,
                        event_data
                    )
                except Exception as e:
                    logger.error(
                        f"Error in {event_type.value} handler "
                        f"{handler_info['name']}: {e}"
                    )
        
        # Dispatch to filtered handlers
        await self._dispatch_filtered_events(event_type, event_data)
    
    async def _dispatch_filtered_events(
        self,
        event_type: StreamEventType,
        event_data: Dict[str, Any]
    ) -> None:
        """Dispatch events to filtered handlers."""
        
        # Market-specific handlers
        if event_type == StreamEventType.MARKET_DATA:
            ticker = event_data.get('market_ticker', '')
            
            for filter_ticker, handlers in self.market_filters.items():
                if self._matches_ticker_filter(ticker, filter_ticker):
                    for handler in handlers:
                        try:
                            await self._call_handler(handler, event_type, event_data)
                        except Exception as e:
                            logger.error(f"Error in market handler {handler.__name__}: {e}")
        
        # Team-specific handlers
        team_code = self._extract_team_from_event(event_data)
        if team_code:
            for filter_team, handlers in self.team_filters.items():
                if team_code.upper() == filter_team:
                    for handler in handlers:
                        try:
                            await self._call_handler(handler, event_type, event_data)
                        except Exception as e:
                            logger.error(f"Error in team handler {handler.__name__}: {e}")
    
    def _should_handle_event(self, handler_info: Dict, event_data: Dict) -> bool:
        """Check if handler should process this event based on filters."""
        filters = handler_info.get('filters', {})
        
        if not filters:
            return True
        
        # Check ticker filter
        if 'ticker' in filters:
            ticker = event_data.get('market_ticker', '')
            if not self._matches_ticker_filter(ticker, filters['ticker']):
                return False
        
        # Check team filter
        if 'team' in filters:
            team = self._extract_team_from_event(event_data)
            if not team or team.upper() != filters['team'].upper():
                return False
        
        # Check price range filter
        if 'price_range' in filters:
            yes_price = event_data.get('yes_price')
            if yes_price is not None:
                min_price, max_price = filters['price_range']
                if not (min_price <= yes_price <= max_price):
                    return False
        
        return True
    
    def _matches_ticker_filter(self, ticker: str, filter_pattern: str) -> bool:
        """Check if ticker matches filter pattern."""
        if filter_pattern.endswith('*'):
            prefix = filter_pattern[:-1]
            return ticker.startswith(prefix)
        return ticker == filter_pattern
    
    def _extract_team_from_event(self, event_data: Dict) -> str:
        """Extract team code from event data."""
        ticker = event_data.get('market_ticker', '')
        
        # Simple extraction - would need more sophisticated logic
        if 'NFL' in ticker:
            parts = ticker.split('-')
            if len(parts) >= 4:
                return parts[-1][:3]
        
        return ''
    
    async def _call_handler(
        self,
        handler: Callable,
        event_type: StreamEventType,
        event_data: Dict[str, Any]
    ) -> None:
        """Call handler function with proper error handling."""
        try:
            # Check if handler expects event_type parameter
            import inspect
            sig = inspect.signature(handler)
            
            if len(sig.parameters) == 1:
                # Handler expects only event_data
                await handler(event_data)
            elif len(sig.parameters) == 2:
                # Handler expects event_type and event_data
                await handler(event_type, event_data)
            else:
                # Default to event_data only
                await handler(event_data)
                
        except Exception as e:
            logger.error(f"Handler {handler.__name__} failed: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get event handler statistics."""
        return {
            'event_counts': dict(self.event_count),
            'handler_counts': {
                event_type.value: len(handlers)
                for event_type, handlers in self.handlers.items()
            },
            'market_filters': len(self.market_filters),
            'team_filters': len(self.team_filters),
            'global_handlers': len(self.global_handlers)
        }
    
    def clear_handlers(self, event_type: StreamEventType = None) -> None:
        """
        Clear registered handlers.
        
        Args:
            event_type: Specific event type to clear, or None for all
        """
        if event_type:
            self.handlers[event_type].clear()
            logger.info(f"Cleared {event_type.value} handlers")
        else:
            for handlers in self.handlers.values():
                handlers.clear()
            self.market_filters.clear()
            self.team_filters.clear()
            self.global_handlers.clear()
            logger.info("Cleared all handlers")


# Convenience decorators
def market_data_handler(
    ticker_filter: str = None,
    team_filter: str = None,
    price_range: tuple = None
):
    """
    Decorator for market data handlers with filtering.
    
    Args:
        ticker_filter: Ticker pattern to filter (e.g., 'NFL-*')
        team_filter: Team code to filter (e.g., 'PHI')
        price_range: Price range tuple (min, max)
    
    Example:
        ```python
        @market_data_handler(ticker_filter='NFL-*', price_range=(0.0, 0.3))
        async def handle_nfl_oversold(market_data):
            print(f"NFL oversold: {market_data['market_ticker']}")
        ```
    """
    def decorator(func):
        func._stream_handler_type = StreamEventType.MARKET_DATA
        func._stream_filters = {
            'ticker': ticker_filter,
            'team': team_filter,
            'price_range': price_range
        }
        # Remove None values
        func._stream_filters = {k: v for k, v in func._stream_filters.items() if v is not None}
        return func
    
    return decorator


def trade_handler(ticker_filter: str = None):
    """
    Decorator for trade execution handlers.
    
    Args:
        ticker_filter: Ticker pattern to filter
    """
    def decorator(func):
        func._stream_handler_type = StreamEventType.TRADE
        func._stream_filters = {'ticker': ticker_filter} if ticker_filter else {}
        return func
    
    return decorator


def price_alert_handler(threshold: float = 0.05):
    """
    Decorator for price alert handlers.
    
    Args:
        threshold: Minimum price change to trigger alert
    """
    def decorator(func):
        func._stream_handler_type = StreamEventType.PRICE_ALERT
        func._stream_filters = {'threshold': threshold}
        return func
    
    return decorator
