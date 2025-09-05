"""
Simple unit tests for Neural SDK Stream Event Handlers.

Tests event handling functionality without external dependencies.
"""

import pytest
from enum import Enum


class StreamEventType(Enum):
    """Types of streaming events."""
    MARKET_DATA = "market_data"
    TRADE = "trade"
    CONNECTION = "connection"
    ERROR = "error"
    PRICE_ALERT = "price_alert"
    VOLUME_ALERT = "volume_alert"


class SimpleStreamEventHandler:
    """Simplified event handler for testing."""
    
    def __init__(self):
        """Initialize event handler registry."""
        self.handlers = {event_type: [] for event_type in StreamEventType}
        self.global_handlers = []
        self.market_filters = {}
        self.team_filters = {}
        self.event_count = {event_type: 0 for event_type in StreamEventType}
    
    def register_handler(self, event_type, handler, priority=0, filters=None):
        """Register an event handler."""
        handler_info = {
            'handler': handler,
            'priority': priority,
            'filters': filters or {},
            'name': handler.__name__
        }
        
        self.handlers[event_type].append(handler_info)
        self.handlers[event_type].sort(key=lambda x: x['priority'], reverse=True)
    
    def register_market_handler(self, ticker, handler):
        """Register handler for specific market ticker."""
        if ticker not in self.market_filters:
            self.market_filters[ticker] = []
        self.market_filters[ticker].append(handler)
    
    def register_team_handler(self, team_code, handler):
        """Register handler for specific team."""
        team_code = team_code.upper()
        if team_code not in self.team_filters:
            self.team_filters[team_code] = []
        self.team_filters[team_code].append(handler)
    
    def register_global_handler(self, handler):
        """Register global handler that receives all events."""
        self.global_handlers.append(handler)
    
    async def dispatch_event(self, event_type, event_data):
        """Dispatch event to registered handlers."""
        self.event_count[event_type] += 1
        
        # Call global handlers
        for handler in self.global_handlers:
            try:
                await handler(event_type, event_data)
            except Exception:
                pass  # Ignore errors for testing
        
        # Call specific handlers
        for handler_info in self.handlers[event_type]:
            if self._should_handle_event(handler_info, event_data):
                try:
                    await handler_info['handler'](event_data)
                except Exception:
                    pass  # Ignore errors for testing
    
    def _should_handle_event(self, handler_info, event_data):
        """Check if handler should process this event based on filters."""
        filters = handler_info.get('filters', {})
        
        if not filters:
            return True
        
        # Check ticker filter
        if 'ticker' in filters:
            ticker = event_data.get('market_ticker', '')
            if not self._matches_ticker_filter(ticker, filters['ticker']):
                return False
        
        # Check price range filter
        if 'price_range' in filters:
            yes_price = event_data.get('yes_price')
            if yes_price is not None:
                min_price, max_price = filters['price_range']
                if not (min_price <= yes_price <= max_price):
                    return False
        
        return True
    
    def _matches_ticker_filter(self, ticker, filter_pattern):
        """Check if ticker matches filter pattern."""
        if filter_pattern.endswith('*'):
            prefix = filter_pattern[:-1]
            return ticker.startswith(prefix)
        return ticker == filter_pattern
    
    def get_statistics(self):
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
    
    def clear_handlers(self, event_type=None):
        """Clear registered handlers."""
        if event_type:
            self.handlers[event_type].clear()
        else:
            for handlers in self.handlers.values():
                handlers.clear()
            self.market_filters.clear()
            self.team_filters.clear()
            self.global_handlers.clear()


class TestStreamEventType:
    """Test cases for StreamEventType enum."""
    
    def test_event_types_exist(self):
        """Test that all expected event types exist."""
        expected_types = [
            "MARKET_DATA",
            "TRADE", 
            "CONNECTION",
            "ERROR",
            "PRICE_ALERT",
            "VOLUME_ALERT"
        ]
        
        for event_type in expected_types:
            assert hasattr(StreamEventType, event_type)
            assert isinstance(getattr(StreamEventType, event_type), StreamEventType)


class TestStreamEventHandler:
    """Test cases for StreamEventHandler class."""
    
    @pytest.fixture
    def handler_registry(self):
        """Create a StreamEventHandler instance for testing."""
        return SimpleStreamEventHandler()
    
    def test_initialization(self, handler_registry):
        """Test StreamEventHandler initialization."""
        assert len(handler_registry.handlers) == len(StreamEventType)
        assert len(handler_registry.global_handlers) == 0
        assert len(handler_registry.market_filters) == 0
        assert len(handler_registry.team_filters) == 0
        
        # Check event counters
        for event_type in StreamEventType:
            assert handler_registry.event_count[event_type] == 0
    
    def test_register_handler(self, handler_registry):
        """Test registering event handlers."""
        async def test_handler(data):
            pass
        
        handler_registry.register_handler(
            StreamEventType.MARKET_DATA,
            test_handler,
            priority=5,
            filters={'ticker': 'NFL-*'}
        )
        
        handlers = handler_registry.handlers[StreamEventType.MARKET_DATA]
        assert len(handlers) == 1
        
        handler_info = handlers[0]
        assert handler_info['handler'] == test_handler
        assert handler_info['priority'] == 5
        assert handler_info['filters'] == {'ticker': 'NFL-*'}
        assert handler_info['name'] == 'test_handler'
    
    def test_handler_priority_sorting(self, handler_registry):
        """Test that handlers are sorted by priority."""
        async def low_priority(data):
            pass
        
        async def high_priority(data):
            pass
        
        async def medium_priority(data):
            pass
        
        # Register in random order
        handler_registry.register_handler(StreamEventType.MARKET_DATA, medium_priority, priority=5)
        handler_registry.register_handler(StreamEventType.MARKET_DATA, high_priority, priority=10)
        handler_registry.register_handler(StreamEventType.MARKET_DATA, low_priority, priority=1)
        
        handlers = handler_registry.handlers[StreamEventType.MARKET_DATA]
        
        # Should be sorted by priority (descending)
        assert handlers[0]['handler'] == high_priority
        assert handlers[1]['handler'] == medium_priority
        assert handlers[2]['handler'] == low_priority
    
    def test_register_market_handler(self, handler_registry):
        """Test registering market-specific handlers."""
        async def nfl_handler(data):
            pass
        
        handler_registry.register_market_handler('NFL-*', nfl_handler)
        
        assert 'NFL-*' in handler_registry.market_filters
        assert nfl_handler in handler_registry.market_filters['NFL-*']
    
    def test_register_team_handler(self, handler_registry):
        """Test registering team-specific handlers."""
        async def eagles_handler(data):
            pass
        
        handler_registry.register_team_handler('PHI', eagles_handler)
        
        assert 'PHI' in handler_registry.team_filters
        assert eagles_handler in handler_registry.team_filters['PHI']
        
        # Test case insensitive
        handler_registry.register_team_handler('kc', eagles_handler)
        assert 'KC' in handler_registry.team_filters
    
    def test_register_global_handler(self, handler_registry):
        """Test registering global handlers."""
        async def global_handler(data):
            pass
        
        handler_registry.register_global_handler(global_handler)
        
        assert global_handler in handler_registry.global_handlers
    
    @pytest.mark.asyncio
    async def test_dispatch_event_basic(self, handler_registry):
        """Test basic event dispatching."""
        handler_called = False
        received_data = None
        
        async def test_handler(data):
            nonlocal handler_called, received_data
            handler_called = True
            received_data = data
        
        handler_registry.register_handler(StreamEventType.MARKET_DATA, test_handler)
        
        test_data = {'market_ticker': 'TEST-MARKET', 'yes_price': 0.55}
        await handler_registry.dispatch_event(StreamEventType.MARKET_DATA, test_data)
        
        assert handler_called is True
        assert received_data == test_data
        assert handler_registry.event_count[StreamEventType.MARKET_DATA] == 1
    
    @pytest.mark.asyncio
    async def test_dispatch_event_global_handlers(self, handler_registry):
        """Test that global handlers receive all events."""
        global_handler_calls = []
        
        async def global_handler(event_type, event_data):
            global_handler_calls.append((event_type, event_data))
        
        handler_registry.register_global_handler(global_handler)
        
        # Dispatch different event types
        await handler_registry.dispatch_event(StreamEventType.MARKET_DATA, {'test': 'market'})
        await handler_registry.dispatch_event(StreamEventType.TRADE, {'test': 'trade'})
        
        assert len(global_handler_calls) == 2
        assert global_handler_calls[0][0] == StreamEventType.MARKET_DATA
        assert global_handler_calls[1][0] == StreamEventType.TRADE
    
    @pytest.mark.asyncio
    async def test_dispatch_event_with_filters(self, handler_registry):
        """Test event dispatching with filters."""
        nfl_handler_calls = []
        all_handler_calls = []
        
        async def nfl_handler(data):
            nfl_handler_calls.append(data)
        
        async def all_handler(data):
            all_handler_calls.append(data)
        
        # Register handlers with different filters
        handler_registry.register_handler(
            StreamEventType.MARKET_DATA,
            nfl_handler,
            filters={'ticker': 'NFL-*'}
        )
        handler_registry.register_handler(
            StreamEventType.MARKET_DATA,
            all_handler
        )
        
        # Dispatch NFL market data
        nfl_data = {'market_ticker': 'NFL-TEST', 'yes_price': 0.55}
        await handler_registry.dispatch_event(StreamEventType.MARKET_DATA, nfl_data)
        
        # Dispatch non-NFL market data
        other_data = {'market_ticker': 'NBA-TEST', 'yes_price': 0.60}
        await handler_registry.dispatch_event(StreamEventType.MARKET_DATA, other_data)
        
        # NFL handler should only receive NFL data
        assert len(nfl_handler_calls) == 1
        assert nfl_handler_calls[0] == nfl_data
        
        # All handler should receive both
        assert len(all_handler_calls) == 2
    
    def test_matches_ticker_filter(self, handler_registry):
        """Test ticker filter matching."""
        # Test exact match
        assert handler_registry._matches_ticker_filter('NFL-TEST', 'NFL-TEST') is True
        assert handler_registry._matches_ticker_filter('NFL-TEST', 'NBA-TEST') is False
        
        # Test wildcard match
        assert handler_registry._matches_ticker_filter('NFL-TEST', 'NFL-*') is True
        assert handler_registry._matches_ticker_filter('NBA-TEST', 'NFL-*') is False
    
    def test_should_handle_event_filters(self, handler_registry):
        """Test event filtering logic."""
        # No filters - should handle
        handler_info = {'filters': {}}
        event_data = {'market_ticker': 'TEST'}
        assert handler_registry._should_handle_event(handler_info, event_data) is True
        
        # Ticker filter match
        handler_info = {'filters': {'ticker': 'TEST'}}
        event_data = {'market_ticker': 'TEST'}
        assert handler_registry._should_handle_event(handler_info, event_data) is True
        
        # Ticker filter no match
        handler_info = {'filters': {'ticker': 'NFL-*'}}
        event_data = {'market_ticker': 'NBA-TEST'}
        assert handler_registry._should_handle_event(handler_info, event_data) is False
        
        # Price range filter
        handler_info = {'filters': {'price_range': (0.0, 0.5)}}
        event_data = {'yes_price': 0.3}
        assert handler_registry._should_handle_event(handler_info, event_data) is True
        
        event_data = {'yes_price': 0.8}
        assert handler_registry._should_handle_event(handler_info, event_data) is False
    
    def test_get_statistics(self, handler_registry):
        """Test getting handler statistics."""
        # Add some handlers
        async def test_handler(data):
            pass
        
        handler_registry.register_handler(StreamEventType.MARKET_DATA, test_handler)
        handler_registry.register_handler(StreamEventType.TRADE, test_handler)
        handler_registry.register_market_handler('NFL-*', test_handler)
        handler_registry.register_team_handler('PHI', test_handler)
        handler_registry.register_global_handler(test_handler)
        
        # Increment some event counts
        handler_registry.event_count[StreamEventType.MARKET_DATA] = 5
        handler_registry.event_count[StreamEventType.TRADE] = 3
        
        stats = handler_registry.get_statistics()
        
        assert stats['event_counts'][StreamEventType.MARKET_DATA] == 5
        assert stats['event_counts'][StreamEventType.TRADE] == 3
        assert stats['handler_counts']['market_data'] == 1
        assert stats['handler_counts']['trade'] == 1
        assert stats['market_filters'] == 1
        assert stats['team_filters'] == 1
        assert stats['global_handlers'] == 1
    
    def test_clear_handlers(self, handler_registry):
        """Test clearing handlers."""
        async def test_handler(data):
            pass
        
        # Add handlers
        handler_registry.register_handler(StreamEventType.MARKET_DATA, test_handler)
        handler_registry.register_handler(StreamEventType.TRADE, test_handler)
        handler_registry.register_market_handler('NFL-*', test_handler)
        handler_registry.register_global_handler(test_handler)
        
        # Clear specific event type
        handler_registry.clear_handlers(StreamEventType.MARKET_DATA)
        assert len(handler_registry.handlers[StreamEventType.MARKET_DATA]) == 0
        assert len(handler_registry.handlers[StreamEventType.TRADE]) == 1
        
        # Clear all handlers
        handler_registry.clear_handlers()
        assert len(handler_registry.handlers[StreamEventType.TRADE]) == 0
        assert len(handler_registry.market_filters) == 0
        assert len(handler_registry.global_handlers) == 0


if __name__ == "__main__":
    pytest.main([__file__])
