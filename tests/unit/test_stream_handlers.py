"""
Unit tests for Neural SDK Stream Event Handlers.

Tests the StreamEventHandler class and related event handling functionality.
"""

import pytest

# Import the classes we're testing
from neural_sdk.streaming.handlers import (
    StreamEventHandler,
    StreamEventType,
    market_data_handler,
    trade_handler,
    price_alert_handler
)


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
        return StreamEventHandler()
    
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
    
    @pytest.mark.asyncio
    async def test_dispatch_event_market_filters(self, handler_registry):
        """Test market-specific event filtering."""
        nfl_handler_calls = []
        
        async def nfl_handler(data):
            nfl_handler_calls.append(data)
        
        handler_registry.register_market_handler('NFL-*', nfl_handler)
        
        # Dispatch NFL market data
        nfl_data = {'market_ticker': 'NFL-TEST', 'yes_price': 0.55}
        await handler_registry.dispatch_event(StreamEventType.MARKET_DATA, nfl_data)
        
        # Dispatch non-NFL market data
        other_data = {'market_ticker': 'NBA-TEST', 'yes_price': 0.60}
        await handler_registry.dispatch_event(StreamEventType.MARKET_DATA, other_data)
        
        # Should only receive NFL data
        assert len(nfl_handler_calls) == 1
        assert nfl_handler_calls[0] == nfl_data
    
    @pytest.mark.asyncio
    async def test_dispatch_event_error_handling(self, handler_registry):
        """Test error handling during event dispatch."""
        success_handler_called = False
        
        async def failing_handler(data):
            raise Exception("Handler failed")
        
        async def success_handler(data):
            nonlocal success_handler_called
            success_handler_called = True
        
        handler_registry.register_handler(StreamEventType.MARKET_DATA, failing_handler)
        handler_registry.register_handler(StreamEventType.MARKET_DATA, success_handler)
        
        # Should not raise exception, but should continue to other handlers
        await handler_registry.dispatch_event(
            StreamEventType.MARKET_DATA,
            {'test': 'data'}
        )
        
        assert success_handler_called is True
    
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


class TestHandlerDecorators:
    """Test cases for handler decorators."""
    
    def test_market_data_handler_decorator(self):
        """Test market_data_handler decorator."""
        @market_data_handler(ticker_filter='NFL-*', price_range=(0.0, 0.5))
        async def test_handler(data):
            pass
        
        assert hasattr(test_handler, '_stream_handler_type')
        assert test_handler._stream_handler_type == StreamEventType.MARKET_DATA
        assert hasattr(test_handler, '_stream_filters')
        assert test_handler._stream_filters['ticker'] == 'NFL-*'
        assert test_handler._stream_filters['price_range'] == (0.0, 0.5)
    
    def test_trade_handler_decorator(self):
        """Test trade_handler decorator."""
        @trade_handler(ticker_filter='NFL-*')
        async def test_handler(data):
            pass
        
        assert test_handler._stream_handler_type == StreamEventType.TRADE
        assert test_handler._stream_filters['ticker'] == 'NFL-*'
    
    def test_price_alert_handler_decorator(self):
        """Test price_alert_handler decorator."""
        @price_alert_handler(threshold=0.1)
        async def test_handler(data):
            pass
        
        assert test_handler._stream_handler_type == StreamEventType.PRICE_ALERT
        assert test_handler._stream_filters['threshold'] == 0.1
    
    def test_decorator_with_no_filters(self):
        """Test decorators with no filters."""
        @market_data_handler()
        async def test_handler(data):
            pass
        
        assert test_handler._stream_handler_type == StreamEventType.MARKET_DATA
        assert test_handler._stream_filters == {}


if __name__ == "__main__":
    pytest.main([__file__])
