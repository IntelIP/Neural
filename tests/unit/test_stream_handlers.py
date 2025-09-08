"""
Unit tests for Neural SDK Stream Event Handlers.

Tests the MessageHandler classes and related event handling functionality.
"""

import pytest
from datetime import datetime

# Import the classes we're testing
from neural_sdk.streaming.handlers import (
    MessageHandler,
    OrderbookHandler,
    TickerHandler,
    TradeHandler
)


class TestMessageHandlers:
    """Test cases for message handlers."""
    
    @pytest.mark.asyncio
    async def test_orderbook_handler(self):
        """Test orderbook handler functionality."""
        handler = OrderbookHandler()
        
        # Test snapshot message
        snapshot_msg = {
            'type': 'orderbook',
            'ticker': 'TEST-MARKET',
            'update_type': 'snapshot',
            'bids': [[0.50, 100], [0.49, 200]],
            'asks': [[0.51, 150], [0.52, 300]]
        }
        
        result = await handler.handle(snapshot_msg)
        
        assert result['ticker'] == 'TEST-MARKET'
        assert len(result['bids']) == 2
        assert len(result['asks']) == 2
        assert result['spread'] == 0.01
        assert result['mid_price'] == 0.505
        
        # Test delta update
        delta_msg = {
            'type': 'orderbook',
            'ticker': 'TEST-MARKET',
            'update_type': 'delta',
            'bid_deltas': [[0.50, 0], [0.495, 150]],  # Remove 0.50, add 0.495
            'ask_deltas': []
        }
        
        result = await handler.handle(delta_msg)
        assert len(result['bids']) == 2  # Should have 0.495 and 0.49
        
    @pytest.mark.asyncio
    async def test_ticker_handler(self):
        """Test ticker handler functionality."""
        handler = TickerHandler(window_size=5)
        
        # Add price updates
        for i in range(5):
            msg = {
                'type': 'ticker',
                'ticker': 'TEST-MARKET',
                'price': 0.50 + i * 0.01,
                'volume': 100 * (i + 1)
            }
            result = await handler.handle(msg)
            
            assert result['ticker'] == 'TEST-MARKET'
            assert result['price'] == 0.50 + i * 0.01
            assert 'metrics' in result
            
        # Check final metrics
        metrics = handler.metrics['TEST-MARKET']
        assert metrics['current'] == 0.54
        assert metrics['high'] == 0.54
        assert metrics['low'] == 0.50
        assert metrics['change'] == 0.04
        
    @pytest.mark.asyncio
    async def test_trade_handler(self):
        """Test trade handler functionality."""
        handler = TradeHandler()
        
        # Add buy trade
        buy_trade = {
            'type': 'trade',
            'ticker': 'TEST-MARKET',
            'price': 0.50,
            'size': 100,
            'side': 'buy',
            'trade_id': 'trade1'
        }
        
        result = await handler.handle(buy_trade)
        assert result['ticker'] == 'TEST-MARKET'
        assert result['trade']['side'] == 'buy'
        assert result['stats']['buy_volume'] == 100
        assert result['stats']['total_volume'] == 100
        
        # Add sell trade
        sell_trade = {
            'type': 'trade',
            'ticker': 'TEST-MARKET',
            'price': 0.51,
            'size': 50,
            'side': 'sell',
            'trade_id': 'trade2'
        }
        
        result = await handler.handle(sell_trade)
        assert result['stats']['sell_volume'] == 50
        assert result['stats']['total_volume'] == 150
        assert result['stats']['vwap'] == (0.50 * 100 + 0.51 * 50) / 150
        
    def test_message_validation(self):
        """Test message validation."""
        handler = OrderbookHandler()
        
        # Valid message
        assert handler.validate_message({'type': 'orderbook'}) == True
        
        # Invalid messages
        assert handler.validate_message(None) == False
        assert handler.validate_message('not a dict') == False
        assert handler.validate_message({}) == False
        

class TestHandlerIntegration:
    """Test handler integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_multiple_handlers_same_market(self):
        """Test multiple handlers processing same market data."""
        orderbook_handler = OrderbookHandler()
        ticker_handler = TickerHandler()
        trade_handler = TradeHandler()
        
        market = 'TEST-MARKET'
        
        # Process orderbook update
        orderbook_msg = {
            'type': 'orderbook',
            'ticker': market,
            'update_type': 'snapshot',
            'bids': [[0.50, 100]],
            'asks': [[0.51, 100]]
        }
        ob_result = await orderbook_handler.handle(orderbook_msg)
        
        # Process ticker update
        ticker_msg = {
            'type': 'ticker',
            'ticker': market,
            'price': 0.505,
            'volume': 1000
        }
        ticker_result = await ticker_handler.handle(ticker_msg)
        
        # Process trade
        trade_msg = {
            'type': 'trade',
            'ticker': market,
            'price': 0.505,
            'size': 10,
            'side': 'buy'
        }
        trade_result = await trade_handler.handle(trade_msg)
        
        # Verify all handlers processed their data
        assert ob_result['mid_price'] == 0.505
        assert ticker_result['price'] == 0.505
        assert trade_result['trade']['price'] == 0.505
        

if __name__ == "__main__":
    pytest.main([__file__])