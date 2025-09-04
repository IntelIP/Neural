"""
Unit tests for Neural SDK client WebSocket integration.

Tests the WebSocket methods added to the main NeuralSDK client class.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Import the classes we're testing
from neural_sdk.core.client import NeuralSDK
from neural_sdk.core.config import SDKConfig


class TestNeuralSDKWebSocketIntegration:
    """Test cases for WebSocket integration in the main SDK client."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock SDK configuration."""
        config = MagicMock(spec=SDKConfig)
        config.environment = "test"
        config.api_key_id = "test_key"
        config.api_secret = "test_secret"
        config.trading = MagicMock()
        config.trading.enable_live_trading = False
        config.trading.default_quantity = 10
        return config
    
    @pytest.fixture
    def sdk(self, mock_config):
        """Create a NeuralSDK instance for testing."""
        return NeuralSDK(mock_config)
    
    def test_create_websocket(self, sdk):
        """Test creating a WebSocket client from SDK."""
        with patch('neural_sdk.core.client.NeuralWebSocket') as MockWebSocket:
            mock_websocket = MagicMock()
            MockWebSocket.return_value = mock_websocket
            
            websocket = sdk.create_websocket()
            
            assert websocket == mock_websocket
            MockWebSocket.assert_called_once_with(sdk.config)
    
    def test_create_nfl_stream(self, sdk):
        """Test creating an NFL stream from SDK."""
        with patch('neural_sdk.core.client.NFLMarketStream') as MockNFLStream:
            mock_nfl_stream = MagicMock()
            MockNFLStream.return_value = mock_nfl_stream
            
            nfl_stream = sdk.create_nfl_stream()
            
            assert nfl_stream == mock_nfl_stream
            MockNFLStream.assert_called_once_with(sdk.config)
    
    @pytest.mark.asyncio
    async def test_start_streaming_new_websocket(self, sdk):
        """Test starting streaming with a new WebSocket."""
        with patch('neural_sdk.core.client.NeuralWebSocket') as MockWebSocket:
            mock_websocket = AsyncMock()
            MockWebSocket.return_value = mock_websocket
            
            markets = ['NFL-TEST1', 'NFL-TEST2']
            await sdk.start_streaming(markets)
            
            # Should create new WebSocket
            assert sdk._websocket == mock_websocket
            MockWebSocket.assert_called_once_with(sdk.config)
            
            # Should connect and subscribe
            mock_websocket.connect.assert_called_once()
            mock_websocket.subscribe_markets.assert_called_once_with(markets)
    
    @pytest.mark.asyncio
    async def test_start_streaming_existing_websocket(self, sdk):
        """Test starting streaming with existing WebSocket."""
        # Set up existing WebSocket
        mock_websocket = AsyncMock()
        sdk._websocket = mock_websocket
        
        markets = ['NFL-TEST1']
        await sdk.start_streaming(markets)
        
        # Should use existing WebSocket
        mock_websocket.connect.assert_called_once()
        mock_websocket.subscribe_markets.assert_called_once_with(markets)
    
    @pytest.mark.asyncio
    async def test_start_streaming_no_markets(self, sdk):
        """Test starting streaming without specific markets."""
        with patch('neural_sdk.core.client.NeuralWebSocket') as MockWebSocket:
            mock_websocket = AsyncMock()
            MockWebSocket.return_value = mock_websocket
            
            await sdk.start_streaming()
            
            # Should connect but not subscribe to any markets
            mock_websocket.connect.assert_called_once()
            mock_websocket.subscribe_markets.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_start_streaming_with_handlers(self, sdk):
        """Test that SDK handlers are connected to WebSocket."""
        # Add handlers to SDK
        market_handler = AsyncMock()
        trade_handler = AsyncMock()
        
        sdk._market_data_handlers.append(market_handler)
        sdk._trade_handlers.append(trade_handler)
        
        with patch('neural_sdk.core.client.NeuralWebSocket') as MockWebSocket:
            mock_websocket = AsyncMock()
            MockWebSocket.return_value = mock_websocket
            
            await sdk.start_streaming()
            
            # Should register handlers with WebSocket
            mock_websocket.on_market_data.assert_called_with(market_handler)
            mock_websocket.on_trade.assert_called_with(trade_handler)
    
    @pytest.mark.asyncio
    async def test_stop_streaming(self, sdk):
        """Test stopping streaming."""
        # Set up WebSocket
        mock_websocket = AsyncMock()
        sdk._websocket = mock_websocket
        
        await sdk.stop_streaming()
        
        # Should disconnect and clear WebSocket
        mock_websocket.disconnect.assert_called_once()
        assert sdk._websocket is None
    
    @pytest.mark.asyncio
    async def test_stop_streaming_no_websocket(self, sdk):
        """Test stopping streaming when no WebSocket exists."""
        # Should not raise an error
        await sdk.stop_streaming()
        assert sdk._websocket is None
    
    def test_on_market_update_decorator(self, sdk):
        """Test the on_market_update decorator (alias for on_market_data)."""
        @sdk.on_market_update
        async def test_handler(data):
            pass
        
        # Should add handler to market data handlers
        assert len(sdk._market_data_handlers) == 1
        assert sdk._market_data_handlers[0] == test_handler
    
    def test_on_market_data_decorator(self, sdk):
        """Test the on_market_data decorator."""
        @sdk.on_market_data
        async def test_handler(data):
            pass
        
        # Should add handler to market data handlers
        assert len(sdk._market_data_handlers) == 1
        assert sdk._market_data_handlers[0] == test_handler
    
    def test_on_trade_decorator(self, sdk):
        """Test the on_trade decorator."""
        @sdk.on_trade
        async def test_handler(data):
            pass
        
        # Should add handler to trade handlers
        assert len(sdk._trade_handlers) == 1
        assert sdk._trade_handlers[0] == test_handler


class TestSDKWebSocketIntegration:
    """Integration tests for SDK WebSocket functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock SDK configuration."""
        config = MagicMock(spec=SDKConfig)
        config.environment = "test"
        config.api_key_id = "test_key"
        config.api_secret = "test_secret"
        config.trading = MagicMock()
        config.trading.enable_live_trading = False
        config.trading.default_quantity = 10
        return config
    
    @pytest.mark.asyncio
    async def test_full_sdk_streaming_workflow(self, mock_config):
        """Test complete SDK streaming workflow."""
        with patch('neural_sdk.core.client.NeuralWebSocket') as MockWebSocket:
            mock_websocket = AsyncMock()
            MockWebSocket.return_value = mock_websocket
            
            # Create SDK
            sdk = NeuralSDK(mock_config)
            
            # Add event handlers
            market_data_received = []
            trade_data_received = []
            
            @sdk.on_market_data
            async def handle_market_data(data):
                market_data_received.append(data)
            
            @sdk.on_trade
            async def handle_trade_data(data):
                trade_data_received.append(data)
            
            # Start streaming
            markets = ['NFL-TEST']
            await sdk.start_streaming(markets)
            
            # Verify WebSocket setup
            assert sdk._websocket == mock_websocket
            mock_websocket.connect.assert_called_once()
            mock_websocket.subscribe_markets.assert_called_once_with(markets)
            
            # Verify handlers were registered
            mock_websocket.on_market_data.assert_called()
            mock_websocket.on_trade.assert_called()
            
            # Stop streaming
            await sdk.stop_streaming()
            mock_websocket.disconnect.assert_called_once()
            assert sdk._websocket is None
    
    @pytest.mark.asyncio
    async def test_sdk_websocket_creation_workflow(self, mock_config):
        """Test SDK WebSocket creation workflow."""
        with patch('neural_sdk.core.client.NeuralWebSocket') as MockWebSocket, \
             patch('neural_sdk.core.client.NFLMarketStream') as MockNFLStream:
            
            mock_websocket = MagicMock()
            mock_nfl_stream = MagicMock()
            MockWebSocket.return_value = mock_websocket
            MockNFLStream.return_value = mock_nfl_stream
            
            # Create SDK
            sdk = NeuralSDK(mock_config)
            
            # Create WebSocket
            websocket = sdk.create_websocket()
            assert websocket == mock_websocket
            MockWebSocket.assert_called_with(sdk.config)
            
            # Create NFL stream
            nfl_stream = sdk.create_nfl_stream()
            assert nfl_stream == mock_nfl_stream
            MockNFLStream.assert_called_with(sdk.config)
    
    @pytest.mark.asyncio
    async def test_sdk_handler_integration(self, mock_config):
        """Test integration between SDK handlers and WebSocket."""
        with patch('neural_sdk.core.client.NeuralWebSocket') as MockWebSocket:
            mock_websocket = AsyncMock()
            MockWebSocket.return_value = mock_websocket
            
            # Create SDK and add handlers before starting streaming
            sdk = NeuralSDK(mock_config)
            
            handler_calls = []
            
            @sdk.on_market_data
            async def market_handler(data):
                handler_calls.append(('market', data))
            
            @sdk.on_trade
            async def trade_handler(data):
                handler_calls.append(('trade', data))
            
            # Start streaming
            await sdk.start_streaming(['TEST-MARKET'])
            
            # Verify handlers were registered with WebSocket
            # The mock should have been called with the handler functions
            mock_websocket.on_market_data.assert_called_with(market_handler)
            mock_websocket.on_trade.assert_called_with(trade_handler)
            
            # Stop streaming
            await sdk.stop_streaming()


if __name__ == "__main__":
    pytest.main([__file__])
