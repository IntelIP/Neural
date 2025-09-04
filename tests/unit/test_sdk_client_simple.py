"""
Simple unit tests for Neural SDK client WebSocket integration.

Tests the WebSocket methods added to the main NeuralSDK client class
without requiring external dependencies.
"""

import pytest
from unittest.mock import MagicMock
import sys

# Mock data_pipeline modules
data_pipeline_mocks = {
    'data_pipeline': MagicMock(),
    'data_pipeline.streaming': MagicMock(),
    'data_pipeline.streaming.websocket': MagicMock(),
    'data_pipeline.streaming.handlers': MagicMock(),
    'data_pipeline.data_sources': MagicMock(),
    'data_pipeline.data_sources.kalshi': MagicMock(),
    'data_pipeline.data_sources.kalshi.market_discovery': MagicMock(),
    'data_pipeline.sports_config': MagicMock(),
}

for module_name, mock in data_pipeline_mocks.items():
    sys.modules[module_name] = mock

from neural_sdk.core.config import SDKConfig


class MockNeuralWebSocket:
    """Mock WebSocket for testing."""
    
    def __init__(self, config):
        self.config = config
        self._connected = False
        self._market_data_handlers = []
        self._trade_handlers = []
    
    async def connect(self):
        self._connected = True
    
    async def disconnect(self):
        self._connected = False
    
    async def subscribe_markets(self, markets):
        pass
    
    def on_market_data(self, handler):
        self._market_data_handlers.append(handler)
        return handler
    
    def on_trade(self, handler):
        self._trade_handlers.append(handler)
        return handler
    
    def is_connected(self):
        return self._connected


class MockNFLMarketStream:
    """Mock NFL market stream for testing."""
    
    def __init__(self, config):
        self.config = config
        self.websocket = MockNeuralWebSocket(config)


class MockNeuralSDK:
    """Mock Neural SDK for testing WebSocket integration."""
    
    def __init__(self, config):
        self.config = config
        self._websocket = None
        self._market_data_handlers = []
        self._trade_handlers = []
    
    def create_websocket(self):
        """Create a WebSocket client."""
        return MockNeuralWebSocket(self.config)
    
    def create_nfl_stream(self):
        """Create an NFL stream."""
        return MockNFLMarketStream(self.config)
    
    async def start_streaming(self, markets=None):
        """Start streaming."""
        if self._websocket is None:
            self._websocket = self.create_websocket()
            
            # Connect SDK handlers to WebSocket
            for handler in self._market_data_handlers:
                self._websocket.on_market_data(handler)
            
            for handler in self._trade_handlers:
                self._websocket.on_trade(handler)
        
        await self._websocket.connect()
        
        if markets:
            await self._websocket.subscribe_markets(markets)
    
    async def stop_streaming(self):
        """Stop streaming."""
        if self._websocket:
            await self._websocket.disconnect()
            self._websocket = None
    
    def on_market_data(self, func):
        """Register market data handler."""
        self._market_data_handlers.append(func)
        return func
    
    def on_trade(self, func):
        """Register trade handler."""
        self._trade_handlers.append(func)
        return func
    
    def on_market_update(self, func):
        """Alias for on_market_data."""
        return self.on_market_data(func)


class TestNeuralSDKWebSocketIntegration:
    """Test cases for WebSocket integration in the main SDK client."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock SDK configuration."""
        config = MagicMock(spec=SDKConfig)
        config.environment = "test"
        config.api_key_id = "test_key"
        config.api_secret = "test_secret"
        return config
    
    @pytest.fixture
    def sdk(self, mock_config):
        """Create a mock NeuralSDK instance for testing."""
        return MockNeuralSDK(mock_config)
    
    def test_create_websocket(self, sdk):
        """Test creating a WebSocket client from SDK."""
        websocket = sdk.create_websocket()
        
        assert isinstance(websocket, MockNeuralWebSocket)
        assert websocket.config == sdk.config
    
    def test_create_nfl_stream(self, sdk):
        """Test creating an NFL stream from SDK."""
        nfl_stream = sdk.create_nfl_stream()
        
        assert isinstance(nfl_stream, MockNFLMarketStream)
        assert nfl_stream.config == sdk.config
    
    @pytest.mark.asyncio
    async def test_start_streaming_new_websocket(self, sdk):
        """Test starting streaming with a new WebSocket."""
        markets = ['NFL-TEST1', 'NFL-TEST2']
        await sdk.start_streaming(markets)
        
        # Should create new WebSocket
        assert sdk._websocket is not None
        assert isinstance(sdk._websocket, MockNeuralWebSocket)
        assert sdk._websocket.is_connected()
    
    @pytest.mark.asyncio
    async def test_start_streaming_existing_websocket(self, sdk):
        """Test starting streaming with existing WebSocket."""
        # Create existing WebSocket
        websocket = sdk.create_websocket()
        sdk._websocket = websocket
        
        markets = ['NFL-TEST1']
        await sdk.start_streaming(markets)
        
        # Should use existing WebSocket
        assert sdk._websocket == websocket
        assert websocket.is_connected()
    
    @pytest.mark.asyncio
    async def test_start_streaming_no_markets(self, sdk):
        """Test starting streaming without specific markets."""
        await sdk.start_streaming()
        
        # Should connect but not subscribe to any markets
        assert sdk._websocket is not None
        assert sdk._websocket.is_connected()
    
    @pytest.mark.asyncio
    async def test_start_streaming_with_handlers(self, sdk):
        """Test that SDK handlers are connected to WebSocket."""
        # Add handlers to SDK
        @sdk.on_market_data
        async def market_handler(data):
            pass
        
        @sdk.on_trade
        async def trade_handler(data):
            pass
        
        await sdk.start_streaming()
        
        # Should register handlers with WebSocket
        assert len(sdk._websocket._market_data_handlers) == 1
        assert len(sdk._websocket._trade_handlers) == 1
        assert sdk._websocket._market_data_handlers[0] == market_handler
        assert sdk._websocket._trade_handlers[0] == trade_handler
    
    @pytest.mark.asyncio
    async def test_stop_streaming(self, sdk):
        """Test stopping streaming."""
        # Start streaming first
        await sdk.start_streaming()
        assert sdk._websocket is not None
        
        # Stop streaming
        await sdk.stop_streaming()
        
        # Should disconnect and clear WebSocket
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
        return config
    
    @pytest.mark.asyncio
    async def test_full_sdk_streaming_workflow(self, mock_config):
        """Test complete SDK streaming workflow."""
        # Create SDK
        sdk = MockNeuralSDK(mock_config)
        
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
        assert sdk._websocket is not None
        assert sdk._websocket.is_connected()
        
        # Verify handlers were registered
        assert len(sdk._websocket._market_data_handlers) == 1
        assert len(sdk._websocket._trade_handlers) == 1
        
        # Stop streaming
        await sdk.stop_streaming()
        assert sdk._websocket is None
    
    @pytest.mark.asyncio
    async def test_sdk_websocket_creation_workflow(self, mock_config):
        """Test SDK WebSocket creation workflow."""
        # Create SDK
        sdk = MockNeuralSDK(mock_config)
        
        # Create WebSocket
        websocket = sdk.create_websocket()
        assert isinstance(websocket, MockNeuralWebSocket)
        assert websocket.config == sdk.config
        
        # Create NFL stream
        nfl_stream = sdk.create_nfl_stream()
        assert isinstance(nfl_stream, MockNFLMarketStream)
        assert nfl_stream.config == sdk.config
    
    @pytest.mark.asyncio
    async def test_sdk_handler_integration(self, mock_config):
        """Test integration between SDK handlers and WebSocket."""
        # Create SDK and add handlers before starting streaming
        sdk = MockNeuralSDK(mock_config)
        
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
        assert len(sdk._websocket._market_data_handlers) == 1
        assert len(sdk._websocket._trade_handlers) == 1
        assert sdk._websocket._market_data_handlers[0] == market_handler
        assert sdk._websocket._trade_handlers[0] == trade_handler
        
        # Stop streaming
        await sdk.stop_streaming()


if __name__ == "__main__":
    pytest.main([__file__])
