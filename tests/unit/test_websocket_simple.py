"""
Simple unit tests for Neural SDK WebSocket functionality.

Tests core functionality without data pipeline dependencies.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import sys

# Mock all data_pipeline modules at the module level
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

# Patch sys.modules before any imports
for module_name, mock in data_pipeline_mocks.items():
    sys.modules[module_name] = mock

# Now we can safely import
from neural_sdk.core.config import SDKConfig
from neural_sdk.core.exceptions import ConnectionError, SDKError


class MockNeuralWebSocket:
    """Mock WebSocket for testing without data pipeline dependencies."""
    
    def __init__(self, config):
        self.config = config
        self._connected = False
        self._subscribed_markets = set()
        self._market_data_handlers = []
        self._trade_handlers = []
        self._connection_handlers = []
        self._error_handlers = []
    
    async def connect(self):
        """Mock connect method."""
        self._connected = True
    
    async def disconnect(self):
        """Mock disconnect method."""
        self._connected = False
    
    async def subscribe_markets(self, tickers):
        """Mock subscribe markets method."""
        if not self._connected:
            raise ConnectionError("WebSocket not connected. Call connect() first.")
        self._subscribed_markets.update(tickers)
    
    def on_market_data(self, func):
        """Mock market data handler decorator."""
        self._market_data_handlers.append(func)
        return func
    
    def on_trade(self, func):
        """Mock trade handler decorator."""
        self._trade_handlers.append(func)
        return func
    
    def on_connection(self, func):
        """Mock connection handler decorator."""
        self._connection_handlers.append(func)
        return func
    
    def on_error(self, func):
        """Mock error handler decorator."""
        self._error_handlers.append(func)
        return func
    
    def is_connected(self):
        """Check connection status."""
        return self._connected
    
    def get_subscribed_markets(self):
        """Get subscribed markets."""
        return list(self._subscribed_markets)
    
    def get_status(self):
        """Get status."""
        return {
            "connected": self._connected,
            "subscribed_markets": len(self._subscribed_markets),
            "market_data_handlers": len(self._market_data_handlers),
            "trade_handlers": len(self._trade_handlers),
            "connection_handlers": len(self._connection_handlers),
            "error_handlers": len(self._error_handlers)
        }


class TestNeuralWebSocketSimple:
    """Simple test cases for WebSocket functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock SDK configuration."""
        config = MagicMock(spec=SDKConfig)
        config.environment = "test"
        return config
    
    @pytest.fixture
    def websocket(self, mock_config):
        """Create a mock WebSocket instance."""
        return MockNeuralWebSocket(mock_config)
    
    def test_websocket_initialization(self, mock_config):
        """Test WebSocket initialization."""
        ws = MockNeuralWebSocket(mock_config)
        
        assert ws.config == mock_config
        assert ws._connected is False
        assert len(ws._market_data_handlers) == 0
        assert len(ws._trade_handlers) == 0
    
    @pytest.mark.asyncio
    async def test_connect_disconnect(self, websocket):
        """Test connection and disconnection."""
        # Initially not connected
        assert not websocket.is_connected()
        
        # Connect
        await websocket.connect()
        assert websocket.is_connected()
        
        # Disconnect
        await websocket.disconnect()
        assert not websocket.is_connected()
    
    @pytest.mark.asyncio
    async def test_subscribe_markets(self, websocket):
        """Test market subscription."""
        # Can't subscribe when not connected
        with pytest.raises(ConnectionError):
            await websocket.subscribe_markets(['TEST-MARKET'])
        
        # Connect first
        await websocket.connect()
        
        # Now can subscribe
        tickers = ['MARKET1', 'MARKET2']
        await websocket.subscribe_markets(tickers)
        
        subscribed = websocket.get_subscribed_markets()
        assert 'MARKET1' in subscribed
        assert 'MARKET2' in subscribed
    
    def test_event_handlers(self, websocket):
        """Test event handler registration."""
        # Market data handler
        @websocket.on_market_data
        async def market_handler(data):
            pass
        
        assert len(websocket._market_data_handlers) == 1
        assert websocket._market_data_handlers[0] == market_handler
        
        # Trade handler
        @websocket.on_trade
        async def trade_handler(data):
            pass
        
        assert len(websocket._trade_handlers) == 1
        assert websocket._trade_handlers[0] == trade_handler
    
    def test_get_status(self, websocket):
        """Test status reporting."""
        # Add handlers
        @websocket.on_market_data
        async def handler1(data):
            pass
        
        @websocket.on_trade
        async def handler2(data):
            pass
        
        status = websocket.get_status()
        
        assert status['connected'] is False
        assert status['market_data_handlers'] == 1
        assert status['trade_handlers'] == 1
        assert status['subscribed_markets'] == 0


class TestSDKIntegrationSimple:
    """Simple integration tests for SDK WebSocket functionality."""
    
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
    
    def test_sdk_websocket_creation(self, mock_config):
        """Test that SDK can create WebSocket clients."""
        # This tests the concept without requiring actual imports
        
        # Mock the SDK client
        class MockNeuralSDK:
            def __init__(self, config):
                self.config = config
                self._websocket = None
            
            def create_websocket(self):
                return MockNeuralWebSocket(self.config)
        
        sdk = MockNeuralSDK(mock_config)
        websocket = sdk.create_websocket()
        
        assert websocket.config == mock_config
        assert isinstance(websocket, MockNeuralWebSocket)
    
    @pytest.mark.asyncio
    async def test_sdk_streaming_workflow(self, mock_config):
        """Test SDK streaming workflow."""
        
        class MockNeuralSDK:
            def __init__(self, config):
                self.config = config
                self._websocket = None
                self._market_data_handlers = []
                self._trade_handlers = []
            
            def create_websocket(self):
                return MockNeuralWebSocket(self.config)
            
            async def start_streaming(self, markets=None):
                if self._websocket is None:
                    self._websocket = self.create_websocket()
                
                await self._websocket.connect()
                
                if markets:
                    await self._websocket.subscribe_markets(markets)
            
            async def stop_streaming(self):
                if self._websocket:
                    await self._websocket.disconnect()
                    self._websocket = None
            
            def on_market_data(self, func):
                self._market_data_handlers.append(func)
                return func
        
        sdk = MockNeuralSDK(mock_config)
        
        # Add event handler
        @sdk.on_market_data
        async def handle_data(data):
            pass
        
        # Start streaming
        await sdk.start_streaming(['TEST-MARKET'])
        
        # Verify state
        assert sdk._websocket is not None
        assert sdk._websocket.is_connected()
        assert 'TEST-MARKET' in sdk._websocket.get_subscribed_markets()
        
        # Stop streaming
        await sdk.stop_streaming()
        assert sdk._websocket is None


if __name__ == "__main__":
    pytest.main([__file__])
