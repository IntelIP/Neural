"""
Unit tests for Neural SDK WebSocket functionality.

Tests the NeuralWebSocket class and related streaming functionality
without requiring actual network connections.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Mock data_pipeline imports before importing neural_sdk
with patch.dict('sys.modules', {
    'data_pipeline': MagicMock(),
    'data_pipeline.streaming': MagicMock(),
    'data_pipeline.streaming.websocket': MagicMock(),
    'data_pipeline.streaming.handlers': MagicMock(),
    'data_pipeline.data_sources': MagicMock(),
    'data_pipeline.data_sources.kalshi': MagicMock(),
    'data_pipeline.data_sources.kalshi.market_discovery': MagicMock(),
    'data_pipeline.sports_config': MagicMock(),
}):
    # Import the classes we're testing
    from neural_sdk.streaming.websocket import NeuralWebSocket
    from neural_sdk.core.config import SDKConfig
    from neural_sdk.core.exceptions import ConnectionError


class TestNeuralWebSocket:
    """Test cases for NeuralWebSocket class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock SDK configuration."""
        config = MagicMock(spec=SDKConfig)
        config.environment = "test"
        return config
    
    @pytest.fixture
    def websocket(self, mock_config):
        """Create a NeuralWebSocket instance for testing."""
        with patch('neural_sdk.streaming.websocket.KalshiWebSocket'), \
             patch('neural_sdk.streaming.websocket.KalshiMarketDiscovery'):
            return NeuralWebSocket(mock_config)
    
    def test_websocket_initialization(self, mock_config):
        """Test WebSocket initialization."""
        with patch('neural_sdk.streaming.websocket.KalshiWebSocket'), \
             patch('neural_sdk.streaming.websocket.KalshiMarketDiscovery'):
            
            ws = NeuralWebSocket(mock_config)
            
            assert ws.config == mock_config
            assert ws._connected is False
            assert len(ws._market_data_handlers) == 0
            assert len(ws._trade_handlers) == 0
            assert len(ws._connection_handlers) == 0
            assert len(ws._error_handlers) == 0
    
    @pytest.mark.asyncio
    async def test_connect_success(self, websocket):
        """Test successful WebSocket connection."""
        # Mock the underlying WebSocket client
        mock_ws_client = AsyncMock()
        mock_market_discovery = MagicMock()
        
        with patch('neural_sdk.streaming.websocket.KalshiWebSocket', return_value=mock_ws_client), \
             patch('neural_sdk.streaming.websocket.KalshiMarketDiscovery', return_value=mock_market_discovery):
            
            await websocket.connect()
            
            assert websocket._connected is True
            assert websocket._ws_client == mock_ws_client
            assert websocket._market_discovery == mock_market_discovery
            mock_ws_client.connect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_connect_failure(self, websocket):
        """Test WebSocket connection failure."""
        mock_ws_client = AsyncMock()
        mock_ws_client.connect.side_effect = Exception("Connection failed")
        
        with patch('neural_sdk.streaming.websocket.KalshiWebSocket', return_value=mock_ws_client), \
             patch('neural_sdk.streaming.websocket.KalshiMarketDiscovery'):
            
            with pytest.raises(ConnectionError, match="WebSocket connection failed"):
                await websocket.connect()
            
            assert websocket._connected is False
    
    @pytest.mark.asyncio
    async def test_disconnect(self, websocket):
        """Test WebSocket disconnection."""
        # Set up connected state
        mock_ws_client = AsyncMock()
        websocket._ws_client = mock_ws_client
        websocket._connected = True
        
        await websocket.disconnect()
        
        assert websocket._connected is False
        mock_ws_client.disconnect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_subscribe_markets_success(self, websocket):
        """Test successful market subscription."""
        # Set up connected state
        mock_ws_client = AsyncMock()
        websocket._ws_client = mock_ws_client
        websocket._connected = True
        
        tickers = ["KXNFLGAME-TEST1", "KXNFLGAME-TEST2"]
        
        await websocket.subscribe_markets(tickers)
        
        mock_ws_client.subscribe_markets.assert_called_once_with(tickers)
        assert websocket._subscribed_markets == set(tickers)
    
    @pytest.mark.asyncio
    async def test_subscribe_markets_not_connected(self, websocket):
        """Test market subscription when not connected."""
        tickers = ["KXNFLGAME-TEST1"]
        
        with pytest.raises(ConnectionError, match="WebSocket not connected"):
            await websocket.subscribe_markets(tickers)
    
    @pytest.mark.asyncio
    async def test_subscribe_nfl_game(self, websocket):
        """Test NFL game subscription."""
        # Set up connected state and mock market discovery
        mock_ws_client = AsyncMock()
        mock_market_discovery = AsyncMock()
        
        # Mock discovered markets
        mock_market1 = MagicMock()
        mock_market1.ticker = "KXNFLGAME-25SEP04DALPHI-PHI"
        mock_market2 = MagicMock()
        mock_market2.ticker = "KXNFLGAME-25SEP04DALPHI-SPREAD"
        
        mock_market_discovery.discover_nfl_markets.return_value = [mock_market1, mock_market2]
        
        websocket._ws_client = mock_ws_client
        websocket._market_discovery = mock_market_discovery
        websocket._connected = True
        
        game_id = "25SEP04DALPHI"
        await websocket.subscribe_nfl_game(game_id)
        
        # Should subscribe to markets containing the game ID
        expected_tickers = ["KXNFLGAME-25SEP04DALPHI-PHI", "KXNFLGAME-25SEP04DALPHI-SPREAD"]
        mock_ws_client.subscribe_markets.assert_called_once_with(expected_tickers)
    
    @pytest.mark.asyncio
    async def test_subscribe_nfl_team(self, websocket):
        """Test NFL team subscription."""
        # Set up connected state and mock market discovery
        mock_ws_client = AsyncMock()
        mock_market_discovery = AsyncMock()
        
        # Mock team markets
        mock_market1 = MagicMock()
        mock_market1.ticker = "KXNFLGAME-PHI-WINNER"
        mock_market2 = MagicMock()
        mock_market2.ticker = "KXNFLGAME-PHI-SPREAD"
        
        mock_market_discovery.find_team_markets.return_value = [mock_market1, mock_market2]
        
        websocket._ws_client = mock_ws_client
        websocket._market_discovery = mock_market_discovery
        websocket._connected = True
        
        team_code = "PHI"
        await websocket.subscribe_nfl_team(team_code)
        
        # Should call find_team_markets and subscribe to results
        from neural_sdk.data_pipeline.sports_config import Sport
        mock_market_discovery.find_team_markets.assert_called_once_with(Sport.NFL, team_code)
        
        expected_tickers = ["KXNFLGAME-PHI-WINNER", "KXNFLGAME-PHI-SPREAD"]
        mock_ws_client.subscribe_markets.assert_called_once_with(expected_tickers)
    
    def test_event_handler_registration(self, websocket):
        """Test event handler registration decorators."""
        # Test market data handler
        @websocket.on_market_data
        async def test_market_handler(data):
            pass
        
        assert len(websocket._market_data_handlers) == 1
        assert websocket._market_data_handlers[0] == test_market_handler
        
        # Test trade handler
        @websocket.on_trade
        async def test_trade_handler(data):
            pass
        
        assert len(websocket._trade_handlers) == 1
        assert websocket._trade_handlers[0] == test_trade_handler
        
        # Test connection handler
        @websocket.on_connection
        async def test_connection_handler(data):
            pass
        
        assert len(websocket._connection_handlers) == 1
        assert websocket._connection_handlers[0] == test_connection_handler
        
        # Test error handler
        @websocket.on_error
        async def test_error_handler(data):
            pass
        
        assert len(websocket._error_handlers) == 1
        assert websocket._error_handlers[0] == test_error_handler
    
    @pytest.mark.asyncio
    async def test_market_data_handler_execution(self, websocket):
        """Test that market data handlers are called correctly."""
        handler_called = False
        received_data = None
        
        @websocket.on_market_data
        async def test_handler(data):
            nonlocal handler_called, received_data
            handler_called = True
            received_data = data
        
        # Simulate market data
        test_data = {
            'market_ticker': 'TEST-MARKET',
            'yes_price': 0.55,
            'volume': 100
        }
        
        await websocket._handle_market_data(test_data)
        
        assert handler_called is True
        assert received_data == test_data
    
    @pytest.mark.asyncio
    async def test_trade_handler_execution(self, websocket):
        """Test that trade handlers are called correctly."""
        handler_called = False
        received_data = None
        
        @websocket.on_trade
        async def test_handler(data):
            nonlocal handler_called, received_data
            handler_called = True
            received_data = data
        
        # Simulate trade data
        test_data = {
            'market_ticker': 'TEST-MARKET',
            'size': 50,
            'price': 0.60
        }
        
        await websocket._handle_trade_data(test_data)
        
        assert handler_called is True
        assert received_data == test_data
    
    @pytest.mark.asyncio
    async def test_error_handler_execution(self, websocket):
        """Test that error handlers are called correctly."""
        handler_called = False
        received_data = None
        
        @websocket.on_error
        async def test_handler(data):
            nonlocal handler_called, received_data
            handler_called = True
            received_data = data
        
        # Simulate error data
        test_data = {
            'code': 'TEST_ERROR',
            'message': 'Test error message'
        }
        
        await websocket._handle_error(test_data)
        
        assert handler_called is True
        assert received_data == test_data
    
    @pytest.mark.asyncio
    async def test_handler_error_handling(self, websocket):
        """Test that handler errors are caught and logged."""
        @websocket.on_market_data
        async def failing_handler(data):
            raise Exception("Handler failed")
        
        # This should not raise an exception
        with patch('neural_sdk.streaming.websocket.logger') as mock_logger:
            await websocket._handle_market_data({'test': 'data'})
            
            # Should log the error
            mock_logger.error.assert_called()
    
    def test_get_status(self, websocket):
        """Test status reporting."""
        # Add some handlers
        @websocket.on_market_data
        async def test_handler(data):
            pass
        
        # Set up some state
        websocket._connected = True
        websocket._subscribed_markets = {"MARKET1", "MARKET2"}
        
        status = websocket.get_status()
        
        expected_status = {
            "connected": True,
            "subscribed_markets": 2,
            "market_data_handlers": 1,
            "trade_handlers": 0,
            "connection_handlers": 0,
            "error_handlers": 0
        }
        
        assert status == expected_status
    
    def test_is_connected(self, websocket):
        """Test connection status check."""
        assert websocket.is_connected() is False
        
        websocket._connected = True
        assert websocket.is_connected() is True
    
    def test_get_subscribed_markets(self, websocket):
        """Test getting subscribed markets list."""
        websocket._subscribed_markets = {"MARKET1", "MARKET2", "MARKET3"}
        
        markets = websocket.get_subscribed_markets()
        
        assert len(markets) == 3
        assert "MARKET1" in markets
        assert "MARKET2" in markets
        assert "MARKET3" in markets
    
    @pytest.mark.asyncio
    async def test_run_forever_not_connected(self, websocket):
        """Test run_forever when not connected."""
        with pytest.raises(ConnectionError, match="WebSocket not connected"):
            await websocket.run_forever()
    
    @pytest.mark.asyncio
    async def test_run_forever_with_interruption(self, websocket):
        """Test run_forever with keyboard interrupt."""
        websocket._connected = True
        websocket._ws_client = AsyncMock()
        
        # Mock asyncio.sleep to raise KeyboardInterrupt after short delay
        with patch('asyncio.sleep', side_effect=KeyboardInterrupt):
            await websocket.run_forever()
        
        # Should have called disconnect
        websocket._ws_client.disconnect.assert_called_once()


class TestWebSocketIntegration:
    """Integration tests for WebSocket functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock SDK configuration."""
        config = MagicMock(spec=SDKConfig)
        config.environment = "test"
        return config
    
    @pytest.mark.asyncio
    async def test_full_websocket_lifecycle(self, mock_config):
        """Test complete WebSocket lifecycle."""
        with patch('neural_sdk.streaming.websocket.KalshiWebSocket') as MockKalshiWS, \
             patch('neural_sdk.streaming.websocket.KalshiMarketDiscovery') as MockDiscovery:
            
            # Set up mocks
            mock_ws_client = AsyncMock()
            mock_discovery = AsyncMock()
            MockKalshiWS.return_value = mock_ws_client
            MockDiscovery.return_value = mock_discovery
            
            # Create WebSocket
            websocket = NeuralWebSocket(mock_config)
            
            # Set up event handlers
            market_data_received = []
            
            @websocket.on_market_data
            async def handle_market_data(data):
                market_data_received.append(data)
            
            # Connect
            await websocket.connect()
            assert websocket.is_connected()
            
            # Subscribe to markets
            await websocket.subscribe_markets(['TEST-MARKET'])
            mock_ws_client.subscribe_markets.assert_called_with(['TEST-MARKET'])
            
            # Simulate receiving market data
            test_market_data = {
                'market_ticker': 'TEST-MARKET',
                'yes_price': 0.75,
                'volume': 200
            }
            await websocket._handle_market_data(test_market_data)
            
            # Verify handler was called
            assert len(market_data_received) == 1
            assert market_data_received[0] == test_market_data
            
            # Disconnect
            await websocket.disconnect()
            assert not websocket.is_connected()
            mock_ws_client.disconnect.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
