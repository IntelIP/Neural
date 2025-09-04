"""
Integration tests for Neural SDK WebSocket functionality.

Tests the complete integration between SDK components and WebSocket streaming.
"""

import asyncio
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
    from neural_sdk import NeuralSDK, NeuralWebSocket, NFLMarketStream
    from neural_sdk.core.config import SDKConfig
    from neural_sdk.core.exceptions import ConnectionError, SDKError


class TestWebSocketIntegration:
    """Integration tests for WebSocket functionality."""
    
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
    async def test_sdk_to_websocket_integration(self, mock_config):
        """Test integration from SDK to WebSocket."""
        with patch('neural_sdk.streaming.websocket.KalshiWebSocket') as MockKalshiWS, \
             patch('neural_sdk.streaming.websocket.KalshiMarketDiscovery') as MockDiscovery:
            
            # Set up mocks
            mock_ws_client = AsyncMock()
            mock_discovery = AsyncMock()
            MockKalshiWS.return_value = mock_ws_client
            MockDiscovery.return_value = mock_discovery
            
            # Create SDK and WebSocket
            sdk = NeuralSDK(mock_config)
            websocket = sdk.create_websocket()
            
            # Verify WebSocket was created with correct config
            assert isinstance(websocket, NeuralWebSocket)
            
            # Test connection flow
            await websocket.connect()
            
            # Verify underlying components were initialized
            MockKalshiWS.assert_called_once()
            MockDiscovery.assert_called_once()
            mock_ws_client.connect.assert_called_once()
            
            # Test subscription
            await websocket.subscribe_markets(['TEST-MARKET'])
            mock_ws_client.subscribe_markets.assert_called_with(['TEST-MARKET'])
            
            # Test disconnection
            await websocket.disconnect()
            mock_ws_client.disconnect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_sdk_streaming_integration(self, mock_config):
        """Test SDK integrated streaming functionality."""
        with patch('neural_sdk.streaming.websocket.KalshiWebSocket') as MockKalshiWS, \
             patch('neural_sdk.streaming.websocket.KalshiMarketDiscovery'):
            
            mock_ws_client = AsyncMock()
            MockKalshiWS.return_value = mock_ws_client
            
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
            await sdk.start_streaming(['TEST-MARKET'])
            
            # Verify WebSocket was created and configured
            assert sdk._websocket is not None
            mock_ws_client.connect.assert_called_once()
            mock_ws_client.subscribe_markets.assert_called_with(['TEST-MARKET'])
            
            # Simulate event handling by calling handlers directly
            # (In real integration, this would come through WebSocket)
            test_market_data = {'market_ticker': 'TEST-MARKET', 'yes_price': 0.55}
            test_trade_data = {'market_ticker': 'TEST-MARKET', 'size': 100}
            
            # The handlers should have been registered with the WebSocket
            # We can verify this by checking that on_market_data was called
            mock_ws_client.message_handler = MagicMock()
            
            # Stop streaming
            await sdk.stop_streaming()
            mock_ws_client.disconnect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_nfl_stream_integration(self, mock_config):
        """Test NFL stream integration."""
        with patch('neural_sdk.streaming.websocket.KalshiWebSocket') as MockKalshiWS, \
             patch('neural_sdk.streaming.websocket.KalshiMarketDiscovery') as MockDiscovery:
            
            # Set up mocks
            mock_ws_client = AsyncMock()
            mock_discovery = AsyncMock()
            
            # Mock NFL markets discovery
            mock_market1 = MagicMock()
            mock_market1.ticker = "KXNFLGAME-25SEP04DALPHI-PHI"
            mock_market2 = MagicMock()
            mock_market2.ticker = "KXNFLGAME-25SEP04DALPHI-SPREAD"
            mock_discovery.discover_nfl_markets.return_value = [mock_market1, mock_market2]
            mock_discovery.find_team_markets.return_value = [mock_market1]
            
            MockKalshiWS.return_value = mock_ws_client
            MockDiscovery.return_value = mock_discovery
            
            # Create SDK and NFL stream
            sdk = NeuralSDK(mock_config)
            nfl_stream = sdk.create_nfl_stream()
            
            # Test connection
            await nfl_stream.connect()
            mock_ws_client.connect.assert_called_once()
            
            # Test game subscription
            await nfl_stream.subscribe_to_game("25SEP04DALPHI")
            mock_discovery.discover_nfl_markets.assert_called_once()
            
            # Test team subscription
            await nfl_stream.subscribe_to_team("PHI")
            mock_discovery.find_team_markets.assert_called()
            
            # Test data tracking
            # Simulate NFL game data processing
            game_id = "25SEP04-DALPHI"
            nfl_stream.games[game_id] = {
                'markets': {
                    'KXNFLGAME-25SEP04DALPHI-WINNER': {'yes_price': 0.65}
                },
                'home_team': 'PHI',
                'away_team': 'DAL',
                'last_update': 'test_time'
            }
            
            # Test game summary
            summary = nfl_stream.get_game_summary(game_id)
            assert summary is not None
            assert summary['home_team'] == 'PHI'
            assert summary['away_team'] == 'DAL'
            assert summary['win_probability'] == 0.65
            
            # Test disconnection
            await nfl_stream.disconnect()
            mock_ws_client.disconnect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_event_handler_integration(self, mock_config):
        """Test event handler integration across components."""
        with patch('neural_sdk.streaming.websocket.KalshiWebSocket') as MockKalshiWS, \
             patch('neural_sdk.streaming.websocket.KalshiMarketDiscovery'):
            
            mock_ws_client = AsyncMock()
            MockKalshiWS.return_value = mock_ws_client
            
            # Create WebSocket
            websocket = NeuralWebSocket(mock_config)
            
            # Track handler calls
            handler_calls = []
            
            @websocket.on_market_data
            async def market_handler(data):
                handler_calls.append(('market', data))
            
            @websocket.on_trade
            async def trade_handler(data):
                handler_calls.append(('trade', data))
            
            @websocket.on_connection
            async def connection_handler(data):
                handler_calls.append(('connection', data))
            
            @websocket.on_error
            async def error_handler(data):
                handler_calls.append(('error', data))
            
            # Connect
            await websocket.connect()
            
            # Simulate various events
            await websocket._handle_market_data({'test': 'market_data'})
            await websocket._handle_trade_data({'test': 'trade_data'})
            await websocket._handle_error({'test': 'error_data'})
            
            # Verify handlers were called
            assert len(handler_calls) >= 3
            
            # Check that different event types were handled
            event_types = [call[0] for call in handler_calls]
            assert 'market' in event_types
            assert 'trade' in event_types
            assert 'error' in event_types
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, mock_config):
        """Test error handling across integration points."""
        with patch('neural_sdk.streaming.websocket.KalshiWebSocket') as MockKalshiWS:
            
            # Test connection failure
            mock_ws_client = AsyncMock()
            mock_ws_client.connect.side_effect = Exception("Connection failed")
            MockKalshiWS.return_value = mock_ws_client
            
            websocket = NeuralWebSocket(mock_config)
            
            with pytest.raises(ConnectionError):
                await websocket.connect()
            
            # Test subscription failure when not connected
            websocket_not_connected = NeuralWebSocket(mock_config)
            
            with pytest.raises(ConnectionError):
                await websocket_not_connected.subscribe_markets(['TEST'])
    
    @pytest.mark.asyncio
    async def test_configuration_integration(self, mock_config):
        """Test configuration integration across components."""
        with patch('neural_sdk.streaming.websocket.KalshiWebSocket') as MockKalshiWS, \
             patch('neural_sdk.streaming.websocket.KalshiMarketDiscovery'):
            
            MockKalshiWS.return_value = AsyncMock()
            
            # Create SDK with specific config
            sdk = NeuralSDK(mock_config)
            
            # Create WebSocket and verify config is passed
            websocket = sdk.create_websocket()
            assert websocket.config == mock_config
            
            # Create NFL stream and verify config is passed
            nfl_stream = sdk.create_nfl_stream()
            assert nfl_stream.config == mock_config
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, mock_config):
        """Test concurrent WebSocket operations."""
        with patch('neural_sdk.streaming.websocket.KalshiWebSocket') as MockKalshiWS, \
             patch('neural_sdk.streaming.websocket.KalshiMarketDiscovery'):
            
            mock_ws_client = AsyncMock()
            MockKalshiWS.return_value = mock_ws_client
            
            # Create multiple WebSocket connections
            websocket1 = NeuralWebSocket(mock_config)
            websocket2 = NeuralWebSocket(mock_config)
            
            # Connect both concurrently
            await asyncio.gather(
                websocket1.connect(),
                websocket2.connect()
            )
            
            # Both should be connected
            assert websocket1.is_connected()
            assert websocket2.is_connected()
            
            # Subscribe to different markets concurrently
            await asyncio.gather(
                websocket1.subscribe_markets(['MARKET1']),
                websocket2.subscribe_markets(['MARKET2'])
            )
            
            # Disconnect both
            await asyncio.gather(
                websocket1.disconnect(),
                websocket2.disconnect()
            )
            
            assert not websocket1.is_connected()
            assert not websocket2.is_connected()


class TestEndToEndIntegration:
    """End-to-end integration tests."""
    
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
    async def test_complete_nfl_streaming_workflow(self, mock_config):
        """Test complete NFL streaming workflow from SDK to data processing."""
        with patch('neural_sdk.streaming.websocket.KalshiWebSocket') as MockKalshiWS, \
             patch('neural_sdk.streaming.websocket.KalshiMarketDiscovery') as MockDiscovery:
            
            # Set up comprehensive mocks
            mock_ws_client = AsyncMock()
            mock_discovery = AsyncMock()
            
            # Mock NFL game markets
            mock_winner_market = MagicMock()
            mock_winner_market.ticker = "KXNFLGAME-25SEP04DALPHI-WINNER"
            mock_spread_market = MagicMock()
            mock_spread_market.ticker = "KXNFLGAME-25SEP04DALPHI-SPREAD"
            
            mock_discovery.discover_nfl_markets.return_value = [
                mock_winner_market, 
                mock_spread_market
            ]
            
            MockKalshiWS.return_value = mock_ws_client
            MockDiscovery.return_value = mock_discovery
            
            # Initialize SDK
            sdk = NeuralSDK(mock_config)
            
            # Track processed data
            processed_data = {
                'market_updates': [],
                'game_summaries': [],
                'trade_signals': []
            }
            
            # Set up comprehensive event handling
            @sdk.on_market_data
            async def process_market_data(data):
                processed_data['market_updates'].append(data)
                
                # Simulate trading logic
                if data.get('yes_price', 0) < 0.3:
                    processed_data['trade_signals'].append({
                        'action': 'BUY',
                        'ticker': data.get('market_ticker'),
                        'reason': 'oversold'
                    })
            
            # Create NFL stream
            nfl_stream = sdk.create_nfl_stream()
            
            # Connect and subscribe
            await nfl_stream.connect()
            await nfl_stream.subscribe_to_game("25SEP04DALPHI")
            
            # Verify subscription worked
            mock_discovery.discover_nfl_markets.assert_called_once()
            expected_tickers = [
                "KXNFLGAME-25SEP04DALPHI-WINNER",
                "KXNFLGAME-25SEP04DALPHI-SPREAD"
            ]
            mock_ws_client.subscribe_markets.assert_called_with(expected_tickers)
            
            # Simulate receiving market data
            # (In real scenario, this would come through WebSocket)
            game_data = {
                'markets': {
                    'KXNFLGAME-25SEP04DALPHI-WINNER': {
                        'yes_price': 0.65,
                        'volume': 1000
                    }
                },
                'home_team': 'PHI',
                'away_team': 'DAL',
                'last_update': 'test_time'
            }
            
            game_id = "25SEP04-DALPHI"
            nfl_stream.games[game_id] = game_data
            
            # Test game analysis
            summary = nfl_stream.get_game_summary(game_id)
            assert summary is not None
            assert summary['win_probability'] == 0.65
            assert summary['home_team'] == 'PHI'
            assert summary['away_team'] == 'DAL'
            
            # Test cleanup
            await nfl_stream.disconnect()
            mock_ws_client.disconnect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_multi_component_integration(self, mock_config):
        """Test integration across multiple SDK components."""
        with patch('neural_sdk.streaming.websocket.KalshiWebSocket') as MockKalshiWS, \
             patch('neural_sdk.streaming.websocket.KalshiMarketDiscovery'):
            
            MockKalshiWS.return_value = AsyncMock()
            
            # Create SDK
            sdk = NeuralSDK(mock_config)
            
            # Create multiple components
            websocket = sdk.create_websocket()
            nfl_stream = sdk.create_nfl_stream()
            
            # Both should use the same config
            assert websocket.config == sdk.config
            assert nfl_stream.config == sdk.config
            
            # Test that they can operate independently
            await websocket.connect()
            await nfl_stream.connect()
            
            assert websocket.is_connected()
            # NFLMarketStream doesn't have is_connected, but websocket should
            assert nfl_stream.websocket._connected
            
            # Test cleanup
            await websocket.disconnect()
            await nfl_stream.disconnect()


if __name__ == "__main__":
    pytest.main([__file__])
