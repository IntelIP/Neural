"""
Unit tests for Neural SDK Market Streaming functionality.

Tests the MarketStream and NFLMarketStream classes.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from typing import Dict, Any

# Mock data_pipeline imports before importing neural_sdk
with patch.dict('sys.modules', {
    'data_pipeline': MagicMock(),
    'data_pipeline.streaming': MagicMock(),
    'data_pipeline.streaming.websocket': MagicMock(),
    'data_pipeline.sports_config': MagicMock(),
}):
    # Import the classes we're testing
    from neural_sdk.streaming.market_stream import MarketStream, NFLMarketStream
    from neural_sdk.core.config import SDKConfig


class TestMarketStream:
    """Test cases for MarketStream base class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock SDK configuration."""
        config = MagicMock(spec=SDKConfig)
        config.environment = "test"
        return config
    
    @pytest.fixture
    def market_stream(self, mock_config):
        """Create a MarketStream instance for testing."""
        with patch('neural_sdk.streaming.market_stream.NeuralWebSocket'):
            return MarketStream(mock_config)
    
    def test_market_stream_initialization(self, mock_config):
        """Test MarketStream initialization."""
        with patch('neural_sdk.streaming.market_stream.NeuralWebSocket') as MockWebSocket:
            mock_websocket = MagicMock()
            MockWebSocket.return_value = mock_websocket
            
            stream = MarketStream(mock_config)
            
            assert stream.config == mock_config
            assert stream.websocket == mock_websocket
            assert len(stream.active_markets) == 0
            assert len(stream.price_history) == 0
    
    @pytest.mark.asyncio
    async def test_connect(self, market_stream):
        """Test stream connection."""
        market_stream.websocket.connect = AsyncMock()
        
        await market_stream.connect()
        
        market_stream.websocket.connect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_disconnect(self, market_stream):
        """Test stream disconnection."""
        market_stream.websocket.disconnect = AsyncMock()
        
        await market_stream.disconnect()
        
        market_stream.websocket.disconnect.assert_called_once()
    
    def test_get_market_data(self, market_stream):
        """Test getting market data for a ticker."""
        # Set up test data
        test_data = {'yes_price': 0.55, 'volume': 100}
        market_stream.active_markets['TEST-MARKET'] = test_data
        
        result = market_stream.get_market_data('TEST-MARKET')
        assert result == test_data
        
        # Test non-existent market
        result = market_stream.get_market_data('NON-EXISTENT')
        assert result is None
    
    def test_get_price_history(self, market_stream):
        """Test getting price history for a ticker."""
        # Set up test data
        test_history = [
            {'timestamp': datetime.now(), 'yes_price': 0.50},
            {'timestamp': datetime.now(), 'yes_price': 0.55}
        ]
        market_stream.price_history['TEST-MARKET'] = test_history
        
        result = market_stream.get_price_history('TEST-MARKET')
        assert result == test_history
        
        # Test non-existent market
        result = market_stream.get_price_history('NON-EXISTENT')
        assert result == []
    
    def test_get_active_markets(self, market_stream):
        """Test getting list of active markets."""
        market_stream.active_markets = {
            'MARKET1': {'yes_price': 0.50},
            'MARKET2': {'yes_price': 0.60},
            'MARKET3': {'yes_price': 0.70}
        }
        
        active_markets = market_stream.get_active_markets()
        
        assert len(active_markets) == 3
        assert 'MARKET1' in active_markets
        assert 'MARKET2' in active_markets
        assert 'MARKET3' in active_markets


class TestNFLMarketStream:
    """Test cases for NFLMarketStream class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock SDK configuration."""
        config = MagicMock(spec=SDKConfig)
        config.environment = "test"
        return config
    
    @pytest.fixture
    def nfl_stream(self, mock_config):
        """Create an NFLMarketStream instance for testing."""
        with patch('neural_sdk.streaming.market_stream.NeuralWebSocket'):
            return NFLMarketStream(mock_config)
    
    def test_nfl_stream_initialization(self, mock_config):
        """Test NFLMarketStream initialization."""
        with patch('neural_sdk.streaming.market_stream.NeuralWebSocket'):
            stream = NFLMarketStream(mock_config)
            
            assert len(stream.games) == 0
            assert len(stream.teams) == 0
            assert hasattr(stream, 'active_markets')  # Inherited from MarketStream
            assert hasattr(stream, 'price_history')   # Inherited from MarketStream
    
    @pytest.mark.asyncio
    async def test_subscribe_to_game(self, nfl_stream):
        """Test subscribing to a specific NFL game."""
        nfl_stream.websocket.subscribe_nfl_game = AsyncMock()
        
        game_id = "25SEP04DALPHI"
        await nfl_stream.subscribe_to_game(game_id)
        
        nfl_stream.websocket.subscribe_nfl_game.assert_called_once_with(game_id)
    
    @pytest.mark.asyncio
    async def test_subscribe_to_team(self, nfl_stream):
        """Test subscribing to a specific NFL team."""
        nfl_stream.websocket.subscribe_nfl_team = AsyncMock()
        
        team_code = "PHI"
        await nfl_stream.subscribe_to_team(team_code)
        
        nfl_stream.websocket.subscribe_nfl_team.assert_called_once_with(team_code)
    
    @pytest.mark.asyncio
    async def test_subscribe_to_all_nfl(self, nfl_stream):
        """Test subscribing to all NFL markets."""
        nfl_stream.websocket.subscribe_markets = AsyncMock()
        
        await nfl_stream.subscribe_to_all_nfl()
        
        nfl_stream.websocket.subscribe_markets.assert_called_once_with(['KXNFLGAME*'])
    
    def test_extract_game_id(self, nfl_stream):
        """Test extracting game ID from ticker."""
        # Test valid NFL ticker
        ticker = "KXNFLGAME-25SEP04-DALPHI-PHI"
        game_id = nfl_stream._extract_game_id(ticker)
        assert game_id == "25SEP04-DALPHI"
        
        # Test invalid ticker
        ticker = "INVALID-TICKER"
        game_id = nfl_stream._extract_game_id(ticker)
        assert game_id is None
    
    def test_extract_team_codes(self, nfl_stream):
        """Test extracting team codes from ticker."""
        ticker = "KXNFLGAME-25SEP04-DALPHI-PHI"
        
        home_team = nfl_stream._extract_home_team(ticker)
        away_team = nfl_stream._extract_away_team(ticker)
        
        assert home_team == "PHI"
        assert len(away_team) == 3  # Should be 3-character team code
    
    def test_get_game_data(self, nfl_stream):
        """Test getting data for a specific game."""
        # Set up test game data
        game_id = "25SEP04-DALPHI"
        test_game_data = {
            'markets': {'MARKET1': {'yes_price': 0.55}},
            'home_team': 'PHI',
            'away_team': 'DAL',
            'last_update': datetime.now()
        }
        nfl_stream.games[game_id] = test_game_data
        
        result = nfl_stream.get_game_data(game_id)
        assert result == test_game_data
        
        # Test non-existent game
        result = nfl_stream.get_game_data('NON-EXISTENT')
        assert result is None
    
    def test_get_team_markets(self, nfl_stream):
        """Test getting markets for a specific team."""
        # Set up test team data
        team_code = "PHI"
        test_markets = ["MARKET1", "MARKET2", "MARKET3"]
        nfl_stream.teams[team_code] = test_markets
        
        result = nfl_stream.get_team_markets(team_code)
        assert result == test_markets
        
        # Test case insensitive
        result = nfl_stream.get_team_markets("phi")
        assert result == test_markets
        
        # Test non-existent team
        result = nfl_stream.get_team_markets("XXX")
        assert result == []
    
    def test_get_active_games(self, nfl_stream):
        """Test getting list of active games."""
        nfl_stream.games = {
            'GAME1': {'markets': {}},
            'GAME2': {'markets': {}},
            'GAME3': {'markets': {}}
        }
        
        active_games = nfl_stream.get_active_games()
        
        assert len(active_games) == 3
        assert 'GAME1' in active_games
        assert 'GAME2' in active_games
        assert 'GAME3' in active_games
    
    def test_get_game_win_probability(self, nfl_stream):
        """Test calculating win probability for a game."""
        game_id = "25SEP04-DALPHI"
        
        # Set up game with winner market
        nfl_stream.games[game_id] = {
            'markets': {
                'KXNFLGAME-25SEP04-DALPHI-WINNER': {'yes_price': 0.65},
                'KXNFLGAME-25SEP04-DALPHI-SPREAD': {'yes_price': 0.50}
            }
        }
        
        win_prob = nfl_stream.get_game_win_probability(game_id)
        assert win_prob == 0.65
        
        # Test game without winner market
        nfl_stream.games[game_id] = {
            'markets': {
                'KXNFLGAME-25SEP04-DALPHI-SPREAD': {'yes_price': 0.50}
            }
        }
        
        win_prob = nfl_stream.get_game_win_probability(game_id)
        assert win_prob is None
        
        # Test non-existent game
        win_prob = nfl_stream.get_game_win_probability('NON-EXISTENT')
        assert win_prob is None
    
    def test_get_game_summary(self, nfl_stream):
        """Test getting comprehensive game summary."""
        game_id = "25SEP04-DALPHI"
        test_time = datetime.now()
        
        # Set up game data
        nfl_stream.games[game_id] = {
            'markets': {
                'KXNFLGAME-25SEP04-DALPHI-WINNER': {'yes_price': 0.65},
                'KXNFLGAME-25SEP04-DALPHI-SPREAD': {'yes_price': 0.50}
            },
            'home_team': 'PHI',
            'away_team': 'DAL',
            'last_update': test_time
        }
        
        summary = nfl_stream.get_game_summary(game_id)
        
        assert summary is not None
        assert summary['game_id'] == game_id
        assert summary['home_team'] == 'PHI'
        assert summary['away_team'] == 'DAL'
        assert summary['markets_count'] == 2
        assert summary['win_probability'] == 0.65
        assert summary['last_update'] == test_time.isoformat()
        
        # Test non-existent game
        summary = nfl_stream.get_game_summary('NON-EXISTENT')
        assert summary is None


class TestMarketStreamIntegration:
    """Integration tests for market streaming functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock SDK configuration."""
        config = MagicMock(spec=SDKConfig)
        config.environment = "test"
        return config
    
    @pytest.mark.asyncio
    async def test_nfl_stream_lifecycle(self, mock_config):
        """Test complete NFL stream lifecycle."""
        with patch('neural_sdk.streaming.market_stream.NeuralWebSocket') as MockWebSocket:
            mock_websocket = AsyncMock()
            MockWebSocket.return_value = mock_websocket
            
            # Create NFL stream
            nfl_stream = NFLMarketStream(mock_config)
            
            # Connect
            await nfl_stream.connect()
            mock_websocket.connect.assert_called_once()
            
            # Subscribe to game
            await nfl_stream.subscribe_to_game("25SEP04DALPHI")
            mock_websocket.subscribe_nfl_game.assert_called_with("25SEP04DALPHI")
            
            # Subscribe to team
            await nfl_stream.subscribe_to_team("PHI")
            mock_websocket.subscribe_nfl_team.assert_called_with("PHI")
            
            # Simulate market data processing
            # (This would normally be handled by the WebSocket handlers)
            game_id = "25SEP04-DALPHI"
            nfl_stream.games[game_id] = {
                'markets': {
                    'TEST-MARKET': {'yes_price': 0.60, 'volume': 100}
                },
                'home_team': 'PHI',
                'away_team': 'DAL',
                'last_update': datetime.now()
            }
            
            # Test data retrieval
            game_data = nfl_stream.get_game_data(game_id)
            assert game_data is not None
            assert game_data['home_team'] == 'PHI'
            
            summary = nfl_stream.get_game_summary(game_id)
            assert summary is not None
            assert summary['markets_count'] == 1
            
            # Disconnect
            await nfl_stream.disconnect()
            mock_websocket.disconnect.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
