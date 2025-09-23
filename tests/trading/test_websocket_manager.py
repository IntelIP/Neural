"""
Unit tests for WebSocket Manager

Tests cover connection management, subscription handling,
and message routing.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from neural.trading.websocket_manager import (
    WebSocketManager, SubscriptionType, WebSocketMessage, Subscription
)
from neural.trading.kalshi_client import KalshiClient, KalshiConfig, Environment


class TestWebSocketMessage:
    """Test WebSocketMessage data class."""
    
    def test_message_creation(self):
        """Test creating a WebSocketMessage."""
        message = WebSocketMessage(
            id="msg_123",
            type="orderbook",
            seq=42,
            msg={"ticker": "TEST", "data": "test_data"},
            timestamp=datetime.now()
        )
        
        assert message.id == "msg_123"
        assert message.type == "orderbook"
        assert message.seq == 42
        assert message.msg["ticker"] == "TEST"
    
    def test_from_raw_message(self):
        """Test creating message from raw data."""
        raw_data = {
            "id": "msg_456",
            "type": "trade",
            "seq": 100,
            "msg": {"ticker": "MARKET", "price": 50}
        }
        
        message = WebSocketMessage.from_raw(raw_data)
        
        assert message.id == "msg_456"
        assert message.type == "trade"
        assert message.seq == 100
        assert message.msg["ticker"] == "MARKET"
        assert message.timestamp is not None


class TestSubscription:
    """Test Subscription data class."""
    
    def test_subscription_creation(self):
        """Test creating a Subscription."""
        subscription = Subscription(
            id="sub_123",
            type=SubscriptionType.ORDERBOOK,
            params={"ticker": "TEST-MARKET"},
            handler=lambda msg: None
        )
        
        assert subscription.id == "sub_123"
        assert subscription.type == SubscriptionType.ORDERBOOK
        assert subscription.params["ticker"] == "TEST-MARKET"
        assert subscription.handler is not None


class TestWebSocketManager:
    """Test WebSocketManager class."""
    
    @pytest.fixture
    def mock_kalshi_client(self):
        """Create mock Kalshi client."""
        client = Mock(spec=KalshiClient)
        client.authenticated = False
        client.access_token = None
        client.config = KalshiConfig(environment=Environment.DEMO)
        return client
    
    @pytest.fixture
    def ws_manager(self, mock_kalshi_client):
        """Create WebSocketManager instance."""
        return WebSocketManager(mock_kalshi_client)
    
    def test_initialization(self, ws_manager, mock_kalshi_client):
        """Test WebSocketManager initialization."""
        assert ws_manager.client == mock_kalshi_client
        assert ws_manager.connected is False
        assert ws_manager.running is False
        assert ws_manager.ws_url == "wss://demo-api.kalshi.co/trade-api/ws/v2"
        assert len(ws_manager.subscriptions) == 0
        assert ws_manager.max_reconnect_attempts == 10
    
    def test_production_url(self):
        """Test production WebSocket URL."""
        client = Mock()
        client.config = KalshiConfig(environment=Environment.PRODUCTION)
        
        ws_manager = WebSocketManager(client)
        
        assert ws_manager.ws_url == "wss://api.elections.kalshi.com/trade-api/ws/v2"
    
    @pytest.mark.asyncio
    async def test_connect_success(self, ws_manager):
        """Test successful WebSocket connection."""
        mock_websocket = AsyncMock()
        
        with patch('websockets.connect') as mock_connect:
            mock_connect.return_value = mock_websocket
            
            await ws_manager.connect()
            
            assert ws_manager.connected is True
            assert ws_manager.websocket == mock_websocket
            assert ws_manager.running is True
            mock_connect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_connect_with_authentication(self, ws_manager):
        """Test connection with authentication headers."""
        ws_manager.client.authenticated = True
        ws_manager.client.access_token = "test_token"
        mock_websocket = AsyncMock()
        
        with patch('websockets.connect') as mock_connect:
            mock_connect.return_value = mock_websocket
            
            await ws_manager.connect()
            
            # Check that authorization header was included
            call_args = mock_connect.call_args
            additional_headers = call_args.kwargs.get('additional_headers')
            assert additional_headers is not None
            assert ('Authorization', 'Bearer test_token') in additional_headers
    
    @pytest.mark.asyncio
    async def test_connect_failure(self, ws_manager):
        """Test WebSocket connection failure."""
        with patch('websockets.connect') as mock_connect:
            mock_connect.side_effect = Exception("Connection failed")
            
            with pytest.raises(Exception, match="Connection failed"):
                await ws_manager.connect()
            
            assert ws_manager.connected is False
    
    @pytest.mark.asyncio
    async def test_disconnect(self, ws_manager):
        """Test WebSocket disconnection."""
        # Setup connected state
        mock_websocket = AsyncMock()
        ws_manager.websocket = mock_websocket
        ws_manager.connected = True
        ws_manager.running = True
        
        await ws_manager.disconnect()
        
        assert ws_manager.should_reconnect is False
        assert ws_manager.running is False
        assert ws_manager.connected is False
        mock_websocket.close.assert_called_once()
    
    def test_get_subscription_type(self, ws_manager):
        """Test subscription type detection from message."""
        # Test orderbook message
        orderbook_msg = WebSocketMessage("1", "orderbook_delta", 1, {}, datetime.now())
        assert ws_manager._get_subscription_type(orderbook_msg) == SubscriptionType.ORDERBOOK
        
        # Test trade message
        trade_msg = WebSocketMessage("2", "trade_update", 2, {}, datetime.now())
        assert ws_manager._get_subscription_type(trade_msg) == SubscriptionType.TRADES
        
        # Test market message
        market_msg = WebSocketMessage("3", "market_status", 3, {}, datetime.now())
        assert ws_manager._get_subscription_type(market_msg) == SubscriptionType.MARKET_STATUS
        
        # Test fill message
        fill_msg = WebSocketMessage("4", "fill_notification", 4, {}, datetime.now())
        assert ws_manager._get_subscription_type(fill_msg) == SubscriptionType.FILLS
        
        # Test order message
        order_msg = WebSocketMessage("5", "order_update", 5, {}, datetime.now())
        assert ws_manager._get_subscription_type(order_msg) == SubscriptionType.ORDERS
        
        # Test unknown message
        unknown_msg = WebSocketMessage("6", "unknown_type", 6, {}, datetime.now())
        assert ws_manager._get_subscription_type(unknown_msg) is None
    
    @pytest.mark.asyncio
    async def test_subscribe_orderbook(self, ws_manager):
        """Test subscribing to orderbook updates."""
        ws_manager.connected = True
        ws_manager.websocket = AsyncMock()
        
        handler = Mock()
        subscription_id = await ws_manager.subscribe_orderbook("TEST-MARKET", handler)
        
        assert subscription_id in ws_manager.subscriptions
        subscription = ws_manager.subscriptions[subscription_id]
        assert subscription.type == SubscriptionType.ORDERBOOK
        assert subscription.params["ticker"] == "TEST-MARKET"
        assert subscription.handler == handler
        
        # Check handler was added to global handlers
        assert handler in ws_manager.message_handlers[SubscriptionType.ORDERBOOK]
    
    @pytest.mark.asyncio
    async def test_subscribe_trades(self, ws_manager):
        """Test subscribing to trade updates."""
        ws_manager.connected = True
        ws_manager.websocket = AsyncMock()
        
        subscription_id = await ws_manager.subscribe_trades("TEST-MARKET")
        
        assert subscription_id in ws_manager.subscriptions
        subscription = ws_manager.subscriptions[subscription_id]
        assert subscription.type == SubscriptionType.TRADES
        assert subscription.params["ticker"] == "TEST-MARKET"
    
    @pytest.mark.asyncio
    async def test_subscribe_fills_requires_auth(self, ws_manager):
        """Test that fill subscription requires authentication."""
        ws_manager.client.authenticated = False
        
        with pytest.raises(RuntimeError, match="Authentication required"):
            await ws_manager.subscribe_fills()
    
    @pytest.mark.asyncio
    async def test_subscribe_fills_authenticated(self, ws_manager):
        """Test subscribing to fills when authenticated."""
        ws_manager.client.authenticated = True
        ws_manager.connected = True
        ws_manager.websocket = AsyncMock()
        
        subscription_id = await ws_manager.subscribe_fills()
        
        assert subscription_id in ws_manager.subscriptions
        subscription = ws_manager.subscriptions[subscription_id]
        assert subscription.type == SubscriptionType.FILLS
    
    @pytest.mark.asyncio
    async def test_unsubscribe(self, ws_manager):
        """Test unsubscribing from subscription."""
        # Setup subscription
        handler = Mock()
        ws_manager.connected = True
        ws_manager.websocket = AsyncMock()
        
        subscription_id = await ws_manager.subscribe_orderbook("TEST-MARKET", handler)
        
        # Verify subscription exists
        assert subscription_id in ws_manager.subscriptions
        assert handler in ws_manager.message_handlers[SubscriptionType.ORDERBOOK]
        
        # Unsubscribe
        await ws_manager.unsubscribe(subscription_id)
        
        # Verify subscription removed
        assert subscription_id not in ws_manager.subscriptions
        assert handler not in ws_manager.message_handlers[SubscriptionType.ORDERBOOK]
    
    @pytest.mark.asyncio
    async def test_unsubscribe_all(self, ws_manager):
        """Test unsubscribing from all subscriptions."""
        ws_manager.connected = True
        ws_manager.websocket = AsyncMock()
        
        # Create multiple subscriptions
        sub1 = await ws_manager.subscribe_orderbook("MARKET1")
        sub2 = await ws_manager.subscribe_trades("MARKET2")
        
        assert len(ws_manager.subscriptions) == 2
        
        # Unsubscribe all
        await ws_manager.unsubscribe_all()
        
        assert len(ws_manager.subscriptions) == 0
    
    def test_add_remove_handlers(self, ws_manager):
        """Test adding and removing global handlers."""
        handler1 = Mock()
        handler2 = Mock()
        
        # Add handlers
        ws_manager.add_handler(SubscriptionType.ORDERBOOK, handler1)
        ws_manager.add_handler(SubscriptionType.ORDERBOOK, handler2)
        
        handlers = ws_manager.message_handlers[SubscriptionType.ORDERBOOK]
        assert len(handlers) == 2
        assert handler1 in handlers
        assert handler2 in handlers
        
        # Remove handler
        ws_manager.remove_handler(SubscriptionType.ORDERBOOK, handler1)
        
        handlers = ws_manager.message_handlers[SubscriptionType.ORDERBOOK]
        assert len(handlers) == 1
        assert handler1 not in handlers
        assert handler2 in handlers
    
    @pytest.mark.asyncio
    async def test_route_message_to_handlers(self, ws_manager):
        """Test message routing to handlers."""
        handler_calls = []
        
        def test_handler(message):
            handler_calls.append(message)
        
        # Add handler
        ws_manager.add_handler(SubscriptionType.ORDERBOOK, test_handler)
        
        # Create and route message
        message = WebSocketMessage(
            "msg_1", "orderbook_delta", 1, {"ticker": "TEST"}, datetime.now()
        )
        
        await ws_manager._route_message(message)
        
        # Verify handler was called
        assert len(handler_calls) == 1
        assert handler_calls[0] == message
    
    @pytest.mark.asyncio
    async def test_route_message_unknown_type(self, ws_manager):
        """Test routing message with unknown type."""
        handler_calls = []
        
        def test_handler(message):
            handler_calls.append(message)
        
        # Add handler for different type
        ws_manager.add_handler(SubscriptionType.ORDERBOOK, test_handler)
        
        # Create message with unknown type
        message = WebSocketMessage(
            "msg_1", "unknown_type", 1, {}, datetime.now()
        )
        
        await ws_manager._route_message(message)
        
        # Handler should not be called for unknown type
        assert len(handler_calls) == 0
    
    @pytest.mark.asyncio
    async def test_call_handler_async(self, ws_manager):
        """Test calling async handler."""
        handler_calls = []
        
        async def async_handler(message):
            handler_calls.append(message)
        
        message = WebSocketMessage("1", "test", 1, {}, datetime.now())
        
        await ws_manager._call_handler(async_handler, message)
        
        assert len(handler_calls) == 1
        assert handler_calls[0] == message
    
    @pytest.mark.asyncio
    async def test_call_handler_sync(self, ws_manager):
        """Test calling sync handler."""
        handler_calls = []
        
        def sync_handler(message):
            handler_calls.append(message)
        
        message = WebSocketMessage("1", "test", 1, {}, datetime.now())
        
        await ws_manager._call_handler(sync_handler, message)
        
        assert len(handler_calls) == 1
        assert handler_calls[0] == message
    
    def test_status_methods(self, ws_manager):
        """Test status and monitoring methods."""
        # Test initial state
        assert ws_manager.is_connected() is False
        assert ws_manager.get_subscription_count() == 0
        assert len(ws_manager.get_subscriptions()) == 0
        
        # Setup connected state with subscriptions
        ws_manager.connected = True
        ws_manager.websocket = Mock()
        ws_manager.subscriptions["sub1"] = Mock()
        ws_manager.subscriptions["sub2"] = Mock()
        ws_manager.last_seq_num = 100
        ws_manager.reconnect_count = 2
        
        assert ws_manager.is_connected() is True
        assert ws_manager.get_subscription_count() == 2
        assert len(ws_manager.get_subscriptions()) == 2
        
        # Test connection status
        status = ws_manager.get_connection_status()
        assert status["connected"] is True
        assert status["subscriptions"] == 2
        assert status["last_seq_num"] == 100
        assert status["reconnect_count"] == 2
    
    @pytest.mark.asyncio
    async def test_context_manager(self, ws_manager):
        """Test using WebSocket manager as async context manager."""
        with patch.object(ws_manager, 'connect') as mock_connect, \
             patch.object(ws_manager, 'disconnect') as mock_disconnect:
            
            async with ws_manager:
                pass
            
            mock_connect.assert_called_once()
            mock_disconnect.assert_called_once()


# Integration tests
class TestWebSocketManagerIntegration:
    """Integration tests for WebSocketManager."""
    
    @pytest.mark.asyncio
    async def test_message_processing_flow(self):
        """Test complete message processing flow."""
        client = Mock()
        client.authenticated = True
        client.access_token = "test_token"
        client.config = KalshiConfig()
        
        ws_manager = WebSocketManager(client)
        
        # Mock message processing
        handler_calls = []
        
        def message_handler(message):
            handler_calls.append(message)
        
        ws_manager.add_handler(SubscriptionType.ORDERBOOK, message_handler)
        
        # Simulate message routing
        test_message = WebSocketMessage(
            "test_id", "orderbook_delta", 1, 
            {"ticker": "TEST", "bid": 50}, 
            datetime.now()
        )
        
        await ws_manager._route_message(test_message)
        
        # Verify message was processed
        assert len(handler_calls) == 1
        assert handler_calls[0].msg["ticker"] == "TEST"
    
    @pytest.mark.asyncio
    async def test_subscription_lifecycle(self):
        """Test complete subscription lifecycle."""
        client = Mock()
        client.authenticated = True
        client.config = KalshiConfig()
        
        ws_manager = WebSocketManager(client)
        ws_manager.connected = True
        ws_manager.websocket = AsyncMock()
        
        # Subscribe to multiple channels
        orderbook_sub = await ws_manager.subscribe_orderbook("MARKET1")
        trades_sub = await ws_manager.subscribe_trades("MARKET2")
        fills_sub = await ws_manager.subscribe_fills()
        
        # Verify all subscriptions created
        assert len(ws_manager.subscriptions) == 3
        
        # Test individual unsubscribe
        await ws_manager.unsubscribe(orderbook_sub)
        assert len(ws_manager.subscriptions) == 2
        
        # Test unsubscribe all
        await ws_manager.unsubscribe_all()
        assert len(ws_manager.subscriptions) == 0
