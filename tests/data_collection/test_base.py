"""
Tests for base data source abstractions.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from neural.data_collection.base import (
    BaseDataSource,
    DataSourceConfig,
    ConnectionState
)
from neural.data_collection.exceptions import (
    ConnectionError,
    ConfigurationError
)


class MockDataSource(BaseDataSource):
    """Mock implementation for testing."""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self._connected = False
        self.connect_called = False
        self.disconnect_called = False
        self.process_called = False
    
    async def _connect(self) -> bool:
        self.connect_called = True
        self._connected = True
        return True
    
    async def _disconnect(self) -> None:
        self.disconnect_called = True
        self._connected = False
    
    async def _process_data(self, data) -> None:
        self.process_called = True
        await self._emit_event("data", data)
    
    def is_connected(self) -> bool:
        return self._connected


class TestConnectionState:
    """Test ConnectionState enum."""
    
    def test_states_exist(self):
        assert ConnectionState.DISCONNECTED
        assert ConnectionState.CONNECTING
        assert ConnectionState.CONNECTED
        assert ConnectionState.RECONNECTING
        assert ConnectionState.ERROR


class TestDataSourceConfig:
    """Test DataSourceConfig class."""
    
    def test_create_config(self):
        config = DataSourceConfig(
            name="test",
            enabled=True,
            reconnect=True,
            reconnect_delay=5.0,
            max_reconnect_attempts=3,
            timeout=30.0
        )
        
        assert config.name == "test"
        assert config.enabled is True
        assert config.reconnect is True
        assert config.reconnect_delay == 5.0
        assert config.max_reconnect_attempts == 3
        assert config.timeout == 30.0
    
    def test_config_defaults(self):
        config = DataSourceConfig(name="test")
        
        assert config.name == "test"
        assert config.enabled is True
        assert config.reconnect is True
        assert config.reconnect_delay == 1.0
        assert config.max_reconnect_attempts == 5
        assert config.timeout == 30.0
    
    def test_config_as_dict(self):
        config = DataSourceConfig(
            name="test",
            metadata={"key": "value"}
        )
        
        config_dict = config.to_dict()
        assert config_dict["name"] == "test"
        assert config_dict["metadata"]["key"] == "value"


class TestBaseDataSource:
    """Test BaseDataSource abstract class."""
    
    @pytest.fixture
    def config(self):
        return DataSourceConfig(
            name="test_source",
            reconnect=True,
            max_reconnect_attempts=3
        )
    
    @pytest.fixture
    def data_source(self, config):
        return MockDataSource(config)
    
    @pytest.mark.asyncio
    async def test_connect_success(self, data_source):
        result = await data_source.connect()
        
        assert result is True
        assert data_source.connect_called is True
        assert data_source.state == ConnectionState.CONNECTED
    
    @pytest.mark.asyncio
    async def test_disconnect(self, data_source):
        await data_source.connect()
        await data_source.disconnect()
        
        assert data_source.disconnect_called is True
        assert data_source.state == ConnectionState.DISCONNECTED
    
    @pytest.mark.asyncio
    async def test_reconnect_on_failure(self, config):
        data_source = MockDataSource(config)
        
        # Mock _connect to fail first time, succeed second time
        call_count = 0
        async def mock_connect():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Connection failed")
            return True
        
        data_source._connect = mock_connect
        
        result = await data_source.connect()
        
        assert result is True
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_max_reconnect_attempts(self, config):
        config.max_reconnect_attempts = 2
        data_source = MockDataSource(config)
        
        # Mock _connect to always fail
        async def mock_connect():
            raise ConnectionError("Connection failed")
        
        data_source._connect = mock_connect
        
        result = await data_source.connect()
        
        assert result is False
        assert data_source.state == ConnectionState.ERROR
    
    def test_register_callback(self, data_source):
        callback = Mock()
        
        data_source.register_callback("data", callback)
        
        assert "data" in data_source._callbacks
        assert callback in data_source._callbacks["data"]
    
    def test_unregister_callback(self, data_source):
        callback = Mock()
        
        data_source.register_callback("data", callback)
        data_source.unregister_callback("data", callback)
        
        assert callback not in data_source._callbacks.get("data", [])
    
    @pytest.mark.asyncio
    async def test_emit_event(self, data_source):
        callback = AsyncMock()
        data_source.register_callback("data", callback)
        
        test_data = {"test": "data"}
        await data_source._process_data(test_data)
        
        callback.assert_called_once_with(test_data)
    
    @pytest.mark.asyncio
    async def test_emit_event_with_sync_callback(self, data_source):
        callback = Mock()
        data_source.register_callback("data", callback)
        
        test_data = {"test": "data"}
        await data_source._process_data(test_data)
        
        callback.assert_called_once_with(test_data)
    
    @pytest.mark.asyncio
    async def test_callback_error_handling(self, data_source):
        # Callback that raises an error
        def bad_callback(data):
            raise Exception("Callback error")
        
        good_callback = Mock()
        
        data_source.register_callback("data", bad_callback)
        data_source.register_callback("data", good_callback)
        
        # Should not crash despite bad callback
        await data_source._process_data({"test": "data"})
        
        # Good callback should still be called
        good_callback.assert_called_once()
    
    def test_get_stats(self, data_source):
        stats = data_source.get_stats()
        
        assert "state" in stats
        assert "connected_at" in stats
        assert "reconnect_count" in stats
        assert "last_error" in stats
        assert stats["state"] == "disconnected"
        assert stats["reconnect_count"] == 0
    
    @pytest.mark.asyncio
    async def test_connected_stats(self, data_source):
        await data_source.connect()
        stats = data_source.get_stats()
        
        assert stats["state"] == "connected"
        assert stats["connected_at"] is not None
    
    @pytest.mark.asyncio
    async def test_disabled_source(self, config):
        config.enabled = False
        data_source = MockDataSource(config)
        
        result = await data_source.connect()
        
        assert result is False
        assert data_source.connect_called is False