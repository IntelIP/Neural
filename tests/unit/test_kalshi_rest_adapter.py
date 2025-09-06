import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
from neural_sdk.data_sources.kalshi.rest_adapter import KalshiRESTAdapter


class TestKalshiRESTAdapter:
    @pytest.fixture
    def adapter(self):
        """Create adapter with mocked dependencies"""
        with patch('neural_sdk.data_sources.kalshi.rest_adapter.KalshiClient') as mock_client, \
             patch('neural_sdk.data_sources.kalshi.rest_adapter.RSASignatureAuth') as mock_auth, \
             patch('neural_sdk.data_sources.kalshi.rest_adapter.RESTDataSource.__init__', return_value=None):

            mock_config = MagicMock()
            mock_config.api_key_id = "test_key"
            mock_config.private_key = "test_private"
            mock_config.api_base_url = "https://api.test.com"
            mock_config.environment = "test"

            mock_client.return_value.config = mock_config

            adapter = KalshiRESTAdapter()
            adapter.kalshi_client = mock_client.return_value
            adapter.config = mock_config
            adapter.fetch = AsyncMock()
            adapter.transform_response = AsyncMock()

            return adapter

    @pytest.mark.asyncio
    async def test_paginate_events_empty_response(self, adapter):
        """Test pagination with empty response"""
        adapter.get_events = AsyncMock(return_value={"data": {"events": []}})

        result = await adapter._paginate_events()

        assert result == []
        assert adapter.get_events.call_count == 1

    @pytest.mark.asyncio
    async def test_paginate_events_with_cursor(self, adapter):
        """Test pagination with cursor"""
        responses = [
            {"data": {"events": [{"id": 1}], "cursor": "cursor1"}},
            {"data": {"events": [{"id": 2}], "cursor": "cursor2"}},
            {"data": {"events": [{"id": 3}]}}  # No cursor, stops
        ]
        adapter.get_events = AsyncMock(side_effect=responses)

        result = await adapter._paginate_events()

        assert len(result) == 3
        assert adapter.get_events.call_count == 3

    @pytest.mark.asyncio
    async def test_paginate_events_network_failure(self, adapter):
        """Test pagination handles network failure"""
        adapter.get_events = AsyncMock(side_effect=Exception("Network error"))

        with pytest.raises(Exception, match="Network error"):
            await adapter._paginate_events()

    @pytest.mark.asyncio
    async def test_get_game_markets_fallback(self, adapter):
        """Test get_game_markets falls back to markets endpoint"""
        # Mock empty events
        adapter._paginate_events = AsyncMock(return_value=[])
        adapter.get_markets = AsyncMock(return_value={"data": {"markets": [{"title": "Test Game"}]}})
        adapter.transform_response = AsyncMock(return_value={"data": {"markets": [{"title": "Test Game"}]}})

        result = await adapter.get_game_markets()

        adapter.get_markets.assert_called_once_with(status="open", limit=500)
        adapter.transform_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_nfl_markets_with_week_filter(self, adapter):
        """Test NFL markets with week filter"""
        events = [{
            "title": "Week 1 Game",
            "markets": [{"title": "NFL Game Winner", "ticker": "NFL123"}]
        }]
        adapter._paginate_events = AsyncMock(return_value=events)
        adapter.transform_response = AsyncMock(return_value={"data": {"markets": []}})

        result = await adapter.get_nfl_markets(week=1)

        adapter._paginate_events.assert_called_once()
        adapter.transform_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_events_with_nested_markets(self, adapter):
        """Test get_events with nested markets parameter"""
        adapter.fetch = AsyncMock(return_value={"data": {"events": []}})

        result = await adapter.get_events(with_nested_markets=True, cursor="test_cursor")

        adapter.fetch.assert_called_once_with("/events", params={
            "limit": 100,
            "with_nested_markets": True,
            "cursor": "test_cursor"
        })

    @pytest.mark.asyncio
    async def test_health_check_success(self, adapter):
        """Test health check success"""
        adapter.fetch = AsyncMock(return_value={"data": {"markets": []}})

        result = await adapter.health_check()

        assert result is True
        adapter.fetch.assert_called_once_with("/markets", params={"limit": 1})

    @pytest.mark.asyncio
    async def test_health_check_failure(self, adapter):
        """Test health check failure"""
        adapter.fetch = AsyncMock(return_value={})

        result = await adapter.health_check()

        assert result is False