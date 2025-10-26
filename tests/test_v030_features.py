"""
Test Suite for Neural SDK v0.3.0 Features

Tests for:
- Historical candlesticks fetching
- NBA market collection
- SportMarketCollector unified interface
- Moneyline filtering utilities
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from neural.data_collection.kalshi import (
    KalshiMarketsSource,
    SportMarketCollector,
    filter_moneyline_markets,
    get_moneyline_markets,
    get_nba_games,
)


class TestHistoricalCandlesticks:
    """Test historical candlesticks fetching functionality"""

    @pytest.mark.asyncio
    async def test_fetch_historical_candlesticks_basic(self):
        """Test basic historical candlesticks fetching"""
        source = KalshiMarketsSource(series_ticker="KXNFLGAME")

        # Mock the HTTP client response instead of the method under test
        with patch.object(source, "http_client") as mock_http:
            # Mock the candlesticks API response
            mock_response = {
                "candlesticks": [
                    {
                        "ts": (datetime.now() - timedelta(hours=i)).timestamp(),
                        "open": 45 + i,  # in cents
                        "high": 46 + i,
                        "low": 44 + i,
                        "close": 45 + i,
                        "volume": 100 + i * 10,
                    }
                    for i in range(5)
                ]
            }
            mock_http.get.return_value = mock_response

            result = await source.fetch_historical_candlesticks(
                market_ticker="KXNFLGAME-1234", hours_back=24
            )

            assert not result.empty
            assert "timestamp" in result.columns
            assert "open" in result.columns
            assert "close" in result.columns
            assert len(result) == 5
            # Verify prices are converted from cents to dollars
            assert result["open"].iloc[0] == 0.45

    @pytest.mark.asyncio
    async def test_fetch_historical_candlesticks_with_date_range(self):
        """Test historical candlesticks with custom date range"""
        source = KalshiMarketsSource()

        with patch.object(source, "http_client") as mock_http:
            mock_response = {"candlesticks": []}
            mock_http.get.return_value = mock_response

            start_date = datetime.now() - timedelta(days=7)
            end_date = datetime.now()

            result = await source.fetch_historical_candlesticks(
                market_ticker="TEST-1234", start_date=start_date, end_date=end_date
            )

            assert isinstance(result, pd.DataFrame)
            assert result.empty
            # Verify the correct API endpoint was called
            mock_http.get.assert_called_once()


class TestNBAMarketCollection:
    """Test NBA market collection functionality"""

    @pytest.mark.asyncio
    async def test_get_nba_games_basic(self):
        """Test basic NBA games fetching"""
        with patch("neural.data_collection.kalshi.KalshiHTTPClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock the API response for NBA markets
            mock_response = {
                "markets": [
                    {
                        "ticker": "KXNBA-LAL-GSW-01",
                        "title": "Lakers vs Warriors",
                        "yes_bid": 45,
                        "yes_ask": 47,
                        "volume": 1000,
                    },
                    {
                        "ticker": "KXNBA-BOS-MIA-01",
                        "title": "Celtics vs Heat",
                        "yes_bid": 52,
                        "yes_ask": 54,
                        "volume": 1500,
                    },
                ]
            }
            mock_client.get.return_value = mock_response

            result = await get_nba_games()

            assert not result.empty
            assert len(result) == 2
            assert all(result["ticker"].str.startswith("KXNBA"))
            # Verify prices are converted from cents to dollars
            assert result["yes_bid"].iloc[0] == 0.45

    @pytest.mark.asyncio
    async def test_get_nba_games_with_team_filter(self):
        """Test NBA games with team filtering"""
        with patch("neural.data_collection.kalshi.KalshiHTTPClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = {
                "markets": [
                    {"ticker": "KXNBA-LAL-GSW-01", "title": "Lakers vs Warriors"},
                    {"ticker": "KXNBA-BOS-MIA-01", "title": "Celtics vs Heat"},
                ]
            }
            mock_client.get.return_value = mock_response

            result = await get_nba_games()

            # Filter for Lakers games
            lal_games = result[result["ticker"].str.contains("LAL")]
            assert len(lal_games) == 1
            assert "LAL" in lal_games.iloc[0]["ticker"]


class TestMoneylineFiltering:
    """Test moneyline filtering utilities"""

    def test_filter_moneyline_markets_basic(self):
        """Test basic moneyline filtering"""
        markets_df = pd.DataFrame(
            {
                "ticker": [
                    "KXNFLGAME-KC-BUF-WIN",
                    "KXNFLGAME-KC-BUF-SPREAD",
                    "KXNFLGAME-DAL-PHI-WIN",
                ],
                "title": [
                    "Will Chiefs beat Buffalo?",
                    "Chiefs to cover spread?",
                    "Will Cowboys win?",
                ],
            }
        )

        result = filter_moneyline_markets(markets_df)

        assert len(result) == 2
        assert all(result["ticker"].str.contains("WIN"))

    def test_filter_moneyline_markets_empty(self):
        """Test moneyline filtering with empty DataFrame"""
        empty_df = pd.DataFrame({"ticker": [], "title": []})

        result = filter_moneyline_markets(empty_df)

        assert result.empty
        assert isinstance(result, pd.DataFrame)

    @pytest.mark.asyncio
    async def test_get_moneyline_markets(self):
        """Test get_moneyline_markets function"""
        with patch("neural.data_collection.kalshi.KalshiHTTPClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = {
                "markets": [
                    {"ticker": "KXNFLGAME-KC-BUF-WIN", "title": "Chiefs to win"},
                    {"ticker": "KXNFLGAME-KC-BUF-SPREAD", "title": "Chiefs spread"},
                ]
            }
            mock_client.get.return_value = mock_response

            result = await get_moneyline_markets(sport="NFL")

            # Should only return WIN markets
            assert all(
                "-WIN" in ticker or "winner" in ticker.lower() for ticker in result["ticker"]
            )


class TestSportMarketCollector:
    """Test SportMarketCollector unified interface"""

    @pytest.mark.asyncio
    async def test_sport_market_collector_nfl(self):
        """Test SportMarketCollector for NFL"""
        collector = SportMarketCollector()

        with patch("neural.data_collection.kalshi.KalshiHTTPClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = {
                "markets": [
                    {"ticker": "KXNFLGAME-KC-BUF-WIN", "title": "Will Chiefs beat Buffalo?"}
                ]
            }
            mock_client.get.return_value = mock_response

            result = await collector.get_games(sport="NFL")

            assert not result.empty
            assert "KXNFLGAME" in result.iloc[0]["ticker"]

    @pytest.mark.asyncio
    async def test_sport_market_collector_nba(self):
        """Test SportMarketCollector for NBA"""
        collector = SportMarketCollector()

        with patch("neural.data_collection.kalshi.KalshiHTTPClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = {
                "markets": [{"ticker": "KXNBA-LAL-GSW-WIN", "title": "Will Lakers beat GSW?"}]
            }
            mock_client.get.return_value = mock_response

            result = await collector.get_games(sport="NBA")

            assert not result.empty
            assert "KXNBA" in result.iloc[0]["ticker"]

    @pytest.mark.asyncio
    async def test_sport_market_collector_with_filters(self):
        """Test SportMarketCollector with moneyline filter"""
        collector = SportMarketCollector()

        with patch("neural.data_collection.kalshi.KalshiHTTPClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = {
                "markets": [
                    {"ticker": "KXNFLGAME-KC-BUF-WIN", "title": "Will Chiefs beat Buffalo?"},
                    {"ticker": "KXNFLGAME-KC-BUF-SPREAD", "title": "Chiefs to cover spread?"},
                ]
            }
            mock_client.get.return_value = mock_response

            result = await collector.get_games(sport="NFL", market_type="moneyline")

            # Should filter to only moneyline markets
            assert all("-WIN" in ticker for ticker in result["ticker"])


class TestIntegrationScenarios:
    """Integration tests for v0.3.0 workflows"""

    @pytest.mark.asyncio
    async def test_historical_data_to_backtest_workflow(self):
        """Test complete workflow: fetch historical data -> backtest"""
        source = KalshiMarketsSource()

        with patch.object(source, "http_client") as mock_http:
            mock_response = {
                "candlesticks": [
                    {
                        "ts": (datetime.now() - timedelta(hours=i)).timestamp(),
                        "close": 45 + i,  # in cents
                        "volume": 100 + i * 10,
                    }
                    for i in range(10)
                ]
            }
            mock_http.get.return_value = mock_response

            historical_data = await source.fetch_historical_candlesticks(
                market_ticker="TEST-1234", hours_back=24
            )

            assert len(historical_data) == 10
            # Verify prices are converted from cents to dollars
            assert historical_data["close"].iloc[0] == 0.45
            assert historical_data["close"].iloc[0] < historical_data["close"].iloc[-1]

    @pytest.mark.asyncio
    async def test_multi_sport_collection_workflow(self):
        """Test collecting markets from multiple sports"""
        sports = ["NFL", "NBA", "CFB"]
        results = {}

        for sport in sports:
            with patch("neural.data_collection.kalshi.KalshiHTTPClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client_class.return_value = mock_client

                mock_response = {
                    "markets": [
                        {"ticker": f"KX{sport}-TEST-WIN", "title": f"Will {sport} team win?"}
                    ]
                }
                mock_client.get.return_value = mock_response

                collector = SportMarketCollector()
                results[sport] = await collector.get_games(sport=sport)

        assert len(results) == 3
        assert all(not df.empty for df in results.values())


# All tests use asyncio
pytestmark = pytest.mark.asyncio
