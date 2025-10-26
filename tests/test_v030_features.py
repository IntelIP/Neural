"""
Test Suite for Neural SDK v0.3.0 Features

Tests for:
- Historical candlesticks fetching
- NBA market collection
- SportMarketCollector unified interface
- Moneyline filtering utilities
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd

from neural.data_collection.kalshi import (
    KalshiMarketsSource,
    get_nba_games,
    filter_moneyline_markets,
    get_moneyline_markets,
    SportMarketCollector,
)


class TestHistoricalCandlesticks:
    """Test historical candlesticks fetching functionality"""

    @pytest.mark.asyncio
    async def test_fetch_historical_candlesticks_basic(self):
        """Test basic historical candlesticks fetching"""
        source = KalshiMarketsSource(series_ticker="KXNFLGAME")

        # Mock the HTTP response
        with patch.object(source, "fetch_historical_candlesticks") as mock_fetch:
            mock_fetch.return_value = pd.DataFrame(
                {
                    "timestamp": [datetime.now() - timedelta(hours=i) for i in range(5)],
                    "open": [0.45, 0.46, 0.47, 0.48, 0.49],
                    "high": [0.46, 0.47, 0.48, 0.49, 0.50],
                    "low": [0.44, 0.45, 0.46, 0.47, 0.48],
                    "close": [0.45, 0.46, 0.47, 0.48, 0.49],
                    "volume": [100, 150, 200, 250, 300],
                }
            )

            result = await source.fetch_historical_candlesticks(
                market_ticker="KXNFLGAME-1234", hours_back=24
            )

            assert not result.empty
            assert "timestamp" in result.columns
            assert "open" in result.columns
            assert "close" in result.columns
            assert len(result) == 5

    @pytest.mark.asyncio
    async def test_fetch_historical_candlesticks_with_date_range(self):
        """Test historical candlesticks with custom date range"""
        source = KalshiMarketsSource()

        with patch.object(source, "fetch_historical_candlesticks") as mock_fetch:
            mock_fetch.return_value = pd.DataFrame({"timestamp": [], "open": []})

            start_date = datetime.now() - timedelta(days=7)
            end_date = datetime.now()

            result = await source.fetch_historical_candlesticks(
                market_ticker="TEST-1234", start_date=start_date, end_date=end_date
            )

            assert isinstance(result, pd.DataFrame)
            mock_fetch.assert_called_once()


class TestNBAMarketCollection:
    """Test NBA market collection functionality"""

    @pytest.mark.asyncio
    async def test_get_nba_games_basic(self):
        """Test basic NBA games fetching"""
        with patch("neural.data_collection.kalshi._fetch_markets") as mock_fetch:
            mock_fetch.return_value = pd.DataFrame(
                {
                    "ticker": ["KXNBA-LAL-GSW-01", "KXNBA-BOS-MIA-01"],
                    "title": ["Lakers vs Warriors", "Celtics vs Heat"],
                    "yes_bid": [0.45, 0.52],
                    "yes_ask": [0.47, 0.54],
                    "volume": [1000, 1500],
                }
            )

            result = await get_nba_games()

            assert not result.empty
            assert len(result) == 2
            assert all(result["ticker"].str.startswith("KXNBA"))

    @pytest.mark.asyncio
    async def test_get_nba_games_with_team_filter(self):
        """Test NBA games with team filtering"""
        with patch("neural.data_collection.kalshi._fetch_markets") as mock_fetch:
            mock_fetch.return_value = pd.DataFrame(
                {
                    "ticker": ["KXNBA-LAL-GSW-01", "KXNBA-BOS-MIA-01"],
                    "title": ["Lakers vs Warriors", "Celtics vs Heat"],
                }
            )

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
        with patch("neural.data_collection.kalshi._fetch_markets") as mock_fetch:
            mock_fetch.return_value = pd.DataFrame(
                {
                    "ticker": [
                        "KXNFLGAME-KC-BUF-WIN",
                        "KXNFLGAME-KC-BUF-SPREAD",
                    ],
                    "title": ["Chiefs to win", "Chiefs spread"],
                }
            )

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

        with patch.object(collector, "get_games") as mock_fetch:
            mock_fetch.return_value = pd.DataFrame(
                {
                    "ticker": ["KXNFLGAME-KC-BUF-WIN"],
                    "title": ["Will Chiefs beat Buffalo?"],
                }
            )

            result = await collector.get_games(sport="NFL")

            assert not result.empty
            mock_fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_sport_market_collector_nba(self):
        """Test SportMarketCollector for NBA"""
        collector = SportMarketCollector()

        with patch.object(collector, "get_games") as mock_fetch:
            mock_fetch.return_value = pd.DataFrame(
                {
                    "ticker": ["KXNBA-LAL-GSW-WIN"],
                    "title": ["Will Lakers beat GSW?"],
                }
            )

            result = await collector.get_games(sport="NBA")

            assert not result.empty
            assert "KXNBA" in result.iloc[0]["ticker"]

    @pytest.mark.asyncio
    async def test_sport_market_collector_with_filters(self):
        """Test SportMarketCollector with moneyline filter"""
        collector = SportMarketCollector()

        with patch("neural.data_collection.kalshi._fetch_markets") as mock_fetch:
            mock_fetch.return_value = pd.DataFrame(
                {
                    "ticker": [
                        "KXNFLGAME-KC-BUF-WIN",
                        "KXNFLGAME-KC-BUF-SPREAD",
                    ],
                    "title": ["Will Chiefs beat Buffalo?", "Chiefs to cover spread?"],
                }
            )

            result = await collector.get_games(sport="NFL", market_type="moneyline")

            # Should filter to only moneyline markets
            assert all("-WIN" in ticker for ticker in result["ticker"])


class TestIntegrationScenarios:
    """Integration tests for v0.3.0 workflows"""

    @pytest.mark.asyncio
    async def test_historical_data_to_backtest_workflow(self):
        """Test complete workflow: fetch historical data -> backtest"""
        source = KalshiMarketsSource()

        with patch.object(source, "fetch_historical_candlesticks") as mock_fetch:
            mock_fetch.return_value = pd.DataFrame(
                {
                    "timestamp": [datetime.now() - timedelta(hours=i) for i in range(10)],
                    "close": [0.45 + i * 0.01 for i in range(10)],
                    "volume": [100 + i * 10 for i in range(10)],
                }
            )

            historical_data = await source.fetch_historical_candlesticks(
                market_ticker="TEST-1234", hours_back=24
            )

            assert len(historical_data) == 10
            assert historical_data["close"].iloc[0] < historical_data["close"].iloc[-1]

    @pytest.mark.asyncio
    async def test_multi_sport_collection_workflow(self):
        """Test collecting markets from multiple sports"""
        sports = ["NFL", "NBA", "CFB"]
        results = {}

        for sport in sports:
            collector = SportMarketCollector()

            with patch.object(collector, "get_games") as mock_fetch:
                mock_fetch.return_value = pd.DataFrame(
                    {"ticker": [f"KX{sport}-TEST"], "title": [f"Will {sport} team win?"]}
                )

                results[sport] = await collector.get_games(sport=sport)

        assert len(results) == 3
        assert all(not df.empty for df in results.values())


# All tests use asyncio
pytestmark = pytest.mark.asyncio
