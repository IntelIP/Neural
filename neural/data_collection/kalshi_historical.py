"""
Kalshi Historical Data Source

Provides comprehensive historical data collection for Kalshi markets using all documented
historical endpoints. Supports trade-level data, market candlesticks, and event-level
aggregated data with proper pagination and error handling.
"""

import asyncio
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from .base import BaseDataSource, DataSourceConfig
from neural.auth.http_client import KalshiHTTPClient

logger = logging.getLogger(__name__)


class KalshiHistoricalDataSource(BaseDataSource):
    """
    Historical data source for Kalshi markets supporting all documented historical endpoints.

    This class provides access to:
    - Trade-level data via GET /markets/trades
    - Market candlestick data via GET /series/{series_ticker}/markets/{ticker}/candlesticks
    - Event-level aggregated data via GET /events/{ticker}/candlesticks

    Follows Neural SDK naming conventions and integrates with the BaseDataSource framework.
    """

    # Supported time intervals for candlestick data (minutes)
    SUPPORTED_INTERVALS = [1, 60, 1440]  # 1min, 1hr, 1day

    def __init__(self, config: DataSourceConfig,
                 api_key: Optional[str] = None,
                 private_key_path: Optional[str] = None):
        """
        Initialize Kalshi historical data source.

        Args:
            config: DataSourceConfig with name and other settings
            api_key: Optional Kalshi API key (defaults to environment)
            private_key_path: Optional path to RSA private key (defaults to environment)
        """
        super().__init__(config)

        # Store credentials if provided
        self.api_key = api_key
        self.private_key_path = private_key_path

        # Initialize HTTP client for API access
        self.http_client = KalshiHTTPClient(
            api_key_id=api_key,
            private_key_pem=None  # Will use env/file defaults
        )

        logger.info(f"Initialized KalshiHistoricalDataSource: {config.name}")

    async def _connect_impl(self) -> bool:
        """
        Connect implementation - no persistent connection needed for REST API.

        Returns:
            bool: Always True for REST API
        """
        return True

    async def _disconnect_impl(self) -> None:
        """
        Disconnect implementation - no persistent connection to close.
        """
        pass

    async def _subscribe_impl(self, channels: List[str]) -> bool:
        """
        Subscribe implementation - not applicable for historical data collection.

        Returns:
            bool: Always True
        """
        return True

    async def collect_trades(self, ticker: str,
                           start_ts: int, end_ts: int,
                           limit: int = 1000) -> pd.DataFrame:
        """
        Collect granular trade data for a specific market using GET /markets/trades.

        Args:
            ticker: Market ticker (e.g., 'KXNFLGAME-25SEP22DETBAL-BAL')
            start_ts: Start timestamp (Unix timestamp)
            end_ts: End timestamp (Unix timestamp)
            limit: Maximum trades per page (1-1000, default 1000)

        Returns:
            DataFrame with trade data containing columns:
            - trade_id, ticker, created_time, yes_price, no_price, count, taker_side
        """
        logger.info(f"Collecting trades for {ticker} from {start_ts} to {end_ts}")

        all_trades = []
        cursor = None

        while True:
            try:
                # Use documented /markets/trades endpoint with filters
                params = {
                    "ticker": ticker,
                    "min_ts": start_ts,
                    "max_ts": end_ts,
                    "limit": min(limit, 1000),  # API max is 1000
                }

                if cursor:
                    params["cursor"] = cursor

                # Use the HTTP client to get real trades
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.http_client.get_trades,
                    ticker,
                    start_ts,
                    end_ts,
                    limit,
                    cursor
                )

                # Kalshi API returns trades directly (not nested in "data")
                trades = response.get("trades", [])
                if not trades:
                    break

                all_trades.extend(trades)

                # Check for next page
                cursor = response.get("cursor")
                if not cursor:
                    break

                # Safety check to prevent infinite loops
                if len(all_trades) > 100000:  # Reasonable upper limit
                    logger.warning(f"Reached maximum trade limit (100k) for {ticker}")
                    break

            except Exception as e:
                logger.error(f"Error collecting trades for {ticker}: {e}", exc_info=True)
                print(f"DEBUG: Full error details: {e}")
                break

        # Convert to DataFrame
        if all_trades:
            df = pd.DataFrame(all_trades)
            df['created_time'] = pd.to_datetime(df['created_time'])
            df = df.sort_values('created_time').reset_index(drop=True)
            logger.info(f"Collected {len(df)} trades for {ticker}")
            return df
        else:
            logger.info(f"No trades found for {ticker}")
            return pd.DataFrame()

    async def collect_market_candlesticks(self, series_ticker: str, market_ticker: str,
                                        start_ts: int, end_ts: int,
                                        period_interval: int = 60) -> pd.DataFrame:
        """
        Collect candlestick data for a specific market using GET /series/{series_ticker}/markets/{ticker}/candlesticks.

        Args:
            series_ticker: Series ticker (e.g., 'KXNFLGAME-25SEP22DETBAL')
            market_ticker: Market ticker within series (e.g., 'BAL')
            start_ts: Start timestamp (Unix timestamp)
            end_ts: End timestamp (Unix timestamp)
            period_interval: Time interval in minutes (1, 60, or 1440)

        Returns:
            DataFrame with candlestick data containing OHLC and volume information
        """
        if period_interval not in self.SUPPORTED_INTERVALS:
            raise ValueError(f"period_interval must be one of {self.SUPPORTED_INTERVALS}")

        logger.info(f"Collecting {period_interval}min candlesticks for {series_ticker}/{market_ticker}")

        try:
            # Use documented market candlesticks endpoint
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                self.http_client.get_market_candlesticks,
                series_ticker,
                market_ticker,
                start_ts,
                end_ts,
                period_interval
            )

            # Response structure may vary - try both nested and direct
            candlesticks = response.get("candlesticks", [])
            if not candlesticks:
                # Try nested structure
                data = response.get("data", {})
                candlesticks = data.get("candlesticks", [])

            if candlesticks:
                # Flatten the nested structure
                processed_data = []
                for candle in candlesticks:
                    price_data = candle.get("price", {})
                    yes_bid = candle.get("yes_bid", {})
                    yes_ask = candle.get("yes_ask", {})

                    processed_data.append({
                        # Timestamps
                        "end_period_ts": candle.get("end_period_ts"),
                        "timestamp": datetime.fromtimestamp(candle.get("end_period_ts", 0)),

                        # Price data (OHLC)
                        "open": price_data.get("open"),
                        "high": price_data.get("high"),
                        "low": price_data.get("low"),
                        "close": price_data.get("close"),
                        "mean": price_data.get("mean"),

                        # Bid/ask data
                        "yes_bid_open": yes_bid.get("open"),
                        "yes_bid_high": yes_bid.get("high"),
                        "yes_bid_low": yes_bid.get("low"),
                        "yes_bid_close": yes_bid.get("close"),
                        "yes_ask_open": yes_ask.get("open"),
                        "yes_ask_high": yes_ask.get("high"),
                        "yes_ask_low": yes_ask.get("low"),
                        "yes_ask_close": yes_ask.get("close"),

                        # Volume and open interest
                        "volume": candle.get("volume"),
                        "open_interest": candle.get("open_interest"),
                    })

                df = pd.DataFrame(processed_data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp').reset_index(drop=True)
                logger.info(f"Collected {len(df)} candlesticks for {series_ticker}/{market_ticker}")
                return df
            else:
                logger.info(f"No candlestick data found for {series_ticker}/{market_ticker}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error collecting candlesticks for {series_ticker}/{market_ticker}: {e}")
            return pd.DataFrame()

    async def collect_event_candlesticks(self, event_ticker: str,
                                       start_ts: int, end_ts: int,
                                       period_interval: int = 60) -> pd.DataFrame:
        """
        Collect aggregated candlestick data for an entire event using GET /events/{ticker}/candlesticks.

        Args:
            event_ticker: Event ticker (e.g., 'KXNFLGAME-25SEP22DETBAL')
            start_ts: Start timestamp (Unix timestamp)
            end_ts: End timestamp (Unix timestamp)
            period_interval: Time interval in minutes (1, 60, or 1440)

        Returns:
            DataFrame with aggregated candlestick data across all markets in the event
        """
        if period_interval not in self.SUPPORTED_INTERVALS:
            raise ValueError(f"period_interval must be one of {self.SUPPORTED_INTERVALS}")

        logger.info(f"Collecting {period_interval}min event candlesticks for {event_ticker}")

        try:
            # Use documented event candlesticks endpoint
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                self.http_client.get_event_candlesticks,
                event_ticker,
                start_ts,
                end_ts,
                period_interval
            )

            # Response structure may vary - try both nested and direct
            market_candlesticks = response.get("market_candlesticks", [])
            market_tickers = response.get("market_tickers", [])
            if not market_candlesticks:
                # Try nested structure
                data = response.get("data", {})
                market_candlesticks = data.get("market_candlesticks", [])
                market_tickers = data.get("market_tickers", [])

            if market_candlesticks and market_tickers:
                # Process each market's candlestick data
                all_data = []
                for market_idx, market_candles in enumerate(market_candlesticks):
                    if market_idx < len(market_tickers):
                        market_ticker = market_tickers[market_idx]

                        for candle in market_candles:
                            all_data.append({
                                "market_ticker": market_ticker,
                                "end_period_ts": candle.get("end_period_ts"),
                                "timestamp": datetime.fromtimestamp(candle.get("end_period_ts", 0)),
                                "open": candle.get("open"),
                                "high": candle.get("high"),
                                "low": candle.get("low"),
                                "close": candle.get("close"),
                                "volume": candle.get("volume"),
                                "open_interest": candle.get("open_interest"),
                            })

                df = pd.DataFrame(all_data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values(['market_ticker', 'timestamp']).reset_index(drop=True)
                logger.info(f"Collected {len(df)} event candlesticks for {event_ticker}")
                return df
            else:
                logger.info(f"No event candlestick data found for {event_ticker}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error collecting event candlesticks for {event_ticker}: {e}")
            return pd.DataFrame()

    async def collect_historical_data(self, ticker: str,
                                    start_ts: int, end_ts: int,
                                    data_type: str = "trades") -> pd.DataFrame:
        """
        Unified method to collect historical data with automatic method selection.

        Args:
            ticker: Market or event ticker
            start_ts: Start timestamp
            end_ts: End timestamp
            data_type: Type of data to collect ("trades", "market_candlesticks", "event_candlesticks")

        Returns:
            DataFrame with requested historical data
        """
        if data_type == "trades":
            return await self.collect_trades(ticker, start_ts, end_ts)
        elif data_type == "market_candlesticks":
            # For market candlesticks, ticker should be in format "series/market"
            if "/" in ticker:
                series_ticker, market_ticker = ticker.split("/", 1)
                return await self.collect_market_candlesticks(series_ticker, market_ticker, start_ts, end_ts)
            else:
                raise ValueError("For market_candlesticks, ticker must be in format 'series/market'")
        elif data_type == "event_candlesticks":
            return await self.collect_event_candlesticks(ticker, start_ts, end_ts)
        else:
            raise ValueError(f"Unsupported data_type: {data_type}. Use 'trades', 'market_candlesticks', or 'event_candlesticks'")