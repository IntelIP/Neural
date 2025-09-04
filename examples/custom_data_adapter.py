"""
Custom Data Adapter Example

Shows how to create a custom data provider for the backtesting module.
This example demonstrates connecting to a REST API for historical data.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from neural_sdk.backtesting import BacktestConfig, BacktestEngine
from neural_sdk.backtesting.providers.base import DataError, DataProvider

logger = logging.getLogger(__name__)


class APIProvider(DataProvider):
    """
    Custom data provider that fetches historical data from prediction market APIs.

    This is a demonstration of how to create your own data provider
    for any data source (API, database, file format, etc.).
    """

    def __init__(self, api_key: str = None, base_url: str = None, **kwargs):
        """
        Initialize Kalshi API provider.

        Args:
            api_key: Kalshi API key (if required)
            base_url: Base URL for Kalshi API
        """
        super().__init__(**kwargs)

        self.api_key = api_key
        self.base_url = base_url or "https://trading-api.kalshi.com/trade-api/v2"

        # Set up session with authentication
        self.session = requests.Session()
        if api_key:
            self.session.headers.update(
                {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
            )

    def fetch(
        self,
        source: str,
        symbols: Optional[List[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Fetch historical data from Kalshi API.

        Args:
            source: API endpoint or market category (e.g., 'markets', 'trades')
            symbols: List of market tickers
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)

        Returns:
            DataFrame with standardized columns
        """
        try:
            # Build API request parameters
            params = {}
            if start:
                params["start_date"] = start
            if end:
                params["end_date"] = end
            if symbols:
                params["tickers"] = ",".join(symbols)

            # Construct URL
            url = f"{self.base_url}/{source}"

            # Make API request
            logger.info(f"Fetching data from Kalshi API: {url}")
            response = self.session.get(url, params=params)
            response.raise_for_status()

            # Parse response
            data = response.json()

            # Convert to DataFrame
            if source == "markets":
                df = self._parse_markets_response(data)
            elif source == "trades":
                df = self._parse_trades_response(data)
            else:
                # Generic parser
                df = self._parse_generic_response(data)

            # Clean and validate
            df = self.clean_data(df)
            self.validate(df)

            logger.info(f"Fetched {len(df)} records from Kalshi API")
            return df

        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise DataError(f"Failed to fetch from Kalshi API: {e}")
        except Exception as e:
            logger.error(f"Data parsing failed: {e}")
            raise DataError(f"Failed to parse Kalshi API response: {e}")

    def _parse_markets_response(self, data: Dict) -> pd.DataFrame:
        """Parse markets API response."""
        markets = data.get("markets", [])

        records = []
        for market in markets:
            records.append(
                {
                    "timestamp": datetime.now(),  # Use current time or market timestamp
                    "symbol": market.get("ticker"),
                    "price": market.get("yes_price", 0)
                    / 100,  # Convert cents to dollars
                    "volume": market.get("volume", 0),
                    "open_interest": market.get("open_interest", 0),
                    "status": market.get("status"),
                }
            )

        return pd.DataFrame(records)

    def _parse_trades_response(self, data: Dict) -> pd.DataFrame:
        """Parse trades API response."""
        trades = data.get("trades", [])

        records = []
        for trade in trades:
            records.append(
                {
                    "timestamp": pd.to_datetime(trade.get("timestamp")),
                    "symbol": trade.get("ticker"),
                    "price": trade.get("price", 0) / 100,  # Convert cents to dollars
                    "volume": trade.get("count", 1),
                    "side": trade.get("side"),  # YES/NO
                }
            )

        return pd.DataFrame(records)

    def _parse_generic_response(self, data: Dict) -> pd.DataFrame:
        """Generic response parser."""
        # Handle different response formats
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif "data" in data:
            return pd.DataFrame(data["data"])
        else:
            return pd.DataFrame([data])


class MockAPIProvider(DataProvider):
    """
    Mock API provider for demonstration purposes.
    Simulates an external data API without actually making network calls.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fetch(
        self,
        source: str,
        symbols: Optional[List[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Generate mock API data."""

        from datetime import datetime, timedelta

        import numpy as np

        # Generate date range
        start_date = pd.to_datetime(start) if start else datetime(2024, 1, 1)
        end_date = pd.to_datetime(end) if end else datetime(2024, 3, 31)

        # Default symbols if none provided
        if not symbols:
            symbols = ["NFL-CHIEFS-WINNER", "NBA-LAKERS-WINNER", "PRES-2024-WINNER"]

        # Generate hourly data
        timestamps = pd.date_range(start_date, end_date, freq="1H")

        records = []
        for symbol in symbols:
            # Create realistic price series
            initial_price = np.random.uniform(0.3, 0.7)
            prices = [initial_price]

            for i in range(1, len(timestamps)):
                # Random walk with slight trend
                change = np.random.normal(0, 0.01)
                new_price = max(0.01, min(0.99, prices[-1] + change))
                prices.append(new_price)

            # Create records
            for timestamp, price in zip(timestamps, prices):
                records.append(
                    {
                        "timestamp": timestamp,
                        "symbol": symbol,
                        "price": price,
                        "volume": np.random.randint(10, 200),
                        "source": "mock_api",
                    }
                )

        return pd.DataFrame(records)


class DatabaseProvider(DataProvider):
    """
    Database data provider example.
    Shows how to load data from SQL databases.
    """

    def __init__(self, connection_string: str, **kwargs):
        super().__init__(**kwargs)
        self.connection_string = connection_string

    def fetch(
        self,
        source: str,
        symbols: Optional[List[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Fetch data from database.

        Args:
            source: SQL query or table name
            symbols: Filter symbols in query
            start: Start date filter
            end: End date filter
        """
        try:
            import sqlalchemy

            engine = sqlalchemy.create_engine(self.connection_string)

            # Build query
            if source.upper().startswith("SELECT"):
                # Custom SQL query
                query = source
            else:
                # Table name - build query
                query = f"SELECT * FROM {source}"
                conditions = []

                if start:
                    conditions.append(f"timestamp >= '{start}'")
                if end:
                    conditions.append(f"timestamp <= '{end}'")
                if symbols:
                    symbol_list = "', '".join(symbols)
                    conditions.append(f"symbol IN ('{symbol_list}')")

                if conditions:
                    query += " WHERE " + " AND ".join(conditions)

            # Execute query
            df = pd.read_sql(query, engine)

            # Ensure timestamp is datetime
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])

            logger.info(f"Loaded {len(df)} records from database")
            return df

        except ImportError:
            raise DataError(
                "Database provider requires SQLAlchemy: pip install sqlalchemy"
            )
        except Exception as e:
            raise DataError(f"Database query failed: {e}")


def advanced_strategy_with_custom_data(market_data):
    """
    Advanced strategy that uses additional data fields from custom provider.

    Demonstrates how to access custom fields in your strategy logic.
    """
    prices = market_data["prices"]
    portfolio = market_data["portfolio"]

    # Access the raw data to get additional fields
    raw_data = market_data.get("data", [])

    for data_point in raw_data:
        symbol = data_point.get("symbol")
        price = data_point.get("price")
        volume = data_point.get("volume", 0)

        if symbol not in prices:
            continue

        # Strategy logic using volume data
        if volume > 100:  # High volume threshold
            current_position = portfolio.get_position_size(symbol)

            # Buy high-volume, low-price opportunities
            if price < 0.35 and current_position == 0:
                return {
                    "action": "BUY",
                    "market": symbol,
                    "size": min(200, int(volume * 0.5)),  # Size based on volume
                    "reason": f"High volume ({volume}) + low price ({price:.2f})",
                }

            # Sell when price high regardless of volume
            elif price > 0.75 and current_position > 0:
                return {
                    "action": "SELL",
                    "market": symbol,
                    "size": current_position,
                    "reason": f"High price exit at {price:.2f}",
                }

    return None


def run_custom_provider_example():
    """Demonstrate custom data provider usage."""

    print("=== Custom Data Provider Example ===\n")

    # 1. Test Mock API Provider
    print("Testing Mock API Provider...")
    mock_provider = MockAPIProvider()

    mock_data = mock_provider.fetch(
        source="markets",
        symbols=["NFL-KC-BUF-WINNER", "NBA-LAL-GSW-WINNER"],
        start="2024-01-01",
        end="2024-01-31",
    )

    print(f"Mock API returned {len(mock_data)} records")
    print(f"Columns: {list(mock_data.columns)}")
    print(
        f"Date range: {mock_data['timestamp'].min()} to {mock_data['timestamp'].max()}"
    )

    # 2. Register custom provider with data loader
    from kalshi_trading_sdk.backtesting import DataLoader

    loader = DataLoader()
    loader.providers["mock_api"] = mock_provider

    # Load data through the loader
    loaded_data = loader.load("mock_api", source="markets")
    print(f"Data loader returned {len(loaded_data)} records")

    # 3. Run backtest with custom data
    print("\nRunning backtest with custom data...")

    config = BacktestConfig(
        start_date="2024-01-01",
        end_date="2024-01-31",
        initial_capital=10000.0,
        commission=0.02,
    )

    engine = BacktestEngine(config)
    engine.add_strategy(advanced_strategy_with_custom_data)
    engine.data = loaded_data

    results = engine.run()

    print(f"Backtest Results:")
    print(f"  Total Return: {results.total_return:.2f}%")
    print(f"  Number of Trades: {results.num_trades}")
    print(f"  Sharpe Ratio: {results.metrics.get('sharpe_ratio', 0):.3f}")

    return results


def create_custom_csv_provider():
    """Example of customizing the CSV provider."""

    from kalshi_trading_sdk.backtesting.providers import CSVProvider

    # Custom column mapping for your specific CSV format
    custom_csv = CSVProvider(
        column_mapping={
            "date_time": "timestamp",
            "market_id": "symbol",
            "last_price": "price",
            "traded_volume": "volume",
            "bid_price": "bid",
            "ask_price": "ask",
        },
        csv_options={"sep": "\t", "decimal": ".", "thousands": ","},  # Tab-separated
    )

    print("Custom CSV provider created with tab-separated format")
    return custom_csv


if __name__ == "__main__":

    print("Kalshi Trading SDK - Custom Data Provider Examples")
    print("=" * 55)

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    try:
        # Run the main example
        results = run_custom_provider_example()

        # Show how to customize existing providers
        custom_csv = create_custom_csv_provider()

        print("\n" + "=" * 55)
        print("Custom provider examples completed!")
        print("\nKey takeaways:")
        print("1. Inherit from DataProvider base class")
        print("2. Implement the fetch() method")
        print("3. Return DataFrame with standard columns (timestamp, symbol, price)")
        print("4. Add custom columns for advanced strategies")
        print("5. Register provider with DataLoader for easy use")

    except Exception as e:
        logger.error(f"Example failed: {e}")
        print(f"\nExample failed: {e}")
        print("This is normal if you don't have all dependencies installed.")
