"""
Base Data Provider Interface

Abstract base class for all data providers in the backtesting module.
Ensures consistent interface across different data sources.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DataProvider(ABC):
    """
    Abstract base class for historical data providers.

    All data providers must implement the fetch() method and should
    implement validate() for data quality assurance.
    """

    def __init__(self, **kwargs):
        """Initialize provider with configuration."""
        self.config = kwargs

    @abstractmethod
    def fetch(
        self,
        source: str,
        symbols: Optional[List[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Fetch historical data from the provider.

        Args:
            source: Source identifier (file path, table name, etc.)
            symbols: List of symbols to fetch (None for all)
            start: Start date (YYYY-MM-DD format)
            end: End date (YYYY-MM-DD format)
            **kwargs: Provider-specific parameters

        Returns:
            DataFrame with standardized columns:
            - timestamp: datetime
            - symbol: str
            - price: float
            - volume: int (optional)
            - additional market data columns

        Raises:
            ValueError: If parameters are invalid
            ConnectionError: If unable to connect to data source
            DataError: If data format is invalid
        """
        pass

    def validate(self, data: pd.DataFrame) -> bool:
        """
        Validate data integrity and format.

        Args:
            data: DataFrame to validate

        Returns:
            True if data is valid, raises exception otherwise

        Raises:
            ValueError: If data format is invalid
        """
        required_columns = ["timestamp", "symbol", "price"]
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            raise ValueError(f"Data missing required columns: {missing_columns}")

        # Check for null values in critical columns
        for col in required_columns:
            if data[col].isnull().any():
                null_count = data[col].isnull().sum()
                raise ValueError(f"Column '{col}' has {null_count} null values")

        # Validate timestamp format
        if not pd.api.types.is_datetime64_any_dtype(data["timestamp"]):
            try:
                data["timestamp"] = pd.to_datetime(data["timestamp"])
            except Exception as e:
                raise ValueError(f"Cannot parse timestamp column: {e}")

        # Validate price data
        if not pd.api.types.is_numeric_dtype(data["price"]):
            raise ValueError("Price column must be numeric")

        if (data["price"] <= 0).any():
            raise ValueError("Price column contains non-positive values")

        # Check for reasonable data ranges (prediction market prices are 0-1)
        if (data["price"] > 1.0).any():
            logger.warning(
                "Price values > 1.0 detected. Are these prediction market prices?"
            )

        logger.info(f"Data validation passed: {len(data)} records")
        return True

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize data format.

        Args:
            data: Raw data DataFrame

        Returns:
            Cleaned DataFrame
        """
        # Make a copy to avoid modifying original
        cleaned = data.copy()

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(cleaned["timestamp"]):
            cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"])

        # Sort by timestamp
        cleaned = cleaned.sort_values("timestamp")

        # Remove duplicates
        initial_len = len(cleaned)
        cleaned = cleaned.drop_duplicates(subset=["timestamp", "symbol"])
        if len(cleaned) < initial_len:
            logger.info(f"Removed {initial_len - len(cleaned)} duplicate records")

        # Fill forward any missing prices (conservative approach)
        if "price" in cleaned.columns:
            cleaned["price"] = cleaned.groupby("symbol")["price"].fillna(method="ffill")

        # Add volume column if missing
        if "volume" not in cleaned.columns:
            cleaned["volume"] = 0

        return cleaned

    def filter_data(
        self,
        data: pd.DataFrame,
        symbols: Optional[List[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Filter data by symbols and date range.

        Args:
            data: DataFrame to filter
            symbols: List of symbols to include
            start: Start date string
            end: End date string

        Returns:
            Filtered DataFrame
        """
        filtered = data.copy()

        # Filter by symbols
        if symbols:
            filtered = filtered[filtered["symbol"].isin(symbols)]

        # Filter by date range
        if start:
            start_date = pd.to_datetime(start)
            filtered = filtered[filtered["timestamp"] >= start_date]

        if end:
            end_date = pd.to_datetime(end)
            filtered = filtered[filtered["timestamp"] <= end_date]

        logger.info(f"Filtered data: {len(filtered)} records")
        return filtered

    def aggregate_data(
        self, data: pd.DataFrame, frequency: str = "1min"
    ) -> pd.DataFrame:
        """
        Aggregate data to specified frequency.

        Args:
            data: Data to aggregate
            frequency: Pandas frequency string (1min, 5min, 1h, 1d)

        Returns:
            Aggregated DataFrame
        """
        if frequency == "raw":
            return data

        # Group by symbol and timestamp
        grouped = (
            data.groupby("symbol")
            .resample(frequency, on="timestamp")
            .agg(
                {
                    "price": "last",  # Use last price in period
                    "volume": "sum",  # Sum volume
                }
            )
            .reset_index()
        )

        # Remove any null prices (no data in period)
        grouped = grouped.dropna(subset=["price"])

        logger.info(f"Aggregated to {frequency}: {len(grouped)} records")
        return grouped

    def get_available_symbols(self, source: str) -> List[str]:
        """
        Get list of available symbols from data source.

        Args:
            source: Data source identifier

        Returns:
            List of symbol strings
        """
        # Default implementation - override in subclasses
        try:
            sample_data = self.fetch(source, start="2024-01-01", end="2024-01-02")
            return sorted(sample_data["symbol"].unique().tolist())
        except Exception as e:
            logger.error(f"Cannot get symbols from {source}: {e}")
            return []

    def get_date_range(self, source: str, symbol: Optional[str] = None) -> tuple:
        """
        Get available date range for data source.

        Args:
            source: Data source identifier
            symbol: Specific symbol (None for all)

        Returns:
            Tuple of (start_date, end_date) as datetime objects
        """
        try:
            # Load small sample to get date range
            sample_data = self.fetch(source, symbols=[symbol] if symbol else None)
            if len(sample_data) == 0:
                return None, None

            start_date = sample_data["timestamp"].min()
            end_date = sample_data["timestamp"].max()
            return start_date, end_date

        except Exception as e:
            logger.error(f"Cannot get date range from {source}: {e}")
            return None, None


class DataError(Exception):
    """Exception raised for data-related errors."""

    pass


class ConnectionError(Exception):
    """Exception raised for connection-related errors."""

    pass
