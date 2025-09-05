"""
CSV Data Provider for Backtesting

Loads historical trading data from CSV files.
Supports standard CSV format with configurable column mapping.
"""

import glob
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .base import DataError, DataProvider

logger = logging.getLogger(__name__)


class CSVProvider(DataProvider):
    """
    CSV file data provider for backtesting.

    Expects CSV files with columns:
    - timestamp (or datetime, date)
    - symbol (or ticker, market)
    - price (or close, last)
    - volume (optional)

    Example:
        ```python
        provider = CSVProvider(
            column_mapping={
                'datetime': 'timestamp',
                'ticker': 'symbol',
                'close': 'price'
            }
        )
        data = provider.fetch('trades.csv')
        ```
    """

    def __init__(self, column_mapping: Optional[Dict[str, str]] = None, **kwargs):
        """
        Initialize CSV provider.

        Args:
            column_mapping: Map CSV columns to standard names
                          e.g., {'datetime': 'timestamp', 'close': 'price'}
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)

        # Default column mapping
        self.column_mapping = {
            "datetime": "timestamp",
            "date": "timestamp",
            "time": "timestamp",
            "ticker": "symbol",
            "market": "symbol",
            "instrument": "symbol",
            "close": "price",
            "last": "price",
            "trade_price": "price",
            "vol": "volume",
            "qty": "volume",
            "size": "volume",
        }

        # Override with user mapping
        if column_mapping:
            self.column_mapping.update(column_mapping)

        # CSV reading options
        self.csv_options = {
            "parse_dates": True,
            "infer_datetime_format": True,
            **kwargs.get("csv_options", {}),
        }

    def fetch(
        self,
        source: str,
        symbols: Optional[List[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load data from CSV file(s).

        Args:
            source: CSV file path or glob pattern
            symbols: Filter by symbols (applied after loading)
            start: Start date filter
            end: End date filter
            **kwargs: Additional CSV reading options

        Returns:
            DataFrame with standardized columns
        """
        try:
            # Handle glob patterns for multiple files
            if "*" in source or "?" in source:
                files = glob.glob(source)
                if not files:
                    raise FileNotFoundError(
                        f"No files found matching pattern: {source}"
                    )
                logger.info(f"Loading {len(files)} CSV files")

                # Load and combine all files
                dataframes = []
                for file_path in files:
                    df = self._load_single_file(file_path, **kwargs)
                    dataframes.append(df)

                data = pd.concat(dataframes, ignore_index=True)

            else:
                # Single file
                data = self._load_single_file(source, **kwargs)

            # Standardize columns
            data = self._standardize_columns(data)

            # Clean and validate
            data = self.clean_data(data)
            self.validate(data)

            # Apply filters
            data = self.filter_data(data, symbols, start, end)

            logger.info(f"Loaded {len(data)} records from CSV: {source}")
            return data

        except Exception as e:
            logger.error(f"Error loading CSV {source}: {e}")
            raise DataError(f"Failed to load CSV data: {e}")

    def _load_single_file(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load single CSV file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        # Merge CSV options
        options = {**self.csv_options, **kwargs}

        # Try to detect delimiter
        if "sep" not in options:
            with open(file_path, "r") as f:
                first_line = f.readline()
                if "\t" in first_line:
                    options["sep"] = "\t"
                elif ";" in first_line:
                    options["sep"] = ";"
                else:
                    options["sep"] = ","

        data = pd.read_csv(file_path, **options)

        # Add filename as symbol if no symbol column
        if (
            "symbol" not in data.columns
            and len(self._find_symbol_column(data.columns)) == 0
        ):
            # Use filename as symbol
            symbol = path.stem
            data["symbol"] = symbol
            logger.info(f"Using filename as symbol: {symbol}")

        return data

    def _standardize_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Map CSV columns to standard names."""
        # Create column mapping
        renamed_columns = {}

        for original_col in data.columns:
            # Check exact matches first
            if original_col.lower() in self.column_mapping:
                renamed_columns[original_col] = self.column_mapping[
                    original_col.lower()
                ]
            # Check partial matches
            else:
                for pattern, standard in self.column_mapping.items():
                    if pattern.lower() in original_col.lower():
                        renamed_columns[original_col] = standard
                        break

        # Apply renaming
        if renamed_columns:
            data = data.rename(columns=renamed_columns)
            logger.debug(f"Renamed columns: {renamed_columns}")

        return data

    def _find_symbol_column(self, columns: List[str]) -> List[str]:
        """Find potential symbol columns."""
        symbol_patterns = ["symbol", "ticker", "market", "instrument", "contract"]
        matches = []

        for col in columns:
            for pattern in symbol_patterns:
                if pattern.lower() in col.lower():
                    matches.append(col)
                    break

        return matches

    def get_available_symbols(self, source: str) -> List[str]:
        """Get unique symbols from CSV file."""
        try:
            # Load just a sample to get symbols quickly
            sample_options = {**self.csv_options, "nrows": 10000}
            data = self._load_single_file(source, **sample_options)
            data = self._standardize_columns(data)

            if "symbol" in data.columns:
                return sorted(data["symbol"].unique().tolist())
            else:
                # Use filename as symbol
                return [Path(source).stem]

        except Exception as e:
            logger.error(f"Error getting symbols from {source}: {e}")
            return []

    def export_template(self, file_path: str):
        """
        Export a CSV template file with proper column headers.

        Args:
            file_path: Path where to save template
        """
        template_data = pd.DataFrame(
            {
                "timestamp": ["2024-01-01 09:30:00", "2024-01-01 09:31:00"],
                "symbol": ["NFL-KC-BUF-WINNER", "NFL-KC-BUF-TOTAL"],
                "price": [0.65, 0.52],
                "volume": [1000, 500],
            }
        )

        template_data.to_csv(file_path, index=False)
        logger.info(f"CSV template saved to: {file_path}")

    def validate_csv_format(self, file_path: str) -> Dict[str, Any]:
        """
        Validate CSV file format without loading all data.

        Args:
            file_path: Path to CSV file

        Returns:
            Dict with validation results
        """
        results = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "columns": [],
            "row_count": 0,
            "date_range": None,
        }

        try:
            # Read just header and a few rows
            sample = pd.read_csv(file_path, nrows=100)
            results["columns"] = sample.columns.tolist()
            results["row_count"] = len(sample)

            # Check for required columns after mapping
            standardized = self._standardize_columns(sample)

            required_columns = ["timestamp", "symbol", "price"]
            missing = [
                col for col in required_columns if col not in standardized.columns
            ]

            if missing:
                results["errors"].append(f"Missing required columns: {missing}")
            else:
                results["valid"] = True

                # Check date range
                if "timestamp" in standardized.columns:
                    try:
                        timestamps = pd.to_datetime(standardized["timestamp"])
                        results["date_range"] = (
                            timestamps.min().strftime("%Y-%m-%d"),
                            timestamps.max().strftime("%Y-%m-%d"),
                        )
                    except:
                        results["warnings"].append("Cannot parse timestamp column")

        except Exception as e:
            results["errors"].append(f"Cannot read CSV file: {e}")

        return results
