"""
Parquet Data Provider for Backtesting

High-performance data loading from Parquet files.
Optimal for large historical datasets with columnar storage.
"""

import glob
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .base import DataError, DataProvider

logger = logging.getLogger(__name__)

try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
    logger.warning("PyArrow not available. Falling back to pandas parquet engine.")


class ParquetProvider(DataProvider):
    """
    Parquet file data provider for backtesting.

    Optimized for large datasets with:
    - Column-based filtering
    - Predicate pushdown
    - Memory-efficient loading
    - Multi-file support

    Example:
        ```python
        provider = ParquetProvider()

        # Load single file
        data = provider.fetch('trades_2024.parquet')

        # Load multiple files with date partitioning
        data = provider.fetch('data/year=2024/month=*/trades.parquet')

        # Load with filtering
        data = provider.fetch(
            'trades.parquet',
            symbols=['NFL-KC-BUF-WINNER'],
            columns=['timestamp', 'symbol', 'price']
        )
        ```
    """

    def __init__(self, use_pyarrow: bool = True, **kwargs):
        """
        Initialize Parquet provider.

        Args:
            use_pyarrow: Use PyArrow engine if available (recommended)
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)

        self.use_pyarrow = use_pyarrow and PYARROW_AVAILABLE
        if use_pyarrow and not PYARROW_AVAILABLE:
            logger.warning("PyArrow requested but not available")

        # Parquet reading options
        self.parquet_options = {
            "engine": "pyarrow" if self.use_pyarrow else "auto",
            **kwargs.get("parquet_options", {}),
        }

    def fetch(
        self,
        source: str,
        symbols: Optional[List[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        columns: Optional[List[str]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load data from Parquet file(s).

        Args:
            source: Parquet file path or glob pattern
            symbols: Filter by symbols (pushed down to file level if supported)
            start: Start date filter
            end: End date filter
            columns: Specific columns to load (memory optimization)
            **kwargs: Additional parquet reading options

        Returns:
            DataFrame with requested data
        """
        try:
            # Handle glob patterns for multiple files
            if "*" in source or "?" in source:
                files = glob.glob(source)
                if not files:
                    raise FileNotFoundError(
                        f"No files found matching pattern: {source}"
                    )
                logger.info(f"Loading {len(files)} Parquet files")

                if self.use_pyarrow and len(files) > 1:
                    # Use PyArrow dataset for efficient multi-file loading
                    data = self._load_dataset(
                        files, symbols, start, end, columns, **kwargs
                    )
                else:
                    # Load individual files
                    dataframes = []
                    for file_path in files:
                        df = self._load_single_file(file_path, columns, **kwargs)
                        dataframes.append(df)
                    data = pd.concat(dataframes, ignore_index=True)

            else:
                # Single file
                data = self._load_single_file(source, columns, **kwargs)

            # Validate data
            self.validate(data)

            # Apply filters (if not already applied at file level)
            data = self.filter_data(data, symbols, start, end)

            logger.info(f"Loaded {len(data)} records from Parquet: {source}")
            return data

        except Exception as e:
            logger.error(f"Error loading Parquet {source}: {e}")
            raise DataError(f"Failed to load Parquet data: {e}")

    def _load_single_file(
        self, file_path: str, columns: Optional[List[str]] = None, **kwargs
    ) -> pd.DataFrame:
        """Load single Parquet file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found: {file_path}")

        # Merge options
        options = {**self.parquet_options, **kwargs}

        # Read parquet file
        if columns:
            options["columns"] = columns
            logger.debug(f"Loading columns: {columns}")

        data = pd.read_parquet(file_path, **options)

        return data

    def _load_dataset(
        self,
        files: List[str],
        symbols: Optional[List[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        columns: Optional[List[str]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Load multiple files efficiently using PyArrow dataset."""
        if not self.use_pyarrow:
            raise NotImplementedError("Dataset loading requires PyArrow")

        # Create dataset
        dataset = pq.ParquetDataset(files)

        # Build filters for predicate pushdown
        filters = []

        if symbols:
            filters.append(("symbol", "in", symbols))

        if start:
            filters.append(("timestamp", ">=", pd.to_datetime(start)))

        if end:
            filters.append(("timestamp", "<=", pd.to_datetime(end)))

        # Read with filters
        table = dataset.read(columns=columns, filters=filters or None)
        data = table.to_pandas()

        if len(filters) > 0:
            logger.info(f"Applied {len(filters)} filters at file level")

        return data

    def get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Get metadata from Parquet file without loading data.

        Args:
            file_path: Path to Parquet file

        Returns:
            Dict with file metadata
        """
        if not PYARROW_AVAILABLE:
            logger.warning("PyArrow not available for metadata extraction")
            return {}

        try:
            parquet_file = pq.ParquetFile(file_path)
            schema = parquet_file.schema_arrow

            metadata = {
                "num_rows": parquet_file.metadata.num_rows,
                "num_columns": len(schema),
                "columns": [field.name for field in schema],
                "file_size": Path(file_path).stat().st_size,
                "compression": None,
            }

            # Get column statistics if available
            if parquet_file.metadata.row_group(0):
                rg_meta = parquet_file.metadata.row_group(0)
                for i in range(rg_meta.num_columns):
                    col_meta = rg_meta.column(i)
                    if col_meta.compression:
                        metadata["compression"] = col_meta.compression
                        break

            return metadata

        except Exception as e:
            logger.error(f"Error reading Parquet metadata: {e}")
            return {}

    def get_available_symbols(self, source: str) -> List[str]:
        """Get unique symbols from Parquet file efficiently."""
        try:
            if self.use_pyarrow:
                # Read just the symbol column
                data = pd.read_parquet(source, columns=["symbol"])
            else:
                # Read small sample
                data = pd.read_parquet(source, engine="auto")
                if len(data) > 10000:
                    data = data.sample(10000)

            return sorted(data["symbol"].unique().tolist())

        except Exception as e:
            logger.error(f"Error getting symbols from {source}: {e}")
            return []

    def optimize_file(
        self,
        input_path: str,
        output_path: str,
        partition_cols: Optional[List[str]] = None,
        compression: str = "snappy",
    ):
        """
        Optimize Parquet file for backtesting performance.

        Args:
            input_path: Source file path
            output_path: Optimized file path
            partition_cols: Columns to partition by (e.g., ['symbol', 'date'])
            compression: Compression algorithm
        """
        if not self.use_pyarrow:
            logger.error("File optimization requires PyArrow")
            return

        try:
            # Read source file
            table = pq.read_table(input_path)

            # Sort by timestamp for better filtering performance
            if "timestamp" in table.column_names:
                indices = pa.compute.sort_indices(
                    table, sort_keys=[("timestamp", "ascending")]
                )
                table = pa.compute.take(table, indices)

            # Write optimized file
            if partition_cols:
                pq.write_to_dataset(
                    table,
                    root_path=output_path,
                    partition_cols=partition_cols,
                    compression=compression,
                )
                logger.info(f"Created partitioned dataset at: {output_path}")
            else:
                pq.write_table(table, output_path, compression=compression)
                logger.info(f"Optimized file saved to: {output_path}")

        except Exception as e:
            logger.error(f"Error optimizing Parquet file: {e}")

    def create_sample_data(self, output_path: str, num_records: int = 10000):
        """
        Create sample Parquet file for testing.

        Args:
            output_path: Where to save sample file
            num_records: Number of sample records
        """
        from datetime import datetime, timedelta

        import numpy as np

        # Generate sample data
        start_time = datetime(2024, 1, 1)
        timestamps = [start_time + timedelta(minutes=i) for i in range(num_records)]

        symbols = ["NFL-KC-BUF-WINNER", "NFL-KC-BUF-TOTAL", "NBA-LAL-GSW-WINNER"]

        data = pd.DataFrame(
            {
                "timestamp": np.random.choice(timestamps, num_records),
                "symbol": np.random.choice(symbols, num_records),
                "price": np.random.uniform(0.3, 0.7, num_records),
                "volume": np.random.randint(1, 1000, num_records),
            }
        )

        # Sort by timestamp
        data = data.sort_values("timestamp")

        # Save to parquet
        data.to_parquet(output_path, compression="snappy")

        logger.info(f"Sample Parquet file created: {output_path}")
        logger.info(f"Records: {num_records}, Symbols: {len(symbols)}")

    def convert_csv_to_parquet(
        self, csv_path: str, parquet_path: str, optimize: bool = True
    ):
        """
        Convert CSV file to optimized Parquet format.

        Args:
            csv_path: Source CSV file
            parquet_path: Output Parquet file
            optimize: Apply optimization during conversion
        """
        try:
            # Load CSV
            from .csv_provider import CSVProvider

            csv_provider = CSVProvider()
            data = csv_provider.fetch(csv_path)

            if optimize:
                # Sort by timestamp for better query performance
                data = data.sort_values("timestamp")

            # Save as Parquet
            compression = "snappy" if optimize else None
            data.to_parquet(parquet_path, compression=compression, index=False)

            # Compare file sizes
            csv_size = Path(csv_path).stat().st_size
            parquet_size = Path(parquet_path).stat().st_size
            compression_ratio = (csv_size - parquet_size) / csv_size * 100

            logger.info(f"Converted CSV to Parquet: {csv_path} -> {parquet_path}")
            logger.info(
                f"Size reduction: {compression_ratio:.1f}% ({csv_size:,} -> {parquet_size:,} bytes)"
            )

        except Exception as e:
            logger.error(f"Error converting CSV to Parquet: {e}")
            raise
