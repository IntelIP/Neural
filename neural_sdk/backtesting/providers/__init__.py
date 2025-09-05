"""
Data Providers for Backtesting Module

Support for loading historical data from various sources:
- CSV files
- Parquet files
- S3 buckets
- PostgreSQL databases
- Remote APIs

Example:
    ```python
    from neural_sdk.backtesting.providers import CSVProvider, S3Provider

    # Load from CSV
    csv_provider = CSVProvider()
    data = csv_provider.fetch("data/trades.csv")

    # Load from S3
    s3_provider = S3Provider(bucket="my-data")
    data = s3_provider.fetch("trades/2024/", start="2024-01-01", end="2024-12-31")
    ```
"""

from .base import DataProvider
from .csv_provider import CSVProvider
from .parquet_provider import ParquetProvider

try:
    from .s3_provider import S3Provider

    __all__ = ["DataProvider", "CSVProvider", "ParquetProvider", "S3Provider"]
except ImportError:
    # S3 provider requires boto3
    __all__ = ["DataProvider", "CSVProvider", "ParquetProvider"]

try:
    from .database_provider import DatabaseProvider

    __all__.append("DatabaseProvider")
except ImportError:
    # Database provider requires sqlalchemy
    pass
