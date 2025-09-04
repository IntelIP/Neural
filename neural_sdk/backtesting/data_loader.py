"""
Data Loader for Backtesting Module

Unified interface for loading historical data from multiple sources.
Handles data validation, caching, and format standardization.
"""

import hashlib
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .providers.base import DataError, DataProvider
from .providers.csv_provider import CSVProvider
from .providers.parquet_provider import ParquetProvider

logger = logging.getLogger(__name__)

try:
    from .providers.s3_provider import S3Provider

    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False

try:
    from .providers.database_provider import DatabaseProvider

    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False


class DataLoader:
    """
    Unified data loader with caching and multi-source support.

    Supports:
    - CSV files
    - Parquet files
    - S3 buckets
    - PostgreSQL databases
    - Data caching for performance
    - Format validation

    Example:
        ```python
        loader = DataLoader(cache_dir="~/.neural_sdk/cache")

        # Load from CSV
        data = loader.load("file", path="trades.csv")

        # Load from S3 with caching
        data = loader.load("s3", bucket="my-data", prefix="2024/")

        # Load from database
        data = loader.load("postgres",
                          connection_string="postgresql://...",
                          query="SELECT * FROM trades")
        ```
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        cache_ttl_hours: int = 24,
        enable_cache: bool = True,
    ):
        """
        Initialize data loader.

        Args:
            cache_dir: Directory for data cache (None to disable)
            cache_ttl_hours: Cache time-to-live in hours
            enable_cache: Enable/disable caching
        """
        self.enable_cache = enable_cache and cache_dir is not None
        self.cache_ttl = timedelta(hours=cache_ttl_hours)

        if self.enable_cache:
            self.cache_dir = Path(cache_dir).expanduser()
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Data cache enabled: {self.cache_dir}")
        else:
            self.cache_dir = None
            logger.info("Data cache disabled")

        # Initialize providers
        self.providers = {
            "file": CSVProvider(),
            "csv": CSVProvider(),
            "parquet": ParquetProvider(),
        }

        # Add optional providers if available
        if S3_AVAILABLE:
            self.providers["s3"] = None  # Lazy initialization
        if DATABASE_AVAILABLE:
            self.providers["postgres"] = None
            self.providers["database"] = None

        logger.info(
            f"Initialized data loader with providers: {list(self.providers.keys())}"
        )

    def load(
        self,
        source_type: str,
        cache_key: Optional[str] = None,
        force_refresh: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load data from specified source.

        Args:
            source_type: Type of data source ('file', 'csv', 'parquet', 's3', 'postgres')
            cache_key: Custom cache key (auto-generated if not provided)
            force_refresh: Force reload even if cached data exists
            **kwargs: Source-specific parameters

        Returns:
            DataFrame with standardized columns

        Raises:
            ValueError: If source type is not supported
            DataError: If data loading fails
        """
        # Generate cache key if not provided
        if cache_key is None:
            cache_key = self._generate_cache_key(source_type, kwargs)

        # Try to load from cache first
        if self.enable_cache and not force_refresh:
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                logger.info(f"Loaded data from cache: {cache_key}")
                return cached_data

        # Load data from source
        data = self._load_from_source(source_type, **kwargs)

        # Cache the loaded data
        if self.enable_cache:
            self._save_to_cache(cache_key, data)

        return data

    def _load_from_source(self, source_type: str, **kwargs) -> pd.DataFrame:
        """Load data from the specified source type."""
        if source_type not in self.providers:
            raise ValueError(
                f"Unsupported source type: {source_type}. Available: {list(self.providers.keys())}"
            )

        # Lazy initialization of providers
        provider = self.providers[source_type]
        if provider is None:
            provider = self._initialize_provider(source_type, **kwargs)

        # Extract source parameter
        source = kwargs.pop("path", None) or kwargs.pop("source", None)
        if source is None and source_type in ["file", "csv", "parquet"]:
            raise ValueError(f"'path' parameter required for {source_type} source")

        # Load data using provider
        data = provider.fetch(source, **kwargs)

        # Validate standard format
        self._validate_data_format(data)

        logger.info(f"Loaded {len(data)} records from {source_type} source")
        return data

    def _initialize_provider(self, source_type: str, **kwargs) -> DataProvider:
        """Initialize provider with configuration."""
        if source_type == "s3":
            if not S3_AVAILABLE:
                raise ImportError("S3 provider requires boto3: pip install boto3")
            from .providers.s3_provider import S3Provider

            provider = S3Provider(**kwargs)
        elif source_type in ["postgres", "database"]:
            if not DATABASE_AVAILABLE:
                raise ImportError(
                    "Database provider requires sqlalchemy: pip install sqlalchemy"
                )
            from .providers.database_provider import DatabaseProvider

            provider = DatabaseProvider(**kwargs)
        else:
            raise ValueError(f"Cannot initialize provider for: {source_type}")

        self.providers[source_type] = provider
        return provider

    def _validate_data_format(self, data: pd.DataFrame):
        """Validate data has required columns and format."""
        required_columns = ["timestamp", "symbol", "price"]
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            raise DataError(f"Data missing required columns: {missing_columns}")

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(data["timestamp"]):
            data["timestamp"] = pd.to_datetime(data["timestamp"])

        # Validate data types
        if not pd.api.types.is_numeric_dtype(data["price"]):
            raise DataError("Price column must be numeric")

    def _generate_cache_key(self, source_type: str, kwargs: Dict[str, Any]) -> str:
        """Generate unique cache key for the data request."""
        # Create hash from source type and parameters
        cache_input = f"{source_type}_{sorted(kwargs.items())}"
        return hashlib.md5(cache_input.encode()).hexdigest()[:12]

    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load data from cache if available and fresh."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if not cache_file.exists():
            return None

        # Check if cache is still fresh
        cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if cache_age > self.cache_ttl:
            logger.debug(f"Cache expired: {cache_key}")
            cache_file.unlink()  # Remove expired cache
            return None

        try:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_key}: {e}")
            cache_file.unlink()  # Remove corrupted cache
            return None

    def _save_to_cache(self, cache_key: str, data: pd.DataFrame):
        """Save data to cache."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(data, f)
            logger.debug(f"Saved to cache: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_key}: {e}")

    def clear_cache(self, cache_key: Optional[str] = None):
        """
        Clear cache files.

        Args:
            cache_key: Specific cache key to clear (None to clear all)
        """
        if not self.enable_cache:
            logger.info("Cache is disabled")
            return

        if cache_key:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                cache_file.unlink()
                logger.info(f"Cleared cache: {cache_key}")
        else:
            # Clear all cache files
            cache_files = list(self.cache_dir.glob("*.pkl"))
            for cache_file in cache_files:
                cache_file.unlink()
            logger.info(f"Cleared {len(cache_files)} cache files")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached data."""
        if not self.enable_cache:
            return {"enabled": False}

        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)

        cache_info = {
            "enabled": True,
            "directory": str(self.cache_dir),
            "files": len(cache_files),
            "total_size_mb": total_size / (1024 * 1024),
            "ttl_hours": self.cache_ttl.total_seconds() / 3600,
        }

        # Get details of each cache file
        cache_details = []
        for cache_file in cache_files:
            stat = cache_file.stat()
            age_hours = (datetime.now().timestamp() - stat.st_mtime) / 3600

            cache_details.append(
                {
                    "key": cache_file.stem,
                    "size_mb": stat.st_size / (1024 * 1024),
                    "age_hours": age_hours,
                    "expired": age_hours > self.cache_ttl.total_seconds() / 3600,
                }
            )

        cache_info["details"] = cache_details
        return cache_info

    def preload_data(
        self, sources: List[Dict[str, Any]], parallel: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Preload multiple data sources.

        Args:
            sources: List of source configurations
                    [{'type': 'csv', 'path': 'data.csv'}, {...}]
            parallel: Load sources in parallel (if supported)

        Returns:
            Dictionary mapping source names to DataFrames
        """
        results = {}

        for i, source_config in enumerate(sources):
            source_type = source_config.pop("type")
            source_name = source_config.pop("name", f"source_{i}")

            try:
                data = self.load(source_type, **source_config)
                results[source_name] = data
                logger.info(f"Preloaded {source_name}: {len(data)} records")
            except Exception as e:
                logger.error(f"Failed to preload {source_name}: {e}")
                results[source_name] = None

        return results

    def get_supported_sources(self) -> List[str]:
        """Get list of supported data source types."""
        return list(self.providers.keys())

    def validate_source_config(self, source_type: str, **kwargs) -> Dict[str, Any]:
        """
        Validate source configuration without loading data.

        Args:
            source_type: Type of data source
            **kwargs: Source configuration

        Returns:
            Validation results
        """
        validation = {"valid": False, "errors": [], "warnings": []}

        try:
            # Check if source type is supported
            if source_type not in self.providers:
                validation["errors"].append(f"Unsupported source type: {source_type}")
                return validation

            # Basic parameter validation
            if source_type in ["file", "csv", "parquet"]:
                if "path" not in kwargs:
                    validation["errors"].append("'path' parameter required")
                else:
                    path = Path(kwargs["path"])
                    if not path.exists():
                        validation["errors"].append(f"File not found: {path}")
                    elif path.is_dir():
                        validation["errors"].append(
                            f"Path is directory, not file: {path}"
                        )

            elif source_type == "s3":
                required_params = ["bucket"]
                missing = [p for p in required_params if p not in kwargs]
                if missing:
                    validation["errors"].append(
                        f"Missing required parameters: {missing}"
                    )

            elif source_type in ["postgres", "database"]:
                if "connection_string" not in kwargs and "host" not in kwargs:
                    validation["errors"].append(
                        "Database connection parameters required"
                    )

            if not validation["errors"]:
                validation["valid"] = True

        except Exception as e:
            validation["errors"].append(f"Validation error: {e}")

        return validation
