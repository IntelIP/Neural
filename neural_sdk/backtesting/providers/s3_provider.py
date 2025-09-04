"""
S3 Data Provider for Backtesting

Loads historical trading data from Amazon S3 buckets.
Supports large-scale data storage and efficient querying.
"""

import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .base import DataError, DataProvider

logger = logging.getLogger(__name__)

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


class S3Provider(DataProvider):
    """
    S3 data provider for backtesting.

    Loads historical data from S3 buckets with support for:
    - Parquet and CSV files
    - Partitioned datasets
    - Prefix-based filtering
    - AWS credential handling

    Example:
        ```python
        provider = S3Provider(
            bucket="my-trading-data",
            aws_profile="default"
        )

        data = provider.fetch(
            source="trades/2024/",
            symbols=["NFL-KC-BUF-WINNER"],
            start="2024-01-01",
            end="2024-03-31"
        )
        ```
    """

    def __init__(
        self,
        bucket: str,
        aws_profile: Optional[str] = None,
        region: str = "us-east-1",
        **kwargs,
    ):
        """
        Initialize S3 provider.

        Args:
            bucket: S3 bucket name
            aws_profile: AWS profile to use (None for default)
            region: AWS region
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)

        if not BOTO3_AVAILABLE:
            raise ImportError("S3 provider requires boto3: pip install boto3")

        self.bucket = bucket
        self.aws_profile = aws_profile
        self.region = region

        # Initialize S3 client
        try:
            session = boto3.Session(profile_name=aws_profile)
            self.s3_client = session.client("s3", region_name=region)
            self.s3_resource = session.resource("s3", region_name=region)

            # Test connection
            self.s3_client.head_bucket(Bucket=bucket)
            logger.info(f"S3 provider initialized for bucket: {bucket}")

        except NoCredentialsError:
            raise DataError(
                "AWS credentials not found. Configure AWS CLI or set environment variables."
            )
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                raise DataError(f"S3 bucket not found: {bucket}")
            elif error_code == "403":
                raise DataError(f"Access denied to S3 bucket: {bucket}")
            else:
                raise DataError(f"S3 connection failed: {e}")

    def fetch(
        self,
        source: str,
        symbols: Optional[List[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        file_format: str = "auto",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Fetch data from S3.

        Args:
            source: S3 prefix/key (e.g., "trades/2024/" or "data.parquet")
            symbols: Filter by symbols (applied after loading)
            start: Start date filter
            end: End date filter
            file_format: File format ("auto", "parquet", "csv")
            **kwargs: Additional parameters

        Returns:
            DataFrame with standardized columns
        """
        try:
            # List objects with the given prefix
            objects = self._list_objects(source)

            if not objects:
                raise DataError(f"No objects found in S3 with prefix: {source}")

            # Filter objects by file type if specified
            if file_format != "auto":
                objects = [obj for obj in objects if obj.endswith(f".{file_format}")]

            logger.info(f"Found {len(objects)} S3 objects to load")

            # Load data from objects
            dataframes = []
            for obj_key in objects:
                try:
                    df = self._load_single_object(obj_key, file_format)
                    if len(df) > 0:
                        dataframes.append(df)
                except Exception as e:
                    logger.warning(f"Failed to load {obj_key}: {e}")

            if not dataframes:
                raise DataError("No valid data loaded from S3 objects")

            # Combine all dataframes
            data = pd.concat(dataframes, ignore_index=True)

            # Clean and validate
            data = self.clean_data(data)
            self.validate(data)

            # Apply filters
            data = self.filter_data(data, symbols, start, end)

            logger.info(f"Loaded {len(data)} records from S3: {source}")
            return data

        except Exception as e:
            logger.error(f"S3 fetch failed: {e}")
            raise DataError(f"Failed to fetch from S3: {e}")

    def _list_objects(self, prefix: str) -> List[str]:
        """List objects in S3 bucket with given prefix."""
        objects = []

        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket, Prefix=prefix)

            for page in pages:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        key = obj["Key"]
                        # Skip directories
                        if not key.endswith("/"):
                            objects.append(key)

        except ClientError as e:
            raise DataError(f"Failed to list S3 objects: {e}")

        return objects

    def _load_single_object(self, key: str, file_format: str = "auto") -> pd.DataFrame:
        """Load single object from S3."""
        try:
            # Get object
            response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
            content = response["Body"].read()

            # Determine file format
            if file_format == "auto":
                if key.endswith(".parquet"):
                    file_format = "parquet"
                elif key.endswith(".csv"):
                    file_format = "csv"
                else:
                    # Try to guess from content
                    file_format = "csv"  # Default fallback

            # Load data based on format
            if file_format == "parquet":
                df = pd.read_parquet(io.BytesIO(content))
            elif file_format == "csv":
                df = pd.read_csv(io.BytesIO(content))
            else:
                raise ValueError(f"Unsupported file format: {file_format}")

            logger.debug(f"Loaded {len(df)} records from {key}")
            return df

        except ClientError as e:
            raise DataError(f"Failed to load S3 object {key}: {e}")

    def upload_data(
        self, data: pd.DataFrame, key: str, file_format: str = "parquet", **kwargs
    ):
        """
        Upload data to S3.

        Args:
            data: DataFrame to upload
            key: S3 key/path
            file_format: File format ("parquet", "csv")
            **kwargs: Format-specific options
        """
        try:
            # Convert to bytes
            if file_format == "parquet":
                buffer = io.BytesIO()
                data.to_parquet(buffer, **kwargs)
                buffer.seek(0)
            elif file_format == "csv":
                buffer = io.StringIO()
                data.to_csv(buffer, index=False, **kwargs)
                buffer = io.BytesIO(buffer.getvalue().encode())
            else:
                raise ValueError(f"Unsupported upload format: {file_format}")

            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket, Key=key, Body=buffer.getvalue()
            )

            logger.info(f"Uploaded {len(data)} records to s3://{self.bucket}/{key}")

        except Exception as e:
            logger.error(f"S3 upload failed: {e}")
            raise DataError(f"Failed to upload to S3: {e}")

    def get_available_symbols(self, source: str) -> List[str]:
        """Get unique symbols from S3 data."""
        try:
            # Load a sample of data to get symbols
            sample_data = self.fetch(source)
            if len(sample_data) > 10000:
                sample_data = sample_data.sample(10000)

            return sorted(sample_data["symbol"].unique().tolist())

        except Exception as e:
            logger.error(f"Error getting symbols from S3: {e}")
            return []

    def create_partitioned_dataset(
        self,
        data: pd.DataFrame,
        base_prefix: str,
        partition_cols: List[str] = None,
        file_format: str = "parquet",
    ):
        """
        Create partitioned dataset in S3.

        Args:
            data: Data to partition and upload
            base_prefix: Base S3 prefix for partitioned data
            partition_cols: Columns to partition by
            file_format: File format for partitioned files
        """
        if not partition_cols:
            partition_cols = ["symbol"]

        try:
            # Group by partition columns
            for group_values, group_data in data.groupby(partition_cols):
                if not isinstance(group_values, tuple):
                    group_values = (group_values,)

                # Build partition path
                partition_path = base_prefix.rstrip("/")
                for col, value in zip(partition_cols, group_values):
                    partition_path += f"/{col}={value}"

                # Create filename
                filename = f"data.{file_format}"
                full_key = f"{partition_path}/{filename}"

                # Upload partition
                self.upload_data(group_data, full_key, file_format)

            logger.info(f"Created partitioned dataset: {base_prefix}")

        except Exception as e:
            logger.error(f"Partitioned dataset creation failed: {e}")
            raise DataError(f"Failed to create partitioned dataset: {e}")
