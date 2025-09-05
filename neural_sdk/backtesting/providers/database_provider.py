"""
Database Data Provider for Backtesting

Loads historical trading data from SQL databases.
Supports PostgreSQL, MySQL, SQLite, and other SQLAlchemy-compatible databases.
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from .base import DataError, DataProvider

logger = logging.getLogger(__name__)

try:
    import sqlalchemy
    from sqlalchemy import create_engine, text

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False


class DatabaseProvider(DataProvider):
    """
    SQL database provider for backtesting.

    Supports loading historical data from:
    - PostgreSQL
    - MySQL
    - SQLite
    - Any SQLAlchemy-compatible database

    Example:
        ```python
        provider = DatabaseProvider(
            connection_string="postgresql://user:pass@host:5432/dbname"
        )

        # Load from table with filtering
        data = provider.fetch(
            source="trades",
            symbols=["NFL-KC-BUF-WINNER"],
            start="2024-01-01",
            end="2024-03-31"
        )

        # Load with custom SQL
        data = provider.fetch(
            source="SELECT * FROM trades WHERE volume > 100",
            start="2024-01-01"
        )
        ```
    """

    def __init__(
        self,
        connection_string: str,
        pool_size: int = 5,
        max_overflow: int = 10,
        **kwargs,
    ):
        """
        Initialize database provider.

        Args:
            connection_string: SQLAlchemy database URL
            pool_size: Connection pool size
            max_overflow: Max pool overflow
            **kwargs: Additional SQLAlchemy engine options
        """
        super().__init__(**kwargs)

        if not SQLALCHEMY_AVAILABLE:
            raise ImportError(
                "Database provider requires SQLAlchemy: pip install sqlalchemy"
            )

        self.connection_string = connection_string

        # Create database engine
        try:
            engine_kwargs = {
                "pool_size": pool_size,
                "max_overflow": max_overflow,
                "pool_timeout": 30,
                "pool_recycle": 3600,
                **kwargs,
            }

            self.engine = create_engine(connection_string, **engine_kwargs)

            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            logger.info("Database provider initialized successfully")

        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise DataError(f"Failed to connect to database: {e}")

    def fetch(
        self,
        source: str,
        symbols: Optional[List[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: Optional[int] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Fetch data from database.

        Args:
            source: Table name or SQL query
            symbols: Filter by symbols
            start: Start date filter (YYYY-MM-DD)
            end: End date filter (YYYY-MM-DD)
            limit: Maximum number of records
            **kwargs: Additional query parameters

        Returns:
            DataFrame with standardized columns
        """
        try:
            # Build SQL query
            if source.upper().strip().startswith("SELECT"):
                # Custom SQL query
                query = source
                params = kwargs
            else:
                # Table name - build query
                query, params = self._build_query(
                    table=source,
                    symbols=symbols,
                    start=start,
                    end=end,
                    limit=limit,
                    **kwargs,
                )

            logger.info(f"Executing database query: {query}")
            logger.debug(f"Query parameters: {params}")

            # Execute query
            data = pd.read_sql(sql=text(query), con=self.engine, params=params)

            # Ensure timestamp column is datetime
            if "timestamp" in data.columns:
                data["timestamp"] = pd.to_datetime(data["timestamp"])

            # Clean and validate
            data = self.clean_data(data)
            self.validate(data)

            logger.info(f"Loaded {len(data)} records from database")
            return data

        except Exception as e:
            logger.error(f"Database query failed: {e}")
            raise DataError(f"Failed to fetch from database: {e}")

    def _build_query(
        self,
        table: str,
        symbols: Optional[List[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: Optional[int] = None,
        order_by: str = "timestamp",
        **kwargs,
    ) -> tuple:
        """Build SQL query and parameters."""

        # Base query
        query = f"SELECT * FROM {table}"
        conditions = []
        params = {}

        # Add filters
        if start:
            conditions.append("timestamp >= :start_date")
            params["start_date"] = start

        if end:
            conditions.append("timestamp <= :end_date")
            params["end_date"] = end

        if symbols:
            # Create IN clause with parameterized values
            symbol_params = []
            for i, symbol in enumerate(symbols):
                param_name = f"symbol_{i}"
                symbol_params.append(f":{param_name}")
                params[param_name] = symbol

            conditions.append(f"symbol IN ({', '.join(symbol_params)})")

        # Add custom conditions from kwargs
        for key, value in kwargs.items():
            if key not in ["order_by"]:  # Skip non-filter kwargs
                conditions.append(f"{key} = :{key}")
                params[key] = value

        # Add WHERE clause if conditions exist
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        # Add ORDER BY
        if order_by:
            query += f" ORDER BY {order_by}"

        # Add LIMIT
        if limit:
            query += f" LIMIT {limit}"

        return query, params

    def get_available_symbols(self, source: str) -> List[str]:
        """Get unique symbols from database table."""
        try:
            if source.upper().strip().startswith("SELECT"):
                # Extract from custom query - just run it and get symbols
                sample_data = self.fetch(source)
                if len(sample_data) > 10000:
                    sample_data = sample_data.sample(10000)
                return sorted(sample_data["symbol"].unique().tolist())
            else:
                # Simple table query
                query = f"SELECT DISTINCT symbol FROM {source} ORDER BY symbol"

                with self.engine.connect() as conn:
                    result = conn.execute(text(query))
                    symbols = [row[0] for row in result]

                return symbols

        except Exception as e:
            logger.error(f"Error getting symbols from database: {e}")
            return []

    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get information about database table."""
        try:
            # Get table schema
            with self.engine.connect() as conn:
                # Get column info
                if "postgresql" in self.connection_string.lower():
                    schema_query = """
                        SELECT column_name, data_type, is_nullable
                        FROM information_schema.columns 
                        WHERE table_name = :table_name
                        ORDER BY ordinal_position
                    """
                elif "mysql" in self.connection_string.lower():
                    schema_query = """
                        SELECT column_name, data_type, is_nullable
                        FROM information_schema.columns 
                        WHERE table_name = :table_name
                        ORDER BY ordinal_position
                    """
                else:
                    # SQLite or generic
                    schema_query = f"PRAGMA table_info({table_name})"

                schema_result = conn.execute(
                    text(schema_query), {"table_name": table_name}
                )
                columns = list(schema_result)

                # Get row count
                count_query = f"SELECT COUNT(*) FROM {table_name}"
                count_result = conn.execute(text(count_query))
                row_count = count_result.scalar()

                # Get date range if timestamp column exists
                date_range = None
                try:
                    date_query = f"""
                        SELECT MIN(timestamp) as min_date, MAX(timestamp) as max_date 
                        FROM {table_name}
                        WHERE timestamp IS NOT NULL
                    """
                    date_result = conn.execute(text(date_query))
                    date_row = date_result.first()
                    if date_row:
                        date_range = (date_row[0], date_row[1])
                except:
                    pass  # timestamp column might not exist

                return {
                    "table_name": table_name,
                    "columns": columns,
                    "row_count": row_count,
                    "date_range": date_range,
                }

        except Exception as e:
            logger.error(f"Error getting table info: {e}")
            return {}

    def execute_query(self, query: str, params: Dict[str, Any] = None) -> pd.DataFrame:
        """Execute custom SQL query."""
        try:
            data = pd.read_sql(sql=text(query), con=self.engine, params=params or {})

            logger.info(f"Query returned {len(data)} records")
            return data

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise DataError(f"Failed to execute query: {e}")

    def list_tables(self) -> List[str]:
        """List all tables in the database."""
        try:
            with self.engine.connect() as conn:
                if "postgresql" in self.connection_string.lower():
                    query = """
                        SELECT table_name 
                        FROM information_schema.tables 
                        WHERE table_schema = 'public'
                        ORDER BY table_name
                    """
                elif "mysql" in self.connection_string.lower():
                    query = "SHOW TABLES"
                else:
                    # SQLite
                    query = """
                        SELECT name FROM sqlite_master 
                        WHERE type='table'
                        ORDER BY name
                    """

                result = conn.execute(text(query))
                tables = [row[0] for row in result]

                return tables

        except Exception as e:
            logger.error(f"Error listing tables: {e}")
            return []

    def close(self):
        """Close database connection."""
        if hasattr(self, "engine"):
            self.engine.dispose()
            logger.info("Database connection closed")
