"""
SQLite database setup and management for Neural SDK Analysis Infrastructure.

This module provides database schema creation, connection management,
and performance optimizations for storing Kalshi market data and analysis results.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages SQLite database connections and operations for market data storage.
    
    This class handles database initialization, schema creation, and provides
    optimized connection settings for high-performance data operations.
    """
    
    # SQL schema definitions
    SCHEMA = """
    -- Core market data table for time-series price data
    CREATE TABLE IF NOT EXISTS market_prices (
        market_id TEXT NOT NULL,
        timestamp INTEGER NOT NULL,
        bid REAL,
        ask REAL,
        last REAL,
        volume INTEGER,
        open_interest INTEGER,
        PRIMARY KEY (market_id, timestamp)
    );

    -- Market metadata and information
    CREATE TABLE IF NOT EXISTS markets (
        market_id TEXT PRIMARY KEY,
        event_ticker TEXT,
        event_name TEXT,
        sport TEXT,
        close_time INTEGER,
        resolution_time INTEGER,
        outcome INTEGER,  -- 1 for YES, 0 for NO, NULL if pending
        created_at INTEGER,
        updated_at INTEGER,
        metadata JSON
    );

    -- Track individual trades for performance analysis
    CREATE TABLE IF NOT EXISTS trades (
        trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
        strategy_id TEXT NOT NULL,
        market_id TEXT NOT NULL,
        side TEXT NOT NULL,  -- 'YES' or 'NO'
        entry_price REAL NOT NULL,
        exit_price REAL,
        quantity INTEGER NOT NULL,
        entry_time INTEGER NOT NULL,
        exit_time INTEGER,
        pnl REAL,
        fees REAL,
        edge_estimate REAL,
        metadata JSON,
        FOREIGN KEY (market_id) REFERENCES markets(market_id)
    );

    -- Store backtest results for strategy evaluation
    CREATE TABLE IF NOT EXISTS backtest_runs (
        run_id TEXT PRIMARY KEY,
        strategy_id TEXT NOT NULL,
        start_date INTEGER NOT NULL,
        end_date INTEGER NOT NULL,
        initial_capital REAL NOT NULL,
        final_capital REAL,
        total_trades INTEGER,
        win_rate REAL,
        sharpe_ratio REAL,
        max_drawdown REAL,
        parameters JSON,
        metrics JSON,
        created_at INTEGER NOT NULL
    );

    -- Store strategy signals for analysis
    CREATE TABLE IF NOT EXISTS signals (
        signal_id INTEGER PRIMARY KEY AUTOINCREMENT,
        strategy_id TEXT NOT NULL,
        market_id TEXT NOT NULL,
        timestamp INTEGER NOT NULL,
        signal_type TEXT NOT NULL,  -- 'BUY_YES', 'BUY_NO', 'SELL_YES', 'SELL_NO', 'HOLD'
        strength REAL,  -- Signal confidence/strength
        edge REAL,
        metadata JSON,
        FOREIGN KEY (market_id) REFERENCES markets(market_id)
    );

    -- Create indexes for optimal query performance
    CREATE INDEX IF NOT EXISTS idx_market_timestamp ON market_prices(timestamp);
    CREATE INDEX IF NOT EXISTS idx_market_id ON market_prices(market_id);
    CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy_id);
    CREATE INDEX IF NOT EXISTS idx_trades_market ON trades(market_id);
    CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp);
    CREATE INDEX IF NOT EXISTS idx_signals_strategy ON signals(strategy_id);
    CREATE INDEX IF NOT EXISTS idx_markets_sport ON markets(sport);
    CREATE INDEX IF NOT EXISTS idx_markets_close ON markets(close_time);
    """
    
    def __init__(self, db_path: str = "data/kalshi_trading.db"):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
        
    def _initialize_database(self):
        """Create database schema and apply optimizations."""
        with self.get_connection() as conn:
            # Apply performance optimizations
            self._optimize_performance(conn)
            
            # Create schema
            conn.executescript(self.SCHEMA)
            conn.commit()
            
            logger.info(f"Database initialized at {self.db_path}")
    
    def _optimize_performance(self, conn: sqlite3.Connection):
        """
        Apply SQLite performance optimizations.
        
        These settings optimize the database for analytical workloads
        with large amounts of time-series data.
        """
        optimizations = [
            "PRAGMA journal_mode = WAL",           # Write-ahead logging for better concurrency
            "PRAGMA synchronous = NORMAL",         # Faster writes with acceptable safety
            "PRAGMA cache_size = -64000",          # 64MB cache (negative = KB)
            "PRAGMA temp_store = MEMORY",          # Use memory for temporary tables
            "PRAGMA mmap_size = 30000000000",      # 30GB memory-mapped I/O
            "PRAGMA page_size = 4096",             # Optimal page size for most systems
            "PRAGMA foreign_keys = ON",            # Enforce foreign key constraints
            "PRAGMA analysis_limit = 1000",        # Analyze up to 1000 rows for query planning
            "PRAGMA optimize",                     # Run ANALYZE on tables that need it
        ]
        
        for pragma in optimizations:
            try:
                conn.execute(pragma)
            except sqlite3.Error as e:
                logger.warning(f"Could not apply optimization '{pragma}': {e}")
    
    @contextmanager
    def get_connection(self) -> sqlite3.Connection:
        """
        Get a database connection with optimal settings.
        
        Yields:
            sqlite3.Connection: Configured database connection
        """
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            isolation_level=None,  # Autocommit mode
            check_same_thread=False
        )
        
        # Enable JSON support
        conn.row_factory = sqlite3.Row
        
        try:
            yield conn
        finally:
            conn.close()
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> list:
        """
        Execute a SELECT query and return results.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            List of Row objects
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor.fetchall()
    
    def execute_insert(self, query: str, params: Optional[tuple] = None) -> int:
        """
        Execute an INSERT query and return the last row ID.
        
        Args:
            query: SQL INSERT query
            params: Query parameters
            
        Returns:
            Last inserted row ID
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            conn.commit()
            return cursor.lastrowid
    
    def execute_many(self, query: str, params_list: list):
        """
        Execute multiple INSERT/UPDATE queries efficiently.
        
        Args:
            query: SQL query with placeholders
            params_list: List of parameter tuples
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            conn.commit()
    
    def vacuum(self):
        """
        Vacuum the database to reclaim space and optimize performance.
        
        Should be run periodically during maintenance windows.
        """
        with self.get_connection() as conn:
            conn.execute("VACUUM")
            conn.execute("ANALYZE")
            logger.info("Database vacuumed and analyzed")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics and health metrics.
        
        Returns:
            Dictionary with database statistics
        """
        stats = {}
        
        with self.get_connection() as conn:
            # Get table row counts
            tables = ['market_prices', 'markets', 'trades', 'backtest_runs', 'signals']
            for table in tables:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                stats[f"{table}_count"] = cursor.fetchone()[0]
            
            # Get database file size
            stats['file_size_mb'] = self.db_path.stat().st_size / (1024 * 1024)
            
            # Get page statistics
            cursor = conn.execute("PRAGMA page_count")
            stats['page_count'] = cursor.fetchone()[0]
            
            cursor = conn.execute("PRAGMA page_size")
            stats['page_size'] = cursor.fetchone()[0]
            
            cursor = conn.execute("PRAGMA cache_size")
            stats['cache_size'] = cursor.fetchone()[0]
            
        return stats
    
    def backup(self, backup_path: str):
        """
        Create a backup of the database.
        
        Args:
            backup_path: Path for the backup file
        """
        backup_path = Path(backup_path)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        
        with self.get_connection() as source_conn:
            with sqlite3.connect(str(backup_path)) as backup_conn:
                source_conn.backup(backup_conn)
                logger.info(f"Database backed up to {backup_path}")


# Convenience function for getting a database instance
_db_instance = None

def get_database(db_path: str = "data/kalshi_trading.db") -> DatabaseManager:
    """
    Get or create a database manager instance.
    
    Args:
        db_path: Path to database file
        
    Returns:
        DatabaseManager instance
    """
    global _db_instance
    if _db_instance is None or _db_instance.db_path != Path(db_path):
        # Create new instance if path is different (important for tests)
        _db_instance = DatabaseManager(db_path)
    return _db_instance