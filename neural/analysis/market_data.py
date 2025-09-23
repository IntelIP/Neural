"""
Market data storage and retrieval for Neural SDK Analysis Infrastructure.

This module provides the MarketDataStore class for managing historical
Kalshi market data, including prices, volumes, and market metadata.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict

from neural.analysis.database import DatabaseManager, get_database

logger = logging.getLogger(__name__)


@dataclass
class PriceUpdate:
    """Represents a single price update for a market."""
    market_id: str
    timestamp: int
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return asdict(self)


@dataclass
class MarketInfo:
    """Market metadata and information."""
    market_id: str
    event_ticker: str
    event_name: str
    sport: str
    close_time: int
    resolution_time: Optional[int] = None
    outcome: Optional[int] = None  # 1 for YES, 0 for NO
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        data = asdict(self)
        if self.metadata:
            data['metadata'] = json.dumps(self.metadata)
        return data


class MarketDataStore:
    """
    Manages storage and retrieval of Kalshi market data.
    
    This class provides high-level methods for working with market data,
    including efficient bulk operations and time-series queries.
    """
    
    def __init__(self, db_path: str = "data/kalshi_trading.db"):
        """
        Initialize market data store.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db = get_database(db_path)
        logger.info(f"MarketDataStore initialized with database at {db_path}")
    
    def store_price_update(self, update: PriceUpdate):
        """
        Store a single price update.
        
        Args:
            update: PriceUpdate object
        """
        query = """
        INSERT OR REPLACE INTO market_prices 
        (market_id, timestamp, bid, ask, last, volume, open_interest)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            update.market_id,
            update.timestamp,
            update.bid,
            update.ask,
            update.last,
            update.volume,
            update.open_interest
        )
        
        self.db.execute_insert(query, params)
    
    def store_price_updates_bulk(self, updates: List[PriceUpdate]):
        """
        Store multiple price updates efficiently.
        
        Args:
            updates: List of PriceUpdate objects
        """
        query = """
        INSERT OR REPLACE INTO market_prices 
        (market_id, timestamp, bid, ask, last, volume, open_interest)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        
        params_list = [
            (u.market_id, u.timestamp, u.bid, u.ask, u.last, u.volume, u.open_interest)
            for u in updates
        ]
        
        self.db.execute_many(query, params_list)
        logger.info(f"Stored {len(updates)} price updates")
    
    def get_price_history(
        self, 
        market_id: str, 
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get historical price data for a market.
        
        Args:
            market_id: Market identifier
            start_time: Start timestamp (Unix epoch)
            end_time: End timestamp (Unix epoch)
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with price history
        """
        query = "SELECT * FROM market_prices WHERE market_id = ?"
        params = [market_id]
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp < ?"
            params.append(end_time)
        
        query += " ORDER BY timestamp"
        
        if limit:
            query += f" LIMIT {limit}"
        
        rows = self.db.execute_query(query, tuple(params))
        
        # Convert to DataFrame
        if rows:
            df = pd.DataFrame([dict(row) for row in rows])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('timestamp', inplace=True)
            return df
        else:
            return pd.DataFrame()
    
    def get_latest_price(self, market_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the most recent price for a market.
        
        Args:
            market_id: Market identifier
            
        Returns:
            Dictionary with latest price data or None
        """
        query = """
        SELECT * FROM market_prices 
        WHERE market_id = ? 
        ORDER BY timestamp DESC 
        LIMIT 1
        """
        
        rows = self.db.execute_query(query, (market_id,))
        
        if rows:
            return dict(rows[0])
        return None
    
    def store_market_info(self, market: MarketInfo):
        """
        Store or update market metadata.
        
        Args:
            market: MarketInfo object
        """
        current_time = int(datetime.now().timestamp())
        
        query = """
        INSERT OR REPLACE INTO markets 
        (market_id, event_ticker, event_name, sport, close_time, 
         resolution_time, outcome, created_at, updated_at, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, 
                COALESCE((SELECT created_at FROM markets WHERE market_id = ?), ?),
                ?, ?)
        """
        
        metadata_json = json.dumps(market.metadata) if market.metadata else None
        
        params = (
            market.market_id,
            market.event_ticker,
            market.event_name,
            market.sport,
            market.close_time,
            market.resolution_time,
            market.outcome,
            market.market_id,  # For COALESCE subquery
            current_time,      # For COALESCE default
            current_time,      # updated_at
            metadata_json
        )
        
        self.db.execute_insert(query, params)
    
    def get_market_info(self, market_id: str) -> Optional[MarketInfo]:
        """
        Get market metadata.
        
        Args:
            market_id: Market identifier
            
        Returns:
            MarketInfo object or None
        """
        query = "SELECT * FROM markets WHERE market_id = ?"
        rows = self.db.execute_query(query, (market_id,))
        
        if rows:
            row = dict(rows[0])
            if row['metadata']:
                row['metadata'] = json.loads(row['metadata'])
            
            # Remove database-specific fields
            row.pop('created_at', None)
            row.pop('updated_at', None)
            
            return MarketInfo(**row)
        return None
    
    def get_markets_by_sport(
        self, 
        sport: str, 
        active_only: bool = True
    ) -> List[MarketInfo]:
        """
        Get all markets for a specific sport.
        
        Args:
            sport: Sport name (e.g., 'NFL', 'NBA')
            active_only: Only return markets that haven't resolved
            
        Returns:
            List of MarketInfo objects
        """
        query = "SELECT * FROM markets WHERE sport = ?"
        params = [sport]
        
        if active_only:
            query += " AND outcome IS NULL"
        
        query += " ORDER BY close_time"
        
        rows = self.db.execute_query(query, tuple(params))
        
        markets = []
        for row in rows:
            row_dict = dict(row)
            if row_dict['metadata']:
                row_dict['metadata'] = json.loads(row_dict['metadata'])
            
            # Remove database-specific fields
            row_dict.pop('created_at', None)
            row_dict.pop('updated_at', None)
            
            markets.append(MarketInfo(**row_dict))
        
        return markets
    
    def get_active_markets(self) -> List[MarketInfo]:
        """
        Get all active (unresolved) markets.
        
        Returns:
            List of active MarketInfo objects
        """
        current_time = int(datetime.now().timestamp())
        
        query = """
        SELECT * FROM markets 
        WHERE outcome IS NULL AND close_time > ?
        ORDER BY close_time
        """
        
        rows = self.db.execute_query(query, (current_time,))
        
        markets = []
        for row in rows:
            row_dict = dict(row)
            if row_dict['metadata']:
                row_dict['metadata'] = json.loads(row_dict['metadata'])
            
            # Remove database-specific fields
            row_dict.pop('created_at', None)
            row_dict.pop('updated_at', None)
            
            markets.append(MarketInfo(**row_dict))
        
        return markets
    
    def calculate_price_stats(
        self,
        market_id: str,
        window_hours: int = 24
    ) -> Dict[str, float]:
        """
        Calculate price statistics for a market over a time window.
        
        Args:
            market_id: Market identifier
            window_hours: Time window in hours
            
        Returns:
            Dictionary with statistics (mean, std, min, max, etc.)
        """
        end_time = int(datetime.now().timestamp())
        start_time = end_time - (window_hours * 3600)
        
        df = self.get_price_history(market_id, start_time, end_time)
        
        if df.empty:
            return {}
        
        stats = {
            'mean_price': df['last'].mean() if 'last' in df else None,
            'std_price': df['last'].std() if 'last' in df else None,
            'min_price': df['last'].min() if 'last' in df else None,
            'max_price': df['last'].max() if 'last' in df else None,
            'total_volume': df['volume'].sum() if 'volume' in df else None,
            'avg_spread': ((df['ask'] - df['bid']).mean() 
                          if 'ask' in df and 'bid' in df else None),
            'price_changes': len(df),
            'volatility': (df['last'].pct_change().std() * np.sqrt(252) 
                          if 'last' in df else None)
        }
        
        # Remove None values
        return {k: v for k, v in stats.items() if v is not None}
    
    def get_price_at_time(
        self,
        market_id: str,
        timestamp: int
    ) -> Optional[Dict[str, Any]]:
        """
        Get the price at or before a specific timestamp.
        
        Args:
            market_id: Market identifier
            timestamp: Unix timestamp
            
        Returns:
            Price data at the specified time or None
        """
        query = """
        SELECT * FROM market_prices 
        WHERE market_id = ? AND timestamp <= ?
        ORDER BY timestamp DESC
        LIMIT 1
        """
        
        rows = self.db.execute_query(query, (market_id, timestamp))
        
        if rows:
            return dict(rows[0])
        return None
    
    def clean_old_data(self, days_to_keep: int = 90):
        """
        Remove old price data to manage database size.
        
        Args:
            days_to_keep: Number of days of data to retain
        """
        cutoff_time = int((datetime.now() - timedelta(days=days_to_keep)).timestamp())
        
        query = "DELETE FROM market_prices WHERE timestamp < ?"
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (cutoff_time,))
            deleted_count = cursor.rowcount
            conn.commit()
        
        if deleted_count > 0:
            logger.info(f"Cleaned {deleted_count} old price records")
            self.db.vacuum()
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the stored data.
        
        Returns:
            Dictionary with database statistics
        """
        return self.db.get_database_stats()