"""
Unit tests for MarketDataStore class.

Tests database operations, data storage/retrieval, and performance.
"""

import pytest
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path

from neural.analysis.market_data import MarketDataStore, PriceUpdate, MarketInfo
from neural.analysis.database import DatabaseManager


class TestMarketDataStore:
    """Test suite for MarketDataStore."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        yield db_path
        
        # Cleanup
        if os.path.exists(db_path):
            os.remove(db_path)
    
    @pytest.fixture
    def store(self, temp_db):
        """Create MarketDataStore instance with temp database."""
        return MarketDataStore(db_path=temp_db)
    
    def test_initialization(self, store):
        """Test store initialization."""
        assert store is not None
        assert store.db is not None
    
    def test_store_price_update(self, store):
        """Test storing a single price update."""
        update = PriceUpdate(
            market_id="TEST_MARKET",
            timestamp=int(datetime.now().timestamp()),
            bid=0.45,
            ask=0.47,
            last=0.46,
            volume=1000,
            open_interest=5000
        )
        
        store.store_price_update(update)
        
        # Retrieve and verify
        latest = store.get_latest_price("TEST_MARKET")
        assert latest is not None
        assert latest['market_id'] == "TEST_MARKET"
        assert latest['bid'] == pytest.approx(0.45, rel=1e-3)
        assert latest['ask'] == pytest.approx(0.47, rel=1e-3)
        assert latest['last'] == pytest.approx(0.46, rel=1e-3)
        assert latest['volume'] == 1000
    
    def test_store_price_updates_bulk(self, store):
        """Test bulk price update storage."""
        base_time = int(datetime.now().timestamp())
        updates = []
        
        for i in range(100):
            updates.append(PriceUpdate(
                market_id="BULK_TEST",
                timestamp=base_time + i,
                bid=0.40 + i * 0.001,
                ask=0.42 + i * 0.001,
                last=0.41 + i * 0.001,
                volume=100 * i
            ))
        
        store.store_price_updates_bulk(updates)
        
        # Retrieve history
        history = store.get_price_history("BULK_TEST")
        assert len(history) == 100
        assert history.iloc[0]['bid'] == pytest.approx(0.40, rel=1e-3)
        assert history.iloc[-1]['bid'] == pytest.approx(0.499, rel=1e-3)
    
    def test_get_price_history_with_time_range(self, store):
        """Test retrieving price history with time filters."""
        base_time = datetime.now()
        
        # Store updates across 3 hours
        updates = []
        for hour in range(3):
            for minute in range(0, 60, 10):
                timestamp = base_time + timedelta(hours=hour, minutes=minute)
                updates.append(PriceUpdate(
                    market_id="TIME_TEST",
                    timestamp=int(timestamp.timestamp()),
                    last=0.50 + hour * 0.01
                ))
        
        store.store_price_updates_bulk(updates)
        
        # Get middle hour only
        start = int((base_time + timedelta(hours=1)).timestamp())
        end = int((base_time + timedelta(hours=2)).timestamp())
        
        history = store.get_price_history("TIME_TEST", start, end)
        assert len(history) == 6  # 6 updates per hour
        assert all(0.51 <= row['last'] <= 0.52 for _, row in history.iterrows())
    
    def test_store_and_retrieve_market_info(self, store):
        """Test market metadata storage and retrieval."""
        market = MarketInfo(
            market_id="NFL_TEST",
            event_ticker="NFL_GAME_123",
            event_name="Patriots vs Bills",
            sport="NFL",
            close_time=int((datetime.now() + timedelta(hours=24)).timestamp()),
            metadata={"home_team": "Patriots", "away_team": "Bills"}
        )
        
        store.store_market_info(market)
        
        # Retrieve
        retrieved = store.get_market_info("NFL_TEST")
        assert retrieved is not None
        assert retrieved.event_ticker == "NFL_GAME_123"
        assert retrieved.event_name == "Patriots vs Bills"
        assert retrieved.sport == "NFL"
        assert retrieved.metadata["home_team"] == "Patriots"
    
    def test_get_markets_by_sport(self, store):
        """Test filtering markets by sport."""
        # Store multiple markets
        sports = ["NFL", "NBA", "NFL", "MLB"]
        for i, sport in enumerate(sports):
            market = MarketInfo(
                market_id=f"MARKET_{i}",
                event_ticker=f"TICKER_{i}",
                event_name=f"Game {i}",
                sport=sport,
                close_time=int((datetime.now() + timedelta(hours=i)).timestamp())
            )
            store.store_market_info(market)
        
        # Get NFL markets
        nfl_markets = store.get_markets_by_sport("NFL")
        assert len(nfl_markets) == 2
        assert all(m.sport == "NFL" for m in nfl_markets)
    
    def test_calculate_price_stats(self, store):
        """Test price statistics calculation."""
        base_time = int(datetime.now().timestamp()) - 10 * 3600  # Start 10 hours ago
        
        # Create price series with known statistics
        updates = []
        prices = [0.40, 0.42, 0.45, 0.43, 0.44, 0.46, 0.45, 0.47, 0.48, 0.45]
        
        for i, price in enumerate(prices):
            updates.append(PriceUpdate(
                market_id="STATS_TEST",
                timestamp=base_time + i * 3600,  # Hourly updates
                bid=price - 0.01,
                ask=price + 0.01,
                last=price,
                volume=1000
            ))
        
        store.store_price_updates_bulk(updates)
        
        # Calculate stats
        stats = store.calculate_price_stats("STATS_TEST", window_hours=24)
        
        assert 'mean_price' in stats
        assert 'std_price' in stats
        assert 'min_price' in stats
        assert 'max_price' in stats
        assert 'avg_spread' in stats
        
        assert stats['mean_price'] == pytest.approx(0.445, rel=1e-2)
        assert stats['min_price'] == pytest.approx(0.40, rel=1e-3)
        assert stats['max_price'] == pytest.approx(0.48, rel=1e-3)
        assert stats['avg_spread'] == pytest.approx(0.02, rel=1e-2)
    
    def test_get_active_markets(self, store):
        """Test retrieving active markets."""
        current_time = datetime.now()
        
        # Store markets with different states
        markets = [
            MarketInfo(
                market_id="ACTIVE_1",
                event_ticker="T1",
                event_name="Active Game 1",
                sport="NFL",
                close_time=int((current_time + timedelta(hours=2)).timestamp()),
                outcome=None  # Not resolved
            ),
            MarketInfo(
                market_id="RESOLVED_1",
                event_ticker="T2",
                event_name="Resolved Game",
                sport="NFL",
                close_time=int((current_time - timedelta(hours=2)).timestamp()),
                outcome=1  # Resolved to YES
            ),
            MarketInfo(
                market_id="ACTIVE_2",
                event_ticker="T3",
                event_name="Active Game 2",
                sport="NBA",
                close_time=int((current_time + timedelta(hours=5)).timestamp()),
                outcome=None  # Not resolved
            )
        ]
        
        for market in markets:
            store.store_market_info(market)
        
        # Get active markets
        active = store.get_active_markets()
        assert len(active) == 2
        assert all(m.outcome is None for m in active)
        assert all("ACTIVE" in m.market_id for m in active)
    
    def test_get_price_at_time(self, store):
        """Test getting price at specific timestamp."""
        base_time = int(datetime.now().timestamp())
        
        # Store prices at different times
        updates = [
            PriceUpdate("TIME_PRICE", base_time + 100, last=0.40),
            PriceUpdate("TIME_PRICE", base_time + 200, last=0.45),
            PriceUpdate("TIME_PRICE", base_time + 300, last=0.50),
        ]
        
        store.store_price_updates_bulk(updates)
        
        # Get price at different times
        price_150 = store.get_price_at_time("TIME_PRICE", base_time + 150)
        assert price_150['last'] == pytest.approx(0.40, rel=1e-3)  # Should get price from time 100
        
        price_250 = store.get_price_at_time("TIME_PRICE", base_time + 250)
        assert price_250['last'] == pytest.approx(0.45, rel=1e-3)  # Should get price from time 200
        
        price_350 = store.get_price_at_time("TIME_PRICE", base_time + 350)
        assert price_350['last'] == pytest.approx(0.50, rel=1e-3)  # Should get price from time 300
    
    def test_clean_old_data(self, store):
        """Test cleaning old price data."""
        current_time = datetime.now()
        
        # Store old and new data
        old_updates = []
        new_updates = []
        
        # Old data (100 days ago)
        old_time = current_time - timedelta(days=100)
        for i in range(50):
            old_updates.append(PriceUpdate(
                market_id="CLEAN_TEST",
                timestamp=int((old_time + timedelta(hours=i)).timestamp()),
                last=0.40
            ))
        
        # Recent data (1 day ago)
        recent_time = current_time - timedelta(days=1)
        for i in range(50):
            new_updates.append(PriceUpdate(
                market_id="CLEAN_TEST",
                timestamp=int((recent_time + timedelta(hours=i)).timestamp()),
                last=0.50
            ))
        
        store.store_price_updates_bulk(old_updates + new_updates)
        
        # Verify all data stored
        all_history = store.get_price_history("CLEAN_TEST")
        assert len(all_history) == 100
        
        # Clean old data (keep 30 days)
        store.clean_old_data(days_to_keep=30)
        
        # Verify old data removed
        remaining_history = store.get_price_history("CLEAN_TEST")
        assert len(remaining_history) == 50
        assert all(pytest.approx(row['last'], rel=1e-3) == 0.50 for _, row in remaining_history.iterrows())
    
    def test_database_stats(self, store):
        """Test getting database statistics."""
        # Store some data
        for i in range(10):
            store.store_price_update(PriceUpdate(
                market_id=f"STATS_{i}",
                timestamp=int(datetime.now().timestamp()),
                last=0.50
            ))
        
        stats = store.get_database_stats()
        
        assert 'market_prices_count' in stats
        assert 'markets_count' in stats
        assert 'file_size_mb' in stats
        assert stats['market_prices_count'] >= 10