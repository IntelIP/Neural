"""
Tests for buffer management.
"""

import asyncio
import pytest
from unittest.mock import Mock
import time

from neural.data_collection.buffer import (
    CircularBuffer,
    AsyncBuffer,
    OverflowStrategy,
    BufferStats
)
from neural.data_collection.exceptions import BufferOverflowError


class TestBufferStats:
    """Test BufferStats class."""
    
    def test_stats_initialization(self):
        stats = BufferStats()
        
        assert stats.total_added == 0
        assert stats.total_removed == 0
        assert stats.total_dropped == 0
        assert stats.current_size == 0
        assert stats.max_size == 0
        assert stats.overflow_count == 0
        assert stats.last_overflow_time is None
    
    def test_drop_rate_calculation(self):
        stats = BufferStats()
        
        # No items added
        assert stats.drop_rate() == 0.0
        
        # Some items added and dropped
        stats.total_added = 100
        stats.total_dropped = 10
        assert stats.drop_rate() == 10.0
        
        # All items dropped
        stats.total_dropped = 100
        assert stats.drop_rate() == 100.0
    
    def test_utilization_calculation(self):
        stats = BufferStats()
        
        # No capacity
        assert stats.utilization() == 0.0
        
        # Half full
        stats.max_size = 100
        stats.current_size = 50
        assert stats.utilization() == 50.0
        
        # Full
        stats.current_size = 100
        assert stats.utilization() == 100.0


class TestCircularBuffer:
    """Test CircularBuffer implementation."""
    
    def test_initialization(self):
        buffer = CircularBuffer(capacity=100)
        
        assert buffer.capacity == 100
        assert buffer.overflow_strategy == OverflowStrategy.DROP_OLDEST
        assert buffer.size() == 0
        assert buffer.is_empty() is True
        assert buffer.is_full() is False
    
    def test_invalid_capacity(self):
        with pytest.raises(ValueError):
            CircularBuffer(capacity=0)
        
        with pytest.raises(ValueError):
            CircularBuffer(capacity=-1)
    
    def test_push_and_pop(self):
        buffer = CircularBuffer(capacity=10)
        
        # Push items
        assert buffer.push("item1") is True
        assert buffer.push("item2") is True
        
        assert buffer.size() == 2
        
        # Pop items (FIFO)
        assert buffer.pop() == "item1"
        assert buffer.pop() == "item2"
        assert buffer.pop() is None
    
    def test_push_many(self):
        buffer = CircularBuffer(capacity=10)
        items = ["item1", "item2", "item3"]
        
        added = buffer.push_many(items)
        
        assert added == 3
        assert buffer.size() == 3
    
    def test_pop_many(self):
        buffer = CircularBuffer(capacity=10)
        buffer.push_many(["item1", "item2", "item3", "item4"])
        
        items = buffer.pop_many(2)
        
        assert len(items) == 2
        assert items == ["item1", "item2"]
        assert buffer.size() == 2
        
        # Request more than available
        items = buffer.pop_many(5)
        assert len(items) == 2
        assert items == ["item3", "item4"]
    
    def test_peek(self):
        buffer = CircularBuffer(capacity=10)
        
        assert buffer.peek() is None
        
        buffer.push("item1")
        buffer.push("item2")
        
        # Peek doesn't remove
        assert buffer.peek() == "item1"
        assert buffer.size() == 2
        assert buffer.peek() == "item1"
    
    def test_drop_oldest_strategy(self):
        buffer = CircularBuffer(
            capacity=3,
            overflow_strategy=OverflowStrategy.DROP_OLDEST
        )
        
        buffer.push("item1")
        buffer.push("item2")
        buffer.push("item3")
        
        # Buffer full, should drop oldest
        assert buffer.push("item4") is True
        
        # item1 should be dropped
        items = buffer.pop_many(3)
        assert items == ["item2", "item3", "item4"]
    
    def test_drop_newest_strategy(self):
        buffer = CircularBuffer(
            capacity=3,
            overflow_strategy=OverflowStrategy.DROP_NEWEST
        )
        
        buffer.push("item1")
        buffer.push("item2")
        buffer.push("item3")
        
        # Buffer full, should reject new
        assert buffer.push("item4") is False
        
        # Original items preserved
        items = buffer.pop_many(3)
        assert items == ["item1", "item2", "item3"]
    
    def test_expand_strategy(self):
        buffer = CircularBuffer(
            capacity=2,
            overflow_strategy=OverflowStrategy.EXPAND
        )
        
        buffer.push("item1")
        buffer.push("item2")
        
        # Should expand capacity
        assert buffer.push("item3") is True
        assert buffer.capacity > 2
        assert buffer.size() == 3
        
        # Test expansion limit
        for i in range(10):
            buffer.push(f"item{i+4}")
        
        # After max expansions, should drop oldest
        stats = buffer.get_stats()
        assert stats.total_dropped > 0
    
    def test_block_strategy(self):
        buffer = CircularBuffer(
            capacity=2,
            overflow_strategy=OverflowStrategy.BLOCK
        )
        
        buffer.push("item1")
        buffer.push("item2")
        
        # Should raise on overflow
        with pytest.raises(BufferOverflowError):
            buffer.push("item3")
    
    def test_overflow_callback(self):
        callback = Mock()
        buffer = CircularBuffer(
            capacity=2,
            overflow_strategy=OverflowStrategy.DROP_OLDEST,
            overflow_callback=callback
        )
        
        buffer.push("item1")
        buffer.push("item2")
        buffer.push("item3")  # Should trigger callback for item1
        
        callback.assert_called_once()
    
    def test_clear(self):
        buffer = CircularBuffer(capacity=10)
        buffer.push_many(["item1", "item2", "item3"])
        
        cleared = buffer.clear()
        
        assert cleared == 3
        assert buffer.size() == 0
        assert buffer.is_empty() is True
    
    def test_thread_safety(self):
        import threading
        
        buffer = CircularBuffer(capacity=100)
        results = []
        
        def push_items():
            for i in range(50):
                buffer.push(f"thread1_{i}")
        
        def pop_items():
            for _ in range(25):
                item = buffer.pop()
                if item:
                    results.append(item)
        
        threads = [
            threading.Thread(target=push_items),
            threading.Thread(target=pop_items)
        ]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        # Should not crash and maintain consistency
        assert buffer.size() >= 0
        assert buffer.size() <= buffer.capacity


class TestAsyncBuffer:
    """Test AsyncBuffer implementation."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        buffer = AsyncBuffer(capacity=100)
        
        assert buffer.capacity == 100
        assert buffer.overflow_strategy == OverflowStrategy.BLOCK
        assert buffer.size() == 0
        assert buffer.is_empty() is True
        assert buffer.is_full() is False
    
    @pytest.mark.asyncio
    async def test_invalid_strategy(self):
        with pytest.raises(ValueError):
            AsyncBuffer(
                capacity=10,
                overflow_strategy=OverflowStrategy.DROP_OLDEST
            )
    
    @pytest.mark.asyncio
    async def test_push_and_pop(self):
        buffer = AsyncBuffer(capacity=10)
        
        # Push items
        assert await buffer.push("item1") is True
        assert await buffer.push("item2") is True
        
        assert buffer.size() == 2
        
        # Pop items
        assert await buffer.pop() == "item1"
        assert await buffer.pop() == "item2"
        
        # Pop with timeout on empty
        item = await buffer.pop(timeout=0.1)
        assert item is None
    
    @pytest.mark.asyncio
    async def test_pop_many(self):
        buffer = AsyncBuffer(capacity=10)
        
        for i in range(5):
            await buffer.push(f"item{i}")
        
        items = await buffer.pop_many(3, timeout=0.1)
        
        assert len(items) == 3
        assert items == ["item0", "item1", "item2"]
    
    @pytest.mark.asyncio
    async def test_block_strategy(self):
        buffer = AsyncBuffer(
            capacity=2,
            overflow_strategy=OverflowStrategy.BLOCK
        )
        
        await buffer.push("item1")
        await buffer.push("item2")
        
        # Should block (timeout)
        result = await buffer.push("item3", timeout=0.1)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_drop_newest_strategy(self):
        buffer = AsyncBuffer(
            capacity=2,
            overflow_strategy=OverflowStrategy.DROP_NEWEST
        )
        
        await buffer.push("item1")
        await buffer.push("item2")
        
        # Should drop new item
        result = await buffer.push("item3")
        assert result is False
        
        stats = buffer.get_stats()
        assert stats.total_dropped == 1
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        buffer = AsyncBuffer(capacity=100)
        
        async def producer():
            for i in range(50):
                await buffer.push(f"item{i}")
                await asyncio.sleep(0.001)
        
        async def consumer():
            items = []
            for _ in range(50):
                item = await buffer.pop(timeout=0.1)
                if item:
                    items.append(item)
                await asyncio.sleep(0.001)
            return items
        
        # Run concurrently
        producer_task = asyncio.create_task(producer())
        consumer_task = asyncio.create_task(consumer())
        
        await producer_task
        consumed = await consumer_task
        
        # Should consume all produced items
        assert len(consumed) > 0
        assert buffer.size() >= 0