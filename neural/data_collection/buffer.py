"""
Buffer management for the Neural SDK Data Collection Infrastructure.

This module provides efficient buffer implementations for managing
high-throughput data streams with different overflow strategies and
memory management techniques.
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Any, Callable, Deque, Generic, List, Optional, TypeVar
import logging

from neural.data_collection.exceptions import BufferOverflowError


# Configure module logger
logger = logging.getLogger(__name__)

# Generic type for buffer items
T = TypeVar('T')


class OverflowStrategy(Enum):
    """
    Strategies for handling buffer overflow.
    
    Attributes:
        DROP_OLDEST: Remove oldest items when buffer is full
        DROP_NEWEST: Reject new items when buffer is full
        BLOCK: Block until space is available
        EXPAND: Dynamically expand buffer size
    """
    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"
    BLOCK = "block"
    EXPAND = "expand"


@dataclass
class BufferStats:
    """
    Statistics for buffer operations.
    
    Attributes:
        total_added: Total items added to buffer
        total_removed: Total items removed from buffer
        total_dropped: Total items dropped due to overflow
        current_size: Current number of items in buffer
        max_size: Maximum buffer capacity
        overflow_count: Number of overflow events
        last_overflow_time: Timestamp of last overflow
    """
    total_added: int = 0
    total_removed: int = 0
    total_dropped: int = 0
    current_size: int = 0
    max_size: int = 0
    overflow_count: int = 0
    last_overflow_time: Optional[float] = None
    
    def drop_rate(self) -> float:
        """Calculate the drop rate as a percentage."""
        if self.total_added == 0:
            return 0.0
        return (self.total_dropped / self.total_added) * 100
    
    def utilization(self) -> float:
        """Calculate buffer utilization as a percentage."""
        if self.max_size == 0:
            return 0.0
        return (self.current_size / self.max_size) * 100


class CircularBuffer(Generic[T]):
    """
    Thread-safe circular buffer implementation.
    
    This class provides a fixed-size circular buffer that efficiently
    handles high-throughput data streams with configurable overflow
    strategies.
    
    Attributes:
        capacity: Maximum number of items the buffer can hold
        overflow_strategy: Strategy for handling overflow
        buffer: Internal deque storing items
        stats: Buffer operation statistics
        
    Example:
        >>> buffer = CircularBuffer(capacity=1000)
        >>> buffer.push({"data": "example"})
        >>> item = buffer.pop()
    """
    
    def __init__(
        self,
        capacity: int,
        overflow_strategy: OverflowStrategy = OverflowStrategy.DROP_OLDEST,
        overflow_callback: Optional[Callable[[T], None]] = None
    ):
        """
        Initialize the circular buffer.
        
        Args:
            capacity: Maximum buffer size
            overflow_strategy: How to handle buffer overflow
            overflow_callback: Optional callback when items are dropped
        """
        if capacity <= 0:
            raise ValueError("Buffer capacity must be positive")
        
        self.capacity = capacity
        self.overflow_strategy = overflow_strategy
        self.overflow_callback = overflow_callback
        
        # Use deque with maxlen for efficient circular buffer
        # Note: maxlen only works with DROP_OLDEST strategy
        if overflow_strategy == OverflowStrategy.DROP_OLDEST:
            self._buffer: Deque[T] = deque(maxlen=capacity)
        else:
            self._buffer: Deque[T] = deque()
        
        # Thread safety
        self._lock = Lock()
        
        # Statistics
        self.stats = BufferStats(max_size=capacity)
        
        # For EXPAND strategy
        self._expansion_factor = 1.5
        self._max_expansions = 3
        self._expansion_count = 0
        
        logger.debug(
            f"Initialized CircularBuffer: capacity={capacity}, "
            f"strategy={overflow_strategy.value}"
        )
    
    def push(self, item: T) -> bool:
        """
        Add an item to the buffer.
        
        Args:
            item: Item to add
            
        Returns:
            bool: True if item was added, False if dropped
            
        Raises:
            BufferOverflowError: If BLOCK strategy and buffer is full
            
        Example:
            >>> success = buffer.push({"id": 1, "data": "test"})
        """
        with self._lock:
            current_size = len(self._buffer)
            
            # Check if buffer is full
            if current_size >= self.capacity:
                return self._handle_overflow(item)
            
            # Add item to buffer
            self._buffer.append(item)
            self.stats.total_added += 1
            self.stats.current_size = len(self._buffer)
            
            return True
    
    def push_many(self, items: List[T]) -> int:
        """
        Add multiple items to the buffer.
        
        Args:
            items: List of items to add
            
        Returns:
            int: Number of items successfully added
            
        Example:
            >>> added = buffer.push_many([item1, item2, item3])
        """
        added_count = 0
        
        for item in items:
            if self.push(item):
                added_count += 1
        
        return added_count
    
    def pop(self) -> Optional[T]:
        """
        Remove and return the oldest item from the buffer.
        
        Returns:
            Oldest item or None if buffer is empty
            
        Example:
            >>> item = buffer.pop()
            >>> if item:
            ...     process(item)
        """
        with self._lock:
            if not self._buffer:
                return None
            
            item = self._buffer.popleft()
            self.stats.total_removed += 1
            self.stats.current_size = len(self._buffer)
            
            return item
    
    def pop_many(self, count: int) -> List[T]:
        """
        Remove and return multiple items from the buffer.
        
        Args:
            count: Maximum number of items to remove
            
        Returns:
            List of items (may be fewer than requested)
            
        Example:
            >>> items = buffer.pop_many(10)
        """
        items = []
        
        for _ in range(count):
            item = self.pop()
            if item is None:
                break
            items.append(item)
        
        return items
    
    def peek(self) -> Optional[T]:
        """
        Return the oldest item without removing it.
        
        Returns:
            Oldest item or None if buffer is empty
            
        Example:
            >>> item = buffer.peek()
        """
        with self._lock:
            if not self._buffer:
                return None
            return self._buffer[0]
    
    def _handle_overflow(self, item: T) -> bool:
        """
        Handle buffer overflow based on strategy.
        
        Args:
            item: Item attempting to be added
            
        Returns:
            bool: True if item was added, False if dropped
        """
        self.stats.overflow_count += 1
        self.stats.last_overflow_time = time.time()
        
        if self.overflow_strategy == OverflowStrategy.DROP_OLDEST:
            # With maxlen deque, this happens automatically
            dropped_item = None
            if len(self._buffer) >= self.capacity:
                dropped_item = self._buffer[0]  # Item that will be dropped
            
            self._buffer.append(item)
            self.stats.total_added += 1
            self.stats.total_dropped += 1
            
            # Call overflow callback if provided
            if self.overflow_callback and dropped_item:
                try:
                    self.overflow_callback(dropped_item)
                except Exception as e:
                    logger.error(f"Overflow callback error: {e}")
            
            return True
        
        elif self.overflow_strategy == OverflowStrategy.DROP_NEWEST:
            # Reject the new item
            self.stats.total_dropped += 1
            
            # Call overflow callback if provided
            if self.overflow_callback:
                try:
                    self.overflow_callback(item)
                except Exception as e:
                    logger.error(f"Overflow callback error: {e}")
            
            logger.debug("Buffer full, dropping new item")
            return False
        
        elif self.overflow_strategy == OverflowStrategy.EXPAND:
            # Dynamically expand buffer if allowed
            if self._expansion_count < self._max_expansions:
                new_capacity = int(self.capacity * self._expansion_factor)
                self.capacity = new_capacity
                self._expansion_count += 1
                
                logger.info(
                    f"Buffer expanded to {new_capacity} "
                    f"(expansion {self._expansion_count}/{self._max_expansions})"
                )
                
                # Update stats
                self.stats.max_size = new_capacity
                
                # Add the item
                self._buffer.append(item)
                self.stats.total_added += 1
                self.stats.current_size = len(self._buffer)
                
                return True
            else:
                # Max expansions reached, drop oldest
                dropped_item = self._buffer.popleft()
                self._buffer.append(item)
                self.stats.total_added += 1
                self.stats.total_dropped += 1
                
                if self.overflow_callback and dropped_item:
                    try:
                        self.overflow_callback(dropped_item)
                    except Exception as e:
                        logger.error(f"Overflow callback error: {e}")
                
                return True
        
        elif self.overflow_strategy == OverflowStrategy.BLOCK:
            # This strategy doesn't work well with synchronous push
            # Would need async version for proper blocking
            raise BufferOverflowError(
                "Buffer full with BLOCK strategy",
                details={"capacity": self.capacity, "size": len(self._buffer)}
            )
        
        return False
    
    def clear(self) -> int:
        """
        Remove all items from the buffer.
        
        Returns:
            int: Number of items cleared
            
        Example:
            >>> cleared = buffer.clear()
            >>> print(f"Cleared {cleared} items")
        """
        with self._lock:
            count = len(self._buffer)
            self._buffer.clear()
            self.stats.current_size = 0
            
            logger.debug(f"Buffer cleared, removed {count} items")
            return count
    
    def size(self) -> int:
        """
        Get current number of items in the buffer.
        
        Returns:
            int: Current buffer size
            
        Example:
            >>> current = buffer.size()
        """
        with self._lock:
            return len(self._buffer)
    
    def is_empty(self) -> bool:
        """
        Check if buffer is empty.
        
        Returns:
            bool: True if buffer is empty
        """
        with self._lock:
            return len(self._buffer) == 0
    
    def is_full(self) -> bool:
        """
        Check if buffer is at capacity.
        
        Returns:
            bool: True if buffer is full
        """
        with self._lock:
            return len(self._buffer) >= self.capacity
    
    def get_stats(self) -> BufferStats:
        """
        Get buffer statistics.
        
        Returns:
            BufferStats object with current statistics
            
        Example:
            >>> stats = buffer.get_stats()
            >>> print(f"Drop rate: {stats.drop_rate():.2f}%")
        """
        with self._lock:
            self.stats.current_size = len(self._buffer)
            return self.stats
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size()
    
    def __repr__(self) -> str:
        """Return string representation of the buffer."""
        return (
            f"CircularBuffer("
            f"size={self.size()}/{self.capacity}, "
            f"strategy={self.overflow_strategy.value})"
        )


class AsyncBuffer(Generic[T]):
    """
    Asynchronous buffer implementation using asyncio.Queue.
    
    This class provides an async-friendly buffer that supports
    blocking operations and is suitable for use in async pipelines.
    
    Attributes:
        capacity: Maximum buffer size
        queue: Internal asyncio.Queue
        stats: Buffer operation statistics
        
    Example:
        >>> buffer = AsyncBuffer(capacity=1000)
        >>> await buffer.push(item)
        >>> item = await buffer.pop()
    """
    
    def __init__(
        self,
        capacity: int,
        overflow_strategy: OverflowStrategy = OverflowStrategy.BLOCK
    ):
        """
        Initialize the async buffer.
        
        Args:
            capacity: Maximum buffer size
            overflow_strategy: How to handle overflow (BLOCK or DROP_NEWEST)
        """
        if capacity <= 0:
            raise ValueError("Buffer capacity must be positive")
        
        if overflow_strategy not in [OverflowStrategy.BLOCK, OverflowStrategy.DROP_NEWEST]:
            raise ValueError(
                "AsyncBuffer only supports BLOCK and DROP_NEWEST strategies"
            )
        
        self.capacity = capacity
        self.overflow_strategy = overflow_strategy
        
        # Use asyncio.Queue for async operations
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=capacity)
        
        # Statistics
        self.stats = BufferStats(max_size=capacity)
        
        logger.debug(
            f"Initialized AsyncBuffer: capacity={capacity}, "
            f"strategy={overflow_strategy.value}"
        )
    
    async def push(
        self,
        item: T,
        timeout: Optional[float] = None
    ) -> bool:
        """
        Add an item to the buffer.
        
        Args:
            item: Item to add
            timeout: Maximum time to wait if blocking
            
        Returns:
            bool: True if item was added, False if dropped
            
        Example:
            >>> success = await buffer.push(item, timeout=5.0)
        """
        try:
            if self.overflow_strategy == OverflowStrategy.BLOCK:
                # Block until space is available
                if timeout:
                    await asyncio.wait_for(
                        self._queue.put(item),
                        timeout=timeout
                    )
                else:
                    await self._queue.put(item)
            else:
                # DROP_NEWEST strategy
                try:
                    self._queue.put_nowait(item)
                except asyncio.QueueFull:
                    self.stats.total_dropped += 1
                    self.stats.overflow_count += 1
                    self.stats.last_overflow_time = time.time()
                    return False
            
            self.stats.total_added += 1
            self.stats.current_size = self._queue.qsize()
            return True
            
        except asyncio.TimeoutError:
            self.stats.total_dropped += 1
            self.stats.overflow_count += 1
            self.stats.last_overflow_time = time.time()
            return False
        except Exception as e:
            logger.error(f"Error pushing to buffer: {e}")
            return False
    
    async def pop(
        self,
        timeout: Optional[float] = None
    ) -> Optional[T]:
        """
        Remove and return an item from the buffer.
        
        Args:
            timeout: Maximum time to wait if buffer is empty
            
        Returns:
            Item or None if timeout/empty
            
        Example:
            >>> item = await buffer.pop(timeout=1.0)
        """
        try:
            if timeout:
                item = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=timeout
                )
            else:
                item = await self._queue.get()
            
            self.stats.total_removed += 1
            self.stats.current_size = self._queue.qsize()
            return item
            
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Error popping from buffer: {e}")
            return None
    
    async def pop_many(
        self,
        count: int,
        timeout: Optional[float] = None
    ) -> List[T]:
        """
        Remove and return multiple items from the buffer.
        
        Args:
            count: Maximum number of items to remove
            timeout: Maximum time to wait for each item
            
        Returns:
            List of items (may be fewer than requested)
            
        Example:
            >>> items = await buffer.pop_many(10, timeout=0.1)
        """
        items = []
        
        for _ in range(count):
            item = await self.pop(timeout=timeout)
            if item is None:
                break
            items.append(item)
        
        return items
    
    def size(self) -> int:
        """
        Get current number of items in the buffer.
        
        Returns:
            int: Current buffer size
        """
        return self._queue.qsize()
    
    def is_empty(self) -> bool:
        """
        Check if buffer is empty.
        
        Returns:
            bool: True if buffer is empty
        """
        return self._queue.empty()
    
    def is_full(self) -> bool:
        """
        Check if buffer is at capacity.
        
        Returns:
            bool: True if buffer is full
        """
        return self._queue.full()
    
    def get_stats(self) -> BufferStats:
        """
        Get buffer statistics.
        
        Returns:
            BufferStats object with current statistics
        """
        self.stats.current_size = self._queue.qsize()
        return self.stats
    
    def __repr__(self) -> str:
        """Return string representation of the buffer."""
        return (
            f"AsyncBuffer("
            f"size={self.size()}/{self.capacity}, "
            f"strategy={self.overflow_strategy.value})"
        )