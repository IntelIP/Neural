"""
Event Buffer - High-performance ring buffer for event queueing
Based on LMAX Disruptor pattern for lock-free message passing
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class Priority(Enum):
    """Event priority levels"""
    CRITICAL = 0  # Immediate processing
    HIGH = 1      # Process within 100ms
    NORMAL = 2    # Process within 1s
    LOW = 3       # Best effort

    def __lt__(self, other):
        return self.value < other.value


@dataclass
class BufferedEvent:
    """Event wrapper with metadata"""
    event_type: str
    data: Dict[str, Any]
    priority: Priority
    timestamp: datetime = field(default_factory=datetime.now)
    sequence: int = 0
    attempts: int = 0
    destinations: List[str] = field(default_factory=list)
    
    def __lt__(self, other):
        """Priority comparison for heap operations"""
        # Higher priority (lower value) comes first
        if self.priority != other.priority:
            return self.priority < other.priority
        # For same priority, earlier timestamp comes first
        return self.timestamp < other.timestamp


class EventBuffer:
    """
    High-performance ring buffer for event queueing
    
    Features:
    - Lock-free single writer, multiple readers
    - Priority-based processing
    - Backpressure handling
    - Event batching support
    """
    
    def __init__(self, size: int = 10000):
        """
        Initialize event buffer
        
        Args:
            size: Buffer capacity (must be power of 2 for optimization)
        """
        # Ensure size is power of 2 for bitwise operations
        self.size = 1 << (size - 1).bit_length()
        self.mask = self.size - 1
        
        # Pre-allocate buffer
        self.buffer: List[Optional[BufferedEvent]] = [None] * self.size
        
        # Positions (using atomic operations)
        self.write_position = 0
        self.read_positions: Dict[str, int] = {}  # Per consumer
        
        # Priority queues for urgent events
        self.priority_queues: Dict[Priority, List[BufferedEvent]] = {
            p: [] for p in Priority
        }
        
        # Statistics
        self.stats = {
            "events_written": 0,
            "events_read": 0,
            "events_dropped": 0,
            "buffer_overruns": 0
        }
        
        # Backpressure control
        self.high_water_mark = int(self.size * 0.8)
        self.low_water_mark = int(self.size * 0.5)
        self.backpressure_active = False
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        logger.info(f"EventBuffer initialized with size {self.size}")
    
    async def write(self, event: BufferedEvent) -> bool:
        """
        Write event to buffer
        
        Args:
            event: Event to write
            
        Returns:
            True if written successfully, False if dropped
        """
        try:
            # Check for critical events that bypass buffer
            if event.priority == Priority.CRITICAL:
                self.priority_queues[Priority.CRITICAL].append(event)
                logger.debug(f"Critical event queued: {event.event_type}")
                return True
            
            # Check backpressure
            if self.backpressure_active and event.priority == Priority.LOW:
                self.stats["events_dropped"] += 1
                logger.warning(f"Dropping low priority event due to backpressure: {event.event_type}")
                return False
            
            # Calculate next position
            next_pos = (self.write_position + 1) & self.mask
            
            # Check if buffer is full (would overwrite unread data)
            min_read_pos = min(self.read_positions.values()) if self.read_positions else 0
            if next_pos == min_read_pos:
                self.stats["buffer_overruns"] += 1
                
                if event.priority in [Priority.HIGH, Priority.NORMAL]:
                    # Try priority queue for important events
                    self.priority_queues[event.priority].append(event)
                    logger.warning(f"Buffer full, queued {event.priority.name} event: {event.event_type}")
                    return True
                else:
                    self.stats["events_dropped"] += 1
                    logger.warning(f"Buffer full, dropping event: {event.event_type}")
                    return False
            
            # Write to buffer
            event.sequence = self.write_position
            self.buffer[self.write_position] = event
            self.write_position = next_pos
            self.stats["events_written"] += 1
            
            # Check water marks
            buffer_usage = self._calculate_usage()
            if buffer_usage > self.high_water_mark and not self.backpressure_active:
                self.backpressure_active = True
                logger.warning(f"Backpressure activated at {buffer_usage}/{self.size}")
            elif buffer_usage < self.low_water_mark and self.backpressure_active:
                self.backpressure_active = False
                logger.info("Backpressure deactivated")
            
            return True
            
        except Exception as e:
            logger.error(f"Error writing event: {e}")
            return False
    
    async def read(self, consumer_id: str, batch_size: int = 1) -> List[BufferedEvent]:
        """
        Read events from buffer
        
        Args:
            consumer_id: Unique consumer identifier
            batch_size: Number of events to read
            
        Returns:
            List of events (may be less than batch_size)
        """
        events = []
        
        try:
            # Initialize read position if new consumer
            if consumer_id not in self.read_positions:
                self.read_positions[consumer_id] = 0
                logger.info(f"New consumer registered: {consumer_id}")
            
            # Check priority queues first
            for priority in [Priority.CRITICAL, Priority.HIGH]:
                if self.priority_queues[priority]:
                    # Get all priority events up to batch size
                    priority_events = self.priority_queues[priority][:batch_size]
                    self.priority_queues[priority] = self.priority_queues[priority][batch_size:]
                    events.extend(priority_events)
                    
                    if len(events) >= batch_size:
                        return events[:batch_size]
            
            # Read from main buffer
            read_pos = self.read_positions[consumer_id]
            remaining = batch_size - len(events)
            
            while remaining > 0 and read_pos != self.write_position:
                event = self.buffer[read_pos]
                if event is not None:
                    events.append(event)
                    self.stats["events_read"] += 1
                    remaining -= 1
                
                read_pos = (read_pos + 1) & self.mask
            
            # Update read position
            self.read_positions[consumer_id] = read_pos
            
            return events
            
        except Exception as e:
            logger.error(f"Error reading events for {consumer_id}: {e}")
            return events
    
    async def read_blocking(self, consumer_id: str, batch_size: int = 1, 
                           timeout: float = 1.0) -> List[BufferedEvent]:
        """
        Read events with blocking wait for data
        
        Args:
            consumer_id: Unique consumer identifier
            batch_size: Number of events to read
            timeout: Maximum wait time in seconds
            
        Returns:
            List of events
        """
        end_time = asyncio.get_event_loop().time() + timeout
        events = []
        
        while len(events) < batch_size:
            batch = await self.read(consumer_id, batch_size - len(events))
            events.extend(batch)
            
            if len(events) >= batch_size:
                break
            
            # Check timeout
            remaining_time = end_time - asyncio.get_event_loop().time()
            if remaining_time <= 0:
                break
            
            # Small sleep to avoid busy waiting
            await asyncio.sleep(min(0.01, remaining_time))
        
        return events
    
    def _calculate_usage(self) -> int:
        """Calculate buffer usage"""
        min_read_pos = min(self.read_positions.values()) if self.read_positions else 0
        
        if self.write_position >= min_read_pos:
            return self.write_position - min_read_pos
        else:
            return self.size - min_read_pos + self.write_position
    
    def register_handler(self, event_type: str, handler: Callable):
        """Register event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
        logger.info(f"Registered handler for {event_type}")
    
    async def process_events(self, consumer_id: str, batch_size: int = 10):
        """
        Process events using registered handlers
        
        Args:
            consumer_id: Consumer identifier
            batch_size: Events to process per iteration
        """
        while True:
            try:
                events = await self.read_blocking(consumer_id, batch_size, timeout=0.1)
                
                for event in events:
                    handlers = self.event_handlers.get(event.event_type, [])
                    for handler in handlers:
                        try:
                            await handler(event)
                        except Exception as e:
                            logger.error(f"Handler error for {event.event_type}: {e}")
                            event.attempts += 1
                            
                            # Retry logic for failed events
                            if event.attempts < 3 and event.priority != Priority.LOW:
                                event.priority = Priority.HIGH
                                await self.write(event)
                
                # Small yield to prevent hogging
                await asyncio.sleep(0)
                
            except Exception as e:
                logger.error(f"Error in event processing loop: {e}")
                await asyncio.sleep(1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        return {
            **self.stats,
            "buffer_size": self.size,
            "buffer_usage": self._calculate_usage(),
            "backpressure": self.backpressure_active,
            "consumers": list(self.read_positions.keys()),
            "priority_queue_sizes": {
                p.name: len(q) for p, q in self.priority_queues.items()
            }
        }
    
    def reset_consumer(self, consumer_id: str):
        """Reset consumer read position"""
        if consumer_id in self.read_positions:
            self.read_positions[consumer_id] = 0
            logger.info(f"Reset consumer position: {consumer_id}")