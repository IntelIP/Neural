"""
Flow Controller - Credit-based flow control for streaming data
Prevents message flooding and ensures smooth data flow
Based on TCP flow control principles adapted for WebSockets
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import deque
import threading

logger = logging.getLogger(__name__)


class FlowState(Enum):
    """Flow control states"""
    FLOWING = "flowing"      # Normal flow
    THROTTLED = "throttled"  # Reduced flow
    PAUSED = "paused"       # Flow paused
    RESUMING = "resuming"   # Resuming from pause


@dataclass
class FlowMetrics:
    """Metrics for flow control"""
    messages_received: int = 0
    messages_processed: int = 0
    messages_dropped: int = 0
    credits_granted: int = 0
    credits_consumed: int = 0
    flow_pauses: int = 0
    flow_resumes: int = 0
    avg_processing_time_ms: float = 0.0
    current_rate: float = 0.0  # Messages per second
    peak_rate: float = 0.0


@dataclass
class FlowWindow:
    """Flow control window"""
    size: int  # Window size in messages
    credits: int  # Available credits
    inflight: int  # Messages in flight
    
    @property
    def available_space(self) -> int:
        """Calculate available space in window"""
        return max(0, self.size - self.inflight)
    
    def can_send(self, count: int = 1) -> bool:
        """Check if can send messages"""
        return self.credits >= count and self.available_space >= count


class CreditBasedFlowController:
    """
    Credit-based flow controller for streaming data
    
    Features:
    - Dynamic credit allocation
    - Automatic flow adjustment
    - Burst handling
    - Backpressure integration
    """
    
    def __init__(
        self,
        initial_credits: int = 100,
        max_credits: int = 1000,
        refill_rate: float = 10.0,  # Credits per second
        window_size: int = 100
    ):
        """
        Initialize flow controller
        
        Args:
            initial_credits: Starting credits
            max_credits: Maximum credits
            refill_rate: Credit refill rate
            window_size: Flow control window size
        """
        self.max_credits = max_credits
        self.refill_rate = refill_rate
        self.window_size = window_size
        
        # Flow window
        self.window = FlowWindow(
            size=window_size,
            credits=initial_credits,
            inflight=0
        )
        
        # Flow state
        self.state = FlowState.FLOWING
        self.last_refill = time.time()
        
        # Metrics
        self.metrics = FlowMetrics()
        self.rate_history = deque(maxlen=60)  # 1 minute of history
        
        # Callbacks
        self.on_pause: Optional[Callable] = None
        self.on_resume: Optional[Callable] = None
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Auto-adjustment parameters
        self.enable_auto_adjust = True
        self.target_utilization = 0.8  # Target 80% utilization
        
        logger.info(
            f"FlowController initialized: credits={initial_credits}, "
            f"max={max_credits}, window={window_size}"
        )
    
    def consume_credits(self, count: int = 1) -> bool:
        """
        Consume credits for sending messages
        
        Args:
            count: Number of credits to consume
            
        Returns:
            True if credits consumed, False if insufficient
        """
        with self._lock:
            # Refill credits first
            self._refill_credits()
            
            # Check if we have enough credits
            if self.window.credits >= count:
                self.window.credits -= count
                self.window.inflight += count
                self.metrics.credits_consumed += count
                self.metrics.messages_received += count
                
                # Check if we should pause
                if self.window.credits < self.window_size * 0.1:
                    self._transition_to(FlowState.PAUSED)
                elif self.window.credits < self.window_size * 0.3:
                    self._transition_to(FlowState.THROTTLED)
                
                return True
            else:
                self.metrics.messages_dropped += count
                return False
    
    def release_credits(self, count: int = 1):
        """
        Release credits after processing messages
        
        Args:
            count: Number of credits to release
        """
        with self._lock:
            self.window.inflight = max(0, self.window.inflight - count)
            self.metrics.messages_processed += count
            
            # Check if we should resume
            if self.state == FlowState.PAUSED and self.window.credits > self.window_size * 0.5:
                self._transition_to(FlowState.RESUMING)
            elif self.state == FlowState.THROTTLED and self.window.credits > self.window_size * 0.7:
                self._transition_to(FlowState.FLOWING)
    
    def grant_credits(self, count: int):
        """
        Grant additional credits
        
        Args:
            count: Number of credits to grant
        """
        with self._lock:
            self.window.credits = min(self.max_credits, self.window.credits + count)
            self.metrics.credits_granted += count
            
            # Check for state transition
            if self.state == FlowState.PAUSED and self.window.credits > self.window_size * 0.3:
                self._transition_to(FlowState.RESUMING)
    
    def _refill_credits(self):
        """Refill credits based on time elapsed"""
        now = time.time()
        elapsed = now - self.last_refill
        
        if elapsed > 0:
            credits_to_add = int(elapsed * self.refill_rate)
            if credits_to_add > 0:
                self.window.credits = min(
                    self.max_credits,
                    self.window.credits + credits_to_add
                )
                self.last_refill = now
    
    def _transition_to(self, new_state: FlowState):
        """Transition to new flow state"""
        if self.state == new_state:
            return
        
        old_state = self.state
        self.state = new_state
        
        logger.info(f"Flow state transition: {old_state.value} -> {new_state.value}")
        
        # Handle state-specific actions
        if new_state == FlowState.PAUSED:
            self.metrics.flow_pauses += 1
            if self.on_pause:
                asyncio.create_task(self.on_pause())
        elif new_state == FlowState.RESUMING:
            self.metrics.flow_resumes += 1
            if self.on_resume:
                asyncio.create_task(self.on_resume())
            # Auto-transition to flowing
            self.state = FlowState.FLOWING
    
    def pause(self):
        """Manually pause flow"""
        with self._lock:
            self._transition_to(FlowState.PAUSED)
    
    def resume(self):
        """Manually resume flow"""
        with self._lock:
            if self.state == FlowState.PAUSED:
                self._transition_to(FlowState.RESUMING)
    
    def is_flowing(self) -> bool:
        """Check if flow is active"""
        return self.state in [FlowState.FLOWING, FlowState.THROTTLED]
    
    def get_flow_rate(self) -> float:
        """Get current flow rate multiplier"""
        rates = {
            FlowState.FLOWING: 1.0,
            FlowState.THROTTLED: 0.5,
            FlowState.PAUSED: 0.0,
            FlowState.RESUMING: 0.7
        }
        return rates.get(self.state, 1.0)
    
    async def wait_for_credits(self, count: int = 1, timeout: float = None) -> bool:
        """
        Wait for credits to become available
        
        Args:
            count: Number of credits needed
            timeout: Maximum wait time
            
        Returns:
            True if credits available, False if timeout
        """
        start_time = time.time()
        
        while True:
            with self._lock:
                self._refill_credits()
                if self.window.credits >= count:
                    return True
            
            if timeout and (time.time() - start_time) >= timeout:
                return False
            
            # Calculate wait time
            deficit = count - self.window.credits
            wait_time = deficit / self.refill_rate if deficit > 0 else 0.01
            await asyncio.sleep(min(wait_time, 0.1))
    
    def update_metrics(self, processing_time_ms: float):
        """
        Update flow metrics
        
        Args:
            processing_time_ms: Message processing time
        """
        with self._lock:
            # Update average processing time
            alpha = 0.1  # Exponential moving average factor
            self.metrics.avg_processing_time_ms = (
                alpha * processing_time_ms +
                (1 - alpha) * self.metrics.avg_processing_time_ms
            )
            
            # Calculate current rate
            now = time.time()
            self.rate_history.append((now, self.metrics.messages_processed))
            
            if len(self.rate_history) > 1:
                oldest_time, oldest_count = self.rate_history[0]
                time_span = now - oldest_time
                if time_span > 0:
                    message_count = self.metrics.messages_processed - oldest_count
                    self.metrics.current_rate = message_count / time_span
                    self.metrics.peak_rate = max(self.metrics.peak_rate, self.metrics.current_rate)
            
            # Auto-adjust window size if enabled
            if self.enable_auto_adjust:
                self._auto_adjust_window()
    
    def _auto_adjust_window(self):
        """Automatically adjust window size based on performance"""
        utilization = self.window.inflight / self.window.size if self.window.size > 0 else 0
        
        # Adjust window size to maintain target utilization
        if utilization > self.target_utilization + 0.1:
            # Increase window size
            new_size = min(self.window_size * 2, int(self.window.size * 1.2))
            if new_size != self.window.size:
                self.window.size = new_size
                logger.info(f"Increased window size to {new_size}")
        elif utilization < self.target_utilization - 0.2:
            # Decrease window size
            new_size = max(10, int(self.window.size * 0.8))
            if new_size != self.window.size:
                self.window.size = new_size
                logger.info(f"Decreased window size to {new_size}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get flow controller statistics"""
        with self._lock:
            return {
                "state": self.state.value,
                "window": {
                    "size": self.window.size,
                    "credits": self.window.credits,
                    "inflight": self.window.inflight,
                    "utilization": f"{(self.window.inflight / self.window.size * 100):.1f}%"
                },
                "metrics": {
                    "messages_received": self.metrics.messages_received,
                    "messages_processed": self.metrics.messages_processed,
                    "messages_dropped": self.metrics.messages_dropped,
                    "flow_pauses": self.metrics.flow_pauses,
                    "flow_resumes": self.metrics.flow_resumes,
                    "avg_processing_time_ms": self.metrics.avg_processing_time_ms,
                    "current_rate": f"{self.metrics.current_rate:.1f}/s",
                    "peak_rate": f"{self.metrics.peak_rate:.1f}/s"
                },
                "credit_flow": {
                    "granted": self.metrics.credits_granted,
                    "consumed": self.metrics.credits_consumed,
                    "refill_rate": f"{self.refill_rate}/s"
                }
            }


class MessageBatcher:
    """
    Message batcher for efficient processing
    
    Features:
    - Time and size based batching
    - Priority message handling
    - Automatic flush
    """
    
    def __init__(
        self,
        max_batch_size: int = 100,
        max_batch_time: float = 0.1,  # 100ms
        flush_callback: Optional[Callable] = None
    ):
        """
        Initialize message batcher
        
        Args:
            max_batch_size: Maximum messages per batch
            max_batch_time: Maximum time before flush
            flush_callback: Callback for batch processing
        """
        self.max_batch_size = max_batch_size
        self.max_batch_time = max_batch_time
        self.flush_callback = flush_callback
        
        # Current batch
        self.batch: List[Any] = []
        self.batch_start_time = time.time()
        
        # Priority queue for urgent messages
        self.priority_queue: List[Any] = []
        
        # Statistics
        self.batches_sent = 0
        self.messages_batched = 0
        
        # Auto-flush task
        self.flush_task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()
        
        logger.info(f"MessageBatcher initialized: size={max_batch_size}, time={max_batch_time}s")
    
    def add_message(self, message: Any, priority: bool = False):
        """
        Add message to batch
        
        Args:
            message: Message to batch
            priority: If True, process immediately
        """
        with self._lock:
            if priority:
                self.priority_queue.append(message)
                # Flush immediately for priority messages
                asyncio.create_task(self._flush())
            else:
                self.batch.append(message)
                self.messages_batched += 1
                
                # Check if we should flush
                if len(self.batch) >= self.max_batch_size:
                    asyncio.create_task(self._flush())
                elif not self.flush_task or self.flush_task.done():
                    # Schedule auto-flush
                    self.flush_task = asyncio.create_task(self._auto_flush())
    
    async def _auto_flush(self):
        """Auto-flush after timeout"""
        await asyncio.sleep(self.max_batch_time)
        await self._flush()
    
    async def _flush(self):
        """Flush current batch"""
        with self._lock:
            # Combine priority and regular messages
            messages = self.priority_queue + self.batch
            
            if not messages:
                return
            
            # Clear buffers
            self.priority_queue = []
            self.batch = []
            self.batch_start_time = time.time()
            self.batches_sent += 1
        
        # Process batch
        if self.flush_callback:
            try:
                await self.flush_callback(messages)
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batcher statistics"""
        with self._lock:
            return {
                "current_batch_size": len(self.batch),
                "priority_queue_size": len(self.priority_queue),
                "batches_sent": self.batches_sent,
                "messages_batched": self.messages_batched,
                "avg_batch_size": (
                    self.messages_batched / self.batches_sent
                    if self.batches_sent > 0 else 0
                )
            }


class StreamFlowController:
    """
    Complete flow control system for streaming data
    
    Features:
    - Credit-based flow control
    - Message batching
    - Backpressure integration
    - Dynamic adjustment
    """
    
    def __init__(
        self,
        credits: int = 1000,
        batch_size: int = 100,
        batch_time: float = 0.1
    ):
        """
        Initialize stream flow controller
        
        Args:
            credits: Initial credits
            batch_size: Batch size
            batch_time: Batch timeout
        """
        self.flow_controller = CreditBasedFlowController(
            initial_credits=credits,
            max_credits=credits * 2
        )
        
        self.batcher = MessageBatcher(
            max_batch_size=batch_size,
            max_batch_time=batch_time
        )
        
        # Link batcher to flow control
        self.batcher.flush_callback = self._process_batch
        
        logger.info("StreamFlowController initialized")
    
    async def handle_message(self, message: Any, priority: bool = False) -> bool:
        """
        Handle incoming message with flow control
        
        Args:
            message: Message to handle
            priority: Priority flag
            
        Returns:
            True if accepted, False if dropped
        """
        # Check flow control
        if not self.flow_controller.consume_credits(1):
            logger.warning("Message dropped due to flow control")
            return False
        
        # Add to batch
        self.batcher.add_message(message, priority)
        return True
    
    async def _process_batch(self, messages: List[Any]):
        """Process a batch of messages"""
        start_time = time.time()
        
        try:
            # Process messages (override in subclass)
            await self.process_messages(messages)
            
            # Release credits
            self.flow_controller.release_credits(len(messages))
            
            # Update metrics
            processing_time_ms = (time.time() - start_time) * 1000
            self.flow_controller.update_metrics(processing_time_ms)
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            # Still release credits to prevent deadlock
            self.flow_controller.release_credits(len(messages))
    
    async def process_messages(self, messages: List[Any]):
        """
        Process messages (override in subclass)
        
        Args:
            messages: Batch of messages to process
        """
        # Default implementation - just log
        logger.debug(f"Processing {len(messages)} messages")
    
    def pause(self):
        """Pause message flow"""
        self.flow_controller.pause()
    
    def resume(self):
        """Resume message flow"""
        self.flow_controller.resume()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get complete flow statistics"""
        return {
            "flow_control": self.flow_controller.get_stats(),
            "batching": self.batcher.get_stats()
        }