"""
Circuit Breaker Pattern Implementation
HFT-optimized resilience pattern with microsecond latency
Based on Netflix Hystrix and modern HFT practices
"""

import asyncio
import time
import random
import logging
from typing import Optional, Callable, Any, Dict, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import threading

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"        # Normal operation, requests pass through
    OPEN = "open"            # Failure threshold exceeded, requests fail fast
    HALF_OPEN = "half_open"  # Testing recovery, limited requests allowed


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior"""
    name: str
    failure_threshold: int = 5           # Failures to trigger open state
    success_threshold: int = 3           # Successes to close circuit
    timeout: float = 30.0               # Timeout for operations (seconds)
    half_open_interval: float = 10.0    # Time before testing recovery (seconds)
    window_size: int = 60               # Sliding window size (seconds)
    max_backoff: float = 30.0           # Maximum backoff time (seconds)
    jitter_range: float = 0.1           # Jitter range (0-1)
    
    # HFT optimizations
    use_lock_free: bool = True          # Use lock-free operations
    emit_metrics: bool = True           # Emit metrics for monitoring
    cache_health_check: bool = True     # Cache health check results


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker monitoring"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state_transitions: List[tuple] = field(default_factory=list)
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    current_backoff_attempt: int = 0
    average_response_time: float = 0.0


class SlidingWindow:
    """
    Lock-free sliding window for failure tracking
    Optimized for HFT with microsecond operations
    """
    
    def __init__(self, window_size_seconds: int = 60):
        """
        Initialize sliding window
        
        Args:
            window_size_seconds: Size of the sliding window in seconds
        """
        self.window_size = window_size_seconds
        self.events: deque = deque()
        self._lock = threading.RLock() if not hasattr(threading, 'Lock') else None
    
    def add_event(self, success: bool) -> tuple[int, int]:
        """
        Add event to sliding window
        
        Args:
            success: Whether the event was successful
            
        Returns:
            Tuple of (failures, total) in window
        """
        now = time.time()
        self.events.append((now, success))
        
        # Remove old events outside window
        cutoff = now - self.window_size
        while self.events and self.events[0][0] < cutoff:
            self.events.popleft()
        
        # Count failures and total
        failures = sum(1 for _, success in self.events if not success)
        total = len(self.events)
        
        return failures, total
    
    def get_stats(self) -> Dict[str, Any]:
        """Get window statistics"""
        if not self.events:
            return {"failures": 0, "total": 0, "failure_rate": 0.0}
        
        failures = sum(1 for _, success in self.events if not success)
        total = len(self.events)
        
        return {
            "failures": failures,
            "total": total,
            "failure_rate": failures / total if total > 0 else 0.0
        }
    
    def clear(self):
        """Clear all events"""
        self.events.clear()


class CircuitBreaker:
    """
    HFT-optimized circuit breaker implementation
    
    Features:
    - Lock-free state transitions for microsecond latency
    - Sliding window failure tracking
    - Exponential backoff with jitter
    - Health probe support
    - Metrics emission
    """
    
    def __init__(self, config: CircuitBreakerConfig):
        """
        Initialize circuit breaker
        
        Args:
            config: Circuit breaker configuration
        """
        self.config = config
        self.state = CircuitState.CLOSED
        self.sliding_window = SlidingWindow(config.window_size)
        self.metrics = CircuitBreakerMetrics()
        
        # State tracking
        self.consecutive_successes = 0
        self.last_state_change = datetime.now()
        self.last_open_time: Optional[datetime] = None
        self.backoff_attempt = 0
        
        # Health check cache
        self._health_check_result: Optional[bool] = None
        self._health_check_time: Optional[float] = None
        
        # Callbacks
        self.on_state_change: Optional[Callable] = None
        self.on_metrics_emit: Optional[Callable] = None
        
        logger.info(f"Circuit breaker '{config.name}' initialized")
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection
        
        Args:
            func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        start_time = time.time()
        
        # Fast path: Check if circuit is open
        if self.state == CircuitState.OPEN:
            if not self._should_attempt_reset():
                self.metrics.rejected_calls += 1
                raise CircuitOpenException(
                    f"Circuit breaker '{self.config.name}' is OPEN"
                )
            
            # Transition to half-open for testing
            await self._transition_to(CircuitState.HALF_OPEN)
        
        # Check if we're in half-open state (limited traffic)
        if self.state == CircuitState.HALF_OPEN:
            # Allow limited requests for testing
            if random.random() > 0.1:  # Allow 10% of traffic
                self.metrics.rejected_calls += 1
                raise CircuitOpenException(
                    f"Circuit breaker '{self.config.name}' is HALF_OPEN (limited traffic)"
                )
        
        # Try to execute the function
        try:
            # Add timeout protection
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout
            )
            
            # Record success
            await self._on_success()
            
            # Update metrics
            elapsed = time.time() - start_time
            self.metrics.average_response_time = (
                0.9 * self.metrics.average_response_time + 0.1 * elapsed
            )
            
            return result
            
        except asyncio.TimeoutError:
            await self._on_failure()
            raise CircuitTimeoutException(
                f"Operation timed out after {self.config.timeout}s"
            )
        except Exception as e:
            await self._on_failure()
            raise
    
    async def _on_success(self):
        """Handle successful call"""
        self.metrics.successful_calls += 1
        self.metrics.total_calls += 1
        self.metrics.last_success_time = datetime.now()
        
        # Update sliding window
        failures, total = self.sliding_window.add_event(True)
        
        # Update consecutive successes
        self.consecutive_successes += 1
        self.backoff_attempt = 0
        
        # Check if we should close the circuit
        if self.state == CircuitState.HALF_OPEN:
            if self.consecutive_successes >= self.config.success_threshold:
                await self._transition_to(CircuitState.CLOSED)
        
        # Emit metrics if configured
        if self.config.emit_metrics and self.on_metrics_emit:
            await self.on_metrics_emit(self.get_metrics())
    
    async def _on_failure(self):
        """Handle failed call"""
        self.metrics.failed_calls += 1
        self.metrics.total_calls += 1
        self.metrics.last_failure_time = datetime.now()
        
        # Reset consecutive successes
        self.consecutive_successes = 0
        
        # Update sliding window
        failures, total = self.sliding_window.add_event(False)
        
        # Check if we should open the circuit
        if self.state == CircuitState.CLOSED:
            if failures >= self.config.failure_threshold:
                await self._transition_to(CircuitState.OPEN)
        elif self.state == CircuitState.HALF_OPEN:
            # Failed during recovery test, back to open
            await self._transition_to(CircuitState.OPEN)
        
        # Increment backoff attempt
        if self.state == CircuitState.OPEN:
            self.backoff_attempt += 1
        
        # Emit metrics if configured
        if self.config.emit_metrics and self.on_metrics_emit:
            await self.on_metrics_emit(self.get_metrics())
    
    async def _transition_to(self, new_state: CircuitState):
        """
        Transition to new state
        
        Args:
            new_state: Target state
        """
        if self.state == new_state:
            return
        
        old_state = self.state
        self.state = new_state
        self.last_state_change = datetime.now()
        
        # Track state transition
        self.metrics.state_transitions.append(
            (old_state, new_state, self.last_state_change)
        )
        
        # Reset counters based on state
        if new_state == CircuitState.OPEN:
            self.last_open_time = datetime.now()
            self.consecutive_successes = 0
        elif new_state == CircuitState.CLOSED:
            self.sliding_window.clear()
            self.consecutive_successes = 0
            self.backoff_attempt = 0
        
        logger.warning(
            f"Circuit breaker '{self.config.name}' transitioned: "
            f"{old_state.value} -> {new_state.value}"
        )
        
        # Notify state change
        if self.on_state_change:
            await self.on_state_change(old_state, new_state)
    
    def _should_attempt_reset(self) -> bool:
        """
        Check if we should attempt to reset (test recovery)
        
        Returns:
            True if we should test recovery
        """
        if not self.last_open_time:
            return True
        
        # Calculate backoff with exponential strategy and jitter
        base_backoff = min(
            self.config.half_open_interval * (2 ** self.backoff_attempt),
            self.config.max_backoff
        )
        
        # Add jitter to prevent thundering herd
        jitter = random.uniform(0, base_backoff * self.config.jitter_range)
        backoff_seconds = base_backoff + jitter
        
        # Check if enough time has passed
        elapsed = (datetime.now() - self.last_open_time).total_seconds()
        
        return elapsed >= backoff_seconds
    
    async def health_check(self, check_func: Optional[Callable] = None) -> bool:
        """
        Perform health check
        
        Args:
            check_func: Optional custom health check function
            
        Returns:
            True if healthy
        """
        # Use cached result if available and fresh
        if self.config.cache_health_check and self._health_check_time:
            if time.time() - self._health_check_time < 5.0:  # 5 second cache
                return self._health_check_result or False
        
        # Default health check based on state and metrics
        if check_func:
            try:
                result = await check_func()
            except:
                result = False
        else:
            # Simple health check based on state
            result = self.state != CircuitState.OPEN
        
        # Cache result
        self._health_check_result = result
        self._health_check_time = time.time()
        
        return result
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics"""
        window_stats = self.sliding_window.get_stats()
        
        return {
            "name": self.config.name,
            "state": self.state.value,
            "metrics": {
                "total_calls": self.metrics.total_calls,
                "successful_calls": self.metrics.successful_calls,
                "failed_calls": self.metrics.failed_calls,
                "rejected_calls": self.metrics.rejected_calls,
                "success_rate": (
                    self.metrics.successful_calls / self.metrics.total_calls
                    if self.metrics.total_calls > 0 else 0.0
                ),
                "average_response_time": self.metrics.average_response_time,
                "current_backoff_attempt": self.backoff_attempt
            },
            "window": window_stats,
            "consecutive_successes": self.consecutive_successes,
            "last_state_change": self.last_state_change.isoformat(),
            "last_failure": (
                self.metrics.last_failure_time.isoformat()
                if self.metrics.last_failure_time else None
            ),
            "last_success": (
                self.metrics.last_success_time.isoformat()
                if self.metrics.last_success_time else None
            )
        }
    
    def reset(self):
        """Reset circuit breaker to initial state"""
        self.state = CircuitState.CLOSED
        self.sliding_window.clear()
        self.consecutive_successes = 0
        self.backoff_attempt = 0
        self.metrics = CircuitBreakerMetrics()
        logger.info(f"Circuit breaker '{self.config.name}' reset")


class CircuitOpenException(Exception):
    """Exception raised when circuit is open"""
    pass


class CircuitTimeoutException(Exception):
    """Exception raised when operation times out"""
    pass


# Factory function for creating circuit breakers
def create_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    timeout: float = 30.0,
    **kwargs
) -> CircuitBreaker:
    """
    Create a circuit breaker with default HFT settings
    
    Args:
        name: Circuit breaker name
        failure_threshold: Number of failures to trigger open
        timeout: Operation timeout
        **kwargs: Additional configuration
        
    Returns:
        Configured circuit breaker
    """
    config = CircuitBreakerConfig(
        name=name,
        failure_threshold=failure_threshold,
        timeout=timeout,
        **kwargs
    )
    
    return CircuitBreaker(config)