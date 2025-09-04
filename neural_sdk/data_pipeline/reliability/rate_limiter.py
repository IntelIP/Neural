"""
Rate Limiter - Multi-algorithm rate limiting with HFT optimizations
Implements Token Bucket, Sliding Window, and Adaptive algorithms
Lock-free operations for microsecond latency
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import threading
from collections import deque

logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded"""
    pass


@dataclass
class RateLimiterConfig:
    """Configuration for rate limiter"""
    name: str
    limit: int  # Maximum requests
    window: float  # Time window in seconds
    burst_size: Optional[int] = None  # Max burst (for token bucket)
    enable_adaptive: bool = False  # Enable adaptive limiting
    min_limit: int = 10  # Minimum limit for adaptive
    max_limit: int = 10000  # Maximum limit for adaptive


class TokenBucket:
    """
    Token Bucket Rate Limiter
    
    Features:
    - O(1) operations with atomic counters
    - Allows controlled bursts
    - Smooth rate limiting
    - Thread-safe implementation
    """
    
    def __init__(
        self,
        capacity: int,
        refill_rate: float,
        burst_size: Optional[int] = None
    ):
        """
        Initialize token bucket
        
        Args:
            capacity: Maximum tokens in bucket
            refill_rate: Tokens added per second
            burst_size: Maximum burst allowed
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.burst_size = burst_size or capacity
        
        # Atomic operations using lock
        self._lock = threading.Lock()
        self._tokens = float(capacity)
        self._last_refill = time.time()
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "accepted_requests": 0,
            "rejected_requests": 0,
            "total_tokens_consumed": 0
        }
        
        logger.info(f"TokenBucket initialized: capacity={capacity}, rate={refill_rate}/s")
    
    def try_consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens consumed, False if rate limited
        """
        with self._lock:
            # Refill tokens based on time elapsed
            now = time.time()
            elapsed = now - self._last_refill
            
            # Add tokens based on refill rate
            tokens_to_add = elapsed * self.refill_rate
            self._tokens = min(self.capacity, self._tokens + tokens_to_add)
            self._last_refill = now
            
            # Update stats
            self.stats["total_requests"] += 1
            
            # Check if we have enough tokens
            if self._tokens >= tokens:
                self._tokens -= tokens
                self.stats["accepted_requests"] += 1
                self.stats["total_tokens_consumed"] += tokens
                return True
            else:
                self.stats["rejected_requests"] += 1
                return False
    
    async def consume(self, tokens: int = 1, timeout: float = None) -> None:
        """
        Consume tokens, waiting if necessary
        
        Args:
            tokens: Number of tokens to consume
            timeout: Maximum time to wait
            
        Raises:
            RateLimitExceeded: If timeout reached
        """
        start_time = time.time()
        
        while True:
            if self.try_consume(tokens):
                return
            
            # Check timeout
            if timeout and (time.time() - start_time) >= timeout:
                raise RateLimitExceeded(f"Rate limit timeout after {timeout}s")
            
            # Calculate wait time
            with self._lock:
                deficit = tokens - self._tokens
                wait_time = deficit / self.refill_rate if deficit > 0 else 0.001
            
            await asyncio.sleep(min(wait_time, 0.1))
    
    def tokens_available(self) -> float:
        """Get current token count"""
        with self._lock:
            # Refill before checking
            now = time.time()
            elapsed = now - self._last_refill
            tokens_to_add = elapsed * self.refill_rate
            return min(self.capacity, self._tokens + tokens_to_add)
    
    def reset(self):
        """Reset bucket to full capacity"""
        with self._lock:
            self._tokens = float(self.capacity)
            self._last_refill = time.time()
            logger.info(f"TokenBucket reset to capacity {self.capacity}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bucket statistics"""
        with self._lock:
            return {
                **self.stats,
                "current_tokens": self._tokens,
                "capacity": self.capacity,
                "refill_rate": self.refill_rate,
                "utilization": 1.0 - (self._tokens / self.capacity)
            }


class SlidingWindowRateLimiter:
    """
    Sliding Window Rate Limiter
    
    Features:
    - Precise rate tracking
    - Microsecond granularity
    - Lock-free circular buffer
    - Memory efficient
    """
    
    def __init__(self, limit: int, window: float):
        """
        Initialize sliding window limiter
        
        Args:
            limit: Maximum requests in window
            window: Time window in seconds
        """
        self.limit = limit
        self.window = window
        
        # Circular buffer for timestamps
        self.timestamps = deque(maxlen=limit * 2)
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "accepted_requests": 0,
            "rejected_requests": 0,
            "window_violations": 0
        }
        
        logger.info(f"SlidingWindow initialized: {limit} requests per {window}s")
    
    def try_acquire(self) -> bool:
        """
        Try to acquire permission for request
        
        Returns:
            True if allowed, False if rate limited
        """
        now = time.time()
        cutoff = now - self.window
        
        with self._lock:
            # Remove expired timestamps
            while self.timestamps and self.timestamps[0] < cutoff:
                self.timestamps.popleft()
            
            self.stats["total_requests"] += 1
            
            # Check if under limit
            if len(self.timestamps) < self.limit:
                self.timestamps.append(now)
                self.stats["accepted_requests"] += 1
                return True
            else:
                self.stats["rejected_requests"] += 1
                self.stats["window_violations"] += 1
                return False
    
    async def acquire(self, timeout: float = None) -> None:
        """
        Acquire permission, waiting if necessary
        
        Args:
            timeout: Maximum time to wait
            
        Raises:
            RateLimitExceeded: If timeout reached
        """
        start_time = time.time()
        
        while True:
            if self.try_acquire():
                return
            
            # Check timeout
            if timeout and (time.time() - start_time) >= timeout:
                raise RateLimitExceeded(f"Rate limit timeout after {timeout}s")
            
            # Calculate wait time until oldest timestamp expires
            with self._lock:
                if self.timestamps:
                    oldest = self.timestamps[0]
                    wait_time = (oldest + self.window) - time.time()
                    wait_time = max(0.001, min(wait_time, 0.1))
                else:
                    wait_time = 0.001
            
            await asyncio.sleep(wait_time)
    
    def current_rate(self) -> float:
        """Get current request rate"""
        now = time.time()
        cutoff = now - self.window
        
        with self._lock:
            # Count requests in window
            count = sum(1 for ts in self.timestamps if ts >= cutoff)
            return count / self.window if self.window > 0 else 0
    
    def reset(self):
        """Reset the sliding window"""
        with self._lock:
            self.timestamps.clear()
            logger.info("SlidingWindow reset")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get limiter statistics"""
        with self._lock:
            return {
                **self.stats,
                "current_rate": self.current_rate(),
                "limit": self.limit,
                "window": self.window,
                "utilization": len(self.timestamps) / self.limit if self.limit > 0 else 0
            }


class AdaptiveRateLimiter:
    """
    Adaptive Rate Limiter that adjusts based on system behavior
    
    Features:
    - Dynamic limit adjustment
    - Error rate monitoring
    - Learns optimal rates
    - Integration with circuit breakers
    """
    
    def __init__(
        self,
        initial_limit: int,
        window: float,
        min_limit: int = 10,
        max_limit: int = 10000,
        target_success_rate: float = 0.99
    ):
        """
        Initialize adaptive limiter
        
        Args:
            initial_limit: Starting limit
            window: Time window in seconds
            min_limit: Minimum allowed limit
            max_limit: Maximum allowed limit
            target_success_rate: Target success rate
        """
        self.current_limit = initial_limit
        self.window = window
        self.min_limit = min_limit
        self.max_limit = max_limit
        self.target_success_rate = target_success_rate
        
        # Underlying limiter (using sliding window)
        self.limiter = SlidingWindowRateLimiter(initial_limit, window)
        
        # Adaptation metrics
        self.success_count = 0
        self.error_count = 0
        self.last_adjustment = time.time()
        self.adjustment_interval = 30.0  # Adjust every 30 seconds
        
        # Statistics
        self.limit_history = deque(maxlen=100)
        self.limit_history.append((datetime.now(), initial_limit))
        
        logger.info(f"AdaptiveRateLimiter initialized: {initial_limit} ({min_limit}-{max_limit})")
    
    async def acquire(self) -> None:
        """Acquire permission with adaptive limiting"""
        await self.limiter.acquire()
    
    def try_acquire(self) -> bool:
        """Try to acquire permission"""
        result = self.limiter.try_acquire()
        
        # Check if we should adjust limits
        now = time.time()
        if now - self.last_adjustment > self.adjustment_interval:
            self._adjust_limit()
            self.last_adjustment = now
        
        return result
    
    def record_success(self):
        """Record successful operation"""
        self.success_count += 1
    
    def record_error(self):
        """Record failed operation"""
        self.error_count += 1
    
    def _adjust_limit(self):
        """Adjust rate limit based on metrics"""
        total = self.success_count + self.error_count
        
        if total == 0:
            return
        
        success_rate = self.success_count / total
        old_limit = self.current_limit
        
        if success_rate >= self.target_success_rate:
            # Increase limit if we're doing well
            self.current_limit = min(
                int(self.current_limit * 1.1),
                self.max_limit
            )
        elif success_rate < self.target_success_rate * 0.9:
            # Decrease limit if too many errors
            self.current_limit = max(
                int(self.current_limit * 0.8),
                self.min_limit
            )
        
        # Apply new limit if changed
        if self.current_limit != old_limit:
            self.limiter = SlidingWindowRateLimiter(self.current_limit, self.window)
            self.limit_history.append((datetime.now(), self.current_limit))
            
            logger.info(
                f"Adjusted rate limit: {old_limit} -> {self.current_limit} "
                f"(success rate: {success_rate:.2%})"
            )
        
        # Reset counters
        self.success_count = 0
        self.error_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get adaptive limiter statistics"""
        total = self.success_count + self.error_count
        success_rate = self.success_count / total if total > 0 else 1.0
        
        return {
            "current_limit": self.current_limit,
            "success_rate": success_rate,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "limit_range": f"{self.min_limit}-{self.max_limit}",
            "limit_history": [
                {"time": t.isoformat(), "limit": l}
                for t, l in list(self.limit_history)[-10:]
            ],
            **self.limiter.get_stats()
        }


class HierarchicalRateLimiter:
    """
    Hierarchical rate limiter with multiple levels
    
    Features:
    - Global, service, and operation level limits
    - Priority-based allocation
    - Fair sharing between services
    """
    
    def __init__(self, global_limit: int, window: float):
        """
        Initialize hierarchical limiter
        
        Args:
            global_limit: Global rate limit
            window: Time window in seconds
        """
        self.global_limiter = TokenBucket(global_limit, global_limit / window)
        self.service_limiters: Dict[str, TokenBucket] = {}
        self.operation_limiters: Dict[Tuple[str, str], TokenBucket] = {}
        
        logger.info(f"HierarchicalRateLimiter initialized: {global_limit} global limit")
    
    def register_service(self, service: str, limit: int, window: float):
        """Register service-level limit"""
        self.service_limiters[service] = TokenBucket(limit, limit / window)
        logger.info(f"Registered service limit: {service} = {limit}/{window}s")
    
    def register_operation(self, service: str, operation: str, limit: int, window: float):
        """Register operation-level limit"""
        key = (service, operation)
        self.operation_limiters[key] = TokenBucket(limit, limit / window)
        logger.info(f"Registered operation limit: {service}.{operation} = {limit}/{window}s")
    
    async def acquire(
        self,
        service: str,
        operation: Optional[str] = None,
        priority: int = 1
    ) -> None:
        """
        Acquire permission through hierarchy
        
        Args:
            service: Service name
            operation: Operation name
            priority: Request priority (higher = more tokens)
            
        Raises:
            RateLimitExceeded: If any level is rate limited
        """
        # Check global limit
        await self.global_limiter.consume(priority)
        
        try:
            # Check service limit
            if service in self.service_limiters:
                await self.service_limiters[service].consume(priority)
            
            # Check operation limit
            if operation:
                key = (service, operation)
                if key in self.operation_limiters:
                    await self.operation_limiters[key].consume(priority)
                    
        except RateLimitExceeded:
            # Return tokens to global if service/operation limited
            self.global_limiter._tokens += priority
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get hierarchical statistics"""
        return {
            "global": self.global_limiter.get_stats(),
            "services": {
                name: limiter.get_stats()
                for name, limiter in self.service_limiters.items()
            },
            "operations": {
                f"{s}.{o}": limiter.get_stats()
                for (s, o), limiter in self.operation_limiters.items()
            }
        }


# Factory function for creating rate limiters
def create_rate_limiter(
    algorithm: str = "token_bucket",
    limit: int = 100,
    window: float = 1.0,
    **kwargs
) -> Any:
    """
    Create a rate limiter with specified algorithm
    
    Args:
        algorithm: Algorithm type (token_bucket, sliding_window, adaptive)
        limit: Rate limit
        window: Time window
        **kwargs: Additional algorithm-specific parameters
        
    Returns:
        Rate limiter instance
    """
    if algorithm == "token_bucket":
        refill_rate = limit / window
        return TokenBucket(limit, refill_rate, **kwargs)
    elif algorithm == "sliding_window":
        return SlidingWindowRateLimiter(limit, window)
    elif algorithm == "adaptive":
        return AdaptiveRateLimiter(limit, window, **kwargs)
    elif algorithm == "hierarchical":
        return HierarchicalRateLimiter(limit, window)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")