"""
Rate Limiting for REST Data Sources

Provides rate limiting functionality to prevent API throttling.
"""

import asyncio
import time
from typing import Optional, Dict
from collections import deque
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter for API requests.
    
    Implements a token bucket algorithm that allows bursts
    while maintaining an average rate limit.
    """
    
    def __init__(
        self,
        requests_per_second: float = 10,
        burst_size: Optional[int] = None,
        name: str = "RateLimiter"
    ):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_second: Maximum average requests per second
            burst_size: Maximum burst size (defaults to requests_per_second)
            name: Name for logging
        """
        self.requests_per_second = requests_per_second
        self.burst_size = burst_size or int(requests_per_second)
        self.name = name
        
        # Token bucket
        self.tokens = float(self.burst_size)
        self.max_tokens = float(self.burst_size)
        self.refill_rate = requests_per_second
        self.last_refill = time.monotonic()
        
        # Lock for thread safety
        self.lock = asyncio.Lock()
        
        # Statistics
        self.stats = {
            'requests': 0,
            'throttled': 0,
            'total_wait_time': 0
        }
    
    def _refill_tokens(self):
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        
        # Add tokens based on elapsed time
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.tokens + tokens_to_add, self.max_tokens)
        self.last_refill = now
    
    async def acquire(self, tokens: float = 1.0) -> float:
        """
        Acquire tokens, waiting if necessary.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            Time waited in seconds
        """
        async with self.lock:
            start_time = time.monotonic()
            self.stats['requests'] += 1
            
            # Refill tokens
            self._refill_tokens()
            
            # Wait if not enough tokens
            wait_time = 0
            if self.tokens < tokens:
                # Calculate wait time
                tokens_needed = tokens - self.tokens
                wait_time = tokens_needed / self.refill_rate
                
                logger.debug(f"{self.name}: Rate limited, waiting {wait_time:.2f}s")
                self.stats['throttled'] += 1
                self.stats['total_wait_time'] += wait_time
                
                await asyncio.sleep(wait_time)
                
                # Refill again after waiting
                self._refill_tokens()
            
            # Consume tokens
            self.tokens -= tokens
            
            actual_wait = time.monotonic() - start_time
            return actual_wait
    
    async def try_acquire(self, tokens: float = 1.0) -> bool:
        """
        Try to acquire tokens without waiting.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            True if acquired, False if would need to wait
        """
        async with self.lock:
            self._refill_tokens()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                self.stats['requests'] += 1
                return True
            
            return False
    
    def get_stats(self) -> Dict:
        """Get rate limiter statistics."""
        return {
            'name': self.name,
            'requests': self.stats['requests'],
            'throttled': self.stats['throttled'],
            'throttle_rate': (
                self.stats['throttled'] / self.stats['requests']
                if self.stats['requests'] > 0 else 0
            ),
            'total_wait_time': self.stats['total_wait_time'],
            'average_wait_time': (
                self.stats['total_wait_time'] / self.stats['throttled']
                if self.stats['throttled'] > 0 else 0
            ),
            'current_tokens': self.tokens,
            'max_tokens': self.max_tokens
        }
    
    def reset(self):
        """Reset the rate limiter to full capacity."""
        self.tokens = self.max_tokens
        self.last_refill = time.monotonic()


class HierarchicalRateLimiter:
    """
    Hierarchical rate limiter for multiple tiers of limits.
    
    Useful for APIs with multiple rate limits (e.g., per second, per minute, per hour).
    """
    
    def __init__(self, name: str = "HierarchicalRateLimiter"):
        """
        Initialize hierarchical rate limiter.
        
        Args:
            name: Name for logging
        """
        self.name = name
        self.limiters: Dict[str, RateLimiter] = {}
    
    def add_limit(
        self,
        tier: str,
        requests: int,
        period_seconds: float,
        burst_size: Optional[int] = None
    ):
        """
        Add a rate limit tier.
        
        Args:
            tier: Name of the tier (e.g., "second", "minute", "hour")
            requests: Number of requests allowed
            period_seconds: Period in seconds
            burst_size: Optional burst size
        """
        rate = requests / period_seconds
        self.limiters[tier] = RateLimiter(
            requests_per_second=rate,
            burst_size=burst_size,
            name=f"{self.name}_{tier}"
        )
    
    async def acquire(self, tokens: float = 1.0) -> float:
        """
        Acquire tokens from all tiers.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            Total time waited
        """
        total_wait = 0
        
        # Acquire from all limiters
        for tier, limiter in self.limiters.items():
            wait_time = await limiter.acquire(tokens)
            total_wait = max(total_wait, wait_time)
        
        return total_wait
    
    async def try_acquire(self, tokens: float = 1.0) -> bool:
        """
        Try to acquire tokens from all tiers without waiting.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            True if acquired from all tiers, False otherwise
        """
        # Check all limiters first
        for limiter in self.limiters.values():
            async with limiter.lock:
                limiter._refill_tokens()
                if limiter.tokens < tokens:
                    return False
        
        # If all have tokens, acquire from all
        for limiter in self.limiters.values():
            await limiter.try_acquire(tokens)
        
        return True
    
    def get_stats(self) -> Dict:
        """Get statistics for all tiers."""
        return {
            tier: limiter.get_stats()
            for tier, limiter in self.limiters.items()
        }


class AdaptiveRateLimiter(RateLimiter):
    """
    Adaptive rate limiter that adjusts based on API responses.
    
    Automatically reduces rate when encountering 429 errors
    and gradually increases back to normal.
    """
    
    def __init__(
        self,
        initial_rate: float = 10,
        min_rate: float = 1,
        max_rate: float = 100,
        name: str = "AdaptiveRateLimiter"
    ):
        """
        Initialize adaptive rate limiter.
        
        Args:
            initial_rate: Initial requests per second
            min_rate: Minimum requests per second
            max_rate: Maximum requests per second
            name: Name for logging
        """
        super().__init__(requests_per_second=initial_rate, name=name)
        
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.initial_rate = initial_rate
        
        # Adaptation parameters
        self.backoff_factor = 0.5  # Reduce rate by 50% on error
        self.recovery_factor = 1.1  # Increase rate by 10% on success
        self.success_streak = 0
        self.recovery_threshold = 10  # Successes before increasing rate
    
    async def on_success(self):
        """Called on successful request."""
        self.success_streak += 1
        
        # Gradually increase rate after consistent success
        if self.success_streak >= self.recovery_threshold:
            new_rate = min(
                self.requests_per_second * self.recovery_factor,
                self.max_rate
            )
            
            if new_rate > self.requests_per_second:
                logger.info(
                    f"{self.name}: Increasing rate from "
                    f"{self.requests_per_second:.1f} to {new_rate:.1f} rps"
                )
                self.requests_per_second = new_rate
                self.refill_rate = new_rate
                self.success_streak = 0
    
    async def on_rate_limit(self, retry_after: Optional[float] = None):
        """
        Called when rate limited by the API.
        
        Args:
            retry_after: Optional retry-after header value in seconds
        """
        self.success_streak = 0
        
        if retry_after:
            # Use retry-after to calculate new rate
            new_rate = max(1.0 / retry_after, self.min_rate)
        else:
            # Reduce rate by backoff factor
            new_rate = max(
                self.requests_per_second * self.backoff_factor,
                self.min_rate
            )
        
        if new_rate < self.requests_per_second:
            logger.warning(
                f"{self.name}: Reducing rate from "
                f"{self.requests_per_second:.1f} to {new_rate:.1f} rps"
            )
            self.requests_per_second = new_rate
            self.refill_rate = new_rate
            
            # Also reduce current tokens to prevent burst
            self.tokens = min(self.tokens, new_rate)
            self.max_tokens = new_rate
            self.burst_size = int(new_rate)
    
    def reset_to_initial(self):
        """Reset rate to initial value."""
        self.requests_per_second = self.initial_rate
        self.refill_rate = self.initial_rate
        self.burst_size = int(self.initial_rate)
        self.max_tokens = float(self.burst_size)
        self.tokens = float(self.burst_size)
        self.success_streak = 0