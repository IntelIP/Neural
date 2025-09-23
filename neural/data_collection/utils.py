"""
Utility functions for the Neural SDK Data Collection Infrastructure.

This module provides common utilities including retry decorators, async helpers,
data transformation functions, and logging configuration.
"""

import asyncio
import functools
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, List, Optional, TypeVar, Union
from collections import deque
import inspect

from neural.data_collection.exceptions import (
    RetryableError,
    TransientError,
    TimeoutError
)


# Configure module logger
logger = logging.getLogger(__name__)

# Type variables
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


class RetryConfig:
    """
    Configuration for retry behavior.
    
    Attributes:
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff calculation
        jitter: Whether to add random jitter to delays
    """
    
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter


def retry(
    config: Optional[RetryConfig] = None,
    exceptions: tuple = (RetryableError, TransientError, TimeoutError)
) -> Callable[[F], F]:
    """
    Decorator for automatic retry with exponential backoff.
    
    Args:
        config: Retry configuration
        exceptions: Tuple of exceptions to retry on
        
    Returns:
        Decorated function with retry logic
        
    Example:
        >>> @retry(RetryConfig(max_attempts=5))
        >>> async def fetch_data():
        ...     return await api_call()
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_attempts - 1:
                        logger.error(
                            f"Max retries ({config.max_attempts}) exceeded for {func.__name__}"
                        )
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        config.initial_delay * (config.exponential_base ** attempt),
                        config.max_delay
                    )
                    
                    # Add jitter if enabled
                    if config.jitter:
                        import random
                        delay *= (0.5 + random.random())
                    
                    logger.warning(
                        f"Retry {attempt + 1}/{config.max_attempts} for {func.__name__} "
                        f"after {delay:.2f}s delay. Error: {e}"
                    )
                    
                    await asyncio.sleep(delay)
            
            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
                
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_attempts - 1:
                        logger.error(
                            f"Max retries ({config.max_attempts}) exceeded for {func.__name__}"
                        )
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        config.initial_delay * (config.exponential_base ** attempt),
                        config.max_delay
                    )
                    
                    # Add jitter if enabled
                    if config.jitter:
                        import random
                        delay *= (0.5 + random.random())
                    
                    logger.warning(
                        f"Retry {attempt + 1}/{config.max_attempts} for {func.__name__} "
                        f"after {delay:.2f}s delay. Error: {e}"
                    )
                    
                    time.sleep(delay)
            
            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class AsyncTaskManager:
    """
    Manager for async tasks with lifecycle management.
    
    This class helps manage background tasks, ensuring they are
    properly started, monitored, and cleaned up.
    
    Example:
        >>> manager = AsyncTaskManager()
        >>> await manager.start_task("heartbeat", heartbeat_coro())
        >>> await manager.stop_task("heartbeat")
    """
    
    def __init__(self):
        self._tasks: Dict[str, asyncio.Task] = {}
        self._shutdown = False
    
    async def start_task(
        self,
        name: str,
        coro: Coroutine,
        restart_on_error: bool = True
    ) -> None:
        """
        Start a managed async task.
        
        Args:
            name: Unique name for the task
            coro: Coroutine to run
            restart_on_error: Whether to restart on error
        """
        if name in self._tasks:
            await self.stop_task(name)
        
        async def wrapped():
            while not self._shutdown:
                try:
                    await coro
                    break  # Task completed normally
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Task {name} failed: {e}")
                    if not restart_on_error:
                        break
                    await asyncio.sleep(1)  # Brief pause before restart
        
        self._tasks[name] = asyncio.create_task(wrapped())
        logger.debug(f"Started task: {name}")
    
    async def stop_task(self, name: str) -> None:
        """
        Stop a managed task.
        
        Args:
            name: Name of the task to stop
        """
        if name in self._tasks:
            task = self._tasks[name]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            del self._tasks[name]
            logger.debug(f"Stopped task: {name}")
    
    async def stop_all(self) -> None:
        """Stop all managed tasks."""
        self._shutdown = True
        tasks = list(self._tasks.keys())
        for name in tasks:
            await self.stop_task(name)
    
    def get_status(self) -> Dict[str, str]:
        """
        Get status of all tasks.
        
        Returns:
            Dict mapping task names to status strings
        """
        status = {}
        for name, task in self._tasks.items():
            if task.done():
                if task.cancelled():
                    status[name] = "cancelled"
                elif task.exception():
                    status[name] = f"error: {task.exception()}"
                else:
                    status[name] = "completed"
            else:
                status[name] = "running"
        return status


class RateLimiter:
    """
    Token bucket rate limiter implementation.
    
    This class provides rate limiting using the token bucket algorithm,
    suitable for controlling API request rates.
    
    Example:
        >>> limiter = RateLimiter(rate=10, burst=20)
        >>> if await limiter.acquire():
        ...     await make_api_call()
    """
    
    def __init__(self, rate: float, burst: Optional[int] = None):
        """
        Initialize rate limiter.
        
        Args:
            rate: Tokens per second
            burst: Maximum burst size (defaults to rate)
        """
        self.rate = rate
        self.burst = burst or int(rate)
        self.tokens = float(self.burst)
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1, wait: bool = True) -> bool:
        """
        Acquire tokens from the bucket.
        
        Args:
            tokens: Number of tokens to acquire
            wait: Whether to wait if not enough tokens
            
        Returns:
            bool: True if tokens were acquired
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.last_update = now
            
            # Add new tokens based on elapsed time
            self.tokens = min(
                self.burst,
                self.tokens + elapsed * self.rate
            )
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            if not wait:
                return False
            
            # Calculate wait time
            needed = tokens - self.tokens
            wait_time = needed / self.rate
            await asyncio.sleep(wait_time)
            
            # Try again after waiting
            return await self.acquire(tokens, wait=False)


def create_logger(
    name: str,
    level: str = "INFO",
    format_string: Optional[str] = None,
    handlers: Optional[List[logging.Handler]] = None
) -> logging.Logger:
    """
    Create a configured logger instance.
    
    Args:
        name: Logger name
        level: Logging level
        format_string: Custom format string
        handlers: Custom handlers
        
    Returns:
        Configured logger instance
        
    Example:
        >>> logger = create_logger("neural.trading", level="DEBUG")
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Default format
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(message)s"
        )
    
    formatter = logging.Formatter(format_string)
    
    # Add handlers
    if handlers is None:
        # Default to console handler
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        for handler in handlers:
            handler.setFormatter(formatter)
            logger.addHandler(handler)
    
    return logger


def generate_id(prefix: str = "", length: int = 8) -> str:
    """
    Generate a unique ID with optional prefix.
    
    Args:
        prefix: Optional prefix for the ID
        length: Length of random part
        
    Returns:
        Unique ID string
        
    Example:
        >>> id = generate_id("trade", 12)
        >>> # Returns: "trade_a3f8b2c1d9e4"
    """
    import secrets
    random_part = secrets.token_hex(length // 2)
    
    if prefix:
        return f"{prefix}_{random_part}"
    return random_part


def safe_json_parse(data: Union[str, bytes], default: Any = None) -> Any:
    """
    Safely parse JSON data with fallback.
    
    Args:
        data: JSON string or bytes
        default: Default value if parsing fails
        
    Returns:
        Parsed data or default value
        
    Example:
        >>> data = safe_json_parse('{"key": "value"}', default={})
    """
    try:
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        return json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.debug(f"JSON parse error: {e}")
        return default


def calculate_checksum(data: Union[str, bytes]) -> str:
    """
    Calculate SHA256 checksum of data.
    
    Args:
        data: Data to checksum
        
    Returns:
        Hex string of checksum
        
    Example:
        >>> checksum = calculate_checksum("important data")
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    return hashlib.sha256(data).hexdigest()


def timestamp_to_iso(timestamp: Union[int, float]) -> str:
    """
    Convert timestamp to ISO format string.
    
    Args:
        timestamp: Unix timestamp
        
    Returns:
        ISO format datetime string
        
    Example:
        >>> iso_str = timestamp_to_iso(1699564800)
    """
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return dt.isoformat()


def iso_to_timestamp(iso_string: str) -> float:
    """
    Convert ISO format string to timestamp.
    
    Args:
        iso_string: ISO format datetime string
        
    Returns:
        Unix timestamp
        
    Example:
        >>> ts = iso_to_timestamp("2024-01-01T00:00:00Z")
    """
    dt = datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
    return dt.timestamp()


class MovingAverage:
    """
    Calculate moving average of values.
    
    Useful for smoothing metrics and detecting trends.
    
    Example:
        >>> avg = MovingAverage(window_size=10)
        >>> avg.add(5.0)
        >>> current_avg = avg.get()
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize moving average calculator.
        
        Args:
            window_size: Number of values to average
        """
        self.window_size = window_size
        self._values: deque = deque(maxlen=window_size)
        self._sum = 0.0
    
    def add(self, value: float) -> None:
        """Add a value to the moving average."""
        if len(self._values) == self.window_size:
            # Remove oldest value from sum
            self._sum -= self._values[0]
        
        self._values.append(value)
        self._sum += value
    
    def get(self) -> Optional[float]:
        """Get current moving average."""
        if not self._values:
            return None
        return self._sum / len(self._values)
    
    def reset(self) -> None:
        """Reset the moving average."""
        self._values.clear()
        self._sum = 0.0


def batch_items(items: List[T], batch_size: int) -> List[List[T]]:
    """
    Split items into batches.
    
    Args:
        items: List of items to batch
        batch_size: Maximum items per batch
        
    Returns:
        List of batches
        
    Example:
        >>> batches = batch_items(range(10), 3)
        >>> # Returns: [[0,1,2], [3,4,5], [6,7,8], [9]]
    """
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    
    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i:i + batch_size])
    
    return batches


async def run_with_timeout(
    coro: Coroutine,
    timeout: float,
    default: Any = None
) -> Any:
    """
    Run coroutine with timeout.
    
    Args:
        coro: Coroutine to run
        timeout: Timeout in seconds
        default: Default value if timeout occurs
        
    Returns:
        Coroutine result or default value
        
    Example:
        >>> result = await run_with_timeout(
        ...     fetch_data(),
        ...     timeout=5.0,
        ...     default={}
        ... )
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"Coroutine timed out after {timeout}s")
        return default


def validate_config(
    config: Dict[str, Any],
    required_fields: List[str],
    optional_fields: Optional[List[str]] = None
) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        required_fields: List of required field names
        optional_fields: List of optional field names
        
    Returns:
        bool: True if valid
        
    Raises:
        ValueError: If validation fails
        
    Example:
        >>> validate_config(
        ...     {"api_key": "xxx", "url": "https://api.example.com"},
        ...     required_fields=["api_key", "url"],
        ...     optional_fields=["timeout"]
        ... )
    """
    # Check required fields
    missing = []
    for field in required_fields:
        if field not in config:
            missing.append(field)
    
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    
    # Check for unknown fields
    known_fields = set(required_fields)
    if optional_fields:
        known_fields.update(optional_fields)
    
    unknown = []
    for field in config:
        if field not in known_fields:
            unknown.append(field)
    
    if unknown:
        logger.warning(f"Unknown configuration fields: {unknown}")
    
    return True


class PerformanceTimer:
    """
    Context manager for timing code execution.
    
    Example:
        >>> with PerformanceTimer("api_call") as timer:
        ...     result = await api.call()
        >>> print(f"Took {timer.elapsed:.3f} seconds")
    """
    
    def __init__(self, name: str = "operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        
        if exc_type is None:
            logger.debug(f"{self.name} took {self.elapsed:.3f}s")
        else:
            logger.debug(f"{self.name} failed after {self.elapsed:.3f}s")


def deep_merge(dict1: Dict, dict2: Dict) -> Dict:
    """
    Deep merge two dictionaries.
    
    Args:
        dict1: Base dictionary
        dict2: Dictionary to merge in
        
    Returns:
        Merged dictionary
        
    Example:
        >>> config = deep_merge(defaults, user_config)
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


async def gather_with_errors(
    *coros,
    return_exceptions: bool = True
) -> List[Union[Any, Exception]]:
    """
    Gather coroutines and handle errors gracefully.
    
    Args:
        coros: Coroutines to run
        return_exceptions: Whether to return exceptions or raise
        
    Returns:
        List of results (and exceptions if return_exceptions=True)
        
    Example:
        >>> results = await gather_with_errors(
        ...     fetch_data1(),
        ...     fetch_data2(),
        ...     fetch_data3()
        ... )
    """
    results = await asyncio.gather(
        *coros,
        return_exceptions=return_exceptions
    )
    
    # Log any exceptions
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Coroutine {i} failed: {result}")
    
    return results