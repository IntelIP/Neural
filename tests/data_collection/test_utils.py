"""
Tests for utility functions.
"""

import asyncio
import json
import time
import pytest
from unittest.mock import Mock, AsyncMock, patch

from neural.data_collection.utils import (
    retry,
    RetryConfig,
    AsyncTaskManager,
    RateLimiter,
    create_logger,
    generate_id,
    safe_json_parse,
    calculate_checksum,
    timestamp_to_iso,
    iso_to_timestamp,
    MovingAverage,
    batch_items,
    run_with_timeout,
    validate_config,
    PerformanceTimer,
    deep_merge,
    gather_with_errors
)
from neural.data_collection.exceptions import (
    RetryableError,
    TransientError,
    TimeoutError
)


class TestRetryDecorator:
    """Test retry decorator functionality."""
    
    @pytest.mark.asyncio
    async def test_async_retry_success(self):
        call_count = 0
        
        @retry(RetryConfig(max_attempts=3))
        async def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RetryableError("Temporary failure")
            return "success"
        
        result = await flaky_function()
        
        assert result == "success"
        assert call_count == 2
    
    def test_sync_retry_success(self):
        call_count = 0
        
        @retry(RetryConfig(max_attempts=3, initial_delay=0.01))
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TransientError("Temporary failure")
            return "success"
        
        result = flaky_function()
        
        assert result == "success"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_retry_max_attempts_exceeded(self):
        @retry(RetryConfig(max_attempts=2, initial_delay=0.01))
        async def always_fails():
            raise RetryableError("Always fails")
        
        with pytest.raises(RetryableError):
            await always_fails()
    
    @pytest.mark.asyncio
    async def test_retry_with_custom_exceptions(self):
        @retry(
            RetryConfig(max_attempts=3, initial_delay=0.01),
            exceptions=(ValueError,)
        )
        async def custom_exception():
            raise ValueError("Custom error")
        
        with pytest.raises(ValueError):
            await custom_exception()
    
    @pytest.mark.asyncio
    async def test_retry_exponential_backoff(self):
        start_time = time.time()
        
        @retry(RetryConfig(
            max_attempts=3,
            initial_delay=0.01,
            exponential_base=2.0,
            jitter=False
        ))
        async def measure_backoff():
            raise RetryableError("Fail")
        
        try:
            await measure_backoff()
        except RetryableError:
            pass
        
        elapsed = time.time() - start_time
        # Should have delays of 0.01 + 0.02 = 0.03 minimum
        assert elapsed >= 0.03


class TestAsyncTaskManager:
    """Test AsyncTaskManager functionality."""
    
    @pytest.mark.asyncio
    async def test_start_and_stop_task(self):
        manager = AsyncTaskManager()
        
        async def simple_task():
            await asyncio.sleep(0.1)
        
        await manager.start_task("test", simple_task())
        
        status = manager.get_status()
        assert "test" in status
        assert status["test"] == "running"
        
        await manager.stop_task("test")
        
        status = manager.get_status()
        assert "test" not in status
    
    @pytest.mark.asyncio
    async def test_task_restart_on_error(self):
        manager = AsyncTaskManager()
        call_count = 0
        
        async def failing_task():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Task failed")
            await asyncio.sleep(10)  # Keep running
        
        await manager.start_task(
            "retry_task",
            failing_task(),
            restart_on_error=True
        )
        
        await asyncio.sleep(0.1)
        await manager.stop_task("retry_task")
        
        assert call_count >= 2
    
    @pytest.mark.asyncio
    async def test_stop_all_tasks(self):
        manager = AsyncTaskManager()
        
        async def long_task():
            await asyncio.sleep(10)
        
        await manager.start_task("task1", long_task())
        await manager.start_task("task2", long_task())
        
        await manager.stop_all()
        
        status = manager.get_status()
        assert len(status) == 0


class TestRateLimiter:
    """Test RateLimiter functionality."""
    
    @pytest.mark.asyncio
    async def test_basic_rate_limiting(self):
        limiter = RateLimiter(rate=10, burst=10)
        
        # Should allow burst
        for _ in range(10):
            assert await limiter.acquire() is True
        
        # Should be rate limited
        assert await limiter.acquire(wait=False) is False
    
    @pytest.mark.asyncio
    async def test_rate_limiter_refill(self):
        limiter = RateLimiter(rate=100, burst=10)
        
        # Use all tokens
        for _ in range(10):
            await limiter.acquire()
        
        # Wait for refill
        await asyncio.sleep(0.1)
        
        # Should have new tokens
        assert await limiter.acquire() is True
    
    @pytest.mark.asyncio
    async def test_rate_limiter_wait(self):
        limiter = RateLimiter(rate=10, burst=1)
        
        # Use the burst
        await limiter.acquire()
        
        # Should wait and then succeed
        start = time.time()
        result = await limiter.acquire(wait=True)
        elapsed = time.time() - start
        
        assert result is True
        assert elapsed >= 0.09  # Should wait ~0.1s


class TestUtilityFunctions:
    """Test various utility functions."""
    
    def test_generate_id(self):
        # Without prefix
        id1 = generate_id()
        assert len(id1) == 8
        
        # With prefix
        id2 = generate_id(prefix="test", length=12)
        assert id2.startswith("test_")
        assert len(id2) == 17  # "test_" + 12 chars
        
        # Should be unique
        id3 = generate_id()
        assert id1 != id3
    
    def test_safe_json_parse(self):
        # Valid JSON
        result = safe_json_parse('{"key": "value"}')
        assert result == {"key": "value"}
        
        # Invalid JSON
        result = safe_json_parse('invalid json', default={})
        assert result == {}
        
        # Bytes input
        result = safe_json_parse(b'{"key": "value"}')
        assert result == {"key": "value"}
    
    def test_calculate_checksum(self):
        # String input
        checksum1 = calculate_checksum("test data")
        assert len(checksum1) == 64  # SHA256 hex length
        
        # Bytes input
        checksum2 = calculate_checksum(b"test data")
        assert checksum1 == checksum2
        
        # Different data
        checksum3 = calculate_checksum("different data")
        assert checksum1 != checksum3
    
    def test_timestamp_conversions(self):
        # Timestamp to ISO
        timestamp = 1699564800
        iso_str = timestamp_to_iso(timestamp)
        assert "2023-11-09" in iso_str
        
        # ISO to timestamp
        converted = iso_to_timestamp(iso_str)
        assert converted == timestamp
    
    def test_moving_average(self):
        avg = MovingAverage(window_size=3)
        
        assert avg.get() is None
        
        avg.add(1.0)
        assert avg.get() == 1.0
        
        avg.add(2.0)
        assert avg.get() == 1.5
        
        avg.add(3.0)
        assert avg.get() == 2.0
        
        # Window full, should drop oldest
        avg.add(4.0)
        assert avg.get() == 3.0  # (2+3+4)/3
        
        avg.reset()
        assert avg.get() is None
    
    def test_batch_items(self):
        items = list(range(10))
        
        # Even batches
        batches = batch_items(items, 3)
        assert len(batches) == 4
        assert batches[0] == [0, 1, 2]
        assert batches[-1] == [9]
        
        # Single batch
        batches = batch_items(items, 20)
        assert len(batches) == 1
        assert batches[0] == items
        
        # Invalid batch size
        with pytest.raises(ValueError):
            batch_items(items, 0)
    
    @pytest.mark.asyncio
    async def test_run_with_timeout(self):
        # Success within timeout
        async def quick_task():
            await asyncio.sleep(0.01)
            return "done"
        
        result = await run_with_timeout(quick_task(), timeout=1.0)
        assert result == "done"
        
        # Timeout with default
        async def slow_task():
            await asyncio.sleep(1.0)
            return "done"
        
        result = await run_with_timeout(
            slow_task(),
            timeout=0.1,
            default="timeout"
        )
        assert result == "timeout"
    
    def test_validate_config(self):
        config = {
            "api_key": "secret",
            "url": "https://api.com",
            "timeout": 30
        }
        
        # Valid config
        result = validate_config(
            config,
            required_fields=["api_key", "url"],
            optional_fields=["timeout"]
        )
        assert result is True
        
        # Missing required field
        with pytest.raises(ValueError) as exc:
            validate_config(
                config,
                required_fields=["api_key", "missing"]
            )
        assert "missing" in str(exc.value)
    
    def test_performance_timer(self):
        with PerformanceTimer("test") as timer:
            time.sleep(0.01)
        
        assert timer.elapsed >= 0.01
        assert timer.name == "test"
    
    def test_deep_merge(self):
        dict1 = {
            "a": 1,
            "b": {"c": 2, "d": 3},
            "e": [1, 2]
        }
        
        dict2 = {
            "b": {"c": 20, "f": 4},
            "g": 5
        }
        
        result = deep_merge(dict1, dict2)
        
        assert result["a"] == 1
        assert result["b"]["c"] == 20  # Overwritten
        assert result["b"]["d"] == 3   # Preserved
        assert result["b"]["f"] == 4   # Added
        assert result["g"] == 5        # Added
        assert result["e"] == [1, 2]   # Preserved
    
    @pytest.mark.asyncio
    async def test_gather_with_errors(self):
        async def success():
            return "success"
        
        async def failure():
            raise ValueError("Failed")
        
        results = await gather_with_errors(
            success(),
            failure(),
            success()
        )
        
        assert len(results) == 3
        assert results[0] == "success"
        assert isinstance(results[1], ValueError)
        assert results[2] == "success"