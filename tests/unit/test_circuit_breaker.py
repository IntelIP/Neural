#!/usr/bin/env python3
"""
Circuit Breaker Test Suite
Tests resilience patterns with simulated failures
"""

import asyncio
import random
import logging
import time

# Import our resilience components
from data_pipeline.reliability.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitOpenException
from data_pipeline.reliability.resilience_coordinator import ResilienceCoordinator, ServicePriority
from data_pipeline.reliability.health_monitor import HealthMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Simulated service functions
async def reliable_service() -> str:
    """Always succeeds"""
    await asyncio.sleep(0.01)  # Simulate some work
    return "success"


async def flaky_service() -> str:
    """Fails 40% of the time"""
    await asyncio.sleep(0.01)
    if random.random() < 0.4:
        raise Exception("Service temporarily unavailable")
    return "success"


async def degrading_service(call_count: int) -> str:
    """Degrades over time"""
    await asyncio.sleep(0.01)
    # Fail more as calls increase
    failure_rate = min(0.9, call_count * 0.05)
    if random.random() < failure_rate:
        raise Exception(f"Service degraded (failure rate: {failure_rate:.1%})")
    return "success"


async def slow_service(delay: float = 5.0) -> str:
    """Slow service that might timeout"""
    await asyncio.sleep(delay)
    return "success"


class CircuitBreakerTests:
    """Test suite for circuit breaker functionality"""
    
    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
    
    async def test_basic_circuit_breaker(self):
        """Test basic circuit breaker operation"""
        logger.info("\n=== Testing Basic Circuit Breaker ===")
        
        config = CircuitBreakerConfig(
            name="test_basic",
            failure_threshold=3,
            success_threshold=2,
            timeout=1.0,
            half_open_interval=1.0
        )
        
        breaker = CircuitBreaker(config)
        
        # Test successful calls
        logger.info("Testing successful calls...")
        for i in range(3):
            result = await breaker.call(reliable_service)
            assert result == "success"
        logger.info("✓ Successful calls work")
        
        # Test failure threshold
        logger.info("Testing failure threshold...")
        failures = 0
        for i in range(10):
            try:
                await breaker.call(flaky_service)
            except Exception:
                failures += 1
                if breaker.state.value == "open":
                    logger.info(f"✓ Circuit opened after {failures} failures")
                    break
        
        # Test circuit open rejection
        logger.info("Testing circuit open rejection...")
        try:
            await breaker.call(reliable_service)
            logger.error("✗ Circuit should reject calls when open")
            self.tests_failed += 1
        except CircuitOpenException:
            logger.info("✓ Circuit correctly rejects calls when open")
            self.tests_passed += 1
        
        # Wait for half-open
        logger.info("Waiting for half-open state...")
        await asyncio.sleep(1.5)
        
        # Test recovery
        logger.info("Testing recovery...")
        for i in range(3):
            try:
                await breaker.call(reliable_service)
                if breaker.state.value == "closed":
                    logger.info("✓ Circuit recovered and closed")
                    self.tests_passed += 1
                    break
            except CircuitOpenException:
                # Still in half-open, limited traffic
                pass
        
        # Get metrics
        metrics = breaker.get_metrics()
        logger.info(f"Circuit breaker metrics: {metrics}")
    
    async def test_timeout_protection(self):
        """Test timeout protection"""
        logger.info("\n=== Testing Timeout Protection ===")
        
        config = CircuitBreakerConfig(
            name="test_timeout",
            failure_threshold=2,
            timeout=0.5  # 500ms timeout
        )
        
        breaker = CircuitBreaker(config)
        
        # Test timeout
        logger.info("Testing slow service timeout...")
        try:
            await breaker.call(slow_service, 2.0)  # Will timeout
            logger.error("✗ Should have timed out")
            self.tests_failed += 1
        except Exception as e:
            if "timed out" in str(e):
                logger.info("✓ Service correctly timed out")
                self.tests_passed += 1
            else:
                logger.error(f"✗ Unexpected error: {e}")
                self.tests_failed += 1
    
    async def test_sliding_window(self):
        """Test sliding window failure tracking"""
        logger.info("\n=== Testing Sliding Window ===")
        
        config = CircuitBreakerConfig(
            name="test_window",
            failure_threshold=5,
            window_size=10  # 10 second window
        )
        
        breaker = CircuitBreaker(config)
        
        # Generate some failures
        for i in range(4):
            try:
                await breaker.call(flaky_service)
            except:
                pass
        
        # Check window stats
        window_stats = breaker.sliding_window.get_stats()
        logger.info(f"Window stats after 4 calls: {window_stats}")
        
        # Wait for window to expire
        logger.info("Waiting for sliding window to expire...")
        await asyncio.sleep(11)
        
        # Check stats again
        window_stats = breaker.sliding_window.get_stats()
        logger.info(f"Window stats after expiry: {window_stats}")
        
        if window_stats['failures'] == 0:
            logger.info("✓ Sliding window correctly expired old events")
            self.tests_passed += 1
        else:
            logger.error("✗ Sliding window did not expire events")
            self.tests_failed += 1
    
    async def test_resilience_coordinator(self):
        """Test resilience coordinator with multiple services"""
        logger.info("\n=== Testing Resilience Coordinator ===")
        
        coordinator = ResilienceCoordinator()
        
        # Register services
        critical_breaker = coordinator.register_service(
            name="critical_service",
            priority=ServicePriority.CRITICAL,
            failure_threshold=2
        )
        
        medium_breaker = coordinator.register_service(
            name="medium_service",
            priority=ServicePriority.MEDIUM,
            failure_threshold=3
        )
        
        low_breaker = coordinator.register_service(
            name="low_service",
            priority=ServicePriority.LOW,
            failure_threshold=5
        )
        
        # Test normal operation
        logger.info("Testing normal operation...")
        result = await coordinator.call_with_resilience(
            "critical_service",
            reliable_service
        )
        assert result == "success"
        logger.info("✓ Normal operation works")
        
        # Simulate failures to trigger degradation
        logger.info("Simulating failures to trigger degradation...")
        
        call_count = 0
        for service_name in ["low_service", "medium_service"]:
            for i in range(10):
                try:
                    call_count += 1
                    await coordinator.call_with_resilience(
                        service_name,
                        degrading_service,
                        call_count
                    )
                except:
                    pass
        
        # Check degradation level
        status = coordinator.get_status()
        logger.info(f"Coordinator status: {status}")
        
        if status['degradation_level'] != "NORMAL":
            logger.info(f"✓ System degraded to {status['degradation_level']}")
            self.tests_passed += 1
        else:
            logger.error("✗ System should be degraded")
            self.tests_failed += 1
    
    async def test_health_monitoring(self):
        """Test health monitoring system"""
        logger.info("\n=== Testing Health Monitoring ===")
        
        monitor = HealthMonitor()
        
        # Register health probe
        probe = monitor.register_probe(
            service_name="test_service",
            check_func=lambda: random.random() > 0.3,  # 70% healthy
            interval=1.0,
            healthy_threshold=2,
            unhealthy_threshold=2
        )
        
        # Start monitoring
        await monitor.start()
        logger.info("Health monitoring started")
        
        # Let it run for a bit
        await asyncio.sleep(5)
        
        # Check status
        status = monitor.get_status()
        logger.info(f"Health monitor status: {status}")
        
        # Get history
        history = monitor.get_health_history("test_service", limit=5)
        logger.info(f"Health check history: {len(history)} checks")
        
        # Stop monitoring
        await monitor.stop()
        
        if len(history) > 0:
            logger.info("✓ Health monitoring collected data")
            self.tests_passed += 1
        else:
            logger.error("✗ No health data collected")
            self.tests_failed += 1
    
    async def test_cascading_failure_prevention(self):
        """Test prevention of cascading failures"""
        logger.info("\n=== Testing Cascading Failure Prevention ===")
        
        coordinator = ResilienceCoordinator()
        
        # Register services with dependencies
        db_breaker = coordinator.register_service(
            name="database",
            priority=ServicePriority.CRITICAL,
            failure_threshold=2
        )
        
        api_breaker = coordinator.register_service(
            name="api",
            priority=ServicePriority.HIGH,
            failure_threshold=3,
            dependencies=["database"]
        )
        
        frontend_breaker = coordinator.register_service(
            name="frontend",
            priority=ServicePriority.MEDIUM,
            failure_threshold=5,
            dependencies=["api"]
        )
        
        # Simulate database failure
        logger.info("Simulating database failure...")
        for i in range(3):
            try:
                await db_breaker.call(lambda: 1/0)  # Force failure
            except:
                pass
        
        # Check if dependent services were preemptively opened
        await asyncio.sleep(0.1)  # Let state changes propagate
        
        status = coordinator.get_status()
        logger.info(f"Service states after database failure: {status['services']}")
        
        prevented = coordinator.metrics.cascading_failures_prevented
        if prevented > 0:
            logger.info(f"✓ Prevented {prevented} cascading failures")
            self.tests_passed += 1
        else:
            logger.warning("⚠ No cascading failures prevented (might be expected)")
    
    async def run_all_tests(self):
        """Run all circuit breaker tests"""
        logger.info("=" * 50)
        logger.info("CIRCUIT BREAKER TEST SUITE")
        logger.info("=" * 50)
        
        start_time = time.time()
        
        # Run tests
        await self.test_basic_circuit_breaker()
        await self.test_timeout_protection()
        await self.test_sliding_window()
        await self.test_resilience_coordinator()
        await self.test_health_monitoring()
        await self.test_cascading_failure_prevention()
        
        # Summary
        elapsed = time.time() - start_time
        total_tests = self.tests_passed + self.tests_failed
        
        logger.info("\n" + "=" * 50)
        logger.info("TEST SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Tests Passed: {self.tests_passed}/{total_tests}")
        logger.info(f"Tests Failed: {self.tests_failed}/{total_tests}")
        logger.info(f"Time Elapsed: {elapsed:.2f}s")
        
        if self.tests_failed == 0:
            logger.info("✅ ALL TESTS PASSED")
        else:
            logger.error(f"❌ {self.tests_failed} TESTS FAILED")
        
        return self.tests_failed == 0


async def main():
    """Main test runner"""
    tests = CircuitBreakerTests()
    success = await tests.run_all_tests()
    
    if not success:
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())