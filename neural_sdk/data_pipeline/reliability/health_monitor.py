"""
Health Probe System - Service health monitoring
Performs periodic health checks and detects recovery
Optimized for low-latency operations
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import aiohttp

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Service health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    service_name: str
    status: HealthStatus
    latency_ms: float
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None


@dataclass
class ServiceHealth:
    """Service health tracking"""
    name: str
    current_status: HealthStatus = HealthStatus.UNKNOWN
    last_check: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    average_latency_ms: float = 0.0
    availability_percentage: float = 100.0
    check_history: List[HealthCheckResult] = field(default_factory=list)


class HealthProbe:
    """
    Health probe for individual service
    
    Features:
    - Async health checks
    - Latency tracking
    - Success/failure criteria
    - Recovery detection
    """
    
    def __init__(
        self,
        service_name: str,
        check_func: Optional[Callable] = None,
        timeout: float = 5.0,
        healthy_threshold: int = 2,
        unhealthy_threshold: int = 2
    ):
        """
        Initialize health probe
        
        Args:
            service_name: Name of the service
            check_func: Custom health check function
            timeout: Timeout for health check
            healthy_threshold: Consecutive successes to mark healthy
            unhealthy_threshold: Consecutive failures to mark unhealthy
        """
        self.service_name = service_name
        self.check_func = check_func
        self.timeout = timeout
        self.healthy_threshold = healthy_threshold
        self.unhealthy_threshold = unhealthy_threshold
        
        self.health = ServiceHealth(name=service_name)
        
    async def check(self) -> HealthCheckResult:
        """
        Perform health check
        
        Returns:
            Health check result
        """
        start_time = time.time()
        
        try:
            if self.check_func:
                # Custom health check
                result = await asyncio.wait_for(
                    self.check_func(),
                    timeout=self.timeout
                )
                
                # Interpret result
                if isinstance(result, bool):
                    status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                    details = {}
                elif isinstance(result, dict):
                    status = HealthStatus(result.get("status", "unknown"))
                    details = result.get("details", {})
                else:
                    status = HealthStatus.HEALTHY
                    details = {"result": str(result)}
            else:
                # Default health check (service is reachable)
                status = HealthStatus.HEALTHY
                details = {"default_check": True}
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Create result
            result = HealthCheckResult(
                service_name=self.service_name,
                status=status,
                latency_ms=latency_ms,
                details=details
            )
            
            # Update health tracking
            await self._update_health(result)
            
            return result
            
        except asyncio.TimeoutError:
            result = HealthCheckResult(
                service_name=self.service_name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=self.timeout * 1000,
                error="Health check timed out"
            )
            await self._update_health(result)
            return result
            
        except Exception as e:
            result = HealthCheckResult(
                service_name=self.service_name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )
            await self._update_health(result)
            return result
    
    async def _update_health(self, result: HealthCheckResult):
        """Update health tracking based on result"""
        self.health.last_check = result.timestamp
        
        # Update history (keep last 100)
        self.health.check_history.append(result)
        if len(self.health.check_history) > 100:
            self.health.check_history.pop(0)
        
        # Update latency average
        self.health.average_latency_ms = (
            0.9 * self.health.average_latency_ms + 0.1 * result.latency_ms
        )
        
        # Update consecutive counts
        if result.status == HealthStatus.HEALTHY:
            self.health.consecutive_successes += 1
            self.health.consecutive_failures = 0
        else:
            self.health.consecutive_failures += 1
            self.health.consecutive_successes = 0
        
        # Determine overall status
        old_status = self.health.current_status
        
        if self.health.consecutive_successes >= self.healthy_threshold:
            self.health.current_status = HealthStatus.HEALTHY
        elif self.health.consecutive_failures >= self.unhealthy_threshold:
            self.health.current_status = HealthStatus.UNHEALTHY
        elif result.status == HealthStatus.DEGRADED:
            self.health.current_status = HealthStatus.DEGRADED
        
        # Log status changes
        if old_status != self.health.current_status:
            logger.info(
                f"Service '{self.service_name}' health changed: "
                f"{old_status.value} -> {self.health.current_status.value}"
            )
        
        # Update availability
        healthy_count = sum(
            1 for h in self.health.check_history
            if h.status == HealthStatus.HEALTHY
        )
        total_count = len(self.health.check_history)
        
        if total_count > 0:
            self.health.availability_percentage = (healthy_count / total_count) * 100


class HealthMonitor:
    """
    Centralized health monitoring system
    
    Features:
    - Multiple service monitoring
    - Configurable probe intervals
    - Aggregate health metrics
    - Alert generation
    """
    
    def __init__(self):
        """Initialize health monitor"""
        self.probes: Dict[str, HealthProbe] = {}
        self.probe_intervals: Dict[str, float] = {}
        self.probe_tasks: Dict[str, asyncio.Task] = {}
        
        self.is_running = False
        self.on_health_change: Optional[Callable] = None
        
        # Aggregate metrics
        self.last_check_time: Optional[datetime] = None
        self.overall_health: HealthStatus = HealthStatus.UNKNOWN
        
        logger.info("HealthMonitor initialized")
    
    def register_probe(
        self,
        service_name: str,
        check_func: Optional[Callable] = None,
        interval: float = 30.0,
        timeout: float = 5.0,
        healthy_threshold: int = 2,
        unhealthy_threshold: int = 2
    ) -> HealthProbe:
        """
        Register a health probe
        
        Args:
            service_name: Service to monitor
            check_func: Health check function
            interval: Check interval in seconds
            timeout: Check timeout
            healthy_threshold: Successes for healthy
            unhealthy_threshold: Failures for unhealthy
            
        Returns:
            Health probe instance
        """
        probe = HealthProbe(
            service_name=service_name,
            check_func=check_func,
            timeout=timeout,
            healthy_threshold=healthy_threshold,
            unhealthy_threshold=unhealthy_threshold
        )
        
        self.probes[service_name] = probe
        self.probe_intervals[service_name] = interval
        
        # Start monitoring if already running
        if self.is_running:
            self.probe_tasks[service_name] = asyncio.create_task(
                self._probe_loop(service_name)
            )
        
        logger.info(f"Registered health probe for '{service_name}' with {interval}s interval")
        
        return probe
    
    async def start(self):
        """Start health monitoring"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start probe tasks
        for service_name in self.probes:
            self.probe_tasks[service_name] = asyncio.create_task(
                self._probe_loop(service_name)
            )
        
        # Start aggregate monitoring
        asyncio.create_task(self._aggregate_health_loop())
        
        logger.info("Health monitoring started")
    
    async def stop(self):
        """Stop health monitoring"""
        self.is_running = False
        
        # Cancel probe tasks
        for task in self.probe_tasks.values():
            task.cancel()
        
        self.probe_tasks.clear()
        
        logger.info("Health monitoring stopped")
    
    async def _probe_loop(self, service_name: str):
        """Health probe loop for a service"""
        probe = self.probes[service_name]
        interval = self.probe_intervals[service_name]
        
        while self.is_running:
            try:
                result = await probe.check()
                
                # Notify if health changed
                if self.on_health_change and result.status != probe.health.current_status:
                    await self.on_health_change(service_name, result)
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Probe error for '{service_name}': {e}")
                await asyncio.sleep(interval)
    
    async def _aggregate_health_loop(self):
        """Calculate aggregate health metrics"""
        while self.is_running:
            try:
                await self._update_overall_health()
                await asyncio.sleep(10)  # Update every 10 seconds
            except Exception as e:
                logger.error(f"Aggregate health error: {e}")
                await asyncio.sleep(10)
    
    async def _update_overall_health(self):
        """Update overall system health"""
        if not self.probes:
            self.overall_health = HealthStatus.UNKNOWN
            return
        
        # Count health states
        healthy_count = sum(
            1 for probe in self.probes.values()
            if probe.health.current_status == HealthStatus.HEALTHY
        )
        degraded_count = sum(
            1 for probe in self.probes.values()
            if probe.health.current_status == HealthStatus.DEGRADED
        )
        unhealthy_count = sum(
            1 for probe in self.probes.values()
            if probe.health.current_status == HealthStatus.UNHEALTHY
        )
        
        total = len(self.probes)
        
        # Determine overall health
        if unhealthy_count > total * 0.5:
            self.overall_health = HealthStatus.UNHEALTHY
        elif degraded_count > total * 0.3 or unhealthy_count > 0:
            self.overall_health = HealthStatus.DEGRADED
        elif healthy_count == total:
            self.overall_health = HealthStatus.HEALTHY
        else:
            self.overall_health = HealthStatus.DEGRADED
        
        self.last_check_time = datetime.now()
    
    async def check_service(self, service_name: str) -> Optional[HealthCheckResult]:
        """
        Manually check a service
        
        Args:
            service_name: Service to check
            
        Returns:
            Health check result
        """
        if service_name not in self.probes:
            return None
        
        return await self.probes[service_name].check()
    
    def get_status(self) -> Dict[str, Any]:
        """Get health monitor status"""
        return {
            "overall_health": self.overall_health.value,
            "services": {
                name: {
                    "status": probe.health.current_status.value,
                    "availability": probe.health.availability_percentage,
                    "average_latency_ms": probe.health.average_latency_ms,
                    "consecutive_failures": probe.health.consecutive_failures,
                    "last_check": (
                        probe.health.last_check.isoformat()
                        if probe.health.last_check else None
                    )
                }
                for name, probe in self.probes.items()
            },
            "is_running": self.is_running,
            "last_aggregate_check": (
                self.last_check_time.isoformat()
                if self.last_check_time else None
            )
        }
    
    def get_health_history(self, service_name: str, limit: int = 10) -> List[HealthCheckResult]:
        """
        Get health check history for a service
        
        Args:
            service_name: Service name
            limit: Number of results to return
            
        Returns:
            List of health check results
        """
        if service_name not in self.probes:
            return []
        
        history = self.probes[service_name].health.check_history
        return history[-limit:] if history else []


# Predefined health check functions for common services

async def http_health_check(url: str, expected_status: int = 200) -> Dict[str, Any]:
    """
    HTTP endpoint health check
    
    Args:
        url: Endpoint URL
        expected_status: Expected HTTP status code
        
    Returns:
        Health check result
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=5) as response:
                is_healthy = response.status == expected_status
                
                return {
                    "status": "healthy" if is_healthy else "unhealthy",
                    "details": {
                        "status_code": response.status,
                        "expected": expected_status
                    }
                }
    except Exception as e:
        return {
            "status": "unhealthy",
            "details": {"error": str(e)}
        }


async def websocket_health_check(ws_client) -> bool:
    """
    WebSocket connection health check
    
    Args:
        ws_client: WebSocket client instance
        
    Returns:
        True if healthy
    """
    try:
        # Check if WebSocket is connected
        if hasattr(ws_client, 'ws') and ws_client.ws:
            # Try to ping
            if hasattr(ws_client.ws, 'ping'):
                await ws_client.ws.ping()
            return True
        return False
    except:
        return False


async def database_health_check(db_session) -> Dict[str, Any]:
    """
    Database connection health check
    
    Args:
        db_session: Database session
        
    Returns:
        Health check result
    """
    try:
        # Simple query to test connection
        result = await db_session.execute("SELECT 1")
        
        return {
            "status": "healthy",
            "details": {"connected": True}
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "details": {"error": str(e)}
        }


# Global health monitor instance
_health_monitor: Optional[HealthMonitor] = None


def get_health_monitor() -> HealthMonitor:
    """Get or create global health monitor"""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    return _health_monitor