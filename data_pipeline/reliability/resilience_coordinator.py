"""
Resilience Coordinator - Manages circuit breakers across services
Coordinates failure handling and degraded mode operations
Prevents cascading failures in the trading system
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field

from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState
from ..state_manager import get_state_manager

logger = logging.getLogger(__name__)


class ServicePriority(Enum):
    """Service priority levels for degradation"""
    CRITICAL = 0    # Must stay up - trading core
    HIGH = 1        # Important - risk management
    MEDIUM = 2      # Useful - market data
    LOW = 3         # Optional - enrichment data


class DegradationLevel(Enum):
    """System degradation levels"""
    NORMAL = 0      # All services operational
    LEVEL_1 = 1     # Disable low priority services
    LEVEL_2 = 2     # Disable medium priority services
    LEVEL_3 = 3     # Read-only mode, no new trades
    EMERGENCY = 4   # Emergency stop, close all positions


@dataclass
class ServiceConfig:
    """Configuration for a managed service"""
    name: str
    priority: ServicePriority
    circuit_breaker: CircuitBreaker
    health_check: Optional[Callable] = None
    fallback_handler: Optional[Callable] = None
    dependencies: List[str] = field(default_factory=list)


@dataclass
class ResilienceMetrics:
    """Metrics for resilience monitoring"""
    total_failures: int = 0
    cascading_failures_prevented: int = 0
    degradation_events: List[tuple] = field(default_factory=list)
    emergency_stops: int = 0
    current_degradation_level: DegradationLevel = DegradationLevel.NORMAL
    service_availability: Dict[str, float] = field(default_factory=dict)
    last_health_check: Optional[datetime] = None


class ResilienceCoordinator:
    """
    Coordinates resilience across multiple services
    
    Features:
    - Manages multiple circuit breakers
    - Coordinates degraded modes
    - Prevents cascading failures
    - Emergency stop capabilities
    - Health monitoring
    """
    
    def __init__(self, agent_context=None):
        """
        Initialize resilience coordinator
        
        Args:
            agent_context: Optional Agentuity context for KV storage
        """
        self.agent_context = agent_context
        self.services: Dict[str, ServiceConfig] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.metrics = ResilienceMetrics()
        self.state_manager = get_state_manager()
        
        # Degradation configuration
        self.degradation_level = DegradationLevel.NORMAL
        self.degradation_thresholds = {
            DegradationLevel.LEVEL_1: 2,  # 2 service failures
            DegradationLevel.LEVEL_2: 4,  # 4 service failures
            DegradationLevel.LEVEL_3: 6,  # 6 service failures
            DegradationLevel.EMERGENCY: 8  # 8 service failures
        }
        
        # Emergency stop callback
        self.emergency_stop_handler: Optional[Callable] = None
        
        # Monitoring
        self.is_monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
        
        logger.info("ResilienceCoordinator initialized")
    
    def register_service(
        self,
        name: str,
        priority: ServicePriority,
        failure_threshold: int = 5,
        timeout: float = 30.0,
        health_check: Optional[Callable] = None,
        fallback_handler: Optional[Callable] = None,
        dependencies: List[str] = None
    ) -> CircuitBreaker:
        """
        Register a service with the coordinator
        
        Args:
            name: Service name
            priority: Service priority level
            failure_threshold: Failures to trigger circuit open
            timeout: Operation timeout
            health_check: Optional health check function
            fallback_handler: Optional fallback handler
            dependencies: List of dependent service names
            
        Returns:
            Circuit breaker for the service
        """
        # Create circuit breaker
        config = CircuitBreakerConfig(
            name=name,
            failure_threshold=failure_threshold,
            timeout=timeout,
            emit_metrics=True
        )
        
        circuit_breaker = CircuitBreaker(config)
        
        # Set callbacks
        circuit_breaker.on_state_change = lambda old, new: asyncio.create_task(
            self._handle_state_change(name, old, new)
        )
        circuit_breaker.on_metrics_emit = lambda metrics: asyncio.create_task(
            self._handle_metrics(name, metrics)
        )
        
        # Store configuration
        service_config = ServiceConfig(
            name=name,
            priority=priority,
            circuit_breaker=circuit_breaker,
            health_check=health_check,
            fallback_handler=fallback_handler,
            dependencies=dependencies or []
        )
        
        self.services[name] = service_config
        self.circuit_breakers[name] = circuit_breaker
        
        logger.info(f"Registered service '{name}' with priority {priority.name}")
        
        return circuit_breaker
    
    async def call_with_resilience(
        self,
        service_name: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Call a function with resilience protection
        
        Args:
            service_name: Name of the service
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result or fallback result
        """
        if service_name not in self.services:
            raise ValueError(f"Unknown service: {service_name}")
        
        service = self.services[service_name]
        
        # Check degradation level
        if self._is_service_disabled(service.priority):
            logger.warning(f"Service '{service_name}' disabled due to degradation level")
            if service.fallback_handler:
                return await service.fallback_handler(*args, **kwargs)
            raise ServiceDegradedException(f"Service '{service_name}' is degraded")
        
        # Check dependencies
        for dep in service.dependencies:
            if dep in self.circuit_breakers:
                dep_breaker = self.circuit_breakers[dep]
                if dep_breaker.state == CircuitState.OPEN:
                    logger.warning(f"Dependency '{dep}' is down for service '{service_name}'")
                    if service.fallback_handler:
                        return await service.fallback_handler(*args, **kwargs)
                    raise DependencyFailureException(f"Dependency '{dep}' is down")
        
        try:
            # Call with circuit breaker protection
            result = await service.circuit_breaker.call(func, *args, **kwargs)
            return result
            
        except Exception as e:
            # Try fallback if available
            if service.fallback_handler:
                logger.info(f"Using fallback for service '{service_name}'")
                return await service.fallback_handler(*args, **kwargs)
            raise
    
    async def _handle_state_change(
        self,
        service_name: str,
        old_state: CircuitState,
        new_state: CircuitState
    ):
        """Handle circuit breaker state change"""
        logger.info(
            f"Service '{service_name}' state changed: "
            f"{old_state.value} -> {new_state.value}"
        )
        
        # Count open circuits
        open_count = sum(
            1 for cb in self.circuit_breakers.values()
            if cb.state == CircuitState.OPEN
        )
        
        # Check for degradation
        await self._evaluate_degradation(open_count)
        
        # Store state in KV if available
        if self.agent_context:
            await self._save_state()
        
        # Check for cascading failure
        if new_state == CircuitState.OPEN:
            await self._check_cascading_failure(service_name)
    
    async def _evaluate_degradation(self, failure_count: int):
        """
        Evaluate and set degradation level
        
        Args:
            failure_count: Number of failed services
        """
        old_level = self.degradation_level
        
        # Determine new level
        if failure_count >= self.degradation_thresholds[DegradationLevel.EMERGENCY]:
            new_level = DegradationLevel.EMERGENCY
        elif failure_count >= self.degradation_thresholds[DegradationLevel.LEVEL_3]:
            new_level = DegradationLevel.LEVEL_3
        elif failure_count >= self.degradation_thresholds[DegradationLevel.LEVEL_2]:
            new_level = DegradationLevel.LEVEL_2
        elif failure_count >= self.degradation_thresholds[DegradationLevel.LEVEL_1]:
            new_level = DegradationLevel.LEVEL_1
        else:
            new_level = DegradationLevel.NORMAL
        
        if new_level != old_level:
            self.degradation_level = new_level
            self.metrics.degradation_events.append(
                (old_level, new_level, datetime.now())
            )
            
            logger.warning(
                f"System degradation level changed: "
                f"{old_level.name} -> {new_level.name}"
            )
            
            # Handle emergency stop
            if new_level == DegradationLevel.EMERGENCY:
                await self._trigger_emergency_stop()
    
    async def _check_cascading_failure(self, failed_service: str):
        """
        Check for potential cascading failures
        
        Args:
            failed_service: Name of the failed service
        """
        # Find services that depend on the failed service
        dependent_services = [
            name for name, config in self.services.items()
            if failed_service in config.dependencies
        ]
        
        if dependent_services:
            logger.warning(
                f"Potential cascading failure: {failed_service} -> "
                f"{', '.join(dependent_services)}"
            )
            
            # Preemptively open dependent service circuit breakers
            for service_name in dependent_services:
                if service_name in self.circuit_breakers:
                    breaker = self.circuit_breakers[service_name]
                    if breaker.state == CircuitState.CLOSED:
                        logger.info(f"Preemptively opening circuit for '{service_name}'")
                        await breaker._transition_to(CircuitState.OPEN)
                        self.metrics.cascading_failures_prevented += 1
    
    async def _trigger_emergency_stop(self):
        """Trigger emergency stop"""
        logger.critical("EMERGENCY STOP TRIGGERED")
        self.metrics.emergency_stops += 1
        
        if self.emergency_stop_handler:
            try:
                await self.emergency_stop_handler()
            except Exception as e:
                logger.error(f"Emergency stop handler failed: {e}")
        
        # Open all circuit breakers
        for breaker in self.circuit_breakers.values():
            if breaker.state != CircuitState.OPEN:
                await breaker._transition_to(CircuitState.OPEN)
    
    def _is_service_disabled(self, priority: ServicePriority) -> bool:
        """
        Check if service is disabled based on degradation level
        
        Args:
            priority: Service priority
            
        Returns:
            True if service should be disabled
        """
        if self.degradation_level == DegradationLevel.NORMAL:
            return False
        elif self.degradation_level == DegradationLevel.LEVEL_1:
            return priority == ServicePriority.LOW
        elif self.degradation_level == DegradationLevel.LEVEL_2:
            return priority in [ServicePriority.LOW, ServicePriority.MEDIUM]
        elif self.degradation_level == DegradationLevel.LEVEL_3:
            return priority != ServicePriority.CRITICAL
        else:  # EMERGENCY
            return True
    
    async def health_check_all(self) -> Dict[str, bool]:
        """
        Perform health checks on all services
        
        Returns:
            Dictionary of service health states
        """
        results = {}
        
        for name, config in self.services.items():
            if config.health_check:
                try:
                    health = await config.health_check()
                except:
                    health = False
            else:
                health = await config.circuit_breaker.health_check()
            
            results[name] = health
            
            # Update availability metric
            if name not in self.metrics.service_availability:
                self.metrics.service_availability[name] = 1.0 if health else 0.0
            else:
                # Exponential moving average
                self.metrics.service_availability[name] = (
                    0.9 * self.metrics.service_availability[name] +
                    0.1 * (1.0 if health else 0.0)
                )
        
        self.metrics.last_health_check = datetime.now()
        
        return results
    
    async def start_monitoring(self, interval: float = 30.0):
        """
        Start health monitoring
        
        Args:
            interval: Check interval in seconds
        """
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_task = asyncio.create_task(
            self._monitor_loop(interval)
        )
        logger.info(f"Started health monitoring with {interval}s interval")
    
    async def _monitor_loop(self, interval: float):
        """Health monitoring loop"""
        while self.is_monitoring:
            try:
                await self.health_check_all()
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(interval)
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        self.is_monitoring = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            self.monitor_task = None
        
        logger.info("Stopped health monitoring")
    
    async def _handle_metrics(self, service_name: str, metrics: Dict[str, Any]):
        """Handle metrics from circuit breaker"""
        # Update total failure count
        if "metrics" in metrics and "failed_calls" in metrics["metrics"]:
            self.metrics.total_failures = sum(
                cb.metrics.failed_calls for cb in self.circuit_breakers.values()
            )
    
    async def _save_state(self):
        """Save coordinator state to KV storage"""
        if not self.agent_context:
            return
        
        state = {
            "degradation_level": self.degradation_level.name,
            "metrics": {
                "total_failures": self.metrics.total_failures,
                "cascading_failures_prevented": self.metrics.cascading_failures_prevented,
                "emergency_stops": self.metrics.emergency_stops,
                "service_availability": self.metrics.service_availability
            },
            "circuit_breakers": {
                name: {
                    "state": cb.state.value,
                    "metrics": cb.get_metrics()
                }
                for name, cb in self.circuit_breakers.items()
            },
            "timestamp": datetime.now().isoformat()
        }
        
        await self.agent_context.kv.set(
            "circuit_breaker",
            "coordinator_state",
            state
        )
    
    async def restore_state(self):
        """Restore coordinator state from KV storage"""
        if not self.agent_context:
            return
        
        result = await self.agent_context.kv.get(
            "circuit_breaker",
            "coordinator_state"
        )
        
        if result.exists:
            state = await result.data.json()
            
            # Restore degradation level
            self.degradation_level = DegradationLevel[state["degradation_level"]]
            
            # Restore metrics
            if "metrics" in state:
                self.metrics.total_failures = state["metrics"]["total_failures"]
                self.metrics.cascading_failures_prevented = state["metrics"]["cascading_failures_prevented"]
                self.metrics.emergency_stops = state["metrics"]["emergency_stops"]
                self.metrics.service_availability = state["metrics"]["service_availability"]
            
            logger.info("Restored coordinator state from KV storage")
    
    def get_status(self) -> Dict[str, Any]:
        """Get coordinator status"""
        return {
            "degradation_level": self.degradation_level.name,
            "services": {
                name: {
                    "priority": config.priority.name,
                    "state": config.circuit_breaker.state.value,
                    "availability": self.metrics.service_availability.get(name, 0.0)
                }
                for name, config in self.services.items()
            },
            "metrics": {
                "total_failures": self.metrics.total_failures,
                "cascading_failures_prevented": self.metrics.cascading_failures_prevented,
                "emergency_stops": self.metrics.emergency_stops,
                "degradation_events": len(self.metrics.degradation_events)
            },
            "is_monitoring": self.is_monitoring,
            "last_health_check": (
                self.metrics.last_health_check.isoformat()
                if self.metrics.last_health_check else None
            )
        }


class ServiceDegradedException(Exception):
    """Exception raised when service is degraded"""
    pass


class DependencyFailureException(Exception):
    """Exception raised when dependency fails"""
    pass


# Global coordinator instance
_coordinator: Optional[ResilienceCoordinator] = None


def get_resilience_coordinator(agent_context=None) -> ResilienceCoordinator:
    """Get or create global resilience coordinator"""
    global _coordinator
    if _coordinator is None:
        _coordinator = ResilienceCoordinator(agent_context)
    return _coordinator