"""
Backpressure Controller - System-wide backpressure coordination
Monitors queue depths, latencies, and memory to prevent overload
Automatically throttles producers when consumers lag
"""

import asyncio
import psutil
import time
import logging
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
from collections import deque

logger = logging.getLogger(__name__)


class PressureLevel(Enum):
    """System pressure levels"""
    NONE = 0      # Normal operation
    LOW = 1       # Slight pressure, monitor closely
    MEDIUM = 2    # Moderate pressure, start throttling
    HIGH = 3      # High pressure, aggressive throttling
    CRITICAL = 4  # Critical pressure, drop non-essential work


@dataclass
class PressureMetrics:
    """Metrics for pressure calculation"""
    queue_depth: int = 0
    queue_capacity: int = 1000
    processing_latency_ms: float = 0.0
    memory_usage_percent: float = 0.0
    error_rate: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def queue_utilization(self) -> float:
        """Calculate queue utilization percentage"""
        return (self.queue_depth / self.queue_capacity * 100) if self.queue_capacity > 0 else 0
    
    def calculate_pressure(self) -> PressureLevel:
        """Calculate pressure level from metrics"""
        # Weighted pressure score
        score = (
            self.queue_utilization * 0.4 +
            min(self.processing_latency_ms / 100, 100) * 0.3 +  # Normalize to 0-100
            self.memory_usage_percent * 0.2 +
            self.error_rate * 100 * 0.1  # Error rate is 0-1, scale to 0-100
        )
        
        if score < 20:
            return PressureLevel.NONE
        elif score < 40:
            return PressureLevel.LOW
        elif score < 60:
            return PressureLevel.MEDIUM
        elif score < 80:
            return PressureLevel.HIGH
        else:
            return PressureLevel.CRITICAL


@dataclass
class BackpressureConfig:
    """Configuration for backpressure controller"""
    name: str
    check_interval: float = 1.0  # Seconds between pressure checks
    
    # Thresholds for pressure levels (queue utilization %)
    low_threshold: float = 30.0
    medium_threshold: float = 50.0
    high_threshold: float = 70.0
    critical_threshold: float = 90.0
    
    # Memory thresholds
    memory_limit_percent: float = 80.0
    
    # Latency thresholds (ms)
    latency_warning_ms: float = 100.0
    latency_critical_ms: float = 500.0
    
    # Response actions
    enable_throttling: bool = True
    enable_dropping: bool = True
    enable_pausing: bool = True


class BackpressureController:
    """
    System-wide backpressure controller
    
    Features:
    - Multi-source pressure monitoring
    - Automatic throttling decisions
    - Graceful degradation
    - Recovery detection
    """
    
    def __init__(self, config: Optional[BackpressureConfig] = None):
        """
        Initialize backpressure controller
        
        Args:
            config: Backpressure configuration
        """
        self.config = config or BackpressureConfig(name="default")
        
        # Pressure tracking
        self.current_pressure = PressureLevel.NONE
        self.pressure_sources: Dict[str, PressureMetrics] = {}
        self.pressure_history = deque(maxlen=100)
        
        # Throttling state
        self.throttle_factors: Dict[str, float] = {}  # 0.0 = stopped, 1.0 = full speed
        self.paused_sources: set = set()
        
        # Callbacks
        self.on_pressure_change: Optional[Callable] = None
        self.on_throttle_change: Optional[Callable] = None
        
        # Monitoring
        self.is_monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            "pressure_changes": 0,
            "throttle_events": 0,
            "pause_events": 0,
            "drop_events": 0,
            "total_dropped": 0
        }
        
        logger.info(f"BackpressureController '{config.name if config else 'default'}' initialized")
    
    def register_source(
        self,
        name: str,
        queue_capacity: int = 1000,
        initial_throttle: float = 1.0
    ):
        """
        Register a pressure source
        
        Args:
            name: Source name
            queue_capacity: Maximum queue size
            initial_throttle: Initial throttle factor
        """
        self.pressure_sources[name] = PressureMetrics(queue_capacity=queue_capacity)
        self.throttle_factors[name] = initial_throttle
        logger.info(f"Registered pressure source: {name}")
    
    def update_metrics(
        self,
        source: str,
        queue_depth: Optional[int] = None,
        latency_ms: Optional[float] = None,
        error_rate: Optional[float] = None
    ):
        """
        Update metrics for a source
        
        Args:
            source: Source name
            queue_depth: Current queue depth
            latency_ms: Processing latency in milliseconds
            error_rate: Error rate (0-1)
        """
        if source not in self.pressure_sources:
            self.register_source(source)
        
        metrics = self.pressure_sources[source]
        
        if queue_depth is not None:
            metrics.queue_depth = queue_depth
        if latency_ms is not None:
            metrics.processing_latency_ms = latency_ms
        if error_rate is not None:
            metrics.error_rate = error_rate
        
        metrics.timestamp = datetime.now()
        
        # Check if we need to adjust pressure
        self._evaluate_pressure()
    
    def _evaluate_pressure(self):
        """Evaluate system pressure and take action"""
        if not self.pressure_sources:
            return
        
        # Calculate overall system pressure
        max_pressure = PressureLevel.NONE
        total_metrics = PressureMetrics()
        
        # Get system memory usage
        memory_percent = psutil.virtual_memory().percent
        
        for source, metrics in self.pressure_sources.items():
            # Update memory usage
            metrics.memory_usage_percent = memory_percent
            
            # Calculate pressure for this source
            source_pressure = metrics.calculate_pressure()
            if source_pressure.value > max_pressure.value:
                max_pressure = source_pressure
            
            # Aggregate metrics
            total_metrics.queue_depth += metrics.queue_depth
            total_metrics.queue_capacity += metrics.queue_capacity
            total_metrics.processing_latency_ms = max(
                total_metrics.processing_latency_ms,
                metrics.processing_latency_ms
            )
            total_metrics.error_rate = max(total_metrics.error_rate, metrics.error_rate)
        
        total_metrics.memory_usage_percent = memory_percent
        
        # Check for pressure change
        old_pressure = self.current_pressure
        self.current_pressure = max_pressure
        
        if old_pressure != self.current_pressure:
            self.stats["pressure_changes"] += 1
            self.pressure_history.append((datetime.now(), self.current_pressure))
            
            logger.info(f"Pressure changed: {old_pressure.name} -> {self.current_pressure.name}")
            
            # Take action based on new pressure
            self._apply_pressure_response()
            
            # Notify callback
            if self.on_pressure_change:
                asyncio.create_task(self.on_pressure_change(old_pressure, self.current_pressure))
    
    def _apply_pressure_response(self):
        """Apply throttling/dropping based on pressure level"""
        if not self.config.enable_throttling:
            return
        
        # Determine throttle factors based on pressure
        base_throttles = {
            PressureLevel.NONE: 1.0,
            PressureLevel.LOW: 0.9,
            PressureLevel.MEDIUM: 0.7,
            PressureLevel.HIGH: 0.5,
            PressureLevel.CRITICAL: 0.2
        }
        
        base_throttle = base_throttles[self.current_pressure]
        
        # Apply throttling to each source based on their individual pressure
        for source, metrics in self.pressure_sources.items():
            source_pressure = metrics.calculate_pressure()
            
            # Calculate source-specific throttle
            if source_pressure == PressureLevel.CRITICAL and self.config.enable_pausing:
                # Pause critical sources
                if source not in self.paused_sources:
                    self.paused_sources.add(source)
                    self.stats["pause_events"] += 1
                    logger.warning(f"Pausing source '{source}' due to critical pressure")
                self.throttle_factors[source] = 0.0
            else:
                # Resume if was paused
                if source in self.paused_sources:
                    self.paused_sources.remove(source)
                    logger.info(f"Resuming source '{source}'")
                
                # Apply graduated throttling
                old_throttle = self.throttle_factors.get(source, 1.0)
                new_throttle = min(base_throttle, base_throttles[source_pressure])
                
                if abs(old_throttle - new_throttle) > 0.05:
                    self.throttle_factors[source] = new_throttle
                    self.stats["throttle_events"] += 1
                    logger.info(f"Throttling '{source}': {old_throttle:.1%} -> {new_throttle:.1%}")
        
        # Notify throttle changes
        if self.on_throttle_change:
            asyncio.create_task(self.on_throttle_change(self.throttle_factors.copy()))
    
    def should_accept(self, source: str, priority: int = 5) -> bool:
        """
        Check if should accept new work
        
        Args:
            source: Source name
            priority: Work priority (1=highest, 10=lowest)
            
        Returns:
            True if work should be accepted
        """
        # Always accept critical priority
        if priority <= 2:
            return True
        
        # Check if source is paused
        if source in self.paused_sources:
            return False
        
        # Check pressure-based acceptance
        if self.current_pressure == PressureLevel.CRITICAL:
            # Only accept high priority in critical state
            accept = priority <= 3
        elif self.current_pressure == PressureLevel.HIGH:
            # Accept medium priority and above
            accept = priority <= 5
        elif self.current_pressure == PressureLevel.MEDIUM:
            # Accept most work
            accept = priority <= 7
        else:
            # Accept everything in normal/low pressure
            accept = True
        
        if not accept:
            self.stats["drop_events"] += 1
            self.stats["total_dropped"] += 1
        
        return accept
    
    def get_throttle_factor(self, source: str) -> float:
        """
        Get current throttle factor for source
        
        Args:
            source: Source name
            
        Returns:
            Throttle factor (0.0 = stopped, 1.0 = full speed)
        """
        return self.throttle_factors.get(source, 1.0)
    
    async def wait_if_pressured(self, source: str, base_delay: float = 0.001):
        """
        Wait proportionally to pressure level
        
        Args:
            source: Source name
            base_delay: Base delay in seconds
        """
        throttle = self.get_throttle_factor(source)
        
        if throttle < 1.0:
            # Calculate additional delay based on throttle
            delay = base_delay * (2.0 - throttle)
            await asyncio.sleep(delay)
    
    async def start_monitoring(self):
        """Start pressure monitoring loop"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Backpressure monitoring started")
    
    async def _monitor_loop(self):
        """Continuous pressure monitoring"""
        while self.is_monitoring:
            try:
                # Re-evaluate pressure periodically
                self._evaluate_pressure()
                
                # Check for recovery
                if self.current_pressure == PressureLevel.NONE:
                    # Gradually increase throttles
                    for source in self.throttle_factors:
                        current = self.throttle_factors[source]
                        if current < 1.0:
                            self.throttle_factors[source] = min(1.0, current + 0.1)
                
                await asyncio.sleep(self.config.check_interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(self.config.check_interval * 2)
    
    async def stop_monitoring(self):
        """Stop pressure monitoring"""
        self.is_monitoring = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Backpressure monitoring stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get controller status"""
        return {
            "current_pressure": self.current_pressure.name,
            "sources": {
                name: {
                    "pressure": metrics.calculate_pressure().name,
                    "queue_utilization": f"{metrics.queue_utilization:.1f}%",
                    "latency_ms": metrics.processing_latency_ms,
                    "throttle_factor": f"{self.throttle_factors.get(name, 1.0):.1%}",
                    "paused": name in self.paused_sources
                }
                for name, metrics in self.pressure_sources.items()
            },
            "stats": self.stats,
            "memory_usage": f"{psutil.virtual_memory().percent:.1f}%",
            "is_monitoring": self.is_monitoring
        }
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            "pressure_changes": 0,
            "throttle_events": 0,
            "pause_events": 0,
            "drop_events": 0,
            "total_dropped": 0
        }
        logger.info("Backpressure statistics reset")


class AdaptiveBackpressure:
    """
    Adaptive backpressure that learns optimal thresholds
    
    Features:
    - ML-inspired threshold adjustment
    - Learns from system behavior
    - Predictive pressure detection
    """
    
    def __init__(self, controller: BackpressureController):
        """
        Initialize adaptive backpressure
        
        Args:
            controller: Base backpressure controller
        """
        self.controller = controller
        
        # Learning parameters
        self.learning_rate = 0.1
        self.history_window = 100
        
        # Performance history
        self.performance_history = deque(maxlen=self.history_window)
        self.threshold_adjustments = []
        
        # Optimal thresholds (learned)
        self.optimal_thresholds = {
            "queue_utilization": 70.0,
            "latency_ms": 100.0,
            "memory_percent": 75.0
        }
        
        logger.info("AdaptiveBackpressure initialized")
    
    def record_performance(
        self,
        throughput: float,
        latency: float,
        error_rate: float,
        pressure: PressureLevel
    ):
        """
        Record performance metrics for learning
        
        Args:
            throughput: Current throughput
            latency: Current latency
            error_rate: Current error rate
            pressure: Current pressure level
        """
        self.performance_history.append({
            "timestamp": datetime.now(),
            "throughput": throughput,
            "latency": latency,
            "error_rate": error_rate,
            "pressure": pressure,
            "thresholds": self.optimal_thresholds.copy()
        })
        
        # Learn from history periodically
        if len(self.performance_history) >= 10:
            self._learn_optimal_thresholds()
    
    def _learn_optimal_thresholds(self):
        """Learn optimal thresholds from performance history"""
        if not self.performance_history:
            return
        
        # Find best performing configurations
        sorted_history = sorted(
            self.performance_history,
            key=lambda x: x["throughput"] / (x["latency"] * (1 + x["error_rate"])),
            reverse=True
        )
        
        # Take top 20% performers
        top_performers = sorted_history[:max(1, len(sorted_history) // 5)]
        
        # Average their thresholds
        if top_performers:
            avg_thresholds = {}
            for key in self.optimal_thresholds:
                values = [p["thresholds"].get(key, self.optimal_thresholds[key]) 
                         for p in top_performers if "thresholds" in p]
                if values:
                    avg_thresholds[key] = sum(values) / len(values)
            
            # Apply learning rate
            for key in self.optimal_thresholds:
                if key in avg_thresholds:
                    old_val = self.optimal_thresholds[key]
                    new_val = (
                        old_val * (1 - self.learning_rate) +
                        avg_thresholds[key] * self.learning_rate
                    )
                    self.optimal_thresholds[key] = new_val
                    
                    if abs(new_val - old_val) > old_val * 0.05:
                        logger.info(f"Adjusted threshold {key}: {old_val:.1f} -> {new_val:.1f}")
                        self.threshold_adjustments.append({
                            "timestamp": datetime.now(),
                            "threshold": key,
                            "old_value": old_val,
                            "new_value": new_val
                        })
    
    def predict_pressure(self, metrics: PressureMetrics) -> PressureLevel:
        """
        Predict pressure level using learned thresholds
        
        Args:
            metrics: Current metrics
            
        Returns:
            Predicted pressure level
        """
        score = 0
        
        # Compare against learned thresholds
        if metrics.queue_utilization > self.optimal_thresholds["queue_utilization"]:
            score += 30
        
        if metrics.processing_latency_ms > self.optimal_thresholds["latency_ms"]:
            score += 30
        
        if metrics.memory_usage_percent > self.optimal_thresholds["memory_percent"]:
            score += 20
        
        # Add error rate contribution
        score += metrics.error_rate * 100 * 20
        
        # Map score to pressure level
        if score < 20:
            return PressureLevel.NONE
        elif score < 40:
            return PressureLevel.LOW
        elif score < 60:
            return PressureLevel.MEDIUM
        elif score < 80:
            return PressureLevel.HIGH
        else:
            return PressureLevel.CRITICAL


# Global backpressure controller instance
_controller: Optional[BackpressureController] = None


def get_backpressure_controller() -> BackpressureController:
    """Get or create global backpressure controller"""
    global _controller
    if _controller is None:
        _controller = BackpressureController()
    return _controller