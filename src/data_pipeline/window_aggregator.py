"""
Window Aggregator - Time-window based data aggregation
Implements sliding and tumbling windows for market data analysis
Based on Apache Flink windowing model
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)


class WindowType(Enum):
    """Window types"""
    TUMBLING = "tumbling"  # Non-overlapping fixed windows
    SLIDING = "sliding"     # Overlapping windows with slide interval
    SESSION = "session"     # Dynamic windows based on activity gaps


@dataclass
class WindowedData:
    """Data point with timestamp"""
    timestamp: datetime
    value: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        return self.timestamp < other.timestamp


@dataclass
class WindowResult:
    """Aggregation result for a window"""
    window_start: datetime
    window_end: datetime
    count: int
    values: List[Any]
    aggregates: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class Window:
    """Base window class"""
    
    def __init__(self, duration_seconds: int):
        """
        Initialize window
        
        Args:
            duration_seconds: Window duration in seconds
        """
        self.duration = timedelta(seconds=duration_seconds)
        self.data: deque = deque()
        self.start_time: Optional[datetime] = None
        
    def add(self, data_point: WindowedData):
        """Add data point to window"""
        self.data.append(data_point)
        if self.start_time is None:
            self.start_time = data_point.timestamp
    
    def clear(self):
        """Clear window data"""
        self.data.clear()
        self.start_time = None
    
    def is_complete(self, current_time: datetime) -> bool:
        """Check if window is complete"""
        if self.start_time is None:
            return False
        return current_time >= self.start_time + self.duration
    
    def get_data(self) -> List[WindowedData]:
        """Get window data"""
        return list(self.data)


class TumblingWindow(Window):
    """
    Non-overlapping fixed time windows
    Example: 5-minute windows [0-5min], [5-10min], [10-15min]
    """
    
    def __init__(self, duration_seconds: int):
        """
        Initialize tumbling window
        
        Args:
            duration_seconds: Window duration in seconds
        """
        super().__init__(duration_seconds)
        self.completed_windows: List[WindowResult] = []
    
    async def add_event(self, value: Any, metadata: Dict[str, Any] = None) -> Optional[WindowResult]:
        """
        Add event to window
        
        Args:
            value: Event value
            metadata: Optional metadata
            
        Returns:
            WindowResult if window completed, None otherwise
        """
        now = datetime.now()
        data_point = WindowedData(now, value, metadata or {})
        
        # Check if we need to close current window
        if self.is_complete(now):
            result = self._aggregate()
            self.clear()
            self.add(data_point)
            return result
        
        self.add(data_point)
        return None
    
    def _aggregate(self) -> WindowResult:
        """Aggregate window data"""
        if not self.data:
            return None
        
        values = [d.value for d in self.data]
        
        # Calculate aggregates based on value types
        aggregates = {}
        
        # Try numeric aggregations
        try:
            numeric_values = [float(v) for v in values if isinstance(v, (int, float))]
            if numeric_values:
                aggregates.update({
                    "min": min(numeric_values),
                    "max": max(numeric_values),
                    "mean": np.mean(numeric_values),
                    "median": np.median(numeric_values),
                    "std": np.std(numeric_values),
                    "sum": sum(numeric_values)
                })
        except:
            pass
        
        return WindowResult(
            window_start=self.start_time,
            window_end=self.start_time + self.duration,
            count=len(self.data),
            values=values,
            aggregates=aggregates
        )


class SlidingWindow:
    """
    Overlapping windows with configurable slide interval
    Example: 5-minute window sliding every 1 minute
    """
    
    def __init__(self, duration_seconds: int, slide_seconds: int):
        """
        Initialize sliding window
        
        Args:
            duration_seconds: Window duration in seconds
            slide_seconds: Slide interval in seconds
        """
        self.duration = timedelta(seconds=duration_seconds)
        self.slide_interval = timedelta(seconds=slide_seconds)
        self.data: deque = deque()
        self.last_emit_time: Optional[datetime] = None
        
    async def add_event(self, value: Any, metadata: Dict[str, Any] = None) -> Optional[WindowResult]:
        """
        Add event to sliding window
        
        Args:
            value: Event value
            metadata: Optional metadata
            
        Returns:
            WindowResult if slide interval reached, None otherwise
        """
        now = datetime.now()
        data_point = WindowedData(now, value, metadata or {})
        
        # Add new data
        self.data.append(data_point)
        
        # Remove old data outside window
        cutoff_time = now - self.duration
        while self.data and self.data[0].timestamp < cutoff_time:
            self.data.popleft()
        
        # Check if we should emit a window
        if self.last_emit_time is None:
            self.last_emit_time = now
            
        if now >= self.last_emit_time + self.slide_interval:
            self.last_emit_time = now
            return self._aggregate(now)
        
        return None
    
    def _aggregate(self, current_time: datetime) -> WindowResult:
        """Aggregate current window data"""
        if not self.data:
            return None
        
        values = [d.value for d in self.data]
        window_start = current_time - self.duration
        
        # Calculate aggregates
        aggregates = {}
        
        try:
            numeric_values = [float(v) for v in values if isinstance(v, (int, float))]
            if numeric_values:
                aggregates.update({
                    "min": min(numeric_values),
                    "max": max(numeric_values),
                    "mean": np.mean(numeric_values),
                    "median": np.median(numeric_values),
                    "std": np.std(numeric_values),
                    "sum": sum(numeric_values),
                    "rate": len(numeric_values) / self.duration.total_seconds()  # Events per second
                })
        except:
            pass
        
        return WindowResult(
            window_start=window_start,
            window_end=current_time,
            count=len(self.data),
            values=values,
            aggregates=aggregates
        )
    
    def get_current_aggregates(self) -> Dict[str, Any]:
        """Get current window aggregates without emitting"""
        if not self.data:
            return {}
        
        result = self._aggregate(datetime.now())
        return result.aggregates if result else {}


class WindowAggregator:
    """
    Manages multiple windows for comprehensive time-based analysis
    """
    
    def __init__(self):
        """Initialize window aggregator"""
        # Standard windows for market analysis
        self.windows: Dict[str, Any] = {
            # Sliding windows for continuous monitoring
            "1m": SlidingWindow(60, 5),       # 1 min window, 5 sec slide
            "5m": SlidingWindow(300, 30),     # 5 min window, 30 sec slide
            "15m": SlidingWindow(900, 60),    # 15 min window, 1 min slide
            
            # Tumbling windows for discrete periods
            "1m_tumbling": TumblingWindow(60),
            "5m_tumbling": TumblingWindow(300),
            "15m_tumbling": TumblingWindow(900),
            "1h_tumbling": TumblingWindow(3600)
        }
        
        # Window results storage
        self.results: Dict[str, List[WindowResult]] = {
            key: [] for key in self.windows
        }
        
        # Aggregation functions
        self.custom_aggregators: Dict[str, Callable] = {}
        
        logger.info("WindowAggregator initialized with standard windows")
    
    async def add_event(self, value: Any, metadata: Dict[str, Any] = None) -> Dict[str, Optional[WindowResult]]:
        """
        Add event to all windows
        
        Args:
            value: Event value
            metadata: Optional metadata
            
        Returns:
            Dictionary of window results (None if window not complete)
        """
        results = {}
        
        for window_key, window in self.windows.items():
            result = await window.add_event(value, metadata)
            
            if result:
                self.results[window_key].append(result)
                # Keep only last 100 results per window
                self.results[window_key] = self.results[window_key][-100:]
                
            results[window_key] = result
        
        return results
    
    async def add_to_window(self, window_key: str, value: Any, metadata: Dict[str, Any] = None) -> Optional[WindowResult]:
        """
        Add event to specific window
        
        Args:
            window_key: Window identifier
            value: Event value
            metadata: Optional metadata
            
        Returns:
            WindowResult if window complete/emitted, None otherwise
        """
        if window_key not in self.windows:
            logger.error(f"Unknown window: {window_key}")
            return None
        
        result = await self.windows[window_key].add_event(value, metadata)
        
        if result:
            self.results[window_key].append(result)
            self.results[window_key] = self.results[window_key][-100:]
        
        return result
    
    def get_aggregates(self, window_key: str) -> Dict[str, Any]:
        """
        Get current aggregates for a window
        
        Args:
            window_key: Window identifier
            
        Returns:
            Current aggregates
        """
        if window_key not in self.windows:
            return {}
        
        window = self.windows[window_key]
        
        # Get current aggregates for sliding windows
        if isinstance(window, SlidingWindow):
            return window.get_current_aggregates()
        
        # Get last result for tumbling windows
        if window_key in self.results and self.results[window_key]:
            last_result = self.results[window_key][-1]
            return last_result.aggregates
        
        return {}
    
    def get_window_history(self, window_key: str, limit: int = 10) -> List[WindowResult]:
        """
        Get historical window results
        
        Args:
            window_key: Window identifier
            limit: Number of results to return
            
        Returns:
            List of window results
        """
        if window_key not in self.results:
            return []
        
        return self.results[window_key][-limit:]
    
    def register_aggregator(self, name: str, func: Callable):
        """
        Register custom aggregation function
        
        Args:
            name: Aggregator name
            func: Aggregation function
        """
        self.custom_aggregators[name] = func
        logger.info(f"Registered custom aggregator: {name}")
    
    def calculate_velocity(self, window_key: str = "5m") -> Dict[str, float]:
        """
        Calculate rate of change (velocity) for windowed data
        
        Args:
            window_key: Window to use for calculation
            
        Returns:
            Velocity metrics
        """
        history = self.get_window_history(window_key, limit=5)
        
        if len(history) < 2:
            return {"velocity": 0.0, "acceleration": 0.0}
        
        # Get mean values from windows
        means = []
        for result in history:
            if "mean" in result.aggregates:
                means.append(result.aggregates["mean"])
        
        if len(means) < 2:
            return {"velocity": 0.0, "acceleration": 0.0}
        
        # Calculate velocity (rate of change)
        velocities = np.diff(means)
        velocity = np.mean(velocities) if len(velocities) > 0 else 0.0
        
        # Calculate acceleration (rate of velocity change)
        acceleration = 0.0
        if len(velocities) > 1:
            acceleration = np.diff(velocities).mean()
        
        return {
            "velocity": float(velocity),
            "acceleration": float(acceleration),
            "momentum": float(means[-1] - means[0]) if len(means) > 1 else 0.0
        }
    
    def detect_anomaly(self, window_key: str = "5m", threshold: float = 3.0) -> bool:
        """
        Detect anomalies using statistical methods
        
        Args:
            window_key: Window to analyze
            threshold: Standard deviation threshold
            
        Returns:
            True if anomaly detected
        """
        aggregates = self.get_aggregates(window_key)
        
        if "mean" not in aggregates or "std" not in aggregates:
            return False
        
        # Check if current values are outside threshold
        history = self.get_window_history(window_key, limit=20)
        if len(history) < 10:
            return False
        
        # Calculate historical statistics
        historical_means = [r.aggregates.get("mean", 0) for r in history[:-1]]
        hist_mean = np.mean(historical_means)
        hist_std = np.std(historical_means)
        
        current_mean = aggregates["mean"]
        
        # Check if current mean is anomalous
        z_score = abs(current_mean - hist_mean) / hist_std if hist_std > 0 else 0
        
        return z_score > threshold
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregator statistics"""
        return {
            "windows": list(self.windows.keys()),
            "result_counts": {
                key: len(results) for key, results in self.results.items()
            },
            "custom_aggregators": list(self.custom_aggregators.keys())
        }