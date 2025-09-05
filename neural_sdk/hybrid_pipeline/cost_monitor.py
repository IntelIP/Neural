"""
API Cost Monitor and Budget Management

Tracks API usage costs, enforces budget limits, and provides
cost optimization recommendations for hybrid data pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
from datetime import datetime, timedelta, date
from enum import Enum
import logging
from collections import defaultdict, deque


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class CostAlert:
    """Cost alert definition"""
    alert_id: str
    level: AlertLevel
    threshold: float
    description: str
    callback: Optional[Callable] = None
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0


@dataclass
class APIEndpoint:
    """API endpoint cost configuration"""
    name: str
    base_cost: float
    cost_per_request: float = 0.0
    cost_per_data_unit: float = 0.0  # per KB, record, etc.
    rate_limit_requests: int = 1000
    rate_limit_window: timedelta = timedelta(hours=1)
    current_usage: int = 0
    window_start: datetime = field(default_factory=datetime.now)


@dataclass
class BudgetPeriod:
    """Budget tracking for a specific time period"""
    period_name: str  # "daily", "weekly", "monthly"
    budget_limit: float
    spent_amount: float = 0.0
    period_start: datetime = field(default_factory=datetime.now)
    period_end: datetime = field(default_factory=lambda: datetime.now() + timedelta(days=1))
    
    @property
    def utilization_percent(self) -> float:
        """Get budget utilization percentage"""
        return (self.spent_amount / self.budget_limit) * 100 if self.budget_limit > 0 else 0
    
    @property
    def remaining_budget(self) -> float:
        """Get remaining budget amount"""
        return max(0, self.budget_limit - self.spent_amount)


class APITracker:
    """
    Tracks API usage, costs, and performance metrics for cost optimization.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # API endpoints configuration
        self.endpoints: Dict[str, APIEndpoint] = {}
        
        # Usage tracking
        self.request_history: deque = deque(maxlen=10000)
        self.cost_history: Dict[date, float] = defaultdict(float)
        
        # Performance metrics
        self.latency_history: deque = deque(maxlen=1000)
        self.error_history: deque = deque(maxlen=1000)
        
        # Cost breakdown
        self.cost_by_endpoint: Dict[str, float] = defaultdict(float)
        self.cost_by_hour: Dict[datetime, float] = defaultdict(float)
        
    def register_endpoint(self, endpoint: APIEndpoint) -> None:
        """Register an API endpoint for tracking"""
        self.endpoints[endpoint.name] = endpoint
        self.logger.info(f"Registered API endpoint: {endpoint.name}")
    
    async def track_request(
        self,
        endpoint_name: str,
        data_size: int = 0,
        latency_ms: float = 0,
        success: bool = True,
        cost_override: Optional[float] = None
    ) -> float:
        """
        Track an API request and return the cost incurred.
        
        Args:
            endpoint_name: Name of the API endpoint
            data_size: Size of data transferred (for cost calculation)
            latency_ms: Request latency in milliseconds
            success: Whether the request was successful
            cost_override: Override calculated cost with specific amount
            
        Returns:
            Cost incurred for this request
        """
        try:
            timestamp = datetime.now()
            
            # Get endpoint configuration
            endpoint = self.endpoints.get(endpoint_name)
            if not endpoint:
                self.logger.warning(f"Unknown endpoint: {endpoint_name}")
                return 0.0
            
            # Check rate limits
            if not self._check_rate_limit(endpoint, timestamp):
                self.logger.warning(f"Rate limit exceeded for {endpoint_name}")
                return 0.0
            
            # Calculate cost
            if cost_override is not None:
                cost = cost_override
            else:
                cost = endpoint.cost_per_request + (endpoint.cost_per_data_unit * data_size / 1024)  # Assume data_size in bytes
            
            # Record request
            request_record = {
                "timestamp": timestamp,
                "endpoint": endpoint_name,
                "data_size": data_size,
                "latency_ms": latency_ms,
                "success": success,
                "cost": cost
            }
            
            self.request_history.append(request_record)
            
            # Update metrics
            self._update_cost_metrics(endpoint_name, cost, timestamp)
            self._update_performance_metrics(latency_ms, success, timestamp)
            
            # Update endpoint usage
            endpoint.current_usage += 1
            
            return cost
            
        except Exception as e:
            self.logger.error(f"Failed to track request: {e}")
            return 0.0
    
    def _check_rate_limit(self, endpoint: APIEndpoint, timestamp: datetime) -> bool:
        """Check if request is within rate limits"""
        try:
            # Reset window if needed
            if timestamp - endpoint.window_start >= endpoint.rate_limit_window:
                endpoint.current_usage = 0
                endpoint.window_start = timestamp
            
            # Check limit
            return endpoint.current_usage < endpoint.rate_limit_requests
            
        except Exception as e:
            self.logger.error(f"Rate limit check failed: {e}")
            return True  # Allow request on error
    
    def _update_cost_metrics(self, endpoint_name: str, cost: float, timestamp: datetime) -> None:
        """Update cost tracking metrics"""
        # Daily cost tracking
        today = timestamp.date()
        self.cost_history[today] += cost
        
        # Endpoint cost tracking
        self.cost_by_endpoint[endpoint_name] += cost
        
        # Hourly cost tracking
        hour_key = timestamp.replace(minute=0, second=0, microsecond=0)
        self.cost_by_hour[hour_key] += cost
    
    def _update_performance_metrics(self, latency_ms: float, success: bool, timestamp: datetime) -> None:
        """Update performance tracking metrics"""
        if latency_ms > 0:
            self.latency_history.append({"timestamp": timestamp, "latency": latency_ms})
        
        self.error_history.append({"timestamp": timestamp, "success": success})
    
    def get_daily_cost(self, target_date: Optional[date] = None) -> float:
        """Get total cost for a specific day"""
        target_date = target_date or datetime.now().date()
        return self.cost_history.get(target_date, 0.0)
    
    def get_hourly_costs(self, hours: int = 24) -> Dict[str, float]:
        """Get hourly cost breakdown for the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return {
            hour.strftime("%Y-%m-%d %H:00"): cost
            for hour, cost in self.cost_by_hour.items()
            if hour >= cutoff_time
        }
    
    def get_endpoint_costs(self) -> Dict[str, Dict[str, Any]]:
        """Get cost breakdown by endpoint"""
        total_cost = sum(self.cost_by_endpoint.values())
        
        return {
            endpoint: {
                "total_cost": cost,
                "percentage": (cost / total_cost) * 100 if total_cost > 0 else 0,
                "requests": sum(1 for r in self.request_history if r["endpoint"] == endpoint),
                "avg_cost_per_request": cost / max(1, sum(1 for r in self.request_history if r["endpoint"] == endpoint))
            }
            for endpoint, cost in self.cost_by_endpoint.items()
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics summary"""
        recent_latencies = [r["latency"] for r in self.latency_history if r["latency"] > 0]
        recent_errors = [not r["success"] for r in self.error_history]
        
        return {
            "avg_latency_ms": sum(recent_latencies) / len(recent_latencies) if recent_latencies else 0,
            "p95_latency_ms": sorted(recent_latencies)[int(len(recent_latencies) * 0.95)] if recent_latencies else 0,
            "error_rate": sum(recent_errors) / len(recent_errors) if recent_errors else 0,
            "total_requests": len(self.request_history),
            "requests_last_hour": sum(1 for r in self.request_history 
                                    if datetime.now() - r["timestamp"] <= timedelta(hours=1))
        }
    
    def generate_cost_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate comprehensive cost report"""
        cutoff_date = datetime.now().date() - timedelta(days=days)
        
        # Daily costs
        daily_costs = {
            str(date_key): cost
            for date_key, cost in self.cost_history.items()
            if date_key >= cutoff_date
        }
        
        # Total cost
        total_cost = sum(daily_costs.values())
        
        return {
            "report_period": f"{days} days",
            "total_cost": total_cost,
            "daily_average": total_cost / days if days > 0 else 0,
            "daily_breakdown": daily_costs,
            "endpoint_breakdown": self.get_endpoint_costs(),
            "hourly_breakdown": self.get_hourly_costs(24 * days),
            "performance_metrics": self.get_performance_metrics(),
            "cost_trend": self._calculate_cost_trend(daily_costs)
        }
    
    def _calculate_cost_trend(self, daily_costs: Dict[str, float]) -> str:
        """Calculate cost trend over time"""
        if len(daily_costs) < 2:
            return "insufficient_data"
        
        costs = list(daily_costs.values())
        first_half_avg = sum(costs[:len(costs)//2]) / (len(costs)//2)
        second_half_avg = sum(costs[len(costs)//2:]) / (len(costs) - len(costs)//2)
        
        if abs(second_half_avg - first_half_avg) < first_half_avg * 0.1:
            return "stable"
        elif second_half_avg > first_half_avg:
            return "increasing"
        else:
            return "decreasing"


class BudgetManager:
    """
    Manages budget limits and enforces cost controls across multiple time periods.
    """
    
    def __init__(self, api_tracker: APITracker):
        self.api_tracker = api_tracker
        self.logger = logging.getLogger(__name__)
        
        # Budget periods
        self.budget_periods: Dict[str, BudgetPeriod] = {}
        
        # Alerts system
        self.alerts: Dict[str, CostAlert] = {}
        self.alert_callbacks: Dict[str, Callable] = {}
        
        # Emergency controls
        self.emergency_stop_enabled = False
        self.emergency_threshold = 500.0  # $500 emergency cutoff
        
        # Cost predictions
        self.prediction_window_hours = 24
        
    def set_budget(self, period_name: str, budget_limit: float, period_duration: timedelta) -> None:
        """Set budget for a specific period"""
        period_end = datetime.now() + period_duration
        
        budget_period = BudgetPeriod(
            period_name=period_name,
            budget_limit=budget_limit,
            period_start=datetime.now(),
            period_end=period_end
        )
        
        self.budget_periods[period_name] = budget_period
        self.logger.info(f"Set budget for {period_name}: ${budget_limit} until {period_end}")
    
    def add_alert(self, alert: CostAlert) -> None:
        """Add a cost alert"""
        self.alerts[alert.alert_id] = alert
        self.logger.info(f"Added cost alert: {alert.alert_id} at {alert.threshold} threshold")
    
    async def check_budgets_and_alerts(self) -> List[Dict[str, Any]]:
        """Check all budgets and trigger alerts if needed"""
        triggered_alerts = []
        
        try:
            # Update budget periods with current costs
            await self._update_budget_periods()
            
            # Check budget violations
            for period_name, budget_period in self.budget_periods.items():
                utilization = budget_period.utilization_percent
                
                # Check alerts for this budget period
                for alert_id, alert in self.alerts.items():
                    if not alert.enabled:
                        continue
                    
                    # Check if alert threshold is met
                    should_trigger = False
                    
                    if alert.level == AlertLevel.INFO and utilization >= 50:
                        should_trigger = True
                    elif alert.level == AlertLevel.WARNING and utilization >= 75:
                        should_trigger = True
                    elif alert.level == AlertLevel.CRITICAL and utilization >= 90:
                        should_trigger = True
                    elif alert.level == AlertLevel.EMERGENCY and utilization >= 95:
                        should_trigger = True
                    
                    # Or check absolute cost threshold
                    if budget_period.spent_amount >= alert.threshold:
                        should_trigger = True
                    
                    if should_trigger:
                        triggered_alert = await self._trigger_alert(alert, budget_period)
                        triggered_alerts.append(triggered_alert)
            
            # Check emergency stop
            total_daily_cost = self.api_tracker.get_daily_cost()
            if total_daily_cost >= self.emergency_threshold:
                await self._trigger_emergency_stop()
            
            return triggered_alerts
            
        except Exception as e:
            self.logger.error(f"Failed to check budgets and alerts: {e}")
            return []
    
    async def _update_budget_periods(self) -> None:
        """Update budget periods with current spending"""
        current_time = datetime.now()
        
        for period_name, budget_period in self.budget_periods.items():
            # Check if period has expired and needs renewal
            if current_time >= budget_period.period_end:
                await self._renew_budget_period(period_name, budget_period)
                continue
            
            # Calculate spent amount for this period
            period_costs = []
            for record in self.api_tracker.request_history:
                if (budget_period.period_start <= record["timestamp"] <= current_time and
                    record["timestamp"] <= budget_period.period_end):
                    period_costs.append(record["cost"])
            
            budget_period.spent_amount = sum(period_costs)
    
    async def _renew_budget_period(self, period_name: str, old_period: BudgetPeriod) -> None:
        """Renew an expired budget period"""
        try:
            # Calculate duration of old period
            duration = old_period.period_end - old_period.period_start
            
            # Create new period with same budget limit
            new_period = BudgetPeriod(
                period_name=period_name,
                budget_limit=old_period.budget_limit,
                period_start=datetime.now(),
                period_end=datetime.now() + duration
            )
            
            self.budget_periods[period_name] = new_period
            
            self.logger.info(f"Renewed budget period {period_name}: ${new_period.budget_limit}")
            
        except Exception as e:
            self.logger.error(f"Failed to renew budget period {period_name}: {e}")
    
    async def _trigger_alert(self, alert: CostAlert, budget_period: BudgetPeriod) -> Dict[str, Any]:
        """Trigger a cost alert"""
        try:
            current_time = datetime.now()
            
            alert_data = {
                "alert_id": alert.alert_id,
                "level": alert.level.value,
                "timestamp": current_time.isoformat(),
                "description": alert.description,
                "budget_period": budget_period.period_name,
                "spent_amount": budget_period.spent_amount,
                "budget_limit": budget_period.budget_limit,
                "utilization_percent": budget_period.utilization_percent,
                "threshold_exceeded": budget_period.spent_amount >= alert.threshold
            }
            
            # Update alert record
            alert.last_triggered = current_time
            alert.trigger_count += 1
            
            # Execute callback if available
            if alert.callback:
                try:
                    await alert.callback(alert_data)
                except Exception as e:
                    self.logger.error(f"Alert callback failed for {alert.alert_id}: {e}")
            
            self.logger.warning(f"Cost alert triggered: {alert.alert_id} - {alert.description}")
            
            return alert_data
            
        except Exception as e:
            self.logger.error(f"Failed to trigger alert {alert.alert_id}: {e}")
            return {"error": str(e)}
    
    async def _trigger_emergency_stop(self) -> None:
        """Trigger emergency stop to halt all API usage"""
        try:
            self.emergency_stop_enabled = True
            
            emergency_data = {
                "timestamp": datetime.now().isoformat(),
                "reason": "emergency_cost_threshold_exceeded",
                "daily_cost": self.api_tracker.get_daily_cost(),
                "emergency_threshold": self.emergency_threshold
            }
            
            self.logger.critical(f"EMERGENCY STOP TRIGGERED: Daily cost ${emergency_data['daily_cost']} exceeded threshold ${self.emergency_threshold}")
            
            # Execute emergency callbacks
            for callback in self.alert_callbacks.get("emergency", []):
                try:
                    await callback(emergency_data)
                except Exception as e:
                    self.logger.error(f"Emergency callback failed: {e}")
                    
        except Exception as e:
            self.logger.error(f"Failed to trigger emergency stop: {e}")
    
    def is_request_allowed(self, estimated_cost: float) -> Tuple[bool, str]:
        """Check if a request is allowed based on budget constraints"""
        try:
            # Emergency stop check
            if self.emergency_stop_enabled:
                return False, "Emergency stop is active"
            
            # Check daily emergency threshold
            current_daily_cost = self.api_tracker.get_daily_cost()
            if current_daily_cost + estimated_cost >= self.emergency_threshold:
                return False, f"Would exceed emergency threshold (${self.emergency_threshold})"
            
            # Check budget periods
            for period_name, budget_period in self.budget_periods.items():
                if budget_period.spent_amount + estimated_cost > budget_period.budget_limit:
                    return False, f"Would exceed {period_name} budget (${budget_period.budget_limit})"
            
            return True, "Request allowed"
            
        except Exception as e:
            self.logger.error(f"Error checking request allowance: {e}")
            return False, "Error in budget check"
    
    def predict_costs(self, hours: int = None) -> Dict[str, Any]:
        """Predict future costs based on current usage patterns"""
        try:
            hours = hours or self.prediction_window_hours
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # Get recent requests for trend analysis
            recent_requests = [
                r for r in self.api_tracker.request_history
                if r["timestamp"] >= cutoff_time
            ]
            
            if not recent_requests:
                return {"error": "No recent data for prediction"}
            
            # Calculate hourly rate
            recent_cost = sum(r["cost"] for r in recent_requests)
            hourly_rate = recent_cost / hours
            
            # Predict next period costs
            predictions = {
                "next_hour": hourly_rate,
                "next_6_hours": hourly_rate * 6,
                "next_24_hours": hourly_rate * 24,
                "end_of_day": hourly_rate * (24 - datetime.now().hour),
                "end_of_week": hourly_rate * ((7 - datetime.now().weekday()) * 24 + (24 - datetime.now().hour))
            }
            
            # Check budget implications
            budget_warnings = []
            for period_name, budget_period in self.budget_periods.items():
                remaining_hours = (budget_period.period_end - datetime.now()).total_seconds() / 3600
                if remaining_hours > 0:
                    predicted_spend = hourly_rate * remaining_hours
                    if budget_period.spent_amount + predicted_spend > budget_period.budget_limit:
                        budget_warnings.append(f"{period_name} budget may be exceeded")
            
            return {
                "hourly_rate": hourly_rate,
                "predictions": predictions,
                "budget_warnings": budget_warnings,
                "confidence": "medium" if len(recent_requests) >= 10 else "low"
            }
            
        except Exception as e:
            self.logger.error(f"Cost prediction failed: {e}")
            return {"error": str(e)}
    
    def get_budget_status(self) -> Dict[str, Any]:
        """Get comprehensive budget status"""
        try:
            status = {
                "timestamp": datetime.now().isoformat(),
                "emergency_stop_active": self.emergency_stop_enabled,
                "daily_cost": self.api_tracker.get_daily_cost(),
                "emergency_threshold": self.emergency_threshold,
                "budget_periods": {},
                "active_alerts": len([a for a in self.alerts.values() if a.enabled]),
                "recent_alerts": [
                    {
                        "alert_id": alert.alert_id,
                        "last_triggered": alert.last_triggered.isoformat() if alert.last_triggered else None,
                        "trigger_count": alert.trigger_count
                    }
                    for alert in self.alerts.values()
                    if alert.last_triggered and 
                    datetime.now() - alert.last_triggered <= timedelta(hours=24)
                ]
            }
            
            # Add budget period details
            for period_name, budget_period in self.budget_periods.items():
                status["budget_periods"][period_name] = {
                    "budget_limit": budget_period.budget_limit,
                    "spent_amount": budget_period.spent_amount,
                    "remaining": budget_period.remaining_budget,
                    "utilization_percent": budget_period.utilization_percent,
                    "period_start": budget_period.period_start.isoformat(),
                    "period_end": budget_period.period_end.isoformat(),
                    "time_remaining": str(budget_period.period_end - datetime.now())
                }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Failed to get budget status: {e}")
            return {"error": str(e)}
    
    def reset_emergency_stop(self) -> None:
        """Reset emergency stop (manual override)"""
        self.emergency_stop_enabled = False
        self.logger.info("Emergency stop reset manually")
    
    def disable_alert(self, alert_id: str) -> None:
        """Disable a specific alert"""
        if alert_id in self.alerts:
            self.alerts[alert_id].enabled = False
            self.logger.info(f"Disabled alert: {alert_id}")
    
    def enable_alert(self, alert_id: str) -> None:
        """Enable a specific alert"""
        if alert_id in self.alerts:
            self.alerts[alert_id].enabled = True
            self.logger.info(f"Enabled alert: {alert_id}")
    
    def register_callback(self, event_type: str, callback: Callable) -> None:
        """Register callback for specific events"""
        if event_type not in self.alert_callbacks:
            self.alert_callbacks[event_type] = []
        
        self.alert_callbacks[event_type].append(callback)
        self.logger.info(f"Registered callback for {event_type}")