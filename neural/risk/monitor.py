"""
Real-Time Risk Monitoring and Alerting System

This module provides comprehensive real-time risk monitoring capabilities
including:

- Continuous portfolio risk assessment
- VaR (Value at Risk) monitoring
- Drawdown tracking and alerts
- Position concentration monitoring  
- Correlation risk assessment
- Performance attribution analysis
- Configurable alert system
- Risk dashboard metrics

The monitoring system integrates with all other risk components to provide
a unified view of portfolio risk and automated alerting.
"""

import logging
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from collections import deque
import asyncio
from concurrent.futures import ThreadPoolExecutor

from neural.risk.limits import RiskLimitManager, LimitViolation
from neural.analysis.metrics import PerformanceCalculator

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertType(Enum):
    """Types of risk alerts."""
    VAR_BREACH = "var_breach"
    DRAWDOWN_ALERT = "drawdown_alert"
    CONCENTRATION_RISK = "concentration_risk"
    CORRELATION_SPIKE = "correlation_spike"
    VOLATILITY_SPIKE = "volatility_spike"
    LIMIT_VIOLATION = "limit_violation"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    LIQUIDITY_SHORTAGE = "liquidity_shortage"
    SYSTEM_ERROR = "system_error"
    CUSTOM = "custom"


@dataclass
class RiskAlert:
    """Risk alert notification."""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    timestamp: datetime
    message: str
    current_value: float
    threshold_value: float
    affected_positions: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False


@dataclass
class RiskMetrics:
    """Current risk metrics snapshot."""
    timestamp: datetime
    total_capital: float
    total_exposure: float
    cash_available: float
    unrealized_pnl: float
    daily_pnl: float
    
    # Risk measures
    portfolio_var_95: float
    portfolio_var_99: float
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    
    # Concentration measures
    largest_position_pct: float
    top_3_concentration: float
    herfindahl_index: float
    
    # Correlation measures
    avg_correlation: float
    max_correlation: float
    correlation_risk_score: float
    
    # Performance measures
    daily_return: float
    volatility: float
    win_rate: float
    profit_factor: float
    
    # System measures  
    num_positions: int
    num_alerts: int
    system_health: float
    
    metadata: Dict[str, Any] = field(default_factory=dict)


class VaRMonitor:
    """
    Value at Risk monitoring system.
    
    Calculates and monitors portfolio VaR using multiple methods
    (historical, parametric, Monte Carlo) with configurable thresholds.
    """
    
    def __init__(
        self,
        lookback_days: int = 252,
        confidence_levels: List[float] = None,
        var_limit: float = 0.05,  # 5% VaR limit
        update_frequency: int = 300  # 5 minutes
    ):
        """
        Initialize VaR monitor.
        
        Args:
            lookback_days: Days of history for VaR calculation
            confidence_levels: VaR confidence levels to monitor
            var_limit: VaR limit threshold
            update_frequency: Update frequency in seconds
        """
        self.lookback_days = lookback_days
        self.confidence_levels = confidence_levels or [0.95, 0.99]
        self.var_limit = var_limit
        self.update_frequency = update_frequency
        
        # VaR history for tracking
        self.var_history = deque(maxlen=1000)
        self.last_update = datetime.now()
        
    def calculate_var(
        self,
        returns: pd.Series,
        method: str = "historical"
    ) -> Dict[str, float]:
        """
        Calculate Value at Risk using specified method.
        
        Args:
            returns: Portfolio returns series
            method: VaR calculation method (historical, parametric)
            
        Returns:
            Dictionary with VaR values for each confidence level
        """
        if returns.empty or len(returns) < 30:
            logger.warning("Insufficient data for VaR calculation")
            return {f"var_{int(cl*100)}": 0.0 for cl in self.confidence_levels}
        
        var_results = {}
        
        for confidence_level in self.confidence_levels:
            if method == "historical":
                var_value = self._historical_var(returns, confidence_level)
            elif method == "parametric":
                var_value = self._parametric_var(returns, confidence_level)
            else:
                var_value = self._historical_var(returns, confidence_level)
            
            var_results[f"var_{int(confidence_level*100)}"] = abs(var_value)
        
        # Store in history
        var_entry = {
            'timestamp': datetime.now(),
            'method': method,
            **var_results
        }
        self.var_history.append(var_entry)
        
        return var_results
    
    def _historical_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate VaR using historical simulation."""
        percentile = (1 - confidence_level) * 100
        return np.percentile(returns, percentile)
    
    def _parametric_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate VaR using parametric (normal distribution) method."""
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Z-score for confidence level
        from scipy import stats
        z_score = stats.norm.ppf(1 - confidence_level)
        
        return mean_return + z_score * std_return
    
    def check_var_breach(self, current_var: Dict[str, float]) -> List[RiskAlert]:
        """Check for VaR limit breaches."""
        alerts = []
        
        for var_type, var_value in current_var.items():
            if var_value > self.var_limit:
                alert = RiskAlert(
                    alert_id=f"var_breach_{var_type}_{datetime.now().isoformat()}",
                    alert_type=AlertType.VAR_BREACH,
                    severity=AlertSeverity.CRITICAL if var_value > self.var_limit * 1.5 else AlertSeverity.WARNING,
                    timestamp=datetime.now(),
                    message=f"VaR breach: {var_type} = {var_value:.1%} exceeds limit {self.var_limit:.1%}",
                    current_value=var_value,
                    threshold_value=self.var_limit,
                    recommended_actions=[
                        "Reduce position sizes",
                        "Increase diversification", 
                        "Review risk limits"
                    ]
                )
                alerts.append(alert)
        
        return alerts


class DrawdownMonitor:
    """
    Drawdown monitoring and alerting system.
    
    Tracks portfolio drawdowns and generates alerts when
    drawdown thresholds are breached or recovery patterns emerge.
    """
    
    def __init__(
        self,
        warning_threshold: float = 0.10,  # 10% warning
        critical_threshold: float = 0.15,  # 15% critical
        recovery_threshold: float = 0.05   # 5% recovery alert
    ):
        """Initialize drawdown monitor."""
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.recovery_threshold = recovery_threshold
        
        # Tracking variables
        self.peak_capital = 0.0
        self.max_drawdown_seen = 0.0
        self.drawdown_start_date = None
        self.in_drawdown = False
        
    def update_drawdown(self, current_capital: float) -> List[RiskAlert]:
        """Update drawdown tracking and generate alerts."""
        alerts = []
        
        # Update peak capital
        if current_capital > self.peak_capital:
            self.peak_capital = current_capital
            
            # Check for recovery
            if self.in_drawdown:
                recovery_pct = (current_capital - (self.peak_capital * (1 - self.max_drawdown_seen))) / self.peak_capital
                
                if recovery_pct >= self.recovery_threshold:
                    alert = RiskAlert(
                        alert_id=f"drawdown_recovery_{datetime.now().isoformat()}",
                        alert_type=AlertType.DRAWDOWN_ALERT,
                        severity=AlertSeverity.INFO,
                        timestamp=datetime.now(),
                        message=f"Drawdown recovery: {recovery_pct:.1%} improvement from peak",
                        current_value=recovery_pct,
                        threshold_value=self.recovery_threshold,
                        recommended_actions=["Continue monitoring", "Consider risk rebalancing"]
                    )
                    alerts.append(alert)
            
            self.in_drawdown = False
            self.drawdown_start_date = None
        
        # Calculate current drawdown
        current_drawdown = (self.peak_capital - current_capital) / self.peak_capital if self.peak_capital > 0 else 0.0
        
        # Check for drawdown alerts
        if current_drawdown > self.warning_threshold:
            if not self.in_drawdown:
                self.drawdown_start_date = datetime.now()
                self.in_drawdown = True
            
            self.max_drawdown_seen = max(self.max_drawdown_seen, current_drawdown)
            
            # Generate appropriate alert
            if current_drawdown > self.critical_threshold:
                severity = AlertSeverity.EMERGENCY
                actions = [
                    "Consider halting trading",
                    "Review all positions",
                    "Implement emergency risk controls"
                ]
            else:
                severity = AlertSeverity.WARNING
                actions = [
                    "Reduce position sizes",
                    "Increase cash allocation",
                    "Review risk management"
                ]
            
            alert = RiskAlert(
                alert_id=f"drawdown_alert_{datetime.now().isoformat()}",
                alert_type=AlertType.DRAWDOWN_ALERT,
                severity=severity,
                timestamp=datetime.now(),
                message=f"Drawdown alert: {current_drawdown:.1%} from peak of ${self.peak_capital:,.2f}",
                current_value=current_drawdown,
                threshold_value=self.critical_threshold if current_drawdown > self.critical_threshold else self.warning_threshold,
                recommended_actions=actions,
                metadata={
                    'peak_capital': self.peak_capital,
                    'current_capital': current_capital,
                    'days_in_drawdown': (datetime.now() - self.drawdown_start_date).days if self.drawdown_start_date else 0
                }
            )
            alerts.append(alert)
        
        return alerts


class RiskMonitor:
    """
    Central real-time risk monitoring system.
    
    Orchestrates all risk monitoring components and provides
    a unified interface for risk assessment and alerting.
    """
    
    def __init__(
        self,
        initial_capital: float,
        risk_limit_manager: RiskLimitManager = None,
        update_interval: int = 60,  # 1 minute
        alert_retention_hours: int = 48
    ):
        """
        Initialize risk monitor.
        
        Args:
            initial_capital: Initial portfolio capital
            risk_limit_manager: Risk limit manager instance
            update_interval: Monitoring update interval in seconds
            alert_retention_hours: Hours to retain alerts
        """
        self.initial_capital = initial_capital
        self.risk_limit_manager = risk_limit_manager or RiskLimitManager(initial_capital)
        self.update_interval = update_interval
        self.alert_retention_hours = alert_retention_hours
        
        # Monitoring components
        self.var_monitor = VaRMonitor()
        self.drawdown_monitor = DrawdownMonitor()
        self.performance_calculator = PerformanceCalculator()
        
        # Alert management
        self.active_alerts: List[RiskAlert] = []
        self.alert_history: deque = deque(maxlen=1000)
        
        # Metrics history
        self.metrics_history: deque = deque(maxlen=1440)  # 24 hours at 1-minute intervals
        
        # Monitoring state
        self.is_monitoring = False
        self.last_update = datetime.now()
        
        # Alert callbacks
        self.alert_callbacks: List[Callable] = []
        
        logger.info("Initialized RiskMonitor")
    
    def add_alert_callback(self, callback: Callable[[RiskAlert], None]) -> None:
        """Add callback function for alert notifications."""
        self.alert_callbacks.append(callback)
    
    async def start_monitoring(self) -> None:
        """Start continuous risk monitoring."""
        if self.is_monitoring:
            logger.warning("Risk monitoring already running")
            return
        
        self.is_monitoring = True
        logger.info(f"Starting risk monitoring with {self.update_interval}s intervals")
        
        while self.is_monitoring:
            try:
                await self._update_risk_metrics()
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in risk monitoring loop: {e}")
                await asyncio.sleep(self.update_interval)
    
    def stop_monitoring(self) -> None:
        """Stop continuous risk monitoring."""
        self.is_monitoring = False
        logger.info("Stopped risk monitoring")
    
    async def _update_risk_metrics(self) -> None:
        """Update risk metrics and check for alerts."""
        # This would be called with real portfolio data
        # For now, we'll create a placeholder structure
        
        current_portfolio = {
            'total_capital': self.initial_capital,  # Would be updated from real data
            'positions': {},  # Would contain actual positions
            'daily_pnl': 0.0,  # Would be calculated from real data
            'peak_capital': self.initial_capital
        }
        
        # Calculate comprehensive risk metrics
        risk_metrics = await self._calculate_comprehensive_metrics(current_portfolio)
        
        # Store metrics in history
        self.metrics_history.append(risk_metrics)
        
        # Check for alerts
        alerts = await self._check_all_risk_conditions(current_portfolio, risk_metrics)
        
        # Process new alerts
        for alert in alerts:
            await self._process_alert(alert)
        
        # Clean up old alerts
        self._cleanup_old_alerts()
        
        self.last_update = datetime.now()
    
    async def _calculate_comprehensive_metrics(
        self, 
        portfolio: Dict[str, Any]
    ) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        
        # Basic portfolio metrics
        total_capital = portfolio.get('total_capital', self.initial_capital)
        positions = portfolio.get('positions', {})
        daily_pnl = portfolio.get('daily_pnl', 0.0)
        
        # Calculate position-based metrics
        total_exposure = sum(pos.get('market_value', 0) for pos in positions.values())
        cash_available = total_capital - total_exposure
        
        # Concentration metrics
        if positions:
            position_values = [pos.get('market_value', 0) for pos in positions.values()]
            position_values.sort(reverse=True)
            
            largest_position_pct = position_values[0] / total_capital if total_capital > 0 else 0
            top_3_concentration = sum(position_values[:3]) / total_capital if total_capital > 0 else 0
            herfindahl_index = sum((pv / total_capital) ** 2 for pv in position_values) if total_capital > 0 else 0
        else:
            largest_position_pct = 0.0
            top_3_concentration = 0.0
            herfindahl_index = 0.0
        
        # Performance metrics (would be calculated from real returns data)
        daily_return = daily_pnl / total_capital if total_capital > 0 else 0.0
        
        # Create risk metrics
        risk_metrics = RiskMetrics(
            timestamp=datetime.now(),
            total_capital=total_capital,
            total_exposure=total_exposure,
            cash_available=cash_available,
            unrealized_pnl=0.0,  # Would be calculated from positions
            daily_pnl=daily_pnl,
            
            # Risk measures (simplified for demo)
            portfolio_var_95=0.02,  # Would be calculated
            portfolio_var_99=0.04,  # Would be calculated
            max_drawdown=0.0,  # Would be tracked
            current_drawdown=0.0,  # Would be calculated
            sharpe_ratio=0.0,  # Would be calculated
            sortino_ratio=0.0,  # Would be calculated
            
            # Concentration measures
            largest_position_pct=largest_position_pct,
            top_3_concentration=top_3_concentration,
            herfindahl_index=herfindahl_index,
            
            # Correlation measures (simplified)
            avg_correlation=0.0,  # Would be calculated from returns
            max_correlation=0.0,  # Would be calculated
            correlation_risk_score=0.0,  # Would be calculated
            
            # Performance measures
            daily_return=daily_return,
            volatility=0.02,  # Would be calculated
            win_rate=0.5,  # Would be calculated
            profit_factor=1.0,  # Would be calculated
            
            # System measures
            num_positions=len(positions),
            num_alerts=len(self.active_alerts),
            system_health=1.0  # Overall system health score
        )
        
        return risk_metrics
    
    async def _check_all_risk_conditions(
        self,
        portfolio: Dict[str, Any],
        risk_metrics: RiskMetrics
    ) -> List[RiskAlert]:
        """Check all risk conditions and generate alerts."""
        alerts = []
        
        # Check risk limits
        limits_ok, limit_violations = self.risk_limit_manager.check_all_limits(portfolio)
        for violation in limit_violations:
            alert = self._create_alert_from_violation(violation)
            alerts.append(alert)
        
        # Check VaR
        if len(self.metrics_history) >= 30:  # Need some history
            returns_data = pd.Series([m.daily_return for m in self.metrics_history])
            var_results = self.var_monitor.calculate_var(returns_data)
            var_alerts = self.var_monitor.check_var_breach(var_results)
            alerts.extend(var_alerts)
        
        # Check drawdown
        drawdown_alerts = self.drawdown_monitor.update_drawdown(risk_metrics.total_capital)
        alerts.extend(drawdown_alerts)
        
        # Check concentration
        if risk_metrics.largest_position_pct > 0.15:  # 15% concentration alert
            alert = RiskAlert(
                alert_id=f"concentration_{datetime.now().isoformat()}",
                alert_type=AlertType.CONCENTRATION_RISK,
                severity=AlertSeverity.WARNING,
                timestamp=datetime.now(),
                message=f"High position concentration: {risk_metrics.largest_position_pct:.1%}",
                current_value=risk_metrics.largest_position_pct,
                threshold_value=0.15,
                recommended_actions=["Reduce largest position", "Increase diversification"]
            )
            alerts.append(alert)
        
        # Check system health
        if risk_metrics.system_health < 0.8:
            alert = RiskAlert(
                alert_id=f"system_health_{datetime.now().isoformat()}",
                alert_type=AlertType.SYSTEM_ERROR,
                severity=AlertSeverity.CRITICAL,
                timestamp=datetime.now(),
                message=f"System health degraded: {risk_metrics.system_health:.1%}",
                current_value=risk_metrics.system_health,
                threshold_value=0.8,
                recommended_actions=["Check system components", "Review data quality"]
            )
            alerts.append(alert)
        
        return alerts
    
    def _create_alert_from_violation(self, violation: LimitViolation) -> RiskAlert:
        """Convert limit violation to risk alert."""
        severity_map = {
            'WARNING': AlertSeverity.WARNING,
            'CRITICAL': AlertSeverity.CRITICAL,
            'EMERGENCY': AlertSeverity.EMERGENCY
        }
        
        return RiskAlert(
            alert_id=f"limit_{violation.limit_id}_{datetime.now().isoformat()}",
            alert_type=AlertType.LIMIT_VIOLATION,
            severity=severity_map.get(violation.severity.name, AlertSeverity.WARNING),
            timestamp=violation.timestamp,
            message=violation.message,
            current_value=violation.current_value,
            threshold_value=violation.limit_value,
            affected_positions=violation.affected_positions,
            recommended_actions=["Review position sizes", "Check risk limits"],
            metadata={'violation_type': violation.limit_type.value}
        )
    
    async def _process_alert(self, alert: RiskAlert) -> None:
        """Process a new risk alert."""
        # Check if alert already exists (avoid duplicates)
        existing = [a for a in self.active_alerts if a.alert_type == alert.alert_type and 
                   abs(a.current_value - alert.current_value) < 0.001]
        
        if existing:
            return  # Skip duplicate alert
        
        # Add to active alerts
        self.active_alerts.append(alert)
        self.alert_history.append(alert)
        
        # Log alert
        logger.warning(f"{alert.severity.value.upper()} ALERT: {alert.message}")
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                await asyncio.get_event_loop().run_in_executor(
                    ThreadPoolExecutor(), callback, alert
                )
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def _cleanup_old_alerts(self) -> None:
        """Remove old resolved alerts."""
        cutoff_time = datetime.now() - timedelta(hours=self.alert_retention_hours)
        
        # Remove old alerts
        self.active_alerts = [
            alert for alert in self.active_alerts 
            if alert.timestamp > cutoff_time and not alert.resolved
        ]
    
    def acknowledge_alert(self, alert_id: str, user: str = "system") -> bool:
        """Acknowledge an alert."""
        for alert in self.active_alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.metadata['acknowledged_by'] = user
                alert.metadata['acknowledged_at'] = datetime.now().isoformat()
                logger.info(f"Alert acknowledged: {alert_id} by {user}")
                return True
        return False
    
    def resolve_alert(self, alert_id: str, user: str = "system") -> bool:
        """Resolve an alert."""
        for alert in self.active_alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                alert.metadata['resolved_by'] = user
                alert.metadata['resolved_at'] = datetime.now().isoformat()
                logger.info(f"Alert resolved: {alert_id} by {user}")
                return True
        return False
    
    def get_current_alerts(
        self, 
        severity: AlertSeverity = None,
        alert_type: AlertType = None
    ) -> List[RiskAlert]:
        """Get current active alerts with optional filtering."""
        alerts = self.active_alerts
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        
        return alerts
    
    def get_risk_dashboard_data(self) -> Dict[str, Any]:
        """Get current risk dashboard data."""
        current_metrics = self.metrics_history[-1] if self.metrics_history else None
        
        if not current_metrics:
            return {'error': 'No metrics available'}
        
        # Alert counts by severity
        alert_counts = {
            'info': len([a for a in self.active_alerts if a.severity == AlertSeverity.INFO]),
            'warning': len([a for a in self.active_alerts if a.severity == AlertSeverity.WARNING]),
            'critical': len([a for a in self.active_alerts if a.severity == AlertSeverity.CRITICAL]),
            'emergency': len([a for a in self.active_alerts if a.severity == AlertSeverity.EMERGENCY])
        }
        
        return {
            'timestamp': current_metrics.timestamp.isoformat(),
            'portfolio': {
                'total_capital': current_metrics.total_capital,
                'total_exposure': current_metrics.total_exposure,
                'cash_available': current_metrics.cash_available,
                'daily_pnl': current_metrics.daily_pnl,
                'num_positions': current_metrics.num_positions
            },
            'risk_metrics': {
                'var_95': current_metrics.portfolio_var_95,
                'var_99': current_metrics.portfolio_var_99,
                'max_drawdown': current_metrics.max_drawdown,
                'current_drawdown': current_metrics.current_drawdown,
                'sharpe_ratio': current_metrics.sharpe_ratio
            },
            'concentration': {
                'largest_position': current_metrics.largest_position_pct,
                'top_3_concentration': current_metrics.top_3_concentration,
                'herfindahl_index': current_metrics.herfindahl_index
            },
            'alerts': {
                'total_active': len(self.active_alerts),
                'by_severity': alert_counts
            },
            'system': {
                'monitoring_status': 'active' if self.is_monitoring else 'inactive',
                'last_update': self.last_update.isoformat(),
                'system_health': current_metrics.system_health
            }
        }
    
    def generate_risk_report(self) -> str:
        """Generate comprehensive risk monitoring report."""
        dashboard = self.get_risk_dashboard_data()
        
        if 'error' in dashboard:
            return "❌ Risk monitoring report unavailable - no data"
        
        report = f"""
🛡️  REAL-TIME RISK MONITORING REPORT
{'=' * 60}

Monitoring Status: {'🟢 ACTIVE' if self.is_monitoring else '🔴 INACTIVE'}
Last Update: {dashboard['system']['last_update']}
System Health: {dashboard['system']['system_health']:.1%}

PORTFOLIO OVERVIEW:
  Total Capital: ${dashboard['portfolio']['total_capital']:,.2f}
  Total Exposure: ${dashboard['portfolio']['total_exposure']:,.2f}
  Cash Available: ${dashboard['portfolio']['cash_available']:,.2f}
  Daily P&L: ${dashboard['portfolio']['daily_pnl']:,.2f}
  Positions: {dashboard['portfolio']['num_positions']}

RISK METRICS:
  VaR (95%): {dashboard['risk_metrics']['var_95']:.2%}
  VaR (99%): {dashboard['risk_metrics']['var_99']:.2%}
  Max Drawdown: {dashboard['risk_metrics']['max_drawdown']:.2%}
  Current Drawdown: {dashboard['risk_metrics']['current_drawdown']:.2%}
  Sharpe Ratio: {dashboard['risk_metrics']['sharpe_ratio']:.2f}

CONCENTRATION ANALYSIS:
  Largest Position: {dashboard['concentration']['largest_position']:.1%}
  Top 3 Positions: {dashboard['concentration']['top_3_concentration']:.1%}
  Herfindahl Index: {dashboard['concentration']['herfindahl_index']:.3f}

ACTIVE ALERTS ({dashboard['alerts']['total_active']}):
  🔵 Info: {dashboard['alerts']['by_severity']['info']}
  🟡 Warning: {dashboard['alerts']['by_severity']['warning']}
  🟠 Critical: {dashboard['alerts']['by_severity']['critical']}
  🔴 Emergency: {dashboard['alerts']['by_severity']['emergency']}
"""
        
        # Add recent alerts
        if self.active_alerts:
            report += "\nRECENT ALERTS:\n"
            recent_alerts = sorted(self.active_alerts, key=lambda x: x.timestamp, reverse=True)[:5]
            for alert in recent_alerts:
                status = "✅" if alert.resolved else "⚠️" if alert.acknowledged else "🚨"
                report += f"  {status} {alert.timestamp.strftime('%H:%M:%S')} | {alert.severity.value.upper()}: {alert.message}\n"
        
        return report
