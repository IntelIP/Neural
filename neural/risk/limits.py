"""
Risk Limits and Constraint Enforcement

This module provides comprehensive risk limit management and enforcement
for trading operations, including:

- Position limits (size, concentration, correlation)
- Loss limits (daily, drawdown, stop-loss)  
- Exposure limits (sector, geographic, temporal)
- Leverage and margin limits
- Liquidity and market impact limits
- Custom constraint definitions

All limits are enforced in real-time with configurable actions
(warning, position reduction, trading halt) when breached.
"""

import logging
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd

from neural.strategy.base import Signal

logger = logging.getLogger(__name__)


class LimitType(Enum):
    """Types of risk limits."""
    POSITION_SIZE = "position_size"          # Max position size
    POSITION_COUNT = "position_count"        # Max number of positions  
    CONCENTRATION = "concentration"          # Max concentration in single asset
    CORRELATION = "correlation"              # Max correlated positions
    DAILY_LOSS = "daily_loss"               # Max daily loss
    TOTAL_LOSS = "total_loss"               # Max total loss (drawdown)
    LEVERAGE = "leverage"                    # Max leverage ratio
    EXPOSURE = "exposure"                    # Max exposure to category
    LIQUIDITY = "liquidity"                  # Min liquidity requirement
    VAR = "var"                             # Value at Risk limit
    CUSTOM = "custom"                        # Custom constraint


class LimitAction(Enum):
    """Actions when limits are breached."""
    WARNING = "warning"                      # Log warning only
    REDUCE_POSITION = "reduce_position"      # Reduce violating positions
    REJECT_TRADE = "reject_trade"           # Reject new trades
    CLOSE_POSITIONS = "close_positions"      # Close violating positions
    HALT_TRADING = "halt_trading"           # Stop all trading
    CUSTOM = "custom"                        # Custom action


class LimitSeverity(Enum):
    """Severity levels for limit breaches."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class LimitViolation:
    """Record of a limit violation."""
    limit_id: str
    limit_type: LimitType
    current_value: float
    limit_value: float
    severity: LimitSeverity
    action_taken: LimitAction
    timestamp: datetime
    affected_positions: List[str] = field(default_factory=list)
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskLimit:
    """Definition of a risk limit constraint."""
    limit_id: str
    limit_type: LimitType
    limit_value: float
    warning_threshold: float = None  # Warning at % of limit
    action: LimitAction = LimitAction.WARNING
    enabled: bool = True
    description: str = ""
    custom_check: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.warning_threshold is None:
            self.warning_threshold = self.limit_value * 0.8  # 80% warning


class PositionLimit(RiskLimit):
    """Position size limit constraint."""
    
    def __init__(
        self,
        limit_id: str,
        max_position_pct: float = 0.1,  # 10% max
        max_position_amount: float = None,
        action: LimitAction = LimitAction.REDUCE_POSITION
    ):
        super().__init__(
            limit_id=limit_id,
            limit_type=LimitType.POSITION_SIZE,
            limit_value=max_position_pct,
            action=action,
            description=f"Maximum position size: {max_position_pct*100:.1f}%"
        )
        self.max_position_amount = max_position_amount


class ExposureLimit(RiskLimit):
    """Exposure limit for categories (sector, sport, etc.)."""
    
    def __init__(
        self,
        limit_id: str,
        category: str,  # e.g., "sport", "event_type"
        max_exposure_pct: float = 0.3,  # 30% max
        action: LimitAction = LimitAction.WARNING
    ):
        super().__init__(
            limit_id=limit_id,
            limit_type=LimitType.EXPOSURE,
            limit_value=max_exposure_pct,
            action=action,
            description=f"Maximum {category} exposure: {max_exposure_pct*100:.1f}%"
        )
        self.category = category


class LossLimit(RiskLimit):
    """Loss limit constraint."""
    
    def __init__(
        self,
        limit_id: str,
        max_loss_pct: float = 0.05,  # 5% max loss
        time_period: str = "daily",  # "daily", "total", "monthly"
        action: LimitAction = LimitAction.HALT_TRADING
    ):
        super().__init__(
            limit_id=limit_id,
            limit_type=LimitType.DAILY_LOSS if time_period == "daily" else LimitType.TOTAL_LOSS,
            limit_value=max_loss_pct,
            action=action,
            description=f"Maximum {time_period} loss: {max_loss_pct*100:.1f}%"
        )
        self.time_period = time_period


class RiskLimitManager:
    """
    Central risk limit management and enforcement system.
    
    This class manages all risk limits, monitors for violations,
    and enforces appropriate actions when limits are breached.
    """
    
    def __init__(
        self,
        initial_capital: float,
        enable_enforcement: bool = True
    ):
        """
        Initialize risk limit manager.
        
        Args:
            initial_capital: Initial trading capital
            enable_enforcement: Whether to enforce limits (vs warn only)
        """
        self.initial_capital = initial_capital
        self.enable_enforcement = enable_enforcement
        
        # Risk limits registry
        self.limits: Dict[str, RiskLimit] = {}
        
        # Violation history
        self.violation_history: List[LimitViolation] = []
        
        # Current state
        self.trading_halted = False
        self.halt_reason = ""
        self.last_check_time = datetime.now()
        
        # Performance tracking
        self.checks_performed = 0
        self.violations_detected = 0
        
        logger.info(f"Initialized RiskLimitManager with ${initial_capital:,.2f} capital")
        
        # Add default limits
        self._add_default_limits()
    
    def add_limit(self, limit: RiskLimit) -> None:
        """Add a risk limit to the manager."""
        self.limits[limit.limit_id] = limit
        logger.info(f"Added risk limit: {limit.limit_id} ({limit.limit_type.value})")
    
    def remove_limit(self, limit_id: str) -> bool:
        """Remove a risk limit."""
        if limit_id in self.limits:
            del self.limits[limit_id]
            logger.info(f"Removed risk limit: {limit_id}")
            return True
        return False
    
    def enable_limit(self, limit_id: str, enabled: bool = True) -> bool:
        """Enable or disable a specific limit."""
        if limit_id in self.limits:
            self.limits[limit_id].enabled = enabled
            logger.info(f"{'Enabled' if enabled else 'Disabled'} limit: {limit_id}")
            return True
        return False
    
    def check_all_limits(
        self,
        current_portfolio: Dict[str, Any],
        proposed_trade: Optional[Dict[str, Any]] = None,
        market_data: Dict[str, Any] = None
    ) -> Tuple[bool, List[LimitViolation]]:
        """
        Check all enabled limits against current state.
        
        Args:
            current_portfolio: Current portfolio state
            proposed_trade: Proposed trade to validate
            market_data: Current market data
            
        Returns:
            Tuple of (all_limits_ok, violations_list)
        """
        if self.trading_halted:
            logger.warning("Trading is halted, all limit checks fail")
            return False, []
        
        self.checks_performed += 1
        self.last_check_time = datetime.now()
        
        violations = []
        
        # Check each enabled limit
        for limit in self.limits.values():
            if not limit.enabled:
                continue
            
            try:
                violation = self._check_single_limit(
                    limit, current_portfolio, proposed_trade, market_data
                )
                
                if violation:
                    violations.append(violation)
                    self.violations_detected += 1
                    
                    # Take action if enforcement is enabled
                    if self.enable_enforcement:
                        self._enforce_limit_action(violation, current_portfolio)
                    
            except Exception as e:
                logger.error(f"Error checking limit {limit.limit_id}: {e}")
        
        # Store violations
        self.violation_history.extend(violations)
        
        # Keep only recent violations (last 1000)
        if len(self.violation_history) > 1000:
            self.violation_history = self.violation_history[-1000:]
        
        all_limits_ok = len(violations) == 0
        
        if not all_limits_ok:
            logger.warning(f"Detected {len(violations)} limit violations")
        
        return all_limits_ok, violations
    
    def check_trade_allowed(
        self,
        signal: Signal,
        current_portfolio: Dict[str, Any],
        market_data: Dict[str, Any] = None
    ) -> Tuple[bool, List[str]]:
        """
        Check if a proposed trade is allowed under current limits.
        
        Args:
            signal: Trading signal to validate
            current_portfolio: Current portfolio state
            market_data: Market data for context
            
        Returns:
            Tuple of (trade_allowed, violation_reasons)
        """
        if self.trading_halted:
            return False, [f"Trading halted: {self.halt_reason}"]
        
        # Simulate the proposed trade
        proposed_trade = {
            'market_id': signal.market_id,
            'side': signal.signal_type.value,
            'quantity': getattr(signal, 'recommended_contracts', 1),
            'price': getattr(signal, 'market_price', 0.5),
            'confidence': signal.confidence
        }
        
        # Check limits
        all_ok, violations = self.check_all_limits(
            current_portfolio, proposed_trade, market_data
        )
        
        rejection_reasons = []
        for violation in violations:
            if violation.action_taken in [LimitAction.REJECT_TRADE, LimitAction.HALT_TRADING]:
                rejection_reasons.append(
                    f"{violation.limit_id}: {violation.message}"
                )
        
        trade_allowed = all_ok or len(rejection_reasons) == 0
        
        return trade_allowed, rejection_reasons
    
    def _check_single_limit(
        self,
        limit: RiskLimit,
        current_portfolio: Dict[str, Any],
        proposed_trade: Optional[Dict[str, Any]] = None,
        market_data: Dict[str, Any] = None
    ) -> Optional[LimitViolation]:
        """Check a single limit constraint."""
        
        if limit.limit_type == LimitType.POSITION_SIZE:
            return self._check_position_size_limit(limit, current_portfolio, proposed_trade)
        
        elif limit.limit_type == LimitType.POSITION_COUNT:
            return self._check_position_count_limit(limit, current_portfolio, proposed_trade)
        
        elif limit.limit_type == LimitType.CONCENTRATION:
            return self._check_concentration_limit(limit, current_portfolio, proposed_trade)
        
        elif limit.limit_type == LimitType.DAILY_LOSS:
            return self._check_daily_loss_limit(limit, current_portfolio)
        
        elif limit.limit_type == LimitType.TOTAL_LOSS:
            return self._check_total_loss_limit(limit, current_portfolio)
        
        elif limit.limit_type == LimitType.EXPOSURE:
            return self._check_exposure_limit(limit, current_portfolio, proposed_trade)
        
        elif limit.limit_type == LimitType.CUSTOM:
            return self._check_custom_limit(limit, current_portfolio, proposed_trade, market_data)
        
        else:
            logger.warning(f"Unknown limit type: {limit.limit_type}")
            return None
    
    def _check_position_size_limit(
        self,
        limit: PositionLimit,
        current_portfolio: Dict[str, Any],
        proposed_trade: Optional[Dict[str, Any]] = None
    ) -> Optional[LimitViolation]:
        """Check position size limits."""
        current_capital = current_portfolio.get('total_capital', self.initial_capital)
        
        # Check existing positions
        positions = current_portfolio.get('positions', {})
        
        for market_id, position in positions.items():
            position_value = position.get('market_value', 0)
            position_pct = position_value / current_capital if current_capital > 0 else 0
            
            if position_pct > limit.limit_value:
                return LimitViolation(
                    limit_id=limit.limit_id,
                    limit_type=limit.limit_type,
                    current_value=position_pct,
                    limit_value=limit.limit_value,
                    severity=LimitSeverity.WARNING if position_pct < limit.limit_value * 1.2 else LimitSeverity.CRITICAL,
                    action_taken=limit.action,
                    timestamp=datetime.now(),
                    affected_positions=[market_id],
                    message=f"Position {market_id} exceeds size limit: {position_pct:.1%} > {limit.limit_value:.1%}"
                )
        
        # Check proposed trade
        if proposed_trade:
            trade_value = proposed_trade.get('quantity', 0) * proposed_trade.get('price', 0)
            trade_pct = trade_value / current_capital if current_capital > 0 else 0
            
            # Add to existing position if any
            market_id = proposed_trade.get('market_id')
            existing_position = positions.get(market_id, {})
            existing_value = existing_position.get('market_value', 0)
            
            total_position_value = existing_value + trade_value
            total_position_pct = total_position_value / current_capital if current_capital > 0 else 0
            
            if total_position_pct > limit.limit_value:
                return LimitViolation(
                    limit_id=limit.limit_id,
                    limit_type=limit.limit_type,
                    current_value=total_position_pct,
                    limit_value=limit.limit_value,
                    severity=LimitSeverity.WARNING,
                    action_taken=limit.action,
                    timestamp=datetime.now(),
                    affected_positions=[market_id],
                    message=f"Proposed trade would exceed position size limit: {total_position_pct:.1%} > {limit.limit_value:.1%}"
                )
        
        return None
    
    def _check_position_count_limit(
        self,
        limit: RiskLimit,
        current_portfolio: Dict[str, Any],
        proposed_trade: Optional[Dict[str, Any]] = None
    ) -> Optional[LimitViolation]:
        """Check maximum number of positions."""
        positions = current_portfolio.get('positions', {})
        current_count = len(positions)
        
        # Check if adding proposed trade would exceed limit
        if proposed_trade and current_count >= limit.limit_value:
            market_id = proposed_trade.get('market_id')
            if market_id not in positions:  # New position
                return LimitViolation(
                    limit_id=limit.limit_id,
                    limit_type=limit.limit_type,
                    current_value=current_count + 1,
                    limit_value=limit.limit_value,
                    severity=LimitSeverity.WARNING,
                    action_taken=limit.action,
                    timestamp=datetime.now(),
                    message=f"Too many positions: {current_count + 1} > {limit.limit_value}"
                )
        
        return None
    
    def _check_concentration_limit(
        self,
        limit: RiskLimit,
        current_portfolio: Dict[str, Any],
        proposed_trade: Optional[Dict[str, Any]] = None
    ) -> Optional[LimitViolation]:
        """Check concentration limits (e.g., top 3 positions < 60%)."""
        positions = current_portfolio.get('positions', {})
        current_capital = current_portfolio.get('total_capital', self.initial_capital)
        
        if current_capital <= 0 or not positions:
            return None
        
        # Calculate position percentages
        position_pcts = []
        for market_id, position in positions.items():
            position_value = position.get('market_value', 0)
            position_pct = position_value / current_capital
            position_pcts.append((market_id, position_pct))
        
        # Sort by size and check top N concentration
        position_pcts.sort(key=lambda x: x[1], reverse=True)
        top_3_concentration = sum(pct for _, pct in position_pcts[:3])
        
        if top_3_concentration > limit.limit_value:
            return LimitViolation(
                limit_id=limit.limit_id,
                limit_type=limit.limit_type,
                current_value=top_3_concentration,
                limit_value=limit.limit_value,
                severity=LimitSeverity.WARNING,
                action_taken=limit.action,
                timestamp=datetime.now(),
                affected_positions=[market_id for market_id, _ in position_pcts[:3]],
                message=f"Portfolio concentration too high: {top_3_concentration:.1%} > {limit.limit_value:.1%}"
            )
        
        return None
    
    def _check_daily_loss_limit(
        self,
        limit: RiskLimit,
        current_portfolio: Dict[str, Any]
    ) -> Optional[LimitViolation]:
        """Check daily loss limits."""
        daily_pnl = current_portfolio.get('daily_pnl', 0)
        current_capital = current_portfolio.get('total_capital', self.initial_capital)
        
        if daily_pnl >= 0:  # No loss
            return None
        
        daily_loss_pct = abs(daily_pnl) / current_capital if current_capital > 0 else 0
        
        if daily_loss_pct > limit.limit_value:
            return LimitViolation(
                limit_id=limit.limit_id,
                limit_type=limit.limit_type,
                current_value=daily_loss_pct,
                limit_value=limit.limit_value,
                severity=LimitSeverity.CRITICAL,
                action_taken=limit.action,
                timestamp=datetime.now(),
                message=f"Daily loss limit exceeded: {daily_loss_pct:.1%} > {limit.limit_value:.1%}"
            )
        
        return None
    
    def _check_total_loss_limit(
        self,
        limit: RiskLimit,
        current_portfolio: Dict[str, Any]
    ) -> Optional[LimitViolation]:
        """Check total loss/drawdown limits."""
        current_capital = current_portfolio.get('total_capital', self.initial_capital)
        peak_capital = current_portfolio.get('peak_capital', self.initial_capital)
        
        if current_capital >= peak_capital:  # No drawdown
            return None
        
        drawdown = (peak_capital - current_capital) / peak_capital
        
        if drawdown > limit.limit_value:
            return LimitViolation(
                limit_id=limit.limit_id,
                limit_type=limit.limit_type,
                current_value=drawdown,
                limit_value=limit.limit_value,
                severity=LimitSeverity.EMERGENCY,
                action_taken=limit.action,
                timestamp=datetime.now(),
                message=f"Maximum drawdown exceeded: {drawdown:.1%} > {limit.limit_value:.1%}"
            )
        
        return None
    
    def _check_exposure_limit(
        self,
        limit: ExposureLimit,
        current_portfolio: Dict[str, Any],
        proposed_trade: Optional[Dict[str, Any]] = None
    ) -> Optional[LimitViolation]:
        """Check exposure limits by category."""
        # This would need category information about positions
        # Placeholder implementation
        return None
    
    def _check_custom_limit(
        self,
        limit: RiskLimit,
        current_portfolio: Dict[str, Any],
        proposed_trade: Optional[Dict[str, Any]] = None,
        market_data: Dict[str, Any] = None
    ) -> Optional[LimitViolation]:
        """Check custom limit with user-defined function."""
        if not limit.custom_check:
            return None
        
        try:
            violation_info = limit.custom_check(
                current_portfolio, proposed_trade, market_data
            )
            
            if violation_info:
                return LimitViolation(
                    limit_id=limit.limit_id,
                    limit_type=limit.limit_type,
                    current_value=violation_info.get('current_value', 0),
                    limit_value=limit.limit_value,
                    severity=LimitSeverity.WARNING,
                    action_taken=limit.action,
                    timestamp=datetime.now(),
                    message=violation_info.get('message', 'Custom limit violation')
                )
        
        except Exception as e:
            logger.error(f"Custom limit check failed for {limit.limit_id}: {e}")
        
        return None
    
    def _enforce_limit_action(
        self,
        violation: LimitViolation,
        current_portfolio: Dict[str, Any]
    ) -> None:
        """Enforce the action specified for a limit violation."""
        
        if violation.action_taken == LimitAction.WARNING:
            logger.warning(f"Risk limit violation: {violation.message}")
        
        elif violation.action_taken == LimitAction.REDUCE_POSITION:
            logger.warning(f"Reducing positions due to limit violation: {violation.message}")
            # Implementation would reduce violating positions
        
        elif violation.action_taken == LimitAction.REJECT_TRADE:
            logger.warning(f"Trade rejected due to limit violation: {violation.message}")
            # Trade rejection handled at higher level
        
        elif violation.action_taken == LimitAction.CLOSE_POSITIONS:
            logger.error(f"Closing positions due to limit violation: {violation.message}")
            # Implementation would close violating positions
        
        elif violation.action_taken == LimitAction.HALT_TRADING:
            self.trading_halted = True
            self.halt_reason = violation.message
            logger.error(f"TRADING HALTED: {violation.message}")
        
        # Custom actions would be handled by callbacks
    
    def _add_default_limits(self) -> None:
        """Add sensible default risk limits."""
        
        # Position size limit: Max 10% per position
        self.add_limit(PositionLimit(
            limit_id="max_position_size",
            max_position_pct=0.10,
            action=LimitAction.REDUCE_POSITION
        ))
        
        # Position count limit: Max 20 positions
        self.add_limit(RiskLimit(
            limit_id="max_positions",
            limit_type=LimitType.POSITION_COUNT,
            limit_value=20,
            action=LimitAction.REJECT_TRADE,
            description="Maximum number of concurrent positions"
        ))
        
        # Daily loss limit: Max 5% daily loss
        self.add_limit(LossLimit(
            limit_id="daily_loss_limit",
            max_loss_pct=0.05,
            time_period="daily",
            action=LimitAction.HALT_TRADING
        ))
        
        # Total loss limit: Max 20% drawdown
        self.add_limit(LossLimit(
            limit_id="max_drawdown",
            max_loss_pct=0.20,
            time_period="total",
            action=LimitAction.HALT_TRADING
        ))
        
        # Concentration limit: Top 3 positions max 60%
        self.add_limit(RiskLimit(
            limit_id="concentration_limit",
            limit_type=LimitType.CONCENTRATION,
            limit_value=0.60,
            action=LimitAction.WARNING,
            description="Maximum concentration in top 3 positions"
        ))
    
    def get_limit_status(self) -> Dict[str, Any]:
        """Get current status of all limits."""
        status = {
            'trading_halted': self.trading_halted,
            'halt_reason': self.halt_reason,
            'total_limits': len(self.limits),
            'enabled_limits': sum(1 for limit in self.limits.values() if limit.enabled),
            'checks_performed': self.checks_performed,
            'violations_detected': self.violations_detected,
            'recent_violations': len([
                v for v in self.violation_history 
                if v.timestamp > datetime.now() - timedelta(hours=24)
            ])
        }
        
        return status
    
    def reset_trading_halt(self, reason: str = "Manual reset") -> None:
        """Reset trading halt status."""
        self.trading_halted = False
        self.halt_reason = ""
        logger.info(f"Trading halt reset: {reason}")
    
    def generate_risk_report(self) -> str:
        """Generate a comprehensive risk limits report."""
        status = self.get_limit_status()
        
        report = f"""
🛡️  RISK LIMITS STATUS REPORT
{'=' * 50}

System Status: {'🚫 HALTED' if status['trading_halted'] else '✅ ACTIVE'}
{f"Halt Reason: {status['halt_reason']}" if status['trading_halted'] else ''}

Limits Overview:
  Total Limits: {status['total_limits']}
  Enabled: {status['enabled_limits']}
  Checks Performed: {status['checks_performed']}
  Violations Detected: {status['violations_detected']}
  Recent Violations (24h): {status['recent_violations']}

ACTIVE LIMITS:
"""
        
        for limit_id, limit in self.limits.items():
            if limit.enabled:
                report += f"  ✅ {limit_id}: {limit.description}\n"
                report += f"     Limit: {limit.limit_value}\n"
                report += f"     Action: {limit.action.value}\n\n"
        
        if self.violation_history:
            recent_violations = [
                v for v in self.violation_history[-10:]  # Last 10 violations
            ]
            
            report += "RECENT VIOLATIONS:\n"
            for violation in recent_violations:
                report += f"  ⚠️  {violation.timestamp.strftime('%H:%M:%S')}: {violation.message}\n"
        
        return report
