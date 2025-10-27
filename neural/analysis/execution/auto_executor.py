"""
Automated Execution Layer for Risk Management

Handles risk event detection and automated order generation for position management.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

try:
    from neural.analysis.risk import RiskEvent, RiskEventHandler

    RISK_MODULE_AVAILABLE = True
except ImportError:
    RISK_MODULE_AVAILABLE = False
    # Define minimal types if risk module not available
    from enum import Enum
    from typing import Protocol

    class RiskEvent(Enum):
        STOP_LOSS_TRIGGERED = "stop_loss_triggered"
        MAX_DRAWDOWN_EXCEEDED = "max_drawdown_exceeded"
        DAILY_LOSS_LIMIT_EXCEEDED = "daily_loss_limit_exceeded"
        POSITION_SIZE_EXCEEDED = "position_size_exceeded"

    class RiskEventHandler(Protocol):
        def on_risk_event(self, event, position, data): ...


from neural.trading.client import TradingClient

_LOG = logging.getLogger(__name__)


class OrderExecutor(Protocol):
    """Protocol for executing orders."""

    def submit_market_order(
        self, market_id: str, side: str, quantity: int, close_position: bool = False
    ) -> dict[str, Any]:
        """Submit a market order."""
        ...

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        ...


@dataclass
class ExecutionConfig:
    """Configuration for automated execution."""

    enable_auto_execution: bool = True
    max_orders_per_minute: int = 10
    require_confirmation: bool = False
    dry_run: bool = False
    emergency_stop_enabled: bool = True


@dataclass
class AutoExecutor(RiskEventHandler):
    """
    Automated execution layer that responds to risk events with order generation.

    Monitors risk events from RiskManager and executes appropriate trading actions
    such as stop-loss orders, position closures, or risk-reducing trades.
    """

    trading_client: TradingClient | None = None
    config: ExecutionConfig = field(default_factory=ExecutionConfig)

    # Runtime state
    active_orders: dict[str, dict[str, Any]] = field(default_factory=dict)
    execution_history: list[dict[str, Any]] = field(default_factory=list)
    rate_limiter: dict[int, int] = field(default_factory=dict)  # Track orders per minute

    def __post_init__(self) -> None:
        """Initialize executor."""
        if self.trading_client is None and not self.config.dry_run:
            _LOG.warning("No trading client provided - operating in dry-run mode")
            self.config.dry_run = True

    def on_risk_event(
        self, event: RiskEvent, position_data: Any, event_data: dict[str, Any]
    ) -> None:
        """
        Handle risk events by generating appropriate orders.

        Args:
            event: The type of risk event
            position_data: Position object or market_id string
            event_data: Additional event context
        """
        if not self.config.enable_auto_execution:
            _LOG.info(f"Auto-execution disabled, skipping {event.value}")
            return

        # Extract position information
        market_id = ""
        if hasattr(position_data, "market_id"):
            market_id = position_data.market_id
        elif isinstance(position_data, str):
            market_id = position_data

        if not market_id:
            _LOG.error(f"Could not extract market_id from position_data: {position_data}")
            return

        # Handle different risk events
        if event == RiskEvent.STOP_LOSS_TRIGGERED:
            self._handle_stop_loss(market_id, position_data, event_data)
        elif event == RiskEvent.MAX_DRAWDOWN_EXCEEDED:
            self._handle_max_drawdown(market_id, event_data)
        elif event == RiskEvent.DAILY_LOSS_LIMIT_EXCEEDED:
            self._handle_daily_loss_limit(event_data)
        elif event == RiskEvent.POSITION_SIZE_EXCEEDED:
            self._handle_position_size_exceeded(market_id, position_data, event_data)

    def _handle_stop_loss(self, market_id: str, position: Any, event_data: dict[str, Any]) -> None:
        """Execute stop-loss by closing the position."""
        _LOG.warning(f"Executing stop-loss for position in {market_id}")

        if self._check_rate_limit():
            return

        try:
            # Determine position side and quantity
            side = "yes" if hasattr(position, "side") and position.side == "yes" else "no"
            quantity = getattr(position, "quantity", 0)

            if quantity == 0:
                _LOG.error(f"Invalid quantity for stop-loss: {quantity}")
                return

            # Submit market order to close position
            close_side = "no" if side == "yes" else "yes"  # Opposite side to close
            order_result = self._execute_market_order(
                market_id=market_id,
                side=close_side,  # Close by taking opposite side
                quantity=quantity,
                reason="stop_loss",
                event_data=event_data,
            )

            if order_result:
                _LOG.info(f"Stop-loss order executed for {market_id}: {order_result}")

        except Exception as e:
            _LOG.error(f"Failed to execute stop-loss for {market_id}: {e}")

    def _handle_max_drawdown(self, market_id: str, event_data: dict[str, Any]) -> None:
        """Handle max drawdown by reducing position sizes."""
        _LOG.warning(f"Max drawdown exceeded, considering position reduction for {market_id}")

        # Could implement position reduction logic here
        # For now, just log the event
        self._log_execution_event("max_drawdown_action", market_id, event_data)

    def _handle_daily_loss_limit(self, event_data: dict[str, Any]) -> None:
        """Handle daily loss limit by stopping all trading."""
        _LOG.critical("Daily loss limit exceeded - initiating emergency stop")

        if self.config.emergency_stop_enabled:
            self._emergency_stop(event_data)

    def _handle_position_size_exceeded(
        self, market_id: str, position: Any, event_data: dict[str, Any]
    ) -> None:
        """Handle oversized position by reducing it."""
        _LOG.warning(f"Position size exceeded for {market_id}, considering reduction")

        # Could implement position size reduction
        self._log_execution_event("position_size_reduction", market_id, event_data)

    def _execute_market_order(
        self, market_id: str, side: str, quantity: int, reason: str, event_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Execute a market order with proper error handling."""
        if self.config.dry_run:
            _LOG.info(
                f"DRY RUN: Would execute {side} order for {quantity} units in {market_id} (reason: {reason})"
            )
            return {"dry_run": True, "market_id": market_id, "side": side, "quantity": quantity}

        if not self.trading_client:
            _LOG.error("No trading client available for order execution")
            return None

        try:
            # Use the trading client's order execution
            # This assumes the TradingClient has a method to place orders
            # We'll need to adapt based on the actual API
            order_result = self._submit_order_via_client(market_id, side, quantity)

            # Track the order
            self.active_orders[order_result.get("order_id", f"order_{len(self.active_orders)}")] = {
                "market_id": market_id,
                "side": side,
                "quantity": quantity,
                "reason": reason,
                "timestamp": event_data.get("timestamp", 0),
                "result": order_result,
            }

            self._log_execution_event(
                "order_executed",
                market_id,
                {**event_data, "order_result": order_result, "reason": reason},
            )

            return order_result

        except Exception as e:
            _LOG.error(f"Order execution failed: {e}")
            self._log_execution_event(
                "order_failed", market_id, {**event_data, "error": str(e), "reason": reason}
            )
            return None

    def _submit_order_via_client(self, market_id: str, side: str, quantity: int) -> dict[str, Any]:
        """Submit order through the trading client."""
        # This is a placeholder - need to implement based on actual TradingClient API
        # Assuming TradingClient has an order method

        if hasattr(self.trading_client, "place_order"):
            # Hypothetical API call
            return self.trading_client.place_order(
                market_id=market_id, side=side, quantity=quantity, order_type="market"
            )
        else:
            raise NotImplementedError("TradingClient does not have place_order method")

    def _check_rate_limit(self) -> bool:
        """Check if we're exceeding order rate limits."""
        # Simple rate limiting - could be enhanced
        import time

        current_minute = int(time.time() // 60)

        if current_minute not in self.rate_limiter:
            self.rate_limiter[current_minute] = 0

        if self.rate_limiter[current_minute] >= self.config.max_orders_per_minute:
            _LOG.warning(
                f"Rate limit exceeded: {self.rate_limiter[current_minute]} orders this minute"
            )
            return True

        self.rate_limiter[current_minute] += 1
        return False

    def _emergency_stop(self, event_data: dict[str, Any]) -> None:
        """Execute emergency stop - cancel all pending orders and stop trading."""
        _LOG.critical("EMERGENCY STOP ACTIVATED")

        # Cancel all active orders
        for order_id, _order_info in self.active_orders.items():
            try:
                if hasattr(self.trading_client, "cancel_order"):
                    self.trading_client.cancel_order(order_id)
                    _LOG.info(f"Cancelled order {order_id} during emergency stop")
            except Exception as e:
                _LOG.error(f"Failed to cancel order {order_id}: {e}")

        # Disable auto-execution
        self.config.enable_auto_execution = False

        self._log_execution_event("emergency_stop", "all", event_data)

    def _log_execution_event(self, event_type: str, market_id: str, data: dict[str, Any]) -> None:
        """Log execution events for monitoring and debugging."""
        event_record = {
            "event_type": event_type,
            "market_id": market_id,
            "timestamp": data.get("timestamp", 0),
            "data": data,
        }
        self.execution_history.append(event_record)
        _LOG.info(f"Execution event: {event_type} for {market_id}")

    def emergency_stop(self, event_data: dict[str, Any] | None = None) -> None:
        """Public method to trigger emergency stop - cancel all pending orders and stop trading."""
        if event_data is None:
            event_data = {"reason": "manual_emergency_stop"}
        self._emergency_stop(event_data)

    def get_execution_summary(self) -> dict[str, Any]:
        """Get summary of execution activity."""
        return {
            "active_orders": len(self.active_orders),
            "total_executions": len(self.execution_history),
            "dry_run": self.config.dry_run,
            "auto_execution_enabled": self.config.enable_auto_execution,
            "recent_events": self.execution_history[-10:] if self.execution_history else [],
        }
