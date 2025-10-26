#!/usr/bin/env python3
"""
Risk Management Example

Demonstrates comprehensive risk management with stop-loss, real-time monitoring,
and automated execution for live trading operations.
"""

import asyncio
import logging
import time
from typing import Dict, Any

from neural.analysis.risk import (
    RiskManager,
    StopLossConfig,
    StopLossType,
    RiskLimits,
    StopLossEngine,
)
from neural.analysis.execution import AutoExecutor, ExecutionConfig
from neural.trading import TradingClient, KalshiWebSocketClient

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RiskManagementDemo:
    """Demonstration of risk management in action."""

    def __init__(self):
        # Initialize components
        self.risk_manager = self._setup_risk_manager()
        self.stop_loss_engine = StopLossEngine()
        self.executor = self._setup_executor()
        self.client = self._setup_client()
        self.ws_client = None

    def _setup_risk_manager(self) -> RiskManager:
        """Configure risk manager with conservative settings."""
        limits = RiskLimits(
            max_drawdown_pct=0.10,  # 10% max drawdown
            max_position_size_pct=0.05,  # 5% of portfolio per position
            daily_loss_limit_pct=0.05,  # 5% daily loss limit
            max_positions=10,
        )

        risk_manager = RiskManager(limits=limits, portfolio_value=10000.0)
        logger.info("Risk manager initialized with limits: %s", limits)
        return risk_manager

    def _setup_executor(self) -> AutoExecutor:
        """Configure automated executor."""
        config = ExecutionConfig(
            enable_auto_execution=True,
            max_orders_per_minute=5,  # Conservative rate limiting
            dry_run=False,  # Set to True for testing without real orders
            emergency_stop_enabled=True,
        )

        executor = AutoExecutor(config=config)
        # Connect executor to risk manager
        self.risk_manager.event_handler = executor

        logger.info("Auto executor initialized with config: %s", config)
        return executor

    def _setup_client(self) -> TradingClient:
        """Configure trading client with risk integration."""
        client = TradingClient(risk_manager=self.risk_manager)
        logger.info("Trading client initialized with risk integration")
        return client

    def demonstrate_stop_loss_types(self):
        """Show different stop-loss configurations."""
        logger.info("=== Demonstrating Stop-Loss Types ===")

        # Percentage stop-loss
        pct_stop = StopLossConfig(type=StopLossType.PERCENTAGE, value=0.05)
        logger.info("Percentage stop-loss (5%%): %s", pct_stop)

        # Absolute stop-loss
        abs_stop = StopLossConfig(type=StopLossType.ABSOLUTE, value=0.45)
        logger.info("Absolute stop-loss ($0.45): %s", abs_stop)

        # Trailing stop-loss
        trail_stop = StopLossConfig(type=StopLossType.TRAILING, value=0.03)
        logger.info("Trailing stop-loss (3%% trail): %s", trail_stop)

    def demonstrate_stop_loss_engine(self):
        """Show advanced stop-loss calculations."""
        logger.info("=== Demonstrating Stop-Loss Engine ===")

        # Fixed stop
        fixed_stop = self.stop_loss_engine.calculate_stop_price(
            entry_price=0.50, current_price=0.52, side="yes", strategy="fixed", stop_pct=0.05
        )
        logger.info("Fixed 5%% stop for long position: %.4f", fixed_stop)

        # Trailing stop
        trail_stop = self.stop_loss_engine.calculate_stop_price(
            entry_price=0.50, current_price=0.55, side="yes", strategy="trailing", trail_pct=0.03
        )
        logger.info("Trailing 3%% stop after price rise: %.4f", trail_stop)

        # Volatility-adjusted stop
        vol_stop = self.stop_loss_engine.calculate_stop_price(
            entry_price=0.50,
            current_price=0.52,
            side="yes",
            strategy="volatility",
            volatility=0.02,
            multiplier=2.0,
        )
        logger.info("Volatility-adjusted stop (2%% vol, 2x multiplier): %.4f", vol_stop)

    def simulate_position_monitoring(self):
        """Simulate real-time position monitoring."""
        logger.info("=== Simulating Position Monitoring ===")

        from neural.analysis.risk import Position

        # Create sample position
        position = Position(
            market_id="demo_market_001",
            side="yes",
            quantity=100,
            entry_price=0.50,
            current_price=0.50,
            stop_loss=StopLossConfig(type=StopLossType.PERCENTAGE, value=0.05),
        )

        self.risk_manager.add_position(position)
        logger.info("Added position: %s", position.market_id)

        # Simulate price changes
        price_changes = [0.52, 0.48, 0.46, 0.44]  # Gradual decline

        for price in price_changes:
            logger.info("Updating price to: %.4f", price)
            events = self.risk_manager.update_position_price("demo_market_001", price)

            if events:
                logger.warning("Risk events triggered: %s", [e.value for e in events])
                break  # Stop-loss triggered

            # Show current metrics
            metrics = self.risk_manager.get_risk_metrics()
            logger.info(
                "Current P&L: $%.2f (%.2f%%)", position.unrealized_pnl, position.pnl_percentage
            )

        # Clean up
        self.risk_manager.remove_position("demo_market_001")

    def demonstrate_risk_limits(self):
        """Show risk limit enforcement."""
        logger.info("=== Demonstrating Risk Limits ===")

        # Test position size limit
        large_position = type(
            "Position",
            (),
            {
                "market_id": "large_pos",
                "current_value": 600.0,  # 6% of $10k portfolio
                "quantity": 100,
            },
        )()

        # This should trigger position size limit
        events = self.risk_manager._check_position_size_limit(large_position)
        if events:
            logger.warning("Position size limit triggered")

        # Test drawdown limit
        self.risk_manager.portfolio_value = 8500.0  # 15% drawdown
        drawdown_events = self.risk_manager._check_drawdown_limit()
        if drawdown_events:
            logger.warning("Drawdown limit triggered")

    def run_websocket_simulation(self):
        """Simulate websocket-based risk monitoring."""
        logger.info("=== WebSocket Risk Monitoring Simulation ===")

        # Create websocket client with risk manager
        self.ws_client = KalshiWebSocketClient(risk_manager=self.risk_manager)

        # Simulate market data messages
        market_updates = [
            {
                "type": "market_price",
                "market": {"id": "market_001", "price": {"latest_price": 0.52}},
            },
            {
                "type": "market_price",
                "market": {"id": "market_001", "price": {"latest_price": 0.48}},
            },
            {
                "type": "market_price",
                "market": {"id": "market_001", "price": {"latest_price": 0.45}},
            },
        ]

        for update in market_updates:
            logger.info("Processing websocket update: %s", update)
            self.ws_client._process_risk_monitoring(update)
            time.sleep(0.1)  # Simulate real-time delay

    def show_execution_summary(self):
        """Display execution summary."""
        logger.info("=== Execution Summary ===")

        summary = self.executor.get_execution_summary()
        logger.info("Auto-execution enabled: %s", summary["auto_execution_enabled"])
        logger.info("Active orders: %d", summary["active_orders"])
        logger.info("Total executions: %d", summary["total_executions"])
        logger.info("Dry run mode: %s", summary["dry_run"])

        metrics = self.risk_manager.get_risk_metrics()
        logger.info("Portfolio value: $%.2f", metrics["portfolio_value"])
        logger.info("Drawdown: %.2f%%", metrics["drawdown_pct"] * 100)
        logger.info("Daily P&L: $%.2f", metrics["daily_pnl"])

    async def run_demo(self):
        """Run the complete risk management demonstration."""
        logger.info("Starting Risk Management Demo")

        try:
            # Basic demonstrations
            self.demonstrate_stop_loss_types()
            print()

            self.demonstrate_stop_loss_engine()
            print()

            self.simulate_position_monitoring()
            print()

            self.demonstrate_risk_limits()
            print()

            # WebSocket simulation (would normally run continuously)
            self.run_websocket_simulation()
            print()

            self.show_execution_summary()

            logger.info("Risk Management Demo completed successfully")

        except Exception as e:
            logger.error("Demo failed: %s", e)
            raise
        finally:
            # Cleanup
            if self.ws_client:
                self.ws_client.close()


def main():
    """Main entry point."""
    demo = RiskManagementDemo()

    # Run async demo
    asyncio.run(demo.run_demo())


if __name__ == "__main__":
    main()
