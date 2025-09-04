#!/usr/bin/env python3
"""
Basic usage example for the Neural SDK.

This example demonstrates:
- SDK initialization
- Configuration management
- Basic trading strategy creation
- Event handling
- System startup and monitoring
"""

import asyncio
import logging
from typing import Optional

from neural_sdk import MarketData, NeuralSDK, TradingSignal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Main example function demonstrating SDK usage."""

    # Initialize SDK from environment variables
    logger.info("Initializing Neural SDK...")
    sdk = NeuralSDK.from_env()

    # Example 1: Simple arbitrage strategy
    @sdk.strategy
    async def arbitrage_strategy(market_data: MarketData) -> Optional[TradingSignal]:
        """
        Simple arbitrage strategy that looks for pricing inefficiencies.

        If the sum of YES and NO prices is less than 0.98 (allowing for fees),
        there's an arbitrage opportunity.
        """
        total_price = market_data.yes_price + market_data.no_price

        if total_price < 0.98:
            # Arbitrage opportunity detected
            logger.info(
                f"Arbitrage opportunity: {market_data.ticker} = {total_price:.3f}"
            )

            return sdk.create_signal(
                action="BUY",
                market_ticker=market_data.ticker,
                side="YES",  # Buy YES, sell NO would be arbitrage
                confidence=min(0.9, (0.98 - total_price) * 50),  # Scale confidence
                quantity=100,
                reason=f"Arbitrage: total price {total_price:.3f} < 0.98",
            )

        return None

    # Example 2: Momentum-based strategy
    @sdk.strategy
    async def momentum_strategy(market_data: MarketData) -> Optional[TradingSignal]:
        """
        Momentum strategy based on extreme price movements.

        Buys when price drops below 30% (oversold)
        Sells when price rises above 70% (overbought)
        """
        if market_data.yes_price < 0.3:
            # Potentially oversold - buy signal
            return sdk.create_signal(
                action="BUY",
                market_ticker=market_data.ticker,
                side="YES",
                confidence=0.7,
                quantity=50,
                reason=f"Oversold signal: price {market_data.yes_price:.3f} < 0.3",
            )

        elif market_data.yes_price > 0.7:
            # Potentially overbought - sell signal
            return sdk.create_signal(
                action="SELL",
                market_ticker=market_data.ticker,
                side="YES",
                confidence=0.7,
                quantity=50,
                reason=f"Overbought signal: price {market_data.yes_price:.3f} > 0.7",
            )

        return None

    # Example 3: Market data event handler
    @sdk.on_market_data
    async def log_market_updates(market_data: MarketData):
        """Log significant market data updates."""
        if market_data.yes_price > 0.8 or market_data.yes_price < 0.2:
            logger.info(
                f"Extreme price: {market_data.ticker} = {market_data.yes_price:.3f}"
            )

    # Example 4: Trading signal event handler
    @sdk.on_signal
    async def log_signals(signal: TradingSignal):
        """Log generated trading signals."""
        logger.info(
            f"Signal generated: {signal.action} {signal.market_ticker} "
            f"(confidence: {signal.confidence:.2f})"
        )

    # Example 5: Trade execution event handler
    @sdk.on_trade
    async def log_trades(trade_result):
        """Log trade execution results."""
        logger.info(
            f"Trade {trade_result.status}: {trade_result.market_ticker} "
            f"x{trade_result.quantity} @ {trade_result.price:.3f}"
        )

    # Display system status
    status = sdk.get_system_status()
    logger.info(f"System status: {status}")

    # Start the trading system
    logger.info("Starting trading system...")
    try:
        await sdk.start_trading_system()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal...")
    finally:
        # Graceful shutdown
        await sdk.stop_trading_system()
        logger.info("Trading system shut down gracefully")


async def advanced_example():
    """Advanced example with custom configuration and multiple strategies."""

    # Custom configuration
    from kalshi_trading_sdk import SDKConfig

    config = SDKConfig(
        environment="development",
        risk_limits__max_position_size_pct=0.03,  # More conservative
        trading__default_quantity=25,
        data__redis_url="redis://localhost:6379",
    )

    sdk = KalshiSDK(config)

    # Advanced strategy with risk management
    @sdk.strategy
    async def risk_aware_strategy(market_data: MarketData) -> Optional[TradingSignal]:
        """Strategy that considers portfolio risk and market conditions."""

        # Check if we have capacity for this trade
        portfolio_status = await sdk.get_portfolio_status()
        exposure_pct = portfolio_status.get("exposure_pct", 0)

        # Don't trade if we're already heavily exposed
        if exposure_pct > 0.8:  # 80% of max exposure
            return None

        # Only trade high-confidence opportunities
        if market_data.yes_price < 0.25 and market_data.volume > 1000:
            return sdk.create_signal(
                action="BUY",
                market_ticker=market_data.ticker,
                side="YES",
                confidence=0.8,
                quantity=25,
                reason="High-confidence oversold with volume",
            )

        return None

    # Monitor portfolio health
    @sdk.on_market_data
    async def monitor_portfolio_health(market_data: MarketData):
        """Monitor overall portfolio health."""
        portfolio = await sdk.get_portfolio_status()

        if portfolio["daily_pnl"] < -100:  # Stop loss threshold
            logger.warning(f"Daily P&L concerning: ${portfolio['daily_pnl']:.2f}")

        if portfolio["exposure_pct"] > 0.9:  # High exposure warning
            logger.warning(f"High exposure: {portfolio['exposure_pct']:.1%}")

    logger.info("Starting advanced trading system...")
    try:
        await sdk.start_trading_system()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await sdk.stop_trading_system()


if __name__ == "__main__":
    # Run basic example
    asyncio.run(main())

    # Uncomment to run advanced example
    # asyncio.run(advanced_example())
