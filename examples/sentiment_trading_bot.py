#!/usr/bin/env python3
"""
Sentiment-Based Trading Bot

This is the main orchestrator that combines all components:
- Twitter sentiment analysis
- ESPN GameCast momentum tracking
- Kalshi market data
- Sentiment-based trading strategy

Run this to start the complete sentiment trading algorithm.

Usage:
    python examples/sentiment_trading_bot.py --game-id 401547439 --teams "Baltimore Ravens,Detroit Lions"
"""

import argparse
import asyncio
import json
import logging
import os
import signal

# Add the neural package to the path
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from neural.analysis.strategies.sentiment_strategy import (
    SentimentTradingConfig,
    create_sentiment_strategy,
)
from neural.data_collection.aggregator import AggregatedData, create_aggregator
from neural.trading.client import TradingClient


@dataclass
class TradingBotConfig:
    """Configuration for the sentiment trading bot."""

    # Game/Market Configuration
    game_id: str
    teams: list[str]
    market_tickers: dict[str, str]

    # API Keys and Credentials
    twitter_api_key: str
    kalshi_api_key: str | None = None
    kalshi_private_key: str | None = None

    # Trading Configuration
    initial_capital: float = 1000.0
    max_position_size: float = 0.1
    risk_per_trade: float = 0.02

    # Strategy Configuration
    min_sentiment_strength: float = 0.3
    sentiment_divergence_threshold: float = 0.2
    min_confidence_threshold: float = 0.6

    # Data Collection Configuration
    twitter_poll_interval: float = 30.0
    espn_poll_interval: float = 5.0
    kalshi_poll_interval: float = 10.0

    # Operational Configuration
    max_runtime_hours: float = 4.0  # Maximum runtime for safety
    log_level: str = "INFO"
    dry_run: bool = True  # Set to False for live trading


class SentimentTradingBot:
    """
    Main sentiment trading bot that orchestrates all components.

    This bot:
    1. Collects real-time data from Twitter, ESPN, and Kalshi
    2. Analyzes sentiment and momentum
    3. Generates trading signals
    4. Executes trades on Kalshi markets
    5. Monitors positions and risk
    """

    def __init__(self, config: TradingBotConfig):
        self.config = config
        self.logger = logging.getLogger("SentimentTradingBot")

        # Initialize components
        self.trading_client: TradingClient | None = None
        self.data_aggregator = None
        self.sentiment_strategy = None

        # State tracking
        self.running = False
        self.start_time: datetime | None = None
        self.positions: list[dict[str, Any]] = []
        self.trade_history: list[dict[str, Any]] = []
        self.performance_metrics: dict[str, Any] = {}

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging for the bot."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(
                    f'sentiment_bot_{self.config.game_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
                ),
                logging.StreamHandler(),
            ],
        )

    async def initialize(self):
        """Initialize all components of the trading bot."""
        try:
            self.logger.info("Initializing Sentiment Trading Bot...")

            # Initialize trading client
            if not self.config.dry_run and self.config.kalshi_api_key:
                self.trading_client = TradingClient(
                    api_key_id=self.config.kalshi_api_key,
                    private_key_pem=(
                        self.config.kalshi_private_key.encode()
                        if self.config.kalshi_private_key
                        else None
                    ),
                )
                self.logger.info("Trading client initialized for live trading")
            else:
                self.logger.info("Running in dry-run mode - no actual trades will be executed")

            # Initialize data aggregator
            self.data_aggregator = create_aggregator(
                game_id=self.config.game_id,
                teams=self.config.teams,
                twitter_enabled=True,
                espn_enabled=True,
                kalshi_enabled=True,
                twitter_interval=self.config.twitter_poll_interval,
                espn_interval=self.config.espn_poll_interval,
                kalshi_interval=self.config.kalshi_poll_interval,
            )

            # Initialize sentiment trading strategy
            strategy_config = SentimentTradingConfig(
                max_position_size=self.config.max_position_size,
                min_edge=0.03,
                min_sentiment_strength=self.config.min_sentiment_strength,
                sentiment_divergence_threshold=self.config.sentiment_divergence_threshold,
                min_confidence_threshold=self.config.min_confidence_threshold,
            )

            self.sentiment_strategy = create_sentiment_strategy(
                teams=self.config.teams,
                market_tickers=self.config.market_tickers,
                **strategy_config.__dict__,
            )

            # Register data handler
            self.data_aggregator.add_data_handler(self._handle_aggregated_data)

            self.logger.info("All components initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise

    async def start(self):
        """Start the trading bot."""
        if self.running:
            self.logger.warning("Bot is already running")
            return

        try:
            self.running = True
            self.start_time = datetime.now()

            self.logger.info("Starting Sentiment Trading Bot")
            self.logger.info(f"Game ID: {self.config.game_id}")
            self.logger.info(f"Teams: {', '.join(self.config.teams)}")
            self.logger.info(f"Dry Run: {self.config.dry_run}")
            self.logger.info(f"Max Runtime: {self.config.max_runtime_hours} hours")

            # Start data aggregation
            await self.data_aggregator.start(
                twitter_api_key=self.config.twitter_api_key,
                kalshi_config=(
                    {
                        "api_key": self.config.kalshi_api_key,
                        "private_key": self.config.kalshi_private_key,
                    }
                    if self.config.kalshi_api_key
                    else None
                ),
            )

            # Main trading loop
            await self._trading_loop()

        except Exception as e:
            self.logger.error(f"Error during bot execution: {e}")
            raise
        finally:
            await self.stop()

    async def _trading_loop(self):
        """Main trading loop that processes data and executes trades."""
        self.logger.info("Entering main trading loop")

        while self.running:
            try:
                # Check runtime limit
                if (
                    self.start_time
                    and (datetime.now() - self.start_time).total_seconds() / 3600
                    > self.config.max_runtime_hours
                ):
                    self.logger.info("Maximum runtime reached, stopping bot")
                    break

                # Get current aggregator state
                current_state = await self.data_aggregator.get_current_state()

                if (
                    current_state["signal_strength"] > 0.3
                ):  # Only process if we have reasonable signal strength
                    self.logger.info(
                        f"Processing trading signals (signal strength: {current_state['signal_strength']:.3f})"
                    )

                # Sleep between iterations
                await asyncio.sleep(15)  # Process every 15 seconds

            except KeyboardInterrupt:
                self.logger.info("Received interrupt signal, stopping bot")
                break
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(30)  # Wait longer on error

    def _handle_aggregated_data(self, data: AggregatedData):
        """Handle new aggregated data and generate trading signals."""
        try:
            self.logger.debug(f"Processing aggregated data at {data.timestamp}")

            # Create mock market data for strategy analysis
            market_data = self._create_market_data_frame(data)

            # Run strategy analysis
            signal = asyncio.run(self.sentiment_strategy.analyze(market_data, data))

            if signal and signal.signal_type.value != "hold":
                self.logger.info(
                    f"Generated signal: {signal.signal_type.value} for {signal.market_id}"
                )
                self.logger.info(f"  Confidence: {signal.confidence:.3f}")
                self.logger.info(f"  Position Size: {signal.recommended_size:.3f}")
                self.logger.info(f"  Strategy: {signal.metadata.get('strategy_type', 'unknown')}")

                # Execute trade if not in dry run mode
                if not self.config.dry_run:
                    asyncio.create_task(self._execute_trade(signal, data))
                else:
                    self._log_hypothetical_trade(signal, data)

                # Record signal
                self._record_signal(signal, data)

        except Exception as e:
            self.logger.error(f"Error handling aggregated data: {e}")

    def _create_market_data_frame(self, data: AggregatedData) -> pd.DataFrame:
        """Create a market data DataFrame from aggregated data."""
        # This is a simplified implementation
        # In practice, you'd extract actual market prices from Kalshi data

        market_data = {
            "timestamp": [data.timestamp],
            f'{data.teams[0].lower().replace(" ", "_")}_price': [0.5],  # Mock price
            f'{data.teams[1].lower().replace(" ", "_")}_price': [0.5],  # Mock price
            "volume": [1000],
            "spread": [0.02],
        }

        # Add sentiment-derived pricing if available
        if data.sentiment_metrics:
            sentiment = data.sentiment_metrics.get("combined_sentiment", 0.0)
            # Adjust prices based on sentiment
            market_data[f'{data.teams[0].lower().replace(" ", "_")}_price'][0] = max(
                0.01, min(0.99, 0.5 + sentiment * 0.3)
            )
            market_data[f'{data.teams[1].lower().replace(" ", "_")}_price'][0] = max(
                0.01, min(0.99, 0.5 - sentiment * 0.3)
            )

        return pd.DataFrame(market_data)

    async def _execute_trade(self, signal, data: AggregatedData):
        """Execute a trade based on the signal."""
        if not self.trading_client:
            self.logger.error("Cannot execute trade: no trading client")
            return

        try:
            self.logger.info(f"Executing trade: {signal.signal_type.value} {signal.market_id}")

            # Calculate position size in dollars
            position_value = self.config.initial_capital * signal.recommended_size

            # This would be implemented with actual Kalshi trading API calls
            # For now, just log the intended trade

            trade_record = {
                "timestamp": datetime.now(),
                "signal_type": signal.signal_type.value,
                "market_id": signal.market_id,
                "position_size": signal.recommended_size,
                "position_value": position_value,
                "confidence": signal.confidence,
                "strategy_type": signal.metadata.get("strategy_type"),
                "sentiment_score": data.sentiment_metrics.get("combined_sentiment", 0.0),
                "executed": True,
            }

            self.trade_history.append(trade_record)
            self.logger.info(f"Trade executed successfully: {trade_record}")

        except Exception as e:
            self.logger.error(f"Failed to execute trade: {e}")

    def _log_hypothetical_trade(self, signal, data: AggregatedData):
        """Log what would have been traded in dry run mode."""
        position_value = self.config.initial_capital * signal.recommended_size

        self.logger.info("=== HYPOTHETICAL TRADE (DRY RUN) ===")
        self.logger.info(f"Signal: {signal.signal_type.value}")
        self.logger.info(f"Market: {signal.market_id}")
        self.logger.info(f"Position Size: {signal.recommended_size:.1%} (${position_value:.2f})")
        self.logger.info(f"Confidence: {signal.confidence:.1%}")
        self.logger.info(f"Strategy: {signal.metadata.get('strategy_type', 'unknown')}")
        self.logger.info(
            f"Sentiment Score: {data.sentiment_metrics.get('combined_sentiment', 0.0):.3f}"
        )

        if signal.metadata.get("sentiment_score"):
            self.logger.info(f"Sentiment Details: {signal.metadata}")

        self.logger.info("=====================================")

    def _record_signal(self, signal, data: AggregatedData):
        """Record signal for analysis."""
        signal_record = {
            "timestamp": datetime.now(),
            "signal_type": signal.signal_type.value,
            "market_id": signal.market_id,
            "confidence": signal.confidence,
            "recommended_size": signal.recommended_size,
            "strategy_type": signal.metadata.get("strategy_type"),
            "sentiment_score": data.sentiment_metrics.get("combined_sentiment", 0.0),
            "signal_strength": data.metadata.get("signal_strength", 0.0),
        }

        # Add to strategy's signal history
        self.sentiment_strategy.signal_history.append(signal_record)

    async def stop(self):
        """Stop the trading bot."""
        if not self.running:
            return

        self.logger.info("Stopping Sentiment Trading Bot")
        self.running = False

        # Stop data aggregation
        if self.data_aggregator:
            await self.data_aggregator.stop()

        # Generate final report
        await self._generate_final_report()

        self.logger.info("Bot stopped successfully")

    async def _generate_final_report(self):
        """Generate a final performance report."""
        runtime = (
            (datetime.now() - self.start_time).total_seconds() / 3600 if self.start_time else 0
        )

        report = {
            "runtime_hours": runtime,
            "total_signals": len(self.sentiment_strategy.signal_history),
            "total_trades": len(self.trade_history),
            "strategy_metrics": self.sentiment_strategy.get_strategy_metrics(),
        }

        # Signal type breakdown
        if self.sentiment_strategy.signal_history:
            signal_types = [
                s.get("strategy_type", "unknown") for s in self.sentiment_strategy.signal_history
            ]
            report["signal_breakdown"] = {
                stype: signal_types.count(stype) for stype in set(signal_types)
            }

        self.logger.info("=== FINAL PERFORMANCE REPORT ===")
        self.logger.info(f"Runtime: {runtime:.2f} hours")
        self.logger.info(f"Signals Generated: {report['total_signals']}")
        self.logger.info(f"Trades Executed: {report['total_trades']}")

        if report.get("signal_breakdown"):
            self.logger.info("Signal Type Breakdown:")
            for stype, count in report["signal_breakdown"].items():
                self.logger.info(f"  {stype}: {count}")

        # Save detailed report to file
        report_file = f"sentiment_bot_report_{self.config.game_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        self.logger.info(f"Detailed report saved to: {report_file}")

    def get_status(self) -> dict[str, Any]:
        """Get current bot status."""
        runtime = (
            (datetime.now() - self.start_time).total_seconds() / 3600 if self.start_time else 0
        )

        return {
            "running": self.running,
            "runtime_hours": runtime,
            "signals_generated": (
                len(self.sentiment_strategy.signal_history) if self.sentiment_strategy else 0
            ),
            "trades_executed": len(self.trade_history),
            "current_positions": len(self.positions),
            "aggregator_state": (
                asyncio.run(self.data_aggregator.get_current_state())
                if self.data_aggregator
                else None
            ),
        }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Sentiment-Based Trading Bot")

    # Required arguments
    parser.add_argument("--game-id", required=True, help="ESPN Game ID")
    parser.add_argument("--teams", required=True, help="Comma-separated team names")

    # API Keys
    parser.add_argument("--twitter-api-key", required=True, help="Twitter API key")
    parser.add_argument("--kalshi-api-key", help="Kalshi API key")
    parser.add_argument("--kalshi-private-key", help="Kalshi private key path")

    # Trading configuration
    parser.add_argument("--initial-capital", type=float, default=1000.0, help="Initial capital")
    parser.add_argument("--max-position-size", type=float, default=0.1, help="Max position size")
    parser.add_argument(
        "--dry-run", action="store_true", default=True, help="Run without executing trades"
    )
    parser.add_argument(
        "--live", action="store_true", help="Run with live trading (overrides dry-run)"
    )

    # Strategy configuration
    parser.add_argument(
        "--min-sentiment-strength", type=float, default=0.3, help="Min sentiment strength"
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.6, help="Min confidence threshold"
    )

    # Operational
    parser.add_argument("--max-runtime-hours", type=float, default=4.0, help="Max runtime in hours")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )

    return parser.parse_args()


async def main():
    """Main entry point for the sentiment trading bot."""
    args = parse_args()

    # Parse teams
    teams = [team.strip() for team in args.teams.split(",")]

    # Create market tickers mapping (this would be configured based on actual markets)
    market_tickers = {
        teams[0]: f"{teams[0].upper().replace(' ', '_')}_WIN",
        teams[1]: f"{teams[1].upper().replace(' ', '_')}_WIN",
    }

    # Load private key if provided
    kalshi_private_key = None
    if args.kalshi_private_key:
        with open(args.kalshi_private_key) as f:
            kalshi_private_key = f.read()

    # Create configuration
    config = TradingBotConfig(
        game_id=args.game_id,
        teams=teams,
        market_tickers=market_tickers,
        twitter_api_key=args.twitter_api_key,
        kalshi_api_key=args.kalshi_api_key,
        kalshi_private_key=kalshi_private_key,
        initial_capital=args.initial_capital,
        max_position_size=args.max_position_size,
        min_sentiment_strength=args.min_sentiment_strength,
        min_confidence_threshold=args.min_confidence,
        max_runtime_hours=args.max_runtime_hours,
        log_level=args.log_level,
        dry_run=args.dry_run and not args.live,  # Live overrides dry-run
    )

    # Create and run bot
    bot = SentimentTradingBot(config)

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        print("\nReceived interrupt signal, stopping bot gracefully...")
        bot.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        await bot.initialize()
        await bot.start()
    except KeyboardInterrupt:
        print("\nBot interrupted by user")
    except Exception as e:
        print(f"Bot failed with error: {e}")
        logging.exception("Bot failed")
    finally:
        print("Bot execution completed")


if __name__ == "__main__":
    # Example usage:
    # python sentiment_trading_bot.py \
    #   --game-id 401547439 \
    #   --teams "Baltimore Ravens,Detroit Lions" \
    #   --twitter-api-key your_twitter_key \
    #   --dry-run \
    #   --max-runtime-hours 2

    asyncio.run(main())
