#!/usr/bin/env python3
"""
Ravens vs Lions Trading Algorithm Example

This example demonstrates building and testing trading algorithms for the
Ravens vs Lions NFL game using the Neural SDK. It compares two strategies:

1. Ravens Win Strategy: Bets on Baltimore Ravens victory
2. Lions Win Strategy: Bets on Detroit Lions victory

The algorithm includes:
- Real-time market data collection
- Strategy implementation with risk management
- Backtesting with historical performance analysis
- Comprehensive visualization of results
- Performance metrics comparison

Usage:
    python examples/ravens_lions_algorithm.py
"""

import asyncio
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

# Plotly imports for visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from neural.analysis.strategies.base import BaseStrategy, Signal, SignalType, StrategyConfig
from neural.data_collection.base import DataSourceConfig
from neural.data_collection.kalshi_historical import KalshiHistoricalDataSource
from neural.auth.env import get_api_key_id, get_private_key_material

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RavensWinStrategy(BaseStrategy):
    """
    Strategy that bets on Baltimore Ravens victory.

    This strategy implements mean reversion logic, betting against
    extreme price movements that suggest Ravens are over/under valued.
    """

    def __init__(self, divergence_threshold: float = 0.08, min_confidence: float = 0.65):
        print("DEBUG: RavensWinStrategy.__init__ - Starting initialization")
        print(f"DEBUG: RavensWinStrategy.__init__ - Creating config with max_position_size=0.15")
        config = StrategyConfig(
            max_position_size=0.15,  # 15% of capital max
            min_edge=0.05  # Minimum 5% edge required
        )
        print(f"DEBUG: RavensWinStrategy.__init__ - Calling super().__init__ with name='RavensWinStrategy'")
        super().__init__(name="RavensWinStrategy", config=config)

        # Strategy-specific parameters
        print(f"DEBUG: RavensWinStrategy.__init__ - Setting strategy parameters: divergence_threshold={divergence_threshold}, min_confidence={min_confidence}")
        self.divergence_threshold = divergence_threshold
        self.min_confidence = min_confidence
        self.ravens_ticker = "KXNFLGAME-24JAN28DETBAL-BAL"
        self.lions_ticker = "KXNFLGAME-24JAN28DETBAL-DET"
        print(f"DEBUG: RavensWinStrategy.__init__ - Set tickers: ravens={self.ravens_ticker}, lions={self.lions_ticker}")

    async def initialize(self) -> None:
        """Initialize the strategy."""
        pass

    async def analyze_market(
        self,
        market_id: str,
        market_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Any:  # Using Any to avoid import issues
        """Analyze market and return signal."""
        return await self.analyze(market_data)

    def get_required_data(self) -> List[str]:
        """Get required data fields."""
        return ['ravens_price', 'lions_price', 'ravens_volume', 'lions_volume', 'timestamp']

    async def analyze(self, market_data: Dict[str, Any]) -> Optional[Signal]:
        """Analyze market data and generate trading signals for Ravens."""
        print(f"DEBUG: RavensWinStrategy.analyze - Starting analysis with market_data keys: {list(market_data.keys())}")

        try:
            # Get Ravens market data
            ravens_price = market_data.get('ravens_price')
            lions_price = market_data.get('lions_price')
            ravens_volume = market_data.get('ravens_volume', 0)
            timestamp = market_data.get('timestamp', datetime.now())
            print(f"DEBUG: RavensWinStrategy.analyze - Extracted data: ravens_price={ravens_price}, lions_price={lions_price}, ravens_volume={ravens_volume}, timestamp={timestamp}")

            if ravens_price is None:
                print("DEBUG: RavensWinStrategy.analyze - Ravens price is None, returning None")
                return None

            # Calculate fair value (should sum to ~1.0)
            total_probability = ravens_price + lions_price if lions_price else ravens_price
            print(f"DEBUG: RavensWinStrategy.analyze - Calculated total_probability: {total_probability}")

            # Mean reversion logic: bet against extreme prices
            if ravens_price < 0.45:  # Ravens undervalued (adjusted threshold)
                print(f"DEBUG: RavensWinStrategy.analyze - Ravens price {ravens_price} < 0.45, calculating edge")
                edge = 0.50 - ravens_price  # Expected fair value vs current price
                confidence = min(0.85, 0.6 + (edge * 0.5) + (ravens_volume / 10000))
                print(f"DEBUG: RavensWinStrategy.analyze - Calculated edge={edge}, confidence={confidence}")

                if confidence >= self.min_confidence and edge >= self.config.min_edge:
                    print(f"DEBUG: RavensWinStrategy.analyze - Conditions met, creating signal")
                    position_size = min(self.config.max_position_size, edge * confidence)
                    print(f"DEBUG: RavensWinStrategy.analyze - position_size={position_size}")

                    signal = Signal(
                        signal_type=SignalType.BUY_YES,
                        market_id=self.ravens_ticker,
                        recommended_size=position_size,
                        confidence=confidence,
                        edge=edge,
                        expected_value=edge * position_size,
                        max_contracts=int(position_size * 1000),  # Assume $1000 per contract
                        stop_loss_price=ravens_price * 0.75,
                        take_profit_price=min(0.65, ravens_price * 1.8),
                        metadata={
                            'strategy': self.name,
                            'edge': edge,
                            'total_probability': total_probability,
                            'entry_price': ravens_price,  # Store in metadata
                            'reasoning': f'Ravens undervalued at {ravens_price:.3f}, fair value ~0.65'
                        },
                        timestamp=timestamp
                    )
                    print(f"DEBUG: RavensWinStrategy.analyze - Created signal: {signal.signal_type} for {signal.market_id}")
                    return signal
                else:
                    print(f"DEBUG: RavensWinStrategy.analyze - Conditions not met: confidence {confidence} >= {self.min_confidence}? {confidence >= self.min_confidence}, edge {edge} >= {self.config.min_edge}? {edge >= self.config.min_edge}")

            elif ravens_price > 0.75:  # Ravens overvalued, but we still bet on them
                print(f"DEBUG: RavensWinStrategy.analyze - Ravens price {ravens_price} > 0.75, strategy only bets on win so no action")
                # This strategy only bets on Ravens win, so we don't short here
                pass

            print("DEBUG: RavensWinStrategy.analyze - No signal generated, returning None")
            return None

        except Exception as e:
            print(f"DEBUG: RavensWinStrategy.analyze - Exception occurred: {e}")
            logger.error(f"Error in {self.name} analysis: {e}")
            return None


class LionsWinStrategy(BaseStrategy):
    """
    Strategy that bets on Detroit Lions victory.

    This strategy implements momentum-based logic, betting with
    price movements that suggest Lions are gaining momentum.
    """

    def __init__(self, momentum_threshold: float = 0.05, min_confidence: float = 0.65):
        print("DEBUG: LionsWinStrategy.__init__ - Starting initialization")
        config = StrategyConfig(
            max_position_size=0.15,
            min_edge=0.05  # Minimum 5% edge required
        )
        super().__init__(name="LionsWinStrategy", config=config)

        # Strategy-specific parameters
        self.momentum_threshold = momentum_threshold
        self.min_confidence = min_confidence
        self.ravens_ticker = "KXNFLGAME-24JAN28DETBAL-BAL"
        self.lions_ticker = "KXNFLGAME-24JAN28DETBAL-DET"

        # Track price history for momentum calculation
        self.price_history: List[Dict[str, Any]] = []
        self.max_history = 10  # Keep last 10 price points

    async def initialize(self) -> None:
        """Initialize the strategy."""
        pass

    async def analyze_market(
        self,
        market_id: str,
        market_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Any:  # Using Any to avoid import issues
        """Analyze market and return signal."""
        return await self.analyze(market_data)

    def get_required_data(self) -> List[str]:
        """Get required data fields."""
        return ['ravens_price', 'lions_price', 'ravens_volume', 'lions_volume', 'timestamp']

    async def analyze(self, market_data: Dict[str, Any]) -> Optional[Signal]:
        """Analyze market data and generate trading signals for Lions."""

        try:
            lions_price = market_data.get('lions_price')
            ravens_price = market_data.get('ravens_price')
            lions_volume = market_data.get('lions_volume', 0)
            timestamp = market_data.get('timestamp', datetime.now())

            if lions_price is None:
                return None

            # Add to price history
            self.price_history.append({
                'price': lions_price,
                'timestamp': timestamp,
                'volume': lions_volume
            })

            # Keep only recent history
            if len(self.price_history) > self.max_history:
                self.price_history = self.price_history[-self.max_history:]

            # Need at least 3 data points for momentum
            if len(self.price_history) < 3:
                return None

            # Calculate momentum (price change over last 3 points)
            recent_prices = [p['price'] for p in self.price_history[-3:]]
            momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]

            # Calculate fair value
            total_probability = lions_price + ravens_price if ravens_price else lions_price

            # Momentum strategy: bet with upward price movement
            if momentum > self.momentum_threshold and lions_price < 0.65:
                edge = momentum * 0.8  # Convert momentum to edge estimate
                confidence = min(0.85, 0.6 + abs(momentum) + (lions_volume / 15000))

                if confidence >= self.min_confidence and edge >= self.config.min_edge:
                    position_size = min(self.config.max_position_size, edge * confidence)

                    return Signal(
                        signal_type=SignalType.BUY_YES,
                        market_id=self.lions_ticker,
                        recommended_size=position_size,
                        confidence=confidence,
                        edge=edge,
                        expected_value=edge * position_size,
                        max_contracts=int(position_size * 1000),  # Assume $1000 per contract
                        stop_loss_price=lions_price * 0.8,
                        take_profit_price=min(0.75, lions_price * 1.6),
                        metadata={
                            'strategy': self.name,
                            'momentum': momentum,
                            'edge': edge,
                            'total_probability': total_probability,
                            'entry_price': lions_price,  # Store in metadata
                            'reasoning': f'Lions momentum {momentum:.1%}, price {lions_price:.3f}'
                        },
                        timestamp=timestamp
                    )

            return None

        except Exception as e:
            logger.error(f"Error in {self.name} analysis: {e}")
            return None


class RavensLionsTradingAlgorithm:
    """
    Main algorithm class that orchestrates data collection, strategy execution,
    backtesting, and performance analysis for the Ravens vs Lions game.
    """

    def __init__(self):
        print("DEBUG: RavensLionsTradingAlgorithm.__init__ - Starting initialization")
        # Use a historical game that actually has data (Jan 2024 game)
        self.event_ticker = os.getenv("KX_EVENT_TICKER", "KXNFLGAME-24JAN28DETBAL")
        self.ravens_ticker = os.getenv("KX_RAVENS_TICKER", "KXNFLGAME-24JAN28DETBAL-BAL")
        self.lions_ticker = os.getenv("KX_LIONS_TICKER", "KXNFLGAME-24JAN28DETBAL-DET")
        print(f"DEBUG: RavensLionsTradingAlgorithm.__init__ - Set tickers: event={self.event_ticker}, ravens={self.ravens_ticker}, lions={self.lions_ticker}")

        logger.info(f"Using event ticker: {self.event_ticker}")

        # Initialize strategies
        print("DEBUG: RavensLionsTradingAlgorithm.__init__ - Initializing strategies")
        self.strategies = {
            'ravens': RavensWinStrategy(),
            'lions': LionsWinStrategy()
        }
        print(f"DEBUG: RavensLionsTradingAlgorithm.__init__ - Strategies initialized: {list(self.strategies.keys())}")

        # Data storage
        self.market_data_history: List[Dict[str, Any]] = []
        self.signals_history: List[Signal] = []

        # Simple configuration
        self.initial_capital = 10000.0
        print(f"DEBUG: RavensLionsTradingAlgorithm.__init__ - Set initial capital: {self.initial_capital}")

        logger.info("Initialized Ravens vs Lions Trading Algorithm")
        print("DEBUG: RavensLionsTradingAlgorithm.__init__ - Initialization complete")

    async def collect_market_data(self) -> pd.DataFrame:
        """
        Collect historical market data for the Ravens vs Lions game (Sept 25, 2025)
        using the new KalshiHistoricalDataSource.

        Returns:
            DataFrame with historical market data
        """
        print("DEBUG: RavensLionsTradingAlgorithm.collect_market_data - Starting data collection")
        logger.info("Collecting historical market data for Ravens vs Lions game...")

        # Use a historical timeframe: January 2024 playoffs
        # Start: 2024-01-20 00:00:00 UTC, End: 2024-01-29 00:00:00 UTC
        start_ts = int(os.getenv("KX_START_TS", "1705708800"))  # Jan 20, 2024
        end_ts = int(os.getenv("KX_END_TS", "1706486400"))    # Jan 29, 2024
        print(f"DEBUG: RavensLionsTradingAlgorithm.collect_market_data - Timeframe: start_ts={start_ts}, end_ts={end_ts}")

        collected_data = []
        print("DEBUG: RavensLionsTradingAlgorithm.collect_market_data - Initialized collected_data list")

        try:
            print("DEBUG: RavensLionsTradingAlgorithm.collect_market_data - Loading Kalshi credentials")
            # Ensure credentials are loaded before adapter initialization
            api_key = get_api_key_id()
            private_key = get_private_key_material()
            print(f"DEBUG: RavensLionsTradingAlgorithm.collect_market_data - Credentials loaded (API Key: {api_key[:10] if api_key else 'None'}...), creating config")
            # Initialize the historical data source
            config = DataSourceConfig(name="ravens_lions_historical")
            print("DEBUG: RavensLionsTradingAlgorithm.collect_market_data - Creating historical data source")
            historical_source = KalshiHistoricalDataSource(config)
            print(f"DEBUG: RavensLionsTradingAlgorithm.collect_market_data - Historical source created: {historical_source}")

            # Collect trade data for both markets
            print(f"DEBUG: RavensLionsTradingAlgorithm.collect_market_data - Collecting Ravens trade data for ticker: {self.ravens_ticker}")
            logger.info("📊 Collecting Ravens trade data...")
            ravens_trades = await historical_source.collect_trades(
                ticker=self.ravens_ticker,
                start_ts=start_ts,
                end_ts=end_ts
            )
            print(f"DEBUG: RavensLionsTradingAlgorithm.collect_market_data - Ravens trades collected: {len(ravens_trades) if hasattr(ravens_trades, '__len__') else 'unknown'}")

            print(f"DEBUG: RavensLionsTradingAlgorithm.collect_market_data - Collecting Lions trade data for ticker: {self.lions_ticker}")
            logger.info("📊 Collecting Lions trade data...")
            lions_trades = await historical_source.collect_trades(
                ticker=self.lions_ticker,
                start_ts=start_ts,
                end_ts=end_ts
            )
            print(f"DEBUG: RavensLionsTradingAlgorithm.collect_market_data - Lions trades collected: {len(lions_trades) if hasattr(lions_trades, '__len__') else 'unknown'}")

            # Merge the trade data by timestamp
            collected_data = []
            if not ravens_trades.empty and not lions_trades.empty:
                # Create a mapping of timestamp to lions data
                lions_data_by_time = {}
                for _, row in lions_trades.iterrows():
                    ts = int(row['created_time'].timestamp())
                    lions_data_by_time[ts] = {
                        'lions_price': row.get('yes_price', 0) / 100,  # Convert from cents to probability
                        'lions_volume': row.get('count', 0),
                    }

                # Process Ravens data and merge with Lions data
                for _, row in ravens_trades.iterrows():
                    ts = int(row['created_time'].timestamp())

                    market_snapshot = {
                        'timestamp': row['created_time'],
                        'ravens_price': row.get('yes_price', 0) / 100,  # Convert from cents to probability
                        'ravens_volume': row.get('count', 0),
                        'lions_price': None,  # Will fill from Lions data
                        'lions_volume': None,
                        'total_probability': None,
                    }

                    # Merge Lions data if available
                    if ts in lions_data_by_time:
                        lions_data = lions_data_by_time[ts]
                        market_snapshot.update(lions_data)
                        # Calculate total probability
                        if market_snapshot['ravens_price'] is not None and market_snapshot['lions_price'] is not None:
                            market_snapshot['total_probability'] = market_snapshot['ravens_price'] + market_snapshot['lions_price']

                    collected_data.append(market_snapshot)

                # Filter out incomplete data points
                collected_data = [d for d in collected_data if d['ravens_price'] is not None and d['lions_price'] is not None]

            elif not ravens_trades.empty:
                # Only Ravens data available
                for _, row in ravens_trades.iterrows():
                    market_snapshot = {
                        'timestamp': row['created_time'],
                        'ravens_price': row.get('yes_price', 0) / 100,
                        'ravens_volume': row.get('count', 0),
                        'lions_price': None,
                        'lions_volume': None,
                        'total_probability': None,
                    }
                    collected_data.append(market_snapshot)

            # Store in history
            self.market_data_history.extend(collected_data)

            logger.info(f"📊 Collected {len(collected_data)} historical data points")
            if collected_data:
                logger.info(".3f"
                          ".3f")

            # Generate signals from strategies for historical data
            for snapshot in collected_data:
                for strategy_name, strategy in self.strategies.items():
                    signal = await strategy.analyze(snapshot)
                    if signal:
                        self.signals_history.append(signal)
                        logger.info(f"📊 {strategy_name.upper()}: {signal.type.value} {signal.ticker} "
                                  f"Size: {signal.size:.1%} Confidence: {signal.confidence:.1%}")

        except Exception as e:
            logger.error(f"Error collecting historical market data: {e}")
            raise  # Re-raise the exception instead of falling back to synthetic data

        # Convert to DataFrame
        df = pd.DataFrame(collected_data)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            # Sort by timestamp
            df = df.sort_index()

        logger.info(f"✅ Collected {len(df)} historical market data points")
        return df



    def _prepare_backtest_data(self, market_data: pd.DataFrame, strategy_name: str) -> pd.DataFrame:
        """Prepare market data for backtesting simulation."""
        # This is a simplified preparation - in production you'd use actual historical data
        # For this demo, we'll simulate price movements based on the collected data

        backtest_data = []

        for timestamp, row in market_data.iterrows():
            if strategy_name == 'ravens':
                price = row['ravens_price']
                volume = row['ravens_volume']
            else:  # lions
                price = row['lions_price']
                volume = row['lions_volume']

            backtest_data.append({
                'timestamp': timestamp,
                'market_id': self.ravens_ticker if strategy_name == 'ravens' else self.lions_ticker,
                'last': price,
                'bid': price * 0.98,  # Simulate bid slightly below ask
                'ask': price,
                'volume': volume or 1000
            })

        return pd.DataFrame(backtest_data)

    async def run_real_time_analysis(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run strategies on collected real-time market data.

        Args:
            market_data: Real market data from collect_market_data()

        Returns:
            Analysis results with strategy performance
        """
        logger.info("Running real-time analysis on collected market data...")

        results = {}

        for strategy_name, strategy in self.strategies.items():
            logger.info(f"Analyzing {strategy_name} strategy on real data...")

            # Reset strategy capital for this analysis
            strategy.capital = self.initial_capital

            trades = []

            for timestamp, row in market_data.iterrows():
                # Create market snapshot for strategy
                market_snapshot = {
                    'ravens_price': row['ravens_price'],
                    'lions_price': row['lions_price'],
                    'ravens_volume': row['ravens_volume'],
                    'lions_volume': row['lions_volume'],
                    'timestamp': timestamp
                }

                # Get signal from strategy
                signal = await strategy.analyze(market_snapshot)

                if signal:
                    # Simulate trade execution
                    entry_price = signal.metadata.get('entry_price') or market_snapshot[f'{strategy_name}_price']

                    # Simplified P&L calculation (in real trading, would track actual fills)
                    if signal.signal_type == SignalType.BUY_YES:
                        pnl = signal.recommended_size * (entry_price * 1.05 - entry_price)  # Assume 5% gain
                    else:
                        pnl = signal.recommended_size * (entry_price - entry_price * 1.03)  # Assume 3% loss

                    trade = {
                        'timestamp': timestamp,
                        'signal': signal,
                        'entry_price': entry_price,
                        'pnl': pnl,
                        'confidence': signal.confidence
                    }

                    trades.append(trade)
                    # Note: record_trade method not implemented in BaseStrategy
                    # Performance tracking handled manually in results

            results[strategy_name] = {
                'final_capital': strategy.capital,
                'total_trades': len(trades),
                'winning_trades': len([t for t in trades if t['pnl'] > 0]),
                'total_pnl': sum(t['pnl'] for t in trades),
                'win_rate': len([t for t in trades if t['pnl'] > 0]) / max(len(trades), 1),
                'trades': trades
            }

            logger.info(f"✅ {strategy_name}: ${strategy.capital:.2f} final, {len(trades)} trades")

        return results

    def create_performance_analysis(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create comprehensive performance analysis and visualizations.

        Args:
            backtest_results: Results from backtesting

        Returns:
            Dictionary with analysis results and chart paths
        """
        logger.info("Creating performance analysis...")

        analysis = {
            'summary': {},
            'charts': {},
            'metrics_comparison': {},
            'recommendations': []
        }

        # Extract performance data
        ravens_result = backtest_results.get('ravens', {}).get('result')
        lions_result = backtest_results.get('lions', {}).get('result')

        if ravens_result and lions_result:
            # Summary statistics
            analysis['summary'] = {
                'ravens_final_capital': ravens_result.final_capital,
                'lions_final_capital': lions_result.final_capital,
                'ravens_total_return': ravens_result.total_return,
                'lions_total_return': lions_result.total_return,
                'ravens_win_rate': ravens_result.win_rate,
                'lions_win_rate': lions_result.win_rate,
                'ravens_total_trades': ravens_result.total_trades,
                'lions_total_trades': lions_result.total_trades,
                'better_strategy': 'ravens' if ravens_result.final_capital > lions_result.final_capital else 'lions'
            }

            # Create equity curve comparison chart
            ravens_equity = ravens_result.equity_curve
            lions_equity = lions_result.equity_curve

            # Create comparison chart
            fig = self._create_strategy_comparison_chart(ravens_equity, lions_equity)
            analysis['charts']['equity_comparison'] = 'ravens_lions_equity_comparison.html'
            fig.write_html(analysis['charts']['equity_comparison'])

            # Create performance metrics comparison
            analysis['metrics_comparison'] = {
                'ravens': {
                    'sharpe_ratio': ravens_result.sharpe_ratio,
                    'max_drawdown': ravens_result.max_drawdown,
                    'win_rate': ravens_result.win_rate,
                    'profit_factor': ravens_result.profit_factor
                },
                'lions': {
                    'sharpe_ratio': lions_result.sharpe_ratio,
                    'max_drawdown': lions_result.max_drawdown,
                    'win_rate': lions_result.win_rate,
                    'profit_factor': lions_result.profit_factor
                }
            }

            # Generate recommendations
            analysis['recommendations'] = self._generate_recommendations(analysis)

        return analysis

    def _create_strategy_comparison_chart(self, ravens_equity: pd.Series, lions_equity: pd.Series):
        """Create equity curve comparison chart."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=["Strategy Equity Curves", "Relative Performance"],
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )

        # Equity curves
        fig.add_trace(
            go.Scatter(
                x=ravens_equity.index,
                y=ravens_equity.values,
                name="Ravens Strategy",
                line=dict(color='#2E8B57', width=3),
                hovertemplate='<b>Ravens Strategy</b><br>Date: %{x}<br>Equity: $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=lions_equity.index,
                y=lions_equity.values,
                name="Lions Strategy",
                line=dict(color='#4169E1', width=3),
                hovertemplate='<b>Lions Strategy</b><br>Date: %{x}<br>Equity: $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )

        # Relative performance (Ravens / Lions)
        if len(ravens_equity) == len(lions_equity):
            relative_perf = ravens_equity.values / lions_equity.values.astype(float)
            fig.add_trace(
                go.Scatter(
                    x=ravens_equity.index,
                    y=relative_perf,
                    name="Ravens/Lions Ratio",
                    line=dict(color='#FF6347', width=2, dash='dot'),
                    hovertemplate='<b>Relative Performance</b><br>Date: %{x}<br>Ratio: %{y:.3f}<extra></extra>'
                ),
                row=2, col=1
            )

            # Add reference line at 1.0
            fig.add_hline(y=1.0, line_dash="solid", line_color="gray", opacity=0.5)

        fig.update_layout(
            title="Ravens vs Lions Strategy Performance Comparison",
            height=800,
            showlegend=True
        )

        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Ravens/Lions Ratio", row=2, col=1)
        fig.update_xaxes(title_text="Time", row=2, col=1)

        return fig

    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate trading recommendations based on analysis."""
        recommendations = []

        summary = analysis.get('summary', {})
        better_strategy = summary.get('better_strategy')

        if better_strategy:
            recommendations.append(f"🎯 **Primary Recommendation**: Use the {better_strategy.upper()} strategy "
                                f"(outperformed by ${(summary[f'{better_strategy}_final_capital'] - summary[f'other_final_capital'.replace('other', 'ravens' if better_strategy == 'lions' else 'lions')]):.2f})")

        # Risk-based recommendations
        metrics = analysis.get('metrics_comparison', {})
        ravens_metrics = metrics.get('ravens', {})
        lions_metrics = metrics.get('lions', {})

        if ravens_metrics.get('max_drawdown', 0) < lions_metrics.get('max_drawdown', 0):
            recommendations.append("🛡️ **Risk Management**: Ravens strategy showed lower maximum drawdown")
        elif lions_metrics.get('max_drawdown', 0) < ravens_metrics.get('max_drawdown', 0):
            recommendations.append("🛡️ **Risk Management**: Lions strategy showed lower maximum drawdown")

        # Sharpe ratio comparison
        if ravens_metrics.get('sharpe_ratio', 0) > lions_metrics.get('sharpe_ratio', 0):
            recommendations.append("📊 **Risk-Adjusted Returns**: Ravens strategy has better Sharpe ratio")
        elif lions_metrics.get('sharpe_ratio', 0) > ravens_metrics.get('sharpe_ratio', 0):
            recommendations.append("📊 **Risk-Adjusted Returns**: Lions strategy has better Sharpe ratio")

        return recommendations

    def print_results_summary(self, analysis: Dict[str, Any]):
        """Print comprehensive results summary."""
        print("\n" + "="*80)
        print("🏈 RAVENS VS LIONS TRADING ALGORITHM RESULTS")
        print("="*80)

        summary = analysis.get('summary', {})
        if summary:
            print("\n💰 FINAL RESULTS:")
            print(".2f")
            print(".2f")
            print(".1%")
            print(".1%")
            print(f"   Ravens Total Trades: {summary.get('ravens_total_trades', 0)}")
            print(f"   Lions Total Trades: {summary.get('lions_total_trades', 0)}")

            better = summary.get('better_strategy', 'unknown')
            print(f"\n🎯 BETTER STRATEGY: {better.upper()}")

        metrics = analysis.get('metrics_comparison', {})
        if metrics:
            print("\n📊 PERFORMANCE METRICS:")
            print("   Ravens Strategy:")
            ravens = metrics.get('ravens', {})
            print(".2f")
            print(".1%")
            print(".2f")
            print(".2f")
            print("   Lions Strategy:")
            lions = metrics.get('lions', {})
            print(".2f")
            print(".1%")
            print(".2f")
            print(".2f")
        recommendations = analysis.get('recommendations', [])
        if recommendations:
            print("\n🎯 RECOMMENDATIONS:")
            for rec in recommendations:
                print(f"   • {rec}")

        charts = analysis.get('charts', {})
        if charts:
            print("\n📈 CHARTS GENERATED:")
            for chart_name, chart_path in charts.items():
                print(f"   • {chart_name}: {chart_path}")

        print("\n" + "="*80)


async def main():
    """Main execution function."""
    print("DEBUG: main() - Starting main function")
    print("🏈 Starting Ravens vs Lions Trading Algorithm")
    print("This will collect REAL Kalshi market data and analyze strategies")

    # Initialize algorithm
    print("DEBUG: main() - Initializing algorithm")
    algorithm = RavensLionsTradingAlgorithm()
    print("DEBUG: main() - Algorithm initialized successfully")

    try:
        # Phase 1: Collect REAL market data from Kalshi API
        print("DEBUG: main() - Starting Phase 1: Data collection")
        print("\n📊 Phase 1: Collecting Historical Market Data...")
        market_data = await algorithm.collect_market_data()
        print(f"DEBUG: main() - Market data collected, shape: {market_data.shape if hasattr(market_data, 'shape') else 'unknown'}")

        if market_data.empty:
            print("DEBUG: main() - Market data is empty, exiting")
            print("❌ No market data collected. Check Kalshi API credentials.")
            return

        print("DEBUG: main() - Market data is not empty, proceeding to Phase 2")
        # Phase 2: Run strategies on real collected data
        print("\n🔬 Phase 2: Running Strategy Analysis on Real Data...")
        analysis_results = await algorithm.run_real_time_analysis(market_data)
        print(f"DEBUG: main() - Analysis results: {analysis_results.keys() if hasattr(analysis_results, 'keys') else 'unknown'}")

        # Phase 3: Create performance analysis
        print("DEBUG: main() - Starting Phase 3: Performance analysis")
        print("\n📈 Phase 3: Creating Performance Analysis...")
        analysis = algorithm.create_performance_analysis(analysis_results)
        print(f"DEBUG: main() - Performance analysis created: {analysis.keys() if hasattr(analysis, 'keys') else 'unknown'}")

        # Phase 4: Print results
        print("DEBUG: main() - Starting Phase 4: Printing results")
        algorithm.print_results_summary(analysis)

        print("DEBUG: main() - All phases completed successfully")
        print("\n✅ Algorithm completed with REAL Kalshi market data!")
        print(f"📊 Analyzed {len(market_data)} real market data points")

    except Exception as e:
        print(f"DEBUG: main() - Exception caught: {e}")
        logger.error(f"Error: {e}")
        print(f"❌ Failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())