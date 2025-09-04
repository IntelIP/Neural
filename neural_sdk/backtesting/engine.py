"""
Backtesting Engine for Neural SDK Trading Strategies

Event-driven backtesting with realistic order execution and portfolio simulation.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from .data_loader import DataLoader
from .metrics import PerformanceMetrics
from .portfolio import Portfolio, Trade

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting engine."""

    start_date: str
    end_date: str
    initial_capital: float = 10000.0
    commission: float = 0.02  # Default 2% commission
    slippage: float = 0.01  # 1% slippage estimate
    data_frequency: str = "1min"
    max_positions: int = 20
    position_size_limit: float = 0.05  # Max 5% per position
    stop_loss: Optional[float] = None  # Global stop loss
    take_profit: Optional[float] = None  # Global take profit

    # Risk management
    daily_loss_limit: float = 0.20  # Stop at 20% daily loss
    max_drawdown_limit: float = 0.30  # Stop at 30% drawdown

    # Market impact simulation
    liquidity_adjustment: bool = True
    market_hours_only: bool = False

    def __post_init__(self):
        """Validate configuration."""
        if pd.to_datetime(self.start_date) >= pd.to_datetime(self.end_date):
            raise ValueError("start_date must be before end_date")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if not 0 <= self.commission <= 1:
            raise ValueError("commission must be between 0 and 1")


@dataclass
class BacktestResults:
    """Results from backtesting run."""

    portfolio_value: pd.Series
    trades: List[Trade]
    positions: pd.DataFrame
    daily_returns: pd.Series
    metrics: Dict[str, float]
    config: BacktestConfig
    start_time: datetime
    end_time: datetime

    @property
    def duration(self) -> timedelta:
        """Duration of backtest."""
        return self.end_time - self.start_time

    @property
    def total_return(self) -> float:
        """Total return percentage."""
        return (self.portfolio_value.iloc[-1] / self.portfolio_value.iloc[0] - 1) * 100

    @property
    def num_trades(self) -> int:
        """Total number of trades."""
        return len(self.trades)

    def plot(self, save_path: Optional[str] = None):
        """Plot backtest results."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # Portfolio value over time
            axes[0, 0].plot(self.portfolio_value.index, self.portfolio_value.values)
            axes[0, 0].set_title("Portfolio Value Over Time")
            axes[0, 0].set_ylabel("Value ($)")

            # Daily returns distribution
            axes[0, 1].hist(self.daily_returns.dropna(), bins=50, alpha=0.7)
            axes[0, 1].set_title("Daily Returns Distribution")
            axes[0, 1].set_xlabel("Daily Return (%)")

            # Cumulative returns
            cum_returns = self.daily_returns.cumsum()
            axes[1, 0].plot(cum_returns.index, cum_returns.values)
            axes[1, 0].set_title("Cumulative Returns")
            axes[1, 0].set_ylabel("Cumulative Return (%)")

            # Drawdown
            rolling_max = self.portfolio_value.expanding().max()
            drawdown = (self.portfolio_value - rolling_max) / rolling_max * 100
            axes[1, 1].fill_between(drawdown.index, drawdown.values, 0, alpha=0.3)
            axes[1, 1].set_title("Drawdown")
            axes[1, 1].set_ylabel("Drawdown (%)")

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
            else:
                plt.show()

        except ImportError:
            logger.warning("Matplotlib not available. Cannot plot results.")

    def export_report(self, path: str):
        """Export detailed HTML report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Neural SDK Backtest Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .metric {{ margin: 10px 0; }}
                .highlight {{ color: #007bff; font-weight: bold; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Neural SDK Backtest Report</h1>
            
            <h2>Summary</h2>
            <div class="metric">Period: <span class="highlight">{self.config.start_date} to {self.config.end_date}</span></div>
            <div class="metric">Initial Capital: <span class="highlight">${self.config.initial_capital:,.2f}</span></div>
            <div class="metric">Final Value: <span class="highlight">${self.portfolio_value.iloc[-1]:,.2f}</span></div>
            <div class="metric">Total Return: <span class="highlight">{self.total_return:.2f}%</span></div>
            <div class="metric">Total Trades: <span class="highlight">{self.num_trades}</span></div>
            
            <h2>Performance Metrics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
        """

        for metric, value in self.metrics.items():
            if isinstance(value, float):
                html += f"<tr><td>{metric.replace('_', ' ').title()}</td><td>{value:.4f}</td></tr>"
            else:
                html += f"<tr><td>{metric.replace('_', ' ').title()}</td><td>{value}</td></tr>"

        html += """
            </table>
            
            <h2>Trade History</h2>
            <table>
                <tr>
                    <th>Date</th>
                    <th>Symbol</th>
                    <th>Action</th>
                    <th>Quantity</th>
                    <th>Price</th>
                    <th>Value</th>
                    <th>P&L</th>
                </tr>
        """

        for trade in self.trades[-50:]:  # Last 50 trades
            pnl = trade.pnl if trade.pnl else 0
            html += f"""
                <tr>
                    <td>{trade.timestamp.strftime('%Y-%m-%d %H:%M')}</td>
                    <td>{trade.symbol}</td>
                    <td>{trade.action}</td>
                    <td>{trade.quantity}</td>
                    <td>${trade.price:.2f}</td>
                    <td>${trade.value:.2f}</td>
                    <td>${pnl:.2f}</td>
                </tr>
            """

        html += """
            </table>
        </body>
        </html>
        """

        with open(path, "w") as f:
            f.write(html)


class BacktestEngine:
    """
    High-performance backtesting engine for prediction market trading strategies.

    Features:
    - Event-driven simulation
    - Realistic order execution
    - Portfolio tracking
    - Risk management
    - Performance analytics
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.portfolio = Portfolio(config.initial_capital)
        self.data_loader = DataLoader()
        self.strategy = None
        self.data = None
        self.current_time = None
        self.results = None

        # Performance tracking
        self.daily_values = []
        self.trades = []

        # Risk management state
        self.daily_start_value = config.initial_capital
        self.max_portfolio_value = config.initial_capital

        logger.info(
            f"Initialized backtest engine: {config.start_date} to {config.end_date}"
        )

    def add_strategy(
        self, strategy: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]
    ):
        """
        Add trading strategy to backtest.

        Args:
            strategy: Function that takes market data and returns trading signals
                     Should return dict with keys: action, size, market, price (optional)
        """
        self.strategy = strategy
        logger.info("Added trading strategy")

    def load_data(self, source: str, **kwargs) -> pd.DataFrame:
        """
        Load historical data from various sources.

        Args:
            source: Data source type ('file', 's3', 'postgres', 'api')
            **kwargs: Source-specific parameters

        Returns:
            DataFrame with OHLCV data
        """
        self.data = self.data_loader.load(source, **kwargs)

        # Validate data
        required_columns = ["timestamp", "symbol", "price", "volume"]
        missing_cols = [col for col in required_columns if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Data missing required columns: {missing_cols}")

        # Filter by date range
        self.data["timestamp"] = pd.to_datetime(self.data["timestamp"])
        mask = (self.data["timestamp"] >= self.config.start_date) & (
            self.data["timestamp"] <= self.config.end_date
        )
        self.data = self.data[mask].sort_values("timestamp")

        logger.info(f"Loaded {len(self.data)} data points from {source}")
        return self.data

    def run(self) -> BacktestResults:
        """
        Execute backtest and return results.

        Returns:
            BacktestResults object with portfolio performance and trades
        """
        if self.strategy is None:
            raise ValueError("No strategy added. Use add_strategy() first.")
        if self.data is None or len(self.data) == 0:
            raise ValueError("No data loaded. Use load_data() first.")

        logger.info("Starting backtest execution...")
        start_time = datetime.now()

        # Group data by timestamp for event-driven simulation
        for timestamp, group in self.data.groupby("timestamp"):
            self.current_time = timestamp

            # Check risk limits before processing
            if self._should_stop_trading():
                logger.warning(f"Risk limit hit at {timestamp}. Stopping backtest.")
                break

            # Process market data for this timestamp
            market_data = self._prepare_market_data(group)

            # Generate trading signals
            signals = self.strategy(market_data)
            if signals:
                self._execute_signals(signals, market_data)

            # Update daily tracking
            if self._is_new_day(timestamp):
                self._update_daily_metrics()

            # Record portfolio value
            current_value = self.portfolio.total_value(market_data)
            self.daily_values.append(
                {"timestamp": timestamp, "portfolio_value": current_value}
            )

        end_time = datetime.now()

        # Calculate final metrics
        metrics = self._calculate_metrics()

        # Create results object
        portfolio_df = pd.DataFrame(self.daily_values)
        portfolio_df.set_index("timestamp", inplace=True)

        daily_returns = portfolio_df["portfolio_value"].pct_change() * 100

        self.results = BacktestResults(
            portfolio_value=portfolio_df["portfolio_value"],
            trades=self.trades,
            positions=self.portfolio.positions_df(),
            daily_returns=daily_returns,
            metrics=metrics,
            config=self.config,
            start_time=start_time,
            end_time=end_time,
        )

        logger.info(f"Backtest completed in {end_time - start_time}")
        logger.info(
            f"Final portfolio value: ${self.results.portfolio_value.iloc[-1]:,.2f}"
        )
        logger.info(f"Total return: {self.results.total_return:.2f}%")

        return self.results

    def _prepare_market_data(self, group: pd.DataFrame) -> Dict[str, Any]:
        """Prepare market data for strategy."""
        return {
            "timestamp": self.current_time,
            "data": group.to_dict("records"),
            "symbols": group["symbol"].unique().tolist(),
            "prices": dict(zip(group["symbol"], group["price"])),
            "volumes": dict(zip(group["symbol"], group["volume"])),
            "portfolio": self.portfolio,
        }

    def _execute_signals(self, signals: Dict[str, Any], market_data: Dict[str, Any]):
        """Execute trading signals with realistic constraints."""
        if not isinstance(signals, dict):
            return

        action = signals.get("action", "").upper()
        symbol = signals.get("market") or signals.get("symbol")
        size = signals.get("size", 0)
        price = signals.get("price")

        if not all([action, symbol, size]):
            return

        # Get market price if not specified
        if price is None:
            price = market_data["prices"].get(symbol)
            if price is None:
                logger.warning(f"No price available for {symbol}")
                return

        # Apply slippage
        if action == "BUY":
            execution_price = price * (1 + self.config.slippage)
        else:
            execution_price = price * (1 - self.config.slippage)

        # Execute trade through portfolio
        success = False
        if action == "BUY":
            success = self.portfolio.buy(
                symbol, size, execution_price, self.current_time
            )
        elif action == "SELL":
            success = self.portfolio.sell(
                symbol, size, execution_price, self.current_time
            )

        if success:
            # Record trade
            trade_value = size * execution_price
            commission = trade_value * self.config.commission

            trade = Trade(
                timestamp=self.current_time,
                symbol=symbol,
                action=action,
                quantity=size,
                price=execution_price,
                commission=commission,
                value=trade_value,
            )
            self.trades.append(trade)

            logger.debug(f"Executed: {action} {size} {symbol} @ ${execution_price:.2f}")

    def _should_stop_trading(self) -> bool:
        """Check if trading should stop due to risk limits."""
        current_value = self.portfolio.cash + sum(
            pos.quantity * pos.current_price
            for pos in self.portfolio.positions.values()
        )

        # Daily loss limit
        daily_loss = (current_value - self.daily_start_value) / self.daily_start_value
        if daily_loss <= -self.config.daily_loss_limit:
            return True

        # Max drawdown limit
        if current_value > self.max_portfolio_value:
            self.max_portfolio_value = current_value
        drawdown = (self.max_portfolio_value - current_value) / self.max_portfolio_value
        if drawdown >= self.config.max_drawdown_limit:
            return True

        return False

    def _is_new_day(self, timestamp: datetime) -> bool:
        """Check if this is a new trading day."""
        if not hasattr(self, "_last_day"):
            self._last_day = timestamp.date()
            return True
        if timestamp.date() > self._last_day:
            self._last_day = timestamp.date()
            return True
        return False

    def _update_daily_metrics(self):
        """Update daily performance metrics."""
        current_value = self.portfolio.cash + sum(
            pos.quantity * pos.current_price
            for pos in self.portfolio.positions.values()
        )
        self.daily_start_value = current_value

    def _calculate_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        if not self.daily_values:
            return {}

        values = pd.DataFrame(self.daily_values)
        values.set_index("timestamp", inplace=True)

        returns = values["portfolio_value"].pct_change().dropna()

        metrics = PerformanceMetrics.calculate_all(
            returns=returns,
            portfolio_values=values["portfolio_value"],
            trades=self.trades,
            config=self.config,
        )

        return metrics
