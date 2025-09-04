"""
Performance Metrics for Backtesting

Comprehensive financial metrics for strategy evaluation:
- Return metrics
- Risk metrics
- Risk-adjusted returns
- Drawdown analysis
- Trade analysis
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics with calculation methods."""

    @staticmethod
    def calculate_all(
        returns: pd.Series, portfolio_values: pd.Series, trades: List[Any], config: Any
    ) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.

        Args:
            returns: Series of period returns (as percentages)
            portfolio_values: Series of portfolio values over time
            trades: List of Trade objects
            config: BacktestConfig object

        Returns:
            Dictionary of all performance metrics
        """
        metrics = {}

        try:
            # Basic return metrics
            metrics.update(
                PerformanceMetrics.calculate_return_metrics(returns, portfolio_values)
            )

            # Risk metrics
            metrics.update(
                PerformanceMetrics.calculate_risk_metrics(returns, portfolio_values)
            )

            # Risk-adjusted metrics
            metrics.update(PerformanceMetrics.calculate_risk_adjusted_metrics(returns))

            # Drawdown metrics
            metrics.update(
                PerformanceMetrics.calculate_drawdown_metrics(portfolio_values)
            )

            # Trade-based metrics
            metrics.update(PerformanceMetrics.calculate_trade_metrics(trades))

            # Time-based metrics
            metrics.update(
                PerformanceMetrics.calculate_time_metrics(returns, portfolio_values)
            )

            # Risk management metrics
            metrics.update(
                PerformanceMetrics.calculate_risk_management_metrics(returns, config)
            )

        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")

        return metrics

    @staticmethod
    def calculate_return_metrics(
        returns: pd.Series, portfolio_values: pd.Series
    ) -> Dict[str, float]:
        """Calculate return-based metrics."""
        if len(returns) == 0 or len(portfolio_values) == 0:
            return {}

        initial_value = portfolio_values.iloc[0]
        final_value = portfolio_values.iloc[-1]

        return {
            "total_return": ((final_value / initial_value) - 1) * 100,
            "total_return_abs": final_value - initial_value,
            "annualized_return": PerformanceMetrics._annualized_return(returns),
            "avg_daily_return": returns.mean(),
            "median_daily_return": returns.median(),
            "geometric_mean_return": PerformanceMetrics._geometric_mean(returns),
            "compound_annual_growth_rate": PerformanceMetrics._cagr(portfolio_values),
        }

    @staticmethod
    def calculate_risk_metrics(
        returns: pd.Series, portfolio_values: pd.Series
    ) -> Dict[str, float]:
        """Calculate risk-based metrics."""
        if len(returns) == 0:
            return {}

        return {
            "volatility": returns.std(),
            "annualized_volatility": returns.std()
            * np.sqrt(252),  # Assuming daily returns
            "downside_volatility": PerformanceMetrics._downside_deviation(returns),
            "value_at_risk_5": returns.quantile(0.05),
            "value_at_risk_1": returns.quantile(0.01),
            "conditional_value_at_risk_5": returns[
                returns <= returns.quantile(0.05)
            ].mean(),
            "skewness": returns.skew(),
            "kurtosis": returns.kurtosis(),
            "positive_periods": (returns > 0).sum(),
            "negative_periods": (returns < 0).sum(),
            "zero_periods": (returns == 0).sum(),
        }

    @staticmethod
    def calculate_risk_adjusted_metrics(
        returns: pd.Series, risk_free_rate: float = 0.02
    ) -> Dict[str, float]:
        """Calculate risk-adjusted return metrics."""
        if len(returns) == 0:
            return {}

        # Convert annual risk-free rate to daily
        daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
        excess_returns = returns - daily_rf

        metrics = {}

        # Sharpe Ratio
        if returns.std() != 0:
            metrics["sharpe_ratio"] = (
                excess_returns.mean() / returns.std() * np.sqrt(252)
            )
        else:
            metrics["sharpe_ratio"] = 0

        # Sortino Ratio
        downside_dev = PerformanceMetrics._downside_deviation(returns)
        if downside_dev != 0:
            metrics["sortino_ratio"] = (
                excess_returns.mean() / downside_dev * np.sqrt(252)
            )
        else:
            metrics["sortino_ratio"] = 0

        # Calmar Ratio (annual return / max drawdown)
        annual_return = PerformanceMetrics._annualized_return(returns)
        max_dd = PerformanceMetrics._max_drawdown_from_returns(returns)
        if max_dd != 0:
            metrics["calmar_ratio"] = annual_return / abs(max_dd)
        else:
            metrics["calmar_ratio"] = 0

        return metrics

    @staticmethod
    def calculate_drawdown_metrics(portfolio_values: pd.Series) -> Dict[str, float]:
        """Calculate drawdown-based metrics."""
        if len(portfolio_values) == 0:
            return {}

        # Calculate drawdown series
        rolling_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - rolling_max) / rolling_max * 100

        return {
            "max_drawdown": drawdown.min(),
            "avg_drawdown": (
                drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0
            ),
            "drawdown_periods": (drawdown < 0).sum(),
            "max_drawdown_duration": PerformanceMetrics._max_drawdown_duration(
                drawdown
            ),
            "recovery_factor": (
                (portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1)
                / abs(drawdown.min())
                if drawdown.min() != 0
                else 0
            ),
            "current_drawdown": drawdown.iloc[-1],
        }

    @staticmethod
    def calculate_trade_metrics(trades: List[Any]) -> Dict[str, float]:
        """Calculate trade-based metrics."""
        if not trades:
            return {}

        # Extract realized P&L from completed trades
        realized_pnls = [
            trade.pnl
            for trade in trades
            if hasattr(trade, "pnl") and trade.pnl is not None
        ]

        if not realized_pnls:
            return {"total_trades": len(trades)}

        realized_pnls = np.array(realized_pnls)
        winning_trades = realized_pnls[realized_pnls > 0]
        losing_trades = realized_pnls[realized_pnls < 0]

        metrics = {
            "total_trades": len(trades),
            "completed_trades": len(realized_pnls),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": (
                len(winning_trades) / len(realized_pnls) * 100
                if len(realized_pnls) > 0
                else 0
            ),
            "avg_win": winning_trades.mean() if len(winning_trades) > 0 else 0,
            "avg_loss": losing_trades.mean() if len(losing_trades) > 0 else 0,
            "largest_win": winning_trades.max() if len(winning_trades) > 0 else 0,
            "largest_loss": losing_trades.min() if len(losing_trades) > 0 else 0,
            "total_realized_pnl": realized_pnls.sum(),
        }

        # Profit factor (gross profit / gross loss)
        gross_profit = winning_trades.sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades.sum()) if len(losing_trades) > 0 else 0
        metrics["profit_factor"] = (
            gross_profit / gross_loss
            if gross_loss > 0
            else float("inf") if gross_profit > 0 else 0
        )

        # Average win/loss ratio
        if len(losing_trades) > 0 and metrics["avg_loss"] != 0:
            metrics["avg_win_loss_ratio"] = abs(
                metrics["avg_win"] / metrics["avg_loss"]
            )
        else:
            metrics["avg_win_loss_ratio"] = 0

        return metrics

    @staticmethod
    def calculate_time_metrics(
        returns: pd.Series, portfolio_values: pd.Series
    ) -> Dict[str, float]:
        """Calculate time-based metrics."""
        if len(returns) == 0:
            return {}

        return {
            "total_periods": len(returns),
            "start_date": returns.index[0] if len(returns) > 0 else None,
            "end_date": returns.index[-1] if len(returns) > 0 else None,
            "trading_days": len(returns),
            "best_day": returns.max(),
            "worst_day": returns.min(),
            "consecutive_wins": PerformanceMetrics._max_consecutive_periods(
                returns > 0
            ),
            "consecutive_losses": PerformanceMetrics._max_consecutive_periods(
                returns < 0
            ),
        }

    @staticmethod
    def calculate_risk_management_metrics(
        returns: pd.Series, config: Any
    ) -> Dict[str, float]:
        """Calculate risk management specific metrics."""
        if len(returns) == 0:
            return {}

        metrics = {}

        # Check against configured limits
        if hasattr(config, "daily_loss_limit"):
            daily_losses = returns[returns < 0]
            worst_daily_loss = daily_losses.min() if len(daily_losses) > 0 else 0
            metrics["worst_daily_loss_pct"] = worst_daily_loss
            metrics["daily_loss_limit_breached"] = (
                worst_daily_loss < -config.daily_loss_limit * 100
            )

        # Calculate tail risk metrics
        if len(returns) > 0:
            metrics["tail_ratio"] = (
                abs(returns.quantile(0.95) / returns.quantile(0.05))
                if returns.quantile(0.05) != 0
                else 0
            )

        return metrics

    # Helper methods
    @staticmethod
    def _annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
        """Calculate annualized return."""
        if len(returns) == 0:
            return 0
        total_periods = len(returns)
        compound_return = (1 + returns / 100).prod()
        return (compound_return ** (periods_per_year / total_periods) - 1) * 100

    @staticmethod
    def _geometric_mean(returns: pd.Series) -> float:
        """Calculate geometric mean return."""
        if len(returns) == 0:
            return 0
        return ((1 + returns / 100).prod() ** (1 / len(returns)) - 1) * 100

    @staticmethod
    def _cagr(portfolio_values: pd.Series) -> float:
        """Calculate Compound Annual Growth Rate."""
        if len(portfolio_values) < 2:
            return 0

        initial_value = portfolio_values.iloc[0]
        final_value = portfolio_values.iloc[-1]

        # Calculate number of years
        time_diff = portfolio_values.index[-1] - portfolio_values.index[0]
        years = time_diff.days / 365.25

        if years == 0 or initial_value == 0:
            return 0

        return ((final_value / initial_value) ** (1 / years) - 1) * 100

    @staticmethod
    def _downside_deviation(returns: pd.Series, target_return: float = 0) -> float:
        """Calculate downside deviation."""
        downside_returns = returns[returns < target_return]
        if len(downside_returns) == 0:
            return 0
        return ((downside_returns - target_return) ** 2).mean() ** 0.5

    @staticmethod
    def _max_drawdown_from_returns(returns: pd.Series) -> float:
        """Calculate max drawdown from returns series."""
        cumulative = (1 + returns / 100).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        return drawdown.min()

    @staticmethod
    def _max_drawdown_duration(drawdown: pd.Series) -> int:
        """Calculate maximum drawdown duration in periods."""
        if len(drawdown) == 0:
            return 0

        # Find periods where we're in drawdown
        in_drawdown = drawdown < 0

        if not in_drawdown.any():
            return 0

        # Find consecutive drawdown periods
        groups = (in_drawdown != in_drawdown.shift()).cumsum()
        drawdown_periods = in_drawdown.groupby(groups).sum()

        return int(drawdown_periods.max()) if len(drawdown_periods) > 0 else 0

    @staticmethod
    def _max_consecutive_periods(condition: pd.Series) -> int:
        """Calculate maximum consecutive periods meeting condition."""
        if len(condition) == 0:
            return 0

        groups = (condition != condition.shift()).cumsum()
        consecutive_counts = condition.groupby(groups).sum()
        return int(consecutive_counts.max()) if len(consecutive_counts) > 0 else 0

    @staticmethod
    def format_metrics(metrics: Dict[str, Any], precision: int = 4) -> Dict[str, str]:
        """
        Format metrics for display with appropriate precision.

        Args:
            metrics: Raw metrics dictionary
            precision: Decimal places for rounding

        Returns:
            Formatted metrics dictionary
        """
        formatted = {}

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if "ratio" in key.lower() or "factor" in key.lower():
                    formatted[key] = f"{value:.{precision}f}"
                elif (
                    "rate" in key.lower()
                    or "return" in key.lower()
                    or "pct" in key.lower()
                ):
                    formatted[key] = f"{value:.2f}%"
                elif "pnl" in key.lower() or "value" in key.lower():
                    formatted[key] = f"${value:,.2f}"
                elif "periods" in key.lower() or "trades" in key.lower():
                    formatted[key] = f"{int(value):,}"
                else:
                    formatted[key] = f"{value:.{precision}f}"
            elif isinstance(value, datetime):
                formatted[key] = value.strftime("%Y-%m-%d")
            else:
                formatted[key] = str(value)

        return formatted

    @staticmethod
    def create_summary_report(metrics: Dict[str, Any]) -> str:
        """Create a formatted summary report."""
        formatted = PerformanceMetrics.format_metrics(metrics)

        report = """
PERFORMANCE SUMMARY
===================

Return Metrics:
- Total Return: {total_return}
- Annualized Return: {annualized_return}
- CAGR: {compound_annual_growth_rate}

Risk Metrics:
- Volatility (Ann.): {annualized_volatility}
- Max Drawdown: {max_drawdown}
- VaR (5%): {value_at_risk_5}

Risk-Adjusted:
- Sharpe Ratio: {sharpe_ratio}
- Sortino Ratio: {sortino_ratio}
- Calmar Ratio: {calmar_ratio}

Trade Analysis:
- Total Trades: {total_trades}
- Win Rate: {win_rate}
- Profit Factor: {profit_factor}
- Avg Win/Loss: {avg_win_loss_ratio}
        """.format(
            **{
                k: formatted.get(k, "N/A")
                for k in [
                    "total_return",
                    "annualized_return",
                    "compound_annual_growth_rate",
                    "annualized_volatility",
                    "max_drawdown",
                    "value_at_risk_5",
                    "sharpe_ratio",
                    "sortino_ratio",
                    "calmar_ratio",
                    "total_trades",
                    "win_rate",
                    "profit_factor",
                    "avg_win_loss_ratio",
                ]
            }
        )

        return report.strip()
