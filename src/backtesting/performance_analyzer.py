"""
Performance Analyzer for Backtesting Results
Provides detailed analytics and visualizations
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dataclasses import dataclass
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from src.backtesting.backtest_engine import BacktestResults, BacktestTrade

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    # Returns
    total_return: float
    annualized_return: float
    monthly_returns: List[float]
    
    # Risk metrics
    volatility: float
    downside_deviation: float
    max_drawdown: float
    max_drawdown_duration: int
    var_95: float  # Value at Risk 95%
    cvar_95: float  # Conditional VaR 95%
    
    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    avg_trade: float
    profit_factor: float
    expectancy: float
    
    # Trade distribution
    largest_win: float
    largest_loss: float
    consecutive_wins: int
    consecutive_losses: int
    avg_holding_period: float
    
    # Kelly metrics
    kelly_criterion: float
    optimal_f: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'returns': {
                'total_return': f"{self.total_return:.2%}",
                'annualized_return': f"{self.annualized_return:.2%}",
                'monthly_avg': f"{np.mean(self.monthly_returns):.2%}"
            },
            'risk': {
                'volatility': f"{self.volatility:.2%}",
                'max_drawdown': f"{self.max_drawdown:.2%}",
                'var_95': f"{self.var_95:.2%}",
                'cvar_95': f"{self.cvar_95:.2%}"
            },
            'risk_adjusted': {
                'sharpe_ratio': f"{self.sharpe_ratio:.2f}",
                'sortino_ratio': f"{self.sortino_ratio:.2f}",
                'calmar_ratio': f"{self.calmar_ratio:.2f}"
            },
            'trades': {
                'total': self.total_trades,
                'win_rate': f"{self.win_rate:.1%}",
                'profit_factor': f"{self.profit_factor:.2f}",
                'expectancy': f"${self.expectancy:.2f}",
                'avg_holding_period': f"{self.avg_holding_period:.1f} hours"
            }
        }


class PerformanceAnalyzer:
    """
    Analyzes backtest performance with detailed metrics
    """
    
    def __init__(self, results_dir: str = "backtesting/results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze(self, backtest_results: BacktestResults) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics
        
        Args:
            backtest_results: Results from backtest
            
        Returns:
            PerformanceMetrics with all calculations
        """
        # Calculate returns metrics
        total_return = backtest_results.total_return
        days = (backtest_results.end_date - backtest_results.start_date).days
        annualized_return = (1 + total_return) ** (365 / max(days, 1)) - 1
        
        # Monthly returns
        monthly_returns = self._calculate_monthly_returns(
            backtest_results.equity_curve,
            backtest_results.start_date
        )
        
        # Risk metrics
        daily_returns = np.array(backtest_results.daily_returns)
        volatility = np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 1 else 0
        
        downside_returns = daily_returns[daily_returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        max_dd, max_dd_duration = self._calculate_max_drawdown_details(
            backtest_results.equity_curve
        )
        
        var_95 = np.percentile(daily_returns, 5) if len(daily_returns) > 0 else 0
        cvar_95 = np.mean(daily_returns[daily_returns <= var_95]) if len(daily_returns[daily_returns <= var_95]) > 0 else var_95
        
        # Risk-adjusted returns
        risk_free_rate = 0  # Assuming 0% risk-free rate
        excess_returns = daily_returns - risk_free_rate / 252
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
        
        sortino = np.mean(excess_returns) / downside_deviation * np.sqrt(252) if downside_deviation > 0 else 0
        
        calmar = annualized_return / max_dd if max_dd > 0 else annualized_return
        
        information_ratio = sharpe  # Simplified (would need benchmark)
        
        # Trade statistics
        trades = backtest_results.trades
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        avg_trade = np.mean([t.pnl for t in trades]) if trades else 0
        
        total_wins = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
        
        # Trade distribution
        largest_win = max([t.pnl for t in trades], default=0)
        largest_loss = min([t.pnl for t in trades], default=0)
        
        consecutive_wins = self._max_consecutive(trades, True)
        consecutive_losses = self._max_consecutive(trades, False)
        
        holding_periods = []
        for trade in trades:
            if trade.exit_timestamp:
                period = (trade.exit_timestamp - trade.timestamp).total_seconds() / 3600
                holding_periods.append(period)
        
        avg_holding_period = np.mean(holding_periods) if holding_periods else 0
        
        # Kelly metrics
        if winning_trades and losing_trades:
            avg_win_ratio = avg_win / abs(avg_loss)
            kelly = (win_rate * avg_win_ratio - (1 - win_rate)) / avg_win_ratio
            kelly = max(0, min(kelly, 1))  # Limit to [0, 1]
        else:
            kelly = 0
        
        optimal_f = kelly * 0.25  # Conservative Kelly
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            monthly_returns=monthly_returns,
            volatility=volatility,
            downside_deviation=downside_deviation,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            var_95=var_95,
            cvar_95=cvar_95,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            information_ratio=information_ratio,
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_trade=avg_trade,
            profit_factor=profit_factor,
            expectancy=expectancy,
            largest_win=largest_win,
            largest_loss=largest_loss,
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses,
            avg_holding_period=avg_holding_period,
            kelly_criterion=kelly,
            optimal_f=optimal_f
        )
    
    def _calculate_monthly_returns(
        self,
        equity_curve: List[float],
        start_date: datetime
    ) -> List[float]:
        """Calculate monthly returns from equity curve"""
        if len(equity_curve) < 30:
            return []
        
        monthly_returns = []
        month_start_value = equity_curve[0]
        days_in_month = 0
        current_month = start_date.month
        
        for i, value in enumerate(equity_curve):
            date = start_date + timedelta(days=i)
            
            if date.month != current_month:
                # Month ended
                if days_in_month > 0:
                    month_return = (value - month_start_value) / month_start_value
                    monthly_returns.append(month_return)
                
                month_start_value = value
                current_month = date.month
                days_in_month = 0
            else:
                days_in_month += 1
        
        return monthly_returns
    
    def _calculate_max_drawdown_details(
        self,
        equity_curve: List[float]
    ) -> Tuple[float, int]:
        """Calculate maximum drawdown and duration"""
        if not equity_curve:
            return 0, 0
        
        peak = equity_curve[0]
        max_dd = 0
        max_dd_duration = 0
        current_dd_start = 0
        
        for i, value in enumerate(equity_curve):
            if value > peak:
                peak = value
                current_dd_start = i
            
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
                max_dd_duration = i - current_dd_start
        
        return max_dd, max_dd_duration
    
    def _max_consecutive(self, trades: List[BacktestTrade], wins: bool) -> int:
        """Calculate maximum consecutive wins or losses"""
        if not trades:
            return 0
        
        max_streak = 0
        current_streak = 0
        
        for trade in trades:
            if (wins and trade.pnl > 0) or (not wins and trade.pnl < 0):
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def create_report(
        self,
        backtest_results: BacktestResults,
        metrics: PerformanceMetrics,
        filename: str = "performance_report.html"
    ):
        """
        Create HTML performance report
        
        Args:
            backtest_results: Backtest results
            metrics: Calculated metrics
            filename: Output filename
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                .metric-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }}
                .metric-card {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
                .metric-title {{ font-weight: bold; color: #666; }}
                .metric-value {{ font-size: 24px; color: #333; margin-top: 5px; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Backtest Performance Report</h1>
            
            <h2>Key Metrics</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-title">Total Return</div>
                    <div class="metric-value {'positive' if metrics.total_return > 0 else 'negative'}">
                        {metrics.total_return:.2%}
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Sharpe Ratio</div>
                    <div class="metric-value">{metrics.sharpe_ratio:.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Win Rate</div>
                    <div class="metric-value">{metrics.win_rate:.1%}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Max Drawdown</div>
                    <div class="metric-value negative">{metrics.max_drawdown:.1%}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Profit Factor</div>
                    <div class="metric-value">{metrics.profit_factor:.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Total Trades</div>
                    <div class="metric-value">{metrics.total_trades}</div>
                </div>
            </div>
            
            <h2>Detailed Statistics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Annualized Return</td><td>{metrics.annualized_return:.2%}</td></tr>
                <tr><td>Volatility</td><td>{metrics.volatility:.2%}</td></tr>
                <tr><td>Sortino Ratio</td><td>{metrics.sortino_ratio:.2f}</td></tr>
                <tr><td>Calmar Ratio</td><td>{metrics.calmar_ratio:.2f}</td></tr>
                <tr><td>Average Win</td><td>${metrics.avg_win:.2f}</td></tr>
                <tr><td>Average Loss</td><td>${metrics.avg_loss:.2f}</td></tr>
                <tr><td>Expectancy</td><td>${metrics.expectancy:.2f}</td></tr>
                <tr><td>Kelly Criterion</td><td>{metrics.kelly_criterion:.2%}</td></tr>
                <tr><td>Avg Holding Period</td><td>{metrics.avg_holding_period:.1f} hours</td></tr>
                <tr><td>Consecutive Wins</td><td>{metrics.consecutive_wins}</td></tr>
                <tr><td>Consecutive Losses</td><td>{metrics.consecutive_losses}</td></tr>
            </table>
            
            <h2>Trade Distribution</h2>
            <table>
                <tr>
                    <th>Winning Trades</th>
                    <th>Losing Trades</th>
                    <th>Largest Win</th>
                    <th>Largest Loss</th>
                </tr>
                <tr>
                    <td>{metrics.winning_trades}</td>
                    <td>{metrics.losing_trades}</td>
                    <td class="positive">${metrics.largest_win:.2f}</td>
                    <td class="negative">${metrics.largest_loss:.2f}</td>
                </tr>
            </table>
            
            <p><small>Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small></p>
        </body>
        </html>
        """
        
        output_path = self.results_dir / filename
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Performance report saved to {output_path}")
    
    def plot_results(
        self,
        backtest_results: BacktestResults,
        save_path: Optional[str] = None
    ):
        """
        Create visualization plots
        
        Args:
            backtest_results: Backtest results
            save_path: Path to save plots
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Equity curve
        ax = axes[0, 0]
        ax.plot(backtest_results.equity_curve)
        ax.set_title('Equity Curve')
        ax.set_xlabel('Days')
        ax.set_ylabel('Portfolio Value ($)')
        ax.grid(True, alpha=0.3)
        
        # Daily returns distribution
        ax = axes[0, 1]
        if backtest_results.daily_returns:
            ax.hist(backtest_results.daily_returns, bins=30, edgecolor='black')
            ax.axvline(0, color='red', linestyle='--', alpha=0.5)
        ax.set_title('Daily Returns Distribution')
        ax.set_xlabel('Return')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Drawdown
        ax = axes[0, 2]
        drawdowns = self._calculate_drawdowns(backtest_results.equity_curve)
        ax.fill_between(range(len(drawdowns)), 0, drawdowns, color='red', alpha=0.3)
        ax.set_title('Drawdown')
        ax.set_xlabel('Days')
        ax.set_ylabel('Drawdown %')
        ax.grid(True, alpha=0.3)
        
        # Trade P&L distribution
        ax = axes[1, 0]
        if backtest_results.trades:
            pnls = [t.pnl for t in backtest_results.trades]
            ax.hist(pnls, bins=20, edgecolor='black')
            ax.axvline(0, color='red', linestyle='--', alpha=0.5)
        ax.set_title('Trade P&L Distribution')
        ax.set_xlabel('P&L ($)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Win rate by month
        ax = axes[1, 1]
        monthly_win_rates = self._calculate_monthly_win_rates(backtest_results)
        if monthly_win_rates:
            ax.bar(range(len(monthly_win_rates)), monthly_win_rates)
        ax.set_title('Monthly Win Rate')
        ax.set_xlabel('Month')
        ax.set_ylabel('Win Rate')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        
        # Rolling Sharpe ratio
        ax = axes[1, 2]
        if len(backtest_results.daily_returns) > 20:
            rolling_sharpe = self._calculate_rolling_sharpe(
                backtest_results.daily_returns, window=20
            )
            ax.plot(rolling_sharpe)
            ax.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax.set_title('Rolling Sharpe Ratio (20-day)')
        ax.set_xlabel('Days')
        ax.set_ylabel('Sharpe Ratio')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Plots saved to {save_path}")
        
        plt.show()
    
    def _calculate_drawdowns(self, equity_curve: List[float]) -> List[float]:
        """Calculate drawdown series"""
        if not equity_curve:
            return []
        
        drawdowns = []
        peak = equity_curve[0]
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            drawdowns.append(-dd)  # Negative for display
        
        return drawdowns
    
    def _calculate_monthly_win_rates(
        self,
        backtest_results: BacktestResults
    ) -> List[float]:
        """Calculate win rate by month"""
        if not backtest_results.trades:
            return []
        
        monthly_trades = {}
        
        for trade in backtest_results.trades:
            month_key = trade.timestamp.strftime('%Y-%m')
            if month_key not in monthly_trades:
                monthly_trades[month_key] = {'wins': 0, 'total': 0}
            
            monthly_trades[month_key]['total'] += 1
            if trade.pnl > 0:
                monthly_trades[month_key]['wins'] += 1
        
        win_rates = []
        for month in sorted(monthly_trades.keys()):
            stats = monthly_trades[month]
            win_rate = stats['wins'] / stats['total'] if stats['total'] > 0 else 0
            win_rates.append(win_rate)
        
        return win_rates
    
    def _calculate_rolling_sharpe(
        self,
        daily_returns: List[float],
        window: int = 20
    ) -> List[float]:
        """Calculate rolling Sharpe ratio"""
        returns = np.array(daily_returns)
        rolling_sharpe = []
        
        for i in range(window, len(returns)):
            window_returns = returns[i-window:i]
            if np.std(window_returns) > 0:
                sharpe = np.mean(window_returns) / np.std(window_returns) * np.sqrt(252)
            else:
                sharpe = 0
            rolling_sharpe.append(sharpe)
        
        return rolling_sharpe
    
    def compare_strategies(
        self,
        results_list: List[Tuple[str, BacktestResults]],
        save_path: Optional[str] = None
    ):
        """
        Compare multiple strategy results
        
        Args:
            results_list: List of (name, results) tuples
            save_path: Path to save comparison
        """
        comparison_data = []
        
        for name, results in results_list:
            metrics = self.analyze(results)
            
            comparison_data.append({
                'Strategy': name,
                'Total Return': f"{metrics.total_return:.2%}",
                'Sharpe Ratio': f"{metrics.sharpe_ratio:.2f}",
                'Max Drawdown': f"{metrics.max_drawdown:.2%}",
                'Win Rate': f"{metrics.win_rate:.1%}",
                'Profit Factor': f"{metrics.profit_factor:.2f}",
                'Total Trades': metrics.total_trades
            })
        
        df = pd.DataFrame(comparison_data)
        
        if save_path:
            df.to_csv(save_path, index=False)
            logger.info(f"Comparison saved to {save_path}")
        
        return df


# Example usage
async def main():
    """Example performance analysis"""
    from src.backtesting.backtest_engine import BacktestEngine, StrategyParameters
    
    # Run backtest
    engine = BacktestEngine()
    params = StrategyParameters()
    
    results = await engine.run_backtest(
        strategy_params=params,
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31),
        markets=["NFL-CHIEFS-WIN"]
    )
    
    # Analyze performance
    analyzer = PerformanceAnalyzer()
    metrics = analyzer.analyze(results)
    
    # Display metrics
    print(json.dumps(metrics.to_dict(), indent=2))
    
    # Create report
    analyzer.create_report(results, metrics)
    
    # Create plots
    analyzer.plot_results(results, save_path="backtesting/results/performance_plots.png")


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())