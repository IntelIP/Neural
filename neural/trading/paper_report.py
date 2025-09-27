"""
Paper Trading Report Generator

Generate comprehensive reports from paper trading data including:
- Performance summary
- Trade analysis
- Portfolio evolution
- Strategy effectiveness
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

logger = logging.getLogger(__name__)


class PaperTradingReporter:
    """Generate reports from paper trading data."""

    def __init__(self, data_dir: str = "paper_trading_data"):
        """
        Initialize reporter.

        Args:
            data_dir: Directory containing paper trading data
        """
        self.data_dir = Path(data_dir)
        self.trades_data: List[Dict[str, Any]] = []
        self.portfolio_snapshots: List[Dict[str, Any]] = []

    def load_data(self, days_back: int = 30) -> bool:
        """
        Load paper trading data from files.

        Args:
            days_back: Number of days of data to load

        Returns:
            True if data loaded successfully
        """
        try:
            # Load trade data from JSON Lines files
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            self.trades_data = []
            current_date = start_date

            while current_date <= end_date:
                date_str = current_date.strftime('%Y%m%d')
                trade_file = self.data_dir / f"trades_{date_str}.jsonl"

                if trade_file.exists():
                    with open(trade_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                trade = json.loads(line.strip())
                                self.trades_data.append(trade)

                current_date += timedelta(days=1)

            # Load portfolio snapshots
            self.portfolio_snapshots = []
            for snapshot_file in self.data_dir.glob("portfolio_snapshot_*.json"):
                try:
                    with open(snapshot_file, 'r') as f:
                        snapshot = json.load(f)
                        self.portfolio_snapshots.append(snapshot)
                except Exception as e:
                    logger.warning(f"Error loading snapshot {snapshot_file}: {e}")

            logger.info(f"Loaded {len(self.trades_data)} trades and {len(self.portfolio_snapshots)} snapshots")
            return True

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False

    def generate_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary."""
        if not self.trades_data:
            return {"error": "No trade data available"}

        if not PANDAS_AVAILABLE:
            return self._generate_performance_summary_basic()

        try:
            # Convert to DataFrame for analysis
            df = pd.DataFrame(self.trades_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Basic metrics
            total_trades = len(df)
            winning_trades = len(df[df['realized_pnl'] > 0])
            losing_trades = len(df[df['realized_pnl'] < 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

            # P&L metrics
            total_realized_pnl = df['realized_pnl'].fillna(0).sum()
            avg_win = df[df['realized_pnl'] > 0]['realized_pnl'].mean() if winning_trades > 0 else 0
            avg_loss = df[df['realized_pnl'] < 0]['realized_pnl'].mean() if losing_trades > 0 else 0

            # Strategy analysis
            strategy_performance = df.groupby('strategy').agg({
                'realized_pnl': ['count', 'sum', 'mean'],
                'sentiment_score': 'mean',
                'confidence': 'mean'
            }).round(3) if 'strategy' in df.columns else None

            # Time analysis
            first_trade = df['timestamp'].min()
            last_trade = df['timestamp'].max()
            trading_period = (last_trade - first_trade).days if total_trades > 1 else 0

            return {
                'period': {
                    'start_date': first_trade.isoformat() if first_trade else None,
                    'end_date': last_trade.isoformat() if last_trade else None,
                    'trading_days': trading_period
                },
                'trade_metrics': {
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'win_rate': win_rate,
                    'avg_trades_per_day': total_trades / max(trading_period, 1)
                },
                'pnl_metrics': {
                    'total_realized_pnl': total_realized_pnl,
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'profit_factor': abs(avg_win / avg_loss) if avg_loss < 0 else float('inf'),
                    'total_commission': df['commission'].sum(),
                    'total_slippage': df['slippage'].sum()
                },
                'strategy_performance': strategy_performance.to_dict() if strategy_performance is not None else None
            }

        except Exception as e:
            logger.error(f"Error generating performance summary: {e}")
            return {"error": str(e)}

    def _generate_performance_summary_basic(self) -> Dict[str, Any]:
        """Generate basic performance summary without pandas."""
        try:
            # Basic calculations without pandas
            total_trades = len(self.trades_data)
            realized_pnls = [trade.get('realized_pnl', 0) for trade in self.trades_data if trade.get('realized_pnl') is not None]

            winning_trades = sum(1 for pnl in realized_pnls if pnl > 0)
            losing_trades = sum(1 for pnl in realized_pnls if pnl < 0)
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

            total_realized_pnl = sum(realized_pnls)
            avg_win = sum(pnl for pnl in realized_pnls if pnl > 0) / max(winning_trades, 1)
            avg_loss = sum(pnl for pnl in realized_pnls if pnl < 0) / max(losing_trades, 1)

            # Time analysis
            timestamps = [trade.get('timestamp') for trade in self.trades_data if trade.get('timestamp')]
            if timestamps:
                first_trade = min(timestamps)
                last_trade = max(timestamps)
                # Basic date parsing
                try:
                    first_dt = datetime.fromisoformat(first_trade.replace('Z', '+00:00'))
                    last_dt = datetime.fromisoformat(last_trade.replace('Z', '+00:00'))
                    trading_period = (last_dt - first_dt).days
                except:
                    trading_period = 0
            else:
                first_trade = last_trade = None
                trading_period = 0

            return {
                'period': {
                    'start_date': first_trade,
                    'end_date': last_trade,
                    'trading_days': trading_period
                },
                'trade_metrics': {
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'win_rate': win_rate,
                    'avg_trades_per_day': total_trades / max(trading_period, 1)
                },
                'pnl_metrics': {
                    'total_realized_pnl': total_realized_pnl,
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'profit_factor': abs(avg_win / avg_loss) if avg_loss < 0 else float('inf'),
                    'total_commission': sum(trade.get('commission', 0) for trade in self.trades_data),
                    'total_slippage': sum(trade.get('slippage', 0) for trade in self.trades_data)
                }
            }

        except Exception as e:
            logger.error(f"Error generating basic performance summary: {e}")
            return {"error": str(e)}

    def generate_sentiment_analysis(self) -> Dict[str, Any]:
        """Analyze performance by sentiment levels."""
        if not self.trades_data:
            return {"error": "No trade data available"}

        if not PANDAS_AVAILABLE:
            return {"error": "Pandas required for sentiment analysis"}

        try:
            df = pd.DataFrame(self.trades_data)

            # Filter trades with sentiment data
            sentiment_trades = df[df['sentiment_score'].notna() & df['confidence'].notna()]

            if sentiment_trades.empty:
                return {"error": "No sentiment data available"}

            # Sentiment bins
            sentiment_trades['sentiment_bin'] = pd.cut(
                sentiment_trades['sentiment_score'],
                bins=[-1, -0.3, 0.3, 1],
                labels=['Bearish', 'Neutral', 'Bullish']
            )

            confidence_trades = sentiment_trades.copy()
            confidence_trades['confidence_bin'] = pd.cut(
                confidence_trades['confidence'],
                bins=[0, 0.6, 0.8, 1],
                labels=['Low', 'Medium', 'High']
            )

            # Performance by sentiment
            sentiment_perf = sentiment_trades.groupby('sentiment_bin').agg({
                'realized_pnl': ['count', 'mean', 'sum'],
                'sentiment_score': 'mean',
                'confidence': 'mean'
            }).round(3)

            # Performance by confidence
            confidence_perf = confidence_trades.groupby('confidence_bin').agg({
                'realized_pnl': ['count', 'mean', 'sum'],
                'sentiment_score': 'mean',
                'confidence': 'mean'
            }).round(3)

            return {
                'sentiment_performance': sentiment_perf.to_dict(),
                'confidence_performance': confidence_perf.to_dict(),
                'correlation_sentiment_pnl': sentiment_trades['sentiment_score'].corr(sentiment_trades['realized_pnl'].fillna(0)),
                'correlation_confidence_pnl': confidence_trades['confidence'].corr(confidence_trades['realized_pnl'].fillna(0))
            }

        except Exception as e:
            logger.error(f"Error generating sentiment analysis: {e}")
            return {"error": str(e)}

    def create_equity_curve_plot(self, save_path: Optional[str] = None) -> str:
        """Create equity curve plot."""
        if not MATPLOTLIB_AVAILABLE or not PANDAS_AVAILABLE:
            return "Error: matplotlib and pandas required for plotting"

        try:
            if not self.trades_data:
                return "No trade data available"

            df = pd.DataFrame(self.trades_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')

            # Calculate cumulative P&L
            df['cumulative_pnl'] = df['realized_pnl'].fillna(0).cumsum()

            # Assume starting capital of $10,000
            starting_capital = 10000
            df['portfolio_value'] = starting_capital + df['cumulative_pnl']

            plt.figure(figsize=(12, 6))
            plt.plot(df['timestamp'], df['portfolio_value'], linewidth=2, color='blue')
            plt.axhline(y=starting_capital, color='gray', linestyle='--', alpha=0.7, label='Starting Capital')

            plt.title('Paper Trading Equity Curve')
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value ($)')
            plt.grid(True, alpha=0.3)
            plt.legend()

            # Format y-axis as currency
            ax = plt.gca()
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                return f"Equity curve saved to {save_path}"
            else:
                plt.show()
                return "Equity curve displayed"

        except Exception as e:
            logger.error(f"Error creating equity curve: {e}")
            return f"Error: {str(e)}"

    def generate_html_report(self, output_file: str = "paper_trading_report.html") -> str:
        """Generate comprehensive HTML report."""
        try:
            performance = self.generate_performance_summary()
            sentiment_analysis = self.generate_sentiment_analysis()

            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Paper Trading Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                    .metric-box {{
                        display: inline-block;
                        background-color: #e8f4fd;
                        padding: 15px;
                        margin: 10px;
                        border-radius: 5px;
                        min-width: 150px;
                    }}
                    .positive {{ color: green; }}
                    .negative {{ color: red; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>📊 Paper Trading Performance Report</h1>
                    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>

                <h2>📈 Performance Summary</h2>
            """

            if "error" not in performance:
                html_content += f"""
                <div class="metric-box">
                    <h3>Total Trades</h3>
                    <p><strong>{performance['trade_metrics']['total_trades']}</strong></p>
                </div>
                <div class="metric-box">
                    <h3>Win Rate</h3>
                    <p><strong>{performance['trade_metrics']['win_rate']:.1f}%</strong></p>
                </div>
                <div class="metric-box">
                    <h3>Total P&L</h3>
                    <p class="{'positive' if performance['pnl_metrics']['total_realized_pnl'] >= 0 else 'negative'}">
                        <strong>${performance['pnl_metrics']['total_realized_pnl']:.2f}</strong>
                    </p>
                </div>
                <div class="metric-box">
                    <h3>Profit Factor</h3>
                    <p><strong>{performance['pnl_metrics']['profit_factor']:.2f}</strong></p>
                </div>
                """

            if "error" not in sentiment_analysis:
                html_content += f"""
                <h2>🎯 Sentiment Analysis</h2>
                <p>Correlation between sentiment and P&L: <strong>{sentiment_analysis.get('correlation_sentiment_pnl', 0):.3f}</strong></p>
                <p>Correlation between confidence and P&L: <strong>{sentiment_analysis.get('correlation_confidence_pnl', 0):.3f}</strong></p>
                """

            html_content += """
                <h2>💡 Recommendations</h2>
                <ul>
                    <li>Focus on trades with high confidence scores (>0.8) for better performance</li>
                    <li>Consider position sizing based on sentiment strength</li>
                    <li>Monitor correlation between sentiment and actual outcomes</li>
                    <li>Review trades with negative P&L to identify patterns</li>
                </ul>

                <div style="margin-top: 40px; padding: 20px; background-color: #f9f9f9; border-radius: 5px;">
                    <p><em>This report is generated from paper trading data. Past performance does not guarantee future results.</em></p>
                </div>
            </body>
            </html>
            """

            with open(output_file, 'w') as f:
                f.write(html_content)

            return f"HTML report saved to {output_file}"

        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
            return f"Error: {str(e)}"

    def print_summary_report(self) -> None:
        """Print a summary report to console."""
        print("\n" + "=" * 60)
        print("📊 PAPER TRADING PERFORMANCE REPORT")
        print("=" * 60)

        performance = self.generate_performance_summary()

        if "error" in performance:
            print(f"❌ Error: {performance['error']}")
            return

        # Trading period
        period = performance['period']
        print(f"\n📅 Trading Period:")
        print(f"   Start: {period.get('start_date', 'N/A')}")
        print(f"   End: {period.get('end_date', 'N/A')}")
        print(f"   Days: {period.get('trading_days', 0)}")

        # Trade metrics
        trades = performance['trade_metrics']
        print(f"\n📈 Trade Statistics:")
        print(f"   Total Trades: {trades['total_trades']}")
        print(f"   Winning Trades: {trades['winning_trades']}")
        print(f"   Losing Trades: {trades['losing_trades']}")
        print(f"   Win Rate: {trades['win_rate']:.1f}%")
        print(f"   Avg Trades/Day: {trades['avg_trades_per_day']:.1f}")

        # P&L metrics
        pnl = performance['pnl_metrics']
        print(f"\n💰 P&L Analysis:")
        print(f"   Total Realized P&L: ${pnl['total_realized_pnl']:.2f}")
        print(f"   Average Win: ${pnl['avg_win']:.2f}")
        print(f"   Average Loss: ${pnl['avg_loss']:.2f}")
        print(f"   Profit Factor: {pnl['profit_factor']:.2f}")
        print(f"   Total Commission: ${pnl['total_commission']:.2f}")
        print(f"   Total Slippage: ${pnl['total_slippage']:.2f}")

        # Sentiment analysis
        sentiment = self.generate_sentiment_analysis()
        if "error" not in sentiment:
            print(f"\n🎯 Sentiment Performance:")
            print(f"   Sentiment-P&L Correlation: {sentiment.get('correlation_sentiment_pnl', 0):.3f}")
            print(f"   Confidence-P&L Correlation: {sentiment.get('correlation_confidence_pnl', 0):.3f}")

        print("\n" + "=" * 60)


# Command line utility functions
def create_report(data_dir: str = "paper_trading_data", days_back: int = 30) -> None:
    """Create and display a paper trading report."""
    reporter = PaperTradingReporter(data_dir)

    if reporter.load_data(days_back):
        reporter.print_summary_report()

        # Generate HTML report
        html_result = reporter.generate_html_report()
        print(f"\n📄 {html_result}")

        # Create equity curve
        plot_result = reporter.create_equity_curve_plot("equity_curve.png")
        print(f"📈 {plot_result}")
    else:
        print("❌ Failed to load paper trading data")


if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "paper_trading_data"
    days_back = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    create_report(data_dir, days_back)