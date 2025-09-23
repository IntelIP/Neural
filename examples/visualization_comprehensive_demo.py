"""
Comprehensive Visualization Demo for Neural SDK

Demonstrates the full capabilities of the Neural SDK visualization framework:
- Interactive performance analytics and P&L tracking
- Real-time dashboard with live data simulation
- Professional report generation (PDF/HTML) 
- Risk visualization and monitoring
- Strategy comparison analysis
- Market data visualization with signal overlays

This demo showcases how to create production-ready visualizations
for Kalshi sports trading analysis and monitoring.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import webbrowser
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from neural.visualization import (
    PerformanceVisualizer, RiskVisualizer, StrategyVisualizer, MarketVisualizer,
    VisualizationConfig, VisualizationTheme,
    DashboardServer, DashboardConfig,
    ReportGenerator, ReportConfig, ReportType, ExportFormat
)
from neural.analysis.metrics import PerformanceCalculator
from neural.strategy.base import Signal, SignalType
from neural.kalshi.markets import KalshiMarket

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisualizationDemo:
    """Comprehensive visualization capabilities demonstration."""
    
    def __init__(self):
        """Initialize demo with synthetic data."""
        self.setup_demo_data()
        self.setup_visualizers()
        
    def setup_demo_data(self):
        """Create synthetic trading data for demonstration."""
        logger.info("🎲 Generating synthetic trading data for demo...")
        
        # Generate date range (6 months of daily data)
        self.dates = pd.date_range(
            start=datetime.now() - timedelta(days=180),
            end=datetime.now(),
            freq='D'
        )
        
        # Generate synthetic returns for multiple strategies
        np.random.seed(42)  # For reproducible demo
        
        # Strategy 1: Mean Reversion (moderate returns, higher volatility)
        mean_rev_returns = np.random.normal(0.0008, 0.015, len(self.dates))
        mean_rev_returns[50:60] = np.random.normal(-0.02, 0.005, 10)  # Drawdown period
        
        # Strategy 2: Momentum (higher returns, moderate volatility)
        momentum_returns = np.random.normal(0.0012, 0.012, len(self.dates))
        momentum_returns[100:110] = np.random.normal(0.03, 0.008, 10)  # Hot streak
        
        # Strategy 3: Arbitrage (lower returns, lower volatility)
        arb_returns = np.random.normal(0.0005, 0.008, len(self.dates))
        
        # Benchmark (market returns)
        benchmark_returns = np.random.normal(0.0006, 0.011, len(self.dates))
        
        # Create DataFrame
        self.returns_data = pd.DataFrame({
            'Mean_Reversion': mean_rev_returns,
            'Momentum': momentum_returns,
            'Arbitrage': arb_returns,
            'Benchmark': benchmark_returns
        }, index=self.dates)
        
        # Combined portfolio (equal weight)
        self.portfolio_returns = self.returns_data[['Mean_Reversion', 'Momentum', 'Arbitrage']].mean(axis=1)
        
        # Generate synthetic positions data
        self.positions = {
            'NFL_CHIEFS_OVER': {
                'market_value': 2500.0,
                'contracts': 25,
                'side': 'YES',
                'avg_price': 0.65,
                'current_price': 0.70,
                'pnl': 125.0,
                'pnl_pct': 5.0,
                'category': 'NFL',
                'returns': 0.05,
                'volatility': 0.12
            },
            'NBA_LAKERS_TOTAL': {
                'market_value': 1800.0,
                'contracts': 18,
                'side': 'NO',
                'avg_price': 0.55,
                'current_price': 0.48,
                'pnl': -126.0,
                'pnl_pct': -7.0,
                'category': 'NBA',
                'returns': -0.07,
                'volatility': 0.15
            },
            'MLB_DODGERS_WIN': {
                'market_value': 3200.0,
                'contracts': 32,
                'side': 'YES',
                'avg_price': 0.42,
                'current_price': 0.58,
                'pnl': 512.0,
                'pnl_pct': 16.0,
                'category': 'MLB',
                'returns': 0.16,
                'volatility': 0.18
            },
            'CFB_ALABAMA_SPREAD': {
                'market_value': 1500.0,
                'contracts': 15,
                'side': 'YES',
                'avg_price': 0.50,
                'current_price': 0.52,
                'pnl': 30.0,
                'pnl_pct': 2.0,
                'category': 'CFB',
                'returns': 0.02,
                'volatility': 0.10
            }
        }
        
        # Generate sample signals
        self.signals = [
            Signal(
                signal_type=SignalType.BUY_YES,
                market_id="NFL_CHIEFS_OVER",
                timestamp=datetime.now() - timedelta(hours=2),
                confidence=0.85,
                edge=0.08,
                expected_value=45.0,
                recommended_size=0.12,
                max_contracts=50
            ),
            Signal(
                signal_type=SignalType.SELL_NO,
                market_id="NBA_LAKERS_TOTAL",
                timestamp=datetime.now() - timedelta(hours=1),
                confidence=0.72,
                edge=0.05,
                expected_value=28.0,
                recommended_size=0.08,
                max_contracts=30
            ),
            Signal(
                signal_type=SignalType.BUY_NO,
                market_id="MLB_DODGERS_WIN",
                timestamp=datetime.now() - timedelta(minutes=30),
                confidence=0.78,
                edge=0.12,
                expected_value=65.0,
                recommended_size=0.15,
                max_contracts=75
            )
        ]
        
        # Generate synthetic market price data
        self.market_data = pd.DataFrame({
            'last': np.random.uniform(0.45, 0.55, len(self.dates)),
            'bid': np.random.uniform(0.42, 0.52, len(self.dates)),
            'ask': np.random.uniform(0.48, 0.58, len(self.dates)),
            'volume': np.random.randint(100, 2000, len(self.dates)),
            'open_interest': np.random.randint(5000, 15000, len(self.dates))
        }, index=self.dates)
        
        # Ensure bid < last < ask
        self.market_data['bid'] = np.minimum(self.market_data['bid'], self.market_data['last'] - 0.01)
        self.market_data['ask'] = np.maximum(self.market_data['ask'], self.market_data['last'] + 0.01)
        
        logger.info("✅ Generated synthetic data:")
        logger.info(f"  • {len(self.dates)} days of returns data")
        logger.info(f"  • {len(self.positions)} active positions")
        logger.info(f"  • {len(self.signals)} recent signals")
        
    def setup_visualizers(self):
        """Setup visualization components."""
        logger.info("🎨 Initializing visualization framework...")
        
        # Create output directory
        self.output_dir = Path("demo_visualizations")
        self.output_dir.mkdir(exist_ok=True)
        
        # Visualization config
        self.viz_config = VisualizationConfig(
            theme=VisualizationTheme.NEURAL_DARK,
            default_width=1200,
            default_height=600,
            export_format=ExportFormat.HTML,
            auto_show=False,  # Set to False for demo control
            save_charts=True,
            output_directory=str(self.output_dir)
        )
        
        # Initialize visualizers
        self.performance_viz = PerformanceVisualizer(self.viz_config)
        self.risk_viz = RiskVisualizer(self.viz_config)
        self.strategy_viz = StrategyVisualizer(self.viz_config)
        self.market_viz = MarketVisualizer(self.viz_config)
        
        # Report generator
        self.report_config = ReportConfig(
            format=ExportFormat.HTML,
            include_charts=True,
            chart_width=800,
            chart_height=600,
            company_name="Neural Trading Demo",
            report_title="Neural SDK Demo Report"
        )
        self.report_generator = ReportGenerator(self.report_config)
        
        logger.info("✅ Visualization framework initialized")
        
    def demo_performance_analytics(self):
        """Demonstrate comprehensive performance analytics."""
        logger.info("\n📊 DEMO: Performance Analytics")
        logger.info("=" * 50)
        
        # Create comprehensive performance dashboard
        logger.info("Creating performance dashboard...")
        performance_charts = self.performance_viz.create_comprehensive_performance_dashboard(
            self.portfolio_returns,
            benchmark_returns=self.returns_data['Benchmark'],
            save_path=str(self.output_dir / "performance_dashboard")
        )
        
        logger.info(f"✅ Generated {len(performance_charts)} performance charts:")
        for chart_name in performance_charts.keys():
            logger.info(f"  • {chart_name}")
            
        # Create rolling metrics analysis
        logger.info("Creating rolling metrics analysis...")
        rolling_metrics_fig = self.performance_viz.create_rolling_metrics_analysis(
            self.portfolio_returns,
            window=60,  # 60-day rolling
            save_path=str(self.output_dir / "rolling_metrics")
        )
        
        # Portfolio allocation analysis
        logger.info("Creating portfolio allocation dashboard...")
        allocation_charts = self.performance_viz.create_portfolio_allocation_dashboard(
            self.positions,
            save_path=str(self.output_dir / "allocation_dashboard")
        )
        
        logger.info(f"✅ Generated {len(allocation_charts)} allocation charts")
        
        return performance_charts, rolling_metrics_fig, allocation_charts
        
    def demo_risk_analytics(self):
        """Demonstrate risk analysis and monitoring."""
        logger.info("\n⚠️  DEMO: Risk Analytics")
        logger.info("=" * 50)
        
        # Create risk dashboard
        logger.info("Creating risk analysis dashboard...")
        risk_limits = {
            'position_size': {'current': 0.08, 'limit': 0.10},
            'daily_var': {'current': 0.025, 'limit': 0.03},
            'max_drawdown': {'current': 0.08, 'limit': 0.15},
            'concentration': {'current': 0.35, 'limit': 0.40}
        }
        
        risk_charts = self.risk_viz.create_risk_dashboard(
            self.portfolio_returns,
            positions=self.positions,
            risk_limits=risk_limits,
            save_path=str(self.output_dir / "risk_dashboard")
        )
        
        logger.info(f"✅ Generated {len(risk_charts)} risk charts")
        
        # Correlation analysis
        logger.info("Creating correlation analysis...")
        correlation_charts = self.risk_viz.create_correlation_analysis(
            self.returns_data,
            save_path=str(self.output_dir / "correlation_analysis")
        )
        
        logger.info(f"✅ Generated {len(correlation_charts)} correlation charts")
        
        return risk_charts, correlation_charts
        
    def demo_strategy_comparison(self):
        """Demonstrate strategy comparison and analysis."""
        logger.info("\n🎯 DEMO: Strategy Comparison")
        logger.info("=" * 50)
        
        # Prepare strategy data
        strategies_data = {}
        for strategy_name in ['Mean_Reversion', 'Momentum', 'Arbitrage']:
            strategies_data[strategy_name] = {
                'returns': self.returns_data[strategy_name],
                'description': f"{strategy_name.replace('_', ' ')} Strategy"
            }
        
        # Create strategy comparison dashboard
        logger.info("Creating strategy comparison dashboard...")
        comparison_charts = self.strategy_viz.create_strategy_comparison_dashboard(
            strategies_data,
            save_path=str(self.output_dir / "strategy_comparison")
        )
        
        logger.info(f"✅ Generated {len(comparison_charts)} strategy comparison charts")
        
        return comparison_charts
        
    def demo_market_analysis(self):
        """Demonstrate market data visualization."""
        logger.info("\n📈 DEMO: Market Analysis")
        logger.info("=" * 50)
        
        # Create market analysis dashboard
        logger.info("Creating market analysis dashboard...")
        market_charts = self.market_viz.create_market_analysis_dashboard(
            self.market_data.tail(30),  # Last 30 days
            signals=self.signals,
            save_path=str(self.output_dir / "market_analysis")
        )
        
        logger.info(f"✅ Generated {len(market_charts)} market charts")
        
        return market_charts
        
    def demo_report_generation(self):
        """Demonstrate professional report generation."""
        logger.info("\n📑 DEMO: Report Generation")
        logger.info("=" * 50)
        
        # Performance report
        logger.info("Generating performance report...")
        performance_calculator = PerformanceCalculator()
        portfolio_metrics = performance_calculator.calculate_comprehensive_metrics(
            self.portfolio_returns
        )
        
        performance_report_data = {
            'returns': self.portfolio_returns,
            'metrics': portfolio_metrics,
            'period': f"{self.dates[0].strftime('%Y-%m-%d')} to {self.dates[-1].strftime('%Y-%m-%d')}",
            'analysis_text': self._generate_demo_analysis_text(portfolio_metrics),
            'risk_metrics': {
                'daily_var_95': 0.025,
                'daily_var_99': 0.038,
                'expected_shortfall': 0.032,
                'maximum_drawdown': abs(portfolio_metrics.max_drawdown),
                'volatility': portfolio_metrics.volatility
            }
        }
        
        performance_report_path = str(self.output_dir / "performance_report.html")
        self.report_generator.generate_report(
            ReportType.PERFORMANCE,
            performance_report_data,
            performance_report_path
        )
        
        logger.info(f"✅ Generated performance report: {performance_report_path}")
        
        # Strategy comparison report
        logger.info("Generating strategy comparison report...")
        strategies_data = {}
        for strategy_name in ['Mean_Reversion', 'Momentum', 'Arbitrage']:
            strategies_data[strategy_name] = {
                'returns': self.returns_data[strategy_name]
            }
            
        comparison_report_path = str(self.output_dir / "strategy_comparison_report.html")
        comparison_report_data = {
            'strategies': strategies_data
        }
        
        self.report_generator.generate_report(
            ReportType.STRATEGY_COMPARISON,
            comparison_report_data,
            comparison_report_path
        )
        
        logger.info(f"✅ Generated strategy comparison report: {comparison_report_path}")
        
        return performance_report_path, comparison_report_path
        
    def demo_dashboard_server(self, duration_seconds: int = 30):
        """Demonstrate real-time dashboard server."""
        logger.info("\n🖥️  DEMO: Real-time Dashboard")
        logger.info("=" * 50)
        
        # Setup dashboard configuration
        dashboard_config = DashboardConfig(
            theme=DashboardServer.NEURAL,
            host="127.0.0.1",
            port=8050,
            debug=False,
            auto_refresh_interval=2000,  # 2 second refresh
            enable_real_time=True
        )
        
        # Create dashboard server
        dashboard = DashboardServer(dashboard_config)
        
        # Simulate real-time data updates
        logger.info(f"Starting dashboard server on http://{dashboard_config.host}:{dashboard_config.port}")
        logger.info(f"Dashboard will run for {duration_seconds} seconds with simulated data...")
        
        # Initialize with some data
        current_portfolio_value = 10000.0
        daily_pnl = 250.0
        
        dashboard.update_data(
            portfolio_value=current_portfolio_value,
            daily_pnl=daily_pnl,
            positions=self.positions,
            signals=self.signals[:2],  # Recent signals
            risk_metrics={
                'current_drawdown': 0.05,
                'var_95': 0.025,
                'sharpe_ratio': 1.8,
                'volatility': 0.15
            },
            alerts=[
                {
                    'level': 'info',
                    'title': 'New Signal',
                    'message': 'High-confidence BUY signal generated for NFL market'
                }
            ]
        )
        
        # Note: In a real application, you would start the server here
        # dashboard.start_server()
        
        logger.info("✅ Dashboard server demo setup complete")
        logger.info("   (Server not started in demo - would run interactively)")
        
        return dashboard
        
    def _generate_demo_analysis_text(self, metrics):
        """Generate demo analysis text for reports."""
        return f"""
        <h3>Portfolio Performance Analysis</h3>
        
        <p>During the analysis period, the portfolio delivered a total return of <strong>{metrics.total_return:.2%}</strong>, 
        demonstrating solid performance in the Kalshi prediction markets.</p>
        
        <h4>Risk-Adjusted Performance</h4>
        <p>The portfolio achieved a Sharpe ratio of <strong>{metrics.sharpe_ratio:.2f}</strong>, indicating 
        {'excellent' if metrics.sharpe_ratio > 1.5 else 'good' if metrics.sharpe_ratio > 1.0 else 'moderate'} 
        risk-adjusted returns. The Sortino ratio of <strong>{metrics.sortino_ratio:.2f}</strong> shows the strategy's 
        ability to generate consistent positive returns while managing downside risk.</p>
        
        <h4>Drawdown Analysis</h4>
        <p>Maximum drawdown was controlled at <strong>{abs(metrics.max_drawdown):.2%}</strong>, demonstrating 
        effective risk management. The portfolio volatility of <strong>{metrics.volatility:.2%}</strong> is 
        appropriate for the strategy's risk profile.</p>
        
        <h4>Trading Performance</h4>
        <p>The strategy maintained a win rate of <strong>{metrics.win_rate:.1%}</strong>, with an average winning 
        trade of <strong>${metrics.avg_win:.2f}</strong> and average losing trade of 
        <strong>-${abs(metrics.avg_loss):.2f}</strong>.</p>
        
        <h4>Conclusion</h4>
        <p>The portfolio demonstrates strong performance characteristics suitable for systematic prediction market trading. 
        Risk management remains effective, and the strategy shows potential for continued profitability.</p>
        """
        
    def create_demo_summary(self):
        """Create a summary HTML file with links to all generated visualizations."""
        logger.info("\n📋 Creating demo summary...")
        
        summary_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Neural SDK Visualization Demo - Summary</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: #1a1a1a; color: #ffffff; }}
                .container {{ max-width: 1000px; margin: 0 auto; }}
                .header {{ text-align: center; border-bottom: 2px solid #00ff88; padding-bottom: 20px; margin-bottom: 40px; }}
                .section {{ background: #2d2d2d; padding: 20px; margin: 20px 0; border-radius: 8px; }}
                .section h3 {{ color: #00ff88; margin-top: 0; }}
                .file-links {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 10px; }}
                .file-link {{ background: #404040; padding: 10px; border-radius: 5px; text-decoration: none; color: #ffffff; display: block; }}
                .file-link:hover {{ background: #505050; }}
                .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }}
                .stat-card {{ background: #404040; padding: 15px; border-radius: 5px; text-align: center; }}
                .stat-value {{ font-size: 24px; font-weight: bold; color: #00ff88; }}
                .timestamp {{ color: #cccccc; font-size: 14px; text-align: center; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚀 Neural SDK Visualization Demo</h1>
                    <p>Comprehensive demonstration of advanced trading visualization capabilities</p>
                </div>
                
                <div class="section">
                    <h3>📊 Demo Statistics</h3>
                    <div class="stats">
                        <div class="stat-card">
                            <div class="stat-value">{len(self.dates)}</div>
                            <div>Days of Data</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{len(self.positions)}</div>
                            <div>Active Positions</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{len(self.signals)}</div>
                            <div>Recent Signals</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{self.portfolio_returns.mean()*252:.1%}</div>
                            <div>Annualized Return</div>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h3>📈 Performance Analytics</h3>
                    <p>Comprehensive portfolio performance analysis with P&L tracking, drawdown analysis, and rolling metrics.</p>
                    <div class="file-links">
        """
        
        # Add links to all generated files
        for file in self.output_dir.glob("performance_*.html"):
            summary_html += f'<a href="{file.name}" class="file-link">📊 {file.stem.replace("_", " ").title()}</a>'
            
        summary_html += """
                    </div>
                </div>
                
                <div class="section">
                    <h3>⚠️ Risk Analytics</h3>
                    <p>Advanced risk analysis including VaR calculations, correlation analysis, and risk limit monitoring.</p>
                    <div class="file-links">
        """
        
        for file in self.output_dir.glob("risk_*.html"):
            summary_html += f'<a href="{file.name}" class="file-link">⚠️ {file.stem.replace("_", " ").title()}</a>'
            
        for file in self.output_dir.glob("correlation_*.html"):
            summary_html += f'<a href="{file.name}" class="file-link">🔗 {file.stem.replace("_", " ").title()}</a>'
        
        summary_html += """
                    </div>
                </div>
                
                <div class="section">
                    <h3>🎯 Strategy Analysis</h3>
                    <p>Multi-strategy performance comparison and analysis tools.</p>
                    <div class="file-links">
        """
        
        for file in self.output_dir.glob("strategy_*.html"):
            summary_html += f'<a href="{file.name}" class="file-link">🎯 {file.stem.replace("_", " ").title()}</a>'
            
        summary_html += """
                    </div>
                </div>
                
                <div class="section">
                    <h3>📈 Market Analysis</h3>
                    <p>Market data visualization with signal overlays and price action analysis.</p>
                    <div class="file-links">
        """
        
        for file in self.output_dir.glob("market_*.html"):
            summary_html += f'<a href="{file.name}" class="file-link">📈 {file.stem.replace("_", " ").title()}</a>'
            
        summary_html += """
                    </div>
                </div>
                
                <div class="section">
                    <h3>📑 Professional Reports</h3>
                    <p>Comprehensive PDF and HTML reports for performance analysis and strategy comparison.</p>
                    <div class="file-links">
        """
        
        for file in self.output_dir.glob("*report*.html"):
            summary_html += f'<a href="{file.name}" class="file-link">📑 {file.stem.replace("_", " ").title()}</a>'
            
        summary_html += f"""
                    </div>
                </div>
                
                <div class="section">
                    <h3>🖥️ Interactive Dashboard</h3>
                    <p>Real-time dashboard server capabilities demonstrated. In a live environment, this would provide:</p>
                    <ul>
                        <li>Real-time portfolio monitoring and P&L tracking</li>
                        <li>Live signal generation and alert system</li>
                        <li>Interactive risk monitoring and limit alerts</li>
                        <li>Market data visualization with live updates</li>
                    </ul>
                    <p><em>Dashboard server demo completed (would run on http://127.0.0.1:8050 in live mode)</em></p>
                </div>
                
                <div class="timestamp">
                    Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by Neural SDK Visualization Framework
                </div>
            </div>
        </body>
        </html>
        """
        
        summary_path = self.output_dir / "demo_summary.html"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_html)
            
        logger.info(f"✅ Created demo summary: {summary_path}")
        return summary_path
        
    def run_complete_demo(self):
        """Run the complete visualization demo."""
        logger.info("🚀 Starting Neural SDK Visualization Framework Demo")
        logger.info("=" * 60)
        
        try:
            # Run all demo sections
            self.demo_performance_analytics()
            self.demo_risk_analytics()  
            self.demo_strategy_comparison()
            self.demo_market_analysis()
            self.demo_report_generation()
            self.demo_dashboard_server()
            
            # Create summary
            summary_path = self.create_demo_summary()
            
            logger.info("\n🎉 DEMO COMPLETE!")
            logger.info("=" * 60)
            logger.info(f"📂 All visualizations saved to: {self.output_dir.absolute()}")
            logger.info(f"📋 Demo summary: {summary_path.absolute()}")
            logger.info("\n✨ Neural SDK Visualization Framework Features Demonstrated:")
            logger.info("  ✅ Interactive performance analytics and P&L tracking")
            logger.info("  ✅ Advanced risk analysis and monitoring")
            logger.info("  ✅ Multi-strategy comparison and analysis") 
            logger.info("  ✅ Market data visualization with signal overlays")
            logger.info("  ✅ Professional report generation (PDF/HTML)")
            logger.info("  ✅ Real-time dashboard server capabilities")
            
            # Open summary in browser
            try:
                webbrowser.open(f'file://{summary_path.absolute()}')
                logger.info("🌐 Opening demo summary in default browser...")
            except Exception:
                logger.info("💡 Open the demo summary manually in your browser")
                
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            raise


def main():
    """Run the visualization demo."""
    demo = VisualizationDemo()
    demo.run_complete_demo()


if __name__ == "__main__":
    main()
