"""
Main Visualization Framework for Neural SDK

Provides the central interface for all visualization capabilities including:
- Unified visualization API for performance, risk, and market analysis
- Theme management and customization
- Integration with charts, dashboards, and reports
- Batch visualization processing and export
- Template-based visualization generation
"""

import os
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path

import pandas as pd
import numpy as np

# Handle optional plotly dependencies
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    px = None

# Import with graceful error handling
try:
    from .charts import (
        ChartManager, ChartTheme, ChartConfig, PerformanceChart, 
        RiskChart, MarketChart, PnLChart, DrawdownChart, CorrelationChart
    )
    CHARTS_AVAILABLE = True
except ImportError:
    CHARTS_AVAILABLE = False

try:
    from .dashboard import DashboardServer, DashboardConfig, RealTimeDashboard
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False

try:
    from .reports import ReportGenerator, ReportConfig, ExportFormat, ReportType
    REPORTS_AVAILABLE = True
except ImportError:
    REPORTS_AVAILABLE = False
from ..analysis.metrics import PerformanceMetrics, PerformanceCalculator
from ..strategy.base import Signal, StrategyResult
from ..risk.limits import LimitViolation

logger = logging.getLogger(__name__)


class VisualizationTheme(Enum):
    """Global visualization themes."""
    NEURAL_DARK = "neural_dark"
    NEURAL_LIGHT = "neural_light" 
    TRADING_DARK = "trading_dark"
    TRADING_LIGHT = "trading_light"
    MINIMAL = "minimal"
    COLORFUL = "colorful"


@dataclass 
class VisualizationConfig:
    """Configuration for visualization framework."""
    theme: VisualizationTheme = VisualizationTheme.NEURAL_DARK
    default_width: int = 1200
    default_height: int = 600
    export_dpi: int = 300
    export_format: ExportFormat = ExportFormat.HTML
    auto_show: bool = True
    save_charts: bool = False
    output_directory: str = "visualizations"
    template_directory: Optional[str] = None


class PerformanceVisualizer:
    """
    Specialized visualizer for performance analytics.
    
    Provides comprehensive performance visualization capabilities
    with automated analysis and professional presentation.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize performance visualizer."""
        if not CHARTS_AVAILABLE:
            raise ImportError("Chart components not available. Install plotly to use PerformanceVisualizer.")
        
        self.config = config or VisualizationConfig()
        self.chart_manager = ChartManager(self._get_chart_config())
        self.performance_chart = PerformanceChart(self.chart_manager)
        self.pnl_chart = PnLChart(self.chart_manager)
        self.drawdown_chart = DrawdownChart(self.chart_manager)
        self.performance_calculator = PerformanceCalculator()
        
    def _get_chart_config(self) -> ChartConfig:
        """Get chart configuration from visualization config."""
        theme_mapping = {
            VisualizationTheme.NEURAL_DARK: ChartTheme.NEURAL,
            VisualizationTheme.TRADING_DARK: ChartTheme.TRADING,
            VisualizationTheme.NEURAL_LIGHT: ChartTheme.WHITE,
            VisualizationTheme.TRADING_LIGHT: ChartTheme.WHITE,
            VisualizationTheme.MINIMAL: ChartTheme.WHITE,
            VisualizationTheme.COLORFUL: ChartTheme.NEURAL
        }
        
        return ChartConfig(
            theme=theme_mapping.get(self.config.theme, ChartTheme.NEURAL),
            width=self.config.default_width,
            height=self.config.default_height
        )
        
    def create_comprehensive_performance_dashboard(
        self, 
        returns_data: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, go.Figure]:
        """Create comprehensive performance dashboard with multiple charts."""
        
        dashboard_charts = {}
        
        # P&L Performance Chart
        pnl_fig = self.performance_chart.create_pnl_chart(
            returns_data, benchmark_returns, "Portfolio Performance Analysis"
        )
        dashboard_charts['performance'] = pnl_fig
        
        # Drawdown Analysis
        drawdown_fig = self.performance_chart.create_drawdown_chart(
            returns_data, "Portfolio Drawdown Analysis"
        )
        dashboard_charts['drawdown'] = drawdown_fig
        
        # Daily P&L
        daily_pnl = returns_data * 10000  # Assuming $10k portfolio for demo
        daily_pnl_fig = self.pnl_chart.create_daily_pnl_chart(
            daily_pnl, "Daily P&L Analysis"
        )
        dashboard_charts['daily_pnl'] = daily_pnl_fig
        
        # Underwater Chart
        underwater_fig = self.drawdown_chart.create_underwater_chart(
            returns_data, "Underwater Equity Curve"
        )
        dashboard_charts['underwater'] = underwater_fig
        
        # Performance Metrics Comparison (if benchmark provided)
        if benchmark_returns is not None:
            strategy_metrics = self.performance_calculator.calculate_comprehensive_metrics(returns_data)
            benchmark_metrics = self.performance_calculator.calculate_comprehensive_metrics(benchmark_returns)
            
            metrics_comparison = {
                'Strategy': strategy_metrics,
                'Benchmark': benchmark_metrics
            }
            
            comparison_fig = self.performance_chart.create_metrics_comparison_chart(
                metrics_comparison, "Strategy vs Benchmark Comparison"
            )
            dashboard_charts['comparison'] = comparison_fig
        
        # Save charts if requested
        if save_path and self.config.save_charts:
            self._save_dashboard_charts(dashboard_charts, save_path)
            
        # Show charts if auto_show enabled
        if self.config.auto_show:
            self._show_dashboard_charts(dashboard_charts)
            
        return dashboard_charts
        
    def create_rolling_metrics_analysis(
        self, 
        returns_data: pd.Series,
        window: int = 252,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """Create rolling performance metrics analysis."""
        
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=[
                "Rolling Sharpe Ratio", 
                "Rolling Volatility", 
                "Rolling Max Drawdown",
                "Rolling Win Rate (if applicable)"
            ],
            vertical_spacing=0.06
        )
        
        # Rolling Sharpe Ratio
        rolling_sharpe = returns_data.rolling(window=window).apply(
            lambda x: self.performance_calculator._calculate_sharpe_ratio(x, 0.02)[1]
        )
        
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe,
                name="Rolling Sharpe",
                line=dict(color='#00ff88', width=2)
            ),
            row=1, col=1
        )
        
        # Rolling Volatility
        rolling_vol = returns_data.rolling(window=window).std() * np.sqrt(252)
        
        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol,
                name="Rolling Volatility",
                line=dict(color='#ff4444', width=2),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Rolling Max Drawdown
        cumulative_returns = (1 + returns_data).cumprod()
        rolling_max = cumulative_returns.rolling(window=window).max()
        rolling_drawdown = (cumulative_returns / rolling_max - 1).rolling(window=window).min()
        
        fig.add_trace(
            go.Scatter(
                x=rolling_drawdown.index,
                y=rolling_drawdown,
                name="Rolling Max DD",
                line=dict(color='#ffaa00', width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 170, 0, 0.2)',
                showlegend=False
            ),
            row=3, col=1
        )
        
        # Rolling Win Rate (simplified - assumes positive returns are "wins")
        rolling_win_rate = returns_data.rolling(window=window).apply(
            lambda x: (x > 0).sum() / len(x)
        )
        
        fig.add_trace(
            go.Scatter(
                x=rolling_win_rate.index,
                y=rolling_win_rate,
                name="Rolling Win Rate",
                line=dict(color='#4488ff', width=2),
                showlegend=False
            ),
            row=4, col=1
        )
        
        # Add benchmark lines
        fig.add_hline(y=1.0, line_dash="dash", line_color="white", opacity=0.5, row=1, col=1)
        fig.add_hline(y=0.5, line_dash="dash", line_color="white", opacity=0.5, row=4, col=1)
        
        # Update layout
        fig.update_layout(
            title="Rolling Performance Metrics Analysis",
            height=800
        )
        
        fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
        fig.update_yaxes(title_text="Volatility", row=2, col=1, tickformat='.1%')
        fig.update_yaxes(title_text="Max Drawdown", row=3, col=1, tickformat='.1%')
        fig.update_yaxes(title_text="Win Rate", row=4, col=1, tickformat='.1%')
        fig.update_xaxes(title_text="Date", row=4, col=1)
        
        themed_fig = self.chart_manager.apply_theme(fig)
        
        if save_path and self.config.save_charts:
            self._save_chart(themed_fig, save_path, "rolling_metrics")
            
        if self.config.auto_show:
            themed_fig.show()
            
        return themed_fig
        
    def create_portfolio_allocation_dashboard(
        self, 
        positions: Dict[str, Dict[str, Any]],
        save_path: Optional[str] = None
    ) -> Dict[str, go.Figure]:
        """Create portfolio allocation visualization dashboard."""
        
        allocation_charts = {}
        
        # Calculate allocation data
        total_value = sum(pos.get('market_value', 0) for pos in positions.values())
        allocation_data = {
            market_id: pos.get('market_value', 0) 
            for market_id, pos in positions.items()
        }
        
        # Allocation pie chart
        allocation_fig = self.pnl_chart.create_portfolio_allocation_chart(
            allocation_data, "Current Portfolio Allocation"
        )
        allocation_charts['allocation'] = allocation_fig
        
        # Position P&L chart
        pnl_data = {
            market_id: pos.get('pnl', 0)
            for market_id, pos in positions.items()
        }
        
        if any(pnl != 0 for pnl in pnl_data.values()):
            pnl_fig = go.Figure(data=[
                go.Bar(
                    x=list(pnl_data.keys()),
                    y=list(pnl_data.values()),
                    marker_color=['#00ff88' if pnl >= 0 else '#ff4444' for pnl in pnl_data.values()],
                    text=[f"${pnl:,.0f}" for pnl in pnl_data.values()],
                    textposition='outside'
                )
            ])
            
            pnl_fig.update_layout(
                title="Position P&L Breakdown",
                xaxis_title="Market",
                yaxis_title="P&L ($)",
                yaxis_tickformat='$,.0f'
            )
            
            allocation_charts['position_pnl'] = self.chart_manager.apply_theme(pnl_fig)
        
        # Risk exposure by category (if available)
        if any('category' in pos for pos in positions.values()):
            category_exposure = {}
            for market_id, pos in positions.items():
                category = pos.get('category', 'Unknown')
                value = pos.get('market_value', 0)
                category_exposure[category] = category_exposure.get(category, 0) + value
                
            exposure_fig = go.Figure(data=[
                go.Bar(
                    x=list(category_exposure.keys()),
                    y=list(category_exposure.values()),
                    marker_color='#4488ff'
                )
            ])
            
            exposure_fig.update_layout(
                title="Risk Exposure by Category",
                xaxis_title="Category", 
                yaxis_title="Exposure ($)",
                yaxis_tickformat='$,.0f'
            )
            
            allocation_charts['category_exposure'] = self.chart_manager.apply_theme(exposure_fig)
        
        if save_path and self.config.save_charts:
            self._save_dashboard_charts(allocation_charts, save_path + "_allocation")
            
        if self.config.auto_show:
            self._show_dashboard_charts(allocation_charts)
            
        return allocation_charts
        
    def _save_dashboard_charts(self, charts: Dict[str, go.Figure], base_path: str):
        """Save dashboard charts to files."""
        os.makedirs(os.path.dirname(base_path) if os.path.dirname(base_path) else self.config.output_directory, exist_ok=True)
        
        for chart_name, fig in charts.items():
            self._save_chart(fig, base_path, chart_name)
            
    def _save_chart(self, fig: go.Figure, base_path: str, chart_name: str):
        """Save individual chart."""
        file_path = f"{base_path}_{chart_name}.{self.config.export_format.value}"
        
        if self.config.export_format == ExportFormat.HTML:
            fig.write_html(file_path)
        elif self.config.export_format == ExportFormat.PDF:
            fig.write_image(file_path, format="pdf")
        else:
            fig.write_image(file_path)
            
        logger.info(f"Saved chart: {file_path}")
        
    def _show_dashboard_charts(self, charts: Dict[str, go.Figure]):
        """Show dashboard charts interactively."""
        for chart_name, fig in charts.items():
            logger.info(f"Displaying {chart_name} chart")
            fig.show()


class RiskVisualizer:
    """Specialized visualizer for risk analysis and monitoring."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize risk visualizer."""
        if not CHARTS_AVAILABLE:
            raise ImportError("Chart components not available. Install plotly to use RiskVisualizer.")
            
        self.config = config or VisualizationConfig()
        self.chart_manager = ChartManager(self._get_chart_config())
        self.risk_chart = RiskChart(self.chart_manager)
        self.correlation_chart = CorrelationChart(self.chart_manager)
        
    def _get_chart_config(self) -> ChartConfig:
        """Get chart configuration."""
        return ChartConfig(
            theme=ChartTheme.NEURAL,
            width=self.config.default_width,
            height=self.config.default_height
        )
        
    def create_risk_dashboard(
        self, 
        returns_data: pd.Series,
        positions: Optional[Dict] = None,
        risk_limits: Optional[Dict] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, go.Figure]:
        """Create comprehensive risk analysis dashboard."""
        
        risk_charts = {}
        
        # VaR Analysis
        var_fig = self.risk_chart.create_var_chart(
            returns_data, title="Value at Risk Analysis"
        )
        risk_charts['var'] = var_fig
        
        # Risk-Return Scatter (if positions provided)
        if positions:
            position_returns = []
            position_risks = []
            position_names = []
            
            for market_id, pos in positions.items():
                if 'returns' in pos and 'volatility' in pos:
                    position_returns.append(pos['returns'])
                    position_risks.append(pos['volatility'])
                    position_names.append(market_id)
                    
            if position_returns:
                risk_return_fig = go.Figure(data=[
                    go.Scatter(
                        x=position_risks,
                        y=position_returns,
                        mode='markers+text',
                        text=position_names,
                        textposition='top center',
                        marker=dict(
                            size=10,
                            color=position_returns,
                            colorscale='RdYlGn',
                            showscale=True,
                            colorbar=dict(title="Returns")
                        ),
                        hovertemplate='<b>%{text}</b><br>Risk: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
                    )
                ])
                
                risk_return_fig.update_layout(
                    title="Risk-Return Analysis by Position",
                    xaxis_title="Risk (Volatility)",
                    yaxis_title="Returns",
                    xaxis_tickformat='.1%',
                    yaxis_tickformat='.1%'
                )
                
                risk_charts['risk_return'] = self.chart_manager.apply_theme(risk_return_fig)
        
        # Risk Limits Monitoring (if provided)
        if risk_limits:
            self._add_risk_limits_chart(risk_charts, risk_limits)
            
        if save_path and self.config.save_charts:
            self._save_dashboard_charts(risk_charts, save_path)
            
        if self.config.auto_show:
            self._show_dashboard_charts(risk_charts)
            
        return risk_charts
        
    def create_correlation_analysis(
        self, 
        returns_matrix: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> Dict[str, go.Figure]:
        """Create correlation analysis dashboard."""
        
        correlation_charts = {}
        
        # Correlation heatmap
        correlation_matrix = returns_matrix.corr()
        heatmap_fig = self.risk_chart.create_correlation_heatmap(
            correlation_matrix, "Asset Correlation Matrix"
        )
        correlation_charts['heatmap'] = heatmap_fig
        
        # Rolling correlation for key pairs
        if len(returns_matrix.columns) >= 2:
            asset1 = returns_matrix.columns[0]
            asset2 = returns_matrix.columns[1]
            
            rolling_corr_fig = self.correlation_chart.create_rolling_correlation_chart(
                returns_matrix[asset1], returns_matrix[asset2], 
                asset1_name=asset1, asset2_name=asset2
            )
            correlation_charts['rolling_correlation'] = rolling_corr_fig
        
        if save_path and self.config.save_charts:
            self._save_dashboard_charts(correlation_charts, save_path)
            
        if self.config.auto_show:
            self._show_dashboard_charts(correlation_charts)
            
        return correlation_charts
        
    def _add_risk_limits_chart(self, charts: Dict[str, go.Figure], risk_limits: Dict):
        """Add risk limits monitoring chart."""
        
        # Implementation would create charts showing current risk metrics vs limits
        # This is a placeholder for the concept
        
        limit_names = list(risk_limits.keys())
        current_values = [risk_limits[name].get('current', 0) for name in limit_names]
        limit_values = [risk_limits[name].get('limit', 1) for name in limit_names]
        
        limits_fig = go.Figure(data=[
            go.Bar(name='Current', x=limit_names, y=current_values, marker_color='#4488ff'),
            go.Bar(name='Limit', x=limit_names, y=limit_values, marker_color='#ff4444', opacity=0.7)
        ])
        
        limits_fig.update_layout(
            title="Risk Limits Monitoring",
            xaxis_title="Risk Metric",
            yaxis_title="Value",
            barmode='group'
        )
        
        charts['risk_limits'] = self.chart_manager.apply_theme(limits_fig)
        
    def _save_dashboard_charts(self, charts: Dict[str, go.Figure], base_path: str):
        """Save risk dashboard charts."""
        # Similar implementation to PerformanceVisualizer
        pass
        
    def _show_dashboard_charts(self, charts: Dict[str, go.Figure]):
        """Show risk dashboard charts."""
        # Similar implementation to PerformanceVisualizer 
        pass


class StrategyVisualizer:
    """Specialized visualizer for strategy analysis and comparison."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize strategy visualizer."""
        if not CHARTS_AVAILABLE:
            raise ImportError("Chart components not available. Install plotly to use StrategyVisualizer.")
            
        self.config = config or VisualizationConfig()
        self.chart_manager = ChartManager(self._get_chart_config())
        self.performance_calculator = PerformanceCalculator()
        
    def _get_chart_config(self) -> ChartConfig:
        """Get chart configuration."""
        return ChartConfig(
            theme=ChartTheme.NEURAL,
            width=self.config.default_width,
            height=self.config.default_height
        )
        
    def create_strategy_comparison_dashboard(
        self, 
        strategies_data: Dict[str, Dict[str, Any]],
        save_path: Optional[str] = None
    ) -> Dict[str, go.Figure]:
        """Create strategy comparison dashboard."""
        
        comparison_charts = {}
        
        # Calculate metrics for each strategy
        strategies_metrics = {}
        for name, data in strategies_data.items():
            if 'returns' in data:
                metrics = self.performance_calculator.calculate_comprehensive_metrics(
                    returns=data['returns']
                )
                strategies_metrics[name] = metrics
        
        if strategies_metrics:
            # Metrics comparison radar chart
            from .charts import PerformanceChart
            performance_chart = PerformanceChart(self.chart_manager)
            
            radar_fig = performance_chart.create_metrics_comparison_chart(
                strategies_metrics, "Strategy Performance Comparison"
            )
            comparison_charts['metrics_comparison'] = radar_fig
            
            # Cumulative returns comparison
            returns_comparison_fig = go.Figure()
            
            for name, data in strategies_data.items():
                if 'returns' in data:
                    cumulative_returns = (1 + data['returns']).cumprod()
                    returns_comparison_fig.add_trace(
                        go.Scatter(
                            x=cumulative_returns.index,
                            y=cumulative_returns,
                            name=name,
                            line=dict(width=2)
                        )
                    )
            
            returns_comparison_fig.update_layout(
                title="Cumulative Returns Comparison",
                xaxis_title="Date",
                yaxis_title="Cumulative Return",
                yaxis_tickformat='.1%'
            )
            
            comparison_charts['returns_comparison'] = self.chart_manager.apply_theme(returns_comparison_fig)
        
        if save_path and self.config.save_charts:
            self._save_dashboard_charts(comparison_charts, save_path)
            
        if self.config.auto_show:
            self._show_dashboard_charts(comparison_charts)
            
        return comparison_charts
        
    def _save_dashboard_charts(self, charts: Dict[str, go.Figure], base_path: str):
        """Save strategy comparison charts."""
        pass
        
    def _show_dashboard_charts(self, charts: Dict[str, go.Figure]):
        """Show strategy comparison charts."""
        pass


class MarketVisualizer:
    """Specialized visualizer for market data and signal analysis."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize market visualizer."""
        if not CHARTS_AVAILABLE:
            raise ImportError("Chart components not available. Install plotly to use MarketVisualizer.")
            
        self.config = config or VisualizationConfig()
        self.chart_manager = ChartManager(self._get_chart_config())
        self.market_chart = MarketChart(self.chart_manager)
        
    def _get_chart_config(self) -> ChartConfig:
        """Get chart configuration."""
        return ChartConfig(
            theme=ChartTheme.NEURAL,
            width=self.config.default_width,
            height=self.config.default_height
        )
        
    def create_market_analysis_dashboard(
        self, 
        price_data: pd.DataFrame,
        signals: Optional[List[Signal]] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, go.Figure]:
        """Create market analysis dashboard."""
        
        market_charts = {}
        
        # Price chart with signals
        price_fig = self.market_chart.create_price_chart(
            price_data, signals, "Market Price Analysis"
        )
        market_charts['price_analysis'] = price_fig
        
        if save_path and self.config.save_charts:
            self._save_dashboard_charts(market_charts, save_path)
            
        if self.config.auto_show:
            self._show_dashboard_charts(market_charts)
            
        return market_charts
        
    def _save_dashboard_charts(self, charts: Dict[str, go.Figure], base_path: str):
        """Save market analysis charts."""
        pass
        
    def _show_dashboard_charts(self, charts: Dict[str, go.Figure]):
        """Show market analysis charts."""
        pass
