"""
Interactive Chart Generation for Neural SDK

Provides comprehensive Plotly-based charting capabilities for:
- Performance analytics and P&L visualization
- Risk metrics and drawdown analysis  
- Market data and price action charts
- Strategy comparison and correlation analysis
- Portfolio allocation and exposure tracking
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import logging

from ..analysis.metrics import PerformanceMetrics
from ..strategy.base import Signal, SignalType

logger = logging.getLogger(__name__)


class ChartTheme(Enum):
    """Chart styling themes."""
    DARK = "plotly_dark"
    WHITE = "plotly_white"
    NEURAL = "neural_custom"
    TRADING = "trading_style"


class ChartType(Enum):
    """Available chart types."""
    LINE = "line"
    CANDLESTICK = "candlestick"
    BAR = "bar"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    HISTOGRAM = "histogram"
    BOX = "box"


@dataclass
class ChartConfig:
    """Configuration for chart generation."""
    theme: ChartTheme = ChartTheme.NEURAL
    width: int = 1200
    height: int = 600
    show_grid: bool = True
    show_legend: bool = True
    interactive: bool = True
    export_format: str = "html"
    font_size: int = 12
    line_width: int = 2


class ChartManager:
    """
    Central manager for all chart generation capabilities.
    
    Provides a unified interface for creating interactive visualizations
    with consistent styling and advanced features.
    """
    
    def __init__(self, config: Optional[ChartConfig] = None):
        """Initialize ChartManager with configuration."""
        self.config = config or ChartConfig()
        self._setup_themes()
        
    def _setup_themes(self):
        """Setup custom chart themes."""
        self.themes = {
            ChartTheme.NEURAL: {
                'layout': {
                    'paper_bgcolor': '#1a1a1a',
                    'plot_bgcolor': '#2d2d2d',
                    'font': {'color': '#ffffff', 'size': self.config.font_size}
                },
                'colors': ['#00ff88', '#ff4444', '#4488ff', '#ffaa00', '#ff88ff'],
                'gridcolor': '#404040',
                'zerolinecolor': '#606060'
            },
            ChartTheme.TRADING: {
                'layout': {
                    'paper_bgcolor': '#0d1421',
                    'plot_bgcolor': '#1e2329',
                    'font': {'color': '#c0c0c0', 'size': self.config.font_size}
                },
                'colors': ['#2eca6a', '#f84960', '#fcd535', '#ad7bee', '#f89880'],
                'gridcolor': '#2b3139',
                'zerolinecolor': '#404040'
            }
        }
    
    def apply_theme(self, fig: go.Figure, theme: ChartTheme = None) -> go.Figure:
        """Apply theme styling to a figure."""
        theme = theme or self.config.theme
        
        if theme in self.themes:
            theme_config = self.themes[theme]
            fig.update_layout(**theme_config['layout'])
            fig.update_layout(colorway=theme_config['colors'])
        else:
            fig.update_layout(template=theme.value)
            
        # Common styling
        fig.update_layout(
            width=self.config.width,
            height=self.config.height,
            showlegend=self.config.show_legend,
            hovermode='x unified'
        )
        
        if self.config.show_grid:
            # Apply grid styling from theme
            grid_color = self.themes.get(theme, {}).get('gridcolor', '#404040')
            zero_color = self.themes.get(theme, {}).get('zerolinecolor', '#606060')
            
            fig.update_xaxes(showgrid=True, gridcolor=grid_color, zerolinecolor=zero_color)
            fig.update_yaxes(showgrid=True, gridcolor=grid_color, zerolinecolor=zero_color)
            
        return fig


class PerformanceChart:
    """Charts for performance analysis and P&L visualization."""
    
    def __init__(self, chart_manager: ChartManager):
        self.chart_manager = chart_manager
        
    def create_pnl_chart(
        self, 
        returns_data: pd.Series,
        benchmark_data: Optional[pd.Series] = None,
        title: str = "Portfolio P&L Performance"
    ) -> go.Figure:
        """Create comprehensive P&L performance chart."""
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=["Cumulative Returns", "Daily Returns", "Rolling Volatility"],
            vertical_spacing=0.08,
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Cumulative returns
        cumulative_returns = (1 + returns_data).cumprod()
        fig.add_trace(
            go.Scatter(
                x=returns_data.index,
                y=cumulative_returns,
                name="Strategy",
                line=dict(color='#00ff88', width=self.chart_manager.config.line_width),
                hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Return: %{y:.2%}<extra></extra>'
            ),
            row=1, col=1
        )
        
        if benchmark_data is not None:
            benchmark_cumulative = (1 + benchmark_data).cumprod()
            fig.add_trace(
                go.Scatter(
                    x=benchmark_data.index,
                    y=benchmark_cumulative,
                    name="Benchmark",
                    line=dict(color='#666666', width=1, dash='dash'),
                    hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Return: %{y:.2%}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Daily returns
        colors = ['#00ff88' if x >= 0 else '#ff4444' for x in returns_data]
        fig.add_trace(
            go.Bar(
                x=returns_data.index,
                y=returns_data,
                name="Daily Returns",
                marker_color=colors,
                opacity=0.7,
                hovertemplate='<b>Daily Return</b><br>Date: %{x}<br>Return: %{y:.2%}<extra></extra>',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Rolling volatility
        rolling_vol = returns_data.rolling(window=30).std() * np.sqrt(252)
        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol,
                name="30-Day Volatility",
                line=dict(color='#ffaa00', width=2),
                hovertemplate='<b>Rolling Volatility</b><br>Date: %{x}<br>Volatility: %{y:.2%}<extra></extra>',
                showlegend=False
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(title=title)
        fig.update_yaxes(title_text="Cumulative Return", row=1, col=1, tickformat='.1%')
        fig.update_yaxes(title_text="Daily Return", row=2, col=1, tickformat='.1%')
        fig.update_yaxes(title_text="Volatility", row=3, col=1, tickformat='.1%')
        fig.update_xaxes(title_text="Date", row=3, col=1)
        
        return self.chart_manager.apply_theme(fig)
    
    def create_drawdown_chart(
        self, 
        returns_data: pd.Series,
        title: str = "Portfolio Drawdown Analysis"
    ) -> go.Figure:
        """Create drawdown analysis chart."""
        
        # Calculate running maximum and drawdown
        cumulative_returns = (1 + returns_data).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns / running_max) - 1
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=["Portfolio Value & Peak", "Drawdown"],
            vertical_spacing=0.1,
            row_heights=[0.6, 0.4]
        )
        
        # Portfolio value and running peak
        fig.add_trace(
            go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns,
                name="Portfolio Value",
                line=dict(color='#00ff88', width=2),
                fill='tonexty',
                fillcolor='rgba(0, 255, 136, 0.1)'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=running_max.index,
                y=running_max,
                name="All-Time High",
                line=dict(color='#ffffff', width=1, dash='dot'),
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Drawdown
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown,
                name="Drawdown",
                line=dict(color='#ff4444', width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 68, 68, 0.3)',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="solid", line_color="white", opacity=0.3, row=2, col=1)
        
        # Update layout
        fig.update_layout(title=title)
        fig.update_yaxes(title_text="Portfolio Value", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown", row=2, col=1, tickformat='.1%')
        fig.update_xaxes(title_text="Date", row=2, col=1)
        
        return self.chart_manager.apply_theme(fig)
    
    def create_metrics_comparison_chart(
        self, 
        strategies_metrics: Dict[str, PerformanceMetrics],
        title: str = "Strategy Performance Comparison"
    ) -> go.Figure:
        """Create performance metrics comparison chart."""
        
        metrics_names = [
            'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 
            'max_drawdown', 'win_rate', 'total_return'
        ]
        
        strategies = list(strategies_metrics.keys())
        
        fig = go.Figure()
        
        for i, strategy in enumerate(strategies):
            metrics = strategies_metrics[strategy]
            values = [
                getattr(metrics, metric_name, 0) for metric_name in metrics_names
            ]
            
            # Normalize some values for better visualization
            normalized_values = values.copy()
            normalized_values[3] = abs(normalized_values[3])  # Max drawdown (make positive)
            normalized_values[5] = normalized_values[5] * 100  # Total return as percentage
            
            fig.add_trace(go.Scatterpolar(
                r=normalized_values,
                theta=['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 
                       'Max Drawdown', 'Win Rate', 'Total Return (%)'],
                fill='toself',
                name=strategy,
                opacity=0.6
            ))
        
        fig.update_layout(
            title=title,
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max([max([getattr(m, metric, 0) for metric in metrics_names]) 
                                  for m in strategies_metrics.values()]) * 1.1]
                )
            ),
            showlegend=True
        )
        
        return self.chart_manager.apply_theme(fig)


class RiskChart:
    """Charts for risk analysis and monitoring."""
    
    def __init__(self, chart_manager: ChartManager):
        self.chart_manager = chart_manager
        
    def create_var_chart(
        self, 
        returns_data: pd.Series,
        confidence_levels: List[float] = [0.95, 0.99],
        title: str = "Value at Risk Analysis"
    ) -> go.Figure:
        """Create Value at Risk visualization."""
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=["Return Distribution", "Rolling VaR"],
            vertical_spacing=0.15
        )
        
        # Return distribution histogram
        fig.add_trace(
            go.Histogram(
                x=returns_data,
                nbinsx=50,
                name="Return Distribution",
                opacity=0.7,
                marker_color='#4488ff',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Add VaR lines
        colors = ['#ff4444', '#ff8800']
        for i, conf_level in enumerate(confidence_levels):
            var_value = returns_data.quantile(1 - conf_level)
            fig.add_vline(
                x=var_value, 
                line_dash="dash", 
                line_color=colors[i],
                annotation_text=f"VaR {conf_level:.0%}: {var_value:.2%}",
                row=1, col=1
            )
        
        # Rolling VaR
        window = 60  # 60-day rolling
        for i, conf_level in enumerate(confidence_levels):
            rolling_var = returns_data.rolling(window=window).quantile(1 - conf_level)
            fig.add_trace(
                go.Scatter(
                    x=rolling_var.index,
                    y=rolling_var,
                    name=f"VaR {conf_level:.0%}",
                    line=dict(color=colors[i], width=2),
                    hovertemplate=f'<b>VaR {conf_level:.0%}</b><br>Date: %{{x}}<br>VaR: %{{y:.2%}}<extra></extra>'
                ),
                row=2, col=1
            )
        
        fig.update_layout(title=title)
        fig.update_xaxes(title_text="Daily Return", row=1, col=1, tickformat='.1%')
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="VaR", row=2, col=1, tickformat='.1%')
        
        return self.chart_manager.apply_theme(fig)
    
    def create_correlation_heatmap(
        self, 
        correlation_matrix: pd.DataFrame,
        title: str = "Asset Correlation Heatmap"
    ) -> go.Figure:
        """Create correlation heatmap."""
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            hovertemplate='<b>Correlation</b><br>%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>',
            colorbar=dict(title="Correlation", tickformat='.2f')
        ))
        
        # Add correlation values as text
        annotations = []
        for i, row in enumerate(correlation_matrix.index):
            for j, col in enumerate(correlation_matrix.columns):
                annotations.append(
                    dict(
                        x=col, y=row,
                        text=f"{correlation_matrix.iloc[i, j]:.2f}",
                        showarrow=False,
                        font=dict(color="white" if abs(correlation_matrix.iloc[i, j]) > 0.5 else "black")
                    )
                )
        
        fig.update_layout(
            title=title,
            annotations=annotations,
            xaxis_title="Assets",
            yaxis_title="Assets"
        )
        
        return self.chart_manager.apply_theme(fig)


class MarketChart:
    """Charts for market data and price action visualization."""
    
    def __init__(self, chart_manager: ChartManager):
        self.chart_manager = chart_manager
        
    def create_price_chart(
        self, 
        price_data: pd.DataFrame,
        signals: Optional[List[Signal]] = None,
        title: str = "Market Price Analysis"
    ) -> go.Figure:
        """Create comprehensive price chart with signals."""
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=["Price & Signals", "Volume", "Spread"],
            vertical_spacing=0.05,
            row_heights=[0.6, 0.2, 0.2],
            shared_xaxes=True
        )
        
        # Price line
        if 'last' in price_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=price_data.index,
                    y=price_data['last'],
                    name="Last Price",
                    line=dict(color='#00ff88', width=2),
                    hovertemplate='<b>Price</b><br>Time: %{x}<br>Price: %{y:.3f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Bid/Ask spread
        if 'bid' in price_data.columns and 'ask' in price_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=price_data.index,
                    y=price_data['bid'],
                    name="Bid",
                    line=dict(color='#ff4444', width=1),
                    showlegend=False
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=price_data.index,
                    y=price_data['ask'],
                    name="Ask",
                    line=dict(color='#4488ff', width=1),
                    fill='tonexty',
                    fillcolor='rgba(68, 136, 255, 0.1)',
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Spread chart
            spread = price_data['ask'] - price_data['bid']
            fig.add_trace(
                go.Scatter(
                    x=price_data.index,
                    y=spread,
                    name="Bid-Ask Spread",
                    line=dict(color='#ffaa00', width=1),
                    fill='tozeroy',
                    fillcolor='rgba(255, 170, 0, 0.2)',
                    showlegend=False
                ),
                row=3, col=1
            )
        
        # Volume
        if 'volume' in price_data.columns:
            colors = ['#00ff88' if i % 2 == 0 else '#4488ff' for i in range(len(price_data))]
            fig.add_trace(
                go.Bar(
                    x=price_data.index,
                    y=price_data['volume'],
                    name="Volume",
                    marker_color=colors,
                    opacity=0.6,
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Add trading signals
        if signals:
            buy_signals = [s for s in signals if s.signal_type == SignalType.BUY_YES]
            sell_signals = [s for s in signals if s.signal_type == SignalType.SELL_YES]
            
            if buy_signals:
                buy_times = [s.timestamp for s in buy_signals]
                buy_prices = [price_data.loc[s.timestamp, 'last'] if s.timestamp in price_data.index else None 
                             for s in buy_signals]
                buy_prices = [p for p in buy_prices if p is not None]
                
                if buy_prices:
                    fig.add_trace(
                        go.Scatter(
                            x=buy_times[:len(buy_prices)],
                            y=buy_prices,
                            mode='markers',
                            name="Buy Signals",
                            marker=dict(color='#00ff00', size=10, symbol='triangle-up'),
                            hovertemplate='<b>Buy Signal</b><br>Time: %{x}<br>Price: %{y:.3f}<extra></extra>'
                        ),
                        row=1, col=1
                    )
            
            if sell_signals:
                sell_times = [s.timestamp for s in sell_signals]
                sell_prices = [price_data.loc[s.timestamp, 'last'] if s.timestamp in price_data.index else None 
                              for s in sell_signals]
                sell_prices = [p for p in sell_prices if p is not None]
                
                if sell_prices:
                    fig.add_trace(
                        go.Scatter(
                            x=sell_times[:len(sell_prices)],
                            y=sell_prices,
                            mode='markers',
                            name="Sell Signals", 
                            marker=dict(color='#ff0000', size=10, symbol='triangle-down'),
                            hovertemplate='<b>Sell Signal</b><br>Time: %{x}<br>Price: %{y:.3f}<extra></extra>'
                        ),
                        row=1, col=1
                    )
        
        # Update layout
        fig.update_layout(title=title)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="Spread", row=3, col=1)
        fig.update_xaxes(title_text="Time", row=3, col=1)
        
        return self.chart_manager.apply_theme(fig)


class PnLChart:
    """Specialized P&L and portfolio tracking charts."""
    
    def __init__(self, chart_manager: ChartManager):
        self.chart_manager = chart_manager
        
    def create_portfolio_allocation_chart(
        self, 
        allocation_data: Dict[str, float],
        title: str = "Portfolio Allocation"
    ) -> go.Figure:
        """Create portfolio allocation pie chart."""
        
        fig = go.Figure(data=[go.Pie(
            labels=list(allocation_data.keys()),
            values=list(allocation_data.values()),
            hole=.3,
            hovertemplate='<b>%{label}</b><br>Allocation: %{percent}<br>Value: $%{value:,.0f}<extra></extra>',
            textfont_size=12,
            marker=dict(line=dict(color='#ffffff', width=2))
        )])
        
        fig.update_layout(
            title=title,
            annotations=[dict(text='Portfolio', x=0.5, y=0.5, font_size=16, showarrow=False)]
        )
        
        return self.chart_manager.apply_theme(fig)
    
    def create_daily_pnl_chart(
        self, 
        daily_pnl: pd.Series,
        title: str = "Daily P&L"
    ) -> go.Figure:
        """Create daily P&L bar chart."""
        
        colors = ['#00ff88' if x >= 0 else '#ff4444' for x in daily_pnl]
        
        fig = go.Figure(data=[go.Bar(
            x=daily_pnl.index,
            y=daily_pnl,
            marker_color=colors,
            name="Daily P&L",
            hovertemplate='<b>Daily P&L</b><br>Date: %{x}<br>P&L: $%{y:,.2f}<extra></extra>'
        )])
        
        # Add zero line
        fig.add_hline(y=0, line_dash="solid", line_color="white", opacity=0.3)
        
        # Add cumulative P&L line
        cumulative_pnl = daily_pnl.cumsum()
        fig.add_trace(go.Scatter(
            x=cumulative_pnl.index,
            y=cumulative_pnl,
            name="Cumulative P&L",
            line=dict(color='#ffaa00', width=3),
            yaxis='y2',
            hovertemplate='<b>Cumulative P&L</b><br>Date: %{x}<br>Total: $%{y:,.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis=dict(title="Daily P&L", side="left"),
            yaxis2=dict(title="Cumulative P&L", side="right", overlaying="y"),
            hovermode='x unified'
        )
        
        return self.chart_manager.apply_theme(fig)


class DrawdownChart:
    """Specialized drawdown analysis charts."""
    
    def __init__(self, chart_manager: ChartManager):
        self.chart_manager = chart_manager
        
    def create_underwater_chart(
        self, 
        returns_data: pd.Series,
        title: str = "Underwater Equity Curve"
    ) -> go.Figure:
        """Create underwater equity curve showing drawdown periods."""
        
        # Calculate drawdown
        cumulative_returns = (1 + returns_data).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns / running_max) - 1
        
        fig = go.Figure()
        
        # Underwater curve
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown,
            fill='tozeroy',
            fillcolor='rgba(255, 68, 68, 0.5)',
            line=dict(color='#ff4444', width=2),
            name="Drawdown",
            hovertemplate='<b>Drawdown</b><br>Date: %{x}<br>Drawdown: %{y:.2%}<extra></extra>'
        ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="solid", line_color="white", opacity=0.5)
        
        # Highlight major drawdown periods (> 10%)
        major_drawdowns = drawdown[drawdown < -0.1]
        if not major_drawdowns.empty:
            fig.add_trace(go.Scatter(
                x=major_drawdowns.index,
                y=major_drawdowns,
                mode='markers',
                name="Major Drawdowns (>10%)",
                marker=dict(color='#ff0000', size=8, symbol='x'),
                hovertemplate='<b>Major Drawdown</b><br>Date: %{x}<br>Drawdown: %{y:.2%}<extra></extra>'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Drawdown",
            yaxis_tickformat='.1%'
        )
        
        return self.chart_manager.apply_theme(fig)


class CorrelationChart:
    """Specialized correlation analysis charts."""
    
    def __init__(self, chart_manager: ChartManager):
        self.chart_manager = chart_manager
        
    def create_rolling_correlation_chart(
        self, 
        asset1_returns: pd.Series,
        asset2_returns: pd.Series,
        window: int = 60,
        asset1_name: str = "Asset 1",
        asset2_name: str = "Asset 2",
        title: str = "Rolling Correlation Analysis"
    ) -> go.Figure:
        """Create rolling correlation chart between two assets."""
        
        # Calculate rolling correlation
        rolling_corr = asset1_returns.rolling(window=window).corr(asset2_returns)
        
        fig = go.Figure()
        
        # Rolling correlation line
        fig.add_trace(go.Scatter(
            x=rolling_corr.index,
            y=rolling_corr,
            name=f"{asset1_name} vs {asset2_name}",
            line=dict(color='#00ff88', width=3),
            hovertemplate=f'<b>Correlation</b><br>Date: %{{x}}<br>{asset1_name} vs {asset2_name}: %{{y:.3f}}<extra></extra>'
        ))
        
        # Add correlation bands
        fig.add_hline(y=0.7, line_dash="dash", line_color="red", opacity=0.5, 
                     annotation_text="High Positive (0.7)")
        fig.add_hline(y=0.3, line_dash="dash", line_color="orange", opacity=0.5,
                     annotation_text="Moderate Positive (0.3)")  
        fig.add_hline(y=0, line_dash="solid", line_color="white", opacity=0.3)
        fig.add_hline(y=-0.3, line_dash="dash", line_color="orange", opacity=0.5,
                     annotation_text="Moderate Negative (-0.3)")
        fig.add_hline(y=-0.7, line_dash="dash", line_color="red", opacity=0.5,
                     annotation_text="High Negative (-0.7)")
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Correlation",
            yaxis=dict(range=[-1, 1], tickformat='.2f')
        )
        
        return self.chart_manager.apply_theme(fig)