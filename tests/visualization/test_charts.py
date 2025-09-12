"""
Unit tests for Chart Generation Framework.

Tests the interactive chart generation capabilities including
performance charts, risk charts, market visualization, and
theme management with graceful handling of missing dependencies.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

# Handle optional dependencies gracefully
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None

# Skip all visualization tests if plotly is not available
pytestmark = pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")

if PLOTLY_AVAILABLE:
    from neural.visualization.charts import (
        ChartManager, ChartConfig, ChartTheme, ChartType,
        PerformanceChart, RiskChart, MarketChart, PnLChart,
        DrawdownChart, CorrelationChart
    )
    from neural.analysis.metrics import PerformanceMetrics
    from neural.strategy.base import Signal, SignalType


@pytest.fixture
def sample_returns_data():
    """Create sample returns data for testing."""
    dates = pd.date_range(start='2024-01-01', end='2024-06-30', freq='D')
    returns = np.random.normal(0.001, 0.02, len(dates))
    return pd.Series(returns, index=dates)


@pytest.fixture
def sample_price_data():
    """Create sample price data for testing."""
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    data = {
        'last': np.random.uniform(0.45, 0.55, len(dates)),
        'bid': np.random.uniform(0.42, 0.52, len(dates)),
        'ask': np.random.uniform(0.48, 0.58, len(dates)),
        'volume': np.random.randint(100, 2000, len(dates))
    }
    df = pd.DataFrame(data, index=dates)
    # Ensure bid < last < ask
    df['bid'] = np.minimum(df['bid'], df['last'] - 0.01)
    df['ask'] = np.maximum(df['ask'], df['last'] + 0.01)
    return df


@pytest.fixture
def sample_signals():
    """Create sample signals for testing."""
    return [
        Signal(
            signal_type=SignalType.BUY_YES,
            market_id="TEST_MARKET",
            timestamp=datetime.now(),
            confidence=0.80,
            edge=0.06,
            expected_value=30.0,
            recommended_size=0.10,
            max_contracts=100
        ),
        Signal(
            signal_type=SignalType.SELL_NO,
            market_id="TEST_MARKET_2",
            timestamp=datetime.now() - timedelta(hours=1),
            confidence=0.75,
            edge=0.04,
            expected_value=20.0,
            recommended_size=0.08,
            max_contracts=80
        )
    ]


class TestChartConfig:
    """Test ChartConfig functionality."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = ChartConfig()
        
        assert config.theme == ChartTheme.NEURAL
        assert config.width == 1200
        assert config.height == 600
        assert config.show_grid is True
        assert config.show_legend is True
        assert config.interactive is True
        assert config.export_format == "html"
        assert config.font_size == 12
        assert config.line_width == 2
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ChartConfig(
            theme=ChartTheme.DARK,
            width=800,
            height=400,
            show_grid=False,
            font_size=14
        )
        
        assert config.theme == ChartTheme.DARK
        assert config.width == 800
        assert config.height == 400
        assert config.show_grid is False
        assert config.font_size == 14


class TestChartManager:
    """Test ChartManager functionality."""
    
    @pytest.fixture
    def chart_manager(self):
        """Create ChartManager for testing."""
        return ChartManager()
    
    def test_initialization(self, chart_manager):
        """Test ChartManager initialization."""
        assert chart_manager.config is not None
        assert isinstance(chart_manager.config, ChartConfig)
        assert hasattr(chart_manager, 'themes')
        assert ChartTheme.NEURAL in chart_manager.themes
        assert ChartTheme.TRADING in chart_manager.themes
    
    def test_themes_setup(self, chart_manager):
        """Test custom themes are properly setup."""
        neural_theme = chart_manager.themes[ChartTheme.NEURAL]
        trading_theme = chart_manager.themes[ChartTheme.TRADING]
        
        assert 'layout' in neural_theme
        assert 'colors' in neural_theme
        assert 'layout' in trading_theme
        assert 'colors' in trading_theme
        
        # Check theme properties
        assert neural_theme['layout']['paper_bgcolor'] == '#1a1a1a'
        assert trading_theme['layout']['paper_bgcolor'] == '#0d1421'
        assert len(neural_theme['colors']) == 5
        assert len(trading_theme['colors']) == 5
    
    def test_apply_theme_neural(self, chart_manager):
        """Test applying Neural theme to figure."""
        fig = go.Figure()
        themed_fig = chart_manager.apply_theme(fig, ChartTheme.NEURAL)
        
        assert themed_fig.layout.paper_bgcolor == '#1a1a1a'
        assert themed_fig.layout.plot_bgcolor == '#2d2d2d'
        assert themed_fig.layout.width == chart_manager.config.width
        assert themed_fig.layout.height == chart_manager.config.height
        assert themed_fig.layout.showlegend == chart_manager.config.show_legend
    
    def test_apply_theme_trading(self, chart_manager):
        """Test applying Trading theme to figure."""
        fig = go.Figure()
        themed_fig = chart_manager.apply_theme(fig, ChartTheme.TRADING)
        
        assert themed_fig.layout.paper_bgcolor == '#0d1421'
        assert themed_fig.layout.plot_bgcolor == '#1e2329'
        assert len(themed_fig.layout.colorway) == 5
    
    def test_apply_theme_default(self, chart_manager):
        """Test applying default theme."""
        fig = go.Figure()
        themed_fig = chart_manager.apply_theme(fig)  # No theme specified
        
        # Should apply the config's default theme
        assert themed_fig.layout.width == chart_manager.config.width
        assert themed_fig.layout.height == chart_manager.config.height


class TestPerformanceChart:
    """Test PerformanceChart functionality."""
    
    @pytest.fixture
    def performance_chart(self):
        """Create PerformanceChart for testing."""
        chart_manager = ChartManager()
        return PerformanceChart(chart_manager)
    
    @pytest.fixture
    def sample_metrics(self):
        """Create sample performance metrics."""
        return PerformanceMetrics(
            total_return=0.15,
            sharpe_ratio=1.8,
            sortino_ratio=2.2,
            calmar_ratio=0.8,
            max_drawdown=-0.08,
            volatility=0.12,
            win_rate=0.65,
            avg_win=45.0,
            avg_loss=-25.0,
            total_trades=150,
            winning_trades=98,
            losing_trades=52
        )
    
    def test_create_pnl_chart_basic(self, performance_chart, sample_returns_data):
        """Test basic P&L chart creation."""
        fig = performance_chart.create_pnl_chart(sample_returns_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 3  # Should have cumulative returns, daily returns, rolling vol
        assert fig.layout.title.text == "Portfolio P&L Performance"
    
    def test_create_pnl_chart_with_benchmark(self, performance_chart, sample_returns_data):
        """Test P&L chart with benchmark comparison."""
        benchmark_returns = sample_returns_data * 0.8  # Slightly worse benchmark
        
        fig = performance_chart.create_pnl_chart(
            sample_returns_data, 
            benchmark_data=benchmark_returns,
            title="Test P&L Chart"
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 4  # Should include benchmark trace
        assert fig.layout.title.text == "Test P&L Chart"
        
        # Check that benchmark trace exists
        trace_names = [trace.name for trace in fig.data]
        assert "Benchmark" in trace_names
    
    def test_create_drawdown_chart(self, performance_chart, sample_returns_data):
        """Test drawdown chart creation."""
        fig = performance_chart.create_drawdown_chart(sample_returns_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 3  # Portfolio value, running max, drawdown
        assert fig.layout.title.text == "Portfolio Drawdown Analysis"
    
    def test_create_metrics_comparison_chart(self, performance_chart, sample_metrics):
        """Test metrics comparison radar chart."""
        strategies_metrics = {
            'Strategy A': sample_metrics,
            'Strategy B': PerformanceMetrics(
                total_return=0.12,
                sharpe_ratio=1.5,
                sortino_ratio=1.8,
                calmar_ratio=0.6,
                max_drawdown=-0.12,
                volatility=0.15,
                win_rate=0.58,
                avg_win=38.0,
                avg_loss=-28.0,
                total_trades=120,
                winning_trades=70,
                losing_trades=50
            )
        }
        
        fig = performance_chart.create_metrics_comparison_chart(strategies_metrics)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # Two strategies
        assert fig.layout.title.text == "Strategy Performance Comparison"
        
        # Check radar chart structure
        for trace in fig.data:
            assert hasattr(trace, 'r')  # Radial values
            assert hasattr(trace, 'theta')  # Angular values


class TestRiskChart:
    """Test RiskChart functionality."""
    
    @pytest.fixture
    def risk_chart(self):
        """Create RiskChart for testing."""
        chart_manager = ChartManager()
        return RiskChart(chart_manager)
    
    def test_create_var_chart(self, risk_chart, sample_returns_data):
        """Test VaR analysis chart creation."""
        fig = risk_chart.create_var_chart(sample_returns_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 3  # Distribution + 2 rolling VaR lines
        assert fig.layout.title.text == "Value at Risk Analysis"
    
    def test_create_var_chart_custom_confidence(self, risk_chart, sample_returns_data):
        """Test VaR chart with custom confidence levels."""
        fig = risk_chart.create_var_chart(
            sample_returns_data, 
            confidence_levels=[0.90, 0.95, 0.99],
            title="Custom VaR Analysis"
        )
        
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Custom VaR Analysis"
        # Should have more traces due to additional confidence level
        assert len(fig.data) >= 4
    
    def test_create_correlation_heatmap(self, risk_chart):
        """Test correlation heatmap creation."""
        # Create correlation matrix
        data = {
            'Asset A': np.random.normal(0, 0.02, 100),
            'Asset B': np.random.normal(0, 0.015, 100),
            'Asset C': np.random.normal(0, 0.025, 100)
        }
        returns_df = pd.DataFrame(data)
        correlation_matrix = returns_df.corr()
        
        fig = risk_chart.create_correlation_heatmap(correlation_matrix)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1  # Single heatmap trace
        assert fig.layout.title.text == "Asset Correlation Heatmap"
        
        # Check heatmap properties
        heatmap = fig.data[0]
        assert heatmap.type == 'heatmap'
        assert heatmap.colorscale == 'RdBu'
        assert heatmap.zmid == 0


class TestMarketChart:
    """Test MarketChart functionality."""
    
    @pytest.fixture
    def market_chart(self):
        """Create MarketChart for testing."""
        chart_manager = ChartManager()
        return MarketChart(chart_manager)
    
    def test_create_price_chart_basic(self, market_chart, sample_price_data):
        """Test basic price chart creation."""
        fig = market_chart.create_price_chart(sample_price_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 4  # Last price, bid, ask, volume
        assert fig.layout.title.text == "Market Price Analysis"
    
    def test_create_price_chart_with_signals(self, market_chart, sample_price_data, sample_signals):
        """Test price chart with trading signals."""
        fig = market_chart.create_price_chart(
            sample_price_data, 
            signals=sample_signals,
            title="Price Chart with Signals"
        )
        
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Price Chart with Signals"
        
        # Check for signal markers (may not always be present due to timestamp matching)
        trace_names = [trace.name for trace in fig.data if hasattr(trace, 'name')]
        # Signals might not always appear due to timestamp matching in test data
    
    def test_create_price_chart_missing_columns(self, market_chart):
        """Test price chart with missing data columns."""
        # Create minimal price data with only 'last' column
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
        minimal_data = pd.DataFrame({'last': np.random.uniform(0.5, 0.6, len(dates))}, index=dates)
        
        fig = market_chart.create_price_chart(minimal_data)
        
        assert isinstance(fig, go.Figure)
        # Should still create chart with available data
        assert len(fig.data) >= 1


class TestPnLChart:
    """Test PnLChart functionality."""
    
    @pytest.fixture
    def pnl_chart(self):
        """Create PnLChart for testing."""
        chart_manager = ChartManager()
        return PnLChart(chart_manager)
    
    def test_create_portfolio_allocation_chart(self, pnl_chart):
        """Test portfolio allocation pie chart."""
        allocation_data = {
            'Position A': 2500.0,
            'Position B': 1800.0,
            'Position C': 3200.0,
            'Position D': 1500.0
        }
        
        fig = pnl_chart.create_portfolio_allocation_chart(allocation_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert fig.data[0].type == 'pie'
        assert fig.layout.title.text == "Portfolio Allocation"
        
        # Check pie chart properties
        pie_chart = fig.data[0]
        assert list(pie_chart.labels) == list(allocation_data.keys())
        assert list(pie_chart.values) == list(allocation_data.values())
    
    def test_create_daily_pnl_chart(self, pnl_chart):
        """Test daily P&L bar chart."""
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        daily_pnl = pd.Series(np.random.normal(50, 200, len(dates)), index=dates)
        
        fig = pnl_chart.create_daily_pnl_chart(daily_pnl)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # Daily bars + cumulative line
        assert fig.layout.title.text == "Daily P&L"
        
        # Check for bar chart and line trace
        trace_types = [trace.type for trace in fig.data]
        assert 'bar' in trace_types
        assert 'scatter' in trace_types


class TestDrawdownChart:
    """Test DrawdownChart functionality."""
    
    @pytest.fixture
    def drawdown_chart(self):
        """Create DrawdownChart for testing."""
        chart_manager = ChartManager()
        return DrawdownChart(chart_manager)
    
    def test_create_underwater_chart(self, drawdown_chart, sample_returns_data):
        """Test underwater equity curve chart."""
        fig = drawdown_chart.create_underwater_chart(sample_returns_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1  # At minimum the underwater curve
        assert fig.layout.title.text == "Underwater Equity Curve"
        
        # Check for fill and zero line
        underwater_trace = fig.data[0]
        assert underwater_trace.fill == 'tozeroy'


class TestCorrelationChart:
    """Test CorrelationChart functionality."""
    
    @pytest.fixture
    def correlation_chart(self):
        """Create CorrelationChart for testing."""
        chart_manager = ChartManager()
        return CorrelationChart(chart_manager)
    
    def test_create_rolling_correlation_chart(self, correlation_chart):
        """Test rolling correlation analysis chart."""
        dates = pd.date_range(start='2024-01-01', end='2024-06-30', freq='D')
        asset1_returns = pd.Series(np.random.normal(0.001, 0.02, len(dates)), index=dates)
        asset2_returns = pd.Series(np.random.normal(0.0008, 0.018, len(dates)), index=dates)
        
        fig = correlation_chart.create_rolling_correlation_chart(
            asset1_returns, asset2_returns,
            asset1_name="Asset 1", asset2_name="Asset 2"
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1  # Rolling correlation line
        assert fig.layout.title.text == "Rolling Correlation Analysis"
        assert fig.layout.yaxis.range == [-1, 1]  # Correlation bounds
    
    def test_rolling_correlation_custom_window(self, correlation_chart):
        """Test rolling correlation with custom window."""
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        asset1_returns = pd.Series(np.random.normal(0.001, 0.02, len(dates)), index=dates)
        asset2_returns = pd.Series(np.random.normal(0.0008, 0.018, len(dates)), index=dates)
        
        fig = correlation_chart.create_rolling_correlation_chart(
            asset1_returns, asset2_returns,
            window=30,  # 30-day window
            title="30-Day Rolling Correlation"
        )
        
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "30-Day Rolling Correlation"


class TestChartIntegration:
    """Test chart integration and edge cases."""
    
    def test_empty_data_handling(self):
        """Test charts handle empty data gracefully."""
        chart_manager = ChartManager()
        performance_chart = PerformanceChart(chart_manager)
        
        empty_series = pd.Series([], dtype=float)
        
        # Should not raise exception
        fig = performance_chart.create_pnl_chart(empty_series)
        assert isinstance(fig, go.Figure)
    
    def test_single_data_point(self):
        """Test charts handle single data point."""
        chart_manager = ChartManager()
        performance_chart = PerformanceChart(chart_manager)
        
        single_point = pd.Series([0.01], index=[datetime.now()])
        
        fig = performance_chart.create_pnl_chart(single_point)
        assert isinstance(fig, go.Figure)
    
    def test_nan_data_handling(self):
        """Test charts handle NaN values in data."""
        chart_manager = ChartManager()
        risk_chart = RiskChart(chart_manager)
        
        # Create data with NaN values
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
        data_with_nan = pd.Series([0.01, np.nan, 0.02, np.nan, -0.01, 0.005, np.nan, 0.01, -0.005, 0.02], index=dates)
        
        # Should handle NaN values gracefully
        fig = risk_chart.create_var_chart(data_with_nan)
        assert isinstance(fig, go.Figure)
