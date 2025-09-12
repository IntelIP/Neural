"""
Unit tests for Visualization Framework.

Tests the main visualization API including PerformanceVisualizer,
RiskVisualizer, StrategyVisualizer, and MarketVisualizer components
with graceful handling of missing dependencies.
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
    from neural.visualization.visualizer import (
        PerformanceVisualizer, RiskVisualizer, StrategyVisualizer, MarketVisualizer,
        VisualizationConfig, VisualizationTheme
    )
    from neural.analysis.metrics import PerformanceMetrics
    from neural.strategy.base import Signal, SignalType


@pytest.fixture
def sample_returns_data():
    """Create sample returns data for testing."""
    dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='D')
    returns = np.random.normal(0.001, 0.015, len(dates))
    return pd.Series(returns, index=dates)


@pytest.fixture
def sample_positions():
    """Create sample positions for testing."""
    return {
        'MARKET_A': {
            'market_value': 2500.0,
            'pnl': 150.0,
            'pnl_pct': 6.0,
            'category': 'NFL',
            'returns': 0.06,
            'volatility': 0.12
        },
        'MARKET_B': {
            'market_value': 1800.0,
            'pnl': -90.0,
            'pnl_pct': -5.0,
            'category': 'NBA',
            'returns': -0.05,
            'volatility': 0.15
        },
        'MARKET_C': {
            'market_value': 3200.0,
            'pnl': 400.0,
            'pnl_pct': 12.5,
            'category': 'MLB',
            'returns': 0.125,
            'volatility': 0.18
        }
    }


@pytest.fixture
def sample_strategies_data():
    """Create sample strategy performance data."""
    dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='D')
    
    return {
        'Strategy_A': {
            'returns': pd.Series(np.random.normal(0.0012, 0.015, len(dates)), index=dates),
            'description': 'Mean Reversion Strategy'
        },
        'Strategy_B': {
            'returns': pd.Series(np.random.normal(0.0008, 0.012, len(dates)), index=dates),
            'description': 'Momentum Strategy'
        },
        'Strategy_C': {
            'returns': pd.Series(np.random.normal(0.0005, 0.008, len(dates)), index=dates),
            'description': 'Arbitrage Strategy'
        }
    }


class TestVisualizationConfig:
    """Test VisualizationConfig functionality."""
    
    def test_default_config(self):
        """Test default visualization configuration."""
        config = VisualizationConfig()
        
        assert config.theme == VisualizationTheme.NEURAL_DARK
        assert config.default_width == 1200
        assert config.default_height == 600
        assert config.export_dpi == 300
        assert config.auto_show is True
        assert config.save_charts is False
        assert config.output_directory == "visualizations"
        assert config.template_directory is None
    
    def test_custom_config(self):
        """Test custom visualization configuration."""
        config = VisualizationConfig(
            theme=VisualizationTheme.TRADING_DARK,
            default_width=800,
            default_height=400,
            auto_show=False,
            save_charts=True,
            output_directory="custom_output"
        )
        
        assert config.theme == VisualizationTheme.TRADING_DARK
        assert config.default_width == 800
        assert config.default_height == 400
        assert config.auto_show is False
        assert config.save_charts is True
        assert config.output_directory == "custom_output"


class TestPerformanceVisualizer:
    """Test PerformanceVisualizer functionality."""
    
    @pytest.fixture
    def performance_visualizer(self):
        """Create PerformanceVisualizer for testing."""
        config = VisualizationConfig(auto_show=False, save_charts=False)
        return PerformanceVisualizer(config)
    
    def test_initialization(self, performance_visualizer):
        """Test PerformanceVisualizer initialization."""
        assert performance_visualizer.config is not None
        assert performance_visualizer.chart_manager is not None
        assert performance_visualizer.performance_chart is not None
        assert performance_visualizer.pnl_chart is not None
        assert performance_visualizer.drawdown_chart is not None
        assert performance_visualizer.performance_calculator is not None
    
    def test_get_chart_config(self, performance_visualizer):
        """Test chart configuration generation."""
        chart_config = performance_visualizer._get_chart_config()
        
        assert chart_config.width == performance_visualizer.config.default_width
        assert chart_config.height == performance_visualizer.config.default_height
    
    def test_create_comprehensive_performance_dashboard(self, performance_visualizer, sample_returns_data):
        """Test comprehensive performance dashboard creation."""
        dashboard_charts = performance_visualizer.create_comprehensive_performance_dashboard(
            sample_returns_data
        )
        
        assert isinstance(dashboard_charts, dict)
        assert 'performance' in dashboard_charts
        assert 'drawdown' in dashboard_charts
        assert 'daily_pnl' in dashboard_charts
        assert 'underwater' in dashboard_charts
        
        # Check chart types
        for chart_name, chart in dashboard_charts.items():
            assert isinstance(chart, go.Figure)
    
    def test_create_performance_dashboard_with_benchmark(
        self, performance_visualizer, sample_returns_data
    ):
        """Test performance dashboard with benchmark comparison."""
        benchmark_returns = sample_returns_data * 0.9  # Slightly worse benchmark
        
        dashboard_charts = performance_visualizer.create_comprehensive_performance_dashboard(
            sample_returns_data, benchmark_returns=benchmark_returns
        )
        
        assert isinstance(dashboard_charts, dict)
        assert 'comparison' in dashboard_charts  # Should include comparison chart
        
        comparison_chart = dashboard_charts['comparison']
        assert isinstance(comparison_chart, go.Figure)
    
    def test_create_rolling_metrics_analysis(self, performance_visualizer, sample_returns_data):
        """Test rolling metrics analysis chart."""
        fig = performance_visualizer.create_rolling_metrics_analysis(
            sample_returns_data, window=30
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 4  # Rolling Sharpe, vol, drawdown, win rate
        assert fig.layout.title.text == "Rolling Performance Metrics Analysis"
    
    def test_create_portfolio_allocation_dashboard(self, performance_visualizer, sample_positions):
        """Test portfolio allocation dashboard."""
        allocation_charts = performance_visualizer.create_portfolio_allocation_dashboard(
            sample_positions
        )
        
        assert isinstance(allocation_charts, dict)
        assert 'allocation' in allocation_charts
        
        if 'position_pnl' in allocation_charts:  # Only if there's P&L data
            assert isinstance(allocation_charts['position_pnl'], go.Figure)
        
        if 'category_exposure' in allocation_charts:  # Only if category data exists
            assert isinstance(allocation_charts['category_exposure'], go.Figure)
    
    @patch('builtins.open', create=True)
    @patch('os.makedirs')
    def test_save_dashboard_charts_with_save_enabled(self, mock_makedirs, mock_open):
        """Test saving charts when save_charts is enabled."""
        config = VisualizationConfig(save_charts=True, auto_show=False)
        visualizer = PerformanceVisualizer(config)
        
        # Mock figure with write methods
        mock_fig = Mock(spec=go.Figure)
        mock_fig.write_html = Mock()
        mock_fig.write_image = Mock()
        
        charts = {'test_chart': mock_fig}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = os.path.join(temp_dir, "test")
            visualizer._save_dashboard_charts(charts, base_path)
            
            # Should call write method
            assert mock_fig.write_html.called or mock_fig.write_image.called


class TestRiskVisualizer:
    """Test RiskVisualizer functionality."""
    
    @pytest.fixture
    def risk_visualizer(self):
        """Create RiskVisualizer for testing."""
        config = VisualizationConfig(auto_show=False, save_charts=False)
        return RiskVisualizer(config)
    
    def test_initialization(self, risk_visualizer):
        """Test RiskVisualizer initialization."""
        assert risk_visualizer.config is not None
        assert risk_visualizer.chart_manager is not None
        assert risk_visualizer.risk_chart is not None
        assert risk_visualizer.correlation_chart is not None
    
    def test_create_risk_dashboard(self, risk_visualizer, sample_returns_data, sample_positions):
        """Test risk dashboard creation."""
        risk_limits = {
            'position_size': {'current': 0.08, 'limit': 0.10},
            'daily_var': {'current': 0.025, 'limit': 0.03}
        }
        
        risk_charts = risk_visualizer.create_risk_dashboard(
            sample_returns_data,
            positions=sample_positions,
            risk_limits=risk_limits
        )
        
        assert isinstance(risk_charts, dict)
        assert 'var' in risk_charts
        
        # Check for risk-return chart if positions have required data
        if 'risk_return' in risk_charts:
            assert isinstance(risk_charts['risk_return'], go.Figure)
        
        # Check for risk limits chart
        if 'risk_limits' in risk_charts:
            assert isinstance(risk_charts['risk_limits'], go.Figure)
    
    def test_create_correlation_analysis(self, risk_visualizer):
        """Test correlation analysis dashboard."""
        # Create sample returns matrix
        dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='D')
        returns_matrix = pd.DataFrame({
            'Asset_A': np.random.normal(0.001, 0.015, len(dates)),
            'Asset_B': np.random.normal(0.0008, 0.012, len(dates)),
            'Asset_C': np.random.normal(0.0005, 0.018, len(dates))
        }, index=dates)
        
        correlation_charts = risk_visualizer.create_correlation_analysis(returns_matrix)
        
        assert isinstance(correlation_charts, dict)
        assert 'heatmap' in correlation_charts
        assert isinstance(correlation_charts['heatmap'], go.Figure)
        
        # Should include rolling correlation for first two assets
        if 'rolling_correlation' in correlation_charts:
            assert isinstance(correlation_charts['rolling_correlation'], go.Figure)


class TestStrategyVisualizer:
    """Test StrategyVisualizer functionality."""
    
    @pytest.fixture
    def strategy_visualizer(self):
        """Create StrategyVisualizer for testing."""
        config = VisualizationConfig(auto_show=False, save_charts=False)
        return StrategyVisualizer(config)
    
    def test_initialization(self, strategy_visualizer):
        """Test StrategyVisualizer initialization."""
        assert strategy_visualizer.config is not None
        assert strategy_visualizer.chart_manager is not None
        assert strategy_visualizer.performance_calculator is not None
    
    def test_create_strategy_comparison_dashboard(
        self, strategy_visualizer, sample_strategies_data
    ):
        """Test strategy comparison dashboard."""
        comparison_charts = strategy_visualizer.create_strategy_comparison_dashboard(
            sample_strategies_data
        )
        
        assert isinstance(comparison_charts, dict)
        
        if comparison_charts:  # Only check if charts were created
            if 'metrics_comparison' in comparison_charts:
                assert isinstance(comparison_charts['metrics_comparison'], go.Figure)
            
            if 'returns_comparison' in comparison_charts:
                assert isinstance(comparison_charts['returns_comparison'], go.Figure)
    
    def test_empty_strategies_data(self, strategy_visualizer):
        """Test handling of empty strategies data."""
        empty_data = {}
        
        comparison_charts = strategy_visualizer.create_strategy_comparison_dashboard(empty_data)
        
        assert isinstance(comparison_charts, dict)
        # Should handle empty data gracefully


class TestMarketVisualizer:
    """Test MarketVisualizer functionality."""
    
    @pytest.fixture
    def market_visualizer(self):
        """Create MarketVisualizer for testing."""
        config = VisualizationConfig(auto_show=False, save_charts=False)
        return MarketVisualizer(config)
    
    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data."""
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        return pd.DataFrame({
            'last': np.random.uniform(0.45, 0.55, len(dates)),
            'bid': np.random.uniform(0.42, 0.52, len(dates)),
            'ask': np.random.uniform(0.48, 0.58, len(dates)),
            'volume': np.random.randint(100, 2000, len(dates))
        }, index=dates)
    
    def test_initialization(self, market_visualizer):
        """Test MarketVisualizer initialization."""
        assert market_visualizer.config is not None
        assert market_visualizer.chart_manager is not None
        assert market_visualizer.market_chart is not None
    
    def test_create_market_analysis_dashboard(self, market_visualizer, sample_price_data):
        """Test market analysis dashboard."""
        signals = [
            Signal(
                signal_type=SignalType.BUY_YES,
                market_id="TEST_MARKET",
                timestamp=datetime.now(),
                confidence=0.80,
                edge=0.06,
                expected_value=30.0,
                recommended_size=0.10,
                max_contracts=100
            )
        ]
        
        market_charts = market_visualizer.create_market_analysis_dashboard(
            sample_price_data, signals=signals
        )
        
        assert isinstance(market_charts, dict)
        assert 'price_analysis' in market_charts
        assert isinstance(market_charts['price_analysis'], go.Figure)


class TestVisualizationIntegration:
    """Test visualization framework integration and edge cases."""
    
    def test_theme_mapping(self):
        """Test theme mapping between visualization and chart themes."""
        config = VisualizationConfig(theme=VisualizationTheme.NEURAL_DARK)
        visualizer = PerformanceVisualizer(config)
        chart_config = visualizer._get_chart_config()
        
        # Should map to appropriate chart theme
        from neural.visualization.charts import ChartTheme
        assert chart_config.theme in [ChartTheme.NEURAL, ChartTheme.WHITE, ChartTheme.TRADING]
    
    def test_empty_data_handling(self):
        """Test visualizers handle empty data gracefully."""
        config = VisualizationConfig(auto_show=False, save_charts=False)
        performance_viz = PerformanceVisualizer(config)
        
        empty_returns = pd.Series([], dtype=float)
        
        # Should not raise exceptions
        dashboard_charts = performance_viz.create_comprehensive_performance_dashboard(empty_returns)
        assert isinstance(dashboard_charts, dict)
    
    def test_single_data_point(self):
        """Test visualizers handle single data point."""
        config = VisualizationConfig(auto_show=False, save_charts=False)
        risk_viz = RiskVisualizer(config)
        
        single_point = pd.Series([0.01], index=[datetime.now()])
        
        risk_charts = risk_viz.create_risk_dashboard(single_point)
        assert isinstance(risk_charts, dict)
    
    def test_configuration_inheritance(self):
        """Test configuration is properly inherited by components."""
        custom_config = VisualizationConfig(
            default_width=800,
            default_height=400,
            theme=VisualizationTheme.TRADING_DARK
        )
        
        visualizer = PerformanceVisualizer(custom_config)
        
        assert visualizer.config.default_width == 800
        assert visualizer.config.default_height == 400
        assert visualizer.config.theme == VisualizationTheme.TRADING_DARK
        
        chart_config = visualizer._get_chart_config()
        assert chart_config.width == 800
        assert chart_config.height == 400
    
    @patch('webbrowser.open')
    def test_auto_show_disabled(self, mock_browser):
        """Test that auto_show=False prevents browser opening."""
        config = VisualizationConfig(auto_show=False, save_charts=False)
        visualizer = PerformanceVisualizer(config)
        
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        returns = pd.Series(np.random.normal(0.001, 0.015, len(dates)), index=dates)
        
        visualizer.create_comprehensive_performance_dashboard(returns)
        
        # Browser should not be opened
        mock_browser.assert_not_called()
    
    def test_missing_optional_data(self):
        """Test handling of missing optional data in visualizations."""
        config = VisualizationConfig(auto_show=False, save_charts=False)
        performance_viz = PerformanceVisualizer(config)
        
        # Test with minimal position data (no categories, returns, etc.)
        minimal_positions = {
            'MARKET_A': {'market_value': 1000.0},
            'MARKET_B': {'market_value': 2000.0}
        }
        
        allocation_charts = performance_viz.create_portfolio_allocation_dashboard(minimal_positions)
        
        assert isinstance(allocation_charts, dict)
        assert 'allocation' in allocation_charts
        # Other charts may not be present due to missing data
    
    def test_nan_and_inf_handling(self):
        """Test handling of NaN and infinite values in data."""
        config = VisualizationConfig(auto_show=False, save_charts=False)
        risk_viz = RiskVisualizer(config)
        
        # Create data with NaN and inf values
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
        problematic_data = pd.Series([
            0.01, np.nan, 0.02, np.inf, -0.01, 
            -np.inf, np.nan, 0.01, -0.005, 0.02
        ], index=dates)
        
        # Should handle problematic data gracefully
        risk_charts = risk_viz.create_risk_dashboard(problematic_data)
        assert isinstance(risk_charts, dict)
