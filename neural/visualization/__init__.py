"""
Neural SDK Visualization & Reporting Framework

This module provides comprehensive visualization and reporting capabilities
for Kalshi sports trading analysis, including:

• Interactive Plotly charts for performance analytics
• Real-time Dash dashboard with live P&L tracking  
• PDF/HTML performance report generation
• Strategy comparison & analysis tools
• Risk metrics visualization & monitoring

Key Components:
- ChartManager: Interactive Plotly chart generation
- DashboardServer: Real-time web dashboard
- ReportGenerator: PDF/HTML report creation
- PerformanceVisualizer: Strategy performance charts
- RiskVisualizer: Risk metrics and monitoring

Note: Visualization features require optional dependencies (plotly, dash, etc.)
Install with: pip install plotly dash dash-bootstrap-components matplotlib seaborn reportlab
"""

# Graceful handling of optional dependencies
__all__ = []

try:
    from .charts import (
        ChartManager, PerformanceChart, RiskChart, MarketChart,
        PnLChart, DrawdownChart, CorrelationChart
    )
    __all__.extend([
        'ChartManager', 'PerformanceChart', 'RiskChart', 'MarketChart',
        'PnLChart', 'DrawdownChart', 'CorrelationChart'
    ])
except ImportError as e:
    import warnings
    warnings.warn(f"Chart generation unavailable: {e}. Install plotly for chart functionality.", ImportWarning)

try:
    from .dashboard import (
        DashboardServer, RealTimeDashboard, PortfolioDashboard,
        StrategyDashboard, RiskDashboard
    )
    __all__.extend([
        'DashboardServer', 'RealTimeDashboard', 'PortfolioDashboard',
        'StrategyDashboard', 'RiskDashboard'
    ])
except ImportError as e:
    import warnings
    warnings.warn(f"Dashboard functionality unavailable: {e}. Install dash for dashboard functionality.", ImportWarning)

try:
    from .reports import (
        ReportGenerator, PerformanceReportBuilder, RiskReportBuilder,
        StrategyComparisonReport, ExportFormat
    )
    __all__.extend([
        'ReportGenerator', 'PerformanceReportBuilder', 'RiskReportBuilder',
        'StrategyComparisonReport', 'ExportFormat'
    ])
except ImportError as e:
    import warnings
    warnings.warn(f"Report generation unavailable: {e}. Install reportlab and jinja2 for report functionality.", ImportWarning)

try:
    from .visualizer import (
        PerformanceVisualizer, RiskVisualizer, StrategyVisualizer,
        MarketVisualizer, VisualizationTheme
    )
    __all__.extend([
        'PerformanceVisualizer', 'RiskVisualizer', 'StrategyVisualizer', 
        'MarketVisualizer', 'VisualizationTheme'
    ])
except ImportError as e:
    import warnings
    warnings.warn(f"Main visualization framework unavailable: {e}. Install visualization dependencies.", ImportWarning)
