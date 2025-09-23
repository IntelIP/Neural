"""
Real-time Dashboard Components for Neural SDK

Provides interactive Dash-based dashboards for:
- Real-time portfolio monitoring and P&L tracking
- Strategy performance comparison and analysis
- Risk monitoring and limit alerting
- Market data visualization and signal tracking
- Live trading activity and order management
"""

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
import logging
from collections import deque

from .charts import ChartManager, ChartTheme, PerformanceChart, RiskChart, MarketChart, PnLChart
from ..analysis.metrics import PerformanceMetrics, PerformanceCalculator
from ..strategy.base import Signal, SignalType
from ..risk.monitor import RiskMonitor

logger = logging.getLogger(__name__)


class DashboardTheme(Enum):
    """Dashboard styling themes."""
    DARK = "dark"
    LIGHT = "light"
    TRADING = "trading"
    NEURAL = "neural"


@dataclass
class DashboardConfig:
    """Configuration for dashboard setup."""
    theme: DashboardTheme = DashboardTheme.NEURAL
    host: str = "127.0.0.1"
    port: int = 8050
    debug: bool = False
    auto_refresh_interval: int = 5000  # milliseconds
    max_data_points: int = 1000
    enable_real_time: bool = True
    enable_alerts: bool = True


class DashboardServer:
    """
    Central server for hosting multiple dashboard applications.
    
    Manages real-time data feeds and coordinates between different
    dashboard components for comprehensive trading monitoring.
    """
    
    def __init__(self, config: Optional[DashboardConfig] = None):
        """Initialize dashboard server."""
        self.config = config or DashboardConfig()
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.themes.GRID],
            suppress_callback_exceptions=True
        )
        
        # Data storage for real-time updates
        self.live_data = {
            'portfolio_value': deque(maxlen=self.config.max_data_points),
            'daily_pnl': deque(maxlen=self.config.max_data_points),
            'positions': {},
            'signals': deque(maxlen=100),
            'risk_metrics': {},
            'alerts': deque(maxlen=50),
            'timestamps': deque(maxlen=self.config.max_data_points)
        }
        
        # Initialize components
        self.chart_manager = ChartManager()
        self.performance_calculator = PerformanceCalculator()
        
        # Setup theme
        self._setup_theme()
        self._setup_callbacks()
        
        # Data update thread
        self._stop_updates = threading.Event()
        self._update_thread = None
        
    def _setup_theme(self):
        """Setup dashboard theme and styling."""
        theme_configs = {
            DashboardTheme.NEURAL: {
                'bg_color': '#1a1a1a',
                'card_color': '#2d2d2d', 
                'text_color': '#ffffff',
                'accent_color': '#00ff88',
                'border_color': '#404040'
            },
            DashboardTheme.TRADING: {
                'bg_color': '#0d1421',
                'card_color': '#1e2329',
                'text_color': '#c0c0c0',
                'accent_color': '#2eca6a',
                'border_color': '#2b3139'
            },
            DashboardTheme.DARK: {
                'bg_color': '#2c3e50',
                'card_color': '#34495e',
                'text_color': '#ecf0f1',
                'accent_color': '#3498db',
                'border_color': '#7f8c8d'
            }
        }
        
        self.theme = theme_configs.get(self.config.theme, theme_configs[DashboardTheme.NEURAL])
        
    def _setup_callbacks(self):
        """Setup dashboard callbacks for interactivity."""
        
        @self.app.callback(
            [Output('portfolio-overview-card', 'children'),
             Output('performance-chart', 'figure'),
             Output('risk-metrics-card', 'children'),
             Output('recent-signals-card', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            return (
                self._generate_portfolio_overview(),
                self._generate_performance_chart(),
                self._generate_risk_metrics(),
                self._generate_recent_signals()
            )
        
        @self.app.callback(
            Output('alerts-container', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_alerts(n):
            return self._generate_alerts()
            
    def _generate_portfolio_overview(self) -> List[Any]:
        """Generate portfolio overview cards."""
        if not self.live_data['portfolio_value']:
            return [html.Div("No portfolio data available", className="text-muted")]
            
        current_value = list(self.live_data['portfolio_value'])[-1]
        daily_pnl = list(self.live_data['daily_pnl'])[-1] if self.live_data['daily_pnl'] else 0
        
        # Calculate daily return percentage
        if len(self.live_data['portfolio_value']) > 1:
            previous_value = list(self.live_data['portfolio_value'])[-2]
            daily_return_pct = ((current_value - previous_value) / previous_value) * 100
        else:
            daily_return_pct = 0
            
        return [
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"${current_value:,.2f}", className="card-title text-success"),
                            html.P("Portfolio Value", className="card-text")
                        ])
                    ], className="mb-3")
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(
                                f"${daily_pnl:,.2f}", 
                                className=f"card-title {'text-success' if daily_pnl >= 0 else 'text-danger'}"
                            ),
                            html.P("Daily P&L", className="card-text")
                        ])
                    ], className="mb-3")
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(
                                f"{daily_return_pct:+.2f}%",
                                className=f"card-title {'text-success' if daily_return_pct >= 0 else 'text-danger'}"
                            ),
                            html.P("Daily Return", className="card-text")
                        ])
                    ], className="mb-3")
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{len(self.live_data['positions'])}", className="card-title text-info"),
                            html.P("Active Positions", className="card-text")
                        ])
                    ], className="mb-3")
                ], width=3)
            ])
        ]
        
    def _generate_performance_chart(self) -> go.Figure:
        """Generate real-time performance chart."""
        if not self.live_data['portfolio_value']:
            # Return empty chart
            fig = go.Figure()
            fig.add_annotation(
                text="No performance data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
            
        # Convert to pandas series for chart generation
        timestamps = list(self.live_data['timestamps'])
        portfolio_values = list(self.live_data['portfolio_value'])
        
        if len(timestamps) != len(portfolio_values):
            # Sync data
            min_len = min(len(timestamps), len(portfolio_values))
            timestamps = timestamps[-min_len:]
            portfolio_values = portfolio_values[-min_len:]
            
        df = pd.DataFrame({
            'timestamp': timestamps,
            'value': portfolio_values
        }).set_index('timestamp')
        
        # Calculate returns
        returns = df['value'].pct_change().dropna()
        
        if len(returns) > 0:
            performance_chart = PerformanceChart(self.chart_manager)
            return performance_chart.create_pnl_chart(returns, title="Live Portfolio Performance")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=timestamps, y=portfolio_values, name="Portfolio Value"))
            return self.chart_manager.apply_theme(fig)
            
    def _generate_risk_metrics(self) -> List[Any]:
        """Generate risk metrics display."""
        risk_metrics = self.live_data.get('risk_metrics', {})
        
        return [
            dbc.Card([
                dbc.CardHeader("Risk Metrics"),
                dbc.CardBody([
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                html.Strong("Current Drawdown:"),
                                html.Span(
                                    f" {risk_metrics.get('current_drawdown', 0):.2%}",
                                    className="text-warning ms-2"
                                )
                            ], width=6),
                            dbc.Col([
                                html.Strong("VaR (95%):"),
                                html.Span(
                                    f" {risk_metrics.get('var_95', 0):.2%}",
                                    className="text-danger ms-2"
                                )
                            ], width=6)
                        ], className="mb-2"),
                        
                        dbc.Row([
                            dbc.Col([
                                html.Strong("Sharpe Ratio:"),
                                html.Span(
                                    f" {risk_metrics.get('sharpe_ratio', 0):.2f}",
                                    className="text-info ms-2"
                                )
                            ], width=6),
                            dbc.Col([
                                html.Strong("Max Positions:"),
                                html.Span(
                                    f" {len(self.live_data['positions'])}/20",
                                    className="text-success ms-2"
                                )
                            ], width=6)
                        ])
                    ])
                ])
            ], className="mb-3")
        ]
        
    def _generate_recent_signals(self) -> List[Any]:
        """Generate recent signals display."""
        recent_signals = list(self.live_data['signals'])[-5:]  # Last 5 signals
        
        if not recent_signals:
            return [
                dbc.Card([
                    dbc.CardHeader("Recent Signals"),
                    dbc.CardBody([
                        html.P("No recent signals", className="text-muted")
                    ])
                ], className="mb-3")
            ]
            
        signal_rows = []
        for signal in reversed(recent_signals):  # Most recent first
            signal_type_color = {
                SignalType.BUY_YES: "success",
                SignalType.BUY_NO: "success", 
                SignalType.SELL_YES: "danger",
                SignalType.SELL_NO: "danger",
                SignalType.HOLD: "secondary"
            }.get(signal.signal_type, "secondary")
            
            signal_rows.append(
                dbc.Row([
                    dbc.Col([
                        dbc.Badge(
                            signal.signal_type.value.upper(),
                            color=signal_type_color,
                            className="me-2"
                        ),
                        html.Small(signal.market_id, className="text-muted")
                    ], width=6),
                    dbc.Col([
                        html.Small(f"Conf: {signal.confidence:.2f}", className="text-info me-2"),
                        html.Small(f"Edge: {signal.edge:.3f}", className="text-warning")
                    ], width=6)
                ], className="mb-1")
            )
            
        return [
            dbc.Card([
                dbc.CardHeader("Recent Signals"),
                dbc.CardBody(signal_rows)
            ], className="mb-3")
        ]
        
    def _generate_alerts(self) -> List[Any]:
        """Generate alerts display."""
        recent_alerts = list(self.live_data['alerts'])[-3:]  # Last 3 alerts
        
        if not recent_alerts:
            return []
            
        alert_components = []
        for alert in reversed(recent_alerts):
            alert_color = {
                'info': 'info',
                'warning': 'warning', 
                'error': 'danger',
                'success': 'success'
            }.get(alert.get('level', 'info'), 'info')
            
            alert_components.append(
                dbc.Alert([
                    html.Strong(f"{alert.get('title', 'Alert')}: "),
                    alert.get('message', 'No message'),
                    html.Small(
                        f" - {alert.get('timestamp', datetime.now()).strftime('%H:%M:%S')}",
                        className="ms-2 text-muted"
                    )
                ], color=alert_color, dismissable=True, className="mb-2")
            )
            
        return alert_components
        
    def create_layout(self) -> html.Div:
        """Create the main dashboard layout."""
        return html.Div([
            # Header
            dbc.Navbar([
                dbc.Container([
                    dbc.NavbarBrand("Neural SDK Trading Dashboard", className="ms-2"),
                    dbc.NavbarToggler(id="navbar-toggler"),
                    dbc.Collapse([
                        dbc.Nav([
                            dbc.NavItem(dbc.NavLink("Portfolio", href="#portfolio")),
                            dbc.NavItem(dbc.NavLink("Strategies", href="#strategies")),
                            dbc.NavItem(dbc.NavLink("Risk", href="#risk")),
                            dbc.NavItem(dbc.NavLink("Markets", href="#markets"))
                        ], navbar=True)
                    ], id="navbar-collapse", navbar=True)
                ])
            ], color="primary", dark=True, className="mb-4"),
            
            # Main Content
            dbc.Container([
                # Alerts section
                html.Div(id="alerts-container"),
                
                # Portfolio overview
                html.Div(id="portfolio-overview-card"),
                
                # Main chart
                dbc.Card([
                    dbc.CardHeader("Portfolio Performance"),
                    dbc.CardBody([
                        dcc.Graph(id="performance-chart")
                    ])
                ], className="mb-4"),
                
                # Secondary panels
                dbc.Row([
                    dbc.Col([
                        html.Div(id="risk-metrics-card")
                    ], width=6),
                    dbc.Col([
                        html.Div(id="recent-signals-card")
                    ], width=6)
                ])
            ], fluid=True),
            
            # Auto-refresh component
            dcc.Interval(
                id='interval-component',
                interval=self.config.auto_refresh_interval,
                n_intervals=0
            ) if self.config.enable_real_time else html.Div()
        ])
        
    def update_data(self, 
                   portfolio_value: Optional[float] = None,
                   daily_pnl: Optional[float] = None,
                   positions: Optional[Dict] = None,
                   signals: Optional[List[Signal]] = None,
                   risk_metrics: Optional[Dict] = None,
                   alerts: Optional[List[Dict]] = None):
        """Update dashboard data."""
        timestamp = datetime.now()
        
        if portfolio_value is not None:
            self.live_data['portfolio_value'].append(portfolio_value)
            self.live_data['timestamps'].append(timestamp)
            
        if daily_pnl is not None:
            self.live_data['daily_pnl'].append(daily_pnl)
            
        if positions is not None:
            self.live_data['positions'].update(positions)
            
        if signals is not None:
            for signal in signals:
                self.live_data['signals'].append(signal)
                
        if risk_metrics is not None:
            self.live_data['risk_metrics'].update(risk_metrics)
            
        if alerts is not None:
            for alert in alerts:
                alert['timestamp'] = timestamp
                self.live_data['alerts'].append(alert)
                
    def start_server(self):
        """Start the dashboard server."""
        self.app.layout = self.create_layout()
        
        logger.info(f"Starting Neural SDK Dashboard on {self.config.host}:{self.config.port}")
        self.app.run_server(
            host=self.config.host,
            port=self.config.port,
            debug=self.config.debug
        )
        
    def stop_server(self):
        """Stop the dashboard server and cleanup."""
        self._stop_updates.set()
        if self._update_thread and self._update_thread.is_alive():
            self._update_thread.join()
            
        logger.info("Neural SDK Dashboard stopped")


class RealTimeDashboard:
    """Specialized real-time trading dashboard."""
    
    def __init__(self, dashboard_server: DashboardServer):
        self.server = dashboard_server
        self.risk_monitor = None
        
    def set_risk_monitor(self, risk_monitor: RiskMonitor):
        """Set risk monitor for real-time alerts."""
        self.risk_monitor = risk_monitor
        
    def start_monitoring(self, data_feed_callback: Callable):
        """Start real-time data monitoring."""
        def monitoring_loop():
            while not self.server._stop_updates.is_set():
                try:
                    # Get data from callback
                    data = data_feed_callback()
                    
                    if data:
                        self.server.update_data(**data)
                        
                    # Check risk alerts if monitor is available
                    if self.risk_monitor:
                        # Implementation would depend on risk monitor integration
                        pass
                        
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    
                time.sleep(1)  # Update every second
                
        self.server._update_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.server._update_thread.start()


class PortfolioDashboard:
    """Portfolio-focused dashboard with detailed position tracking."""
    
    def __init__(self, chart_manager: ChartManager):
        self.chart_manager = chart_manager
        
    def create_position_table(self, positions: Dict) -> html.Table:
        """Create detailed positions table."""
        if not positions:
            return html.Div("No active positions", className="text-muted")
            
        headers = ["Market", "Side", "Contracts", "Avg Price", "Current Price", "P&L", "P&L %"]
        
        rows = []
        for market_id, position in positions.items():
            pnl_color = "text-success" if position.get('pnl', 0) >= 0 else "text-danger"
            
            rows.append(html.Tr([
                html.Td(market_id),
                html.Td(position.get('side', 'N/A')),
                html.Td(f"{position.get('contracts', 0):,}"),
                html.Td(f"${position.get('avg_price', 0):.3f}"),
                html.Td(f"${position.get('current_price', 0):.3f}"),
                html.Td(f"${position.get('pnl', 0):,.2f}", className=pnl_color),
                html.Td(f"{position.get('pnl_pct', 0):+.2f}%", className=pnl_color)
            ]))
            
        return dbc.Table([
            html.Thead([html.Tr([html.Th(header) for header in headers])]),
            html.Tbody(rows)
        ], striped=True, hover=True, responsive=True)


class StrategyDashboard:
    """Strategy performance comparison dashboard."""
    
    def __init__(self, chart_manager: ChartManager):
        self.chart_manager = chart_manager
        self.performance_calculator = PerformanceCalculator()
        
    def create_strategy_comparison_chart(self, strategies_data: Dict) -> go.Figure:
        """Create strategy performance comparison."""
        if not strategies_data:
            fig = go.Figure()
            fig.add_annotation(
                text="No strategy data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
            
        # Calculate metrics for each strategy
        strategies_metrics = {}
        for name, data in strategies_data.items():
            if 'returns' in data:
                metrics = self.performance_calculator.calculate_comprehensive_metrics(
                    returns=data['returns'],
                    benchmark_returns=data.get('benchmark_returns')
                )
                strategies_metrics[name] = metrics
                
        if strategies_metrics:
            performance_chart = PerformanceChart(self.chart_manager)
            return performance_chart.create_metrics_comparison_chart(
                strategies_metrics,
                "Strategy Performance Comparison"
            )
        else:
            return go.Figure()


class RiskDashboard:
    """Risk management focused dashboard."""
    
    def __init__(self, chart_manager: ChartManager):
        self.chart_manager = chart_manager
        
    def create_risk_overview(self, risk_data: Dict) -> List[Any]:
        """Create comprehensive risk overview."""
        return [
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Position Risk"),
                        dbc.CardBody([
                            html.H5(f"{risk_data.get('position_count', 0)}/20", className="text-info"),
                            html.P("Active Positions")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Daily VaR"),
                        dbc.CardBody([
                            html.H5(f"{risk_data.get('daily_var', 0):.2%}", className="text-warning"),
                            html.P("Value at Risk")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Max Drawdown"),
                        dbc.CardBody([
                            html.H5(f"{risk_data.get('max_drawdown', 0):.2%}", className="text-danger"),
                            html.P("Peak to Trough")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Risk Score"),
                        dbc.CardBody([
                            html.H5(f"{risk_data.get('risk_score', 0):.1f}/10", 
                                   className=self._get_risk_score_color(risk_data.get('risk_score', 0))),
                            html.P("Overall Risk")
                        ])
                    ])
                ], width=3)
            ])
        ]
        
    def _get_risk_score_color(self, score: float) -> str:
        """Get color class based on risk score."""
        if score <= 3:
            return "text-success"
        elif score <= 6:
            return "text-warning"
        else:
            return "text-danger"
