"""Dashboard components module."""

from .profit_loss_chart import create_pnl_chart, create_performance_metrics_cards
from .trades_table import create_trades_table, create_simple_trades_table
from .positions_table import create_positions_table, create_position_risk_metrics
from .control_panel import (
    create_control_panel,
    create_agent_status_panel,
    create_quick_controls,
    create_connection_status
)

__all__ = [
    'create_pnl_chart',
    'create_performance_metrics_cards',
    'create_trades_table',
    'create_simple_trades_table',
    'create_positions_table',
    'create_position_risk_metrics',
    'create_control_panel',
    'create_agent_status_panel',
    'create_quick_controls',
    'create_connection_status'
]