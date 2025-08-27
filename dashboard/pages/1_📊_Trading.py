"""Trading Dashboard - Main P&L and positions monitoring page."""

import streamlit as st
import os
import sys
from datetime import datetime, timedelta
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import dashboard modules
from data import DatabaseManager, RedisStreamHandler, RedisDataProcessor
from components import (
    create_pnl_chart,
    create_performance_metrics_cards,
    create_trades_table,
    create_simple_trades_table,
    create_positions_table,
    create_connection_status
)
from utils import (
    calculate_portfolio_metrics,
    format_currency,
    format_percentage
)

# Page configuration
st.set_page_config(
    page_title="Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.2);
        padding: 5px 15px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)


def main():
    """Main trading dashboard page."""
    
    # Page title
    st.title("📊 Trading Dashboard")
    
    # Get shared resources from session state
    db_manager = st.session_state.get('db_manager')
    redis_handler = st.session_state.get('redis_handler')
    db_connected = st.session_state.get('db_connected', False)
    redis_connected = st.session_state.get('redis_connected', False)
    
    # Initialize session state for this page
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = True
    if 'date_range' not in st.session_state:
        st.session_state.date_range = (datetime.now() - timedelta(days=30), datetime.now())
    
    # Get data from database
    if db_manager:
        trades_data = db_manager.get_recent_trades(limit=100)
        # Convert Decimal to float for JSON serialization
        for trade in trades_data:
            if trade['price']:
                trade['price'] = float(trade['price'])
            if trade['fill_price']:
                trade['fill_price'] = float(trade['fill_price'])
            if trade['total_cost']:
                trade['total_cost'] = float(trade['total_cost'])
            if trade['realized_pnl']:
                trade['realized_pnl'] = float(trade['realized_pnl'])
            else:
                trade['realized_pnl'] = 0.0
        
        positions_data = db_manager.get_active_positions()
        # Convert Decimal to float for JSON serialization
        for pos in positions_data:
            if pos['entry_price']:
                pos['entry_price'] = float(pos['entry_price'])
            if pos['current_price']:
                pos['current_price'] = float(pos['current_price'])
            else:
                pos['current_price'] = float(pos['entry_price'])
            if pos['market_value']:
                pos['market_value'] = float(pos['market_value'])
            else:
                pos['market_value'] = 0.0
            if pos['unrealized_pnl']:
                pos['unrealized_pnl'] = float(pos['unrealized_pnl'])
            else:
                pos['unrealized_pnl'] = 0.0
            if pos['unrealized_pnl_pct']:
                pos['unrealized_pnl_pct'] = float(pos['unrealized_pnl_pct'])
            else:
                pos['unrealized_pnl_pct'] = 0.0
        
        pnl_data = db_manager.get_pnl_history(start_date=st.session_state.date_range[0])
        # Convert Decimal to float for JSON serialization
        for pnl in pnl_data:
            if pnl['total_pnl']:
                pnl['total_pnl'] = float(pnl['total_pnl'])
            if pnl['daily_pnl']:
                pnl['daily_pnl'] = float(pnl['daily_pnl'])
            else:
                pnl['daily_pnl'] = 0.0
            if pnl['realized_pnl']:
                pnl['realized_pnl'] = float(pnl['realized_pnl'])
            else:
                pnl['realized_pnl'] = 0.0
            if pnl['unrealized_pnl']:
                pnl['unrealized_pnl'] = float(pnl['unrealized_pnl'])
            else:
                pnl['unrealized_pnl'] = 0.0
            if pnl['win_rate']:
                pnl['win_rate'] = float(pnl['win_rate'])
            else:
                pnl['win_rate'] = 0.0
    else:
        trades_data = []
        positions_data = []
        pnl_data = []
    
    # Check for real-time updates from Redis
    if redis_handler and 'redis_data' in st.session_state:
        # Process real-time trade updates
        if 'kalshi:trades' in st.session_state.redis_data:
            new_trade = RedisDataProcessor.process_trade_update(
                st.session_state.redis_data['kalshi:trades']['data']
            )
            trades_data.insert(0, new_trade)  # Add to beginning of list
        
        # Process real-time position updates
        if 'kalshi:positions' in st.session_state.redis_data:
            new_position = RedisDataProcessor.process_position_update(
                st.session_state.redis_data['kalshi:positions']['data']
            )
            # Update or add position
            for i, p in enumerate(positions_data):
                if p['position_id'] == new_position['position_id']:
                    positions_data[i] = new_position
                    break
            else:
                positions_data.append(new_position)
    
    # Performance metrics cards
    metrics = calculate_portfolio_metrics(trades_data, positions_data)
    perf_metrics = create_performance_metrics_cards(pnl_data, trades_data)
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric(
            "Total P&L",
            format_currency(perf_metrics['total_pnl']),
            delta=format_percentage(perf_metrics['total_pnl'] / 10000 * 100) if perf_metrics['total_pnl'] else None
        )
    
    with col2:
        st.metric(
            "Daily P&L",
            format_currency(perf_metrics['daily_pnl']),
            delta="Today"
        )
    
    with col3:
        st.metric(
            "Win Rate",
            format_percentage(perf_metrics['win_rate']),
            delta=f"{metrics.get('win_count', 0)}/{metrics.get('total_trades', 0)}"
        )
    
    with col4:
        st.metric(
            "Total Trades",
            perf_metrics['total_trades'],
            delta="All time"
        )
    
    with col5:
        st.metric(
            "Sharpe Ratio",
            f"{perf_metrics['sharpe_ratio']:.2f}",
            delta="Risk-adjusted"
        )
    
    with col6:
        st.metric(
            "Max Drawdown",
            format_currency(abs(perf_metrics['max_drawdown'])),
            delta="Peak to trough"
        )
    
    st.markdown("---")
    
    # Date range selector for P&L chart
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.markdown("### Profit & Loss Chart")
    
    with col2:
        date_range = st.date_input(
            "Date Range",
            value=(st.session_state.date_range[0], st.session_state.date_range[1]),
            key="date_selector"
        )
        if len(date_range) == 2:
            st.session_state.date_range = date_range
    
    # P&L Chart
    chart = create_pnl_chart(pnl_data, trades_data, st.session_state.date_range)
    st.plotly_chart(chart, use_container_width=True)
    
    st.markdown("---")
    
    # Tables section - two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Recent Trades")
        # Use simple table if AgGrid not available
        try:
            create_trades_table(trades_data, height=400)
        except:
            create_simple_trades_table(trades_data)
    
    with col2:
        st.markdown("### Active Positions")
        create_positions_table(positions_data, auto_refresh=st.session_state.auto_refresh)
    
    # Connection status footer
    st.markdown("---")
    create_connection_status(redis_connected, db_connected)
    
    # Auto-refresh
    if st.session_state.auto_refresh:
        time.sleep(1)
        st.rerun()


# Sidebar controls (specific to this page)
with st.sidebar:
    st.markdown("## Trading Controls")
    
    # Auto-refresh toggle
    auto_refresh = st.checkbox(
        "Auto Refresh",
        value=st.session_state.get('auto_refresh', True),
        help="Automatically refresh dashboard every second"
    )
    st.session_state.auto_refresh = auto_refresh
    
    st.markdown("---")
    
    # Data settings
    st.markdown("### Data Settings")
    
    # P&L calculation period
    pnl_period = st.selectbox(
        "P&L Period",
        options=["1D", "1W", "1M", "3M", "YTD", "All"],
        index=2
    )
    
    st.markdown("---")
    
    # Export options
    st.markdown("### Export Data")
    
    if st.button("📊 Export to CSV"):
        # Export functionality would go here
        st.info("Export functionality coming soon")
    
    if st.button("📈 Generate Report"):
        # Report generation would go here
        st.info("Report generation coming soon")


if __name__ == "__main__":
    main()