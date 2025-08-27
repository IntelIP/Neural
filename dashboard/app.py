"""Main Streamlit application for Kalshi Trading Dashboard - Multi-page setup."""

import streamlit as st
import os
import sys
import threading
from typing import Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import dashboard modules
from data import DatabaseManager, RedisStreamHandler

# Page configuration
st.set_page_config(
    page_title="Kalshi Trading Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 48px;
        font-weight: bold;
        background: linear-gradient(90deg, #1e3a8a, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 30px;
    }
    .connection-badge {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.2);
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def init_database():
    """Initialize database connection."""
    try:
        db_manager = DatabaseManager()
        return db_manager, True
    except Exception as e:
        st.error(f"Failed to connect to database: {e}")
        return None, False


@st.cache_resource
def init_redis():
    """Initialize Redis connection."""
    try:
        redis_handler = RedisStreamHandler()
        redis_handler.subscribe_to_trading_channels()
        return redis_handler, True
    except Exception as e:
        st.error(f"Failed to connect to Redis: {e}")
        return None, False


def start_redis_listener(redis_handler):
    """Start Redis listener in background thread."""
    if 'redis_thread' not in st.session_state:
        thread = threading.Thread(target=redis_handler.start_listening, daemon=True)
        thread.start()
        st.session_state.redis_thread = thread


def init_shared_resources():
    """Initialize shared resources for all pages."""
    # Initialize database connection
    if 'db_manager' not in st.session_state:
        db_manager, db_connected = init_database()
        st.session_state.db_manager = db_manager
        st.session_state.db_connected = db_connected
    
    # Initialize Redis connection
    if 'redis_handler' not in st.session_state:
        redis_handler, redis_connected = init_redis()
        st.session_state.redis_handler = redis_handler
        st.session_state.redis_connected = redis_connected
        
        # Start Redis listener if connected
        if redis_connected and redis_handler:
            start_redis_listener(redis_handler)


def main():
    """Main dashboard landing page."""
    # Initialize shared resources
    init_shared_resources()
    
    # Main page content
    st.markdown('<h1 class="main-header">Kalshi Trading Dashboard</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the Kalshi Trading Dashboard. This multi-page application provides comprehensive 
    monitoring and control capabilities for your trading system.
    """)
    
    # Navigation cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📊 Trading Dashboard
        Monitor your trading performance with:
        - Real-time P&L tracking
        - Trade history and active positions
        - Performance metrics and analytics
        - Interactive profit & loss charts
        
        **[Go to Trading Dashboard →](Trading)**
        """)
        
    with col2:
        st.markdown("""
        ### 🤖 Agent Monitor
        Control and monitor your trading agents:
        - Agent status and health monitoring
        - Emergency stop controls
        - Quick action buttons
        - System statistics and logs
        
        **[Go to Agent Monitor →](Agent_Monitor)**
        """)
    
    st.markdown("---")
    
    # Connection status
    st.markdown("### System Status")
    
    col1, col2, col3 = st.columns(3)
    
    db_connected = st.session_state.get('db_connected', False)
    redis_connected = st.session_state.get('redis_connected', False)
    
    with col1:
        if db_connected:
            st.success("✅ Database Connected")
        else:
            st.error("❌ Database Disconnected")
    
    with col2:
        if redis_connected:
            st.success("✅ Redis Connected")
        else:
            st.error("❌ Redis Disconnected")
    
    with col3:
        if db_connected and redis_connected:
            st.success("✅ All Systems Operational")
        elif db_connected or redis_connected:
            st.warning("⚠️ Partial System Connection")
        else:
            st.error("❌ Systems Offline")
    
    # Quick stats
    if st.session_state.get('db_manager'):
        st.markdown("---")
        st.markdown("### Quick Statistics")
        
        db_manager = st.session_state.db_manager
        
        # Get quick stats
        try:
            agent_statuses = db_manager.get_agent_statuses()
            trades_data = db_manager.get_recent_trades(limit=10)
            positions_data = db_manager.get_active_positions()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                running_agents = len([a for a in agent_statuses if a.get('status') == 'running'])
                st.metric(
                    "Active Agents",
                    f"{running_agents}/{len(agent_statuses)}",
                    delta="Running" if running_agents else "Stopped"
                )
            
            with col2:
                st.metric(
                    "Active Positions",
                    len(positions_data),
                    delta="Open positions"
                )
            
            with col3:
                recent_trade_count = len(trades_data)
                st.metric(
                    "Recent Trades",
                    recent_trade_count,
                    delta="Last 10"
                )
            
            with col4:
                # Calculate simple P&L
                total_pnl = sum(float(t.get('realized_pnl', 0)) for t in trades_data)
                st.metric(
                    "Recent P&L",
                    f"${total_pnl:,.2f}",
                    delta="From recent trades"
                )
        except Exception as e:
            st.warning(f"Unable to load statistics: {e}")


# Sidebar navigation
with st.sidebar:
    st.markdown("## 🏠 Navigation")
    
    st.page_link("app.py", label="Home", icon="🏠")
    st.page_link("pages/1_📊_Trading.py", label="Trading Dashboard", icon="📊")
    st.page_link("pages/2_🤖_Agent_Monitor.py", label="Agent Monitor", icon="🤖")
    
    st.markdown("---")
    
    # System info
    st.markdown("### System Information")
    
    db_connected = st.session_state.get('db_connected', False)
    redis_connected = st.session_state.get('redis_connected', False)
    
    st.markdown(f"**Database:** {'🟢' if db_connected else '🔴'} {'Connected' if db_connected else 'Disconnected'}")
    st.markdown(f"**Redis:** {'🟢' if redis_connected else '🔴'} {'Connected' if redis_connected else 'Disconnected'}")
    
    st.markdown("---")
    
    # Quick links
    st.markdown("### Quick Links")
    
    if st.button("🔄 Refresh Connections"):
        # Clear cached resources to force reconnection
        st.cache_resource.clear()
        st.rerun()
    
    if st.button("📚 Documentation"):
        st.info("Documentation coming soon")
    
    if st.button("⚙️ Settings"):
        st.info("Settings page coming soon")


if __name__ == "__main__":
    main()