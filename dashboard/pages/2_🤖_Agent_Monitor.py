"""Agent Monitor - System status and control page."""

import streamlit as st
import os
import sys
from datetime import datetime
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import dashboard modules
from data import DatabaseManager, RedisStreamHandler
from components import (
    create_control_panel,
    create_agent_status_panel,
    create_quick_controls
)

# Page configuration
st.set_page_config(
    page_title="Agent Monitor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for agent status cards
st.markdown("""
    <style>
    .agent-card {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.2);
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .stop-button {
        background-color: #ff4444;
        color: white;
        font-weight: bold;
        font-size: 18px;
    }
    </style>
    """, unsafe_allow_html=True)


def main():
    """Agent monitoring and control page."""
    
    # Page title
    st.title("🤖 Agent Monitor & Control")
    
    # Get shared resources from session state
    db_manager = st.session_state.get('db_manager')
    redis_handler = st.session_state.get('redis_handler')
    db_connected = st.session_state.get('db_connected', False)
    redis_connected = st.session_state.get('redis_connected', False)
    
    # Initialize auto-refresh for this page
    if 'agent_auto_refresh' not in st.session_state:
        st.session_state.agent_auto_refresh = True
    
    # Get agent status data
    if db_manager:
        agent_statuses = db_manager.get_agent_statuses()
    else:
        agent_statuses = []
    
    # Check for real-time agent updates from Redis
    if redis_handler and 'redis_data' in st.session_state:
        if 'agent:status' in st.session_state.redis_data:
            # Update agent statuses with real-time data
            status_update = st.session_state.redis_data['agent:status']['data']
            for agent in agent_statuses:
                if agent['agent_name'] == status_update.get('agent_name'):
                    agent.update(status_update)
                    break
    
    # Control panel with emergency stop button
    st.markdown("## System Controls")
    create_control_panel(redis_handler, db_manager, agent_statuses)
    
    # Agent status panel
    st.markdown("---")
    st.markdown("## Agent Status Overview")
    create_agent_status_panel(agent_statuses)
    
    # Quick controls
    st.markdown("---")
    st.markdown("## Quick Actions")
    if redis_handler and db_manager:
        create_quick_controls(db_manager, redis_handler)
    else:
        st.warning("Quick controls unavailable - Redis or Database not connected")
    
    # System Statistics
    st.markdown("---")
    st.markdown("## System Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate statistics
    if agent_statuses:
        running_agents = [a for a in agent_statuses if a.get('status') == 'running']
        total_messages = sum(a.get('messages_processed', 0) for a in agent_statuses)
        total_errors = sum(a.get('errors_count', 0) for a in agent_statuses)
        avg_uptime = 0  # Would calculate from start_time if available
        
        with col1:
            st.metric(
                "Active Agents",
                f"{len(running_agents)}/{len(agent_statuses)}",
                delta="Running" if running_agents else "All Stopped"
            )
        
        with col2:
            st.metric(
                "Messages Processed",
                f"{total_messages:,}",
                delta="Total"
            )
        
        with col3:
            st.metric(
                "Error Count",
                total_errors,
                delta="⚠️ Errors" if total_errors > 0 else "✅ No Errors",
                delta_color="inverse"
            )
        
        with col4:
            st.metric(
                "System Status",
                "Operational" if running_agents else "Stopped",
                delta="🟢 Live" if running_agents else "🔴 Offline"
            )
    else:
        st.info("No agent data available")
    
    # Agent Details Table
    st.markdown("---")
    st.markdown("## Agent Details")
    
    if agent_statuses:
        # Create a detailed table of agent information
        import pandas as pd
        
        agent_df = pd.DataFrame(agent_statuses)
        
        # Select columns to display
        display_columns = ['agent_name', 'status', 'messages_processed', 'errors_count', 'last_heartbeat']
        if all(col in agent_df.columns for col in display_columns):
            agent_df = agent_df[display_columns]
            
            # Format the dataframe for display
            agent_df['last_heartbeat'] = pd.to_datetime(agent_df['last_heartbeat'])
            agent_df['last_heartbeat'] = agent_df['last_heartbeat'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Apply styling
            def style_status(val):
                if val == 'running':
                    return 'color: #00ff88'
                elif val == 'stopped':
                    return 'color: #ff4444'
                elif val == 'error':
                    return 'color: #ff4444; font-weight: bold'
                elif val == 'paused':
                    return 'color: #ffaa00'
                return ''
            
            def style_errors(val):
                if val > 0:
                    return 'color: #ff4444; font-weight: bold'
                return 'color: #00ff88'
            
            styled_df = agent_df.style.applymap(style_status, subset=['status'])
            if 'errors_count' in agent_df.columns:
                styled_df = styled_df.applymap(style_errors, subset=['errors_count'])
            
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
    else:
        st.info("No agent details to display")
    
    # Connection Status
    st.markdown("---")
    st.markdown("## Connection Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if redis_connected:
            st.success("✅ Redis Connected")
            if redis_handler:
                st.caption(f"URL: {redis_handler.redis_url}")
        else:
            st.error("❌ Redis Disconnected")
    
    with col2:
        if db_connected:
            st.success("✅ Database Connected")
            if db_manager:
                st.caption("PostgreSQL Active")
        else:
            st.error("❌ Database Disconnected")
    
    with col3:
        system_health = "Healthy" if redis_connected and db_connected else "Degraded"
        if redis_connected and db_connected:
            st.success(f"✅ System {system_health}")
        else:
            st.warning(f"⚠️ System {system_health}")
    
    # Auto-refresh
    if st.session_state.agent_auto_refresh:
        time.sleep(2)  # Refresh every 2 seconds for agent monitoring
        st.rerun()


# Sidebar controls (specific to this page)
with st.sidebar:
    st.markdown("## Monitor Controls")
    
    # Auto-refresh toggle for agent monitor
    agent_auto_refresh = st.checkbox(
        "Auto Refresh",
        value=st.session_state.get('agent_auto_refresh', True),
        key="agent_refresh_toggle",
        help="Automatically refresh agent status every 2 seconds"
    )
    st.session_state.agent_auto_refresh = agent_auto_refresh
    
    if agent_auto_refresh:
        refresh_interval = st.slider(
            "Refresh Interval (seconds)",
            min_value=1,
            max_value=10,
            value=2,
            help="How often to refresh agent status"
        )
    
    st.markdown("---")
    
    # Agent filters
    st.markdown("### Filter Agents")
    
    status_filter = st.multiselect(
        "Status",
        options=["running", "stopped", "error", "paused"],
        default=["running", "stopped", "error", "paused"]
    )
    
    st.markdown("---")
    
    # Agent actions
    st.markdown("### Bulk Actions")
    
    if st.button("🔄 Restart All Agents", use_container_width=True):
        st.info("Restart functionality coming soon")
    
    if st.button("📝 View Logs", use_container_width=True):
        st.info("Log viewer coming soon")
    
    if st.button("⚙️ Configure Agents", use_container_width=True):
        st.info("Configuration panel coming soon")


if __name__ == "__main__":
    main()