"""Control panel component with emergency stop functionality."""

import streamlit as st
from datetime import datetime
from typing import Dict, Any, List, Callable, Optional
import time


def create_control_panel(redis_handler, db_manager, agent_statuses: List[Dict[str, Any]]) -> None:
    """Create the control panel with emergency stop button and agent status.
    
    Args:
        redis_handler: Redis stream handler instance
        db_manager: Database manager instance
        agent_statuses: List of agent status records
    """
    # Create header with title and stop button
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.markdown("## 📊 Kalshi Trading Agent Monitor")
    
    with col3:
        # Emergency stop button - prominent and red
        if st.button("🛑 **EMERGENCY STOP**", 
                    type="primary", 
                    use_container_width=True,
                    help="Stop all agent operations immediately"):
            
            # Confirmation dialog
            if 'stop_confirmed' not in st.session_state:
                st.session_state.stop_confirmed = False
            
            if not st.session_state.stop_confirmed:
                st.warning("⚠️ **Are you sure?** This will stop all trading operations!")
                col_yes, col_no = st.columns(2)
                
                with col_yes:
                    if st.button("Yes, STOP ALL", type="primary"):
                        st.session_state.stop_confirmed = True
                        st.rerun()
                
                with col_no:
                    if st.button("Cancel"):
                        st.session_state.stop_confirmed = False
                        st.rerun()
            
            if st.session_state.get('stop_confirmed', False):
                # Execute emergency stop
                with st.spinner("Executing emergency stop..."):
                    # Send stop command via Redis
                    redis_handler.send_emergency_stop()
                    
                    # Update database
                    db_manager.emergency_stop_all_agents()
                    
                    # Log the action
                    st.error("🛑 **EMERGENCY STOP ACTIVATED**")
                    st.info("All agents have been stopped. Trading operations halted.")
                    
                    # Reset confirmation flag
                    st.session_state.stop_confirmed = False
                    
                    # Wait for visual feedback
                    time.sleep(2)
                    st.rerun()
    
    with col2:
        # System status indicator
        if agent_statuses:
            running_agents = [a for a in agent_statuses if a.get('status') == 'running']
            if len(running_agents) == len(agent_statuses):
                st.success("🟢 **RUNNING**")
            elif len(running_agents) > 0:
                st.warning(f"🟡 **PARTIAL** ({len(running_agents)}/{len(agent_statuses)})")
            else:
                st.error("🔴 **STOPPED**")
        else:
            st.info("⚪ **INITIALIZING**")


def create_agent_status_panel(agent_statuses: List[Dict[str, Any]]) -> None:
    """Create agent status panel showing health of each agent.
    
    Args:
        agent_statuses: List of agent status records
    """
    st.markdown("### Agent Status")
    
    if not agent_statuses:
        st.info("No agent status data available")
        return
    
    # Create columns for agent status cards
    cols = st.columns(5)  # 5 agents
    
    agent_names = ['DataCoordinator', 'StrategyAnalyst', 'MarketEngineer', 'TradeExecutor', 'RiskManager']
    agent_icons = {'DataCoordinator': '📡', 'StrategyAnalyst': '📈', 'MarketEngineer': '⚙️', 
                  'TradeExecutor': '💰', 'RiskManager': '🛡️'}
    
    for idx, agent_name in enumerate(agent_names):
        with cols[idx]:
            # Find agent status
            agent_data = next((a for a in agent_statuses if a.get('agent_name') == agent_name), None)
            
            if agent_data:
                status = agent_data.get('status', 'unknown')
                last_heartbeat = agent_data.get('last_heartbeat')
                messages = agent_data.get('messages_processed', 0)
                errors = agent_data.get('errors_count', 0)
                
                # Status color
                if status == 'running':
                    status_color = "🟢"
                    container_color = "green"
                elif status == 'paused':
                    status_color = "🟡"
                    container_color = "orange"
                elif status == 'error':
                    status_color = "🔴"
                    container_color = "red"
                else:
                    status_color = "⚫"
                    container_color = "gray"
                
                # Create status card
                with st.container():
                    st.markdown(f"{agent_icons.get(agent_name, '🤖')} **{agent_name}**")
                    st.markdown(f"{status_color} {status.upper()}")
                    
                    # Show metrics
                    if status == 'running':
                        st.caption(f"Messages: {messages}")
                        if errors > 0:
                            st.caption(f"⚠️ Errors: {errors}")
                    
                    # Heartbeat status
                    if last_heartbeat:
                        heartbeat_time = datetime.fromisoformat(last_heartbeat) if isinstance(last_heartbeat, str) else last_heartbeat
                        time_diff = (datetime.utcnow() - heartbeat_time).total_seconds()
                        
                        if time_diff < 60:
                            st.caption("💚 Active")
                        elif time_diff < 300:
                            st.caption("💛 Slow")
                        else:
                            st.caption("💔 Stale")
            else:
                # No data for this agent
                with st.container():
                    st.markdown(f"{agent_icons.get(agent_name, '🤖')} **{agent_name}**")
                    st.markdown("⚫ NO DATA")
    
    # Add separator
    st.markdown("---")


def create_quick_controls(db_manager, redis_handler) -> None:
    """Create quick control buttons for common operations.
    
    Args:
        db_manager: Database manager instance
        redis_handler: Redis handler instance
    """
    st.markdown("### Quick Controls")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("▶️ Start All Agents", use_container_width=True):
            with st.spinner("Starting agents..."):
                # Send start command via Redis
                start_command = {
                    'command': 'START',
                    'timestamp': datetime.utcnow().isoformat(),
                    'source': 'dashboard'
                }
                redis_handler.publish('agent:control', start_command)
                st.success("Start command sent to all agents")
                time.sleep(1)
                st.rerun()
    
    with col2:
        if st.button("⏸️ Pause Trading", use_container_width=True):
            with st.spinner("Pausing trading..."):
                pause_command = {
                    'command': 'PAUSE',
                    'timestamp': datetime.utcnow().isoformat(),
                    'source': 'dashboard',
                    'reason': 'User initiated pause'
                }
                redis_handler.publish('agent:control', pause_command)
                st.warning("Trading paused")
                time.sleep(1)
                st.rerun()
    
    with col3:
        if st.button("🔄 Resume Trading", use_container_width=True):
            with st.spinner("Resuming trading..."):
                resume_command = {
                    'command': 'RESUME',
                    'timestamp': datetime.utcnow().isoformat(),
                    'source': 'dashboard'
                }
                redis_handler.publish('agent:control', resume_command)
                st.success("Trading resumed")
                time.sleep(1)
                st.rerun()
    
    with col4:
        if st.button("📸 Snapshot P&L", use_container_width=True):
            with st.spinner("Creating P&L snapshot..."):
                # Calculate current P&L
                positions = db_manager.get_active_positions()
                trades = db_manager.get_recent_trades(limit=1000)
                
                # Calculate metrics (now working with dictionaries)
                total_realized = sum(float(t['realized_pnl']) for t in trades if t.get('realized_pnl'))
                total_unrealized = sum(float(p['unrealized_pnl']) for p in positions if p.get('unrealized_pnl'))
                total_pnl = total_realized + total_unrealized
                
                winning_trades = len([t for t in trades if float(t.get('realized_pnl', 0)) > 0])
                losing_trades = len([t for t in trades if float(t.get('realized_pnl', 0)) < 0])
                win_rate = winning_trades / (winning_trades + losing_trades) if (winning_trades + losing_trades) > 0 else 0
                
                # Record snapshot
                pnl_data = {
                    'total_pnl': total_pnl,
                    'realized_pnl': total_realized,
                    'unrealized_pnl': total_unrealized,
                    'win_count': winning_trades,
                    'loss_count': losing_trades,
                    'win_rate': win_rate,
                    'portfolio_value': sum(float(p.get('market_value', 0)) for p in positions)
                }
                
                db_manager.record_pnl_snapshot(pnl_data)
                st.success("P&L snapshot saved")
                time.sleep(1)
                st.rerun()


def create_connection_status(redis_connected: bool, db_connected: bool) -> None:
    """Display connection status for Redis and Database.
    
    Args:
        redis_connected: Redis connection status
        db_connected: Database connection status
    """
    col1, col2 = st.columns(2)
    
    with col1:
        if redis_connected:
            st.success("✅ Redis Connected")
        else:
            st.error("❌ Redis Disconnected")
    
    with col2:
        if db_connected:
            st.success("✅ Database Connected")
        else:
            st.error("❌ Database Disconnected")