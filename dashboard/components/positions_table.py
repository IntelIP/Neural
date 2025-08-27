"""Positions table component for the dashboard."""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any


def create_positions_table(positions_data: List[Dict[str, Any]], 
                          auto_refresh: bool = True) -> None:
    """Create an interactive positions table with real-time updates.
    
    Args:
        positions_data: List of position records
        auto_refresh: Enable auto-refresh for real-time updates
    """
    if not positions_data:
        st.info("No active positions")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(positions_data)
    
    # Calculate additional metrics if not present
    if 'current_price' in df.columns and 'entry_price' in df.columns:
        if 'unrealized_pnl' not in df.columns:
            df['unrealized_pnl'] = (df['current_price'] - df['entry_price']) * df['quantity']
        if 'unrealized_pnl_pct' not in df.columns:
            df['unrealized_pnl_pct'] = ((df['current_price'] - df['entry_price']) / df['entry_price'] * 100)
        if 'market_value' not in df.columns:
            df['market_value'] = df['current_price'] * df['quantity']
    
    # Format timestamp
    if 'opened_at' in df.columns:
        df['opened_at'] = pd.to_datetime(df['opened_at'])
        df['open_time'] = df['opened_at'].dt.strftime('%Y-%m-%d %H:%M')
    
    if 'updated_at' in df.columns:
        df['updated_at'] = pd.to_datetime(df['updated_at'])
        df['last_update'] = df['updated_at'].dt.strftime('%H:%M:%S')
    
    # Select columns to display
    display_columns = ['market_ticker', 'side', 'quantity', 'entry_price', 
                      'current_price', 'market_value', 'unrealized_pnl', 
                      'unrealized_pnl_pct', 'last_update']
    
    # Filter to only existing columns
    display_columns = [col for col in display_columns if col in df.columns]
    df_display = df[display_columns].copy()
    
    # Format columns for display
    format_config = {
        'entry_price': '{:.4f}',
        'current_price': '{:.4f}',
        'market_value': '{:.2f}',
        'unrealized_pnl': '{:.2f}',
        'unrealized_pnl_pct': '{:.2f}%'
    }
    
    # Apply formatting
    for col, fmt in format_config.items():
        if col in df_display.columns:
            if col == 'unrealized_pnl_pct':
                df_display[col] = df_display[col].apply(lambda x: fmt.format(x))
            else:
                df_display[col] = df_display[col].apply(lambda x: f"${fmt.format(x)}" if pd.notnull(x) else "")
    
    # Rename columns for display
    column_names = {
        'market_ticker': 'Market',
        'side': 'Side',
        'quantity': 'Qty',
        'entry_price': 'Entry',
        'current_price': 'Current',
        'market_value': 'Value',
        'unrealized_pnl': 'Unreal. P&L',
        'unrealized_pnl_pct': 'P&L %',
        'last_update': 'Updated'
    }
    
    df_display.rename(columns=column_names, inplace=True)
    
    # Apply color styling
    def style_dataframe(df):
        """Apply color styling to the dataframe."""
        def color_pnl(val):
            """Color P&L values based on profit/loss."""
            if isinstance(val, str) and '$' in val:
                numeric_val = float(val.replace('$', '').replace(',', ''))
                if numeric_val > 0:
                    return 'color: #00ff88; font-weight: bold'
                elif numeric_val < 0:
                    return 'color: #ff4444; font-weight: bold'
            return ''
        
        def color_pnl_pct(val):
            """Color P&L percentage values."""
            if isinstance(val, str) and '%' in val:
                numeric_val = float(val.replace('%', ''))
                if numeric_val > 0:
                    return 'color: #00ff88; font-weight: bold'
                elif numeric_val < 0:
                    return 'color: #ff4444; font-weight: bold'
            return ''
        
        def color_side(val):
            """Color side values."""
            if val == 'yes':
                return 'color: #00ff88'
            elif val == 'no':
                return 'color: #ff4444'
            return ''
        
        # Create styler
        styler = df.style
        
        # Apply styles to specific columns
        if 'Unreal. P&L' in df.columns:
            styler = styler.applymap(color_pnl, subset=['Unreal. P&L'])
        if 'P&L %' in df.columns:
            styler = styler.applymap(color_pnl_pct, subset=['P&L %'])
        if 'Side' in df.columns:
            styler = styler.applymap(color_side, subset=['Side'])
        
        return styler
    
    # Display the table with styling
    styled_df = style_dataframe(df_display)
    
    # Add auto-refresh indicator
    if auto_refresh:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown("##### Active Positions")
        with col2:
            st.markdown("🔄 **Live**")
    else:
        st.markdown("##### Active Positions")
    
    # Display the dataframe
    st.dataframe(
        styled_df,
        height=400,
        use_container_width=True,
        hide_index=True
    )
    
    # Display position summary
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        
        total_positions = len(df)
        if 'market_value' in df.columns:
            total_value = df['market_value'].sum() if df['market_value'].dtype in ['float64', 'int64'] else 0
        else:
            total_value = 0
        
        if 'unrealized_pnl' in df.columns:
            total_unrealized = df['unrealized_pnl'].sum() if df['unrealized_pnl'].dtype in ['float64', 'int64'] else 0
            profitable_positions = len(df[df['unrealized_pnl'] > 0])
        else:
            total_unrealized = 0
            profitable_positions = 0
        
        col1.metric("Open Positions", total_positions)
        col2.metric("Market Value", f"${total_value:,.2f}")
        col3.metric("Unrealized P&L", f"${total_unrealized:,.2f}", 
                   delta="Profit" if total_unrealized > 0 else "Loss" if total_unrealized < 0 else None)
        col4.metric("Profitable", f"{profitable_positions}/{total_positions}", 
                   delta=f"{profitable_positions/total_positions*100:.1f}%" if total_positions > 0 else "0%")


def create_position_risk_metrics(positions_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate risk metrics for positions.
    
    Args:
        positions_data: List of position records
        
    Returns:
        Dictionary of risk metrics
    """
    if not positions_data:
        return {
            'total_exposure': 0,
            'max_position_size': 0,
            'concentration_risk': 0,
            'total_stop_loss_risk': 0
        }
    
    df = pd.DataFrame(positions_data)
    
    # Calculate metrics
    metrics = {}
    
    # Total exposure (sum of market values)
    if 'market_value' in df.columns:
        metrics['total_exposure'] = float(df['market_value'].sum())
        metrics['max_position_size'] = float(df['market_value'].max())
        
        # Concentration risk (largest position as % of total)
        if metrics['total_exposure'] > 0:
            metrics['concentration_risk'] = (metrics['max_position_size'] / metrics['total_exposure'] * 100)
        else:
            metrics['concentration_risk'] = 0
    
    # Stop loss risk
    if 'stop_loss_price' in df.columns and 'current_price' in df.columns and 'quantity' in df.columns:
        df['stop_loss_risk'] = df.apply(
            lambda row: (row['current_price'] - row['stop_loss_price']) * row['quantity'] 
            if pd.notnull(row['stop_loss_price']) else 0,
            axis=1
        )
        metrics['total_stop_loss_risk'] = float(df['stop_loss_risk'].sum())
    else:
        metrics['total_stop_loss_risk'] = 0
    
    return metrics