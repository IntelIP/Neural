"""Trades table component for the dashboard."""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode


def create_trades_table(trades_data: List[Dict[str, Any]], 
                       height: int = 400,
                       enable_filtering: bool = True,
                       enable_sorting: bool = True) -> None:
    """Create an interactive trades table using AgGrid.
    
    Args:
        trades_data: List of trade records
        height: Table height in pixels
        enable_filtering: Enable column filtering
        enable_sorting: Enable column sorting
    """
    if not trades_data:
        st.info("No trades to display yet")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(trades_data)
    
    # Format timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['time'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Select columns to display
    display_columns = ['time', 'market_ticker', 'side', 'quantity', 'price', 
                      'total_cost', 'realized_pnl', 'status', 'strategy']
    
    # Filter to only existing columns
    display_columns = [col for col in display_columns if col in df.columns or col == 'time']
    df_display = df[display_columns].copy()
    
    # Format numeric columns
    if 'price' in df_display.columns:
        df_display['price'] = df_display['price'].apply(lambda x: f"${x:.4f}")
    if 'total_cost' in df_display.columns:
        df_display['total_cost'] = df_display['total_cost'].apply(lambda x: f"${x:.2f}")
    if 'realized_pnl' in df_display.columns:
        df_display['realized_pnl'] = df_display['realized_pnl'].apply(lambda x: f"${x:.2f}")
    
    # Configure AgGrid
    gb = GridOptionsBuilder.from_dataframe(df_display)
    
    # Configure column properties
    gb.configure_column("time", header_name="Time", width=180)
    gb.configure_column("market_ticker", header_name="Market", width=150)
    gb.configure_column("side", header_name="Side", width=80)
    gb.configure_column("quantity", header_name="Quantity", width=100)
    gb.configure_column("price", header_name="Price", width=100)
    gb.configure_column("total_cost", header_name="Total Cost", width=120)
    gb.configure_column("status", header_name="Status", width=100)
    gb.configure_column("strategy", header_name="Strategy", width=120)
    
    # Configure P&L column with color coding
    if 'realized_pnl' in df_display.columns:
        pnl_style = JsCode("""
            function(params) {
                if (params.value) {
                    const value = parseFloat(params.value.replace('$', '').replace(',', ''));
                    if (value > 0) {
                        return {
                            'color': '#00ff88',
                            'fontWeight': 'bold'
                        }
                    } else if (value < 0) {
                        return {
                            'color': '#ff4444',
                            'fontWeight': 'bold'
                        }
                    }
                }
                return {}
            }
        """)
        gb.configure_column("realized_pnl", header_name="P&L", width=120, cellStyle=pnl_style)
    
    # Configure status column with color coding
    if 'status' in df_display.columns:
        status_style = JsCode("""
            function(params) {
                if (params.value === 'filled') {
                    return {'color': '#00ff88'}
                } else if (params.value === 'pending') {
                    return {'color': '#ffaa00'}
                } else if (params.value === 'cancelled' || params.value === 'failed') {
                    return {'color': '#ff4444'}
                }
                return {}
            }
        """)
        gb.configure_column("status", cellStyle=status_style)
    
    # Configure side column with color coding
    if 'side' in df_display.columns:
        side_style = JsCode("""
            function(params) {
                if (params.value === 'buy') {
                    return {'color': '#00ff88'}
                } else if (params.value === 'sell') {
                    return {'color': '#ff4444'}
                }
                return {}
            }
        """)
        gb.configure_column("side", cellStyle=side_style)
    
    # Configure grid options
    gb.configure_default_column(
        filterable=enable_filtering,
        sortable=enable_sorting,
        resizable=True
    )
    
    gb.configure_grid_options(
        domLayout='normal',
        enableRangeSelection=True,
        rowSelection='multiple',
        suppressRowClickSelection=False
    )
    
    gb.configure_pagination(enabled=True, paginationAutoPageSize=False, paginationPageSize=20)
    
    grid_options = gb.build()
    
    # Display the grid
    grid_response = AgGrid(
        df_display,
        gridOptions=grid_options,
        height=height,
        theme='streamlit',  # 'streamlit', 'light', 'dark', 'blue', 'fresh', 'material'
        update_mode=GridUpdateMode.MODEL_CHANGED,
        allow_unsafe_jscode=True,
        key='trades_table'
    )
    
    # Display summary statistics
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        
        total_trades = len(df)
        if 'realized_pnl' in df.columns:
            total_pnl = df['realized_pnl'].sum()
            winning_trades = len(df[df['realized_pnl'] > 0])
            losing_trades = len(df[df['realized_pnl'] < 0])
        else:
            total_pnl = 0
            winning_trades = 0
            losing_trades = 0
        
        col1.metric("Total Trades", total_trades)
        col2.metric("Winning Trades", winning_trades, delta=f"{winning_trades/total_trades*100:.1f}%" if total_trades > 0 else "0%")
        col3.metric("Losing Trades", losing_trades, delta=f"{losing_trades/total_trades*100:.1f}%" if total_trades > 0 else "0%")
        col4.metric("Total P&L", f"${total_pnl:,.2f}", delta="Profit" if total_pnl > 0 else "Loss")


def create_simple_trades_table(trades_data: List[Dict[str, Any]]) -> None:
    """Create a simple trades table using standard Streamlit dataframe.
    
    Args:
        trades_data: List of trade records
    """
    if not trades_data:
        st.info("No trades to display yet")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(trades_data)
    
    # Format timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['time'] = df['timestamp'].dt.strftime('%H:%M:%S')
        df['date'] = df['timestamp'].dt.strftime('%Y-%m-%d')
    
    # Select and reorder columns for display
    display_columns = []
    if 'time' in df.columns:
        display_columns.append('time')
    if 'market_ticker' in df.columns:
        display_columns.append('market_ticker')
    if 'side' in df.columns:
        display_columns.append('side')
    if 'quantity' in df.columns:
        display_columns.append('quantity')
    if 'price' in df.columns:
        display_columns.append('price')
    if 'realized_pnl' in df.columns:
        display_columns.append('realized_pnl')
    if 'status' in df.columns:
        display_columns.append('status')
    
    df_display = df[display_columns].copy() if display_columns else df
    
    # Apply color styling
    def color_pnl(val):
        """Color P&L values."""
        if isinstance(val, (int, float)):
            color = '#00ff88' if val > 0 else '#ff4444' if val < 0 else 'white'
            return f'color: {color}; font-weight: bold'
        return ''
    
    def color_side(val):
        """Color side values."""
        if val == 'buy':
            return 'color: #00ff88'
        elif val == 'sell':
            return 'color: #ff4444'
        return ''
    
    def color_status(val):
        """Color status values."""
        if val == 'filled':
            return 'color: #00ff88'
        elif val == 'pending':
            return 'color: #ffaa00'
        elif val in ['cancelled', 'failed']:
            return 'color: #ff4444'
        return ''
    
    # Create style dict
    style_dict = {}
    if 'realized_pnl' in df_display.columns:
        style_dict['realized_pnl'] = color_pnl
    if 'side' in df_display.columns:
        style_dict['side'] = color_side
    if 'status' in df_display.columns:
        style_dict['status'] = color_status
    
    # Apply styling and display
    if style_dict:
        styled_df = df_display.style.applymap(
            lambda x: style_dict.get(df_display.columns[df_display.columns.get_loc(x.name)], lambda y: '')(x),
            subset=list(style_dict.keys())
        )
        st.dataframe(styled_df, height=400, use_container_width=True)
    else:
        st.dataframe(df_display, height=400, use_container_width=True)