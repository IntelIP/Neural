"""Profit and Loss chart component for the dashboard."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import numpy as np


def create_pnl_chart(pnl_data: List[Dict[str, Any]], 
                     trades_data: Optional[List[Dict[str, Any]]] = None,
                     date_range: Optional[tuple] = None) -> go.Figure:
    """Create an interactive P&L line chart with trade markers.
    
    Args:
        pnl_data: List of P&L records with timestamp and values
        trades_data: Optional list of trades to overlay on chart
        date_range: Optional tuple of (start_date, end_date) for filtering
        
    Returns:
        Plotly figure object
    """
    # Convert to DataFrame for easier manipulation
    if not pnl_data:
        # Return empty chart if no data
        fig = go.Figure()
        fig.add_annotation(
            text="No P&L data available yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(
            title="Profit & Loss Over Time",
            height=500,
            template="plotly_dark"
        )
        return fig
    
    df = pd.DataFrame(pnl_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Apply date range filter if provided
    if date_range:
        start_date, end_date = date_range
        df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("Cumulative P&L", "Daily P&L")
    )
    
    # Main P&L line
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['total_pnl'],
            mode='lines',
            name='Total P&L',
            line=dict(color='#00d4ff', width=3),
            fill='tozeroy',
            fillcolor='rgba(0, 212, 255, 0.1)',
            hovertemplate='<b>Total P&L</b>: $%{y:,.2f}<br>Date: %{x}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add realized and unrealized P&L if available
    if 'realized_pnl' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['realized_pnl'],
                mode='lines',
                name='Realized P&L',
                line=dict(color='#00ff88', width=2, dash='dot'),
                hovertemplate='<b>Realized P&L</b>: $%{y:,.2f}<br>Date: %{x}<extra></extra>'
            ),
            row=1, col=1
        )
    
    if 'unrealized_pnl' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['unrealized_pnl'],
                mode='lines',
                name='Unrealized P&L',
                line=dict(color='#ff9500', width=2, dash='dash'),
                hovertemplate='<b>Unrealized P&L</b>: $%{y:,.2f}<br>Date: %{x}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Add zero line for reference
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
    
    # Daily P&L bar chart
    if 'daily_pnl' in df.columns:
        colors = ['green' if x > 0 else 'red' for x in df['daily_pnl']]
        fig.add_trace(
            go.Bar(
                x=df['timestamp'],
                y=df['daily_pnl'],
                name='Daily P&L',
                marker_color=colors,
                hovertemplate='<b>Daily P&L</b>: $%{y:,.2f}<br>Date: %{x}<extra></extra>'
            ),
            row=2, col=1
        )
        fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.3, row=2, col=1)
    
    # Add trade markers if provided
    if trades_data:
        trades_df = pd.DataFrame(trades_data)
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        
        # Filter trades within date range
        if date_range:
            trades_df = trades_df[
                (trades_df['timestamp'] >= date_range[0]) & 
                (trades_df['timestamp'] <= date_range[1])
            ]
        
        # Group trades by strategy/type for different colors
        for strategy in trades_df['strategy'].unique() if 'strategy' in trades_df.columns else ['default']:
            strategy_trades = trades_df[trades_df['strategy'] == strategy] if 'strategy' in trades_df.columns else trades_df
            
            # Get P&L values at trade times (interpolate if needed)
            trade_pnl_values = []
            for trade_time in strategy_trades['timestamp']:
                closest_idx = df['timestamp'].searchsorted(trade_time)
                if closest_idx < len(df):
                    trade_pnl_values.append(df.iloc[closest_idx]['total_pnl'])
                else:
                    trade_pnl_values.append(df.iloc[-1]['total_pnl'])
            
            # Color based on trade profitability
            colors = ['green' if float(t.get('realized_pnl', 0)) > 0 else 'red' 
                     for _, t in strategy_trades.iterrows()]
            
            fig.add_trace(
                go.Scatter(
                    x=strategy_trades['timestamp'],
                    y=trade_pnl_values,
                    mode='markers',
                    name=f'Trades ({strategy})',
                    marker=dict(
                        size=10,
                        color=colors,
                        symbol='triangle-up' if strategy == 'buy' else 'triangle-down',
                        line=dict(color='white', width=1)
                    ),
                    hovertemplate='<b>Trade</b><br>' +
                                 'Market: %{customdata[0]}<br>' +
                                 'Side: %{customdata[1]}<br>' +
                                 'Quantity: %{customdata[2]}<br>' +
                                 'Price: $%{customdata[3]:,.2f}<br>' +
                                 'P&L: $%{customdata[4]:,.2f}<extra></extra>',
                    customdata=strategy_trades[['market_ticker', 'side', 'quantity', 'price', 'realized_pnl']].values
                ),
                row=1, col=1
            )
    
    # Calculate and add statistics annotations
    if len(df) > 0:
        total_return = df.iloc[-1]['total_pnl']
        max_drawdown = calculate_max_drawdown(df['total_pnl'].values)
        win_rate = df.iloc[-1].get('win_rate', 0) * 100 if 'win_rate' in df.columns else 0
        
        stats_text = f"Total Return: ${total_return:,.2f} | Max Drawdown: ${max_drawdown:,.2f} | Win Rate: {win_rate:.1f}%"
        fig.add_annotation(
            text=stats_text,
            xref="paper", yref="paper",
            x=0.5, y=1.15,
            showarrow=False,
            font=dict(size=12, color="white"),
            bgcolor="rgba(0,0,0,0.5)",
            borderpad=4
        )
    
    # Update layout
    fig.update_layout(
        title={
            'text': "Profit & Loss Performance",
            'x': 0.5,
            'xanchor': 'center'
        },
        template="plotly_dark",
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified',
        xaxis=dict(
            rangeslider=dict(visible=False),
            type='date'
        ),
        yaxis=dict(
            title="P&L ($)",
            tickformat="$,.0f",
            zeroline=True,
            zerolinecolor='gray',
            zerolinewidth=1
        ),
        yaxis2=dict(
            title="Daily P&L ($)",
            tickformat="$,.0f"
        )
    )
    
    # Add range selector buttons
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1D", step="day", stepmode="backward"),
                dict(count=7, label="1W", step="day", stepmode="backward"),
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(step="all", label="All")
            ]),
            bgcolor="rgba(0,0,0,0.5)",
            activecolor="rgba(0,212,255,0.5)",
            font=dict(color="white")
        ),
        row=1, col=1
    )
    
    return fig


def calculate_max_drawdown(pnl_values: np.ndarray) -> float:
    """Calculate maximum drawdown from P&L values.
    
    Args:
        pnl_values: Array of P&L values
        
    Returns:
        Maximum drawdown value
    """
    if len(pnl_values) == 0:
        return 0.0
    
    cumulative_max = np.maximum.accumulate(pnl_values)
    drawdown = pnl_values - cumulative_max
    return float(np.min(drawdown))


def create_performance_metrics_cards(pnl_data: List[Dict[str, Any]], 
                                    trades_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create performance metrics for display cards.
    
    Args:
        pnl_data: List of P&L records
        trades_data: List of trade records
        
    Returns:
        Dictionary of performance metrics
    """
    if not pnl_data:
        return {
            'total_pnl': 0,
            'daily_pnl': 0,
            'win_rate': 0,
            'total_trades': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0
        }
    
    df = pd.DataFrame(pnl_data)
    latest_record = df.iloc[-1] if len(df) > 0 else {}
    
    # Calculate additional metrics
    total_trades = len(trades_data) if trades_data else 0
    winning_trades = len([t for t in trades_data if float(t.get('realized_pnl', 0)) > 0]) if trades_data else 0
    
    return {
        'total_pnl': float(latest_record.get('total_pnl', 0)),
        'daily_pnl': float(latest_record.get('daily_pnl', 0)),
        'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
        'total_trades': total_trades,
        'sharpe_ratio': float(latest_record.get('sharpe_ratio', 0)),
        'max_drawdown': calculate_max_drawdown(df['total_pnl'].values) if 'total_pnl' in df.columns else 0,
        'portfolio_value': float(latest_record.get('portfolio_value', 0)),
        'cash_balance': float(latest_record.get('cash_balance', 0))
    }