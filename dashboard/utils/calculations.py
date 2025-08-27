"""Utility functions for calculations."""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
import numpy as np
import pandas as pd


def calculate_portfolio_metrics(trades: List[Dict[str, Any]], 
                               positions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate comprehensive portfolio metrics.
    
    Args:
        trades: List of trade records
        positions: List of position records
        
    Returns:
        Dictionary of portfolio metrics
    """
    metrics = {}
    
    # Trade metrics
    if trades:
        trades_df = pd.DataFrame(trades)
        
        # Basic counts
        metrics['total_trades'] = len(trades)
        
        # P&L calculations
        if 'realized_pnl' in trades_df.columns:
            trades_df['realized_pnl'] = pd.to_numeric(trades_df['realized_pnl'], errors='coerce')
            metrics['total_realized_pnl'] = float(trades_df['realized_pnl'].sum())
            metrics['avg_trade_pnl'] = float(trades_df['realized_pnl'].mean())
            metrics['max_win'] = float(trades_df['realized_pnl'].max())
            metrics['max_loss'] = float(trades_df['realized_pnl'].min())
            
            # Win/loss statistics
            winning_trades = trades_df[trades_df['realized_pnl'] > 0]
            losing_trades = trades_df[trades_df['realized_pnl'] < 0]
            
            metrics['win_count'] = len(winning_trades)
            metrics['loss_count'] = len(losing_trades)
            metrics['win_rate'] = (metrics['win_count'] / metrics['total_trades'] * 100) if metrics['total_trades'] > 0 else 0
            
            # Average win/loss
            metrics['avg_win'] = float(winning_trades['realized_pnl'].mean()) if len(winning_trades) > 0 else 0
            metrics['avg_loss'] = float(losing_trades['realized_pnl'].mean()) if len(losing_trades) > 0 else 0
            
            # Profit factor
            total_wins = winning_trades['realized_pnl'].sum() if len(winning_trades) > 0 else 0
            total_losses = abs(losing_trades['realized_pnl'].sum()) if len(losing_trades) > 0 else 0
            metrics['profit_factor'] = float(total_wins / total_losses) if total_losses > 0 else float('inf') if total_wins > 0 else 0
    else:
        metrics.update({
            'total_trades': 0,
            'total_realized_pnl': 0,
            'win_count': 0,
            'loss_count': 0,
            'win_rate': 0,
            'profit_factor': 0
        })
    
    # Position metrics
    if positions:
        positions_df = pd.DataFrame(positions)
        
        metrics['open_positions'] = len(positions)
        
        if 'market_value' in positions_df.columns:
            positions_df['market_value'] = pd.to_numeric(positions_df['market_value'], errors='coerce')
            metrics['total_market_value'] = float(positions_df['market_value'].sum())
        
        if 'unrealized_pnl' in positions_df.columns:
            positions_df['unrealized_pnl'] = pd.to_numeric(positions_df['unrealized_pnl'], errors='coerce')
            metrics['total_unrealized_pnl'] = float(positions_df['unrealized_pnl'].sum())
            
            # Profitable positions
            profitable = positions_df[positions_df['unrealized_pnl'] > 0]
            metrics['profitable_positions'] = len(profitable)
            metrics['profitable_positions_pct'] = (metrics['profitable_positions'] / metrics['open_positions'] * 100) if metrics['open_positions'] > 0 else 0
    else:
        metrics.update({
            'open_positions': 0,
            'total_market_value': 0,
            'total_unrealized_pnl': 0,
            'profitable_positions': 0,
            'profitable_positions_pct': 0
        })
    
    # Total P&L
    metrics['total_pnl'] = metrics.get('total_realized_pnl', 0) + metrics.get('total_unrealized_pnl', 0)
    
    return metrics


def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio for given returns.
    
    Args:
        returns: List of period returns
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Sharpe ratio
    """
    if not returns or len(returns) < 2:
        return 0.0
    
    returns_array = np.array(returns)
    
    # Calculate excess returns
    daily_rf_rate = risk_free_rate / 252  # Assuming 252 trading days
    excess_returns = returns_array - daily_rf_rate
    
    # Calculate Sharpe ratio
    mean_excess_return = np.mean(excess_returns)
    std_excess_return = np.std(excess_returns)
    
    if std_excess_return == 0:
        return 0.0
    
    # Annualize the Sharpe ratio
    sharpe_ratio = mean_excess_return / std_excess_return * np.sqrt(252)
    
    return float(sharpe_ratio)


def calculate_max_drawdown(values: List[float]) -> Tuple[float, float]:
    """Calculate maximum drawdown and percentage.
    
    Args:
        values: List of portfolio values or P&L over time
        
    Returns:
        Tuple of (max_drawdown_value, max_drawdown_percentage)
    """
    if not values or len(values) < 2:
        return (0.0, 0.0)
    
    values_array = np.array(values)
    cumulative_max = np.maximum.accumulate(values_array)
    drawdown = values_array - cumulative_max
    
    max_drawdown = float(np.min(drawdown))
    
    # Calculate percentage drawdown
    max_value_at_drawdown = cumulative_max[np.argmin(drawdown)]
    max_drawdown_pct = (max_drawdown / max_value_at_drawdown * 100) if max_value_at_drawdown != 0 else 0
    
    return (max_drawdown, max_drawdown_pct)


def calculate_kelly_fraction(win_probability: float, 
                            win_amount: float, 
                            loss_amount: float) -> float:
    """Calculate Kelly Criterion fraction for position sizing.
    
    Args:
        win_probability: Probability of winning (0-1)
        win_amount: Average win amount
        loss_amount: Average loss amount (positive value)
        
    Returns:
        Kelly fraction (0-1)
    """
    if loss_amount <= 0 or win_amount <= 0:
        return 0.0
    
    # Kelly formula: f = (p*b - q) / b
    # where p = win probability, q = loss probability, b = win/loss ratio
    loss_probability = 1 - win_probability
    win_loss_ratio = win_amount / loss_amount
    
    kelly = (win_probability * win_loss_ratio - loss_probability) / win_loss_ratio
    
    # Cap Kelly fraction at 25% for safety
    kelly = max(0, min(kelly, 0.25))
    
    return float(kelly)


def calculate_rolling_metrics(pnl_data: List[Dict[str, Any]], 
                             window: int = 30) -> pd.DataFrame:
    """Calculate rolling metrics for P&L data.
    
    Args:
        pnl_data: List of P&L records with timestamp and values
        window: Rolling window size in days
        
    Returns:
        DataFrame with rolling metrics
    """
    if not pnl_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(pnl_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    df.set_index('timestamp', inplace=True)
    
    # Calculate rolling metrics
    metrics_df = pd.DataFrame(index=df.index)
    
    if 'total_pnl' in df.columns:
        metrics_df['rolling_mean'] = df['total_pnl'].rolling(window=window).mean()
        metrics_df['rolling_std'] = df['total_pnl'].rolling(window=window).std()
        metrics_df['rolling_sharpe'] = (metrics_df['rolling_mean'] / metrics_df['rolling_std']) * np.sqrt(252)
    
    if 'daily_pnl' in df.columns:
        metrics_df['rolling_daily_mean'] = df['daily_pnl'].rolling(window=window).mean()
        metrics_df['rolling_daily_volatility'] = df['daily_pnl'].rolling(window=window).std()
    
    return metrics_df


def calculate_risk_metrics(positions: List[Dict[str, Any]], 
                          portfolio_value: float) -> Dict[str, Any]:
    """Calculate risk metrics for current positions.
    
    Args:
        positions: List of position records
        portfolio_value: Total portfolio value
        
    Returns:
        Dictionary of risk metrics
    """
    if not positions or portfolio_value <= 0:
        return {
            'value_at_risk': 0,
            'concentration_risk': 0,
            'max_position_pct': 0,
            'leverage': 0
        }
    
    positions_df = pd.DataFrame(positions)
    
    metrics = {}
    
    # Value at Risk (simplified - using position sizes)
    if 'market_value' in positions_df.columns:
        positions_df['market_value'] = pd.to_numeric(positions_df['market_value'], errors='coerce')
        position_values = positions_df['market_value'].values
        
        # 95% VaR using historical simulation (simplified)
        metrics['value_at_risk'] = float(np.percentile(position_values, 5)) if len(position_values) > 0 else 0
        
        # Concentration risk
        max_position = positions_df['market_value'].max()
        metrics['max_position_pct'] = float(max_position / portfolio_value * 100) if portfolio_value > 0 else 0
        
        # HHI (Herfindahl-Hirschman Index) for concentration
        position_weights = position_values / portfolio_value
        metrics['concentration_risk'] = float(np.sum(position_weights ** 2))
        
        # Leverage
        total_exposure = positions_df['market_value'].sum()
        metrics['leverage'] = float(total_exposure / portfolio_value) if portfolio_value > 0 else 0
    
    return metrics