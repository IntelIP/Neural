"""
Performance metrics calculations for Neural SDK Analysis Infrastructure.

This module provides comprehensive metrics for evaluating trading strategies,
including risk-adjusted returns, drawdown analysis, and statistical measures.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of performance metrics."""
    RETURN = "return"
    RISK = "risk"
    RISK_ADJUSTED = "risk_adjusted"
    DRAWDOWN = "drawdown"
    TRADING = "trading"
    CONSISTENCY = "consistency"


@dataclass
class PerformanceMetrics:
    """
    Comprehensive performance metrics for a trading strategy.
    """
    # Return Metrics
    total_return: float = 0.0
    annual_return: float = 0.0
    monthly_returns: List[float] = None
    
    # Risk Metrics
    volatility: float = 0.0
    annual_volatility: float = 0.0
    downside_deviation: float = 0.0
    value_at_risk_95: float = 0.0
    
    # Risk-Adjusted Metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    omega_ratio: float = 0.0
    
    # Drawdown Metrics
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    current_drawdown: float = 0.0
    drawdown_periods: int = 0
    
    # Trading Metrics
    total_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    kelly_criterion: float = 0.0
    
    # Consistency Metrics
    hit_ratio: float = 0.0
    consistency_score: float = 0.0
    tail_ratio: float = 0.0
    
    # Metadata
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    days_traded: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        result = {}
        for field_name, field_value in self.__dict__.items():
            if field_name in ['start_date', 'end_date'] and field_value:
                result[field_name] = field_value.isoformat()
            elif field_name == 'monthly_returns' and field_value:
                result[field_name] = field_value
            else:
                result[field_name] = field_value
        return result


class PerformanceCalculator:
    """
    Calculates comprehensive performance metrics for trading strategies.
    
    This class provides methods to analyze P&L series, trade data, and
    portfolio performance with industry-standard metrics.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize performance calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_comprehensive_metrics(
        self,
        pnl_series: pd.Series,
        trades_data: Optional[List[Dict[str, Any]]] = None,
        initial_capital: float = 1000
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            pnl_series: Time series of P&L values (cumulative)
            trades_data: Optional individual trade data
            initial_capital: Starting capital amount
            
        Returns:
            PerformanceMetrics object with all calculated metrics
        """
        if len(pnl_series) < 2:
            return PerformanceMetrics()
        
        # Calculate returns series
        returns = pnl_series.pct_change().dropna()
        
        # Initialize metrics object
        metrics = PerformanceMetrics()
        
        # Basic info
        metrics.start_date = pnl_series.index[0] if hasattr(pnl_series.index[0], 'date') else None
        metrics.end_date = pnl_series.index[-1] if hasattr(pnl_series.index[-1], 'date') else None
        metrics.days_traded = len(pnl_series)
        
        # Return metrics
        metrics.total_return = (pnl_series.iloc[-1] - pnl_series.iloc[0]) / initial_capital
        metrics.annual_return = self._annualize_return(returns)
        metrics.monthly_returns = self._calculate_monthly_returns(returns)
        
        # Risk metrics
        metrics.volatility = returns.std()
        metrics.annual_volatility = returns.std() * np.sqrt(252)  # Assuming daily data
        metrics.downside_deviation = self._calculate_downside_deviation(returns)
        metrics.value_at_risk_95 = self._calculate_var(returns, 0.95)
        
        # Risk-adjusted metrics
        metrics.sharpe_ratio = self._calculate_sharpe_ratio(returns)
        metrics.sortino_ratio = self._calculate_sortino_ratio(returns)
        metrics.calmar_ratio = self._calculate_calmar_ratio(returns, metrics.max_drawdown)
        metrics.omega_ratio = self._calculate_omega_ratio(returns)
        
        # Drawdown metrics
        drawdown_metrics = self._calculate_drawdown_metrics(pnl_series)
        metrics.max_drawdown = drawdown_metrics['max_drawdown']
        metrics.max_drawdown_duration = drawdown_metrics['max_drawdown_duration']
        metrics.current_drawdown = drawdown_metrics['current_drawdown']
        metrics.drawdown_periods = drawdown_metrics['drawdown_periods']
        
        # Trading metrics (if trade data available)
        if trades_data:
            trade_metrics = self._calculate_trade_metrics(trades_data)
            metrics.total_trades = trade_metrics['total_trades']
            metrics.win_rate = trade_metrics['win_rate']
            metrics.avg_win = trade_metrics['avg_win']
            metrics.avg_loss = trade_metrics['avg_loss']
            metrics.profit_factor = trade_metrics['profit_factor']
            metrics.kelly_criterion = trade_metrics['kelly_criterion']
        
        # Consistency metrics
        consistency_metrics = self._calculate_consistency_metrics(returns)
        metrics.hit_ratio = consistency_metrics['hit_ratio']
        metrics.consistency_score = consistency_metrics['consistency_score']
        metrics.tail_ratio = consistency_metrics['tail_ratio']
        
        return metrics
    
    def _annualize_return(self, returns: pd.Series) -> float:
        """Calculate annualized return."""
        if len(returns) == 0:
            return 0.0
        
        # Assume daily returns
        periods_per_year = 252
        
        # Compound returns
        cumulative_return = (1 + returns).prod() - 1
        years = len(returns) / periods_per_year
        
        if years == 0 or cumulative_return <= -1:
            return 0.0
        
        return (1 + cumulative_return) ** (1 / years) - 1
    
    def _calculate_monthly_returns(self, returns: pd.Series) -> List[float]:
        """Calculate monthly returns."""
        if not hasattr(returns.index, 'to_period'):
            return []
        
        try:
            monthly = returns.groupby(returns.index.to_period('M')).apply(
                lambda x: (1 + x).prod() - 1
            )
            return monthly.tolist()
        except:
            return []
    
    def _calculate_downside_deviation(self, returns: pd.Series, threshold: float = 0.0) -> float:
        """Calculate downside deviation."""
        downside_returns = returns[returns < threshold]
        if len(downside_returns) == 0:
            return 0.0
        return downside_returns.std()
    
    def _calculate_var(self, returns: pd.Series, confidence: float) -> float:
        """Calculate Value at Risk."""
        if len(returns) == 0:
            return 0.0
        return np.percentile(returns, (1 - confidence) * 100)
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0:
            return 0.0
        
        volatility = returns.std()
        if volatility == 0 or volatility < 1e-10:  # More robust zero check
            return 0.0
        
        # Convert annual risk-free rate to period rate
        rf_period = self.risk_free_rate / 252  # Daily risk-free rate
        excess_returns = returns - rf_period
        
        return excess_returns.mean() / volatility * np.sqrt(252)
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio."""
        if len(returns) == 0:
            return 0.0
        
        rf_period = self.risk_free_rate / 252
        excess_returns = returns - rf_period
        downside_std = self._calculate_downside_deviation(returns)
        
        if downside_std == 0:
            return 0.0
        
        return excess_returns.mean() / downside_std * np.sqrt(252)
    
    def _calculate_calmar_ratio(self, returns: pd.Series, max_drawdown: float) -> float:
        """Calculate Calmar ratio."""
        if max_drawdown == 0:
            return 0.0
        
        annual_return = self._annualize_return(returns)
        return annual_return / abs(max_drawdown)
    
    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0.0) -> float:
        """Calculate Omega ratio."""
        if len(returns) == 0:
            return 0.0
        
        gains = returns[returns > threshold].sum()
        losses = abs(returns[returns <= threshold].sum())
        
        if losses == 0:
            return float('inf') if gains > 0 else 0.0
        
        return gains / losses
    
    def _calculate_drawdown_metrics(self, pnl_series: pd.Series) -> Dict[str, Any]:
        """
        Calculate comprehensive drawdown metrics.
        
        Args:
            pnl_series: Cumulative P&L series
            
        Returns:
            Dictionary with drawdown metrics
        """
        # Calculate running maximum (peak)
        peak = pnl_series.expanding().max()
        
        # Calculate drawdown series
        drawdown = (pnl_series - peak) / peak.where(peak != 0, 1)
        
        # Maximum drawdown
        max_dd = drawdown.min()
        
        # Current drawdown
        current_dd = drawdown.iloc[-1]
        
        # Drawdown duration analysis
        drawdown_periods = 0
        max_dd_duration = 0
        current_dd_duration = 0
        
        in_drawdown = False
        dd_start = None
        
        for i, dd in enumerate(drawdown):
            if dd < -0.001:  # In drawdown (>0.1%)
                if not in_drawdown:
                    in_drawdown = True
                    dd_start = i
                    drawdown_periods += 1
                current_dd_duration = i - dd_start + 1
            else:  # Out of drawdown
                if in_drawdown:
                    max_dd_duration = max(max_dd_duration, current_dd_duration)
                    in_drawdown = False
                    current_dd_duration = 0
        
        # If still in drawdown at the end
        if in_drawdown:
            max_dd_duration = max(max_dd_duration, current_dd_duration)
        
        return {
            'max_drawdown': abs(max_dd),
            'current_drawdown': abs(current_dd),
            'max_drawdown_duration': max_dd_duration,
            'drawdown_periods': drawdown_periods
        }
    
    def _calculate_trade_metrics(self, trades_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate trading-specific metrics.
        
        Args:
            trades_data: List of trade dictionaries with PnL data
            
        Returns:
            Dictionary with trading metrics
        """
        if not trades_data:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'kelly_criterion': 0.0
            }
        
        # Extract PnL values
        pnls = [trade.get('pnl', 0) for trade in trades_data if 'pnl' in trade]
        
        if not pnls:
            return {
                'total_trades': len(trades_data),
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'kelly_criterion': 0.0
            }
        
        pnls = np.array(pnls)
        
        # Basic trade statistics
        total_trades = len(pnls)
        winning_trades = pnls[pnls > 0]
        losing_trades = pnls[pnls < 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
        avg_loss = abs(losing_trades.mean()) if len(losing_trades) > 0 else 0
        
        # Profit factor
        total_wins = winning_trades.sum() if len(winning_trades) > 0 else 0
        total_losses = abs(losing_trades.sum()) if len(losing_trades) > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Kelly criterion
        kelly = 0.0
        if avg_loss > 0 and win_rate > 0:
            # Kelly = (bp - q) / b, where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - win_rate
            kelly = (b * p - q) / b
            kelly = max(0, min(kelly, 0.25))  # Cap at 25% for safety
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'kelly_criterion': kelly
        }
    
    def _calculate_consistency_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Calculate consistency and distribution metrics.
        
        Args:
            returns: Return series
            
        Returns:
            Dictionary with consistency metrics
        """
        if len(returns) == 0:
            return {
                'hit_ratio': 0.0,
                'consistency_score': 0.0,
                'tail_ratio': 0.0
            }
        
        # Hit ratio (percentage of positive returns)
        hit_ratio = (returns > 0).sum() / len(returns)
        
        # Consistency score (1 - coefficient of variation)
        if returns.mean() != 0:
            cv = abs(returns.std() / returns.mean())
            consistency_score = max(0, 1 - cv)
        else:
            consistency_score = 0.0
        
        # Tail ratio (95th percentile / 5th percentile)
        p95 = returns.quantile(0.95)
        p5 = returns.quantile(0.05)
        
        if p5 < 0:
            tail_ratio = abs(p95 / p5)
        else:
            tail_ratio = 1.0
        
        return {
            'hit_ratio': hit_ratio,
            'consistency_score': consistency_score,
            'tail_ratio': tail_ratio
        }
    
    def compare_strategies(
        self,
        strategy_metrics: Dict[str, PerformanceMetrics]
    ) -> Dict[str, Any]:
        """
        Compare multiple strategies across key metrics.
        
        Args:
            strategy_metrics: Dictionary mapping strategy names to their metrics
            
        Returns:
            Dictionary with comparison results
        """
        if not strategy_metrics:
            return {}
        
        comparison = {}
        
        # Key metrics to compare
        key_metrics = [
            'annual_return', 'sharpe_ratio', 'max_drawdown', 
            'win_rate', 'profit_factor', 'total_trades'
        ]
        
        for metric in key_metrics:
            values = {}
            for name, metrics in strategy_metrics.items():
                if hasattr(metrics, metric):
                    values[name] = getattr(metrics, metric)
            
            if values:
                # Rank strategies for this metric
                sorted_strategies = sorted(
                    values.items(), 
                    key=lambda x: x[1], 
                    reverse=(metric != 'max_drawdown')  # Lower is better for drawdown
                )
                
                comparison[metric] = {
                    'ranking': [name for name, _ in sorted_strategies],
                    'values': values,
                    'best': sorted_strategies[0][0] if sorted_strategies else None,
                    'worst': sorted_strategies[-1][0] if sorted_strategies else None
                }
        
        # Overall score (composite ranking)
        overall_scores = {}
        for name in strategy_metrics.keys():
            score = 0
            for metric in key_metrics:
                if metric in comparison:
                    ranking = comparison[metric]['ranking']
                    if name in ranking:
                        # Score based on rank (higher rank = higher score)
                        rank_score = (len(ranking) - ranking.index(name)) / len(ranking)
                        score += rank_score
            
            overall_scores[name] = score / len(key_metrics) if key_metrics else 0
        
        # Sort by overall score
        sorted_overall = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
        
        comparison['overall'] = {
            'ranking': [name for name, _ in sorted_overall],
            'scores': overall_scores,
            'best': sorted_overall[0][0] if sorted_overall else None
        }
        
        return comparison
    
    def generate_performance_report(
        self,
        metrics: PerformanceMetrics,
        strategy_name: str = "Strategy"
    ) -> str:
        """
        Generate a formatted performance report.
        
        Args:
            metrics: Performance metrics
            strategy_name: Name of the strategy
            
        Returns:
            Formatted text report
        """
        report = f"""
=== {strategy_name} Performance Report ===

RETURN METRICS:
  Total Return:        {metrics.total_return:>8.2%}
  Annual Return:       {metrics.annual_return:>8.2%}
  
RISK METRICS:
  Annual Volatility:   {metrics.annual_volatility:>8.2%}
  Downside Deviation:  {metrics.downside_deviation:>8.2%}
  Value at Risk (95%): {metrics.value_at_risk_95:>8.2%}

RISK-ADJUSTED METRICS:
  Sharpe Ratio:        {metrics.sharpe_ratio:>8.2f}
  Sortino Ratio:       {metrics.sortino_ratio:>8.2f}
  Calmar Ratio:        {metrics.calmar_ratio:>8.2f}

DRAWDOWN METRICS:
  Max Drawdown:        {metrics.max_drawdown:>8.2%}
  Max DD Duration:     {metrics.max_drawdown_duration:>8d} periods
  Current Drawdown:    {metrics.current_drawdown:>8.2%}

TRADING METRICS:
  Total Trades:        {metrics.total_trades:>8d}
  Win Rate:            {metrics.win_rate:>8.2%}
  Average Win:         ${metrics.avg_win:.2f}
  Average Loss:        ${metrics.avg_loss:.2f}
  Profit Factor:       {metrics.profit_factor:>8.2f}

CONSISTENCY:
  Hit Ratio:           {metrics.hit_ratio:>8.2%}
  Consistency Score:   {metrics.consistency_score:>8.2f}
  Tail Ratio:          {metrics.tail_ratio:>8.2f}

PERIOD: {metrics.start_date or 'N/A'} to {metrics.end_date or 'N/A'}
DAYS TRADED: {metrics.days_traded}
        """.strip()
        
        return report
