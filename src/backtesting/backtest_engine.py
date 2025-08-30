"""
Backtesting Engine for Kalshi Trading System
Allows testing strategies with historical data and parameter optimization
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class TradeSide(Enum):
    YES = "yes"
    NO = "no"


@dataclass
class BacktestTrade:
    """Represents a trade in backtesting"""
    timestamp: datetime
    market_ticker: str
    side: TradeSide
    entry_price: float
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    quantity: int = 100
    confidence: float = 0.5
    kelly_fraction: float = 0.05
    pnl: float = 0.0
    pnl_pct: float = 0.0
    trigger_reason: str = ""
    exit_reason: str = ""
    
    def calculate_pnl(self):
        """Calculate P&L for the trade"""
        if self.exit_price is not None:
            if self.side == TradeSide.YES:
                self.pnl = (self.exit_price - self.entry_price) * self.quantity
            else:
                self.pnl = (self.entry_price - self.exit_price) * self.quantity
            
            self.pnl_pct = self.pnl / (self.entry_price * self.quantity)


@dataclass
class StrategyParameters:
    """Adjustable strategy parameters for backtesting"""
    # Position sizing
    kelly_multiplier: float = 0.25  # Fraction of Kelly to use
    max_position_pct: float = 0.05  # Max 5% per position
    min_position_size: int = 10
    max_position_size: int = 100
    
    # Entry triggers
    price_spike_threshold: float = 0.05  # 5% price change
    volume_surge_multiplier: float = 3.0  # 3x average volume
    sentiment_shift_threshold: float = 0.3  # 30% sentiment change
    arbitrage_min_profit: float = 0.02  # 2% minimum arbitrage
    
    # Exit rules
    stop_loss_pct: float = 0.10  # 10% stop loss
    take_profit_pct: float = 0.30  # 30% take profit
    time_based_exit_hours: int = 24  # Exit after 24 hours
    
    # Risk management
    max_daily_loss_pct: float = 0.20  # 20% daily loss limit
    max_correlation: float = 0.7  # Max correlation between positions
    max_concurrent_positions: int = 5
    
    # Market filters
    min_liquidity: float = 1000  # Minimum volume
    max_spread: float = 0.10  # Maximum bid-ask spread
    min_confidence: float = 0.60  # Minimum confidence to trade
    
    # Time filters
    avoid_first_minutes: int = 5  # Avoid first 5 min after game start
    avoid_last_minutes: int = 5  # Avoid last 5 min before game end
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary"""
        return {
            'kelly_multiplier': self.kelly_multiplier,
            'max_position_pct': self.max_position_pct,
            'min_position_size': self.min_position_size,
            'max_position_size': self.max_position_size,
            'price_spike_threshold': self.price_spike_threshold,
            'volume_surge_multiplier': self.volume_surge_multiplier,
            'sentiment_shift_threshold': self.sentiment_shift_threshold,
            'arbitrage_min_profit': self.arbitrage_min_profit,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'time_based_exit_hours': self.time_based_exit_hours,
            'max_daily_loss_pct': self.max_daily_loss_pct,
            'max_correlation': self.max_correlation,
            'max_concurrent_positions': self.max_concurrent_positions,
            'min_liquidity': self.min_liquidity,
            'max_spread': self.max_spread,
            'min_confidence': self.min_confidence,
            'avoid_first_minutes': self.avoid_first_minutes,
            'avoid_last_minutes': self.avoid_last_minutes
        }


@dataclass
class BacktestResults:
    """Results from a backtest run"""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    trades: List[BacktestTrade] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_metrics(self):
        """Calculate performance metrics"""
        if not self.trades:
            return
        
        # Win rate
        self.winning_trades = sum(1 for t in self.trades if t.pnl > 0)
        self.losing_trades = sum(1 for t in self.trades if t.pnl < 0)
        self.win_rate = self.winning_trades / max(self.total_trades, 1)
        
        # Average win/loss
        wins = [t.pnl for t in self.trades if t.pnl > 0]
        losses = [t.pnl for t in self.trades if t.pnl < 0]
        
        self.avg_win = np.mean(wins) if wins else 0
        self.avg_loss = np.mean(losses) if losses else 0
        
        # Profit factor
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 1
        self.profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Sharpe ratio (assuming 0% risk-free rate)
        if self.daily_returns:
            returns_array = np.array(self.daily_returns)
            if len(returns_array) > 1:
                self.sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)
            else:
                self.sharpe_ratio = 0
        
        # Max drawdown
        if self.equity_curve:
            peak = self.equity_curve[0]
            max_dd = 0
            for value in self.equity_curve:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak
                if dd > max_dd:
                    max_dd = dd
            self.max_drawdown = max_dd


class BacktestEngine:
    """
    Main backtesting engine for strategy evaluation
    """
    
    def __init__(
        self,
        initial_capital: float = 10000,
        data_path: str = "backtesting/historical_data"
    ):
        self.initial_capital = initial_capital
        self.data_path = Path(data_path)
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.current_positions: Dict[str, BacktestTrade] = {}
        self.completed_trades: List[BacktestTrade] = []
        self.equity_curve: List[float] = []
        self.daily_returns: List[float] = []
        
    async def load_historical_data(
        self,
        start_date: datetime,
        end_date: datetime,
        markets: List[str]
    ):
        """
        Load historical data for backtesting
        
        Args:
            start_date: Start of backtest period
            end_date: End of backtest period
            markets: List of market tickers to load
        """
        for market in markets:
            file_path = self.data_path / f"{market}_historical.csv"
            
            if file_path.exists():
                df = pd.read_csv(file_path, parse_dates=['timestamp'])
                # Filter by date range
                df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
                self.historical_data[market] = df
                logger.info(f"Loaded {len(df)} records for {market}")
            else:
                logger.warning(f"No historical data found for {market}")
    
    async def run_backtest(
        self,
        strategy_params: StrategyParameters,
        start_date: datetime,
        end_date: datetime,
        markets: List[str]
    ) -> BacktestResults:
        """
        Run backtest with given parameters
        
        Args:
            strategy_params: Strategy parameters to test
            start_date: Start of backtest period
            end_date: End of backtest period
            markets: Markets to trade
            
        Returns:
            BacktestResults with performance metrics
        """
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Reset state
        self.current_positions = {}
        self.completed_trades = []
        self.equity_curve = [self.initial_capital]
        self.daily_returns = []
        
        # Load data if not already loaded
        if not self.historical_data:
            await self.load_historical_data(start_date, end_date, markets)
        
        # Initialize portfolio
        current_capital = self.initial_capital
        current_date = start_date
        
        # Main backtest loop
        while current_date <= end_date:
            daily_pnl = 0
            
            # Process each market
            for market in markets:
                if market not in self.historical_data:
                    continue
                
                df = self.historical_data[market]
                
                # Get data for current timestamp
                current_data = df[
                    (df['timestamp'] >= current_date) & 
                    (df['timestamp'] < current_date + timedelta(days=1))
                ]
                
                if current_data.empty:
                    continue
                
                for _, row in current_data.iterrows():
                    # Check exit conditions for open positions
                    if market in self.current_positions:
                        position = self.current_positions[market]
                        should_exit, exit_reason = self._check_exit_conditions(
                            position, row, strategy_params
                        )
                        
                        if should_exit:
                            # Close position
                            position.exit_price = row['yes_price']
                            position.exit_timestamp = row['timestamp']
                            position.exit_reason = exit_reason
                            position.calculate_pnl()
                            
                            daily_pnl += position.pnl
                            current_capital += position.pnl
                            
                            self.completed_trades.append(position)
                            del self.current_positions[market]
                            
                            logger.info(
                                f"Closed {market}: {exit_reason}, "
                                f"P&L: ${position.pnl:.2f} ({position.pnl_pct:.1%})"
                            )
                    
                    # Check entry conditions
                    elif len(self.current_positions) < strategy_params.max_concurrent_positions:
                        should_enter, entry_reason, confidence = self._check_entry_conditions(
                            row, strategy_params
                        )
                        
                        if should_enter and confidence >= strategy_params.min_confidence:
                            # Calculate position size
                            position_size = self._calculate_position_size(
                                current_capital, confidence, strategy_params
                            )
                            
                            # Open position
                            side = self._determine_side(row, entry_reason)
                            
                            position = BacktestTrade(
                                timestamp=row['timestamp'],
                                market_ticker=market,
                                side=side,
                                entry_price=row['yes_price'] if side == TradeSide.YES else row['no_price'],
                                quantity=position_size,
                                confidence=confidence,
                                trigger_reason=entry_reason
                            )
                            
                            self.current_positions[market] = position
                            current_capital -= position.entry_price * position.quantity
                            
                            logger.info(
                                f"Opened {market}: {side.value} @ {position.entry_price:.2f}, "
                                f"Size: {position_size}, Reason: {entry_reason}"
                            )
            
            # Update equity curve
            total_value = current_capital + sum(
                p.quantity * p.entry_price for p in self.current_positions.values()
            )
            self.equity_curve.append(total_value)
            
            # Calculate daily return
            if len(self.equity_curve) > 1:
                daily_return = (self.equity_curve[-1] - self.equity_curve[-2]) / self.equity_curve[-2]
                self.daily_returns.append(daily_return)
            
            # Check daily loss limit
            daily_loss = (total_value - self.initial_capital) / self.initial_capital
            if daily_loss < -strategy_params.max_daily_loss_pct:
                logger.warning(f"Daily loss limit reached: {daily_loss:.1%}")
                # Close all positions
                for market, position in list(self.current_positions.items()):
                    position.exit_price = position.entry_price * 0.95  # Assume 5% slippage
                    position.exit_timestamp = current_date
                    position.exit_reason = "Daily loss limit"
                    position.calculate_pnl()
                    self.completed_trades.append(position)
                    del self.current_positions[market]
            
            # Move to next day
            current_date += timedelta(days=1)
        
        # Close any remaining positions
        for market, position in self.current_positions.items():
            position.exit_price = position.entry_price  # Exit at current price
            position.exit_timestamp = end_date
            position.exit_reason = "Backtest end"
            position.calculate_pnl()
            self.completed_trades.append(position)
        
        # Calculate final metrics
        final_capital = current_capital + sum(
            t.pnl for t in self.completed_trades
        )
        
        results = BacktestResults(
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_trades=len(self.completed_trades),
            winning_trades=0,
            losing_trades=0,
            total_pnl=final_capital - self.initial_capital,
            total_return=(final_capital - self.initial_capital) / self.initial_capital,
            sharpe_ratio=0,
            max_drawdown=0,
            win_rate=0,
            avg_win=0,
            avg_loss=0,
            profit_factor=0,
            trades=self.completed_trades,
            daily_returns=self.daily_returns,
            equity_curve=self.equity_curve,
            parameters=strategy_params.to_dict()
        )
        
        results.calculate_metrics()
        
        return results
    
    def _check_entry_conditions(
        self,
        data: pd.Series,
        params: StrategyParameters
    ) -> Tuple[bool, str, float]:
        """
        Check if entry conditions are met
        
        Returns:
            (should_enter, reason, confidence)
        """
        # Check liquidity
        if data.get('volume', 0) < params.min_liquidity:
            return False, "", 0.0
        
        # Check spread
        spread = abs(data.get('yes_price', 0) + data.get('no_price', 0) - 1.0)
        if spread > params.max_spread:
            return False, "", 0.0
        
        # Price spike detection
        if 'price_change' in data and abs(data['price_change']) > params.price_spike_threshold:
            confidence = min(0.5 + abs(data['price_change']), 0.9)
            return True, "price_spike", confidence
        
        # Volume surge detection
        if 'volume_ratio' in data and data['volume_ratio'] > params.volume_surge_multiplier:
            confidence = min(0.5 + data['volume_ratio'] / 10, 0.85)
            return True, "volume_surge", confidence
        
        # Arbitrage opportunity
        yes_price = data.get('yes_price', 0)
        no_price = data.get('no_price', 0)
        if yes_price > 0 and no_price > 0:
            total = yes_price + no_price
            if total < (1.0 - params.arbitrage_min_profit):
                confidence = min(0.7 + (1.0 - total), 0.95)
                return True, "arbitrage", confidence
        
        # Sentiment shift
        if 'sentiment_change' in data and abs(data['sentiment_change']) > params.sentiment_shift_threshold:
            confidence = min(0.6 + abs(data['sentiment_change']), 0.85)
            return True, "sentiment_shift", confidence
        
        return False, "", 0.0
    
    def _check_exit_conditions(
        self,
        position: BacktestTrade,
        data: pd.Series,
        params: StrategyParameters
    ) -> Tuple[bool, str]:
        """
        Check if exit conditions are met
        
        Returns:
            (should_exit, reason)
        """
        current_price = data['yes_price'] if position.side == TradeSide.YES else data['no_price']
        
        # Calculate current P&L
        if position.side == TradeSide.YES:
            pnl_pct = (current_price - position.entry_price) / position.entry_price
        else:
            pnl_pct = (position.entry_price - current_price) / position.entry_price
        
        # Stop loss
        if pnl_pct <= -params.stop_loss_pct:
            return True, f"Stop loss ({pnl_pct:.1%})"
        
        # Take profit
        if pnl_pct >= params.take_profit_pct:
            return True, f"Take profit ({pnl_pct:.1%})"
        
        # Time-based exit
        time_held = data['timestamp'] - position.timestamp
        if time_held.total_seconds() / 3600 > params.time_based_exit_hours:
            return True, f"Time exit ({time_held.total_seconds()/3600:.1f} hours)"
        
        # Game ended (if we have that data)
        if data.get('game_ended', False):
            return True, "Game ended"
        
        return False, ""
    
    def _determine_side(self, data: pd.Series, entry_reason: str) -> TradeSide:
        """Determine which side to trade"""
        if entry_reason == "arbitrage":
            # For arbitrage, buy both but we'll simplify to YES
            return TradeSide.YES
        
        # For other signals, trade in direction of signal
        if 'sentiment_change' in data and data['sentiment_change'] > 0:
            return TradeSide.YES
        elif 'price_change' in data and data['price_change'] > 0:
            return TradeSide.YES
        
        return TradeSide.NO
    
    def _calculate_position_size(
        self,
        capital: float,
        confidence: float,
        params: StrategyParameters
    ) -> int:
        """Calculate position size using Kelly Criterion"""
        # Kelly fraction
        kelly = confidence - (1 - confidence)
        kelly = max(0, kelly)  # Ensure non-negative
        
        # Apply multiplier
        adjusted_kelly = kelly * params.kelly_multiplier
        
        # Calculate dollar amount
        position_value = capital * adjusted_kelly
        
        # Apply position limits
        max_position_value = capital * params.max_position_pct
        position_value = min(position_value, max_position_value)
        
        # Convert to quantity (assuming $1 per contract)
        quantity = int(position_value)
        
        # Apply min/max limits
        quantity = max(params.min_position_size, quantity)
        quantity = min(params.max_position_size, quantity)
        
        return quantity


# Example usage
async def main():
    """Example backtest run"""
    
    # Initialize engine
    engine = BacktestEngine(initial_capital=10000)
    
    # Define strategy parameters
    params = StrategyParameters(
        kelly_multiplier=0.25,
        stop_loss_pct=0.10,
        take_profit_pct=0.30,
        price_spike_threshold=0.05,
        min_confidence=0.65
    )
    
    # Run backtest
    results = await engine.run_backtest(
        strategy_params=params,
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31),
        markets=["NFL-CHIEFS-WIN", "NFL-BILLS-WIN"]
    )
    
    # Display results
    print(f"Total Return: {results.total_return:.1%}")
    print(f"Win Rate: {results.win_rate:.1%}")
    print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {results.max_drawdown:.1%}")
    print(f"Profit Factor: {results.profit_factor:.2f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())