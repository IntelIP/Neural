"""
Event-Driven Backtesting Engine

This module implements a comprehensive backtesting engine that simulates
realistic trading conditions for strategy validation and optimization.

Key Features:
- Event-driven architecture for realistic simulation
- Integration with strategy framework
- Realistic fill simulation and slippage modeling
- Comprehensive performance analytics
- Multiple asset support
- Risk management integration

The engine processes historical market data chronologically, allowing
strategies to react to market events in real-time simulation.
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from pathlib import Path

from neural.strategy.base import BaseStrategy, Signal
from neural.strategy.builder import StrategyComposer
from neural.analysis.market_data import MarketDataStore
from neural.analysis.metrics import PerformanceCalculator, PerformanceMetrics
from neural.kalshi.fees import calculate_kalshi_fee

logger = logging.getLogger(__name__)


class BacktestState(Enum):
    """Backtest execution states."""
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BacktestConfig:
    """Configuration for backtest execution."""
    initial_capital: float = 10000.0
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    commission_rate: float = 0.0  # Additional commission beyond Kalshi fees
    slippage_model: str = "linear"  # linear, impact, fixed
    max_slippage_bps: int = 50  # 50 basis points max slippage
    risk_free_rate: float = 0.02  # 2% risk-free rate
    benchmark_return: float = 0.0  # Benchmark for comparison
    rebalance_frequency: str = "daily"  # daily, hourly, trade
    position_sizing_method: str = "signal_based"  # signal_based, equal_weight, kelly
    max_positions: int = 20  # Maximum concurrent positions
    max_position_size: float = 0.20  # Max 20% of capital per position
    stop_loss_pct: Optional[float] = None  # Global stop loss
    take_profit_pct: Optional[float] = None  # Global take profit
    warmup_periods: int = 0  # Periods to warm up before trading
    enable_shorting: bool = False  # Enable short positions (buy NO)
    margin_requirement: float = 0.0  # Margin requirement for positions
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class Trade:
    """Record of a completed trade."""
    trade_id: str
    strategy_id: str
    market_id: str
    side: str  # YES or NO
    entry_price: float
    exit_price: float
    quantity: int
    entry_time: datetime
    exit_time: datetime
    duration: timedelta
    pnl: float
    fees: float
    slippage: float
    commission: float
    net_pnl: float
    edge_estimate: float
    confidence: float
    signal_strength: str
    exit_reason: str  # profit, loss, time, strategy
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Current market position."""
    position_id: str
    strategy_id: str
    market_id: str
    side: str  # YES or NO
    quantity: int
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float
    entry_time: datetime
    last_update: datetime
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    max_hold_time: Optional[timedelta] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def market_value(self) -> float:
        """Current market value of position."""
        return self.quantity * self.current_price

    @property 
    def cost_basis(self) -> float:
        """Cost basis of position."""
        return self.quantity * self.avg_entry_price

    def update_price(self, new_price: float, timestamp: datetime) -> None:
        """Update position with new market price."""
        self.current_price = new_price
        self.last_update = timestamp
        
        # Calculate unrealized P&L
        if self.side == "YES":
            price_change = new_price - self.avg_entry_price
        else:  # NO position
            price_change = self.avg_entry_price - new_price
            
        self.unrealized_pnl = self.quantity * price_change


@dataclass
class BacktestResult:
    """Results of a completed backtest."""
    config: BacktestConfig
    start_date: datetime
    end_date: datetime  
    duration: timedelta
    initial_capital: float
    final_capital: float
    total_return: float
    annual_return: float
    max_drawdown: float
    sharpe_ratio: float
    total_trades: int
    win_rate: float
    profit_factor: float
    performance_metrics: PerformanceMetrics
    equity_curve: pd.Series
    trades: List[Trade]
    daily_returns: pd.Series
    strategy_breakdown: Dict[str, Any]
    risk_metrics: Dict[str, Any]
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BacktestEngine:
    """
    Event-driven backtesting engine for strategy validation.
    
    This engine simulates realistic trading conditions by processing
    historical market data chronologically and allowing strategies
    to react to market events in real-time.
    """
    
    def __init__(self, config: BacktestConfig = None):
        """
        Initialize backtesting engine.
        
        Args:
            config: Backtest configuration
        """
        self.config = config or BacktestConfig()
        self.state = BacktestState.INITIALIZED
        
        # Portfolio tracking
        self.current_capital = self.config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[Trade] = []
        
        # Performance tracking
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.daily_returns: List[Tuple[datetime, float]] = []
        
        # Strategy management
        self.strategies: List[Union[BaseStrategy, StrategyComposer]] = []
        self.strategy_performance: Dict[str, Dict[str, float]] = {}
        
        # Market data
        self.market_data_store: Optional[MarketDataStore] = None
        
        # Event tracking
        self.current_time: Optional[datetime] = None
        self.events_processed: int = 0
        
        # Performance calculator
        self.performance_calc = PerformanceCalculator(
            risk_free_rate=self.config.risk_free_rate
        )
        
        logger.info(f"Initialized BacktestEngine with ${self.config.initial_capital:,.2f} capital")
    
    def add_strategy(self, strategy: Union[BaseStrategy, StrategyComposer]) -> None:
        """
        Add a strategy to the backtest.
        
        Args:
            strategy: Strategy or strategy composer to add
        """
        self.strategies.append(strategy)
        
        strategy_name = getattr(strategy, 'name', strategy.__class__.__name__)
        self.strategy_performance[strategy_name] = {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0.0,
            'total_fees': 0.0
        }
        
        logger.info(f"Added strategy: {strategy_name}")
    
    def set_data_source(self, data_source: Union[MarketDataStore, pd.DataFrame, Dict[str, pd.DataFrame]]) -> None:
        """
        Set the historical data source for backtesting.
        
        Args:
            data_source: Historical market data source
        """
        if isinstance(data_source, MarketDataStore):
            self.market_data_store = data_source
        else:
            # Handle DataFrame or dictionary of DataFrames
            # For now, we'll focus on MarketDataStore integration
            raise NotImplementedError("DataFrame data sources not yet implemented")
        
        logger.info("Data source configured")
    
    async def run(
        self, 
        market_ids: Optional[List[str]] = None,
        progress_callback: Optional[callable] = None
    ) -> BacktestResult:
        """
        Run the backtest simulation.
        
        Args:
            market_ids: Specific markets to backtest (None for all)
            progress_callback: Optional callback for progress updates
            
        Returns:
            BacktestResult with comprehensive performance metrics
        """
        if not self.strategies:
            raise ValueError("No strategies added to backtest")
        
        if not self.market_data_store:
            raise ValueError("No data source configured")
        
        logger.info("🚀 Starting backtest simulation")
        start_time = datetime.now()
        self.state = BacktestState.RUNNING
        
        try:
            # Initialize backtest
            await self._initialize_backtest()
            
            # Get historical data
            historical_data = await self._load_historical_data(market_ids)
            
            # Process events chronologically
            await self._process_events(historical_data, progress_callback)
            
            # Finalize backtest
            result = await self._finalize_backtest(start_time)
            
            self.state = BacktestState.COMPLETED
            logger.info(f"✅ Backtest completed in {result.execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.state = BacktestState.FAILED
            logger.error(f"❌ Backtest failed: {e}")
            raise
    
    async def _initialize_backtest(self) -> None:
        """Initialize backtest state and tracking."""
        self.current_capital = self.config.initial_capital
        self.positions = {}
        self.closed_trades = []
        self.equity_curve = [(datetime.now(), self.current_capital)]
        self.daily_returns = []
        self.events_processed = 0
        
        # Initialize strategy performance tracking
        for strategy in self.strategies:
            strategy_name = getattr(strategy, 'name', strategy.__class__.__name__)
            self.strategy_performance[strategy_name] = {
                'trades': 0, 'wins': 0, 'losses': 0, 
                'total_pnl': 0.0, 'total_fees': 0.0
            }
    
    async def _load_historical_data(self, market_ids: Optional[List[str]]) -> pd.DataFrame:
        """
        Load and prepare historical market data.
        
        Args:
            market_ids: Markets to load data for
            
        Returns:
            DataFrame with chronologically ordered market events
        """
        if not market_ids:
            # Get all available markets in date range
            market_ids = await self._get_available_markets()
        
        all_data = []
        
        for market_id in market_ids:
            # Get price history for market
            market_data = self.market_data_store.get_price_history(
                market_id=market_id,
                start_time=int(self.config.start_date.timestamp()) if self.config.start_date else None,
                end_time=int(self.config.end_date.timestamp()) if self.config.end_date else None
            )
            
            if not market_data.empty:
                market_data['market_id'] = market_id
                all_data.append(market_data)
        
        if not all_data:
            raise ValueError("No historical data found for specified markets and date range")
        
        # Combine and sort chronologically
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data = combined_data.sort_index()  # Sort by timestamp index
        
        logger.info(f"Loaded {len(combined_data)} price events across {len(market_ids)} markets")
        return combined_data
    
    async def _get_available_markets(self) -> List[str]:
        """Get list of available markets from data store."""
        # This is a simplified implementation
        # In practice, would query the database for available markets
        return ["EXAMPLE_MARKET"]  # Placeholder
    
    async def _process_events(
        self, 
        historical_data: pd.DataFrame, 
        progress_callback: Optional[callable] = None
    ) -> None:
        """
        Process historical events chronologically.
        
        Args:
            historical_data: Historical market data
            progress_callback: Optional progress callback
        """
        total_events = len(historical_data)
        
        for i, (timestamp, row) in enumerate(historical_data.iterrows()):
            self.current_time = timestamp
            self.events_processed = i + 1
            
            # Update position valuations
            await self._update_positions(row)
            
            # Generate strategy signals
            signals = await self._generate_signals(row)
            
            # Process signals and execute trades
            await self._process_signals(signals)
            
            # Check for position exits (stop loss, take profit, time decay)
            await self._check_position_exits(row)
            
            # Update equity curve
            self._update_equity_curve(timestamp)
            
            # Progress callback
            if progress_callback and i % 100 == 0:  # Every 100 events
                progress_callback(i, total_events)
        
        logger.info(f"Processed {self.events_processed} market events")
    
    async def _update_positions(self, market_event: pd.Series) -> None:
        """Update existing positions with new market data."""
        market_id = market_event['market_id']
        current_price = market_event.get('last', market_event.get('bid', 0))
        
        # Update positions for this market
        for position in self.positions.values():
            if position.market_id == market_id:
                position.update_price(current_price, self.current_time)
    
    async def _generate_signals(self, market_event: pd.Series) -> List[Signal]:
        """Generate signals from all strategies for current market event."""
        signals = []
        
        # Prepare market data for strategies
        market_data = {
            'current_price': market_event.get('last', market_event.get('bid', 0)),
            'bid': market_event.get('bid'),
            'ask': market_event.get('ask'),
            'volume': market_event.get('volume', 0),
            'timestamp': self.current_time
        }
        
        # Get signals from each strategy
        for strategy in self.strategies:
            try:
                signal = await strategy.analyze(market_event['market_id'], market_data)
                if signal and signal.action != 'HOLD':
                    signals.append(signal)
            except Exception as e:
                logger.warning(f"Strategy {strategy.__class__.__name__} error: {e}")
        
        return signals
    
    async def _process_signals(self, signals: List[Signal]) -> None:
        """Process trading signals and execute trades."""
        for signal in signals:
            await self._execute_signal(signal)
    
    async def _execute_signal(self, signal: Signal) -> None:
        """
        Execute a trading signal.
        
        Args:
            signal: Trading signal to execute
        """
        # Check if we already have a position in this market
        existing_position = None
        for pos in self.positions.values():
            if pos.market_id == signal.market_id and pos.strategy_id == signal.strategy_id:
                existing_position = pos
                break
        
        # Determine action (open new position or modify existing)
        if existing_position is None:
            await self._open_position(signal)
        else:
            await self._modify_position(existing_position, signal)
    
    async def _open_position(self, signal: Signal) -> None:
        """Open a new position based on signal."""
        # Calculate position size
        position_size = self._calculate_position_size(signal)
        
        if position_size <= 0:
            return
        
        # Simulate market impact and slippage
        fill_price = self._simulate_fill_price(signal, position_size)
        
        # Calculate fees
        fees = calculate_kalshi_fee(fill_price, position_size)
        
        # Check if we have enough capital
        total_cost = position_size * fill_price + fees
        if total_cost > self.current_capital:
            logger.debug(f"Insufficient capital for {signal.market_id}: need ${total_cost:.2f}, have ${self.current_capital:.2f}")
            return
        
        # Create position
        position_id = f"{signal.strategy_id}_{signal.market_id}_{self.current_time.isoformat()}"
        
        position = Position(
            position_id=position_id,
            strategy_id=signal.strategy_id,
            market_id=signal.market_id,
            side=signal.action.replace('BUY_', ''),  # YES or NO
            quantity=position_size,
            avg_entry_price=fill_price,
            current_price=fill_price,
            unrealized_pnl=0.0,
            entry_time=self.current_time,
            last_update=self.current_time,
            stop_loss_price=getattr(signal, 'stop_loss_price', None),
            take_profit_price=getattr(signal, 'take_profit_price', None),
            max_hold_time=getattr(signal, 'max_hold_time', None),
            metadata={
                'edge_estimate': getattr(signal, 'edge', 0),
                'confidence': signal.confidence,
                'signal_strength': signal.signal_strength.value
            }
        )
        
        self.positions[position_id] = position
        
        # Update capital
        self.current_capital -= total_cost
        
        logger.debug(f"Opened {signal.action} position: {signal.market_id} @ ${fill_price:.3f} x{position_size}")
    
    async def _modify_position(self, position: Position, signal: Signal) -> None:
        """Modify existing position based on new signal."""
        # For now, we'll keep it simple and not modify positions
        # In a more sophisticated system, this would handle position sizing changes
        pass
    
    async def _check_position_exits(self, market_event: pd.Series) -> None:
        """Check for position exit conditions."""
        positions_to_close = []
        
        for position in self.positions.values():
            if position.market_id != market_event['market_id']:
                continue
            
            exit_reason = None
            
            # Check stop loss
            if position.stop_loss_price and position.current_price <= position.stop_loss_price:
                exit_reason = "stop_loss"
            
            # Check take profit
            elif position.take_profit_price and position.current_price >= position.take_profit_price:
                exit_reason = "take_profit"
            
            # Check time limit
            elif position.max_hold_time:
                if self.current_time - position.entry_time >= position.max_hold_time:
                    exit_reason = "time_limit"
            
            if exit_reason:
                positions_to_close.append((position, exit_reason))
        
        # Close positions
        for position, exit_reason in positions_to_close:
            await self._close_position(position, exit_reason)
    
    async def _close_position(self, position: Position, exit_reason: str) -> None:
        """Close a position and record the trade."""
        # Simulate fill price
        exit_price = position.current_price
        
        # Calculate P&L
        if position.side == "YES":
            price_diff = exit_price - position.avg_entry_price
        else:  # NO position
            price_diff = position.avg_entry_price - exit_price
        
        gross_pnl = position.quantity * price_diff
        
        # Calculate fees (entry + exit)
        entry_fees = calculate_kalshi_fee(position.avg_entry_price, position.quantity)
        exit_fees = calculate_kalshi_fee(exit_price, position.quantity)
        total_fees = entry_fees + exit_fees
        
        # Net P&L
        net_pnl = gross_pnl - total_fees
        
        # Create trade record
        trade = Trade(
            trade_id=f"T_{len(self.closed_trades)+1}_{position.position_id}",
            strategy_id=position.strategy_id,
            market_id=position.market_id,
            side=position.side,
            entry_price=position.avg_entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            entry_time=position.entry_time,
            exit_time=self.current_time,
            duration=self.current_time - position.entry_time,
            pnl=gross_pnl,
            fees=total_fees,
            slippage=0.0,  # Simplified for now
            commission=0.0,
            net_pnl=net_pnl,
            edge_estimate=position.metadata.get('edge_estimate', 0),
            confidence=position.metadata.get('confidence', 0),
            signal_strength=position.metadata.get('signal_strength', ''),
            exit_reason=exit_reason,
            metadata=position.metadata.copy()
        )
        
        self.closed_trades.append(trade)
        
        # Update capital
        self.current_capital += net_pnl + (position.quantity * position.avg_entry_price)
        
        # Update strategy performance
        strategy_perf = self.strategy_performance.get(position.strategy_id, {})
        strategy_perf['trades'] = strategy_perf.get('trades', 0) + 1
        strategy_perf['total_pnl'] = strategy_perf.get('total_pnl', 0) + net_pnl
        strategy_perf['total_fees'] = strategy_perf.get('total_fees', 0) + total_fees
        
        if net_pnl > 0:
            strategy_perf['wins'] = strategy_perf.get('wins', 0) + 1
        else:
            strategy_perf['losses'] = strategy_perf.get('losses', 0) + 1
        
        # Remove position
        del self.positions[position.position_id]
        
        logger.debug(f"Closed position: {position.market_id} for ${net_pnl:.2f} ({exit_reason})")
    
    def _calculate_position_size(self, signal: Signal) -> int:
        """Calculate position size based on signal and risk management."""
        # Use signal's recommended size as base
        target_allocation = signal.position_size
        
        # Apply maximum position size limit
        target_allocation = min(target_allocation, self.config.max_position_size)
        
        # Calculate number of contracts
        market_price = getattr(signal, 'market_price', 0.50)  # Default to 50 cents
        max_contracts = int((self.current_capital * target_allocation) / market_price)
        
        # Apply signal's max contracts limit if specified
        if hasattr(signal, 'max_contracts') and signal.max_contracts:
            max_contracts = min(max_contracts, signal.max_contracts)
        
        return max(0, max_contracts)
    
    def _simulate_fill_price(self, signal: Signal, quantity: int) -> float:
        """Simulate realistic fill price with slippage."""
        # Get base price from signal
        base_price = getattr(signal, 'market_price', 0.50)
        
        # Simple linear slippage model
        if self.config.slippage_model == "linear":
            # Small amount of slippage based on position size
            slippage_factor = min(quantity * 0.0001, self.config.max_slippage_bps / 10000)
            
            if signal.action == 'BUY_YES':
                # Buying moves price up slightly
                fill_price = base_price * (1 + slippage_factor)
            else:  # BUY_NO
                # Buying NO is equivalent to selling YES, moves price down
                fill_price = base_price * (1 - slippage_factor)
        else:
            fill_price = base_price
        
        # Ensure price stays within valid bounds
        return max(0.01, min(fill_price, 0.99))
    
    def _update_equity_curve(self, timestamp: datetime) -> None:
        """Update equity curve with current portfolio value."""
        # Current capital + unrealized P&L from positions
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        position_values = sum(pos.market_value for pos in self.positions.values())
        
        total_equity = self.current_capital + unrealized_pnl
        
        self.equity_curve.append((timestamp, total_equity))
    
    async def _finalize_backtest(self, start_time: datetime) -> BacktestResult:
        """Finalize backtest and calculate comprehensive results."""
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Close any remaining open positions
        await self._close_remaining_positions()
        
        # Calculate performance metrics
        equity_series = pd.Series(
            data=[eq[1] for eq in self.equity_curve],
            index=[eq[0] for eq in self.equity_curve]
        )
        
        returns_series = equity_series.pct_change().dropna()
        
        # Calculate comprehensive metrics
        performance_metrics = self.performance_calc.calculate_comprehensive_metrics(
            returns=returns_series.values,
            initial_capital=self.config.initial_capital,
            final_capital=self.current_capital,
            trade_history=self.closed_trades
        )
        
        # Strategy breakdown
        strategy_breakdown = {}
        for strategy_name, perf in self.strategy_performance.items():
            if perf['trades'] > 0:
                strategy_breakdown[strategy_name] = {
                    **perf,
                    'win_rate': perf['wins'] / perf['trades'] if perf['trades'] > 0 else 0,
                    'avg_pnl_per_trade': perf['total_pnl'] / perf['trades'] if perf['trades'] > 0 else 0
                }
        
        return BacktestResult(
            config=self.config,
            start_date=self.config.start_date or self.equity_curve[0][0],
            end_date=self.config.end_date or self.equity_curve[-1][0],
            duration=self.equity_curve[-1][0] - self.equity_curve[0][0],
            initial_capital=self.config.initial_capital,
            final_capital=self.current_capital,
            total_return=(self.current_capital - self.config.initial_capital) / self.config.initial_capital,
            annual_return=performance_metrics.annual_return,
            max_drawdown=performance_metrics.max_drawdown,
            sharpe_ratio=performance_metrics.sharpe_ratio,
            total_trades=len(self.closed_trades),
            win_rate=performance_metrics.win_rate,
            profit_factor=getattr(performance_metrics, 'profit_factor', 0),
            performance_metrics=performance_metrics,
            equity_curve=equity_series,
            trades=self.closed_trades,
            daily_returns=returns_series,
            strategy_breakdown=strategy_breakdown,
            risk_metrics={
                'max_drawdown': performance_metrics.max_drawdown,
                'volatility': performance_metrics.volatility,
                'var_95': getattr(performance_metrics, 'var_95', 0),
                'sortino_ratio': getattr(performance_metrics, 'sortino_ratio', 0)
            },
            execution_time=execution_time
        )
    
    async def _close_remaining_positions(self) -> None:
        """Close any remaining open positions at the end of backtest."""
        for position in list(self.positions.values()):
            await self._close_position(position, "backtest_end")
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current backtest status and metrics."""
        return {
            'state': self.state.value,
            'events_processed': self.events_processed,
            'current_capital': self.current_capital,
            'open_positions': len(self.positions),
            'closed_trades': len(self.closed_trades),
            'current_time': self.current_time.isoformat() if self.current_time else None,
            'unrealized_pnl': sum(pos.unrealized_pnl for pos in self.positions.values())
        }
