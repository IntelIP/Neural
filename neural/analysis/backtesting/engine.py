"""
Backtesting Engine for Neural Analysis Stack

Provides comprehensive backtesting capabilities with support for:
- Historical market data from Kalshi
- ESPN play-by-play integration
- Multiple strategies running in parallel
- Detailed performance metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import asyncio
from concurrent.futures import ThreadPoolExecutor


@dataclass
class BacktestResult:
    """Results from a backtest run"""
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    profit_factor: float
    total_fees: float
    trades: List[Dict] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    daily_returns: pd.Series = field(default_factory=pd.Series)
    metrics: Dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"""
Backtest Results: {self.strategy_name}
{'='*50}
Period: {self.start_date.date()} to {self.end_date.date()}
Initial Capital: ${self.initial_capital:,.2f}
Final Capital: ${self.final_capital:,.2f}
Total Return: ${self.total_return:,.2f} ({self.total_return_pct:.2f}%)

Performance Metrics:
- Sharpe Ratio: {self.sharpe_ratio:.3f}
- Max Drawdown: ${self.max_drawdown:,.2f} ({self.max_drawdown_pct:.2f}%)
- Win Rate: {self.win_rate:.2%}
- Total Trades: {self.total_trades}
- Avg Win: ${self.avg_win:.2f}
- Avg Loss: ${self.avg_loss:.2f}
- Profit Factor: {self.profit_factor:.2f}
- Total Fees: ${self.total_fees:.2f}
"""

    def to_dataframe(self) -> pd.DataFrame:
        """Convert trades to DataFrame"""
        return pd.DataFrame(self.trades)


class Backtester:
    """
    Main backtesting engine for strategy evaluation.

    Supports both synchronous and asynchronous data sources.
    """

    def __init__(
        self,
        data_source: Optional[Any] = None,
        espn_source: Optional[Any] = None,
        fee_model: str = "kalshi",
        slippage: float = 0.01,  # 1 cent slippage
        commission: float = 0.0,  # Additional commission if any
        initial_capital: float = 1000.0,
        max_workers: int = 4
    ):
        """
        Initialize backtesting engine.

        Args:
            data_source: Data source for market prices
            espn_source: ESPN data source for play-by-play
            fee_model: Fee calculation model ("kalshi" or "fixed")
            slippage: Estimated slippage per trade
            commission: Additional commission per trade
            initial_capital: Starting capital for backtest
            max_workers: Number of parallel workers
        """
        self.data_source = data_source
        self.espn_source = espn_source
        self.fee_model = fee_model
        self.slippage = slippage
        self.commission = commission
        self.initial_capital = initial_capital
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def backtest(
        self,
        strategy,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        markets: Optional[List[str]] = None,
        use_espn: bool = False,
        parallel: bool = False
    ) -> BacktestResult:
        """
        Run backtest for a strategy.

        Args:
            strategy: Strategy instance to test
            start_date: Start date for backtest
            end_date: End date for backtest
            markets: List of market tickers to trade (None = all)
            use_espn: Whether to include ESPN data
            parallel: Run in parallel mode

        Returns:
            BacktestResult with performance metrics
        """
        # Convert dates
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # Reset strategy state
        strategy.reset()
        strategy.current_capital = self.initial_capital

        # Load historical data
        market_data = self._load_market_data(start_date, end_date, markets)
        espn_data = self._load_espn_data(start_date, end_date, markets) if use_espn else None

        # Run backtest
        if parallel and len(market_data) > 1000:
            results = self._run_parallel_backtest(strategy, market_data, espn_data)
        else:
            results = self._run_sequential_backtest(strategy, market_data, espn_data)

        # Calculate metrics
        return self._calculate_results(strategy, results, start_date, end_date)

    def _run_sequential_backtest(
        self,
        strategy,
        market_data: pd.DataFrame,
        espn_data: Optional[Dict]
    ) -> List[Dict]:
        """Run backtest sequentially"""
        trades = []
        positions = {}
        equity_curve = [self.initial_capital]

        # Group by timestamp for synchronized processing
        for timestamp in market_data['timestamp'].unique():
            current_data = market_data[market_data['timestamp'] == timestamp]

            # Get ESPN data for this timestamp if available
            current_espn = None
            if espn_data and timestamp in espn_data:
                current_espn = espn_data[timestamp]

            # Process each market at this timestamp
            for _, market in current_data.iterrows():
                ticker = market['ticker']

                # Update existing positions
                if ticker in positions:
                    position = positions[ticker]
                    position.current_price = market['yes_ask'] if position.side == 'yes' else market['no_ask']

                    # Check exit conditions
                    if strategy.should_close_position(position):
                        # Close position
                        exit_price = self._apply_slippage(
                            position.current_price,
                            'sell'
                        )
                        pnl = self._calculate_pnl(
                            position,
                            exit_price
                        )
                        fees = self._calculate_fees(
                            exit_price,
                            position.size
                        )
                        net_pnl = pnl - fees

                        trades.append({
                            'timestamp': timestamp,
                            'ticker': ticker,
                            'action': 'close',
                            'side': position.side,
                            'size': position.size,
                            'entry_price': position.entry_price,
                            'exit_price': exit_price,
                            'pnl': net_pnl,
                            'fees': fees
                        })

                        strategy.update_capital(net_pnl)
                        del positions[ticker]

                # Generate new signal
                signal = strategy.analyze(
                    current_data,
                    espn_data=current_espn
                )

                # Process signal
                if signal.type.value in ['buy_yes', 'buy_no'] and strategy.can_open_position():
                    # Open new position
                    side = 'yes' if signal.type.value == 'buy_yes' else 'no'
                    entry_price = self._apply_slippage(
                        market['yes_ask'] if side == 'yes' else market['no_ask'],
                        'buy'
                    )
                    fees = self._calculate_fees(entry_price, signal.size)

                    from ..strategies.base import Position
                    position = Position(
                        ticker=ticker,
                        side=side,
                        size=signal.size,
                        entry_price=entry_price,
                        current_price=entry_price,
                        entry_time=timestamp,
                        metadata=signal.metadata
                    )

                    positions[ticker] = position
                    strategy.positions.append(position)

                    trades.append({
                        'timestamp': timestamp,
                        'ticker': ticker,
                        'action': 'open',
                        'side': side,
                        'size': signal.size,
                        'entry_price': entry_price,
                        'exit_price': None,
                        'pnl': -fees,  # Initial cost is fees
                        'fees': fees,
                        'confidence': signal.confidence
                    })

                    strategy.update_capital(-entry_price * signal.size - fees)

            # Record equity
            total_value = strategy.current_capital
            for pos in positions.values():
                total_value += pos.size * pos.current_price
            equity_curve.append(total_value)

        # Close any remaining positions at end
        for ticker, position in positions.items():
            exit_price = position.current_price
            pnl = self._calculate_pnl(position, exit_price)
            fees = self._calculate_fees(exit_price, position.size)
            net_pnl = pnl - fees

            trades.append({
                'timestamp': market_data['timestamp'].iloc[-1],
                'ticker': ticker,
                'action': 'close',
                'side': position.side,
                'size': position.size,
                'entry_price': position.entry_price,
                'exit_price': exit_price,
                'pnl': net_pnl,
                'fees': fees,
                'forced_close': True
            })

            strategy.update_capital(net_pnl)

        return trades

    def _run_parallel_backtest(
        self,
        strategy,
        market_data: pd.DataFrame,
        espn_data: Optional[Dict]
    ) -> List[Dict]:
        """Run backtest in parallel (for large datasets)"""
        # Split data into chunks
        chunks = np.array_split(market_data, self.max_workers)

        # Process chunks in parallel
        futures = []
        for chunk in chunks:
            future = self.executor.submit(
                self._run_sequential_backtest,
                strategy,
                chunk,
                espn_data
            )
            futures.append(future)

        # Combine results
        all_trades = []
        for future in futures:
            trades = future.result()
            all_trades.extend(trades)

        return all_trades

    def _load_market_data(
        self,
        start_date: datetime,
        end_date: datetime,
        markets: Optional[List[str]]
    ) -> pd.DataFrame:
        """Load historical market data"""
        if self.data_source:
            # Use provided data source
            return self.data_source.load(start_date, end_date, markets)
        else:
            # Generate synthetic data for testing
            return self._generate_synthetic_data(start_date, end_date, markets)

    def _load_espn_data(
        self,
        start_date: datetime,
        end_date: datetime,
        markets: Optional[List[str]]
    ) -> Optional[Dict]:
        """Load ESPN play-by-play data"""
        if self.espn_source:
            return self.espn_source.load(start_date, end_date, markets)
        return None

    def _generate_synthetic_data(
        self,
        start_date: datetime,
        end_date: datetime,
        markets: Optional[List[str]]
    ) -> pd.DataFrame:
        """Generate synthetic market data for testing"""
        if markets is None:
            markets = ['KXNFLGAME-TEST1', 'KXNFLGAME-TEST2']

        dates = pd.date_range(start_date, end_date, freq='5min')
        data = []

        for date in dates:
            for market in markets:
                # Random walk for prices
                yes_price = np.random.uniform(0.3, 0.7)
                spread = np.random.uniform(0.01, 0.05)

                data.append({
                    'timestamp': date,
                    'ticker': market,
                    'yes_bid': yes_price - spread/2,
                    'yes_ask': yes_price + spread/2,
                    'no_bid': (1 - yes_price) - spread/2,
                    'no_ask': (1 - yes_price) + spread/2,
                    'volume': np.random.randint(100, 10000),
                    'open_interest': np.random.randint(1000, 50000)
                })

        return pd.DataFrame(data)

    def _apply_slippage(self, price: float, direction: str) -> float:
        """Apply slippage to execution price"""
        if direction == 'buy':
            return min(price + self.slippage, 0.99)
        else:
            return max(price - self.slippage, 0.01)

    def _calculate_fees(self, price: float, size: int) -> float:
        """Calculate trading fees"""
        if self.fee_model == "kalshi":
            # Kalshi fee formula
            fee = 0.07 * price * (1 - price) * size
        else:
            # Fixed fee model
            fee = self.commission * size

        return fee

    def _calculate_pnl(self, position, exit_price: float) -> float:
        """Calculate P&L for a position"""
        if position.side == 'yes':
            return (exit_price - position.entry_price) * position.size
        else:
            return (position.entry_price - exit_price) * position.size

    def _calculate_results(
        self,
        strategy,
        trades: List[Dict],
        start_date: datetime,
        end_date: datetime
    ) -> BacktestResult:
        """Calculate comprehensive backtest results"""
        if not trades:
            return BacktestResult(
                strategy_name=strategy.name,
                start_date=start_date,
                end_date=end_date,
                initial_capital=self.initial_capital,
                final_capital=self.initial_capital,
                total_return=0,
                total_return_pct=0,
                sharpe_ratio=0,
                max_drawdown=0,
                max_drawdown_pct=0,
                win_rate=0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                avg_win=0,
                avg_loss=0,
                profit_factor=0,
                total_fees=0
            )

        trades_df = pd.DataFrame(trades)

        # Calculate metrics
        total_pnl = trades_df['pnl'].sum()
        total_fees = trades_df['fees'].sum()
        final_capital = self.initial_capital + total_pnl

        # Win/loss statistics
        completed_trades = trades_df[trades_df['action'] == 'close']
        if len(completed_trades) > 0:
            wins = completed_trades[completed_trades['pnl'] > 0]
            losses = completed_trades[completed_trades['pnl'] <= 0]
            win_rate = len(wins) / len(completed_trades)
            avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
            avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0
            profit_factor = abs(wins['pnl'].sum() / losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0

        # Calculate returns and risk metrics
        equity_curve = self._build_equity_curve(trades_df, self.initial_capital)
        daily_returns = equity_curve.pct_change().dropna()

        # Sharpe ratio (annualized)
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0

        # Maximum drawdown
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown_pct = abs(drawdown.min()) * 100 if len(drawdown) > 0 else 0
        max_drawdown = max_drawdown_pct * self.initial_capital / 100

        return BacktestResult(
            strategy_name=strategy.name,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return=total_pnl,
            total_return_pct=(final_capital / self.initial_capital - 1) * 100,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            win_rate=win_rate,
            total_trades=len(completed_trades),
            winning_trades=len(wins) if 'wins' in locals() else 0,
            losing_trades=len(losses) if 'losses' in locals() else 0,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            total_fees=total_fees,
            trades=trades,
            equity_curve=equity_curve,
            daily_returns=daily_returns
        )

    def _build_equity_curve(
        self,
        trades_df: pd.DataFrame,
        initial_capital: float
    ) -> pd.Series:
        """Build equity curve from trades"""
        if trades_df.empty:
            return pd.Series([initial_capital])

        # Sort by timestamp
        trades_df = trades_df.sort_values('timestamp')

        # Calculate cumulative P&L
        equity = [initial_capital]
        current = initial_capital

        for _, trade in trades_df.iterrows():
            current += trade['pnl']
            equity.append(current)

        return pd.Series(equity, index=range(len(equity)))

    def compare_strategies(
        self,
        strategies: List,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        markets: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare multiple strategies on the same data.

        Args:
            strategies: List of strategy instances
            start_date: Start date for comparison
            end_date: End date for comparison
            markets: Markets to test on

        Returns:
            DataFrame comparing strategy performance
        """
        results = []

        for strategy in strategies:
            result = self.backtest(strategy, start_date, end_date, markets)
            results.append({
                'Strategy': strategy.name,
                'Total Return (%)': result.total_return_pct,
                'Sharpe Ratio': result.sharpe_ratio,
                'Max Drawdown (%)': result.max_drawdown_pct,
                'Win Rate (%)': result.win_rate * 100,
                'Total Trades': result.total_trades,
                'Profit Factor': result.profit_factor
            })

        return pd.DataFrame(results).sort_values('Sharpe Ratio', ascending=False)