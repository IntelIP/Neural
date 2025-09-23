"""
Order Manager for executing analysis signals

Bridges the analysis stack with the trading stack for order execution.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
from ..strategies.base import Signal, SignalType, Position


class OrderManager:
    """
    Manages order execution from strategy signals.

    Handles:
    - Signal to order conversion
    - Order routing to trading client
    - Position tracking
    - Risk checks before execution
    """

    def __init__(
        self,
        trading_client=None,
        max_slippage: float = 0.02,
        require_confirmation: bool = False,
        dry_run: bool = False
    ):
        """
        Initialize order manager.

        Args:
            trading_client: Neural trading client instance
            max_slippage: Maximum acceptable slippage
            require_confirmation: Require manual confirmation
            dry_run: Simulate orders without execution
        """
        self.trading_client = trading_client
        self.max_slippage = max_slippage
        self.require_confirmation = require_confirmation
        self.dry_run = dry_run

        # State tracking
        self.pending_orders: List[Dict] = []
        self.executed_orders: List[Dict] = []
        self.active_positions: Dict[str, Position] = {}
        self.order_history: List[Dict] = []

    async def execute_signal(self, signal: Signal) -> Optional[Dict]:
        """
        Execute a trading signal.

        Args:
            signal: Trading signal from strategy

        Returns:
            Order result or None if not executed
        """
        if signal.type == SignalType.HOLD:
            return None

        # Risk checks
        if not self._pass_risk_checks(signal):
            print(f"Signal failed risk checks: {signal.ticker}")
            return None

        # Manual confirmation if required
        if self.require_confirmation:
            if not await self._get_confirmation(signal):
                return None

        # Route to appropriate handler
        if signal.type == SignalType.BUY_YES:
            return await self._execute_buy_yes(signal)
        elif signal.type == SignalType.BUY_NO:
            return await self._execute_buy_no(signal)
        elif signal.type == SignalType.SELL_YES:
            return await self._execute_sell_yes(signal)
        elif signal.type == SignalType.SELL_NO:
            return await self._execute_sell_no(signal)
        elif signal.type == SignalType.CLOSE:
            return await self._execute_close(signal)

        return None

    async def _execute_buy_yes(self, signal: Signal) -> Optional[Dict]:
        """Execute BUY_YES order"""
        if self.dry_run:
            return self._simulate_order(signal, "buy", "yes")

        if not self.trading_client:
            raise ValueError("Trading client not configured")

        # Check for arbitrage (need to buy both sides)
        if signal.metadata and signal.metadata.get('also_buy') == 'no':
            # Execute arbitrage trades
            yes_order = await self._place_order(
                signal.ticker,
                "buy",
                "yes",
                signal.size,
                signal.entry_price
            )

            no_order = await self._place_order(
                signal.ticker,
                "buy",
                "no",
                signal.metadata.get('no_size', signal.size),
                signal.metadata.get('no_price')
            )

            return {
                'type': 'arbitrage',
                'yes_order': yes_order,
                'no_order': no_order,
                'signal': signal
            }

        # Regular buy YES
        return await self._place_order(
            signal.ticker,
            "buy",
            "yes",
            signal.size,
            signal.entry_price
        )

    async def _execute_buy_no(self, signal: Signal) -> Optional[Dict]:
        """Execute BUY_NO order"""
        if self.dry_run:
            return self._simulate_order(signal, "buy", "no")

        if not self.trading_client:
            raise ValueError("Trading client not configured")

        return await self._place_order(
            signal.ticker,
            "buy",
            "no",
            signal.size,
            signal.entry_price
        )

    async def _execute_sell_yes(self, signal: Signal) -> Optional[Dict]:
        """Execute SELL_YES order"""
        if self.dry_run:
            return self._simulate_order(signal, "sell", "yes")

        if not self.trading_client:
            raise ValueError("Trading client not configured")

        return await self._place_order(
            signal.ticker,
            "sell",
            "yes",
            signal.size,
            signal.entry_price
        )

    async def _execute_sell_no(self, signal: Signal) -> Optional[Dict]:
        """Execute SELL_NO order"""
        if self.dry_run:
            return self._simulate_order(signal, "sell", "no")

        if not self.trading_client:
            raise ValueError("Trading client not configured")

        return await self._place_order(
            signal.ticker,
            "sell",
            "no",
            signal.size,
            signal.entry_price
        )

    async def _execute_close(self, signal: Signal) -> Optional[Dict]:
        """Close existing position"""
        if signal.ticker not in self.active_positions:
            print(f"No position to close for {signal.ticker}")
            return None

        position = self.active_positions[signal.ticker]

        if self.dry_run:
            del self.active_positions[signal.ticker]
            return {
                'type': 'close',
                'position': position,
                'pnl': position.pnl
            }

        # Close through trading client
        if position.side == "yes":
            return await self._place_order(
                signal.ticker,
                "sell",
                "yes",
                position.size,
                None  # Market order
            )
        else:
            return await self._place_order(
                signal.ticker,
                "sell",
                "no",
                position.size,
                None
            )

    async def _place_order(
        self,
        ticker: str,
        action: str,
        side: str,
        size: int,
        limit_price: Optional[float] = None
    ) -> Dict:
        """
        Place order through trading client.

        Args:
            ticker: Market ticker
            action: 'buy' or 'sell'
            side: 'yes' or 'no'
            size: Number of contracts
            limit_price: Limit price (None for market)

        Returns:
            Order result
        """
        try:
            # Get current market price
            market = await self.trading_client.markets.get_market(ticker)

            if limit_price:
                # Limit order
                order = await self.trading_client.orders.place_limit_order(
                    ticker=ticker,
                    side=side,
                    action=action,
                    count=size,
                    limit_price=int(limit_price * 100)  # Convert to cents
                )
            else:
                # Market order
                order = await self.trading_client.orders.place_market_order(
                    ticker=ticker,
                    side=side,
                    action=action,
                    count=size
                )

            # Track order
            order_record = {
                'timestamp': datetime.now(),
                'ticker': ticker,
                'action': action,
                'side': side,
                'size': size,
                'price': limit_price or market.get(f'{side}_ask'),
                'order_id': order.get('order_id'),
                'status': 'executed'
            }

            self.executed_orders.append(order_record)
            self.order_history.append(order_record)

            # Update positions
            if action == "buy":
                self._add_position(ticker, side, size, order_record['price'])
            elif action == "sell":
                self._remove_position(ticker, side, size)

            return order_record

        except Exception as e:
            print(f"Order execution failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'ticker': ticker
            }

    def _simulate_order(self, signal: Signal, action: str, side: str) -> Dict:
        """Simulate order for dry run mode"""
        order = {
            'timestamp': datetime.now(),
            'ticker': signal.ticker,
            'action': action,
            'side': side,
            'size': signal.size,
            'price': signal.entry_price,
            'confidence': signal.confidence,
            'simulated': True,
            'signal': signal
        }

        self.executed_orders.append(order)
        self.order_history.append(order)

        if action == "buy":
            self._add_position(signal.ticker, side, signal.size, signal.entry_price)

        return order

    def _add_position(self, ticker: str, side: str, size: int, price: float):
        """Add or update position"""
        if ticker in self.active_positions:
            # Update existing position (average in)
            pos = self.active_positions[ticker]
            total_cost = pos.entry_price * pos.size + price * size
            pos.size += size
            pos.entry_price = total_cost / pos.size
        else:
            # New position
            self.active_positions[ticker] = Position(
                ticker=ticker,
                side=side,
                size=size,
                entry_price=price,
                current_price=price,
                entry_time=datetime.now()
            )

    def _remove_position(self, ticker: str, side: str, size: int):
        """Remove or reduce position"""
        if ticker not in self.active_positions:
            return

        pos = self.active_positions[ticker]
        pos.size -= size

        if pos.size <= 0:
            del self.active_positions[ticker]

    def _pass_risk_checks(self, signal: Signal) -> bool:
        """Perform risk checks before execution"""
        # Check position limits
        if len(self.active_positions) >= 20:
            return False

        # Check concentration
        if signal.ticker in self.active_positions:
            pos = self.active_positions[signal.ticker]
            if pos.size + signal.size > 1000:  # Max 1000 contracts per market
                return False

        # Check confidence threshold
        if signal.confidence < 0.3:
            return False

        return True

    async def _get_confirmation(self, signal: Signal) -> bool:
        """Get manual confirmation for order"""
        print(f"\n{'='*50}")
        print(f"CONFIRM ORDER:")
        print(f"  Ticker: {signal.ticker}")
        print(f"  Type: {signal.type.value}")
        print(f"  Size: {signal.size} contracts")
        print(f"  Price: ${signal.entry_price:.2f}")
        print(f"  Confidence: {signal.confidence:.1%}")

        if signal.metadata:
            print(f"  Metadata: {signal.metadata}")

        response = input("Execute? (y/n): ").lower()
        return response == 'y'

    def update_prices(self, market_data: pd.DataFrame):
        """Update current prices for positions"""
        for ticker, position in self.active_positions.items():
            ticker_data = market_data[market_data['ticker'] == ticker]
            if not ticker_data.empty:
                latest = ticker_data.iloc[-1]
                if position.side == "yes":
                    position.current_price = latest['yes_ask']
                else:
                    position.current_price = latest['no_ask']

    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary"""
        total_value = sum(
            pos.size * pos.current_price
            for pos in self.active_positions.values()
        )

        total_cost = sum(
            pos.size * pos.entry_price
            for pos in self.active_positions.values()
        )

        total_pnl = sum(pos.pnl for pos in self.active_positions.values())

        return {
            'positions': len(self.active_positions),
            'total_value': total_value,
            'total_cost': total_cost,
            'total_pnl': total_pnl,
            'total_orders': len(self.executed_orders),
            'active_positions': {
                ticker: {
                    'side': pos.side,
                    'size': pos.size,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'pnl': pos.pnl,
                    'pnl_pct': pos.pnl_percentage
                }
                for ticker, pos in self.active_positions.items()
            }
        }

    async def close_all_positions(self) -> List[Dict]:
        """Close all open positions"""
        results = []

        for ticker in list(self.active_positions.keys()):
            signal = Signal(
                type=SignalType.CLOSE,
                ticker=ticker,
                size=0,
                confidence=1.0
            )
            result = await self.execute_signal(signal)
            if result:
                results.append(result)

        return results