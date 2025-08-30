"""
Portfolio Monitor Agent - Always-On Agent
Continuously monitors positions and enforces risk limits
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import redis.asyncio as redis

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a trading position"""
    market_ticker: str
    side: str  # 'yes' or 'no'
    quantity: int
    entry_price: float
    current_price: float
    timestamp: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L"""
        price_diff = self.current_price - self.entry_price
        return self.quantity * price_diff
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """Calculate unrealized P&L percentage"""
        if self.entry_price == 0:
            return 0
        return (self.current_price - self.entry_price) / self.entry_price
    
    @property
    def cost_basis(self) -> float:
        """Calculate position cost basis"""
        return self.quantity * self.entry_price
    
    def should_stop_loss(self) -> bool:
        """Check if stop-loss should trigger"""
        if self.stop_loss is None:
            return self.unrealized_pnl_pct <= -0.10  # Default 10% stop-loss
        return self.current_price <= self.stop_loss
    
    def should_take_profit(self) -> bool:
        """Check if take-profit should trigger"""
        if self.take_profit is None:
            return self.unrealized_pnl_pct >= 0.30  # Default 30% take-profit
        return self.current_price >= self.take_profit


@dataclass
class Portfolio:
    """Portfolio state and metrics"""
    positions: List[Position] = field(default_factory=list)
    cash_balance: float = 10000.0  # Starting balance
    starting_balance: float = 10000.0
    daily_pnl: float = 0.0
    daily_trades: int = 0
    max_drawdown: float = 0.0
    
    @property
    def total_value(self) -> float:
        """Calculate total portfolio value"""
        position_value = sum(p.current_price * p.quantity for p in self.positions)
        return self.cash_balance + position_value
    
    @property
    def unrealized_pnl(self) -> float:
        """Calculate total unrealized P&L"""
        return sum(p.unrealized_pnl for p in self.positions)
    
    @property
    def exposure(self) -> float:
        """Calculate total market exposure"""
        return sum(p.cost_basis for p in self.positions)
    
    @property
    def exposure_pct(self) -> float:
        """Calculate exposure as percentage of portfolio"""
        if self.total_value == 0:
            return 0
        return self.exposure / self.total_value
    
    @property
    def daily_return(self) -> float:
        """Calculate daily return percentage"""
        if self.starting_balance == 0:
            return 0
        return self.daily_pnl / self.starting_balance
    
    def update_drawdown(self):
        """Update maximum drawdown"""
        current_drawdown = (self.starting_balance - self.total_value) / self.starting_balance
        self.max_drawdown = max(self.max_drawdown, current_drawdown)


class PortfolioMonitorAgent:
    """
    Portfolio Monitor Agent - Always-On
    
    Responsibilities:
    - Track all open positions
    - Monitor P&L in real-time  
    - Enforce position limits (max 5% per position)
    - Trigger stop-losses (10% loss)
    - Trigger take-profits (30% gain)
    - Alert on margin requirements
    - Calculate portfolio Greeks
    - Halt trading on daily loss limit (20%)
    """
    
    # Risk parameters
    MAX_POSITION_PCT = 0.05  # Max 5% per position
    MAX_DAILY_LOSS_PCT = 0.20  # Max 20% daily loss
    DEFAULT_STOP_LOSS_PCT = 0.10  # 10% stop-loss
    DEFAULT_TAKE_PROFIT_PCT = 0.30  # 30% take-profit
    MAX_CORRELATION = 0.7  # Max correlation between positions
    MARGIN_REQUIREMENT = 2.0  # 2x Kelly fraction
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client = None
        self.pubsub = None
        self.is_running = False
        self.trading_halted = False
        
        # Portfolio state
        self.portfolio = Portfolio()
        self.positions_by_market: Dict[str, Position] = {}
        
        # Price cache for position updates
        self.price_cache: Dict[str, float] = {}
        
        # Statistics
        self.stops_triggered = 0
        self.profits_taken = 0
        self.risk_alerts = 0
    
    async def connect(self):
        """Connect to Redis"""
        self.redis_client = redis.from_url(self.redis_url)
        self.pubsub = self.redis_client.pubsub()
        
        # Subscribe to relevant channels
        await self.pubsub.subscribe(
            "kalshi:markets",  # Price updates
            "trades:executed",  # New positions
            "trades:closed",    # Closed positions
            "risk:override"     # Risk overrides
        )
        
        logger.info("Portfolio Monitor connected to Redis")
    
    async def start(self):
        """Start portfolio monitoring"""
        if self.is_running:
            logger.warning("Portfolio Monitor already running")
            return
        
        self.is_running = True
        self.trading_halted = False
        
        logger.info("Portfolio Monitor started")
        
        # Start monitoring tasks
        asyncio.create_task(self._monitor_positions())
        asyncio.create_task(self._process_messages())
        asyncio.create_task(self._periodic_risk_check())
    
    async def _process_messages(self):
        """Process Redis messages"""
        async for message in self.pubsub.listen():
            if not self.is_running:
                break
            
            if message['type'] == 'message':
                try:
                    channel = message['channel'].decode('utf-8')
                    data = json.loads(message['data'])
                    
                    if channel == "kalshi:markets":
                        await self._handle_price_update(data)
                    elif channel == "trades:executed":
                        await self._handle_new_position(data)
                    elif channel == "trades:closed":
                        await self._handle_closed_position(data)
                    elif channel == "risk:override":
                        await self._handle_risk_override(data)
                        
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
    
    async def _monitor_positions(self):
        """Monitor positions every 10 seconds"""
        while self.is_running:
            try:
                # Update position prices
                await self._update_position_prices()
                
                # Check each position
                for market_ticker, position in self.positions_by_market.items():
                    # Check stop-loss
                    if position.should_stop_loss():
                        await self._execute_stop_loss(position)
                    
                    # Check take-profit
                    elif position.should_take_profit():
                        await self._execute_take_profit(position)
                
                # Check portfolio-level limits
                await self._check_portfolio_limits()
                
            except Exception as e:
                logger.error(f"Error monitoring positions: {e}")
            
            await asyncio.sleep(10)
    
    async def _periodic_risk_check(self):
        """Perform periodic risk checks every minute"""
        while self.is_running:
            try:
                # Calculate portfolio metrics
                self.portfolio.update_drawdown()
                
                # Check correlation between positions
                if len(self.portfolio.positions) > 1:
                    await self._check_correlation()
                
                # Check margin requirements
                await self._check_margin_requirements()
                
                # Publish portfolio status
                await self._publish_portfolio_status()
                
            except Exception as e:
                logger.error(f"Error in risk check: {e}")
            
            await asyncio.sleep(60)
    
    async def _handle_price_update(self, data: Dict[str, Any]):
        """Handle market price updates"""
        market_data = data.get('data', {})
        market_ticker = market_data.get('market_ticker')
        yes_price = market_data.get('yes_price')
        
        if market_ticker and yes_price:
            # Update price cache
            self.price_cache[market_ticker] = yes_price
            
            # Update position if we have one
            if market_ticker in self.positions_by_market:
                self.positions_by_market[market_ticker].current_price = yes_price
    
    async def _handle_new_position(self, data: Dict[str, Any]):
        """Handle new position opened"""
        position_data = data.get('data', {})
        
        position = Position(
            market_ticker=position_data['market_ticker'],
            side=position_data['side'],
            quantity=position_data['quantity'],
            entry_price=position_data['price'],
            current_price=position_data['price'],
            timestamp=datetime.fromisoformat(position_data['timestamp']),
            stop_loss=position_data.get('stop_loss'),
            take_profit=position_data.get('take_profit')
        )
        
        # Add to portfolio
        self.portfolio.positions.append(position)
        self.positions_by_market[position.market_ticker] = position
        
        # Update cash balance
        self.portfolio.cash_balance -= position.cost_basis
        
        # Update daily trades
        self.portfolio.daily_trades += 1
        
        logger.info(f"New position added: {position.market_ticker} {position.side} x{position.quantity}")
    
    async def _handle_closed_position(self, data: Dict[str, Any]):
        """Handle position closed"""
        close_data = data.get('data', {})
        market_ticker = close_data['market_ticker']
        
        if market_ticker in self.positions_by_market:
            position = self.positions_by_market[market_ticker]
            
            # Calculate realized P&L
            realized_pnl = position.unrealized_pnl
            
            # Update portfolio
            self.portfolio.cash_balance += position.current_price * position.quantity
            self.portfolio.daily_pnl += realized_pnl
            self.portfolio.positions.remove(position)
            del self.positions_by_market[market_ticker]
            
            logger.info(f"Position closed: {market_ticker}, P&L: ${realized_pnl:.2f}")
    
    async def _handle_risk_override(self, data: Dict[str, Any]):
        """Handle risk override commands"""
        command = data.get('command')
        
        if command == 'halt_trading':
            self.trading_halted = True
            logger.warning("Trading halted by risk override")
        elif command == 'resume_trading':
            self.trading_halted = False
            logger.info("Trading resumed")
        elif command == 'close_all':
            await self._close_all_positions()
    
    async def _update_position_prices(self):
        """Update current prices for all positions"""
        for position in self.portfolio.positions:
            if position.market_ticker in self.price_cache:
                position.current_price = self.price_cache[position.market_ticker]
    
    async def _execute_stop_loss(self, position: Position):
        """Execute stop-loss for position"""
        logger.warning(f"STOP-LOSS triggered for {position.market_ticker}")
        
        # Publish stop-loss order
        stop_order = {
            "action": "STOP_LOSS",
            "market_ticker": position.market_ticker,
            "side": "sell" if position.side == "yes" else "buy",
            "quantity": position.quantity,
            "reason": f"Stop-loss at {position.unrealized_pnl_pct:.1%}",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.redis_client.publish("trades:orders", json.dumps(stop_order))
        
        # Publish risk alert
        await self._publish_risk_alert(
            f"Stop-loss executed: {position.market_ticker}",
            {"position": position.__dict__, "loss_pct": position.unrealized_pnl_pct}
        )
        
        self.stops_triggered += 1
    
    async def _execute_take_profit(self, position: Position):
        """Execute take-profit for position"""
        logger.info(f"TAKE-PROFIT triggered for {position.market_ticker}")
        
        # Publish take-profit order
        profit_order = {
            "action": "TAKE_PROFIT",
            "market_ticker": position.market_ticker,
            "side": "sell" if position.side == "yes" else "buy",
            "quantity": position.quantity,
            "reason": f"Take-profit at {position.unrealized_pnl_pct:.1%}",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.redis_client.publish("trades:orders", json.dumps(profit_order))
        
        self.profits_taken += 1
    
    async def _check_portfolio_limits(self):
        """Check portfolio-level risk limits"""
        # Check daily loss limit
        if self.portfolio.daily_return <= -self.MAX_DAILY_LOSS_PCT:
            if not self.trading_halted:
                await self._halt_trading(f"Daily loss limit reached: {self.portfolio.daily_return:.1%}")
        
        # Check position concentration
        for position in self.portfolio.positions:
            position_pct = position.cost_basis / self.portfolio.total_value
            if position_pct > self.MAX_POSITION_PCT:
                await self._publish_risk_alert(
                    f"Position too large: {position.market_ticker}",
                    {"position_pct": position_pct, "limit": self.MAX_POSITION_PCT}
                )
    
    async def _check_correlation(self):
        """Check correlation between positions"""
        # Simplified correlation check - in production would use actual correlation matrix
        similar_markets = {}
        
        for position in self.portfolio.positions:
            # Group by similar market types (e.g., same game, same team)
            market_type = position.market_ticker.split('-')[0]
            if market_type not in similar_markets:
                similar_markets[market_type] = []
            similar_markets[market_type].append(position)
        
        # Alert if too many correlated positions
        for market_type, positions in similar_markets.items():
            if len(positions) > 2:
                await self._publish_risk_alert(
                    f"High correlation risk: {len(positions)} positions in {market_type}",
                    {"market_type": market_type, "position_count": len(positions)}
                )
    
    async def _check_margin_requirements(self):
        """Check margin requirements"""
        required_margin = self.portfolio.exposure * self.MARGIN_REQUIREMENT
        available_margin = self.portfolio.cash_balance
        
        if available_margin < required_margin:
            await self._publish_risk_alert(
                "Insufficient margin",
                {
                    "required": required_margin,
                    "available": available_margin,
                    "shortfall": required_margin - available_margin
                }
            )
    
    async def _halt_trading(self, reason: str):
        """Halt all trading"""
        self.trading_halted = True
        logger.critical(f"TRADING HALTED: {reason}")
        
        halt_message = {
            "action": "HALT",
            "reason": reason,
            "portfolio_value": self.portfolio.total_value,
            "daily_pnl": self.portfolio.daily_pnl,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.redis_client.publish("risk:halt", json.dumps(halt_message))
        await self._publish_risk_alert(f"Trading halted: {reason}", halt_message)
    
    async def _close_all_positions(self):
        """Close all open positions"""
        logger.warning("Closing all positions")
        
        for position in self.portfolio.positions:
            close_order = {
                "action": "CLOSE_ALL",
                "market_ticker": position.market_ticker,
                "side": "sell" if position.side == "yes" else "buy",
                "quantity": position.quantity,
                "reason": "Risk management: close all",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self.redis_client.publish("trades:orders", json.dumps(close_order))
    
    async def _publish_risk_alert(self, message: str, data: Dict[str, Any]):
        """Publish risk alert"""
        alert = {
            "type": "RISK_ALERT",
            "message": message,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.redis_client.publish("events:portfolio", json.dumps(alert))
        self.risk_alerts += 1
    
    async def _publish_portfolio_status(self):
        """Publish portfolio status update"""
        status = {
            "total_value": self.portfolio.total_value,
            "cash_balance": self.portfolio.cash_balance,
            "unrealized_pnl": self.portfolio.unrealized_pnl,
            "daily_pnl": self.portfolio.daily_pnl,
            "daily_return": self.portfolio.daily_return,
            "exposure": self.portfolio.exposure,
            "exposure_pct": self.portfolio.exposure_pct,
            "position_count": len(self.portfolio.positions),
            "daily_trades": self.portfolio.daily_trades,
            "max_drawdown": self.portfolio.max_drawdown,
            "trading_halted": self.trading_halted,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.redis_client.publish("portfolio:status", json.dumps(status))
    
    async def check_capacity(self, proposed_size: float) -> Dict[str, Any]:
        """
        Check if portfolio has capacity for new position
        
        Args:
            proposed_size: Proposed position size in dollars
            
        Returns:
            Approval status and details
        """
        # Check if trading is halted
        if self.trading_halted:
            return {
                "approved": False,
                "reason": "Trading is halted"
            }
        
        # Check position size limit
        position_pct = proposed_size / self.portfolio.total_value
        if position_pct > self.MAX_POSITION_PCT:
            return {
                "approved": False,
                "reason": f"Position too large: {position_pct:.1%} > {self.MAX_POSITION_PCT:.1%}"
            }
        
        # Check daily loss limit
        if self.portfolio.daily_return <= -self.MAX_DAILY_LOSS_PCT * 0.8:  # 80% of limit
            return {
                "approved": False,
                "reason": f"Approaching daily loss limit: {self.portfolio.daily_return:.1%}"
            }
        
        # Check available cash
        if proposed_size > self.portfolio.cash_balance:
            return {
                "approved": False,
                "reason": f"Insufficient cash: ${proposed_size:.2f} > ${self.portfolio.cash_balance:.2f}"
            }
        
        return {
            "approved": True,
            "available_cash": self.portfolio.cash_balance,
            "current_exposure": self.portfolio.exposure_pct,
            "daily_return": self.portfolio.daily_return
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get portfolio monitor statistics"""
        return {
            "is_running": self.is_running,
            "trading_halted": self.trading_halted,
            "portfolio": {
                "total_value": self.portfolio.total_value,
                "positions": len(self.portfolio.positions),
                "unrealized_pnl": self.portfolio.unrealized_pnl,
                "daily_pnl": self.portfolio.daily_pnl,
                "daily_return": f"{self.portfolio.daily_return:.1%}",
                "max_drawdown": f"{self.portfolio.max_drawdown:.1%}"
            },
            "risk_events": {
                "stops_triggered": self.stops_triggered,
                "profits_taken": self.profits_taken,
                "risk_alerts": self.risk_alerts
            }
        }
    
    async def stop(self):
        """Stop portfolio monitoring"""
        self.is_running = False
        
        # Close all positions if any remain
        if self.portfolio.positions:
            await self._close_all_positions()
        
        if self.pubsub:
            await self.pubsub.unsubscribe()
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info(f"Portfolio Monitor stopped. Final stats: {self.get_statistics()}")


# Example usage
async def main():
    """Example of running the Portfolio Monitor"""
    monitor = PortfolioMonitorAgent()
    await monitor.connect()
    await monitor.start()
    
    # Simulate adding a position
    await asyncio.sleep(2)
    
    # Check capacity for new position
    capacity = await monitor.check_capacity(500.0)
    print(f"Capacity check: {capacity}")
    
    # Get statistics
    stats = monitor.get_statistics()
    print(f"Portfolio stats: {json.dumps(stats, indent=2)}")
    
    # Run for a while
    await asyncio.sleep(60)
    
    await monitor.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())