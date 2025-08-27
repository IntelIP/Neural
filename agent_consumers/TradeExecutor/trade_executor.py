"""
Trade Executor Agent
Executes trades on Kalshi based on signals from other agents
Powered by Google Gemini 2.5 Flash
"""

import os
import asyncio
from typing import Dict, Any, List
from datetime import datetime
import logging

from trading_logic.llm_client import get_llm_client
from trading_logic.stop_loss import DynamicStopLoss

# Stub Kalshi client (will use actual kalshi_web_infra when needed)
class KalshiHttpClient:
    def __init__(self, **kwargs):
        # Will be properly initialized when needed
        pass

# Simple Kelly Calculator (was in tools.kelly_tools)
class KellyCalculator:
    def calculate_kelly_fraction(self, prob_win: float, odds: float) -> float:
        """Calculate Kelly fraction for position sizing"""
        q = 1 - prob_win
        b = odds - 1
        return (prob_win * b - q) / b if b > 0 else 0

logger = logging.getLogger(__name__)


class TradeExecutorAgent:
    """
    Trade Executor Agent - Executes trades on Kalshi markets.
    
    Responsibilities:
    - Execute buy/sell orders based on signals
    - Manage order lifecycle (placement, monitoring, cancellation)
    - Apply Kelly Criterion position sizing
    - Set and manage stop-losses
    - Track execution quality and slippage
    """
    
    def __init__(self):
        """Initialize Trade Executor Agent."""
        # Initialize LLM client
        self.llm_client = get_llm_client()
        
        # Initialize Kalshi API
        self.kalshi = KalshiHttpClient(
            host="https://api.elections.kalshi.com/trade-api/v2",
            key_id=os.getenv("KALSHI_API_KEY_ID"),
            private_key=os.getenv("KALSHI_PRIVATE_KEY")
        )
        
        # Initialize calculators
        self.kelly_calculator = KellyCalculator()
        self.stop_loss_calculator = DynamicStopLoss()
        
        # Message handler will be set by the handler wrapper
        self.message_handler = None
        
        # Trading parameters
        self.bankroll = float(os.getenv("TRADING_BANKROLL", "100000"))
        self.kelly_fraction = float(os.getenv("KELLY_FRACTION", "0.25"))  # Quarter Kelly
        self.max_position_pct = float(os.getenv("MAX_POSITION_PCT", "0.05"))  # 5% max per position
        self.min_edge = float(os.getenv("MIN_EDGE", "0.05"))  # 5% minimum edge
        
        # State tracking
        self.active_orders: Dict[str, Dict] = {}
        self.positions: Dict[str, Dict] = {}
        self.is_running = False
        self.tasks: List[asyncio.Task] = []
    
    async def start(self):
        """Start the Trade Executor agent."""
        if self.is_running:
            logger.warning("Trade Executor already running")
            return
        
        self.is_running = True
        logger.info("Starting Trade Executor Agent...")
        
        # Initialize Kalshi connection
        await self.kalshi.initialize()
        
        # Connection and message handlers handled by Agentuity
        
        # Load existing positions
        await self._load_positions()
        
        # Start monitoring tasks
        self.tasks.append(
            asyncio.create_task(self._order_monitoring_loop())
        )
        self.tasks.append(
            asyncio.create_task(self._stop_loss_monitoring_loop())
        )
        
        logger.info("Trade Executor Agent started successfully")
    
    async def execute_trade(
        self,
        market_ticker: str,
        action: str,  # BUY_YES, BUY_NO, SELL_YES, SELL_NO
        probability: float,
        confidence: float,
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a trade based on signal.
        
        Args:
            market_ticker: Kalshi market ticker
            action: Trading action
            probability: Estimated win probability
            confidence: Confidence in estimate
            market_data: Current market prices and data
        
        Returns:
            Execution result
        """
        try:
            # Use Gemini to analyze execution strategy
            analysis_prompt = f"""
            Analyze optimal execution strategy for:
            - Market: {market_ticker}
            - Action: {action}
            - Our Probability: {probability:.2f}
            - Confidence: {confidence:.2f}
            - Current Bid/Ask: ${market_data.get('yes_bid', 0):.2f}/${market_data.get('yes_ask', 0):.2f}
            - Volume 24h: {market_data.get('volume_24h', 0)}
            
            Recommend:
            1. Order type (market vs limit)
            2. If limit, what price?
            3. Execution urgency
            4. Potential slippage concerns
            """
            
            response_text = await self.llm_client.complete(
                analysis_prompt,
                temperature=0.0,  # Zero temperature for precise execution
                system_prompt="You are a trade execution expert analyzing optimal order placement."
            )
            logger.info(f"Execution analysis: {response_text}")
            
            # Calculate position size using Kelly
            if "YES" in action:
                market_price = market_data.get('yes_ask' if 'BUY' in action else 'yes_bid', 0.5)
                edge = probability - market_price
            else:  # NO
                market_price = market_data.get('no_ask' if 'BUY' in action else 'no_bid', 0.5)
                edge = (1 - probability) - market_price
            
            # Check minimum edge requirement
            if edge < self.min_edge:
                logger.info(f"Insufficient edge: {edge:.3f} < {self.min_edge}")
                return {
                    "status": "rejected",
                    "reason": "insufficient_edge",
                    "edge": edge
                }
            
            # Calculate Kelly position
            odds = (1 - market_price) / market_price
            kelly = self.kelly_calculator.calculate_kelly_fraction(
                prob_win=probability if "YES" in action else (1 - probability),
                odds=odds
            )
            
            position_size = self.kelly_calculator.calculate_position_size(
                kelly_fraction=kelly,
                bankroll=self.bankroll,
                safety_factor=self.kelly_fraction,
                max_position_pct=self.max_position_pct
            )
            
            if position_size == 0:
                return {
                    "status": "rejected",
                    "reason": "position_too_small"
                }
            
            # Calculate number of contracts
            contracts = int(position_size / market_price)
            
            # Place order on Kalshi
            if 'BUY' in action:
                order = await self._place_buy_order(
                    market_ticker=market_ticker,
                    side="YES" if "YES" in action else "NO",
                    contracts=contracts,
                    limit_price=market_price
                )
            else:  # SELL
                order = await self._place_sell_order(
                    market_ticker=market_ticker,
                    side="YES" if "YES" in action else "NO",
                    contracts=contracts,
                    limit_price=market_price
                )
            
            if order.get("success"):
                # Track order
                self.active_orders[order["order_id"]] = {
                    "market_ticker": market_ticker,
                    "action": action,
                    "contracts": contracts,
                    "price": market_price,
                    "kelly": kelly,
                    "probability": probability,
                    "created_at": datetime.now()
                }
                
                # Calculate stop-loss
                stop_loss = self.stop_loss_calculator.calculate_stop_loss(
                    entry_price=market_price,
                    current_price=market_price,
                    sentiment_momentum=market_data.get("sentiment_momentum", 0),
                    volatility=market_data.get("volatility", 0.1),
                    time_to_event_hours=market_data.get("hours_to_close")
                )
                
                logger.info(f"Order placed: {order['order_id']} - {contracts} contracts at ${market_price:.2f}")
                logger.info(f"Stop-loss set at ${stop_loss['stop_price']:.2f}")
                
                return {
                    "status": "success",
                    "order_id": order["order_id"],
                    "contracts": contracts,
                    "price": market_price,
                    "position_size": position_size,
                    "kelly_fraction": kelly,
                    "stop_loss": stop_loss["stop_price"]
                }
            else:
                return {
                    "status": "failed",
                    "reason": order.get("error", "order_placement_failed")
                }
            
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _place_buy_order(
        self,
        market_ticker: str,
        side: str,
        contracts: int,
        limit_price: float
    ) -> Dict[str, Any]:
        """Place buy order on Kalshi."""
        try:
            # Convert price to centi-cents for Kalshi API
            price_centicents = int(limit_price * 10000)
            
            order_params = {
                "market_ticker": market_ticker,
                "side": side,
                "quantity": contracts,
                "price": price_centicents,
                "order_type": "limit",
                "time_in_force": "GTC"  # Good till cancelled
            }
            
            result = await self.kalshi.place_order(order_params)
            
            return {
                "success": True,
                "order_id": result.get("order_id"),
                "status": result.get("status")
            }
            
        except Exception as e:
            logger.error(f"Order placement error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _place_sell_order(
        self,
        market_ticker: str,
        side: str,
        contracts: int,
        limit_price: float
    ) -> Dict[str, Any]:
        """Place sell order on Kalshi."""
        # Similar to buy order but for selling existing position
        return await self._place_buy_order(
            market_ticker, 
            "NO" if side == "YES" else "YES",  # Opposite side to close
            contracts,
            1 - limit_price  # Inverse price for closing
        )
    
    async def _handle_trading_signal(self, data: Dict[str, Any]):
        """Handle incoming trading signal."""
        try:
            market_ticker = data.get("market_ticker")
            action = data.get("action")
            probability = data.get("probability")
            confidence = data.get("confidence")
            market_data = data.get("market_data", {})
            
            logger.info(f"Received signal: {market_ticker} - {action}")
            
            # Execute trade
            result = await self.execute_trade(
                market_ticker=market_ticker,
                action=action,
                probability=probability,
                confidence=confidence,
                market_data=market_data
            )
            
            # Send execution result to handler
            if self.message_handler:
                await self.message_handler('trade_executed', {
                    "market_ticker": market_ticker,
                    "action": action,
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                })
            
        except Exception as e:
            logger.error(f"Error handling trading signal: {e}")
    
    async def _order_monitoring_loop(self):
        """Monitor active orders for fills and cancellations."""
        while self.is_running:
            try:
                for order_id, order_data in list(self.active_orders.items()):
                    # Check order status with Kalshi
                    status = await self.kalshi.get_order_status(order_id)
                    
                    if status.get("status") == "FILLED":
                        # Order filled - update position
                        await self._update_position(order_id, order_data, status)
                        del self.active_orders[order_id]
                        
                    elif status.get("status") == "CANCELLED":
                        # Order cancelled
                        logger.info(f"Order {order_id} cancelled")
                        del self.active_orders[order_id]
                        
                    elif (datetime.now() - order_data["created_at"]).seconds > 300:
                        # Cancel stale orders (5 minutes)
                        logger.warning(f"Cancelling stale order: {order_id}")
                        await self.kalshi.cancel_order(order_id)
                        del self.active_orders[order_id]
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Order monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def _stop_loss_monitoring_loop(self):
        """Monitor positions for stop-loss triggers."""
        while self.is_running:
            try:
                for market_ticker, position in list(self.positions.items()):
                    # Get current market price
                    market = await self.kalshi.get_market(market_ticker)
                    current_price = market.get("yes_price", 0.5)
                    
                    # Check stop-loss
                    if position["side"] == "YES":
                        if current_price <= position["stop_loss"]:
                            logger.warning(f"Stop-loss triggered for {market_ticker}")
                            await self._execute_stop_loss(market_ticker, position)
                    else:  # NO position
                        if current_price >= (1 - position["stop_loss"]):
                            logger.warning(f"Stop-loss triggered for {market_ticker}")
                            await self._execute_stop_loss(market_ticker, position)
                    
                    # Update trailing stop if profitable
                    if position["side"] == "YES":
                        profit_pct = (current_price - position["entry_price"]) / position["entry_price"]
                    else:
                        profit_pct = ((1 - current_price) - position["entry_price"]) / position["entry_price"]
                    
                    if profit_pct > 0.1:  # 10% profit
                        # Update trailing stop
                        new_stop = self.stop_loss_calculator.calculate_trailing_stop(
                            entry_price=position["entry_price"],
                            current_price=current_price,
                            current_stop=position["stop_loss"],
                            profit_percentage=profit_pct
                        )
                        
                        if new_stop["new_stop"] > position["stop_loss"]:
                            position["stop_loss"] = new_stop["new_stop"]
                            logger.info(f"Updated trailing stop for {market_ticker}: ${new_stop['new_stop']:.2f}")
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Stop-loss monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _update_position(self, order_id: str, order_data: Dict, fill_data: Dict):
        """Update position after order fill."""
        market_ticker = order_data["market_ticker"]
        
        if market_ticker not in self.positions:
            self.positions[market_ticker] = {
                "side": "YES" if "YES" in order_data["action"] else "NO",
                "contracts": 0,
                "entry_price": 0,
                "stop_loss": 0
            }
        
        position = self.positions[market_ticker]
        
        # Update position
        if "BUY" in order_data["action"]:
            # Adding to position
            total_value = (position["contracts"] * position["entry_price"]) + \
                         (order_data["contracts"] * fill_data["fill_price"])
            position["contracts"] += order_data["contracts"]
            position["entry_price"] = total_value / position["contracts"] if position["contracts"] > 0 else 0
        else:
            # Reducing position
            position["contracts"] -= order_data["contracts"]
            
        # Remove position if closed
        if position["contracts"] == 0:
            del self.positions[market_ticker]
        else:
            # Update stop-loss
            position["stop_loss"] = order_data.get("stop_loss", position["stop_loss"])
        
        logger.info(f"Position updated: {market_ticker} - {position}")
    
    async def _execute_stop_loss(self, market_ticker: str, position: Dict):
        """Execute stop-loss order."""
        try:
            # Place market order to close position
            action = "SELL_YES" if position["side"] == "YES" else "SELL_NO"
            
            order = await self._place_sell_order(
                market_ticker=market_ticker,
                side=position["side"],
                contracts=position["contracts"],
                limit_price=0  # Market order
            )
            
            if order.get("success"):
                logger.info(f"Stop-loss executed: {market_ticker} - {position['contracts']} contracts")
                
                # Remove position
                del self.positions[market_ticker]
                
                # Notify risk manager
                if self.message_handler:
                    await self.message_handler('stop_loss_triggered', {
                        "market_ticker": market_ticker,
                        "position": position,
                        "timestamp": datetime.now().isoformat()
                    })
            
        except Exception as e:
            logger.error(f"Stop-loss execution error: {e}")
    
    async def _load_positions(self):
        """Load existing positions from database."""
        # Implementation would load from database
        # For now, start with empty positions
        self.positions = {}
        logger.info("Positions loaded")
    
    async def _handle_emergency_stop(self, data: Dict[str, Any]):
        """Handle emergency stop signal."""
        logger.warning(f"Emergency stop received: {data}")
        
        # Cancel all active orders
        for order_id in list(self.active_orders.keys()):
            try:
                await self.kalshi.cancel_order(order_id)
            except:
                pass
        
        # Close all positions
        for market_ticker, position in list(self.positions.items()):
            try:
                await self._execute_stop_loss(market_ticker, position)
            except:
                pass
        
        await self.stop()
    
    async def stop(self):
        """Stop the Trade Executor agent."""
        self.is_running = False
        
        # Cancel tasks
        for task in self.tasks:
            task.cancel()
        
        # Close connections
        # Cleanup handled by Agentuity
        
        logger.info("Trade Executor Agent stopped")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        return {
            "is_running": self.is_running,
            "active_orders": len(self.active_orders),
            "open_positions": len(self.positions),
            "total_exposure": sum(
                p["contracts"] * p["entry_price"] 
                for p in self.positions.values()
            ),
            "bankroll": self.bankroll,
            "positions": self.positions
        }


# Example usage
async def main():
    """Example of running the Trade Executor agent."""
    
    # Initialize agent
    agent = TradeExecutorAgent()
    
    # Start agent
    await agent.start()
    
    # Example trade signal
    await agent.execute_trade(
        market_ticker="SUPERBOWL-2025",
        action="BUY_YES",
        probability=0.65,
        confidence=0.8,
        market_data={
            "yes_bid": 0.58,
            "yes_ask": 0.60,
            "volume_24h": 150000,
            "hours_to_close": 48
        }
    )
    
    # Run for a while
    await asyncio.sleep(60)
    
    # Get status
    status = await agent.get_status()
    print(f"Agent status: {status}")
    
    # Stop agent
    await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())