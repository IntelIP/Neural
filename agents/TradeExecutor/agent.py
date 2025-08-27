from agentuity import AgentRequest, AgentResponse, AgentContext
import asyncio
import json

from agent_consumers.TradeExecutor import trade_executor_agent

def welcome():
    return {
        "welcome": "💰 Trade Executor Agent - I execute trades on Kalshi with Kelly Criterion position sizing and dynamic stop-losses.",
        "prompts": [
            {
                "data": "Execute BUY_YES on SUPERBOWL-2025 with 0.75 probability",
                "contentType": "text/plain"
            },
            {
                "data": "Close position on market NFL-WINNER-WEEK15",
                "contentType": "text/plain"
            },
            {
                "data": "Get current portfolio status",
                "contentType": "text/plain"
            }
        ]
    }

async def run(request: AgentRequest, response: AgentResponse, context: AgentContext):
    """
    Trade Executor Agent - Executes trades on Kalshi markets.
    
    Applies Kelly Criterion position sizing and manages order lifecycle.
    """
    try:
        # Get request data
        data = await request.data.json() if request.data.contentType == "application/json" else None
        prompt = await request.data.text() if not data else json.dumps(data)
        
        context.logger.info(f"[TradeExecutor] Processing: {prompt[:100]}")
        
        # Check for specific commands
        if isinstance(data, dict):
            command = data.get("command", "")
            
            if command == "execute_trade":
                # Execute approved trading signal
                market_ticker = data.get("market_ticker")
                action = data.get("action")  # BUY_YES, BUY_NO, SELL_YES, SELL_NO
                probability = data.get("probability")
                confidence = data.get("confidence")
                risk_params = data.get("risk_params", {})
                
                context.logger.info(f"Executing trade: {market_ticker} - {action}")
                
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: trade_executor_agent.execute_trade(
                        market_ticker=market_ticker,
                        action=action,
                        probability=probability,
                        confidence=confidence,
                        market_data=data.get("market_data", {}),
                        risk_params=risk_params
                    )
                )
                
                if result.get("status") == "success":
                    # Store result for tracking
                    await context.kv.set("trades", f"result_{market_ticker}", result)
                    
                    # Handoff to Risk Manager for position tracking
                    position_data = {
                        "command": "position_opened",
                        "order_id": result.get("order_id"),
                        "market_ticker": market_ticker,
                        "position_size": result.get("position_size"),
                        "kelly_fraction": result.get("kelly_fraction"),
                        "stop_loss": result.get("stop_loss")
                    }
                    
                    context.logger.info(f"Trade executed: {result.get('order_id')}, handing off to RiskManager")
                    
                    return response.handoff(
                        {"name": "RiskManager"},
                        position_data,
                        "application/json",
                        {"source": "TradeExecutor", "trade_result": result}
                    )
                else:
                    context.logger.error(f"Trade failed: {result.get('reason')}")
                    return response.json(result)
            
            elif command == "close_position":
                # Close existing position
                market_ticker = data.get("market_ticker")
                reason = data.get("reason", "manual")
                
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: trade_executor_agent.close_position(
                        market_ticker=market_ticker,
                        reason=reason
                    )
                )
                
                if result.get("status") == "success":
                    # Store result for tracking
                    await context.kv.set("trades", f"closed_{market_ticker}", result)
                    
                    # Handoff to Risk Manager for portfolio update
                    return response.handoff(
                        {"name": "RiskManager"},
                        {
                            "command": "position_closed",
                            "market_ticker": market_ticker,
                            "pnl": result.get("pnl"),
                            "reason": reason
                        },
                        "application/json",
                        {"source": "TradeExecutor"}
                    )
                
                return response.json(result)
            
            elif command == "emergency_stop":
                # Emergency stop - close all positions
                reason = data.get("reason")
                
                context.logger.warning(f"EMERGENCY STOP: {reason}")
                
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: trade_executor_agent.emergency_stop(reason)
                )
                
                return response.json({
                    "status": "stopped",
                    "positions_closed": result.get("positions_closed"),
                    "reason": reason
                })
            
            elif command == "portfolio_status":
                # Get current portfolio status
                loop = asyncio.get_running_loop()
                status = await loop.run_in_executor(
                    None,
                    lambda: trade_executor_agent.get_portfolio_status()
                )
                
                return response.json(status)
        
        # Default: Return status or helpful message
        if "status" in prompt.lower():
            # Get portfolio status
            loop = asyncio.get_running_loop()
            status = await loop.run_in_executor(
                None,
                lambda: trade_executor_agent.get_portfolio_status()
            )
            return response.json(status)
        else:
            # Return a helpful message about available commands
            return response.text(
                "💰 Trade Executor Agent\n\n"
                "I execute trades on Kalshi markets with Kelly Criterion position sizing.\n"
                "Try commands like:\n"
                "• 'status' - Get portfolio status\n"
                "• Send trades via JSON with 'command': 'execute_trade'\n"
                "• Close positions with 'command': 'close_position'"
            )
        
    except Exception as exc:
        context.logger.error(f"[TradeExecutor] Error: {exc}", exc_info=True)
        return response.json({
            "status": "error",
            "message": str(exc)
        })