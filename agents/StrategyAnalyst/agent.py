from agentuity import AgentRequest, AgentResponse, AgentContext
import asyncio
import json

from agent_consumers.StrategyAnalyst import strategy_analyst_agent

def welcome():
    return {
        "welcome": "📈 Strategy Analyst Agent - I analyze market opportunities and generate trading signals using multi-source divergence analysis.",
        "prompts": [
            {
                "data": "Analyze opportunity for SUPERBOWL-2025",
                "contentType": "text/plain"
            },
            {
                "data": "Evaluate position for NFL-WINNER-WEEK15",
                "contentType": "text/plain"
            },
            {
                "data": "Show current trading signals",
                "contentType": "text/plain"
            }
        ]
    }

async def run(request: AgentRequest, response: AgentResponse, context: AgentContext):
    """
    Strategy Analyst Agent - Analyzes opportunities and generates trading signals.
    
    Wraps the Agno-based strategy analyst with Agentuity handler for deployment.
    """
    try:
        # Get request data
        data = await request.data.json() if request.data.contentType == "application/json" else None
        prompt = await request.data.text() if not data else json.dumps(data)
        
        context.logger.info(f"[StrategyAnalyst] Processing: {prompt[:100]}")
        
        # Check for specific commands
        if isinstance(data, dict):
            command = data.get("command", "")
            
            if command == "analyze_opportunity":
                # Analyze market opportunity from DataCoordinator
                context.logger.info(f"Analyzing opportunity: {data.get('market_ticker')}")
                
                loop = asyncio.get_running_loop()
                signal = await loop.run_in_executor(
                    None,
                    lambda: strategy_analyst_agent.analyze_opportunity(data)
                )
                
                if signal:
                    context.logger.info(f"Signal generated: {signal.signal_type.value}")
                    
                    if signal.signal_type.value not in ["hold", "close"]:
                        # Handoff to RiskManager for evaluation
                        signal_data = {
                            "command": "evaluate_signal",
                            "market_ticker": signal.market_ticker,
                            "signal_type": signal.signal_type.value,
                            "confidence": signal.confidence,
                            "probability": signal.probability,
                            "reasoning": signal.reasoning,
                            "risk_level": signal.risk_level,
                            "suggested_size_pct": signal.suggested_size_pct,
                            "stop_loss": signal.stop_loss,
                            "take_profit": signal.take_profit
                        }
                        
                        context.logger.info("Handing off signal to RiskManager")
                        return response.handoff(
                            {"name": "RiskManager"},
                            signal_data,
                            "application/json",
                            {"source": "StrategyAnalyst"}
                        )
                    
                    return response.json({
                        "status": "no_action",
                        "signal": signal.signal_type.value,
                        "reasoning": signal.reasoning
                    })
                else:
                    return response.json({
                        "status": "no_signal",
                        "message": "No trading opportunity identified"
                    })
            
            elif command == "evaluate_position":
                # Evaluate existing position
                market_ticker = data.get("market_ticker")
                
                loop = asyncio.get_running_loop()
                exit_signal = await loop.run_in_executor(
                    None,
                    lambda: strategy_analyst_agent.evaluate_position(
                        market_ticker,
                        data.get("current_data", {})
                    )
                )
                
                if exit_signal:
                    # Create exit signal
                    signal_data = {
                        "command": "close_position",
                        "market_ticker": market_ticker,
                        "reason": exit_signal.value
                    }
                    
                    # Handoff to TradeExecutor for closing
                    return response.handoff(
                        {"name": "TradeExecutor"},
                        signal_data,
                        "application/json",
                        {"source": "StrategyAnalyst"}
                    )
                
                return response.json({
                    "status": "hold_position",
                    "market_ticker": market_ticker
                })
            
            elif command == "get_status":
                # Get agent status
                status = strategy_analyst_agent.get_status()
                return response.json(status)
        
        # Default: Return status or helpful message
        if "status" in prompt.lower():
            status = strategy_analyst_agent.get_status()
            return response.json(status)
        else:
            # Return a helpful message about available commands
            return response.text(
                "📈 Strategy Analyst Agent\n\n"
                "I analyze market opportunities and generate trading signals.\n"
                "Try commands like:\n"
                "• 'status' - Get current status\n"
                "• Send opportunities via JSON with 'command': 'analyze_opportunity'\n"
                "• Evaluate positions with 'command': 'evaluate_position'"
            )
        
    except Exception as exc:
        context.logger.error(f"[StrategyAnalyst] Error: {exc}", exc_info=True)
        return response.json({
            "status": "error",
            "message": str(exc)
        })