from agentuity import AgentRequest, AgentResponse, AgentContext
import asyncio
import json

from agent_consumers.RiskManager import risk_manager_agent

def welcome():
    return {
        "welcome": "🛡️ Risk Manager Agent - I monitor portfolio risk, calculate VaR using E2B, and enforce position limits.",
        "prompts": [
            {
                "data": "Calculate portfolio Value at Risk",
                "contentType": "text/plain"
            },
            {
                "data": "Check correlation risk across positions",
                "contentType": "text/plain"
            },
            {
                "data": "Get current risk metrics",
                "contentType": "text/plain"
            }
        ]
    }

async def run(request: AgentRequest, response: AgentResponse, context: AgentContext):
    """
    Risk Manager Agent - Portfolio risk management with E2B analytics.
    
    Monitors exposure, calculates VaR, and triggers emergency stops.
    """
    try:
        # Get request data
        data = await request.data.json() if request.data.contentType == "application/json" else None
        prompt = await request.data.text() if not data else json.dumps(data)
        
        context.logger.info(f"[RiskManager] Processing: {prompt[:100]}")
        
        # Check for specific commands
        if isinstance(data, dict):
            command = data.get("command", "")
            
            if command == "evaluate_signal":
                # Evaluate trading signal for risk approval
                market_ticker = data.get("market_ticker")
                action = data.get("action")
                
                context.logger.info(f"Evaluating signal: {market_ticker} - {action}")
                
                loop = asyncio.get_running_loop()
                risk_check = await loop.run_in_executor(
                    None,
                    lambda: risk_manager_agent.evaluate_signal_risk(data)
                )
                
                if not risk_check.get("approved"):
                    context.logger.warning(f"Signal rejected: {risk_check.get('reason')}")
                    return response.json(risk_check)
                
                # Calculate position sizing with Kelly
                position_params = await loop.run_in_executor(
                    None,
                    lambda: risk_manager_agent.calculate_position_parameters(data)
                )
                
                # Add risk parameters to signal
                data["risk_params"] = position_params
                
                context.logger.info(f"Signal approved: {market_ticker}, handing off to TradeExecutor")
                
                # Handoff approved signal to Trade Executor
                return response.handoff(
                    {"name": "TradeExecutor"},
                    {"command": "execute_trade", **data},
                    "application/json",
                    {"source": "RiskManager", "approved": True}
                )
            
            elif command == "position_opened":
                # Track new position
                
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None,
                    lambda: risk_manager_agent.track_position(data)
                )
                
                # Recalculate portfolio metrics
                metrics = await loop.run_in_executor(
                    None,
                    lambda: risk_manager_agent.update_portfolio_metrics()
                )
                
                # Check if risk limits exceeded
                if metrics.get("risk_level") in ["high", "critical"]:
                    # Trigger risk alert - returns handoff to TradeExecutor if critical
                    alert_data = {
                        "alert_type": metrics.get("alert_type", "high_risk"),
                        "severity": metrics.get("risk_level"),
                        "metrics": metrics,
                        "timestamp": asyncio.get_event_loop().time()
                    }
                    
                    # Store alert for other agents to access
                    await context.kv.set("alerts", "latest_risk_alert", alert_data)
                    
                    # Emergency stop if critical - handoff to TradeExecutor
                    if metrics.get("risk_level") == "critical":
                        return response.handoff(
                            {"name": "TradeExecutor"},
                            {"command": "emergency_stop", "reason": alert_data["alert_type"]},
                            "application/json",
                            {"source": "RiskManager", "alert": alert_data}
                        )
                    
                    # For non-critical, just log and continue
                    context.logger.warning(f"RISK ALERT: {alert_data['alert_type']}")
                
                return response.json({
                    "status": "tracked",
                    "portfolio_metrics": metrics
                })
            
            elif command == "calculate_var":
                # Calculate Value at Risk using E2B
                loop = asyncio.get_running_loop()
                var_result = await loop.run_in_executor(
                    None,
                    lambda: risk_manager_agent.calculate_portfolio_var()
                )
                
                context.logger.info(f"VaR calculated: ${abs(var_result.get('value_at_risk', 0)):.2f}")
                
                return response.json(var_result)
            
            elif command == "check_correlation":
                # Check correlation risk
                loop = asyncio.get_running_loop()
                correlation = await loop.run_in_executor(
                    None,
                    lambda: risk_manager_agent.check_correlation_risk()
                )
                
                return response.json(correlation)
            
            elif command == "risk_metrics":
                # Get current risk metrics
                loop = asyncio.get_running_loop()
                metrics = await loop.run_in_executor(
                    None,
                    lambda: risk_manager_agent.get_risk_metrics()
                )
                
                return response.json(metrics)
        
        # Default: Use LLM for general risk queries
        if "var" in prompt.lower() or "value at risk" in prompt.lower():
            # Calculate VaR
            var_result = await risk_manager_agent.calculate_portfolio_var()
            
            output = f"📊 Value at Risk Analysis\n\n"
            output += f"VaR (95% confidence): ${abs(var_result.get('value_at_risk', 0)):.2f}\n"
            output += f"Expected Shortfall: ${abs(var_result.get('expected_shortfall', 0)):.2f}\n"
            output += f"Risk Level: {var_result.get('risk_level', 'unknown')}\n"
            output += f"Total Exposure: ${var_result.get('total_exposure', 0):.2f}"
            
            return response.text(output)
        
        elif "correlation" in prompt.lower():
            # Check correlations
            correlation = await risk_manager_agent.check_correlation_risk()
            
            output = f"🔗 Correlation Risk Analysis\n\n"
            output += f"Risk Level: {correlation.get('correlation_risk', 'unknown')}\n"
            output += f"Average Correlation: {correlation.get('average_correlation', 0):.2f}\n"
            
            if correlation.get('high_correlation_pairs'):
                output += "\nHighly Correlated Pairs:\n"
                for pair in correlation['high_correlation_pairs'][:3]:
                    output += f"• {pair['pair'][0]} ↔ {pair['pair'][1]}: {pair['correlation']:.2f}\n"
            
            return response.text(output)
        
        elif "status" in prompt.lower() or "metrics" in prompt.lower():
            # Get status
            status = await risk_manager_agent.get_status()
            
            output = f"🛡️ Risk Manager Status\n\n"
            output += f"✅ Running: {status['is_running']}\n"
            output += f"📈 Positions: {status['num_positions']}\n"
            output += f"💰 Total Exposure: ${status['total_exposure']:.2f}\n"
            output += f"📊 Daily P&L: ${status['daily_pnl']:.2f}\n"
            output += f"⚠️ Max Risk: {status['risk_limits']['max_portfolio_risk']:.0%}"
            
            return response.text(output)
        
        else:
            # Use LLM for general queries
            output = await risk_manager_agent.llm_client.complete(
                prompt,
                temperature=0.1,
                system_prompt="You are a risk management expert. Provide concise risk analysis and recommendations."
            )
            
            return response.text(output)
        
    except Exception as exc:
        context.logger.error(f"[RiskManager] Error: {exc}", exc_info=True)
        return response.json({
            "status": "error",
            "message": str(exc)
        })

