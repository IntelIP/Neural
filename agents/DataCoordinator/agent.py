from agentuity import AgentRequest, AgentResponse, AgentContext
import asyncio
import json

from agent_consumers.DataCoordinator import data_coordinator_agent

def welcome():
    return {
        "welcome": "📊 Data Coordinator Agent - I manage all streaming data from Kalshi, ESPN, and Twitter via the unified StreamManager.",
        "prompts": [
            {
                "data": "Track market SUPERBOWL-2025 with Chiefs vs Bills",
                "contentType": "text/plain"
            },
            {
                "data": "Get market summary for SUPERBOWL-2025",
                "contentType": "text/plain"
            },
            {
                "data": "Show current status",
                "contentType": "text/plain"
            }
        ]
    }

async def run(request: AgentRequest, response: AgentResponse, context: AgentContext):
    """
    Data Coordinator Agent - Manages all data streams via StreamManager.
    
    Uses Agentuity KV storage for persistence instead of database.
    """
    try:
        # Get request data
        data = await request.data.json() if request.data.contentType == "application/json" else None
        prompt = await request.data.text() if not data else json.dumps(data)
        
        context.logger.info(f"[DataCoordinator] Processing: {prompt[:100]}")
        
        # Load agent state from KV storage
        state_result = await context.kv.get("agent_state", "data_coordinator")
        if state_result.exists:
            agent_state = await state_result.data.json()
            data_coordinator_agent.tracked_markets = agent_state.get("tracked_markets", {})
            data_coordinator_agent.events_received = agent_state.get("events_received", 0)
            data_coordinator_agent.events_routed = agent_state.get("events_routed", 0)
        
        # Start agent if not running
        if not data_coordinator_agent.is_running:
            context.logger.info("Starting Data Coordinator with StreamManager...")
            await data_coordinator_agent.start(agent_context=context)
        
        # Check for specific commands
        if isinstance(data, dict):
            command = data.get("command", "")
            
            if command == "track_market":
                # Track a new market
                market_ticker = data.get("market_ticker")
                game_id = data.get("game_id")
                home_team = data.get("home_team")
                away_team = data.get("away_team")
                sport = data.get("sport", "nfl")
                
                await data_coordinator_agent.track_market(
                    market_ticker=market_ticker,
                    game_id=game_id,
                    home_team=home_team,
                    away_team=away_team,
                    sport=sport
                )
                
                return response.json({
                    "status": "tracking",
                    "market": market_ticker,
                    "game": f"{home_team} vs {away_team}"
                })
            
            elif command == "get_summary":
                # Get market summary
                market_ticker = data.get("market_ticker")
                summary = await data_coordinator_agent.get_market_summary(market_ticker)
                
                return response.json(summary)
            
            elif command == "status":
                # Get agent status
                status = await data_coordinator_agent.get_status()
                return response.json(status)
            
            elif command == "buffered_event":
                # Handle buffered event from EventBuffer
                event = data.get("event", {})
                event_type = event.get("type")
                priority = event.get("priority", "NORMAL")
                event_data = event.get("data", {})
                
                context.logger.debug(f"Buffered event: {event_type} (Priority: {priority})")
                
                # Update statistics
                data_coordinator_agent.events_received += 1
                
                # Route based on priority and type
                if priority == "CRITICAL":
                    # Immediate handoff for critical events
                    if event_type in ["DIVERGENCE_DETECTED", "MARKET_OPPORTUNITY"]:
                        await context.handoff("StrategyAnalyst", json.dumps({
                            "command": "analyze_opportunity",
                            "event": event_data
                        }))
                        data_coordinator_agent.events_routed += 1
                    elif event_type in ["INJURY_ALERT", "RISK_ALERT", "ANOMALY_DETECTED"]:
                        await context.handoff("RiskManager", json.dumps({
                            "command": "assess_risk",
                            "event": event_data
                        }))
                        data_coordinator_agent.events_routed += 1
                
                # Save state periodically
                if data_coordinator_agent.events_received % 100 == 0:
                    await context.kv.set("agent_state", "data_coordinator", {
                        "tracked_markets": data_coordinator_agent.tracked_markets,
                        "events_received": data_coordinator_agent.events_received,
                        "events_routed": data_coordinator_agent.events_routed,
                        "is_running": data_coordinator_agent.is_running
                    })
                
                return response.json({
                    "status": "processed",
                    "event_type": event_type,
                    "priority": priority
                })
            
            elif command == "stream_event":
                # Legacy support for direct stream events
                event_type = data.get("type")
                
                context.logger.debug(f"Direct stream event: {event_type}")
                
                # Route to appropriate agent based on event type and impact
                if data.get("requires_action"):
                    # High-priority event
                    if event_type in ["divergence_detected", "market_opportunity"]:
                        # Handoff to Strategy Analyst
                        return response.handoff(
                            {"name": "StrategyAnalyst"},
                            {"command": "analyze_opportunity", **data},
                            "application/json",
                            {"source": "DataCoordinator"}
                        )
                    
                    elif event_type in ["injury_alert", "sentiment_shift"]:
                        # Handoff to Market Engineer
                        return response.handoff(
                            {"name": "MarketEngineer"},
                            {"command": "high_impact_event", **data},
                            "application/json",
                            {"source": "DataCoordinator"}
                        )
                    
                    elif event_type == "risk_alert":
                        # Handoff to Risk Manager
                        return response.handoff(
                            {"name": "RiskManager"},
                            {"command": "risk_alert", **data},
                            "application/json",
                            {"source": "DataCoordinator"}
                        )
                
                # Log all events
                data_coordinator_agent.events_received += 1
                
                return response.json({"status": "processed"})
        
        # Default: Parse natural language commands
        if "track" in prompt.lower() and "market" in prompt.lower():
            # Extract market info from prompt
            # Simple parsing - could use LLM for better extraction
            parts = prompt.split()
            market_ticker = None
            teams = []
            
            for i, part in enumerate(parts):
                if "BOWL" in part.upper() or "-20" in part:
                    market_ticker = part.upper()
                elif "vs" in part.lower() and i > 0 and i < len(parts) - 1:
                    teams = [parts[i-1], parts[i+1]]
            
            if market_ticker:
                home_team = teams[0] if len(teams) > 0 else None
                away_team = teams[1] if len(teams) > 1 else None
                
                await data_coordinator_agent.track_market(
                    market_ticker=market_ticker,
                    home_team=home_team,
                    away_team=away_team
                )
                
                return response.text(f"✅ Now tracking {market_ticker}" + 
                                   (f" ({home_team} vs {away_team})" if teams else ""))
        
        elif "summary" in prompt.lower():
            # Get summary for mentioned market
            for market in data_coordinator_agent.tracked_markets.keys():
                if market in prompt.upper():
                    summary = await data_coordinator_agent.get_market_summary(market)
                    
                    # Format summary
                    output = f"📊 Market Summary: {market}\n\n"
                    output += f"💰 Price: Yes ${summary['price']['yes']:.2f} | No ${summary['price']['no']:.2f}\n"
                    output += f"🏈 Score: {summary['game']['home_team']} {summary['game']['home_score']} - "
                    output += f"{summary['game']['away_team']} {summary['game']['away_score']}\n"
                    output += f"💭 Sentiment: {summary['sentiment']['score']:+.2f} "
                    output += f"({summary['sentiment']['volume']} tweets)\n"
                    output += f"🎯 Opportunity: {summary['analysis']['opportunity_score']:.2f} "
                    output += f"(Risk: {summary['analysis']['risk_level']})\n"
                    
                    return response.text(output)
            
            return response.text("No tracked markets found. Track a market first.")
        
        elif "status" in prompt.lower():
            status = await data_coordinator_agent.get_status()
            
            output = "📊 Data Coordinator Status\n\n"
            output += f"✅ Running: {status['is_running']}\n"
            output += f"📈 Markets: {', '.join(status['tracked_markets']) or 'None'}\n"
            output += f"📨 Events: {status['events_received']} received, "
            output += f"{status['events_routed']} routed ({status['routing_rate']})\n"
            output += f"🔌 Sources: Kalshi={status['stream_manager']['data_sources']['kalshi']}, "
            output += f"ESPN={status['stream_manager']['data_sources']['espn']}, "
            output += f"Twitter={status['stream_manager']['data_sources']['twitter']}"
            
            return response.text(output)
        
        else:
            # Return a helpful message about available commands
            return response.text(
                "📊 Data Coordinator Agent\n\n"
                "I coordinate data collection from Kalshi, ESPN, and Twitter.\n"
                "Try commands like:\n"
                "• 'status' - Get system status\n"
                "• 'summary' - Get market summary\n" 
                "• 'track [market]' - Track a specific market\n"
                "• Use JSON commands for detailed control"
            )
        
    except Exception as exc:
        context.logger.error(f"[DataCoordinator] Error: {exc}", exc_info=True)
        return response.json({
            "status": "error",
            "message": str(exc)
        })