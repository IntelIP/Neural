from agentuity import AgentRequest, AgentResponse, AgentContext
import asyncio
import json

from agent_consumers.MarketEngineer import market_engineer_agent
from data_pipeline.window_aggregator import WindowAggregator
from data_pipeline.state_manager import get_state_manager

def welcome():
    return {
        "welcome": "🧠 Market Engineer Agent - I analyze sentiment and identify trading opportunities using E2B sandboxes and Gemini 2.5 Flash.",
        "prompts": [
            {
                "data": "Analyze sentiment for these tweets about the Chiefs game",
                "contentType": "text/plain"
            },
            {
                "data": "Calculate market opportunity score for SUPERBOWL-2025",
                "contentType": "text/plain"
            },
            {
                "data": "Run sentiment velocity analysis",
                "contentType": "text/plain"
            }
        ]
    }

async def run(request: AgentRequest, response: AgentResponse, context: AgentContext):
    """
    Market Engineer Agent - Analyzes sentiment and identifies opportunities.
    
    Uses E2B sandboxes for advanced sentiment analysis and market scoring.
    """
    try:
        # Initialize window aggregator and state manager
        window_aggregator = WindowAggregator()
        state_manager = get_state_manager()
        
        # Get request data
        data = await request.data.json() if request.data.contentType == "application/json" else None
        prompt = await request.data.text() if not data else json.dumps(data)
        
        context.logger.info(f"[MarketEngineer] Processing: {prompt[:100]}")
        
        # Check for specific commands
        if isinstance(data, dict):
            command = data.get("command", "")
            
            if command == "analyze_sentiment":
                # Process tweet batch for sentiment
                tweets = data.get("tweets", [])
                market_ticker = data.get("market_ticker")
                
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: market_engineer_agent.analyze_sentiment_batch(tweets)
                )
                
                # Add sentiment to window aggregator for trend analysis
                sentiment_score = result.get("aggregate_sentiment", 0)
                await window_aggregator.add_event(
                    value=sentiment_score,
                    metadata={
                        "type": "sentiment",
                        "market": market_ticker,
                        "tweet_count": len(tweets)
                    }
                )
                
                # Get velocity metrics to detect sentiment shifts
                velocity_metrics = window_aggregator.calculate_velocity("5m")
                
                # Check for anomalies in sentiment
                is_anomaly = window_aggregator.detect_anomaly("5m")
                
                # Enhance result with window analysis
                result["window_analysis"] = {
                    "current_5m": window_aggregator.get_aggregates("5m"),
                    "current_15m": window_aggregator.get_aggregates("15m"),
                    "velocity": velocity_metrics,
                    "anomaly_detected": is_anomaly
                }
                
                # Save window state to KV storage
                await state_manager.save_state(
                    context,
                    "windows",
                    f"sentiment_{market_ticker}",
                    {
                        "last_sentiment": sentiment_score,
                        "velocity": velocity_metrics,
                        "aggregates_5m": window_aggregator.get_aggregates("5m"),
                        "timestamp": asyncio.get_event_loop().time()
                    },
                    ttl=3600  # 1 hour TTL
                )
                
                # Check for trading opportunity with enhanced metrics
                opportunity_score = result.get("opportunity_score", 0)
                
                # Boost opportunity score if we detect rapid sentiment shift
                if velocity_metrics.get("velocity", 0) > 0.1:
                    opportunity_score = min(1.0, opportunity_score * 1.2)
                    context.logger.info(f"Sentiment velocity boost: {velocity_metrics['velocity']:.3f}")
                
                if opportunity_score > 0.7 or is_anomaly:
                    # Handoff to Risk Manager for signal evaluation
                    return response.handoff(
                        {"name": "RiskManager"},
                        {
                            "command": "evaluate_signal",
                            "market_ticker": market_ticker,
                            "sentiment_score": sentiment_score,
                            "opportunity_score": opportunity_score,
                            "window_analysis": result["window_analysis"],
                            "analysis": result.get("analysis")
                        },
                        "application/json",
                        {"source": "MarketEngineer"}
                    )
                
                return response.json(result)
            
            elif command == "market_data":
                # Process market data update
                market_data = data.get("data", {})
                
                # Update internal market context
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None,
                    lambda: market_engineer_agent.update_market_context(market_data)
                )
                
                # Check for divergence opportunities
                divergence = await loop.run_in_executor(
                    None,
                    lambda: market_engineer_agent.check_sentiment_divergence(
                        market_data.get("market_ticker")
                    )
                )
                
                if divergence and divergence.get("is_significant"):
                    # Generate trading signal
                    signal = {
                        "market_ticker": market_data.get("market_ticker"),
                        "action": divergence.get("action"),
                        "probability": divergence.get("probability"),
                        "confidence": divergence.get("confidence"),
                        "analysis": divergence.get("analysis")
                    }
                    
                    # Handoff to Risk Manager for evaluation
                    return response.handoff(
                        {"name": "RiskManager"},
                        {"command": "evaluate_signal", **signal},
                        "application/json",
                        {"source": "MarketEngineer", "divergence_detected": True}
                    )
                
                return response.json({
                    "status": "processed",
                    "divergence_found": False
                })
            
            elif command == "calculate_opportunity":
                # Calculate opportunity score using E2B
                market_ticker = data.get("market_ticker")
                
                loop = asyncio.get_running_loop()
                opportunity = await loop.run_in_executor(
                    None,
                    lambda: market_engineer_agent.calculate_opportunity_score(market_ticker)
                )
                
                return response.json(opportunity)
        
        # Default: Use status or basic info
        if "status" in prompt.lower():
            status_info = {
                "is_running": market_engineer_agent.is_running,
                "tracked_markets": len(market_engineer_agent.market_sentiments),
                "message": "Market Engineer ready for sentiment analysis"
            }
            return response.json(status_info)
        else:
            # Return a helpful message
            return response.text(
                "🔍 Market Engineer Agent\n\n"
                "I analyze sentiment and identify market inefficiencies.\n"
                "Try commands like:\n"
                "• 'status' - Get current status\n"
                "• Send market data via JSON with 'command': 'analyze_sentiment'\n"
                "• Calculate divergence with 'command': 'calculate_divergence'"
            )
        
    except Exception as exc:
        context.logger.error(f"[MarketEngineer] Error: {exc}", exc_info=True)
        return response.json({
            "status": "error",
            "message": str(exc)
        })