"""
Strategy Analyst Agent
Analyzes market opportunities and generates trading signals
Uses LLM for complex pattern recognition and strategy decisions
"""

import os
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from trading_logic.llm_client import get_llm_client

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Trading signal types"""
    BUY_YES = "buy_yes"
    BUY_NO = "buy_no"
    SELL_YES = "sell_yes"
    SELL_NO = "sell_no"
    HOLD = "hold"
    CLOSE = "close"


@dataclass
class TradingSignal:
    """Trading signal structure"""
    market_ticker: str
    signal_type: SignalType
    confidence: float  # 0-1 scale
    probability: float  # Market probability estimate
    reasoning: str
    risk_level: str  # low, medium, high
    suggested_size_pct: float  # Percentage of capital
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class StrategyAnalystAgent:
    """
    Strategy Analyst Agent - Analyzes opportunities and generates signals
    
    Responsibilities:
    - Analyze market divergences and opportunities
    - Generate trading signals with confidence levels
    - Evaluate sentiment vs price discrepancies
    - Determine optimal entry/exit points
    
    Uses LLM for complex pattern recognition
    """
    
    def __init__(self):
        """Initialize Strategy Analyst Agent"""
        # Initialize LLM client
        self.llm_client = get_llm_client()
        
        # Signal history
        self.signals: List[TradingSignal] = []
        self.active_positions: Dict[str, TradingSignal] = {}
        
        # Message handler (set by Agentuity)
        self.message_handler = None
        
        # Statistics
        self.opportunities_analyzed = 0
        self.signals_generated = 0
    
    async def analyze_opportunity(self, opportunity_data: Dict[str, Any]) -> Optional[TradingSignal]:
        """
        Analyze a market opportunity and generate trading signal
        
        Args:
            opportunity_data: Opportunity data from DataCoordinator
            
        Returns:
            Trading signal if opportunity is valid
        """
        self.opportunities_analyzed += 1
        
        market_ticker = opportunity_data.get("market_ticker")
        
        # Use LLM to analyze the opportunity
        analysis_prompt = f"""
        Analyze this trading opportunity:
        
        Market: {market_ticker}
        
        Price Data:
        - Market Price: {opportunity_data.get('market_price', 'N/A')}
        - Sentiment Implied: {opportunity_data.get('sentiment_implied', 'N/A')}
        - ESPN Implied: {opportunity_data.get('espn_implied', 'N/A')}
        - Divergence: {opportunity_data.get('divergence', 'N/A')}
        - Opportunity Score: {opportunity_data.get('opportunity_score', 0)}
        
        Context:
        {opportunity_data.get('data', {})}
        
        Provide analysis with:
        1. Signal: BUY_YES, BUY_NO, SELL_YES, SELL_NO, or HOLD
        2. Confidence: 0-1 scale
        3. Probability: Your estimate of true probability
        4. Risk Level: low, medium, or high
        5. Position Size: Percentage of capital (using Kelly principles)
        6. Stop Loss: Price level to exit at loss
        7. Take Profit: Price level to take profits
        8. Reasoning: Clear explanation
        
        Format as JSON.
        """
        
        try:
            # Get LLM analysis
            response = await self.llm_client.complete_json(
                analysis_prompt,
                temperature=0.3
            )
            
            # Parse LLM response if available
            if isinstance(response, dict) and not response.get('error'):
                signal_map = {
                    'BUY_YES': SignalType.BUY_YES,
                    'BUY_NO': SignalType.BUY_NO,
                    'SELL_YES': SignalType.SELL_YES,
                    'SELL_NO': SignalType.SELL_NO,
                    'HOLD': SignalType.HOLD
                }
                
                signal_str = response.get('signal', 'HOLD')
                if signal_str in signal_map:
                    return TradingSignal(
                        market_ticker=market_ticker,
                        signal_type=signal_map[signal_str],
                        confidence=response.get('confidence', 0.5),
                        probability=response.get('probability', 0.5),
                        reasoning=response.get('reasoning', 'LLM analysis'),
                        risk_level=response.get('risk_level', 'medium'),
                        suggested_size_pct=response.get('position_size', 0.05),
                        stop_loss=response.get('stop_loss'),
                        take_profit=response.get('take_profit')
                    )
            
            # Parse response (would need better parsing in production)
            # For now, create signal based on opportunity score
            if opportunity_data.get('opportunity_score', 0) > 0.7:
                # Determine signal type based on divergence
                market_price = opportunity_data.get('market_price', 0.5)
                sentiment_implied = opportunity_data.get('sentiment_implied', 0.5)
                
                if sentiment_implied > market_price + 0.1:
                    signal_type = SignalType.BUY_YES
                elif sentiment_implied < market_price - 0.1:
                    signal_type = SignalType.BUY_NO
                else:
                    signal_type = SignalType.HOLD
                
                # Calculate position size (simplified Kelly)
                edge = abs(sentiment_implied - market_price)
                kelly_fraction = min(edge * 2, 0.25)  # Cap at 25% of capital
                
                signal = TradingSignal(
                    market_ticker=market_ticker,
                    signal_type=signal_type,
                    confidence=min(opportunity_data.get('opportunity_score', 0.5), 0.9),
                    probability=sentiment_implied,
                    reasoning=f"Divergence detected: Market at {market_price:.2f}, Sentiment implies {sentiment_implied:.2f}",
                    risk_level="medium" if edge > 0.15 else "low",
                    suggested_size_pct=kelly_fraction,
                    stop_loss=market_price - 0.15 if signal_type == SignalType.BUY_YES else market_price + 0.15,
                    take_profit=market_price + 0.25 if signal_type == SignalType.BUY_YES else market_price - 0.25
                )
                
                # Store signal
                self.signals.append(signal)
                self.signals_generated += 1
                
                # Track active position
                if signal.signal_type not in [SignalType.HOLD, SignalType.CLOSE]:
                    self.active_positions[market_ticker] = signal
                
                logger.info(f"Generated signal: {signal.signal_type.value} for {market_ticker}")
                
                return signal
            
        except Exception as e:
            logger.error(f"Error analyzing opportunity: {e}")
        
        return None
    
    async def evaluate_position(self, market_ticker: str, current_data: Dict[str, Any]) -> Optional[SignalType]:
        """
        Evaluate existing position for exit signals
        
        Args:
            market_ticker: Market to evaluate
            current_data: Current market data
            
        Returns:
            Exit signal if warranted
        """
        if market_ticker not in self.active_positions:
            return None
        
        position = self.active_positions[market_ticker]
        current_price = current_data.get('yes_price', 0.5)
        
        # Check stop loss
        if position.stop_loss:
            if position.signal_type == SignalType.BUY_YES and current_price <= position.stop_loss:
                logger.warning(f"Stop loss triggered for {market_ticker}")
                return SignalType.SELL_YES
            elif position.signal_type == SignalType.BUY_NO and current_price >= position.stop_loss:
                logger.warning(f"Stop loss triggered for {market_ticker}")
                return SignalType.SELL_NO
        
        # Check take profit
        if position.take_profit:
            if position.signal_type == SignalType.BUY_YES and current_price >= position.take_profit:
                logger.info(f"Take profit triggered for {market_ticker}")
                return SignalType.SELL_YES
            elif position.signal_type == SignalType.BUY_NO and current_price <= position.take_profit:
                logger.info(f"Take profit triggered for {market_ticker}")
                return SignalType.SELL_NO
        
        return None
    
    async def handle_event(self, event_type: str, event_data: Dict[str, Any]):
        """
        Handle events from DataCoordinator
        
        Args:
            event_type: Type of event
            event_data: Event data
        """
        if event_type == "analyze_opportunity":
            # Analyze market opportunity
            signal = await self.analyze_opportunity(event_data)
            
            if signal and signal.signal_type != SignalType.HOLD:
                # Send signal to ExecutionManager
                if self.message_handler:
                    await self.message_handler("trading_signal", {
                        "signal": signal.__dict__,
                        "timestamp": datetime.now().isoformat()
                    })
        
        elif event_type == "price_update":
            # Check existing positions
            market_ticker = event_data.get("market_ticker")
            exit_signal = await self.evaluate_position(market_ticker, event_data)
            
            if exit_signal:
                signal = TradingSignal(
                    market_ticker=market_ticker,
                    signal_type=exit_signal,
                    confidence=0.9,
                    probability=0,  # Not relevant for exit
                    reasoning="Exit condition met",
                    risk_level="low",
                    suggested_size_pct=1.0  # Close full position
                )
                
                if self.message_handler:
                    await self.message_handler("exit_signal", {
                        "signal": signal.__dict__,
                        "timestamp": datetime.now().isoformat()
                    })
                
                # Remove from active positions
                del self.active_positions[market_ticker]
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "opportunities_analyzed": self.opportunities_analyzed,
            "signals_generated": self.signals_generated,
            "active_positions": len(self.active_positions),
            "recent_signals": [
                {
                    "market": s.market_ticker,
                    "type": s.signal_type.value,
                    "confidence": s.confidence,
                    "time": s.timestamp.isoformat()
                }
                for s in self.signals[-5:]  # Last 5 signals
            ]
        }


# Create singleton instance
strategy_analyst_agent = StrategyAnalystAgent()


# Example usage
async def main():
    """Example of running the Strategy Analyst"""
    
    agent = StrategyAnalystAgent()
    
    # Analyze an opportunity
    opportunity = {
        "market_ticker": "SUPERBOWL-2025",
        "market_price": 0.45,
        "sentiment_implied": 0.62,
        "espn_implied": 0.58,
        "divergence": 0.17,
        "opportunity_score": 0.75
    }
    
    signal = await agent.analyze_opportunity(opportunity)
    if signal:
        print(f"Signal: {signal.signal_type.value}")
        print(f"Confidence: {signal.confidence}")
        print(f"Size: {signal.suggested_size_pct:.1%} of capital")
        print(f"Reasoning: {signal.reasoning}")
    
    # Get status
    status = agent.get_status()
    print(f"Status: {status}")


if __name__ == "__main__":
    asyncio.run(main())