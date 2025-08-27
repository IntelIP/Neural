"""
Market Engineer Agent
Analyzes sentiment and identifies market inefficiencies
Powered by Google Gemini 2.5 Flash with E2B Sandbox
Now with Redis consumer for real-time market data
"""

import os
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import logging
import json

from trading_logic.llm_client import get_llm_client
from trading_logic.e2b_executor import get_e2b_executor
from agent_consumers.base_consumer import BaseAgentRedisConsumer

logger = logging.getLogger(__name__)


class MarketEngineerAgent:
    """
    Market Engineer Agent - Analyzes sentiment with E2B sandbox.
    
    Responsibilities:
    - Consume tweets from Data Engineer
    - Analyze sentiment using Gemini 2.5 Flash + E2B sandbox
    - Calculate sentiment velocity and momentum
    - Identify sentiment-price divergences
    - Generate opportunity scores and trading signals
    """
    
    def __init__(self):
        """Initialize Market Engineer Agent."""
        # Initialize LLM client
        self.llm_client = get_llm_client()
        
        # E2B executor for sentiment analysis
        self.e2b_executor = get_e2b_executor()
        
        # Message handler will be set by the handler wrapper
        self.message_handler = None
        
        # Market sentiment tracking
        self.market_sentiments: Dict[str, Dict] = {}
        
        # State
        self.is_running = False
        self.tasks: List[asyncio.Task] = []
    
    async def start(self, agent_context=None):
        """Start the Market Engineer agent."""
        if self.is_running:
            logger.warning("Market Engineer already running")
            return
        
        self.is_running = True
        logger.info("Starting Market Engineer Agent...")
        
        # Connection handled by Agentuity
        
        # Message handlers registered by Agentuity handler
        # Handlers will be called directly by the wrapper
        
        # Start analysis loop
        self.tasks.append(
            asyncio.create_task(self._analysis_loop())
        )
        
        # Start Redis consumer
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis_consumer = MarketEngineerRedisConsumer(self, agent_context)
        self.redis_consumer.redis_url = redis_url
        
        await self.redis_consumer.connect()
        await self.redis_consumer.subscribe([
            "kalshi:markets",
            "espn:games"
        ])
        
        # Start consuming in background
        asyncio.create_task(self.redis_consumer.start_consuming())
        
        logger.info("Market Engineer Agent started successfully with Redis consumer")
    
    async def analyze_sentiment_batch(self, tweets: List[Dict]) -> Dict[str, Any]:
        """
        Run advanced sentiment analysis in E2B sandbox.
        
        Args:
            tweets: List of tweet dictionaries
            
        Returns:
            Aggregated sentiment analysis results
        """
        if not tweets:
            return {"error": "No tweets to analyze"}
        
        # Prepare tweet texts
        tweet_texts = [t.get("text", "") for t in tweets]
        
        # E2B sandbox code for sentiment analysis
        sentiment_code = f'''
import json
import numpy as np

# Install required packages
import subprocess
import sys

def install_packages():
    packages = ["vaderSentiment", "textblob", "nltk"]
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

install_packages()

# Import after installation
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk

# Download required NLTK data
nltk.download('brown', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('vader_lexicon', quiet=True)

def analyze_tweets(tweets):
    """Analyze sentiment of tweets using multiple methods."""
    # Initialize analyzers
    vader = SentimentIntensityAnalyzer()
    
    results = []
    for tweet in tweets:
        # VADER sentiment (optimized for social media)
        vader_scores = vader.polarity_scores(tweet)
        
        # TextBlob sentiment
        try:
            blob = TextBlob(tweet)
            textblob_sentiment = blob.sentiment.polarity
            textblob_subjectivity = blob.sentiment.subjectivity
        except:
            textblob_sentiment = 0.0
            textblob_subjectivity = 0.5
        
        # Custom sports betting keywords
        positive_keywords = ["win", "dominating", "crushing", "healthy", "cleared", "returning", "confident"]
        negative_keywords = ["injury", "out", "questionable", "doubtful", "struggling", "benched", "suspended"]
        
        positive_count = sum(1 for word in positive_keywords if word in tweet.lower())
        negative_count = sum(1 for word in negative_keywords if word in tweet.lower())
        keyword_score = (positive_count - negative_count) * 0.1
        
        # Combined weighted score
        combined_score = (
            vader_scores['compound'] * 0.5 +  # VADER weighted heavily for social media
            textblob_sentiment * 0.3 +         # TextBlob for general sentiment
            keyword_score * 0.2                # Domain-specific keywords
        )
        
        # Normalize to -1 to 1
        combined_score = max(-1, min(1, combined_score))
        
        # Classify sentiment
        if combined_score > 0.1:
            sentiment = 'positive'
            confidence = min(abs(combined_score), 1.0)
        elif combined_score < -0.1:
            sentiment = 'negative'
            confidence = min(abs(combined_score), 1.0)
        else:
            sentiment = 'neutral'
            confidence = 1 - abs(combined_score)
        
        results.append({{
            'text': tweet[:100] + '...' if len(tweet) > 100 else tweet,
            'vader_compound': vader_scores['compound'],
            'vader_positive': vader_scores['pos'],
            'vader_negative': vader_scores['neg'],
            'textblob_polarity': textblob_sentiment,
            'textblob_subjectivity': textblob_subjectivity,
            'keyword_score': keyword_score,
            'combined_score': combined_score,
            'sentiment': sentiment,
            'confidence': confidence
        }})
    
    # Calculate aggregate metrics
    scores = [r['combined_score'] for r in results]
    
    # Sentiment momentum (recent vs older tweets)
    if len(scores) > 1:
        recent_avg = np.mean(scores[-min(5, len(scores)):])
        older_avg = np.mean(scores[:max(1, len(scores)-5)])
        momentum = recent_avg - older_avg
    else:
        momentum = 0.0
    
    # Sentiment velocity (rate of change)
    if len(scores) > 2:
        velocity = (scores[-1] - scores[0]) / len(scores)
    else:
        velocity = 0.0
    
    aggregate = {{
        'mean_sentiment': float(np.mean(scores)),
        'sentiment_std': float(np.std(scores)),
        'sentiment_momentum': float(momentum),
        'sentiment_velocity': float(velocity),
        'positive_ratio': sum(1 for r in results if r['sentiment'] == 'positive') / len(results),
        'negative_ratio': sum(1 for r in results if r['sentiment'] == 'negative') / len(results),
        'neutral_ratio': sum(1 for r in results if r['sentiment'] == 'neutral') / len(results),
        'confidence': float(np.mean([r['confidence'] for r in results])),
        'tweet_count': len(results),
        'detailed_results': results[:5]  # Return top 5 for inspection
    }}
    
    return aggregate

# Run analysis
tweets = {json.dumps(tweet_texts)}
result = analyze_tweets(tweets)
print(json.dumps(result, indent=2))
'''
        
        try:
            # Execute sentiment analysis in E2B sandbox
            result = await self.e2b_executor.run_python_code(sentiment_code)
            
            # Parse results
            if result and not result.get("error"):
                return result.get("result", {"error": "No result returned"})
            
            return {"error": "Failed to parse sentiment results"}
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return {"error": str(e)}
    
    async def calculate_divergence(
        self,
        market_ticker: str,
        market_price: float,
        sentiment_score: float
    ) -> Dict[str, Any]:
        """
        Calculate sentiment-price divergence.
        
        Args:
            market_ticker: Market identifier
            market_price: Current market price (0-1)
            sentiment_score: Sentiment score (-1 to 1)
            
        Returns:
            Divergence analysis
        """
        # Convert sentiment to implied probability
        sentiment_implied_prob = 0.5 + (sentiment_score * 0.4)
        sentiment_implied_prob = max(0.01, min(0.99, sentiment_implied_prob))
        
        # Calculate divergence
        divergence = sentiment_implied_prob - market_price
        
        # Use Gemini for analysis
        analysis_prompt = f"""
        Analyze this sentiment-price divergence:
        - Market: {market_ticker}
        - Current Price: {market_price:.2%}
        - Sentiment Score: {sentiment_score:.2f}
        - Implied Probability: {sentiment_implied_prob:.2%}
        - Divergence: {divergence:.2%}
        
        Determine:
        1. Is this divergence significant? (>15% is typically significant)
        2. What direction is the opportunity? (buy YES or NO)
        3. Confidence level (0-1)
        4. Key risks
        
        Be concise and focus on actionable insights.
        """
        
        response_text = await self.llm_client.complete(
            analysis_prompt,
            temperature=0.3,
            system_prompt="You are a market analysis expert identifying trading opportunities."
        )
        
        return {
            "market_ticker": market_ticker,
            "market_price": market_price,
            "sentiment_score": sentiment_score,
            "sentiment_implied_prob": sentiment_implied_prob,
            "divergence": divergence,
            "is_significant": abs(divergence) > 0.15,
            "analysis": response_text
        }
    
    async def _handle_market_data(self, data: Dict[str, Any]):
        """Handle market data updates."""
        market_ticker = data.get("market_ticker")
        
        if market_ticker:
            # Update market sentiment tracking
            if market_ticker not in self.market_sentiments:
                self.market_sentiments[market_ticker] = {
                    "prices": [],
                    "sentiments": [],
                    "last_analysis": None
                }
            
            # Store price data
            self.market_sentiments[market_ticker]["prices"].append({
                "timestamp": datetime.now(),
                "yes_bid": data.get("yes_bid"),
                "yes_ask": data.get("yes_ask")
            })
            
            # Keep only last 100 data points
            self.market_sentiments[market_ticker]["prices"] = \
                self.market_sentiments[market_ticker]["prices"][-100:]
    
    async def _handle_tweet_batch(self, data: Dict[str, Any]):
        """Handle batch of tweets for sentiment analysis."""
        tweets = data.get("tweets", [])
        market_ticker = data.get("market_ticker")
        
        if not tweets or not market_ticker:
            return
        
        # Analyze sentiment batch
        sentiment_results = await self.analyze_sentiment_batch(tweets)
        
        if "error" not in sentiment_results:
            # Store sentiment results
            if market_ticker in self.market_sentiments:
                self.market_sentiments[market_ticker]["sentiments"].append({
                    "timestamp": datetime.now(),
                    "results": sentiment_results
                })
                
                # Keep only last 50 analyses
                self.market_sentiments[market_ticker]["sentiments"] = \
                    self.market_sentiments[market_ticker]["sentiments"][-50:]
            
            # Check for divergence if we have price data
            if market_ticker in self.market_sentiments:
                prices = self.market_sentiments[market_ticker]["prices"]
                if prices:
                    latest_price = prices[-1]
                    market_price = (latest_price["yes_bid"] + latest_price["yes_ask"]) / 2
                    
                    divergence = await self.calculate_divergence(
                        market_ticker=market_ticker,
                        market_price=market_price,
                        sentiment_score=sentiment_results["mean_sentiment"]
                    )
                    
                    # Publish sentiment analysis
                    await self.communicator.send_sentiment(
                        market_ticker=market_ticker,
                        sentiment={
                            "score": sentiment_results["mean_sentiment"],
                            "momentum": sentiment_results["sentiment_momentum"],
                            "velocity": sentiment_results["sentiment_velocity"],
                            "confidence": sentiment_results["confidence"],
                            "divergence": divergence["divergence"],
                            "is_significant": divergence["is_significant"]
                        }
                    )
                    
                    # Generate trade signal if significant divergence
                    if divergence["is_significant"]:
                        await self.generate_trade_signal(divergence)
    
    async def generate_trade_signal(self, divergence: Dict[str, Any]):
        """Generate trade signal based on divergence analysis."""
        signal = {
            "market_ticker": divergence["market_ticker"],
            "timestamp": datetime.now().isoformat(),
            "divergence": divergence["divergence"],
            "sentiment_score": divergence["sentiment_score"],
            "market_price": divergence["market_price"],
            "action": "BUY_YES" if divergence["divergence"] > 0 else "BUY_NO",
            "confidence": abs(divergence["divergence"]) / 0.5,  # Normalize confidence
            "analysis": divergence["analysis"]
        }
        
        # Send trade signal to handler
        if self.message_handler:
            await self.message_handler('trade_signal', signal)
        
        logger.info(f"Trade signal generated: {signal['market_ticker']} - {signal['action']}")
    
    async def _process_market_message(self, message: Dict[str, Any]):
        """Process messages from market data channel."""
        msg_type = message.get("type")
        data = message.get("data")
        
        # Message type handling moved to Agentuity handler
        pass
    
    async def _analysis_loop(self):
        """Periodic analysis loop."""
        while self.is_running:
            try:
                # Analyze trends for each tracked market
                for market_ticker, data in self.market_sentiments.items():
                    if data["sentiments"] and data["prices"]:
                        # Check if analysis is needed
                        last_analysis = data.get("last_analysis")
                        if not last_analysis or (datetime.now() - last_analysis).seconds > 60:
                            # Perform trend analysis
                            await self._analyze_market_trends(market_ticker)
                            data["last_analysis"] = datetime.now()
                
                await asyncio.sleep(30)  # Analyze every 30 seconds
                
            except Exception as e:
                logger.error(f"Analysis loop error: {e}")
                await asyncio.sleep(60)
    
    async def _analyze_market_trends(self, market_ticker: str):
        """Analyze market trends using historical data."""
        data = self.market_sentiments.get(market_ticker)
        if not data:
            return
        
        # Get recent sentiments
        recent_sentiments = data["sentiments"][-10:] if data["sentiments"] else []
        recent_prices = data["prices"][-10:] if data["prices"] else []
        
        if recent_sentiments and recent_prices:
            # Calculate trends
            sentiment_scores = [s["results"]["mean_sentiment"] for s in recent_sentiments]
            price_values = [(p["yes_bid"] + p["yes_ask"]) / 2 for p in recent_prices]
            
            # Simple trend detection
            sentiment_trend = sentiment_scores[-1] - sentiment_scores[0] if len(sentiment_scores) > 1 else 0
            price_trend = price_values[-1] - price_values[0] if len(price_values) > 1 else 0
            
            # Check for trend divergence
            if abs(sentiment_trend) > 0.2 and abs(price_trend) < 0.05:
                logger.info(f"Trend divergence detected for {market_ticker}")
                logger.info(f"Sentiment trend: {sentiment_trend:.2f}, Price trend: {price_trend:.2f}")
    
    async def _handle_emergency_stop(self, data: Dict[str, Any]):
        """Handle emergency stop signal."""
        logger.warning(f"Emergency stop received: {data}")
        await self.stop()
    
    async def stop(self):
        """Stop the Market Engineer agent."""
        self.is_running = False
        
        # Cancel tasks
        for task in self.tasks:
            task.cancel()
        
        # Stop Redis consumer if running
        if hasattr(self, 'redis_consumer') and self.redis_consumer:
            await self.redis_consumer.disconnect()
        
        # Cleanup handled by Agentuity
        
        logger.info("Market Engineer Agent stopped")


class MarketEngineerRedisConsumer(BaseAgentRedisConsumer):
    """
    Redis consumer for MarketEngineer
    Identifies trading opportunities from real-time data
    """
    
    def __init__(self, engineer: MarketEngineerAgent, agent_context=None):
        super().__init__("MarketEngineer", agent_context=agent_context)
        self.engineer = engineer
        self.opportunities_found = 0
        
    async def process_message(self, channel: str, data: Dict[str, Any]):
        """Process incoming Redis messages"""
        
        if channel == "kalshi:markets":
            await self._analyze_market_opportunity(data)
        elif channel == "espn:games":
            await self._analyze_game_impact(data)
    
    async def _analyze_market_opportunity(self, data: Dict[str, Any]):
        """Analyze market data for opportunities"""
        market_data = data.get('data', {})
        market_ticker = market_data.get('market_ticker')
        
        if not market_ticker:
            return
        
        yes_price = market_data.get('yes_price', 0)
        no_price = market_data.get('no_price', 0)
        
        # Check for arbitrage opportunity
        if yes_price > 0 and no_price > 0:
            total = yes_price + no_price
            
            if total < 0.98:  # Arbitrage opportunity
                opportunity = {
                    "type": "arbitrage",
                    "action": "BUY_BOTH",
                    "market_ticker": market_ticker,
                    "yes_price": yes_price,
                    "no_price": no_price,
                    "profit_potential": 1.0 - total,
                    "confidence": 0.95,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                logger.info(f"Arbitrage opportunity found: {opportunity}")
                self.opportunities_found += 1
                
                # Publish signal for TradeExecutor
                await self.publish_signal(opportunity)
            
            # Check for extreme pricing
            elif yes_price < 0.1 or yes_price > 0.9:
                # Potential mispricing
                signal = {
                    "type": "mispricing",
                    "action": "BUY_NO" if yes_price > 0.9 else "BUY_YES",
                    "market_ticker": market_ticker,
                    "yes_price": yes_price,
                    "confidence": 0.7,
                    "reason": f"Extreme pricing detected: {yes_price:.2f}"
                }
                
                await self.publish_signal(signal)
    
    async def _analyze_game_impact(self, data: Dict[str, Any]):
        """Analyze game events for market impact"""
        game_data = data.get('data', {})
        
        # Check for major game events
        if 'touchdown' in str(game_data).lower() or 'injury' in str(game_data).lower():
            logger.info(f"Major game event detected: {game_data}")
            
            # Could correlate with market positions here
            # For now, just log the event


# Create singleton instance
market_engineer_agent = MarketEngineerAgent()


# Example usage
async def main():
    """Example of running the Market Engineer agent."""
    # Initialize agent
    agent = MarketEngineerAgent()
    
    # Start agent
    await agent.start()
    
    # Test sentiment analysis
    test_tweets = [
        {"text": "Chiefs looking dominant! Mahomes is on fire today! 🔥"},
        {"text": "Bills injury report: Key player questionable for Sunday"},
        {"text": "Weather conditions perfect for the game"},
        {"text": "Betting heavily on the Chiefs to cover the spread"},
        {"text": "Bills struggling in practice, not looking good"}
    ]
    
    results = await agent.analyze_sentiment_batch(test_tweets)
    print(f"Sentiment results: {json.dumps(results, indent=2)}")
    
    # Run for a while
    await asyncio.sleep(60)
    
    # Stop agent
    await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())