"""
Game Analyst Agent - On-Demand Agent
Deep analysis of specific games when triggered
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import redis.asyncio as redis

from src.trading.llm_client import get_llm_client

logger = logging.getLogger(__name__)


@dataclass
class GameAnalysis:
    """Comprehensive game analysis result"""
    game_id: str
    teams: Tuple[str, str]
    recommended_position: Optional[str] = None  # 'yes', 'no', or None
    confidence: float = 0.0
    kelly_fraction: float = 0.0
    key_factors: List[str] = None
    risk_warnings: List[str] = None
    analysis_timestamp: datetime = None
    
    def __post_init__(self):
        if self.key_factors is None:
            self.key_factors = []
        if self.risk_warnings is None:
            self.risk_warnings = []
        if self.analysis_timestamp is None:
            self.analysis_timestamp = datetime.utcnow()


class GameAnalystAgent:
    """
    Game Analyst Agent - On-Demand
    
    Triggered by:
    - User requests analysis of specific game
    - Data Collection Agent detects high-opportunity game
    - Pre-game analysis (1 hour before kickoff)
    - Major game events (injuries, momentum shifts)
    
    Analysis includes:
    - Historical performance
    - Current form and injuries
    - Market analysis
    - Sentiment analysis
    - Weather and venue factors
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client = None
        self.pubsub = None
        self.is_active = False
        
        # LLM client for analysis
        self.llm_client = get_llm_client()
        
        # Cache for recent analyses
        self.analysis_cache: Dict[str, GameAnalysis] = {}
        self.cache_duration = timedelta(minutes=30)
        
        # Statistics
        self.analyses_performed = 0
        self.successful_predictions = 0
    
    async def connect(self):
        """Connect to Redis"""
        self.redis_client = redis.from_url(self.redis_url)
        self.pubsub = self.redis_client.pubsub()
        
        # Subscribe to activation channel
        await self.pubsub.subscribe("agent:activate:gameanalyst")
        
        logger.info("Game Analyst connected to Redis")
    
    async def start(self):
        """Start listening for activation triggers"""
        if self.is_active:
            logger.warning("Game Analyst already active")
            return
        
        self.is_active = True
        logger.info("Game Analyst ready for activation")
        
        # Listen for activation requests
        asyncio.create_task(self._listen_for_activation())
    
    async def _listen_for_activation(self):
        """Listen for activation triggers from TriggerService"""
        async for message in self.pubsub.listen():
            if not self.is_active:
                break
            
            if message['type'] == 'message':
                try:
                    activation_data = json.loads(message['data'])
                    logger.info(f"Game Analyst activated: {activation_data['trigger']}")
                    
                    # Perform analysis
                    await self._handle_activation(activation_data)
                    
                except Exception as e:
                    logger.error(f"Error handling activation: {e}")
    
    async def _handle_activation(self, activation_data: Dict[str, Any]):
        """Handle activation request"""
        trigger = activation_data.get('trigger')
        data = activation_data.get('data', {})
        priority = activation_data.get('priority')
        
        # Extract game information
        game_id = data.get('game_id')
        teams = (data.get('home_team'), data.get('away_team'))
        
        if not game_id or not all(teams):
            # Try to extract from market ticker
            market_ticker = data.get('market_ticker', '')
            if market_ticker:
                # Parse market ticker for game info
                game_id = self._extract_game_id(market_ticker)
                teams = self._extract_teams(market_ticker)
        
        if game_id and all(teams):
            # Perform analysis
            analysis = await self.analyze_game(game_id, teams, trigger_context=data)
            
            # Publish results
            await self._publish_analysis(analysis, priority)
    
    async def analyze_game(
        self, 
        game_id: str, 
        teams: Tuple[str, str],
        trigger_context: Dict[str, Any] = None
    ) -> GameAnalysis:
        """
        Perform comprehensive game analysis
        
        Args:
            game_id: Game identifier
            teams: (home_team, away_team)
            trigger_context: Context data from trigger
            
        Returns:
            GameAnalysis with recommendations
        """
        # Check cache first
        cache_key = f"{game_id}:{teams[0]}:{teams[1]}"
        if cache_key in self.analysis_cache:
            cached = self.analysis_cache[cache_key]
            if datetime.utcnow() - cached.analysis_timestamp < self.cache_duration:
                logger.info(f"Returning cached analysis for {cache_key}")
                return cached
        
        logger.info(f"Analyzing game: {teams[0]} vs {teams[1]} (ID: {game_id})")
        
        # Gather all data sources
        historical = await self._get_historical_data(teams)
        current_form = await self._get_current_form(teams)
        injuries = await self._get_injury_report(teams)
        market_data = await self._get_market_data(game_id)
        sentiment = await self._get_sentiment_data(teams)
        weather = await self._get_weather_conditions(game_id)
        
        # Use LLM for comprehensive analysis
        analysis = await self._llm_analyze(
            game_id=game_id,
            teams=teams,
            historical=historical,
            form=current_form,
            injuries=injuries,
            market=market_data,
            sentiment=sentiment,
            weather=weather,
            trigger_context=trigger_context
        )
        
        # Cache the analysis
        self.analysis_cache[cache_key] = analysis
        self.analyses_performed += 1
        
        return analysis
    
    async def _get_historical_data(self, teams: Tuple[str, str]) -> Dict[str, Any]:
        """Get historical performance data"""
        # Query historical data from Redis or database
        historical_key = f"historical:{teams[0]}:{teams[1]}"
        data = await self.redis_client.get(historical_key)
        
        if data:
            return json.loads(data)
        
        # Default historical data
        return {
            "h2h_record": {"home_wins": 0, "away_wins": 0, "draws": 0},
            "home_recent_form": "WWLWL",  # Last 5 games
            "away_recent_form": "LWWLL",
            "avg_total_points": 45.5
        }
    
    async def _get_current_form(self, teams: Tuple[str, str]) -> Dict[str, Any]:
        """Get current form and recent performance"""
        form_data = {}
        
        for team in teams:
            form_key = f"form:{team}"
            data = await self.redis_client.get(form_key)
            
            if data:
                form_data[team] = json.loads(data)
            else:
                form_data[team] = {
                    "last_5_games": "WWLWL",
                    "points_for_avg": 27.5,
                    "points_against_avg": 21.3,
                    "home_away_split": {"home": "3-1", "away": "2-2"}
                }
        
        return form_data
    
    async def _get_injury_report(self, teams: Tuple[str, str]) -> Dict[str, Any]:
        """Get injury reports for both teams"""
        injuries = {}
        
        for team in teams:
            injury_key = f"injuries:{team}"
            data = await self.redis_client.get(injury_key)
            
            if data:
                injuries[team] = json.loads(data)
            else:
                injuries[team] = []
        
        return injuries
    
    async def _get_market_data(self, game_id: str) -> Dict[str, Any]:
        """Get current market data from Kalshi"""
        market_key = f"market:{game_id}"
        data = await self.redis_client.get(market_key)
        
        if data:
            return json.loads(data)
        
        return {
            "yes_price": 0.50,
            "no_price": 0.50,
            "volume": 0,
            "open_interest": 0,
            "implied_probability": 0.50
        }
    
    async def _get_sentiment_data(self, teams: Tuple[str, str]) -> Dict[str, Any]:
        """Get sentiment analysis from Twitter/social media"""
        sentiment = {}
        
        for team in teams:
            sentiment_key = f"sentiment:{team}"
            data = await self.redis_client.get(sentiment_key)
            
            if data:
                sentiment[team] = json.loads(data)
            else:
                sentiment[team] = {
                    "score": 0.0,  # -1 to 1
                    "volume": 0,
                    "trending": False
                }
        
        return sentiment
    
    async def _get_weather_conditions(self, game_id: str) -> Dict[str, Any]:
        """Get weather conditions for outdoor games"""
        weather_key = f"weather:{game_id}"
        data = await self.redis_client.get(weather_key)
        
        if data:
            return json.loads(data)
        
        return {
            "temperature": 72,
            "wind_speed": 5,
            "precipitation": 0,
            "indoor": True
        }
    
    async def _llm_analyze(
        self,
        game_id: str,
        teams: Tuple[str, str],
        historical: Dict[str, Any],
        form: Dict[str, Any],
        injuries: Dict[str, Any],
        market: Dict[str, Any],
        sentiment: Dict[str, Any],
        weather: Dict[str, Any],
        trigger_context: Dict[str, Any] = None
    ) -> GameAnalysis:
        """Use LLM to analyze all data and generate recommendations"""
        
        prompt = f"""
        Analyze this game and provide trading recommendations:
        
        Game: {teams[0]} (home) vs {teams[1]} (away)
        Game ID: {game_id}
        
        Historical Data:
        - H2H Record: {historical['h2h_record']}
        - Home Recent Form: {historical['home_recent_form']}
        - Away Recent Form: {historical['away_recent_form']}
        
        Current Form:
        - {teams[0]}: {form.get(teams[0], {})}
        - {teams[1]}: {form.get(teams[1], {})}
        
        Injuries:
        - {teams[0]}: {len(injuries.get(teams[0], []))} players injured
        - {teams[1]}: {len(injuries.get(teams[1], []))} players injured
        
        Market Data:
        - Current Odds: Yes {market['yes_price']:.2f} / No {market['no_price']:.2f}
        - Volume: {market['volume']}
        - Implied Probability: {market['implied_probability']:.2%}
        
        Sentiment:
        - {teams[0]}: Score {sentiment.get(teams[0], {}).get('score', 0):.2f}
        - {teams[1]}: Score {sentiment.get(teams[1], {}).get('score', 0):.2f}
        
        Weather:
        - Temperature: {weather['temperature']}°F
        - Wind: {weather['wind_speed']} mph
        - Indoor: {weather['indoor']}
        
        Trigger Context: {trigger_context}
        
        Provide:
        1. Recommended position (yes/no/none)
        2. Confidence level (0-1)
        3. Kelly fraction recommendation (0-0.25)
        4. Key factors (list 3-5)
        5. Risk warnings (list any concerns)
        
        Format as JSON.
        """
        
        try:
            response = await self.llm_client.complete(prompt, temperature=0.2)
            
            # Parse LLM response
            # In production, would use structured output
            analysis_data = self._parse_llm_response(response)
            
            return GameAnalysis(
                game_id=game_id,
                teams=teams,
                recommended_position=analysis_data.get('position'),
                confidence=analysis_data.get('confidence', 0.5),
                kelly_fraction=analysis_data.get('kelly', 0.05),
                key_factors=analysis_data.get('factors', []),
                risk_warnings=analysis_data.get('risks', [])
            )
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            
            # Fallback to simple analysis
            return GameAnalysis(
                game_id=game_id,
                teams=teams,
                recommended_position=None,
                confidence=0.0,
                kelly_fraction=0.0,
                key_factors=["Analysis failed"],
                risk_warnings=["Unable to complete analysis"]
            )
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured data"""
        try:
            # Try to parse as JSON
            if '{' in response and '}' in response:
                json_str = response[response.find('{'):response.rfind('}')+1]
                return json.loads(json_str)
        except:
            pass
        
        # Fallback parsing
        result = {
            'position': None,
            'confidence': 0.5,
            'kelly': 0.05,
            'factors': [],
            'risks': []
        }
        
        # Simple keyword extraction
        if 'yes' in response.lower() and 'recommend' in response.lower():
            result['position'] = 'yes'
        elif 'no' in response.lower() and 'recommend' in response.lower():
            result['position'] = 'no'
        
        return result
    
    async def _publish_analysis(self, analysis: GameAnalysis, priority: str):
        """Publish analysis results"""
        result = {
            "type": "GAME_ANALYSIS",
            "game_id": analysis.game_id,
            "teams": analysis.teams,
            "recommendation": {
                "position": analysis.recommended_position,
                "confidence": analysis.confidence,
                "kelly_fraction": analysis.kelly_fraction
            },
            "factors": analysis.key_factors,
            "risks": analysis.risk_warnings,
            "priority": priority,
            "timestamp": analysis.analysis_timestamp.isoformat()
        }
        
        # Publish to analysis channel
        await self.redis_client.publish("analysis:game", json.dumps(result))
        
        # If high confidence, publish as signal
        if analysis.confidence > 0.7 and analysis.recommended_position:
            signal = {
                "action": "TRADE_SIGNAL",
                "source": "GameAnalyst",
                "market_ticker": f"{analysis.teams[0]}-{analysis.teams[1]}",
                "side": analysis.recommended_position,
                "confidence": analysis.confidence,
                "kelly_fraction": analysis.kelly_fraction,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self.redis_client.publish("kalshi:signals", json.dumps(signal))
            logger.info(f"Published trade signal: {signal}")
    
    def _extract_game_id(self, market_ticker: str) -> Optional[str]:
        """Extract game ID from market ticker"""
        # Implementation depends on market ticker format
        # Example: "NFL-CHIEFS-BILLS-20240115" -> "20240115"
        parts = market_ticker.split('-')
        if len(parts) >= 4:
            return parts[-1]
        return None
    
    def _extract_teams(self, market_ticker: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract teams from market ticker"""
        # Example: "NFL-CHIEFS-BILLS-20240115" -> ("CHIEFS", "BILLS")
        parts = market_ticker.split('-')
        if len(parts) >= 3:
            return (parts[1], parts[2])
        return (None, None)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            "is_active": self.is_active,
            "analyses_performed": self.analyses_performed,
            "cache_size": len(self.analysis_cache),
            "successful_predictions": self.successful_predictions
        }
    
    async def stop(self):
        """Stop the agent"""
        self.is_active = False
        
        if self.pubsub:
            await self.pubsub.unsubscribe()
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info(f"Game Analyst stopped. Stats: {self.get_statistics()}")


# Example usage
async def main():
    """Example of running the Game Analyst"""
    analyst = GameAnalystAgent()
    await analyst.connect()
    await analyst.start()
    
    # Simulate manual analysis
    await asyncio.sleep(2)
    
    analysis = await analyst.analyze_game(
        game_id="401547435",
        teams=("Chiefs", "Bills")
    )
    
    print(f"Analysis: {analysis}")
    
    # Get statistics
    stats = analyst.get_statistics()
    print(f"Stats: {json.dumps(stats, indent=2)}")
    
    # Run for a while
    await asyncio.sleep(60)
    
    await analyst.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())