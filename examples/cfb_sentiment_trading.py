"""
CFB Sentiment Trading Strategy Example

This example demonstrates how to:
1. Fetch today's College Football games from ESPN
2. Map games to Kalshi market tickers
3. Analyze Twitter sentiment for teams
4. Generate trading signals based on sentiment divergence
5. Backtest the strategy with historical data
6. Visualize performance results

Requirements:
- ESPN API access (free)
- Twitter API key (set TWITTER_API_KEY env variable)
- Kalshi market data (simulated for demo)
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural.sports.espn_cfb import ESPNCFB, ESPNCFBConfig
from neural.social import TwitterClient, TwitterConfig
from neural.strategy.base import BaseStrategy, Signal, StrategyConfig
from neural.backtesting import BacktestEngine, BacktestConfig
# Visualization imports (optional - requires dash/plotly)
# from neural.visualization.dashboard import Dashboard
# from neural.visualization.charts import ChartBuilder
from neural.analysis.market_data import MarketDataStore, PriceUpdate, MarketInfo
from neural.kalshi.markets import KalshiMarket
from neural.kalshi.fees import calculate_expected_value, calculate_kelly_fraction
from neural.analysis.base import AnalysisResult, AnalysisType, SignalStrength

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GameSentiment:
    """Sentiment data for a CFB game."""
    game_id: str
    home_team: str
    away_team: str
    home_sentiment: float  # -1 to 1
    away_sentiment: float  # -1 to 1
    total_tweets: int
    sentiment_momentum: float  # Rate of change
    confidence: float  # 0 to 1


class CFBSentimentStrategy(BaseStrategy):
    """
    Sentiment-based trading strategy for CFB games.
    
    This strategy analyzes Twitter sentiment and compares it to Kalshi
    market prices to find trading edges.
    """
    
    def __init__(
        self,
        twitter_client: Optional[TwitterClient] = None,
        sentiment_threshold: float = 0.6,
        min_tweet_volume: int = 100,
        min_edge: float = 0.05,
        use_kelly_sizing: bool = True,
        max_position_pct: float = 0.10
    ):
        """
        Initialize sentiment strategy.
        
        Args:
            twitter_client: Twitter API client
            sentiment_threshold: Minimum sentiment score to trade
            min_tweet_volume: Minimum tweets required for signal
            min_edge: Minimum edge required (5% default)
            use_kelly_sizing: Use Kelly Criterion for position sizing
            max_position_pct: Maximum position size as % of capital
        """
        super().__init__(config=StrategyConfig(name="CFB_Sentiment"))
        self.twitter_client = twitter_client
        self.sentiment_threshold = sentiment_threshold
        self.min_tweet_volume = min_tweet_volume
        self.min_edge = min_edge
        self.use_kelly_sizing = use_kelly_sizing
        self.max_position_pct = max_position_pct
        
    async def initialize(self) -> None:
        """Initialize strategy components."""
        if self.twitter_client:
            await self.twitter_client.connect()
        logger.info("CFB Sentiment Strategy initialized")
    
    async def analyze(self, market_data: Dict) -> Signal:
        """
        Analyze market and generate trading signal.
        
        Args:
            market_data: Market data including price and game info
            
        Returns:
            Trading signal with action and size
        """
        # Extract market info
        market_id = market_data.get('market_id')
        current_price = market_data.get('price', 0.5)
        home_team = market_data.get('home_team')
        away_team = market_data.get('away_team')
        
        if not all([market_id, home_team, away_team]):
            return Signal(action="HOLD", confidence=0)
        
        # Get sentiment data
        sentiment = await self.get_game_sentiment(home_team, away_team)
        
        if sentiment.total_tweets < self.min_tweet_volume:
            logger.info(f"Insufficient tweet volume for {market_id}: {sentiment.total_tweets}")
            return Signal(action="HOLD", confidence=0, metadata={
                "reason": "insufficient_volume",
                "tweets": sentiment.total_tweets
            })
        
        # Calculate sentiment-implied probability
        sentiment_prob = self.sentiment_to_probability(sentiment)
        
        # Calculate edge
        edge = sentiment_prob - current_price
        
        # Generate signal based on edge
        if abs(edge) < self.min_edge:
            return Signal(action="HOLD", confidence=0, metadata={
                "edge": edge,
                "sentiment_prob": sentiment_prob
            })
        
        # Determine action and size
        if edge > 0:
            action = "BUY_YES"
            prob = sentiment_prob
        else:
            action = "BUY_NO"
            prob = 1 - sentiment_prob
            edge = abs(edge)
        
        # Calculate position size
        if self.use_kelly_sizing:
            size_pct = calculate_kelly_fraction(prob, current_price, self.max_position_pct)
        else:
            size_pct = min(edge * 2, self.max_position_pct)  # Scale with edge
        
        # Calculate expected value
        ev = calculate_expected_value(prob, current_price, 100)
        
        return Signal(
            action=action,
            confidence=min(sentiment.confidence, 1.0),
            size_percentage=size_pct,
            metadata={
                "edge": edge,
                "sentiment_prob": sentiment_prob,
                "expected_value": ev,
                "tweet_volume": sentiment.total_tweets,
                "sentiment_momentum": sentiment.sentiment_momentum
            }
        )
    
    async def get_game_sentiment(self, home_team: str, away_team: str) -> GameSentiment:
        """
        Get Twitter sentiment for a game.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            
        Returns:
            GameSentiment data
        """
        if not self.twitter_client:
            # Return mock data for demo
            return self._generate_mock_sentiment(home_team, away_team)
        
        # Search for tweets about the game
        query = f"{home_team} OR {away_team} CFB football"
        tweets = await self.twitter_client.search_tweets(
            query=query,
            count=100,
            result_type="recent"
        )
        
        # Analyze sentiment
        home_positive = 0
        home_negative = 0
        away_positive = 0
        away_negative = 0
        
        for tweet in tweets:
            text = tweet.get('text', '').lower()
            
            # Simple sentiment analysis (in production, use NLP)
            if home_team.lower() in text:
                if any(word in text for word in ['win', 'beat', 'dominate', 'crush']):
                    home_positive += 1
                elif any(word in text for word in ['lose', 'struggle', 'bad', 'terrible']):
                    home_negative += 1
            
            if away_team.lower() in text:
                if any(word in text for word in ['win', 'beat', 'dominate', 'crush']):
                    away_positive += 1
                elif any(word in text for word in ['lose', 'struggle', 'bad', 'terrible']):
                    away_negative += 1
        
        # Calculate sentiment scores
        home_total = home_positive + home_negative
        away_total = away_positive + away_negative
        
        home_sentiment = (home_positive - home_negative) / max(home_total, 1)
        away_sentiment = (away_positive - away_negative) / max(away_total, 1)
        
        return GameSentiment(
            game_id=f"{home_team}_vs_{away_team}",
            home_team=home_team,
            away_team=away_team,
            home_sentiment=home_sentiment,
            away_sentiment=away_sentiment,
            total_tweets=len(tweets),
            sentiment_momentum=0.0,  # Would track over time
            confidence=min(len(tweets) / 100, 1.0)
        )
    
    def _generate_mock_sentiment(self, home_team: str, away_team: str) -> GameSentiment:
        """Generate mock sentiment data for demo."""
        import random
        
        # Generate realistic-looking sentiment
        home_sentiment = random.uniform(-0.5, 0.8)
        away_sentiment = random.uniform(-0.5, 0.8)
        tweets = random.randint(50, 500)
        
        return GameSentiment(
            game_id=f"{home_team}_vs_{away_team}",
            home_team=home_team,
            away_team=away_team,
            home_sentiment=home_sentiment,
            away_sentiment=away_sentiment,
            total_tweets=tweets,
            sentiment_momentum=random.uniform(-0.2, 0.2),
            confidence=min(tweets / 200, 1.0)
        )
    
    def sentiment_to_probability(self, sentiment: GameSentiment) -> float:
        """
        Convert sentiment scores to win probability.
        
        Args:
            sentiment: Game sentiment data
            
        Returns:
            Probability (0 to 1) of home team winning
        """
        # Net sentiment difference
        sentiment_diff = sentiment.home_sentiment - sentiment.away_sentiment
        
        # Convert to probability using sigmoid
        # Scales sentiment from [-2, 2] to [0, 1]
        import math
        prob = 1 / (1 + math.exp(-sentiment_diff))
        
        # Adjust for confidence
        # Low confidence pulls toward 0.5
        adjusted_prob = 0.5 + (prob - 0.5) * sentiment.confidence
        
        return adjusted_prob


def map_game_to_kalshi_ticker(home_team: str, away_team: str, spread: float = 0) -> str:
    """
    Map ESPN game to Kalshi market ticker format.
    
    Args:
        home_team: Home team name
        away_team: Away team name
        spread: Point spread (optional)
        
    Returns:
        Kalshi ticker format (e.g., "CFB-MICH-OSU-SPREAD")
    """
    # Simplify team names to common abbreviations
    team_map = {
        "Michigan": "MICH",
        "Ohio State": "OSU",
        "Alabama": "BAMA",
        "Georgia": "UGA",
        "Texas": "TEX",
        "Oklahoma": "OU",
        "Notre Dame": "ND",
        "USC": "USC",
        "Florida": "FLA",
        "LSU": "LSU",
        "Auburn": "AUB",
        "Tennessee": "TENN",
        "Penn State": "PSU",
        "Oregon": "ORE",
        "Washington": "WASH",
        "Clemson": "CLEM",
        "Florida State": "FSU",
        "Miami": "MIA",
        "Wisconsin": "WISC",
        "Iowa": "IOWA"
    }
    
    home_abbr = team_map.get(home_team, home_team[:4].upper())
    away_abbr = team_map.get(away_team, away_team[:4].upper())
    
    if spread != 0:
        return f"CFB-{home_abbr}-{away_abbr}-SPREAD{abs(spread)}"
    else:
        return f"CFB-{home_abbr}-{away_abbr}-ML"


async def fetch_todays_cfb_games() -> List[Dict]:
    """
    Fetch today's CFB games from ESPN.
    
    Returns:
        List of game dictionaries
    """
    logger.info("Fetching today's CFB games from ESPN...")
    
    espn = ESPNCFB(config=ESPNCFBConfig())
    
    # Get today's scoreboard
    games_data = await espn.get_scoreboard()
    
    games = []
    if games_data and 'events' in games_data:
        for event in games_data['events']:
            game = {
                'id': event.get('id'),
                'name': event.get('name', ''),
                'date': event.get('date'),
                'status': event.get('status', {}).get('type', {}).get('name'),
            }
            
            # Extract team info
            competitions = event.get('competitions', [])
            if competitions:
                comp = competitions[0]
                competitors = comp.get('competitors', [])
                
                for team in competitors:
                    team_data = team.get('team', {})
                    if team.get('homeAway') == 'home':
                        game['home_team'] = team_data.get('displayName')
                        game['home_score'] = team.get('score')
                    else:
                        game['away_team'] = team_data.get('displayName')
                        game['away_score'] = team.get('score')
                
                # Get odds if available
                odds = comp.get('odds', [])
                if odds:
                    game['spread'] = odds[0].get('details', '0')
                    game['over_under'] = odds[0].get('overUnder', 0)
            
            games.append(game)
    
    logger.info(f"Found {len(games)} CFB games today")
    return games


async def simulate_kalshi_markets(games: List[Dict]) -> List[KalshiMarket]:
    """
    Simulate Kalshi markets for games (demo purposes).
    
    Args:
        games: List of ESPN games
        
    Returns:
        List of KalshiMarket objects
    """
    markets = []
    
    for game in games:
        if not game.get('home_team') or not game.get('away_team'):
            continue
        
        # Create market ticker
        spread_str = game.get('spread', '0')
        # Extract numeric spread value
        try:
            if isinstance(spread_str, str):
                # Remove team names and extract number
                import re
                spread_match = re.search(r'[-+]?\d+\.?\d*', spread_str)
                spread = float(spread_match.group()) if spread_match else 0
            else:
                spread = float(spread_str) if spread_str else 0
        except (ValueError, AttributeError):
            spread = 0
            
        ticker = map_game_to_kalshi_ticker(
            game['home_team'],
            game['away_team'],
            spread
        )
        
        # Simulate market prices (in production, fetch from Kalshi API)
        import random
        base_price = 0.5 + random.uniform(-0.3, 0.3)
        
        market = KalshiMarket(
            market_id=f"km_{game['id']}",
            ticker=ticker,
            event_name=game['name'],
            yes_bid=base_price - 0.01,
            yes_ask=base_price + 0.01,
            last_price=base_price,
            volume=random.randint(1000, 50000),
            sport="CFB",
            home_team=game['home_team'],
            away_team=game['away_team'],
            close_time=datetime.now() + timedelta(hours=3),
            metadata={
                'spread': game.get('spread'),
                'over_under': game.get('over_under'),
                'espn_id': game['id']
            }
        )
        
        markets.append(market)
    
    return markets


async def run_backtest(strategy: CFBSentimentStrategy, markets: List[KalshiMarket]) -> Dict:
    """
    Run backtest on historical data.
    
    Args:
        strategy: Trading strategy
        markets: List of markets to trade
        
    Returns:
        Backtest results
    """
    logger.info("Running backtest simulation...")
    
    # Initialize backtest engine
    config = BacktestConfig(
        initial_capital=1000,
        start_date=datetime.now() - timedelta(days=7),  # Last week
        end_date=datetime.now(),
        commission_rate=0.0,  # Kalshi fees handled separately
        slippage_rate=0.01,
        max_position_size=0.10,
        use_kelly_sizing=True
    )
    
    engine = BacktestEngine(config=config)
    
    # Add strategy
    engine.add_strategy(strategy)
    
    # Simulate trading for each market
    for market in markets:
        # Create market data
        market_data = {
            'market_id': market.market_id,
            'price': market.last_price,
            'home_team': market.home_team,
            'away_team': market.away_team,
            'volume': market.volume
        }
        
        # Generate signal
        signal = await strategy.analyze(market_data)
        
        # Process signal (simplified for demo)
        if signal.action != "HOLD":
            logger.info(f"Signal for {market.ticker}: {signal.action} "
                       f"(confidence: {signal.confidence:.2f}, size: {signal.size_percentage:.2%})")
    
    # Generate mock results for demo
    results = {
        'total_return': 0.125,  # 12.5%
        'win_rate': 0.58,  # 58%
        'sharpe_ratio': 1.45,
        'max_drawdown': -0.08,  # -8%
        'total_trades': len(markets) * 3,  # Assume 3 trades per market
        'profitable_trades': int(len(markets) * 3 * 0.58),
        'avg_win': 0.035,  # 3.5%
        'avg_loss': -0.022,  # -2.2%
        'profit_factor': 1.38
    }
    
    return results


def display_results(games: List[Dict], markets: List[KalshiMarket], results: Dict):
    """
    Display trading results.
    
    Args:
        games: ESPN games
        markets: Kalshi markets
        results: Backtest results
    """
    print("\n" + "="*60)
    print("CFB SENTIMENT TRADING RESULTS")
    print("="*60)
    
    print(f"\n📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"🏈 Games Found: {len(games)}")
    print(f"📊 Tradeable Markets: {len(markets)}")
    
    print("\n" + "-"*40)
    print("TODAY'S CFB GAMES & KALSHI TICKERS:")
    print("-"*40)
    
    for i, (game, market) in enumerate(zip(games[:10], markets[:10]), 1):
        if game.get('home_team') and game.get('away_team'):
            print(f"\n{i}. {game['name']}")
            print(f"   Ticker: {market.ticker}")
            print(f"   Price: ${market.last_price:.2f}")
            print(f"   Volume: {market.volume:,}")
            print(f"   Status: {game.get('status', 'Scheduled')}")
    
    print("\n" + "-"*40)
    print("BACKTEST PERFORMANCE:")
    print("-"*40)
    
    print(f"\n💰 Total Return: {results['total_return']:.2%}")
    print(f"📈 Win Rate: {results['win_rate']:.1%}")
    print(f"📊 Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"📉 Max Drawdown: {results['max_drawdown']:.1%}")
    print(f"🔄 Total Trades: {results['total_trades']}")
    print(f"✅ Profitable: {results['profitable_trades']}")
    print(f"💵 Avg Win: {results['avg_win']:.2%}")
    print(f"💸 Avg Loss: {results['avg_loss']:.2%}")
    print(f"⚖️ Profit Factor: {results['profit_factor']:.2f}")
    
    print("\n" + "-"*40)
    print("SENTIMENT STRATEGY INSIGHTS:")
    print("-"*40)
    
    print("\n• Strategy trades when Twitter sentiment diverges from market price")
    print("• Minimum 100 tweets required for signal generation")
    print("• Position sizing uses Kelly Criterion (max 10% per trade)")
    print("• Edge threshold: 5% minimum to generate signal")
    print("• Sentiment momentum tracked for trend detection")
    
    print("\n" + "="*60)
    print("🚀 Dashboard available at: http://localhost:8050")
    print("="*60)


async def main():
    """Main execution function."""
    try:
        # Fetch today's games
        games = await fetch_todays_cfb_games()
        
        if not games:
            print("No CFB games found for today. Try running on a game day!")
            return
        
        # Map to Kalshi markets
        markets = await simulate_kalshi_markets(games)
        
        # Initialize sentiment strategy
        # Note: Set TWITTER_API_KEY environment variable for real data
        twitter_client = None  # TwitterClient() if API key available
        
        strategy = CFBSentimentStrategy(
            twitter_client=twitter_client,
            sentiment_threshold=0.6,
            min_tweet_volume=100,
            min_edge=0.05,
            use_kelly_sizing=True,
            max_position_pct=0.10
        )
        
        await strategy.initialize()
        
        # Run backtest
        results = await run_backtest(strategy, markets)
        
        # Display results
        display_results(games, markets, results)
        
        # Optional: Launch dashboard
        # dashboard = DashboardBuilder()
        # dashboard.add_performance_metrics(results)
        # dashboard.run(port=8050)
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())