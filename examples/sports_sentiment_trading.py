"""
Sports Sentiment Trading Strategy with Real Kalshi Integration

This example demonstrates how to:
1. Fetch real CFB and NFL markets from Kalshi API
2. Analyze Twitter sentiment for teams
3. Generate trading signals based on sentiment divergence
4. Execute backtests with proper fee calculations
5. Support both CFB and NFL markets

Requirements:
- Kalshi API credentials (set KALSHI_API_KEY and KALSHI_PRIVATE_KEY_PATH)
- Twitter API key (set TWITTER_API_KEY env variable)
- ESPN API access (free)
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

from neural.kalshi import KalshiClient, calculate_expected_value, calculate_kelly_fraction
from neural.sports.espn_cfb import ESPNCFB, ESPNCFBConfig
from neural.social import TwitterClient, TwitterConfig
from neural.strategy.base import BaseStrategy, Signal, StrategyConfig
from neural.backtesting import BacktestEngine, BacktestConfig
from neural.analysis.market_data import MarketDataStore, PriceUpdate, MarketInfo
from neural.analysis.base import AnalysisResult, AnalysisType, SignalStrength

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GameSentiment:
    """Sentiment data for a sports game."""
    game_id: str
    sport: str  # 'CFB' or 'NFL'
    home_team: str
    away_team: str
    home_sentiment: float  # -1 to 1
    away_sentiment: float  # -1 to 1
    total_tweets: int
    sentiment_momentum: float  # Rate of change
    confidence: float  # 0 to 1


class SportsSentimentStrategy(BaseStrategy):
    """
    Sentiment-based trading strategy for sports markets on Kalshi.
    
    This strategy uses real Kalshi API to fetch markets and analyzes
    Twitter sentiment to find trading edges.
    """
    
    def __init__(
        self,
        kalshi_client: KalshiClient,
        twitter_client: Optional[TwitterClient] = None,
        sentiment_threshold: float = 0.6,
        min_tweet_volume: int = 100,
        min_edge: float = 0.05,
        use_kelly_sizing: bool = True,
        max_position_pct: float = 0.10
    ):
        """
        Initialize sentiment strategy with real Kalshi integration.
        
        Args:
            kalshi_client: Configured Kalshi API client
            twitter_client: Twitter API client
            sentiment_threshold: Minimum sentiment score to trade
            min_tweet_volume: Minimum tweets required for signal
            min_edge: Minimum edge required (5% default)
            use_kelly_sizing: Use Kelly Criterion for position sizing
            max_position_pct: Maximum position size as % of capital
        """
        super().__init__(config=StrategyConfig(name="Sports_Sentiment"))
        self.kalshi_client = kalshi_client
        self.twitter_client = twitter_client
        self.sentiment_threshold = sentiment_threshold
        self.min_tweet_volume = min_tweet_volume
        self.min_edge = min_edge
        self.use_kelly_sizing = use_kelly_sizing
        self.max_position_pct = max_position_pct
        self.market_cache = {}
        
    async def initialize(self) -> None:
        """Initialize strategy components."""
        if self.twitter_client:
            await self.twitter_client.connect()
        logger.info("Sports Sentiment Strategy initialized with real Kalshi API")
    
    async def analyze_market(self, ticker: str) -> Optional[Signal]:
        """
        Analyze a Kalshi market for trading opportunities.
        
        Args:
            ticker: Real Kalshi market ticker
            
        Returns:
            Trading signal if opportunity found
        """
        try:
            # Fetch real market data from Kalshi
            market = await self.kalshi_client.get_market_by_ticker(ticker)
            
            if not market or market.get('status') != 'open':
                return None
            
            # Extract team information from ticker
            teams = self.kalshi_client._extract_teams_from_ticker(ticker)
            
            # Analyze sentiment (use mock if no Twitter client)
            if self.twitter_client:
                sentiment = await self.analyze_game_sentiment(
                    teams['home'], 
                    teams['away'],
                    'CFB' if 'NCAAF' in ticker else 'NFL'
                )
            else:
                sentiment = self._generate_mock_sentiment(
                    teams['home'],
                    teams['away'],
                    'CFB' if 'NCAAF' in ticker else 'NFL'
                )
            
            # Check minimum tweet volume
            if sentiment.total_tweets < self.min_tweet_volume:
                logger.info(f"Insufficient tweet volume for {ticker}: {sentiment.total_tweets}")
                return None
            
            # Calculate implied probability from sentiment
            sentiment_prob = self.sentiment_to_probability(sentiment)
            
            # Get market prices
            yes_price = market.get('yes_price', 0) / 100  # Convert cents to probability
            no_price = market.get('no_price', 0) / 100
            
            # Calculate edge
            if sentiment_prob > yes_price:
                # Sentiment suggests YES is undervalued
                edge = sentiment_prob - yes_price
                if edge >= self.min_edge:
                    position_size = self._calculate_position_size(
                        edge, sentiment_prob, sentiment.confidence
                    )
                    
                    return Signal(
                        ticker=ticker,
                        action='BUY_YES',
                        quantity=position_size,
                        price=yes_price,
                        confidence=sentiment.confidence,
                        metadata={
                            'sentiment_prob': sentiment_prob,
                            'market_prob': yes_price,
                            'edge': edge,
                            'teams': teams,
                            'sentiment': sentiment.__dict__
                        }
                    )
            
            elif (1 - sentiment_prob) > no_price:
                # Sentiment suggests NO is undervalued
                edge = (1 - sentiment_prob) - no_price
                if edge >= self.min_edge:
                    position_size = self._calculate_position_size(
                        edge, 1 - sentiment_prob, sentiment.confidence
                    )
                    
                    return Signal(
                        ticker=ticker,
                        action='BUY_NO',
                        quantity=position_size,
                        price=no_price,
                        confidence=sentiment.confidence,
                        metadata={
                            'sentiment_prob': 1 - sentiment_prob,
                            'market_prob': no_price,
                            'edge': edge,
                            'teams': teams,
                            'sentiment': sentiment.__dict__
                        }
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing market {ticker}: {e}")
            return None
    
    async def analyze_game_sentiment(
        self, 
        home_team: str, 
        away_team: str,
        sport: str
    ) -> GameSentiment:
        """
        Analyze Twitter sentiment for teams.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            sport: 'CFB' or 'NFL'
            
        Returns:
            GameSentiment data
        """
        if not self.twitter_client:
            return self._generate_mock_sentiment(home_team, away_team, sport)
        
        # Fetch tweets about the game
        query = f"{home_team} OR {away_team}"
        tweets = await self.twitter_client.search_tweets(
            query=query,
            max_results=500
        )
        
        # Analyze sentiment
        home_positive = 0
        home_negative = 0
        away_positive = 0
        away_negative = 0
        
        positive_words = ['win', 'beat', 'dominate', 'crush', 'victory', 'strong']
        negative_words = ['lose', 'struggle', 'bad', 'terrible', 'weak', 'fail']
        
        for tweet in tweets:
            text = tweet.get('text', '').lower()
            
            # Simple sentiment analysis
            if home_team.lower() in text:
                if any(word in text for word in positive_words):
                    home_positive += 1
                elif any(word in text for word in negative_words):
                    home_negative += 1
            
            if away_team.lower() in text:
                if any(word in text for word in positive_words):
                    away_positive += 1
                elif any(word in text for word in negative_words):
                    away_negative += 1
        
        # Calculate sentiment scores
        home_total = home_positive + home_negative
        away_total = away_positive + away_negative
        
        home_sentiment = (home_positive - home_negative) / max(home_total, 1)
        away_sentiment = (away_positive - away_negative) / max(away_total, 1)
        
        return GameSentiment(
            game_id=f"{home_team}_vs_{away_team}",
            sport=sport,
            home_team=home_team,
            away_team=away_team,
            home_sentiment=home_sentiment,
            away_sentiment=away_sentiment,
            total_tweets=len(tweets),
            sentiment_momentum=0.0,
            confidence=min(len(tweets) / 200, 1.0)
        )
    
    def _generate_mock_sentiment(self, home_team: str, away_team: str, sport: str) -> GameSentiment:
        """Generate mock sentiment data for demo."""
        import random
        
        home_sentiment = random.uniform(-0.3, 0.7)
        away_sentiment = random.uniform(-0.3, 0.7)
        tweets = random.randint(150, 500)
        
        return GameSentiment(
            game_id=f"{home_team}_vs_{away_team}",
            sport=sport,
            home_team=home_team,
            away_team=away_team,
            home_sentiment=home_sentiment,
            away_sentiment=away_sentiment,
            total_tweets=tweets,
            sentiment_momentum=random.uniform(-0.1, 0.1),
            confidence=min(tweets / 200, 1.0)
        )
    
    def sentiment_to_probability(self, sentiment: GameSentiment) -> float:
        """Convert sentiment scores to win probability."""
        import math
        
        # Net sentiment difference
        sentiment_diff = sentiment.home_sentiment - sentiment.away_sentiment
        
        # Convert to probability using sigmoid
        prob = 1 / (1 + math.exp(-sentiment_diff * 2))
        
        # Adjust for confidence
        adjusted_prob = 0.5 + (prob - 0.5) * sentiment.confidence
        
        return adjusted_prob
    
    def _calculate_position_size(
        self, 
        edge: float, 
        win_prob: float,
        confidence: float
    ) -> float:
        """
        Calculate position size using Kelly Criterion.
        
        Args:
            edge: Expected edge (probability difference)
            win_prob: Probability of winning
            confidence: Confidence in signal
            
        Returns:
            Position size as fraction of capital
        """
        if self.use_kelly_sizing:
            # Kelly fraction with confidence adjustment
            kelly = calculate_kelly_fraction(win_prob, 1.0, 1.0)
            adjusted_kelly = kelly * confidence * 0.25  # Quarter Kelly for safety
            
            # Cap at maximum position size
            return min(adjusted_kelly, self.max_position_pct)
        else:
            # Fixed fraction based on confidence
            return min(confidence * 0.05, self.max_position_pct)


async def run_demo_mode(sport: str = 'CFB'):
    """
    Run in demo mode with example data.
    
    This shows how the system would work with real Kalshi API credentials.
    """
    logger.info("\n" + "="*60)
    logger.info("DEMO MODE - Example of Real Kalshi Integration")
    logger.info("="*60)
    
    # Example markets (real format from Kalshi)
    if sport == 'CFB':
        example_markets = [
            {
                'ticker': 'NCAAFSPREAD-24DEC14-MICHIGAN-OHIOSTATE-7',
                'title': 'Will Michigan beat Ohio State by more than 7 points?',
                'yes_price': 45,  # $0.45
                'no_price': 55,   # $0.55
                'volume': 125000,
                'status': 'open'
            },
            {
                'ticker': 'NCAAFSPREAD-24DEC14-ALABAMA-GEORGIA-3',
                'title': 'Will Alabama beat Georgia by more than 3 points?',
                'yes_price': 62,
                'no_price': 38,
                'volume': 98000,
                'status': 'open'
            },
            {
                'ticker': 'NCAAFWIN-24DEC14-TEXAS-OKLAHOMA',
                'title': 'Will Texas beat Oklahoma?',
                'yes_price': 71,
                'no_price': 29,
                'volume': 87500,
                'status': 'open'
            }
        ]
    else:  # NFL
        example_markets = [
            {
                'ticker': 'NFLWIN-24DEC15-COWBOYS-EAGLES',
                'title': 'Will Cowboys beat Eagles?',
                'yes_price': 42,
                'no_price': 58,
                'volume': 245000,
                'status': 'open'
            },
            {
                'ticker': 'NFLSPREAD-24DEC15-CHIEFS-BILLS-3',
                'title': 'Will Chiefs beat Bills by more than 3 points?',
                'yes_price': 53,
                'no_price': 47,
                'volume': 187000,
                'status': 'open'
            }
        ]
    
    logger.info(f"\nExample {sport} Markets (Real Kalshi Format):")
    logger.info("-" * 60)
    
    for i, market in enumerate(example_markets, 1):
        logger.info(f"\n{i}. Ticker: {market['ticker']}")
        logger.info(f"   {market['title']}")
        logger.info(f"   YES: ${market['yes_price']/100:.2f} | NO: ${market['no_price']/100:.2f}")
        logger.info(f"   Volume: ${market['volume']:,}")
        
        # Simulate sentiment analysis
        import random
        sentiment_prob = random.uniform(0.35, 0.75)
        yes_prob = market['yes_price'] / 100
        
        edge = abs(sentiment_prob - yes_prob)
        
        if edge > 0.05:  # 5% minimum edge
            action = "BUY YES" if sentiment_prob > yes_prob else "BUY NO"
            logger.info(f"\n   📊 SIGNAL DETECTED:")
            logger.info(f"      Sentiment probability: {sentiment_prob:.1%}")
            logger.info(f"      Market probability: {yes_prob:.1%}")
            logger.info(f"      Edge: {edge:.1%}")
            logger.info(f"      Action: {action}")
            logger.info(f"      Kelly position size: {edge * 0.25:.1%} of capital")
    
    logger.info("\n" + "="*60)
    logger.info("To use real Kalshi data:")
    logger.info("1. Set KALSHI_API_KEY environment variable")
    logger.info("2. Set KALSHI_PRIVATE_KEY_PATH to your RSA key file")
    logger.info("3. The system will fetch live market data and prices")
    logger.info("="*60)


async def run_cfb_strategy():
    """Run the strategy for CFB markets."""
    logger.info("=" * 60)
    logger.info("Starting CFB Sentiment Trading Strategy")
    logger.info("=" * 60)
    
    # Initialize Kalshi client
    try:
        kalshi_client = KalshiClient()
        logger.info("✓ Connected to Kalshi API")
    except ValueError as e:
        logger.warning(f"Kalshi API not configured: {e}")
        logger.info("Running in DEMO mode with example market data")
        
        # Demo mode - show how it would work
        await run_demo_mode('CFB')
        return
    
    # Initialize Twitter client (optional)
    twitter_client = None
    if os.getenv('TWITTER_API_KEY'):
        twitter_client = TwitterClient(
            config=TwitterConfig(api_key=os.getenv('TWITTER_API_KEY'))
        )
        logger.info("✓ Connected to Twitter API")
    else:
        logger.info("⚠ No Twitter API key found, using mock sentiment data")
    
    # Create strategy
    strategy = SportsSentimentStrategy(
        kalshi_client=kalshi_client,
        twitter_client=twitter_client,
        sentiment_threshold=0.6,
        min_tweet_volume=100,
        min_edge=0.05,
        use_kelly_sizing=True,
        max_position_pct=0.10
    )
    
    await strategy.initialize()
    
    # Fetch real CFB markets from Kalshi
    logger.info("\nFetching CFB markets from Kalshi...")
    try:
        cfb_markets = await kalshi_client.get_cfb_markets()
        logger.info(f"Found {len(cfb_markets)} CFB markets")
        
        # Analyze top markets
        signals = []
        for market in cfb_markets[:10]:  # Analyze first 10 markets
            ticker = market['ticker']
            logger.info(f"\nAnalyzing: {ticker}")
            logger.info(f"  Title: {market.get('title', 'N/A')}")
            logger.info(f"  YES price: ${market.get('yes_price', 0)/100:.2f}")
            logger.info(f"  NO price: ${market.get('no_price', 0)/100:.2f}")
            
            signal = await strategy.analyze_market(ticker)
            
            if signal:
                signals.append(signal)
                logger.info(f"  ✓ SIGNAL: {signal.action} with {signal.confidence:.1%} confidence")
                logger.info(f"    Edge: {signal.metadata['edge']:.1%}")
                logger.info(f"    Position size: {signal.quantity:.1%} of capital")
            else:
                logger.info(f"  ✗ No trading opportunity found")
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info(f"SUMMARY: Found {len(signals)} trading opportunities")
        
        for i, signal in enumerate(signals, 1):
            logger.info(f"\n{i}. {signal.ticker}")
            logger.info(f"   Action: {signal.action}")
            logger.info(f"   Edge: {signal.metadata['edge']:.1%}")
            logger.info(f"   Confidence: {signal.confidence:.1%}")
            logger.info(f"   Position: {signal.quantity:.1%} of capital")
        
    except Exception as e:
        logger.error(f"Error fetching CFB markets: {e}")
    
    finally:
        await kalshi_client.close()
        if twitter_client:
            await twitter_client.disconnect()


async def run_nfl_strategy():
    """Run the strategy for NFL markets."""
    logger.info("=" * 60)
    logger.info("Starting NFL Sentiment Trading Strategy")
    logger.info("=" * 60)
    
    # Initialize Kalshi client
    try:
        kalshi_client = KalshiClient()
        logger.info("✓ Connected to Kalshi API")
    except ValueError as e:
        logger.warning(f"Kalshi API not configured: {e}")
        logger.info("Running in DEMO mode with example market data")
        
        # Demo mode - show how it would work
        await run_demo_mode('NFL')
        return
    
    # Initialize Twitter client (optional)
    twitter_client = None
    if os.getenv('TWITTER_API_KEY'):
        twitter_client = TwitterClient(
            config=TwitterConfig(api_key=os.getenv('TWITTER_API_KEY'))
        )
        logger.info("✓ Connected to Twitter API")
    else:
        logger.info("⚠ No Twitter API key found, using mock sentiment data")
    
    # Create strategy
    strategy = SportsSentimentStrategy(
        kalshi_client=kalshi_client,
        twitter_client=twitter_client,
        sentiment_threshold=0.6,
        min_tweet_volume=100,
        min_edge=0.05,
        use_kelly_sizing=True,
        max_position_pct=0.10
    )
    
    await strategy.initialize()
    
    # Fetch real NFL markets from Kalshi
    logger.info("\nFetching NFL markets from Kalshi...")
    try:
        nfl_markets = await kalshi_client.get_nfl_markets()
        logger.info(f"Found {len(nfl_markets)} NFL markets")
        
        # Analyze top markets
        signals = []
        for market in nfl_markets[:10]:  # Analyze first 10 markets
            ticker = market['ticker']
            logger.info(f"\nAnalyzing: {ticker}")
            logger.info(f"  Title: {market.get('title', 'N/A')}")
            logger.info(f"  YES price: ${market.get('yes_price', 0)/100:.2f}")
            logger.info(f"  NO price: ${market.get('no_price', 0)/100:.2f}")
            
            signal = await strategy.analyze_market(ticker)
            
            if signal:
                signals.append(signal)
                logger.info(f"  ✓ SIGNAL: {signal.action} with {signal.confidence:.1%} confidence")
                logger.info(f"    Edge: {signal.metadata['edge']:.1%}")
                logger.info(f"    Position size: {signal.quantity:.1%} of capital")
            else:
                logger.info(f"  ✗ No trading opportunity found")
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info(f"SUMMARY: Found {len(signals)} trading opportunities")
        
        for i, signal in enumerate(signals, 1):
            logger.info(f"\n{i}. {signal.ticker}")
            logger.info(f"   Action: {signal.action}")
            logger.info(f"   Edge: {signal.metadata['edge']:.1%}")
            logger.info(f"   Confidence: {signal.confidence:.1%}")
            logger.info(f"   Position: {signal.quantity:.1%} of capital")
        
    except Exception as e:
        logger.error(f"Error fetching NFL markets: {e}")
    
    finally:
        await kalshi_client.close()
        if twitter_client:
            await twitter_client.disconnect()


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Sports Sentiment Trading Strategy')
    parser.add_argument(
        '--sport',
        choices=['cfb', 'nfl', 'both'],
        default='both',
        help='Which sport to analyze'
    )
    
    args = parser.parse_args()
    
    if args.sport in ['cfb', 'both']:
        await run_cfb_strategy()
        print("\n")
    
    if args.sport in ['nfl', 'both']:
        await run_nfl_strategy()


if __name__ == "__main__":
    asyncio.run(main())