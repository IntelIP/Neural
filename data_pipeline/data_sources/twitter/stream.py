"""
Twitter Stream Adapter
Real-time sentiment streaming with game context
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from collections import deque

from .client import AsyncTwitterWebSocketClient
from .sentiment import SentimentAnalyzer
from .filters import FilterManager
from .models import (
    Tweet, TweetSentiment, SentimentEvent, GameSentimentSummary
)

logger = logging.getLogger(__name__)


class TwitterStreamAdapter:
    """
    Adapter for streaming Twitter sentiment with game context
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Twitter stream adapter
        
        Args:
            api_key: TwitterAPI.io API key
        """
        # Initialize components
        self.client = AsyncTwitterWebSocketClient(api_key)
        self.filter_manager = FilterManager(api_key)
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # State tracking
        self.is_running = False
        self.current_game: Optional[Dict[str, Any]] = None
        self.game_summaries: Dict[str, GameSentimentSummary] = {}
        
        # Tweet buffer for aggregation
        self.tweet_buffer = deque(maxlen=1000)
        self.sentiment_window = deque(maxlen=100)
        
        # Callbacks
        self.on_sentiment_event: Optional[Callable[[SentimentEvent], None]] = None
        self.on_sentiment_shift: Optional[Callable[[Dict[str, Any]], None]] = None
        self.on_high_impact_tweet: Optional[Callable[[Tweet, TweetSentiment], None]] = None
        
        # Aggregation settings
        self.aggregation_interval = 30  # seconds
        self.aggregation_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the Twitter stream adapter"""
        self.is_running = True
        
        # Set up client callbacks
        self.client.on_tweet(self._handle_tweet)
        self.client.on_error(self._handle_error)
        
        # Connect to WebSocket
        await self.client.connect()
        
        # Start aggregation task
        self.aggregation_task = asyncio.create_task(self._aggregation_loop())
        
        logger.info("Twitter stream adapter started")
    
    async def stop(self):
        """Stop the Twitter stream adapter"""
        self.is_running = False
        
        # Cancel aggregation task
        if self.aggregation_task:
            self.aggregation_task.cancel()
        
        # Disconnect client
        await self.client.disconnect()
        
        # Clean up old filters
        await self.filter_manager.cleanup_old_rules(hours=12)
        
        logger.info("Twitter stream adapter stopped")
    
    async def monitor_game(
        self,
        game_id: str,
        home_team: str,
        away_team: str,
        sport: str = 'nfl',
        players: Optional[List[str]] = None
    ):
        """
        Set up monitoring for a specific game
        
        Args:
            game_id: Unique game identifier
            home_team: Home team name
            away_team: Away team name
            sport: Sport type
            players: Key players to monitor
        """
        # Create game filter
        filter_rule = await self.filter_manager.create_game_filter(
            home_team=home_team,
            away_team=away_team,
            sport=sport,
            players=players
        )
        
        # Set current game context
        self.current_game = {
            'id': game_id,
            'home_team': home_team,
            'away_team': away_team,
            'sport': sport,
            'filter_id': filter_rule.id,
            'start_time': datetime.now()
        }
        
        # Initialize game summary
        self.game_summaries[game_id] = GameSentimentSummary(
            game_id=game_id,
            home_team=home_team,
            away_team=away_team,
            start_time=datetime.now()
        )
        
        logger.info(f"Monitoring game: {home_team} vs {away_team}")
    
    async def monitor_player(self, player_name: str, team: Optional[str] = None):
        """
        Set up monitoring for a specific player
        
        Args:
            player_name: Player's name
            team: Optional team name
        """
        await self.filter_manager.create_player_filter(
            player_name=player_name,
            team=team
        )
        
        logger.info(f"Monitoring player: {player_name}")
    
    def _handle_tweet(self, tweet: Tweet):
        """Handle incoming tweet"""
        try:
            # Add to buffer
            self.tweet_buffer.append(tweet)
            
            # Analyze sentiment
            sentiment = self.sentiment_analyzer.analyze_tweet(tweet)
            
            # Update game summary if applicable
            if self.current_game:
                game_id = self.current_game['id']
                if game_id in self.game_summaries:
                    self.game_summaries[game_id].total_tweets += 1
            
            # Check for high-impact tweets
            if self._is_high_impact(tweet, sentiment):
                logger.info(
                    f"High-impact tweet from @{tweet.author.username}: "
                    f"{tweet.text[:100]}... (Sentiment: {sentiment.score:.2f})"
                )
                
                if self.on_high_impact_tweet:
                    # Run callback in thread to avoid blocking
                    asyncio.create_task(
                        self._run_callback(self.on_high_impact_tweet, tweet, sentiment)
                    )
            
            # Log periodic stats
            if len(self.tweet_buffer) % 100 == 0:
                stats = self.client.get_stats()
                logger.info(
                    f"Stream stats: {stats['tweets_received']} tweets, "
                    f"{stats['tweets_per_minute']:.1f}/min"
                )
        
        except Exception as e:
            logger.error(f"Error handling tweet: {e}")
    
    def _handle_error(self, error: Dict[str, Any]):
        """Handle stream errors"""
        logger.error(f"Stream error: {error}")
    
    def _is_high_impact(self, tweet: Tweet, sentiment: TweetSentiment) -> bool:
        """Check if tweet is high impact"""
        # Verified author with strong sentiment
        if tweet.author.verified and abs(sentiment.score) > 0.7:
            return True
        
        # High credibility author with market keywords
        if tweet.author.credibility_score > 0.7 and sentiment.market_keywords:
            return True
        
        # High engagement with significant sentiment
        if tweet.metrics.engagement_score > 0.8 and sentiment.is_significant:
            return True
        
        # Critical market keywords
        critical_keywords = ['injury', 'injured', 'out', 'suspended', 'ejected']
        if any(kw in tweet.text.lower() for kw in critical_keywords):
            if tweet.author.followers_count > 10000:
                return True
        
        return False
    
    async def _aggregation_loop(self):
        """Periodically aggregate sentiment"""
        while self.is_running:
            try:
                await asyncio.sleep(self.aggregation_interval)
                
                # Get recent tweets
                cutoff_time = datetime.now() - timedelta(seconds=self.aggregation_interval)
                recent_tweets = [
                    t for t in self.tweet_buffer
                    if isinstance(t, Tweet) and t.created_at > cutoff_time
                ]
                
                if recent_tweets:
                    # Aggregate sentiment
                    sentiment_event = self.sentiment_analyzer.aggregate_sentiment(
                        recent_tweets,
                        window_minutes=self.aggregation_interval // 60
                    )
                    
                    # Add to history
                    self.sentiment_analyzer.add_to_history(sentiment_event)
                    
                    # Check for sentiment shift
                    shift = self.sentiment_analyzer.detect_sentiment_shift(sentiment_event)
                    if shift:
                        logger.info(
                            f"Sentiment shift detected: {shift['type']} "
                            f"(magnitude: {shift['magnitude']:.2f})"
                        )
                        
                        if self.on_sentiment_shift:
                            await self._run_callback(self.on_sentiment_shift, shift)
                        
                        # Add to game summary if applicable
                        if self.current_game:
                            game_id = self.current_game['id']
                            if game_id in self.game_summaries:
                                self.game_summaries[game_id].add_key_moment(
                                    description=f"Sentiment shift: {shift['type']}",
                                    sentiment_score=sentiment_event.avg_sentiment_score,
                                    tweet_surge=int(shift.get('tweet_surge', 1))
                                )
                    
                    # Emit sentiment event
                    if self.on_sentiment_event:
                        await self._run_callback(self.on_sentiment_event, sentiment_event)
                    
                    # Log aggregation stats
                    logger.info(
                        f"Sentiment aggregation: {sentiment_event.tweet_count} tweets, "
                        f"Score: {sentiment_event.avg_sentiment_score:.2f}, "
                        f"Impact: {sentiment_event.market_impact.value}"
                    )
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in aggregation loop: {e}")
    
    async def _run_callback(self, callback: Callable, *args):
        """Run callback asynchronously"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args)
            else:
                callback(*args)
        except Exception as e:
            logger.error(f"Error in callback: {e}")
    
    def get_current_sentiment(self) -> Optional[float]:
        """
        Get current sentiment score
        
        Returns:
            Current average sentiment or None
        """
        if not self.sentiment_analyzer.sentiment_history:
            return None
        
        recent = list(self.sentiment_analyzer.sentiment_history)[-1]
        return recent.avg_sentiment_score
    
    def get_sentiment_trend(self, periods: int = 5) -> List[float]:
        """
        Get sentiment trend over recent periods
        
        Args:
            periods: Number of periods to include
            
        Returns:
            List of sentiment scores
        """
        history = list(self.sentiment_analyzer.sentiment_history)[-periods:]
        return [event.avg_sentiment_score for event in history]
    
    def get_game_summary(self, game_id: str) -> Optional[GameSentimentSummary]:
        """
        Get sentiment summary for a game
        
        Args:
            game_id: Game identifier
            
        Returns:
            GameSentimentSummary or None
        """
        return self.game_summaries.get(game_id)
    
    def get_momentum_shifts(self, game_id: str) -> List[Dict[str, Any]]:
        """
        Get momentum shifts for a game
        
        Args:
            game_id: Game identifier
            
        Returns:
            List of momentum shift events
        """
        summary = self.game_summaries.get(game_id)
        if summary:
            return summary.calculate_momentum_shifts()
        return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get stream statistics"""
        client_stats = self.client.get_stats()
        
        return {
            **client_stats,
            'buffer_size': len(self.tweet_buffer),
            'sentiment_windows': len(self.sentiment_analyzer.sentiment_history),
            'current_sentiment': self.get_current_sentiment(),
            'active_game': self.current_game.get('id') if self.current_game else None,
            'games_tracked': len(self.game_summaries)
        }