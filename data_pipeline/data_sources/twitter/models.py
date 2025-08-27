"""
Twitter Data Models
Structured representations for tweets, sentiment, and filter rules
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class MarketImpact(Enum):
    """Market impact levels for tweets"""
    CRITICAL = "critical"  # Breaking news from verified insiders
    HIGH = "high"          # Verified accounts, high engagement
    MEDIUM = "medium"      # Moderate credibility or engagement
    LOW = "low"            # Regular users, low engagement
    NOISE = "noise"        # Likely irrelevant


class SentimentType(Enum):
    """Types of sentiment"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


@dataclass
class Author:
    """Tweet author information"""
    id: str
    username: str
    name: str
    verified: bool = False
    followers_count: int = 0
    following_count: int = 0
    tweet_count: int = 0
    description: Optional[str] = None
    
    @property
    def credibility_score(self) -> float:
        """Calculate author credibility (0-1)"""
        score = 0.0
        
        if self.verified:
            score += 0.4
        
        # Follower tiers
        if self.followers_count > 1000000:
            score += 0.3
        elif self.followers_count > 100000:
            score += 0.25
        elif self.followers_count > 10000:
            score += 0.15
        elif self.followers_count > 1000:
            score += 0.05
        
        # Engagement ratio
        if self.following_count > 0:
            ratio = self.followers_count / self.following_count
            if ratio > 10:
                score += 0.2
            elif ratio > 5:
                score += 0.1
        
        # Activity level
        if self.tweet_count > 10000:
            score += 0.1
        
        return min(score, 1.0)


@dataclass
class TweetMetrics:
    """Tweet engagement metrics"""
    retweet_count: int = 0
    reply_count: int = 0
    like_count: int = 0
    quote_count: int = 0
    impression_count: Optional[int] = None
    
    @property
    def engagement_score(self) -> float:
        """Calculate engagement score"""
        total = (
            self.retweet_count * 3 +  # Retweets weighted highest
            self.quote_count * 2 +     # Quotes show engagement
            self.like_count +           # Likes are easy
            self.reply_count * 1.5     # Replies show discussion
        )
        
        # Normalize to 0-1 scale (1000+ engagement = 1.0)
        return min(total / 1000, 1.0)


@dataclass
class Tweet:
    """Individual tweet data"""
    id: str
    text: str
    author: Author
    created_at: datetime
    metrics: TweetMetrics = field(default_factory=TweetMetrics)
    lang: str = "en"
    conversation_id: Optional[str] = None
    in_reply_to_user_id: Optional[str] = None
    referenced_tweets: List[Dict[str, str]] = field(default_factory=list)
    entities: Dict[str, Any] = field(default_factory=dict)
    context_annotations: List[Dict[str, Any]] = field(default_factory=list)
    possibly_sensitive: bool = False
    
    def is_reply(self) -> bool:
        """Check if tweet is a reply"""
        return self.in_reply_to_user_id is not None
    
    def is_retweet(self) -> bool:
        """Check if tweet is a retweet"""
        return any(ref.get('type') == 'retweeted' for ref in self.referenced_tweets)
    
    def is_quote(self) -> bool:
        """Check if tweet is a quote tweet"""
        return any(ref.get('type') == 'quoted' for ref in self.referenced_tweets)
    
    def get_hashtags(self) -> List[str]:
        """Extract hashtags from tweet"""
        hashtags = self.entities.get('hashtags', [])
        return [tag.get('tag', '') for tag in hashtags]
    
    def get_mentions(self) -> List[str]:
        """Extract mentioned usernames"""
        mentions = self.entities.get('mentions', [])
        return [mention.get('username', '') for mention in mentions]


@dataclass
class TweetSentiment:
    """Sentiment analysis result for a tweet"""
    tweet_id: str
    sentiment_type: SentimentType
    score: float  # -1 to 1
    confidence: float  # 0 to 1
    positive_keywords: List[str] = field(default_factory=list)
    negative_keywords: List[str] = field(default_factory=list)
    market_keywords: List[str] = field(default_factory=list)  # injury, trade, etc.
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def is_significant(self) -> bool:
        """Check if sentiment is significant (high confidence and strong score)"""
        return self.confidence > 0.7 and abs(self.score) > 0.5
    
    @property
    def sentiment_label(self) -> str:
        """Get human-readable sentiment label"""
        if self.score > 0.3:
            return "strongly_positive" if self.score > 0.7 else "positive"
        elif self.score < -0.3:
            return "strongly_negative" if self.score < -0.7 else "negative"
        else:
            return "neutral"


@dataclass
class FilterRule:
    """Twitter stream filter rule"""
    id: Optional[str] = None
    value: str = ""  # The actual filter query
    tag: str = ""    # Human-readable tag
    polling_interval: float = 1.0  # Seconds between polls
    active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    matched_count: int = 0
    
    def to_api_format(self) -> Dict[str, Any]:
        """Convert to TwitterAPI.io format"""
        return {
            "value": self.value,
            "tag": self.tag,
            "polling_interval": self.polling_interval
        }
    
    @staticmethod
    def create_game_filter(
        home_team: str,
        away_team: str,
        players: List[str] = None,
        hashtags: List[str] = None
    ) -> 'FilterRule':
        """Create a filter rule for a specific game"""
        query_parts = []
        
        # Team mentions
        query_parts.append(f"({home_team} OR {away_team})")
        
        # Hashtags
        game_hashtag = f"#{home_team}vs{away_team}"
        all_hashtags = [game_hashtag] + (hashtags or [])
        if all_hashtags:
            hashtag_query = " OR ".join(all_hashtags)
            query_parts.append(f"({hashtag_query})")
        
        # Player mentions with context
        if players:
            player_queries = []
            for player in players:
                player_queries.append(
                    f'("{player}" AND (injury OR injured OR questionable OR '
                    f'doubtful OR out OR return OR touchdown OR interception))'
                )
            query_parts.append(f"({' OR '.join(player_queries)})")
        
        # Combine all parts
        query = " OR ".join(query_parts)
        
        return FilterRule(
            value=query,
            tag=f"game_{home_team}_{away_team}_{datetime.now().strftime('%Y%m%d')}",
            polling_interval=0.5  # Fast polling for live games
        )


@dataclass
class SentimentEvent:
    """Aggregated sentiment over a time window"""
    start_time: datetime
    end_time: datetime
    tweet_count: int
    unique_authors: int
    avg_sentiment_score: float
    sentiment_distribution: Dict[str, int]  # positive/negative/neutral counts
    top_positive_tweets: List[Tweet] = field(default_factory=list)
    top_negative_tweets: List[Tweet] = field(default_factory=list)
    market_impact: MarketImpact = MarketImpact.LOW
    trending_keywords: List[str] = field(default_factory=list)
    
    @property
    def duration_seconds(self) -> float:
        """Get duration of the sentiment window"""
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def tweets_per_minute(self) -> float:
        """Calculate tweet velocity"""
        if self.duration_seconds == 0:
            return 0
        return (self.tweet_count / self.duration_seconds) * 60
    
    @property
    def sentiment_consensus(self) -> float:
        """Calculate how much consensus there is (0=mixed, 1=unanimous)"""
        if self.tweet_count == 0:
            return 0
        
        max_sentiment = max(self.sentiment_distribution.values(), default=0)
        return max_sentiment / self.tweet_count
    
    def is_sentiment_shift(self, prev_score: float, threshold: float = 0.3) -> bool:
        """Check if this represents a significant sentiment shift"""
        return abs(self.avg_sentiment_score - prev_score) > threshold


@dataclass
class MarketSentimentCorrelation:
    """Correlation between Twitter sentiment and market movement"""
    sentiment_event: SentimentEvent
    market_ticker: str
    price_before: float
    price_after: float
    correlation_score: float  # -1 to 1
    lag_seconds: int  # Time lag between sentiment and price movement
    confidence: float  # 0 to 1
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def price_change(self) -> float:
        """Calculate price change"""
        return self.price_after - self.price_before
    
    @property
    def price_change_pct(self) -> float:
        """Calculate percentage price change"""
        if self.price_before == 0:
            return 0
        return (self.price_change / self.price_before) * 100
    
    @property
    def is_correlated(self) -> bool:
        """Check if sentiment and price are correlated"""
        # Positive sentiment should correlate with positive price movement
        sentiment_positive = self.sentiment_event.avg_sentiment_score > 0
        price_positive = self.price_change > 0
        
        return (sentiment_positive == price_positive) and abs(self.correlation_score) > 0.3


@dataclass
class GameSentimentSummary:
    """Summary of sentiment for an entire game"""
    game_id: str
    home_team: str
    away_team: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_tweets: int = 0
    unique_authors: int = 0
    sentiment_events: List[SentimentEvent] = field(default_factory=list)
    key_moments: List[Dict[str, Any]] = field(default_factory=list)  # High-impact moments
    final_sentiment: Optional[float] = None
    winning_team_sentiment: Optional[float] = None
    losing_team_sentiment: Optional[float] = None
    
    def add_key_moment(self, description: str, sentiment_score: float, tweet_surge: int):
        """Add a key moment to the game summary"""
        self.key_moments.append({
            'time': datetime.now(),
            'description': description,
            'sentiment_score': sentiment_score,
            'tweet_surge': tweet_surge
        })
    
    def calculate_momentum_shifts(self) -> List[Dict[str, Any]]:
        """Identify momentum shifts based on sentiment changes"""
        shifts = []
        
        for i in range(1, len(self.sentiment_events)):
            prev = self.sentiment_events[i-1]
            curr = self.sentiment_events[i]
            
            if curr.is_sentiment_shift(prev.avg_sentiment_score):
                shifts.append({
                    'time': curr.start_time,
                    'from_sentiment': prev.avg_sentiment_score,
                    'to_sentiment': curr.avg_sentiment_score,
                    'change': curr.avg_sentiment_score - prev.avg_sentiment_score,
                    'tweet_velocity': curr.tweets_per_minute
                })
        
        return shifts