"""
Reddit Game Thread Adapter
Monitors Reddit game threads for real-time sentiment and reactions
"""

import asyncio
import aiohttp
import asyncpraw
import json
import re
from typing import Dict, Any, Optional, AsyncGenerator, List, Set
from datetime import datetime, timedelta
import logging
from collections import deque
from textblob import TextBlob

from ..core.base_adapter import (
    DataSourceAdapter,
    DataSourceMetadata,
    StandardizedEvent,
    EventType,
    SignalStrength
)

logger = logging.getLogger(__name__)


class RedditAdapter(DataSourceAdapter):
    """
    Reddit game thread monitor
    Analyzes real-time comments for sentiment and significant events
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Reddit adapter
        
        Config should include:
        - client_id: Reddit API client ID
        - client_secret: Reddit API client secret
        - user_agent: User agent string
        - subreddits: List of subreddits to monitor
        - keywords: Keywords to track for events
        - min_comment_karma: Minimum karma for comment to be considered
        """
        super().__init__(config)
        
        self.reddit = None
        self.subreddits = config.get('subreddits', ['nfl', 'nba', 'baseball'])
        self.keywords = config.get('keywords', self._get_default_keywords())
        self.min_karma = config.get('min_comment_karma', 5)
        
        # Sentiment tracking
        self.sentiment_window = deque(maxlen=100)  # Last 100 comments
        self.baseline_sentiment = 0.0
        
        # Rate tracking
        self.comment_rate_window = deque(maxlen=60)  # Last 60 seconds
        self.baseline_rate = 0.0
        
        # Tracked threads
        self.active_threads: Dict[str, Dict] = {}
        
    def get_metadata(self) -> DataSourceMetadata:
        """Return Reddit adapter metadata"""
        return DataSourceMetadata(
            name="Reddit",
            version="1.0.0",
            author="Neural Trading Platform",
            description="Real-time sentiment from Reddit game threads",
            source_type="social",
            latency_ms=1000,
            reliability=0.85,
            requires_auth=True,
            rate_limits={"requests_per_minute": 60},
            supported_sports=["NFL", "NBA", "MLB", "NHL", "Soccer"],
            supported_markets=["sentiment", "momentum"]
        )
    
    def _get_default_keywords(self) -> Dict[str, List[str]]:
        """Get default keywords for event detection"""
        return {
            "touchdown": ["touchdown", "td", "score", "scored"],
            "injury": ["injury", "injured", "hurt", "down", "limping"],
            "turnover": ["interception", "fumble", "turnover", "picked", "strip"],
            "big_play": ["holy shit", "wow", "omg", "incredible", "amazing"],
            "momentum": ["momentum", "turning point", "game changer", "tide turning"],
            "referee": ["ref", "referee", "flag", "penalty", "bullshit call", "rigged"]
        }
    
    async def connect(self) -> bool:
        """Connect to Reddit API"""
        try:
            # Initialize async Reddit client
            self.reddit = asyncpraw.Reddit(
                client_id=self.config['client_id'],
                client_secret=self.config['client_secret'],
                user_agent=self.config.get('user_agent', 'NeuralTradingPlatform/1.0')
            )
            
            # Test connection
            subreddit = await self.reddit.subreddit('nfl')
            async for submission in subreddit.hot(limit=1):
                pass  # Just checking we can fetch
            
            self.logger.info("Connected to Reddit API")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Reddit: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Reddit"""
        if self.reddit:
            await self.reddit.close()
            self.reddit = None
        self.logger.info("Disconnected from Reddit")
    
    async def validate_connection(self) -> bool:
        """Validate Reddit connection"""
        if not self.reddit:
            return False
        
        try:
            # Try to fetch user info
            await self.reddit.user.me()
            return True
        except:
            return False
    
    async def stream(self) -> AsyncGenerator[StandardizedEvent, None]:
        """Stream events from Reddit game threads"""
        while self.is_connected:
            try:
                # Find and monitor game threads
                await self._update_game_threads()
                
                # Stream comments from active threads
                for thread_id, thread_info in self.active_threads.items():
                    async for event in self._monitor_thread(thread_id, thread_info):
                        yield event
                        self._increment_event_count()
                
                # Brief pause between checks
                await asyncio.sleep(2)
                
            except Exception as e:
                self.logger.error(f"Error in Reddit stream: {e}")
                self._increment_error_count()
                await asyncio.sleep(5)
    
    async def _update_game_threads(self):
        """Find and update active game threads"""
        for subreddit_name in self.subreddits:
            try:
                subreddit = await self.reddit.subreddit(subreddit_name)
                
                # Look for game threads in hot and new
                async for submission in subreddit.hot(limit=25):
                    if self._is_game_thread(submission.title):
                        if submission.id not in self.active_threads:
                            self.active_threads[submission.id] = {
                                "title": submission.title,
                                "subreddit": subreddit_name,
                                "created": datetime.fromtimestamp(submission.created_utc),
                                "last_checked": datetime.now(),
                                "comment_count": 0,
                                "teams": self._extract_teams(submission.title)
                            }
                            self.logger.info(f"Tracking game thread: {submission.title}")
                            
            except Exception as e:
                self.logger.error(f"Error updating threads for r/{subreddit_name}: {e}")
    
    def _is_game_thread(self, title: str) -> bool:
        """Check if a post title indicates a game thread"""
        game_indicators = [
            "game thread", "game day thread", "match thread",
            "gamethread", "[game thread]", "gdt", "gdthread"
        ]
        title_lower = title.lower()
        return any(indicator in title_lower for indicator in game_indicators)
    
    def _extract_teams(self, title: str) -> Dict[str, str]:
        """Extract team names from thread title"""
        # Simple pattern matching for "Team1 vs Team2" or "Team1 @ Team2"
        pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:vs\.?|@|at)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        match = re.search(pattern, title)
        
        if match:
            return {
                "away": match.group(1),
                "home": match.group(2)
            }
        return {}
    
    async def _monitor_thread(
        self, 
        thread_id: str, 
        thread_info: Dict
    ) -> AsyncGenerator[StandardizedEvent, None]:
        """
        Monitor a specific game thread for events
        
        Args:
            thread_id: Reddit submission ID
            thread_info: Thread metadata
            
        Yields:
            StandardizedEvent objects
        """
        try:
            submission = await self.reddit.submission(thread_id)
            
            # Get new comments
            submission.comments.replace_more(limit=0)
            await submission.comments.fetch_async()
            
            # Process recent comments
            recent_comments = []
            for comment in submission.comments.list():
                # Skip low-karma comments
                if comment.score < self.min_karma:
                    continue
                
                # Skip old comments
                comment_time = datetime.fromtimestamp(comment.created_utc)
                if datetime.now() - comment_time > timedelta(minutes=2):
                    continue
                
                recent_comments.append({
                    "text": comment.body,
                    "score": comment.score,
                    "time": comment_time,
                    "author": str(comment.author) if comment.author else "deleted"
                })
            
            # Analyze comments for events
            if recent_comments:
                # Check for sentiment shifts
                sentiment_event = self._analyze_sentiment_shift(recent_comments, thread_info)
                if sentiment_event:
                    yield sentiment_event
                
                # Check for keyword events
                for keyword_event in self._detect_keyword_events(recent_comments, thread_info):
                    yield keyword_event
                
                # Check for volume spikes
                volume_event = self._detect_volume_spike(recent_comments, thread_info)
                if volume_event:
                    yield volume_event
                    
        except Exception as e:
            self.logger.error(f"Error monitoring thread {thread_id}: {e}")
    
    def _analyze_sentiment_shift(
        self, 
        comments: List[Dict], 
        thread_info: Dict
    ) -> Optional[StandardizedEvent]:
        """Detect significant sentiment shifts"""
        
        # Calculate current sentiment
        sentiments = []
        for comment in comments:
            blob = TextBlob(comment['text'])
            sentiments.append(blob.sentiment.polarity)
        
        if not sentiments:
            return None
        
        current_sentiment = sum(sentiments) / len(sentiments)
        
        # Update rolling window
        self.sentiment_window.append(current_sentiment)
        
        # Need enough history
        if len(self.sentiment_window) < 10:
            return None
        
        # Calculate baseline (older half of window)
        window_list = list(self.sentiment_window)
        baseline = sum(window_list[:len(window_list)//2]) / (len(window_list)//2)
        recent = sum(window_list[len(window_list)//2:]) / (len(window_list) - len(window_list)//2)
        
        # Detect significant shift
        shift = recent - baseline
        if abs(shift) > 0.3:  # 0.3 sentiment shift threshold
            return StandardizedEvent(
                source="Reddit",
                event_type=EventType.SENTIMENT_SHIFT,
                timestamp=datetime.now(),
                game_id=thread_info.get('title', 'unknown'),
                data={
                    "baseline_sentiment": baseline,
                    "current_sentiment": recent,
                    "shift": shift,
                    "direction": "positive" if shift > 0 else "negative",
                    "comment_sample": [c['text'][:100] for c in comments[:3]],
                    "thread_title": thread_info['title']
                },
                confidence=min(1.0, abs(shift) / 0.5),
                impact="high" if abs(shift) > 0.5 else "medium",
                metadata={
                    "subreddit": thread_info['subreddit'],
                    "teams": thread_info.get('teams', {})
                }
            )
        
        return None
    
    def _detect_keyword_events(
        self, 
        comments: List[Dict], 
        thread_info: Dict
    ) -> List[StandardizedEvent]:
        """Detect events based on keyword mentions"""
        events = []
        
        # Count keyword mentions
        keyword_counts = {category: 0 for category in self.keywords}
        
        for comment in comments:
            text_lower = comment['text'].lower()
            for category, keywords in self.keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    keyword_counts[category] += 1
        
        # Generate events for significant mentions
        for category, count in keyword_counts.items():
            if count >= 3:  # At least 3 mentions
                events.append(StandardizedEvent(
                    source="Reddit",
                    event_type=EventType.SOCIAL_MENTION,
                    timestamp=datetime.now(),
                    game_id=thread_info.get('title', 'unknown'),
                    data={
                        "event_category": category,
                        "mention_count": count,
                        "total_comments": len(comments),
                        "mention_rate": count / max(len(comments), 1),
                        "thread_title": thread_info['title']
                    },
                    confidence=min(1.0, count / 10),
                    impact="high" if category in ["injury", "big_play"] else "medium",
                    metadata={
                        "subreddit": thread_info['subreddit'],
                        "teams": thread_info.get('teams', {})
                    }
                ))
        
        return events
    
    def _detect_volume_spike(
        self, 
        comments: List[Dict], 
        thread_info: Dict
    ) -> Optional[StandardizedEvent]:
        """Detect unusual comment volume (indicates something happened)"""
        
        # Update comment rate
        current_rate = len(comments)
        self.comment_rate_window.append(current_rate)
        
        # Need history
        if len(self.comment_rate_window) < 10:
            return None
        
        # Calculate baseline and current rate
        rates = list(self.comment_rate_window)
        baseline_rate = sum(rates[:-5]) / (len(rates) - 5)
        current_rate = sum(rates[-5:]) / 5
        
        # Detect spike (2x normal rate)
        if current_rate > baseline_rate * 2 and current_rate > 10:
            return StandardizedEvent(
                source="Reddit",
                event_type=EventType.VOLUME_SPIKE,
                timestamp=datetime.now(),
                game_id=thread_info.get('title', 'unknown'),
                data={
                    "baseline_rate": baseline_rate,
                    "current_rate": current_rate,
                    "spike_ratio": current_rate / max(baseline_rate, 1),
                    "thread_title": thread_info['title'],
                    "sample_comments": [c['text'][:100] for c in comments[:5]]
                },
                confidence=min(1.0, (current_rate / baseline_rate - 1) / 3),
                impact="high",
                metadata={
                    "subreddit": thread_info['subreddit'],
                    "teams": thread_info.get('teams', {})
                }
            )
        
        return None
    
    def transform(self, raw_data: Any) -> Optional[StandardizedEvent]:
        """Transform raw Reddit data to standardized event"""
        # Transformation happens in the monitoring methods
        return None


class RedditSentimentAnalyzer:
    """
    Advanced sentiment analysis for Reddit comments
    """
    
    def __init__(self):
        self.emoji_sentiments = {
            "🔥": 1.0, "💪": 0.8, "🎉": 1.0, "😭": -0.8,
            "😤": -0.5, "🤦": -0.7, "💀": -0.9, "🐐": 1.0
        }
        
        self.intensity_words = {
            "insane": 2.0, "incredible": 2.0, "horrible": 2.0,
            "amazing": 1.8, "terrible": 1.8, "perfect": 1.5
        }
    
    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment with sports context
        
        Args:
            text: Comment text
            
        Returns:
            Sentiment scores
        """
        # Base sentiment from TextBlob
        blob = TextBlob(text)
        base_sentiment = blob.sentiment.polarity
        
        # Adjust for emojis
        emoji_score = 0
        emoji_count = 0
        for emoji, score in self.emoji_sentiments.items():
            if emoji in text:
                emoji_score += score
                emoji_count += 1
        
        if emoji_count > 0:
            emoji_sentiment = emoji_score / emoji_count
            # Weight emojis heavily in sports context
            base_sentiment = (base_sentiment + emoji_sentiment * 2) / 3
        
        # Adjust for intensity
        intensity = 1.0
        text_lower = text.lower()
        for word, multiplier in self.intensity_words.items():
            if word in text_lower:
                intensity = max(intensity, multiplier)
        
        final_sentiment = base_sentiment * intensity
        
        # Clip to [-1, 1]
        final_sentiment = max(-1.0, min(1.0, final_sentiment))
        
        return {
            "sentiment": final_sentiment,
            "intensity": intensity,
            "confidence": blob.sentiment.subjectivity
        }