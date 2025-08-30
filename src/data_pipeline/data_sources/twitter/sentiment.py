"""
Twitter Sentiment Analysis Processor
Advanced sentiment analysis for market correlation
"""

import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
from collections import Counter, deque

from .models import (
    Tweet, TweetSentiment, SentimentType, SentimentEvent,
    MarketImpact
)

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Advanced sentiment analyzer for tweets
    Optimized for sports and market-related content
    """
    
    def __init__(self):
        """Initialize sentiment analyzer"""
        self.sentiment_lexicon = self._build_lexicon()
        self.market_keywords = self._build_market_keywords()
        self.pattern_cache = {}
        
        # Sliding window for aggregation
        self.tweet_buffer = deque(maxlen=1000)
        self.sentiment_history = deque(maxlen=100)
    
    def _build_lexicon(self) -> Dict[str, Dict[str, float]]:
        """Build sentiment lexicon for sports/market context"""
        return {
            'positive': {
                # Game performance
                'win': 0.8, 'winning': 0.7, 'won': 0.9, 'victory': 0.9,
                'dominating': 0.8, 'crushing': 0.7, 'destroying': 0.7,
                'comeback': 0.9, 'rally': 0.7, 'surge': 0.6,
                'lead': 0.5, 'leading': 0.6, 'ahead': 0.5,
                
                # Player status
                'healthy': 0.8, 'cleared': 0.9, 'active': 0.7,
                'returning': 0.7, 'back': 0.6, 'ready': 0.6,
                'impressive': 0.7, 'excellent': 0.8, 'great': 0.7,
                'amazing': 0.8, 'incredible': 0.8, 'fantastic': 0.8,
                
                # Momentum
                'momentum': 0.6, 'rolling': 0.5, 'hot': 0.6,
                'fire': 0.6, '🔥': 0.7, '💪': 0.6, '🚀': 0.7,
                
                # Score
                'touchdown': 0.7, 'goal': 0.6, 'score': 0.5,
                'points': 0.4, 'basket': 0.5
            },
            'negative': {
                # Game performance
                'loss': -0.8, 'losing': -0.7, 'lost': -0.9, 'defeat': -0.9,
                'struggling': -0.7, 'collapsing': -0.8, 'blown': -0.7,
                'trailing': -0.6, 'behind': -0.5, 'down': -0.4,
                
                # Injuries
                'injury': -0.9, 'injured': -0.9, 'hurt': -0.8,
                'questionable': -0.7, 'doubtful': -0.8, 'out': -0.8,
                'sidelined': -0.8, 'benched': -0.6, 'inactive': -0.7,
                'concussion': -0.9, 'torn': -0.9, 'broken': -0.9,
                
                # Performance issues
                'terrible': -0.8, 'awful': -0.8, 'horrible': -0.8,
                'bad': -0.6, 'poor': -0.6, 'weak': -0.5,
                'turnover': -0.7, 'interception': -0.7, 'fumble': -0.7,
                'penalty': -0.5, 'flag': -0.4, 'foul': -0.5,
                
                # Sentiment
                'worried': -0.6, 'concerned': -0.5, 'nervous': -0.5,
                '😢': -0.6, '😭': -0.7, '💔': -0.8, '😡': -0.7
            },
            'intensifiers': {
                'very': 1.3, 'extremely': 1.5, 'absolutely': 1.4,
                'totally': 1.3, 'completely': 1.4, 'really': 1.2,
                'so': 1.2, 'too': 1.2, 'super': 1.3
            },
            'negations': {
                'not': -1, 'no': -1, 'never': -1.2, 'neither': -1,
                'none': -1, 'nobody': -1, 'nothing': -1, 'nowhere': -1,
                "don't": -1, "doesn't": -1, "didn't": -1, "won't": -1,
                "wouldn't": -1, "couldn't": -1, "shouldn't": -1
            }
        }
    
    def _build_market_keywords(self) -> Dict[str, List[str]]:
        """Build market-moving keyword patterns"""
        return {
            'injury_critical': [
                'season ending', 'torn acl', 'torn mcl', 'surgery required',
                'out indefinitely', 'placed on ir', 'career threatening'
            ],
            'injury_major': [
                'ruled out', 'will not play', 'inactive', 'emergency',
                'hospitalized', 'concussion protocol', 'significant injury'
            ],
            'injury_moderate': [
                'questionable', 'game time decision', 'limited practice',
                'doubtful', 'day to day', 'minor injury'
            ],
            'momentum': [
                'turning point', 'game changer', 'momentum shift',
                'comeback', 'collapse', 'choke', 'clutch'
            ],
            'referee': [
                'bad call', 'terrible call', 'referee', 'refs',
                'robbed', 'rigged', 'controversial'
            ],
            'weather': [
                'weather delay', 'rain delay', 'lightning', 'postponed',
                'suspended', 'weather conditions'
            ]
        }
    
    def analyze_tweet(self, tweet: Tweet) -> TweetSentiment:
        """
        Analyze sentiment of a single tweet
        
        Args:
            tweet: Tweet to analyze
            
        Returns:
            TweetSentiment object with analysis results
        """
        text = tweet.text.lower()
        
        # Calculate base sentiment
        score, positive_words, negative_words = self._calculate_sentiment_score(text)
        
        # Adjust for author credibility
        credibility = tweet.author.credibility_score
        score *= (0.5 + credibility * 0.5)  # Scale by credibility
        
        # Adjust for engagement
        engagement = tweet.metrics.engagement_score
        if engagement > 0.5:
            score *= 1.1  # Boost high-engagement tweets
        
        # Detect market keywords
        market_keywords = self._detect_market_keywords(text)
        
        # Determine sentiment type
        if score > 0.3:
            sentiment_type = SentimentType.POSITIVE
        elif score < -0.3:
            sentiment_type = SentimentType.NEGATIVE
        elif positive_words and negative_words:
            sentiment_type = SentimentType.MIXED
        else:
            sentiment_type = SentimentType.NEUTRAL
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            text, score, credibility, engagement
        )
        
        return TweetSentiment(
            tweet_id=tweet.id,
            sentiment_type=sentiment_type,
            score=max(-1, min(1, score)),  # Clamp to [-1, 1]
            confidence=confidence,
            positive_keywords=positive_words,
            negative_keywords=negative_words,
            market_keywords=market_keywords,
            timestamp=tweet.created_at
        )
    
    def _calculate_sentiment_score(self, text: str) -> Tuple[float, List[str], List[str]]:
        """Calculate sentiment score from text"""
        words = text.split()
        score = 0.0
        positive_words = []
        negative_words = []
        
        # Check for negation context
        negation_window = 3
        
        for i, word in enumerate(words):
            # Clean word
            word_clean = re.sub(r'[^\w\s]', '', word)
            
            # Check for negation
            is_negated = False
            for j in range(max(0, i - negation_window), i):
                if words[j] in self.sentiment_lexicon['negations']:
                    is_negated = True
                    break
            
            # Check for intensifier
            intensifier = 1.0
            if i > 0:
                prev_word = words[i-1]
                if prev_word in self.sentiment_lexicon['intensifiers']:
                    intensifier = self.sentiment_lexicon['intensifiers'][prev_word]
            
            # Calculate word sentiment
            if word_clean in self.sentiment_lexicon['positive']:
                word_score = self.sentiment_lexicon['positive'][word_clean]
                if is_negated:
                    word_score *= -1
                    negative_words.append(f"not_{word_clean}")
                else:
                    positive_words.append(word_clean)
                score += word_score * intensifier
            
            elif word_clean in self.sentiment_lexicon['negative']:
                word_score = self.sentiment_lexicon['negative'][word_clean]
                if is_negated:
                    word_score *= -1
                    positive_words.append(f"not_{word_clean}")
                else:
                    negative_words.append(word_clean)
                score += word_score * intensifier
        
        # Normalize by text length
        if len(words) > 0:
            score /= (len(words) ** 0.5)  # Square root normalization
        
        return score, positive_words, negative_words
    
    def _detect_market_keywords(self, text: str) -> List[str]:
        """Detect market-moving keywords in text"""
        detected = []
        
        for category, patterns in self.market_keywords.items():
            for pattern in patterns:
                if pattern.lower() in text:
                    detected.append(f"{category}:{pattern}")
        
        return detected
    
    def _calculate_confidence(
        self,
        text: str,
        score: float,
        credibility: float,
        engagement: float
    ) -> float:
        """Calculate confidence in sentiment analysis"""
        confidence = 0.5  # Base confidence
        
        # Strong sentiment increases confidence
        if abs(score) > 0.7:
            confidence += 0.2
        elif abs(score) > 0.4:
            confidence += 0.1
        
        # Credible authors increase confidence
        confidence += credibility * 0.2
        
        # High engagement increases confidence
        if engagement > 0.7:
            confidence += 0.1
        
        # Market keywords increase confidence
        if self._detect_market_keywords(text):
            confidence += 0.1
        
        # Text length affects confidence
        word_count = len(text.split())
        if word_count > 20:
            confidence += 0.1
        elif word_count < 5:
            confidence -= 0.1
        
        return max(0, min(1, confidence))
    
    def aggregate_sentiment(
        self,
        tweets: List[Tweet],
        window_minutes: int = 5
    ) -> SentimentEvent:
        """
        Aggregate sentiment over a time window
        
        Args:
            tweets: List of tweets to aggregate
            window_minutes: Time window in minutes
            
        Returns:
            SentimentEvent with aggregated sentiment
        """
        if not tweets:
            now = datetime.now()
            return SentimentEvent(
                start_time=now - timedelta(minutes=window_minutes),
                end_time=now,
                tweet_count=0,
                unique_authors=0,
                avg_sentiment_score=0,
                sentiment_distribution={'positive': 0, 'negative': 0, 'neutral': 0},
                market_impact=MarketImpact.NOISE
            )
        
        # Sort tweets by time
        tweets_sorted = sorted(tweets, key=lambda t: t.created_at)
        
        # Analyze each tweet
        sentiments = [self.analyze_tweet(tweet) for tweet in tweets]
        
        # Calculate aggregates
        start_time = tweets_sorted[0].created_at
        end_time = tweets_sorted[-1].created_at
        
        # Sentiment scores
        scores = [s.score for s in sentiments]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # Sentiment distribution
        distribution = Counter(s.sentiment_type.value for s in sentiments)
        
        # Unique authors
        unique_authors = len(set(t.author.id for t in tweets))
        
        # Top tweets by engagement
        tweets_by_engagement = sorted(
            tweets,
            key=lambda t: t.metrics.engagement_score,
            reverse=True
        )
        
        top_positive = [
            t for t in tweets_by_engagement
            if self.analyze_tweet(t).score > 0.3
        ][:3]
        
        top_negative = [
            t for t in tweets_by_engagement
            if self.analyze_tweet(t).score < -0.3
        ][:3]
        
        # Trending keywords
        all_words = ' '.join(t.text.lower() for t in tweets).split()
        word_counts = Counter(
            word for word in all_words
            if len(word) > 4 and not word.startswith('http')
        )
        trending_keywords = [word for word, _ in word_counts.most_common(10)]
        
        # Determine market impact
        market_impact = self._calculate_market_impact(
            tweets, sentiments, unique_authors
        )
        
        return SentimentEvent(
            start_time=start_time,
            end_time=end_time,
            tweet_count=len(tweets),
            unique_authors=unique_authors,
            avg_sentiment_score=avg_score,
            sentiment_distribution=dict(distribution),
            top_positive_tweets=top_positive,
            top_negative_tweets=top_negative,
            market_impact=market_impact,
            trending_keywords=trending_keywords
        )
    
    def _calculate_market_impact(
        self,
        tweets: List[Tweet],
        sentiments: List[TweetSentiment],
        unique_authors: int
    ) -> MarketImpact:
        """Calculate market impact level"""
        # Check for critical keywords
        critical_keywords = any(
            any(kw.startswith('injury_critical') for kw in s.market_keywords)
            for s in sentiments
        )
        
        if critical_keywords:
            return MarketImpact.CRITICAL
        
        # Check for verified high-impact authors
        verified_count = sum(1 for t in tweets if t.author.verified)
        high_follower_count = sum(
            1 for t in tweets
            if t.author.followers_count > 100000
        )
        
        if verified_count >= 3 or high_follower_count >= 2:
            return MarketImpact.HIGH
        
        # Check for high engagement
        total_engagement = sum(t.metrics.engagement_score for t in tweets)
        avg_engagement = total_engagement / len(tweets) if tweets else 0
        
        if avg_engagement > 0.5 or unique_authors > 50:
            return MarketImpact.MEDIUM
        
        # Check for consensus
        if sentiments:
            consensus = abs(sum(s.score for s in sentiments) / len(sentiments))
            if consensus > 0.6 and len(tweets) > 10:
                return MarketImpact.MEDIUM
        
        # Default based on volume
        if len(tweets) > 5:
            return MarketImpact.LOW
        
        return MarketImpact.NOISE
    
    def detect_sentiment_shift(
        self,
        current: SentimentEvent,
        threshold: float = 0.3
    ) -> Optional[Dict[str, Any]]:
        """
        Detect significant sentiment shifts
        
        Args:
            current: Current sentiment event
            threshold: Minimum change to be considered significant
            
        Returns:
            Shift details if detected, None otherwise
        """
        if not self.sentiment_history:
            return None
        
        # Get recent history
        recent = list(self.sentiment_history)[-5:]
        avg_recent = sum(e.avg_sentiment_score for e in recent) / len(recent)
        
        # Check for shift
        change = current.avg_sentiment_score - avg_recent
        
        if abs(change) > threshold:
            # Determine shift type
            if change > 0:
                shift_type = "positive" if change > 0.5 else "moderate_positive"
            else:
                shift_type = "negative" if change < -0.5 else "moderate_negative"
            
            return {
                'type': shift_type,
                'magnitude': abs(change),
                'from_sentiment': avg_recent,
                'to_sentiment': current.avg_sentiment_score,
                'tweet_surge': current.tweets_per_minute / max(
                    sum(e.tweets_per_minute for e in recent) / len(recent), 1
                ),
                'timestamp': current.end_time
            }
        
        return None
    
    def add_to_history(self, sentiment_event: SentimentEvent):
        """Add sentiment event to history"""
        self.sentiment_history.append(sentiment_event)