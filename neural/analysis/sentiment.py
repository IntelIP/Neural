"""
Sentiment Analysis Module for Trading

This module provides comprehensive sentiment analysis capabilities for social media
and sports data, designed to generate trading signals based on public sentiment
and game momentum shifts.
"""

import re
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import asyncio

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("TextBlob not available. Install with: pip install textblob")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("VADER not available. Install with: pip install vaderSentiment")


class SentimentEngine(Enum):
    """Available sentiment analysis engines."""
    VADER = "vader"
    TEXTBLOB = "textblob"
    COMBINED = "combined"
    NEURAL_CUSTOM = "neural_custom"


class SentimentStrength(Enum):
    """Sentiment strength categories."""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"


@dataclass
class SentimentScore:
    """Comprehensive sentiment score with metadata."""
    overall_score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    strength: SentimentStrength
    positive: float
    negative: float
    neutral: float
    compound: float
    subjectivity: float  # 0.0 (objective) to 1.0 (subjective)
    magnitude: float  # Overall intensity
    engine_used: SentimentEngine
    metadata: Dict[str, Any]


@dataclass
class TimeSeries:
    """Time series data for sentiment tracking."""
    timestamps: List[datetime]
    values: List[float]
    window_size: int = 50

    def add_value(self, timestamp: datetime, value: float):
        """Add a new value to the time series."""
        self.timestamps.append(timestamp)
        self.values.append(value)

        # Keep only recent values
        if len(self.values) > self.window_size:
            self.timestamps = self.timestamps[-self.window_size:]
            self.values = self.values[-self.window_size:]

    def get_trend(self, minutes: int = 5) -> float:
        """Calculate sentiment trend over last N minutes."""
        if len(self.values) < 2:
            return 0.0

        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_indices = [
            i for i, ts in enumerate(self.timestamps)
            if ts >= cutoff_time
        ]

        if len(recent_indices) < 2:
            return 0.0

        recent_values = [self.values[i] for i in recent_indices]
        x = np.arange(len(recent_values))
        coefficients = np.polyfit(x, recent_values, 1)
        return coefficients[0]  # Slope indicates trend

    def get_volatility(self, minutes: int = 5) -> float:
        """Calculate sentiment volatility over last N minutes."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_values = [
            self.values[i] for i, ts in enumerate(self.timestamps)
            if ts >= cutoff_time
        ]

        if len(recent_values) < 2:
            return 0.0

        return np.std(recent_values)


class SentimentAnalyzer:
    """
    Comprehensive sentiment analyzer supporting multiple engines.

    Provides sentiment analysis for text data from various sources including
    social media, news, and sports commentary.
    """

    def __init__(
        self,
        engine: SentimentEngine = SentimentEngine.COMBINED,
        custom_lexicon: Optional[Dict[str, float]] = None
    ):
        self.engine = engine
        self.custom_lexicon = custom_lexicon or {}

        # Initialize available engines
        self.vader_analyzer = None
        if VADER_AVAILABLE:
            self.vader_analyzer = SentimentIntensityAnalyzer()

        # Sports-specific sentiment lexicon
        self.sports_lexicon = {
            # Positive sports terms
            'touchdown': 0.8, 'score': 0.6, 'win': 0.7, 'victory': 0.8,
            'champion': 0.9, 'excellent': 0.7, 'amazing': 0.8, 'incredible': 0.8,
            'fantastic': 0.7, 'perfect': 0.8, 'clutch': 0.8, 'dominant': 0.7,
            'brilliant': 0.7, 'spectacular': 0.8, 'outstanding': 0.7,

            # Negative sports terms
            'fumble': -0.6, 'interception': -0.7, 'penalty': -0.4, 'foul': -0.4,
            'miss': -0.5, 'fail': -0.6, 'lose': -0.6, 'defeat': -0.7,
            'terrible': -0.7, 'awful': -0.8, 'disaster': -0.8, 'mistake': -0.5,
            'error': -0.5, 'bad': -0.4, 'poor': -0.4, 'worst': -0.8,

            # Intensity modifiers
            'very': 1.3, 'extremely': 1.5, 'incredibly': 1.4, 'absolutely': 1.3,
            'totally': 1.2, 'completely': 1.2, 'really': 1.1, 'so': 1.1,

            # Excitement indicators
            'wow': 0.6, 'omg': 0.5, 'holy': 0.4, 'insane': 0.7, 'crazy': 0.4,
            'unreal': 0.6, 'sick': 0.5, 'fire': 0.6, 'beast': 0.5
        }

        # Combine lexicons
        self.combined_lexicon = {**self.sports_lexicon, **self.custom_lexicon}

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis."""
        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove mentions and hashtags (but keep the text)
        text = re.sub(r'[@#]\w+', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def _analyze_with_vader(self, text: str) -> Optional[Dict[str, float]]:
        """Analyze sentiment using VADER."""
        if not self.vader_analyzer:
            return None

        scores = self.vader_analyzer.polarity_scores(text)
        return scores

    def _analyze_with_textblob(self, text: str) -> Optional[Dict[str, float]]:
        """Analyze sentiment using TextBlob."""
        if not TEXTBLOB_AVAILABLE:
            return None

        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }

    def _analyze_with_custom(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using custom sports lexicon."""
        words = text.lower().split()
        scores = []
        intensity_modifier = 1.0

        for i, word in enumerate(words):
            # Check for intensity modifiers
            if word in ['very', 'extremely', 'incredibly', 'absolutely']:
                intensity_modifier = self.combined_lexicon.get(word, 1.0)
                continue

            # Get word sentiment
            if word in self.combined_lexicon:
                word_score = self.combined_lexicon[word] * intensity_modifier
                scores.append(word_score)
                intensity_modifier = 1.0  # Reset after use

        if not scores:
            return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}

        compound = np.mean(scores)
        positive = np.mean([s for s in scores if s > 0]) if any(s > 0 for s in scores) else 0.0
        negative = abs(np.mean([s for s in scores if s < 0])) if any(s < 0 for s in scores) else 0.0
        neutral = 1.0 - (positive + negative)

        return {
            'compound': compound,
            'pos': positive,
            'neu': max(0.0, neutral),
            'neg': negative
        }

    def analyze_text(self, text: str) -> SentimentScore:
        """
        Analyze sentiment of a text string.

        Args:
            text: Text to analyze

        Returns:
            SentimentScore with comprehensive metrics
        """
        if not text or not text.strip():
            return SentimentScore(
                overall_score=0.0,
                confidence=0.0,
                strength=SentimentStrength.NEUTRAL,
                positive=0.0,
                negative=0.0,
                neutral=1.0,
                compound=0.0,
                subjectivity=0.0,
                magnitude=0.0,
                engine_used=self.engine,
                metadata={'text_length': 0, 'word_count': 0}
            )

        preprocessed_text = self._preprocess_text(text)
        word_count = len(preprocessed_text.split())

        # Get scores from different engines
        vader_scores = self._analyze_with_vader(preprocessed_text)
        textblob_scores = self._analyze_with_textblob(preprocessed_text)
        custom_scores = self._analyze_with_custom(preprocessed_text)

        # Combine scores based on engine choice
        if self.engine == SentimentEngine.VADER and vader_scores:
            compound = vader_scores['compound']
            positive = vader_scores['pos']
            negative = vader_scores['neg']
            neutral = vader_scores['neu']
            subjectivity = 0.5  # VADER doesn't provide subjectivity
        elif self.engine == SentimentEngine.TEXTBLOB and textblob_scores:
            compound = textblob_scores['polarity']
            positive = max(0, compound)
            negative = max(0, -compound)
            neutral = 1 - (positive + negative)
            subjectivity = textblob_scores['subjectivity']
        elif self.engine == SentimentEngine.COMBINED:
            # Weighted combination of available engines
            weights = []
            compounds = []

            if vader_scores:
                weights.append(0.4)
                compounds.append(vader_scores['compound'])
            if textblob_scores:
                weights.append(0.3)
                compounds.append(textblob_scores['polarity'])
            if custom_scores:
                weights.append(0.3)
                compounds.append(custom_scores['compound'])

            if compounds:
                compound = np.average(compounds, weights=weights)
            else:
                compound = 0.0

            # Use VADER scores if available, otherwise estimate
            if vader_scores:
                positive = vader_scores['pos']
                negative = vader_scores['neg']
                neutral = vader_scores['neu']
            else:
                positive = max(0, compound)
                negative = max(0, -compound)
                neutral = 1 - (positive + negative)

            subjectivity = textblob_scores['subjectivity'] if textblob_scores else 0.5
        else:
            # Use custom engine
            compound = custom_scores['compound']
            positive = custom_scores['pos']
            negative = custom_scores['neg']
            neutral = custom_scores['neu']
            subjectivity = 0.5

        # Calculate overall score and confidence
        overall_score = compound
        magnitude = abs(compound)

        # Confidence based on magnitude, text length, and subjectivity
        confidence = min(1.0, magnitude * (1 + min(word_count / 20, 1.0)) * (1 - subjectivity * 0.5))

        # Determine sentiment strength
        if compound >= 0.5:
            strength = SentimentStrength.VERY_POSITIVE
        elif compound >= 0.1:
            strength = SentimentStrength.POSITIVE
        elif compound <= -0.5:
            strength = SentimentStrength.VERY_NEGATIVE
        elif compound <= -0.1:
            strength = SentimentStrength.NEGATIVE
        else:
            strength = SentimentStrength.NEUTRAL

        return SentimentScore(
            overall_score=overall_score,
            confidence=confidence,
            strength=strength,
            positive=positive,
            negative=negative,
            neutral=neutral,
            compound=compound,
            subjectivity=subjectivity,
            magnitude=magnitude,
            engine_used=self.engine,
            metadata={
                'text_length': len(text),
                'word_count': word_count,
                'preprocessed_length': len(preprocessed_text),
                'engines_used': [
                    'vader' if vader_scores else None,
                    'textblob' if textblob_scores else None,
                    'custom'
                ]
            }
        )

    def analyze_batch(self, texts: List[str]) -> List[SentimentScore]:
        """Analyze sentiment for a batch of texts."""
        return [self.analyze_text(text) for text in texts]

    def get_aggregate_sentiment(
        self,
        texts: List[str],
        weights: Optional[List[float]] = None
    ) -> SentimentScore:
        """
        Get aggregate sentiment from multiple texts.

        Args:
            texts: List of texts to analyze
            weights: Optional weights for each text

        Returns:
            Aggregated sentiment score
        """
        if not texts:
            return self.analyze_text("")

        scores = self.analyze_batch(texts)

        if weights and len(weights) == len(scores):
            overall = np.average([s.overall_score for s in scores], weights=weights)
            confidence = np.average([s.confidence for s in scores], weights=weights)
            magnitude = np.average([s.magnitude for s in scores], weights=weights)
        else:
            overall = np.mean([s.overall_score for s in scores])
            confidence = np.mean([s.confidence for s in scores])
            magnitude = np.mean([s.magnitude for s in scores])

        # Determine aggregate strength
        if overall >= 0.5:
            strength = SentimentStrength.VERY_POSITIVE
        elif overall >= 0.1:
            strength = SentimentStrength.POSITIVE
        elif overall <= -0.5:
            strength = SentimentStrength.VERY_NEGATIVE
        elif overall <= -0.1:
            strength = SentimentStrength.NEGATIVE
        else:
            strength = SentimentStrength.NEUTRAL

        return SentimentScore(
            overall_score=overall,
            confidence=confidence,
            strength=strength,
            positive=np.mean([s.positive for s in scores]),
            negative=np.mean([s.negative for s in scores]),
            neutral=np.mean([s.neutral for s in scores]),
            compound=overall,
            subjectivity=np.mean([s.subjectivity for s in scores]),
            magnitude=magnitude,
            engine_used=self.engine,
            metadata={
                'text_count': len(texts),
                'total_length': sum(s.metadata['text_length'] for s in scores),
                'individual_scores': [s.overall_score for s in scores]
            }
        )


class GameSentimentTracker:
    """
    Tracks sentiment over time for a specific game or event.

    Combines Twitter sentiment, ESPN play momentum, and other sources
    to provide comprehensive sentiment tracking.
    """

    def __init__(
        self,
        game_id: str,
        teams: List[str],
        sentiment_analyzer: Optional[SentimentAnalyzer] = None,
        window_minutes: int = 10
    ):
        self.game_id = game_id
        self.teams = teams
        self.analyzer = sentiment_analyzer or SentimentAnalyzer(SentimentEngine.COMBINED)
        self.window_minutes = window_minutes

        # Time series tracking
        self.twitter_sentiment = TimeSeries([], [], window_size=100)
        self.espn_momentum = TimeSeries([], [], window_size=50)
        self.combined_sentiment = TimeSeries([], [], window_size=100)

        # Team-specific tracking
        self.team_sentiment = {team: TimeSeries([], [], window_size=50) for team in teams}

    def add_twitter_data(self, tweets: List[Dict[str, Any]]) -> None:
        """Process and add Twitter sentiment data."""
        if not tweets:
            return

        # Extract text and metadata
        texts = [tweet['text'] for tweet in tweets]
        weights = [
            1.0 + (tweet.get('metrics', {}).get('like_count', 0) / 100)
            for tweet in tweets
        ]

        # Analyze aggregate sentiment
        aggregate_score = self.analyzer.get_aggregate_sentiment(texts, weights)
        timestamp = datetime.now()

        self.twitter_sentiment.add_value(timestamp, aggregate_score.overall_score)

        # Team-specific sentiment
        for team in self.teams:
            team_tweets = [
                tweet['text'] for tweet in tweets
                if team.lower() in tweet['text'].lower()
            ]

            if team_tweets:
                team_score = self.analyzer.get_aggregate_sentiment(team_tweets)
                self.team_sentiment[team].add_value(timestamp, team_score.overall_score)

    def add_espn_data(self, espn_data: Dict[str, Any]) -> None:
        """Process and add ESPN momentum data."""
        timestamp = datetime.now()

        # Extract momentum from ESPN data
        momentum_home = espn_data.get('momentum_home', 0.0)
        momentum_away = espn_data.get('momentum_away', 0.0)

        # Use overall momentum as ESPN sentiment
        overall_momentum = (momentum_home + momentum_away) / 2
        self.espn_momentum.add_value(timestamp, overall_momentum)

        # Team-specific momentum
        if len(self.teams) >= 2:
            self.team_sentiment[self.teams[0]].add_value(timestamp, momentum_home)
            self.team_sentiment[self.teams[1]].add_value(timestamp, momentum_away)

    def get_current_sentiment(self) -> Dict[str, Any]:
        """Get current comprehensive sentiment metrics."""
        now = datetime.now()

        # Calculate combined sentiment (weighted average)
        twitter_weight = 0.6
        espn_weight = 0.4

        twitter_current = self.twitter_sentiment.values[-1] if self.twitter_sentiment.values else 0.0
        espn_current = self.espn_momentum.values[-1] if self.espn_momentum.values else 0.0

        combined_score = (twitter_current * twitter_weight + espn_current * espn_weight)
        self.combined_sentiment.add_value(now, combined_score)

        return {
            'timestamp': now,
            'twitter_sentiment': twitter_current,
            'espn_momentum': espn_current,
            'combined_sentiment': combined_score,
            'twitter_trend': self.twitter_sentiment.get_trend(5),
            'espn_trend': self.espn_momentum.get_trend(5),
            'combined_trend': self.combined_sentiment.get_trend(5),
            'twitter_volatility': self.twitter_sentiment.get_volatility(5),
            'sentiment_strength': self._classify_sentiment(combined_score),
            'team_sentiment': {
                team: {
                    'current': ts.values[-1] if ts.values else 0.0,
                    'trend': ts.get_trend(5),
                    'volatility': ts.get_volatility(5)
                }
                for team, ts in self.team_sentiment.items()
            }
        }

    def _classify_sentiment(self, score: float) -> str:
        """Classify sentiment score into categories."""
        if score >= 0.5:
            return "very_positive"
        elif score >= 0.2:
            return "positive"
        elif score <= -0.5:
            return "very_negative"
        elif score <= -0.2:
            return "negative"
        else:
            return "neutral"

    def get_trading_signal_strength(self) -> float:
        """
        Calculate trading signal strength based on sentiment metrics.

        Returns:
            Signal strength from 0.0 to 1.0
        """
        current = self.get_current_sentiment()

        # Factors that increase signal strength
        magnitude = abs(current['combined_sentiment'])
        trend_strength = abs(current['combined_trend'])
        volatility = current['twitter_volatility']

        # Strong sentiment with strong trend and low volatility = high signal
        signal_strength = (
            magnitude * 0.5 +
            trend_strength * 0.3 +
            max(0, 0.5 - volatility) * 0.2  # Lower volatility = higher confidence
        )

        return min(1.0, signal_strength)


# Factory function for easy setup
def create_sentiment_analyzer(
    engine: str = "combined",
    custom_words: Optional[Dict[str, float]] = None
) -> SentimentAnalyzer:
    """
    Create sentiment analyzer with specified configuration.

    Args:
        engine: Engine type ("vader", "textblob", "combined", "custom")
        custom_words: Custom word-sentiment mappings

    Returns:
        Configured sentiment analyzer
    """
    engine_enum = SentimentEngine(engine)
    return SentimentAnalyzer(engine_enum, custom_words)


# Example usage
if __name__ == "__main__":
    # Create analyzer
    analyzer = create_sentiment_analyzer("combined")

    # Test sentiment analysis
    test_texts = [
        "What an amazing touchdown! Best play of the game!",
        "Terrible fumble, this team is playing awful",
        "Great defensive play, they're dominating the field",
        "Missed field goal, another disappointing performance"
    ]

    for text in test_texts:
        score = analyzer.analyze_text(text)
        print(f"Text: {text}")
        print(f"Score: {score.overall_score:.3f} ({score.strength.value})")
        print(f"Confidence: {score.confidence:.3f}")
        print("---")