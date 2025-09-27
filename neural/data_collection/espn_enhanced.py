"""
Enhanced ESPN Integration with GameCast Real-time Data

This module extends the basic ESPN integration to provide real-time play-by-play
data, game flow analysis, and momentum tracking for sentiment-based trading.
"""

import asyncio
import re
from datetime import datetime
from typing import Dict, List, Optional, AsyncGenerator, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from .rest_api import RestApiSource
from .base import DataSource


class PlayType(Enum):
    """Types of plays that can affect game momentum."""
    TOUCHDOWN = "touchdown"
    FIELD_GOAL = "field_goal"
    INTERCEPTION = "interception"
    FUMBLE = "fumble"
    SAFETY = "safety"
    TURNOVER_ON_DOWNS = "turnover_on_downs"
    PUNT = "punt"
    KICKOFF = "kickoff"
    TWO_POINT = "two_point"
    PENALTY = "penalty"
    TIMEOUT = "timeout"
    END_QUARTER = "end_quarter"
    INJURY = "injury"
    UNKNOWN = "unknown"


class MomentumDirection(Enum):
    """Direction of momentum shift."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class PlayData:
    """Structured play-by-play data."""
    id: str
    sequence_number: int
    quarter: int
    time_remaining: str
    down: Optional[int]
    distance: Optional[int]
    yard_line: Optional[str]
    play_type: PlayType
    description: str
    team: Optional[str]
    scoring_play: bool
    turnover: bool
    momentum_score: float  # -1 to 1, calculated based on play impact
    momentum_direction: MomentumDirection
    timestamp: datetime
    raw_data: Dict[str, Any]


@dataclass
class GameState:
    """Current game state for momentum analysis."""
    game_id: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    quarter: int
    time_remaining: str
    possession: Optional[str]
    down: Optional[int]
    distance: Optional[int]
    yard_line: Optional[str]
    game_status: str
    recent_plays: List[PlayData]
    momentum_home: float  # Running momentum score for home team
    momentum_away: float  # Running momentum score for away team


class ESPNGameCastSource(DataSource):
    """
    Enhanced ESPN data source with real-time GameCast data.

    Provides play-by-play analysis, momentum tracking, and sentiment scoring
    for individual plays and game flow.
    """

    def __init__(
        self,
        game_id: str,
        sport: str = "football/nfl",
        poll_interval: float = 5.0,
        momentum_window: int = 10
    ):
        super().__init__(name=f"espn_gamecast_{game_id}")
        self.game_id = game_id
        self.sport = sport
        self.poll_interval = poll_interval
        self.momentum_window = momentum_window
        self.last_play_id = None
        self.game_state = None

        # Momentum keywords for play analysis
        self.positive_keywords = [
            'touchdown', 'score', 'interception', 'fumble recovery',
            'sack', 'big gain', 'converted', 'first down', 'field goal'
        ]
        self.negative_keywords = [
            'fumble', 'interception', 'sacked', 'penalty', 'incomplete',
            'punt', 'turnover', 'missed', 'blocked'
        ]

    async def connect(self) -> None:
        """Establish connection and initialize game state."""
        self._connected = True

    async def disconnect(self) -> None:
        """Close connection."""
        self._connected = False

    async def get_game_summary(self) -> Dict[str, Any]:
        """Get current game summary data."""
        import aiohttp

        url = f"https://site.api.espn.com/apis/site/v2/sports/{self.sport}/summary"
        params = {"event": self.game_id}

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise RuntimeError(f"ESPN API error {response.status}")

    async def get_play_by_play(self) -> Dict[str, Any]:
        """Get detailed play-by-play data."""
        import aiohttp

        url = f"https://site.api.espn.com/apis/site/v2/sports/{self.sport}/playbyplay"
        params = {"event": self.game_id}

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise RuntimeError(f"ESPN Play-by-Play API error {response.status}")

    def _extract_play_type(self, play_text: str) -> PlayType:
        """Extract play type from play description."""
        play_text_lower = play_text.lower()

        if any(word in play_text_lower for word in ['touchdown', 'td']):
            return PlayType.TOUCHDOWN
        elif 'field goal' in play_text_lower:
            return PlayType.FIELD_GOAL
        elif 'interception' in play_text_lower:
            return PlayType.INTERCEPTION
        elif 'fumble' in play_text_lower and 'recovered' not in play_text_lower:
            return PlayType.FUMBLE
        elif 'safety' in play_text_lower:
            return PlayType.SAFETY
        elif 'punt' in play_text_lower:
            return PlayType.PUNT
        elif 'two point' in play_text_lower or '2-pt' in play_text_lower:
            return PlayType.TWO_POINT
        elif 'penalty' in play_text_lower:
            return PlayType.PENALTY
        elif 'timeout' in play_text_lower:
            return PlayType.TIMEOUT
        elif 'end of' in play_text_lower and 'quarter' in play_text_lower:
            return PlayType.END_QUARTER
        elif 'injury' in play_text_lower:
            return PlayType.INJURY
        else:
            return PlayType.UNKNOWN

    def _calculate_momentum_score(self, play: Dict[str, Any]) -> Tuple[float, MomentumDirection]:
        """
        Calculate momentum score for a play (-1 to 1).

        Args:
            play: Raw play data from ESPN API

        Returns:
            Tuple of (momentum_score, momentum_direction)
        """
        text = play.get('text', '').lower()
        play_type = self._extract_play_type(text)

        # Base momentum scores by play type
        base_scores = {
            PlayType.TOUCHDOWN: 0.8,
            PlayType.FIELD_GOAL: 0.3,
            PlayType.INTERCEPTION: -0.7,
            PlayType.FUMBLE: -0.6,
            PlayType.SAFETY: -0.9,
            PlayType.TURNOVER_ON_DOWNS: -0.4,
            PlayType.PUNT: -0.1,
            PlayType.TWO_POINT: 0.5,
            PlayType.PENALTY: -0.2,
            PlayType.UNKNOWN: 0.0
        }

        base_score = base_scores.get(play_type, 0.0)

        # Adjust based on keywords
        positive_count = sum(1 for keyword in self.positive_keywords if keyword in text)
        negative_count = sum(1 for keyword in self.negative_keywords if keyword in text)

        keyword_adjustment = (positive_count - negative_count) * 0.1
        final_score = max(-1.0, min(1.0, base_score + keyword_adjustment))

        # Determine direction
        if final_score > 0.1:
            direction = MomentumDirection.POSITIVE
        elif final_score < -0.1:
            direction = MomentumDirection.NEGATIVE
        else:
            direction = MomentumDirection.NEUTRAL

        return final_score, direction

    def _process_play(self, play: Dict[str, Any], drive_info: Dict[str, Any] = None) -> PlayData:
        """Process raw play data into structured format."""
        play_id = play.get('id', str(play.get('sequenceNumber', 0)))
        description = play.get('text', '')

        momentum_score, momentum_direction = self._calculate_momentum_score(play)

        return PlayData(
            id=play_id,
            sequence_number=play.get('sequenceNumber', 0),
            quarter=play.get('period', {}).get('number', 0),
            time_remaining=play.get('clock', {}).get('displayValue', ''),
            down=play.get('start', {}).get('down'),
            distance=play.get('start', {}).get('distance'),
            yard_line=play.get('start', {}).get('yardLine'),
            play_type=self._extract_play_type(description),
            description=description,
            team=play.get('start', {}).get('team', {}).get('abbreviation'),
            scoring_play=play.get('scoringPlay', False),
            turnover='turnover' in description.lower() or 'interception' in description.lower(),
            momentum_score=momentum_score,
            momentum_direction=momentum_direction,
            timestamp=datetime.now(),
            raw_data=play
        )

    def _update_game_state(self, game_data: Dict[str, Any], plays: List[PlayData]) -> GameState:
        """Update game state with latest data."""
        header = game_data.get('header', {})
        competitions = header.get('competitions', [{}])

        if competitions:
            competition = competitions[0]
            competitors = competition.get('competitors', [])

            home_team = next((c for c in competitors if c.get('homeAway') == 'home'), {})
            away_team = next((c for c in competitors if c.get('homeAway') == 'away'), {})

            # Calculate running momentum
            recent_plays = plays[-self.momentum_window:] if len(plays) > self.momentum_window else plays

            home_momentum = sum(
                p.momentum_score for p in recent_plays
                if p.team == home_team.get('team', {}).get('abbreviation')
            ) / len(recent_plays) if recent_plays else 0.0

            away_momentum = sum(
                p.momentum_score for p in recent_plays
                if p.team == away_team.get('team', {}).get('abbreviation')
            ) / len(recent_plays) if recent_plays else 0.0

            return GameState(
                game_id=self.game_id,
                home_team=home_team.get('team', {}).get('displayName', ''),
                away_team=away_team.get('team', {}).get('displayName', ''),
                home_score=int(home_team.get('score', '0')),
                away_score=int(away_team.get('score', '0')),
                quarter=competition.get('status', {}).get('period', 0),
                time_remaining=competition.get('status', {}).get('displayClock', ''),
                possession=None,  # Would need additional parsing
                down=None,
                distance=None,
                yard_line=None,
                game_status=competition.get('status', {}).get('type', {}).get('description', ''),
                recent_plays=recent_plays,
                momentum_home=home_momentum,
                momentum_away=away_momentum
            )

        return None

    async def collect(self) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Continuously collect ESPN GameCast data.

        Yields:
            Processed game data with play analysis and momentum scores
        """
        all_plays = []

        while self._connected:
            try:
                # Get current game data
                game_data = await self.get_game_summary()

                try:
                    # Get play-by-play data
                    pbp_data = await self.get_play_by_play()

                    # Process new plays
                    new_plays = []
                    if 'drives' in pbp_data:
                        for drive in pbp_data['drives']:
                            if 'plays' in drive:
                                for play in drive['plays']:
                                    play_id = play.get('id', str(play.get('sequenceNumber', 0)))

                                    # Only process new plays
                                    if self.last_play_id is None or play_id != self.last_play_id:
                                        processed_play = self._process_play(play, drive)
                                        new_plays.append(processed_play)
                                        all_plays.append(processed_play)

                                        # Update last play ID
                                        if processed_play.sequence_number > (
                                            int(self.last_play_id) if self.last_play_id and self.last_play_id.isdigit() else 0
                                        ):
                                            self.last_play_id = play_id

                    # Update game state
                    self.game_state = self._update_game_state(game_data, all_plays)

                    # Yield data if we have new plays or game state updates
                    if new_plays or self.game_state:
                        yield {
                            'source': 'espn_gamecast',
                            'game_id': self.game_id,
                            'timestamp': datetime.now(),
                            'game_state': self.game_state.__dict__ if self.game_state else None,
                            'new_plays': [play.__dict__ for play in new_plays],
                            'momentum_home': self.game_state.momentum_home if self.game_state else 0.0,
                            'momentum_away': self.game_state.momentum_away if self.game_state else 0.0,
                            'total_plays': len(all_plays),
                            'raw_game_data': game_data
                        }

                except Exception as pbp_error:
                    # If play-by-play fails, still provide game state
                    print(f"Play-by-play error: {pbp_error}")
                    yield {
                        'source': 'espn_gamecast',
                        'game_id': self.game_id,
                        'timestamp': datetime.now(),
                        'game_state': None,
                        'new_plays': [],
                        'momentum_home': 0.0,
                        'momentum_away': 0.0,
                        'total_plays': len(all_plays),
                        'raw_game_data': game_data,
                        'error': str(pbp_error)
                    }

            except Exception as e:
                print(f"ESPN GameCast error: {e}")
                yield {
                    'source': 'espn_gamecast',
                    'game_id': self.game_id,
                    'timestamp': datetime.now(),
                    'error': str(e)
                }

            await asyncio.sleep(self.poll_interval)


class ESPNSentimentSource(ESPNGameCastSource):
    """
    ESPN source focused on sentiment extraction from play descriptions.

    Extends GameCast source with enhanced sentiment analysis of plays.
    """

    def __init__(self, game_id: str, sport: str = "football/nfl", poll_interval: float = 5.0):
        super().__init__(game_id, sport, poll_interval)

        # Enhanced sentiment keywords
        self.excitement_words = [
            'amazing', 'incredible', 'fantastic', 'spectacular', 'huge', 'big',
            'clutch', 'perfect', 'brilliant', 'outstanding', 'explosive'
        ]

        self.negative_words = [
            'terrible', 'awful', 'disaster', 'mistake', 'error', 'bad',
            'poor', 'miss', 'fail', 'drop', 'overthrow', 'underthrow'
        ]

        self.intensity_words = [
            'crushing', 'devastating', 'dominant', 'powerful', 'fierce',
            'aggressive', 'massive', 'enormous', 'critical', 'crucial'
        ]

    def _extract_play_sentiment(self, play_text: str) -> Dict[str, float]:
        """
        Extract sentiment metrics from play description.

        Args:
            play_text: Play description text

        Returns:
            Dictionary with sentiment scores
        """
        text_lower = play_text.lower()

        # Count sentiment indicators
        excitement_score = sum(1 for word in self.excitement_words if word in text_lower)
        negative_score = sum(1 for word in self.negative_words if word in text_lower)
        intensity_score = sum(1 for word in self.intensity_words if word in text_lower)

        # Calculate overall sentiment (-1 to 1)
        raw_sentiment = (excitement_score - negative_score) / max(1, excitement_score + negative_score)

        # Adjust for intensity
        intensity_multiplier = 1 + (intensity_score * 0.2)
        final_sentiment = raw_sentiment * intensity_multiplier

        return {
            'sentiment_score': max(-1.0, min(1.0, final_sentiment)),
            'excitement_level': excitement_score,
            'negative_level': negative_score,
            'intensity_level': intensity_score,
            'text_length': len(play_text.split())
        }

    async def collect(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Collect ESPN data with enhanced sentiment analysis."""
        async for data in super().collect():
            # Add sentiment analysis to new plays
            if 'new_plays' in data:
                for play_data in data['new_plays']:
                    sentiment_metrics = self._extract_play_sentiment(play_data['description'])
                    play_data['sentiment'] = sentiment_metrics

            # Add overall game sentiment
            if data.get('game_state') and data.get('new_plays'):
                recent_plays = data['new_plays']
                if recent_plays:
                    avg_sentiment = sum(
                        play.get('sentiment', {}).get('sentiment_score', 0)
                        for play in recent_plays
                    ) / len(recent_plays)

                    data['game_sentiment'] = {
                        'average_sentiment': avg_sentiment,
                        'sentiment_trend': 'positive' if avg_sentiment > 0.1 else 'negative' if avg_sentiment < -0.1 else 'neutral',
                        'play_count': len(recent_plays)
                    }

            yield data


# Factory functions for easy creation
def create_gamecast_source(
    game_id: str,
    sport: str = "football/nfl",
    poll_interval: float = 5.0,
    enhanced_sentiment: bool = True
) -> ESPNGameCastSource:
    """
    Create ESPN GameCast source with options.

    Args:
        game_id: ESPN game identifier
        sport: Sport type (e.g., "football/nfl", "basketball/nba")
        poll_interval: Polling interval in seconds
        enhanced_sentiment: Whether to use enhanced sentiment analysis

    Returns:
        Configured ESPN GameCast source
    """
    if enhanced_sentiment:
        return ESPNSentimentSource(game_id, sport, poll_interval)
    else:
        return ESPNGameCastSource(game_id, sport, poll_interval)


# Example usage
if __name__ == "__main__":
    async def example():
        # Example game ID (would be from actual ESPN)
        game_source = create_gamecast_source(
            game_id="401547439",  # Example NFL game ID
            sport="football/nfl",
            poll_interval=10.0
        )

        async with game_source:
            async for data in game_source.collect():
                print(f"Game momentum - Home: {data.get('momentum_home', 0):.2f}, Away: {data.get('momentum_away', 0):.2f}")
                if data.get('new_plays'):
                    for play in data['new_plays']:
                        print(f"  Play: {play['description'][:50]}... (momentum: {play['momentum_score']:.2f})")
                break

    asyncio.run(example())