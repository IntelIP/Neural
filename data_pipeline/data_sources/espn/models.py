"""
ESPN Data Models
Structured representations of play-by-play data
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class PlayType(Enum):
    """Types of plays in football"""
    RUSH = "rush"
    PASS = "pass"
    PUNT = "punt"
    KICKOFF = "kickoff"
    FIELD_GOAL = "field_goal"
    EXTRA_POINT = "extra_point"
    TWO_POINT = "two_point_conversion"
    PENALTY = "penalty"
    TIMEOUT = "timeout"
    SPIKE = "spike"
    KNEEL = "kneel"
    OTHER = "other"


class EventType(Enum):
    """Market-moving event types"""
    TOUCHDOWN = "touchdown"
    FIELD_GOAL_MADE = "field_goal_made"
    FIELD_GOAL_MISSED = "field_goal_missed"
    INTERCEPTION = "interception"
    FUMBLE = "fumble"
    FUMBLE_RECOVERY = "fumble_recovery"
    SAFETY = "safety"
    TURNOVER_ON_DOWNS = "turnover_on_downs"
    INJURY = "injury"
    PENALTY_MAJOR = "penalty_major"
    TWO_MINUTE_WARNING = "two_minute_warning"
    QUARTER_END = "quarter_end"
    HALF_END = "half_end"
    GAME_END = "game_end"


@dataclass
class Play:
    """Individual play data"""
    id: str
    text: str
    play_type: PlayType
    down: Optional[int] = None
    distance: Optional[int] = None
    yard_line: Optional[int] = None
    yards_gained: Optional[int] = None
    timestamp: Optional[str] = None
    clock: Optional[str] = None
    quarter: Optional[int] = None
    scoring_play: bool = False
    turnover: bool = False
    events: List[EventType] = field(default_factory=list)
    players_involved: List[str] = field(default_factory=list)
    impact_score: float = 0.0  # Market impact score (0-1)
    
    def is_high_impact(self) -> bool:
        """Check if play has high market impact"""
        high_impact_events = {
            EventType.TOUCHDOWN,
            EventType.INTERCEPTION,
            EventType.FUMBLE,
            EventType.SAFETY,
            EventType.INJURY
        }
        return any(event in high_impact_events for event in self.events)
    
    def is_scoring(self) -> bool:
        """Check if play resulted in score"""
        scoring_events = {
            EventType.TOUCHDOWN,
            EventType.FIELD_GOAL_MADE,
            EventType.SAFETY
        }
        return any(event in scoring_events for event in self.events)


@dataclass
class Drive:
    """Drive information"""
    id: str
    team: str
    start_yard_line: int
    end_yard_line: Optional[int] = None
    plays: List[Play] = field(default_factory=list)
    result: Optional[str] = None  # TD, FG, Punt, Turnover, etc.
    time_of_possession: Optional[str] = None
    yards: int = 0
    first_downs: int = 0
    
    def calculate_efficiency(self) -> float:
        """Calculate drive efficiency (yards per play)"""
        if not self.plays:
            return 0.0
        return self.yards / len(self.plays)
    
    def is_scoring_drive(self) -> bool:
        """Check if drive resulted in points"""
        return self.result in ["TOUCHDOWN", "FIELD_GOAL"]


@dataclass
class TeamStats:
    """Team statistics"""
    score: int = 0
    total_yards: int = 0
    passing_yards: int = 0
    rushing_yards: int = 0
    turnovers: int = 0
    time_of_possession: str = "0:00"
    third_down_efficiency: str = "0/0"
    penalties: int = 0
    penalty_yards: int = 0


@dataclass
class GameState:
    """Current game state"""
    game_id: str
    home_team: str
    away_team: str
    home_score: int = 0
    away_score: int = 0
    quarter: int = 1
    clock: str = "15:00"
    down: Optional[int] = None
    distance: Optional[int] = None
    yard_line: Optional[int] = None
    possession: Optional[str] = None
    home_timeouts: int = 3
    away_timeouts: int = 3
    home_win_probability: float = 50.0
    away_win_probability: float = 50.0
    is_redzone: bool = False
    is_two_minute_warning: bool = False
    home_stats: TeamStats = field(default_factory=TeamStats)
    away_stats: TeamStats = field(default_factory=TeamStats)
    last_update: datetime = field(default_factory=datetime.now)
    
    def is_close_game(self, threshold: int = 7) -> bool:
        """Check if game is close (within threshold points)"""
        return abs(self.home_score - self.away_score) <= threshold
    
    def is_late_game(self) -> bool:
        """Check if in late game situation"""
        return self.quarter >= 4 or (self.quarter == 3 and self._parse_clock_minutes() < 5)
    
    def is_critical_situation(self) -> bool:
        """Check if in critical game situation"""
        return (
            (self.is_late_game() and self.is_close_game()) or
            self.is_two_minute_warning or
            (self.is_redzone and self.is_close_game()) or
            (self.down == 4 and self.distance <= 3)
        )
    
    def _parse_clock_minutes(self) -> int:
        """Parse minutes from clock string"""
        try:
            minutes = int(self.clock.split(':')[0])
            return minutes
        except:
            return 15


@dataclass
class GameEvent:
    """Game event for streaming"""
    type: EventType
    game_id: str
    description: str
    impact_score: float
    game_state: GameState
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'type': self.type.value,
            'game_id': self.game_id,
            'description': self.description,
            'impact_score': self.impact_score,
            'timestamp': self.timestamp.isoformat(),
            'game_state': {
                'home_score': self.game_state.home_score,
                'away_score': self.game_state.away_score,
                'quarter': self.game_state.quarter,
                'clock': self.game_state.clock,
                'home_win_prob': self.game_state.home_win_probability,
                'away_win_prob': self.game_state.away_win_probability
            },
            'metadata': self.metadata
        }