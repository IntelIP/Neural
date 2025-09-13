"""
ESPN Play-by-Play Processor
Analyzes plays and extracts market-relevant information
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple

from .models import (
    Play, Drive, GameState, PlayType, EventType, 
    TeamStats, GameEvent
)

logger = logging.getLogger(__name__)


class PlayByPlayProcessor:
    """Process and analyze play-by-play data"""
    
    def __init__(self):
        """Initialize the processor"""
        self.play_patterns = self._compile_patterns()
        self.event_impact_scores = self._define_impact_scores()
    
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for play parsing"""
        return {
            'touchdown': re.compile(r'touchdown|TD', re.IGNORECASE),
            'field_goal': re.compile(r'field goal|FG', re.IGNORECASE),
            'interception': re.compile(r'intercepted|interception|INT', re.IGNORECASE),
            'fumble': re.compile(r'fumble|fumbled', re.IGNORECASE),
            'safety': re.compile(r'safety', re.IGNORECASE),
            'punt': re.compile(r'punt|punts', re.IGNORECASE),
            'kickoff': re.compile(r'kickoff|kicks off', re.IGNORECASE),
            'pass': re.compile(r'pass|passes|thrown|complete|incomplete', re.IGNORECASE),
            'rush': re.compile(r'rush|run|carry|carries', re.IGNORECASE),
            'penalty': re.compile(r'penalty|flag', re.IGNORECASE),
            'timeout': re.compile(r'timeout', re.IGNORECASE),
            'spike': re.compile(r'spike', re.IGNORECASE),
            'kneel': re.compile(r'kneel|kneels', re.IGNORECASE),
            'yards': re.compile(r'(\d+)\s*yard'),
            'down_distance': re.compile(r'(\d+)(?:st|nd|rd|th)\s*(?:and|&)\s*(\d+)'),
            'yard_line': re.compile(r'at\s+(?:the\s+)?(\w+)\s+(\d+)'),
            'clock': re.compile(r'(\d{1,2}):(\d{2})')
        }
    
    def _define_impact_scores(self) -> Dict[EventType, float]:
        """Define market impact scores for events"""
        return {
            EventType.TOUCHDOWN: 0.9,
            EventType.INTERCEPTION: 0.85,
            EventType.FUMBLE: 0.8,
            EventType.SAFETY: 0.95,
            EventType.FIELD_GOAL_MADE: 0.6,
            EventType.FIELD_GOAL_MISSED: 0.5,
            EventType.INJURY: 0.7,
            EventType.TURNOVER_ON_DOWNS: 0.7,
            EventType.TWO_MINUTE_WARNING: 0.4,
            EventType.QUARTER_END: 0.3,
            EventType.HALF_END: 0.5,
            EventType.GAME_END: 1.0,
            EventType.PENALTY_MAJOR: 0.4,
            EventType.FUMBLE_RECOVERY: 0.75
        }
    
    def process_play(self, play_data: Dict[str, Any], game_state: GameState) -> Play:
        """
        Process raw play data into structured Play object
        
        Args:
            play_data: Raw play data from ESPN API
            game_state: Current game state
            
        Returns:
            Processed Play object
        """
        text = play_data.get('text', '')
        play_id = play_data.get('id', '')
        
        # Determine play type
        play_type = self._determine_play_type(text)
        
        # Extract play details
        down, distance = self._extract_down_distance(text)
        yard_line = self._extract_yard_line(text)
        yards_gained = self._extract_yards_gained(text)
        clock = self._extract_clock(text)
        
        # Detect events
        events = self._detect_events(text)
        
        # Extract players
        players = self._extract_players(text)
        
        # Calculate impact score
        impact_score = self._calculate_impact_score(events, game_state)
        
        # Check if scoring play
        scoring_play = any(e in [EventType.TOUCHDOWN, EventType.FIELD_GOAL_MADE, EventType.SAFETY] 
                          for e in events)
        
        # Check if turnover
        turnover = any(e in [EventType.INTERCEPTION, EventType.FUMBLE, EventType.TURNOVER_ON_DOWNS] 
                      for e in events)
        
        return Play(
            id=play_id,
            text=text,
            play_type=play_type,
            down=down,
            distance=distance,
            yard_line=yard_line,
            yards_gained=yards_gained,
            timestamp=play_data.get('wallclock'),
            clock=clock,
            quarter=game_state.quarter,
            scoring_play=scoring_play,
            turnover=turnover,
            events=events,
            players_involved=players,
            impact_score=impact_score
        )
    
    def _determine_play_type(self, text: str) -> PlayType:
        """Determine the type of play from text"""
        text_lower = text.lower()
        
        if self.play_patterns['touchdown'].search(text):
            if self.play_patterns['pass'].search(text):
                return PlayType.PASS
            return PlayType.RUSH
        elif self.play_patterns['field_goal'].search(text):
            return PlayType.FIELD_GOAL
        elif self.play_patterns['punt'].search(text):
            return PlayType.PUNT
        elif self.play_patterns['kickoff'].search(text):
            return PlayType.KICKOFF
        elif 'extra point' in text_lower:
            return PlayType.EXTRA_POINT
        elif 'two point' in text_lower or '2 point' in text_lower:
            return PlayType.TWO_POINT
        elif self.play_patterns['penalty'].search(text):
            return PlayType.PENALTY
        elif self.play_patterns['timeout'].search(text):
            return PlayType.TIMEOUT
        elif self.play_patterns['spike'].search(text):
            return PlayType.SPIKE
        elif self.play_patterns['kneel'].search(text):
            return PlayType.KNEEL
        elif self.play_patterns['pass'].search(text):
            return PlayType.PASS
        elif self.play_patterns['rush'].search(text):
            return PlayType.RUSH
        else:
            return PlayType.OTHER
    
    def _extract_down_distance(self, text: str) -> Tuple[Optional[int], Optional[int]]:
        """Extract down and distance from play text"""
        match = self.play_patterns['down_distance'].search(text)
        if match:
            return int(match.group(1)), int(match.group(2))
        return None, None
    
    def _extract_yard_line(self, text: str) -> Optional[int]:
        """Extract yard line from play text"""
        match = self.play_patterns['yard_line'].search(text)
        if match:
            try:
                return int(match.group(2))
            except:
                pass
        return None
    
    def _extract_yards_gained(self, text: str) -> Optional[int]:
        """Extract yards gained from play text"""
        match = self.play_patterns['yards'].search(text)
        if match:
            try:
                yards = int(match.group(1))
                # Check for loss
                if 'loss' in text.lower():
                    return -yards
                return yards
            except:
                pass
        return None
    
    def _extract_clock(self, text: str) -> Optional[str]:
        """Extract game clock from play text"""
        match = self.play_patterns['clock'].search(text)
        if match:
            return f"{match.group(1)}:{match.group(2)}"
        return None
    
    def _detect_events(self, text: str) -> List[EventType]:
        """Detect market-moving events in play text"""
        events = []
        text_lower = text.lower()
        
        if self.play_patterns['touchdown'].search(text):
            events.append(EventType.TOUCHDOWN)
        if self.play_patterns['interception'].search(text):
            events.append(EventType.INTERCEPTION)
        if self.play_patterns['fumble'].search(text):
            events.append(EventType.FUMBLE)
            if 'recovered' in text_lower:
                events.append(EventType.FUMBLE_RECOVERY)
        if self.play_patterns['safety'].search(text):
            events.append(EventType.SAFETY)
        if 'field goal' in text_lower:
            if 'good' in text_lower or 'made' in text_lower:
                events.append(EventType.FIELD_GOAL_MADE)
            elif 'no good' in text_lower or 'missed' in text_lower:
                events.append(EventType.FIELD_GOAL_MISSED)
        if 'injury' in text_lower or 'injured' in text_lower:
            events.append(EventType.INJURY)
        if 'turnover on downs' in text_lower:
            events.append(EventType.TURNOVER_ON_DOWNS)
        
        return events
    
    def _extract_players(self, text: str) -> List[str]:
        """Extract player names from play text"""
        # Simple pattern for common name formats
        # This could be enhanced with a player database
        players = []
        
        # Pattern for "FirstName LastName" or "F.LastName"
        player_pattern = re.compile(r'\b([A-Z][a-z]*\.?\s+[A-Z][a-z]+)\b')
        matches = player_pattern.findall(text)
        
        for match in matches:
            # Filter out common non-player phrases
            if not any(word in match.lower() for word in ['yard', 'down', 'quarter', 'timeout']):
                players.append(match)
        
        return players
    
    def _calculate_impact_score(self, events: List[EventType], game_state: GameState) -> float:
        """
        Calculate market impact score for a play
        
        Args:
            events: Events detected in the play
            game_state: Current game state
            
        Returns:
            Impact score between 0 and 1
        """
        if not events:
            return 0.1  # Base score for regular play
        
        # Get max event score
        max_score = max(self.event_impact_scores.get(event, 0.1) for event in events)
        
        # Apply situational modifiers
        if game_state.is_critical_situation():
            max_score *= 1.5
        
        if game_state.is_late_game():
            max_score *= 1.3
        
        if game_state.is_close_game():
            max_score *= 1.2
        
        if game_state.is_redzone:
            max_score *= 1.1
        
        # Cap at 1.0
        return min(max_score, 1.0)
    
    def process_drive(self, drive_data: Dict[str, Any], game_state: GameState) -> Drive:
        """
        Process raw drive data into structured Drive object
        
        Args:
            drive_data: Raw drive data from ESPN API
            game_state: Current game state
            
        Returns:
            Processed Drive object
        """
        drive = Drive(
            id=drive_data.get('id', ''),
            team=drive_data.get('team', {}).get('displayName', ''),
            start_yard_line=drive_data.get('start', {}).get('yardLine', 0),
            end_yard_line=drive_data.get('end', {}).get('yardLine'),
            result=drive_data.get('result'),
            time_of_possession=drive_data.get('timeOfPossession'),
            yards=drive_data.get('yards', 0),
            first_downs=drive_data.get('firstDowns', 0)
        )
        
        # Process plays in drive
        for play_data in drive_data.get('plays', []):
            play = self.process_play(play_data, game_state)
            drive.plays.append(play)
        
        return drive
    
    def extract_game_state(self, summary_data: Dict[str, Any]) -> GameState:
        """
        Extract current game state from ESPN summary
        
        Args:
            summary_data: Full game summary from ESPN API
            
        Returns:
            Current GameState object
        """
        header = summary_data.get('header', {})
        competitions = header.get('competitions', [{}])[0]
        competitors = competitions.get('competitors', [{}, {}])
        
        # Determine home/away teams
        home_team = None
        away_team = None
        home_score = 0
        away_score = 0
        
        for competitor in competitors:
            if competitor.get('homeAway') == 'home':
                home_team = competitor.get('team', {}).get('displayName', '')
                home_score = int(competitor.get('score', 0))
            else:
                away_team = competitor.get('team', {}).get('displayName', '')
                away_score = int(competitor.get('score', 0))
        
        # Get game status
        status = competitions.get('status', {})
        quarter = status.get('period', 1)
        clock = status.get('displayClock', '15:00')
        
        # Get possession and situation
        situation = summary_data.get('situation', {})
        possession = situation.get('possession')
        down = situation.get('down')
        distance = situation.get('distance')
        yard_line = situation.get('yardLine')
        is_redzone = situation.get('isRedZone', False)
        
        # Get win probability
        win_prob_data = summary_data.get('winprobability', [])
        home_win_prob = 50.0
        away_win_prob = 50.0
        
        if win_prob_data:
            latest_prob = win_prob_data[-1]
            home_win_prob = latest_prob.get('homeWinPercentage', 50.0)
            away_win_prob = 100 - home_win_prob
        
        # Get timeouts
        home_timeouts = situation.get('homeTimeouts', 3)
        away_timeouts = situation.get('awayTimeouts', 3)
        
        # Create game state
        game_state = GameState(
            game_id=summary_data.get('header', {}).get('id', ''),
            home_team=home_team,
            away_team=away_team,
            home_score=home_score,
            away_score=away_score,
            quarter=quarter,
            clock=clock,
            down=down,
            distance=distance,
            yard_line=yard_line,
            possession=possession,
            home_timeouts=home_timeouts,
            away_timeouts=away_timeouts,
            home_win_probability=home_win_prob,
            away_win_probability=away_win_prob,
            is_redzone=is_redzone
        )
        
        # Extract team stats if available
        game_state.home_stats = self._extract_team_stats(summary_data, 'home')
        game_state.away_stats = self._extract_team_stats(summary_data, 'away')
        
        return game_state
    
    def _extract_team_stats(self, summary_data: Dict[str, Any], team_type: str) -> TeamStats:
        """Extract team statistics from summary"""
        stats = TeamStats()
        
        # This would need to be enhanced based on actual ESPN API structure
        # Placeholder implementation
        team_stats_data = summary_data.get('boxscore', {}).get(team_type, {})
        
        if team_stats_data:
            stats.total_yards = team_stats_data.get('totalYards', 0)
            stats.passing_yards = team_stats_data.get('passingYards', 0)
            stats.rushing_yards = team_stats_data.get('rushingYards', 0)
            stats.turnovers = team_stats_data.get('turnovers', 0)
            stats.penalties = team_stats_data.get('penalties', 0)
            stats.penalty_yards = team_stats_data.get('penaltyYards', 0)
        
        return stats
    
    def create_game_event(
        self,
        event_type: EventType,
        game_state: GameState,
        description: str,
        impact_score: float = None,
        metadata: Dict[str, Any] = None
    ) -> GameEvent:
        """
        Create a game event for streaming
        
        Args:
            event_type: Type of event
            game_state: Current game state
            description: Event description
            impact_score: Optional impact score override
            metadata: Additional event metadata
            
        Returns:
            GameEvent object
        """
        if impact_score is None:
            impact_score = self.event_impact_scores.get(event_type, 0.5)
        
        return GameEvent(
            type=event_type,
            game_id=game_state.game_id,
            description=description,
            impact_score=impact_score,
            game_state=game_state,
            metadata=metadata or {}
        )