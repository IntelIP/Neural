"""
Synthetic Game Engine

Uses fine-tuned LFM2 models to generate realistic NFL game scenarios
for agent training with unlimited synthetic data.
"""

import logging
import random
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json

from ..preprocessing.nfl_dataset_processor import ProcessedNFLPlay
from ..storage.chromadb_manager import ChromaDBManager
from src.sdk.core.base_adapter import StandardizedEvent, EventType

logger = logging.getLogger(__name__)


@dataclass
class GameContext:
    """Current game state context"""
    game_id: str
    home_team: str
    away_team: str
    quarter: int = 1
    time_remaining: str = "15:00"
    time_seconds: int = 900
    
    # Situation
    down: int = 1
    distance: int = 10
    yard_line: int = 25
    yards_to_goal: int = 75
    possession_team: str = ""
    
    # Score
    score_home: int = 0
    score_away: int = 0
    score_differential: int = 0
    
    # Game factors
    weather_impact: float = 0.0  # -1 to 1
    crowd_noise: float = 0.0     # 0 to 1
    momentum: float = 0.0        # -1 to 1 (negative = away, positive = home)
    
    def __post_init__(self):
        if not self.possession_team:
            self.possession_team = self.home_team
        self.score_differential = self.score_home - self.score_away


@dataclass 
class SyntheticPlay:
    """Generated synthetic play"""
    play_id: str
    context: GameContext
    play_type: str
    play_description: str
    yards_gained: int
    time_elapsed: int = 30
    
    # Outcomes
    touchdown: bool = False
    field_goal: bool = False
    turnover: bool = False
    safety: bool = False
    penalty: bool = False
    
    # Advanced metrics
    epa: float = 0.0
    wpa: float = 0.0
    excitement_factor: float = 0.5  # 0 to 1


@dataclass
class SyntheticGame:
    """Complete synthetic game"""
    game_id: str
    home_team: str
    away_team: str
    season: int
    week: int
    plays: List[SyntheticPlay] = field(default_factory=list)
    final_score: Tuple[int, int] = (0, 0)
    total_plays: int = 0
    game_duration_minutes: int = 180
    created_at: datetime = field(default_factory=datetime.now)
    
    # Game characteristics
    game_type: str = "regular"  # regular, close, blowout, overtime, weather
    excitement_score: float = 0.5
    market_impact: float = 0.5
    

class SyntheticGameEngine:
    """
    Generates realistic NFL game scenarios using fine-tuned LFM2 models
    and historical pattern analysis from ChromaDB
    """
    
    def __init__(self, 
                 chromadb_manager: ChromaDBManager = None,
                 model_name: str = "nfl_playbypay_lfm2"):
        """
        Initialize synthetic game engine
        
        Args:
            chromadb_manager: ChromaDB manager for historical patterns
            model_name: Fine-tuned LFM2 model name
        """
        self.chromadb = chromadb_manager or ChromaDBManager()
        self.model_name = model_name
        
        # Team data
        self.nfl_teams = [
            "KC", "BUF", "CIN", "BAL", "MIA", "NYJ", "NE", "CLE",
            "DAL", "PHI", "WAS", "NYG", "GB", "MIN", "CHI", "DET", 
            "LAR", "SF", "SEA", "ARI", "NO", "TB", "ATL", "CAR",
            "LV", "LAC", "DEN", "PIT", "TEN", "IND", "HOU", "JAX"
        ]
        
        # Game templates
        self.game_templates = self._load_game_templates()
        
        logger.info(f"Initialized SyntheticGameEngine with model: {model_name}")
    
    def _load_game_templates(self) -> Dict[str, Dict]:
        """Load game scenario templates"""
        return {
            "regular": {
                "avg_plays": 140,
                "avg_points": 45,
                "variance": 0.2,
                "overtime_probability": 0.05
            },
            "high_scoring": {
                "avg_plays": 160, 
                "avg_points": 65,
                "variance": 0.3,
                "overtime_probability": 0.03
            },
            "defensive": {
                "avg_plays": 120,
                "avg_points": 28,
                "variance": 0.15,
                "overtime_probability": 0.08
            },
            "weather": {
                "avg_plays": 110,
                "avg_points": 35,
                "variance": 0.25,
                "weather_impact": 0.7
            },
            "blowout": {
                "avg_plays": 135,
                "avg_points": 52,
                "variance": 0.4,
                "comeback_probability": 0.15
            }
        }
    
    async def generate_single_game(self, 
                                 home_team: str = None, 
                                 away_team: str = None,
                                 game_type: str = "regular",
                                 season: int = 2024,
                                 week: int = 1) -> SyntheticGame:
        """
        Generate a complete synthetic game
        
        Args:
            home_team: Home team code (random if None)
            away_team: Away team code (random if None) 
            game_type: Type of game scenario
            season: Season year
            week: Week number
            
        Returns:
            Complete synthetic game
        """
        # Select random teams if not specified
        if not home_team or not away_team:
            teams = random.sample(self.nfl_teams, 2)
            home_team = home_team or teams[0]
            away_team = away_team or teams[1]
        
        game_id = f"{season}{week:02d}{home_team}{away_team}"
        
        logger.info(f"Generating synthetic game: {away_team} @ {home_team} ({game_type})")
        
        # Initialize game context
        context = GameContext(
            game_id=game_id,
            home_team=home_team,
            away_team=away_team,
            possession_team=away_team if random.random() > 0.5 else home_team
        )
        
        # Get game template
        template = self.game_templates.get(game_type, self.game_templates["regular"])
        
        # Generate game
        game = SyntheticGame(
            game_id=game_id,
            home_team=home_team,
            away_team=away_team,
            season=season,
            week=week,
            game_type=game_type
        )
        
        # Generate plays using LFM2 and historical patterns
        await self._generate_game_plays(game, context, template)
        
        # Post-process game statistics
        self._calculate_game_metrics(game)
        
        logger.info(f"Generated game complete: {game.away_team} {game.final_score[1]} - {game.final_score[0]} {game.home_team}")
        return game
    
    async def _generate_game_plays(self, 
                                 game: SyntheticGame,
                                 context: GameContext, 
                                 template: Dict) -> None:
        """Generate all plays for a game"""
        
        # Estimate total plays
        target_plays = int(template["avg_plays"] * (1 + random.gauss(0, template["variance"])))
        target_plays = max(100, min(200, target_plays))
        
        play_count = 0
        drive_number = 1
        
        while context.quarter <= 4 and play_count < target_plays:
            # Generate drive
            drive_plays = await self._generate_drive(context, drive_number, template)
            game.plays.extend(drive_plays)
            play_count += len(drive_plays)
            drive_number += 1
            
            # Update quarter/time
            self._advance_game_time(context, len(drive_plays))
            
            # Switch possession
            self._switch_possession(context)
            
            # Check for quarter end
            if context.time_seconds <= 0:
                context.quarter += 1
                context.time_seconds = 900  # 15 minutes
                context.time_remaining = "15:00"
        
        # Handle overtime if needed
        if context.score_home == context.score_away and random.random() < template.get("overtime_probability", 0.05):
            await self._generate_overtime(game, context)
        
        game.total_plays = len(game.plays)
        game.final_score = (context.score_home, context.score_away)
    
    async def _generate_drive(self, 
                            context: GameContext,
                            drive_number: int,
                            template: Dict) -> List[SyntheticPlay]:
        """Generate plays for a single drive"""
        
        drive_plays = []
        starting_field_pos = context.yards_to_goal
        
        # Reset downs
        context.down = 1
        context.distance = 10
        
        while True:
            # Generate single play
            play = await self._generate_single_play(context, drive_number, len(drive_plays) + 1)
            drive_plays.append(play)
            
            # Apply play results
            self._apply_play_results(context, play)
            
            # Check drive ending conditions
            if self._is_drive_over(context, play):
                break
                
            # Prevent infinite drives
            if len(drive_plays) >= 20:
                break
        
        return drive_plays
    
    async def _generate_single_play(self, 
                                  context: GameContext,
                                  drive_number: int,
                                  play_number: int) -> SyntheticPlay:
        """Generate a single play using LFM2 and pattern matching"""
        
        # Search for similar situations in ChromaDB
        situation_query = self._build_situation_query(context)
        similar_plays = self.chromadb.search_similar_plays(
            query=situation_query,
            n_results=5
        )
        
        # Generate play using LFM2 (simulated for now)
        play_type, yards_gained, outcomes = await self._llm_generate_play(context, similar_plays)
        
        # Create play description
        description = self._create_play_description(context, play_type, yards_gained, outcomes)
        
        play_id = f"{context.game_id}_{drive_number}_{play_number}"
        
        return SyntheticPlay(
            play_id=play_id,
            context=GameContext(**context.__dict__),  # Copy context
            play_type=play_type,
            play_description=description,
            yards_gained=yards_gained,
            touchdown=outcomes.get("touchdown", False),
            field_goal=outcomes.get("field_goal", False),
            turnover=outcomes.get("turnover", False),
            safety=outcomes.get("safety", False),
            penalty=outcomes.get("penalty", False),
            epa=self._calculate_epa(context, yards_gained, outcomes),
            wpa=self._calculate_wpa(context, yards_gained, outcomes)
        )
    
    def _build_situation_query(self, context: GameContext) -> str:
        """Build ChromaDB query for similar situations"""
        situation_parts = []
        
        # Down and distance
        situation_parts.append(f"{context.down} and {context.distance}")
        
        # Field position
        if context.yards_to_goal <= 20:
            situation_parts.append("red zone")
        elif context.yards_to_goal >= 80:
            situation_parts.append("deep in own territory")
        else:
            situation_parts.append(f"{context.yards_to_goal} yards to goal")
        
        # Time situation
        if context.quarter >= 4 and context.time_seconds < 120:
            situation_parts.append("two minute warning")
        elif context.quarter >= 4 and context.time_seconds < 300:
            situation_parts.append("fourth quarter")
        
        # Score situation
        if abs(context.score_differential) <= 3:
            situation_parts.append("close game")
        elif context.score_differential > 14:
            situation_parts.append("large lead")
        elif context.score_differential < -14:
            situation_parts.append("large deficit")
        
        return ". ".join(situation_parts)
    
    async def _llm_generate_play(self, 
                               context: GameContext, 
                               similar_plays: List[Dict]) -> Tuple[str, int, Dict]:
        """Use LFM2 to generate realistic play outcome"""
        
        # For now, simulate LFM2 with pattern-based generation
        # In production, this would call the fine-tuned Ollama model
        
        # Analyze similar plays for patterns
        play_types = []
        yard_totals = []
        
        for play in similar_plays:
            metadata = play.get('metadata', {})
            play_types.append(metadata.get('play_type', 'RUN'))
            yard_totals.append(metadata.get('yards_gained', 0))
        
        # Generate play type based on situation
        play_type = self._select_play_type(context, play_types)
        
        # Generate yards based on play type and situation
        yards_gained = self._generate_yards(context, play_type, yard_totals)
        
        # Generate special outcomes
        outcomes = self._generate_outcomes(context, play_type, yards_gained)
        
        return play_type, yards_gained, outcomes
    
    def _select_play_type(self, context: GameContext, historical_types: List[str]) -> str:
        """Select play type based on situation and patterns"""
        
        # Use historical patterns if available
        if historical_types:
            type_weights = {}
            for ptype in historical_types:
                type_weights[ptype] = type_weights.get(ptype, 0) + 1
            
            # Add situational bias
            if context.distance <= 2:
                type_weights["RUN"] = type_weights.get("RUN", 0) + 2
            elif context.distance >= 10:
                type_weights["PASS"] = type_weights.get("PASS", 0) + 2
            elif context.yards_to_goal <= 5 and context.down <= 2:
                type_weights["RUN"] = type_weights.get("RUN", 0) + 1
            
            # Weighted random selection
            total_weight = sum(type_weights.values())
            if total_weight > 0:
                rand = random.random() * total_weight
                cumsum = 0
                for ptype, weight in type_weights.items():
                    cumsum += weight
                    if rand <= cumsum:
                        return ptype
        
        # Fallback to situational logic
        if context.down >= 3 and context.distance >= 7:
            return "PASS"
        elif context.distance <= 2:
            return "RUN"  
        elif context.yards_to_goal <= 35 and context.down == 4:
            return "FIELD_GOAL"
        elif context.down == 4:
            return "PUNT"
        else:
            return random.choice(["RUN", "PASS"])
    
    def _generate_yards(self, context: GameContext, play_type: str, historical_yards: List[int]) -> int:
        """Generate realistic yards gained"""
        
        # Base yards by play type
        if play_type == "PASS":
            base_yards = random.choice([0, 3, 5, 7, 12, 15, 18, 22, 35])
            variance = 5
        elif play_type == "RUN":
            base_yards = random.choice([-1, 0, 1, 2, 3, 4, 5, 6, 8, 12])
            variance = 3
        elif play_type == "FIELD_GOAL":
            return 0  # Handled separately
        elif play_type == "PUNT":
            return -(40 + random.randint(-10, 10))  # Field position change
        else:
            base_yards = 2
            variance = 2
        
        # Adjust for historical patterns
        if historical_yards:
            avg_historical = sum(historical_yards) / len(historical_yards)
            base_yards = int(0.7 * base_yards + 0.3 * avg_historical)
        
        # Add random variance
        yards = base_yards + random.randint(-variance, variance)
        
        # Constrain to field boundaries
        max_yards = min(99, context.yards_to_goal)
        min_yards = max(-99, -(100 - context.yards_to_goal))
        
        return max(min_yards, min(max_yards, yards))
    
    def _generate_outcomes(self, context: GameContext, play_type: str, yards_gained: int) -> Dict:
        """Generate special play outcomes"""
        outcomes = {}
        
        # Touchdown
        if yards_gained >= context.yards_to_goal:
            outcomes["touchdown"] = True
            return outcomes
        
        # Field goal
        if play_type == "FIELD_GOAL":
            # Success probability based on distance
            distance = context.yards_to_goal + 17  # Add endzone + snap distance
            success_rate = max(0.5, 1.0 - (distance - 20) / 100)
            outcomes["field_goal"] = random.random() < success_rate
            return outcomes
        
        # Turnover 
        turnover_rate = 0.02
        if play_type == "PASS":
            turnover_rate = 0.025
        elif play_type == "RUN":
            turnover_rate = 0.015
        
        if random.random() < turnover_rate:
            outcomes["turnover"] = True
        
        # Penalty (rare)
        if random.random() < 0.08:
            outcomes["penalty"] = True
            # Penalties don't count as yards gained typically
            yards_gained = 0
        
        # Safety (very rare)
        if context.yards_to_goal >= 98 and yards_gained <= -2:
            outcomes["safety"] = True
        
        return outcomes
    
    def _create_play_description(self, context: GameContext, play_type: str, yards_gained: int, outcomes: Dict) -> str:
        """Create realistic play description"""
        
        # Player names (simplified)
        qb_name = f"{context.possession_team}.QB"
        rb_name = f"{context.possession_team}.RB" 
        wr_name = f"{context.possession_team}.WR"
        
        if outcomes.get("touchdown"):
            if play_type == "PASS":
                return f"{qb_name} pass complete to {wr_name} for {yards_gained} yards, TOUCHDOWN"
            elif play_type == "RUN":
                return f"{rb_name} rush for {yards_gained} yards, TOUCHDOWN"
        elif outcomes.get("field_goal"):
            return f"Field goal attempt is GOOD"
        elif outcomes.get("turnover"):
            if play_type == "PASS":
                return f"{qb_name} pass INTERCEPTED"
            else:
                return f"{rb_name} FUMBLES, recovered by defense"
        elif play_type == "PASS":
            if yards_gained <= 0:
                return f"{qb_name} pass incomplete"
            else:
                return f"{qb_name} pass complete to {wr_name} for {yards_gained} yards"
        elif play_type == "RUN":
            return f"{rb_name} rush for {yards_gained} yards"
        elif play_type == "PUNT":
            return f"Punt for {abs(yards_gained)} yards"
        else:
            return f"{play_type} for {yards_gained} yards"
    
    def _calculate_epa(self, context: GameContext, yards_gained: int, outcomes: Dict) -> float:
        """Calculate Expected Points Added (simplified)"""
        
        # Simplified EPA calculation
        base_ep = 0.0
        
        # Base expected points by field position
        if context.yards_to_goal <= 5:
            base_ep = 6.0
        elif context.yards_to_goal <= 15:
            base_ep = 4.5
        elif context.yards_to_goal <= 35:
            base_ep = 3.0
        elif context.yards_to_goal <= 65:
            base_ep = 1.0
        else:
            base_ep = 0.2
        
        # Adjust for outcomes
        if outcomes.get("touchdown"):
            return 7.0 - base_ep
        elif outcomes.get("field_goal"):
            return 3.0 - base_ep
        elif outcomes.get("turnover"):
            return -base_ep - 2.0
        else:
            # Rough EPA for yard gain
            yard_value = yards_gained * 0.1
            return yard_value - 0.5  # Small negative for not scoring
    
    def _calculate_wpa(self, context: GameContext, yards_gained: int, outcomes: Dict) -> float:
        """Calculate Win Probability Added (simplified)"""
        
        # Simplified WPA calculation
        time_factor = context.time_seconds / 3600  # Normalize to game time
        score_factor = 1.0 / (1.0 + abs(context.score_differential) / 7.0)
        
        base_wpa = 0.0
        
        if outcomes.get("touchdown"):
            base_wpa = 0.15 * score_factor
        elif outcomes.get("field_goal"):
            base_wpa = 0.08 * score_factor
        elif outcomes.get("turnover"):
            base_wpa = -0.12 * score_factor
        else:
            # First down conversion impact
            if yards_gained >= context.distance:
                base_wpa = 0.03 * score_factor
            else:
                base_wpa = -0.01 * score_factor
        
        # Time pressure multiplier
        if context.quarter >= 4:
            base_wpa *= (2.0 - time_factor)
        
        return base_wpa
    
    def _apply_play_results(self, context: GameContext, play: SyntheticPlay) -> None:
        """Apply play results to game context"""
        
        # Update score
        if play.touchdown:
            if context.possession_team == context.home_team:
                context.score_home += 7  # TD + XP
            else:
                context.score_away += 7
            context.score_differential = context.score_home - context.score_away
            
        elif play.field_goal:
            if context.possession_team == context.home_team:
                context.score_home += 3
            else:
                context.score_away += 3
            context.score_differential = context.score_home - context.score_away
            
        elif play.safety:
            if context.possession_team == context.home_team:
                context.score_away += 2  # Defense gets points
            else:
                context.score_home += 2
            context.score_differential = context.score_home - context.score_away
        
        # Update field position
        if not (play.touchdown or play.field_goal or play.turnover):
            context.yards_to_goal -= play.yards_gained
            context.yards_to_goal = max(1, min(99, context.yards_to_goal))
            context.yard_line = 100 - context.yards_to_goal
        
        # Update down and distance
        if play.turnover or play.touchdown or play.field_goal or play.safety:
            # Possession will change
            pass
        elif play.yards_gained >= context.distance:
            # First down
            context.down = 1
            context.distance = 10
        else:
            # Next down
            context.down += 1
            context.distance -= play.yards_gained
            context.distance = max(1, context.distance)
    
    def _is_drive_over(self, context: GameContext, play: SyntheticPlay) -> bool:
        """Check if drive should end"""
        
        # Scoring plays end drives
        if play.touchdown or play.field_goal or play.safety:
            return True
        
        # Turnovers end drives
        if play.turnover:
            return True
        
        # Fourth down stops (punt/failed conversion)
        if context.down >= 4 and play.yards_gained < context.distance:
            return True
        
        return False
    
    def _advance_game_time(self, context: GameContext, num_plays: int) -> None:
        """Advance game time based on plays"""
        
        # Average 30 seconds per play
        time_elapsed = num_plays * 30
        context.time_seconds -= time_elapsed
        
        if context.time_seconds <= 0:
            context.time_seconds = 0
            context.time_remaining = "0:00"
        else:
            minutes = context.time_seconds // 60
            seconds = context.time_seconds % 60
            context.time_remaining = f"{minutes}:{seconds:02d}"
    
    def _switch_possession(self, context: GameContext) -> None:
        """Switch possession between teams"""
        
        if context.possession_team == context.home_team:
            context.possession_team = context.away_team
        else:
            context.possession_team = context.home_team
        
        # Reset field position for new drive
        context.yards_to_goal = random.randint(70, 85)  # Typical starting position
        context.yard_line = 100 - context.yards_to_goal
    
    async def _generate_overtime(self, game: SyntheticGame, context: GameContext) -> None:
        """Generate overtime period"""
        logger.info(f"Generating overtime for {game.game_id}")
        
        context.quarter = 5
        context.time_seconds = 900  # 15 minutes
        context.time_remaining = "15:00"
        
        # Simple overtime - first score wins
        drive_number = 100  # High number to distinguish OT
        
        while context.score_home == context.score_away:
            drive_plays = await self._generate_drive(context, drive_number, self.game_templates["regular"])
            game.plays.extend(drive_plays)
            
            # Check for scoring
            if any(play.touchdown or play.field_goal or play.safety for play in drive_plays):
                break
            
            self._switch_possession(context)
            drive_number += 1
            
            # Prevent infinite overtime
            if drive_number > 110:
                # Simulate coin flip winner
                if random.random() > 0.5:
                    context.score_home += 3
                else:
                    context.score_away += 3
                break
    
    def _calculate_game_metrics(self, game: SyntheticGame) -> None:
        """Calculate final game metrics"""
        
        if game.plays:
            # Calculate excitement score
            big_plays = sum(1 for play in game.plays if abs(play.yards_gained) >= 20)
            scoring_plays = sum(1 for play in game.plays if play.touchdown or play.field_goal)
            turnovers = sum(1 for play in game.plays if play.turnover)
            
            excitement = (big_plays * 0.1 + scoring_plays * 0.2 + turnovers * 0.15)
            game.excitement_score = min(1.0, excitement / 10.0)
            
            # Calculate market impact (based on score differential and lead changes)
            score_diff = abs(game.final_score[0] - game.final_score[1])
            if score_diff <= 3:
                game.market_impact = 0.9  # High impact for close games
            elif score_diff <= 7:
                game.market_impact = 0.7
            elif score_diff <= 14:
                game.market_impact = 0.5
            else:
                game.market_impact = 0.3  # Low impact for blowouts
    
    def convert_to_standardized_events(self, game: SyntheticGame) -> List[StandardizedEvent]:
        """Convert synthetic game to StandardizedEvent format"""
        
        events = []
        
        for play in game.plays:
            # Determine event type and impact
            if play.touchdown or play.field_goal:
                event_type = EventType.GAME_EVENT
                impact = "high"
            elif play.turnover:
                event_type = EventType.GAME_EVENT
                impact = "high" 
            elif abs(play.yards_gained) >= 15:
                event_type = EventType.GAME_EVENT
                impact = "medium"
            else:
                event_type = EventType.GAME_EVENT
                impact = "low"
            
            # Build event data
            event_data = {
                "play_type": play.play_type,
                "description": play.play_description,
                "quarter": play.context.quarter,
                "time_remaining": play.context.time_remaining,
                "down": play.context.down,
                "distance": play.context.distance,
                "yards_gained": play.yards_gained,
                "score": {
                    "home": play.context.score_home,
                    "away": play.context.score_away,
                    "differential": play.context.score_differential
                },
                "outcomes": {
                    "touchdown": play.touchdown,
                    "field_goal": play.field_goal,
                    "turnover": play.turnover,
                    "safety": play.safety,
                    "penalty": play.penalty
                },
                "synthetic": True  # Mark as synthetic data
            }
            
            event = StandardizedEvent(
                source="synthetic_nfl_engine",
                event_type=event_type,
                timestamp=datetime.now(),
                game_id=play.context.game_id,
                data=event_data,
                confidence=0.99,  # High confidence in synthetic data
                impact=impact,
                metadata={
                    "home_team": game.home_team,
                    "away_team": game.away_team,
                    "season": game.season,
                    "week": game.week,
                    "epa": play.epa,
                    "wpa": play.wpa,
                    "excitement": play.excitement_factor,
                    "synthetic": True
                },
                raw_data=play
            )
            
            events.append(event)
        
        logger.info(f"Converted {len(events)} synthetic plays to StandardizedEvents")
        return events
    
    async def generate_batch_games(self, 
                                 num_games: int = 100,
                                 game_types: List[str] = None,
                                 season: int = 2024,
                                 start_week: int = 1) -> List[SyntheticGame]:
        """
        Generate a batch of synthetic games
        
        Args:
            num_games: Number of games to generate
            game_types: List of game types to include
            season: Season year
            start_week: Starting week number
            
        Returns:
            List of synthetic games
        """
        if game_types is None:
            game_types = ["regular", "high_scoring", "defensive", "weather", "blowout"]
        
        games = []
        week = start_week
        
        logger.info(f"Generating batch of {num_games} synthetic games...")
        
        for i in range(num_games):
            # Select random game type
            game_type = random.choice(game_types)
            
            # Select random teams
            teams = random.sample(self.nfl_teams, 2)
            
            # Generate game
            game = await self.generate_single_game(
                home_team=teams[0],
                away_team=teams[1],
                game_type=game_type,
                season=season,
                week=week
            )
            
            games.append(game)
            
            # Progress logging
            if (i + 1) % 25 == 0:
                logger.info(f"Generated {i + 1}/{num_games} synthetic games")
            
            # Increment week
            week += 1
            if week > 18:
                week = 1
        
        logger.info(f"Batch generation complete: {len(games)} games generated")
        return games


# Example usage and testing
if __name__ == "__main__":
    import sys
    import asyncio
    sys.path.append('/Users/hudson/Documents/GitHub/IntelIP/PROJECTS/Neural/Kalshi_Agentic_Agent')
    
    from src.synthetic_data.storage.chromadb_manager import ChromaDBManager
    
    async def test_game_engine():
        # Initialize components
        chromadb = ChromaDBManager()
        engine = SyntheticGameEngine(chromadb)
        
        # Generate single game
        game = await engine.generate_single_game(
            home_team="KC",
            away_team="BUF", 
            game_type="regular"
        )
        
        print(f"Generated Game: {game.away_team} @ {game.home_team}")
        print(f"Final Score: {game.away_team} {game.final_score[1]} - {game.final_score[0]} {game.home_team}")
        print(f"Total Plays: {game.total_plays}")
        print(f"Game Type: {game.game_type}")
        print(f"Excitement Score: {game.excitement_score:.2f}")
        
        # Show some plays
        print(f"\nFirst 5 plays:")
        for play in game.plays[:5]:
            print(f"  {play.play_type}: {play.play_description}")
        
        # Convert to StandardizedEvents
        events = engine.convert_to_standardized_events(game)
        print(f"Converted to {len(events)} StandardizedEvents")
    
    # Run test
    asyncio.run(test_game_engine())