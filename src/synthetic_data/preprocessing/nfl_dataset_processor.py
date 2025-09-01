"""
NFL Dataset Processor

Processes historical NFL play-by-play CSV data (2009-2016/2017/2018)
and converts to standardized format for machine learning training.
"""

import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

from src.sdk.core.base_adapter import StandardizedEvent, EventType

logger = logging.getLogger(__name__)


@dataclass
class ProcessedNFLPlay:
    """Standardized NFL play data structure"""
    game_id: str
    play_id: str
    date: str
    season: int
    week: int
    
    # Game context
    quarter: int
    time_remaining: str
    time_seconds: int
    home_team: str
    away_team: str
    possession_team: str
    defensive_team: str
    
    # Situation
    down: Optional[int]
    distance: Optional[int]
    yard_line: Optional[int]
    yards_to_goal: Optional[int]
    field_position: str
    
    # Play details
    play_type: str
    play_description: str
    yards_gained: Optional[int]
    
    # Advanced metrics
    expected_points: Optional[float]
    epa: Optional[float]  # Expected Points Added
    win_probability_pre: Optional[float]
    win_probability_post: Optional[float]
    wpa: Optional[float]  # Win Probability Added
    
    # Score context
    score_home: Optional[int]
    score_away: Optional[int]
    score_differential: Optional[int]
    
    # Play outcomes
    touchdown: bool
    field_goal: bool
    turnover: bool
    safety: bool
    penalty: bool
    

class NFLDatasetProcessor:
    """
    Processes NFL CSV datasets and converts to standardized format
    Compatible with existing StandardizedEvent system
    """
    
    def __init__(self, data_dir: str = "data/nfl_source"):
        """
        Initialize processor
        
        Args:
            data_dir: Directory containing NFL CSV files
        """
        self.data_dir = Path(data_dir)
        self.datasets = self._discover_datasets()
        logger.info(f"Discovered {len(self.datasets)} NFL datasets")
        
    def _discover_datasets(self) -> Dict[str, Path]:
        """Discover available NFL datasets"""
        datasets = {}
        
        for csv_file in self.data_dir.glob("*.csv"):
            if "2009-2016" in csv_file.name:
                datasets["2009-2016"] = csv_file
            elif "2009-2017" in csv_file.name:
                datasets["2009-2017"] = csv_file
            elif "2009-2018" in csv_file.name:
                datasets["2009-2018"] = csv_file
                
        return datasets
    
    def load_dataset(self, version: str = "2009-2016") -> pd.DataFrame:
        """
        Load NFL dataset
        
        Args:
            version: Dataset version to load
            
        Returns:
            Pandas DataFrame with NFL data
        """
        if version not in self.datasets:
            raise ValueError(f"Dataset {version} not found. Available: {list(self.datasets.keys())}")
        
        logger.info(f"Loading NFL dataset: {version}")
        
        try:
            df = pd.read_csv(
                self.datasets[version],
                encoding='utf-8-sig',  # Handle BOM
                low_memory=False
            )
            
            logger.info(f"Loaded {len(df)} plays from {version} dataset")
            return df
            
        except Exception as e:
            logger.error(f"Error loading dataset {version}: {e}")
            raise
    
    def process_plays(self, df: pd.DataFrame, limit: Optional[int] = None) -> List[ProcessedNFLPlay]:
        """
        Process raw NFL data into structured plays
        
        Args:
            df: Raw NFL DataFrame
            limit: Optional limit on number of plays to process
            
        Returns:
            List of processed NFL plays
        """
        if limit:
            df = df.head(limit)
        
        processed_plays = []
        
        logger.info(f"Processing {len(df)} NFL plays...")
        
        for idx, row in df.iterrows():
            try:
                play = self._process_single_play(row, idx)
                if play:
                    processed_plays.append(play)
                    
            except Exception as e:
                logger.warning(f"Error processing play {idx}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(processed_plays)} plays")
        return processed_plays
    
    def _process_single_play(self, row: pd.Series, idx: int) -> Optional[ProcessedNFLPlay]:
        """Process a single play row"""
        try:
            # Skip invalid plays
            if pd.isna(row.get('desc')) or pd.isna(row.get('GameID')):
                return None
            
            # Parse play type
            play_type = self._determine_play_type(row)
            
            # Calculate field position
            field_pos = self._calculate_field_position(row)
            
            # Determine outcomes
            touchdown = bool(row.get('Touchdown', 0))
            field_goal = 'Field Goal' in str(row.get('desc', ''))
            turnover = bool(row.get('InterceptionThrown', 0)) or bool(row.get('Fumble', 0))
            safety = bool(row.get('Safety', 0))
            penalty = bool(row.get('Accepted.Penalty', 0))
            
            # Create truly unique play ID using multiple NFL dataset fields
            play_unique_id = self._generate_unique_play_id(row, idx)
            
            return ProcessedNFLPlay(
                game_id=str(row['GameID']),
                play_id=play_unique_id,
                date=str(row['Date']),
                season=int(row.get('Season', 0)),
                week=self._extract_week_from_date(str(row['Date'])),
                
                # Game context
                quarter=self._safe_int(row.get('qtr')),
                time_remaining=str(row.get('time', '')),
                time_seconds=self._safe_int(row.get('TimeSecs')),
                home_team=str(row.get('HomeTeam', '')),
                away_team=str(row.get('AwayTeam', '')),
                possession_team=str(row.get('posteam', '')),
                defensive_team=str(row.get('DefensiveTeam', '')),
                
                # Situation
                down=self._safe_int(row.get('down')),
                distance=self._safe_int(row.get('ydstogo')),
                yard_line=self._safe_int(row.get('yrdln')),
                yards_to_goal=self._safe_int(row.get('yrdline100')),
                field_position=field_pos,
                
                # Play details
                play_type=play_type,
                play_description=str(row.get('desc', '')),
                yards_gained=self._safe_int(row.get('Yards.Gained')),
                
                # Advanced metrics
                expected_points=self._safe_float(row.get('ExpPts')),
                epa=self._safe_float(row.get('EPA')),
                win_probability_pre=self._safe_float(row.get('Win_Prob')),
                win_probability_post=self._safe_float(row.get('Home_WP_post')) if row.get('posteam') == row.get('HomeTeam') else self._safe_float(row.get('Away_WP_post')),
                wpa=self._safe_float(row.get('WPA')),
                
                # Score context  
                score_home=self._safe_int(row.get('PosTeamScore')) if row.get('posteam') == row.get('HomeTeam') else self._safe_int(row.get('DefTeamScore')),
                score_away=self._safe_int(row.get('DefTeamScore')) if row.get('posteam') == row.get('HomeTeam') else self._safe_int(row.get('PosTeamScore')),
                score_differential=self._safe_int(row.get('ScoreDiff')),
                
                # Outcomes
                touchdown=touchdown,
                field_goal=field_goal,
                turnover=turnover,
                safety=safety,
                penalty=penalty
            )
            
        except Exception as e:
            logger.warning(f"Error processing play: {e}")
            return None
    
    def _determine_play_type(self, row: pd.Series) -> str:
        """Determine standardized play type from raw data"""
        play_type = str(row.get('PlayType', ''))
        
        # Map to standardized types
        type_mapping = {
            'Pass': 'PASS',
            'Rush': 'RUN', 
            'Run': 'RUN',
            'Punt': 'PUNT',
            'Field Goal': 'FIELD_GOAL',
            'Kickoff': 'KICKOFF',
            'Sack': 'SACK',
            'Spike': 'SPIKE',
            'Kneel': 'KNEEL',
            'No Play': 'PENALTY'
        }
        
        return type_mapping.get(play_type, 'UNKNOWN')
    
    def _calculate_field_position(self, row: pd.Series) -> str:
        """Calculate field position description"""
        side = str(row.get('SideofField', ''))
        yard_line = self._safe_int(row.get('yrdln'))
        
        if side and yard_line:
            return f"{side} {yard_line}"
        return "UNKNOWN"
    
    def _extract_week_from_date(self, date_str: str) -> int:
        """Extract week number from date (simplified)"""
        try:
            # This is a simplified approach - you might want to implement 
            # proper NFL week calculation based on season start dates
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            # September games are generally weeks 1-4
            if date_obj.month == 9:
                return min(4, (date_obj.day // 7) + 1)
            elif date_obj.month == 10:
                return min(8, 4 + (date_obj.day // 7) + 1)
            elif date_obj.month == 11:
                return min(12, 8 + (date_obj.day // 7) + 1)
            elif date_obj.month == 12:
                return min(16, 12 + (date_obj.day // 7) + 1)
            else:
                return 1  # Default
        except:
            return 1
    
    def _safe_int(self, value) -> Optional[int]:
        """Safely convert to int"""
        if pd.isna(value) or value == 'NA':
            return None
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return None
    
    def _safe_float(self, value) -> Optional[float]:
        """Safely convert to float"""
        if pd.isna(value) or value == 'NA':
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _generate_unique_play_id(self, row: pd.Series, idx: int) -> str:
        """Generate truly unique play ID using multiple NFL dataset fields"""
        import hashlib
        
        # Core identifiers
        game_id = str(row.get('GameID', ''))
        time_secs = str(row.get('TimeSecs', ''))
        drive = str(row.get('Drive', ''))
        play_desc = str(row.get('desc', ''))
        
        # Additional uniqueness factors
        qtr = str(row.get('qtr', ''))
        down = str(row.get('down', ''))
        ydstogo = str(row.get('ydstogo', ''))
        yrdln = str(row.get('yrdln', ''))
        
        # Create composite string for hashing
        composite_string = f"{game_id}_{time_secs}_{drive}_{qtr}_{down}_{ydstogo}_{yrdln}_{play_desc}"
        
        # Generate hash of the description and situation for extra uniqueness
        play_hash = hashlib.md5(composite_string.encode()).hexdigest()[:8]
        
        # Combine with fallback to row index
        unique_id = f"{game_id}_{time_secs}_{drive}_{play_hash}_{idx}"
        
        return unique_id
    
    def to_standardized_events(self, processed_plays: List[ProcessedNFLPlay]) -> List[StandardizedEvent]:
        """
        Convert processed plays to StandardizedEvent format
        Compatible with existing event system
        """
        events = []
        
        for play in processed_plays:
            try:
                # Determine event type
                if play.touchdown:
                    event_type = EventType.GAME_EVENT
                    impact = "high"
                elif play.turnover:
                    event_type = EventType.GAME_EVENT  
                    impact = "high"
                elif play.field_goal:
                    event_type = EventType.GAME_EVENT
                    impact = "medium"
                else:
                    event_type = EventType.GAME_EVENT
                    impact = "low"
                
                # Build event data
                event_data = {
                    "play_type": play.play_type,
                    "description": play.play_description,
                    "quarter": play.quarter,
                    "time_remaining": play.time_remaining,
                    "down": play.down,
                    "distance": play.distance,
                    "field_position": play.field_position,
                    "yards_gained": play.yards_gained,
                    "possession_team": play.possession_team,
                    "score": {
                        "home": play.score_home,
                        "away": play.score_away,
                        "differential": play.score_differential
                    },
                    "outcomes": {
                        "touchdown": play.touchdown,
                        "field_goal": play.field_goal,
                        "turnover": play.turnover,
                        "safety": play.safety,
                        "penalty": play.penalty
                    }
                }
                
                # Create standardized event
                event = StandardizedEvent(
                    source="nfl_historical",
                    event_type=event_type,
                    timestamp=datetime.strptime(play.date, '%Y-%m-%d'),
                    game_id=play.game_id,
                    data=event_data,
                    confidence=0.99,  # Historical data is highly reliable
                    impact=impact,
                    metadata={
                        "season": play.season,
                        "week": play.week,
                        "home_team": play.home_team,
                        "away_team": play.away_team,
                        "epa": play.epa,
                        "wpa": play.wpa
                    },
                    raw_data=play
                )
                
                events.append(event)
                
            except Exception as e:
                logger.warning(f"Error converting play to StandardizedEvent: {e}")
                continue
        
        logger.info(f"Converted {len(events)} plays to StandardizedEvents")
        return events


# Example usage and testing
if __name__ == "__main__":
    import sys
    sys.path.append('/Users/hudson/Documents/GitHub/IntelIP/PROJECTS/Neural/Kalshi_Agentic_Agent')
    
    # Initialize processor
    processor = NFLDatasetProcessor()
    
    # Load small sample for testing
    df = processor.load_dataset("2009-2016")
    sample_df = df.head(1000)  # Test with first 1000 plays
    
    # Process plays
    processed_plays = processor.process_plays(sample_df)
    print(f"Processed {len(processed_plays)} plays")
    
    # Convert to StandardizedEvents
    events = processor.to_standardized_events(processed_plays)
    print(f"Created {len(events)} StandardizedEvents")
    
    # Show sample
    if events:
        sample_event = events[0]
        print(f"\nSample event:")
        print(f"Type: {sample_event.event_type}")
        print(f"Description: {sample_event.data['description']}")
        print(f"Impact: {sample_event.impact}")