"""
NFL Play Data Validator

Validates the completeness and quality of NFL play data
before processing and storage.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import pandas as pd

from ..preprocessing.nfl_dataset_processor import ProcessedNFLPlay

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Results of data validation"""
    is_valid: bool
    total_plays: int
    valid_plays: int
    invalid_plays: int
    warnings: List[str]
    errors: List[str]
    quality_score: float  # 0.0 to 1.0
    
    def __post_init__(self):
        """Calculate quality score after initialization"""
        if self.total_plays > 0:
            self.quality_score = self.valid_plays / self.total_plays
        else:
            self.quality_score = 0.0


@dataclass 
class PlayValidationRule:
    """Individual validation rule for NFL plays"""
    name: str
    required: bool
    validator_func: callable
    error_message: str
    weight: float = 1.0  # For weighted scoring


class PlayDataValidator:
    """
    Validates NFL play data for completeness and quality
    """
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize validator
        
        Args:
            strict_mode: If True, apply stricter validation rules
        """
        self.strict_mode = strict_mode
        self.validation_rules = self._initialize_validation_rules()
        
    def _initialize_validation_rules(self) -> List[PlayValidationRule]:
        """Initialize validation rules for NFL plays"""
        rules = [
            # Critical fields
            PlayValidationRule(
                name="game_id_present",
                required=True,
                validator_func=lambda play: play.game_id and play.game_id != "nan",
                error_message="Game ID is missing or invalid",
                weight=2.0
            ),
            PlayValidationRule(
                name="play_id_present", 
                required=True,
                validator_func=lambda play: play.play_id and play.play_id != "nan",
                error_message="Play ID is missing or invalid",
                weight=2.0
            ),
            PlayValidationRule(
                name="description_present",
                required=True,
                validator_func=lambda play: play.play_description and len(play.play_description.strip()) > 5,
                error_message="Play description is missing or too short",
                weight=2.0
            ),
            
            # Game context validation
            PlayValidationRule(
                name="valid_quarter",
                required=True,
                validator_func=lambda play: play.quarter and 1 <= play.quarter <= 5,
                error_message="Quarter must be between 1 and 5",
                weight=1.5
            ),
            PlayValidationRule(
                name="valid_down",
                required=False,
                validator_func=lambda play: play.down is None or (1 <= play.down <= 4),
                error_message="Down must be between 1 and 4 when present",
                weight=1.0
            ),
            PlayValidationRule(
                name="valid_distance",
                required=False,
                validator_func=lambda play: play.distance is None or (0 <= play.distance <= 99),
                error_message="Distance must be between 0 and 99 yards when present",
                weight=1.0
            ),
            PlayValidationRule(
                name="valid_yard_line",
                required=False,
                validator_func=lambda play: play.yard_line is None or (0 <= play.yard_line <= 100),
                error_message="Yard line must be between 0 and 100 when present", 
                weight=1.0
            ),
            
            # Team information
            PlayValidationRule(
                name="home_team_present",
                required=True,
                validator_func=lambda play: play.home_team and len(play.home_team) >= 2,
                error_message="Home team is missing or invalid",
                weight=1.5
            ),
            PlayValidationRule(
                name="away_team_present",
                required=True,
                validator_func=lambda play: play.away_team and len(play.away_team) >= 2, 
                error_message="Away team is missing or invalid",
                weight=1.5
            ),
            PlayValidationRule(
                name="possession_team_present",
                required=True,
                validator_func=lambda play: play.possession_team and len(play.possession_team) >= 2,
                error_message="Possession team is missing or invalid",
                weight=1.5
            ),
            
            # Play type validation
            PlayValidationRule(
                name="valid_play_type",
                required=True,
                validator_func=lambda play: play.play_type in ['PASS', 'RUN', 'PUNT', 'FIELD_GOAL', 'KICKOFF', 'SACK', 'SPIKE', 'KNEEL', 'PENALTY', 'UNKNOWN'],
                error_message="Play type must be a recognized value",
                weight=1.5
            ),
            
            # Season and date validation
            PlayValidationRule(
                name="valid_season",
                required=True,
                validator_func=lambda play: play.season and 2009 <= play.season <= datetime.now().year,
                error_message="Season must be between 2009 and current year",
                weight=1.0
            ),
            PlayValidationRule(
                name="valid_week",
                required=True,
                validator_func=lambda play: play.week and 1 <= play.week <= 22,
                error_message="Week must be between 1 and 22",
                weight=1.0
            ),
            
            # Data quality checks
            PlayValidationRule(
                name="reasonable_yards_gained", 
                required=False,
                validator_func=lambda play: play.yards_gained is None or (-50 <= play.yards_gained <= 99),
                error_message="Yards gained seems unreasonable (should be between -50 and 99)",
                weight=0.5
            ),
            PlayValidationRule(
                name="consistent_score_differential",
                required=False,
                validator_func=lambda play: (
                    play.score_differential is None or 
                    play.score_home is None or 
                    play.score_away is None or
                    play.score_differential == (play.score_home - play.score_away)
                ),
                error_message="Score differential doesn't match home/away scores",
                weight=0.5
            )
        ]
        
        # Add strict mode rules
        if self.strict_mode:
            rules.extend([
                PlayValidationRule(
                    name="advanced_metrics_present",
                    required=True,
                    validator_func=lambda play: play.epa is not None and play.wpa is not None,
                    error_message="Advanced metrics (EPA, WPA) required in strict mode",
                    weight=1.0
                ),
                PlayValidationRule(
                    name="complete_score_info",
                    required=True,
                    validator_func=lambda play: (
                        play.score_home is not None and 
                        play.score_away is not None and
                        play.score_differential is not None
                    ),
                    error_message="Complete score information required in strict mode",
                    weight=1.0
                )
            ])
        
        return rules
    
    def validate_single_play(self, play: ProcessedNFLPlay) -> Tuple[bool, List[str], List[str]]:
        """
        Validate a single NFL play
        
        Args:
            play: Processed NFL play to validate
            
        Returns:
            Tuple of (is_valid, warnings, errors)
        """
        warnings = []
        errors = []
        
        for rule in self.validation_rules:
            try:
                result = rule.validator_func(play)
                
                if not result:
                    if rule.required:
                        errors.append(f"{rule.name}: {rule.error_message}")
                    else:
                        warnings.append(f"{rule.name}: {rule.error_message}")
                        
            except Exception as e:
                error_msg = f"{rule.name}: Validation failed - {e}"
                if rule.required:
                    errors.append(error_msg)
                else:
                    warnings.append(error_msg)
        
        is_valid = len(errors) == 0
        return is_valid, warnings, errors
    
    def validate_play_list(self, plays: List[ProcessedNFLPlay]) -> ValidationResult:
        """
        Validate a list of NFL plays
        
        Args:
            plays: List of processed NFL plays
            
        Returns:
            ValidationResult with detailed validation info
        """
        if not plays:
            return ValidationResult(
                is_valid=False,
                total_plays=0,
                valid_plays=0,
                invalid_plays=0,
                warnings=[],
                errors=["No plays provided for validation"],
                quality_score=0.0
            )
        
        all_warnings = []
        all_errors = []
        valid_count = 0
        
        logger.info(f"Validating {len(plays)} plays...")
        
        for i, play in enumerate(plays):
            try:
                is_valid, warnings, errors = self.validate_single_play(play)
                
                if is_valid:
                    valid_count += 1
                
                # Add play context to messages
                if warnings:
                    for warning in warnings:
                        all_warnings.append(f"Play {i+1} ({play.play_id}): {warning}")
                        
                if errors:
                    for error in errors:
                        all_errors.append(f"Play {i+1} ({play.play_id}): {error}")
                        
            except Exception as e:
                error_msg = f"Play {i+1}: Validation exception - {e}"
                all_errors.append(error_msg)
                logger.error(error_msg)
        
        result = ValidationResult(
            is_valid=(valid_count == len(plays)),
            total_plays=len(plays),
            valid_plays=valid_count,
            invalid_plays=len(plays) - valid_count,
            warnings=all_warnings,
            errors=all_errors,
            quality_score=0.0  # Will be calculated in __post_init__
        )
        
        logger.info(f"Validation complete: {result.valid_plays}/{result.total_plays} plays valid (quality: {result.quality_score:.2%})")
        
        return result
    
    def filter_valid_plays(self, plays: List[ProcessedNFLPlay]) -> Tuple[List[ProcessedNFLPlay], ValidationResult]:
        """
        Filter plays to return only valid ones
        
        Args:
            plays: List of processed NFL plays
            
        Returns:
            Tuple of (valid_plays, validation_result)
        """
        if not plays:
            return [], ValidationResult(
                is_valid=False,
                total_plays=0,
                valid_plays=0,
                invalid_plays=0,
                warnings=[],
                errors=["No plays provided"],
                quality_score=0.0
            )
        
        valid_plays = []
        validation_result = self.validate_play_list(plays)
        
        for i, play in enumerate(plays):
            is_valid, _, _ = self.validate_single_play(play)
            if is_valid:
                valid_plays.append(play)
        
        logger.info(f"Filtered to {len(valid_plays)} valid plays from {len(plays)} total")
        
        return valid_plays, validation_result
    
    def get_data_quality_report(self, plays: List[ProcessedNFLPlay]) -> Dict[str, Any]:
        """
        Generate comprehensive data quality report
        
        Args:
            plays: List of processed NFL plays
            
        Returns:
            Detailed quality report
        """
        validation_result = self.validate_play_list(plays)
        
        # Calculate rule-specific failure rates
        rule_failures = {}
        for rule in self.validation_rules:
            failures = 0
            for play in plays:
                try:
                    if not rule.validator_func(play):
                        failures += 1
                except:
                    failures += 1
            rule_failures[rule.name] = {
                'failures': failures,
                'failure_rate': failures / len(plays) if plays else 0.0,
                'required': rule.required
            }
        
        # Calculate completeness metrics
        completeness = {}
        if plays:
            sample_play = plays[0]
            for field in ['game_id', 'play_description', 'quarter', 'down', 'distance', 
                         'possession_team', 'yards_gained', 'epa', 'wpa', 'score_home']:
                non_null_count = sum(1 for play in plays if getattr(play, field) is not None)
                completeness[field] = non_null_count / len(plays)
        
        return {
            'validation_summary': {
                'total_plays': validation_result.total_plays,
                'valid_plays': validation_result.valid_plays,
                'invalid_plays': validation_result.invalid_plays,
                'quality_score': validation_result.quality_score,
                'is_valid': validation_result.is_valid
            },
            'rule_failures': rule_failures,
            'completeness_metrics': completeness,
            'validation_errors': validation_result.errors[:50],  # Limit for readability
            'validation_warnings': validation_result.warnings[:50]
        }


# Example usage and testing
if __name__ == "__main__":
    import sys
    sys.path.append('/Users/hudson/Documents/GitHub/IntelIP/PROJECTS/Neural/Kalshi_Agentic_Agent')
    
    from src.synthetic_data.preprocessing.nfl_dataset_processor import NFLDatasetProcessor
    
    # Test validation
    processor = NFLDatasetProcessor(data_dir='data/nfl_source')
    validator = PlayDataValidator(strict_mode=False)
    
    # Load and process sample data
    df = processor.load_dataset('2009-2016')
    sample_df = df.head(50)
    processed_plays = processor.process_plays(sample_df)
    
    # Validate
    validation_result = validator.validate_play_list(processed_plays)
    print(f"Validation Result: {validation_result.quality_score:.2%} quality score")
    print(f"Valid plays: {validation_result.valid_plays}/{validation_result.total_plays}")
    
    if validation_result.errors:
        print(f"Sample errors: {validation_result.errors[:3]}")
    
    # Get quality report
    quality_report = validator.get_data_quality_report(processed_plays)
    print(f"Data completeness: {quality_report['completeness_metrics']}")
    
    print("PlayDataValidator test completed!")