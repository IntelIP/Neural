"""
Training Data Builder for LFM2 Fine-tuning

Converts processed NFL plays into training sequences
suitable for fine-tuning language models.
"""

import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from .nfl_dataset_processor import ProcessedNFLPlay

logger = logging.getLogger(__name__)


@dataclass 
class TrainingSequence:
    """Training sequence for language model fine-tuning"""
    context: str          # Game context and situation
    target: str          # Expected play outcome
    metadata: Dict[str, Any]  # Additional information


class TrainingDataBuilder:
    """
    Builds training sequences for LFM2 fine-tuning
    from processed NFL play data
    """
    
    def __init__(self):
        """Initialize training data builder"""
        self.sequences = []
        
    def build_sequences(self, plays: List[ProcessedNFLPlay], sequence_length: int = 5) -> List[TrainingSequence]:
        """
        Build training sequences from NFL plays
        
        Args:
            plays: List of processed NFL plays
            sequence_length: Number of plays per sequence
            
        Returns:
            List of training sequences
        """
        # Group plays by game
        games = self._group_plays_by_game(plays)
        
        sequences = []
        
        for game_id, game_plays in games.items():
            game_sequences = self._create_game_sequences(game_plays, sequence_length)
            sequences.extend(game_sequences)
        
        logger.info(f"Built {len(sequences)} training sequences from {len(plays)} plays")
        return sequences
    
    def _group_plays_by_game(self, plays: List[ProcessedNFLPlay]) -> Dict[str, List[ProcessedNFLPlay]]:
        """Group plays by game ID"""
        games = {}
        
        for play in plays:
            if play.game_id not in games:
                games[play.game_id] = []
            games[play.game_id].append(play)
        
        # Sort plays within each game by time
        for game_id in games:
            games[game_id].sort(key=lambda p: (p.quarter or 0, -(p.time_seconds or 0)))
        
        return games
    
    def _create_game_sequences(self, plays: List[ProcessedNFLPlay], sequence_length: int) -> List[TrainingSequence]:
        """Create training sequences from a single game"""
        sequences = []
        
        for i in range(len(plays) - sequence_length + 1):
            sequence_plays = plays[i:i + sequence_length]
            
            # Use first N-1 plays as context, last play as target
            context_plays = sequence_plays[:-1]
            target_play = sequence_plays[-1]
            
            context = self._build_context(context_plays)
            target = self._build_target(target_play)
            
            metadata = {
                "game_id": target_play.game_id,
                "season": target_play.season,
                "sequence_start": i,
                "plays_count": len(context_plays)
            }
            
            sequence = TrainingSequence(
                context=context,
                target=target, 
                metadata=metadata
            )
            
            sequences.append(sequence)
        
        return sequences
    
    def _build_context(self, plays: List[ProcessedNFLPlay]) -> str:
        """Build context string from sequence of plays"""
        context_parts = []
        
        for play in plays:
            play_context = self._format_play_context(play)
            context_parts.append(play_context)
        
        return " | ".join(context_parts)
    
    def _format_play_context(self, play: ProcessedNFLPlay) -> str:
        """Format single play as context"""
        parts = []
        
        # Game situation
        if play.quarter and play.time_remaining:
            parts.append(f"Q{play.quarter} {play.time_remaining}")
        
        # Down and distance
        if play.down and play.distance:
            parts.append(f"{play.down}&{play.distance}")
        
        # Field position
        if play.yards_to_goal:
            parts.append(f"Y{play.yards_to_goal}")
        
        # Score differential
        if play.score_differential is not None:
            if play.score_differential > 0:
                parts.append(f"+{play.score_differential}")
            elif play.score_differential < 0:
                parts.append(f"{play.score_differential}")
            else:
                parts.append("TIE")
        
        # Play type and result
        parts.append(f"{play.play_type}")
        if play.yards_gained is not None:
            parts.append(f"{play.yards_gained}yd")
        
        # Special outcomes
        if play.touchdown:
            parts.append("TD")
        elif play.field_goal:
            parts.append("FG")
        elif play.turnover:
            parts.append("TO")
        
        return " ".join(parts)
    
    def _build_target(self, play: ProcessedNFLPlay) -> str:
        """Build target string for play prediction"""
        target_parts = []
        
        # Play type
        target_parts.append(f"PLAY:{play.play_type}")
        
        # Expected outcome
        if play.yards_gained is not None:
            target_parts.append(f"YARDS:{play.yards_gained}")
        
        # Special outcomes
        outcomes = []
        if play.touchdown:
            outcomes.append("TD")
        if play.field_goal:
            outcomes.append("FG") 
        if play.turnover:
            outcomes.append("TO")
        if play.safety:
            outcomes.append("SAFETY")
        
        if outcomes:
            target_parts.append(f"OUTCOME:{','.join(outcomes)}")
        
        # Performance metrics
        if play.epa is not None:
            target_parts.append(f"EPA:{play.epa:.2f}")
        
        return " ".join(target_parts)
    
    def export_for_training(self, sequences: List[TrainingSequence], format: str = "jsonl") -> str:
        """
        Export training sequences in specified format
        
        Args:
            sequences: Training sequences
            format: Export format ('jsonl', 'csv', 'txt')
            
        Returns:
            Formatted training data string
        """
        if format == "jsonl":
            return self._export_jsonl(sequences)
        elif format == "csv":
            return self._export_csv(sequences)
        elif format == "txt":
            return self._export_txt(sequences)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_jsonl(self, sequences: List[TrainingSequence]) -> str:
        """Export as JSONL format"""
        import json
        
        lines = []
        for seq in sequences:
            record = {
                "context": seq.context,
                "target": seq.target,
                "metadata": seq.metadata
            }
            lines.append(json.dumps(record))
        
        return "\n".join(lines)
    
    def _export_csv(self, sequences: List[TrainingSequence]) -> str:
        """Export as CSV format"""
        import csv
        from io import StringIO
        
        output = StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow(["context", "target", "game_id", "season"])
        
        # Data
        for seq in sequences:
            writer.writerow([
                seq.context,
                seq.target,
                seq.metadata.get("game_id", ""),
                seq.metadata.get("season", "")
            ])
        
        return output.getvalue()
    
    def _export_txt(self, sequences: List[TrainingSequence]) -> str:
        """Export as text format for language model training"""
        lines = []
        
        for seq in sequences:
            # Format as: Context -> Target
            line = f"{seq.context} -> {seq.target}"
            lines.append(line)
        
        return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    # This would be used with processed NFL plays
    builder = TrainingDataBuilder()
    print("TrainingDataBuilder initialized successfully!")