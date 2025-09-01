"""
ChromaDB Manager for NFL Data

Manages ChromaDB collections for storing and retrieving
NFL play patterns, game scenarios, and synthetic data.
"""

import chromadb
from chromadb.config import Settings
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import json

from ..preprocessing.nfl_dataset_processor import ProcessedNFLPlay
from src.sdk.core.base_adapter import StandardizedEvent

logger = logging.getLogger(__name__)


class ChromaDBManager:
    """
    Manages ChromaDB collections for NFL synthetic data system
    """
    
    def __init__(self, persist_directory: str = "data/chromadb"):
        """
        Initialize ChromaDB manager
        
        Args:
            persist_directory: Directory to store ChromaDB data
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )
        
        # Initialize collections
        self.collections = self._initialize_collections()
        logger.info(f"Initialized ChromaDB with {len(self.collections)} collections")
    
    def _initialize_collections(self) -> Dict[str, chromadb.Collection]:
        """Initialize all NFL data collections"""
        collections = {}
        
        # Collection 1: NFL Play Patterns
        collections["nfl_play_patterns"] = self.client.get_or_create_collection(
            name="nfl_play_patterns",
            metadata={
                "description": "Historical NFL play sequences and patterns 2009-2018",
                "source": "nfl_dataset",
                "purpose": "pattern_learning"
            }
        )
        
        # Collection 2: Game Scenarios
        collections["game_scenarios"] = self.client.get_or_create_collection(
            name="game_scenarios", 
            metadata={
                "description": "Specific game situation templates for generation",
                "source": "synthetic",
                "purpose": "scenario_generation"
            }
        )
        
        # Collection 3: Team Tendencies
        collections["team_tendencies"] = self.client.get_or_create_collection(
            name="team_tendencies",
            metadata={
                "description": "Team-specific play calling and behavioral patterns",
                "source": "nfl_dataset",
                "purpose": "team_modeling"
            }
        )
        
        # Collection 4: Synthetic Learnings
        collections["synthetic_learnings"] = self.client.get_or_create_collection(
            name="synthetic_learnings",
            metadata={
                "description": "Agent-discovered patterns and successful strategies",
                "source": "agent_training",
                "purpose": "adaptive_learning"
            }
        )
        
        return collections
    
    def add_nfl_plays(self, plays: List[ProcessedNFLPlay], upsert: bool = False) -> int:
        """
        Add NFL plays to the patterns collection with duplicate detection
        
        Args:
            plays: List of processed NFL plays
            upsert: If True, update existing plays instead of skipping
            
        Returns:
            Number of plays added/updated
        """
        if not plays:
            return 0
        
        collection = self.collections["nfl_play_patterns"]
        
        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        
        # Check for existing plays if not upserting
        existing_ids = set()
        if not upsert:
            try:
                # Get all existing IDs
                existing_data = collection.get()
                existing_ids = set(existing_data['ids']) if existing_data['ids'] else set()
                logger.debug(f"Found {len(existing_ids)} existing plays in ChromaDB")
            except Exception as e:
                logger.warning(f"Could not check for existing plays: {e}")
        
        skipped_count = 0
        for play in plays:
            # Create unique ID using play_id directly (already unique from processor)
            play_id = f"{play.game_id}_play_{play.play_id}"
            
            # Skip duplicates if not upserting
            if not upsert and play_id in existing_ids:
                skipped_count += 1
                continue
            
            ids.append(play_id)
            
            # Create searchable document text
            doc_text = self._create_play_document(play)
            documents.append(doc_text)
            
            # Create metadata
            metadata = self._create_play_metadata(play)
            metadatas.append(metadata)
        
        if not ids:
            logger.info(f"No new plays to add. Skipped {skipped_count} duplicates")
            return 0
        
        try:
            if upsert:
                # Use upsert for updating existing plays
                collection.upsert(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
                operation = "upserted"
            else:
                # Regular add operation
                collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
                operation = "added"
            
            logger.info(f"Successfully {operation} {len(ids)} NFL plays to ChromaDB. Skipped {skipped_count} duplicates")
            return len(ids)
            
        except Exception as e:
            logger.error(f"Error {operation.replace('ed', 'ing')} plays to ChromaDB: {e}")
            return 0
    
    def _create_play_document(self, play: ProcessedNFLPlay) -> str:
        """Create searchable text document from play data"""
        
        # Build contextual description
        context_parts = []
        
        # Game situation
        if play.quarter and play.time_remaining:
            context_parts.append(f"Quarter {play.quarter}, {play.time_remaining} remaining")
        
        # Down and distance
        if play.down and play.distance:
            context_parts.append(f"{play.down} and {play.distance}")
        
        # Field position
        if play.field_position:
            context_parts.append(f"at {play.field_position}")
        
        # Team context
        if play.possession_team:
            context_parts.append(f"{play.possession_team} has possession")
        
        # Score situation
        if play.score_differential is not None:
            if play.score_differential > 0:
                context_parts.append(f"leading by {play.score_differential}")
            elif play.score_differential < 0:
                context_parts.append(f"trailing by {abs(play.score_differential)}")
            else:
                context_parts.append("tied game")
        
        context = ". ".join(context_parts)
        
        # Play description and outcome
        outcome_parts = []
        if play.yards_gained is not None:
            outcome_parts.append(f"Gained {play.yards_gained} yards")
        
        if play.touchdown:
            outcome_parts.append("TOUCHDOWN")
        elif play.field_goal:
            outcome_parts.append("FIELD GOAL")
        elif play.turnover:
            outcome_parts.append("TURNOVER")
        elif play.safety:
            outcome_parts.append("SAFETY")
        
        outcome = ". ".join(outcome_parts)
        
        # Combine into searchable document
        document = f"{context}. Play: {play.play_description}. Result: {outcome}"
        
        # Add performance metrics if available
        if play.epa is not None:
            document += f". EPA: {play.epa:.2f}"
        if play.wpa is not None:
            document += f". WPA: {play.wpa:.3f}"
        
        return document
    
    def _create_play_metadata(self, play: ProcessedNFLPlay) -> Dict[str, Any]:
        """Create metadata dict for play"""
        metadata = {
            "game_id": play.game_id,
            "season": play.season,
            "week": play.week,
            "quarter": play.quarter or 0,
            "down": play.down or 0,
            "distance": play.distance or 0,
            "yards_to_goal": play.yards_to_goal or 0,
            "play_type": play.play_type,
            "possession_team": play.possession_team,
            "home_team": play.home_team,
            "away_team": play.away_team,
            "yards_gained": play.yards_gained or 0,
            "touchdown": play.touchdown,
            "field_goal": play.field_goal,
            "turnover": play.turnover,
            "safety": play.safety,
            "penalty": play.penalty
        }
        
        # Add optional metrics
        if play.epa is not None:
            metadata["epa"] = round(play.epa, 3)
        if play.wpa is not None:
            metadata["wpa"] = round(play.wpa, 4)
        if play.score_differential is not None:
            metadata["score_diff"] = play.score_differential
        
        return metadata
    
    def search_similar_plays(
        self, 
        query: str, 
        collection_name: str = "nfl_play_patterns",
        n_results: int = 10,
        filters: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Search for similar plays based on situation
        
        Args:
            query: Natural language description of situation
            collection_name: Collection to search
            n_results: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            Search results with documents and metadata
        """
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} not found")
        
        collection = self.collections[collection_name]
        
        try:
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filters,
                include=['documents', 'metadatas', 'distances']
            )
            
            logger.debug(f"Found {len(results['ids'][0])} similar plays for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching plays: {e}")
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    
    def add_game_scenario(
        self, 
        scenario_name: str, 
        description: str, 
        template_data: Dict[str, Any]
    ) -> bool:
        """
        Add a game scenario template
        
        Args:
            scenario_name: Unique name for scenario
            description: Description of the scenario
            template_data: Template parameters
            
        Returns:
            Success status
        """
        collection = self.collections["game_scenarios"]
        
        try:
            collection.add(
                ids=[scenario_name],
                documents=[description],
                metadatas=[{
                    "scenario_type": template_data.get("type", "general"),
                    "weather": template_data.get("weather", "clear"),
                    "game_stakes": template_data.get("stakes", "regular"),
                    "expected_plays": template_data.get("expected_plays", 140),
                    "template_json": json.dumps(template_data)
                }]
            )
            
            logger.info(f"Added scenario: {scenario_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding scenario: {e}")
            return False
    
    def get_team_tendencies(self, team: str, situation: str = None) -> List[Dict]:
        """
        Get team-specific play calling tendencies
        
        Args:
            team: Team code (e.g., 'KC', 'NE')
            situation: Optional situation filter
            
        Returns:
            List of relevant plays/tendencies
        """
        query = f"{team} team tendencies"
        if situation:
            query += f" in {situation}"
        
        results = self.search_similar_plays(
            query=query,
            collection_name="team_tendencies", 
            filters={"possession_team": team} if team else None
        )
        
        return self._format_search_results(results)
    
    def add_synthetic_learning(
        self, 
        agent_name: str,
        pattern_description: str,
        success_metrics: Dict[str, float],
        context: Dict[str, Any]
    ) -> bool:
        """
        Add agent-discovered pattern
        
        Args:
            agent_name: Name of discovering agent
            pattern_description: Description of discovered pattern
            success_metrics: Performance metrics
            context: Context information
            
        Returns:
            Success status
        """
        collection = self.collections["synthetic_learnings"]
        
        learning_id = f"{agent_name}_{len(collection.get()['ids'])}"
        
        try:
            metadata = {
                "agent": agent_name,
                "success_rate": success_metrics.get("success_rate", 0.0),
                "profit": success_metrics.get("profit", 0.0),
                "pattern_type": context.get("pattern_type", "unknown"),
                "discovery_date": context.get("date", "unknown")
            }
            
            collection.add(
                ids=[learning_id],
                documents=[pattern_description],
                metadatas=[metadata]
            )
            
            logger.info(f"Added synthetic learning from {agent_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding synthetic learning: {e}")
            return False
    
    def _format_search_results(self, results: Dict[str, Any]) -> List[Dict]:
        """Format ChromaDB search results into convenient structure"""
        formatted = []
        
        if not results.get('ids') or not results['ids'][0]:
            return formatted
        
        for i in range(len(results['ids'][0])):
            formatted.append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })
        
        return formatted
    
    def add_nfl_plays_batch(self, plays: List[ProcessedNFLPlay], batch_size: int = 1000, upsert: bool = False) -> int:
        """
        Add NFL plays in batches with transaction rollback capability
        
        Args:
            plays: List of processed NFL plays
            batch_size: Size of each batch
            upsert: If True, update existing plays instead of skipping
            
        Returns:
            Total number of plays added/updated successfully
        """
        if not plays:
            return 0
        
        collection = self.collections["nfl_play_patterns"]
        total_added = 0
        successful_batches = []
        
        # Process in batches
        for i in range(0, len(plays), batch_size):
            batch = plays[i:i + batch_size]
            batch_number = i // batch_size + 1
            
            logger.info(f"Processing batch {batch_number}/{(len(plays) + batch_size - 1) // batch_size} ({len(batch)} plays)")
            
            try:
                # Create backup of batch IDs for potential rollback
                batch_ids = [f"{play.game_id}_play_{play.play_id}" for play in batch]
                
                # Process the batch
                batch_added = self.add_nfl_plays(batch, upsert=upsert)
                
                if batch_added > 0:
                    successful_batches.append({
                        'batch_number': batch_number,
                        'ids': batch_ids[:batch_added],  # Only IDs that were actually added
                        'count': batch_added
                    })
                    total_added += batch_added
                    logger.info(f"Batch {batch_number} completed: {batch_added} plays added")
                else:
                    logger.warning(f"Batch {batch_number} added 0 plays (likely all duplicates)")
                
            except Exception as e:
                logger.error(f"Batch {batch_number} failed: {e}")
                
                # Rollback all successful batches
                if successful_batches:
                    logger.warning(f"Rolling back {len(successful_batches)} successful batches due to failure")
                    rollback_count = self._rollback_batches(successful_batches)
                    logger.info(f"Rolled back {rollback_count} plays")
                
                raise Exception(f"Batch processing failed at batch {batch_number}. All changes have been rolled back.") from e
        
        logger.info(f"Batch processing completed successfully. Total: {total_added} plays added across {len(successful_batches)} batches")
        return total_added
    
    def _rollback_batches(self, successful_batches: List[Dict]) -> int:
        """
        Rollback successfully added batches
        
        Args:
            successful_batches: List of batch info dicts
            
        Returns:
            Number of plays rolled back
        """
        collection = self.collections["nfl_play_patterns"]
        rollback_count = 0
        
        for batch_info in successful_batches:
            try:
                # Delete the batch using ChromaDB delete
                collection.delete(ids=batch_info['ids'])
                rollback_count += batch_info['count']
                logger.debug(f"Rolled back batch {batch_info['batch_number']}: {batch_info['count']} plays")
            except Exception as e:
                logger.error(f"Failed to rollback batch {batch_info['batch_number']}: {e}")
        
        return rollback_count
    
    def get_collection_stats(self) -> Dict[str, int]:
        """Get statistics for all collections"""
        stats = {}
        
        for name, collection in self.collections.items():
            try:
                count = collection.count()
                stats[name] = count
            except Exception as e:
                logger.warning(f"Error getting stats for {name}: {e}")
                stats[name] = 0
        
        return stats
    
    def clear_collection(self, collection_name: str) -> bool:
        """Clear all data from a collection"""
        if collection_name not in self.collections:
            return False
        
        try:
            # Delete and recreate collection
            self.client.delete_collection(collection_name)
            # Recreate will happen automatically on next access
            return True
        except Exception as e:
            logger.error(f"Error clearing collection {collection_name}: {e}")
            return False


# Example usage and testing
if __name__ == "__main__":
    import sys
    sys.path.append('/Users/hudson/Documents/GitHub/IntelIP/PROJECTS/Neural/Kalshi_Agentic_Agent')
    
    # Initialize ChromaDB manager
    chromadb_manager = ChromaDBManager()
    
    # Get collection stats
    stats = chromadb_manager.get_collection_stats()
    print(f"Collection stats: {stats}")
    
    # Test search (will be empty initially)
    results = chromadb_manager.search_similar_plays(
        "3rd down and 8 in red zone, 2 minutes remaining"
    )
    print(f"Search results: {len(results['ids'][0])} found")
    
    print("ChromaDB Manager initialized successfully!")