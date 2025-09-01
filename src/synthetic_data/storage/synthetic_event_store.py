"""
Synthetic Event Store

Extends existing EventStore to handle synthetic NFL events
and integrates with ChromaDB for knowledge storage.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

# Import existing event store functionality
try:
    from src.backtesting.event_store.storage import EventStore
    from src.backtesting.event_store.models import EventQuery, StoredEvent
except ImportError:
    # Fallback if imports not available
    EventStore = None
    EventQuery = None
    StoredEvent = None

from src.sdk.core.base_adapter import StandardizedEvent
from .chromadb_manager import ChromaDBManager

logger = logging.getLogger(__name__)


class SyntheticEventStore:
    """
    Extended event store for synthetic NFL events
    Integrates with ChromaDB for enhanced knowledge storage
    """
    
    def __init__(self, 
                 sqlite_path: str = "data/synthetic_events.db",
                 chromadb_path: str = "data/chromadb"):
        """
        Initialize synthetic event store
        
        Args:
            sqlite_path: Path to SQLite database for events
            chromadb_path: Path to ChromaDB storage
        """
        # Initialize traditional event store if available
        if EventStore:
            self.event_store = EventStore(db_path=sqlite_path)
        else:
            self.event_store = None
            logger.warning("EventStore not available, using ChromaDB only")
        
        # Initialize ChromaDB for enhanced knowledge storage
        self.chromadb = ChromaDBManager(persist_directory=chromadb_path)
        
        logger.info("Initialized SyntheticEventStore")
    
    def store_synthetic_events(self, events: List[StandardizedEvent]) -> int:
        """
        Store synthetic events in both traditional and knowledge stores
        
        Args:
            events: List of synthetic events
            
        Returns:
            Number of events stored
        """
        if not events:
            return 0
        
        stored_count = 0
        
        # Store in traditional event store
        if self.event_store:
            try:
                for event in events:
                    event_id = self.event_store.save_event(event)
                    if event_id:
                        stored_count += 1
            except Exception as e:
                logger.error(f"Error storing events in EventStore: {e}")
        
        # Store in ChromaDB for knowledge-based retrieval
        try:
            # Convert events to play format for ChromaDB
            from ..preprocessing.nfl_dataset_processor import ProcessedNFLPlay
            
            plays = []
            for event in events:
                if hasattr(event.raw_data, 'game_id'):  # NFL play data
                    plays.append(event.raw_data)
            
            if plays:
                chromadb_count = self.chromadb.add_nfl_plays(plays)
                logger.info(f"Stored {chromadb_count} plays in ChromaDB")
        
        except Exception as e:
            logger.error(f"Error storing events in ChromaDB: {e}")
        
        logger.info(f"Stored {stored_count} synthetic events")
        return stored_count
    
    def search_similar_events(self, 
                            query: str,
                            n_results: int = 10,
                            filters: Optional[Dict] = None) -> List[Dict]:
        """
        Search for similar events using ChromaDB
        
        Args:
            query: Natural language query
            n_results: Number of results
            filters: Optional metadata filters
            
        Returns:
            List of similar events with metadata
        """
        results = self.chromadb.search_similar_plays(
            query=query,
            n_results=n_results,
            filters=filters
        )
        
        return self.chromadb._format_search_results(results)
    
    def get_team_patterns(self, team: str, situation: str = None) -> List[Dict]:
        """
        Get team-specific patterns from knowledge base
        
        Args:
            team: Team code
            situation: Optional situation description
            
        Returns:
            List of relevant patterns
        """
        return self.chromadb.get_team_tendencies(team, situation)
    
    def add_agent_learning(self,
                          agent_name: str, 
                          pattern_description: str,
                          success_metrics: Dict[str, float],
                          context: Dict[str, Any]) -> bool:
        """
        Add agent-discovered pattern to knowledge base
        
        Args:
            agent_name: Name of agent
            pattern_description: Description of discovered pattern  
            success_metrics: Performance metrics
            context: Additional context
            
        Returns:
            Success status
        """
        return self.chromadb.add_synthetic_learning(
            agent_name=agent_name,
            pattern_description=pattern_description,
            success_metrics=success_metrics,
            context=context
        )
    
    def query_traditional_events(self, query: EventQuery) -> List[StandardizedEvent]:
        """
        Query traditional event store if available
        
        Args:
            query: Event query
            
        Returns:
            List of events
        """
        if not self.event_store:
            logger.warning("Traditional EventStore not available")
            return []
        
        try:
            events = list(self.event_store.stream_events(query))
            return events
        except Exception as e:
            logger.error(f"Error querying traditional events: {e}")
            return []
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get statistics for all storage systems"""
        stats = {
            "chromadb_collections": self.chromadb.get_collection_stats(),
            "traditional_events": 0
        }
        
        if self.event_store:
            try:
                # This would need to be implemented in the EventStore
                # For now, just indicate it's available
                stats["traditional_store"] = "available"
            except Exception:
                stats["traditional_store"] = "unavailable"
        
        return stats


# Example usage
if __name__ == "__main__":
    # Initialize synthetic event store
    store = SyntheticEventStore()
    
    # Get storage statistics
    stats = store.get_storage_stats()
    print(f"Storage stats: {stats}")
    
    print("SyntheticEventStore initialized successfully!")