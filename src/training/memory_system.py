"""
ChromaDB Memory Integration System

Integrates ChromaDB with agent memory systems for persistent learning
and experience replay from synthetic training scenarios.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib

from ..synthetic_data.storage.chromadb_manager import ChromaDBManager
from .synthetic_env import AgentPerformanceMetrics, AgentAction
from src.sdk.core.base_adapter import StandardizedEvent

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of agent memories"""
    EXPERIENCE = "experience"          # Trading experiences
    PATTERN = "pattern"               # Recognized patterns
    STRATEGY = "strategy"             # Successful strategies
    MISTAKE = "mistake"              # Failed decisions for avoidance
    CONTEXT = "context"              # Situational context
    PERFORMANCE = "performance"       # Performance records


@dataclass
class AgentMemory:
    """Individual agent memory entry"""
    memory_id: str
    agent_id: str
    memory_type: MemoryType
    timestamp: datetime
    
    # Core content
    description: str
    context: Dict[str, Any]
    outcome: Dict[str, Any]
    
    # Learning metadata
    importance_score: float = 0.5  # 0-1 importance for retention
    confidence: float = 0.5        # 0-1 confidence in the memory
    usage_count: int = 0          # How often this memory was accessed
    success_rate: float = 0.5     # Success rate when this memory was used
    
    # Retrieval metadata  
    tags: List[str] = field(default_factory=list)
    related_scenarios: List[str] = field(default_factory=list)
    embedding_metadata: Dict[str, Any] = field(default_factory=dict)
    
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: Optional[datetime] = None
    last_updated: Optional[datetime] = None


@dataclass
class MemoryQuery:
    """Query for retrieving memories"""
    query_text: str
    agent_id: Optional[str] = None
    memory_types: List[MemoryType] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    min_importance: float = 0.0
    max_results: int = 10
    include_context: bool = True


@dataclass 
class MemorySearchResult:
    """Result from memory search"""
    memory: AgentMemory
    relevance_score: float
    similarity_score: float
    context_match_score: float


class AgentMemorySystem:
    """
    ChromaDB-backed memory system for agent learning and experience replay
    """
    
    def __init__(self, chromadb_manager: ChromaDBManager = None):
        """
        Initialize agent memory system
        
        Args:
            chromadb_manager: ChromaDB manager instance
        """
        self.chromadb = chromadb_manager or ChromaDBManager()
        
        # Initialize memory collections in ChromaDB
        self._initialize_memory_collections()
        
        # Memory management
        self.memory_cache: Dict[str, AgentMemory] = {}
        self.access_patterns: Dict[str, List[datetime]] = {}
        
        # Performance tracking
        self.memory_stats = {
            "total_memories": 0,
            "memories_by_type": {},
            "memories_by_agent": {},
            "avg_retrieval_time_ms": 0.0,
            "cache_hit_rate": 0.0
        }
        
        logger.info("Initialized AgentMemorySystem with ChromaDB integration")
    
    def _initialize_memory_collections(self):
        """Initialize ChromaDB collections for agent memories"""
        
        # Agent experiences collection
        try:
            self.experiences_collection = self.chromadb.client.get_or_create_collection(
                name="agent_experiences",
                metadata={
                    "description": "Agent trading experiences and outcomes",
                    "purpose": "experience_replay",
                    "source": "agent_training"
                }
            )
        except Exception as e:
            logger.error(f"Failed to create experiences collection: {e}")
            self.experiences_collection = None
        
        # Pattern recognition collection
        try:
            self.patterns_collection = self.chromadb.client.get_or_create_collection(
                name="agent_patterns", 
                metadata={
                    "description": "Recognized market and game patterns",
                    "purpose": "pattern_matching",
                    "source": "agent_learning"
                }
            )
        except Exception as e:
            logger.error(f"Failed to create patterns collection: {e}")
            self.patterns_collection = None
        
        # Strategy knowledge collection
        try:
            self.strategies_collection = self.chromadb.client.get_or_create_collection(
                name="agent_strategies",
                metadata={
                    "description": "Successful trading strategies and tactics",
                    "purpose": "strategy_recall",
                    "source": "agent_optimization"
                }
            )
        except Exception as e:
            logger.error(f"Failed to create strategies collection: {e}")
            self.strategies_collection = None
    
    async def store_agent_experience(self, 
                                   agent_id: str,
                                   scenario_id: str, 
                                   action: AgentAction,
                                   outcome: Dict[str, Any],
                                   context: Dict[str, Any],
                                   importance: float = 0.5) -> str:
        """
        Store agent experience in memory system
        
        Args:
            agent_id: Agent identifier
            scenario_id: Training scenario ID
            action: Action taken by agent
            outcome: Result of the action
            context: Context when action was taken
            importance: Importance score (0-1)
            
        Returns:
            Memory ID
        """
        # Create memory entry
        memory_id = self._generate_memory_id(agent_id, scenario_id, action.timestamp)
        
        # Determine memory type based on outcome
        memory_type = self._classify_experience_type(action, outcome)
        
        # Create description
        description = self._create_experience_description(action, outcome, context)
        
        # Extract tags
        tags = self._extract_experience_tags(action, outcome, context)
        
        memory = AgentMemory(
            memory_id=memory_id,
            agent_id=agent_id,
            memory_type=memory_type,
            timestamp=action.timestamp,
            description=description,
            context=context,
            outcome=outcome,
            importance_score=importance,
            confidence=action.confidence or 0.5,
            tags=tags,
            related_scenarios=[scenario_id]
        )
        
        # Store in ChromaDB
        await self._store_memory_in_chromadb(memory)
        
        # Update cache
        self.memory_cache[memory_id] = memory
        
        # Update statistics
        self._update_memory_stats(memory)
        
        logger.debug(f"Stored experience memory: {memory_id}")
        return memory_id
    
    async def store_discovered_pattern(self,
                                     agent_id: str,
                                     pattern_description: str,
                                     pattern_context: Dict[str, Any],
                                     success_rate: float,
                                     confidence: float = 0.7) -> str:
        """
        Store discovered pattern in memory system
        
        Args:
            agent_id: Agent that discovered pattern
            pattern_description: Description of the pattern
            pattern_context: Context where pattern applies
            success_rate: Historical success rate of pattern
            confidence: Confidence in pattern validity
            
        Returns:
            Memory ID
        """
        memory_id = self._generate_memory_id(agent_id, "pattern", datetime.now())
        
        memory = AgentMemory(
            memory_id=memory_id,
            agent_id=agent_id,
            memory_type=MemoryType.PATTERN,
            timestamp=datetime.now(),
            description=pattern_description,
            context=pattern_context,
            outcome={"success_rate": success_rate},
            importance_score=min(1.0, success_rate + 0.2),  # Higher success = higher importance
            confidence=confidence,
            success_rate=success_rate,
            tags=self._extract_pattern_tags(pattern_description, pattern_context)
        )
        
        # Store in ChromaDB patterns collection
        await self._store_pattern_in_chromadb(memory)
        
        # Update cache and stats
        self.memory_cache[memory_id] = memory
        self._update_memory_stats(memory)
        
        logger.info(f"Stored pattern memory: {pattern_description[:50]}...")
        return memory_id
    
    async def store_strategy_knowledge(self,
                                     agent_id: str,
                                     strategy_name: str,
                                     strategy_details: Dict[str, Any],
                                     performance_metrics: Dict[str, float],
                                     applicable_contexts: List[str]) -> str:
        """
        Store successful strategy in memory system
        
        Args:
            agent_id: Agent that developed strategy
            strategy_name: Name/description of strategy
            strategy_details: Detailed strategy parameters
            performance_metrics: Performance metrics
            applicable_contexts: Contexts where strategy works
            
        Returns:
            Memory ID
        """
        memory_id = self._generate_memory_id(agent_id, "strategy", datetime.now())
        
        # Calculate importance based on performance
        profit_factor = performance_metrics.get("profit_factor", 1.0)
        win_rate = performance_metrics.get("win_rate", 0.5)
        importance = min(1.0, (profit_factor * win_rate) / 2.0)
        
        memory = AgentMemory(
            memory_id=memory_id,
            agent_id=agent_id,
            memory_type=MemoryType.STRATEGY,
            timestamp=datetime.now(),
            description=f"Strategy: {strategy_name}",
            context={
                "strategy_details": strategy_details,
                "applicable_contexts": applicable_contexts,
                "performance_metrics": performance_metrics
            },
            outcome=performance_metrics,
            importance_score=importance,
            confidence=min(1.0, win_rate + 0.3),
            success_rate=win_rate,
            tags=["strategy"] + applicable_contexts
        )
        
        # Store in ChromaDB strategies collection
        await self._store_strategy_in_chromadb(memory)
        
        # Update cache and stats
        self.memory_cache[memory_id] = memory
        self._update_memory_stats(memory)
        
        logger.info(f"Stored strategy: {strategy_name}")
        return memory_id
    
    async def retrieve_memories(self, query: MemoryQuery) -> List[MemorySearchResult]:
        """
        Retrieve relevant memories based on query
        
        Args:
            query: Memory query parameters
            
        Returns:
            List of relevant memories with scores
        """
        results = []
        
        # Search in ChromaDB collections
        if query.memory_types:
            for memory_type in query.memory_types:
                type_results = await self._search_by_memory_type(query, memory_type)
                results.extend(type_results)
        else:
            # Search all types
            for memory_type in MemoryType:
                type_results = await self._search_by_memory_type(query, memory_type)
                results.extend(type_results)
        
        # Filter by importance
        if query.min_importance > 0:
            results = [r for r in results if r.memory.importance_score >= query.min_importance]
        
        # Sort by relevance score
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Limit results
        results = results[:query.max_results]
        
        # Update access patterns
        for result in results:
            await self._record_memory_access(result.memory)
        
        logger.debug(f"Retrieved {len(results)} memories for query: {query.query_text[:50]}...")
        return results
    
    async def _search_by_memory_type(self, query: MemoryQuery, memory_type: MemoryType) -> List[MemorySearchResult]:
        """Search specific memory type collection"""
        
        results = []
        collection = None
        
        # Select appropriate collection
        if memory_type == MemoryType.EXPERIENCE:
            collection = self.experiences_collection
        elif memory_type == MemoryType.PATTERN:
            collection = self.patterns_collection
        elif memory_type == MemoryType.STRATEGY:
            collection = self.strategies_collection
        else:
            # Use general experiences collection for other types
            collection = self.experiences_collection
        
        if not collection:
            return results
        
        try:
            # Search ChromaDB
            search_results = collection.query(
                query_texts=[query.query_text],
                n_results=min(query.max_results * 2, 50),  # Get more to filter
                include=['documents', 'metadatas', 'distances']
            )
            
            # Convert to MemorySearchResult objects
            if search_results['ids'] and search_results['ids'][0]:
                for i, memory_id in enumerate(search_results['ids'][0]):
                    try:
                        # Check if memory is in cache
                        if memory_id in self.memory_cache:
                            memory = self.memory_cache[memory_id]
                        else:
                            # Reconstruct memory from ChromaDB metadata
                            memory = self._reconstruct_memory_from_metadata(
                                memory_id, 
                                search_results['metadatas'][0][i]
                            )
                            self.memory_cache[memory_id] = memory
                        
                        # Filter by agent if specified
                        if query.agent_id and memory.agent_id != query.agent_id:
                            continue
                        
                        # Filter by tags if specified
                        if query.tags and not any(tag in memory.tags for tag in query.tags):
                            continue
                        
                        # Calculate scores
                        similarity_score = 1.0 - search_results['distances'][0][i]
                        relevance_score = self._calculate_relevance_score(memory, query)
                        context_match_score = self._calculate_context_match_score(memory, query)
                        
                        result = MemorySearchResult(
                            memory=memory,
                            relevance_score=relevance_score,
                            similarity_score=similarity_score,
                            context_match_score=context_match_score
                        )
                        
                        results.append(result)
                        
                    except Exception as e:
                        logger.warning(f"Error processing search result {i}: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Error searching {memory_type.value} collection: {e}")
        
        return results
    
    async def get_agent_learning_summary(self, agent_id: str) -> Dict[str, Any]:
        """
        Get learning summary for specific agent
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Learning summary with key insights
        """
        # Query all memories for agent
        query = MemoryQuery(
            query_text="",  # Empty query to get all
            agent_id=agent_id,
            max_results=1000
        )
        
        memories = await self.retrieve_memories(query)
        
        if not memories:
            return {"agent_id": agent_id, "total_memories": 0}
        
        # Analyze memories
        summary = {
            "agent_id": agent_id,
            "total_memories": len(memories),
            "memory_types": {},
            "success_patterns": [],
            "improvement_areas": [],
            "key_strategies": [],
            "learning_progress": {}
        }
        
        # Count by memory type
        for result in memories:
            mem_type = result.memory.memory_type.value
            summary["memory_types"][mem_type] = summary["memory_types"].get(mem_type, 0) + 1
        
        # Extract successful patterns (high success rate + high importance)
        successful_memories = [
            r.memory for r in memories 
            if r.memory.success_rate >= 0.7 and r.memory.importance_score >= 0.6
        ]
        
        summary["success_patterns"] = [
            {
                "description": mem.description,
                "success_rate": mem.success_rate,
                "importance": mem.importance_score,
                "usage_count": mem.usage_count
            }
            for mem in successful_memories[:5]  # Top 5
        ]
        
        # Identify improvement areas (failures with high importance)
        failure_memories = [
            r.memory for r in memories
            if r.memory.memory_type == MemoryType.MISTAKE and r.memory.importance_score >= 0.5
        ]
        
        summary["improvement_areas"] = [
            {
                "description": mem.description,
                "frequency": mem.usage_count,
                "impact": mem.importance_score
            }
            for mem in failure_memories[:3]  # Top 3 areas
        ]
        
        # Extract key strategies
        strategy_memories = [
            r.memory for r in memories
            if r.memory.memory_type == MemoryType.STRATEGY
        ]
        
        summary["key_strategies"] = [
            {
                "name": mem.description,
                "success_rate": mem.success_rate,
                "confidence": mem.confidence
            }
            for mem in sorted(strategy_memories, key=lambda x: x.success_rate, reverse=True)[:3]
        ]
        
        return summary
    
    def _generate_memory_id(self, agent_id: str, identifier: str, timestamp: datetime) -> str:
        """Generate unique memory ID"""
        content = f"{agent_id}_{identifier}_{timestamp.isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _classify_experience_type(self, action: AgentAction, outcome: Dict[str, Any]) -> MemoryType:
        """Classify experience type based on action and outcome"""
        
        if outcome.get("success", True):
            if outcome.get("profit", 0) > 0:
                return MemoryType.EXPERIENCE
            else:
                return MemoryType.CONTEXT
        else:
            return MemoryType.MISTAKE
    
    def _create_experience_description(self, action: AgentAction, outcome: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Create human-readable description of experience"""
        
        action_desc = f"{action.action_type.upper()}"
        if action.market_ticker:
            action_desc += f" on {action.market_ticker}"
        if action.side and action.size:
            action_desc += f" - {action.side} {action.size}"
        
        outcome_desc = "SUCCESS" if outcome.get("success", True) else "FAILED"
        if "profit" in outcome:
            outcome_desc += f" (P&L: {outcome['profit']:+.2f})"
        
        return f"{action_desc} - {outcome_desc}"
    
    def _extract_experience_tags(self, action: AgentAction, outcome: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Extract relevant tags from experience"""
        tags = []
        
        # Action type tags
        tags.append(action.action_type)
        
        # Outcome tags
        if outcome.get("success", True):
            tags.append("success")
            if outcome.get("profit", 0) > 0:
                tags.append("profitable")
        else:
            tags.append("failure")
        
        # Context tags
        if context.get("market_condition"):
            tags.append(f"market_{context['market_condition']}")
        
        if context.get("game_situation"):
            tags.append(f"game_{context['game_situation']}")
        
        # Confidence tags
        if action.confidence:
            if action.confidence >= 0.8:
                tags.append("high_confidence")
            elif action.confidence <= 0.3:
                tags.append("low_confidence")
        
        return tags
    
    def _extract_pattern_tags(self, description: str, context: Dict[str, Any]) -> List[str]:
        """Extract tags from pattern description and context"""
        tags = ["pattern"]
        
        # Extract key terms from description
        keywords = ["momentum", "reversal", "breakout", "support", "resistance", "volume", "sentiment"]
        for keyword in keywords:
            if keyword in description.lower():
                tags.append(keyword)
        
        # Context-based tags
        if context.get("market_type"):
            tags.append(f"market_{context['market_type']}")
        
        if context.get("game_phase"):
            tags.append(f"phase_{context['game_phase']}")
        
        return tags
    
    async def _store_memory_in_chromadb(self, memory: AgentMemory):
        """Store memory in appropriate ChromaDB collection"""
        
        if memory.memory_type == MemoryType.EXPERIENCE and self.experiences_collection:
            collection = self.experiences_collection
        elif memory.memory_type == MemoryType.PATTERN and self.patterns_collection:
            collection = self.patterns_collection
        elif memory.memory_type == MemoryType.STRATEGY and self.strategies_collection:
            collection = self.strategies_collection
        else:
            # Default to experiences collection
            collection = self.experiences_collection
        
        if not collection:
            logger.warning(f"No collection available for memory type: {memory.memory_type}")
            return
        
        try:
            collection.add(
                ids=[memory.memory_id],
                documents=[memory.description],
                metadatas=[{
                    "agent_id": memory.agent_id,
                    "memory_type": memory.memory_type.value,
                    "timestamp": memory.timestamp.isoformat(),
                    "importance_score": memory.importance_score,
                    "confidence": memory.confidence,
                    "success_rate": memory.success_rate,
                    "tags": json.dumps(memory.tags),
                    "context": json.dumps(memory.context, default=str),
                    "outcome": json.dumps(memory.outcome, default=str)
                }]
            )
        except Exception as e:
            logger.error(f"Error storing memory in ChromaDB: {e}")
    
    async def _store_pattern_in_chromadb(self, memory: AgentMemory):
        """Store pattern memory in ChromaDB patterns collection"""
        await self._store_memory_in_chromadb(memory)
    
    async def _store_strategy_in_chromadb(self, memory: AgentMemory):
        """Store strategy memory in ChromaDB strategies collection"""
        await self._store_memory_in_chromadb(memory)
    
    def _reconstruct_memory_from_metadata(self, memory_id: str, metadata: Dict[str, Any]) -> AgentMemory:
        """Reconstruct AgentMemory object from ChromaDB metadata"""
        
        return AgentMemory(
            memory_id=memory_id,
            agent_id=metadata.get("agent_id", "unknown"),
            memory_type=MemoryType(metadata.get("memory_type", "experience")),
            timestamp=datetime.fromisoformat(metadata.get("timestamp", datetime.now().isoformat())),
            description=metadata.get("description", ""),
            context=json.loads(metadata.get("context", "{}")),
            outcome=json.loads(metadata.get("outcome", "{}")),
            importance_score=metadata.get("importance_score", 0.5),
            confidence=metadata.get("confidence", 0.5),
            success_rate=metadata.get("success_rate", 0.5),
            tags=json.loads(metadata.get("tags", "[]"))
        )
    
    def _calculate_relevance_score(self, memory: AgentMemory, query: MemoryQuery) -> float:
        """Calculate relevance score for memory given query"""
        
        score = 0.0
        
        # Base score from importance and confidence
        score += memory.importance_score * 0.3
        score += memory.confidence * 0.2
        
        # Success rate bonus
        score += memory.success_rate * 0.2
        
        # Tag matching bonus
        if query.tags:
            matching_tags = len(set(query.tags) & set(memory.tags))
            score += (matching_tags / len(query.tags)) * 0.3
        
        return min(1.0, score)
    
    def _calculate_context_match_score(self, memory: AgentMemory, query: MemoryQuery) -> float:
        """Calculate context matching score"""
        
        # Simple context matching based on query text
        context_str = json.dumps(memory.context, default=str).lower()
        query_terms = query.query_text.lower().split()
        
        matches = sum(1 for term in query_terms if term in context_str)
        
        if query_terms:
            return matches / len(query_terms)
        else:
            return 0.5  # Neutral score for empty query
    
    async def _record_memory_access(self, memory: AgentMemory):
        """Record that memory was accessed"""
        
        memory.usage_count += 1
        memory.last_accessed = datetime.now()
        
        # Update access patterns
        if memory.memory_id not in self.access_patterns:
            self.access_patterns[memory.memory_id] = []
        
        self.access_patterns[memory.memory_id].append(datetime.now())
        
        # Keep only recent access patterns (last 100)
        if len(self.access_patterns[memory.memory_id]) > 100:
            self.access_patterns[memory.memory_id] = self.access_patterns[memory.memory_id][-100:]
    
    def _update_memory_stats(self, memory: AgentMemory):
        """Update memory statistics"""
        
        self.memory_stats["total_memories"] += 1
        
        mem_type = memory.memory_type.value
        self.memory_stats["memories_by_type"][mem_type] = self.memory_stats["memories_by_type"].get(mem_type, 0) + 1
        
        self.memory_stats["memories_by_agent"][memory.agent_id] = self.memory_stats["memories_by_agent"].get(memory.agent_id, 0) + 1
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics"""
        return {
            **self.memory_stats,
            "cache_size": len(self.memory_cache),
            "access_patterns_tracked": len(self.access_patterns)
        }


# Example usage and testing  
if __name__ == "__main__":
    import sys
    import asyncio
    sys.path.append('/Users/hudson/Documents/GitHub/IntelIP/PROJECTS/Neural/Kalshi_Agentic_Agent')
    
    from src.synthetic_data.storage.chromadb_manager import ChromaDBManager
    
    async def test_memory_system():
        # Initialize memory system
        chromadb = ChromaDBManager()
        memory_system = AgentMemorySystem(chromadb)
        
        # Test storing experience
        test_action = AgentAction(
            agent_id="test_agent",
            timestamp=datetime.now(),
            action_type="trade",
            market_ticker="NFL-KC-BUF-20240115",
            side="yes",
            size=100.0,
            confidence=0.8
        )
        
        test_outcome = {
            "success": True,
            "profit": 25.0,
            "execution_time_ms": 150
        }
        
        test_context = {
            "market_condition": "volatile",
            "game_situation": "fourth_quarter",
            "score_differential": 3
        }
        
        # Store experience
        memory_id = await memory_system.store_agent_experience(
            agent_id="test_agent",
            scenario_id="test_scenario_001", 
            action=test_action,
            outcome=test_outcome,
            context=test_context,
            importance=0.8
        )
        
        print(f"Stored experience memory: {memory_id}")
        
        # Store pattern discovery
        pattern_id = await memory_system.store_discovered_pattern(
            agent_id="test_agent",
            pattern_description="Fourth quarter momentum reversal pattern",
            pattern_context={"game_phase": "fourth_quarter", "score_situation": "close"},
            success_rate=0.75,
            confidence=0.8
        )
        
        print(f"Stored pattern memory: {pattern_id}")
        
        # Test memory retrieval
        query = MemoryQuery(
            query_text="fourth quarter trading volatile market",
            agent_id="test_agent",
            max_results=10
        )
        
        results = await memory_system.retrieve_memories(query)
        print(f"Retrieved {len(results)} memories")
        
        for result in results:
            print(f"  Memory: {result.memory.description}")
            print(f"  Relevance: {result.relevance_score:.3f}")
            print(f"  Similarity: {result.similarity_score:.3f}")
        
        # Get learning summary
        summary = await memory_system.get_agent_learning_summary("test_agent")
        print(f"\nLearning Summary:")
        print(f"  Total memories: {summary['total_memories']}")
        print(f"  Memory types: {summary['memory_types']}")
        
        # Get statistics
        stats = memory_system.get_memory_statistics()
        print(f"\nMemory System Stats: {stats}")
    
    # Run test
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_memory_system())