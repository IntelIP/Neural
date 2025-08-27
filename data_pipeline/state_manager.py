"""
State Manager - Manages agent state with hot/warm/cold storage tiers
Provides consistent state management across all agents using Agentuity KV storage
"""

import json
import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import OrderedDict
import hashlib

logger = logging.getLogger(__name__)


# KV Storage Namespaces
KV_NAMESPACES = {
    "agent_state": "Persistent agent configuration and state",
    "market_data": "Current market snapshots and prices",
    "positions": "Active trading positions",
    "signals": "Generated trading signals",
    "alerts": "Risk and system alerts", 
    "metrics": "Performance metrics and statistics",
    "cache": "Temporary computation cache",
    "windows": "Windowed aggregation data"
}


@dataclass
class StateEntry:
    """State entry with metadata"""
    key: str
    value: Any
    timestamp: datetime
    ttl: Optional[int] = None  # Time to live in seconds
    version: int = 1
    checksum: Optional[str] = None


class HotCache:
    """
    In-memory LRU cache for frequently accessed data
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize hot cache
        
        Args:
            max_size: Maximum number of entries
        """
        self.cache: OrderedDict[str, StateEntry] = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            entry = self.cache[key]
            
            # Check TTL
            if entry.ttl:
                elapsed = (datetime.now() - entry.timestamp).seconds
                if elapsed > entry.ttl:
                    del self.cache[key]
                    self.misses += 1
                    return None
            
            self.hits += 1
            return entry.value
        
        self.misses += 1
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache"""
        # Remove oldest if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            self.cache.popitem(last=False)
        
        # Create entry
        entry = StateEntry(
            key=key,
            value=value,
            timestamp=datetime.now(),
            ttl=ttl
        )
        
        # Add/update cache
        self.cache[key] = entry
        self.cache.move_to_end(key)
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        hit_rate = self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate
        }


class AgentStateManager:
    """
    Manages agent state with three-tier storage:
    - Hot: In-memory LRU cache (last 100-1000 entries)
    - Warm: KV storage (last hour)
    - Cold: External database (historical data)
    """
    
    def __init__(self, hot_cache_size: int = 100):
        """
        Initialize state manager
        
        Args:
            hot_cache_size: Size of in-memory cache per namespace
        """
        # Hot caches per namespace
        self.hot_caches: Dict[str, HotCache] = {
            namespace: HotCache(hot_cache_size)
            for namespace in KV_NAMESPACES
        }
        
        # Default states for agents
        self.default_states = {
            "DataCoordinator": {
                "tracked_markets": {},
                "events_received": 0,
                "events_routed": 0,
                "is_running": False
            },
            "MarketEngineer": {
                "market_sentiments": {},
                "is_running": False
            },
            "RiskManager": {
                "positions": {},
                "risk_limits": {
                    "max_position_size": 100,
                    "max_portfolio_risk": 0.4,
                    "max_correlation": 0.7
                },
                "daily_pnl": 0.0
            },
            "StrategyAnalyst": {
                "active_signals": [],
                "signal_history": [],
                "is_running": False
            },
            "TradeExecutor": {
                "open_orders": {},
                "portfolio_value": 1000.0,
                "is_running": False
            }
        }
        
        logger.info("AgentStateManager initialized")
    
    async def save_state(
        self, 
        context,  # Agentuity context
        namespace: str,
        key: str,
        value: Any,
        ttl: Optional[int] = 3600
    ) -> bool:
        """
        Save state to appropriate tier
        
        Args:
            context: Agentuity context with KV access
            namespace: Storage namespace
            key: State key
            value: State value
            ttl: Time to live in seconds (default 1 hour)
            
        Returns:
            True if saved successfully
        """
        try:
            # Validate namespace
            if namespace not in KV_NAMESPACES:
                logger.error(f"Invalid namespace: {namespace}")
                return False
            
            # Save to hot cache
            self.hot_caches[namespace].set(key, value, ttl)
            
            # Save to KV storage (warm tier)
            await context.kv.set(namespace, key, value)
            
            logger.debug(f"Saved state: {namespace}/{key}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving state: {e}")
            return False
    
    async def restore_state(
        self,
        context,  # Agentuity context
        namespace: str,
        key: str,
        default: Any = None
    ) -> Any:
        """
        Restore state from storage tiers
        
        Args:
            context: Agentuity context with KV access
            namespace: Storage namespace
            key: State key
            default: Default value if not found
            
        Returns:
            State value or default
        """
        try:
            # Check hot cache first
            value = self.hot_caches[namespace].get(key)
            if value is not None:
                return value
            
            # Check KV storage (warm tier)
            result = await context.kv.get(namespace, key)
            if result.exists:
                value = await result.data.json()
                
                # Populate hot cache
                self.hot_caches[namespace].set(key, value)
                
                return value
            
            # Return default
            return default
            
        except Exception as e:
            logger.error(f"Error restoring state: {e}")
            return default
    
    async def save_agent_state(self, context, agent_name: str, state: Dict[str, Any]) -> bool:
        """
        Save complete agent state
        
        Args:
            context: Agentuity context
            agent_name: Name of the agent
            state: Agent state dictionary
            
        Returns:
            True if saved successfully
        """
        return await self.save_state(
            context,
            "agent_state",
            agent_name,
            state,
            ttl=3600  # 1 hour TTL
        )
    
    async def restore_agent_state(self, context, agent_name: str) -> Dict[str, Any]:
        """
        Restore agent state
        
        Args:
            context: Agentuity context
            agent_name: Name of the agent
            
        Returns:
            Agent state or default state
        """
        state = await self.restore_state(
            context,
            "agent_state",
            agent_name,
            default=self.default_states.get(agent_name, {})
        )
        
        # Merge with defaults to ensure all keys exist
        default = self.default_states.get(agent_name, {})
        return {**default, **state}
    
    def get_default_state(self, agent_name: str) -> Dict[str, Any]:
        """Get default state for agent"""
        return self.default_states.get(agent_name, {})
    
    def clear_namespace(self, namespace: str):
        """Clear hot cache for namespace"""
        if namespace in self.hot_caches:
            self.hot_caches[namespace].clear()
            logger.info(f"Cleared hot cache for {namespace}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for all namespaces"""
        return {
            namespace: cache.get_stats()
            for namespace, cache in self.hot_caches.items()
        }


class ComputationCache:
    """
    Cache for expensive computations with automatic expiration
    """
    
    def __init__(self, state_manager: AgentStateManager):
        """
        Initialize computation cache
        
        Args:
            state_manager: AgentStateManager instance
        """
        self.state_manager = state_manager
        self.computation_times: Dict[str, float] = {}
        
    async def get_or_compute(
        self,
        context,
        key: str,
        compute_fn: Callable,
        ttl: int = 300,
        force_refresh: bool = False
    ) -> Any:
        """
        Get cached value or compute it
        
        Args:
            context: Agentuity context
            key: Cache key
            compute_fn: Async function to compute value
            ttl: Time to live in seconds (default 5 minutes)
            force_refresh: Force recomputation
            
        Returns:
            Cached or computed value
        """
        # Generate cache key with hash for uniqueness
        cache_key = f"compute_{hashlib.md5(key.encode()).hexdigest()[:8]}_{key}"
        
        # Check cache unless forced refresh
        if not force_refresh:
            cached = await self.state_manager.restore_state(
                context,
                "cache",
                cache_key
            )
            
            if cached is not None:
                logger.debug(f"Cache hit for {key}")
                return cached
        
        # Compute value
        logger.debug(f"Computing value for {key}")
        start_time = datetime.now()
        
        try:
            value = await compute_fn()
            
            # Track computation time
            elapsed = (datetime.now() - start_time).total_seconds()
            self.computation_times[key] = elapsed
            
            # Cache result
            await self.state_manager.save_state(
                context,
                "cache",
                cache_key,
                value,
                ttl=ttl
            )
            
            logger.info(f"Computed and cached {key} in {elapsed:.2f}s")
            return value
            
        except Exception as e:
            logger.error(f"Computation failed for {key}: {e}")
            raise
    
    async def invalidate(self, context, key: str):
        """Invalidate cached value"""
        cache_key = f"compute_{hashlib.md5(key.encode()).hexdigest()[:8]}_{key}"
        
        # Clear from hot cache
        self.state_manager.hot_caches["cache"].cache.pop(cache_key, None)
        
        # Would need to implement delete in KV storage
        logger.info(f"Invalidated cache for {key}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_stats": self.state_manager.hot_caches["cache"].get_stats(),
            "computation_times": self.computation_times
        }


# Global instance for easy access
_state_manager: Optional[AgentStateManager] = None


def get_state_manager(hot_cache_size: int = 100) -> AgentStateManager:
    """Get or create global state manager instance"""
    global _state_manager
    if _state_manager is None:
        _state_manager = AgentStateManager(hot_cache_size)
    return _state_manager