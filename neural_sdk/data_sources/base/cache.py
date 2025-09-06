"""
Response Caching for REST Data Sources

Provides caching functionality to reduce API calls and improve performance.
"""

import time
import hashlib
import json
from typing import Any, Optional, Dict
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


class ResponseCache:
    """
    LRU cache for API responses with TTL support.
    
    Features:
    - Least Recently Used (LRU) eviction
    - Time-to-live (TTL) for entries
    - Size limits
    - Statistics tracking
    """
    
    def __init__(
        self,
        ttl: int = 60,
        max_size: int = 1000,
        name: str = "ResponseCache"
    ):
        """
        Initialize response cache.
        
        Args:
            ttl: Time-to-live in seconds
            max_size: Maximum number of entries
            name: Name for logging
        """
        self.ttl = ttl
        self.max_size = max_size
        self.name = name
        
        # Use OrderedDict for LRU behavior
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'expired': 0
        }
    
    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if cache entry is expired."""
        if self.ttl <= 0:
            return False
        
        age = time.time() - entry['timestamp']
        return age > self.ttl
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        if key not in self.cache:
            self.stats['misses'] += 1
            return None
        
        entry = self.cache[key]
        
        # Check expiration
        if self._is_expired(entry):
            self.stats['expired'] += 1
            self.stats['misses'] += 1
            del self.cache[key]
            return None
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        self.stats['hits'] += 1
        
        return entry['value']
    
    def set(self, key: str, value: Any) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # Remove if already exists (to update position)
        if key in self.cache:
            del self.cache[key]
        
        # Add to end (most recently used)
        self.cache[key] = {
            'value': value,
            'timestamp': time.time()
        }
        
        # Evict oldest if over size limit
        while len(self.cache) > self.max_size:
            oldest = next(iter(self.cache))
            del self.cache[oldest]
            self.stats['evictions'] += 1
    
    def delete(self, key: str) -> bool:
        """
        Delete entry from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted, False if not found
        """
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.
        
        Returns:
            Number of entries removed
        """
        if self.ttl <= 0:
            return 0
        
        expired_keys = [
            key for key, entry in self.cache.items()
            if self._is_expired(entry)
        ]
        
        for key in expired_keys:
            del self.cache[key]
            self.stats['expired'] += 1
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = (
            self.stats['hits'] / total_requests
            if total_requests > 0 else 0
        )
        
        return {
            'name': self.name,
            'size': len(self.cache),
            'max_size': self.max_size,
            'ttl': self.ttl,
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'hit_rate': hit_rate,
            'evictions': self.stats['evictions'],
            'expired': self.stats['expired']
        }
    
    def __len__(self) -> int:
        """Get number of cached entries."""
        return len(self.cache)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        if key not in self.cache:
            return False
        
        if self._is_expired(self.cache[key]):
            del self.cache[key]
            return False
        
        return True


class MultiLevelCache:
    """
    Multi-level cache with memory and optional persistent storage.
    
    Provides a two-level cache system:
    - L1: Fast in-memory cache
    - L2: Optional persistent cache (Redis, disk, etc.)
    """
    
    def __init__(
        self,
        memory_ttl: int = 60,
        memory_size: int = 1000,
        persistent_cache: Optional[Any] = None,
        name: str = "MultiLevelCache"
    ):
        """
        Initialize multi-level cache.
        
        Args:
            memory_ttl: TTL for memory cache
            memory_size: Max size for memory cache
            persistent_cache: Optional persistent cache backend
            name: Name for logging
        """
        self.memory_cache = ResponseCache(
            ttl=memory_ttl,
            max_size=memory_size,
            name=f"{name}_L1"
        )
        self.persistent_cache = persistent_cache
        self.name = name
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache (checks both levels).
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        # Check L1 (memory)
        value = self.memory_cache.get(key)
        if value is not None:
            return value
        
        # Check L2 (persistent) if available
        if self.persistent_cache:
            try:
                value = await self.persistent_cache.get(key)
                if value is not None:
                    # Promote to L1
                    self.memory_cache.set(key, value)
                    return value
            except Exception as e:
                logger.error(f"{self.name}: Error reading from L2 cache: {e}")
        
        return None
    
    async def set(self, key: str, value: Any) -> None:
        """
        Set value in cache (both levels).
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # Set in L1 (memory)
        self.memory_cache.set(key, value)
        
        # Set in L2 (persistent) if available
        if self.persistent_cache:
            try:
                await self.persistent_cache.set(key, value)
            except Exception as e:
                logger.error(f"{self.name}: Error writing to L2 cache: {e}")
    
    async def delete(self, key: str) -> bool:
        """
        Delete from both cache levels.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted from at least one level
        """
        deleted_l1 = self.memory_cache.delete(key)
        deleted_l2 = False
        
        if self.persistent_cache:
            try:
                deleted_l2 = await self.persistent_cache.delete(key)
            except Exception as e:
                logger.error(f"{self.name}: Error deleting from L2 cache: {e}")
        
        return deleted_l1 or deleted_l2
    
    def get_stats(self) -> Dict:
        """Get statistics for both cache levels."""
        stats = {
            'name': self.name,
            'L1': self.memory_cache.get_stats()
        }
        
        if self.persistent_cache and hasattr(self.persistent_cache, 'get_stats'):
            stats['L2'] = self.persistent_cache.get_stats()
        
        return stats


def make_cache_key(*args, **kwargs) -> str:
    """
    Generate a cache key from arguments.
    
    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Cache key string
    """
    # Create a unique string from all arguments
    key_parts = []
    
    # Add positional arguments
    for arg in args:
        if isinstance(arg, (dict, list)):
            key_parts.append(json.dumps(arg, sort_keys=True))
        else:
            key_parts.append(str(arg))
    
    # Add keyword arguments
    for k, v in sorted(kwargs.items()):
        if isinstance(v, (dict, list)):
            key_parts.append(f"{k}={json.dumps(v, sort_keys=True)}")
        else:
            key_parts.append(f"{k}={v}")
    
    # Create hash of the key for consistent length
    key_str = "|".join(key_parts)
    key_hash = hashlib.md5(key_str.encode()).hexdigest()
    
    return key_hash