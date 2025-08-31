"""
Base Data Source Adapter for Neural Trading Platform SDK
Provides standardized interface for all data source integrations
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, AsyncGenerator, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import logging

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Standardized event types across all data sources"""
    ODDS_CHANGE = "odds_change"
    SENTIMENT_SHIFT = "sentiment_shift"
    GAME_EVENT = "game_event"
    WEATHER_UPDATE = "weather_update"
    VOLUME_SPIKE = "volume_spike"
    NEWS_ALERT = "news_alert"
    SOCIAL_MENTION = "social_mention"
    MARKET_MOVEMENT = "market_movement"


class SignalStrength(Enum):
    """Signal strength classification"""
    WEAK = "weak"        # <60% confidence
    MEDIUM = "medium"    # 60-80% confidence
    STRONG = "strong"    # >80% confidence


@dataclass
class DataSourceMetadata:
    """Metadata describing a data source"""
    name: str
    version: str
    author: str
    description: str
    source_type: str  # 'sportsbook', 'social', 'weather', 'news', etc.
    latency_ms: int  # Expected latency in milliseconds
    reliability: float  # 0.0 to 1.0
    requires_auth: bool
    rate_limits: Optional[Dict[str, int]] = None
    supported_sports: List[str] = field(default_factory=list)
    supported_markets: List[str] = field(default_factory=list)


@dataclass
class StandardizedEvent:
    """Standardized event format for all data sources"""
    # Core fields
    source: str  # Name of data source
    event_type: EventType
    timestamp: datetime
    game_id: Optional[str] = None
    market_id: Optional[str] = None
    
    # Event data
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Signal quality
    confidence: float = 0.0  # 0.0 to 1.0
    signal_strength: SignalStrength = SignalStrength.WEAK
    impact: str = "low"  # 'low', 'medium', 'high', 'critical'
    
    # Optional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_data: Optional[Any] = None  # Original data for debugging
    
    def __post_init__(self):
        """Calculate signal strength based on confidence"""
        if self.confidence >= 0.8:
            self.signal_strength = SignalStrength.STRONG
        elif self.confidence >= 0.6:
            self.signal_strength = SignalStrength.MEDIUM
        else:
            self.signal_strength = SignalStrength.WEAK


class DataSourceAdapter(ABC):
    """
    Base adapter class for all data sources
    Inherit from this to create new data source integrations
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize adapter with configuration
        
        Args:
            config: Configuration dictionary for the adapter
        """
        self.config = config
        self.is_connected = False
        self.metadata = self.get_metadata()
        self._rate_limiter = None
        self._error_count = 0
        self._event_count = 0
        self._last_event_time = None
        
        # Set up logging
        self.logger = logging.getLogger(f"{__name__}.{self.metadata.name}")
        
    @abstractmethod
    def get_metadata(self) -> DataSourceMetadata:
        """
        Return metadata describing this data source
        
        Returns:
            DataSourceMetadata object with source information
        """
        pass
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to data source
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """
        Close connection to data source
        """
        pass
    
    @abstractmethod
    async def validate_connection(self) -> bool:
        """
        Validate that connection is still active
        
        Returns:
            True if connection is valid, False otherwise
        """
        pass
    
    @abstractmethod
    async def stream(self) -> AsyncGenerator[StandardizedEvent, None]:
        """
        Stream standardized events from data source
        
        Yields:
            StandardizedEvent objects
        """
        pass
    
    @abstractmethod
    def transform(self, raw_data: Any) -> Optional[StandardizedEvent]:
        """
        Transform raw data from source into standardized event
        
        Args:
            raw_data: Raw data from the source
            
        Returns:
            StandardizedEvent or None if data cannot be transformed
        """
        pass
    
    async def start(self) -> None:
        """
        Start the adapter (connect and begin streaming)
        """
        try:
            self.logger.info(f"Starting {self.metadata.name} adapter...")
            
            # Connect with retry logic
            connected = await self._connect_with_retry()
            if not connected:
                raise ConnectionError(f"Failed to connect to {self.metadata.name}")
            
            self.is_connected = True
            self.logger.info(f"{self.metadata.name} adapter started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start {self.metadata.name}: {e}")
            raise
    
    async def stop(self) -> None:
        """
        Stop the adapter (disconnect and cleanup)
        """
        try:
            self.logger.info(f"Stopping {self.metadata.name} adapter...")
            await self.disconnect()
            self.is_connected = False
            self.logger.info(f"{self.metadata.name} adapter stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping {self.metadata.name}: {e}")
    
    async def _connect_with_retry(self, max_retries: int = 3) -> bool:
        """
        Connect with exponential backoff retry
        
        Args:
            max_retries: Maximum number of connection attempts
            
        Returns:
            True if connected, False otherwise
        """
        for attempt in range(max_retries):
            try:
                if await self.connect():
                    return True
                    
            except Exception as e:
                self.logger.warning(
                    f"Connection attempt {attempt + 1} failed: {e}"
                )
                
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                self.logger.info(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
        
        return False
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the adapter
        
        Returns:
            Dictionary with health status information
        """
        is_healthy = await self.validate_connection()
        
        return {
            "adapter": self.metadata.name,
            "healthy": is_healthy,
            "connected": self.is_connected,
            "error_count": self._error_count,
            "event_count": self._event_count,
            "last_event": self._last_event_time.isoformat() if self._last_event_time else None,
            "uptime_seconds": self._calculate_uptime(),
            "events_per_minute": self._calculate_event_rate()
        }
    
    def _calculate_uptime(self) -> float:
        """Calculate adapter uptime in seconds"""
        # Implementation would track start time
        return 0.0
    
    def _calculate_event_rate(self) -> float:
        """Calculate events per minute"""
        # Implementation would track event timestamps
        return 0.0
    
    def _increment_event_count(self) -> None:
        """Track event statistics"""
        self._event_count += 1
        self._last_event_time = datetime.now()
    
    def _increment_error_count(self) -> None:
        """Track error statistics"""
        self._error_count += 1
    
    @property
    def statistics(self) -> Dict[str, Any]:
        """
        Get adapter statistics
        
        Returns:
            Dictionary with adapter statistics
        """
        return {
            "events_processed": self._event_count,
            "errors_encountered": self._error_count,
            "error_rate": self._error_count / max(self._event_count, 1),
            "last_event": self._last_event_time,
            "is_connected": self.is_connected
        }


class RateLimiter:
    """Rate limiting for API calls"""
    
    def __init__(self, calls_per_second: float):
        """
        Initialize rate limiter
        
        Args:
            calls_per_second: Maximum calls per second
        """
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0
    
    async def acquire(self) -> None:
        """Wait if necessary to respect rate limit"""
        now = asyncio.get_event_loop().time()
        time_since_last = now - self.last_call
        
        if time_since_last < self.min_interval:
            await asyncio.sleep(self.min_interval - time_since_last)
        
        self.last_call = asyncio.get_event_loop().time()


class AuthenticationError(Exception):
    """Raised when authentication fails"""
    pass


class DataSourceError(Exception):
    """Base exception for data source errors"""
    pass


class RateLimitError(DataSourceError):
    """Raised when rate limit is exceeded"""
    pass


class DataValidationError(DataSourceError):
    """Raised when data validation fails"""
    pass