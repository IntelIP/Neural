"""
Base abstractions for the Neural SDK Data Collection Infrastructure.

This module provides abstract base classes that define the common interface
for all data sources in the Neural SDK. These abstractions ensure consistency
across different data source implementations (WebSocket, REST, etc.) and 
provide a foundation for building custom data sources.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
import asyncio
import logging

from neural.data_collection.exceptions import (
    ConnectionError,
    ConfigurationError,
    DataSourceError
)


# Configure module logger
logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """
    Enumeration of possible connection states for a data source.
    
    States:
        DISCONNECTED: Not connected to the data source
        CONNECTING: Connection attempt in progress
        CONNECTED: Successfully connected and operational
        RECONNECTING: Attempting to reconnect after disconnection
        ERROR: Connection failed or encountered an error
        CLOSING: Connection is being closed
    """
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    CLOSING = "closing"


@dataclass
class DataSourceConfig:
    """
    Base configuration class for all data sources.
    
    This dataclass defines common configuration parameters that all
    data sources should support. Specific implementations can extend
    this class to add protocol-specific configurations.
    
    Attributes:
        name: Unique identifier for this data source instance
        enabled: Whether this data source is active
        max_retries: Maximum number of connection retry attempts
        retry_delay: Initial delay between retries in seconds
        retry_backoff: Multiplier for exponential backoff
        timeout: Connection timeout in seconds
        buffer_size: Maximum number of messages to buffer
        metadata: Additional configuration parameters
    
    Example:
        >>> config = DataSourceConfig(
        ...     name="espn_api",
        ...     enabled=True,
        ...     max_retries=5,
        ...     timeout=30.0
        ... )
    """
    name: str
    enabled: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0
    timeout: float = 30.0
    buffer_size: int = 10000
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ConfigurationError: If configuration is invalid
        """
        if not self.name:
            raise ConfigurationError("Data source name cannot be empty")
        
        if self.max_retries < 0:
            raise ConfigurationError(
                "max_retries must be non-negative",
                details={"value": self.max_retries}
            )
        
        if self.retry_delay <= 0:
            raise ConfigurationError(
                "retry_delay must be positive",
                details={"value": self.retry_delay}
            )
        
        if self.retry_backoff < 1:
            raise ConfigurationError(
                "retry_backoff must be >= 1",
                details={"value": self.retry_backoff}
            )
        
        if self.timeout <= 0:
            raise ConfigurationError(
                "timeout must be positive",
                details={"value": self.timeout}
            )
        
        if self.buffer_size <= 0:
            raise ConfigurationError(
                "buffer_size must be positive",
                details={"value": self.buffer_size}
            )


class BaseDataSource(ABC):
    """
    Abstract base class for all data sources in the Neural SDK.
    
    This class defines the common interface that all data sources must
    implement. It provides basic functionality for connection management,
    event handling, and lifecycle operations.
    
    Subclasses must implement the abstract methods to provide
    protocol-specific functionality.
    
    Attributes:
        config: Configuration for this data source
        state: Current connection state
        callbacks: Registered event callbacks
        metrics: Performance and operational metrics
    
    Example:
        >>> class MyDataSource(BaseDataSource):
        ...     async def _connect_impl(self) -> bool:
        ...         # Implementation specific connection logic
        ...         return True
    """
    
    def __init__(self, config: DataSourceConfig):
        """
        Initialize the base data source.
        
        Args:
            config: Configuration for this data source
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        # Validate configuration before initialization
        config.validate()
        
        self.config = config
        self.state = ConnectionState.DISCONNECTED
        self._callbacks: Dict[str, List[Callable]] = {
            "data": [],      # Called when new data arrives
            "error": [],     # Called on errors
            "connect": [],   # Called on successful connection
            "disconnect": [] # Called on disconnection
        }
        
        # Performance metrics
        self._metrics = {
            "messages_received": 0,
            "messages_processed": 0,
            "errors_count": 0,
            "reconnect_attempts": 0,
            "last_message_time": None,
            "connection_start_time": None,
            "total_uptime": 0.0
        }
        
        # Connection management
        self._connection_lock = asyncio.Lock()
        self._retry_count = 0
        self._connection_task: Optional[asyncio.Task] = None
        
        logger.info(f"Initialized data source: {config.name}")
    
    @abstractmethod
    async def _connect_impl(self) -> bool:
        """
        Protocol-specific connection implementation.
        
        This method must be implemented by subclasses to establish
        the actual connection to the data source.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def _disconnect_impl(self) -> None:
        """
        Protocol-specific disconnection implementation.
        
        This method must be implemented by subclasses to properly
        close the connection to the data source.
        """
        pass
    
    @abstractmethod
    async def _subscribe_impl(self, channels: List[str]) -> bool:
        """
        Protocol-specific subscription implementation.
        
        This method must be implemented by subclasses to subscribe
        to specific data channels or topics.
        
        Args:
            channels: List of channels to subscribe to
            
        Returns:
            bool: True if subscription successful, False otherwise
        """
        pass
    
    async def connect(self) -> bool:
        """
        Establish connection to the data source.
        
        This method handles the connection lifecycle including retries
        and state management. It calls the protocol-specific
        _connect_impl method.
        
        Returns:
            bool: True if connection successful, False otherwise
            
        Raises:
            ConnectionError: If connection cannot be established
            
        Example:
            >>> source = MyDataSource(config)
            >>> if await source.connect():
            ...     print("Connected successfully")
        """
        async with self._connection_lock:
            # Check if already connected
            if self.state == ConnectionState.CONNECTED:
                logger.debug(f"{self.config.name}: Already connected")
                return True
            
            # Update state
            self.state = ConnectionState.CONNECTING
            self._metrics["connection_start_time"] = datetime.now()
            
            # Attempt connection with retries
            while self._retry_count <= self.config.max_retries:
                try:
                    logger.info(
                        f"{self.config.name}: Attempting connection "
                        f"(attempt {self._retry_count + 1}/{self.config.max_retries + 1})"
                    )
                    
                    # Call protocol-specific implementation
                    success = await asyncio.wait_for(
                        self._connect_impl(),
                        timeout=self.config.timeout
                    )
                    
                    if success:
                        self.state = ConnectionState.CONNECTED
                        self._retry_count = 0
                        logger.info(f"{self.config.name}: Connected successfully")
                        
                        # Trigger connect callbacks
                        await self._trigger_callbacks("connect", self)
                        return True
                    
                except asyncio.TimeoutError:
                    logger.warning(
                        f"{self.config.name}: Connection timeout after "
                        f"{self.config.timeout} seconds"
                    )
                except Exception as e:
                    logger.error(f"{self.config.name}: Connection error: {e}")
                
                # Calculate retry delay with exponential backoff
                if self._retry_count < self.config.max_retries:
                    delay = self.config.retry_delay * (
                        self.config.retry_backoff ** self._retry_count
                    )
                    logger.info(f"{self.config.name}: Retrying in {delay:.1f} seconds")
                    await asyncio.sleep(delay)
                    self._retry_count += 1
                    self._metrics["reconnect_attempts"] += 1
                else:
                    break
            
            # Connection failed
            self.state = ConnectionState.ERROR
            error_msg = (
                f"{self.config.name}: Failed to connect after "
                f"{self.config.max_retries + 1} attempts"
            )
            logger.error(error_msg)
            raise ConnectionError(error_msg)
    
    async def disconnect(self) -> None:
        """
        Disconnect from the data source.
        
        This method handles graceful disconnection including cleanup
        and state management.
        
        Example:
            >>> await source.disconnect()
        """
        async with self._connection_lock:
            if self.state == ConnectionState.DISCONNECTED:
                logger.debug(f"{self.config.name}: Already disconnected")
                return
            
            logger.info(f"{self.config.name}: Disconnecting")
            self.state = ConnectionState.CLOSING
            
            try:
                # Call protocol-specific implementation
                await self._disconnect_impl()
                
                # Update metrics
                if self._metrics["connection_start_time"]:
                    uptime = (
                        datetime.now() - self._metrics["connection_start_time"]
                    ).total_seconds()
                    self._metrics["total_uptime"] += uptime
                
                self.state = ConnectionState.DISCONNECTED
                logger.info(f"{self.config.name}: Disconnected successfully")
                
                # Trigger disconnect callbacks
                await self._trigger_callbacks("disconnect", self)
                
            except Exception as e:
                logger.error(f"{self.config.name}: Error during disconnect: {e}")
                self.state = ConnectionState.ERROR
                raise
    
    async def reconnect(self) -> bool:
        """
        Reconnect to the data source.
        
        This method disconnects and then attempts to reconnect
        to the data source.
        
        Returns:
            bool: True if reconnection successful, False otherwise
            
        Example:
            >>> if await source.reconnect():
            ...     print("Reconnected successfully")
        """
        logger.info(f"{self.config.name}: Initiating reconnection")
        self.state = ConnectionState.RECONNECTING
        
        # Disconnect if connected
        if self.state in [ConnectionState.CONNECTED, ConnectionState.ERROR]:
            try:
                await self.disconnect()
            except Exception as e:
                logger.warning(f"{self.config.name}: Error during disconnect: {e}")
        
        # Reset retry counter for fresh connection attempt
        self._retry_count = 0
        
        # Attempt to connect
        return await self.connect()
    
    def is_connected(self) -> bool:
        """
        Check if the data source is currently connected.
        
        Returns:
            bool: True if connected, False otherwise
            
        Example:
            >>> if source.is_connected():
            ...     print("Source is active")
        """
        return self.state == ConnectionState.CONNECTED
    
    def register_callback(
        self,
        event: str,
        callback: Callable[[Any], None]
    ) -> None:
        """
        Register a callback for a specific event.
        
        Args:
            event: Event name ("data", "error", "connect", "disconnect")
            callback: Function to call when event occurs
            
        Raises:
            ValueError: If event type is not recognized
            
        Example:
            >>> def on_data(data):
            ...     print(f"Received: {data}")
            >>> source.register_callback("data", on_data)
        """
        if event not in self._callbacks:
            raise ValueError(
                f"Unknown event type: {event}. "
                f"Valid events: {list(self._callbacks.keys())}"
            )
        
        self._callbacks[event].append(callback)
        logger.debug(f"{self.config.name}: Registered callback for {event}")
    
    def unregister_callback(
        self,
        event: str,
        callback: Callable[[Any], None]
    ) -> None:
        """
        Unregister a callback for a specific event.
        
        Args:
            event: Event name
            callback: Function to remove
            
        Example:
            >>> source.unregister_callback("data", on_data)
        """
        if event in self._callbacks and callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)
            logger.debug(f"{self.config.name}: Unregistered callback for {event}")
    
    async def _trigger_callbacks(self, event: str, data: Any) -> None:
        """
        Trigger all callbacks for a specific event.
        
        Args:
            event: Event name
            data: Data to pass to callbacks
        """
        for callback in self._callbacks.get(event, []):
            try:
                # Handle both sync and async callbacks
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(
                    f"{self.config.name}: Error in {event} callback: {e}",
                    exc_info=True
                )
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics.
        
        Returns:
            Dictionary containing performance metrics
            
        Example:
            >>> metrics = source.get_metrics()
            >>> print(f"Messages received: {metrics['messages_received']}")
        """
        return self._metrics.copy()
    
    def __repr__(self) -> str:
        """Return string representation of the data source."""
        return (
            f"{self.__class__.__name__}("
            f"name='{self.config.name}', "
            f"state={self.state.value})"
        )