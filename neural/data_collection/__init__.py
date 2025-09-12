"""
Neural SDK Data Collection Infrastructure.

This module provides the foundation for collecting data from various sources
including WebSocket streams and REST APIs. It offers a unified interface
for managing different data sources with features like automatic reconnection,
configuration management, and error handling.
"""

from neural.data_collection.base import (
    BaseDataSource,
    DataSourceConfig,
    ConnectionState
)
from neural.data_collection.config import ConfigManager
from neural.data_collection.exceptions import (
    NeuralException,
    ConnectionError,
    ConfigurationError,
    DataSourceError,
    RateLimitError,
    AuthenticationError,
    TimeoutError,
    BufferOverflowError,
    ValidationError,
    RetryableError,
    TransientError
)
from neural.data_collection.websocket_handler import (
    WebSocketDataSource,
    WebSocketConfig
)

__all__ = [
    # Base classes
    "BaseDataSource",
    "DataSourceConfig", 
    "ConnectionState",
    
    # Configuration
    "ConfigManager",
    
    # WebSocket
    "WebSocketDataSource",
    "WebSocketConfig",
    
    # Exceptions
    "NeuralException",
    "ConnectionError",
    "ConfigurationError",
    "DataSourceError",
    "RateLimitError",
    "AuthenticationError",
    "TimeoutError",
    "BufferOverflowError",
    "ValidationError",
    "RetryableError",
    "TransientError"
]