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
from neural.data_collection.rest_handler import (
    RestDataSource,
    RestConfig,
    HttpMethod
)
from neural.data_collection.buffer import (
    CircularBuffer,
    AsyncBuffer,
    OverflowStrategy,
    BufferStats
)
from neural.data_collection.pipeline import (
    DataPipeline,
    TransformStage
)
from neural.data_collection.utils import (
    retry,
    RetryConfig,
    AsyncTaskManager,
    RateLimiter,
    create_logger,
    generate_id,
    safe_json_parse,
    MovingAverage,
    PerformanceTimer
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
    
    # REST
    "RestDataSource",
    "RestConfig",
    "HttpMethod",
    
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
    "TransientError",
    
    # Buffer
    "CircularBuffer",
    "AsyncBuffer",
    "OverflowStrategy",
    "BufferStats",
    
    # Pipeline
    "DataPipeline",
    "TransformStage",
    
    # Utils
    "retry",
    "RetryConfig",
    "AsyncTaskManager",
    "RateLimiter",
    "create_logger",
    "generate_id",
    "safe_json_parse",
    "MovingAverage",
    "PerformanceTimer"
]