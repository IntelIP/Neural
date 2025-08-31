"""
Neural Trading Platform SDK
Easy integration of custom data sources
"""

from .core.base_adapter import (
    DataSourceAdapter,
    DataSourceMetadata,
    StandardizedEvent,
    EventType,
    SignalStrength,
    DataSourceError,
    AuthenticationError,
    RateLimitError,
    DataValidationError
)

from .core.sdk_manager import (
    SDKManager,
    AdapterConfig
)

# Version
__version__ = "1.0.0"

# Expose main classes
__all__ = [
    "DataSourceAdapter",
    "DataSourceMetadata",
    "StandardizedEvent",
    "EventType",
    "SignalStrength",
    "SDKManager",
    "AdapterConfig",
    "DataSourceError",
    "AuthenticationError",
    "RateLimitError",
    "DataValidationError"
]