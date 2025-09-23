"""
Custom exceptions for the Neural SDK Data Collection Infrastructure.

This module defines all custom exceptions used throughout the data collection
layer. Each exception provides clear context about what went wrong and 
suggestions for resolution.
"""

from typing import Optional, Dict, Any


class NeuralException(Exception):
    """
    Base exception class for all Neural SDK exceptions.
    
    All custom exceptions in the Neural SDK should inherit from this class
    to maintain consistency in error handling and reporting.
    
    Attributes:
        message: Human-readable error description
        details: Additional context about the error
        error_code: Optional error code for programmatic handling
    """
    
    def __init__(
        self, 
        message: str, 
        details: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None
    ):
        """
        Initialize a Neural exception.
        
        Args:
            message: Primary error message
            details: Additional error context as key-value pairs
            error_code: Optional error code for categorization
        """
        self.message = message
        self.details = details or {}
        self.error_code = error_code
        super().__init__(self.message)
    
    def __str__(self) -> str:
        """Return formatted error message with details."""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class ConnectionError(NeuralException):
    """
    Raised when a connection to a data source cannot be established or is lost.
    
    This exception indicates issues with network connectivity, authentication,
    or endpoint availability.
    
    Example:
        >>> raise ConnectionError(
        ...     "Failed to connect to WebSocket",
        ...     details={"url": "wss://api.example.com", "timeout": 30}
        ... )
    """
    pass


class ConfigurationError(NeuralException):
    """
    Raised when configuration is invalid or missing required parameters.
    
    This exception helps identify configuration issues early in the
    initialization process.
    
    Example:
        >>> raise ConfigurationError(
        ...     "Missing required configuration field",
        ...     details={"field": "api_key", "config_file": "config.yaml"}
        ... )
    """
    pass


class DataSourceError(NeuralException):
    """
    Raised when a data source encounters an error during operation.
    
    This is a general exception for data source issues that don't fit
    into more specific categories.
    
    Example:
        >>> raise DataSourceError(
        ...     "Data source returned invalid response",
        ...     details={"source": "ESPN API", "status_code": 500}
        ... )
    """
    pass


class RateLimitError(NeuralException):
    """
    Raised when rate limits are exceeded for a data source.
    
    This exception includes information about when the rate limit
    will reset and suggested retry timing.
    
    Attributes:
        retry_after: Seconds to wait before retrying
        reset_time: Unix timestamp when rate limit resets
    
    Example:
        >>> raise RateLimitError(
        ...     "Rate limit exceeded",
        ...     details={"limit": 100, "window": "1m", "retry_after": 60}
        ... )
    """
    
    def __init__(
        self, 
        message: str, 
        retry_after: Optional[int] = None,
        reset_time: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize a rate limit error.
        
        Args:
            message: Error description
            retry_after: Seconds to wait before retry
            reset_time: Unix timestamp for rate limit reset
            **kwargs: Additional details for parent class
        """
        super().__init__(message, **kwargs)
        self.retry_after = retry_after
        self.reset_time = reset_time
        
        if retry_after:
            self.details["retry_after"] = retry_after
        if reset_time:
            self.details["reset_time"] = reset_time


class AuthenticationError(NeuralException):
    """
    Raised when authentication fails for a data source.
    
    This exception indicates issues with API keys, tokens, or
    other authentication credentials.
    
    Example:
        >>> raise AuthenticationError(
        ...     "Invalid API key",
        ...     details={"source": "Kalshi API", "key_id": "xxx...xxx"}
        ... )
    """
    pass


class TimeoutError(NeuralException):
    """
    Raised when an operation exceeds its timeout limit.
    
    This exception helps distinguish between connection timeouts
    and other types of failures.
    
    Example:
        >>> raise TimeoutError(
        ...     "Connection timeout",
        ...     details={"operation": "connect", "timeout": 30}
        ... )
    """
    pass


class BufferOverflowError(NeuralException):
    """
    Raised when a data buffer exceeds its maximum capacity.
    
    This exception indicates that data is arriving faster than
    it can be processed.
    
    Example:
        >>> raise BufferOverflowError(
        ...     "Message buffer full",
        ...     details={"buffer_size": 10000, "dropped_messages": 50}
        ... )
    """
    pass


class ValidationError(NeuralException):
    """
    Raised when data validation fails.
    
    This exception is used when incoming data doesn't match
    expected schemas or contains invalid values.
    
    Example:
        >>> raise ValidationError(
        ...     "Invalid data format",
        ...     details={"field": "price", "expected": "float", "received": "string"}
        ... )
    """
    pass


class RetryableError(NeuralException):
    """
    Base class for errors that should trigger automatic retry.
    
    Subclasses of this exception will be automatically retried
    by the retry logic in the data collection infrastructure.
    
    Attributes:
        attempt: Current retry attempt number
        max_attempts: Maximum number of retry attempts
    """
    
    def __init__(
        self,
        message: str,
        attempt: int = 1,
        max_attempts: int = 3,
        **kwargs
    ):
        """
        Initialize a retryable error.
        
        Args:
            message: Error description
            attempt: Current attempt number
            max_attempts: Maximum attempts allowed
            **kwargs: Additional details for parent class
        """
        super().__init__(message, **kwargs)
        self.attempt = attempt
        self.max_attempts = max_attempts
        self.details["attempt"] = f"{attempt}/{max_attempts}"


class TransientError(RetryableError):
    """
    Raised for temporary errors that are likely to resolve.
    
    These errors trigger automatic retry with exponential backoff.
    
    Example:
        >>> raise TransientError(
        ...     "Temporary network issue",
        ...     attempt=2,
        ...     max_attempts=5
        ... )
    """
    pass