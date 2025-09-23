"""
Tests for exception hierarchy.
"""

import pytest
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


class TestNeuralException:
    """Test base exception class."""
    
    def test_basic_exception(self):
        exc = NeuralException("Test error")
        
        assert "Test error" in str(exc)
        assert exc.details == {}  # Default is empty dict, not None
        assert exc.error_code is None
    
    def test_exception_with_details(self):
        details = {"key": "value", "code": 123}
        exc = NeuralException("Test error", details=details)
        
        assert "Test error" in str(exc)
        assert exc.details == details
        assert exc.details["key"] == "value"
    
    def test_exception_with_error_code(self):
        exc = NeuralException("Test error", error_code="ERR_001")
        
        assert exc.error_code == "ERR_001"
    
    def test_exception_with_all_params(self):
        details = {"retry_after": 60}
        exc = NeuralException(
            "Rate limited",
            details=details,
            error_code="RATE_LIMIT"
        )
        
        assert "Rate limited" in str(exc)
        assert exc.details["retry_after"] == 60
        assert exc.error_code == "RATE_LIMIT"
    
    def test_exception_inheritance(self):
        exc = ConnectionError("Connection failed")
        
        assert isinstance(exc, NeuralException)
        assert isinstance(exc, ConnectionError)


class TestSpecificExceptions:
    """Test specific exception types."""
    
    def test_connection_error(self):
        exc = ConnectionError("Failed to connect", details={"host": "localhost"})
        
        assert isinstance(exc, NeuralException)
        assert exc.details["host"] == "localhost"
    
    def test_configuration_error(self):
        exc = ConfigurationError("Invalid config")
        
        assert isinstance(exc, NeuralException)
    
    def test_rate_limit_error(self):
        exc = RateLimitError(
            "Too many requests",
            details={"retry_after": 30, "limit": 100}
        )
        
        assert isinstance(exc, NeuralException)
        assert exc.details["retry_after"] == 30
        assert exc.details["limit"] == 100
    
    def test_authentication_error(self):
        exc = AuthenticationError(
            "Invalid API key",
            error_code="AUTH_INVALID_KEY"
        )
        
        assert isinstance(exc, NeuralException)
        assert exc.error_code == "AUTH_INVALID_KEY"
    
    def test_timeout_error(self):
        exc = TimeoutError(
            "Request timed out",
            details={"timeout": 30, "endpoint": "/api/data"}
        )
        
        assert isinstance(exc, NeuralException)
        assert exc.details["timeout"] == 30
    
    def test_buffer_overflow_error(self):
        exc = BufferOverflowError(
            "Buffer full",
            details={"capacity": 1000, "current_size": 1000}
        )
        
        assert isinstance(exc, NeuralException)
        assert exc.details["capacity"] == 1000
    
    def test_validation_error(self):
        exc = ValidationError(
            "Invalid data format",
            details={"field": "email", "value": "invalid"}
        )
        
        assert isinstance(exc, NeuralException)
        assert exc.details["field"] == "email"
    
    def test_retryable_error(self):
        exc = RetryableError("Temporary failure")
        
        assert isinstance(exc, NeuralException)
    
    def test_transient_error(self):
        exc = TransientError(
            "Network glitch",
            details={"retry_count": 2}
        )
        
        assert isinstance(exc, RetryableError)
        assert isinstance(exc, NeuralException)


class TestExceptionHierarchy:
    """Test exception inheritance hierarchy."""
    
    def test_retryable_hierarchy(self):
        # TransientError should be retryable
        exc = TransientError("Temporary issue")
        
        assert isinstance(exc, RetryableError)
        assert isinstance(exc, NeuralException)
    
    def test_all_exceptions_inherit_from_base(self):
        exceptions = [
            ConnectionError("test"),
            ConfigurationError("test"),
            DataSourceError("test"),
            RateLimitError("test"),
            AuthenticationError("test"),
            TimeoutError("test"),
            BufferOverflowError("test"),
            ValidationError("test"),
            RetryableError("test"),
            TransientError("test")
        ]
        
        for exc in exceptions:
            assert isinstance(exc, NeuralException)
    
    def test_exception_catching(self):
        # Should be able to catch by base class
        try:
            raise ConnectionError("Test")
        except NeuralException as e:
            assert isinstance(e, ConnectionError)
        
        # Should be able to catch retryable errors
        try:
            raise TransientError("Test")
        except RetryableError as e:
            assert isinstance(e, TransientError)