"""
Custom exceptions for the Neural Trading SDK.

This module defines all custom exceptions used throughout the SDK
to provide clear error messages and proper error handling.
"""

from typing import Any, Optional


class SDKError(Exception):
    """Base exception for all SDK errors."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(SDKError):
    """Raised when there's a configuration error."""

    def __init__(self, message: str, config_key: Optional[str] = None) -> None:
        super().__init__(message, {"config_key": config_key} if config_key else None)
        self.config_key = config_key


class ConnectionError(SDKError):
    """Raised when there's a connection error."""

    def __init__(
        self, message: str, service: Optional[str] = None, retry_count: int = 0
    ) -> None:
        super().__init__(message, {"service": service, "retry_count": retry_count})
        self.service = service
        self.retry_count = retry_count


class TradingError(SDKError):
    """Raised when there's a trading-related error."""

    def __init__(
        self,
        message: str,
        market_ticker: Optional[str] = None,
        order_id: Optional[str] = None,
    ) -> None:
        super().__init__(
            message, {"market_ticker": market_ticker, "order_id": order_id}
        )
        self.market_ticker = market_ticker
        self.order_id = order_id


class ValidationError(SDKError):
    """Raised when there's a validation error."""

    def __init__(
        self, message: str, field: Optional[str] = None, value: Optional[Any] = None
    ) -> None:
        super().__init__(message, {"field": field, "value": value})
        self.field = field
        self.value = value


class AuthenticationError(SDKError):
    """Raised when there's an authentication error."""

    def __init__(self, message: str, service: Optional[str] = None) -> None:
        super().__init__(message, {"service": service})
        self.service = service


class RateLimitError(SDKError):
    """Raised when rate limits are exceeded."""

    def __init__(self, message: str, retry_after: Optional[int] = None) -> None:
        super().__init__(message, {"retry_after": retry_after})
        self.retry_after = retry_after


class InsufficientFundsError(TradingError):
    """Raised when there are insufficient funds for a trade."""

    def __init__(self, message: str, required: float, available: float) -> None:
        super().__init__(message, None, None)
        self.required = required
        self.available = available
        self.details.update({"required": required, "available": available})


class MarketNotFoundError(TradingError):
    """Raised when a market is not found."""

    def __init__(self, message: str, market_ticker: str) -> None:
        super().__init__(message, market_ticker, None)
        self.market_ticker = market_ticker


class OrderRejectedError(TradingError):
    """Raised when an order is rejected."""

    def __init__(self, message: str, market_ticker: str, reason: str) -> None:
        super().__init__(message, market_ticker, None)
        self.reason = reason
        self.details["reason"] = reason
