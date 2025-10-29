"""
Custom exceptions for the Neural SDK deployment module.

This module defines deployment-specific exceptions for better error handling
and debugging of deployment operations.
"""


class DeploymentError(Exception):
    """Base exception for all deployment-related errors."""

    pass


class ProviderNotFoundError(DeploymentError):
    """Raised when a deployment provider is not available or not installed."""

    pass


class ContainerNotFoundError(DeploymentError):
    """Raised when a Docker container cannot be found."""

    pass


class ResourceLimitExceededError(DeploymentError):
    """Raised when resource limits (CPU, memory, etc.) are exceeded."""

    pass


class DeploymentTimeoutError(DeploymentError):
    """Raised when a deployment operation times out."""

    pass


class ConfigurationError(DeploymentError):
    """Raised when deployment configuration is invalid."""

    pass


class ImageBuildError(DeploymentError):
    """Raised when Docker image building fails."""

    pass


class NetworkError(DeploymentError):
    """Raised when network-related deployment operations fail."""

    pass


class DatabaseError(DeploymentError):
    """Raised when database operations fail during deployment."""

    pass


class MonitoringError(DeploymentError):
    """Raised when monitoring setup or data collection fails."""

    pass
