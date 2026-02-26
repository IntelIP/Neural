"""
Neural SDK Deployment Module (Experimental)

Provides Docker-based deployment infrastructure for trading bots with
database persistence, monitoring, and multi-environment support.

  EXPERIMENTAL: This module is experimental in Neural SDK Beta v0.4.0.
Use with caution in production environments.

Example:
    ```python
    from neural.deployment import DockerDeploymentProvider, DeploymentConfig, deploy

    # Configure deployment
    config = DeploymentConfig(
        bot_name="MyNFLBot",
        strategy_type="NFL",
        environment="paper",
        algorithm_config={"algorithm_type": "mean_reversion"}
    )

    # Deploy with context manager
    provider = DockerDeploymentProvider()
    async with deploy(provider, config) as deployment:
        print(f"Deployed: {deployment.deployment_id}")

        # Get status
        status = await provider.status(deployment.deployment_id)
        print(f"Status: {status.status}")
    # Deployment is automatically stopped when exiting context
    ```
"""

from typing import Any

# Core abstractions
from neural.deployment.base import DeploymentContext, DeploymentProvider

# Configuration models
from neural.deployment.config import (
    DatabaseConfig,
    DeploymentConfig,
    DeploymentInfo,
    DeploymentResult,
    DeploymentStatus,
    DockerConfig,
    MonitoringConfig,
)

# Exceptions
from neural.deployment.exceptions import (
    ConfigurationError,
    ContainerNotFoundError,
    DatabaseError,
    DeploymentError,
    DeploymentTimeoutError,
    ImageBuildError,
    MonitoringError,
    NetworkError,
    ProviderNotFoundError,
    ResourceLimitExceededError,
)
from neural.deployment.registry import create_provider, list_providers, register_provider

_DOCKER_AVAILABLE = True
_DOCKER_IMPORT_ERROR: Exception | None = None

try:
    # Docker provider
    from neural.deployment.docker import (
        DockerDeploymentProvider,
        render_compose_file,
        render_dockerfile,
        render_dockerignore,
        write_compose_file,
    )
except Exception as exc:  # pragma: no cover - depends on optional dependency presence
    _DOCKER_AVAILABLE = False
    _DOCKER_IMPORT_ERROR = exc

    class DockerDeploymentProvider:  # type: ignore[no-redef]
        """Placeholder that raises when Docker deployment extras are missing."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ProviderNotFoundError(
                "Docker deployment provider is unavailable. "
                "Install optional dependencies with: pip install 'neural-sdk[deployment]'"
            ) from _DOCKER_IMPORT_ERROR

    def render_compose_file(*args: Any, **kwargs: Any) -> str:
        raise ProviderNotFoundError(
            "Docker compose rendering is unavailable. "
            "Install optional dependencies with: pip install 'neural-sdk[deployment]'"
        ) from _DOCKER_IMPORT_ERROR

    def render_dockerfile(*args: Any, **kwargs: Any) -> str:
        raise ProviderNotFoundError(
            "Dockerfile rendering is unavailable. "
            "Install optional dependencies with: pip install 'neural-sdk[deployment]'"
        ) from _DOCKER_IMPORT_ERROR

    def render_dockerignore() -> str:
        raise ProviderNotFoundError(
            "Docker ignore rendering is unavailable. "
            "Install optional dependencies with: pip install 'neural-sdk[deployment]'"
        ) from _DOCKER_IMPORT_ERROR

    def write_compose_file(*args: Any, **kwargs: Any) -> Any:
        raise ProviderNotFoundError(
            "Docker compose writing is unavailable. "
            "Install optional dependencies with: pip install 'neural-sdk[deployment]'"
        ) from _DOCKER_IMPORT_ERROR


__all__ = [
    # Core abstractions
    "DeploymentProvider",
    "DeploymentContext",
    "deploy",
    # Configuration
    "DeploymentConfig",
    "DockerConfig",
    "DatabaseConfig",
    "MonitoringConfig",
    "DeploymentResult",
    "DeploymentStatus",
    "DeploymentInfo",
    # Providers
    "DockerDeploymentProvider",
    "register_provider",
    "create_provider",
    "list_providers",
    # Docker utilities
    "render_dockerfile",
    "render_dockerignore",
    "render_compose_file",
    "write_compose_file",
    # Exceptions
    "DeploymentError",
    "ProviderNotFoundError",
    "ContainerNotFoundError",
    "ResourceLimitExceededError",
    "DeploymentTimeoutError",
    "ConfigurationError",
    "ImageBuildError",
    "NetworkError",
    "DatabaseError",
    "MonitoringError",
]

# Register built-in providers so callers can use create_provider("docker", ...).
if _DOCKER_AVAILABLE:
    register_provider("docker", DockerDeploymentProvider, replace=True)


# Convenience function for deploying with context manager
def deploy(
    provider: DeploymentProvider,
    config: DeploymentConfig,
    auto_stop: bool = True,
) -> DeploymentContext:
    """Deploy a trading bot using a deployment provider with automatic cleanup.

    This is a convenience function that creates a DeploymentContext for use
    with Python's async context manager protocol.

    Args:
        provider: Deployment provider to use (e.g., DockerDeploymentProvider)
        config: Deployment configuration
        auto_stop: Whether to automatically stop deployment on context exit (default: True)

    Returns:
        DeploymentContext that can be used with 'async with'

    Example:
        ```python
        provider = DockerDeploymentProvider()
        config = DeploymentConfig(bot_name="MyBot", strategy_type="NFL")

        async with deploy(provider, config) as deployment:
            status = await provider.status(deployment.deployment_id)
            print(f"Bot is {status.status}")
        # Deployment automatically stopped here
        ```
    """
    return DeploymentContext(provider, config, auto_stop=auto_stop)
