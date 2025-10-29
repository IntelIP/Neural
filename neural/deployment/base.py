"""
Abstract base classes for the Neural SDK deployment module.

This module defines the provider pattern for deployment backends,
allowing extensibility to support Docker, E2B, cloud providers, etc.
"""

from abc import ABC, abstractmethod
from typing import Any

from neural.deployment.config import DeploymentConfig, DeploymentInfo, DeploymentResult, DeploymentStatus


class DeploymentProvider(ABC):
    """Abstract base class for deployment providers.

    All deployment providers (Docker, E2B, AWS, GCP, etc.) must inherit from
    this class and implement its abstract methods.

    This enables a consistent API across different deployment backends while
    allowing provider-specific implementations.

    Example:
        ```python
        class DockerDeploymentProvider(DeploymentProvider):
            async def deploy(self, config: DeploymentConfig) -> DeploymentResult:
                # Docker-specific deployment logic
                ...
        ```
    """

    @abstractmethod
    async def deploy(self, config: DeploymentConfig) -> DeploymentResult:
        """Deploy a trading bot using this provider.

        Args:
            config: Deployment configuration

        Returns:
            DeploymentResult with deployment details

        Raises:
            DeploymentError: If deployment fails
        """
        pass

    @abstractmethod
    async def stop(self, deployment_id: str) -> bool:
        """Stop a running deployment.

        Args:
            deployment_id: Unique identifier for the deployment

        Returns:
            True if successfully stopped, False otherwise

        Raises:
            DeploymentError: If stop operation fails
            ContainerNotFoundError: If deployment doesn't exist
        """
        pass

    @abstractmethod
    async def status(self, deployment_id: str) -> DeploymentStatus:
        """Get the current status of a deployment.

        Args:
            deployment_id: Unique identifier for the deployment

        Returns:
            DeploymentStatus with current status info

        Raises:
            DeploymentError: If status check fails
            ContainerNotFoundError: If deployment doesn't exist
        """
        pass

    @abstractmethod
    async def logs(self, deployment_id: str, tail: int = 100) -> list[str]:
        """Get recent logs from a deployment.

        Args:
            deployment_id: Unique identifier for the deployment
            tail: Number of recent log lines to return

        Returns:
            List of log lines (most recent last)

        Raises:
            DeploymentError: If log retrieval fails
            ContainerNotFoundError: If deployment doesn't exist
        """
        pass

    @abstractmethod
    async def list_deployments(self) -> list[DeploymentInfo]:
        """List all active deployments.

        Returns:
            List of DeploymentInfo for all active deployments

        Raises:
            DeploymentError: If listing fails
        """
        pass

    async def restart(self, deployment_id: str) -> bool:
        """Restart a deployment (stop then deploy again).

        Default implementation using stop() and deploy().
        Providers can override for more efficient restart logic.

        Args:
            deployment_id: Unique identifier for the deployment

        Returns:
            True if successfully restarted, False otherwise

        Raises:
            DeploymentError: If restart fails
        """
        # Get current config from status
        status_info = await self.status(deployment_id)

        # Stop the deployment
        await self.stop(deployment_id)

        # This is a simplified implementation - in practice, you'd need to
        # store the original config or retrieve it from the deployment metadata
        raise NotImplementedError(
            "Restart requires storing deployment configs. "
            "Providers should override this method."
        )

    async def cleanup(self) -> None:
        """Clean up provider resources.

        Optional method for providers to clean up resources, connections, etc.
        Called when the provider is being shut down.
        """
        pass


class DeploymentContext:
    """Async context manager for deployments.

    Provides a convenient way to deploy and automatically clean up
    resources using Python's async context manager protocol.

    Example:
        ```python
        async with DeploymentContext(provider, config) as deployment:
            status = await provider.status(deployment.deployment_id)
            logs = await provider.logs(deployment.deployment_id)
        # Deployment is automatically stopped when exiting the context
        ```
    """

    def __init__(
        self,
        provider: DeploymentProvider,
        config: DeploymentConfig,
        auto_stop: bool = True,
    ):
        """Initialize deployment context.

        Args:
            provider: Deployment provider to use
            config: Deployment configuration
            auto_stop: Whether to automatically stop deployment on exit (default: True)
        """
        self.provider = provider
        self.config = config
        self.auto_stop = auto_stop
        self.deployment_result: DeploymentResult | None = None

    async def __aenter__(self) -> DeploymentResult:
        """Deploy when entering the context.

        Returns:
            DeploymentResult from the deployment operation
        """
        self.deployment_result = await self.provider.deploy(self.config)
        return self.deployment_result

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Stop deployment when exiting the context (if auto_stop is True).

        Args:
            exc_type: Exception type (if an exception occurred)
            exc_val: Exception value (if an exception occurred)
            exc_tb: Exception traceback (if an exception occurred)
        """
        if self.auto_stop and self.deployment_result:
            await self.provider.stop(self.deployment_result.deployment_id)

    async def status(self) -> DeploymentStatus:
        """Get current deployment status.

        Convenience method to check status without keeping track of deployment_id.

        Returns:
            Current deployment status

        Raises:
            RuntimeError: If called before deployment is created
        """
        if not self.deployment_result:
            raise RuntimeError("Deployment has not been created yet")
        return await self.provider.status(self.deployment_result.deployment_id)

    async def logs(self, tail: int = 100) -> list[str]:
        """Get deployment logs.

        Convenience method to get logs without keeping track of deployment_id.

        Args:
            tail: Number of recent log lines to return

        Returns:
            List of log lines

        Raises:
            RuntimeError: If called before deployment is created
        """
        if not self.deployment_result:
            raise RuntimeError("Deployment has not been created yet")
        return await self.provider.logs(self.deployment_result.deployment_id, tail=tail)
