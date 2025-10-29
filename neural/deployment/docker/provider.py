"""
Docker deployment provider for the Neural SDK.

This module implements the Docker-based deployment provider for running
trading bots in containers.
"""

import asyncio
import concurrent.futures
import json
import logging
import os
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import docker
from docker.errors import DockerException, NotFound

from neural.deployment.base import DeploymentProvider
from neural.deployment.config import (
    DeploymentConfig,
    DeploymentInfo,
    DeploymentResult,
    DeploymentStatus,
    DockerConfig,
)
from neural.deployment.docker.compose import write_compose_file
from neural.deployment.docker.templates import render_dockerfile, render_dockerignore
from neural.deployment.exceptions import (
    ConfigurationError,
    ContainerNotFoundError,
    DeploymentError,
    ImageBuildError,
    ResourceLimitExceededError,
)

logger = logging.getLogger(__name__)


class DockerDeploymentProvider(DeploymentProvider):
    """Docker-based deployment provider.

    This provider deploys trading bots as Docker containers,
    supporting both individual containers and Docker Compose stacks.

    Example:
        ```python
        from neural.deployment import DockerDeploymentProvider, DeploymentConfig

        provider = DockerDeploymentProvider()
        config = DeploymentConfig(
            bot_name="MyBot",
            strategy_type="NFL",
            environment="paper"
        )

        result = await provider.deploy(config)
        print(f"Deployed: {result.deployment_id}")
        ```
    """

    def __init__(
        self,
        docker_client: docker.DockerClient | None = None,
        project_root: Path | None = None,
    ):
        """Initialize Docker deployment provider.

        Args:
            docker_client: Docker client (created automatically if None)
            project_root: Root directory for project files (defaults to cwd)
        """
        try:
            self.docker_client = docker_client or docker.from_env()
        except DockerException as e:
            raise DeploymentError(f"Failed to connect to Docker: {e}") from e

        self.project_root = Path(project_root or os.getcwd())
        self.active_deployments: dict[str, dict[str, Any]] = {}
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    async def deploy(self, config: DeploymentConfig) -> DeploymentResult:
        """Deploy a trading bot to Docker container.

        Args:
            config: Deployment configuration

        Returns:
            DeploymentResult with deployment details

        Raises:
            DeploymentError: If deployment fails
            ImageBuildError: If Docker image build fails
        """
        deployment_id = str(uuid.uuid4())
        container_name = self._generate_container_name(config.bot_name, deployment_id)

        logger.info(f"Starting deployment {deployment_id} for bot: {config.bot_name}")

        try:
            # Build Docker image
            image_tag = await self._build_image(config, deployment_id)

            # Prepare environment variables
            env_vars = self._prepare_env_vars(config)

            # Create container configuration
            container_config = self._create_container_config(
                config, image_tag, container_name, env_vars
            )

            # Create and start container (run in executor to avoid blocking event loop)
            loop = asyncio.get_event_loop()
            container = await loop.run_in_executor(
                self._executor, lambda: self.docker_client.containers.create(**container_config)
            )
            await loop.run_in_executor(self._executor, container.start)

            # Store deployment info
            deployment_info = {
                "container_id": container.id,
                "container_name": container_name,
                "config": config.model_dump(),
                "status": "running",
                "created_at": datetime.now(),
                "image_tag": image_tag,
            }
            self.active_deployments[deployment_id] = deployment_info

            logger.info(f"Deployment {deployment_id} started successfully")

            return DeploymentResult(
                deployment_id=deployment_id,
                status="running",
                container_id=container.id,
                container_name=container_name,
                created_at=deployment_info["created_at"],
                endpoints={},
                metadata={"image_tag": image_tag},
            )

        except DockerException as e:
            logger.error(f"Docker deployment failed: {e}")
            raise DeploymentError(f"Failed to deploy container: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected deployment error: {e}")
            raise DeploymentError(f"Deployment failed: {e}") from e

    async def stop(self, deployment_id: str) -> bool:
        """Stop a running deployment.

        Args:
            deployment_id: Unique identifier for the deployment

        Returns:
            True if successfully stopped

        Raises:
            ContainerNotFoundError: If deployment doesn't exist
        """
        if deployment_id not in self.active_deployments:
            raise ContainerNotFoundError(f"Deployment not found: {deployment_id}")

        deployment = self.active_deployments[deployment_id]

        try:
            loop = asyncio.get_event_loop()
            container = await loop.run_in_executor(
                self._executor, self.docker_client.containers.get, deployment["container_id"]
            )
            await loop.run_in_executor(self._executor, lambda: container.stop(timeout=30))
            await loop.run_in_executor(self._executor, container.remove)

            del self.active_deployments[deployment_id]
            logger.info(f"Stopped deployment: {deployment_id}")
            return True

        except NotFound:
            del self.active_deployments[deployment_id]
            raise ContainerNotFoundError(f"Container not found: {deployment_id}")
        except DockerException as e:
            logger.error(f"Failed to stop deployment {deployment_id}: {e}")
            raise DeploymentError(f"Failed to stop deployment: {e}") from e

    async def status(self, deployment_id: str) -> DeploymentStatus:
        """Get current deployment status.

        Args:
            deployment_id: Unique identifier for the deployment

        Returns:
            DeploymentStatus with current status info

        Raises:
            ContainerNotFoundError: If deployment doesn't exist
        """
        if deployment_id not in self.active_deployments:
            raise ContainerNotFoundError(f"Deployment not found: {deployment_id}")

        deployment = self.active_deployments[deployment_id]

        try:
            loop = asyncio.get_event_loop()
            container = await loop.run_in_executor(
                self._executor, self.docker_client.containers.get, deployment["container_id"]
            )
            await loop.run_in_executor(self._executor, container.reload)

            # Calculate uptime
            created_at = deployment["created_at"]
            uptime = (datetime.now() - created_at).total_seconds()

            # Get container stats (run in executor to avoid blocking)
            stats = await loop.run_in_executor(
                self._executor, lambda: container.stats(stream=False)
            )

            return DeploymentStatus(
                deployment_id=deployment_id,
                status=container.status,
                uptime_seconds=uptime,
                logs=[],  # Use logs() method for full logs
                health_status=container.attrs.get("State", {}).get("Health", {}).get("Status"),
                metrics=self._extract_metrics(stats),
            )

        except NotFound:
            raise ContainerNotFoundError(f"Container not found: {deployment_id}")
        except DockerException as e:
            raise DeploymentError(f"Failed to get status: {e}") from e

    async def logs(self, deployment_id: str, tail: int = 100) -> list[str]:
        """Get recent logs from deployment.

        Args:
            deployment_id: Unique identifier for the deployment
            tail: Number of recent log lines to return

        Returns:
            List of log lines

        Raises:
            ContainerNotFoundError: If deployment doesn't exist
        """
        if deployment_id not in self.active_deployments:
            raise ContainerNotFoundError(f"Deployment not found: {deployment_id}")

        deployment = self.active_deployments[deployment_id]

        try:
            loop = asyncio.get_event_loop()
            container = await loop.run_in_executor(
                self._executor, self.docker_client.containers.get, deployment["container_id"]
            )
            logs = await loop.run_in_executor(
                self._executor, lambda: container.logs(tail=tail, timestamps=True).decode("utf-8")
            )
            return logs.strip().split("\n") if logs else []

        except NotFound:
            raise ContainerNotFoundError(f"Container not found: {deployment_id}")
        except DockerException as e:
            raise DeploymentError(f"Failed to get logs: {e}") from e

    async def list_deployments(self) -> list[DeploymentInfo]:
        """List all active deployments.

        Returns:
            List of DeploymentInfo for all active deployments
        """
        deployments = []
        loop = asyncio.get_event_loop()

        for dep_id, deployment in self.active_deployments.items():
            try:
                container = await loop.run_in_executor(
                    self._executor, self.docker_client.containers.get, deployment["container_id"]
                )
                await loop.run_in_executor(self._executor, container.reload)

                deployments.append(
                    DeploymentInfo(
                        deployment_id=dep_id,
                        bot_name=deployment["config"]["bot_name"],
                        status=container.status,
                        environment=deployment["config"]["environment"],
                        created_at=deployment["created_at"],
                        deployment_type="docker",
                    )
                )
            except NotFound:
                # Container was removed externally
                continue

        return deployments

    # Private helper methods

    def _generate_container_name(self, bot_name: str, deployment_id: str) -> str:
        """Generate a unique container name."""
        safe_name = bot_name.lower().replace(" ", "-").replace("_", "-")
        short_id = deployment_id[:8]
        return f"neural-bot-{safe_name}-{short_id}"

    async def _build_image(self, config: DeploymentConfig, deployment_id: str) -> str:
        """Build Docker image for the deployment."""
        image_tag = f"neural-trading-bot:{config.environment}-{deployment_id[:8]}"

        try:
            # Generate Dockerfile
            dockerfile_content = render_dockerfile(
                algorithm_type=config.algorithm_config.get("algorithm_type", "mean_reversion"),
                environment=config.environment,
                bot_name=config.bot_name,
                database_enabled=config.database_enabled,
                websocket_enabled=config.websocket_enabled,
                monitoring_enabled=config.monitoring_enabled,
            )

            # Write Dockerfile to temp location
            build_dir = self.project_root / "build" / deployment_id
            build_dir.mkdir(parents=True, exist_ok=True)

            dockerfile_path = build_dir / "Dockerfile"
            dockerfile_path.write_text(dockerfile_content)

            # Build image (run in executor - can take several minutes)
            logger.info(f"Building Docker image: {image_tag}")
            loop = asyncio.get_event_loop()
            image, build_logs = await loop.run_in_executor(
                self._executor,
                lambda: self.docker_client.images.build(
                    path=str(self.project_root), dockerfile=str(dockerfile_path), tag=image_tag, rm=True
                ),
            )

            logger.info(f"Image built successfully: {image_tag}")
            return image_tag

        except DockerException as e:
            raise ImageBuildError(f"Failed to build Docker image: {e}") from e

    def _prepare_env_vars(self, config: DeploymentConfig) -> dict[str, str]:
        """Prepare environment variables for the container."""
        return {
            "BOT_NAME": config.bot_name,
            "STRATEGY_TYPE": config.strategy_type,
            "ENVIRONMENT": config.environment,
            "DATABASE_ENABLED": str(config.database_enabled).lower(),
            "WEBSOCKET_ENABLED": str(config.websocket_enabled).lower(),
            "MONITORING_ENABLED": str(config.monitoring_enabled).lower(),
            "KALSHI_API_KEY_ID": os.getenv("KALSHI_API_KEY_ID", ""),
            "RISK_CONFIG": json.dumps(config.risk_config),
            "ALGORITHM_CONFIG": json.dumps(config.algorithm_config),
        }

    def _create_container_config(
        self,
        config: DeploymentConfig,
        image_tag: str,
        container_name: str,
        env_vars: dict[str, str],
    ) -> dict[str, Any]:
        """Create Docker container configuration."""
        # Validate secrets directory exists
        secrets_path = self.project_root / "secrets"
        if not secrets_path.exists():
            raise ConfigurationError(
                f"Secrets directory not found: {secrets_path}\n"
                f"Please create the secrets directory with your Kalshi API credentials:\n"
                f"  mkdir -p {secrets_path}\n"
                f"  echo 'KALSHI_API_KEY=your_key' > {secrets_path}/.env"
            )

        if not secrets_path.is_dir():
            raise ConfigurationError(f"Secrets path exists but is not a directory: {secrets_path}")

        container_config: dict[str, Any] = {
            "image": image_tag,
            "name": container_name,
            "environment": env_vars,
            "volumes": {
                str(secrets_path): {"bind": "/secrets", "mode": "ro"},
                f"{container_name}_data": {"bind": "/app/data", "mode": "rw"},
            },
            "restart_policy": {"Name": "unless-stopped"},
            "labels": {
                "app": "neural-trading-bot",
                "bot_name": config.bot_name,
                "strategy_type": config.strategy_type,
                "environment": config.environment,
            },
            "detach": True,
        }

        # Add resource limits
        resources = config.compute_resources
        if resources:
            if "cpu_limit" in resources:
                container_config["nano_cpus"] = int(float(resources["cpu_limit"]) * 1e9)
            if "memory_limit" in resources:
                container_config["mem_limit"] = resources["memory_limit"]

        return container_config

    def _extract_metrics(self, stats: dict[str, Any]) -> dict[str, Any]:
        """Extract key metrics from container stats."""
        try:
            cpu_stats = stats.get("cpu_stats", {})
            memory_stats = stats.get("memory_stats", {})

            return {
                "cpu_percent": self._calculate_cpu_percent(stats),
                "memory_usage_mb": memory_stats.get("usage", 0) / (1024 * 1024),
                "memory_limit_mb": memory_stats.get("limit", 0) / (1024 * 1024),
            }
        except Exception as e:
            logger.warning(f"Failed to extract metrics: {e}")
            return {}

    def _calculate_cpu_percent(self, stats: dict[str, Any]) -> float:
        """Calculate CPU percentage from stats."""
        try:
            cpu_stats = stats.get("cpu_stats", {})
            precpu_stats = stats.get("precpu_stats", {})

            cpu_delta = cpu_stats.get("cpu_usage", {}).get("total_usage", 0) - precpu_stats.get(
                "cpu_usage", {}
            ).get("total_usage", 0)
            system_delta = cpu_stats.get("system_cpu_usage", 0) - precpu_stats.get(
                "system_cpu_usage", 0
            )

            if system_delta > 0 and cpu_delta > 0:
                num_cpus = len(cpu_stats.get("cpu_usage", {}).get("percpu_usage", [1]))
                return (cpu_delta / system_delta) * num_cpus * 100.0

            return 0.0
        except Exception:
            return 0.0

    async def cleanup(self) -> None:
        """Cleanup resources including the thread pool executor.

        This should be called when the provider is no longer needed
        to ensure proper shutdown of background threads.

        Example:
            ```python
            provider = DockerDeploymentProvider()
            try:
                # Use provider...
                pass
            finally:
                await provider.cleanup()
            ```
        """
        logger.info("Shutting down deployment provider executor")
        self._executor.shutdown(wait=True)
