"""
Configuration models for the Neural SDK deployment module.

This module provides Pydantic models for configuring trading bot deployments,
including Docker containers, resource limits, and deployment parameters.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class DeploymentConfig(BaseModel):
    """Configuration for trading bot deployment.

    This model defines all parameters needed to deploy a trading bot,
    including resource limits, environment settings, and feature flags.

    Attributes:
        bot_name: Unique name for the trading bot
        strategy_type: Type of trading strategy (e.g., "NFL", "NBA")
        risk_config: Risk management configuration dict
        algorithm_config: Algorithm-specific parameters
        environment: Deployment environment ("sandbox", "paper", or "live")
        compute_resources: CPU and memory resource limits
        database_enabled: Whether to enable database persistence
        websocket_enabled: Whether to enable WebSocket trading
        monitoring_enabled: Whether to enable performance monitoring
    """

    bot_name: str = Field(..., description="Name of the trading bot")
    strategy_type: str = Field(..., description="Type of trading strategy (NFL, NBA, etc.)")
    risk_config: dict[str, Any] = Field(
        default_factory=dict, description="Risk management configuration"
    )
    algorithm_config: dict[str, Any] = Field(
        default_factory=dict, description="Algorithm parameters"
    )
    environment: str = Field(default="sandbox", description="Deployment environment")
    compute_resources: dict[str, Any] = Field(
        default_factory=lambda: {
            "cpu_limit": "1.0",
            "memory_limit": "2g",
            "cpu_request": "0.5",
            "memory_request": "1g",
        },
        description="Docker resource limits",
    )
    database_enabled: bool = Field(default=True, description="Enable database persistence")
    websocket_enabled: bool = Field(default=True, description="Enable WebSocket trading")
    monitoring_enabled: bool = Field(default=True, description="Enable performance monitoring")

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment is one of the allowed values."""
        allowed = ["sandbox", "paper", "live"]
        if v.lower() not in allowed:
            raise ValueError(f"Environment must be one of {allowed}, got: {v}")
        return v.lower()


class DockerConfig(BaseModel):
    """Docker-specific configuration for container deployments.

    Attributes:
        image_name: Name of the Docker image to use
        image_tag: Tag/version of the Docker image
        cpu_limit: Maximum CPU cores (e.g., 1.0 = 1 core)
        memory_limit: Maximum memory (e.g., "2g" = 2 gigabytes)
        cpu_request: Requested CPU cores (for scheduling)
        memory_request: Requested memory (for scheduling)
        restart_policy: Container restart policy
        network_mode: Docker network mode
        labels: Container labels for organization
    """

    image_name: str = Field(default="neural-trading-bot", description="Docker image name")
    image_tag: str = Field(default="latest", description="Docker image tag")
    cpu_limit: float = Field(default=1.0, description="CPU core limit", ge=0.1, le=16.0)
    memory_limit: str = Field(default="2g", description="Memory limit (e.g., '2g', '512m')")
    cpu_request: float = Field(default=0.5, description="Requested CPU cores", ge=0.1, le=16.0)
    memory_request: str = Field(default="1g", description="Requested memory")
    restart_policy: str = Field(default="unless-stopped", description="Container restart policy")
    network_mode: str = Field(default="bridge", description="Docker network mode")
    labels: dict[str, str] = Field(default_factory=dict, description="Container labels")

    @field_validator("restart_policy")
    @classmethod
    def validate_restart_policy(cls, v: str) -> str:
        """Validate restart policy is one of Docker's allowed values."""
        allowed = ["no", "always", "on-failure", "unless-stopped"]
        if v not in allowed:
            raise ValueError(f"Restart policy must be one of {allowed}, got: {v}")
        return v

    @field_validator("network_mode")
    @classmethod
    def validate_network_mode(cls, v: str) -> str:
        """Validate network mode is one of Docker's allowed values."""
        allowed = ["bridge", "host", "none", "container"]
        if v not in allowed and not v.startswith("container:"):
            raise ValueError(
                f"Network mode must be one of {allowed} or 'container:<name>', got: {v}"
            )
        return v


class DeploymentResult(BaseModel):
    """Result of a deployment operation.

    Attributes:
        deployment_id: Unique identifier for the deployment
        status: Current deployment status
        container_id: Docker container ID (if applicable)
        container_name: Human-readable container name
        created_at: Timestamp when deployment was created
        endpoints: Dict of service endpoints (e.g., {"api": "http://localhost:8000"})
        metadata: Additional deployment metadata
    """

    deployment_id: str = Field(..., description="Unique deployment identifier")
    status: str = Field(..., description="Deployment status")
    container_id: str | None = Field(None, description="Docker container ID")
    container_name: str | None = Field(None, description="Container name")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    endpoints: dict[str, str] = Field(default_factory=dict, description="Service endpoints")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class DeploymentStatus(BaseModel):
    """Current status of a deployment.

    Attributes:
        deployment_id: Unique identifier for the deployment
        status: Current status ("running", "stopped", "error", "starting")
        uptime_seconds: Number of seconds the deployment has been running
        logs: Recent log lines from the deployment
        health_status: Health check status (if available)
        metrics: Current resource metrics (CPU, memory, etc.)
    """

    deployment_id: str = Field(..., description="Unique deployment identifier")
    status: str = Field(..., description="Current status")
    uptime_seconds: float | None = Field(None, description="Uptime in seconds")
    logs: list[str] = Field(default_factory=list, description="Recent log lines")
    health_status: str | None = Field(None, description="Health check status")
    metrics: dict[str, Any] = Field(default_factory=dict, description="Current metrics")


class DeploymentInfo(BaseModel):
    """Summary information about a deployment.

    Used for listing multiple deployments.

    Attributes:
        deployment_id: Unique identifier for the deployment
        bot_name: Name of the trading bot
        status: Current deployment status
        environment: Deployment environment (sandbox/paper/live)
        created_at: When the deployment was created
        deployment_type: Type of deployment (docker, compose, etc.)
    """

    deployment_id: str = Field(..., description="Unique deployment identifier")
    bot_name: str = Field(..., description="Trading bot name")
    status: str = Field(..., description="Current status")
    environment: str = Field(..., description="Deployment environment")
    created_at: datetime = Field(..., description="Creation timestamp")
    deployment_type: str = Field(..., description="Deployment type")


class DatabaseConfig(BaseModel):
    """Database configuration for deployment persistence.

    Attributes:
        host: Database host address
        port: Database port number
        user: Database username
        password: Database password
        database: Database name
        connection_pool_size: Size of the connection pool
        max_overflow: Maximum overflow connections
    """

    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port", ge=1, le=65535)
    user: str = Field(default="trading_user", description="Database username")
    password: str = Field(default="", description="Database password")
    database: str = Field(default="trading_db", description="Database name")
    connection_pool_size: int = Field(default=5, description="Connection pool size", ge=1)
    max_overflow: int = Field(default=10, description="Max overflow connections", ge=0)


class MonitoringConfig(BaseModel):
    """Monitoring configuration for deployment metrics.

    Attributes:
        enabled: Whether monitoring is enabled
        collection_interval: Seconds between metric collections
        metrics_port: Port for exposing metrics
        log_level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
        export_prometheus: Whether to export Prometheus metrics
    """

    enabled: bool = Field(default=True, description="Enable monitoring")
    collection_interval: int = Field(
        default=60, description="Metric collection interval (seconds)", ge=1
    )
    metrics_port: int = Field(default=9090, description="Metrics export port", ge=1024, le=65535)
    log_level: str = Field(default="INFO", description="Logging level")
    export_prometheus: bool = Field(default=False, description="Export Prometheus metrics")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is one of the standard logging levels."""
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed:
            raise ValueError(f"Log level must be one of {allowed}, got: {v}")
        return v.upper()
