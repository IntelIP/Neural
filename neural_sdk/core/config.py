"""
Configuration management for the Neural SDK.

This module provides configuration classes and utilities for managing
SDK settings, environment variables, and validation.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .exceptions import ConfigurationError, ValidationError


@dataclass
class RiskLimits:
    """Risk management limits and thresholds."""

    max_position_size_pct: float = 0.05  # Max 5% of portfolio per position
    max_daily_loss_pct: float = 0.20  # Max 20% daily loss
    max_correlation: float = 0.7  # Max correlation between positions
    stop_loss_pct: float = 0.10  # Default 10% stop-loss
    take_profit_pct: float = 0.30  # Default 30% take-profit
    kelly_fraction: float = 0.25  # Use 25% of Kelly criterion

    def __post_init__(self):
        """Validate risk limits."""
        if not 0 < self.max_position_size_pct <= 1:
            raise ValidationError(
                f"max_position_size_pct must be between 0 and 1, got {self.max_position_size_pct}"
            )
        if not 0 < self.max_daily_loss_pct <= 1:
            raise ValidationError(
                f"max_daily_loss_pct must be between 0 and 1, got {self.max_daily_loss_pct}"
            )
        if not -1 <= self.max_correlation <= 1:
            raise ValidationError(
                f"max_correlation must be between -1 and 1, got {self.max_correlation}"
            )
        if not 0 < self.kelly_fraction <= 1:
            raise ValidationError(
                f"kelly_fraction must be between 0 and 1, got {self.kelly_fraction}"
            )


@dataclass
class TradingConfig:
    """Trading-specific configuration."""

    default_quantity: int = 100
    max_order_size: int = 1000
    min_order_size: int = 10
    enable_paper_trading: bool = True
    enable_live_trading: bool = False
    auto_cancel_stale_orders: bool = True
    stale_order_timeout_seconds: int = 300  # 5 minutes


@dataclass
class DataConfig:
    """Data source configuration."""

    redis_url: str = "redis://localhost:6379"
    enable_stream: bool = True
    enable_espn_stream: bool = True
    enable_twitter_stream: bool = False
    enable_weather_stream: bool = False
    stream_buffer_size: int = 1000
    data_retention_days: int = 30


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    enable_file_logging: bool = False
    log_file_path: Optional[str] = None
    enable_structured_logging: bool = True
    log_performance_metrics: bool = True


@dataclass
class SDKConfig:
    """
    Main configuration class for the Neural SDK.

    This class centralizes all configuration options and provides
    validation and environment variable loading.
    """

    # API Credentials
    api_key_id: Optional[str] = None
    api_secret: Optional[str] = None

    # Core Settings
    environment: str = "development"  # development, staging, production
    log_level: str = "INFO"

    # Component Configurations
    risk_limits: RiskLimits = field(default_factory=RiskLimits)
    trading: TradingConfig = field(default_factory=TradingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Advanced Settings
    enable_monitoring: bool = True
    enable_metrics: bool = True
    enable_debug_mode: bool = False
    config_file_path: Optional[str] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_configuration()

    def _validate_configuration(self):
        """Validate the entire configuration."""
        valid_environments = ["development", "staging", "production"]
        if self.environment not in valid_environments:
            raise ConfigurationError(
                f"Invalid environment '{self.environment}'. Must be one of: {valid_environments}"
            )

        # Validate API credentials for non-development environments
        if self.environment != "development":
            if not self.api_key_id or not self.api_secret:
                raise ConfigurationError(
                    "API credentials required for non-development environments"
                )

        # Validate trading configuration
        if self.trading.max_order_size < self.trading.min_order_size:
            raise ConfigurationError(
                "max_order_size must be greater than min_order_size"
            )

    @classmethod
    def from_env(cls, prefix: str = "KALSHI_") -> "SDKConfig":
        """
        Create configuration from environment variables.

        Args:
            prefix: Environment variable prefix (default: "KALSHI_")

        Returns:
            SDKConfig instance with values from environment

        Example:
            >>> config = SDKConfig.from_env()
            >>> # Reads KALSHI_KALSHI_API_KEY_ID, etc.
        """
        env_data = {}

        # API Credentials
        env_data["api_key_id"] = os.getenv(f"{prefix}API_KEY_ID")
        env_data["api_secret"] = os.getenv(f"{prefix}API_SECRET")

        # Core Settings
        env_data["environment"] = os.getenv(f"{prefix}ENVIRONMENT", "development")
        env_data["log_level"] = os.getenv(f"{prefix}LOG_LEVEL", "INFO")

        # Risk Limits
        risk_limits = {}
        risk_limits["max_position_size_pct"] = float(
            os.getenv(f"{prefix}MAX_POSITION_SIZE_PCT", "0.05")
        )
        risk_limits["max_daily_loss_pct"] = float(
            os.getenv(f"{prefix}MAX_DAILY_LOSS_PCT", "0.20")
        )
        risk_limits["max_correlation"] = float(
            os.getenv(f"{prefix}MAX_CORRELATION", "0.7")
        )
        risk_limits["stop_loss_pct"] = float(
            os.getenv(f"{prefix}STOP_LOSS_PCT", "0.10")
        )
        risk_limits["take_profit_pct"] = float(
            os.getenv(f"{prefix}TAKE_PROFIT_PCT", "0.30")
        )
        risk_limits["kelly_fraction"] = float(
            os.getenv(f"{prefix}KELLY_FRACTION", "0.25")
        )
        env_data["risk_limits"] = RiskLimits(**risk_limits)

        # Data Configuration
        data_config = {}
        data_config["redis_url"] = os.getenv(
            f"{prefix}REDIS_URL", "redis://localhost:6379"
        )
        data_config["enable_stream"] = (
            os.getenv(f"{prefix}ENABLE_STREAM", "true").lower() == "true"
        )
        data_config["enable_espn_stream"] = (
            os.getenv(f"{prefix}ENABLE_ESPN_STREAM", "true").lower() == "true"
        )
        data_config["enable_twitter_stream"] = (
            os.getenv(f"{prefix}ENABLE_TWITTER_STREAM", "false").lower() == "true"
        )
        data_config["enable_weather_stream"] = (
            os.getenv(f"{prefix}ENABLE_WEATHER_STREAM", "false").lower() == "true"
        )
        data_config["stream_buffer_size"] = int(
            os.getenv(f"{prefix}STREAM_BUFFER_SIZE", "1000")
        )
        data_config["data_retention_days"] = int(
            os.getenv(f"{prefix}DATA_RETENTION_DAYS", "30")
        )
        env_data["data"] = DataConfig(**data_config)

        # Trading Configuration
        trading_config = {}
        trading_config["default_quantity"] = int(
            os.getenv(f"{prefix}DEFAULT_QUANTITY", "100")
        )
        trading_config["max_order_size"] = int(
            os.getenv(f"{prefix}MAX_ORDER_SIZE", "1000")
        )
        trading_config["min_order_size"] = int(
            os.getenv(f"{prefix}MIN_ORDER_SIZE", "10")
        )
        trading_config["enable_paper_trading"] = (
            os.getenv(f"{prefix}ENABLE_PAPER_TRADING", "true").lower() == "true"
        )
        trading_config["enable_live_trading"] = (
            os.getenv(f"{prefix}ENABLE_LIVE_TRADING", "false").lower() == "true"
        )
        trading_config["auto_cancel_stale_orders"] = (
            os.getenv(f"{prefix}AUTO_CANCEL_STALE_ORDERS", "true").lower() == "true"
        )
        trading_config["stale_order_timeout_seconds"] = int(
            os.getenv(f"{prefix}STALE_ORDER_TIMEOUT", "300")
        )
        env_data["trading"] = TradingConfig(**trading_config)

        # Logging Configuration
        logging_config = {}
        logging_config["level"] = os.getenv(f"{prefix}LOG_LEVEL", "INFO")
        logging_config["enable_file_logging"] = (
            os.getenv(f"{prefix}ENABLE_FILE_LOGGING", "false").lower() == "true"
        )
        logging_config["log_file_path"] = os.getenv(f"{prefix}LOG_FILE_PATH")
        logging_config["enable_structured_logging"] = (
            os.getenv(f"{prefix}ENABLE_STRUCTURED_LOGGING", "true").lower() == "true"
        )
        logging_config["log_performance_metrics"] = (
            os.getenv(f"{prefix}LOG_PERFORMANCE_METRICS", "true").lower() == "true"
        )
        env_data["logging"] = LoggingConfig(**logging_config)

        # Advanced Settings
        env_data["enable_monitoring"] = (
            os.getenv(f"{prefix}ENABLE_MONITORING", "true").lower() == "true"
        )
        env_data["enable_metrics"] = (
            os.getenv(f"{prefix}ENABLE_METRICS", "true").lower() == "true"
        )
        env_data["enable_debug_mode"] = (
            os.getenv(f"{prefix}ENABLE_DEBUG_MODE", "false").lower() == "true"
        )
        env_data["config_file_path"] = os.getenv(f"{prefix}CONFIG_FILE_PATH")

        return cls(**env_data)

    @classmethod
    def from_file(cls, file_path: str) -> "SDKConfig":
        """
        Create configuration from a YAML file.

        Args:
            file_path: Path to YAML configuration file

        Returns:
            SDKConfig instance with values from file
        """
        try:
            import yaml
        except ImportError:
            raise ConfigurationError(
                "PyYAML is required for file-based configuration. Install with: pip install PyYAML"
            )

        path = Path(file_path)
        if not path.exists():
            raise ConfigurationError(f"Configuration file not found: {file_path}")

        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            raise ConfigurationError(f"Failed to parse configuration file: {e}")

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SDKConfig":
        """
        Create configuration from a dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            SDKConfig instance
        """
        # Handle nested configurations
        if "risk_limits" in data:
            data["risk_limits"] = RiskLimits(**data["risk_limits"])
        if "trading" in data:
            data["trading"] = TradingConfig(**data["trading"])
        if "data" in data:
            data["data"] = DataConfig(**data["data"])
        if "logging" in data:
            data["logging"] = LoggingConfig(**data["logging"])

        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "api_key_id": self.api_key_id,
            "api_secret": self.api_secret,
            "environment": self.environment,
            "log_level": self.log_level,
            "risk_limits": {
                "max_position_size_pct": self.risk_limits.max_position_size_pct,
                "max_daily_loss_pct": self.risk_limits.max_daily_loss_pct,
                "max_correlation": self.risk_limits.max_correlation,
                "stop_loss_pct": self.risk_limits.stop_loss_pct,
                "take_profit_pct": self.risk_limits.take_profit_pct,
                "kelly_fraction": self.risk_limits.kelly_fraction,
            },
            "trading": {
                "default_quantity": self.trading.default_quantity,
                "max_order_size": self.trading.max_order_size,
                "min_order_size": self.trading.min_order_size,
                "enable_paper_trading": self.trading.enable_paper_trading,
                "enable_live_trading": self.trading.enable_live_trading,
                "auto_cancel_stale_orders": self.trading.auto_cancel_stale_orders,
                "stale_order_timeout_seconds": self.trading.stale_order_timeout_seconds,
            },
            "data": {
                "redis_url": self.data.redis_url,
                "enable_stream": self.data.enable_stream,
                "enable_espn_stream": self.data.enable_espn_stream,
                "enable_twitter_stream": self.data.enable_twitter_stream,
                "enable_weather_stream": self.data.enable_weather_stream,
                "stream_buffer_size": self.data.stream_buffer_size,
                "data_retention_days": self.data.data_retention_days,
            },
            "logging": {
                "level": self.logging.level,
                "format": self.logging.format,
                "enable_file_logging": self.logging.enable_file_logging,
                "log_file_path": self.logging.log_file_path,
                "enable_structured_logging": self.logging.enable_structured_logging,
                "log_performance_metrics": self.logging.log_performance_metrics,
            },
            "enable_monitoring": self.enable_monitoring,
            "enable_metrics": self.enable_metrics,
            "enable_debug_mode": self.enable_debug_mode,
            "config_file_path": self.config_file_path,
        }

    def save_to_file(self, file_path: str):
        """Save configuration to a YAML file."""
        try:
            import yaml
        except ImportError:
            raise ConfigurationError(
                "PyYAML is required to save configuration. Install with: pip install PyYAML"
            )

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(path, "w") as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration file: {e}")

    def is_production_ready(self) -> bool:
        """
        Check if configuration is ready for production use.

        Returns:
            True if configuration is production-ready
        """
        checks = [
            self.api_key_id is not None,
            self.api_secret is not None,
            self.environment == "production",
            self.trading.enable_live_trading,
            not self.enable_debug_mode,
        ]
        return all(checks)

    def get_summary(self) -> str:
        """Get a human-readable summary of the configuration."""
        return f"""
Neural SDK Configuration
================================
Environment: {self.environment}
Log Level: {self.log_level}

API Credentials: {'✅ Configured' if self.api_key_id else '❌ Missing'}

Risk Limits:
  - Max Position Size: {self.risk_limits.max_position_size_pct:.1%}
  - Max Daily Loss: {self.risk_limits.max_daily_loss_pct:.1%}
  - Kelly Fraction: {self.risk_limits.kelly_fraction:.1%}

Trading:
  - Paper Trading: {'✅ Enabled' if self.trading.enable_paper_trading else '❌ Disabled'}
  - Live Trading: {'✅ Enabled' if self.trading.enable_live_trading else '❌ Disabled'}
  - Default Quantity: {self.trading.default_quantity}

Data Sources:
  - Redis: {self.data.redis_url}
  - Stream: {'✅ Enabled' if self.data.enable_stream else '❌ Disabled'}
  - ESPN Stream: {'✅ Enabled' if self.data.enable_espn_stream else '❌ Disabled'}

Production Ready: {'✅ Yes' if self.is_production_ready() else '❌ No'}
        """.strip()
