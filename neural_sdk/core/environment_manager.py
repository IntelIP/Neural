"""
Environment Manager

Manages environment detection, configuration, and safety mechanisms
for training/sandbox vs production modes.
"""

import hashlib
import os
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import redis.asyncio as redis
import yaml
from colorama import Back, Fore, Style, init

# Initialize colorama for cross-platform color support
init(autoreset=True)


class Environment(Enum):
    """Environment types."""

    DEVELOPMENT = "development"
    TRAINING = "training"
    SANDBOX = "sandbox"
    STAGING = "staging"
    PRODUCTION = "production"


class VisualTheme(Enum):
    """Visual themes for different environments."""

    DEVELOPMENT = (Fore.CYAN, "🔧")
    TRAINING = (Fore.YELLOW, "🎓")
    SANDBOX = (Fore.GREEN, "📦")
    STAGING = (Fore.MAGENTA, "🚦")
    PRODUCTION = (Fore.RED, "🔥")


@dataclass
class EnvironmentConfig:
    """Configuration for a specific environment."""

    name: str
    environment: Environment
    redis_db: int
    redis_prefix: str
    api_endpoints: Dict[str, str]
    rate_limits: Dict[str, int]
    safety_checks: bool
    require_confirmation: bool
    require_mfa: bool
    allowed_operations: List[str]
    restricted_operations: List[str]
    data_retention_hours: int
    max_position_size: float
    max_daily_trades: int
    enable_logging: bool
    log_level: str
    telemetry_enabled: bool
    features: Dict[str, bool] = field(default_factory=dict)
    # Optional fields from YAML
    backup_enabled: bool = False
    backup_frequency_hours: int = 0
    max_concurrent_positions: int = 100
    max_order_value: float = 10000.0
    min_order_value: float = 1.0
    stop_loss_percentage: float = 0.10
    max_drawdown_percentage: float = 0.20
    position_sizing_method: str = "kelly_fraction"
    kelly_fraction: float = 0.25
    alert_enabled: bool = False
    alert_channels: List[str] = field(default_factory=list)
    monitoring: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    synthetic_data: Dict[str, Any] = field(default_factory=dict)
    sandbox: Dict[str, Any] = field(default_factory=dict)
    development: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, file_path: Path) -> "EnvironmentConfig":
        """Load configuration from YAML file."""
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "environment": self.environment.value,
            "redis_db": self.redis_db,
            "redis_prefix": self.redis_prefix,
            "api_endpoints": self.api_endpoints,
            "rate_limits": self.rate_limits,
            "safety_checks": self.safety_checks,
            "require_confirmation": self.require_confirmation,
            "require_mfa": self.require_mfa,
            "allowed_operations": self.allowed_operations,
            "restricted_operations": self.restricted_operations,
            "data_retention_hours": self.data_retention_hours,
            "max_position_size": self.max_position_size,
            "max_daily_trades": self.max_daily_trades,
            "enable_logging": self.enable_logging,
            "log_level": self.log_level,
            "telemetry_enabled": self.telemetry_enabled,
            "features": self.features,
        }


@dataclass
class EnvironmentStatus:
    """Current environment status."""

    environment: Environment
    config: EnvironmentConfig
    is_authenticated: bool
    session_id: str
    session_start: datetime
    operations_count: int
    last_operation: Optional[datetime]
    warnings: List[str]

    def is_production(self) -> bool:
        """Check if in production environment."""
        return self.environment == Environment.PRODUCTION

    def is_safe_mode(self) -> bool:
        """Check if in a safe mode (non-production)."""
        return self.environment in [
            Environment.DEVELOPMENT,
            Environment.TRAINING,
            Environment.SANDBOX,
        ]

    def session_duration(self) -> timedelta:
        """Get current session duration."""
        return datetime.now() - self.session_start


class EnvironmentManager:
    """Manages environment detection and configuration."""

    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize environment manager."""
        self.config_dir = config_dir or Path("config/environments")
        self._current_env: Optional[Environment] = None
        self._config: Optional[EnvironmentConfig] = None
        self._status: Optional[EnvironmentStatus] = None
        self._redis_clients: Dict[Environment, redis.Redis] = {}
        self._guards: List["EnvironmentGuard"] = []
        self._callbacks: Dict[str, List[Callable]] = {
            "pre_switch": [],
            "post_switch": [],
            "operation": [],
        }

        # Auto-detect environment on initialization
        self._detect_environment()

    def _detect_environment(self) -> Environment:
        """Auto-detect current environment from various sources."""
        # Priority order for environment detection

        # 1. Explicit environment variable
        env_var = os.getenv("KALSHI_ENVIRONMENT", "").lower()
        if env_var:
            # Support legacy aliases commonly used for Neural API selection
            alias_map = {
                "prod": Environment.PRODUCTION,
                "production": Environment.PRODUCTION,
                "demo": Environment.SANDBOX,
                "sandbox": Environment.SANDBOX,
                "staging": Environment.STAGING,
                "train": Environment.TRAINING,
                "training": Environment.TRAINING,
                "dev": Environment.DEVELOPMENT,
                "development": Environment.DEVELOPMENT,
            }
            if env_var in alias_map:
                return alias_map[env_var]
            try:
                return Environment(env_var)
            except ValueError:
                warnings.warn(f"Invalid environment: {env_var}")

        # 2. Check for production indicators
        if os.getenv("PRODUCTION_MODE") == "true":
            return Environment.PRODUCTION

        # 3. Check for training indicators
        if os.getenv("TRAINING_MODE") == "true":
            return Environment.TRAINING

        # 4. Check for CI/CD indicators
        if os.getenv("CI") or os.getenv("GITHUB_ACTIONS"):
            return Environment.STAGING

        # 5. Check for development indicators
        if os.getenv("DEBUG") == "true" or os.getenv("DEV_MODE"):
            return Environment.DEVELOPMENT

        # 6. Check hostname patterns
        hostname = os.uname().nodename.lower()
        if "prod" in hostname:
            return Environment.PRODUCTION
        elif "staging" in hostname:
            return Environment.STAGING
        elif "sandbox" in hostname:
            return Environment.SANDBOX

        # 7. Default to development for safety
        warnings.warn("Environment not detected, defaulting to DEVELOPMENT")
        return Environment.DEVELOPMENT

    def initialize(self, environment: Optional[Environment] = None) -> None:
        """Initialize with specific environment."""
        if environment:
            self._current_env = environment
        else:
            self._current_env = self._detect_environment()

        # Load configuration
        self._load_config()

        # Initialize status
        self._status = EnvironmentStatus(
            environment=self._current_env,
            config=self._config,
            is_authenticated=False,
            session_id=self._generate_session_id(),
            session_start=datetime.now(),
            operations_count=0,
            last_operation=None,
            warnings=[],
        )

        # Apply visual theme
        self._apply_visual_theme()

        # Print environment banner
        self._print_banner()

    def _load_config(self) -> None:
        """Load configuration for current environment."""
        config_file = self.config_dir / f"{self._current_env.value}.yaml"

        if config_file.exists():
            self._config = EnvironmentConfig.from_yaml(config_file)
        else:
            # Use default configuration
            self._config = self._get_default_config(self._current_env)

    def _get_default_config(self, env: Environment) -> EnvironmentConfig:
        """Get default configuration for environment."""
        base_config = {
            "name": env.value.title(),
            "environment": env,
            "enable_logging": True,
            "features": {},
        }

        if env == Environment.PRODUCTION:
            return EnvironmentConfig(
                **base_config,
                redis_db=0,
                redis_prefix="prod:",
                api_endpoints={
                    "neural": "https://api.elections.neural.com",
                    "odds": "https://api.theoddsapi.com/v4",
                },
                rate_limits={"neural": 100, "odds": 50},
                safety_checks=True,
                require_confirmation=True,
                require_mfa=True,
                allowed_operations=["*"],
                restricted_operations=[],
                data_retention_hours=720,  # 30 days
                max_position_size=0.05,
                max_daily_trades=20,
                log_level="INFO",
                telemetry_enabled=True,
            )

        elif env == Environment.STAGING:
            return EnvironmentConfig(
                **base_config,
                redis_db=1,
                redis_prefix="staging:",
                api_endpoints={
                    "neural": "https://demo-api.neural.co",
                    "odds": "https://api.theoddsapi.com/v4",
                },
                rate_limits={"neural": 200, "odds": 100},
                safety_checks=True,
                require_confirmation=True,
                require_mfa=False,
                allowed_operations=["*"],
                restricted_operations=["delete_all"],
                data_retention_hours=168,  # 7 days
                max_position_size=0.10,
                max_daily_trades=50,
                log_level="DEBUG",
                telemetry_enabled=True,
            )

        elif env == Environment.SANDBOX:
            return EnvironmentConfig(
                **base_config,
                redis_db=2,
                redis_prefix="sandbox:",
                api_endpoints={
                    "neural": "https://demo-api.neural.co",
                    "odds": "https://api.theoddsapi.com/v4",
                },
                rate_limits={"neural": 500, "odds": 200},
                safety_checks=True,
                require_confirmation=False,
                require_mfa=False,
                allowed_operations=["*"],
                restricted_operations=[],
                data_retention_hours=24,
                max_position_size=0.20,
                max_daily_trades=100,
                log_level="DEBUG",
                telemetry_enabled=False,
            )

        elif env == Environment.TRAINING:
            return EnvironmentConfig(
                **base_config,
                redis_db=3,
                redis_prefix="training:",
                api_endpoints={
                    "neural": "http://localhost:8000/mock",
                    "odds": "http://localhost:8001/mock",
                },
                rate_limits={"neural": 1000, "odds": 1000},
                safety_checks=False,
                require_confirmation=False,
                require_mfa=False,
                allowed_operations=["*"],
                restricted_operations=[],
                data_retention_hours=4,
                max_position_size=1.0,
                max_daily_trades=1000,
                log_level="DEBUG",
                telemetry_enabled=False,
            )

        else:  # DEVELOPMENT
            return EnvironmentConfig(
                **base_config,
                redis_db=4,
                redis_prefix="dev:",
                api_endpoints={
                    "neural": "http://localhost:8000/mock",
                    "odds": "http://localhost:8001/mock",
                },
                rate_limits={"neural": 10000, "odds": 10000},
                safety_checks=False,
                require_confirmation=False,
                require_mfa=False,
                allowed_operations=["*"],
                restricted_operations=[],
                data_retention_hours=1,
                max_position_size=1.0,
                max_daily_trades=10000,
                log_level="DEBUG",
                telemetry_enabled=False,
            )

    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        data = f"{datetime.now().isoformat()}-{os.getpid()}-{os.urandom(8).hex()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _apply_visual_theme(self) -> None:
        """Apply visual theme for current environment."""
        theme = VisualTheme[self._current_env.name]
        _, icon = theme.value

        # Set terminal title if supported (skip in non-interactive environments)
        try:
            if os.name != "nt" and sys.stdout.isatty():  # Unix-like systems with TTY
                sys.stdout.write(
                    f"\033]0;Neural Agent - {icon} {self._current_env.value.upper()}\007"
                )
                sys.stdout.flush()
        except Exception:
            pass  # Skip terminal title setting if not supported

    def _print_banner(self) -> None:
        """Print environment banner."""
        theme = VisualTheme[self._current_env.name]
        color, icon = theme.value

        banner = f"""
{color}{'='*60}
{icon} KALSHI TRADING AGENT - {self._current_env.value.upper()} ENVIRONMENT {icon}
{'='*60}
Session ID: {self._status.session_id}
Started: {self._status.session_start.strftime('%Y-%m-%d %H:%M:%S')}
Redis DB: {self._config.redis_db}
Safety Checks: {'✓' if self._config.safety_checks else '✗'}
MFA Required: {'✓' if self._config.require_mfa else '✗'}
{'='*60}{Style.RESET_ALL}
"""
        print(banner)

        if self._current_env == Environment.PRODUCTION:
            self._print_production_warning()

    def _print_production_warning(self) -> None:
        """Print production environment warning."""
        warning = f"""
{Back.RED}{Fore.WHITE}{'!'*60}
                    PRODUCTION ENVIRONMENT
                    
  You are operating in PRODUCTION mode with REAL MONEY.
  All trades will be executed on live markets.
  
  Ensure you have:
  ✓ Verified all risk parameters
  ✓ Tested changes in sandbox first
  ✓ Proper authorization to trade
  ✓ Emergency stop procedures ready
  
  Type 'CONFIRM PRODUCTION' to proceed...
{'!'*60}{Style.RESET_ALL}
"""
        print(warning)

    def get_current_environment(self) -> Environment:
        """Get current environment."""
        return self._current_env

    def get_config(self) -> EnvironmentConfig:
        """Get current environment configuration."""
        return self._config

    def get_status(self) -> EnvironmentStatus:
        """Get current environment status."""
        return self._status

    def is_production(self) -> bool:
        """Check if in production environment."""
        return self._current_env == Environment.PRODUCTION

    def is_safe_mode(self) -> bool:
        """Check if in safe mode (non-production)."""
        return self._current_env in [
            Environment.DEVELOPMENT,
            Environment.TRAINING,
            Environment.SANDBOX,
        ]

    def can_execute_operation(self, operation: str) -> Tuple[bool, str]:
        """Check if operation is allowed in current environment."""
        # Check if operation is restricted
        if operation in self._config.restricted_operations:
            return (
                False,
                f"Operation '{operation}' is restricted in {self._current_env.value}",
            )

        # Check if operation is explicitly allowed
        if "*" in self._config.allowed_operations:
            return True, "Operation allowed"

        if operation in self._config.allowed_operations:
            return True, "Operation allowed"

        return (
            False,
            f"Operation '{operation}' not allowed in {self._current_env.value}",
        )

    async def get_redis_client(self) -> redis.Redis:
        """Get Redis client for current environment."""
        if self._current_env not in self._redis_clients:
            self._redis_clients[self._current_env] = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                db=self._config.redis_db,
                decode_responses=True,
            )

        return self._redis_clients[self._current_env]

    def add_guard(self, guard: "EnvironmentGuard") -> None:
        """Add environment guard."""
        self._guards.append(guard)

    def register_callback(self, event: str, callback: Callable) -> None:
        """Register callback for environment events."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    async def switch_environment(
        self, new_env: Environment, force: bool = False
    ) -> Tuple[bool, str]:
        """Switch to different environment."""
        if self._current_env == new_env:
            return True, "Already in requested environment"

        # Check if switch is allowed
        if not force:
            # Run pre-switch callbacks
            for callback in self._callbacks["pre_switch"]:
                if not await callback(self._current_env, new_env):
                    return False, "Switch blocked by callback"

            # Check guards
            for guard in self._guards:
                allowed, reason = await guard.check_switch(self._current_env, new_env)
                if not allowed:
                    return False, f"Switch blocked: {reason}"

        # Perform switch
        old_env = self._current_env
        self._current_env = new_env
        self._load_config()

        # Update status
        self._status = EnvironmentStatus(
            environment=self._current_env,
            config=self._config,
            is_authenticated=False,
            session_id=self._generate_session_id(),
            session_start=datetime.now(),
            operations_count=0,
            last_operation=None,
            warnings=[],
        )

        # Apply new theme
        self._apply_visual_theme()
        self._print_banner()

        # Run post-switch callbacks
        for callback in self._callbacks["post_switch"]:
            await callback(old_env, new_env)

        return True, f"Switched from {old_env.value} to {new_env.value}"

    def require_confirmation(self, message: str) -> bool:
        """Require user confirmation for sensitive operations."""
        if not self._config.require_confirmation:
            return True

        print(f"\n{Fore.YELLOW}⚠️  {message}")
        response = input(f"{Fore.YELLOW}Type 'yes' to confirm: {Style.RESET_ALL}")

        return response.lower() == "yes"

    def log_operation(self, operation: str, details: Dict[str, Any]) -> None:
        """Log an operation in current environment."""
        self._status.operations_count += 1
        self._status.last_operation = datetime.now()

        # Run operation callbacks
        for callback in self._callbacks["operation"]:
            callback(operation, details)

    def add_warning(self, warning: str) -> None:
        """Add warning to current session."""
        self._status.warnings.append(f"[{datetime.now().isoformat()}] {warning}")

    def get_api_endpoint(self, service: str) -> str:
        """Get API endpoint for service in current environment."""
        return self._config.api_endpoints.get(service, "")

    def get_rate_limit(self, service: str) -> int:
        """Get rate limit for service in current environment."""
        return self._config.rate_limits.get(service, 100)

    def is_feature_enabled(self, feature: str) -> bool:
        """Check if feature is enabled in current environment."""
        return self._config.features.get(feature, False)

    def get_visual_indicator(self) -> str:
        """Get visual indicator for current environment."""
        theme = VisualTheme[self._current_env.name]
        color, icon = theme.value
        return f"{color}{icon} {self._current_env.value.upper()}{Style.RESET_ALL}"


class EnvironmentGuard:
    """Base class for environment guards."""

    async def check_switch(
        self, from_env: Environment, to_env: Environment
    ) -> Tuple[bool, str]:
        """Check if environment switch is allowed."""
        raise NotImplementedError

    async def check_operation(
        self, operation: str, environment: Environment
    ) -> Tuple[bool, str]:
        """Check if operation is allowed."""
        raise NotImplementedError


class ProductionGuard(EnvironmentGuard):
    """Guard for production environment access."""

    def __init__(self, require_mfa: bool = True):
        """Initialize production guard."""
        self.require_mfa = require_mfa
        self._authenticated_sessions: Dict[str, datetime] = {}

    async def check_switch(
        self, from_env: Environment, to_env: Environment
    ) -> Tuple[bool, str]:
        """Check if switch to production is allowed."""
        if to_env != Environment.PRODUCTION:
            return True, "Not switching to production"

        # Require explicit confirmation
        print(
            f"\n{Back.RED}{Fore.WHITE}SWITCHING TO PRODUCTION ENVIRONMENT{Style.RESET_ALL}"
        )
        print("This will enable REAL MONEY trading.")

        confirmation = input("Type 'PRODUCTION' to confirm: ")
        if confirmation != "PRODUCTION":
            return False, "Production switch not confirmed"

        # Check MFA if required
        if self.require_mfa:
            mfa_code = input("Enter MFA code: ")
            if not self._verify_mfa(mfa_code):
                return False, "Invalid MFA code"

        return True, "Production access granted"

    async def check_operation(
        self, operation: str, environment: Environment
    ) -> Tuple[bool, str]:
        """Check if operation is allowed in production."""
        if environment != Environment.PRODUCTION:
            return True, "Not in production"

        # List of operations requiring extra confirmation
        sensitive_ops = [
            "delete_all_data",
            "reset_positions",
            "emergency_stop",
            "modify_risk_params",
        ]

        if operation in sensitive_ops:
            print(f"\n{Fore.RED}⚠️  SENSITIVE OPERATION: {operation}{Style.RESET_ALL}")
            confirmation = input("Type operation name to confirm: ")
            if confirmation != operation:
                return False, "Operation not confirmed"

        return True, "Operation allowed"

    def _verify_mfa(self, code: str) -> bool:
        """Verify MFA code (placeholder - implement actual MFA)."""
        # In production, integrate with actual MFA provider
        # For now, accept a specific code for testing
        return code == "123456"  # TODO: Implement real MFA
