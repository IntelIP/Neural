"""
Environment Guards

Safety mechanisms and validation for environment operations.
"""

import asyncio
import hashlib
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from .environment_manager import Environment, EnvironmentGuard


class OperationType(Enum):
    """Types of operations that can be guarded."""

    TRADE_EXECUTION = "trade_execution"
    DATA_DELETION = "data_deletion"
    CONFIG_MODIFICATION = "config_modification"
    API_CALL = "api_call"
    DATABASE_WRITE = "database_write"
    DATABASE_READ = "database_read"
    MODEL_TRAINING = "model_training"
    SYSTEM_RESTART = "system_restart"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class GuardPolicy:
    """Policy for environment guards."""

    operation_type: OperationType
    allowed_environments: List[Environment]
    require_confirmation: bool
    require_mfa: bool
    cooldown_seconds: int
    max_daily_count: int
    audit_log: bool

    def is_allowed_in(self, environment: Environment) -> bool:
        """Check if operation is allowed in environment."""
        return environment in self.allowed_environments


class RateLimitGuard(EnvironmentGuard):
    """Guard that enforces rate limits on operations."""

    def __init__(self, default_limit: int = 100):
        """Initialize rate limit guard."""
        self.default_limit = default_limit
        self.operation_counts: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.operation_times: Dict[str, List[datetime]] = defaultdict(list)
        self.limits: Dict[Tuple[Environment, str], int] = {}

    def set_limit(self, environment: Environment, operation: str, limit: int) -> None:
        """Set rate limit for operation in environment."""
        self.limits[(environment, operation)] = limit

    async def check_operation(
        self, operation: str, environment: Environment
    ) -> Tuple[bool, str]:
        """Check if operation is within rate limits."""
        # Get limit for this operation
        limit = self.limits.get((environment, operation), self.default_limit)

        # Clean old entries (older than 1 hour)
        now = datetime.now()
        cutoff = now - timedelta(hours=1)

        if operation in self.operation_times:
            self.operation_times[operation] = [
                t for t in self.operation_times[operation] if t > cutoff
            ]

        # Check current count
        current_count = len(self.operation_times[operation])

        if current_count >= limit:
            return (
                False,
                f"Rate limit exceeded: {current_count}/{limit} operations in last hour",
            )

        # Record operation
        self.operation_times[operation].append(now)

        return True, f"Operation allowed: {current_count + 1}/{limit}"

    async def check_switch(
        self, from_env: Environment, to_env: Environment
    ) -> Tuple[bool, str]:
        """Check if environment switch is allowed."""
        # Always allow switching for rate limit guard
        return True, "Switch allowed"


class DataIntegrityGuard(EnvironmentGuard):
    """Guard that ensures data integrity across environments."""

    def __init__(self):
        """Initialize data integrity guard."""
        self.data_checksums: Dict[Environment, Dict[str, str]] = defaultdict(dict)
        self.migration_history: List[Dict[str, Any]] = []

    async def check_switch(
        self, from_env: Environment, to_env: Environment
    ) -> Tuple[bool, str]:
        """Check if environment switch maintains data integrity."""
        # Don't allow direct switch from training to production
        if from_env == Environment.TRAINING and to_env == Environment.PRODUCTION:
            return (
                False,
                "Direct switch from training to production not allowed. Use staging first.",
            )

        # Warn about data loss when switching from production
        if from_env == Environment.PRODUCTION and to_env != Environment.STAGING:
            print("\n⚠️  WARNING: Switching from production may result in data loss.")
            confirm = input("Type 'ACKNOWLEDGE' to continue: ")
            if confirm != "ACKNOWLEDGE":
                return False, "Switch cancelled - data loss not acknowledged"

        return True, "Data integrity check passed"

    async def check_operation(
        self, operation: str, environment: Environment
    ) -> Tuple[bool, str]:
        """Check if operation maintains data integrity."""
        # Prevent data deletion in production
        if environment == Environment.PRODUCTION and "delete" in operation.lower():
            return False, "Data deletion operations not allowed in production"

        return True, "Operation allowed"

    def record_checksum(
        self, environment: Environment, data_type: str, data: str
    ) -> str:
        """Record checksum for data integrity tracking."""
        checksum = hashlib.sha256(data.encode()).hexdigest()
        self.data_checksums[environment][data_type] = checksum
        return checksum

    def verify_checksum(
        self, environment: Environment, data_type: str, data: str
    ) -> bool:
        """Verify data checksum."""
        expected = self.data_checksums[environment].get(data_type)
        if not expected:
            return True  # No checksum recorded

        actual = hashlib.sha256(data.encode()).hexdigest()
        return actual == expected


class ComplianceGuard(EnvironmentGuard):
    """Guard that ensures regulatory compliance."""

    def __init__(self):
        """Initialize compliance guard."""
        self.audit_log: List[Dict[str, Any]] = []
        self.compliance_rules: Dict[str, GuardPolicy] = {}
        self._setup_default_rules()

    def _setup_default_rules(self) -> None:
        """Setup default compliance rules."""
        # Trading must be audited in production
        self.compliance_rules["trade_execution"] = GuardPolicy(
            operation_type=OperationType.TRADE_EXECUTION,
            allowed_environments=list(Environment),
            require_confirmation=True,
            require_mfa=True,
            cooldown_seconds=1,
            max_daily_count=100,
            audit_log=True,
        )

        # Data deletion requires special permission
        self.compliance_rules["data_deletion"] = GuardPolicy(
            operation_type=OperationType.DATA_DELETION,
            allowed_environments=[
                Environment.DEVELOPMENT,
                Environment.TRAINING,
                Environment.SANDBOX,
            ],
            require_confirmation=True,
            require_mfa=False,
            cooldown_seconds=60,
            max_daily_count=10,
            audit_log=True,
        )

    async def check_operation(
        self, operation: str, environment: Environment
    ) -> Tuple[bool, str]:
        """Check if operation is compliant."""
        # Check if operation has compliance rules
        if operation in self.compliance_rules:
            policy = self.compliance_rules[operation]

            if not policy.is_allowed_in(environment):
                self._audit(operation, environment, False, "Environment not allowed")
                return (
                    False,
                    f"Operation '{operation}' not allowed in {environment.value}",
                )

            # Audit if required
            if policy.audit_log:
                self._audit(operation, environment, True, "Operation allowed")

        return True, "Compliance check passed"

    async def check_switch(
        self, from_env: Environment, to_env: Environment
    ) -> Tuple[bool, str]:
        """Check if environment switch is compliant."""
        # Audit all environment switches
        self._audit(
            f"switch_{from_env.value}_to_{to_env.value}",
            to_env,
            True,
            "Environment switch",
        )

        return True, "Switch compliant"

    def _audit(
        self, operation: str, environment: Environment, allowed: bool, reason: str
    ) -> None:
        """Add entry to audit log."""
        self.audit_log.append(
            {
                "timestamp": datetime.now().isoformat(),
                "operation": operation,
                "environment": environment.value,
                "allowed": allowed,
                "reason": reason,
                "user": os.getenv("USER", "unknown"),
            }
        )

    def get_audit_log(
        self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get audit log entries."""
        if not start_time and not end_time:
            return self.audit_log

        filtered = []
        for entry in self.audit_log:
            entry_time = datetime.fromisoformat(entry["timestamp"])

            if start_time and entry_time < start_time:
                continue
            if end_time and entry_time > end_time:
                continue

            filtered.append(entry)

        return filtered


class ResourceGuard(EnvironmentGuard):
    """Guard that monitors and limits resource usage."""

    def __init__(self):
        """Initialize resource guard."""
        self.resource_limits: Dict[Environment, Dict[str, float]] = {
            Environment.PRODUCTION: {
                "memory_gb": 4.0,
                "cpu_percent": 80.0,
                "disk_gb": 100.0,
                "connections": 100,
            },
            Environment.STAGING: {
                "memory_gb": 8.0,
                "cpu_percent": 90.0,
                "disk_gb": 200.0,
                "connections": 200,
            },
            Environment.SANDBOX: {
                "memory_gb": 4.0,
                "cpu_percent": 90.0,
                "disk_gb": 50.0,
                "connections": 50,
            },
            Environment.TRAINING: {
                "memory_gb": 16.0,
                "cpu_percent": 95.0,
                "disk_gb": 500.0,
                "connections": 500,
            },
            Environment.DEVELOPMENT: {
                "memory_gb": 32.0,
                "cpu_percent": 100.0,
                "disk_gb": 1000.0,
                "connections": 1000,
            },
        }
        self.current_usage: Dict[str, float] = defaultdict(float)

    async def check_operation(
        self, operation: str, environment: Environment
    ) -> Tuple[bool, str]:
        """Check if operation can proceed given resource constraints."""
        # Get current resource usage (simplified - in production use psutil)
        import random

        self.current_usage["memory_gb"] = random.uniform(1, 10)
        self.current_usage["cpu_percent"] = random.uniform(10, 100)

        limits = self.resource_limits[environment]

        # Check memory
        if self.current_usage["memory_gb"] > limits["memory_gb"]:
            return (
                False,
                f"Memory limit exceeded: {self.current_usage['memory_gb']:.1f}GB > {limits['memory_gb']}GB",
            )

        # Check CPU
        if self.current_usage["cpu_percent"] > limits["cpu_percent"]:
            return (
                False,
                f"CPU limit exceeded: {self.current_usage['cpu_percent']:.1f}% > {limits['cpu_percent']}%",
            )

        return True, "Resource check passed"

    async def check_switch(
        self, from_env: Environment, to_env: Environment
    ) -> Tuple[bool, str]:
        """Check if environment switch is possible given resources."""
        # Check if target environment has enough resources
        target_limits = self.resource_limits[to_env]

        if self.current_usage["memory_gb"] > target_limits["memory_gb"]:
            return (
                False,
                f"Insufficient memory for {to_env.value}: need {self.current_usage['memory_gb']:.1f}GB, limit {target_limits['memory_gb']}GB",
            )

        return True, "Resources available"


class CrossEnvironmentGuard(EnvironmentGuard):
    """Guard that prevents cross-environment contamination."""

    def __init__(self):
        """Initialize cross-environment guard."""
        self.environment_boundaries: Dict[Environment, Set[str]] = {
            Environment.PRODUCTION: {"prod_db", "prod_api", "prod_redis"},
            Environment.STAGING: {"staging_db", "staging_api", "staging_redis"},
            Environment.SANDBOX: {"sandbox_db", "sandbox_api", "sandbox_redis"},
            Environment.TRAINING: {"training_db", "training_api", "training_redis"},
            Environment.DEVELOPMENT: {"dev_db", "dev_api", "dev_redis"},
        }
        self.active_connections: Dict[str, Environment] = {}

    async def check_operation(
        self, operation: str, environment: Environment
    ) -> Tuple[bool, str]:
        """Check if operation respects environment boundaries."""
        # Parse operation for resource access
        resources = self._extract_resources(operation)

        allowed_resources = self.environment_boundaries[environment]

        for resource in resources:
            # Check if resource belongs to current environment
            if resource not in allowed_resources:
                # Check if it belongs to another environment
                for env, env_resources in self.environment_boundaries.items():
                    if env != environment and resource in env_resources:
                        return (
                            False,
                            f"Cross-environment access denied: {resource} belongs to {env.value}",
                        )

        return True, "Environment boundaries respected"

    async def check_switch(
        self, from_env: Environment, to_env: Environment
    ) -> Tuple[bool, str]:
        """Check if environment switch is clean."""
        # Check for active connections to old environment
        active_in_old = [
            conn for conn, env in self.active_connections.items() if env == from_env
        ]

        if active_in_old:
            return (
                False,
                f"Active connections to {from_env.value}: {', '.join(active_in_old)}",
            )

        return True, "No active connections to close"

    def _extract_resources(self, operation: str) -> Set[str]:
        """Extract resource names from operation."""
        # Simplified extraction - in production use proper parsing
        resources = set()

        for env_resources in self.environment_boundaries.values():
            for resource in env_resources:
                if resource in operation:
                    resources.add(resource)

        return resources

    def register_connection(self, resource: str, environment: Environment) -> None:
        """Register active connection to resource."""
        self.active_connections[resource] = environment

    def unregister_connection(self, resource: str) -> None:
        """Unregister connection to resource."""
        self.active_connections.pop(resource, None)


class TransitionGuard(EnvironmentGuard):
    """Guard that manages safe environment transitions."""

    def __init__(self):
        """Initialize transition guard."""
        self.transition_rules: Dict[Tuple[Environment, Environment], List[str]] = {
            # From Dev to others
            (Environment.DEVELOPMENT, Environment.TRAINING): [],
            (Environment.DEVELOPMENT, Environment.SANDBOX): ["run_tests"],
            (Environment.DEVELOPMENT, Environment.STAGING): [
                "run_tests",
                "code_review",
            ],
            (Environment.DEVELOPMENT, Environment.PRODUCTION): ["forbidden"],
            # From Training to others
            (Environment.TRAINING, Environment.DEVELOPMENT): [],
            (Environment.TRAINING, Environment.SANDBOX): ["save_model"],
            (Environment.TRAINING, Environment.STAGING): ["validate_model"],
            (Environment.TRAINING, Environment.PRODUCTION): ["forbidden"],
            # From Sandbox to others
            (Environment.SANDBOX, Environment.DEVELOPMENT): [],
            (Environment.SANDBOX, Environment.TRAINING): [],
            (Environment.SANDBOX, Environment.STAGING): ["integration_test"],
            (Environment.SANDBOX, Environment.PRODUCTION): ["forbidden"],
            # From Staging to others
            (Environment.STAGING, Environment.DEVELOPMENT): [],
            (Environment.STAGING, Environment.TRAINING): [],
            (Environment.STAGING, Environment.SANDBOX): [],
            (Environment.STAGING, Environment.PRODUCTION): [
                "approval",
                "backup",
                "health_check",
            ],
            # From Production to others
            (Environment.PRODUCTION, Environment.STAGING): ["backup"],
            (Environment.PRODUCTION, Environment.SANDBOX): ["forbidden"],
            (Environment.PRODUCTION, Environment.TRAINING): ["forbidden"],
            (Environment.PRODUCTION, Environment.DEVELOPMENT): ["forbidden"],
        }

        self.completed_checks: Set[str] = set()

    async def check_switch(
        self, from_env: Environment, to_env: Environment
    ) -> Tuple[bool, str]:
        """Check if environment transition is allowed."""
        if from_env == to_env:
            return True, "Same environment"

        # Get transition rules
        rules = self.transition_rules.get((from_env, to_env), [])

        # Check if transition is forbidden
        if "forbidden" in rules:
            return (
                False,
                f"Direct transition from {from_env.value} to {to_env.value} is forbidden",
            )

        # Check if all required checks are completed
        missing_checks = [
            check for check in rules if check not in self.completed_checks
        ]

        if missing_checks:
            return False, f"Required checks not completed: {', '.join(missing_checks)}"

        # Clear completed checks after successful transition
        self.completed_checks.clear()

        return True, f"Transition from {from_env.value} to {to_env.value} allowed"

    async def check_operation(
        self, operation: str, environment: Environment
    ) -> Tuple[bool, str]:
        """Check if operation is allowed in current environment."""
        return True, "Operation allowed"

    def complete_check(self, check_name: str) -> None:
        """Mark a transition check as completed."""
        self.completed_checks.add(check_name)

    def get_required_checks(
        self, from_env: Environment, to_env: Environment
    ) -> List[str]:
        """Get list of required checks for transition."""
        return self.transition_rules.get((from_env, to_env), [])


class EmergencyStopGuard(EnvironmentGuard):
    """Guard that implements emergency stop functionality."""

    def __init__(self):
        """Initialize emergency stop guard."""
        self.emergency_active = False
        self.emergency_reason = ""
        self.emergency_time: Optional[datetime] = None
        self.allowed_during_emergency = [
            "get_status",
            "cancel_order",
            "close_position",
            "emergency_stop",
        ]

    def activate_emergency_stop(self, reason: str) -> None:
        """Activate emergency stop."""
        self.emergency_active = True
        self.emergency_reason = reason
        self.emergency_time = datetime.now()
        print(f"\n🚨 EMERGENCY STOP ACTIVATED: {reason}")

    def deactivate_emergency_stop(self) -> None:
        """Deactivate emergency stop."""
        if self.emergency_active:
            duration = datetime.now() - self.emergency_time
            print(f"\n✅ Emergency stop deactivated after {duration}")

        self.emergency_active = False
        self.emergency_reason = ""
        self.emergency_time = None

    async def check_operation(
        self, operation: str, environment: Environment
    ) -> Tuple[bool, str]:
        """Check if operation is allowed during emergency."""
        if not self.emergency_active:
            return True, "No emergency"

        # Only allow specific operations during emergency
        if operation in self.allowed_during_emergency:
            return True, f"Operation allowed during emergency: {operation}"

        return False, f"Emergency stop active: {self.emergency_reason}"

    async def check_switch(
        self, from_env: Environment, to_env: Environment
    ) -> Tuple[bool, str]:
        """Check if environment switch is allowed during emergency."""
        if self.emergency_active:
            return (
                False,
                f"Environment switch blocked: emergency stop active ({self.emergency_reason})",
            )

        return True, "No emergency"

    def is_emergency(self) -> bool:
        """Check if emergency stop is active."""
        return self.emergency_active

    def get_emergency_status(self) -> Dict[str, Any]:
        """Get emergency stop status."""
        if not self.emergency_active:
            return {"active": False}

        return {
            "active": True,
            "reason": self.emergency_reason,
            "start_time": self.emergency_time.isoformat(),
            "duration": str(datetime.now() - self.emergency_time),
        }
