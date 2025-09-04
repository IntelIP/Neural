"""
Environment Validator

Validation and testing utilities for environment configurations
and transitions.
"""

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pytest
import redis.asyncio as redis

from .environment_guards import (
    ComplianceGuard,
    CrossEnvironmentGuard,
    DataIntegrityGuard,
    EmergencyStopGuard,
    RateLimitGuard,
    ResourceGuard,
    TransitionGuard,
)
from .environment_manager import Environment, EnvironmentConfig, EnvironmentManager
from .redis_config import EnvironmentRedisManager, RedisConfig
from .transition_protocol import ModeTransitionProtocol, TransitionPlan


@dataclass
class ValidationResult:
    """Result of validation check."""

    check_name: str
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "check_name": self.check_name,
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ValidationReport:
    """Complete validation report."""

    environment: Environment
    total_checks: int
    passed_checks: int
    failed_checks: int
    results: List[ValidationResult]
    duration: timedelta
    generated_at: datetime

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_checks == 0:
            return 0.0
        return self.passed_checks / self.total_checks

    def get_failures(self) -> List[ValidationResult]:
        """Get failed validation results."""
        return [r for r in self.results if not r.passed]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "environment": self.environment.value,
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "success_rate": self.success_rate,
            "duration": str(self.duration),
            "generated_at": self.generated_at.isoformat(),
            "results": [r.to_dict() for r in self.results],
        }


class EnvironmentValidator:
    """Validator for environment configurations."""

    def __init__(self, manager: EnvironmentManager):
        """Initialize validator."""
        self.manager = manager
        self.redis_manager = EnvironmentRedisManager(manager)
        self.checks = {
            "config": self.validate_config,
            "redis": self.validate_redis,
            "api": self.validate_api_endpoints,
            "guards": self.validate_guards,
            "isolation": self.validate_isolation,
            "permissions": self.validate_permissions,
            "resources": self.validate_resources,
            "data_integrity": self.validate_data_integrity,
        }

    async def validate_environment(
        self,
        environment: Optional[Environment] = None,
        checks: Optional[List[str]] = None,
    ) -> ValidationReport:
        """Run complete environment validation."""
        env = environment or self.manager.get_current_environment()
        checks_to_run = checks or list(self.checks.keys())

        start_time = datetime.now()
        results = []

        for check_name in checks_to_run:
            if check_name in self.checks:
                result = await self.checks[check_name](env)
                results.append(result)

        # Calculate statistics
        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed

        report = ValidationReport(
            environment=env,
            total_checks=len(results),
            passed_checks=passed,
            failed_checks=failed,
            results=results,
            duration=datetime.now() - start_time,
            generated_at=datetime.now(),
        )

        return report

    async def validate_config(self, env: Environment) -> ValidationResult:
        """Validate environment configuration."""
        try:
            config = self.manager.get_config()

            # Check required fields
            required_fields = [
                "name",
                "environment",
                "redis_db",
                "redis_prefix",
                "api_endpoints",
                "rate_limits",
                "max_position_size",
            ]

            missing = []
            for field in required_fields:
                if not hasattr(config, field):
                    missing.append(field)

            if missing:
                return ValidationResult(
                    check_name="config",
                    passed=False,
                    message=f"Missing required fields: {missing}",
                    details={"missing_fields": missing},
                )

            # Validate field values
            issues = []

            if config.redis_db < 0 or config.redis_db > 15:
                issues.append("Redis DB must be between 0 and 15")

            if config.max_position_size < 0 or config.max_position_size > 1:
                issues.append("Max position size must be between 0 and 1")

            if config.max_daily_trades < 0:
                issues.append("Max daily trades must be non-negative")

            if issues:
                return ValidationResult(
                    check_name="config",
                    passed=False,
                    message="Configuration validation issues",
                    details={"issues": issues},
                )

            return ValidationResult(
                check_name="config",
                passed=True,
                message="Configuration valid",
                details={"config": config.name},
            )

        except Exception as e:
            return ValidationResult(
                check_name="config",
                passed=False,
                message=f"Configuration validation error: {str(e)}",
                details={"error": str(e)},
            )

    async def validate_redis(self, env: Environment) -> ValidationResult:
        """Validate Redis connectivity."""
        try:
            # Test connection
            connected = await self.redis_manager.ping(env)

            if not connected:
                return ValidationResult(
                    check_name="redis",
                    passed=False,
                    message="Redis connection failed",
                    details={"environment": env.value},
                )

            # Test basic operations
            test_key = f"validation:test:{env.value}"
            test_value = f"test_{datetime.now().timestamp()}"

            # Set
            await self.redis_manager.set_key(
                test_key, test_value, ttl=60, environment=env
            )

            # Get
            retrieved = await self.redis_manager.get_key(test_key, environment=env)

            if retrieved != test_value:
                return ValidationResult(
                    check_name="redis",
                    passed=False,
                    message="Redis read/write test failed",
                    details={"expected": test_value, "retrieved": retrieved},
                )

            # Clean up
            if env != Environment.PRODUCTION:
                await self.redis_manager.delete_key(test_key, environment=env)

            return ValidationResult(
                check_name="redis",
                passed=True,
                message="Redis connection and operations successful",
                details={"db": RedisConfig.for_environment(env).db},
            )

        except Exception as e:
            return ValidationResult(
                check_name="redis",
                passed=False,
                message=f"Redis validation error: {str(e)}",
                details={"error": str(e)},
            )

    async def validate_api_endpoints(self, env: Environment) -> ValidationResult:
        """Validate API endpoint configuration."""
        try:
            config = self.manager.get_config()
            endpoints = config.api_endpoints

            if not endpoints:
                return ValidationResult(
                    check_name="api",
                    passed=False,
                    message="No API endpoints configured",
                    details={},
                )

            # Check endpoint formats
            invalid = []
            for service, endpoint in endpoints.items():
                if not endpoint:
                    invalid.append(f"{service}: empty")
                elif not (
                    endpoint.startswith("http://") or endpoint.startswith("https://")
                ):
                    invalid.append(f"{service}: invalid protocol")

            if invalid:
                return ValidationResult(
                    check_name="api",
                    passed=False,
                    message="Invalid API endpoints",
                    details={"invalid": invalid},
                )

            # Check production requirements
            if env == Environment.PRODUCTION:
                if any(
                    "localhost" in ep or "127.0.0.1" in ep for ep in endpoints.values()
                ):
                    return ValidationResult(
                        check_name="api",
                        passed=False,
                        message="Production cannot use localhost endpoints",
                        details={"endpoints": endpoints},
                    )

            return ValidationResult(
                check_name="api",
                passed=True,
                message="API endpoints valid",
                details={"endpoints": list(endpoints.keys())},
            )

        except Exception as e:
            return ValidationResult(
                check_name="api",
                passed=False,
                message=f"API validation error: {str(e)}",
                details={"error": str(e)},
            )

    async def validate_guards(self, env: Environment) -> ValidationResult:
        """Validate environment guards."""
        try:
            # Test each guard type
            guards_tested = []
            guards_failed = []

            # Rate limit guard
            rate_guard = RateLimitGuard()
            allowed, msg = await rate_guard.check_operation("test_op", env)
            guards_tested.append("rate_limit")
            if not allowed and env != Environment.PRODUCTION:
                guards_failed.append(f"rate_limit: {msg}")

            # Data integrity guard
            integrity_guard = DataIntegrityGuard()
            allowed, msg = await integrity_guard.check_operation("read_data", env)
            guards_tested.append("data_integrity")
            if not allowed:
                guards_failed.append(f"data_integrity: {msg}")

            # Compliance guard
            compliance_guard = ComplianceGuard()
            allowed, msg = await compliance_guard.check_operation("get_status", env)
            guards_tested.append("compliance")
            if not allowed:
                guards_failed.append(f"compliance: {msg}")

            # Resource guard
            resource_guard = ResourceGuard()
            allowed, msg = await resource_guard.check_operation("test_op", env)
            guards_tested.append("resource")
            if not allowed:
                guards_failed.append(f"resource: {msg}")

            if guards_failed:
                return ValidationResult(
                    check_name="guards",
                    passed=False,
                    message="Some guards failed validation",
                    details={"tested": guards_tested, "failed": guards_failed},
                )

            return ValidationResult(
                check_name="guards",
                passed=True,
                message="All guards operational",
                details={"guards_tested": guards_tested},
            )

        except Exception as e:
            return ValidationResult(
                check_name="guards",
                passed=False,
                message=f"Guard validation error: {str(e)}",
                details={"error": str(e)},
            )

    async def validate_isolation(self, env: Environment) -> ValidationResult:
        """Validate environment isolation."""
        try:
            # Check Redis isolation
            redis_config = RedisConfig.for_environment(env)

            # Verify different DBs for different environments
            db_mapping = {}
            for test_env in Environment:
                test_config = RedisConfig.for_environment(test_env)
                if test_config.db in db_mapping:
                    return ValidationResult(
                        check_name="isolation",
                        passed=False,
                        message="Redis DB collision detected",
                        details={
                            "collision": f"{test_env.value} and {db_mapping[test_config.db]} use same DB {test_config.db}"
                        },
                    )
                db_mapping[test_config.db] = test_env.value

            # Check channel isolation
            test_channel = f"test:channel:{env.value}"
            allowed = self.redis_manager.is_channel_allowed(test_channel, env)

            # Production should have restricted channels
            if env == Environment.PRODUCTION:
                dev_channel = "dev:test"
                if self.redis_manager.is_channel_allowed(dev_channel, env):
                    return ValidationResult(
                        check_name="isolation",
                        passed=False,
                        message="Production allows dev channels",
                        details={"channel": dev_channel},
                    )

            return ValidationResult(
                check_name="isolation",
                passed=True,
                message="Environment properly isolated",
                details={"redis_db": redis_config.db, "prefix": redis_config.prefix},
            )

        except Exception as e:
            return ValidationResult(
                check_name="isolation",
                passed=False,
                message=f"Isolation validation error: {str(e)}",
                details={"error": str(e)},
            )

    async def validate_permissions(self, env: Environment) -> ValidationResult:
        """Validate environment permissions."""
        try:
            config = self.manager.get_config()

            # Check critical operations
            critical_ops = [
                "delete_all_data",
                "reset_positions",
                "modify_system_config",
            ]

            issues = []

            for op in critical_ops:
                allowed, reason = self.manager.can_execute_operation(op)

                # Critical ops should be restricted in production
                if env == Environment.PRODUCTION and allowed:
                    issues.append(f"{op} allowed in production")

                # Should be allowed in development
                if env == Environment.DEVELOPMENT and not allowed:
                    issues.append(f"{op} blocked in development")

            if issues:
                return ValidationResult(
                    check_name="permissions",
                    passed=False,
                    message="Permission configuration issues",
                    details={"issues": issues},
                )

            # Check authentication requirements
            if env == Environment.PRODUCTION and not config.require_mfa:
                return ValidationResult(
                    check_name="permissions",
                    passed=False,
                    message="Production must require MFA",
                    details={"require_mfa": config.require_mfa},
                )

            return ValidationResult(
                check_name="permissions",
                passed=True,
                message="Permissions properly configured",
                details={
                    "require_confirmation": config.require_confirmation,
                    "require_mfa": config.require_mfa,
                },
            )

        except Exception as e:
            return ValidationResult(
                check_name="permissions",
                passed=False,
                message=f"Permission validation error: {str(e)}",
                details={"error": str(e)},
            )

    async def validate_resources(self, env: Environment) -> ValidationResult:
        """Validate resource limits."""
        try:
            config = self.manager.get_config()

            issues = []

            # Check trading limits
            if config.max_position_size <= 0:
                issues.append("Invalid max position size")

            if config.max_daily_trades <= 0:
                issues.append("Invalid max daily trades")

            # Production should have conservative limits
            if env == Environment.PRODUCTION:
                if config.max_position_size > 0.1:  # 10%
                    issues.append("Production position size too high")

                if config.max_daily_trades > 100:
                    issues.append("Production daily trade limit too high")

            # Development should have relaxed limits
            if env == Environment.DEVELOPMENT:
                if config.max_position_size < 0.5:
                    issues.append("Development position size too restrictive")

            if issues:
                return ValidationResult(
                    check_name="resources",
                    passed=False,
                    message="Resource limit issues",
                    details={"issues": issues},
                )

            return ValidationResult(
                check_name="resources",
                passed=True,
                message="Resource limits appropriate",
                details={
                    "max_position_size": config.max_position_size,
                    "max_daily_trades": config.max_daily_trades,
                },
            )

        except Exception as e:
            return ValidationResult(
                check_name="resources",
                passed=False,
                message=f"Resource validation error: {str(e)}",
                details={"error": str(e)},
            )

    async def validate_data_integrity(self, env: Environment) -> ValidationResult:
        """Validate data integrity measures."""
        try:
            config = self.manager.get_config()

            issues = []

            # Check data retention
            if config.data_retention_hours <= 0:
                issues.append("Invalid data retention period")

            # Production should have longer retention
            if env == Environment.PRODUCTION:
                if config.data_retention_hours < 168:  # 1 week
                    issues.append("Production retention too short")

            # Check logging
            if env == Environment.PRODUCTION and not config.enable_logging:
                issues.append("Logging must be enabled in production")

            # Check telemetry
            if env == Environment.PRODUCTION and not config.telemetry_enabled:
                issues.append("Telemetry should be enabled in production")

            if issues:
                return ValidationResult(
                    check_name="data_integrity",
                    passed=False,
                    message="Data integrity issues",
                    details={"issues": issues},
                )

            return ValidationResult(
                check_name="data_integrity",
                passed=True,
                message="Data integrity measures in place",
                details={
                    "retention_hours": config.data_retention_hours,
                    "logging": config.enable_logging,
                    "telemetry": config.telemetry_enabled,
                },
            )

        except Exception as e:
            return ValidationResult(
                check_name="data_integrity",
                passed=False,
                message=f"Data integrity validation error: {str(e)}",
                details={"error": str(e)},
            )


class EnvironmentTester:
    """Testing utilities for environment transitions."""

    def __init__(self, manager: EnvironmentManager):
        """Initialize tester."""
        self.manager = manager
        self.validator = EnvironmentValidator(manager)
        self.transition_protocol = ModeTransitionProtocol(manager)

    async def test_transition(
        self, from_env: Environment, to_env: Environment, dry_run: bool = True
    ) -> Dict[str, Any]:
        """Test environment transition."""
        results = {
            "from_env": from_env.value,
            "to_env": to_env.value,
            "dry_run": dry_run,
            "tests": [],
        }

        # Validate source environment
        from_validation = await self.validator.validate_environment(from_env)
        results["tests"].append(
            {
                "name": f"Validate {from_env.value}",
                "passed": from_validation.success_rate == 1.0,
                "details": from_validation.to_dict(),
            }
        )

        # Create transition plan
        plan = self.transition_protocol.create_transition_plan(
            from_env, to_env, gradual=False
        )

        results["tests"].append(
            {
                "name": "Create transition plan",
                "passed": True,
                "details": {
                    "phases": [p.value for p in plan.phases],
                    "auth_required": [a.value for a in plan.auth_required],
                    "estimated_duration": str(plan.estimated_duration),
                },
            }
        )

        # Test guards
        guard = TransitionGuard()
        allowed, reason = await guard.check_switch(from_env, to_env)

        results["tests"].append(
            {
                "name": "Guard check",
                "passed": allowed or (not allowed and "forbidden" in reason),
                "details": {"allowed": allowed, "reason": reason},
            }
        )

        # Test data migration (dry run)
        if plan.data_migration and not dry_run:
            redis_manager = EnvironmentRedisManager(self.manager)
            migration_result = await redis_manager.migrate_data(
                from_env, to_env, pattern="test:*", dry_run=True
            )

            results["tests"].append(
                {
                    "name": "Data migration test",
                    "passed": migration_result["errors"] == 0,
                    "details": migration_result,
                }
            )

        # Calculate overall result
        all_passed = all(test["passed"] for test in results["tests"])
        results["overall_passed"] = all_passed
        results["recommendation"] = (
            "Safe to proceed"
            if all_passed
            else "Issues detected - review before proceeding"
        )

        return results

    async def test_isolation(self) -> Dict[str, Any]:
        """Test environment isolation."""
        results = {"timestamp": datetime.now().isoformat(), "tests": []}

        # Test Redis isolation
        redis_manager = EnvironmentRedisManager(self.manager)

        for env in Environment:
            # Try to access other environment's data
            test_key = f"isolation_test:{env.value}"

            # Set in one environment
            await redis_manager.set_key(
                test_key, f"data_{env.value}", ttl=60, environment=env
            )

            # Try to read from different environment
            for other_env in Environment:
                if other_env != env:
                    value = await redis_manager.get_key(test_key, environment=other_env)

                    # Should not see other environment's data
                    if value == f"data_{env.value}":
                        results["tests"].append(
                            {
                                "name": f"Isolation {env.value} from {other_env.value}",
                                "passed": False,
                                "message": f"Data leaked from {env.value} to {other_env.value}",
                            }
                        )
                    else:
                        results["tests"].append(
                            {
                                "name": f"Isolation {env.value} from {other_env.value}",
                                "passed": True,
                                "message": "Properly isolated",
                            }
                        )

        results["overall_passed"] = all(test["passed"] for test in results["tests"])
        return results

    async def test_guards(self) -> Dict[str, Any]:
        """Test all environment guards."""
        results = {"timestamp": datetime.now().isoformat(), "guards": []}

        guards_to_test = [
            ("RateLimitGuard", RateLimitGuard()),
            ("DataIntegrityGuard", DataIntegrityGuard()),
            ("ComplianceGuard", ComplianceGuard()),
            ("ResourceGuard", ResourceGuard()),
            ("CrossEnvironmentGuard", CrossEnvironmentGuard()),
            ("TransitionGuard", TransitionGuard()),
            ("EmergencyStopGuard", EmergencyStopGuard()),
        ]

        for guard_name, guard in guards_to_test:
            guard_results = {"name": guard_name, "tests": []}

            # Test operation checks
            for env in Environment:
                allowed, reason = await guard.check_operation("test_operation", env)
                guard_results["tests"].append(
                    {
                        "type": "operation",
                        "environment": env.value,
                        "allowed": allowed,
                        "reason": reason,
                    }
                )

            # Test transition checks
            for from_env in [Environment.DEVELOPMENT, Environment.TRAINING]:
                for to_env in [Environment.STAGING, Environment.PRODUCTION]:
                    allowed, reason = await guard.check_switch(from_env, to_env)
                    guard_results["tests"].append(
                        {
                            "type": "transition",
                            "from": from_env.value,
                            "to": to_env.value,
                            "allowed": allowed,
                            "reason": reason,
                        }
                    )

            results["guards"].append(guard_results)

        return results


# Pytest fixtures and tests
@pytest.fixture
async def env_manager():
    """Create environment manager for testing."""
    manager = EnvironmentManager()
    manager.initialize(Environment.DEVELOPMENT)
    return manager


@pytest.fixture
async def validator(env_manager):
    """Create validator for testing."""
    return EnvironmentValidator(env_manager)


@pytest.mark.asyncio
async def test_environment_validation(validator):
    """Test environment validation."""
    report = await validator.validate_environment(Environment.DEVELOPMENT)
    assert report.total_checks > 0
    assert report.success_rate >= 0.8  # At least 80% should pass


@pytest.mark.asyncio
async def test_redis_isolation(env_manager):
    """Test Redis isolation between environments."""
    tester = EnvironmentTester(env_manager)
    results = await tester.test_isolation()
    assert results["overall_passed"]


@pytest.mark.asyncio
async def test_guard_functionality(env_manager):
    """Test environment guards."""
    tester = EnvironmentTester(env_manager)
    results = await tester.test_guards()
    assert len(results["guards"]) > 0
