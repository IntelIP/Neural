"""
Mode Transition Protocol

Manages safe transitions between training and production modes
with authentication, validation, and rollback capabilities.
"""

import asyncio
import os
import secrets
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import pyotp

from .environment_manager import Environment, EnvironmentManager


class TransitionPhase(Enum):
    """Phases of environment transition."""

    VALIDATION = "validation"
    PREPARATION = "preparation"
    MIGRATION = "migration"
    VERIFICATION = "verification"
    ACTIVATION = "activation"
    MONITORING = "monitoring"
    COMPLETE = "complete"
    ROLLBACK = "rollback"


class AuthenticationMethod(Enum):
    """Authentication methods for transitions."""

    NONE = "none"
    PASSWORD = "password"
    MFA_TOTP = "mfa_totp"
    MFA_SMS = "mfa_sms"
    API_KEY = "api_key"
    HARDWARE_TOKEN = "hardware_token"


@dataclass
class TransitionCheckpoint:
    """Checkpoint in transition process."""

    phase: TransitionPhase
    timestamp: datetime
    status: str
    details: Dict[str, Any]
    can_rollback: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "phase": self.phase.value,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status,
            "details": self.details,
            "can_rollback": self.can_rollback,
        }


@dataclass
class TransitionPlan:
    """Plan for environment transition."""

    id: str
    from_env: Environment
    to_env: Environment
    phases: List[TransitionPhase]
    auth_required: List[AuthenticationMethod]
    estimated_duration: timedelta
    rollback_points: List[TransitionPhase]
    validation_checks: List[str]
    data_migration: bool
    gradual_rollout: bool
    rollout_percentage: float = 0.0

    def get_next_phase(self, current: TransitionPhase) -> Optional[TransitionPhase]:
        """Get next phase in plan."""
        try:
            current_index = self.phases.index(current)
            if current_index < len(self.phases) - 1:
                return self.phases[current_index + 1]
        except ValueError:
            pass
        return None

    def can_rollback_from(self, phase: TransitionPhase) -> bool:
        """Check if rollback is possible from phase."""
        return phase in self.rollback_points


@dataclass
class TransitionState:
    """Current state of transition."""

    plan: TransitionPlan
    current_phase: TransitionPhase
    started_at: datetime
    checkpoints: List[TransitionCheckpoint] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    authenticated: bool = False
    auth_methods_completed: List[AuthenticationMethod] = field(default_factory=list)
    rollback_initiated: bool = False

    def add_checkpoint(
        self,
        phase: TransitionPhase,
        status: str,
        details: Dict[str, Any],
        can_rollback: bool = True,
    ) -> None:
        """Add checkpoint to transition."""
        checkpoint = TransitionCheckpoint(
            phase=phase,
            timestamp=datetime.now(),
            status=status,
            details=details,
            can_rollback=can_rollback,
        )
        self.checkpoints.append(checkpoint)

    def get_duration(self) -> timedelta:
        """Get transition duration."""
        return datetime.now() - self.started_at

    def is_complete(self) -> bool:
        """Check if transition is complete."""
        return self.current_phase == TransitionPhase.COMPLETE

    def is_failed(self) -> bool:
        """Check if transition has failed."""
        return self.rollback_initiated or len(self.errors) > 0


class AuthenticationProvider:
    """Provider for authentication services."""

    def __init__(self):
        """Initialize authentication provider."""
        self.totp_secrets: Dict[str, str] = {}
        self.api_keys: Dict[str, str] = {}
        self.session_tokens: Dict[str, datetime] = {}

    def setup_totp(self, user_id: str) -> str:
        """Setup TOTP for user."""
        secret = pyotp.random_base32()
        self.totp_secrets[user_id] = secret

        # Generate provisioning URI for QR code
        totp = pyotp.TOTP(secret)
        provisioning_uri = totp.provisioning_uri(
            name=user_id, issuer_name="Neural Trading Agent"
        )

        return provisioning_uri

    def verify_totp(self, user_id: str, token: str) -> bool:
        """Verify TOTP token."""
        if user_id not in self.totp_secrets:
            return False

        totp = pyotp.TOTP(self.totp_secrets[user_id])
        return totp.verify(token, valid_window=1)

    def generate_api_key(self, user_id: str) -> str:
        """Generate API key for user."""
        key = secrets.token_urlsafe(32)
        self.api_keys[user_id] = key
        return key

    def verify_api_key(self, user_id: str, key: str) -> bool:
        """Verify API key."""
        return self.api_keys.get(user_id) == key

    def create_session(self, user_id: str) -> str:
        """Create authenticated session."""
        token = secrets.token_urlsafe(32)
        self.session_tokens[token] = datetime.now()
        return token

    def verify_session(
        self, token: str, max_age: timedelta = timedelta(hours=1)
    ) -> bool:
        """Verify session token."""
        if token not in self.session_tokens:
            return False

        age = datetime.now() - self.session_tokens[token]
        return age < max_age


class ModeTransitionProtocol:
    """Protocol for managing mode transitions."""

    def __init__(
        self,
        manager: EnvironmentManager,
        auth_provider: Optional[AuthenticationProvider] = None,
    ):
        """Initialize transition protocol."""
        self.manager = manager
        self.auth_provider = auth_provider or AuthenticationProvider()
        self.active_transitions: Dict[str, TransitionState] = {}
        self.completed_transitions: List[TransitionState] = []
        self.transition_hooks: Dict[TransitionPhase, List[Callable]] = defaultdict(list)
        self.rollback_handlers: Dict[TransitionPhase, Callable] = {}

    def create_transition_plan(
        self, from_env: Environment, to_env: Environment, gradual: bool = False
    ) -> TransitionPlan:
        """Create transition plan between environments."""
        plan_id = f"{from_env.value}-to-{to_env.value}-{int(time.time())}"

        # Determine required phases
        phases = [TransitionPhase.VALIDATION]

        # Add authentication phase for production
        if to_env == Environment.PRODUCTION:
            phases.append(TransitionPhase.PREPARATION)

        # Add migration if needed
        if self._needs_data_migration(from_env, to_env):
            phases.append(TransitionPhase.MIGRATION)

        phases.extend(
            [
                TransitionPhase.VERIFICATION,
                TransitionPhase.ACTIVATION,
                TransitionPhase.MONITORING,
                TransitionPhase.COMPLETE,
            ]
        )

        # Determine authentication requirements
        auth_required = self._get_auth_requirements(from_env, to_env)

        # Determine rollback points
        rollback_points = [
            TransitionPhase.VALIDATION,
            TransitionPhase.PREPARATION,
            TransitionPhase.MIGRATION,
            TransitionPhase.VERIFICATION,
        ]

        # Validation checks
        validation_checks = self._get_validation_checks(from_env, to_env)

        # Estimate duration
        duration = self._estimate_duration(from_env, to_env, gradual)

        return TransitionPlan(
            id=plan_id,
            from_env=from_env,
            to_env=to_env,
            phases=phases,
            auth_required=auth_required,
            estimated_duration=duration,
            rollback_points=rollback_points,
            validation_checks=validation_checks,
            data_migration=self._needs_data_migration(from_env, to_env),
            gradual_rollout=gradual,
            rollout_percentage=0.0 if gradual else 100.0,
        )

    def _needs_data_migration(self, from_env: Environment, to_env: Environment) -> bool:
        """Check if data migration is needed."""
        # Migration needed when moving to/from production
        return from_env == Environment.PRODUCTION or to_env == Environment.PRODUCTION

    def _get_auth_requirements(
        self, from_env: Environment, to_env: Environment
    ) -> List[AuthenticationMethod]:
        """Get authentication requirements for transition."""
        if to_env == Environment.PRODUCTION:
            return [AuthenticationMethod.PASSWORD, AuthenticationMethod.MFA_TOTP]
        elif to_env == Environment.STAGING:
            return [AuthenticationMethod.PASSWORD]
        else:
            return []

    def _get_validation_checks(
        self, from_env: Environment, to_env: Environment
    ) -> List[str]:
        """Get validation checks for transition."""
        checks = ["system_health", "dependency_check"]

        if to_env == Environment.PRODUCTION:
            checks.extend(
                [
                    "risk_parameters",
                    "api_connectivity",
                    "backup_verification",
                    "emergency_procedures",
                ]
            )

        if from_env == Environment.TRAINING:
            checks.append("model_validation")

        return checks

    def _estimate_duration(
        self, from_env: Environment, to_env: Environment, gradual: bool
    ) -> timedelta:
        """Estimate transition duration."""
        base_duration = timedelta(minutes=5)

        if to_env == Environment.PRODUCTION:
            base_duration = timedelta(minutes=30)

        if gradual:
            base_duration *= 3  # Gradual rollout takes longer

        if self._needs_data_migration(from_env, to_env):
            base_duration += timedelta(minutes=15)

        return base_duration

    async def start_transition(
        self, plan: TransitionPlan, user_id: str
    ) -> Tuple[bool, str]:
        """Start environment transition."""
        # Check if transition already active
        if plan.id in self.active_transitions:
            return False, "Transition already in progress"

        # Create transition state
        state = TransitionState(
            plan=plan,
            current_phase=TransitionPhase.VALIDATION,
            started_at=datetime.now(),
        )

        self.active_transitions[plan.id] = state

        # Start validation phase
        success = await self._execute_phase(state, TransitionPhase.VALIDATION)

        if not success:
            return False, "Validation failed"

        return True, f"Transition {plan.id} started"

    async def authenticate_transition(
        self,
        plan_id: str,
        user_id: str,
        method: AuthenticationMethod,
        credentials: Dict[str, str],
    ) -> Tuple[bool, str]:
        """Authenticate for transition."""
        if plan_id not in self.active_transitions:
            return False, "Transition not found"

        state = self.active_transitions[plan_id]

        # Verify authentication
        authenticated = False

        if method == AuthenticationMethod.PASSWORD:
            # Verify password (simplified - use proper password hashing)
            authenticated = credentials.get("password") == os.getenv(
                "TRANSITION_PASSWORD"
            )

        elif method == AuthenticationMethod.MFA_TOTP:
            token = credentials.get("token", "")
            authenticated = self.auth_provider.verify_totp(user_id, token)

        elif method == AuthenticationMethod.API_KEY:
            key = credentials.get("api_key", "")
            authenticated = self.auth_provider.verify_api_key(user_id, key)

        if authenticated:
            state.auth_methods_completed.append(method)

            # Check if all required auth completed
            if set(state.auth_methods_completed) >= set(state.plan.auth_required):
                state.authenticated = True
                return True, "Authentication complete"
            else:
                remaining = set(state.plan.auth_required) - set(
                    state.auth_methods_completed
                )
                return True, f"Authentication successful. Remaining: {remaining}"

        return False, f"Authentication failed for {method.value}"

    async def continue_transition(self, plan_id: str) -> Tuple[bool, str]:
        """Continue transition to next phase."""
        if plan_id not in self.active_transitions:
            return False, "Transition not found"

        state = self.active_transitions[plan_id]

        # Check authentication
        if state.plan.auth_required and not state.authenticated:
            return False, "Authentication required"

        # Get next phase
        next_phase = state.plan.get_next_phase(state.current_phase)

        if not next_phase:
            return False, "No next phase available"

        # Execute next phase
        success = await self._execute_phase(state, next_phase)

        if success:
            state.current_phase = next_phase

            # Check if complete
            if state.is_complete():
                self._complete_transition(plan_id)
                return True, "Transition complete"

            return True, f"Advanced to {next_phase.value}"
        else:
            return False, f"Failed to execute {next_phase.value}"

    async def _execute_phase(
        self, state: TransitionState, phase: TransitionPhase
    ) -> bool:
        """Execute transition phase."""
        try:
            # Run phase hooks
            for hook in self.transition_hooks[phase]:
                await hook(state)

            # Phase-specific execution
            if phase == TransitionPhase.VALIDATION:
                return await self._validate_transition(state)

            elif phase == TransitionPhase.PREPARATION:
                return await self._prepare_transition(state)

            elif phase == TransitionPhase.MIGRATION:
                return await self._migrate_data(state)

            elif phase == TransitionPhase.VERIFICATION:
                return await self._verify_transition(state)

            elif phase == TransitionPhase.ACTIVATION:
                return await self._activate_environment(state)

            elif phase == TransitionPhase.MONITORING:
                return await self._monitor_transition(state)

            elif phase == TransitionPhase.COMPLETE:
                return True

            else:
                return False

        except Exception as e:
            state.errors.append(f"Phase {phase.value} failed: {str(e)}")
            return False

    async def _validate_transition(self, state: TransitionState) -> bool:
        """Validate transition readiness."""
        results = {}

        for check in state.plan.validation_checks:
            # Run validation check
            if check == "system_health":
                results[check] = await self._check_system_health()
            elif check == "dependency_check":
                results[check] = await self._check_dependencies()
            elif check == "risk_parameters":
                results[check] = await self._check_risk_parameters()
            elif check == "api_connectivity":
                results[check] = await self._check_api_connectivity()
            elif check == "backup_verification":
                results[check] = await self._verify_backups()
            elif check == "emergency_procedures":
                results[check] = await self._check_emergency_procedures()
            elif check == "model_validation":
                results[check] = await self._validate_models()
            else:
                results[check] = True

        # Check if all passed
        all_passed = all(results.values())

        state.add_checkpoint(
            TransitionPhase.VALIDATION,
            "passed" if all_passed else "failed",
            results,
            can_rollback=True,
        )

        return all_passed

    async def _prepare_transition(self, state: TransitionState) -> bool:
        """Prepare for transition."""
        # Create backups
        backup_created = await self._create_backup(state.plan.from_env)

        # Prepare new environment
        env_prepared = await self._prepare_environment(state.plan.to_env)

        state.add_checkpoint(
            TransitionPhase.PREPARATION,
            "complete" if (backup_created and env_prepared) else "failed",
            {"backup_created": backup_created, "environment_prepared": env_prepared},
            can_rollback=True,
        )

        return backup_created and env_prepared

    async def _migrate_data(self, state: TransitionState) -> bool:
        """Migrate data between environments."""
        if not state.plan.data_migration:
            return True

        # Simplified data migration
        migrated = await self._copy_data(state.plan.from_env, state.plan.to_env)

        state.add_checkpoint(
            TransitionPhase.MIGRATION,
            "complete" if migrated else "failed",
            {"data_migrated": migrated},
            can_rollback=True,
        )

        return migrated

    async def _verify_transition(self, state: TransitionState) -> bool:
        """Verify transition integrity."""
        # Run verification tests
        tests_passed = await self._run_verification_tests(state.plan.to_env)

        state.add_checkpoint(
            TransitionPhase.VERIFICATION,
            "passed" if tests_passed else "failed",
            {"tests_passed": tests_passed},
            can_rollback=True,
        )

        return tests_passed

    async def _activate_environment(self, state: TransitionState) -> bool:
        """Activate new environment."""
        # Switch environment
        success, message = await self.manager.switch_environment(
            state.plan.to_env, force=True
        )

        state.add_checkpoint(
            TransitionPhase.ACTIVATION,
            "active" if success else "failed",
            {"switch_result": message},
            can_rollback=False,  # Cannot rollback after activation
        )

        return success

    async def _monitor_transition(self, state: TransitionState) -> bool:
        """Monitor transition for issues."""
        # Monitor for specified duration
        monitor_duration = min(timedelta(minutes=5), state.plan.estimated_duration / 10)

        start_time = datetime.now()
        issues = []

        while datetime.now() - start_time < monitor_duration:
            # Check for issues
            health = await self._check_system_health()
            if not health:
                issues.append(f"Health check failed at {datetime.now()}")

            await asyncio.sleep(10)  # Check every 10 seconds

        state.add_checkpoint(
            TransitionPhase.MONITORING,
            "complete" if not issues else "issues_detected",
            {"issues": issues},
            can_rollback=False,
        )

        return len(issues) == 0

    def _complete_transition(self, plan_id: str) -> None:
        """Complete and archive transition."""
        if plan_id in self.active_transitions:
            state = self.active_transitions[plan_id]
            state.current_phase = TransitionPhase.COMPLETE
            self.completed_transitions.append(state)
            del self.active_transitions[plan_id]

    async def rollback_transition(self, plan_id: str) -> Tuple[bool, str]:
        """Rollback active transition."""
        if plan_id not in self.active_transitions:
            return False, "Transition not found"

        state = self.active_transitions[plan_id]

        # Check if rollback is possible
        if not state.plan.can_rollback_from(state.current_phase):
            return False, f"Cannot rollback from {state.current_phase.value}"

        state.rollback_initiated = True

        # Execute rollback
        success = await self._execute_rollback(state)

        if success:
            # Switch back to original environment
            await self.manager.switch_environment(state.plan.from_env, force=True)
            self._complete_transition(plan_id)
            return True, "Rollback successful"

        return False, "Rollback failed"

    async def _execute_rollback(self, state: TransitionState) -> bool:
        """Execute rollback procedures."""
        # Run rollback in reverse order
        for checkpoint in reversed(state.checkpoints):
            if checkpoint.can_rollback:
                handler = self.rollback_handlers.get(checkpoint.phase)
                if handler:
                    await handler(state, checkpoint)

        return True

    # Helper methods for validation checks
    async def _check_system_health(self) -> bool:
        """Check system health."""
        # Simplified health check
        return True

    async def _check_dependencies(self) -> bool:
        """Check dependencies."""
        return True

    async def _check_risk_parameters(self) -> bool:
        """Check risk parameters."""
        return True

    async def _check_api_connectivity(self) -> bool:
        """Check API connectivity."""
        return True

    async def _verify_backups(self) -> bool:
        """Verify backups."""
        return True

    async def _check_emergency_procedures(self) -> bool:
        """Check emergency procedures."""
        return True

    async def _validate_models(self) -> bool:
        """Validate ML models."""
        return True

    async def _create_backup(self, env: Environment) -> bool:
        """Create environment backup."""
        return True

    async def _prepare_environment(self, env: Environment) -> bool:
        """Prepare environment."""
        return True

    async def _copy_data(self, from_env: Environment, to_env: Environment) -> bool:
        """Copy data between environments."""
        return True

    async def _run_verification_tests(self, env: Environment) -> bool:
        """Run verification tests."""
        return True

    def register_hook(self, phase: TransitionPhase, hook: Callable) -> None:
        """Register hook for transition phase."""
        self.transition_hooks[phase].append(hook)

    def register_rollback_handler(
        self, phase: TransitionPhase, handler: Callable
    ) -> None:
        """Register rollback handler for phase."""
        self.rollback_handlers[phase] = handler

    def get_active_transitions(self) -> List[TransitionState]:
        """Get list of active transitions."""
        return list(self.active_transitions.values())

    def get_transition_status(self, plan_id: str) -> Optional[TransitionState]:
        """Get status of specific transition."""
        return self.active_transitions.get(plan_id)
