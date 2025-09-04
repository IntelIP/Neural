"""
Type definitions and enums for the agent framework.

This module defines the core types, enums, and data structures
used throughout the agent system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class AgentType(Enum):
    """Types of agents in the system."""

    DATA_COORDINATOR = "data_coordinator"
    PORTFOLIO_MONITOR = "portfolio_monitor"
    GAME_ANALYST = "game_analyst"
    ARBITRAGE_HUNTER = "arbitrage_hunter"
    STRATEGY_OPTIMIZER = "strategy_optimizer"
    RISK_MANAGER = "risk_manager"
    MARKET_ENGINEER = "market_engineer"


class AgentStatus(Enum):
    """Status of an agent."""

    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class TriggerPriority(Enum):
    """Priority levels for agent activation triggers."""

    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


@dataclass
class AgentStats:
    """Statistics for an agent."""

    agent_name: str
    status: AgentStatus
    messages_received: int = 0
    messages_processed: int = 0
    messages_published: int = 0
    last_message_time: Optional[datetime] = None
    uptime_seconds: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None

    @property
    def processing_rate(self) -> float:
        """Calculate messages processed per second."""
        if self.uptime_seconds == 0:
            return 0.0
        return self.messages_processed / self.uptime_seconds


@dataclass
class TriggerCondition:
    """Represents a trigger condition for agent activation."""

    name: str
    condition_func: Callable[[Dict[str, Any]], bool]
    agent_type: AgentType
    priority: TriggerPriority
    cooldown_seconds: int = 60
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0

    def can_trigger(self) -> bool:
        """Check if trigger is off cooldown."""
        if self.last_triggered is None:
            return True

        elapsed = (datetime.now() - self.last_triggered).total_seconds()
        return elapsed >= self.cooldown_seconds

    def evaluate(self, data: Dict[str, Any]) -> bool:
        """Evaluate if condition is met."""
        if not self.can_trigger():
            return False

        try:
            return self.condition_func(data)
        except Exception:
            return False

    def mark_triggered(self):
        """Mark this trigger as having fired."""
        self.last_triggered = datetime.now()
        self.trigger_count += 1


@dataclass
class AgentMessage:
    """Standardized message format for agent communication."""

    message_type: str
    source_agent: str
    target_agent: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None
    priority: TriggerPriority = TriggerPriority.MEDIUM


@dataclass
class AgentConfig:
    """Configuration for an individual agent."""

    agent_type: AgentType
    name: str
    redis_url: str = "redis://localhost:6379"
    channels: List[str] = field(default_factory=list)
    max_concurrent_tasks: int = 10
    enable_health_checks: bool = True
    health_check_interval: int = 30
    enable_metrics: bool = True
    custom_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestratorConfig:
    """Configuration for the agent orchestrator."""

    redis_url: str = "redis://localhost:6379"
    enable_auto_restart: bool = True
    max_restart_attempts: int = 3
    restart_delay_seconds: int = 5
    enable_health_monitoring: bool = True
    health_check_interval: int = 30
    enable_metrics_collection: bool = True
    metrics_retention_hours: int = 24
