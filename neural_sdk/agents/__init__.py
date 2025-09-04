"""
Agent framework for the Neural Trading SDK.

This module provides the core agent architecture including:
- BaseAgentRedisConsumer: Core agent with Redis pub/sub
- AgentOrchestrator: Multi-agent coordination
- Always-on agents: DataCoordinator, PortfolioMonitor
- On-demand agents: GameAnalyst, ArbitrageHunter
- TriggerService: Agent activation coordination
"""

from .base import AgentOrchestrator, BaseAgentRedisConsumer
from .types import AgentStatus, AgentType, TriggerPriority

__all__ = [
    "BaseAgentRedisConsumer",
    "AgentOrchestrator",
    "AgentType",
    "AgentStatus",
    "TriggerPriority",
]
