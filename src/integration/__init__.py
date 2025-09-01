"""
Integration Module

Bridges synthetic data generation and training systems with 
the existing Redis-based agent infrastructure.
"""

from .training_bridge import (
    TrainingBridge,
    TrainingMode,
    TrainingSession,
    TrainingConfig
)
from .synthetic_injector import (
    SyntheticDataInjector,
    InjectionConfig,
    EventTiming
)
from .training_harness import (
    AgentTrainingHarness,
    TrainingScenario,
    HarnessConfig
)
# from .decision_tracker import (
#     DecisionTracker,
#     DecisionRecord,
#     TrackingConfig
# )
# from .training_controller import (
#     TrainingModeController,
#     ModeConfig,
#     DataSourceMode
# )

__all__ = [
    'TrainingBridge',
    'TrainingMode',
    'TrainingSession',
    'TrainingConfig',
    'SyntheticDataInjector',
    'InjectionConfig',
    'EventTiming',
    'AgentTrainingHarness',
    'TrainingScenario',
    'HarnessConfig',
    'DecisionTracker',
    'DecisionRecord',
    'TrackingConfig',
    'TrainingModeController',
    'ModeConfig',
    'DataSourceMode'
]