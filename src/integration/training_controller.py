"""
Training Mode Controller

Centralized controller for managing training/production mode transitions
across all agents in the system. Coordinates data sources, enables/disables
training features, and ensures smooth transitions without service interruption.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class DataSourceMode(Enum):
    """System operational modes"""
    TRAINING = "training"          # Full training with synthetic data
    PRODUCTION = "production"      # Live trading with real data  
    HYBRID = "hybrid"             # Mixed mode for transitions
    BACKTESTING = "backtesting"   # Historical data replay
    DEVELOPMENT = "development"    # Development/debugging
    MAINTENANCE = "maintenance"    # Read-only maintenance


class TransitionState(Enum):
    """States during mode transition"""
    IDLE = "idle"
    PREPARING = "preparing"
    TRANSITIONING = "transitioning"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    ROLLING_BACK = "rolling_back"
    FAILED = "failed"


class AgentState(Enum):
    """Individual agent states"""
    REGISTERED = "registered"
    READY = "ready"
    TRANSITIONING = "transitioning"
    ACTIVE = "active"
    ERROR = "error"
    DISCONNECTED = "disconnected"


@dataclass
class ModeConfig:
    """Configuration for mode management"""
    # Mode settings
    default_mode: DataSourceMode = DataSourceMode.DEVELOPMENT
    allow_hybrid: bool = True
    allow_partial_transitions: bool = True
    transition_delay_seconds: float = 2.0
    
    # Data source configuration
    training_data_sources: List[str] = field(default_factory=lambda: ["synthetic", "cached"])
    production_data_sources: List[str] = field(default_factory=lambda: ["live", "realtime"])
    hybrid_data_ratio: float = 0.5  # Ratio of synthetic to live data
    
    # Feature toggles
    enable_decision_tracking: bool = True
    enable_confidence_calibration: bool = True
    enable_memory_storage: bool = True
    enable_performance_monitoring: bool = True
    enable_replay_analysis: bool = False
    
    # Safety settings
    require_confirmation: bool = True
    auto_rollback_on_error: bool = True
    max_transition_attempts: int = 3
    health_check_interval: float = 5.0
    error_threshold_percent: float = 5.0
    min_healthy_agents_percent: float = 95.0
    
    # Timeouts
    agent_response_timeout: float = 10.0
    transition_timeout: float = 60.0
    rollback_timeout: float = 30.0
    
    # Redis settings
    redis_url: str = "redis://localhost:6379"
    mode_channel: str = "mode:controller"
    agent_channel_prefix: str = "agent:mode"
    
    # Agent-specific overrides
    agent_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def get_agent_config(self, agent_id: str) -> Dict[str, Any]:
        """Get configuration for specific agent with overrides"""
        base_config = asdict(self)
        if agent_id in self.agent_overrides:
            base_config.update(self.agent_overrides[agent_id])
        return base_config


@dataclass
class AgentInfo:
    """Information about a registered agent"""
    agent_id: str
    agent_type: str
    current_mode: DataSourceMode
    state: AgentState
    registered_at: datetime
    last_heartbeat: datetime
    capabilities: Set[str]
    health_score: float = 1.0
    error_count: int = 0
    transition_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_healthy(self) -> bool:
        """Check if agent is healthy"""
        return (
            self.state in [AgentState.READY, AgentState.ACTIVE] and
            self.health_score > 0.8 and
            (datetime.now() - self.last_heartbeat).total_seconds() < 30
        )


@dataclass
class TransitionPlan:
    """Plan for mode transition"""
    transition_id: str
    from_mode: DataSourceMode
    to_mode: DataSourceMode
    agents: List[str]
    groups: List[List[str]]  # Agents grouped for gradual rollout
    started_at: datetime
    estimated_duration: float
    rollback_checkpoint: Dict[str, Any]
    
    # Progress tracking
    completed_agents: Set[str] = field(default_factory=set)
    failed_agents: Set[str] = field(default_factory=set)
    current_group: int = 0
    
    @property
    def progress_percent(self) -> float:
        """Calculate transition progress"""
        if not self.agents:
            return 100.0
        return len(self.completed_agents) / len(self.agents) * 100


@dataclass
class ModeStatus:
    """Current system mode status"""
    global_mode: DataSourceMode
    transition_state: TransitionState
    agent_modes: Dict[str, DataSourceMode]
    healthy_agents: int
    total_agents: int
    last_transition: Optional[datetime]
    active_transition: Optional[TransitionPlan]
    
    @property
    def health_percent(self) -> float:
        """Calculate system health percentage"""
        if self.total_agents == 0:
            return 100.0
        return self.healthy_agents / self.total_agents * 100


class TrainingModeController:
    """
    Central controller for managing training/production modes.
    
    Coordinates mode transitions across all agents, manages data source
    switching, and ensures system stability during transitions.
    """
    
    def __init__(self, config: ModeConfig):
        self.config = config
        self.redis_client: Optional[redis.Redis] = None
        
        # Current state
        self.global_mode = config.default_mode
        self.transition_state = TransitionState.IDLE
        
        # Agent registry
        self.agents: Dict[str, AgentInfo] = {}
        self.agent_modes: Dict[str, DataSourceMode] = {}
        
        # Transition management
        self.active_transition: Optional[TransitionPlan] = None
        self.transition_history: List[TransitionPlan] = []
        
        # Callbacks
        self.mode_change_callbacks: List[Callable] = []
        self.transition_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []
        
        # Background tasks
        self.health_monitor_task: Optional[asyncio.Task] = None
        self.heartbeat_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.transition_count = 0
        self.rollback_count = 0
        self.error_count = 0
        
    async def initialize(self):
        """Initialize the mode controller"""
        # Connect to Redis
        self.redis_client = redis.from_url(self.config.redis_url)
        
        # Start background tasks
        self.health_monitor_task = asyncio.create_task(self._monitor_health())
        self.heartbeat_task = asyncio.create_task(self._process_heartbeats())
        
        # Subscribe to agent channels
        asyncio.create_task(self._subscribe_to_agents())
        
        logger.info(f"Training Mode Controller initialized in {self.global_mode.value} mode")
        
    async def register_agent(
        self,
        agent_id: str,
        agent_type: str,
        capabilities: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Register an agent with the controller.
        
        Args:
            agent_id: Unique agent identifier
            agent_type: Type of agent
            capabilities: Agent capabilities
            metadata: Additional metadata
            
        Returns:
            Success status
        """
        if agent_id in self.agents:
            logger.warning(f"Agent {agent_id} already registered")
            return False
            
        agent_info = AgentInfo(
            agent_id=agent_id,
            agent_type=agent_type,
            current_mode=self.global_mode,
            state=AgentState.REGISTERED,
            registered_at=datetime.now(),
            last_heartbeat=datetime.now(),
            capabilities=capabilities or set(),
            metadata=metadata or {}
        )
        
        self.agents[agent_id] = agent_info
        self.agent_modes[agent_id] = self.global_mode
        
        # Notify agent of current mode
        await self._notify_agent_mode(agent_id, self.global_mode)
        
        logger.info(f"Registered agent {agent_id} ({agent_type})")
        return True
        
    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent"""
        if agent_id not in self.agents:
            return False
            
        del self.agents[agent_id]
        del self.agent_modes[agent_id]
        
        logger.info(f"Unregistered agent {agent_id}")
        return True
        
    async def set_global_mode(
        self,
        new_mode: DataSourceMode,
        force: bool = False,
        gradual: bool = True
    ) -> Tuple[bool, str]:
        """
        Set global system mode.
        
        Args:
            new_mode: Target mode
            force: Force transition even with unhealthy agents
            gradual: Use gradual rollout
            
        Returns:
            Success status and message
        """
        # Validate transition
        if not self._is_valid_transition(self.global_mode, new_mode):
            return False, f"Invalid transition from {self.global_mode.value} to {new_mode.value}"
            
        # Check system health
        if not force:
            health_check = await self._check_system_health()
            if not health_check['healthy']:
                return False, f"System health check failed: {health_check['reason']}"
                
        # Require confirmation if configured
        if self.config.require_confirmation and not force:
            # In production, this would wait for operator confirmation
            logger.warning(f"Mode transition to {new_mode.value} requires confirmation")
            
        # Create transition plan
        plan = self._create_transition_plan(
            self.global_mode,
            new_mode,
            list(self.agents.keys()),
            gradual
        )
        
        # Execute transition
        success = await self._execute_transition(plan)
        
        if success:
            self.global_mode = new_mode
            self.transition_count += 1
            return True, f"Successfully transitioned to {new_mode.value}"
        else:
            return False, f"Transition to {new_mode.value} failed"
            
    async def set_agent_mode(
        self,
        agent_id: str,
        new_mode: DataSourceMode
    ) -> Tuple[bool, str]:
        """
        Set mode for individual agent.
        
        Args:
            agent_id: Agent to update
            new_mode: Target mode
            
        Returns:
            Success status and message
        """
        if agent_id not in self.agents:
            return False, f"Agent {agent_id} not registered"
            
        agent = self.agents[agent_id]
        
        # Check if agent supports mode
        if not self._agent_supports_mode(agent, new_mode):
            return False, f"Agent {agent_id} does not support {new_mode.value} mode"
            
        # Transition single agent
        success = await self._transition_agent(agent_id, new_mode)
        
        if success:
            agent.current_mode = new_mode
            self.agent_modes[agent_id] = new_mode
            return True, f"Agent {agent_id} transitioned to {new_mode.value}"
        else:
            return False, f"Failed to transition agent {agent_id}"
            
    async def get_mode_status(self) -> ModeStatus:
        """Get current mode status"""
        healthy_count = sum(1 for a in self.agents.values() if a.is_healthy())
        
        return ModeStatus(
            global_mode=self.global_mode,
            transition_state=self.transition_state,
            agent_modes=self.agent_modes.copy(),
            healthy_agents=healthy_count,
            total_agents=len(self.agents),
            last_transition=self.transition_history[-1].started_at if self.transition_history else None,
            active_transition=self.active_transition
        )
        
    async def transition_to_training(
        self,
        session_id: Optional[str] = None,
        agents: Optional[List[str]] = None
    ) -> Tuple[bool, str]:
        """
        Orchestrated transition to training mode.
        
        Args:
            session_id: Training session ID
            agents: Specific agents to transition (None for all)
            
        Returns:
            Success status and message
        """
        # Prepare training environment
        await self._prepare_training_environment(session_id)
        
        # Switch to training mode
        target_agents = agents or list(self.agents.keys())
        
        # Transition in groups for safety
        success = await self._gradual_mode_transition(
            target_agents,
            DataSourceMode.TRAINING,
            session_id
        )
        
        if success:
            # Enable training features
            await self._enable_training_features()
            return True, "Successfully transitioned to training mode"
        else:
            return False, "Training mode transition failed"
            
    async def transition_to_production(
        self,
        validation_required: bool = True
    ) -> Tuple[bool, str]:
        """
        Orchestrated transition to production mode.
        
        Args:
            validation_required: Require validation before transition
            
        Returns:
            Success status and message
        """
        # Validate readiness
        if validation_required:
            validation = await self._validate_production_readiness()
            if not validation['ready']:
                return False, f"Production validation failed: {validation['reason']}"
                
        # Disable training features
        await self._disable_training_features()
        
        # Switch to production mode
        success = await self._gradual_mode_transition(
            list(self.agents.keys()),
            DataSourceMode.PRODUCTION,
            None
        )
        
        if success:
            return True, "Successfully transitioned to production mode"
        else:
            # Rollback if failed
            await self.emergency_rollback()
            return False, "Production transition failed, rolled back"
            
    async def emergency_rollback(self) -> bool:
        """
        Emergency rollback to previous stable state.
        
        Returns:
            Success status
        """
        if not self.active_transition:
            logger.warning("No active transition to rollback")
            return False
            
        logger.warning("Initiating emergency rollback")
        self.transition_state = TransitionState.ROLLING_BACK
        
        # Get rollback checkpoint
        checkpoint = self.active_transition.rollback_checkpoint
        
        try:
            # Restore agent modes
            for agent_id, mode in checkpoint['agent_modes'].items():
                await self._transition_agent(agent_id, mode)
                
            # Restore global mode
            self.global_mode = checkpoint['global_mode']
            
            # Restore configurations
            await self._restore_configurations(checkpoint)
            
            self.rollback_count += 1
            self.transition_state = TransitionState.IDLE
            self.active_transition = None
            
            logger.info("Emergency rollback completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Emergency rollback failed: {e}")
            self.transition_state = TransitionState.FAILED
            return False
            
    def add_mode_change_callback(self, callback: Callable):
        """Add callback for mode changes"""
        self.mode_change_callbacks.append(callback)
        
    def add_transition_callback(self, callback: Callable):
        """Add callback for transitions"""
        self.transition_callbacks.append(callback)
        
    def add_error_callback(self, callback: Callable):
        """Add callback for errors"""
        self.error_callbacks.append(callback)
        
    async def _execute_transition(self, plan: TransitionPlan) -> bool:
        """Execute a transition plan"""
        self.active_transition = plan
        self.transition_state = TransitionState.PREPARING
        
        try:
            # Pre-transition preparation
            await self._prepare_transition(plan)
            
            self.transition_state = TransitionState.TRANSITIONING
            
            # Execute group by group
            for group_idx, group in enumerate(plan.groups):
                plan.current_group = group_idx
                
                # Transition group
                success = await self._transition_group(group, plan.to_mode)
                
                if not success:
                    # Check if we should rollback
                    if self.config.auto_rollback_on_error:
                        await self.emergency_rollback()
                        return False
                        
                # Delay between groups
                await asyncio.sleep(self.config.transition_delay_seconds)
                
                # Health check after each group
                health = await self._check_group_health(group)
                if health < self.config.min_healthy_agents_percent:
                    logger.error(f"Group {group_idx} health check failed")
                    if self.config.auto_rollback_on_error:
                        await self.emergency_rollback()
                        return False
                        
            # Verify transition
            self.transition_state = TransitionState.VERIFYING
            verification = await self._verify_transition(plan)
            
            if verification:
                self.transition_state = TransitionState.COMPLETED
                self.transition_history.append(plan)
                self.active_transition = None
                
                # Trigger callbacks
                for callback in self.transition_callbacks:
                    asyncio.create_task(callback(plan))
                    
                return True
            else:
                if self.config.auto_rollback_on_error:
                    await self.emergency_rollback()
                return False
                
        except Exception as e:
            logger.error(f"Transition failed: {e}")
            self.error_count += 1
            
            # Trigger error callbacks
            for callback in self.error_callbacks:
                asyncio.create_task(callback(e))
                
            if self.config.auto_rollback_on_error:
                await self.emergency_rollback()
            return False
            
    async def _transition_agent(
        self,
        agent_id: str,
        new_mode: DataSourceMode
    ) -> bool:
        """Transition individual agent"""
        if agent_id not in self.agents:
            return False
            
        agent = self.agents[agent_id]
        agent.state = AgentState.TRANSITIONING
        
        try:
            # Notify agent
            await self._notify_agent_mode(agent_id, new_mode)
            
            # Wait for acknowledgment
            ack = await self._wait_for_agent_ack(agent_id)
            
            if ack:
                agent.current_mode = new_mode
                agent.state = AgentState.ACTIVE
                agent.transition_count += 1
                
                # Trigger callbacks
                for callback in self.mode_change_callbacks:
                    asyncio.create_task(callback(agent_id, new_mode))
                    
                return True
            else:
                agent.state = AgentState.ERROR
                agent.error_count += 1
                return False
                
        except asyncio.TimeoutError:
            logger.error(f"Agent {agent_id} transition timeout")
            agent.state = AgentState.ERROR
            return False
            
    async def _transition_group(
        self,
        group: List[str],
        new_mode: DataSourceMode
    ) -> bool:
        """Transition a group of agents"""
        tasks = []
        for agent_id in group:
            task = asyncio.create_task(self._transition_agent(agent_id, new_mode))
            tasks.append(task)
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check success rate
        success_count = sum(1 for r in results if r is True)
        success_rate = success_count / len(group) * 100 if group else 100
        
        if success_rate >= self.config.min_healthy_agents_percent:
            return True
        else:
            logger.error(f"Group transition failed: {success_rate:.1f}% success rate")
            return False
            
    async def _gradual_mode_transition(
        self,
        agents: List[str],
        new_mode: DataSourceMode,
        session_id: Optional[str]
    ) -> bool:
        """Gradual mode transition with monitoring"""
        # Create groups for gradual rollout
        group_size = max(1, len(agents) // 5)  # 20% at a time
        groups = [agents[i:i+group_size] for i in range(0, len(agents), group_size)]
        
        plan = self._create_transition_plan(
            self.global_mode,
            new_mode,
            agents,
            gradual=True
        )
        
        return await self._execute_transition(plan)
        
    def _create_transition_plan(
        self,
        from_mode: DataSourceMode,
        to_mode: DataSourceMode,
        agents: List[str],
        gradual: bool
    ) -> TransitionPlan:
        """Create a transition plan"""
        # Create checkpoint for rollback
        checkpoint = {
            'global_mode': self.global_mode,
            'agent_modes': self.agent_modes.copy(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Group agents for gradual rollout
        if gradual and len(agents) > 5:
            group_size = max(1, len(agents) // 5)
            groups = [agents[i:i+group_size] for i in range(0, len(agents), group_size)]
        else:
            groups = [agents]  # Single group
            
        plan = TransitionPlan(
            transition_id=f"transition_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            from_mode=from_mode,
            to_mode=to_mode,
            agents=agents,
            groups=groups,
            started_at=datetime.now(),
            estimated_duration=len(groups) * self.config.transition_delay_seconds,
            rollback_checkpoint=checkpoint
        )
        
        return plan
        
    async def _notify_agent_mode(self, agent_id: str, mode: DataSourceMode):
        """Notify agent of mode change"""
        if not self.redis_client:
            return
            
        message = {
            'type': 'mode_change',
            'agent_id': agent_id,
            'mode': mode.value,
            'timestamp': datetime.now().isoformat(),
            'config': self.config.get_agent_config(agent_id)
        }
        
        channel = f"{self.config.agent_channel_prefix}:{agent_id}"
        await self.redis_client.publish(channel, json.dumps(message))
        
    async def _wait_for_agent_ack(
        self,
        agent_id: str,
        timeout: Optional[float] = None
    ) -> bool:
        """Wait for agent acknowledgment"""
        timeout = timeout or self.config.agent_response_timeout
        
        # In production, would wait for Redis response
        # For now, simulate with delay
        await asyncio.sleep(0.1)
        
        # Check if agent updated heartbeat
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            if (datetime.now() - agent.last_heartbeat).total_seconds() < timeout:
                return True
                
        return False
        
    async def _check_system_health(self) -> Dict[str, Any]:
        """Check overall system health"""
        total_agents = len(self.agents)
        if total_agents == 0:
            return {'healthy': True, 'reason': 'No agents registered'}
            
        healthy_agents = sum(1 for a in self.agents.values() if a.is_healthy())
        health_percent = healthy_agents / total_agents * 100
        
        if health_percent < self.config.min_healthy_agents_percent:
            return {
                'healthy': False,
                'reason': f'Only {health_percent:.1f}% agents healthy'
            }
            
        return {'healthy': True, 'health_percent': health_percent}
        
    async def _check_group_health(self, group: List[str]) -> float:
        """Check health of agent group"""
        if not group:
            return 100.0
            
        healthy = sum(1 for aid in group if aid in self.agents and self.agents[aid].is_healthy())
        return healthy / len(group) * 100
        
    async def _prepare_transition(self, plan: TransitionPlan):
        """Prepare for transition"""
        logger.info(f"Preparing transition {plan.transition_id}")
        
        # Notify all agents of upcoming transition
        for agent_id in plan.agents:
            await self._notify_agent_prepare(agent_id, plan)
            
        # Warm up caches if needed
        if plan.to_mode == DataSourceMode.PRODUCTION:
            await self._warm_production_caches()
        elif plan.to_mode == DataSourceMode.TRAINING:
            await self._prepare_training_data()
            
    async def _verify_transition(self, plan: TransitionPlan) -> bool:
        """Verify transition completed successfully"""
        # Check all agents transitioned
        for agent_id in plan.agents:
            if agent_id not in self.agents:
                continue
                
            agent = self.agents[agent_id]
            if agent.current_mode != plan.to_mode:
                logger.error(f"Agent {agent_id} failed to transition")
                return False
                
        # Check system health
        health = await self._check_system_health()
        if not health['healthy']:
            logger.error(f"Post-transition health check failed: {health['reason']}")
            return False
            
        return True
        
    async def _prepare_training_environment(self, session_id: Optional[str]):
        """Prepare training environment"""
        logger.info(f"Preparing training environment (session: {session_id})")
        
        # Would coordinate with training systems
        # Enable synthetic data generation
        # Configure decision tracking
        # Set up performance monitoring
        
    async def _enable_training_features(self):
        """Enable training-specific features"""
        # Enable decision tracking
        if self.config.enable_decision_tracking:
            await self._enable_decision_tracking()
            
        # Enable confidence calibration
        if self.config.enable_confidence_calibration:
            await self._enable_confidence_calibration()
            
        # Enable memory storage
        if self.config.enable_memory_storage:
            await self._enable_memory_storage()
            
    async def _disable_training_features(self):
        """Disable training-specific features"""
        # Disable non-production features
        await self._disable_decision_tracking()
        await self._disable_replay_analysis()
        
    async def _validate_production_readiness(self) -> Dict[str, Any]:
        """Validate system is ready for production"""
        checks = {
            'agents_ready': True,
            'data_sources_available': True,
            'risk_limits_set': True,
            'monitoring_active': True
        }
        
        # Check all agents are healthy
        health = await self._check_system_health()
        if not health['healthy']:
            return {'ready': False, 'reason': 'System health check failed'}
            
        # Verify production data sources
        # Check risk management systems
        # Verify monitoring is active
        
        return {'ready': True, 'checks': checks}
        
    async def _restore_configurations(self, checkpoint: Dict[str, Any]):
        """Restore configurations from checkpoint"""
        # Restore agent configurations
        # Restore data source settings
        # Restore feature toggles
        pass
        
    async def _warm_production_caches(self):
        """Warm up production caches"""
        # Pre-load market data
        # Initialize connections
        # Warm decision caches
        pass
        
    async def _prepare_training_data(self):
        """Prepare training data"""
        # Load synthetic data
        # Initialize generators
        # Set up replay data
        pass
        
    async def _notify_agent_prepare(self, agent_id: str, plan: TransitionPlan):
        """Notify agent to prepare for transition"""
        if not self.redis_client:
            return
            
        message = {
            'type': 'prepare_transition',
            'agent_id': agent_id,
            'transition_id': plan.transition_id,
            'from_mode': plan.from_mode.value,
            'to_mode': plan.to_mode.value,
            'estimated_start': (datetime.now() + timedelta(seconds=5)).isoformat()
        }
        
        channel = f"{self.config.agent_channel_prefix}:{agent_id}"
        await self.redis_client.publish(channel, json.dumps(message))
        
    async def _enable_decision_tracking(self):
        """Enable decision tracking for all agents"""
        # Would coordinate with DecisionTracker
        pass
        
    async def _disable_decision_tracking(self):
        """Disable decision tracking"""
        # Would coordinate with DecisionTracker
        pass
        
    async def _enable_confidence_calibration(self):
        """Enable confidence calibration"""
        # Would coordinate with ConfidenceCalibrator
        pass
        
    async def _enable_memory_storage(self):
        """Enable memory storage"""
        # Would coordinate with MemorySystem
        pass
        
    async def _disable_replay_analysis(self):
        """Disable replay analysis"""
        # Would coordinate with ReplayEngine
        pass
        
    def _is_valid_transition(
        self,
        from_mode: DataSourceMode,
        to_mode: DataSourceMode
    ) -> bool:
        """Check if transition is valid"""
        # Define valid transitions
        valid_transitions = {
            DataSourceMode.DEVELOPMENT: [DataSourceMode.TRAINING, DataSourceMode.BACKTESTING],
            DataSourceMode.TRAINING: [DataSourceMode.DEVELOPMENT, DataSourceMode.HYBRID, DataSourceMode.BACKTESTING],
            DataSourceMode.HYBRID: [DataSourceMode.TRAINING, DataSourceMode.PRODUCTION],
            DataSourceMode.PRODUCTION: [DataSourceMode.HYBRID, DataSourceMode.MAINTENANCE],
            DataSourceMode.MAINTENANCE: [DataSourceMode.PRODUCTION, DataSourceMode.DEVELOPMENT],
            DataSourceMode.BACKTESTING: [DataSourceMode.DEVELOPMENT, DataSourceMode.TRAINING]
        }
        
        return to_mode in valid_transitions.get(from_mode, [])
        
    def _agent_supports_mode(self, agent: AgentInfo, mode: DataSourceMode) -> bool:
        """Check if agent supports mode"""
        # Check agent capabilities
        mode_capabilities = {
            DataSourceMode.TRAINING: {'training', 'synthetic_data'},
            DataSourceMode.PRODUCTION: {'trading', 'live_data'},
            DataSourceMode.HYBRID: {'training', 'trading'},
            DataSourceMode.BACKTESTING: {'replay', 'historical_data'}
        }
        
        required = mode_capabilities.get(mode, set())
        return required.issubset(agent.capabilities)
        
    async def _monitor_health(self):
        """Background task to monitor system health"""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                # Update agent health scores
                for agent in self.agents.values():
                    heartbeat_age = (datetime.now() - agent.last_heartbeat).total_seconds()
                    
                    if heartbeat_age > 60:
                        agent.state = AgentState.DISCONNECTED
                        agent.health_score = 0
                    elif heartbeat_age > 30:
                        agent.health_score = max(0, agent.health_score - 0.1)
                    else:
                        agent.health_score = min(1.0, agent.health_score + 0.05)
                        
                # Check for stuck transitions
                if self.active_transition:
                    duration = (datetime.now() - self.active_transition.started_at).total_seconds()
                    if duration > self.config.transition_timeout:
                        logger.error("Transition timeout, initiating rollback")
                        await self.emergency_rollback()
                        
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                
    async def _process_heartbeats(self):
        """Process agent heartbeats"""
        if not self.redis_client:
            return
            
        pubsub = self.redis_client.pubsub()
        await pubsub.subscribe("agent:heartbeat:*")
        
        async for message in pubsub.listen():
            if message['type'] == 'message':
                try:
                    data = json.loads(message['data'])
                    agent_id = data.get('agent_id')
                    
                    if agent_id in self.agents:
                        self.agents[agent_id].last_heartbeat = datetime.now()
                        
                except Exception as e:
                    logger.error(f"Heartbeat processing error: {e}")
                    
    async def _subscribe_to_agents(self):
        """Subscribe to agent status updates"""
        if not self.redis_client:
            return
            
        pubsub = self.redis_client.pubsub()
        await pubsub.subscribe("agent:status:*")
        
        async for message in pubsub.listen():
            if message['type'] == 'message':
                try:
                    data = json.loads(message['data'])
                    agent_id = data.get('agent_id')
                    
                    if agent_id in self.agents:
                        # Update agent status
                        if 'state' in data:
                            self.agents[agent_id].state = AgentState(data['state'])
                        if 'mode' in data:
                            self.agents[agent_id].current_mode = DataSourceMode(data['mode'])
                            
                except Exception as e:
                    logger.error(f"Agent status processing error: {e}")
                    
    async def cleanup(self):
        """Clean up resources"""
        # Cancel background tasks
        if self.health_monitor_task:
            self.health_monitor_task.cancel()
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
            
        logger.info("Training Mode Controller cleaned up")