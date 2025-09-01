"""
Adaptive Scheduler for Hybrid Data Pipeline

Intelligently schedules data requests and training sessions based on
agent performance, cost optimization, and system resources.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import logging
import json
from collections import defaultdict, deque
import heapq
import random

from .data_orchestrator import DataSource, DataMode, HybridDataOrchestrator
from .cost_monitor import BudgetManager, APITracker
from ..training.agent_analytics import AgentAnalytics


class TrainingPhase(Enum):
    """Different phases of agent training"""
    INITIAL = "initial"
    FOUNDATION = "foundation"
    ADVANCED = "advanced"
    VALIDATION = "validation"
    PRODUCTION_PREP = "production_prep"
    MAINTENANCE = "maintenance"


class SchedulingPolicy(Enum):
    """Scheduling policy options"""
    COST_OPTIMIZED = "cost_optimized"
    PERFORMANCE_OPTIMIZED = "performance_optimized"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"


class Priority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class TrainingTask:
    """Individual training task definition"""
    task_id: str
    agent_id: str
    task_type: str  # "scenario_training", "validation", "edge_case_practice"
    priority: Priority
    estimated_cost: float
    estimated_duration: timedelta
    data_requirements: Dict[str, Any]
    preferred_data_source: DataSource
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"  # pending, scheduled, running, completed, failed
    retry_count: int = 0
    max_retries: int = 3
    
    def __lt__(self, other):
        """For priority queue ordering"""
        return (self.priority.value, self.scheduled_at or datetime.max) > (other.priority.value, other.scheduled_at or datetime.max)


@dataclass
class SchedulingContext:
    """Context for scheduling decisions"""
    current_time: datetime
    available_budget: float
    system_load: float
    api_rate_limits: Dict[str, int]
    agent_performance_metrics: Dict[str, Dict]
    active_tasks: List[TrainingTask]
    failed_tasks: List[TrainingTask]
    time_window: timedelta = timedelta(hours=24)


class AdaptiveScheduler:
    """
    Adaptive scheduler that optimizes training task scheduling based on:
    - Agent performance and learning needs
    - API cost constraints and budget optimization
    - System resources and capacity
    - Training phase requirements
    - Historical performance data
    """
    
    def __init__(
        self,
        orchestrator: HybridDataOrchestrator,
        budget_manager: BudgetManager,
        api_tracker: APITracker,
        agent_analytics: AgentAnalytics
    ):
        self.orchestrator = orchestrator
        self.budget_manager = budget_manager
        self.api_tracker = api_tracker
        self.agent_analytics = agent_analytics
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.scheduling_policy = SchedulingPolicy.BALANCED
        self.max_concurrent_tasks = 5
        self.planning_horizon = timedelta(hours=24)
        self.rebalance_interval = timedelta(minutes=30)
        
        # Task management
        self.task_queue: List[TrainingTask] = []
        self.active_tasks: Dict[str, TrainingTask] = {}
        self.completed_tasks: List[TrainingTask] = []
        self.failed_tasks: List[TrainingTask] = []
        
        # Agent training phases
        self.agent_phases: Dict[str, TrainingPhase] = {}
        self.phase_requirements: Dict[TrainingPhase, Dict[str, Any]] = {
            TrainingPhase.INITIAL: {
                "synthetic_data_ratio": 1.0,
                "scenario_diversity": 0.3,
                "edge_case_ratio": 0.05,
                "validation_frequency": 0.1
            },
            TrainingPhase.FOUNDATION: {
                "synthetic_data_ratio": 0.9,
                "scenario_diversity": 0.5,
                "edge_case_ratio": 0.1,
                "validation_frequency": 0.15
            },
            TrainingPhase.ADVANCED: {
                "synthetic_data_ratio": 0.7,
                "scenario_diversity": 0.8,
                "edge_case_ratio": 0.2,
                "validation_frequency": 0.2
            },
            TrainingPhase.VALIDATION: {
                "synthetic_data_ratio": 0.3,
                "scenario_diversity": 1.0,
                "edge_case_ratio": 0.3,
                "validation_frequency": 0.5
            },
            TrainingPhase.PRODUCTION_PREP: {
                "synthetic_data_ratio": 0.1,
                "scenario_diversity": 0.6,
                "edge_case_ratio": 0.1,
                "validation_frequency": 0.8
            }
        }
        
        # Scheduling metrics
        self.scheduling_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, float] = defaultdict(float)
        
        # Task execution callbacks
        self.task_callbacks: Dict[str, Callable] = {}
        
    async def initialize(self) -> None:
        """Initialize the adaptive scheduler"""
        try:
            self.logger.info("Initializing Adaptive Scheduler")
            
            # Initialize agent phases
            await self._initialize_agent_phases()
            
            # Start background scheduling loop
            asyncio.create_task(self._scheduling_loop())
            
            self.logger.info("Adaptive Scheduler initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize scheduler: {e}")
            raise
    
    async def _initialize_agent_phases(self) -> None:
        """Initialize training phases for all agents"""
        try:
            # Get all agents from analytics
            for agent_id in self.agent_analytics.decision_history.keys():
                # Determine initial phase based on agent experience
                analytics = await self.agent_analytics.get_agent_analytics(agent_id)
                
                if analytics.get("total_decisions", 0) == 0:
                    phase = TrainingPhase.INITIAL
                elif analytics.get("total_decisions", 0) < 100:
                    phase = TrainingPhase.FOUNDATION
                elif analytics.get("latest_snapshot", {}).get("win_rate", 0) < 0.5:
                    phase = TrainingPhase.ADVANCED
                else:
                    phase = TrainingPhase.VALIDATION
                
                self.agent_phases[agent_id] = phase
                self.logger.info(f"Agent {agent_id} initialized in {phase.value} phase")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize agent phases: {e}")
    
    async def schedule_training_task(
        self,
        agent_id: str,
        task_type: str,
        priority: Priority = Priority.NORMAL,
        data_requirements: Optional[Dict[str, Any]] = None,
        preferred_time: Optional[datetime] = None
    ) -> str:
        """Schedule a new training task"""
        try:
            task_id = f"{agent_id}_{task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Get agent's current phase
            agent_phase = self.agent_phases.get(agent_id, TrainingPhase.INITIAL)
            
            # Determine optimal data source and requirements
            optimal_source, requirements = await self._determine_optimal_task_config(
                agent_id, task_type, agent_phase, data_requirements
            )
            
            # Estimate cost and duration
            estimated_cost = await self._estimate_task_cost(task_type, requirements, optimal_source)
            estimated_duration = self._estimate_task_duration(task_type, requirements)
            
            # Create task
            task = TrainingTask(
                task_id=task_id,
                agent_id=agent_id,
                task_type=task_type,
                priority=priority,
                estimated_cost=estimated_cost,
                estimated_duration=estimated_duration,
                data_requirements=requirements,
                preferred_data_source=optimal_source
            )
            
            # Schedule the task
            await self._schedule_task(task, preferred_time)
            
            self.logger.info(f"Scheduled task {task_id} for agent {agent_id}")
            return task_id
            
        except Exception as e:
            self.logger.error(f"Failed to schedule training task: {e}")
            raise
    
    async def _determine_optimal_task_config(
        self,
        agent_id: str,
        task_type: str,
        agent_phase: TrainingPhase,
        data_requirements: Optional[Dict[str, Any]]
    ) -> Tuple[DataSource, Dict[str, Any]]:
        """Determine optimal configuration for a training task"""
        try:
            # Get phase requirements
            phase_config = self.phase_requirements[agent_phase]
            
            # Base requirements from phase
            requirements = {
                "count": data_requirements.get("count", 100) if data_requirements else 100,
                "scenario_diversity": phase_config["scenario_diversity"],
                "edge_case_ratio": phase_config["edge_case_ratio"],
                "include_edge_cases": phase_config["edge_case_ratio"] > 0.1,
                "training_phase": agent_phase.value,
                "agent_id": agent_id
            }
            
            # Override with specific requirements
            if data_requirements:
                requirements.update(data_requirements)
            
            # Determine optimal data source
            synthetic_ratio = phase_config["synthetic_data_ratio"]
            
            # Adjust based on scheduling policy
            if self.scheduling_policy == SchedulingPolicy.COST_OPTIMIZED:
                # Favor synthetic data for cost savings
                synthetic_ratio = min(1.0, synthetic_ratio * 1.2)
            elif self.scheduling_policy == SchedulingPolicy.PERFORMANCE_OPTIMIZED:
                # Favor live data for realism
                synthetic_ratio = max(0.1, synthetic_ratio * 0.8)
            
            # Select data source based on ratio and current context
            if synthetic_ratio > 0.8:
                optimal_source = DataSource.SYNTHETIC
            elif synthetic_ratio < 0.3:
                optimal_source = DataSource.LIVE_API
            else:
                optimal_source = DataSource.HYBRID
            
            # Consider current budget constraints
            daily_budget_used = self.budget_manager.get_budget_status().get("daily_cost", 0)
            if daily_budget_used > 50:  # If already spent significant amount
                optimal_source = DataSource.SYNTHETIC
            
            return optimal_source, requirements
            
        except Exception as e:
            self.logger.error(f"Failed to determine optimal task config: {e}")
            return DataSource.SYNTHETIC, {"count": 100}
    
    async def _estimate_task_cost(
        self,
        task_type: str,
        requirements: Dict[str, Any],
        data_source: DataSource
    ) -> float:
        """Estimate the cost of a training task"""
        try:
            base_cost = 0.0
            data_count = requirements.get("count", 100)
            
            if data_source == DataSource.SYNTHETIC:
                base_cost = data_count * 0.001  # Very low cost for synthetic
            elif data_source == DataSource.LIVE_API:
                base_cost = data_count * 0.01   # Higher cost for live API
            elif data_source == DataSource.CACHED:
                base_cost = 0.0  # No cost for cached data
            else:  # HYBRID
                # Mix of costs
                base_cost = data_count * 0.005
            
            # Adjust for task complexity
            complexity_multiplier = 1.0
            if task_type == "edge_case_practice":
                complexity_multiplier = 1.5
            elif task_type == "validation":
                complexity_multiplier = 2.0
            
            # Adjust for special requirements
            if requirements.get("include_edge_cases", False):
                complexity_multiplier *= 1.2
            
            if requirements.get("scenario_diversity", 0) > 0.8:
                complexity_multiplier *= 1.3
            
            return base_cost * complexity_multiplier
            
        except Exception as e:
            self.logger.error(f"Failed to estimate task cost: {e}")
            return 1.0  # Default cost
    
    def _estimate_task_duration(self, task_type: str, requirements: Dict[str, Any]) -> timedelta:
        """Estimate the duration of a training task"""
        try:
            base_duration = timedelta(minutes=30)  # Base 30 minutes
            data_count = requirements.get("count", 100)
            
            # Scale with data count
            duration_multiplier = max(1.0, data_count / 100)
            
            # Adjust for task type
            if task_type == "scenario_training":
                duration_multiplier *= 1.0
            elif task_type == "validation":
                duration_multiplier *= 1.5
            elif task_type == "edge_case_practice":
                duration_multiplier *= 2.0
            
            # Adjust for complexity
            if requirements.get("scenario_diversity", 0) > 0.8:
                duration_multiplier *= 1.2
            
            return base_duration * duration_multiplier
            
        except Exception as e:
            self.logger.error(f"Failed to estimate task duration: {e}")
            return timedelta(hours=1)  # Default duration
    
    async def _schedule_task(self, task: TrainingTask, preferred_time: Optional[datetime] = None) -> None:
        """Schedule a task in the queue"""
        try:
            # Determine optimal scheduling time
            if preferred_time:
                scheduled_time = preferred_time
            else:
                scheduled_time = await self._find_optimal_schedule_time(task)
            
            task.scheduled_at = scheduled_time
            task.status = "scheduled"
            
            # Add to priority queue
            heapq.heappush(self.task_queue, task)
            
            # Record scheduling decision
            self.scheduling_history.append({
                "task_id": task.task_id,
                "scheduled_at": scheduled_time.isoformat(),
                "estimated_cost": task.estimated_cost,
                "priority": task.priority.value,
                "data_source": task.preferred_data_source.value,
                "scheduling_policy": self.scheduling_policy.value
            })
            
        except Exception as e:
            self.logger.error(f"Failed to schedule task: {e}")
            raise
    
    async def _find_optimal_schedule_time(self, task: TrainingTask) -> datetime:
        """Find optimal time to schedule a task"""
        try:
            current_time = datetime.now()
            
            # Get scheduling context
            context = await self._build_scheduling_context()
            
            # Base scheduling logic
            if task.priority == Priority.CRITICAL:
                return current_time  # Schedule immediately
            
            # Consider budget constraints
            if task.estimated_cost > context.available_budget:
                # Schedule for next budget period
                return current_time + timedelta(hours=24)
            
            # Consider system load
            if context.system_load > 0.8:
                # Schedule when load is lower
                return current_time + timedelta(hours=2)
            
            # Consider API rate limits
            if task.preferred_data_source == DataSource.LIVE_API:
                api_usage = context.api_rate_limits.get("live_api", 0)
                if api_usage > 80:  # Near rate limit
                    return current_time + timedelta(hours=1)
            
            # Policy-based scheduling
            if self.scheduling_policy == SchedulingPolicy.COST_OPTIMIZED:
                # Schedule during low-cost periods (example: off-peak hours)
                if current_time.hour >= 9 and current_time.hour <= 17:
                    return current_time.replace(hour=20, minute=0, second=0)
            
            # Default: schedule with small random delay to spread load
            delay_minutes = random.randint(0, 30)
            return current_time + timedelta(minutes=delay_minutes)
            
        except Exception as e:
            self.logger.error(f"Failed to find optimal schedule time: {e}")
            return datetime.now() + timedelta(minutes=5)  # Default short delay
    
    async def _build_scheduling_context(self) -> SchedulingContext:
        """Build context for scheduling decisions"""
        try:
            current_time = datetime.now()
            
            # Get budget information
            budget_status = self.budget_manager.get_budget_status()
            available_budget = 0.0
            if budget_status.get("budget_periods"):
                for period_data in budget_status["budget_periods"].values():
                    available_budget += period_data.get("remaining", 0)
            
            # Get system metrics
            performance_metrics = self.api_tracker.get_performance_metrics()
            system_load = performance_metrics.get("error_rate", 0) + (performance_metrics.get("avg_latency_ms", 0) / 1000)
            
            # Get API rate limit status (simplified)
            api_rate_limits = {
                "live_api": performance_metrics.get("requests_last_hour", 0),
                "synthetic": 0  # No limits for synthetic
            }
            
            # Get agent performance metrics
            agent_metrics = {}
            for agent_id in self.agent_phases.keys():
                try:
                    analytics = await self.agent_analytics.get_agent_analytics(agent_id, timedelta(hours=6))
                    agent_metrics[agent_id] = analytics
                except Exception as e:
                    self.logger.warning(f"Failed to get analytics for {agent_id}: {e}")
                    agent_metrics[agent_id] = {}
            
            return SchedulingContext(
                current_time=current_time,
                available_budget=available_budget,
                system_load=min(1.0, system_load),
                api_rate_limits=api_rate_limits,
                agent_performance_metrics=agent_metrics,
                active_tasks=list(self.active_tasks.values()),
                failed_tasks=self.failed_tasks[-10:]  # Last 10 failures
            )
            
        except Exception as e:
            self.logger.error(f"Failed to build scheduling context: {e}")
            return SchedulingContext(
                current_time=datetime.now(),
                available_budget=100.0,
                system_load=0.5,
                api_rate_limits={},
                agent_performance_metrics={},
                active_tasks=[],
                failed_tasks=[]
            )
    
    async def _scheduling_loop(self) -> None:
        """Main scheduling loop"""
        try:
            while True:
                await asyncio.sleep(60)  # Check every minute
                
                current_time = datetime.now()
                
                # Execute scheduled tasks
                await self._execute_ready_tasks(current_time)
                
                # Rebalance schedule periodically
                if hasattr(self, '_last_rebalance'):
                    if current_time - self._last_rebalance >= self.rebalance_interval:
                        await self._rebalance_schedule()
                        self._last_rebalance = current_time
                else:
                    self._last_rebalance = current_time
                
                # Update agent phases based on performance
                await self._update_agent_phases()
                
                # Clean up old completed tasks
                self._cleanup_old_tasks()
                
        except Exception as e:
            self.logger.error(f"Scheduling loop error: {e}")
            await asyncio.sleep(300)  # Wait 5 minutes before retrying
    
    async def _execute_ready_tasks(self, current_time: datetime) -> None:
        """Execute tasks that are ready to run"""
        try:
            # Check for tasks ready to execute
            ready_tasks = []
            
            while (self.task_queue and 
                   self.task_queue[0].scheduled_at and 
                   self.task_queue[0].scheduled_at <= current_time and
                   len(self.active_tasks) < self.max_concurrent_tasks):
                
                task = heapq.heappop(self.task_queue)
                ready_tasks.append(task)
            
            # Execute ready tasks
            for task in ready_tasks:
                try:
                    # Check if task is still viable (budget, etc.)
                    can_execute, reason = await self._can_execute_task(task)
                    
                    if can_execute:
                        asyncio.create_task(self._execute_task(task))
                    else:
                        self.logger.warning(f"Cannot execute task {task.task_id}: {reason}")
                        # Reschedule or fail the task
                        if task.retry_count < task.max_retries:
                            task.retry_count += 1
                            task.scheduled_at = current_time + timedelta(minutes=30)
                            heapq.heappush(self.task_queue, task)
                        else:
                            task.status = "failed"
                            self.failed_tasks.append(task)
                
                except Exception as e:
                    self.logger.error(f"Failed to execute task {task.task_id}: {e}")
                    task.status = "failed"
                    self.failed_tasks.append(task)
                    
        except Exception as e:
            self.logger.error(f"Failed to execute ready tasks: {e}")
    
    async def _can_execute_task(self, task: TrainingTask) -> Tuple[bool, str]:
        """Check if a task can be executed now"""
        try:
            # Budget check
            is_allowed, budget_reason = self.budget_manager.is_request_allowed(task.estimated_cost)
            if not is_allowed:
                return False, budget_reason
            
            # Concurrency check
            if len(self.active_tasks) >= self.max_concurrent_tasks:
                return False, "Maximum concurrent tasks reached"
            
            # Agent-specific checks
            agent_tasks = [t for t in self.active_tasks.values() if t.agent_id == task.agent_id]
            if len(agent_tasks) >= 2:  # Max 2 tasks per agent
                return False, "Agent has too many active tasks"
            
            return True, "Task can be executed"
            
        except Exception as e:
            self.logger.error(f"Failed to check task executability: {e}")
            return False, str(e)
    
    async def _execute_task(self, task: TrainingTask) -> None:
        """Execute a training task"""
        try:
            task.started_at = datetime.now()
            task.status = "running"
            self.active_tasks[task.task_id] = task
            
            self.logger.info(f"Executing task {task.task_id} for agent {task.agent_id}")
            
            # Get training data from orchestrator
            training_data = await self.orchestrator.get_training_data(
                data_type=task.task_type,
                count=task.data_requirements.get("count", 100),
                specific_requirements=task.data_requirements
            )
            
            # Execute training callback if available
            callback = self.task_callbacks.get(task.task_type)
            if callback:
                await callback(task, training_data)
            else:
                # Default training execution (simplified)
                await self._default_training_execution(task, training_data)
            
            # Mark task as completed
            task.completed_at = datetime.now()
            task.status = "completed"
            self.completed_tasks.append(task)
            
            # Remove from active tasks
            del self.active_tasks[task.task_id]
            
            # Update performance metrics
            await self._update_task_performance_metrics(task, success=True)
            
            self.logger.info(f"Completed task {task.task_id}")
            
        except Exception as e:
            self.logger.error(f"Task execution failed {task.task_id}: {e}")
            task.status = "failed"
            self.failed_tasks.append(task)
            
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            
            await self._update_task_performance_metrics(task, success=False)
    
    async def _default_training_execution(self, task: TrainingTask, training_data: List) -> None:
        """Default training execution (placeholder)"""
        # Simulate training time
        await asyncio.sleep(2)  # 2 second simulation
        
        # In real implementation, this would:
        # 1. Pass data to agent training system
        # 2. Execute training scenarios
        # 3. Collect performance metrics
        # 4. Update agent memory systems
        
        self.logger.info(f"Simulated training execution for task {task.task_id} with {len(training_data)} data points")
    
    async def _update_task_performance_metrics(self, task: TrainingTask, success: bool) -> None:
        """Update performance metrics for completed tasks"""
        try:
            # Calculate actual vs estimated metrics
            if task.started_at and task.completed_at:
                actual_duration = task.completed_at - task.started_at
                estimated_duration = task.estimated_duration
                
                duration_accuracy = 1 - abs((actual_duration - estimated_duration).total_seconds()) / estimated_duration.total_seconds()
                self.performance_metrics["duration_accuracy"] = (
                    self.performance_metrics["duration_accuracy"] * 0.9 + duration_accuracy * 0.1
                )
            
            # Update success rate
            current_success_rate = self.performance_metrics.get("success_rate", 0.5)
            new_success_rate = current_success_rate * 0.9 + (1.0 if success else 0.0) * 0.1
            self.performance_metrics["success_rate"] = new_success_rate
            
            # Update cost accuracy (simplified)
            self.performance_metrics["cost_accuracy"] = 0.85  # Placeholder
            
        except Exception as e:
            self.logger.error(f"Failed to update task performance metrics: {e}")
    
    async def _rebalance_schedule(self) -> None:
        """Rebalance the task schedule based on current conditions"""
        try:
            self.logger.info("Rebalancing task schedule")
            
            # Get current context
            context = await self._build_scheduling_context()
            
            # Identify tasks that should be rescheduled
            tasks_to_rebalance = []
            
            for task in list(self.task_queue):
                # Check if task should be rescheduled based on new context
                if self._should_reschedule_task(task, context):
                    tasks_to_rebalance.append(task)
            
            # Remove and reschedule identified tasks
            for task in tasks_to_rebalance:
                self.task_queue.remove(task)
                new_schedule_time = await self._find_optimal_schedule_time(task)
                task.scheduled_at = new_schedule_time
                heapq.heappush(self.task_queue, task)
            
            if tasks_to_rebalance:
                self.logger.info(f"Rescheduled {len(tasks_to_rebalance)} tasks")
            
        except Exception as e:
            self.logger.error(f"Failed to rebalance schedule: {e}")
    
    def _should_reschedule_task(self, task: TrainingTask, context: SchedulingContext) -> bool:
        """Determine if a task should be rescheduled"""
        try:
            # Don't reschedule high priority tasks
            if task.priority in [Priority.HIGH, Priority.CRITICAL]:
                return False
            
            # Reschedule if budget is tight and task is expensive
            if task.estimated_cost > context.available_budget * 0.5:
                return True
            
            # Reschedule if system load is high
            if context.system_load > 0.8 and task.priority == Priority.LOW:
                return True
            
            # Reschedule based on agent performance
            agent_metrics = context.agent_performance_metrics.get(task.agent_id, {})
            if agent_metrics.get("latest_snapshot", {}).get("learning_velocity", 0) < 0.2:
                # Agent learning slowly, might want to delay advanced tasks
                if task.task_type in ["validation", "edge_case_practice"]:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to check if task should be rescheduled: {e}")
            return False
    
    async def _update_agent_phases(self) -> None:
        """Update agent training phases based on performance"""
        try:
            for agent_id, current_phase in self.agent_phases.items():
                try:
                    # Get recent analytics
                    analytics = await self.agent_analytics.get_agent_analytics(agent_id, timedelta(hours=24))
                    
                    if not analytics or "latest_snapshot" not in analytics:
                        continue
                    
                    snapshot = analytics["latest_snapshot"]
                    
                    # Phase progression logic
                    new_phase = self._determine_agent_phase(current_phase, snapshot, analytics)
                    
                    if new_phase != current_phase:
                        self.agent_phases[agent_id] = new_phase
                        self.logger.info(f"Agent {agent_id} progressed from {current_phase.value} to {new_phase.value}")
                        
                        # Schedule phase transition tasks
                        await self._schedule_phase_transition_tasks(agent_id, current_phase, new_phase)
                
                except Exception as e:
                    self.logger.warning(f"Failed to update phase for agent {agent_id}: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Failed to update agent phases: {e}")
    
    def _determine_agent_phase(self, current_phase: TrainingPhase, snapshot: Dict, analytics: Dict) -> TrainingPhase:
        """Determine appropriate training phase for an agent"""
        try:
            total_decisions = analytics.get("total_decisions", 0)
            win_rate = snapshot.get("win_rate", 0)
            kelly_adherence = snapshot.get("kelly_adherence_score", 0)
            learning_velocity = snapshot.get("learning_velocity", 0)
            
            # Phase transition criteria
            if current_phase == TrainingPhase.INITIAL:
                if total_decisions >= 50 and win_rate > 0.3:
                    return TrainingPhase.FOUNDATION
            
            elif current_phase == TrainingPhase.FOUNDATION:
                if total_decisions >= 200 and win_rate > 0.45 and kelly_adherence > 0.6:
                    return TrainingPhase.ADVANCED
            
            elif current_phase == TrainingPhase.ADVANCED:
                if total_decisions >= 500 and win_rate > 0.55 and kelly_adherence > 0.75:
                    return TrainingPhase.VALIDATION
            
            elif current_phase == TrainingPhase.VALIDATION:
                if total_decisions >= 1000 and win_rate > 0.6 and kelly_adherence > 0.8:
                    return TrainingPhase.PRODUCTION_PREP
            
            # Regression checks
            if win_rate < 0.3 or learning_velocity < 0.1:
                # Agent struggling, may need to go back
                if current_phase == TrainingPhase.PRODUCTION_PREP:
                    return TrainingPhase.VALIDATION
                elif current_phase == TrainingPhase.VALIDATION:
                    return TrainingPhase.ADVANCED
            
            return current_phase
            
        except Exception as e:
            self.logger.error(f"Failed to determine agent phase: {e}")
            return current_phase
    
    async def _schedule_phase_transition_tasks(self, agent_id: str, old_phase: TrainingPhase, new_phase: TrainingPhase) -> None:
        """Schedule tasks appropriate for phase transition"""
        try:
            # Schedule validation task to confirm readiness
            await self.schedule_training_task(
                agent_id=agent_id,
                task_type="validation",
                priority=Priority.HIGH,
                data_requirements={
                    "count": 50,
                    "include_edge_cases": True,
                    "phase_transition": True,
                    "old_phase": old_phase.value,
                    "new_phase": new_phase.value
                }
            )
            
            # Schedule new phase introduction task
            await self.schedule_training_task(
                agent_id=agent_id,
                task_type="scenario_training",
                priority=Priority.NORMAL,
                data_requirements={
                    "count": 100,
                    "training_phase": new_phase.value,
                    "phase_introduction": True
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to schedule phase transition tasks: {e}")
    
    def _cleanup_old_tasks(self) -> None:
        """Clean up old completed and failed tasks"""
        try:
            cutoff_time = datetime.now() - timedelta(days=7)
            
            # Clean completed tasks
            self.completed_tasks = [
                task for task in self.completed_tasks
                if task.completed_at and task.completed_at > cutoff_time
            ]
            
            # Clean failed tasks
            self.failed_tasks = [
                task for task in self.failed_tasks
                if task.created_at > cutoff_time
            ]
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old tasks: {e}")
    
    def register_task_callback(self, task_type: str, callback: Callable) -> None:
        """Register callback for specific task type"""
        self.task_callbacks[task_type] = callback
        self.logger.info(f"Registered callback for task type: {task_type}")
    
    def set_scheduling_policy(self, policy: SchedulingPolicy) -> None:
        """Set the scheduling policy"""
        old_policy = self.scheduling_policy
        self.scheduling_policy = policy
        self.logger.info(f"Changed scheduling policy from {old_policy.value} to {policy.value}")
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get comprehensive scheduler status"""
        try:
            return {
                "timestamp": datetime.now().isoformat(),
                "scheduling_policy": self.scheduling_policy.value,
                "queue_size": len(self.task_queue),
                "active_tasks": len(self.active_tasks),
                "completed_tasks": len(self.completed_tasks),
                "failed_tasks": len(self.failed_tasks),
                "agent_phases": {agent_id: phase.value for agent_id, phase in self.agent_phases.items()},
                "performance_metrics": dict(self.performance_metrics),
                "next_scheduled_task": self.task_queue[0].scheduled_at.isoformat() if self.task_queue else None,
                "system_load": self.performance_metrics.get("system_load", 0.5)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get scheduler status: {e}")
            return {"error": str(e)}
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the scheduler"""
        try:
            self.logger.info("Shutting down Adaptive Scheduler")
            
            # Wait for active tasks to complete (with timeout)
            timeout = 300  # 5 minutes
            start_time = datetime.now()
            
            while self.active_tasks and (datetime.now() - start_time).seconds < timeout:
                await asyncio.sleep(10)
            
            # Force stop remaining tasks
            for task in self.active_tasks.values():
                task.status = "cancelled"
                self.failed_tasks.append(task)
            
            self.active_tasks.clear()
            
            self.logger.info("Adaptive Scheduler shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during scheduler shutdown: {e}")