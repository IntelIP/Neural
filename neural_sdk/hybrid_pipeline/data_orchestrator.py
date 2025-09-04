"""
Hybrid Data Orchestrator

Intelligently switches between live API data and synthetic data generation
based on training requirements, API costs, and performance metrics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import logging
from datetime import datetime, timedelta

from ..synthetic_data.generators.game_engine import SyntheticGameEngine
from ..synthetic_data.generators.market_simulator import MarketSimulator
from ..synthetic_data.generators.scenario_builder import ScenarioBuilder
from ..synthetic_data.storage.chromadb_manager import ChromaDBManager
from ..sdk.core.base_adapter import StandardizedEvent
# Training module imports commented out - not yet available


class DataSource(Enum):
    """Available data sources"""
    LIVE_API = "live_api"
    SYNTHETIC = "synthetic"
    HYBRID = "hybrid"
    CACHED = "cached"


class DataMode(Enum):
    """Operating modes for data pipeline"""
    TRAINING = "training"
    BACKTESTING = "backtesting"
    LIVE_TRADING = "live_trading"
    DEVELOPMENT = "development"


class SwitchingStrategy(Enum):
    """Strategies for switching between data sources"""
    COST_BASED = "cost_based"
    PERFORMANCE_BASED = "performance_based"
    TIME_BASED = "time_based"
    MANUAL = "manual"
    INTELLIGENT = "intelligent"


@dataclass
class CostThreshold:
    """Cost thresholds for data source switching"""
    daily_limit: float = 100.0
    hourly_limit: float = 10.0
    per_request_limit: float = 0.50
    emergency_cutoff: float = 200.0
    synthetic_cost_per_event: float = 0.001  # Much lower cost


@dataclass
class DataSourceMetrics:
    """Metrics for a specific data source"""
    source: DataSource
    requests_made: int = 0
    cost_incurred: float = 0.0
    latency_ms: float = 0.0
    error_rate: float = 0.0
    data_quality_score: float = 1.0
    last_used: Optional[datetime] = None
    availability: bool = True


@dataclass
class HybridConfig:
    """Configuration for hybrid data pipeline"""
    default_mode: DataMode = DataMode.TRAINING
    primary_source: DataSource = DataSource.SYNTHETIC
    fallback_source: DataSource = DataSource.CACHED
    switching_strategy: SwitchingStrategy = SwitchingStrategy.INTELLIGENT
    cost_thresholds: CostThreshold = field(default_factory=CostThreshold)
    
    # Performance thresholds
    max_latency_ms: float = 1000.0
    min_data_quality: float = 0.8
    max_error_rate: float = 0.05
    
    # Synthetic data preferences
    synthetic_scenario_count: int = 1000
    edge_case_probability: float = 0.1
    information_asymmetry_level: float = 0.3
    
    # Caching settings
    cache_duration_hours: int = 24
    cache_size_limit: int = 10000


class HybridDataOrchestrator:
    """
    Orchestrates hybrid data pipeline for optimal cost-performance balance.
    
    Automatically switches between live API data and synthetic data based on:
    - API costs and budget constraints
    - Agent training performance requirements
    - Data quality and availability
    - Training phase and objectives
    """
    
    def __init__(
        self,
        config: HybridConfig,
        chroma_manager: ChromaDBManager,
        synthetic_engine: SyntheticGameEngine,
        market_simulator: MarketSimulator,
        scenario_builder: ScenarioBuilder,
        agent_analytics: AgentAnalytics
    ):
        self.config = config
        self.chroma_manager = chroma_manager
        self.synthetic_engine = synthetic_engine
        self.market_simulator = market_simulator
        self.scenario_builder = scenario_builder
        self.agent_analytics = agent_analytics
        
        self.logger = logging.getLogger(__name__)
        
        # Data source metrics
        self.source_metrics = {
            DataSource.LIVE_API: DataSourceMetrics(DataSource.LIVE_API),
            DataSource.SYNTHETIC: DataSourceMetrics(DataSource.SYNTHETIC),
            DataSource.CACHED: DataSourceMetrics(DataSource.CACHED)
        }
        
        # Current state
        self.active_source = self.config.primary_source
        self.current_mode = self.config.default_mode
        self.daily_costs = 0.0
        self.last_cost_reset = datetime.now().date()
        
        # Cache for processed data
        self.data_cache: Dict[str, Any] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        
        # Decision history for intelligent switching
        self.switching_history: List[Dict[str, Any]] = []
        
    async def initialize(self) -> None:
        """Initialize the hybrid data orchestrator"""
        try:
            self.logger.info("Initializing Hybrid Data Orchestrator")
            
            # Reset daily costs if needed
            self._check_daily_cost_reset()
            
            # Verify synthetic data systems
            await self._verify_synthetic_systems()
            
            # Load cached decisions if available
            await self._load_switching_history()
            
            # Initial data source selection
            await self._select_optimal_data_source()
            
            self.logger.info(f"Hybrid orchestrator initialized with {self.active_source} as active source")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize hybrid orchestrator: {e}")
            raise
    
    async def get_training_data(
        self,
        data_type: str,
        count: int = 100,
        specific_requirements: Optional[Dict[str, Any]] = None
    ) -> List[StandardizedEvent]:
        """
        Get training data using optimal data source selection.
        
        Args:
            data_type: Type of data needed ("game_events", "market_events", etc.)
            count: Number of events requested
            specific_requirements: Specific requirements (edge cases, scenarios, etc.)
            
        Returns:
            List of standardized events for training
        """
        try:
            # Evaluate current data source optimality
            should_switch = await self._should_switch_data_source(data_type, count, specific_requirements)
            
            if should_switch:
                await self._switch_data_source(should_switch)
            
            # Get data from active source
            if self.active_source == DataSource.SYNTHETIC:
                return await self._get_synthetic_data(data_type, count, specific_requirements)
            elif self.active_source == DataSource.LIVE_API:
                return await self._get_live_api_data(data_type, count, specific_requirements)
            elif self.active_source == DataSource.CACHED:
                return await self._get_cached_data(data_type, count, specific_requirements)
            else:
                raise ValueError(f"Unsupported data source: {self.active_source}")
                
        except Exception as e:
            self.logger.error(f"Failed to get training data: {e}")
            # Fallback to synthetic data
            if self.active_source != DataSource.SYNTHETIC:
                self.logger.info("Falling back to synthetic data")
                return await self._get_synthetic_data(data_type, count, specific_requirements)
            raise
    
    async def _should_switch_data_source(
        self,
        data_type: str,
        count: int,
        requirements: Optional[Dict[str, Any]]
    ) -> Optional[DataSource]:
        """Determine if data source should be switched and to which source"""
        try:
            current_metrics = self.source_metrics[self.active_source]
            
            # Cost-based switching
            if self.config.switching_strategy in [SwitchingStrategy.COST_BASED, SwitchingStrategy.INTELLIGENT]:
                estimated_cost = await self._estimate_request_cost(self.active_source, count)
                
                # Check if cost exceeds thresholds
                if (self.daily_costs + estimated_cost > self.config.cost_thresholds.daily_limit or
                    estimated_cost > self.config.cost_thresholds.per_request_limit):
                    
                    # Switch to synthetic if currently using live API
                    if self.active_source == DataSource.LIVE_API:
                        return DataSource.SYNTHETIC
            
            # Performance-based switching
            if self.config.switching_strategy in [SwitchingStrategy.PERFORMANCE_BASED, SwitchingStrategy.INTELLIGENT]:
                if (current_metrics.latency_ms > self.config.max_latency_ms or
                    current_metrics.error_rate > self.config.max_error_rate or
                    current_metrics.data_quality_score < self.config.min_data_quality):
                    
                    # Find best alternative source
                    return await self._find_best_alternative_source()
            
            # Intelligence-based switching
            if self.config.switching_strategy == SwitchingStrategy.INTELLIGENT:
                return await self._intelligent_source_selection(data_type, count, requirements)
            
            return None  # No switch needed
            
        except Exception as e:
            self.logger.error(f"Error in switch evaluation: {e}")
            return None
    
    async def _intelligent_source_selection(
        self,
        data_type: str,
        count: int,
        requirements: Optional[Dict[str, Any]]
    ) -> Optional[DataSource]:
        """Intelligent data source selection based on multiple factors"""
        try:
            # Factors to consider
            factors = {}
            
            # 1. Training phase analysis
            if requirements and requirements.get("training_phase"):
                phase = requirements["training_phase"]
                if phase == "initial":
                    # Use synthetic for initial training (cheaper, unlimited scenarios)
                    factors["phase_preference"] = DataSource.SYNTHETIC
                elif phase == "advanced":
                    # Mix of synthetic and live for advanced training
                    factors["phase_preference"] = DataSource.HYBRID
                elif phase == "validation":
                    # Live data for final validation
                    factors["phase_preference"] = DataSource.LIVE_API
            
            # 2. Agent performance analysis
            if requirements and requirements.get("agent_id"):
                agent_id = requirements["agent_id"]
                # Get recent analytics to determine if agent needs diverse scenarios
                analytics = await self.agent_analytics.get_agent_analytics(agent_id, timedelta(hours=24))
                
                if analytics.get("latest_snapshot"):
                    snapshot = analytics["latest_snapshot"]
                    
                    # If learning velocity is low, provide more diverse synthetic scenarios
                    if snapshot.get("learning_velocity", 0) < 0.3:
                        factors["performance_need"] = DataSource.SYNTHETIC
                    
                    # If agent is performing well, validate with live data
                    elif snapshot.get("win_rate", 0) > 0.6:
                        factors["performance_need"] = DataSource.LIVE_API
            
            # 3. Cost efficiency analysis
            synthetic_cost = self.config.cost_thresholds.synthetic_cost_per_event * count
            api_cost = await self._estimate_request_cost(DataSource.LIVE_API, count)
            
            if api_cost > synthetic_cost * 10:  # API is 10x more expensive
                factors["cost_efficiency"] = DataSource.SYNTHETIC
            
            # 4. Data requirements analysis
            if requirements:
                # Edge cases are better handled by synthetic data
                if requirements.get("include_edge_cases", False):
                    factors["data_requirement"] = DataSource.SYNTHETIC
                
                # Real market conditions need live data
                if requirements.get("market_realism", False):
                    factors["data_requirement"] = DataSource.LIVE_API
            
            # 5. System load analysis
            current_load = await self._get_system_load()
            if current_load > 0.8:  # High load, prefer cached data
                factors["system_load"] = DataSource.CACHED
            
            # Decision algorithm: weighted voting
            votes = {source: 0 for source in DataSource}
            for factor_name, preferred_source in factors.items():
                weight = self._get_factor_weight(factor_name)
                votes[preferred_source] += weight
            
            # Select highest voted source
            best_source = max(votes, key=votes.get)
            
            # Only switch if significantly better
            if votes[best_source] > votes[self.active_source] * 1.2:
                return best_source
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in intelligent source selection: {e}")
            return None
    
    def _get_factor_weight(self, factor_name: str) -> float:
        """Get weight for different decision factors"""
        weights = {
            "cost_efficiency": 0.4,
            "phase_preference": 0.3,
            "performance_need": 0.2,
            "data_requirement": 0.25,
            "system_load": 0.15
        }
        return weights.get(factor_name, 0.1)
    
    async def _estimate_request_cost(self, source: DataSource, count: int) -> float:
        """Estimate cost for a data request"""
        if source == DataSource.SYNTHETIC:
            return self.config.cost_thresholds.synthetic_cost_per_event * count
        elif source == DataSource.LIVE_API:
            # Estimate based on typical API costs
            return count * 0.01  # $0.01 per event (example)
        elif source == DataSource.CACHED:
            return 0.0  # No cost for cached data
        else:
            return 0.0
    
    async def _get_system_load(self) -> float:
        """Get current system load (simplified implementation)"""
        try:
            # In real implementation, would check CPU, memory, network
            # For now, return random load between 0.1 and 0.9
            import random
            return random.uniform(0.1, 0.9)
        except Exception:
            return 0.5  # Default moderate load
    
    async def _find_best_alternative_source(self) -> DataSource:
        """Find the best alternative data source based on current metrics"""
        best_source = None
        best_score = -1
        
        for source, metrics in self.source_metrics.items():
            if source == self.active_source:
                continue
                
            if not metrics.availability:
                continue
            
            # Score based on multiple factors
            score = (
                (1 - metrics.error_rate) * 0.3 +
                min(1.0, 1000.0 / max(metrics.latency_ms, 1)) * 0.3 +
                metrics.data_quality_score * 0.4
            )
            
            if score > best_score:
                best_score = score
                best_source = source
        
        return best_source or DataSource.SYNTHETIC  # Fallback to synthetic
    
    async def _switch_data_source(self, new_source: DataSource) -> None:
        """Switch to a new data source"""
        try:
            old_source = self.active_source
            self.active_source = new_source
            
            # Record the switch
            switch_record = {
                "timestamp": datetime.now().isoformat(),
                "from_source": old_source.value,
                "to_source": new_source.value,
                "reason": "intelligent_switching",
                "daily_cost": self.daily_costs
            }
            self.switching_history.append(switch_record)
            
            # Update metrics
            self.source_metrics[new_source].last_used = datetime.now()
            
            self.logger.info(f"Switched data source from {old_source.value} to {new_source.value}")
            
        except Exception as e:
            self.logger.error(f"Failed to switch data source: {e}")
            raise
    
    async def _get_synthetic_data(
        self,
        data_type: str,
        count: int,
        requirements: Optional[Dict[str, Any]]
    ) -> List[StandardizedEvent]:
        """Get synthetic training data"""
        try:
            start_time = datetime.now()
            
            if data_type == "game_events":
                # Generate synthetic games
                scenarios = await self.scenario_builder.build_comprehensive_training_set(
                    num_scenarios=max(count // 20, 1),  # ~20 events per game
                    include_edge_cases=requirements.get("include_edge_cases", True) if requirements else True
                )
                
                # Convert scenarios to standardized events
                events = []
                for scenario in scenarios.scenarios[:count]:
                    game_events = await self._convert_scenario_to_events(scenario)
                    events.extend(game_events)
                
                events = events[:count]  # Trim to requested count
                
            elif data_type == "market_events":
                # Generate market-specific scenarios
                trading_scenarios = await self.scenario_builder.build_market_scenarios(count)
                events = await self._convert_market_scenarios_to_events(trading_scenarios)
                
            else:
                # Default: generate mixed scenarios
                scenarios = await self.scenario_builder.build_comprehensive_training_set(
                    num_scenarios=count // 10,
                    include_edge_cases=True
                )
                events = []
                for scenario in scenarios.scenarios:
                    scenario_events = await self._convert_scenario_to_events(scenario)
                    events.extend(scenario_events)
                events = events[:count]
            
            # Update metrics
            latency = (datetime.now() - start_time).total_seconds() * 1000
            cost = len(events) * self.config.cost_thresholds.synthetic_cost_per_event
            
            await self._update_source_metrics(
                DataSource.SYNTHETIC,
                requests=1,
                cost=cost,
                latency=latency,
                success=True
            )
            
            return events
            
        except Exception as e:
            await self._update_source_metrics(DataSource.SYNTHETIC, requests=1, success=False)
            self.logger.error(f"Failed to get synthetic data: {e}")
            raise
    
    async def _get_live_api_data(
        self,
        data_type: str,
        count: int,
        requirements: Optional[Dict[str, Any]]
    ) -> List[StandardizedEvent]:
        """Get data from live APIs (placeholder implementation)"""
        try:
            start_time = datetime.now()
            
            # This would integrate with actual live APIs
            # For now, return cached data as proxy for live data
            events = await self._get_cached_data(data_type, count, requirements)
            
            # Simulate API cost and latency
            latency = 500  # 500ms typical API latency
            cost = count * 0.01  # $0.01 per event
            
            await self._update_source_metrics(
                DataSource.LIVE_API,
                requests=1,
                cost=cost,
                latency=latency,
                success=True
            )
            
            return events
            
        except Exception as e:
            await self._update_source_metrics(DataSource.LIVE_API, requests=1, success=False)
            self.logger.error(f"Failed to get live API data: {e}")
            raise
    
    async def _get_cached_data(
        self,
        data_type: str,
        count: int,
        requirements: Optional[Dict[str, Any]]
    ) -> List[StandardizedEvent]:
        """Get cached training data"""
        try:
            # Check cache freshness
            cache_key = f"{data_type}_{count}_{hash(str(requirements))}"
            
            if (cache_key in self.data_cache and 
                cache_key in self.cache_timestamps and
                datetime.now() - self.cache_timestamps[cache_key] < timedelta(hours=self.config.cache_duration_hours)):
                
                cached_events = self.data_cache[cache_key]
                await self._update_source_metrics(DataSource.CACHED, requests=1, cost=0.0, latency=10, success=True)
                return cached_events
            
            # If no cached data, generate synthetic data and cache it
            events = await self._get_synthetic_data(data_type, count, requirements)
            
            # Cache the results
            if len(self.data_cache) < self.config.cache_size_limit:
                self.data_cache[cache_key] = events
                self.cache_timestamps[cache_key] = datetime.now()
            
            return events
            
        except Exception as e:
            await self._update_source_metrics(DataSource.CACHED, requests=1, success=False)
            self.logger.error(f"Failed to get cached data: {e}")
            raise
    
    async def _convert_scenario_to_events(self, scenario) -> List[StandardizedEvent]:
        """Convert training scenario to standardized events"""
        try:
            events = []
            
            # Extract game events from scenario
            if hasattr(scenario, 'game') and scenario.game:
                for play in scenario.game.plays:
                    event = StandardizedEvent(
                        event_id=f"synth_{play.play_id}",
                        game_id=scenario.game.game_id,
                        timestamp=play.timestamp,
                        event_type=play.play_type,
                        description=play.description,
                        team_possession=play.team_possession,
                        score_home=play.score_home,
                        score_away=play.score_away,
                        quarter=play.quarter,
                        time_remaining=play.time_remaining,
                        down=getattr(play, 'down', None),
                        yards_to_go=getattr(play, 'yards_to_go', None),
                        yard_line=getattr(play, 'yard_line', None),
                        metadata={
                            "synthetic": True,
                            "scenario_id": scenario.scenario_id,
                            "source": "hybrid_orchestrator"
                        }
                    )
                    events.append(event)
            
            return events
            
        except Exception as e:
            self.logger.error(f"Failed to convert scenario to events: {e}")
            return []
    
    async def _convert_market_scenarios_to_events(self, scenarios) -> List[StandardizedEvent]:
        """Convert market scenarios to standardized events"""
        try:
            events = []
            
            for scenario in scenarios:
                # Convert market events to standardized format
                for market_event in scenario.events:
                    event = StandardizedEvent(
                        event_id=f"market_{market_event.event_id}",
                        game_id=scenario.market_ticker,
                        timestamp=market_event.timestamp,
                        event_type="market_update",
                        description=f"Market price update: {market_event.price_change}",
                        metadata={
                            "synthetic": True,
                            "scenario_id": scenario.scenario_id,
                            "price_change": market_event.price_change,
                            "volume": getattr(market_event, 'volume', 0),
                            "source": "hybrid_orchestrator"
                        }
                    )
                    events.append(event)
            
            return events
            
        except Exception as e:
            self.logger.error(f"Failed to convert market scenarios to events: {e}")
            return []
    
    async def _update_source_metrics(
        self,
        source: DataSource,
        requests: int = 0,
        cost: float = 0.0,
        latency: float = 0.0,
        success: bool = True
    ) -> None:
        """Update metrics for a data source"""
        try:
            metrics = self.source_metrics[source]
            
            metrics.requests_made += requests
            metrics.cost_incurred += cost
            
            if latency > 0:
                # Exponential moving average for latency
                alpha = 0.1
                metrics.latency_ms = (1 - alpha) * metrics.latency_ms + alpha * latency
            
            # Update error rate
            if requests > 0:
                alpha = 0.1
                new_error_rate = 0.0 if success else 1.0
                metrics.error_rate = (1 - alpha) * metrics.error_rate + alpha * new_error_rate
            
            # Update daily costs
            self.daily_costs += cost
            
            metrics.last_used = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Failed to update source metrics: {e}")
    
    async def _verify_synthetic_systems(self) -> None:
        """Verify that synthetic data systems are available"""
        try:
            # Test synthetic game engine
            test_game = await self.synthetic_engine.generate_single_game()
            if not test_game or not test_game.plays:
                raise Exception("Synthetic game engine not working")
            
            # Test scenario builder
            test_scenarios = await self.scenario_builder.build_comprehensive_training_set(num_scenarios=1)
            if not test_scenarios or not test_scenarios.scenarios:
                raise Exception("Scenario builder not working")
            
            self.logger.info("Synthetic systems verified successfully")
            
        except Exception as e:
            self.logger.error(f"Synthetic systems verification failed: {e}")
            # Mark synthetic as unavailable
            self.source_metrics[DataSource.SYNTHETIC].availability = False
            raise
    
    def _check_daily_cost_reset(self) -> None:
        """Check if daily costs should be reset"""
        today = datetime.now().date()
        if today > self.last_cost_reset:
            self.daily_costs = 0.0
            self.last_cost_reset = today
            self.logger.info("Daily costs reset")
    
    async def _load_switching_history(self) -> None:
        """Load switching history from persistent storage"""
        try:
            # In real implementation, would load from database
            # For now, initialize empty history
            self.switching_history = []
            
        except Exception as e:
            self.logger.error(f"Failed to load switching history: {e}")
            self.switching_history = []
    
    async def _select_optimal_data_source(self) -> None:
        """Select optimal initial data source"""
        try:
            # For training mode, prefer synthetic data
            if self.current_mode == DataMode.TRAINING:
                self.active_source = DataSource.SYNTHETIC
            # For live trading, prefer live API
            elif self.current_mode == DataMode.LIVE_TRADING:
                self.active_source = DataSource.LIVE_API
            # For backtesting, prefer cached data
            elif self.current_mode == DataMode.BACKTESTING:
                self.active_source = DataSource.CACHED
            else:
                # Default to configured primary source
                self.active_source = self.config.primary_source
            
            self.logger.info(f"Selected {self.active_source.value} as optimal data source for {self.current_mode.value} mode")
            
        except Exception as e:
            self.logger.error(f"Failed to select optimal data source: {e}")
            self.active_source = DataSource.SYNTHETIC  # Safe fallback
    
    async def get_cost_summary(self) -> Dict[str, Any]:
        """Get comprehensive cost summary"""
        try:
            return {
                "daily_costs": self.daily_costs,
                "daily_limit": self.config.cost_thresholds.daily_limit,
                "utilization_percent": (self.daily_costs / self.config.cost_thresholds.daily_limit) * 100,
                "source_breakdown": {
                    source.value: {
                        "requests": metrics.requests_made,
                        "cost": metrics.cost_incurred,
                        "avg_latency": metrics.latency_ms,
                        "error_rate": metrics.error_rate,
                        "last_used": metrics.last_used.isoformat() if metrics.last_used else None
                    }
                    for source, metrics in self.source_metrics.items()
                },
                "active_source": self.active_source.value,
                "switching_history": self.switching_history[-10:]  # Last 10 switches
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate cost summary: {e}")
            return {"error": str(e)}
    
    async def force_data_source(self, source: DataSource) -> None:
        """Manually force a specific data source"""
        try:
            old_source = self.active_source
            self.active_source = source
            
            # Record manual switch
            switch_record = {
                "timestamp": datetime.now().isoformat(),
                "from_source": old_source.value,
                "to_source": source.value,
                "reason": "manual_override",
                "daily_cost": self.daily_costs
            }
            self.switching_history.append(switch_record)
            
            self.logger.info(f"Manually switched data source to {source.value}")
            
        except Exception as e:
            self.logger.error(f"Failed to force data source: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the hybrid orchestrator"""
        try:
            self.logger.info("Shutting down Hybrid Data Orchestrator")
            
            # Save switching history
            await self._save_switching_history()
            
            # Clear cache
            self.data_cache.clear()
            self.cache_timestamps.clear()
            
            self.logger.info("Hybrid Data Orchestrator shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    async def _save_switching_history(self) -> None:
        """Save switching history to persistent storage"""
        try:
            # In real implementation, would save to database
            # For now, just log summary
            self.logger.info(f"Saving switching history: {len(self.switching_history)} records")
            
        except Exception as e:
            self.logger.error(f"Failed to save switching history: {e}")