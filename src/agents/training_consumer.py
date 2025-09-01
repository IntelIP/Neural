"""
Training-Enhanced Redis Consumer

Extends BaseAgentRedisConsumer with training mode support,
performance tracking, and integration with learning systems.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from abc import abstractmethod
import json

from .base_consumer import BaseAgentRedisConsumer
from ..training.agent_analytics import DecisionMetrics
from ..training.memory_system import AgentMemorySystem, AgentMemory, MemoryType
from ..confidence_calibration.calibrator import ConfidenceCalibrator


class TrainingConsumer(BaseAgentRedisConsumer):
    """
    Enhanced Redis consumer with training mode capabilities.
    
    Adds:
    - Training/production mode switching
    - Decision tracking and analytics
    - Memory system integration
    - Confidence calibration
    - Performance monitoring
    """
    
    def __init__(
        self,
        agent_name: str,
        redis_url: str = "redis://localhost:6379",
        agent_context: Optional[Any] = None,
        training_mode: bool = False
    ):
        """
        Initialize training-enhanced consumer.
        
        Args:
            agent_name: Unique agent identifier
            redis_url: Redis connection URL
            agent_context: Agentuity context (optional)
            training_mode: Whether to start in training mode
        """
        super().__init__(agent_name, redis_url, agent_context)
        
        self.training_mode = training_mode
        self.training_session_id: Optional[str] = None
        self.logger = logging.getLogger(__name__)
        
        # Training systems (will be injected)
        self.analytics: Optional[Any] = None
        self.memory_system: Optional[AgentMemorySystem] = None
        self.calibrator: Optional[ConfidenceCalibrator] = None
        
        # Performance tracking
        self.decision_buffer: List[Dict[str, Any]] = []
        self.pending_decisions: Dict[str, Dict[str, Any]] = {}  # Track outcomes
        
        # Training configuration
        self.training_config = {
            "track_all_decisions": True,
            "confidence_threshold": 0.3,
            "max_position_size": 0.1,
            "use_calibrated_confidence": True,
            "store_experiences": True,
            "learn_from_mistakes": True
        }
        
        # Metrics
        self.training_metrics = {
            "decisions_made": 0,
            "successful_decisions": 0,
            "failed_decisions": 0,
            "total_pnl": 0.0,
            "confidence_sum": 0.0,
            "kelly_adherence_sum": 0.0
        }
    
    def set_training_mode(self, enabled: bool, session_id: Optional[str] = None) -> None:
        """
        Enable or disable training mode.
        
        Args:
            enabled: Whether to enable training mode
            session_id: Optional training session identifier
        """
        self.training_mode = enabled
        self.training_session_id = session_id
        
        if enabled:
            self.logger.info(f"{self.agent_name} entering training mode (session: {session_id})")
            self._reset_training_metrics()
        else:
            self.logger.info(f"{self.agent_name} exiting training mode")
            self._finalize_training_metrics()
    
    def inject_training_systems(
        self,
        analytics: Any,
        memory_system: AgentMemorySystem,
        calibrator: ConfidenceCalibrator
    ) -> None:
        """
        Inject training system dependencies.
        
        Args:
            analytics: Agent analytics system
            memory_system: Memory storage system
            calibrator: Confidence calibration system
        """
        self.analytics = analytics
        self.memory_system = memory_system
        self.calibrator = calibrator
        
        self.logger.info(f"Training systems injected for {self.agent_name}")
    
    async def process_message(self, channel: str, data: Dict[str, Any]) -> None:
        """
        Process message with training enhancements.
        
        Wraps the agent's process_message to add training functionality.
        
        Args:
            channel: Redis channel the message came from
            data: Message data
        """
        # Pre-process for training
        if self.training_mode:
            await self._pre_process_training(channel, data)
        
        # Call agent's implementation
        await self.process_training_message(channel, data)
        
        # Post-process for training
        if self.training_mode:
            await self._post_process_training(channel, data)
    
    @abstractmethod
    async def process_training_message(self, channel: str, data: Dict[str, Any]) -> None:
        """
        Process message - to be implemented by specific agents.
        
        This replaces the original process_message for training-aware agents.
        
        Args:
            channel: Redis channel
            data: Message data
        """
        pass
    
    async def _pre_process_training(self, channel: str, data: Dict[str, Any]) -> None:
        """Pre-process message for training tracking"""
        try:
            # Add training context
            data['_training_context'] = {
                'received_at': datetime.now().isoformat(),
                'channel': channel,
                'session_id': self.training_session_id
            }
            
            # Store in memory if configured
            if self.memory_system and self.training_config["store_experiences"]:
                await self._store_incoming_event(channel, data)
                
        except Exception as e:
            self.logger.error(f"Training pre-process error: {e}")
    
    async def _post_process_training(self, channel: str, data: Dict[str, Any]) -> None:
        """Post-process message for training tracking"""
        try:
            # Check if a decision was made
            if hasattr(self, '_last_decision') and self._last_decision:
                await self._track_training_decision(self._last_decision)
                self._last_decision = None
                
        except Exception as e:
            self.logger.error(f"Training post-process error: {e}")
    
    async def make_trading_decision(
        self,
        market_ticker: str,
        decision_type: str,
        confidence: float,
        position_size: float = 0.0,
        expected_value: float = 0.0,
        kelly_fraction: float = 0.0,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make a trading decision with training tracking.
        
        Args:
            market_ticker: Market to trade
            decision_type: "buy", "sell", "hold"
            confidence: Raw confidence score (0-1)
            position_size: Position size to take
            expected_value: Expected value of decision
            kelly_fraction: Optimal Kelly fraction
            context: Additional context
            
        Returns:
            Decision details with potential calibration
        """
        try:
            # Apply confidence calibration if available
            calibrated_confidence = confidence
            uncertainty = 0.0
            
            if self.calibrator and self.training_config["use_calibrated_confidence"]:
                calibration_result = await self.calibrator.calibrate_confidence(
                    agent_id=self.agent_name,
                    raw_confidence=confidence,
                    context=context or {}
                )
                calibrated_confidence = calibration_result.calibrated_confidence
                uncertainty = calibration_result.uncertainty
            
            # Check confidence threshold
            if calibrated_confidence < self.training_config["confidence_threshold"]:
                decision_type = "hold"
                position_size = 0.0
            
            # Enforce position limits
            if position_size > self.training_config["max_position_size"]:
                position_size = self.training_config["max_position_size"]
            
            # Calculate actual Kelly used
            actual_kelly_used = position_size / kelly_fraction if kelly_fraction > 0 else 0.0
            
            # Create decision record
            decision = {
                "agent_id": self.agent_name,
                "decision_id": f"{self.agent_name}_{datetime.now().timestamp()}",
                "timestamp": datetime.now().isoformat(),
                "market_ticker": market_ticker,
                "decision_type": decision_type,
                "raw_confidence": confidence,
                "calibrated_confidence": calibrated_confidence,
                "uncertainty": uncertainty,
                "position_size": position_size,
                "expected_value": expected_value,
                "kelly_fraction": kelly_fraction,
                "actual_kelly_used": actual_kelly_used,
                "training_mode": self.training_mode,
                "session_id": self.training_session_id,
                "context": context
            }
            
            # Store for tracking
            self._last_decision = decision
            
            # Track in training mode
            if self.training_mode:
                self.training_metrics["decisions_made"] += 1
                self.training_metrics["confidence_sum"] += calibrated_confidence
                self.training_metrics["kelly_adherence_sum"] += abs(actual_kelly_used - 1.0)
                
                # Store pending for outcome tracking
                self.pending_decisions[decision["decision_id"]] = decision
                
                # Publish to training channel
                await self._publish_training_decision(decision)
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Error making trading decision: {e}")
            return {
                "error": str(e),
                "decision_type": "hold",
                "position_size": 0.0
            }
    
    async def _track_training_decision(self, decision: Dict[str, Any]) -> None:
        """Track a decision for training analytics"""
        try:
            if not self.analytics:
                return
            
            # Create DecisionMetrics object
            metrics = DecisionMetrics(
                decision_id=decision["decision_id"],
                agent_id=self.agent_name,
                scenario_id=self.training_session_id or "unknown",
                timestamp=datetime.fromisoformat(decision["timestamp"]),
                market_ticker=decision["market_ticker"],
                decision_type=decision["decision_type"],
                confidence=decision["calibrated_confidence"],
                expected_value=decision["expected_value"],
                kelly_fraction=decision["kelly_fraction"],
                actual_kelly_used=decision["actual_kelly_used"],
                position_size=decision["position_size"],
                market_efficiency=decision.get("context", {}).get("market_efficiency", 0.8),
                information_advantage=decision.get("context", {}).get("information_advantage", 0.0),
                execution_latency=(datetime.now() - datetime.fromisoformat(decision["timestamp"])).total_seconds()
            )
            
            # Record in analytics
            await self.analytics.record_decision(metrics)
            
            # Buffer for batch processing
            self.decision_buffer.append(decision)
            
            # Process buffer if full
            if len(self.decision_buffer) >= 10:
                await self._process_decision_buffer()
                
        except Exception as e:
            self.logger.error(f"Failed to track training decision: {e}")
    
    async def _process_decision_buffer(self) -> None:
        """Process buffered decisions for batch analytics"""
        try:
            if not self.decision_buffer:
                return
            
            # Batch process decisions
            for decision in self.decision_buffer:
                # Store in memory system if available
                if self.memory_system and self.training_config["store_experiences"]:
                    await self._store_decision_memory(decision)
            
            # Clear buffer
            self.decision_buffer.clear()
            
        except Exception as e:
            self.logger.error(f"Failed to process decision buffer: {e}")
    
    async def _store_decision_memory(self, decision: Dict[str, Any]) -> None:
        """Store decision in memory system"""
        try:
            if not self.memory_system:
                return
            
            # Create memory entry
            memory = AgentMemory(
                memory_id=f"decision_{decision['decision_id']}",
                agent_id=self.agent_name,
                memory_type=MemoryType.EXPERIENCE,
                timestamp=datetime.fromisoformat(decision["timestamp"]),
                description=f"{decision['decision_type']} on {decision['market_ticker']} with confidence {decision['calibrated_confidence']:.2f}",
                context=decision.get("context", {}),
                outcome={"pending": True}  # Will be updated when outcome known
            )
            
            # Store in memory system
            await self.memory_system.store_memory(memory)
            
        except Exception as e:
            self.logger.error(f"Failed to store decision memory: {e}")
    
    async def _store_incoming_event(self, channel: str, data: Dict[str, Any]) -> None:
        """Store incoming event in memory for pattern learning"""
        try:
            if not self.memory_system:
                return
            
            # Create memory entry for significant events
            if self._is_significant_event(channel, data):
                memory = AgentMemory(
                    memory_id=f"event_{self.agent_name}_{datetime.now().timestamp()}",
                    agent_id=self.agent_name,
                    memory_type=MemoryType.PATTERN,
                    timestamp=datetime.now(),
                    description=f"Event from {channel}: {data.get('type', 'unknown')}",
                    context={
                        "channel": channel,
                        "data": data,
                        "training_session": self.training_session_id
                    },
                    outcome={}
                )
                
                await self.memory_system.store_memory(memory)
                
        except Exception as e:
            self.logger.error(f"Failed to store incoming event: {e}")
    
    def _is_significant_event(self, channel: str, data: Dict[str, Any]) -> bool:
        """Determine if an event is significant enough to store"""
        # Store market updates, big plays, trades, high-impact events
        significant_types = ["market_update", "trade_executed", "big_play", "signal", "injury_alert"]
        event_type = data.get("type", "").lower()
        
        return any(sig in event_type for sig in significant_types)
    
    async def report_decision_outcome(
        self,
        decision_id: str,
        outcome: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Report the outcome of a previous decision.
        
        Args:
            decision_id: ID of the decision
            outcome: P&L outcome
            metadata: Additional outcome information
        """
        try:
            # Find pending decision
            decision = self.pending_decisions.get(decision_id)
            if not decision:
                return
            
            # Update metrics
            self.training_metrics["total_pnl"] += outcome
            if outcome > 0:
                self.training_metrics["successful_decisions"] += 1
            else:
                self.training_metrics["failed_decisions"] += 1
            
            # Update analytics if available
            if self.analytics:
                # Find and update the decision metrics
                # This would need to be implemented in analytics
                pass
            
            # Update memory with outcome
            if self.memory_system:
                await self._update_memory_outcome(decision_id, outcome, metadata)
            
            # Learn from mistakes if configured
            if self.training_config["learn_from_mistakes"] and outcome < 0:
                await self._learn_from_mistake(decision, outcome, metadata)
            
            # Remove from pending
            del self.pending_decisions[decision_id]
            
        except Exception as e:
            self.logger.error(f"Failed to report decision outcome: {e}")
    
    async def _update_memory_outcome(
        self,
        decision_id: str,
        outcome: float,
        metadata: Optional[Dict[str, Any]]
    ) -> None:
        """Update stored memory with decision outcome"""
        try:
            if not self.memory_system:
                return
            
            # Update the memory entry with outcome
            memory_id = f"decision_{decision_id}"
            
            # This would need memory_system to support updates
            # For now, store a new memory linking to the decision
            outcome_memory = AgentMemory(
                memory_id=f"outcome_{decision_id}",
                agent_id=self.agent_name,
                memory_type=MemoryType.EXPERIENCE,
                timestamp=datetime.now(),
                description=f"Outcome for decision {decision_id}: {'profit' if outcome > 0 else 'loss'} of {outcome:.2f}",
                context={"decision_id": decision_id, "metadata": metadata},
                outcome={"pnl": outcome, "success": outcome > 0}
            )
            
            await self.memory_system.store_memory(outcome_memory)
            
        except Exception as e:
            self.logger.error(f"Failed to update memory outcome: {e}")
    
    async def _learn_from_mistake(
        self,
        decision: Dict[str, Any],
        outcome: float,
        metadata: Optional[Dict[str, Any]]
    ) -> None:
        """Learn from a failed decision"""
        try:
            if not self.memory_system:
                return
            
            # Store as a mistake to avoid in future
            mistake_memory = AgentMemory(
                memory_id=f"mistake_{decision['decision_id']}",
                agent_id=self.agent_name,
                memory_type=MemoryType.MISTAKE,
                timestamp=datetime.now(),
                description=f"Failed {decision['decision_type']} on {decision['market_ticker']}: lost {abs(outcome):.2f}",
                context={
                    "decision": decision,
                    "outcome": outcome,
                    "metadata": metadata,
                    "lesson": self._extract_lesson(decision, outcome, metadata)
                },
                outcome={"loss": abs(outcome)}
            )
            
            await self.memory_system.store_memory(mistake_memory)
            
            self.logger.info(f"Learned from mistake: {mistake_memory.description}")
            
        except Exception as e:
            self.logger.error(f"Failed to learn from mistake: {e}")
    
    def _extract_lesson(
        self,
        decision: Dict[str, Any],
        outcome: float,
        metadata: Optional[Dict[str, Any]]
    ) -> str:
        """Extract a lesson from a failed decision"""
        lessons = []
        
        # Check confidence calibration
        if decision["calibrated_confidence"] > 0.7 and outcome < 0:
            lessons.append("High confidence was misplaced")
        
        # Check Kelly adherence
        if abs(decision["actual_kelly_used"] - 1.0) > 0.5:
            lessons.append("Position sizing was suboptimal")
        
        # Check context factors
        if metadata and "market_volatility" in metadata and metadata["market_volatility"] > 0.5:
            lessons.append("Failed to account for high volatility")
        
        return "; ".join(lessons) if lessons else "General trading loss"
    
    async def _publish_training_decision(self, decision: Dict[str, Any]) -> None:
        """Publish decision to training channels for tracking"""
        try:
            # Determine channel based on training namespace
            channel = "training:agent_decisions"
            if self.training_session_id:
                channel = f"{channel}:{self.training_session_id}"
            
            # Publish decision
            await self.publish(channel, decision)
            
        except Exception as e:
            self.logger.error(f"Failed to publish training decision: {e}")
    
    def _reset_training_metrics(self) -> None:
        """Reset training metrics for new session"""
        self.training_metrics = {
            "decisions_made": 0,
            "successful_decisions": 0,
            "failed_decisions": 0,
            "total_pnl": 0.0,
            "confidence_sum": 0.0,
            "kelly_adherence_sum": 0.0
        }
        self.decision_buffer.clear()
        self.pending_decisions.clear()
    
    def _finalize_training_metrics(self) -> None:
        """Finalize training metrics at end of session"""
        try:
            # Calculate averages
            if self.training_metrics["decisions_made"] > 0:
                avg_confidence = self.training_metrics["confidence_sum"] / self.training_metrics["decisions_made"]
                avg_kelly_deviation = self.training_metrics["kelly_adherence_sum"] / self.training_metrics["decisions_made"]
                win_rate = self.training_metrics["successful_decisions"] / self.training_metrics["decisions_made"]
                
                self.logger.info(
                    f"Training session complete for {self.agent_name}: "
                    f"{self.training_metrics['decisions_made']} decisions, "
                    f"Win rate: {win_rate:.2%}, "
                    f"P&L: {self.training_metrics['total_pnl']:.2f}, "
                    f"Avg confidence: {avg_confidence:.2f}, "
                    f"Avg Kelly deviation: {avg_kelly_deviation:.2f}"
                )
            
            # Process any remaining buffered decisions
            asyncio.create_task(self._process_decision_buffer())
            
        except Exception as e:
            self.logger.error(f"Failed to finalize training metrics: {e}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get current training statistics"""
        stats = {
            "agent": self.agent_name,
            "training_mode": self.training_mode,
            "session_id": self.training_session_id,
            "metrics": dict(self.training_metrics),
            "pending_decisions": len(self.pending_decisions),
            "buffered_decisions": len(self.decision_buffer)
        }
        
        # Add win rate if decisions made
        if self.training_metrics["decisions_made"] > 0:
            stats["win_rate"] = self.training_metrics["successful_decisions"] / self.training_metrics["decisions_made"]
            stats["avg_confidence"] = self.training_metrics["confidence_sum"] / self.training_metrics["decisions_made"]
        
        return stats