"""
Decision Tracking Middleware

Comprehensive decision tracking system that captures, analyzes, and stores
all agent trading decisions during both training and production modes.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import redis.asyncio as redis
import numpy as np

logger = logging.getLogger(__name__)


class DecisionType(Enum):
    """Types of trading decisions"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"
    ADJUST = "adjust"
    HEDGE = "hedge"


class DecisionOutcome(Enum):
    """Outcome status of decisions"""
    PENDING = "pending"
    PROFITABLE = "profitable"
    LOSS = "loss"
    BREAKEVEN = "breakeven"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class SignalSource(Enum):
    """Source of trading signals"""
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    NEWS = "news"
    PATTERN = "pattern"
    ML_MODEL = "ml_model"
    HYBRID = "hybrid"


@dataclass
class MarketContext:
    """Market conditions at decision time"""
    ticker: str
    price: float
    bid: float
    ask: float
    volume: int
    volatility: float
    momentum: float
    liquidity: float
    spread: float
    market_phase: str  # "pre_game", "in_game", "post_game"
    time_remaining: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return asdict(self)


@dataclass
class DecisionReasoning:
    """Reasoning behind a trading decision"""
    strategy: str
    signals_used: List[SignalSource]
    confidence_factors: Dict[str, float]
    risk_assessment: Dict[str, float]
    expected_edge: float
    time_horizon: str  # "seconds", "minutes", "hours", "days"
    correlation_considered: List[str]
    alternative_actions: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['signals_used'] = [s.value for s in self.signals_used]
        return data


@dataclass
class DecisionRecord:
    """Complete record of a trading decision"""
    # Metadata
    decision_id: str
    agent_id: str
    session_id: Optional[str]
    timestamp: datetime
    
    # Decision details
    decision_type: DecisionType
    market_ticker: str
    position_size: float
    entry_price: float
    confidence: float
    
    # Kelly Criterion
    kelly_fraction: float
    actual_kelly_used: float
    expected_value: float
    probability_win: float
    
    # Context
    market_context: MarketContext
    reasoning: DecisionReasoning
    
    # Outcome (updated later)
    outcome: DecisionOutcome = DecisionOutcome.PENDING
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    profit_loss: Optional[float] = None
    actual_probability: Optional[float] = None
    
    # Performance metrics
    sharpe_contribution: Optional[float] = None
    drawdown_impact: Optional[float] = None
    execution_latency_ms: Optional[float] = None
    slippage: Optional[float] = None
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    parent_decision_id: Optional[str] = None  # For linked decisions
    
    @property
    def kelly_deviation(self) -> float:
        """Calculate deviation from optimal Kelly"""
        return abs(self.actual_kelly_used - self.kelly_fraction)
    
    @property
    def risk_adjusted_return(self) -> Optional[float]:
        """Calculate risk-adjusted return"""
        if self.profit_loss is None or self.position_size == 0:
            return None
        return self.profit_loss / self.position_size
    
    @property
    def confidence_accuracy(self) -> Optional[float]:
        """How accurate was the confidence estimate"""
        if self.actual_probability is None:
            return None
        return 1 - abs(self.confidence - self.actual_probability)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['decision_type'] = self.decision_type.value
        data['timestamp'] = self.timestamp.isoformat()
        data['outcome'] = self.outcome.value
        data['market_context'] = self.market_context.to_dict()
        data['reasoning'] = self.reasoning.to_dict()
        if self.exit_timestamp:
            data['exit_timestamp'] = self.exit_timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DecisionRecord':
        """Create from dictionary"""
        data['decision_type'] = DecisionType(data['decision_type'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['outcome'] = DecisionOutcome(data['outcome'])
        data['market_context'] = MarketContext(**data['market_context'])
        
        # Convert signal sources back to enums
        reasoning_data = data['reasoning']
        reasoning_data['signals_used'] = [
            SignalSource(s) for s in reasoning_data['signals_used']
        ]
        data['reasoning'] = DecisionReasoning(**reasoning_data)
        
        if data.get('exit_timestamp'):
            data['exit_timestamp'] = datetime.fromisoformat(data['exit_timestamp'])
            
        return cls(**data)


@dataclass
class TrackingConfig:
    """Configuration for decision tracking"""
    # Tracking settings
    enabled: bool = True
    tracking_mode: str = "full"  # "full", "partial", "minimal"
    track_production: bool = True
    track_training: bool = True
    
    # Storage settings
    storage_backend: str = "redis"  # "redis", "postgres", "file", "hybrid"
    redis_url: str = "redis://localhost:6379"
    redis_ttl_hours: int = 24
    
    # Retention policy
    hot_storage_hours: int = 24  # In Redis
    warm_storage_days: int = 30  # In PostgreSQL
    cold_storage_days: int = 365  # In files
    
    # Performance settings
    batch_size: int = 100
    flush_interval_seconds: float = 5.0
    max_memory_decisions: int = 10000
    
    # Analytics settings
    real_time_analytics: bool = True
    pattern_detection: bool = True
    anomaly_detection: bool = True
    
    # Integration settings
    publish_decisions: bool = True
    decision_channel: str = "decisions:tracked"
    outcome_channel: str = "decisions:outcomes"


class DecisionTracker:
    """
    Central middleware for tracking all agent trading decisions.
    
    Intercepts, tracks, analyzes, and stores all trading decisions
    with full context and outcome tracking.
    """
    
    def __init__(self, config: TrackingConfig):
        self.config = config
        self.redis_client: Optional[redis.Redis] = None
        
        # In-memory storage
        self.pending_decisions: Dict[str, DecisionRecord] = {}
        self.completed_decisions: deque = deque(maxlen=config.max_memory_decisions)
        self.decision_buffer: List[DecisionRecord] = []
        
        # Analytics state
        self.decision_counts: Dict[str, int] = defaultdict(int)
        self.outcome_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.performance_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Callbacks
        self.decision_callbacks: List[Callable] = []
        self.outcome_callbacks: List[Callable] = []
        
        # Background tasks
        self.flush_task: Optional[asyncio.Task] = None
        self.analytics_task: Optional[asyncio.Task] = None
        
    async def initialize(self):
        """Initialize the decision tracker"""
        if self.config.storage_backend in ["redis", "hybrid"]:
            self.redis_client = redis.from_url(self.config.redis_url)
            
        # Start background tasks
        if self.config.enabled:
            self.flush_task = asyncio.create_task(self._flush_decisions_periodically())
            
        if self.config.real_time_analytics:
            self.analytics_task = asyncio.create_task(self._update_analytics_periodically())
            
        logger.info("Decision tracker initialized")
        
    async def track_decision(
        self,
        agent_id: str,
        decision_type: DecisionType,
        market_ticker: str,
        position_size: float,
        entry_price: float,
        confidence: float,
        kelly_fraction: float,
        actual_kelly_used: float,
        expected_value: float,
        probability_win: float,
        market_context: MarketContext,
        reasoning: DecisionReasoning,
        session_id: Optional[str] = None,
        execution_latency_ms: Optional[float] = None,
        tags: Optional[List[str]] = None,
        parent_decision_id: Optional[str] = None
    ) -> str:
        """
        Track a new trading decision.
        
        Args:
            agent_id: ID of the agent making the decision
            decision_type: Type of decision (buy/sell/hold)
            market_ticker: Market identifier
            position_size: Size of position
            entry_price: Entry price
            confidence: Confidence level (0-1)
            kelly_fraction: Optimal Kelly fraction
            actual_kelly_used: Actual Kelly fraction used
            expected_value: Expected value of trade
            probability_win: Probability of winning
            market_context: Current market conditions
            reasoning: Reasoning behind decision
            session_id: Optional training session ID
            execution_latency_ms: Execution latency in milliseconds
            tags: Optional tags for categorization
            parent_decision_id: ID of parent decision if linked
            
        Returns:
            Decision ID for tracking
        """
        if not self.config.enabled:
            return ""
            
        # Check if we should track this decision
        if session_id and not self.config.track_training:
            return ""
        if not session_id and not self.config.track_production:
            return ""
            
        # Create decision record
        decision_id = str(uuid.uuid4())
        
        record = DecisionRecord(
            decision_id=decision_id,
            agent_id=agent_id,
            session_id=session_id,
            timestamp=datetime.now(),
            decision_type=decision_type,
            market_ticker=market_ticker,
            position_size=position_size,
            entry_price=entry_price,
            confidence=confidence,
            kelly_fraction=kelly_fraction,
            actual_kelly_used=actual_kelly_used,
            expected_value=expected_value,
            probability_win=probability_win,
            market_context=market_context,
            reasoning=reasoning,
            execution_latency_ms=execution_latency_ms,
            tags=tags or [],
            parent_decision_id=parent_decision_id
        )
        
        # Store in pending decisions
        self.pending_decisions[decision_id] = record
        
        # Add to buffer for batch processing
        self.decision_buffer.append(record)
        
        # Update counters
        self.decision_counts[agent_id] += 1
        
        # Trigger callbacks
        for callback in self.decision_callbacks:
            asyncio.create_task(callback(record))
            
        # Publish to Redis if configured
        if self.config.publish_decisions and self.redis_client:
            await self.redis_client.publish(
                f"{self.config.decision_channel}:{agent_id}",
                json.dumps(record.to_dict())
            )
            
        # Flush if buffer is full
        if len(self.decision_buffer) >= self.config.batch_size:
            asyncio.create_task(self._flush_decisions())
            
        logger.debug(f"Tracked decision {decision_id} for agent {agent_id}")
        return decision_id
        
    async def update_outcome(
        self,
        decision_id: str,
        outcome: DecisionOutcome,
        exit_price: float,
        profit_loss: float,
        actual_probability: Optional[float] = None,
        sharpe_contribution: Optional[float] = None,
        drawdown_impact: Optional[float] = None,
        slippage: Optional[float] = None,
        notes: Optional[str] = None
    ) -> bool:
        """
        Update decision with actual outcome.
        
        Args:
            decision_id: ID of decision to update
            outcome: Outcome status
            exit_price: Exit price
            profit_loss: Actual profit/loss
            actual_probability: Actual probability (for calibration)
            sharpe_contribution: Contribution to Sharpe ratio
            drawdown_impact: Impact on drawdown
            slippage: Execution slippage
            notes: Optional notes
            
        Returns:
            True if updated successfully
        """
        # Find decision
        record = self.pending_decisions.get(decision_id)
        if not record:
            # Try to load from storage
            record = await self._load_decision(decision_id)
            if not record:
                logger.warning(f"Decision {decision_id} not found")
                return False
                
        # Update outcome
        record.outcome = outcome
        record.exit_price = exit_price
        record.exit_timestamp = datetime.now()
        record.profit_loss = profit_loss
        record.actual_probability = actual_probability
        record.sharpe_contribution = sharpe_contribution
        record.drawdown_impact = drawdown_impact
        record.slippage = slippage
        
        if notes:
            record.notes = notes
            
        # Move to completed
        if decision_id in self.pending_decisions:
            del self.pending_decisions[decision_id]
        self.completed_decisions.append(record)
        
        # Update outcome counters
        self.outcome_counts[record.agent_id][outcome.value] += 1
        
        # Trigger callbacks
        for callback in self.outcome_callbacks:
            asyncio.create_task(callback(record))
            
        # Publish outcome if configured
        if self.config.publish_decisions and self.redis_client:
            await self.redis_client.publish(
                f"{self.config.outcome_channel}:{record.agent_id}",
                json.dumps({
                    'decision_id': decision_id,
                    'outcome': outcome.value,
                    'profit_loss': profit_loss,
                    'timestamp': datetime.now().isoformat()
                })
            )
            
        # Store updated record
        await self._store_decision(record)
        
        logger.debug(f"Updated outcome for decision {decision_id}: {outcome.value}")
        return True
        
    async def get_decision_history(
        self,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        market_ticker: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        outcome_filter: Optional[DecisionOutcome] = None,
        limit: int = 100
    ) -> List[DecisionRecord]:
        """
        Retrieve historical decisions with filters.
        
        Args:
            agent_id: Filter by agent
            session_id: Filter by session
            market_ticker: Filter by market
            start_time: Start time filter
            end_time: End time filter
            outcome_filter: Filter by outcome
            limit: Maximum records to return
            
        Returns:
            List of matching decision records
        """
        decisions = []
        
        # Search in-memory decisions first
        for record in list(self.completed_decisions):
            if self._matches_filters(
                record, agent_id, session_id, market_ticker,
                start_time, end_time, outcome_filter
            ):
                decisions.append(record)
                if len(decisions) >= limit:
                    break
                    
        # If need more, search in storage
        if len(decisions) < limit and self.redis_client:
            stored_decisions = await self._search_stored_decisions(
                agent_id, session_id, market_ticker,
                start_time, end_time, outcome_filter,
                limit - len(decisions)
            )
            decisions.extend(stored_decisions)
            
        return decisions
        
    async def analyze_patterns(
        self,
        agent_id: str,
        window_size: int = 100
    ) -> Dict[str, Any]:
        """
        Analyze decision patterns for an agent.
        
        Args:
            agent_id: Agent to analyze
            window_size: Number of recent decisions to analyze
            
        Returns:
            Pattern analysis results
        """
        # Get recent decisions
        decisions = await self.get_decision_history(
            agent_id=agent_id,
            limit=window_size
        )
        
        if not decisions:
            return {"error": "No decisions found"}
            
        # Basic statistics
        total_decisions = len(decisions)
        profitable = sum(1 for d in decisions if d.outcome == DecisionOutcome.PROFITABLE)
        losses = sum(1 for d in decisions if d.outcome == DecisionOutcome.LOSS)
        
        # Kelly adherence
        kelly_deviations = [d.kelly_deviation for d in decisions]
        avg_kelly_deviation = np.mean(kelly_deviations) if kelly_deviations else 0
        
        # Confidence calibration
        confidence_scores = [d.confidence for d in decisions if d.actual_probability is not None]
        actual_probabilities = [d.actual_probability for d in decisions if d.actual_probability is not None]
        
        if confidence_scores and actual_probabilities:
            calibration_error = np.mean(np.abs(np.array(confidence_scores) - np.array(actual_probabilities)))
        else:
            calibration_error = None
            
        # Time patterns
        hour_distribution = defaultdict(int)
        for d in decisions:
            hour_distribution[d.timestamp.hour] += 1
            
        # Market patterns
        market_performance = defaultdict(lambda: {"count": 0, "profit": 0})
        for d in decisions:
            if d.profit_loss is not None:
                market_performance[d.market_ticker]["count"] += 1
                market_performance[d.market_ticker]["profit"] += d.profit_loss
                
        # Streak analysis
        current_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        
        for d in sorted(decisions, key=lambda x: x.timestamp):
            if d.outcome == DecisionOutcome.PROFITABLE:
                if current_streak >= 0:
                    current_streak += 1
                    max_win_streak = max(max_win_streak, current_streak)
                else:
                    current_streak = 1
            elif d.outcome == DecisionOutcome.LOSS:
                if current_streak <= 0:
                    current_streak -= 1
                    max_loss_streak = max(max_loss_streak, abs(current_streak))
                else:
                    current_streak = -1
                    
        return {
            "total_decisions": total_decisions,
            "win_rate": profitable / total_decisions if total_decisions > 0 else 0,
            "profitable_decisions": profitable,
            "losing_decisions": losses,
            "avg_kelly_deviation": avg_kelly_deviation,
            "calibration_error": calibration_error,
            "hour_distribution": dict(hour_distribution),
            "market_performance": dict(market_performance),
            "max_win_streak": max_win_streak,
            "max_loss_streak": max_loss_streak,
            "current_streak": current_streak
        }
        
    async def export_for_training(
        self,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        format: str = "json"
    ) -> Any:
        """
        Export decisions for model training.
        
        Args:
            agent_id: Filter by agent
            session_id: Filter by session
            format: Export format ("json", "parquet", "csv")
            
        Returns:
            Exported data in requested format
        """
        # Get all relevant decisions
        decisions = await self.get_decision_history(
            agent_id=agent_id,
            session_id=session_id,
            limit=100000  # Get all
        )
        
        if format == "json":
            return [d.to_dict() for d in decisions]
            
        elif format == "parquet":
            # Would need pandas/pyarrow for this
            # For now, return JSON with a note
            return {
                "format": "json",
                "note": "Parquet export requires pandas/pyarrow",
                "data": [d.to_dict() for d in decisions]
            }
            
        elif format == "csv":
            # Flatten to CSV-friendly format
            rows = []
            for d in decisions:
                row = {
                    "decision_id": d.decision_id,
                    "agent_id": d.agent_id,
                    "timestamp": d.timestamp.isoformat(),
                    "decision_type": d.decision_type.value,
                    "market_ticker": d.market_ticker,
                    "position_size": d.position_size,
                    "entry_price": d.entry_price,
                    "confidence": d.confidence,
                    "kelly_fraction": d.kelly_fraction,
                    "actual_kelly_used": d.actual_kelly_used,
                    "outcome": d.outcome.value,
                    "profit_loss": d.profit_loss
                }
                rows.append(row)
            return rows
            
        else:
            raise ValueError(f"Unsupported format: {format}")
            
    def add_decision_callback(self, callback: Callable):
        """Add callback for new decisions"""
        self.decision_callbacks.append(callback)
        
    def add_outcome_callback(self, callback: Callable):
        """Add callback for decision outcomes"""
        self.outcome_callbacks.append(callback)
        
    async def get_statistics(
        self,
        agent_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get tracking statistics.
        
        Args:
            agent_id: Optional agent filter
            
        Returns:
            Statistics dictionary
        """
        if agent_id:
            return {
                "agent_id": agent_id,
                "total_decisions": self.decision_counts.get(agent_id, 0),
                "outcomes": dict(self.outcome_counts.get(agent_id, {})),
                "pending_decisions": sum(
                    1 for d in self.pending_decisions.values()
                    if d.agent_id == agent_id
                ),
                "performance_metrics": self.performance_metrics.get(agent_id, {})
            }
        else:
            return {
                "total_decisions": sum(self.decision_counts.values()),
                "agents_tracked": len(self.decision_counts),
                "pending_decisions": len(self.pending_decisions),
                "completed_decisions": len(self.completed_decisions),
                "buffer_size": len(self.decision_buffer)
            }
            
    async def _flush_decisions(self):
        """Flush decision buffer to storage"""
        if not self.decision_buffer:
            return
            
        decisions_to_flush = self.decision_buffer.copy()
        self.decision_buffer.clear()
        
        for decision in decisions_to_flush:
            await self._store_decision(decision)
            
        logger.debug(f"Flushed {len(decisions_to_flush)} decisions to storage")
        
    async def _flush_decisions_periodically(self):
        """Background task to flush decisions periodically"""
        while True:
            await asyncio.sleep(self.config.flush_interval_seconds)
            await self._flush_decisions()
            
    async def _update_analytics_periodically(self):
        """Background task to update analytics"""
        while True:
            await asyncio.sleep(30)  # Update every 30 seconds
            
            for agent_id in self.decision_counts.keys():
                # Calculate performance metrics
                recent_decisions = await self.get_decision_history(
                    agent_id=agent_id,
                    limit=100
                )
                
                if recent_decisions:
                    profitable = sum(
                        1 for d in recent_decisions
                        if d.outcome == DecisionOutcome.PROFITABLE
                    )
                    total = len(recent_decisions)
                    
                    self.performance_metrics[agent_id] = {
                        "win_rate": profitable / total if total > 0 else 0,
                        "avg_confidence": np.mean([d.confidence for d in recent_decisions]),
                        "avg_kelly_deviation": np.mean([d.kelly_deviation for d in recent_decisions]),
                        "total_pnl": sum(d.profit_loss or 0 for d in recent_decisions)
                    }
                    
    async def _store_decision(self, decision: DecisionRecord):
        """Store decision in backend"""
        if self.redis_client:
            # Store in Redis with TTL
            key = f"decision:{decision.decision_id}"
            await self.redis_client.setex(
                key,
                self.config.redis_ttl_hours * 3600,
                json.dumps(decision.to_dict())
            )
            
            # Add to agent's decision set
            agent_key = f"agent:decisions:{decision.agent_id}"
            await self.redis_client.zadd(
                agent_key,
                {decision.decision_id: decision.timestamp.timestamp()}
            )
            
    async def _load_decision(self, decision_id: str) -> Optional[DecisionRecord]:
        """Load decision from storage"""
        if self.redis_client:
            key = f"decision:{decision_id}"
            data = await self.redis_client.get(key)
            if data:
                return DecisionRecord.from_dict(json.loads(data))
        return None
        
    async def _search_stored_decisions(
        self,
        agent_id: Optional[str],
        session_id: Optional[str],
        market_ticker: Optional[str],
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        outcome_filter: Optional[DecisionOutcome],
        limit: int
    ) -> List[DecisionRecord]:
        """Search for decisions in storage"""
        decisions = []
        
        if self.redis_client and agent_id:
            # Get decision IDs for agent
            agent_key = f"agent:decisions:{agent_id}"
            
            # Get decisions within time range
            min_score = start_time.timestamp() if start_time else "-inf"
            max_score = end_time.timestamp() if end_time else "+inf"
            
            decision_ids = await self.redis_client.zrangebyscore(
                agent_key,
                min_score,
                max_score,
                start=0,
                num=limit * 2  # Get extra to account for filters
            )
            
            # Load and filter decisions
            for decision_id in decision_ids:
                record = await self._load_decision(decision_id.decode() if isinstance(decision_id, bytes) else decision_id)
                if record and self._matches_filters(
                    record, agent_id, session_id, market_ticker,
                    start_time, end_time, outcome_filter
                ):
                    decisions.append(record)
                    if len(decisions) >= limit:
                        break
                        
        return decisions
        
    def _matches_filters(
        self,
        record: DecisionRecord,
        agent_id: Optional[str],
        session_id: Optional[str],
        market_ticker: Optional[str],
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        outcome_filter: Optional[DecisionOutcome]
    ) -> bool:
        """Check if record matches filters"""
        if agent_id and record.agent_id != agent_id:
            return False
        if session_id and record.session_id != session_id:
            return False
        if market_ticker and record.market_ticker != market_ticker:
            return False
        if start_time and record.timestamp < start_time:
            return False
        if end_time and record.timestamp > end_time:
            return False
        if outcome_filter and record.outcome != outcome_filter:
            return False
        return True
        
    async def cleanup(self):
        """Clean up resources"""
        # Cancel background tasks
        if self.flush_task:
            self.flush_task.cancel()
        if self.analytics_task:
            self.analytics_task.cancel()
            
        # Final flush
        await self._flush_decisions()
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
            
        logger.info("Decision tracker cleaned up")