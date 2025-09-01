"""
Agent Performance Analytics System

Comprehensive analytics and metrics tracking for agent training
performance across synthetic scenarios with Kelly Criterion validation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import asyncio
import json
import logging
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, deque
import statistics

from ..synthetic_data.storage.chromadb_manager import ChromaDBManager
from ..synthetic_data.preprocessing.nfl_dataset_processor import ProcessedNFLPlay


class MetricType(Enum):
    """Types of performance metrics tracked"""
    DECISION_QUALITY = "decision_quality"
    KELLY_ADHERENCE = "kelly_adherence" 
    RISK_MANAGEMENT = "risk_management"
    PROFITABILITY = "profitability"
    LEARNING_PROGRESS = "learning_progress"
    BEHAVIORAL_PATTERNS = "behavioral_patterns"


@dataclass
class DecisionMetrics:
    """Metrics for individual trading decisions"""
    decision_id: str
    agent_id: str
    scenario_id: str
    timestamp: datetime
    market_ticker: str
    decision_type: str  # "buy", "sell", "hold"
    confidence: float
    expected_value: float
    kelly_fraction: float
    actual_kelly_used: float
    position_size: float
    outcome: Optional[float] = None  # Actual P&L
    market_efficiency: float = 0.8
    information_advantage: float = 0.0
    execution_latency: float = 0.0
    
    @property
    def kelly_deviation(self) -> float:
        """How much the agent deviated from optimal Kelly"""
        return abs(self.actual_kelly_used - self.kelly_fraction)
    
    @property
    def risk_adjusted_return(self) -> float:
        """Return adjusted for risk taken"""
        if self.outcome is None or self.position_size == 0:
            return 0.0
        return self.outcome / self.position_size


@dataclass
class AgentPerformanceSnapshot:
    """Performance snapshot for an agent at a point in time"""
    agent_id: str
    timestamp: datetime
    scenarios_completed: int
    total_decisions: int
    win_rate: float
    total_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    avg_kelly_deviation: float
    avg_confidence: float
    learning_velocity: float  # Rate of improvement
    behavioral_consistency: float
    risk_score: float
    decision_speed: float  # Avg latency
    
    # Advanced metrics
    kelly_adherence_score: float = 0.0
    information_utilization: float = 0.0
    pattern_recognition_score: float = 0.0
    edge_case_performance: float = 0.0


class AgentAnalytics:
    """
    Comprehensive analytics engine for agent training performance.
    
    Tracks decision quality, Kelly Criterion adherence, learning progress,
    and behavioral patterns across synthetic training scenarios.
    """
    
    def __init__(self, chroma_manager: ChromaDBManager):
        self.chroma_manager = chroma_manager
        self.decision_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.performance_snapshots: Dict[str, List[AgentPerformanceSnapshot]] = defaultdict(list)
        self.behavioral_patterns: Dict[str, Dict] = defaultdict(dict)
        self.logger = logging.getLogger(__name__)
        
        # Analytics configuration
        self.snapshot_interval = timedelta(hours=1)
        self.pattern_window = 100  # Decisions to analyze for patterns
        self.performance_windows = [10, 50, 100, 500]  # Different time horizons
    
    async def record_decision(self, decision: DecisionMetrics) -> None:
        """Record a single trading decision with comprehensive metrics"""
        try:
            # Store in memory for fast access
            self.decision_history[decision.agent_id].append(decision)
            
            # Store in ChromaDB for persistent analytics
            await self._store_decision_in_chromadb(decision)
            
            # Update behavioral patterns
            await self._update_behavioral_patterns(decision)
            
            # Check if snapshot needed
            await self._check_snapshot_trigger(decision.agent_id)
            
        except Exception as e:
            self.logger.error(f"Failed to record decision {decision.decision_id}: {e}")
            raise
    
    async def _store_decision_in_chromadb(self, decision: DecisionMetrics) -> None:
        """Store decision metrics in ChromaDB for semantic search"""
        try:
            # Create searchable description
            description = self._create_decision_description(decision)
            
            # Prepare metadata
            metadata = {
                "agent_id": decision.agent_id,
                "scenario_id": decision.scenario_id,
                "decision_type": decision.decision_type,
                "market_ticker": decision.market_ticker,
                "timestamp": decision.timestamp.isoformat(),
                "confidence": decision.confidence,
                "kelly_fraction": decision.kelly_fraction,
                "actual_kelly_used": decision.actual_kelly_used,
                "kelly_deviation": decision.kelly_deviation,
                "position_size": decision.position_size,
                "expected_value": decision.expected_value,
                "outcome": decision.outcome if decision.outcome is not None else 0.0,
                "risk_adjusted_return": decision.risk_adjusted_return,
                "execution_latency": decision.execution_latency,
                "market_efficiency": decision.market_efficiency,
                "information_advantage": decision.information_advantage
            }
            
            # Store in agent decisions collection
            decisions_collection = self.chroma_manager.get_collection("agent_decisions")
            if not decisions_collection:
                decisions_collection = await self.chroma_manager.create_collection(
                    "agent_decisions", 
                    "Agent trading decisions with performance metrics"
                )
            
            decisions_collection.add(
                ids=[decision.decision_id],
                documents=[description],
                metadatas=[metadata]
            )
            
        except Exception as e:
            self.logger.error(f"Failed to store decision in ChromaDB: {e}")
    
    def _create_decision_description(self, decision: DecisionMetrics) -> str:
        """Create semantic description of decision for ChromaDB storage"""
        outcome_desc = "profitable" if decision.outcome and decision.outcome > 0 else "losing" if decision.outcome and decision.outcome < 0 else "pending"
        kelly_adherence = "optimal" if decision.kelly_deviation < 0.1 else "suboptimal"
        confidence_level = "high" if decision.confidence > 0.7 else "medium" if decision.confidence > 0.4 else "low"
        
        return (f"{decision.decision_type.upper()} decision on {decision.market_ticker} "
                f"with {confidence_level} confidence ({decision.confidence:.2f}). "
                f"Used {kelly_adherence} Kelly sizing (deviation: {decision.kelly_deviation:.3f}). "
                f"Position size: {decision.position_size:.2f}, Expected value: {decision.expected_value:.3f}. "
                f"Market efficiency: {decision.market_efficiency:.2f}, Information advantage: {decision.information_advantage:.3f}. "
                f"Execution latency: {decision.execution_latency:.3f}s. Outcome: {outcome_desc}")
    
    async def _update_behavioral_patterns(self, decision: DecisionMetrics) -> None:
        """Update behavioral pattern analysis for the agent"""
        agent_decisions = self.decision_history[decision.agent_id]
        
        if len(agent_decisions) < 10:
            return  # Need minimum decisions for pattern analysis
        
        # Analyze recent decisions for patterns
        recent_decisions = list(agent_decisions)[-self.pattern_window:]
        
        patterns = {
            "avg_confidence": statistics.mean([d.confidence for d in recent_decisions]),
            "confidence_volatility": statistics.stdev([d.confidence for d in recent_decisions]) if len(recent_decisions) > 1 else 0,
            "kelly_consistency": 1 - statistics.mean([d.kelly_deviation for d in recent_decisions]),
            "decision_type_distribution": self._calculate_decision_distribution(recent_decisions),
            "risk_taking_tendency": statistics.mean([d.position_size for d in recent_decisions]),
            "market_timing_score": await self._calculate_market_timing_score(recent_decisions),
            "learning_trend": self._calculate_learning_trend(recent_decisions),
            "execution_speed_trend": self._calculate_speed_trend(recent_decisions)
        }
        
        self.behavioral_patterns[decision.agent_id] = patterns
    
    def _calculate_decision_distribution(self, decisions: List[DecisionMetrics]) -> Dict[str, float]:
        """Calculate distribution of decision types"""
        total = len(decisions)
        if total == 0:
            return {"buy": 0, "sell": 0, "hold": 0}
        
        counts = defaultdict(int)
        for decision in decisions:
            counts[decision.decision_type] += 1
        
        return {decision_type: count / total for decision_type, count in counts.items()}
    
    async def _calculate_market_timing_score(self, decisions: List[DecisionMetrics]) -> float:
        """Calculate how well agent times market entries"""
        if len(decisions) < 5:
            return 0.5  # Neutral score
        
        # Analyze correlation between confidence and outcomes
        profitable_decisions = [d for d in decisions if d.outcome and d.outcome > 0]
        if not profitable_decisions:
            return 0.3  # Below average
        
        # High confidence decisions should be more profitable
        high_conf_decisions = [d for d in decisions if d.confidence > 0.7]
        if not high_conf_decisions:
            return 0.4
        
        high_conf_profitability = sum(1 for d in high_conf_decisions if d.outcome and d.outcome > 0) / len(high_conf_decisions)
        return min(high_conf_profitability * 1.2, 1.0)  # Cap at 1.0
    
    def _calculate_learning_trend(self, decisions: List[DecisionMetrics]) -> float:
        """Calculate if agent is improving over time"""
        if len(decisions) < 20:
            return 0.0
        
        # Split into early and recent halves
        mid_point = len(decisions) // 2
        early_decisions = decisions[:mid_point]
        recent_decisions = decisions[mid_point:]
        
        # Compare performance metrics
        early_avg_deviation = statistics.mean([d.kelly_deviation for d in early_decisions])
        recent_avg_deviation = statistics.mean([d.kelly_deviation for d in recent_decisions])
        
        early_avg_confidence = statistics.mean([d.confidence for d in early_decisions])
        recent_avg_confidence = statistics.mean([d.confidence for d in recent_decisions])
        
        # Improvement is less deviation and higher confidence
        deviation_improvement = max(0, early_avg_deviation - recent_avg_deviation)
        confidence_improvement = max(0, recent_avg_confidence - early_avg_confidence)
        
        return min((deviation_improvement + confidence_improvement) / 2, 1.0)
    
    def _calculate_speed_trend(self, decisions: List[DecisionMetrics]) -> float:
        """Calculate trend in execution speed"""
        latencies = [d.execution_latency for d in decisions if d.execution_latency > 0]
        if len(latencies) < 5:
            return 0.0
        
        # Simple linear trend
        x = range(len(latencies))
        trend = np.polyfit(x, latencies, 1)[0]
        
        # Negative trend (getting faster) is good
        return max(0, -trend)
    
    async def _check_snapshot_trigger(self, agent_id: str) -> None:
        """Check if it's time to create a performance snapshot"""
        last_snapshot = self.performance_snapshots[agent_id][-1] if self.performance_snapshots[agent_id] else None
        
        if (not last_snapshot or 
            datetime.now() - last_snapshot.timestamp >= self.snapshot_interval):
            await self.create_performance_snapshot(agent_id)
    
    async def create_performance_snapshot(self, agent_id: str) -> AgentPerformanceSnapshot:
        """Create comprehensive performance snapshot for an agent"""
        try:
            decisions = list(self.decision_history[agent_id])
            if not decisions:
                # Return empty snapshot
                return AgentPerformanceSnapshot(
                    agent_id=agent_id,
                    timestamp=datetime.now(),
                    scenarios_completed=0,
                    total_decisions=0,
                    win_rate=0.0,
                    total_pnl=0.0,
                    sharpe_ratio=0.0,
                    max_drawdown=0.0,
                    avg_kelly_deviation=0.0,
                    avg_confidence=0.0,
                    learning_velocity=0.0,
                    behavioral_consistency=0.0,
                    risk_score=0.0,
                    decision_speed=0.0
                )
            
            # Calculate basic metrics
            completed_decisions = [d for d in decisions if d.outcome is not None]
            win_rate = sum(1 for d in completed_decisions if d.outcome > 0) / len(completed_decisions) if completed_decisions else 0
            total_pnl = sum(d.outcome for d in completed_decisions if d.outcome is not None)
            
            # Calculate Sharpe ratio
            returns = [d.risk_adjusted_return for d in completed_decisions if d.outcome is not None]
            sharpe_ratio = (statistics.mean(returns) / statistics.stdev(returns)) if len(returns) > 1 and statistics.stdev(returns) > 0 else 0
            
            # Calculate max drawdown
            max_drawdown = self._calculate_max_drawdown(completed_decisions)
            
            # Kelly metrics
            avg_kelly_deviation = statistics.mean([d.kelly_deviation for d in decisions])
            kelly_adherence_score = max(0, 1 - avg_kelly_deviation)
            
            # Behavioral metrics
            avg_confidence = statistics.mean([d.confidence for d in decisions])
            behavioral_consistency = self._calculate_behavioral_consistency(decisions)
            
            # Advanced metrics
            patterns = self.behavioral_patterns.get(agent_id, {})
            learning_velocity = patterns.get("learning_trend", 0.0)
            decision_speed = statistics.mean([d.execution_latency for d in decisions if d.execution_latency > 0]) or 0.0
            
            # Risk score
            risk_score = self._calculate_risk_score(decisions)
            
            # Information utilization
            information_utilization = statistics.mean([d.information_advantage for d in decisions])
            
            # Count unique scenarios
            unique_scenarios = len(set(d.scenario_id for d in decisions))
            
            snapshot = AgentPerformanceSnapshot(
                agent_id=agent_id,
                timestamp=datetime.now(),
                scenarios_completed=unique_scenarios,
                total_decisions=len(decisions),
                win_rate=win_rate,
                total_pnl=total_pnl,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                avg_kelly_deviation=avg_kelly_deviation,
                avg_confidence=avg_confidence,
                learning_velocity=learning_velocity,
                behavioral_consistency=behavioral_consistency,
                risk_score=risk_score,
                decision_speed=decision_speed,
                kelly_adherence_score=kelly_adherence_score,
                information_utilization=information_utilization,
                pattern_recognition_score=patterns.get("market_timing_score", 0.5),
                edge_case_performance=await self._calculate_edge_case_performance(agent_id)
            )
            
            self.performance_snapshots[agent_id].append(snapshot)
            await self._store_snapshot_in_chromadb(snapshot)
            
            return snapshot
            
        except Exception as e:
            self.logger.error(f"Failed to create performance snapshot for {agent_id}: {e}")
            raise
    
    def _calculate_max_drawdown(self, decisions: List[DecisionMetrics]) -> float:
        """Calculate maximum drawdown from peak equity"""
        if not decisions:
            return 0.0
        
        cumulative_pnl = 0.0
        running_max = 0.0
        max_drawdown = 0.0
        
        for decision in decisions:
            if decision.outcome is not None:
                cumulative_pnl += decision.outcome
                running_max = max(running_max, cumulative_pnl)
                drawdown = running_max - cumulative_pnl
                max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def _calculate_behavioral_consistency(self, decisions: List[DecisionMetrics]) -> float:
        """Calculate how consistent the agent's behavior is"""
        if len(decisions) < 10:
            return 0.5
        
        # Measure consistency in confidence levels
        confidence_std = statistics.stdev([d.confidence for d in decisions])
        confidence_consistency = max(0, 1 - confidence_std)
        
        # Measure consistency in Kelly adherence
        kelly_deviations = [d.kelly_deviation for d in decisions]
        kelly_std = statistics.stdev(kelly_deviations) if len(kelly_deviations) > 1 else 0
        kelly_consistency = max(0, 1 - kelly_std)
        
        return (confidence_consistency + kelly_consistency) / 2
    
    def _calculate_risk_score(self, decisions: List[DecisionMetrics]) -> float:
        """Calculate overall risk score (0 = low risk, 1 = high risk)"""
        if not decisions:
            return 0.5
        
        # Average position size relative to Kelly
        avg_position_ratio = statistics.mean([d.actual_kelly_used for d in decisions])
        position_risk = min(avg_position_ratio / 0.25, 1.0)  # Normalize to 25% Kelly
        
        # Kelly deviation risk
        kelly_risk = statistics.mean([d.kelly_deviation for d in decisions])
        
        # Concentration risk (are decisions spread across markets?)
        unique_markets = len(set(d.market_ticker for d in decisions))
        concentration_risk = max(0, 1 - unique_markets / 10)  # Normalize to 10 markets
        
        return (position_risk + kelly_risk + concentration_risk) / 3
    
    async def _calculate_edge_case_performance(self, agent_id: str) -> float:
        """Calculate performance specifically on edge case scenarios"""
        try:
            # Query ChromaDB for edge case scenarios
            edge_case_collection = self.chroma_manager.get_collection("training_scenarios")
            if not edge_case_collection:
                return 0.5  # Neutral score if no data
            
            # Search for edge case scenarios
            edge_cases = edge_case_collection.query(
                query_texts=["edge case", "rare event", "unusual situation", "outlier scenario"],
                where={"agent_id": agent_id},
                n_results=100
            )
            
            if not edge_cases['documents']:
                return 0.5  # No edge cases found
            
            # Analyze performance on these scenarios
            edge_case_decisions = [d for d in self.decision_history[agent_id] 
                                 if any(scenario_id in edge_cases['ids'] for scenario_id in edge_cases['ids'])]
            
            if not edge_case_decisions:
                return 0.5
            
            # Calculate win rate on edge cases
            completed_edge_decisions = [d for d in edge_case_decisions if d.outcome is not None]
            if not completed_edge_decisions:
                return 0.5
            
            edge_win_rate = sum(1 for d in completed_edge_decisions if d.outcome > 0) / len(completed_edge_decisions)
            return edge_win_rate
            
        except Exception as e:
            self.logger.error(f"Failed to calculate edge case performance: {e}")
            return 0.5
    
    async def _store_snapshot_in_chromadb(self, snapshot: AgentPerformanceSnapshot) -> None:
        """Store performance snapshot in ChromaDB"""
        try:
            description = (f"Performance snapshot for agent {snapshot.agent_id}: "
                         f"{snapshot.total_decisions} decisions across {snapshot.scenarios_completed} scenarios. "
                         f"Win rate: {snapshot.win_rate:.2f}, Total P&L: {snapshot.total_pnl:.2f}, "
                         f"Sharpe: {snapshot.sharpe_ratio:.2f}, Max DD: {snapshot.max_drawdown:.2f}. "
                         f"Kelly adherence: {snapshot.kelly_adherence_score:.2f}, "
                         f"Learning velocity: {snapshot.learning_velocity:.2f}")
            
            metadata = {
                "agent_id": snapshot.agent_id,
                "timestamp": snapshot.timestamp.isoformat(),
                "scenarios_completed": snapshot.scenarios_completed,
                "total_decisions": snapshot.total_decisions,
                "win_rate": snapshot.win_rate,
                "total_pnl": snapshot.total_pnl,
                "sharpe_ratio": snapshot.sharpe_ratio,
                "max_drawdown": snapshot.max_drawdown,
                "avg_kelly_deviation": snapshot.avg_kelly_deviation,
                "kelly_adherence_score": snapshot.kelly_adherence_score,
                "learning_velocity": snapshot.learning_velocity,
                "risk_score": snapshot.risk_score
            }
            
            # Store in performance snapshots collection
            snapshots_collection = self.chroma_manager.get_collection("performance_snapshots")
            if not snapshots_collection:
                snapshots_collection = await self.chroma_manager.create_collection(
                    "performance_snapshots",
                    "Agent performance snapshots over time"
                )
            
            snapshot_id = f"{snapshot.agent_id}_{snapshot.timestamp.strftime('%Y%m%d_%H%M%S')}"
            snapshots_collection.add(
                ids=[snapshot_id],
                documents=[description],
                metadatas=[metadata]
            )
            
        except Exception as e:
            self.logger.error(f"Failed to store snapshot in ChromaDB: {e}")
    
    async def get_agent_analytics(self, agent_id: str, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get comprehensive analytics for a specific agent"""
        try:
            decisions = list(self.decision_history[agent_id])
            
            # Filter by time window if specified
            if time_window:
                cutoff_time = datetime.now() - time_window
                decisions = [d for d in decisions if d.timestamp >= cutoff_time]
            
            if not decisions:
                return {"error": f"No decisions found for agent {agent_id}"}
            
            # Get latest snapshot
            latest_snapshot = self.performance_snapshots[agent_id][-1] if self.performance_snapshots[agent_id] else None
            
            # Calculate detailed metrics
            analytics = {
                "agent_id": agent_id,
                "analysis_timestamp": datetime.now().isoformat(),
                "time_window": str(time_window) if time_window else "all_time",
                "total_decisions": len(decisions),
                "latest_snapshot": {
                    "timestamp": latest_snapshot.timestamp.isoformat() if latest_snapshot else None,
                    "win_rate": latest_snapshot.win_rate if latest_snapshot else 0,
                    "total_pnl": latest_snapshot.total_pnl if latest_snapshot else 0,
                    "sharpe_ratio": latest_snapshot.sharpe_ratio if latest_snapshot else 0,
                    "kelly_adherence_score": latest_snapshot.kelly_adherence_score if latest_snapshot else 0,
                    "learning_velocity": latest_snapshot.learning_velocity if latest_snapshot else 0
                } if latest_snapshot else None,
                
                "behavioral_patterns": self.behavioral_patterns.get(agent_id, {}),
                
                "decision_breakdown": self._get_decision_breakdown(decisions),
                "performance_by_market": self._get_performance_by_market(decisions),
                "kelly_analysis": self._get_kelly_analysis(decisions),
                "confidence_analysis": self._get_confidence_analysis(decisions),
                "learning_progression": await self._get_learning_progression(agent_id),
                "risk_analysis": self._get_risk_analysis(decisions)
            }
            
            return analytics
            
        except Exception as e:
            self.logger.error(f"Failed to get analytics for {agent_id}: {e}")
            return {"error": str(e)}
    
    def _get_decision_breakdown(self, decisions: List[DecisionMetrics]) -> Dict[str, Any]:
        """Detailed breakdown of decision types and outcomes"""
        breakdown = defaultdict(lambda: {"count": 0, "wins": 0, "total_pnl": 0.0})
        
        for decision in decisions:
            key = decision.decision_type
            breakdown[key]["count"] += 1
            if decision.outcome is not None:
                if decision.outcome > 0:
                    breakdown[key]["wins"] += 1
                breakdown[key]["total_pnl"] += decision.outcome
        
        # Calculate win rates
        for key in breakdown:
            completed = breakdown[key]["count"]
            breakdown[key]["win_rate"] = breakdown[key]["wins"] / completed if completed > 0 else 0
        
        return dict(breakdown)
    
    def _get_performance_by_market(self, decisions: List[DecisionMetrics]) -> Dict[str, Any]:
        """Performance breakdown by market ticker"""
        market_performance = defaultdict(lambda: {"decisions": 0, "wins": 0, "total_pnl": 0.0, "avg_confidence": 0.0})
        
        for decision in decisions:
            market = decision.market_ticker
            market_performance[market]["decisions"] += 1
            market_performance[market]["avg_confidence"] += decision.confidence
            
            if decision.outcome is not None:
                if decision.outcome > 0:
                    market_performance[market]["wins"] += 1
                market_performance[market]["total_pnl"] += decision.outcome
        
        # Finalize calculations
        for market in market_performance:
            data = market_performance[market]
            data["avg_confidence"] /= data["decisions"] if data["decisions"] > 0 else 1
            data["win_rate"] = data["wins"] / data["decisions"] if data["decisions"] > 0 else 0
        
        return dict(market_performance)
    
    def _get_kelly_analysis(self, decisions: List[DecisionMetrics]) -> Dict[str, Any]:
        """Detailed analysis of Kelly Criterion adherence"""
        kelly_deviations = [d.kelly_deviation for d in decisions]
        kelly_fractions = [d.kelly_fraction for d in decisions]
        actual_kelly_used = [d.actual_kelly_used for d in decisions]
        
        return {
            "avg_kelly_fraction": statistics.mean(kelly_fractions),
            "avg_actual_kelly": statistics.mean(actual_kelly_used),
            "avg_deviation": statistics.mean(kelly_deviations),
            "deviation_std": statistics.stdev(kelly_deviations) if len(kelly_deviations) > 1 else 0,
            "optimal_decisions": sum(1 for d in kelly_deviations if d < 0.1),
            "suboptimal_decisions": sum(1 for d in kelly_deviations if d >= 0.1),
            "adherence_score": max(0, 1 - statistics.mean(kelly_deviations)),
            "over_betting_frequency": sum(1 for d in decisions if d.actual_kelly_used > d.kelly_fraction) / len(decisions),
            "under_betting_frequency": sum(1 for d in decisions if d.actual_kelly_used < d.kelly_fraction) / len(decisions)
        }
    
    def _get_confidence_analysis(self, decisions: List[DecisionMetrics]) -> Dict[str, Any]:
        """Analysis of agent confidence patterns"""
        confidences = [d.confidence for d in decisions]
        
        # Confidence vs outcome correlation
        completed_decisions = [d for d in decisions if d.outcome is not None]
        high_conf_decisions = [d for d in completed_decisions if d.confidence > 0.7]
        low_conf_decisions = [d for d in completed_decisions if d.confidence < 0.4]
        
        return {
            "avg_confidence": statistics.mean(confidences),
            "confidence_std": statistics.stdev(confidences) if len(confidences) > 1 else 0,
            "high_confidence_decisions": len(high_conf_decisions),
            "low_confidence_decisions": len(low_conf_decisions),
            "high_conf_win_rate": sum(1 for d in high_conf_decisions if d.outcome > 0) / len(high_conf_decisions) if high_conf_decisions else 0,
            "low_conf_win_rate": sum(1 for d in low_conf_decisions if d.outcome > 0) / len(low_conf_decisions) if low_conf_decisions else 0,
            "confidence_calibration": self._calculate_confidence_calibration(completed_decisions)
        }
    
    def _calculate_confidence_calibration(self, decisions: List[DecisionMetrics]) -> Dict[str, float]:
        """Calculate how well calibrated the agent's confidence is"""
        if len(decisions) < 10:
            return {"score": 0.5, "note": "Insufficient data"}
        
        # Bucket decisions by confidence level
        buckets = {"0.0-0.2": [], "0.2-0.4": [], "0.4-0.6": [], "0.6-0.8": [], "0.8-1.0": []}
        
        for decision in decisions:
            if decision.confidence <= 0.2:
                buckets["0.0-0.2"].append(decision)
            elif decision.confidence <= 0.4:
                buckets["0.2-0.4"].append(decision)
            elif decision.confidence <= 0.6:
                buckets["0.4-0.6"].append(decision)
            elif decision.confidence <= 0.8:
                buckets["0.6-0.8"].append(decision)
            else:
                buckets["0.8-1.0"].append(decision)
        
        calibration_scores = []
        for bucket_name, bucket_decisions in buckets.items():
            if not bucket_decisions:
                continue
            
            expected_win_rate = sum(d.confidence for d in bucket_decisions) / len(bucket_decisions)
            actual_win_rate = sum(1 for d in bucket_decisions if d.outcome and d.outcome > 0) / len(bucket_decisions)
            
            # Calibration is how close actual matches expected
            calibration_scores.append(1 - abs(expected_win_rate - actual_win_rate))
        
        overall_calibration = statistics.mean(calibration_scores) if calibration_scores else 0.5
        
        return {
            "score": overall_calibration,
            "bucket_analysis": {k: len(v) for k, v in buckets.items() if v}
        }
    
    async def _get_learning_progression(self, agent_id: str) -> Dict[str, Any]:
        """Analyze learning progression over time"""
        snapshots = self.performance_snapshots[agent_id]
        if len(snapshots) < 2:
            return {"note": "Insufficient snapshots for progression analysis"}
        
        # Track key metrics over time
        timestamps = [s.timestamp for s in snapshots]
        win_rates = [s.win_rate for s in snapshots]
        kelly_scores = [s.kelly_adherence_score for s in snapshots]
        learning_velocities = [s.learning_velocity for s in snapshots]
        
        return {
            "snapshot_count": len(snapshots),
            "time_span": str(timestamps[-1] - timestamps[0]),
            "win_rate_trend": self._calculate_trend(win_rates),
            "kelly_adherence_trend": self._calculate_trend(kelly_scores),
            "learning_velocity_trend": self._calculate_trend(learning_velocities),
            "overall_improvement_score": statistics.mean([s.learning_velocity for s in snapshots[-5:]]) if len(snapshots) >= 5 else 0
        }
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, Any]:
        """Calculate trend in a series of values"""
        if len(values) < 3:
            return {"trend": "insufficient_data"}
        
        x = range(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if abs(slope) < 0.001:
            trend = "stable"
        elif slope > 0:
            trend = "improving"
        else:
            trend = "declining"
        
        return {
            "trend": trend,
            "slope": slope,
            "start_value": values[0],
            "end_value": values[-1],
            "change": values[-1] - values[0]
        }
    
    def _get_risk_analysis(self, decisions: List[DecisionMetrics]) -> Dict[str, Any]:
        """Comprehensive risk analysis"""
        if not decisions:
            return {"error": "No decisions to analyze"}
        
        position_sizes = [d.position_size for d in decisions]
        outcomes = [d.outcome for d in decisions if d.outcome is not None]
        
        return {
            "avg_position_size": statistics.mean(position_sizes),
            "max_position_size": max(position_sizes),
            "position_size_std": statistics.stdev(position_sizes) if len(position_sizes) > 1 else 0,
            "largest_loss": min(outcomes) if outcomes else 0,
            "largest_gain": max(outcomes) if outcomes else 0,
            "risk_reward_ratio": abs(max(outcomes)) / abs(min(outcomes)) if outcomes and min(outcomes) < 0 else 0,
            "consecutive_losses": self._calculate_max_consecutive_losses(decisions),
            "var_95": np.percentile(outcomes, 5) if outcomes else 0,  # Value at Risk 95th percentile
            "risk_score": self._calculate_risk_score(decisions)
        }
    
    def _calculate_max_consecutive_losses(self, decisions: List[DecisionMetrics]) -> int:
        """Calculate maximum consecutive losing trades"""
        max_consecutive = 0
        current_consecutive = 0
        
        for decision in decisions:
            if decision.outcome is not None:
                if decision.outcome <= 0:
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else:
                    current_consecutive = 0
        
        return max_consecutive
    
    async def generate_training_report(self, agent_ids: List[str] = None, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Generate comprehensive training report for specified agents"""
        try:
            target_agents = agent_ids or list(self.decision_history.keys())
            
            if not target_agents:
                return {"error": "No agents found"}
            
            report = {
                "report_timestamp": datetime.now().isoformat(),
                "time_window": str(time_window) if time_window else "all_time",
                "agents_analyzed": len(target_agents),
                "summary": {},
                "agent_details": {},
                "comparative_analysis": {},
                "recommendations": []
            }
            
            # Gather analytics for each agent
            all_agent_analytics = {}
            for agent_id in target_agents:
                agent_analytics = await self.get_agent_analytics(agent_id, time_window)
                all_agent_analytics[agent_id] = agent_analytics
                report["agent_details"][agent_id] = agent_analytics
            
            # Generate summary statistics
            report["summary"] = self._generate_summary_statistics(all_agent_analytics)
            
            # Comparative analysis
            report["comparative_analysis"] = self._generate_comparative_analysis(all_agent_analytics)
            
            # Generate recommendations
            report["recommendations"] = await self._generate_training_recommendations(all_agent_analytics)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate training report: {e}")
            return {"error": str(e)}
    
    def _generate_summary_statistics(self, all_analytics: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate summary statistics across all agents"""
        valid_analytics = [a for a in all_analytics.values() if "error" not in a]
        
        if not valid_analytics:
            return {"error": "No valid analytics data"}
        
        total_decisions = sum(a["total_decisions"] for a in valid_analytics)
        
        # Aggregate latest snapshots
        latest_snapshots = [a["latest_snapshot"] for a in valid_analytics if a["latest_snapshot"]]
        
        if not latest_snapshots:
            return {"total_decisions": total_decisions, "note": "No snapshot data available"}
        
        return {
            "total_decisions": total_decisions,
            "avg_win_rate": statistics.mean([s["win_rate"] for s in latest_snapshots]),
            "total_pnl": sum(s["total_pnl"] for s in latest_snapshots),
            "avg_sharpe_ratio": statistics.mean([s["sharpe_ratio"] for s in latest_snapshots]),
            "avg_kelly_adherence": statistics.mean([s["kelly_adherence_score"] for s in latest_snapshots]),
            "avg_learning_velocity": statistics.mean([s["learning_velocity"] for s in latest_snapshots]),
            "best_performer": max(latest_snapshots, key=lambda s: s["total_pnl"])["agent_id"] if latest_snapshots else None,
            "fastest_learner": max(latest_snapshots, key=lambda s: s["learning_velocity"])["agent_id"] if latest_snapshots else None
        }
    
    def _generate_comparative_analysis(self, all_analytics: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate comparative analysis between agents"""
        valid_analytics = {k: v for k, v in all_analytics.items() if "error" not in v}
        
        if len(valid_analytics) < 2:
            return {"note": "Need at least 2 agents for comparison"}
        
        # Compare key metrics
        comparisons = {}
        
        for metric in ["win_rate", "total_pnl", "sharpe_ratio", "kelly_adherence_score", "learning_velocity"]:
            values = {}
            for agent_id, analytics in valid_analytics.items():
                if analytics.get("latest_snapshot") and analytics["latest_snapshot"].get(metric) is not None:
                    values[agent_id] = analytics["latest_snapshot"][metric]
            
            if values:
                comparisons[metric] = {
                    "best": max(values, key=values.get),
                    "worst": min(values, key=values.get),
                    "best_value": max(values.values()),
                    "worst_value": min(values.values()),
                    "spread": max(values.values()) - min(values.values())
                }
        
        return comparisons
    
    async def _generate_training_recommendations(self, all_analytics: Dict[str, Dict]) -> List[str]:
        """Generate specific training recommendations based on analytics"""
        recommendations = []
        
        valid_analytics = {k: v for k, v in all_analytics.items() if "error" not in v}
        
        for agent_id, analytics in valid_analytics.items():
            latest_snapshot = analytics.get("latest_snapshot")
            if not latest_snapshot:
                continue
            
            # Kelly adherence recommendations
            if latest_snapshot["kelly_adherence_score"] < 0.7:
                recommendations.append(f"Agent {agent_id}: Improve Kelly Criterion adherence (current: {latest_snapshot['kelly_adherence_score']:.2f}). Consider additional training on position sizing.")
            
            # Learning velocity recommendations
            if latest_snapshot["learning_velocity"] < 0.3:
                recommendations.append(f"Agent {agent_id}: Low learning velocity ({latest_snapshot['learning_velocity']:.2f}). Increase scenario diversity or adjust learning parameters.")
            
            # Win rate recommendations
            if latest_snapshot["win_rate"] < 0.45:
                recommendations.append(f"Agent {agent_id}: Below-average win rate ({latest_snapshot['win_rate']:.2f}). Focus on signal quality and market timing training.")
            
            # Sharpe ratio recommendations
            if latest_snapshot["sharpe_ratio"] < 0.5:
                recommendations.append(f"Agent {agent_id}: Low risk-adjusted returns (Sharpe: {latest_snapshot['sharpe_ratio']:.2f}). Emphasize risk management training.")
        
        # Global recommendations
        if len(valid_analytics) > 1:
            best_performer = max(valid_analytics.items(), key=lambda x: x[1]["latest_snapshot"]["total_pnl"] if x[1].get("latest_snapshot") else 0)
            recommendations.append(f"Consider using Agent {best_performer[0]}'s strategies as training templates for other agents.")
        
        return recommendations
    
    async def export_analytics_data(self, agent_id: str, format: str = "json") -> str:
        """Export comprehensive analytics data for external analysis"""
        try:
            analytics = await self.get_agent_analytics(agent_id)
            
            if format.lower() == "json":
                return json.dumps(analytics, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            self.logger.error(f"Failed to export analytics data: {e}")
            raise