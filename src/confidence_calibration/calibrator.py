"""
Confidence Calibration Engine

Calibrates agent confidence scores to improve decision-making accuracy
and uncertainty estimation in trading scenarios.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
import logging
import numpy as np
from collections import defaultdict, deque
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from ..training.agent_analytics import DecisionMetrics


class CalibrationMethod(Enum):
    """Methods for confidence calibration"""
    PLATT_SCALING = "platt_scaling"
    ISOTONIC_REGRESSION = "isotonic_regression"
    TEMPERATURE_SCALING = "temperature_scaling"
    BAYESIAN_CALIBRATION = "bayesian_calibration"
    HISTOGRAM_BINNING = "histogram_binning"


@dataclass
class ConfidenceScore:
    """Enhanced confidence score with uncertainty quantification"""
    raw_confidence: float  # Original agent confidence
    calibrated_confidence: float  # Calibrated confidence
    uncertainty: float  # Uncertainty estimate
    confidence_interval: Tuple[float, float]  # Confidence bounds
    method_used: CalibrationMethod
    sample_size: int  # Number of samples used for calibration
    
    @property
    def reliability(self) -> float:
        """Get reliability score based on sample size and uncertainty"""
        base_reliability = min(1.0, self.sample_size / 100)  # More samples = more reliable
        uncertainty_penalty = self.uncertainty * 0.5
        return max(0.0, base_reliability - uncertainty_penalty)


@dataclass
class CalibrationMetrics:
    """Metrics for evaluating calibration quality"""
    brier_score: float  # Lower is better
    reliability: float  # Expected Calibration Error (ECE)
    resolution: float  # Ability to discriminate
    sharpness: float  # Confidence in predictions
    calibration_slope: float  # Slope of calibration curve
    calibration_intercept: float  # Intercept of calibration curve
    sample_count: int
    
    @property
    def calibration_quality(self) -> str:
        """Overall calibration quality assessment"""
        if self.reliability < 0.05:
            return "excellent"
        elif self.reliability < 0.10:
            return "good"
        elif self.reliability < 0.20:
            return "fair"
        else:
            return "poor"


@dataclass
class CalibrationData:
    """Data structure for calibration training"""
    confidences: List[float]
    outcomes: List[bool]  # True for success, False for failure
    weights: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.confidences)
    
    @property
    def is_valid(self) -> bool:
        """Check if calibration data is valid"""
        return (len(self.confidences) == len(self.outcomes) and
                len(self.confidences) > 10 and  # Minimum sample size
                all(0 <= c <= 1 for c in self.confidences))


class UncertaintyQuantifier:
    """
    Quantifies uncertainty in agent decisions using multiple approaches.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Historical data for uncertainty estimation
        self.decision_history: deque = deque(maxlen=1000)
        self.outcome_patterns: Dict[str, List[float]] = defaultdict(list)
        
    def estimate_uncertainty(
        self,
        confidence: float,
        context: Dict[str, Any],
        method: str = "ensemble"
    ) -> float:
        """
        Estimate uncertainty for a given confidence score and context.
        
        Args:
            confidence: Raw confidence score
            context: Decision context (market conditions, agent state, etc.)
            method: Uncertainty estimation method
            
        Returns:
            Uncertainty estimate (0 = certain, 1 = maximum uncertainty)
        """
        try:
            if method == "ensemble":
                return self._ensemble_uncertainty(confidence, context)
            elif method == "variance":
                return self._variance_based_uncertainty(confidence, context)
            elif method == "entropy":
                return self._entropy_based_uncertainty(confidence, context)
            else:
                return self._default_uncertainty(confidence, context)
                
        except Exception as e:
            self.logger.error(f"Failed to estimate uncertainty: {e}")
            return 0.5  # Default moderate uncertainty
    
    def _ensemble_uncertainty(self, confidence: float, context: Dict[str, Any]) -> float:
        """Ensemble uncertainty estimation combining multiple methods"""
        try:
            # Get uncertainty from different methods
            variance_unc = self._variance_based_uncertainty(confidence, context)
            entropy_unc = self._entropy_based_uncertainty(confidence, context)
            pattern_unc = self._pattern_based_uncertainty(confidence, context)
            
            # Weighted combination
            weights = [0.4, 0.3, 0.3]
            uncertainties = [variance_unc, entropy_unc, pattern_unc]
            
            ensemble_uncertainty = sum(w * u for w, u in zip(weights, uncertainties))
            return min(1.0, max(0.0, ensemble_uncertainty))
            
        except Exception as e:
            self.logger.error(f"Ensemble uncertainty estimation failed: {e}")
            return 0.5
    
    def _variance_based_uncertainty(self, confidence: float, context: Dict[str, Any]) -> float:
        """Uncertainty based on confidence variance in similar situations"""
        try:
            # Find similar historical decisions
            similar_decisions = self._find_similar_decisions(context)
            
            if len(similar_decisions) < 5:
                return 0.7  # High uncertainty with limited data
            
            # Calculate confidence variance in similar situations
            confidences = [d["confidence"] for d in similar_decisions]
            variance = np.var(confidences)
            
            # Normalize variance to uncertainty scale
            uncertainty = min(1.0, variance * 2)  # Scale factor
            return uncertainty
            
        except Exception as e:
            self.logger.error(f"Variance-based uncertainty failed: {e}")
            return 0.5
    
    def _entropy_based_uncertainty(self, confidence: float, context: Dict[str, Any]) -> float:
        """Uncertainty based on information entropy"""
        try:
            # Create probability distribution
            p = confidence
            q = 1 - confidence
            
            # Calculate entropy (maximum at p=0.5)
            if p == 0 or p == 1:
                entropy = 0
            else:
                entropy = -p * np.log2(p) - q * np.log2(q)
            
            # Normalize to 0-1 scale (max entropy = 1)
            uncertainty = entropy
            
            # Adjust based on context complexity
            context_complexity = self._assess_context_complexity(context)
            uncertainty = uncertainty * (1 + context_complexity * 0.3)
            
            return min(1.0, uncertainty)
            
        except Exception as e:
            self.logger.error(f"Entropy-based uncertainty failed: {e}")
            return 0.5
    
    def _pattern_based_uncertainty(self, confidence: float, context: Dict[str, Any]) -> float:
        """Uncertainty based on outcome patterns for similar confidence levels"""
        try:
            # Find decisions with similar confidence levels
            confidence_range = 0.1  # ±10% confidence range
            similar_confidences = []
            
            for decision in self.decision_history:
                if abs(decision["confidence"] - confidence) <= confidence_range:
                    similar_confidences.append(decision)
            
            if len(similar_confidences) < 3:
                return 0.6  # Moderate uncertainty with limited pattern data
            
            # Calculate outcome variance for this confidence level
            outcomes = [1.0 if d["outcome"] else 0.0 for d in similar_confidences]
            outcome_variance = np.var(outcomes)
            
            # Higher variance = higher uncertainty
            uncertainty = min(1.0, outcome_variance * 2)
            return uncertainty
            
        except Exception as e:
            self.logger.error(f"Pattern-based uncertainty failed: {e}")
            return 0.5
    
    def _find_similar_decisions(self, context: Dict[str, Any], similarity_threshold: float = 0.7) -> List[Dict]:
        """Find historically similar decision contexts"""
        try:
            similar_decisions = []
            
            for decision in self.decision_history:
                similarity = self._calculate_context_similarity(
                    context, decision.get("context", {})
                )
                
                if similarity >= similarity_threshold:
                    similar_decisions.append(decision)
            
            return similar_decisions
            
        except Exception as e:
            self.logger.error(f"Failed to find similar decisions: {e}")
            return []
    
    def _calculate_context_similarity(self, context1: Dict, context2: Dict) -> float:
        """Calculate similarity between two decision contexts"""
        try:
            if not context1 or not context2:
                return 0.0
            
            # Key context features to compare
            features = ["market_ticker", "event_type", "quarter", "time_remaining"]
            
            matches = 0
            total_features = 0
            
            for feature in features:
                if feature in context1 and feature in context2:
                    total_features += 1
                    if context1[feature] == context2[feature]:
                        matches += 1
            
            if total_features == 0:
                return 0.0
            
            return matches / total_features
            
        except Exception as e:
            self.logger.error(f"Context similarity calculation failed: {e}")
            return 0.0
    
    def _assess_context_complexity(self, context: Dict[str, Any]) -> float:
        """Assess complexity of decision context (0 = simple, 1 = complex)"""
        try:
            complexity_score = 0.0
            
            # Number of context features
            feature_complexity = min(1.0, len(context) / 20)  # More features = more complex
            complexity_score += feature_complexity * 0.3
            
            # Time pressure (less time = more complex)
            time_remaining = context.get("time_remaining", 3600)  # Default 1 hour
            time_complexity = max(0.0, 1 - time_remaining / 3600)  # Normalize to hour
            complexity_score += time_complexity * 0.3
            
            # Market volatility proxy
            market_events = context.get("recent_events", [])
            volatility_complexity = min(1.0, len(market_events) / 10)
            complexity_score += volatility_complexity * 0.4
            
            return min(1.0, complexity_score)
            
        except Exception as e:
            self.logger.error(f"Context complexity assessment failed: {e}")
            return 0.5
    
    def _default_uncertainty(self, confidence: float, context: Dict[str, Any]) -> float:
        """Default uncertainty estimation"""
        # Simple heuristic: uncertainty is highest at confidence = 0.5
        base_uncertainty = 2 * min(confidence, 1 - confidence)
        
        # Add some noise based on context
        context_factor = min(0.2, len(context) * 0.01)
        return min(1.0, base_uncertainty + context_factor)
    
    def update_decision_history(self, decision_data: Dict[str, Any]) -> None:
        """Update decision history with new outcome"""
        try:
            self.decision_history.append(decision_data)
            
            # Update outcome patterns
            confidence_bucket = int(decision_data["confidence"] * 10) / 10  # Bucket to 0.1 precision
            outcome_value = 1.0 if decision_data.get("outcome", False) else 0.0
            self.outcome_patterns[str(confidence_bucket)].append(outcome_value)
            
        except Exception as e:
            self.logger.error(f"Failed to update decision history: {e}")


class ConfidenceCalibrator:
    """
    Main confidence calibration engine that trains calibration models
    and provides calibrated confidence scores for agent decisions.
    """
    
    def __init__(self, uncertainty_quantifier: Optional[UncertaintyQuantifier] = None):
        self.logger = logging.getLogger(__name__)
        
        # Uncertainty quantification
        self.uncertainty_quantifier = uncertainty_quantifier or UncertaintyQuantifier()
        
        # Calibration models for different methods
        self.calibration_models: Dict[CalibrationMethod, Any] = {}
        
        # Training data storage
        self.training_data: Dict[str, CalibrationData] = {}  # By agent_id
        
        # Calibration performance tracking
        self.calibration_metrics: Dict[str, CalibrationMetrics] = {}
        
        # Default calibration parameters
        self.default_method = CalibrationMethod.ISOTONIC_REGRESSION
        self.min_training_samples = 50
        self.calibration_update_frequency = timedelta(hours=6)
        self.last_calibration_update: Dict[str, datetime] = {}
        
    async def train_calibration(self, agent_id: str, decisions: List[DecisionMetrics]) -> CalibrationMetrics:
        """
        Train calibration model for a specific agent.
        
        Args:
            agent_id: Agent identifier
            decisions: List of historical decisions with outcomes
            
        Returns:
            Calibration metrics for the trained model
        """
        try:
            # Prepare training data
            training_data = self._prepare_training_data(decisions)
            
            if not training_data.is_valid:
                raise ValueError(f"Invalid training data for agent {agent_id}")
            
            if len(training_data) < self.min_training_samples:
                self.logger.warning(f"Insufficient training data for agent {agent_id}: {len(training_data)} samples")
                return self._create_default_metrics()
            
            # Store training data
            self.training_data[agent_id] = training_data
            
            # Train multiple calibration models
            models = {}
            for method in [CalibrationMethod.PLATT_SCALING, 
                          CalibrationMethod.ISOTONIC_REGRESSION,
                          CalibrationMethod.TEMPERATURE_SCALING]:
                try:
                    model = await self._train_single_method(method, training_data)
                    models[method] = model
                except Exception as e:
                    self.logger.warning(f"Failed to train {method.value} for agent {agent_id}: {e}")
            
            # Select best model
            best_method, best_model = self._select_best_model(models, training_data)
            self.calibration_models[agent_id] = (best_method, best_model)
            
            # Calculate and store metrics
            metrics = self._calculate_calibration_metrics(best_model, best_method, training_data)
            self.calibration_metrics[agent_id] = metrics
            
            # Update timestamp
            self.last_calibration_update[agent_id] = datetime.now()
            
            self.logger.info(f"Trained calibration for agent {agent_id} using {best_method.value}")
            self.logger.info(f"Calibration quality: {metrics.calibration_quality} (reliability: {metrics.reliability:.3f})")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to train calibration for agent {agent_id}: {e}")
            return self._create_default_metrics()
    
    def _prepare_training_data(self, decisions: List[DecisionMetrics]) -> CalibrationData:
        """Prepare training data from decision metrics"""
        try:
            confidences = []
            outcomes = []
            weights = []
            
            for decision in decisions:
                if decision.outcome is not None and 0 <= decision.confidence <= 1:
                    confidences.append(decision.confidence)
                    outcomes.append(decision.outcome > 0)  # Convert to boolean
                    
                    # Weight more recent decisions higher
                    age_days = (datetime.now() - decision.timestamp).days
                    weight = max(0.1, 1.0 - age_days * 0.1)  # Decay weight over time
                    weights.append(weight)
            
            return CalibrationData(
                confidences=confidences,
                outcomes=outcomes,
                weights=weights,
                metadata={"sample_count": len(confidences)}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to prepare training data: {e}")
            return CalibrationData([], [])
    
    async def _train_single_method(self, method: CalibrationMethod, data: CalibrationData) -> Any:
        """Train a single calibration method"""
        try:
            X = np.array(data.confidences).reshape(-1, 1)
            y = np.array(data.outcomes)
            sample_weights = np.array(data.weights) if data.weights else None
            
            if method == CalibrationMethod.PLATT_SCALING:
                model = LogisticRegression()
                model.fit(X, y, sample_weight=sample_weights)
                return model
                
            elif method == CalibrationMethod.ISOTONIC_REGRESSION:
                model = IsotonicRegression(out_of_bounds='clip')
                model.fit(data.confidences, data.outcomes, sample_weight=sample_weights)
                return model
                
            elif method == CalibrationMethod.TEMPERATURE_SCALING:
                # Temperature scaling using logistic regression
                model = self._train_temperature_scaling(X, y, sample_weights)
                return model
                
            elif method == CalibrationMethod.HISTOGRAM_BINNING:
                model = self._train_histogram_binning(data.confidences, data.outcomes, sample_weights)
                return model
                
            else:
                raise ValueError(f"Unsupported calibration method: {method}")
                
        except Exception as e:
            self.logger.error(f"Failed to train {method.value}: {e}")
            raise
    
    def _train_temperature_scaling(self, X: np.ndarray, y: np.ndarray, sample_weights: Optional[np.ndarray]) -> Dict:
        """Train temperature scaling model"""
        try:
            # Use logistic regression to find optimal temperature
            model = LogisticRegression()
            model.fit(X, y, sample_weight=sample_weights)
            
            # Extract temperature parameter (inverse of coefficient)
            temperature = 1.0 / abs(model.coef_[0][0]) if model.coef_[0][0] != 0 else 1.0
            
            return {
                "type": "temperature_scaling",
                "temperature": temperature,
                "intercept": model.intercept_[0]
            }
            
        except Exception as e:
            self.logger.error(f"Temperature scaling training failed: {e}")
            return {"type": "temperature_scaling", "temperature": 1.0, "intercept": 0.0}
    
    def _train_histogram_binning(self, confidences: List[float], outcomes: List[bool], weights: Optional[List[float]]) -> Dict:
        """Train histogram binning calibration"""
        try:
            n_bins = min(10, len(confidences) // 10)  # Adaptive number of bins
            if n_bins < 2:
                n_bins = 2
            
            # Create bins
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
            
            # Calculate calibrated probabilities for each bin
            calibrated_probs = []
            
            for i in range(n_bins):
                # Find samples in this bin
                in_bin = [(confidences[j] >= bin_boundaries[i] and confidences[j] < bin_boundaries[i+1])
                         for j in range(len(confidences))]
                
                if i == n_bins - 1:  # Last bin includes upper boundary
                    in_bin = [(confidences[j] >= bin_boundaries[i] and confidences[j] <= bin_boundaries[i+1])
                             for j in range(len(confidences))]
                
                # Calculate weighted average outcome for this bin
                bin_outcomes = [outcomes[j] for j in range(len(outcomes)) if in_bin[j]]
                bin_weights = [weights[j] for j in range(len(weights)) if in_bin[j]] if weights else None
                
                if bin_outcomes:
                    if bin_weights:
                        weighted_sum = sum(o * w for o, w in zip(bin_outcomes, bin_weights))
                        weight_sum = sum(bin_weights)
                        calibrated_prob = weighted_sum / weight_sum if weight_sum > 0 else 0.5
                    else:
                        calibrated_prob = sum(bin_outcomes) / len(bin_outcomes)
                else:
                    calibrated_prob = bin_centers[i]  # Use bin center as fallback
                
                calibrated_probs.append(calibrated_prob)
            
            return {
                "type": "histogram_binning",
                "bin_boundaries": bin_boundaries.tolist(),
                "calibrated_probs": calibrated_probs
            }
            
        except Exception as e:
            self.logger.error(f"Histogram binning training failed: {e}")
            return {"type": "histogram_binning", "bin_boundaries": [0, 1], "calibrated_probs": [0.5]}
    
    def _select_best_model(self, models: Dict[CalibrationMethod, Any], data: CalibrationData) -> Tuple[CalibrationMethod, Any]:
        """Select best calibration model based on validation performance"""
        try:
            if not models:
                return self.default_method, None
            
            best_method = None
            best_model = None
            best_score = float('inf')
            
            # Use Brier score for model selection (lower is better)
            for method, model in models.items():
                try:
                    score = self._calculate_brier_score(model, method, data)
                    if score < best_score:
                        best_score = score
                        best_method = method
                        best_model = model
                except Exception as e:
                    self.logger.warning(f"Failed to evaluate {method.value}: {e}")
                    continue
            
            if best_method is None:
                # Fallback to first available model
                best_method = list(models.keys())[0]
                best_model = models[best_method]
            
            return best_method, best_model
            
        except Exception as e:
            self.logger.error(f"Model selection failed: {e}")
            return self.default_method, None
    
    def _calculate_brier_score(self, model: Any, method: CalibrationMethod, data: CalibrationData) -> float:
        """Calculate Brier score for model evaluation"""
        try:
            # Get calibrated predictions
            calibrated_probs = self._apply_calibration_model(data.confidences, model, method)
            
            # Calculate Brier score
            brier_score = np.mean([(prob - outcome)**2 for prob, outcome in zip(calibrated_probs, data.outcomes)])
            
            return brier_score
            
        except Exception as e:
            self.logger.error(f"Brier score calculation failed: {e}")
            return 1.0  # Worst possible score
    
    def _apply_calibration_model(self, confidences: List[float], model: Any, method: CalibrationMethod) -> List[float]:
        """Apply calibration model to get calibrated probabilities"""
        try:
            if method == CalibrationMethod.PLATT_SCALING:
                X = np.array(confidences).reshape(-1, 1)
                return model.predict_proba(X)[:, 1].tolist()
                
            elif method == CalibrationMethod.ISOTONIC_REGRESSION:
                return model.predict(confidences).tolist()
                
            elif method == CalibrationMethod.TEMPERATURE_SCALING:
                temperature = model["temperature"]
                intercept = model["intercept"]
                
                calibrated = []
                for conf in confidences:
                    # Apply temperature scaling
                    logit = np.log(conf / (1 - conf)) if 0 < conf < 1 else 0
                    scaled_logit = logit / temperature + intercept
                    calibrated_prob = 1 / (1 + np.exp(-scaled_logit))
                    calibrated.append(max(0.001, min(0.999, calibrated_prob)))
                
                return calibrated
                
            elif method == CalibrationMethod.HISTOGRAM_BINNING:
                bin_boundaries = model["bin_boundaries"]
                calibrated_probs = model["calibrated_probs"]
                
                calibrated = []
                for conf in confidences:
                    # Find appropriate bin
                    bin_idx = np.digitize(conf, bin_boundaries) - 1
                    bin_idx = max(0, min(len(calibrated_probs) - 1, bin_idx))
                    calibrated.append(calibrated_probs[bin_idx])
                
                return calibrated
                
            else:
                return confidences  # No calibration
                
        except Exception as e:
            self.logger.error(f"Failed to apply calibration model: {e}")
            return confidences
    
    def _calculate_calibration_metrics(self, model: Any, method: CalibrationMethod, data: CalibrationData) -> CalibrationMetrics:
        """Calculate comprehensive calibration metrics"""
        try:
            # Get calibrated predictions
            calibrated_probs = self._apply_calibration_model(data.confidences, model, method)
            
            # Brier score
            brier_score = np.mean([(prob - outcome)**2 for prob, outcome in zip(calibrated_probs, data.outcomes)])
            
            # Reliability (Expected Calibration Error)
            reliability = self._calculate_expected_calibration_error(calibrated_probs, data.outcomes)
            
            # Resolution
            resolution = self._calculate_resolution(calibrated_probs, data.outcomes)
            
            # Sharpness
            sharpness = np.var(calibrated_probs)
            
            # Calibration curve statistics
            slope, intercept = self._calculate_calibration_curve_stats(calibrated_probs, data.outcomes)
            
            return CalibrationMetrics(
                brier_score=brier_score,
                reliability=reliability,
                resolution=resolution,
                sharpness=sharpness,
                calibration_slope=slope,
                calibration_intercept=intercept,
                sample_count=len(data)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to calculate calibration metrics: {e}")
            return self._create_default_metrics()
    
    def _calculate_expected_calibration_error(self, predictions: List[float], outcomes: List[bool]) -> float:
        """Calculate Expected Calibration Error (ECE)"""
        try:
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            
            ece = 0.0
            total_samples = len(predictions)
            
            for i in range(n_bins):
                # Find samples in this bin
                in_bin_mask = [(predictions[j] >= bin_boundaries[i] and predictions[j] < bin_boundaries[i+1])
                              for j in range(len(predictions))]
                
                if i == n_bins - 1:  # Last bin includes upper boundary
                    in_bin_mask = [(predictions[j] >= bin_boundaries[i] and predictions[j] <= bin_boundaries[i+1])
                                  for j in range(len(predictions))]
                
                bin_predictions = [predictions[j] for j in range(len(predictions)) if in_bin_mask[j]]
                bin_outcomes = [outcomes[j] for j in range(len(outcomes)) if in_bin_mask[j]]
                
                if bin_predictions:
                    bin_size = len(bin_predictions)
                    bin_confidence = np.mean(bin_predictions)
                    bin_accuracy = np.mean(bin_outcomes)
                    
                    ece += (bin_size / total_samples) * abs(bin_confidence - bin_accuracy)
            
            return ece
            
        except Exception as e:
            self.logger.error(f"ECE calculation failed: {e}")
            return 1.0
    
    def _calculate_resolution(self, predictions: List[float], outcomes: List[bool]) -> float:
        """Calculate resolution (ability to discriminate between classes)"""
        try:
            # Resolution is the variance of conditional probabilities weighted by frequency
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            
            resolution = 0.0
            total_samples = len(predictions)
            overall_base_rate = np.mean(outcomes)
            
            for i in range(n_bins):
                # Find samples in this bin
                in_bin_mask = [(predictions[j] >= bin_boundaries[i] and predictions[j] < bin_boundaries[i+1])
                              for j in range(len(predictions))]
                
                if i == n_bins - 1:  # Last bin includes upper boundary
                    in_bin_mask = [(predictions[j] >= bin_boundaries[i] and predictions[j] <= bin_boundaries[i+1])
                                  for j in range(len(predictions))]
                
                bin_outcomes = [outcomes[j] for j in range(len(outcomes)) if in_bin_mask[j]]
                
                if bin_outcomes:
                    bin_size = len(bin_outcomes)
                    bin_accuracy = np.mean(bin_outcomes)
                    
                    resolution += (bin_size / total_samples) * (bin_accuracy - overall_base_rate)**2
            
            return resolution
            
        except Exception as e:
            self.logger.error(f"Resolution calculation failed: {e}")
            return 0.0
    
    def _calculate_calibration_curve_stats(self, predictions: List[float], outcomes: List[bool]) -> Tuple[float, float]:
        """Calculate slope and intercept of calibration curve"""
        try:
            # Linear regression of outcomes vs predictions
            slope, intercept, _, _, _ = stats.linregress(predictions, outcomes)
            return slope, intercept
            
        except Exception as e:
            self.logger.error(f"Calibration curve stats calculation failed: {e}")
            return 1.0, 0.0  # Perfect calibration
    
    def _create_default_metrics(self) -> CalibrationMetrics:
        """Create default calibration metrics"""
        return CalibrationMetrics(
            brier_score=0.25,  # Random prediction
            reliability=0.5,   # Poor reliability
            resolution=0.0,    # No resolution
            sharpness=0.25,    # Moderate sharpness
            calibration_slope=1.0,
            calibration_intercept=0.0,
            sample_count=0
        )
    
    async def calibrate_confidence(
        self,
        agent_id: str,
        raw_confidence: float,
        context: Dict[str, Any]
    ) -> ConfidenceScore:
        """
        Calibrate confidence score for an agent's decision.
        
        Args:
            agent_id: Agent identifier
            raw_confidence: Original confidence score (0-1)
            context: Decision context for uncertainty estimation
            
        Returns:
            Calibrated confidence score with uncertainty bounds
        """
        try:
            # Check if we have calibration model for this agent
            if agent_id not in self.calibration_models:
                # Try to train calibration if we have enough data
                if agent_id in self.training_data and len(self.training_data[agent_id]) >= self.min_training_samples:
                    # Use existing training data to create temporary model
                    decisions = []  # Would need to reconstruct from training data
                    await self.train_calibration(agent_id, decisions)
                else:
                    # Return uncalibrated confidence with high uncertainty
                    return self._create_default_confidence_score(raw_confidence, context)
            
            method, model = self.calibration_models[agent_id]
            
            # Apply calibration
            calibrated_conf = self._apply_calibration_model([raw_confidence], model, method)[0]
            
            # Estimate uncertainty
            uncertainty = self.uncertainty_quantifier.estimate_uncertainty(
                raw_confidence, context, method="ensemble"
            )
            
            # Calculate confidence interval
            conf_interval = self._calculate_confidence_interval(
                calibrated_conf, uncertainty, context
            )
            
            # Get sample size used for calibration
            sample_size = self.training_data.get(agent_id, CalibrationData([], [])).sample_count if agent_id in self.training_data else 0
            
            return ConfidenceScore(
                raw_confidence=raw_confidence,
                calibrated_confidence=calibrated_conf,
                uncertainty=uncertainty,
                confidence_interval=conf_interval,
                method_used=method,
                sample_size=sample_size
            )
            
        except Exception as e:
            self.logger.error(f"Failed to calibrate confidence for agent {agent_id}: {e}")
            return self._create_default_confidence_score(raw_confidence, context)
    
    def _create_default_confidence_score(self, raw_confidence: float, context: Dict[str, Any]) -> ConfidenceScore:
        """Create default confidence score when calibration is not available"""
        # Use simple uncertainty estimation
        uncertainty = max(0.3, 2 * min(raw_confidence, 1 - raw_confidence))  # Higher at extremes
        
        # Wide confidence interval due to lack of calibration
        margin = uncertainty * 0.5
        conf_interval = (
            max(0.0, raw_confidence - margin),
            min(1.0, raw_confidence + margin)
        )
        
        return ConfidenceScore(
            raw_confidence=raw_confidence,
            calibrated_confidence=raw_confidence,  # No calibration applied
            uncertainty=uncertainty,
            confidence_interval=conf_interval,
            method_used=CalibrationMethod.HISTOGRAM_BINNING,  # Default method
            sample_size=0
        )
    
    def _calculate_confidence_interval(
        self,
        calibrated_confidence: float,
        uncertainty: float,
        context: Dict[str, Any],
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval for calibrated score"""
        try:
            # Use uncertainty to determine interval width
            z_score = stats.norm.ppf((1 + confidence_level) / 2)  # 95% confidence
            
            # Scale uncertainty by z-score
            margin = uncertainty * z_score * 0.5
            
            # Adjust margin based on context
            context_adjustment = self._get_context_adjustment(context)
            margin = margin * (1 + context_adjustment)
            
            # Calculate bounds
            lower_bound = max(0.0, calibrated_confidence - margin)
            upper_bound = min(1.0, calibrated_confidence + margin)
            
            return (lower_bound, upper_bound)
            
        except Exception as e:
            self.logger.error(f"Confidence interval calculation failed: {e}")
            return (max(0.0, calibrated_confidence - 0.2), min(1.0, calibrated_confidence + 0.2))
    
    def _get_context_adjustment(self, context: Dict[str, Any]) -> float:
        """Get adjustment factor based on decision context"""
        try:
            adjustment = 0.0
            
            # Time pressure increases uncertainty
            time_remaining = context.get("time_remaining", 3600)
            if time_remaining < 300:  # Less than 5 minutes
                adjustment += 0.2
            elif time_remaining < 900:  # Less than 15 minutes
                adjustment += 0.1
            
            # Market volatility increases uncertainty
            recent_events = context.get("recent_events", [])
            if len(recent_events) > 5:
                adjustment += 0.15
            
            # Edge case scenarios increase uncertainty
            if context.get("is_edge_case", False):
                adjustment += 0.25
            
            return min(0.5, adjustment)  # Cap adjustment
            
        except Exception as e:
            self.logger.error(f"Context adjustment calculation failed: {e}")
            return 0.1  # Small default adjustment
    
    async def should_update_calibration(self, agent_id: str) -> bool:
        """Check if calibration model should be updated"""
        try:
            if agent_id not in self.last_calibration_update:
                return True  # Never been updated
            
            last_update = self.last_calibration_update[agent_id]
            time_since_update = datetime.now() - last_update
            
            # Time-based update
            if time_since_update >= self.calibration_update_frequency:
                return True
            
            # Performance-based update
            if agent_id in self.calibration_metrics:
                metrics = self.calibration_metrics[agent_id]
                if metrics.reliability > 0.2:  # Poor calibration
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to check update status for agent {agent_id}: {e}")
            return False
    
    def get_calibration_status(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get calibration status for agent(s)"""
        try:
            if agent_id:
                # Single agent status
                if agent_id not in self.calibration_models:
                    return {"agent_id": agent_id, "status": "not_calibrated", "reason": "no_model"}
                
                method, _ = self.calibration_models[agent_id]
                metrics = self.calibration_metrics.get(agent_id)
                last_update = self.last_calibration_update.get(agent_id)
                
                return {
                    "agent_id": agent_id,
                    "status": "calibrated",
                    "method": method.value,
                    "metrics": {
                        "calibration_quality": metrics.calibration_quality if metrics else "unknown",
                        "reliability": metrics.reliability if metrics else 0.5,
                        "brier_score": metrics.brier_score if metrics else 0.25,
                        "sample_count": metrics.sample_count if metrics else 0
                    },
                    "last_update": last_update.isoformat() if last_update else None,
                    "needs_update": False  # Will be checked separately if needed
                }
            else:
                # All agents status
                all_status = {}
                for agent_id in set(list(self.calibration_models.keys()) + list(self.training_data.keys())):
                    all_status[agent_id] = self.get_calibration_status(agent_id)
                
                return {
                    "timestamp": datetime.now().isoformat(),
                    "total_agents": len(all_status),
                    "calibrated_agents": len([s for s in all_status.values() if s.get("status") == "calibrated"]),
                    "agents": all_status
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get calibration status: {e}")
            return {"error": str(e)}