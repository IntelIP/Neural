"""
Bootstrap Confidence Estimator

Uses bootstrap sampling to estimate confidence intervals and
uncertainty for agent decision-making scenarios.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable
import numpy as np
import asyncio
import logging
from datetime import datetime
import random


@dataclass
class BootstrapConfig:
    """Configuration for bootstrap estimation"""
    n_bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    sample_size_fraction: float = 1.0  # Fraction of original data to sample
    random_seed: Optional[int] = None
    parallel_execution: bool = True
    max_workers: int = 4
    
    # Advanced options
    bias_correction: bool = True
    acceleration_correction: bool = True  # BCa intervals
    minimum_sample_size: int = 30


@dataclass 
class ConfidenceInterval:
    """Bootstrap confidence interval result"""
    lower_bound: float
    upper_bound: float
    point_estimate: float
    confidence_level: float
    method: str  # "percentile", "bias_corrected", "bca"
    n_bootstrap_samples: int
    
    @property
    def width(self) -> float:
        """Width of confidence interval"""
        return self.upper_bound - self.lower_bound
    
    @property
    def margin_of_error(self) -> float:
        """Margin of error (half-width)"""
        return self.width / 2


class BootstrapConfidenceEstimator:
    """
    Bootstrap-based confidence estimation for agent decision metrics.
    
    Uses resampling techniques to estimate confidence intervals and
    uncertainty quantification for various performance metrics.
    """
    
    def __init__(self, config: BootstrapConfig = None):
        self.config = config or BootstrapConfig()
        self.logger = logging.getLogger(__name__)
        
        # Set random seed for reproducibility
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
            random.seed(self.config.random_seed)
        
        # Cache for bootstrap results
        self.bootstrap_cache: Dict[str, Any] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        
        # Performance tracking
        self.estimation_history: List[Dict[str, Any]] = []
    
    async def estimate_win_rate_confidence(
        self,
        outcomes: List[bool],
        agent_id: Optional[str] = None
    ) -> ConfidenceInterval:
        """
        Estimate confidence interval for win rate using bootstrap.
        
        Args:
            outcomes: List of decision outcomes (True for win, False for loss)
            agent_id: Optional agent identifier for caching
            
        Returns:
            Confidence interval for win rate
        """
        try:
            if len(outcomes) < self.config.minimum_sample_size:
                self.logger.warning(f"Sample size too small for bootstrap: {len(outcomes)}")
                return self._create_wide_interval(np.mean(outcomes), "insufficient_data")
            
            # Check cache
            cache_key = f"win_rate_{agent_id}_{hash(str(outcomes))}" if agent_id else f"win_rate_{hash(str(outcomes))}"
            if cache_key in self.bootstrap_cache:
                cached_result = self.bootstrap_cache[cache_key]
                if datetime.now() - self.cache_timestamps[cache_key] < timedelta(minutes=30):
                    return cached_result
            
            # Bootstrap function for win rate
            def win_rate_statistic(sample):
                return np.mean(sample)
            
            # Perform bootstrap
            bootstrap_estimates = await self._bootstrap_statistic(
                data=outcomes,
                statistic_func=win_rate_statistic,
                description="win_rate"
            )
            
            # Calculate confidence interval
            point_estimate = np.mean(outcomes)
            interval = self._calculate_confidence_interval(
                bootstrap_estimates, 
                point_estimate,
                outcomes  # Original data for BCa
            )
            
            # Cache result
            self.bootstrap_cache[cache_key] = interval
            self.cache_timestamps[cache_key] = datetime.now()
            
            return interval
            
        except Exception as e:
            self.logger.error(f"Win rate confidence estimation failed: {e}")
            return self._create_wide_interval(np.mean(outcomes) if outcomes else 0.5, "error")
    
    async def estimate_kelly_adherence_confidence(
        self,
        kelly_deviations: List[float],
        agent_id: Optional[str] = None
    ) -> ConfidenceInterval:
        """
        Estimate confidence interval for Kelly Criterion adherence.
        
        Args:
            kelly_deviations: List of Kelly fraction deviations
            agent_id: Optional agent identifier for caching
            
        Returns:
            Confidence interval for Kelly adherence score
        """
        try:
            if len(kelly_deviations) < self.config.minimum_sample_size:
                return self._create_wide_interval(0.5, "insufficient_data")
            
            # Kelly adherence score = 1 - average deviation
            def kelly_adherence_statistic(sample):
                avg_deviation = np.mean(np.abs(sample))
                return max(0.0, 1.0 - avg_deviation)
            
            # Perform bootstrap
            bootstrap_estimates = await self._bootstrap_statistic(
                data=kelly_deviations,
                statistic_func=kelly_adherence_statistic,
                description="kelly_adherence"
            )
            
            # Calculate point estimate
            point_estimate = kelly_adherence_statistic(kelly_deviations)
            
            # Calculate confidence interval
            interval = self._calculate_confidence_interval(
                bootstrap_estimates,
                point_estimate,
                kelly_deviations
            )
            
            return interval
            
        except Exception as e:
            self.logger.error(f"Kelly adherence confidence estimation failed: {e}")
            return self._create_wide_interval(0.5, "error")
    
    async def estimate_sharpe_ratio_confidence(
        self,
        returns: List[float],
        agent_id: Optional[str] = None
    ) -> ConfidenceInterval:
        """
        Estimate confidence interval for Sharpe ratio.
        
        Args:
            returns: List of return values
            agent_id: Optional agent identifier for caching
            
        Returns:
            Confidence interval for Sharpe ratio
        """
        try:
            if len(returns) < self.config.minimum_sample_size:
                return self._create_wide_interval(0.0, "insufficient_data")
            
            def sharpe_ratio_statistic(sample):
                if len(sample) < 2:
                    return 0.0
                mean_return = np.mean(sample)
                std_return = np.std(sample, ddof=1)  # Sample standard deviation
                return mean_return / std_return if std_return > 0 else 0.0
            
            # Perform bootstrap
            bootstrap_estimates = await self._bootstrap_statistic(
                data=returns,
                statistic_func=sharpe_ratio_statistic,
                description="sharpe_ratio"
            )
            
            # Calculate point estimate
            point_estimate = sharpe_ratio_statistic(returns)
            
            # Calculate confidence interval
            interval = self._calculate_confidence_interval(
                bootstrap_estimates,
                point_estimate,
                returns
            )
            
            return interval
            
        except Exception as e:
            self.logger.error(f"Sharpe ratio confidence estimation failed: {e}")
            return self._create_wide_interval(0.0, "error")
    
    async def estimate_confidence_calibration_interval(
        self,
        confidences: List[float],
        outcomes: List[bool],
        agent_id: Optional[str] = None
    ) -> ConfidenceInterval:
        """
        Estimate confidence interval for confidence calibration score.
        
        Args:
            confidences: List of confidence scores
            outcomes: List of corresponding outcomes
            agent_id: Optional agent identifier for caching
            
        Returns:
            Confidence interval for calibration score
        """
        try:
            if len(confidences) != len(outcomes) or len(confidences) < self.config.minimum_sample_size:
                return self._create_wide_interval(0.5, "insufficient_data")
            
            def calibration_statistic(indices):
                # Resample both confidences and outcomes using same indices
                sample_confidences = [confidences[i] for i in indices]
                sample_outcomes = [outcomes[i] for i in indices]
                
                # Calculate Expected Calibration Error
                return self._calculate_expected_calibration_error(sample_confidences, sample_outcomes)
            
            # Perform bootstrap with paired resampling
            bootstrap_estimates = await self._bootstrap_paired_statistic(
                data1=confidences,
                data2=outcomes,
                statistic_func=calibration_statistic,
                description="confidence_calibration"
            )
            
            # Calculate point estimate
            point_estimate = self._calculate_expected_calibration_error(confidences, outcomes)
            
            # For ECE, lower is better, so we want 1 - ECE as the score
            calibration_scores = [max(0.0, 1.0 - ece) for ece in bootstrap_estimates]
            point_calibration_score = max(0.0, 1.0 - point_estimate)
            
            # Calculate confidence interval
            interval = self._calculate_confidence_interval(
                calibration_scores,
                point_calibration_score,
                calibration_scores  # Use bootstrap estimates as "original data"
            )
            
            return interval
            
        except Exception as e:
            self.logger.error(f"Confidence calibration interval estimation failed: {e}")
            return self._create_wide_interval(0.5, "error")
    
    def _calculate_expected_calibration_error(self, confidences: List[float], outcomes: List[bool]) -> float:
        """Calculate Expected Calibration Error for bootstrap sampling"""
        try:
            if not confidences or len(confidences) != len(outcomes):
                return 1.0  # Maximum error
            
            n_bins = min(10, len(confidences) // 5)  # Adaptive binning
            if n_bins < 2:
                n_bins = 2
            
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            
            ece = 0.0
            total_samples = len(confidences)
            
            for i in range(n_bins):
                # Find samples in this bin
                in_bin = []
                for j, conf in enumerate(confidences):
                    if i == n_bins - 1:  # Last bin includes upper boundary
                        if bin_boundaries[i] <= conf <= bin_boundaries[i + 1]:
                            in_bin.append(j)
                    else:
                        if bin_boundaries[i] <= conf < bin_boundaries[i + 1]:
                            in_bin.append(j)
                
                if in_bin:
                    bin_confidences = [confidences[j] for j in in_bin]
                    bin_outcomes = [outcomes[j] for j in in_bin]
                    
                    bin_size = len(bin_confidences)
                    bin_conf_avg = np.mean(bin_confidences)
                    bin_accuracy = np.mean(bin_outcomes)
                    
                    ece += (bin_size / total_samples) * abs(bin_conf_avg - bin_accuracy)
            
            return ece
            
        except Exception as e:
            self.logger.error(f"ECE calculation failed: {e}")
            return 1.0
    
    async def _bootstrap_statistic(
        self,
        data: List[Any],
        statistic_func: Callable,
        description: str
    ) -> List[float]:
        """
        Perform bootstrap resampling for a single dataset.
        
        Args:
            data: Original dataset
            statistic_func: Function to compute statistic on sample
            description: Description for logging
            
        Returns:
            List of bootstrap estimates
        """
        try:
            n_original = len(data)
            sample_size = max(1, int(n_original * self.config.sample_size_fraction))
            
            if self.config.parallel_execution and self.config.n_bootstrap_samples > 100:
                return await self._bootstrap_parallel(data, statistic_func, sample_size, description)
            else:
                return await self._bootstrap_sequential(data, statistic_func, sample_size, description)
                
        except Exception as e:
            self.logger.error(f"Bootstrap sampling failed for {description}: {e}")
            return [statistic_func(data)]  # Return original estimate
    
    async def _bootstrap_paired_statistic(
        self,
        data1: List[Any],
        data2: List[Any],
        statistic_func: Callable,
        description: str
    ) -> List[float]:
        """
        Perform bootstrap resampling for paired datasets.
        
        Args:
            data1: First dataset
            data2: Second dataset (must be same length as data1)
            statistic_func: Function that takes indices and computes statistic
            description: Description for logging
            
        Returns:
            List of bootstrap estimates
        """
        try:
            if len(data1) != len(data2):
                raise ValueError("Paired datasets must have same length")
            
            n_original = len(data1)
            sample_size = max(1, int(n_original * self.config.sample_size_fraction))
            
            bootstrap_estimates = []
            
            for i in range(self.config.n_bootstrap_samples):
                # Sample indices with replacement
                sample_indices = np.random.choice(n_original, size=sample_size, replace=True)
                
                # Compute statistic
                estimate = statistic_func(sample_indices)
                bootstrap_estimates.append(estimate)
                
                # Progress logging
                if (i + 1) % 250 == 0:
                    self.logger.debug(f"Bootstrap {description}: {i + 1}/{self.config.n_bootstrap_samples}")
            
            return bootstrap_estimates
            
        except Exception as e:
            self.logger.error(f"Paired bootstrap sampling failed for {description}: {e}")
            return [0.5]  # Default estimate
    
    async def _bootstrap_sequential(
        self,
        data: List[Any],
        statistic_func: Callable,
        sample_size: int,
        description: str
    ) -> List[float]:
        """Sequential bootstrap sampling"""
        try:
            bootstrap_estimates = []
            
            for i in range(self.config.n_bootstrap_samples):
                # Sample with replacement
                sample = np.random.choice(data, size=sample_size, replace=True)
                
                # Compute statistic
                estimate = statistic_func(sample)
                bootstrap_estimates.append(estimate)
                
                # Yield control occasionally for async
                if i % 100 == 0:
                    await asyncio.sleep(0)
                    
                # Progress logging
                if (i + 1) % 250 == 0:
                    self.logger.debug(f"Bootstrap {description}: {i + 1}/{self.config.n_bootstrap_samples}")
            
            return bootstrap_estimates
            
        except Exception as e:
            self.logger.error(f"Sequential bootstrap failed: {e}")
            return [statistic_func(data)]
    
    async def _bootstrap_parallel(
        self,
        data: List[Any],
        statistic_func: Callable,
        sample_size: int,
        description: str
    ) -> List[float]:
        """Parallel bootstrap sampling using asyncio"""
        try:
            # Split work into batches
            batch_size = max(1, self.config.n_bootstrap_samples // self.config.max_workers)
            
            # Create tasks
            tasks = []
            remaining_samples = self.config.n_bootstrap_samples
            
            for worker in range(self.config.max_workers):
                if remaining_samples <= 0:
                    break
                
                worker_samples = min(batch_size, remaining_samples)
                remaining_samples -= worker_samples
                
                task = asyncio.create_task(
                    self._bootstrap_worker(data, statistic_func, sample_size, worker_samples, f"{description}_worker_{worker}")
                )
                tasks.append(task)
            
            # Wait for all workers to complete
            worker_results = await asyncio.gather(*tasks)
            
            # Combine results
            bootstrap_estimates = []
            for result in worker_results:
                bootstrap_estimates.extend(result)
            
            return bootstrap_estimates
            
        except Exception as e:
            self.logger.error(f"Parallel bootstrap failed: {e}")
            # Fallback to sequential
            return await self._bootstrap_sequential(data, statistic_func, sample_size, description)
    
    async def _bootstrap_worker(
        self,
        data: List[Any],
        statistic_func: Callable,
        sample_size: int,
        n_samples: int,
        worker_id: str
    ) -> List[float]:
        """Bootstrap worker for parallel execution"""
        try:
            estimates = []
            
            for i in range(n_samples):
                # Sample with replacement
                sample = np.random.choice(data, size=sample_size, replace=True)
                
                # Compute statistic
                estimate = statistic_func(sample)
                estimates.append(estimate)
                
                # Yield control occasionally
                if i % 50 == 0:
                    await asyncio.sleep(0)
            
            return estimates
            
        except Exception as e:
            self.logger.error(f"Bootstrap worker {worker_id} failed: {e}")
            return []
    
    def _calculate_confidence_interval(
        self,
        bootstrap_estimates: List[float],
        point_estimate: float,
        original_data: List[Any]
    ) -> ConfidenceInterval:
        """
        Calculate confidence interval from bootstrap estimates.
        
        Uses bias-corrected and accelerated (BCa) method if enabled,
        otherwise falls back to percentile method.
        """
        try:
            if not bootstrap_estimates:
                return self._create_wide_interval(point_estimate, "no_bootstrap_data")
            
            bootstrap_estimates = np.array(bootstrap_estimates)
            
            # Remove any invalid estimates
            valid_estimates = bootstrap_estimates[np.isfinite(bootstrap_estimates)]
            if len(valid_estimates) == 0:
                return self._create_wide_interval(point_estimate, "invalid_estimates")
            
            alpha = 1 - self.config.confidence_level
            
            if self.config.bias_correction and self.config.acceleration_correction:
                # BCa intervals
                return self._calculate_bca_interval(valid_estimates, point_estimate, original_data, alpha)
            elif self.config.bias_correction:
                # Bias-corrected intervals
                return self._calculate_bc_interval(valid_estimates, point_estimate, alpha)
            else:
                # Simple percentile intervals
                return self._calculate_percentile_interval(valid_estimates, point_estimate, alpha)
                
        except Exception as e:
            self.logger.error(f"Confidence interval calculation failed: {e}")
            return self._create_wide_interval(point_estimate, "calculation_error")
    
    def _calculate_percentile_interval(
        self,
        bootstrap_estimates: np.ndarray,
        point_estimate: float,
        alpha: float
    ) -> ConfidenceInterval:
        """Calculate simple percentile confidence interval"""
        try:
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            lower_bound = np.percentile(bootstrap_estimates, lower_percentile)
            upper_bound = np.percentile(bootstrap_estimates, upper_percentile)
            
            return ConfidenceInterval(
                lower_bound=float(lower_bound),
                upper_bound=float(upper_bound),
                point_estimate=point_estimate,
                confidence_level=self.config.confidence_level,
                method="percentile",
                n_bootstrap_samples=len(bootstrap_estimates)
            )
            
        except Exception as e:
            self.logger.error(f"Percentile interval calculation failed: {e}")
            return self._create_wide_interval(point_estimate, "percentile_error")
    
    def _calculate_bc_interval(
        self,
        bootstrap_estimates: np.ndarray,
        point_estimate: float,
        alpha: float
    ) -> ConfidenceInterval:
        """Calculate bias-corrected confidence interval"""
        try:
            # Calculate bias correction
            n_less = np.sum(bootstrap_estimates < point_estimate)
            p_less = n_less / len(bootstrap_estimates)
            
            if p_less == 0:
                z0 = -np.inf
            elif p_less == 1:
                z0 = np.inf
            else:
                z0 = stats.norm.ppf(p_less)
            
            # Calculate corrected percentiles
            z_alpha_2 = stats.norm.ppf(alpha / 2)
            z_1_alpha_2 = stats.norm.ppf(1 - alpha / 2)
            
            # Apply bias correction
            p1 = stats.norm.cdf(2 * z0 + z_alpha_2)
            p2 = stats.norm.cdf(2 * z0 + z_1_alpha_2)
            
            # Ensure percentiles are within valid range
            p1 = max(0.001, min(0.999, p1))
            p2 = max(0.001, min(0.999, p2))
            
            lower_bound = np.percentile(bootstrap_estimates, p1 * 100)
            upper_bound = np.percentile(bootstrap_estimates, p2 * 100)
            
            return ConfidenceInterval(
                lower_bound=float(lower_bound),
                upper_bound=float(upper_bound),
                point_estimate=point_estimate,
                confidence_level=self.config.confidence_level,
                method="bias_corrected",
                n_bootstrap_samples=len(bootstrap_estimates)
            )
            
        except Exception as e:
            self.logger.error(f"Bias-corrected interval calculation failed: {e}")
            return self._calculate_percentile_interval(bootstrap_estimates, point_estimate, alpha)
    
    def _calculate_bca_interval(
        self,
        bootstrap_estimates: np.ndarray,
        point_estimate: float,
        original_data: List[Any],
        alpha: float
    ) -> ConfidenceInterval:
        """Calculate bias-corrected and accelerated (BCa) confidence interval"""
        try:
            # Calculate bias correction (same as BC method)
            n_less = np.sum(bootstrap_estimates < point_estimate)
            p_less = n_less / len(bootstrap_estimates)
            
            if p_less == 0:
                z0 = -np.inf
            elif p_less == 1:
                z0 = np.inf
            else:
                z0 = stats.norm.ppf(p_less)
            
            # Calculate acceleration using jackknife
            acceleration = self._calculate_acceleration(original_data, point_estimate)
            
            # Calculate corrected percentiles with acceleration
            z_alpha_2 = stats.norm.ppf(alpha / 2)
            z_1_alpha_2 = stats.norm.ppf(1 - alpha / 2)
            
            # Apply bias and acceleration corrections
            p1_num = z0 + z_alpha_2
            p1_denom = 1 - acceleration * (z0 + z_alpha_2)
            p1 = stats.norm.cdf(z0 + p1_num / p1_denom) if p1_denom != 0 else stats.norm.cdf(z0 + z_alpha_2)
            
            p2_num = z0 + z_1_alpha_2
            p2_denom = 1 - acceleration * (z0 + z_1_alpha_2)
            p2 = stats.norm.cdf(z0 + p2_num / p2_denom) if p2_denom != 0 else stats.norm.cdf(z0 + z_1_alpha_2)
            
            # Ensure percentiles are within valid range
            p1 = max(0.001, min(0.999, p1))
            p2 = max(0.001, min(0.999, p2))
            
            lower_bound = np.percentile(bootstrap_estimates, p1 * 100)
            upper_bound = np.percentile(bootstrap_estimates, p2 * 100)
            
            return ConfidenceInterval(
                lower_bound=float(lower_bound),
                upper_bound=float(upper_bound),
                point_estimate=point_estimate,
                confidence_level=self.config.confidence_level,
                method="bca",
                n_bootstrap_samples=len(bootstrap_estimates)
            )
            
        except Exception as e:
            self.logger.error(f"BCa interval calculation failed: {e}")
            return self._calculate_bc_interval(bootstrap_estimates, point_estimate, alpha)
    
    def _calculate_acceleration(self, original_data: List[Any], point_estimate: float) -> float:
        """Calculate acceleration parameter for BCa intervals using jackknife"""
        try:
            n = len(original_data)
            
            if n < 10:  # Not enough data for reliable acceleration
                return 0.0
            
            # Jackknife estimates (leave-one-out)
            jackknife_estimates = []
            
            for i in range(n):
                # Create jackknife sample (all data except index i)
                jackknife_sample = [original_data[j] for j in range(n) if j != i]
                
                # For this implementation, we'll use a simple approximation
                # In practice, you would apply the same statistic function used for bootstrap
                if isinstance(original_data[0], bool):
                    # For boolean outcomes (like win rate)
                    jackknife_estimate = np.mean(jackknife_sample)
                else:
                    # For continuous outcomes
                    jackknife_estimate = np.mean(jackknife_sample)
                
                jackknife_estimates.append(jackknife_estimate)
            
            # Calculate acceleration
            jackknife_mean = np.mean(jackknife_estimates)
            numerator = np.sum((jackknife_mean - np.array(jackknife_estimates))**3)
            denominator = 6 * (np.sum((jackknife_mean - np.array(jackknife_estimates))**2))**1.5
            
            acceleration = numerator / denominator if denominator != 0 else 0.0
            
            # Limit acceleration to reasonable range
            acceleration = max(-0.25, min(0.25, acceleration))
            
            return acceleration
            
        except Exception as e:
            self.logger.error(f"Acceleration calculation failed: {e}")
            return 0.0  # No acceleration
    
    def _create_wide_interval(self, point_estimate: float, reason: str) -> ConfidenceInterval:
        """Create a wide confidence interval when normal calculation fails"""
        # Use 40% margin of error as fallback
        margin = 0.4
        lower_bound = max(0.0, point_estimate - margin)
        upper_bound = min(1.0, point_estimate + margin)
        
        return ConfidenceInterval(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            point_estimate=point_estimate,
            confidence_level=self.config.confidence_level,
            method=f"fallback_{reason}",
            n_bootstrap_samples=0
        )
    
    async def estimate_multiple_metrics(
        self,
        agent_data: Dict[str, List],
        agent_id: Optional[str] = None
    ) -> Dict[str, ConfidenceInterval]:
        """
        Estimate confidence intervals for multiple metrics simultaneously.
        
        Args:
            agent_data: Dictionary with metric names as keys and data lists as values
            agent_id: Optional agent identifier
            
        Returns:
            Dictionary of confidence intervals for each metric
        """
        try:
            results = {}
            
            # Create tasks for parallel estimation
            tasks = []
            
            if "outcomes" in agent_data:
                tasks.append(("win_rate", self.estimate_win_rate_confidence(agent_data["outcomes"], agent_id)))
            
            if "kelly_deviations" in agent_data:
                tasks.append(("kelly_adherence", self.estimate_kelly_adherence_confidence(agent_data["kelly_deviations"], agent_id)))
            
            if "returns" in agent_data:
                tasks.append(("sharpe_ratio", self.estimate_sharpe_ratio_confidence(agent_data["returns"], agent_id)))
            
            if "confidences" in agent_data and "outcomes" in agent_data:
                tasks.append(("confidence_calibration", self.estimate_confidence_calibration_interval(
                    agent_data["confidences"], agent_data["outcomes"], agent_id)))
            
            # Execute all tasks
            if tasks:
                task_results = await asyncio.gather(*[task for _, task in tasks])
                
                # Collect results
                for (metric_name, _), result in zip(tasks, task_results):
                    results[metric_name] = result
            
            return results
            
        except Exception as e:
            self.logger.error(f"Multiple metrics estimation failed: {e}")
            return {}
    
    def get_estimation_summary(self, intervals: Dict[str, ConfidenceInterval]) -> Dict[str, Any]:
        """Generate summary of confidence interval estimations"""
        try:
            summary = {
                "timestamp": datetime.now().isoformat(),
                "total_metrics": len(intervals),
                "confidence_level": self.config.confidence_level,
                "bootstrap_samples": self.config.n_bootstrap_samples,
                "metrics_summary": {}
            }
            
            for metric_name, interval in intervals.items():
                summary["metrics_summary"][metric_name] = {
                    "point_estimate": interval.point_estimate,
                    "confidence_interval": [interval.lower_bound, interval.upper_bound],
                    "interval_width": interval.width,
                    "margin_of_error": interval.margin_of_error,
                    "method": interval.method,
                    "bootstrap_samples": interval.n_bootstrap_samples
                }
            
            # Overall quality assessment
            narrow_intervals = sum(1 for interval in intervals.values() if interval.width < 0.2)
            summary["estimation_quality"] = {
                "narrow_intervals": narrow_intervals,
                "narrow_interval_ratio": narrow_intervals / len(intervals) if intervals else 0,
                "quality": "high" if narrow_intervals / len(intervals) > 0.7 else "medium" if narrow_intervals / len(intervals) > 0.4 else "low"
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Summary generation failed: {e}")
            return {"error": str(e)}