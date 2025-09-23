"""
Backtest Validation Framework

This module provides sophisticated validation methods for ensuring
strategy robustness and preventing overfitting, including:

- Walk-forward analysis
- Out-of-sample testing  
- Cross-validation for time series
- Monte Carlo simulation
- Bootstrap validation
- Rolling window analysis

These validation techniques help ensure strategies will perform
in live trading conditions rather than just on historical data.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

from neural.backtesting.engine import BacktestEngine, BacktestResult, BacktestConfig
from neural.strategy.base import BaseStrategy
from neural.strategy.builder import StrategyComposer
from neural.analysis.market_data import MarketDataStore

logger = logging.getLogger(__name__)


class ValidationMethod(Enum):
    """Types of validation methods."""
    WALK_FORWARD = "walk_forward"
    OUT_OF_SAMPLE = "out_of_sample"
    ROLLING_WINDOW = "rolling_window"
    MONTE_CARLO = "monte_carlo"
    BOOTSTRAP = "bootstrap"
    CROSS_VALIDATION = "cross_validation"


@dataclass
class ValidationConfig:
    """Configuration for validation testing."""
    method: ValidationMethod
    train_period_days: int = 90  # Training period length
    test_period_days: int = 30   # Test period length
    step_days: int = 7           # Step size for rolling windows
    min_train_days: int = 30     # Minimum training period
    out_of_sample_pct: float = 0.2  # 20% for out-of-sample
    n_splits: int = 5            # Number of cross-validation splits
    monte_carlo_runs: int = 1000 # Monte Carlo iterations
    bootstrap_samples: int = 500 # Bootstrap samples
    confidence_level: float = 0.95  # Confidence level for statistics
    parallel_execution: bool = True  # Enable parallel processing
    max_workers: int = 4         # Maximum worker threads
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Results from validation testing."""
    method: ValidationMethod
    config: ValidationConfig
    individual_results: List[BacktestResult]
    aggregate_metrics: Dict[str, float]
    statistical_significance: Dict[str, float]
    consistency_metrics: Dict[str, float]
    risk_metrics: Dict[str, float]
    execution_time: float
    validation_passed: bool
    failure_reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    """Specific results for walk-forward analysis."""
    validation_result: ValidationResult
    training_periods: List[Tuple[datetime, datetime]]
    testing_periods: List[Tuple[datetime, datetime]]
    rolling_performance: pd.DataFrame
    degradation_analysis: Dict[str, Any]
    stability_metrics: Dict[str, Any]


class BacktestValidator:
    """
    Comprehensive validation framework for strategy backtesting.
    
    This class provides multiple validation methods to ensure strategy
    robustness and prevent overfitting to historical data.
    """
    
    def __init__(self, data_store: MarketDataStore):
        """
        Initialize backtest validator.
        
        Args:
            data_store: Market data source for validation
        """
        self.data_store = data_store
        self.validation_history: List[ValidationResult] = []
        
        logger.info("Initialized BacktestValidator")
    
    async def validate_strategy(
        self,
        strategy: Union[BaseStrategy, StrategyComposer],
        config: ValidationConfig,
        market_ids: List[str],
        date_range: Tuple[datetime, datetime],
        backtest_config: BacktestConfig = None,
        progress_callback: Optional[Callable] = None
    ) -> ValidationResult:
        """
        Validate a strategy using specified validation method.
        
        Args:
            strategy: Strategy to validate
            config: Validation configuration
            market_ids: Markets to test on
            date_range: Date range for validation
            backtest_config: Backtest configuration
            progress_callback: Optional progress callback
            
        Returns:
            ValidationResult with comprehensive validation metrics
        """
        logger.info(f"🔬 Starting {config.method.value} validation")
        start_time = datetime.now()
        
        try:
            # Dispatch to appropriate validation method
            if config.method == ValidationMethod.WALK_FORWARD:
                result = await self._walk_forward_validation(
                    strategy, config, market_ids, date_range, backtest_config, progress_callback
                )
            elif config.method == ValidationMethod.OUT_OF_SAMPLE:
                result = await self._out_of_sample_validation(
                    strategy, config, market_ids, date_range, backtest_config, progress_callback
                )
            elif config.method == ValidationMethod.ROLLING_WINDOW:
                result = await self._rolling_window_validation(
                    strategy, config, market_ids, date_range, backtest_config, progress_callback
                )
            elif config.method == ValidationMethod.MONTE_CARLO:
                result = await self._monte_carlo_validation(
                    strategy, config, market_ids, date_range, backtest_config, progress_callback
                )
            elif config.method == ValidationMethod.BOOTSTRAP:
                result = await self._bootstrap_validation(
                    strategy, config, market_ids, date_range, backtest_config, progress_callback
                )
            else:
                raise ValueError(f"Unsupported validation method: {config.method}")
            
            # Calculate execution time
            result.execution_time = (datetime.now() - start_time).total_seconds()
            
            # Store in history
            self.validation_history.append(result)
            
            logger.info(f"✅ Validation completed in {result.execution_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            raise
    
    async def _walk_forward_validation(
        self,
        strategy: Union[BaseStrategy, StrategyComposer],
        config: ValidationConfig,
        market_ids: List[str],
        date_range: Tuple[datetime, datetime],
        backtest_config: BacktestConfig = None,
        progress_callback: Optional[Callable] = None
    ) -> ValidationResult:
        """
        Perform walk-forward analysis.
        
        This method trains on historical data and tests on subsequent periods,
        rolling the window forward to validate strategy robustness over time.
        """
        start_date, end_date = date_range
        backtest_config = backtest_config or BacktestConfig()
        
        # Generate training and testing periods
        periods = self._generate_walk_forward_periods(
            start_date, end_date, config.train_period_days, 
            config.test_period_days, config.step_days
        )
        
        if len(periods) < 2:
            raise ValueError("Insufficient data for walk-forward analysis")
        
        logger.info(f"Generated {len(periods)} walk-forward periods")
        
        # Run backtests for each period
        backtest_results = []
        
        if config.parallel_execution:
            # Run backtests in parallel
            tasks = []
            for i, (train_start, train_end, test_start, test_end) in enumerate(periods):
                task = self._run_single_backtest(
                    strategy, backtest_config, market_ids, test_start, test_end, f"WF_{i}"
                )
                tasks.append(task)
            
            # Execute all backtests
            backtest_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter successful results
            successful_results = [
                r for r in backtest_results 
                if isinstance(r, BacktestResult)
            ]
            
        else:
            # Run backtests sequentially
            for i, (train_start, train_end, test_start, test_end) in enumerate(periods):
                try:
                    result = await self._run_single_backtest(
                        strategy, backtest_config, market_ids, test_start, test_end, f"WF_{i}"
                    )
                    backtest_results.append(result)
                    
                    if progress_callback:
                        progress_callback(i + 1, len(periods))
                        
                except Exception as e:
                    logger.warning(f"Walk-forward period {i} failed: {e}")
            
            successful_results = backtest_results
        
        if not successful_results:
            return self._create_failed_validation_result(
                config, "No successful backtest periods", []
            )
        
        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_aggregate_metrics(successful_results)
        
        # Calculate consistency metrics
        consistency_metrics = self._calculate_consistency_metrics(successful_results)
        
        # Statistical significance testing
        significance_metrics = self._calculate_statistical_significance(successful_results, config)
        
        # Risk metrics
        risk_metrics = self._calculate_risk_metrics(successful_results)
        
        # Determine if validation passed
        validation_passed, failure_reasons = self._assess_validation_success(
            successful_results, aggregate_metrics, consistency_metrics
        )
        
        return ValidationResult(
            method=config.method,
            config=config,
            individual_results=successful_results,
            aggregate_metrics=aggregate_metrics,
            statistical_significance=significance_metrics,
            consistency_metrics=consistency_metrics,
            risk_metrics=risk_metrics,
            execution_time=0,  # Will be set by caller
            validation_passed=validation_passed,
            failure_reasons=failure_reasons,
            metadata={
                'periods_tested': len(successful_results),
                'periods_failed': len(periods) - len(successful_results)
            }
        )
    
    async def _out_of_sample_validation(
        self,
        strategy: Union[BaseStrategy, StrategyComposer],
        config: ValidationConfig,
        market_ids: List[str],
        date_range: Tuple[datetime, datetime],
        backtest_config: BacktestConfig = None,
        progress_callback: Optional[Callable] = None
    ) -> ValidationResult:
        """
        Perform out-of-sample validation.
        
        Reserves a portion of data for final validation that wasn't used
        in strategy development or parameter optimization.
        """
        start_date, end_date = date_range
        backtest_config = backtest_config or BacktestConfig()
        
        # Split data into in-sample and out-of-sample periods
        total_days = (end_date - start_date).days
        out_of_sample_days = int(total_days * config.out_of_sample_pct)
        
        # Use the last portion as out-of-sample
        split_date = end_date - timedelta(days=out_of_sample_days)
        
        # In-sample period (for reference/comparison)
        in_sample_start = start_date
        in_sample_end = split_date
        
        # Out-of-sample period (the actual test)
        out_of_sample_start = split_date
        out_of_sample_end = end_date
        
        logger.info(f"Out-of-sample period: {out_of_sample_start} to {out_of_sample_end}")
        
        # Run backtests
        results = []
        
        # In-sample backtest (for comparison)
        in_sample_result = await self._run_single_backtest(
            strategy, backtest_config, market_ids, 
            in_sample_start, in_sample_end, "In_Sample"
        )
        results.append(in_sample_result)
        
        # Out-of-sample backtest (the key test)
        out_of_sample_result = await self._run_single_backtest(
            strategy, backtest_config, market_ids,
            out_of_sample_start, out_of_sample_end, "Out_Of_Sample"
        )
        results.append(out_of_sample_result)
        
        # Calculate metrics focused on out-of-sample performance
        aggregate_metrics = {
            'in_sample_return': in_sample_result.total_return,
            'out_of_sample_return': out_of_sample_result.total_return,
            'performance_degradation': in_sample_result.total_return - out_of_sample_result.total_return,
            'sharpe_degradation': in_sample_result.sharpe_ratio - out_of_sample_result.sharpe_ratio,
            'out_of_sample_sharpe': out_of_sample_result.sharpe_ratio,
            'out_of_sample_max_dd': out_of_sample_result.max_drawdown,
            'consistency_ratio': out_of_sample_result.total_return / max(in_sample_result.total_return, 0.01)
        }
        
        # Validation criteria for out-of-sample
        validation_passed = (
            out_of_sample_result.total_return > 0 and
            out_of_sample_result.sharpe_ratio > 0.5 and
            aggregate_metrics['performance_degradation'] < 0.5  # Less than 50% degradation
        )
        
        failure_reasons = []
        if out_of_sample_result.total_return <= 0:
            failure_reasons.append("Negative out-of-sample returns")
        if out_of_sample_result.sharpe_ratio <= 0.5:
            failure_reasons.append("Poor out-of-sample Sharpe ratio")
        if aggregate_metrics['performance_degradation'] >= 0.5:
            failure_reasons.append("Excessive performance degradation")
        
        return ValidationResult(
            method=config.method,
            config=config,
            individual_results=results,
            aggregate_metrics=aggregate_metrics,
            statistical_significance={},
            consistency_metrics={},
            risk_metrics=self._calculate_risk_metrics([out_of_sample_result]),
            execution_time=0,
            validation_passed=validation_passed,
            failure_reasons=failure_reasons,
            metadata={
                'in_sample_period': f"{in_sample_start} to {in_sample_end}",
                'out_of_sample_period': f"{out_of_sample_start} to {out_of_sample_end}",
                'out_of_sample_pct': config.out_of_sample_pct
            }
        )
    
    async def _rolling_window_validation(
        self,
        strategy: Union[BaseStrategy, StrategyComposer],
        config: ValidationConfig,
        market_ids: List[str],
        date_range: Tuple[datetime, datetime],
        backtest_config: BacktestConfig = None,
        progress_callback: Optional[Callable] = None
    ) -> ValidationResult:
        """
        Perform rolling window validation.
        
        Tests strategy performance across multiple fixed-size time windows
        to assess consistency and stability.
        """
        start_date, end_date = date_range
        backtest_config = backtest_config or BacktestConfig()
        
        # Generate rolling windows
        windows = self._generate_rolling_windows(
            start_date, end_date, config.test_period_days, config.step_days
        )
        
        if len(windows) < 3:
            raise ValueError("Insufficient data for rolling window analysis")
        
        logger.info(f"Testing {len(windows)} rolling windows")
        
        # Run backtests for each window
        results = []
        for i, (window_start, window_end) in enumerate(windows):
            try:
                result = await self._run_single_backtest(
                    strategy, backtest_config, market_ids,
                    window_start, window_end, f"Window_{i}"
                )
                results.append(result)
                
                if progress_callback:
                    progress_callback(i + 1, len(windows))
                    
            except Exception as e:
                logger.warning(f"Window {i} failed: {e}")
        
        if not results:
            return self._create_failed_validation_result(
                config, "No successful window periods", []
            )
        
        # Calculate metrics
        aggregate_metrics = self._calculate_aggregate_metrics(results)
        consistency_metrics = self._calculate_consistency_metrics(results)
        significance_metrics = self._calculate_statistical_significance(results, config)
        risk_metrics = self._calculate_risk_metrics(results)
        
        # Assess validation success
        validation_passed, failure_reasons = self._assess_validation_success(
            results, aggregate_metrics, consistency_metrics
        )
        
        return ValidationResult(
            method=config.method,
            config=config,
            individual_results=results,
            aggregate_metrics=aggregate_metrics,
            statistical_significance=significance_metrics,
            consistency_metrics=consistency_metrics,
            risk_metrics=risk_metrics,
            execution_time=0,
            validation_passed=validation_passed,
            failure_reasons=failure_reasons,
            metadata={'windows_tested': len(results)}
        )
    
    async def _monte_carlo_validation(
        self,
        strategy: Union[BaseStrategy, StrategyComposer],
        config: ValidationConfig,
        market_ids: List[str],
        date_range: Tuple[datetime, datetime],
        backtest_config: BacktestConfig = None,
        progress_callback: Optional[Callable] = None
    ) -> ValidationResult:
        """
        Perform Monte Carlo validation.
        
        Runs multiple simulations with randomized market conditions
        to test strategy robustness across various scenarios.
        """
        # This is a placeholder for Monte Carlo simulation
        # In practice, this would involve:
        # 1. Bootstrapping historical returns
        # 2. Generating synthetic price paths
        # 3. Running multiple backtest scenarios
        # 4. Analyzing distribution of outcomes
        
        logger.warning("Monte Carlo validation not yet fully implemented")
        
        # Run a single backtest as placeholder
        start_date, end_date = date_range
        result = await self._run_single_backtest(
            strategy, backtest_config or BacktestConfig(), 
            market_ids, start_date, end_date, "Monte_Carlo"
        )
        
        return ValidationResult(
            method=config.method,
            config=config,
            individual_results=[result],
            aggregate_metrics={'total_return': result.total_return},
            statistical_significance={},
            consistency_metrics={},
            risk_metrics=self._calculate_risk_metrics([result]),
            execution_time=0,
            validation_passed=result.total_return > 0,
            failure_reasons=[] if result.total_return > 0 else ["Negative returns"],
            metadata={'note': 'Monte Carlo validation is placeholder implementation'}
        )
    
    async def _bootstrap_validation(
        self,
        strategy: Union[BaseStrategy, StrategyComposer],
        config: ValidationConfig,
        market_ids: List[str],
        date_range: Tuple[datetime, datetime],
        backtest_config: BacktestConfig = None,
        progress_callback: Optional[Callable] = None
    ) -> ValidationResult:
        """
        Perform bootstrap validation.
        
        Uses bootstrap sampling of historical periods to assess
        strategy performance distribution.
        """
        # Placeholder implementation
        logger.warning("Bootstrap validation not yet fully implemented")
        
        start_date, end_date = date_range
        result = await self._run_single_backtest(
            strategy, backtest_config or BacktestConfig(),
            market_ids, start_date, end_date, "Bootstrap"
        )
        
        return ValidationResult(
            method=config.method,
            config=config,
            individual_results=[result],
            aggregate_metrics={'total_return': result.total_return},
            statistical_significance={},
            consistency_metrics={},
            risk_metrics=self._calculate_risk_metrics([result]),
            execution_time=0,
            validation_passed=result.total_return > 0,
            failure_reasons=[] if result.total_return > 0 else ["Negative returns"],
            metadata={'note': 'Bootstrap validation is placeholder implementation'}
        )
    
    def _generate_walk_forward_periods(
        self,
        start_date: datetime,
        end_date: datetime,
        train_days: int,
        test_days: int,
        step_days: int
    ) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """Generate training and testing periods for walk-forward analysis."""
        periods = []
        current_start = start_date
        
        while current_start < end_date:
            # Training period
            train_start = current_start
            train_end = train_start + timedelta(days=train_days)
            
            # Testing period
            test_start = train_end
            test_end = test_start + timedelta(days=test_days)
            
            # Check if we have enough data
            if test_end <= end_date:
                periods.append((train_start, train_end, test_start, test_end))
            
            # Move window forward
            current_start += timedelta(days=step_days)
        
        return periods
    
    def _generate_rolling_windows(
        self,
        start_date: datetime,
        end_date: datetime,
        window_days: int,
        step_days: int
    ) -> List[Tuple[datetime, datetime]]:
        """Generate rolling time windows."""
        windows = []
        current_start = start_date
        
        while current_start < end_date:
            window_end = current_start + timedelta(days=window_days)
            
            if window_end <= end_date:
                windows.append((current_start, window_end))
            
            current_start += timedelta(days=step_days)
        
        return windows
    
    async def _run_single_backtest(
        self,
        strategy: Union[BaseStrategy, StrategyComposer],
        config: BacktestConfig,
        market_ids: List[str],
        start_date: datetime,
        end_date: datetime,
        run_id: str
    ) -> BacktestResult:
        """Run a single backtest for validation."""
        # Create backtest configuration for this specific run
        test_config = BacktestConfig(
            initial_capital=config.initial_capital,
            start_date=start_date,
            end_date=end_date,
            commission_rate=config.commission_rate,
            slippage_model=config.slippage_model,
            max_slippage_bps=config.max_slippage_bps,
            risk_free_rate=config.risk_free_rate,
            max_positions=config.max_positions,
            max_position_size=config.max_position_size,
            metadata={'validation_run_id': run_id}
        )
        
        # Create and run backtest
        engine = BacktestEngine(test_config)
        engine.add_strategy(strategy)
        engine.set_data_source(self.data_store)
        
        result = await engine.run(market_ids)
        return result
    
    def _calculate_aggregate_metrics(self, results: List[BacktestResult]) -> Dict[str, float]:
        """Calculate aggregate metrics across multiple backtest results."""
        if not results:
            return {}
        
        returns = [r.total_return for r in results]
        sharpe_ratios = [r.sharpe_ratio for r in results]
        max_drawdowns = [r.max_drawdown for r in results]
        
        return {
            'mean_return': np.mean(returns),
            'median_return': np.median(returns),
            'std_return': np.std(returns),
            'min_return': np.min(returns),
            'max_return': np.max(returns),
            'mean_sharpe': np.mean(sharpe_ratios),
            'median_sharpe': np.median(sharpe_ratios),
            'mean_max_drawdown': np.mean(max_drawdowns),
            'worst_drawdown': np.max(max_drawdowns),
            'positive_periods': sum(1 for r in returns if r > 0) / len(returns)
        }
    
    def _calculate_consistency_metrics(self, results: List[BacktestResult]) -> Dict[str, float]:
        """Calculate consistency metrics across results."""
        if not results:
            return {}
        
        returns = [r.total_return for r in results]
        
        # Consistency measures
        positive_ratio = sum(1 for r in returns if r > 0) / len(returns)
        
        # Coefficient of variation (lower is more consistent)
        mean_return = np.mean(returns)
        cv = np.std(returns) / abs(mean_return) if mean_return != 0 else float('inf')
        
        return {
            'positive_periods_ratio': positive_ratio,
            'coefficient_of_variation': cv,
            'return_consistency': 1 / (1 + cv) if cv != float('inf') else 0
        }
    
    def _calculate_statistical_significance(
        self, 
        results: List[BacktestResult], 
        config: ValidationConfig
    ) -> Dict[str, float]:
        """Calculate statistical significance of results."""
        if not results:
            return {}
        
        returns = [r.total_return for r in results]
        
        # Simple t-test against zero return
        from scipy import stats
        
        if len(returns) > 1:
            t_stat, p_value = stats.ttest_1samp(returns, 0)
            
            return {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant_at_95': p_value < 0.05,
                'significant_at_99': p_value < 0.01
            }
        else:
            return {'insufficient_data': True}
    
    def _calculate_risk_metrics(self, results: List[BacktestResult]) -> Dict[str, float]:
        """Calculate risk metrics from results."""
        if not results:
            return {}
        
        returns = [r.total_return for r in results]
        max_drawdowns = [r.max_drawdown for r in results]
        
        return {
            'value_at_risk_95': np.percentile(returns, 5),  # 5th percentile
            'expected_shortfall': np.mean([r for r in returns if r <= np.percentile(returns, 5)]),
            'maximum_drawdown': np.max(max_drawdowns),
            'average_drawdown': np.mean(max_drawdowns)
        }
    
    def _assess_validation_success(
        self,
        results: List[BacktestResult],
        aggregate_metrics: Dict[str, float],
        consistency_metrics: Dict[str, float]
    ) -> Tuple[bool, List[str]]:
        """Assess whether validation passed based on results."""
        failure_reasons = []
        
        # Check minimum performance criteria
        if aggregate_metrics.get('mean_return', -1) <= 0:
            failure_reasons.append("Negative mean returns")
        
        if aggregate_metrics.get('mean_sharpe', -1) <= 0.5:
            failure_reasons.append("Poor Sharpe ratio")
        
        if aggregate_metrics.get('positive_periods', 0) < 0.6:
            failure_reasons.append("Low percentage of positive periods")
        
        if consistency_metrics.get('coefficient_of_variation', float('inf')) > 2.0:
            failure_reasons.append("High return variability")
        
        validation_passed = len(failure_reasons) == 0
        
        return validation_passed, failure_reasons
    
    def _create_failed_validation_result(
        self,
        config: ValidationConfig,
        reason: str,
        results: List[BacktestResult]
    ) -> ValidationResult:
        """Create a failed validation result."""
        return ValidationResult(
            method=config.method,
            config=config,
            individual_results=results,
            aggregate_metrics={},
            statistical_significance={},
            consistency_metrics={},
            risk_metrics={},
            execution_time=0,
            validation_passed=False,
            failure_reasons=[reason]
        )
    
    def get_validation_history(self) -> List[ValidationResult]:
        """Get history of all validation runs."""
        return self.validation_history.copy()
    
    def generate_validation_report(self, result: ValidationResult) -> str:
        """Generate a human-readable validation report."""
        report = f"""
🔬 VALIDATION REPORT
{'=' * 50}

Method: {result.method.value.upper()}
Status: {'✅ PASSED' if result.validation_passed else '❌ FAILED'}
Execution Time: {result.execution_time:.2f}s

AGGREGATE METRICS:
"""
        
        for metric, value in result.aggregate_metrics.items():
            if isinstance(value, float):
                report += f"  {metric.replace('_', ' ').title()}: {value:.4f}\n"
            else:
                report += f"  {metric.replace('_', ' ').title()}: {value}\n"
        
        if result.consistency_metrics:
            report += "\nCONSISTENCY METRICS:\n"
            for metric, value in result.consistency_metrics.items():
                report += f"  {metric.replace('_', ' ').title()}: {value:.4f}\n"
        
        if result.failure_reasons:
            report += "\nFAILURE REASONS:\n"
            for reason in result.failure_reasons:
                report += f"  • {reason}\n"
        
        report += f"\nIndividual Results: {len(result.individual_results)} backtests\n"
        
        return report
