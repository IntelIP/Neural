"""
Parameter Optimization for Strategy Backtesting

This module provides sophisticated parameter optimization capabilities
for finding optimal strategy parameters through various methods:

- Grid search optimization
- Random search optimization  
- Bayesian optimization
- Genetic algorithm optimization
- Multi-objective optimization
- Robust optimization

The optimizer helps prevent overfitting while finding parameters
that generalize well to out-of-sample data.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Callable, Union, Iterator
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import copy

from neural.backtesting.engine import BacktestEngine, BacktestResult, BacktestConfig
from neural.backtesting.validator import BacktestValidator, ValidationConfig, ValidationMethod
from neural.strategy.base import BaseStrategy
from neural.analysis.market_data import MarketDataStore

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """Optimization methods available."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    GENETIC_ALGORITHM = "genetic"
    MULTI_OBJECTIVE = "multi_objective"


class OptimizationMetric(Enum):
    """Metrics to optimize for."""
    TOTAL_RETURN = "total_return"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    OMEGA_RATIO = "omega_ratio"
    MAX_DRAWDOWN = "max_drawdown"  # Minimize
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    RISK_ADJUSTED_RETURN = "risk_adjusted_return"


@dataclass
class ParameterSpace:
    """Definition of parameter search space."""
    name: str
    param_type: str  # float, int, categorical, bool
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    step_size: Optional[Union[int, float]] = None
    values: Optional[List[Any]] = None  # For categorical parameters
    log_scale: bool = False  # Use log scale for sampling
    constraints: Optional[Callable] = None  # Parameter constraints


@dataclass
class OptimizationConfig:
    """Configuration for parameter optimization."""
    method: OptimizationMethod
    primary_metric: OptimizationMetric
    secondary_metrics: List[OptimizationMetric] = field(default_factory=list)
    parameter_spaces: Dict[str, ParameterSpace] = field(default_factory=dict)
    max_iterations: int = 100
    n_random_samples: int = 50  # For random search
    cross_validation_folds: int = 3
    validation_split: float = 0.2  # Holdout validation
    early_stopping: bool = True
    early_stopping_rounds: int = 10
    parallel_execution: bool = True
    max_workers: int = 4
    random_seed: int = 42
    convergence_threshold: float = 0.001
    minimize_metric: bool = False  # True for metrics like max_drawdown
    robustness_weight: float = 0.3  # Weight for robustness vs performance
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """Results from parameter optimization."""
    method: OptimizationMethod
    config: OptimizationConfig
    best_parameters: Dict[str, Any]
    best_score: float
    best_backtest_result: BacktestResult
    all_results: List[Tuple[Dict[str, Any], BacktestResult]]
    optimization_history: pd.DataFrame
    convergence_analysis: Dict[str, Any]
    robustness_metrics: Dict[str, Any]
    execution_time: float
    iterations_completed: int
    early_stopped: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class ParameterOptimizer:
    """
    Advanced parameter optimization for strategy backtesting.
    
    This class provides multiple optimization methods to find optimal
    strategy parameters while preventing overfitting through proper
    validation and robustness testing.
    """
    
    def __init__(self, data_store: MarketDataStore, validator: BacktestValidator = None):
        """
        Initialize parameter optimizer.
        
        Args:
            data_store: Market data source
            validator: Backtest validator for robustness testing
        """
        self.data_store = data_store
        self.validator = validator or BacktestValidator(data_store)
        self.optimization_history: List[OptimizationResult] = []
        
        logger.info("Initialized ParameterOptimizer")
    
    async def optimize_parameters(
        self,
        strategy_class: type,
        config: OptimizationConfig,
        market_ids: List[str],
        date_range: Tuple[datetime, datetime],
        backtest_config: BacktestConfig = None,
        progress_callback: Optional[Callable] = None
    ) -> OptimizationResult:
        """
        Optimize strategy parameters using specified method.
        
        Args:
            strategy_class: Strategy class to optimize (not instance)
            config: Optimization configuration
            market_ids: Markets to test on
            date_range: Date range for optimization
            backtest_config: Base backtest configuration
            progress_callback: Optional progress callback
            
        Returns:
            OptimizationResult with best parameters and analysis
        """
        logger.info(f"🎯 Starting {config.method.value} optimization")
        start_time = datetime.now()
        
        # Set random seed for reproducibility
        np.random.seed(config.random_seed)
        
        try:
            # Split data for validation
            train_range, validation_range = self._split_date_range(date_range, config.validation_split)
            
            # Dispatch to appropriate optimization method
            if config.method == OptimizationMethod.GRID_SEARCH:
                result = await self._grid_search_optimization(
                    strategy_class, config, market_ids, train_range, validation_range,
                    backtest_config, progress_callback
                )
            elif config.method == OptimizationMethod.RANDOM_SEARCH:
                result = await self._random_search_optimization(
                    strategy_class, config, market_ids, train_range, validation_range,
                    backtest_config, progress_callback
                )
            elif config.method == OptimizationMethod.BAYESIAN:
                result = await self._bayesian_optimization(
                    strategy_class, config, market_ids, train_range, validation_range,
                    backtest_config, progress_callback
                )
            elif config.method == OptimizationMethod.GENETIC_ALGORITHM:
                result = await self._genetic_algorithm_optimization(
                    strategy_class, config, market_ids, train_range, validation_range,
                    backtest_config, progress_callback
                )
            else:
                raise ValueError(f"Unsupported optimization method: {config.method}")
            
            # Calculate execution time
            result.execution_time = (datetime.now() - start_time).total_seconds()
            
            # Perform final validation on holdout data
            await self._final_validation(result, strategy_class, market_ids, validation_range, backtest_config)
            
            # Store in history
            self.optimization_history.append(result)
            
            logger.info(f"✅ Optimization completed in {result.execution_time:.2f}s")
            logger.info(f"   Best score: {result.best_score:.4f}")
            logger.info(f"   Best parameters: {result.best_parameters}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Optimization failed: {e}")
            raise
    
    async def _grid_search_optimization(
        self,
        strategy_class: type,
        config: OptimizationConfig,
        market_ids: List[str],
        train_range: Tuple[datetime, datetime],
        validation_range: Tuple[datetime, datetime],
        backtest_config: BacktestConfig = None,
        progress_callback: Optional[Callable] = None
    ) -> OptimizationResult:
        """
        Perform grid search optimization.
        
        Tests all combinations of parameters in the specified grid.
        """
        # Generate parameter grid
        param_grid = self._generate_parameter_grid(config.parameter_spaces)
        
        if not param_grid:
            raise ValueError("No parameter grid generated")
        
        logger.info(f"Testing {len(param_grid)} parameter combinations")
        
        # Limit grid size for practical execution
        if len(param_grid) > config.max_iterations:
            logger.warning(f"Grid size ({len(param_grid)}) exceeds max iterations, sampling randomly")
            param_grid = np.random.choice(param_grid, size=config.max_iterations, replace=False)
        
        # Test all parameter combinations
        results = []
        best_score = -np.inf if not config.minimize_metric else np.inf
        best_params = None
        best_result = None
        
        for i, params in enumerate(param_grid):
            try:
                # Create strategy with parameters
                strategy = self._create_strategy_with_params(strategy_class, params)
                
                # Run backtest
                backtest_result = await self._run_optimization_backtest(
                    strategy, market_ids, train_range, backtest_config, f"Grid_{i}"
                )
                
                # Calculate score
                score = self._calculate_optimization_score(backtest_result, config)
                
                results.append((params, backtest_result))
                
                # Check if this is the best so far
                if self._is_better_score(score, best_score, config.minimize_metric):
                    best_score = score
                    best_params = params.copy()
                    best_result = backtest_result
                
                if progress_callback:
                    progress_callback(i + 1, len(param_grid))
                
                logger.debug(f"Grid {i}: score={score:.4f}, params={params}")
                
            except Exception as e:
                logger.warning(f"Grid search iteration {i} failed: {e}")
        
        if not results:
            raise ValueError("No successful parameter combinations found")
        
        # Create optimization history DataFrame
        history_data = []
        for i, (params, result) in enumerate(results):
            row = params.copy()
            row['iteration'] = i
            row['score'] = self._calculate_optimization_score(result, config)
            row['total_return'] = result.total_return
            row['sharpe_ratio'] = result.sharpe_ratio
            row['max_drawdown'] = result.max_drawdown
            history_data.append(row)
        
        optimization_history = pd.DataFrame(history_data)
        
        return OptimizationResult(
            method=config.method,
            config=config,
            best_parameters=best_params,
            best_score=best_score,
            best_backtest_result=best_result,
            all_results=results,
            optimization_history=optimization_history,
            convergence_analysis=self._analyze_convergence(optimization_history),
            robustness_metrics={},  # Will be filled by final validation
            execution_time=0,  # Will be set by caller
            iterations_completed=len(results),
            early_stopped=False
        )
    
    async def _random_search_optimization(
        self,
        strategy_class: type,
        config: OptimizationConfig,
        market_ids: List[str],
        train_range: Tuple[datetime, datetime],
        validation_range: Tuple[datetime, datetime],
        backtest_config: BacktestConfig = None,
        progress_callback: Optional[Callable] = None
    ) -> OptimizationResult:
        """
        Perform random search optimization.
        
        Randomly samples parameter combinations within specified ranges.
        """
        max_iterations = min(config.max_iterations, config.n_random_samples)
        
        logger.info(f"Random search with {max_iterations} iterations")
        
        results = []
        best_score = -np.inf if not config.minimize_metric else np.inf
        best_params = None
        best_result = None
        no_improvement_count = 0
        
        for i in range(max_iterations):
            try:
                # Sample random parameters
                params = self._sample_random_parameters(config.parameter_spaces)
                
                # Create strategy with parameters
                strategy = self._create_strategy_with_params(strategy_class, params)
                
                # Run backtest
                backtest_result = await self._run_optimization_backtest(
                    strategy, market_ids, train_range, backtest_config, f"Random_{i}"
                )
                
                # Calculate score
                score = self._calculate_optimization_score(backtest_result, config)
                
                results.append((params, backtest_result))
                
                # Check if this is the best so far
                if self._is_better_score(score, best_score, config.minimize_metric):
                    best_score = score
                    best_params = params.copy()
                    best_result = backtest_result
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                
                # Early stopping check
                if (config.early_stopping and 
                    no_improvement_count >= config.early_stopping_rounds):
                    logger.info(f"Early stopping after {i+1} iterations")
                    break
                
                if progress_callback:
                    progress_callback(i + 1, max_iterations)
                
                logger.debug(f"Random {i}: score={score:.4f}, params={params}")
                
            except Exception as e:
                logger.warning(f"Random search iteration {i} failed: {e}")
        
        if not results:
            raise ValueError("No successful parameter combinations found")
        
        # Create optimization history
        history_data = []
        for i, (params, result) in enumerate(results):
            row = params.copy()
            row['iteration'] = i
            row['score'] = self._calculate_optimization_score(result, config)
            row['total_return'] = result.total_return
            row['sharpe_ratio'] = result.sharpe_ratio
            row['max_drawdown'] = result.max_drawdown
            history_data.append(row)
        
        optimization_history = pd.DataFrame(history_data)
        
        return OptimizationResult(
            method=config.method,
            config=config,
            best_parameters=best_params,
            best_score=best_score,
            best_backtest_result=best_result,
            all_results=results,
            optimization_history=optimization_history,
            convergence_analysis=self._analyze_convergence(optimization_history),
            robustness_metrics={},
            execution_time=0,
            iterations_completed=len(results),
            early_stopped=no_improvement_count >= config.early_stopping_rounds
        )
    
    async def _bayesian_optimization(
        self,
        strategy_class: type,
        config: OptimizationConfig,
        market_ids: List[str],
        train_range: Tuple[datetime, datetime],
        validation_range: Tuple[datetime, datetime],
        backtest_config: BacktestConfig = None,
        progress_callback: Optional[Callable] = None
    ) -> OptimizationResult:
        """
        Perform Bayesian optimization.
        
        Uses Gaussian Process to model the objective function and
        intelligently select next parameters to test.
        """
        # Placeholder implementation - would require scikit-optimize or similar
        logger.warning("Bayesian optimization not fully implemented, falling back to random search")
        
        return await self._random_search_optimization(
            strategy_class, config, market_ids, train_range, validation_range,
            backtest_config, progress_callback
        )
    
    async def _genetic_algorithm_optimization(
        self,
        strategy_class: type,
        config: OptimizationConfig,
        market_ids: List[str],
        train_range: Tuple[datetime, datetime],
        validation_range: Tuple[datetime, datetime],
        backtest_config: BacktestConfig = None,
        progress_callback: Optional[Callable] = None
    ) -> OptimizationResult:
        """
        Perform genetic algorithm optimization.
        
        Evolves parameter combinations using genetic operators
        like selection, crossover, and mutation.
        """
        # Placeholder implementation
        logger.warning("Genetic algorithm optimization not fully implemented, falling back to random search")
        
        return await self._random_search_optimization(
            strategy_class, config, market_ids, train_range, validation_range,
            backtest_config, progress_callback
        )
    
    def _generate_parameter_grid(self, parameter_spaces: Dict[str, ParameterSpace]) -> List[Dict[str, Any]]:
        """Generate all combinations of parameters for grid search."""
        if not parameter_spaces:
            return [{}]
        
        param_lists = {}
        
        for param_name, space in parameter_spaces.items():
            if space.param_type == 'categorical':
                param_lists[param_name] = space.values
            elif space.param_type == 'bool':
                param_lists[param_name] = [True, False]
            elif space.param_type in ['int', 'float']:
                if space.step_size:
                    if space.param_type == 'int':
                        param_lists[param_name] = list(range(
                            int(space.min_value), int(space.max_value) + 1, int(space.step_size)
                        ))
                    else:  # float
                        param_lists[param_name] = np.arange(
                            space.min_value, space.max_value + space.step_size, space.step_size
                        ).tolist()
                else:
                    # Default to 5 values across the range
                    param_lists[param_name] = np.linspace(
                        space.min_value, space.max_value, 5
                    ).tolist()
                    if space.param_type == 'int':
                        param_lists[param_name] = [int(x) for x in param_lists[param_name]]
        
        # Generate all combinations
        keys = param_lists.keys()
        combinations = list(product(*param_lists.values()))
        
        return [dict(zip(keys, combo)) for combo in combinations]
    
    def _sample_random_parameters(self, parameter_spaces: Dict[str, ParameterSpace]) -> Dict[str, Any]:
        """Sample random parameters from the specified spaces."""
        params = {}
        
        for param_name, space in parameter_spaces.items():
            if space.param_type == 'categorical':
                params[param_name] = np.random.choice(space.values)
            elif space.param_type == 'bool':
                params[param_name] = np.random.choice([True, False])
            elif space.param_type == 'int':
                params[param_name] = np.random.randint(space.min_value, space.max_value + 1)
            elif space.param_type == 'float':
                if space.log_scale:
                    log_min = np.log10(space.min_value)
                    log_max = np.log10(space.max_value)
                    log_val = np.random.uniform(log_min, log_max)
                    params[param_name] = 10 ** log_val
                else:
                    params[param_name] = np.random.uniform(space.min_value, space.max_value)
        
        return params
    
    def _create_strategy_with_params(self, strategy_class: type, params: Dict[str, Any]) -> BaseStrategy:
        """Create strategy instance with specified parameters."""
        try:
            return strategy_class(**params)
        except Exception as e:
            logger.error(f"Failed to create strategy with params {params}: {e}")
            raise
    
    async def _run_optimization_backtest(
        self,
        strategy: BaseStrategy,
        market_ids: List[str],
        date_range: Tuple[datetime, datetime],
        backtest_config: BacktestConfig = None,
        run_id: str = "opt"
    ) -> BacktestResult:
        """Run a single backtest for optimization."""
        start_date, end_date = date_range
        
        test_config = backtest_config or BacktestConfig()
        test_config.start_date = start_date
        test_config.end_date = end_date
        test_config.metadata = test_config.metadata or {}
        test_config.metadata['optimization_run'] = run_id
        
        engine = BacktestEngine(test_config)
        engine.add_strategy(strategy)
        engine.set_data_source(self.data_store)
        
        return await engine.run(market_ids)
    
    def _calculate_optimization_score(self, result: BacktestResult, config: OptimizationConfig) -> float:
        """Calculate optimization score from backtest result."""
        primary_score = self._get_metric_value(result, config.primary_metric)
        
        # Add secondary metrics with lower weights
        total_score = primary_score
        
        for i, metric in enumerate(config.secondary_metrics):
            weight = 0.1 / (i + 1)  # Diminishing weights for secondary metrics
            secondary_score = self._get_metric_value(result, metric)
            total_score += weight * secondary_score
        
        return total_score
    
    def _get_metric_value(self, result: BacktestResult, metric: OptimizationMetric) -> float:
        """Extract metric value from backtest result."""
        if metric == OptimizationMetric.TOTAL_RETURN:
            return result.total_return
        elif metric == OptimizationMetric.SHARPE_RATIO:
            return result.sharpe_ratio
        elif metric == OptimizationMetric.MAX_DRAWDOWN:
            return -result.max_drawdown  # Negative because we want to minimize drawdown
        elif metric == OptimizationMetric.WIN_RATE:
            return result.win_rate
        else:
            # For other metrics, try to get from performance_metrics
            return getattr(result.performance_metrics, metric.value, 0.0)
    
    def _is_better_score(self, new_score: float, best_score: float, minimize: bool) -> bool:
        """Check if new score is better than current best."""
        if minimize:
            return new_score < best_score
        else:
            return new_score > best_score
    
    def _split_date_range(
        self, 
        date_range: Tuple[datetime, datetime], 
        validation_split: float
    ) -> Tuple[Tuple[datetime, datetime], Tuple[datetime, datetime]]:
        """Split date range into training and validation periods."""
        start_date, end_date = date_range
        total_days = (end_date - start_date).days
        
        train_days = int(total_days * (1 - validation_split))
        split_date = start_date + timedelta(days=train_days)
        
        train_range = (start_date, split_date)
        validation_range = (split_date, end_date)
        
        return train_range, validation_range
    
    def _analyze_convergence(self, optimization_history: pd.DataFrame) -> Dict[str, Any]:
        """Analyze optimization convergence characteristics."""
        if optimization_history.empty:
            return {}
        
        scores = optimization_history['score'].values
        iterations = optimization_history['iteration'].values
        
        # Running maximum (or minimum for minimization problems)
        running_best = np.maximum.accumulate(scores)  # Assuming maximization
        
        # Find when convergence occurred (no improvement for several iterations)
        improvement_iterations = np.where(np.diff(running_best) > 0)[0] + 1
        
        convergence_analysis = {
            'final_score': scores[-1],
            'best_score': np.max(scores),
            'best_iteration': int(np.argmax(scores)),
            'improvement_iterations': len(improvement_iterations),
            'convergence_rate': len(improvement_iterations) / len(scores) if scores.size > 0 else 0,
            'score_std': float(np.std(scores)),
            'score_trend': 'improving' if scores[-1] > scores[0] else 'declining'
        }
        
        return convergence_analysis
    
    async def _final_validation(
        self,
        result: OptimizationResult,
        strategy_class: type,
        market_ids: List[str],
        validation_range: Tuple[datetime, datetime],
        backtest_config: BacktestConfig = None
    ) -> None:
        """Perform final validation on holdout data."""
        try:
            # Create strategy with best parameters
            best_strategy = self._create_strategy_with_params(strategy_class, result.best_parameters)
            
            # Run validation backtest
            validation_result = await self._run_optimization_backtest(
                best_strategy, market_ids, validation_range, backtest_config, "Final_Validation"
            )
            
            # Calculate robustness metrics
            train_performance = result.best_score
            validation_performance = self._calculate_optimization_score(validation_result, result.config)
            
            performance_degradation = train_performance - validation_performance
            degradation_pct = performance_degradation / abs(train_performance) if train_performance != 0 else 0
            
            result.robustness_metrics = {
                'validation_score': validation_performance,
                'performance_degradation': performance_degradation,
                'degradation_percentage': degradation_pct,
                'robustness_score': max(0, 1 - abs(degradation_pct)),
                'validation_sharpe': validation_result.sharpe_ratio,
                'validation_return': validation_result.total_return,
                'validation_max_drawdown': validation_result.max_drawdown
            }
            
            logger.info(f"Validation performance: {validation_performance:.4f} vs training: {train_performance:.4f}")
            
        except Exception as e:
            logger.warning(f"Final validation failed: {e}")
            result.robustness_metrics = {'validation_failed': True, 'error': str(e)}
    
    def create_parameter_space(
        self,
        name: str,
        param_type: str,
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        step_size: Optional[Union[int, float]] = None,
        values: Optional[List[Any]] = None,
        log_scale: bool = False
    ) -> ParameterSpace:
        """Helper method to create parameter space."""
        return ParameterSpace(
            name=name,
            param_type=param_type,
            min_value=min_value,
            max_value=max_value,
            step_size=step_size,
            values=values,
            log_scale=log_scale
        )
    
    def get_optimization_history(self) -> List[OptimizationResult]:
        """Get history of all optimization runs."""
        return self.optimization_history.copy()
    
    def generate_optimization_report(self, result: OptimizationResult) -> str:
        """Generate human-readable optimization report."""
        report = f"""
🎯 PARAMETER OPTIMIZATION REPORT
{'=' * 50}

Method: {result.method.value.upper()}
Iterations: {result.iterations_completed}
Execution Time: {result.execution_time:.2f}s
Early Stopped: {'Yes' if result.early_stopped else 'No'}

BEST PARAMETERS:
"""
        
        for param, value in result.best_parameters.items():
            if isinstance(value, float):
                report += f"  {param}: {value:.6f}\n"
            else:
                report += f"  {param}: {value}\n"
        
        report += f"\nBEST SCORE: {result.best_score:.6f}\n"
        
        if result.best_backtest_result:
            report += f"""
BEST RESULT METRICS:
  Total Return: {result.best_backtest_result.total_return:.4f}
  Sharpe Ratio: {result.best_backtest_result.sharpe_ratio:.4f}
  Max Drawdown: {result.best_backtest_result.max_drawdown:.4f}
  Win Rate: {result.best_backtest_result.win_rate:.4f}
  Total Trades: {result.best_backtest_result.total_trades}
"""
        
        if result.robustness_metrics:
            report += "\nROBUSTNESS METRICS:\n"
            for metric, value in result.robustness_metrics.items():
                if isinstance(value, float):
                    report += f"  {metric.replace('_', ' ').title()}: {value:.4f}\n"
                else:
                    report += f"  {metric.replace('_', ' ').title()}: {value}\n"
        
        if result.convergence_analysis:
            report += "\nCONVERGENCE ANALYSIS:\n"
            for metric, value in result.convergence_analysis.items():
                if isinstance(value, float):
                    report += f"  {metric.replace('_', ' ').title()}: {value:.4f}\n"
                else:
                    report += f"  {metric.replace('_', ' ').title()}: {value}\n"
        
        return report
