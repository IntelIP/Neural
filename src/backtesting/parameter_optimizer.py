"""
Parameter Optimizer for Strategy Optimization
Uses grid search and genetic algorithms to find optimal parameters
"""

import asyncio
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import numpy as np
import pandas as pd
from dataclasses import dataclass
import json
import itertools
from pathlib import Path

from src.backtesting.backtest_engine import BacktestEngine, StrategyParameters, BacktestResults

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Results from parameter optimization"""
    best_params: StrategyParameters
    best_performance: BacktestResults
    all_results: List[Tuple[StrategyParameters, BacktestResults]]
    optimization_time: float
    parameter_importance: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for saving"""
        return {
            'best_params': self.best_params.to_dict(),
            'best_metrics': {
                'total_return': self.best_performance.total_return,
                'sharpe_ratio': self.best_performance.sharpe_ratio,
                'win_rate': self.best_performance.win_rate,
                'max_drawdown': self.best_performance.max_drawdown,
                'profit_factor': self.best_performance.profit_factor
            },
            'optimization_time': self.optimization_time,
            'parameter_importance': self.parameter_importance,
            'total_combinations_tested': len(self.all_results)
        }


class ParameterOptimizer:
    """
    Optimizer for finding best strategy parameters
    
    Optimization methods:
    - Grid search: Test all combinations
    - Random search: Test random samples
    - Genetic algorithm: Evolve parameters
    - Bayesian optimization: Smart sampling
    """
    
    def __init__(
        self,
        backtest_engine: BacktestEngine,
        objective_metric: str = "sharpe_ratio"
    ):
        self.engine = backtest_engine
        self.objective_metric = objective_metric
        self.results_cache: Dict[str, BacktestResults] = {}
    
    async def grid_search(
        self,
        param_grid: Dict[str, List[Any]],
        start_date: datetime,
        end_date: datetime,
        markets: List[str],
        max_combinations: int = 100
    ) -> OptimizationResult:
        """
        Grid search optimization
        
        Args:
            param_grid: Dictionary of parameter names and values to test
            start_date: Backtest start date
            end_date: Backtest end date
            markets: Markets to trade
            max_combinations: Maximum combinations to test
            
        Returns:
            OptimizationResult with best parameters
        """
        import time
        start_time = time.time()
        
        logger.info("Starting grid search optimization")
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))
        
        # Limit combinations if too many
        if len(combinations) > max_combinations:
            logger.warning(f"Limiting from {len(combinations)} to {max_combinations} combinations")
            import random
            combinations = random.sample(combinations, max_combinations)
        
        logger.info(f"Testing {len(combinations)} parameter combinations")
        
        # Test each combination
        all_results = []
        best_result = None
        best_params = None
        best_score = -float('inf')
        
        for i, combo in enumerate(combinations):
            # Create parameter set
            params_dict = dict(zip(param_names, combo))
            params = self._create_params_from_dict(params_dict)
            
            # Run backtest
            logger.info(f"Testing combination {i+1}/{len(combinations)}: {params_dict}")
            
            try:
                result = await self.engine.run_backtest(
                    strategy_params=params,
                    start_date=start_date,
                    end_date=end_date,
                    markets=markets
                )
                
                # Get objective score
                score = self._get_objective_score(result)
                
                all_results.append((params, result))
                
                # Track best
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_result = result
                    logger.info(f"New best score: {score:.4f}")
                
            except Exception as e:
                logger.error(f"Error testing combination {params_dict}: {e}")
        
        # Calculate parameter importance
        parameter_importance = self._calculate_parameter_importance(
            all_results, param_names
        )
        
        optimization_time = time.time() - start_time
        
        return OptimizationResult(
            best_params=best_params,
            best_performance=best_result,
            all_results=all_results,
            optimization_time=optimization_time,
            parameter_importance=parameter_importance
        )
    
    async def random_search(
        self,
        param_distributions: Dict[str, Tuple[float, float]],
        n_iterations: int = 50,
        start_date: datetime = None,
        end_date: datetime = None,
        markets: List[str] = None
    ) -> OptimizationResult:
        """
        Random search optimization
        
        Args:
            param_distributions: Parameter ranges (min, max)
            n_iterations: Number of random samples
            start_date: Backtest start date
            end_date: Backtest end date
            markets: Markets to trade
            
        Returns:
            OptimizationResult with best parameters
        """
        import time
        start_time = time.time()
        
        logger.info(f"Starting random search with {n_iterations} iterations")
        
        all_results = []
        best_result = None
        best_params = None
        best_score = -float('inf')
        
        for i in range(n_iterations):
            # Sample random parameters
            params_dict = {}
            for param_name, (min_val, max_val) in param_distributions.items():
                if isinstance(min_val, int):
                    value = np.random.randint(min_val, max_val + 1)
                else:
                    value = np.random.uniform(min_val, max_val)
                params_dict[param_name] = value
            
            params = self._create_params_from_dict(params_dict)
            
            logger.info(f"Iteration {i+1}/{n_iterations}: Testing {params_dict}")
            
            try:
                result = await self.engine.run_backtest(
                    strategy_params=params,
                    start_date=start_date,
                    end_date=end_date,
                    markets=markets
                )
                
                score = self._get_objective_score(result)
                all_results.append((params, result))
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_result = result
                    logger.info(f"New best score: {score:.4f}")
                
            except Exception as e:
                logger.error(f"Error in iteration {i+1}: {e}")
        
        parameter_importance = self._calculate_parameter_importance(
            all_results, list(param_distributions.keys())
        )
        
        optimization_time = time.time() - start_time
        
        return OptimizationResult(
            best_params=best_params,
            best_performance=best_result,
            all_results=all_results,
            optimization_time=optimization_time,
            parameter_importance=parameter_importance
        )
    
    async def genetic_algorithm(
        self,
        param_ranges: Dict[str, Tuple[float, float]],
        population_size: int = 20,
        generations: int = 10,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        start_date: datetime = None,
        end_date: datetime = None,
        markets: List[str] = None
    ) -> OptimizationResult:
        """
        Genetic algorithm optimization
        
        Args:
            param_ranges: Parameter ranges (min, max)
            population_size: Size of each generation
            generations: Number of generations to evolve
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            start_date: Backtest start date
            end_date: Backtest end date
            markets: Markets to trade
            
        Returns:
            OptimizationResult with best parameters
        """
        import time
        start_time = time.time()
        
        logger.info(f"Starting genetic algorithm: {generations} generations, population {population_size}")
        
        all_results = []
        
        # Initialize population
        population = []
        for _ in range(population_size):
            params_dict = {}
            for param_name, (min_val, max_val) in param_ranges.items():
                if isinstance(min_val, int):
                    value = np.random.randint(min_val, max_val + 1)
                else:
                    value = np.random.uniform(min_val, max_val)
                params_dict[param_name] = value
            population.append(params_dict)
        
        best_overall_params = None
        best_overall_result = None
        best_overall_score = -float('inf')
        
        # Evolution loop
        for gen in range(generations):
            logger.info(f"Generation {gen+1}/{generations}")
            
            # Evaluate fitness
            fitness_scores = []
            gen_results = []
            
            for individual in population:
                params = self._create_params_from_dict(individual)
                
                try:
                    result = await self.engine.run_backtest(
                        strategy_params=params,
                        start_date=start_date,
                        end_date=end_date,
                        markets=markets
                    )
                    
                    score = self._get_objective_score(result)
                    fitness_scores.append(score)
                    gen_results.append((params, result))
                    all_results.append((params, result))
                    
                    if score > best_overall_score:
                        best_overall_score = score
                        best_overall_params = params
                        best_overall_result = result
                        logger.info(f"New best in generation {gen+1}: {score:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error evaluating individual: {e}")
                    fitness_scores.append(-float('inf'))
            
            # Selection (tournament selection)
            new_population = []
            for _ in range(population_size):
                # Tournament of 3
                tournament_indices = np.random.choice(len(population), 3, replace=False)
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                new_population.append(population[winner_idx].copy())
            
            # Crossover
            for i in range(0, population_size - 1, 2):
                if np.random.random() < crossover_rate:
                    # Single-point crossover
                    crossover_point = np.random.randint(1, len(param_ranges))
                    param_names = list(param_ranges.keys())
                    
                    for j in range(crossover_point):
                        param_name = param_names[j]
                        temp = new_population[i][param_name]
                        new_population[i][param_name] = new_population[i+1][param_name]
                        new_population[i+1][param_name] = temp
            
            # Mutation
            for individual in new_population:
                for param_name, (min_val, max_val) in param_ranges.items():
                    if np.random.random() < mutation_rate:
                        if isinstance(min_val, int):
                            individual[param_name] = np.random.randint(min_val, max_val + 1)
                        else:
                            # Gaussian mutation
                            current = individual[param_name]
                            mutation = np.random.normal(0, (max_val - min_val) * 0.1)
                            individual[param_name] = np.clip(
                                current + mutation, min_val, max_val
                            )
            
            population = new_population
        
        parameter_importance = self._calculate_parameter_importance(
            all_results, list(param_ranges.keys())
        )
        
        optimization_time = time.time() - start_time
        
        return OptimizationResult(
            best_params=best_overall_params,
            best_performance=best_overall_result,
            all_results=all_results,
            optimization_time=optimization_time,
            parameter_importance=parameter_importance
        )
    
    def _create_params_from_dict(self, params_dict: Dict[str, Any]) -> StrategyParameters:
        """Create StrategyParameters from dictionary"""
        base_params = StrategyParameters()
        
        for key, value in params_dict.items():
            if hasattr(base_params, key):
                setattr(base_params, key, value)
        
        return base_params
    
    def _get_objective_score(self, result: BacktestResults) -> float:
        """Get objective score from backtest result"""
        if self.objective_metric == "sharpe_ratio":
            return result.sharpe_ratio
        elif self.objective_metric == "total_return":
            return result.total_return
        elif self.objective_metric == "profit_factor":
            return result.profit_factor
        elif self.objective_metric == "calmar_ratio":
            # Return / Max Drawdown
            if result.max_drawdown > 0:
                return result.total_return / result.max_drawdown
            return result.total_return
        elif self.objective_metric == "sortino_ratio":
            # Similar to Sharpe but only considers downside deviation
            if result.daily_returns:
                returns = np.array(result.daily_returns)
                downside_returns = returns[returns < 0]
                if len(downside_returns) > 0:
                    downside_std = np.std(downside_returns)
                    if downside_std > 0:
                        return np.mean(returns) / downside_std * np.sqrt(252)
            return 0
        else:
            return result.total_return
    
    def _calculate_parameter_importance(
        self,
        all_results: List[Tuple[StrategyParameters, BacktestResults]],
        param_names: List[str]
    ) -> Dict[str, float]:
        """Calculate importance of each parameter"""
        if len(all_results) < 10:
            return {name: 0.0 for name in param_names}
        
        # Extract scores and parameter values
        scores = []
        param_values = {name: [] for name in param_names}
        
        for params, result in all_results:
            scores.append(self._get_objective_score(result))
            for name in param_names:
                if hasattr(params, name):
                    param_values[name].append(getattr(params, name))
        
        # Calculate correlation between each parameter and score
        importance = {}
        scores_array = np.array(scores)
        
        for name in param_names:
            if name in param_values and param_values[name]:
                values = np.array(param_values[name])
                
                # Handle categorical or constant values
                if len(np.unique(values)) == 1:
                    importance[name] = 0.0
                else:
                    # Calculate correlation
                    correlation = np.corrcoef(values, scores_array)[0, 1]
                    importance[name] = abs(correlation) if not np.isnan(correlation) else 0.0
            else:
                importance[name] = 0.0
        
        # Normalize importance scores
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v / total_importance for k, v in importance.items()}
        
        return importance
    
    async def walk_forward_optimization(
        self,
        param_ranges: Dict[str, Tuple[float, float]],
        start_date: datetime,
        end_date: datetime,
        markets: List[str],
        window_months: int = 3,
        step_months: int = 1
    ) -> List[OptimizationResult]:
        """
        Walk-forward optimization for robust parameter selection
        
        Args:
            param_ranges: Parameter ranges to optimize
            start_date: Overall start date
            end_date: Overall end date
            markets: Markets to trade
            window_months: Training window size
            step_months: Step size for moving window
            
        Returns:
            List of optimization results for each window
        """
        from dateutil.relativedelta import relativedelta
        
        results = []
        current_start = start_date
        
        while current_start + relativedelta(months=window_months) <= end_date:
            train_end = current_start + relativedelta(months=window_months)
            test_start = train_end
            test_end = min(test_start + relativedelta(months=step_months), end_date)
            
            logger.info(f"Walk-forward window: Train {current_start} to {train_end}, Test {test_start} to {test_end}")
            
            # Optimize on training period
            opt_result = await self.random_search(
                param_distributions=param_ranges,
                n_iterations=20,
                start_date=current_start,
                end_date=train_end,
                markets=markets
            )
            
            # Test on out-of-sample period
            test_result = await self.engine.run_backtest(
                strategy_params=opt_result.best_params,
                start_date=test_start,
                end_date=test_end,
                markets=markets
            )
            
            logger.info(f"Out-of-sample return: {test_result.total_return:.1%}")
            
            results.append(opt_result)
            current_start += relativedelta(months=step_months)
        
        return results
    
    def save_results(self, result: OptimizationResult, filename: str):
        """Save optimization results to file"""
        output_path = Path(f"backtesting/optimization_results/{filename}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        logger.info(f"Saved optimization results to {output_path}")
    
    def analyze_sensitivity(
        self,
        base_params: StrategyParameters,
        param_name: str,
        param_range: Tuple[float, float],
        n_points: int = 10
    ) -> pd.DataFrame:
        """
        Analyze sensitivity of results to a single parameter
        
        Args:
            base_params: Base parameters
            param_name: Parameter to vary
            param_range: Range to test
            n_points: Number of points to test
            
        Returns:
            DataFrame with sensitivity analysis
        """
        results = []
        
        min_val, max_val = param_range
        if isinstance(min_val, int):
            values = np.linspace(min_val, max_val, n_points, dtype=int)
        else:
            values = np.linspace(min_val, max_val, n_points)
        
        for value in values:
            params = base_params
            setattr(params, param_name, value)
            
            # Run backtest (would be async in practice)
            # result = await self.engine.run_backtest(params, ...)
            
            results.append({
                param_name: value,
                # 'return': result.total_return,
                # 'sharpe': result.sharpe_ratio,
                # 'max_dd': result.max_drawdown
            })
        
        return pd.DataFrame(results)


# Example usage
async def main():
    """Example optimization run"""
    
    # Initialize components
    engine = BacktestEngine(initial_capital=10000)
    optimizer = ParameterOptimizer(engine, objective_metric="sharpe_ratio")
    
    # Define parameter grid
    param_grid = {
        'kelly_multiplier': [0.1, 0.25, 0.5],
        'stop_loss_pct': [0.05, 0.10, 0.15],
        'take_profit_pct': [0.20, 0.30, 0.40],
        'min_confidence': [0.60, 0.70, 0.80]
    }
    
    # Run grid search
    result = await optimizer.grid_search(
        param_grid=param_grid,
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31),
        markets=["NFL-CHIEFS-WIN"],
        max_combinations=50
    )
    
    # Display results
    print(f"Best parameters: {result.best_params.to_dict()}")
    print(f"Best Sharpe ratio: {result.best_performance.sharpe_ratio:.2f}")
    print(f"Parameter importance: {result.parameter_importance}")
    
    # Save results
    optimizer.save_results(result, "optimization_2024.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())