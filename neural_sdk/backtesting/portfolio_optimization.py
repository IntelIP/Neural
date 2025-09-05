"""
Portfolio Optimization Module for Neural SDK

Advanced portfolio allocation strategies for prediction market trading.
Integrates Kelly Criterion, risk management, and multi-asset optimization.

Features:
- Kelly Criterion optimization
- Risk parity allocation
- Concentration constraints
- Monte Carlo simulation
- Asset selection algorithms
- Performance comparison tools

Example:
    ```python
    from neural_sdk.backtesting import PortfolioOptimizer, OptimizationConfig
    
    config = OptimizationConfig(
        total_budget=1000,
        max_concentration=0.25,
        min_assets=3,
        strategies=['kelly', 'equal_weight', 'risk_parity']
    )
    
    optimizer = PortfolioOptimizer(config)
    optimizer.add_assets(market_data)
    results = optimizer.optimize()
    optimal_allocation = results.best_allocation
    ```
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from itertools import combinations
from enum import Enum

import numpy as np
import pandas as pd
from scipy import optimize

logger = logging.getLogger(__name__)


class AllocationStrategy(Enum):
    """Available portfolio allocation strategies."""
    KELLY = "kelly"
    EQUAL_WEIGHT = "equal_weight"
    RISK_PARITY = "risk_parity"
    PROBABILITY_WEIGHTED = "probability_weighted"
    VALUE_WEIGHTED = "value_weighted"
    CONSTRAINED_KELLY = "constrained_kelly"


@dataclass
class Asset:
    """Represents a tradeable asset in prediction markets."""
    symbol: str
    name: str
    favorite_price: float
    underdog_price: float
    implied_probability: float
    volume: int = 0
    
    @property
    def expected_return(self) -> float:
        """Calculate expected return based on true probability vs market price."""
        if self.favorite_price < 0.5:
            # High probability favorite
            return (1 - self.favorite_price) * self.implied_probability - self.favorite_price * (1 - self.implied_probability)
        else:
            # Lower probability favorite
            return (1/self.favorite_price - 1) * self.implied_probability - (1 - self.implied_probability)
    
    @property
    def kelly_fraction(self) -> float:
        """Calculate Kelly fraction for optimal bet sizing."""
        p = self.implied_probability
        price = self.favorite_price
        
        # Expected value calculation
        ev = self.expected_return
        
        # Kelly fraction
        if price < 0.5:
            kelly_frac = ev / (1 - price)
        else:
            kelly_frac = ev / (1/price - 1)
            
        return max(0, min(1, kelly_frac))


@dataclass
class OptimizationConfig:
    """Configuration for portfolio optimization."""
    total_budget: float = 1000.0
    max_concentration: float = 0.25  # Max 25% per asset
    min_assets: int = 1
    max_assets: Optional[int] = None
    strategies: List[str] = field(default_factory=lambda: ['kelly', 'equal_weight'])
    monte_carlo_runs: int = 10000
    risk_free_rate: float = 0.02  # 2% annual risk-free rate
    
    # Risk management
    max_drawdown_tolerance: float = 0.20
    min_win_probability: float = 0.40
    min_sharpe_ratio: float = 0.0
    
    def __post_init__(self):
        """Validate configuration."""
        if self.total_budget <= 0:
            raise ValueError("total_budget must be positive")
        if not 0 < self.max_concentration <= 1:
            raise ValueError("max_concentration must be between 0 and 1")
        if self.min_assets < 1:
            raise ValueError("min_assets must be at least 1")


@dataclass
class AllocationResult:
    """Results of portfolio allocation strategy."""
    strategy: AllocationStrategy
    allocation: Dict[str, float]
    assets_used: List[str]
    expected_return: float
    std_return: float
    sharpe_ratio: float
    win_probability: float
    max_concentration: float
    herfindahl_index: float
    var_95: float
    monte_carlo_results: Optional[np.ndarray] = None
    
    @property
    def total_invested(self) -> float:
        """Total amount invested."""
        return sum(self.allocation.values())
    
    @property
    def diversification_ratio(self) -> float:
        """Diversification ratio (lower HHI = better diversification)."""
        return 1 / self.herfindahl_index if self.herfindahl_index > 0 else 0
    
    @property
    def risk_adjusted_return(self) -> float:
        """Risk-adjusted return (Sharpe ratio)."""
        return self.sharpe_ratio


@dataclass
class OptimizationResults:
    """Complete optimization results."""
    timestamp: datetime = field(default_factory=datetime.now)
    config: OptimizationConfig = field(default_factory=OptimizationConfig)
    results: List[AllocationResult] = field(default_factory=list)
    
    @property
    def best_allocation(self) -> AllocationResult:
        """Best allocation by Sharpe ratio."""
        if not self.results:
            raise ValueError("No optimization results available")
        return max(self.results, key=lambda x: x.sharpe_ratio)
    
    @property
    def highest_return(self) -> AllocationResult:
        """Highest expected return allocation."""
        if not self.results:
            raise ValueError("No optimization results available")
        return max(self.results, key=lambda x: x.expected_return)
    
    @property
    def most_diversified(self) -> AllocationResult:
        """Most diversified allocation (lowest HHI)."""
        if not self.results:
            raise ValueError("No optimization results available")
        return min(self.results, key=lambda x: x.herfindahl_index)


class PortfolioOptimizer:
    """Advanced portfolio optimization for prediction markets."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.assets: Dict[str, Asset] = {}
        
    def add_asset(self, asset: Asset):
        """Add an asset to the optimization universe."""
        self.assets[asset.symbol] = asset
        
    def add_assets(self, assets: List[Asset]):
        """Add multiple assets to the optimization universe."""
        for asset in assets:
            self.add_asset(asset)
    
    def clear_assets(self):
        """Clear all assets from the optimization universe."""
        self.assets.clear()
    
    def get_asset_values(self) -> Dict[str, float]:
        """Calculate value score for each asset (Kelly fraction * expected return)."""
        return {
            symbol: asset.kelly_fraction * asset.expected_return
            for symbol, asset in self.assets.items()
        }
    
    def select_best_assets(self, n: int) -> List[str]:
        """Select the best N assets by value score."""
        values = self.get_asset_values()
        sorted_assets = sorted(values.items(), key=lambda x: x[1], reverse=True)
        return [symbol for symbol, _ in sorted_assets[:n]]
    
    def get_asset_combinations(self, n: int) -> List[List[str]]:
        """Get all possible combinations of N assets."""
        if n > len(self.assets):
            n = len(self.assets)
        return list(combinations(self.assets.keys(), n))
    
    def strategy_kelly_criterion(self, selected_assets: List[str]) -> Dict[str, float]:
        """Pure Kelly criterion allocation."""
        allocation = {}
        kelly_fractions = {}
        
        # Calculate Kelly fractions for selected assets
        for symbol in selected_assets:
            asset = self.assets[symbol]
            kelly_fractions[symbol] = asset.kelly_fraction
        
        # Scale to budget
        total_kelly = sum(kelly_fractions.values())
        if total_kelly > 0:
            for symbol in selected_assets:
                allocation[symbol] = (kelly_fractions[symbol] / total_kelly) * self.config.total_budget
        else:
            # Equal allocation fallback
            per_asset = self.config.total_budget / len(selected_assets)
            for symbol in selected_assets:
                allocation[symbol] = per_asset
                
        return allocation
    
    def strategy_equal_weight(self, selected_assets: List[str]) -> Dict[str, float]:
        """Equal weight allocation."""
        per_asset = self.config.total_budget / len(selected_assets)
        return {symbol: per_asset for symbol in selected_assets}
    
    def strategy_probability_weighted(self, selected_assets: List[str]) -> Dict[str, float]:
        """Weight by implied probability."""
        weights = {}
        for symbol in selected_assets:
            asset = self.assets[symbol]
            weights[symbol] = asset.implied_probability
        
        total_weight = sum(weights.values())
        allocation = {}
        for symbol in selected_assets:
            allocation[symbol] = (weights[symbol] / total_weight) * self.config.total_budget
        
        return allocation
    
    def strategy_value_weighted(self, selected_assets: List[str]) -> Dict[str, float]:
        """Weight by value score (Kelly * expected return)."""
        values = self.get_asset_values()
        selected_values = {k: values[k] for k in selected_assets if values[k] > 0}
        
        if not selected_values:
            return self.strategy_equal_weight(selected_assets)
        
        total_value = sum(selected_values.values())
        allocation = {}
        for symbol in selected_assets:
            if symbol in selected_values:
                allocation[symbol] = (selected_values[symbol] / total_value) * self.config.total_budget
            else:
                allocation[symbol] = 0
                
        return allocation
    
    def strategy_constrained_kelly(self, selected_assets: List[str]) -> Dict[str, float]:
        """Kelly criterion with concentration constraints."""
        # Start with unconstrained Kelly
        allocation = self.strategy_kelly_criterion(selected_assets)
        max_per_asset = self.config.total_budget * self.config.max_concentration
        
        # Apply constraints iteratively
        for _ in range(10):  # Max iterations
            violations = [(k, v) for k, v in allocation.items() if v > max_per_asset]
            if not violations:
                break
            
            # Reduce violations and redistribute
            excess_total = 0
            for symbol, amount in violations:
                excess = amount - max_per_asset
                allocation[symbol] = max_per_asset
                excess_total += excess
            
            # Redistribute excess to non-violating assets
            non_violating = [k for k in allocation.keys() if allocation[k] < max_per_asset]
            if non_violating and excess_total > 0:
                remaining_capacity = sum(max_per_asset - allocation[k] for k in non_violating)
                if remaining_capacity > 0:
                    for symbol in non_violating:
                        capacity = max_per_asset - allocation[symbol]
                        additional = (capacity / remaining_capacity) * excess_total
                        allocation[symbol] += additional
        
        return allocation
    
    def strategy_risk_parity(self, selected_assets: List[str]) -> Dict[str, float]:
        """Risk parity allocation (equal risk contribution)."""
        # Simple implementation: inverse volatility weighting
        volatilities = {}
        for symbol in selected_assets:
            asset = self.assets[symbol]
            # Estimate volatility from price (lower prices = higher volatility)
            vol = 1 / max(asset.favorite_price, 0.01)  # Avoid division by zero
            volatilities[symbol] = vol
        
        # Inverse volatility weights
        inv_vol_weights = {k: 1/v for k, v in volatilities.items()}
        total_weight = sum(inv_vol_weights.values())
        
        allocation = {}
        for symbol in selected_assets:
            weight = inv_vol_weights[symbol] / total_weight
            allocation[symbol] = weight * self.config.total_budget
            
        return allocation
    
    def run_monte_carlo(self, allocation: Dict[str, float]) -> Dict[str, float]:
        """Run Monte Carlo simulation for allocation."""
        results = []
        
        for _ in range(self.config.monte_carlo_runs):
            total_return = 0
            
            for symbol, amount in allocation.items():
                if amount > 0:
                    asset = self.assets[symbol]
                    # Simulate outcome based on implied probability
                    if np.random.random() < asset.implied_probability:
                        # Win
                        profit = (1/asset.favorite_price - 1) * amount
                        total_return += profit
                    else:
                        # Loss
                        total_return -= amount
            
            results.append(total_return)
        
        results = np.array(results)
        
        return {
            'mean_return': np.mean(results),
            'std_return': np.std(results),
            'win_probability': np.mean(results > 0),
            'var_95': np.percentile(results, 5),
            'results': results
        }
    
    def calculate_portfolio_metrics(self, allocation: Dict[str, float]) -> Dict[str, float]:
        """Calculate portfolio concentration and diversification metrics."""
        total_invested = sum(allocation.values())
        if total_invested == 0:
            return {
                'max_concentration': 0,
                'herfindahl_index': 0,
                'diversification_ratio': 0
            }
        
        weights = {k: v/total_invested for k, v in allocation.items() if v > 0}
        
        # Maximum concentration
        max_concentration = max(weights.values()) if weights else 0
        
        # Herfindahl-Hirschman Index
        hhi = sum(w**2 for w in weights.values())
        
        return {
            'max_concentration': max_concentration,
            'herfindahl_index': hhi,
            'diversification_ratio': 1/hhi if hhi > 0 else 0
        }
    
    def evaluate_allocation(self, strategy: AllocationStrategy, allocation: Dict[str, float]) -> AllocationResult:
        """Evaluate a complete allocation strategy."""
        # Run Monte Carlo simulation
        mc_results = self.run_monte_carlo(allocation)
        
        # Calculate portfolio metrics
        portfolio_metrics = self.calculate_portfolio_metrics(allocation)
        
        # Calculate Sharpe ratio
        if mc_results['std_return'] > 0:
            sharpe_ratio = mc_results['mean_return'] / mc_results['std_return']
        else:
            sharpe_ratio = 0
        
        # Filter non-zero allocations
        assets_used = [k for k, v in allocation.items() if v > 0]
        
        return AllocationResult(
            strategy=strategy,
            allocation=allocation,
            assets_used=assets_used,
            expected_return=mc_results['mean_return'],
            std_return=mc_results['std_return'],
            sharpe_ratio=sharpe_ratio,
            win_probability=mc_results['win_probability'],
            max_concentration=portfolio_metrics['max_concentration'],
            herfindahl_index=portfolio_metrics['herfindahl_index'],
            var_95=mc_results['var_95'],
            monte_carlo_results=mc_results['results']
        )
    
    def optimize_for_asset_count(self, n_assets: int) -> List[AllocationResult]:
        """Optimize portfolio for a specific number of assets."""
        if n_assets > len(self.assets):
            logger.warning(f"Requested {n_assets} assets but only {len(self.assets)} available")
            n_assets = len(self.assets)
        
        # Strategy mapping
        strategy_functions = {
            'kelly': (AllocationStrategy.KELLY, self.strategy_kelly_criterion),
            'equal_weight': (AllocationStrategy.EQUAL_WEIGHT, self.strategy_equal_weight),
            'probability_weighted': (AllocationStrategy.PROBABILITY_WEIGHTED, self.strategy_probability_weighted),
            'value_weighted': (AllocationStrategy.VALUE_WEIGHTED, self.strategy_value_weighted),
            'constrained_kelly': (AllocationStrategy.CONSTRAINED_KELLY, self.strategy_constrained_kelly),
            'risk_parity': (AllocationStrategy.RISK_PARITY, self.strategy_risk_parity),
        }
        
        results = []
        
        # Test strategies on best N assets
        best_assets = self.select_best_assets(n_assets)
        
        for strategy_name in self.config.strategies:
            if strategy_name in strategy_functions:
                strategy_enum, strategy_func = strategy_functions[strategy_name]
                allocation = strategy_func(best_assets)
                result = self.evaluate_allocation(strategy_enum, allocation)
                results.append(result)
        
        # Also test all possible combinations if feasible (limited to avoid explosion)
        if n_assets <= 4 and len(self.assets) <= 6:  # Reasonable limits
            all_combinations = self.get_asset_combinations(n_assets)
            
            for combo in all_combinations:
                combo_list = list(combo)
                # Test equal weight on this combination
                allocation = self.strategy_equal_weight(combo_list)
                result = self.evaluate_allocation(AllocationStrategy.EQUAL_WEIGHT, allocation)
                result.strategy = AllocationStrategy.EQUAL_WEIGHT  # Mark as combination test
                results.append(result)
        
        return results
    
    def optimize(self) -> OptimizationResults:
        """Run complete portfolio optimization."""
        if not self.assets:
            raise ValueError("No assets added to optimization universe")
        
        logger.info(f"Starting portfolio optimization with {len(self.assets)} assets")
        
        all_results = []
        max_assets = self.config.max_assets or len(self.assets)
        
        # Test different numbers of assets
        for n_assets in range(self.config.min_assets, min(max_assets + 1, len(self.assets) + 1)):
            logger.info(f"Optimizing for {n_assets} assets")
            results = self.optimize_for_asset_count(n_assets)
            all_results.extend(results)
        
        # Filter results based on constraints
        filtered_results = []
        for result in all_results:
            if (result.win_probability >= self.config.min_win_probability and
                result.sharpe_ratio >= self.config.min_sharpe_ratio and
                result.max_concentration <= self.config.max_concentration + 0.01):  # Small tolerance
                filtered_results.append(result)
        
        if not filtered_results:
            logger.warning("No results met the specified constraints, returning all results")
            filtered_results = all_results
        
        # Sort by Sharpe ratio
        filtered_results.sort(key=lambda x: x.sharpe_ratio, reverse=True)
        
        logger.info(f"Optimization complete. Found {len(filtered_results)} valid allocations")
        
        return OptimizationResults(
            config=self.config,
            results=filtered_results
        )
    
    def compare_strategies(self, asset_list: List[str]) -> pd.DataFrame:
        """Compare all strategies on a specific set of assets."""
        strategy_functions = {
            'Kelly Criterion': self.strategy_kelly_criterion,
            'Equal Weight': self.strategy_equal_weight,
            'Probability Weighted': self.strategy_probability_weighted,
            'Value Weighted': self.strategy_value_weighted,
            'Constrained Kelly': self.strategy_constrained_kelly,
            'Risk Parity': self.strategy_risk_parity,
        }
        
        comparison_data = []
        
        for strategy_name, strategy_func in strategy_functions.items():
            allocation = strategy_func(asset_list)
            mc_results = self.run_monte_carlo(allocation)
            portfolio_metrics = self.calculate_portfolio_metrics(allocation)
            
            sharpe_ratio = (mc_results['mean_return'] / mc_results['std_return'] 
                          if mc_results['std_return'] > 0 else 0)
            
            comparison_data.append({
                'Strategy': strategy_name,
                'Expected Return': mc_results['mean_return'],
                'Std Dev': mc_results['std_return'],
                'Sharpe Ratio': sharpe_ratio,
                'Win Probability': mc_results['win_probability'],
                'Max Concentration': portfolio_metrics['max_concentration'],
                'Herfindahl Index': portfolio_metrics['herfindahl_index'],
                'VaR 95%': mc_results['var_95']
            })
        
        return pd.DataFrame(comparison_data).sort_values('Sharpe Ratio', ascending=False)