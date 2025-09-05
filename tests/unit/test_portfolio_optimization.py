"""
Unit tests for Portfolio Optimization module.

Tests comprehensive portfolio optimization functionality including:
- Asset definition and management
- Allocation strategies (Kelly, Equal Weight, Risk Parity, etc.)
- Monte Carlo simulation
- Risk metrics and constraints
- Portfolio optimization and comparison
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from neural_sdk.backtesting.portfolio_optimization import (
    Asset,
    OptimizationConfig,
    PortfolioOptimizer,
    AllocationResult,
    AllocationStrategy,
    OptimizationResults
)


class TestAsset:
    """Test Asset class functionality."""
    
    def test_asset_creation(self):
        """Test basic asset creation."""
        asset = Asset(
            symbol="GAME1",
            name="Chiefs vs Chargers",
            favorite_price=0.61,
            underdog_price=0.39,
            implied_probability=0.65,
            volume=100000
        )
        
        assert asset.symbol == "GAME1"
        assert asset.name == "Chiefs vs Chargers"
        assert asset.favorite_price == 0.61
        assert asset.underdog_price == 0.39
        assert asset.implied_probability == 0.65
        assert asset.volume == 100000
    
    def test_expected_return_calculation(self):
        """Test expected return calculation for different asset types."""
        # High probability favorite (price < 0.5)
        high_prob_asset = Asset("HIGH", "High Prob Game", 0.30, 0.70, 0.75)
        expected_return = high_prob_asset.expected_return
        # Expected: (1-0.30) * 0.75 - 0.30 * (1-0.75) = 0.70 * 0.75 - 0.30 * 0.25 = 0.525 - 0.075 = 0.45
        assert abs(expected_return - 0.45) < 0.001
        
        # Lower probability favorite (price >= 0.5)
        low_prob_asset = Asset("LOW", "Low Prob Game", 0.61, 0.39, 0.65)
        expected_return = low_prob_asset.expected_return
        # Expected: (1/0.61 - 1) * 0.65 - (1-0.65) = 0.639 * 0.65 - 0.35 ≈ 0.0654
        assert expected_return > 0.05 and expected_return < 0.1
    
    def test_kelly_fraction_calculation(self):
        """Test Kelly fraction calculation."""
        # Asset with positive expected value
        profitable_asset = Asset("PROFIT", "Profitable Game", 0.61, 0.39, 0.70)
        kelly_frac = profitable_asset.kelly_fraction
        assert kelly_frac > 0
        assert kelly_frac <= 1
        
        # Asset with zero/negative expected value
        unprofitable_asset = Asset("LOSS", "Unprofitable Game", 0.61, 0.39, 0.50)
        kelly_frac = unprofitable_asset.kelly_fraction
        assert kelly_frac >= 0  # Should be clamped to 0


class TestOptimizationConfig:
    """Test OptimizationConfig validation and setup."""
    
    def test_valid_config(self):
        """Test valid configuration creation."""
        config = OptimizationConfig(
            total_budget=1000,
            max_concentration=0.25,
            strategies=['kelly', 'equal_weight']
        )
        
        assert config.total_budget == 1000
        assert config.max_concentration == 0.25
        assert config.strategies == ['kelly', 'equal_weight']
    
    def test_invalid_budget(self):
        """Test invalid budget validation."""
        with pytest.raises(ValueError, match="total_budget must be positive"):
            OptimizationConfig(total_budget=-100)
        
        with pytest.raises(ValueError, match="total_budget must be positive"):
            OptimizationConfig(total_budget=0)
    
    def test_invalid_concentration(self):
        """Test invalid concentration validation."""
        with pytest.raises(ValueError, match="max_concentration must be between 0 and 1"):
            OptimizationConfig(max_concentration=0)
        
        with pytest.raises(ValueError, match="max_concentration must be between 0 and 1"):
            OptimizationConfig(max_concentration=1.5)
    
    def test_invalid_min_assets(self):
        """Test invalid min_assets validation."""
        with pytest.raises(ValueError, match="min_assets must be at least 1"):
            OptimizationConfig(min_assets=0)


class TestPortfolioOptimizer:
    """Test PortfolioOptimizer functionality."""
    
    @pytest.fixture
    def sample_assets(self):
        """Create sample assets for testing."""
        return [
            Asset("GAME1", "Chiefs vs Chargers", 0.61, 0.39, 0.65, 100000),
            Asset("GAME2", "Northwestern vs Western Illinois", 0.99, 0.01, 0.98, 50000),
            Asset("GAME3", "Maryland vs Northern Illinois", 0.88, 0.12, 0.85, 75000),
            Asset("GAME4", "Louisville vs James Madison", 0.86, 0.14, 0.86, 80000),
            Asset("GAME5", "Boise vs Eastern Washington", 0.98, 0.02, 0.98, 30000)
        ]
    
    @pytest.fixture
    def optimizer(self, sample_assets):
        """Create optimizer with sample data."""
        config = OptimizationConfig(
            total_budget=100,
            max_concentration=0.30,
            strategies=['kelly', 'equal_weight'],
            monte_carlo_runs=1000  # Reduced for testing
        )
        
        optimizer = PortfolioOptimizer(config)
        optimizer.add_assets(sample_assets)
        return optimizer
    
    def test_add_assets(self, optimizer, sample_assets):
        """Test asset addition and management."""
        assert len(optimizer.assets) == 5
        assert "GAME1" in optimizer.assets
        assert optimizer.assets["GAME1"].name == "Chiefs vs Chargers"
    
    def test_clear_assets(self, optimizer):
        """Test asset clearing."""
        optimizer.clear_assets()
        assert len(optimizer.assets) == 0
    
    def test_get_asset_values(self, optimizer):
        """Test asset value calculation."""
        values = optimizer.get_asset_values()
        
        assert len(values) == 5
        assert all(isinstance(v, float) for v in values.values())
        
        # Chiefs should have highest value (best odds with reasonable probability)
        assert values["GAME1"] == max(values.values())
    
    def test_select_best_assets(self, optimizer):
        """Test best asset selection."""
        best_3 = optimizer.select_best_assets(3)
        
        assert len(best_3) == 3
        assert "GAME1" in best_3  # Chiefs should be in top 3
        
        # Test edge case: more assets requested than available
        best_10 = optimizer.select_best_assets(10)
        assert len(best_10) == 5  # Should return all available
    
    def test_asset_combinations(self, optimizer):
        """Test asset combination generation."""
        combos_3 = optimizer.get_asset_combinations(3)
        
        # Should be C(5,3) = 10 combinations
        assert len(combos_3) == 10
        assert all(len(combo) == 3 for combo in combos_3)
    
    def test_equal_weight_strategy(self, optimizer):
        """Test equal weight allocation strategy."""
        selected_assets = ["GAME1", "GAME2", "GAME3"]
        allocation = optimizer.strategy_equal_weight(selected_assets)
        
        assert len(allocation) == 3
        assert all(amount == pytest.approx(100/3, rel=1e-6) for amount in allocation.values())
        assert sum(allocation.values()) == pytest.approx(100, rel=1e-6)
    
    def test_kelly_criterion_strategy(self, optimizer):
        """Test Kelly criterion allocation strategy."""
        selected_assets = ["GAME1", "GAME2", "GAME3"]
        allocation = optimizer.strategy_kelly_criterion(selected_assets)
        
        assert len(allocation) == 3
        assert all(amount >= 0 for amount in allocation.values())
        assert sum(allocation.values()) == pytest.approx(100, rel=1e-3)
        
        # Assets with better value should get more allocation
        assert allocation["GAME1"] > allocation["GAME2"]  # Chiefs > Northwestern
    
    def test_probability_weighted_strategy(self, optimizer):
        """Test probability weighted allocation strategy."""
        selected_assets = ["GAME1", "GAME2", "GAME3"]
        allocation = optimizer.strategy_probability_weighted(selected_assets)
        
        assert len(allocation) == 3
        assert all(amount >= 0 for amount in allocation.values())
        assert sum(allocation.values()) == pytest.approx(100, rel=1e-6)
        
        # Higher probability assets should get more allocation
        assert allocation["GAME2"] > allocation["GAME1"]  # Northwestern (98%) > Chiefs (65%)
    
    def test_constrained_kelly_strategy(self, optimizer):
        """Test constrained Kelly criterion strategy."""
        selected_assets = ["GAME1", "GAME2", "GAME3"]
        allocation = optimizer.strategy_constrained_kelly(selected_assets)
        
        assert len(allocation) == 3
        assert all(amount >= 0 for amount in allocation.values())
        assert sum(allocation.values()) <= 100 + 1e-6  # Allow small floating point error
        
        # No asset should exceed max concentration
        total_budget = optimizer.config.total_budget
        max_allowed = total_budget * optimizer.config.max_concentration
        assert all(amount <= max_allowed + 1e-6 for amount in allocation.values())
    
    def test_risk_parity_strategy(self, optimizer):
        """Test risk parity allocation strategy."""
        selected_assets = ["GAME1", "GAME2", "GAME3"]
        allocation = optimizer.strategy_risk_parity(selected_assets)
        
        assert len(allocation) == 3
        assert all(amount >= 0 for amount in allocation.values())
        assert sum(allocation.values()) == pytest.approx(100, rel=1e-6)
    
    def test_monte_carlo_simulation(self, optimizer):
        """Test Monte Carlo simulation."""
        allocation = {"GAME1": 50, "GAME2": 30, "GAME3": 20}
        mc_results = optimizer.run_monte_carlo(allocation)
        
        assert 'mean_return' in mc_results
        assert 'std_return' in mc_results
        assert 'win_probability' in mc_results
        assert 'var_95' in mc_results
        assert 'results' in mc_results
        
        assert len(mc_results['results']) == 1000  # matches monte_carlo_runs
        assert 0 <= mc_results['win_probability'] <= 1
        assert mc_results['std_return'] >= 0
    
    def test_portfolio_metrics(self, optimizer):
        """Test portfolio metrics calculation."""
        allocation = {"GAME1": 40, "GAME2": 30, "GAME3": 30}
        metrics = optimizer.calculate_portfolio_metrics(allocation)
        
        assert 'max_concentration' in metrics
        assert 'herfindahl_index' in metrics
        assert 'diversification_ratio' in metrics
        
        assert metrics['max_concentration'] == 0.4  # 40/100
        assert 0 <= metrics['herfindahl_index'] <= 1
        assert metrics['diversification_ratio'] >= 0
    
    def test_evaluate_allocation(self, optimizer):
        """Test complete allocation evaluation."""
        allocation = {"GAME1": 40, "GAME2": 30, "GAME3": 30}
        result = optimizer.evaluate_allocation(AllocationStrategy.EQUAL_WEIGHT, allocation)
        
        assert isinstance(result, AllocationResult)
        assert result.strategy == AllocationStrategy.EQUAL_WEIGHT
        assert result.allocation == allocation
        assert len(result.assets_used) == 3
        assert isinstance(result.expected_return, float)
        assert isinstance(result.sharpe_ratio, float)
        assert 0 <= result.win_probability <= 1
        assert 0 <= result.max_concentration <= 1
        assert result.herfindahl_index > 0
    
    def test_optimize_for_asset_count(self, optimizer):
        """Test optimization for specific asset count."""
        results = optimizer.optimize_for_asset_count(3)
        
        assert len(results) >= 2  # At least kelly and equal_weight
        assert all(isinstance(r, AllocationResult) for r in results)
        assert all(len(r.assets_used) <= 3 for r in results)
    
    def test_full_optimization(self, optimizer):
        """Test complete portfolio optimization."""
        results = optimizer.optimize()
        
        assert isinstance(results, OptimizationResults)
        assert len(results.results) > 0
        assert isinstance(results.best_allocation, AllocationResult)
        assert isinstance(results.highest_return, AllocationResult)
        assert isinstance(results.most_diversified, AllocationResult)
    
    def test_compare_strategies(self, optimizer):
        """Test strategy comparison."""
        asset_list = ["GAME1", "GAME2", "GAME3"]
        comparison_df = optimizer.compare_strategies(asset_list)
        
        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) >= 3  # At least 3 strategies
        assert 'Strategy' in comparison_df.columns
        assert 'Sharpe Ratio' in comparison_df.columns
        assert 'Expected Return' in comparison_df.columns
        
        # Should be sorted by Sharpe ratio descending
        sharpe_values = comparison_df['Sharpe Ratio'].tolist()
        assert sharpe_values == sorted(sharpe_values, reverse=True)


class TestAllocationResult:
    """Test AllocationResult class functionality."""
    
    def test_allocation_result_properties(self):
        """Test AllocationResult property calculations."""
        allocation = {"GAME1": 40, "GAME2": 30, "GAME3": 30}
        
        result = AllocationResult(
            strategy=AllocationStrategy.EQUAL_WEIGHT,
            allocation=allocation,
            assets_used=["GAME1", "GAME2", "GAME3"],
            expected_return=5.0,
            std_return=10.0,
            sharpe_ratio=0.5,
            win_probability=0.6,
            max_concentration=0.4,
            herfindahl_index=0.34,
            var_95=-15.0
        )
        
        assert result.total_invested == 100
        assert result.diversification_ratio == pytest.approx(1/0.34, rel=1e-6)
        assert result.risk_adjusted_return == 0.5


class TestOptimizationResults:
    """Test OptimizationResults class functionality."""
    
    def test_optimization_results_properties(self):
        """Test OptimizationResults property access."""
        config = OptimizationConfig()
        
        result1 = AllocationResult(
            strategy=AllocationStrategy.KELLY,
            allocation={"GAME1": 100},
            assets_used=["GAME1"],
            expected_return=10.0,
            std_return=20.0,
            sharpe_ratio=0.5,
            win_probability=0.7,
            max_concentration=1.0,
            herfindahl_index=1.0,
            var_95=-25.0
        )
        
        result2 = AllocationResult(
            strategy=AllocationStrategy.EQUAL_WEIGHT,
            allocation={"GAME1": 50, "GAME2": 50},
            assets_used=["GAME1", "GAME2"],
            expected_return=8.0,
            std_return=15.0,
            sharpe_ratio=0.53,
            win_probability=0.65,
            max_concentration=0.5,
            herfindahl_index=0.5,
            var_95=-20.0
        )
        
        results = OptimizationResults(config=config, results=[result1, result2])
        
        # Best allocation should be result2 (higher Sharpe ratio)
        assert results.best_allocation == result2
        
        # Highest return should be result1
        assert results.highest_return == result1
        
        # Most diversified should be result2 (lower HHI)
        assert results.most_diversified == result2
    
    def test_empty_results_error(self):
        """Test error handling for empty results."""
        results = OptimizationResults(results=[])
        
        with pytest.raises(ValueError, match="No optimization results available"):
            _ = results.best_allocation
        
        with pytest.raises(ValueError, match="No optimization results available"):
            _ = results.highest_return
        
        with pytest.raises(ValueError, match="No optimization results available"):
            _ = results.most_diversified


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_asset_universe(self):
        """Test optimization with no assets."""
        config = OptimizationConfig()
        optimizer = PortfolioOptimizer(config)
        
        with pytest.raises(ValueError, match="No assets added to optimization universe"):
            optimizer.optimize()
    
    def test_zero_total_budget(self):
        """Test with zero allocation amounts."""
        config = OptimizationConfig(total_budget=100)
        optimizer = PortfolioOptimizer(config)
        
        # Add assets with zero expected returns
        assets = [Asset("ZERO", "Zero Return Game", 0.5, 0.5, 0.5)]
        optimizer.add_assets(assets)
        
        results = optimizer.optimize()
        assert len(results.results) > 0  # Should still return results
    
    def test_single_asset_optimization(self):
        """Test optimization with only one asset."""
        config = OptimizationConfig(total_budget=100, min_assets=1, max_assets=1)
        optimizer = PortfolioOptimizer(config)
        
        asset = Asset("SINGLE", "Single Game", 0.61, 0.39, 0.65)
        optimizer.add_asset(asset)
        
        results = optimizer.optimize()
        best = results.best_allocation
        
        assert len(best.assets_used) == 1
        assert best.max_concentration == 1.0  # 100% concentration
        assert best.allocation["SINGLE"] == pytest.approx(100, rel=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])