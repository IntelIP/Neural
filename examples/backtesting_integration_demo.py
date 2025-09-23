"""
Comprehensive Backtesting Integration Demo

This example demonstrates the full Neural SDK backtesting capabilities,
showing how to:

1. Run event-driven backtests with realistic market simulation
2. Validate strategies with walk-forward analysis
3. Optimize parameters with multiple methods
4. Analyze performance and robustness
5. Generate comprehensive reports

The demo integrates the complete backtesting framework with the
strategy framework for comprehensive strategy development.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Neural SDK imports
from neural.backtesting import (
    BacktestEngine, BacktestConfig, BacktestValidator, ValidationConfig,
    ValidationMethod, ParameterOptimizer, OptimizationConfig, OptimizationMethod,
    OptimizationMetric
)
from neural.backtesting.simulator import FillSimulation
from neural.strategy.library.mean_reversion import BasicMeanReversionStrategy
from neural.strategy.library.volume_anomaly import VolumeAnomalyStrategy
from neural.strategy.builder import StrategyComposer, AggregationMethod, AllocationMethod
from neural.analysis.market_data import MarketDataStore
from neural.analysis.database import get_database

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BacktestingDemo:
    """
    Comprehensive backtesting demonstration showcasing all framework capabilities.
    """
    
    def __init__(self):
        """Initialize the demo with test data and configurations."""
        self.db_path = "demo_backtest.db"
        self.data_store = MarketDataStore(get_database(self.db_path))
        
        # Demo market IDs
        self.market_ids = ["NFL_CHIEFS_WIN", "NBA_LAKERS_WIN", "ELECTION_2024"]
        
        # Date ranges for testing
        self.full_range = (
            datetime(2024, 1, 1),
            datetime(2024, 12, 31)
        )
        
        logger.info("🚀 Initialized BacktestingDemo")
    
    async def run_full_demo(self):
        """Run the complete backtesting demo."""
        logger.info("=" * 60)
        logger.info("🎯 NEURAL SDK BACKTESTING FRAMEWORK DEMO")
        logger.info("=" * 60)
        
        # Generate some demo data first
        await self._generate_demo_data()
        
        # 1. Basic Backtesting
        logger.info("\n1️⃣  BASIC BACKTESTING")
        await self._demo_basic_backtesting()
        
        # 2. Multi-Strategy Composition Backtesting
        logger.info("\n2️⃣  MULTI-STRATEGY BACKTESTING")
        await self._demo_strategy_composition_backtesting()
        
        # 3. Market Simulation Comparison
        logger.info("\n3️⃣  MARKET SIMULATION COMPARISON")
        await self._demo_market_simulation_comparison()
        
        # 4. Strategy Validation
        logger.info("\n4️⃣  STRATEGY VALIDATION")
        await self._demo_strategy_validation()
        
        # 5. Parameter Optimization
        logger.info("\n5️⃣  PARAMETER OPTIMIZATION")
        await self._demo_parameter_optimization()
        
        # 6. Comprehensive Analysis
        logger.info("\n6️⃣  COMPREHENSIVE ANALYSIS")
        await self._demo_comprehensive_analysis()
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ BACKTESTING DEMO COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
    
    async def _generate_demo_data(self):
        """Generate synthetic market data for demonstration."""
        logger.info("📊 Generating synthetic market data...")
        
        # This would normally load real historical data
        # For demo purposes, we'll create synthetic data
        
        import numpy as np
        from datetime import datetime
        
        base_time = int(datetime(2024, 1, 1).timestamp())
        
        for market_id in self.market_ids:
            # Generate 365 days of hourly price data
            for day in range(365):
                for hour in range(24):
                    timestamp = base_time + (day * 24 * 3600) + (hour * 3600)
                    
                    # Create realistic price movement
                    if hour == 0:  # Start of day
                        base_price = 0.5 + np.random.normal(0, 0.1)
                        base_price = max(0.05, min(0.95, base_price))
                    
                    # Add some random walk with mean reversion
                    price_change = np.random.normal(0, 0.02) - (base_price - 0.5) * 0.05
                    base_price = max(0.05, min(0.95, base_price + price_change))
                    
                    # Create bid-ask spread
                    spread = np.random.uniform(0.01, 0.03)
                    bid = base_price - spread / 2
                    ask = base_price + spread / 2
                    
                    # Volume varies with activity
                    volume = max(10, int(np.random.exponential(200) * (1 + 0.5 * np.sin(hour * np.pi / 12))))
                    
                    # Store price update
                    price_update = {
                        'market_id': market_id,
                        'timestamp': timestamp,
                        'bid': bid,
                        'ask': ask,
                        'last': base_price,
                        'volume': volume,
                        'open_interest': volume * 10
                    }
                    
                    await self._store_price_update(price_update)
        
        logger.info(f"✅ Generated synthetic data for {len(self.market_ids)} markets")
    
    async def _store_price_update(self, update: dict):
        """Store a price update (simplified for demo)."""
        # In real implementation, this would use the MarketDataStore properly
        pass
    
    async def _demo_basic_backtesting(self):
        """Demonstrate basic backtesting functionality."""
        logger.info("🔧 Setting up basic backtest...")
        
        # Create a simple mean reversion strategy
        strategy = BasicMeanReversionStrategy(
            lookback_period=24,  # 24 hours
            threshold_std=2.0,
            position_size=0.05,
            max_hold_hours=12
        )
        
        # Configure backtest
        config = BacktestConfig(
            initial_capital=10000.0,
            start_date=datetime(2024, 6, 1),
            end_date=datetime(2024, 8, 31),
            max_position_size=0.10,
            max_positions=5
        )
        
        # Run backtest
        engine = BacktestEngine(config)
        engine.add_strategy(strategy)
        engine.set_data_source(self.data_store)
        
        logger.info("🚀 Running basic backtest...")
        result = await engine.run(self.market_ids[:2])  # Test on 2 markets
        
        # Display results
        self._display_backtest_results("Basic Mean Reversion", result)
    
    async def _demo_strategy_composition_backtesting(self):
        """Demonstrate multi-strategy backtesting."""
        logger.info("🎼 Setting up multi-strategy composition...")
        
        # Create multiple strategies
        mean_reversion = BasicMeanReversionStrategy(
            lookback_period=20,
            threshold_std=1.8,
            position_size=0.04
        )
        
        volume_anomaly = VolumeAnomalyStrategy(
            lookback_period=48,
            volume_threshold=2.5,
            position_size=0.03
        )
        
        # Create strategy composer
        composer = StrategyComposer(
            name="Multi-Strategy Demo",
            aggregation_method=AggregationMethod.WEIGHTED_AVERAGE,
            allocation_method=AllocationMethod.CONFIDENCE_WEIGHTED,
            max_allocation_per_strategy=0.4
        )
        
        # Add strategies with weights
        composer.add_strategy(mean_reversion, weight=0.6)
        composer.add_strategy(volume_anomaly, weight=0.4)
        
        # Configure backtest
        config = BacktestConfig(
            initial_capital=25000.0,
            start_date=datetime(2024, 3, 1),
            end_date=datetime(2024, 11, 30),
            max_position_size=0.15,
            max_positions=8
        )
        
        # Run backtest
        engine = BacktestEngine(config)
        engine.add_strategy(composer)
        engine.set_data_source(self.data_store)
        
        logger.info("🚀 Running multi-strategy backtest...")
        result = await engine.run(self.market_ids)
        
        # Display results
        self._display_backtest_results("Multi-Strategy Composition", result)
    
    async def _demo_market_simulation_comparison(self):
        """Demonstrate different market simulation models."""
        logger.info("🎭 Comparing market simulation models...")
        
        strategy = BasicMeanReversionStrategy(lookback_period=24, position_size=0.05)
        
        base_config = BacktestConfig(
            initial_capital=15000.0,
            start_date=datetime(2024, 7, 1),
            end_date=datetime(2024, 9, 30),
            max_positions=3
        )
        
        # Test different simulation models
        simulation_models = [
            ("Conservative", FillSimulation.conservative_model()),
            ("Realistic", FillSimulation.realistic_model()),
            ("Optimistic", FillSimulation.optimistic_model()),
            ("Market Impact", FillSimulation.market_impact_model())
        ]
        
        results = {}
        
        for model_name, simulator in simulation_models:
            logger.info(f"   Testing {model_name} simulation...")
            
            # Create engine with specific simulator
            engine = BacktestEngine(base_config)
            engine.add_strategy(strategy)
            engine.set_data_source(self.data_store)
            # Note: In full implementation, would set simulator here
            
            result = await engine.run(self.market_ids[:1])  # Single market for speed
            results[model_name] = result
        
        # Compare results
        logger.info("📊 Simulation Model Comparison:")
        for model_name, result in results.items():
            logger.info(f"   {model_name:15} | Return: {result.total_return:7.4f} | "
                       f"Sharpe: {result.sharpe_ratio:5.2f} | Trades: {result.total_trades}")
    
    async def _demo_strategy_validation(self):
        """Demonstrate strategy validation with walk-forward analysis."""
        logger.info("🔬 Setting up strategy validation...")
        
        strategy = BasicMeanReversionStrategy(lookback_period=36, position_size=0.06)
        
        # Configure validation
        validation_config = ValidationConfig(
            method=ValidationMethod.WALK_FORWARD,
            train_period_days=60,
            test_period_days=20,
            step_days=10,
            parallel_execution=False  # Set to False for demo simplicity
        )
        
        validator = BacktestValidator(self.data_store)
        
        logger.info("🚀 Running walk-forward validation...")
        validation_result = await validator.validate_strategy(
            strategy=strategy,
            config=validation_config,
            market_ids=self.market_ids[:2],
            date_range=(datetime(2024, 4, 1), datetime(2024, 10, 31)),
            backtest_config=BacktestConfig(initial_capital=12000.0)
        )
        
        # Display validation results
        logger.info("📊 Validation Results:")
        logger.info(f"   Validation Passed: {'✅' if validation_result.validation_passed else '❌'}")
        logger.info(f"   Individual Tests: {len(validation_result.individual_results)}")
        
        if validation_result.aggregate_metrics:
            logger.info("   Aggregate Metrics:")
            for metric, value in validation_result.aggregate_metrics.items():
                logger.info(f"     {metric}: {value:.4f}")
        
        if validation_result.failure_reasons:
            logger.info("   Failure Reasons:")
            for reason in validation_result.failure_reasons:
                logger.info(f"     • {reason}")
    
    async def _demo_parameter_optimization(self):
        """Demonstrate parameter optimization."""
        logger.info("🎯 Setting up parameter optimization...")
        
        # Define parameter spaces for BasicMeanReversionStrategy
        from neural.backtesting.optimizer import ParameterSpace
        
        parameter_spaces = {
            'lookback_period': ParameterSpace(
                name='lookback_period',
                param_type='int',
                min_value=12,
                max_value=72,
                step_size=12
            ),
            'threshold_std': ParameterSpace(
                name='threshold_std',
                param_type='float',
                min_value=1.0,
                max_value=3.0,
                step_size=0.5
            ),
            'position_size': ParameterSpace(
                name='position_size',
                param_type='float',
                min_value=0.02,
                max_value=0.08,
                step_size=0.02
            )
        }
        
        # Configure optimization
        optimization_config = OptimizationConfig(
            method=OptimizationMethod.GRID_SEARCH,
            primary_metric=OptimizationMetric.SHARPE_RATIO,
            secondary_metrics=[OptimizationMetric.TOTAL_RETURN],
            parameter_spaces=parameter_spaces,
            max_iterations=20,  # Limit for demo
            validation_split=0.3,
            parallel_execution=False
        )
        
        optimizer = ParameterOptimizer(self.data_store)
        
        logger.info("🚀 Running parameter optimization...")
        optimization_result = await optimizer.optimize_parameters(
            strategy_class=BasicMeanReversionStrategy,
            config=optimization_config,
            market_ids=self.market_ids[:1],  # Single market for speed
            date_range=(datetime(2024, 5, 1), datetime(2024, 10, 31)),
            backtest_config=BacktestConfig(initial_capital=10000.0)
        )
        
        # Display optimization results
        logger.info("📊 Optimization Results:")
        logger.info(f"   Best Score: {optimization_result.best_score:.4f}")
        logger.info(f"   Iterations: {optimization_result.iterations_completed}")
        logger.info("   Best Parameters:")
        for param, value in optimization_result.best_parameters.items():
            logger.info(f"     {param}: {value}")
        
        if optimization_result.robustness_metrics:
            logger.info("   Robustness Metrics:")
            for metric, value in optimization_result.robustness_metrics.items():
                if isinstance(value, (int, float)):
                    logger.info(f"     {metric}: {value:.4f}")
    
    async def _demo_comprehensive_analysis(self):
        """Demonstrate comprehensive analysis combining all components."""
        logger.info("📈 Running comprehensive analysis...")
        
        # Create optimized strategy (normally from optimization results)
        optimized_strategy = BasicMeanReversionStrategy(
            lookback_period=36,
            threshold_std=2.0,
            position_size=0.06,
            max_hold_hours=18
        )
        
        # Run full backtest
        config = BacktestConfig(
            initial_capital=20000.0,
            start_date=datetime(2024, 1, 15),
            end_date=datetime(2024, 11, 15),
            max_position_size=0.12,
            max_positions=6,
            risk_free_rate=0.025
        )
        
        engine = BacktestEngine(config)
        engine.add_strategy(optimized_strategy)
        engine.set_data_source(self.data_store)
        
        logger.info("🚀 Running comprehensive backtest...")
        result = await engine.run(self.market_ids)
        
        # Generate comprehensive report
        self._generate_comprehensive_report(result)
    
    def _display_backtest_results(self, strategy_name: str, result):
        """Display formatted backtest results."""
        logger.info(f"📊 {strategy_name} Results:")
        logger.info(f"   Initial Capital: ${result.initial_capital:,.2f}")
        logger.info(f"   Final Capital:   ${result.final_capital:,.2f}")
        logger.info(f"   Total Return:    {result.total_return:.4f} ({result.total_return*100:.2f}%)")
        logger.info(f"   Annual Return:   {result.annual_return:.4f} ({result.annual_return*100:.2f}%)")
        logger.info(f"   Sharpe Ratio:    {result.sharpe_ratio:.3f}")
        logger.info(f"   Max Drawdown:    {result.max_drawdown:.4f} ({result.max_drawdown*100:.2f}%)")
        logger.info(f"   Total Trades:    {result.total_trades}")
        logger.info(f"   Win Rate:        {result.win_rate:.3f} ({result.win_rate*100:.1f}%)")
        logger.info(f"   Execution Time:  {result.execution_time:.2f}s")
    
    def _generate_comprehensive_report(self, result):
        """Generate a comprehensive analysis report."""
        logger.info("📄 COMPREHENSIVE ANALYSIS REPORT")
        logger.info("=" * 50)
        
        # Performance Summary
        logger.info("PERFORMANCE SUMMARY:")
        self._display_backtest_results("Final Strategy", result)
        
        # Risk Analysis
        logger.info("\nRISK ANALYSIS:")
        logger.info(f"   Volatility:      {result.performance_metrics.volatility:.4f}")
        logger.info(f"   VaR (95%):       {getattr(result.performance_metrics, 'var_95', 'N/A')}")
        logger.info(f"   Sortino Ratio:   {getattr(result.performance_metrics, 'sortino_ratio', 'N/A')}")
        logger.info(f"   Calmar Ratio:    {getattr(result.performance_metrics, 'calmar_ratio', 'N/A')}")
        
        # Trade Analysis
        if result.trades:
            profitable_trades = [t for t in result.trades if t.net_pnl > 0]
            losing_trades = [t for t in result.trades if t.net_pnl < 0]
            
            logger.info("\nTRADE ANALYSIS:")
            logger.info(f"   Profitable Trades: {len(profitable_trades)}")
            logger.info(f"   Losing Trades:     {len(losing_trades)}")
            
            if profitable_trades:
                avg_win = sum(t.net_pnl for t in profitable_trades) / len(profitable_trades)
                logger.info(f"   Average Win:       ${avg_win:.2f}")
            
            if losing_trades:
                avg_loss = sum(t.net_pnl for t in losing_trades) / len(losing_trades)
                logger.info(f"   Average Loss:      ${avg_loss:.2f}")
        
        # Strategy Breakdown
        if result.strategy_breakdown:
            logger.info("\nSTRATEGY BREAKDOWN:")
            for strategy, metrics in result.strategy_breakdown.items():
                logger.info(f"   {strategy}:")
                for metric, value in metrics.items():
                    logger.info(f"     {metric}: {value:.4f}" if isinstance(value, float) else f"     {metric}: {value}")


# Demo execution functions
async def run_quick_demo():
    """Run a quick version of the demo."""
    demo = BacktestingDemo()
    
    logger.info("🚀 Running Quick Backtesting Demo...")
    
    # Just run basic backtesting for speed
    await demo._generate_demo_data()
    await demo._demo_basic_backtesting()
    
    logger.info("✅ Quick demo completed!")


async def run_full_demo():
    """Run the complete comprehensive demo."""
    demo = BacktestingDemo()
    await demo.run_full_demo()


if __name__ == "__main__":
    # Run the demo
    print("🎯 Neural SDK Backtesting Framework Demo")
    print("Choose demo type:")
    print("1. Quick Demo (basic backtesting only)")
    print("2. Full Demo (all features)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        asyncio.run(run_quick_demo())
    else:
        asyncio.run(run_full_demo())
