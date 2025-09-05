#!/usr/bin/env python3
"""
Main Backtesting Runner
Complete backtesting workflow with parameter optimization
"""

import asyncio
import logging
from datetime import datetime
from typing import List
import argparse

import sys
import os
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtesting.backtest_engine import BacktestEngine, StrategyParameters
from src.backtesting.historical_data_collector import HistoricalDataCollector
from src.backtesting.parameter_optimizer import ParameterOptimizer
from src.backtesting.performance_analyzer import PerformanceAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BacktestRunner:
    """
    Main backtesting orchestrator
    """
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.engine = BacktestEngine(initial_capital=initial_capital)
        self.collector = HistoricalDataCollector()
        self.optimizer = ParameterOptimizer(self.engine)
        self.analyzer = PerformanceAnalyzer()
    
    async def run_simple_backtest(
        self,
        start_date: datetime,
        end_date: datetime,
        markets: List[str],
        params: StrategyParameters = None
    ):
        """
        Run a simple backtest with default or specified parameters
        """
        logger.info("="*60)
        logger.info("SIMPLE BACKTEST")
        logger.info("="*60)
        
        if params is None:
            params = StrategyParameters()
        
        logger.info(f"Period: {start_date.date()} to {end_date.date()}")
        logger.info(f"Markets: {markets}")
        logger.info(f"Initial Capital: ${self.initial_capital:,.2f}")
        
        # Collect historical data if needed
        for market in markets:
            await self.collector.load_or_collect(
                market_ticker=market,
                start_date=start_date,
                end_date=end_date
            )
        
        # Run backtest
        results = await self.engine.run_backtest(
            strategy_params=params,
            start_date=start_date,
            end_date=end_date,
            markets=markets
        )
        
        # Analyze performance
        metrics = self.analyzer.analyze(results)
        
        # Display results
        self._display_results(metrics)
        
        # Create report
        self.analyzer.create_report(
            results, 
            metrics,
            filename=f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        )
        
        # Create plots
        self.analyzer.plot_results(
            results,
            save_path=f"backtesting/results/plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        
        return results, metrics
    
    async def run_optimization(
        self,
        start_date: datetime,
        end_date: datetime,
        markets: List[str],
        optimization_type: str = "grid"
    ):
        """
        Run parameter optimization
        """
        logger.info("="*60)
        logger.info("PARAMETER OPTIMIZATION")
        logger.info("="*60)
        
        logger.info(f"Optimization Type: {optimization_type}")
        logger.info(f"Period: {start_date.date()} to {end_date.date()}")
        logger.info(f"Markets: {markets}")
        
        # Collect historical data
        for market in markets:
            await self.collector.load_or_collect(
                market_ticker=market,
                start_date=start_date,
                end_date=end_date
            )
        
        if optimization_type == "grid":
            # Grid search optimization
            param_grid = {
                'kelly_multiplier': [0.1, 0.25, 0.4],
                'stop_loss_pct': [0.05, 0.10, 0.15],
                'take_profit_pct': [0.20, 0.30, 0.40],
                'min_confidence': [0.60, 0.65, 0.70],
                'price_spike_threshold': [0.03, 0.05, 0.07]
            }
            
            result = await self.optimizer.grid_search(
                param_grid=param_grid,
                start_date=start_date,
                end_date=end_date,
                markets=markets,
                max_combinations=50
            )
            
        elif optimization_type == "random":
            # Random search optimization
            param_distributions = {
                'kelly_multiplier': (0.05, 0.5),
                'stop_loss_pct': (0.05, 0.20),
                'take_profit_pct': (0.15, 0.50),
                'min_confidence': (0.50, 0.80),
                'price_spike_threshold': (0.02, 0.10),
                'volume_surge_multiplier': (2.0, 5.0),
                'sentiment_shift_threshold': (0.2, 0.5)
            }
            
            result = await self.optimizer.random_search(
                param_distributions=param_distributions,
                n_iterations=30,
                start_date=start_date,
                end_date=end_date,
                markets=markets
            )
            
        elif optimization_type == "genetic":
            # Genetic algorithm optimization
            param_ranges = {
                'kelly_multiplier': (0.05, 0.5),
                'stop_loss_pct': (0.05, 0.20),
                'take_profit_pct': (0.15, 0.50),
                'min_confidence': (0.50, 0.80),
                'price_spike_threshold': (0.02, 0.10)
            }
            
            result = await self.optimizer.genetic_algorithm(
                param_ranges=param_ranges,
                population_size=20,
                generations=5,
                start_date=start_date,
                end_date=end_date,
                markets=markets
            )
        else:
            raise ValueError(f"Unknown optimization type: {optimization_type}")
        
        # Display best parameters
        logger.info("\n" + "="*60)
        logger.info("OPTIMIZATION RESULTS")
        logger.info("="*60)
        
        logger.info("\nBest Parameters:")
        for key, value in result.best_params.to_dict().items():
            logger.info(f"  {key}: {value}")
        
        logger.info("\nBest Performance:")
        metrics = self.analyzer.analyze(result.best_performance)
        self._display_results(metrics)
        
        logger.info("\nParameter Importance:")
        for param, importance in sorted(
            result.parameter_importance.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            logger.info(f"  {param}: {importance:.2%}")
        
        # Save optimization results
        self.optimizer.save_results(
            result,
            f"optimization_{optimization_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        return result
    
    async def run_walk_forward(
        self,
        start_date: datetime,
        end_date: datetime,
        markets: List[str],
        window_months: int = 3,
        step_months: int = 1
    ):
        """
        Run walk-forward analysis
        """
        logger.info("="*60)
        logger.info("WALK-FORWARD ANALYSIS")
        logger.info("="*60)
        
        logger.info(f"Period: {start_date.date()} to {end_date.date()}")
        logger.info(f"Window: {window_months} months, Step: {step_months} months")
        logger.info(f"Markets: {markets}")
        
        # Collect all historical data upfront
        for market in markets:
            await self.collector.load_or_collect(
                market_ticker=market,
                start_date=start_date,
                end_date=end_date
            )
        
        param_ranges = {
            'kelly_multiplier': (0.1, 0.4),
            'stop_loss_pct': (0.05, 0.15),
            'take_profit_pct': (0.20, 0.40),
            'min_confidence': (0.60, 0.75)
        }
        
        results = await self.optimizer.walk_forward_optimization(
            param_ranges=param_ranges,
            start_date=start_date,
            end_date=end_date,
            markets=markets,
            window_months=window_months,
            step_months=step_months
        )
        
        logger.info(f"\nCompleted {len(results)} walk-forward windows")
        
        # Analyze out-of-sample performance
        oos_returns = []
        for i, result in enumerate(results):
            if result.best_performance:
                oos_returns.append(result.best_performance.total_return)
                logger.info(f"Window {i+1}: OOS Return = {result.best_performance.total_return:.2%}")
        
        if oos_returns:
            avg_oos = sum(oos_returns) / len(oos_returns)
            logger.info(f"\nAverage Out-of-Sample Return: {avg_oos:.2%}")
        
        return results
    
    async def compare_strategies(
        self,
        start_date: datetime,
        end_date: datetime,
        markets: List[str]
    ):
        """
        Compare different strategy configurations
        """
        logger.info("="*60)
        logger.info("STRATEGY COMPARISON")
        logger.info("="*60)
        
        strategies = {
            "Conservative": StrategyParameters(
                kelly_multiplier=0.10,
                stop_loss_pct=0.05,
                take_profit_pct=0.15,
                min_confidence=0.75
            ),
            "Moderate": StrategyParameters(
                kelly_multiplier=0.25,
                stop_loss_pct=0.10,
                take_profit_pct=0.30,
                min_confidence=0.65
            ),
            "Aggressive": StrategyParameters(
                kelly_multiplier=0.40,
                stop_loss_pct=0.15,
                take_profit_pct=0.50,
                min_confidence=0.55
            ),
            "Arbitrage-Focus": StrategyParameters(
                kelly_multiplier=0.30,
                arbitrage_min_profit=0.01,
                min_confidence=0.70,
                stop_loss_pct=0.03
            )
        }
        
        comparison_results = []
        
        for name, params in strategies.items():
            logger.info(f"\nTesting {name} strategy...")
            
            results = await self.engine.run_backtest(
                strategy_params=params,
                start_date=start_date,
                end_date=end_date,
                markets=markets
            )
            
            comparison_results.append((name, results))
        
        # Compare strategies
        comparison_df = self.analyzer.compare_strategies(
            comparison_results,
            save_path="backtesting/results/strategy_comparison.csv"
        )
        
        logger.info("\n" + "="*60)
        logger.info("COMPARISON RESULTS")
        logger.info("="*60)
        print(comparison_df.to_string())
        
        return comparison_results
    
    def _display_results(self, metrics):
        """Display performance metrics"""
        logger.info("\n" + "="*60)
        logger.info("PERFORMANCE METRICS")
        logger.info("="*60)
        
        logger.info("\nReturns:")
        logger.info(f"  Total Return: {metrics.total_return:.2%}")
        logger.info(f"  Annualized Return: {metrics.annualized_return:.2%}")
        
        logger.info("\nRisk Metrics:")
        logger.info(f"  Volatility: {metrics.volatility:.2%}")
        logger.info(f"  Max Drawdown: {metrics.max_drawdown:.2%}")
        logger.info(f"  VaR (95%): {metrics.var_95:.2%}")
        
        logger.info("\nRisk-Adjusted Returns:")
        logger.info(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        logger.info(f"  Sortino Ratio: {metrics.sortino_ratio:.2f}")
        logger.info(f"  Calmar Ratio: {metrics.calmar_ratio:.2f}")
        
        logger.info("\nTrade Statistics:")
        logger.info(f"  Total Trades: {metrics.total_trades}")
        logger.info(f"  Win Rate: {metrics.win_rate:.1%}")
        logger.info(f"  Profit Factor: {metrics.profit_factor:.2f}")
        logger.info(f"  Expectancy: ${metrics.expectancy:.2f}")
        logger.info(f"  Avg Holding Period: {metrics.avg_holding_period:.1f} hours")
        
        logger.info("\nKelly Metrics:")
        logger.info(f"  Kelly Criterion: {metrics.kelly_criterion:.2%}")
        logger.info(f"  Optimal f: {metrics.optimal_f:.2%}")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Kalshi Backtesting System")
    parser.add_argument(
        "--mode",
        choices=["simple", "optimize", "walk-forward", "compare"],
        default="simple",
        help="Backtesting mode"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2024-01-01",
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-12-31",
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--markets",
        type=str,
        nargs="+",
        default=["NFL-CHIEFS-WIN", "NFL-BILLS-WIN"],
        help="Markets to backtest"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10000,
        help="Initial capital"
    )
    parser.add_argument(
        "--optimization",
        choices=["grid", "random", "genetic"],
        default="grid",
        help="Optimization type"
    )
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    
    # Create runner
    runner = BacktestRunner(initial_capital=args.capital)
    
    # Run selected mode
    if args.mode == "simple":
        await runner.run_simple_backtest(
            start_date=start_date,
            end_date=end_date,
            markets=args.markets
        )
    
    elif args.mode == "optimize":
        await runner.run_optimization(
            start_date=start_date,
            end_date=end_date,
            markets=args.markets,
            optimization_type=args.optimization
        )
    
    elif args.mode == "walk-forward":
        await runner.run_walk_forward(
            start_date=start_date,
            end_date=end_date,
            markets=args.markets,
            window_months=3,
            step_months=1
        )
    
    elif args.mode == "compare":
        await runner.compare_strategies(
            start_date=start_date,
            end_date=end_date,
            markets=args.markets
        )
    
    logger.info("\n" + "="*60)
    logger.info("BACKTEST COMPLETE")
    logger.info("="*60)


if __name__ == "__main__":
    asyncio.run(main())