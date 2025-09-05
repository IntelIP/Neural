#!/usr/bin/env python3
"""
Portfolio Optimization Example

Demonstrates the Neural SDK's portfolio optimization capabilities using real 
prediction market data. This example shows how to:

1. Define prediction market assets
2. Configure portfolio optimization
3. Compare allocation strategies
4. Analyze risk and return metrics
5. Select optimal allocations

Based on our analysis of September 5, 2025 college football and NFL games.
"""

from neural_sdk.backtesting import (
    PortfolioOptimizer,
    OptimizationConfig,
    Asset,
    AllocationStrategy
)


def create_september_5_games():
    """Create assets for September 5, 2025 games with current market data."""
    return [
        Asset(
            symbol="BOISE",
            name="Eastern Washington at Boise State",
            favorite_price=0.98,  # Boise State to win
            underdog_price=0.02,  # Eastern Washington to win
            implied_probability=0.98,  # Your probability estimate
            volume=29776
        ),
        Asset(
            symbol="NORTHWESTERN",
            name="Western Illinois at Northwestern", 
            favorite_price=0.99,  # Northwestern to win
            underdog_price=0.01,  # Western Illinois to win
            implied_probability=0.99,
            volume=39629
        ),
        Asset(
            symbol="MARYLAND",
            name="Northern Illinois at Maryland",
            favorite_price=0.88,  # Maryland to win
            underdog_price=0.12,  # Northern Illinois to win
            implied_probability=0.85,  # Slightly lower than market
            volume=282572
        ),
        Asset(
            symbol="LOUISVILLE", 
            name="James Madison at Louisville",
            favorite_price=0.86,  # Louisville to win
            underdog_price=0.14,  # James Madison to win
            implied_probability=0.86,
            volume=529265
        ),
        Asset(
            symbol="CHIEFS",
            name="Kansas City Chiefs at Los Angeles Chargers",
            favorite_price=0.62,  # Chiefs to win
            underdog_price=0.38,  # Chargers to win
            implied_probability=0.65,  # Slightly higher than market
            volume=853104
        )
    ]


def demonstrate_basic_optimization():
    """Demonstrate basic portfolio optimization."""
    print("🏈 BASIC PORTFOLIO OPTIMIZATION")
    print("=" * 60)
    
    # Create assets
    games = create_september_5_games()
    
    # Configure optimization for $50 investment
    config = OptimizationConfig(
        total_budget=50,
        max_concentration=0.35,  # Max 35% per asset
        strategies=['kelly', 'equal_weight', 'constrained_kelly'],
        monte_carlo_runs=10000
    )
    
    # Run optimization
    optimizer = PortfolioOptimizer(config)
    optimizer.add_assets(games)
    results = optimizer.optimize()
    
    # Display results
    best = results.best_allocation
    print(f"\n🏆 BEST STRATEGY: {best.strategy.value}")
    print(f"Expected Return: ${best.expected_return:+.2f} ({best.expected_return/50*100:+.1f}%)")
    print(f"Sharpe Ratio: {best.sharpe_ratio:.4f}")
    print(f"Win Probability: {best.win_probability:.1%}")
    print(f"Max Concentration: {best.max_concentration:.1%}")
    
    print(f"\n💰 ALLOCATION:")
    for asset, amount in best.allocation.items():
        if amount > 0:
            pct = amount / 50 * 100
            game_name = next(g.name for g in games if g.symbol == asset)
            print(f"  {game_name:<40} ${amount:6.2f} ({pct:5.1f}%)")
    print(f"  {'TOTAL':<40} ${best.total_invested:6.2f} (100.0%)")


def demonstrate_strategy_comparison():
    """Demonstrate comparison of different allocation strategies."""
    print("\n📊 STRATEGY COMPARISON")
    print("=" * 60)
    
    # Create assets and optimizer
    games = create_september_5_games()
    config = OptimizationConfig(
        total_budget=50,
        max_concentration=0.30,
        strategies=['kelly', 'equal_weight', 'risk_parity', 'constrained_kelly'],
        monte_carlo_runs=5000  # Reduced for faster comparison
    )
    
    optimizer = PortfolioOptimizer(config)
    optimizer.add_assets(games)
    
    # Compare strategies on best 3 assets
    best_3_assets = optimizer.select_best_assets(3)
    print(f"\nSelected Assets: {', '.join(best_3_assets)}")
    
    comparison_df = optimizer.compare_strategies(best_3_assets)
    print(f"\nStrategy Performance Comparison:")
    print(comparison_df.round(4).to_string(index=False))


def demonstrate_risk_analysis():
    """Demonstrate risk analysis and concentration limits."""
    print("\n🛡️ RISK ANALYSIS")
    print("=" * 60)
    
    games = create_september_5_games()
    
    # Test different concentration limits
    concentration_limits = [0.20, 0.30, 0.50, 1.00]  # 20%, 30%, 50%, unlimited
    
    print(f"{'Concentration Limit':<20} {'Expected Return':<15} {'Sharpe Ratio':<12} {'Max Conc':<10}")
    print("-" * 60)
    
    for limit in concentration_limits:
        config = OptimizationConfig(
            total_budget=50,
            max_concentration=limit,
            strategies=['constrained_kelly'],
            monte_carlo_runs=2000
        )
        
        optimizer = PortfolioOptimizer(config)
        optimizer.add_assets(games)
        results = optimizer.optimize()
        
        if results.results:
            best = results.best_allocation
            limit_str = f"{limit:.0%}" if limit < 1 else "Unlimited"
            print(f"{limit_str:<20} ${best.expected_return:+8.2f}     {best.sharpe_ratio:8.4f}    {best.max_concentration:6.1%}")


def demonstrate_asset_selection():
    """Demonstrate asset selection optimization."""
    print("\n🎯 ASSET SELECTION OPTIMIZATION")
    print("=" * 60)
    
    games = create_september_5_games()
    
    # Test different numbers of assets
    asset_counts = [3, 4, 5]
    
    print(f"{'Assets Used':<12} {'Strategy':<20} {'Expected Return':<15} {'Sharpe Ratio':<12}")
    print("-" * 65)
    
    for n_assets in asset_counts:
        config = OptimizationConfig(
            total_budget=50,
            min_assets=n_assets,
            max_assets=n_assets,
            strategies=['equal_weight', 'constrained_kelly'],
            monte_carlo_runs=2000
        )
        
        optimizer = PortfolioOptimizer(config)
        optimizer.add_assets(games)
        results = optimizer.optimize()
        
        # Show best result for this asset count
        if results.results:
            best = results.best_allocation
            strategy_name = best.strategy.value.replace('_', ' ').title()
            print(f"{n_assets:<12} {strategy_name:<20} ${best.expected_return:+8.2f}     {best.sharpe_ratio:8.4f}")


def demonstrate_profit_analysis():
    """Demonstrate profit potential analysis."""
    print("\n💡 PROFIT POTENTIAL ANALYSIS")
    print("=" * 60)
    
    games = create_september_5_games()
    
    print(f"Individual Asset Analysis:")
    print(f"{'Asset':<20} {'Price':<8} {'Profit/Loss on $10':<18} {'Expected Value':<15}")
    print("-" * 65)
    
    for game in games:
        profit_if_win = (1/game.favorite_price - 1) * 10
        loss_if_lose = -10
        expected_value = game.implied_probability * profit_if_win + (1 - game.implied_probability) * loss_if_lose
        
        print(f"{game.symbol:<20} ${game.favorite_price:.2f}    "
              f"+${profit_if_win:6.2f} / ${loss_if_lose:6.2f}   "
              f"${expected_value:+8.2f}")
    
    print(f"\nKey Insights:")
    print(f"• Boise State and Northwestern offer minimal profit potential (<$0.20 on $10)")
    print(f"• Chiefs offer highest profit potential but with higher risk")
    print(f"• Maryland and Louisville provide balanced risk/reward profiles")


def main():
    """Run all portfolio optimization demonstrations."""
    print("Neural SDK Portfolio Optimization Demo")
    print("September 5, 2025 College Football & NFL Games")
    print("=" * 80)
    
    # Run demonstrations
    demonstrate_basic_optimization()
    demonstrate_strategy_comparison()
    demonstrate_risk_analysis()
    demonstrate_asset_selection()
    demonstrate_profit_analysis()
    
    print("\n" + "=" * 80)
    print("🎉 PORTFOLIO OPTIMIZATION COMPLETE")
    print("\nKey Takeaways:")
    print("1. Focus on games with reasonable profit margins (avoid >$0.95 prices)")
    print("2. Diversify across 3-4 assets for optimal risk-adjusted returns")
    print("3. Use concentration limits to manage single-asset risk")
    print("4. Consider Kelly Criterion for growth optimization")
    print("5. Equal weight provides good baseline diversification")
    
    print(f"\n💡 For your $50 investment, the optimal strategy is:")
    print(f"   Focus on 3 profitable games with equal or constrained Kelly allocation")
    print(f"   Avoid high-price, low-return favorites like Boise State and Northwestern")


if __name__ == "__main__":
    main()