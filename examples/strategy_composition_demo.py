"""
Strategy Composition Demo

This example demonstrates how to build sophisticated multi-strategy trading
systems using the Neural SDK's strategy framework. It showcases:

1. Individual strategy configuration
2. Multi-strategy composition
3. Advanced signal processing
4. Risk management integration
5. Performance monitoring

This is an educational example that shows the framework's power while
keeping the actual trading logic simple and demonstrative.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

# Neural SDK imports
from neural.strategy.library import (
    BasicMeanReversionStrategy,
    VolumeAnomalyStrategy, 
    SimpleArbitrageStrategy,
    LineMovementStrategy,
    NewsReactionFramework
)
from neural.strategy.builder import StrategyComposer, StrategyConfig, AggregationMethod, AllocationMethod
from neural.strategy.signals import SignalProcessor, SignalFilter, DecayFunction
from neural.analysis.base import SignalStrength

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_multi_strategy_system():
    """
    Create a sophisticated multi-strategy trading system.
    
    This demonstrates the full power of the strategy framework
    while keeping individual strategies simple and educational.
    """
    logger.info("🚀 Creating Multi-Strategy Trading System")
    
    # Step 1: Initialize individual strategies with custom parameters
    logger.info("📚 Initializing Individual Strategies")
    
    # Mean reversion strategy (conservative)
    mean_reversion = BasicMeanReversionStrategy(
        divergence_threshold=0.04,  # 4% divergence required
        min_confidence=0.65
    )
    
    # Volume anomaly strategy (aggressive)
    volume_strategy = VolumeAnomalyStrategy(
        volume_threshold=2.5,  # 2.5 standard deviations
        lookback_hours=12,
        min_confidence=0.60
    )
    
    # Simple arbitrage strategy (high precision)
    arbitrage_strategy = SimpleArbitrageStrategy(
        min_arbitrage_pct=0.025,  # 2.5% minimum profit
        transaction_cost=0.015,   # 1.5% transaction costs
        max_execution_time=180    # 3 minutes max
    )
    
    # Line movement strategy (momentum)
    momentum_strategy = LineMovementStrategy(
        movement_threshold=0.05,  # 5% price movement
        time_window=4,           # 4-hour windows
        volume_confirmation=True
    )
    
    # News reaction framework (event-driven)
    news_strategy = NewsReactionFramework(
        reaction_window=20,      # 20-minute reaction window
        min_event_score=0.65
    )
    
    # Step 2: Create strategy configurations with different weights/allocations
    logger.info("⚖️ Configuring Strategy Weights and Allocations")
    
    strategy_configs = [
        StrategyConfig(
            strategy=mean_reversion,
            weight=0.3,              # 30% weight in decisions
            max_allocation=0.15,     # Max 15% of capital
            min_confidence=0.65
        ),
        StrategyConfig(
            strategy=volume_strategy,
            weight=0.25,             # 25% weight
            max_allocation=0.12,     # Max 12% of capital  
            min_confidence=0.60
        ),
        StrategyConfig(
            strategy=arbitrage_strategy,
            weight=0.20,             # 20% weight
            max_allocation=0.20,     # Higher allocation for arbitrage
            min_confidence=0.75      # Higher confidence required
        ),
        StrategyConfig(
            strategy=momentum_strategy,
            weight=0.15,             # 15% weight
            max_allocation=0.10,     # Conservative allocation
            min_confidence=0.55
        ),
        StrategyConfig(
            strategy=news_strategy,
            weight=0.10,             # 10% weight
            max_allocation=0.08,     # Small allocation for news
            min_confidence=0.70
        )
    ]
    
    # Step 3: Create strategy composer with sophisticated aggregation
    logger.info("🎼 Creating Strategy Composer")
    
    composer = StrategyComposer(
        aggregation_method=AggregationMethod.WEIGHTED_AVERAGE,
        allocation_method=AllocationMethod.CONFIDENCE_WEIGHTED,
        max_total_allocation=0.25,    # Max 25% total capital deployed
        consensus_threshold=0.65,     # For consensus methods
        rebalance_frequency=7         # Weekly rebalancing
    )
    
    # Add all strategies to composer
    for config in strategy_configs:
        composer.add_strategy(config)
    
    # Step 4: Initialize advanced signal processing
    logger.info("🔧 Setting Up Advanced Signal Processing")
    
    signal_processor = SignalProcessor(
        decay_function=DecayFunction.EXPONENTIAL,
        decay_half_life=25,          # 25-minute half-life
        min_confidence=0.55,         # Minimum processed confidence
        max_signal_age=90,           # 90-minute max age
        enable_temporal_filtering=True
    )
    
    # Configure signal filters
    signal_processor.add_filter(
        SignalFilter.CONFIDENCE_THRESHOLD,
        threshold=0.60
    )
    
    signal_processor.add_filter(
        SignalFilter.STRENGTH_MINIMUM,
        min_strength=SignalStrength.WEAK
    )
    
    signal_processor.add_filter(
        SignalFilter.TEMPORAL_CONSISTENCY,
        lookback_minutes=45,
        min_consistency=0.55
    )
    
    logger.info("✅ Multi-Strategy System Ready!")
    return composer, signal_processor


async def demo_strategy_analysis():
    """
    Demo the multi-strategy system with sample market data.
    """
    logger.info("📊 Running Strategy Analysis Demo")
    
    # Create the system
    composer, signal_processor = await create_multi_strategy_system()
    
    # Sample market data (educational example)
    sample_markets = {
        "NFL_CHIEFS_WIN": {
            'current_price': 0.45,
            'sportsbook_consensus': 0.52,      # 7-cent divergence
            'volume_24h': 8500,                # High volume
            'volume_history': [3000, 3200, 2800, 3100, 8500],  # Volume spike
            'spread': 0.02,
            'sportsbook_prices': {             # For arbitrage
                'pinnacle': 0.48,
                'bet365': 0.50,
                'draftkings': 0.49
            },
            'price_history': [              # For line movement
                {'timestamp': datetime.now().timestamp() - 3600, 'price': 0.52},
                {'timestamp': datetime.now().timestamp() - 1800, 'price': 0.48},
                {'timestamp': datetime.now().timestamp() - 900, 'price': 0.45}
            ],
            'hours_to_close': 18,
            'recent_events': [              # For news strategy
                {
                    'text': 'Chiefs star player confirmed healthy for upcoming game',
                    'timestamp': datetime.now() - timedelta(minutes=30),
                    'source': 'ESPN'
                }
            ],
            'market_context': {
                'teams': ['Kansas City Chiefs', 'Buffalo Bills'],
                'sport': 'NFL'
            }
        },
        
        "NBA_LAKERS_SPREAD": {
            'current_price': 0.65,
            'sportsbook_consensus': 0.60,      # Smaller divergence
            'volume_24h': 2200,                # Lower volume
            'volume_history': [2000, 2100, 2200, 2300, 2200],  # Stable volume
            'spread': 0.03,
            'sportsbook_prices': {
                'pinnacle': 0.62,
                'bet365': 0.61
            },
            'price_history': [
                {'timestamp': datetime.now().timestamp() - 3600, 'price': 0.60},
                {'timestamp': datetime.now().timestamp() - 1800, 'price': 0.63},
                {'timestamp': datetime.now().timestamp() - 900, 'price': 0.65}
            ],
            'hours_to_close': 6,
            'recent_events': [],
            'market_context': {
                'teams': ['Los Angeles Lakers', 'Boston Celtics'],
                'sport': 'NBA'
            }
        }
    }
    
    # Analyze each market
    for market_id, market_data in sample_markets.items():
        logger.info(f"\n🎯 Analyzing Market: {market_id}")
        
        # Get composite signal from all strategies
        composite_signal = await composer.analyze(market_id, market_data)
        
        if composite_signal:
            logger.info(f"🎯 Composite Signal Generated!")
            logger.info(f"   Action: {composite_signal.action}")
            logger.info(f"   Confidence: {composite_signal.confidence:.2f}")
            logger.info(f"   Signal Strength: {composite_signal.signal_strength.value}")
            logger.info(f"   Position Size: {composite_signal.position_size:.1%}")
            logger.info(f"   Contributing Strategies: {len(composite_signal.contributing_strategies)}")
            logger.info(f"   Reasoning: {composite_signal.reasoning}")
            
            # Process through signal processor
            if composite_signal.individual_signals:
                processed_signals = []
                for signal in composite_signal.individual_signals:
                    processed = signal_processor.process_signal(signal)
                    if processed:
                        processed_signals.append(processed)
                
                if processed_signals:
                    logger.info(f"🔧 Signal Processing Results:")
                    logger.info(f"   Signals Passed Filters: {len(processed_signals)}/{len(composite_signal.individual_signals)}")
                    avg_decay = sum(p.decay_factor for p in processed_signals) / len(processed_signals)
                    logger.info(f"   Average Decay Factor: {avg_decay:.2f}")
            
            # Get consensus metrics
            consensus = signal_processor.get_signal_consensus(market_id, 60)
            logger.info(f"📊 Consensus Metrics:")
            logger.info(f"   Consensus Score: {consensus['consensus_score']:.2f}")
            logger.info(f"   Signal Count: {consensus['signal_count']}")
            logger.info(f"   Dominant Action: {consensus.get('dominant_action', 'N/A')}")
            
        else:
            logger.info(f"❌ No composite signal generated for {market_id}")
    
    # Show system summary
    logger.info(f"\n📈 System Summary:")
    summary = composer.get_strategy_summary()
    logger.info(f"   Total Strategies: {summary['total_strategies']}")
    logger.info(f"   Enabled Strategies: {summary['enabled_strategies']}")
    logger.info(f"   Aggregation Method: {summary['aggregation_method']}")
    logger.info(f"   Max Total Allocation: {summary['max_total_allocation']:.1%}")
    
    processing_stats = signal_processor.get_processing_stats()
    logger.info(f"   Signal Processing Stats:")
    logger.info(f"     Markets Tracked: {processing_stats['total_markets_tracked']}")
    logger.info(f"     Signals Processed: {processing_stats['total_signals_processed']}")
    logger.info(f"     Active Filters: {len(processing_stats['active_filters'])}")


async def demo_dynamic_rebalancing():
    """
    Demo dynamic strategy rebalancing based on performance.
    """
    logger.info("\n⚡ Dynamic Rebalancing Demo")
    
    composer, signal_processor = await create_multi_strategy_system()
    
    # Simulate performance updates
    performance_updates = {
        'BasicMeanReversion': 1.2,    # 20% above average
        'VolumeAnomaly': 0.8,         # 20% below average  
        'SimpleArbitrage': 1.5,       # 50% above average
        'LineMovement': 1.0,          # Average performance
        'NewsReaction': 0.6           # 40% below average
    }
    
    logger.info("📊 Updating Strategy Performance Scores:")
    for strategy_id, performance in performance_updates.items():
        signal_processor.update_strategy_performance(strategy_id, performance)
        logger.info(f"   {strategy_id}: {performance:.1f}")
    
    # Adjust strategy weights based on performance
    logger.info("⚖️ Rebalancing Strategy Weights:")
    weight_adjustments = {
        'BasicMeanReversion': 0.35,   # Increase weight (good performance)
        'VolumeAnomaly': 0.20,        # Decrease weight (poor performance)
        'SimpleArbitrage': 0.30,      # Increase weight (excellent performance)
        'LineMovement': 0.10,         # Decrease weight (average performance)
        'NewsReaction': 0.05          # Decrease weight (poor performance)
    }
    
    for strategy_id, new_weight in weight_adjustments.items():
        success = composer.update_strategy_weight(strategy_id, new_weight)
        if success:
            logger.info(f"   Updated {strategy_id} weight to {new_weight:.1%}")
    
    logger.info("✅ Rebalancing Complete!")


def showcase_framework_capabilities():
    """
    Print a summary of the framework's capabilities.
    """
    logger.info("\n🎯 Neural SDK Strategy Framework Capabilities")
    logger.info("=" * 60)
    
    capabilities = [
        "✅ Educational Strategy Library (5 example strategies)",
        "✅ Multi-Strategy Composition Framework", 
        "✅ Advanced Signal Processing & Filtering",
        "✅ Multiple Aggregation Methods (weighted, consensus, etc.)",
        "✅ Risk-Aware Capital Allocation",
        "✅ Performance-Based Dynamic Rebalancing",
        "✅ Temporal Signal Decay & Validation",
        "✅ Signal Consensus & Confidence Scoring",
        "✅ Modular & Extensible Architecture",
        "✅ Production-Ready Error Handling"
    ]
    
    for capability in capabilities:
        logger.info(f"  {capability}")
    
    logger.info("\n🔒 Protected Competitive Advantages:")
    logger.info("  • Keep sophisticated probability models proprietary")
    logger.info("  • Preserve advanced edge detection algorithms")  
    logger.info("  • Maintain complex multi-factor confidence scoring")
    logger.info("  • Protect real alpha-generating strategies")
    
    logger.info("\n📚 Educational Value:")
    logger.info("  • Demonstrates framework power without revealing secrets")
    logger.info("  • Provides templates for building custom strategies")
    logger.info("  • Shows best practices for system composition")
    logger.info("  • Enables rapid strategy development & testing")


async def main():
    """
    Run the complete strategy composition demo.
    """
    logger.info("🚀 Neural SDK Strategy Framework Demo")
    logger.info("=" * 60)
    
    # Show capabilities overview
    showcase_framework_capabilities()
    
    # Run main demo
    await demo_strategy_analysis()
    
    # Demo dynamic rebalancing
    await demo_dynamic_rebalancing()
    
    logger.info("\n🎉 Demo Complete! The Neural SDK Strategy Framework is ready.")
    logger.info("💡 Use these educational examples as templates for your own strategies.")
    logger.info("🔐 Keep your alpha-generating algorithms proprietary for competitive advantage!")


if __name__ == "__main__":
    asyncio.run(main())
