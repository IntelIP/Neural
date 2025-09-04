"""
Example: ESPN Play-by-Play to Kalshi Market Correlation
Demonstrates real-time correlation between game events and market movements
"""

import asyncio
import logging

from data_pipeline.data_sources.espn.models import EventType
from data_pipeline.unified_stream import MarketCorrelation, UnifiedStreamManager
from data_pipeline.utils import setup_logging


async def track_game_with_markets():
    """Example of tracking a specific game with its Kalshi markets"""
    setup_logging(level="INFO")
    logger = logging.getLogger(__name__)

    manager = UnifiedStreamManager()

    # Track correlations
    correlations_found = []

    async def handle_correlation(correlation: MarketCorrelation):
        """Handle detected correlations"""
        correlations_found.append(correlation)

        logger.info("=" * 60)
        logger.info("🎯 MARKET CORRELATION DETECTED!")
        logger.info(f"Game Event: {correlation.event.type.value}")
        logger.info(f"Description: {correlation.event.description}")
        logger.info(f"Impact Score: {correlation.event.impact_score:.2f}")
        logger.info(f"Market: {correlation.market_ticker}")
        logger.info(
            f"Price Movement: {correlation.price_before:.3f} → {correlation.price_after:.3f}"
        )
        logger.info(f"Change: {correlation.price_change_pct:+.2f}%")
        logger.info(f"Latency: {correlation.latency_ms}ms")
        logger.info("=" * 60)

    async def handle_game_event(event):
        """Handle game events"""
        if event.impact_score > 0.6:
            logger.info(
                f"⚡ High Impact Event: {event.description} (Score: {event.impact_score:.2f})"
            )

    async def handle_market_update(update):
        """Handle market updates"""
        if abs(update["change"]) > 0.02:  # Log significant changes
            logger.info(
                f"📊 Market Move: {update['ticker']} {update['change']:+.3f} to {update['price']:.3f}"
            )

    # Set callbacks
    manager.on_correlation = handle_correlation
    manager.on_game_event = handle_game_event
    manager.on_market_update = handle_market_update

    try:
        await manager.start()

        # Example: Track an NFL game with winner and total points markets
        # You would replace these with actual game ID and market tickers
        game_id = "401547435"  # Example game ID
        market_tickers = [
            "NFL-2024-WEEK1-HOMETEAM-WIN",  # Home team win market
            "NFL-2024-WEEK1-TOTAL-OVER-45",  # Total points over/under
        ]

        logger.info(f"Tracking game {game_id} with markets: {market_tickers}")
        await manager.track_game(game_id, market_tickers, "nfl")

        # Run for 2 minutes
        await asyncio.sleep(120)

        # Print correlation summary
        if correlations_found:
            logger.info("\n" + "=" * 60)
            logger.info("CORRELATION SUMMARY")
            logger.info("=" * 60)

            for corr in correlations_found:
                logger.info(
                    f"• {corr.event.type.value}: {corr.price_change_pct:+.2f}% on {corr.market_ticker}"
                )

            # Calculate average impact by event type
            by_type = {}
            for corr in correlations_found:
                event_type = corr.event.type
                if event_type not in by_type:
                    by_type[event_type] = []
                by_type[event_type].append(abs(corr.price_change_pct))

            logger.info("\nAverage Price Impact by Event Type:")
            for event_type, changes in by_type.items():
                avg_change = sum(changes) / len(changes)
                logger.info(f"  {event_type.value}: {avg_change:.2f}%")

    finally:
        await manager.stop()


async def monitor_all_nfl_games():
    """Example of monitoring all NFL games for patterns"""
    setup_logging(level="INFO")
    logger = logging.getLogger(__name__)

    manager = UnifiedStreamManager()

    # Pattern tracking
    touchdown_impacts = []
    turnover_impacts = []

    async def handle_correlation(correlation: MarketCorrelation):
        """Track patterns in correlations"""
        if correlation.event.type == EventType.TOUCHDOWN:
            touchdown_impacts.append(abs(correlation.price_change_pct))
        elif correlation.event.type in [EventType.INTERCEPTION, EventType.FUMBLE]:
            turnover_impacts.append(abs(correlation.price_change_pct))

        # Log significant correlations
        if abs(correlation.price_change_pct) > 3:  # 3% threshold
            logger.info(
                f"💥 Significant Impact: {correlation.event.type.value} caused {correlation.price_change_pct:+.2f}% move in {correlation.market_ticker}"
            )

    manager.on_correlation = handle_correlation

    try:
        await manager.start()

        # Monitor all NFL games
        logger.info("Monitoring all NFL games for market patterns...")
        await manager.track_sport("nfl")

        # For demo, we'll also track some example markets
        # In production, you'd dynamically subscribe based on live games
        example_markets = ["NFL-GENERIC-HOME-WIN", "NFL-GENERIC-TOTAL-OVER"]
        await manager.kalshi_ws.subscribe_markets(example_markets)

        # Run for 3 minutes
        await asyncio.sleep(180)

        # Print pattern analysis
        logger.info("\n" + "=" * 60)
        logger.info("PATTERN ANALYSIS")
        logger.info("=" * 60)

        if touchdown_impacts:
            avg_td_impact = sum(touchdown_impacts) / len(touchdown_impacts)
            logger.info(
                f"Touchdowns: {len(touchdown_impacts)} events, avg impact {avg_td_impact:.2f}%"
            )

        if turnover_impacts:
            avg_to_impact = sum(turnover_impacts) / len(turnover_impacts)
            logger.info(
                f"Turnovers: {len(turnover_impacts)} events, avg impact {avg_to_impact:.2f}%"
            )

        # Get overall statistics
        stats = manager.get_correlation_stats()
        logger.info("\nOverall Statistics:")
        logger.info(f"  Total Correlations: {stats['total_correlations']}")
        logger.info(f"  Average Price Change: {stats['avg_price_change']:.3f}")
        logger.info(f"  Average Latency: {stats['avg_latency_ms']}ms")

        if stats["strongest_correlation"]:
            strongest = stats["strongest_correlation"]
            logger.info("\nStrongest Correlation:")
            logger.info(f"  Event: {strongest.event.description}")
            logger.info(f"  Impact: {strongest.price_change_pct:+.2f}%")

    finally:
        await manager.stop()


async def simulate_correlation_scenarios():
    """Simulate different correlation scenarios for testing"""
    setup_logging(level="INFO")
    logger = logging.getLogger(__name__)

    from data_pipeline.data_sources.espn.models import GameEvent, GameState

    # Create a mock game state
    game_state = GameState(
        game_id="TEST123",
        home_team="Home Team",
        away_team="Away Team",
        home_score=14,
        away_score=10,
        quarter=3,
        clock="7:30",
        home_win_probability=65.0,
        away_win_probability=35.0,
    )

    # Simulate different events and their typical market impacts
    scenarios = [
        {
            "event": GameEvent(
                type=EventType.TOUCHDOWN,
                game_id="TEST123",
                description="Home team scores touchdown!",
                impact_score=0.9,
                game_state=game_state,
            ),
            "expected_impact": 5.0,  # 5% price movement
        },
        {
            "event": GameEvent(
                type=EventType.INTERCEPTION,
                game_id="TEST123",
                description="Away team throws interception in red zone!",
                impact_score=0.85,
                game_state=game_state,
            ),
            "expected_impact": -4.5,  # -4.5% price movement
        },
        {
            "event": GameEvent(
                type=EventType.INJURY,
                game_id="TEST123",
                description="Star quarterback injured",
                impact_score=0.8,
                game_state=game_state,
            ),
            "expected_impact": -6.0,  # -6% price movement
        },
        {
            "event": GameEvent(
                type=EventType.FIELD_GOAL_MISSED,
                game_id="TEST123",
                description="Home team misses 35-yard field goal",
                impact_score=0.5,
                game_state=game_state,
            ),
            "expected_impact": -2.0,  # -2% price movement
        },
    ]

    logger.info("=" * 60)
    logger.info("CORRELATION SCENARIO SIMULATION")
    logger.info("=" * 60)

    for scenario in scenarios:
        event = scenario["event"]
        expected = scenario["expected_impact"]

        logger.info(f"\nScenario: {event.type.value}")
        logger.info(f"  Description: {event.description}")
        logger.info(f"  Impact Score: {event.impact_score:.2f}")
        logger.info(f"  Expected Market Impact: {expected:+.1f}%")

        # Simulate market reaction
        if event.game_state.is_critical_situation():
            actual_impact = expected * 1.5  # Amplified in critical situations
            logger.info("  🔥 Critical Situation Multiplier: 1.5x")
        else:
            actual_impact = expected

        logger.info(f"  Simulated Actual Impact: {actual_impact:+.1f}%")

        # Determine trading opportunity
        if abs(actual_impact) > 4:
            logger.info("  💰 TRADING OPPORTUNITY: Strong correlation detected!")
        elif abs(actual_impact) > 2:
            logger.info("  📊 Moderate correlation - monitor closely")
        else:
            logger.info("  📉 Weak correlation - no immediate action")

    logger.info("\n" + "=" * 60)
    logger.info("Key Insights:")
    logger.info("• Touchdowns typically move markets 3-7%")
    logger.info("• Turnovers have 2-5% impact depending on field position")
    logger.info("• Injuries to key players can cause 4-8% swings")
    logger.info("• Critical situations amplify all impacts by 1.5-2x")
    logger.info("• Market reaction typically occurs within 2-5 seconds")


async def main():
    """Run correlation examples"""
    print("=" * 60)
    print("ESPN to Kalshi Market Correlation Examples")
    print("=" * 60)

    # Choose which example to run
    print("\nSelect an example to run:")
    print("1. Track specific game with markets")
    print("2. Monitor all NFL games for patterns")
    print("3. Simulate correlation scenarios")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == "1":
        await track_game_with_markets()
    elif choice == "2":
        await monitor_all_nfl_games()
    elif choice == "3":
        await simulate_correlation_scenarios()
    else:
        print("Running simulation example...")
        await simulate_correlation_scenarios()


if __name__ == "__main__":
    asyncio.run(main())
