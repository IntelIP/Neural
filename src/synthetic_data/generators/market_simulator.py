"""
Market Event Simulator

Simulates Kalshi market events and trading scenarios based on 
synthetic game data for agent training.
"""

import logging
import random
from typing import List, Dict, Any, Optional, Tuple, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import math

from .game_engine import SyntheticGame, SyntheticPlay
from src.sdk.core.base_adapter import StandardizedEvent, EventType

logger = logging.getLogger(__name__)


class MarketEventType(Enum):
    """Types of market events"""
    PRICE_MOVEMENT = "price_movement"
    VOLUME_SPIKE = "volume_spike"
    SPREAD_CHANGE = "spread_change"
    NEWS_EVENT = "news_event"
    MOMENTUM_SHIFT = "momentum_shift"
    INJURY_REPORT = "injury_report"
    WEATHER_UPDATE = "weather_update"


@dataclass
class MarketState:
    """Current market state for a contract"""
    market_ticker: str
    yes_price: float = 0.50
    no_price: float = 0.50
    volume: int = 0
    open_interest: int = 1000
    bid_ask_spread: float = 0.02
    last_trade_price: float = 0.50
    price_change_24h: float = 0.0
    
    # Market dynamics
    momentum: float = 0.0  # -1 to 1
    volatility: float = 0.1  # 0 to 1
    liquidity: float = 0.5   # 0 to 1
    
    def __post_init__(self):
        """Ensure prices sum to ~1.00"""
        total = self.yes_price + self.no_price
        if total != 1.0:
            self.yes_price = self.yes_price / total
            self.no_price = self.no_price / total


@dataclass
class MarketEvent:
    """Market event that affects pricing"""
    event_type: MarketEventType
    market_ticker: str
    timestamp: datetime
    description: str
    
    # Price impact
    price_impact: float = 0.0  # -1 to 1 (negative = price down)
    volume_impact: float = 0.0  # 0 to 1
    confidence_impact: float = 0.0  # -1 to 1
    
    # Context
    game_context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradingScenario:
    """Complete trading scenario for agent training"""
    scenario_id: str
    market_ticker: str
    game: SyntheticGame
    
    # Market timeline
    events: List[MarketEvent] = field(default_factory=list)
    price_history: List[Tuple[datetime, float]] = field(default_factory=list)
    volume_history: List[Tuple[datetime, int]] = field(default_factory=list)
    
    # Information asymmetry
    public_events: List[MarketEvent] = field(default_factory=list)
    private_events: List[MarketEvent] = field(default_factory=list)
    delayed_events: List[MarketEvent] = field(default_factory=list)
    
    # Scenario characteristics
    scenario_type: str = "regular"  # regular, volatile, trending, reversal
    information_delay: float = 0.0  # 0 to 1 (fraction of events delayed)
    market_efficiency: float = 0.8  # 0 to 1 (how quickly prices adjust)
    
    created_at: datetime = field(default_factory=datetime.now)


class MarketSimulator:
    """
    Simulates realistic Kalshi market events and trading scenarios
    for comprehensive agent training
    """
    
    def __init__(self):
        """Initialize market simulator"""
        
        # Market configuration
        self.base_liquidity = 1000
        self.min_spread = 0.01
        self.max_spread = 0.10
        self.volatility_decay = 0.95
        
        # Event probabilities (per game)
        self.event_probabilities = {
            MarketEventType.PRICE_MOVEMENT: 1.0,    # Always present
            MarketEventType.VOLUME_SPIKE: 0.3,      # 30% of games
            MarketEventType.SPREAD_CHANGE: 0.4,     # 40% of games
            MarketEventType.NEWS_EVENT: 0.2,        # 20% of games
            MarketEventType.MOMENTUM_SHIFT: 0.6,    # 60% of games
            MarketEventType.INJURY_REPORT: 0.15,    # 15% of games
            MarketEventType.WEATHER_UPDATE: 0.1     # 10% of games
        }
        
        logger.info("Initialized MarketSimulator")
    
    def create_trading_scenario(self, 
                              game: SyntheticGame,
                              scenario_type: str = "regular",
                              information_delay: float = 0.1,
                              market_efficiency: float = 0.8) -> TradingScenario:
        """
        Create complete trading scenario from synthetic game
        
        Args:
            game: Synthetic game to base scenario on
            scenario_type: Type of market scenario
            information_delay: Fraction of events with delayed information
            market_efficiency: Speed of price adjustment (0-1)
            
        Returns:
            Complete trading scenario
        """
        market_ticker = f"NFL-{game.away_team}-{game.home_team}-{game.game_id[-8:]}"
        scenario_id = f"scenario_{game.game_id}_{scenario_type}"
        
        logger.info(f"Creating trading scenario: {market_ticker} ({scenario_type})")
        
        scenario = TradingScenario(
            scenario_id=scenario_id,
            market_ticker=market_ticker,
            game=game,
            scenario_type=scenario_type,
            information_delay=information_delay,
            market_efficiency=market_efficiency
        )
        
        # Initialize market state
        initial_state = self._initialize_market_state(market_ticker, game)
        
        # Generate market events throughout game
        self._generate_market_events(scenario, initial_state)
        
        # Separate events by information type
        self._classify_information_events(scenario)
        
        # Generate price and volume history
        self._simulate_market_timeline(scenario, initial_state)
        
        logger.info(f"Created scenario with {len(scenario.events)} market events")
        return scenario
    
    def _initialize_market_state(self, market_ticker: str, game: SyntheticGame) -> MarketState:
        """Initialize market state based on game characteristics"""
        
        # Determine initial odds based on team strength (simplified)
        home_advantage = 0.55  # Typical home field advantage
        
        # Add random noise for team strength
        team_strength_diff = random.gauss(0, 0.15)  # Random team strength difference
        
        initial_prob = home_advantage + team_strength_diff
        initial_prob = max(0.2, min(0.8, initial_prob))  # Keep within reasonable bounds
        
        # Set initial pricing
        yes_price = initial_prob
        no_price = 1.0 - yes_price
        
        # Adjust for game type
        if game.game_type == "blowout":
            # More extreme pricing for expected blowouts
            if yes_price > 0.5:
                yes_price = min(0.85, yes_price + 0.2)
            else:
                yes_price = max(0.15, yes_price - 0.2)
            no_price = 1.0 - yes_price
        
        # Calculate spread and liquidity
        spread = self._calculate_spread(yes_price, game.game_type)
        liquidity = self._calculate_liquidity(game.game_type)
        
        return MarketState(
            market_ticker=market_ticker,
            yes_price=yes_price,
            no_price=no_price,
            volume=random.randint(500, 2000),
            open_interest=random.randint(800, 5000),
            bid_ask_spread=spread,
            last_trade_price=yes_price,
            momentum=random.gauss(0, 0.1),
            volatility=self._get_base_volatility(game.game_type),
            liquidity=liquidity
        )
    
    def _calculate_spread(self, price: float, game_type: str) -> float:
        """Calculate bid-ask spread based on price and game type"""
        
        # Spreads wider at extremes
        extreme_factor = 4 * price * (1 - price)  # 0 at extremes, 1 at 0.5
        base_spread = self.min_spread + (self.max_spread - self.min_spread) * (1 - extreme_factor)
        
        # Game type modifiers
        type_multipliers = {
            "regular": 1.0,
            "high_scoring": 0.9,    # More liquid
            "defensive": 1.1,       # Less liquid
            "weather": 1.3,         # Less liquid due to uncertainty
            "blowout": 0.8          # More liquid due to clarity
        }
        
        multiplier = type_multipliers.get(game_type, 1.0)
        return max(self.min_spread, base_spread * multiplier)
    
    def _calculate_liquidity(self, game_type: str) -> float:
        """Calculate market liquidity factor"""
        
        liquidity_scores = {
            "regular": 0.7,
            "high_scoring": 0.8,    # High interest
            "defensive": 0.6,       # Lower interest
            "weather": 0.5,         # Uncertainty reduces participation
            "blowout": 0.9          # Clear outcomes attract traders
        }
        
        base_liquidity = liquidity_scores.get(game_type, 0.7)
        return base_liquidity + random.gauss(0, 0.1)
    
    def _get_base_volatility(self, game_type: str) -> float:
        """Get base volatility for game type"""
        
        volatility_scores = {
            "regular": 0.15,
            "high_scoring": 0.25,   # More volatile due to scoring
            "defensive": 0.10,      # Lower volatility
            "weather": 0.30,        # Weather creates uncertainty
            "blowout": 0.20         # Moderate volatility
        }
        
        return volatility_scores.get(game_type, 0.15)
    
    def _generate_market_events(self, scenario: TradingScenario, initial_state: MarketState):
        """Generate market events throughout the game timeline"""
        
        current_state = initial_state
        game_start = datetime.now()
        
        # Pre-game events (2 hours before)
        self._generate_pregame_events(scenario, current_state, game_start - timedelta(hours=2))
        
        # During game events (based on plays)
        self._generate_ingame_events(scenario, current_state, game_start)
        
        # Post-game events
        self._generate_postgame_events(scenario, current_state, game_start + timedelta(hours=3))
    
    def _generate_pregame_events(self, scenario: TradingScenario, state: MarketState, start_time: datetime):
        """Generate pre-game market events"""
        
        events = []
        current_time = start_time
        
        # Injury reports
        if random.random() < self.event_probabilities[MarketEventType.INJURY_REPORT]:
            affected_team = random.choice([scenario.game.home_team, scenario.game.away_team])
            severity = random.choice(["questionable", "doubtful", "out"])
            
            impact = -0.15 if severity == "out" else -0.05
            if affected_team == scenario.game.away_team:
                impact *= -1  # Flip for away team
            
            event = MarketEvent(
                event_type=MarketEventType.INJURY_REPORT,
                market_ticker=state.market_ticker,
                timestamp=current_time + timedelta(minutes=random.randint(15, 90)),
                description=f"Key player from {affected_team} listed as {severity}",
                price_impact=impact,
                volume_impact=0.3,
                confidence_impact=-0.2
            )
            events.append(event)
        
        # Weather updates
        if random.random() < self.event_probabilities[MarketEventType.WEATHER_UPDATE]:
            weather_conditions = random.choice(["rain", "wind", "snow", "cold"])
            
            event = MarketEvent(
                event_type=MarketEventType.WEATHER_UPDATE,
                market_ticker=state.market_ticker,
                timestamp=current_time + timedelta(minutes=random.randint(30, 120)),
                description=f"Weather update: {weather_conditions} conditions expected",
                price_impact=random.gauss(0, 0.05),
                volume_impact=0.2,
                confidence_impact=-0.1,
                metadata={"weather": weather_conditions}
            )
            events.append(event)
        
        # News events
        if random.random() < self.event_probabilities[MarketEventType.NEWS_EVENT]:
            news_types = ["coaching_decision", "team_news", "analyst_prediction", "betting_trends"]
            news_type = random.choice(news_types)
            
            event = MarketEvent(
                event_type=MarketEventType.NEWS_EVENT,
                market_ticker=state.market_ticker,
                timestamp=current_time + timedelta(minutes=random.randint(45, 120)),
                description=f"News: {news_type.replace('_', ' ')}",
                price_impact=random.gauss(0, 0.08),
                volume_impact=0.15,
                confidence_impact=random.gauss(0, 0.05)
            )
            events.append(event)
        
        scenario.events.extend(events)
    
    def _generate_ingame_events(self, scenario: TradingScenario, state: MarketState, game_start: datetime):
        """Generate market events based on game plays"""
        
        current_time = game_start
        last_momentum = 0.0
        
        for i, play in enumerate(scenario.game.plays):
            # Advance time
            current_time += timedelta(seconds=30)  # Average time per play
            
            # Always generate price movement for significant plays
            if self._is_significant_play(play):
                price_impact = self._calculate_play_price_impact(play, scenario.game)
                
                event = MarketEvent(
                    event_type=MarketEventType.PRICE_MOVEMENT,
                    market_ticker=state.market_ticker,
                    timestamp=current_time,
                    description=f"Play update: {play.play_description}",
                    price_impact=price_impact,
                    volume_impact=min(0.5, abs(price_impact) * 2),
                    game_context={
                        "quarter": play.context.quarter,
                        "score_home": play.context.score_home,
                        "score_away": play.context.score_away,
                        "play_type": play.play_type
                    }
                )
                scenario.events.append(event)
            
            # Volume spikes on exciting plays
            if play.excitement_factor > 0.8 and random.random() < 0.5:
                event = MarketEvent(
                    event_type=MarketEventType.VOLUME_SPIKE,
                    market_ticker=state.market_ticker,
                    timestamp=current_time + timedelta(seconds=random.randint(5, 30)),
                    description=f"High volume trading after {play.play_type}",
                    volume_impact=0.8,
                    price_impact=0.0
                )
                scenario.events.append(event)
            
            # Momentum shifts
            current_momentum = self._calculate_momentum(play, last_momentum)
            if abs(current_momentum - last_momentum) > 0.3:
                event = MarketEvent(
                    event_type=MarketEventType.MOMENTUM_SHIFT,
                    market_ticker=state.market_ticker,
                    timestamp=current_time + timedelta(seconds=random.randint(10, 60)),
                    description=f"Momentum shift detected",
                    price_impact=current_momentum * 0.1,
                    volume_impact=0.3,
                    confidence_impact=0.1
                )
                scenario.events.append(event)
            
            last_momentum = current_momentum
        
        # Add quarter-end events
        for quarter in range(1, 5):
            if any(play.context.quarter == quarter for play in scenario.game.plays):
                quarter_time = current_time + timedelta(minutes=quarter * 30)
                
                event = MarketEvent(
                    event_type=MarketEventType.PRICE_MOVEMENT,
                    market_ticker=state.market_ticker,
                    timestamp=quarter_time,
                    description=f"End of quarter {quarter} update",
                    price_impact=random.gauss(0, 0.02),
                    volume_impact=0.1
                )
                scenario.events.append(event)
    
    def _generate_postgame_events(self, scenario: TradingScenario, state: MarketState, end_time: datetime):
        """Generate post-game market events"""
        
        # Final settlement
        final_event = MarketEvent(
            event_type=MarketEventType.PRICE_MOVEMENT,
            market_ticker=state.market_ticker,
            timestamp=end_time,
            description=f"Game final: {scenario.game.away_team} {scenario.game.final_score[1]} - {scenario.game.final_score[0]} {scenario.game.home_team}",
            price_impact=1.0 if scenario.game.final_score[0] > scenario.game.final_score[1] else -1.0,  # Complete resolution
            volume_impact=0.9
        )
        scenario.events.append(final_event)
    
    def _is_significant_play(self, play: SyntheticPlay) -> bool:
        """Check if play is significant enough to generate market event"""
        
        # Always significant
        if play.touchdown or play.field_goal or play.turnover or play.safety:
            return True
        
        # Big yardage plays
        if abs(play.yards_gained) >= 20:
            return True
        
        # Fourth quarter plays
        if play.context.quarter >= 4:
            return True
        
        # Red zone plays
        if play.context.yards_to_goal <= 20:
            return True
        
        # High leverage situations
        if play.context.down >= 3 and play.context.distance >= 7:
            return True
        
        return False
    
    def _calculate_play_price_impact(self, play: SyntheticPlay, game: SyntheticGame) -> float:
        """Calculate market price impact of a play"""
        
        base_impact = 0.0
        
        # Scoring plays
        if play.touchdown:
            base_impact = 0.20
        elif play.field_goal:
            base_impact = 0.10
        elif play.safety:
            base_impact = 0.15
        
        # Turnovers
        elif play.turnover:
            base_impact = 0.15
        
        # Big plays
        elif abs(play.yards_gained) >= 30:
            base_impact = 0.08
        elif abs(play.yards_gained) >= 20:
            base_impact = 0.05
        elif abs(play.yards_gained) >= 10:
            base_impact = 0.02
        
        # Adjust for possession team
        if play.context.possession_team == game.away_team:
            base_impact *= -1  # Away team positive impact = negative price impact
        
        # Time pressure multiplier
        if play.context.quarter >= 4:
            time_factor = 1 + (4 - play.context.quarter) * 0.5
            base_impact *= time_factor
        
        # Game situation multiplier  
        score_diff = abs(play.context.score_differential)
        if score_diff <= 3:
            base_impact *= 1.5  # Close games have higher impact
        elif score_diff <= 7:
            base_impact *= 1.2
        elif score_diff >= 21:
            base_impact *= 0.5  # Blowouts have lower impact
        
        # Add random noise
        noise = random.gauss(0, 0.02)
        return base_impact + noise
    
    def _calculate_momentum(self, play: SyntheticPlay, last_momentum: float) -> float:
        """Calculate current momentum factor"""
        
        play_momentum = 0.0
        
        # Big positive plays increase momentum
        if play.touchdown:
            play_momentum = 0.8
        elif play.field_goal:
            play_momentum = 0.4
        elif play.yards_gained >= 20:
            play_momentum = 0.3
        elif play.yards_gained >= 10:
            play_momentum = 0.1
        
        # Negative plays decrease momentum
        elif play.turnover:
            play_momentum = -0.8
        elif play.yards_gained <= -5:
            play_momentum = -0.3
        elif play.yards_gained <= 0:
            play_momentum = -0.1
        
        # Adjust for possession team (home = positive momentum)
        if play.context.possession_team != play.context.home_team:
            play_momentum *= -1
        
        # Momentum decay and update
        decayed_momentum = last_momentum * 0.9
        new_momentum = decayed_momentum + play_momentum * 0.3
        
        return max(-1.0, min(1.0, new_momentum))
    
    def _classify_information_events(self, scenario: TradingScenario):
        """Separate events into public, private, and delayed categories"""
        
        for event in scenario.events:
            # Determine information classification
            if random.random() < scenario.information_delay:
                # Delayed information
                scenario.delayed_events.append(event)
            elif event.event_type in [MarketEventType.NEWS_EVENT, MarketEventType.INJURY_REPORT]:
                # Some news might be private initially
                if random.random() < 0.3:
                    scenario.private_events.append(event)
                else:
                    scenario.public_events.append(event)
            else:
                # Most game events are public
                scenario.public_events.append(event)
    
    def _simulate_market_timeline(self, scenario: TradingScenario, initial_state: MarketState):
        """Simulate price and volume over time"""
        
        current_state = initial_state
        timeline_events = sorted(scenario.events, key=lambda x: x.timestamp)
        
        for event in timeline_events:
            # Apply event to market state
            self._apply_event_to_state(current_state, event, scenario)
            
            # Record price point
            scenario.price_history.append((event.timestamp, current_state.yes_price))
            scenario.volume_history.append((event.timestamp, current_state.volume))
        
        logger.debug(f"Simulated {len(scenario.price_history)} price points for {scenario.market_ticker}")
    
    def _apply_event_to_state(self, state: MarketState, event: MarketEvent, scenario: TradingScenario):
        """Apply market event to current state"""
        
        # Price impact
        if event.price_impact != 0:
            # Adjust price based on impact and market efficiency
            price_change = event.price_impact * scenario.market_efficiency
            
            # Apply to yes price
            new_yes_price = state.yes_price + price_change
            new_yes_price = max(0.01, min(0.99, new_yes_price))
            
            state.yes_price = new_yes_price
            state.no_price = 1.0 - new_yes_price
            state.last_trade_price = new_yes_price
            
            # Update price change
            state.price_change_24h += price_change
        
        # Volume impact
        if event.volume_impact > 0:
            volume_multiplier = 1.0 + event.volume_impact
            additional_volume = int(state.volume * volume_multiplier * 0.1)
            state.volume += additional_volume
        
        # Update momentum and volatility
        if hasattr(event, 'confidence_impact'):
            state.momentum += event.confidence_impact * 0.1
            state.momentum = max(-1.0, min(1.0, state.momentum))
        
        # Volatility increases with price movements
        if abs(event.price_impact) > 0.05:
            state.volatility = min(1.0, state.volatility * 1.1)
        else:
            state.volatility *= self.volatility_decay
    
    def convert_to_standardized_events(self, scenario: TradingScenario) -> List[StandardizedEvent]:
        """Convert market scenario to StandardizedEvent format"""
        
        events = []
        
        for market_event in scenario.events:
            # Determine impact level
            if abs(market_event.price_impact) >= 0.15:
                impact = "high"
            elif abs(market_event.price_impact) >= 0.05:
                impact = "medium"
            else:
                impact = "low"
            
            # Build event data
            event_data = {
                "market_ticker": market_event.market_ticker,
                "event_type": market_event.event_type.value,
                "description": market_event.description,
                "price_impact": market_event.price_impact,
                "volume_impact": market_event.volume_impact,
                "confidence_impact": market_event.confidence_impact,
                "game_context": market_event.game_context,
                "synthetic": True
            }
            
            event = StandardizedEvent(
                source="synthetic_market_simulator",
                event_type=EventType.MARKET_EVENT,
                timestamp=market_event.timestamp,
                game_id=scenario.game.game_id,
                data=event_data,
                confidence=0.99,
                impact=impact,
                metadata={
                    "scenario_id": scenario.scenario_id,
                    "scenario_type": scenario.scenario_type,
                    "information_delay": scenario.information_delay,
                    "market_efficiency": scenario.market_efficiency,
                    "synthetic": True
                },
                raw_data=market_event
            )
            
            events.append(event)
        
        logger.info(f"Converted {len(events)} market events to StandardizedEvents")
        return events
    
    def generate_batch_scenarios(self, 
                                games: List[SyntheticGame],
                                scenario_types: List[str] = None,
                                information_delays: List[float] = None) -> List[TradingScenario]:
        """
        Generate multiple trading scenarios from games
        
        Args:
            games: List of synthetic games
            scenario_types: Types of scenarios to generate
            information_delays: Different information delay settings
            
        Returns:
            List of trading scenarios
        """
        if scenario_types is None:
            scenario_types = ["regular", "volatile", "trending", "reversal"]
        
        if information_delays is None:
            information_delays = [0.0, 0.1, 0.2, 0.3]
        
        scenarios = []
        
        logger.info(f"Generating trading scenarios for {len(games)} games...")
        
        for i, game in enumerate(games):
            # Select scenario parameters
            scenario_type = random.choice(scenario_types)
            info_delay = random.choice(information_delays)
            market_efficiency = random.uniform(0.6, 0.9)
            
            # Create scenario
            scenario = self.create_trading_scenario(
                game=game,
                scenario_type=scenario_type,
                information_delay=info_delay,
                market_efficiency=market_efficiency
            )
            
            scenarios.append(scenario)
            
            # Progress logging
            if (i + 1) % 25 == 0:
                logger.info(f"Generated {i + 1}/{len(games)} trading scenarios")
        
        logger.info(f"Generated {len(scenarios)} trading scenarios")
        return scenarios


# Example usage and testing
if __name__ == "__main__":
    import sys
    import asyncio
    sys.path.append('/Users/hudson/Documents/GitHub/IntelIP/PROJECTS/Neural/Kalshi_Agentic_Agent')
    
    from src.synthetic_data.generators.game_engine import SyntheticGameEngine
    from src.synthetic_data.storage.chromadb_manager import ChromaDBManager
    
    async def test_market_simulator():
        # Initialize components
        chromadb = ChromaDBManager()
        game_engine = SyntheticGameEngine(chromadb)
        market_sim = MarketSimulator()
        
        # Generate game
        game = await game_engine.generate_single_game(
            home_team="KC",
            away_team="BUF",
            game_type="regular"
        )
        
        # Create trading scenario
        scenario = market_sim.create_trading_scenario(
            game=game,
            scenario_type="volatile",
            information_delay=0.2
        )
        
        print(f"Created Trading Scenario: {scenario.market_ticker}")
        print(f"Scenario Type: {scenario.scenario_type}")
        print(f"Total Events: {len(scenario.events)}")
        print(f"Public Events: {len(scenario.public_events)}")
        print(f"Private Events: {len(scenario.private_events)}")
        print(f"Delayed Events: {len(scenario.delayed_events)}")
        print(f"Price History Points: {len(scenario.price_history)}")
        
        # Show some events
        print(f"\nSample Market Events:")
        for event in scenario.events[:5]:
            print(f"  {event.timestamp.strftime('%H:%M:%S')} - {event.event_type.value}: {event.description}")
            print(f"    Price Impact: {event.price_impact:+.3f}, Volume Impact: {event.volume_impact:.3f}")
        
        # Convert to StandardizedEvents
        events = market_sim.convert_to_standardized_events(scenario)
        print(f"\nConverted to {len(events)} StandardizedEvents")
        
        # Show price evolution
        if scenario.price_history:
            print(f"\nPrice Evolution:")
            print(f"  Start: {scenario.price_history[0][1]:.3f}")
            print(f"  End: {scenario.price_history[-1][1]:.3f}")
            print(f"  Change: {scenario.price_history[-1][1] - scenario.price_history[0][1]:+.3f}")
    
    # Run test
    asyncio.run(test_market_simulator())