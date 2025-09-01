"""
Scenario Builder

Creates specialized training scenarios including edge cases, 
rare events, and specific situations for comprehensive agent training.
"""

import logging
import random
from typing import List, Dict, Any, Optional, Tuple, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import math

from .game_engine import SyntheticGameEngine, SyntheticGame, GameContext
from .market_simulator import MarketSimulator, TradingScenario, MarketEvent, MarketEventType
from ..storage.chromadb_manager import ChromaDBManager

logger = logging.getLogger(__name__)


class ScenarioCategory(Enum):
    """Categories of training scenarios"""
    EDGE_CASES = "edge_cases"
    MARKET_CONDITIONS = "market_conditions" 
    GAME_SITUATIONS = "game_situations"
    INFORMATION_ASYMMETRY = "information_asymmetry"
    RISK_MANAGEMENT = "risk_management"
    BEHAVIORAL_PATTERNS = "behavioral_patterns"


@dataclass
class ScenarioTemplate:
    """Template for generating specific scenario types"""
    name: str
    category: ScenarioCategory
    description: str
    frequency: float  # How often this occurs in real data (0-1)
    difficulty: float  # Training difficulty (0-1)
    
    # Game parameters
    game_type: str = "regular"
    game_modifiers: Dict[str, Any] = field(default_factory=dict)
    
    # Market parameters  
    market_type: str = "regular"
    information_delay: float = 0.1
    market_efficiency: float = 0.8
    volatility_modifier: float = 1.0
    
    # Learning objectives
    learning_objectives: List[str] = field(default_factory=list)
    success_criteria: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    tags: List[str] = field(default_factory=list)


@dataclass
class TrainingScenarioSet:
    """Complete set of training scenarios for agent development"""
    set_id: str
    name: str
    description: str
    scenarios: List[TradingScenario] = field(default_factory=list)
    
    # Set characteristics
    total_scenarios: int = 0
    difficulty_distribution: Dict[str, int] = field(default_factory=dict)
    category_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Learning progression
    beginner_scenarios: List[str] = field(default_factory=list)
    intermediate_scenarios: List[str] = field(default_factory=list)
    advanced_scenarios: List[str] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=datetime.now)


class ScenarioBuilder:
    """
    Builds comprehensive training scenario sets with edge cases,
    rare events, and specialized situations for agent training
    """
    
    def __init__(self, 
                 game_engine: SyntheticGameEngine = None,
                 market_simulator: MarketSimulator = None,
                 chromadb_manager: ChromaDBManager = None):
        """
        Initialize scenario builder
        
        Args:
            game_engine: Synthetic game engine
            market_simulator: Market event simulator
            chromadb_manager: ChromaDB for pattern analysis
        """
        self.game_engine = game_engine or SyntheticGameEngine()
        self.market_simulator = market_simulator or MarketSimulator()
        self.chromadb = chromadb_manager or ChromaDBManager()
        
        # Load scenario templates
        self.templates = self._initialize_scenario_templates()
        
        logger.info(f"Initialized ScenarioBuilder with {len(self.templates)} scenario templates")
    
    def _initialize_scenario_templates(self) -> Dict[str, ScenarioTemplate]:
        """Initialize all scenario templates"""
        
        templates = {}
        
        # Edge Case Scenarios
        templates["overtime_thriller"] = ScenarioTemplate(
            name="Overtime Thriller",
            category=ScenarioCategory.EDGE_CASES,
            description="Game goes to overtime with multiple lead changes",
            frequency=0.05,  # ~5% of games
            difficulty=0.9,
            game_type="regular",
            game_modifiers={"force_overtime": True, "close_game": True},
            market_type="volatile",
            volatility_modifier=2.0,
            learning_objectives=["handle_extreme_volatility", "overtime_betting", "live_adjustments"],
            success_criteria={"max_drawdown": 0.15, "profit_factor": 1.2},
            tags=["overtime", "volatile", "rare"]
        )
        
        templates["weather_chaos"] = ScenarioTemplate(
            name="Weather Chaos",
            category=ScenarioCategory.EDGE_CASES,
            description="Severe weather dramatically changes game dynamics",
            frequency=0.02,
            difficulty=0.8,
            game_type="weather",
            game_modifiers={"severe_weather": True, "low_scoring": True},
            market_type="uncertain",
            information_delay=0.3,
            learning_objectives=["weather_impact", "uncertainty_handling", "information_delays"],
            success_criteria={"accuracy": 0.6, "kelly_adherence": 0.8},
            tags=["weather", "uncertainty", "external_factors"]
        )
        
        templates["injury_impact"] = ScenarioTemplate(
            name="Key Injury Impact",
            category=ScenarioCategory.EDGE_CASES,
            description="Star player injury during game creates major shift",
            frequency=0.08,
            difficulty=0.7,
            game_modifiers={"injury_event": True, "momentum_shift": True},
            market_type="news_driven",
            information_delay=0.4,  # Injury news takes time to spread
            learning_objectives=["injury_assessment", "information_arbitrage", "quick_adaptation"],
            success_criteria={"reaction_time": 30, "position_adjustment": 0.3},
            tags=["injury", "news", "information_asymmetry"]
        )
        
        templates["referee_controversy"] = ScenarioTemplate(
            name="Referee Controversy",
            category=ScenarioCategory.EDGE_CASES,
            description="Controversial referee calls affecting game outcome",
            frequency=0.03,
            difficulty=0.8,
            game_modifiers={"controversial_calls": True, "momentum_swings": True},
            market_type="sentiment_driven",
            volatility_modifier=1.5,
            learning_objectives=["sentiment_analysis", "controversy_handling", "noise_filtering"],
            success_criteria={"emotional_control": 0.9, "signal_noise_ratio": 0.7},
            tags=["controversy", "sentiment", "noise"]
        )
        
        # Market Condition Scenarios
        templates["low_liquidity"] = ScenarioTemplate(
            name="Low Liquidity Market",
            category=ScenarioCategory.MARKET_CONDITIONS,
            description="Market with very low liquidity and wide spreads",
            frequency=0.15,
            difficulty=0.6,
            market_type="illiquid",
            game_modifiers={"unpopular_matchup": True},
            learning_objectives=["liquidity_assessment", "spread_management", "position_sizing"],
            success_criteria={"spread_cost": 0.05, "execution_quality": 0.8},
            tags=["liquidity", "spreads", "execution"]
        )
        
        templates["high_volume_spike"] = ScenarioTemplate(
            name="High Volume Trading",
            category=ScenarioCategory.MARKET_CONDITIONS,
            description="Unusual high volume creates pricing inefficiencies",
            frequency=0.12,
            difficulty=0.5,
            market_type="volume_spike",
            volatility_modifier=1.3,
            learning_objectives=["volume_analysis", "arbitrage_opportunities", "momentum_trading"],
            success_criteria={"volume_utilization": 0.7, "timing_accuracy": 0.6},
            tags=["volume", "momentum", "arbitrage"]
        )
        
        templates["market_manipulation"] = ScenarioTemplate(
            name="Market Manipulation",
            category=ScenarioCategory.MARKET_CONDITIONS,
            description="Artificial price movements from large traders",
            frequency=0.05,
            difficulty=0.9,
            market_type="manipulated",
            learning_objectives=["manipulation_detection", "contrarian_strategy", "risk_management"],
            success_criteria={"detection_accuracy": 0.8, "avoided_losses": 0.9},
            tags=["manipulation", "detection", "contrarian"]
        )
        
        # Game Situation Scenarios
        templates["fourth_quarter_comeback"] = ScenarioTemplate(
            name="Fourth Quarter Comeback",
            category=ScenarioCategory.GAME_SITUATIONS,
            description="Team mounting dramatic fourth quarter comeback",
            frequency=0.25,
            difficulty=0.6,
            game_type="comeback",
            game_modifiers={"large_deficit": True, "fourth_quarter_focus": True},
            market_type="trending",
            learning_objectives=["comeback_probability", "live_betting", "momentum_analysis"],
            success_criteria={"timing_precision": 0.7, "trend_following": 0.6},
            tags=["comeback", "momentum", "live_betting"]
        )
        
        templates["defensive_battle"] = ScenarioTemplate(
            name="Defensive Battle",
            category=ScenarioCategory.GAME_SITUATIONS,
            description="Low-scoring defensive game with few opportunities",
            frequency=0.20,
            difficulty=0.4,
            game_type="defensive",
            game_modifiers={"low_scoring": True, "field_goals": True},
            market_type="stable",
            learning_objectives=["low_scoring_dynamics", "patience", "value_betting"],
            success_criteria={"patience_score": 0.8, "value_identification": 0.6},
            tags=["defense", "patience", "value"]
        )
        
        templates["shootout_game"] = ScenarioTemplate(
            name="High-Scoring Shootout",
            category=ScenarioCategory.GAME_SITUATIONS,
            description="High-scoring game with minimal defense",
            frequency=0.18,
            difficulty=0.5,
            game_type="high_scoring",
            game_modifiers={"high_scoring": True, "fast_pace": True},
            volatility_modifier=1.4,
            learning_objectives=["high_scoring_dynamics", "pace_analysis", "over_under_betting"],
            success_criteria={"pace_recognition": 0.7, "scoring_prediction": 0.6},
            tags=["high_scoring", "pace", "over_under"]
        )
        
        # Information Asymmetry Scenarios
        templates["insider_information"] = ScenarioTemplate(
            name="Insider Information",
            category=ScenarioCategory.INFORMATION_ASYMMETRY,
            description="Some traders have early access to key information",
            frequency=0.10,
            difficulty=0.8,
            information_delay=0.5,  # 50% of information is delayed
            market_efficiency=0.6,   # Less efficient due to asymmetry
            learning_objectives=["information_arbitrage", "timing_advantage", "pattern_recognition"],
            success_criteria={"information_speed": 0.8, "arbitrage_capture": 0.7},
            tags=["information", "arbitrage", "timing"]
        )
        
        templates["media_narrative"] = ScenarioTemplate(
            name="Media Narrative Bias",
            category=ScenarioCategory.INFORMATION_ASYMMETRY,
            description="Strong media narrative creates betting bias",
            frequency=0.30,
            difficulty=0.6,
            market_type="narrative_driven",
            learning_objectives=["narrative_analysis", "bias_detection", "contrarian_thinking"],
            success_criteria={"bias_identification": 0.7, "contrarian_profit": 0.5},
            tags=["media", "narrative", "bias"]
        )
        
        # Risk Management Scenarios  
        templates["black_swan_event"] = ScenarioTemplate(
            name="Black Swan Event",
            category=ScenarioCategory.RISK_MANAGEMENT,
            description="Extremely rare event with major market impact",
            frequency=0.001,  # Very rare
            difficulty=0.95,
            game_modifiers={"unprecedented_event": True},
            market_type="crisis",
            volatility_modifier=3.0,
            learning_objectives=["crisis_management", "position_sizing", "stop_losses"],
            success_criteria={"maximum_loss": 0.10, "recovery_time": 5},
            tags=["black_swan", "crisis", "risk_management"]
        )
        
        templates["correlation_breakdown"] = ScenarioTemplate(
            name="Correlation Breakdown",
            category=ScenarioCategory.RISK_MANAGEMENT,
            description="Normal market correlations break down unexpectedly",
            frequency=0.02,
            difficulty=0.85,
            market_type="decorrelated",
            learning_objectives=["correlation_monitoring", "portfolio_risk", "hedging"],
            success_criteria={"correlation_detection": 0.8, "hedge_effectiveness": 0.7},
            tags=["correlation", "portfolio", "hedging"]
        )
        
        # Behavioral Pattern Scenarios
        templates["herd_mentality"] = ScenarioTemplate(
            name="Herd Mentality",
            category=ScenarioCategory.BEHAVIORAL_PATTERNS,
            description="Market exhibits strong herding behavior",
            frequency=0.25,
            difficulty=0.5,
            market_type="herding",
            learning_objectives=["crowd_psychology", "contrarian_signals", "behavioral_finance"],
            success_criteria={"herd_identification": 0.7, "contrarian_timing": 0.6},
            tags=["herding", "psychology", "behavioral"]
        )
        
        templates["overreaction_pattern"] = ScenarioTemplate(
            name="Market Overreaction",
            category=ScenarioCategory.BEHAVIORAL_PATTERNS,
            description="Market overreacts to news then corrects",
            frequency=0.40,
            difficulty=0.4,
            market_type="overreaction",
            learning_objectives=["overreaction_detection", "mean_reversion", "patience"],
            success_criteria={"overreaction_timing": 0.6, "reversion_capture": 0.5},
            tags=["overreaction", "mean_reversion", "behavioral"]
        )
        
        return templates
    
    async def build_comprehensive_training_set(self, 
                                             num_scenarios: int = 1000,
                                             difficulty_progression: bool = True,
                                             include_edge_cases: bool = True,
                                             custom_weights: Dict[str, float] = None) -> TrainingScenarioSet:
        """
        Build comprehensive training scenario set
        
        Args:
            num_scenarios: Total number of scenarios to generate
            difficulty_progression: Whether to include progressive difficulty
            include_edge_cases: Whether to include rare edge cases
            custom_weights: Custom weights for scenario types
            
        Returns:
            Complete training scenario set
        """
        set_id = f"training_set_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Building comprehensive training set with {num_scenarios} scenarios")
        
        training_set = TrainingScenarioSet(
            set_id=set_id,
            name="Comprehensive Agent Training Set",
            description=f"Complete training set with {num_scenarios} diverse scenarios including edge cases"
        )
        
        # Calculate scenario distribution
        distribution = self._calculate_scenario_distribution(
            num_scenarios, 
            include_edge_cases,
            custom_weights
        )
        
        # Generate scenarios by template
        all_scenarios = []
        
        for template_name, count in distribution.items():
            template = self.templates[template_name]
            
            logger.info(f"Generating {count} scenarios for: {template.name}")
            
            scenarios = await self._generate_scenarios_from_template(template, count)
            all_scenarios.extend(scenarios)
            
            # Update distribution tracking
            category = template.category.value
            training_set.category_distribution[category] = training_set.category_distribution.get(category, 0) + count
            
            difficulty_level = self._get_difficulty_level(template.difficulty)
            training_set.difficulty_distribution[difficulty_level] = training_set.difficulty_distribution.get(difficulty_level, 0) + count
        
        # Sort scenarios for progressive training
        if difficulty_progression:
            all_scenarios = self._sort_scenarios_by_difficulty(all_scenarios)
        else:
            random.shuffle(all_scenarios)
        
        training_set.scenarios = all_scenarios
        training_set.total_scenarios = len(all_scenarios)
        
        # Classify scenarios by difficulty
        self._classify_scenarios_by_difficulty(training_set)
        
        logger.info(f"Generated {len(all_scenarios)} total training scenarios")
        logger.info(f"Category distribution: {training_set.category_distribution}")
        logger.info(f"Difficulty distribution: {training_set.difficulty_distribution}")
        
        return training_set
    
    def _calculate_scenario_distribution(self, 
                                       total_scenarios: int,
                                       include_edge_cases: bool,
                                       custom_weights: Dict[str, float] = None) -> Dict[str, int]:
        """Calculate how many scenarios to generate for each template"""
        
        distribution = {}
        
        if custom_weights:
            # Use custom weights
            total_weight = sum(custom_weights.values())
            for template_name, weight in custom_weights.items():
                if template_name in self.templates:
                    count = int(total_scenarios * weight / total_weight)
                    distribution[template_name] = count
        else:
            # Use frequency-based distribution
            total_weight = 0
            weights = {}
            
            for template_name, template in self.templates.items():
                # Skip very rare events if not including edge cases
                if not include_edge_cases and template.frequency < 0.01:
                    continue
                
                weight = template.frequency
                weights[template_name] = weight
                total_weight += weight
            
            # Distribute scenarios
            remaining = total_scenarios
            for template_name, weight in weights.items():
                if remaining <= 0:
                    break
                
                count = max(1, int(total_scenarios * weight / total_weight))
                count = min(count, remaining)
                distribution[template_name] = count
                remaining -= count
            
            # Distribute any remaining scenarios
            while remaining > 0:
                for template_name in weights.keys():
                    if remaining <= 0:
                        break
                    distribution[template_name] = distribution.get(template_name, 0) + 1
                    remaining -= 1
        
        return distribution
    
    async def _generate_scenarios_from_template(self, 
                                              template: ScenarioTemplate, 
                                              count: int) -> List[TradingScenario]:
        """Generate multiple scenarios from a template"""
        
        scenarios = []
        
        for i in range(count):
            try:
                # Generate base game with template modifiers
                game = await self._generate_game_from_template(template)
                
                # Create trading scenario
                scenario = self._create_scenario_from_template(template, game, i)
                
                scenarios.append(scenario)
                
            except Exception as e:
                logger.warning(f"Failed to generate scenario {i} for {template.name}: {e}")
                continue
        
        return scenarios
    
    async def _generate_game_from_template(self, template: ScenarioTemplate) -> SyntheticGame:
        """Generate game based on template specifications"""
        
        # Select teams
        teams = random.sample(self.game_engine.nfl_teams, 2)
        
        # Generate base game
        game = await self.game_engine.generate_single_game(
            home_team=teams[0],
            away_team=teams[1],
            game_type=template.game_type,
            season=2024,
            week=random.randint(1, 18)
        )
        
        # Apply template modifiers
        self._apply_game_modifiers(game, template.game_modifiers)
        
        return game
    
    def _apply_game_modifiers(self, game: SyntheticGame, modifiers: Dict[str, Any]):
        """Apply template modifiers to generated game"""
        
        if modifiers.get("force_overtime"):
            # Ensure overtime by adjusting final score
            game.final_score = (21, 21)  # Will trigger overtime
        
        if modifiers.get("close_game"):
            # Ensure close final score
            diff = abs(game.final_score[0] - game.final_score[1])
            if diff > 7:
                if game.final_score[0] > game.final_score[1]:
                    game.final_score = (game.final_score[0], game.final_score[0] - 3)
                else:
                    game.final_score = (game.final_score[1] - 3, game.final_score[1])
        
        if modifiers.get("large_deficit"):
            # Create large deficit scenario
            if random.random() > 0.5:
                game.final_score = (35, 14)  # Home team wins big
            else:
                game.final_score = (14, 35)  # Away team wins big
        
        if modifiers.get("low_scoring"):
            # Ensure low-scoring game
            total_points = sum(game.final_score)
            if total_points > 35:
                ratio = game.final_score[0] / game.final_score[1] if game.final_score[1] > 0 else 1
                game.final_score = (int(20 * ratio), 20)
        
        if modifiers.get("high_scoring"):
            # Ensure high-scoring game
            total_points = sum(game.final_score)
            if total_points < 50:
                ratio = game.final_score[0] / game.final_score[1] if game.final_score[1] > 0 else 1
                game.final_score = (int(35 * ratio), 35)
        
        # Update game characteristics
        if modifiers.get("severe_weather"):
            game.game_type = "weather"
        
        if modifiers.get("controversial_calls"):
            game.excitement_score = min(1.0, game.excitement_score + 0.3)
    
    def _create_scenario_from_template(self, 
                                     template: ScenarioTemplate,
                                     game: SyntheticGame,
                                     index: int) -> TradingScenario:
        """Create trading scenario based on template"""
        
        # Create base scenario
        scenario = self.market_simulator.create_trading_scenario(
            game=game,
            scenario_type=template.market_type,
            information_delay=template.information_delay,
            market_efficiency=template.market_efficiency
        )
        
        # Apply template-specific modifications
        self._apply_scenario_modifiers(scenario, template)
        
        # Add template metadata
        scenario.scenario_id = f"{template.name.lower().replace(' ', '_')}_{index}"
        
        # Add learning metadata
        if hasattr(scenario, 'metadata'):
            scenario.metadata.update({
                "template": template.name,
                "category": template.category.value,
                "difficulty": template.difficulty,
                "learning_objectives": template.learning_objectives,
                "success_criteria": template.success_criteria,
                "tags": template.tags
            })
        else:
            scenario.metadata = {
                "template": template.name,
                "category": template.category.value,
                "difficulty": template.difficulty,
                "learning_objectives": template.learning_objectives,
                "success_criteria": template.success_criteria,
                "tags": template.tags
            }
        
        return scenario
    
    def _apply_scenario_modifiers(self, scenario: TradingScenario, template: ScenarioTemplate):
        """Apply template-specific modifications to scenario"""
        
        # Volatility modifications
        if template.volatility_modifier != 1.0:
            for event in scenario.events:
                event.price_impact *= template.volatility_modifier
                event.volume_impact *= min(1.0, template.volatility_modifier)
        
        # Add template-specific events
        if template.category == ScenarioCategory.EDGE_CASES:
            self._add_edge_case_events(scenario, template)
        elif template.category == ScenarioCategory.INFORMATION_ASYMMETRY:
            self._add_information_asymmetry_events(scenario, template)
        elif template.category == ScenarioCategory.RISK_MANAGEMENT:
            self._add_risk_events(scenario, template)
    
    def _add_edge_case_events(self, scenario: TradingScenario, template: ScenarioTemplate):
        """Add edge case specific events"""
        
        if "injury" in template.tags:
            # Add sudden injury event
            injury_time = random.choice(scenario.events).timestamp
            injury_event = MarketEvent(
                event_type=MarketEventType.NEWS_EVENT,
                market_ticker=scenario.market_ticker,
                timestamp=injury_time,
                description="Star player injured during play",
                price_impact=random.choice([-0.25, 0.25]),
                volume_impact=0.8,
                confidence_impact=-0.3
            )
            scenario.events.append(injury_event)
        
        if "weather" in template.tags:
            # Add weather update events
            weather_event = MarketEvent(
                event_type=MarketEventType.WEATHER_UPDATE,
                market_ticker=scenario.market_ticker,
                timestamp=scenario.events[0].timestamp - timedelta(hours=1),
                description="Severe weather conditions developing",
                price_impact=random.gauss(0, 0.15),
                volume_impact=0.4,
                confidence_impact=-0.2
            )
            scenario.events.insert(0, weather_event)
    
    def _add_information_asymmetry_events(self, scenario: TradingScenario, template: ScenarioTemplate):
        """Add information asymmetry specific events"""
        
        # Move some events to private/delayed
        num_private = int(len(scenario.events) * 0.3)
        private_events = random.sample(scenario.events, num_private)
        
        for event in private_events:
            scenario.private_events.append(event)
            if event in scenario.public_events:
                scenario.public_events.remove(event)
    
    def _add_risk_events(self, scenario: TradingScenario, template: ScenarioTemplate):
        """Add risk management specific events"""
        
        if "black_swan" in template.tags:
            # Add extreme market event
            extreme_event = MarketEvent(
                event_type=MarketEventType.NEWS_EVENT,
                market_ticker=scenario.market_ticker,
                timestamp=random.choice(scenario.events).timestamp,
                description="Unprecedented event affects game",
                price_impact=random.choice([-0.5, 0.5]),
                volume_impact=1.0,
                confidence_impact=-0.5
            )
            scenario.events.append(extreme_event)
    
    def _get_difficulty_level(self, difficulty: float) -> str:
        """Convert difficulty score to level"""
        if difficulty < 0.4:
            return "beginner"
        elif difficulty < 0.7:
            return "intermediate"
        else:
            return "advanced"
    
    def _sort_scenarios_by_difficulty(self, scenarios: List[TradingScenario]) -> List[TradingScenario]:
        """Sort scenarios by difficulty for progressive training"""
        
        def get_scenario_difficulty(scenario):
            return scenario.metadata.get("difficulty", 0.5)
        
        return sorted(scenarios, key=get_scenario_difficulty)
    
    def _classify_scenarios_by_difficulty(self, training_set: TrainingScenarioSet):
        """Classify scenarios into difficulty levels"""
        
        for scenario in training_set.scenarios:
            difficulty = scenario.metadata.get("difficulty", 0.5)
            level = self._get_difficulty_level(difficulty)
            
            if level == "beginner":
                training_set.beginner_scenarios.append(scenario.scenario_id)
            elif level == "intermediate":
                training_set.intermediate_scenarios.append(scenario.scenario_id)
            else:
                training_set.advanced_scenarios.append(scenario.scenario_id)
    
    def export_training_set(self, training_set: TrainingScenarioSet, output_path: str):
        """Export training set to file"""
        
        export_data = {
            "set_id": training_set.set_id,
            "name": training_set.name,
            "description": training_set.description,
            "total_scenarios": training_set.total_scenarios,
            "difficulty_distribution": training_set.difficulty_distribution,
            "category_distribution": training_set.category_distribution,
            "beginner_scenarios": training_set.beginner_scenarios,
            "intermediate_scenarios": training_set.intermediate_scenarios,
            "advanced_scenarios": training_set.advanced_scenarios,
            "created_at": training_set.created_at.isoformat(),
            "scenarios": [
                {
                    "scenario_id": scenario.scenario_id,
                    "market_ticker": scenario.market_ticker,
                    "scenario_type": scenario.scenario_type,
                    "information_delay": scenario.information_delay,
                    "market_efficiency": scenario.market_efficiency,
                    "num_events": len(scenario.events),
                    "game_final_score": scenario.game.final_score,
                    "metadata": getattr(scenario, 'metadata', {})
                }
                for scenario in training_set.scenarios
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported training set to {output_path}")
    
    def get_scenario_statistics(self, training_set: TrainingScenarioSet) -> Dict[str, Any]:
        """Get comprehensive statistics about training set"""
        
        stats = {
            "overview": {
                "total_scenarios": training_set.total_scenarios,
                "categories": len(training_set.category_distribution),
                "difficulty_levels": len(training_set.difficulty_distribution)
            },
            "category_distribution": training_set.category_distribution,
            "difficulty_distribution": training_set.difficulty_distribution,
            "progression": {
                "beginner": len(training_set.beginner_scenarios),
                "intermediate": len(training_set.intermediate_scenarios),
                "advanced": len(training_set.advanced_scenarios)
            }
        }
        
        # Calculate learning objective coverage
        all_objectives = set()
        objective_counts = {}
        
        for scenario in training_set.scenarios:
            objectives = scenario.metadata.get("learning_objectives", [])
            for obj in objectives:
                all_objectives.add(obj)
                objective_counts[obj] = objective_counts.get(obj, 0) + 1
        
        stats["learning_objectives"] = {
            "total_unique": len(all_objectives),
            "coverage": objective_counts
        }
        
        # Calculate tag distribution
        tag_counts = {}
        for scenario in training_set.scenarios:
            tags = scenario.metadata.get("tags", [])
            for tag in tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        stats["tags"] = tag_counts
        
        return stats


# Example usage and testing
if __name__ == "__main__":
    import sys
    import asyncio
    sys.path.append('/Users/hudson/Documents/GitHub/IntelIP/PROJECTS/Neural/Kalshi_Agentic_Agent')
    
    from src.synthetic_data.generators.game_engine import SyntheticGameEngine
    from src.synthetic_data.generators.market_simulator import MarketSimulator
    from src.synthetic_data.storage.chromadb_manager import ChromaDBManager
    
    async def test_scenario_builder():
        # Initialize components
        chromadb = ChromaDBManager()
        game_engine = SyntheticGameEngine(chromadb)
        market_simulator = MarketSimulator()
        scenario_builder = ScenarioBuilder(game_engine, market_simulator, chromadb)
        
        # Build comprehensive training set
        training_set = await scenario_builder.build_comprehensive_training_set(
            num_scenarios=100,  # Small test set
            difficulty_progression=True,
            include_edge_cases=True
        )
        
        print(f"Built Training Set: {training_set.name}")
        print(f"Total Scenarios: {training_set.total_scenarios}")
        print(f"Categories: {training_set.category_distribution}")
        print(f"Difficulty: {training_set.difficulty_distribution}")
        
        # Get statistics
        stats = scenario_builder.get_scenario_statistics(training_set)
        print(f"\nDetailed Statistics:")
        print(f"Learning Objectives: {stats['learning_objectives']['total_unique']}")
        print(f"Tag Distribution: {list(stats['tags'].keys())}")
        
        # Show sample scenarios
        print(f"\nSample Scenarios:")
        for scenario in training_set.scenarios[:3]:
            print(f"  {scenario.scenario_id}: {scenario.metadata.get('template', 'Unknown')}")
            print(f"    Difficulty: {scenario.metadata.get('difficulty', 0):.2f}")
            print(f"    Events: {len(scenario.events)}")
            print(f"    Objectives: {scenario.metadata.get('learning_objectives', [])}")
        
        # Export training set
        scenario_builder.export_training_set(training_set, "test_training_set.json")
        print(f"\nTraining set exported to test_training_set.json")
    
    # Run test
    asyncio.run(test_scenario_builder())