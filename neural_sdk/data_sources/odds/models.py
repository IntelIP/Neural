"""
Data models for The Odds API.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum


class Sport(Enum):
    """Supported sports."""
    NFL = "americanfootball_nfl"
    NCAAF = "americanfootball_ncaaf"
    NBA = "basketball_nba"
    NCAAB = "basketball_ncaab"
    MLB = "baseball_mlb"
    NHL = "icehockey_nhl"
    MMA = "mma_mixed_martial_arts"
    SOCCER_EPL = "soccer_epl"
    SOCCER_MLS = "soccer_usa_mls"


class Market(Enum):
    """Betting market types."""
    H2H = "h2h"  # Moneyline
    SPREADS = "spreads"  # Point spreads
    TOTALS = "totals"  # Over/under
    OUTRIGHTS = "outrights"  # Futures
    H2H_LAY = "h2h_lay"  # Lay betting
    
    
class OddsFormat(Enum):
    """Odds display format."""
    AMERICAN = "american"
    DECIMAL = "decimal"
    

class Bookmaker(Enum):
    """Major US bookmakers."""
    DRAFTKINGS = "draftkings"
    FANDUEL = "fanduel"
    BETMGM = "betmgm"
    CAESARS = "caesars"
    POINTSBET = "pointsbetus"
    BETRIVERS = "betrivers"
    BOVADA = "bovada"
    BETONLINE = "betonlineag"


@dataclass
class MarketOutcome:
    """Single outcome within a market."""
    name: str  # Team name or Over/Under
    price: float  # Odds in specified format
    point: Optional[float] = None  # Spread or total points


@dataclass
class MarketOdds:
    """Odds for a specific market."""
    key: str  # Market type (h2h, spreads, totals)
    last_update: datetime
    outcomes: List[MarketOutcome]
    
    def get_best_price(self, outcome_name: str) -> Optional[float]:
        """Get best price for specific outcome."""
        for outcome in self.outcomes:
            if outcome.name == outcome_name:
                return outcome.price
        return None
    
    def get_spread(self, team_name: str) -> Optional[float]:
        """Get spread for specific team."""
        if self.key != "spreads":
            return None
        for outcome in self.outcomes:
            if outcome.name == team_name:
                return outcome.point
        return None


@dataclass
class BookmakerOdds:
    """Odds from a single bookmaker."""
    key: str  # Bookmaker identifier
    title: str  # Bookmaker display name
    last_update: datetime
    markets: List[MarketOdds]
    
    def get_market(self, market_type: str) -> Optional[MarketOdds]:
        """Get specific market odds."""
        for market in self.markets:
            if market.key == market_type:
                return market
        return None
    
    def get_moneyline(self, team_name: str) -> Optional[float]:
        """Get moneyline odds for team."""
        h2h = self.get_market("h2h")
        if h2h:
            return h2h.get_best_price(team_name)
        return None


@dataclass
class OddsData:
    """Complete odds data for a game."""
    id: str  # Unique game identifier
    sport_key: str  # Sport identifier
    sport_title: str  # Sport display name
    commence_time: datetime  # Game start time
    home_team: str
    away_team: str
    bookmakers: List[BookmakerOdds]
    
    @property
    def is_live(self) -> bool:
        """Check if game is currently live."""
        return datetime.utcnow() > self.commence_time
    
    def get_consensus_moneyline(self) -> Dict[str, float]:
        """Get average moneyline odds across all bookmakers."""
        home_odds = []
        away_odds = []
        
        for bookmaker in self.bookmakers:
            home = bookmaker.get_moneyline(self.home_team)
            away = bookmaker.get_moneyline(self.away_team)
            if home:
                home_odds.append(home)
            if away:
                away_odds.append(away)
        
        return {
            self.home_team: sum(home_odds) / len(home_odds) if home_odds else 0,
            self.away_team: sum(away_odds) / len(away_odds) if away_odds else 0
        }
    
    def get_best_odds(self, market_type: str = "h2h") -> Dict[str, Dict[str, Any]]:
        """Find best odds across all bookmakers for a market."""
        best_odds = {}
        
        for bookmaker in self.bookmakers:
            market = bookmaker.get_market(market_type)
            if not market:
                continue
                
            for outcome in market.outcomes:
                if outcome.name not in best_odds:
                    best_odds[outcome.name] = {
                        "price": outcome.price,
                        "bookmaker": bookmaker.title,
                        "point": outcome.point
                    }
                else:
                    # For American odds, higher positive or less negative is better
                    current_best = best_odds[outcome.name]["price"]
                    if outcome.price > current_best:
                        best_odds[outcome.name] = {
                            "price": outcome.price,
                            "bookmaker": bookmaker.title,
                            "point": outcome.point
                        }
        
        return best_odds
    
    def find_arbitrage(self) -> Optional[Dict[str, Any]]:
        """
        Find arbitrage opportunities across bookmakers.
        Returns opportunity details if found.
        """
        # Convert American odds to implied probability
        def american_to_prob(odds: float) -> float:
            if odds > 0:
                return 100 / (odds + 100)
            else:
                return abs(odds) / (abs(odds) + 100)
        
        best_home = {"price": -10000, "bookmaker": None}
        best_away = {"price": -10000, "bookmaker": None}
        
        for bookmaker in self.bookmakers:
            home = bookmaker.get_moneyline(self.home_team)
            away = bookmaker.get_moneyline(self.away_team)
            
            if home and home > best_home["price"]:
                best_home = {"price": home, "bookmaker": bookmaker.title}
            if away and away > best_away["price"]:
                best_away = {"price": away, "bookmaker": bookmaker.title}
        
        if best_home["bookmaker"] and best_away["bookmaker"]:
            # Calculate combined probability
            home_prob = american_to_prob(best_home["price"])
            away_prob = american_to_prob(best_away["price"])
            total_prob = home_prob + away_prob
            
            if total_prob < 1.0:  # Arbitrage exists
                profit_margin = (1.0 - total_prob) * 100
                return {
                    "exists": True,
                    "profit_margin": profit_margin,
                    "home_bet": {
                        "team": self.home_team,
                        "odds": best_home["price"],
                        "bookmaker": best_home["bookmaker"],
                        "stake_percentage": home_prob / total_prob
                    },
                    "away_bet": {
                        "team": self.away_team,
                        "odds": best_away["price"],
                        "bookmaker": best_away["bookmaker"],
                        "stake_percentage": away_prob / total_prob
                    }
                }
        
        return None


@dataclass
class LineMovement:
    """Track odds movement over time."""
    game_id: str
    market_type: str
    bookmaker: str
    team: str
    movements: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_movement(self, price: float, point: Optional[float] = None):
        """Record new odds."""
        self.movements.append({
            "timestamp": datetime.utcnow(),
            "price": price,
            "point": point
        })
    
    def get_trend(self) -> str:
        """Determine if line is moving up, down, or stable."""
        if len(self.movements) < 2:
            return "stable"
        
        recent = self.movements[-5:]  # Last 5 movements
        if len(recent) < 2:
            return "stable"
        
        first_price = recent[0]["price"]
        last_price = recent[-1]["price"]
        
        if last_price > first_price + 5:
            return "improving"  # Better odds for bettor
        elif last_price < first_price - 5:
            return "worsening"  # Worse odds for bettor
        else:
            return "stable"