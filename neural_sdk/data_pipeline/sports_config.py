"""
Sports Configuration Module
Define sports and their Kalshi series tickers for standardized market discovery
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class Sport(Enum):
    """Supported sports"""
    NFL = "nfl"
    NCAAF = "ncaaf"
    CFP = "cfp"


@dataclass
class SportConfig:
    """Configuration for a specific sport"""
    name: str
    display_name: str
    series_ticker: str
    season_active: bool = True
    market_types: List[str] = None
    championship_ticker: Optional[str] = None

    def __post_init__(self):
        if self.market_types is None:
            self.market_types = ["game_winner", "spread", "total"]


# Sport configurations
SPORT_CONFIGS: Dict[Sport, SportConfig] = {
    Sport.NFL: SportConfig(
        name="nfl",
        display_name="NFL",
        series_ticker="KXNFLGAME",
        season_active=True,
        market_types=["game_winner", "spread", "total", "player_props"],
        championship_ticker="KXSB"  # Super Bowl
    ),
    
    Sport.NCAAF: SportConfig(
        name="ncaaf",
        display_name="NCAA Football",
        series_ticker="KXNCAAFGAME",  # Individual games
        season_active=True,
        market_types=["game_winner", "spread", "total"],
        championship_ticker="KXNCAAF"  # Championship winner
    ),
    
    Sport.CFP: SportConfig(
        name="cfp",
        display_name="College Football Playoff", 
        series_ticker="KXCFPGAME",  # Will need to verify actual ticker
        season_active=False,  # Not yet active
        market_types=["game_winner", "spread", "total"]
    )
}


def get_sport_config(sport: Sport) -> SportConfig:
    """Get configuration for a sport"""
    return SPORT_CONFIGS[sport]


def get_active_sports() -> List[Sport]:
    """Get list of currently active sports"""
    return [sport for sport, config in SPORT_CONFIGS.items() if config.season_active]


def get_series_ticker(sport: Sport) -> str:
    """Get Kalshi series ticker for a sport"""
    return SPORT_CONFIGS[sport].series_ticker