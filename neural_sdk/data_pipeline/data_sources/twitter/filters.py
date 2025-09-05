"""
Twitter Filter Rule Management
Dynamic filter creation and management for game-specific monitoring
"""

import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import httpx

from .models import FilterRule

logger = logging.getLogger(__name__)


class FilterManager:
    """
    Manages Twitter stream filter rules via TwitterAPI.io
    """
    
    # Key Twitter accounts for sports
    INSIDER_ACCOUNTS = {
        'nfl': [
            'AdamSchefter',      # ESPN NFL insider
            'RapSheet',          # NFL Network insider
            'JayGlazer',         # Fox Sports insider
            'AlbertBreer',       # SI NFL reporter
            'MikeGarafolo',      # NFL Network
            'TomPelissero',      # NFL Network
            'FieldYates',        # ESPN Fantasy
            'BleacherReport',    # General sports
            'ESPNFantasy',       # Fantasy updates
            'NFLGameDay'         # Official NFL
        ],
        'nba': [
            'wojespn',           # Adrian Wojnarowski
            'ShamsCharania',     # The Athletic
            'ChrisBHaynes',      # TNT/Bleacher Report
            'TheSteinLine',      # Marc Stein
            'WindhorstESPN',     # Brian Windhorst
            'ZachLowe_NBA',      # ESPN
            'TimBontemps',       # ESPN
            'BleacherReport',    # General sports
            'ESPNFantasy',       # Fantasy updates
            'NBAOfficial'        # Official NBA
        ],
        'mlb': [
            'JeffPassan',        # ESPN MLB insider
            'Ken_Rosenthal',     # The Athletic
            'JonHeyman',         # MLB Network
            'BNightengale',      # USA Today
            'MLB',               # Official MLB
            'BleacherReport'     # General sports
        ]
    }
    
    # Team-specific accounts mapping
    TEAM_ACCOUNTS = {
        # NFL Teams
        'Chiefs': ['Chiefs', 'ChiefsReporter', 'ArrowheadPride'],
        'Bills': ['BuffaloBills', 'BillsWire', 'BuffRumblings'],
        'Eagles': ['Eagles', 'EaglesNation', 'BleedingGreen'],
        'Cowboys': ['dallascowboys', 'CowboysNation', 'BloggingTheBoys'],
        'Packers': ['packers', 'PackersNews', 'CheeseheadTV'],
        '49ers': ['49ers', 'NinersNation', 'NinersWire'],
        
        # NBA Teams
        'Lakers': ['Lakers', 'LakersReporter', 'LakersSBN'],
        'Warriors': ['warriors', 'WarriorsWorld', 'GSWReddit'],
        'Celtics': ['celtics', 'CelticsHub', 'CelticsSBN'],
        'Heat': ['MiamiHEAT', 'HeatNation', 'HotHotHoops'],
        'Nets': ['BrooklynNets', 'NetsDaily', 'NetsSBN'],
        'Bucks': ['Bucks', 'BucksNewsNow', 'BrewHoop']
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize filter manager
        
        Args:
            api_key: TwitterAPI.io API key
        """
        # Prefer TWITTERAPI_KEY (TwitterAPI.io). Fall back to TWITTER_BEARER_TOKEN for convenience.
        self.api_key = api_key or os.getenv('TWITTERAPI_KEY') or os.getenv('TWITTER_BEARER_TOKEN')
        if not self.api_key:
            raise ValueError("TwitterAPI.io API key required")
        if not api_key and not os.getenv('TWITTERAPI_KEY') and os.getenv('TWITTER_BEARER_TOKEN'):
            logger.warning("Using TWITTER_BEARER_TOKEN for TwitterAPI.io filters. Ensure it's a TwitterAPI.io API key.")
        
        # Allow overriding API base via env for compatibility with provider changes
        self.api_base_url = os.getenv("TWITTER_API_BASE", "https://api.twitterapi.io")
        self.active_rules: Dict[str, FilterRule] = {}
    
    async def create_game_filter(
        self,
        home_team: str,
        away_team: str,
        sport: str = 'nfl',
        players: Optional[List[str]] = None,
        additional_hashtags: Optional[List[str]] = None
    ) -> FilterRule:
        """
        Create a comprehensive filter for a specific game
        
        Args:
            home_team: Home team name
            away_team: Away team name
            sport: Sport type (nfl, nba, mlb)
            players: Key players to monitor
            additional_hashtags: Extra hashtags to track
            
        Returns:
            Created FilterRule
        """
        query_parts = []
        
        # 1. Insider accounts for breaking news
        insiders = self.INSIDER_ACCOUNTS.get(sport.lower(), [])
        if insiders:
            insider_query = ' OR '.join(f'from:{account}' for account in insiders)
            query_parts.append(f'({insider_query})')
        
        # 2. Team-specific accounts
        home_accounts = self.TEAM_ACCOUNTS.get(home_team, [home_team])
        away_accounts = self.TEAM_ACCOUNTS.get(away_team, [away_team])
        team_accounts = home_accounts + away_accounts
        
        if team_accounts:
            team_query = ' OR '.join(f'from:{account}' for account in team_accounts)
            query_parts.append(f'({team_query})')
        
        # 3. Team mentions and hashtags
        team_hashtags = [
            f'#{home_team}',
            f'#{away_team}',
            f'#{home_team}vs{away_team}',
            f'#{away_team}vs{home_team}',
            f'#{home_team}{away_team}'
        ]
        
        # Add sport-specific hashtags
        sport_hashtags = {
            'nfl': ['#NFL', '#NFLGameDay', '#FantasyFootball'],
            'nba': ['#NBA', '#NBATwitter', '#FantasyBasketball'],
            'mlb': ['#MLB', '#BaseballTwitter', '#FantasyBaseball']
        }
        
        all_hashtags = team_hashtags + sport_hashtags.get(sport.lower(), [])
        if additional_hashtags:
            all_hashtags.extend(additional_hashtags)
        
        hashtag_query = ' OR '.join(all_hashtags)
        query_parts.append(f'({hashtag_query})')
        
        # 4. Player-specific monitoring with injury/status keywords
        if players:
            player_queries = []
            status_keywords = [
                'injury', 'injured', 'questionable', 'doubtful', 'out',
                'active', 'inactive', 'starting', 'benched', 'return',
                'update', 'status', 'report', 'news', 'breaking'
            ]
            
            for player in players:
                # Create comprehensive player query
                player_query = f'"{player}" ({" OR ".join(status_keywords)})'
                player_queries.append(player_query)
            
            if player_queries:
                query_parts.append(f'({" OR ".join(player_queries)})')
        
        # 5. High-impact keywords (always monitor)
        impact_keywords = [
            f'("{home_team}" OR "{away_team}") AND ("breaking" OR "BREAKING")',
            f'("{home_team}" OR "{away_team}") AND ("injury" OR "injured")',
            f'("{home_team}" OR "{away_team}") AND ("suspended" OR "ejected")',
            f'("{home_team}" OR "{away_team}") AND ("lineup" OR "starting")'
        ]
        query_parts.append(f'({" OR ".join(impact_keywords)})')
        
        # Combine all parts
        final_query = ' OR '.join(query_parts)
        
        # Create filter rule
        rule = FilterRule(
            value=final_query,
            tag=f'{sport}_{home_team}_{away_team}_{datetime.now().strftime("%Y%m%d_%H%M")}',
            polling_interval=0.1,  # 100ms for live games (fastest available)
            active=True
        )
        
        # Apply the rule
        await self.apply_rule(rule)
        
        return rule
    
    async def create_player_filter(
        self,
        player_name: str,
        team: Optional[str] = None,
        keywords: Optional[List[str]] = None
    ) -> FilterRule:
        """
        Create a filter for a specific player
        
        Args:
            player_name: Player's name
            team: Optional team name
            keywords: Optional specific keywords to monitor
            
        Returns:
            Created FilterRule
        """
        query_parts = []
        
        # Base player query
        player_query = f'"{player_name}"'
        
        # Add team context if provided
        if team:
            player_query = f'{player_query} AND ("{team}" OR #{team})'
        
        query_parts.append(player_query)
        
        # Add keyword monitoring
        default_keywords = [
            'injury', 'injured', 'status', 'update', 'questionable',
            'doubtful', 'out', 'return', 'active', 'inactive',
            'starting', 'benched', 'news', 'report'
        ]
        
        all_keywords = keywords or default_keywords
        keyword_query = f'"{player_name}" AND ({" OR ".join(all_keywords)})'
        query_parts.append(keyword_query)
        
        # Combine queries
        final_query = f'({" OR ".join(query_parts)})'
        
        rule = FilterRule(
            value=final_query,
            tag=f'player_{player_name.replace(" ", "_")}_{datetime.now().strftime("%Y%m%d")}',
            polling_interval=1.0,  # 1 second for player monitoring
            active=True
        )
        
        await self.apply_rule(rule)
        return rule
    
    async def create_event_filter(
        self,
        event_name: str,
        keywords: List[str],
        accounts: Optional[List[str]] = None
    ) -> FilterRule:
        """
        Create a filter for a specific event
        
        Args:
            event_name: Name of the event
            keywords: Keywords to monitor
            accounts: Optional specific accounts to monitor
            
        Returns:
            Created FilterRule
        """
        query_parts = []
        
        # Keyword monitoring
        keyword_query = ' OR '.join(f'"{kw}"' for kw in keywords)
        query_parts.append(f'({keyword_query})')
        
        # Account monitoring
        if accounts:
            account_query = ' OR '.join(f'from:{acc}' for acc in accounts)
            query_parts.append(f'({account_query})')
        
        final_query = ' AND '.join(query_parts) if len(query_parts) > 1 else query_parts[0]
        
        rule = FilterRule(
            value=final_query,
            tag=f'event_{event_name}_{datetime.now().strftime("%Y%m%d")}',
            polling_interval=2.0,  # 2 seconds for event monitoring
            active=True
        )
        
        await self.apply_rule(rule)
        return rule
    
    async def apply_rule(self, rule: FilterRule) -> Dict[str, Any]:
        """
        Apply a filter rule locally (TwitterAPI.io uses search, not webhook rules)
        
        Args:
            rule: FilterRule to apply
            
        Returns:
            Rule info
        """
        # Generate ID if not set
        if not rule.id:
            rule.id = f"rule_{datetime.now().timestamp()}"
        
        # Store rule locally for search queries
        self.active_rules[rule.id] = rule
        
        logger.info(f"Filter rule stored locally: {rule.tag}")
        logger.info(f"Query: {rule.value}")
        
        # Return rule info
        return {
            'data': {
                'id': rule.id,
                'value': rule.value,
                'tag': rule.tag
            }
        }
    
    async def delete_rule(self, rule_id: str) -> bool:
        """
        Delete a filter rule
        
        Args:
            rule_id: ID of the rule to delete
            
        Returns:
            True if successful
        """
        async with httpx.AsyncClient() as client:
            headers = {"X-API-Key": self.api_key}
            
            try:
                response = await client.delete(
                    f"{self.api_base_url}/twitter/webhook/rules/{rule_id}",
                    headers=headers
                )
                
                if response.status_code in [200, 204]:
                    if rule_id in self.active_rules:
                        del self.active_rules[rule_id]
                    logger.info(f"Filter rule deleted: {rule_id}")
                    return True
                else:
                    logger.error(f"Failed to delete filter: {response.text}")
                    return False
                    
            except Exception as e:
                logger.error(f"Error deleting filter rule: {e}")
                return False
    
    async def list_rules(self) -> List[Dict[str, Any]]:
        """
        List all active filter rules
        
        Returns:
            List of active rules
        """
        async with httpx.AsyncClient() as client:
            headers = {"X-API-Key": self.api_key}
            
            try:
                response = await client.get(
                    f"{self.api_base_url}/twitter/webhook/rules",
                    headers=headers
                )
                
                if response.status_code == 200:
                    return response.json().get('data', [])
                else:
                    logger.error(f"Failed to list filters: {response.text}")
                    return []
                    
            except Exception as e:
                logger.error(f"Error listing filter rules: {e}")
                return []
    
    async def cleanup_old_rules(self, hours: int = 24):
        """
        Delete filter rules older than specified hours
        
        Args:
            hours: Age threshold in hours
        """
        rules = await self.list_rules()
        current_time = datetime.now()
        
        for rule in rules:
            # Parse rule tag for timestamp
            tag = rule.get('tag', '')
            if '_' in tag:
                try:
                    # Extract timestamp from tag
                    timestamp_str = tag.split('_')[-2] + tag.split('_')[-1]
                    rule_time = datetime.strptime(timestamp_str, '%Y%m%d%H%M')
                    
                    # Check age
                    age_hours = (current_time - rule_time).total_seconds() / 3600
                    
                    if age_hours > hours:
                        await self.delete_rule(rule.get('id'))
                        logger.info(f"Deleted old rule: {tag} (age: {age_hours:.1f} hours)")
                        
                except Exception as e:
                    logger.debug(f"Could not parse timestamp from tag {tag}: {e}")
