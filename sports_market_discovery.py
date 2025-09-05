#!/usr/bin/env python3
"""
Sports Market Discovery - Kalshi Recommended Approach
Implements proper series-first market discovery following Kalshi best practices
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from neural_sdk.data_pipeline.data_sources.kalshi.client import KalshiClient

class SportsMarketDiscovery:
    """
    Sports market discovery using Kalshi's recommended series-first approach.
    
    Follows the proper workflow:
    1. Discover sports series using /series endpoint
    2. Filter by sports category 
    3. Query specific markets using series_ticker
    4. Apply status and pagination filters
    """
    
    def __init__(self):
        self.client = KalshiClient()
        self._sports_series_cache = None
        self._nfl_series_cache = None
    
    def close(self):
        """Close client connection"""
        self.client.close()
    
    def discover_sports_series(self, leagues: List[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Discover sports series using Kalshi's recommended tags-based approach.
        
        Args:
            leagues: List of league tags (e.g., ['NFL', 'NBA', 'CFP'])
        
        Returns:
            Dict with league as key and list of series as value
        """
        if leagues is None:
            leagues = ['NFL', 'NBA', 'CFP']
            
        if self._sports_series_cache is None:
            try:
                print("🔍 Discovering sports series using CORRECTED Kalshi tags approach...")
                
                # Use CORRECTED Kalshi tags filtering (Football, Basketball, Baseball, etc.)
                sports_data = self.client.get_sports_series(leagues=leagues)
                
                total_series = sum(len(series_list) for series_list in sports_data.values())
                print(f"✅ Found {total_series} total sports series across {len(sports_data)} leagues")
                
                # Log discovered series for debugging
                for league, series_list in sports_data.items():
                    print(f"📊 {league}: {len(series_list)} series")
                    for series in series_list[:2]:  # Show first 2 per league
                        ticker = series.get('series_ticker', 'Unknown')
                        title = series.get('title', 'No title')
                        print(f"   • {ticker} - {title}")
                
                self._sports_series_cache = sports_data
                
            except Exception as e:
                print(f"❌ Error discovering sports series: {e}")
                self._sports_series_cache = {}
        
        return self._sports_series_cache
    
    def discover_nfl_series(self) -> List[Dict[str, Any]]:
        """
        Discover NFL-specific series using CORRECTED tags filtering.
        
        Returns:
            List of NFL series
        """
        if self._nfl_series_cache is None:
            try:
                print("🏈 Discovering NFL series using tags='Football' (CORRECTED approach)...")
                
                # CORRECTED: Use 'Football' tag, not 'NFL' tag
                nfl_series = self.client.get_nfl_series()
                
                print(f"🏈 Found {len(nfl_series)} NFL/Football series:")
                for series in nfl_series:
                    ticker = series.get('ticker', 'Unknown') 
                    title = series.get('title', 'No title')
                    print(f"   • {ticker} - {title}")
                
                self._nfl_series_cache = nfl_series
                
            except Exception as e:
                print(f"❌ Error discovering NFL series: {e}")
                self._nfl_series_cache = []
        
        return self._nfl_series_cache
    
    def get_markets_for_series(
        self, 
        series_ticker: str, 
        status: str = 'open',
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get markets for a specific series using proper API filtering.
        
        Args:
            series_ticker: Series ticker to query
            status: Market status filter ('open', 'active', 'closed')
            limit: Number of markets to retrieve
            
        Returns:
            List of markets for the series
        """
        try:
            print(f"📊 Getting markets for series: {series_ticker}")
            
            # Use proper series_ticker filtering (Kalshi recommended approach)
            response = self.client.get_markets(
                series_ticker=series_ticker,
                status=status,
                limit=limit
            )
            
            markets = response.get('markets', [])
            print(f"   ✅ Found {len(markets)} markets with status '{status}'")
            
            return markets
            
        except Exception as e:
            print(f"   ❌ Error getting markets for {series_ticker}: {e}")
            return []
    
    def find_nfl_markets(
        self, 
        status: str = 'open',
        limit_per_series: int = 20
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Find all NFL markets using proper series-first approach.
        
        Args:
            status: Market status to filter by
            limit_per_series: Max markets per series
            
        Returns:
            Dict with series_ticker as key and list of markets as value
        """
        nfl_series = self.discover_nfl_series()
        
        if not nfl_series:
            print("❌ No NFL series found - may be off-season or API access issue")
            return {}
        
        print(f"🏈 Querying {len(nfl_series)} NFL series for markets...")
        
        nfl_markets = {}
        total_markets = 0
        
        for series in nfl_series:
            series_ticker = series.get('series_ticker')
            if series_ticker:
                markets = self.get_markets_for_series(
                    series_ticker, 
                    status=status, 
                    limit=limit_per_series
                )
                
                if markets:
                    nfl_markets[series_ticker] = markets
                    total_markets += len(markets)
        
        print(f"🎯 Total NFL markets found: {total_markets}")
        return nfl_markets
    
    def get_current_nfl_games(self) -> List[Dict[str, Any]]:
        """
        Get current/upcoming NFL games using proper discovery.
        
        Returns:
            List of current NFL game markets
        """
        print("🏈 Finding current NFL games...")
        
        # Get all open NFL markets
        nfl_markets = self.find_nfl_markets(status='open')
        
        current_games = []
        for series_ticker, markets in nfl_markets.items():
            for market in markets:
                ticker = market.get('ticker', '')
                title = market.get('title', '')
                
                # Look for game-specific markets (vs team-vs-team patterns)
                if any(pattern in ticker.upper() or pattern in title.upper() 
                       for pattern in ['VS', 'GAME', 'WIN', 'SPREAD']):
                    current_games.append({
                        'series_ticker': series_ticker,
                        'market': market
                    })
        
        print(f"🎮 Found {len(current_games)} current NFL game markets")
        
        # Show sample games
        for game in current_games[:3]:
            market = game['market']
            ticker = market.get('ticker', 'Unknown')
            title = market.get('title', 'No title')
            print(f"   🏈 {ticker} - {title}")
        
        return current_games
    
    def find_team_markets(self, team_code: str) -> List[Dict[str, Any]]:
        """
        Find markets for a specific team (e.g., 'DAL', 'PHI').
        
        Args:
            team_code: Team code to search for
            
        Returns:
            List of markets involving the team
        """
        print(f"🏈 Finding markets for team: {team_code}")
        
        nfl_markets = self.find_nfl_markets(status='open')
        team_markets = []
        
        for series_ticker, markets in nfl_markets.items():
            for market in markets:
                ticker = market.get('ticker', '').upper()
                title = market.get('title', '').upper()
                
                # Check if team is mentioned in ticker or title
                if team_code.upper() in ticker or team_code.upper() in title:
                    team_markets.append(market)
        
        print(f"   ✅ Found {len(team_markets)} markets for {team_code}")
        return team_markets
    
    def get_working_example_ticker(self) -> Optional[str]:
        """
        Get a working market ticker for testing purposes.
        
        Returns:
            A valid market ticker or None if no markets available
        """
        print("🔧 Finding working market ticker for testing...")
        
        # Try NFL first
        nfl_markets = self.find_nfl_markets(status='open')
        
        if nfl_markets:
            for series_ticker, markets in nfl_markets.items():
                if markets:
                    ticker = markets[0].get('ticker')
                    print(f"✅ Found working NFL ticker: {ticker}")
                    return ticker
        
        # Fallback to any sports market
        sports_series = self.discover_sports_series()
        
        for series in sports_series:
            series_ticker = series.get('series_ticker')
            if series_ticker:
                markets = self.get_markets_for_series(series_ticker, limit=1)
                if markets:
                    ticker = markets[0].get('ticker')
                    print(f"✅ Found working sports ticker: {ticker}")
                    return ticker
        
        print("❌ No working market tickers found")
        return None

def main():
    """Test the new sports market discovery approach"""
    print("🚀 SPORTS MARKET DISCOVERY - KALSHI RECOMMENDED APPROACH")
    print("=" * 70)
    print("Testing proper series-first market discovery...")
    print()
    
    discovery = SportsMarketDiscovery()
    
    try:
        # Step 1: Discover sports series
        print("📋 STEP 1: Discover Sports Series")
        print("-" * 40)
        sports_series = discovery.discover_sports_series()
        print()
        
        # Step 2: Focus on NFL series
        print("🏈 STEP 2: Discover NFL Series")
        print("-" * 40)
        nfl_series = discovery.discover_nfl_series()
        print()
        
        # Step 3: Get NFL markets
        print("🎯 STEP 3: Query NFL Markets")
        print("-" * 40)
        nfl_markets = discovery.find_nfl_markets()
        print()
        
        # Step 4: Find current games
        print("🎮 STEP 4: Find Current Games")
        print("-" * 40)
        current_games = discovery.get_current_nfl_games()
        print()
        
        # Step 5: Get working ticker
        print("🔧 STEP 5: Get Working Ticker")
        print("-" * 40)
        working_ticker = discovery.get_working_example_ticker()
        print()
        
        # Summary
        print("📊 SUMMARY")
        print("-" * 40)
        print(f"Sports Series Found: {len(sports_series)}")
        print(f"NFL Series Found: {len(nfl_series)}")
        print(f"NFL Market Series: {len(nfl_markets)}")
        print(f"Current Games: {len(current_games)}")
        print(f"Working Ticker: {working_ticker or 'None'}")
        
        if working_ticker:
            print("\n✅ SUCCESS: Found working markets using proper Kalshi approach!")
            print("🔄 Your Cowboys vs Eagles trader can now use this method.")
        else:
            print("\n⚠️ No current markets found - may be NFL off-season")
            print("🔄 But the discovery method is now properly implemented!")
        
    except Exception as e:
        print(f"❌ Error in discovery: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        discovery.close()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())