"""
Kalshi WebSocket Infrastructure - HTTP Client Module
Provides authenticated HTTP client for Kalshi API
"""

from typing import Any, Dict, Optional, List
import requests
from urllib.parse import urljoin

from .auth import KalshiAuth

# Import for type hints - will be redefined in __init__ if needed
try:
    from ...config.settings import KalshiConfig
except ImportError:
    KalshiConfig = None


class KalshiClient:
    """HTTP client for Kalshi API with RSA-PSS authentication"""
    
    def __init__(self, config: Optional[KalshiConfig] = None, auth: Optional[KalshiAuth] = None):
        """
        Initialize HTTP client
        
        Args:
            config: Optional KalshiConfig instance
            auth: Optional KalshiAuth instance
        """
        if config is None:
            # Create basic config from environment
            import os
            from pathlib import Path
            from dotenv import load_dotenv
            
            # Load env
            env_path = Path(__file__).parent.parent.parent.parent.parent / '.env'
            load_dotenv(env_path)
            
            from ...config.settings import KalshiConfig
            
            environment = os.getenv("KALSHI_ENVIRONMENT", "prod")
            # FORCE PRODUCTION ENDPOINTS - NO DEMO ALLOWED
            api_base_url = "https://api.elections.kalshi.com/trade-api/v2"
            
            # Log which endpoint we're using  
            print(f"🏭 Kalshi Client: Forcing production endpoints")
            print(f"🔗 API: {api_base_url}")
            print(f"🌍 Environment variable: {environment}")
            
            # Load private key from file if not in environment
            private_key = os.getenv("KALSHI_PRIVATE_KEY")
            if not private_key:
                private_key_file = os.getenv("KALSHI_PRIVATE_KEY_FILE")
                if private_key_file:
                    with open(private_key_file, 'r') as f:
                        private_key = f.read()
            
            config = KalshiConfig(
                api_key_id=os.getenv("KALSHI_API_KEY_ID"),
                private_key=private_key,
                environment=environment,
                api_base_url=api_base_url
            )
        
        self.config = config
        self.auth = auth or KalshiAuth(config)
        self.base_url = config.api_base_url
        self.session = requests.Session()
    
    def _make_request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None
    ) -> requests.Response:
        """
        Make an authenticated request to the API
        
        Args:
            method: HTTP method
            path: API path
            params: Query parameters
            json_data: JSON body data
        
        Returns:
            Response object
        """
        # Ensure path starts with /
        if not path.startswith('/'):
            path = '/' + path
        
        # Get authentication headers with full path for signing
        # The signature needs the full path including /trade-api/v2
        full_path = path
        if '/trade-api/v2' in self.base_url and not path.startswith('/trade-api/v2'):
            full_path = '/trade-api/v2' + path
        
        headers = self.auth.get_auth_headers(method, full_path)
        
        # Add content type for JSON requests
        if json_data is not None:
            headers['Content-Type'] = 'application/json'
        
        # Build full URL - ensure base_url ends with / for proper joining
        base_url = self.base_url.rstrip('/') + '/'
        url = urljoin(base_url, path.lstrip('/'))
        
        # Make request
        response = self.session.request(
            method=method,
            url=url,
            headers=headers,
            params=params,
            json=json_data
        )
        
        response.raise_for_status()
        return response
    
    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a GET request
        
        Args:
            path: API path
            params: Query parameters
        
        Returns:
            JSON response data
        """
        response = self._make_request('GET', path, params=params)
        return response.json()
    
    def post(self, path: str, json_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a POST request
        
        Args:
            path: API path
            json_data: JSON body data
        
        Returns:
            JSON response data
        """
        response = self._make_request('POST', path, json_data=json_data)
        return response.json()
    
    def get_markets(
        self,
        limit: int = 100,
        cursor: Optional[str] = None,
        event_ticker: Optional[str] = None,
        series_ticker: Optional[str] = None,
        status: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get markets from the API
        
        Args:
            limit: Number of markets to retrieve
            cursor: Pagination cursor
            event_ticker: Filter by event ticker
            series_ticker: Filter by series ticker
            status: Filter by market status
        
        Returns:
            Markets response with 'markets' list and optional 'cursor'
        """
        params = {'limit': limit}
        if cursor:
            params['cursor'] = cursor
        if event_ticker:
            params['event_ticker'] = event_ticker
        if series_ticker:
            params['series_ticker'] = series_ticker
        if status:
            params['status'] = status
        
        return self.get('/markets', params=params)
    
    def get_all_markets(
        self,
        batch_size: int = 100,
        **filters
    ) -> List[Dict[str, Any]]:
        """
        Get all markets using pagination
        
        Args:
            batch_size: Number of markets per request
            **filters: Additional filters for markets
        
        Returns:
            List of all markets
        """
        all_markets = []
        cursor = None
        
        while True:
            response = self.get_markets(
                limit=batch_size,
                cursor=cursor,
                **filters
            )
            
            markets = response.get('markets', [])
            all_markets.extend(markets)
            
            cursor = response.get('cursor')
            if not cursor:
                break
        
        return all_markets
    
    def get_market(self, ticker: str) -> Dict[str, Any]:
        """
        Get a specific market by ticker
        
        Args:
            ticker: Market ticker
        
        Returns:
            Market data
        """
        return self.get(f'/markets/{ticker}')
    
    def get_events(
        self,
        limit: int = 100,
        cursor: Optional[str] = None,
        status: Optional[str] = None,
        series_ticker: Optional[str] = None,
        with_nested_markets: bool = False
    ) -> Dict[str, Any]:
        """
        Get events from the API
        
        Args:
            limit: Number of events to retrieve
            cursor: Pagination cursor
            status: Filter by event status
            series_ticker: Filter by series ticker
            with_nested_markets: Include nested market data in response
        
        Returns:
            Events response
        """
        params = {'limit': limit}
        if cursor:
            params['cursor'] = cursor
        if status:
            params['status'] = status
        if series_ticker:
            params['series_ticker'] = series_ticker
        if with_nested_markets:
            params['with_nested_markets'] = True
        
        return self.get('/events', params=params)
    
    def get_events_with_markets(
        self,
        series_ticker: Optional[str] = None,
        status: str = 'open'
    ) -> List[Dict[str, Any]]:
        """
        Get events with their nested markets
        
        Args:
            series_ticker: Filter by series ticker (e.g., 'KXNFLGAME')
            status: Event status filter
        
        Returns:
            List of events with nested markets
        """
        response = self.get_events(
            limit=200,
            status=status,
            series_ticker=series_ticker,
            with_nested_markets=True
        )
        return response.get('events', [])
    
    def get_series(
        self,
        limit: int = 100,
        cursor: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get series from the API
        
        Args:
            limit: Number of series to retrieve
            cursor: Pagination cursor
            category: Filter by series category (deprecated - use tags instead)
            tags: Filter by series tags (e.g., 'NFL', 'NBA', 'CFP') - RECOMMENDED
        
        Returns:
            Series response with 'series' list and optional 'cursor'
        """
        params = {'limit': limit}
        if cursor:
            params['cursor'] = cursor
        if category:
            params['category'] = category
        if tags:
            params['tags'] = tags
        
        return self.get('/series', params=params)
    
    def get_all_series(
        self,
        batch_size: int = 100,
        **filters
    ) -> List[Dict[str, Any]]:
        """
        Get all series using pagination
        
        Args:
            batch_size: Number of series per request
            **filters: Additional filters for series (e.g., category)
        
        Returns:
            List of all series
        """
        all_series = []
        cursor = None
        
        while True:
            response = self.get_series(
                limit=batch_size,
                cursor=cursor,
                **filters
            )
            
            series = response.get('series', [])
            all_series.extend(series)
            
            cursor = response.get('cursor')
            if not cursor:
                break
        
        return all_series
    
    def get_nfl_series(self) -> List[Dict[str, Any]]:
        """
        Get NFL/Pro Football series using proper Kalshi tags filtering.
        
        Returns:
            List of NFL series with series_ticker and metadata
        """
        try:
            print("🏈 Getting NFL series using tags='Football' (CORRECTED Kalshi approach)")
            
            # CORRECTED: Kalshi uses 'Football' tag, not 'NFL'
            football_series = self.get_all_series(tags='Football')
            
            print(f"✅ Found {len(football_series)} NFL/Football series using correct tags filter")
            
            return football_series
            
        except Exception as e:
            print(f"❌ Error getting Football series with tags: {e}")
            return []
    
    def get_nba_series(self) -> List[Dict[str, Any]]:
        """
        Get NBA/Basketball series using proper Kalshi tags filtering.
        
        Returns:
            List of NBA series with series_ticker and metadata
        """
        try:
            print("🏀 Getting NBA series using tags='Basketball' (CORRECTED Kalshi approach)")
            # CORRECTED: Kalshi uses 'Basketball' tag, not 'NBA'
            return self.get_all_series(tags='Basketball')
        except Exception as e:
            print(f"❌ Error getting Basketball series: {e}")
            return []
    
    def get_mlb_series(self) -> List[Dict[str, Any]]:
        """
        Get MLB/Baseball series using proper Kalshi tags filtering.
        
        Returns:
            List of MLB series with series_ticker and metadata
        """
        try:
            print("⚾ Getting MLB series using tags='Baseball' (CORRECTED Kalshi approach)")
            # CORRECTED: Kalshi uses 'Baseball' tag, not 'MLB'
            return self.get_all_series(tags='Baseball')
        except Exception as e:
            print(f"❌ Error getting Baseball series: {e}")
            return []
    
    def get_soccer_series(self) -> List[Dict[str, Any]]:
        """
        Get Soccer series using proper Kalshi tags filtering.
        
        Returns:
            List of Soccer series with series_ticker and metadata
        """
        try:
            print("⚽ Getting Soccer series using tags='Soccer'")
            return self.get_all_series(tags='Soccer')
        except Exception as e:
            print(f"❌ Error getting Soccer series: {e}")
            return []
    
    def get_sports_series(self, leagues: List[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get sports series for multiple leagues using CORRECTED Kalshi tags filtering.
        
        Args:
            leagues: List of user-friendly league names (e.g., ['NFL', 'NBA', 'MLB', 'Soccer'])
                    Defaults to ['NFL', 'NBA', 'MLB', 'Soccer']
        
        Returns:
            Dict with user league name as key and list of series as value
        """
        if leagues is None:
            leagues = ['NFL', 'NBA', 'MLB', 'Soccer']
        
        print(f"🏟️ Getting sports series for leagues: {leagues} (using corrected Kalshi tags)")
        
        # Map user-friendly names to actual Kalshi tags
        tag_mapping = {
            'NFL': 'Football',      # CORRECTED: NFL uses 'Football' tag
            'NBA': 'Basketball',    # CORRECTED: NBA uses 'Basketball' tag  
            'MLB': 'Baseball',      # CORRECTED: MLB uses 'Baseball' tag
            'Soccer': 'Soccer',     # Soccer uses 'Soccer' tag
            'Hockey': 'Hockey',     # Hockey uses 'Hockey' tag
            'Tennis': 'Tennis'      # Tennis uses 'Tennis' tag
        }
        
        sports_data = {}
        
        for league in leagues:
            actual_tag = tag_mapping.get(league, league)
            
            try:
                print(f"📊 Getting {league} series using tags='{actual_tag}'...")
                league_series = self.get_all_series(tags=actual_tag)
                sports_data[league] = league_series
                
                print(f"✅ {league}: {len(league_series)} series found")
                
                # Show sample series for debugging
                for series in league_series[:2]:
                    ticker = series.get('ticker', 'Unknown')
                    title = series.get('title', 'No title')
                    print(f"   • {ticker} - {title}")
                    
            except Exception as e:
                print(f"❌ Error getting {league} series: {e}")
                sports_data[league] = []
        
        total_series = sum(len(series_list) for series_list in sports_data.values())
        print(f"🎯 Total sports series found: {total_series}")
        
        return sports_data
    
    def close(self) -> None:
        """Close the HTTP session"""
        self.session.close()
