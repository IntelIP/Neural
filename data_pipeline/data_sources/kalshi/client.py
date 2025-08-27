"""
Kalshi WebSocket Infrastructure - HTTP Client Module
Provides authenticated HTTP client for Kalshi API
"""

from typing import Any, Dict, Optional, List
import requests
from urllib.parse import urljoin

from .auth import KalshiAuth
from ..config import KalshiConfig


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
            from ..config import get_config
            config = get_config()
        
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
        
        # Get authentication headers
        headers = self.auth.get_auth_headers(method, path)
        
        # Add content type for JSON requests
        if json_data is not None:
            headers['Content-Type'] = 'application/json'
        
        # Build full URL
        url = urljoin(self.base_url, path.lstrip('/'))
        
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
        
        return self.get('/trade-api/v2/markets', params=params)
    
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
        return self.get(f'/trade-api/v2/markets/{ticker}')
    
    def get_events(
        self,
        limit: int = 100,
        cursor: Optional[str] = None,
        status: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get events from the API
        
        Args:
            limit: Number of events to retrieve
            cursor: Pagination cursor
            status: Filter by event status
        
        Returns:
            Events response
        """
        params = {'limit': limit}
        if cursor:
            params['cursor'] = cursor
        if status:
            params['status'] = status
        
        return self.get('/trade-api/v2/events', params=params)
    
    def close(self) -> None:
        """Close the HTTP session"""
        self.session.close()