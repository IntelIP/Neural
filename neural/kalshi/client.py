"""
Kalshi API client for Neural SDK.

This module provides a high-level interface to the Kalshi REST API,
properly integrating with the REST adapter from the main branch.
"""

import os
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class KalshiClient:
    """
    High-level Kalshi API client that wraps the REST adapter.
    
    This client provides methods to fetch real market data from Kalshi,
    including CFB and NFL markets with proper ticker formats.
    """
    
    def __init__(self, api_key: Optional[str] = None, private_key_path: Optional[str] = None):
        """
        Initialize Kalshi client.
        
        Args:
            api_key: Kalshi API key (defaults to KALSHI_API_KEY env var)
            private_key_path: Path to RSA private key (defaults to KALSHI_PRIVATE_KEY_PATH env var)
        """
        self.api_key = api_key or os.getenv('KALSHI_API_KEY')
        self.private_key_path = private_key_path or os.getenv('KALSHI_PRIVATE_KEY_PATH')
        
        if not self.api_key or not self.private_key_path:
            raise ValueError(
                "Kalshi API credentials required. Set KALSHI_API_KEY and KALSHI_PRIVATE_KEY_PATH "
                "environment variables or pass them as arguments."
            )
        
        # Import here to avoid circular dependency
        from neural_sdk.data_sources.kalshi import KalshiRESTAdapter
        
        self.adapter = KalshiRESTAdapter(
            api_key=self.api_key,
            private_key_path=self.private_key_path
        )
        
    async def get_cfb_markets(self, week: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get College Football markets from Kalshi.
        
        Args:
            week: Optional week number to filter by
            
        Returns:
            List of CFB market dictionaries with real Kalshi tickers
        """
        try:
            markets = await self.adapter.get_cfb_markets(week=week)
            
            # Process and return markets with proper structure
            processed_markets = []
            for market in markets:
                processed_markets.append({
                    'ticker': market.get('ticker'),  # Real format: NCAAFSPREAD-24DEC14-MICHIGAN-OHIOSTATE-7
                    'title': market.get('title'),
                    'yes_price': market.get('yes_ask'),
                    'no_price': market.get('no_ask'),
                    'volume': market.get('volume'),
                    'open_interest': market.get('open_interest'),
                    'expiration': market.get('expiration_time'),
                    'status': market.get('status'),
                    'spread': self._extract_spread_from_ticker(market.get('ticker', '')),
                    'teams': self._extract_teams_from_ticker(market.get('ticker', ''))
                })
            
            return processed_markets
            
        except Exception as e:
            logger.error(f"Error fetching CFB markets: {e}")
            raise
    
    async def get_nfl_markets(self, week: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get NFL markets from Kalshi.
        
        Args:
            week: Optional week number to filter by
            
        Returns:
            List of NFL market dictionaries with real Kalshi tickers
        """
        try:
            markets = await self.adapter.get_nfl_markets(week=week)
            
            # Process and return markets with proper structure
            processed_markets = []
            for market in markets:
                processed_markets.append({
                    'ticker': market.get('ticker'),  # Real format: NFLWIN-24DEC15-COWBOYS-EAGLES
                    'title': market.get('title'),
                    'yes_price': market.get('yes_ask'),
                    'no_price': market.get('no_ask'),
                    'volume': market.get('volume'),
                    'open_interest': market.get('open_interest'),
                    'expiration': market.get('expiration_time'),
                    'status': market.get('status'),
                    'teams': self._extract_teams_from_ticker(market.get('ticker', ''))
                })
            
            return processed_markets
            
        except Exception as e:
            logger.error(f"Error fetching NFL markets: {e}")
            raise
    
    async def get_market_by_ticker(self, ticker: str) -> Dict[str, Any]:
        """
        Get a specific market by its ticker.
        
        Args:
            ticker: Kalshi market ticker (e.g., 'NCAAFSPREAD-24DEC14-MICHIGAN-OHIOSTATE-7')
            
        Returns:
            Market dictionary with full details
        """
        try:
            market = await self.adapter.get_market(ticker)
            
            return {
                'ticker': market.get('ticker'),
                'title': market.get('title'),
                'yes_price': market.get('yes_ask'),
                'no_price': market.get('no_ask'),
                'yes_bid': market.get('yes_bid'),
                'no_bid': market.get('no_bid'),
                'volume': market.get('volume'),
                'open_interest': market.get('open_interest'),
                'expiration': market.get('expiration_time'),
                'status': market.get('status'),
                'result': market.get('result'),
                'can_close_early': market.get('can_close_early'),
                'estimated_close_time': market.get('estimated_close_time')
            }
            
        except Exception as e:
            logger.error(f"Error fetching market {ticker}: {e}")
            raise
    
    async def get_market_orderbook(self, ticker: str) -> Dict[str, Any]:
        """
        Get the orderbook for a specific market.
        
        Args:
            ticker: Kalshi market ticker
            
        Returns:
            Orderbook with bids and asks
        """
        try:
            orderbook = await self.adapter.get_orderbook(ticker)
            return orderbook
            
        except Exception as e:
            logger.error(f"Error fetching orderbook for {ticker}: {e}")
            raise
    
    async def get_market_history(self, ticker: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get trade history for a specific market.
        
        Args:
            ticker: Kalshi market ticker
            limit: Maximum number of trades to return
            
        Returns:
            List of historical trades
        """
        try:
            history = await self.adapter.get_trades(
                ticker=ticker,
                limit=limit
            )
            return history
            
        except Exception as e:
            logger.error(f"Error fetching history for {ticker}: {e}")
            raise
    
    async def search_markets(self, query: str, status: str = 'open') -> List[Dict[str, Any]]:
        """
        Search for markets by query string.
        
        Args:
            query: Search query
            status: Market status filter ('open', 'closed', 'settled')
            
        Returns:
            List of matching markets
        """
        try:
            events = await self.adapter._paginate_events(
                status=status,
                with_nested_markets=True,
                limit_per_page=200,
                max_pages=5
            )
            
            # Filter markets by query
            matching_markets = []
            query_lower = query.lower()
            
            for event in events:
                for market in event.get('markets', []):
                    if (query_lower in market.get('ticker', '').lower() or
                        query_lower in market.get('title', '').lower()):
                        matching_markets.append({
                            'ticker': market.get('ticker'),
                            'title': market.get('title'),
                            'yes_price': market.get('yes_ask'),
                            'no_price': market.get('no_ask'),
                            'volume': market.get('volume'),
                            'status': market.get('status')
                        })
            
            return matching_markets
            
        except Exception as e:
            logger.error(f"Error searching markets: {e}")
            raise
    
    def _extract_spread_from_ticker(self, ticker: str) -> Optional[float]:
        """Extract spread value from CFB ticker format."""
        if 'SPREAD' in ticker:
            parts = ticker.split('-')
            for part in parts:
                if part and part[-1].isdigit():
                    try:
                        # Handle both positive and negative spreads
                        return float(part.lstrip('SPREAD'))
                    except ValueError:
                        continue
        return None
    
    def _extract_teams_from_ticker(self, ticker: str) -> Dict[str, str]:
        """Extract team names from ticker format."""
        parts = ticker.split('-')
        
        # For CFB: NCAAFSPREAD-24DEC14-MICHIGAN-OHIOSTATE-7
        # For NFL: NFLWIN-24DEC15-COWBOYS-EAGLES
        
        if len(parts) >= 4:
            if 'NCAAF' in parts[0]:
                # Skip date and get teams
                return {
                    'away': parts[2] if len(parts) > 2 else '',
                    'home': parts[3] if len(parts) > 3 else ''
                }
            elif 'NFL' in parts[0]:
                return {
                    'away': parts[2] if len(parts) > 2 else '',
                    'home': parts[3] if len(parts) > 3 else ''
                }
        
        return {'away': '', 'home': ''}
    
    async def close(self):
        """Close the adapter connection."""
        if hasattr(self.adapter, 'close'):
            await self.adapter.close()


def get_kalshi_client(api_key: Optional[str] = None, 
                      private_key_path: Optional[str] = None) -> KalshiClient:
    """
    Factory function to create a Kalshi client.
    
    Args:
        api_key: Optional Kalshi API key
        private_key_path: Optional path to RSA private key
        
    Returns:
        Configured KalshiClient instance
    """
    return KalshiClient(api_key=api_key, private_key_path=private_key_path)