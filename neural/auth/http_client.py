"""
HTTP Client with Kalshi Authentication

Provides an authenticated HTTP client for making requests to the Kalshi API
using the KalshiSigner for request signing.
"""

import requests
from typing import Optional, Dict, Any
import logging
from time import sleep
from urllib.parse import urljoin

from neural.auth.signers.kalshi import KalshiSigner
from neural.auth.env import get_api_key_id, get_private_key_material, get_base_url

logger = logging.getLogger(__name__)


class KalshiHTTPClient:
    """
    HTTP client for authenticated Kalshi API requests.

    This client handles:
    - Request signing using KalshiSigner
    - Automatic retry on rate limits
    - Error handling and logging
    """

    def __init__(
        self,
        api_key_id: Optional[str] = None,
        private_key_pem: Optional[bytes] = None,
        base_url: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize the Kalshi HTTP client.

        Args:
            api_key_id: Kalshi API key ID (defaults to env)
            private_key_pem: RSA private key PEM bytes (defaults to env)
            base_url: API base URL (defaults to production)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries on failure
        """
        # Load credentials
        self.api_key_id = api_key_id or get_api_key_id()
        self.private_key_pem = private_key_pem or get_private_key_material()
        self.base_url = base_url or get_base_url()

        # Initialize signer
        self.signer = KalshiSigner(self.api_key_id, self.private_key_pem)

        # HTTP session
        self.session = requests.Session()
        self.timeout = timeout
        self.max_retries = max_retries

        logger.info(f"Initialized KalshiHTTPClient for {self.base_url}")

    def _make_request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        retry_count: int = 0
    ) -> Dict[str, Any]:
        """
        Make an authenticated HTTP request to Kalshi API.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: API endpoint path
            params: Query parameters
            json_data: JSON body data
            retry_count: Current retry attempt

        Returns:
            Response data as dictionary
        """
        # Ensure path starts with /
        if not path.startswith('/'):
            path = f'/{path}'

        # Build full URL
        url = urljoin(self.base_url, f"/trade-api/v2{path}")

        # Get authentication headers
        auth_headers = self.signer.headers(method, f"/trade-api/v2{path}")

        # Prepare headers
        headers = {
            **auth_headers,
            'Content-Type': 'application/json'
        }

        try:
            # Make request
            logger.debug(f"{method} {url} with params: {params}")
            response = self.session.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                json=json_data,
                timeout=self.timeout
            )

            # Handle rate limiting
            if response.status_code == 429:
                if retry_count < self.max_retries:
                    # Get retry-after header if available
                    retry_after = int(response.headers.get('Retry-After', 2))
                    logger.warning(f"Rate limited, retrying after {retry_after} seconds...")
                    sleep(retry_after)
                    return self._make_request(
                        method, path, params, json_data, retry_count + 1
                    )
                else:
                    logger.error(f"Max retries exceeded for {method} {path}")
                    response.raise_for_status()

            # Check for other errors
            if response.status_code >= 400:
                logger.error(f"API error {response.status_code}: {response.text}")
                response.raise_for_status()

            # Parse JSON response
            return response.json()

        except requests.exceptions.Timeout:
            if retry_count < self.max_retries:
                logger.warning(f"Request timeout, retry {retry_count + 1}/{self.max_retries}")
                sleep(2 ** retry_count)  # Exponential backoff
                return self._make_request(
                    method, path, params, json_data, retry_count + 1
                )
            else:
                logger.error(f"Request timeout after {self.max_retries} retries")
                raise

        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a GET request to the Kalshi API.

        Args:
            path: API endpoint path
            params: Query parameters

        Returns:
            Response data as dictionary
        """
        return self._make_request('GET', path, params=params)

    def post(
        self,
        path: str,
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make a POST request to the Kalshi API.

        Args:
            path: API endpoint path
            json_data: JSON body data
            params: Query parameters

        Returns:
            Response data as dictionary
        """
        return self._make_request('POST', path, params=params, json_data=json_data)

    def get_trades(
        self,
        ticker: str,
        min_ts: Optional[int] = None,
        max_ts: Optional[int] = None,
        limit: int = 1000,
        cursor: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get historical trades for a market.

        Args:
            ticker: Market ticker
            min_ts: Minimum timestamp (Unix seconds)
            max_ts: Maximum timestamp (Unix seconds)
            limit: Number of trades per page (max 1000)
            cursor: Pagination cursor

        Returns:
            API response with trades data
        """
        params = {
            'ticker': ticker,
            'limit': min(limit, 1000)
        }

        if min_ts is not None:
            params['min_ts'] = min_ts
        if max_ts is not None:
            params['max_ts'] = max_ts
        if cursor is not None:
            params['cursor'] = cursor

        return self.get('/markets/trades', params=params)

    def get_market_candlesticks(
        self,
        series_ticker: str,
        ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int
    ) -> Dict[str, Any]:
        """
        Get candlestick data for a specific market.

        Args:
            series_ticker: Series ticker
            ticker: Market ticker within series
            start_ts: Start timestamp
            end_ts: End timestamp
            period_interval: Time interval in minutes (1, 60, or 1440)

        Returns:
            API response with candlestick data
        """
        path = f'/series/{series_ticker}/markets/{ticker}/candlesticks'
        params = {
            'start_ts': start_ts,
            'end_ts': end_ts,
            'period_interval': period_interval
        }

        return self.get(path, params=params)

    def get_event_candlesticks(
        self,
        ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int
    ) -> Dict[str, Any]:
        """
        Get aggregated candlestick data for an event.

        Args:
            ticker: Event ticker
            start_ts: Start timestamp
            end_ts: End timestamp
            period_interval: Time interval in minutes (1, 60, or 1440)

        Returns:
            API response with event candlestick data
        """
        path = f'/events/{ticker}/candlesticks'
        params = {
            'start_ts': start_ts,
            'end_ts': end_ts,
            'period_interval': period_interval
        }

        return self.get(path, params=params)

    def close(self):
        """Close the HTTP session."""
        self.session.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()