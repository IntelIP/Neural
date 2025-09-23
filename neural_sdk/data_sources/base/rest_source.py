"""
REST Data Source Base Class

Abstract base class for all REST API data sources in Neural SDK.
Provides common functionality for authentication, rate limiting,
caching, and error handling.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
import asyncio
import logging
from datetime import datetime, timedelta
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

from .rate_limiter import RateLimiter
from .cache import ResponseCache
from .auth_strategies import AuthStrategy, NoAuth

logger = logging.getLogger(__name__)


class RESTDataSource(ABC):
    """
    Abstract base class for REST API data sources.
    
    Provides:
    - Automatic authentication
    - Rate limiting
    - Response caching
    - Retry logic with exponential backoff
    - Error handling
    - Response transformation
    """
    
    def __init__(
        self,
        base_url: str,
        name: str = None,
        auth_strategy: AuthStrategy = None,
        timeout: int = 30,
        cache_ttl: int = 60,
        rate_limit: int = 10,  # requests per second
        max_retries: int = 3
    ):
        """
        Initialize REST data source.
        
        Args:
            base_url: Base URL for the API
            name: Name of the data source
            auth_strategy: Authentication strategy to use
            timeout: Request timeout in seconds
            cache_ttl: Cache time-to-live in seconds
            rate_limit: Maximum requests per second
            max_retries: Maximum number of retry attempts
        """
        self.base_url = base_url.rstrip('/')
        self.name = name or self.__class__.__name__
        self.auth_strategy = auth_strategy or NoAuth()
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Initialize components
        self.cache = ResponseCache(ttl=cache_ttl)
        self.rate_limiter = RateLimiter(requests_per_second=rate_limit)
        self.session: Optional[httpx.AsyncClient] = None
        
        # Statistics
        self.stats = {
            'requests': 0,
            'cache_hits': 0,
            'errors': 0,
            'total_latency': 0
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
    
    async def connect(self):
        """Initialize HTTP session."""
        if not self.session:
            self.session = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout
            )
            logger.info(f"{self.name}: Connected to {self.base_url}")
    
    async def disconnect(self):
        """Close HTTP session."""
        if self.session:
            await self.session.aclose()
            self.session = None
            logger.info(f"{self.name}: Disconnected")
    
    @abstractmethod
    async def transform_response(self, data: Any, endpoint: str) -> Dict:
        """
        Transform API response to standardized format.
        
        Args:
            data: Raw response data
            endpoint: The endpoint that was called
            
        Returns:
            Standardized response dictionary
        """
        pass
    
    @abstractmethod
    async def validate_response(self, response: httpx.Response) -> bool:
        """
        Validate that the response is successful.
        
        Args:
            response: HTTP response object
            
        Returns:
            True if response is valid, False otherwise
        """
        pass
    
    def _get_cache_key(self, method: str, endpoint: str, params: Dict = None) -> str:
        """Generate cache key for request."""
        param_str = "&".join(f"{k}={v}" for k, v in sorted((params or {}).items()))
        return f"{self.name}:{method}:{endpoint}:{param_str}"
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError))
    )
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None,
        use_cache: bool = True
    ) -> Any:
        """
        Make HTTP request with retry logic.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            json_data: JSON body data
            use_cache: Whether to use caching
            
        Returns:
            Response data
        """
        # Check cache for GET requests
        cache_key = self._get_cache_key(method, endpoint, params)
        if use_cache and method == "GET":
            cached = self.cache.get(cache_key)
            if cached is not None:
                self.stats['cache_hits'] += 1
                logger.debug(f"{self.name}: Cache hit for {endpoint}")
                return cached
        
        # Apply rate limiting
        await self.rate_limiter.acquire()
        
        # Get authentication headers
        auth_headers = await self.auth_strategy.get_headers(method, endpoint)
        
        # Make request
        start_time = datetime.utcnow()
        self.stats['requests'] += 1
        
        try:
            response = await self.session.request(
                method=method,
                url=endpoint,
                params=params,
                json=json_data,
                headers=auth_headers
            )
            
            # Calculate latency
            latency = (datetime.utcnow() - start_time).total_seconds()
            self.stats['total_latency'] += latency
            
            # Validate response
            if not await self.validate_response(response):
                raise ValueError(f"Invalid response from {endpoint}: {response.status_code}")
            
            # Parse response
            data = response.json() if response.content else {}
            
            # Cache successful GET responses
            if use_cache and method == "GET":
                self.cache.set(cache_key, data)
            
            return data
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"{self.name}: Error fetching {endpoint}: {e}")
            raise
    
    async def fetch(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        use_cache: bool = True,
        transform: bool = True
    ) -> Dict:
        """
        Fetch data from REST endpoint.
        
        Args:
            endpoint: API endpoint (relative to base_url)
            params: Query parameters
            use_cache: Whether to use caching
            transform: Whether to transform response
            
        Returns:
            Transformed response data
        """
        if not self.session:
            await self.connect()
        
        # Ensure endpoint starts with /
        if not endpoint.startswith('/'):
            endpoint = '/' + endpoint
        
        # Make request
        data = await self._make_request(
            method="GET",
            endpoint=endpoint,
            params=params,
            use_cache=use_cache
        )
        
        # Transform response if requested
        if transform:
            return await self.transform_response(data, endpoint)
        
        return data
    
    async def post(
        self,
        endpoint: str,
        json_data: Optional[Dict] = None,
        params: Optional[Dict] = None,
        transform: bool = True
    ) -> Dict:
        """
        Post data to REST endpoint.
        
        Args:
            endpoint: API endpoint
            json_data: JSON body data
            params: Query parameters
            transform: Whether to transform response
            
        Returns:
            Response data
        """
        if not self.session:
            await self.connect()
        
        if not endpoint.startswith('/'):
            endpoint = '/' + endpoint
        
        data = await self._make_request(
            method="POST",
            endpoint=endpoint,
            params=params,
            json_data=json_data,
            use_cache=False
        )
        
        if transform:
            return await self.transform_response(data, endpoint)
        
        return data
    
    async def batch_fetch(
        self,
        requests: List[Dict[str, Any]],
        max_concurrent: int = 5
    ) -> List[Dict]:
        """
        Fetch multiple endpoints concurrently.
        
        Args:
            requests: List of request configurations
            max_concurrent: Maximum concurrent requests
            
        Returns:
            List of responses
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def fetch_with_semaphore(request):
            async with semaphore:
                return await self.fetch(
                    endpoint=request['endpoint'],
                    params=request.get('params'),
                    use_cache=request.get('use_cache', True)
                )
        
        tasks = [fetch_with_semaphore(req) for req in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_stats(self) -> Dict:
        """Get data source statistics."""
        avg_latency = (
            self.stats['total_latency'] / self.stats['requests']
            if self.stats['requests'] > 0 else 0
        )
        
        cache_hit_rate = (
            self.stats['cache_hits'] / self.stats['requests']
            if self.stats['requests'] > 0 else 0
        )
        
        return {
            'name': self.name,
            'total_requests': self.stats['requests'],
            'cache_hits': self.stats['cache_hits'],
            'cache_hit_rate': cache_hit_rate,
            'errors': self.stats['errors'],
            'average_latency': avg_latency
        }
    
    async def health_check(self) -> bool:
        """
        Check if the data source is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Make a simple request to check connectivity
            await self.fetch("/", use_cache=False)
            return True
        except Exception as e:
            logger.error(f"{self.name}: Health check failed: {e}")
            return False