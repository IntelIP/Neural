"""
REST API data source implementation for the Neural SDK.

This module provides a robust REST API client that handles HTTP requests
with features like automatic retries, rate limiting, response caching,
and pagination support.
"""

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from urllib.parse import urljoin, urlparse, parse_qs, urlencode
import aiohttp
from aiohttp import ClientSession, ClientResponse
import logging

from neural.data_collection.base import BaseDataSource, DataSourceConfig, ConnectionState
from neural.data_collection.exceptions import (
    ConnectionError,
    RateLimitError,
    AuthenticationError,
    TimeoutError,
    DataSourceError,
    ValidationError,
    TransientError
)


# Configure module logger
logger = logging.getLogger(__name__)


class HttpMethod(Enum):
    """HTTP methods supported by the REST handler."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


@dataclass
class RestConfig(DataSourceConfig):
    """
    Configuration specific to REST API data sources.
    
    Extends the base DataSourceConfig with REST-specific parameters
    for HTTP requests, authentication, rate limiting, and caching.
    
    Attributes:
        base_url: Base URL for the REST API
        headers: Default headers to include in all requests
        auth_type: Authentication type ("bearer", "api_key", "basic", "custom")
        auth_credentials: Authentication credentials
        rate_limit_requests: Maximum requests per second
        rate_limit_burst: Burst capacity for rate limiting
        cache_enabled: Whether to enable response caching
        cache_ttl: Cache time-to-live in seconds
        pagination_type: Type of pagination ("offset", "cursor", "page", "none")
        pagination_params: Parameters for pagination
        
    Example:
        >>> config = RestConfig(
        ...     name="espn_api",
        ...     base_url="https://api.espn.com/v1",
        ...     headers={"Accept": "application/json"},
        ...     rate_limit_requests=10,
        ...     cache_enabled=True
        ... )
    """
    base_url: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    auth_type: Optional[str] = None
    auth_credentials: Dict[str, str] = field(default_factory=dict)
    rate_limit_requests: float = 10.0  # Requests per second
    rate_limit_burst: int = 20  # Burst capacity
    cache_enabled: bool = False
    cache_ttl: int = 300  # 5 minutes default
    pagination_type: str = "none"  # "offset", "cursor", "page", "none"
    pagination_params: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> None:
        """
        Validate REST-specific configuration.
        
        Raises:
            ConfigurationError: If configuration is invalid
        """
        # Call parent validation
        super().validate()
        
        # Validate REST-specific fields
        if not self.base_url:
            raise ConfigurationError("REST base_url cannot be empty")
        
        # Validate URL format
        parsed = urlparse(self.base_url)
        if not parsed.scheme or not parsed.netloc:
            raise ConfigurationError(
                "Invalid base_url format",
                details={"url": self.base_url}
            )
        
        if self.rate_limit_requests <= 0:
            raise ConfigurationError(
                "rate_limit_requests must be positive",
                details={"value": self.rate_limit_requests}
            )
        
        if self.rate_limit_burst < 1:
            raise ConfigurationError(
                "rate_limit_burst must be at least 1",
                details={"value": self.rate_limit_burst}
            )
        
        if self.cache_ttl < 0:
            raise ConfigurationError(
                "cache_ttl must be non-negative",
                details={"value": self.cache_ttl}
            )
        
        # Validate auth type
        valid_auth_types = ["bearer", "api_key", "basic", "custom", None]
        if self.auth_type not in valid_auth_types:
            raise ConfigurationError(
                f"Invalid auth_type: {self.auth_type}",
                details={"valid_types": valid_auth_types}
            )
        
        # Validate pagination type
        valid_pagination = ["offset", "cursor", "page", "none"]
        if self.pagination_type not in valid_pagination:
            raise ConfigurationError(
                f"Invalid pagination_type: {self.pagination_type}",
                details={"valid_types": valid_pagination}
            )


class RateLimiter:
    """
    Token bucket rate limiter for REST API requests.
    
    This class implements a token bucket algorithm to enforce
    rate limits on API requests.
    
    Attributes:
        rate: Requests per second
        burst: Maximum burst capacity
        tokens: Current available tokens
        last_update: Last token update time
    """
    
    def __init__(self, rate: float, burst: int):
        """
        Initialize the rate limiter.
        
        Args:
            rate: Requests per second
            burst: Maximum burst capacity
        """
        self.rate = rate
        self.burst = burst
        self.tokens = burst
        self.last_update = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> float:
        """
        Acquire tokens from the bucket.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            float: Time to wait before request can proceed
        """
        async with self._lock:
            now = time.time()
            
            # Calculate tokens accumulated since last update
            elapsed = now - self.last_update
            self.tokens = min(
                self.burst,
                self.tokens + elapsed * self.rate
            )
            self.last_update = now
            
            # Check if we have enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return 0.0  # No wait needed
            
            # Calculate wait time
            tokens_needed = tokens - self.tokens
            wait_time = tokens_needed / self.rate
            
            # Reserve the tokens
            self.tokens = 0
            
            return wait_time


class ResponseCache:
    """
    Simple in-memory cache for REST API responses.
    
    This class provides a basic caching mechanism to reduce
    redundant API calls.
    
    Attributes:
        ttl: Time-to-live for cache entries in seconds
        cache: Dictionary storing cached responses
    """
    
    def __init__(self, ttl: int = 300):
        """
        Initialize the response cache.
        
        Args:
            ttl: Time-to-live for cache entries in seconds
        """
        self.ttl = ttl
        self.cache: Dict[str, tuple[Any, datetime]] = {}
    
    def _get_cache_key(
        self,
        method: str,
        url: str,
        params: Optional[Dict] = None,
        data: Optional[Any] = None
    ) -> str:
        """
        Generate a cache key for a request.
        
        Args:
            method: HTTP method
            url: Request URL
            params: Query parameters
            data: Request body data
            
        Returns:
            str: Cache key
        """
        # Create a unique key from request parameters
        key_parts = [method, url]
        
        if params:
            # Sort params for consistent keys
            sorted_params = sorted(params.items())
            key_parts.append(str(sorted_params))
        
        if data:
            if isinstance(data, dict):
                sorted_data = sorted(data.items())
                key_parts.append(str(sorted_data))
            else:
                key_parts.append(str(data))
        
        # Hash the key for consistent length
        key_str = "|".join(key_parts)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def get(
        self,
        method: str,
        url: str,
        params: Optional[Dict] = None,
        data: Optional[Any] = None
    ) -> Optional[Any]:
        """
        Get a cached response if available and not expired.
        
        Args:
            method: HTTP method
            url: Request URL
            params: Query parameters
            data: Request body data
            
        Returns:
            Cached response or None
        """
        key = self._get_cache_key(method, url, params, data)
        
        if key in self.cache:
            response, timestamp = self.cache[key]
            
            # Check if cache entry is still valid
            if datetime.now() - timestamp < timedelta(seconds=self.ttl):
                logger.debug(f"Cache hit for {method} {url}")
                return response
            else:
                # Remove expired entry
                del self.cache[key]
                logger.debug(f"Cache expired for {method} {url}")
        
        return None
    
    def set(
        self,
        method: str,
        url: str,
        response: Any,
        params: Optional[Dict] = None,
        data: Optional[Any] = None
    ) -> None:
        """
        Store a response in the cache.
        
        Args:
            method: HTTP method
            url: Request URL
            response: Response to cache
            params: Query parameters
            data: Request body data
        """
        key = self._get_cache_key(method, url, params, data)
        self.cache[key] = (response, datetime.now())
        logger.debug(f"Cached response for {method} {url}")
    
    def clear(self) -> None:
        """Clear all cached responses."""
        self.cache.clear()
        logger.debug("Cache cleared")


class RestDataSource(BaseDataSource):
    """
    REST API implementation of a data source.
    
    This class provides a complete REST client with features like:
    - Automatic retry with exponential backoff
    - Rate limiting with token bucket algorithm
    - Response caching
    - Pagination support
    - Authentication handling
    - Request/response hooks
    
    The class handles all HTTP operations and provides a simple
    interface for making API requests.
    
    Attributes:
        config: REST configuration
        session: aiohttp client session
        rate_limiter: Token bucket rate limiter
        cache: Response cache
        
    Example:
        >>> config = RestConfig(
        ...     name="api_client",
        ...     base_url="https://api.example.com"
        ... )
        >>> source = RestDataSource(config)
        >>> 
        >>> # Make a request
        >>> await source.connect()
        >>> data = await source.get("/users", params={"limit": 10})
    """
    
    def __init__(self, config: RestConfig):
        """
        Initialize the REST data source.
        
        Args:
            config: REST configuration
        """
        super().__init__(config)
        self.config: RestConfig = config
        
        # HTTP session
        self.session: Optional[ClientSession] = None
        
        # Rate limiting
        self.rate_limiter = RateLimiter(
            config.rate_limit_requests,
            config.rate_limit_burst
        )
        
        # Response caching
        self.cache = ResponseCache(config.cache_ttl) if config.cache_enabled else None
        
        # Request statistics
        self._request_count = 0
        self._cache_hits = 0
        
        logger.info(f"Initialized RestDataSource: {config.name}")
    
    async def _connect_impl(self) -> bool:
        """
        Establish HTTP session for REST API requests.
        
        Returns:
            bool: True if session created successfully
        """
        try:
            # Create session with timeout
            timeout = aiohttp.ClientTimeout(
                total=self.config.timeout,
                connect=self.config.timeout / 2
            )
            
            # Build default headers
            headers = self.config.headers.copy()
            
            # Add authentication headers
            auth_headers = self._build_auth_headers()
            headers.update(auth_headers)
            
            self.session = ClientSession(
                base_url=self.config.base_url,
                headers=headers,
                timeout=timeout
            )
            
            logger.info(f"{self.config.name}: REST session created")
            return True
            
        except Exception as e:
            logger.error(f"{self.config.name}: Failed to create session: {e}")
            raise ConnectionError(
                f"Failed to create REST session: {e}",
                details={"base_url": self.config.base_url}
            )
    
    async def _disconnect_impl(self) -> None:
        """Close HTTP session and cleanup resources."""
        if self.session and not self.session.closed:
            await self.session.close()
            # Small delay to allow graceful closure
            await asyncio.sleep(0.1)
            logger.debug(f"{self.config.name}: Session closed")
        
        self.session = None
        
        # Clear cache if enabled
        if self.cache:
            self.cache.clear()
        
        logger.info(f"{self.config.name}: Disconnected and cleaned up")
    
    async def _subscribe_impl(self, channels: List[str]) -> bool:
        """
        REST doesn't have subscriptions, but we can use this for
        setting up polling or webhook endpoints.
        
        Args:
            channels: List of resources to poll
            
        Returns:
            bool: Always True for REST
        """
        # Store channels for potential polling implementation
        # Subclasses can override to implement polling
        logger.debug(f"{self.config.name}: REST subscription not applicable")
        return True
    
    def _build_auth_headers(self) -> Dict[str, str]:
        """
        Build authentication headers based on config.
        
        Returns:
            Dict containing authentication headers
        """
        headers = {}
        
        if not self.config.auth_type:
            return headers
        
        if self.config.auth_type == "bearer":
            token = self.config.auth_credentials.get("token")
            if token:
                headers["Authorization"] = f"Bearer {token}"
        
        elif self.config.auth_type == "api_key":
            key_name = self.config.auth_credentials.get("key_name", "X-API-Key")
            key_value = self.config.auth_credentials.get("key_value")
            if key_value:
                headers[key_name] = key_value
        
        elif self.config.auth_type == "basic":
            import base64
            username = self.config.auth_credentials.get("username")
            password = self.config.auth_credentials.get("password")
            if username and password:
                credentials = base64.b64encode(
                    f"{username}:{password}".encode()
                ).decode()
                headers["Authorization"] = f"Basic {credentials}"
        
        elif self.config.auth_type == "custom":
            # Custom auth headers provided directly
            headers.update(self.config.auth_credentials)
        
        return headers
    
    async def _make_request(
        self,
        method: HttpMethod,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Union[Dict, str, bytes]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Any:
        """
        Make an HTTP request with rate limiting and retries.
        
        Args:
            method: HTTP method
            endpoint: API endpoint (relative to base_url)
            params: Query parameters
            data: Request body data
            headers: Additional headers
            **kwargs: Additional request parameters
            
        Returns:
            Response data (parsed JSON or text)
            
        Raises:
            Various exceptions based on response status
        """
        if not self.session:
            raise ConnectionError("Not connected to REST API")
        
        # Build full URL
        url = endpoint if endpoint.startswith("http") else urljoin(self.config.base_url, endpoint)
        
        # Check cache if enabled and method is GET
        if self.cache and method == HttpMethod.GET:
            cached = self.cache.get(method.value, url, params, data)
            if cached is not None:
                self._cache_hits += 1
                await self._trigger_callbacks("data", cached)
                return cached
        
        # Apply rate limiting
        wait_time = await self.rate_limiter.acquire()
        if wait_time > 0:
            logger.debug(f"{self.config.name}: Rate limit wait: {wait_time:.2f}s")
            await asyncio.sleep(wait_time)
        
        # Prepare request
        request_headers = headers or {}
        
        # Retry logic
        last_error = None
        for attempt in range(self.config.max_retries + 1):
            try:
                # Make the request
                self._request_count += 1
                self._metrics["messages_received"] += 1
                
                async with self.session.request(
                    method.value,
                    url,
                    params=params,
                    json=data if isinstance(data, dict) else None,
                    data=data if not isinstance(data, dict) else None,
                    headers=request_headers,
                    **kwargs
                ) as response:
                    # Check response status
                    if response.status == 401:
                        raise AuthenticationError(
                            "Authentication failed",
                            details={"status": response.status, "url": url}
                        )
                    
                    if response.status == 429:
                        # Rate limit exceeded
                        retry_after = response.headers.get("Retry-After")
                        raise RateLimitError(
                            "Rate limit exceeded",
                            retry_after=int(retry_after) if retry_after else 60,
                            details={"status": response.status, "url": url}
                        )
                    
                    if response.status >= 500:
                        # Server error - retry
                        raise TransientError(
                            f"Server error: {response.status}",
                            attempt=attempt + 1,
                            max_attempts=self.config.max_retries + 1
                        )
                    
                    response.raise_for_status()
                    
                    # Parse response
                    content_type = response.headers.get("Content-Type", "")
                    
                    if "application/json" in content_type:
                        result = await response.json()
                    else:
                        result = await response.text()
                    
                    # Cache successful GET responses
                    if self.cache and method == HttpMethod.GET:
                        self.cache.set(method.value, url, result, params, data)
                    
                    # Update metrics
                    self._metrics["messages_processed"] += 1
                    
                    # Trigger data callback
                    await self._trigger_callbacks("data", result)
                    
                    return result
                    
            except (AuthenticationError, ValidationError):
                # Don't retry these errors
                raise
                
            except RateLimitError as e:
                # Wait for rate limit reset
                if e.retry_after:
                    logger.warning(
                        f"{self.config.name}: Rate limited, "
                        f"waiting {e.retry_after}s"
                    )
                    await asyncio.sleep(e.retry_after)
                else:
                    raise
                    
            except (TransientError, aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = e
                
                # Calculate retry delay
                if attempt < self.config.max_retries:
                    delay = self.config.retry_delay * (
                        self.config.retry_backoff ** attempt
                    )
                    logger.warning(
                        f"{self.config.name}: Request failed "
                        f"(attempt {attempt + 1}), retrying in {delay:.1f}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    # Max retries exceeded
                    break
        
        # All retries failed
        self._metrics["errors_count"] += 1
        error_msg = f"Request failed after {self.config.max_retries + 1} attempts"
        
        if last_error:
            await self._trigger_callbacks("error", last_error)
            raise DataSourceError(
                error_msg,
                details={
                    "url": url,
                    "method": method.value,
                    "last_error": str(last_error)
                }
            )
        else:
            raise DataSourceError(error_msg, details={"url": url})
    
    # Convenience methods for HTTP verbs
    async def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """
        Make a GET request.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            **kwargs: Additional request parameters
            
        Returns:
            Response data
            
        Example:
            >>> data = await source.get("/users", params={"limit": 10})
        """
        return await self._make_request(
            HttpMethod.GET,
            endpoint,
            params=params,
            **kwargs
        )
    
    async def post(
        self,
        endpoint: str,
        data: Optional[Union[Dict, str, bytes]] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """
        Make a POST request.
        
        Args:
            endpoint: API endpoint
            data: Request body
            params: Query parameters
            **kwargs: Additional request parameters
            
        Returns:
            Response data
            
        Example:
            >>> result = await source.post("/users", data={"name": "Alice"})
        """
        return await self._make_request(
            HttpMethod.POST,
            endpoint,
            data=data,
            params=params,
            **kwargs
        )
    
    async def put(
        self,
        endpoint: str,
        data: Optional[Union[Dict, str, bytes]] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """
        Make a PUT request.
        
        Args:
            endpoint: API endpoint
            data: Request body
            params: Query parameters
            **kwargs: Additional request parameters
            
        Returns:
            Response data
        """
        return await self._make_request(
            HttpMethod.PUT,
            endpoint,
            data=data,
            params=params,
            **kwargs
        )
    
    async def delete(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """
        Make a DELETE request.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            **kwargs: Additional request parameters
            
        Returns:
            Response data
        """
        return await self._make_request(
            HttpMethod.DELETE,
            endpoint,
            params=params,
            **kwargs
        )
    
    async def paginate(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        max_pages: Optional[int] = None,
        **kwargs
    ) -> List[Any]:
        """
        Automatically paginate through API results.
        
        Args:
            endpoint: API endpoint
            params: Initial query parameters
            max_pages: Maximum number of pages to fetch
            **kwargs: Additional request parameters
            
        Returns:
            List of all results from all pages
            
        Example:
            >>> all_users = await source.paginate("/users", params={"limit": 100})
        """
        if self.config.pagination_type == "none":
            # No pagination, just return single response
            result = await self.get(endpoint, params=params, **kwargs)
            return [result] if not isinstance(result, list) else result
        
        all_results = []
        current_params = params or {}
        page_count = 0
        
        while True:
            # Check page limit
            if max_pages and page_count >= max_pages:
                break
            
            # Make request
            response = await self.get(endpoint, params=current_params, **kwargs)
            page_count += 1
            
            # Extract results based on response structure
            if isinstance(response, dict):
                # Look for common data keys
                data_keys = ["data", "results", "items", "records"]
                results = None
                for key in data_keys:
                    if key in response:
                        results = response[key]
                        break
                
                if results is None:
                    # Assume the whole response is the data
                    results = [response]
                
                all_results.extend(results if isinstance(results, list) else [results])
                
                # Check for next page
                if self.config.pagination_type == "cursor":
                    # Cursor-based pagination
                    cursor_key = self.config.pagination_params.get("cursor_key", "cursor")
                    next_key = self.config.pagination_params.get("next_key", "next")
                    
                    next_cursor = response.get(next_key)
                    if not next_cursor:
                        break
                    
                    current_params[cursor_key] = next_cursor
                
                elif self.config.pagination_type == "offset":
                    # Offset-based pagination
                    offset_key = self.config.pagination_params.get("offset_key", "offset")
                    limit_key = self.config.pagination_params.get("limit_key", "limit")
                    
                    limit = current_params.get(limit_key, 100)
                    offset = current_params.get(offset_key, 0)
                    
                    # Check if we got fewer results than limit
                    if len(results) < limit:
                        break
                    
                    current_params[offset_key] = offset + limit
                
                elif self.config.pagination_type == "page":
                    # Page-based pagination
                    page_key = self.config.pagination_params.get("page_key", "page")
                    total_pages_key = self.config.pagination_params.get("total_pages_key", "total_pages")
                    
                    current_page = current_params.get(page_key, 1)
                    total_pages = response.get(total_pages_key)
                    
                    if total_pages and current_page >= total_pages:
                        break
                    
                    current_params[page_key] = current_page + 1
                
            else:
                # Response is a list
                all_results.extend(response)
                break  # Can't paginate without metadata
        
        logger.info(
            f"{self.config.name}: Paginated {page_count} pages, "
            f"retrieved {len(all_results)} total results"
        )
        
        return all_results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get request statistics.
        
        Returns:
            Dictionary with request stats
            
        Example:
            >>> stats = source.get_stats()
            >>> print(f"Requests: {stats['total_requests']}")
        """
        stats = {
            "total_requests": self._request_count,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": (
                self._cache_hits / self._request_count 
                if self._request_count > 0 else 0
            )
        }
        
        # Add base metrics
        stats.update(self.get_metrics())
        
        return stats