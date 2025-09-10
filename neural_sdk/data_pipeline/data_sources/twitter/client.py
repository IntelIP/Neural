import httpx
import logging
from typing import Dict, Any, List, Optional
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

logger = logging.getLogger(__name__)

class TwitterAPIClient:
    BASE_URL = "https://api.twitterapi.io/api/v1"

    def __init__(self, api_key: str, timeout: int = 30):
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=timeout)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError))
    )
    async def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/{endpoint}"
        headers = {"X-API-KEY": self.api_key}
        
        logger.info(f"Making Twitter API request to {url} with params {params}")
        response = await self.client.get(url, headers=headers, params=params)
        response.raise_for_status() # Raise an exception for 4xx or 5xx responses
        return response.json()

    async def search_tweets(self, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        logger.info(f"Searching tweets for query: '{query}' with limit: {limit}")
        params = {"q": query, "limit": limit}
        data = await self._make_request("search", params=params)
        return data.get("tweets", [])

    async def get_tweet_details(self, tweet_id: str) -> Optional[Dict[str, Any]]:
        logger.info(f"Getting details for tweet ID: {tweet_id}")
        data = await self._make_request(f"tweet/{tweet_id}")
        return data.get("tweet")

    async def close(self):
        await self.client.aclose()