import logging
from typing import Dict, Any, List

from neural_sdk.data_pipeline.data_sources.twitter.client import TwitterAPIClient

logger = logging.getLogger(__name__)

class TwitterRESTAdapter:
    def __init__(self, api_key: str):
        self.client = TwitterAPIClient(api_key=api_key)

    async def get_tweets_by_query(self, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        logger.info(f"Adapter: Fetching tweets for query: '{query}'")
        raw_tweets = await self.client.search_tweets(query=query, limit=limit)
        
        processed_tweets = []
        for tweet in raw_tweets:
            processed_tweets.append({
                "id": tweet.get("id"),
                "text": tweet.get("text"),
                "created_at": tweet.get("created_at"),
                "author_id": tweet.get("author_id"),
                "public_metrics": tweet.get("public_metrics", {}),
                # Add more fields as needed
            })
        return processed_tweets

    async def close(self):
        await self.client.aclose()
