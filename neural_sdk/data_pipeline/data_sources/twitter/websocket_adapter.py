import asyncio
import logging
from typing import Callable, Awaitable, Dict, Any

from neural_sdk.data_pipeline.data_sources.twitter.rest_adapter import TwitterRESTAdapter
from neural_sdk.data_pipeline.data_sources.twitter.sentiment import SentimentAnalyzer

logger = logging.getLogger(__name__)

class TwitterWebSocketAdapter:
    def __init__(
        self,
        api_key: str,
        query: str,
        poll_interval: int = 60,
        on_tweet: Callable[[Dict[str, Any]], Awaitable[None]] = None
    ):
        self.rest_adapter = TwitterRESTAdapter(api_key=api_key)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.query = query
        self.poll_interval = poll_interval
        self.on_tweet = on_tweet
        self._running = False
        self._last_tweet_id = None # To track new tweets

    async def _poll_for_tweets(self):
        while self._running:
            try:
                logger.info(f"Polling Twitter API for new tweets matching '{self.query}'...")
                # Fetch a small number of recent tweets
                tweets = await self.rest_adapter.get_tweets_by_query(self.query, limit=10)

                new_tweets = []
                # Simple logic to find new tweets since last poll
                if self._last_tweet_id:
                    for tweet in tweets:
                        if tweet["id"] == self._last_tweet_id:
                            break
                        new_tweets.append(tweet)
                    new_tweets.reverse() # Process oldest first
                else:
                    new_tweets = tweets # First poll, process all

                if new_tweets:
                    logger.info(f"Found {len(new_tweets)} new tweets.")
                    for tweet in new_tweets:
                        # Perform sentiment analysis
                        sentiment = self.sentiment_analyzer.analyze(tweet.get("text", ""))
                        tweet["sentiment"] = sentiment

                        # Push to handler if provided
                        if self.on_tweet:
                            await self.on_tweet(tweet)
                    self._last_tweet_id = new_tweets[-1]["id"] # Update last seen tweet
                else:
                    logger.info("No new tweets found.")

            except Exception as e:
                logger.error(f"Error during Twitter polling: {e}")

            await asyncio.sleep(self.poll_interval)

    async def start(self):
        logger.info("Starting Twitter WebSocket Adapter (polling mode)...")
        self._running = True
        self._polling_task = asyncio.create_task(self._poll_for_tweets())

    async def stop(self):
        logger.info("Stopping Twitter WebSocket Adapter...")
        self._running = False
        if self._polling_task:
            self._polling_task.cancel()
            try:
                await self._polling_task
            except asyncio.CancelledError:
                pass
        await self.rest_adapter.close()

# Example usage (for testing purposes)
async def handle_incoming_tweet(tweet: Dict[str, Any]):
    print(f"\n--- Incoming Tweet ---")
    print(f"ID: {tweet.get('id')}")
    print(f"Text: {tweet.get('text')}")
    print(f"Sentiment: {tweet.get('sentiment', {}).get('label')}")
    print(f"Score: {tweet.get('sentiment', {}).get('score')}")

async def run_example():
    # Replace with your actual twitterapi.io API key
    # You would typically load this from an environment variable
    twitter_api_key = "YOUR_TWITTERAPI_IO_KEY"
    query = "#CFB OR #CollegeFootball"

    if twitter_api_key == "YOUR_TWITTERAPI_IO_KEY":
        print("Please replace 'YOUR_TWITTERAPI_IO_KEY' with your actual twitterapi.io API key in the script.")
        return

    adapter = TwitterWebSocketAdapter(
        api_key=twitter_api_key,
        query=query,
        poll_interval=10, # Poll every 10 seconds for demo
        on_tweet=handle_incoming_tweet
    )

    await adapter.start()
    print(f"Listening for tweets matching '{query}' (polling every 10s). Press Ctrl+C to stop.")
    try:
        while True:
            await asyncio.sleep(1) # Keep main loop alive
    except asyncio.CancelledError:
        pass
    finally:
        await adapter.stop()

if __name__ == "__main__":
    # This example requires an actual API key and will continuously poll.
    # It's best to run this in a controlled environment.
    try:
        asyncio.run(run_example())
    except KeyboardInterrupt:
        print("\nTwitter WebSocket Adapter stopped by user.")
