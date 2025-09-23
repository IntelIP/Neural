#!/usr/bin/env python3
"""
Twitter WebSocket Example

This example demonstrates how to use the Neural SDK to connect to a
Twitter WebSocket stream and process real-time tweet data.

This showcases the simplicity of the SDK - just configure and connect!
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any

from neural.data_collection import (
    WebSocketDataSource,
    WebSocketConfig,
    DataPipeline,
    TransformStage,
    create_logger
)


# Configure logging
logger = create_logger("twitter_example", level="INFO")


class TweetProcessor(TransformStage):
    """Transform raw Twitter data into structured format."""
    
    async def process(self, data: Any) -> Any:
        """Extract key information from tweet."""
        if isinstance(data, dict):
            # Extract essential tweet information
            return {
                "id": data.get("id"),
                "text": data.get("text"),
                "author": data.get("author", {}).get("username"),
                "created_at": data.get("created_at"),
                "metrics": {
                    "likes": data.get("public_metrics", {}).get("like_count", 0),
                    "retweets": data.get("public_metrics", {}).get("retweet_count", 0),
                    "replies": data.get("public_metrics", {}).get("reply_count", 0)
                },
                "processed_at": datetime.utcnow().isoformat()
            }
        return data


class SentimentAnalyzer(TransformStage):
    """Simple sentiment analysis stage."""
    
    async def process(self, data: Any) -> Any:
        """Add basic sentiment analysis."""
        if isinstance(data, dict) and "text" in data:
            text = data["text"].lower()
            
            # Very basic sentiment scoring
            positive_words = ["good", "great", "excellent", "amazing", "love", "best"]
            negative_words = ["bad", "terrible", "awful", "hate", "worst", "poor"]
            
            positive_score = sum(1 for word in positive_words if word in text)
            negative_score = sum(1 for word in negative_words if word in text)
            
            if positive_score > negative_score:
                sentiment = "positive"
            elif negative_score > positive_score:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            data["sentiment"] = {
                "label": sentiment,
                "positive_score": positive_score,
                "negative_score": negative_score
            }
        
        return data


async def handle_tweet(data: Dict[str, Any]) -> None:
    """Handle processed tweet data."""
    logger.info(f"Tweet from @{data.get('author')}: {data.get('sentiment', {}).get('label', 'unknown')}")
    logger.debug(f"Full data: {json.dumps(data, indent=2)}")


async def handle_error(error: Exception) -> None:
    """Handle stream errors."""
    logger.error(f"Stream error: {error}")


async def main():
    """Main example function."""
    
    # Create WebSocket configuration
    # In production, these would come from environment variables
    config = WebSocketConfig(
        url="wss://api.twitter.com/2/tweets/search/stream",
        headers={
            "Authorization": "Bearer ${TWITTER_BEARER_TOKEN}"
        },
        reconnect=True,
        max_reconnect_attempts=5,
        heartbeat_interval=30
    )
    
    # Create data source
    twitter_ws = WebSocketDataSource(config)
    
    # Register event handlers
    twitter_ws.register_callback("error", handle_error)
    
    # Create data pipeline
    pipeline = DataPipeline()
    
    # Add the Twitter WebSocket as a source
    await pipeline.add_source("twitter", twitter_ws)
    
    # Add transformation stages
    pipeline.add_stage(TweetProcessor())
    pipeline.add_stage(SentimentAnalyzer())
    
    # Add consumer for processed data
    await pipeline.add_consumer(handle_tweet)
    
    logger.info("Starting Twitter WebSocket stream...")
    
    # Connect and start processing
    try:
        await twitter_ws.connect()
        
        # Start the pipeline
        await pipeline.start()
        
        # Keep running
        logger.info("Stream connected. Processing tweets...")
        logger.info("Press Ctrl+C to stop")
        
        # Run forever
        await asyncio.Event().wait()
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        # Clean shutdown
        await pipeline.stop()
        await twitter_ws.disconnect()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())