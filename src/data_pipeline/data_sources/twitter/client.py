"""
TwitterAPI.io WebSocket Client
Real-time Twitter data streaming with proper authentication
"""

import os
import json
import asyncio
import logging
import time
import traceback
from typing import Dict, Any, Optional, Callable
from datetime import datetime

from websocket import WebSocketApp, WebSocketTimeoutException, WebSocketBadStatusException

from .models import Tweet, Author, TweetMetrics

logger = logging.getLogger(__name__)


class TwitterWebSocketClient:
    """
    WebSocket client for TwitterAPI.io real-time streaming
    Based on official documentation
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Twitter WebSocket client
        
        Args:
            api_key: TwitterAPI.io API key (or from TWITTERAPI_KEY env var)
        """
        # Prefer TWITTERAPI_KEY (TwitterAPI.io). Fall back to TWITTER_BEARER_TOKEN if set.
        self.api_key = api_key or os.getenv('TWITTERAPI_KEY') or os.getenv('TWITTER_BEARER_TOKEN')
        if not self.api_key:
            raise ValueError("TwitterAPI.io API key required. Set TWITTERAPI_KEY env var or pass api_key")
        if not api_key and not os.getenv('TWITTERAPI_KEY') and os.getenv('TWITTER_BEARER_TOKEN'):
            logger.warning("Using TWITTER_BEARER_TOKEN for TwitterAPI.io client. Ensure this is a TwitterAPI.io API key, not Twitter v2 bearer token.")
        
        # WebSocket configuration
        self.ws_url = "wss://ws.twitterapi.io/twitter/tweet/websocket"
        self.ws: Optional[WebSocketApp] = None
        
        # Connection settings
        self.ping_interval = 40  # seconds
        self.ping_timeout = 30   # seconds
        self.reconnect_delay = 90  # seconds (per documentation)
        
        # State tracking
        self.is_connected = False
        self.connection_start: Optional[datetime] = None
        self.tweets_received = 0
        self.last_ping: Optional[datetime] = None
        
        # Callbacks
        self.on_tweet: Optional[Callable[[Tweet], None]] = None
        self.on_error: Optional[Callable[[Dict[str, Any]], None]] = None
        self.on_connected: Optional[Callable[[], None]] = None
        self.on_disconnected: Optional[Callable[[int, str], None]] = None
    
    def _on_message(self, ws: WebSocketApp, message: str):
        """Handle incoming WebSocket messages"""
        try:
            logger.debug(f"Received message: {message[:200]}...")
            result_json = json.loads(message)
            event_type = result_json.get("event_type")
            
            if event_type == "connected":
                self._handle_connected(result_json)
            elif event_type == "ping":
                self._handle_ping(result_json)
            elif event_type == "tweet":
                self._handle_tweet(result_json)
            else:
                logger.warning(f"Unknown event type: {event_type}")
        
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}. Message: {message[:200]}")
        except Exception as e:
            logger.error(f"Error processing message: {e}. Traceback: {traceback.format_exc()}")
    
    def _handle_connected(self, data: Dict[str, Any]):
        """Handle connection confirmation"""
        self.is_connected = True
        self.connection_start = datetime.now()
        logger.info("WebSocket connection confirmed")
        
        if self.on_connected:
            self.on_connected()
    
    def _handle_ping(self, data: Dict[str, Any]):
        """Handle ping/heartbeat message"""
        timestamp = data.get("timestamp", 0)
        current_time_ms = time.time() * 1000
        diff_time_ms = current_time_ms - timestamp
        
        self.last_ping = datetime.now()
        
        logger.debug(f"Ping received. Latency: {diff_time_ms:.0f}ms")
    
    def _handle_tweet(self, data: Dict[str, Any]):
        """Handle tweet event"""
        try:
            rule_id = data.get("rule_id")
            rule_tag = data.get("rule_tag")
            tweets = data.get("tweets", [])
            timestamp = data.get("timestamp")
            
            logger.info(f"Tweet event - Rule: {rule_tag}, Count: {len(tweets)}")
            
            # Calculate latency
            current_time_ms = time.time() * 1000
            latency_ms = current_time_ms - timestamp if timestamp else 0
            
            # Process each tweet
            for tweet_data in tweets:
                self.tweets_received += 1
                tweet = self._parse_tweet(tweet_data)
                
                if tweet and self.on_tweet:
                    self.on_tweet(tweet)
            
            # Log performance metrics
            if self.tweets_received % 100 == 0:
                logger.info(f"Processed {self.tweets_received} tweets. Latency: {latency_ms:.0f}ms")
        
        except Exception as e:
            logger.error(f"Error handling tweet: {e}")
    
    def _parse_tweet(self, data: Dict[str, Any]) -> Optional[Tweet]:
        """Parse raw tweet data into Tweet object"""
        try:
            # Parse author
            author_data = data.get("author", {})
            author = Author(
                id=author_data.get("id", ""),
                username=author_data.get("username", ""),
                name=author_data.get("name", ""),
                verified=author_data.get("verified", False),
                followers_count=author_data.get("public_metrics", {}).get("followers_count", 0),
                following_count=author_data.get("public_metrics", {}).get("following_count", 0),
                tweet_count=author_data.get("public_metrics", {}).get("tweet_count", 0),
                description=author_data.get("description")
            )
            
            # Parse metrics
            metrics_data = data.get("public_metrics", {})
            metrics = TweetMetrics(
                retweet_count=metrics_data.get("retweet_count", 0),
                reply_count=metrics_data.get("reply_count", 0),
                like_count=metrics_data.get("like_count", 0),
                quote_count=metrics_data.get("quote_count", 0),
                impression_count=metrics_data.get("impression_count")
            )
            
            # Parse created_at
            created_at_str = data.get("created_at", "")
            try:
                # Twitter format: "Sat Mar 15 05:31:28 +0000 2025"
                created_at = datetime.strptime(created_at_str, "%a %b %d %H:%M:%S %z %Y")
            except:
                created_at = datetime.now()
            
            # Create Tweet object
            tweet = Tweet(
                id=data.get("id", ""),
                text=data.get("text", ""),
                author=author,
                created_at=created_at,
                metrics=metrics,
                lang=data.get("lang", "en"),
                conversation_id=data.get("conversation_id"),
                in_reply_to_user_id=data.get("in_reply_to_user_id"),
                referenced_tweets=data.get("referenced_tweets", []),
                entities=data.get("entities", {}),
                context_annotations=data.get("context_annotations", []),
                possibly_sensitive=data.get("possibly_sensitive", False)
            )
            
            return tweet
        
        except Exception as e:
            logger.error(f"Error parsing tweet: {e}")
            return None
    
    def _on_error(self, ws: WebSocketApp, error: Exception):
        """Handle WebSocket errors"""
        logger.error(f"WebSocket error: {error}. Traceback: {traceback.format_exc()}")
        
        error_info = {
            "error": str(error),
            "type": type(error).__name__,
            "timestamp": datetime.now().isoformat()
        }
        
        if isinstance(error, WebSocketTimeoutException):
            error_info["description"] = "Connection timeout. Check server status or network."
        elif isinstance(error, WebSocketBadStatusException):
            error_info["description"] = f"Server returned error status: {error}. Check API key."
        elif isinstance(error, ConnectionRefusedError):
            error_info["description"] = "Connection refused. Check server address."
        
        if self.on_error:
            self.on_error(error_info)
    
    def _on_close(self, ws: WebSocketApp, close_status_code: Optional[int], close_msg: Optional[str]):
        """Handle WebSocket connection close"""
        self.is_connected = False
        
        logger.info(f"Connection closed: code={close_status_code}, message={close_msg}")
        
        # Interpret close codes
        close_descriptions = {
            1000: "Normal closure",
            1001: "Server shutting down or client navigating away",
            1002: "Protocol error",
            1003: "Unacceptable data type received",
            1006: "Abnormal closure (network issues)",
            1008: "Policy violation",
            1011: "Server internal error",
            1013: "Server overloaded"
        }
        
        description = close_descriptions.get(close_status_code, "Unknown reason")
        logger.info(f"Close reason: {description}")
        
        if self.on_disconnected:
            self.on_disconnected(close_status_code or 0, close_msg or "")
        
        # Log session statistics
        if self.connection_start:
            duration = (datetime.now() - self.connection_start).total_seconds()
            logger.info(
                f"Session stats: Duration={duration:.0f}s, "
                f"Tweets={self.tweets_received}, "
                f"Rate={self.tweets_received/duration:.2f}/s"
            )
    
    def _on_open(self, ws: WebSocketApp):
        """Handle WebSocket connection open"""
        logger.info("WebSocket connection established")
    
    def connect(self):
        """Establish WebSocket connection"""
        headers = {"x-api-key": self.api_key}
        
        self.ws = WebSocketApp(
            self.ws_url,
            header=headers,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open
        )
        
        logger.info(f"Connecting to {self.ws_url}")
        
        # Run forever with automatic reconnection
        self.ws.run_forever(
            ping_interval=self.ping_interval,
            ping_timeout=self.ping_timeout,
            reconnect=self.reconnect_delay
        )
    
    def disconnect(self):
        """Close WebSocket connection"""
        if self.ws:
            self.ws.close()
            self.ws = None
            logger.info("WebSocket disconnected")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        uptime = None
        if self.connection_start:
            uptime = (datetime.now() - self.connection_start).total_seconds()
        
        last_ping_ago = None
        if self.last_ping:
            last_ping_ago = (datetime.now() - self.last_ping).total_seconds()
        
        return {
            "connected": self.is_connected,
            "tweets_received": self.tweets_received,
            "uptime_seconds": uptime,
            "tweets_per_minute": (self.tweets_received / uptime * 60) if uptime else 0,
            "last_ping_seconds_ago": last_ping_ago,
            "connection_start": self.connection_start.isoformat() if self.connection_start else None
        }


# Async wrapper for compatibility
class AsyncTwitterWebSocketClient:
    """Async wrapper for TwitterWebSocketClient"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = TwitterWebSocketClient(api_key)
        self.running = False
        self._task: Optional[asyncio.Task] = None
    
    async def connect(self):
        """Connect to WebSocket in background"""
        self.running = True
        
        # Run WebSocket in executor (since websocket-client is sync)
        loop = asyncio.get_event_loop()
        self._task = loop.run_in_executor(None, self.client.connect)
        
        # Wait for connection
        for _ in range(10):  # 10 second timeout
            await asyncio.sleep(1)
            if self.client.is_connected:
                return
        
        raise TimeoutError("Failed to connect to Twitter WebSocket")
    
    async def disconnect(self):
        """Disconnect from WebSocket"""
        self.running = False
        self.client.disconnect()
        
        if self._task:
            self._task.cancel()
    
    def on_tweet(self, callback: Callable[[Tweet], None]):
        """Set tweet callback"""
        self.client.on_tweet = callback
    
    def on_error(self, callback: Callable[[Dict[str, Any]], None]):
        """Set error callback"""
        self.client.on_error = callback
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return self.client.get_stats()
