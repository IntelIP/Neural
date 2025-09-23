"""
REST API Polling for Real-Time-Like Market Data Streaming

Provides market data streaming via REST API polling as a fallback
when WebSocket or FIX market data is unavailable.
"""

import asyncio
from datetime import datetime
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass
import pandas as pd

from neural.data_collection import KalshiMarketsSource


@dataclass
class MarketSnapshot:
    """Market data snapshot from REST API"""
    timestamp: datetime
    ticker: str
    title: str
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float
    volume: int
    open_interest: int
    last_price: Optional[float] = None

    @property
    def yes_spread(self) -> float:
        """Calculate YES bid-ask spread"""
        return self.yes_ask - self.yes_bid

    @property
    def yes_mid(self) -> float:
        """Calculate YES mid price"""
        return (self.yes_bid + self.yes_ask) / 2

    @property
    def implied_probability(self) -> float:
        """Get implied probability from YES mid price"""
        return self.yes_mid * 100  # Convert to percentage

    @property
    def arbitrage_opportunity(self) -> float:
        """Check if YES + NO prices sum to less than 100"""
        total = self.yes_ask + self.no_ask
        if total < 100:
            return 100 - total  # Profit opportunity in cents
        return 0


class RESTStreamingClient:
    """
    Simulates real-time streaming using REST API polling.

    Provides market data updates via callbacks when prices change.
    More reliable than WebSocket when permissions are limited.
    """

    def __init__(
        self,
        on_market_update: Optional[Callable[[MarketSnapshot], None]] = None,
        on_price_change: Optional[Callable[[str, float, float], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        poll_interval: float = 1.0,
        min_price_change: float = 0.001  # Minimum change to trigger update (0.1 cent)
    ):
        """
        Initialize REST streaming client.

        Args:
            on_market_update: Callback for any market update
            on_price_change: Callback for significant price changes (ticker, old_price, new_price)
            on_error: Callback for errors
            poll_interval: Seconds between polls (minimum 0.5)
            min_price_change: Minimum price change to trigger callback
        """
        self.on_market_update = on_market_update
        self.on_price_change = on_price_change
        self.on_error = on_error
        self.poll_interval = max(0.5, poll_interval)  # Enforce minimum interval
        self.min_price_change = min_price_change

        self.client: Optional[KalshiMarketsSource] = None
        self.market_cache: Dict[str, MarketSnapshot] = {}
        self.subscribed_tickers: List[str] = []
        self._running = False
        self._poll_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        """Connect to Kalshi REST API"""
        try:
            self.client = KalshiMarketsSource()
            print(f"[{self._timestamp()}] ✅ Connected to Kalshi REST API")
            self._running = True
        except Exception as e:
            if self.on_error:
                self.on_error(f"Connection failed: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect and stop polling"""
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        self.client = None
        print(f"[{self._timestamp()}] 👋 Disconnected from REST API")

    async def subscribe(self, tickers: List[str]) -> None:
        """
        Subscribe to market tickers for polling.

        Args:
            tickers: List of market tickers to monitor
        """
        for ticker in tickers:
            if ticker not in self.subscribed_tickers:
                self.subscribed_tickers.append(ticker)

        print(f"[{self._timestamp()}] 📊 Subscribed to {len(tickers)} markets")

        # Start polling if not already running
        if self._running and not self._poll_task:
            self._poll_task = asyncio.create_task(self._poll_loop())

    async def unsubscribe(self, tickers: List[str]) -> None:
        """Unsubscribe from market tickers"""
        for ticker in tickers:
            if ticker in self.subscribed_tickers:
                self.subscribed_tickers.remove(ticker)

        print(f"[{self._timestamp()}] 🚫 Unsubscribed from {len(tickers)} markets")

    async def _poll_loop(self) -> None:
        """Main polling loop"""
        poll_count = 0

        while self._running and self.subscribed_tickers:
            try:
                poll_count += 1

                # Fetch market data for all subscribed tickers
                for ticker in self.subscribed_tickers:
                    await self._fetch_market(ticker)

                # Show periodic status
                if poll_count % 10 == 0:  # Every 10 polls
                    active_markets = len([m for m in self.market_cache.values()
                                         if (datetime.now() - m.timestamp).seconds < 5])
                    print(f"[{self._timestamp()}] 📈 Polling {len(self.subscribed_tickers)} markets, "
                          f"{active_markets} active")

                # Wait before next poll
                await asyncio.sleep(self.poll_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.on_error:
                    self.on_error(f"Polling error: {e}")
                await asyncio.sleep(self.poll_interval)

    async def _fetch_market(self, ticker: str) -> None:
        """Fetch and process single market"""
        try:
            if not self.client:
                return

            # Get market data using the client's method
            markets_df = self.client.get_markets_for_ticker(ticker)

            if markets_df.empty:
                return

            # Get first market (should be the only one for a specific ticker)
            market = markets_df.iloc[0].to_dict()

            # Create snapshot
            snapshot = MarketSnapshot(
                timestamp=datetime.now(),
                ticker=ticker,
                title=market.get('title', ''),
                yes_bid=market.get('yes_bid', 0) / 100,  # Convert cents to dollars
                yes_ask=market.get('yes_ask', 0) / 100,
                no_bid=market.get('no_bid', 0) / 100,
                no_ask=market.get('no_ask', 0) / 100,
                volume=market.get('volume', 0),
                open_interest=market.get('open_interest', 0),
                last_price=market.get('last_price', 0) / 100 if market.get('last_price') else None
            )

            # Check for price changes
            if ticker in self.market_cache:
                old_snapshot = self.market_cache[ticker]
                price_change = abs(snapshot.yes_mid - old_snapshot.yes_mid)

                # Trigger callback on significant change
                if price_change >= self.min_price_change:
                    if self.on_price_change:
                        self.on_price_change(ticker, old_snapshot.yes_mid, snapshot.yes_mid)

                    # Show significant changes
                    if price_change >= 0.01:  # 1 cent or more
                        direction = "📈" if snapshot.yes_mid > old_snapshot.yes_mid else "📉"
                        print(f"[{self._timestamp()}] {direction} {ticker}: "
                              f"${old_snapshot.yes_mid:.3f} → ${snapshot.yes_mid:.3f} "
                              f"({price_change*100:.1f}¢ move)")

            # Update cache
            self.market_cache[ticker] = snapshot

            # Trigger update callback
            if self.on_market_update:
                self.on_market_update(snapshot)

        except Exception as e:
            if self.on_error:
                self.on_error(f"Error fetching {ticker}: {e}")

    def get_snapshot(self, ticker: str) -> Optional[MarketSnapshot]:
        """Get latest snapshot for a ticker"""
        return self.market_cache.get(ticker)

    def get_all_snapshots(self) -> Dict[str, MarketSnapshot]:
        """Get all cached snapshots"""
        return self.market_cache.copy()

    def _timestamp(self) -> str:
        """Get current timestamp string"""
        return datetime.now().strftime('%H:%M:%S')

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()


async def stream_via_rest(
    tickers: List[str],
    duration_seconds: int = 60,
    poll_interval: float = 1.0,
    on_update: Optional[Callable[[MarketSnapshot], None]] = None
) -> pd.DataFrame:
    """
    Stream market data via REST API polling.

    Args:
        tickers: List of market tickers to monitor
        duration_seconds: How long to stream
        poll_interval: Seconds between polls
        on_update: Optional callback for each update

    Returns:
        DataFrame with all collected market data
    """
    history = []

    def handle_update(snapshot: MarketSnapshot):
        # Record to history
        history.append({
            'timestamp': snapshot.timestamp,
            'ticker': snapshot.ticker,
            'yes_bid': snapshot.yes_bid,
            'yes_ask': snapshot.yes_ask,
            'yes_spread': snapshot.yes_spread,
            'yes_mid': snapshot.yes_mid,
            'implied_prob': snapshot.implied_probability,
            'volume': snapshot.volume,
            'open_interest': snapshot.open_interest,
            'arbitrage': snapshot.arbitrage_opportunity
        })

        # Call user callback
        if on_update:
            on_update(snapshot)

    # Create streaming client
    client = RESTStreamingClient(
        on_market_update=handle_update,
        poll_interval=poll_interval
    )

    try:
        async with client:
            # Subscribe to tickers
            await client.subscribe(tickers)

            # Stream for specified duration
            print(f"\n🔄 Streaming {len(tickers)} markets for {duration_seconds} seconds...")
            await asyncio.sleep(duration_seconds)

    except KeyboardInterrupt:
        print("\n⏹️ Streaming stopped by user")

    # Convert history to DataFrame
    if history:
        df = pd.DataFrame(history)
        df = df.sort_values(['ticker', 'timestamp'])
        return df
    else:
        return pd.DataFrame()