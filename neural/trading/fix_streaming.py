"""
FIX API Streaming Client for Real-Time Market Data

Provides real-time streaming of Kalshi market data via FIX protocol.
"""

import asyncio
from datetime import datetime
from typing import Optional, Callable, Dict, Any, List
import simplefix
from dataclasses import dataclass
import pandas as pd

from .fix import KalshiFIXClient, FIXConnectionConfig


@dataclass
class MarketDataSnapshot:
    """Represents a market data snapshot"""
    timestamp: datetime
    symbol: str
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int
    last_price: Optional[float] = None
    volume: Optional[int] = None

    @property
    def spread(self) -> float:
        """Calculate bid-ask spread"""
        return self.ask_price - self.bid_price

    @property
    def mid_price(self) -> float:
        """Calculate mid-point price"""
        return (self.bid_price + self.ask_price) / 2

    @property
    def implied_probability(self) -> float:
        """Get implied probability from mid price"""
        return self.mid_price * 100  # Convert to percentage


class FIXStreamingClient:
    """
    Real-time streaming client for Kalshi market data via FIX.

    Provides:
    - Real-time bid/ask updates
    - Market depth information
    - Trade execution reports
    - Automatic reconnection
    """

    def __init__(
        self,
        on_market_data: Optional[Callable[[MarketDataSnapshot], None]] = None,
        on_execution: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        auto_reconnect: bool = True,
        heartbeat_interval: int = 30
    ):
        """
        Initialize streaming client.

        Args:
            on_market_data: Callback for market data updates
            on_execution: Callback for execution reports
            on_error: Callback for errors
            auto_reconnect: Automatically reconnect on disconnect
            heartbeat_interval: Heartbeat interval in seconds
        """
        self.on_market_data = on_market_data
        self.on_execution = on_execution
        self.on_error = on_error
        self.auto_reconnect = auto_reconnect
        self.heartbeat_interval = heartbeat_interval

        self.client: Optional[KalshiFIXClient] = None
        self.connected = False
        self.subscribed_symbols: List[str] = []
        self.market_data_cache: Dict[str, MarketDataSnapshot] = {}
        self._running = False
        self._reconnect_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        """Connect to FIX gateway"""
        if self.connected:
            return

        config = FIXConnectionConfig(
            heartbeat_interval=self.heartbeat_interval,
            reset_seq_num=True,
            listener_session=True,  # Enable market data
            cancel_on_disconnect=True
        )

        self.client = KalshiFIXClient(
            config=config,
            on_message=self._handle_message
        )

        try:
            await self.client.connect(timeout=10)
            self.connected = True
            self._running = True
            print(f"[{self._timestamp()}] ✅ Connected to FIX gateway")

            # Resubscribe to symbols if reconnecting
            if self.subscribed_symbols:
                for symbol in self.subscribed_symbols:
                    await self._send_market_data_request(symbol)

        except Exception as e:
            self.connected = False
            if self.on_error:
                self.on_error(f"Connection failed: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from FIX gateway"""
        self._running = False
        self.connected = False

        if self._reconnect_task:
            self._reconnect_task.cancel()

        if self.client:
            await self.client.close()
            self.client = None

        print(f"[{self._timestamp()}] 👋 Disconnected from FIX gateway")

    async def subscribe(self, symbol: str) -> None:
        """
        Subscribe to market data for a symbol.

        Args:
            symbol: Market symbol (e.g., 'KXNFLGAME-25SEP25SEAARI-SEA')
        """
        if not self.connected:
            await self.connect()

        if symbol not in self.subscribed_symbols:
            self.subscribed_symbols.append(symbol)

        await self._send_market_data_request(symbol)
        print(f"[{self._timestamp()}] 📊 Subscribed to {symbol}")

    async def unsubscribe(self, symbol: str) -> None:
        """Unsubscribe from market data"""
        if symbol in self.subscribed_symbols:
            self.subscribed_symbols.remove(symbol)

        # Send unsubscribe request
        await self._send_market_data_request(symbol, subscribe=False)
        print(f"[{self._timestamp()}] 🚫 Unsubscribed from {symbol}")

    async def _send_market_data_request(self, symbol: str, subscribe: bool = True) -> None:
        """Send market data subscription request"""
        if not self.client:
            return

        # Market Data Request (MsgType = V)
        fields = [
            (262, f"MDR_{datetime.now().strftime('%Y%m%d%H%M%S')}"),  # MDReqID
            (263, "1" if subscribe else "2"),  # SubscriptionRequestType (1=Subscribe, 2=Unsubscribe)
            (264, "0"),  # MarketDepth (0=Full book)
            (265, "1"),  # MDUpdateType (1=Incremental refresh)
            (267, "2"),  # NoMDEntryTypes (2 types: Bid and Offer)
            (269, "0"),  # MDEntryType: Bid
            (269, "1"),  # MDEntryType: Offer
            (146, "1"),  # NoRelatedSym (1 symbol)
            (55, symbol),  # Symbol
        ]

        await self.client._send_message("V", fields)

    def _handle_message(self, message: simplefix.FixMessage) -> None:
        """Handle incoming FIX message"""
        try:
            msg_dict = KalshiFIXClient.to_dict(message)
            msg_type = msg_dict.get(35)

            if msg_type == 'W':  # Market Data Snapshot/Full Refresh
                self._handle_market_data_snapshot(msg_dict)
            elif msg_type == 'X':  # Market Data Incremental Refresh
                self._handle_market_data_update(msg_dict)
            elif msg_type == '8':  # Execution Report
                self._handle_execution_report(msg_dict)
            elif msg_type == 'Y':  # Market Data Request Reject
                self._handle_market_data_reject(msg_dict)
            elif msg_type == '5':  # Logout
                self.connected = False
                if self.auto_reconnect and self._running:
                    self._reconnect_task = asyncio.create_task(self._reconnect())

        except Exception as e:
            if self.on_error:
                self.on_error(f"Error handling message: {e}")

    def _handle_market_data_snapshot(self, msg: Dict[int, Any]) -> None:
        """Handle market data snapshot"""
        symbol = msg.get(55)  # Symbol
        if not symbol:
            return

        # Extract bid/ask data
        bid_price = self._parse_price(msg.get(132))  # BidPx
        ask_price = self._parse_price(msg.get(133))  # OfferPx
        bid_size = int(msg.get(134, 0))  # BidSize
        ask_size = int(msg.get(135, 0))  # OfferSize

        # Create snapshot
        snapshot = MarketDataSnapshot(
            timestamp=datetime.now(),
            symbol=symbol,
            bid_price=bid_price,
            ask_price=ask_price,
            bid_size=bid_size,
            ask_size=ask_size
        )

        # Cache and notify
        self.market_data_cache[symbol] = snapshot
        if self.on_market_data:
            self.on_market_data(snapshot)

    def _handle_market_data_update(self, msg: Dict[int, Any]) -> None:
        """Handle incremental market data update"""
        # Parse incremental updates
        # This would contain multiple entries for bid/ask updates
        # Implementation depends on Kalshi's specific FIX format
        pass

    def _handle_execution_report(self, msg: Dict[int, Any]) -> None:
        """Handle execution report"""
        if self.on_execution:
            exec_report = {
                'order_id': msg.get(11),  # ClOrdID
                'symbol': msg.get(55),  # Symbol
                'side': msg.get(54),  # Side
                'quantity': msg.get(38),  # OrderQty
                'price': self._parse_price(msg.get(44)),  # Price
                'status': msg.get(39),  # OrdStatus
                'exec_type': msg.get(150),  # ExecType
                'timestamp': datetime.now()
            }
            self.on_execution(exec_report)

    def _handle_market_data_reject(self, msg: Dict[int, Any]) -> None:
        """Handle market data request rejection"""
        reason = msg.get(58, "Unknown reason")
        if self.on_error:
            self.on_error(f"Market data request rejected: {reason}")

    async def _reconnect(self) -> None:
        """Attempt to reconnect after disconnect"""
        retry_count = 0
        max_retries = 5
        retry_delay = 5

        while self._running and retry_count < max_retries:
            await asyncio.sleep(retry_delay)
            print(f"[{self._timestamp()}] 🔄 Attempting reconnection... (attempt {retry_count + 1})")

            try:
                await self.connect()
                print(f"[{self._timestamp()}] ✅ Reconnected successfully")
                break
            except Exception as e:
                retry_count += 1
                retry_delay = min(retry_delay * 2, 60)  # Exponential backoff
                if self.on_error:
                    self.on_error(f"Reconnection failed: {e}")

    def _parse_price(self, value: Any) -> float:
        """Parse price from FIX message (cents to dollars)"""
        if value is None:
            return 0.0
        return float(value) / 100

    def _timestamp(self) -> str:
        """Get current timestamp string"""
        return datetime.now().strftime('%H:%M:%S')

    def get_snapshot(self, symbol: str) -> Optional[MarketDataSnapshot]:
        """Get latest market data snapshot for symbol"""
        return self.market_data_cache.get(symbol)

    def get_all_snapshots(self) -> Dict[str, MarketDataSnapshot]:
        """Get all cached market data snapshots"""
        return self.market_data_cache.copy()

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()


async def stream_market_data(
    symbols: List[str],
    duration_seconds: int = 60,
    on_update: Optional[Callable[[MarketDataSnapshot], None]] = None
) -> pd.DataFrame:
    """
    Stream market data for specified symbols.

    Args:
        symbols: List of market symbols
        duration_seconds: How long to stream
        on_update: Optional callback for each update

    Returns:
        DataFrame with all collected market data
    """
    history = []

    def handle_market_data(snapshot: MarketDataSnapshot):
        # Record to history
        history.append({
            'timestamp': snapshot.timestamp,
            'symbol': snapshot.symbol,
            'bid': snapshot.bid_price,
            'ask': snapshot.ask_price,
            'spread': snapshot.spread,
            'mid': snapshot.mid_price,
            'implied_prob': snapshot.implied_probability,
            'bid_size': snapshot.bid_size,
            'ask_size': snapshot.ask_size
        })

        # Call user callback
        if on_update:
            on_update(snapshot)

        # Print update
        print(f"[{snapshot.timestamp.strftime('%H:%M:%S')}] "
              f"{snapshot.symbol}: "
              f"Bid ${snapshot.bid_price:.2f} x {snapshot.bid_size} | "
              f"Ask ${snapshot.ask_price:.2f} x {snapshot.ask_size} | "
              f"Spread ${snapshot.spread:.2f}")

    # Create streaming client
    client = FIXStreamingClient(on_market_data=handle_market_data)

    try:
        async with client:
            # Subscribe to all symbols
            for symbol in symbols:
                await client.subscribe(symbol)

            # Stream for specified duration
            await asyncio.sleep(duration_seconds)

    except KeyboardInterrupt:
        print("\n⏹️ Streaming stopped by user")

    # Convert history to DataFrame
    if history:
        return pd.DataFrame(history)
    else:
        return pd.DataFrame()