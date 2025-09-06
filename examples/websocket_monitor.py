#!/usr/bin/env python3
"""
WebSocket Connection Monitor

Monitors WebSocket connection health and data flow metrics.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys
from collections import defaultdict, deque
from typing import Dict, List, Deque

sys.path.insert(0, str(Path(__file__).parent.parent))

from neural_sdk.data_sources.kalshi.websocket_adapter import KalshiWebSocketAdapter, KalshiChannel
from neural_sdk.data_sources.unified.stream_manager import UnifiedStreamManager, EventType, StreamConfig


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WebSocketMonitor:
    """Monitors WebSocket connections and data flow."""
    
    def __init__(self):
        self.stream_manager = None
        self.metrics: Dict[str, Dict] = defaultdict(lambda: {
            "messages_received": 0,
            "bytes_received": 0,
            "last_message": None,
            "message_rate": deque(maxlen=60),  # Last 60 seconds
            "errors": 0,
            "reconnections": 0,
            "latency_ms": deque(maxlen=100),
        })
        self.start_time = datetime.utcnow()
        
    async def setup(self):
        """Setup monitoring components."""
        print("\n" + "="*60)
        print("📡 WEBSOCKET CONNECTION MONITOR")
        print("="*60)
        
        stream_config = StreamConfig(
            enable_kalshi=True,
            enable_odds_polling=True,
            odds_poll_interval=60,
            correlation_window=10,
            divergence_threshold=0.05
        )
        
        self.stream_manager = UnifiedStreamManager(stream_config)
        
        # Register monitoring handlers
        self.stream_manager.on(EventType.PRICE_UPDATE, self.on_message)
        self.stream_manager.on(EventType.ORDERBOOK_UPDATE, self.on_message)
        self.stream_manager.on(EventType.TRADE, self.on_message)
        self.stream_manager.on(EventType.CONNECTION_ERROR, self.on_error)
        self.stream_manager.on(EventType.RECONNECTED, self.on_reconnect)
        
        print("✅ Monitor initialized")
    
    async def on_message(self, event):
        """Track message metrics."""
        source = event.get("source", "unknown")
        ticker = event.get("ticker", "")
        
        metrics = self.metrics[source]
        metrics["messages_received"] += 1
        metrics["last_message"] = datetime.utcnow()
        
        # Calculate message rate
        now = datetime.utcnow()
        metrics["message_rate"].append(now)
        
        # Estimate latency (if timestamp available)
        if "timestamp" in event:
            try:
                event_time = datetime.fromisoformat(event["timestamp"])
                latency = (now - event_time).total_seconds() * 1000
                metrics["latency_ms"].append(latency)
            except:
                pass
    
    async def on_error(self, event):
        """Track errors."""
        source = event.get("source", "unknown")
        self.metrics[source]["errors"] += 1
        print(f"❌ Error on {source}: {event.get('error')}")
    
    async def on_reconnect(self, event):
        """Track reconnections."""
        source = event.get("source", "unknown")
        self.metrics[source]["reconnections"] += 1
        print(f"🔄 Reconnected to {source}")
    
    def calculate_message_rate(self, timestamps: Deque) -> float:
        """Calculate messages per second."""
        if len(timestamps) < 2:
            return 0.0
        
        now = datetime.utcnow()
        recent = [t for t in timestamps if (now - t).total_seconds() <= 60]
        
        if not recent:
            return 0.0
        
        time_span = (now - recent[0]).total_seconds()
        if time_span > 0:
            return len(recent) / time_span
        return 0.0
    
    def get_average_latency(self, latencies: Deque) -> float:
        """Calculate average latency."""
        if not latencies:
            return 0.0
        return sum(latencies) / len(latencies)
    
    def display_metrics(self):
        """Display current metrics."""
        runtime = (datetime.utcnow() - self.start_time).total_seconds()
        
        print("\n" + "="*60)
        print(f"📊 METRICS (Runtime: {runtime:.0f}s)")
        print("="*60)
        
        for source, metrics in self.metrics.items():
            msg_rate = self.calculate_message_rate(metrics["message_rate"])
            avg_latency = self.get_average_latency(metrics["latency_ms"])
            
            print(f"\n📡 {source.upper()}")
            print(f"  Messages: {metrics['messages_received']:,}")
            print(f"  Rate: {msg_rate:.1f} msg/s")
            print(f"  Avg Latency: {avg_latency:.1f}ms")
            print(f"  Errors: {metrics['errors']}")
            print(f"  Reconnections: {metrics['reconnections']}")
            
            if metrics["last_message"]:
                age = (datetime.utcnow() - metrics["last_message"]).total_seconds()
                status = "🟢 Active" if age < 5 else "🟡 Idle" if age < 30 else "🔴 Stale"
                print(f"  Status: {status} (last msg {age:.1f}s ago)")
    
    async def run_monitor(self):
        """Run the connection monitor."""
        await self.setup()
        
        try:
            # Start stream manager
            print("\n🔌 Starting data streams...")
            await self.stream_manager.start()
            
            # Track some test markets
            print("📍 Tracking test markets...")
            test_markets = [
                ("NFLGAME-TEST-YES", "test_001"),
                ("NFLGAME-DEMO-YES", "test_002"),
            ]
            
            for kalshi_ticker, game_id in test_markets:
                try:
                    await self.stream_manager.track_market(
                        kalshi_ticker,
                        game_id,
                        [KalshiChannel.TICKER, KalshiChannel.ORDERBOOK_DELTA]
                    )
                    print(f"  ✓ Tracking {kalshi_ticker}")
                except Exception as e:
                    print(f"  ✗ Failed to track {kalshi_ticker}: {e}")
            
            print("\n" + "-"*60)
            print("Monitoring active. Press Ctrl+C to stop.")
            print("-"*60)
            
            # Display metrics periodically
            while True:
                await asyncio.sleep(10)
                self.display_metrics()
                
                # Health check
                if self.stream_manager.kalshi_ws:
                    kalshi_health = "🟢 Connected" if self.stream_manager.kalshi_ws.connected else "🔴 Disconnected"
                    print(f"\n💓 Health Check - Kalshi: {kalshi_health}")
                
        except KeyboardInterrupt:
            print("\n\n⏹️  Stopping monitor...")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("\n🧹 Cleaning up...")
            if self.stream_manager:
                await self.stream_manager.stop()
            
            # Final metrics
            self.display_metrics()
            print("\n✅ Monitor stopped!")


async def main():
    """Run the WebSocket monitor."""
    monitor = WebSocketMonitor()
    await monitor.run_monitor()


if __name__ == "__main__":
    asyncio.run(main())