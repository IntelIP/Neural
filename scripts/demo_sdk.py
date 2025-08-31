#!/usr/bin/env python3
"""
SDK Demo - Shows how to use the Data Source SDK
"""

import asyncio
import logging
from typing import Dict, Any
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sdk import SDKManager, StandardizedEvent, EventType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EventProcessor:
    """Process events from data sources"""
    
    def __init__(self):
        self.event_counts = {}
        self.high_impact_events = []
        
    async def process_event(self, event: StandardizedEvent):
        """
        Process a standardized event
        
        Args:
            event: Event from a data source
        """
        # Track event counts
        source = event.source
        self.event_counts[source] = self.event_counts.get(source, 0) + 1
        
        # Log high-impact events
        if event.impact in ["high", "critical"]:
            self.high_impact_events.append(event)
            logger.info(
                f"🔴 HIGH IMPACT: {event.source} - {event.event_type.value}\n"
                f"   Confidence: {event.confidence:.2f}\n"
                f"   Data: {event.data}"
            )
        
        # Process by event type
        if event.event_type == EventType.ODDS_CHANGE:
            await self.handle_odds_change(event)
        elif event.event_type == EventType.SENTIMENT_SHIFT:
            await self.handle_sentiment_shift(event)
        elif event.event_type == EventType.WEATHER_UPDATE:
            await self.handle_weather_update(event)
    
    async def handle_odds_change(self, event: StandardizedEvent):
        """Handle odds change event from sportsbook"""
        data = event.data
        logger.info(
            f"📊 Odds Change: {data.get('event_name')}\n"
            f"   Market: {data.get('market_type')}\n"
            f"   Change: {data.get('previous_odds'):.3f} → {data.get('current_odds'):.3f}\n"
            f"   Direction: {data.get('direction')}"
        )
        
        # Check for arbitrage opportunity
        if event.confidence > 0.8:
            logger.warning("⚡ Potential arbitrage opportunity detected!")
    
    async def handle_sentiment_shift(self, event: StandardizedEvent):
        """Handle sentiment shift from Reddit"""
        data = event.data
        logger.info(
            f"💬 Sentiment Shift: {data.get('thread_title', 'Unknown')}\n"
            f"   Shift: {data.get('shift'):.3f} ({data.get('direction')})\n"
            f"   Sample: {data.get('comment_sample', [''])[0][:50]}..."
        )
    
    async def handle_weather_update(self, event: StandardizedEvent):
        """Handle weather update"""
        data = event.data
        logger.info(
            f"🌤️ Weather Alert: {data.get('stadium')}\n"
            f"   Condition: {data.get('condition')}\n"
            f"   Impact: {event.metadata.get('affects', [])}"
        )
    
    def print_summary(self):
        """Print processing summary"""
        print("\n" + "="*50)
        print("EVENT PROCESSING SUMMARY")
        print("="*50)
        
        print("\nEvent Counts by Source:")
        for source, count in self.event_counts.items():
            print(f"  {source}: {count} events")
        
        print(f"\nHigh Impact Events: {len(self.high_impact_events)}")
        for event in self.high_impact_events[-5:]:  # Last 5
            print(f"  - {event.source}: {event.event_type.value}")
        
        print("="*50)


async def demo_basic_usage():
    """Demonstrate basic SDK usage"""
    print("\n" + "="*50)
    print("NEURAL TRADING PLATFORM - DATA SOURCE SDK DEMO")
    print("="*50)
    
    # Initialize SDK Manager
    sdk = SDKManager(config_path="config/data_sources.yaml")
    processor = EventProcessor()
    
    try:
        # Initialize adapters
        print("\n📡 Initializing data sources...")
        await sdk.initialize()
        
        # Show loaded adapters
        print(f"\n✅ Loaded {len(sdk.adapters)} adapters:")
        for name, adapter in sdk.adapters.items():
            metadata = adapter.metadata
            print(f"  • {metadata.name} v{metadata.version}")
            print(f"    Type: {metadata.source_type}")
            print(f"    Latency: {metadata.latency_ms}ms")
            print(f"    Reliability: {metadata.reliability:.1%}")
        
        # Start streaming
        print("\n🚀 Starting data streams...")
        await sdk.start()
        
        # Process events for demo duration
        print("\n📊 Processing events (30 seconds)...")
        print("-" * 50)
        
        start_time = asyncio.get_event_loop().time()
        duration = 30  # seconds
        
        async for event in sdk.get_events():
            await processor.process_event(event)
            
            # Check if demo duration exceeded
            if asyncio.get_event_loop().time() - start_time > duration:
                break
        
        # Print summary
        processor.print_summary()
        
        # Health check
        print("\n🏥 Health Check:")
        health = await sdk.health_check()
        for adapter_name, status in health['adapters'].items():
            print(f"  {adapter_name}: {'✅' if status['healthy'] else '❌'} "
                  f"({status['event_count']} events)")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo error: {e}")
    finally:
        # Cleanup
        print("\n🛑 Stopping SDK...")
        await sdk.stop()
        print("✅ SDK stopped cleanly")


async def demo_custom_adapter():
    """Demonstrate creating a custom adapter"""
    print("\n" + "="*50)
    print("CUSTOM ADAPTER DEMO")
    print("="*50)
    
    from src.sdk import DataSourceAdapter, DataSourceMetadata, StandardizedEvent, EventType
    
    class CustomNewsAdapter(DataSourceAdapter):
        """Example custom adapter for news feeds"""
        
        def get_metadata(self) -> DataSourceMetadata:
            return DataSourceMetadata(
                name="CustomNews",
                version="1.0.0",
                author="Demo",
                description="Custom news feed adapter",
                source_type="news",
                latency_ms=1000,
                reliability=0.95,
                requires_auth=False
            )
        
        async def connect(self) -> bool:
            logger.info("Custom adapter connected!")
            return True
        
        async def disconnect(self) -> None:
            logger.info("Custom adapter disconnected")
        
        async def validate_connection(self) -> bool:
            return True
        
        async def stream(self):
            """Generate fake news events"""
            for i in range(5):
                yield StandardizedEvent(
                    source="CustomNews",
                    event_type=EventType.NEWS_ALERT,
                    timestamp=datetime.now(),
                    data={
                        "headline": f"Breaking: Team makes major announcement #{i+1}",
                        "impact": "medium"
                    },
                    confidence=0.75,
                    impact="medium"
                )
                await asyncio.sleep(2)
        
        def transform(self, raw_data):
            return None
    
    # Use the custom adapter
    adapter = CustomNewsAdapter({})
    await adapter.start()
    
    print("\n📰 Streaming from custom adapter...")
    async for event in adapter.stream():
        print(f"  Received: {event.data['headline']}")
    
    await adapter.stop()
    print("✅ Custom adapter demo complete")


async def demo_testing():
    """Demonstrate adapter testing"""
    print("\n" + "="*50)
    print("ADAPTER TESTING DEMO")
    print("="*50)
    
    sdk = SDKManager(config_path="config/data_sources.yaml")
    await sdk.initialize()
    
    # Test DraftKings adapter if available
    if "draftkings" in sdk.adapters:
        print("\n🧪 Testing DraftKings adapter...")
        results = await sdk.test_adapter("draftkings", duration=10)
        
        print(f"\nTest Results:")
        print(f"  Events received: {results['events_received']}")
        print(f"  Events/second: {results['events_per_second']:.2f}")
        print(f"  Errors: {results['errors']}")
        
        if results.get('avg_latency_ms'):
            print(f"  Avg latency: {results['avg_latency_ms']:.0f}ms")
        
        if results.get('event_types'):
            print(f"  Event types: {list(results['event_types'].keys())}")
    
    await sdk.stop()
    print("✅ Testing complete")


async def main():
    """Run all demos"""
    
    # Basic usage
    await demo_basic_usage()
    
    # Custom adapter
    print("\n" + "="*50)
    input("Press Enter to continue to custom adapter demo...")
    await demo_custom_adapter()
    
    # Testing
    print("\n" + "="*50)
    input("Press Enter to continue to testing demo...")
    await demo_testing()
    
    print("\n" + "="*50)
    print("🎉 SDK DEMO COMPLETE!")
    print("="*50)
    print("\nNext steps:")
    print("1. Add your API credentials to config/data_sources.yaml")
    print("2. Enable the adapters you want to use")
    print("3. Create custom adapters for your data sources")
    print("4. Integrate with the trading platform")


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════╗
    ║     NEURAL TRADING PLATFORM - DATA SOURCE SDK       ║
    ║                                                      ║
    ║  This demo shows how to:                           ║
    ║  • Use the SDK to stream from multiple sources     ║
    ║  • Create custom data adapters                     ║
    ║  • Process standardized events                     ║
    ║  • Test adapter performance                        ║
    ╚══════════════════════════════════════════════════════╝
    """)
    
    asyncio.run(main())