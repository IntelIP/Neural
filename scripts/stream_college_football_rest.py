#!/usr/bin/env python3
"""
College Football REST API Market Streaming
==========================================
Stream live market data using REST API polling for College Football games.

This script demonstrates how to use the Neural SDK to:
- Discover college football game markets
- Filter by date, team, or conference
- Poll market data using REST API
- Display real-time price updates and changes
- Track volume and trading activity

Usage:
    python scripts/stream_college_football_rest.py [--date DATE] [--team TEAM] [--conference CONF] [--duration SECONDS]
    
Examples:
    # Stream today's games
    python scripts/stream_college_football_rest.py
    
    # Stream specific team
    python scripts/stream_college_football_rest.py --team "Ohio State"
    
    # Stream SEC games
    python scripts/stream_college_football_rest.py --conference SEC
    
    # Stream specific date
    python scripts/stream_college_football_rest.py --date 2025-09-13
"""

import asyncio
import logging
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any, Optional, List
import argparse
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from neural_sdk.streaming.rest_stream import RESTMarketStream, MarketSnapshot
from neural_sdk.data_pipeline.data_sources.kalshi.cfb_discovery import CFBMarketDiscovery
from neural_sdk.core.exceptions import ConfigurationError

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class CollegeFootballRESTStream:
    """Stream real-time market data for College Football games using REST API."""
    
    def __init__(self, poll_interval: float = 2.0, debug: bool = False):
        """
        Initialize the REST streaming client.
        
        Args:
            poll_interval: Seconds between polls
            debug: Enable debug logging if True
        """
        # Load environment variables
        load_dotenv()
        
        # Validate credentials
        if not os.getenv('KALSHI_API_KEY_ID'):
            raise ConfigurationError(
                "Missing KALSHI_API_KEY_ID in environment. "
                "Please add it to your .env file."
            )
        
        # Set logging level
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Initialize components
        self.streamer = RESTMarketStream(poll_interval=poll_interval)
        self.discovery = CFBMarketDiscovery()
        
        # Market tracking
        self.market_data = {}
        self.price_alerts = []
        self.session_start = None
        
        logger.info(f"✅ College Football streaming client initialized (poll interval: {poll_interval}s)")
    
    def setup_handlers(self):
        """Configure event handlers for market updates."""
        
        async def handle_market_update(data: Dict[str, Any]):
            """Process market updates."""
            ticker = data.get('ticker')
            snapshot: MarketSnapshot = data.get('snapshot')
            changes = data.get('changes', [])
            
            if not snapshot:
                return
            
            # Update tracking
            self.market_data[ticker] = snapshot
            
            # Display update if there are changes or initial snapshot
            if ticker not in self.market_data or changes:
                self.display_market_update(ticker, snapshot, changes)
        
        async def handle_price_change(data: Dict[str, Any]):
            """Process significant price changes."""
            ticker = data.get('ticker')
            field = data.get('field')
            old_price = data.get('old_price')
            new_price = data.get('new_price')
            change_percent = data.get('change_percent')
            
            # Alert on significant changes (> 2%)
            if change_percent and abs(change_percent) > 2:
                self.display_price_alert(ticker, field, old_price, new_price, change_percent)
        
        async def handle_error(data: Dict[str, Any]):
            """Handle errors."""
            ticker = data.get('ticker')
            error = data.get('error')
            logger.error(f"Error for {ticker}: {error}")
        
        # Set handlers
        self.streamer.on_market_update = handle_market_update
        self.streamer.on_price_change = handle_price_change
        self.streamer.on_error = handle_error
    
    def display_market_update(self, ticker: str, snapshot: MarketSnapshot, changes: List[Any]):
        """Display formatted market update."""
        # Extract team names from ticker (format: KXNCAAFGAME-DATE-TEAM1TEAM2-TEAM)
        parts = ticker.split('-')
        if len(parts) >= 3:
            teams = parts[-2]  # Team codes
            winner = parts[-1]  # Which team this market is for
            market_type = f"🏈 {winner} to Win"
        else:
            market_type = "🏈 College Football Market"
        
        # Show initial snapshot or updates with changes
        if ticker not in self.market_data or changes:
            print(f"\n{'='*60}")
            print(f"📊 {market_type}")
            print(f"   Ticker: {ticker}")
            print(f"   Time: {snapshot.timestamp.strftime('%H:%M:%S')}")
            
            # Display prices
            if snapshot.yes_price and snapshot.no_price:
                print(f"   YES: ${snapshot.yes_price:.4f} | NO: ${snapshot.no_price:.4f}")
                
                # Show implied probability
                implied_prob = snapshot.implied_probability * 100
                print(f"   Implied Probability: {implied_prob:.1f}%")
            elif snapshot.yes_price:
                print(f"   YES Price: ${snapshot.yes_price:.4f}")
            elif snapshot.no_price:
                print(f"   NO Price: ${snapshot.no_price:.4f}")
            else:
                print(f"   No active prices available")
            
            # Display bid/ask spread if available
            if snapshot.spread:
                print(f"   Spread: ${snapshot.spread:.4f}")
            
            # Display volume
            print(f"   Volume: {snapshot.volume:,} contracts")
            
            # Show changes if any
            if changes:
                price_changes = [c for c in changes if 'price' in c.field]
                for change in price_changes:
                    direction = "📈" if change.new_value > change.old_value else "📉"
                    field_name = change.field.replace('_', ' ').title()
                    print(f"   {direction} {field_name}: ${change.old_value:.4f} → ${change.new_value:.4f}")
    
    def display_price_alert(self, ticker: str, field: str, old_price: float, new_price: float, change_percent: float):
        """Display price alert for significant changes."""
        direction = "🚨📈" if new_price > old_price else "🚨📉"
        
        print(f"\n{direction} PRICE ALERT @ {datetime.now().strftime('%H:%M:%S')}")
        print(f"   Market: {ticker}")
        print(f"   {field.replace('_', ' ').title()}: ${old_price:.4f} → ${new_price:.4f}")
        print(f"   Change: {change_percent:+.2f}%")
        
        self.price_alerts.append({
            'time': datetime.now(),
            'ticker': ticker,
            'change': change_percent
        })
    
    async def discover_markets(
        self, 
        target_date: Optional[date] = None,
        team: Optional[str] = None,
        conference: Optional[str] = None
    ) -> List[str]:
        """
        Discover available college football markets.
        
        Args:
            target_date: Date to filter games (None for today)
            team: Specific team to filter
            conference: Conference to filter
            
        Returns:
            List of market tickers to stream
        """
        print("\n🔍 Discovering College Football markets...")
        print("="*60)
        
        try:
            markets_to_stream = []
            
            if team:
                # Get markets for specific team
                print(f"   Searching for {team} games...")
                events = self.discovery.get_team_events(team, status='open')
            elif conference:
                # Get markets for conference
                print(f"   Searching for {conference} games...")
                events = self.discovery.get_conference_events(conference, status='open')
            elif target_date:
                # Get markets for specific date
                print(f"   Searching for games on {target_date}...")
                events = self.discovery.get_events_by_date(target_date, status='open')
            else:
                # Get today's games
                print(f"   Searching for today's games...")
                events = self.discovery.get_events_by_date(datetime.now().date(), status='open')
            
            # Extract market tickers from events
            for event in events:
                markets = event.get('markets', [])
                for market in markets:
                    ticker = market.get('ticker')
                    if ticker:
                        markets_to_stream.append(ticker)
            
            # Remove duplicates
            markets_to_stream = list(set(markets_to_stream))
            
            if markets_to_stream:
                print(f"   ✓ Found {len(markets_to_stream)} markets")
                print("\n📋 Markets to stream:")
                for i, ticker in enumerate(markets_to_stream[:10], 1):
                    print(f"   {i}. {ticker}")
                if len(markets_to_stream) > 10:
                    print(f"   ... and {len(markets_to_stream) - 10} more")
            else:
                print("   ⚠️  No active markets found")
                
                # Show some available games
                print("\n📅 Available games:")
                all_events = self.discovery.get_all_cfb_events(status='open')
                for i, event in enumerate(all_events[:5], 1):
                    info = self.discovery.format_game_info(event)
                    print(f"   {i}. {info['title']} ({info['date']})")
            
            return markets_to_stream
            
        except Exception as e:
            logger.error(f"Market discovery error: {e}")
            return []
    
    async def stream_markets(
        self, 
        duration_seconds: int = 300,
        target_date: Optional[date] = None,
        team: Optional[str] = None,
        conference: Optional[str] = None
    ):
        """
        Stream market data for specified duration.
        
        Args:
            duration_seconds: Duration to stream in seconds
            target_date: Date to filter games
            team: Specific team to filter
            conference: Conference to filter
        """
        self.session_start = datetime.now()
        
        print(f"\n🚀 Starting REST API stream for {duration_seconds} seconds...")
        print("="*60)
        
        # Set up event handlers
        self.setup_handlers()
        
        try:
            # Discover markets
            markets = await self.discover_markets(target_date, team, conference)
            
            if not markets:
                print("\n⚠️  No markets to stream")
                return
            
            # Start streaming
            print(f"\n📡 Streaming {len(markets)} markets...")
            print("="*60)
            print("🎮 LIVE MARKET DATA (via REST API)")
            print("="*60)
            print("Press Ctrl+C to stop early\n")
            
            # Stream markets
            await self.streamer.stream_markets(
                tickers=markets,
                duration=duration_seconds
            )
            
        except KeyboardInterrupt:
            print("\n⚠️  Stream interrupted by user")
        except Exception as e:
            logger.error(f"❌ Streaming error: {e}")
        finally:
            # Display session summary
            self.display_summary()
    
    def display_summary(self):
        """Display streaming session summary."""
        if not self.session_start:
            return
        
        duration = (datetime.now() - self.session_start).total_seconds()
        
        print("\n" + "="*60)
        print("📊 STREAMING SESSION SUMMARY")
        print("="*60)
        print(f"⏱️  Duration: {duration:.0f} seconds")
        print(f"📈 Markets Tracked: {len(self.market_data)}")
        print(f"🔄 Polls: {self.streamer.polls_count}")
        print(f"📝 Changes Detected: {self.streamer.changes_count}")
        print(f"🚨 Price Alerts: {len(self.price_alerts)}")
        print(f"❌ Errors: {self.streamer.errors_count}")
        
        if self.market_data:
            print("\n🎯 Final Market Prices:")
            print("-"*40)
            
            for ticker, snapshot in sorted(self.market_data.items()):
                if snapshot.yes_price:
                    print(f"{ticker:40} ${snapshot.yes_price:.4f} ({snapshot.volume:,} vol)")
        
        if self.price_alerts:
            print("\n🚨 Significant Price Movements:")
            print("-"*40)
            for alert in self.price_alerts[:5]:
                print(f"   {alert['time'].strftime('%H:%M:%S')} - {alert['ticker']}: {alert['change']:+.2f}%")
        
        print("\n✅ Stream completed successfully!")


async def main():
    """Main function to run the streaming client."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Stream College Football market data via REST API'
    )
    parser.add_argument(
        '--date',
        type=str,
        help='Date to filter games (YYYY-MM-DD format)'
    )
    parser.add_argument(
        '--team',
        type=str,
        help='Specific team to stream'
    )
    parser.add_argument(
        '--conference',
        type=str,
        choices=['SEC', 'Big Ten', 'ACC', 'Big 12', 'Pac-12', 'Independent'],
        help='Conference to stream'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=300,
        help='Stream duration in seconds (default: 300)'
    )
    parser.add_argument(
        '--interval',
        type=float,
        default=2.0,
        help='Poll interval in seconds (default: 2.0)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Parse date if provided
    target_date = None
    if args.date:
        try:
            target_date = datetime.strptime(args.date, '%Y-%m-%d').date()
        except ValueError:
            print(f"❌ Invalid date format: {args.date}. Use YYYY-MM-DD")
            sys.exit(1)
    
    # Display header
    print("\n" + "="*60)
    print("🏈 COLLEGE FOOTBALL REST API STREAMING")
    print("="*60)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.team:
        print(f"🎯 Team: {args.team}")
    elif args.conference:
        print(f"🏆 Conference: {args.conference}")
    elif target_date:
        print(f"📅 Games on: {target_date}")
    else:
        print(f"📅 Today's Games")
    
    print(f"⏱️  Duration: {args.duration} seconds")
    print(f"🔄 Poll Interval: {args.interval} seconds")
    print("="*60)
    
    try:
        # Create and run streaming client
        streamer = CollegeFootballRESTStream(
            poll_interval=args.interval,
            debug=args.debug
        )
        await streamer.stream_markets(
            duration_seconds=args.duration,
            target_date=target_date,
            team=args.team,
            conference=args.conference
        )
    except ConfigurationError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("\n📝 Please ensure you have a .env file with:")
        print("   KALSHI_API_KEY_ID=your_key_id")
        print("   KALSHI_PRIVATE_KEY_FILE=path/to/private.key")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n👋 Stream stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())