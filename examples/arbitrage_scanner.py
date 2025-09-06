#!/usr/bin/env python3
"""
Real-Time Arbitrage Scanner

Scans for arbitrage opportunities between Kalshi and sportsbooks.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, List, Optional
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

from neural_sdk.data_sources.kalshi.websocket_adapter import KalshiWebSocketAdapter, KalshiChannel
from neural_sdk.data_sources.unified.stream_manager import UnifiedStreamManager, EventType, StreamConfig
from neural_sdk.data_sources.unified.stream_manager import UnifiedMarketData


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ArbitrageOpportunity:
    """Represents an arbitrage opportunity."""
    timestamp: datetime
    market_ticker: str
    game_id: str
    kalshi_price: float
    sportsbook_implied: float
    divergence: float
    expected_profit: float
    confidence: float
    recommendation: str
    
    @property
    def profit_percentage(self) -> float:
        """Calculate profit as percentage."""
        return self.expected_profit * 100


class ArbitrageScanner:
    """Scans for real-time arbitrage opportunities."""
    
    def __init__(self, min_divergence: float = 0.05, min_confidence: float = 0.7):
        self.stream_manager = None
        self.min_divergence = min_divergence
        self.min_confidence = min_confidence
        self.opportunities: List[ArbitrageOpportunity] = []
        self.active_arbs: Dict[str, ArbitrageOpportunity] = {}
        self.total_opportunities = 0
        self.profitable_opportunities = 0
        
    async def setup(self):
        """Setup scanner components."""
        print("\n" + "="*60)
        print("💰 REAL-TIME ARBITRAGE SCANNER")
        print("="*60)
        print(f"Min Divergence: {self.min_divergence:.1%}")
        print(f"Min Confidence: {self.min_confidence:.1%}")
        
        stream_config = StreamConfig(
            enable_kalshi=True,
            enable_odds_polling=True,
            odds_poll_interval=15,  # Fast polling for arbitrage
            correlation_window=3,    # Quick correlation
            divergence_threshold=self.min_divergence
        )
        
        self.stream_manager = UnifiedStreamManager(stream_config)
        
        # Register arbitrage handlers
        self.stream_manager.on(EventType.ARBITRAGE_OPPORTUNITY, self.on_arbitrage)
        self.stream_manager.on(EventType.DIVERGENCE_DETECTED, self.on_divergence)
        self.stream_manager.on(EventType.PRICE_UPDATE, self.on_price_update)
        
        print("✅ Scanner initialized")
    
    def calculate_arbitrage_profit(self, kalshi_price: float, sportsbook_implied: float) -> float:
        """Calculate potential arbitrage profit."""
        # Simple arbitrage calculation
        # Buy on Kalshi at kalshi_price, hedge on sportsbook
        # Profit = |kalshi_price - sportsbook_implied| - transaction_costs
        
        transaction_cost = 0.02  # 2% estimated transaction cost
        gross_profit = abs(kalshi_price - sportsbook_implied)
        net_profit = gross_profit - transaction_cost
        
        return max(0, net_profit)
    
    def generate_recommendation(self, opportunity: ArbitrageOpportunity) -> str:
        """Generate trading recommendation."""
        if opportunity.kalshi_price < opportunity.sportsbook_implied:
            action = "BUY on Kalshi"
            hedge = "SELL equivalent on sportsbook"
        else:
            action = "SELL on Kalshi"
            hedge = "BUY equivalent on sportsbook"
        
        return f"{action} @ ${opportunity.kalshi_price:.4f}, {hedge}"
    
    async def on_arbitrage(self, event):
        """Handle arbitrage opportunity."""
        ticker = event.get("ticker")
        data: UnifiedMarketData = event.get("data")
        
        if not data or not data.kalshi_yes_price or not data.odds_implied_prob_home:
            return
        
        divergence = abs(data.kalshi_yes_price - data.odds_implied_prob_home)
        
        if divergence >= self.min_divergence:
            profit = self.calculate_arbitrage_profit(
                data.kalshi_yes_price,
                data.odds_implied_prob_home
            )
            
            if profit > 0:
                confidence = min(1.0, divergence / 0.10)  # Max confidence at 10% divergence
                
                opportunity = ArbitrageOpportunity(
                    timestamp=datetime.utcnow(),
                    market_ticker=ticker,
                    game_id=data.game_id or "",
                    kalshi_price=data.kalshi_yes_price,
                    sportsbook_implied=data.odds_implied_prob_home,
                    divergence=divergence,
                    expected_profit=profit,
                    confidence=confidence,
                    recommendation=""
                )
                
                opportunity.recommendation = self.generate_recommendation(opportunity)
                
                self.opportunities.append(opportunity)
                self.active_arbs[ticker] = opportunity
                self.total_opportunities += 1
                
                if profit > 0.01:  # Profitable after costs
                    self.profitable_opportunities += 1
                
                self.display_opportunity(opportunity)
    
    async def on_divergence(self, event):
        """Handle divergence detection."""
        ticker = event.get("ticker")
        divergence = event.get("divergence", 0)
        
        if divergence >= self.min_divergence:
            print(f"⚠️  Divergence Alert: {ticker} - {divergence:.1%}")
    
    async def on_price_update(self, event):
        """Monitor price updates for existing opportunities."""
        ticker = event.get("ticker")
        
        if ticker in self.active_arbs:
            data = event.get("data")
            if data and data.kalshi_yes_price and data.odds_implied_prob_home:
                current_divergence = abs(data.kalshi_yes_price - data.odds_implied_prob_home)
                
                if current_divergence < self.min_divergence * 0.5:
                    # Opportunity closed
                    print(f"❌ Arbitrage closed: {ticker}")
                    del self.active_arbs[ticker]
    
    def display_opportunity(self, opp: ArbitrageOpportunity):
        """Display arbitrage opportunity."""
        print("\n" + "="*60)
        print("🎯 ARBITRAGE OPPORTUNITY DETECTED!")
        print("="*60)
        print(f"Market: {opp.market_ticker}")
        print(f"Kalshi Price: ${opp.kalshi_price:.4f}")
        print(f"Sportsbook Implied: ${opp.sportsbook_implied:.4f}")
        print(f"Divergence: {opp.divergence:.1%}")
        print(f"Expected Profit: {opp.profit_percentage:.2f}%")
        print(f"Confidence: {opp.confidence:.1%}")
        print(f"📋 Recommendation: {opp.recommendation}")
        print("="*60)
    
    def display_summary(self):
        """Display scanning summary."""
        print("\n" + "-"*60)
        print("📊 SCANNING SUMMARY")
        print("-"*60)
        print(f"Total Opportunities: {self.total_opportunities}")
        print(f"Profitable (>1%): {self.profitable_opportunities}")
        print(f"Active Arbs: {len(self.active_arbs)}")
        
        if self.opportunities:
            avg_profit = sum(o.profit_percentage for o in self.opportunities) / len(self.opportunities)
            max_profit = max(o.profit_percentage for o in self.opportunities)
            print(f"Avg Profit: {avg_profit:.2f}%")
            print(f"Max Profit: {max_profit:.2f}%")
        
        if self.active_arbs:
            print("\n🔥 Active Opportunities:")
            for ticker, opp in self.active_arbs.items():
                print(f"  • {ticker}: {opp.divergence:.1%} divergence, {opp.profit_percentage:.2f}% profit")
    
    async def scan_markets(self, markets: List[tuple]):
        """Scan specific markets for arbitrage."""
        print(f"\n📡 Scanning {len(markets)} markets...")
        
        for kalshi_ticker, game_id in markets:
            try:
                await self.stream_manager.track_market(
                    kalshi_ticker,
                    game_id,
                    [KalshiChannel.TICKER, KalshiChannel.ORDERBOOK_DELTA]
                )
                print(f"  ✓ Tracking {kalshi_ticker}")
            except Exception as e:
                print(f"  ✗ Failed to track {kalshi_ticker}: {e}")
        
        print("✅ Market scanning initiated")
    
    async def run_scanner(self):
        """Run the arbitrage scanner."""
        await self.setup()
        
        try:
            # Start stream manager
            print("\n🔌 Starting data streams...")
            await self.stream_manager.start()
            
            # Example markets to scan (NFL games)
            markets_to_scan = [
                ("NFLGAME-KC-BUF-YES", "nfl_001"),
                ("NFLGAME-DAL-PHI-YES", "nfl_002"),
                ("NFLGAME-SF-SEA-YES", "nfl_003"),
                ("NFLGAME-GB-CHI-YES", "nfl_004"),
                ("NFLGAME-TB-NO-YES", "nfl_005"),
            ]
            
            await self.scan_markets(markets_to_scan)
            
            print("\n" + "="*60)
            print("🔍 ARBITRAGE SCANNER ACTIVE")
            print("="*60)
            print("Monitoring for opportunities...")
            print("Press Ctrl+C to stop")
            print("-"*60)
            
            # Run scanner
            scan_interval = 30  # Display summary every 30 seconds
            last_summary = datetime.utcnow()
            
            while True:
                await asyncio.sleep(5)
                
                # Display periodic summary
                if (datetime.utcnow() - last_summary).total_seconds() >= scan_interval:
                    self.display_summary()
                    last_summary = datetime.utcnow()
                
        except KeyboardInterrupt:
            print("\n\n⏹️  Stopping scanner...")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("\n🧹 Cleaning up...")
            if self.stream_manager:
                await self.stream_manager.stop()
            
            # Final summary
            print("\n" + "="*60)
            print("📈 FINAL ARBITRAGE REPORT")
            print("="*60)
            print(f"Total Opportunities Found: {self.total_opportunities}")
            print(f"Profitable Opportunities: {self.profitable_opportunities}")
            
            if self.opportunities:
                # Top opportunities
                top_opps = sorted(self.opportunities, key=lambda x: x.profit_percentage, reverse=True)[:5]
                print("\n🏆 Top 5 Opportunities:")
                for i, opp in enumerate(top_opps, 1):
                    print(f"{i}. {opp.market_ticker}: {opp.profit_percentage:.2f}% profit")
            
            print("\n✅ Scanner completed!")


async def main():
    """Run the arbitrage scanner."""
    scanner = ArbitrageScanner(
        min_divergence=0.03,  # 3% minimum divergence
        min_confidence=0.6    # 60% minimum confidence
    )
    await scanner.run_scanner()


if __name__ == "__main__":
    asyncio.run(main())