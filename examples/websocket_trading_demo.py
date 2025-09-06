#!/usr/bin/env python3
"""
WebSocket Real-Time Trading Demo

Demonstrates the complete WebSocket infrastructure with real-time trading.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neural_sdk.data_sources.kalshi.websocket_adapter import KalshiWebSocketAdapter, KalshiChannel
from neural_sdk.data_sources.unified.stream_manager import UnifiedStreamManager, EventType, StreamConfig
from neural_sdk.trading.real_time_engine import (
    RealTimeTradingEngine, SignalType, TradingSignal, RiskLimits
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Example trading strategies

async def momentum_strategy(market_data, engine):
    """
    Simple momentum strategy.
    Buy when price is rising rapidly.
    """
    ticker = market_data.ticker
    
    # Get price history
    history = engine.stream_manager.get_market_history(ticker, limit=10)
    if len(history) < 5:
        return None
    
    # Calculate momentum
    recent_prices = [h.kalshi_yes_price for h in history[-5:] if h.kalshi_yes_price]
    if len(recent_prices) < 3:
        return None
    
    # Check if price is rising
    if recent_prices[-1] > recent_prices[0] * 1.02:  # 2% increase
        return TradingSignal(
            signal_id=f"momentum_{datetime.utcnow().timestamp()}",
            timestamp=datetime.utcnow(),
            market_ticker=ticker,
            signal_type=SignalType.BUY,
            confidence=0.7,
            size=50,
            reason="Positive momentum detected"
        )
    
    return None


async def arbitrage_strategy(market_data, engine):
    """
    Arbitrage strategy based on Kalshi vs sportsbook odds.
    """
    if not market_data.arbitrage_exists:
        return None
    
    # Check divergence threshold
    if market_data.divergence_score and market_data.divergence_score > 0.08:  # 8% divergence
        return TradingSignal(
            signal_id=f"arb_{datetime.utcnow().timestamp()}",
            timestamp=datetime.utcnow(),
            market_ticker=market_data.ticker,
            signal_type=SignalType.BUY,
            confidence=0.9,
            size=100,
            reason=f"Arbitrage: {market_data.divergence_score:.1%} divergence",
            metadata={
                "kalshi_price": market_data.kalshi_yes_price,
                "odds_implied": market_data.odds_implied_prob_home
            }
        )
    
    return None


async def mean_reversion_strategy(market_data, engine):
    """
    Mean reversion strategy.
    Trade when price deviates significantly from average.
    """
    ticker = market_data.ticker
    
    # Calculate volatility
    volatility = engine.stream_manager.calculate_volatility(ticker, window=20)
    if not volatility or volatility < 0.01:
        return None
    
    # Get average price
    history = engine.stream_manager.get_market_history(ticker, limit=20)
    if len(history) < 10:
        return None
    
    prices = [h.kalshi_yes_price for h in history if h.kalshi_yes_price]
    if not prices:
        return None
    
    avg_price = sum(prices) / len(prices)
    current_price = market_data.kalshi_yes_price
    
    if not current_price:
        return None
    
    # Check for mean reversion opportunity
    deviation = (current_price - avg_price) / avg_price
    
    if deviation < -0.05:  # Price 5% below average
        return TradingSignal(
            signal_id=f"meanrev_{datetime.utcnow().timestamp()}",
            timestamp=datetime.utcnow(),
            market_ticker=ticker,
            signal_type=SignalType.BUY,
            confidence=0.6,
            size=30,
            reason=f"Mean reversion: {deviation:.1%} below average"
        )
    elif deviation > 0.05 and ticker in engine.get_positions():  # Price 5% above average
        return TradingSignal(
            signal_id=f"meanrev_sell_{datetime.utcnow().timestamp()}",
            timestamp=datetime.utcnow(),
            market_ticker=ticker,
            signal_type=SignalType.SELL,
            confidence=0.6,
            reason=f"Mean reversion: {deviation:.1%} above average"
        )
    
    return None


class TradingDemo:
    """Demo application for WebSocket trading."""
    
    def __init__(self):
        self.stream_manager = None
        self.trading_engine = None
        self.kalshi_ws = None
        
    async def setup(self):
        """Setup demo components."""
        print("\n" + "="*60)
        print("🚀 WEBSOCKET REAL-TIME TRADING DEMO")
        print("="*60)
        
        # Configure stream manager
        stream_config = StreamConfig(
            enable_kalshi=True,
            enable_odds_polling=True,
            odds_poll_interval=30,
            correlation_window=5,
            divergence_threshold=0.05
        )
        
        # Initialize components
        self.stream_manager = UnifiedStreamManager(stream_config)
        
        # Configure risk limits
        risk_limits = RiskLimits(
            max_position_size=200,
            max_order_size=50,
            max_daily_loss=500.0,
            max_daily_trades=20,
            max_open_positions=5,
            stop_loss_percentage=0.10,
            take_profit_percentage=0.15
        )
        
        # Initialize trading engine
        self.trading_engine = RealTimeTradingEngine(
            stream_manager=self.stream_manager,
            risk_limits=risk_limits
        )
        
        # Add strategies
        self.trading_engine.add_strategy(momentum_strategy)
        self.trading_engine.add_strategy(arbitrage_strategy)
        self.trading_engine.add_strategy(mean_reversion_strategy)
        
        # Register event handlers
        self.setup_event_handlers()
        
        print("✅ Demo components initialized")
    
    def setup_event_handlers(self):
        """Setup event handlers for monitoring."""
        # Stream events
        self.stream_manager.on(EventType.PRICE_UPDATE, self.on_price_update)
        self.stream_manager.on(EventType.ARBITRAGE_OPPORTUNITY, self.on_arbitrage)
        self.stream_manager.on(EventType.DIVERGENCE_DETECTED, self.on_divergence)
        
        # Trading events
        self.trading_engine.on_signal(self.on_trading_signal)
        self.trading_engine.on_order(self.on_order)
    
    async def on_price_update(self, event):
        """Handle price updates."""
        data = event.get("data")
        if data and data.kalshi_yes_price:
            print(f"📊 {data.ticker}: ${data.kalshi_yes_price:.4f} "
                  f"(Vol: {data.kalshi_volume or 0:.0f})")
    
    async def on_arbitrage(self, event):
        """Handle arbitrage opportunities."""
        ticker = event.get("ticker")
        data = event.get("data")
        print(f"💰 ARBITRAGE: {ticker} - "
              f"Kalshi: ${data.kalshi_yes_price:.4f}, "
              f"Odds implied: {data.odds_implied_prob_home:.4f}")
    
    async def on_divergence(self, event):
        """Handle divergence detection."""
        ticker = event.get("ticker")
        divergence = event.get("divergence")
        print(f"📈 DIVERGENCE: {ticker} - {divergence:.1%}")
    
    async def on_trading_signal(self, signal):
        """Handle trading signals."""
        print(f"🎯 SIGNAL: {signal.signal_type.value.upper()} {signal.market_ticker} "
              f"(Confidence: {signal.confidence:.1%}) - {signal.reason}")
    
    async def on_order(self, order):
        """Handle order updates."""
        print(f"📝 ORDER: {order.status.value} - {order.side} {order.size} "
              f"{order.market_ticker} @ ${order.average_fill_price or 0:.4f}")
    
    async def run_demo(self):
        """Run the trading demo."""
        # Setup components
        await self.setup()
        
        try:
            # Start stream manager
            print("\n🔌 Starting data streams...")
            await self.stream_manager.start()
            
            # Start trading engine
            print("💹 Starting trading engine...")
            await self.trading_engine.start()
            
            # Track some markets (example tickers)
            print("\n📍 Tracking markets...")
            
            # Example: Track NFL game markets
            markets_to_track = [
                ("NFLGAME-ARI-NO-YES", "game_001"),  # Example mapping
                ("NFLGAME-KC-BUF-YES", "game_002"),
                ("NFLGAME-DAL-PHI-YES", "game_003")
            ]
            
            for kalshi_ticker, game_id in markets_to_track:
                try:
                    await self.stream_manager.track_market(
                        kalshi_ticker,
                        game_id,
                        [KalshiChannel.TICKER, KalshiChannel.ORDERBOOK_DELTA]
                    )
                    print(f"  ✓ Tracking {kalshi_ticker}")
                except Exception as e:
                    print(f"  ✗ Failed to track {kalshi_ticker}: {e}")
            
            # Display status
            print("\n" + "="*60)
            print("🎮 LIVE TRADING ACTIVE")
            print("="*60)
            print("Strategies: Momentum, Arbitrage, Mean Reversion")
            print("Risk Limits: $500 daily loss, 20 trades/day")
            print("Press Ctrl+C to stop")
            print("-"*60)
            
            # Run for demo duration
            start_time = datetime.utcnow()
            
            while True:
                await asyncio.sleep(10)
                
                # Print periodic status
                runtime = (datetime.utcnow() - start_time).total_seconds()
                stats = self.trading_engine.get_stats()
                positions = self.trading_engine.get_positions()
                
                print(f"\n📊 STATUS ({runtime:.0f}s)")
                print(f"  Trades: {stats.total_trades} | "
                      f"Win Rate: {stats.win_rate:.1%}" if stats.win_rate else "N/A")
                print(f"  P&L: ${stats.total_pnl:.2f} | "
                      f"Positions: {len(positions)}")
                
                # Show positions
                if positions:
                    print("  Positions:")
                    for ticker, pos in positions.items():
                        print(f"    • {ticker}: {pos.size} @ ${pos.average_price:.4f} "
                              f"(P&L: ${pos.unrealized_pnl:.2f})")
                
        except KeyboardInterrupt:
            print("\n\n⏹️  Stopping demo...")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Cleanup
            print("\n🧹 Cleaning up...")
            
            # Stop trading engine
            if self.trading_engine:
                await self.trading_engine.stop()
            
            # Stop stream manager
            if self.stream_manager:
                await self.stream_manager.stop()
            
            # Final stats
            if self.trading_engine:
                stats = self.trading_engine.get_stats()
                print("\n" + "="*60)
                print("📈 FINAL STATISTICS")
                print("="*60)
                print(f"Total Trades: {stats.total_trades}")
                print(f"Winning Trades: {stats.winning_trades}")
                print(f"Losing Trades: {stats.losing_trades}")
                print(f"Win Rate: {stats.win_rate:.1%}" if stats.win_rate else "N/A")
                print(f"Total P&L: ${stats.total_pnl:.2f}")
                print(f"Daily P&L: ${stats.daily_pnl:.2f}")
            
            print("\n✅ Demo completed!")


async def main():
    """Run the WebSocket trading demo."""
    demo = TradingDemo()
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())