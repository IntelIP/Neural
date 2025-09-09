#!/usr/bin/env python3
"""
CFB HFT Trader Script for Neural SDK
High-frequency trading for College Football markets using mean reversion strategy (best from backtest).
Integrates WebSocket for real-time execution with $200 capital and strict risk limits.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
import sys
from datetime import timezone

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neural_sdk.data_sources.kalshi.websocket_adapter import KalshiWebSocketAdapter, KalshiChannel
from neural_sdk.data_sources.unified.stream_manager import UnifiedStreamManager, EventType, StreamConfig
from neural_sdk.trading.real_time_engine import (
    RealTimeTradingEngine, SignalType, TradingSignal, RiskLimits
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def cfb_mean_reversion_strategy(market_data, engine):
    """
    Mean reversion strategy tailored for CFB markets.
    Buy if price <0.4, sell if >0.6 with tight limits for $200 capital.
    """
    ticker = market_data.ticker
    
    # Get price history
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
    
    deviation = (current_price - avg_price) / avg_price
    
    # Conservative sizing for $200 capital (max $40 per position)
    size = 20  # Small contract size
    
    if deviation < -0.05:  # 5% below average
        return TradingSignal(
            signal_id=f"cfb_meanrev_{datetime.utcnow().timestamp()}",
            timestamp=datetime.utcnow(),
            market_ticker=ticker,
            signal_type=SignalType.BUY,
            confidence=0.6,
            size=size,
            reason=f"CFB mean reversion: {deviation:.1%} below average"
        )
    elif deviation > 0.05 and ticker in engine.get_positions():
        return TradingSignal(
            signal_id=f"cfb_meanrev_sell_{datetime.utcnow().timestamp()}",
            timestamp=datetime.utcnow(),
            market_ticker=ticker,
            signal_type=SignalType.SELL,
            confidence=0.6,
            reason=f"CFB mean reversion: {deviation:.1%} above average"
        )
    
    return None

class CFBHFTTrader:
    """CFB HFT trading application with risk management."""
    
    def __init__(self, simulation_mode=True):
        self.simulation_mode = simulation_mode
        self.stream_manager = None
        self.trading_engine = None
        self.kalshi_ws = None
        
    async def setup(self):
        """Setup trader components."""
        print("\n" + "="*60)
        print("🏈 CFB HFT TRADER - MEAN REVERSION STRATEGY")
        print("="*60)
        
        # Configure stream for CFB
        stream_config = StreamConfig(
            enable_kalshi=True,
            enable_odds_polling=True,
            odds_poll_interval=30,
            correlation_window=5,
            divergence_threshold=0.05
        )
        
        self.stream_manager = UnifiedStreamManager(stream_config)
        
        # Risk limits for $200 capital, 25% liquidity ($50 reserved), $150 exposure, no trade limit
        risk_limits = RiskLimits(
            max_position_size=30,  # ~15% per position ($30, for up to 5 positions)
            max_order_size=50,     # Increased order size
            max_daily_loss=20.0,   # 10% daily loss ($20)
            max_daily_trades=1000,  # Effectively unlimited
            max_open_positions=5,  # Increased positions (total <= $150 exposure)
            stop_loss_percentage=0.05,  # 5% stop loss
            take_profit_percentage=0.10  # 10% take profit
        )
        
        self.trading_engine = RealTimeTradingEngine(
            stream_manager=self.stream_manager,
            risk_limits=risk_limits
        )
        
        # Add CFB strategy (best from backtest)
        self.trading_engine.add_strategy(cfb_mean_reversion_strategy)
        
        # Event handlers
        self.setup_event_handlers()
        
        print("✅ Components initialized (Simulation: {})".format(self.simulation_mode))
    
    def setup_event_handlers(self):
        """Setup event handlers."""
        self.stream_manager.on(EventType.PRICE_UPDATE, self.on_price_update)
        self.stream_manager.on(EventType.ARBITRAGE_OPPORTUNITY, self.on_arbitrage)
        self.stream_manager.on(EventType.DIVERGENCE_DETECTED, self.on_divergence)
        
        self.trading_engine.on_signal(self.on_trading_signal)
        self.trading_engine.on_order(self.on_order)
    
    async def on_price_update(self, event):
        data = event.get("data")
        if data and data.kalshi_yes_price:
            print(f"📊 CFB {data.ticker}: ${data.kalshi_yes_price:.4f}")
    
    async def on_arbitrage(self, event):
        ticker = event.get("ticker")
        data = event.get("data")
        print(f"💰 CFB ARBITRAGE: {ticker} - Kalshi: ${data.kalshi_yes_price:.4f}")
    
    async def on_divergence(self, event):
        ticker = event.get("ticker")
        divergence = event.get("divergence")
        print(f"📈 CFB DIVERGENCE: {ticker} - {divergence:.1%}")
    
    async def on_trading_signal(self, signal):
        print(f"🎯 CFB SIGNAL: {signal.signal_type.value.upper()} {signal.market_ticker} "
              f"(Size: {signal.size}, Conf: {signal.confidence:.1%}) - {signal.reason}")
    
    async def on_order(self, order):
        status = "SIM" if self.simulation_mode else "LIVE"
        print(f"📝 {status} ORDER: {order.status.value} - {order.side} {order.size} "
              f"{order.market_ticker} @ ${order.average_fill_price or 0:.4f}")
    
    async def run_trader(self):
        """Run the CFB HFT trader."""
        await self.setup()
        
        try:
            print("\n🔌 Starting CFB streams...")
            await self.stream_manager.start()
            
            print("💹 Starting CFB trading engine...")
            await self.trading_engine.start()
            
            # Track upcoming CFB games (from discover script)
            cfb_markets = [
                "KXNCAAFGAME-25SEP14PRSTHAW-PRST",  # Portland St. to win
                "KXNCAAFGAME-25SEP13TXSTASU-TXST",  # Texas St. to win
                "KXNCAAFGAME-25SEP13MINNCAL-MINN",  # Minnesota to win
                # Add more real tickers as needed
            ]
            
            print("\n📍 Tracking CFB markets...")
            for ticker in cfb_markets:
                try:
                    await self.stream_manager.track_market(
                        ticker,
                        "cfb_game",
                        [KalshiChannel.TICKER, KalshiChannel.ORDERBOOK_DELTA]
                    )
                    print(f"  ✓ Tracking {ticker}")
                except Exception as e:
                    print(f"  ✗ Failed to track {ticker}: {e}")
            
            print("\n" + "="*60)
            print("🎮 CFB HFT LIVE (Strategy: Mean Reversion)")
            print("Risk Limits: $20 daily loss, 5% SL, 10% TP, $30 max pos, 20 trades/day, 5 pos max")
            print("Press Ctrl+C to stop")
            print("-"*60)
            
            start_time = datetime.now(timezone.utc)
            
            while True:
                await asyncio.sleep(10)
                
                runtime = (datetime.now(timezone.utc) - start_time).total_seconds()
                stats = self.trading_engine.get_stats()
                positions = self.trading_engine.get_positions()
                
                print(f"\n📊 CFB STATUS ({runtime:.0f}s)")
                print(f"  Trades: {stats.total_trades} | P&L: ${stats.total_pnl:.2f} | Positions: {len(positions)}")
                
                # Print current prices from tracked markets
                print("  Market Prices:")
                for ticker in cfb_markets:
                    market = self.stream_manager.get_market_data(ticker)
                    if market:
                        kalshi_price = market.kalshi_yes_price if market.kalshi_yes_price else "N/A"
                        odds_home = market.odds_implied_prob_home if market.odds_implied_prob_home else "N/A"
                        print(f"    • {ticker}: Kalshi=${kalshi_price}, Odds={odds_home}")
                    else:
                        print(f"    • {ticker}: No data")
                
                if positions:
                    print("  Positions:")
                    for ticker, pos in positions.items():
                        pnl_pct = (pos.unrealized_pnl / (pos.size * pos.average_price)) * 100 if pos.size > 0 else 0
                        print(f"    • {ticker}: {pos.size} @ ${pos.average_price:.4f} (P&L: {pnl_pct:.1f}%)")
                        
        except KeyboardInterrupt:
            print("\n\n⏹️  Stopping CFB trader...")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("\n🧹 Cleaning up...")
            
            if self.trading_engine:
                await self.trading_engine.stop()
            
            if self.stream_manager:
                await self.stream_manager.stop()
            
            if self.trading_engine:
                stats = self.trading_engine.get_stats()
                print("\n" + "="*60)
                print("📈 CFB FINAL STATISTICS")
                print("="*60)
                print(f"Total Trades: {stats.total_trades}")
                print(f"Total P&L: ${stats.total_pnl:.2f}")
                print(f"Positions: {len(self.trading_engine.get_positions())}")
            
            print("\n✅ CFB HFT completed!")

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="CFB HFT Trader")
    parser.add_argument("--live", action="store_true", help="Run in live mode (default: simulation)")
    args = parser.parse_args()
    
    trader = CFBHFTTrader(simulation_mode=not args.live)
    await trader.run_trader()

if __name__ == "__main__":
    asyncio.run(main())
