#!/usr/bin/env python3
"""
Neural SDK Portfolio Management Example

This demonstrates the clean, user-friendly API for portfolio management.
Users no longer need to understand the internal data pipeline architecture.
"""

import asyncio
from neural_sdk import NeuralSDK

async def main():
    """Show the new clean SDK interface."""
    
    # Initialize SDK from environment variables
    sdk = NeuralSDK.from_env()
    
    print("💼 NEURAL SDK PORTFOLIO MANAGEMENT")
    print("=" * 50)
    
    # Get comprehensive portfolio summary
    portfolio = await sdk.get_portfolio_summary()
    
    print(f"💰 Balance: ${portfolio.balance:.2f}")
    print(f"📊 Total Value: ${portfolio.total_value:.2f}")
    print(f"📈 Total Exposure: ${portfolio.total_exposure:.2f}")
    print(f"🎯 Active Positions: {portfolio.position_count}")
    print(f"💸 Total Fees Paid: ${portfolio.total_fees_paid:.2f}")
    print()
    
    # Show positions with clean market names
    print("📋 CURRENT POSITIONS:")
    print("-" * 30)
    for position in portfolio.positions:
        print(f"🎯 {position.market_name}")
        print(f"   Shares: {position.position}")
        print(f"   Exposure: ${position.market_exposure:.2f}")
        print(f"   Avg Price: ${position.avg_price:.3f}")
        print()
    
    # Get recent orders
    orders = await sdk.get_orders(limit=3)
    
    print(f"📝 RECENT ORDERS ({len(orders)}):")
    print("-" * 30)
    for order in orders:
        print(f"📋 {order.action} {order.quantity} {order.side}")
        print(f"   Price: ${order.price:.2f}")
        print(f"   Status: {order.status}")
        print(f"   Fill: {order.fill_percentage:.1f}%")
        print()
    
    # Example: Place a new order (commented out for safety)
    """
    # Place an order to buy Eagles win
    order = await sdk.place_order(
        ticker="KXNFLGAME-25SEP04DALPHI-PHI",
        side="YES", 
        quantity=5,
        price=0.70
    )
    print(f"✅ Order placed: {order.order_id}")
    """
    
    print("🎉 PORTFOLIO MANAGEMENT COMPLETE!")

if __name__ == "__main__":
    asyncio.run(main())