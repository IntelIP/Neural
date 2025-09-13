"""
Neural SDK Trading Infrastructure

This module provides the complete trading infrastructure stack for executing
sentiment-based trading strategies on Kalshi prediction markets.

Components:
- KalshiClient: REST API client for market data and trading
- WebSocketManager: Real-time data streaming and order updates  
- OrderManager: Order lifecycle management and execution
- PositionTracker: Real-time position and P&L tracking
- TradingEngine: Strategy execution and signal-to-trade conversion
- RiskManager: Pre-trade and real-time risk controls
- PortfolioManager: Portfolio-level management and optimization

Example:
    >>> from neural.trading import TradingEngine, KalshiClient
    >>> 
    >>> # Initialize trading components
    >>> client = KalshiClient(api_key="your_key", private_key_path="key.pem")
    >>> engine = TradingEngine(client)
    >>> 
    >>> # Execute trading strategy
    >>> await engine.start_trading()
"""

from .kalshi_client import KalshiClient, KalshiConfig, Environment
from .websocket_manager import WebSocketManager, SubscriptionType
from .order_manager import OrderManager, Order, OrderStatus, OrderType, OrderSide, OrderAction
from .position_tracker import PositionTracker, Position
from .trading_engine import TradingEngine, TradingConfig, TradingMode, ExecutionMode
from .risk_manager import TradingRiskManager, RiskRule
from .portfolio_manager import PortfolioManager

__all__ = [
    "KalshiClient",
    "KalshiConfig",
    "Environment", 
    "WebSocketManager",
    "SubscriptionType",
    "OrderManager",
    "Order",
    "OrderStatus", 
    "OrderType",
    "OrderSide",
    "OrderAction",
    "PositionTracker",
    "Position",
    "TradingEngine",
    "TradingConfig",
    "TradingMode",
    "ExecutionMode",
    "TradingRiskManager",
    "RiskRule",
    "PortfolioManager"
]
