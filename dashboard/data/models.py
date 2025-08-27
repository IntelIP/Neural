"""Database models for the Kalshi trading dashboard."""

from datetime import datetime
from decimal import Decimal
from typing import Optional
from sqlalchemy import Column, String, Integer, DateTime, Numeric, Boolean, Text, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class Trade(Base):
    """Model for storing trade history."""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(50), unique=True, nullable=False)
    agent_id = Column(String(50), nullable=False)
    market_ticker = Column(String(100), nullable=False)
    market_title = Column(Text)
    side = Column(String(10), nullable=False)  # 'buy' or 'sell'
    quantity = Column(Integer, nullable=False)
    price = Column(Numeric(10, 4), nullable=False)
    fill_price = Column(Numeric(10, 4))
    total_cost = Column(Numeric(12, 4), nullable=False)
    status = Column(String(20), nullable=False)  # 'pending', 'filled', 'cancelled', 'failed'
    realized_pnl = Column(Numeric(12, 4))
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    execution_time = Column(DateTime)
    order_type = Column(String(20))  # 'market', 'limit'
    strategy = Column(String(50))  # Strategy that triggered the trade
    
    __table_args__ = (
        Index('idx_trade_timestamp', 'timestamp'),
        Index('idx_trade_market', 'market_ticker'),
        Index('idx_trade_status', 'status'),
    )


class Position(Base):
    """Model for tracking current active positions."""
    __tablename__ = 'positions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    position_id = Column(String(50), unique=True, nullable=False)
    market_ticker = Column(String(100), nullable=False)
    market_title = Column(Text)
    side = Column(String(10), nullable=False)  # 'yes' or 'no'
    quantity = Column(Integer, nullable=False)
    entry_price = Column(Numeric(10, 4), nullable=False)
    current_price = Column(Numeric(10, 4))
    market_value = Column(Numeric(12, 4))
    unrealized_pnl = Column(Numeric(12, 4))
    unrealized_pnl_pct = Column(Numeric(8, 4))
    opened_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=func.now())
    stop_loss_price = Column(Numeric(10, 4))
    take_profit_price = Column(Numeric(10, 4))
    kelly_fraction = Column(Numeric(6, 4))  # Kelly criterion position size
    
    __table_args__ = (
        Index('idx_position_market', 'market_ticker'),
        Index('idx_position_updated', 'updated_at'),
    )


class ProfitLoss(Base):
    """Model for tracking P&L over time."""
    __tablename__ = 'profit_loss'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    total_pnl = Column(Numeric(12, 4), nullable=False)
    daily_pnl = Column(Numeric(12, 4))
    realized_pnl = Column(Numeric(12, 4))
    unrealized_pnl = Column(Numeric(12, 4))
    win_count = Column(Integer, default=0)
    loss_count = Column(Integer, default=0)
    win_rate = Column(Numeric(6, 4))
    sharpe_ratio = Column(Numeric(8, 4))
    max_drawdown = Column(Numeric(12, 4))
    portfolio_value = Column(Numeric(12, 4))
    cash_balance = Column(Numeric(12, 4))
    
    __table_args__ = (
        Index('idx_pnl_timestamp', 'timestamp'),
    )


class AgentStatus(Base):
    """Model for tracking agent operational status."""
    __tablename__ = 'agent_status'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_name = Column(String(50), nullable=False, unique=True)
    status = Column(String(20), nullable=False)  # 'running', 'stopped', 'error', 'paused'
    last_heartbeat = Column(DateTime, nullable=False, default=datetime.utcnow)
    messages_processed = Column(Integer, default=0)
    errors_count = Column(Integer, default=0)
    start_time = Column(DateTime)
    stop_time = Column(DateTime)
    emergency_stop = Column(Boolean, default=False)
    version = Column(String(20))
    config = Column(Text)  # JSON configuration
    
    __table_args__ = (
        Index('idx_agent_status', 'status'),
        Index('idx_agent_heartbeat', 'last_heartbeat'),
    )


class MarketSnapshot(Base):
    """Model for storing market snapshots for analysis."""
    __tablename__ = 'market_snapshots'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    market_ticker = Column(String(100), nullable=False)
    yes_price = Column(Numeric(10, 4))
    no_price = Column(Numeric(10, 4))
    volume = Column(Integer)
    open_interest = Column(Integer)
    sentiment_score = Column(Numeric(6, 4))  # Twitter sentiment
    kelly_signal = Column(Numeric(6, 4))  # Kelly criterion signal
    
    __table_args__ = (
        Index('idx_snapshot_timestamp_market', 'timestamp', 'market_ticker'),
    )