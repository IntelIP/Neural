"""
Database schema for Neural SDK deployment module.

SQLAlchemy models for storing deployment, trade, and performance data.
"""

from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Integer, Numeric, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class Trade(Base):
    """Trade record model."""

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    deployment_id = Column(String(255), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    ticker = Column(String(255), nullable=False)
    side = Column(String(10), nullable=False)  # 'buy' or 'sell'
    quantity = Column(Integer, nullable=False)
    price = Column(Numeric(10, 2), nullable=False)
    pnl = Column(Numeric(10, 2))
    strategy = Column(String(255))
    metadata = Column(JSON)


class Position(Base):
    """Current position model."""

    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    deployment_id = Column(String(255), nullable=False, index=True)
    ticker = Column(String(255), nullable=False)
    quantity = Column(Integer, nullable=False)
    entry_price = Column(Numeric(10, 2), nullable=False)
    current_price = Column(Numeric(10, 2))
    unrealized_pnl = Column(Numeric(10, 2))
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)


class Performance(Base):
    """Performance metrics model."""

    __tablename__ = "performance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    deployment_id = Column(String(255), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    total_pnl = Column(Numeric(10, 2))
    daily_pnl = Column(Numeric(10, 2))
    sharpe_ratio = Column(Numeric(10, 4))
    max_drawdown = Column(Numeric(10, 4))
    win_rate = Column(Numeric(5, 4))
    num_trades = Column(Integer)


class Deployment(Base):
    """Deployment record model."""

    __tablename__ = "deployments"

    id = Column(String(255), primary_key=True)
    bot_name = Column(String(255), nullable=False)
    strategy_type = Column(String(255))
    environment = Column(String(50))
    status = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    config = Column(JSON)
    container_id = Column(String(255))
    sandbox_id = Column(String(255))


def create_tables(database_url: str) -> None:
    """Create all database tables.

    Args:
        database_url: SQLAlchemy database URL
    """
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)


def get_session(database_url: str):
    """Get a database session.

    Args:
        database_url: SQLAlchemy database URL

    Returns:
        SQLAlchemy Session
    """
    engine = create_engine(database_url)
    Session = sessionmaker(bind=engine)
    return Session()
