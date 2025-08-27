"""Database connection manager for PostgreSQL."""

import os
import logging
from contextlib import contextmanager
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from decimal import Decimal

from sqlalchemy import create_engine, MetaData, select, and_, or_, desc
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool
from sqlalchemy.exc import SQLAlchemyError

from .models import Base, Trade, Position, ProfitLoss, AgentStatus, MarketSnapshot

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages PostgreSQL database connections and operations."""
    
    def __init__(self, connection_url: Optional[str] = None):
        """Initialize database manager.
        
        Args:
            connection_url: PostgreSQL connection URL. If not provided, uses environment variable.
        """
        self.connection_url = connection_url or os.getenv(
            'DATABASE_URL',
            'postgresql://postgres:postgres@localhost:5432/kalshi_trading'
        )
        
        # Create engine with connection pooling
        self.engine = create_engine(
            self.connection_url,
            pool_pre_ping=True,  # Verify connections before using
            pool_size=10,
            max_overflow=20,
            echo=False  # Set to True for SQL debugging
        )
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        # Initialize database tables
        self.init_database()
    
    def init_database(self):
        """Create all database tables if they don't exist."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables initialized successfully")
        except SQLAlchemyError as e:
            logger.error(f"Failed to initialize database tables: {e}")
            raise
    
    @contextmanager
    def get_session(self) -> Session:
        """Context manager for database sessions.
        
        Yields:
            Session: SQLAlchemy session
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def record_trade(self, trade_data: Dict[str, Any]) -> Trade:
        """Record a new trade in the database.
        
        Args:
            trade_data: Trade information dictionary
            
        Returns:
            Trade: Created trade object
        """
        with self.get_session() as session:
            trade = Trade(
                trade_id=trade_data['trade_id'],
                agent_id=trade_data.get('agent_id', 'TradeExecutor'),
                market_ticker=trade_data['market_ticker'],
                market_title=trade_data.get('market_title'),
                side=trade_data['side'],
                quantity=trade_data['quantity'],
                price=Decimal(str(trade_data['price'])),
                fill_price=Decimal(str(trade_data.get('fill_price', trade_data['price']))),
                total_cost=Decimal(str(trade_data['total_cost'])),
                status=trade_data.get('status', 'pending'),
                realized_pnl=Decimal(str(trade_data.get('realized_pnl', 0))),
                timestamp=trade_data.get('timestamp', datetime.utcnow()),
                execution_time=trade_data.get('execution_time'),
                order_type=trade_data.get('order_type', 'market'),
                strategy=trade_data.get('strategy')
            )
            session.add(trade)
            session.commit()
            logger.info(f"Recorded trade: {trade.trade_id}")
            return trade
    
    def update_position(self, position_data: Dict[str, Any]) -> Position:
        """Update or create a position in the database.
        
        Args:
            position_data: Position information dictionary
            
        Returns:
            Position: Updated or created position object
        """
        with self.get_session() as session:
            position = session.query(Position).filter_by(
                position_id=position_data['position_id']
            ).first()
            
            if position:
                # Update existing position
                position.quantity = position_data['quantity']
                position.current_price = Decimal(str(position_data.get('current_price', 0)))
                position.market_value = Decimal(str(position_data.get('market_value', 0)))
                position.unrealized_pnl = Decimal(str(position_data.get('unrealized_pnl', 0)))
                position.unrealized_pnl_pct = Decimal(str(position_data.get('unrealized_pnl_pct', 0)))
                position.updated_at = datetime.utcnow()
            else:
                # Create new position
                position = Position(
                    position_id=position_data['position_id'],
                    market_ticker=position_data['market_ticker'],
                    market_title=position_data.get('market_title'),
                    side=position_data['side'],
                    quantity=position_data['quantity'],
                    entry_price=Decimal(str(position_data['entry_price'])),
                    current_price=Decimal(str(position_data.get('current_price', position_data['entry_price']))),
                    market_value=Decimal(str(position_data.get('market_value', 0))),
                    unrealized_pnl=Decimal(str(position_data.get('unrealized_pnl', 0))),
                    unrealized_pnl_pct=Decimal(str(position_data.get('unrealized_pnl_pct', 0))),
                    opened_at=position_data.get('opened_at', datetime.utcnow()),
                    stop_loss_price=Decimal(str(position_data['stop_loss_price'])) if position_data.get('stop_loss_price') else None,
                    take_profit_price=Decimal(str(position_data['take_profit_price'])) if position_data.get('take_profit_price') else None,
                    kelly_fraction=Decimal(str(position_data['kelly_fraction'])) if position_data.get('kelly_fraction') else None
                )
                session.add(position)
            
            session.commit()
            logger.info(f"Updated position: {position.position_id}")
            return position
    
    def close_position(self, position_id: str) -> bool:
        """Close a position by removing it from the database.
        
        Args:
            position_id: Unique position identifier
            
        Returns:
            bool: True if position was closed, False if not found
        """
        with self.get_session() as session:
            position = session.query(Position).filter_by(position_id=position_id).first()
            if position:
                session.delete(position)
                session.commit()
                logger.info(f"Closed position: {position_id}")
                return True
            return False
    
    def record_pnl_snapshot(self, pnl_data: Dict[str, Any]) -> ProfitLoss:
        """Record a P&L snapshot.
        
        Args:
            pnl_data: P&L information dictionary
            
        Returns:
            ProfitLoss: Created P&L record
        """
        with self.get_session() as session:
            pnl = ProfitLoss(
                timestamp=pnl_data.get('timestamp', datetime.utcnow()),
                total_pnl=Decimal(str(pnl_data['total_pnl'])),
                daily_pnl=Decimal(str(pnl_data.get('daily_pnl', 0))),
                realized_pnl=Decimal(str(pnl_data.get('realized_pnl', 0))),
                unrealized_pnl=Decimal(str(pnl_data.get('unrealized_pnl', 0))),
                win_count=pnl_data.get('win_count', 0),
                loss_count=pnl_data.get('loss_count', 0),
                win_rate=Decimal(str(pnl_data['win_rate'])) if pnl_data.get('win_rate') else None,
                sharpe_ratio=Decimal(str(pnl_data['sharpe_ratio'])) if pnl_data.get('sharpe_ratio') else None,
                max_drawdown=Decimal(str(pnl_data['max_drawdown'])) if pnl_data.get('max_drawdown') else None,
                portfolio_value=Decimal(str(pnl_data.get('portfolio_value', 0))),
                cash_balance=Decimal(str(pnl_data.get('cash_balance', 0)))
            )
            session.add(pnl)
            session.commit()
            logger.info(f"Recorded P&L snapshot at {pnl.timestamp}")
            return pnl
    
    def update_agent_status(self, agent_name: str, status: str, **kwargs) -> AgentStatus:
        """Update agent status.
        
        Args:
            agent_name: Name of the agent
            status: Current status ('running', 'stopped', 'error', 'paused')
            **kwargs: Additional status fields
            
        Returns:
            AgentStatus: Updated agent status
        """
        with self.get_session() as session:
            agent = session.query(AgentStatus).filter_by(agent_name=agent_name).first()
            
            if not agent:
                agent = AgentStatus(agent_name=agent_name, status=status)
                session.add(agent)
            
            agent.status = status
            agent.last_heartbeat = datetime.utcnow()
            
            # Update optional fields
            for key, value in kwargs.items():
                if hasattr(agent, key):
                    setattr(agent, key, value)
            
            session.commit()
            logger.info(f"Updated agent status: {agent_name} -> {status}")
            return agent
    
    def get_recent_trades(self, limit: int = 100, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recent trades.
        
        Args:
            limit: Maximum number of trades to return
            status: Filter by trade status (optional)
            
        Returns:
            List[Dict[str, Any]]: List of recent trades as dictionaries
        """
        with self.get_session() as session:
            query = session.query(Trade)
            if status:
                query = query.filter(Trade.status == status)
            trades = query.order_by(desc(Trade.timestamp)).limit(limit).all()
            return [
                {
                    'trade_id': t.trade_id,
                    'agent_id': t.agent_id,
                    'market_ticker': t.market_ticker,
                    'market_title': t.market_title,
                    'side': t.side,
                    'quantity': t.quantity,
                    'price': t.price,
                    'fill_price': t.fill_price,
                    'total_cost': t.total_cost,
                    'status': t.status,
                    'realized_pnl': t.realized_pnl,
                    'timestamp': t.timestamp,
                    'execution_time': t.execution_time,
                    'order_type': t.order_type,
                    'strategy': t.strategy
                }
                for t in trades
            ]
    
    def get_active_positions(self) -> List[Dict[str, Any]]:
        """Get all active positions.
        
        Returns:
            List[Dict[str, Any]]: List of active positions as dictionaries
        """
        with self.get_session() as session:
            positions = session.query(Position).order_by(desc(Position.updated_at)).all()
            return [
                {
                    'position_id': p.position_id,
                    'market_ticker': p.market_ticker,
                    'market_title': p.market_title,
                    'side': p.side,
                    'quantity': p.quantity,
                    'entry_price': p.entry_price,
                    'current_price': p.current_price,
                    'market_value': p.market_value,
                    'unrealized_pnl': p.unrealized_pnl,
                    'unrealized_pnl_pct': p.unrealized_pnl_pct,
                    'opened_at': p.opened_at,
                    'updated_at': p.updated_at,
                    'stop_loss_price': p.stop_loss_price,
                    'take_profit_price': p.take_profit_price,
                    'kelly_fraction': p.kelly_fraction
                }
                for p in positions
            ]
    
    def get_pnl_history(self, start_date: Optional[datetime] = None, 
                       end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get P&L history within date range.
        
        Args:
            start_date: Start of date range (optional)
            end_date: End of date range (optional)
            
        Returns:
            List[Dict[str, Any]]: List of P&L records as dictionaries
        """
        with self.get_session() as session:
            query = session.query(ProfitLoss)
            
            if start_date:
                query = query.filter(ProfitLoss.timestamp >= start_date)
            if end_date:
                query = query.filter(ProfitLoss.timestamp <= end_date)
            
            pnl_records = query.order_by(ProfitLoss.timestamp).all()
            return [
                {
                    'timestamp': pl.timestamp,
                    'total_pnl': pl.total_pnl,
                    'daily_pnl': pl.daily_pnl,
                    'realized_pnl': pl.realized_pnl,
                    'unrealized_pnl': pl.unrealized_pnl,
                    'win_count': pl.win_count,
                    'loss_count': pl.loss_count,
                    'win_rate': pl.win_rate,
                    'sharpe_ratio': pl.sharpe_ratio,
                    'max_drawdown': pl.max_drawdown,
                    'portfolio_value': pl.portfolio_value,
                    'cash_balance': pl.cash_balance
                }
                for pl in pnl_records
            ]
    
    def get_agent_statuses(self) -> List[Dict[str, Any]]:
        """Get status of all agents.
        
        Returns:
            List[Dict[str, Any]]: List of agent statuses as dictionaries
        """
        with self.get_session() as session:
            agents = session.query(AgentStatus).all()
            return [
                {
                    'agent_name': a.agent_name,
                    'status': a.status,
                    'last_heartbeat': a.last_heartbeat,
                    'messages_processed': a.messages_processed,
                    'errors_count': a.errors_count,
                    'start_time': a.start_time,
                    'stop_time': a.stop_time,
                    'emergency_stop': a.emergency_stop,
                    'version': a.version,
                    'config': a.config
                }
                for a in agents
            ]
    
    def emergency_stop_all_agents(self) -> bool:
        """Set emergency stop flag for all agents.
        
        Returns:
            bool: True if successful
        """
        with self.get_session() as session:
            session.query(AgentStatus).update({
                'status': 'stopped',
                'emergency_stop': True,
                'stop_time': datetime.utcnow()
            })
            session.commit()
            logger.warning("Emergency stop activated for all agents")
            return True
    
    def record_market_snapshot(self, market_data: Dict[str, Any]) -> MarketSnapshot:
        """Record a market snapshot for analysis.
        
        Args:
            market_data: Market information dictionary
            
        Returns:
            MarketSnapshot: Created market snapshot
        """
        with self.get_session() as session:
            snapshot = MarketSnapshot(
                timestamp=market_data.get('timestamp', datetime.utcnow()),
                market_ticker=market_data['market_ticker'],
                yes_price=Decimal(str(market_data.get('yes_price', 0))),
                no_price=Decimal(str(market_data.get('no_price', 0))),
                volume=market_data.get('volume'),
                open_interest=market_data.get('open_interest'),
                sentiment_score=Decimal(str(market_data['sentiment_score'])) if market_data.get('sentiment_score') else None,
                kelly_signal=Decimal(str(market_data['kelly_signal'])) if market_data.get('kelly_signal') else None
            )
            session.add(snapshot)
            session.commit()
            return snapshot