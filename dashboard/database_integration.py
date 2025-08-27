"""Database integration patch for base_consumer.py

This file contains the modifications needed for the base_consumer.py 
to integrate with the PostgreSQL database for the dashboard.

Add this to base_consumer.py after the imports section:
"""

# Add to imports section:
import os
import sys
from pathlib import Path

# Add database path to system path if dashboard exists
dashboard_path = Path(__file__).parent.parent / 'dashboard'
if dashboard_path.exists():
    sys.path.insert(0, str(dashboard_path))
    try:
        from data import DatabaseManager
        HAS_DATABASE = True
    except ImportError:
        HAS_DATABASE = False
else:
    HAS_DATABASE = False


# Add this method to BaseAgentRedisConsumer class:
async def init_database(self) -> None:
    """Initialize database connection for dashboard integration."""
    if HAS_DATABASE and os.getenv('ENABLE_DASHBOARD_DB', 'false').lower() == 'true':
        try:
            self.db_manager = DatabaseManager()
            logger.info(f"{self.agent_name} connected to dashboard database")
        except Exception as e:
            logger.warning(f"{self.agent_name} could not connect to database: {e}")
            self.db_manager = None
    else:
        self.db_manager = None


# Modify the connect method to include database initialization:
# Add after establishing Redis connections:
# await self.init_database()


# Add these methods to BaseAgentRedisConsumer class:
async def record_trade_to_db(self, trade: Dict[str, Any]) -> None:
    """Record trade to database if connected."""
    if self.db_manager:
        try:
            trade_data = {
                'trade_id': trade.get('trade_id', f"{self.agent_name}_{datetime.utcnow().timestamp()}"),
                'agent_id': self.agent_name,
                'market_ticker': trade['market'],
                'side': trade.get('side', 'buy'),
                'quantity': trade.get('quantity', 0),
                'price': trade.get('price', 0),
                'total_cost': trade.get('quantity', 0) * trade.get('price', 0),
                'status': trade.get('status', 'pending'),
                'realized_pnl': trade.get('realized_pnl', 0),
                'strategy': trade.get('strategy', self.agent_name)
            }
            self.db_manager.record_trade(trade_data)
        except Exception as e:
            logger.error(f"Failed to record trade to database: {e}")


async def update_position_in_db(self, position: Dict[str, Any]) -> None:
    """Update position in database if connected."""
    if self.db_manager:
        try:
            position_data = {
                'position_id': position.get('position_id', f"{position['market']}_{position.get('side', 'yes')}"),
                'market_ticker': position['market'],
                'side': position.get('side', 'yes'),
                'quantity': position.get('quantity', 0),
                'entry_price': position.get('entry_price', 0),
                'current_price': position.get('current_price', position.get('entry_price', 0)),
                'kelly_fraction': position.get('kelly_fraction')
            }
            self.db_manager.update_position(position_data)
        except Exception as e:
            logger.error(f"Failed to update position in database: {e}")


async def update_agent_status_in_db(self, status: str = 'running', **kwargs) -> None:
    """Update agent status in database if connected."""
    if self.db_manager:
        try:
            self.db_manager.update_agent_status(
                agent_name=self.agent_name,
                status=status,
                messages_processed=self.messages_processed,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Failed to update agent status in database: {e}")


async def check_emergency_stop(self) -> bool:
    """Check if emergency stop has been triggered."""
    if self.db_manager:
        try:
            agents = self.db_manager.get_agent_statuses()
            for agent in agents:
                if agent.agent_name == self.agent_name and agent.emergency_stop:
                    logger.warning(f"{self.agent_name} received emergency stop signal")
                    self.is_running = False
                    return True
        except Exception as e:
            logger.error(f"Failed to check emergency stop status: {e}")
    return False


# Modify the publish_trade method to include database recording:
# Add after successful publish:
# await self.record_trade_to_db(trade)


# Modify the start_consuming method to include periodic status updates:
# Add inside the main loop:
"""
# Update agent status periodically (every 100 messages or 60 seconds)
if self.messages_received % 100 == 0:
    await self.update_agent_status_in_db()

# Check for emergency stop
if await self.check_emergency_stop():
    logger.warning(f"{self.agent_name} stopping due to emergency stop")
    break
"""


# Add to the disconnect method:
# await self.update_agent_status_in_db(status='stopped')