"""
Trigger Service - Coordinates between always-on and on-demand agents
Evaluates conditions and activates appropriate agents based on events
"""

import asyncio
import json
import logging
from typing import Dict, Any, Callable, Optional, List
from datetime import datetime, time
from enum import Enum
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class Priority(Enum):
    """Priority levels for agent activation"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


class TriggerCondition:
    """Represents a trigger condition for agent activation"""
    
    def __init__(
        self,
        name: str,
        condition: Callable[[Dict[str, Any]], bool],
        agent: str,
        priority: Priority,
        cooldown_seconds: int = 60
    ):
        self.name = name
        self.condition = condition
        self.agent = agent
        self.priority = priority
        self.cooldown_seconds = cooldown_seconds
        self.last_triggered: Optional[datetime] = None
    
    def can_trigger(self) -> bool:
        """Check if trigger is off cooldown"""
        if self.last_triggered is None:
            return True
        
        elapsed = (datetime.now() - self.last_triggered).total_seconds()
        return elapsed >= self.cooldown_seconds
    
    def evaluate(self, data: Dict[str, Any]) -> bool:
        """Evaluate if condition is met"""
        if not self.can_trigger():
            return False
        
        try:
            return self.condition(data)
        except Exception as e:
            logger.error(f"Error evaluating trigger {self.name}: {e}")
            return False
    
    def mark_triggered(self):
        """Mark this trigger as having fired"""
        self.last_triggered = datetime.now()


class TriggerService:
    """
    Coordinates between always-on and on-demand agents
    Evaluates trigger conditions and activates appropriate agents
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client = None
        self.pubsub = None
        self.is_running = False
        
        # Active agent instances (for on-demand agents)
        self.active_agents: Dict[str, Any] = {}
        
        # Define trigger conditions
        self.triggers = self._initialize_triggers()
        
        # Track activation statistics
        self.activation_stats: Dict[str, int] = {}
    
    def _initialize_triggers(self) -> List[TriggerCondition]:
        """Initialize all trigger conditions"""
        return [
            # Market-based triggers
            TriggerCondition(
                name="price_spike",
                condition=lambda d: abs(d.get('price_change', 0)) > 0.05,
                agent="ArbitrageHunter",
                priority=Priority.HIGH,
                cooldown_seconds=30
            ),
            
            TriggerCondition(
                name="volume_surge",
                condition=lambda d: d.get('volume_ratio', 0) > 3.0,
                agent="GameAnalyst",
                priority=Priority.HIGH,
                cooldown_seconds=60
            ),
            
            TriggerCondition(
                name="arbitrage_opportunity",
                condition=lambda d: (
                    d.get('yes_price', 0) + d.get('no_price', 0) < 0.98
                    and d.get('yes_price', 0) > 0
                    and d.get('no_price', 0) > 0
                ),
                agent="ArbitrageHunter",
                priority=Priority.CRITICAL,
                cooldown_seconds=10
            ),
            
            # Game-based triggers
            TriggerCondition(
                name="game_starting_soon",
                condition=lambda d: 0 < d.get('time_to_game', float('inf')) < 3600,
                agent="GameAnalyst",
                priority=Priority.MEDIUM,
                cooldown_seconds=300
            ),
            
            TriggerCondition(
                name="major_game_event",
                condition=lambda d: d.get('event_type') in ['touchdown', 'field_goal', 'interception', 'fumble'],
                agent="GameAnalyst",
                priority=Priority.HIGH,
                cooldown_seconds=30
            ),
            
            TriggerCondition(
                name="injury_reported",
                condition=lambda d: d.get('event_type') == 'injury' and d.get('player_importance', 0) > 0.7,
                agent="GameAnalyst",
                priority=Priority.CRITICAL,
                cooldown_seconds=60
            ),
            
            # Sentiment-based triggers
            TriggerCondition(
                name="sentiment_shift",
                condition=lambda d: abs(d.get('sentiment_change', 0)) > 0.3,
                agent="MarketEngineer",
                priority=Priority.HIGH,
                cooldown_seconds=120
            ),
            
            TriggerCondition(
                name="viral_tweet",
                condition=lambda d: d.get('tweet_engagement', 0) > 10000,
                agent="MarketEngineer",
                priority=Priority.MEDIUM,
                cooldown_seconds=180
            ),
            
            # Portfolio/Risk triggers
            TriggerCondition(
                name="position_stop_loss",
                condition=lambda d: d.get('position_pnl', 0) < -0.10,
                agent="RiskManager",
                priority=Priority.CRITICAL,
                cooldown_seconds=0  # No cooldown for stop-loss
            ),
            
            TriggerCondition(
                name="portfolio_drawdown",
                condition=lambda d: d.get('portfolio_drawdown', 0) > 0.15,
                agent="RiskManager",
                priority=Priority.CRITICAL,
                cooldown_seconds=60
            ),
            
            # Scheduled triggers
            TriggerCondition(
                name="daily_optimization",
                condition=lambda d: (
                    d.get('event_type') == 'scheduled' 
                    and d.get('schedule_name') == 'daily_review'
                ),
                agent="StrategyOptimizer",
                priority=Priority.LOW,
                cooldown_seconds=86400  # Once per day
            ),
            
            TriggerCondition(
                name="performance_review",
                condition=lambda d: d.get('trades_completed', 0) >= 20,
                agent="StrategyOptimizer",
                priority=Priority.MEDIUM,
                cooldown_seconds=3600
            ),
            
            # Cross-market opportunities
            TriggerCondition(
                name="correlated_divergence",
                condition=lambda d: (
                    d.get('correlation_break', False) 
                    and d.get('divergence_magnitude', 0) > 0.1
                ),
                agent="ArbitrageHunter",
                priority=Priority.HIGH,
                cooldown_seconds=120
            ),
            
            # User-initiated triggers
            TriggerCondition(
                name="user_analysis_request",
                condition=lambda d: d.get('event_type') == 'user_request',
                agent="GameAnalyst",  # Default to GameAnalyst, can be overridden
                priority=Priority.HIGH,
                cooldown_seconds=0  # No cooldown for user requests
            )
        ]
    
    async def connect(self):
        """Connect to Redis"""
        self.redis_client = redis.from_url(self.redis_url)
        self.pubsub = self.redis_client.pubsub()
        logger.info("Trigger Service connected to Redis")
    
    async def start(self):
        """Start the trigger service"""
        if self.is_running:
            logger.warning("Trigger Service already running")
            return
        
        self.is_running = True
        
        # Subscribe to event channels
        await self.pubsub.subscribe(
            "events:market",
            "events:game",
            "events:sentiment",
            "events:portfolio",
            "events:user"
        )
        
        logger.info("Trigger Service started")
        
        # Start evaluation loop
        asyncio.create_task(self._evaluation_loop())
        
        # Start scheduled triggers
        asyncio.create_task(self._scheduled_triggers())
    
    async def _evaluation_loop(self):
        """Main loop for evaluating trigger conditions"""
        async for message in self.pubsub.listen():
            if not self.is_running:
                break
            
            if message['type'] == 'message':
                try:
                    channel = message['channel'].decode('utf-8')
                    data = json.loads(message['data'])
                    
                    # Evaluate triggers
                    await self.evaluate_triggers(data, channel)
                    
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
    
    async def evaluate_triggers(self, event_data: Dict[str, Any], source_channel: str):
        """
        Evaluate all triggers against incoming event
        
        Args:
            event_data: Event data to evaluate
            source_channel: Source channel of the event
        """
        triggered = []
        
        for trigger in self.triggers:
            if trigger.evaluate(event_data):
                triggered.append(trigger)
                trigger.mark_triggered()
                logger.info(f"Trigger '{trigger.name}' activated for agent '{trigger.agent}'")
        
        # Sort by priority
        triggered.sort(key=lambda t: t.priority.value)
        
        # Activate agents
        for trigger in triggered:
            await self.activate_agent(
                agent_name=trigger.agent,
                trigger_name=trigger.name,
                event_data=event_data,
                priority=trigger.priority
            )
    
    async def activate_agent(
        self,
        agent_name: str,
        trigger_name: str,
        event_data: Dict[str, Any],
        priority: Priority
    ):
        """
        Activate an on-demand agent
        
        Args:
            agent_name: Name of agent to activate
            trigger_name: Name of trigger that fired
            event_data: Event data that triggered activation
            priority: Priority level
        """
        activation_data = {
            "agent": agent_name,
            "trigger": trigger_name,
            "priority": priority.name,
            "data": event_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Publish activation request
        channel = f"agent:activate:{agent_name.lower()}"
        await self.redis_client.publish(channel, json.dumps(activation_data))
        
        # Update statistics
        self.activation_stats[agent_name] = self.activation_stats.get(agent_name, 0) + 1
        
        logger.info(
            f"Activated {agent_name} via trigger '{trigger_name}' "
            f"with priority {priority.name}"
        )
        
        # For critical priority, also send alert
        if priority == Priority.CRITICAL:
            await self._send_critical_alert(agent_name, trigger_name, event_data)
    
    async def _send_critical_alert(
        self,
        agent_name: str,
        trigger_name: str,
        event_data: Dict[str, Any]
    ):
        """Send alert for critical triggers"""
        alert = {
            "type": "CRITICAL_TRIGGER",
            "agent": agent_name,
            "trigger": trigger_name,
            "summary": f"Critical trigger '{trigger_name}' activated {agent_name}",
            "data": event_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.redis_client.publish("alerts:critical", json.dumps(alert))
        logger.critical(f"CRITICAL: {alert['summary']}")
    
    async def _scheduled_triggers(self):
        """Handle scheduled triggers (daily reviews, etc.)"""
        while self.is_running:
            now = datetime.now()
            
            # Daily review at 11 PM
            if now.hour == 23 and now.minute == 0:
                await self.evaluate_triggers(
                    {
                        "event_type": "scheduled",
                        "schedule_name": "daily_review",
                        "timestamp": now.isoformat()
                    },
                    "scheduled"
                )
                
                # Wait to avoid duplicate triggers
                await asyncio.sleep(60)
            
            # Check every minute
            await asyncio.sleep(60)
    
    async def manual_trigger(
        self,
        agent_name: str,
        data: Dict[str, Any],
        reason: str = "manual"
    ):
        """
        Manually trigger an agent activation
        
        Args:
            agent_name: Agent to activate
            data: Data to pass to agent
            reason: Reason for manual trigger
        """
        await self.activate_agent(
            agent_name=agent_name,
            trigger_name=f"manual:{reason}",
            event_data=data,
            priority=Priority.HIGH
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get trigger service statistics"""
        return {
            "is_running": self.is_running,
            "triggers_configured": len(self.triggers),
            "activation_stats": self.activation_stats,
            "trigger_status": [
                {
                    "name": t.name,
                    "agent": t.agent,
                    "priority": t.priority.name,
                    "last_triggered": t.last_triggered.isoformat() if t.last_triggered else None,
                    "on_cooldown": not t.can_trigger()
                }
                for t in self.triggers
            ]
        }
    
    async def stop(self):
        """Stop the trigger service"""
        self.is_running = False
        
        if self.pubsub:
            await self.pubsub.unsubscribe()
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info(f"Trigger Service stopped. Activations: {self.activation_stats}")


# Example usage
async def main():
    """Example of running the trigger service"""
    service = TriggerService()
    await service.connect()
    await service.start()
    
    # Simulate some events
    await asyncio.sleep(2)
    
    # Manual trigger example
    await service.manual_trigger(
        "GameAnalyst",
        {"game_id": "12345", "teams": ["Chiefs", "Bills"]},
        reason="user_request"
    )
    
    # Get statistics
    stats = service.get_statistics()
    print(f"Trigger Service Stats: {json.dumps(stats, indent=2)}")
    
    # Run for a while
    await asyncio.sleep(60)
    
    await service.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())