"""
ESPN Streaming Adapter
Converts ESPN polling to real-time streaming with optimized intervals
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from enum import Enum

from .client import ESPNClient
from .processor import PlayByPlayProcessor
from .models import GameState, GameEvent, EventType
from ..circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitOpenException
from ..health_probe import http_health_check
from ..rate_limiter import TokenBucket, AdaptiveRateLimiter

logger = logging.getLogger(__name__)


class PollInterval(Enum):
    """Polling intervals based on game state"""
    PREGAME = 60  # 1 minute before game
    ACTIVE_PLAY = 5  # 5 seconds during active play
    CRITICAL = 3  # 3 seconds in critical situations
    TIMEOUT = 15  # 15 seconds during timeouts
    QUARTER_BREAK = 30  # 30 seconds between quarters
    HALFTIME = 60  # 1 minute at halftime
    POSTGAME = 120  # 2 minutes after game


class ESPNStreamAdapter:
    """
    Adaptive streaming adapter for ESPN data
    Optimizes polling based on game state
    """
    
    def __init__(
        self,
        on_event: Optional[Callable[[GameEvent], None]] = None,
        on_error: Optional[Callable[[Dict[str, Any]], None]] = None,
        enable_circuit_breaker: bool = True
    ):
        """
        Initialize ESPN stream adapter
        
        Args:
            on_event: Callback for game events
            on_error: Callback for errors
            enable_circuit_breaker: Enable circuit breaker protection
        """
        self.client = ESPNClient()
        self.processor = PlayByPlayProcessor()
        self.on_event = on_event or self._default_event_handler
        self.on_error = on_error or self._default_error_handler
        
        # State tracking
        self.is_running = False
        self.monitored_games: Dict[str, Dict[str, Any]] = {}
        self.game_tasks: Dict[str, asyncio.Task] = {}
        self.sport_tasks: Dict[str, asyncio.Task] = {}
        
        # Event queue
        self.event_queue: asyncio.Queue = asyncio.Queue()
        
        # Circuit breaker for ESPN API protection
        self.enable_circuit_breaker = enable_circuit_breaker
        if enable_circuit_breaker:
            config = CircuitBreakerConfig(
                name="espn_api",
                failure_threshold=5,
                success_threshold=2,
                timeout=10.0,
                half_open_interval=15.0,
                window_size=60
            )
            self.circuit_breaker = CircuitBreaker(config)
        else:
            self.circuit_breaker = None
        
        # Rate limiting for API calls (100 requests per minute)
        self.api_rate_limiter = TokenBucket(
            capacity=100,
            refill_rate=100/60,  # 100 requests per 60 seconds
            burst_size=20  # Allow burst of 20 requests
        )
        
        # Adaptive rate limiter that learns optimal rates
        self.adaptive_limiter = AdaptiveRateLimiter(
            initial_limit=100,
            window=60,
            min_limit=20,
            max_limit=200,
            target_success_rate=0.99
        )
    
    async def start(self):
        """Start the streaming adapter"""
        self.is_running = True
        
        # Start event processor
        asyncio.create_task(self._process_events())
        
        logger.info("ESPN streaming adapter started")
    
    async def stop(self):
        """Stop the streaming adapter"""
        self.is_running = False
        
        # Cancel all tasks
        for task in list(self.game_tasks.values()) + list(self.sport_tasks.values()):
            task.cancel()
        
        self.game_tasks.clear()
        self.sport_tasks.clear()
        self.monitored_games.clear()
        
        await self.client.close()
        logger.info("ESPN streaming adapter stopped")
    
    async def monitor_sport(self, sport: str = "nfl"):
        """
        Monitor all live games in a sport
        
        Args:
            sport: Sport to monitor
        """
        if sport in self.sport_tasks:
            logger.warning(f"Already monitoring {sport}")
            return
        
        task = asyncio.create_task(self._monitor_sport_loop(sport))
        self.sport_tasks[sport] = task
        
        logger.info(f"Started monitoring {sport}")
    
    async def monitor_game(self, game_id: str, sport: str = "nfl"):
        """
        Monitor a specific game
        
        Args:
            game_id: ESPN game ID
            sport: Sport type
        """
        if game_id in self.monitored_games:
            logger.warning(f"Already monitoring game {game_id}")
            return
        
        self.monitored_games[game_id] = {
            'sport': sport,
            'state': None,
            'last_play_id': None,
            'last_update': datetime.now()
        }
        
        task = asyncio.create_task(self._monitor_game_loop(game_id, sport))
        self.game_tasks[game_id] = task
        
        logger.info(f"Started monitoring game {game_id}")
    
    async def stop_monitoring_game(self, game_id: str):
        """Stop monitoring a specific game"""
        if game_id not in self.monitored_games:
            return
        
        # Cancel task
        if game_id in self.game_tasks:
            self.game_tasks[game_id].cancel()
            del self.game_tasks[game_id]
        
        del self.monitored_games[game_id]
        logger.info(f"Stopped monitoring game {game_id}")
    
    def _calculate_poll_interval(self, game_state: GameState) -> int:
        """
        Calculate optimal polling interval based on game state
        
        Args:
            game_state: Current game state
            
        Returns:
            Polling interval in seconds
        """
        # Game not started
        if game_state.quarter == 0:
            return PollInterval.PREGAME.value
        
        # Game finished
        if game_state.quarter > 4 and game_state.clock == "0:00":
            return PollInterval.POSTGAME.value
        
        # Critical situation - poll more frequently
        if game_state.is_critical_situation():
            return PollInterval.CRITICAL.value
        
        # Two minute warning
        if game_state.is_two_minute_warning:
            return PollInterval.CRITICAL.value
        
        # Halftime
        if game_state.quarter == 2 and game_state.clock == "0:00":
            return PollInterval.HALFTIME.value
        
        # Quarter break
        if game_state.clock == "0:00":
            return PollInterval.QUARTER_BREAK.value
        
        # Timeout (would need to detect from play text)
        # This is a simplified check
        clock_minutes = game_state._parse_clock_minutes()
        if clock_minutes == 15:  # Likely a break
            return PollInterval.TIMEOUT.value
        
        # Default active play interval
        return PollInterval.ACTIVE_PLAY.value
    
    async def _monitor_sport_loop(self, sport: str):
        """Monitor sport scoreboard for live games"""
        check_interval = 60  # Check scoreboard every minute
        
        while self.is_running:
            try:
                # Apply rate limiting before API call
                await self.api_rate_limiter.consume(1, timeout=5.0)
                
                # Get live games with circuit breaker protection
                if self.circuit_breaker:
                    try:
                        live_games = await self.circuit_breaker.call(
                            self.client.get_live_games, sport
                        )
                        self.adaptive_limiter.record_success()
                    except CircuitOpenException:
                        logger.warning(f"Circuit breaker open for ESPN API (sport: {sport})")
                        self.adaptive_limiter.record_error()
                        await asyncio.sleep(check_interval * 2)
                        continue
                else:
                    live_games = await self.client.get_live_games(sport)
                    self.adaptive_limiter.record_success()
                
                # Start monitoring new live games
                for game in live_games:
                    game_id = game['id']
                    if game_id not in self.monitored_games:
                        await self.monitor_game(game_id, sport)
                
                # Stop monitoring finished games
                live_game_ids = {g['id'] for g in live_games}
                finished_games = [
                    game_id for game_id in self.monitored_games
                    if game_id not in live_game_ids
                ]
                
                for game_id in finished_games:
                    await self.stop_monitoring_game(game_id)
                
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Error monitoring {sport}: {e}")
                await self.on_error({
                    'type': 'sport_monitor_error',
                    'sport': sport,
                    'error': str(e)
                })
                await asyncio.sleep(check_interval * 2)
    
    async def _monitor_game_loop(self, game_id: str, sport: str):
        """Monitor a specific game with adaptive polling"""
        game_info = self.monitored_games[game_id]
        
        while self.is_running and game_id in self.monitored_games:
            try:
                # Apply rate limiting before API call
                await self.api_rate_limiter.consume(1, timeout=5.0)
                
                # Get play-by-play data with circuit breaker protection
                if self.circuit_breaker:
                    try:
                        pbp_data = await self.circuit_breaker.call(
                            self.client.get_play_by_play, game_id, sport
                        )
                        self.adaptive_limiter.record_success()
                    except CircuitOpenException:
                        logger.warning(f"Circuit breaker open for ESPN API (game: {game_id})")
                        self.adaptive_limiter.record_error()
                        await asyncio.sleep(PollInterval.TIMEOUT.value * 2)
                        continue
                else:
                    pbp_data = await self.client.get_play_by_play(game_id, sport)
                    self.adaptive_limiter.record_success()
                game_state = pbp_data['game_state']
                
                # Check for new plays
                await self._process_new_plays(game_id, pbp_data, game_info)
                
                # Check for state changes
                await self._process_state_changes(game_id, game_state, game_info)
                
                # Update game info
                game_info['state'] = game_state
                game_info['last_update'] = datetime.now()
                
                # Calculate next poll interval
                base_interval = self._calculate_poll_interval(game_state)
                
                # Adjust interval based on rate limit availability
                tokens_available = self.api_rate_limiter.tokens_available()
                if tokens_available < 10:
                    # Low on tokens, increase interval
                    poll_interval = base_interval * 2
                    logger.info(f"Low on API tokens ({tokens_available:.0f}), doubling poll interval to {poll_interval}s")
                elif tokens_available < 30:
                    # Getting low, slightly increase interval
                    poll_interval = base_interval * 1.5
                else:
                    poll_interval = base_interval
                
                # Check if game is over
                if game_state.quarter > 4 and game_state.clock == "0:00":
                    # Emit game end event
                    event = self.processor.create_game_event(
                        EventType.GAME_END,
                        game_state,
                        f"Game ended: {game_state.home_team} {game_state.home_score} - {game_state.away_team} {game_state.away_score}"
                    )
                    await self._emit_event(event)
                    
                    # Stop monitoring after a delay
                    await asyncio.sleep(PollInterval.POSTGAME.value)
                    await self.stop_monitoring_game(game_id)
                    break
                
                await asyncio.sleep(poll_interval)
                
            except Exception as e:
                logger.error(f"Error monitoring game {game_id}: {e}")
                await self.on_error({
                    'type': 'game_monitor_error',
                    'game_id': game_id,
                    'error': str(e)
                })
                await asyncio.sleep(PollInterval.TIMEOUT.value)
    
    async def _process_new_plays(
        self,
        game_id: str,
        pbp_data: Dict[str, Any],
        game_info: Dict[str, Any]
    ):
        """Process new plays and emit events"""
        all_plays = pbp_data['plays']
        
        if not all_plays:
            return
        
        # Find new plays
        last_play_id = game_info.get('last_play_id')
        new_plays = []
        
        if last_play_id is None:
            # First update, don't emit all historical plays
            if all_plays:
                game_info['last_play_id'] = all_plays[-1].id
            return
        
        # Find plays after last known play
        found_last = False
        for play in all_plays:
            if found_last:
                new_plays.append(play)
            elif play.id == last_play_id:
                found_last = True
        
        # If we didn't find the last play, assume all are new
        if not found_last and all_plays:
            new_plays = all_plays[-5:]  # Limit to last 5 plays
        
        # Process new plays
        for play in new_plays:
            # Emit events for high-impact plays
            if play.is_high_impact():
                for event_type in play.events:
                    event = self.processor.create_game_event(
                        event_type,
                        pbp_data['game_state'],
                        play.text,
                        play.impact_score,
                        {'play': play.__dict__}
                    )
                    await self._emit_event(event)
        
        # Update last play ID
        if all_plays:
            game_info['last_play_id'] = all_plays[-1].id
    
    async def _process_state_changes(
        self,
        game_id: str,
        game_state: GameState,
        game_info: Dict[str, Any]
    ):
        """Process game state changes"""
        old_state = game_info.get('state')
        
        if old_state is None:
            return
        
        # Check for score changes
        if (old_state.home_score != game_state.home_score or
            old_state.away_score != game_state.away_score):
            
            score_diff = abs((game_state.home_score - game_state.away_score) -
                           (old_state.home_score - old_state.away_score))
            
            description = f"Score update: {game_state.home_team} {game_state.home_score} - {game_state.away_team} {game_state.away_score}"
            
            # Higher impact for scores that change the lead
            impact = 0.7 if score_diff > 0 else 0.5
            
            event = GameEvent(
                type=EventType.TOUCHDOWN if score_diff >= 6 else EventType.FIELD_GOAL_MADE,
                game_id=game_id,
                description=description,
                impact_score=impact,
                game_state=game_state
            )
            await self._emit_event(event)
        
        # Check for quarter changes
        if old_state.quarter != game_state.quarter:
            event = self.processor.create_game_event(
                EventType.QUARTER_END if game_state.quarter <= 4 else EventType.GAME_END,
                game_state,
                f"Quarter {old_state.quarter} ended"
            )
            await self._emit_event(event)
        
        # Check for significant win probability swings
        prob_swing = abs(game_state.home_win_probability - old_state.home_win_probability)
        if prob_swing > 10:  # 10% swing
            description = f"Win probability swing: {game_state.home_team} {game_state.home_win_probability:.1f}%"
            event = GameEvent(
                type=EventType.TOUCHDOWN,  # Use high-impact type
                game_id=game_id,
                description=description,
                impact_score=min(prob_swing / 20, 1.0),  # Scale impact
                game_state=game_state,
                metadata={'probability_swing': prob_swing}
            )
            await self._emit_event(event)
    
    async def _emit_event(self, event: GameEvent):
        """Emit an event to subscribers"""
        await self.event_queue.put(event)
        await self.on_event(event)
    
    async def _process_events(self):
        """Process event queue"""
        while self.is_running:
            try:
                event = await asyncio.wait_for(
                    self.event_queue.get(),
                    timeout=1.0
                )
                # Event already processed by emit
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    async def get_next_event(self, timeout: float = 1.0) -> Optional[GameEvent]:
        """
        Get next event from queue
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            GameEvent or None if timeout
        """
        try:
            return await asyncio.wait_for(
                self.event_queue.get(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            return None
    
    async def health_check(self) -> bool:
        """Check ESPN API health"""
        try:
            # Try to get scoreboard for NFL as health check
            if self.circuit_breaker:
                result = await self.circuit_breaker.health_check(
                    lambda: http_health_check("https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard")
                )
            else:
                result = await http_health_check("https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard")
            
            return result.get('status') == 'healthy' if isinstance(result, dict) else result
        except Exception:
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get adapter metrics"""
        return {
            'is_running': self.is_running,
            'monitored_games': len(self.monitored_games),
            'active_game_tasks': len(self.game_tasks),
            'active_sport_tasks': len(self.sport_tasks),
            'circuit_breaker': self.circuit_breaker.get_metrics() if self.circuit_breaker else None,
            'rate_limiter': self.api_rate_limiter.get_stats(),
            'adaptive_limiter': self.adaptive_limiter.get_stats()
        }
    
    # Default handlers
    async def _default_event_handler(self, event: GameEvent):
        """Default event handler"""
        logger.info(f"ESPN Event: {event.type.value} - {event.description} (Impact: {event.impact_score:.2f})")
    
    async def _default_error_handler(self, error: Dict[str, Any]):
        """Default error handler"""
        logger.error(f"ESPN Error: {error}")