"""
ESPN Adapter for Neural SDK
Integrates ESPN GameCast and real-time play-by-play data
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime

from ...data_pipeline.data_sources.espn.stream import ESPNStreamAdapter
from ...data_pipeline.data_sources.espn.client import ESPNClient
from ...data_pipeline.data_sources.espn.models import GameEvent, EventType

logger = logging.getLogger(__name__)


class ESPNAdapter:
    """
    ESPN Adapter for Neural Trading Platform

    Integrates ESPN's real-time play-by-play data with the Neural SDK.
    Provides streaming game events for trading signals.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ESPN adapter.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.name = "ESPN"
        self.stream_adapter = None
        self.client = ESPNClient()
        self.is_running = False

        # Event callbacks
        self.on_event: Optional[Callable[[GameEvent], None]] = None
        self.on_error: Optional[Callable[[Dict[str, Any]], None]] = None

        # Configuration
        self.sports = self.config.get("sports", ["nfl"])
        self.update_interval = self.config.get("update_interval", 5)
        self.critical_interval = self.config.get("critical_interval", 3)
        self.game_ids = self.config.get("games", [])

        logger.info(f"ESPN Adapter initialized for sports: {self.sports}")

    async def start(self):
        """Start the ESPN adapter."""
        if self.is_running:
            logger.warning("ESPN adapter already running")
            return

        self.is_running = True

        # Create stream adapter
        self.stream_adapter = ESPNStreamAdapter(
            on_event=self._handle_game_event_sync,
            on_error=self._handle_error_sync
        )

        # Start streaming
        await self.stream_adapter.start()

        # Start monitoring configured sports
        for sport in self.sports:
            await self.stream_adapter.monitor_sport(sport)
            logger.info(f"Started monitoring {sport}")

        # Monitor specific games if configured
        for game_id in self.game_ids:
            sport = self._infer_sport_from_game_id(game_id)
            await self.stream_adapter.monitor_game(game_id, sport)
            logger.info(f"Started monitoring game {game_id}")

        logger.info("ESPN adapter started successfully")

    async def stop(self):
        """Stop the ESPN adapter."""
        if not self.is_running:
            return

        self.is_running = False

        if self.stream_adapter:
            await self.stream_adapter.stop()

        await self.client.close()
        logger.info("ESPN adapter stopped")

    def set_event_callback(self, callback: Callable[[GameEvent], None]):
        """Set event callback function."""
        self.on_event = callback

    def set_error_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Set error callback function."""
        self.on_error = callback

    async def get_live_games(self, sport: str = "nfl") -> List[Dict[str, Any]]:
        """
        Get currently live games for a sport.

        Args:
            sport: Sport identifier

        Returns:
            List of live games
        """
        try:
            return await self.client.get_live_games(sport)
        except Exception as e:
            logger.error(f"Failed to get live games for {sport}: {e}")
            return []

    async def get_game_summary(self, game_id: str, sport: str = "nfl") -> Dict[str, Any]:
        """
        Get detailed game summary.

        Args:
            game_id: ESPN game ID
            sport: Sport identifier

        Returns:
            Game summary data
        """
        try:
            return await self.client.get_game_summary(game_id, sport)
        except Exception as e:
            logger.error(f"Failed to get game summary for {game_id}: {e}")
            return {}

    async def get_play_by_play(self, game_id: str, sport: str = "nfl") -> Dict[str, Any]:
        """
        Get play-by-play data for a game.

        Args:
            game_id: ESPN game ID
            sport: Sport identifier

        Returns:
            Play-by-play data
        """
        try:
            return await self.client.get_play_by_play(game_id, sport)
        except Exception as e:
            logger.error(f"Failed to get play-by-play for {game_id}: {e}")
            return {}

    async def get_win_probability(self, game_id: str, sport: str = "nfl") -> List[Dict[str, float]]:
        """
        Get win probability history for a game.

        Args:
            game_id: ESPN game ID
            sport: Sport identifier

        Returns:
            Win probability data points
        """
        try:
            return await self.client.get_win_probability_history(game_id, sport)
        except Exception as e:
            logger.error(f"Failed to get win probability for {game_id}: {e}")
            return []

    async def get_scoring_plays(self, game_id: str, sport: str = "nfl") -> List[Dict[str, Any]]:
        """
        Get scoring plays for a game.

        Args:
            game_id: ESPN game ID
            sport: Sport identifier

        Returns:
            List of scoring plays
        """
        try:
            return await self.client.get_scoring_plays(game_id, sport)
        except Exception as e:
            logger.error(f"Failed to get scoring plays for {game_id}: {e}")
            return []

    async def get_injuries(self, game_id: str, sport: str = "nfl") -> List[Dict[str, Any]]:
        """
        Get injury information for a game.

        Args:
            game_id: ESPN game ID
            sport: Sport identifier

        Returns:
            List of injuries
        """
        try:
            return await self.client.get_injuries(game_id, sport)
        except Exception as e:
            logger.error(f"Failed to get injuries for {game_id}: {e}")
            return []

    async def health_check(self) -> bool:
        """
        Check ESPN service health.

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Try to get NFL scoreboard
            result = await self.client.get_scoreboard("nfl")
            return bool(result and "events" in result)
        except Exception as e:
            logger.error(f"ESPN health check failed: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get adapter statistics.

        Returns:
            Statistics dictionary
        """
        if self.stream_adapter:
            return self.stream_adapter.get_metrics()
        return {
            "is_running": self.is_running,
            "sports": self.sports,
            "game_ids": self.game_ids
        }

    def _infer_sport_from_game_id(self, game_id: str) -> str:
        """Infer sport from game ID (basic implementation)."""
        # This could be enhanced with actual game ID patterns
        return "nfl"  # Default to NFL

    def _handle_game_event_sync(self, event: GameEvent):
        """Handle game events from the stream adapter (synchronous)."""
        logger.info(f"ESPN Event: {event.type.value} - {event.description} (Impact: {event.impact_score:.2f})")

        if self.on_event:
            try:
                # Schedule async callback if needed
                if asyncio.iscoroutinefunction(self.on_event):
                    asyncio.create_task(self._handle_game_event_async(event))
                else:
                    self.on_event(event)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")

    def _handle_error_sync(self, error: Dict[str, Any]):
        """Handle errors from the stream adapter (synchronous)."""
        logger.error(f"ESPN Error: {error}")

        if self.on_error:
            try:
                # Schedule async callback if needed
                if asyncio.iscoroutinefunction(self.on_error):
                    asyncio.create_task(self._handle_error_async(error))
                else:
                    self.on_error(error)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")

    async def _handle_game_event_async(self, event: GameEvent):
        """Handle async game event callback."""
        if self.on_event and asyncio.iscoroutinefunction(self.on_event):
            await self.on_event(event)

    async def _handle_error_async(self, error: Dict[str, Any]):
        """Handle async error callback."""
        if self.on_error and asyncio.iscoroutinefunction(self.on_error):
            await self.on_error(error)

    # Context manager support
    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()