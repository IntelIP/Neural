"""
Historical Data Collector for Backtesting
Collects and stores historical data from Kalshi, ESPN, and Twitter
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import aiohttp
import os

from src.data_pipeline.data_sources.kalshi.client import KalshiClient as KalshiAPIClient
from src.data_pipeline.data_sources.espn.client import ESPNClient as ESPNDataAPI
from src.data_pipeline.data_sources.twitter.search import TwitterSearchClient as TwitterSearchAPI

logger = logging.getLogger(__name__)


class HistoricalDataCollector:
    """
    Collects historical data for backtesting
    
    Data sources:
    - Kalshi: Market prices, volumes, trades
    - ESPN: Game scores, events, statistics
    - Twitter: Sentiment data, tweet volumes
    """
    
    def __init__(self, output_dir: str = "backtesting/historical_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize API clients
        self.kalshi_client = KalshiAPIClient()
        self.espn_client = ESPNDataAPI()
        self.twitter_client = TwitterSearchAPI()
        
        # Data storage
        self.market_data: Dict[str, List[Dict]] = {}
        self.game_data: Dict[str, List[Dict]] = {}
        self.sentiment_data: Dict[str, List[Dict]] = {}
    
    async def collect_historical_data(
        self,
        start_date: datetime,
        end_date: datetime,
        markets: List[str],
        interval_minutes: int = 5
    ):
        """
        Collect historical data for specified period and markets
        
        Args:
            start_date: Start of collection period
            end_date: End of collection period
            markets: List of market tickers to collect
            interval_minutes: Data collection interval
        """
        logger.info(f"Collecting data from {start_date} to {end_date}")
        
        # Collect data for each market
        for market in markets:
            logger.info(f"Processing market: {market}")
            
            # Collect Kalshi market data
            await self._collect_kalshi_data(market, start_date, end_date, interval_minutes)
            
            # Extract game info from market ticker
            game_info = self._extract_game_info(market)
            if game_info:
                # Collect ESPN game data
                await self._collect_espn_data(
                    game_info['game_id'],
                    game_info['teams'],
                    start_date,
                    end_date
                )
                
                # Collect Twitter sentiment
                await self._collect_twitter_data(
                    game_info['teams'],
                    start_date,
                    end_date,
                    interval_minutes
                )
            
            # Save collected data
            await self._save_market_data(market)
    
    async def _collect_kalshi_data(
        self,
        market_ticker: str,
        start_date: datetime,
        end_date: datetime,
        interval_minutes: int
    ):
        """Collect historical Kalshi market data"""
        data_points = []
        current_time = start_date
        
        while current_time <= end_date:
            try:
                # Get market snapshot
                market_data = await self._get_kalshi_snapshot(market_ticker, current_time)
                
                if market_data:
                    data_points.append({
                        'timestamp': current_time,
                        'market_ticker': market_ticker,
                        'yes_price': market_data.get('yes_price', 0),
                        'no_price': market_data.get('no_price', 0),
                        'yes_bid': market_data.get('yes_bid', 0),
                        'yes_ask': market_data.get('yes_ask', 0),
                        'no_bid': market_data.get('no_bid', 0),
                        'no_ask': market_data.get('no_ask', 0),
                        'volume': market_data.get('volume', 0),
                        'open_interest': market_data.get('open_interest', 0),
                        'last_trade_time': market_data.get('last_trade_time'),
                        'price_change': self._calculate_price_change(data_points, market_data),
                        'volume_ratio': self._calculate_volume_ratio(data_points, market_data)
                    })
                
            except Exception as e:
                logger.error(f"Error collecting Kalshi data at {current_time}: {e}")
            
            current_time += timedelta(minutes=interval_minutes)
        
        self.market_data[market_ticker] = data_points
        logger.info(f"Collected {len(data_points)} Kalshi data points for {market_ticker}")
    
    async def _get_kalshi_snapshot(
        self,
        market_ticker: str,
        timestamp: datetime
    ) -> Optional[Dict[str, Any]]:
        """Get Kalshi market snapshot at specific time"""
        # In production, this would query historical data API
        # For now, simulate with mock data
        
        # Mock implementation
        import random
        
        base_price = 0.5 + random.gauss(0, 0.1)
        base_price = max(0.01, min(0.99, base_price))
        
        return {
            'yes_price': base_price,
            'no_price': 1 - base_price,
            'yes_bid': base_price - 0.01,
            'yes_ask': base_price + 0.01,
            'no_bid': (1 - base_price) - 0.01,
            'no_ask': (1 - base_price) + 0.01,
            'volume': random.randint(1000, 50000),
            'open_interest': random.randint(5000, 100000),
            'last_trade_time': timestamp.isoformat()
        }
    
    async def _collect_espn_data(
        self,
        game_id: str,
        teams: tuple,
        start_date: datetime,
        end_date: datetime
    ):
        """Collect historical ESPN game data"""
        data_points = []
        
        try:
            # Get game events
            game_events = await self._get_espn_game_events(game_id, start_date, end_date)
            
            for event in game_events:
                data_points.append({
                    'timestamp': event['timestamp'],
                    'game_id': game_id,
                    'home_team': teams[0],
                    'away_team': teams[1],
                    'home_score': event.get('home_score', 0),
                    'away_score': event.get('away_score', 0),
                    'quarter': event.get('quarter', 1),
                    'time_remaining': event.get('time_remaining', '15:00'),
                    'event_type': event.get('event_type'),
                    'event_description': event.get('description'),
                    'win_probability': event.get('win_probability', 0.5),
                    'momentum': event.get('momentum', 0)
                })
                
        except Exception as e:
            logger.error(f"Error collecting ESPN data for game {game_id}: {e}")
        
        self.game_data[game_id] = data_points
        logger.info(f"Collected {len(data_points)} ESPN data points for game {game_id}")
    
    async def _get_espn_game_events(
        self,
        game_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get ESPN game events"""
        # Mock implementation
        events = []
        
        # Simulate game progression
        game_time = start_date
        quarter = 1
        home_score = 0
        away_score = 0
        
        while game_time <= end_date and quarter <= 4:
            # Random scoring events
            import random
            
            if random.random() < 0.1:  # 10% chance of scoring
                if random.random() < 0.5:
                    home_score += random.choice([3, 7])
                    event_type = "touchdown" if home_score % 7 == 0 else "field_goal"
                else:
                    away_score += random.choice([3, 7])
                    event_type = "touchdown" if away_score % 7 == 0 else "field_goal"
            else:
                event_type = "play"
            
            events.append({
                'timestamp': game_time,
                'home_score': home_score,
                'away_score': away_score,
                'quarter': quarter,
                'time_remaining': f"{15 - (game_time - start_date).seconds // 60 % 15}:00",
                'event_type': event_type,
                'description': f"Q{quarter} - {event_type}",
                'win_probability': 0.5 + (home_score - away_score) * 0.02,
                'momentum': (home_score - away_score) / 10
            })
            
            game_time += timedelta(minutes=5)
            
            # Advance quarter
            if (game_time - start_date).seconds // 60 % 60 == 0:
                quarter += 1
        
        return events
    
    async def _collect_twitter_data(
        self,
        teams: tuple,
        start_date: datetime,
        end_date: datetime,
        interval_minutes: int
    ):
        """Collect historical Twitter sentiment data"""
        data_points = []
        current_time = start_date
        
        while current_time <= end_date:
            try:
                # Get sentiment for each team
                for team in teams:
                    sentiment = await self._get_twitter_sentiment(team, current_time)
                    
                    if sentiment:
                        data_points.append({
                            'timestamp': current_time,
                            'team': team,
                            'sentiment_score': sentiment['score'],
                            'tweet_volume': sentiment['volume'],
                            'positive_count': sentiment['positive'],
                            'negative_count': sentiment['negative'],
                            'neutral_count': sentiment['neutral'],
                            'engagement_rate': sentiment['engagement'],
                            'influencer_sentiment': sentiment['influencer_sentiment'],
                            'sentiment_velocity': self._calculate_sentiment_velocity(
                                data_points, team, sentiment
                            )
                        })
                
            except Exception as e:
                logger.error(f"Error collecting Twitter data at {current_time}: {e}")
            
            current_time += timedelta(minutes=interval_minutes)
        
        teams_key = f"{teams[0]}_vs_{teams[1]}"
        self.sentiment_data[teams_key] = data_points
        logger.info(f"Collected {len(data_points)} Twitter data points for {teams_key}")
    
    async def _get_twitter_sentiment(
        self,
        team: str,
        timestamp: datetime
    ) -> Optional[Dict[str, Any]]:
        """Get Twitter sentiment for team at specific time"""
        # Mock implementation
        import random
        
        base_sentiment = random.gauss(0, 0.3)
        base_sentiment = max(-1, min(1, base_sentiment))
        
        volume = random.randint(100, 5000)
        positive = int(volume * (0.5 + base_sentiment / 2))
        negative = int(volume * (0.5 - base_sentiment / 2))
        neutral = volume - positive - negative
        
        return {
            'score': base_sentiment,
            'volume': volume,
            'positive': positive,
            'negative': negative,
            'neutral': neutral,
            'engagement': random.uniform(0.01, 0.1),
            'influencer_sentiment': base_sentiment * 1.2
        }
    
    def _calculate_price_change(
        self,
        history: List[Dict],
        current: Dict
    ) -> float:
        """Calculate price change from previous data point"""
        if not history:
            return 0.0
        
        prev_price = history[-1].get('yes_price', 0.5)
        curr_price = current.get('yes_price', 0.5)
        
        if prev_price == 0:
            return 0.0
        
        return (curr_price - prev_price) / prev_price
    
    def _calculate_volume_ratio(
        self,
        history: List[Dict],
        current: Dict
    ) -> float:
        """Calculate volume ratio vs average"""
        if len(history) < 10:
            return 1.0
        
        # Get last 10 volumes
        recent_volumes = [h.get('volume', 0) for h in history[-10:]]
        avg_volume = sum(recent_volumes) / len(recent_volumes) if recent_volumes else 1
        
        if avg_volume == 0:
            return 1.0
        
        return current.get('volume', 0) / avg_volume
    
    def _calculate_sentiment_velocity(
        self,
        history: List[Dict],
        team: str,
        current: Dict
    ) -> float:
        """Calculate rate of sentiment change"""
        team_history = [h for h in history if h.get('team') == team]
        
        if not team_history:
            return 0.0
        
        prev_sentiment = team_history[-1].get('sentiment_score', 0)
        curr_sentiment = current.get('score', 0)
        
        return curr_sentiment - prev_sentiment
    
    def _extract_game_info(self, market_ticker: str) -> Optional[Dict[str, Any]]:
        """Extract game information from market ticker"""
        # Example: "NFL-CHIEFS-BILLS-20240115"
        parts = market_ticker.split('-')
        
        if len(parts) >= 4:
            return {
                'sport': parts[0],
                'teams': (parts[1], parts[2]),
                'game_id': parts[3] if len(parts) > 3 else 'unknown'
            }
        
        return None
    
    async def _save_market_data(self, market_ticker: str):
        """Save collected data to CSV files"""
        # Save market data
        if market_ticker in self.market_data:
            df = pd.DataFrame(self.market_data[market_ticker])
            output_file = self.output_dir / f"{market_ticker}_historical.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"Saved market data to {output_file}")
        
        # Save game data
        game_info = self._extract_game_info(market_ticker)
        if game_info and game_info['game_id'] in self.game_data:
            df = pd.DataFrame(self.game_data[game_info['game_id']])
            output_file = self.output_dir / f"{game_info['game_id']}_game.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"Saved game data to {output_file}")
        
        # Save sentiment data
        if game_info:
            teams_key = f"{game_info['teams'][0]}_vs_{game_info['teams'][1]}"
            if teams_key in self.sentiment_data:
                df = pd.DataFrame(self.sentiment_data[teams_key])
                output_file = self.output_dir / f"{teams_key}_sentiment.csv"
                df.to_csv(output_file, index=False)
                logger.info(f"Saved sentiment data to {output_file}")
    
    async def load_or_collect(
        self,
        market_ticker: str,
        start_date: datetime,
        end_date: datetime,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Load existing data or collect new data
        
        Args:
            market_ticker: Market to load/collect
            start_date: Start date
            end_date: End date
            force_refresh: Force new collection even if data exists
            
        Returns:
            DataFrame with historical data
        """
        output_file = self.output_dir / f"{market_ticker}_historical.csv"
        
        if output_file.exists() and not force_refresh:
            logger.info(f"Loading existing data from {output_file}")
            return pd.read_csv(output_file, parse_dates=['timestamp'])
        
        # Collect new data
        await self.collect_historical_data(
            start_date=start_date,
            end_date=end_date,
            markets=[market_ticker],
            interval_minutes=5
        )
        
        if output_file.exists():
            return pd.read_csv(output_file, parse_dates=['timestamp'])
        
        return pd.DataFrame()


# Example usage
async def main():
    """Example data collection"""
    collector = HistoricalDataCollector()
    
    # Collect historical data
    await collector.collect_historical_data(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 31),
        markets=[
            "NFL-CHIEFS-BILLS-20240115",
            "NFL-RAVENS-TEXANS-20240120"
        ],
        interval_minutes=5
    )
    
    logger.info("Historical data collection complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())