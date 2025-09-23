"""
Simple Arbitrage Strategy - Educational Example

This strategy demonstrates basic arbitrage concepts by identifying price
discrepancies between different markets or data sources for the same event.

Key Features:
- Cross-market price comparison
- Basic arbitrage opportunity detection
- Educational risk-free profit identification
- Framework demonstration

Note: This is a simplified educational example. Real arbitrage requires:
- Real-time execution capabilities
- Advanced risk management
- Transaction cost modeling
- Sophisticated opportunity validation
"""

from typing import Dict, Optional, Any, List
import logging
from datetime import datetime, timedelta

from neural.analysis.base import AnalysisResult, AnalysisType, SignalStrength
from neural.strategy.base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class SimpleArbitrageStrategy(BaseStrategy):
    """
    Educational simple arbitrage strategy.
    
    This strategy identifies basic price discrepancies that might represent
    arbitrage opportunities between different markets or pricing sources.
    
    Key Concepts Demonstrated:
    - Cross-market price comparison
    - Risk-free profit identification
    - Basic execution logic
    - Transaction cost consideration
    
    This is an educational implementation. Real arbitrage strategies require:
    - Sub-second execution
    - Advanced risk management  
    - Real-time market access
    - Sophisticated validation logic
    """
    
    def __init__(
        self, 
        min_arbitrage_pct: float = 0.03,
        transaction_cost: float = 0.02,
        max_execution_time: int = 300  # 5 minutes
    ):
        """
        Initialize simple arbitrage strategy.
        
        Args:
            min_arbitrage_pct: Minimum profit margin to consider (default: 3%)
            transaction_cost: Estimated total transaction costs (default: 2%)
            max_execution_time: Maximum time to execute trade (seconds)
        """
        super().__init__("SimpleArbitrage")
        self.min_arbitrage_pct = min_arbitrage_pct
        self.transaction_cost = transaction_cost
        self.max_execution_time = max_execution_time
        
        # Net minimum required after costs
        self.net_min_profit = min_arbitrage_pct + transaction_cost
        
        logger.info(f"Initialized {self.name} strategy")
    
    async def analyze(self, market_id: str, market_data: Dict[str, Any]) -> Optional[Signal]:
        """
        Identify potential arbitrage opportunities.
        
        Args:
            market_id: Market identifier
            market_data: Dictionary containing market information
            
        Returns:
            Signal object or None if no arbitrage found
        """
        try:
            # Extract price sources
            kalshi_price = market_data.get('kalshi_price')
            sportsbook_prices = market_data.get('sportsbook_prices', {})
            
            if kalshi_price is None or not sportsbook_prices:
                logger.debug(f"Insufficient price data for arbitrage analysis: {market_id}")
                return None
            
            # Find best arbitrage opportunity
            best_opportunity = self._find_best_arbitrage(kalshi_price, sportsbook_prices)
            
            if not best_opportunity:
                return None
                
            # Validate opportunity
            if not self._validate_opportunity(best_opportunity, market_data):
                return None
                
            # Generate signal
            return self._generate_arbitrage_signal(
                market_id, best_opportunity, market_data
            )
            
        except Exception as e:
            logger.error(f"Error analyzing arbitrage for {market_id}: {e}")
            return None
    
    def _find_best_arbitrage(
        self, 
        kalshi_price: float, 
        sportsbook_prices: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """
        Find the best arbitrage opportunity among available prices.
        
        This is a basic implementation for educational purposes.
        Real arbitrage systems need sophisticated validation and execution.
        """
        best_opportunity = None
        max_profit = 0
        
        for source, sb_price in sportsbook_prices.items():
            if sb_price is None:
                continue
                
            # Check both directions of the arbitrage
            
            # Opportunity 1: Buy Kalshi YES, Sell Sportsbook YES (implied)
            # This happens when Kalshi price < Sportsbook implied price
            if kalshi_price < sb_price:
                profit_pct = (sb_price - kalshi_price) / kalshi_price
                if profit_pct > self.net_min_profit and profit_pct > max_profit:
                    max_profit = profit_pct
                    best_opportunity = {
                        'type': 'buy_kalshi_yes',
                        'kalshi_price': kalshi_price,
                        'sportsbook_price': sb_price,
                        'source': source,
                        'profit_pct': profit_pct,
                        'action': 'BUY_YES'
                    }
            
            # Opportunity 2: Buy Kalshi NO, Sell Sportsbook NO (implied)  
            # This happens when Kalshi price > Sportsbook implied price
            elif kalshi_price > sb_price:
                # For NO side, we compare (1 - kalshi_price) vs (1 - sb_price)
                kalshi_no_price = 1 - kalshi_price
                sb_no_price = 1 - sb_price
                
                if kalshi_no_price < sb_no_price:
                    profit_pct = (sb_no_price - kalshi_no_price) / kalshi_no_price
                    if profit_pct > self.net_min_profit and profit_pct > max_profit:
                        max_profit = profit_pct
                        best_opportunity = {
                            'type': 'buy_kalshi_no',
                            'kalshi_price': kalshi_price,
                            'sportsbook_price': sb_price,
                            'source': source,
                            'profit_pct': profit_pct,
                            'action': 'BUY_NO'
                        }
        
        return best_opportunity
    
    def _validate_opportunity(
        self, 
        opportunity: Dict[str, Any], 
        market_data: Dict[str, Any]
    ) -> bool:
        """
        Validate that the arbitrage opportunity is real and executable.
        
        Basic validation for educational purposes.
        Real systems need comprehensive validation.
        """
        # Check market liquidity
        volume = market_data.get('volume_24h', 0)
        if volume < 1000:  # Low liquidity might indicate stale prices
            logger.debug("Rejecting arbitrage: insufficient volume")
            return False
        
        # Check spread (wide spreads indicate poor liquidity)
        spread = market_data.get('spread', 0)
        if spread > 0.05:  # 5% spread might eat into arbitrage profit
            logger.debug("Rejecting arbitrage: spread too wide")
            return False
        
        # Check data freshness (basic implementation)
        last_update = market_data.get('last_update_time')
        if last_update:
            age_seconds = (datetime.now() - last_update).total_seconds()
            if age_seconds > 300:  # 5 minutes old
                logger.debug("Rejecting arbitrage: stale data")
                return False
        
        # Check if profit margin is still sufficient after deeper analysis
        expected_profit = opportunity['profit_pct'] - self.transaction_cost
        if expected_profit < self.min_arbitrage_pct:
            logger.debug("Rejecting arbitrage: insufficient profit after costs")
            return False
        
        return True
    
    def _generate_arbitrage_signal(
        self, 
        market_id: str, 
        opportunity: Dict[str, Any], 
        market_data: Dict[str, Any]
    ) -> Signal:
        """
        Generate trading signal for the arbitrage opportunity.
        """
        # Calculate position size (conservative for arbitrage)
        position_size = self._calculate_arbitrage_position_size(opportunity, market_data)
        
        # High confidence for validated arbitrage
        confidence = min(0.85 + (opportunity['profit_pct'] * 2), 0.95)
        
        # Signal strength based on profit margin
        signal_strength = self._calculate_signal_strength(opportunity['profit_pct'])
        
        return Signal(
            strategy_id=self.name,
            market_id=market_id,
            action=opportunity['action'],
            confidence=confidence,
            signal_strength=signal_strength,
            position_size=position_size,
            timestamp=datetime.now(),
            reasoning=f"Arbitrage vs {opportunity['source']}: {opportunity['profit_pct']:.1%} profit",
            metadata={
                'opportunity_type': opportunity['type'],
                'kalshi_price': opportunity['kalshi_price'],
                'sportsbook_price': opportunity['sportsbook_price'],
                'source': opportunity['source'],
                'gross_profit_pct': opportunity['profit_pct'],
                'net_profit_pct': opportunity['profit_pct'] - self.transaction_cost,
                'execution_urgency': 'HIGH'  # Arbitrage requires fast execution
            }
        )
    
    def _calculate_signal_strength(self, profit_pct: float) -> SignalStrength:
        """
        Calculate signal strength based on arbitrage profit margin.
        """
        if profit_pct >= 0.10:  # 10%+ arbitrage
            return SignalStrength.STRONG
        elif profit_pct >= 0.06:  # 6%+ arbitrage  
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK
    
    def _calculate_arbitrage_position_size(
        self, 
        opportunity: Dict[str, Any], 
        market_data: Dict[str, Any]
    ) -> float:
        """
        Calculate position size for arbitrage opportunity.
        
        Arbitrage position sizing should be more aggressive than directional
        strategies since the risk is theoretically lower.
        """
        base_size = 0.05  # 5% base for arbitrage
        
        # Scale with profit margin
        profit_multiplier = min(opportunity['profit_pct'] / 0.03, 3.0)  # Up to 3x for high profit
        
        # Scale with liquidity
        volume = market_data.get('volume_24h', 0)
        if volume > 10000:
            liquidity_multiplier = 1.2
        elif volume > 5000:
            liquidity_multiplier = 1.0
        else:
            liquidity_multiplier = 0.8
        
        position_size = base_size * profit_multiplier * liquidity_multiplier
        
        # Conservative cap even for arbitrage
        return min(position_size, 0.15)  # Max 15% for single arbitrage
    
    def get_required_data_sources(self) -> list:
        """Return list of required data sources for this strategy."""
        return [
            'kalshi_price',
            'sportsbook_prices',  # Dict of source -> price
            'volume_24h',
            'spread', 
            'last_update_time'
        ]
    
    def get_strategy_description(self) -> str:
        """Return human-readable strategy description."""
        return (
            f"Simple arbitrage strategy that identifies price discrepancies "
            f"≥{self.min_arbitrage_pct:.1%} between Kalshi and other markets. "
            f"Educational example - real arbitrage requires sophisticated execution."
        )
