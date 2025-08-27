"""
Risk Manager Agent
Monitors portfolio risk and enforces position limits
Powered by Google Gemini 2.5 Flash with E2B for advanced risk analytics
"""

import os
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
import json

from trading_logic.llm_client import get_llm_client
from trading_logic.e2b_executor import get_e2b_executor
from trading_logic.stop_loss import DynamicStopLoss

# Simple Kelly Calculator
class KellyCalculator:
    def calculate_kelly_fraction(self, prob_win: float, odds: float) -> float:
        """Calculate Kelly fraction for position sizing"""
        q = 1 - prob_win
        b = odds - 1
        return (prob_win * b - q) / b if b > 0 else 0

logger = logging.getLogger(__name__)


class RiskManagerAgent:
    """
    Risk Manager Agent - Portfolio risk management with E2B analytics.
    
    Responsibilities:
    - Monitor portfolio exposure and concentration
    - Calculate Value at Risk (VaR) using E2B
    - Run Monte Carlo risk simulations
    - Enforce position limits and Kelly constraints
    - Trigger emergency stops when needed
    - Manage correlated position risk
    """
    
    def __init__(self):
        """Initialize Risk Manager Agent."""
        # Initialize LLM client
        self.llm_client = get_llm_client()
        
        # E2B executor for risk analytics
        self.e2b_executor = get_e2b_executor()
        
        # Risk calculators
        self.kelly_calculator = KellyCalculator()
        self.stop_loss_calculator = DynamicStopLoss()
        
        # Message handler will be set by the handler wrapper
        self.message_handler = None
        
        # Risk parameters
        self.max_portfolio_risk = float(os.getenv("MAX_PORTFOLIO_RISK", "0.40"))  # 40% max exposure
        self.max_daily_loss = float(os.getenv("MAX_DAILY_LOSS", "0.05"))  # 5% daily loss limit
        self.max_position_concentration = float(os.getenv("MAX_POSITION_CONCENTRATION", "0.15"))  # 15% max single position
        self.correlation_threshold = float(os.getenv("CORRELATION_THRESHOLD", "0.7"))  # High correlation threshold
        self.var_confidence = float(os.getenv("VAR_CONFIDENCE", "0.95"))  # 95% VaR
        
        # Portfolio tracking
        self.positions: Dict[str, Dict] = {}
        self.daily_pnl = 0.0
        self.risk_metrics: Dict[str, Any] = {}
        
        # State
        self.is_running = False
        self.tasks: List[asyncio.Task] = []
    
    async def start(self):
        """Start the Risk Manager agent."""
        if self.is_running:
            logger.warning("Risk Manager already running")
            return
        
        self.is_running = True
        logger.info("Starting Risk Manager Agent...")
        
        # Connection and message handlers handled by Agentuity
        # (handlers will be called directly by wrapper)
        
        # Start monitoring tasks
        self.tasks.append(
            asyncio.create_task(self._risk_monitoring_loop())
        )
        self.tasks.append(
            asyncio.create_task(self._var_calculation_loop())
        )
        
        logger.info("Risk Manager Agent started successfully")
    
    async def calculate_portfolio_var(self) -> Dict[str, Any]:
        """
        Calculate Value at Risk using E2B Monte Carlo simulation.
        
        Returns:
            VaR metrics and risk assessment
        """
        if not self.positions:
            return {"var": 0, "expected_shortfall": 0, "risk_level": "low"}
        
        # Prepare position data
        position_data = []
        for ticker, pos in self.positions.items():
            position_data.append({
                "ticker": ticker,
                "contracts": pos.get("contracts", 0),
                "entry_price": pos.get("entry_price", 0.5),
                "current_price": pos.get("current_price", 0.5),
                "volatility": pos.get("volatility", 0.1),
                "probability": pos.get("probability", 0.5)
            })
        
        # E2B code for VaR calculation
        var_code = f"""
import numpy as np
import pandas as pd
from scipy import stats

# Position data
positions = {json.dumps(position_data)}

# Monte Carlo parameters
num_simulations = 10000
time_horizon = 1  # 1 day VaR
confidence_level = {self.var_confidence}

# Run Monte Carlo simulation
portfolio_returns = []

for _ in range(num_simulations):
    daily_return = 0
    
    for pos in positions:
        # Simulate price movement based on volatility
        price_change = np.random.normal(0, pos['volatility'] * np.sqrt(time_horizon))
        new_price = pos['current_price'] * (1 + price_change)
        new_price = max(0.01, min(0.99, new_price))  # Bound between 0.01 and 0.99
        
        # Calculate P&L
        if pos['contracts'] > 0:  # Long position
            pnl = (new_price - pos['current_price']) * pos['contracts'] * 100  # $100 per point
        else:  # Short position
            pnl = (pos['current_price'] - new_price) * abs(pos['contracts']) * 100
        
        daily_return += pnl
    
    portfolio_returns.append(daily_return)

# Calculate VaR and Expected Shortfall
portfolio_returns = np.array(portfolio_returns)
var_percentile = (1 - confidence_level) * 100
value_at_risk = np.percentile(portfolio_returns, var_percentile)

# Expected Shortfall (Conditional VaR)
losses_beyond_var = portfolio_returns[portfolio_returns <= value_at_risk]
expected_shortfall = np.mean(losses_beyond_var) if len(losses_beyond_var) > 0 else value_at_risk

# Risk metrics
mean_return = np.mean(portfolio_returns)
std_return = np.std(portfolio_returns)
skewness = stats.skew(portfolio_returns)
kurtosis = stats.kurtosis(portfolio_returns)

# Risk level assessment
total_exposure = sum(abs(p['contracts'] * p['current_price'] * 100) for p in positions)
risk_ratio = abs(value_at_risk) / total_exposure if total_exposure > 0 else 0

if risk_ratio < 0.05:
    risk_level = "low"
elif risk_ratio < 0.10:
    risk_level = "moderate"
elif risk_ratio < 0.15:
    risk_level = "high"
else:
    risk_level = "critical"

results = {{
    "value_at_risk": float(value_at_risk),
    "expected_shortfall": float(expected_shortfall),
    "mean_return": float(mean_return),
    "std_return": float(std_return),
    "skewness": float(skewness),
    "kurtosis": float(kurtosis),
    "total_exposure": float(total_exposure),
    "risk_ratio": float(risk_ratio),
    "risk_level": risk_level,
    "confidence_level": confidence_level,
    "num_simulations": num_simulations
}}

print(f"VaR at {{confidence_level*100}}% confidence: ${{abs(value_at_risk):.2f}}")
print(f"Expected Shortfall: ${{abs(expected_shortfall):.2f}}")
print(f"Risk Level: {{risk_level}}")

results
        """
        
        try:
            # Run VaR calculation in E2B
            result = await self.e2b_executor.run_python_code(var_code, timeout=300)
            
            if result and not result.get("error"):
                var_metrics = result.get("results", {})
                self.risk_metrics["var"] = var_metrics
                return var_metrics
            else:
                logger.error(f"VaR calculation error: {result.get('error')}")
                return {"error": "VaR calculation failed"}
                
        except Exception as e:
            logger.error(f"VaR calculation exception: {e}")
            return {"error": str(e)}
    
    async def check_correlation_risk(self) -> Dict[str, Any]:
        """
        Check for correlated positions using E2B.
        
        Returns:
            Correlation matrix and risk assessment
        """
        if len(self.positions) < 2:
            return {"correlation_risk": "low", "pairs": []}
        
        # Use Gemini to analyze correlations
        position_list = [
            {"ticker": ticker, "contracts": pos["contracts"]}
            for ticker, pos in self.positions.items()
        ]
        
        analysis_prompt = f"""
        Analyze correlation risk for these positions:
        {json.dumps(position_list, indent=2)}
        
        Consider:
        1. Same team/player correlations
        2. Same sport/league correlations
        3. Same event time correlations
        4. Inverse correlations (opposing outcomes)
        
        Identify high-risk correlated pairs.
        """
        
        response_text = await self.llm_client.complete(
            analysis_prompt,
            temperature=0.1,
            system_prompt="You are a risk analysis expert analyzing portfolio correlations."
        )
        
        # E2B code for numerical correlation analysis
        correlation_code = f"""
import numpy as np
import pandas as pd

# Simulated correlation matrix (would use historical data in production)
positions = {json.dumps(list(self.positions.keys()))}
n_positions = len(positions)

# Create correlation matrix based on market relationships
correlation_matrix = np.eye(n_positions)

# Identify correlated pairs
high_correlation_pairs = []
correlation_threshold = {self.correlation_threshold}

for i in range(n_positions):
    for j in range(i+1, n_positions):
        # Check for same sport/event
        pos1, pos2 = positions[i], positions[j]
        
        # Simple heuristic: positions in same market are highly correlated
        if pos1.split('-')[0] == pos2.split('-')[0]:  # Same event prefix
            correlation = 0.8
        else:
            correlation = np.random.uniform(-0.3, 0.3)  # Random low correlation
        
        correlation_matrix[i, j] = correlation
        correlation_matrix[j, i] = correlation
        
        if abs(correlation) > correlation_threshold:
            high_correlation_pairs.append({{
                "pair": [pos1, pos2],
                "correlation": float(correlation),
                "risk": "high" if correlation > 0 else "hedge"
            }})

# Calculate portfolio correlation risk
avg_correlation = np.mean(np.abs(correlation_matrix[np.triu_indices(n_positions, k=1)]))

if avg_correlation < 0.3:
    risk_level = "low"
elif avg_correlation < 0.5:
    risk_level = "moderate"
else:
    risk_level = "high"

results = {{
    "correlation_risk": risk_level,
    "average_correlation": float(avg_correlation),
    "high_correlation_pairs": high_correlation_pairs,
    "correlation_matrix": correlation_matrix.tolist()
}}

print(f"Correlation Risk: {{risk_level}}")
print(f"Found {{len(high_correlation_pairs)}} highly correlated pairs")

results
        """
        
        try:
            result = await self.e2b_executor.run_python_code(correlation_code)
            
            if result and not result.get("error"):
                correlation_metrics = result.get("results", {})
                correlation_metrics["gemini_analysis"] = response_text
                self.risk_metrics["correlation"] = correlation_metrics
                return correlation_metrics
            
        except Exception as e:
            logger.error(f"Correlation analysis error: {e}")
            return {"error": str(e)}
    
    async def optimize_portfolio_risk(self) -> Dict[str, Any]:
        """
        Optimize portfolio using E2B Kelly functions.
        
        Returns:
            Recommended position adjustments
        """
        if not self.positions:
            return {"adjustments": []}
        
        # Prepare position data for Kelly optimization
        positions_data = []
        for ticker, pos in self.positions.items():
            positions_data.append({
                "probability": pos.get("probability", 0.5),
                "odds": pos.get("odds", 1.0),
                "current_size": pos.get("contracts", 0) * pos.get("entry_price", 0.5)
            })
        
        # Get correlation matrix from previous analysis
        correlation_matrix = self.risk_metrics.get("correlation", {}).get("correlation_matrix")
        
        # Run portfolio Kelly optimization using E2B
        optimization_code = f"""
import numpy as np
import json

positions = {json.dumps(positions_data)}
correlation_matrix = {json.dumps(correlation_matrix) if correlation_matrix else 'None'}
max_position_pct = {self.max_position_concentration}
max_total_exposure = {self.max_portfolio_risk}

# Portfolio Kelly optimization
optimal_sizes = []
total_allocation = 0

for pos in positions:
    # Basic Kelly fraction
    p = pos['probability']
    b = pos['odds'] - 1
    q = 1 - p
    
    kelly_fraction = (p * b - q) / b if b > 0 else 0
    
    # Apply constraints
    constrained_fraction = min(kelly_fraction * 0.25, max_position_pct)  # Quarter Kelly with position limit
    
    # Check total exposure
    if total_allocation + constrained_fraction > max_total_exposure:
        constrained_fraction = max(0, max_total_exposure - total_allocation)
    
    total_allocation += constrained_fraction
    
    adjustment = constrained_fraction - (pos['current_size'] / 100000)  # Assume $100k capital
    
    optimal_sizes.append({{
        "optimal_fraction": float(constrained_fraction),
        "current_fraction": float(pos['current_size'] / 100000),
        "adjustment_needed": float(adjustment),
        "action": "increase" if adjustment > 0.01 else "decrease" if adjustment < -0.01 else "hold"
    }})

results = {{
    "adjustments": optimal_sizes,
    "total_allocation": float(total_allocation),
    "remaining_capacity": float(max_total_exposure - total_allocation)
}}

print(json.dumps(results, indent=2))
results
"""
        
        result = await self.e2b_executor.run_python_code(optimization_code)
        
        if result and not result.get("error"):
            return result.get("result", {"adjustments": []})
        else:
            return {"adjustments": []}
        
        return optimization
    
    async def _handle_trade_executed(self, data: Dict[str, Any]):
        """Handle trade execution notification."""
        try:
            market_ticker = data.get("market_ticker")
            action = data.get("action")
            result = data.get("result", {})
            
            if result.get("status") == "success":
                # Update position tracking
                if market_ticker not in self.positions:
                    self.positions[market_ticker] = {
                        "contracts": 0,
                        "entry_price": 0,
                        "side": "YES" if "YES" in action else "NO"
                    }
                
                position = self.positions[market_ticker]
                
                if "BUY" in action:
                    # Adding to position
                    new_contracts = result.get("contracts", 0)
                    new_price = result.get("price", 0.5)
                    
                    total_value = (position["contracts"] * position["entry_price"]) + \
                                 (new_contracts * new_price)
                    position["contracts"] += new_contracts
                    position["entry_price"] = total_value / position["contracts"] if position["contracts"] > 0 else 0
                    position["probability"] = data.get("probability", 0.5)
                    position["stop_loss"] = result.get("stop_loss", 0)
                else:
                    # Reducing position
                    position["contracts"] -= result.get("contracts", 0)
                
                # Check risk limits
                await self._check_risk_limits()
            
        except Exception as e:
            logger.error(f"Error handling trade execution: {e}")
    
    async def _check_risk_limits(self):
        """Check if portfolio exceeds risk limits."""
        # Calculate total exposure
        total_exposure = sum(
            abs(pos["contracts"] * pos.get("entry_price", 0.5))
            for pos in self.positions.values()
        )
        
        # Check concentration risk
        for ticker, pos in self.positions.items():
            position_size = abs(pos["contracts"] * pos.get("entry_price", 0.5))
            concentration = position_size / total_exposure if total_exposure > 0 else 0
            
            if concentration > self.max_position_concentration:
                logger.warning(f"Position concentration exceeded for {ticker}: {concentration:.2%}")
                
                # Send risk alert
                if self.message_handler:
                    await self.message_handler('risk_alert', {
                        "alert_type": "concentration_exceeded",
                        "ticker": ticker,
                        "concentration": concentration,
                        "limit": self.max_position_concentration
                    })
        
        # Check daily loss limit
        if abs(self.daily_pnl) > self.max_daily_loss * 100000:  # Assuming $100k account
            logger.critical(f"Daily loss limit exceeded: ${abs(self.daily_pnl):.2f}")
            
            # Trigger emergency stop
            await self._trigger_emergency_stop("daily_loss_exceeded")
    
    async def _trigger_emergency_stop(self, reason: str):
        """Trigger emergency stop across all agents."""
        logger.critical(f"EMERGENCY STOP TRIGGERED: {reason}")
        
        # Send emergency stop signal
        if self.message_handler:
            await self.message_handler('emergency_stop', {
                "reason": reason,
                "timestamp": datetime.now().isoformat(),
                "positions": self.positions,
                "daily_pnl": self.daily_pnl
            })
        
        # Use Gemini for post-mortem analysis
        analysis_prompt = f"""
        Emergency stop triggered: {reason}
        
        Portfolio state:
        - Positions: {len(self.positions)}
        - Total exposure: ${sum(abs(p['contracts'] * p.get('entry_price', 0.5) * 100) for p in self.positions.values()):.2f}
        - Daily P&L: ${self.daily_pnl:.2f}
        
        Provide brief analysis of what went wrong and recommendations.
        """
        
        response_text = await self.llm_client.complete(
            analysis_prompt,
            temperature=0.1,
            system_prompt="You are a risk analysis expert analyzing portfolio correlations."
        )
        logger.info(f"Post-mortem analysis: {response_text}")
    
    async def _risk_monitoring_loop(self):
        """Main risk monitoring loop."""
        while self.is_running:
            try:
                # Calculate VaR
                var_metrics = await self.calculate_portfolio_var()
                
                # Check correlations
                correlation_risk = await self.check_correlation_risk()
                
                # Optimize portfolio if needed
                if var_metrics.get("risk_level") in ["high", "critical"]:
                    optimization = await self.optimize_portfolio_risk()
                    
                    # Send optimization recommendations
                    if self.message_handler:
                        await self.message_handler('portfolio_optimization', {
                            "var_metrics": var_metrics,
                            "correlation_risk": correlation_risk,
                            "optimization": optimization,
                            "timestamp": datetime.now().isoformat()
                        })
                
                # Log risk metrics
                logger.info(f"Risk Metrics - VaR: ${abs(var_metrics.get('value_at_risk', 0)):.2f}, "
                          f"Risk Level: {var_metrics.get('risk_level', 'unknown')}, "
                          f"Correlation Risk: {correlation_risk.get('correlation_risk', 'unknown')}")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Risk monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _var_calculation_loop(self):
        """Periodic VaR calculation and risk of ruin analysis."""
        while self.is_running:
            try:
                # Run risk of ruin calculation for each position
                for ticker, pos in self.positions.items():
                    if pos.get("contracts", 0) > 0:
                        # Calculate risk of ruin using E2B
                        kelly_fraction = self.kelly_calculator.calculate_kelly_fraction(
                            prob_win=pos.get("probability", 0.5),
                            odds=pos.get("odds", 1.0)
                        ) * 0.25  # Quarter Kelly
                        
                        risk_code = f"""
import numpy as np
import json

prob_win = {pos.get("probability", 0.5)}
odds = {pos.get("odds", 1.0)}
kelly_fraction = {kelly_fraction}
num_simulations = 1000

# Risk of ruin simulation
ruin_count = 0
initial_bankroll = 1.0

for _ in range(num_simulations):
    bankroll = initial_bankroll
    
    # Simulate 100 bets
    for _ in range(100):
        if bankroll <= 0.1:  # Consider 90% loss as ruin
            ruin_count += 1
            break
        
        # Simulate bet outcome
        if np.random.random() < prob_win:
            # Win
            bankroll += kelly_fraction * bankroll * (odds - 1)
        else:
            # Loss
            bankroll -= kelly_fraction * bankroll

probability_of_ruin = ruin_count / num_simulations

result = {{
    "probability_of_ruin": float(probability_of_ruin),
    "kelly_fraction": float(kelly_fraction),
    "expected_growth": float((prob_win * np.log(1 + kelly_fraction * (odds - 1)) + 
                              (1 - prob_win) * np.log(1 - kelly_fraction)))
}}

print(json.dumps(result))
result
"""
                        
                        risk_result = await self.e2b_executor.run_python_code(risk_code)
                        risk_of_ruin = risk_result.get("result", {}) if not risk_result.get("error") else {}
                        
                        if risk_of_ruin.get("probability_of_ruin", 0) > 0.05:
                            logger.warning(f"High risk of ruin for {ticker}: {risk_of_ruin.get('probability_of_ruin', 0):.2%}")
                
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                logger.error(f"VaR calculation error: {e}")
                await asyncio.sleep(300)
    
    async def _update_portfolio_values(self, data: Dict[str, Any]):
        """Update portfolio values with latest market data."""
        market_ticker = data.get("market_ticker")
        
        if market_ticker in self.positions:
            position = self.positions[market_ticker]
            
            # Update current price
            if position["side"] == "YES":
                position["current_price"] = data.get("yes_bid", position.get("entry_price", 0.5))
            else:
                position["current_price"] = data.get("no_bid", 1 - position.get("entry_price", 0.5))
            
            # Calculate unrealized P&L
            if position["side"] == "YES":
                unrealized_pnl = (position["current_price"] - position["entry_price"]) * position["contracts"] * 100
            else:
                unrealized_pnl = ((1 - position["current_price"]) - position["entry_price"]) * position["contracts"] * 100
            
            position["unrealized_pnl"] = unrealized_pnl
            
            # Update volatility estimate
            position["volatility"] = data.get("volatility", 0.1)
    
    async def _handle_stop_loss(self, data: Dict[str, Any]):
        """Handle stop-loss trigger notification."""
        market_ticker = data.get("market_ticker")
        position = data.get("position", {})
        
        logger.info(f"Stop-loss triggered for {market_ticker}")
        
        # Remove from tracked positions
        if market_ticker in self.positions:
            # Update daily P&L
            realized_pnl = position.get("unrealized_pnl", 0)
            self.daily_pnl += realized_pnl
            
            del self.positions[market_ticker]
        
        # Check if we need to reduce risk further
        await self._check_risk_limits()
    
    async def stop(self):
        """Stop the Risk Manager agent."""
        self.is_running = False
        
        # Cancel tasks
        for task in self.tasks:
            task.cancel()
        
        # Close connections
        # Cleanup handled by Agentuity
        
        logger.info("Risk Manager Agent stopped")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        total_exposure = sum(
            abs(pos["contracts"] * pos.get("entry_price", 0.5) * 100)
            for pos in self.positions.values()
        )
        
        return {
            "is_running": self.is_running,
            "num_positions": len(self.positions),
            "total_exposure": total_exposure,
            "daily_pnl": self.daily_pnl,
            "risk_metrics": self.risk_metrics,
            "risk_limits": {
                "max_portfolio_risk": self.max_portfolio_risk,
                "max_daily_loss": self.max_daily_loss,
                "max_concentration": self.max_position_concentration
            }
        }


# Example usage
async def main():
    """Example of running the Risk Manager agent."""
    
    # Initialize agent
    agent = RiskManagerAgent()
    
    # Start agent
    await agent.start()
    
    # Simulate some positions
    agent.positions = {
        "SUPERBOWL-2025": {
            "contracts": 100,
            "entry_price": 0.60,
            "side": "YES",
            "probability": 0.65,
            "odds": 0.67,
            "current_price": 0.62,
            "volatility": 0.15
        },
        "NBA-FINALS-2025": {
            "contracts": 50,
            "entry_price": 0.45,
            "side": "NO",
            "probability": 0.40,
            "odds": 1.22,
            "current_price": 0.43,
            "volatility": 0.12
        }
    }
    
    # Calculate VaR
    var_metrics = await agent.calculate_portfolio_var()
    print(f"VaR Metrics: {var_metrics}")
    
    # Check correlations
    correlation_risk = await agent.check_correlation_risk()
    print(f"Correlation Risk: {correlation_risk}")
    
    # Optimize portfolio
    optimization = await agent.optimize_portfolio_risk()
    print(f"Portfolio Optimization: {optimization}")
    
    # Run for a while
    await asyncio.sleep(60)
    
    # Get status
    status = await agent.get_status()
    print(f"Agent status: {status}")
    
    # Stop agent
    await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())