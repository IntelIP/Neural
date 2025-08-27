"""
E2B Sandbox Executor
Direct E2B SDK integration without Agno wrapper
"""

import os
import logging
from typing import Dict, Any, Optional
from e2b import Sandbox

logger = logging.getLogger(__name__)


class E2BExecutor:
    """Direct E2B SDK client for sandbox execution"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize E2B executor
        
        Args:
            api_key: E2B API key (or from environment)
        """
        self.api_key = api_key or os.getenv("E2B_API_KEY")
        if not self.api_key:
            logger.warning("E2B API key not provided - sandbox features disabled")
    
    async def run_python_code(
        self,
        code: str,
        timeout: int = 30,
        packages: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Execute Python code in E2B sandbox
        
        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds
            packages: Optional list of packages to install
            
        Returns:
            Execution result with output and any errors
        """
        if not self.api_key:
            return {"error": "E2B API key not configured"}
        
        try:
            # Create sandbox
            sandbox = await Sandbox.create(
                api_key=self.api_key,
                template="python3"
            )
            
            # Install packages if needed
            if packages:
                for package in packages:
                    await sandbox.run(f"pip install {package}")
            
            # Execute code
            result = await sandbox.run_python(code, timeout=timeout)
            
            # Parse result
            output = {
                "success": result.exit_code == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.exit_code
            }
            
            # Try to extract returned value
            if result.stdout and "Result:" in result.stdout:
                # Parse structured output if available
                lines = result.stdout.split('\n')
                for line in lines:
                    if line.startswith("Result:"):
                        import json
                        try:
                            output["result"] = json.loads(line.replace("Result:", "").strip())
                        except:
                            output["result"] = line.replace("Result:", "").strip()
            
            # Close sandbox
            await sandbox.close()
            
            return output
            
        except Exception as e:
            logger.error(f"E2B execution error: {e}")
            return {"error": str(e), "success": False}
    
    async def calculate_var(
        self,
        positions: list,
        confidence_level: float = 0.95,
        num_simulations: int = 10000
    ) -> Dict[str, Any]:
        """
        Calculate Value at Risk using E2B sandbox
        
        Args:
            positions: List of position dictionaries
            confidence_level: VaR confidence level
            num_simulations: Number of Monte Carlo simulations
            
        Returns:
            VaR calculation results
        """
        var_code = f"""
import numpy as np
import json
from scipy import stats

positions = {positions}
confidence_level = {confidence_level}
num_simulations = {num_simulations}

# Monte Carlo simulation
np.random.seed(42)
portfolio_returns = []

for _ in range(num_simulations):
    daily_return = 0
    for position in positions:
        # Simulate price movement
        price_change = np.random.normal(0, 0.1)  # 10% daily volatility
        position_return = position['contracts'] * price_change * 100
        daily_return += position_return
    portfolio_returns.append(daily_return)

# Calculate VaR
portfolio_returns = np.array(portfolio_returns)
var_percentile = (1 - confidence_level) * 100
value_at_risk = np.percentile(portfolio_returns, var_percentile)

# Calculate Expected Shortfall (CVaR)
tail_losses = portfolio_returns[portfolio_returns <= value_at_risk]
expected_shortfall = np.mean(tail_losses) if len(tail_losses) > 0 else value_at_risk

# Calculate statistics
mean_return = np.mean(portfolio_returns)
std_return = np.std(portfolio_returns)

result = {{
    "value_at_risk": float(value_at_risk),
    "expected_shortfall": float(expected_shortfall),
    "mean_return": float(mean_return),
    "std_return": float(std_return),
    "confidence_level": confidence_level,
    "num_simulations": num_simulations
}}

print(f"Result: {{json.dumps(result)}}")
"""
        
        return await self.run_python_code(
            var_code,
            packages=["numpy", "scipy"]
        )
    
    async def run_kelly_calculation(
        self,
        probability: float,
        odds: float,
        confidence: float
    ) -> Dict[str, Any]:
        """
        Calculate Kelly Criterion position size
        
        Args:
            probability: Win probability (0-1)
            odds: Decimal odds
            confidence: Confidence in prediction (0-1)
            
        Returns:
            Kelly fraction and adjusted size
        """
        kelly_code = f"""
import json

# Kelly Criterion calculation
probability = {probability}
odds = {odds}
confidence = {confidence}

# Basic Kelly formula: f = (p * b - q) / b
# where p = win probability, q = loss probability, b = odds
q = 1 - probability
b = odds - 1  # Convert to net odds

kelly_fraction = (probability * b - q) / b if b > 0 else 0

# Adjust for confidence
adjusted_kelly = kelly_fraction * confidence

# Apply Kelly cap (max 25% of capital)
final_size = min(max(adjusted_kelly, 0), 0.25)

result = {{
    "kelly_fraction": float(kelly_fraction),
    "confidence_adjusted": float(adjusted_kelly),
    "final_size": float(final_size),
    "probability": probability,
    "odds": odds,
    "confidence": confidence
}}

print(f"Result: {{json.dumps(result)}}")
"""
        
        return await self.run_python_code(kelly_code)


# Singleton instance
e2b_executor = None

def get_e2b_executor() -> E2BExecutor:
    """Get or create E2B executor singleton"""
    global e2b_executor
    if e2b_executor is None:
        e2b_executor = E2BExecutor()
    return e2b_executor