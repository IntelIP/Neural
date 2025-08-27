"""
E2B Kelly Functions for Advanced Calculations.
Runs Monte Carlo simulations and backtesting in E2B sandboxes.
"""

from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
from e2b_code_interpreter import Sandbox

logger = logging.getLogger(__name__)


class E2BKellyFunctions:
    """
    Advanced Kelly Criterion calculations using E2B sandboxes.
    
    Runs computationally intensive simulations in isolated environments:
    - Monte Carlo simulations for optimal Kelly fraction
    - Portfolio optimization with multiple positions
    - Backtesting Kelly strategies
    - Risk of ruin calculations
    """
    
    def __init__(self, timeout: int = 300):
        """
        Initialize E2B Kelly functions.
        
        Args:
            timeout: Sandbox timeout in seconds
        """
        self.timeout = timeout
        self.sandbox: Optional[Sandbox] = None
    
    async def initialize(self):
        """Initialize E2B sandbox with required libraries."""
        try:
            self.sandbox = await Sandbox.create(timeout=self.timeout)
            
            # Install required packages
            setup_code = """
import numpy as np
import pandas as pd
from scipy import stats, optimize
import warnings
warnings.filterwarnings('ignore')

# Utility functions for Kelly calculations
def kelly_growth_rate(f, p, b):
    '''Calculate expected growth rate for Kelly fraction f'''
    if f <= 0 or f >= 1:
        return -float('inf')
    return p * np.log(1 + b * f) + (1 - p) * np.log(1 - f)

def optimal_kelly(p, b):
    '''Find optimal Kelly fraction numerically'''
    result = optimize.minimize_scalar(
        lambda f: -kelly_growth_rate(f, p, b),
        bounds=(0.001, 0.999),
        method='bounded'
    )
    return result.x if result.success else (p * b - (1 - p)) / b

print("Kelly functions initialized")
            """
            
            result = await self.sandbox.run_code(setup_code)
            if result.error:
                logger.error(f"E2B setup error: {result.error}")
                return False
            
            logger.info("E2B Kelly sandbox initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize E2B sandbox: {e}")
            return False
    
    async def monte_carlo_kelly(
        self,
        prob_win: float,
        odds: float,
        num_simulations: int = 10000,
        num_bets: int = 100,
        bankroll: float = 10000
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation to find optimal Kelly fraction.
        
        Args:
            prob_win: True winning probability
            odds: Betting odds (profit/stake)
            num_simulations: Number of simulation runs
            num_bets: Number of bets per simulation
            bankroll: Starting bankroll
        
        Returns:
            Optimal Kelly fraction and performance metrics
        """
        if not self.sandbox:
            await self.initialize()
        
        simulation_code = f"""
import numpy as np

def run_kelly_simulation(kelly_fraction, prob_win, odds, num_bets, bankroll):
    '''Simulate betting with given Kelly fraction'''
    capital = bankroll
    history = [capital]
    
    for _ in range(num_bets):
        if capital <= 0:
            break
        
        bet_size = capital * kelly_fraction
        if np.random.random() < prob_win:
            capital += bet_size * odds
        else:
            capital -= bet_size
        
        history.append(capital)
    
    return capital, history

# Parameters
prob_win = {prob_win}
odds = {odds}
num_simulations = {num_simulations}
num_bets = {num_bets}
bankroll = {bankroll}

# Test different Kelly fractions
fractions = np.linspace(0.05, 0.5, 20)
results = {{}}

for fraction in fractions:
    final_capitals = []
    bankruptcies = 0
    
    for _ in range(num_simulations):
        final_capital, _ = run_kelly_simulation(
            fraction, prob_win, odds, num_bets, bankroll
        )
        final_capitals.append(final_capital)
        if final_capital <= bankroll * 0.01:  # 99% loss = bankruptcy
            bankruptcies += 1
    
    results[float(fraction)] = {{
        'mean_return': np.mean(final_capitals) / bankroll,
        'median_return': np.median(final_capitals) / bankroll,
        'std_return': np.std(final_capitals) / bankroll,
        'bankruptcy_rate': bankruptcies / num_simulations,
        'sharpe': (np.mean(final_capitals) - bankroll) / (np.std(final_capitals) + 1e-6),
        'max_return': np.max(final_capitals) / bankroll,
        'min_return': np.min(final_capitals) / bankroll
    }}

# Find optimal fraction
best_fraction = max(results.keys(), key=lambda k: results[k]['median_return'])
theoretical_kelly = (prob_win * odds - (1 - prob_win)) / odds

# Growth rate analysis
growth_rates = {{}}
for fraction in fractions:
    expected_growth = prob_win * np.log(1 + odds * fraction) + (1 - prob_win) * np.log(1 - fraction)
    growth_rates[float(fraction)] = expected_growth

best_growth_fraction = max(growth_rates.keys(), key=lambda k: growth_rates[k])

analysis = {{
    'optimal_fraction': best_fraction,
    'theoretical_kelly': theoretical_kelly,
    'best_growth_fraction': best_growth_fraction,
    'results_by_fraction': results,
    'growth_rates': growth_rates,
    'recommendation': {{
        'full_kelly': theoretical_kelly,
        'half_kelly': theoretical_kelly * 0.5,
        'quarter_kelly': theoretical_kelly * 0.25,
        'simulated_optimal': best_fraction
    }}
}}

print(analysis)
analysis
        """
        
        try:
            result = await self.sandbox.run_code(simulation_code)
            if result.error:
                logger.error(f"Monte Carlo simulation error: {result.error}")
                return {"error": str(result.error)}
            
            return result.results[-1].data if result.results else {}
            
        except Exception as e:
            logger.error(f"Monte Carlo Kelly error: {e}")
            return {"error": str(e)}
    
    async def portfolio_kelly(
        self,
        positions: List[Dict[str, float]],
        correlation_matrix: Optional[List[List[float]]] = None,
        constraints: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Calculate optimal Kelly allocation for portfolio of bets.
        
        Args:
            positions: List of positions with prob, odds, current_allocation
            correlation_matrix: Correlation between positions
            constraints: Max position size, total exposure limits
        
        Returns:
            Optimal portfolio allocation
        """
        if not self.sandbox:
            await self.initialize()
        
        positions_str = str(positions)
        corr_str = str(correlation_matrix) if correlation_matrix else "None"
        constraints_str = str(constraints) if constraints else "{}"
        
        portfolio_code = f"""
import numpy as np
from scipy.optimize import minimize
import pandas as pd

positions = {positions_str}
correlation_matrix = {corr_str}
constraints = {constraints_str}

# Extract position data
probs = np.array([p['probability'] for p in positions])
odds = np.array([p['odds'] for p in positions])
n_positions = len(positions)

# Default correlation matrix if not provided
if correlation_matrix is None:
    correlation_matrix = np.eye(n_positions)
else:
    correlation_matrix = np.array(correlation_matrix)

def portfolio_growth_rate(weights, probs, odds, corr_matrix):
    '''Calculate expected log growth rate for portfolio'''
    weights = np.array(weights)
    
    # Expected returns
    expected_returns = probs * odds - (1 - probs)
    portfolio_return = np.dot(weights, expected_returns)
    
    # Variance calculation with correlations
    variances = []
    for i in range(len(probs)):
        var_i = probs[i] * (1 - probs[i]) * (odds[i] + 1)**2
        variances.append(var_i)
    
    var_matrix = np.outer(np.sqrt(variances), np.sqrt(variances)) * corr_matrix
    portfolio_variance = np.dot(weights, np.dot(var_matrix, weights))
    
    # Log growth approximation
    growth_rate = portfolio_return - 0.5 * portfolio_variance
    return -growth_rate  # Negative for minimization

# Optimization constraints
max_position = constraints.get('max_position_pct', 0.1)
max_total = constraints.get('max_total_exposure', 0.4)

constraint_list = [
    {{'type': 'ineq', 'fun': lambda w: max_total - np.sum(w)}},  # Total exposure limit
    {{'type': 'ineq', 'fun': lambda w: w}}  # Non-negative weights
]

# Individual position limits
for i in range(n_positions):
    constraint_list.append(
        {{'type': 'ineq', 'fun': lambda w, idx=i: max_position - w[idx]}}
    )

# Initial guess (equal weight up to limits)
initial_weights = np.ones(n_positions) * min(max_position, max_total / n_positions) * 0.5

# Optimize
result = minimize(
    portfolio_growth_rate,
    initial_weights,
    args=(probs, odds, correlation_matrix),
    method='SLSQP',
    constraints=constraint_list,
    options={{'maxiter': 1000}}
)

optimal_weights = result.x if result.success else initial_weights

# Calculate metrics
individual_kellys = (probs * odds - (1 - probs)) / odds
portfolio_kelly = np.sum(optimal_weights)

# Risk metrics
portfolio_return = np.dot(optimal_weights, probs * odds - (1 - probs))
variances = [probs[i] * (1 - probs[i]) * (odds[i] + 1)**2 for i in range(n_positions)]
var_matrix = np.outer(np.sqrt(variances), np.sqrt(variances)) * correlation_matrix
portfolio_std = np.sqrt(np.dot(optimal_weights, np.dot(var_matrix, optimal_weights)))

allocation = {{
    'optimal_weights': optimal_weights.tolist(),
    'individual_kellys': individual_kellys.tolist(),
    'total_allocation': portfolio_kelly,
    'expected_return': portfolio_return,
    'portfolio_std': portfolio_std,
    'sharpe_ratio': portfolio_return / (portfolio_std + 1e-6),
    'allocations': [
        {{
            'position': i,
            'weight': optimal_weights[i],
            'kelly': individual_kellys[i],
            'expected_return': (probs[i] * odds[i] - (1 - probs[i])) * optimal_weights[i]
        }}
        for i in range(n_positions)
    ]
}}

print(allocation)
allocation
        """
        
        try:
            result = await self.sandbox.run_code(portfolio_code)
            if result.error:
                logger.error(f"Portfolio Kelly error: {result.error}")
                return {"error": str(result.error)}
            
            return result.results[-1].data if result.results else {}
            
        except Exception as e:
            logger.error(f"Portfolio optimization error: {e}")
            return {"error": str(e)}
    
    async def backtest_kelly_strategy(
        self,
        historical_data: List[Dict[str, Any]],
        strategy: str = "quarter_kelly",
        initial_bankroll: float = 10000
    ) -> Dict[str, Any]:
        """
        Backtest Kelly betting strategy on historical data.
        
        Args:
            historical_data: List of historical bets with outcomes
            strategy: Kelly fraction strategy (full, half, quarter)
            initial_bankroll: Starting capital
        
        Returns:
            Backtest results with performance metrics
        """
        if not self.sandbox:
            await self.initialize()
        
        data_str = str(historical_data)
        
        backtest_code = f"""
import numpy as np
import pandas as pd

historical_data = {data_str}
strategy = '{strategy}'
initial_bankroll = {initial_bankroll}

# Strategy multipliers
strategy_map = {{
    'full_kelly': 1.0,
    'half_kelly': 0.5,
    'quarter_kelly': 0.25,
    'eighth_kelly': 0.125
}}

kelly_multiplier = strategy_map.get(strategy, 0.25)

# Run backtest
bankroll = initial_bankroll
history = []
trades = []

for i, bet in enumerate(historical_data):
    prob = bet.get('estimated_prob', 0.5)
    market_price = bet.get('market_price', 0.5)
    outcome = bet.get('outcome', 0)  # 1 for win, 0 for loss
    
    # Calculate Kelly
    odds = (1 - market_price) / market_price
    kelly = (prob * odds - (1 - prob)) / odds
    
    # Apply strategy
    position_size = kelly * kelly_multiplier * bankroll
    
    # Skip if no edge or position too small
    if kelly <= 0 or position_size < 10:
        history.append({{
            'bet_num': i,
            'bankroll': bankroll,
            'position': 0,
            'outcome': 'skip'
        }})
        continue
    
    # Cap position size
    position_size = min(position_size, bankroll * 0.1)  # Max 10% per bet
    
    # Calculate P&L
    if outcome == 1:
        profit = position_size * odds
        bankroll += profit
        result = 'win'
    else:
        bankroll -= position_size
        result = 'loss'
    
    trades.append({{
        'bet_num': i,
        'estimated_prob': prob,
        'market_price': market_price,
        'kelly': kelly,
        'position_size': position_size,
        'outcome': result,
        'bankroll_after': bankroll
    }})
    
    history.append({{
        'bet_num': i,
        'bankroll': bankroll,
        'position': position_size,
        'outcome': result
    }})

# Calculate metrics
df_trades = pd.DataFrame(trades)
if len(df_trades) > 0:
    total_return = (bankroll - initial_bankroll) / initial_bankroll
    win_rate = len(df_trades[df_trades['outcome'] == 'win']) / len(df_trades)
    
    # Calculate max drawdown
    cumulative = [initial_bankroll]
    for h in history:
        cumulative.append(h['bankroll'])
    
    peak = cumulative[0]
    max_dd = 0
    for value in cumulative:
        if value > peak:
            peak = value
        dd = (peak - value) / peak
        if dd > max_dd:
            max_dd = dd
    
    # Sharpe ratio
    if len(df_trades) > 1:
        returns = df_trades['bankroll_after'].pct_change().dropna()
        sharpe = np.sqrt(252) * returns.mean() / (returns.std() + 1e-6)
    else:
        sharpe = 0
    
    avg_position = df_trades['position_size'].mean()
    avg_kelly = df_trades['kelly'].mean()
else:
    total_return = 0
    win_rate = 0
    max_dd = 0
    sharpe = 0
    avg_position = 0
    avg_kelly = 0

results = {{
    'strategy': strategy,
    'initial_bankroll': initial_bankroll,
    'final_bankroll': bankroll,
    'total_return': total_return,
    'num_trades': len(trades),
    'win_rate': win_rate,
    'max_drawdown': max_dd,
    'sharpe_ratio': sharpe,
    'avg_position_size': avg_position,
    'avg_kelly': avg_kelly,
    'kelly_multiplier': kelly_multiplier,
    'trades': trades[-10:] if trades else []  # Last 10 trades
}}

print(f"Backtest Complete: {{total_return:.2%}} return, {{win_rate:.2%}} win rate")
results
        """
        
        try:
            result = await self.sandbox.run_code(backtest_code)
            if result.error:
                logger.error(f"Backtest error: {result.error}")
                return {"error": str(result.error)}
            
            return result.results[-1].data if result.results else {}
            
        except Exception as e:
            logger.error(f"Backtest Kelly error: {e}")
            return {"error": str(e)}
    
    async def calculate_risk_of_ruin(
        self,
        prob_win: float,
        odds: float,
        kelly_fraction: float,
        target_multiplier: float = 0.1,
        num_simulations: int = 10000
    ) -> Dict[str, float]:
        """
        Calculate probability of ruin for given Kelly fraction.
        
        Args:
            prob_win: Winning probability
            odds: Betting odds
            kelly_fraction: Fraction of Kelly to use
            target_multiplier: Ruin threshold (0.1 = 90% loss)
            num_simulations: Number of simulations
        
        Returns:
            Risk metrics including probability of ruin
        """
        if not self.sandbox:
            await self.initialize()
        
        ruin_code = f"""
import numpy as np

prob_win = {prob_win}
odds = {odds}
kelly_fraction = {kelly_fraction}
target_multiplier = {target_multiplier}
num_simulations = {num_simulations}

# Theoretical Kelly
theoretical_kelly = (prob_win * odds - (1 - prob_win)) / odds
actual_fraction = theoretical_kelly * kelly_fraction

ruins = 0
times_to_ruin = []
max_drawdowns = []

for sim in range(num_simulations):
    bankroll = 1.0  # Start with 1 unit
    peak = 1.0
    max_dd = 0
    
    for bet_num in range(10000):  # Max 10k bets
        if bankroll <= target_multiplier:
            ruins += 1
            times_to_ruin.append(bet_num)
            break
        
        bet_size = bankroll * actual_fraction
        
        if np.random.random() < prob_win:
            bankroll += bet_size * odds
        else:
            bankroll -= bet_size
        
        # Track drawdown
        if bankroll > peak:
            peak = bankroll
        dd = (peak - bankroll) / peak
        if dd > max_dd:
            max_dd = dd
    
    max_drawdowns.append(max_dd)

# Calculate metrics
prob_of_ruin = ruins / num_simulations
avg_time_to_ruin = np.mean(times_to_ruin) if times_to_ruin else float('inf')
avg_max_drawdown = np.mean(max_drawdowns)
percentile_95_dd = np.percentile(max_drawdowns, 95)

# Theoretical calculations
if actual_fraction >= 1:
    theoretical_ruin = 1.0
else:
    # Using gambler's ruin formula approximation
    p = prob_win
    q = 1 - prob_win
    if p != q:
        theoretical_ruin = ((q/p) ** (1/actual_fraction) - 1) / ((q/p) ** (100/actual_fraction) - 1)
    else:
        theoretical_ruin = 1 - 1/100  # Fair game approximation

risk_metrics = {{
    'kelly_fraction_used': kelly_fraction,
    'actual_betting_fraction': actual_fraction,
    'probability_of_ruin': prob_of_ruin,
    'theoretical_ruin_prob': theoretical_ruin,
    'avg_time_to_ruin': avg_time_to_ruin,
    'avg_max_drawdown': avg_max_drawdown,
    'percentile_95_drawdown': percentile_95_dd,
    'risk_assessment': {{
        'very_safe': prob_of_ruin < 0.01,
        'safe': prob_of_ruin < 0.05,
        'moderate': prob_of_ruin < 0.15,
        'risky': prob_of_ruin < 0.30,
        'very_risky': prob_of_ruin >= 0.30
    }}
}}

print(f"Risk of Ruin: {{prob_of_ruin:.2%}}")
print(f"Average Max Drawdown: {{avg_max_drawdown:.2%}}")
risk_metrics
        """
        
        try:
            result = await self.sandbox.run_code(ruin_code)
            if result.error:
                logger.error(f"Risk of ruin error: {result.error}")
                return {"error": str(result.error)}
            
            return result.results[-1].data if result.results else {}
            
        except Exception as e:
            logger.error(f"Risk calculation error: {e}")
            return {"error": str(e)}
    
    async def optimize_kelly_constraints(
        self,
        historical_performance: List[Dict[str, float]],
        target_sharpe: float = 1.5,
        max_drawdown: float = 0.20
    ) -> Dict[str, float]:
        """
        Find optimal Kelly fraction given risk constraints.
        
        Args:
            historical_performance: Past betting performance
            target_sharpe: Minimum Sharpe ratio
            max_drawdown: Maximum acceptable drawdown
        
        Returns:
            Optimal Kelly fraction with constraints
        """
        if not self.sandbox:
            await self.initialize()
        
        optimization_code = f"""
import numpy as np
from scipy.optimize import minimize_scalar
import pandas as pd

historical = {str(historical_performance)}
target_sharpe = {target_sharpe}
max_drawdown = {max_drawdown}

def evaluate_kelly_fraction(fraction, data):
    '''Evaluate performance metrics for given Kelly fraction'''
    
    results = []
    for record in data:
        prob = record.get('probability', 0.5)
        odds = record.get('odds', 1.0)
        outcome = record.get('outcome', 0)
        
        kelly = (prob * odds - (1 - prob)) / odds
        bet_fraction = kelly * fraction
        
        if bet_fraction > 0:
            if outcome == 1:
                ret = bet_fraction * odds
            else:
                ret = -bet_fraction
            results.append(ret)
    
    if not results:
        return 0, float('inf'), 0
    
    # Calculate metrics
    returns = pd.Series(results)
    sharpe = np.sqrt(252) * returns.mean() / (returns.std() + 1e-6)
    
    # Calculate drawdown
    cumsum = (1 + returns).cumprod()
    running_max = cumsum.cummax()
    dd = ((running_max - cumsum) / running_max).max()
    
    return sharpe, dd, returns.mean()

# Grid search for optimal fraction
fractions = np.linspace(0.05, 1.0, 20)
best_fraction = 0.25  # Default
best_score = -float('inf')

for f in fractions:
    sharpe, dd, mean_ret = evaluate_kelly_fraction(f, historical)
    
    # Check constraints
    if sharpe >= target_sharpe and dd <= max_drawdown:
        # Score based on return while meeting constraints
        score = mean_ret
        if score > best_score:
            best_score = score
            best_fraction = f

# Evaluate best fraction
final_sharpe, final_dd, final_return = evaluate_kelly_fraction(best_fraction, historical)

# Also calculate unconstrained optimal
unconstrained_fractions = []
for f in fractions:
    _, _, ret = evaluate_kelly_fraction(f, historical)
    unconstrained_fractions.append((f, ret))

unconstrained_optimal = max(unconstrained_fractions, key=lambda x: x[1])[0]

result = {{
    'constrained_optimal': best_fraction,
    'unconstrained_optimal': unconstrained_optimal,
    'achieved_sharpe': final_sharpe,
    'achieved_drawdown': final_dd,
    'expected_return': final_return,
    'target_sharpe': target_sharpe,
    'max_drawdown_limit': max_drawdown,
    'recommendation': {{
        'conservative': best_fraction * 0.5,
        'moderate': best_fraction * 0.75,
        'aggressive': best_fraction,
        'description': f"Use {{best_fraction:.1%}} Kelly for Sharpe>{{target_sharpe}} and DD<{{max_drawdown:.0%}}"
    }}
}}

print(f"Optimal Kelly with constraints: {{best_fraction:.1%}}")
result
        """
        
        try:
            result = await self.sandbox.run_code(optimization_code)
            if result.error:
                logger.error(f"Optimization error: {result.error}")
                return {"error": str(result.error)}
            
            return result.results[-1].data if result.results else {}
            
        except Exception as e:
            logger.error(f"Kelly optimization error: {e}")
            return {"error": str(e)}
    
    async def cleanup(self):
        """Clean up E2B sandbox."""
        if self.sandbox:
            await self.sandbox.close()
            self.sandbox = None
            logger.info("E2B Kelly sandbox closed")


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        kelly = E2BKellyFunctions()
        await kelly.initialize()
        
        # Monte Carlo simulation
        mc_results = await kelly.monte_carlo_kelly(
            prob_win=0.55,
            odds=1.0,  # Even money bet
            num_simulations=10000,
            num_bets=100
        )
        print(f"Monte Carlo Results: {mc_results}")
        
        # Portfolio optimization
        positions = [
            {"probability": 0.6, "odds": 0.8, "current_allocation": 0},
            {"probability": 0.55, "odds": 1.2, "current_allocation": 0},
            {"probability": 0.65, "odds": 0.6, "current_allocation": 0}
        ]
        
        portfolio = await kelly.portfolio_kelly(positions)
        print(f"Portfolio Optimization: {portfolio}")
        
        # Risk of ruin
        risk = await kelly.calculate_risk_of_ruin(
            prob_win=0.55,
            odds=1.0,
            kelly_fraction=0.25  # Quarter Kelly
        )
        print(f"Risk Metrics: {risk}")
        
        await kelly.cleanup()
    
    asyncio.run(main())