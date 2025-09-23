"""
Comprehensive Risk Management Integration Demo

This example demonstrates the complete Neural SDK risk management system,
showing how to:

1. Configure position sizing with Kelly Criterion
2. Set up portfolio optimization with correlation analysis
3. Enforce risk limits and constraints
4. Monitor real-time risk metrics and alerts
5. Integrate risk management with strategy execution
6. Handle risk-based position adjustments

The demo shows enterprise-grade risk management that ensures
trading strategies operate within safe parameters while
optimizing risk-adjusted returns.
"""

import asyncio
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Neural SDK imports
from neural.risk import (
    PositionSizer, PositionSizingMethod,
    PortfolioOptimizer, AllocationMethod, 
    RiskLimitManager, PositionLimit, LossLimit, ExposureLimit,
    RiskMonitor, AlertSeverity
)
from neural.strategy.library.mean_reversion import BasicMeanReversionStrategy
from neural.strategy.library.volume_anomaly import VolumeAnomalyStrategy
from neural.strategy.builder import StrategyComposer
from neural.strategy.base import Signal, SignalType, SignalStrength
from neural.backtesting import BacktestEngine, BacktestConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskManagedTradingSystem:
    """
    Complete risk-managed trading system integrating all components.
    
    This system demonstrates how to build a production-ready trading
    system with comprehensive risk management and monitoring.
    """
    
    def __init__(self, initial_capital: float = 50000.0):
        """Initialize the risk-managed trading system."""
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # Initialize risk management components
        self.position_sizer = PositionSizer(
            primary_method=PositionSizingMethod.KELLY_CRITERION,
            fallback_method=PositionSizingMethod.FIXED_PERCENTAGE
        )
        
        self.portfolio_optimizer = PortfolioOptimizer(
            max_weight=0.15,  # Max 15% per position
            max_concentration=0.45  # Max 45% in top 3
        )
        
        self.risk_limit_manager = RiskLimitManager(initial_capital)
        self.risk_monitor = RiskMonitor(initial_capital, self.risk_limit_manager)
        
        # Portfolio state
        self.positions = {}
        self.signals_history = []
        
        # Performance tracking
        self.trades = []
        self.daily_pnls = []
        
        logger.info(f"🚀 Initialized Risk-Managed Trading System with ${initial_capital:,.2f}")
    
    async def run_comprehensive_demo(self):
        """Run the complete risk management demo."""
        logger.info("=" * 70)
        logger.info("🛡️  NEURAL SDK RISK MANAGEMENT FRAMEWORK DEMO")
        logger.info("=" * 70)
        
        # 1. Setup Risk Configuration
        logger.info("\n1️⃣  RISK CONFIGURATION SETUP")
        await self._setup_risk_configuration()
        
        # 2. Position Sizing Demo
        logger.info("\n2️⃣  POSITION SIZING ALGORITHMS")
        await self._demo_position_sizing()
        
        # 3. Portfolio Optimization Demo
        logger.info("\n3️⃣  PORTFOLIO OPTIMIZATION")
        await self._demo_portfolio_optimization()
        
        # 4. Risk Limits Enforcement
        logger.info("\n4️⃣  RISK LIMITS ENFORCEMENT")
        await self._demo_risk_limits()
        
        # 5. Real-time Risk Monitoring
        logger.info("\n5️⃣  REAL-TIME RISK MONITORING")
        await self._demo_risk_monitoring()
        
        # 6. Integrated Trading Simulation
        logger.info("\n6️⃣  INTEGRATED TRADING SIMULATION")
        await self._demo_integrated_trading()
        
        # 7. Risk Reports and Analytics
        logger.info("\n7️⃣  RISK REPORTS & ANALYTICS")
        await self._demo_risk_reports()
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ RISK MANAGEMENT DEMO COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)
    
    async def _setup_risk_configuration(self):
        """Demonstrate risk configuration setup."""
        logger.info("🔧 Configuring risk management parameters...")
        
        # Add custom risk limits
        self.risk_limit_manager.add_limit(PositionLimit(
            limit_id="large_position_limit",
            max_position_pct=0.12,  # 12% max per position
        ))
        
        self.risk_limit_manager.add_limit(ExposureLimit(
            limit_id="sport_exposure_limit", 
            category="sport",
            max_exposure_pct=0.40  # 40% max per sport
        ))
        
        self.risk_limit_manager.add_limit(LossLimit(
            limit_id="weekly_loss_limit",
            max_loss_pct=0.08,  # 8% weekly loss limit
            time_period="weekly"
        ))
        
        # Add alert callback
        def alert_handler(alert):
            severity_emoji = {
                AlertSeverity.INFO: "ℹ️",
                AlertSeverity.WARNING: "⚠️", 
                AlertSeverity.CRITICAL: "🚨",
                AlertSeverity.EMERGENCY: "🆘"
            }
            
            emoji = severity_emoji.get(alert.severity, "❓")
            logger.info(f"{emoji} ALERT: {alert.message}")
        
        self.risk_monitor.add_alert_callback(alert_handler)
        
        logger.info("✅ Risk configuration completed")
        logger.info(f"   • {len(self.risk_limit_manager.limits)} risk limits configured")
        logger.info("   • Alert monitoring enabled")
    
    async def _demo_position_sizing(self):
        """Demonstrate position sizing algorithms."""
        logger.info("📊 Testing position sizing algorithms...")
        
        # Create sample signals with varying characteristics
        signals = [
            Signal(
                signal_type=SignalType.BUY_YES,
                market_id="NFL_CHIEFS_WIN",
                timestamp=datetime.now(),
                confidence=0.75,
                edge=0.08,  # 8% edge
                signal_strength=SignalStrength.STRONG,
                market_price=0.45,
                recommended_size=0.10
            ),
            Signal(
                signal_type=SignalType.BUY_NO,
                market_id="NBA_LAKERS_WIN", 
                timestamp=datetime.now(),
                confidence=0.60,
                edge=0.03,  # 3% edge
                signal_strength=SignalStrength.WEAK,
                market_price=0.65,
                recommended_size=0.05
            ),
            Signal(
                signal_type=SignalType.BUY_YES,
                market_id="ELECTION_2024",
                timestamp=datetime.now(),
                confidence=0.90,
                edge=0.12,  # 12% edge
                signal_strength=SignalStrength.VERY_STRONG,
                market_price=0.38,
                recommended_size=0.15
            )
        ]
        
        logger.info("🎯 Position Sizing Results:")
        total_allocation = 0.0
        
        for signal in signals:
            result = self.position_sizer.calculate_position_size(
                signal=signal,
                current_capital=self.current_capital,
                current_positions=self.positions
            )
            
            allocation_pct = result.recommended_size / self.current_capital * 100
            total_allocation += allocation_pct
            
            logger.info(f"   {signal.market_id}:")
            logger.info(f"     Method: {result.method.value}")
            logger.info(f"     Size: ${result.recommended_size:,.2f} ({allocation_pct:.1f}%)")
            logger.info(f"     Contracts: {result.recommended_contracts}")
            logger.info(f"     Rationale: {result.rationale}")
            
            if result.warnings:
                for warning in result.warnings:
                    logger.info(f"     ⚠️  {warning}")
        
        logger.info(f"📈 Total Allocation: {total_allocation:.1f}% of capital")
    
    async def _demo_portfolio_optimization(self):
        """Demonstrate portfolio optimization with correlation analysis."""
        logger.info("🎼 Testing portfolio optimization...")
        
        # Generate synthetic returns data for correlation analysis
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        markets = ["NFL_CHIEFS", "NBA_LAKERS", "MLB_DODGERS", "ELECTION_2024", "CRYPTO_BTC"]
        
        # Create correlated returns (some markets more correlated than others)
        np.random.seed(42)
        base_returns = np.random.normal(0.001, 0.02, (100, 5))
        
        # Add correlation structure
        correlation_factor = np.array([[1.0, 0.6, 0.4, 0.1, 0.0],
                                     [0.6, 1.0, 0.5, 0.1, 0.1],
                                     [0.4, 0.5, 1.0, 0.2, 0.1],
                                     [0.1, 0.1, 0.2, 1.0, 0.3],
                                     [0.0, 0.1, 0.1, 0.3, 1.0]])
        
        correlated_returns = base_returns @ np.linalg.cholesky(correlation_factor).T
        returns_df = pd.DataFrame(correlated_returns, index=dates, columns=markets)
        
        # Create signals for optimization
        signals = []
        for i, market in enumerate(markets):
            signal = Signal(
                signal_type=SignalType.BUY_YES,
                market_id=market,
                timestamp=datetime.now(),
                confidence=0.6 + i * 0.05,  # Varying confidence
                edge=0.02 + i * 0.01,
                signal_strength=SignalStrength.MEDIUM
            )
            signals.append(signal)
        
        # Test different allocation methods
        methods = [
            AllocationMethod.EQUAL_WEIGHT,
            AllocationMethod.SIGNAL_WEIGHTED,
            AllocationMethod.RISK_PARITY
        ]
        
        logger.info("🔍 Portfolio Allocation Comparison:")
        
        for method in methods:
            allocation = self.portfolio_optimizer.optimize_portfolio(
                method=method,
                signals=signals,
                returns_data=returns_df
            )
            
            logger.info(f"   {method.value.upper()}:")
            logger.info(f"     Success: {'✅' if allocation.optimization_success else '❌'}")
            logger.info(f"     Expected Vol: {allocation.expected_volatility:.3f}")
            logger.info(f"     Concentration Risk: {allocation.concentration_risk:.3f}")
            logger.info("     Weights:")
            
            for asset, weight in allocation.weights.items():
                if weight > 0.001:  # Only show significant weights
                    logger.info(f"       {asset}: {weight:.1%}")
            
            if allocation.warnings:
                for warning in allocation.warnings:
                    logger.info(f"     ⚠️  {warning}")
    
    async def _demo_risk_limits(self):
        """Demonstrate risk limits enforcement."""
        logger.info("🚦 Testing risk limits enforcement...")
        
        # Simulate a large portfolio to test limits
        large_portfolio = {
            'total_capital': self.current_capital,
            'positions': {
                'BIG_POSITION': {'market_value': self.current_capital * 0.18},  # 18% - exceeds limit
                'MEDIUM_POSITION': {'market_value': self.current_capital * 0.08},
                'SMALL_POSITION': {'market_value': self.current_capital * 0.04}
            },
            'daily_pnl': -self.current_capital * 0.02  # -2% daily loss
        }
        
        # Test limit checking
        limits_ok, violations = self.risk_limit_manager.check_all_limits(large_portfolio)
        
        logger.info(f"📊 Risk Limits Check:")
        logger.info(f"   All Limits OK: {'✅' if limits_ok else '❌'}")
        logger.info(f"   Violations Found: {len(violations)}")
        
        for violation in violations:
            logger.info(f"   🚨 VIOLATION: {violation.message}")
            logger.info(f"      Current: {violation.current_value:.1%}")
            logger.info(f"      Limit: {violation.limit_value:.1%}")
            logger.info(f"      Action: {violation.action_taken.value}")
        
        # Test trade approval
        large_signal = Signal(
            signal_type=SignalType.BUY_YES,
            market_id="HUGE_BET",
            timestamp=datetime.now(),
            confidence=0.80,
            edge=0.05,
            recommended_contracts=int(self.current_capital * 0.25 / 100)  # Would be 25% position
        )
        
        trade_allowed, rejection_reasons = self.risk_limit_manager.check_trade_allowed(
            signal=large_signal,
            current_portfolio=large_portfolio
        )
        
        logger.info(f"📋 Trade Approval Test:")
        logger.info(f"   Trade Allowed: {'✅' if trade_allowed else '❌'}")
        if rejection_reasons:
            for reason in rejection_reasons:
                logger.info(f"   ❌ Rejection: {reason}")
    
    async def _demo_risk_monitoring(self):
        """Demonstrate real-time risk monitoring."""
        logger.info("📡 Testing real-time risk monitoring...")
        
        # Create a portfolio for monitoring
        monitoring_portfolio = {
            'total_capital': self.current_capital,
            'positions': {
                'POSITION_1': {'market_value': self.current_capital * 0.12},
                'POSITION_2': {'market_value': self.current_capital * 0.08},
                'POSITION_3': {'market_value': self.current_capital * 0.06}
            },
            'daily_pnl': self.current_capital * 0.01,  # +1% today
            'peak_capital': self.current_capital * 1.05  # 5% below peak
        }
        
        # Simulate monitoring update
        risk_metrics = await self.risk_monitor._calculate_comprehensive_metrics(monitoring_portfolio)
        
        logger.info("📊 Current Risk Metrics:")
        logger.info(f"   Total Capital: ${risk_metrics.total_capital:,.2f}")
        logger.info(f"   Total Exposure: ${risk_metrics.total_exposure:,.2f}")
        logger.info(f"   Daily P&L: ${risk_metrics.daily_pnl:,.2f}")
        logger.info(f"   Largest Position: {risk_metrics.largest_position_pct:.1%}")
        logger.info(f"   Top 3 Concentration: {risk_metrics.top_3_concentration:.1%}")
        logger.info(f"   Active Positions: {risk_metrics.num_positions}")
        
        # Test alert generation
        alerts = await self.risk_monitor._check_all_risk_conditions(monitoring_portfolio, risk_metrics)
        
        if alerts:
            logger.info(f"🚨 Generated {len(alerts)} alerts:")
            for alert in alerts:
                logger.info(f"   {alert.severity.value.upper()}: {alert.message}")
        else:
            logger.info("✅ No risk alerts generated")
        
        # Get dashboard data
        dashboard = self.risk_monitor.get_risk_dashboard_data()
        
        if 'error' not in dashboard:
            logger.info("📊 Risk Dashboard Summary:")
            logger.info(f"   System Health: {dashboard['system']['system_health']:.1%}")
            logger.info(f"   VaR (95%): {dashboard['risk_metrics']['var_95']:.2%}")
            logger.info(f"   Active Alerts: {dashboard['alerts']['total_active']}")
    
    async def _demo_integrated_trading(self):
        """Demonstrate integrated trading with risk management."""
        logger.info("🎯 Simulating integrated risk-managed trading...")
        
        # Create strategies
        mean_reversion = BasicMeanReversionStrategy(
            lookback_period=24,
            threshold_std=2.0,
            position_size=0.08
        )
        
        volume_anomaly = VolumeAnomalyStrategy(
            lookback_period=36,
            volume_threshold=2.5,
            position_size=0.06
        )
        
        # Simulate market conditions
        markets = ["NFL_GAME_1", "NBA_GAME_1", "MLB_GAME_1"]
        
        logger.info("🔄 Processing trading signals with risk management...")
        
        total_allocated = 0.0
        approved_trades = 0
        rejected_trades = 0
        
        for i, market_id in enumerate(markets):
            # Generate mock signal
            signal = Signal(
                signal_type=SignalType.BUY_YES,
                market_id=market_id,
                timestamp=datetime.now(),
                confidence=0.65 + i * 0.1,
                edge=0.04 + i * 0.02,
                signal_strength=SignalStrength.MEDIUM,
                market_price=0.45 + i * 0.05
            )
            
            # Size the position
            sizing_result = self.position_sizer.calculate_position_size(
                signal=signal,
                current_capital=self.current_capital,
                current_positions=self.positions
            )
            
            # Check if trade is allowed
            trade_allowed, rejection_reasons = self.risk_limit_manager.check_trade_allowed(
                signal=signal,
                current_portfolio={
                    'total_capital': self.current_capital,
                    'positions': self.positions
                }
            )
            
            logger.info(f"   📈 {market_id}:")
            logger.info(f"      Signal: {signal.signal_type.value} @ {signal.market_price:.2f}")
            logger.info(f"      Confidence: {signal.confidence:.1%}")
            logger.info(f"      Edge: {signal.edge:.1%}")
            logger.info(f"      Recommended Size: ${sizing_result.recommended_size:,.2f}")
            
            if trade_allowed:
                logger.info(f"      Status: ✅ APPROVED")
                total_allocated += sizing_result.recommended_size
                approved_trades += 1
                
                # Add to positions (simplified)
                self.positions[market_id] = {
                    'market_value': sizing_result.recommended_size,
                    'entry_price': signal.market_price,
                    'contracts': sizing_result.recommended_contracts
                }
            else:
                logger.info(f"      Status: ❌ REJECTED")
                rejected_trades += 1
                for reason in rejection_reasons:
                    logger.info(f"        Reason: {reason}")
        
        allocation_pct = total_allocated / self.current_capital * 100
        
        logger.info(f"📊 Trading Session Summary:")
        logger.info(f"   Approved Trades: {approved_trades}")
        logger.info(f"   Rejected Trades: {rejected_trades}")
        logger.info(f"   Total Allocated: ${total_allocated:,.2f} ({allocation_pct:.1f}%)")
        logger.info(f"   Cash Remaining: ${self.current_capital - total_allocated:,.2f}")
    
    async def _demo_risk_reports(self):
        """Demonstrate comprehensive risk reporting."""
        logger.info("📄 Generating comprehensive risk reports...")
        
        # Generate risk limits report
        limits_report = self.risk_limit_manager.generate_risk_report()
        logger.info("\n📋 RISK LIMITS REPORT:")
        logger.info(limits_report)
        
        # Generate risk monitoring report
        monitoring_report = self.risk_monitor.generate_risk_report()
        logger.info("\n📊 RISK MONITORING REPORT:")
        logger.info(monitoring_report)
        
        # Generate position sizing statistics
        sample_signals = [
            Signal(
                signal_type=SignalType.BUY_YES,
                market_id=f"SAMPLE_{i}",
                timestamp=datetime.now(),
                confidence=0.5 + i * 0.1,
                edge=0.02 + i * 0.01
            ) for i in range(5)
        ]
        
        sizing_stats = self.position_sizer.get_sizing_statistics(
            signals=sample_signals,
            current_capital=self.current_capital
        )
        
        logger.info("\n📈 POSITION SIZING STATISTICS:")
        for metric, value in sizing_stats.items():
            if isinstance(value, float):
                if 'percentage' in metric or 'ratio' in metric:
                    logger.info(f"   {metric.replace('_', ' ').title()}: {value:.1%}")
                else:
                    logger.info(f"   {metric.replace('_', ' ').title()}: ${value:,.2f}")
            else:
                logger.info(f"   {metric.replace('_', ' ').title()}: {value}")


# Standalone demo functions
async def run_risk_management_demo():
    """Run the comprehensive risk management demo."""
    system = RiskManagedTradingSystem(initial_capital=100000.0)
    await system.run_comprehensive_demo()


async def run_position_sizing_demo():
    """Run focused position sizing demo."""
    logger.info("🎯 Position Sizing Algorithm Demo")
    logger.info("=" * 50)
    
    position_sizer = PositionSizer(
        primary_method=PositionSizingMethod.KELLY_CRITERION
    )
    
    # Test different signal scenarios
    scenarios = [
        ("High Confidence, High Edge", 0.85, 0.10, 0.42),
        ("Medium Confidence, Medium Edge", 0.65, 0.05, 0.55),
        ("Low Confidence, Small Edge", 0.45, 0.02, 0.48),
        ("High Confidence, No Edge", 0.90, 0.00, 0.50),
        ("Medium Confidence, Negative Edge", 0.70, -0.03, 0.62)
    ]
    
    capital = 50000.0
    
    for scenario_name, confidence, edge, price in scenarios:
        signal = Signal(
            signal_type=SignalType.BUY_YES,
            market_id="TEST_MARKET",
            timestamp=datetime.now(),
            confidence=confidence,
            edge=edge,
            market_price=price,
            signal_strength=SignalStrength.MEDIUM
        )
        
        result = position_sizer.calculate_position_size(signal, capital, {})
        
        logger.info(f"\n📊 {scenario_name}:")
        logger.info(f"   Signal: Confidence={confidence:.0%}, Edge={edge:.1%}, Price=${price:.2f}")
        logger.info(f"   Method: {result.method.value}")
        logger.info(f"   Recommended Size: ${result.recommended_size:.2f}")
        logger.info(f"   Contracts: {result.recommended_contracts}")
        logger.info(f"   Risk %: {result.risk_percentage:.1%}")
        logger.info(f"   Rationale: {result.rationale}")


if __name__ == "__main__":
    print("🛡️ Neural SDK Risk Management Framework Demo")
    print("Choose demo type:")
    print("1. Complete Risk Management Demo")
    print("2. Position Sizing Focus Demo")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        asyncio.run(run_position_sizing_demo())
    else:
        asyncio.run(run_risk_management_demo())
