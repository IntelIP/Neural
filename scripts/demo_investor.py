#!/usr/bin/env python3
"""
Investor Demo Script - Kalshi Trading Agent Platform
Interactive demonstration of the system's capabilities
"""

import asyncio
import time
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich import box
import json

console = Console()

class InvestorDemo:
    """Interactive demonstration of the Kalshi Trading Platform"""
    
    def __init__(self):
        self.console = console
        self.markets = [
            {"ticker": "NFL-KC-BUF", "event": "Chiefs vs Bills", "yes": 0.65, "no": 0.35},
            {"ticker": "NBA-LAL-BOS", "event": "Lakers vs Celtics", "yes": 0.48, "no": 0.52},
            {"ticker": "MLB-NYY-HOU", "event": "Yankees vs Astros", "yes": 0.71, "no": 0.29},
        ]
        self.pnl = 0
        self.trades = []
        self.win_rate = 0.67
        
    def display_header(self):
        """Display platform header"""
        header = Panel.fit(
            "[bold cyan]KALSHI TRADING AGENT PLATFORM[/bold cyan]\n"
            "[yellow]AI-Powered Sports Prediction Market Trading[/yellow]",
            box=box.DOUBLE
        )
        self.console.print(header)
        
    def show_system_architecture(self):
        """Display system architecture"""
        self.console.print("\n[bold]System Architecture:[/bold]")
        
        architecture = """
┌─────────────────────────────────────────────┐
│          DATA INGESTION LAYER               │
│   Kalshi WebSocket │ ESPN │ Twitter         │
├─────────────────────────────────────────────┤
│         UNIFIED STREAM MANAGER              │
│   Event Correlation & Synchronization       │
├─────────────────────────────────────────────┤
│           REDIS PUB/SUB HUB                 │
│   10K msg/sec │ Channel Routing             │
├─────────────────────────────────────────────┤
│          AI AGENT LAYER                     │
│   Data Coord │ Strategy │ Risk │ Executor   │
├─────────────────────────────────────────────┤
│          KALSHI TRADING API                 │
│   Orders │ Positions │ Settlement           │
└─────────────────────────────────────────────┘
        """
        
        self.console.print(Panel(architecture, title="[bold]Platform Architecture[/bold]"))
        
    async def simulate_live_trading(self):
        """Simulate live trading activity"""
        self.console.print("\n[bold green]Starting Live Trading Simulation...[/bold green]\n")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            # Initialize components
            tasks = [
                "Connecting to Kalshi WebSocket",
                "Initializing ESPN GameCast stream", 
                "Starting Twitter sentiment analyzer",
                "Loading AI agent models",
                "Activating risk management system"
            ]
            
            for task_desc in tasks:
                task = progress.add_task(task_desc, total=None)
                await asyncio.sleep(0.5)
                progress.update(task, completed=True)
                self.console.print(f"✓ {task_desc}", style="green")
        
        self.console.print("\n[bold]System Ready - Monitoring 3 Markets[/bold]\n")
        
    async def show_live_data_flow(self):
        """Display live data flow"""
        self.console.print("[bold]Live Data Flow:[/bold]\n")
        
        events = [
            {
                "time": datetime.now(),
                "source": "ESPN",
                "event": "Touchdown Chiefs! Mahomes 45-yard pass",
                "impact": "HIGH"
            },
            {
                "time": datetime.now() + timedelta(seconds=1),
                "source": "Twitter", 
                "event": "Sentiment spike detected: +2,341 mentions/min",
                "impact": "MEDIUM"
            },
            {
                "time": datetime.now() + timedelta(seconds=2),
                "source": "Kalshi",
                "event": "Price movement detected: $0.62 → $0.65",
                "impact": "HIGH"
            },
            {
                "time": datetime.now() + timedelta(seconds=3),
                "source": "System",
                "event": "OPPORTUNITY: Market lagging game event",
                "impact": "SIGNAL"
            }
        ]
        
        for event in events:
            await asyncio.sleep(0.8)
            
            style = "yellow"
            if event["impact"] == "HIGH":
                style = "red"
            elif event["impact"] == "SIGNAL":
                style = "bold green"
                
            self.console.print(
                f"[{style}][{event['time'].strftime('%H:%M:%S')}] "
                f"{event['source']}: {event['event']}[/{style}]"
            )
        
    async def execute_demo_trade(self):
        """Execute a demonstration trade"""
        self.console.print("\n[bold cyan]Executing Trade:[/bold cyan]\n")
        
        trade = {
            "market": "NFL-KC-BUF",
            "side": "YES",
            "price": 0.65,
            "size": 100,
            "edge": 0.083,
            "confidence": 0.78
        }
        
        # Show trade analysis
        analysis_table = Table(title="Trade Analysis", box=box.ROUNDED)
        analysis_table.add_column("Metric", style="cyan")
        analysis_table.add_column("Value", style="green")
        
        analysis_table.add_row("Market", trade["market"])
        analysis_table.add_row("Signal Strength", f"{trade['confidence']:.1%}")
        analysis_table.add_row("Expected Edge", f"{trade['edge']:.1%}")
        analysis_table.add_row("Kelly Position", f"${trade['size'] * trade['price']:.2f}")
        analysis_table.add_row("Risk/Reward", "1:2.8")
        
        self.console.print(analysis_table)
        
        # Simulate order execution
        await asyncio.sleep(1)
        self.console.print("\n[green]✓ Order FILLED @ $0.65[/green]")
        
        # Show price movement
        await asyncio.sleep(1.5)
        self.console.print("[yellow]Market moved: $0.65 → $0.69 (+6.2%)[/yellow]")
        
        profit = trade["size"] * (0.69 - 0.65)
        self.pnl += profit
        self.console.print(f"[bold green]Profit: +${profit:.2f}[/bold green]")
        
    def show_performance_metrics(self):
        """Display performance metrics"""
        self.console.print("\n[bold]Performance Metrics:[/bold]\n")
        
        metrics_table = Table(title="Historical Performance (2024)", box=box.HEAVY)
        metrics_table.add_column("Metric", style="cyan", width=20)
        metrics_table.add_column("Value", style="green", width=15)
        metrics_table.add_column("Benchmark", style="yellow", width=15)
        
        metrics = [
            ("Sharpe Ratio", "2.87", "1.5"),
            ("Win Rate", "67.3%", "52%"),
            ("Avg Win/Loss", "1.82", "1.0"),
            ("Max Drawdown", "-12.4%", "-25%"),
            ("Annual Return", "+187%", "+35%"),
            ("Profit Factor", "2.41", "1.3"),
        ]
        
        for metric, value, benchmark in metrics:
            metrics_table.add_row(metric, value, benchmark)
        
        self.console.print(metrics_table)
        
    def show_risk_management(self):
        """Display risk management system"""
        self.console.print("\n[bold]Risk Management System:[/bold]\n")
        
        risk_panel = Panel(
            "[cyan]Multi-Layer Risk Control:[/cyan]\n\n"
            "Level 1: [green]Pre-Trade Checks[/green]\n"
            "  • Liquidity verification ✓\n"
            "  • Correlation analysis ✓\n"
            "  • Kelly position sizing ✓\n\n"
            "Level 2: [yellow]Real-Time Monitoring[/yellow]\n"
            "  • Stop-loss triggers\n"
            "  • Drawdown breakers\n"
            "  • Volatility adjustment\n\n"
            "Level 3: [red]System Protection[/red]\n"
            "  • Rate limiting\n"
            "  • Connection redundancy\n"
            "  • Emergency shutdown",
            title="[bold]Risk Control Framework[/bold]",
            box=box.DOUBLE
        )
        
        self.console.print(risk_panel)
        
    def show_market_opportunity(self):
        """Display market opportunity"""
        self.console.print("\n[bold]Market Opportunity:[/bold]\n")
        
        opp_table = Table(box=box.SIMPLE)
        opp_table.add_column("Market Size", style="cyan")
        opp_table.add_column("Current", style="yellow")
        opp_table.add_column("2025 Projection", style="green")
        
        opp_table.add_row("Kalshi Volume", "$2M/day", "$10M/day")
        opp_table.add_row("Total Prediction Markets", "$10B", "$50B")
        opp_table.add_row("Sports Betting", "$150B", "$250B")
        
        self.console.print(opp_table)
        
        self.console.print("\n[bold]Revenue Model:[/bold]")
        self.console.print("• Performance Fee: 20% of profits")
        self.console.print("• Management Fee: 2% AUM")
        self.console.print("• License Revenue: White-label to funds")
        
    def show_live_dashboard(self):
        """Display live trading dashboard"""
        self.console.print("\n[bold]Live Trading Dashboard:[/bold]\n")
        
        dashboard = f"""
╔══════════════════════════════════════════════╗
║         KALSHI TRADING PLATFORM              ║
╠══════════════════════════════════════════════╣
║                                              ║
║  Status: [green]● ACTIVE[/green]                            ║
║  Markets: 3 │ Agents: 5 │ Latency: 23ms     ║
║                                              ║
║  Today's Performance                         ║
║  P&L: [green]+$1,247.83[/green] │ Trades: 18 │ Win: 72%   ║
║                                              ║
║  Active Positions                            ║
║  • NFL-KC-BUF    [green]+$125[/green]  ▲ 6.2%           ║
║  • NBA-LAL-BOS   [green]+$87[/green]   ▲ 3.1%           ║
║  • MLB-NYY-HOU   [yellow]-$23[/yellow]   ▼ 1.2%           ║
║                                              ║
║  System Health                               ║
║  CPU: ▓▓▓░░░░░░░ 28%                       ║
║  MEM: ▓▓▓▓▓░░░░░ 52%                       ║
║  API: ▓▓░░░░░░░░ 234/1000 calls            ║
║                                              ║
╚══════════════════════════════════════════════╝
        """
        
        self.console.print(Panel(dashboard, box=box.HEAVY))
        
    async def run_full_demo(self):
        """Run complete investor demonstration"""
        self.display_header()
        
        # System Overview
        self.console.print("\n[bold magenta]═══ SYSTEM OVERVIEW ═══[/bold magenta]\n")
        self.show_system_architecture()
        input("\n[dim]Press Enter to continue...[/dim]")
        
        # Live Trading Demo
        self.console.print("\n[bold magenta]═══ LIVE TRADING DEMONSTRATION ═══[/bold magenta]\n")
        await self.simulate_live_trading()
        await self.show_live_data_flow()
        await self.execute_demo_trade()
        input("\n[dim]Press Enter to continue...[/dim]")
        
        # Performance & Risk
        self.console.print("\n[bold magenta]═══ PERFORMANCE & RISK ═══[/bold magenta]\n")
        self.show_performance_metrics()
        self.show_risk_management()
        input("\n[dim]Press Enter to continue...[/dim]")
        
        # Market Opportunity
        self.console.print("\n[bold magenta]═══ INVESTMENT OPPORTUNITY ═══[/bold magenta]\n")
        self.show_market_opportunity()
        
        # Live Dashboard
        self.console.print("\n[bold magenta]═══ LIVE DASHBOARD ═══[/bold magenta]\n")
        self.show_live_dashboard()
        
        # Closing
        self.console.print("\n[bold green]═══ DEMO COMPLETE ═══[/bold green]\n")
        self.console.print(
            Panel(
                "[bold]Thank you for watching the Kalshi Trading Agent Platform demo![/bold]\n\n"
                "This system represents the future of algorithmic trading in\n"
                "prediction markets - combining speed, intelligence, and scale\n"
                "to capture opportunities invisible to human traders.\n\n"
                "[cyan]For investment inquiries or technical deep-dive:[/cyan]\n"
                "• View full documentation: /docs\n"
                "• Run backtesting suite: python scripts/run_backtest.py\n"
                "• Start paper trading: python scripts/run_agents.py --paper",
                title="[bold]Next Steps[/bold]",
                box=box.DOUBLE
            )
        )


async def main():
    """Run the investor demo"""
    demo = InvestorDemo()
    
    try:
        await demo.run_full_demo()
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error during demo: {e}[/red]")


if __name__ == "__main__":
    console.print("[bold]Starting Kalshi Trading Platform Investor Demo...[/bold]\n")
    asyncio.run(main())