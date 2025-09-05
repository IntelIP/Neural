"""
Command-line interface for the Neural Trading SDK.

This module provides a CLI for managing the SDK, running trading systems,
and performing administrative tasks.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .core.client import NeuralSDK
from .core.config import SDKConfig
from .core.exceptions import ConfigurationError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create CLI app
app = typer.Typer(
    name="neural-sdk",
    help="Neural Trading SDK - Algorithmic sports trading platform",
    add_completion=False,
)

console = Console()


@app.callback()
def callback():
    """Neural Trading SDK CLI"""
    pass


@app.command()
def init(
    config_file: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
    environment: str = typer.Option(
        "development",
        "--env",
        "-e",
        help="Environment (development/staging/production)",
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Overwrite existing configuration"
    ),
):
    """
    Initialize SDK configuration.

    Creates a default configuration file that can be customized for your needs.
    """
    try:
        if config_file:
            config_path = Path(config_file)
        else:
            config_path = Path("neural_config.yaml")

        if config_path.exists() and not force:
            console.print(
                f"[red]Configuration file already exists: {config_path}[/red]"
            )
            console.print("[yellow]Use --force to overwrite[/yellow]")
            return

        # Create default configuration
        config = SDKConfig(
            environment=environment, enable_debug_mode=environment == "development"
        )

        # Save configuration
        config.save_to_file(str(config_path))

        console.print(f"[green]✅ Configuration created: {config_path}[/green]")
        console.print("\n[bold]Next steps:[/bold]")
        console.print("1. Edit the configuration file with your API keys")
        console.print("2. Run 'neural-sdk start' to begin trading")

    except Exception as e:
        console.print(f"[red]❌ Failed to initialize configuration: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def start(
    config_file: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
    env_prefix: str = typer.Option(
        "KALSHI_", "--env-prefix", "-p", help="Environment variable prefix"
    ),
):
    """
    Start the trading system.

    Initializes all components and begins automated trading based on your configuration.
    """
    try:
        # Load configuration
        if config_file:
            config = SDKConfig.from_file(config_file)
        else:
            config = SDKConfig.from_env(env_prefix)

        console.print(
            f"[blue]🚀 Starting Neural Trading SDK ({config.environment})[/blue]"
        )

        # Create SDK instance
        sdk = NeuralSDK(config)

        # Display configuration summary
        console.print(
            Panel.fit(
                config.get_summary(),
                title="📊 Configuration Summary",
                border_style="blue",
            )
        )

        # Start trading system
        asyncio.run(_run_trading_system(sdk))

    except ConfigurationError as e:
        console.print(f"[red]❌ Configuration error: {e}[/red]")
        console.print(
            "[yellow]Run 'neural-sdk init' to create a configuration file[/yellow]"
        )
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("[yellow]⚠️  Trading system interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]❌ Failed to start trading system: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def status(
    config_file: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
    env_prefix: str = typer.Option(
        "KALSHI_", "--env-prefix", "-p", help="Environment variable prefix"
    ),
    watch: bool = typer.Option(
        False, "--watch", "-w", help="Continuously monitor status"
    ),
):
    """
    Show system status and health information.
    """
    try:
        # Load configuration
        if config_file:
            config = SDKConfig.from_file(config_file)
        else:
            config = SDKConfig.from_env(env_prefix)

        sdk = NeuralSDK(config)

        if watch:
            console.print("[green]📊 Monitoring system status (Ctrl+C to stop)[/green]")
            asyncio.run(_monitor_status(sdk))
        else:
            asyncio.run(_show_status(sdk))

    except Exception as e:
        console.print(f"[red]❌ Failed to get status: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def validate(
    config_file: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
    env_prefix: str = typer.Option(
        "KALSHI_", "--env-prefix", "-p", help="Environment variable prefix"
    ),
):
    """
    Validate configuration and check system health.
    """
    try:
        # Load configuration
        if config_file:
            config = SDKConfig.from_file(config_file)
        else:
            config = SDKConfig.from_env(env_prefix)

        console.print("[blue]🔍 Validating configuration...[/blue]")

        # Create SDK instance (this will validate config)
        sdk = NeuralSDK(config)

        # Perform health check
        health = asyncio.run(sdk.health_check())

        # Display results
        if health["overall"] == "healthy":
            console.print(
                "[green]✅ Configuration is valid and system is healthy[/green]"
            )
        else:
            console.print(f"[red]❌ System health: {health['overall']}[/red]")

        # Show component status
        table = Table(title="Component Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")

        for component, status in health["components"].items():
            status_color = {
                "healthy": "green",
                "degraded": "yellow",
                "unhealthy": "red",
            }.get(status, "white")
            table.add_row(component, f"[{status_color}]{status}[/{status_color}]")

        console.print(table)

    except Exception as e:
        console.print(f"[red]❌ Validation failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def config(
    action: str = typer.Argument(..., help="Action: show, edit, or validate"),
    config_file: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
    env_prefix: str = typer.Option(
        "KALSHI_", "--env-prefix", "-p", help="Environment variable prefix"
    ),
):
    """
    Manage SDK configuration.

    Actions:
    - show: Display current configuration
    - edit: Open configuration file in editor
    - validate: Check configuration validity
    """
    try:
        if action == "show":
            if config_file:
                config = SDKConfig.from_file(config_file)
            else:
                config = SDKConfig.from_env(env_prefix)

            console.print(
                Panel.fit(
                    config.get_summary(),
                    title="📋 Current Configuration",
                    border_style="blue",
                )
            )

        elif action == "edit":
            if not config_file:
                config_file = "neural_config.yaml"

            config_path = Path(config_file)
            if not config_path.exists():
                console.print(f"[red]Configuration file not found: {config_path}[/red]")
                console.print("[yellow]Run 'neural-sdk init' to create one[/yellow]")
                return

            # Try to open in editor
            import os
            import subprocess

            editor = os.getenv("EDITOR", "nano")
            try:
                subprocess.run([editor, str(config_path)])
                console.print(f"[green]✅ Configuration edited: {config_path}[/green]")
            except Exception as e:
                console.print(f"[red]❌ Failed to open editor: {e}[/red]")
                console.print(f"Edit the file manually: {config_path}")

        elif action == "validate":
            # Reuse the validate command logic
            validate(config_file, env_prefix)

        else:
            console.print(f"[red]❌ Unknown action: {action}[/red]")
            console.print("Available actions: show, edit, validate")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]❌ Config command failed: {e}[/red]")
        raise typer.Exit(1)


async def _run_trading_system(sdk: NeuralSDK):
    """Run the trading system with proper error handling."""
    try:
        await sdk.start_trading_system()
    except KeyboardInterrupt:
        console.print("[yellow]⚠️  Received shutdown signal[/yellow]")
    finally:
        await sdk.stop_trading_system()
        console.print("[green]✅ Trading system shut down gracefully[/green]")


async def _show_status(sdk: NeuralSDK):
    """Show current system status."""
    try:
        # Get system status
        status = sdk.get_system_status()
        health = await sdk.health_check()

        # Display status
        console.print(
            Panel.fit(
                f"""
Environment: {status['environment']}
Running: {'✅ Yes' if status['is_running'] else '❌ No'}
Data Sources: {status['data_sources_connected']}
Active Agents: {status['agents_active']}
Loaded Strategies: {status['strategies_loaded']}
System Health: {health['overall']}
Last Update: {status['last_update']}
            """.strip(),
                title="📊 System Status",
                border_style="blue",
            )
        )

    except Exception as e:
        console.print(f"[red]❌ Failed to get status: {e}[/red]")


async def _monitor_status(sdk: NeuralSDK):
    """Continuously monitor system status."""

    try:
        while True:
            console.clear()
            await _show_status(sdk)
            console.print(
                "\n[yellow]Refreshing in 5 seconds... (Ctrl+C to stop)[/yellow]"
            )
            await asyncio.sleep(5)
    except KeyboardInterrupt:
        console.print("[green]✅ Monitoring stopped[/green]")


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
