"""
Environment UI Components

Visual interface components for environment management with
clear distinctions between training/sandbox and production modes.
"""

import asyncio
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from colorama import Back, Fore
from colorama import Style as ColoramaStyle
from colorama import init
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.style import Style
from rich.table import Table
from rich.text import Text

from .environment_manager import Environment, EnvironmentManager, EnvironmentStatus

# Initialize colorama
init(autoreset=True)


class UITheme:
    """UI themes for different environments."""

    THEMES = {
        Environment.DEVELOPMENT: {
            "primary": "cyan",
            "secondary": "blue",
            "accent": "white",
            "background": "black",
            "warning": "yellow",
            "error": "red",
            "success": "green",
            "border_style": "cyan",
            "title_style": "bold cyan",
            "icon": "🔧",
        },
        Environment.TRAINING: {
            "primary": "yellow",
            "secondary": "orange",
            "accent": "white",
            "background": "black",
            "warning": "magenta",
            "error": "red",
            "success": "green",
            "border_style": "yellow",
            "title_style": "bold yellow",
            "icon": "🎓",
        },
        Environment.SANDBOX: {
            "primary": "green",
            "secondary": "lime",
            "accent": "white",
            "background": "black",
            "warning": "yellow",
            "error": "red",
            "success": "bright_green",
            "border_style": "green",
            "title_style": "bold green",
            "icon": "📦",
        },
        Environment.STAGING: {
            "primary": "magenta",
            "secondary": "purple",
            "accent": "white",
            "background": "black",
            "warning": "yellow",
            "error": "red",
            "success": "green",
            "border_style": "magenta",
            "title_style": "bold magenta",
            "icon": "🚦",
        },
        Environment.PRODUCTION: {
            "primary": "red",
            "secondary": "bright_red",
            "accent": "white",
            "background": "black",
            "warning": "bright_yellow",
            "error": "bright_red",
            "success": "green",
            "border_style": "bold red",
            "title_style": "bold white on red",
            "icon": "🔥",
        },
    }

    @classmethod
    def get_theme(cls, environment: Environment) -> Dict[str, Any]:
        """Get theme for environment."""
        return cls.THEMES.get(environment, cls.THEMES[Environment.DEVELOPMENT])


class EnvironmentUI:
    """Main UI class for environment management."""

    def __init__(self, manager: EnvironmentManager):
        """Initialize environment UI."""
        self.manager = manager
        self.console = Console()
        self.current_theme = UITheme.get_theme(manager.get_current_environment())

    def display_banner(self) -> None:
        """Display environment banner with visual indicators."""
        env = self.manager.get_current_environment()
        theme = UITheme.get_theme(env)
        status = self.manager.get_status()

        # Create banner content
        banner_text = Text()
        banner_text.append(f"\n{theme['icon']} ", style=theme["primary"])
        banner_text.append(f"KALSHI TRADING AGENT", style="bold white")
        banner_text.append(f" - ", style="white")
        banner_text.append(f"{env.value.upper()}", style=f"bold {theme['primary']}")
        banner_text.append(f" {theme['icon']}\n", style=theme["primary"])

        # Add status information
        status_table = Table(show_header=False, box=None, padding=0)
        status_table.add_column("Key", style="dim")
        status_table.add_column("Value", style=theme["accent"])

        status_table.add_row("Session ID:", status.session_id)
        status_table.add_row(
            "Started:", status.session_start.strftime("%Y-%m-%d %H:%M:%S")
        )
        status_table.add_row("Duration:", str(status.session_duration()).split(".")[0])
        status_table.add_row("Operations:", str(status.operations_count))
        status_table.add_row("Redis DB:", str(status.config.redis_db))
        status_table.add_row(
            "Safety Checks:", "✓" if status.config.safety_checks else "✗"
        )
        status_table.add_row("MFA Required:", "✓" if status.config.require_mfa else "✗")

        # Create panel
        panel = Panel(
            status_table,
            title=banner_text,
            border_style=theme["border_style"],
            padding=(1, 2),
        )

        self.console.print(panel)

        # Show production warning if needed
        if env == Environment.PRODUCTION:
            self._display_production_warning()

    def _display_production_warning(self) -> None:
        """Display prominent production warning."""
        warning = Panel(
            Text.from_markup(
                "[bold white on red] ⚠️  PRODUCTION ENVIRONMENT ⚠️  [/]\n\n"
                "[bold yellow]You are operating with REAL MONEY[/]\n"
                "All trades will be executed on live markets\n\n"
                "[dim]Ensure you have:[/]\n"
                "  • Verified all risk parameters\n"
                "  • Tested changes in sandbox\n"
                "  • Proper authorization\n"
                "  • Emergency procedures ready"
            ),
            title="[bold red]CRITICAL WARNING[/]",
            border_style="bold red",
            padding=(1, 2),
        )
        self.console.print(warning)

    def display_status(self) -> None:
        """Display current environment status."""
        status = self.manager.get_status()
        theme = self.current_theme

        # Create status table
        table = Table(
            title=f"{theme['icon']} Environment Status",
            title_style=theme["title_style"],
            border_style=theme["border_style"],
        )

        table.add_column("Property", style="cyan", no_wrap=True)
        table.add_column("Value", style="white")

        # Add rows
        table.add_row("Environment", status.environment.value.upper())
        table.add_row("Session ID", status.session_id)
        table.add_row("Authenticated", "✓" if status.is_authenticated else "✗")
        table.add_row("Session Duration", str(status.session_duration()).split(".")[0])
        table.add_row("Operations Count", str(status.operations_count))

        if status.last_operation:
            table.add_row("Last Operation", status.last_operation.strftime("%H:%M:%S"))

        if status.warnings:
            table.add_row("Warnings", str(len(status.warnings)))

        table.add_row("Safe Mode", "✓" if status.is_safe_mode() else "✗")

        self.console.print(table)

    def display_config(self) -> None:
        """Display environment configuration."""
        config = self.manager.get_config()
        theme = self.current_theme

        # Create config panels
        layout = Layout()
        layout.split_column(
            Layout(name="limits"), Layout(name="features"), Layout(name="api")
        )

        # Trading limits table
        limits_table = Table(title="Trading Limits", border_style=theme["border_style"])
        limits_table.add_column("Limit", style="cyan")
        limits_table.add_column("Value", style="white")

        limits_table.add_row("Max Position Size", f"{config.max_position_size:.1%}")
        limits_table.add_row("Max Daily Trades", str(config.max_daily_trades))
        limits_table.add_row("Data Retention", f"{config.data_retention_hours}h")

        # Features table
        features_table = Table(title="Features", border_style=theme["border_style"])
        features_table.add_column("Feature", style="cyan")
        features_table.add_column("Status", style="white")

        for feature, enabled in config.features.items():
            status = "✓" if enabled else "✗"
            color = "green" if enabled else "red"
            features_table.add_row(
                feature.replace("_", " ").title(), f"[{color}]{status}[/]"
            )

        # API endpoints table
        api_table = Table(title="API Endpoints", border_style=theme["border_style"])
        api_table.add_column("Service", style="cyan")
        api_table.add_column("Endpoint", style="white")

        for service, endpoint in config.api_endpoints.items():
            api_table.add_row(service, endpoint)

        # Display all tables
        self.console.print(limits_table)
        self.console.print(features_table)
        self.console.print(api_table)

    async def confirm_operation(
        self, operation: str, details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Confirm sensitive operation with visual prompt."""
        env = self.manager.get_current_environment()
        theme = UITheme.get_theme(env)

        # Build confirmation message
        message = Text()
        message.append(f"\n{theme['icon']} ", style=theme["primary"])
        message.append("Operation Confirmation\n", style="bold white")
        message.append(f"Operation: ", style="dim")
        message.append(f"{operation}\n", style="yellow")

        if details:
            for key, value in details.items():
                message.append(f"{key}: ", style="dim")
                message.append(f"{value}\n", style="white")

        # Add environment warning
        if env == Environment.PRODUCTION:
            message.append("\n⚠️  ", style="red")
            message.append("This is a PRODUCTION operation!\n", style="bold red")

        # Create confirmation panel
        panel = Panel(
            message,
            border_style=(
                theme["warning"] if env != Environment.PRODUCTION else "bold red"
            ),
            padding=(1, 2),
        )

        self.console.print(panel)

        # Get confirmation
        return Confirm.ask(
            f"Do you want to proceed?",
            default=False if env == Environment.PRODUCTION else True,
        )

    async def select_environment(self) -> Optional[Environment]:
        """Interactive environment selection."""
        theme = self.current_theme

        # Create environment table
        table = Table(
            title="Available Environments",
            title_style=theme["title_style"],
            border_style=theme["border_style"],
        )

        table.add_column("#", style="cyan", width=3)
        table.add_column("Environment", style="white")
        table.add_column("Icon", width=3)
        table.add_column("Description", style="dim")

        descriptions = {
            Environment.DEVELOPMENT: "Local development and testing",
            Environment.TRAINING: "Agent training with synthetic data",
            Environment.SANDBOX: "Safe testing with demo APIs",
            Environment.STAGING: "Pre-production validation",
            Environment.PRODUCTION: "Live trading with real money",
        }

        environments = list(Environment)
        for i, env in enumerate(environments, 1):
            env_theme = UITheme.get_theme(env)
            style = "red bold" if env == Environment.PRODUCTION else "white"
            table.add_row(
                str(i),
                env.value.upper(),
                env_theme["icon"],
                descriptions[env],
                style=style,
            )

        self.console.print(table)

        # Get selection
        choice = Prompt.ask("Select environment (number or name)", default="cancel")

        if choice.lower() == "cancel":
            return None

        # Parse choice
        try:
            if choice.isdigit():
                index = int(choice) - 1
                if 0 <= index < len(environments):
                    return environments[index]
            else:
                return Environment(choice.lower())
        except (ValueError, IndexError):
            self.console.print("[red]Invalid selection[/]")
            return None

    def display_transition_plan(
        self, from_env: Environment, to_env: Environment, checks: List[str]
    ) -> None:
        """Display environment transition plan."""
        theme = UITheme.get_theme(to_env)

        # Create transition panel
        content = Text()
        content.append("Transition Plan\n\n", style="bold white")
        content.append(f"From: ", style="dim")
        content.append(f"{from_env.value.upper()}\n", style="cyan")
        content.append(f"To: ", style="dim")
        content.append(f"{to_env.value.upper()}\n", style=theme["primary"])

        if checks:
            content.append(f"\nRequired Checks:\n", style="yellow")
            for check in checks:
                content.append(f"  • {check}\n", style="white")

        panel = Panel(
            content,
            title="Environment Transition",
            border_style=theme["border_style"],
            padding=(1, 2),
        )

        self.console.print(panel)

    async def show_progress(self, task: str, total: int = 100) -> Progress:
        """Show progress bar for long operations."""
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        )

        task_id = progress.add_task(task, total=total)
        return progress

    def display_error(self, error: str) -> None:
        """Display error message."""
        theme = self.current_theme

        error_panel = Panel(
            Text(error, style="bold red"),
            title="❌ Error",
            border_style="red",
            padding=(1, 2),
        )

        self.console.print(error_panel)

    def display_success(self, message: str) -> None:
        """Display success message."""
        theme = self.current_theme

        success_panel = Panel(
            Text(message, style="bold green"),
            title="✅ Success",
            border_style="green",
            padding=(1, 2),
        )

        self.console.print(success_panel)

    def display_warning(self, warning: str) -> None:
        """Display warning message."""
        theme = self.current_theme

        warning_panel = Panel(
            Text(warning, style="bold yellow"),
            title="⚠️  Warning",
            border_style="yellow",
            padding=(1, 2),
        )

        self.console.print(warning_panel)


class EnvironmentCLI:
    """Command-line interface for environment management."""

    def __init__(self, manager: EnvironmentManager):
        """Initialize CLI."""
        self.manager = manager
        self.ui = EnvironmentUI(manager)
        self.commands = {
            "status": self.cmd_status,
            "config": self.cmd_config,
            "switch": self.cmd_switch,
            "check": self.cmd_check,
            "help": self.cmd_help,
            "exit": self.cmd_exit,
        }

    async def run(self) -> None:
        """Run interactive CLI."""
        self.ui.display_banner()

        while True:
            try:
                # Get command
                env_indicator = self.manager.get_visual_indicator()
                command = input(f"\n{env_indicator} > ").strip().lower()

                if not command:
                    continue

                # Parse command
                parts = command.split()
                cmd = parts[0]
                args = parts[1:] if len(parts) > 1 else []

                # Execute command
                if cmd in self.commands:
                    result = await self.commands[cmd](args)
                    if result == "exit":
                        break
                else:
                    self.ui.display_error(f"Unknown command: {cmd}")
                    self.cmd_help([])

            except KeyboardInterrupt:
                print("\n")
                if self.manager.is_production():
                    if await self.ui.confirm_operation(
                        "Exit from production environment"
                    ):
                        break
                else:
                    break
            except Exception as e:
                self.ui.display_error(f"Error: {str(e)}")

    async def cmd_status(self, args: List[str]) -> None:
        """Show environment status."""
        self.ui.display_status()

    async def cmd_config(self, args: List[str]) -> None:
        """Show environment configuration."""
        self.ui.display_config()

    async def cmd_switch(self, args: List[str]) -> None:
        """Switch environment."""
        # Select new environment
        new_env = await self.ui.select_environment()
        if not new_env:
            return

        # Get transition requirements
        from .environment_guards import TransitionGuard

        guard = TransitionGuard()
        checks = guard.get_required_checks(
            self.manager.get_current_environment(), new_env
        )

        # Display transition plan
        self.ui.display_transition_plan(
            self.manager.get_current_environment(), new_env, checks
        )

        # Confirm switch
        if await self.ui.confirm_operation(f"Switch to {new_env.value}"):
            success, message = await self.manager.switch_environment(new_env)

            if success:
                self.ui.display_success(message)
                self.ui.display_banner()
            else:
                self.ui.display_error(message)

    async def cmd_check(self, args: List[str]) -> None:
        """Check if operation is allowed."""
        if not args:
            self.ui.display_error("Usage: check <operation>")
            return

        operation = " ".join(args)
        allowed, reason = self.manager.can_execute_operation(operation)

        if allowed:
            self.ui.display_success(f"Operation '{operation}' is allowed: {reason}")
        else:
            self.ui.display_warning(f"Operation '{operation}' is blocked: {reason}")

    def cmd_help(self, args: List[str]) -> None:
        """Show help."""
        help_text = """
Available Commands:
  status  - Show environment status
  config  - Show environment configuration
  switch  - Switch to different environment
  check   - Check if operation is allowed
  help    - Show this help message
  exit    - Exit the CLI
"""
        self.ui.console.print(help_text)

    async def cmd_exit(self, args: List[str]) -> str:
        """Exit CLI."""
        if self.manager.is_production():
            if await self.ui.confirm_operation("Exit from production environment"):
                return "exit"
        else:
            return "exit"


def create_environment_cli() -> EnvironmentCLI:
    """Create and initialize environment CLI."""
    manager = EnvironmentManager()
    manager.initialize()
    return EnvironmentCLI(manager)
