"""
Neural SDK - Professional-grade SDK for algorithmic trading on prediction markets.

This package provides tools for:
- Authentication with Kalshi API
- Historical and real-time market data collection
- Trading strategy development and backtesting
- Risk management and position sizing
- Order execution via REST and FIX protocols

⚠️ BETA NOTICE: This package is in beta. Core features are stable, but advanced
modules (sentiment analysis, FIX streaming) are experimental.
"""

__author__ = "Neural Contributors"
__license__ = "MIT"

import warnings
from importlib import import_module
from typing import Any

from ._version import __version__

_LAZY_SUBMODULES = {"analysis", "auth", "data_collection", "deployment", "exchanges", "trading"}

# Track which experimental features have been used
_experimental_features_used: set[str] = set()

# Track if beta warning has been issued
_beta_warning_issued = False


def _warn_experimental(feature: str, module: str | None = None) -> None:
    """Issue a warning for experimental features."""
    if feature not in _experimental_features_used:
        _experimental_features_used.add(feature)
        module_info = f" in {module}" if module else ""
        warnings.warn(
            f"⚠️  {feature}{module_info} is experimental in Neural SDK Beta v{__version__}. "
            "Use with caution in production environments. "
            "See https://github.com/IntelIP/Neural#module-status for details.",
            UserWarning,
            stacklevel=3,
        )


def _warn_beta() -> None:
    """Issue a one-time beta warning."""
    global _beta_warning_issued
    if not _beta_warning_issued:
        warnings.warn(
            f"⚠️  Neural SDK Beta v{__version__} is in BETA. "
            "Core features are stable, but advanced modules are experimental. "
            "See https://github.com/IntelIP/Neural#module-status for details.",
            UserWarning,
            stacklevel=2,
        )
        _beta_warning_issued = True


def __getattr__(name: str) -> Any:
    """Load heavy SDK submodules on first access instead of at package import time."""
    if name not in _LAZY_SUBMODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if name == "deployment":
        try:
            module = import_module(f".{name}", __name__)
        except ImportError as exc:
            if not any(dep in str(exc) for dep in ("docker", "pydantic", "jinja2")):
                raise
            module = None
    else:
        module = import_module(f".{name}", __name__)

    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_SUBMODULES)

__all__ = [
    "__version__",
    "auth",
    "data_collection",
    "analysis",
    "trading",
    "exchanges",
    "deployment",  # v0.4.0: Docker deployment module (experimental)
    "_warn_experimental",  # For internal use by modules
]
