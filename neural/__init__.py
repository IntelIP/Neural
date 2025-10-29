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

__version__ = "0.4.0"
__author__ = "Neural Contributors"
__license__ = "MIT"

import warnings
from typing import Set  # noqa: UP035

from neural import analysis, auth, data_collection, deployment, trading

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


# Issue beta warning on import
_warn_beta()

__all__ = [
    "__version__",
    "auth",
    "data_collection",
    "analysis",
    "trading",
    "deployment",  # v0.4.0: Docker deployment module (experimental)
    "_warn_experimental",  # For internal use by modules
]
