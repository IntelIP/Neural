"""Neural SDK public package surface."""

from __future__ import annotations

import importlib
import warnings
from types import ModuleType

__version__ = "0.4.0"
__author__ = "Neural Contributors"
__license__ = "MIT"

_OPTIONAL_SUBMODULES = {
    "analysis",
    "auth",
    "data_collection",
    "deployment",
    "exchanges",
    "trading",
}

_experimental_features_used: set[str] = set()
_beta_warning_issued = False


def _warn_experimental(feature: str, module: str | None = None) -> None:
    """Issue a warning for experimental features."""
    if feature not in _experimental_features_used:
        _experimental_features_used.add(feature)
        module_info = f" in {module}" if module else ""
        warnings.warn(
            f"{feature}{module_info} is experimental in Neural SDK Beta v{__version__}. "
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
            f"Neural SDK Beta v{__version__} is in beta. "
            "Core features are stable, but advanced modules are experimental. "
            "See https://github.com/IntelIP/Neural#module-status for details.",
            UserWarning,
            stacklevel=2,
        )
        _beta_warning_issued = True


def __getattr__(name: str) -> ModuleType:
    if name not in _OPTIONAL_SUBMODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        module = importlib.import_module(f".{name}", __name__)
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None) or "unknown"
        raise ModuleNotFoundError(
            f"Neural SDK submodule {name!r} is unavailable because dependency {missing!r} is not installed. "
            "Install the matching optional extra before importing it."
        ) from exc

    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(set(globals()) | _OPTIONAL_SUBMODULES)



__all__ = [
    "__version__",
    "analysis",
    "auth",
    "data_collection",
    "deployment",
    "exchanges",
    "trading",
    "_warn_experimental",
]

