"""
Provider registry and plugin discovery for deployment backends.

External packages can register deployment providers via setuptools entry points
under the group ``neural.deployment.providers``.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import metadata
from typing import Any

from neural.deployment.base import DeploymentProvider
from neural.deployment.exceptions import ConfigurationError, ProviderNotFoundError

PROVIDER_ENTRYPOINT_GROUP = "neural.deployment.providers"

ProviderFactory = Callable[..., DeploymentProvider]

_provider_factories: dict[str, ProviderFactory] = {}
_provider_load_errors: dict[str, str] = {}
_plugins_discovered = False


def _normalize_provider_name(name: str) -> str:
    return name.strip().lower()


def register_provider(name: str, factory: ProviderFactory, *, replace: bool = False) -> None:
    """Register a deployment provider factory.

    Args:
        name: Provider name (e.g., "docker", "daytona")
        factory: Callable that returns a DeploymentProvider instance
        replace: Whether to overwrite an existing provider with the same name

    Raises:
        ConfigurationError: If the provider name/factory is invalid or name is duplicated
    """
    normalized_name = _normalize_provider_name(name)
    if not normalized_name:
        raise ConfigurationError("Provider name must be a non-empty string.")
    if not callable(factory):
        raise ConfigurationError(f"Provider '{normalized_name}' factory must be callable.")
    if normalized_name in _provider_factories and not replace:
        raise ConfigurationError(
            f"Provider '{normalized_name}' is already registered. "
            "Use replace=True to override it."
        )

    _provider_factories[normalized_name] = factory
    # Clear stale loader error if the provider was successfully registered later.
    _provider_load_errors.pop(normalized_name, None)


def discover_providers(*, force: bool = False) -> None:
    """Discover provider plugins from Python entry points.

    Discovery is cached after the first successful pass unless ``force=True``.
    """
    global _plugins_discovered
    if _plugins_discovered and not force:
        return

    entry_points = metadata.entry_points()
    if hasattr(entry_points, "select"):
        providers = entry_points.select(group=PROVIDER_ENTRYPOINT_GROUP)
    else:  # pragma: no cover - compatibility branch for older runtimes
        providers = entry_points.get(PROVIDER_ENTRYPOINT_GROUP, [])

    for entry_point in providers:
        provider_name = _normalize_provider_name(entry_point.name)
        if provider_name in _provider_factories:
            # Keep built-in providers deterministic; external plugins can use a different name.
            continue
        try:
            loaded_factory = entry_point.load()
            if not callable(loaded_factory):
                raise TypeError("entry point did not resolve to a callable provider factory")
            register_provider(provider_name, loaded_factory)
        except Exception as exc:
            _provider_load_errors[provider_name] = (
                f"Provider plugin '{provider_name}' failed to load from '{entry_point.value}': {exc}"
            )

    _plugins_discovered = True


def list_providers() -> list[str]:
    """Return all successfully registered provider names."""
    discover_providers()
    return sorted(_provider_factories.keys())


def create_provider(name: str, **kwargs: Any) -> DeploymentProvider:
    """Create a provider instance by name.

    Args:
        name: Registered provider name
        **kwargs: Arguments forwarded to the provider factory

    Raises:
        ProviderNotFoundError: If provider is missing or failed to load
        ConfigurationError: If the factory returns the wrong object type
    """
    discover_providers()
    normalized_name = _normalize_provider_name(name)

    load_error = _provider_load_errors.get(normalized_name)
    if load_error:
        raise ProviderNotFoundError(load_error)

    factory = _provider_factories.get(normalized_name)
    if not factory:
        available = ", ".join(sorted(_provider_factories.keys())) or "(none)"
        raise ProviderNotFoundError(
            f"Provider '{normalized_name}' is not registered. Available providers: {available}."
        )

    provider = factory(**kwargs)
    if not isinstance(provider, DeploymentProvider):
        raise ConfigurationError(
            f"Provider '{normalized_name}' factory returned {type(provider).__name__}, "
            "expected a DeploymentProvider instance."
        )
    return provider


def _reset_registry_for_tests() -> None:
    """Reset global state for unit tests."""
    global _plugins_discovered
    _provider_factories.clear()
    _provider_load_errors.clear()
    _plugins_discovered = False
