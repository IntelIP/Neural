from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from neural.deployment.base import DeploymentProvider
from neural.deployment.config import DeploymentInfo, DeploymentResult, DeploymentStatus
from neural.deployment.exceptions import ConfigurationError, ProviderNotFoundError
from neural.deployment.registry import (
    _reset_registry_for_tests,
    create_provider,
    discover_providers,
    list_providers,
    register_provider,
)


@dataclass
class _DummyProvider(DeploymentProvider):
    marker: str = "ok"

    async def deploy(self, config) -> DeploymentResult:  # pragma: no cover - not used in tests
        return DeploymentResult(deployment_id="x", status="running")

    async def stop(self, deployment_id: str) -> bool:  # pragma: no cover - not used in tests
        return True

    async def status(self, deployment_id: str) -> DeploymentStatus:  # pragma: no cover
        return DeploymentStatus(deployment_id=deployment_id, status="running")

    async def logs(self, deployment_id: str, tail: int = 100) -> list[str]:  # pragma: no cover
        return []

    async def list_deployments(self) -> list[DeploymentInfo]:  # pragma: no cover
        return []

    async def cleanup(self) -> None:  # pragma: no cover
        return None


@pytest.fixture(autouse=True)
def _reset_registry_state() -> None:
    _reset_registry_for_tests()
    yield
    _reset_registry_for_tests()


def test_register_list_and_create_provider() -> None:
    register_provider("dummy", _DummyProvider)

    assert list_providers() == ["dummy"]

    provider = create_provider("dummy", marker="custom")
    assert isinstance(provider, _DummyProvider)
    assert provider.marker == "custom"


def test_register_duplicate_provider_without_replace_fails() -> None:
    register_provider("dummy", _DummyProvider)
    with pytest.raises(ConfigurationError, match="already registered"):
        register_provider("dummy", _DummyProvider)


def test_create_provider_unknown_name_shows_available() -> None:
    register_provider("dummy", _DummyProvider)

    with pytest.raises(ProviderNotFoundError, match="Available providers: dummy"):
        create_provider("missing")


def test_discover_providers_loads_entry_points_and_tracks_broken_plugins(monkeypatch) -> None:
    class _FakeEntryPoint:
        def __init__(self, name: str, value: str, loader: Any):
            self.name = name
            self.value = value
            self._loader = loader

        def load(self):
            return self._loader()

    class _FakeEntryPoints:
        def __init__(self, items: list[_FakeEntryPoint]):
            self._items = items

        def select(self, **kwargs):
            if kwargs.get("group") == "neural.deployment.providers":
                return self._items
            return []

    def _good_loader():
        return _DummyProvider

    def _bad_loader():
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "neural.deployment.registry.metadata.entry_points",
        lambda: _FakeEntryPoints(
            [
                _FakeEntryPoint("good_plugin", "pkg.good:factory", _good_loader),
                _FakeEntryPoint("bad_plugin", "pkg.bad:factory", _bad_loader),
            ]
        ),
    )

    discover_providers(force=True)

    assert "good_plugin" in list_providers()
    provider = create_provider("good_plugin", marker="plugin")
    assert isinstance(provider, _DummyProvider)
    assert provider.marker == "plugin"

    with pytest.raises(ProviderNotFoundError, match="failed to load"):
        create_provider("bad_plugin")


def test_factory_must_return_provider_instance() -> None:
    register_provider("bad", lambda **_: object())

    with pytest.raises(ConfigurationError, match="expected a DeploymentProvider instance"):
        create_provider("bad")

