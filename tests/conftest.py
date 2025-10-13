import asyncio
import os

import pytest


@pytest.fixture(autouse=True)
def _set_default_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide sane defaults so tests don't hit real services by accident.
    Env vars can be overridden in CI via repository secrets.
    """
    monkeypatch.setenv("KALSHI_EMAIL", os.getenv("KALSHI_EMAIL", "test@example.com"))
    monkeypatch.setenv("KALSHI_PASSWORD", os.getenv("KALSHI_PASSWORD", "password123"))
    monkeypatch.setenv(
        "KALSHI_API_BASE", os.getenv("KALSHI_API_BASE", "https://api.elections.kalshi.com")
    )


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
