from __future__ import annotations

import base64

import pytest

from neural.auth.polymarket_us_env import get_polymarket_us_api_secret


def test_env_secret_accepts_valid_base64(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYMARKET_US_API_SECRET", base64.b64encode(b"x" * 32).decode("ascii"))
    assert get_polymarket_us_api_secret() == b"x" * 32


def test_env_secret_accepts_pem_string(monkeypatch: pytest.MonkeyPatch) -> None:
    pem = "-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----"
    monkeypatch.setenv("POLYMARKET_US_API_SECRET", pem)
    assert get_polymarket_us_api_secret() == pem.encode("utf-8")


def test_env_secret_rejects_invalid_base64(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYMARKET_US_API_SECRET", "not-base64!!")
    with pytest.raises(ValueError, match="must be valid base64 or PEM"):
        get_polymarket_us_api_secret()
