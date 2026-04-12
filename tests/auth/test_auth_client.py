from __future__ import annotations

import requests

from neural.auth.client import AuthClient


class _StubResponse:
    def __init__(self, json_exc: Exception) -> None:
        self.text = "plain-text-error"
        self._json_exc = json_exc

    def raise_for_status(self) -> None:
        raise requests.HTTPError("boom")

    def json(self) -> object:
        raise self._json_exc


def test_raise_for_status_falls_back_to_text_for_invalid_json() -> None:
    response = _StubResponse(ValueError("not-json"))

    try:
        AuthClient._raise_for_status(response)  # type: ignore[arg-type]
    except requests.HTTPError as exc:
        assert "plain-text-error" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("HTTPError was not raised")


def test_raise_for_status_does_not_swallow_unexpected_json_errors() -> None:
    response = _StubResponse(RuntimeError("json parser exploded"))

    try:
        AuthClient._raise_for_status(response)  # type: ignore[arg-type]
    except RuntimeError as exc:
        assert "json parser exploded" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Unexpected JSON error was swallowed")
