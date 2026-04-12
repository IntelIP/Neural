from __future__ import annotations

import json

import pytest

from neural._version import __version__
from neural.cli import main


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--version"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == __version__
    assert captured.err == ""


def test_cli_doctor_json(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delenv("KALSHI_API_KEY_ID", raising=False)
    monkeypatch.delenv("KALSHI_PRIVATE_KEY_BASE64", raising=False)
    monkeypatch.delenv("POLYMARKET_US_API_KEY", raising=False)
    monkeypatch.delenv("POLYMARKET_US_API_SECRET_BASE64", raising=False)
    monkeypatch.delenv("POLYMARKET_US_API_PASSPHRASE", raising=False)

    exit_code = main(["doctor", "--json"])

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert exit_code == 0
    assert report["version"] == __version__
    assert "credentials" in report
    assert "optional_dependencies" in report
    assert captured.err == ""


def test_cli_doctor_detects_kalshi_env(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("KALSHI_API_KEY_ID", "test-key")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY_BASE64", "dGVzdA==")

    exit_code = main(["doctor"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Kalshi: ready" in captured.out


def test_cli_importable_from_module() -> None:
    import neural.cli as cli

    assert hasattr(cli, "main")
