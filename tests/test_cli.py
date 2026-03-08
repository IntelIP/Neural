from __future__ import annotations

import json
import os
import subprocess
import sys
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

from neural import cli


def _capture(monkeypatch: pytest.MonkeyPatch) -> tuple[StringIO, StringIO]:
    stdout = StringIO()
    stderr = StringIO()
    monkeypatch.setattr("sys.stdout", stdout)
    monkeypatch.setattr("sys.stderr", stderr)
    return stdout, stderr


def test_doctor_json_output(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout, stderr = _capture(monkeypatch)
    monkeypatch.setattr(cli, "_safe_list_providers", lambda: ["daytona"])

    exit_code = cli.main(["--json", "doctor"])

    assert exit_code == 0
    assert stderr.getvalue() == ""
    payload = json.loads(stdout.getvalue())
    assert payload["ok"] is True
    assert payload["data"]["providers"] == ["daytona"]
    assert "doctor" in payload["data"]["commands"]
    assert "daytona_cli_available" in payload["data"]["tooling"]


def test_capabilities_json_output(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout, _ = _capture(monkeypatch)
    monkeypatch.setattr(cli, "_safe_list_providers", lambda: ["daytona"])

    exit_code = cli.main(["--json", "capabilities"])

    assert exit_code == 0
    payload = json.loads(stdout.getvalue())
    assert payload["ok"] is True
    assert payload["data"]["platform"]["private_provider_installed"] is True
    assert "stop" in payload["data"]["platform"]["deployment_controls"]


def test_paper_order_json_output(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout, _ = _capture(monkeypatch)

    class _FakePaperTradingClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

        def update_market_price(self, *args: Any, **kwargs: Any) -> None:
            return None

        async def place_order(self, **kwargs: Any) -> dict[str, Any]:
            return {
                "success": True,
                "order_id": "PAPER_1",
                "message": f"placed {kwargs['quantity']}",
            }

        def get_portfolio(self) -> dict[str, Any]:
            return {"portfolio_value": 10050.0}

    monkeypatch.setattr(cli, "_get_paper_trading_client_class", lambda: _FakePaperTradingClient)

    exit_code = cli.main(
        [
            "--json",
            "paper",
            "order",
            "--market-id",
            "TEST-1",
            "--side",
            "yes",
            "--quantity",
            "5",
            "--price",
            "0.55",
            "--data-dir",
            str(Path.cwd() / ".tmp-paper-cli"),
        ]
    )

    assert exit_code == 0
    payload = json.loads(stdout.getvalue())
    assert payload["ok"] is True
    assert payload["data"]["order"]["order_id"] == "PAPER_1"


def test_paper_order_json_output_real_client(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout, stderr = _capture(monkeypatch)

    exit_code = cli.main(
        [
            "--json",
            "paper",
            "order",
            "--market-id",
            "TEST-1",
            "--side",
            "yes",
            "--quantity",
            "1",
            "--price",
            "0.55",
            "--data-dir",
            str(Path.cwd() / ".tmp-paper-cli"),
        ]
    )

    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert "Infinity" not in stdout.getvalue()
    payload = json.loads(stdout.getvalue())
    assert payload["ok"] is True
    assert payload["data"]["order"]["success"] is True


def test_deployments_list_json_output_with_default_provider_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout, _ = _capture(monkeypatch)

    class _FakeProvider:
        async def list_deployments(self) -> list[dict[str, Any]]:
            return [{"deployment_id": "daytona-123", "status": "running", "environment": "paper"}]

    def _fake_create_provider(name: str, **kwargs: Any) -> _FakeProvider:
        assert name == "docker"
        assert kwargs["project_name"] == "neural"
        return _FakeProvider()

    monkeypatch.setattr(cli, "_safe_list_providers", lambda: ["docker"])
    monkeypatch.setattr(cli, "_create_provider", _fake_create_provider)

    exit_code = cli.main(["--json", "deployments", "list"])

    assert exit_code == 0
    payload = json.loads(stdout.getvalue())
    assert payload["ok"] is True
    assert payload["data"]["provider"] == "docker"
    assert payload["data"]["count"] == 1


def test_deployments_stop_json_output(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout, _ = _capture(monkeypatch)

    class _FakeProvider:
        async def stop(self, deployment_id: str) -> bool:
            assert deployment_id == "daytona-123"
            return True

    monkeypatch.setattr(cli, "_build_provider", lambda args: ("daytona", _FakeProvider()))

    exit_code = cli.main(["--json", "deployments", "stop", "daytona-123"])

    assert exit_code == 0
    payload = json.loads(stdout.getvalue())
    assert payload["ok"] is True
    assert payload["data"]["stopped"] is True


def test_deployments_list_json_output_without_discovered_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout, _ = _capture(monkeypatch)
    monkeypatch.setattr(cli, "_safe_list_providers", lambda: [])

    exit_code = cli.main(["--json", "deployments", "list"])

    assert exit_code == 1
    payload = json.loads(stdout.getvalue())
    assert payload["ok"] is False
    assert payload["error"]["code"] == "RuntimeError"
    assert "No deployment providers discovered" in payload["error"]["message"]


def test_python_module_cli_json_is_clean_on_stderr() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pythonpath = str(repo_root)
    if os.environ.get("PYTHONPATH"):
        pythonpath = os.pathsep.join([str(repo_root), os.environ["PYTHONPATH"]])

    result = subprocess.run(
        [sys.executable, "-m", "neural", "--json", "providers", "list"],
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": pythonpath},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload["ok"] is True

def test_missing_subcommand_json_output(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout, stderr = _capture(monkeypatch)

    exit_code = cli.main(["--json", "deployments"])

    assert exit_code == 1
    assert stderr.getvalue() == ""
    payload = json.loads(stdout.getvalue())
    assert payload["ok"] is False
    assert payload["error"]["code"] == "ValueError"
    assert "Missing sub-command" in payload["error"]["message"]
