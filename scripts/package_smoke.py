#!/usr/bin/env python3
from __future__ import annotations

import builtins
import importlib
import json
import subprocess
import sys
from pathlib import Path


def _assert_optional_dependency_contract() -> None:
    blocked_imports = {"certifi", "simplefix", "websockets"}
    module_names = [
        "neural.data_collection.websocket",
        "neural.trading.fix",
        "neural.trading.websocket",
    ]
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in blocked_imports:
            raise ModuleNotFoundError(f"No module named '{name}'", name=name)
        return original_import(name, globals, locals, fromlist, level)

    builtins.__import__ = fake_import
    try:
        import neural.data_collection as data_collection
        import neural.trading as trading

        for module_name in module_names:
            sys.modules.pop(module_name, None)
        for module, names in (
            (data_collection, ("WebSocketSource",)),
            (trading, ("FIXConnectionConfig", "KalshiFIXClient", "KalshiWebSocketClient")),
        ):
            for name in names:
                module.__dict__.pop(name, None)

        importlib.invalidate_caches()

        from neural.data_collection import WebSocketSource
        from neural.trading import (
            FIXConnectionConfig,
            KalshiFIXClient,
            KalshiWebSocketClient,
        )

        for symbol in (WebSocketSource, FIXConnectionConfig, KalshiFIXClient, KalshiWebSocketClient):
            try:
                symbol()
            except ImportError as exc:
                assert "neural-sdk[trading]" in str(exc)
            else:
                raise AssertionError(f"{symbol.__name__} should require trading extras")
    finally:
        builtins.__import__ = original_import


def _assert_clean_cli() -> None:
    cli_path = Path(sys.executable).with_name("neural")

    version = subprocess.run(
        [str(cli_path), "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert version.stdout.strip() == "0.4.1"
    assert version.stderr == ""

    doctor = subprocess.run(
        [str(cli_path), "doctor", "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert doctor.stderr == ""
    payload = json.loads(doctor.stdout)
    assert payload["version"] == "0.4.1"
    assert "credentials" in payload
    assert "optional_dependencies" in payload


def _assert_import_surface() -> None:
    import neural
    import neural.auth as auth
    import neural.cli
    import neural.data_collection as data_collection
    import neural.trading as trading

    assert neural.__version__ == "0.4.1"
    assert neural.cli.main
    assert auth.__all__ and data_collection.__all__ and trading.__all__
    assert "site-packages" in neural.__file__
    assert "neural.auth.client" not in sys.modules
    assert "neural.data_collection.kalshi" not in sys.modules
    assert "neural.trading.client" not in sys.modules

    from neural.analysis import Strategy
    from neural.auth import AuthClient
    from neural.data_collection import DataSource, WebSocketSource
    from neural.trading import (
        FIXConnectionConfig,
        KalshiFIXClient,
        KalshiWebSocketClient,
        TradingClient,
    )

    assert AuthClient and Strategy and DataSource and TradingClient

    _assert_optional_dependency_contract()


def main() -> int:
    _assert_import_surface()
    _assert_clean_cli()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
