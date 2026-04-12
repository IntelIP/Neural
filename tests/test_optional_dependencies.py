from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path


def test_optional_trading_dependencies_fall_back_to_placeholders() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        """
        import builtins
        import json
        import sys

        original_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name in {"websockets", "simplefix"}:
                raise ModuleNotFoundError(f"No module named '{name}'", name=name)
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = fake_import

        for module_name in [
            "neural",
            "neural.trading",
            "neural.trading.websocket",
            "neural.trading.fix",
            "neural.data_collection",
            "neural.data_collection.websocket",
        ]:
            sys.modules.pop(module_name, None)

        import neural  # noqa: F401
        from neural.data_collection import WebSocketSource
        from neural.trading import FIXConnectionConfig, KalshiFIXClient, KalshiWebSocketClient

        payload = {}
        for name, obj in {
            "WebSocketSource": WebSocketSource,
            "FIXConnectionConfig": FIXConnectionConfig,
            "KalshiFIXClient": KalshiFIXClient,
            "KalshiWebSocketClient": KalshiWebSocketClient,
        }.items():
            try:
                obj()
            except Exception as exc:  # noqa: BLE001
                payload[name] = {"type": type(exc).__name__, "message": str(exc)}
            else:
                payload[name] = {"type": None, "message": "constructed"}

        print(json.dumps(payload))
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["WebSocketSource"]["type"] == "ImportError"
    assert "neural-sdk[trading]" in payload["WebSocketSource"]["message"]
    assert payload["FIXConnectionConfig"]["type"] == "ImportError"
    assert "neural-sdk[trading]" in payload["FIXConnectionConfig"]["message"]
    assert payload["KalshiFIXClient"]["type"] == "ImportError"
    assert "neural-sdk[trading]" in payload["KalshiFIXClient"]["message"]
    assert payload["KalshiWebSocketClient"]["type"] == "ImportError"
    assert "neural-sdk[trading]" in payload["KalshiWebSocketClient"]["message"]
