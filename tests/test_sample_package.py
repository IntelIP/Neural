import json
import subprocess
import sys
from importlib import import_module
from pathlib import Path


def test_can_import_top_level() -> None:
    mod = import_module("neural")
    assert hasattr(mod, "__version__") or hasattr(mod, "__all__")


def test_top_level_import_is_lazy() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, sys; import neural; "
                "print(json.dumps({"
                "\"has_version\": hasattr(neural, \"__version__\"), "
                "\"analysis_loaded\": \"neural.analysis\" in sys.modules, "
                "\"trading_loaded\": \"neural.trading\" in sys.modules"
                "}))"
            ),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["has_version"] is True
    assert payload["analysis_loaded"] is False
    assert payload["trading_loaded"] is False


def test_trading_and_data_collection_imports_are_lazy() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, sys; import neural.data_collection as dc; import neural.trading as tr; "
                "print(json.dumps({"
                "\"dc_loaded\": \"neural.data_collection\" in sys.modules, "
                "\"dc_kalshi_loaded\": \"neural.data_collection.kalshi\" in sys.modules, "
                "\"dc_websocket_loaded\": \"neural.data_collection.websocket\" in sys.modules, "
                "\"tr_loaded\": \"neural.trading\" in sys.modules, "
                "\"tr_client_loaded\": \"neural.trading.client\" in sys.modules, "
                "\"tr_fix_loaded\": \"neural.trading.fix\" in sys.modules, "
                "\"tr_websocket_loaded\": \"neural.trading.websocket\" in sys.modules, "
                "\"dc_has_all\": hasattr(dc, \"__all__\"), "
                "\"tr_has_all\": hasattr(tr, \"__all__\")"
                "}))"
            ),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["dc_loaded"] is True
    assert payload["dc_kalshi_loaded"] is False
    assert payload["dc_websocket_loaded"] is False
    assert payload["tr_loaded"] is True
    assert payload["tr_client_loaded"] is False
    assert payload["tr_fix_loaded"] is False
    assert payload["tr_websocket_loaded"] is False
    assert payload["dc_has_all"] is True
    assert payload["tr_has_all"] is True


def test_auth_import_is_lazy() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, sys; import neural.auth as auth; "
                "print(json.dumps({"
                "\"auth_loaded\": \"neural.auth\" in sys.modules, "
                "\"auth_client_loaded\": \"neural.auth.client\" in sys.modules, "
                "\"auth_poly_env_loaded\": \"neural.auth.polymarket_us_env\" in sys.modules, "
                "\"requests_loaded\": \"requests\" in sys.modules, "
                "\"auth_has_all\": hasattr(auth, \"__all__\")"
                "}))"
            ),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["auth_loaded"] is True
    assert payload["auth_client_loaded"] is False
    assert payload["auth_poly_env_loaded"] is False
    assert payload["requests_loaded"] is False
    assert payload["auth_has_all"] is True
