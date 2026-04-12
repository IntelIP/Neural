from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "AuthClient": (".client", "AuthClient"),
    "KalshiSigner": (".signers.kalshi", "KalshiSigner"),
    "PolymarketUSSigner": (".signers.polymarket_us", "PolymarketUSSigner"),
    "get_polymarket_us_base_url": (".polymarket_us_env", "get_polymarket_us_base_url"),
    "get_polymarket_us_api_key": (".polymarket_us_env", "get_polymarket_us_api_key"),
    "get_polymarket_us_api_secret": (".polymarket_us_env", "get_polymarket_us_api_secret"),
    "get_polymarket_us_passphrase": (".polymarket_us_env", "get_polymarket_us_passphrase"),
    "get_polymarket_us_credentials": (".polymarket_us_env", "get_polymarket_us_credentials"),
}


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        value = getattr(import_module(module_name, __name__), attr_name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


__all__ = [
    "AuthClient",
    "KalshiSigner",
    "PolymarketUSSigner",
    "get_polymarket_us_base_url",
    "get_polymarket_us_api_key",
    "get_polymarket_us_api_secret",
    "get_polymarket_us_passphrase",
    "get_polymarket_us_credentials",
]
