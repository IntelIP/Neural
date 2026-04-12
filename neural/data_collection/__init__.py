from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "DataSource": (".base", "DataSource"),
    "RestApiSource": (".rest_api", "RestApiSource"),
    "DataTransformer": (".transformer", "DataTransformer"),
    "DataSourceRegistry": (".registry", "DataSourceRegistry"),
    "registry": (".registry", "registry"),
    "register_source": (".registry", "register_source"),
    "KalshiApiSource": (".kalshi_api_source", "KalshiApiSource"),
    "PolymarketUSMarketsSource": (".polymarket_us", "PolymarketUSMarketsSource"),
    "PolymarketUSConfig": (".polymarket_us", "PolymarketUSConfig"),
    "KalshiMarketsSource": (".kalshi", "KalshiMarketsSource"),
    "SportMarketCollector": (".kalshi", "SportMarketCollector"),
    "filter_moneyline_markets": (".kalshi", "filter_moneyline_markets"),
    "get_all_sports_markets": (".kalshi", "get_all_sports_markets"),
    "get_cfb_games": (".kalshi", "get_cfb_games"),
    "get_game_markets": (".kalshi", "get_game_markets"),
    "get_live_sports": (".kalshi", "get_live_sports"),
    "get_markets_by_sport": (".kalshi", "get_markets_by_sport"),
    "get_moneyline_markets": (".kalshi", "get_moneyline_markets"),
    "get_nba_games": (".kalshi", "get_nba_games"),
    "get_nfl_games": (".kalshi", "get_nfl_games"),
    "get_sports_series": (".kalshi", "get_sports_series"),
    "search_markets": (".kalshi", "search_markets"),
}


def _placeholder(message: str, exc: ImportError) -> type[Any]:
    class _MissingDependency:  # type: ignore[too-many-ancestors]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(message) from exc

    _MissingDependency.__name__ = "WebSocketSource"
    return _MissingDependency


def __getattr__(name: str) -> Any:
    if name == "WebSocketSource":
        try:
            value = getattr(import_module(".websocket", __name__), name)
        except ImportError as exc:
            if "websockets" not in str(exc):
                raise
            value = _placeholder(
                "WebSocketSource requires optional trading extras. "
                "Install with: pip install 'neural-sdk[trading]'",
                exc,
            )
        globals()[name] = value
        return value

    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        value = getattr(import_module(module_name, __name__), attr_name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS) | {"WebSocketSource"})


__all__ = [
    "DataSource",
    "RestApiSource",
    "WebSocketSource",
    "DataTransformer",
    "DataSourceRegistry",
    "registry",
    "register_source",
    "KalshiApiSource",
    "PolymarketUSMarketsSource",
    "PolymarketUSConfig",
    "KalshiMarketsSource",
    "SportMarketCollector",
    "filter_moneyline_markets",
    "get_all_sports_markets",
    "get_cfb_games",
    "get_game_markets",
    "get_live_sports",
    "get_markets_by_sport",
    "get_moneyline_markets",
    "get_nba_games",
    "get_nfl_games",
    "get_sports_series",
    "search_markets",
]
