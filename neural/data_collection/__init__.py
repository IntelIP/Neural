"""Neural data-collection surface with lazy imports for provider-specific sources."""

from __future__ import annotations

import importlib

_ATTRIBUTE_EXPORTS = {
    "DataSource": (".base", "DataSource"),
    "RestApiSource": (".rest_api", "RestApiSource"),
    "WebSocketSource": (".websocket", "WebSocketSource"),
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


def __getattr__(name: str) -> object:
    target = _ATTRIBUTE_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = target
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_ATTRIBUTE_EXPORTS))


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
