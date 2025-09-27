def test_api_surface_imports() -> None:
    # Data collection exports
    from neural.data_collection import (
        DataSource,
        RestApiSource,
        WebSocketSource,
        DataTransformer,
        DataSourceRegistry,
        registry,
        register_source,
        KalshiApiSource,
        KalshiMarketsSource,
        get_sports_series,
        get_markets_by_sport,
        get_all_sports_markets,
        search_markets,
        get_game_markets,
        get_live_sports,
    )

    # Analysis exports
    from neural.analysis import (
        Strategy,
        Signal,
        Position,
        Backtester,
        OrderManager,
        kelly_criterion,
        fixed_percentage,
        edge_proportional,
    )

    # Trading exports
    from neural.trading import (
        TradingClient,
        KalshiWebSocketClient,
        KalshiFIXClient,
        FIXConnectionConfig,
    )

    # Simple asserts to silence linters
    assert Strategy and TradingClient and DataSource
