def test_api_surface_imports() -> None:
    from neural.analysis import Strategy
    from neural.data_collection import DataSource, PolymarketUSMarketsSource
    from neural.trading import (
        PolymarketUSAdapter,
        PolymarketUSMarketWebSocketClient,
        PolymarketUSUserWebSocketClient,
        TradingClient,
        TradingPolicy,
    )

    # Simple asserts to silence linters
    assert Strategy and TradingClient and DataSource and PolymarketUSMarketsSource
    assert PolymarketUSAdapter and TradingPolicy
    assert PolymarketUSMarketWebSocketClient and PolymarketUSUserWebSocketClient
