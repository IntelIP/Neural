def test_api_surface_imports() -> None:
    # Data collection exports
    # Analysis exports
    from neural.analysis import (
        Strategy,
    )
    from neural.data_collection import (
        DataSource,
    )

    # Trading exports
    from neural.trading import (
        TradingClient,
    )

    # Simple asserts to silence linters
    assert Strategy and TradingClient and DataSource
