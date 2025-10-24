# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and this project adheres to Semantic Versioning.

## [0.3.0] - 2025-10-24

### Added
- **Historical Data Fetching:** Added `fetch_historical_candlesticks()` to KalshiMarketsSource with OHLCV support for backtesting
- **Enhanced Backtesting Engine:** Multi-sport support with Plotly visualization and caching capabilities
- **NBA Market Collection:** Implemented `get_nba_games()` with team parsing and date extraction
- **Enhanced CFB Collection:** Improved `get_cfb_games()` with better market discovery
- **Moneyline Market Discovery:** Added `filter_moneyline_markets()` utility and `get_moneyline_markets()` sport-agnostic function
- **Unified Sports Interface:** New `SportMarketCollector` class for multi-sport market collection
- **Complete v0.3.0 Demo:** End-to-end example showcasing NBA/NFL collection, moneyline filtering, and historical data analysis

### Changed
- **Data Collection Exports:** Updated `data_collection/__init__.py` to include new sports market utilities
- **Version Management:** Synchronized version numbers across `pyproject.toml`, `neural/__init__.py`, and `.bumpversion.cfg`

### Documentation
- Added comprehensive examples for historical data fetching and sports market analysis
- Updated README with sports examples and backtesting guides
- Enhanced documentation for multi-sport market discovery

### Performance
- Historical data fetching benchmarks: <1s per market
- Backtesting performance optimized with caching
- Real-time workflow demonstrations showing 43% price improvement potential

## [0.2.0] - 2025-01-13 (Beta)

### Fixed
- **Twitter API Domain (Bug #1):** Corrected base URL from `twitter-api.io` to `api.twitterapi.io`
- **Twitter API Authentication (Bug #1):** Updated authentication headers to use `x-api-key` format instead of Bearer token
- **Import Error (Bug #2):** Fixed import name from `KalshiAPISource` to `KalshiApiSource` in aggregator
- **Kalshi Game Discovery (Bugs #4, #12):** Fixed `get_nfl_games()` and `get_cfb_games()` to use `ticker` field instead of non-existent `series_ticker` field
- **SSL Certificate Verification (Bug #5):** Added helpful 404 error messages with guidance for endpoint verification
- **WebSocket Authentication (Bug #11):** Added comprehensive documentation for PSS signature generation and SSL/TLS configuration
- **WebSocket Subscribe Method (Bug #14):** Added `market_tickers` parameter to `subscribe()` method for server-side filtering

### Added
- **certifi dependency (Bug #5):** Added `certifi>=2023.0.0` for proper SSL certificate verification
- **Comprehensive Documentation:**
  - `BUG_FIXES_COMPLETED.md` - Complete fix summary with deployment guide
  - `BETA_BUGS_TRACKING.md` - Detailed bug reports from beta testing (15 bugs documented)
  - `SDK_FIXES_REQUIRED.md` - Technical specifications for SDK fixes
  - `WEBSOCKET_INTEGRATION_GUIDE.md` - Production-ready WebSocket usage patterns
  - `LIVE_TESTING_FINDINGS.md` - Live testing results and performance metrics

### Changed
- **NumPy Compatibility (Bugs #3, #13):** Added inline documentation explaining `numpy<2.0` requirement
- **WebSocket API:** Enhanced `subscribe()` method signature to support optional `market_tickers` parameter for efficient server-side filtering

### Documentation
- Added inline comments explaining all bug fixes
- Documented NumPy version requirements and compatibility constraints
- Added SSL/TLS configuration examples with certifi
- Enhanced WebSocket authentication documentation with working examples

### Code Quality
- Fixed 1,513 ruff linting errors (99.7% improvement)
- Applied black formatting to entire codebase
- Resolved critical mypy type errors in 4 core modules
- All tests passing (17 passed, 2 skipped)

### Notes
- All changes are backward compatible (no breaking changes)
- All existing tests pass
- No new linter errors introduced
- Fixes address 15 documented bugs: 5 critical, 4 high priority, 4 medium priority, 2 minor

## [0.1.0] - 2025-09-24

### Added
- Initial release of neural-sdk with data collection, trading clients, and example strategies.
- CI workflow for tests and code quality.
- Publish workflow for PyPI releases.
- Development tooling: pytest, mypy, ruff, black.
