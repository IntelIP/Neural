# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and this project adheres to Semantic Versioning.

## [0.2.0] - 2025-10-11 (Beta)

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
