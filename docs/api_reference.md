# API Reference

Complete API documentation for the Neural SDK.

## Core Classes

### NeuralSDK
Main SDK class for trading operations.

### BacktestEngine  
Event-driven backtesting engine.

### BacktestConfig
Configuration for backtesting parameters.

## Data Providers

### DataProvider (Base)
Abstract base class for data providers.

### CSVProvider
Load data from CSV files.

### ParquetProvider  
Load data from Parquet files with PyArrow support.

### S3Provider (Optional)
Load data from Amazon S3 buckets.

### DatabaseProvider (Optional)
Load data from SQL databases.

## Portfolio Management

### Portfolio
Portfolio simulation with realistic costs.

### Position
Individual market position tracking.

### Trade
Trade execution record.

## Performance Analytics

### PerformanceMetrics
Comprehensive trading performance metrics.

---

*Full API documentation will be available at [neural-sdk.readthedocs.io](https://neural-sdk.readthedocs.io/)*