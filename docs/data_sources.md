# Data Sources Guide

Guide to setting up and using different data sources with the Neural SDK.

## Supported Sources

| Source | Installation | Use Case |
|--------|-------------|----------|
| CSV Files | Built-in | Local historical data |
| Parquet Files | `pip install neural-sdk[parquet]` | High-performance data |
| S3 Buckets | `pip install neural-sdk[s3]` | Cloud data storage |
| SQL Databases | `pip install neural-sdk[database]` | Large datasets |

## Configuration

### CSV Files
```python
from neural_sdk.backtesting import DataLoader

loader = DataLoader()
data = loader.load("csv", path="trades.csv")
```

### S3 Buckets
```python
data = loader.load("s3", bucket="my-data", prefix="trades/2024/")
```

### SQL Databases
```python
data = loader.load("postgres", 
    connection_string="postgresql://user:pass@host/db",
    source="trades_table")
```

## Custom Data Sources

Create custom data providers by inheriting from `DataProvider`:

```python
from neural_sdk.backtesting.providers import DataProvider

class MyAPIProvider(DataProvider):
    def fetch(self, source, **kwargs):
        # Your custom data loading logic
        return pd.DataFrame(data)
```

## Data Format

All data sources must provide:
- `timestamp` (datetime)
- `symbol` (string)  
- `price` (float)
- `volume` (optional, integer)

See `examples/custom_data_adapter.py` for detailed examples.