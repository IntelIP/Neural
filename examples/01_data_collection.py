"""
Example usage of the neural data collection module.

This example demonstrates how to implement custom data sources for REST APIs and WebSockets,
and how to collect and transform data for analysis.
"""

import asyncio
import sys
import os

# Add the neural package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neural.data_collection import RestApiSource, WebSocketSource, DataTransformer, register_source


# Example custom REST API source
@register_source()
class WeatherApiSource(RestApiSource):
    """Custom REST API source for weather data."""

    def __init__(self, api_key: str, city: str = "New York"):
        super().__init__(
            name=f"weather_{city}",
            url=f"https://api.openweathermap.org/data/2.5/weather",
            params={"q": city, "appid": api_key, "units": "metric"},
            interval=300.0  # 5 minutes
        )


# Example custom WebSocket source
@register_source()
class CryptoPriceSource(WebSocketSource):
    """Custom WebSocket source for cryptocurrency prices."""

    def __init__(self, symbol: str = "btcusdt"):
        super().__init__(
            name=f"crypto_{symbol}",
            uri=f"wss://stream.binance.com:9443/ws/{symbol}@ticker"
        )


async def collect_weather_data():
    """Example of collecting weather data."""
    # Note: Replace with actual API key
    api_key = "your_openweather_api_key_here"
    
    transformer = DataTransformer()
    transformer.add_transformation(DataTransformer.normalize_types)
    transformer.add_transformation(DataTransformer.flatten_keys)
    
    source = WeatherApiSource(api_key, "London")
    
    async with source:
        async for data in source.collect():
            transformed = transformer.transform(data)
            print(f"Weather data: {transformed}")
            break  # Just one sample


async def collect_crypto_data():
    """Example of collecting crypto price data."""
    transformer = DataTransformer()
    transformer.add_transformation(lambda d: {k: v for k, v in d.items() if k in ['s', 'c', 'P']})  # Filter relevant fields
    
    source = CryptoPriceSource("ethusdt")
    
    async with source:
        count = 0
        async for data in source.collect():
            transformed = transformer.transform(data)
            print(f"Crypto data: {transformed}")
            count += 1
            if count >= 5:  # Collect 5 messages
                break


async def main():
    """Run the examples."""
    print("Collecting weather data...")
    try:
        await collect_weather_data()
    except Exception as e:
        print(f"Weather collection failed: {e}")
    
    print("\nCollecting crypto data...")
    try:
        await collect_crypto_data()
    except Exception as e:
        print(f"Crypto collection failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())