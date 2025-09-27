import asyncio
from neural.data_collection.kalshi_api_source import KalshiApiSource
from neural.data_collection.transformer import DataTransformer
from neural.data_collection.registry import registry
import time

class KalshiMarketPoller(KalshiApiSource):
    def __init__(self, ticker):
        super().__init__(
            name='kalshi_poller',
            url=f'https://api.elections.kalshi.com/trade-api/v2/markets/{ticker}',
            interval=5.0
        )

async def stream_prices(ticker, duration=30):
    source = KalshiMarketPoller(ticker)
    transformer = DataTransformer([lambda d: d.get('market', {})])
    registry.sources['kalshi_poller'] = source
    registry.transformers['kalshi_poller'] = transformer
    
    print(f'Streaming {ticker} for {duration}s (poll every 5s)...')
    start = time.time()
    updates = 0
    
    async with source:
        async for raw_data in source.collect():
            if time.time() - start > duration:
                break
            transformed = transformer.transform(raw_data)
            yes_ask = transformed.get('yes_ask', 'N/A')
            no_ask = transformed.get('no_ask', 'N/A')
            volume = transformed.get('volume', 'N/A')
            print(f'[{time.strftime("%H:%M:%S")}] Yes Ask: {yes_ask}, No Ask: {no_ask}, Volume: {volume}')
            updates += 1

if __name__ == '__main__':
    asyncio.run(stream_prices('KXNFLGAME-25SEP25SEAARI-SEA', 30))
