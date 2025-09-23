#!/usr/bin/env python3
import asyncio
from neural.data_collection import KalshiMarketsSource

async def test():
    print("Starting debug test...")
    source = KalshiMarketsSource(
        series_ticker="NFL",
        status="open",
        use_authenticated=False,
        interval=float('inf')
    )

    print("\nCalling connect...")
    await source.connect()

    print("\nStarting collect...")
    # Just try to get one result
    try:
        gen = source.collect()
        df = await anext(gen)
        print(f"Got DataFrame with shape: {df.shape}")
    except StopAsyncIteration:
        print("No data returned")
    finally:
        await source.disconnect()

if __name__ == "__main__":
    asyncio.run(test())