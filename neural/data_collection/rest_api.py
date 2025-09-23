import asyncio
import requests
from typing import Dict, Any, Optional, AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from .base import DataSource


class RestApiSource(DataSource):
    """Data source for REST API endpoints."""

    def __init__(self, name: str, url: str, method: str = 'GET',
                 headers: Optional[Dict[str, str]] = None,
                 params: Optional[Dict[str, Any]] = None,
                 interval: float = 60.0,  # seconds
                 config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.url = url
        self.method = method.upper()
        self.headers = headers or {}
        self.params = params or {}
        self.interval = interval
        self._executor = ThreadPoolExecutor(max_workers=1)

    async def connect(self) -> None:
        """No persistent connection for REST APIs."""
        self._connected = True

    async def disconnect(self) -> None:
        """Close the executor."""
        self._executor.shutdown(wait=True)
        self._connected = False

    async def _fetch_data(self) -> Dict[str, Any]:
        """Fetch data from the REST API using requests in a thread."""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            self._executor,
            lambda: requests.request(
                self.method, self.url,
                headers=self.headers, params=self.params
            )
        )
        response.raise_for_status()
        return response.json()

    async def collect(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Continuously fetch data at intervals."""
        retry_count = 0
        max_retries = 3
        while self._connected:
            try:
                data = await self._fetch_data()
                yield data
                retry_count = 0  # Reset on success
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"Max retries reached for {self.name}: {e}")
                    break
                print(f"Error fetching from {self.name} (retry {retry_count}/{max_retries}): {e}")
                await asyncio.sleep(self.interval / 2)  # Shorter wait on error
            await asyncio.sleep(self.interval)