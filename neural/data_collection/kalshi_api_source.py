import asyncio
import requests
from typing import Dict, Any, Optional, AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from .base import DataSource
from neural.auth.signers.kalshi import KalshiSigner
from neural.auth.env import get_api_key_id, get_private_key_material


class KalshiApiSource(DataSource):
    """Authenticated data source for Kalshi REST API endpoints."""

    def __init__(self, name: str, url: str, method: str = 'GET',
                 params: Optional[Dict[str, Any]] = None,
                 interval: float = 60.0,
                 config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.url = url
        self.method = method.upper()
        self.params = params or {}
        self.interval = interval
        self._executor = ThreadPoolExecutor(max_workers=1)

        api_key_id = get_api_key_id()
        private_key_pem = get_private_key_material()
        self.signer = KalshiSigner(api_key_id, private_key_pem)

    async def connect(self) -> None:
        """No persistent connection for REST APIs."""
        self._connected = True

    async def disconnect(self) -> None:
        """Close the executor."""
        self._executor.shutdown(wait=True)
        self._connected = False

    async def _fetch_data(self) -> Dict[str, Any]:
        """Fetch data from the Kalshi API with authentication."""
        loop = asyncio.get_event_loop()

        from urllib.parse import urlparse
        parsed = urlparse(self.url)
        path = parsed.path

        auth_headers = self.signer.headers(self.method, path)

        response = await loop.run_in_executor(
            self._executor,
            lambda: requests.request(
                self.method, self.url,
                headers=auth_headers, params=self.params
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
                retry_count = 0
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"Max retries reached for {self.name}: {e}")
                    break
                print(f"Error fetching from {self.name} (retry {retry_count}/{max_retries}): {e}")
                await asyncio.sleep(self.interval / 2)
            await asyncio.sleep(self.interval)