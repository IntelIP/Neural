from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, Any, Optional
import asyncio


class DataSource(ABC):
    """Base class for all data sources in the neural SDK."""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self._connected = False

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the data source."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the data source."""
        pass

    @abstractmethod
    async def collect(self):
        """Collect data from the source. Should yield data."""
        pass

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()