from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, Any, Optional, List
from dataclasses import dataclass
import asyncio


@dataclass
class DataSourceConfig:
    """Configuration for data sources."""
    name: str
    config: Optional[Dict[str, Any]] = None


class BaseDataSource(ABC):
    """Abstract base class for all data sources."""

    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.name = config.name
        self._connected = False

    @abstractmethod
    async def _connect_impl(self) -> bool:
        """Implementation-specific connection logic."""
        pass

    @abstractmethod
    async def _disconnect_impl(self) -> None:
        """Implementation-specific disconnection logic."""
        pass

    @abstractmethod
    async def _subscribe_impl(self, channels: List[str]) -> bool:
        """Implementation-specific subscription logic."""
        pass

    async def connect(self) -> bool:
        """Connect to the data source."""
        if not self._connected:
            self._connected = await self._connect_impl()
        return self._connected

    async def disconnect(self) -> None:
        """Disconnect from the data source."""
        if self._connected:
            await self._disconnect_impl()
            self._connected = False


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