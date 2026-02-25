from __future__ import annotations

import asyncio
from dataclasses import dataclass, fields
from typing import Any

import pandas as pd

from neural.trading.polymarket_us_adapter import PolymarketUSAdapter

from .base import DataSource


@dataclass(slots=True)
class PolymarketUSConfig:
    sport: str | None = None
    limit: int = 100
    sports_only: bool = True
    poll_interval: float = 5.0


class PolymarketUSMarketsSource(DataSource):
    """Polling source for Polymarket US market snapshots."""

    def __init__(
        self,
        name: str = "polymarket_us_markets",
        *,
        config: dict[str, Any] | None = None,
        adapter: PolymarketUSAdapter | None = None,
    ) -> None:
        super().__init__(name=name, config=config)
        self.adapter = adapter or PolymarketUSAdapter()
        raw_config = config or {}
        allowed_keys = {f.name for f in fields(PolymarketUSConfig)}
        source_cfg = {k: v for k, v in raw_config.items() if k in allowed_keys}
        self._source_cfg = PolymarketUSConfig(**source_cfg)

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False
        self.adapter.close()

    async def collect(self):
        while self._connected:
            markets = self.adapter.list_markets(
                sport=self._source_cfg.sport,
                limit=self._source_cfg.limit,
                sports_only=self._source_cfg.sports_only,
            )
            yield [m.metadata.get("raw", {}) for m in markets]
            await asyncio.sleep(self._source_cfg.poll_interval)

    def get_markets_df(self) -> pd.DataFrame:
        markets = self.adapter.list_markets(
            sport=self._source_cfg.sport,
            limit=self._source_cfg.limit,
            sports_only=self._source_cfg.sports_only,
        )
        rows = [
            {
                "market_id": m.market_id,
                "ticker": m.ticker,
                "title": m.title,
                "status": m.status,
                "yes_price": m.yes_price,
                "no_price": m.no_price,
                "sport": m.sport,
                "category": m.category,
            }
            for m in markets
        ]
        return pd.DataFrame(rows)

    def get_market_history(
        self,
        market_id: str,
        interval: str = "1h",
        limit: int = 200,
    ) -> pd.DataFrame:
        # Path is intentionally isolated here so endpoint changes are easy to update.
        payload = self.adapter._request(  # noqa: SLF001
            "GET",
            f"/api/v1/markets/{market_id}/candles",
            params={"interval": interval, "limit": limit},
        )
        rows = payload.get("candles") or payload.get("data") or []
        if not isinstance(rows, list):
            rows = []

        normalized = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            normalized.append(
                {
                    "timestamp": row.get("timestamp") or row.get("time"),
                    "open": row.get("open"),
                    "high": row.get("high"),
                    "low": row.get("low"),
                    "close": row.get("close"),
                    "volume": row.get("volume"),
                }
            )

        df = pd.DataFrame(normalized)
        if not df.empty and "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="ignore")
        return df
