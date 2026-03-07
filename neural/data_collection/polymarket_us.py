from __future__ import annotations

import asyncio
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

from neural.trading.polymarket_us_adapter import PolymarketUSAdapter

from .base import DataSource


@dataclass(slots=True)
class PolymarketUSConfig:
    sport: str | None = None
    limit: int = 100
    sports_only: bool = True
    poll_interval: float = 5.0
    replay_page_size: int = 500


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
        if adapter is not None:
            self.adapter = adapter
        else:
            try:
                self.adapter = PolymarketUSAdapter()
            except Exception as exc:
                raise RuntimeError(
                    "Failed to initialize PolymarketUSMarketsSource adapter"
                ) from exc
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
        import pandas as pd

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
        start_ts_ms: int | None = None,
        end_ts_ms: int | None = None,
    ) -> pd.DataFrame:
        import pandas as pd

        rows = self.adapter.get_candles(
            market_id,
            interval=interval,
            limit=limit,
            start_ts_ms=start_ts_ms,
            end_ts_ms=end_ts_ms,
        )

        normalized = []
        for row in rows:
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
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
        return df

    def get_market_replay_df(
        self,
        market_id: str,
        *,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> tuple[pd.DataFrame, str | None]:
        import pandas as pd

        page_size = limit or self._source_cfg.replay_page_size
        payload = self.adapter.get_trade_replay(market_id, limit=page_size, cursor=cursor)
        rows = payload.get("items", [])

        normalized = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            normalized.append(
                {
                    "timestamp": row.get("timestamp") or row.get("time"),
                    "market_id": row.get("market_id") or row.get("market"),
                    "side": row.get("side"),
                    "price": row.get("price"),
                    "size": row.get("size") or row.get("quantity"),
                    "trade_id": row.get("trade_id") or row.get("id"),
                    "sequence": row.get("sequence") or row.get("seq"),
                }
            )

        df = pd.DataFrame(normalized)
        if not df.empty and "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
        return df, payload.get("next_cursor")

    def iter_market_replay(
        self,
        market_id: str,
        *,
        limit_per_page: int | None = None,
        max_pages: int | None = None,
    ):
        cursor: str | None = None
        pages = 0
        while True:
            df, next_cursor = self.get_market_replay_df(
                market_id,
                limit=limit_per_page,
                cursor=cursor,
            )
            yield df
            pages += 1

            if not next_cursor:
                break
            if max_pages is not None and pages >= max_pages:
                break
            cursor = next_cursor

    def get_market_events_df(
        self,
        market_id: str,
        *,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> tuple[pd.DataFrame, str | None]:
        import pandas as pd

        page_size = limit or self._source_cfg.replay_page_size
        payload = self.adapter.get_market_events(market_id, limit=page_size, cursor=cursor)
        rows = payload.get("items", [])

        normalized = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            normalized.append(
                {
                    "timestamp": row.get("timestamp") or row.get("time"),
                    "market_id": row.get("market_id") or row.get("market"),
                    "event_type": row.get("event_type") or row.get("type"),
                    "payload": row,
                    "sequence": row.get("sequence") or row.get("seq"),
                }
            )

        df = pd.DataFrame(normalized)
        if not df.empty and "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
        return df, payload.get("next_cursor")
