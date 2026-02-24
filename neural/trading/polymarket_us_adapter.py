from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from neural.auth.polymarket_us_env import (
    get_polymarket_us_base_url,
    get_polymarket_us_credentials,
)
from neural.auth.signers.polymarket_us import PolymarketUSSigner
from neural.exchanges.base import BaseExchangeAdapter
from neural.exchanges.types import (
    ExchangeCapabilities,
    NormalizedMarket,
    NormalizedOrderRequest,
    NormalizedOrderResult,
    NormalizedPosition,
    NormalizedQuote,
    TradingPolicy,
)

_LOG = logging.getLogger(__name__)


@dataclass(slots=True)
class PolymarketUSAdapter(BaseExchangeAdapter):
    """Polymarket US adapter with sports-first defaults."""

    api_key: str | None = None
    api_secret: bytes | None = None
    passphrase: str | None = None
    base_url: str | None = None
    timeout: int = 30
    max_retries: int = 3
    session: requests.Session | None = None

    name: str = "polymarket_us"
    _http: requests.Session = field(init=False)
    _signer: PolymarketUSSigner = field(init=False)

    def __post_init__(self) -> None:
        BaseExchangeAdapter.__init__(self)
        creds: dict[str, Any] = {}
        if self.api_key is None or self.api_secret is None or self.passphrase is None:
            creds = get_polymarket_us_credentials()
        self.api_key = self.api_key or str(creds["api_key"])
        self.api_secret = self.api_secret or bytes(creds["api_secret"])
        self.passphrase = self.passphrase or str(creds["passphrase"])
        self.base_url = (self.base_url or get_polymarket_us_base_url()).rstrip("/")

        self._signer = PolymarketUSSigner(
            api_key=self.api_key,
            api_secret=self.api_secret,
            passphrase=self.passphrase,
        )
        self._http = self.session or requests.Session()

    def capabilities(self) -> ExchangeCapabilities:
        return ExchangeCapabilities(read=True, paper=False, live=False, streaming=False)

    def list_markets(
        self, *, sport: str | None = None, limit: int = 100, sports_only: bool = True
    ) -> list[NormalizedMarket]:
        path = "/api/v1/sports/markets" if sports_only else "/api/v1/markets"
        params: dict[str, Any] = {"limit": limit}
        if sport:
            params["sport"] = sport

        payload = self._request("GET", path, params=params)
        rows = _extract_rows(payload)
        out = [self._normalize_market(r) for r in rows]

        if sports_only:
            out = [m for m in out if _is_sports_market(m)]
        if sport:
            out = [m for m in out if (m.sport or "").lower() == sport.lower()]
        return out

    def get_quote(self, market_id: str) -> NormalizedQuote:
        path = f"/api/v1/markets/{market_id}/book"
        payload = self._request("GET", path)
        row = payload.get("book") or payload.get("data") or payload
        yes_bid = _to_prob(row.get("yes_bid") or row.get("best_bid_yes") or row.get("bid"))
        yes_ask = _to_prob(row.get("yes_ask") or row.get("best_ask_yes") or row.get("ask"))
        no_bid = _to_prob(row.get("no_bid") or row.get("best_bid_no"))
        no_ask = _to_prob(row.get("no_ask") or row.get("best_ask_no"))

        if no_bid is None and yes_ask is not None:
            no_bid = 1.0 - yes_ask
        if no_ask is None and yes_bid is not None:
            no_ask = 1.0 - yes_bid

        return NormalizedQuote(
            market_id=market_id,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=no_bid,
            no_ask=no_ask,
            last_price=_to_prob(row.get("last_price") or row.get("last")),
            volume=_to_float(row.get("volume")),
            timestamp=_to_float(row.get("timestamp") or row.get("ts")),
            metadata={"exchange": self.name, "raw": row},
        )

    def place_order(
        self,
        order: NormalizedOrderRequest,
        *,
        policy: TradingPolicy | None = None,
    ) -> NormalizedOrderResult:
        raise NotImplementedError(
            "Polymarket US live order placement is introduced in a later PR."
        )

    def cancel_order(self, order_id: str) -> NormalizedOrderResult:
        raise NotImplementedError(
            "Polymarket US order cancellation is introduced in a later PR."
        )

    def get_order_status(self, order_id: str) -> NormalizedOrderResult:
        raise NotImplementedError(
            "Polymarket US order status retrieval is introduced in a later PR."
        )

    def get_positions(self) -> list[NormalizedPosition]:
        raw = self._request("GET", "/api/v1/portfolio/positions")
        rows = _extract_rows(raw)
        out: list[NormalizedPosition] = []
        for row in rows:
            out.append(
                NormalizedPosition(
                    market_id=str(row.get("market_id") or row.get("market") or ""),
                    side="yes" if str(row.get("side", "yes")).lower() == "yes" else "no",
                    quantity=int(row.get("quantity") or row.get("size") or 0),
                    entry_price=_to_prob(row.get("entry_price") or row.get("avg_price")),
                    current_price=_to_prob(row.get("current_price") or row.get("mark_price")),
                    unrealized_pnl=_to_float(row.get("unrealized_pnl") or row.get("pnl")),
                    metadata={"exchange": self.name, "raw": row},
                )
            )
        return out

    def close(self) -> None:
        self._http.close()

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = json.dumps(json_data, separators=(",", ":")) if json_data else ""
        headers = {
            "Content-Type": "application/json",
            **self._signer.headers(method, path, body=body),
        }

        url = f"{self.base_url}{path}"
        retry = 0
        while True:
            response = self._http.request(
                method=method,
                url=url,
                params=params,
                data=body if body else None,
                headers=headers,
                timeout=self.timeout,
            )
            if response.status_code < 400:
                data = response.json() if response.text else {}
                if isinstance(data, dict):
                    return data
                return {"data": data}

            if response.status_code in (429, 500, 502, 503, 504) and retry < self.max_retries:
                retry += 1
                sleep_s = 2**retry
                _LOG.warning(
                    "Polymarket US request failed (%s), retrying in %ss (%s/%s)",
                    response.status_code,
                    sleep_s,
                    retry,
                    self.max_retries,
                )
                time.sleep(sleep_s)
                continue

            response.raise_for_status()

    @staticmethod
    def _normalize_market(raw: dict[str, Any]) -> NormalizedMarket:
        market_id = str(raw.get("id") or raw.get("market_id") or raw.get("slug") or "")
        title = str(raw.get("title") or raw.get("question") or raw.get("name") or market_id)
        category = str(raw.get("category") or raw.get("topic") or "sports")
        sport = str(raw.get("sport") or raw.get("league") or category)

        yes_price = _to_prob(raw.get("yes_price") or raw.get("price_yes") or raw.get("best_ask_yes"))
        no_price = _to_prob(raw.get("no_price") or raw.get("price_no") or raw.get("best_ask_no"))
        if no_price is None and yes_price is not None:
            no_price = 1.0 - yes_price

        return NormalizedMarket(
            market_id=market_id,
            ticker=str(raw.get("ticker") or raw.get("slug") or market_id),
            title=title,
            status=str(raw.get("status") or "open"),
            yes_price=yes_price,
            no_price=no_price,
            category=category,
            sport=sport,
            metadata={"exchange": "polymarket_us", "raw": raw},
        )


def _extract_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(payload.get("data"), list):
        return [r for r in payload["data"] if isinstance(r, dict)]
    if isinstance(payload.get("markets"), list):
        return [r for r in payload["markets"] if isinstance(r, dict)]
    if isinstance(payload.get("positions"), list):
        return [r for r in payload["positions"] if isinstance(r, dict)]
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if payload:
        return [payload]
    return []


def _is_sports_market(market: NormalizedMarket) -> bool:
    haystack = f"{market.category or ''} {market.sport or ''} {market.title}".lower()
    return "sport" in haystack or any(
        k in haystack
        for k in ("nfl", "nba", "mlb", "nhl", "soccer", "football", "basketball", "baseball")
    )


def _to_prob(value: Any) -> float | None:
    if value is None:
        return None
    out = float(value)
    if out > 1:
        return out / 100.0
    return out


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
