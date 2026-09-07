"""Current Kalshi read-only data interface (NRCL-87).

Explicitly separate from the legacy float-valued trading adapter. No order
submission, credential discovery, or implicit conversion from integer cents.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from time import sleep
from typing import Any
from urllib.parse import quote

import requests

BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
DATA_VERSION = "1.0.0"


class KalshiDataError(ValueError):
    """A response cannot safely represent current market state."""


def _number(value: Any, name: str, places: int, *, price: bool = False) -> Decimal:
    pattern = rf"(?:0|[1-9][0-9]{{0,19}})(?:\.[0-9]{{1,{places}}})?"
    if not isinstance(value, str) or re.fullmatch(pattern, value) is None:
        raise KalshiDataError(f"{name}: expected fixed-point string with <= {places} decimals")
    result = Decimal(value)
    if price and result > 1:
        raise KalshiDataError(f"{name}: binary price outside [0, 1]")
    return result


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 4096:
        raise KalshiDataError(f"{name}: expected nonempty string")
    return value


def _ticker(value: Any) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}", value) is None
    ):
        raise KalshiDataError("ticker: invalid native identifier")
    return value


def _timestamp(value: Any, name: str) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(_text(value, name).replace("Z", "+00:00"))
        if parsed.utcoffset() is None:
            raise ValueError("timezone required")
        return parsed.astimezone(timezone.utc)
    except ValueError as exc:
        raise KalshiDataError(f"{name}: expected timezone-aware ISO timestamp") from exc


def _received(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise KalshiDataError("received_at: timezone-aware datetime required")
    return value.astimezone(timezone.utc)


def _wire(value: Any) -> Any:
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _wire(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_wire(item) for item in value]
    return value


@dataclass(frozen=True)
class PriceRange:
    start: Decimal
    end: Decimal
    step: Decimal


@dataclass(frozen=True)
class MarketSnapshot:
    ticker: str
    event_ticker: str
    title: str
    status: str
    yes_bid: Decimal | None
    yes_ask: Decimal | None
    no_bid: Decimal | None
    no_ask: Decimal | None
    last_price: Decimal | None
    volume: Decimal | None
    price_ranges: tuple[PriceRange, ...]
    exchange_index: int | None
    updated_at: datetime | None
    received_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": DATA_VERSION, "venue": "kalshi", **_wire(asdict(self))}


@dataclass(frozen=True)
class BookLevel:
    price: Decimal
    quantity: Decimal


@dataclass(frozen=True)
class OrderBookSnapshot:
    ticker: str
    yes_bids: tuple[BookLevel, ...]
    no_bids: tuple[BookLevel, ...]
    received_at: datetime
    requested_depth: int

    def asks(self, outcome: str) -> tuple[BookLevel, ...]:
        """Opposite bids imply asks, preserving quantities and ascending prices."""
        if outcome not in ("yes", "no"):
            raise ValueError("outcome must be yes or no")
        bids = self.no_bids if outcome == "yes" else self.yes_bids
        with localcontext() as context:
            context.prec = 40
            return tuple(BookLevel(Decimal(1) - level.price, level.quantity) for level in bids)

    def to_dict(self) -> dict[str, Any]:
        # REST book response provides no exchange event timestamp/sequence.
        return {"schema_version": DATA_VERSION, "venue": "kalshi", **_wire(asdict(self))}


def parse_market(raw: Any, *, received_at: datetime) -> MarketSnapshot:
    if not isinstance(raw, dict):
        raise KalshiDataError("market: expected object")
    if (
        raw.get("market_type") != "binary"
        or raw.get("mve_collection_ticker")
        or raw.get("mve_selected_legs")
    ):
        raise KalshiDataError("only standard binary event markets are supported")

    def optional(field: str, places: int, price: bool = False) -> Decimal | None:
        legacy = field.removesuffix("_dollars").removesuffix("_fp")
        if field not in raw and legacy in raw:
            raise KalshiDataError(f"{field}: legacy-only response is unsupported")
        value = raw.get(field)
        return None if value is None else _number(value, field, places, price=price)

    ranges = raw.get("price_ranges")
    if not isinstance(ranges, list):
        raise KalshiDataError("price_ranges: expected array; never infer a tick size")
    parsed_ranges = []
    for band in ranges:
        if not isinstance(band, dict) or not {"start", "end", "step"} <= band.keys():
            raise KalshiDataError("price_ranges: incomplete band")
        start, end, step = (_number(band[k], k, 4, price=True) for k in ("start", "end", "step"))
        if start >= end or step == 0:
            raise KalshiDataError("price_ranges: invalid band bounds or step")
        parsed_ranges.append(PriceRange(start, end, step))
    parsed_ranges.sort(key=lambda band: band.start)
    if any(
        left.end > right.start
        for left, right in zip(parsed_ranges, parsed_ranges[1:], strict=False)
    ):
        raise KalshiDataError("price_ranges: overlapping bands")
    shard = raw.get("exchange_index")
    if shard is not None and (type(shard) is not int or shard < 0):
        raise KalshiDataError("exchange_index: expected nonnegative integer")
    return MarketSnapshot(
        ticker=_ticker(raw.get("ticker")),
        event_ticker=_ticker(raw.get("event_ticker")),
        title=_text(raw.get("title"), "title"),
        status=_text(raw.get("status"), "status"),
        yes_bid=optional("yes_bid_dollars", 4, True),
        yes_ask=optional("yes_ask_dollars", 4, True),
        no_bid=optional("no_bid_dollars", 4, True),
        no_ask=optional("no_ask_dollars", 4, True),
        last_price=optional("last_price_dollars", 4, True),
        volume=optional("volume_fp", 2),
        price_ranges=tuple(parsed_ranges),
        exchange_index=shard,
        updated_at=_timestamp(raw.get("updated_time"), "updated_time"),
        received_at=_received(received_at),
    )


def parse_orderbook(
    payload: Any, *, ticker: str, received_at: datetime, depth: int = 0
) -> OrderBookSnapshot:
    _depth(depth)
    book = payload.get("orderbook_fp") if isinstance(payload, dict) else None
    if not isinstance(book, dict):
        raise KalshiDataError("orderbook_fp: required; legacy cent books are unsupported")

    def levels(side: str) -> tuple[BookLevel, ...]:
        rows = book.get(side)
        if not isinstance(rows, list):
            raise KalshiDataError(f"{side}: required array (empty array means no liquidity)")
        result = []
        seen = set()
        for row in rows:
            if not isinstance(row, list) or len(row) != 2:
                raise KalshiDataError(f"{side}: expected [price, quantity]")
            price = _number(row[0], "price", 4, price=True)
            quantity = _number(row[1], "quantity", 2)
            if quantity == 0 or price in seen:
                raise KalshiDataError(f"{side}: zero quantity or duplicate price level")
            seen.add(price)
            result.append(BookLevel(price, quantity))
        return tuple(sorted(result, key=lambda level: level.price, reverse=True))

    return OrderBookSnapshot(
        _ticker(ticker), levels("yes_dollars"), levels("no_dollars"), _received(received_at), depth
    )


def _depth(value: int) -> None:
    if type(value) is not int or not 0 <= value <= 100:
        raise ValueError("depth must be an integer in [0, 100]")


class KalshiDataClient:
    """Bounded synchronous GET client. Use outside an async execution loop.

    No environment credentials are loaded. Optional signer_headers explicitly
    supplies authentication for venues/accounts requiring it on book reads.
    Injected sessions are caller-owned; default sessions ignore netrc/proxy env.
    """

    def __init__(
        self,
        *,
        timeout: float = 10,
        max_retries: int = 2,
        session: Any = None,
        signer_headers: Callable[[str, str], Mapping[str, str]] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if isinstance(timeout, bool) or not math.isfinite(timeout) or not 0 < timeout <= 60:
            raise ValueError("timeout must be finite and in (0, 60]")
        if type(max_retries) is not int or not 0 <= max_retries <= 3:
            raise ValueError("max_retries must be an integer in [0, 3]")
        self.timeout = timeout
        self.max_retries = max_retries
        self._owned_session = session is None
        self._session = requests.Session() if session is None else session
        if self._owned_session:
            self._session.trust_env = False
        self._signer_headers = signer_headers
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._closed = False

    def _get(
        self, path: str, params: dict[str, Any], *, authenticated: bool = False
    ) -> tuple[dict[str, Any], datetime]:
        if self._closed:
            raise RuntimeError("client is closed")
        for attempt in range(self.max_retries + 1):
            headers = (
                dict(self._signer_headers("GET", "/trade-api/v2" + path))
                if authenticated and self._signer_headers
                else {}
            )
            try:
                response = self._session.get(
                    BASE_URL + path,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                    allow_redirects=False,
                )
            except (requests.Timeout, requests.ConnectionError):
                if attempt == self.max_retries:
                    raise
                sleep(0.25 * 2**attempt)
                continue
            try:
                if response.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    retry_after = response.headers.get("Retry-After")
                    delay = 0.25 * 2**attempt
                    if retry_after is not None:
                        # Do not retry earlier than a long or unrecognized server directive.
                        if not retry_after.isdigit() or int(retry_after) > 30:
                            response.raise_for_status()
                        delay = float(retry_after)
                else:
                    response.raise_for_status()
                    if response.status_code != 200:
                        raise KalshiDataError("expected HTTP 200; redirects are not followed")
                    received_at = _received(self._clock())
                    payload = response.json()
                    if not isinstance(payload, dict):
                        raise KalshiDataError("expected JSON object")
                    return payload, received_at
            finally:
                response.close()
            sleep(delay)
        raise RuntimeError("unreachable retry state")

    def get_market(self, ticker: str) -> MarketSnapshot:
        ticker = _ticker(ticker)
        payload, received = self._get("/markets/" + quote(ticker, safe=""), {})
        market = parse_market(payload.get("market"), received_at=received)
        if market.ticker != ticker:
            raise KalshiDataError("response ticker does not match request")
        return market

    def iter_markets(
        self,
        *,
        status: str | None = "open",
        series_ticker: str | None = None,
        page_size: int = 100,
        max_pages: int = 10,
    ) -> Iterator[MarketSnapshot]:
        if status not in (None, "unopened", "open", "paused", "closed", "settled"):
            raise ValueError("unsupported market status filter")
        if type(page_size) is not int or not 1 <= page_size <= 1000:
            raise ValueError("page_size must be in [1, 1000]")
        if type(max_pages) is not int or not 1 <= max_pages <= 1000:
            raise ValueError("max_pages must be in [1, 1000]")
        params: dict[str, Any] = {"limit": page_size, "mve_filter": "exclude"}
        if status is not None:
            params["status"] = status
        if series_ticker is not None:
            params["series_ticker"] = _ticker(series_ticker)
        cursors: set[str] = set()
        for _ in range(max_pages):
            payload, received = self._get("/markets", dict(params))
            rows = payload.get("markets")
            if not isinstance(rows, list):
                raise KalshiDataError("markets: required array")
            for row in rows:
                yield parse_market(row, received_at=received)
            cursor = payload.get("cursor")
            if cursor == "":
                return
            _text(cursor, "cursor")
            if cursor in cursors:
                raise KalshiDataError("repeated pagination cursor")
            cursors.add(cursor)
            params["cursor"] = cursor
        raise KalshiDataError("max_pages reached; discovery is incomplete")

    def get_orderbook(self, ticker: str, *, depth: int = 0) -> OrderBookSnapshot:
        _depth(depth)
        ticker = _ticker(ticker)
        payload, received = self._get(
            "/markets/" + quote(ticker, safe="") + "/orderbook",
            {"depth": depth},
            authenticated=True,
        )
        return parse_orderbook(payload, ticker=ticker, received_at=received, depth=depth)

    def close(self) -> None:
        if not self._closed and self._owned_session:
            self._session.close()
        self._closed = True

    def __enter__(self) -> KalshiDataClient:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def snapshot_json(snapshot: MarketSnapshot | OrderBookSnapshot) -> str:
    """Portable decimal-string recording; timestamps remain part of the data."""
    return json.dumps(snapshot.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
