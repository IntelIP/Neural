"""Offline, decimal-safe book boundary for paper simulation (NRCL-100).

The normalized format currently accepts synthetic fixtures only. It is not a
venue capture client and does not certify source depth, fees or data rights.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any

from neural.kalshi import BookLevel, OrderBookSnapshot, _ticker, _timestamp
from neural.kalshi_stream import (
    MAX_RECORD,
    BookUpdate,
    StreamEvent,
    _json,
)
from neural.kalshi_stream import (
    RECORD_VERSION as KALSHI_VERSION,
)
from neural.kalshi_stream import (
    replay_book_recording as replay_kalshi_recording,
)
from neural.sports import SportsMarket
from neural.strategy import _decimal

RECORD_VERSION = "neural-book/1"
MAX_SOURCE_AGE_SECONDS = 30


def _records(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, "rb") as source:
        while line := source.readline(MAX_RECORD + 1):
            if len(line) > MAX_RECORD or not line.endswith(b"\n"):
                raise ValueError("oversized or truncated recording line")
            yield _json(line)


def read_recording_metadata(path: str | Path) -> dict[str, Any]:
    """Read and validate the first record only; replay must still validate EOF."""
    records = _records(path)
    try:
        header = next(records, None)
    finally:
        records.close()
    return _metadata(header)


def _metadata(header: dict[str, Any] | None) -> dict[str, Any]:
    if header is None:
        raise ValueError("empty recording")
    if header.get("version") == KALSHI_VERSION:
        return {
            "version": KALSHI_VERSION,
            "venue": "kalshi",
            "market_id": _ticker(header.get("ticker")),
            "outcome": None,
            "sports_market": None,
            "provenance": "legacy_recording_unverified",
        }
    if (
        set(header)
        != {"version", "kind", "venue", "market_id", "outcome", "sports_market", "provenance"}
        or header.get("version") != RECORD_VERSION
        or header.get("kind") != "header"
    ):
        raise ValueError("unsupported recording header or version")
    if header["venue"] not in ("kalshi", "polymarket_us"):
        raise ValueError("unsupported recording venue")
    _ticker(header["market_id"])
    if header["outcome"] != "yes":
        raise ValueError("normalized sports recordings support the YES team-wins outcome only")
    if header["provenance"] != "synthetic":
        raise ValueError("normalized recording currently supports synthetic provenance only")
    market = SportsMarket.from_dict(header["sports_market"])
    if (market.venue, market.market_id) != (header["venue"], header["market_id"]):
        raise ValueError("sports market does not match recording venue/market")
    return {key: value for key, value in header.items() if key != "kind"}


def _at(value: Any, name: str) -> datetime:
    at = _timestamp(value, name)
    if at is None:
        raise ValueError(f"{name} is required")
    return at


def _levels(value: Any, side: str) -> tuple[BookLevel, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("full-depth recording requires both nonempty book sides")
    result = []
    previous = None
    for row in value:
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError("book level must contain price and quantity decimal strings")
        price, quantity = _decimal(row[0], "price"), _decimal(row[1], "quantity")
        if price > 1 or quantity <= 0:
            raise ValueError("book price must be in [0,1] and quantity positive")
        if previous is not None and (
            (side == "bids" and price >= previous) or (side == "asks" and price <= previous)
        ):
            raise ValueError("book levels must have unique prices in executable order")
        result.append(BookLevel(price, quantity))
        previous = price
    return tuple(result)


def replay_book_recording(
    path: str | Path, *, expected_metadata: dict[str, Any] | None = None
) -> Iterator[StreamEvent]:
    """Replay either format; exhaust the iterator to validate the final boundary.

    Normalized records contain full bid/ask ladders for the named YES team-wins
    proposition only. The existing binary book type supplies the boundary.
    """
    records = _records(path)
    try:
        metadata = _metadata(next(records, None))
        if expected_metadata is not None and metadata != expected_metadata:
            raise ValueError("recording metadata changed before replay")
        yield from _replay_records(path, records, metadata)
    finally:
        records.close()


def _replay_records(
    path: str | Path, records: Iterator[dict[str, Any]], metadata: dict[str, Any]
) -> Iterator[StreamEvent]:
    if metadata["version"] == KALSHI_VERSION:
        for event in replay_kalshi_recording(path):
            if event.update is not None and event.update.book.ticker != metadata["market_id"]:
                raise ValueError("recording market changed before replay")
            yield event
        return
    active = False
    session = sequence = 0
    previous: datetime | None = None
    previous_source: datetime | None = None
    for record in records:
        kind = record.get("kind")
        extra = (
            {"reason"} if kind == "reset" else {"source_at", "sequence", "quality", "bids", "asks"}
        )
        if set(record) != {"version", "kind", "received_at"} | extra or (
            record.get("version") != RECORD_VERSION
        ):
            raise ValueError("unsupported normalized record shape or version")
        at = _at(record["received_at"], "received_at")
        if previous is not None and at < previous:
            raise ValueError("recording receive timestamps must not regress")
        previous = at
        if kind == "reset":
            if record["reason"] not in ("connecting", "disconnected"):
                raise ValueError("invalid reset reason")
            connecting = record["reason"] == "connecting"
            if connecting == active:
                raise ValueError("invalid recording session boundary")
            active = connecting
            if connecting:
                session += 1
                sequence = 0
                previous_source = None
            yield StreamEvent("reset", at, reason=record["reason"])
            continue
        if kind != "book" or not active:
            raise ValueError("book outside recording session or unknown kind")
        if record["quality"] != "full_depth":
            raise ValueError("paper replay requires explicit full_depth quality")
        if type(record["sequence"]) is not int or record["sequence"] != sequence + 1:
            raise ValueError("recording sequence gap or regression")
        sequence = record["sequence"]
        source_at = _at(record["source_at"], "source_at")
        age = (at - source_at).total_seconds()
        if age < 0 or age > MAX_SOURCE_AGE_SECONDS:
            raise ValueError("recorded source timestamp is future or stale")
        if previous_source is not None and source_at < previous_source:
            raise ValueError("recording source timestamps must not regress within a session")
        previous_source = source_at
        bids, asks = _levels(record["bids"], "bids"), _levels(record["asks"], "asks")
        if bids[0].price > asks[0].price:
            raise ValueError("crossed recorded book")
        with localcontext() as context:
            context.prec = 80
            opposite = tuple(BookLevel(Decimal(1) - level.price, level.quantity) for level in asks)
        book = OrderBookSnapshot(metadata["market_id"], bids, opposite, at, 0)
        yield StreamEvent("book", at, BookUpdate(book, session, sequence, source_at))
    if active or previous is None:
        raise ValueError("empty or incomplete recording: terminal reset required")


def describe_recording(path: str | Path, *, max_events: int = 10000) -> dict[str, Any]:
    """Return metadata and bounded summary only after validating the entire file."""
    if type(max_events) is not int or max_events <= 0:
        raise ValueError("max_events must be a positive integer")
    result = read_recording_metadata(path)
    count = books = resets = connections = disconnects = 0
    start = end = last_source = None
    for count, event in enumerate(replay_book_recording(path, expected_metadata=result), 1):
        if count > max_events:
            raise ValueError("recording exceeds max_events")
        if end is not None and event.received_at < end:
            raise ValueError("recording receive timestamps must not regress")
        start = start or event.received_at
        end = event.received_at
        if event.kind == "book":
            books += 1
            assert event.update is not None
            last_source = event.update.source_at
        else:
            resets += 1
            connections += event.reason == "connecting"
            disconnects += event.reason == "disconnected"
    if not books:
        raise ValueError("recording contains no books")
    result.update(
        event_count=count,
        book_count=books,
        reset_count=resets,
        reconnects=max(0, connections - 1),
        disconnects=disconnects,
        start_at=start.isoformat() if start else None,
        end_at=end.isoformat() if end else None,
        source_at=last_source.isoformat() if last_source else None,
    )
    return result
