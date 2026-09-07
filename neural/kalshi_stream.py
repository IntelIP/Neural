"""Single-market Kalshi book recovery and recording (NRCL-88).

Read-only companion to neural.kalshi; not a trading or strategy runner.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any

from neural.kalshi import (
    BookLevel,
    KalshiDataError,
    OrderBookSnapshot,
    _number,
    _received,
    _ticker,
    _timestamp,
    parse_orderbook,
)

WS_PATH = "/trade-api/ws/v2"
WS_URL = "wss://external-api-ws.kalshi.com" + WS_PATH
DEMO_WS_URL = "wss://external-api-ws.demo.kalshi.co" + WS_PATH
RECORD_VERSION = "kalshi-book/1"
MAX_RECORD = 2 * 1024 * 1024


class RecoveryRequired(KalshiDataError):
    """Discard the current book and establish a new snapshot boundary."""


def _integer(value: Any, name: str) -> int:
    if type(value) is not int or not 0 <= value < 2**63:
        raise KalshiDataError(f"{name}: nonnegative integer required")
    return value


def _json(text: str | bytes) -> dict[str, Any]:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise KalshiDataError("duplicate JSON key")
            result[key] = value
        return result

    def constant(value):
        raise KalshiDataError("non-finite JSON number")

    try:
        value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise KalshiDataError("invalid JSON") from exc
    if not isinstance(value, dict):
        raise KalshiDataError("JSON object required")
    return value


@dataclass(frozen=True)
class BookUpdate:
    book: OrderBookSnapshot
    sid: int
    seq: int
    source_at: datetime | None


@dataclass(frozen=True)
class StreamEvent:
    kind: str
    received_at: datetime
    update: BookUpdate | None = None
    reason: str | None = None


class KalshiBook:
    """Sequence-valid state for exactly one subscription and market.

    A valid sequence is not proof of market freshness or tradability.
    """

    def __init__(self, ticker: str) -> None:
        self.ticker = _ticker(ticker)
        self.latest: BookUpdate | None = None

    def reset(self) -> None:
        self.latest = None

    def apply(self, frame: Any, *, received_at: datetime) -> BookUpdate:
        try:
            update = self._apply(frame, _received(received_at))
        except (KalshiDataError, OverflowError) as exc:
            self.reset()
            raise RecoveryRequired(str(exc)) from exc
        self.latest = update
        return update

    def _apply(self, frame: Any, received_at: datetime) -> BookUpdate:
        if not isinstance(frame, dict) or not isinstance(frame.get("msg"), dict):
            raise KalshiDataError("book frame and msg must be objects")
        sid = _integer(frame.get("sid"), "sid")
        seq = _integer(frame.get("seq"), "seq")
        msg = frame["msg"]
        if msg.get("market_ticker") != self.ticker:
            raise KalshiDataError("unexpected market in single-market subscription")
        if self.latest and (sid != self.latest.sid or seq != self.latest.seq + 1):
            raise KalshiDataError("subscription changed or sequence discontinuity")
        if frame.get("type") == "orderbook_snapshot":
            book = parse_orderbook(
                {
                    "orderbook_fp": {
                        "yes_dollars": msg.get("yes_dollars_fp"),
                        "no_dollars": msg.get("no_dollars_fp"),
                    }
                },
                ticker=self.ticker,
                received_at=received_at,
            )
            return BookUpdate(book, sid, seq, None)
        if frame.get("type") != "orderbook_delta" or self.latest is None:
            raise KalshiDataError("delta requires a preceding snapshot")
        side = msg.get("side")
        if side not in ("yes", "no"):
            raise KalshiDataError("unknown book side")
        price = _number(msg.get("price_dollars"), "price_dollars", 4, price=True)
        value = msg.get("delta_fp")
        if not isinstance(value, str):
            raise KalshiDataError("delta_fp: signed fixed-point string required")
        negative = value.startswith("-")
        delta = _number(value[1:] if negative else value, "delta_fp", 2)
        source_at = _timestamp(msg.get("ts"), "ts")
        if msg.get("ts_ms") is not None:
            ms = _integer(msg["ts_ms"], "ts_ms")
            precise = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(milliseconds=ms)
            if source_at is not None and abs(precise - source_at) >= timedelta(seconds=1):
                raise KalshiDataError("conflicting source timestamps")
            source_at = precise
        old = self.latest.book
        levels = {level.price: level.quantity for level in getattr(old, side + "_bids")}
        with localcontext() as context:
            context.prec = 50
            quantity = levels.get(price, Decimal(0)) + (-delta if negative else delta)
        _number(format(quantity, "f"), "resulting quantity", 2)
        if quantity == 0:
            levels.pop(price, None)
        else:
            levels[price] = quantity
        changed = tuple(BookLevel(p, q) for p, q in sorted(levels.items(), reverse=True))
        book = OrderBookSnapshot(
            self.ticker,
            changed if side == "yes" else old.yes_bids,
            changed if side == "no" else old.no_bids,
            received_at,
            0,
        )
        return BookUpdate(book, sid, seq, source_at)


class BookRecording:
    """Exclusive-create JSONL file; flush before publishing every event.

    durable=True also fsyncs each record. No append, overwrite, or tail repair.
    """

    def __init__(self, path: str | Path, *, durable: bool = False) -> None:
        self._file = open(path, "x", encoding="utf-8")
        self._durable = durable

    def write(self, record: dict[str, Any]) -> None:
        try:
            line = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
            if len(line.encode("utf-8")) + 1 > MAX_RECORD:
                raise ValueError("record too large")
            self._file.write(line + "\n")
            self._file.flush()
            if self._durable:
                os.fsync(self._file.fileno())
        except (OSError, ValueError) as exc:
            raise RuntimeError("recording failed; stop stream") from exc

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> BookRecording:
        return self

    def __exit__(self, *args) -> None:
        self.close()


def replay_book_recording(path: str | Path) -> Iterator[StreamEvent]:
    """Replay file order, including invalidations; reject corrupt/incomplete files.

    Exhaust the iterator to validate the final boundary. Not legacy price replay.
    """
    state = None
    last_kind = None
    active = False
    with open(path, "rb") as source:
        while line := source.readline(MAX_RECORD + 1):
            if len(line) > MAX_RECORD or not line.endswith(b"\n"):
                raise KalshiDataError("oversized or truncated recording line")
            record = _json(line)
            kind = record.get("kind")
            keys = {
                "version",
                "ticker",
                "kind",
                "received_at",
                "reason" if kind == "reset" else "frame",
            }
            if set(record) != keys or record.get("version") != RECORD_VERSION:
                raise KalshiDataError("unsupported recording shape or version")
            ticker = _ticker(record.get("ticker"))
            at = _timestamp(record.get("received_at"), "received_at")
            if at is None:
                raise KalshiDataError("recording timestamp required")
            if state is None:
                if kind != "reset":
                    raise KalshiDataError("recording must start with reset")
                state = KalshiBook(ticker)
            if ticker != state.ticker:
                raise KalshiDataError("recording market changed")
            if kind == "reset":
                if record["reason"] not in ("connecting", "disconnected"):
                    raise KalshiDataError("invalid reset reason")
                connecting = record["reason"] == "connecting"
                if connecting == active:
                    raise KalshiDataError("invalid recording session boundary")
                active = connecting
                state.reset()
                yield StreamEvent("reset", at, reason=record["reason"])
            elif kind == "frame":
                if not active:
                    raise KalshiDataError("frame outside recording session")
                yield StreamEvent("book", at, state.apply(record["frame"], received_at=at))
            else:
                raise KalshiDataError("unknown record kind")
            last_kind = kind
    if last_kind != "reset" or active:
        raise KalshiDataError("empty or incomplete recording: terminal reset required")


def _connect(url: str, headers: Mapping[str, str]):
    # The legacy interface supports the project's websockets>=11 floor.
    from websockets.exceptions import SecurityError
    from websockets.legacy.client import Connect

    class DirectConnect(Connect):
        def handle_redirect(self, uri: str) -> None:
            raise SecurityError("Kalshi credential redirects are forbidden")

    return DirectConnect(
        url,
        extra_headers=headers,
        open_timeout=10,
        close_timeout=2,
        ping_interval=20,
        ping_timeout=10,
        max_size=1024 * 1024,
        max_queue=16,
    )


class KalshiBookStream:
    """Run a bounded, read-only, single-market stream with explicit auth.

    on_event is synchronous. Cancel the run task to stop. Caller owns recording.
    """

    def __init__(
        self,
        ticker: str,
        *,
        signer_headers: Callable[[str, str], Mapping[str, str]],
        demo: bool = False,
        max_reconnects: int = 3,
        idle_timeout: float = 30,
    ) -> None:
        if not callable(signer_headers):
            raise ValueError("explicit signer_headers callable required")
        if type(max_reconnects) is not int or not 0 <= max_reconnects <= 10:
            raise ValueError("max_reconnects must be in [0, 10]")
        if (
            isinstance(idle_timeout, bool)
            or not math.isfinite(idle_timeout)
            or not 0 < idle_timeout <= 300
        ):
            raise ValueError("idle_timeout must be finite and in (0, 300]")
        if type(demo) is not bool:
            raise ValueError("demo must be boolean")
        self.book = KalshiBook(ticker)
        self._signer = signer_headers
        self._url = DEMO_WS_URL if demo else WS_URL
        self._max_reconnects = max_reconnects
        self._idle_timeout = idle_timeout
        self._running = False

    async def run(
        self,
        on_event: Callable[[StreamEvent], None],
        *,
        recording: BookRecording | None = None,
    ) -> None:
        from websockets.exceptions import ConnectionClosed

        if self._running:
            raise RuntimeError("stream already running")
        self._running = True

        def emit(kind: str, *, frame=None, reason=None) -> None:
            at = datetime.now(timezone.utc)
            if kind == "reset":
                self.book.reset()
                event = StreamEvent(kind, at, reason=reason)
            else:
                event = StreamEvent("book", at, self.book.apply(frame, received_at=at))
            record = {
                "version": RECORD_VERSION,
                "ticker": self.book.ticker,
                "kind": kind,
                "received_at": at.isoformat(),
                **({"reason": reason} if kind == "reset" else {"frame": frame}),
            }
            if recording is not None:
                recording.write(record)
            try:
                on_event(event)
            except Exception as exc:
                raise RuntimeError("stream consumer failed") from exc

        try:
            for attempt in range(self._max_reconnects + 1):
                emit("reset", reason="connecting")
                # Fresh authentication for every connection; never persisted.
                headers = dict(self._signer("GET", WS_PATH))
                try:
                    async with _connect(self._url, headers) as socket:
                        acknowledged_sid = None
                        await socket.send(
                            json.dumps(
                                {
                                    "id": 1,
                                    "cmd": "subscribe",
                                    "params": {
                                        "channels": ["orderbook_delta"],
                                        "market_tickers": [self.book.ticker],
                                    },
                                }
                            )
                        )
                        while True:
                            raw = await asyncio.wait_for(socket.recv(), self._idle_timeout)
                            try:
                                frame = _json(raw)
                                if frame.get("type") == "subscribed":
                                    msg = frame.get("msg")
                                    if (
                                        acknowledged_sid is not None
                                        or self.book.latest
                                        or frame.get("id") != 1
                                        or not isinstance(msg, dict)
                                        or msg.get("channel") != "orderbook_delta"
                                    ):
                                        raise KalshiDataError(
                                            "unexpected subscription acknowledgement"
                                        )
                                    acknowledged_sid = _integer(msg.get("sid"), "sid")
                                    continue
                                if frame.get("type") == "error":
                                    raise RuntimeError("Kalshi rejected subscription")
                                if (
                                    acknowledged_sid is not None
                                    and frame.get("sid") != acknowledged_sid
                                ):
                                    raise KalshiDataError(
                                        "book subscription differs from acknowledgement"
                                    )
                                emit("frame", frame=frame)
                            except KalshiDataError as exc:
                                self.book.reset()
                                raise RecoveryRequired(str(exc)) from exc
                except (ConnectionClosed, OSError, asyncio.TimeoutError, RecoveryRequired) as exc:
                    if attempt == self._max_reconnects:
                        raise RecoveryRequired("stream recovery budget exhausted") from exc
                finally:
                    emit("reset", reason="disconnected")
                await asyncio.sleep(min(0.25 * 2**attempt, 4))
        finally:
            self.book.reset()
            self._running = False
