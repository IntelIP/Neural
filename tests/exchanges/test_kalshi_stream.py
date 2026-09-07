"""NRCL-88 synthetic wire fixtures; no exchange access or real credentials."""

import asyncio
import json
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from http import HTTPStatus

import pytest

import neural.kalshi_stream as streaming
from neural.kalshi import KalshiDataError
from neural.kalshi_stream import (
    BookRecording,
    KalshiBook,
    KalshiBookStream,
    RecoveryRequired,
    replay_book_recording,
)

NOW = datetime(2026, 9, 6, tzinfo=timezone.utc)
TICKER = "KX-EXAMPLE"


def snapshot(seq=2, sid=7):
    return {
        "type": "orderbook_snapshot",
        "sid": sid,
        "seq": seq,
        "msg": {
            "market_ticker": TICKER,
            "yes_dollars_fp": [["0.3001", "2.50"]],
            "no_dollars_fp": [["0.5999", "4.75"]],
        },
    }


def delta(seq=3, value="-1.25", sid=7):
    return {
        "type": "orderbook_delta",
        "sid": sid,
        "seq": seq,
        "msg": {
            "market_ticker": TICKER,
            "side": "yes",
            "price_dollars": "0.3001",
            "delta_fp": value,
            "ts": "2026-09-06T00:00:00Z",
            "ts_ms": 1788652800123,
        },
    }


def test_exact_snapshot_delta_removal_and_recovery():
    book = KalshiBook(TICKER)
    first = book.apply(snapshot(), received_at=NOW)
    assert first.source_at is None
    assert first.book.asks("yes")[0].price == Decimal("0.4001")
    with localcontext() as context:
        context.prec = 2
        changed = book.apply(delta(), received_at=NOW)
    assert changed.book.yes_bids[0].quantity == Decimal("1.25")
    assert first.book.yes_bids[0].quantity == Decimal("2.50")
    assert changed.source_at.microsecond == 123000
    assert not book.apply(delta(4), received_at=NOW).book.yes_bids
    with pytest.raises(RecoveryRequired):
        book.apply(delta(6), received_at=NOW)
    assert book.latest is None
    with pytest.raises(RecoveryRequired):
        book.apply(delta(7), received_at=NOW)
    assert book.apply(snapshot(30, sid=9), received_at=NOW).sid == 9


@pytest.mark.parametrize(
    "field,value",
    [
        ("delta_fp", "-2.51"),
        ("delta_fp", 1),
        ("delta_fp", "NaN"),
        ("delta_fp", "+1.00"),
        ("delta_fp", "0.001"),
        ("side", "maybe"),
        ("price_dollars", "1.0001"),
        ("market_ticker", "OTHER"),
        ("ts", "bad"),
        ("ts_ms", True),
        ("ts_ms", 1),
    ],
)
def test_bad_delta_invalidates(field, value):
    book = KalshiBook(TICKER)
    book.apply(snapshot(), received_at=NOW)
    frame = delta()
    frame["msg"][field] = value
    with pytest.raises(RecoveryRequired):
        book.apply(frame, received_at=NOW)
    assert book.latest is None


@pytest.mark.parametrize(
    "frame", [delta(2), delta(4), delta(sid=8), snapshot(2), {"type": "orderbook_delta", "msg": []}]
)
def test_sequence_and_shape_fail_closed(frame):
    book = KalshiBook(TICKER)
    book.apply(snapshot(), received_at=NOW)
    with pytest.raises(RecoveryRequired):
        book.apply(frame, received_at=NOW)
    assert book.latest is None


class Socket:
    def __init__(self, frames):
        self.frames = iter(frames)
        self.sent = []
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True

    async def send(self, value):
        self.sent.append(json.loads(value))

    async def recv(self):
        value = next(self.frames, asyncio.CancelledError())
        if isinstance(value, BaseException):
            raise value
        return json.dumps(value)


def install_sockets(monkeypatch, sessions):
    sockets = [Socket(frames) for frames in sessions]
    pending = iter(sockets)
    calls = []

    def connect(url, headers):
        calls.append((url, headers))
        return next(pending)

    monkeypatch.setattr(streaming, "_connect", connect)
    return sockets, calls


@pytest.mark.asyncio
async def test_gap_reconnect_recording_and_replay(monkeypatch, tmp_path):
    sockets, calls = install_sockets(
        monkeypatch,
        [
            [snapshot(), delta(), delta(5)],
            [snapshot(20, sid=8), delta(21, sid=8)],
        ],
    )
    signatures = []

    def sign(method, path):
        signatures.append((method, path))
        return {"KALSHI-ACCESS-KEY": "fake-secret"}

    stream = KalshiBookStream(TICKER, signer_headers=sign, max_reconnects=1)
    events = []
    path = tmp_path / "book.jsonl"
    with BookRecording(path, durable=True) as recording:
        with pytest.raises(asyncio.CancelledError):
            await stream.run(events.append, recording=recording)
    assert stream.book.latest is None
    assert all(socket.closed for socket in sockets)
    assert len(signatures) == len(calls) == 2
    assert signatures == [("GET", streaming.WS_PATH)] * 2
    assert sockets[0].sent == [
        {
            "id": 1,
            "cmd": "subscribe",
            "params": {
                "channels": ["orderbook_delta"],
                "market_tickers": [TICKER],
            },
        }
    ]
    assert [e.update.seq for e in events if e.update] == [2, 3, 20, 21]
    assert [e.kind for e in events] == ["reset", "book", "book", "reset"] * 2
    assert "fake-secret" not in path.read_text()
    assert list(replay_book_recording(path)) == events
    assert list(replay_book_recording(path)) == list(replay_book_recording(path))
    with pytest.raises(FileExistsError):
        BookRecording(path)


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [OSError("offline"), asyncio.TimeoutError(), delta(50)])
async def test_retry_budget_and_invalidation(monkeypatch, bad):
    sockets, calls = install_sockets(monkeypatch, [[snapshot(), bad]] * 2)
    stream = KalshiBookStream(TICKER, signer_headers=lambda *args: {}, max_reconnects=1)
    events = []
    with pytest.raises(RecoveryRequired, match="budget"):
        await stream.run(events.append)
    assert len(calls) == 2 and all(s.closed for s in sockets)
    assert stream.book.latest is None and events[-1].kind == "reset"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["consumer", "recording", "server"])
async def test_fatal_errors_not_retried(monkeypatch, tmp_path, failure):
    frames = [{"type": "error"}] if failure == "server" else [snapshot()]
    sockets, calls = install_sockets(monkeypatch, [frames])
    stream = KalshiBookStream(TICKER, signer_headers=lambda *args: {})

    def consume(event):
        if failure == "consumer" and event.update:
            raise OSError("consumer failed")

    with BookRecording(tmp_path / "closed.jsonl") as recording:
        if failure == "recording":
            recording.close()
        with pytest.raises(RuntimeError):
            await stream.run(consume, recording=recording)
    assert len(calls) <= 1 and stream.book.latest is None
    if calls:
        assert sockets[0].closed


@pytest.mark.asyncio
async def test_local_wire_transport_and_auth(monkeypatch):
    from websockets.legacy.server import serve

    requests = []

    async def handler(socket, path):
        requests.append(
            (path, socket.request_headers["KALSHI-ACCESS-KEY"], json.loads(await socket.recv()))
        )
        await socket.send(
            json.dumps(
                {
                    "id": 1,
                    "type": "subscribed",
                    "msg": {
                        "channel": "orderbook_delta",
                        "sid": 7,
                    },
                }
            )
        )
        await socket.send(json.dumps(snapshot()))
        await socket.send(json.dumps(delta()))
        await socket.close()

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        stream = KalshiBookStream(
            TICKER,
            signer_headers=lambda *args: {"KALSHI-ACCESS-KEY": "test-only"},
            max_reconnects=0,
        )
        stream._url = f"ws://127.0.0.1:{port}" + streaming.WS_PATH
        events = []
        with pytest.raises(RecoveryRequired):
            await stream.run(events.append)
    assert requests[0][:2] == (streaming.WS_PATH, "test-only")
    assert [e.update.seq for e in events if e.update] == [2, 3]
    assert stream.book.latest is None


@pytest.mark.asyncio
async def test_redirect_rejected_before_credentials_forwarded():
    from websockets.exceptions import SecurityError
    from websockets.legacy.server import serve

    async def redirect(path, headers):
        return HTTPStatus.FOUND, [("Location", "ws://127.0.0.1:1/stolen")], b""

    async def handler(socket, path):
        pytest.fail("redirect must not establish websocket")

    async with serve(handler, "127.0.0.1", 0, process_request=redirect) as server:
        port = server.sockets[0].getsockname()[1]
        with pytest.raises(SecurityError, match="redirects"):
            async with streaming._connect(f"ws://127.0.0.1:{port}", {"KALSHI-ACCESS-KEY": "fake"}):
                pytest.fail("redirect accepted")


@pytest.mark.parametrize(
    "text",
    [
        "",
        "{",
        '{"kind":"reset","kind":"frame"}\n',
        '{"value":NaN}\n',
        json.dumps({"version": "old"}) + "\n",
    ],
)
def test_corrupt_recordings_rejected(tmp_path, text):
    path = tmp_path / "bad.jsonl"
    path.write_text(text)
    with pytest.raises(KalshiDataError):
        list(replay_book_recording(path))


def test_incomplete_and_cross_market_recordings(tmp_path):
    base = {"version": streaming.RECORD_VERSION, "ticker": TICKER, "received_at": NOW.isoformat()}
    start = {**base, "kind": "reset", "reason": "connecting"}
    frame = {**base, "kind": "frame", "frame": snapshot()}
    end = {**base, "kind": "reset", "reason": "disconnected"}
    cases = [
        [start],
        [start, frame],
        [frame, end],
        [start, end, frame, end],
        [start, {**frame, "ticker": "OTHER"}, end],
    ]
    for index, records in enumerate(cases):
        path = tmp_path / f"bad-{index}.jsonl"
        with BookRecording(path) as recording:
            for record in records:
                recording.write(record)
        with pytest.raises(KalshiDataError):
            list(replay_book_recording(path))
